"""Schema helpers for RADAR Avro definitions."""

import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pyarrow as pa

from .fs import resolve_paths


def _load_schema(schema, *, fs=None, storage_options=None):
    """Load a schema from a dict or JSON/AVSC file."""
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, (str, Path)):
        fs, fs_paths = resolve_paths(schema, fs=fs, storage_options=storage_options)
        if not fs_paths:
            raise ValueError("Schema path is empty")
        path = fs_paths[0]
        if not fs.exists(path):
            raise FileNotFoundError(path)
        with fs.open(path, "r") as handle:
            text = handle.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON schema in {path}") from exc
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid embedded JSON schema in {path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Schema in {path} is not a JSON object")
        return data
    raise ValueError("schema must be a dict or path-like to a JSON schema file")


def _is_null_type(type_obj):
    return type_obj == "null" or (
        isinstance(type_obj, dict) and type_obj.get("type") == "null"
    )


def _extract_record_type(type_obj):
    if isinstance(type_obj, dict) and type_obj.get("type") == "record":
        return type_obj
    if isinstance(type_obj, list):
        for entry in type_obj:
            if isinstance(entry, dict) and entry.get("type") == "record":
                return entry
    return None


def _extract_array_record_type(type_obj):
    if isinstance(type_obj, list):
        for entry in type_obj:
            record = _extract_array_record_type(entry)
            if record is not None:
                return record
        return None
    if isinstance(type_obj, dict):
        if type_obj.get("type") == "array":
            return _extract_record_type(type_obj.get("items"))
        inner = type_obj.get("type")
        if isinstance(inner, (dict, list)):
            return _extract_array_record_type(inner)
    return None


def _iter_observed_headers(observed_columns):
    if not observed_columns:
        return []
    if isinstance(observed_columns, (list, tuple)) and all(
        isinstance(item, str) for item in observed_columns
    ):
        return [observed_columns]
    return observed_columns


def _collect_array_record_columns(prefix, array_record, observed_columns):
    if not observed_columns:
        return []
    item_fields = array_record.get("fields")
    if not isinstance(item_fields, list):
        raise ValueError(f"Array record field '{prefix}' missing 'fields'")
    subfield_dtypes = {}
    for item_field in item_fields:
        if not isinstance(item_field, dict) or "name" not in item_field:
            raise ValueError(f"Array record field '{prefix}' has invalid subfield")
        subfield_dtypes[item_field["name"]] = _avro_type_to_pandas(
            item_field.get("type")
        )
    expanded = []
    seen = set()
    match_prefix = f"{prefix}."
    for header in _iter_observed_headers(observed_columns):
        if not isinstance(header, (list, tuple)):
            continue
        for column in header:
            if not isinstance(column, str) or not column.startswith(match_prefix):
                continue
            suffix = column[len(match_prefix) :]
            parts = suffix.split(".")
            if len(parts) != 2:
                continue
            index, item_name = parts
            if not index.isdigit() or item_name not in subfield_dtypes:
                continue
            if column in seen:
                continue
            seen.add(column)
            expanded.append((column, subfield_dtypes[item_name]))
    return expanded


def _avro_type_to_pandas(type_obj):
    if isinstance(type_obj, list):
        non_null = [entry for entry in type_obj if not _is_null_type(entry)]
        if len(non_null) == 1:
            return _avro_type_to_pandas(non_null[0])
        return "object"
    if isinstance(type_obj, dict):
        inner = type_obj.get("type")
        if inner == "enum":
            return "object"
        if inner in {"record", "array", "map"}:
            return "object"
        return _avro_type_to_pandas(inner)
    if isinstance(type_obj, str):
        if type_obj == "string":
            return "object"
        if type_obj == "double":
            return "float64"
        if type_obj == "float":
            return "float32"
        if type_obj in {"int", "long"}:
            return "Int64"
        if type_obj == "boolean":
            return "boolean"
        return "object"
    return "object"


def _pandas_dtype_to_arrow(dtype):
    mapping = {
        "string": pa.string(),
        "float64": pa.float64(),
        "float32": pa.float32(),
        "Int64": pa.int64(),
        "boolean": pa.bool_(),
        "object": pa.string(),
    }
    return mapping.get(dtype, pa.string())


def build_schema(
    schema,
    *,
    flatten_key="key",
    flatten_value="value",
    prefer_value_record=True,
    observed_columns=None,
    fs=None,
    storage_options=None,
):
    """
    Build schema artifacts for reading flattened RADAR CSV exports.

    Parameters
    ----------
    schema : dict or str or pathlib.Path
        Schema dictionary or path to a JSON/AVSC file.
    flatten_key : str, default "key"
        Field name used for key record flattening.
    flatten_value : str, default "value"
        Field name used for value record flattening.
    prefer_value_record : bool, default True
        If True, choose the nested value record for measurement metadata.
    observed_columns : list[list[str]] or list[str], optional
        Flattened CSV headers used to expand array-of-record fields into
        concrete indexed columns (for example ``value.answers.0.questionId``).
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance used to load a schema path.
    storage_options : dict, optional
        Storage options passed to fsspec when resolving a schema path.

    Returns
    -------
    dict
        Dictionary containing measurement metadata, flattened columns,
        pandas dtype mapping, meta DataFrame, and pyarrow schema.
    """

    schema_obj = _load_schema(schema, fs=fs, storage_options=storage_options)
    fields = schema_obj.get("fields")
    if not isinstance(fields, list):
        raise ValueError("Schema missing 'fields' list")

    fields_by_name = {}
    for field in fields:
        if not isinstance(field, dict) or "name" not in field:
            raise ValueError("Schema field missing name")
        fields_by_name[field["name"]] = field

    preferred_record = None
    if prefer_value_record:
        value_field = fields_by_name.get(flatten_value)
        if value_field is not None:
            preferred_record = _extract_record_type(value_field.get("type"))
    if preferred_record is None:
        preferred_record = schema_obj

    measurement = (
        preferred_record.get("name") if isinstance(preferred_record, dict) else None
    )
    if not measurement:
        raise ValueError("Schema missing record name")

    namespace = None
    if isinstance(preferred_record, dict):
        namespace = preferred_record.get("namespace") or schema_obj.get("namespace")
    fqn = f"{namespace}.{measurement}" if namespace else measurement

    if flatten_key in fields_by_name and flatten_value in fields_by_name:
        ordered_names = [
            flatten_key,
            flatten_value,
            *[
                field["name"]
                for field in fields
                if field["name"] not in {flatten_key, flatten_value}
            ],
        ]
    else:
        ordered_names = [field["name"] for field in fields]

    columns = []
    pandas_dtypes = OrderedDict()

    for name in ordered_names:
        field = fields_by_name.get(name)
        if field is None:
            continue
        field_type = field.get("type")
        record = _extract_record_type(field_type)
        if record is not None:
            subfields = record.get("fields")
            if not isinstance(subfields, list):
                raise ValueError(f"Record field '{name}' missing 'fields'")
            for subfield in subfields:
                if not isinstance(subfield, dict) or "name" not in subfield:
                    raise ValueError(f"Record field '{name}' has invalid subfield")
                column = f"{name}.{subfield['name']}"
                array_record = _extract_array_record_type(subfield.get("type"))
                if array_record is not None:
                    expanded = _collect_array_record_columns(
                        column,
                        array_record,
                        observed_columns,
                    )
                    if expanded:
                        for expanded_column, expanded_dtype in expanded:
                            columns.append(expanded_column)
                            pandas_dtypes[expanded_column] = expanded_dtype
                        continue
                columns.append(column)
                pandas_dtypes[column] = _avro_type_to_pandas(subfield.get("type"))
        else:
            array_record = _extract_array_record_type(field_type)
            if array_record is not None:
                expanded = _collect_array_record_columns(
                    name,
                    array_record,
                    observed_columns,
                )
                if expanded:
                    for expanded_column, expanded_dtype in expanded:
                        columns.append(expanded_column)
                        pandas_dtypes[expanded_column] = expanded_dtype
                    continue
            columns.append(name)
            pandas_dtypes[name] = _avro_type_to_pandas(field_type)

    meta = pd.DataFrame(
        {col: pd.Series(dtype=dtype) for col, dtype in pandas_dtypes.items()}
    ).reindex(columns=columns)

    pyarrow_schema = pa.schema(
        [
            pa.field(col, _pandas_dtype_to_arrow(dtype))
            for col, dtype in pandas_dtypes.items()
        ]
    )

    return {
        "measurement": measurement,
        "fqn": fqn,
        "columns": columns,
        "pandas_dtypes": dict(pandas_dtypes),
        "meta": meta,
        "pyarrow_schema": pyarrow_schema,
    }
