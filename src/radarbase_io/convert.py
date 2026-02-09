"""CSV to parquet conversion helpers."""

import math
import os
import re
import uuid
from collections import OrderedDict

import dask.dataframe as dd
import fsspec
import pandas as pd
import pyarrow as pa
from dask import delayed
from dask.utils import parse_bytes
from tlz import partition_all

from .fs import call_with_fs_json, fs_to_json, resolve_paths
from .performance import compute_dask
from .utils import align_and_coerce_dataframe

_INDEXED_ARRAY_COLUMN_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<index>\d+)\.(?P<field>[^.]+)$"
)


def _split_parent_name(path):
    normalized = path.rstrip("/")
    if "/" not in normalized:
        return "", normalized
    parent, name = normalized.rsplit("/", 1)
    return parent, name


def _join_fs_path(parent, name):
    if not parent:
        return name
    return f"{parent.rstrip('/')}/{name}"


def _infer_csv_compression(path):
    if path.endswith(".gz"):
        return "gzip"
    return None


def _protocol_set(protocol):
    if protocol is None:
        return set()
    if isinstance(protocol, (tuple, list, set)):
        return {str(item) for item in protocol if item}
    return {str(protocol)}


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


def _validate_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("schema must be a dict produced by build_schema")

    required = {"columns", "pandas_dtypes", "meta", "pyarrow_schema"}
    missing = [key for key in required if key not in schema]
    if missing:
        raise ValueError(
            f"schema is missing required keys: {', '.join(sorted(missing))}"
        )

    columns = schema["columns"]
    pandas_dtypes = schema["pandas_dtypes"]
    meta = schema["meta"]
    pyarrow_schema = schema["pyarrow_schema"]
    array_record_dtypes = schema.get("array_record_dtypes", {})

    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise ValueError("schema['columns'] must be a list of column names")
    if not isinstance(pandas_dtypes, dict):
        raise ValueError("schema['pandas_dtypes'] must be a dict")
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("schema['meta'] must be a pandas DataFrame")
    if not isinstance(pyarrow_schema, pa.Schema):
        raise ValueError("schema['pyarrow_schema'] must be a pyarrow.Schema")
    if array_record_dtypes is None:
        array_record_dtypes = {}
    if not isinstance(array_record_dtypes, dict):
        raise ValueError("schema['array_record_dtypes'] must be a dict when provided")

    missing_dtype_columns = [col for col in columns if col not in pandas_dtypes]
    if missing_dtype_columns:
        raise ValueError(
            "schema['pandas_dtypes'] is missing columns: "
            + ", ".join(missing_dtype_columns)
        )

    if list(pyarrow_schema.names) != columns:
        raise ValueError("schema['pyarrow_schema'] column order must match columns")

    normalized_array_record_dtypes = {}
    for prefix, subfield_map in array_record_dtypes.items():
        if not isinstance(prefix, str):
            raise ValueError("schema['array_record_dtypes'] keys must be strings")
        if not isinstance(subfield_map, dict):
            raise ValueError(
                "schema['array_record_dtypes'] values must be dicts of field dtypes"
            )
        normalized_subfield_map = OrderedDict()
        for field_name, field_dtype in subfield_map.items():
            if not isinstance(field_name, str):
                raise ValueError(
                    "schema['array_record_dtypes'] field names must be strings"
                )
            if not isinstance(field_dtype, str):
                raise ValueError(
                    "schema['array_record_dtypes'] dtype values must be strings"
                )
            normalized_subfield_map[field_name] = field_dtype
        normalized_array_record_dtypes[prefix] = dict(normalized_subfield_map)

    normalized_meta = meta.reindex(columns=columns)

    return (
        columns,
        pandas_dtypes,
        normalized_meta,
        pyarrow_schema,
        normalized_array_record_dtypes,
    )


def _sample_paths_for_header_scan(paths, *, sample_size=200):
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if len(paths) <= sample_size:
        return paths

    step = max(1, len(paths) // sample_size)
    return paths[::step][:sample_size]


def _read_header_columns(fs, path):
    compression = _infer_csv_compression(path)
    with fs.open(path, "rb") as handle:
        header = pd.read_csv(handle, compression=compression, nrows=0)
    return header.columns.tolist()


def _expand_array_record_columns_from_headers(prefix, subfield_dtypes, header_columns):
    observed_pairs = set()
    for header_column in header_columns:
        match = _INDEXED_ARRAY_COLUMN_RE.match(header_column)
        if match is None:
            continue
        if match.group("prefix") != prefix:
            continue
        subfield_name = match.group("field")
        if subfield_name not in subfield_dtypes:
            continue
        observed_pairs.add((int(match.group("index")), subfield_name))

    if not observed_pairs:
        return []

    expanded_columns = []
    ordered_subfields = list(subfield_dtypes.keys())
    for index in sorted({index for index, _ in observed_pairs}):
        for subfield_name in ordered_subfields:
            if (index, subfield_name) not in observed_pairs:
                continue
            expanded_columns.append(
                (
                    f"{prefix}.{index}.{subfield_name}",
                    subfield_dtypes[subfield_name],
                )
            )
    return expanded_columns


def _expand_schema_with_observed_array_columns(
    fs,
    csv_paths,
    columns,
    pandas_dtypes,
    array_record_dtypes,
    *,
    header_sample_size=200,
):
    if not array_record_dtypes:
        return columns, pandas_dtypes

    if not columns:
        return columns, pandas_dtypes

    header_columns = []
    sampled_paths = _sample_paths_for_header_scan(
        csv_paths,
        sample_size=header_sample_size,
    )
    for path in sampled_paths:
        header_columns.extend(_read_header_columns(fs, path))

    if not header_columns:
        return columns, pandas_dtypes

    replacement_columns = {}
    for prefix, subfield_dtypes in array_record_dtypes.items():
        if prefix not in columns:
            continue
        expanded = _expand_array_record_columns_from_headers(
            prefix,
            subfield_dtypes,
            header_columns,
        )
        if expanded:
            replacement_columns[prefix] = expanded

    if not replacement_columns:
        return columns, pandas_dtypes

    expanded_columns = []
    expanded_dtypes = OrderedDict()
    for column in columns:
        replacements = replacement_columns.get(column)
        if replacements:
            for replacement_column, replacement_dtype in replacements:
                expanded_columns.append(replacement_column)
                expanded_dtypes[replacement_column] = replacement_dtype
            continue
        expanded_columns.append(column)
        expanded_dtypes[column] = pandas_dtypes[column]

    return expanded_columns, dict(expanded_dtypes)


def _build_meta_and_pyarrow_schema(columns, pandas_dtypes):
    meta = pd.DataFrame(
        {column: pd.Series(dtype=pandas_dtypes[column]) for column in columns}
    ).reindex(columns=columns)
    pyarrow_schema = pa.schema(
        [
            pa.field(column, _pandas_dtype_to_arrow(pandas_dtypes[column]))
            for column in columns
        ]
    )
    return meta, pyarrow_schema


def _estimate_npartitions_from_size(fs, paths, partition_size, *, sample_size=200):
    target_bytes = parse_bytes(partition_size)
    if target_bytes <= 0:
        raise ValueError("repartition target size must be greater than 0")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    if not paths:
        return 1

    if len(paths) <= sample_size:
        sample_paths = paths
    else:
        step = max(1, len(paths) // sample_size)
        sample_paths = paths[::step][:sample_size]

    sampled_total = 0
    for path in sample_paths:
        sampled_total += fs.size(path)

    if len(sample_paths) == len(paths):
        estimated_total = sampled_total
    else:
        avg_size = sampled_total / len(sample_paths)
        estimated_total = int(avg_size * len(paths))

    return max(1, math.ceil(estimated_total / target_bytes))


def _read_batch(paths, columns, pandas_dtypes, *, fs=None, fs_json=None):
    if fs is None:
        if fs_json is None:
            raise ValueError("Pass either fs= or fs_json=.")
    elif fs_json is not None:
        raise ValueError("Pass either fs= or fs_json=, not both.")

    frames = []
    for path in paths:
        compression = _infer_csv_compression(path)

        def _read_one(active_fs):
            with active_fs.open(path, "rb") as handle:
                frame = pd.read_csv(handle, compression=compression)
            return align_and_coerce_dataframe(frame, columns, pandas_dtypes)

        if fs is not None:
            frame = _read_one(fs)
        else:
            frame = call_with_fs_json(
                fs_json,
                _read_one,
                operation=f"Reading CSV batch file {path!r}",
            )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            {col: pd.Series(dtype=pandas_dtypes[col]) for col in columns}
        )

    return pd.concat(frames, ignore_index=True)


def _build_temp_target(output_fs_path):
    parent, name = _split_parent_name(output_fs_path)
    if not name:
        name = "parquet"
    tmp_name = f".{name}.tmp-{uuid.uuid4().hex}"
    return _join_fs_path(parent, tmp_name)


def _remove_if_exists(fs, path):
    if fs.exists(path):
        fs.rm(path, recursive=True)


def _prepare_target(fs, output_fs_path, overwrite):
    if fs.exists(output_fs_path):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_fs_path}. "
                "Pass overwrite=True to replace it."
            )
        fs.rm(output_fs_path, recursive=True)

    parent, _ = _split_parent_name(output_fs_path)
    if parent and not fs.exists(parent):
        fs.makedirs(parent, exist_ok=True)


def csv_to_parquet(
    csv_paths,
    output_path,
    schema,
    fs=None,
    storage_options=None,
    *,
    output_mode="file",
    files_per_task=150,
    repartition=None,
    overwrite=False,
    client=None,
):
    """
    Convert CSV/CSV.GZ files into parquet using a build_schema artifact.

    Parameters
    ----------
    csv_paths : str or sequence of str
        Input CSV paths.
    output_path : str
        Output parquet file path (file mode) or dataset directory (dataset mode).
    schema : dict
        Output from radarbase_io.schema.build_schema.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance to read `csv_paths`. When `output_path` is a local
        path (file protocol), output is written to local filesystem.
    storage_options : dict, optional
        Storage options passed to fsspec when resolving paths.
    output_mode : {"file", "dataset"}, default "file"
        Output parquet layout mode.
    files_per_task : int, default 150
        Number of CSV files to process in one delayed batch task.
    repartition : int or str, optional
        Dataset-mode repartition hint. If `int`, sets target partition count
        (`npartitions`). If `str`, interpreted as Dask partition size (for
        example `"256MB"`). Only supported when `output_mode="dataset"`. When
        using `client=...`, size strings are converted into an estimated
        `npartitions` from total input file size.
    overwrite : bool, default False
        Whether to overwrite existing output.
    client : dask.distributed.Client, optional
        Dask client for distributed execution.
    """

    if output_mode not in {"file", "dataset"}:
        raise ValueError("output_mode must be one of {'file', 'dataset'}")

    (
        columns,
        pandas_dtypes,
        meta,
        pyarrow_schema,
        array_record_dtypes,
    ) = _validate_schema(schema)

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    else:
        csv_paths = list(csv_paths)

    if not csv_paths:
        raise ValueError("csv_paths is empty")
    if files_per_task <= 0:
        raise ValueError("files_per_task must be greater than 0")
    if output_mode == "file" and repartition is not None:
        raise ValueError("repartition is only supported when output_mode='dataset'")
    if isinstance(repartition, int) and repartition <= 0:
        raise ValueError("repartition must be greater than 0 when passed as int")
    if repartition is not None and not isinstance(repartition, (int, str)):
        raise ValueError(
            "repartition must be None, a positive int, or a partition-size string"
        )

    input_fs, csv_fs_paths = resolve_paths(
        csv_paths, fs=fs, storage_options=storage_options
    )

    if fs is None:
        output_protocol = fsspec.utils.get_protocol(os.fspath(output_path))
        input_protocols = _protocol_set(getattr(input_fs, "protocol", None))
        if not input_protocols:
            input_protocols = {fsspec.utils.get_protocol(os.fspath(csv_paths[0]))}

        if output_protocol == "file":
            output_fs, output_paths = resolve_paths(output_path)
        elif output_protocol in input_protocols:
            output_fs, output_paths = resolve_paths(output_path, fs=input_fs)
        else:
            raise ValueError(
                "Protocol mismatch: when fs is None, input and output protocols "
                "must match unless output_path is local (file). "
                f"Input protocol(s): {sorted(input_protocols)}, "
                f"output protocol: {output_protocol}."
            )
        output_fs_path = output_paths[0]
    else:
        output_protocol = fsspec.utils.get_protocol(os.fspath(output_path))
        if output_protocol == "file":
            output_fs, output_paths = resolve_paths(output_path)
        else:
            output_fs, output_paths = resolve_paths(output_path, fs=fs)
        output_fs_path = output_paths[0]

    if not csv_fs_paths:
        raise ValueError("No CSV paths were resolved")

    columns, pandas_dtypes = _expand_schema_with_observed_array_columns(
        input_fs,
        csv_fs_paths,
        columns,
        pandas_dtypes,
        array_record_dtypes,
    )
    meta, pyarrow_schema = _build_meta_and_pyarrow_schema(columns, pandas_dtypes)

    _prepare_target(output_fs, output_fs_path, overwrite=overwrite)

    batch_kwargs = (
        {"fs": input_fs} if client is None else {"fs_json": fs_to_json(input_fs)}
    )
    tasks = [
        delayed(_read_batch)(list(chunk), columns, pandas_dtypes, **batch_kwargs)
        for chunk in partition_all(files_per_task, csv_fs_paths)
    ]
    ddf = dd.from_delayed(tasks, meta=meta)

    row_count_task = ddf.map_partitions(len).sum()

    temp_target = None
    output_target = output_fs_path
    ddf_to_write = ddf
    if output_mode == "file":
        temp_target = _build_temp_target(output_fs_path)
        _remove_if_exists(output_fs, temp_target)
        output_target = temp_target
        ddf_to_write = ddf.repartition(npartitions=1)
    elif repartition is not None:
        if isinstance(repartition, int):
            ddf_to_write = ddf.repartition(npartitions=repartition)
        else:
            if client is None:
                ddf_to_write = ddf.repartition(partition_size=repartition)
            else:
                npartitions = _estimate_npartitions_from_size(
                    input_fs,
                    csv_fs_paths,
                    repartition,
                )
                ddf_to_write = ddf.repartition(npartitions=npartitions)

    write_task = ddf_to_write.to_parquet(
        output_target,
        engine="pyarrow",
        schema=pyarrow_schema,
        write_index=False,
        filesystem=output_fs,
        compute=False,
    )

    try:
        results = compute_dask(
            [write_task, row_count_task],
            client=client,
            scheduler="threads",
        )
        rows_written = int(results[1])

        if output_mode == "file":
            part_files = [
                path
                for path in output_fs.find(output_target)
                if path.endswith(".parquet")
            ]
            if len(part_files) != 1:
                raise ValueError(
                    "Expected a single parquet part file in file mode, "
                    f"found {len(part_files)}"
                )
            output_fs.mv(part_files[0], output_fs_path)
    finally:
        if temp_target is not None:
            _remove_if_exists(output_fs, temp_target)

    return {
        "output_mode": output_mode,
        "output_path": output_fs_path,
        "files_processed": len(csv_fs_paths),
        "rows_written": rows_written,
        "columns": columns,
    }
