"""Tests for schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from radarbase_io.index import build_index
from radarbase_io.schema import build_schema


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


INDEX_ROOT = _repo_root() / "tests" / "data" / "mockdata" / "mockdata"
SCHEMAS_DIR = _repo_root() / "tests" / "data" / "schemas"


def _schema_paths_from_index() -> list[Path]:
    if not INDEX_ROOT.exists():
        return []
    try:
        index = build_index(INDEX_ROOT)
    except ValueError:
        return []
    if "schema_path" not in index.columns:
        return []
    paths = [Path(path) for path in index["schema_path"].dropna().unique()]
    return sorted(path for path in paths if path.exists())


def _schema_paths_from_raw() -> list[Path]:
    if not SCHEMAS_DIR.exists():
        return []
    return sorted(
        path for path in SCHEMAS_DIR.iterdir() if path.suffix in {".json", ".avsc"}
    )


INDEX_SCHEMA_PATHS = _schema_paths_from_index()
RAW_SCHEMA_PATHS = _schema_paths_from_raw()

if not INDEX_SCHEMA_PATHS and not RAW_SCHEMA_PATHS:
    pytest.skip(
        "No schema files found via build_index or tests/data/schemas",
        allow_module_level=True,
    )


def _load_schema_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError(f"Schema {path} is not a JSON object")
    return data


def _extract_record_type(type_obj: Any) -> dict | None:
    if isinstance(type_obj, dict) and type_obj.get("type") == "record":
        return type_obj
    if isinstance(type_obj, list):
        for entry in type_obj:
            if isinstance(entry, dict) and entry.get("type") == "record":
                return entry
    return None


def _select_schema_path() -> Path:
    if INDEX_SCHEMA_PATHS:
        return INDEX_SCHEMA_PATHS[0]
    return RAW_SCHEMA_PATHS[0]


def _iter_flattened_fields(schema: dict) -> list[tuple[str, dict]]:
    flattened: list[tuple[str, dict]] = []
    for field in schema.get("fields", []):
        if not isinstance(field, dict) or "name" not in field:
            continue
        record = _extract_record_type(field.get("type"))
        if record is None:
            continue
        for subfield in record.get("fields", []):
            if isinstance(subfield, dict) and "name" in subfield:
                flattened.append((f"{field['name']}.{subfield['name']}", subfield))
    return flattened


def _is_union_with_null(type_obj: Any, primitive: str) -> bool:
    if not isinstance(type_obj, list):
        return False
    types: list[str | None] = []
    for entry in type_obj:
        if isinstance(entry, dict):
            types.append(entry.get("type"))
        else:
            types.append(entry)
    return "null" in types and primitive in types and types.count("null") == 1


def _is_enum_type(type_obj: Any) -> bool:
    if isinstance(type_obj, dict) and type_obj.get("type") == "enum":
        return True
    if isinstance(type_obj, list):
        return any(_is_enum_type(entry) for entry in type_obj)
    return False


def _is_double_type(type_obj: Any) -> bool:
    if isinstance(type_obj, str):
        return type_obj == "double"
    if isinstance(type_obj, dict):
        return type_obj.get("type") == "double"
    if isinstance(type_obj, list):
        return _is_union_with_null(type_obj, "double")
    return False


def _select_schema_for_dtype() -> Path | None:
    seen = set()
    all_paths = []
    for path in INDEX_SCHEMA_PATHS + RAW_SCHEMA_PATHS:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        all_paths.append(path)
    for path in all_paths:
        try:
            schema = _load_schema_file(path)
        except Exception:
            continue
        flattened = _iter_flattened_fields(schema)
        has_union_int = any(
            _is_union_with_null(field.get("type"), "int") for _, field in flattened
        )
        has_enum = any(_is_enum_type(field.get("type")) for _, field in flattened)
        has_double = any(_is_double_type(field.get("type")) for _, field in flattened)
        if has_union_int and has_enum and has_double:
            return path
    return None


def _smoke_cases() -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    cases.extend(("index", path) for path in INDEX_SCHEMA_PATHS)
    cases.extend(("raw", path) for path in RAW_SCHEMA_PATHS)
    return cases


@pytest.mark.parametrize(
    "case",
    _smoke_cases(),
    ids=lambda case: f"{case[0]}:{case[1].name}",
)
def test_build_schema_smoke(case: tuple[str, Path]):
    _, schema_path = case
    schema = _load_schema_file(schema_path)

    result = build_schema(schema_path)

    required_keys = {
        "measurement",
        "fqn",
        "columns",
        "pandas_dtypes",
        "meta",
        "pyarrow_schema",
    }
    assert required_keys.issubset(result.keys())

    columns = result["columns"]
    assert list(result["pandas_dtypes"].keys()) == columns

    top_names = [
        field.get("name")
        for field in schema.get("fields", [])
        if isinstance(field, dict)
    ]
    if "key" in top_names and "value" in top_names:
        assert any(col.startswith("key.") for col in columns)
        assert any(col.startswith("value.") for col in columns)
        key_cols = [col for col in columns if col.startswith("key.")]
        value_cols = [col for col in columns if col.startswith("value.")]
        if key_cols and value_cols:
            assert columns.index(key_cols[-1]) < columns.index(value_cols[0])

    value_record = None
    for field in schema.get("fields", []):
        if isinstance(field, dict) and field.get("name") == "value":
            value_record = _extract_record_type(field.get("type"))
            break

    expected_measurement = (
        value_record.get("name") if value_record is not None else schema.get("name")
    )
    assert result["measurement"] == expected_measurement

    namespace = None
    if value_record is not None:
        namespace = value_record.get("namespace") or schema.get("namespace")
    else:
        namespace = schema.get("namespace")
    expected_fqn = (
        f"{namespace}.{expected_measurement}" if namespace else expected_measurement
    )
    assert result["fqn"] == expected_fqn


def test_build_schema_dtype_mapping_and_meta():
    schema_path = _select_schema_for_dtype()
    if schema_path is None:
        pytest.skip("No schema with union int, enum, and double fields found")
    schema = _load_schema_file(schema_path)
    result = build_schema(schema)

    flattened = _iter_flattened_fields(schema)

    union_int_col = next(
        (
            col
            for col, field in flattened
            if _is_union_with_null(field.get("type"), "int")
        ),
        None,
    )
    if union_int_col is None:
        pytest.skip("No nullable int field found in selected schema")
    assert result["pandas_dtypes"][union_int_col] == "Int64"

    enum_col = next(
        (col for col, field in flattened if _is_enum_type(field.get("type"))), None
    )
    if enum_col is None:
        pytest.skip("No enum field found in selected schema")
    assert result["pandas_dtypes"][enum_col] == "object"

    double_col = next(
        (col for col, field in flattened if _is_double_type(field.get("type"))), None
    )
    if double_col is None:
        pytest.skip("No double field found in selected schema")
    assert result["pandas_dtypes"][double_col] == "float64"

    meta = result["meta"]
    assert isinstance(meta, pd.DataFrame)
    assert len(meta) == 0
    assert list(meta.columns) == result["columns"]

    meta_dtypes = {col: str(dtype) for col, dtype in meta.dtypes.items()}
    assert meta_dtypes == result["pandas_dtypes"]


def test_build_schema_pyarrow_optional():
    schema_path = _select_schema_path()
    result = build_schema(schema_path)

    import pyarrow as pa

    assert isinstance(result["pyarrow_schema"], pa.Schema)
    assert result["pyarrow_schema"].names == result["columns"]


def test_build_schema_invalid_schema_type_raises():
    with pytest.raises(ValueError, match="schema must be a dict or path-like"):
        build_schema(123)


def test_build_schema_missing_file_raises(tmp_path):
    missing = tmp_path / "missing.avsc"
    with pytest.raises(FileNotFoundError):
        build_schema(missing)


def test_build_schema_invalid_json_raises(tmp_path):
    path = tmp_path / "invalid.avsc"
    path.write_text("{")

    with pytest.raises(ValueError, match="Invalid JSON schema"):
        build_schema(path)


def test_build_schema_invalid_embedded_json_raises(tmp_path):
    path = tmp_path / "embedded-invalid.avsc"
    path.write_text(json.dumps("{bad"))

    with pytest.raises(ValueError, match="Invalid embedded JSON schema"):
        build_schema(path)


def test_build_schema_non_object_json_raises(tmp_path):
    path = tmp_path / "list-json.avsc"
    path.write_text("[]")

    with pytest.raises(ValueError, match="not a JSON object"):
        build_schema(path)


def test_build_schema_missing_fields_raises():
    with pytest.raises(ValueError, match="missing 'fields' list"):
        build_schema({"type": "record", "name": "X"})


def test_build_schema_field_missing_name_raises():
    schema = {"type": "record", "name": "X", "fields": [{"type": "string"}]}

    with pytest.raises(ValueError, match="Schema field missing name"):
        build_schema(schema)


def test_build_schema_missing_record_name_raises():
    schema = {"type": "record", "fields": [{"name": "value", "type": "string"}]}

    with pytest.raises(ValueError, match="Schema missing record name"):
        build_schema(schema)


def test_build_schema_record_missing_fields_raises():
    schema = {
        "type": "record",
        "name": "Top",
        "fields": [
            {
                "name": "value",
                "type": {"type": "record", "name": "Inner"},
            }
        ],
    }

    with pytest.raises(ValueError, match="missing 'fields'"):
        build_schema(schema)


def test_build_schema_record_invalid_subfield_raises():
    schema = {
        "type": "record",
        "name": "Top",
        "fields": [
            {
                "name": "value",
                "type": {"type": "record", "name": "Inner", "fields": [{}]},
            }
        ],
    }

    with pytest.raises(ValueError, match="invalid subfield"):
        build_schema(schema)


def test_build_schema_handles_union_record_and_non_record_fields():
    schema = {
        "type": "record",
        "name": "Top",
        "namespace": "org.example",
        "fields": [
            {
                "name": "key",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "KeyRecord",
                        "fields": [{"name": "id", "type": "string"}],
                    },
                ],
            },
            {
                "name": "value",
                "type": [
                    "null",
                    {
                        "type": "record",
                        "name": "ValueRecord",
                        "fields": [
                            {"name": "mixed_union", "type": ["int", "string"]},
                            {"name": "unknown_token", "type": 123},
                        ],
                    },
                ],
            },
            {"name": "meta", "type": "double"},
            {"name": "labels", "type": ["null", "string"]},
        ],
    }

    result = build_schema(schema)

    assert result["columns"] == [
        "key.id",
        "value.mixed_union",
        "value.unknown_token",
        "meta",
        "labels",
    ]
    assert result["pandas_dtypes"]["value.mixed_union"] == "object"
    assert result["pandas_dtypes"]["value.unknown_token"] == "object"
    assert result["pandas_dtypes"]["meta"] == "float64"
    assert result["pandas_dtypes"]["labels"] == "object"


def test_build_schema_without_key_value_uses_top_level_order():
    schema = {
        "type": "record",
        "name": "Top",
        "fields": [
            {"name": "alpha", "type": "double"},
            {
                "name": "nested",
                "type": {
                    "type": "record",
                    "name": "Nested",
                    "fields": [{"name": "beta", "type": "int"}],
                },
            },
        ],
    }

    result = build_schema(schema)
    assert result["columns"] == ["alpha", "nested.beta"]
    assert result["measurement"] == "Top"
