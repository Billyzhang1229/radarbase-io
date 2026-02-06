"""Tests for CSV-to-parquet conversion helpers."""

import csv
import gzip
import json
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow as pa
import pytest

import radarbase_io.convert as convert_module
from radarbase_io.convert import csv_to_parquet
from radarbase_io.schema import build_schema

TEST_SCHEMA = {
    "type": "record",
    "name": "ObservationEnvelope",
    "fields": [
        {
            "name": "key",
            "type": {
                "type": "record",
                "name": "ObservationKey",
                "fields": [
                    {"name": "projectId", "type": ["null", "string"], "default": None},
                    {"name": "userId", "type": "string"},
                ],
            },
        },
        {
            "name": "value",
            "type": {
                "type": "record",
                "name": "Measurement",
                "fields": [
                    {"name": "time", "type": "double"},
                    {"name": "count", "type": ["null", "long"], "default": None},
                    {"name": "flag", "type": ["null", "boolean"], "default": None},
                    {"name": "note", "type": ["null", "string"], "default": None},
                ],
            },
        },
    ],
}


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _find_mockdata_data_type(data_type):
    root = _repo_root() / "tests" / "data" / "mockdata" / "mockdata"
    if not root.exists():
        return None, []

    schema_paths = sorted(root.rglob(f"schema-{data_type}.json"))
    for schema_path in schema_paths:
        csv_paths = sorted(schema_path.parent.glob("*.csv.gz"))
        if csv_paths:
            return schema_path, csv_paths[:5]

    return None, []


def _read_gzip_header(path):
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle))


def _write_csv(path, rows):
    columns = list(rows[0].keys())
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row[column]
                values.append("" if value is None else str(value))
            handle.write(",".join(values) + "\n")


def _build_schema_output(tmp_path):
    schema_path = tmp_path / "schema-test.json"
    schema_path.write_text(json.dumps(TEST_SCHEMA))
    return build_schema(schema_path)


def test_csv_to_parquet_file_mode_local(tmp_path):
    schema = _build_schema_output(tmp_path)

    input_a = tmp_path / "a.csv.gz"
    input_b = tmp_path / "b.csv.gz"

    _write_csv(
        input_a,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 1.0,
                "value.count": 5,
                "value.flag": "true",
                "extra.col": "drop-me",
            },
            {
                "key.projectId": "projA",
                "key.userId": "u2",
                "value.time": 2.0,
                "value.count": "bad-int",
                "value.flag": "0",
                "extra.col": "drop-me-too",
            },
        ],
    )
    _write_csv(
        input_b,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u3",
                "value.time": 3.5,
                "value.note": "hello",
            },
        ],
    )

    output_file = tmp_path / "out.parquet"
    result = csv_to_parquet(
        [input_a, input_b],
        output_path=output_file,
        schema=schema,
        output_mode="file",
    )

    assert output_file.exists()
    assert result["output_mode"] == "file"
    assert result["files_processed"] == 2
    assert result["rows_written"] == 3
    assert result["columns"] == schema["columns"]

    df = pd.read_parquet(output_file)
    assert list(df.columns) == schema["columns"]
    assert "extra.col" not in df.columns
    assert len(df) == 3
    assert pd.isna(df.loc[1, "value.count"])


def test_csv_to_parquet_dataset_mode_local(tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "only.csv.gz"
    _write_csv(
        input_file,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 10.0,
                "value.count": 1,
                "value.flag": "false",
                "value.note": "n1",
            },
            {
                "key.projectId": "projA",
                "key.userId": "u2",
                "value.time": 11.0,
                "value.count": 2,
                "value.flag": "true",
                "value.note": "n2",
            },
        ],
    )

    output_dir = tmp_path / "dataset_out"
    result = csv_to_parquet(
        [input_file],
        output_path=output_dir,
        schema=schema,
        output_mode="dataset",
    )

    assert output_dir.exists()
    assert output_dir.is_dir()
    assert result["output_mode"] == "dataset"
    assert result["rows_written"] == 2

    df = pd.read_parquet(output_dir)
    assert len(df) == 2
    assert list(df.columns) == schema["columns"]


def test_csv_to_parquet_with_explicit_fs(tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 1.0,
                "value.count": 7,
                "value.flag": "1",
                "value.note": "ok",
            },
        ],
    )

    output_file = tmp_path / "explicit.parquet"
    fs = fsspec.filesystem("file")

    result = csv_to_parquet(
        [input_file.as_uri()],
        output_path=output_file.as_uri(),
        schema=schema,
        fs=fs,
        output_mode="file",
    )

    assert result["files_processed"] == 1
    assert output_file.exists()


def test_csv_to_parquet_rejects_fs_and_storage_options(tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 1.0,
            },
        ],
    )

    output_file = tmp_path / "conflict.parquet"
    fs = fsspec.filesystem("file")

    with pytest.raises(ValueError, match="either fs= or storage_options="):
        csv_to_parquet(
            [input_file],
            output_path=output_file,
            schema=schema,
            fs=fs,
            storage_options={"anon": True},
        )


def test_csv_to_parquet_protocol_mismatch(tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 1.0,
            },
        ],
    )

    with pytest.raises(ValueError):
        csv_to_parquet(
            [input_file],
            output_path="memory://bucket/out.parquet",
            schema=schema,
        )


def test_csv_to_parquet_overwrite_behavior(tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 1.0,
                "value.count": 1,
                "value.flag": "true",
                "value.note": "first",
            },
        ],
    )

    output_file = tmp_path / "overwrite.parquet"
    csv_to_parquet(
        [input_file], output_path=output_file, schema=schema, output_mode="file"
    )

    with pytest.raises(FileExistsError, match="already exists"):
        csv_to_parquet(
            [input_file], output_path=output_file, schema=schema, output_mode="file"
        )

    result = csv_to_parquet(
        [input_file],
        output_path=output_file,
        schema=schema,
        output_mode="file",
        overwrite=True,
    )
    assert result["rows_written"] == 1


def test_csv_to_parquet_schema_validation(tmp_path):
    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [
            {
                "key.projectId": "projA",
                "key.userId": "u1",
                "value.time": 1.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="missing required keys"):
        csv_to_parquet([input_file], output_path=tmp_path / "out.parquet", schema={})


@pytest.mark.parametrize(
    "data_type",
    ["android_phone_gyroscope", "questionnaire_app_event"],
)
def test_csv_to_parquet_mockdata_submodule_smoke(tmp_path, data_type):
    schema_path, csv_paths = _find_mockdata_data_type(data_type)
    if schema_path is None or not csv_paths:
        pytest.skip(f"No usable schema/csv pair found for {data_type!r} under mockdata")

    schema = build_schema(schema_path)
    output_file = tmp_path / "mockdata-out.parquet"

    result = csv_to_parquet(
        [str(path) for path in csv_paths],
        output_path=output_file,
        schema=schema,
        output_mode="file",
    )

    df = pd.read_parquet(output_file)

    assert output_file.exists()
    assert result["files_processed"] == len(csv_paths)
    assert result["rows_written"] == len(df)
    assert result["rows_written"] > 0
    assert list(df.columns) == schema["columns"]


def test_csv_to_parquet_questionnaire_with_observed_columns(tmp_path):
    data_type = "questionnaire_phq8"
    schema_path, csv_paths = _find_mockdata_data_type(data_type)
    if schema_path is None or not csv_paths:
        pytest.skip(f"No usable schema/csv pair found for {data_type!r} under mockdata")

    observed_columns = [_read_gzip_header(path) for path in csv_paths]
    schema = build_schema(schema_path, observed_columns=observed_columns)

    answer_columns = [
        column for column in observed_columns[0] if column.startswith("value.answers.")
    ]
    assert answer_columns
    assert set(answer_columns).issubset(set(schema["columns"]))

    output_file = tmp_path / "questionnaire-phq8-out.parquet"
    result = csv_to_parquet(
        [str(path) for path in csv_paths],
        output_path=output_file,
        schema=schema,
        output_mode="file",
    )

    df = pd.read_parquet(output_file)
    assert output_file.exists()
    assert result["rows_written"] > 0
    assert result["rows_written"] == len(df)
    assert set(answer_columns).issubset(set(df.columns))
    assert df[answer_columns].notna().any().any()


def test_convert_private_path_helpers():
    assert convert_module._split_parent_name("out.parquet") == ("", "out.parquet")
    assert convert_module._join_fs_path("", "out.parquet") == "out.parquet"
    assert convert_module._infer_csv_compression("input.csv") is None

    temp_target = convert_module._build_temp_target("")
    assert temp_target.startswith(".parquet.tmp-")


def test_validate_schema_error_paths():
    with pytest.raises(ValueError, match="dict produced by build_schema"):
        convert_module._validate_schema("not-a-dict")

    base = {
        "columns": ["a"],
        "pandas_dtypes": {"a": "string"},
        "meta": pd.DataFrame({"a": pd.Series(dtype="string")}),
        "pyarrow_schema": pa.schema([pa.field("a", pa.string())]),
    }

    bad = dict(base)
    bad["columns"] = "a"
    with pytest.raises(ValueError, match="columns"):
        convert_module._validate_schema(bad)

    bad = dict(base)
    bad["pandas_dtypes"] = []
    with pytest.raises(ValueError, match="pandas_dtypes"):
        convert_module._validate_schema(bad)

    bad = dict(base)
    bad["meta"] = {}
    with pytest.raises(ValueError, match="meta"):
        convert_module._validate_schema(bad)

    bad = dict(base)
    bad["pyarrow_schema"] = "not-schema"
    with pytest.raises(ValueError, match="pyarrow_schema"):
        convert_module._validate_schema(bad)

    bad = dict(base)
    bad["pandas_dtypes"] = {}
    with pytest.raises(ValueError, match="missing columns"):
        convert_module._validate_schema(bad)

    bad = dict(base)
    bad["pyarrow_schema"] = pa.schema([pa.field("b", pa.string())])
    with pytest.raises(ValueError, match="column order must match"):
        convert_module._validate_schema(bad)


def test_read_batch_error_and_empty_paths(tmp_path):
    columns = ["a"]
    dtypes = {"a": "string"}
    fs = fsspec.filesystem("file")

    with pytest.raises(ValueError, match="either fs= or fs_json="):
        convert_module._read_batch([], columns, dtypes)

    with pytest.raises(ValueError, match="either fs= or fs_json=, not both"):
        convert_module._read_batch([], columns, dtypes, fs=fs, fs_json=fs.to_json())

    out = convert_module._read_batch([], columns, dtypes, fs=fs)
    assert list(out.columns) == columns
    assert out.empty


def test_read_batch_with_fs_json(tmp_path):
    columns = ["a"]
    dtypes = {"a": "string"}
    input_file = tmp_path / "tiny.csv"
    input_file.write_text("a\nx\n")

    fs = fsspec.filesystem("file")
    out = convert_module._read_batch(
        [str(input_file)],
        columns,
        dtypes,
        fs_json=fs.to_json(),
    )

    assert list(out.columns) == columns
    assert out.iloc[0, 0] == "x"


def test_prepare_target_creates_parent_dir_with_memory_fs():
    fs = fsspec.filesystem("memory")
    output_path = "bucket/newdir/out.parquet"
    convert_module._prepare_target(fs, output_path, overwrite=False)
    assert fs.exists("bucket/newdir")


def test_csv_to_parquet_invalid_output_mode(tmp_path):
    schema = _build_schema_output(tmp_path)
    with pytest.raises(ValueError, match="output_mode"):
        csv_to_parquet(
            [],
            output_path=tmp_path / "out.parquet",
            schema=schema,
            output_mode="bad",
        )


def test_csv_to_parquet_string_csv_path(tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "single.csv.gz"
    _write_csv(
        input_file,
        [{"key.projectId": "projA", "key.userId": "u1", "value.time": 1.0}],
    )
    output_file = tmp_path / "single.parquet"

    result = csv_to_parquet(
        str(input_file),
        output_path=output_file,
        schema=schema,
        output_mode="file",
    )
    assert result["files_processed"] == 1


def test_csv_to_parquet_empty_paths_and_invalid_chunk_size(tmp_path):
    schema = _build_schema_output(tmp_path)
    with pytest.raises(ValueError, match="csv_paths is empty"):
        csv_to_parquet([], output_path=tmp_path / "out.parquet", schema=schema)

    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [{"key.projectId": "projA", "key.userId": "u1", "value.time": 1.0}],
    )
    with pytest.raises(ValueError, match="files_per_task"):
        csv_to_parquet(
            [input_file],
            output_path=tmp_path / "out.parquet",
            schema=schema,
            files_per_task=0,
        )


def test_csv_to_parquet_no_resolved_csv_paths(monkeypatch, tmp_path):
    schema = _build_schema_output(tmp_path)
    fs = fsspec.filesystem("file")

    def _fake_resolve_paths(paths, *, fs=None, storage_options=None):  # noqa: ARG001
        return fs, ["only-output.parquet"]

    monkeypatch.setattr(convert_module, "resolve_paths", _fake_resolve_paths)

    with pytest.raises(ValueError, match="No CSV paths were resolved"):
        csv_to_parquet(
            ["dummy.csv.gz"],
            output_path="dummy-out.parquet",
            schema=schema,
            fs=fs,
        )


def test_csv_to_parquet_file_mode_requires_single_part(monkeypatch, tmp_path):
    schema = _build_schema_output(tmp_path)
    input_file = tmp_path / "input.csv.gz"
    _write_csv(
        input_file,
        [{"key.projectId": "projA", "key.userId": "u1", "value.time": 1.0}],
    )

    def _fake_compute_dask(  # noqa: ARG001
        tasks, *, client=None, show_progress=False, scheduler="threads"
    ):
        return [None, 1]

    monkeypatch.setattr(convert_module, "compute_dask", _fake_compute_dask)

    with pytest.raises(ValueError, match="Expected a single parquet part file"):
        csv_to_parquet(
            [input_file],
            output_path=tmp_path / "out.parquet",
            schema=schema,
            output_mode="file",
        )
