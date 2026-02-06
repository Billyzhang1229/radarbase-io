"""Tests for utility helpers."""

import numpy as np
import pandas as pd
import pytest

from radarbase_io.utils import (
    align_and_coerce_dataframe,
    coerce_boolean_series,
    coerce_series_dtype,
    parse_participant_uuids,
    parse_radar_path,
)

UUID1 = "11111111-1111-1111-1111-111111111111"
UUID2 = "22222222-2222-2222-2222-222222222222"


def test_parse_participant_uuids_single_valid():
    assert parse_participant_uuids(UUID1) == UUID1


def test_parse_participant_uuids_single_invalid():
    assert parse_participant_uuids("not-a-uuid") is None


def test_parse_participant_uuids_trailing_slash():
    assert parse_participant_uuids(f"/x/{UUID1}/") == UUID1


def test_parse_participant_uuids_sequence_filters_invalid():
    paths = [f"/root/{UUID1}", "/root/not-a-uuid", f"/root/{UUID2}/"]
    out = parse_participant_uuids(paths)
    assert isinstance(out, np.ndarray)
    assert set(out.tolist()) == {UUID1, UUID2}


def test_parse_radar_path_basic():
    path = f"/data/proj/{UUID1}/accel"
    result = parse_radar_path(path)

    assert result["project_id"] == "proj"
    assert result["user_id"] == UUID1
    assert result["data_type"] == "accel"
    assert result["schema_file"] == "schema-accel.json"
    assert result["path"] == path
    assert result["schema_path"] == f"/data/proj/{UUID1}/accel/schema-accel.json"


def test_parse_radar_path_trailing_slash():
    result = parse_radar_path(f"/data/proj/{UUID1}/accel/")
    assert result["data_type"] == "accel"


def test_parse_radar_path_windows_path():
    path = f"C:\\data\\proj\\{UUID1}\\accel"
    result = parse_radar_path(path)

    assert result["project_id"] == "proj"
    assert result["user_id"] == UUID1
    assert result["data_type"] == "accel"
    assert result["schema_path"].replace("\\", "/") == (
        f"C:/data/proj/{UUID1}/accel/schema-accel.json"
    )


def test_parse_radar_path_too_short_raises():
    with pytest.raises(ValueError, match="Expected .*<project_id>"):
        parse_radar_path("a/b")


def test_coerce_boolean_series():
    series = pd.Series(["true", "FALSE", "1", "0", "unknown", None])
    out = coerce_boolean_series(series)

    assert str(out.dtype) == "boolean"
    assert out.tolist() == [True, False, True, False, pd.NA, pd.NA]


def test_coerce_boolean_series_numeric_unknown():
    series = pd.Series([2, -1, 0, 1])
    out = coerce_boolean_series(series)
    assert out.tolist() == [pd.NA, pd.NA, False, True]


def test_coerce_series_dtype_numeric():
    float_series = pd.Series(["1.2", "bad", None])
    float_out = coerce_series_dtype(float_series, "float64")
    assert str(float_out.dtype) == "float64"
    assert pd.isna(float_out.iloc[1])

    int_series = pd.Series(["1", "bad", None])
    int_out = coerce_series_dtype(int_series, "Int64")
    assert str(int_out.dtype) == "Int64"
    assert pd.isna(int_out.iloc[1])


def test_coerce_series_dtype_fallback():
    series = pd.Series(["a", "b", "a"])
    out = coerce_series_dtype(series, "category")
    assert str(out.dtype) == "category"


def test_align_and_coerce_dataframe():
    df = pd.DataFrame(
        {
            "col_a": ["1", "bad"],
            "col_extra": ["x", "y"],
            "col_b": ["true", "0"],
        }
    )
    columns = ["col_b", "col_c", "col_a"]
    pandas_dtypes = {"col_b": "boolean", "col_c": "string", "col_a": "Int64"}

    out = align_and_coerce_dataframe(df, columns, pandas_dtypes)

    assert list(out.columns) == columns
    assert "col_extra" not in out.columns
    assert str(out["col_b"].dtype) == "boolean"
    assert str(out["col_c"].dtype) == "string"
    assert str(out["col_a"].dtype) == "Int64"
    assert out["col_c"].isna().all()
    assert pd.isna(out.loc[1, "col_a"])
