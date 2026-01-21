"""Tests for index helpers."""

import fsspec
import pandas as pd
import pytest

from radarbase_io.index import _build_index, build_index, list_one_participant

UUID1 = "11111111-1111-1111-1111-111111111111"
UUID2 = "22222222-2222-2222-2222-222222222222"


def _make_tree(root):
    (root / UUID1 / "accel").mkdir(parents=True)
    (root / UUID1 / "gps").mkdir()
    (root / UUID2 / "accel").mkdir(parents=True)
    (root / "not-a-uuid").mkdir()


def test_list_one_participant_requires_fs_or_fs_json():
    with pytest.raises(ValueError, match="either fs= or fs_json="):
        list_one_participant("/some/path")


def test_list_one_participant_fs_json(tmp_path):
    root = tmp_path / "projA"
    participant = root / UUID1
    (participant / "accel").mkdir(parents=True)

    fs = fsspec.filesystem("file")
    rows = list_one_participant(str(participant), fs_json=fs.to_json())

    assert len(rows) == 1
    assert rows[0]["user_id"] == UUID1
    assert rows[0]["data_type"] == "accel"


def test_list_one_participant_rejects_fs_and_fs_json(tmp_path):
    root = tmp_path / "projA"
    participant = root / UUID1
    (participant / "accel").mkdir(parents=True)

    fs = fsspec.filesystem("file")
    with pytest.raises(ValueError, match="either fs= or fs_json="):
        list_one_participant(str(participant), fs=fs, fs_json=fs.to_json())


def test_build_index_local(tmp_path):
    root = tmp_path / "projA"
    _make_tree(root)

    df = build_index(root)

    assert isinstance(df, pd.DataFrame)
    assert {
        "project_id",
        "user_id",
        "data_type",
        "schema_file",
        "schema_path",
        "path",
    }.issubset(df.columns)
    assert len(df) == 3
    assert set(df["project_id"].unique()) == {"projA"}
    assert set(df["user_id"].unique()) == {UUID1, UUID2}
    assert set(df["data_type"].unique()) == {"accel", "gps"}


def test_build_index_multiple_roots(tmp_path):
    root1 = tmp_path / "projA"
    root2 = tmp_path / "projB"
    _make_tree(root1)
    _make_tree(root2)

    df = build_index([root1, root2])
    assert set(df["project_id"].unique()) == {"projA", "projB"}


def test_build_index_str_path(tmp_path):
    root = tmp_path / "projA"
    _make_tree(root)

    df = build_index(str(root))
    assert set(df["project_id"].unique()) == {"projA"}


def test_build_index_no_data_raises(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    with pytest.raises(ValueError, match="No data found"):
        build_index(empty_root)


def test_build_index_empty_root_returns_empty_dataframe():
    df = _build_index([])
    assert df.empty
