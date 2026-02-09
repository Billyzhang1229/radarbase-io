"""Tests for index helpers."""

import fsspec
import pandas as pd
import pytest

import radarbase_io.fs as fs_module
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


def test_list_one_participant_skips_files(tmp_path):
    root = tmp_path / "projA"
    participant = root / UUID1
    (participant / "accel").mkdir(parents=True)
    (participant / "schema-accel.json").write_text("{}")
    (participant / ".DS_Store").write_text("noise")
    (participant / "notes.txt").write_text("ignore me")

    fs = fsspec.filesystem("file")
    rows = list_one_participant(str(participant), fs=fs)

    data_types = {row["data_type"] for row in rows}
    assert data_types == {"accel"}


def test_list_one_participant_accepts_dict_ls_entries(tmp_path, monkeypatch):
    root = tmp_path / "projA"
    participant = root / UUID1
    accel = participant / "accel"
    gps = participant / "gps"
    accel.mkdir(parents=True)
    gps.mkdir()

    fs = fsspec.filesystem("file")

    def fake_ls(_path, detail=True):
        assert detail is True
        return {
            "accel": {"name": str(accel), "type": "directory"},
            "gps": {"name": str(gps)},
        }

    monkeypatch.setattr(fs, "ls", fake_ls)
    rows = list_one_participant(str(participant), fs=fs)

    data_types = {row["data_type"] for row in rows}
    assert data_types == {"accel", "gps"}


def test_list_one_participant_accepts_non_dict_ls_entries(tmp_path, monkeypatch):
    root = tmp_path / "projA"
    participant = root / UUID1
    accel = participant / "accel"
    accel.mkdir(parents=True)
    file_path = participant / "notes.txt"
    file_path.write_text("ignore")

    fs = fsspec.filesystem("file")

    def fake_ls(_path, detail=True):
        assert detail is True
        return [str(accel), str(file_path)]

    monkeypatch.setattr(fs, "ls", fake_ls)
    rows = list_one_participant(str(participant), fs=fs)

    assert len(rows) == 1
    assert rows[0]["data_type"] == "accel"


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


def test_list_one_participant_fs_json_reuses_cached_fs(monkeypatch, tmp_path):
    fs_module._clear_fs_json_cache()
    root = tmp_path / "projA"
    participant = root / UUID1
    (participant / "accel").mkdir(parents=True)

    real_fs_from_json = fs_module.fs_from_json
    calls = {"count": 0}

    def counted_fs_from_json(fs_json):
        calls["count"] += 1
        return real_fs_from_json(fs_json)

    monkeypatch.setattr(fs_module, "fs_from_json", counted_fs_from_json)

    fs = fsspec.filesystem("file")
    rows1 = list_one_participant(str(participant), fs_json=fs.to_json())
    rows2 = list_one_participant(str(participant), fs_json=fs.to_json())

    assert len(rows1) == 1
    assert len(rows2) == 1
    assert calls["count"] == 1
    fs_module._clear_fs_json_cache()


def test_list_one_participant_fs_json_retries_connection_error(monkeypatch, tmp_path):
    fs_module._clear_fs_json_cache()
    root = tmp_path / "projA"
    participant = root / UUID1
    (participant / "accel").mkdir(parents=True)

    fs = fsspec.filesystem("file")
    real_fs_from_json = fs_module.fs_from_json
    base_fs = real_fs_from_json(fs.to_json())

    class FailingLsFs:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.failed = False

        def ls(self, path, detail=True):
            if not self.failed:
                self.failed = True
                raise ConnectionError("Connection lost")
            return self.wrapped.ls(path, detail=detail)

        def isdir(self, path):
            return self.wrapped.isdir(path)

    calls = {"count": 0}
    flaky_fs = FailingLsFs(base_fs)

    def fake_fs_from_json(_fs_json):
        calls["count"] += 1
        if calls["count"] == 1:
            return flaky_fs
        return real_fs_from_json(fs.to_json())

    monkeypatch.setattr(fs_module, "fs_from_json", fake_fs_from_json)

    rows = list_one_participant(str(participant), fs_json=fs.to_json())

    assert len(rows) == 1
    assert rows[0]["data_type"] == "accel"
    assert calls["count"] == 2
    fs_module._clear_fs_json_cache()
