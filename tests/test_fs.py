"""Tests for filesystem helpers."""

import os

import fsspec
import pytest

from radarbase_io.fs import fs_to_json, resolve_paths


def test_resolve_paths_local(tmp_path):
    path = tmp_path / "root"
    fs, paths = resolve_paths(path)
    assert paths == [os.fspath(path)]
    assert fs is not None


def test_resolve_paths_explicit_fs(tmp_path):
    fs = fsspec.filesystem("file")
    path = tmp_path / "root"
    uri = path.as_uri()
    fs_out, paths = resolve_paths(uri, fs=fs)
    assert fs_out is fs
    assert paths == [fs._strip_protocol(uri)]


def test_resolve_paths_rejects_fs_and_storage_options(tmp_path):
    fs = fsspec.filesystem("file")
    with pytest.raises(ValueError, match="either fs= or storage_options="):
        resolve_paths(tmp_path, fs=fs, storage_options={"anon": True})


def test_resolve_paths_protocol_mismatch(tmp_path):
    paths = [tmp_path / "root", "memory://bucket/root"]
    with pytest.raises(ValueError):
        resolve_paths(paths)


def test_fs_to_json_roundtrip_smoke():
    fs = fsspec.filesystem("file")
    serialized = fs_to_json(fs)
    assert isinstance(serialized, str)
    assert serialized


def test_resolve_paths_explicit_fs_strip_protocol_failure_falls_back():
    class BrokenStripProtocolFs:
        def _strip_protocol(self, _path):
            raise RuntimeError("boom")

    fs = BrokenStripProtocolFs()
    fs_out, paths = resolve_paths("memory://bucket/root", fs=fs)
    assert fs_out is fs
    assert paths == ["memory://bucket/root"]
