"""Tests for filesystem helpers."""

import concurrent.futures
import os
import threading

import fsspec
import pytest

import radarbase_io.fs as fs_module
from radarbase_io.fs import call_with_fs_json, fs_to_json, resolve_paths


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


def test_call_with_fs_json_uses_cache(monkeypatch):
    fs_module._clear_fs_json_cache()
    calls = {"count": 0}

    class FakeFs:
        value = "ok"

    def fake_fs_from_json(_fs_json):
        calls["count"] += 1
        return FakeFs()

    monkeypatch.setattr(fs_module, "fs_from_json", fake_fs_from_json)

    out1 = call_with_fs_json("serialized", lambda fs: fs.value, operation="op")
    out2 = call_with_fs_json("serialized", lambda fs: fs.value, operation="op")

    assert out1 == "ok"
    assert out2 == "ok"
    assert calls["count"] == 1
    fs_module._clear_fs_json_cache()


def test_call_with_fs_json_retries_and_refreshes(monkeypatch):
    fs_module._clear_fs_json_cache()
    calls = {"count": 0}

    class FakeFs:
        pass

    def fake_fs_from_json(_fs_json):
        calls["count"] += 1
        return FakeFs()

    monkeypatch.setattr(fs_module, "fs_from_json", fake_fs_from_json)

    attempts = {"count": 0}

    def flaky_op(_fs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionError("Connection lost")
        return "done"

    result = call_with_fs_json(
        "serialized",
        flaky_op,
        operation="flaky op",
        attempts=2,
        backoff_seconds=0,
    )

    assert result == "done"
    assert attempts["count"] == 2
    assert calls["count"] == 2
    fs_module._clear_fs_json_cache()


def test_call_with_fs_json_retry_exhausted_raises(monkeypatch):
    fs_module._clear_fs_json_cache()

    class FakeFs:
        pass

    monkeypatch.setattr(fs_module, "fs_from_json", lambda _fs_json: FakeFs())

    def always_fail(_fs):
        raise ConnectionError("Connection lost")

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        call_with_fs_json(
            "serialized",
            always_fail,
            operation="failing op",
            attempts=2,
            backoff_seconds=0,
        )

    fs_module._clear_fs_json_cache()


def test_call_with_fs_json_uses_thread_local_cache(monkeypatch):
    fs_module._clear_fs_json_cache()
    calls = {"count": 0}

    class FakeFs:
        pass

    def fake_fs_from_json(_fs_json):
        calls["count"] += 1
        return FakeFs()

    monkeypatch.setattr(fs_module, "fs_from_json", fake_fs_from_json)

    barrier = threading.Barrier(2)

    def read_once():
        def use_fs(fs):
            barrier.wait()
            return id(fs)

        return call_with_fs_json("serialized", use_fs, operation="threaded op")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read_once) for _ in range(2)]
        fs_ids = [future.result() for future in futures]

    assert calls["count"] == 2
    assert len(set(fs_ids)) == 2
    fs_module._clear_fs_json_cache()
