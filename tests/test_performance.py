"""Tests for performance helpers."""

import dask.dataframe as dd
import pandas as pd
from dask import delayed

import radarbase_io.performance as perf
from radarbase_io.performance import compute_dask


def _add_one(value):
    return value + 1


def _return_ok():
    return "ok"


def test_compute_dask_empty():
    assert compute_dask([]) == []


def test_compute_dask_none_tasks():
    assert compute_dask(None) == []


def test_compute_dask_non_iterable_task():
    out = compute_dask(123)
    assert out == [123]


def test_compute_dask_single_delayed():
    task = delayed(_add_one)(1)
    assert compute_dask(task, scheduler="threads") == [2]


def test_compute_dask_local_compute():
    tasks = [delayed(_add_one)(1), delayed(_return_ok)()]
    out = compute_dask(tasks, scheduler="threads")
    assert out == [2, "ok"]


def test_compute_dask_single_dask_collection():
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ddf = dd.from_pandas(expected, npartitions=1)

    out = compute_dask(ddf, scheduler="threads")

    assert len(out) == 1
    pd.testing.assert_frame_equal(out[0].reset_index(drop=True), expected)


class _FakeClient:
    def __init__(self):
        self.computed = None

    def compute(self, tasks):
        self.computed = list(tasks)
        return [("future", i) for i, _ in enumerate(self.computed)]

    def gather(self, futures):
        return [f"result-{item[1]}" for item in futures]


def test_compute_dask_client_compute_and_gather():
    tasks = [delayed(_add_one)(1), delayed(_return_ok)()]
    client = _FakeClient()

    out = compute_dask(tasks, client=client)

    assert len(client.computed) == 2
    assert out == ["result-0", "result-1"]


def test_compute_dask_client_with_progress(monkeypatch):
    task = delayed(_add_one)(1)
    client = _FakeClient()
    seen = {}

    def _fake_progress(futures):
        seen["futures"] = futures

    monkeypatch.setattr(perf, "progress", _fake_progress)

    out = compute_dask([task], client=client, show_progress=True)

    assert seen["futures"] == [("future", 0)]
    assert out == ["result-0"]


def test_compute_dask_client_single_dask_collection_not_iterated():
    ddf = dd.from_pandas(pd.DataFrame({"a": [1], "b": [2]}), npartitions=1)
    client = _FakeClient()

    out = compute_dask(ddf, client=client)

    assert len(client.computed) == 1
    assert client.computed[0] is ddf
    assert out == ["result-0"]
