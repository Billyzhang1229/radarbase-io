"""Tests for performance helpers."""

from dask import delayed

import radarbase_io.performance as perf
from radarbase_io.performance import run_dask_tasks


def _add_one(value):
    return value + 1


def _return_ok():
    return "ok"


def test_run_dask_tasks_empty():
    assert run_dask_tasks([]) == []


def test_run_dask_tasks_single_delayed():
    task = delayed(_add_one)(1)
    assert run_dask_tasks(task, scheduler="threads") == [2]


def test_run_dask_tasks_local_compute():
    tasks = [delayed(_add_one)(1), delayed(_return_ok)()]
    out = run_dask_tasks(tasks, scheduler="threads")
    assert out == [2, "ok"]


class _FakeClient:
    def __init__(self):
        self.computed = None

    def compute(self, tasks):
        self.computed = list(tasks)
        return [("future", i) for i, _ in enumerate(self.computed)]

    def gather(self, futures):
        return [f"result-{item[1]}" for item in futures]


def test_run_dask_tasks_client_compute_and_gather():
    tasks = [delayed(_add_one)(1), delayed(_return_ok)()]
    client = _FakeClient()

    out = run_dask_tasks(tasks, client=client)

    assert len(client.computed) == 2
    assert out == ["result-0", "result-1"]


def test_run_dask_tasks_client_with_progress(monkeypatch):
    task = delayed(_add_one)(1)
    client = _FakeClient()
    seen = {}

    def _fake_progress(futures):
        seen["futures"] = futures

    monkeypatch.setattr(perf, "progress", _fake_progress)

    out = run_dask_tasks([task], client=client, show_progress=True)

    assert seen["futures"] == [("future", 0)]
    assert out == ["result-0"]
