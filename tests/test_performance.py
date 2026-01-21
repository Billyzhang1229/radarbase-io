"""Tests for performance helpers."""

from dask import delayed

from radarbase_io.performance import run_dask_tasks


def _add_one(value):
    return value + 1


def _return_ok():
    return "ok"


def test_run_dask_tasks_empty():
    assert run_dask_tasks([]) == []


def test_run_dask_tasks_local_compute():
    tasks = [delayed(_add_one)(1), delayed(_return_ok)()]
    out = run_dask_tasks(tasks, scheduler="threads")
    assert out == [2, "ok"]
