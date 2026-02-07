"""Acceleration helpers (Dask/Numba/JIT related)."""

from dask import compute
from dask.base import is_dask_collection
from dask.delayed import Delayed
from dask.distributed import progress


def _normalize_tasks(tasks):
    if tasks is None:
        return []
    if isinstance(tasks, Delayed) or is_dask_collection(tasks):
        return [tasks]
    try:
        return list(tasks)
    except TypeError:
        return [tasks]


def compute_dask(tasks, *, client=None, show_progress=False, scheduler="threads"):
    """
    Execute Dask tasks locally or via a distributed client.

    Parameters
    ----------
    tasks : object or iterable
        A Dask task/collection or a sequence of tasks/collections.
    client : dask.distributed.Client, optional
        Distributed client used to schedule tasks.
    show_progress : bool, default False
        If True, display a progress bar when using a distributed client.
    scheduler : str, default "threads"
        Dask scheduler name used when no client is provided.
    """
    tasks = _normalize_tasks(tasks)
    if not tasks:
        return []

    if client is None:
        return list(compute(*tasks, scheduler=scheduler))

    futures = client.compute(tasks)
    if show_progress:
        progress(futures)
    return client.gather(futures)
