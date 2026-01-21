"""Acceleration helpers (Dask/Numba/JIT related)."""

from collections.abc import Iterable

from dask import compute as dask_compute
from dask.distributed import progress as dask_progress


def run_dask_tasks(
    tasks: Iterable,
    *,
    client=None,
    show_progress: bool = False,
    scheduler: str = "threads",
):
    """
    Execute Dask delayed tasks locally or via a distributed client.

    Parameters
    ----------
    tasks : iterable
        Dask delayed objects to execute.
    client : dask.distributed.Client, optional
        Distributed client used to schedule tasks.
    show_progress : bool, default False
        If True, display a progress bar when using a distributed client.
    scheduler : str, default "threads"
        Dask scheduler name used when no client is provided.
    """
    tasks = list(tasks)
    if not tasks:
        return []

    if client is None:
        return list(dask_compute(*tasks, scheduler=scheduler))

    futures = client.compute(tasks)
    if show_progress:
        dask_progress(futures)
    return client.gather(futures)
