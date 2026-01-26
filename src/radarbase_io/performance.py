"""Acceleration helpers (Dask/Numba/JIT related)."""

from dask import compute
from dask.delayed import Delayed
from dask.distributed import progress


def run_dask_tasks(tasks, *, client=None, show_progress=False, scheduler="threads"):
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
    if isinstance(tasks, Delayed):
        tasks = [tasks]
    else:
        tasks = list(tasks)
    if not tasks:
        return []

    if client is None:
        return list(compute(*tasks, scheduler=scheduler))

    futures = client.compute(tasks)
    if show_progress:
        progress(futures)
    return client.gather(futures)
