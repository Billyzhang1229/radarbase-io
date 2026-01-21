"""Index/manifest build, load, and query helpers."""

from itertools import chain

import pandas as pd
from dask import delayed

from .fs import fs_from_json, fs_to_json, resolve_paths
from .performance import run_dask_tasks
from .utils import parse_participant_uuids, parse_radar_path


def list_one_participant(participant_dir, fs=None, *, fs_json=None):
    """
    List one participant directory and parse its immediate children.

    Parameters
    ----------
    participant_dir : str
        Participant directory path.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance.
    fs_json : str, optional
        JSON representation of a filesystem created with `fs.to_json()`.
    Returns
    -------
    list of dict
        Parsed records for each data_type directory under the participant.

    Raises
    ------
    ValueError
        If neither `fs` nor `fs_json` is provided, if both are passed, or if a
        child path does not match the expected RADAR layout.
    """
    if fs is None:
        if fs_json is None:
            raise ValueError("Pass either fs= or fs_json=.")
        fs = fs_from_json(fs_json)
    elif fs_json is not None:
        raise ValueError("Pass either fs= or fs_json=, not both.")

    dir_paths = fs.ls(participant_dir, detail=False)
    rows = [parse_radar_path(path) for path in dir_paths]

    return rows


def _build_index(
    root,
    fs=None,
    *,
    client=None,
    show_progress=False,
    storage_options=None,
):
    """Build an index DataFrame for a single project root."""
    fs, root_paths = resolve_paths(root, fs=fs, storage_options=storage_options)
    if not root_paths:
        return pd.DataFrame()
    root_path = root_paths[0]

    entries = fs.ls(root_path, detail=False)
    candidate_paths = entries

    participant_ids = parse_participant_uuids(candidate_paths)
    root_norm = root_path.rstrip("/")
    participant_dirs = [f"{root_norm}/{pid}" for pid in participant_ids]

    kwargs = {"fs": fs} if client is None else {"fs_json": fs_to_json(fs)}
    tasks = [delayed(list_one_participant)(path, **kwargs) for path in participant_dirs]
    rows_nested = run_dask_tasks(
        tasks, client=client, show_progress=show_progress, scheduler="threads"
    )

    records = list(chain.from_iterable(rows_nested))
    df = pd.DataFrame.from_records(records)

    return df


def build_index(paths, fs=None, *, client=None, storage_options=None, **kwargs):
    """
    Build an index DataFrame across multiple project roots.

    Parameters
    ----------
    paths : str or sequence of str
        One project root or many.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance (shared across roots if provided).
    client : dask.distributed.Client, optional
        Dask client for parallel listing.
    storage_options : dict, optional
        Storage options used when resolving paths with fsspec.
    **kwargs
        Forwarded to `_build_index`.

    Returns
    -------
    pandas.DataFrame
        Concatenated index across all roots.

    Raises
    ------
    ValueError
        If no data is found in any of the provided paths.
    """
    if isinstance(paths, str):
        paths = [paths]

    fs, root_paths = resolve_paths(paths, fs=fs, storage_options=storage_options)
    child_storage_options = None if fs is not None else storage_options
    dfs = [
        _build_index(
            path,
            fs=fs,
            client=client,
            storage_options=child_storage_options,
            **kwargs,
        )
        for path in root_paths
    ]
    dfs = [frame for frame in dfs if not frame.empty]

    if not dfs:
        raise ValueError("No data found in any of the provided paths.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["project_id", "user_id", "data_type"]).reset_index(drop=True)
    return df
