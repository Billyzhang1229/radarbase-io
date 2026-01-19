"""Index/manifest build, load, and query helpers."""

from itertools import chain

import pandas as pd
from dask import compute, delayed
from fsspec.spec import AbstractFileSystem

from .fs import normalize_ls, resolve_fs
from .utils import parse_participant_uuids, parse_radar_path


def list_one_participant(participant_dir, fs, *, check_schema=False):
    """
    List one participant directory and parse its immediate children.

    Parameters
    ----------
    participant_dir : str
        Participant directory path.
    fs : fsspec.AbstractFileSystem
        Filesystem instance.
    check_schema : bool, default False
        If True, add `schema_exists` by checking `fs.exists(schema_path)`.

    Returns
    -------
    list of dict
        Parsed records for each data_type directory under the participant.

    Raises
    ------
    ValueError
        If a child path does not match the expected RADAR layout.
    """
    dir_paths = fs.ls(participant_dir, detail=False)
    rows = [parse_radar_path(path) for path in dir_paths]

    if check_schema:
        for record in rows:
            record["schema_exists"] = fs.exists(record["schema_path"])

    return rows


def _list_one_participant_from_json(participant_dir, fs_json, check_schema=False):
    """Reconstruct a filesystem from JSON and list a participant directory."""
    fs = AbstractFileSystem.from_json(fs_json)
    return list_one_participant(participant_dir, fs, check_schema=check_schema)


def _build_index(
    root,
    fs=None,
    *,
    ensure_dirs=False,
    check_schema=False,
    client=None,
    use_dask=True,
    show_progress=False,
    storage_options=None,
):
    """Build an index DataFrame for a single project root."""
    fs, root_path = resolve_fs(root, fs=fs, storage_options=storage_options)

    entries = fs.ls(root_path, detail=ensure_dirs)

    if ensure_dirs:
        entries = normalize_ls(entries)  # only meaningful for detail=True
        candidate_paths = [e["name"] for e in entries if e.get("type") == "directory"]
    else:
        candidate_paths = entries  # list[str]

    participant_ids = parse_participant_uuids(candidate_paths)
    root_norm = root_path.rstrip("/")
    participant_dirs = [f"{root_norm}/{pid}" for pid in participant_ids]

    # execute
    if use_dask and client is not None:
        fs_json = fs.to_json()
        futures = client.map(
            _list_one_participant_from_json,
            participant_dirs,
            fs_json=fs_json,
            check_schema=check_schema,
        )
        if show_progress:
            from dask.distributed import progress

            progress(futures)

        rows_nested = client.gather(futures)

    elif use_dask:
        tasks = [
            delayed(list_one_participant)(path, fs, check_schema=check_schema)
            for path in participant_dirs
        ]

        if show_progress:
            try:
                from dask.diagnostics import ProgressBar

                with ProgressBar():
                    rows_nested = compute(*tasks, scheduler="threads")
            except Exception:
                rows_nested = compute(*tasks, scheduler="threads")
        else:
            rows_nested = compute(*tasks, scheduler="threads")

    else:
        rows_nested = [
            list_one_participant(path, fs, check_schema=check_schema)
            for path in participant_dirs
        ]

    records = list(chain.from_iterable(rows_nested))
    df = pd.DataFrame.from_records(records)

    if not df.empty:
        df = df.sort_values(["project_id", "user_id", "data_type"]).reset_index(
            drop=True
        )

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
        Storage options passed to `fsspec.core.url_to_fs`.
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

    dfs = [
        _build_index(
            path,
            fs=fs,
            client=client,
            storage_options=storage_options,
            **kwargs,
        )
        for path in paths
    ]
    dfs = [frame for frame in dfs if not frame.empty]

    if not dfs:
        raise ValueError("No data found in any of the provided paths.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["project_id", "user_id", "data_type"]).reset_index(drop=True)
    return df
