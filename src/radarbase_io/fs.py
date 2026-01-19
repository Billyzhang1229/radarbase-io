"""Filesystem helpers built on fsspec."""

import os

import fsspec

__all__ = ["normalize_ls", "resolve_fs", "resolve_many"]


def normalize_ls(entries):
    """
    Normalize `fs.ls(..., detail=True)` output.

    Parameters
    ----------
    entries : dict or list of dict
        Output from `fs.ls(..., detail=True)`.

    Returns
    -------
    list of dict
        Normalized list of entry dictionaries. If `entries` is a mapping, each
        value is copied and the key is stored as `name` when missing.
    """
    if isinstance(entries, dict):
        out = []
        for name, info in entries.items():
            record = dict(info) if isinstance(info, dict) else {}
            record.setdefault("name", name)
            out.append(record)
        return out
    return entries


def resolve_fs(path, *, fs=None, storage_options=None):
    """
    Resolve a path into an (fs, fs_path) pair.

    Parameters
    ----------
    path : str or os.PathLike
        Local path or URL (for example, file://, s3://, sftp://, ssh://).
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance to use.
    storage_options : dict, optional
        Passed to `url_to_fs` when `fs` is not provided.

    Returns
    -------
    tuple
        `(fs, fs_path)` where `fs_path` is suitable for methods on `fs`.

    Raises
    ------
    ValueError
        If `storage_options` is provided together with `fs`.

    Notes
    -----
    When `fs` is provided, the path is stripped of protocol with
    `fs._strip_protocol` when available.
    """
    path = os.fspath(path)

    if fs is not None:
        if storage_options:
            raise ValueError("storage_options is ignored when fs is provided.")
        try:
            path = fs._strip_protocol(path)
        except Exception:
            pass
        return fs, path

    return fsspec.core.url_to_fs(path, **(storage_options or {}))


def resolve_many(paths, *, fs=None, storage_options=None):
    """
    Resolve multiple paths into a list of (fs, fs_path) pairs.

    Parameters
    ----------
    paths : sequence of str or os.PathLike
        Paths/URLs to resolve.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance to use for all paths.
    storage_options : dict, optional
        Passed to `url_to_fs` when `fs` is not provided.

    Returns
    -------
    list of (fs, fs_path)

    Raises
    ------
    ValueError
        If `storage_options` is provided together with `fs`.

    Notes
    -----
    If `fs` is not provided, `storage_options` is applied to every path. This
    assumes the paths share compatible options (typically same protocol).
    """
    paths = list(paths)

    if fs is not None:
        if storage_options:
            raise ValueError("storage_options is ignored when fs is provided.")
        out = []
        for path in paths:
            fs_path = os.fspath(path)
            try:
                fs_path = fs._strip_protocol(fs_path)
            except Exception:
                pass
            out.append((fs, fs_path))
        return out

    return [resolve_fs(path, storage_options=storage_options) for path in paths]
