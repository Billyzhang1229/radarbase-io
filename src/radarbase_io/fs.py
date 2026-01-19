"""Filesystem helpers built on fsspec."""

import os

import fsspec


def fs_from_json(fs_json):
    """Reconstruct a filesystem from JSON produced by fs.to_json()."""
    return fsspec.spec.AbstractFileSystem.from_json(fs_json)


def fs_to_json(fs):
    """Serialize a filesystem into JSON for transport."""
    return fs.to_json()


def normalize_ls(entries):
    """
    Normalize `fs.ls(..., detail=True)` output.

    Parameters
    ----------
    entries : dict or list of dict or list of str
        Output from `fs.ls(..., detail=True)` or compatible listing.

    Returns
    -------
    list of dict
        Normalized list of entry dictionaries. If `entries` is a mapping, each
        value is copied and the key is stored as `name` when missing. If
        `entries` is a list of strings, each entry is mapped to `{"name": ...}`.
    """
    if isinstance(entries, dict):
        out = []
        for name, info in entries.items():
            record = dict(info) if isinstance(info, dict) else {}
            record.setdefault("name", name)
            out.append(record)
        return out
    if isinstance(entries, (list, tuple)):
        out = []
        for item in entries:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"name": item})
        return out
    return []


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
            raise ValueError("Pass either fs= or storage_options=, not both.")
        try:
            path = fs._strip_protocol(path)
        except Exception:
            pass
        return fs, path

    opts = dict(storage_options or {})
    return fsspec.core.url_to_fs(path, **opts)


def resolve_many(paths, *, fs=None, storage_options=None, enforce_same_fs=True):
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
    enforce_same_fs : bool, default True
        If True, require all paths to resolve to the same filesystem and return
        a single shared fs instance. If False, resolve each path independently.

    Returns
    -------
    list of (fs, fs_path)

    Raises
    ------
    ValueError
        If `storage_options` is provided together with `fs`.

    """
    paths = [os.fspath(path) for path in paths]

    if fs is not None:
        if storage_options:
            raise ValueError("Pass either fs= or storage_options=, not both.")
        out = []
        for fs_path in paths:
            try:
                fs_path = fs._strip_protocol(fs_path)
            except Exception:
                pass
            out.append((fs, fs_path))
        return out

    if not paths:
        return []

    if not enforce_same_fs:
        return [resolve_fs(path, storage_options=storage_options) for path in paths]

    opts = dict(storage_options or {})
    fs0, _token, fs_paths = fsspec.core.get_fs_token_paths(
        paths, mode="rb", storage_options=opts, expand=False
    )
    return [(fs0, path) for path in fs_paths]
