"""Filesystem helpers built on fsspec."""

import os
from collections.abc import Sequence

import fsspec


def fs_from_json(fs_json):
    """Reconstruct a filesystem from JSON produced by fs.to_json()."""
    return fsspec.spec.AbstractFileSystem.from_json(fs_json)


def fs_to_json(fs):
    """Serialize a filesystem into JSON for transport."""
    return fs.to_json()


def _resolve_paths(
    paths: str | os.PathLike | Sequence[str | os.PathLike],
    *,
    fs: fsspec.spec.AbstractFileSystem | None = None,
    storage_options: dict | None = None,
) -> tuple[
    fsspec.spec.AbstractFileSystem | None,
    list[str],
]:
    if fs is not None and storage_options:
        raise ValueError("Pass either fs= or storage_options=, not both.")

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    fs_paths = [os.fspath(path) for path in paths]
    if not fs_paths:
        return fs, []

    if fs is not None:
        stripped = []
        for path in fs_paths:
            try:
                path = fs._strip_protocol(path)
            except Exception:
                pass
            stripped.append(path)
        return fs, stripped

    opts = dict(storage_options or {})
    fs_out, _token, fs_paths = fsspec.core.get_fs_token_paths(
        fs_paths, mode="rb", storage_options=opts, expand=False
    )
    return fs_out, list(fs_paths)


def resolve_paths(
    paths: str | os.PathLike | Sequence[str | os.PathLike],
    *,
    fs: fsspec.spec.AbstractFileSystem | None = None,
    storage_options: dict | None = None,
) -> tuple[
    fsspec.spec.AbstractFileSystem | None,
    list[str],
]:
    """
    Resolve one or many paths into a shared filesystem and fs paths.

    Parameters
    ----------
    paths : str or os.PathLike or sequence
        Path/URL or collection of paths/URLs.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance to use.
    storage_options : dict, optional
        Passed to fsspec when resolving paths.

    Returns
    -------
    tuple
        `(fs, fs_paths)` where `fs_paths` are suitable for methods on `fs`.

    Raises
    ------
    ValueError
        If `storage_options` is provided together with `fs`.
    """
    return _resolve_paths(paths, fs=fs, storage_options=storage_options)
