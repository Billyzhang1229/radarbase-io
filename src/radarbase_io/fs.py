"""Filesystem helpers built on fsspec."""

import os
import threading
import time

import fsspec

_FS_JSON_CACHE = {}
_CONNECTION_ERROR_MARKERS = (
    "connection lost",
    "connection reset",
    "broken pipe",
    "timed out",
    "timeout",
    "network is unreachable",
    "host is unreachable",
    "temporarily unavailable",
    "ssh",
    "sftp",
)


def fs_from_json(fs_json):
    """Reconstruct a filesystem from JSON produced by fs.to_json()."""
    return fsspec.spec.AbstractFileSystem.from_json(fs_json)


def fs_to_json(fs):
    """Serialize a filesystem into JSON for transport."""
    return fs.to_json()


def _clear_fs_json_cache():
    """Clear cached filesystems (primarily for tests)."""
    _FS_JSON_CACHE.clear()


def _invalidate_fs_json_cache(fs_json):
    stale_keys = [key for key in _FS_JSON_CACHE if key[1] == fs_json]
    for key in stale_keys:
        _FS_JSON_CACHE.pop(key, None)


def _cache_key(fs_json):
    return (threading.get_ident(), fs_json)


def fs_from_json_cached(fs_json, *, refresh=False):
    """
    Reconstruct and cache a filesystem from JSON.

    A cached filesystem is reused within the same process to avoid expensive
    reconnects for every delayed task.
    """
    if refresh:
        _invalidate_fs_json_cache(fs_json)

    key = _cache_key(fs_json)
    if key not in _FS_JSON_CACHE:
        _FS_JSON_CACHE[key] = fs_from_json(fs_json)
    return _FS_JSON_CACHE[key]


def is_connection_error(error):
    """Return True when an exception looks like a transient connection issue."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    if isinstance(error, OSError):
        message = str(error).lower()
        return any(marker in message for marker in _CONNECTION_ERROR_MARKERS)
    message = str(error).lower()
    return any(marker in message for marker in _CONNECTION_ERROR_MARKERS)


def with_connection_retries(
    operation,
    fn,
    *,
    attempts=3,
    backoff_seconds=0.2,
    retry_filter=is_connection_error,
    on_retry=None,
):
    """
    Run an operation with bounded retries for transient connection failures.
    """
    if attempts <= 0:
        raise ValueError("attempts must be greater than 0")

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            should_retry = retry_filter(exc)
            if attempt >= attempts or not should_retry:
                if should_retry:
                    raise RuntimeError(
                        f"{operation} failed after {attempts} attempts due to "
                        f"connection errors. Last error: {exc}"
                    ) from exc
                raise

            if on_retry is not None:
                on_retry(attempt, exc)

            if backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))


def call_with_fs_json(
    fs_json,
    fn,
    *,
    operation,
    attempts=3,
    backoff_seconds=0.2,
):
    """
    Execute a callable with a cached filesystem reconstructed from JSON.
    """
    if fs_json is None:
        raise ValueError("fs_json must not be None")

    def _run_once():
        fs = fs_from_json_cached(fs_json)
        return fn(fs)

    def _on_retry(_attempt, _error):
        _invalidate_fs_json_cache(fs_json)

    return with_connection_retries(
        operation,
        _run_once,
        attempts=attempts,
        backoff_seconds=backoff_seconds,
        on_retry=_on_retry,
    )


def _resolve_paths(paths, *, fs=None, storage_options=None):
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


def resolve_paths(paths, *, fs=None, storage_options=None):
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
