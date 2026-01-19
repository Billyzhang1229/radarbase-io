"""RADAR folder parsing and discovery helpers."""

from .fs import resolve_fs
from .utils import parse_participant_uuids


def list_participants(root, *, fs=None, storage_options=None):
    """
    List participant directories under a RADAR-like project root.

    Parameters
    ----------
    root : str or os.PathLike
        Root path to RADAR-like project data.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance to use.
    storage_options : dict, optional
        Storage options passed to `fsspec.core.url_to_fs` when `fs` is not
        provided.

    Returns
    -------
    numpy.ndarray
        Array of participant UUID strings found directly under `root`.

    Notes
    -----
    Only the immediate children of `root` are inspected; non-UUID folders are
    ignored.
    """
    fs, root_path = resolve_fs(root, fs=fs, storage_options=storage_options)
    entries = fs.ls(root_path, detail=False)
    participants = parse_participant_uuids(entries)

    return participants
