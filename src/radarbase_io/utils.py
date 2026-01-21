import os
import re

import numpy as np


def parse_participant_uuids(paths):
    """
    Parse participant UUIDs from one or more paths.

    Parameters
    ----------
    paths : str or sequence of str
        Path or collection of paths. Only the final path component is checked.

    Returns
    -------
    str or None or numpy.ndarray
        For a single string input, returns the UUID string or None if invalid.
        For a sequence, returns an array of valid UUID strings.

    Notes
    -----
    Trailing slashes are ignored before extracting the basename.
    """
    uuid_pattern = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )

    def extract_uuid(path):
        base = os.path.basename(path.rstrip("/"))
        return base if uuid_pattern.match(base) else None

    if isinstance(paths, str):
        return extract_uuid(paths)

    paths = np.asarray(paths)
    extracted = np.vectorize(extract_uuid, otypes=[object])(paths)
    return extracted[extracted != np.array(None)]


def parse_radar_path(path):
    """
    Parse a RADAR-CNS directory path into structured metadata.

    The function assumes the following directory layout::

        .../<project_id>/<user_id>/<data_type>

    where ``data_type`` corresponds to a RADAR data category or modality
    and is associated with a schema file named
    ``schema-<data_type>.json``.

    Parameters
    ----------
    path : str
        Full filesystem path to a RADAR data category directory.
        Trailing slashes are allowed and will be ignored.

    Returns
    -------
    dict
        Parsed metadata with the following keys:

        - ``project_id`` : str
          RADAR project identifier.

        - ``user_id`` : str
          Participant / subject UUID.

        - ``data_type`` : str
          Data category or modality name.

        - ``schema_file`` : str
          Expected schema filename
          (``schema-<data_type>.json``).

        - ``path`` : str
          Original input path.

        - ``schema_path`` : str
          Path to the expected schema file, located alongside the
          ``data_type`` directory.

    Raises
    ------
    ValueError
        If the path does not contain at least three components.

    Notes
    -----
    This function performs no filesystem I/O and does not validate that the
    schema file exists. It relies purely on string parsing and the assumed
    RADAR directory structure.
    """
    parts = path.rstrip("/").split("/")
    if len(parts) < 3:
        message = f"Expected .../<project_id>/<user_id>/<data_type>, got: {path!r}"
        raise ValueError(message)

    data_type = parts[-1]
    user_id = parts[-2]
    project_id = parts[-3]

    schema_file = f"schema-{data_type}.json"

    return {
        "project_id": project_id,
        "user_id": user_id,
        "data_type": data_type,
        "schema_file": schema_file,
        "path": path,
        "schema_path": "/".join(parts[:-1] + [schema_file]),
    }
