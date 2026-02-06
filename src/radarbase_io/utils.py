import os
import re

import numpy as np
import pandas as pd


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
    normalized = path.replace("\\", "/").rstrip("/")
    parts = normalized.split("/")
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
        "schema_path": "/".join(parts + [schema_file]),
    }


def coerce_boolean_series(series):
    """
    Coerce a Series into pandas nullable boolean dtype.

    String/int truthy and falsy values are normalized; unknown values become
    missing (``pd.NA``).
    """
    true_values = {"true", "1", "t", "y", "yes"}
    false_values = {"false", "0", "f", "n", "no"}

    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return pd.NA
        value_str = str(value).strip().lower()
        if value_str in true_values:
            return True
        if value_str in false_values:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def coerce_series_dtype(series, dtype):
    """Coerce a Series into a schema-driven dtype."""
    if dtype in {"float32", "float64"}:
        return pd.to_numeric(series, errors="coerce").astype(dtype)
    if dtype == "Int64":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if dtype == "boolean":
        return coerce_boolean_series(series)
    if dtype == "string":
        return series.astype("string")
    if dtype == "object":
        return series.astype("object")
    return series.astype(dtype)


def align_and_coerce_dataframe(df, columns, pandas_dtypes):
    """
    Align DataFrame columns to schema and coerce values to schema dtypes.

    Missing columns are added with null values, extra columns are dropped, and
    output order follows ``columns``.
    """
    df = df.reindex(columns=columns)
    for column in columns:
        df[column] = coerce_series_dtype(df[column], pandas_dtypes[column])
    return df
