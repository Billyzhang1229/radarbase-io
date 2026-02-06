"""CSV to parquet conversion helpers."""

import uuid

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from dask import delayed
from tlz import partition_all

from .fs import fs_from_json, fs_to_json, resolve_paths
from .performance import compute_dask
from .utils import align_and_coerce_dataframe


def _split_parent_name(path):
    normalized = path.rstrip("/")
    if "/" not in normalized:
        return "", normalized
    parent, name = normalized.rsplit("/", 1)
    return parent, name


def _join_fs_path(parent, name):
    if not parent:
        return name
    return f"{parent.rstrip('/')}/{name}"


def _infer_csv_compression(path):
    if path.endswith(".gz"):
        return "gzip"
    return None


def _validate_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("schema must be a dict produced by build_schema")

    required = {"columns", "pandas_dtypes", "meta", "pyarrow_schema"}
    missing = [key for key in required if key not in schema]
    if missing:
        raise ValueError(
            f"schema is missing required keys: {', '.join(sorted(missing))}"
        )

    columns = schema["columns"]
    pandas_dtypes = schema["pandas_dtypes"]
    meta = schema["meta"]
    pyarrow_schema = schema["pyarrow_schema"]

    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise ValueError("schema['columns'] must be a list of column names")
    if not isinstance(pandas_dtypes, dict):
        raise ValueError("schema['pandas_dtypes'] must be a dict")
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("schema['meta'] must be a pandas DataFrame")
    if not isinstance(pyarrow_schema, pa.Schema):
        raise ValueError("schema['pyarrow_schema'] must be a pyarrow.Schema")

    missing_dtype_columns = [col for col in columns if col not in pandas_dtypes]
    if missing_dtype_columns:
        raise ValueError(
            "schema['pandas_dtypes'] is missing columns: "
            + ", ".join(missing_dtype_columns)
        )

    if list(pyarrow_schema.names) != columns:
        raise ValueError("schema['pyarrow_schema'] column order must match columns")

    normalized_meta = meta.reindex(columns=columns)

    return columns, pandas_dtypes, normalized_meta, pyarrow_schema


def _read_batch(paths, columns, pandas_dtypes, *, fs=None, fs_json=None):
    if fs is None:
        if fs_json is None:
            raise ValueError("Pass either fs= or fs_json=.")
        fs = fs_from_json(fs_json)
    elif fs_json is not None:
        raise ValueError("Pass either fs= or fs_json=, not both.")

    frames = []
    for path in paths:
        compression = _infer_csv_compression(path)
        with fs.open(path, "rb") as handle:
            frame = pd.read_csv(handle, compression=compression)
        frames.append(align_and_coerce_dataframe(frame, columns, pandas_dtypes))

    if not frames:
        return pd.DataFrame(
            {col: pd.Series(dtype=pandas_dtypes[col]) for col in columns}
        )

    return pd.concat(frames, ignore_index=True)


def _build_temp_target(output_fs_path):
    parent, name = _split_parent_name(output_fs_path)
    if not name:
        name = "parquet"
    tmp_name = f".{name}.tmp-{uuid.uuid4().hex}"
    return _join_fs_path(parent, tmp_name)


def _remove_if_exists(fs, path):
    if fs.exists(path):
        fs.rm(path, recursive=True)


def _prepare_target(fs, output_fs_path, overwrite):
    if fs.exists(output_fs_path):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_fs_path}. "
                "Pass overwrite=True to replace it."
            )
        fs.rm(output_fs_path, recursive=True)

    parent, _ = _split_parent_name(output_fs_path)
    if parent and not fs.exists(parent):
        fs.makedirs(parent, exist_ok=True)


def csv_to_parquet(
    csv_paths,
    output_path,
    schema,
    fs=None,
    storage_options=None,
    *,
    output_mode="file",
    files_per_task=150,
    overwrite=False,
    client=None,
):
    """
    Convert CSV/CSV.GZ files into parquet using a build_schema artifact.

    Parameters
    ----------
    csv_paths : str or sequence of str
        Input CSV paths.
    output_path : str
        Output parquet file path (file mode) or dataset directory (dataset mode).
    schema : dict
        Output from radarbase_io.schema.build_schema.
    fs : fsspec.AbstractFileSystem, optional
        Filesystem instance to use.
    storage_options : dict, optional
        Storage options passed to fsspec when resolving paths.
    output_mode : {"file", "dataset"}, default "file"
        Output parquet layout mode.
    files_per_task : int, default 150
        Number of CSV files to process in one delayed batch task.
    overwrite : bool, default False
        Whether to overwrite existing output.
    client : dask.distributed.Client, optional
        Dask client for distributed execution.
    """

    if output_mode not in {"file", "dataset"}:
        raise ValueError("output_mode must be one of {'file', 'dataset'}")

    columns, pandas_dtypes, meta, pyarrow_schema = _validate_schema(schema)

    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    else:
        csv_paths = list(csv_paths)

    if not csv_paths:
        raise ValueError("csv_paths is empty")
    if files_per_task <= 0:
        raise ValueError("files_per_task must be greater than 0")

    fs, fs_paths = resolve_paths(
        [*csv_paths, output_path], fs=fs, storage_options=storage_options
    )
    csv_fs_paths = fs_paths[:-1]
    output_fs_path = fs_paths[-1]

    if not csv_fs_paths:
        raise ValueError("No CSV paths were resolved")

    _prepare_target(fs, output_fs_path, overwrite=overwrite)

    batch_kwargs = {"fs": fs} if client is None else {"fs_json": fs_to_json(fs)}
    tasks = [
        delayed(_read_batch)(list(chunk), columns, pandas_dtypes, **batch_kwargs)
        for chunk in partition_all(files_per_task, csv_fs_paths)
    ]
    ddf = dd.from_delayed(tasks, meta=meta)

    row_count_task = ddf.map_partitions(len).sum()

    temp_target = None
    output_target = output_fs_path
    ddf_to_write = ddf
    if output_mode == "file":
        temp_target = _build_temp_target(output_fs_path)
        _remove_if_exists(fs, temp_target)
        output_target = temp_target
        ddf_to_write = ddf.repartition(npartitions=1)

    write_task = ddf_to_write.to_parquet(
        output_target,
        engine="pyarrow",
        schema=pyarrow_schema,
        write_index=False,
        compute=False,
    )

    try:
        results = compute_dask(
            [write_task, row_count_task],
            client=client,
            scheduler="threads",
        )
        rows_written = int(results[1])

        if output_mode == "file":
            part_files = [
                path for path in fs.find(output_target) if path.endswith(".parquet")
            ]
            if len(part_files) != 1:
                raise ValueError(
                    "Expected a single parquet part file in file mode, "
                    f"found {len(part_files)}"
                )
            fs.mv(part_files[0], output_fs_path)
    finally:
        if temp_target is not None:
            _remove_if_exists(fs, temp_target)

    return {
        "output_mode": output_mode,
        "output_path": output_fs_path,
        "files_processed": len(csv_fs_paths),
        "rows_written": rows_written,
        "columns": columns,
    }
