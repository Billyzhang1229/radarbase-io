# API reference

Minimal, hand-written API notes will live here.

Planned modules:
- radarbase_io.exceptions
- radarbase_io.fs
- radarbase_io.layout
- radarbase_io.index
- radarbase_io.read_table
- radarbase_io.read_xarray
- radarbase_io.convert

## `radarbase_io.convert`

- `csv_to_parquet(csv_paths, output_path, schema, fs=None, storage_options=None, *, output_mode="file", files_per_task=150, overwrite=False, client=None)`
  - Converts one or many CSV/CSV.GZ files into parquet using schema artifacts returned by `radarbase_io.schema.build_schema(...)`.
  - `output_mode="file"` writes a single parquet file.
  - `output_mode="dataset"` writes a parquet dataset directory.
