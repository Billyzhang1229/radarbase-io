# radarbase-io

I/O utilities and helpers for RADAR-base datasets. This repository is currently a
scaffold; modules, examples, and tests will be filled in over time.

## Quickstart

```bash
uv sync --dev
```

## Storage configuration

This project uses `fsspec`-style URLs so the same code can read from SFTP, S3,
or local paths. Configure the root with an environment variable (use
`.env.local`, which is gitignored), and keep credentials in native tools (SSH
config for SFTP, AWS credentials for S3).

Example `.env.local`:

```
RADARBASE_STORAGE_URL=sftp://radarbase/path/to/project
RADARBASE_STORAGE_OPTIONS={"anon": false}
```

Example URLs:
- `sftp://radarbase/path/to/project`
- `s3://my-bucket/path/to/project`
- `/data/radar` (local path)

## Test data

This repo uses the RADAR-base mockdata repository as a git submodule at
`tests/data/mockdata`. Initialize it with:

```bash
git submodule update --init --recursive
```

The mockdata repository is Apache-2.0 licensed; its LICENSE is kept within the
submodule.

## Development

```bash
uv run -m pytest
uv run ruff check .
uv run ruff format --check .
```
