# radarbase-io

I/O utilities and helpers for RADAR-base datasets. This repository is currently a
scaffold; modules, examples, and tests will be filled in over time.

## Quickstart

```bash
uv sync --dev
```

## Storage configuration

This project uses `fsspec`-style URLs so the same code can read from SFTP, S3,
or local paths. Keep credentials in native tools where possible (for example,
SSH config + ssh-agent for SFTP).

Example URLs:
- `sftp://radarbase/path/to/project`
- `s3://my-bucket/path/to/project`
- `/data/radar` (local path)

## Build index example (local + remote)

Local usage with explicit path:

```bash
./.venv/bin/python examples/build_index.py /data/radar/project
```

Local usage with mockdata fixture:

```bash
./.venv/bin/python examples/build_index.py tests/data/mockdata/mockdata
```

Remote usage:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set:
   - `HOST_NAME`
   - `USER_NAME`
   - `KEY_PATH`
3. Recommended: load your encrypted SSH key into `ssh-agent`:
   ```bash
   ssh-add "$KEY_PATH"
   ```
4. Optional fallback (not recommended): set `SSH_KEY_PASSWORD` in `.env`.
5. Optional: set `RADARBASE_STORAGE_OPTIONS` as JSON for extra sshfs options
   (for example `{"connect_timeout": 20, "login_timeout": 20}`).
6. Run the example in mock mode first:
   ```bash
   ./.venv/bin/python examples/build_index.py /your/remote/project/path --remote
   ```
7. Run the live remote call:
   ```bash
   ./.venv/bin/python examples/build_index.py /your/remote/project/path --remote --live
   ```

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
