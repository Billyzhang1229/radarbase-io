# radarbase-io

I/O utilities and helpers for RADAR-base datasets. This repository is currently a
scaffold; modules, examples, and tests will be filled in over time.

## Quickstart

```bash
uv sync --dev
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
