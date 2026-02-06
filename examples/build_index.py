"""Build an index DataFrame from local paths or remote SFTP paths."""

import argparse
import json
import os
from pathlib import Path

import fsspec
from dotenv import load_dotenv

from radarbase_io import build_index

REQUIRED_REMOTE_ENV_VARS = ("HOST_NAME", "USER_NAME", "KEY_PATH")
OPTIONAL_PASSPHRASE_ENV = "SSH_KEY_PASSWORD"
SENSITIVE_KEYS = ("passphrase", "password", "token", "secret", "key", "user", "host")


def _parse_storage_options(raw_storage_options):
    if not raw_storage_options:
        return None

    try:
        parsed = json.loads(raw_storage_options)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid JSON for storage options: {raw_storage_options!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise SystemExit("Storage options must decode to a JSON object.")

    return parsed


def _load_remote_sftp_options(extra_options):
    missing = [name for name in REQUIRED_REMOTE_ENV_VARS if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: " + ", ".join(sorted(missing))
        )

    storage_options = dict(extra_options)
    storage_options.update(
        {
            "host": os.getenv("HOST_NAME"),
            "username": os.getenv("USER_NAME"),
            "client_keys": [os.path.expanduser(os.getenv("KEY_PATH", ""))],
            "keepalive_interval": 30,
            "keepalive_count_max": 3,
        }
    )
    passphrase = os.getenv(OPTIONAL_PASSPHRASE_ENV)
    if "passphrase" not in storage_options and passphrase:
        storage_options["passphrase"] = passphrase
    return storage_options


def _redact_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return "***"
    return value


def _redact_storage_options(storage_options):
    redacted = {}
    for key, value in storage_options.items():
        lower_key = key.lower()
        if any(token in lower_key for token in SENSITIVE_KEYS):
            if isinstance(value, list):
                redacted[key] = [_redact_value(item) for item in value]
            else:
                redacted[key] = _redact_value(value)
            continue
        redacted[key] = value
    return redacted


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Build a RADAR index table from one or more project roots."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Project root path(s): local paths/URLs, or remote paths with --remote.",
    )
    parser.add_argument(
        "--storage-options",
        default=None,
        help=(
            "Storage options JSON string. If omitted, RADARBASE_STORAGE_OPTIONS "
            "from the environment is used."
        ),
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use SSH/SFTP settings from .env for remote filesystem paths.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute remote call when --remote is set. Without this, run mock mode.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=20,
        help="Number of rows to print from the top of the index table.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the full index as CSV.",
    )
    return parser


def _run_remote_mode(paths, storage_options, *, live):
    remote_options = _load_remote_sftp_options(storage_options or {})

    if not live:
        print("mock mode enabled (no remote connection attempted)")
        print("paths:")
        for path in paths:
            print(f"- {path}")
        print("resolved storage options (redacted):")
        print(json.dumps(_redact_storage_options(remote_options), indent=2))
        print("use --live with --remote to execute build_index against remote storage")
        return None

    fs = fsspec.filesystem("sftp", **remote_options)
    return build_index(paths, fs=fs)


def _print_and_save(index_df, *, head_rows, output):
    print(
        f"rows={len(index_df)} "
        f"projects={index_df['project_id'].nunique()} "
        f"participants={index_df['user_id'].nunique()}"
    )
    print(index_df.head(head_rows).to_string(index=False))

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(output_path, index=False)
        print(f"saved CSV: {output_path}")


def main():
    load_dotenv(override=False)
    args = _build_parser().parse_args()

    raw_storage_options = (
        args.storage_options
        if args.storage_options is not None
        else os.getenv("RADARBASE_STORAGE_OPTIONS")
    )
    storage_options = _parse_storage_options(raw_storage_options)

    if args.remote:
        index_df = _run_remote_mode(args.paths, storage_options, live=args.live)
        if index_df is None:
            return
    else:
        index_df = build_index(args.paths, storage_options=storage_options)

    _print_and_save(index_df, head_rows=args.head, output=args.output)


if __name__ == "__main__":
    main()
