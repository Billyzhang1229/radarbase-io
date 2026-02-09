"""Top-level package for radarbase_io."""

from importlib import import_module

__all__ = ["build_index", "build_schema", "list_participants", "csv_to_parquet"]

_LAZY_ATTRS = {
    "build_index": (".index", "build_index"),
    "build_schema": (".schema", "build_schema"),
    "list_participants": (".layout", "list_participants"),
    "csv_to_parquet": (".convert", "csv_to_parquet"),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message)
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
