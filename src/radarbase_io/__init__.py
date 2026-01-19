"""Top-level package for radarbase_io."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["build_index", "list_participants"]

if TYPE_CHECKING:
    from .index import build_index
    from .layout import list_participants

_LAZY_ATTRS = {
    "build_index": (".index", "build_index"),
    "list_participants": (".layout", "list_participants"),
}


def __getattr__(name: str):
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
