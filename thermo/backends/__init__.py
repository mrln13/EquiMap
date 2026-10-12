# thermo/backends/__init__.py
"""
Backend registry.

Backends are imported lazily so that a missing optional dependency
(e.g. juliacall for magemin) does not prevent importing the thermo package.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thermo.backends.base import AbstractBackend

_REGISTRY = {
    "magemin":   ("thermo.backends.magemin",   "MAGEMinDirectBackend"),
    "pycalphad": ("thermo.backends.pycalphad", "PycalphadBackend"),
    "perplex":   ("thermo.backends.perplex",   "PerplexBackend"),
}


def get_backend_class(name: str):
    """Return the backend class for *name* (lazy import)."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    module_path, class_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def list_backends() -> list:
    """Return names of all registered backends."""
    return list(_REGISTRY.keys())
