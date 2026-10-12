# thermo/cache.py
"""On-disk cache for P-T grid results.

Why this exists
---------------
A thermodynamic P-T grid is a pure function of (backend + its config,
composition, P-T range, grid resolution).  Computing one is expensive —
MAGEMin boots Julia (~20 s) and minimises hundreds of points; the
composition-vs-scale *sweep* runs a dozen grids back to back.  Re-running an
identical grid (re-opening the viewer, repeating a sweep, comparing the same
ROI again) repeats all that work for a bit-identical answer.

This module memoises grid results on disk, keyed by a SHA-256 of everything
that determines the output.  A cache *hit* returns
:func:`thermo.result.result_from_portable_dict` results instantly — and,
crucially, lets the orchestrator skip booting a worker pool / Julia entirely
when every grid in a sweep is cached (see :mod:`thermo.parallel`).

Determinism & safety
--------------------
Backends are deterministic for a fixed config, so caching is sound.  The key
includes the backend *name* and its constructor kwargs (``db``, ``tdb_path``,
``perplex_dir``, ``fe3_fraction`` …) but NOT the byte-contents of external
database files — if you swap the actual MAGEMin/Perple_X database on disk for a
different one under the same name, clear the cache (``GridCache().clear()`` or
``python -m thermo cache clear``).  ``CACHE_VERSION`` is bumped whenever the
result schema or a backend's output semantics change, which orphans (ignores)
all older entries.

This module is deliberately free of any Qt / viewer imports so it can be unit
tested and used from worker processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from thermo.result import (
    ThermoResult,
    result_from_portable_dict,
    result_to_portable_dict,
)

__all__ = [
    "CACHE_VERSION",
    "GridCache",
    "default_cache_dir",
    "cache_disabled_by_env",
]

# Bump when the stored result schema or a backend's output semantics change.
# Old entries (keyed with a different version) are then simply never matched.
CACHE_VERSION = 1

# Rounding used when canonicalising floats into the key.  Compositions and P-T
# bounds are produced deterministically, so this only guards against trailing
# float noise — 6 dp is far finer than any physically meaningful difference.
_KEY_ROUND = 6

_ENV_DIR = "THERMOD_GRID_CACHE_DIR"
_ENV_OFF = "THERMOD_GRID_CACHE"  # set to 0/off/false/no to disable globally
_OFF_VALUES = {"0", "off", "false", "no", "disable", "disabled"}


def cache_disabled_by_env() -> bool:
    """True if ``THERMOD_GRID_CACHE`` is set to an off-like value."""
    return os.environ.get(_ENV_OFF, "").strip().lower() in _OFF_VALUES


def default_cache_dir() -> Path:
    """User-level grid-cache directory (``$THERMOD_GRID_CACHE_DIR`` overrides).

    Defaults to ``~/.thermod/grid_cache``.  The cache is keyed by a fully
    self-describing hash, so a single shared location is safe across datasets.
    """
    env = os.environ.get(_ENV_DIR)
    if env:
        return Path(env)
    return Path.home() / ".thermod" / "grid_cache"


# ---------------------------------------------------------------------------
# Canonicalisation helpers (stable JSON → stable hash)
# ---------------------------------------------------------------------------

def _round(x: Any) -> Any:
    try:
        return round(float(x), _KEY_ROUND)
    except (TypeError, ValueError):
        return x


def _canon(obj: Any) -> Any:
    """Recursively coerce *obj* into a JSON-stable, hashable-by-text form."""
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return _round(obj)
    if obj is None:
        return None
    return str(obj)


def _canon_comp(composition: Any) -> Dict[str, float]:
    """Element wt% dict, rounded and sorted — the composition's cache identity.

    Accepts a :class:`thermo.composition.Composition` (duck-typed via
    ``to_element_wt``) or a plain ``{element: wt%}`` dict.
    """
    if hasattr(composition, "to_element_wt"):
        wt = composition.to_element_wt()
    elif isinstance(composition, dict):
        wt = composition
    else:
        raise TypeError(
            f"composition must be a Composition or dict, got {type(composition).__name__}"
        )
    return {str(el): _round(v) for el, v in sorted(wt.items()) if float(v) != 0.0}


# ---------------------------------------------------------------------------
# GridCache
# ---------------------------------------------------------------------------

class GridCache:
    """A disk-backed memo of P-T grid results.

    Parameters
    ----------
    cache_dir :
        Where to store entries.  Defaults to :func:`default_cache_dir`.
    enabled :
        Master switch.  Forced ``False`` if the ``THERMOD_GRID_CACHE`` env var
        is set to an off-like value, so a single env flag disables caching for
        the whole process tree.
    """

    def __init__(self, cache_dir: Optional[os.PathLike] = None, enabled: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.enabled = bool(enabled) and not cache_disabled_by_env()

    # ---- Key ----------------------------------------------------------------

    def key_for(
        self,
        backend_name: str,
        backend_kwargs: Optional[Dict[str, Any]],
        composition: Any,
        p_range,
        t_range,
        n_p: int,
        n_t: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """SHA-256 hex over everything that determines the grid output."""
        payload = {
            "v": CACHE_VERSION,
            "backend": str(backend_name or "").lower(),
            "backend_kwargs": _canon(backend_kwargs or {}),
            "comp": _canon_comp(composition),
            "p_range": [_round(p_range[0]), _round(p_range[1])],
            "t_range": [_round(t_range[0]), _round(t_range[1])],
            "n_p": int(n_p),
            "n_t": int(n_t),
            "extra": _canon(extra) if extra else None,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ---- Read ---------------------------------------------------------------

    def get(self, key: str) -> Optional[List[Dict]]:
        """Return the cached portable-result dicts for *key*, or None on miss.

        Tolerates a missing / unreadable / half-written file by treating it as
        a miss (the grid will simply be recomputed and re-stored).
        """
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict) or doc.get("version") != CACHE_VERSION:
            return None
        results = doc.get("results")
        if not isinstance(results, list):
            return None
        return results

    def get_results(self, key: str) -> Optional[List[ThermoResult]]:
        """Like :meth:`get` but returns reconstructed :class:`ThermoResult`."""
        dicts = self.get(key)
        if dicts is None:
            return None
        return [result_from_portable_dict(d) for d in dicts]

    # ---- Write --------------------------------------------------------------

    def put(self, key: str, results_portable: List[Dict],
            meta: Optional[Dict[str, Any]] = None) -> bool:
        """Store *results_portable* (a list of portable dicts) under *key*.

        Written atomically (temp file + ``os.replace``).  Any failure is
        swallowed and reported as ``False`` — caching must never break a run.
        """
        if not self.enabled:
            return False
        doc = {
            "version": CACHE_VERSION,
            "key": key,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_results": len(results_portable),
            "meta": _canon(meta) if meta else {},
            "results": results_portable,
        }
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.cache_dir), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh)
                os.replace(tmp, str(self._path(key)))
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            return True
        except OSError:
            return False

    def put_results(self, key: str, results: List[ThermoResult],
                    meta: Optional[Dict[str, Any]] = None) -> bool:
        """Like :meth:`put` but takes :class:`ThermoResult` objects."""
        return self.put(key, [result_to_portable_dict(r) for r in results], meta=meta)

    # ---- Maintenance --------------------------------------------------------

    def clear(self) -> int:
        """Delete every cache entry.  Returns the number of files removed."""
        n = 0
        try:
            entries = list(self.cache_dir.glob("*.json"))
        except OSError:
            return 0
        for p in entries:
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n

    def stats(self) -> Dict[str, Any]:
        """Summary of the cache: entry count, total size, directory, version."""
        n = 0
        total = 0
        try:
            for p in self.cache_dir.glob("*.json"):
                try:
                    total += p.stat().st_size
                    n += 1
                except OSError:
                    pass
        except OSError:
            pass
        return {
            "dir": str(self.cache_dir),
            "exists": self.cache_dir.exists(),
            "enabled": self.enabled,
            "version": CACHE_VERSION,
            "n_entries": n,
            "total_bytes": total,
        }
