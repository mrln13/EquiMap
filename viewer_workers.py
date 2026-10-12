# viewer_workers.py
"""
QThread workers used by viewer.py:
- `_EDSWorker`        — per-tile full-res ROI EDS extraction (`eds_map.roi_to_banded_maps`).
- `_EDSCacheWorker`   — cache fast-path ROI EDS extraction (`eds_cache.roi_from_cache`).
- `_ThermoGridWorker` — runs `ThermoCalc.run_grid` off the Qt main thread (perplex/pycalphad).
- `_ThermoSweepWorker` — runs one *or many* MAGEMin grids across a process pool.
- `_CompositionScaleWorker` — quadtree per-window composition computation.

Moved out of viewer.py as part of the tier-B refactor.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
from qtpy import QtCore

from eds_map import roi_to_banded_maps
from viewer_thermo_helpers import _net_count_fractions


class _EDSWorker(QtCore.QThread):
    """Runs roi_to_banded_maps() off the Qt main thread so the UI stays responsive."""
    progress = QtCore.Signal(int, int)        # (tiles_done, tiles_total)
    finished = QtCore.Signal(object, object)  # (band_maps, diag_or_None)
    error    = QtCore.Signal(str)

    def __init__(self, manifest, roi, elements, kwargs, parent=None):
        super().__init__(parent)
        self._manifest  = manifest
        self._roi       = roi
        self._elements  = elements
        self._kwargs    = kwargs
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            result = roi_to_banded_maps(
                self._manifest, self._roi, self._elements,
                progress=True,
                _cancel_flag=lambda: self._cancelled,
                _progress_cb=lambda done, total: self.progress.emit(done, total),
                **self._kwargs,
            )
            band_maps, diag = result if isinstance(result, tuple) else (result, None)
            self.finished.emit(band_maps, diag)
        except Exception as exc:
            self.error.emit(str(exc))


class _EDSCacheWorker(QtCore.QThread):
    """Runs eds_cache.roi_from_cache() off the Qt main thread.

    Emits per-band progress so the user can see that slow mass storage
    (e.g. an HDD or sparse-file fragmented reads) is actually doing work
    rather than freezing.
    """
    progress = QtCore.Signal(int, int)  # (bands_done, bands_total)
    finished = QtCore.Signal(object, object)  # (band_maps, diag)
    error    = QtCore.Signal(str)

    def __init__(self, cache_dir, roi, elements, blend_mode, roi_mask,
                 out_ds=1, parent=None):
        super().__init__(parent)
        self._cache_dir = cache_dir
        self._roi = roi
        self._elements = elements
        self._blend_mode = blend_mode
        self._roi_mask = roi_mask
        self._out_ds = max(1, int(out_ds))
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import eds_cache as _edsc
            band_maps, w_slice = _edsc.roi_from_cache(
                self._cache_dir, self._roi, self._elements,
                blend_mode=self._blend_mode,
                roi_mask=self._roi_mask,
                return_weight=True,
                out_ds=self._out_ds,
                progress_cb=lambda done, total: self.progress.emit(done, total),
                cancel_flag=lambda: self._cancelled,
            )
            diag = {"weight": w_slice, "source": "cache"}
            self.finished.emit(band_maps, diag)
        except Exception as exc:
            self.error.emit(str(exc))


class _ThermoGridWorker(QtCore.QThread):
    """Runs calc.run_grid() off the main thread so the UI stays responsive."""
    finished = QtCore.Signal(list)
    error    = QtCore.Signal(str)

    def __init__(self, calc, comp, opts, parent=None):
        super().__init__(parent)
        self._calc = calc
        self._comp = comp
        self._opts = opts

    def cancel(self):
        """Cancel the running calculation - kills the underlying subprocess."""
        if hasattr(self._calc, "cancel"):
            self._calc.cancel()

    def run(self):
        try:
            results = self._calc.run_grid(
                self._comp,
                p_range=self._opts["p_range"],
                t_range=self._opts["t_range"],
                n_p=self._opts["n_p"],
                n_t=self._opts["n_t"],
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class _ThermoSweepWorker(QtCore.QThread):
    """Run a batch of P-T grids in worker processes off the Qt main thread.

    Used for the MAGEMin backend, whose juliacall ``@threads :static`` calls
    forbid running in a QThread but are fine in a separate process. Each
    worker process boots Julia + MAGEMin once and reuses it for every grid it
    receives (see ``thermo/parallel.py``). A single grid is just a one-element
    ``jobs`` list with ``max_workers=1`` — the same machinery keeps the UI
    responsive and Cancel working.

    Signals
    -------
    progress(done, total)        — emitted after each grid completes/fails
    grid_done(job_index, dicts)  — one grid finished; ``dicts`` are portable
                                    ThermoResult dicts (thermo.result)
    grid_failed(job_index, msg)  — one grid raised (others keep running)
    pool_failed(msg)             — the whole pool died; caller should fall back
    finished_all()               — every job processed (or cancelled)
    """
    progress     = QtCore.Signal(int, int)
    grid_done    = QtCore.Signal(int, object)
    grid_failed  = QtCore.Signal(int, str)
    pool_failed  = QtCore.Signal(str)
    finished_all = QtCore.Signal()

    def __init__(self, jobs, backend_name, backend_kwargs,
                 max_workers=None, threads_per_worker=None, parent=None):
        super().__init__(parent)
        self._jobs = list(jobs)
        self._backend_name = backend_name
        self._backend_kwargs = dict(backend_kwargs or {})
        self._max_workers = max_workers
        self._threads_per_worker = threads_per_worker
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from thermo.parallel import iter_grid_results, SweepPoolBroken
        total = len(self._jobs)
        done = 0
        try:
            for idx, dicts, err in iter_grid_results(
                self._jobs,
                self._backend_name,
                self._backend_kwargs,
                max_workers=self._max_workers,
                threads_per_worker=self._threads_per_worker,
                cancel_flag=lambda: self._cancelled,
            ):
                if err is not None:
                    self.grid_failed.emit(idx, err)
                else:
                    self.grid_done.emit(idx, dicts)
                done += 1
                self.progress.emit(done, total)
            self.finished_all.emit()
        except SweepPoolBroken as exc:
            self.pool_failed.emit(str(exc))
        except Exception as exc:
            self.pool_failed.emit(str(exc))


class _CompositionScaleWorker(QtCore.QThread):
    """Runs per-window composition computation off the Qt main thread.

    Emits ``progress(done, total)`` after each window and
    ``finished_with(records, summary)`` when done.

    ``records`` is a flat list - one dict per window with keys ``level``,
    ``bbox`` (mosaic-coords), ``n_pixels``, ``mask_frac``, ``wt_pct``
    (element wt% summing to 100).

    ``summary`` is keyed by quadtree level and gives ``mean``, ``std``,
    ``n_windows`` (number of usable windows), and ``elements`` (sorted).

    EDS source has two modes: an in-memory ``band_maps`` (sliced per window;
    coordinates divided by ``band_maps_ds`` if those maps were downsampled), or
    — when ``cache_dir`` is given — **full-res cache mode**, reading each
    window's native-resolution band maps straight from the mosaic cache
    (``eds_cache.roi_from_cache``) so a large ROI can be analysed at full
    resolution with bounded, per-window memory.
    """
    progress = QtCore.Signal(int, int)
    finished_with = QtCore.Signal(list, dict)
    error = QtCore.Signal(str)

    def __init__(self,
                 *,
                 source: str,
                 levels: Dict[int, List[Tuple[int, int, int, int]]],
                 roi_bbox: Tuple[int, int, int, int],
                 roi_mask: Optional[np.ndarray],
                 band_maps: Optional[Dict[str, Dict[str, np.ndarray]]],
                 spx_callback,
                 min_mask: float = 0.5,
                 band_maps_ds: int = 1,
                 cache_dir=None,
                 cache_elements: Optional[List[str]] = None,
                 cache_blend_mode: str = "mean",
                 k_factors: Optional[Dict[str, float]] = None,
                 parent=None):
        super().__init__(parent)
        self._source = source
        self._levels = levels
        self._roi_bbox = roi_bbox
        self._roi_mask = roi_mask
        self._band_maps = band_maps
        self._spx_callback = spx_callback  # callable(bbox) -> Optional[Dict[str,float]]
        self._min_mask = float(min_mask)
        # EDS band maps may have been computed at reduced resolution to fit a
        # large ROI in RAM. ROI-local coordinates (and the mask) are then in
        # full-res mosaic pixels but must be divided by this factor to index
        # the decimated arrays. SPX path is unaffected (no band_maps indexing).
        self._band_maps_ds = max(1, int(band_maps_ds))
        # Cache mode (full-res, large ROI): when ``cache_dir`` is set the EDS
        # path reads each window's full-resolution band maps directly from the
        # mosaic cache memmaps (window-by-window, bounded RAM) instead of
        # slicing an in-memory ``band_maps``. Coordinates stay full-res (ds=1).
        self._cache_dir = cache_dir
        self._cache_elements = cache_elements
        self._cache_blend_mode = cache_blend_mode
        # SPX-anchored calibration: when set, per-window EDS net fractions are
        # converted to calibrated wt% (eds_quantify) instead of raw fraction×100.
        self._k_factors = k_factors
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _fractions_to_wt(self, frac: Dict[str, float]) -> Dict[str, float]:
        """Per-window net fractions -> wt%. Calibrated when k-factors are set,
        else the raw fraction×100 approximation."""
        if self._k_factors:
            import eds_quantify as _eq
            return _eq.apply_calibration_fractions(frac, self._k_factors)
        return {el: 100.0 * float(v) for el, v in frac.items()}

    def run(self):
        try:
            records: List[dict] = []
            ry0, ry1, rx0, rx1 = self._roi_bbox
            cache_mode = self._source == "eds" and self._cache_dir is not None
            ds = 1 if cache_mode else self._band_maps_ds
            _reader = None
            if cache_mode:
                import eds_cache as _edsc
                # Open the cache memmaps ONCE; slicing per window is then cheap
                # (no per-window file opens / thread pools).
                _reader = _edsc.CacheWindowReader(
                    self._cache_dir, self._cache_elements, self._cache_blend_mode)
            # Working grid: full-res for SPX and cache mode; the (possibly
            # decimated) band-map shape for the in-RAM EDS path.
            if self._source == "eds" and not cache_mode and self._band_maps:
                _sample = next(iter(self._band_maps.values()))["main"]
                roi_h, roi_w = _sample.shape[:2]
            else:
                roi_h, roi_w = ry1 - ry0, rx1 - rx0

            total = sum(len(v) for v in self._levels.values())
            done = 0

            for level in sorted(self._levels):
                for bbox in self._levels[level]:
                    if self._cancelled:
                        self.finished_with.emit(records,
                                                self._summarise(records))
                        return
                    by0, by1, bx0, bx1 = bbox
                    # ROI-local sub-bbox (relative to band_maps origin),
                    # divided by ds for the (possibly decimated) EDS grid.
                    sub_local = ((by0 - ry0) // ds, (by1 - ry0) // ds,
                                 (bx0 - rx0) // ds, (bx1 - rx0) // ds)
                    # ROI mask slice for this sub-window
                    if self._roi_mask is not None:
                        sm = self._roi_mask[
                            max(0, sub_local[0]):min(roi_h, sub_local[1]),
                            max(0, sub_local[2]):min(roi_w, sub_local[3]),
                        ]
                        sm_bool = sm.astype(bool)
                        n_total = sm.size
                        n_in = int(sm_bool.sum())
                        mask_frac = (n_in / n_total) if n_total else 0.0
                    else:
                        sm_bool = None
                        mask_frac = 1.0
                        n_in = (sub_local[1] - sub_local[0]) * (sub_local[3] - sub_local[2])

                    record: dict = {
                        "level": level,
                        "bbox": tuple(bbox),
                        "n_pixels": int(n_in),
                        "mask_frac": float(mask_frac),
                        "wt_pct": {},
                    }

                    if mask_frac < self._min_mask:
                        record["skipped"] = "mask_frac<min"
                    elif self._source == "eds":
                        if cache_mode:
                            # Slice this window's full-res band maps from the
                            # already-open cache memmaps, mask + integrate.
                            try:
                                frac = _reader.net_fractions(tuple(bbox), sm_bool)
                            except Exception as exc:
                                record["skipped"] = f"cache_read_error: {exc}"
                                frac = None
                            if frac:
                                record["wt_pct"] = self._fractions_to_wt(frac)
                            elif "skipped" not in record:
                                record["skipped"] = "no signal"
                        elif not self._band_maps:
                            record["skipped"] = "no band_maps"
                        else:
                            frac, _np = _net_count_fractions(
                                self._band_maps,
                                sub_bbox=sub_local,
                                sub_mask=sm_bool,
                            )
                            if frac:
                                record["wt_pct"] = self._fractions_to_wt(frac)
                            else:
                                record["skipped"] = "no signal"
                    else:  # spx
                        comp = None
                        try:
                            comp = self._spx_callback(tuple(bbox))
                        except Exception as exc:
                            record["skipped"] = f"spx_error: {exc}"
                        if comp:
                            record["wt_pct"] = {el: float(v) for el, v in comp.items()}
                        elif "skipped" not in record:
                            record["skipped"] = "no SPX tile"

                    records.append(record)
                    done += 1
                    self.progress.emit(done, total)

            self.finished_with.emit(records, self._summarise(records))
        except Exception as exc:
            self.error.emit(str(exc))

    @staticmethod
    def _summarise(records: List[dict]) -> dict:
        by_level: Dict[int, List[dict]] = {}
        for r in records:
            by_level.setdefault(r["level"], []).append(r)
        summary: Dict[int, dict] = {}
        all_els: List[str] = []
        seen = set()
        for r in records:
            for el in r.get("wt_pct", {}):
                if el not in seen:
                    seen.add(el); all_els.append(el)
        all_els.sort()

        for level, rs in by_level.items():
            usable = [r for r in rs if r.get("wt_pct")]
            n = len(usable)
            mean: Dict[str, float] = {}
            std: Dict[str, float] = {}
            for el in all_els:
                vals = np.array([r["wt_pct"].get(el, 0.0) for r in usable],
                                dtype=float)
                if vals.size:
                    mean[el] = float(vals.mean())
                    std[el] = float(vals.std(ddof=0))
                else:
                    mean[el] = 0.0
                    std[el] = 0.0
            summary[level] = {
                "mean": mean,
                "std": std,
                "n_windows": n,
                "n_total": len(rs),
                "elements": all_els,
            }
        return summary
