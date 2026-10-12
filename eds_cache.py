"""EDS mosaic cache.

Stores per-element, per-band (main/sb_lo/sb_hi) accumulated sums plus a shared
weight mosaic at the full-resolution BSE grid (or a downsampled version). ROI
reads become memmap slicing instead of per-tile BCF re-parsing.

Layout under ``<base_dir>/eds_cache/``::

    manifest.json            metadata + fingerprint (see CACHE_MANIFEST_VERSION)
    weight.npy               float32, shape (H_ds, W_ds) — shared across elements
    {El}__{band}.npy         float32, shape (H_ds, W_ds) — raw weighted sums

Raw sums are stored so callers can pick blend_mode at read time
(mean = sum / weight; sum = raw array). Width_eV is a cache-wide property
(one value per cache), to keep the per-file names stable.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CACHE_DIRNAME = "eds_cache"
CACHE_MANIFEST = "manifest.json"
CACHE_MANIFEST_VERSION = 1
CACHE_LINK_FILENAME = "eds_cache_link.txt"  # sidecar pointer for relocated caches

BANDS = ("main", "sb_lo", "sb_hi")


# ---------- Paths ----------

def cache_dir_for(base_dir: Path) -> Path:
    """Default cache location. Use ``resolve_cache_dir`` to also follow link files."""
    return Path(base_dir) / CACHE_DIRNAME


def _link_file_path(base_dir: Path) -> Path:
    return Path(base_dir) / CACHE_LINK_FILENAME


def resolve_cache_dir(base_dir: Path) -> Path:
    """Return the actual cache directory for a stitch base_dir.

    Resolution order:
      1. If ``<base_dir>/eds_cache_link.txt`` exists and points at a directory
         that contains a manifest, return that path (cache was built with
         ``--eds-cache-dir`` to put it on a different drive).
      2. Otherwise return the default ``<base_dir>/eds_cache/``.
    """
    base_dir = Path(base_dir)
    link = _link_file_path(base_dir)
    if link.is_file():
        try:
            target = Path(link.read_text(encoding="utf-8").strip())
            if target.is_dir():
                return target
        except Exception:
            pass
    return cache_dir_for(base_dir)


def write_cache_link(base_dir: Path, target: Path) -> None:
    """Write the sidecar link file so future runs find the relocated cache."""
    base_dir = Path(base_dir)
    target = Path(target).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    _link_file_path(base_dir).write_text(str(target), encoding="utf-8")


def clear_cache_link(base_dir: Path) -> None:
    """Remove a stale link file (e.g. after moving the cache back to the default)."""
    link = _link_file_path(base_dir)
    if link.exists():
        try:
            link.unlink()
        except OSError:
            pass


def _element_path(cache_dir: Path, element: str, band: str) -> Path:
    return cache_dir / f"{element}__{band}.npy"


def _weight_path(cache_dir: Path) -> Path:
    return cache_dir / "weight.npy"


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_MANIFEST


# ---------- Fingerprint ----------

def stitch_fingerprint(manifest_csv: Path) -> str:
    """SHA-256 of the stitch manifest CSV content.

    Any change to tile positions, shears, or file mappings re-hashes, which
    invalidates the EDS cache against a new stitch result.
    """
    with open(manifest_csv, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------- Manifest ----------

def load_manifest(cache_dir: Path) -> Optional[dict]:
    mp = _manifest_path(cache_dir)
    try:
        if not mp.exists():
            return None
        with open(mp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        # Unreadable manifest — e.g. an interrupted write or NTFS corruption
        # (WinError 1392). Even Path.exists() raises here. Treat as absent so
        # the caller rebuilds the cache instead of crashing.
        return None
    except Exception:
        return None
    if data.get("version") != CACHE_MANIFEST_VERSION:
        return None
    return data


def save_manifest(cache_dir: Path, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["version"] = CACHE_MANIFEST_VERSION
    with open(_manifest_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------- State queries ----------

def is_valid_for(cache_dir: Path, stitch_manifest_csv: Path,
                 required_elements: Optional[List[str]] = None,
                 required_width_eV: Optional[float] = None) -> bool:
    """Return True if the cache is present and matches the given stitch/elements.

    When ``required_elements`` / ``required_width_eV`` is None, those aren't
    checked — useful for "cache exists at all" queries.

    The ``width_eV`` check is skipped for caches whose ``data_type`` is not
    ``"bcf_net_counts"`` (e.g. Oxford montaged-Wt% caches don't carry an
    energy window — the value in their manifest is a placeholder).
    """
    try:
        if not cache_dir.exists():
            return False
    except OSError:
        # Corrupt/unreadable cache dir (WinError 1392) — treat as invalid.
        return False
    data = load_manifest(cache_dir)
    if not data:
        return False
    try:
        if data.get("stitch_fingerprint") != stitch_fingerprint(stitch_manifest_csv):
            return False
    except Exception:
        return False
    if required_width_eV is not None:
        # Treat a missing/null data_type as Bruker net-counts (older caches stored
        # it as JSON null); only those carry a real energy window to check.
        data_type = data.get("data_type") or "bcf_net_counts"
        if data_type == "bcf_net_counts":
            cached_w = float(data.get("width_eV", 0.0) or 0.0)
            if abs(cached_w - float(required_width_eV)) > 1e-6:
                return False
    if required_elements is not None:
        cached_els = set(data.get("elements", []))
        for e in required_elements:
            if e not in cached_els:
                return False
    return True


def status_for(cache_dir: Path, stitch_manifest_csv: Path) -> dict:
    """Describe the cache state. Keys: exists, valid, stale, elements, width_eV, ds, H, W, fingerprint, data_type."""
    out = {"exists": cache_dir.exists(), "valid": False, "stale": False,
           "elements": [], "width_eV": None, "ds": None, "H": None, "W": None,
           "fingerprint": None, "data_type": None, "source": None}
    data = load_manifest(cache_dir)
    if not data:
        return out
    out["elements"] = list(data.get("elements", []))
    out["width_eV"] = data.get("width_eV")
    out["ds"]      = data.get("ds")
    out["H"]       = data.get("H")
    out["W"]       = data.get("W")
    out["fingerprint"] = data.get("stitch_fingerprint")
    out["data_type"]   = data.get("data_type") or "bcf_net_counts"
    out["source"]      = data.get("source")
    try:
        current = stitch_fingerprint(stitch_manifest_csv)
        out["valid"] = (data.get("stitch_fingerprint") == current)
        out["stale"] = not out["valid"]
    except Exception:
        pass
    return out


def invalidate(cache_dir: Path) -> None:
    """Remove the cache directory entirely."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------- Memmap helpers ----------

def allocate_memmaps(cache_dir: Path, elements: List[str], shape: Tuple[int, int]
                     ) -> Tuple[Dict[str, Dict[str, np.memmap]], np.memmap]:
    """Create w+ memmaps for all (element, band) + weight.

    Intentionally does NOT write zeros across the full arrays. A freshly
    created memmap file on Windows/Linux reads as zeros from any page that
    hasn't been written yet (VDL / sparse-file semantics), so an explicit
    ``mm[:] = 0`` is redundant and catastrophic at scale — it would dirty
    every page of every file (up to ~170 GB for the full-res geological
    dataset) before a single tile is processed, exhausting RAM and forcing
    Windows to page other processes out to the swap file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    H, W = shape

    weight = np.lib.format.open_memmap(_weight_path(cache_dir),
                                       mode="w+", dtype=np.float32, shape=(H, W))

    accum: Dict[str, Dict[str, np.memmap]] = {}
    for el in elements:
        accum[el] = {}
        for band in BANDS:
            mm = np.lib.format.open_memmap(_element_path(cache_dir, el, band),
                                           mode="w+", dtype=np.float32, shape=(H, W))
            accum[el][band] = mm
    return accum, weight


def open_for_read(cache_dir: Path, elements: List[str]
                  ) -> Tuple[Dict[str, Dict[str, np.memmap]], np.memmap]:
    """Open existing cache memmaps read-only. Raises if any required file is missing."""
    weight = np.lib.format.open_memmap(_weight_path(cache_dir), mode="r")
    accum: Dict[str, Dict[str, np.memmap]] = {}
    for el in elements:
        accum[el] = {}
        for band in BANDS:
            pth = _element_path(cache_dir, el, band)
            if not pth.exists():
                raise FileNotFoundError(f"Cache missing {pth.name}")
            accum[el][band] = np.lib.format.open_memmap(pth, mode="r")
    return accum, weight


# ---------- ROI reads ----------

def roi_from_cache(cache_dir: Path, roi: Tuple[int, int, int, int],
                   elements: List[str], blend_mode: str = "mean",
                   roi_mask: Optional[np.ndarray] = None,
                   return_weight: bool = False,
                   progress_cb=None, cancel_flag=None,
                   progress: bool = True, max_workers: int = 1,
                   chunks_per_slice: int = 8, out_ds: int = 1,
                   ) -> Dict[str, Dict[str, np.ndarray]] | Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    """Return banded ROI maps by slicing the mosaic cache.

    Output matches ``roi_to_banded_maps``: ``{el: {'main', 'sb_lo', 'sb_hi'}}``,
    float32 arrays of shape (y1-y0, x1-x0).

    ``out_ds`` (default 1) decimates the output grid by that integer stride
    (nearest-pixel), so a very large ROI can be read at ``1/out_ds`` resolution
    to fit in RAM. Output arrays then have shape ``(ceil(H/out_ds),
    ceil(W/out_ds))``. A ``roi_mask`` given at full ROI resolution is decimated
    to match. ``out_ds=1`` is a no-op (identical to the original behaviour).

    Each slice is read in ``chunks_per_slice`` row-chunks (full-width rows,
    column-sliced in RAM — see cost note in _slice_chunked). Progress ticks
    after each chunk.

    ``max_workers`` defaults to **1** (serial file reads). On HDDs this is
    optimal: multiple parallel readers cause head thrashing between files
    (the disk has one mechanical arm and can only service one sequential
    stream at full speed). On SSDs you can bump to 4–8 to parallelise, but
    the full-row read pattern is already dominated by sequential bandwidth,
    so the marginal speedup from higher concurrency is modest on this
    workload.

    ``roi_mask`` (optional) is applied multiplicatively after normalization.
    ``progress_cb(done, total)``: called after each chunk. Total is
    (bands + 1) × chunks_per_slice (approx; last chunk may be smaller).
    ``cancel_flag()``: True to abort.
    ``progress`` (default True): tqdm bar on stderr with rate + ETA.
    ``max_workers``: parallel I/O fan-out (default 1; raise on SSD).
    ``chunks_per_slice``: progress granularity per file (default 8).

    Assumes cache was built at ds=1. For ds>1, call ``roi_from_cache_ds``.
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import threading

    data = load_manifest(cache_dir)
    if not data:
        raise RuntimeError("No cache manifest; build the cache first.")
    ds = int(data.get("ds", 1))
    if ds != 1:
        return roi_from_cache_ds(cache_dir, roi, elements, blend_mode=blend_mode,
                                 roi_mask=roi_mask, return_weight=return_weight,
                                 progress_cb=progress_cb, cancel_flag=cancel_flag,
                                 progress=progress, max_workers=max_workers,
                                 chunks_per_slice=chunks_per_slice, out_ds=out_ds)

    y0, y1, x0, x1 = map(int, roi)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("ROI must have positive size.")

    out_ds = max(1, int(out_ds))
    H_roi = y1 - y0
    W_roi = x1 - x0
    W_cache = int(data.get("W", W_roi))

    # Output grid after optional nearest-pixel decimation. src_rows are the
    # source-row offsets we keep; the column stride is applied per read.
    src_rows = list(range(0, H_roi, out_ds))
    H_out = len(src_rows)
    W_out = len(range(0, W_roi, out_ds))

    # Fixed-row chunks so total-chunk count (and the progress bar) scales with
    # the ROI height. Previously we used chunks_per_slice as a count, which
    # made the progress bar show the same total ticks for any ROI size. Now
    # the user sees a total that reflects actual work.
    #
    # chunks_per_slice is retained only as a historical parameter name; the
    # effective chunk sizing is controlled by max_chunk_rows. Chunking is over
    # OUTPUT rows so it stays correct under decimation.
    max_chunk_rows = max(1, int(500 if chunks_per_slice <= 1 else
                                max(100, 500)))  # keep default 500
    chunk_rows = max(1, min(H_out, max_chunk_rows))
    n_chunks_per_file = max(1, (H_out + chunk_rows - 1) // chunk_rows)
    n_files = len(elements) * len(BANDS) + 1  # weight + bands

    # Adaptive read strategy: if the ROI's column range is a small fraction of
    # the cache width, strided reads (one [y, x0:x1] per row) transfer much
    # less data, which can beat full-row-then-slice on fragmented HDDs where
    # sequential throughput doesn't actually hit the 100 MB/s ideal. For wide
    # ROIs (≥25% of cache width), full-row reads hide the per-row seek cost.
    fullrow_threshold = 0.25
    use_full_rows = (W_roi / max(1, W_cache)) >= fullrow_threshold

    total_steps = n_files * n_chunks_per_file
    pbar = tqdm(total=total_steps, desc="ROI from cache",
                unit="chunk", miniters=1,
                postfix={"mode": "full-rows" if use_full_rows else "strided",
                         "chunk_rows": chunk_rows,
                         "files": n_files}) if progress else None
    done_count = [0]
    tick_lock = threading.Lock()

    def _tick():
        with tick_lock:
            done_count[0] += 1
            d = done_count[0]
        if pbar is not None:
            pbar.update(1)
        if progress_cb is not None:
            progress_cb(d, total_steps)

    def _slice_chunked(path: Path) -> np.ndarray:
        mm = np.lib.format.open_memmap(path, mode="r")
        out = np.empty((H_out, W_out), dtype=np.float32)
        for k0 in range(0, H_out, chunk_rows):
            if cancel_flag is not None and cancel_flag():
                raise RuntimeError("Cancelled")
            k1 = min(H_out, k0 + chunk_rows)
            s0 = src_rows[k0]            # source-row start (inclusive)
            s1 = src_rows[k1 - 1] + 1    # source-row end (exclusive)
            if use_full_rows:
                full_rows = np.asarray(
                    mm[y0 + s0:y0 + s1:out_ds, :], dtype=np.float32)
                out[k0:k1] = full_rows[:, x0:x1:out_ds]
            else:
                out[k0:k1] = np.asarray(
                    mm[y0 + s0:y0 + s1:out_ds, x0:x1:out_ds], dtype=np.float32)
            _tick()
        return out

    try:
        for el in elements:
            for band in BANDS:
                pth = _element_path(cache_dir, el, band)
                if not pth.exists():
                    raise FileNotFoundError(f"Cache missing {pth.name}")

        workers = max(1, min(int(max_workers), n_files))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            weight_fut = pool.submit(_slice_chunked, _weight_path(cache_dir))
            band_futs: Dict[Tuple[str, str], "Future"] = {}
            for el in elements:
                for band in BANDS:
                    band_futs[(el, band)] = pool.submit(
                        _slice_chunked, _element_path(cache_dir, el, band)
                    )

            w_slice = weight_fut.result()
            if cancel_flag is not None and cancel_flag():
                raise RuntimeError("Cancelled")

            if blend_mode not in ("mean", "sum"):
                blend_mode = "mean"

            m = w_slice > 0 if blend_mode == "mean" else None
            invw = None
            if blend_mode == "mean" and m.any():
                invw = np.zeros_like(w_slice)
                invw[m] = 1.0 / w_slice[m]

            # Decimate a full-resolution ROI mask to the output grid so the
            # element-wise multiply below has matching shapes under out_ds>1.
            mask_f = None
            if roi_mask is not None:
                rm = np.asarray(roi_mask)
                if rm.shape != (H_out, W_out):
                    rm = rm[::out_ds, ::out_ds][:H_out, :W_out]
                mask_f = rm.astype(np.float32, copy=False)

            out: Dict[str, Dict[str, np.ndarray]] = {}
            for (el, band), fut in band_futs.items():
                arr = fut.result()
                if blend_mode == "mean" and invw is not None:
                    arr = arr * invw
                if mask_f is not None:
                    arr = arr * mask_f
                out.setdefault(el, {})[band] = arr
    finally:
        if pbar is not None:
            pbar.close()

    if return_weight:
        return out, w_slice
    return out


class CacheWindowReader:
    """Open the mosaic-cache memmaps once, then compute per-window net-count
    fractions cheaply.

    Why: the viewer's composition-vs-scale full-res path evaluates many
    quadtree windows. Calling :func:`roi_from_cache` once per window re-opens
    ~``len(elements)·3 + 1`` memmaps and spins up a thread pool every time,
    which is very slow for a large ROI. This holds the memmaps open and slices
    each window directly (lazy disk reads served by the OS page cache, which
    helps for the nested/overlapping quadtree windows).

    ``net_fractions(bbox, mask)`` reproduces ``roi_from_cache(blend_mode)``
    followed by ``_net_count_fractions`` exactly, so it is a drop-in fast path.
    Requires a ``ds=1`` cache (raises otherwise; callers fall back).
    """

    def __init__(self, cache_dir: Path, elements: List[str],
                 blend_mode: str = "mean", scheme: Optional[str] = None):
        data = load_manifest(cache_dir)
        if not data:
            raise RuntimeError("No cache manifest; build the cache first.")
        if int(data.get("ds", 1)) != 1:
            raise ValueError("CacheWindowReader requires a ds=1 cache.")
        from eds_background import (anchor_elements, background_plan,
                                    default_scheme)
        self.blend_mode = blend_mode if blend_mode in ("mean", "sum") else "mean"
        self.scheme = scheme or default_scheme()
        self.elements = list(elements)
        # The background plan is judged on EVERY element the cache holds (a
        # line contaminates a window whether or not it is being read), and
        # the anchor bands it needs are opened alongside the requested ones.
        cached = list(data.get("elements") or elements)
        self.plan = background_plan(
            cached, width_eV=float(data.get("width_eV") or 150.0),
            sideband_mult=float(data.get("sideband_mult") or 1.5))
        extra = [e for e in anchor_elements(self.plan, self.elements) if e in cached]
        self._weight = np.lib.format.open_memmap(_weight_path(cache_dir), mode="r")
        self._bands: Dict[str, Dict[str, np.ndarray]] = {}
        for el in list(self.elements) + extra:
            try:
                self._bands[el] = {
                    b: np.lib.format.open_memmap(_element_path(cache_dir, el, b), mode="r")
                    for b in BANDS
                }
            except (OSError, ValueError):
                if el in self.elements:
                    raise
                # an anchor element absent on disk: its users fall back to
                # their own sidebands (eds_background.net_count_map notes it)

    def _net_totals(self, bbox: Tuple[int, int, int, int],
                    mask: Optional[np.ndarray] = None,
                    step: int = 1) -> Dict[str, float]:
        """Raw (un-normalised) per-element SIGNED net-count sums over *bbox*.

        Reproduces the integration in ``roi_from_cache`` + ``_net_count_sums``:
        the net per pixel is ``main - background`` from
        ``eds_background.net_count_map`` (the reader's ``plan``/``scheme``),
        weight-normalised per pixel when ``blend_mode == "mean"``, summed
        WITHOUT a per-pixel clip (that rectified the noise of low-dose maps
        into a positive bias, 2026-09-16). Shared by :meth:`net_fractions`
        (which clips at the aggregate) and :meth:`net_sums`.

        ``step`` > 1 reads only every ``step``-th row/column and scales the
        result up to the full-region area — an unbiased estimate of the true
        sum that reads ``step²`` fewer bytes. This is the fast path for
        whole-mosaic calibration, where an exact sum is unnecessary (the k-fit
        only needs per-tile relative magnitudes) but reading every pixel of a
        billion-pixel cache is prohibitive.
        """
        from eds_background import net_count_map
        y0, y1, x0, x1 = (int(v) for v in bbox)
        step = max(1, int(step))
        ys, xs = slice(y0, y1, step), slice(x0, x1, step)
        invw = None
        if self.blend_mode == "mean":
            w = np.asarray(self._weight[ys, xs], dtype=np.float32)
            invw = np.zeros_like(w)
            np.divide(1.0, w, out=invw, where=w > 0)
        m = None
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if step > 1:
                m = m[::step, ::step]

        def get_band(el: str, band: str):
            b = self._bands.get(el)
            if b is None:
                return None
            return np.asarray(b[band][ys, xs], dtype=np.float32)

        totals: Dict[str, float] = {}
        sampled = 0
        for el in self.elements:
            net = net_count_map(el, get_band, plan=self.plan, scheme=self.scheme)
            if net is None:
                continue
            if invw is not None:
                net *= invw
            if m is not None:
                net *= m
            totals[el] = float(np.nansum(net))
            sampled = net.size
        if step > 1 and sampled > 0:
            full = max(1, y1 - y0) * max(1, x1 - x0)
            scale = full / float(sampled)
            totals = {el: v * scale for el, v in totals.items()}
        return totals

    def net_fractions(self, bbox: Tuple[int, int, int, int],
                      mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Net-count fractions (summing to 1) over *bbox* (mosaic coords).

        ``mask`` is an optional bool array of shape ``(y1-y0, x1-x0)`` applied
        after weight-normalisation. Returns ``{}`` when the window has no
        positive net signal.
        """
        totals = {el: max(0.0, v) for el, v in self._net_totals(bbox, mask).items()}
        s = sum(totals.values())
        if s <= 0:
            return {}
        return {el: v / s for el, v in totals.items()}

    def net_sums(self, bbox: Tuple[int, int, int, int],
                 mask: Optional[np.ndarray] = None,
                 step: int = 1) -> Dict[str, float]:
        """Raw per-element net-count sums over *bbox* (mosaic coords).

        The un-normalised counterpart of :meth:`net_fractions` — the input the
        SPX-anchored k-factor fit (``eds_quantify.derive_k_factors``) needs.
        ``step`` > 1 subsamples for speed (see :meth:`_net_totals`). Returns
        ``{}`` only when there are no elements; an all-zero window yields zeros
        (the caller's quality filter decides what to drop).
        """
        return self._net_totals(bbox, mask, step=step)

    def coverage_frac(self, bbox: Tuple[int, int, int, int],
                      step: int = 1) -> float:
        """Fraction of pixels in *bbox* with positive tile coverage (weight > 0).

        A cheap "hole" signal for the calibration quality filter: a tile bbox
        sitting over a hole / unmapped region of the mosaic reads low here.
        ``step`` > 1 subsamples (the fraction is scale-free). Returns 1.0 for an
        empty bbox (nothing to flag).
        """
        y0, y1, x0, x1 = (int(v) for v in bbox)
        step = max(1, int(step))
        w = np.asarray(self._weight[y0:y1:step, x0:x1:step])
        if w.size == 0:
            return 1.0
        return float(np.count_nonzero(w > 0) / w.size)


def roi_from_cache_ds(cache_dir: Path, roi: Tuple[int, int, int, int],
                      elements: List[str], blend_mode: str = "mean",
                      roi_mask: Optional[np.ndarray] = None,
                      return_weight: bool = False,
                      progress_cb=None, cancel_flag=None,
                      progress: bool = True, max_workers: int = 1,
                      chunks_per_slice: int = 8, out_ds: int = 1,
                      ) -> Dict[str, Dict[str, np.ndarray]] | Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    """Downsampled cache read: slice at the cache ds grid, resize to BSE ROI size.

    Keeps the caller-visible API identical to ``roi_from_cache`` / ``roi_to_banded_maps``.

    ``out_ds`` (default 1) further decimates the *output* grid by that integer
    stride — the large-ROI memory guard relies on it to fit a huge ROI in RAM
    even when the cache itself is stored at ``ds>1`` (e.g. a two-resolution
    Oxford cache, BSE native + coarser element maps). Output arrays then have
    shape ``(len(range(0, H_bse, out_ds)), len(range(0, W_bse, out_ds)))``,
    matching ``roi_from_cache``'s ``out_ds`` convention. A full-resolution
    ``roi_mask`` is decimated to match. ``out_ds=1`` is the original behaviour
    (resize straight to the BSE ROI size).
    """
    from skimage.transform import resize

    data = load_manifest(cache_dir)
    if not data:
        raise RuntimeError("No cache manifest; build the cache first.")
    ds = max(1, int(data.get("ds", 1)))

    y0, y1, x0, x1 = map(int, roi)
    H_bse = y1 - y0
    W_bse = x1 - x0
    if H_bse <= 0 or W_bse <= 0:
        raise ValueError("ROI must have positive size.")

    # Output grid after optional nearest-pixel decimation of the BSE ROI.
    out_ds = max(1, int(out_ds))
    H_out = len(range(0, H_bse, out_ds))
    W_out = len(range(0, W_bse, out_ds))

    # Map BSE ROI → cache-resolution ROI (inclusive outer bound)
    cy0 = y0 // ds
    cx0 = x0 // ds
    cy1 = (y1 + ds - 1) // ds
    cx1 = (x1 + ds - 1) // ds

    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    import threading

    H_cache_roi = cy1 - cy0
    W_cache_roi = cx1 - cx0
    W_cache_full = int(data.get("W", W_cache_roi))

    max_chunk_rows = 500
    chunk_rows = max(1, min(H_cache_roi, max_chunk_rows))
    n_chunks_per_file = max(1, (H_cache_roi + chunk_rows - 1) // chunk_rows)
    n_files = len(elements) * len(BANDS) + 1
    total_steps = n_files * n_chunks_per_file

    fullrow_threshold = 0.25
    use_full_rows = (W_cache_roi / max(1, W_cache_full)) >= fullrow_threshold

    pbar = tqdm(total=total_steps, desc="ROI from cache (ds>1)",
                unit="chunk", miniters=1,
                postfix={"mode": "full-rows" if use_full_rows else "strided",
                         "chunk_rows": chunk_rows,
                         "files": n_files}) if progress else None
    done_count = [0]
    tick_lock = threading.Lock()

    def _tick():
        with tick_lock:
            done_count[0] += 1
            d = done_count[0]
        if pbar is not None:
            pbar.update(1)
        if progress_cb is not None:
            progress_cb(d, total_steps)

    def _slice_chunked(path: Path) -> np.ndarray:
        mm = np.lib.format.open_memmap(path, mode="r")
        out = np.empty((H_cache_roi, W_cache_roi), dtype=np.float32)
        for y0_chunk in range(0, H_cache_roi, chunk_rows):
            if cancel_flag is not None and cancel_flag():
                raise RuntimeError("Cancelled")
            y1_chunk = min(H_cache_roi, y0_chunk + chunk_rows)
            if use_full_rows:
                full_rows = np.asarray(
                    mm[cy0 + y0_chunk:cy0 + y1_chunk, :], dtype=np.float32)
                out[y0_chunk:y1_chunk] = full_rows[:, cx0:cx1]
            else:
                out[y0_chunk:y1_chunk] = np.asarray(
                    mm[cy0 + y0_chunk:cy0 + y1_chunk, cx0:cx1], dtype=np.float32)
            _tick()
        return out

    try:
        for el in elements:
            for band in BANDS:
                pth = _element_path(cache_dir, el, band)
                if not pth.exists():
                    raise FileNotFoundError(f"Cache missing {pth.name}")

        workers = max(1, min(int(max_workers), n_files))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            weight_fut = pool.submit(_slice_chunked, _weight_path(cache_dir))
            band_futs: Dict[Tuple[str, str], "Future"] = {}
            for el in elements:
                for band in BANDS:
                    band_futs[(el, band)] = pool.submit(
                        _slice_chunked, _element_path(cache_dir, el, band)
                    )

            w_ds = weight_fut.result()
            if cancel_flag is not None and cancel_flag():
                raise RuntimeError("Cancelled")

            if blend_mode not in ("mean", "sum"):
                blend_mode = "mean"

            m = w_ds > 0 if blend_mode == "mean" else None
            invw = None
            if blend_mode == "mean" and m.any():
                invw = np.zeros_like(w_ds)
                invw[m] = 1.0 / w_ds[m]

            # Decimate a full-resolution ROI mask to the output grid so the
            # element-wise multiply below has matching shapes under out_ds>1.
            mask_f = None
            if roi_mask is not None:
                rm = np.asarray(roi_mask)
                if rm.shape != (H_out, W_out):
                    rm = rm[::out_ds, ::out_ds][:H_out, :W_out]
                mask_f = rm.astype(np.float32, copy=False)

            out: Dict[str, Dict[str, np.ndarray]] = {}
            for (el, band), fut in band_futs.items():
                arr_ds = fut.result()
                if blend_mode == "mean" and invw is not None:
                    arr_ds = arr_ds * invw
                arr = resize(arr_ds, (H_out, W_out), order=1,
                             preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)
                if mask_f is not None:
                    arr = arr * mask_f
                out.setdefault(el, {})[band] = arr
    finally:
        if pbar is not None:
            pbar.close()

    if return_weight:
        w_full = resize(w_ds, (H_out, W_out), order=1,
                        preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)
        return out, w_full
    return out


# ---------- Cache build ----------

def build_cache(
    manifest_path: Path,
    elements: List[str],
    width_eV: float = 150.0,
    sideband_mult: float = 1.5,
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    ds: int = 1,
    max_workers: int = 0,
    progress: bool = True,
    shift_method: str = "bilinear",
    flush_every: int = 50,
    cache_dir: Optional[Path] = None,
    coating_elements: Sequence[str] = ("C",),
) -> Path:
    """Build the EDS mosaic cache for every tile in the stitch manifest.

    Accumulates raw weighted sums (per element × band) + weight into memmapped
    .npy files under ``<manifest.parent>/eds_cache/``. Writes ``manifest.json``.

    ``ds`` downsamples the cache resolution relative to BSE: cache shape is
    ``(H_mosaic // ds, W_mosaic // ds)``. ROI reads handle the upscaling.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os
    from tqdm import tqdm

    from eds_map import (
        read_manifest, load_eds_signal_lazy, extract_eds_metadata_from_signal,
        _resolve_manifest_path, _band_indices, _sideband_centers,
        LINE_ENERGIES_KEV, _process_one_tile,
    )

    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    if cache_dir is None:
        cache_dir = cache_dir_for(base_dir)
        clear_cache_link(base_dir)  # fresh build at default location — drop any stale link
    else:
        cache_dir = Path(cache_dir).resolve()
        write_cache_link(base_dir, cache_dir)

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty.")

    # Energy axis: prefer manifest, fall back to first readable tile
    E0 = dE = None
    nE = None
    for r in rows:
        try:
            _nE = int(r.get("nE") or 0)
            if _nE and r.get("dE") is not None and r.get("E0") is not None:
                E0 = float(r["E0"]); dE = float(r["dE"]); nE = _nE
                break
        except (ValueError, TypeError):
            continue
    if nE is None:
        for r in rows:
            p = _resolve_manifest_path(r["file"], base_dir)
            sig = load_eds_signal_lazy(p)
            if sig is None:
                continue
            meta = extract_eds_metadata_from_signal(sig)
            if meta.get("nE", 0) and meta.get("dE") is not None and meta.get("E0") is not None:
                E0 = float(meta["E0"]); dE = float(meta["dE"]); nE = int(meta["nE"])
                break
    if nE is None:
        raise RuntimeError("No EDS signals found in manifest (or missing energy axis).")

    elements = [e for e in elements if e in LINE_ENERGIES_KEV]
    if not elements:
        raise ValueError("No valid elements requested.")

    # Mosaic extent at BSE resolution
    H_bse = 0
    W_bse = 0
    for r in rows:
        y_end = int(r["y"]) + int(r["bse_h"])
        x_end = int(r["x"]) + int(r["bse_w"])
        if y_end > H_bse:
            H_bse = y_end
        if x_end > W_bse:
            W_bse = x_end
    if H_bse <= 0 or W_bse <= 0:
        raise RuntimeError("Degenerate mosaic dimensions from manifest.")

    ds = max(1, int(ds))
    H = H_bse // ds
    W = W_bse // ds
    if H <= 0 or W <= 0:
        raise ValueError(f"ds={ds} collapses mosaic to zero size.")

    bands_def = {}
    for e in elements:
        Ec = LINE_ENERGIES_KEV[e]
        lo_main, hi_main = _band_indices(E0, dE, nE, Ec, width_eV)
        Ec_lo, Ec_hi = _sideband_centers(Ec, width_eV, sideband_mult)
        lo_lo, hi_lo = _band_indices(E0, dE, nE, Ec_lo, width_eV)
        lo_hi, hi_hi = _band_indices(E0, dE, nE, Ec_hi, width_eV)
        bands_def[e] = {"main": (lo_main, hi_main),
                        "sb_lo": (lo_lo, hi_lo),
                        "sb_hi": (lo_hi, hi_hi)}

    # Clear any prior files (a new build always overwrites)
    invalidate(cache_dir)
    accum, weight = allocate_memmaps(cache_dir, elements, (H, W))

    # Use BSE-resolution ROI covering the whole mosaic; _process_one_tile
    # returns patches in mosaic coordinates (yB0..y_end, xB0..x_end).
    roi_full = (0, H_bse, 0, W_bse)
    # Default to 4 workers: each tile load materialises a subcube (~0.8-1.5 GB
    # for a typical Bruker tile) so 4 workers keep peak subcube RAM around
    # 3-6 GB. Caller can override with max_workers > 0.
    n_workers = min(
        max_workers if max_workers > 0 else 4,
        max(len(rows), 1),
        8,
    )

    def _accum_result(result):
        yB0 = result["yB0"]; yB1 = result["yB1"]
        xB0 = result["xB0"]; xB1 = result["xB1"]
        y_end = result["y_end"]; x_end = result["x_end"]
        ww = result["ww"]; ww_eff = result["ww_eff"]
        patches = result["patches"]

        if ds == 1:
            for e in elements:
                pe = patches.get(e, {})
                for band in BANDS:
                    patch = pe.get(band)
                    if patch is None:
                        continue
                    accum[e][band][yB0:y_end, xB0:x_end] += patch * ww_eff
            if result["has_mask"]:
                weight[yB0:y_end, xB0:x_end] += ww_eff
            else:
                weight[yB0:yB1, xB0:xB1] += ww
        else:
            # Downsampled accumulation: block-average patch and weight onto ds grid.
            # Target bounds in cache pixels
            ty0 = yB0 // ds; tx0 = xB0 // ds
            ty1 = (y_end + ds - 1) // ds; tx1 = (x_end + ds - 1) // ds
            if ty1 <= ty0 or tx1 <= tx0:
                return
            ph = (ty1 - ty0) * ds
            pw = (tx1 - tx0) * ds

            def _fit(arr):
                out = np.zeros((ph, pw), dtype=np.float32)
                h_src, w_src = arr.shape
                out[:h_src, :w_src] = arr
                return out.reshape(ty1 - ty0, ds, tx1 - tx0, ds).mean(axis=(1, 3))

            ww_src = ww_eff if result["has_mask"] else ww
            ww_ds = _fit(ww_src)
            weight[ty0:ty1, tx0:tx1] += ww_ds

            for e in elements:
                pe = patches.get(e, {})
                for band in BANDS:
                    patch = pe.get(band)
                    if patch is None:
                        continue
                    accum[e][band][ty0:ty1, tx0:tx1] += _fit(patch * ww_eff)

    from concurrent.futures import wait, FIRST_COMPLETED

    flush_every = max(1, int(flush_every))

    def _flush_all():
        weight.flush()
        for _e in elements:
            for _b in BANDS:
                accum[_e][_b].flush()

    def _submit(row):
        return pool.submit(
            _process_one_tile,
            row, base_dir, elements, bands_def, E0, dE, nE,
            roi_full, feather, tukey_alpha, apply_subpixel, None, None,
            shift_method,
        )

    # Bounded submission window. Submitting all 2014 futures up-front keeps
    # every completed Future alive until the whole batch finishes — each
    # Future holds its ~100 MB result dict, so memory blows up at scale.
    # With a window of 2*workers, at most that many result dicts are live.
    window = max(n_workers * 2, 4)
    tiles_done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        row_iter = iter(rows)
        pending = set()
        for _ in range(window):
            try:
                pending.add(_submit(next(row_iter)))
            except StopIteration:
                break

        pbar = tqdm(total=len(rows), desc="EDS cache", unit="tile") if progress else None
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    res = fut.result()
                    if res is not None:
                        _accum_result(res)
                    if pbar is not None:
                        pbar.update(1)
                    tiles_done += 1
                    if tiles_done % flush_every == 0:
                        _flush_all()
                    try:
                        pending.add(_submit(next(row_iter)))
                    except StopIteration:
                        pass
        finally:
            if pbar is not None:
                pbar.close()

    # Flush memmaps to disk before writing the manifest
    weight.flush()
    for e in elements:
        for band in BANDS:
            accum[e][band].flush()

    save_manifest(cache_dir, {
        "stitch_fingerprint": stitch_fingerprint(manifest_path),
        "elements": elements,
        "width_eV": float(width_eV),
        "sideband_mult": float(sideband_mult),
        "ds": ds,
        "H": int(H),
        "W": int(W),
        "H_bse": int(H_bse),
        "W_bse": int(W_bse),
        "E0": float(E0), "dE": float(dE), "nE": int(nE),
        "shift_method": str(shift_method),
        "data_type": "bcf_net_counts",
        "source": "Bruker BCF tiles",
        "coating_elements": list(coating_elements),
    })

    return cache_dir


# ---------- Preview PNG derivation ----------

def write_preview_pngs(cache_dir: Path, out_dir: Path, preview_ds: int = 8,
                       pct: Tuple[float, float] = (1.0, 99.0),
                       elements: Optional[List[str]] = None) -> Path:
    """Derive 8-bit preview PNGs from the EDS mosaic cache.

    For each element, reads ``{El}__main.npy`` + ``weight.npy`` in row-chunks,
    block-averages to the preview resolution (weighted mean: sum(main) /
    sum(weight) per block), percentile-clips, and writes
    ``<out_dir>/eds_preview_ds{preview_ds}/{El}.png``.

    ``preview_ds`` is interpreted relative to the BSE mosaic. If the cache was
    built at ``cache_ds`` > 1, the effective reduction from cache is
    ``preview_ds // cache_ds`` (clamped to ≥1). If ``preview_ds`` isn't a clean
    multiple of ``cache_ds``, it's floored (a note is printed).

    Only the ``main`` band is rendered — sidebands exist for background
    subtraction at ROI time and are never useful as a standalone display.
    """
    from PIL import Image

    cache_dir = Path(cache_dir)
    data = load_manifest(cache_dir)
    if not data:
        raise RuntimeError(f"No cache manifest at {cache_dir}; build the cache first.")

    cache_ds = max(1, int(data.get("ds", 1)))
    H_cache = int(data["H"])
    W_cache = int(data["W"])
    if elements is None:
        elements = list(data.get("elements", []))
    elements = [e for e in elements if (cache_dir / f"{e}__main.npy").exists()]
    if not elements:
        raise RuntimeError("No cached elements available for preview rendering.")

    preview_ds = max(1, int(preview_ds))
    reduce_factor = max(1, preview_ds // cache_ds)
    effective_ds = cache_ds * reduce_factor
    if effective_ds != preview_ds:
        print(f"[eds-preview] Requested preview_ds={preview_ds} not a multiple of "
              f"cache_ds={cache_ds}; using effective ds={effective_ds}.")

    outp = Path(out_dir) / f"eds_preview_ds{effective_ds}"
    outp.mkdir(parents=True, exist_ok=True)

    weight_mm = np.lib.format.open_memmap(_weight_path(cache_dir), mode="r")
    Hd = H_cache // reduce_factor
    Wd = W_cache // reduce_factor
    if Hd <= 0 or Wd <= 0:
        raise RuntimeError(f"Downsample factor collapses shape to zero "
                           f"(cache {H_cache}x{W_cache}, reduce {reduce_factor}).")

    # Target ~256 MB per chunk for main+weight together
    target_bytes = 256 * 1024 * 1024
    rows_per_chunk = max(1, target_bytes // max(1, W_cache * 4 * reduce_factor * 2))

    for el in elements:
        main_mm = np.lib.format.open_memmap(_element_path(cache_dir, el, "main"), mode="r")

        out = np.zeros((Hd, Wd), dtype=np.float32)
        if reduce_factor == 1:
            m = np.asarray(main_mm[:Hd, :Wd], dtype=np.float32)
            w = np.asarray(weight_mm[:Hd, :Wd], dtype=np.float32)
            mask = w > 0
            out[mask] = m[mask] / w[mask]
        else:
            Wc = Wd * reduce_factor
            for y0_out in range(0, Hd, rows_per_chunk):
                y1_out = min(Hd, y0_out + rows_per_chunk)
                y0_in = y0_out * reduce_factor
                y1_in = y1_out * reduce_factor
                m_chunk = np.asarray(main_mm[y0_in:y1_in, :Wc], dtype=np.float32)
                w_chunk = np.asarray(weight_mm[y0_in:y1_in, :Wc], dtype=np.float32)
                rows = y1_out - y0_out
                m_sum = m_chunk.reshape(rows, reduce_factor, Wd, reduce_factor).sum(axis=(1, 3))
                w_sum = w_chunk.reshape(rows, reduce_factor, Wd, reduce_factor).sum(axis=(1, 3))
                mask = w_sum > 0
                chunk_out = np.zeros_like(m_sum)
                chunk_out[mask] = m_sum[mask] / w_sum[mask]
                out[y0_out:y1_out] = chunk_out

        # Percentile-clip to uint8
        finite = out[np.isfinite(out)]
        if finite.size:
            lo, hi = np.percentile(finite, pct)
        else:
            lo, hi = 0.0, 1.0
        img8 = (np.clip((out - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8) \
            if hi > lo else np.zeros_like(out, dtype=np.uint8)
        Image.fromarray(img8).save(outp / f"{el}.png")

    print(f"[eds-preview] Rendered {len(elements)} PNGs from cache → {outp}")
    return outp


# ---------- Defragmentation ----------

def defragment(cache_dir: Path, progress: bool = True) -> None:
    """Rewrite each cache file to force a fresh NTFS extent allocation.

    Cache files built tile-by-tile can accumulate thousands of non-contiguous
    extents on HDDs, causing slow ROI reads (observed ~1 MB/s on a fragmented
    60 GB cache vs ~100 MB/s on a freshly-allocated contiguous one). Copying
    each file to a new name and renaming forces NTFS to allocate a new extent
    for the new file, typically contiguous if the disk has enough free space.

    Original files are replaced atomically (``os.replace``). Safe to abort —
    partial ``.defrag-tmp`` files are cleaned up on the next run. Requires the
    cache to be closed elsewhere (no open memmaps), so close the viewer first.

    Requires free disk space at least as large as the biggest file
    (~5.4 GB for a full-res 500 Mpx mosaic). The existing file is deleted
    only after the new one is successfully written.
    """
    import shutil as _shutil
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files in {cache_dir}")

    # Clean up any leftover temp files from a prior aborted defrag
    for stale in cache_dir.glob("*.defrag-tmp"):
        try:
            stale.unlink()
        except OSError:
            pass

    largest = max(f.stat().st_size for f in files)
    try:
        free = _shutil.disk_usage(cache_dir).free
    except OSError:
        free = largest  # can't check; optimistic
    if free < largest:
        raise RuntimeError(
            f"Not enough free disk space: need {largest/1e9:.1f} GB, "
            f"have {free/1e9:.1f} GB. Free some space and retry."
        )

    total_bytes = sum(f.stat().st_size for f in files)
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True,
                desc="Defragmenting cache", miniters=1) if progress else None

    CHUNK = 64 * 1024 * 1024  # 64 MB reads keep throughput sequential

    try:
        for pth in files:
            tmp_path = pth.with_suffix(pth.suffix + ".defrag-tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            with open(pth, "rb") as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
                dst.flush()
                try:
                    import os as _os
                    _os.fsync(dst.fileno())
                except OSError:
                    pass
            import os as _os
            _os.replace(tmp_path, pth)
    finally:
        if pbar is not None:
            pbar.close()


# ---------- Disk estimate ----------

def estimate_disk_bytes(H: int, W: int, n_elements: int, ds: int = 1) -> int:
    """Approximate on-disk size for a cache of (H, W) at n_elements × 3 bands + weight."""
    H_ds = max(1, H // max(1, ds))
    W_ds = max(1, W // max(1, ds))
    n_arrays = 3 * max(0, n_elements) + 1  # + weight
    return int(H_ds * W_ds * 4 * n_arrays)


def format_bytes(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} TB"


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        prog="python eds_cache.py",
        description="EDS mosaic cache utilities.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser(
        "defragment",
        help="Rewrite each .npy in the cache directory to a fresh NTFS extent "
             "(one-time pass for HDD-stored caches that suffer from tile-order "
             "write fragmentation).",
    )
    pd.add_argument("cache_dir", help="Path to the <stitch_dir>/eds_cache/ folder.")

    ps = sub.add_parser(
        "status",
        help="Print a short summary of an EDS cache (fingerprint, elements, ds, size).",
    )
    ps.add_argument("cache_dir", help="Path to the <stitch_dir>/eds_cache/ folder.")

    args = p.parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f"Not a directory: {cache_dir}")

    if args.cmd == "defragment":
        defragment(cache_dir)
        print(f"[eds-cache] Defragmented {cache_dir}.")
    elif args.cmd == "status":
        data = load_manifest(cache_dir)
        if not data:
            print(f"No manifest at {cache_dir}.")
        else:
            total = sum(f.stat().st_size for f in cache_dir.glob("*.npy"))
            print(f"Cache: {cache_dir}")
            print(f"  data_type    : {data.get('data_type', 'bcf_net_counts')}")
            print(f"  source       : {data.get('source', 'unknown')}")
            print(f"  elements     : {data.get('elements')}")
            print(f"  width_eV     : {data.get('width_eV')}")
            print(f"  ds           : {data.get('ds')}")
            print(f"  shape (H,W)  : ({data.get('H')}, {data.get('W')})")
            print(f"  shift_method : {data.get('shift_method', 'unknown')}")
            print(f"  fingerprint  : {data.get('stitch_fingerprint','?')[:16]}…")
            print(f"  total size   : {format_bytes(total)}")
