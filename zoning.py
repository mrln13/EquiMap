# zoning.py
"""
Per-pixel compositional index maps and line (traverse) profiles.

The rest of the pipeline reduces a phase domain to its *mean* composition;
this module exposes the spatial variation inside a domain — the classic
metamorphic use case being growth zoning (Mg# across a garnet, An% across a
plagioclase) read from a core-to-rim traverse.

Index maps are molar cation ratios computed per pixel from the ROI band maps:

    value = Σ_num m_i / Σ_den m_i,    m_i = k_i · net_i / A_i

with ``net_i`` the (line-aware) net count, smoothed then clipped at 0, ``A_i`` the atomic weight,
and ``k_i`` the SPX-anchored calibration factor when available. With k-factors
(or already-quantified Oxford wt% maps) the ratio is a true molar ratio; on
raw Bruker net counts it is an *intensity* ratio — still monotonic in the real
zoning, but not absolute (callers should label it as uncalibrated). Because
indices are ratios, the per-pixel renormalisation constant cancels — only the
relative k's matter.

Per-element maps return the element's share of the (calibrated) total signal
in percent, so they read as quasi-wt% when calibrated and net-count % when not.

Counting statistics: smoothing (``smooth_sigma``) is applied to the *net count
maps before any division* — smoothing numerator and denominator separately is
the statistically correct way to stabilise a ratio of noisy counts.

Qt-free / napari-free; only numpy + scipy + thermo.composition.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from thermo.composition import ATOMIC_WEIGHTS

# index name -> (numerator elements, denominator elements); all molar
INDEX_DEFS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "Mg#": (("Mg",), ("Mg", "Fe")),
    "Fe#": (("Fe",), ("Fe", "Mg")),
    "An":  (("Ca",), ("Ca", "Na", "K")),
    "Or":  (("K",),  ("Ca", "Na", "K")),
    "Cr#": (("Cr",), ("Cr", "Al")),
}


def _net_map(band_maps: Dict[str, Dict[str, np.ndarray]], el: str,
             smooth_sigma: float = 0.0) -> Optional[np.ndarray]:
    """Non-negative net-count map for one element, optionally smoothed.

    The signed net (``eds_background.net_count_maps``, line-aware background
    by default) is smoothed FIRST and clipped at zero after, so the
    smoothing pools the unbiased signed noise instead of its rectified
    half (a per-pixel clip before pooling biases every sparse element up).
    """
    if el not in band_maps:
        return None
    from eds_background import net_count_maps
    net = net_count_maps(band_maps, elements=[el]).get(el)
    if net is None:
        return None
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        net = gaussian_filter(net, sigma=float(smooth_sigma))
    return np.maximum(net, 0.0)


def _molar_map(band_maps, el: str, k_factors: Optional[Dict[str, float]],
               smooth_sigma: float) -> Optional[np.ndarray]:
    net = _net_map(band_maps, el, smooth_sigma)
    if net is None:
        return None
    k = float((k_factors or {}).get(el, 1.0))
    return net * (k / ATOMIC_WEIGHTS[el])


def index_map(band_maps: Dict[str, Dict[str, np.ndarray]],
              index: str,
              k_factors: Optional[Dict[str, float]] = None,
              smooth_sigma: float = 0.0,
              sub_mask: Optional[np.ndarray] = None,
              ) -> np.ndarray:
    """Per-pixel molar index map (float32, NaN where undefined).

    ``index`` is a key of :data:`INDEX_DEFS`. Pixels with zero denominator
    signal, missing required elements, or outside ``sub_mask`` are NaN.
    Raises ``KeyError`` for an unknown index; missing *numerator* elements
    count as zero (e.g. K-free maps still give An), but a fully missing
    denominator yields all-NaN.
    """
    if index not in INDEX_DEFS:
        raise KeyError(f"unknown index {index!r}; have {sorted(INDEX_DEFS)}")
    num_els, den_els = INDEX_DEFS[index]

    sample = next(iter(band_maps.values()))["main"]
    num = np.zeros(sample.shape[:2], dtype=np.float32)
    den = np.zeros(sample.shape[:2], dtype=np.float32)
    for el in den_els:
        m = _molar_map(band_maps, el, k_factors, smooth_sigma)
        if m is None:
            continue
        den += m
        if el in num_els:
            num += m

    out = np.full(sample.shape[:2], np.nan, dtype=np.float32)
    ok = den > 0
    if sub_mask is not None:
        ok &= np.asarray(sub_mask, dtype=bool)
    np.divide(num, den, out=out, where=ok)
    out[~ok] = np.nan
    return out


def element_map(band_maps: Dict[str, Dict[str, np.ndarray]],
                el: str,
                k_factors: Optional[Dict[str, float]] = None,
                smooth_sigma: float = 0.0,
                sub_mask: Optional[np.ndarray] = None,
                ) -> np.ndarray:
    """Per-pixel share of one element in the total signal, in percent.

    With k-factors the value is quasi-wt% over the calibrated elements (the
    same renormalisation ``eds_quantify.apply_calibration_maps`` uses); without
    them it is the net-count percentage. NaN where the total signal is zero or
    outside ``sub_mask``.
    """
    if el not in band_maps:
        raise KeyError(f"element {el!r} not in band maps")
    # Calibrated mode only when the *requested* element has a k-factor —
    # mixing calibrated and raw intensities in one total would be meaningless.
    if k_factors and el in k_factors:
        els = [e for e in band_maps if e in k_factors]
        use_k: Optional[Dict[str, float]] = k_factors
    else:
        els = list(band_maps)
        use_k = None

    sample = next(iter(band_maps.values()))["main"]
    total = np.zeros(sample.shape[:2], dtype=np.float32)
    target = None
    for e in els:
        net = _net_map(band_maps, e, smooth_sigma)
        if net is None:
            continue
        w = net * float((use_k or {}).get(e, 1.0))
        total += w
        if e == el:
            target = w
    out = np.full(sample.shape[:2], np.nan, dtype=np.float32)
    if target is None:
        return out
    ok = total > 0
    if sub_mask is not None:
        ok &= np.asarray(sub_mask, dtype=bool)
    np.divide(100.0 * target, total, out=out, where=ok)
    out[~ok] = np.nan
    return out


def line_profile(value_map: np.ndarray,
                 p0: Sequence[float], p1: Sequence[float],
                 n_samples: int = 200,
                 width: int = 1,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a value map along the segment p0→p1 ((y, x), map coordinates).

    Bilinear interpolation via ``scipy.ndimage.map_coordinates``; NaNs in the
    map propagate to samples that touch them. ``width`` > 1 averages that many
    parallel lines spaced 1 px along the segment normal (``nanmean``), the
    usual way to beat per-pixel counting noise on a traverse.

    Returns ``(distances_px, values)`` with distances measured from p0 in map
    pixels (callers convert to mosaic px / µm).
    """
    from scipy.ndimage import map_coordinates

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    n_samples = max(2, int(n_samples))
    length = float(np.hypot(*(p1 - p0)))
    t = np.linspace(0.0, 1.0, n_samples)
    ys = p0[0] + t * (p1[0] - p0[0])
    xs = p0[1] + t * (p1[1] - p0[1])

    if length > 0:
        ny, nx = (p1[1] - p0[1]) / length, -(p1[0] - p0[0]) / length
    else:
        ny, nx = 0.0, 0.0
    width = max(1, int(width))
    offsets = np.arange(width, dtype=float) - (width - 1) / 2.0

    vm = np.asarray(value_map, dtype=float)
    rows = []
    for off in offsets:
        coords = np.vstack([ys + off * ny, xs + off * nx])
        rows.append(map_coordinates(vm, coords, order=1, mode="constant",
                                    cval=np.nan))
    import warnings
    with warnings.catch_warnings():
        # all-NaN columns (samples fully off-map) are a legitimate NaN result
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vals = np.nanmean(np.vstack(rows), axis=0)
    return t * length, vals
