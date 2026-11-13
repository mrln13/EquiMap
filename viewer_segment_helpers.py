# viewer_segment_helpers.py
"""
Phase segmentation helpers for viewer.py:
- Map collection from preview/full-res sources (`_collect_segment_maps`).
- Spatial smoothing and morphological cleanup
  (`_smooth_element_maps`, `_clean_label_map`).
- Overlap-mask construction (`_compute_overlap_mask_from_weight`,
  `_compute_overlap_mask_from_manifest`).
- Feature matrix + normalisation (`_build_feature_matrix`,
  `_normalize_features`).
- Auto-k selection via ICL + knee fallback in PCA space
  (`_find_knee`, `_select_k_auto`, `_run_clustering`).
- Cluster naming + napari layer construction (`_cluster_dominant_elements`,
  `_label_map_to_rgba`, `_remove_phase_layers`, `_add_phase_cluster_layers`).

Moved out of viewer.py as part of the tier-B refactor.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Sequence

import numpy as np
import matplotlib


_PHASE_LAYER_PREFIX = "Phase: "


def _collect_segment_maps(
    viewer,
    elements: List[str],
    source: str,
    roi: Optional[Tuple[int, int, int, int]],
    band_maps: Optional[Dict],
) -> Tuple[Dict[str, np.ndarray], Tuple[float, float]]:
    """Return {element: 2D float32 map} and (translate_y, translate_x) for napari placement."""
    maps: Dict[str, np.ndarray] = {}
    translate = (0.0, 0.0)

    if source == "preview_full":
        for el in elements:
            lyr = viewer.layers[f"{el} preview"]
            maps[el] = lyr.data.astype(np.float32)

    elif source == "preview_roi":
        if roi is None:
            raise ValueError("No ROI available for preview crop.")
        y0, y1, x0, x1 = roi
        first_lyr = viewer.layers[f"{elements[0]} preview"]
        ds = int(round(first_lyr.scale[0]))
        py0, py1 = y0 // ds, y1 // ds
        px0, px1 = x0 // ds, x1 // ds
        translate = (float(py0 * ds), float(px0 * ds))
        for el in elements:
            arr = viewer.layers[f"{el} preview"].data.astype(np.float32)
            maps[el] = arr[py0:py1, px0:px1]

    elif source == "fullres_roi":
        if band_maps is None:
            raise ValueError("No full-resolution band maps available.")
        translate = (float(roi[0]), float(roi[2])) if roi else (0.0, 0.0)
        # SIGNED nets (line-aware background, plan over every element in the
        # dict so a neighbour's line is known even if it is not clustered on).
        # They are clipped at zero only after Gaussian smoothing, in
        # _smooth_element_maps: pooling ~4*pi*sigma^2 px of unbiased signed
        # noise first is what makes sparse elements (Mg, Na) usable features
        # instead of rectified noise.
        from eds_background import default_scheme, net_count_maps
        legacy = default_scheme() == "sideband"
        # Legacy replay ("sideband" scheme): the old pipeline clipped per
        # pixel BEFORE smoothing; keep that so a pre-2026-09 recipe
        # reproduces its map byte for byte.
        nets = net_count_maps(band_maps, elements=[el for el in elements if el in band_maps],
                              clip=legacy)
        for el in elements:
            if el in nets:
                maps[el] = nets[el].astype(np.float32)
    else:
        raise ValueError(f"Unknown source: {source}")

    if not maps:
        raise ValueError("No element maps could be collected.")
    return maps, translate


def _noise_floor(smoothed: np.ndarray, min_tail: int = 100) -> float:
    """Noise level of a smoothed SIGNED net map: 1.4826 x MAD of its negative
    tail (the values a signal-free pixel takes). 0 when there is no tail."""
    neg = smoothed[smoothed < 0]
    if neg.size < min_tail:
        return 0.0
    return float(1.4826 * np.median(np.abs(neg)))


def _smooth_element_maps(
    maps: Dict[str, np.ndarray],
    sigma: float = 2.0,
    noise_floor: bool = True,
) -> Dict[str, np.ndarray]:
    """Apply Gaussian smoothing to each element map (spatial noise reduction),
    then clip at zero and add the element's noise level as a pseudo-count.

    This is critical for EDS data: per-pixel spectra are very noisy, so
    clustering raw pixels produces salt-and-pepper results.  Smoothing
    with a modest sigma (2-4 px) averages out per-pixel noise while
    preserving phase boundaries. The input maps are SIGNED nets
    (eds_background), so the clip comes after the pooling and the noise is
    averaged rather than rectified.

    ``noise_floor`` (default True): clipping alone leaves EXACT zeros
    wherever the smoothed net stayed negative (15-25 % of pixels for the
    sparse elements of a low-dose map). The row-normalised log features
    then put those pixels on a spike far from the small positive values,
    and the GMM clusters on that artefact (2026-09-16 on C23S1A: clean
    quartz swallowed by a mixed domain, pixel agreement with the July map
    79 %; with the floor 88 %). Adding each element's noise level
    (:func:`_noise_floor`, the negative-tail MAD of the smoothed map) keeps
    the features continuous; it is the smoothed-map analogue of the Poisson
    noise the legacy per-pixel clip rectified. Maps without a negative tail
    (legacy clipped nets) get a floor of 0, so old recipes replay unchanged.
    With ``sigma <= 0`` the maps are not smoothed, only floored.
    """
    from scipy.ndimage import gaussian_filter
    out: Dict[str, np.ndarray] = {}
    for el, arr in maps.items():
        sm = np.asarray(arr, dtype=np.float32)
        if sigma > 0:
            sm = gaussian_filter(sm, sigma=sigma)
        floor = _noise_floor(sm) if noise_floor else 0.0
        out[el] = np.maximum(sm, 0.0) + np.float32(floor)
    return out


def _clean_label_map(
    label_map: np.ndarray,
    min_size: int = 50,
) -> np.ndarray:
    """Morphological cleanup: remove tiny isolated clusters.

    Regions smaller than min_size pixels are reassigned to the nearest
    neighboring cluster via median filtering.
    """
    from scipy.ndimage import label as ndlabel, median_filter
    cleaned = label_map.copy()

    # Label ALL connected components across all clusters in one pass:
    # offset each cluster's labels so they don't collide.
    valid = cleaned >= 0
    all_labeled = np.zeros_like(cleaned, dtype=np.int32)
    offset = 0
    unique_cids = [c for c in np.unique(cleaned) if c >= 0]
    for cid in unique_cids:
        mask = cleaned == cid
        labeled, n_regions = ndlabel(mask)
        if n_regions > 0:
            labeled[labeled > 0] += offset
            all_labeled[mask] = labeled[mask]
            offset += n_regions

    if offset == 0:
        return cleaned

    # Vectorized size check across all regions at once
    sizes = np.bincount(all_labeled.ravel())  # index 0 = background
    small_mask = (sizes < min_size)
    small_mask[0] = False  # don't touch background
    needs_reassign = small_mask[all_labeled]

    if not needs_reassign.any():
        return cleaned

    # Mark small-region pixels for reassignment
    cleaned[needs_reassign] = -2

    # Reassign using median filter (size=3 is fast and sufficient)
    filt = median_filter(cleaned, size=3)
    still_bad = cleaned == -2
    cleaned[still_bad] = filt[still_bad]
    # Any remaining -2 (edge cases)
    still_bad = cleaned == -2
    if still_bad.any():
        cleaned[still_bad] = -1
    return cleaned


def _compute_overlap_mask_from_weight(
    weight: np.ndarray,
    threshold: float = 1.05,
) -> np.ndarray:
    """Boolean mask from the raw accumulation weight array.

    True = overlap pixel (received signal from multiple tiles).
    Threshold slightly above 1.0 to account for floating-point Tukey
    taper values that sum to just over 1.0 at tile centres.
    """
    return weight > threshold


def _compute_overlap_mask_from_manifest(
    rows: List[dict],
    shape: Tuple[int, int],
    roi: Optional[Tuple[int, int, int, int]] = None,
    ds: int = 1,
) -> np.ndarray:
    """Build a tile-overlap mask from manifest tile positions.

    Counts how many tiles cover each pixel and returns a boolean mask
    where True = pixel covered by more than one tile (overlap zone).
    Works for both full-resolution and downsampled preview coordinates.
    """
    H, W = shape
    coverage = np.zeros((H, W), dtype=np.int16)
    y_off = roi[0] if roi else 0
    x_off = roi[2] if roi else 0
    for r in rows:
        try:
            th = int(float(r["bse_h"]))
            tw = int(float(r["bse_w"]))
            iy = int(np.floor(float(r["y"])))
            ix = int(np.floor(float(r["x"])))
        except (KeyError, ValueError):
            continue
        cy0 = max(0, (iy - y_off) // ds)
        cx0 = max(0, (ix - x_off) // ds)
        cy1 = min(H, (iy + th - y_off) // ds)
        cx1 = min(W, (ix + tw - x_off) // ds)
        if cy1 > cy0 and cx1 > cx0:
            coverage[cy0:cy1, cx0:cx1] += 1
    return coverage > 1


# Count-based smoothing (opt-in; the dialog default stays a fixed sigma).
# The Gaussian pools ~4*pi*sigma**2 pixels, so pooling N counts needs
# sigma = sqrt(N / (4*pi*rate)), whatever the pixel size, dwell time, beam
# current or decimation. N = 1000 measures an element that carries 10 % of
# the counts to about 10 % (1/sqrt(100)).
AUTO_SMOOTH_TARGET_COUNTS = 1000.0
AUTO_SMOOTH_SIGMA_RANGE = (0.5, 10.0)


def _min_region_for_sigma(sigma: float) -> int:
    """Smallest region the cleanup keeps for a given smoothing sigma (px).

    The area of a disc one FWHM of the kernel across (FWHM = 2.355 sigma):
    a region smaller than that is below the resolution the smoothing leaves.
    """
    import math as _math
    if sigma <= 0:
        return 0
    return int(round(_math.pi * (1.1774 * float(sigma)) ** 2))


def _auto_smoothing(
    maps: Dict[str, np.ndarray],
    valid_pixels: Optional[np.ndarray] = None,
    target_counts: float = AUTO_SMOOTH_TARGET_COUNTS,
    exclude: Sequence[str] = ("C",),
    sigma_range: Tuple[float, float] = AUTO_SMOOTH_SIGMA_RANGE,
) -> Dict[str, float]:
    """Smoothing sigma (px) that pools ``target_counts`` per pixel, and the
    matching minimum region size.

    ``maps`` are the (signed) net-count maps that will be clustered. The rate
    is the median summed count per measured pixel, over every element except
    ``exclude`` (the coating). Only meaningful for COUNT maps: weight-percent
    maps (Oxford Wt% CSV) are not counts, and the caller must not use it there.

    Returns ``{"sigma", "min_region_size", "counts_per_px", "target_counts"}``;
    sigma is clipped to ``sigma_range`` (a starved map gets the upper bound).
    """
    import math as _math
    use = [el for el in maps if el not in set(exclude)]
    if not use:
        raise ValueError("no element maps to count")
    total = np.zeros_like(np.asarray(maps[use[0]], dtype=np.float64))
    for el in use:
        total += np.nan_to_num(np.asarray(maps[el], dtype=np.float64))
    if valid_pixels is None:
        valid_pixels = np.zeros(total.shape, dtype=bool)
        for el in maps:
            valid_pixels |= np.nan_to_num(np.asarray(maps[el])) != 0
    vals = total[valid_pixels]
    rate = float(np.median(vals)) if vals.size else 0.0
    lo, hi = float(sigma_range[0]), float(sigma_range[1])
    if rate <= 0:
        sigma = hi
    else:
        sigma = _math.sqrt(float(target_counts) / (4.0 * _math.pi * rate))
        sigma = min(hi, max(lo, sigma))
    sigma = round(sigma, 1)
    return {"sigma": sigma, "min_region_size": _min_region_for_sigma(sigma),
            "counts_per_px": rate, "target_counts": float(target_counts)}


def _sparse_element_exclusions(
    maps: Dict[str, np.ndarray],
    valid_pixels: np.ndarray,
    max_zero_pct: float = 90.0,
    sigma: float = 3.0,
    min_smoothed_counts: float = 3.0,
) -> Tuple[List[str], List[Tuple[str, float, float, bool]]]:
    """Decide which elements to exclude as too sparse to cluster on.

    On low-dose maps the raw zero fraction is almost exactly the Poisson
    ``exp(-rate)``, so a plain zero-fraction threshold is a hidden count-rate
    cutoff that throws away legitimately sparse discriminators (Mg/Na/Ca at a
    few counts per hundred pixels) together with real contamination (C).  The
    features are clustered *after* Gaussian smoothing, which pools roughly
    ``4*pi*sigma**2`` pixels — so an element whose pooled count
    ``rate * 4*pi*sigma**2`` is usable survives even when almost every raw
    pixel is zero.  An element is excluded only when it is BOTH raw-sparse
    (zero fraction above ``max_zero_pct``) AND still count-starved after
    smoothing (pooled counts below ``min_smoothed_counts``).  With
    ``sigma == 0`` there is no pooling and the behaviour reduces to the plain
    zero-fraction threshold.

    Returns ``(excluded_elements, report)`` where report rows are
    ``(element, zero_frac_pct, pooled_counts, excluded)``.
    """
    import math as _math
    pooling = 4.0 * _math.pi * float(sigma) ** 2 if sigma > 0 else 1.0
    excluded: List[str] = []
    report: List[Tuple[str, float, float, bool]] = []
    n_valid = int(valid_pixels.sum())
    for el in sorted(maps.keys()):
        arr = maps[el]
        if n_valid > 0:
            vals = arr[valid_pixels]
            zero_frac = 100.0 * float((vals < 0.5).mean())
            rate = float(vals.mean())
        else:
            zero_frac, rate = 100.0, 0.0
        pooled = rate * pooling
        drop = (zero_frac > max_zero_pct) and (pooled < min_smoothed_counts)
        if drop:
            excluded.append(el)
        report.append((el, zero_frac, pooled, drop))
    return excluded, report


def _build_feature_matrix(
    maps: Dict[str, np.ndarray],
    min_count_pct: float = 0.0,
    min_count_ratio: float = 0.1,
    overlap_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], List[str]]:
    """Stack element maps -> (n_valid, n_el) feature matrix.

    Parameters
    ----------
    maps : dict of element -> 2-D float32 array (net counts)
    min_count_pct : float in [0, 100)
        Discard pixels whose total count falls below this percentile of all
        non-zero pixels.
    min_count_ratio : float in [0, 1)
        Discard pixels whose total count is below this fraction of the
        **median** total count among non-zero pixels.
    overlap_mask : optional bool array (H, W)
        True = tile overlap pixel; excluded from the feature matrix.

    The final threshold is the **maximum** of the two filters.

    Returns (X, valid_mask, (H, W), element_names).
    """
    el_names = sorted(maps.keys())
    arrs = [maps[el] for el in el_names]

    # Crop to minimum common shape
    H = min(a.shape[0] for a in arrs)
    W = min(a.shape[1] for a in arrs)
    arrs = [a[:H, :W] for a in arrs]

    stack = np.stack(arrs, axis=-1)  # (H, W, n_el)
    flat = stack.reshape(-1, len(el_names))

    total = flat.sum(axis=1)
    finite = np.all(np.isfinite(flat), axis=1)
    nonzero = total > 0
    valid_base = finite & nonzero

    # Tile overlap exclusion - remove pixels where multiple tiles contribute
    if overlap_mask is not None:
        ol = overlap_mask[:H, :W].ravel()
        n_overlap = int((valid_base & ol).sum())
        valid_base = valid_base & ~ol
        pct_ol = 100.0 * n_overlap / max(int((finite & nonzero).sum()), 1)
        print(
            f"[segment] Overlap filter: excluded {n_overlap:,} pixels "
            f"({pct_ol:.1f}% of non-zero pixels)",
            flush=True,
        )

    # Low-count pixel filter - two complementary thresholds.
    apply_pct = min_count_pct > 0
    apply_ratio = min_count_ratio > 0
    if (apply_pct or apply_ratio) and valid_base.any():
        total_valid = total[valid_base]
        thresh_pct = 0.0
        thresh_ratio = 0.0

        if apply_pct:
            thresh_pct = float(np.percentile(total_valid, min_count_pct))
        if apply_ratio:
            median_total = float(np.median(total_valid))
            thresh_ratio = min_count_ratio * median_total

        threshold = max(thresh_pct, thresh_ratio, 1.0)

        count_ok = total >= threshold
        valid_flat = finite & count_ok
        n_excluded = int(valid_base.sum()) - int(valid_flat.sum())
        pct_excluded = 100.0 * n_excluded / max(int(valid_base.sum()), 1)
        parts = []
        if apply_pct:
            parts.append(f"pct={min_count_pct}->{thresh_pct:.1f}")
        if apply_ratio:
            parts.append(f"ratio={min_count_ratio}*median({median_total:.1f})->{thresh_ratio:.1f}")
        print(
            f"[segment] Low-count filter: threshold={threshold:.1f} counts "
            f"({', '.join(parts)}), excluded {n_excluded:,} pixels "
            f"({pct_excluded:.1f}% of non-zero pixels)",
            flush=True,
        )
    else:
        valid_flat = valid_base

    # Drop all-zero columns
    col_sums = flat[valid_flat].sum(axis=0)
    keep_cols = col_sums > 0
    if not np.all(keep_cols):
        dropped = [el_names[i] for i in range(len(el_names)) if not keep_cols[i]]
        print(f"[segment] Dropped all-zero elements: {dropped}", flush=True)
        el_names = [el_names[i] for i in range(len(el_names)) if keep_cols[i]]
        flat = flat[:, keep_cols]

    if len(el_names) < 2:
        raise ValueError("Need at least 2 non-zero elements for segmentation.")

    print(
        f"[segment] Feature matrix: {int(valid_flat.sum()):,} valid pixels, "
        f"{len(el_names)} elements",
        flush=True,
    )

    X = flat[valid_flat].astype(np.float32)
    valid_mask = valid_flat.reshape(H, W)
    return X, valid_mask, (H, W), el_names


def _normalize_features(
    X: np.ndarray,
    norm_method: str = "row",
    log_scale: bool = True,
) -> np.ndarray:
    """Normalize EDS feature matrix for clustering.

    Parameters
    ----------
    norm_method : "row" or "anscombe"
    log_scale : bool   Applied after row normalization when norm_method="row".

    Both paths end with a per-column robust z-score (median / MAD).
    """
    X = X.copy().astype(np.float64)

    if norm_method == "anscombe":
        X = np.sqrt(np.maximum(X, 0.0))
    else:
        # Row-normalize: each pixel -> apparent composition (sum=1)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums
        if log_scale:
            X = np.log1p(X * 1000.0)

    # Per-column robust z-score: (x - median) / MAD
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.median(col)
        mad = np.median(np.abs(col - med))
        if mad > 0:
            X[:, j] = (col - med) / mad
        else:
            X[:, j] = col - med

    return X.astype(np.float32)


def _find_knee(ks: List[int], scores: List[float]) -> int:
    """Find the knee/elbow point via max perpendicular distance from chord."""
    x = np.array(ks, dtype=np.float64)
    y = np.array(scores, dtype=np.float64)
    x_range = x[-1] - x[0]
    y_range = y[-1] - y[0]
    if x_range == 0 or y_range == 0:
        return ks[0]
    xn = (x - x[0]) / x_range
    yn = (y - y[0]) / y_range
    dists = np.abs(yn - xn)
    return ks[int(np.argmax(dists))]


def _select_k_auto(
    X: np.ndarray,
    k_max: int,
    on_progress=None,
    random_state: int = 42,
) -> Tuple[int, Dict[int, float], Dict[int, float], object, str]:
    """Select k via ICL (Integrated Classification Likelihood) with knee fallback.

    Returns (k_best, bic_scores, icl_scores, best_gmm, method_used).
    """
    from sklearn.mixture import GaussianMixture

    bic_scores: Dict[int, float] = {}
    icl_scores: Dict[int, float] = {}
    entropies: Dict[int, float] = {}
    models: Dict[int, object] = {}

    for i, k in enumerate(range(2, k_max + 1)):
        if on_progress:
            on_progress(i, k_max - 1, f"GMM-ICL sweep k={k} ...")
        gmm = GaussianMixture(
            n_components=k, n_init=5, max_iter=200,
            random_state=random_state, reg_covar=1e-4,
        )
        gmm.fit(X)
        bic = float(gmm.bic(X))
        resp = gmm.predict_proba(X)
        # Safe entropy: use 0*log(0) = 0 convention via np.where
        safe_log = np.where(resp > 0, np.log(np.maximum(resp, 1e-300)), 0.0)
        ent = float(-np.sum(resp * safe_log))
        bic_scores[k] = bic
        icl_scores[k] = bic + 2.0 * ent
        entropies[k] = ent
        models[k] = gmm

    # Diagnostic table
    ks = sorted(icl_scores)
    # ASCII only: a redirected stdout on Windows is cp1252 and a Greek delta
    # here raised UnicodeEncodeError after the whole GMM sweep had run.
    print(f"  {'k':>4}  {'BIC':>12}  {'ICL':>12}  {'EN(k)':>10}  {'dICL':>10}",
          flush=True)
    for i, k in enumerate(ks):
        d_icl = icl_scores[k] - icl_scores[ks[i - 1]] if i > 0 else 0.0
        print(f"  {k:4d}  {bic_scores[k]:12.1f}  {icl_scores[k]:12.1f}"
              f"  {entropies[k]:10.1f}  {d_icl:+10.1f}", flush=True)

    # Primary: argmin ICL (with NaN guard - fall back to BIC if ICL failed)
    finite_icl = {k: v for k, v in icl_scores.items() if np.isfinite(v)}
    if finite_icl:
        k_icl = min(finite_icl, key=finite_icl.__getitem__)
    else:
        print("  [warn] All ICL scores are NaN; falling back to BIC", flush=True)
        k_icl = min(bic_scores, key=bic_scores.__getitem__)

    if k_icl < k_max:
        method = "icl_min"
        k_best = k_icl
        print(f"  -> ICL minimum at k={k_best}", flush=True)
    else:
        # ICL monotonically decreasing - fall back to knee detection
        if finite_icl:
            ks_list = sorted(finite_icl)
            vals_list = [finite_icl[k] for k in ks_list]
        else:
            ks_list = sorted(bic_scores)
            vals_list = [bic_scores[k] for k in ks_list]
        k_best = _find_knee(ks_list, vals_list)
        method = "icl_knee"
        print(f"  -> ICL monotone; knee detection chose k={k_best}", flush=True)

    return k_best, bic_scores, icl_scores, models[k_best], method


def _density_balanced_sample(X_norm: np.ndarray, n: int, n_bins: int = 200,
                             probe: int = 300_000, seed: int = 42) -> np.ndarray:
    """Indices of ``n`` pixels drawn evenly across FEATURE space.

    A uniform pixel sample reproduces the areal abundance of every phase, so
    a phase that covers 1-2 % of the map contributes a few hundred points to
    the mixture fit and, when its boundary pixels grade into a host phase,
    EM absorbs it as a tail of the host's component (C23S1A: the sillimanite
    cores of the muscovite nodules were merged into muscovite/matrix at every
    k and pitch tried, 2026-09-17). Binning feature space with a quick
    k-means on a random probe and drawing the same number of points from
    every occupied bin gives minor populations the same weight as major
    ones in the fit; all pixels are still labelled by the fitted model.
    """
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    N = len(X_norm)
    probe_idx = rng.choice(N, min(probe, N), replace=False)
    km = MiniBatchKMeans(n_clusters=min(n_bins, len(probe_idx)), random_state=seed,
                         batch_size=4096, n_init=3).fit(X_norm[probe_idx])
    bins = np.empty(N, dtype=np.int32)
    chunk = 2_000_000
    for i in range(0, N, chunk):
        bins[i:i + chunk] = km.predict(X_norm[i:i + chunk])
    occupied = np.unique(bins)
    per_bin = max(1, n // len(occupied))
    out = []
    for b in occupied:
        members = np.flatnonzero(bins == b)
        out.append(rng.choice(members, min(per_bin, len(members)), replace=False))
    return np.concatenate(out)


def _run_clustering(
    X_norm: np.ndarray,
    n_clusters="auto",
    k_max: int = 12,
    min_k: int = 2,
    fit_sample: int = 30_000,
    pca_variance: float = 0.92,
    posterior_min: float = 0.0,
    density_pct: float = 0.0,
    on_progress=None,
    fit_balance: str = "uniform",
    seed_means: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, int, Optional[Dict[int, float]],
           Optional[Dict[int, float]], Optional[str]]:
    """Cluster pixels into phases.

    ``fit_balance``: ``"uniform"`` (default, the classic random pixel sample)
    or ``"density"`` (:func:`_density_balanced_sample`: the fit sample is
    drawn evenly across feature space so minor phases are represented).

    ``seed_means``: optional ``(n_seed, n_features)`` array of NORMALISED
    feature vectors (the same space as ``X_norm``) for phases the unseeded
    mixture is known to merge into a host (a minor phase whose boundary
    pixels grade into its host, e.g. the sillimanite cores of muscovite
    nodules). After the unseeded fit, one extra component per seed is added
    with its mean initialised at the seed (PCA-projected) and EM is re-run
    from that initialisation; the seeds only set the starting point, the
    data decide where the components end up.

    Auto-k uses ICL (BIC + classification entropy) on PCA-reduced features.

    ``posterior_min`` > 0 leaves *ambiguous* pixels unclassified (label -1):
    a pixel whose max GMM posterior falls below the threshold sits genuinely
    between two components (grain-boundary mixed signal) and forcing it into
    either cluster creates spurious intermediate domains. ``density_pct`` > 0
    additionally rejects *atypical* pixels whose mixture log-density falls in
    the lowest given percentile — outliers (cracks, epoxy, contamination) that
    resemble no phase; posteriors cannot catch these because they are relative
    (a far outlier still gets a confident posterior for its nearest component).
    Both default to 0 = off, preserving the classic assign-every-pixel result.

    Returns (labels, k_chosen, bic_scores, icl_scores, method_used); labels
    may contain -1 (rejected/unclassified) when either rejection is active.
    """
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
    except ImportError:
        raise RuntimeError(
            "scikit-learn is required for phase segmentation.\n"
            "Install it with:  pip install scikit-learn"
        )

    # ``random_state`` seeds everything stochastic here: the fit sample (and its
    # density-balancing bins) and every mixture initialisation. The default, 42,
    # is the recipe value and reproduces every stored phase map; other values
    # exist to test seed-to-seed stability (a seeded fit is reproducible, which
    # is not the same as robust).
    rng = np.random.default_rng(random_state)

    def _sub(X, n):
        return X[rng.choice(len(X), n, replace=False)] if len(X) > n else X

    if fit_balance == "density" and len(X_norm) > fit_sample:
        X_fit = X_norm[_density_balanced_sample(X_norm, fit_sample, seed=random_state)]
    else:
        X_fit = _sub(X_norm, fit_sample)
    print(
        f"[segment] Clustering: fit_sample={len(X_fit):,} ({fit_balance})  total={len(X_norm):,}",
        flush=True,
    )

    # ---- PCA reduction -------------------------------------------------------
    pca_full = PCA(random_state=42)
    pca_full.fit(X_fit.astype(np.float64))
    cumvar_arr = pca_full.explained_variance_ratio_.cumsum()
    n_pca = int(np.searchsorted(cumvar_arr, pca_variance)) + 1
    n_pca = max(2, min(n_pca, X_fit.shape[1] - 1))
    pca = PCA(n_components=n_pca, random_state=42)
    X_fit_r = pca.fit_transform(X_fit.astype(np.float64)).astype(np.float32)
    X_norm_r = pca.transform(X_norm.astype(np.float64)).astype(np.float32)
    print(
        f"[segment] PCA: {X_fit.shape[1]}D -> {n_pca}D "
        f"({cumvar_arr[n_pca-1]*100:.1f}% variance explained)",
        flush=True,
    )
    # --------------------------------------------------------------------------

    if n_clusters == "auto":
        k_best, bic_scores, icl_scores, best_gmm, method = _select_k_auto(
            X_fit_r, k_max, on_progress, random_state=random_state,
        )
        k_raw = k_best
        k_best = max(k_best, min_k)
        if k_best > k_raw:
            print(f"  -> min_k floor applied: {k_raw} -> {k_best}", flush=True)
            best_gmm = GaussianMixture(
                n_components=k_best, n_init=5, max_iter=200,
                random_state=random_state, reg_covar=1e-4,
            )
            best_gmm.fit(X_fit_r)
    else:
        k_best = int(n_clusters)
        bic_scores = None
        icl_scores = None
        method = None
        best_gmm = GaussianMixture(
            n_components=k_best, n_init=5, max_iter=200,
            random_state=random_state, reg_covar=1e-4,
        )
        best_gmm.fit(X_fit_r)

    if seed_means is not None and len(seed_means):
        seeds_r = pca.transform(np.asarray(seed_means, dtype=np.float64)).astype(np.float32)
        means0 = np.vstack([best_gmm.means_, seeds_r])
        k_seeded = k_best + len(seeds_r)
        w0 = np.concatenate([best_gmm.weights_ * (1.0 - 0.01 * len(seeds_r)),
                             np.full(len(seeds_r), 0.01)])
        seeded = GaussianMixture(
            n_components=k_seeded, n_init=1, max_iter=200, random_state=random_state,
            reg_covar=1e-4, means_init=means0, weights_init=w0 / w0.sum(),
        )
        seeded.fit(X_fit_r)
        d = np.linalg.norm(seeded.means_[k_best:] - seeds_r, axis=1)
        print(f"[segment] Seeded components: {len(seeds_r)} (k {k_best} -> {k_seeded}); "
              f"mean drift from seed {d.round(2).tolist()}; weights {seeded.weights_[k_best:].round(4).tolist()}",
              flush=True)
        best_gmm, k_best = seeded, k_seeded
    if on_progress:
        on_progress(k_max - 1, k_max, f"Labelling all pixels (k={k_best}) ...")
    labels = best_gmm.predict(X_norm_r).astype(np.int32)

    # Optional rejection: leave ambiguous / atypical pixels unclassified (-1)
    # instead of forcing every pixel into a component. Chunked so the
    # (n_pixels x k) responsibility matrix never materialises whole.
    if posterior_min > 0.0 or density_pct > 0.0:
        n = len(X_norm_r)
        maxpost = np.empty(n, dtype=np.float32) if posterior_min > 0.0 else None
        logdens = np.empty(n, dtype=np.float32) if density_pct > 0.0 else None
        chunk = 500_000
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            if maxpost is not None:
                maxpost[sl] = best_gmm.predict_proba(
                    X_norm_r[sl]).max(axis=1)
            if logdens is not None:
                logdens[sl] = best_gmm.score_samples(X_norm_r[sl])
        reject = np.zeros(n, dtype=bool)
        n_amb = n_atyp = 0
        if maxpost is not None:
            amb = maxpost < posterior_min
            n_amb = int(amb.sum())
            reject |= amb
        if logdens is not None:
            thr = float(np.percentile(logdens, density_pct))
            atyp = logdens < thr
            n_atyp = int(atyp.sum())
            reject |= atyp
        n_rej = int(reject.sum())
        labels[reject] = -1
        print(
            f"[segment] Rejection: {n_rej:,}/{n:,} px left unclassified "
            f"({100.0 * n_rej / max(n, 1):.1f}%) — "
            f"ambiguous (max posterior < {posterior_min:g}): {n_amb:,}; "
            f"atypical (log-density below p{density_pct:g}): {n_atyp:,}",
            flush=True,
        )

    return labels, k_best, bic_scores, icl_scores, method


def _fill_unlabeled(
    label_map: np.ndarray,
    protect: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Nearest-neighbour fill of unlabeled (< 0) pixels, in place.

    ``protect`` is a boolean mask of pixels that must STAY unclassified
    (GMM-rejected ambiguous/atypical pixels): they are neither filled nor —
    because the distance transform treats every ``< 0`` pixel as a gap —
    ever used as a fill source. Only the incidental holes (tile overlap
    exclusion, low-count filter, small-region cleanup) are filled.
    """
    unlabeled = label_map < 0
    fill = unlabeled if protect is None else (unlabeled & ~protect)
    if not fill.any() or not (~unlabeled).any():
        return label_map
    from scipy.ndimage import distance_transform_edt
    nearest_idx = distance_transform_edt(
        unlabeled, return_distances=False, return_indices=True,
    )
    label_map[fill] = label_map[tuple(nearest_idx[:, fill])]
    return label_map


def _zero_signal_mask(maps: Dict[str, np.ndarray]) -> np.ndarray:
    """True where NO element carries any signal (raw, pre-smoothing maps).

    These pixels hold no measurement at all — off-sample area, unscanned
    montage margins — so they are not a phase and must be neither clustered
    (Gaussian smoothing bleeds sample counts a few sigma into the void,
    which otherwise makes a spurious edge fringe pass the low-count filter)
    nor nearest-neighbour filled. Judged on the raw maps: after smoothing
    the void is no longer zero.
    """
    total = None
    for arr in maps.values():
        # abs: the nets are signed, and a noise pixel that happens to be
        # negative in every element is a measurement, not a void.
        a = np.abs(np.asarray(arr, dtype=np.float64))
        total = a.copy() if total is None else total + a
    if total is None:
        raise ValueError("no element maps supplied")
    return total <= 0


def _cluster_dominant_elements(
    X_norm: np.ndarray,
    labels: np.ndarray,
    el_names: List[str],
    top_n: int = 3,
) -> Dict[int, str]:
    """Return {cluster_id: 'El1+El2+El3'} based on per-cluster column means.

    Negative labels (rejected/unclassified pixels) are skipped — they are not
    a phase and must not surface as a domain row in the classification dialog.
    """
    unique = sorted(c for c in set(int(v) for v in labels) if c >= 0)
    result: Dict[int, str] = {}
    for cid in unique:
        mask = labels == cid
        if not mask.any():
            result[cid] = "(empty)"
            continue
        means = X_norm[mask].mean(axis=0)
        top_idx = np.argsort(means)[::-1][:top_n]
        result[cid] = "+".join(el_names[i] for i in top_idx)
    return result


def _label_map_to_rgba(label_map: np.ndarray, n_clusters: int) -> np.ndarray:
    """Convert int label map (H,W) to uint8 RGBA (H,W,4). -1 = transparent.

    Used for the matplotlib preview in the classification dialog.
    """
    tab20 = matplotlib.colormaps['tab20']
    H, W = label_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for cid in range(n_clusters):
        mask = label_map == cid
        if mask.any():
            color = np.array(tab20(cid % 20))  # float 0-1
            rgba[mask] = (color * 255).astype(np.uint8)
    return rgba


def _remove_phase_layers(viewer) -> None:
    """Remove all existing phase layers (per-cluster and legacy combined)."""
    to_remove = [lyr for lyr in viewer.layers
                 if lyr.name.startswith(_PHASE_LAYER_PREFIX)
                 or lyr.name == "Phase segmentation"]
    for lyr in to_remove:
        viewer.layers.remove(lyr)


def _add_phase_cluster_layers(
    viewer,
    label_map: np.ndarray,
    phase_names: Dict[int, str],
    translate: Tuple[float, float],
    opacity: float = 0.7,
    scale: Tuple[float, float] = (1.0, 1.0),
) -> None:
    """Add one napari layer per phase cluster so each can be toggled individually.

    ``scale`` upsamples a decimated phase map (segmented at 1/ds to fit RAM) back
    to mosaic resolution so it overlays the BSE correctly; ``translate`` stays in
    full-resolution mosaic pixels.
    """
    _remove_phase_layers(viewer)
    tab20 = matplotlib.colormaps['tab20']
    H, W = label_map.shape
    ty, tx = translate

    cluster_ids = sorted(set(int(v) for v in np.unique(label_map) if v >= 0))
    for cid in cluster_ids:
        mask = label_map == cid
        if not mask.any():
            continue
        color_f = np.array(tab20(cid % 20))
        color_u8 = (color_f * 255).astype(np.uint8)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[mask] = color_u8

        name = _PHASE_LAYER_PREFIX + phase_names.get(cid, f"Cluster {cid}")
        viewer.add_image(
            rgba, name=name, rgb=True,
            blending='translucent_no_depth', opacity=opacity,
            scale=scale, translate=(ty, tx), visible=True,
        )
