# eds_quantify.py
"""
SPX-anchored empirical calibration of EDS net-count maps.

EDS net-count maps are uncalibrated: per-element net counts are proportional to
X-ray *intensity*, not *mass*. Each element has a different generation+detection
efficiency (the Z-bias) plus matrix absorption/fluorescence, so treating count
fractions as wt% gives non-physical bulk compositions (e.g. SiO2 ~2 wt%). Bruker
SPX files, by contrast, are matrix-corrected quantified wt% - but only as a bulk
value per tile/ROI, never per-pixel.

For Bruker mosaics we hold *both*, spatially registered: per-pixel EDS net-count
maps and per-tile SPX wt%. This module self-calibrates from that pairing - it
regresses integrated EDS net-counts against SPX wt% to recover per-element
empirical sensitivity factors (working-curve slopes that bundle the detector
efficiency, the beam conditions and the average ZAF correction carried by the
SPX quantification; NOT thin-film Cliff-Lorimer factors, which neglect
absorption and fluorescence), then applies them to the count maps to produce
quasi-quantified wt% everywhere, including the sub-tile scales where SPX gives
nothing.

Model (per element i, linear through the origin):

    wt%_i ~= k_i * I_i

with ``I_i`` the integrated net-count for element i over a tile. The no-intercept
least-squares slope over the SPX-covered tiles t is

    k_i = sum_t (I_{i,t} * W_{i,t}) / sum_t (I_{i,t}^2)

Because SPX is already matrix-corrected, the fit absorbs the *average* A/F
correction into k_i. Applying k and renormalising a window/pixel to 100 wt%
yields quasi-quantified composition.

LIMITATIONS (documented, not bugs):
- Captures only the average matrix correction baked into SPX, not per-pixel
  matrix variation (a true ZAF/phi-rho-z would, but needs take-off angle/beam
  geometry the data lacks, plus full spectra rather than 3-band sums). This is
  the accepted second-order simplification - surfaced downstream as a
  ``calibrated_avg_matrix`` caveat.
- k quality degrades for elements anchored by few tiles or low counts (reported
  via ``n_tiles`` / ``r2`` on each KFactor).
- Calibrated wt% omits SPX-only / EDS-only elements (no anchor or no signal),
  so renormalised majors are slightly inflated.
- Needs SPX present (Bruker). Oxford CSV is already wt%; Oxford h5oina counts
  have no SPX anchor and stay ratio-only.

The module is Qt-free and numpy-only so it can be unit-tested headless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# Mirrors mineral_id's anion/coating split. ANION_ELEMENTS: O is the anion
# (added by stoichiometry, not measured as a cation by SPX) — always dropped, and
# the drop is load-bearing here (unlike mineral_id, nothing downstream re-drops O).
# DEFAULT_COATING_ELEMENTS: the conductive coating (carbon by default; may be
# Pt/Au/… or none) — configurable per dataset. Callers that know the coating pass
# ``drop=ANION_ELEMENTS | coating`` to derive_k_factors / cross_validate_k_factors.
# DROP_ELEMENTS is the union kept as the backward-compatible default.
ANION_ELEMENTS = frozenset({"O"})
DEFAULT_COATING_ELEMENTS = frozenset({"C"})
DROP_ELEMENTS = ANION_ELEMENTS | DEFAULT_COATING_ELEMENTS


@dataclass(frozen=True)
class KFactor:
    """Per-element empirical sensitivity factor ``wt% ~= k * net_count``."""
    k: float
    n_tiles: int          # number of valid (I>0, W>0) tile pairs used in the fit
    r2: Optional[float]   # coefficient of determination; None when n_tiles < 2
    sum_counts: float     # sum_t I_{i,t}, diagnostic
    sum_wt: float         # sum_t W_{i,t}, diagnostic


@dataclass(frozen=True)
class Calibration:
    """Result of :func:`derive_k_factors` - the per-element k-factors + diagnostics."""
    k_factors: Dict[str, KFactor]
    n_tiles: int               # number of paired tiles considered
    elements: List[str]        # sorted calibrated element set (got a k-factor)
    spx_only: List[str]        # in SPX but never anchored to EDS counts (info)
    eds_only: List[str]        # in EDS counts but never in SPX (dropped, info)
    # Background scheme the anchoring net counts were computed with
    # (``eds_background.SCHEMES``). k-factors are only valid for nets of the
    # same scheme: the legacy "sideband" nets are ~2x smaller for Al/Si.
    background: str = "line-aware"

    def k(self) -> Dict[str, float]:
        """Plain ``{el: k}`` mapping for the apply_* functions / workers."""
        return {el: kf.k for el, kf in self.k_factors.items()}

    def to_dict(self) -> dict:
        """JSON-friendly round-trip representation (for session persistence)."""
        return {
            "k_factors": {
                el: {"k": kf.k, "n_tiles": kf.n_tiles, "r2": kf.r2,
                     "sum_counts": kf.sum_counts, "sum_wt": kf.sum_wt}
                for el, kf in self.k_factors.items()
            },
            "n_tiles": self.n_tiles,
            "elements": list(self.elements),
            "spx_only": list(self.spx_only),
            "eds_only": list(self.eds_only),
            "background": self.background,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        kfs = {}
        for el, kd in (d.get("k_factors") or {}).items():
            r2 = kd.get("r2", None)
            kfs[el] = KFactor(
                k=float(kd["k"]),
                n_tiles=int(kd.get("n_tiles", 0)),
                r2=(None if r2 is None else float(r2)),
                sum_counts=float(kd.get("sum_counts", 0.0)),
                sum_wt=float(kd.get("sum_wt", 0.0)),
            )
        return cls(
            k_factors=kfs,
            n_tiles=int(d.get("n_tiles", 0)),
            elements=list(d.get("elements") or sorted(kfs)),
            spx_only=list(d.get("spx_only") or []),
            eds_only=list(d.get("eds_only") or []),
            # Sessions saved before 2026-09-16 were anchored on sideband nets.
            background=str(d.get("background") or "sideband"),
        )


def _present_elements(rows: Sequence[Dict[str, float]], drop: frozenset) -> set:
    """Elements that appear with a strictly positive value in at least one row."""
    out = set()
    for row in rows:
        if not row:
            continue
        for el, v in row.items():
            if el in drop:
                continue
            try:
                if float(v) > 0.0:
                    out.add(el)
            except (TypeError, ValueError):
                continue
    return out


def derive_k_factors(spx_by_tile: Sequence[Dict[str, float]],
                     counts_by_tile: Sequence[Dict[str, float]],
                     drop: frozenset = DROP_ELEMENTS,
                     background: Optional[str] = None) -> Calibration:
    """Derive per-element k-factors from paired (SPX wt%, EDS net-count) tiles.

    ``spx_by_tile[t]`` is ``{el: wt%}`` and ``counts_by_tile[t]`` is
    ``{el: raw_integrated_net_count}`` for the same tile t (the two sequences
    are paired positionally and must be the same length). An element contributes
    a tile only where it has ``I > 0`` *and* ``W > 0`` there. Elements valid in a
    single tile get ``k = W / I`` with ``r2 = None`` (no scatter estimate).

    Elements present only in SPX (no EDS anchor) land in ``spx_only``; elements
    present only in EDS (no SPX reference) land in ``eds_only``; neither is
    calibrated. Returns a :class:`Calibration`; ``k_factors`` is empty when no
    element can be anchored.
    """
    n = min(len(spx_by_tile), len(counts_by_tile))
    spx_by_tile = list(spx_by_tile[:n])
    counts_by_tile = list(counts_by_tile[:n])

    spx_elems = _present_elements(spx_by_tile, drop)
    eds_elems = _present_elements(counts_by_tile, drop)
    shared = sorted(spx_elems & eds_elems)

    k_factors: Dict[str, KFactor] = {}
    for el in shared:
        I_vals: List[float] = []
        W_vals: List[float] = []
        for spx, cnt in zip(spx_by_tile, counts_by_tile):
            if not spx or not cnt:
                continue
            try:
                I = float(cnt.get(el, 0.0))
                W = float(spx.get(el, 0.0))
            except (TypeError, ValueError):
                continue
            if I > 0.0 and W > 0.0:
                I_vals.append(I)
                W_vals.append(W)
        if not I_vals:
            continue
        I_arr = np.asarray(I_vals, dtype=float)
        W_arr = np.asarray(W_vals, dtype=float)
        sse_x = float(np.sum(I_arr * I_arr))
        if sse_x <= 0.0:
            continue
        k = float(np.sum(I_arr * W_arr) / sse_x)
        n_t = int(I_arr.size)

        if n_t >= 2:
            resid = W_arr - k * I_arr
            ss_res = float(np.sum(resid * resid))
            ss_tot = float(np.sum((W_arr - W_arr.mean()) ** 2))
            r2: Optional[float] = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None
        else:
            r2 = None

        k_factors[el] = KFactor(
            k=k, n_tiles=n_t, r2=r2,
            sum_counts=float(I_arr.sum()), sum_wt=float(W_arr.sum()),
        )

    calibrated = set(k_factors)
    # SPX elements that never got a k-factor: either SPX-only (no EDS signal) or
    # shared-but-never-co-positive-in-one-tile. Both mean "couldn't be anchored".
    spx_only = sorted(spx_elems - calibrated)
    # EDS elements with no SPX reference at all.
    eds_only = sorted(eds_elems - spx_elems)
    if background is None:
        from eds_background import default_scheme
        background = default_scheme()
    return Calibration(
        k_factors=k_factors,
        n_tiles=n,
        elements=sorted(calibrated),
        spx_only=spx_only,
        eds_only=eds_only,
        background=background,
    )


def filter_calibration_tiles(spx_by_tile: Sequence[Dict[str, float]],
                             counts_by_tile: Sequence[Dict[str, float]],
                             *,
                             weights: Optional[Sequence[float]] = None,
                             valid_fracs: Optional[Sequence[float]] = None,
                             min_total_counts: float = 1e-9,
                             min_valid_frac: float = 0.5,
                             min_weight: float = 0.0):
    """Drop spurious tiles before fitting k-factors.

    Spurious tiles — empty/near-empty EDS, holes in the sample, very low count
    rate — corrupt the k-factor fit, not just its validation. This conservative,
    paired filter removes a tile when it has no usable SPX/EDS, a near-zero total
    net signal (``< min_total_counts``), a coverage fraction below
    ``min_valid_frac`` (when ``valid_fracs`` is supplied — the hole guard), or a
    near-zero SPX weight (when ``weights`` + ``min_weight`` are supplied).

    Defaults are deliberately permissive (only truly-empty tiles dropped, the
    coverage/weight guards off unless the caller passes the per-tile metadata and
    a threshold) so nothing borderline-good is silently removed; callers tune the
    thresholds. Inputs are paired positionally with ``weights``/``valid_fracs``.

    Returns ``(kept_spx, kept_counts, dropped)`` where ``dropped`` is a list of
    ``{"index": t, "reasons": [...]}`` for reporting.
    """
    n = min(len(spx_by_tile), len(counts_by_tile))
    kept_spx: List[Dict[str, float]] = []
    kept_counts: List[Dict[str, float]] = []
    dropped: List[dict] = []
    for t in range(n):
        s, c = spx_by_tile[t], counts_by_tile[t]
        reasons: List[str] = []
        if not s:
            reasons.append("no_spx")
        if not c:
            reasons.append("no_counts")
        else:
            tot = 0.0
            for v in c.values():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv > 0.0:
                    tot += fv
            if tot < min_total_counts:
                reasons.append("low_counts")
        if valid_fracs is not None and t < len(valid_fracs):
            vf = valid_fracs[t]
            if vf is not None and float(vf) < min_valid_frac:
                reasons.append("low_valid_frac")
        if weights is not None and min_weight > 0.0 and t < len(weights):
            w = weights[t]
            if w is not None and float(w) < min_weight:
                reasons.append("low_weight")
        if reasons:
            dropped.append({"index": t, "reasons": reasons})
        else:
            kept_spx.append(s)
            kept_counts.append(c)
    return kept_spx, kept_counts, dropped


@dataclass(frozen=True)
class CVElementStats:
    """Hold-out prediction quality for one element (wt% points)."""
    n_tiles: int          # held-out predictions for this element (tile x fold)
    bias: float           # mean(predicted - measured); signed systematic error
    rmse: float           # root-mean-square of (predicted - measured), pooled
    mean_measured: float  # mean held-out SPX wt%, for relative-error context
    rmse_std: float = 0.0  # spread of the per-fold RMSE (the error bar)


@dataclass(frozen=True)
class CrossValidation:
    """Hold-out validation of the k-factor calibration.

    Two schemes (``method``):

    - ``"holdout"`` (default): repeated random leave-a-proportion-out. Each of
      ``n_folds`` seeded folds holds out ``ceil(holdout_frac*N)`` tiles, fits the
      k-factors on the rest, and predicts the held-out tiles. Pooling over folds
      gives a robust mean RMSE/bias plus a per-fold spread (``rmse_std``) — and a
      single spurious tile cannot dominate one unlucky residual.
    - ``"loo"``: exhaustive leave-one-tile-out (closed form; see
      :func:`cross_validate_k_factors`).

    In both, the held-out tile's bulk is predicted from its EDS counts alone via
    :func:`apply_calibration_fractions`; predicted and measured (SPX) are
    renormalised to 100 over their shared element set so residuals are
    like-for-like wt% points.
    """
    n_tiles: int                            # held-out predictions pooled (tile x fold)
    elements: List[str]
    per_element: Dict[str, CVElementStats]
    records: List[dict]                     # {tile, fold, predicted, measured}
    rmse: float                             # overall pooled RMSE
    method: str = "holdout"
    n_folds: int = 1                        # folds with a usable prediction
    rmse_std: float = 0.0                   # spread of per-fold overall RMSE

    def to_dict(self) -> dict:
        return {
            "n_tiles": self.n_tiles,
            "elements": list(self.elements),
            "per_element": {
                el: {"n_tiles": s.n_tiles, "bias": s.bias, "rmse": s.rmse,
                     "mean_measured": s.mean_measured, "rmse_std": s.rmse_std}
                for el, s in self.per_element.items()
            },
            "records": self.records,
            "rmse": self.rmse,
            "method": self.method,
            "n_folds": self.n_folds,
            "rmse_std": self.rmse_std,
        }


def _cv_member_sums(spx: List[Dict[str, float]], cnt: List[Dict[str, float]],
                    drop: frozenset):
    """Per-element member contributions for closed-form re-fitting.

    Mirrors :func:`derive_k_factors` membership (element not dropped, ``I > 0``
    and ``W > 0`` in that tile). Returns ``(members, totals)`` where
    ``members[el] = {tile_idx: (I, W)}`` and ``totals[el] = [ΣIW, ΣI²]`` over all
    member tiles — so a fit on any train subset is ``total minus the held-out
    members``, identical to re-running ``derive_k_factors`` on that subset.
    """
    spx_elems = _present_elements(spx, drop)
    eds_elems = _present_elements(cnt, drop)
    shared = spx_elems & eds_elems
    members: Dict[str, Dict[int, tuple]] = {el: {} for el in shared}
    totals: Dict[str, list] = {el: [0.0, 0.0] for el in shared}
    for t in range(len(spx)):
        s, c = spx[t], cnt[t]
        if not s or not c:
            continue
        for el in shared:
            try:
                I = float(c.get(el, 0.0))
                W = float(s.get(el, 0.0))
            except (TypeError, ValueError):
                continue
            if I > 0.0 and W > 0.0:
                members[el][t] = (I, W)
                totals[el][0] += I * W
                totals[el][1] += I * I
    return members, totals


def _fit_k_excluding(members: Dict[str, Dict[int, tuple]],
                     totals: Dict[str, list], test_set: set) -> Dict[str, float]:
    """k-factors fit on every tile *except* ``test_set`` (closed form)."""
    k: Dict[str, float] = {}
    for el, mem in members.items():
        IW, II = totals[el]
        for t in test_set:
            hit = mem.get(t)
            if hit is not None:
                I, W = hit
                IW -= I * W
                II -= I * I
        if II > 0.0:
            k[el] = IW / II
    return k


def cross_validate_k_factors(spx_by_tile: Sequence[Dict[str, float]],
                             counts_by_tile: Sequence[Dict[str, float]],
                             drop: frozenset = DROP_ELEMENTS,
                             *,
                             method: str = "holdout",
                             holdout_frac: float = 0.2,
                             n_repeats: int = 50,
                             seed: int = 0,
                             ) -> Optional[CrossValidation]:
    """Hold-out cross-validation of :func:`derive_k_factors`.

    Same paired inputs as :func:`derive_k_factors`.

    ``method="holdout"`` (default): repeated random leave-a-proportion-out —
    ``n_repeats`` seeded folds each hold out ``ceil(holdout_frac*N)`` tiles, fit
    on the rest, and predict the held-out tiles. Robust to a few spurious tiles
    and yields an error bar (``rmse_std``). ``method="loo"``: exhaustive
    leave-one-tile-out. Both use the same closed-form re-fit (per-element
    ``k₋S = (ΣIW − Σ_S IW)/(ΣI² − Σ_S I²)``), numerically identical to re-running
    ``derive_k_factors`` on the train subset, so LOO needs only one O(N·E) pass.

    With too few tiles for a meaningful proportion split, ``"holdout"`` auto-falls
    back to ``"loo"``. Returns ``None`` when fewer than 2 held-out predictions are
    usable (no scatter to estimate).
    """
    n = min(len(spx_by_tile), len(counts_by_tile))
    spx = list(spx_by_tile[:n])
    cnt = list(counts_by_tile[:n])
    if n < 2:
        return None

    method = (method or "holdout").lower()
    if method not in ("holdout", "loo"):
        method = "holdout"
    if method == "holdout":
        # Need enough tiles that each fold's train set can still anchor elements.
        min_for_holdout = int(np.ceil(1.0 / max(holdout_frac, 1e-9))) + 1
        if n < max(4, min_for_holdout):
            method = "loo"

    members, totals = _cv_member_sums(spx, cnt, drop)

    if method == "loo":
        folds: List[List[int]] = [[t] for t in range(n)]
    else:
        rng = np.random.default_rng(seed)
        k_out = max(1, int(np.ceil(holdout_frac * n)))
        k_out = min(k_out, n - 1)  # always leave ≥1 tile to fit on
        folds = [sorted(rng.choice(n, size=k_out, replace=False).tolist())
                 for _ in range(max(1, n_repeats))]

    records: List[dict] = []
    residuals: Dict[str, List[float]] = {}
    measured_acc: Dict[str, List[float]] = {}
    per_el_fold_rmse: Dict[str, List[float]] = {}
    fold_rmses: List[float] = []

    for fi, test in enumerate(folds):
        k_train = _fit_k_excluding(members, totals, set(test))
        if not k_train:
            continue
        fold_res: List[float] = []
        el_fold_res: Dict[str, List[float]] = {}
        for t in test:
            spx_t, cnt_t = spx[t], cnt[t]
            if not spx_t or not cnt_t:
                continue
            pred = apply_calibration_fractions(cnt_t, k_train)
            if not pred:
                continue
            meas: Dict[str, float] = {}
            for el, v in spx_t.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if el in pred and fv > 0.0:
                    meas[el] = fv
            shared = sorted(set(pred) & set(meas))
            if len(shared) < 2:
                continue
            p_tot = sum(pred[el] for el in shared)
            m_tot = sum(meas[el] for el in shared)
            if p_tot <= 0.0 or m_tot <= 0.0:
                continue
            pred_n = {el: 100.0 * pred[el] / p_tot for el in shared}
            meas_n = {el: 100.0 * meas[el] / m_tot for el in shared}
            records.append({"tile": t, "fold": fi,
                            "predicted": pred_n, "measured": meas_n})
            for el in shared:
                r = pred_n[el] - meas_n[el]
                residuals.setdefault(el, []).append(r)
                measured_acc.setdefault(el, []).append(meas_n[el])
                fold_res.append(r)
                el_fold_res.setdefault(el, []).append(r)
        if fold_res:
            fold_rmses.append(float(np.sqrt(np.mean(np.square(fold_res)))))
            for el, rs in el_fold_res.items():
                per_el_fold_rmse.setdefault(el, []).append(
                    float(np.sqrt(np.mean(np.square(rs)))))

    if len(records) < 2:
        return None

    per_element: Dict[str, CVElementStats] = {}
    all_res: List[float] = []
    for el, rs in residuals.items():
        arr = np.asarray(rs, dtype=float)
        fold_rmse = per_el_fold_rmse.get(el, [])
        per_element[el] = CVElementStats(
            n_tiles=int(arr.size),
            bias=float(arr.mean()),
            rmse=float(np.sqrt(np.mean(arr * arr))),
            mean_measured=float(np.mean(measured_acc[el])),
            rmse_std=float(np.std(fold_rmse)) if len(fold_rmse) > 1 else 0.0,
        )
        all_res.extend(rs)
    overall = float(np.sqrt(np.mean(np.square(np.asarray(all_res, dtype=float)))))
    return CrossValidation(
        n_tiles=len(records),
        elements=sorted(per_element),
        per_element=per_element,
        records=records,
        rmse=overall,
        method=method,
        n_folds=len(fold_rmses),
        rmse_std=float(np.std(fold_rmses)) if len(fold_rmses) > 1 else 0.0,
    )


def apply_calibration_fractions(net_fractions: Dict[str, float],
                                k_factors: Dict[str, float]) -> Dict[str, float]:
    """Convert per-element relative intensities to calibrated wt% (sum=100).

    ``net_fractions`` may be normalised fractions (sum~1) *or* raw net-counts -
    the per-pixel/window renormalisation makes the input scale irrelevant. Only
    elements present in both ``net_fractions`` and ``k_factors`` survive; the
    rest are dropped. Returns ``{}`` when there is no overlap or the calibrated
    total is non-positive.
    """
    if not net_fractions or not k_factors:
        return {}
    raw: Dict[str, float] = {}
    for el, frac in net_fractions.items():
        k = k_factors.get(el)
        if k is None:
            continue
        try:
            val = float(k) * float(frac)
        except (TypeError, ValueError):
            continue
        if val > 0.0:
            raw[el] = val
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {el: 100.0 * v / total for el, v in raw.items()}


def apply_calibration_maps(band_maps: Dict[str, Dict[str, np.ndarray]],
                           k_factors: Dict[str, float],
                           sub_bbox: Optional[tuple] = None,
                           sub_mask: Optional[np.ndarray] = None,
                           ) -> Dict[str, np.ndarray]:
    """Per-pixel calibrated wt% maps from net-count band maps and k-factors.

    For each calibrated element (intersection of ``band_maps`` and
    ``k_factors``): the signed net (``eds_background.net_count_map``, line-aware
    by default, plan from the dict's own elements) over the optional
    sub-window, clipped at zero PER PIXEL because a per-pixel composition
    cannot be negative (so these maps keep the rectified-noise bias of a
    low-dose map; smooth before reading ratios, as ``zoning`` does),
    multiplied by ``k``, then renormalised across the calibrated
    element axis per pixel so each pixel's wt% sums to 100 (pixels with zero
    total signal stay 0). Returns ``{el: HxW float32}`` for calibrated elements
    only. ``{}`` when no element overlaps.
    """
    if not band_maps or not k_factors:
        return {}
    elems = [el for el in band_maps if el in k_factors]
    if not elems:
        return {}

    sample = band_maps[elems[0]]['main']
    H, W = sample.shape[:2]
    if sub_bbox is None:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = sub_bbox
        y0 = max(0, int(y0)); y1 = min(H, int(y1))
        x0 = max(0, int(x0)); x1 = min(W, int(x1))
    if y1 <= y0 or x1 <= x0:
        return {}

    m = None
    if sub_mask is not None:
        m = np.asarray(sub_mask).astype(np.float32)

    from eds_background import net_count_maps
    nets = net_count_maps(band_maps, elements=elems, clip=True,
                          sub_bbox=(y0, y1, x0, x1))
    cal: Dict[str, np.ndarray] = {}
    total = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for el in elems:
        net = nets[el].astype(np.float32)
        if m is not None:
            net = net * m
        c = net * float(k_factors[el])
        cal[el] = c
        total += c

    safe = total > 0.0
    out: Dict[str, np.ndarray] = {}
    for el in elems:
        wt = np.zeros_like(total)
        np.divide(100.0 * cal[el], total, out=wt, where=safe)
        out[el] = wt.astype(np.float32)
    return out
