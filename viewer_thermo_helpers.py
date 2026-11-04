# viewer_thermo_helpers.py
"""
Thermodynamic-modelling helpers for viewer.py:
- Pseudosection / phase-field utilities (`_make_phase_field_map`,
  `_pt_estimate_from_results`).
- Assemblage matching + modal misfit (`PHASE_ALIASES`, `_phase_matches`,
  `_observed_modes_from_phase_map`, `_modal_misfit_grid`).
- Multi-scale composition helpers (`_net_count_fractions`,
  `_max_useful_quadtree_depth`, `_quadtree_windows`).
- Spatial-structure metrics (`_semivariogram`, `_morans_i`,
  `_eligible_variogram_levels`, `_usable_windows_at_level`).
- Run-record serialisation (`_serialize_phase_result`,
  `_deserialize_phase_result`, `_serialize_thermod_result`,
  `_deserialize_thermod_result`).
- Display helpers (`_format_composition_lines`,
  `_thermod_parameters_text`, `_export_thermod_bundle`).
- Inline assemblage-selector dialog (`_assemblage_selector_dialog`).

Moved out of viewer.py as part of the tier-B refactor.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable, Union

import numpy as np
from qtpy import QtWidgets, QtCore


def _make_phase_field_map(results, n_p: int, n_t: int):
    """Assign a colour index to each P-T grid cell based on its unique stable assemblage.

    Returns
    -------
    field_idx : np.ndarray shape (n_p, n_t), int
    ranked    : list of (frozenset[str], count) sorted by frequency descending
    """
    from collections import Counter
    assem_list = [frozenset(ph.name for ph in r.stable_phases) for r in results]
    counts = Counter(assem_list)
    ranked = counts.most_common()
    assem_to_idx = {fs: i for i, (fs, _) in enumerate(ranked)}
    field_idx = np.array([assem_to_idx[a] for a in assem_list], dtype=int)
    return field_idx.reshape(n_p, n_t), ranked


# -------------------------
# Phase-name matching (observed assemblage vs backend phase names)
# -------------------------
# The mineral-ID assemblage speaks in thermo keys ("g", "pl", "cpx" — the
# values of mineral_id.THERMO_ABBREV), while backends spell the same phase as
# "Gt(W)" / "Grt" (Perple_X solution models) or "g" / "fsp" (MAGEMin), and
# users / EBSD legends may type full names ("Garnet", "Diopside"). Each entry
# below groups every spelling of one phase; matching is by normalised-token
# membership in the same group (plus plain exact equality), never by prefix —
# so "sp" (spinel) can't accidentally match "sph" (sphene).
# "fsp"/"feldspar" appear under BOTH pl and kf on purpose: a ternary-feldspar
# solution satisfies either observation.
PHASE_ALIASES: Dict[str, frozenset] = {
    "q":    frozenset({"q", "qz", "qtz", "quartz", "coe", "coesite"}),
    "pl":   frozenset({"pl", "plg", "plag", "plagioclase", "fsp", "feldspar",
                       "pl4t", "pl4tr", "afs", "ab", "an", "albite", "anorthite"}),
    "kf":   frozenset({"kf", "kfs", "kspar", "k-feldspar", "alkali feldspar",
                       "fsp", "feldspar", "afs", "san", "sanidine", "mic",
                       "microcline", "or", "orthoclase"}),
    "ne":   frozenset({"ne", "neph", "nepheline"}),
    "ol":   frozenset({"ol", "o", "oliv", "olivine", "fo", "fa",
                       "forsterite", "fayalite"}),
    "opx":  frozenset({"opx", "orthopyroxene", "en", "fs", "enstatite",
                       "ferrosilite"}),
    "cpx":  frozenset({"cpx", "clinopyroxene", "di", "hed", "aug", "augite",
                       "omph", "omphacite", "jd", "jadeite", "diopside", "acm",
                       "pig", "pigeonite"}),
    "g":    frozenset({"g", "gt", "grt", "gar", "garnet", "alm", "py", "gr",
                       "sps", "andr", "maj", "almandine", "pyrope",
                       "grossular", "spessartine", "andradite"}),
    "mu":   frozenset({"mu", "ms", "mus", "muscovite", "mica", "wm", "pheng",
                       "phengite", "cel", "pa", "paragonite", "white mica"}),
    "bi":   frozenset({"bi", "bio", "bt", "biotite", "phl", "phlogopite", "ann",
                       "annite"}),
    "amph": frozenset({"amph", "amp", "camph", "oamp", "oamph", "hb", "hbl",
                       "hornblende", "amphibole", "act", "actinolite", "tr",
                       "tremolite", "gl", "glaucophane", "anth", "cumm", "gru"}),
    "chl":  frozenset({"chl", "chlorite", "clin", "daph", "ames",
                       "clinochlore"}),
    "ep":   frozenset({"ep", "epidote", "zo", "zoisite", "cz", "clinozoisite"}),
    # Al2SiO5: a generic "als" observation matches ANY polymorph (EDS can't
    # distinguish them), but a specific polymorph key (ky/sill/and — set by the
    # user via the Override / add-phase list) matches ONLY that polymorph, so the
    # choice actually changes the modelled stability field.
    "ky":   frozenset({"ky", "kyanite"}),
    "sill": frozenset({"sill", "sil", "sillimanite"}),
    "and":  frozenset({"and", "andalusite"}),
    "als":  frozenset({"als", "al2sio5", "aluminosilicate", "ky", "kyanite",
                       "sill", "sil", "sillimanite", "and", "andalusite"}),
    "cd":   frozenset({"cd", "crd", "hcrd", "cord", "cordierite"}),
    "mt":   frozenset({"mt", "mag", "magnetite", "mtul", "usp", "ulvospinel"}),
    "ilm":  frozenset({"ilm", "ilmm", "ilhm", "ilmenite"}),
    "ru":   frozenset({"ru", "rt", "rut", "rutile"}),
    "hem":  frozenset({"hem", "hematite", "ilhm"}),
    "sp":   frozenset({"sp", "spl", "spn", "spinel", "spi", "chr", "cm",
                       "chromite", "herc", "hercynite"}),
    "cc":   frozenset({"cc", "cal", "calcite", "carb", "arag", "aragonite"}),
    "do":   frozenset({"do", "dol", "dolomite", "ank", "ankerite"}),
    "ap":   frozenset({"ap", "apatite", "fap"}),
    "sph":  frozenset({"sph", "ttn", "sphene", "titanite"}),
    "zrc":  frozenset({"zrc", "zr", "zrn", "zircon"}),
}

# Free-fluid phases: their modal fraction isn't preserved in the rock, so they
# are excluded from the predicted modal vector (melt is excluded via is_melt).
_FLUID_TOKENS = frozenset({"h2o", "fl", "fluid", "f", "aq", "water", "steam",
                           "co2", "vapor"})


def _normalize_phase_token(name: str) -> str:
    """Lowercase a phase name and strip solution-model decoration.

    ``"Gt(W)"`` → ``"gt"``, ``"melt(G)"`` → ``"melt"``, ``" Cpx(JH) "`` → ``"cpx"``.
    """
    s = str(name).strip().lower()
    i = s.find("(")
    if i > 0:
        s = s[:i]
    return s.strip().strip("_-")


def _phase_matches(observed: str, backend_name: str) -> bool:
    """Alias-aware test that *backend_name* is the phase the user observed.

    Exact normalised equality always matches (so hand-typed backend names keep
    working); otherwise both tokens must sit in the same ``PHASE_ALIASES``
    group. The observed side may be a canonical thermo key ("g"), any alias
    ("Grt", "Garnet"), or a raw backend spelling.
    """
    obs_tok = _normalize_phase_token(observed)
    name_tok = _normalize_phase_token(backend_name)
    if not obs_tok or not name_tok:
        return False
    if obs_tok == name_tok:
        return True
    aliases = PHASE_ALIASES.get(obs_tok)
    if aliases is not None:
        return name_tok in aliases
    for group in PHASE_ALIASES.values():
        if obs_tok in group and name_tok in group:
            return True
    return False


def _is_fluid_phase(name: str) -> bool:
    return _normalize_phase_token(name) in _FLUID_TOKENS


def _pt_estimate_from_results(results, observed_phases: List[str], n_p: int,
                              n_t: int, exact: bool = False) -> np.ndarray:
    """Return bool mask (n_p x n_t) where all observed phases are stable.

    By default matching is alias-aware (:func:`_phase_matches`), so mineral-ID
    thermo keys ("g", "pl") and full names ("Garnet") match backend spellings
    ("Gt(W)", "fsp") — correct when the observed assemblage is in a *foreign*
    vocabulary (mineral-ID / EBSD names).

    When *exact* is True, phases are matched only by case-insensitive name
    equality, with no alias broadening. Use this when ``observed_phases`` are
    drawn from the grid's *own* phase names (e.g. the Select-assemblage dialog):
    alias matching is strictly broader than exact equality (it can add cells but
    never remove them), so it would outline a superset of the true stability
    field — e.g. selecting plagioclase would also light up cells where only
    alkali feldspar is stable, because both share ``fsp``/``afs`` in the alias
    table. Exact matching makes the outline agree with the per-phase 'highlight
    phases' overlays and the mol-fraction property maps.
    """
    if exact:
        obs_norm = [str(o).strip().lower() for o in observed_phases]
        compatible = []
        for r in results:
            names = {ph.name.strip().lower() for ph in r.stable_phases}
            compatible.append(all(o in names for o in obs_norm))
        return np.array(compatible, dtype=bool).reshape(n_p, n_t)
    compatible = []
    for r in results:
        names = [ph.name for ph in r.stable_phases]
        compatible.append(all(any(_phase_matches(obs, nm) for nm in names)
                              for obs in observed_phases))
    return np.array(compatible, dtype=bool).reshape(n_p, n_t)


def _observed_modes_from_phase_map(phase_map: np.ndarray,
                                   key_by_cid: Dict[int, str],
                                   unmix: Optional[Dict[int, Dict[int, float]]] = None,
                                   ) -> Tuple[Dict[str, float], float]:
    """Observed modal fractions per phase from a segmentation label map.

    Pixel counts are areal fractions, which estimate volume fractions
    (standard stereology) — directly comparable to the backends'
    ``vol_fraction``. ``key_by_cid`` maps each domain to its observed phase key
    (mineral-ID ``thermo_abbrev`` or a display name); domains mapping to the
    same key merge. Domains with a missing/empty key and unsegmented pixels
    (cid < 0) are excluded.

    ``unmix`` (optional) is ``{source_cid: {endmember_cid: fraction}}`` from
    :func:`_unmix_domain`: a source domain known to be a sub-pixel *mixture*
    of other phases has its pixel count redistributed onto the endmember
    domains in proportion to the (renormalised) fractions, instead of being
    counted under its own key — the resolution-bias correction for
    fine-grained matrix domains. An unmix entry whose fractions are empty is
    ignored (the source keeps its own key).

    Returns ``({key: fraction}, excluded_fraction)`` where the fractions sum to
    1 over the keyed domains and ``excluded_fraction`` is the share of
    *segmented* pixels that had no key. ``({}, 1.0)`` when nothing is keyed.
    """
    pm = np.asarray(phase_map)
    cids, ncounts = np.unique(pm[pm >= 0], return_counts=True)
    counts_by_cid = {int(c): float(n) for c, n in zip(cids, ncounts)}
    seg_px = sum(counts_by_cid.values())
    if unmix:
        for src, fr in unmix.items():
            src = int(src)
            fr = {int(e): float(v) for e, v in (fr or {}).items()
                  if float(v) > 0 and int(e) != src}
            if src not in counts_by_cid or not fr:
                continue
            n_src = counts_by_cid.pop(src)
            tot_f = sum(fr.values())
            for ecid, f in fr.items():
                counts_by_cid[ecid] = counts_by_cid.get(ecid, 0.0) \
                    + n_src * (f / tot_f)
    counts: Dict[str, float] = {}
    keyed_px = 0.0
    for cid, n in counts_by_cid.items():
        key = (key_by_cid or {}).get(int(cid))
        if not key or not str(key).strip():
            continue
        key = str(key).strip()
        counts[key] = counts.get(key, 0.0) + float(n)
        keyed_px += float(n)
    if keyed_px == 0 or seg_px == 0:
        return {}, 1.0
    modes = {k: v / keyed_px for k, v in counts.items()}
    return modes, 1.0 - keyed_px / seg_px


def _unmix_domain(mean_counts_by_cid: Dict[int, Dict[str, float]],
                  source_cid: int,
                  endmember_cids: List[int],
                  ) -> Tuple[Dict[int, float], float]:
    """Areal decomposition of a mixed segmentation domain by NNLS.

    EDS net counts add linearly over sub-pixel phase fractions, so a mixed
    domain's mean per-pixel count vector is a non-negative linear combination
    of the (clean) endmember domains' mean vectors. The NNLS coefficients are
    the areal fractions of each endmember phase inside the mixed domain,
    under the assumption that each phase emits at the count rate measured in
    its clean domain — which is why the inputs must be *raw mean counts per
    pixel* (``_net_count_sums / n_px``), never normalised fractions.

    ``mean_counts_by_cid``: ``{cid: {element: mean net counts per pixel}}``.
    Elements are aligned over the union of keys (missing = 0).

    Returns ``({endmember_cid: fraction}, relative_residual)``. Fractions are
    NOT renormalised — their sum is diagnostic (well below 1 means the
    endmembers do not explain the source's signal); the relative residual is
    ``||A f − y|| / ||y||`` (a few % or less indicates a good linear mixture).
    Consumers that redistribute pixel counts renormalise
    (:func:`_observed_modes_from_phase_map`).
    """
    from scipy.optimize import nnls
    src = int(source_cid)
    ecids = [int(c) for c in endmember_cids if int(c) != src]
    if not ecids:
        raise ValueError("Need at least one endmember domain besides the source.")
    els = sorted({el for c in [src] + ecids
                  for el in (mean_counts_by_cid.get(c) or {})})
    if not els:
        raise ValueError("No element data in the supplied mean count vectors.")
    A = np.array([[float((mean_counts_by_cid.get(c) or {}).get(el, 0.0))
                   for c in ecids] for el in els], dtype=np.float64)
    y = np.array([float((mean_counts_by_cid.get(src) or {}).get(el, 0.0))
                  for el in els], dtype=np.float64)
    f, rnorm = nnls(A, y)
    denom = float(np.linalg.norm(y))
    rel = float(rnorm) / denom if denom > 0 else float("nan")
    return {c: float(fi) for c, fi in zip(ecids, f)}, rel


def _best_fit_pt(misfit: np.ndarray, p_range, t_range, n_p: int, n_t: int):
    """``(P_GPa, T_C)`` of the minimum-misfit cell of a grid, or ``None``.

    The argmin of the modal-misfit grid is the best-fit P-T estimate for an
    assemblage (robust: always defined when any cell is finite, unlike a strict
    all-phases-stable intersection which can be empty or span many cells).
    """
    misfit = np.asarray(misfit, dtype=float).reshape(n_p, n_t)
    if not np.isfinite(misfit).any():
        return None
    i_p, j_t = np.unravel_index(int(np.nanargmin(misfit)), (n_p, n_t))
    P = float(np.linspace(p_range[0], p_range[1], n_p)[i_p])
    T = float(np.linspace(t_range[0], t_range[1], n_t)[j_t])
    return P, T


def _pt_region_from_misfit(misfit, p_range, t_range, n_p: int, n_t: int,
                           delta: float = 0.02):
    """Near-minimal P-T region of a misfit grid: the cells the data *accept*.

    Returns ``{"pt": (P, T), "p_lo", "p_hi", "t_lo", "t_hi"}`` where ``pt``
    is the argmin cell and the ranges span every cell with
    ``misfit <= min + delta``, or ``None`` when no cell is finite.

    Why a region and not just the argmin: the modal misfit of a
    well-matching assemblage forms a nearly *flat valley* along its
    stability band, and the argmin of a flat function is unstable — small
    bulk perturbations hop it along the valley, so an envelope built from
    argmin points measures valley-hopping noise rather than how well the
    data constrain P-T. The near-minimal region is stable where the misfit
    surface is stable and genuinely widens when the constraint degrades.
    """
    misfit = np.asarray(misfit, dtype=float).reshape(n_p, n_t)
    if not np.isfinite(misfit).any():
        return None
    p_ax = np.linspace(p_range[0], p_range[1], n_p)
    t_ax = np.linspace(t_range[0], t_range[1], n_t)
    i_p, j_t = np.unravel_index(int(np.nanargmin(misfit)), (n_p, n_t))
    with np.errstate(invalid="ignore"):
        sel = misfit <= np.nanmin(misfit) + float(delta)
    ps = p_ax[sel.any(axis=1)]
    ts = t_ax[sel.any(axis=0)]
    return {"pt": (float(p_ax[i_p]), float(t_ax[j_t])),
            "p_lo": float(ps.min()), "p_hi": float(ps.max()),
            "t_lo": float(ts.min()), "t_hi": float(ts.max())}


def _pt_scale_rows(pt_by_level: Dict[int, Dict[str, object]],
                   roi_h: float, roi_w: float) -> List[dict]:
    """Per-scale P-T rows from each level's per-variant estimates.

    ``pt_by_level[level][rep]`` for ``rep`` in {mean, mean+sigma, mean-sigma}
    is either a plain ``(P, T)`` best-fit point (legacy) or a region dict
    from :func:`_pt_region_from_misfit`. Per level the best estimate is the
    *mean* variant's point (else any), and the lo/hi envelope is the union
    over the available variants — of their near-minimal regions when regions
    are given, else of the points. The region-based envelope is preferred:
    on a flat misfit valley an argmin-point spread measures valley-hopping
    noise, while the union of accepted regions is stable at coarse scales
    and genuinely widens when the constraint degrades. ``window_px`` is the
    level's geometric-mean window side (``sqrt(roi_h/2^L * roi_w/2^L)``).
    Returns rows matching the pt_scale.csv schema, largest window first.
    """
    def _norm(rep):
        if isinstance(rep, dict):
            P, T = rep["pt"]
            return (float(P), float(T),
                    float(rep.get("p_lo", P)), float(rep.get("p_hi", P)),
                    float(rep.get("t_lo", T)), float(rep.get("t_hi", T)))
        P, T = rep
        return float(P), float(T), float(P), float(P), float(T), float(T)

    rows: List[dict] = []
    nan = float("nan")
    for lv in sorted(pt_by_level):
        reps = pt_by_level[lv] or {}
        if not reps:
            continue
        norm = {r: _norm(v) for r, v in reps.items()}
        best = norm.get("mean") or next(iter(norm.values()))
        plus = norm.get("mean+sigma")
        minus = norm.get("mean-sigma")
        window_px = float(np.sqrt((roi_h / (2 ** lv)) * (roi_w / (2 ** lv))))
        rows.append({
            # --- original schema: best cell + union envelope ---
            "window_px": window_px,
            "T_best_C": best[1],
            "T_lo_C": min(v[4] for v in norm.values()),
            "T_hi_C": max(v[5] for v in norm.values()),
            "P_best_GPa": best[0],
            "P_lo_GPa": min(v[2] for v in norm.values()),
            "P_hi_GPa": max(v[3] for v in norm.values()),
            # --- decomposition of the envelope (2026-09-16) ---
            # The mean variant's OWN accepted region: how flat the misfit
            # valley is on the window-mean bulk at this scale. The union
            # above adds the ±σ variants; where the two part, the widening
            # comes from the compositional spread rather than from the valley.
            "T_lo_mean_C": best[4], "T_hi_mean_C": best[5],
            "P_lo_mean_GPa": best[2], "P_hi_mean_GPa": best[3],
            # Best cells of the ±σ variants (argmin hopping along the valley).
            "T_best_plus_C": plus[1] if plus else nan,
            "P_best_plus_GPa": plus[0] if plus else nan,
            "T_best_minus_C": minus[1] if minus else nan,
            "P_best_minus_GPa": minus[0] if minus else nan,
        })
    return sorted(rows, key=lambda d: -d["window_px"])


def _pca_perturbed_compositions(window_wt: List[Dict[str, float]],
                                elements: List[str],
                                n_sigma: float = 1.0,
                                ) -> Tuple[Dict[str, float],
                                           Dict[str, float],
                                           Dict[str, float]]:
    """Per-scale sweep compositions perturbed along the dominant variation axis.

    Given the per-window wt% vectors at one length scale, return
    ``(mean, plus, minus)`` wt% dicts where *plus*/*minus* step ``±n_sigma``
    standard deviations along **PC1** — the first principal component of the
    between-window composition covariance.

    Why PC1 and not a marginal ``±σ`` per element: the bulk handed to a
    thermodynamic backend is a *closed* composition (the backend only sees
    element ratios), so any perturbation parallel to the mean is invisible after
    renormalisation. A vector of marginal standard deviations is non-negative,
    hence dominated by that invisible mean-parallel component — renormalising it
    largely (and, for an equal coefficient-of-variation spread, *exactly*)
    cancels the perturbation, so the modelled trio can be insensitive to the
    actual magnitude of the heterogeneity. PC1 of closed compositions is itself
    a (sum-to-≈zero) ratio-changing direction the windows genuinely vary along,
    and the step is the *true* standard deviation of the data projected onto it
    — so the perturbation reflects both the direction and the magnitude of the
    real heterogeneity. Multi-axis variation is captured only to the extent it
    aligns with PC1 (the trio is a 2-sided bracket, by design).

    Each returned composition is clipped at 0 and renormalised to 100 wt%. With
    fewer than two windows or no variance, ``plus == minus == mean``.
    """
    els = list(elements)

    def _close(vec) -> Optional[Dict[str, float]]:
        v = np.clip(np.asarray(vec, dtype=float), 0.0, None)
        s = float(v.sum())
        if s <= 0.0:
            return None
        v = v * (100.0 / s)
        return {el: float(x) for el, x in zip(els, v)}

    if not els:
        return {}, {}, {}
    X = np.array([[float((w or {}).get(el, 0.0)) for el in els]
                  for w in (window_wt or [])], dtype=float)
    if X.shape[0] == 0:
        return {}, {}, {}
    mu = X.mean(axis=0)
    mean_c = _close(mu) or {el: 0.0 for el in els}

    plus_c = minus_c = mean_c
    if X.shape[0] >= 2:
        Xc = X - mu
        try:
            _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            S = np.array([])
        if S.size and S[0] > 1e-12:
            v1 = Vt[0]
            # Deterministic sign: largest-|component| positive (SVD sign is
            # arbitrary; this keeps plus/minus reproducible across LAPACK builds).
            if v1[int(np.argmax(np.abs(v1)))] < 0:
                v1 = -v1
            # Population std (ddof=0) of the window projections onto PC1:
            # Σ(Xc·v1)² = S[0]², so std = S[0]/sqrt(n).
            std1 = float(S[0]) / np.sqrt(X.shape[0])
            step = n_sigma * std1 * v1
            plus_c = _close(mu + step) or mean_c
            minus_c = _close(mu - step) or mean_c
    return mean_c, plus_c, minus_c


def _modal_misfit_grid(results, observed_modes: Dict[str, float],
                       n_p: int, n_t: int, *,
                       weights: Optional[Dict[str, float]] = None,
                       presence_keys: Optional[Iterable[str]] = None,
                       open_other_tol: float = 0.0,
                       presence_penalty: float = 0.15,
                       ) -> Tuple[np.ndarray, str, List[str]]:
    """Per-cell modal misfit between observed and predicted assemblage modes.

    For each grid cell the predicted modal vector is built over the non-melt,
    non-fluid stable phases (per cell: ``vol_fraction`` preferred, then
    ``wt_fraction``, then ``mol_fraction`` — the first basis defined for every
    phase in that cell), renormalised to 1. Each predicted phase's fraction is
    split equally among the observed keys it matches (alias-aware); unmatched
    predicted mass pools into an "other" bucket that the observation claims is
    absent. With ``obs`` and ``pred`` both summing to 1,

        misfit = 0.5 * ( Σ_k |obs_k − pred_k|  +  pred_other )  ∈ [0, 1]

    i.e. the L1/2 (Bray-Curtis-style) distance: 0 = predicted modes equal the
    observed ones exactly, 1 = no overlap at all. Cells where no fraction basis
    covers all phases (or only melt/fluid is stable) are NaN.

    The keyword arguments relax the "the observed assemblage is exact, complete,
    and equally trustworthy" assumption (all default to the original behaviour):

    - ``weights`` — per-observed-key reliability (e.g. mineral-ID confidence).
      Down-weighting a shaky phase means a mismatch on it costs less. Weights are
      renormalised to mean 1 over the proportion keys, so a uniform set reproduces
      the unweighted distance and the metric stays ~[0, 1]. Weights also scale the
      per-phase presence penalty below.
    - ``presence_keys`` — phases the user knows are present but with **no modal
      target** (e.g. a trace phase segmentation missed, or one added from
      petrography/EBSD). They are matched like proportion keys but excluded from
      the proportion comparison (their predicted mass is set aside, not penalised
      as "other"); a fixed ``presence_penalty`` (×weight) is added per such phase
      that is **absent** from a cell. Keys also in ``observed_modes`` stay
      proportion keys.
    - ``open_other_tol`` — treat the observation as possibly **incomplete**: each
      unmatched predicted phase contributes only ``max(0, f − open_other_tol)`` to
      the "other" penalty, so a minor predicted phase the user simply didn't
      segment (≤ tol) isn't punished, while an abundant unobserved phase still is.
      ``0.0`` = the original (closed-assemblage) penalty.

    Returns ``(misfit (n_p, n_t), basis_label, unmatched_observed)`` where
    ``unmatched_observed`` lists observed/presence keys that match no stable phase
    anywhere in the grid (alias gap or genuinely never stable).
    """
    total = sum(float(v) for v in (observed_modes or {}).values())
    pres_keys = {str(k) for k in (presence_keys or [])}
    if total <= 0.0 and not pres_keys:
        return np.full((n_p, n_t), np.nan), "none", sorted(observed_modes or {})
    obs = ({str(k): float(v) / total for k, v in observed_modes.items()}
           if total > 0.0 else {})
    obs_keys = sorted(obs)
    pres_keys -= set(obs_keys)             # proportion keys win over presence

    # Weights → mean 1 over the proportion keys (uniform reproduces unweighted).
    w = {k: float((weights or {}).get(k, 1.0)) for k in obs_keys}
    if obs_keys:
        wsum = sum(w.values())
        scale = (len(obs_keys) / wsum) if wsum > 0 else 0.0
        w = {k: (w[k] * scale if scale > 0 else 1.0) for k in obs_keys}
    pw = {k: float((weights or {}).get(k, 1.0)) for k in pres_keys}
    tol = max(0.0, float(open_other_tol))

    misfit = np.full(n_p * n_t, np.nan)
    bases_used = set()
    matched_anywhere = {k: False for k in obs_keys}
    pres_seen_anywhere = {k: False for k in pres_keys}

    for idx, r in enumerate(results):
        solids = [ph for ph in r.stable_phases
                  if not ph.is_melt and not _is_fluid_phase(ph.name)]
        if not solids:
            continue
        fracs = None
        for basis in ("vol", "wt", "mol"):
            attr = f"{basis}_fraction"
            vals = [getattr(ph, attr, None) for ph in solids]
            if all(v is not None and v == v for v in vals):
                fracs = [float(v) for v in vals]
                bases_used.add(basis)
                break
        if fracs is None:
            continue
        ftot = sum(fracs)
        if ftot <= 0.0:
            continue

        pred = {k: 0.0 for k in obs_keys}
        other = 0.0           # penalised unmatched mass (after forgiveness)
        m_pres = 0.0          # mass routed to presence phases (set aside)
        pres_here = {k: False for k in pres_keys}
        for ph, f in zip(solids, fracs):
            f = f / ftot
            hits = [k for k in obs_keys if _phase_matches(k, ph.name)]
            if hits:
                for k in hits:
                    pred[k] += f / len(hits)
                    matched_anywhere[k] = True
                continue
            phits = [k for k in pres_keys if _phase_matches(k, ph.name)]
            if phits:
                m_pres += f
                for k in phits:
                    pres_here[k] = True
                    pres_seen_anywhere[k] = True
            else:
                other += max(0.0, f - tol)

        # Proportion distance in the non-presence subspace (renormalise out the
        # set-aside presence mass so a present trace phase doesn't inflate it).
        denom = 1.0 - m_pres
        if denom <= 1e-12:
            prop_term = 0.5 * sum(w[k] * obs[k] for k in obs_keys)
        else:
            prop_term = 0.5 * (sum(w[k] * abs(obs[k] - pred[k] / denom)
                                   for k in obs_keys) + other / denom)
        pres_term = sum(presence_penalty * pw[k]
                        for k in pres_keys if not pres_here[k])
        misfit[idx] = min(1.0, prop_term + pres_term)

    basis_label = "/".join(b for b in ("vol", "wt", "mol") if b in bases_used) or "none"
    unmatched = [k for k in obs_keys if not matched_anywhere[k]]
    unmatched += [k for k in sorted(pres_keys) if not pres_seen_anywhere[k]]
    return misfit.reshape(n_p, n_t), basis_label, unmatched


# -------------------------
# Monte Carlo bulk-composition uncertainty
# -------------------------

def _mc_sample_compositions(comp: Dict[str, float],
                            sigma,
                            n: int,
                            seed: int = 0) -> List[Dict[str, float]]:
    """``n`` perturbed copies of a composition for Monte Carlo propagation.

    Each component is drawn independently from a Gaussian truncated at zero.
    ``sigma`` is either a single *relative* standard deviation (e.g. ``0.05``
    = 5% of each component's value) or a per-component dict of *absolute*
    standard deviations in the composition's own units. Every sample is
    renormalised back to the input's total, so wt% stays wt% and only the
    *proportions* vary — which is what feeds a thermodynamic model.

    Deterministic for a given ``seed`` (and the grid cache then makes a
    repeated MC with identical settings instant).
    """
    rng = np.random.default_rng(int(seed))
    base_total = sum(float(v) for v in comp.values())
    rel = None if isinstance(sigma, dict) else float(sigma)
    out: List[Dict[str, float]] = []
    for _ in range(max(1, int(n))):
        d: Dict[str, float] = {}
        for k, v in comp.items():
            v = float(v)
            s = (rel * v) if rel is not None else float(sigma.get(k, 0.0))
            d[k] = max(0.0, v + s * rng.standard_normal())
        tot = sum(d.values())
        if tot > 0 and base_total > 0:
            f = base_total / tot
            d = {k: val * f for k, val in d.items()}
        out.append(d)
    return out


def _mc_sample_modes(observed_modes: Dict[str, float], n: int,
                     seed: int = 0, rel_sigma: float = 0.2,
                     ) -> List[Dict[str, float]]:
    """``n`` resampled copies of an observed modal vector (proportion uncertainty).

    Each mode fraction is perturbed by a Gaussian of ``rel_sigma × fraction``,
    truncated at 0 and renormalised to 1, so every sample stays a valid modal
    vector. This propagates **observation** uncertainty (segmentation areas /
    mineral-ID errors) — and because it is used only to *re-score grids that are
    already computed*, it adds essentially no compute (no new Gibbs
    minimisations). Deterministic for a given ``seed``.
    """
    keys = sorted(observed_modes or {})
    base = np.array([float(observed_modes[k]) for k in keys], dtype=float)
    tot = base.sum()
    if tot <= 0.0:
        return []
    base = base / tot
    rng = np.random.default_rng(int(seed))
    out: List[Dict[str, float]] = []
    for _ in range(max(1, int(n))):
        v = np.clip(base + rel_sigma * base * rng.standard_normal(len(base)),
                    0.0, None)
        s = v.sum()
        if s <= 0.0:
            continue
        out.append({k: float(x / s) for k, x in zip(keys, v)})
    return out


def _mc_aggregate(grids, n_p: int, n_t: int,
                  observed_phases: Optional[List[str]] = None,
                  observed_modes: Optional[Dict[str, float]] = None,
                  misfit_kwargs: Optional[dict] = None,
                  mode_samples: Optional[List[Dict[str, float]]] = None) -> dict:
    """Aggregate Monte Carlo grid runs into uncertainty maps.

    ``grids`` is one results-list per MC sample (``None``/empty = failed run,
    skipped). Returns a dict with ``n_runs``/``n_failed`` and, depending on
    what was observed:

    - ``stability_prob`` (n_p, n_t): fraction of runs in which ALL observed
      phases are stable in that cell (alias-aware).
    - ``misfit_mean`` / ``misfit_std`` (n_p, n_t): per-cell statistics of the
      modal misfit across runs (NaN-aware).
    - ``best_fits``: ``(i, j, misfit)`` of each run's best-fit cell — the scatter
      of the P-T estimate under bulk-composition uncertainty.

    ``misfit_kwargs`` (per-key weights / presence keys / open-assemblage tol)
    are forwarded to every misfit evaluation so MC honours the same relaxed
    assumptions as the rest of the pipeline. ``mode_samples`` (from
    :func:`_mc_sample_modes`) additionally folds **observation** uncertainty into
    ``best_fits``: each already-computed grid is re-scored against every
    resampled observation (cheap — no new grids), so the cloud reflects bulk
    *and* modal/ID uncertainty together.
    """
    mk = misfit_kwargs or {}
    valid = [g for g in grids if g]
    agg: dict = {"n_runs": len(valid), "n_failed": len(grids) - len(valid)}
    if not valid:
        return agg

    if observed_phases:
        stack = np.stack([
            _pt_estimate_from_results(g, observed_phases, n_p, n_t).astype(float)
            for g in valid])
        agg["stability_prob"] = stack.mean(axis=0)

    if observed_modes:
        mis = np.stack([
            _modal_misfit_grid(g, observed_modes, n_p, n_t, **mk)[0]
            for g in valid])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            agg["misfit_mean"] = np.nanmean(mis, axis=0)
            agg["misfit_std"] = np.nanstd(mis, axis=0)
        best = []
        for m in mis:
            if np.isfinite(m).any():
                i, j = np.unravel_index(int(np.nanargmin(m)), m.shape)
                best.append((int(i), int(j), float(m[i, j])))
        # Fold observation uncertainty into the best-fit cloud by re-scoring the
        # SAME grids against each resampled observed mode vector (no new grids).
        if mode_samples:
            for g in valid:
                for om in mode_samples:
                    mm = _modal_misfit_grid(g, om, n_p, n_t, **mk)[0]
                    if np.isfinite(mm).any():
                        i, j = np.unravel_index(int(np.nanargmin(mm)), mm.shape)
                        best.append((int(i), int(j), float(mm[i, j])))
            agg["mode_resampled"] = len(mode_samples)
        agg["best_fits"] = best
    return agg


# -------------------------
# Multi-scale composition helpers
# -------------------------

def _net_count_sums(band_maps: Dict[str, Dict[str, np.ndarray]],
                    sub_bbox: Optional[Tuple[int, int, int, int]] = None,
                    sub_mask: Optional[np.ndarray] = None,
                    scheme: Optional[str] = None,
                    plan: Optional[Dict[str, dict]] = None,
                    ) -> Tuple[Dict[str, float], int]:
    """Per-element *raw* integrated net-counts over a sub-window of an ROI's band maps.

    Same windowing/masking contract as :func:`_net_count_fractions`, but returns
    the un-normalised per-element sums instead of fractions. The net per pixel
    is ``main - background`` with the background from
    ``eds_background.net_count_map`` (line-aware by default; ``scheme`` /
    ``plan`` override the process default and the plan derived from the
    dict's own elements). The sums are SIGNED: a per-pixel clip at zero would
    rectify the Poisson noise of a 1-4 counts/px map into a positive bias
    (2026-09-16). Callers clip at the aggregate where a non-negative value is
    required. The raw sums are what the SPX-anchored k-factor fit
    (``eds_quantify.derive_k_factors``) needs: normalising to fractions first
    would couple every element's intensity to the others, which breaks the
    per-element linear-through-origin model when tiles have different element
    sets. ``_net_count_fractions`` is defined in terms of this function.

    Returns ``({el: net_count}, n_pixels_used)``; ``({}, 0)`` for an empty slice.
    """
    if not band_maps:
        return {}, 0
    sample = next(iter(band_maps.values()))['main']
    H, W = sample.shape[:2]
    if sub_bbox is None:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = sub_bbox
        y0 = max(0, int(y0)); y1 = min(H, int(y1))
        x0 = max(0, int(x0)); x1 = min(W, int(x1))
    if y1 <= y0 or x1 <= x0:
        return {}, 0

    if sub_mask is not None:
        m = np.asarray(sub_mask).astype(bool)
        n_pix = int(m.sum())
    else:
        m = None
        n_pix = (y1 - y0) * (x1 - x0)
    if n_pix <= 0:
        return {}, 0

    from eds_background import background_plan, net_count_map
    scheme = scheme or _bg_default_scheme()
    if plan is None and scheme != "sideband":
        plan = background_plan(list(band_maps))

    def get_band(e: str, b: str):
        arr = (band_maps.get(e) or {}).get(b)
        return None if arr is None else arr[y0:y1, x0:x1]

    net_counts: Dict[str, float] = {}
    for el in band_maps:
        net = net_count_map(el, get_band, plan=plan, scheme=scheme)
        if net is None:
            continue
        if m is not None:
            net = net * m
        net_counts[el] = float(np.nansum(net))
    return net_counts, n_pix


def _bg_default_scheme() -> str:
    from eds_background import default_scheme
    return default_scheme()


def _net_count_fractions(band_maps: Dict[str, Dict[str, np.ndarray]],
                         sub_bbox: Optional[Tuple[int, int, int, int]] = None,
                         sub_mask: Optional[np.ndarray] = None,
                         scheme: Optional[str] = None,
                         plan: Optional[Dict[str, dict]] = None,
                         ) -> Tuple[Dict[str, float], int]:
    """Per-element net-count fractions over a sub-window of an ROI's band maps.

    ``band_maps`` is the ROI-local dict produced by ``roi_to_banded_maps`` /
    ``roi_from_cache`` (each element has 'main', 'sb_lo', 'sb_hi' float32
    arrays of identical shape). ``sub_bbox`` is in ROI-local coordinates
    ``(y0, y1, x0, x1)``; if None, the whole ROI is used. ``sub_mask`` is
    multiplied pointwise into the (clipped) net signal - pass the relevant
    slice of the parent ROI polygon mask to honour non-rectangular ROIs.

    Returns ``({el: fraction}, n_pixels_used)``. Fractions sum to 1 unless
    the slice carries no positive net signal, in which case ``{}`` and the
    pixel count are returned. The signed per-element sums are clipped at zero
    HERE, at the aggregate, before normalising (an element whose window sum
    is negative is background-only and reads as 0).
    """
    net_counts, n_pix = _net_count_sums(band_maps, sub_bbox=sub_bbox, sub_mask=sub_mask,
                                        scheme=scheme, plan=plan)
    if not net_counts:
        return {}, n_pix
    clipped = {el: max(0.0, v) for el, v in net_counts.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return {}, n_pix
    return {el: v / total for el, v in clipped.items()}, n_pix


def _max_useful_quadtree_depth(bbox: Tuple[int, int, int, int],
                                 min_window_px: int = 4) -> int:
    """Largest quadtree depth that keeps both window dimensions >= ``min_window_px``.

    Integer halving means level-k window is roughly ``min_dim // 2**k`` on its
    short side. We stop one level before either dimension falls below the
    threshold. Returns 0 for degenerate / very small bboxes - caller should
    handle that case explicitly.
    """
    y0, y1, x0, x1 = bbox
    h = max(0, int(y1) - int(y0))
    w = max(0, int(x1) - int(x0))
    if h < int(min_window_px) or w < int(min_window_px):
        return 0
    md = 0
    while True:
        nh = h // (2 ** (md + 1))
        nw = w // (2 ** (md + 1))
        if nh < int(min_window_px) or nw < int(min_window_px):
            break
        md += 1
    return md


def _quadtree_windows(bbox: Tuple[int, int, int, int], depth: int
                      ) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """Quadtree tiling of ``bbox`` (y0, y1, x0, x1) into ``depth`` levels.

    Level 0 is the full bbox. Level k splits each level-(k-1) window into
    four equal quadrants. Returns ``{level: [bbox, ...]}``.
    """
    y0, y1, x0, x1 = bbox
    levels: Dict[int, List[Tuple[int, int, int, int]]] = {0: [tuple(bbox)]}
    cur = [tuple(bbox)]
    for k in range(1, max(0, int(depth)) + 1):
        nxt: List[Tuple[int, int, int, int]] = []
        for (a0, a1, b0, b1) in cur:
            ymid = (a0 + a1) // 2
            xmid = (b0 + b1) // 2
            # Skip degenerate splits (window too small to halve)
            if ymid <= a0 or ymid >= a1 or xmid <= b0 or xmid >= b1:
                continue
            nxt.append((a0, ymid, b0, xmid))
            nxt.append((a0, ymid, xmid, b1))
            nxt.append((ymid, a1, b0, xmid))
            nxt.append((ymid, a1, xmid, b1))
        if not nxt:
            break
        levels[k] = nxt
        cur = nxt
    return levels


def _quadtree_axis_segments(lo: int, hi: int, level: int
                            ) -> List[Tuple[int, int]]:
    """1-D recursive equal-halving of ``[lo, hi)`` to ``level`` (O(2**level)).

    A quadtree with equal halving is the Cartesian product of these per-axis
    segments, so windows can be generated from two axis splittings without ever
    materialising all ``4**level`` cells. A segment too small to halve is kept
    as a leaf (mirrors the degenerate-split guard in :func:`_quadtree_windows`).
    """
    segs: List[Tuple[int, int]] = [(int(lo), int(hi))]
    for _ in range(max(0, int(level))):
        nxt: List[Tuple[int, int]] = []
        for a, b in segs:
            m = (a + b) // 2
            if m <= a or m >= b:
                nxt.append((a, b))
            else:
                nxt.append((a, m))
                nxt.append((m, b))
        segs = nxt
    return segs


def _quadtree_level_dims(bbox: Tuple[int, int, int, int], level: int):
    """``(rows, cols, (win_h, win_w))`` at ``level`` without building windows.

    Lets the settings dialog show per-level window counts and sizes for a deep
    quadtree on a huge ROI instantly — counting via two O(2**level) axis splits
    instead of materialising O(4**level) windows.
    """
    y0, y1, x0, x1 = bbox
    yseg = _quadtree_axis_segments(y0, y1, level)
    xseg = _quadtree_axis_segments(x0, x1, level)
    wh = (yseg[0][1] - yseg[0][0]) if yseg else 0
    ww = (xseg[0][1] - xseg[0][0]) if xseg else 0
    return len(yseg), len(xseg), (wh, ww)


def _quadtree_sampled_windows(bbox: Tuple[int, int, int, int], level: int,
                              cap: Optional[int] = None, rng=None
                              ) -> List[Tuple[int, int, int, int]]:
    """Up to ``cap`` window bboxes at ``level`` from the axis-segment grid.

    Windows are cells of the ``rows × cols`` Cartesian product of the per-axis
    segments. When ``cap`` is below the total, cell indices are sampled (random
    when ``rng`` is given, else the first ``cap`` in scan order) and only those
    cells' bboxes are built — ``O(cap + 2**level)``, never ``O(4**level)``. This
    is what keeps composition-vs-scale tractable at depth on a large ROI.

    The random sample is spatially uniform over the ROI and has exactly
    ``cap`` windows, returned in scan order. (Until 2026-10-02 the deduplicated
    draws, which ``np.unique`` returns sorted, were truncated to ``cap``
    without shuffling, which kept the LOWEST cell indices: the capped levels
    covered only the top of the ROI — 79 % and 57 % of the worked-example
    section at its two finest levels — and could fall short of ``cap``. The
    "first N" mode is spatially biased by design.)
    """
    y0, y1, x0, x1 = bbox
    yseg = _quadtree_axis_segments(y0, y1, level)
    xseg = _quadtree_axis_segments(x0, x1, level)
    rows, cols = len(yseg), len(xseg)
    total = rows * cols
    if total == 0:
        return []
    if cap is None or int(cap) >= total:
        return [(a0, a1, b0, b1) for (a0, a1) in yseg for (b0, b1) in xseg]
    cap = max(1, int(cap))
    if rng is not None:
        if cap * 4 >= total and total <= 4_000_000:
            # Dense request: an exact permutation is cheap enough and always
            # yields ``cap`` distinct cells.
            flat = rng.permutation(total)[:cap]
        else:
            # Sparse request: sample with replacement + dedup
            # (Generator.choice(replace=False) would permute the whole
            # 4**level population, defeating the purpose), top up if the
            # dedup fell short, then SHUFFLE before truncating: np.unique
            # returns the draws sorted, so truncating them directly keeps
            # only the lowest indices (the top of the ROI).
            uniq = np.unique(rng.integers(0, total, size=cap * 2 + 16))
            while len(uniq) < cap:
                uniq = np.unique(np.concatenate(
                    [uniq, rng.integers(0, total, size=cap)]))
            flat = rng.permutation(uniq)[:cap]
        flat = np.sort(flat)                 # scan order for the caller
    else:
        flat = np.arange(min(cap, total))
    out: List[Tuple[int, int, int, int]] = []
    for f in flat.tolist():
        r, c = divmod(int(f), cols)
        a0, a1 = yseg[r]
        b0, b1 = xseg[c]
        out.append((a0, a1, b0, b1))
    return out


# -------------------------
# Spatial-structure metrics (composition vs scale)
# -------------------------
# Per-scale *variance* says how much composition spreads at a given window
# size, but ignores *where* the windows sit. A semivariogram and Moran's I add
# the spatial dimension: they reveal the characteristic length over which
# composition decorrelates (the variogram range) and whether like values
# cluster in space (Moran's I). Both operate on the quadtree window records,
# using window centres (in mosaic pixels) as point locations.

def _usable_windows_at_level(records: List[dict], level: int
                             ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Centres ``(N, 2)`` as ``[cy, cx]`` and the wt% dicts of usable windows.

    "Usable" = the window has a non-empty ``wt_pct`` (not skipped). Centres are
    window-bbox midpoints in mosaic-pixel coordinates.
    """
    centers: List[Tuple[float, float]] = []
    comps: List[Dict[str, float]] = []
    for r in records:
        if r.get("level") != level:
            continue
        wt = r.get("wt_pct")
        if not wt:
            continue
        y0, y1, x0, x1 = r["bbox"]
        centers.append(((y0 + y1) / 2.0, (x0 + x1) / 2.0))
        comps.append(wt)
    arr = np.asarray(centers, dtype=float).reshape(-1, 2)
    return arr, comps


def _eligible_variogram_levels(records: List[dict], min_windows: int = 4
                               ) -> List[int]:
    """Quadtree levels with at least ``min_windows`` usable windows (sorted)."""
    by_level: Dict[int, int] = {}
    for r in records:
        if r.get("wt_pct"):
            by_level[r["level"]] = by_level.get(r["level"], 0) + 1
    return sorted(lv for lv, n in by_level.items() if n >= int(min_windows))


def _semivariogram(records: List[dict], level: int, elements: List[str],
                   n_bins: int = 12, max_lag_frac: float = 0.6,
                   max_points: int = 3000, seed: int = 0,
                   direction_deg: Optional[float] = None,
                   tol_deg: float = 22.5) -> dict:
    """Empirical semivariogram at one quadtree level, omnidirectional or directional.

    ``γ(h) = ½ · mean over window pairs separated by ≈h of (z_i − z_j)²``.
    Computed per element, plus a ``total`` curve that uses the squared
    Euclidean distance summed over ``elements`` (a compositional dissimilarity)
    so its rise summarises the whole composition. Lag ``h`` is in mosaic pixels.

    ``direction_deg`` restricts the pairs to those whose separation vector lies
    within ``±tol_deg`` of that direction (degrees from the +x mosaic axis
    towards +y, modulo 180: 0 = along x, 90 = along y). A layered rock
    decorrelates faster across the layering than along it, so the directional
    curves give the correlation length per direction; ``None`` uses every pair.

    Pairs farther apart than ``max_lag_frac × max_pair_distance`` are dropped
    (the far tail is sparse and noisy). Returns ``{}`` if fewer than 4 usable
    windows (or no pair in the chosen sector). Otherwise::

        {"level", "n_windows", "n_used", "bin_centers"[K], "counts"[K],
         "gamma": {el: ndarray[K]}, "gamma_total": ndarray[K], "max_lag",
         "direction_deg", "tol_deg"}

    Bins containing no pairs hold ``NaN`` γ and ``0`` count.

    The pairwise computation is O(N²); to keep memory bounded on deep quadtrees
    (a fine level can have tens of thousands of windows) the windows are
    randomly subsampled to ``max_points`` (seeded → reproducible) before pairing.
    A few-thousand-point sample estimates the variogram well.
    """
    centers, comps = _usable_windows_at_level(records, level)
    n = len(comps)
    if n < 4:
        return {}
    n_total = n
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        idx.sort()
        centers = centers[idx]
        comps = [comps[i] for i in idx]
        n = max_points

    iu = np.triu_indices(n, k=1)
    dy = centers[iu[0], 0] - centers[iu[1], 0]
    dx = centers[iu[0], 1] - centers[iu[1], 1]
    hh = np.sqrt(dy * dy + dx * dx)
    if hh.size == 0 or not np.isfinite(hh).any() or hh.max() <= 0:
        return {}

    # Directional sector: angle of the separation vector modulo 180° (a pair
    # has no orientation), kept when within tol_deg of the requested direction.
    sector = np.ones(hh.shape, dtype=bool)
    if direction_deg is not None:
        ang = np.degrees(np.arctan2(dy, dx)) % 180.0
        dev = np.abs(ang - (float(direction_deg) % 180.0))
        dev = np.minimum(dev, 180.0 - dev)
        sector = dev <= float(tol_deg)
        if not sector.any():
            return {}

    max_lag = float(max_lag_frac) * float(hh.max())
    if max_lag <= 0:
        max_lag = float(hh.max())
    keep = (hh <= max_lag) & sector
    hh = hh[keep]
    bins = np.linspace(0.0, max_lag, int(n_bins) + 1)
    idx = np.clip(np.digitize(hh, bins) - 1, 0, int(n_bins) - 1)
    counts = np.bincount(idx, minlength=int(n_bins)).astype(int)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    def _gamma(weights: np.ndarray) -> np.ndarray:
        s = np.bincount(idx, weights=weights, minlength=int(n_bins))
        with np.errstate(invalid="ignore", divide="ignore"):
            g = 0.5 * s / counts
        g[counts == 0] = np.nan
        return g

    z_by_el = {el: np.array([c.get(el, 0.0) for c in comps], dtype=float)
               for el in elements}
    gamma: Dict[str, np.ndarray] = {}
    total_sq = np.zeros(keep.sum(), dtype=float)
    for el in elements:
        z = z_by_el[el]
        d = (z[iu[0]] - z[iu[1]])[keep]
        sq = d * d
        total_sq += sq
        gamma[el] = _gamma(sq)
    gamma_total = _gamma(total_sq)

    return {
        "level": int(level),
        "n_windows": int(n_total),
        "n_used": int(n),
        "bin_centers": bin_centers,
        "counts": counts,
        "gamma": gamma,
        "gamma_total": gamma_total,
        "max_lag": float(max_lag),
        "direction_deg": (None if direction_deg is None else float(direction_deg)),
        "tol_deg": float(tol_deg),
    }


def _spread_by_level(records: List[dict], elements: List[str],
                     ddof: int = 1) -> Dict[int, Dict[str, float]]:
    """Between-window standard deviation of each element at every quadtree level.

    ``{level: {el: std}}`` over the usable (non-skipped) windows; levels with
    fewer than two usable windows are omitted. ``ddof=1`` (sample std) is the
    convention the manuscript quotes; the worker's ``summary`` uses ddof=0.
    """
    out: Dict[int, Dict[str, float]] = {}
    for lv in sorted({r.get("level") for r in records if r.get("wt_pct")}):
        _c, comps = _usable_windows_at_level(records, lv)
        if len(comps) < 2:
            continue
        out[int(lv)] = {}
        for el in elements:
            vals = np.array([c.get(el, 0.0) for c in comps], dtype=float)
            out[int(lv)][el] = float(vals.std(ddof=ddof)) if vals.size > ddof else float("nan")
    return out


def _representative_level(spread: Dict[int, Dict[str, float]],
                          tolerance: Union[float, Dict[str, float]],
                          means: Optional[Dict[int, Dict[str, float]]] = None,
                          elements: Optional[List[str]] = None
                          ) -> Tuple[Optional[int], Dict[int, Dict[str, float]]]:
    """The representative averaging scale as a quadtree level.

    A window size is representative when the between-window spread of EVERY
    element is at or below its tolerance; the representative level is the
    deepest (smallest-window) level that passes together with every coarser
    level, so one noisy dip at a fine level does not qualify. The spread never
    plateaus on a heterogeneous sample (it keeps falling as the window grows),
    which is why a tolerance is needed to define the scale at all.

    ``tolerance`` is either a per-element absolute dict (wt%; the natural
    choice is the calibration's hold-out RMSE: a window whose placement moves
    the bulk by less than the maps resolve is representative) or a single
    relative fraction of each element's mean at that level (``means`` required
    then; e.g. 0.10 = 10 %). Elements absent from an absolute dict are not
    tested. Returns ``(level or None, {level: {el: spread / tolerance}})``.
    """
    ratios: Dict[int, Dict[str, float]] = {}
    levels = sorted(spread)
    for lv in levels:
        ratios[lv] = {}
        for el, sd in spread[lv].items():
            if elements is not None and el not in elements:
                continue
            if isinstance(tolerance, dict):
                tol = tolerance.get(el)
                if tol is None:
                    continue
            else:
                mu = (means or {}).get(lv, {}).get(el)
                if mu is None:
                    continue
                tol = float(tolerance) * float(mu)
            if tol is None or not np.isfinite(tol) or tol <= 0:
                continue
            ratios[lv][el] = float(sd) / float(tol)
    best: Optional[int] = None
    for lv in levels:
        r = ratios[lv]
        if not r:
            break
        if all(np.isfinite(v) and v <= 1.0 for v in r.values()):
            best = lv
        else:
            break
    return best, ratios


def _morans_i(records: List[dict], level: int, elements: List[str],
              max_points: int = 3000, seed: int = 0) -> Dict[str, float]:
    """Inverse-distance Moran's I per element at one quadtree level.

    Spatial weights are ``w_ij = 1/distance`` (``i≠j``; mosaic-pixel distance).
    Returns ``{el: I}`` where I is roughly in ``[-1, 1]``: positive → like
    values cluster spatially, ≈0 → no spatial pattern, negative → neighbouring
    windows tend to differ. ``NaN`` when there are fewer than 3 usable windows
    or an element has zero variance.

    Builds an N×N weight matrix, so on deep quadtrees the windows are randomly
    subsampled to ``max_points`` (seeded) first to keep memory bounded.
    """
    centers, comps = _usable_windows_at_level(records, level)
    n = len(comps)
    out: Dict[str, float] = {el: float("nan") for el in elements}
    if n < 3:
        return out
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        idx.sort()
        centers = centers[idx]
        comps = [comps[i] for i in idx]
        n = max_points
    dy = centers[:, 0][:, None] - centers[:, 0][None, :]
    dx = centers[:, 1][:, None] - centers[:, 1][None, :]
    dist = np.sqrt(dy * dy + dx * dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / dist
    w[~np.isfinite(w)] = 0.0          # zeroes the diagonal (dist=0) too
    w_sum = w.sum()
    if w_sum <= 0:
        return out
    for el in elements:
        z = np.array([c.get(el, 0.0) for c in comps], dtype=float)
        dz = z - z.mean()
        denom = float((dz * dz).sum())
        if denom <= 0:
            continue
        num = float((w * (dz[:, None] * dz[None, :])).sum())
        out[el] = (n / w_sum) * num / denom
    return out


# -------------------------
# Thermodynamic run records: serialisation + display helpers
# -------------------------

def _serialize_phase_result(p) -> dict:
    return {
        "name": p.name,
        "mol_fraction": float(p.mol_fraction) if p.mol_fraction is not None else None,
        "wt_fraction": float(p.wt_fraction) if p.wt_fraction is not None else None,
        "vol_fraction": float(p.vol_fraction) if p.vol_fraction is not None else None,
        "composition_oxide_wt": (dict(p.composition_oxide_wt)
                                  if p.composition_oxide_wt else None),
        "composition_oxide_mol": (dict(p.composition_oxide_mol)
                                   if p.composition_oxide_mol else None),
        "is_melt": bool(p.is_melt),
        "endmembers": dict(p.endmembers) if p.endmembers else None,
    }


def _deserialize_phase_result(d: dict):
    from thermo.result import PhaseResult
    return PhaseResult(
        name=d.get("name", ""),
        mol_fraction=float(d.get("mol_fraction") or 0.0),
        wt_fraction=(float(d["wt_fraction"]) if d.get("wt_fraction") is not None else None),
        vol_fraction=(float(d["vol_fraction"]) if d.get("vol_fraction") is not None else None),
        composition_oxide_wt=d.get("composition_oxide_wt") or None,
        composition_oxide_mol=d.get("composition_oxide_mol") or None,
        is_melt=bool(d.get("is_melt", False)),
        endmembers=d.get("endmembers") or None,
    )


def _serialize_thermod_result(r) -> dict:
    return {
        "p_gpa": float(r.p_gpa),
        "t_c": float(r.t_c),
        "stable_phases": [_serialize_phase_result(p) for p in r.stable_phases],
        "bulk_properties": {k: (None if v is None else float(v))
                            for k, v in (r.bulk_properties or {}).items()},
        "backend": str(r.backend or ""),
        "converged": bool(r.converged),
    }


def _deserialize_thermod_result(d: dict):
    from thermo.result import ThermoResult
    return ThermoResult(
        p_gpa=float(d.get("p_gpa", 0.0)),
        t_c=float(d.get("t_c", 0.0)),
        stable_phases=[_deserialize_phase_result(p) for p in d.get("stable_phases", [])],
        bulk_properties=dict(d.get("bulk_properties") or {}),
        backend=str(d.get("backend", "")),
        converged=bool(d.get("converged", True)),
        raw=None,
    )


def _format_composition_lines(comp_el: dict, comp_source: str) -> List[str]:
    if not comp_el:
        return ["(no composition recorded)"]
    units = "oxide wt%" if comp_source == "custom" else "element wt%"
    lines = [f"Composition ({comp_source}, {units}):"]
    for k in sorted(comp_el.keys()):
        try:
            lines.append(f"  {k:<8s} {float(comp_el[k]):8.3f}")
        except (TypeError, ValueError):
            lines.append(f"  {k}: {comp_el[k]}")
    return lines


def _thermod_parameters_text(run: dict) -> str:
    """Human-readable parameters block for a saved run."""
    opts = run.get("opts", {}) or {}
    bkw = opts.get("backend_kwargs", {}) or {}
    lines: List[str] = []
    lines.append(f"Run ID:       {run.get('id', '?')}")
    lines.append(f"Created:      {run.get('timestamp', '?')}")
    lines.append(f"Mode:         {run.get('mode', '?')}")
    lines.append(f"Backend:      {opts.get('backend', '?')}")
    if bkw:
        lines.append("Backend kwargs:")
        for k, v in bkw.items():
            lines.append(f"  {k}: {v}")
    if run.get("mode") == "single":
        lines.append(f"P:            {opts.get('p_gpa', '?')} GPa")
        lines.append(f"T:            {opts.get('t_c', '?')} \N{DEGREE SIGN}C")
    else:
        pr = opts.get("p_range", (None, None))
        tr = opts.get("t_range", (None, None))
        lines.append(f"P range:      {pr[0]} … {pr[1]} GPa  ({opts.get('n_p', '?')} pts)")
        lines.append(f"T range:      {tr[0]} … {tr[1]} \N{DEGREE SIGN}C  ({opts.get('n_t', '?')} pts)")
        lines.append(f"Workflow:     {opts.get('workflow', '?')}")
        if opts.get("observed_phases"):
            lines.append(f"Observed:     {', '.join(opts['observed_phases'])}")
        if opts.get("observed_modes"):
            modes = opts["observed_modes"]
            lines.append("Observed modes: "
                         + ", ".join(f"{k} {100 * float(v):.0f}%"
                                     for k, v in sorted(modes.items(),
                                                        key=lambda kv: -kv[1])))
    lines.append(f"Fe3+ frac:    {opts.get('fe3', 0.0)}")
    _basis = ("of the bulk" if opts.get("extra_basis") == "bulk"
              else "legacy: added to a cation-normalised total, about half this in the bulk")
    lines.append(f"H2O extra:    {opts.get('h2o', 0.0)} wt% ({_basis})")
    lines.append(f"CO2 extra:    {opts.get('co2', 0.0)} wt% ({_basis})")
    lines.append("")
    lines.append(f"ROI label:    {run.get('roi_label', '(none)')}")
    bbox = run.get("roi_bbox")
    if bbox:
        y0, y1, x0, x1 = bbox
        lines.append(f"ROI bbox:     y=[{y0}, {y1}]  x=[{x0}, {x1}]   ({y1-y0} × {x1-x0} px)")
    lines.append(f"Manifest:     {run.get('manifest', '(unknown)')}")
    lines.append("")
    lines.extend(_format_composition_lines(
        run.get("comp_el", {}), run.get("comp_source", "?"),
    ))
    return "\n".join(lines)


def _export_thermod_bundle(run: dict, out_dir: Path, fig=None) -> List[Path]:
    """Write parameters.json + data CSVs (+ figure for grid runs) to *out_dir*.

    Returns the list of files written.
    """
    import json as _json
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # 1. parameters.json - full run record minus the heavy data arrays
    params = {k: run[k] for k in run.keys()
              if k not in ("result", "results")}
    p_path = out_dir / "parameters.json"
    p_path.write_text(_json.dumps(params, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    written.append(p_path)

    # 2. summary.txt - readable
    s_path = out_dir / "summary.txt"
    s_path.write_text(_thermod_parameters_text(run), encoding="utf-8")
    written.append(s_path)

    if run.get("mode") == "single" and run.get("result"):
        r = _deserialize_thermod_result(run["result"])
        # Phases (mol/wt/vol)
        rows = []
        for ph in r.stable_phases:
            rows.append({
                "phase": ph.name,
                "mol_fraction": ph.mol_fraction,
                "wt_fraction": ph.wt_fraction,
                "vol_fraction": ph.vol_fraction,
                "is_melt": ph.is_melt,
            })
        ph_path = out_dir / "phases.csv"
        pd.DataFrame(rows).to_csv(ph_path, index=False)
        written.append(ph_path)

        # Phase compositions (oxide wt% wide)
        comp_rows = [ph for ph in r.stable_phases if ph.composition_oxide_wt]
        if comp_rows:
            all_ox = sorted({ox for ph in comp_rows for ox in ph.composition_oxide_wt})
            cdata = []
            for ph in comp_rows:
                row = {"phase": ph.name}
                for ox in all_ox:
                    row[ox] = ph.composition_oxide_wt.get(ox)
                cdata.append(row)
            c_path = out_dir / "phase_compositions.csv"
            pd.DataFrame(cdata).to_csv(c_path, index=False)
            written.append(c_path)

        # Bulk properties
        if r.bulk_properties:
            bp_path = out_dir / "bulk_properties.csv"
            pd.DataFrame(
                [{"property": k, "value": v} for k, v in r.bulk_properties.items()]
            ).to_csv(bp_path, index=False)
            written.append(bp_path)
    elif run.get("mode") == "grid" and run.get("results"):
        from thermo import results_to_dataframe
        results_obj = [_deserialize_thermod_result(d) for d in run["results"]]
        g_path = out_dir / "grid.csv"
        results_to_dataframe(results_obj).to_csv(g_path, index=False)
        written.append(g_path)
        if fig is not None:
            fig_path = out_dir / "pseudosection.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            written.append(fig_path)

    return written


def _assemblage_selector_dialog(results, parent=None) -> Optional[List[str]]:
    """Post-run dialog: checklist of all phases in the grid; returns selected names or None."""
    seen: Dict[str, str] = {}   # lower -> original casing (first occurrence)
    for r in results:
        for ph in r.stable_phases:
            key = ph.name.strip().lower()
            if key not in seen:
                seen[key] = ph.name.strip()
    all_names = sorted(seen.values(), key=str.lower)

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Select observed assemblage")
    lay = QtWidgets.QVBoxLayout(dlg)
    lay.addWidget(QtWidgets.QLabel("Check all phases present in your observed assemblage:"))
    lay.addWidget(QtWidgets.QLabel("(Matching is case-insensitive)"))

    lw = QtWidgets.QListWidget()
    for name in all_names:
        item = QtWidgets.QListWidgetItem(name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Unchecked)
        lw.addItem(item)
    lay.addWidget(lw)

    btn_row = QtWidgets.QHBoxLayout()
    btn_all = QtWidgets.QPushButton("Select all")
    btn_none = QtWidgets.QPushButton("Clear all")
    btn_all.clicked.connect(lambda: [lw.item(i).setCheckState(QtCore.Qt.Checked)
                                     for i in range(lw.count())])
    btn_none.clicked.connect(lambda: [lw.item(i).setCheckState(QtCore.Qt.Unchecked)
                                      for i in range(lw.count())])
    btn_row.addWidget(btn_all); btn_row.addWidget(btn_none); btn_row.addStretch()
    lay.addLayout(btn_row)

    bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                    QtWidgets.QDialogButtonBox.Cancel)
    bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    dlg.resize(300, 400)

    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return [lw.item(i).text() for i in range(lw.count())
            if lw.item(i).checkState() == QtCore.Qt.Checked]
