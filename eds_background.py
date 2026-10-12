"""Background estimation for EDS band maps (Qt-free, hyperspy-free, numpy-only).

Every Bruker element map in this toolkit is stored as three energy windows of
equal width (default 150 eV): a *main* window on the K-alpha line and two
*sidebands* centred ``sideband_mult`` window widths (default 1.5, i.e. 225 eV)
below and above it. The net characteristic intensity is ``main - background``.

The original scheme (``"sideband"``) took the background as the mean of the
two sidebands regardless of what else lies there. For the Na-Mg-Al-Si-P
K-alpha lines (1.04, 1.25, 1.49, 1.74, 2.01 keV, i.e. 210-270 eV apart) a
sideband at +/-225 eV sits ON the neighbouring peak: Al's upper sideband holds
most of the Si K-alpha peak, Si's lower sideband holds Al, Mg's sidebands hold
Na and Al, P's lower sideband holds Si. The "background" then scales with the
neighbour, the net goes strongly negative in neighbour-rich phases, and
clipping the net at zero per pixel turns that into rectified noise of the
right sign, so nothing looked broken. On the C23S1A mosaic this cost the Al
calibration its predictive power (hold-out r2 = -0.19) and put a 2 wt% Al
deficit into every calibrated bulk (found 2026-09-16).

``"line-aware"`` (the default) inspects every window for the K-alpha, K-beta
and L-alpha lines of every mapped element and for the K-alpha pile-up (sum)
peaks, and only uses windows free of them: the mean of both sidebands when
both are clean, the clean one when only one is, and otherwise a linear
interpolation between the nearest clean sideband windows of *other* elements
below and above the line (on C23S1A: Na's lower sideband at 0.82 keV and P's
upper sideband at 2.24 keV serve Na, Mg, Al and Si). An element whose line is
not bracketed by clean windows keeps the legacy mean and is flagged
``approximate`` (O and C on a carbon-coated silicate). All windows have the
same width, so their counts are directly comparable and no rescaling is
needed. Nets are SIGNED: sum them as they are and clip only what has to be
non-negative (fractions, display images, per-pixel wt% maps) at the
aggregate, never per pixel before summing.

The scheme is a process-wide default (``default_scheme()``; env
``THERMOD_EDS_BACKGROUND`` overrides) so that legacy sessions can be replayed
with ``set_default_scheme("sideband")``; every function also takes an
explicit ``scheme=``.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Canonical K-alpha (L-alpha for the heavy tail) line table, shared by
# eds_map / viewer_helpers (both re-export it).
LINE_ENERGIES_KEV: Dict[str, float] = {
    'B': 0.183, 'C': 0.277, 'N': 0.392, 'O': 0.525,
    'F': 0.676, 'Na': 1.041, 'Mg': 1.254, 'Al': 1.486, 'Si': 1.740,
    'P': 2.013, 'S': 2.307, 'Cl': 2.622, 'K': 3.312, 'Ca': 3.691,
    'Ti': 4.511, 'V': 4.952, 'Cr': 5.415, 'Mn': 5.900, 'Fe': 6.404,
    'Co': 6.930, 'Ni': 7.478, 'Cu': 8.048, 'Zn': 8.638,
    'Ga': 9.251, 'Ge': 9.886, 'As': 10.543, 'Se': 11.222,
    'Br': 11.924, 'Kr': 12.657,
    'Ag': 2.984, 'Cd': 3.133, 'Sn': 3.443, 'Sb': 3.605, 'Te': 3.769,
    'Ba': 4.466, 'W': 8.398, 'Au': 9.713, 'Pb': 10.551, 'U': 13.614,
}

# Secondary lines that put counts into a window without ever being mapped
# themselves. Only used to judge whether a window is clean.
K_BETA_KEV: Dict[str, float] = {
    'Si': 1.836, 'P': 2.139, 'S': 2.464, 'Cl': 2.816, 'K': 3.590, 'Ca': 4.013,
    'Ti': 4.932, 'V': 5.427, 'Cr': 5.947, 'Mn': 6.490, 'Fe': 7.058,
    'Co': 7.649, 'Ni': 8.265, 'Cu': 8.905, 'Zn': 9.572,
}
L_ALPHA_KEV: Dict[str, float] = {
    'Ti': 0.452, 'V': 0.511, 'Cr': 0.573, 'Mn': 0.637, 'Fe': 0.705,
    'Co': 0.776, 'Ni': 0.851, 'Cu': 0.930, 'Zn': 1.012,
}

DEFAULT_WIDTH_EV = 150.0
DEFAULT_SIDEBAND_MULT = 1.5
# Windows reaching below this sit in the detector noise peak / cutoff and are
# never usable as a background estimate.
E_MIN_KEV = 0.2
# Pile-up (sum) peaks are only modelled for lines up to this energy: the
# light framework lines dominate a silicate spectrum, heavier lines are too
# weak for their sum peak to matter.
PILEUP_MAX_KEV = 2.5

SCHEMES = ("line-aware", "sideband")
_DEFAULT_SCHEME = os.environ.get("THERMOD_EDS_BACKGROUND", "line-aware")
if _DEFAULT_SCHEME not in SCHEMES:
    _DEFAULT_SCHEME = "line-aware"

BandGetter = Callable[[str, str], Optional[np.ndarray]]


def default_scheme() -> str:
    """Process-wide background scheme (``line-aware`` unless overridden)."""
    return _DEFAULT_SCHEME


def set_default_scheme(scheme: str) -> str:
    """Set the process-wide scheme (``"line-aware"`` | ``"sideband"``)."""
    global _DEFAULT_SCHEME
    scheme = str(scheme or "").strip().lower()
    if scheme not in SCHEMES:
        raise ValueError(f"unknown background scheme {scheme!r}; use one of {SCHEMES}")
    _DEFAULT_SCHEME = scheme
    return scheme


def sideband_centers(E_center_keV: float, width_eV: float = DEFAULT_WIDTH_EV,
                     mult: float = DEFAULT_SIDEBAND_MULT) -> Tuple[float, float]:
    """Centres (keV) of the two sidebands, ``mult`` window widths either side."""
    delta_keV = mult * width_eV / 1000.0
    return E_center_keV - delta_keV, E_center_keV + delta_keV


def _window(E_center_keV: float, width_eV: float) -> Tuple[float, float]:
    half = 0.5 * width_eV / 1000.0
    return E_center_keV - half, E_center_keV + half


def contaminant_lines(elements: Iterable[str],
                      lines: Optional[Dict[str, float]] = None,
                      ) -> List[Tuple[str, float]]:
    """Every line that can put counts into a window, as ``(label, keV)``.

    For each mapped element: its own K-alpha, its K-beta and L-alpha where
    tabulated, and, for the intense light lines below ``PILEUP_MAX_KEV``,
    the K-alpha pile-up (sum) peak at twice the line energy (the Si sum peak
    at 3.48 keV lands in K's upper and Ca's lower sideband on a quartz-rich
    section; sum peaks of the weaker high-energy lines are negligible).
    Labels read ``"Si"``, ``"Si Kb"``, ``"Fe La"``, ``"Si+Si"``.
    """
    lines = LINE_ENERGIES_KEV if lines is None else lines
    out: List[Tuple[str, float]] = []
    for el in elements:
        E = lines.get(el)
        if E is None:
            continue
        out.append((el, float(E)))
        if el in K_BETA_KEV:
            out.append((f"{el} Kb", K_BETA_KEV[el]))
        if el in L_ALPHA_KEV:
            out.append((f"{el} La", L_ALPHA_KEV[el]))
        if float(E) <= PILEUP_MAX_KEV:
            out.append((f"{el}+{el}", 2.0 * float(E)))
    return out


def background_plan(elements: Sequence[str],
                    width_eV: float = DEFAULT_WIDTH_EV,
                    sideband_mult: float = DEFAULT_SIDEBAND_MULT,
                    lines: Optional[Dict[str, float]] = None,
                    ) -> Dict[str, dict]:
    """Decide, per element, which stored windows estimate its background.

    ``elements`` should be EVERY element mapped for the dataset (the cache
    manifest's list), not just the ones being read: an element's line
    contaminates windows whether or not it is currently displayed.

    Returns ``{el: plan}`` with ``plan["mode"]`` one of ``"both"`` (mean of
    its own two sidebands), ``"lo"`` / ``"hi"`` (its one clean sideband),
    ``"interp"`` (linear interpolation between the nearest clean sideband
    windows of other elements below and above the line) or ``"legacy"``
    (both sidebands, nothing better available; ``plan["approximate"]`` is
    True). ``plan["anchors"]`` lists ``(element, band)`` pairs with matching
    ``plan["weights"]``; ``plan["dirty"]`` names the lines found in each of
    the element's own sidebands; ``plan["windows"]`` gives the (lo, hi) keV
    edges of its three windows.
    """
    lines = LINE_ENERGIES_KEV if lines is None else lines
    els = [e for e in elements if e in lines]
    contam = contaminant_lines(els, lines)
    win: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for e in els:
        Ec = float(lines[e])
        lo_c, hi_c = sideband_centers(Ec, width_eV, sideband_mult)
        win[e] = {"main": _window(Ec, width_eV), "sb_lo": _window(lo_c, width_eV),
                  "sb_hi": _window(hi_c, width_eV)}

    def hits(e: str, band: str) -> List[str]:
        lo, hi = win[e][band]
        found = []
        for lab, E in contam:
            if band == "main" and lab == e:
                continue      # its own line, by design
            if lo <= E <= hi:
                found.append(lab)
        return found

    def usable(e: str, band: str) -> bool:
        return win[e][band][0] >= E_MIN_KEV

    clean_cache: Dict[Tuple[str, str], bool] = {}

    def clean(e: str, band: str) -> bool:
        key = (e, band)
        if key not in clean_cache:
            clean_cache[key] = usable(e, band) and not hits(e, band)
        return clean_cache[key]

    clean_windows = []   # (centre_keV, element, band)
    for e in els:
        for band in ("sb_lo", "sb_hi"):
            if clean(e, band):
                lo, hi = win[e][band]
                clean_windows.append((0.5 * (lo + hi), e, band))

    plan: Dict[str, dict] = {}
    for e in els:
        Ec = float(lines[e])
        dirty = {b: hits(e, b) for b in ("sb_lo", "sb_hi")}
        c_lo, c_hi = clean(e, "sb_lo"), clean(e, "sb_hi")
        approximate = False
        if c_lo and c_hi:
            mode, anchors, weights = "both", [(e, "sb_lo"), (e, "sb_hi")], [0.5, 0.5]
        elif c_lo:
            mode, anchors, weights = "lo", [(e, "sb_lo")], [1.0]
        elif c_hi:
            mode, anchors, weights = "hi", [(e, "sb_hi")], [1.0]
        else:
            below = [cw for cw in clean_windows if cw[0] < Ec and cw[1] != e]
            above = [cw for cw in clean_windows if cw[0] > Ec and cw[1] != e]
            if below and above:
                b = max(below); a = min(above)
                w = (Ec - b[0]) / (a[0] - b[0])
                mode, anchors, weights = "interp", [(b[1], b[2]), (a[1], a[2])], [1.0 - w, w]
            else:
                mode, anchors, weights = "legacy", [(e, "sb_lo"), (e, "sb_hi")], [0.5, 0.5]
                approximate = True
        anchor_keV = [0.5 * sum(win[ae][ab]) for ae, ab in anchors]
        plan[e] = {
            "mode": mode, "anchors": anchors, "weights": weights,
            "anchor_keV": anchor_keV, "dirty": dirty, "approximate": approximate,
            "line_keV": Ec, "windows": dict(win[e]),
        }
    return plan


def anchor_elements(plan: Dict[str, dict], elements: Iterable[str]) -> List[str]:
    """Extra elements whose bands the plan needs for ``elements`` (sorted)."""
    want = set(elements)
    extra = set()
    for e in want:
        for ae, _ab in (plan.get(e) or {}).get("anchors", []):
            if ae not in want:
                extra.add(ae)
    return sorted(extra)


def describe_plan(plan: Dict[str, dict], scheme: Optional[str] = None) -> str:
    """One line per element for logs / the calibration dialog."""
    scheme = scheme or default_scheme()
    if scheme == "sideband":
        return "background: legacy sideband mean for every element"
    rows = []
    for e, p in plan.items():
        m = p["mode"]
        if m == "both":
            what = "mean of both sidebands"
        elif m in ("lo", "hi"):
            what = f"{'lower' if m == 'lo' else 'upper'} sideband only"
        elif m == "interp":
            (ae, ab), (be, bb) = p["anchors"]
            k0, k1 = p["anchor_keV"]
            what = f"interpolated between {ae} {ab} ({k0:.2f} keV) and {be} {bb} ({k1:.2f} keV)"
        else:
            what = "legacy sideband mean (APPROXIMATE: no clean window brackets the line)"
        dirty = "; ".join(f"{b} holds {', '.join(v)}" for b, v in p["dirty"].items() if v)
        rows.append(f"  {e:<2s} {p['line_keV']:.3f} keV: {what}" + (f"  [{dirty}]" if dirty else ""))
    return "background (line-aware):\n" + "\n".join(rows)


def net_count_map(el: str, get_band: BandGetter,
                  plan: Optional[Dict[str, dict]] = None,
                  scheme: Optional[str] = None,
                  clip: bool = False,
                  notes: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """Signed net-count array for one element.

    ``get_band(element, band)`` returns the window array (any shape, all
    identical) or ``None`` when that element/band is unavailable; this is how
    memmap slices, ROI-local dicts and sub-windows all share one code path.
    When an anchor of the plan is unavailable the element falls back to its
    own sidebands and a line is appended to ``notes``. Returns ``None`` when
    the element's main window is unavailable. ``clip=True`` floors the result
    at zero (display / per-pixel maps only; never before a sum).
    """
    scheme = scheme or default_scheme()
    main = get_band(el, "main")
    if main is None:
        return None
    main = np.asarray(main, dtype=np.float32)

    def own_sidebands() -> Optional[np.ndarray]:
        lo = get_band(el, "sb_lo"); hi = get_band(el, "sb_hi")
        if lo is None or hi is None:
            return None
        return 0.5 * (np.asarray(lo, dtype=np.float32) + np.asarray(hi, dtype=np.float32))

    bg: Optional[np.ndarray] = None
    if scheme == "sideband" or not plan or el not in plan:
        bg = own_sidebands()
    else:
        p = plan[el]
        acc = None
        ok = True
        for (ae, ab), w in zip(p["anchors"], p["weights"]):
            arr = get_band(ae, ab)
            if arr is None:
                ok = False
                break
            term = float(w) * np.asarray(arr, dtype=np.float32)
            acc = term if acc is None else acc + term
        if ok and acc is not None:
            bg = acc
        else:
            if notes is not None:
                notes.append(f"{el}: anchor bands unavailable, using its own sidebands")
            bg = own_sidebands()
    if bg is None:
        net = main.copy()
    else:
        net = main - bg
    if clip:
        np.maximum(net, 0, out=net)
    return net


def net_count_maps(band_maps: Dict[str, Dict[str, np.ndarray]],
                   scheme: Optional[str] = None,
                   plan: Optional[Dict[str, dict]] = None,
                   clip: bool = False,
                   elements: Optional[Iterable[str]] = None,
                   sub_bbox: Optional[Tuple[int, int, int, int]] = None,
                   width_eV: float = DEFAULT_WIDTH_EV,
                   sideband_mult: float = DEFAULT_SIDEBAND_MULT,
                   notes: Optional[List[str]] = None,
                   ) -> Dict[str, np.ndarray]:
    """Signed net-count maps ``{el: array}`` for an ROI-local band-map dict.

    ``band_maps`` is ``{el: {'main', 'sb_lo', 'sb_hi'}}`` as produced by
    ``roi_to_banded_maps`` / ``roi_from_cache``. ``plan`` defaults to
    :func:`background_plan` over the dict's own elements (pass the plan built
    from the dataset's full element list when the dict holds a subset).
    ``sub_bbox`` = ``(y0, y1, x0, x1)`` slices every band before use.
    """
    if not band_maps:
        return {}
    if plan is None and (scheme or default_scheme()) != "sideband":
        plan = background_plan(list(band_maps), width_eV=width_eV, sideband_mult=sideband_mult)
    if sub_bbox is not None:
        y0, y1, x0, x1 = (int(v) for v in sub_bbox)

        def get_band(e: str, b: str):
            arr = (band_maps.get(e) or {}).get(b)
            return None if arr is None else arr[y0:y1, x0:x1]
    else:
        def get_band(e: str, b: str):
            return (band_maps.get(e) or {}).get(b)
    out: Dict[str, np.ndarray] = {}
    for el in (elements if elements is not None else list(band_maps)):
        net = net_count_map(el, get_band, plan=plan, scheme=scheme, clip=clip, notes=notes)
        if net is not None:
            out[el] = net
    return out
