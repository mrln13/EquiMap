#!/usr/bin/env python
"""Figure: composition as a function of length scale (the equilibration-volume tool).

Reads the per-window CSV exported by EquiMap's *Composition vs scale …* dialog
("Export CSV …"), one row per quadtree window:

    level, y0, y1, x0, x1, n_pixels, mask_frac, skipped, <El1>, <El2>, ...

where the trailing columns are each window's composition (wt% for quantified /
SPX-calibrated data). Produces a three-panel publication figure:

  (a) between-window spread (sample s.d.) of the main elements vs. window size,
      divided by each element's calibration hold-out error (the tolerance,
      eds_calibration.json; dashed line at 1): the representative averaging
      scale is the window size at which every curve is at or below the line.
      The spread never plateaus, which is why a tolerance is needed
      (co-author comment, 2026-10-28). Raw wt% when no calibration JSON;
  (b) empirical semivariogram gamma(h) at the finest well-populated level,
      elements standardised and averaged (sill 1 by construction), over all
      window pairs and within +-22.5 deg of the banding (--banding-deg) and of
      its normal; dotted lines where each reaches 95 % of the sill;
  (c) per-window P-T inversions: overlap of each window's accepted region with
      the section's, vs. window size (pt_windows.csv).

If the CSV is absent the script generates a synthetic spatially-correlated
dataset (so the semivariogram is meaningful) and watermarks the figure
"SYNTHETIC DEMO". Replace by exporting the real CSV and re-running.

Usage:
    python fig_scale.py [--csv composition_scale.csv] [--out fig_scale.pdf]
                        [--dpi 300] [--pixel-um PIXEL_UM] [--top 4]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (ScalarFormatter, NullFormatter, NullLocator,
                               FuncFormatter, FormatStrFormatter)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import element_colour, ELEMENT_ORDER as _ELEMENT_ORDER  # noqa: E402

_META_COLS = {"level", "y0", "y1", "x0", "x1", "n_pixels", "mask_frac", "skipped"}


def _synthetic(seed: int = 1) -> pd.DataFrame:
    """Quadtree windows over a spatially-correlated composition field.

    A few smooth blobs give a genuine correlation length, so the semivariogram
    rises to a sill; fine windows resolve the heterogeneity, coarse ones average
    it out.
    """
    rng = np.random.default_rng(seed)
    size = 4096                       # ROI side in px
    base = {"Si": 24.0, "Al": 9.0, "Fe": 8.0, "Mg": 7.5, "Ca": 6.0, "Na": 1.8}
    # Smooth field: sum of Gaussian blobs per element (different centres/signs).
    blobs = {el: [(rng.uniform(0, size), rng.uniform(0, size),
                   rng.uniform(size * 0.04, size * 0.10),
                   rng.normal(0, base[el] * 0.55)) for _ in range(8)]
             for el in base}

    def field(el, cx, cy):
        v = base[el]
        for bx, by, r, amp in blobs[el]:
            v += amp * np.exp(-((cx - bx) ** 2 + (cy - by) ** 2) / (2 * r ** 2))
        return max(0.0, v + rng.normal(0, base[el] * 0.04))

    rows = []
    for level in range(5):            # 0..4 -> 1,4,16,64,256 windows
        n = 2 ** level
        step = size / n
        for iy in range(n):
            for ix in range(n):
                x0, y0 = ix * step, iy * step
                x1, y1 = x0 + step, y0 + step
                cx, cy = x0 + step / 2, y0 + step / 2
                row = {"level": level, "y0": int(y0), "y1": int(y1),
                       "x0": int(x0), "x1": int(x1),
                       "n_pixels": int(step * step), "mask_frac": 1.0,
                       "skipped": ""}
                for el in base:
                    row[el] = field(el, cx, cy)
                rows.append(row)
    return pd.DataFrame(rows)


def _element_columns(df: pd.DataFrame) -> list[str]:
    els = [c for c in df.columns if c not in _META_COLS
           and pd.api.types.is_numeric_dtype(df[c])]
    known = [e for e in _ELEMENT_ORDER if e in els]
    rest = sorted(e for e in els if e not in _ELEMENT_ORDER)
    return known + rest


# P-T-vs-scale CSV schema (panel c). One row per quadtree level / window size:
#   window_px, T_best_C, T_lo_C, T_hi_C, P_best_GPa, P_lo_GPa, P_hi_GPa
# T_best/P_best = estimate for the level's MEAN composition; lo/hi = the spread
# from the mean+/-sigma composition variants (so the band widens at small windows).
PT_COLUMNS = ["window_px", "T_best_C", "T_lo_C", "T_hi_C",
              "P_best_GPa", "P_lo_GPa", "P_hi_GPa"]


def _synthetic_pt(side_by_level, seed: int = 2) -> pd.DataFrame:
    """A plausible P-T-vs-scale table for layout preview (panel c).

    Roughly scale-invariant best-fit P-T with an envelope that widens as the
    window shrinks (the equilibration-volume signal): the +/- sigma composition
    variants disagree more at fine scale.
    """
    rng = np.random.default_rng(seed)
    sizes = sorted(side_by_level.values(), reverse=True)  # large -> small
    smax = max(sizes) if sizes else 1.0
    rows = []
    for sz in sizes:
        frac = (smax / sz) ** 0.6              # grows as the window shrinks
        tw, pw = 7.0 * frac, 0.035 * frac      # half-widths in degC / GPa
        T = 650.0 + float(rng.normal(0, 1.5))
        P = 1.00 + float(rng.normal(0, 0.004))
        rows.append({"window_px": sz,
                     "T_best_C": T, "T_lo_C": T - tw, "T_hi_C": T + tw,
                     "P_best_GPa": P, "P_lo_GPa": P - pw, "P_hi_GPa": P + pw})
    return pd.DataFrame(rows)


def _window_side(df: pd.DataFrame, pixel_um: float | None) -> pd.Series:
    side_px = np.sqrt((df["y1"] - df["y0"]).astype(float) * (df["x1"] - df["x0"]).astype(float))
    return side_px * pixel_um if pixel_um else side_px


def _semivariogram(cx, cy, z, n_bins=10, max_lag_frac=0.55, max_points=1500,
                   seed=0, direction_deg=None, tol_deg=22.5, max_lag=None,
                   min_pairs=30):
    """Classic empirical semivariogram, omnidirectional or within a sector.

    Only lags up to ``max_lag`` (same units as cx/cy) or, when that is None,
    ``max_lag_frac`` of the maximum separation are reported: beyond about half
    the domain the pair count collapses and gamma(h) is unreliable (the
    standard rule of thumb). ``direction_deg`` keeps only the pairs whose
    separation lies within ``tol_deg`` of that direction (degrees from +x
    towards +y, modulo 180), e.g. along and across a banding. Bins with fewer
    than ``min_pairs`` pairs are NaN.

    The pairwise computation is O(N^2) in windows, so when a level has more than
    ``max_points`` windows they are seeded-subsampled first — otherwise a deep
    quadtree (thousands of windows) would build billions of pairs and hang.
    """
    cx, cy = np.asarray(cx, dtype=float), np.asarray(cy, dtype=float)
    Z = np.asarray(z, dtype=float)
    Z = Z if Z.ndim == 2 else Z[:, None]   # (N,) single element or (N,k) combined
    ok = np.isfinite(cx) & np.isfinite(cy) & np.all(np.isfinite(Z), axis=1)
    cx, cy, Z = cx[ok], cy[ok], Z[ok]
    if Z.shape[0] < 6:
        return None, None
    if Z.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(Z.shape[0], size=max_points,
                                                 replace=False)
        cx, cy, Z = cx[idx], cy[idx], Z[idx]
    i, j = np.triu_indices(Z.shape[0], k=1)
    dx, dy = cx[i] - cx[j], cy[i] - cy[j]
    h = np.sqrt(dx ** 2 + dy ** 2)
    d = 0.5 * np.mean((Z[i] - Z[j]) ** 2, axis=1)  # mean over elements; sill ~ 1
    h_max = float(max_lag) if max_lag else h.max() * max_lag_frac
    keep = h <= h_max
    if direction_deg is not None:
        ang = np.degrees(np.arctan2(dy, dx)) % 180.0
        dev = np.abs(ang - (float(direction_deg) % 180.0))
        keep &= np.minimum(dev, 180.0 - dev) <= float(tol_deg)
    h, d = h[keep], d[keep]
    if h.size < 6:
        return None, None
    edges = np.linspace(0, h_max * 1.001, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (h >= edges[b]) & (h < edges[b + 1])
        if m.sum() >= min_pairs:
            gamma[b] = d[m].mean()
    return centres, gamma


def _phase_domain_band(path: str):
    """(smallest, largest) phase-domain size in micrometres from
    ``phase_domain_sizes.csv`` (phase_domain_sizes.py): the area-weighted
    median equivalent diameters of the phases scored as modal proportions.
    None when the file is absent. The band is measured on the phase map,
    independently of the per-window curve it is drawn against.
    """
    if not path or not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d["scored"] == 1]
    if not len(d):
        return None
    return float(d["d50_um"].min()), float(d["d50_um"].max())


def make_figure(df: pd.DataFrame, out: str, dpi: int, pixel_um: float | None,
                top: int, synthetic: bool, vario_elements: int = 0,
                pt_df: pd.DataFrame | None = None, vario_bins: int = 24,
                pt_df2: pd.DataFrame | None = None,
                pt_win: pd.DataFrame | None = None,
                domain_band: tuple | None = None,
                tolerance: dict | None = None,
                banding_deg: float = 0.0,
                vario_max_lag: float | None = None) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "savefig.bbox": "tight",
    })
    # Keep only windows with no skip reason. NaN-safe: pandas >=3 preserves NA
    # through astype(str) rather than yielding "nan", so an empty (NaN) skipped
    # field must be treated as "not skipped" or every real window is dropped.
    if "skipped" in df.columns:
        reason = df["skipped"].astype("string").fillna("").str.strip()
        df = df[reason.eq("")].copy()
    else:
        df = df.copy()
    elements = _element_columns(df)
    # Rank by overall mean; keep the strongest `top`.
    ranking = {el: float(np.nanmean(df[el])) for el in elements}
    top_els = sorted((e for e in elements if ranking[e] > 0),
                     key=lambda e: -ranking[e])[:top]
    df["_side"] = _window_side(df, pixel_um)
    unit = "um" if pixel_um else "px"
    colours = {el: element_colour(el) for el in top_els}

    # Three panels: (a) and (b) side by side, (c) full width below them. A
    # fourth panel, the 10-90 % spread of the windows' best-fit cells, was
    # dropped on 2026-10-27 (co-author comment: a best cell on a flat misfit
    # valley is grid quantisation; a width-of-region version added nothing that
    # panel (c) does not show). A 1 x 3 row was tried and rejected: at the
    # 5.4-inch text width each panel came out under two inches wide.
    fig = plt.figure(figsize=(7.4, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # ── (a) composition vs window size (mean +/- s.d. band) ─────────────────
    # The spread keeps falling as the window grows and never plateaus, so the
    # representative averaging scale needs a tolerance (co-author comment,
    # 2026-10-28). With `tolerance` (per-element wt%, the calibration's hold-out
    # RMSE from eds_calibration.json) each curve is spread / tolerance and one
    # dashed line at 1 marks it: the representative scale is where every curve
    # is at or below the line. Without a tolerance the raw wt% spread is drawn.
    # Sample s.d. (ddof=1), the convention the text quotes.
    levels = sorted(df["level"].unique())
    side_by_level = {lv: float(df[df["level"] == lv]["_side"].mean()) for lv in levels}
    if pt_df is None and synthetic:          # placeholder P-T-vs-scale for panel (c)
        pt_df = _synthetic_pt(side_by_level)
    use_tol = bool(tolerance)
    for el in top_els:
        std_l, side_l = [], []
        tol = (tolerance or {}).get(el)
        if use_tol and not (tol and tol > 0):
            continue                         # no error estimate: not testable
        for lv in levels:
            sub = df[df["level"] == lv][el].dropna()
            if sub.size < 2:                 # need >=2 windows for a spread
                continue
            sd = float(sub.std(ddof=1))
            std_l.append(sd / tol if use_tol else sd)
            side_l.append(side_by_level[lv])
        if side_l:
            axA.plot(side_l, std_l, "-o", ms=3, color=colours[el], label=el, lw=1.0)
            if use_tol:
                print(f"(a) {el}: spread / error per level "
                      + " ".join(f"{s:.2f}" for s in std_l))
    if use_tol:
        axA.axhline(1.0, color="0.4", ls="--", lw=0.9)
        axA.set_yscale("log")
        axA.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        axA.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        axA.yaxis.set_minor_formatter(NullFormatter())
    # (A 1/L reference for uncorrelated pixels was tried on a log y-axis and
    # dropped, 2026-10-07: the observed spread falls about half as fast, a point
    # the paper does not use.)
    axA.set_xscale("log")
    axA.invert_xaxis()                       # large windows left -> small right
    axA.set_xlabel(f"window size ({unit})")
    axA.set_ylabel("spread / calibration error" if use_tol
                   else "between-window s.d. (wt%)")
    axA.set_title("(a) compositional spread vs. scale")
    axA.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.3,
               columnspacing=0.8)

    # ── (b) combined compositional semivariogram at the finest level ────────
    # Per-element semivariograms have wildly different magnitudes (a major
    # element dwarfs the minors), which is hard to read. Instead each element is
    # standardised (z-score) and ONE combined curve is computed and normalised to
    # its sill, so it shows the correlation length of the overall composition.
    counts = df.groupby("level").size()
    fine = max((lv for lv in levels if counts.get(lv, 0) >= 6), default=None)
    drew_variogram = False
    if fine is not None:
        sub = df[df["level"] == fine]
        cx = (sub["x0"] + sub["x1"]).values / 2.0
        cy = (sub["y0"] + sub["y1"]).values / 2.0
        if pixel_um:
            cx, cy = cx * pixel_um, cy * pixel_um
        cols, std_elems = [], []
        for el in top_els:
            v = sub[el].values.astype(float)
            sd = np.nanstd(v)
            if sd > 0:
                cols.append((v - np.nanmean(v)) / sd)
                std_elems.append(el)
        if cols:
            Z = np.column_stack(cols)
            # The variables are standardised, so the sill is 1 by construction.
            # (Until 2026-10-28 the sill was taken as the maximum of the binned
            # curve, a noise peak of about 1.06 with 0.76 mm bins and a 1500-
            # window subsample; the 95 % rule then found where the curve reached
            # its noise ceiling, 2.7 mm, instead of 95 % of the variance.)
            SILL = 1.0

            def _crossing(ce, ge, frac=0.95):
                ok = np.isfinite(ge)
                hit = np.where(ok & (ge >= frac * SILL))[0]
                return float(ce[hit[0]]) if len(hit) else None

            def _fmt_len(a):
                return (f"{a / 1000:.1f} mm" if unit == "um" and a >= 100
                        else f"{a:.0f} {unit}")

            # Omnidirectional plus the two directions of a layered sample:
            # along the banding (banding_deg from +x) and across it.
            sectors = [("all directions", None, "0.1", "-o"),
                       ("along banding", float(banding_deg), "C3", "-s"),
                       ("across banding", float(banding_deg) + 90.0, "C0", "-^")]
            drew_any = False
            for lab, sec, col, style in sectors:
                ce, ge = _semivariogram(cx, cy, Z, n_bins=vario_bins,
                                        direction_deg=sec, max_lag=vario_max_lag,
                                        max_points=20000)
                if ce is None:
                    continue
                a = _crossing(ce, ge)
                axB.plot(ce, ge, style, ms=2.8, lw=1.1, color=col,
                         label=(f"{lab}, $a$ = {_fmt_len(a)}" if a else lab),
                         zorder=5 if sec is None else 4)
                if a:
                    axB.axvline(a, color=col, ls=":", lw=0.8)
                    print(f"(b) {lab}: 95 % of the sill at {_fmt_len(a)}")
                drew_any = True
            # Optional individual element curves (faint, omnidirectional).
            for el, zc in list(zip(std_elems, cols))[:max(0, int(vario_elements))]:
                ce, ge = _semivariogram(cx, cy, zc, n_bins=vario_bins,
                                        max_lag=vario_max_lag, max_points=20000)
                if ce is not None:
                    axB.plot(ce, ge, "-", lw=0.8, alpha=0.7,
                             color=colours.get(el, "0.6"), label=el)
            if drew_any:
                axB.axhline(0.95 * SILL, color="0.6", ls="--", lw=0.7)
                axB.margins(y=0.12)   # autoscale to the data range (no forced 0)
                axB.set_xlabel(f"lag distance $h$ ({unit})")
                axB.set_ylabel(r"standardised $\gamma(h)$")
                sz = side_by_level[fine]
                sz_txt = (f"{sz / 1000:.2f} mm" if unit == "um" and sz >= 1000
                          else f"{sz:.0f} µm" if unit == "um" else f"{sz:.0f} px")
                axB.set_title(f"(b) semivariogram at {sz_txt} windows")
                axB.legend(loc="lower right", frameon=False, fontsize=6.5,
                           handletextpad=0.4)
                drew_variogram = True
    if not drew_variogram:
        axB.text(0.5, 0.5, "too few windows\nfor a semivariogram",
                 ha="center", va="center", transform=axB.transAxes, fontsize=8)
        axB.set_title("(b) compositional semivariogram")

    # ── (c) P-T estimate vs scale ───────
    # The payoff: how the inferred P-T estimate's envelope widens as the
    # averaging window shrinks (the equilibration-volume signal). T on the left
    # axis, P on the right; the bands are the spread from the +/- sigma
    # composition variants. Needs the thermodynamic sensitivity sweep export
    # (pt_scale.csv); see _synthetic_pt for the column schema.
    scale = (pixel_um if pixel_um else 1.0)
    tcol, pcol, p2col = "0.45", "#1f77b4", "#d62728"
    if pt_win is not None and len(pt_win):
        # PER-WINDOW inversions (window_pt_run.py): every sampled window is
        # inverted with its OWN calibrated bulk and its OWN modes read from
        # the phase map inside it. The panel shows ONE quantity against
        # window size: the fraction of a window's accepted P-T region that
        # lies inside the section's accepted region (median line, inter-
        # quartile band). A representative window scores ~1; below the
        # representative averaging scale the windows drift, lose phases and
        # the overlap declines slowly, within the window-to-window scatter,
        # down to ~0.4 mm and then drops once windows no longer sample the
        # whole assemblage (the phase-domain scale). The compositional
        # correlation length of panel (b) leaves no mark on the curve: each
        # window's bulk and modes co-vary, so the inversion tolerates the
        # bulk drift. The shaded interval is the range of PHASE-DOMAIN
        # SIZES on the phase map (phase_domain_sizes.csv), measured
        # independently of this curve: the overlap is lost as windows
        # shrink through it. (An onset derived from the curve itself moved
        # by a factor of two between 12 and 36 windows per size, on a
        # knife-edge between discrete overlap values, and was dropped.)
        w = pt_win[pt_win["skipped"].isna() | (pt_win["skipped"] == "")].copy()
        # same size convention as panel (a): sqrt(window area), not the height
        # (the windows keep the ROI's aspect ratio, 1.85:1 on the worked example)
        w["side"] = np.sqrt((w["y1"] - w["y0"]).astype(float)
                            * (w["x1"] - w["x0"]).astype(float)) * scale
        g = w.groupby("level")
        sizes = g["side"].mean().sort_index()
        pos = sizes.to_numpy(float)
        q25 = g["overlap_with_section"].quantile(0.25).reindex(sizes.index).to_numpy(float)
        q50 = g["overlap_with_section"].median().reindex(sizes.index).to_numpy(float)
        q75 = g["overlap_with_section"].quantile(0.75).reindex(sizes.index).to_numpy(float)
        n_win = g.size().reindex(sizes.index).to_numpy(int)
        axC.set_xscale("log")
        axC.invert_xaxis()
        axC.set_xlabel(f"window size ({unit})")
        axC.fill_between(pos, q25, q75, color="0.45", alpha=0.25, lw=0, zorder=1)
        # every window as a faint dot, so the distribution behind the median
        # and the band is visible (a small seeded jitter in x separates ties)
        jit = np.exp(np.random.default_rng(0).normal(0.0, 0.045, size=len(w)))
        xs = w["level"].map(sizes).to_numpy(float) * jit
        if "overlap_own" in w.columns:
            ocol = "#d95f02"
            hs = w["has_sill"].to_numpy(int) == 1
            yo = w["overlap_own"].to_numpy(float)
            dot_s = axC.scatter(xs[hs], yo[hs], s=9, color=ocol, alpha=0.55, lw=0, zorder=2,
                                label="window containing sillimanite")
            dot_n = axC.scatter(xs[~hs], yo[~hs], s=9, facecolors="none", edgecolors=ocol,
                                alpha=0.7, lw=0.6, zorder=2, label="window without sillimanite")
        else:
            axC.scatter(xs, w["overlap_with_section"].to_numpy(float),
                        s=7, color="0.25", alpha=0.35, lw=0, zorder=2)
        line_sec, = axC.plot(pos, q50, "-o", color="0.1", ms=3.5, lw=1.3, zorder=3,
                             label="section's constraint in every window")
        handles = [line_sec]
        if "overlap_own" in w.columns:
            # second scoring (window_pt_run.py): the window's OWN assemblage,
            # sillimanite required only where the window contains it
            ocol = "#d95f02"
            o25 = g["overlap_own"].quantile(0.25).reindex(sizes.index).to_numpy(float)
            o50 = g["overlap_own"].median().reindex(sizes.index).to_numpy(float)
            o75 = g["overlap_own"].quantile(0.75).reindex(sizes.index).to_numpy(float)
            axC.fill_between(pos, o25, o75, color=ocol, alpha=0.18, lw=0, zorder=1)
            line_own, = axC.plot(pos, o50, "-s", color=ocol, ms=3.2, lw=1.3, zorder=3,
                                 label="window's own assemblage")
            handles += [line_own, dot_s, dot_n]
        # legend in the empty lower-left corner of the wide panel (direct text
        # labels were tried and read as unattached once the panel was widened);
        # the lines are the per-size medians, the bands their interquartile
        # ranges (caption), the dots the individual windows of the own scoring
        axC.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.5,
                   handlelength=2.4, borderaxespad=0.5, labelspacing=0.5)

        if domain_band is not None:
            lo, hi = domain_band       # smallest / largest phase-domain size (um)
            axC.axvspan(lo, hi, color="0.5", alpha=0.13, lw=0, zorder=0)
            axC.text(hi, 0.03, f"phase-domain\nsizes\n{lo / 1000:.2f} to {hi / 1000:.2f} mm ",
                     ha="right", va="bottom", fontsize=6.5, color="0.3", linespacing=1.1)
            print(f"[fig_scale] phase-domain sizes of the scored phases: {lo:.0f} to {hi:.0f} um")
        axC.set_ylim(0, 1.05)
        axC.set_ylabel("overlap of window's accepted\nP-T region with the section's")
    elif pt_df is not None and len(pt_df):
        s = pt_df.sort_values("window_px", ascending=False)
        wpx = s["window_px"].to_numpy(dtype=float) * scale
        cols = set(s.columns)
        # One quantity per axis: the ACCEPTED REGION (union over the
        # {mean, ±σ} composition variants of each grid's near-minimal cells),
        # i.e. the range a single window of that size would give. The
        # calibrated-map sweep is the filled band; the SPX-source sweep
        # (tier-3 cross-check, possible down to the tile size) the dashed
        # outline in the matching colour. Best cells, the ±σ variants' best
        # cells and the mean variant's own region are deliberately NOT drawn
        # (they are in pt_scale*.csv and the viewer tab): on a flat misfit
        # valley the best cell is arbitrary, and the mean variant's region is
        # the section bulk's region at every scale (a constant).
        tdash, pdash = "0.1", "#0b4f8a"
        if {"T_lo_C", "T_hi_C"} <= cols:
            axC.fill_between(wpx, s["T_lo_C"].to_numpy(float),
                             s["T_hi_C"].to_numpy(float), color=tcol,
                             alpha=0.30, lw=0)
        axC.set_xscale("log")
        axC.invert_xaxis()
        axC.set_xlabel(f"window size ({unit})")
        axC.set_ylabel("T estimate (°C)")
        axC2 = None
        if {"P_lo_GPa", "P_hi_GPa"} <= cols:
            axC2 = axC.twinx()
            axC2.fill_between(wpx, s["P_lo_GPa"].to_numpy(float),
                              s["P_hi_GPa"].to_numpy(float),
                              color=pcol, alpha=0.18, lw=0)
            axC2.set_ylabel("P estimate (GPa)", color=pcol, fontsize=8)
            axC2.tick_params(axis="y", colors=pcol, labelsize=7)
        if pt_df2 is not None and len(pt_df2):
            s2 = pt_df2.sort_values("window_px", ascending=False)
            w2 = s2["window_px"].to_numpy(dtype=float) * scale
            if {"T_lo_C", "T_hi_C"} <= set(s2.columns):
                for k in ("T_lo_C", "T_hi_C"):
                    axC.plot(w2, s2[k].to_numpy(float), "--", color=tdash,
                             lw=1.0)
            if axC2 is not None and {"P_lo_GPa", "P_hi_GPa"} <= set(s2.columns):
                for k in ("P_lo_GPa", "P_hi_GPa"):
                    axC2.plot(w2, s2[k].to_numpy(float), "--", color=pdash,
                              lw=1.0)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        handles = [Patch(facecolor=tcol, alpha=0.5, label="calibrated maps (filled)")]
        if pt_df2 is not None and len(pt_df2):
            handles.append(Line2D([], [], color=tdash, ls="--", lw=1.0,
                                  label="SPX quantifications (dashed)"))
        axC.legend(handles=handles, loc="lower left", frameon=True,
                   framealpha=0.85, edgecolor="none", fontsize=6,
                   handlelength=1.8, borderpad=0.4, title="accepted P–T region",
                   title_fontsize=6)
    else:
        axC.set_xscale("log")
        axC.invert_xaxis()
        axC.set_xlabel(f"window size ({unit})")
        axC.text(0.5, 0.5, "run the sensitivity sweep\nand export pt_scale.csv",
                 ha="center", va="center", transform=axC.transAxes,
                 fontsize=7.5, color="0.45")
    axC.set_title("(c) P-T estimate vs. scale")

    # Clean discrete window-size ticks on the two log axes (a) and (c). With many
    # quadtree levels the per-level sizes crowd the axis, so label at most ~6
    # (always keeping the extremes).
    sizes = sorted({round(s, 1) for s in side_by_level.values()})
    if len(sizes) > 6:
        step = int(np.ceil(len(sizes) / 6))
        ticks = sizes[::step]
        if sizes[-1] not in ticks:
            ticks.append(sizes[-1])
    else:
        ticks = sizes
    # label lengths in mm when the windows span millimetres (5-digit micron
    # labels collide on the narrow panels); (b)'s lag axis follows suit
    use_mm = unit == "um" and max(sizes) >= 1000
    xunit = "mm" if use_mm else unit

    def _len_label(v):
        if not use_mm:
            return f"{int(round(v))}"
        v = v / 1000.0
        return f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")

    for ax in (axA, axC):
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())   # no unlabeled minor tick marks
        ax.set_xticklabels([_len_label(s) for s in ticks])
        ax.set_xlabel(f"window size ({xunit})")
    if use_mm and drew_variogram:
        axB.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000.0:g}"))
        axB.set_xlabel("lag distance $h$ (mm)")

    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DEMO", fontsize=34, color="0.85",
                 ha="center", va="center", rotation=18, zorder=0, alpha=0.6)

    fig.tight_layout(w_pad=1.4, h_pad=1.6)
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}  ({len(df)} windows, levels {levels}, "
          f"elements {top_els}" + ("; SYNTHETIC" if synthetic else "") + ")")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "composition_scale.csv"),
                    help="per-window CSV from the Composition-vs-scale dialog")
    ap.add_argument("--out", default=os.path.join(here, "fig_scale.pdf"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pixel-um", type=float, default=None,
                    help="px size in microns; if given, axes are in um not px")
    ap.add_argument("--top", type=int, default=4    , help="elements to plot")
    ap.add_argument("--variogram-elements", type=int, default=0,
                    help="individual element curves to overlay in panel (b) "
                         "alongside the combined curves (0 = combined only)")
    ap.add_argument("--variogram-bins", type=int, default=24,
                    help="number of lag bins in the semivariogram panel (b)")
    ap.add_argument("--variogram-max-lag", type=float, default=6000.0,
                    help="largest lag drawn in panel (b), in the length unit of "
                         "the axes (um when --pixel-um is given, else px); 24 "
                         "bins to 6000 um = 0.25 mm bins (0 = 0.55 of the "
                         "largest separation, the old rule)")
    ap.add_argument("--banding-deg", type=float, default=0.0,
                    help="direction of the banding/foliation trace in degrees "
                         "from the +x (long-side) axis towards +y; panel (b) "
                         "draws the semivariogram along and across it (+-22.5 deg)")
    ap.add_argument("--calibration-json", default=os.path.join(here, "eds_calibration.json"),
                    help="calibrate_mosaic.py output; its hold-out per-element "
                         "RMSE is the tolerance of panel (a) (spread / error, "
                         "line at 1); '' or a missing file draws the raw wt% spread")
    ap.add_argument("--pt-csv", default=os.path.join(here, "pt_scale.csv"),
                    help="P-T-vs-scale CSV for panel (c) "
                         f"(columns {PT_COLUMNS}; optional *_mean_* and "
                         "*_best_plus/minus_* columns draw the inner band and "
                         "the ±σ best cells); from the sensitivity sweep")
    ap.add_argument("--pt-csv-spx", default=os.path.join(here, "pt_scale_spx.csv"),
                    help="optional second P-T-vs-scale CSV (the SPX-source "
                         "sweep) overlaid in panel (c) as dashed outlines; "
                         "skipped when absent")
    ap.add_argument("--domain-sizes", default=os.path.join(here, "phase_domain_sizes.csv"),
                    help="phase-domain sizes (phase_domain_sizes.py); the range over the "
                         "scored phases is shaded in panel (c) ('' to disable)")
    ap.add_argument("--pt-windows", default=os.path.join(here, "pt_windows.csv"),
                    help="per-window inversions (window_pt_run.py); when present "
                         "panel (c) shows their per-level distributions instead of "
                         "the ±σ union bands ('' to disable)")
    args = ap.parse_args()

    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        synthetic = False
    else:
        print(f"note: {args.csv} not found — using synthetic demonstration data.")
        df = _synthetic()
        synthetic = True

    pt_df = None
    if os.path.exists(args.pt_csv):
        pt_df = pd.read_csv(args.pt_csv)
    elif not synthetic:
        print(f"note: {args.pt_csv} not found — panel (c) will prompt to run the "
              "sensitivity sweep.")

    pt_df2 = pd.read_csv(args.pt_csv_spx) if os.path.exists(args.pt_csv_spx) else None

    pt_win = (pd.read_csv(args.pt_windows)
              if args.pt_windows and os.path.exists(args.pt_windows) else None)
    tolerance = None
    if args.calibration_json and os.path.exists(args.calibration_json):
        import json
        with open(args.calibration_json) as fh:
            cal = json.load(fh)
        per_el = ((cal.get("holdout") or {}).get("per_element") or {})
        tolerance = {el: float(v["rmse"]) for el, v in per_el.items()
                     if v.get("rmse") is not None and float(v["rmse"]) > 0}
        if tolerance:
            print("(a) tolerance = hold-out RMSE (wt%): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in sorted(tolerance.items())))
    elif not synthetic:
        print(f"note: {args.calibration_json} not found — panel (a) shows the raw spread.")
    make_figure(df, args.out, args.dpi, args.pixel_um, args.top, synthetic,
                vario_elements=args.variogram_elements, pt_df=pt_df,
                vario_bins=args.variogram_bins, pt_df2=pt_df2, pt_win=pt_win,
                domain_band=(_phase_domain_band(args.domain_sizes) if args.pixel_um else None),
                tolerance=tolerance, banding_deg=args.banding_deg,
                vario_max_lag=(args.variogram_max_lag or None))


if __name__ == "__main__":
    main()
