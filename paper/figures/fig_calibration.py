#!/usr/bin/env python
"""Figure: SPX-anchored EDS calibration — hold-out validation.

Reads the long-format CSV exported by EquiMap's *Calibrate EDS (SPX k-factors)*
dialog ("Export validation CSV …"), which contains one row per held-out
tile x element with columns:

    tile, element, predicted_wt_pct, measured_wt_pct, residual_wt_pct

where ``predicted`` is the bulk wt% predicted for a tile from k-factors fitted on
the *other* tiles (leave-one-tile-out) and ``measured`` is that tile's own SPX
wt%. Produces a three-panel publication figure:

  (a) predicted vs. measured wt% for every held-out tile x element, with the 1:1
      line — the headline agreement;
  (b) per-element root-mean-square error (how large the residuals are);
  (c) per-element bias = mean(predicted - measured) (systematic over/under-call).

If the CSV is absent the script generates a synthetic demonstration dataset so the
layout can be previewed; the figure is then watermarked "SYNTHETIC DEMO". Replace
it by exporting the real CSV from the viewer and re-running.

Usage:
    python fig_calibration.py [--csv calibration_validation.csv]
                              [--out fig_calibration.pdf] [--dpi 300]
"""
from __future__ import annotations


import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import element_colour, order_elements as _order_elements  # noqa: E402

COLUMNS = ["tile", "element", "predicted_wt_pct", "measured_wt_pct"]


def _synthetic(seed: int = 0, n_folds: int = 12) -> pd.DataFrame:
    """A plausible repeated hold-out validation table for layout preview.

    Each tile is predicted in several folds (different train subsets), so the
    per-fold spread that drives the error bars is present.
    """
    rng = np.random.default_rng(seed)
    elements = {            # nominal mean wt%, per-element systematic bias, scatter
        "Si": (24.0, 0.3, 0.9), "Al": (9.0, -0.2, 0.6), "Fe": (8.0, 0.4, 0.7),
        "Mg": (7.5, -0.3, 0.6), "Ca": (6.0, 0.1, 0.5), "Na": (1.8, 0.05, 0.3),
        "K": (1.2, -0.1, 0.25), "Ti": (0.8, 0.02, 0.15),
    }
    meas_by_tile = {tile: {el: max(0.0, rng.normal(mu, mu * 0.18))
                           for el, (mu, _b, _s) in elements.items()}
                    for tile in range(14)}
    rows = []
    for fold in range(n_folds):
        for tile in range(14):
            for el, (mu, bias, sig) in elements.items():
                meas = meas_by_tile[tile][el]
                pred = max(0.0, meas + bias + rng.normal(0.0, sig))
                rows.append((f"tile_{tile:02d}", fold, el, pred, meas))
    df = pd.DataFrame(rows, columns=["tile", "fold", "element",
                                     "predicted_wt_pct", "measured_wt_pct"])
    df["residual_wt_pct"] = df["predicted_wt_pct"] - df["measured_wt_pct"]
    return df


def _per_element_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled per-element RMSE/bias; when a ``fold`` column is present, also the
    per-fold spread (std of per-fold RMSE/bias) for error bars."""
    d = df.assign(_resid=df["predicted_wt_pct"] - df["measured_wt_pct"])
    g = d.groupby("element")["_resid"]
    stats = pd.DataFrame({
        "rmse": g.apply(lambda r: float(np.sqrt(np.mean(r ** 2)))),
        "bias": g.mean(),
        "n": g.size(),
        "rmse_std": 0.0,
        "bias_std": 0.0,
    })
    if "fold" in d.columns and d["fold"].nunique() > 1:
        per_fold = d.groupby(["element", "fold"])["_resid"].agg(
            _rmse=lambda r: float(np.sqrt(np.mean(r ** 2))), _bias="mean")
        spread = per_fold.groupby("element").agg(rmse_std=("_rmse", "std"),
                                                 bias_std=("_bias", "std"))
        stats["rmse_std"] = spread["rmse_std"].reindex(stats.index).fillna(0.0)
        stats["bias_std"] = spread["bias_std"].reindex(stats.index).fillna(0.0)
    return stats.reindex(_order_elements(stats.index.tolist()))


def make_figure(df: pd.DataFrame, out: str, dpi: int, synthetic: bool) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "savefig.bbox": "tight",
    })
    stats = _per_element_stats(df)
    elements = stats.index.tolist()
    colours = {el: element_colour(el) for el in elements}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.4, 2.7))

    # ── (a) predicted vs measured ───────────────────────────────────────────
    hi = float(np.nanmax([df["measured_wt_pct"].max(), df["predicted_wt_pct"].max()]))
    hi *= 1.05
    axA.plot([0, hi], [0, hi], color="0.4", lw=0.8, ls="--", zorder=1)
    # Thin the displayed points per element (stats still use all data) and
    # rasterise, so thousands of overplotted hold-out predictions stay legible
    # and the PDF stays small.
    rng = np.random.default_rng(0)
    cap = max(150, 3000 // max(len(elements), 1))
    for el in elements:
        sub = df[df["element"] == el]
        if len(sub) > cap:
            sub = sub.iloc[rng.choice(len(sub), size=cap, replace=False)]
        axA.scatter(sub["measured_wt_pct"], sub["predicted_wt_pct"],
                    s=5, color=colours[el], edgecolor="none", alpha=0.35,
                    rasterized=True, label=el, zorder=2)
    axA.set_xlim(0, hi)
    axA.set_ylim(0, hi)
    # Equal limits keep the 1:1 line at the diagonal; no equal-aspect, so panel
    # (a) fills the same height as the bar panels and their labels stay aligned.
    axA.set_xlabel("SPX-measured (wt%)")
    axA.set_ylabel("Predicted, hold-out (wt%)")
    overall_rmse = float(np.sqrt(np.mean((df["predicted_wt_pct"] - df["measured_wt_pct"]) ** 2)))
    axA.text(0.05, 0.95, f"RMSE = {overall_rmse:.2f} wt%\n$n$ = {len(df)}",
             transform=axA.transAxes, va="top", ha="left", fontsize=8)
    axA.legend(loc="lower right", ncol=2, fontsize=6.5, markerscale=2.0,
               handletextpad=0.2, columnspacing=0.7, borderpad=0.3,
               framealpha=0.75, edgecolor="0.8")
    axA.set_title("(a) predicted vs. measured")

    # ── (b) per-element RMSE (± per-fold spread) ───────────────────────────
    x = np.arange(len(elements))
    has_err = float(np.nansum(stats["rmse_std"].values)) > 0
    bcol = [colours[e] for e in elements]
    axB.bar(x, stats["rmse"].values, color=bcol, width=0.7,
            yerr=(stats["rmse_std"].values if has_err else None),
            error_kw=dict(ecolor="0.3", lw=0.8, capsize=2))
    axB.set_xticks(x)
    axB.set_xticklabels(elements, rotation=0)
    # Relative error (RMSE as % of the element's mean measured value): the
    # absolute RMSE of a minor element looks small only because the element is.
    mean_meas = df.groupby("element")["measured_wt_pct"].mean()
    err = stats["rmse_std"].values if has_err else np.zeros(len(elements))
    for xi, el, r, e in zip(x, elements, stats["rmse"].values, err):
        m = float(mean_meas.get(el, np.nan))
        if m > 0:
            axB.text(xi, r + e + 0.06, f"{100 * r / m:.0f}%", ha="center",
                     va="bottom", fontsize=6.3, color="0.25", rotation=90)
    axB.set_ylim(0, float(np.nanmax(stats["rmse"].values + err)) * 1.32)
    axB.set_ylabel("RMSE (wt%)")
    axB.set_title("(b) per-element RMSE (labels: % of mean)")
    axB.margins(x=0.02)

    # ── (c) per-element bias (± per-fold spread) ───────────────────────────
    axC.axhline(0.0, color="0.4", lw=0.8)
    axC.bar(x, stats["bias"].values, color=bcol, width=0.7,
            yerr=(stats["bias_std"].values if has_err else None),
            error_kw=dict(ecolor="0.3", lw=0.8, capsize=2))
    axC.set_xticks(x)
    axC.set_xticklabels(elements, rotation=0)
    axC.set_ylabel("Bias = mean(pred $-$ meas) (wt%)")
    axC.set_title("(c) per-element bias")
    axC.margins(x=0.02)

    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DEMO", fontsize=34, color="0.85",
                 ha="center", va="center", rotation=18, zorder=0, alpha=0.6)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}  ({len(df)} pairs, {len(elements)} elements, "
          f"overall RMSE {overall_rmse:.3f} wt%"
          + ("; SYNTHETIC" if synthetic else "") + ")")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "calibration_validation.csv"),
                    help="validation CSV exported from the Calibrate-EDS dialog")
    ap.add_argument("--out", default=os.path.join(here, "fig_calibration.pdf"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"{args.csv} missing columns: {missing}")
        synthetic = False
    else:
        print(f"note: {args.csv} not found — using synthetic demonstration data.")
        df = _synthetic()
        synthetic = True

    make_figure(df, args.out, args.dpi, synthetic)


if __name__ == "__main__":
    main()
