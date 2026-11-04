"""How much can the P-T consistency check of Section 5.3 discriminate?

Re-scores the staged section-level grid (pt_grid.csv, 21 x 21 over
550-750 C / 0.2-1.0 GPa) with deliberately WRONG observations and reports,
for each, the misfit floor and the accepted region (cells within
REGION_DELTA of the minimum), next to the independent literature box
(620-700 C, 0.3-0.74 GPa). No thermodynamic computation: pure re-scoring,
as fig_pt.py / fig_reconstruction.py do.

Variants:
  adopted            the published observation (unmixed modes, weights,
                     sillimanite presence-only)
  matrix -> quartz   both matrix domains merged wholly into quartz (the
                     July 2026 quartz over-estimate, Section 4.2)
  Ms -> cordierite   the muscovite domain mis-identified as cordierite
  Bt -> chlorite     the biotite domain mis-identified as chlorite
  Ms dropped         muscovite left out of the observed list

Also prints the stability field of the five observed phases. Numbers are
quoted only in the %% provenance comment of Section 5.3 and in README.md
(the paper states the criterion, not these examples).

Usage:  python consistency_counterfactuals.py [--repo <path>]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIT_T = (620.0, 700.0)      # independent peak-temperature estimates, C
LIT_P = (0.30, 0.74)        # independent pressure estimates, GPa
OBSERVED = ("bi", "mu", "q", "fsp", "sill")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--grid-csv", default=str(HERE / "pt_grid.csv"))
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    sys.path.insert(0, str(HERE))
    import fig_pt
    from viewer_thermo_helpers import _modal_misfit_grid, _phase_matches

    g, keys = fig_pt.load_grid(Path(args.grid_csv))
    res = fig_pt.build_results(g, keys, args.repo)
    n_p, n_t = g.P_GPa.nunique(), g.T_C.nunique()
    p_ax, t_ax = np.sort(g.P_GPa.unique()), np.sort(g.T_C.unique())
    modes, w = fig_pt.observed_from_exports()
    print("adopted modes:", {k: round(v, 3) for k, v in modes.items()})

    TT, PP = np.meshgrid(t_ax, p_ax)
    box = ((TT >= LIT_T[0]) & (TT <= LIT_T[1])
           & (PP >= LIT_P[0]) & (PP <= LIT_P[1]))
    print(f"grid {TT.size} cells; literature box {LIT_T[0]:.0f}-{LIT_T[1]:.0f} C "
          f"x {LIT_P[0]:.2f}-{LIT_P[1]:.2f} GPa = {int(box.sum())} cells "
          f"({100 * box.mean():.0f} %)")

    # Both matrix domains merged into quartz (the pre-unmix state).
    seg = json.loads((HERE / "segmentation_params.json").read_text())
    unmixed = set((seg.get("unmix") or {}).keys())
    merged = {}
    with open(HERE / "phase_legend.csv") as f:
        for row in csv.DictReader(f):
            cid = row["cluster_id"].strip()
            key = "q" if cid in unmixed else fig_pt.NAME_TO_KEY.get(row["name"].strip())
            if key in ("q", "bi", "mu", "fsp"):
                merged[key] = merged.get(key, 0.0) + float(row["pixel_count"])
    tot = sum(merged.values())
    merged = {k: v / tot for k, v in merged.items()}

    def swap(obs, wts, old, new):
        o, ww = dict(obs), dict(wts)
        o[new] = o.pop(old)
        ww[new] = ww.pop(old)
        return o, ww

    variants = [("adopted", modes, w),
                ("matrix -> quartz", merged, {k: w.get(k, 1.0) for k in merged}),
                ("Ms -> cordierite", *swap(modes, w, "mu", "cd")),
                ("Bt -> chlorite", *swap(modes, w, "bi", "chl")),
                ("Ms dropped", {k: v for k, v in modes.items() if k != "mu"},
                 {k: v for k, v in w.items() if k != "mu"})]
    for tag, obs, ww in variants:
        s = sum(obs.values())
        obs = {k: v / s for k, v in obs.items()}
        mis = np.asarray(_modal_misfit_grid(res, obs, n_p, n_t, weights=ww,
                                            presence_keys=fig_pt.PRESENCE)[0],
                         dtype=float)
        mn = float(np.nanmin(mis))
        acc = mis <= mn + fig_pt.REGION_DELTA
        pp, tt = np.where(acc)
        verdict = ("FAILS" if mn >= 0.19 or not (acc & box).any() else "passes")
        print(f"{tag:<18s} floor {mn:.3f}; region {t_ax[tt].min():.0f}-"
              f"{t_ax[tt].max():.0f} C, {p_ax[pp].min():.2f}-{p_ax[pp].max():.2f} GPa, "
              f"{int(acc.sum())} cells, {int((acc & box).sum())} in the box; {verdict}")

    stab = np.zeros((n_p, n_t), bool)
    for i, r in enumerate(res):
        present = [p.name for p in r.stable_phases if p.vol_fraction > 0]
        stab[divmod(i, n_t)] = all(any(_phase_matches(k, p) for p in present)
                                   for k in OBSERVED)
    pp, tt = np.where(stab)
    print(f"five observed phases coexist in {int(stab.sum())} cells: "
          f"{t_ax[tt].min():.0f}-{t_ax[tt].max():.0f} C, "
          f"{p_ax[pp].min():.2f}-{p_ax[pp].max():.2f} GPa")


if __name__ == "__main__":
    main()
