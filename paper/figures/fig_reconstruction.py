"""Figure: sensitivity of the P-T estimate to the pseudomorph interpretation
(fig:reconstruction).

Three treatments of the muscovite nodules (pseudomorphs after peak
K-feldspar), each scored by re-using the SAME computed grid — no additional
thermodynamic computation, only re-scoring of the modal misfit:

  (a) assemblage exactly as observed;
  (b) partial reconstruction: the nodule-hosted muscovite and the epidote
      cores reassigned to their K-feldspar precursor, matrix muscovite
      retained (illustrated with half the muscovite mode attributed to the
      nodules);
  (c) full conversion: the entire muscovite mode plus epidote converted to
      K-feldspar (an over-correction, shown deliberately).

Inputs as fig_pt.py: pt_grid.csv + phase_legend.csv + phase_minerals.csv.

Usage:  python fig_reconstruction.py [--repo <path>] [--out ...]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NAME_TO_KEY = {"Biotite": "bi", "Quartz": "q", "Muscovite": "mu",
               "Plagioclase": "fsp", "Epidote": "ep"}
PRESENCE = ["sill"]
REGION_DELTA = 0.02   # accepted region = cells within this of each panel's minimum
NODULE_MS_FRACTION = 0.5   # illustrative share of the Ms mode in the nodules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--grid-csv", default=str(HERE / "pt_grid.csv"))
    ap.add_argument("--out", default=str(HERE / "fig_reconstruction.pdf"))
    args = ap.parse_args()

    import pandas as pd
    g = pd.read_csv(args.grid_csv)
    keys = [c.replace("vol_frac_", "") for c in g.columns
            if c.startswith("vol_frac_")]
    sys.path.insert(0, args.repo)
    from thermo.result import ThermoResult, PhaseResult, phase_base_name
    from viewer_thermo_helpers import _modal_misfit_grid

    results = []
    for _, r in g.iterrows():
        # "fsp_2" etc. = second instance of a solvus pair (ThermoResult.to_dict);
        # keep the base name so alias matching sees "fsp" and both count.
        phases = [PhaseResult(phase_base_name(k),
                              float(r.get(f"mol_frac_{k}") or 0),
                              float(r.get(f"wt_frac_{k}") or 0),
                              float(r.get(f"vol_frac_{k}")),
                              is_melt=(phase_base_name(k) == "liq"))
                  for k in keys
                  if pd.notna(r.get(f"vol_frac_{k}"))
                  and r.get(f"vol_frac_{k}") > 0]
        results.append(ThermoResult(p_gpa=float(r.P_GPa), t_c=float(r.T_C),
                                    stable_phases=phases, backend="magemin"))
    n_p = g.P_GPa.nunique()
    n_t = g.T_C.nunique()
    p_ax = np.sort(g.P_GPa.unique())
    t_ax = np.sort(g.T_C.unique())

    # Observed modes + confidence weights from the classification exports.
    # The matrix-domain unmixing (segmentation_params.json "unmix") is applied
    # first, so every variant starts from the resolution-corrected modes.
    import json as _json
    px, names_by_cid = {}, {}
    with open(HERE / "phase_legend.csv") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            px[cid] = float(row["pixel_count"])
            names_by_cid[cid] = row["name"].strip()
    seg_json = HERE / "segmentation_params.json"
    if seg_json.exists():
        seg = _json.loads(seg_json.read_text())
        for src, d in (seg.get("unmix") or {}).items():
            src = int(src)
            frs = {int(e): float(v) for e, v in (d.get("fractions") or {}).items()
                   if float(v) > 0}
            if src in px and frs:
                n = px.pop(src)
                tot_f = sum(frs.values())
                for e, v in frs.items():
                    px[e] = px.get(e, 0.0) + n * v / tot_f
    modes = {}
    for cid, n in px.items():
        key = NAME_TO_KEY.get(names_by_cid.get(cid, ""))
        if key:
            modes[key] = modes.get(key, 0) + n
    tot = sum(modes.values())
    modes = {k: v / tot for k, v in modes.items()}
    wts = {}
    with open(HERE / "phase_minerals.csv") as f:
        for row in csv.DictReader(f):
            key = NAME_TO_KEY.get(row["name"].strip())
            if key and row.get("top1_score"):
                wts[key] = max(wts.get(key, 0.0), float(row["top1_score"]))

    f = NODULE_MS_FRACTION
    variants = [
        ("(a) as observed", modes, wts),
        ("(b) nodule Ms → Kfs",
         {"bi": modes["bi"], "q": modes["q"], "mu": modes["mu"] * (1 - f),
          "kf": modes["mu"] * f + modes.get("ep", 0.0), "pl": modes["fsp"]},
         {"bi": wts["bi"], "q": wts["q"], "mu": wts["mu"], "kf": 1.0,
          "pl": wts["fsp"]}),
        ("(c) all Ms → Kfs",
         {"bi": modes["bi"], "q": modes["q"],
          "kf": modes["mu"] + modes.get("ep", 0.0), "pl": modes["fsp"]},
         {"bi": wts["bi"], "q": wts["q"], "kf": 1.0, "pl": wts["fsp"]}),
    ]

    grids, stars = [], []
    for tag, obs, w in variants:
        s = sum(obs.values())
        obs = {k: v / s for k, v in obs.items()}
        grid, _basis, _un = _modal_misfit_grid(results, obs, n_p, n_t,
                                               weights=w,
                                               presence_keys=PRESENCE)
        arr = np.asarray(grid, dtype=float)
        i = np.nanargmin(arr)
        ip, it = divmod(i, n_t)
        grids.append(arr)
        stars.append((t_ax[it], p_ax[ip], arr.flat[i]))
        print(f"{tag:<28s} min={arr.flat[i]:.3f} at "
              f"T={t_ax[it]:.1f} P={p_ax[ip]:.3f}")

    vmax = min(1.0, max(np.nanmax(a) for a in grids))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    for ax, (tag, _o, _w), arr, (st, sp, sm) in zip(axes, variants, grids,
                                                    stars):
        im = ax.pcolormesh(t_ax, p_ax, arr, cmap="RdYlGn_r", vmin=0.0,
                           vmax=vmax, shading="nearest")
        ax.plot(st, sp, marker="*", ms=13, mec="black", mfc="white", lw=0)
        # Accepted region (misfit <= min + delta): the width of the valley
        # is the constraint quality, which the star alone does not show.
        ax.contour(t_ax, p_ax, arr, levels=[sm + REGION_DELTA],
                   colors="white", linestyles="--", linewidths=1.0)
        ax.set_title(tag, fontsize=9.5)
        ax.set_xlabel("T (°C)")
    axes[0].set_ylabel("P (GPa)")
    fig.colorbar(im, ax=axes, label="modal misfit", fraction=0.03, pad=0.02)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
