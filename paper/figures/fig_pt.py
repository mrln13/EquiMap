"""Figure: assemblage-to-P-T inversion for the worked example (fig:pt).

(a) modal-misfit map over the computed P-T grid with the best-fit star;
(b) Monte Carlo probability that the observed assemblage is stable.

Inputs (this directory; exported from the viewer):
  pt_grid.csv        grid export (per-cell stable phases + vol fractions)
  monte_carlo.csv    MC stability-probability export
  phase_legend.csv   observed modes (pixel counts per named domain)
  phase_minerals.csv identification scores (reliability weights)

The misfit is recomputed with the toolkit's own scorer
(viewer_thermo_helpers._modal_misfit_grid), with per-domain weights from the
identification confidence and sillimanite as a presence-only constraint —
the same configuration as the worked example.

Usage:  python fig_pt.py [--repo <path>] [--out fig_pt.pdf]
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

# Thermo keys for the observed domain names (mineral_id.THERMO_ABBREV).
NAME_TO_KEY = {"Biotite": "bi", "Quartz": "q", "Muscovite": "mu",
               "Plagioclase": "fsp", "Epidote": "ep", "Alkali feldspar": "fsp",
               "Sillimanite": "sill"}
# Chlorite (retrograde after biotite) has no key: excluded from the peak-assemblage modes.
# Sillimanite is a presence-only constraint even though the map now resolves
# it: at ~1 % of the map its proportion is neither a reliable modal target
# nor an informative one (a 1 % target costs almost nothing when the phase
# is absent), whereas its presence excludes every cell without an
# aluminosilicate. Its pixels are dropped from the modal vector.
PRESENCE = ["sill"]
REGION_DELTA = 0.02   # accepted region = cells within this of the minimum misfit


def load_grid(path):
    import pandas as pd
    g = pd.read_csv(path)
    keys = [c.replace("vol_frac_", "") for c in g.columns
            if c.startswith("vol_frac_")]
    return g, keys


def build_results(g, keys, repo):
    sys.path.insert(0, str(repo))
    from thermo.result import ThermoResult, PhaseResult, phase_base_name
    results = []
    for _, r in g.iterrows():
        phases = []
        for k in keys:
            v = r.get(f"vol_frac_{k}")
            if v == v and v is not None and v > 0:   # NaN-safe
                # "fsp_2" etc. = second instance of a solvus pair (see
                # ThermoResult.to_dict); the phase keeps its base name so
                # alias matching still sees "fsp" and both instances count.
                name = phase_base_name(k)
                phases.append(PhaseResult(
                    name, float(r.get(f"mol_frac_{k}") or 0),
                    float(r.get(f"wt_frac_{k}") or 0), float(v),
                    is_melt=(name == "liq")))
        results.append(ThermoResult(p_gpa=float(r.P_GPa), t_c=float(r.T_C),
                                    stable_phases=phases, backend="magemin"))
    return results


def observed_from_exports():
    import json
    px, names = {}, {}
    with open(HERE / "phase_legend.csv") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            px[cid] = float(row["pixel_count"])
            names[cid] = row["name"].strip()
    # Mixed-domain unmixing (segmentation_params.json "unmix" block): a
    # domain that is a sub-pixel mixture has its pixel count redistributed
    # onto the endmember domains (renormalised NNLS fractions) before the
    # name -> key mapping, so its pixels are not all counted as one phase.
    seg_json = HERE / "segmentation_params.json"
    if seg_json.exists():
        seg = json.loads(seg_json.read_text())
        for src, d in (seg.get("unmix") or {}).items():
            src = int(src)
            fr = {int(e): float(v) for e, v in (d.get("fractions") or {}).items()
                  if float(v) > 0}
            if src in px and fr:
                n = px.pop(src)
                tot_f = sum(fr.values())
                for e, v in fr.items():
                    px[e] = px.get(e, 0.0) + n * v / tot_f
    modes_px = {}
    for cid, n in px.items():
        key = NAME_TO_KEY.get(names.get(cid, ""))
        if key and key not in PRESENCE:
            modes_px[key] = modes_px.get(key, 0.0) + n
    tot = sum(modes_px.values())
    modes = {k: v / tot for k, v in modes_px.items()}
    weights = {}
    with open(HERE / "phase_minerals.csv") as f:
        for row in csv.DictReader(f):
            key = NAME_TO_KEY.get(row["name"].strip())
            if key and row.get("top1_score"):
                weights[key] = max(weights.get(key, 0.0),
                                   float(row["top1_score"]))
    for k in PRESENCE:
        weights.pop(k, None)
    return modes, weights


def load_mc(path):
    Ts, Ps, rows = None, [], []
    for line in open(path):
        line = line.strip()
        if line.startswith("stability_prob"):
            Ts = [float(x.split("=")[1]) for x in line.split(",")[1:]]
        elif line.startswith("P="):
            parts = line.split(",")
            Ps.append(float(parts[0].split("=")[1]))
            rows.append([float(x) for x in parts[1:]])
    return np.array(Ts), np.array(Ps), np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent.parent),
                    help="EquiMap repo root (for thermo/viewer helper imports)")
    ap.add_argument("--grid-csv", default=str(HERE / "pt_grid.csv"))
    ap.add_argument("--mc-csv", default=str(HERE / "monte_carlo.csv"))
    ap.add_argument("--out", default=str(HERE / "fig_pt.pdf"))
    args = ap.parse_args()

    g, keys = load_grid(args.grid_csv)
    n_p = g.P_GPa.nunique()
    n_t = g.T_C.nunique()
    p_ax = np.sort(g.P_GPa.unique())
    t_ax = np.sort(g.T_C.unique())

    results = build_results(g, keys, Path(args.repo))
    from viewer_thermo_helpers import _modal_misfit_grid
    modes, weights = observed_from_exports()
    grid, basis, unmatched = _modal_misfit_grid(
        results, modes, n_p, n_t, weights=weights, presence_keys=PRESENCE)
    mis = np.asarray(grid, dtype=float)
    i = np.nanargmin(mis)
    ip, it = divmod(i, n_t)
    print(f"misfit min {mis.flat[i]:.3f} at T={t_ax[it]:.1f} C, "
          f"P={p_ax[ip]:.3f} GPa (basis {basis}; unmatched {unmatched})")

    # Presence-only view of the same grid: where EVERY observed phase
    # (proportion keys + presence-only keys) is predicted stable, and where
    # melt is predicted. The accepted region must lie inside the first; the
    # second shows whether accepted or stable cells are supersolidus (melt is
    # excluded from the modal vector, so the misfit alone does not say).
    from viewer_thermo_helpers import _phase_matches
    obs_all = list(modes) + list(PRESENCE)
    stab = np.array([all(any(_phase_matches(k, ph.name) for ph in r.stable_phases)
                         for k in obs_all) for r in results], dtype=float).reshape(n_p, n_t)
    melt = np.array([sum((ph.vol_fraction or 0.0) for ph in r.stable_phases if ph.is_melt)
                     for r in results], dtype=float).reshape(n_p, n_t)
    acc = mis <= np.nanmin(mis) + REGION_DELTA
    sp, st = np.where(stab > 0)
    print(f"stability field (all of {obs_all}): {int(stab.sum())} cells, "
          f"{t_ax[st].min():.0f}-{t_ax[st].max():.0f} C, {p_ax[sp].min():.2f}-{p_ax[sp].max():.2f} GPa; "
          f"misfit inside {np.nanmin(mis[stab > 0]):.3f}-{np.nanmax(mis[stab > 0]):.3f}, "
          f"lowest outside {np.nanmin(mis[stab == 0]):.3f}; accepted {int(acc.sum())} cells "
          f"({int((acc & (stab > 0)).sum())} inside the field, {int((acc & (melt > 0)).sum())} with melt); "
          f"melt-bearing cells in the field: {int(((stab > 0) & (melt > 0)).sum())}")

    Ts, Ps, prob = load_mc(args.mc_csv)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    im_a = ax_a.pcolormesh(t_ax, p_ax, mis, cmap="RdYlGn_r",
                           vmin=0.0, vmax=min(1.0, np.nanmax(mis)),
                           shading="nearest")
    ax_a.plot(t_ax[it], p_ax[ip], marker="*", ms=14, mec="black",
              mfc="white", lw=0)
    # Accepted region: the cells the data accept (misfit <= min + delta) —
    # the estimate the text reports, as opposed to the single best cell.
    ax_a.contour(t_ax, p_ax, mis, levels=[float(np.nanmin(mis)) + REGION_DELTA],
                 colors="white", linestyles="--", linewidths=1.1)
    # Solid: the field where every observed phase is stable (the presence-
    # only constraint). Dotted: melt-in (the wet solidus of this bulk).
    ax_a.contour(t_ax, p_ax, stab, levels=[0.5], colors="black", linewidths=1.0)
    if np.nanmax(melt) > 0:
        ax_a.contour(t_ax, p_ax, (melt > 0).astype(float), levels=[0.5],
                     colors="black", linewidths=0.9, linestyles=":")
    from matplotlib.lines import Line2D
    ax_a.legend(handles=[Line2D([], [], color="white", ls="--", lw=1.1, label="accepted region"),
                         Line2D([], [], color="black", ls="-", lw=1.0, label="assemblage stable"),
                         Line2D([], [], color="black", ls=":", lw=0.9, label="melt-in")],
                loc="upper left", fontsize=6, frameon=True, facecolor="0.62",
                framealpha=0.85, edgecolor="none", handlelength=2.2, borderpad=0.5)
    fig.colorbar(im_a, ax=ax_a, label="modal misfit")
    ax_a.set_title("(a) assemblage misfit", fontsize=10)
    ax_a.set_xlabel("T (°C)")
    ax_a.set_ylabel("P (GPa)")

    im_b = ax_b.pcolormesh(Ts, Ps, prob, cmap="Blues", vmin=0.0, vmax=1.0,
                           shading="nearest")
    if np.nanmax(prob) >= 0.5:
        # crimson, not black: black vanished against the dark cells it ran through
        ax_b.contour(Ts, Ps, prob, levels=[0.5], colors="crimson",
                     linewidths=1.4, linestyles="--")
    ax_b.plot(t_ax[it], p_ax[ip], marker="*", ms=14, mec="black",
              mfc="white", lw=0)
    if np.nanmax(melt) > 0:
        ax_b.contour(t_ax, p_ax, (melt > 0).astype(float), levels=[0.5],
                     colors="black", linewidths=0.9, linestyles=":")
    fig.colorbar(im_b, ax=ax_b, label="P(assemblage stable)")
    ax_b.set_title("(b) Monte Carlo stability", fontsize=10)
    ax_b.set_xlabel("T (°C)")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
