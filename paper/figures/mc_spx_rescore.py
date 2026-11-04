"""Re-score the SPX-bulk Monte Carlo (Fig. 5 stability contour) for a new phase map.

The viewer's Monte Carlo perturbed the SPX bulk of Fig. 5 (session run #3)
with 10 % relative sigma, 16 draws, seed 0, and ran one MAGEMin grid per
draw. The grids depend only on the compositions (served from the result
cache, or recomputed here if the cache was cleared); the stability
probability and misfit statistics depend on the observed assemblage and
modes, which change with the segmentation. This script regenerates the same
draws, fetches/computes their grids, aggregates them against the CURRENT
phase exports (fig_pt.observed_from_exports: unmix-corrected modes,
confidence weights, sillimanite presence-only) and writes

  monte_carlo.csv           stability-probability block in the viewer's export
                            layout (what fig_pt.load_mc reads)
  monte_carlo_summary.json  misfit-mean region, best-fit cloud, draw list

The stability set is the peak assemblage the July analysis used
(bi, mu, q, fsp + sill presence): epidote is retrograde and counted in the
modes/misfit but not required for "assemblage stable".
"""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("fig_pt", HERE / "fig_pt.py")
fig_pt = importlib.util.module_from_spec(spec); spec.loader.exec_module(fig_pt)
from viewer_thermo_helpers import _mc_sample_compositions, _mc_aggregate   # noqa: E402
from thermo import Composition, ThermoCalc                                 # noqa: E402
from thermo.cache import GridCache                                         # noqa: E402

SESSION = Path(r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.json")
from pt_grid_spec import P_RANGE, T_RANGE, NP, NT                 # the Fig 5 grid (pt_grid.csv)
FE3, H2O = 0.1, 4.0
N, SIGMA, SEED = 16, 0.10, 0
BK = {"db": "mpe", "fe3_fraction": FE3}
OBS = ["bi", "mu", "q", "fsp", "sill"]
p_ax = np.linspace(*P_RANGE, NP); t_ax = np.linspace(*T_RANGE, NT)
t0 = time.time()

modes, wts = fig_pt.observed_from_exports()
MK = {"weights": wts, "presence_keys": list(fig_pt.PRESENCE), "open_other_tol": 0.0}
print("modes", {k: round(v, 3) for k, v in modes.items()}, "weights", {k: round(v, 2) for k, v in wts.items()}, flush=True)
s = json.load(open(SESSION, encoding="utf-8"))
runs = {r["index"]: r for r in s["thermod_runs"]}
spx_el = {k: float(v) for k, v in runs[3]["comp_el"].items()}


def build(comp_el):
    import model_bulk      # shared recipe: elements passed, Fe3+, H2O as wt% of the bulk
    return model_bulk.model_composition(comp_el, fe3=FE3, h2o=H2O)


cache = GridCache()
samples = _mc_sample_compositions(spx_el, SIGMA, N, SEED)
results = []
calc = None
for i, c in enumerate(samples):
    comp = build(c)
    res = cache.get_results(cache.key_for("magemin", BK, comp, P_RANGE, T_RANGE, NP, NT))
    if res is None:
        if calc is None:
            calc = ThermoCalc("magemin", **BK)
        t1 = time.time()
        res = calc.run_grid(comp, P_RANGE, T_RANGE, n_p=NP, n_t=NT)
        print(f"draw {i}: computed {len(res)} cells in {time.time() - t1:.0f} s", flush=True)
    else:
        print(f"draw {i}: cache hit", flush=True)
    results.append(res)

agg = _mc_aggregate(results, NP, NT, observed_phases=OBS, observed_modes=modes, misfit_kwargs=MK)
sp = np.asarray(agg["stability_prob"], dtype=float)
mm = np.asarray(agg["misfit_mean"], dtype=float)
with open(HERE / "monte_carlo.csv", "w", newline="") as f:
    f.write(f"# Monte Carlo,n={N},sigma_pct={SIGMA * 100:.1f},seed={SEED}\n\n")
    w = csv.writer(f)
    w.writerow(["stability_prob"] + [f"T={t:.1f}" for t in t_ax])
    for ip in range(NP):
        w.writerow([f"P={p_ax[ip]:.4f}"] + [f"{sp[ip, it]:.4f}" for it in range(NT)])


def region(mis, delta=fig_pt.REGION_DELTA):
    i = int(np.nanargmin(mis)); ip, it = divmod(i, NT)
    reg = ~np.isnan(mis) & (mis <= mis.flat[i] + delta)
    ii, jj = np.where(reg)
    return dict(best_T=float(t_ax[it]), best_P=float(p_ax[ip]), best_misfit=float(mis.flat[i]),
                T_lo=float(t_ax[jj.min()]), T_hi=float(t_ax[jj.max()]),
                P_lo=float(p_ax[ii.min()]), P_hi=float(p_ax[ii.max()]), n_cells=int(reg.sum()))


ok = sp >= 0.5
ii, jj = np.where(ok)
stab = (dict(T_lo=float(t_ax[jj.min()]), T_hi=float(t_ax[jj.max()]), P_lo=float(p_ax[ii.min()]),
             P_hi=float(p_ax[ii.max()]), n_cells=int(ok.sum())) if ok.any() else None)
bf = [(float(t_ax[j]), float(p_ax[i]), float(m)) for i, j, m in agg["best_fits"]]
summary = {"n": N, "sigma_pct": SIGMA * 100, "seed": SEED, "stability_set": OBS,
           "modes": modes, "weights": wts, "stability_ge_0.5": stab, "stability_max": float(np.nanmax(sp)),
           "misfit_mean_region": region(mm),
           "best_fit_T_C": sorted(round(b[0]) for b in bf), "best_fit_P_GPa": sorted(round(b[1], 2) for b in bf),
           "best_fit_T_median": float(np.median([b[0] for b in bf])),
           "best_fit_P_median": float(np.median([b[1] for b in bf])),
           "spx_bulk_el": spx_el, "elapsed_s": round(time.time() - t0)}
json.dump(summary, open(HERE / "monte_carlo_summary.json", "w"), indent=1)
print("P(stable) >= 0.5 region:", stab, "\nmisfit-mean region:", summary["misfit_mean_region"],
      "\nbest-fit T:", summary["best_fit_T_C"], "\nbest-fit P:", summary["best_fit_P_GPa"], flush=True)
print(f"wrote monte_carlo.csv + monte_carlo_summary.json ({time.time() - t0:.0f} s)")
