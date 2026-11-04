"""Tier-2 check: propagate the SPX-calibration hold-out error into the P-T
estimate obtained from the calibrated-map bulk (worked example, C23S1A).

- composition  = sweep L0 mean calibrated bulk (eds_sweep_comps.json), + 4 wt% H2O,
                 Fe3+/Fe = 0.1, MAGEMin mpe, 20x20, 300-800 C / 0.1-1.5 GPa
                 (the Fig. 5 grid)
- perturbation = per-element ABSOLUTE sigma = |hold-out bias| (wt% points on the
                 cation-only basis of calibration_validation.csv); n=16, seed=0
- scoring      = identical to fig_pt.py (modes + confidence weights + sill
                 presence-only); stability set reconstructed against the
                 exported monte_carlo.csv from the cached SPX draws.
Outputs (figures/): monte_carlo_calibrated.csv (fig_pt.load_mc layout),
monte_carlo_calibrated_long.csv, mc_calibrated_samples.csv, mc_calibrated_summary.json
"""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "_Manuscript" / "figures"
SESSION = Path(r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.json")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(FIG))

spec = importlib.util.spec_from_file_location("fig_pt", FIG / "fig_pt.py")
fig_pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig_pt)

from viewer_thermo_helpers import (_mc_sample_compositions, _mc_aggregate,
                                   _modal_misfit_grid)
from thermo import Composition, ThermoCalc
from thermo.cache import GridCache

sys.path.insert(0, str(FIG))
from pt_grid_spec import P_RANGE, T_RANGE, NP, NT   # the Fig 5 grid
FE3, H2O = 0.1, 4.0
N, SEED = 16, 0
BK = {"db": "mpe", "fe3_fraction": FE3}
p_ax = np.linspace(*P_RANGE, NP)
t_ax = np.linspace(*T_RANGE, NT)


def log(*a):
    print(*a, flush=True)


def build(comp_el):
    import model_bulk      # shared recipe: elements passed, Fe3+, H2O as wt% of the bulk
    return model_bulk.model_composition(comp_el, fe3=FE3, h2o=H2O)


def region(mis):
    ok = ~np.isnan(mis)
    i = int(np.nanargmin(mis))
    ip, it = divmod(i, NT)
    reg = ok & (mis <= mis.flat[i] + 0.02)
    ii, jj = np.where(reg)
    return dict(best_T=float(t_ax[it]), best_P=float(p_ax[ip]), best_misfit=float(mis.flat[i]),
                T_lo=float(t_ax[jj.min()]), T_hi=float(t_ax[jj.max()]),
                P_lo=float(p_ax[ii.min()]), P_hi=float(p_ax[ii.max()]), n_cells=int(reg.sum()))


def load_mc_export(path):
    return fig_pt.load_mc(path)


t0 = time.time()
modes, wts = fig_pt.observed_from_exports()
MK = {"weights": wts, "presence_keys": list(fig_pt.PRESENCE), "open_other_tol": 0.0}
log("modes", {k: round(v, 3) for k, v in modes.items()}, "weights", {k: round(v, 2) for k, v in wts.items()})

s = json.load(open(SESSION, encoding="utf-8"))
runs = {r["index"]: r for r in s["thermod_runs"]}
spx_el = {k: float(v) for k, v in runs[3]["comp_el"].items()}
# calibrated L0 bulk: the mean variant of the re-run sweep (eds_sweep_run.py on
# the line-aware nets), not the July session run #4 (sideband nets)
cal_el = {k: float(v) for k, v in json.load(open(FIG / "eds_sweep_comps.json"))["0"]["variants"]["mean"].items()}

# per-element |bias| from the hold-out validation (cation-only basis, wt% points)
import pandas as pd
d = pd.read_csv(FIG / "calibration_validation.csv")
d["res"] = d["predicted_wt_pct"] - d["measured_wt_pct"]
bias = d.groupby("element")["res"].mean().to_dict()
rmse = d.groupby("element")["res"].apply(lambda x: float(np.sqrt((x ** 2).mean()))).to_dict()
sig = {el: abs(bias.get(el, 0.0)) for el in cal_el}
log("sigma (|bias|, wt%):", {k: round(v, 2) for k, v in sig.items()})

calc = ThermoCalc("magemin", **BK)
cache = GridCache()


def key(c):
    return cache.key_for("magemin", BK, c, P_RANGE, T_RANGE, NP, NT)


# ---------------------------------------------------------------- SPX check
# Reconstruct the July stability set from cached SPX draws (10% relative, n=16,
# seed 0) against the exported monte_carlo.csv. Skipped if the draws aren't cached.
Ts_x, Ps_x, prob_x = load_mc_export(FIG / "monte_carlo.csv")
spx_samples = _mc_sample_compositions(spx_el, 0.10, N, SEED)
hits = [cache.get_results(key(build(c))) for c in spx_samples]
n_hit = sum(h is not None for h in hits)
log(f"SPX 10% draws cached: {n_hit}/{N}")
OBS = ["bi", "mu", "q", "fsp", "sill"]
if n_hit == N:
    cands = {"bi mu q fsp": ["bi", "mu", "q", "fsp"],
             "bi mu q fsp sill": ["bi", "mu", "q", "fsp", "sill"],
             "bi mu q fsp sill ep": ["bi", "mu", "q", "fsp", "sill", "ep"],
             "bi q fsp sill": ["bi", "q", "fsp", "sill"]}
    best = None
    for name, obs in cands.items():
        agg = _mc_aggregate(hits, NP, NT, observed_phases=obs)
        sp = np.asarray(agg["stability_prob"], dtype=float)
        err = float(np.abs(sp - prob_x).max())
        log(f"  candidate [{name}] max |dP| vs monte_carlo.csv = {err:.4f}")
        if best is None or err < best[1]:
            best = (obs, err)
    OBS = best[0]
    log("stability set adopted:", OBS, "(max |dP| %.4f)" % best[1])
    agg_spx = _mc_aggregate(hits, NP, NT, observed_phases=OBS, observed_modes=modes, misfit_kwargs=MK)
    mm = np.asarray(agg_spx["misfit_mean"], dtype=float)
    log("SPX 10% MC: misfit-mean region", region(mm))
    bf = [(float(t_ax[j]), float(p_ax[i]), m) for i, j, m in agg_spx["best_fits"]]
    log("SPX 10%% MC best-fit cloud T: %s" % sorted(round(b[0]) for b in bf))
    log("SPX 10%% MC best-fit cloud P: %s" % sorted(round(b[1], 2) for b in bf))
else:
    log("SPX draws not fully cached -> stability set assumed:", OBS)

# ---------------------------------------------------------- calibrated MC
samples = _mc_sample_compositions(cal_el, sig, N, SEED)
with open(FIG / "mc_calibrated_samples.csv", "w", newline="") as f:
    w = csv.writer(f)
    els = list(cal_el)
    w.writerow(["draw"] + els)
    w.writerow(["unperturbed"] + [round(cal_el[e], 4) for e in els])
    for k, c in enumerate(samples):
        w.writerow([k] + [round(c.get(e, 0.0), 4) for e in els])

log("running unperturbed calibrated bulk on the Fig 5 grid ...")
t1 = time.time()
base = calc.run_grid(build(cal_el), P_RANGE, T_RANGE, n_p=NP, n_t=NT)
log(f"  done in {time.time() - t1:.0f} s ({len(base)} cells)")
mis0 = np.asarray(_modal_misfit_grid(base, modes, NP, NT, **MK)[0], dtype=float)
r0 = region(mis0)
log("calibrated bulk, Fig-5 grid:", r0)
# what is stable at the SPX best cell and along the 0.395/0.468 rows
by_pt = {(round(r.p_gpa, 3), round(r.t_c, 1)): r for r in base}

grids = []
for k, c in enumerate(samples):
    t1 = time.time()
    try:
        res = calc.run_grid(build(c), P_RANGE, T_RANGE, n_p=NP, n_t=NT)
        grids.append(res)
        log(f"draw {k:2d}: {len(res)} cells in {time.time() - t1:.0f} s  (Al {c.get('Al', 0):.2f}, Si {c.get('Si', 0):.2f})")
    except Exception as e:  # keep going; aggregate skips failures
        grids.append(None)
        log(f"draw {k:2d}: FAILED {e!r}")

agg = _mc_aggregate(grids, NP, NT, observed_phases=OBS, observed_modes=modes, misfit_kwargs=MK)
sp = np.asarray(agg["stability_prob"], dtype=float)
mm = np.asarray(agg["misfit_mean"], dtype=float)
ms = np.asarray(agg["misfit_std"], dtype=float)


def extent(mask):
    ii, jj = np.where(mask)
    if len(ii) == 0:
        return None
    return dict(T_lo=float(t_ax[jj.min()]), T_hi=float(t_ax[jj.max()]),
                P_lo=float(p_ax[ii.min()]), P_hi=float(p_ax[ii.max()]), n_cells=int(mask.sum()))


half = extent(sp >= 0.5)
full = extent(sp >= 0.999)
imax = int(np.nanargmax(sp)); ipm, itm = divmod(imax, NT)
log(f"stability_prob max {sp.flat[imax]:.3f} at T={t_ax[itm]:.0f} P={p_ax[ipm]:.2f}; cells >=0.5: {(sp >= 0.5).sum()}, >0: {(sp > 0).sum()}")
# --- diagnostics: which observed phase limits the stability set, per draw
from viewer_thermo_helpers import _pt_estimate_from_results, _phase_matches
diag = []
for k, g in enumerate(grids):
    if g is None:
        continue
    names_any = set(p.name for r in g for p in r.stable_phases)
    stable_all = int(np.asarray(_pt_estimate_from_results(g, OBS, NP, NT)).sum())
    per = {}
    for ob in OBS:
        per[ob] = sum(1 for r in g if any(_phase_matches(ob, p.name) for p in r.stable_phases))
    co = sum(1 for r in g if all(any(_phase_matches(ob, p.name) for p in r.stable_phases) for ob in ("mu", "sill")))
    diag.append(dict(draw=k, all_stable_cells=stable_all, per_phase_cells=per, mu_and_sill_cells=co))
    log(f"draw {k:2d}: all-5-stable cells={stable_all:3d}  per-phase {per}  mu+sill cells={co}")
# reference: stability without the presence-only sillimanite
agg4 = _mc_aggregate(grids, NP, NT, observed_phases=["bi", "mu", "q", "fsp"])
sp4 = np.asarray(agg4["stability_prob"], dtype=float)
log("stability (bi mu q fsp, no sill) >=0.5:", extent(sp4 >= 0.5), " ==1:", extent(sp4 >= 0.999))
# per-draw: sill-in temperature along the P=0.395 and 0.468 rows
for k, g in enumerate(grids):
    if g is None:
        continue
    rows = {}
    for pr in (0.3947, 0.4684):
        cells = sorted((r.t_c for r in g if abs(r.p_gpa - pr) < 1e-3 and any(_phase_matches("sill", p.name) for p in r.stable_phases)))
        rows[pr] = (round(cells[0]) if cells else None)
    log(f"draw {k:2d}: sill-in T at P=0.39 -> {rows[0.3947]}, at P=0.47 -> {rows[0.4684]}")
rm = region(mm)
bf = [(float(t_ax[j]), float(p_ax[i]), float(m)) for i, j, m in agg["best_fits"]]
# per-draw accepted-region bounds
per_draw = []
for g in grids:
    if g is None:
        continue
    per_draw.append(region(np.asarray(_modal_misfit_grid(g, modes, NP, NT, **MK)[0], dtype=float)))
T_lo_draws = sorted(round(r["T_lo"]) for r in per_draw)
T_hi_draws = sorted(round(r["T_hi"]) for r in per_draw)
best_T = sorted(round(b[0]) for b in bf)
best_P = sorted(round(b[1], 2) for b in bf)

summary = {
    "n_runs": agg.get("n_runs"), "n_failed": agg.get("n_failed"), "sigma_abs_wt": sig, "bias": bias, "rmse": rmse,
    "stability_set": OBS, "calibrated_unperturbed_region": r0,
    "stability_ge_0.5": half, "stability_eq_1": full, "stability_max": float(sp.flat[imax]),
    "stability_no_sill_ge_0.5": extent(sp4 >= 0.5), "diag": diag, "misfit_mean_region": rm,
    "best_fit_T_C": best_T, "best_fit_P_GPa": best_P, "best_fit_T_median": float(np.median(best_T)),
    "best_fit_P_median": float(np.median(best_P)),
    "per_draw_T_lo": T_lo_draws, "per_draw_T_hi": T_hi_draws,
    "elapsed_s": round(time.time() - t0),
}
log(json.dumps(summary, indent=1))
(FIG / "mc_calibrated_summary.json").write_text(json.dumps(summary, indent=1))

# fig_pt.load_mc-compatible export (stability block only) + long format
with open(FIG / "monte_carlo_calibrated.csv", "w", newline="") as f:
    f.write(f"# Monte Carlo (calibrated-map bulk),n={N},sigma=per-element |hold-out bias| wt%,seed={SEED}\n\n")
    f.write("stability_prob," + ",".join(f"T={t:.1f}" for t in t_ax) + "\n")
    for i, p in enumerate(p_ax):
        f.write(f"P={p:.3f}," + ",".join(f"{v:.4f}" for v in sp[i]) + "\n")
with open(FIG / "monte_carlo_calibrated_long.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T_C", "P_GPa", "stability_prob", "misfit_mean", "misfit_std", "misfit_unperturbed"])
    for i, p in enumerate(p_ax):
        for j, t in enumerate(t_ax):
            w.writerow([f"{t:.1f}", f"{p:.3f}", f"{sp[i, j]:.4f}", f"{mm[i, j]:.4f}", f"{ms[i, j]:.4f}", f"{mis0[i, j]:.4f}"])
log("wrote monte_carlo_calibrated.csv / _long.csv / mc_calibrated_samples.csv / mc_calibrated_summary.json")
log(f"total {time.time() - t0:.0f} s")
