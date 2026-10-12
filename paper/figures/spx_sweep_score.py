"""Score P-T-vs-scale sweeps into pt_scale*.csv (region reduction, all columns).

  --which eds   score the calibrated-map sweep written by eds_sweep_run.py
                (eds_sweep_grids.json; line-aware nets, 2026-09-16) -> pt_scale.csv
                [--eds-from-session scores the July session's sideband-net
                sweep instead, for the record]
  --which spx   score the SPX-source sweep written by spx_sweep_run.py
                (spx_sweep_grids.json) -> pt_scale_spx.csv
  --which both  (default)

Scoring = fig_pt.py's inputs (unmix-corrected modes, confidence weights,
sillimanite presence-only) and viewer_thermo_helpers._pt_region_from_misfit
with delta = REGION_DELTA, unioned per level by _pt_scale_rows.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "_Manuscript" / "figures"
SESSION = Path(r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.json")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("fig_pt", FIG / "fig_pt.py")
fig_pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig_pt)
from viewer_thermo_helpers import (_deserialize_thermod_result,      # noqa: E402
                                   _modal_misfit_grid, _pt_region_from_misfit,
                                   _pt_scale_rows)
from thermo.result import result_from_portable_dict                 # noqa: E402

REGION_DELTA = fig_pt.REGION_DELTA
modes, wts = fig_pt.observed_from_exports()
MK = dict(weights=wts, presence_keys=list(fig_pt.PRESENCE), open_other_tol=0.0)


def region(results, p_range, t_range, n_p, n_t):
    mis = _modal_misfit_grid(results, modes, n_p, n_t, **MK)[0]
    return _pt_region_from_misfit(mis, p_range, t_range, n_p, n_t, delta=REGION_DELTA)


def write_rows(rows, path):
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda d: -d["window_px"]):
            w.writerow(["" if (isinstance(r[c], float) and np.isnan(r[c]))
                        else (f"{r[c]:.4f}" if c.startswith("P_") else f"{r[c]:.1f}")
                        for c in cols])
    print(f"wrote {path.name}: {len(rows)} levels, {len(cols)} columns")


def show(rows, label):
    print(f"--- {label} (delta {REGION_DELTA}) ---")
    print(f"{'window_px':>9} {'T_best':>6} {'T mean-region':>14} {'T union':>12} {'P_best':>6} {'P mean-region':>14} {'P union':>12}")
    for r in rows:
        print(f"{r['window_px']:9.0f} {r['T_best_C']:6.0f} {r['T_lo_mean_C']:6.0f}-{r['T_hi_mean_C']:<6.0f} "
              f"{r['T_lo_C']:5.0f}-{r['T_hi_C']:<5.0f} {r['P_best_GPa']:6.2f} {r['P_lo_mean_GPa']:6.2f}-{r['P_hi_mean_GPa']:<6.2f} "
              f"{r['P_lo_GPa']:5.2f}-{r['P_hi_GPa']:<5.2f}")


ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["eds", "spx", "both"], default="both")
ap.add_argument("--eds-from-session", action="store_true",
                help="score the July session's (sideband-net) calibrated sweep instead of eds_sweep_grids.json")
args = ap.parse_args()


def load_grids_file(path):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def rows_from_grids_file(path, roi):
    g = load_grids_file(path)
    by_level = {}
    for key, portable in g["grids"].items():
        lv, rep = key.split("|")
        res = [result_from_portable_dict(d) for d in portable]
        reg = region(res, tuple(g["p_range"]), tuple(g["t_range"]), int(g["n_p"]), int(g["n_t"]))
        if reg is not None:
            by_level.setdefault(int(lv), {})[rep] = reg
    return _pt_scale_rows(by_level, float(roi[1] - roi[0]), float(roi[3] - roi[2]))


def grids_path(stem):
    # newest first: a fresh .json beside an older .gz must win
    cands = [c for c in (FIG / f"{stem}.json.gz", FIG / f"{stem}.json") if c.exists()]
    if cands:
        return max(cands, key=lambda c: c.stat().st_mtime)
    raise SystemExit(f"{stem}.json(.gz) not found in {FIG}")

roi = None
if args.which in ("eds", "both") and not args.eds_from_session:
    s = json.load(open(SESSION, encoding="utf-8"))
    roi = tuple(int(v) for v in s["comp_scale"]["roi_bbox"])
    rows = rows_from_grids_file(grids_path("eds_sweep_grids"), roi)
    show(rows, "EDS (calibrated-map, line-aware nets) sweep")
    write_rows(rows, FIG / "pt_scale.csv")
elif args.which in ("eds", "both"):
    s = json.load(open(SESSION, encoding="utf-8"))
    roi = tuple(int(v) for v in s["comp_scale"]["roi_bbox"])
    by_level = {}
    for r in s["thermod_runs"]:
        o = r.get("opts") or {}
        if r.get("mode") == "grid" and o.get("sweep_level") is not None:
            res = [_deserialize_thermod_result(d) for d in r["results"]]
            reg = region(res, o["p_range"], o["t_range"], int(o["n_p"]), int(o["n_t"]))
            if reg is not None:
                by_level.setdefault(int(o["sweep_level"]), {})[str(o["sweep_rep"])] = reg
    rows = _pt_scale_rows(by_level, float(roi[1] - roi[0]), float(roi[3] - roi[2]))
    old = FIG / "pt_scale.csv"
    if old.exists():
        import pandas as pd
        o_df = pd.read_csv(old).sort_values("window_px", ascending=False).reset_index(drop=True)
        n_df = pd.DataFrame(sorted(rows, key=lambda d: -d["window_px"]))
        same = all(np.allclose(o_df[c].to_numpy(float), n_df[c].to_numpy(float), atol=0.06)
                   for c in ["window_px", "T_best_C", "T_lo_C", "T_hi_C", "P_best_GPa", "P_lo_GPa", "P_hi_GPa"]
                   if c in o_df.columns)
        print("eds: original 7 columns unchanged vs the staged pt_scale.csv:", same)
    show(rows, "EDS (calibrated-map) sweep")
    write_rows(rows, FIG / "pt_scale.csv")

if args.which in ("spx", "both"):
    if roi is None:
        s = json.load(open(SESSION, encoding="utf-8"))
        roi = tuple(int(v) for v in s["comp_scale"]["roi_bbox"])
    rows = rows_from_grids_file(grids_path("spx_sweep_grids"), roi)
    show(rows, "SPX-source sweep")
    write_rows(rows, FIG / "pt_scale_spx.csv")
