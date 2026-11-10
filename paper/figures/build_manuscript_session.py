"""Build the viewer session of the worked example as the manuscript presents it.

The July 2026 session (viewer_session_22_07_26.json/.npz) holds the band maps
of the whole section on the 1/5 grid. Everything the paper reports was later
derived from those band maps by the headless scripts of README.md, with the
line-aware net-count background (September 2026). That session's own phase
map, calibration, compositions and modelling runs predate the correction, so
opening it in the viewer does not show the paper's results.

This script writes a session that does: the same band maps, plus the paper's
state rebuilt from the staged files of this folder and the viewer's own code
paths, so the viewer shows what the paper shows and recomputes it the same way:

  background        line-aware
  calibration       eds_calibration.json (all 2,014 SPX tiles)
  ROI bulk          net-count fractions and calibrated bulk on line-aware nets
  phase map         phase_map_labels.npz + phase_legend.csv, mineral
                    assignments (segmentation_report.json), per-domain
                    compositions (computed as the phase dialog does), the
                    unmixing and recipe (segmentation_params.json)
  P-T settings      the paper's (fig_pt.py): chlorite left out, sillimanite as
                    a presence-only phase, domain weights = top identification
                    score, no open-assemblage tolerance
  modelling runs    run 1 = the Fig. 5 grid (section SPX bulk, model_bulk.py,
                    pt_grid_spec.py), computed with MAGEMin (grid cache aware);
                    runs 2-28 = the calibrated-map scale sweep
                    (eds_sweep_grids.json.gz) with its level / variant tags
  comp. vs scale    the windows of composition_scale.csv

The manifest is stored as "stitch_manifest.csv" (relative), so the session
matches the dataset folder it is unpacked beside.

    python build_manuscript_session.py [--out <dir>] [--check-only]

Checks printed at the end: the Fig. 5 grid against pt_grid.csv, the
observed modes and accepted region against fig_pt.py, and the whole-section
calibrated bulk against level 0 of composition_scale.csv.
"""
import argparse
import csv
import gzip
import json
import sys
import time
import types
from pathlib import Path
from _dataset import DATASET_DIR, SESSION_JSON, SESSION_NPZ   # dataset folder: env EQUIMAP_DATASET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--session-json", default=str(SESSION_JSON))
ap.add_argument("--session-npz", default=str(SESSION_NPZ))
ap.add_argument("--out", default=str(DATASET_DIR / "viewer_session_manuscript.json"))
ap.add_argument("--manifest-sha", default="",
                help="calibration fingerprint to record (SHA-256 of the published manifest)")
ap.add_argument("--check-only", action="store_true", help="build and check, write nothing")
args = ap.parse_args()
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)


import eds_background as bg                                       # noqa: E402
bg.set_default_scheme("line-aware")
import eds_quantify as eq                                         # noqa: E402
import model_bulk                                                 # noqa: E402
import fig_pt                                                     # noqa: E402
from pt_grid_spec import P_RANGE, T_RANGE, NP, NT                 # noqa: E402
from thermo import ThermoCalc                                     # noqa: E402
from thermo.result import result_from_portable_dict               # noqa: E402
from viewer_thermo_helpers import (_net_count_fractions, _serialize_thermod_result,   # noqa: E402
                                   _modal_misfit_grid, _pt_region_from_misfit)
from viewer_workers import _CompositionScaleWorker                # noqa: E402
from viewer import EDSControlPanel                                # noqa: E402

july = json.load(open(args.session_json, encoding="utf-8"))
log(f"July session: {len(july['thermod_runs'])} runs, {len(july['comp_scale']['records'])} windows")

# ---- calibration --------------------------------------------------------------
cal_file = json.load(open(HERE / "eds_calibration.json"))
cal = eq.Calibration.from_dict(cal_file["calibration"])
assert cal.background == "line-aware", cal.background
k = cal.k()

# ---- band maps (raw window sums; the background is applied on top) ------------
z = np.load(args.session_npz)
band_maps = {}
for key in z.files:
    if "__" in key:
        el, band = key.split("__", 1)
        band_maps.setdefault(el, {})[band] = z[key].astype(np.float32)
H, W = band_maps["Si"]["main"].shape
log(f"band maps: {len(band_maps)} elements, {H} x {W}")

frac_roi, _ = _net_count_fractions(band_maps)
cal_roi = eq.apply_calibration_fractions(frac_roi, k)

# ---- phase map and its classification -----------------------------------------
labels = np.load(HERE / "phase_map_labels.npz")["labels"].astype(np.int16)
assert labels.shape == (H, W), (labels.shape, (H, W))
names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(HERE / "phase_legend.csv"))}
report = json.load(open(HERE / "segmentation_report.json"))
minerals = {int(c): m for c, m in report["assignments"].items()}
seg_params = json.load(open(HERE / "segmentation_params.json"))
unmix = {int(c): {"fractions": {int(e): float(f) for e, f in d["fractions"].items()},
                  "residual": float(d.get("residual", float("nan")))}
         for c, d in (seg_params.get("unmix") or {}).items()}

comps = {}
for cid in sorted(names):
    f, n = _net_count_fractions(band_maps, sub_mask=(labels == cid))
    wt = eq.apply_calibration_fractions(f, k) if f else {}
    tot = sum(wt.values())
    comps[cid] = {el: v / tot for el, v in wt.items()} if tot > 0 else dict(f)
log(f"per-domain compositions for {len(comps)} domains")

# P-T settings of the paper (fig_pt.py): domains whose name has a key, minus the
# presence-only phases; the unmixed matrix domains are kept so their pixels are
# redistributed; weight = the domain's top identification score.
top1 = {r["name"].strip(): float(r["top1_score"])
        for r in csv.DictReader(open(HERE / "phase_minerals.csv")) if r.get("top1_score")}
scored = {cid for cid, nm in names.items()
          if fig_pt.NAME_TO_KEY.get(nm) and fig_pt.NAME_TO_KEY[nm] not in fig_pt.PRESENCE}
pt_include = sorted(scored | set(unmix))
pt_weights = {cid: top1[names[cid]] for cid in sorted(scored) if names[cid] in top1}
pt_extra = [{"key": k_, "mode": None, "weight": 1.0} for k_ in fig_pt.PRESENCE]

panel = types.SimpleNamespace(
    last_phase_pt_include=set(pt_include), last_phase_pt_weights=pt_weights,
    last_phase_extra=pt_extra, last_phase_open_tol=0.0,
    last_phase_unmix=unmix, last_phase_map=labels,
    _confidence_weight=EDSControlPanel._confidence_weight)
inputs = EDSControlPanel._assemblage_inputs_from_state(panel, minerals, names)
log(f"observed modes (viewer path): { {k_: round(v, 4) for k_, v in inputs['modes'].items()} }")

# ---- run 1: the Fig. 5 grid -----------------------------------------------------
july_runs = {r["index"]: r for r in july["thermod_runs"]}
spx_run = july_runs[3]
spx_el = {e: float(v) for e, v in spx_run["comp_el"].items()}
kept, _dropped = model_bulk.split_elements(spx_el)
calc = ThermoCalc("magemin", db=model_bulk.DATABASE, fe3_fraction=model_bulk.FE3_FRACTION)
log("Fig. 5 grid: MAGEMin (served from the grid cache when present) ...")
grid = calc.run_grid(model_bulk.model_composition(spx_el), P_RANGE, T_RANGE, n_p=NP, n_t=NT)
log(f"Fig. 5 grid: {len(grid)} cells")

base_opts = {"backend": "magemin",
             "backend_kwargs": {"db": model_bulk.DATABASE, "fe3_fraction": model_bulk.FE3_FRACTION},
             "fe3": model_bulk.FE3_FRACTION, "h2o": model_bulk.H2O_WT_PCT, "co2": 0.0,
             "extra_basis": "bulk", "mode": "grid", "p_gpa": 1.0, "t_c": 700.0,
             "workflow": "both", "assem_method": "after"}
misfit_opts = {"observed_phases": sorted(inputs["modes"]),
               "observed_modes": {str(k_): float(v) for k_, v in inputs["modes"].items()},
               "modal_weights": {str(k_): float(v) for k_, v in inputs["weights"].items()},
               "presence_keys": list(inputs["presence_keys"]),
               "segmentation_params": seg_params}


def record(index, comp_el, opts, results, label_extra=""):
    r = {k_: spx_run[k_] for k_ in ("comp_source", "roi_bbox", "roi_shape")}
    r.update({"id": f"ms{index:03d}", "index": index, "timestamp": "2026-11-10T00:00:00",
              "mode": "grid", "opts": dict(opts, comp_el=dict(comp_el)), "comp_el": dict(comp_el),
              "manifest": "stitch_manifest.csv",
              "roi_label": f"ROI #{index} [y=0:{spx_run['roi_bbox'][1]}, x=0:{spx_run['roi_bbox'][3]}]{label_extra}",
              "results": [_serialize_thermod_result(x) for x in results]})
    return r


runs = [record(1, kept, dict(base_opts, p_range=list(P_RANGE), t_range=list(T_RANGE), n_p=NP, n_t=NT,
                             comp_label="section SPX bulk (Fig. 5)", **misfit_opts), grid,
               " section SPX bulk (Fig. 5)")]

# ---- runs 2-28: the calibrated-map scale sweep -----------------------------------
sweep = json.load(gzip.open(HERE / "eds_sweep_grids.json.gz"))
scomps = json.load(open(HERE / "eds_sweep_comps.json"))
parent = july["comp_scale"]["parent_label"]
for key in sorted(sweep["grids"], key=lambda s: (int(s.split("|")[0]), ["mean", "mean+sigma", "mean-sigma"].index(s.split("|")[1]))):
    lv, rep = key.split("|")
    comp = scomps[lv]["variants"][rep]
    kept_v, _ = model_bulk.split_elements(comp)
    res = [result_from_portable_dict(d) for d in sweep["grids"][key]]
    opts = dict(base_opts, p_range=list(sweep["p_range"]), t_range=list(sweep["t_range"]),
                n_p=int(sweep["n_p"]), n_t=int(sweep["n_t"]), observed_phases=[],
                sweep_level=int(lv), sweep_rep=rep, sweep_parent_label=parent)
    runs.append(record(len(runs) + 1, kept_v, opts, res, f" sweep L{lv} {rep}"))
log(f"{len(runs)} modelling runs")

# ---- composition vs scale ------------------------------------------------------
cs_records = []
with open(HERE / "composition_scale.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        rec = {"bbox": [int(row["y0"]), int(row["y1"]), int(row["x0"]), int(row["x1"])],
               "level": int(row["level"]), "mask_frac": float(row["mask_frac"]),
               "n_pixels": int(row["n_pixels"])}
        if row["skipped"]:
            rec["skipped"] = row["skipped"]
        else:
            rec["wt_pct"] = {el: float(v) for el, v in row.items()
                             if el not in ("level", "y0", "y1", "x0", "x1", "n_pixels", "mask_frac", "skipped")
                             and v != ""}
        cs_records.append(rec)
cs_summary = _CompositionScaleWorker._summarise(cs_records)
cs_opts = dict(july["comp_scale"]["opts"], data_type="eds_calibrated_wt_pct",
               background="line-aware")

# ---- checks ---------------------------------------------------------------------
import pandas as pd                                               # noqa: E402
g_csv, keys_csv = fig_pt.load_grid(HERE / "pt_grid.csv")
res_csv = fig_pt.build_results(g_csv, keys_csv, REPO)
dmax = 0.0
for a, b in zip(res_csv, grid):
    va = {}
    for p in a.stable_phases:
        va[p.name] = va.get(p.name, 0.0) + (p.vol_fraction or 0.0)
    vb = {}
    for p in b.stable_phases:
        vb[p.name] = vb.get(p.name, 0.0) + (p.vol_fraction or 0.0)
    dmax = max([dmax] + [abs(va.get(n, 0.0) - vb.get(n, 0.0)) for n in set(va) | set(vb)])
print(f"CHECK Fig. 5 grid vs pt_grid.csv: max |d vol fraction| = {dmax:.2e}")

m_paper, w_paper = fig_pt.observed_from_exports()
mis_paper = np.asarray(_modal_misfit_grid(res_csv, m_paper, NP, NT, weights=w_paper,
                                          presence_keys=fig_pt.PRESENCE)[0], float)
mis_sess = np.asarray(_modal_misfit_grid(grid, inputs["modes"], NP, NT, weights=inputs["weights"],
                                         presence_keys=inputs["presence_keys"])[0], float)
reg_p = _pt_region_from_misfit(mis_paper, P_RANGE, T_RANGE, NP, NT, delta=fig_pt.REGION_DELTA)
reg_s = _pt_region_from_misfit(mis_sess, P_RANGE, T_RANGE, NP, NT, delta=fig_pt.REGION_DELTA)
print(f"CHECK misfit grids: max |diff| = {np.nanmax(np.abs(mis_paper - mis_sess)):.2e}")
print(f"CHECK accepted region  paper {reg_p}\n                       session {reg_s}")
print(f"CHECK modes  paper   { {k_: round(v, 4) for k_, v in m_paper.items()} }")
print(f"             session { {k_: round(v, 4) for k_, v in inputs['modes'].items()} }")
l0 = next(r for r in cs_records if r["level"] == 0 and "wt_pct" in r)["wt_pct"]
print("CHECK section calibrated bulk vs composition_scale.csv L0: max |d wt%| = "
      f"{max(abs(cal_roi.get(e, 0.0) - v) for e, v in l0.items()):.2e}")
if args.check_only:
    sys.exit(0)

# ---- write -----------------------------------------------------------------------
meta = {k_: july[k_] for k_ in ("available_elements", "selected_elements", "ui", "roi",
                                 "band_maps_ds", "blend_mode", "last_elements", "roi_shapes")
        if k_ in july}
meta.update({
    "manifest": "stitch_manifest.csv",
    "description": "EquiMap worked example (C23-S1-A) in the state the manuscript presents; "
                   "built by paper/figures/build_manuscript_session.py from the July 2026 band maps "
                   "and the staged results",
    "last_composition": frac_roi,
    "last_spx_composition": july.get("last_spx_composition"),
    "last_calibrated_composition": cal_roi,
    "eds_calibration": cal.to_dict(),
    "calibration_scope": "mosaic",
    "calibration_fingerprint": args.manifest_sha or july.get("calibration_fingerprint"),
    "background_scheme": "line-aware",
    "phase_segmentation": {
        "names": {str(c): n for c, n in names.items()},
        "translate": [0.0, 0.0],
        "ds": int(seg_params.get("seg_ds", 5)),
        "minerals": {str(c): m for c, m in minerals.items()},
        "fe3_fraction": float(seg_params.get("fe3_fraction", 0.1)),
        "unmix": {str(c): {"fractions": {str(e): f for e, f in d["fractions"].items()},
                           "residual": d["residual"]} for c, d in unmix.items()},
        "composition": {str(c): {el: round(float(v), 6) for el, v in f.items()} for c, f in comps.items()},
        "seg_params": seg_params,
        "coating_elements": ["C"],
        "pt_include": pt_include,
        "pt_weights": {str(c): w for c, w in pt_weights.items()},
        "pt_extra": pt_extra,
    },
    "thermod_runs": runs,
    "comp_scale": {"records": cs_records, "summary": {str(k_): v for k_, v in cs_summary.items()},
                   "opts": cs_opts, "parent_label": parent,
                   "roi_bbox": july["comp_scale"]["roi_bbox"]},
})
out = Path(args.out)
out.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
log(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB); writing the arrays ...")
arrays = {key: z[key] for key in z.files if "__" in key}
arrays["phase_map"] = labels
np.savez_compressed(out.with_suffix(".npz"), **arrays)
log(f"wrote {out.with_suffix('.npz')} ({out.with_suffix('.npz').stat().st_size / 1e9:.2f} GB)")
