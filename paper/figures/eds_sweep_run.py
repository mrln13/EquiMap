"""P-T-vs-scale sweep on the calibrated-map source (Fig. 3c filled bands).

Counterpart of spx_sweep_run.py for the EDS source: takes the per-window
calibrated compositions of composition_scale.csv (eds_windows_run.py; the
July quadtree windows on the line-aware nets), builds the {mean, mean+sigma,
mean-sigma} PC1 variants per level with the viewer's own helper
(_pca_perturbed_compositions, the sweep's definition), and runs the same
MAGEMin grid as the SPX sweep (mpe, Fe3+/Fe 0.1, +4 wt% H2O, 300-800 C,
0.1-1.0 GPa, 20x20). Levels default to L0-L5, the SPX-comparable range.

Outputs (this directory):
  eds_sweep_comps.json    per-level variants actually modelled
  eds_sweep_grids.json    the grids as portable result dicts (cache-independent)
Scoring into pt_scale.csv is done by spx_sweep_score.py --which eds.
Grids are served from the result cache when identical inputs were run before.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from viewer_thermo_helpers import _pca_perturbed_compositions      # noqa: E402
from thermo import Composition, ThermoCalc                         # noqa: E402
from thermo.result import result_to_portable_dict                  # noqa: E402

P_RANGE, T_RANGE, NP, NT = (0.1, 1.0), (300.0, 800.0), 20, 20
FE3, H2O = 0.1, 4.0

ap = argparse.ArgumentParser()
ap.add_argument("--windows", default=str(HERE / "composition_scale.csv"))
ap.add_argument("--max-level", type=int, default=5)
ap.add_argument("--min-level", type=int, default=0)
ap.add_argument("--append", action="store_true", help="merge into an existing *_grids.json(.gz) / *_comps.json")
ap.add_argument("--out-prefix", default=str(HERE / "eds_sweep"))
args = ap.parse_args()
t0 = time.time()


def log(*a):
    print(*a, flush=True)


META = {"level", "y0", "y1", "x0", "x1", "n_pixels", "mask_frac", "skipped"}
wins_by_level, tot_by_level = {}, {}
with open(args.windows, newline="") as f:
    for r in csv.DictReader(f):
        lv = int(r["level"])
        if lv > args.max_level or lv < args.min_level:
            continue
        tot_by_level[lv] = tot_by_level.get(lv, 0) + 1
        if r.get("skipped"):
            continue
        comp = {k: float(v) for k, v in r.items() if k not in META and v not in ("", None)}
        if sum(comp.values()) > 0:
            wins_by_level.setdefault(lv, []).append(comp)
comps = {}
for lv, wins in sorted(wins_by_level.items()):
    els = sorted({e for w in wins for e in w})
    mean_c, plus_c, minus_c = _pca_perturbed_compositions(wins, els, n_sigma=1.0)
    comps[lv] = {"n_windows": len(wins), "n_total": tot_by_level[lv], "elements": els,
                 "variants": {"mean": mean_c, "mean+sigma": plus_c, "mean-sigma": minus_c}}
    log(f"L{lv} ({len(wins)} windows) mean: " + " ".join(f"{e} {mean_c.get(e, 0):.2f}" for e in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca"))
        + " | +sigma Al %.2f  -sigma Al %.2f" % (plus_c.get("Al", 0), minus_c.get("Al", 0)))
import gzip
comps_path = Path(args.out_prefix + "_comps.json")
if args.append and comps_path.exists():
    prev = json.loads(comps_path.read_text()); prev.update({str(k): v for k, v in comps.items()}); comps_all = prev
else:
    comps_all = {str(k): v for k, v in comps.items()}
comps_path.write_text(json.dumps(comps_all, indent=1))


def build(comp_el):
    import model_bulk      # shared recipe: elements passed, Fe3+, H2O as wt% of the bulk
    return model_bulk.model_composition(comp_el, fe3=FE3, h2o=H2O)


calc = ThermoCalc("magemin", db="mpe", fe3_fraction=FE3)
grids = {}
out_grids = Path(args.out_prefix + "_grids.json")
gz_path = Path(args.out_prefix + "_grids.json.gz")
if args.append:
    src = gz_path if gz_path.exists() else out_grids
    if src.exists():
        opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt") as f:
            grids = json.load(f)["grids"]
        log(f"appending to {len(grids)} existing grids from {src.name}")
for lv, d in sorted(comps.items()):
    for rep, comp in d["variants"].items():
        if not comp or sum(comp.values()) <= 0:
            continue
        t1 = time.time()
        res = calc.run_grid(build(comp), P_RANGE, T_RANGE, n_p=NP, n_t=NT)
        grids[f"{lv}|{rep}"] = [result_to_portable_dict(r) for r in res]
        log(f"L{lv} {rep:<10s}: {len(res)} cells in {time.time() - t1:.0f} s")
        out_grids.write_text(json.dumps(
            {"p_range": P_RANGE, "t_range": T_RANGE, "n_p": NP, "n_t": NT,
             "backend": {"db": "mpe", "fe3_fraction": FE3, "h2o": H2O},
             "source": "eds_calibrated (line-aware nets, eds_windows_run.py)", "grids": grids}))
# The staged dump is the gzip (the scorer reads it); always refresh it so a re-run
# cannot leave a stale .gz beside a fresh .json.
if out_grids.exists():
    with open(out_grids, "rb") as fi, gzip.open(gz_path, "wb", compresslevel=6) as fo:
        import shutil; shutil.copyfileobj(fi, fo)
    out_grids.unlink()
log(f"done: {len(grids)} grids in {time.time() - t0:.0f} s -> {gz_path}")
