"""Tier-3 check for Fig. 3c: the P-T-vs-scale sweep on the SPX source.

Reuses the EXACT quadtree windows of the July composition-vs-scale run (stored
in the viewer session's ``comp_scale.records``), computes each window's bulk
from the per-tile SPX quantifications the way the viewer's SPX source does
(tiles >= 95% inside the window, weighted by SPX NetIntens, renormalised to
100 wt%), builds the {mean, mean+sigma, mean-sigma} PC1 variants per level
with the viewer's own helper, and runs the same MAGEMin grid as the July EDS
sweep (mpe, Fe3+/Fe 0.1, +4 wt% H2O, 300-800 C, 0.1-1.0 GPa, 20x20).

Levels are limited by the SPX floor (a window must be able to hold a whole
tile): here L0-L5 (18 mm down to ~0.6 mm), which covers the L4->L5 breakdown
seen on the calibrated-map sweep.

Outputs (this directory):
  spx_windows.csv         per-window SPX bulk (level, bbox, wt%) for L0-L5
  spx_sweep_comps.json    per-level variants actually modelled
  spx_sweep_grids.json.gz the 18 grids as portable result dicts (cache-independent)
Scoring into pt_scale_spx.csv is done by spx_sweep_score.py.
"""
import csv
import json
import sys
import time
from pathlib import Path
from _dataset import DATASET_DIR   # dataset folder: env EQUIMAP_DATASET

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = DATASET_DIR
FIG = REPO / "_Manuscript" / "figures"
SESSION = DATA / "viewer_session_22_07_26.json"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))

from eds_map import read_manifest                                  # noqa: E402
from viewer_helpers import (_row_bbox_on_canvas, _overlap_area,    # noqa: E402
                            _match_spx_for_bcf, _parse_spx_quant)
from viewer_thermo_helpers import (_pca_perturbed_compositions,    # noqa: E402
                                   _max_useful_quadtree_depth)
from thermo import Composition, ThermoCalc                         # noqa: E402
from thermo.result import result_to_portable_dict                  # noqa: E402

P_RANGE, T_RANGE, NP, NT = (0.1, 1.0), (300.0, 800.0), 20, 20
FE3, H2O = 0.1, 4.0
MIN_OVERLAP = 0.95


def log(*a):
    print(*a, flush=True)


t0 = time.time()
manifest = DATA / "stitch_manifest.csv"
rows = read_manifest(manifest)
base_dir = manifest.parent.resolve()
tiles = [(bb, r) for r in rows if (bb := _row_bbox_on_canvas(r)) is not None]
log(f"{len(tiles)} tiles")

_spx_cache = {}


def spx_for_row(r):
    rel = r.get("file") or r.get("path") or ""
    p = (base_dir / rel).resolve()
    if not p.exists():
        p = (base_dir / Path(rel).name).resolve()
    if p in _spx_cache:
        return _spx_cache[p]
    out = ({}, 0.0, [])
    if p.exists():
        spx = _match_spx_for_bcf(p)
        if spx is not None and spx.exists():
            out = _parse_spx_quant(spx)
    _spx_cache[p] = out
    return out


def spx_for_bbox(bbox):
    sum_w, sum_wt, n = 0.0, {}, 0
    for bb, r in tiles:
        a_tile = (bb[1] - bb[0]) * (bb[3] - bb[2])
        if a_tile <= 0 or _overlap_area(bb, bbox) / a_tile < MIN_OVERLAP:
            continue
        comp, w, _ = spx_for_row(r)
        if not comp:
            continue
        n += 1
        sum_w += w
        for el, v in comp.items():
            sum_wt[el] = sum_wt.get(el, 0.0) + v * w
    if n == 0:
        return None, 0
    c = {el: v / sum_w for el, v in sum_wt.items()} if sum_w > 0 else dict(sum_wt)
    s = sum(c.values())
    return ({k: 100.0 * v / s for k, v in c.items()} if s > 0 else c), n


s = json.load(open(SESSION, encoding="utf-8"))
cs = s["comp_scale"]
roi = tuple(int(v) for v in cs["roi_bbox"])
tile_short = float(np.median([min(bb[1] - bb[0], bb[3] - bb[2]) for bb, _ in tiles]))
max_depth = _max_useful_quadtree_depth(roi, int(tile_short))
log(f"ROI {roi}; median tile short side {tile_short:.0f} px -> SPX max depth L{max_depth}")

records = [r for r in cs["records"] if int(r["level"]) <= max_depth]
win_rows, wins_by_level, tot_by_level = [], {}, {}
for r in records:
    lv = int(r["level"])
    tot_by_level[lv] = tot_by_level.get(lv, 0) + 1
    comp, n = spx_for_bbox(tuple(int(v) for v in r["bbox"]))
    if comp is None:
        continue
    wins_by_level.setdefault(lv, []).append(comp)
    y0, y1, x0, x1 = r["bbox"]
    win_rows.append(dict(level=lv, y0=y0, y1=y1, x0=x0, x1=x1, n_tiles=n, **comp))
els_all = sorted({e for w in win_rows for e in w if e not in ("level", "y0", "y1", "x0", "x1", "n_tiles")})
with open(FIG / "spx_windows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["level", "y0", "y1", "x0", "x1", "n_tiles"] + els_all)
    w.writeheader()
    for row in win_rows:
        w.writerow({k: row.get(k, "") for k in w.fieldnames})
for lv in sorted(tot_by_level):
    log(f"L{lv}: {len(wins_by_level.get(lv, []))} / {tot_by_level[lv]} windows have >=1 whole SPX tile")

comps = {}
for lv, wins in sorted(wins_by_level.items()):
    els = sorted({e for w in wins for e in w})
    mean_c, plus_c, minus_c = _pca_perturbed_compositions(wins, els, n_sigma=1.0)
    comps[lv] = {"n_windows": len(wins), "n_total": tot_by_level[lv], "elements": els,
                 "variants": {"mean": mean_c, "mean+sigma": plus_c, "mean-sigma": minus_c}}
    log(f"L{lv} mean: " + " ".join(f"{e} {mean_c.get(e, 0):.2f}" for e in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca", "O")))
(FIG / "spx_sweep_comps.json").write_text(json.dumps(comps, indent=1))


def build(comp_el):
    import model_bulk      # shared recipe: elements passed, Fe3+, H2O as wt% of the bulk
    return model_bulk.model_composition(comp_el, fe3=FE3, h2o=H2O)


calc = ThermoCalc("magemin", db="mpe", fe3_fraction=FE3)
grids = {}
for lv, d in sorted(comps.items()):
    for rep, comp in d["variants"].items():
        if not comp or sum(comp.values()) <= 0:
            continue
        t1 = time.time()
        res = calc.run_grid(build(comp), P_RANGE, T_RANGE, n_p=NP, n_t=NT)
        grids[f"{lv}|{rep}"] = [result_to_portable_dict(r) for r in res]
        log(f"L{lv} {rep:<10s}: {len(res)} cells in {time.time() - t1:.0f} s")
        (FIG / "spx_sweep_grids.json").write_text(json.dumps(
            {"p_range": P_RANGE, "t_range": T_RANGE, "n_p": NP, "n_t": NT,
             "backend": {"db": "mpe", "fe3_fraction": FE3, "h2o": H2O}, "grids": grids}))
# The staged dump is the gzip (the scorer reads it); always refresh it so a re-run
# cannot leave a stale .gz beside a fresh .json.
import gzip
import shutil
_raw = FIG / "spx_sweep_grids.json"
if _raw.exists():
    with open(_raw, "rb") as fi, gzip.open(FIG / "spx_sweep_grids.json.gz", "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    _raw.unlink()
log(f"done: {len(grids)} grids in {time.time() - t0:.0f} s -> spx_sweep_grids.json.gz")
