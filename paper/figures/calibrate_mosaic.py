"""Headless whole-mosaic SPX calibration (Section 3.3 / 5.1, Fig. 7 data).

Replays the viewer's "Calibrate EDS (SPX k-factors)" button on the C23S1A
mosaic without the GUI, through the same code path (eds_cache.CacheWindowReader
net sums subsampled to ~2000 px per tile, eds_quantify.filter_calibration_tiles,
derive_k_factors, cross_validate_k_factors with 50 seeded 20 % hold-out folds),
and writes

  calibration_validation.csv   long format tile/fold/element/predicted/measured/
                               residual, the input of fig_calibration.py
  eds_calibration.json         k-factors + r2, hold-out RMSE/bias per element,
                               the background plan, tile counts

Options: --background {line-aware,sideband} (default: the toolkit default),
--dataset, --out-dir, --samples-per-tile. Run time ~5 min (cache reads).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
import eds_background as bg                                   # noqa: E402
import eds_cache as edsc                                      # noqa: E402
import eds_quantify as eq                                     # noqa: E402
from viewer_helpers import (_row_bbox_on_canvas, _match_spx_for_bcf,   # noqa: E402
                            _parse_spx_quant)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=r"D:/Actief/C23S1A_Quant")
ap.add_argument("--out-dir", default=str(HERE))
ap.add_argument("--background", choices=list(bg.SCHEMES), default=None)
ap.add_argument("--samples-per-tile", type=int, default=2000)
ap.add_argument("--min-valid-frac", type=float, default=0.5)
ap.add_argument("--holdout-frac", type=float, default=0.2)
ap.add_argument("--n-repeats", type=int, default=50)
ap.add_argument("--coating", default="C")
args = ap.parse_args()
if args.background:
    bg.set_default_scheme(args.background)
scheme = bg.default_scheme()

DATA = Path(args.dataset)
OUT = Path(args.out_dir)
cache_dir = edsc.resolve_cache_dir(DATA)
man = edsc.load_manifest(cache_dir)
elements = list(man["elements"])
H, W = int(man["H"]), int(man["W"])
reader = edsc.CacheWindowReader(cache_dir, elements, blend_mode="mean")
print(f"dataset {DATA}: {H}x{W}, elements {elements}, background {scheme}")
print(bg.describe_plan(reader.plan, scheme))

rows = list(csv.DictReader(open(DATA / "stitch_manifest.csv", newline="")))
t0 = time.time()
spx_l, cnt_l, wt_l, vf_l, ids = [], [], [], [], []
for i, r in enumerate(rows):
    bb = _row_bbox_on_canvas(r)
    if bb is None:
        continue
    rel = r.get("file") or ""
    bcf = (DATA / rel).resolve()
    if not bcf.exists():
        bcf = (DATA / Path(rel).name).resolve()
    spx = _match_spx_for_bcf(bcf) if bcf.exists() else None
    if spx is None:
        continue
    comp, weight, _notes = _parse_spx_quant(spx)
    if not comp:
        continue
    y0 = max(0, min(bb[0], H)); y1 = max(0, min(bb[1], H))
    x0 = max(0, min(bb[2], W)); x1 = max(0, min(bb[3], W))
    if y1 <= y0 or x1 <= x0:
        continue
    area = (y1 - y0) * (x1 - x0)
    step = 1
    if area > args.samples_per_tile:
        step = max(1, min(int(round((area / args.samples_per_tile) ** 0.5)), y1 - y0, x1 - x0))
    sums = reader.net_sums((y0, y1, x0, x1), step=step)
    vf = reader.coverage_frac((y0, y1, x0, x1), step=step)
    spx_l.append(comp); cnt_l.append(sums); wt_l.append(float(weight)); vf_l.append(float(vf))
    ids.append(i)
    if (len(ids)) % 250 == 0:
        print(f"  {len(ids)} tiles read, {time.time() - t0:.0f} s", flush=True)
print(f"{len(ids)} SPX tiles read in {time.time() - t0:.0f} s")

n_cand = len(cnt_l)
spx_f, cnt_f, dropped = eq.filter_calibration_tiles(
    spx_l, cnt_l, weights=wt_l, valid_fracs=vf_l, min_valid_frac=args.min_valid_frac)
print(f"quality filter: {len(dropped)} of {n_cand} dropped")
drop = eq.ANION_ELEMENTS | (frozenset({args.coating}) if args.coating else frozenset())
cal = eq.derive_k_factors(spx_f, cnt_f, drop=drop, background=scheme)
cv = eq.cross_validate_k_factors(spx_f, cnt_f, drop=drop, method="holdout",
                                 holdout_frac=args.holdout_frac, n_repeats=args.n_repeats, seed=0)

print(f"\n{'el':<3s} {'k':>10s} {'n':>5s} {'r2':>7s} | {'RMSE':>6s} {'bias':>6s} {'+-fold':>6s}")
for el, kf in sorted(cal.k_factors.items()):
    st = cv.per_element.get(el) if cv else None
    print(f"{el:<3s} {kf.k:10.3e} {kf.n_tiles:5d} {('%.2f' % kf.r2) if kf.r2 is not None else '-':>7s} | "
          + (f"{st.rmse:6.2f} {st.bias:+6.2f} {st.rmse_std:6.2f}" if st else ""))
if cv:
    print(f"overall hold-out RMSE {cv.rmse:.2f} +- {cv.rmse_std:.2f} wt% over {cv.n_tiles} predictions, {cv.n_folds} folds")

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "calibration_validation.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["tile", "fold", "element", "predicted_wt_pct", "measured_wt_pct", "residual_wt_pct"])
    for rec in (cv.records if cv else []):
        fold = rec.get("fold", 0)
        for el in sorted(rec["predicted"]):
            p = rec["predicted"][el]; m = rec["measured"].get(el)
            if m is None:
                continue
            w.writerow([rec["tile"], fold, el, f"{p:.4f}", f"{m:.4f}", f"{p - m:.4f}"])
summary = {
    "dataset": str(DATA), "background": scheme, "n_manifest_tiles": len(rows),
    "n_spx_tiles": n_cand, "n_dropped_by_filter": len(dropped), "n_fitted": len(cnt_f),
    "samples_per_tile": args.samples_per_tile, "coating": args.coating,
    "calibration": cal.to_dict(),
    "holdout": None if cv is None else {
        "method": cv.method, "holdout_frac": args.holdout_frac, "n_repeats": args.n_repeats,
        "seed": 0, "n_predictions": cv.n_tiles, "n_folds": cv.n_folds,
        "rmse": cv.rmse, "rmse_std": cv.rmse_std,
        "per_element": {el: {"rmse": s.rmse, "bias": s.bias, "rmse_std": s.rmse_std, "n": getattr(s, "n", None)}
                        for el, s in cv.per_element.items()}},
    "plan": bg.describe_plan(reader.plan, scheme),
}
json.dump(summary, open(OUT / "eds_calibration.json", "w"), indent=1)
print(f"wrote {OUT / 'calibration_validation.csv'} and eds_calibration.json ({time.time() - t0:.0f} s)")
