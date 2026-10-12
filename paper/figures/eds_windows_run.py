"""Per-window calibrated compositions for the composition-vs-scale figure (Fig. 3).

Re-evaluates the EXACT quadtree windows of the July composition-vs-scale run
(stored in the viewer session's ``comp_scale.records``: 22,845 windows over
levels L0-L8, the same windows the SPX sweep re-used) on the session's 1/5
band maps with the line-aware signed net counts (eds_background) and the
k-factors of eds_calibration.json, and writes composition_scale.csv in the
layout of the viewer's export (level, y0, y1, x0, x1, n_pixels, mask_frac,
skipped, element wt% columns). The ROI was the whole rectangle (no polygon
mask), so mask_frac is 1 for every window.
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
import eds_background as bg                                       # noqa: E402
import eds_quantify as eq                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--session-json", default=r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.json")
ap.add_argument("--session-npz", default=r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.npz")
ap.add_argument("--calibration", default=str(HERE / "eds_calibration.json"))
ap.add_argument("--out", default=str(HERE / "composition_scale.csv"))
ap.add_argument("--background", choices=list(bg.SCHEMES), default=None)
ap.add_argument("--keep-session-windows", action="store_true",
                help="use the session's window list verbatim. Default: levels the session "
                     "had capped are re-drawn with the (fixed) uniform sampler, because the "
                     "session's capped levels cover only the top of the section")
ap.add_argument("--seed", type=int, default=0, help="seed of the re-drawn capped levels")
args = ap.parse_args()
if args.background:
    bg.set_default_scheme(args.background)
scheme = bg.default_scheme()
t0 = time.time()

cal = eq.Calibration.from_dict(json.load(open(args.calibration))["calibration"])
if cal.background != scheme:
    raise SystemExit(f"calibration background {cal.background!r} != scheme {scheme!r}")
k = cal.k()
sess = json.load(open(args.session_json, encoding="utf-8"))
cs = sess["comp_scale"]
records = cs["records"]
ry0, ry1, rx0, rx1 = (int(v) for v in cs["roi_bbox"])
ds = int(sess.get("band_maps_ds") or 1)
if not args.keep_session_windows:
    # The session's windows came from the viewer's capped random sampler, which
    # until 2026-10-02 kept only the lowest cell indices of a capped level (the
    # top of the ROI). Complete levels are kept as they are; capped levels are
    # re-drawn, same number of windows, uniformly over the ROI.
    from viewer_thermo_helpers import _quadtree_level_dims, _quadtree_sampled_windows
    by_lv = {}
    for r in records:
        by_lv.setdefault(int(r["level"]), []).append(r)
    records = []
    for lv in sorted(by_lv):
        n_rows, n_cols, _ = _quadtree_level_dims((ry0, ry1, rx0, rx1), lv)
        total = n_rows * n_cols
        if len(by_lv[lv]) >= total:
            records += by_lv[lv]
            continue
        wins = _quadtree_sampled_windows((ry0, ry1, rx0, rx1), lv, cap=len(by_lv[lv]),
                                         rng=np.random.default_rng(args.seed + lv))
        records += [{"level": lv, "bbox": list(w)} for w in wins]
        ys = np.array([w[0] for w in wins])
        print(f"L{lv}: re-drew {len(wins)} of {total} windows uniformly "
              f"(session's reached y0 <= {max(int(r['bbox'][0]) for r in by_lv[lv])} of {ry1}; "
              f"new y0 <= {int(ys.max())}, top-half share {np.mean(ys < (ry0 + ry1) / 2):.2f})", flush=True)
print(f"{len(records)} windows, ROI {(ry0, ry1, rx0, rx1)}, band maps at 1/{ds}, background {scheme}, "
      f"k from {cal.n_tiles} tiles", flush=True)

z = np.load(args.session_npz)
elements = sorted({key.split("__")[0] for key in z.keys() if "__" in key})
plan = bg.background_plan(elements)
nets = {}
for el in elements:
    nets[el] = bg.net_count_map(el, lambda e, b: z[f"{e}__{b}"] if f"{e}__{b}" in z else None, plan=plan)
H, W = nets[elements[0]].shape
print(f"signed nets ready ({time.time() - t0:.0f} s)", flush=True)

cal_els = [e for e in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca", "Ti", "P") if e in k] + \
          sorted(e for e in k if e not in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca", "Ti", "P"))
rows = []
n_skip = 0
for i, r in enumerate(records):
    lv = int(r["level"]); by0, by1, bx0, bx1 = (int(v) for v in r["bbox"])
    sy0, sy1 = max(0, (by0 - ry0) // ds), min(H, (by1 - ry0) // ds)
    sx0, sx1 = max(0, (bx0 - rx0) // ds), min(W, (bx1 - rx0) // ds)
    row = {"level": lv, "y0": by0, "y1": by1, "x0": bx0, "x1": bx1,
           "n_pixels": int(r.get("n_pixels", (sy1 - sy0) * (sx1 - sx0))),
           "mask_frac": float(r.get("mask_frac", 1.0)), "skipped": ""}
    if sy1 <= sy0 or sx1 <= sx0 or r.get("skipped"):
        row["skipped"] = r.get("skipped") or "empty"; n_skip += 1
    else:
        sums = {el: float(nets[el][sy0:sy1, sx0:sx1].sum()) for el in elements}
        clipped = {el: max(0.0, v) for el, v in sums.items()}
        tot = sum(clipped.values())
        if tot <= 0:
            row["skipped"] = "no signal"; n_skip += 1
        else:
            wt = eq.apply_calibration_fractions({el: v / tot for el, v in clipped.items()}, k)
            row.update({el: wt.get(el, 0.0) for el in cal_els})
    rows.append(row)
    if (i + 1) % 5000 == 0:
        print(f"  {i + 1} windows ({time.time() - t0:.0f} s)", flush=True)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["level", "y0", "y1", "x0", "x1", "n_pixels", "mask_frac", "skipped"] + cal_els)
    w.writeheader()
    for row in rows:
        w.writerow({c: row.get(c, "") for c in w.fieldnames})
by_level = {}
for row in rows:
    if not row["skipped"]:
        by_level.setdefault(row["level"], []).append(row)
for lv in sorted(by_level):
    al = np.array([r["Al"] for r in by_level[lv]]); si = np.array([r["Si"] for r in by_level[lv]])
    print(f"L{lv}: {len(by_level[lv]):5d} windows  Al {al.mean():.2f} +- {al.std():.2f}  Si {si.mean():.2f} +- {si.std():.2f}")
print(f"wrote {args.out}: {len(rows)} rows, {n_skip} skipped ({time.time() - t0:.0f} s)")
