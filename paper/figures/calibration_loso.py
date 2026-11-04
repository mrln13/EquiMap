# -*- coding: utf-8 -*-
"""Leave-one-session-out validation of the SPX-anchored calibration (section 5.1).

The C23S1A mosaic was acquired in three Bruker mapping jobs (2025-09-29,
-09-30, -10-01), each covering tiles over the whole grid. The published
validation is a random 20 % tile hold-out; with one slope per element and
2,014 tiles that is nearly a training residual. Here the tiles are loaded
exactly as calibrate_mosaic.py loads them, then:
  1. k fitted per session; relative differences between sessions, after
     normalising each session's k to its Si k (an overall intensity scale
     cancels in the renormalised fractions; only element-relative drift matters);
  2. leave-one-session-out: fit on two sessions, predict the third; per-element
     RMSE and bias defined exactly as in eds_quantify.cross_validate_k_factors
     (renormalised over the shared elements), compared with the random hold-out;
  3. per-session mean SPX composition and mean total counts per tile (beam /
     live-time proxy), to separate instrument drift from composition.
Writes calibration_loso.json beside this script; prints the tables.
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = Path(r"D:/Actief/C23S1A_Quant")
sys.path.insert(0, str(REPO))
import eds_background as bg  # noqa: E402
import eds_cache as edsc  # noqa: E402
import eds_quantify as eq  # noqa: E402
from viewer_helpers import _row_bbox_on_canvas, _match_spx_for_bcf, _parse_spx_quant  # noqa: E402

SAMPLES = 2000
scheme = bg.default_scheme()
cache_dir = edsc.resolve_cache_dir(DATA)
man = edsc.load_manifest(cache_dir)
elements = list(man["elements"])
H, W = int(man["H"]), int(man["W"])
reader = edsc.CacheWindowReader(cache_dir, elements, blend_mode="mean")
rows = list(csv.DictReader(open(DATA / "stitch_manifest.csv", newline="")))
t0 = time.time()
spx_l, cnt_l, wt_l, vf_l, sess = [], [], [], [], []
for i, r in enumerate(rows):
    bb = _row_bbox_on_canvas(r)
    if bb is None:
        continue
    rel = r.get("file") or ""
    m = re.search(r"Mapping job_(2025_\d\d_\d\d)", rel)
    bcf = (DATA / rel).resolve()
    if not bcf.exists():
        bcf = (DATA / Path(rel).name).resolve()
    spx = _match_spx_for_bcf(bcf) if bcf.exists() else None
    if spx is None or m is None:
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
    if area > SAMPLES:
        step = max(1, min(int(round((area / SAMPLES) ** 0.5)), y1 - y0, x1 - x0))
    spx_l.append(comp); cnt_l.append(reader.net_sums((y0, y1, x0, x1), step=step))
    wt_l.append(float(weight)); vf_l.append(float(reader.coverage_frac((y0, y1, x0, x1), step=step)))
    sess.append(m.group(1))
    if len(sess) % 250 == 0:
        print(f"  {len(sess)} tiles, {time.time() - t0:.0f} s", flush=True)
print(f"{len(sess)} SPX tiles read in {time.time() - t0:.0f} s; sessions {sorted(set(sess))}")
spx_f, cnt_f, dropped = eq.filter_calibration_tiles(spx_l, cnt_l, weights=wt_l, valid_fracs=vf_l, min_valid_frac=0.5)
assert not dropped, dropped
drop = eq.ANION_ELEMENTS | frozenset({"C"})
sessions = sorted(set(sess))
sess = np.array(sess)


def fit(idx):
    return eq.derive_k_factors([spx_l[i] for i in idx], [cnt_l[i] for i in idx], drop=drop, background=scheme).k()


def score(idx, k):
    """Residuals exactly as cross_validate_k_factors: renormalised over shared elements."""
    res, meas_acc, recs = {}, {}, []
    for t in idx:
        pred = eq.apply_calibration_fractions(cnt_l[t], k)
        if not pred:
            continue
        meas = {el: float(v) for el, v in spx_l[t].items() if el in pred and float(v) > 0}
        shared = sorted(set(pred) & set(meas))
        if len(shared) < 2:
            continue
        pt, mt = sum(pred[e] for e in shared), sum(meas[e] for e in shared)
        for e in shared:
            r = 100 * pred[e] / pt - 100 * meas[e] / mt
            res.setdefault(e, []).append(r); meas_acc.setdefault(e, []).append(100 * meas[e] / mt)
    out = {e: {"rmse": float(np.sqrt(np.mean(np.square(v)))), "bias": float(np.mean(v)), "n": len(v),
               "mean_measured": float(np.mean(meas_acc[e]))} for e, v in res.items()}
    allr = np.concatenate([np.asarray(v) for v in res.values()])
    out["_overall"] = {"rmse": float(np.sqrt(np.mean(allr ** 2))), "n": int(allr.size)}
    return out


all_idx = list(range(len(sess)))
k_all = fit(all_idx)
els = sorted(e for e in k_all if e != "P")
report = {"sessions": {}, "loso": {}, "k_all": {e: k_all[e] for e in k_all}}
print("\n=== per-session k relative to the pooled k, after normalising to Si (intensity scale cancels) ===")
print(f"{'session':10s} {'tiles':>5s} " + " ".join(f"{e:>7s}" for e in els) + "   mean counts/px (all cations)")
for s in sessions:
    idx = [i for i in all_idx if sess[i] == s]
    ks = fit(idx)
    rel = {e: (ks[e] / ks["Si"]) / (k_all[e] / k_all["Si"]) - 1 for e in els if e in ks}
    tot = float(np.mean([sum(v for e, v in cnt_l[i].items() if e not in ("O", "C")) for i in idx]))
    spx_mean = {e: float(np.mean([spx_l[i].get(e, 0.0) for i in idx])) for e in els}
    report["sessions"][s] = {"n": len(idx), "k": ks, "rel_k_vs_pooled_Si_normalised": rel, "mean_counts": tot, "mean_spx": spx_mean}
    print(f"{s:10s} {len(idx):5d} " + " ".join(f"{100 * rel[e]:+6.1f}%" for e in els) + f"   {tot:,.0f}")
print("\n=== per-session mean SPX wt% (composition confound check) ===")
print(f"{'session':10s} " + " ".join(f"{e:>6s}" for e in els))
for s in sessions:
    print(f"{s:10s} " + " ".join(f"{report['sessions'][s]['mean_spx'][e]:6.2f}" for e in els))

hold = json.load(open(HERE / "eds_calibration.json"))["holdout"]["per_element"]
print("\n=== leave-one-session-out: RMSE / bias (wt% points) vs the random 20 % hold-out ===")
print(f"{'element':8s} {'random':>13s} | " + " | ".join(f"{'held-out ' + s[5:]:>14s}" for s in sessions))
loso_overall = {}
for s in sessions:
    train = [i for i in all_idx if sess[i] != s]; test = [i for i in all_idx if sess[i] == s]
    sc = score(test, fit(train))
    report["loso"][s] = sc
    loso_overall[s] = sc["_overall"]["rmse"]
for e in els:
    line = f"{e:8s} {hold[e]['rmse']:6.2f} {hold[e]['bias']:+6.2f} | "
    line += " | ".join(f"{report['loso'][s][e]['rmse']:6.2f} {report['loso'][s][e]['bias']:+6.2f}" if e in report["loso"][s] else f"{'-':>14s}" for s in sessions)
    print(line)
print(f"{'overall':8s} {json.load(open(HERE / 'eds_calibration.json'))['holdout']['rmse']:6.2f}        | "
      + " | ".join(f"{loso_overall[s]:6.2f}        " for s in sessions))
# within-session random hold-out for reference (is the LOSO error drift or just that session's scatter?)
print("\n=== within-session random 20 % hold-out (10 folds) for reference ===")
for s in sessions:
    idx = [i for i in all_idx if sess[i] == s]
    cv = eq.cross_validate_k_factors([spx_l[i] for i in idx], [cnt_l[i] for i in idx], drop=drop, method="holdout",
                                     holdout_frac=0.2, n_repeats=10, seed=0)
    report["sessions"][s]["within_holdout_rmse"] = cv.rmse
    print(f"{s}: overall RMSE {cv.rmse:.2f}; " + ", ".join(f"{e} {cv.per_element[e].rmse:.2f}/{cv.per_element[e].bias:+.2f}" for e in els if e in cv.per_element))
json.dump(report, open(HERE / "calibration_loso.json", "w"), indent=1)
print(f"\nwrote calibration_loso.json ({time.time() - t0:.0f} s)")
