"""Per-phase stoichiometric check of the SPX-anchored calibration (Section 5.1).

The tile hold-out only tests the k-factor model at the tile-bulk scale. This
script asks the question the segmentation, identification and sub-tile
analyses actually depend on: does the calibrated mean composition of each
clean domain make a sensible mineral formula?

Inputs (this directory + the July session):
  phase_map_labels.npz          the six-domain label map of Fig. 4 (seg_ds = 5)
  phase_legend.csv              cluster id -> name
  viewer_session_22_07_26.npz   band maps on the same 1/5 grid
  eds_calibration.json          the mosaic calibration (calibrate_mosaic.py)
Per domain and erosion depth (0 / 3 / 6 px at 2.5 um/px): raw net-count sums
-> calibrated cation wt% (eds_quantify.apply_calibration_fractions, the same
call the viewer's classification dialog makes; nets are the signed line-aware
nets of eds_background) -> structural formula on the
mineral's oxygen basis (mineral_id.identify, quantified=True, Fe3+/Fe = 0.1).
Writes phase_stoichiometry.csv and prints the erosion-3 table.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

REPO = Path(__file__).resolve().parents[2]
DATA = Path(r"D:/Actief/C23S1A_Quant")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import eds_quantify as eq          # noqa: E402
import mineral_id as mi            # noqa: E402

EROSIONS = (0, 3, 6)
FE3 = 0.1
TARGET = {"Quartz": "Quartz", "Plagioclase": "Plagioclase", "Biotite": "Biotite",
          "Muscovite": "Muscovite", "Epidote": "Epidote"}

labels = np.load(HERE / "phase_map_labels.npz")["labels"]
legend = {int(r["cluster_id"]): r["name"].strip()
          for r in csv.DictReader(open(HERE / "phase_legend.csv"))}
# k-factors from the headless mosaic calibration (calibrate_mosaic.py), which
# must have been derived on the SAME background scheme as the nets below.
import eds_background as bg        # noqa: E402
calj = json.load(open(HERE / "eds_calibration.json"))
cal = eq.Calibration.from_dict(calj["calibration"])
if cal.background != bg.default_scheme():
    raise SystemExit(f"calibration background {cal.background!r} != scheme {bg.default_scheme()!r}")
k = cal.k()
print(f"k-factors ({cal.background}, {cal.n_tiles} tiles):",
      {e: f"{v:.3g}" for e, v in k.items()})
r2 = {e: cal.k_factors[e].r2 for e in k}
print("in-sample r2:", {e: (None if v is None else round(v, 2)) for e, v in r2.items()})

z = np.load(DATA / "viewer_session_22_07_26.npz")
elements = sorted({key.split("__")[0] for key in z.keys() if "__" in key})
print("elements in the band maps:", elements)

# masks per (cid, erosion); eroded-away domains fall back to the full mask
masks = {}
for cid, name in legend.items():
    full = labels == cid
    for e in EROSIONS:
        m = full if e == 0 else binary_erosion(full, iterations=e)
        if e > 0 and not m.any():
            m = full
        masks[(cid, e)] = m
npx = {key: int(m.sum()) for key, m in masks.items()}

# Signed line-aware nets, one element at a time (the band getter pulls the
# anchor windows of other elements straight from the session NPZ).
from eds_background import background_plan, net_count_map   # noqa: E402
plan = background_plan(elements)
sums = {key: {} for key in masks}
for el in elements:
    net = net_count_map(el, lambda e, b: z[f"{e}__{b}"] if f"{e}__{b}" in z else None, plan=plan)
    for key, m in masks.items():
        sums[key][el] = float(net[m].sum())
    del net
    print(f"  summed {el}", flush=True)

rows = []
for (cid, e), s in sorted(sums.items()):
    name = legend[cid]
    tot = sum(v for v in s.values() if v > 0)
    raw_pct = {el: 100.0 * max(v, 0.0) / tot for el, v in s.items()} if tot > 0 else {}
    calw = eq.apply_calibration_fractions(s, k)
    row = {"cluster_id": cid, "name": name, "erosion_px": e, "n_px": npx[(cid, e)]}
    row.update({f"raw_{el}": round(raw_pct.get(el, 0.0), 3) for el in elements})
    row.update({f"cal_{el}": round(calw.get(el, 0.0), 3) for el in sorted(k)})
    target = TARGET.get(name)
    if target:
        matches = mi.identify(calw, quantified=True, fe3_fraction=FE3, top_n=40)
        m = next((x for x in matches if x.name == target), None)
        if m is not None:
            row["score"] = round(m.score, 3)
            for el, v in (m.apfu or {}).items():
                row[f"apfu_{el}"] = round(v, 3)
            for site, v in (m.site_closure or {}).items():
                row[f"closure_{site}"] = round(v, 3)
            for idx, v in (m.indices or {}).items():
                row[f"idx_{idx}"] = round(v, 3)
    rows.append(row)

cols = []
for r in rows:
    for c in r:
        if c not in cols:
            cols.append(c)
with open(HERE / "phase_stoichiometry.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
print("wrote phase_stoichiometry.csv")

for e in EROSIONS:
    print(f"\n=== erosion {e} px ({e * 2.5:.1f} um) ===")
    for r in rows:
        if r["erosion_px"] != e or r["name"] not in TARGET:
            continue
        ap = {c[5:]: r[c] for c in r if c.startswith("apfu_") and r[c] > 0.02}
        cl = {c[8:]: r[c] for c in r if c.startswith("closure_")}
        ix = {c[4:]: r[c] for c in r if c.startswith("idx_")}
        calw = {c[4:]: r[c] for c in r if c.startswith("cal_") and r[c] > 0.3}
        print(f"{r['name']:<12s} n={r['n_px']:>9d}  cal wt% " + " ".join(f"{el} {v:.1f}" for el, v in sorted(calw.items(), key=lambda kv: -kv[1])))
        print(f"{'':12s} apfu " + " ".join(f"{el} {v:.2f}" for el, v in sorted(ap.items(), key=lambda kv: -kv[1]))
              + " | closure " + " ".join(f"{s} {v:.2f}" for s, v in cl.items())
              + (" | " + " ".join(f"{i} {v:.2f}" for i, v in ix.items()) if ix else ""))
