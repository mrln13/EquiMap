"""Sizes of the phase domains on the staged phase map (Section 4.4, Fig 3c/d).

For every domain of ``phase_map_labels.npz`` the connected regions are
measured on the segmentation grid and summarised by the AREA-WEIGHTED median
(and 90th percentile) of their equivalent diameters, ``2 * sqrt(area / pi)``:
the size of the region a randomly chosen pixel of that phase sits in. These
are domains, not grains (touching grains of one phase merge), which is the
scale that decides whether a window still contains a mix of the phases.

The range over the phases scored as modal proportions in the P-T inversion
(``fig_pt.NAME_TO_KEY`` minus the presence-only keys) is the shaded band of
``fig_scale.py`` panels (c) and (d): the per-window overlap is lost as windows
shrink through that range.

Writes ``phase_domain_sizes.csv`` beside this script.

Usage:  python phase_domain_sizes.py [--pixel-um 0.5] [--data-dir .]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_pt                                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(HERE))
ap.add_argument("--pixel-um", type=float, default=0.5, help="mosaic pixel size; the map pitch is seg_ds times this")
ap.add_argument("--out", default=str(HERE / "phase_domain_sizes.csv"))
args = ap.parse_args()
DATA = Path(args.data_dir)

labels = np.load(DATA / "phase_map_labels.npz")["labels"]
names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(DATA / "phase_legend.csv"))}
seg = json.loads((DATA / "segmentation_params.json").read_text())
pitch = float(seg.get("seg_ds", 1)) * args.pixel_um
scored = {n for n, k in fig_pt.NAME_TO_KEY.items() if k not in fig_pt.PRESENCE}
total = int((labels >= 0).sum())

rows = []
for cid, name in sorted(names.items()):
    mask = labels == cid
    if not mask.any():
        continue
    cc, n = ndi.label(mask)
    area = np.bincount(cc.ravel())[1:].astype(float)
    diam = 2.0 * np.sqrt(area / np.pi) * pitch
    order = np.argsort(diam)
    cum = np.cumsum(area[order]) / area.sum()
    d50 = float(diam[order][np.searchsorted(cum, 0.5)])
    d90 = float(diam[order][np.searchsorted(cum, 0.9)])
    rows.append({"cluster_id": cid, "name": name, "areal_pct": round(100.0 * mask.sum() / total, 2),
                 "n_regions": int(n), "d50_um": round(d50, 1), "d90_um": round(d90, 1),
                 "scored": int(name in scored)})
    print(f"{name:20s} {rows[-1]['areal_pct']:5.1f} %  regions {n:6d}  area-weighted median diameter "
          f"{d50:6.0f} um, 90th percentile {d90:6.0f} um{'   [scored]' if name in scored else ''}")

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
sc = [r["d50_um"] for r in rows if r["scored"]]
print(f"map pitch {pitch} um; scored phases span {min(sc):.0f} to {max(sc):.0f} um; wrote {args.out}")
