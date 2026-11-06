# -*- coding: utf-8 -*-
"""Would a tolerance-based criterion return a representative averaging scale on
the worked example?  (a) per-level between-window spread vs the calibration
hold-out RMSE and vs 10 % relative; (b) pt_scale.csv region widths per level;
(c) directional semivariograms (4 sectors) at the finest level."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIG = Path(__file__).resolve().parent
df = pd.read_csv(FIG / "composition_scale.csv")
els = [c for c in df.columns if c in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca", "Ti", "P")]
W_px = df.x1.max() - df.x0.min()
H_px = df.y1.max() - df.y0.min()
sys.path.insert(0, str(FIG))
from _figstyle import MOSAIC_PIXEL_UM
px_mm = MOSAIC_PIXEL_UM / 1000.0
print(f"section {W_px} x {H_px} px -> {px_mm*1000:.3f} um/px; {W_px*px_mm:.1f} x {H_px*px_mm:.1f} mm")
rmse = json.loads((FIG / "mc_calibrated_summary.json").read_text())["rmse"]

print("\n(a) per level: n windows, side (mm), per-element std (wt%) [rel %], flags: '>R' std > calibration RMSE, '>10' rel > 10 %")
hdr = "lvl    n  side_mm " + " ".join(f"{e:>14s}" for e in els)
print(hdr)
for lvl, sub in df.groupby("level"):
    sub = sub[~sub.skipped.astype(str).str.lower().isin(("true","1"))] if "skipped" in sub else sub
    if len(sub) < 2:
        continue
    side = np.sqrt((sub.x1 - sub.x0) * (sub.y1 - sub.y0)).mean() * px_mm
    cells = []
    for e in els:
        sd, mu = sub[e].std(ddof=0), sub[e].mean()
        rel = 100 * sd / mu if mu > 0 else np.nan
        flag = ("R" if sd > rmse.get(e, 9e9) else "-") + ("T" if rel > 10 else "-")
        cells.append(f"{sd:5.2f}[{rel:4.0f}%]{flag}")
    print(f"{lvl:3d} {len(sub):4d} {side:8.3f} " + " ".join(f"{c:>14s}" for c in cells))

print("\n(b) pt_scale.csv per level")
pt = pd.read_csv(FIG / "pt_scale.csv")
cols = [c for c in pt.columns if c.startswith(("level", "window", "T_", "P_", "n_"))]
print(pt[cols].to_string(index=False))

print("\n(c) directional semivariograms at the finest level (elements standardised, averaged)")
sub = df[df.level == df.level.max()]
sub = sub[~sub.skipped.astype(str).str.lower().isin(("true","1"))] if "skipped" in sub else sub
cx = ((sub.x0 + sub.x1) / 2).to_numpy() * px_mm
cy = ((sub.y0 + sub.y1) / 2).to_numpy() * px_mm
Z = np.column_stack([(sub[e] - sub[e].mean()) / sub[e].std(ddof=0) for e in els if sub[e].std(ddof=0) > 0])
i, j = np.triu_indices(len(cx), 1)
dx, dy = cx[j] - cx[i], cy[j] - cy[i]
h = np.hypot(dx, dy)
ang = np.degrees(np.arctan2(dy, dx)) % 180
g = 0.5 * ((Z[i] - Z[j]) ** 2).mean(axis=1)
bins = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 13])
print(f"n windows {len(cx)}, pairs {len(h)}; lag bins (mm): {bins.tolist()}")
for name, a0 in (("x (0 deg, along the long side)", 0), ("45 deg", 45), ("y (90 deg, along the short side)", 90), ("135 deg", 135)):
    d = np.minimum(abs(ang - a0), 180 - abs(ang - a0))
    sel = d <= 22.5
    row = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = sel & (h >= lo) & (h < hi)
        row.append(f"{g[m].mean():4.2f}" if m.sum() >= 15 else "  . ")
    print(f"{name:<34s} " + " ".join(row))
m_all = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (h >= lo) & (h < hi)
    m_all.append(f"{g[m].mean():4.2f}" if m.sum() >= 15 else "  . ")
print(f"{'omnidirectional':<34s} " + " ".join(m_all))
