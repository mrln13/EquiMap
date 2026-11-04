# -*- coding: utf-8 -*-
"""Directional correlation lengths at the finest level, all 9192 windows, fine
lag bins, two estimators: (i) the paper's rule, first lag with gamma >= 0.95 of
the sill (sill = 1 for standardised elements averaged), (ii) exponential model
gamma = 1 - exp(-h/r) fitted by least squares on 0-6 mm, range = 3r."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

FIG = Path(__file__).resolve().parent
df = pd.read_csv(FIG / "composition_scale.csv")
els = [c for c in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca", "Ti", "P") if c in df.columns]
px_mm = 24.8 / (df.x1.max() - df.x0.min())
for lvl in (df.level.max(), df.level.max() - 2):
    sub = df[df.level == lvl]
    cx = ((sub.x0 + sub.x1) / 2).to_numpy() * px_mm
    cy = ((sub.y0 + sub.y1) / 2).to_numpy() * px_mm
    Z = np.column_stack([(sub[e] - sub[e].mean()) / sub[e].std(ddof=0) for e in els if sub[e].std(ddof=0) > 0])
    i, j = np.triu_indices(len(cx), 1)
    dx, dy = cx[j] - cx[i], cy[j] - cy[i]
    h = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180
    g = 0.5 * ((Z[i] - Z[j]) ** 2).mean(axis=1)
    edges = np.arange(0, 6.01, 0.25)
    cen = 0.5 * (edges[:-1] + edges[1:])
    side = np.sqrt(((sub.x1 - sub.x0) * (sub.y1 - sub.y0)).mean()) * px_mm
    print(f"\nlevel {lvl}: {len(cx)} windows of {side*1000:.0f} um, {len(h)} pairs, bins 0.25 mm to 6 mm")
    for name, sec in (("omnidirectional", None), ("along x (long side)", 0), ("along y (short side)", 90)):
        keep = np.ones(len(h), bool) if sec is None else (np.minimum(abs(ang - sec), 180 - abs(ang - sec)) <= 22.5)
        gam = np.array([g[keep & (h >= lo) & (h < hi)].mean() if (keep & (h >= lo) & (h < hi)).sum() >= 30 else np.nan
                        for lo, hi in zip(edges[:-1], edges[1:])])
        ok = np.isfinite(gam)
        a95 = cen[ok][np.where(gam[ok] >= 0.95)[0][0]] if (gam[ok] >= 0.95).any() else np.nan
        (r,), _ = curve_fit(lambda hh, r: 1 - np.exp(-hh / r), cen[ok], gam[ok], p0=[1.0], bounds=(0.01, 50))
        print(f"  {name:<22s} 0.95-sill crossing {a95:4.2f} mm; exponential r = {r:4.2f} mm (practical range 3r = {3*r:4.1f} mm); "
              f"gamma at 0.125/0.375/0.625/1.1/2.1 mm: " + " ".join(f"{gam[k]:4.2f}" for k in (0, 1, 2, 4, 8)))
