# -*- coding: utf-8 -*-
"""Which way does the banding run on the worked-example phase map?  Indicator
variograms of the biotite domain along x and along y, plus a downsampled PNG."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parent
S = FIG
z = np.load(FIG / "phase_map_labels.npz")
print("npz keys:", list(z.keys()), {k: (z[k].shape, z[k].dtype) for k in z.keys()})
lab = z[list(z.keys())[0]]
import csv
names = {int(r["cluster_id"]): r["name"] for r in csv.DictReader(open(FIG / "phase_legend.csv"))}
print("labels present:", {int(v): names.get(int(v), "?") for v in np.unique(lab) if v >= 0})
ds = 8
sub = lab[::ds, ::ds]
px_um = 24.8e3 / lab.shape[1]
print(f"map {lab.shape} px, {px_um:.3f} um/px full-res; downsampled x{ds}")
bt = [c for c, n in names.items() if n == "Biotite"][0]
I = (sub == bt).astype(float)
var = I.var()
lags_um = []
gx, gy = [], []
for h in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    lags_um.append(h * ds * px_um)
    gx.append(0.5 * np.mean((I[:, h:] - I[:, :-h]) ** 2) / var)
    gy.append(0.5 * np.mean((I[h:, :] - I[:-h, :]) ** 2) / var)
print("lag (mm)      :", " ".join(f"{l/1000:6.2f}" for l in lags_um))
print("gamma along x :", " ".join(f"{g:6.2f}" for g in gx))
print("gamma along y :", " ".join(f"{g:6.2f}" for g in gy))
print("(biotite indicator, normalised by its variance; slower rise = longer correlation = along the banding)")
fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(np.where(sub >= 0, sub, np.nan), cmap="tab20", interpolation="nearest")
ax.set_title("phase map labels (downsampled); x = long side")
fig.savefig(S / "phase_map_overview.png", dpi=110, bbox_inches="tight")
print("wrote", S / "phase_map_overview.png")
