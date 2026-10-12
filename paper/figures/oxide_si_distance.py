# -*- coding: utf-8 -*-
"""Apparent Si inside EBSD-indexed oxide grains against distance from the grain
edge (co-author comment 20), on the two Oxford h5oina maps of section 5.2.

Fluorescence or lateral mixing fades with distance from the edge; a continuum
(background) artefact of the oxide's higher mean Z does not. A second
discriminant: windows of elements absent everywhere (P, S, Cl, K, Y) hold
background only, so their apparent share inside the oxides is the background
level a Si window would also carry.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
DATASETS = {1: Path(r"D:\data\EBSD\serpdi_1_EBSD"), 2: Path(r"D:\data\EBSD\serpdi_2_EBSD")}
OXIDES = {3: "Ilmenite", 4: "Magnetite"}
DROP = {"O", "C"}
ABSENT = ["P", "S", "Cl", "K", "Y"]          # background-only windows
MAXD = 8                                      # px (6 um each)

results = {}
for n, d in DATASETS.items():
    print("=" * 78); print(d)
    man = json.load(open(d / "eds_cache" / "manifest.json"))
    els = list(man["elements"])
    ebsd = np.load(d / "ebsd_phase_map.npy")
    H, W = ebsd.shape
    print(f" grid {H}x{W}, elements {els}")
    oxide = np.isin(ebsd, list(OXIDES))
    print(f" oxide pixels: {oxide.sum()} ({100*oxide.mean():.2f} %); "
          + "; ".join(f"{OXIDES[o]} {(ebsd==o).sum()}" for o in OXIDES))
    # inward distance (to the nearest non-oxide pixel) and outward distance
    d_in = ndimage.distance_transform_edt(oxide)
    d_out = ndimage.distance_transform_edt(~oxide)
    sel_in = oxide
    sel_out = (~oxide) & (d_out <= MAXD)
    idx_in = np.flatnonzero(sel_in.ravel()); idx_out = np.flatnonzero(sel_out.ravel())
    bin_in = np.minimum(np.floor(d_in.ravel()[idx_in]).astype(int), MAXD)   # 1..MAXD
    bin_out = np.minimum(np.floor(d_out.ravel()[idx_out]).astype(int), MAXD)
    ph_in = ebsd.ravel()[idx_in]
    # read every element once; keep only the selected pixels
    cnt_in = {}; cnt_out = {}
    for el in els:
        a = np.load(d / "eds_cache" / f"{el}__main.npy", mmap_mode="r").ravel()
        cnt_in[el] = np.asarray(a[idx_in], dtype=np.float64)
        cnt_out[el] = np.asarray(a[idx_out], dtype=np.float64)
    cat = [e for e in els if e not in DROP]
    cat_noW = [e for e in cat if e != "W"]
    tot_all_in = sum(cnt_in[e] for e in els)
    tot_cat_in = sum(cnt_in[e] for e in cat)
    tot_cat_noW_in = sum(cnt_in[e] for e in cat_noW)
    tot_cat_out = sum(cnt_out[e] for e in cat)
    # also the total counts of the two silicate phases for the mean-Z comparison
    tot_phase = {}
    for ordv, name in {1: "Diopside", 2: "Chlorite"}.items():
        m = (ebsd == ordv)
        ii = np.flatnonzero(m.ravel())
        rng = np.random.default_rng(0); ii = rng.choice(ii, size=min(200000, ii.size), replace=False)
        t = np.zeros(ii.size)
        for el in els:
            a = np.load(d / "eds_cache" / f"{el}__main.npy", mmap_mode="r").ravel(); t += a[ii]
        tot_phase[name] = float(t.mean())
    tot_phase["oxide interior (d>=3)"] = float(tot_all_in[bin_in >= 3].mean()) if (bin_in >= 3).any() else np.nan
    tot_phase["oxide edge (d=1)"] = float(tot_all_in[bin_in == 1].mean())
    print(" mean total window counts per pixel:", {k: round(v) for k, v in tot_phase.items()})

    def share(el, which="in", wsel=None, denom=None):
        c = cnt_in if which == "in" else cnt_out
        den = denom if denom is not None else (tot_cat_in if which == "in" else tot_cat_out)
        b = bin_in if which == "in" else bin_out
        rows = []
        for k in range(1, MAXD + 1):
            m = (b == k) if wsel is None else ((b == k) & wsel)
            rows.append((k, int(m.sum()), 100 * c[el][m].sum() / den[m].sum() if m.any() else np.nan))
        return rows

    res = {"n_px": {}, "inward": {}, "outward": {}, "absent": {}, "totals": tot_phase}
    print("\n inward: apparent share (% of the cation windows) by distance from the grain edge, px of 6 um")
    hdr = " el  phase      " + " ".join(f"d={k:<7d}" for k in range(1, MAXD + 1))
    print(hdr)
    for el in ["Si", "Mg", "Al", "Fe", "Ti"] + (["W"] if "W" in els else []):
        for o, name in OXIDES.items():
            rows = share(el, "in", wsel=(ph_in == o))
            res["inward"][f"{el}|{name}"] = rows
            print(f" {el:<3s} {name:<10s} " + " ".join(f"{v:5.1f} ({c:>5d})"[:9].ljust(9) for _, c, v in rows))
    if "W" in els:
        print(" (map 2, W window excluded from the denominator)")
        for o, name in OXIDES.items():
            rows = share("Si", "in", wsel=(ph_in == o), denom=tot_cat_noW_in)
            res["inward"][f"Si_noW|{name}"] = rows
            print(f" Si* {name:<10s} " + " ".join(f"{v:5.1f}".ljust(9) for _, c, v in rows))
    print("\n background-only windows inside the oxides (share of the cation windows, interior d>=3 vs edge d=1):")
    for el in ABSENT:
        if el not in els:
            continue
        inner = (bin_in >= 3); edge = (bin_in == 1)
        vi = 100 * cnt_in[el][inner].sum() / tot_cat_in[inner].sum(); ve = 100 * cnt_in[el][edge].sum() / tot_cat_in[edge].sum()
        res["absent"][el] = (float(ve), float(vi))
        print(f"  {el:<2s} edge {ve:5.2f}  interior {vi:5.2f}")
    print("\n outward: Fe and Ti share in the non-oxide pixels by distance from the oxide (mixing/fluorescence of the oxide into the silicate)")
    for el in ["Fe", "Ti", "Si"]:
        rows = share(el, "out")
        res["outward"][el] = rows
        print(f" {el:<3s}            " + " ".join(f"{v:5.1f}".ljust(9) for _, c, v in rows))
    results[n] = res

json.dump(results, open(OUT / "oxide_si_distance.json", "w"), indent=1, default=float)

# ---- plot ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, n in zip(axes, results):
    r = results[n]
    for key, st in (("Si|Ilmenite", "o-"), ("Si|Magnetite", "s-"), ("Mg|Ilmenite", "o--"), ("Mg|Magnetite", "s--")):
        rows = r["inward"][key]
        ax.plot([k * 6 for k, _, _ in rows], [v for _, _, v in rows], st, label=key.replace("|", " in "))
    if "Si_noW|Ilmenite" in r["inward"]:
        rows = r["inward"]["Si_noW|Ilmenite"]
        ax.plot([k * 6 for k, _, _ in rows], [v for _, _, v in rows], "o:", color="gray", label="Si in Ilmenite, W window dropped")
    for el, (ve, vi) in r["absent"].items():
        ax.axhline(vi, color="lightgray", lw=0.8)
    ax.set_title(f"map {n}: apparent share inside EBSD oxide grains"); ax.set_xlabel("distance from the grain edge, um (pixel 6 um; last bin = beyond)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("% of the cation windows (O, C dropped)")
fig.text(0.5, -0.02, "grey lines: share of the background-only windows (P, S, Cl, K, Y) in the oxide interior", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "oxide_si_distance.png", dpi=110, bbox_inches="tight")
print("wrote", OUT / "oxide_si_distance.png")
