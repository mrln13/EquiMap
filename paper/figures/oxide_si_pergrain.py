# -*- coding: utf-8 -*-
"""Two discriminants for the flat interior Si plateau inside the EBSD oxides:

1. Per-grain spread of the interior Si share. Sub-pixel intergrowth or thin
   grains over silicate vary from grain to grain (many grains near 0 %, some
   high); a systematic effect (remote excitation by backscattered electrons at
   the 70 degree EBSD tilt, detector/window effects) is uniform across grains.
2. Map 1 re-quantified by AZtec (binning 4, background subtracted, ZAF): the Si
   wt% in bins lying wholly inside oxide grains. Background subtraction cannot
   remove a characteristic Si signal.
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
MIN_INTERIOR = 6      # grains with at least this many interior (d>=2) pixels

for n, d in DATASETS.items():
    print("=" * 78); print(d)
    man = json.load(open(d / "eds_cache" / "manifest.json")); els = list(man["elements"])
    ebsd = np.load(d / "ebsd_phase_map.npy")
    oxide = np.isin(ebsd, list(OXIDES))
    d_in = ndimage.distance_transform_edt(oxide)
    interior = oxide & (d_in >= 2)
    lab, nlab = ndimage.label(oxide)
    idx = np.flatnonzero(interior.ravel())
    lab_i = lab.ravel()[idx]; ph_i = ebsd.ravel()[idx]
    cat = [e for e in els if e not in DROP]
    si = np.asarray(np.load(d / "eds_cache" / "Si__main.npy", mmap_mode="r").ravel()[idx], float)
    mg = np.asarray(np.load(d / "eds_cache" / "Mg__main.npy", mmap_mode="r").ravel()[idx], float)
    tot = np.zeros(idx.size)
    for el in cat:
        tot += np.load(d / "eds_cache" / f"{el}__main.npy", mmap_mode="r").ravel()[idx]
    npx = np.bincount(lab_i, minlength=nlab + 1)
    s_si = np.bincount(lab_i, weights=si, minlength=nlab + 1)
    s_mg = np.bincount(lab_i, weights=mg, minlength=nlab + 1)
    s_tot = np.bincount(lab_i, weights=tot, minlength=nlab + 1)
    # phase of each grain = majority ordinal of its interior pixels
    ph_grain = np.zeros(nlab + 1, int)
    for o in OXIDES:
        c = np.bincount(lab_i[ph_i == o], minlength=nlab + 1); ph_grain[c > npx / 2] = o
    keep = npx >= MIN_INTERIOR
    share = 100 * s_si[keep] / s_tot[keep]; mgs = 100 * s_mg[keep] / s_tot[keep]; sz = npx[keep]; ph = ph_grain[keep]
    print(f" grains with >= {MIN_INTERIOR} interior px: {keep.sum()} of {nlab} oxide components; interior px used {sz.sum()}")
    for o, name in OXIDES.items():
        m = ph == o
        if not m.any():
            continue
        q = np.percentile(share[m], [5, 25, 50, 75, 95])
        print(f"  {name:<10s} n={m.sum():4d}: interior Si share percentiles 5/25/50/75/95 = "
              + "/".join(f"{v:.1f}" for v in q) + f";  grains below 3 %: {(share[m] < 3).sum()}, above 20 %: {(share[m] > 20).sum()}")
        # pixel-weighted vs grain-median, and size dependence
        big = m & (sz >= np.percentile(sz[m], 75)); small = m & (sz <= np.percentile(sz[m], 25))
        print(f"             pixel-weighted mean {100*s_si[keep][m].sum()/s_tot[keep][m].sum():.1f}; "
              f"largest-quartile grains median {np.median(share[big]):.1f}, smallest-quartile {np.median(share[small]):.1f}; "
              f"Si/Mg median {np.median(share[m]/np.maximum(mgs[m], 1e-9)):.2f}")
    np.savez(OUT / f"oxide_grains_map{n}.npz", share=share, mg=mgs, size=sz, phase=ph)

    if n == 1:
        z = np.load(r"D:\Actief\THERMOD\_scratch\background_fix\serpdi_1_wt_maps.npz")
        wt = {k: z[k] for k in z.files}
        bh, bw = wt["Si"].shape; B = 4
        print(f"\n AZtec-quantified map 1 ({bh}x{bw} bins of {B}x{B} px): Si wt% in bins wholly inside the oxides")
        H, W = ebsd.shape
        Hc, Wc = bh * B, bw * B
        ox_b = oxide[:Hc, :Wc].reshape(bh, B, bw, B).all(axis=(1, 3))
        int_b = interior[:Hc, :Wc].reshape(bh, B, bw, B).all(axis=(1, 3))
        ph_b = np.where(ebsd[:Hc, :Wc].reshape(bh, B, bw, B)[:, 0, :, 0] == 3, 3, 4)
        live = np.isfinite(wt["Si"]) & (sum(wt[k] for k in wt) > 50)
        for name, m in (("any pure-oxide bin", ox_b & live), ("interior bins (all 16 px at d>=2)", int_b & live)):
            print(f"  {name}: n={m.sum()}")
            for o, pname in OXIDES.items():
                mm = m & (ph_b == o)
                if mm.sum() == 0:
                    continue
                print(f"    {pname:<10s} n={mm.sum():5d}: Si {np.nanmean(wt['Si'][mm]):5.2f} wt% (median {np.nanmedian(wt['Si'][mm]):5.2f}), "
                      f"Mg {np.nanmean(wt['Mg'][mm]):5.2f}, Al {np.nanmean(wt['Al'][mm]):5.2f}, Fe {np.nanmean(wt['Fe'][mm]):5.1f}, Ti {np.nanmean(wt['Ti'][mm]):5.1f}, O {np.nanmean(wt['O'][mm]):5.1f}, C {np.nanmean(wt['C'][mm]):5.1f}")
        # the same for silicate reference bins
        for ordv, pname in ((1, "Diopside"), (2, "Chlorite")):
            mb = (ebsd[:Hc, :Wc] == ordv).reshape(bh, B, bw, B).all(axis=(1, 3)) & live
            print(f"    {pname:<10s} pure bins n={mb.sum():6d}: Si {np.nanmean(wt['Si'][mb]):5.2f} wt%, Mg {np.nanmean(wt['Mg'][mb]):5.2f}, Fe {np.nanmean(wt['Fe'][mb]):5.1f}, P {np.nanmean(wt['P'][mb]):4.2f}")
