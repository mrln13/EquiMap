# -*- coding: utf-8 -*-
"""Share of the muscovite mode that sits in the millimetre-scale nodules,
measured on the staged phase map (co-author comment, 2026-10-28).

The partial peak-assemblage reconstruction (fig_reconstruction.py panel b,
Table 1 row 3) converts the NODULE-hosted muscovite to its K-feldspar
precursor and keeps the foliation-defining matrix muscovite. Until now the
nodule share of the muscovite mode was an illustrative 0.5. Here it is
measured on ``phase_map_labels.npz`` (2.5 um/px) in two independent ways:

  size   : morphological opening of the muscovite mask with a disc of radius
           r removes the thin foliation-parallel flakes; the surviving blobs
           with an equivalent diameter >= D are the nodules; the nodule
           muscovite is every muscovite pixel within r of such a blob (the
           opening shaves the irregular rims, the dilation restores them
           without reaching the flakes further out).
  cores  : muscovite regions (closed over 2 px to bridge cracks) that touch
           a sillimanite (Al2SiO5 core) pixel, the nodules' defining feature
           on the map; size-filtered the same way.

The muscovite MODE also receives muscovite from the two unmixed matrix
domains (segmentation_params.json "unmix"), none of which is nodule
muscovite, so the nodule share of the mode is
    nodule_px / (muscovite_domain_px + sum(matrix_px * f_ms)).
A sweep over r and D gives the sensitivity; the adopted value is the
size-based share at r = 12.5 um (flakes thinner than 25 um removed) and
D = 0.5 mm (the smallest nodules Rinia describes are about half a
millimetre): 42 nodules of 0.5 to 1.9 mm holding 0.35 of the muscovite mode.
Result of the 2026-10-28 sweep: 0.25 (r = 25 um) to 0.45 (D = 0.3 mm) under
the size definition, 0.43 to 0.47 for every muscovite region attached to a
core, 0.19 to 0.33 for the muscovite within 0.5 to 1.0 mm of a large core;
every definition puts the share below the illustrative half used before,
and the partial reconstruction's accepted region is the same for any share
up to 0.45 (fig_reconstruction.py re-scored over f).

Writes ``nodule_fraction.json`` (read by fig_reconstruction.py for the
adopted share) and a diagnostic ``nodule_fraction.png`` beside this script.

Usage:  python nodule_fraction.py [--pixel-um 2.5] [--r-um 12.5] [--d-mm 0.5] [--sweep]
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

ap = argparse.ArgumentParser()
ap.add_argument("--pixel-um", type=float, default=2.7375,
                help="phase-map pitch (measured 0.5475 um mosaic pixel x seg_ds 5; was a nominal 2.5 "
                     "until 2026-11-06, which gave 0.35 instead of 0.37)")
ap.add_argument("--r-um", type=float, default=12.5, help="opening radius of the adopted estimate")
ap.add_argument("--d-mm", type=float, default=0.5, help="minimum nodule equivalent diameter of the adopted estimate")
ap.add_argument("--sweep", action="store_true", help="also sweep r and D")
ap.add_argument("--png", default=str(HERE / "nodule_fraction.png"))
ap.add_argument("--out", default=str(HERE / "nodule_fraction.json"))
args = ap.parse_args()

labels = np.load(HERE / "phase_map_labels.npz")["labels"]
names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(HERE / "phase_legend.csv"))}
by_name = {v: k for k, v in names.items()}
MS, SIL = by_name["Muscovite"], by_name["Sillimanite"]
seg = json.loads((HERE / "segmentation_params.json").read_text())
px = {cid: int((labels == cid).sum()) for cid in names}

# muscovite pixel-equivalents that the unmixed matrix domains add to the mode
ms_from_matrix = 0.0
for src, d in (seg.get("unmix") or {}).items():
    fr = {int(e): float(v) for e, v in (d.get("fractions") or {}).items() if float(v) > 0}
    tot_f = sum(fr.values())
    if MS in fr and tot_f > 0:
        ms_from_matrix += px[int(src)] * fr[MS] / tot_f
        print(f"unmixed {names[int(src)]:20s}: {px[int(src)]:>10d} px, muscovite share {fr[MS] / tot_f:.3f}")
ms_px = px[MS]
ms_total = ms_px + ms_from_matrix
print(f"muscovite domain {ms_px} px; from unmixed matrix {ms_from_matrix:.0f} px-eq; mode total {ms_total:.0f} px-eq "
      f"(domain = {ms_px / ms_total:.3f} of the mode)")

ms = labels == MS
sil = labels == SIL
um = args.pixel_um


# Disc erosions / dilations through the Euclidean distance transform (a
# structuring element of radius 10-20 px on a 53-Mpx map is far too slow
# for scipy's direct binary morphology).
def erode(a, r_px):
    return ndi.distance_transform_edt(a) > r_px


def dilate(a, r_px):
    return ndi.distance_transform_edt(~a) <= r_px


_opened = {}


def size_based(r_um, d_mm):
    r_px = r_um / um
    if r_px not in _opened:
        _opened[r_px] = dilate(erode(ms, r_px), r_px)
    lab, n = ndi.label(_opened[r_px])
    if n == 0:
        return np.zeros_like(ms), 0, np.array([]), np.zeros_like(ms)
    areas = np.bincount(lab.ravel())[1:].astype(float)
    eqd_mm = 2 * np.sqrt(areas / np.pi) * um / 1000.0
    keep = np.where(eqd_mm >= d_mm)[0] + 1
    seeds = np.isin(lab, keep)
    region = ndi.binary_fill_holes(dilate(seeds, r_px))
    nod = ms & region
    return nod, int(keep.size), eqd_mm[keep - 1], region


_closed = None
_sil_near = None


def core_based(d_mm, close_px=2):
    global _closed, _sil_near
    if _closed is None:
        _closed = erode(dilate(ms, close_px), close_px) | ms
        _sil_near = dilate(sil, 2)
    lab, n = ndi.label(_closed)
    touch = np.unique(lab[_sil_near & (lab > 0)])
    areas = np.bincount(lab.ravel()).astype(float)
    eqd_mm = 2 * np.sqrt(areas / np.pi) * um / 1000.0
    keep = np.array([c for c in touch if c > 0 and eqd_mm[c] >= d_mm], dtype=int)
    nod = ms & np.isin(lab, keep)
    return nod, int(keep.size), eqd_mm[keep] if keep.size else np.array([])


# Per-pixel muscovite content: 1 in the muscovite domain, the unmixed
# muscovite share in the two matrix domains (their rims wrap the nodules:
# half the pixels bordering a sillimanite core are "mica-rich matrix").
ms_content = ms.astype(np.float32)
for src, d in (seg.get("unmix") or {}).items():
    fr = {int(e): float(v) for e, v in (d.get("fractions") or {}).items() if float(v) > 0}
    tot_f = sum(fr.values())
    if MS in fr and tot_f > 0:
        ms_content[labels == int(src)] = fr[MS] / tot_f
# the larger sillimanite cores (>= 0.2 mm equivalent diameter): the nodules'
# defining feature, used to check that a nodule definition captures them
_sl, _n = ndi.label(sil)
_ar = np.bincount(_sl.ravel()); _ar[0] = 0
_big_ids = np.where(2 * np.sqrt(_ar / np.pi) * um / 1000.0 >= 0.2)[0]
_big = np.isin(_sl, _big_ids)
_n_big = int(_big_ids.size)
# restrict the labelled core map to the large cores (the capture loop runs per core)
_sl = np.where(_big, _sl, 0)
_slices = ndi.find_objects(_sl)
print(f"sillimanite cores >= 0.2 mm: {_n_big} of {int((_ar > 0).sum())} regions, {_big.sum()} of {sil.sum()} px")


def report(tag, nod, n, eqd, region=None):
    """nod = nodule pixels of the clean muscovite domain; region = the filled
    nodule outline (clean muscovite + matrix rims + cores), when given."""
    f_dom = nod.sum() / ms_px
    f_mode = nod.sum() / ms_total
    sizes = (f"d50 {np.median(eqd):.2f} mm, max {eqd.max():.2f} mm" if eqd.size else "-")
    extra = {}
    if region is not None:
        f_region = float(ms_content[region].sum() / ms_total)
        # a large core counts as captured when at least half of its 3-px
        # boundary ring lies in the nodule region (the ring, not the core's
        # interior, is what a muscovite nodule can contain)
        cap_px, cap_n = 0, 0
        for c, sl in enumerate(_slices, start=1):
            if sl is None:
                continue
            ys, xs = sl
            ys = slice(max(0, ys.start - 4), ys.stop + 4); xs = slice(max(0, xs.start - 4), xs.stop + 4)
            core = _sl[ys, xs] == c
            ring = ndi.binary_dilation(core, iterations=3) & ~core
            if region[ys, xs][ring].mean() >= 0.5:
                cap_px += int(core.sum()); cap_n += 1
        cores_in = cap_px / max(1, _big.sum())
        extra = {"frac_of_mode_with_rims": f_region, "large_cores_captured_px": cores_in, "large_cores_captured_n": cap_n}
        tail = f"; with matrix rims {f_region:.3f} of the mode; large cores captured {cap_n}/{_n_big} ({100 * cores_in:.0f} % of their px)"
    else:
        tail = ""
    print(f"{tag:34s} nodules {n:4d}  nodule Ms {nod.sum():>9d} px = {f_dom:.3f} of the domain, "
          f"{f_mode:.3f} of the mode; {sizes}{tail}")
    return {"n_nodules": n, "nodule_px": int(nod.sum()), "frac_of_domain": float(f_dom), "frac_of_mode": float(f_mode),
            "d50_mm": (float(np.median(eqd)) if eqd.size else None), "dmax_mm": (float(eqd.max()) if eqd.size else None), **extra}


out = {"pixel_um": um, "muscovite_domain_px": ms_px, "muscovite_from_unmixed_matrix_px": ms_from_matrix,
       "muscovite_mode_px_equiv": ms_total, "adopted": {"r_um": args.r_um, "d_mm": args.d_mm}, "size": {}, "cores": {}}
nod_a, n_a, eqd_a, reg_a = size_based(args.r_um, args.d_mm)
out["size"][f"r{args.r_um:g}_d{args.d_mm:g}"] = report(f"size r={args.r_um:g} um D={args.d_mm:g} mm", nod_a, n_a, eqd_a, reg_a)
nod_c, n_c, eqd_c = core_based(args.d_mm)
out["cores"][f"d{args.d_mm:g}"] = report(f"cores D={args.d_mm:g} mm", nod_c, n_c, eqd_c)
both = nod_a & nod_c
print(f"agreement: {both.sum() / max(1, (nod_a | nod_c).sum()):.2f} Jaccard; size-only {(nod_a & ~nod_c).sum()} px, cores-only {(nod_c & ~nod_a).sum()} px")
out["union"] = report("union of the two", nod_a | nod_c, 0, np.array([]))
out["adopted"]["frac_of_mode"] = out["size"][f"r{args.r_um:g}_d{args.d_mm:g}"]["frac_of_mode"]
out["adopted"]["frac_of_mode_with_rims"] = out["size"][f"r{args.r_um:g}_d{args.d_mm:g}"]["frac_of_mode_with_rims"]

if args.sweep:
    for r_um in (12.5, 25.0, 50.0):
        for d_mm in (0.3, 0.5, 0.75, 1.0):
            nod, n, eqd, reg = size_based(r_um, d_mm)
            out["size"][f"r{r_um:g}_d{d_mm:g}"] = report(f"size r={r_um:g} um D={d_mm:g} mm", nod, n, eqd, reg)
    for d_mm in (0.0, 0.3, 0.5, 0.75, 1.0):
        nod, n, eqd = core_based(d_mm)
        out["cores"][f"d{d_mm:g}"] = report(f"cores D={d_mm:g} mm", nod, n, eqd)

# diagnostic picture (5x decimated): muscovite grey, nodule muscovite red, sillimanite blue
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = 5
rgb = np.ones(labels[::d, ::d].shape + (3,), float) * 0.96
rgb[ms[::d, ::d]] = (0.55, 0.55, 0.55)
rgb[nod_a[::d, ::d]] = (0.85, 0.15, 0.1)
rgb[dilate(sil, 2)[::d, ::d]] = (0.1, 0.3, 0.9)
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.imshow(rgb, interpolation="nearest")
ax.set_axis_off()
ax.set_title(f"muscovite (grey), nodule muscovite r={args.r_um:g} um, D={args.d_mm:g} mm (red), sillimanite cores (blue): "
             f"{out['adopted']['frac_of_mode']:.2f} of the Ms mode in {n_a} nodules", fontsize=9)
fig.savefig(args.png, dpi=150, bbox_inches="tight")
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"wrote {args.out} and {args.png}")
