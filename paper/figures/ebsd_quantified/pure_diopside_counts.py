# -*- coding: utf-8 -*-
"""Why does interior-only erosion not rescue the diopside name on the 6 um
COUNT maps when it does on the quantified wt% maps?

Takes the EBSD-indexed diopside pixels themselves (no segmentation involved),
eroded by 3 px so grain rims are excluded, and runs the ratio-only mineral ID
on their mean net-count fractions, exactly as the pipeline names a domain.
If even EBSD-pure interior diopside scores amphibole above clinopyroxene in
counts, the failure is intrinsic to the uncalibrated intensities (plus the
matrix contamination floor) and no mask erosion can fix it. The same pixels'
mean wt% from the binning-2 QuantMap export is scored alongside for contrast.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

REPO = Path(__file__).resolve().parents[3]   # <repo>/_Manuscript/figures/ebsd_quantified
sys.path.insert(0, str(REPO))
import eds_cache  # noqa: E402
import mineral_id  # noqa: E402
from viewer_thermo_helpers import _net_count_fractions  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = {1: Path(r"D:/data/EBSD/serpdi_1_EBSD"), 2: Path(r"D:/data/EBSD/serpdi_2_EBSD")}
ERODE_PX = 3
out = {}
for m in (1, 2):
    ds = SRC[m]
    cache_dir = eds_cache.resolve_cache_dir(str(ds))
    man = eds_cache.load_manifest(cache_dir)
    H, W = int(man["H"]), int(man["W"])
    els = list(man["elements"])
    band = eds_cache.roi_from_cache(cache_dir, (0, H, 0, W), els, blend_mode="mean", progress=False)
    ebsd = np.load(ds / "ebsd_phase_map.npy")[:H, :W]
    legend = json.loads((ds / "ebsd_phases.json").read_text(encoding="utf-8"))
    names = {int(k): v["name"] for k, v in legend.items() if int(k) > 0}
    dio = [k for k, v in names.items() if "iopside" in v][0]
    mask = ebsd == dio
    inner = binary_erosion(mask, iterations=ERODE_PX)
    print(f"\nmap {m}: EBSD diopside {mask.sum():,} px, interior ({ERODE_PX} px erosion) {inner.sum():,} px")
    out[m] = {}
    for tag, mk in (("all EBSD diopside", mask), ("interior diopside", inner)):
        frac, n = _net_count_fractions(band, sub_mask=mk)
        comp = {k: 100.0 * float(v) for k, v in frac.items()}
        mm = mineral_id.identify(comp, quantified=False, fe3_fraction=0.0, top_n=4,
                                 coating_elements=frozenset({"C"}))
        top = [(x.name, round(x.confidence_pct, 1)) for x in mm]
        major = {k: round(comp[k], 1) for k in ("Si", "Al", "Mg", "Fe", "Ca", "O") if k in comp}
        print(f"  counts  {tag:20s} n={n:>9,}  {major}  -> {top}")
        out[m][f"counts/{tag}"] = {"n": int(n), "comp": comp, "top": top}
    # same pixels in the quantified wt% maps (bin 2): bin covers EBSD 2i..2i+1
    z = np.load(HERE / f"serpdi_{m}_wt_maps_bin2.npz")
    wt = {k: z[k] for k in z.files}
    h, w = wt["Si"].shape
    # a bin is "interior diopside" when all four EBSD pixels are interior diopside
    inn = inner[:2 * h, :2 * w]
    binmask = inn[0::2, 0::2] & inn[1::2, 0::2] & inn[0::2, 1::2] & inn[1::2, 1::2]
    live = sum(wt.values()) > 0
    sel = binmask & live
    comp = {el: float(wt[el][sel].mean()) for el in wt}
    for q, label in ((False, "ratio"), (True, "structural")):
        mm = mineral_id.identify(comp, quantified=q, fe3_fraction=0.0, top_n=4,
                                 coating_elements=frozenset())
        top = [(x.name, round(x.confidence_pct, 1)) for x in mm]
        major = {k: round(comp[k], 1) for k in ("Si", "Al", "Mg", "Fe", "Ca", "O")}
        print(f"  wt%     interior bins {label:10s} n={int(sel.sum()):>9,}  {major}  -> {top}")
        out[m][f"wt/{label}"] = {"n": int(sel.sum()), "comp": comp, "top": top}
    # count-fraction vs wt%: the Ca/Mg and Ca/Si proxies
    fc, _ = _net_count_fractions(band, sub_mask=inner)
    print(f"  Ca/Mg: counts {fc['Ca'] / fc['Mg']:.2f}  wt% {comp['Ca'] / comp['Mg']:.2f}  "
          f"(diopside mass ratio 1.65, molar 1.0);  Ca/Si: counts {fc['Ca'] / fc['Si']:.2f}  "
          f"wt% {comp['Ca'] / comp['Si']:.2f} (mass 0.71)")
(HERE / "pure_diopside_counts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
