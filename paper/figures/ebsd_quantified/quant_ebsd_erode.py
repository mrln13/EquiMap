# -*- coding: utf-8 -*-
"""Follow-up to quant_ebsd_run.py: interior-only domain compositions.

The diopside-dominated domains (published count-based and GMM-on-wt%) hold a
quarter of unindexed chlorite matrix plus boundary pixels, and their MEAN
wt% is a diopside-chlorite mixture that the library reads as amphibole. The
toolkit's remedy is the phase dialog's "interior-only" option (binary erosion
of the domain mask before averaging, viewer_dialogs._domain_composition).
This script re-names every saved label set from eroded-mask means (0, 1, 2
bins = 0, 12, 24 um) in ratio and structural mode on three element sets and
re-scores each against EBSD.

Reads quant_ebsd_<m>_labels.npz + serpdi_<m>_wt_maps_bin2.npz; writes
quant_ebsd_erode.json and quant_ebsd_erode_summary.txt beside this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

REPO = Path(__file__).resolve().parents[3]   # <repo>/_Manuscript/figures/ebsd_quantified
sys.path.insert(0, str(REPO))
import mineral_id  # noqa: E402
from ebsd_validation import confusion_matrix  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = {1: Path(r"D:/data/EBSD/serpdi_1_EBSD"), 2: Path(r"D:/data/EBSD/serpdi_2_EBSD")}
TRACES = {"Cr", "Mn", "Na", "K", "Ni"}
ELSETS = {"all12": set(), "noTrace": TRACES, "majors": TRACES | {"Ti"}}
MODES = {"ratio": False, "structural": True}
FE3 = 0.0
ERODE = (0, 1, 2)


def purity(res):
    return float(res.counts.max(axis=0).sum() / max(res.counts.sum(), 1))


def means_eroded(maps, labels, live, erode):
    out = {}
    for cid in np.unique(labels):
        if cid < 0:
            continue
        mask = labels == cid
        if erode > 0:
            inner = binary_erosion(mask, iterations=erode)
            if inner.any():
                mask = inner
        sel = mask & live
        if sel.sum() == 0:
            sel = (labels == cid) & live
        out[int(cid)] = ({el: float(maps[el][sel].mean()) for el in maps}, int(sel.sum()))
    return out


def main():
    report, lines = {}, []
    lines.append(f"{'map':>3s} {'labels':16s} {'erode':>5s} {'naming':20s} {'acc':>6s}  names")
    for m in (1, 2):
        z = np.load(HERE / f"serpdi_{m}_wt_maps_bin2.npz")
        maps = {k: z[k] for k in z.files}
        live = sum(maps.values()) > 0
        L = np.load(HERE / f"quant_ebsd_{m}_labels.npz")
        ebsd_crop = L["ebsd"].astype(np.int32)
        legend = json.loads((SRC[m] / "ebsd_phases.json").read_text(encoding="utf-8"))
        ebsd_names = {int(k): v["name"] for k, v in legend.items() if int(k) > 0}
        report[m] = {}
        for lset in [k for k in L.files if k != "ebsd"]:
            labels = L[lset]
            report[m][lset] = {}
            for erode in ERODE:
                means = means_eroded(maps, labels, live, erode)
                for mode, quantified in MODES.items():
                    for elset, drop in ELSETS.items():
                        names, top = {}, {}
                        for cid, (comp, n) in means.items():
                            c = {el: v for el, v in comp.items() if el not in drop}
                            mm = mineral_id.identify(c, quantified=quantified, fe3_fraction=FE3,
                                                     top_n=3, coating_elements=frozenset())
                            names[cid] = mm[0].name if mm else f"Phase {cid + 1}"
                            top[cid] = [(x.name, round(x.score, 3)) for x in mm]
                        res = confusion_matrix(ebsd_crop, labels, ebsd_names, names)
                        key = f"e{erode}/{mode}/{elset}"
                        report[m][lset][key] = {
                            "accuracy": round(res.accuracy, 4), "purity": round(purity(res), 4),
                            "names": {str(c): n for c, n in names.items()},
                            "top3": {str(c): t for c, t in top.items()},
                            "n_px": {str(c): means[c][1] for c in means}}
                        lines.append(f"{m:3d} {lset:16s} {erode:5d} {mode + '/' + elset:20s} "
                                     f"{res.accuracy:6.3f}  {[names[c] for c in sorted(names)]}")
                        print(lines[-1], flush=True)
    (HERE / "quant_ebsd_erode.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (HERE / "quant_ebsd_erode_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
