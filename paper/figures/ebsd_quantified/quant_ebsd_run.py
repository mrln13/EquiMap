# -*- coding: utf-8 -*-
"""Quantified-tier EBSD validation (manuscript section 5.2) on the binning-2
AZtec QuantMap wt% exports of the two serpdi maps (12 um per bin).

Three stages per map, every result scored against the EBSD phase map with
ebsd_validation.confusion_matrix (alias-matched accuracy) and a pixel-weighted
purity (each EDS domain's majority-phase share, weighted by domain size, over
the pixels valid in both maps):

  0. baseline  : the published count-based GMM domains at 6 um with their
                 ratio-only names (checks the purity definition against the
                 manuscript's 86.9 / 89.9 %), then the same domains decimated
                 to the 12 um bin grid (the apples-to-apples baseline);
  1. re-ID     : the decimated published domains re-identified from their
                 mean wt% (isolates the quantification's effect on naming);
  2. full      : GMM segmentation of the wt% maps themselves with the
                 section-5.2 pipeline (sigma / min_region scaled to the bin),
                 clustered on all 12 elements and on the trace-free set.

Naming variants in stages 1 and 2: ratio-only and structural mode, each on
all 12 elements, without the five noise-level traces (Cr Mn Na K Ni), and on
the majors only (also without Ti).

Inputs : serpdi_{1,2}_wt_maps_bin2.npz (check_wt_bin2.py), the EBSD maps and
         the staged section-5.2 run in _Manuscript/figures.
Outputs: quant_ebsd_<m>.json, quant_ebsd_<m>_labels.npz, confusion/*.csv,
         quant_ebsd_summary.txt (all beside this script).
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]   # <repo>/_Manuscript/figures/ebsd_quantified
sys.path.insert(0, str(REPO))
import mineral_id  # noqa: E402
from viewer_segment_helpers import (  # noqa: E402
    _sparse_element_exclusions, _smooth_element_maps, _build_feature_matrix,
    _normalize_features, _run_clustering, _clean_label_map, _fill_unlabeled,
    _zero_signal_mask,
)
from ebsd_validation import (  # noqa: E402
    crop_ebsd_to_phase_grid, confusion_matrix, confusion_to_csv_rows,
)

HERE = Path(__file__).resolve().parent
FIG = REPO / "_Manuscript/figures"
SRC = {1: Path(r"D:/data/EBSD/serpdi_1_EBSD"), 2: Path(r"D:/data/EBSD/serpdi_2_EBSD")}
(HERE / "confusion").mkdir(exist_ok=True)

BIN = 2
SIGMA = 2.0 / BIN      # 12 um, the width used at 6 um/px in section 5.2
MIN_SIZE = max(1, round(10 / BIN ** 2))   # 10 px at 6 um = 360 um^2 -> 2.5 px
TRACES = {"Cr", "Mn", "Na", "K", "Ni"}
ELSETS = {"all12": set(), "noTrace": TRACES, "majors": TRACES | {"Ti"}}
MODES = {"ratio": False, "structural": True}
FE3 = 0.0              # as in ebsd_confusion_prep.py

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def purity(res):
    return float(res.counts.max(axis=0).sum() / max(res.counts.sum(), 1))


def majority_rows(res):
    """Table-2 rows: for each EBSD phase the domain holding most of its pixels,
    that domain's share of the phase and its precision for the phase."""
    rows = []
    col_tot = res.counts.sum(axis=0)
    for i, lab in enumerate(res.row_labels):
        j = int(np.argmax(res.counts[i]))
        n = int(res.counts[i].sum())
        rows.append({"phase": lab, "n": n, "domain": res.col_labels[j],
                     "cid": int(res.col_cids[j]),
                     "share": float(res.counts[i, j] / max(n, 1)),
                     "precision": float(res.counts[i, j] / max(col_tot[j], 1))})
    return rows


def domain_means(maps, labels, live):
    out = {}
    for cid in np.unique(labels):
        if cid < 0:
            continue
        sel = (labels == cid) & live
        if sel.sum() == 0:
            continue
        out[int(cid)] = ({el: float(maps[el][sel].mean()) for el in maps}, int(sel.sum()))
    return out


def name_domains(means, mode, elset):
    quantified = MODES[mode]
    drop = ELSETS[elset]
    names, detail = {}, {}
    for cid, (comp, n) in means.items():
        c = {el: v for el, v in comp.items() if el not in drop}
        m = mineral_id.identify(c, quantified=quantified, fe3_fraction=FE3,
                                top_n=3, coating_elements=frozenset())
        names[cid] = m[0].name if m else f"Phase {cid + 1}"
        detail[cid] = [(x.name, round(x.confidence_pct, 1), x.method) for x in m]
    return names, detail


def score(labels, ebsd_crop, ebsd_names, names, tag):
    res = confusion_matrix(ebsd_crop, labels, ebsd_names, names)
    with open(HERE / "confusion" / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(confusion_to_csv_rows(res))
    return res


def evaluate(labels, ebsd_crop, ebsd_names, maps, live, tag, out):
    means = domain_means(maps, labels, live)
    out["domain_means"] = {str(c): {"n": n, **{k: round(v, 3) for k, v in comp.items()}}
                           for c, (comp, n) in means.items()}
    out["variants"] = {}
    for mode in MODES:
        for elset in ELSETS:
            names, detail = name_domains(means, mode, elset)
            res = score(labels, ebsd_crop, ebsd_names, names, f"{tag}_{mode}_{elset}")
            out["variants"][f"{mode}/{elset}"] = {
                "accuracy": round(res.accuracy, 4), "purity": round(purity(res), 4),
                "names": {str(c): n for c, n in names.items()},
                "top3": {str(c): d for c, d in detail.items()},
                "majority": majority_rows(res),
            }
            log(f"  {tag:18s} {mode:10s} {elset:8s} acc {res.accuracy:.3f}  "
                f"names {[names[c] for c in sorted(names)]}")
    return means


def main():
    summary = []
    for m in (1, 2):
        log(f"########## map {m} ##########")
        z = np.load(HERE / f"serpdi_{m}_wt_maps_bin2.npz")
        maps = {k: z[k] for k in z.files}
        H, W = maps["Si"].shape
        live = sum(maps[e] for e in maps) > 0
        ebsd = np.load(SRC[m] / "ebsd_phase_map.npy")
        legend = json.loads((SRC[m] / "ebsd_phases.json").read_text(encoding="utf-8"))
        ebsd_names = {int(k): v["name"] for k, v in legend.items() if int(k) > 0}
        report = {"map": m, "shape": [H, W], "sigma": SIGMA, "min_size": MIN_SIZE,
                  "fe3": FE3, "stages": {}}

        # ---- stage 0: published run, at 6 um and decimated to the bin grid ----
        pub = np.load(FIG / f"ebsd_val_serpdi_{m}_EBSD.npz")
        meta = json.loads((FIG / f"serpdi_{m}_EBSD_seg_meta.json").read_text(encoding="utf-8"))
        pub_names = {int(k): v for k, v in meta["eds_names"].items()}
        res6 = confusion_matrix(pub["ebsd"].astype(np.int32), pub["eds_labels"], ebsd_names, pub_names)
        log(f"published 6 um: acc {res6.accuracy:.4f} purity {purity(res6):.4f} (ms: 0.32/0.27, 86.9/89.9)")
        lab_pub = pub["eds_labels"][::BIN, ::BIN][:H, :W].astype(np.int16)
        ebsd_crop = crop_ebsd_to_phase_grid(ebsd, (0, BIN * H, 0, BIN * W), BIN, (H, W), None)
        assert ebsd_crop is not None and ebsd_crop.shape == (H, W), "EBSD crop mismatch"
        assert lab_pub.shape == (H, W)
        lab_pub[~live] = -1
        res_b = score(lab_pub, ebsd_crop, ebsd_names, pub_names, f"map{m}_baseline_bin2")
        report["stages"]["baseline"] = {
            "acc_6um": round(res6.accuracy, 4), "purity_6um": round(purity(res6), 4),
            "acc_bin2": round(res_b.accuracy, 4), "purity_bin2": round(purity(res_b), 4),
            "names": {str(c): n for c, n in pub_names.items()},
            "majority_bin2": majority_rows(res_b)}
        log(f"published domains on bin grid: acc {res_b.accuracy:.4f} purity {purity(res_b):.4f}")
        summary.append((m, "published counts, 6 um", "ratio/all17", res6.accuracy, purity(res6)))
        summary.append((m, "published domains, bin2", "ratio/all17", res_b.accuracy, purity(res_b)))

        # ---- stage 1: re-identify the published domains from wt% ----
        st = {}
        evaluate(lab_pub, ebsd_crop, ebsd_names, maps, live, f"map{m}_reid", st)
        report["stages"]["reid"] = st
        for k, v in st["variants"].items():
            summary.append((m, "published domains re-ID wt%", k, v["accuracy"], v["purity"]))

        # ---- stage 2: full GMM on the wt% maps ----
        labels_out = {"ebsd": ebsd_crop.astype(np.int8), "published_bin2": lab_pub}
        for feat_tag, drop in (("all12", set()), ("noTrace", TRACES)):
            fmaps = {el: a.copy() for el, a in maps.items() if el not in drop}
            raw_total = sum(fmaps.values())
            excluded, _rep = _sparse_element_exclusions(fmaps, raw_total > 0,
                                                        max_zero_pct=90.0, sigma=SIGMA)
            fmaps = {el: a for el, a in fmaps.items() if el not in excluded}
            raw_zero = _zero_signal_mask(fmaps)
            log(f"GMM[{feat_tag}] features {sorted(fmaps)} excluded {sorted(excluded)}; "
                f"zero-signal {100 * raw_zero.mean():.1f}%")
            fmaps = _smooth_element_maps(fmaps, sigma=SIGMA)
            X, valid, shape, el_names = _build_feature_matrix(
                fmaps, min_count_pct=0, min_count_ratio=0.1, overlap_mask=None)
            del fmaps
            X_norm = _normalize_features(X, norm_method="row")
            del X
            labels_flat, k, _bic, _icl, method = _run_clustering(
                X_norm, n_clusters="auto", k_max=12, min_k=2, pca_variance=0.92,
                posterior_min=0.0, density_pct=0.0)
            del X_norm
            log(f"GMM[{feat_tag}] k={k} ({method})")
            lab = np.full(shape, -1, dtype=np.int16)
            lab[valid] = labels_flat.astype(np.int16)
            rejected = valid & (lab < 0)
            lab[raw_zero] = -1
            lab = _clean_label_map(lab, min_size=MIN_SIZE)
            _fill_unlabeled(lab, protect=rejected | raw_zero)
            st = {"k": int(k), "method": method, "features": el_names,
                  "excluded": sorted(excluded)}
            evaluate(lab, ebsd_crop, ebsd_names, maps, live, f"map{m}_gmm{feat_tag}", st)
            report["stages"][f"gmm_{feat_tag}"] = st
            labels_out[f"gmm_{feat_tag}"] = lab
            for kk, v in st["variants"].items():
                summary.append((m, f"GMM on wt% [{feat_tag}] k={k}", kk, v["accuracy"], v["purity"]))
        np.savez_compressed(HERE / f"quant_ebsd_{m}_labels.npz", **labels_out)
        (HERE / f"quant_ebsd_{m}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = [f"{'map':>3s}  {'segmentation':34s} {'naming':20s} {'acc':>6s} {'purity':>7s}"]
    for m, seg, nm, acc, pur in summary:
        lines.append(f"{m:3d}  {seg:34s} {nm:20s} {acc:6.3f} {pur:7.3f}")
    txt = "\n".join(lines)
    (HERE / "quant_ebsd_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    log("done")


if __name__ == "__main__":
    main()
