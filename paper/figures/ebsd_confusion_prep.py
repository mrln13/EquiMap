# -*- coding: utf-8 -*-
"""Prep for fig_ebsd.py: EBSD-vs-EDS confusion on an h5oina-converted dataset.

Replicates the GUI path with one fixed parameter set for both maps:
full-map ROI -> band maps from the eds_cache -> sparse-element check ->
sigma=2 smoothing -> feature matrix (min_count_ratio=0.1) -> row-normalise
-> PCA(0.92) + GMM auto-k (ICL, seeded) -> min_region 10 cleanup -> NN fill
-> ratio-only mineral ID (Fe3+=0.0, coating={"C"}) ->
ebsd_validation.confusion_matrix.
GMM only (never "Use EBSD phases"), so the comparison is independent.

sigma=2 / min_region=10 instead of the dialog defaults (3 / 50) because at
6 um/px the defaults smooth over 18 um and erase sub-1800 um^2 grains --
resolution-dependent quantities, sized here to the coarse pixel; the same
values are used for both maps, untuned per dataset. The defaults give the
same structure with softer boundaries (purity 82.4/83.6% instead of
86.9/89.9%, k=5 with a split diopside domain instead of k=4).

Interior-only naming (2026-10-09, --erode-px): the composition that names a
domain is taken from the domain mask eroded by N pixels, the classification
dialog's "Interior only" option (viewer_dialogs._domain_composition: binary
erosion, full-mask fallback when a domain vanishes). On these maps the
diopside domain holds a quarter of unindexed chlorite matrix and boundary
pixels, and its full-mask mean is a diopside-chlorite mixture that the
library reads as amphibole. --erode-scan reports the naming and accuracy for
several erosions from the SAME label map (naming only; the segmentation
never changes), so the sensitivity is on record in the meta file.
--erode-px 0 reproduces the pre-2026-10-09 staging exactly.

Writes into this directory, per dataset:
  ebsd_val_<tag>.npz       eds_labels (int16) + ebsd ordinals (int8)
  <tag>_seg_meta.json      auto-ID names per domain (+ top-5 candidates,
                           comp_erosion_px, erosion_scan) + EBSD legend
  <tag>_confusion.csv      full confusion export (counts + recall/precision)

The same result comes from the viewer: open the dataset, whole-map ROI,
Compute, Segment phases (sigma=2, min region 10, other options default),
tick "Interior only" (N px), Identify minerals, then "EBSD vs EDS
confusion ..." and its CSV export.

Usage:  python ebsd_confusion_prep.py <dataset_dir> [--repo <root>]
            [--erode-px N] [--erode-scan 0,2,3,4,6,8] [--reuse-labels]
        e.g. python ebsd_confusion_prep.py "D:\\data\\EBSD\\serpdi_2_EBSD" --erode-px 3
--reuse-labels re-names the staged label map (ebsd_val_<tag>.npz) at another
erosion without re-running the clustering.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

HERE = Path(__file__).resolve().parent

SIGMA = 2.0        # px; 12 um at the 6 um/px map resolution
MIN_SIZE = 10      # px; smallest surviving domain region

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--repo", default=str(HERE.parent.parent))
ap.add_argument("--erode-px", type=int, default=0,
                help="interior-only erosion (px) for the staged naming; 0 = full mask")
ap.add_argument("--erode-scan", default="0,2,3,4,6,8",
                help="comma-separated erosions (px) for the naming-sensitivity report")
ap.add_argument("--reuse-labels", action="store_true",
                help="load the staged label map instead of re-clustering")
args = ap.parse_args()
sys.path.insert(0, args.repo)

import eds_cache
import mineral_id
from viewer_segment_helpers import (
    _sparse_element_exclusions, _smooth_element_maps, _build_feature_matrix,
    _normalize_features, _run_clustering, _clean_label_map, _fill_unlabeled,
    _zero_signal_mask,
)
from viewer_thermo_helpers import _net_count_fractions
from ebsd_validation import (
    crop_ebsd_to_phase_grid, confusion_matrix, confusion_to_csv_rows,
)

t0 = time.time()
DATASET = args.dataset
TAG = Path(DATASET).name
ERODE = max(0, int(args.erode_px))
SCAN = sorted({max(0, int(s)) for s in args.erode_scan.split(",") if s.strip()} | {ERODE})


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


cache_dir = eds_cache.resolve_cache_dir(DATASET)
manifest = eds_cache.load_manifest(cache_dir)
H, W = int(manifest["H"]), int(manifest["W"])
elements = list(manifest["elements"])
roi = (0, H, 0, W)
log(f"{TAG}: {H}x{W} px, {len(elements)} elements; naming erosion {ERODE} px, scan {SCAN}")

band_maps = eds_cache.roi_from_cache(cache_dir, roi, elements,
                                     blend_mode="mean", progress=False)

if args.reuse_labels:
    staged = np.load(HERE / f"ebsd_val_{TAG}.npz")
    label_map = staged["eds_labels"].astype(np.int16)
    prev = json.loads((HERE / f"{TAG}_seg_meta.json").read_text())
    k_chosen, auto_method = int(prev["k"]), prev["auto_method"]
    el_names, excluded = list(prev["elements_used"]), list(prev["excluded"])
    log(f"reusing staged labels: k={k_chosen} ({auto_method})")
else:
    # Signed nets through the toolkit's background module (Oxford h5oina
    # counts carry zero sidebands, so every scheme reduces to the window
    # integral here); clipped after smoothing inside _smooth_element_maps,
    # as the viewer does.
    from eds_background import net_count_maps
    maps = net_count_maps(band_maps)

    raw_total = np.stack(list(maps.values()), axis=-1).sum(axis=-1)
    excluded, _rep = _sparse_element_exclusions(
        maps, raw_total > 0, max_zero_pct=90.0, sigma=SIGMA)
    maps = {el: a for el, a in maps.items() if el not in excluded}
    del raw_total

    # Zero-signal (no-data) pixels (off-sample area / unscanned margins) are
    # left unclassified (protect_background, dialog default ON): judged on the
    # raw pre-smoothing maps, cleared from the clustered labels and protected
    # from the nearest-neighbour fill.
    raw_zero = _zero_signal_mask(maps)
    log(f"zero-signal px: {int(raw_zero.sum()):,} "
        f"({100.0 * raw_zero.mean():.1f}% of the map), left unclassified")

    maps = _smooth_element_maps(maps, sigma=SIGMA)
    X, valid, shape, el_names = _build_feature_matrix(
        maps, min_count_pct=0, min_count_ratio=0.1, overlap_mask=None)
    del maps
    X_norm = _normalize_features(X, norm_method="row")
    del X

    labels_flat, k_chosen, _bic, _icl, auto_method = _run_clustering(
        X_norm, n_clusters="auto", k_max=12, min_k=2, pca_variance=0.92,
        posterior_min=0.0, density_pct=0.0)
    del X_norm
    log(f"k={k_chosen} ({auto_method})")

    label_map = np.full(shape, -1, dtype=np.int16)
    label_map[valid] = labels_flat.astype(np.int16)
    rejected = valid & (label_map < 0)
    label_map[raw_zero] = -1
    label_map = _clean_label_map(label_map, min_size=MIN_SIZE)
    _fill_unlabeled(label_map, protect=rejected | raw_zero)

ebsd = np.load(str(Path(DATASET) / "ebsd_phase_map.npy"))
legend = json.loads((Path(DATASET) / "ebsd_phases.json").read_text())
ebsd_names = {int(k): v["name"] for k, v in legend.items() if int(k) > 0}
crop = crop_ebsd_to_phase_grid(ebsd, roi, 1, label_map.shape, None)


def name_domains(erode):
    """Ratio-only ID of every domain from its (interior-only) mean composition.

    Mirrors viewer_dialogs._domain_composition: the domain mask is eroded by
    ``erode`` px before averaging; a domain that vanishes falls back to its
    full mask. Returns names, top-5 candidates and the pixel count used.
    """
    names, cands, n_used = {}, {}, {}
    for cid in range(k_chosen):
        mask = label_map == cid
        if erode > 0 and mask.any():
            inner = binary_erosion(mask, iterations=erode)
            if inner.any():
                mask = inner
            else:
                log(f"  cid {cid}: erosion {erode} px left no pixels, full mask used")
        frac, n_px = _net_count_fractions(band_maps, sub_mask=mask)
        comp_wt = {k: 100.0 * float(v) for k, v in frac.items()}
        matches = mineral_id.identify(comp_wt, quantified=False, fe3_fraction=0.0,
                                      top_n=5, coating_elements=frozenset({"C"}))
        names[cid] = matches[0].name if matches else f"Phase {cid + 1}"
        cands[cid] = [(m.name, round(float(m.confidence_pct), 1)) for m in matches]
        n_used[cid] = int(n_px)
    return names, cands, n_used


scan_report = {}
for erode in SCAN:
    names, cands, n_used = name_domains(erode)
    res = confusion_matrix(crop, label_map, ebsd_names, names)
    scan_report[erode] = {"accuracy": round(float(res.accuracy), 4),
                          "names": {str(c): n for c, n in names.items()},
                          "candidates": {str(c): v for c, v in cands.items()},
                          "n_px_used": {str(c): n for c, n in n_used.items()}}
    log(f"erosion {erode:2d} px: accuracy {res.accuracy:.4f}  "
        f"{[names[c] for c in sorted(names)]}")

eds_names, cands, n_used = name_domains(ERODE)
for cid in range(k_chosen):
    log(f"  cid {cid} ({n_used[cid]:,} px at {ERODE} px erosion) -> {eds_names[cid]}  "
        f"{cands[cid]}")
res = confusion_matrix(crop, label_map, ebsd_names, eds_names)
log(f"accuracy {res.accuracy:.4f} over {res.n_both:,} both-valid px")

np.savez_compressed(HERE / f"ebsd_val_{TAG}.npz",
                    eds_labels=label_map.astype(np.int16),
                    ebsd=crop.astype(np.int8))
json.dump({"eds_names": {str(c): n for c, n in eds_names.items()},
           "candidates": {str(c): v for c, v in cands.items()},
           "comp_erosion_px": ERODE,
           "n_px_naming": {str(c): n for c, n in n_used.items()},
           "erosion_scan": {str(e): v for e, v in scan_report.items()},
           "k": k_chosen, "auto_method": auto_method,
           "sigma": SIGMA, "min_size": MIN_SIZE,
           "elements_used": el_names, "excluded": list(excluded),
           "ebsd_phases": legend},
          open(HERE / f"{TAG}_seg_meta.json", "w"), indent=2)
with open(HERE / f"{TAG}_confusion.csv", "w", newline="",
          encoding="utf-8") as f:
    csv.writer(f).writerows(confusion_to_csv_rows(res))
log(f"staged ebsd_val_{TAG}.npz + {TAG}_seg_meta.json + {TAG}_confusion.csv")
