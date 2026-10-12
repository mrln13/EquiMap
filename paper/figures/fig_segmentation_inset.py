"""Prep for fig_segmentation panel (b): re-segment the nodule window locally.

The whole-section map of panel (a) is segmented at 1/5 resolution with
sigma=2.5, which blurs the nodule-interior texture. This script re-segments
JUST the inset window at full cache resolution with lighter smoothing
(sigma=1), using the same headless pipeline as the viewer, and maps the local
clusters onto the global phase names by majority overlap with the
whole-section label map - so panel (b) keeps panel (a)'s legend and colours.

Writes:  phase_map_inset.npz   (int16 labels, GLOBAL cluster ids, -1 invalid;
                                compressed, key "labels")
         phase_map_inset.json  (window coords, pixel pitch, recipe)

Usage:   python fig_segmentation_inset.py --dataset <stitched_dir>
         [--window Y0 X0] [--size 4500] [--sigma 1.0] [--k 6]
The default window is the densest-epidote region of the global label map
(the same auto-pick as fig_segmentation.py).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"D:\Actief\C23S1A_Quant")
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--window", nargs=2, type=int, default=None,
                    metavar=("Y0", "X0"),
                    help="window top-left in MOSAIC px (default: auto-pick)")
    ap.add_argument("--size", type=int, default=4500,
                    help="window edge in mosaic px")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--als-name", default="Sillimanite",
                    help="name for a local cluster the mineral ID calls Al2SiO5 (the polymorph is "
                         "resolved petrographically, not by EDS); '' disables")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import eds_cache
    from viewer_segment_helpers import (
        _smooth_element_maps, _build_feature_matrix, _normalize_features,
        _run_clustering, _clean_label_map,
    )

    glob_labels = np.load(HERE / "phase_map_labels.npz")["labels"]
    with open(HERE / "phase_legend.csv") as f:
        legend = list(csv.DictReader(f))
    seg_json = json.loads((HERE / "segmentation_params.json").read_text())
    seg_ds = int(seg_json.get("seg_ds", 1))
    elements = [e for e in seg_json["elements_used"]]

    if args.window is None:
        ep_cid = next(int(r["cluster_id"]) for r in legend
                      if r["name"].strip() == "Epidote")
        from scipy.ndimage import uniform_filter
        win_g = max(3, args.size // seg_ds // 4)
        m = (glob_labels == ep_cid)[::4, ::4].astype(np.float32)
        dens = uniform_filter(m, size=max(3, win_g // 4))
        iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
        gy = int(iy * 4 - args.size / seg_ds / 2)
        gx = int(ix * 4 - args.size / seg_ds / 2)
        gy = max(0, min(glob_labels.shape[0] - args.size // seg_ds, gy))
        gx = max(0, min(glob_labels.shape[1] - args.size // seg_ds, gx))
        y0, x0 = gy * seg_ds, gx * seg_ds
    else:
        y0, x0 = args.window
    y1, x1 = y0 + args.size, x0 + args.size
    print(f"window: mosaic y {y0}:{y1}, x {x0}:{x1}")

    cache_dir = eds_cache.resolve_cache_dir(args.dataset)
    bands = eds_cache.roi_from_cache(cache_dir, (y0, y1, x0, x1), elements,
                                     blend_mode="mean", progress=False)
    # Signed line-aware nets (the plan needs every cached element's line and
    # the anchor bands, so read them all and keep the recipe's elements);
    # clipped after smoothing in _smooth_element_maps, exactly as the viewer.
    from eds_background import anchor_elements, background_plan, net_count_maps
    cached = list(eds_cache.load_manifest(cache_dir).get("elements") or elements)
    plan = background_plan(cached)
    extra = [e for e in anchor_elements(plan, elements) if e in cached and e not in bands]
    if extra:
        bands.update(eds_cache.roi_from_cache(cache_dir, (y0, y1, x0, x1), extra,
                                              blend_mode="mean", progress=False))
    maps = {el: a.astype(np.float32)
            for el, a in net_count_maps(bands, plan=plan, elements=elements).items()}
    del bands
    # Raw (un-smoothed) copies: the compositional cluster-to-phase assignment
    # below needs mean net-count vectors on the additive basis.
    raw_maps = {el: maps[el].copy() for el in elements}

    # Elements fixed to the global recipe's list (C/P already excluded there);
    # no sparsity re-check: at sigma=1 the pooling rescue would drop the very
    # discriminators the global run validated (Mg/Na/Ca), and this window is
    # exactly where they matter.
    maps = _smooth_element_maps(maps, sigma=args.sigma)
    X, valid, shape, el_names = _build_feature_matrix(
        maps,
        min_count_pct=seg_json.get("min_count_pct", 5.0),
        min_count_ratio=seg_json.get("min_count_ratio", 0.1),
        overlap_mask=None)
    X_norm = _normalize_features(X, norm_method=seg_json.get("norm_method", "row"))
    labels_flat, k_chosen, *_ = _run_clustering(
        X_norm, n_clusters=args.k, k_max=15, min_k=2,
        pca_variance=seg_json.get("pca_variance", 0.92))
    local = np.full(shape, -1, dtype=np.int16)
    local[valid] = labels_flat.astype(np.int16)
    local = _clean_label_map(local,
                             min_size=int(seg_json.get("min_region_size", 15)))
    # Nearest-neighbour fill of excluded pixels, as in the viewer pipeline
    # (gap-free domain map).
    unlabeled = local < 0
    if unlabeled.any() and (~unlabeled).any():
        from scipy.ndimage import distance_transform_edt
        nearest = distance_transform_edt(unlabeled, return_distances=False,
                                         return_indices=True)
        local[unlabeled] = local[tuple(nearest[:, unlabeled])]

    # Assign each local cluster to a section-wide phase by COMPOSITIONAL
    # similarity, not by spatial majority against the global map. Spatial
    # majority is wrong here by construction: over the fine-grained matrix
    # the global map says "quartz-mica matrix" (a sub-pixel mixture class at
    # 1/seg_ds resolution), so resolved clean-quartz grains would inherit
    # the mixture label even though at full resolution they are quartz. The
    # inset's point is the opposite: mixed-at-map-resolution areas RESOLVE
    # into their constituents. Reference vectors are the global domains'
    # mean net-count vectors measured over this window (upsampled global
    # labels), compared by cosine similarity of the normalised vectors.
    gwin = glob_labels[y0 // seg_ds:(y1 + seg_ds - 1) // seg_ds,
                       x0 // seg_ds:(x1 + seg_ds - 1) // seg_ds]
    gup = np.repeat(np.repeat(gwin, seg_ds, 0), seg_ds, 1)[:shape[0], :shape[1]]

    def mean_vec(mask):
        n = int(mask.sum())
        if n == 0:
            return None
        v = np.array([float(raw_maps[el][:shape[0], :shape[1]][mask].sum())
                      for el in el_names]) / n
        s = v.sum()
        return v / s if s > 0 else None

    MIN_REF_PX = 1000
    ref = {}
    for gcid in sorted(int(v) for v in np.unique(gup) if v >= 0):
        if int((gup == gcid).sum()) < MIN_REF_PX:
            continue
        v = mean_vec(gup == gcid)
        if v is not None:
            ref[gcid] = v

    names = {int(r["cluster_id"]): r["name"] for r in legend}
    # Mineral ID of each local cluster (calibrated mean composition): a
    # confident library mineral that has NO global domain (epidote in the
    # nodule cores, below the global map's resolution) gets an inset-only
    # id above the global range instead of being forced onto the nearest
    # global name. Recorded in phase_map_inset.json ("extra_names").
    import eds_quantify as _eq
    import mineral_id as _mi
    _cal = json.loads((HERE / "eds_calibration.json").read_text())["calibration"]
    _k = _eq.Calibration.from_dict(_cal).k()
    _global_names = set(names.values())
    _rank = _mi._CONFIDENCE_RANK
    extra_names = {}
    next_extra = max(names) + 1

    def ident(mask):
        n = int(mask.sum())
        raw = {el: max(0.0, float(raw_maps[el][:shape[0], :shape[1]][mask].sum()) / n)
               for el in el_names}
        tot = sum(raw.values())
        if tot <= 0:
            return None
        wt = _eq.apply_calibration_fractions({el: v / tot for el, v in raw.items()}, _k)
        m = _mi.identify(wt, quantified=True, fe3_fraction=0.1, top_n=3,
                         coating_elements=frozenset({"C"}))
        return (m[0], wt) if m else None

    out = np.full(shape, -1, dtype=np.int16)
    mapping = {}
    majority = {}
    ids = {}
    for cid in range(k_chosen):
        sel = local == cid
        if not sel.any():
            continue
        gl = gup[sel]
        gl = gl[gl >= 0]
        majority[cid] = int(np.bincount(gl).argmax()) if gl.size else -1
        v = mean_vec(sel)
        if v is None:
            mapping[cid] = majority[cid]
        else:
            sims = {g: float(np.dot(v, r) /
                             (np.linalg.norm(v) * np.linalg.norm(r)))
                    for g, r in ref.items()}
            mapping[cid] = max(sims, key=sims.get)
        top = ident(sel)
        if top is not None:
            m, wt = top
            ids[cid] = f"{m.name} {m.confidence_pct:.0f}% ({m.confidence})"
            # The Al2SiO5 entry is the merged polymorph group; the user names
            # the polymorph (sillimanite here) and any confidence counts,
            # because a nodule core is small and boundary-contaminated.
            nm = args.als_name if (args.als_name and m.name.startswith("Al2SiO5")) else m.name
            confident = (nm != m.name) or _rank.get(m.confidence, 0) >= _rank.get("moderate", 2)
            if nm not in _global_names and nm not in extra_names.values() and confident:
                extra_names[next_extra] = nm
                mapping[cid] = next_extra
                next_extra += 1
            elif nm in extra_names.values():
                mapping[cid] = next(i for i, n_ in extra_names.items() if n_ == nm)
        out[sel] = mapping[cid]
    print("local cluster mineral ID:", ids)
    print("inset-only phases (no global domain):", extra_names)
    print("local -> global (composition):",
          {c: names.get(g, extra_names.get(g, g)) for c, g in mapping.items()})
    print("local -> global (spatial majority, for reference):",
          {c: names.get(g, g) for c, g in majority.items()})

    np.savez_compressed(HERE / "phase_map_inset.npz", labels=out)
    (HERE / "phase_map_inset.json").write_text(json.dumps({
        "window_mosaic_px": [y0, y1, x0, x1],
        "sigma": args.sigma, "k": args.k, "elements": el_names,
        "seg_ds": 1, "extra_names": {int(i): nm for i, nm in extra_names.items()},
        "cluster_mapping": {str(c): int(g)
                                         for c, g in mapping.items()},
        "note": "local full-resolution re-segmentation for the fig 4 inset; "
                "clusters assigned to the global phase names by compositional "
                "similarity (cosine on mean net-count vectors), so areas that "
                "are mixed at map resolution resolve into their constituents",
    }, indent=2))
    print(f"wrote phase_map_inset.npz ({out.shape}) + phase_map_inset.json")


if __name__ == "__main__":
    main()
