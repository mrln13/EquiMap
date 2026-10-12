"""Headless re-segmentation + classification of the worked example (Section 4.2).

Replays the viewer's Segment-phases pipeline on the July session's 1/5 band
maps with the recorded recipe (segmentation_params.json: sigma 2.5, row
normalisation, PCA 0.92, k = 6 fixed, min region 15 px, overlap exclusion,
sparse-element check at 97 %) but on the LINE-AWARE signed net counts
(eds_background, 2026-09-16), then the classification-dialog steps: per-domain
calibrated composition (k-factors from eds_calibration.json), mineral ID
(mineral_id.identify, Fe3+/Fe 0.1, C coating), naming, NNLS unmixing of the
quartz-mica matrix domain, and the same export files the dialog writes:

  phase_map_labels.npz      int16 label map (key "labels"; -1 = unclassified)
  phase_map_combined.png    tab20-coloured map (fig_segmentation.py)
  phase_legend.csv          cluster_id, name, pixel_count (full masks -> modes)
  phase_minerals.csv        dialog export layout (fig_pt.py weights)
  phase_composition.csv     calibrated wt% per domain (erosion --comp-erosion)
  segmentation_params.json  recipe + background + unmix block + provenance
  segmentation_report.json  sparsity report, k-selection scores, contingency
                            with the previous (sideband-net) map, ID details

Domain NAMES follow the July classification wherever the pixels say so: a
domain whose pixels fall mostly (>50 %) on one previous domain inherits that
domain's decision. A previous mixed domain ("Quartz-mica matrix") makes the
new domain a MIXED domain too: it is named by its dominant NNLS endmember
("Quartz-rich matrix" / "Mica-rich matrix") and unmixed onto the clean
domains (never onto another mixed one). A previous mineral name that the
auto-ID does not reproduce becomes a manual override (method "manual"),
exactly what the dialog's Override column did in July for Quartz and
Epidote. Everything is printed so the mapping can be reviewed;
--n-clusters overrides the recipe's k for diagnostics.
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
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
import eds_background as bg                                       # noqa: E402
import eds_cache as edsc                                          # noqa: E402
import eds_quantify as eq                                         # noqa: E402
import mineral_id as mi                                           # noqa: E402
from viewer_segment_helpers import (                              # noqa: E402
    _sparse_element_exclusions, _smooth_element_maps, _build_feature_matrix,
    _normalize_features, _run_clustering, _clean_label_map, _fill_unlabeled,
    _compute_overlap_mask_from_weight, _cluster_dominant_elements,
    _label_map_to_rgba)
from viewer_thermo_helpers import _unmix_domain                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=r"D:/Actief/C23S1A_Quant")
ap.add_argument("--session-npz", default=r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.npz")
ap.add_argument("--recipe", default=str(HERE / "segmentation_params.json"))
ap.add_argument("--calibration", default=str(HERE / "eds_calibration.json"))
ap.add_argument("--old-labels", default=str(HERE / "phase_map_labels.npz"))
ap.add_argument("--old-legend", default=str(HERE / "phase_legend.csv"))
ap.add_argument("--out-dir", default=str(HERE))
ap.add_argument("--comp-erosion", type=int, default=0,
                help="interior erosion (px) for the per-domain composition / ID (dialog default 0)")
ap.add_argument("--fe3", type=float, default=0.1)
ap.add_argument("--coating", default="C")
ap.add_argument("--background", choices=list(bg.SCHEMES), default=None)
ap.add_argument("--n-clusters", default=None, help="override the recipe's k (an integer or 'auto' for the ICL sweep)")
ap.add_argument("--min-majority", type=float, default=0.5)
ap.add_argument("--pca-variance", type=float, default=None, help="override the recipe's PCA retained variance")
ap.add_argument("--elements", nargs="+", default=None, help="cluster on exactly these elements (diagnostic)")
ap.add_argument("--sigma", type=float, default=None, help="override the recipe's smoothing sigma")
ap.add_argument("--min-region", type=int, default=None, help="override the recipe's min_region_size")
ap.add_argument("--fit-balance", choices=["uniform", "density"], default=None, help="GMM fit sample: uniform pixels (recipe) or density-balanced over feature space")
ap.add_argument("--seed-mask", default=None, help="npz with a boolean 'mask' on the map grid: pixels of a phase to seed one extra GMM component with (its mean normalised feature vector)")
ap.add_argument("--seed-erode", type=int, default=3)
ap.add_argument("--merge-same-name", action="store_true", help="merge domains that received the same name (duplicated components), as the dialog's Merge selected does; mixed domains are merged by type before unmixing")
ap.add_argument("--seed", type=int, default=42,
                help="random seed of the fit sample and the mixture initialisations (42 = the recipe; "
                     "other values test seed-to-seed stability)")
ap.add_argument("--max-zero-pct", type=float, default=None, help="override the sparsity cut (100 disables it)")
ap.add_argument("--smooth-floor", choices=["zero", "noise"], default="noise",
                help="after smoothing: clip at zero and add the per-element noise level (toolkit default) or "
                     "clip at zero only (diagnostic: the exact-zero features that split clean quartz)")
args = ap.parse_args()
if args.background:
    bg.set_default_scheme(args.background)
scheme = bg.default_scheme()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)


recipe = json.load(open(args.recipe))
recipe_in = {k: v for k, v in recipe.items() if k not in ("unmix", "unclassified_px", "unclassified_frac",
                                                          "comp_erosion_px", "background", "elements_used",
                                                          "k_chosen", "auto_method", "provenance")}
log(f"recipe: {json.dumps(recipe_in)}")
cal_json = json.load(open(args.calibration))
cal = eq.Calibration.from_dict(cal_json["calibration"])
k = cal.k()
if cal.background != scheme:
    raise SystemExit(f"calibration background {cal.background!r} != current scheme {scheme!r}")
log(f"k-factors ({cal.background}): " + " ".join(f"{e} {v:.3g}" for e, v in sorted(k.items())))

z = np.load(args.session_npz)
elements = sorted({key.split("__")[0] for key in z.keys() if "__" in key})
shape = z[f"{elements[0]}__main"].shape
seg_ds = int(recipe.get("seg_ds", 5))
roi = recipe.get("roi") or [0, shape[0] * seg_ds, 0, shape[1] * seg_ds]
log(f"band maps {shape} at 1/{seg_ds}, elements {elements}, background {scheme}")
plan = bg.background_plan(elements)
print(bg.describe_plan(plan, scheme), flush=True)


def get_band(e, b):
    key = f"{e}__{b}"
    return z[key] if key in z else None


# 1. signed nets for every element (the plan knows every line in the dict)
maps = {}
for el in elements:
    maps[el] = bg.net_count_map(el, get_band, plan=plan, scheme=scheme)
log("signed nets computed")

# 2. manual element filter (recipe: none; --elements overrides)
el_filter = args.elements or recipe.get("element_filter")
if el_filter:
    maps = {el: a for el, a in maps.items() if el in el_filter}

# 3. zero-signal protection (recipe: off)
raw_zero = None
if recipe.get("protect_background", False):
    from viewer_segment_helpers import _zero_signal_mask
    raw_zero = _zero_signal_mask(maps)
    if not raw_zero.any():
        raw_zero = None

# 4. sparse-element exclusion (smoothing-aware), on the raw signed nets
max_zero_pct = float(args.max_zero_pct if args.max_zero_pct is not None else recipe.get("max_zero_pct", 90.0))
sigma = float(args.sigma if args.sigma is not None else recipe.get("smooth_sigma", 3.0))
sparsity_report = []
if max_zero_pct < 100.0:
    raw_total = np.zeros(shape, np.float32)
    for a in maps.values():
        raw_total += a
    valid_pixels = raw_total > 0
    del raw_total
    excluded, sparsity_report = _sparse_element_exclusions(
        maps, valid_pixels, max_zero_pct=max_zero_pct, sigma=sigma)
    del valid_pixels
    for el, zf, pooled, drop in sparsity_report:
        log(f"  sparsity {el:2s}: {zf:5.1f}% near-zero, ~{pooled:.1f} counts pooled" + ("  <-- EXCLUDED" if drop else ""))
    maps = {el: a for el, a in maps.items() if el not in excluded}
log(f"elements clustered on: {sorted(maps)}")

# 5. smoothing: clip after, plus the per-element noise floor (toolkit default;
#    --smooth-floor zero reproduces the exact-zero features for comparison)
from viewer_segment_helpers import _noise_floor
if args.smooth_floor == "noise":
    from scipy.ndimage import gaussian_filter
    for el, a in maps.items():
        sm = gaussian_filter(a.astype(np.float32), sigma=sigma) if sigma > 0 else a
        log(f"  noise floor {el}: {_noise_floor(sm):.4f} counts (negative-tail MAD)")
maps = _smooth_element_maps(maps, sigma=sigma, noise_floor=(args.smooth_floor == "noise"))
log(f"smoothed (sigma {sigma}, floor {args.smooth_floor})")

# 6. tile-overlap mask from the cache weight, decimated like the band maps
overlap_mask = None
if recipe.get("exclude_overlap", True):
    cache_dir = edsc.resolve_cache_dir(Path(args.dataset))
    w = np.lib.format.open_memmap(edsc._weight_path(cache_dir), mode="r")
    y0, y1, x0, x1 = (int(v) for v in roi)
    wds = np.asarray(w[y0:y1:seg_ds, x0:x1:seg_ds], dtype=np.float32)
    if wds.shape != shape:
        raise SystemExit(f"weight decimation {wds.shape} != band maps {shape}")
    overlap_mask = _compute_overlap_mask_from_weight(wds)
    del w, wds
    log(f"overlap mask: {overlap_mask.mean() * 100:.1f}% of pixels excluded")

# 7-9. features -> normalise -> cluster
X, valid, shp, el_names = _build_feature_matrix(
    maps, min_count_pct=float(recipe.get("min_count_pct", 0)),
    min_count_ratio=float(recipe.get("min_count_ratio", 0.1)), overlap_mask=overlap_mask)
del maps, overlap_mask
log(f"feature matrix {X.shape} ({valid.mean() * 100:.1f}% of pixels valid), elements {el_names}")
X_norm = _normalize_features(X, norm_method=recipe.get("norm_method", "row"))
seed_means = None
if args.seed_mask:
    from scipy.ndimage import binary_erosion as _erode
    sm_ = np.load(args.seed_mask)["mask"].astype(bool)
    if args.seed_erode > 0:
        sm_ = _erode(sm_, iterations=args.seed_erode)
    sel_ = sm_.reshape(-1)[valid.reshape(-1)]
    seed_means = X_norm[sel_].mean(axis=0, keepdims=True)
    log(f"seed component from {int(sel_.sum())} masked pixels (normalised feature mean {np.round(seed_means[0], 2).tolist()})")
del X
n_clusters = (("auto" if str(args.n_clusters).lower() == "auto" else int(args.n_clusters))
              if args.n_clusters else recipe.get("n_clusters", "auto"))
labels_flat, k_chosen, bic_scores, icl_scores, auto_method = _run_clustering(
    X_norm, n_clusters=n_clusters, k_max=int(recipe.get("k_max") or 12),
    min_k=int(recipe.get("min_k") or 2),
    pca_variance=float(args.pca_variance if args.pca_variance else recipe.get("pca_variance", 0.92)),
    posterior_min=float(recipe.get("reject_posterior_min", 0.0)),
    density_pct=float(recipe.get("reject_density_pct", 0.0)),
    fit_balance=(args.fit_balance or recipe.get("fit_balance", "uniform")), seed_means=seed_means,
    random_state=int(args.seed))
log(f"clustered: k = {k_chosen} (requested {n_clusters}, auto_method {auto_method})")
dominant = _cluster_dominant_elements(X_norm, labels_flat, el_names)
del X_norm

label_map = np.full(shp, -1, dtype=np.int16)
label_map[valid] = labels_flat.astype(np.int16)
rejected = valid & (label_map < 0)
if raw_zero is not None:
    label_map[raw_zero] = -1
    protect = rejected | raw_zero
else:
    protect = rejected
min_size = int(args.min_region if args.min_region is not None else recipe.get("min_region_size", 50))
if min_size > 0:
    label_map = _clean_label_map(label_map, min_size=min_size)
_fill_unlabeled(label_map, protect=protect)
del valid, rejected, protect, labels_flat
n_unc = int((label_map < 0).sum())
cids = sorted(int(c) for c in np.unique(label_map) if c >= 0)
px = {c: int((label_map == c).sum()) for c in cids}
log(f"label map done: {len(cids)} domains, {n_unc} unclassified px; pixels {px}")

# 10. contingency with the previous (sideband-net) map
old = np.load(args.old_labels)["labels"]
old_names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(args.old_legend))}
contingency = {}
for c in cids:
    m = label_map == c
    oc, on = np.unique(old[m], return_counts=True)
    contingency[c] = {old_names.get(int(o), str(o)): float(n) / px[c] for o, n in zip(oc, on) if int(o) >= 0}
    top = sorted(contingency[c].items(), key=lambda kv: -kv[1])[:3]
    log(f"  new {c} ({px[c] / label_map.size * 100:5.1f}%): " + ", ".join(f"{n} {f * 100:.0f}%" for n, f in top))
agree = float(sum(max(contingency[c].values()) * px[c] for c in cids) / sum(px.values()))
log(f"pixel agreement with the previous map under the majority mapping: {agree * 100:.1f}%")

# 11. per-domain composition (full masks and the requested erosion)
def domain_sums(mask):
    out = {}
    for el in elements:
        net = bg.net_count_map(el, get_band, plan=plan, scheme=scheme)
        out[el] = float(net[mask].sum())
    return out


masks_full = {c: label_map == c for c in cids}
masks_comp = {}
for c in cids:
    m = masks_full[c]
    if args.comp_erosion > 0:
        me = binary_erosion(m, iterations=args.comp_erosion)
        m = me if me.any() else m
    masks_comp[c] = m
sums_full, sums_comp = {}, {}
for el in elements:
    net = bg.net_count_map(el, get_band, plan=plan, scheme=scheme)
    for c in cids:
        sums_full.setdefault(c, {})[el] = float(net[masks_full[c]].sum())
        sums_comp.setdefault(c, {})[el] = float(net[masks_comp[c]].sum())
    del net
log("per-domain net sums done")


def fractions(sums):
    clipped = {el: max(0.0, v) for el, v in sums.items()}
    tot = sum(clipped.values())
    return {el: v / tot for el, v in clipped.items()} if tot > 0 else {}


comp_wt = {c: eq.apply_calibration_fractions(fractions(sums_comp[c]), k) for c in cids}
n_comp = {c: int(masks_comp[c].sum()) for c in cids}

# 12. mineral ID + naming
coating = frozenset({args.coating}) if args.coating else frozenset()
matches = {c: mi.identify(comp_wt[c], quantified=True, fe3_fraction=args.fe3, top_n=5,
                          coating_elements=coating) for c in cids}
names, assignments, mixed = {}, {}, []
LIB_NAMES = {e.name for e in mi.DEFAULT_LIBRARY.values()}
for c in cids:
    top = matches[c][0] if matches[c] else None
    auto_name = top.name if top else f"Phase {c + 1}"
    maj_name, maj_frac = max(contingency[c].items(), key=lambda kv: kv[1]) if contingency[c] else ("", 0.0)
    # A component is a clean mineral only when ONE mineral of the previous map
    # holds the majority of its pixels. Everything else is a mixed domain and is
    # unmixed: a component of matrix pixels, also when it straddles the two
    # matrix types ("Mica-rich 49 %, Quartz-rich 46 %", neither a majority on
    # its own), and a component in which no phase reaches the majority. Without
    # this such components fell through to their automatic match, a junk
    # identification at a few percent confidence ("Calcite 5 %") that either
    # dropped them from the modal budget or merged them into a clean phase.
    min_frac = max((f for n, f in contingency[c].items() if "matrix" not in n.lower()), default=0.0)
    name, override = auto_name, None
    if top is not None and top.name.startswith("Al2SiO5"):
        name = "Sillimanite"
    elif contingency[c] and min_frac <= args.min_majority:
        name = "Mixed matrix"            # renamed after unmixing below
        mixed.append(c)
    elif (maj_frac > args.min_majority and maj_name in LIB_NAMES and auto_name != maj_name
          and (top is None or (top.confidence_pct or 0) < 75)):
        override = maj_name          # weak auto-ID: inherit the July decision
        name = maj_name
    names[c] = name
    cands = [{"name": m.name, "abbrev": m.abbrev, "confidence_pct": m.confidence_pct,
              "confidence": m.confidence} for m in matches[c][:5]]
    if override:
        entry = next(e for e in mi.DEFAULT_LIBRARY.values() if e.name == override)
        assignments[c] = {"abbrev": entry.abbrev, "name": entry.name, "group": entry.group,
                          "score": 1.0, "method": "manual", "confidence": "high",
                          "confidence_pct": 100.0, "indices": {}, "flags": ["manual_override"],
                          "thermo_abbrev": mi.THERMO_ABBREV.get(entry.abbrev), "candidates": cands}
    elif top:
        assignments[c] = {"abbrev": top.abbrev, "name": top.name, "group": top.group,
                          "score": round(float(top.score), 4), "method": top.method,
                          "confidence": top.confidence, "confidence_pct": top.confidence_pct,
                          "indices": {kk: round(float(v), 4) for kk, v in top.indices.items()},
                          "flags": list(top.flags), "thermo_abbrev": top.thermo_abbrev,
                          "candidates": cands}
    log(f"  domain {c} -> {name!r} (auto {auto_name} {top.confidence_pct if top else 0:.0f}%, "
        f"majority {maj_name} {maj_frac * 100:.0f}%" + (f", OVERRIDE {override}" if override else "") + ")  "
        + " ".join(f"{el} {v:.1f}" for el, v in sorted(comp_wt[c].items(), key=lambda kv: -kv[1]) if v > 0.3))

# 12b. merge duplicated components by name (the dialog's Merge selected):
#      mixed domains are merged into one "Mixed matrix" first and split by
#      type after unmixing below. Sums/pixels are additive, so the merged
#      domain's composition and unmix inputs follow without re-reading.
if args.merge_same_name:
    # mixed components: type them by their dominant unmix endmember first so
    # quartz-rich and mica-rich matrix stay separate domains
    _clean = [c for c in cids if c not in mixed]
    _means = {c: {el: sums_full[c][el] / px[c] for el in elements} for c in cids}
    for src in list(mixed):
        fr, _r = _unmix_domain(_means, src, _clean)
        dom = max(fr.items(), key=lambda kv: kv[1])[0] if fr else None
        names[src] = "Quartz-rich matrix" if (dom is not None and names[dom] == "Quartz") else "Mica-rich matrix"
    by_name = {}
    for c in cids:
        by_name.setdefault(names[c], []).append(c)
    merged_note = {}
    for nm, group in by_name.items():
        if len(group) < 2:
            continue
        keep, others = group[0], group[1:]
        for o in others:
            label_map[label_map == o] = keep
            for el in elements:
                sums_full[keep][el] += sums_full[o][el]
                sums_comp[keep][el] += sums_comp[o][el]
            px[keep] += px.pop(o); n_comp[keep] += n_comp.pop(o)
            masks_full[keep] |= masks_full.pop(o); masks_comp[keep] |= masks_comp.pop(o)
            for d in (names, comp_wt, matches, assignments, contingency):
                d.pop(o, None)
            if o in mixed:
                mixed.remove(o)
        merged_note[nm] = group
        log(f"  merged {group} -> {keep} ({nm!r}, {100 * px[keep] / label_map.size:.1f} %)")
    cids = sorted(px)
    # recompute the merged domains' composition / ID / assignment on the pooled sums
    for nm, group in merged_note.items():
        keep = group[0]
        comp_wt[keep] = eq.apply_calibration_fractions(fractions(sums_comp[keep]), k)
        matches[keep] = mi.identify(comp_wt[keep], quantified=True, fe3_fraction=args.fe3, top_n=5, coating_elements=coating)
        top = matches[keep][0] if matches[keep] else None
        cands = [{"name": m_.name, "abbrev": m_.abbrev, "confidence_pct": m_.confidence_pct, "confidence": m_.confidence}
                 for m_ in matches[keep][:5]]
        flags_extra = [f"merged_from_{'_'.join(map(str, group))}"]
        if top is not None and top.name == nm:
            # the pooled composition identifies as the merged name: auto assignment
            assignments[keep] = {"abbrev": top.abbrev, "name": top.name, "group": top.group,
                                 "score": round(float(top.score), 4), "method": top.method,
                                 "confidence": top.confidence, "confidence_pct": top.confidence_pct,
                                 "indices": {kk: round(float(v), 4) for kk, v in top.indices.items()},
                                 "flags": list(top.flags) + flags_extra, "thermo_abbrev": top.thermo_abbrev,
                                 "candidates": cands}
        elif nm in LIB_NAMES:
            entry = next(e for e in mi.DEFAULT_LIBRARY.values() if e.name == nm)
            assignments[keep] = {"abbrev": entry.abbrev, "name": entry.name, "group": entry.group, "score": 1.0,
                                 "method": "manual", "confidence": "high", "confidence_pct": 100.0, "indices": {},
                                 "flags": ["manual_override"] + flags_extra,
                                 "thermo_abbrev": mi.THERMO_ABBREV.get(entry.abbrev), "candidates": cands}
        elif keep in assignments:
            assignments[keep]["flags"] = list(assignments[keep].get("flags", [])) + flags_extra
        contingency[keep] = {}
        m = masks_full[keep]
        oc, on = np.unique(old[m], return_counts=True)
        contingency[keep] = {old_names.get(int(o), str(o)): float(n) / px[keep] for o, n in zip(oc, on) if int(o) >= 0}

# 13. unmix every mixed domain on raw mean signed counts per pixel (full
#     masks) onto the CLEAN domains only; name it by its dominant endmember
unmix = {}
clean = [c for c in cids if c not in mixed]
means = {c: {el: sums_full[c][el] / px[c] for el in elements} for c in cids}
for src in mixed:
    frac, resid = _unmix_domain(means, src, clean)
    unmix[src] = {"fractions": {int(e): float(f) for e, f in frac.items()}, "residual": float(resid)}
    dom = max(frac.items(), key=lambda kv: kv[1])[0] if frac else None
    names[src] = "Quartz-rich matrix" if (dom is not None and names[dom] == "Quartz") else "Mica-rich matrix"
    if sum(1 for m in mixed if names[m] == names[src]) > 1 and not args.merge_same_name:
        names[src] += f" {src}"
    log(f"unmix domain {src} -> {names[src]!r}: " + " ".join(f"{names[e]} {f:.3f}" for e, f in frac.items())
        + f" (sum {sum(frac.values()):.3f}, residual {resid:.4f}; dominant {names[dom] if dom is not None else '-'})")

# 14. exports (dialog layout)
import imageio.v2 as iio
np.savez_compressed(OUT / "phase_map_labels.npz", labels=label_map)
iio.imwrite(str(OUT / "phase_map_combined.png"), _label_map_to_rgba(label_map, max(cids) + 1))
with open(OUT / "phase_legend.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cluster_id", "name", "pixel_count"])
    for c in cids:
        w.writerow([c, names[c], px[c]])
with open(OUT / "phase_minerals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cluster_id", "name", "pixel_count", "top1_mineral", "top1_abbrev", "top1_score",
                "top1_confidence", "top1_method", "Mg#", "An", "top2_mineral", "top2_score",
                "top3_mineral", "top3_score", "flags"])
    for c in cids:
        a = assignments.get(c)
        if not a:
            w.writerow([c, names[c], px[c]] + [""] * 12); continue
        ms = matches[c]; ix = a["indices"]; cands = a["candidates"]
        sc = lambda i: f"{ms[i].score:.4f}" if i < len(ms) else ""
        w.writerow([c, names[c], px[c], a["name"], a["abbrev"], f"{a['score']:.4f}", a["confidence"],
                    a["method"], f"{ix['Mg#']:.4f}" if "Mg#" in ix else "", f"{ix['An']:.4f}" if "An" in ix else "",
                    cands[1]["name"] if len(cands) > 1 else "", sc(1),
                    cands[2]["name"] if len(cands) > 2 else "", sc(2), ";".join(a["flags"])])
el_order = sorted(k, key=lambda e: -np.mean([comp_wt[c].get(e, 0.0) for c in cids]))
with open(OUT / "phase_composition.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cluster_id", "name", "pixel_count", "units"] + el_order)
    for c in cids:
        w.writerow([c, names[c], n_comp[c], "wt%"] + [f"{comp_wt[c].get(e, 0.0):.4f}" for e in el_order])
seg_params = dict(recipe_in)
seg_params.update({
    "elements_used": list(el_names), "n_clusters": n_clusters, "k_chosen": int(k_chosen),
    "pca_variance": float(args.pca_variance if args.pca_variance else recipe.get("pca_variance", 0.92)),
    "element_filter": el_filter, "max_zero_pct": max_zero_pct, "smooth_floor": args.smooth_floor,
    "smooth_sigma": sigma, "min_region_size": min_size,
    "fit_balance": (args.fit_balance or recipe.get("fit_balance", "uniform")),
    "random_state": int(args.seed),
    "seed_mask": args.seed_mask, "seed_erode": args.seed_erode, "merge_same_name": bool(args.merge_same_name),
    "auto_method": auto_method, "background": scheme,
    "calibration": {"source": Path(args.calibration).name, "background": cal.background,
                    "n_tiles": cal.n_tiles, "k_factors": {e: kf.k for e, kf in cal.k_factors.items()}},
    "fe3_fraction": args.fe3, "coating_elements": [args.coating] if args.coating else [],
    "provenance": "segment_worked_example.py (headless replay of the viewer pipeline, 2026-09-16)",
})
if args.comp_erosion:
    seg_params["comp_erosion_px"] = int(args.comp_erosion)
if unmix:
    seg_params["unmix"] = {str(s): {"fractions": {str(e): round(f, 6) for e, f in d["fractions"].items()},
                                    "residual": round(d["residual"], 6)} for s, d in unmix.items()}
if n_unc:
    seg_params["unclassified_px"] = n_unc
    seg_params["unclassified_frac"] = round(n_unc / label_map.size, 6)
json.dump(seg_params, open(OUT / "segmentation_params.json", "w"), indent=2, default=str)
report = {
    "sparsity": [{"element": e, "zero_frac_pct": zf, "pooled_counts": p, "excluded": d} for e, zf, p, d in sparsity_report],
    "bic": bic_scores, "icl": icl_scores, "k_chosen": int(k_chosen), "dominant": dominant,
    "pixels": px, "n_unclassified": n_unc, "contingency_vs_previous": contingency,
    "pixel_agreement_majority": agree, "names": names, "assignments": assignments,
    "composition_wt": comp_wt, "composition_n_px": n_comp, "unmix": unmix,
    "elapsed_s": round(time.time() - t0),
}
json.dump(report, open(OUT / "segmentation_report.json", "w"), indent=1, default=str)
log(f"wrote exports to {OUT}")
