"""Full-resolution phase segmentation of the whole worked-example section.

The staged map is segmented on the 1/5 grid (2.5 um/px, 53 M px) because the
viewer's memory guard decimates a whole-section ROI. At that pitch, with the
recipe's 2.5 px smoothing, the sillimanite cores of the muscovite nodules
(50-300 um) are sub-resolution mixtures and never get their own GMM
component. This script segments the section at the native 0.5 um pitch
(1.33 G px) with the SAME feature pipeline (line-aware signed nets from the
mosaic cache, Gaussian smoothing, noise floor, row-normalised log features,
robust z-score, PCA, GMM), fitted on a random block sample and applied tile
by tile from the cache, with the label map kept on disk.

Outputs
  <export-dir>/phase_map_fullres.npy   int8 label map at native pitch (1.3 GB,
                                       outside the repository)
  phase_map_labels.npz                 the same map decimated to the 1/5 grid
                                       (labels[::5, ::5]; the staged file every
                                       downstream script reads)
  phase_map_inset.{npz,json}           the Fig 4b window cut from the native map
  phase_legend.csv / phase_minerals.csv / phase_composition.csv /
  segmentation_params.json / segmentation_report.json   as segment_worked_example.py
Naming follows segment_worked_example.py (July decisions by pixel majority,
mixed domains unmixed onto clean ones, Al2SiO5 -> --als-name).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
import eds_background as bg                                       # noqa: E402
import eds_cache as edsc                                          # noqa: E402
import eds_quantify as eq                                         # noqa: E402
import mineral_id as mi                                           # noqa: E402
from viewer_segment_helpers import (_noise_floor, _clean_label_map, _fill_unlabeled,   # noqa: E402
                                    _compute_overlap_mask_from_weight, _label_map_to_rgba)
from viewer_thermo_helpers import _unmix_domain                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=r"D:/Actief/C23S1A_Quant")
ap.add_argument("--export-dir", default=r"D:/Actief/C23S1A_Quant/export")
ap.add_argument("--out-dir", default=str(HERE))
ap.add_argument("--calibration", default=str(HERE / "eds_calibration.json"))
ap.add_argument("--old-labels", default=str(REPO / "_scratch/pre_background_fix_staged/phase_map_labels.npz"))
ap.add_argument("--old-legend", default=str(REPO / "_scratch/pre_background_fix_staged/phase_legend.csv"))
ap.add_argument("--elements", nargs="+", default=["Al", "Ca", "Fe", "K", "Mg", "Na", "O", "Si"],
                help="feature elements (the adopted 1/5 recipe's list)")
ap.add_argument("--sigma", type=float, default=2.0, help="smoothing sigma in map pixels (at --ds)")
ap.add_argument("--k", default="auto", help="components: an integer, or auto (ICL sweep up to --k-max)")
ap.add_argument("--k-max", type=int, default=15)
ap.add_argument("--ds", type=int, default=2, help="nearest-pixel decimation of the native grid (1 = native 0.5 um)")
ap.add_argument("--min-region", type=int, default=190, help="min region size in map pixels (30 px at 1/5 = 750 native = 190 at ds 2)")
ap.add_argument("--pca-variance", type=float, default=0.92)
ap.add_argument("--min-count-pct", type=float, default=5.0)
ap.add_argument("--min-count-ratio", type=float, default=0.1)
ap.add_argument("--tile", type=int, default=2048)
ap.add_argument("--margin", type=int, default=48)
ap.add_argument("--sample-blocks", type=int, default=600)
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--fit-sample", type=int, default=30000)
ap.add_argument("--limit-tiles", type=int, default=0, help="smoke test: process only the first N tiles")
ap.add_argument("--fe3", type=float, default=0.1)
ap.add_argument("--als-name", default="Sillimanite")
ap.add_argument("--min-majority", type=float, default=0.5)
ap.add_argument("--seg-ds", type=int, default=5)
args = ap.parse_args()
t0 = time.time()
OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
EXP = Path(args.export_dir); EXP.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)


# ---------------------------------------------------------------- cache
cache_dir = edsc.resolve_cache_dir(Path(args.dataset))
man = edsc.load_manifest(cache_dir)
H0, W0 = int(man["H"]), int(man["W"])
DS = int(args.ds)
H, W = (H0 + DS - 1) // DS, (W0 + DS - 1) // DS          # map grid (nearest decimation)
cached = list(man["elements"])
plan = bg.background_plan(cached, width_eV=float(man.get("width_eV") or 150.0),
                          sideband_mult=float(man.get("sideband_mult") or 1.5))
feat_els = [e for e in args.elements if e in cached]
comp_els = [e for e in cached if e != "C"]                       # composition / ID / unmix
need = sorted(set(feat_els) | set(comp_els) | set(bg.anchor_elements(plan, feat_els + comp_els)))
weight = np.lib.format.open_memmap(edsc._weight_path(cache_dir), mode="r")
bands = {e: {b: np.lib.format.open_memmap(edsc._element_path(cache_dir, e, b), mode="r") for b in edsc.BANDS}
         for e in need}
log(f"cache {H0}x{W0} -> map {H}x{W} at ds {DS}; features {feat_els}; composition {comp_els}; reading {need}")
print(bg.describe_plan(plan), flush=True)
cal = eq.Calibration.from_dict(json.load(open(args.calibration))["calibration"])
k_fac = cal.k()


def read_nets(y0, y1, x0, x1, elements):
    """Signed line-aware nets (weight-normalised, blend 'mean') for a window."""
    sy, sx = slice(y0 * DS, y1 * DS, DS), slice(x0 * DS, x1 * DS, DS)     # map coords -> native stride
    w = np.asarray(weight[sy, sx], dtype=np.float32)
    invw = np.zeros_like(w); np.divide(1.0, w, out=invw, where=w > 0)

    def get_band(e, b):
        arr = bands.get(e)
        return None if arr is None else np.asarray(arr[b][sy, sx], dtype=np.float32) * invw
    out = {e: bg.net_count_map(e, get_band, plan=plan) for e in elements}
    return out, w


def features_from_nets(nets, floors):
    """Smoothed + floored maps -> (H*W, n) feature rows in feat_els order, plus totals."""
    cols = []
    for e in feat_els:
        sm = gaussian_filter(nets[e], sigma=args.sigma) if args.sigma > 0 else nets[e]
        cols.append(np.maximum(sm, 0.0) + np.float32(floors[e]))
    X = np.stack(cols, axis=-1).reshape(-1, len(feat_els))
    return X


def normalise(X, med, mad):
    rs = X.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    Z = np.log1p(X / rs * 1000.0)
    return ((Z - med) / np.where(mad > 0, mad, 1.0)).astype(np.float32)


# ---------------------------------------------------------------- 1. sample: noise floors, normalisation, PCA, GMM
rng = np.random.default_rng(42)
B, M = args.block, int(4 * args.sigma) + 2
ys = rng.integers(M, H - B - M, args.sample_blocks); xs = rng.integers(M, W - B - M, args.sample_blocks)
neg = {e: [] for e in feat_els}; rows = []; totals = []
for i, (y, x) in enumerate(zip(ys, xs)):
    nets, w = read_nets(y - M, y + B + M, x - M, x + B + M, feat_els)
    if (w <= 0).mean() > 0.5:
        continue
    for e in feat_els:
        sm = gaussian_filter(nets[e], sigma=args.sigma) if args.sigma > 0 else nets[e]
        neg[e].append(sm[M:-M, M:-M][sm[M:-M, M:-M] < 0])
    inner = {e: nets[e][M:-M, M:-M] for e in feat_els}
    rows.append((inner, w[M:-M, M:-M]))
    if (i + 1) % 100 == 0:
        log(f"  sample blocks {i + 1}/{len(ys)}")
floors = {e: (float(1.4826 * np.median(np.abs(np.concatenate(neg[e])))) if sum(a.size for a in neg[e]) >= 100 else 0.0)
          for e in feat_els}
log("noise floors: " + " ".join(f"{e} {v:.3f}" for e, v in floors.items()))
X_all = []
for inner, w in rows:
    X = features_from_nets(inner, floors)
    ov = _compute_overlap_mask_from_weight(w).reshape(-1)
    X_all.append(X[~ov])
X_all = np.concatenate(X_all)
tot = X_all.sum(axis=1)
valid = np.isfinite(tot) & (tot > 0)
thr = max(float(np.percentile(tot[valid], args.min_count_pct)) if args.min_count_pct > 0 else 0.0,
          args.min_count_ratio * float(np.median(tot[valid])) if args.min_count_ratio > 0 else 0.0, 1.0)
X_all = X_all[valid & (tot >= thr)]
log(f"sample features {X_all.shape}; low-count threshold {thr:.2f} counts")
rs = X_all.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
Z = np.log1p(X_all / rs * 1000.0)
med = np.median(Z, axis=0); mad = np.median(np.abs(Z - med), axis=0)
Zn = normalise(X_all, med, mad)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
fit = Zn[rng.choice(len(Zn), min(args.fit_sample, len(Zn)), replace=False)]
pca_full = PCA(random_state=42).fit(fit.astype(np.float64))
cum = pca_full.explained_variance_ratio_.cumsum()
n_pca = max(2, min(int(np.searchsorted(cum, args.pca_variance)) + 1, fit.shape[1] - 1))
pca = PCA(n_components=n_pca, random_state=42)
fit_r = pca.fit_transform(fit.astype(np.float64)).astype(np.float32)
if str(args.k).lower() == "auto":
    from viewer_segment_helpers import _select_k_auto
    k_best, bic_scores, icl_scores, gmm, k_method = _select_k_auto(fit_r, args.k_max, None)
    K = int(k_best)
else:
    K = int(args.k); bic_scores = icl_scores = None; k_method = None
    gmm = GaussianMixture(n_components=K, n_init=5, max_iter=200, random_state=42, reg_covar=1e-4).fit(fit_r)
log(f"model: PCA {fit.shape[1]}D -> {n_pca}D ({cum[n_pca - 1] * 100:.1f}%), GMM k={K} ({k_method or 'fixed'})")

# ---------------------------------------------------------------- 2. predict tile by tile
lab_path = EXP / (f"phase_map_ds{DS}.npy" if DS > 1 else "phase_map_fullres.npy")
labels = np.lib.format.open_memmap(lab_path, mode="w+", dtype=np.int8, shape=(H, W))
T, Mg_ = args.tile, args.margin
tiles = [(y, x) for y in range(0, H, T) for x in range(0, W, T)]
if args.limit_tiles:
    tiles = tiles[:args.limit_tiles]
sums = {}   # cid -> {el: signed net sum}; px counts from the map afterwards
n_unc = 0
for i, (ty, tx) in enumerate(tiles):
    y0, y1 = max(0, ty - Mg_), min(H, ty + T + Mg_); x0, x1 = max(0, tx - Mg_), min(W, tx + T + Mg_)
    nets, w = read_nets(y0, y1, x0, x1, need)
    if not (w > 0).any():
        labels[ty:min(H, ty + T), tx:min(W, tx + T)] = -1
        continue
    X = features_from_nets(nets, floors)
    tot = X.sum(axis=1)
    ov = _compute_overlap_mask_from_weight(w).reshape(-1)
    ok = np.isfinite(tot) & (tot >= thr) & ~ov
    lab = np.full(X.shape[0], -1, np.int16)
    if ok.any():
        Zt = normalise(X[ok], med, mad)
        lab[ok] = gmm.predict(pca.transform(Zt.astype(np.float64)).astype(np.float32)).astype(np.int16)
    lab = lab.reshape(y1 - y0, x1 - x0)
    lab = _clean_label_map(lab, min_size=args.min_region)
    _fill_unlabeled(lab)
    lab[w <= 0] = -1
    iy0, ix0 = ty - y0, tx - x0
    core = lab[iy0:iy0 + min(T, H - ty), ix0:ix0 + min(T, W - tx)]
    labels[ty:ty + core.shape[0], tx:tx + core.shape[1]] = core.astype(np.int8)
    # per-domain signed net sums on the interior (un-smoothed nets, composition elements)
    for c in np.unique(core):
        if c < 0:
            continue
        m = core == c
        acc = sums.setdefault(int(c), {e: 0.0 for e in comp_els})
        for e in comp_els:
            acc[e] += float(nets[e][iy0:iy0 + core.shape[0], ix0:ix0 + core.shape[1]][m].sum())
    n_unc += int((core < 0).sum())
    if (i + 1) % 20 == 0 or i + 1 == len(tiles):
        log(f"  tile {i + 1}/{len(tiles)}")
labels.flush()
log(f"map at ds {DS} written: {lab_path}; unclassified {n_unc} px")

# ---------------------------------------------------------------- 3. classification
cids = sorted(sums)
px = {c: int((labels == c).sum()) for c in cids} if not args.limit_tiles else {c: 0 for c in cids}
if args.limit_tiles:
    for c in cids:
        px[c] = sum(int((labels[ty:ty + T, tx:tx + T] == c).sum()) for ty, tx in tiles)
tot_px = sum(px.values())
old = np.load(args.old_labels)["labels"]
old_names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(args.old_legend))}
ri = (np.arange(0, H0, args.seg_ds) // DS).clip(0, H - 1); ci = (np.arange(0, W0, args.seg_ds) // DS).clip(0, W - 1)
dec = np.asarray(labels)[np.ix_(ri, ci)]
dec = dec[:old.shape[0], :old.shape[1]]
contingency, names, assignments, mixed = {}, {}, {}, []
LIB = {e.name for e in mi.DEFAULT_LIBRARY.values()}
for c in cids:
    m = dec == c
    oc, on = np.unique(old[:dec.shape[0], :dec.shape[1]][m], return_counts=True)
    n_m = max(1, int(m.sum()))
    contingency[c] = {old_names.get(int(o), str(o)): float(n) / n_m for o, n in zip(oc, on) if int(o) >= 0}
    s = {e: max(0.0, v) for e, v in sums[c].items()}; t = sum(s.values())
    wt = eq.apply_calibration_fractions({e: v / t for e, v in s.items()}, k_fac)
    ids = mi.identify(wt, quantified=True, fe3_fraction=args.fe3, top_n=5, coating_elements=frozenset({"C"}))
    top = ids[0] if ids else None
    auto = top.name if top else f"Phase {c + 1}"
    maj, majf = max(contingency[c].items(), key=lambda kv: kv[1]) if contingency[c] else ("", 0.0)
    name, override = auto, None
    if top and top.name.startswith("Al2SiO5"):
        name = args.als_name; override = None
    elif majf > args.min_majority and "matrix" in maj.lower():
        name = "Mixed matrix"; mixed.append(c)
    elif majf > args.min_majority and maj in LIB and auto != maj:
        override = maj; name = maj
    names[c] = name
    cands = [{"name": x.name, "abbrev": x.abbrev, "confidence_pct": x.confidence_pct, "confidence": x.confidence} for x in ids[:5]]
    if override:
        entry = next(e for e in mi.DEFAULT_LIBRARY.values() if e.name == override)
        assignments[c] = {"abbrev": entry.abbrev, "name": entry.name, "group": entry.group, "score": 1.0, "method": "manual",
                          "confidence": "high", "confidence_pct": 100.0, "indices": {}, "flags": ["manual_override"],
                          "thermo_abbrev": mi.THERMO_ABBREV.get(entry.abbrev), "candidates": cands}
    elif top:
        assignments[c] = {"abbrev": top.abbrev, "name": top.name, "group": top.group, "score": round(float(top.score), 4),
                          "method": top.method, "confidence": top.confidence, "confidence_pct": top.confidence_pct,
                          "indices": {kk: round(float(v), 4) for kk, v in top.indices.items()}, "flags": list(top.flags),
                          "thermo_abbrev": ("sill" if name == args.als_name else top.thermo_abbrev), "candidates": cands}
    log(f"  domain {c} ({100 * px[c] / max(1, tot_px):5.1f}%) -> {name!r} (auto {auto} {top.confidence_pct if top else 0:.0f}%, "
        f"majority {maj} {majf * 100:.0f}%" + (f", OVERRIDE {override}" if override else "") + ")  "
        + " ".join(f"{e} {v:.1f}" for e, v in sorted(wt.items(), key=lambda kv: -kv[1]) if v > 0.3))
    sums[c]["_wt"] = wt
unmix = {}
clean = [c for c in cids if c not in mixed]
means = {c: {e: sums[c][e] / max(1, px[c]) for e in comp_els} for c in cids}
for src in mixed:
    frac, resid = _unmix_domain(means, src, clean)
    unmix[src] = {"fractions": {int(e): float(f) for e, f in frac.items()}, "residual": float(resid)}
    dom = max(frac.items(), key=lambda kv: kv[1])[0] if frac else None
    names[src] = "Quartz-rich matrix" if (dom is not None and names[dom] == "Quartz") else "Mica-rich matrix"
    if sum(1 for m_ in mixed if names[m_] == names[src]) > 1:
        names[src] += f" {src}"
    log(f"unmix {src} -> {names[src]!r}: " + " ".join(f"{names[e]} {f:.3f}" for e, f in frac.items()) + f" (sum {sum(frac.values()):.3f}, residual {resid:.4f})")
agree = float(sum(max(contingency[c].values(), default=0) * (dec == c).sum() for c in cids) / max(1, (dec >= 0).sum()))
log(f"pixel agreement with the July map (1/{args.seg_ds} grid, majority mapping): {agree * 100:.1f}%")

# ---------------------------------------------------------------- 4. exports
np.savez_compressed(OUT / "phase_map_labels.npz", labels=dec.astype(np.int16))
import imageio.v2 as iio
iio.imwrite(str(OUT / "phase_map_combined.png"), _label_map_to_rgba(dec.astype(np.int16), max(cids) + 1))
# (the Fig 4b inset stays a native-pitch re-segmentation of the nodule window: fig_segmentation_inset.py)
with open(OUT / "phase_legend.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cluster_id", "name", "pixel_count"])
    for c in cids:
        w.writerow([c, names[c], px[c]])
with open(OUT / "phase_minerals.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cluster_id", "name", "pixel_count", "top1_mineral", "top1_abbrev", "top1_score", "top1_confidence",
                "top1_method", "Mg#", "An", "top2_mineral", "top2_score", "top3_mineral", "top3_score", "flags"])
    for c in cids:
        a = assignments.get(c)
        if not a:
            w.writerow([c, names[c], px[c]] + [""] * 12); continue
        cands = a["candidates"]; ix = a["indices"]
        w.writerow([c, names[c], px[c], a["name"], a["abbrev"], f"{a['score']:.4f}", a["confidence"], a["method"],
                    f"{ix['Mg#']:.4f}" if "Mg#" in ix else "", f"{ix['An']:.4f}" if "An" in ix else "",
                    cands[1]["name"] if len(cands) > 1 else "", "", cands[2]["name"] if len(cands) > 2 else "", "",
                    ";".join(a["flags"])])
el_order = sorted(k_fac, key=lambda e: -np.mean([sums[c]["_wt"].get(e, 0.0) for c in cids]))
with open(OUT / "phase_composition.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cluster_id", "name", "pixel_count", "units"] + el_order)
    for c in cids:
        w.writerow([c, names[c], px[c], "wt%"] + [f"{sums[c]['_wt'].get(e, 0.0):.4f}" for e in el_order])
seg_params = {
    "source": "streamed_cache", "seg_ds": DS, "roi": [0, H0, 0, W0], "elements_used": feat_els, "element_filter": feat_els,
    "n_clusters": (K if str(args.k).lower() != "auto" else "auto"), "k_chosen": K, "auto_method": k_method,
    "k_max": args.k_max, "bic": bic_scores, "icl": icl_scores, "pca_variance": args.pca_variance,
    "smooth_sigma": args.sigma, "min_region_size": args.min_region, "norm_method": "row",
    "min_count_pct": args.min_count_pct, "min_count_ratio": args.min_count_ratio, "max_zero_pct": 100.0,
    "exclude_overlap": True, "reject_posterior_min": 0.0, "reject_density_pct": 0.0, "protect_background": True,
    "background": bg.default_scheme(), "noise_floor": True, "noise_floors": floors,
    "fit": {"sample_blocks": args.sample_blocks, "block": args.block, "fit_sample": args.fit_sample, "seed": 42,
            "low_count_threshold": thr, "n_pca": int(n_pca)},
    "tile": args.tile, "margin": args.margin, "fullres_labels": str(lab_path),
    "staged_labels_note": f"phase_map_labels.npz = the ds-{DS} map sampled at the 1/{args.seg_ds} grid (nearest); legend pixel counts are at ds {DS}",
    "calibration": {"source": Path(args.calibration).name, "background": cal.background, "n_tiles": cal.n_tiles},
    "fe3_fraction": args.fe3, "coating_elements": ["C"],
    "unmix": {str(s): {"fractions": {str(e): round(f, 6) for e, f in d["fractions"].items()}, "residual": round(d["residual"], 6)}
              for s, d in unmix.items()},
    "unclassified_px": n_unc, "provenance": f"segment_fullres.py (streamed GMM at ds {DS}, 2026-09-17)",
}
json.dump(seg_params, open(OUT / "segmentation_params.json", "w"), indent=2, default=str)
json.dump({"pixels": px, "names": names, "contingency_vs_july": contingency, "pixel_agreement_majority": agree,
           "assignments": assignments, "composition_wt": {c: sums[c]["_wt"] for c in cids}, "unmix": unmix,
           "elapsed_s": round(time.time() - t0)}, open(OUT / "segmentation_report.json", "w"), indent=1, default=str)
log(f"wrote exports to {OUT}")
