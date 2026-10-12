"""Second-stage clustering of ONE segmentation domain (Section 4.2).

The global GMM allots its components to the large populations; a small phase
that is compositionally distinct but only a percent of the map (the
sillimanite cores of the muscovite nodules, 1.5 % of C23S1A) ends up inside
a mixed domain at every k tried. This re-clusters the pixels of one domain
on the SAME features the global run used (line-aware signed nets, smoothed,
noise floor, row-normalised log features, PCA), with a small k, identifies
each sub-cluster from its calibrated composition, and, when a sub-cluster
matches --target (a library mineral name, default the Al2SiO5 group), splits
it off as a new domain appended to the legend. All other exports are
rewritten consistently (labels, legend, minerals, composition, params with
a "split" provenance block). Diagnostic by default (--out-dir); pass
--out-dir . to overwrite the staged files.
"""
import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
import eds_background as bg                                       # noqa: E402
import eds_quantify as eq                                         # noqa: E402
import mineral_id as mi                                           # noqa: E402
from viewer_segment_helpers import (_smooth_element_maps, _build_feature_matrix,   # noqa: E402
                                    _normalize_features, _run_clustering, _clean_label_map,
                                    _label_map_to_rgba)

ap = argparse.ArgumentParser()
ap.add_argument("--session-npz", default=r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.npz")
ap.add_argument("--data-dir", default=str(HERE), help="folder with the staged exports to split")
ap.add_argument("--out-dir", default=str(REPO / "_scratch/background_fix/seg_split"))
ap.add_argument("--domain", nargs="+", default=["Mica-rich matrix"], help="name(s) of the domain(s) to re-cluster together")
ap.add_argument("--k", default="3", help="sub-clusters: an integer or auto (ICL up to --k-max)")
ap.add_argument("--k-max", type=int, default=10)
ap.add_argument("--target", default="Al2SiO5", help="library mineral name prefix to split off")
ap.add_argument("--new-name", default="Sillimanite")
ap.add_argument("--min-confidence", default="low")
args = ap.parse_args()
t0 = time.time()
DATA = Path(args.data_dir); OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.time() - t0:5.0f} s] {m}", flush=True)


labels = np.load(DATA / "phase_map_labels.npz")["labels"].copy()
legend = list(csv.DictReader(open(DATA / "phase_legend.csv")))
names = {int(r["cluster_id"]): r["name"].strip() for r in legend}
seg = json.loads((DATA / "segmentation_params.json").read_text())
cal = eq.Calibration.from_dict(json.load(open(DATA / "eds_calibration.json"))["calibration"])
k_fac = cal.k()
srcs = [c for c, n in names.items() if n in args.domain]
assert srcs, f"none of {args.domain} in the legend {names}"
src = srcs[0]
mask = np.isin(labels, srcs)
log(f"domains {srcs} {args.domain}: {mask.sum()} px ({100 * mask.mean():.1f} % of the map)")

# features exactly as the global run: signed nets -> smooth (+ noise floor) -> row/log -> z-score
z = np.load(args.session_npz)
elements = sorted({k.split("__")[0] for k in z.keys() if "__" in k})
plan = bg.background_plan(elements)
use = seg.get("elements_used") or elements
maps = {}
for el in use:
    maps[el] = bg.net_count_map(el, lambda e, b: z[f"{e}__{b}"] if f"{e}__{b}" in z else None, plan=plan)
raw = {el: maps[el] for el in use}
sm = _smooth_element_maps(maps, sigma=float(seg.get("smooth_sigma", 2.5)), noise_floor=bool(seg.get("noise_floor", True)))
del maps
X, valid, shape, el_names = _build_feature_matrix(sm, min_count_pct=0.0, min_count_ratio=0.0, overlap_mask=None)
del sm
sel = mask.reshape(-1)[valid.reshape(-1)] if valid.shape == mask.shape else mask.reshape(-1)
Xd = X[sel]
del X
log(f"features for the domain: {Xd.shape}")
Xn = _normalize_features(Xd, norm_method=seg.get("norm_method", "row"))
kk = "auto" if str(args.k).lower() == "auto" else int(args.k)
lab_flat, k_chosen, *_ = _run_clustering(Xn, n_clusters=kk, k_max=(args.k_max if kk == "auto" else kk), min_k=2,
                                        pca_variance=float(seg.get("pca_variance", 0.92)))
sub = np.full(shape, -1, np.int16)
idx = np.flatnonzero(valid.reshape(-1))[sel] if valid.shape == mask.shape else np.flatnonzero(mask.reshape(-1))
sub.reshape(-1)[idx] = lab_flat
sub = _clean_label_map(sub, min_size=int(seg.get("min_region_size", 15)))

# identify each sub-cluster from its calibrated composition (signed nets, aggregate clip)
matches = {}
for c in range(k_chosen):
    m = sub == c
    if not m.any():
        continue
    sums = {el: max(0.0, float(raw[el][m].sum())) for el in use}
    tot = sum(sums.values())
    wt = eq.apply_calibration_fractions({el: v / tot for el, v in sums.items()}, k_fac)
    ids = mi.identify(wt, quantified=True, fe3_fraction=0.1, top_n=3, coating_elements=frozenset({"C"}))
    matches[c] = (m.sum(), wt, ids)
    log(f"  sub-cluster {c}: {m.sum():8d} px  " + " ".join(f"{e} {v:.1f}" for e, v in sorted(wt.items(), key=lambda kv: -kv[1]) if v > 0.4)
        + " | " + "; ".join(f"{x.name} {x.confidence_pct:.0f}% ({x.confidence})" for x in ids))

rank = mi._CONFIDENCE_RANK
hits = [c for c, (n, wt, ids) in matches.items()
        if ids and ids[0].name.startswith(args.target) and rank.get(ids[0].confidence, 0) >= rank.get(args.min_confidence, 0)]
if not hits:
    log(f"no sub-cluster identified as {args.target}; nothing split"); sys.exit(0)
new_cid = max(names) + 1
split_mask = np.isin(sub, hits)
labels[split_mask] = new_cid
names[new_cid] = args.new_name
log(f"split off sub-cluster(s) {hits} as domain {new_cid} '{args.new_name}': {split_mask.sum()} px ({100 * split_mask.mean():.2f} % of the map)")

# rewrite the exports
for f in ("eds_calibration.json", "phase_map_inset.npz", "phase_map_inset.json"):
    if (DATA / f).exists() and (DATA / f).resolve() != (OUT / f).resolve():
        shutil.copy2(DATA / f, OUT / f)
np.savez_compressed(OUT / "phase_map_labels.npz", labels=labels)
import imageio.v2 as iio
iio.imwrite(str(OUT / "phase_map_combined.png"), _label_map_to_rgba(labels, new_cid + 1))
px = {c: int((labels == c).sum()) for c in sorted(names)}
with open(OUT / "phase_legend.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cluster_id", "name", "pixel_count"])
    for c in sorted(names):
        w.writerow([c, names[c], px[c]])
# minerals: copy rows, append the new domain (structural ID of its own composition)
rows = list(csv.DictReader(open(DATA / "phase_minerals.csv")))
cols = list(rows[0].keys())
n_new, wt_new, ids_new = max(((matches[c][0], matches[c][1], matches[c][2]) for c in hits), key=lambda t: t[0])
sums = {el: max(0.0, float(raw[el][split_mask].sum())) for el in use}; tot = sum(sums.values())
wt_new = eq.apply_calibration_fractions({el: v / tot for el, v in sums.items()}, k_fac)
ids_new = mi.identify(wt_new, quantified=True, fe3_fraction=0.1, top_n=3, coating_elements=frozenset({"C"}))
top = ids_new[0]
entry = next(e for e in mi.DEFAULT_LIBRARY.values() if e.name == top.name)
new_row = {c: "" for c in cols}
new_row.update({"cluster_id": new_cid, "name": args.new_name, "pixel_count": px[new_cid],
                "top1_mineral": top.name, "top1_abbrev": top.abbrev, "top1_score": f"{top.score:.4f}",
                "top1_confidence": top.confidence, "top1_method": top.method,
                "top2_mineral": ids_new[1].name if len(ids_new) > 1 else "", "top2_score": f"{ids_new[1].score:.4f}" if len(ids_new) > 1 else "",
                "top3_mineral": ids_new[2].name if len(ids_new) > 2 else "", "top3_score": f"{ids_new[2].score:.4f}" if len(ids_new) > 2 else "",
                "flags": ";".join(list(top.flags) + ["second_stage_split", f"named_{args.new_name}_by_texture"])})
for r in rows:
    if int(r["cluster_id"]) in srcs:
        r["pixel_count"] = str(px[int(r["cluster_id"])])
with open(OUT / "phase_minerals.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
    w.writerow(new_row)
# composition: recompute the source and the new domain (full masks), keep the rest
crows = list(csv.DictReader(open(DATA / "phase_composition.csv")))
ccols = list(crows[0].keys()); el_cols = [c for c in ccols if c not in ("cluster_id", "name", "pixel_count", "units")]
def comp_row(cid, nm, m):
    sums = {el: max(0.0, float(raw[el][m].sum())) for el in use}; tot = sum(sums.values())
    wt = eq.apply_calibration_fractions({el: v / tot for el, v in sums.items()}, k_fac)
    return {"cluster_id": cid, "name": nm, "pixel_count": int(m.sum()), "units": "wt%", **{e: f"{wt.get(e, 0.0):.4f}" for e in el_cols}}
out_rows = [comp_row(int(r["cluster_id"]), names[int(r["cluster_id"])], labels == int(r["cluster_id"])) if int(r["cluster_id"]) in srcs else r for r in crows]
out_rows.append(comp_row(new_cid, args.new_name, split_mask))
with open(OUT / "phase_composition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ccols); w.writeheader()
    for r in out_rows: w.writerow(r)
# params: unmix of the source is re-estimated by the caller (segment_worked_example unmix step is
# not re-run here); record the split
seg["split"] = {"source_cids": srcs, "source_names": args.domain, "k": (int(k_chosen)), "target": args.target,
                "new_cid": new_cid, "new_name": args.new_name, "n_px": int(split_mask.sum()),
                "sub_clusters": {str(c): {"n_px": int(n), "top1": (ids[0].name if ids else None),
                                          "confidence_pct": (ids[0].confidence_pct if ids else None)}
                                 for c, (n, wt, ids) in matches.items()},
                "provenance": "split_domain.py (second-stage GMM on one domain, same features)"}
json.dump(seg, open(OUT / "segmentation_params.json", "w"), indent=2, default=str)
log(f"wrote exports to {OUT}; new domain composition: " + " ".join(f"{e} {v:.1f}" for e, v in sorted(wt_new.items(), key=lambda kv: -kv[1]) if v > 0.4)
    + f" | ID {top.name} {top.confidence_pct:.0f}%")
