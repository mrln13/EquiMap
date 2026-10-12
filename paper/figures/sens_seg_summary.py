# -*- coding: utf-8 -*-
"""Summarise the segmentation sensitivity runs: modal proportions after
unmixing, agreement with the adopted map, and the P-T estimate each variant
gives on the Fig-5 grid (a re-scoring; no thermodynamic computation)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "_Manuscript" / "figures"
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "_scratch" / "background_fix" / "sens_seg2"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(FIG))
import fig_pt                                                     # noqa: E402
from viewer_thermo_helpers import _modal_misfit_grid              # noqa: E402

g, keys = fig_pt.load_grid(FIG / "pt_grid.csv")
results = fig_pt.build_results(g, keys, REPO)
n_p, n_t = g.P_GPa.nunique(), g.T_C.nunique()
p_ax, t_ax = np.sort(g.P_GPa.unique()), np.sort(g.T_C.unique())


def read_dir(d):
    px, names = {}, {}
    for r in csv.DictReader(open(d / "phase_legend.csv")):
        px[int(r["cluster_id"])] = float(r["pixel_count"]); names[int(r["cluster_id"])] = r["name"].strip()
    seg = json.loads((d / "segmentation_params.json").read_text())
    tot = sum(px.values())
    mixed = sum(v for c, v in px.items() if "matrix" in names[c].lower()) / tot
    sill = sum(v for c, v in px.items() if names[c] == "Sillimanite") / tot
    raw = dict(px)
    for src, u in (seg.get("unmix") or {}).items():
        src = int(src); fr = {int(e): float(v) for e, v in (u.get("fractions") or {}).items() if float(v) > 0}
        if src in px and fr:
            n = px.pop(src); s = sum(fr.values())
            for e, v in fr.items():
                px[e] = px.get(e, 0.0) + n * v / s
    modes = {}
    for c, n in px.items():
        k = fig_pt.NAME_TO_KEY.get(names.get(c, ""))
        if k and k not in fig_pt.PRESENCE:
            modes[k] = modes.get(k, 0.0) + n
    t = sum(modes.values()); modes = {k: v / t for k, v in modes.items()}
    wts = {}
    for r in csv.DictReader(open(d / "phase_minerals.csv")):
        k = fig_pt.NAME_TO_KEY.get(r["name"].strip())
        if k and r.get("top1_score") and k not in fig_pt.PRESENCE:
            wts[k] = max(wts.get(k, 0.0), float(r["top1_score"]))
    return names, raw, seg, mixed, sill, modes, wts


ref_lab = np.load(FIG / "phase_map_labels.npz")["labels"]
ref_names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(FIG / "phase_legend.csv"))}
all_names = sorted(set(ref_names.values()))


def name_map(lab, names):
    lut = np.full(max(names) + 2, -1, np.int16)
    for c, n in names.items():
        if n not in all_names:
            all_names.append(n)
        lut[c] = all_names.index(n)
    out = np.full(lab.shape, -1, np.int16); ok = lab >= 0; out[ok] = lut[lab[ok]]
    return out


rows = []
dirs = [("adopted (staged)", FIG)] + [(d.name, d) for d in sorted(BASE.iterdir()) if (d / "phase_legend.csv").exists()]
ref_n = name_map(ref_lab, ref_names)
for tag, d in dirs:
    names, raw, seg, mixed, sill, modes, wts = read_dir(d)
    if not all(k in modes for k in ("bi", "q", "mu", "fsp")):
        print(tag, "incomplete modes", modes); continue
    lab = np.load(d / "phase_map_labels.npz")["labels"]
    agree = float((name_map(lab, names) == ref_n)[(lab >= 0) & (ref_lab >= 0)].mean())
    mis = np.asarray(_modal_misfit_grid(results, modes, n_p, n_t, weights=wts, presence_keys=fig_pt.PRESENCE)[0], float)
    j = int(np.nanargmin(mis)); ip, it = divmod(j, n_t)
    acc = mis <= np.nanmin(mis) + fig_pt.REGION_DELTA; pp, tt = np.where(acc)
    row = dict(run=tag, sigma=seg.get("smooth_sigma"), min_region=seg.get("min_region_size"), k=seg.get("k_chosen"),
               seed=seg.get("random_state", 42), n_domains=len(names), mixed_pct=round(100 * mixed, 1), sill_pct=round(100 * sill, 2),
               agree_pct=round(100 * agree, 1), Bt=round(100 * modes["bi"], 1), Qz=round(100 * modes["q"], 1),
               Ms=round(100 * modes["mu"], 1), Pl=round(100 * modes["fsp"], 1),
               best_T=round(float(t_ax[it])), best_P=round(float(p_ax[ip]), 2), misfit=round(float(mis[ip, it]), 3),
               T_lo=round(float(t_ax[tt].min())), T_hi=round(float(t_ax[tt].max())),
               P_lo=round(float(p_ax[pp].min()), 2), P_hi=round(float(p_ax[pp].max()), 2), n_acc=int(acc.sum()))
    rows.append(row)
cols = list(rows[0].keys())
with open(FIG / "sens_segmentation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"{'run':18s} sig minR  k seed dom mixed  sill agree |   Bt   Qz   Ms   Pl | best        misfit region")
for r in rows:
    print(f"{r['run']:18s} {r['sigma']!s:3s} {r['min_region']!s:4s} {r['k']!s:2s} {r['seed']!s:4s} {r['n_domains']:3d} {r['mixed_pct']:5.1f} {r['sill_pct']:5.2f} {r['agree_pct']:5.1f} | "
          f"{r['Bt']:4.1f} {r['Qz']:4.1f} {r['Ms']:4.1f} {r['Pl']:4.1f} | {r['best_T']}/{r['best_P']:.2f}  {r['misfit']:.3f}  {r['T_lo']}-{r['T_hi']} / {r['P_lo']:.2f}-{r['P_hi']:.2f} ({r['n_acc']})")
a = np.array([[r["Bt"], r["Qz"], r["Ms"], r["Pl"]] for r in rows])
print("modes over all runs: min", a.min(0), "max", a.max(0), "sd", np.round(a.std(0), 2))
