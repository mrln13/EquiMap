"""Per-window P-T inversion versus window size (Fig. 3c).

The representative averaging scale of Section 3.7 is the window size below
which a randomly placed window stops representing the section. The direct
test is to invert INDIVIDUAL windows with their OWN observation: the
window's calibrated bulk (composition_scale.csv, eds_windows_run.py) and the
window's own modal proportions read from the phase map inside it (with the
same NNLS unmixing of the mixed domains as the section-wide budget), scored
with the section's per-phase weights, on the same MAGEMin grid as Fig. 5,
and scored TWICE: with the section's sillimanite presence constraint in
every window (columns as before), and with the window's OWN assemblage,
sillimanite required only where the window's phase map contains it and
the open-assemblage tolerance otherwise (*_own columns, 2026-10-27). A
representative window returns the section's accepted region; a
non-representative one loses phases or lands elsewhere.

Windows: every window at L1 and L2, and --n-per-level random windows
(seed 0) at each deeper level up to --max-level. Grids run through the
toolkit's worker pool (thermo.parallel, result-cache aware). Outputs:
  pt_windows.csv            one row per window: level, bbox, local modes,
                            best cell, min misfit, accepted region, overlap
                            with the section region, phases missing locally
  pt_windows_grids.json.gz  the grids as portable dicts (cache-independent)
--from-grids re-scores that dump (windows, bulks and grids taken from it)
without MAGEMin, e.g. after a change of the scoring.
"""
import argparse
import csv
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("fig_pt", HERE / "fig_pt.py")
fig_pt = importlib.util.module_from_spec(spec); spec.loader.exec_module(fig_pt)
from viewer_thermo_helpers import _modal_misfit_grid, _pt_region_from_misfit   # noqa: E402
from thermo import Composition                                               # noqa: E402
from thermo.parallel import iter_grid_results                                # noqa: E402
from thermo.result import result_from_portable_dict                          # noqa: E402

P_RANGE, T_RANGE, NP, NT = (0.1, 1.5), (300.0, 800.0), 20, 20
FE3, H2O = 0.1, 4.0
BK = {"db": "mpe", "fe3_fraction": FE3}
OWN_TOL = 0.10   # open-assemblage tolerance for windows without sillimanite (own-assemblage scoring)

def main():
    # Everything runs under a main guard: the worker pool re-imports this
    # module on Windows (spawn), so no work may happen at import time.
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=str(HERE / "composition_scale.csv"))
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--n-per-level", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seg-ds", type=int, default=5)
    ap.add_argument("--from-grids", action="store_true",
                    help="re-score the stored pt_windows_grids.json.gz (no MAGEMin); "
                         "windows, bulks and grids are taken from the dump")
    args = ap.parse_args()
    t0 = time.time()


    def log(*a):
        print(*a, flush=True)


    # ---- section-wide observation (modes, weights) and region, as in fig_pt.py
    modes_sec, weights = fig_pt.observed_from_exports()
    MK = {"weights": weights, "presence_keys": list(fig_pt.PRESENCE), "open_other_tol": 0.0}
    labels = np.load(HERE / "phase_map_labels.npz")["labels"]
    names = {int(r["cluster_id"]): r["name"].strip() for r in csv.DictReader(open(HERE / "phase_legend.csv"))}
    sill_cid = next((c for c, n in names.items() if n == "Sillimanite"), None)
    seg = json.loads((HERE / "segmentation_params.json").read_text())
    unmix = {int(s): {int(e): float(v) for e, v in (d.get("fractions") or {}).items() if float(v) > 0}
             for s, d in (seg.get("unmix") or {}).items()}
    p_ax = np.linspace(*P_RANGE, NP); t_ax = np.linspace(*T_RANGE, NT)


    def local_modes(bbox):
        y0, y1, x0, x1 = (int(v) // args.seg_ds for v in bbox)
        sub = labels[y0:y1, x0:x1]
        cids, n = np.unique(sub[sub >= 0], return_counts=True)
        px = {int(c): float(k) for c, k in zip(cids, n)}
        sill_px = int(px.get(sill_cid, 0.0)) if sill_cid is not None else 0
        for src, fr in unmix.items():
            if src in px and fr:
                k = px.pop(src); tot = sum(fr.values())
                for e, v in fr.items():
                    px[e] = px.get(e, 0.0) + k * v / tot
        mp = {}
        for cid, k in px.items():
            key = fig_pt.NAME_TO_KEY.get(names.get(cid, ""))
            if key and key not in fig_pt.PRESENCE:      # presence-only phases carry no modal target
                mp[key] = mp.get(key, 0.0) + k
        tot = sum(mp.values())
        return ({k: v / tot for k, v in mp.items()} if tot > 0 else {}), int(sub.size), sill_px


    def region_of(mis):
        return _pt_region_from_misfit(mis, P_RANGE, T_RANGE, NP, NT, delta=fig_pt.REGION_DELTA)


    def build(comp_el):
        import model_bulk      # shared recipe: elements passed, Fe3+, H2O as wt% of the bulk
        return model_bulk.model_composition(comp_el, fe3=FE3, h2o=H2O)


    # ---- sample windows
    META = {"level", "y0", "y1", "x0", "x1", "n_pixels", "mask_frac", "skipped"}
    by_level = {}
    with open(args.windows, newline="") as f:
        for r in csv.DictReader(f):
            lv = int(r["level"])
            if lv == 0 or lv > args.max_level or r.get("skipped"):
                continue
            comp = {k: float(v) for k, v in r.items() if k not in META and v not in ("", None)}
            by_level.setdefault(lv, []).append((tuple(int(r[k]) for k in ("y0", "y1", "x0", "x1")), comp))
    rng = np.random.default_rng(args.seed)
    chosen = []
    for lv, wins in sorted(by_level.items()):
        if len(wins) > args.n_per_level and lv > 2:
            idx = sorted(rng.choice(len(wins), size=args.n_per_level, replace=False).tolist())
            wins = [wins[i] for i in idx]
        chosen += [(lv, bb, comp) for bb, comp in wins]
    log(f"{len(chosen)} windows: " + ", ".join(f"L{lv} {sum(1 for c in chosen if c[0] == lv)}" for lv in sorted(by_level)))

    # ---- section reference: the L0 window (the section bulk) with the section modes
    sec_row = next(r for r in csv.DictReader(open(args.windows, newline="")) if int(r["level"]) == 0)
    sec_comp = {k: float(v) for k, v in sec_row.items() if k not in META and v not in ("", None)}
    if args.from_grids:
        with gzip.open(HERE / "pt_windows_grids.json.gz", "rt") as f:
            G = json.load(f)
        chosen = [(int(w["level"]), tuple(int(v) for v in w["bbox"]), w["bulk"]) for w in G["windows"]]
        results = {int(k): [result_from_portable_dict(d) for d in v] for k, v in G["grids"].items()}
        grids_out = None
        log(f"re-scoring {len(results)} stored grids (no MAGEMin)")
    else:
        jobs = [{"comp": build(sec_comp), "p_range": P_RANGE, "t_range": T_RANGE, "n_p": NP, "n_t": NT}]
        jobs += [{"comp": build(comp), "p_range": P_RANGE, "t_range": T_RANGE, "n_p": NP, "n_t": NT} for _, _, comp in chosen]

        results = {}
        grids_out = {}
        n_done = 0
        for idx, dicts, err in iter_grid_results(jobs, "magemin", BK, max_workers=args.workers):
            n_done += 1
            if err is not None:
                log(f"  job {idx} FAILED: {err}"); continue
            results[idx] = [result_from_portable_dict(d) for d in dicts]
            grids_out[str(idx)] = dicts
            if n_done % 10 == 0 or n_done == len(jobs):
                log(f"  {n_done}/{len(jobs)} grids ({time.time() - t0:.0f} s)")

    sec_mis = np.asarray(_modal_misfit_grid(results[0], modes_sec, NP, NT, **MK)[0], float)
    sec_reg = region_of(sec_mis)
    sec_cells = ~np.isnan(sec_mis) & (sec_mis <= np.nanmin(sec_mis) + fig_pt.REGION_DELTA)
    log(f"section (L0 bulk + section modes): best {sec_reg['pt'][1]:.0f} C / {sec_reg['pt'][0]:.2f} GPa, "
        f"region {sec_reg['t_lo']:.0f}-{sec_reg['t_hi']:.0f} / {sec_reg['p_lo']:.2f}-{sec_reg['p_hi']:.2f}")

    rows = []
    for j, (lv, bb, comp) in enumerate(chosen, start=1):
        if j not in results:
            continue
        lm, n_px, sill_px = local_modes(bb)
        missing = sorted(k for k in modes_sec if k not in lm)
        if not lm:
            rows.append(dict(level=lv, y0=bb[0], y1=bb[1], x0=bb[2], x1=bb[3], n_px=n_px, skipped="no modes"))
            continue
        mis = np.asarray(_modal_misfit_grid(results[j], lm, NP, NT, **MK)[0], float)
        reg = region_of(mis)
        cells = ~np.isnan(mis) & (mis <= np.nanmin(mis) + fig_pt.REGION_DELTA)
        overlap = float((cells & sec_cells).sum() / max(1, cells.sum()))
        # the window's OWN assemblage: sillimanite required only where the
        # window's phase map contains it; elsewhere the open-assemblage
        # tolerance, so predicted minor sillimanite is not penalised either
        MK_own = MK if sill_px > 0 else {"weights": weights, "presence_keys": [], "open_other_tol": OWN_TOL}
        mis_o = np.asarray(_modal_misfit_grid(results[j], lm, NP, NT, **MK_own)[0], float)
        reg_o = region_of(mis_o)
        cells_o = ~np.isnan(mis_o) & (mis_o <= np.nanmin(mis_o) + fig_pt.REGION_DELTA)
        overlap_o = float((cells_o & sec_cells).sum() / max(1, cells_o.sum()))
        recall_o = float((cells_o & sec_cells).sum() / max(1, sec_cells.sum()))
        rows.append(dict(level=lv, y0=bb[0], y1=bb[1], x0=bb[2], x1=bb[3], n_px=n_px, skipped="",
                         best_T_C=reg["pt"][1], best_P_GPa=reg["pt"][0], min_misfit=float(np.nanmin(mis)),
                         T_lo_C=reg["t_lo"], T_hi_C=reg["t_hi"], P_lo_GPa=reg["p_lo"], P_hi_GPa=reg["p_hi"],
                         n_cells=int(cells.sum()), overlap_with_section=overlap,
                         contains_section_best=bool(cells[np.unravel_index(int(np.nanargmin(sec_mis)), sec_mis.shape)]),
                         missing_phases=" ".join(missing),
                         sill_px=sill_px, has_sill=int(sill_px > 0),
                         overlap_own=overlap_o, recall_own=recall_o, n_cells_own=int(cells_o.sum()),
                         best_T_C_own=reg_o["pt"][1], best_P_GPa_own=reg_o["pt"][0],
                         min_misfit_own=float(np.nanmin(mis_o)),
                         T_lo_C_own=reg_o["t_lo"], T_hi_C_own=reg_o["t_hi"],
                         P_lo_GPa_own=reg_o["p_lo"], P_hi_GPa_own=reg_o["p_hi"],
                         **{f"mode_{k}": round(lm.get(k, 0.0), 4) for k in sorted(modes_sec)},
                         **{f"wt_{e}": round(comp.get(e, 0.0), 3) for e in ("Si", "Al", "Fe", "K", "Mg", "Na", "Ca")}))
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(HERE / "pt_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    if grids_out is not None:
        with gzip.open(HERE / "pt_windows_grids.json.gz", "wt") as f:
            json.dump({"p_range": P_RANGE, "t_range": T_RANGE, "n_p": NP, "n_t": NT, "backend": BK,
                       "section": {"bulk": sec_comp, "modes": modes_sec, "region": sec_reg},
                       "windows": [{"level": lv, "bbox": bb, "bulk": comp} for lv, bb, comp in chosen],
                       "grids": grids_out}, f)
    for lv in sorted(by_level):
        rr = [r for r in rows if r["level"] == lv and not r["skipped"]]
        if not rr:
            continue
        T = np.array([r["best_T_C"] for r in rr]); P = np.array([r["best_P_GPa"] for r in rr])
        ov = np.array([r["overlap_with_section"] for r in rr])
        ov_o = np.array([r["overlap_own"] for r in rr])
        miss = sum(1 for r in rr if r["missing_phases"])
        log(f"L{lv}: {len(rr):2d} windows  best T {np.median(T):.0f} [{T.min():.0f}-{T.max():.0f}]  "
            f"best P {np.median(P):.2f} [{P.min():.2f}-{P.max():.2f}]  region overlap with section median {np.median(ov):.2f} (own assemblage {np.median(ov_o):.2f})  "
            f"windows missing a phase {miss}")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
