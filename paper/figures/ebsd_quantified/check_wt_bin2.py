# -*- coding: utf-8 -*-
"""Check the binning-2 AZtec QuantMap wt% exports of the two serpdi EBSD maps.

Parses every per-element CSV in D:/data/EBSD/serpdi_{1,2}_EBSD/wt% with the
repo's own reader (decimal-comma repair included), caches the maps as .npz
beside this script, then reports:
  * grid size vs the EBSD grid / 2, live-pixel fraction, closure to 100 %;
  * per-element statistics on live pixels;
  * whether O follows cation stoichiometry (oxygen "by stoichiometry");
  * EBSD registration (which 2x2 block each bin covers);
  * per-EBSD-phase mean compositions, and for map 1 the same from the
    binning-4 run of 2026-10-05 (trace-element inflation check).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]   # <repo>/_Manuscript/figures/ebsd_quantified
sys.path.insert(0, str(REPO))
import oxford_loader as ol  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = {1: Path(r"D:/data/EBSD/serpdi_1_EBSD"), 2: Path(r"D:/data/EBSD/serpdi_2_EBSD")}
BIN4 = REPO / "_scratch/background_fix/serpdi_1_wt_maps.npz"

# oxygen per unit cation mass for the standard oxides (O wt per cation wt)
AW = {"Si": 28.085, "Al": 26.982, "Mg": 24.305, "Fe": 55.845, "Ca": 40.078,
      "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Na": 22.990, "K": 39.098,
      "Ni": 58.693, "O": 15.999}
O_PER_CAT = {"Si": 2, "Al": 1.5, "Mg": 1, "Fe": 1, "Ca": 1, "Ti": 2, "Cr": 1.5,
             "Mn": 1, "Na": 0.5, "K": 0.5, "Ni": 1}


def load_maps(m):
    npz = HERE / f"serpdi_{m}_wt_maps_bin2.npz"
    if npz.exists():
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    maps = {}
    for p in sorted((SRC[m] / "wt%").glob("*.csv")):
        mo = ol._ELEMENT_CSV_RE_TAIL.search(p.stem)
        if not mo:
            print("  skip", p.name)
            continue
        el = mo.group("el")
        maps[el] = ol._read_element_csv(p)
        print(f"  {el:2s} {maps[el].shape}")
    np.savez_compressed(npz, **maps)
    return maps


def phase_means(maps, ebsd_sub, names, live):
    out = {}
    for ordv, nm in names.items():
        sel = (ebsd_sub == ordv) & live
        if sel.sum() == 0:
            continue
        out[nm] = {el: float(maps[el][sel].mean()) for el in maps}
        out[nm]["_n"] = int(sel.sum())
    return out


def fmt_row(d, els):
    return " ".join(f"{d.get(el, float('nan')):6.2f}" for el in els)


def main():
    els_order = ["Si", "Al", "Mg", "Fe", "Ca", "Ti", "Cr", "Mn", "Na", "K", "Ni", "O"]
    report = {}
    for m in (1, 2):
        print(f"\n################ map {m} ################")
        maps = load_maps(m)
        els = [e for e in els_order if e in maps] + [e for e in maps if e not in els_order]
        shapes = {maps[e].shape for e in maps}
        ebsd = np.load(SRC[m] / "ebsd_phase_map.npy")
        names_raw = json.loads((SRC[m] / "ebsd_phases.json").read_text(encoding="utf-8"))
        # legend keyed by ordinal (string keys in JSON)
        names = {}
        for k, v in (names_raw.items() if isinstance(names_raw, dict) else []):
            try:
                names[int(k)] = v["name"] if isinstance(v, dict) else str(v)
            except (ValueError, TypeError, KeyError):
                pass
        print("elements:", els)
        print("shapes:", shapes, " EBSD grid:", ebsd.shape, " EBSD/2:",
              (ebsd.shape[0] / 2, ebsd.shape[1] / 2))
        H, W = next(iter(shapes))
        stack = np.stack([maps[e] for e in els], axis=0)
        total = stack.sum(axis=0)
        live = total > 0
        print(f"live pixels: {live.mean()*100:.1f} %  ({live.sum()} of {live.size})")
        t = total[live]
        print(f"closure on live px: min {t.min():.3f}  p1 {np.percentile(t,1):.3f}  "
              f"median {np.median(t):.3f}  p99 {np.percentile(t,99):.3f}  max {t.max():.3f}")
        print("partial live (some elements zero, total>0):",
              f"{((stack == 0).any(axis=0) & live).mean()*100:.1f} % of all px")
        print("\nper element on live px:  mean  median   p99    max   zero%")
        for e in els:
            v = maps[e][live]
            print(f"  {e:2s} {v.mean():7.2f} {np.median(v):7.2f} {np.percentile(v,99):7.2f} "
                  f"{v.max():7.2f} {100*(v==0).mean():6.1f}")
        # oxygen by stoichiometry?
        if "O" in maps:
            o_pred = sum(maps[e] * O_PER_CAT[e] * AW["O"] / AW[e] for e in els if e in O_PER_CAT)
            r = maps["O"][live] / np.maximum(o_pred[live], 1e-6)
            print(f"\nO / O_stoich(cations): median {np.median(r):.3f}  "
                  f"p5 {np.percentile(r,5):.3f}  p95 {np.percentile(r,95):.3f}")
        # registration: compare Ca contrast diopside vs chlorite over offsets
        dio = [k for k, v in names.items() if "iopside" in v]
        chl = [k for k, v in names.items() if "hlorite" in v]
        print("\nEBSD legend:", names)
        if dio and chl and "Ca" in maps:
            best = None
            print("registration scan (dy, dx in EBSD px) -> Ca(diopside) - Ca(chlorite), "
                  "indexed-vs-live agreement")
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    sub = np.full((H, W), -1, dtype=np.int32)
                    ys = np.arange(H) * 2 + dy
                    xs = np.arange(W) * 2 + dx
                    oky = (ys >= 0) & (ys < ebsd.shape[0])
                    okx = (xs >= 0) & (xs < ebsd.shape[1])
                    sub[np.ix_(oky, okx)] = ebsd[np.ix_(ys[oky], xs[okx])]
                    d = np.isin(sub, dio) & live
                    c = np.isin(sub, chl) & live
                    if d.sum() == 0 or c.sum() == 0:
                        continue
                    contrast = maps["Ca"][d].mean() - maps["Ca"][c].mean()
                    agree = ((sub > 0) == live).mean()
                    print(f"  ({dy:+d},{dx:+d}) {contrast:6.2f}  {agree:.4f}")
                    if best is None or contrast > best[0]:
                        best = (contrast, dy, dx)
            print("best offset:", best[1:], "(0,0 = bin i covers EBSD rows 2i,2i+1)")
        # per-phase means at (0,0)
        sub = ebsd[0:2 * H:2, 0:2 * W:2]
        sub2 = np.full((H, W), -1, dtype=np.int32)
        sub2[:sub.shape[0], :sub.shape[1]] = sub
        pm = phase_means(maps, sub2, names, live)
        print("\nper-EBSD-phase mean wt% (bin 2):          " + " ".join(f"{e:>6s}" for e in els) + "      n")
        for nm, d in pm.items():
            print(f"  {nm:22s} {fmt_row(d, els)} {d['_n']:7d}")
        report[m] = {"shape": [H, W], "live_frac": float(live.mean()),
                     "closure_median": float(np.median(t)), "phase_means": pm}
        if m == 1 and BIN4.exists():
            z = np.load(BIN4)
            m4 = {k: z[k] for k in z.files}
            H4, W4 = m4["Si"].shape
            live4 = sum(m4[e] for e in m4) > 0
            sub4 = np.full((H4, W4), -1, dtype=np.int32)
            s = ebsd[0:4 * H4:4, 0:4 * W4:4]
            sub4[:s.shape[0], :s.shape[1]] = s
            els4 = [e for e in els if e in m4]
            pm4 = phase_means({e: m4[e] for e in els4}, sub4, names, live4)
            extra = [e for e in m4 if e not in els]
            print(f"\nbinning-4 run (2026-10-05) for comparison, live {live4.mean()*100:.1f} %, "
                  f"extra elements then: {extra}")
            print("per-EBSD-phase mean wt% (bin 4):          " + " ".join(f"{e:>6s}" for e in els4) + "      n")
            for nm, d in pm4.items():
                print(f"  {nm:22s} {fmt_row(d, els4)} {d['_n']:7d}")
                if nm in pm:
                    diff = {e: pm[nm][e] - d[e] for e in els4}
                    print(f"  {'  bin2 - bin4':22s} {fmt_row(diff, els4)}")
            for e in extra:
                v = m4[e][live4]
                print(f"  bin4 {e}: mean {v.mean():.2f} on live px")
            report["bin4_phase_means"] = pm4
    (HERE / "check_wt_bin2.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
