# -*- coding: utf-8 -*-
"""Source of the nodule-core formula quoted in section 4.2 (2026-10-27).

What is in the nodule cores? Full calibrated composition and formula of the
Al2SiO5 cluster of the Fig-4b window at NATIVE resolution (0.5 um/px), with
rims stripped, against the global 2.5 um/px domain, and against epidote,
clinozoisite, margarite and Al2SiO5 reference formulas. Also a pixel-level
look inside the cores: do Ca, Fe, Na, P co-vary (inclusions) or sit flat?
"""
import json
import sys
from pathlib import Path
from _dataset import DATASET_DIR   # dataset folder: env EQUIMAP_DATASET

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIG = HERE
DATA = DATASET_DIR
sys.path.insert(0, str(REPO))
import eds_cache  # noqa: E402
import eds_quantify as eq  # noqa: E402
import mineral_id as mi  # noqa: E402
from eds_background import anchor_elements, background_plan, net_count_maps  # noqa: E402
from thermo.composition import ATOMIC_WEIGHTS  # noqa: E402

O_PER = {"Si": 2, "Al": 1.5, "Fe": 1.0, "Mg": 1, "Ca": 1, "Na": 0.5, "K": 0.5, "Ti": 2, "P": 2.5, "Mn": 1}
CHARGE = {"Si": 4, "Al": 3, "Fe": 2, "Mg": 2, "Ca": 2, "Na": 1, "K": 1, "Ti": 4, "P": 5, "Mn": 2}
ORDER = ["Si", "Al", "Fe", "Mg", "Ca", "Na", "K", "Ti", "P"]


def formula(wt, n_o=5.0, fe3=0.1):
    """apfu on n_o oxygens from cation wt% (Fe split by fe3 for the O budget)."""
    mol = {el: wt.get(el, 0.0) / ATOMIC_WEIGHTS[el] for el in ORDER if el in ATOMIC_WEIGHTS}
    o = sum(mol[el] * O_PER[el] for el in mol) + mol.get("Fe", 0.0) * 0.5 * fe3
    f = n_o / o
    apfu = {el: mol[el] * f for el in mol}
    charge = sum(apfu[el] * CHARGE[el] for el in apfu) + apfu.get("Fe", 0.0) * fe3
    return apfu, charge


def fmt(apfu):
    return "  ".join(f"{el} {apfu[el]:.3f}" for el in ORDER if apfu.get(el, 0) >= 0.0005)


def report(tag, wt, n_px=None):
    apfu5, q5 = formula(wt, 5.0)
    apfu12, _ = formula(wt, 12.5)
    m = mi.identify(wt, quantified=True, fe3_fraction=0.1, top_n=3, coating_elements=frozenset({"C"}))
    top = ", ".join(f"{x.name.split(' ')[0]} {x.confidence_pct:.0f}% ({x.confidence})" for x in m)
    print(f"{tag}" + (f"  [{n_px:,} px]" if n_px else ""))
    print(f"   wt% cations: " + "  ".join(f"{el} {wt.get(el, 0):.1f}" for el in ORDER if el in wt))
    print(f"   per 5 O:    {fmt(apfu5)}   (charge {q5:.2f}; Al/Si {apfu5['Al'] / apfu5['Si']:.2f})")
    print(f"   per 12.5 O: {fmt(apfu12)}")
    print(f"   ID: {top}")


print("=== reference formulas per 5 O ===")
refs = {"Al2SiO5": {"Al": 2, "Si": 1}, "epidote Ca2Al2Fe3+Si3O12(OH)": {"Ca": 0.8, "Al": 0.8, "Fe": 0.4, "Si": 1.2},
        "clinozoisite Ca2Al3Si3O12(OH)": {"Ca": 0.8, "Al": 1.2, "Si": 1.2},
        "margarite CaAl2(Al2Si2O10)(OH)2": {"Ca": 0.417, "Al": 1.667, "Si": 0.833},
        "muscovite KAl2(AlSi3O10)(OH)2": {"K": 0.417, "Al": 1.25, "Si": 1.25}}
for n, f in refs.items():
    print(f"   {n:34s} " + "  ".join(f"{el} {v:.2f}" for el, v in f.items()))

print("\n=== global 2.5 um/px domain 'Sillimanite' (phase_stoichiometry.csv, calibrated cation wt%) ===")
import csv
for r in csv.DictReader(open(FIG / "phase_stoichiometry.csv")):
    if r["name"] == "Sillimanite":
        wt = {k[4:]: float(v) for k, v in r.items() if k.startswith("cal_") and v}
        report(f"erosion {r['erosion_px']} px ({2.5 * int(r['erosion_px']):.0f} um)", wt, int(r["n_px"]))

print("\n=== native-resolution window of Fig 4b (0.5 um/px, staged seed-42 k=6 labels) ===")
meta = json.loads((FIG / "phase_map_inset.json").read_text())
y0, y1, x0, x1 = meta["window_mosaic_px"]
lab = np.load(FIG / "phase_map_inset.npz")["labels"]
legend = {int(r.split(",")[0]): r.split(",")[1].strip() for r in (FIG / "phase_legend.csv").read_text().splitlines()[1:]}
seg = json.loads((FIG / "segmentation_params.json").read_text())
cache_dir = eds_cache.resolve_cache_dir(str(DATA))
cached = list(eds_cache.load_manifest(cache_dir).get("elements") or [])
cations = [e for e in ORDER if e in cached]
plan = background_plan(cached)
need = sorted(set(cations) | set(anchor_elements(plan, cations)))
bands = eds_cache.roi_from_cache(cache_dir, (y0, y1, x0, x1), need, blend_mode="mean", progress=False)
nets = {el: a.astype(np.float32) for el, a in net_count_maps(bands, plan=plan, elements=cations).items()}
del bands
H, W = nets["Si"].shape
lab = lab[:H, :W]
k = eq.Calibration.from_dict(json.loads((FIG / "eds_calibration.json").read_text())["calibration"]).k()


def comp(mask):
    n = int(mask.sum())
    raw = {el: max(0.0, float(nets[el][mask].sum())) / n for el in cations}
    tot = sum(raw.values())
    return eq.apply_calibration_fractions({el: v / tot for el, v in raw.items()}, k), n


sill = next(i for i, n in legend.items() if n == "Sillimanite")
for name_, gid in (("Al2SiO5 cores", sill), ("Muscovite (reference)", 5), ("Plagioclase (reference)", 11)):
    mask0 = lab == gid
    for er in (0, 3, 6, 10):
        mask = binary_erosion(mask0, iterations=er) if er else mask0
        if mask.sum() < 500:
            print(f"{name_} erosion {er}: too few pixels"); continue
        wt, n = comp(mask)
        report(f"{name_}, erosion {er} px ({0.5 * er:.1f} um)", wt, n)

print("\n=== inside the cores (erosion 6 px): pixel-level structure, sigma=1 smoothed nets ===")
core = binary_erosion(lab == sill, iterations=6)
sm = {el: gaussian_filter(nets[el], 1.0)[core] for el in ("Ca", "Fe", "Na", "K", "Mg", "P", "Al", "Si")}
tot = sum(sm.values())
fr = {el: sm[el] / np.maximum(tot, 1e-6) for el in sm}
print("   fraction of total counts, mean / 90th / 99th percentile:")
for el in ("Ca", "Fe", "Na", "K", "P"):
    v = fr[el]
    print(f"     {el}: {v.mean():.3f} / {np.percentile(v, 90):.3f} / {np.percentile(v, 99):.3f}")
print("   correlation of count fractions across core pixels:")
for a, b in (("Ca", "P"), ("Ca", "Fe"), ("Ca", "Na"), ("Ca", "Al"), ("Fe", "Mg"), ("Na", "K")):
    print(f"     r({a},{b}) = {np.corrcoef(fr[a], fr[b])[0, 1]:+.2f}")
hi = fr["Ca"] > np.percentile(fr["Ca"], 95)
print(f"   top-5% Ca pixels: Ca {fr['Ca'][hi].mean():.3f} P {fr['P'][hi].mean():.3f} Fe {fr['Fe'][hi].mean():.3f} "
      f"Na {fr['Na'][hi].mean():.3f} Al {fr['Al'][hi].mean():.3f} Si {fr['Si'][hi].mean():.3f}")
lo = fr["Ca"] < np.percentile(fr["Ca"], 50)
print(f"   lower-half Ca pixels: Ca {fr['Ca'][lo].mean():.3f} P {fr['P'][lo].mean():.3f} Fe {fr['Fe'][lo].mean():.3f} "
      f"Na {fr['Na'][lo].mean():.3f} Al {fr['Al'][lo].mean():.3f} Si {fr['Si'][lo].mean():.3f}")
# formula of the Ca-poor half of the cores
mask_lo = np.zeros_like(core); mask_lo[core] = lo
wt, n = comp(mask_lo)
report("Ca-poor half of the cores (erosion 6 px)", wt, n)
