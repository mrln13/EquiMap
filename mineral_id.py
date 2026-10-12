# mineral_id.py
"""
Mineral identification from EDS composition.

Given the mean composition of a phase-segmentation domain, rank candidate
minerals from a curated reference library and (for *quantified* data) recalculate
a structural formula (atoms per formula unit + crystallographic site closure +
indices like Mg#, An).

Reliability depends on the data source, which the caller signals via
``quantified``:

* ``quantified=False`` (Bruker ``bcf_net_counts``) - the input "wt%" are really
  uncalibrated net X-ray intensities, skewed by element-dependent yields. Only a
  **cation-ratio** match is honest; every result is flagged ``approximate`` and
  no APFU is reported.
* ``quantified=True`` (Oxford ``oxford_wt_pct`` / SPX) - true wt%. The full
  **structural-formula** recalculation is applied on top of the ratio match.

Source of the reference data
----------------------------
There is no mineral database in the repo or dependencies, so the library is an
authored table of *ideal end-member stoichiometric formulas* (forsterite Mg2SiO4,
anorthite CaAl2Si2O8, almandine Fe3Al2Si3O12, ...). These are definitional, not
measured, so they carry no analytical uncertainty. Formulas / oxygen bases / site
models follow standard references:

* Deer, Howie & Zussman, *An Introduction to the Rock-Forming Minerals* (3rd ed.)
* IMA-CNMNC approved mineral list
* Klein & Hurlbut, *Manual of Mineral Science*

This is the same basis EPMA recalculation tools (NORM / MINPET / AX) use. The
table is JSON-overridable via :func:`load_library` so it can be corrected/extended.

This module is intentionally Qt-free and napari-free so it is unit-testable
headless. It builds on ``thermo.composition`` for all chemistry.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from thermo.composition import (
    Composition,
    ATOMIC_WEIGHTS,
    OXIDE_FORMULAS,
    oxide_mw,
)

# Data types that mean "true wt%" (structural-formula recalc applies). The
# Oxford/SPX ones come from the eds_cache manifest; "eds_calibrated_wt_pct" is a
# runtime overlay set by the viewer when SPX-anchored k-factors (eds_quantify)
# have been applied to Bruker net-counts - the cache still stores net counts.
QUANTIFIED_TYPES = frozenset({"oxford_wt_pct", "spx_wt_pct", "eds_calibrated_wt_pct"})

# Elements dropped before the cation-ratio match, split by *why* they are dropped:
#
#   ANION_ELEMENTS — O is the anion, not a cation. Always dropped (the cation-only
#   model also re-drops it in Composition.from_element_wt, so its membership here
#   is belt-and-braces). Not configurable.
#
#   DEFAULT_COATING_ELEMENTS — the conductive coating / mounting medium, which is
#   not a rock-forming cation and (for C) unreliable in EDS regardless. It lands in
#   "unassigned" and skews every match (badly for restrictive-site phases like
#   chlorite and the Fe-Ti oxides), so it is dropped. Carbon is the usual coating
#   and the default, but a sample may be uncoated or coated with Pt/Au/Pd/Ir/… —
#   so this set is CONFIGURABLE per dataset (identify(coating_elements=...), the
#   cache manifest's "coating_elements", or the viewer's "Coating to ignore" knob).
#
# IGNORED_ELEMENTS is kept as the union for backward compatibility (the default
# drop set: O + the carbon coating).
ANION_ELEMENTS = frozenset({"O"})
DEFAULT_COATING_ELEMENTS = frozenset({"C"})
IGNORED_ELEMENTS = ANION_ELEMENTS | DEFAULT_COATING_ELEMENTS

# L1 (city-block on the site-fraction simplex) is the default ratio metric: it
# penalises magnitude differences between site occupancies, so chemically
# distinct minerals separate cleanly. Cosine ("cosine") is direction-only and
# scores unrelated minerals deceptively high (set RATIO_METRIC="cosine" to use it).
RATIO_METRIC = "l1"

# ratio_similarity multiplies its L1 score by
# ``EMPTY_SITE_FLOOR + (1 - EMPTY_SITE_FLOOR) * min(site closure)`` so a
# candidate missing an essential constituent (an ~empty crystallographic site)
# is capped at this fraction of its raw score, while a filled-site match is
# untouched. See the comment in ratio_similarity.
EMPTY_SITE_FLOOR = 0.5

# Tolerance of the structural site test: each site's closure (actual/ideal) is
# scored exp(-0.5 ((closure - 1) / sigma)^2), so a 10 % mis-fill costs ~40 %
# and a 30 % mis-fill two orders of magnitude. The same width is used for the
# carbonate exclusion test below.
SITE_CLOSURE_SIGMA = 0.10

# In quantified mode a carbonate can't be confirmed structurally (EDS misses C),
# and its permissive cation site matches almost any divalent cation. Cap its
# score below a structurally-confirmed phase so e.g. magnetite outranks calcite
# for pure Fe. Stays well above the 0.6 "moderate" floor. The cap alone is not
# enough: a carbonate's ratio score for ANY Mg-Ca silicate is ~0.3 almost
# independent of composition, so on a mixed domain where every silicate fails
# site closure the carbonate won by default (serpdi EBSD maps, 2026-10-08: a
# diopside-chlorite domain with 23 wt% Si named "Calcite"). Carbonates are
# therefore also scored on the structural test they CAN take, exclusion: the
# fraction of cations EDS does see (Si, Al, Ti, ...) that no carbonate site
# hosts, with the site-closure tolerance (see identify()).
RATIO_ONLY_CEILING = 0.84

# Order in which cations are placed into structural sites (tetrahedral fillers
# first, then octahedral, then large cations). Fe is kept as *total* Fe.
CATION_FILL_ORDER: Tuple[str, ...] = (
    "Si", "Al", "Ti", "Cr", "Fe", "Mn", "Mg", "Ni", "Zn", "V",
    "Ca", "Na", "K", "Ba", "Sr", "P", "Zr", "Hf", "Nb", "Ta",
)

# entry flag -> human caveat string surfaced on every match of that mineral
_FLAG_CAVEATS = {
    "hydrous": "hydrous_unmeasured_h2o",
    "no_h_measured": "hydrous_unmeasured_h2o",
    "carbonate": "carbonate_unmeasured_co2",
    "no_c_measured": "carbonate_unmeasured_co2",
    "polymorph_ambiguous": "polymorph_ambiguous",
    "fe_redox_sensitive": "fe_redox_assumed",
    "halogen_bearing": "halogen_unmeasured",
    "no_f_measured": "halogen_unmeasured",
}


# ---------------------------------------------------------------------------
# Reference-library data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteSpec:
    """One crystallographic site: the cations it may host + its ideal total."""
    cations: Tuple[str, ...]
    ideal: float


@dataclass(frozen=True)
class MineralEntry:
    name: str                       # display name, e.g. "Olivine"
    abbrev: str                     # short tag, e.g. "Ol"
    group: str                      # solid-solution family (the match target)
    formula: str                    # display formula (end-member anchor)
    cations: Dict[str, float]       # end-member cation counts, e.g. {"Mg":2,"Si":1}
    oxygen: float                   # O atoms per formula unit (APFU recalc basis)
    sites: Dict[str, SiteSpec]      # ordered: fill priority follows insertion order
    flags: Tuple[str, ...] = ()     # "anhydrous"/"hydrous"/"carbonate"/"polymorph_ambiguous"...
    indices: Tuple[str, ...] = ()   # derived indices to report: "Mg#","An","Or"


@dataclass(frozen=True)
class MineralMatch:
    name: str
    abbrev: str
    group: str
    formula: str
    score: float                            # 0..1 overall goodness-of-fit (ranking key)
    method: str                             # "ratio" | "structural"
    confidence: str                         # "high"|"moderate"|"low"|"approximate"
    confidence_pct: float                   # 0..100, independent per-candidate fit
    apfu: Optional[Dict[str, float]]        # atoms per formula unit (structural only)
    site_closure: Optional[Dict[str, float]]  # {site: actual/ideal} (structural only)
    indices: Dict[str, float]               # {"Mg#":0.81, "An":0.43}
    flags: Tuple[str, ...]                  # caveats
    thermo_abbrev: str                      # phase abbrev for the future P-T step

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.confidence_pct:.0f}%, {self.confidence})"


# Local abbrev -> thermodynamic-model phase abbreviation. Used by the future
# "find the P-T field where this assemblage is stable" feature, which feeds
# observed_phases into viewer_thermo_helpers._pt_estimate_from_results.
THERMO_ABBREV: Dict[str, str] = {
    "Qz": "q", "Pl": "pl", "Kfs": "kf", "Ne": "ne", "Ol": "ol",
    "Opx": "opx", "Cpx": "cpx", "Grt": "g", "Ms": "mu", "Bt": "bi",
    "Phl": "bi", "Amph": "amph", "Chl": "chl", "Ep": "ep",
    "Als": "als", "Crd": "cd",
    "Mag": "mt", "Ilm": "ilm", "Rt": "ru", "Hem": "hem", "Spl": "sp",
    "Chr": "sp", "Cal": "cc", "Dol": "do", "Ap": "ap", "Ttn": "sph",
    "Zrn": "zrc",
}

# The three Al2SiO5 polymorphs as explicit *modelling* choices. EDS cannot tell
# them apart, so `identify` only ever returns the merged "Als" entry (thermo key
# "als", which matches any polymorph). When the user resolves the polymorph by
# texture — in the Override column or the add-missing-phase list — these feed a
# *specific* thermo phase key (ky / sill / and) that the pseudosection
# distinguishes. Not part of DEFAULT_LIBRARY (the matcher must never emit them).
MODELLING_POLYMORPHS: List[dict] = [
    {"name": "Kyanite (Al2SiO5)", "abbrev": "Ky", "group": "Al2SiO5",
     "thermo_abbrev": "ky"},
    {"name": "Sillimanite (Al2SiO5)", "abbrev": "Sil", "group": "Al2SiO5",
     "thermo_abbrev": "sill"},
    {"name": "Andalusite (Al2SiO5)", "abbrev": "And", "group": "Al2SiO5",
     "thermo_abbrev": "and"},
]


def _S(cations: Sequence[str], ideal: float) -> SiteSpec:
    return SiteSpec(tuple(cations), float(ideal))


# Curated rock-forming library. Sites are inserted tetrahedral-first so the
# greedy site fill places Si (then Al) into T before octahedral/large sites.
DEFAULT_LIBRARY: Dict[str, MineralEntry] = {
    # --- framework / chain / sheet silicates --------------------------------
    "Qz": MineralEntry(
        "Quartz", "Qz", "quartz", "SiO2",
        {"Si": 1}, 2.0, {"T": _S(["Si"], 1.0)},
        flags=("anhydrous", "polymorph_ambiguous")),
    "Pl": MineralEntry(
        "Plagioclase", "Pl", "plagioclase", "(Na,Ca)Al(Si,Al)Si2O8",
        {"Ca": 1, "Al": 2, "Si": 2}, 8.0,
        # A-site diagnostic cation is Ca (Na shared with alkali feldspar via the
        # albite end-member; K excluded so orthoclase doesn't masquerade as Pl).
        {"T": _S(["Si", "Al"], 4.0), "A": _S(["Ca", "Na"], 1.0)},
        flags=("anhydrous", "solid_solution"), indices=("An",)),
    "Kfs": MineralEntry(
        "Alkali feldspar", "Kfs", "alkali_feldspar", "(K,Na)AlSi3O8",
        {"K": 1, "Al": 1, "Si": 3}, 8.0,
        # Diagnostic cation K (Ca excluded so anorthite doesn't masquerade as Kfs).
        {"T": _S(["Si", "Al"], 4.0), "A": _S(["K", "Na", "Ba"], 1.0)},
        flags=("anhydrous", "solid_solution"), indices=("Or",)),
    "Ne": MineralEntry(
        "Nepheline", "Ne", "feldspathoid", "NaAlSiO4",
        {"Na": 1, "Al": 1, "Si": 1}, 4.0,
        {"T": _S(["Si", "Al"], 2.0), "A": _S(["Na", "K", "Ca"], 1.0)},
        flags=("anhydrous",)),
    "Ol": MineralEntry(
        "Olivine", "Ol", "olivine", "(Mg,Fe)2SiO4",
        {"Mg": 2, "Si": 1}, 4.0,
        {"T": _S(["Si"], 1.0), "M": _S(["Mg", "Fe", "Mn", "Ni", "Ca"], 2.0)},
        flags=("anhydrous", "solid_solution"), indices=("Mg#",)),
    "Opx": MineralEntry(
        "Orthopyroxene", "Opx", "orthopyroxene", "(Mg,Fe)2Si2O6",
        {"Mg": 2, "Si": 2}, 6.0,
        # T = Si only: real opx has negligible tetrahedral Al, so a Si-deficient
        # sheet silicate (chlorite, Si≈3 of 10 cations) can't borrow Al to fill the
        # framework and masquerade as pyroxene (Si=2 of 4). Octahedral Al also kept
        # out of M (minor in opx) so almandine's Al2 isn't absorbed.
        {"T": _S(["Si"], 2.0), "M": _S(["Mg", "Fe", "Mn", "Ni", "Ti"], 2.0)},
        flags=("anhydrous", "solid_solution"), indices=("Mg#",)),
    "Cpx": MineralEntry(
        "Clinopyroxene", "Cpx", "clinopyroxene", "Ca(Mg,Fe)Si2O6",
        {"Ca": 1, "Mg": 1, "Si": 2}, 6.0,
        # M is split into M1 (octahedral) + M2, with Ca/Na diagnostic of M2, so a
        # Ca-poor Mg-silicate (=chlorite) underfills M2 and no longer matches cpx
        # perfectly. T keeps Al (real cpx has tetrahedral Al via the Tschermak
        # substitution) so genuine, slightly Si-variable diopside still fills it.
        {"T": _S(["Si", "Al"], 2.0), "M1": _S(["Mg", "Fe", "Mn", "Ti", "Cr", "Al"], 1.0),
         "M2": _S(["Ca", "Na"], 1.0)},
        flags=("anhydrous", "solid_solution"), indices=("Mg#",)),
    "Grt": MineralEntry(
        "Garnet", "Grt", "garnet", "(Fe,Mg,Ca,Mn)3Al2Si3O12",
        {"Fe": 3, "Al": 2, "Si": 3}, 12.0,
        # T = Si only (garnet is Si3 exactly — no tetrahedral Al). Garnet's three
        # accommodating sites otherwise make it a "sponge" that absorbs almost any
        # Si+Al+Fe+Mg+Ca mix (e.g. mixed-pixel Fe-Ti-oxide grains); requiring a full
        # Si3 tetrahedral site penalises Si-deficient mixed pixels.
        {"T": _S(["Si"], 3.0), "Y": _S(["Al", "Fe", "Cr", "Ti"], 2.0),
         "X": _S(["Fe", "Mg", "Ca", "Mn"], 3.0)},
        flags=("anhydrous", "solid_solution"), indices=("Mg#",)),
    "Ms": MineralEntry(
        "Muscovite", "Ms", "muscovite", "KAl2(AlSi3)O10(OH)2",
        # O-basis = cation-oxide oxygens (framework 10 + OH/2 = 11); EDS sees no H,
        # so the recalc is driven by the measured cations' oxides, not framework O.
        {"K": 1, "Al": 3, "Si": 3}, 11.0,
        # Dioctahedral (Al2): Mg/Fe kept OUT of the octahedral M-site so a
        # trioctahedral (Mg,Fe)3 biotite isn't absorbed and mislabelled muscovite.
        {"T": _S(["Si", "Al"], 4.0), "M": _S(["Al", "Ti"], 2.0),
         "A": _S(["K", "Na", "Ca", "Ba"], 1.0)},
        flags=("hydrous", "no_h_measured", "solid_solution")),
    "Bt": MineralEntry(
        "Biotite", "Bt", "biotite", "K(Mg,Fe)3(AlSi3)O10(OH)2",
        {"K": 1, "Mg": 3, "Al": 1, "Si": 3}, 11.0,   # framework 10 + OH/2
        {"T": _S(["Si", "Al"], 4.0), "M": _S(["Mg", "Fe", "Mn", "Ti", "Al"], 3.0),
         "A": _S(["K", "Na", "Ca", "Ba"], 1.0)},
        flags=("hydrous", "no_h_measured", "solid_solution"), indices=("Mg#",)),
    "Amph": MineralEntry(
        "Amphibole", "Amph", "amphibole", "Ca2(Mg,Fe)5Si8O22(OH)2",
        {"Ca": 2, "Mg": 5, "Si": 8}, 23.0,   # framework 22 + OH/2
        # No A-site: it is vacancy-bearing (0-1), so a fixed ideal would wrongly
        # penalise A-empty amphiboles like tremolite. T8 + M5 + B2 define it.
        {"T": _S(["Si", "Al"], 8.0), "M": _S(["Mg", "Fe", "Mn", "Al", "Ti"], 5.0),
         "B": _S(["Ca", "Na", "Fe", "Mn"], 2.0)},
        flags=("hydrous", "no_h_measured", "solid_solution"), indices=("Mg#",)),
    "Chl": MineralEntry(
        "Chlorite", "Chl", "chlorite", "(Mg,Fe)5Al(AlSi3)O10(OH)8",
        {"Mg": 5, "Al": 2, "Si": 3}, 14.0,   # framework 10 + OH(8)/2
        {"T": _S(["Si", "Al"], 4.0), "M": _S(["Mg", "Fe", "Mn", "Al"], 6.0)},
        flags=("hydrous", "no_h_measured", "solid_solution"), indices=("Mg#",)),
    "Ep": MineralEntry(
        "Epidote", "Ep", "epidote", "Ca2(Al,Fe)3Si3O12(OH)",
        {"Ca": 2, "Al": 2, "Fe": 1, "Si": 3}, 12.0,
        {"T": _S(["Si"], 3.0), "M": _S(["Al", "Fe", "Mn", "Ti"], 3.0),
         "A": _S(["Ca", "Na"], 2.0)},
        flags=("hydrous", "no_h_measured")),
    # --- Al2SiO5 polymorphs: one entry (kyanite/sillimanite/andalusite are
    #     compositionally identical, so they would always tie; the polymorph is
    #     resolved by texture/context, not EDS). User can rename to the specific one.
    "Als": MineralEntry(
        "Al2SiO5 (kyanite/sillimanite/andalusite)", "Als", "Al2SiO5", "Al2SiO5",
        {"Al": 2, "Si": 1}, 5.0,
        {"T": _S(["Si"], 1.0), "M": _S(["Al"], 2.0)},
        flags=("anhydrous", "polymorph_ambiguous")),
    "Crd": MineralEntry(
        "Cordierite", "Crd", "cordierite", "(Mg,Fe)2Al4Si5O18",
        {"Mg": 2, "Al": 4, "Si": 5}, 18.0,
        {"T": _S(["Si", "Al"], 9.0), "M": _S(["Mg", "Fe", "Mn"], 2.0)},
        flags=("anhydrous", "solid_solution"), indices=("Mg#",)),
    # --- Fe-Ti oxides / spinels ---------------------------------------------
    "Mag": MineralEntry(
        "Magnetite", "Mag", "spinel", "Fe3O4",
        {"Fe": 3}, 4.0,
        {"A": _S(["Fe", "Mg", "Mn", "Zn"], 1.0),
         "B": _S(["Fe", "Ti", "Al", "Cr", "V"], 2.0)},
        flags=("anhydrous", "fe_redox_sensitive")),
    "Spl": MineralEntry(
        "Spinel", "Spl", "spinel", "(Mg,Fe)Al2O4",
        {"Mg": 1, "Al": 2}, 4.0,
        {"A": _S(["Mg", "Fe", "Mn", "Zn"], 1.0),
         "B": _S(["Al", "Cr", "Fe"], 2.0)},
        flags=("anhydrous", "solid_solution"), indices=("Mg#",)),
    "Chr": MineralEntry(
        "Chromite", "Chr", "spinel", "FeCr2O4",
        {"Fe": 1, "Cr": 2}, 4.0,
        {"A": _S(["Fe", "Mg", "Mn"], 1.0),
         "B": _S(["Cr", "Al", "Fe"], 2.0)},
        flags=("anhydrous",)),
    "Ilm": MineralEntry(
        "Ilmenite", "Ilm", "ilmenite", "FeTiO3",
        {"Fe": 1, "Ti": 1}, 3.0,
        {"A": _S(["Fe", "Mg", "Mn"], 1.0), "B": _S(["Ti", "Fe"], 1.0)},
        flags=("anhydrous",)),
    "Rt": MineralEntry(
        "Rutile", "Rt", "rutile", "TiO2",
        {"Ti": 1}, 2.0, {"M": _S(["Ti", "Fe", "Nb", "Ta", "V"], 1.0)},
        flags=("anhydrous", "polymorph_ambiguous")),
    "Hem": MineralEntry(
        "Hematite", "Hem", "hematite", "Fe2O3",
        {"Fe": 2}, 3.0, {"M": _S(["Fe", "Ti", "Al"], 2.0)},
        flags=("anhydrous", "fe_redox_sensitive")),
    # --- carbonates (C / CO2 not seen by EDS) -------------------------------
    "Cal": MineralEntry(
        "Calcite", "Cal", "carbonate", "CaCO3",
        {"Ca": 1}, 3.0, {"M": _S(["Ca", "Mg", "Fe", "Mn"], 1.0)},
        flags=("carbonate", "no_c_measured", "solid_solution")),
    "Dol": MineralEntry(
        "Dolomite", "Dol", "carbonate", "CaMg(CO3)2",
        {"Ca": 1, "Mg": 1}, 6.0,
        {"A": _S(["Ca"], 1.0), "B": _S(["Mg", "Fe", "Mn"], 1.0)},
        flags=("carbonate", "no_c_measured", "solid_solution")),
    # --- phosphate / accessory ----------------------------------------------
    "Ap": MineralEntry(
        "Apatite", "Ap", "apatite", "Ca5(PO4)3(OH,F,Cl)",
        {"Ca": 5, "P": 3}, 12.5,   # cation-oxide O (Ca5 + P3 as P2O5)
        {"M": _S(["Ca", "Na", "Mn", "Sr", "Fe"], 5.0), "T": _S(["P", "Si"], 3.0)},
        flags=("halogen_bearing", "no_f_measured")),
    "Ttn": MineralEntry(
        "Titanite", "Ttn", "titanite", "CaTiSiO5",
        {"Ca": 1, "Ti": 1, "Si": 1}, 5.0,
        {"T": _S(["Si"], 1.0), "B": _S(["Ti", "Fe", "Al"], 1.0), "M": _S(["Ca", "Na"], 1.0)},
        flags=("anhydrous",)),
    "Zrn": MineralEntry(
        "Zircon", "Zrn", "zircon", "ZrSiO4",
        {"Zr": 1, "Si": 1}, 4.0,
        {"T": _S(["Si"], 1.0), "M": _S(["Zr", "Hf", "Th", "U"], 1.0)},
        flags=("anhydrous",)),
}


# ---------------------------------------------------------------------------
# Small numeric helpers (no numpy needed)
# ---------------------------------------------------------------------------

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da <= 0 or db <= 0:
        return 0.0
    return max(0.0, min(1.0, num / (da * db)))


def _l1_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - 0.5*Σ|a-b| on L1-normalised vectors (both in [0,1])."""
    sa = sum(a) or 1.0
    sb = sum(b) or 1.0
    an = [x / sa for x in a]
    bn = [y / sb for y in b]
    return max(0.0, 1.0 - 0.5 * sum(abs(x - y) for x, y in zip(an, bn)))


def _vec_sim(a: Sequence[float], b: Sequence[float]) -> float:
    return _l1_sim(a, b) if RATIO_METRIC == "l1" else _cosine(a, b)


def _geomean(values: Sequence[float]) -> float:
    vals = [max(1e-9, float(v)) for v in values]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _cation_fracs_total_fe(comp: Composition, fe3_fraction: float) -> Dict[str, float]:
    """Cation mole fractions with Fe2+/Fe3+ merged back to total 'Fe'."""
    out: Dict[str, float] = {}
    for k, v in comp.to_cation_mol_frac(fe3_fraction).items():
        key = "Fe" if k in ("Fe2+", "Fe3+") else k
        out[key] = out.get(key, 0.0) + v
    return out


def _assign_sites(amounts: Dict[str, float], entry: MineralEntry
                  ) -> Tuple[Dict[str, float], float]:
    """Greedy fill of ``amounts`` (cation -> atoms) into ``entry`` sites.

    Cations are placed in :data:`CATION_FILL_ORDER`; each cation flows through
    its allowed sites in site-insertion (tetrahedral-first) order, filling each
    to its ideal before spilling; the last allowed site absorbs any remainder
    (so over-filling shows up as a site_closure > 1). Cations no site can host
    are returned as the unassigned total. Returns ``(site_actual, unassigned)``.
    """
    site_actual = {s: 0.0 for s in entry.sites}
    unassigned = 0.0
    ordered = list(CATION_FILL_ORDER) + [c for c in amounts if c not in CATION_FILL_ORDER]
    for cat in ordered:
        amt = float(amounts.get(cat, 0.0))
        if amt <= 0:
            continue
        allowed = [s for s, spec in entry.sites.items() if cat in spec.cations]
        if not allowed:
            unassigned += amt
            continue
        for i, s in enumerate(allowed):
            if i == len(allowed) - 1:
                site_actual[s] += amt
                amt = 0.0
            else:
                cap = max(0.0, entry.sites[s].ideal - site_actual[s])
                take = min(amt, cap)
                site_actual[s] += take
                amt -= take
    return site_actual, unassigned


# ---------------------------------------------------------------------------
# Matching primitives (public for unit testing)
# ---------------------------------------------------------------------------

def ratio_similarity(comp: Composition, entry: MineralEntry,
                     fe3_fraction: float = 0.0) -> float:
    """Cation-ratio match in (site + unassigned) space, in [0, 1].

    Observed cation fractions are scaled to the entry's total cation count then
    assigned to sites; the resulting (per-site occupancy + unassigned) vector is
    compared to the entry's ideal site totals (unassigned ideal = 0) via
    :data:`RATIO_METRIC` (default L1 on the simplex; cosine optional). Because
    substituting cations share a site, this matches a solid solution at any
    composition (e.g. Fo80 still scores ~1 vs olivine), while cations the
    mineral cannot host (e.g. Ca for orthopyroxene) lower the score. L1 is the
    default because cosine is direction-only and scores unrelated minerals too
    high (the "several 100% matches" failure mode).
    """
    cat = _cation_fracs_total_fe(comp, fe3_fraction)
    if not cat:
        return 0.0
    total_cat = sum(spec.ideal for spec in entry.sites.values()) or 1.0
    amounts = {c: f * total_cat for c, f in cat.items()}
    site_actual, unassigned = _assign_sites(amounts, entry)
    sites = list(entry.sites)
    obs = [site_actual[s] for s in sites] + [unassigned]
    ideal = [entry.sites[s].ideal for s in sites] + [0.0]
    base = _vec_sim(obs, ideal)
    # Essential-site emptiness penalty. The library's sites are all essential
    # (vacancy-bearing sites like the amphibole A-site are deliberately not
    # modelled), so a genuine match fills every site via its substitution
    # alternates. Without this factor a composition can score deceptively
    # high against a mineral missing one of its essential constituents,
    # because a merged framework site absorbs the excess of a single cation:
    # nearly pure Si scores ~0.8 vs cordierite/alkali feldspar (the whole
    # (Si,Al) framework fills with Si, leaving only the small octahedral or
    # alkali site as L1 deficit). Scaling by the emptiest site's closure
    # keeps solid solutions at ~1 while capping missing-constituent matches.
    closures = [min(1.0, site_actual[s] / spec.ideal)
                for s, spec in entry.sites.items() if spec.ideal > 0]
    c_min = min(closures) if closures else 1.0
    return base * (EMPTY_SITE_FLOOR + (1.0 - EMPTY_SITE_FLOOR) * c_min)


def unassigned_fraction(comp: Composition, entry: MineralEntry,
                        fe3_fraction: float = 0.0) -> float:
    """Fraction of the cations (total Fe) that no site of ``entry`` can host.

    Zero for a composition the mineral can accommodate at any solid-solution
    ratio; ~0.5 for a silicate matched against a carbonate (all the Si and Al
    are unassigned). This is the structural test available to minerals whose
    O-basis recalculation is meaningless (carbonates: EDS misses the CO2).
    """
    cat = _cation_fracs_total_fe(comp, fe3_fraction)
    if not cat:
        return 0.0
    total_cat = sum(spec.ideal for spec in entry.sites.values()) or 1.0
    amounts = {c: f * total_cat for c, f in cat.items()}
    _site_actual, unassigned = _assign_sites(amounts, entry)
    return float(unassigned / total_cat)


def _closure_score(deviations: Sequence[float]) -> float:
    """Geometric mean of Gaussian penalties on closure deviations (0 = perfect)."""
    return _geomean([math.exp(-0.5 * (d / SITE_CLOSURE_SIGMA) ** 2) for d in deviations])


def _apfu(comp: Composition, entry: MineralEntry, fe3_fraction: float) -> Dict[str, float]:
    """Atoms per formula unit on ``entry.oxygen`` basis (Fe kept as total)."""
    ox_wt = comp.to_oxide_wt(fe3_fraction=fe3_fraction, normalize=False)
    cat_moles: Dict[str, float] = {}
    o_moles = 0.0
    for ox, wt in ox_wt.items():
        if wt <= 0:
            continue
        n = wt / oxide_mw(ox)
        for el, cnt in OXIDE_FORMULAS[ox].items():
            if el == "O":
                o_moles += n * cnt
            else:
                key = "Fe" if el == "Fe" else el
                cat_moles[key] = cat_moles.get(key, 0.0) + n * cnt
    if o_moles <= 0:
        return {}
    scale = entry.oxygen / o_moles
    return {c: m * scale for c, m in cat_moles.items()}


def structural_formula(comp: Composition, entry: MineralEntry,
                       fe3_fraction: float = 0.0
                       ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return ``(apfu, site_closure, indices)`` for ``comp`` on ``entry``'s basis.

    ``site_closure[site] = assigned_atoms / ideal_total`` (1.0 = perfect). Only
    meaningful for quantified data; the caller gates on that.
    """
    apfu = _apfu(comp, entry, fe3_fraction)
    if not apfu:
        return {}, {}, {}
    site_actual, _ = _assign_sites(apfu, entry)
    site_closure = {s: site_actual[s] / entry.sites[s].ideal
                    for s in entry.sites if entry.sites[s].ideal > 0}
    indices = _compute_indices(apfu, entry)
    return apfu, site_closure, indices


def _compute_indices(amounts: Dict[str, float], entry: MineralEntry) -> Dict[str, float]:
    """Derived mineral-chemistry indices from cation amounts (apfu or fractions)."""
    out: Dict[str, float] = {}
    g = lambda k: float(amounts.get(k, 0.0))
    for ix in entry.indices:
        if ix == "Mg#":
            den = g("Mg") + g("Fe")
            if den > 0:
                out["Mg#"] = g("Mg") / den
        elif ix == "An":
            den = g("Ca") + g("Na") + g("K")
            if den > 0:
                out["An"] = g("Ca") / den
        elif ix == "Or":
            den = g("K") + g("Na") + g("Ca")
            if den > 0:
                out["Or"] = g("K") / den
    return out


def _caveat_flags(entry: MineralEntry) -> Tuple[str, ...]:
    seen: List[str] = []
    for f in entry.flags:
        c = _FLAG_CAVEATS.get(f)
        if c and c not in seen:
            seen.append(c)
    return tuple(seen)


def grade_confidence(score: float, site_closure: Optional[Dict[str, float]],
                     entry: MineralEntry, quantified: bool) -> Tuple[str, Tuple[str, ...]]:
    """Qualitative confidence band + caveat flags for one match."""
    caveats = _caveat_flags(entry)
    if not quantified:
        return "approximate", ("uncalibrated_intensities",) + caveats
    # Carbonates (and any entry without a reliable structural basis) are scored
    # on cation ratio alone -> cap at moderate, never high.
    carbonate = "carbonate" in entry.flags or "no_c_measured" in entry.flags
    if carbonate or not site_closure:
        return ("moderate" if score >= 0.6 else "low"), caveats
    maxdev = max(abs(c - 1.0) for c in site_closure.values())
    if score >= 0.85 and maxdev <= 0.07:
        band = "high"
    elif score >= 0.6 and maxdev <= 0.15:
        band = "moderate"
    else:
        band = "low"
    # EDS can't see structural OH: never claim "high" for hydrous phases.
    if band == "high" and any(f in entry.flags for f in ("hydrous", "no_h_measured")):
        band = "moderate"
    if maxdev > 0.25 and band == "low":
        caveats = caveats + ("mixed_pixel_suspect",)
    return band, caveats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def identify(composition_wt: Dict[str, float], quantified: bool,
             fe3_fraction: float = 0.0, top_n: int = 3,
             library: Optional[Dict[str, MineralEntry]] = None,
             coating_elements: Optional[Iterable[str]] = None) -> List[MineralMatch]:
    """Rank library minerals for one mean composition (element wt%).

    Parameters
    ----------
    composition_wt :
        ``{element: wt%}``. For ``quantified=False`` these may be uncalibrated
        net-count fractions (×100); treated as a wt% proxy and flagged.
    quantified :
        True for Oxford/SPX true wt% (enables structural recalculation), False
        for Bruker net counts (ratio-only, flagged ``approximate``).
    fe3_fraction :
        Fraction of total Fe taken as Fe3+ (affects the oxide O-basis). Default 0.
    top_n :
        Number of ranked candidates to return.
    library :
        Override the default library (see :func:`load_library`).
    coating_elements :
        Conductive coating / contaminant element symbols to drop before matching.
        ``None`` (default) uses :data:`DEFAULT_COATING_ELEMENTS` (``{"C"}`` — carbon
        coat); pass an explicit set for a different coating (e.g. ``{"Pt"}``) or an
        empty set for an uncoated sample (keeps measured C). O is always dropped as
        the anion regardless of this argument.

    Returns an empty list for empty / all-zero / non-elemental input.
    """
    lib = library if library is not None else DEFAULT_LIBRARY
    coat = DEFAULT_COATING_ELEMENTS if coating_elements is None else frozenset(coating_elements)
    drop = ANION_ELEMENTS | coat
    clean = {k: float(v) for k, v in (composition_wt or {}).items()
             if k not in drop and k in ATOMIC_WEIGHTS and float(v) > 0}
    if not clean:
        return []
    try:
        comp = Composition.from_element_wt(clean)
    except ValueError:
        return []

    matches: List[MineralMatch] = []
    cat_frac = _cation_fracs_total_fe(comp, fe3_fraction)
    for abbrev, entry in lib.items():
        ratio = ratio_similarity(comp, entry, fe3_fraction)
        # Carbonates: the C-site cation and 2/3 of the oxygens are CO2, which
        # EDS can't see, so an O-basis structural recalc is meaningless. Score
        # them on cation ratio only (confidence is capped at moderate below).
        carbonate = "carbonate" in entry.flags or "no_c_measured" in entry.flags
        unassigned = 0.0
        if quantified and not carbonate:
            apfu, closure, indices = structural_formula(comp, entry, fe3_fraction)
            if closure:
                score = ratio * _closure_score([c - 1.0 for c in closure.values()])
            else:
                score = ratio  # recalc unavailable -> fall back to ratio
            method = "structural"
        else:
            apfu, closure = None, None
            indices = _compute_indices(cat_frac, entry)
            if quantified:
                # Quantified carbonate: ratio-only, kept below a structural match,
                # AND penalised for the cations EDS does see that no carbonate
                # site hosts (Si, Al, Ti, ...). Without this exclusion test the
                # ~0.3 ratio score of calcite outranked every silicate whose
                # sites failed closure on a mixed domain (the "catch-all").
                unassigned = unassigned_fraction(comp, entry, fe3_fraction)
                score = ratio * RATIO_ONLY_CEILING * _closure_score([unassigned])
            else:
                score = ratio
            method = "ratio"
        band, flags = grade_confidence(score, closure, entry, quantified)
        if carbonate and quantified and unassigned > 0.25:
            flags = flags + ("mixed_pixel_suspect",)
        matches.append(MineralMatch(
            name=entry.name, abbrev=entry.abbrev, group=entry.group,
            formula=entry.formula, score=score, method=method,
            confidence=band, confidence_pct=round(100.0 * score, 1),
            apfu=apfu, site_closure=closure, indices=indices, flags=flags,
            thermo_abbrev=THERMO_ABBREV.get(entry.abbrev, entry.abbrev.lower())))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:max(1, int(top_n))]


# ---------------------------------------------------------------------------
# Library override (JSON) + forward hook for the pseudosection step
# ---------------------------------------------------------------------------

def load_library(path) -> Dict[str, MineralEntry]:
    """Load a mineral library from JSON, merged over :data:`DEFAULT_LIBRARY`.

    JSON schema: ``{abbrev: {name, group, formula, cations, oxygen, sites,
    flags, indices}}`` where ``sites`` is ``{site: {cations: [...], ideal: n}}``.
    Entries with the same abbrev override the default; new abbrevs are added.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lib = dict(DEFAULT_LIBRARY)
    for abbrev, d in data.items():
        sites = {s: _S(spec["cations"], spec["ideal"]) for s, spec in d["sites"].items()}
        lib[abbrev] = MineralEntry(
            name=d.get("name", abbrev), abbrev=abbrev,
            group=d.get("group", abbrev), formula=d.get("formula", ""),
            cations={k: float(v) for k, v in d["cations"].items()},
            oxygen=float(d["oxygen"]), sites=sites,
            flags=tuple(d.get("flags", ())), indices=tuple(d.get("indices", ())))
    return lib


_CONFIDENCE_RANK = {"approximate": 0, "low": 1, "moderate": 2, "high": 3}


def assemblage(matches_by_cid: Dict[int, List[MineralMatch]],
               min_confidence: str = "moderate") -> List[str]:
    """Collapse per-domain top matches into a deduped thermo-phase assemblage.

    Returns the sorted list of ``thermo_abbrev`` for each domain whose top
    match meets ``min_confidence`` - the exact ``observed_phases`` shape that
    ``viewer_thermo_helpers._pt_estimate_from_results`` consumes. This is the
    forward hook for the future "find the P-T field where this assemblage is
    stable" feature; nothing here runs the model.
    """
    floor = _CONFIDENCE_RANK.get(min_confidence, 2)
    out = set()
    for matches in (matches_by_cid or {}).values():
        if not matches:
            continue
        top = matches[0]
        if _CONFIDENCE_RANK.get(top.confidence, 0) >= floor:
            out.add(top.thermo_abbrev)
    return sorted(out)
