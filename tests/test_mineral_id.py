"""Mineral identification from EDS composition (mineral_id.py).

Pure-function tests: no Qt/napari. Synthetic compositions are built from ideal
cation counts (O is dropped/renormalised internally, so only mole ratios matter).
"""
import json

import mineral_id as mid
from thermo.composition import ATOMIC_WEIGHTS


def cw(cations):
    """Element wt% input from ideal cation counts."""
    return {el: n * ATOMIC_WEIGHTS[el] for el, n in cations.items()}


def find(matches, group):
    return next((m for m in matches if m.group == group), None)


# ---------------------------------------------------------------------------
# Quantified: structural-formula recalculation
# ---------------------------------------------------------------------------

def test_forsterite_matches_olivine_quantified():
    m = mid.identify(cw({"Mg": 2, "Si": 1}), quantified=True)
    t = m[0]
    assert t.group == "olivine"
    assert t.method == "structural"
    assert t.confidence == "high"
    assert abs(t.indices["Mg#"] - 1.0) < 1e-3
    assert all(abs(c - 1.0) < 0.05 for c in t.site_closure.values())
    assert t.confidence_pct > 90


def test_fo80_still_matches_olivine():
    # Solid-solution assertion: Fo80 must still match the olivine *group*.
    m = mid.identify(cw({"Mg": 1.6, "Fe": 0.4, "Si": 1}), quantified=True)
    t = m[0]
    assert t.group == "olivine"
    assert t.score > 0.95
    assert 0.78 < t.indices["Mg#"] < 0.82


def test_pure_silica_matches_quartz_with_polymorph_flag():
    m = mid.identify(cw({"Si": 1}), quantified=True)
    t = m[0]
    assert t.abbrev == "Qz"
    assert "polymorph_ambiguous" in t.flags


def test_anorthite_is_plagioclase_not_kfs():
    m = mid.identify(cw({"Ca": 1, "Al": 2, "Si": 2}), quantified=True, top_n=99)
    t = m[0]
    assert t.group == "plagioclase"
    assert abs(t.indices["An"] - 1.0) < 1e-3
    kfs = find(m, "alkali_feldspar")
    assert kfs is not None and t.score > kfs.score   # Ca end-member != K-feldspar


def test_almandine_is_garnet_not_pyroxene():
    # Garnet and pyroxene share a cation:O ratio; the M-site model must not let
    # pyroxene absorb garnet's octahedral Al2 (else all three tie at 100%).
    m = mid.identify(cw({"Fe": 3, "Al": 2, "Si": 3}), quantified=True, top_n=99)
    t = m[0]
    assert t.group == "garnet"
    assert t.confidence == "high"
    opx = find(m, "orthopyroxene")
    cpx = find(m, "clinopyroxene")
    assert t.score > opx.score and t.score > cpx.score


def test_grossular_still_garnet():
    # Ca end-member of the garnet solid solution must still match garnet.
    m = mid.identify(cw({"Ca": 3, "Al": 2, "Si": 3}), quantified=True)
    assert m[0].group == "garnet"


def test_al2sio5_single_polymorph_entry():
    # The three Al2SiO5 polymorphs are merged into one entry (no triple-100% tie).
    m = mid.identify(cw({"Al": 2, "Si": 1}), quantified=True)
    assert m[0].abbrev == "Als"
    assert "polymorph_ambiguous" in m[0].flags
    als = [e for e in mid.DEFAULT_LIBRARY.values() if e.group == "Al2SiO5"]
    assert len(als) == 1


def test_als_thermo_abbrev_is_generic_and_polymorph_choices():
    # The merged Als maps to the generic "als" key (matches any polymorph);
    # the specific polymorphs are modelling-only choices, not library entries.
    assert mid.THERMO_ABBREV["Als"] == "als"
    keys = {p["thermo_abbrev"] for p in mid.MODELLING_POLYMORPHS}
    assert keys == {"ky", "sill", "and"}
    assert not any(e.group == "Al2SiO5" and e.abbrev in ("Ky", "Sil", "And")
                   for e in mid.DEFAULT_LIBRARY.values())


def test_magnetite_matches_spinel():
    # Fe3O4 = 1 Fe2+ : 2 Fe3+  ->  fe3_fraction = 2/3
    m = mid.identify(cw({"Fe": 3}), quantified=True, fe3_fraction=2.0 / 3.0)
    assert m[0].group == "spinel"


def test_biotite_not_mislabelled_muscovite():
    # Trioctahedral (Mg,Fe)3 mica must beat dioctahedral Al2 muscovite, in both
    # quantified (O-basis = framework+OH/2 closes the hydrous formula) and ratio
    # modes (muscovite M excludes Mg/Fe so it can't absorb biotite's octahedral).
    comp = cw({"K": 0.95, "Mg": 1.4, "Fe": 1.3, "Al": 1.25, "Si": 2.8, "Ti": 0.2})
    for q in (True, False):
        m = mid.identify(comp, quantified=q, top_n=99)
        t = m[0]
        assert t.group == "biotite", (q, t.group)
        ms = find(m, "muscovite")
        assert ms is not None and t.score > ms.score
    # quantified biotite should recalc well (the O-basis fix), not score near zero
    assert mid.identify(comp, quantified=True)[0].confidence_pct > 80


def test_muscovite_still_matches_muscovite():
    m = mid.identify(cw({"K": 0.95, "Al": 2.9, "Si": 3.1}), quantified=True)
    assert m[0].group == "muscovite"
    assert m[0].confidence_pct > 80


def test_tremolite_is_amphibole_not_pyroxene():
    # A-empty amphibole (tremolite) must not be penalised into clinopyroxene.
    m = mid.identify(cw({"Ca": 2, "Mg": 5, "Si": 8}), quantified=True, top_n=99)
    assert m[0].group == "amphibole"
    cpx = find(m, "clinopyroxene")
    assert cpx is not None and m[0].score > cpx.score


def test_carbonate_capped_at_moderate():
    # EDS sees only Ca for calcite (C / CO2 are invisible) -> no APFU, never "high".
    m = mid.identify({"Ca": 100.0}, quantified=True)
    t = m[0]
    assert t.group == "carbonate"
    assert t.confidence == "moderate"
    assert "carbonate_unmeasured_co2" in t.flags
    assert t.apfu is None


def test_carbonate_is_not_a_catch_all_for_mixed_silicates():
    # Diopside-chlorite mixed domain (serpdi EBSD maps, 2026-10-08). In
    # structural mode every silicate fails site closure, and calcite, exempt
    # from the recalc, used to win with a ~0.3 ratio score that hardly depends
    # on composition. EDS does see the Si and Al, so a carbonate must pay for
    # cations no carbonate site hosts.
    mix = {"Si": 23.1, "Al": 2.3, "Mg": 10.9, "Fe": 3.3, "Ca": 13.0}
    m = mid.identify(mix, quantified=True, top_n=99)
    assert m[0].group != "carbonate"
    carbonates = [x for x in m if x.group == "carbonate"]
    assert carbonates and all(x.score < 0.05 for x in carbonates)
    assert all("mixed_pixel_suspect" in x.flags for x in carbonates)
    comp = mid.Composition.from_element_wt(mix)
    assert 0.4 < mid.unassigned_fraction(comp, mid.DEFAULT_LIBRARY["Cal"]) < 0.6
    assert mid.unassigned_fraction(mid.Composition.from_element_wt({"Ca": 100.0}),
                                   mid.DEFAULT_LIBRARY["Cal"]) == 0.0


def test_slightly_impure_calcite_stays_calcite():
    # A few % Si from a neighbouring grain must not kill a genuine carbonate.
    m = mid.identify({"Ca": 38.0, "Si": 1.5}, quantified=True)
    assert m[0].group == "carbonate"
    assert m[0].confidence == "moderate"
    assert "mixed_pixel_suspect" not in m[0].flags


def test_site_closure_good_beats_bad():
    good = mid.identify(cw({"Mg": 2, "Si": 1}), quantified=True, top_n=99)  # olivine
    bad = mid.identify(cw({"Mg": 1, "Si": 1}), quantified=True, top_n=99)   # Si-overfilled
    good_ol = find(good, "olivine")
    bad_ol = find(bad, "olivine")
    assert good_ol.score > bad_ol.score
    assert good_ol.confidence == "high"
    assert bad_ol.confidence != "high"


def test_garbage_input_is_low_confidence():
    m = mid.identify({"Cu": 60.0, "Pb": 40.0}, quantified=True)
    assert m[0].confidence == "low"
    assert m[0].score < 0.6


def test_empty_site_penalty_pure_si_vs_framework_absorbers():
    # Regression for the merged-(Si,Al)-framework leak: nearly pure Si used to
    # score ~0.8 against cordierite and alkali feldspar (Si fills the whole
    # T-site; only the small empty octahedral/alkali site penalised), so those
    # appeared as confident-looking candidates on quartz-dominated domains.
    # The empty-essential-site factor caps a candidate whose octahedral or
    # alkali site is empty; quartz itself is untouched and must win by a wide
    # margin.
    m = mid.identify(cw({"Si": 1}), quantified=False, top_n=99)
    qz = find(m, "quartz")
    crd = find(m, "cordierite")
    kfs = find(m, "alkali_feldspar")
    assert qz is m[0] and qz.score > 0.95
    assert crd is not None and crd.score < 0.6
    assert kfs is not None and kfs.score < 0.6


def test_empty_site_penalty_spares_filled_solid_solutions():
    # A genuine solid solution fills every essential site via its alternates,
    # so the penalty factor must be ~1: Fo80 keeps its near-perfect score and
    # an intermediate plagioclase still matches plagioclase confidently.
    m = mid.identify(cw({"Mg": 1.6, "Fe": 0.4, "Si": 1}), quantified=False)
    assert m[0].group == "olivine" and m[0].score > 0.95
    # An50: Na0.5 Ca0.5 Al1.5 Si2.5 O8
    m = mid.identify(cw({"Na": 0.5, "Ca": 0.5, "Al": 1.5, "Si": 2.5}),
                     quantified=False)
    assert m[0].group == "plagioclase" and m[0].score > 0.9


# ---------------------------------------------------------------------------
# Net counts: ratio-only, flagged approximate
# ---------------------------------------------------------------------------

def test_netcounts_flagged_approximate():
    m = mid.identify(cw({"Mg": 2, "Si": 1}), quantified=False)
    assert m[0].group == "olivine"
    for mm in m:
        assert mm.confidence == "approximate"
        assert "uncalibrated_intensities" in mm.flags
        assert mm.apfu is None
        assert mm.method == "ratio"


# ---------------------------------------------------------------------------
# Robustness + contracts
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty():
    assert mid.identify({}, quantified=True) == []
    assert mid.identify({"O": 50.0}, quantified=True) == []     # O is dropped
    assert mid.identify({"Si": 0.0}, quantified=False) == []


def test_top_n_and_sorted():
    m = mid.identify(cw({"Mg": 2, "Si": 1}), quantified=True, top_n=5)
    assert len(m) == 5
    assert [x.score for x in m] == sorted((x.score for x in m), reverse=True)


def test_thermo_abbrev_contract():
    for ab in mid.DEFAULT_LIBRARY:
        assert mid.THERMO_ABBREV.get(ab)            # every entry maps to a phase
    for mm in mid.identify(cw({"Mg": 2, "Si": 1}), quantified=True, top_n=3):
        assert mm.thermo_abbrev


def test_assemblage_collapses_to_phases():
    ol = mid.identify(cw({"Mg": 2, "Si": 1}), quantified=True)
    qz = mid.identify(cw({"Si": 1}), quantified=True)
    asm = mid.assemblage({0: ol, 1: qz, 2: []}, min_confidence="moderate")
    assert "ol" in asm and "q" in asm
    assert asm == sorted(asm)


def test_decimation_preserves_domain_identification():
    """A domain segmented/read at 1/ds (the large-ROI guard) IDs the same mineral.

    Exercises the exact per-domain path the dialog uses (mask → _net_count_fractions
    → identify) on both a full-res grid and a nearest-pixel-decimated one. Locks in
    that ds>1 support doesn't change the result for an olivine domain.
    """
    import numpy as np
    from viewer_thermo_helpers import _net_count_fractions

    H = W = 40

    def bm(val):  # quantified (Oxford-style): net = main = wt%, sidebands 0
        return {"main": np.full((H, W), val, np.float32),
                "sb_lo": np.zeros((H, W), np.float32),
                "sb_hi": np.zeros((H, W), np.float32)}

    band = {"Mg": bm(2 * ATOMIC_WEIGHTS["Mg"]), "Si": bm(ATOMIC_WEIGHTS["Si"])}
    label = np.zeros((H, W), np.int16)

    def top_for(bands, lab):
        frac, _ = _net_count_fractions(bands, sub_mask=(lab == 0))
        return mid.identify({k: 100 * v for k, v in frac.items()}, quantified=True)[0]

    full = top_for(band, label)
    ds = 4
    band_d = {el: {b: a[::ds, ::ds] for b, a in bands.items()} for el, bands in band.items()}
    dec = top_for(band_d, label[::ds, ::ds])

    assert full.group == dec.group == "olivine"
    assert abs(full.confidence_pct - dec.confidence_pct) < 1e-6


def test_carbon_is_stripped_like_oxygen():
    # Carbon (coating / epoxy) is not a rock-forming cation here; it must be
    # ignored so it doesn't pollute the cation-ratio match. Forsterite + a slug
    # of spurious C must still ID as olivine, identically to the C-free input.
    base = cw({"Mg": 2, "Si": 1})
    withc = dict(base); withc["C"] = 30.0
    assert "C" in mid.IGNORED_ELEMENTS
    a = mid.identify(base, quantified=False)
    b = mid.identify(withc, quantified=False)
    assert a[0].abbrev == b[0].abbrev == "Ol"
    assert abs(a[0].confidence_pct - b[0].confidence_pct) < 1e-6


def test_anion_coating_split_constants():
    # O is the always-dropped anion; C is the default (configurable) coating;
    # IGNORED_ELEMENTS stays the union for backward compatibility.
    assert mid.ANION_ELEMENTS == frozenset({"O"})
    assert mid.DEFAULT_COATING_ELEMENTS == frozenset({"C"})
    assert mid.IGNORED_ELEMENTS == mid.ANION_ELEMENTS | mid.DEFAULT_COATING_ELEMENTS
    assert "C" in mid.IGNORED_ELEMENTS and "O" in mid.IGNORED_ELEMENTS


def test_coating_elements_configurable():
    base = cw({"Mg": 2, "Si": 1})              # forsterite
    stripped = mid.identify(base, quantified=False)        # default coating {"C"}

    # Default: spurious C is dropped → identical to the clean input.
    withc = dict(base); withc["C"] = 30.0
    assert mid.identify(withc, quantified=False)[0].confidence_pct == \
        stripped[0].confidence_pct

    # Uncoated (empty set): measured C is now RETAINED → it lands in
    # "unassigned" and changes the score (so the result must differ).
    kept = mid.identify(withc, quantified=False, coating_elements=frozenset())
    assert kept[0].confidence_pct != stripped[0].confidence_pct

    # Pt-coated: Pt is stripped (→ same as clean), while C is now retained.
    withpt = dict(base); withpt["Pt"] = 40.0
    pt = mid.identify(withpt, quantified=False, coating_elements={"Pt"})
    assert pt[0].abbrev == "Ol"
    assert abs(pt[0].confidence_pct - stripped[0].confidence_pct) < 1e-6
    # …and with a Pt coating, a spurious C slug DOES perturb the match.
    withptc = dict(withpt); withptc["C"] = 30.0
    ptc = mid.identify(withptc, quantified=False, coating_elements={"Pt"})
    assert ptc[0].confidence_pct != stripped[0].confidence_pct


def test_oxygen_always_dropped_regardless_of_coating():
    # O must be excluded as the anion even when the coating set is empty.
    base = cw({"Mg": 2, "Si": 1})
    witho = dict(base); witho["O"] = 50.0
    a = mid.identify(base, quantified=False, coating_elements=frozenset())
    b = mid.identify(witho, quantified=False, coating_elements=frozenset())
    assert a[0].abbrev == b[0].abbrev == "Ol"
    assert abs(a[0].confidence_pct - b[0].confidence_pct) < 1e-6


def test_chlorite_not_mislabelled_pyroxene():
    # A Mg-rich, Ca-poor sheet silicate (chlorite) read from uncalibrated net
    # counts (Si slightly inflated, a little Ca contamination, carbon coating)
    # must beat clinopyroxene and orthopyroxene. The fix: Ca is diagnostic of
    # cpx's split M2 site, and opx/garnet tetrahedral sites take Si only, so a
    # Si-deficient sheet silicate can't borrow Al to masquerade as a chain
    # silicate. Mirrors the real serpentinite chlorite domain (net-count %).
    comp = {"Mg": 33.0, "Si": 32.0, "Al": 13.0, "Fe": 5.0, "Ca": 4.0,
            "C": 2.0, "Cr": 1.0}
    m = mid.identify(comp, quantified=False, top_n=99)
    assert m[0].group == "chlorite", m[0].group
    cpx = find(m, "clinopyroxene")
    opx = find(m, "orthopyroxene")
    assert cpx is not None and m[0].score > cpx.score
    assert opx is not None and m[0].score > opx.score


def test_garnet_not_a_sponge_for_si_deficient_mix():
    # Garnet's tetrahedral site is Si-only (it is Si3 exactly), so a Si-deficient
    # Fe-Ti-oxide-like composition (the mixed-pixel signal of a tiny oxide grain)
    # must NOT score as confidently garnet as a real garnet does.
    almandine = mid.identify(cw({"Fe": 3, "Al": 2, "Si": 3}),
                             quantified=False, top_n=99)
    grt_alm = find(almandine, "garnet")
    oxide_mix = mid.identify({"Fe": 43.0, "Si": 16.0, "Mg": 13.0, "Al": 6.0,
                              "Ca": 4.0, "Cr": 3.0, "Ti": 1.0},
                             quantified=False, top_n=99)
    grt_mix = find(oxide_mix, "garnet")
    assert grt_alm.score > grt_mix.score + 0.05      # real garnet clearly higher


def test_diopside_still_clinopyroxene_after_msite_split():
    # The cpx M-site split (M2 = Ca,Na diagnostic) must not regress real diopside.
    m = mid.identify(cw({"Ca": 1, "Mg": 1, "Si": 2}), quantified=True, top_n=99)
    assert m[0].group == "clinopyroxene"
    assert m[0].confidence_pct > 90


def test_json_library_override(tmp_path):
    lib_json = {
        "Per": {
            "name": "Periclase", "group": "periclase", "formula": "MgO",
            "cations": {"Mg": 1}, "oxygen": 1.0,
            "sites": {"M": {"cations": ["Mg", "Fe"], "ideal": 1.0}},
            "flags": ["anhydrous"], "indices": [],
        }
    }
    p = tmp_path / "lib.json"
    p.write_text(json.dumps(lib_json), encoding="utf-8")
    lib = mid.load_library(p)
    assert "Per" in lib and "Ol" in lib            # merged over the default
    m = mid.identify(cw({"Mg": 1}), quantified=True, library=lib)
    assert m[0].abbrev == "Per"


# --- Fe-Ti oxides and carbonates at the default Fe3+ setting (2026-11-05) ---
# The analyst's single Fe3+/total-Fe knob (default 0) overfilled magnetite and
# hematite, so pure Fe oxides were named calcite at the ratio-only ceiling.

def test_charge_balance_fe3_for_iron_oxides():
    comp = mid.Composition.from_element_wt(cw({"Fe": 3}))
    assert abs(mid.charge_balance_fe3(comp, mid.DEFAULT_LIBRARY["Mag"]) - 2 / 3) < 1e-6
    assert abs(mid.charge_balance_fe3(comp, mid.DEFAULT_LIBRARY["Hem"]) - 1.0) < 1e-6


def test_magnetite_found_without_setting_fe3():
    m = mid.identify(cw({"Fe": 3}), quantified=True)       # fe3_fraction = 0
    assert m[0].group in ("spinel", "hematite")              # Fe valence unseen
    assert {x.abbrev for x in m[:2]} == {"Mag", "Hem"}
    assert all(x.group != "carbonate" for x in m[:2])
    mag = next(x for x in m if x.abbrev == "Mag")
    assert abs(mag.indices["Fe3+/Fe"] - 2 / 3) < 1e-6
    assert mag.confidence == "moderate"                     # closure fitted, never "high"


def test_ilmenite_does_not_tie_with_hematite():
    m = mid.identify(cw({"Fe": 1, "Ti": 1}), quantified=True)
    assert m[0].abbrev == "Ilm"
    hem = next((x for x in m if x.abbrev == "Hem"), None)
    assert hem is None or hem.score < 0.5


def test_dolomite_composition_is_not_calcite():
    m = mid.identify(cw({"Ca": 1, "Mg": 1}), quantified=True)
    assert m[0].abbrev == "Dol"
    cal = next((x for x in m if x.abbrev == "Cal"), None)
    assert cal is None or cal.score < 0.1


def test_mg_calcite_stays_calcite():
    m = mid.identify(cw({"Ca": 0.9, "Mg": 0.1}), quantified=True)
    assert m[0].abbrev == "Cal"


def test_dominance_from_json_library(tmp_path):
    lib_json = {"Sd": {"name": "Siderite", "group": "carbonate", "formula": "FeCO3",
                       "cations": {"Fe": 1}, "oxygen": 3.0,
                       "sites": {"M": {"cations": ["Fe", "Mg", "Ca", "Mn"], "ideal": 1.0}},
                       "flags": ["carbonate", "no_c_measured"], "indices": [],
                       "dominant": {"M": ["Fe", 0.5]}}}
    p = tmp_path / "lib.json"
    p.write_text(json.dumps(lib_json), encoding="utf-8")
    lib = mid.load_library(p)
    assert lib["Sd"].dominant == {"M": ("Fe", 0.5)}
    assert mid.dominance_factor(mid.Composition.from_element_wt(cw({"Ca": 1})),
                                lib["Sd"]) < 1e-3
