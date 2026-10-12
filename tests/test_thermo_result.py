"""ThermoResult portable serialisation + Composition round-trips."""
import pickle

import pytest

from thermo.composition import Composition
from thermo.result import (
    ThermoResult, PhaseResult,
    result_to_portable_dict, result_from_portable_dict,
    phase_base_name,
)


def _sample_result():
    return ThermoResult(
        p_gpa=1.25, t_c=712.0,
        stable_phases=[
            PhaseResult("liq", 0.3, 0.28, 0.31,
                        composition_oxide_wt={"SiO2": 60.0, "Al2O3": 18.0},
                        is_melt=True),
            PhaseResult("q", 0.7, 0.72, 0.69),
        ],
        bulk_properties={"rho_kgm3": 2680.0, "Vp_kms": 6.1, "fO2": None},
        backend="magemin/ig", converged=True, raw=object(),
    )


def test_portable_roundtrip_drops_raw_and_preserves_fields():
    r = _sample_result()
    d = result_to_portable_dict(r)
    assert "raw" not in d
    r2 = result_from_portable_dict(d)
    assert r2.p_gpa == r.p_gpa and r2.t_c == r.t_c
    assert r2.converged is True and r2.backend == "magemin/ig"
    assert [p.name for p in r2.stable_phases] == ["liq", "q"]
    liq = r2.phase("liq")
    assert liq.is_melt is True
    assert liq.composition_oxide_wt == {"SiO2": 60.0, "Al2O3": 18.0}
    assert r2.has_melt() is True
    assert r2.raw is None


def test_portable_dict_is_picklable():
    d = result_to_portable_dict(_sample_result())
    r2 = result_from_portable_dict(pickle.loads(pickle.dumps(d)))
    assert r2.bulk_properties["rho_kgm3"] == 2680.0
    assert r2.phase("q").mol_fraction == 0.7


def test_to_dict_keeps_every_instance_of_a_repeated_phase():
    # MAGEMin reports a solvus pair as two instances of one solution model
    # (two "fsp" = plagioclase + K-feldspar). The flat export used to key
    # columns by name, so the second instance silently overwrote the first
    # and 5-25 vol% of feldspar vanished from every two-feldspar cell.
    r = ThermoResult(
        p_gpa=0.32, t_c=642.0,
        stable_phases=[
            PhaseResult("fsp", 0.10, 0.11, 0.089),
            PhaseResult("bi", 0.35, 0.36, 0.351),
            PhaseResult("fsp", 0.15, 0.14, 0.134),
            PhaseResult("q", 0.40, 0.39, 0.426),
        ],
        backend="magemin",
    )
    d = r.to_dict()
    assert d["vol_frac_fsp"] == 0.089
    assert d["vol_frac_fsp_2"] == 0.134
    assert d["mol_frac_fsp_2"] == 0.15 and d["wt_frac_fsp_2"] == 0.14
    assert "vol_frac_bi_2" not in d
    assert d["n_phases"] == 4 and d["phases"] == "fsp bi fsp q"
    # the volume fractions of all instances still add up
    tot = sum(v for k, v in d.items() if k.startswith("vol_frac_"))
    assert abs(tot - 1.0) < 1e-9


def test_phase_base_name_strips_instance_suffix_only():
    assert phase_base_name("fsp_2") == "fsp"
    assert phase_base_name("fsp_10") == "fsp"
    assert phase_base_name("fsp") == "fsp"
    assert phase_base_name("H2O") == "H2O"
    assert phase_base_name("Gt(W)") == "Gt(W)"


def test_composition_element_wt_sums_to_100():
    comp = Composition.from_element_wt({"Si": 25.0, "Fe": 10.0, "O": 40.0})
    wt = comp.to_element_wt()
    # 'O' is ignored internally; remaining elements renormalise to 100.
    assert abs(sum(wt.values()) - 100.0) < 1e-6
    assert "O" not in wt


def test_composition_oxide_roundtrip():
    ox = {"SiO2": 50.0, "Al2O3": 15.0, "FeO": 10.0, "MgO": 25.0}
    comp = Composition.from_oxide_wt(ox)
    back = comp.to_oxide_wt()
    assert abs(sum(back.values()) - 100.0) < 1e-6
    # Si should dominate the recovered oxide vector
    assert back.get("SiO2", 0) == max(back.values())


def test_composition_is_picklable():
    comp = Composition.from_element_wt({"Si": 20.0, "Mg": 30.0})
    comp2 = pickle.loads(pickle.dumps(comp))
    assert comp2.to_element_wt().keys() == comp.to_element_wt().keys()


def test_extra_oxides_are_wt_percent_of_the_bulk():
    # "4 wt% H2O" must be 4 wt% of the final bulk, with the measured oxides
    # keeping their ratios. The legacy arithmetic added the value to the
    # oxide sum of a cation-normalised composition (~180), which halved it.
    from thermo import Composition
    c = Composition.from_element_wt({"Si": 26.6, "Al": 10.5, "Fe": 6.3, "K": 4.3, "Mg": 2.6, "Na": 1.0, "O": 45.1})
    dry = c.to_oxide_wt(fe3_fraction=0.1)
    wet = c.to_oxide_wt(fe3_fraction=0.1, extra_oxides={"H2O": 4.0})
    assert abs(sum(wet.values()) - 100.0) < 1e-9
    assert abs(wet["H2O"] - 4.0) < 1e-9
    for k in dry:
        assert abs(wet[k] / wet["SiO2"] - dry[k] / dry["SiO2"]) < 1e-9
    assert abs(wet["SiO2"] - 0.96 * dry["SiO2"]) < 1e-9
    # two extras, and the same result whatever `normalize` says
    two = c.to_oxide_wt(extra_oxides={"H2O": 4.0, "CO2": 1.0}, normalize=False)
    assert abs(two["H2O"] - 4.0) < 1e-9 and abs(two["CO2"] - 1.0) < 1e-9 and abs(sum(two.values()) - 100.0) < 1e-9
    # legacy replay: the old number, well below the nominal value
    old = c.to_oxide_wt(fe3_fraction=0.1, extra_oxides={"H2O": 4.0}, extra_basis="legacy")
    assert 1.9 < old["H2O"] < 2.4 and abs(sum(old.values()) - 100.0) < 1e-9
    import pytest
    with pytest.raises(ValueError):
        c.to_oxide_wt(extra_oxides={"H2O": 100.0})
    with pytest.raises(ValueError):
        c.to_oxide_wt(extra_oxides={"H2O": 4.0}, extra_basis="nonsense")
