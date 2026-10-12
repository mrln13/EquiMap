"""MAGEMin backend: per-instance phase compositions (no Julia needed).

The fixture is a verbatim dump of a MAGEMin_C `gmin_struct` at 642 C /
0.32 GPa for the worked-example bulk (2026-09-15): a two-feldspar cell.
`SS_syms`/`PP_syms` are unordered unique-name lists, while `SS_vec`/`PP_vec`
hold one composition per *instance* in the order given by `ph_id` within
`ph_type`. Zipping the name lists with the vectors (the old code) scrambled
every composition.
"""
import pytest

pytest.importorskip("numpy")
from thermo.backends.magemin import MAGEMinDirectBackend  # noqa: E402

PH = ["bi", "fl", "fsp", "fsp", "q", "sill", "ru"]
PH_TYPE = [1, 1, 1, 1, 0, 0, 0]
PH_ID = [0, 1, 2, 3, 0, 1, 2]
SS_SYMS = ["fl", "bi", "fsp"]          # unordered, unique -> NOT aligned
PP_SYMS = ["ru", "q", "sill"]
SS_VEC = [
    {"SiO2": 33.3, "Al2O3": 19.1, "MgO": 10.8, "FeO": 20.8, "K2O": 10.2, "TiO2": 2.0, "H2O": 3.4},  # biotite
    {"H2O": 100.0},                                                                            # fluid
    {"SiO2": 63.5, "Al2O3": 22.8, "CaO": 4.0, "K2O": 0.8, "Na2O": 8.9},                        # plagioclase
    {"SiO2": 65.5, "Al2O3": 18.8, "CaO": 0.2, "K2O": 12.8, "Na2O": 2.8},                       # K-feldspar
]
PP_VEC = [
    {"SiO2": 100.0},                       # quartz
    {"SiO2": 37.1, "Al2O3": 62.9},         # sillimanite
    {"TiO2": 100.0},                       # rutile
]


def test_instance_mapping_follows_ph_type_and_ph_id():
    comps = MAGEMinDirectBackend._instance_compositions(PH_TYPE, PH_ID, SS_VEC, PP_VEC)
    assert len(comps) == len(PH)
    by = dict(zip(range(len(PH)), comps))
    assert by[0]["K2O"] == 10.2 and by[0]["MgO"] == 10.8       # bi is biotite, not fluid
    assert by[1] == {"H2O": 100.0}                              # fl is the fluid
    assert by[2]["Na2O"] == 8.9 and by[3]["K2O"] == 12.8        # two DISTINCT feldspars
    assert by[4] == {"SiO2": 100.0}                             # q is quartz, not rutile
    assert by[5]["Al2O3"] == 62.9 and by[6] == {"TiO2": 100.0}  # sill, ru


def test_old_name_zip_would_have_scrambled():
    # Documents the failure mode: name-keyed zip of SS_syms with SS_vec.
    scrambled = dict(zip(SS_SYMS, SS_VEC))
    assert scrambled["bi"] == {"H2O": 100.0}           # biotite "was" the fluid
    assert scrambled["fl"]["K2O"] == 10.2              # the fluid "was" biotite
    # only one feldspar composition survives a name-keyed dict; the K-feldspar
    # instance (K2O 12.8) is unreachable
    assert all(c.get("K2O") != 12.8 for c in scrambled.values())
    assert dict(zip(PP_SYMS, PP_VEC))["q"]["Al2O3"] == 62.9  # quartz "was" sillimanite


def test_instance_mapping_tolerates_missing_or_bad_ids():
    comps = MAGEMinDirectBackend._instance_compositions([1, 0, 1], [0, 5, "x"], SS_VEC, PP_VEC)
    assert comps[0] == SS_VEC[0]
    assert comps[1] is None and comps[2] is None
    assert MAGEMinDirectBackend._instance_compositions([], [], SS_VEC, PP_VEC) == []
