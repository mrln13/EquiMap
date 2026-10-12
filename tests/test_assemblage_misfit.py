"""Assemblage -> P-T inversion helpers (viewer_thermo_helpers).

Alias-aware phase-name matching (mineral-ID thermo keys vs MAGEMin/Perple_X
spellings), observed modal proportions from a segmentation label map, and the
per-cell modal misfit grid. All Qt-free.
"""
import numpy as np
import pytest

from thermo.result import PhaseResult, ThermoResult
from viewer_thermo_helpers import (
    _normalize_phase_token, _phase_matches, _pt_estimate_from_results,
    _observed_modes_from_phase_map, _modal_misfit_grid,
    _best_fit_pt, _pt_scale_rows,
    _mc_sample_compositions, _mc_aggregate, _mc_sample_modes,
)


def _res(phases, p=1.0, t=700.0):
    """ThermoResult from [(name, vol, wt, mol, is_melt), ...] tuples."""
    sp = [PhaseResult(name=n, mol_fraction=mol, wt_fraction=wt,
                      vol_fraction=vol, is_melt=melt)
          for (n, vol, wt, mol, melt) in phases]
    return ThermoResult(p_gpa=p, t_c=t, stable_phases=sp)


# ---------------------------------------------------------------------------
# Phase-name matching
# ---------------------------------------------------------------------------

def test_normalize_phase_token():
    assert _normalize_phase_token("Gt(W)") == "gt"
    assert _normalize_phase_token(" Cpx(JH) ") == "cpx"
    assert _normalize_phase_token("melt(G)") == "melt"
    assert _normalize_phase_token("q") == "q"


def test_phase_matches_aliases():
    # mineral-ID thermo keys vs Perple_X solution-model names
    assert _phase_matches("g", "Gt(W)")
    assert _phase_matches("ol", "O(HP)")
    assert _phase_matches("bi", "Bio(TCC)")
    assert _phase_matches("mu", "Mica(CHA1)")
    assert _phase_matches("chl", "Chl(W)")
    # ... vs MAGEMin names
    assert _phase_matches("g", "g")
    assert _phase_matches("amph", "amp")
    assert _phase_matches("amph", "hb")
    # ternary feldspar satisfies either feldspar observation
    assert _phase_matches("pl", "fsp")
    assert _phase_matches("kf", "fsp")
    # full mineral names (EBSD legends / user-edited phase names) work too
    assert _phase_matches("Diopside", "cpx")
    assert _phase_matches("Olivine", "ol")
    assert _phase_matches("Grt", "garnet")          # non-canonical observed alias


def test_phase_matches_exact_fallback_for_unknown_names():
    # A phase outside the alias table still matches by exact (normalised) name,
    # so hand-typed backend names keep working (e.g. staurolite).
    assert _phase_matches("st", "St(W)")
    assert _phase_matches("St(W)", "st")


def test_phase_matches_no_prefix_false_positives():
    assert not _phase_matches("sp", "sph")           # spinel != sphene
    assert not _phase_matches("q", "ru")
    assert not _phase_matches("g", "Opx(JH)")


def test_phase_matches_al2sio5_polymorphs():
    # Generic "als" (the EDS-ambiguous default) matches ANY polymorph...
    assert _phase_matches("als", "ky")
    assert _phase_matches("als", "sill")
    assert _phase_matches("als", "and")
    assert _phase_matches("als", "Als")
    # ...but a user-resolved polymorph matches ONLY itself, so the modelling
    # choice actually changes the stability field / misfit.
    assert _phase_matches("ky", "kyanite")
    assert not _phase_matches("ky", "sill")
    assert not _phase_matches("ky", "and")
    assert _phase_matches("sill", "sillimanite")
    assert not _phase_matches("sill", "ky")
    assert not _phase_matches("and", "sill")


def test_pt_estimate_alias_aware():
    results = [
        _res([("Gt(W)", .5, .5, .5, False), ("fsp", .5, .5, .5, False)]),
        _res([("Gt(W)", 1., 1., 1., False)]),
        _res([("fsp", 1., 1., 1., False)]),
        _res([("Opx(JH)", 1., 1., 1., False)]),
    ]
    mask = _pt_estimate_from_results(results, ["g", "pl"], 2, 2)
    assert mask.tolist() == [[True, False], [False, False]]


def test_pt_estimate_exact_matches_grid_names_only():
    # The Select-assemblage dialog returns the grid's OWN phase names, so the
    # outline must match those exactly — alias broadening would light up cells
    # where a *different* phase in the same alias group is stable (here "Grt",
    # a second garnet spelling, and "fsp", which shares the pl alias group).
    results = [
        _res([("Gt(W)", 1., 1., 1., False)]),    # literal selection
        _res([("Grt", 1., 1., 1., False)]),      # same mineral, other spelling
        _res([("fsp", 1., 1., 1., False)]),      # alkali/ternary feldspar
        _res([("Gt(W)", 1., 1., 1., False)]),    # literal selection
    ]
    # Alias-aware (default): "Gt(W)" also matches "Grt"; "Pl(JH)" matches "fsp".
    alias = _pt_estimate_from_results(results, ["Gt(W)", "Pl(JH)"], 2, 2)
    # No cell has BOTH garnet and feldspar, so the intersection is empty here...
    assert alias.tolist() == [[False, False], [False, False]]
    # ...but garnet alone over-broadens to the "Grt" cell under aliasing:
    alias_g = _pt_estimate_from_results(results, ["Gt(W)"], 2, 2)
    assert alias_g.tolist() == [[True, True], [False, True]]
    # Exact matching outlines ONLY the literal "Gt(W)" cells (== highlight view).
    exact_g = _pt_estimate_from_results(results, ["Gt(W)"], 2, 2, exact=True)
    assert exact_g.tolist() == [[True, False], [False, True]]
    # Case/whitespace-insensitive, but no alias broadening.
    assert _pt_estimate_from_results(
        results, [" gt(w) "], 2, 2, exact=True).tolist() == [[True, False],
                                                             [False, True]]


# ---------------------------------------------------------------------------
# Observed modes from the phase map
# ---------------------------------------------------------------------------

def test_observed_modes_merges_and_normalises():
    pm = np.full((10, 10), -1, dtype=np.int16)
    pm[0:6, :] = 0     # 60 px
    pm[6:9, :] = 1     # 30 px
    pm[9:, :] = 2      # 10 px ; -1 nowhere now -> make some unsegmented
    pm[0, 0:5] = -1    # 5 px unsegmented (excluded from denominator entirely)
    modes, excluded = _observed_modes_from_phase_map(
        pm, {0: "ol", 1: "ol", 2: "cpx"})
    assert excluded == pytest.approx(0.0)
    assert modes["ol"] == pytest.approx(85 / 95)     # cids 0+1 merged
    assert modes["cpx"] == pytest.approx(10 / 95)
    assert sum(modes.values()) == pytest.approx(1.0)


def test_observed_modes_unkeyed_domains_excluded():
    pm = np.zeros((10, 10), dtype=np.int16)
    pm[5:, :] = 1
    modes, excluded = _observed_modes_from_phase_map(pm, {0: "ol"})
    assert modes == {"ol": pytest.approx(1.0)}
    assert excluded == pytest.approx(0.5)            # cid 1 had no key


def test_observed_modes_nothing_keyed():
    pm = np.zeros((4, 4), dtype=np.int16)
    assert _observed_modes_from_phase_map(pm, {}) == ({}, 1.0)
    assert _observed_modes_from_phase_map(pm, {0: "  "}) == ({}, 1.0)


# ---------------------------------------------------------------------------
# Modal misfit grid
# ---------------------------------------------------------------------------

def test_misfit_perfect_match_is_zero():
    results = [_res([("ol", .6, None, .6, False), ("cpx", .4, None, .4, False)])]
    grid, basis, unmatched = _modal_misfit_grid(
        results, {"ol": .6, "cpx": .4}, 1, 1)
    assert grid[0, 0] == pytest.approx(0.0)
    assert basis == "vol"
    assert unmatched == []


def test_misfit_disjoint_is_one():
    results = [_res([("q", 1.0, None, 1.0, False)])]
    grid, _, unmatched = _modal_misfit_grid(results, {"ol": 1.0}, 1, 1)
    assert grid[0, 0] == pytest.approx(1.0)
    assert unmatched == ["ol"]


def test_misfit_half_overlap():
    # obs all olivine; predicted half olivine half quartz -> 0.5
    results = [_res([("ol", .5, None, .5, False), ("q", .5, None, .5, False)])]
    grid, _, _ = _modal_misfit_grid(results, {"ol": 1.0}, 1, 1)
    assert grid[0, 0] == pytest.approx(0.5)


def test_misfit_excludes_melt_and_fluid():
    # solids renormalise to ol=1.0 after dropping melt + H2O -> perfect match
    results = [_res([("ol", .3, None, .3, False),
                     ("liq", .5, None, .5, True),
                     ("H2O", .2, None, .2, False)])]
    grid, _, _ = _modal_misfit_grid(results, {"ol": 1.0}, 1, 1)
    assert grid[0, 0] == pytest.approx(0.0)


def test_misfit_basis_fallback_to_wt():
    # one phase lacks vol_fraction -> the cell falls back to the wt basis
    results = [_res([("ol", None, .6, .5, False), ("cpx", .4, .4, .5, False)])]
    grid, basis, _ = _modal_misfit_grid(results, {"ol": .6, "cpx": .4}, 1, 1)
    assert grid[0, 0] == pytest.approx(0.0)
    assert basis == "wt"


def test_misfit_nan_when_no_fractions():
    results = [_res([("ol", None, None, None, False)])]
    grid, basis, _ = _modal_misfit_grid(results, {"ol": 1.0}, 1, 1)
    assert np.isnan(grid[0, 0])
    assert basis == "none"


def test_misfit_alias_matching_and_solvus_split():
    # Perple_X garnet name matches "g"; one fsp splits between observed pl+kf
    results = [_res([("Gt(W)", .5, None, .5, False), ("fsp", .5, None, .5, False)])]
    grid, _, unmatched = _modal_misfit_grid(
        results, {"g": .5, "pl": .25, "kf": .25}, 1, 1)
    assert grid[0, 0] == pytest.approx(0.0)
    assert unmatched == []


def test_misfit_observed_renormalised():
    # observed modes given as pixel counts (not fractions) work unchanged
    results = [_res([("ol", .75, None, .75, False), ("q", .25, None, .25, False)])]
    grid, _, _ = _modal_misfit_grid(results, {"ol": 300.0, "q": 100.0}, 1, 1)
    assert grid[0, 0] == pytest.approx(0.0)


def test_misfit_empty_observed():
    results = [_res([("ol", 1.0, None, 1.0, False)])]
    grid, basis, _ = _modal_misfit_grid(results, {}, 1, 1)
    assert np.isnan(grid[0, 0])


# ---------------------------------------------------------------------------
# Relaxed-assumption misfit: weights / presence / open assemblage (2026-06-23)
# ---------------------------------------------------------------------------

def test_misfit_weights_downweight_shaky_phase():
    # obs equal thirds; predicted over-represents ol (largest error). Down-
    # weighting ol (a shaky ID) must lower the misfit; uniform weights == default.
    results = [_res([("ol", .6, None, .6, False),
                     ("cpx", .3, None, .3, False),
                     ("q", .1, None, .1, False)])]
    obs = {"ol": 1 / 3, "cpx": 1 / 3, "q": 1 / 3}
    m_eq = _modal_misfit_grid(results, obs, 1, 1)[0][0, 0]
    m_uniform = _modal_misfit_grid(
        results, obs, 1, 1, weights={"ol": 1.0, "cpx": 1.0, "q": 1.0})[0][0, 0]
    m_dw = _modal_misfit_grid(results, obs, 1, 1, weights={"ol": 0.05})[0][0, 0]
    assert m_uniform == pytest.approx(m_eq)      # uniform reproduces unweighted
    assert m_dw < m_eq


def test_misfit_presence_constraint():
    present = [_res([("ol", .8, None, .8, False), ("ru", .2, None, .2, False)])]
    absent = [_res([("ol", 1.0, None, 1.0, False)])]
    obs = {"ol": 1.0}
    # ru as presence-only: its mass is set aside (not penalised as "other"),
    # and it's present here -> perfect proportion match on ol.
    g_present = _modal_misfit_grid(present, obs, 1, 1, presence_keys=["ru"])[0][0, 0]
    assert g_present == pytest.approx(0.0)
    # ru absent -> fixed presence penalty (default 0.15) and reported unmatched.
    g_absent, _, unm = _modal_misfit_grid(absent, obs, 1, 1, presence_keys=["ru"])
    assert g_absent[0, 0] == pytest.approx(0.15)
    assert "ru" in unm
    # Without the presence constraint the ru mass is the penalised "other".
    assert _modal_misfit_grid(present, obs, 1, 1)[0][0, 0] == pytest.approx(0.2)


def test_misfit_open_assemblage_forgives_minor_unobserved():
    # obs ol only; predicted ol .95 + a 5% phase segmentation didn't pick up.
    results = [_res([("ol", .95, None, .95, False),
                     ("ilm", .05, None, .05, False)])]
    obs = {"ol": 1.0}
    closed = _modal_misfit_grid(results, obs, 1, 1)[0][0, 0]          # tol=0
    assert closed == pytest.approx(0.05)                              # other=.05
    open_ = _modal_misfit_grid(results, obs, 1, 1, open_other_tol=0.05)[0][0, 0]
    assert open_ == pytest.approx(0.025)                              # other forgiven
    assert open_ < closed


# ---------------------------------------------------------------------------
# P-T-vs-scale reduction (_best_fit_pt + _pt_scale_rows)
# ---------------------------------------------------------------------------

def test_best_fit_pt_argmin_maps_to_axes():
    # 3x4 misfit grid; minimum at (i_p=2, j_t=1)
    n_p, n_t = 3, 4
    m = np.ones((n_p, n_t))
    m[2, 1] = 0.0
    pt = _best_fit_pt(m, (0.5, 1.5), (600.0, 900.0), n_p, n_t)
    # P axis linspace(0.5,1.5,3) -> [0.5,1.0,1.5]; T linspace(600,900,4) -> [600,700,800,900]
    assert pt == pytest.approx((1.5, 700.0))


def test_best_fit_pt_flat_and_allnan():
    n_p, n_t = 2, 2
    flat = _best_fit_pt(np.zeros((n_p, n_t)), (1.0, 2.0), (500.0, 600.0), n_p, n_t)
    assert flat == pytest.approx((1.0, 500.0))           # first finite argmin
    assert _best_fit_pt(np.full((n_p, n_t), np.nan), (1, 2), (5, 6), n_p, n_t) is None
    # accepts a flat (n_p*n_t,) misfit too
    flat2 = _best_fit_pt(np.array([1.0, 0.0, 1.0, 1.0]), (1.0, 2.0),
                         (500.0, 700.0), 2, 2)
    assert flat2 == pytest.approx((1.0, 700.0))          # idx1 -> (i_p=0, j_t=1)


def test_pt_scale_rows_best_and_envelope():
    # two levels, each with mean / +sigma / -sigma best-fit (P, T)
    pt_by_level = {
        0: {"mean": (1.0, 700.0), "mean+sigma": (1.0, 705.0),
            "mean-sigma": (1.0, 695.0)},
        2: {"mean": (1.1, 710.0), "mean+sigma": (1.3, 740.0),
            "mean-sigma": (0.9, 680.0)},
    }
    rows = _pt_scale_rows(pt_by_level, roi_h=4000.0, roi_w=4000.0)
    assert [r["window_px"] for r in rows] == sorted(
        [r["window_px"] for r in rows], reverse=True)         # largest first
    r0 = next(r for r in rows if r["window_px"] == pytest.approx(4000.0))
    assert r0["T_best_C"] == 700.0                            # mean variant
    assert (r0["T_lo_C"], r0["T_hi_C"]) == (695.0, 705.0)
    r2 = next(r for r in rows if r["window_px"] == pytest.approx(1000.0))  # L2: 4000/4
    assert r2["T_best_C"] == 710.0                            # mean variant
    assert (r2["T_lo_C"], r2["T_hi_C"]) == (680.0, 740.0)     # wider envelope at fine scale
    assert (r2["P_lo_GPa"], r2["P_hi_GPa"]) == (0.9, 1.3)


def test_pt_scale_rows_best_falls_back_without_mean():
    rows = _pt_scale_rows({1: {"mean+sigma": (1.2, 720.0)}}, 2000.0, 2000.0)
    assert len(rows) == 1
    assert rows[0]["T_best_C"] == 720.0 and rows[0]["P_best_GPa"] == 1.2


# ---------------------------------------------------------------------------
# Monte Carlo sampling + aggregation
# ---------------------------------------------------------------------------

def test_mc_sample_seeded_and_renormalised():
    comp = {"Si": 25.0, "Mg": 15.0, "Fe": 10.0}
    a = _mc_sample_compositions(comp, 0.05, 8, seed=1)
    b = _mc_sample_compositions(comp, 0.05, 8, seed=1)
    c = _mc_sample_compositions(comp, 0.05, 8, seed=2)
    assert len(a) == 8
    assert a == b                              # deterministic per seed
    assert a != c
    for s in a:
        assert sum(s.values()) == pytest.approx(50.0)   # total preserved
        assert all(v >= 0 for v in s.values())
        assert s != comp                       # actually perturbed


def test_mc_sample_per_component_absolute_sigma():
    comp = {"Si": 25.0, "Mg": 15.0, "Fe": 10.0}
    samples = _mc_sample_compositions(
        comp, {"Si": 5.0, "Mg": 0.0, "Fe": 0.0}, 50, seed=0)
    si = np.array([s["Si"] for s in samples])
    assert si.std() > 1.0                      # Si varies
    # Mg and Fe had sigma 0: renormalisation scales them identically, so
    # their ratio is exactly preserved while Si/Mg moves.
    for s in samples:
        assert s["Mg"] / s["Fe"] == pytest.approx(1.5)
    si_mg = np.array([s["Si"] / s["Mg"] for s in samples])
    assert si_mg.std() > 0.05


def test_mc_sample_scale_of_spread():
    comp = {"Si": 50.0, "Mg": 50.0}
    samples = _mc_sample_compositions(comp, 0.10, 400, seed=3)
    si = np.array([s["Si"] for s in samples])
    # relative σ 10% on two anticorrelated (renormalised) components
    assert 2.0 < si.std() < 8.0


def test_mc_aggregate_stability_probability():
    stable = _res([("g", .5, None, .5, False), ("ol", .5, None, .5, False)])
    unstable = _res([("ol", 1.0, None, 1.0, False)])
    grids = [[stable], [stable], [unstable], None]      # one failed run
    agg = _mc_aggregate(grids, 1, 1, observed_phases=["g"])
    assert agg["n_runs"] == 3
    assert agg["n_failed"] == 1
    assert agg["stability_prob"][0, 0] == pytest.approx(2 / 3)


def test_mc_aggregate_misfit_stats_and_best_fits():
    perfect = _res([("ol", 1.0, None, 1.0, False)])
    half = _res([("ol", .5, None, .5, False), ("q", .5, None, .5, False)])
    grids = [[perfect], [half]]
    agg = _mc_aggregate(grids, 1, 1, observed_modes={"ol": 1.0})
    assert agg["misfit_mean"][0, 0] == pytest.approx(0.25)
    assert agg["misfit_std"][0, 0] == pytest.approx(0.25)
    assert agg["best_fits"] == [(0, 0, 0.0), (0, 0, 0.5)]


def test_mc_aggregate_all_failed():
    agg = _mc_aggregate([None, [], None], 2, 2, observed_phases=["g"])
    assert agg["n_runs"] == 0
    assert "stability_prob" not in agg


def test_mc_sample_modes_seeded_and_on_simplex():
    obs = {"ol": 0.6, "cpx": 0.3, "q": 0.1}
    a = _mc_sample_modes(obs, 16, seed=1)
    b = _mc_sample_modes(obs, 16, seed=1)
    c = _mc_sample_modes(obs, 16, seed=2)
    assert a == b and a != c                       # deterministic per seed
    for s in a:
        assert sum(s.values()) == pytest.approx(1.0)   # stays on the simplex
        assert all(v >= 0.0 for v in s.values())
    # mean of many samples ~ the observed vector (unbiased proportions)
    big = _mc_sample_modes(obs, 4000, seed=0, rel_sigma=0.15)
    mean_ol = np.mean([s["ol"] for s in big])
    assert mean_ol == pytest.approx(0.6, abs=0.03)


def test_mc_aggregate_misfit_kwargs_forwarded():
    # open-assemblage tol forgives the unobserved q -> lower misfit than closed
    grids = [[_res([("ol", .95, None, .95, False), ("q", .05, None, .05, False)])]]
    closed = _mc_aggregate(grids, 1, 1, observed_modes={"ol": 1.0})
    opened = _mc_aggregate(grids, 1, 1, observed_modes={"ol": 1.0},
                           misfit_kwargs={"open_other_tol": 0.05})
    assert opened["misfit_mean"][0, 0] < closed["misfit_mean"][0, 0]


def test_mc_aggregate_mode_resampling_enlarges_cloud():
    grids = [[_res([("ol", .7, None, .7, False), ("cpx", .3, None, .3, False)])]]
    obs = {"ol": 0.7, "cpx": 0.3}
    base = _mc_aggregate(grids, 1, 1, observed_modes=obs)
    samples = _mc_sample_modes(obs, 10, seed=0)
    folded = _mc_aggregate(grids, 1, 1, observed_modes=obs, mode_samples=samples)
    # 1 grid -> base has 1 best-fit; folding 10 mode samples adds 10 more.
    assert len(base["best_fits"]) == 1
    assert len(folded["best_fits"]) == 1 + 10
    assert folded["mode_resampled"] == 10
