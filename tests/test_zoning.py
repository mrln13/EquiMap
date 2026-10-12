"""Zoning index maps + traverse profiles (zoning.py). Qt-free."""
import numpy as np
import pytest

import zoning as z
from thermo.composition import ATOMIC_WEIGHTS


def _band(arr):
    arr = np.asarray(arr, dtype=np.float32)
    zz = np.zeros_like(arr)
    return {"main": arr, "sb_lo": zz, "sb_hi": zz}


def _wt_bands(wt_by_el, shape=(4, 4)):
    """Band maps whose net counts ARE wt% (k=1 ground truth)."""
    return {el: _band(np.full(shape, w, np.float32))
            for el, w in wt_by_el.items()}


def test_mgnum_from_known_wt():
    # Fo80-ish: Mg# = (Mg/A_Mg) / (Mg/A_Mg + Fe/A_Fe) with wt inputs
    bm = _wt_bands({"Mg": 20.0, "Fe": 10.0, "Si": 20.0})
    m = z.index_map(bm, "Mg#")
    mg = 20.0 / ATOMIC_WEIGHTS["Mg"]
    fe = 10.0 / ATOMIC_WEIGHTS["Fe"]
    assert np.allclose(m, mg / (mg + fe))


def test_index_uses_k_factors():
    # raw counts with k-factors equal to the wt conversion give the true Mg#
    counts = {"Mg": 1000.0, "Fe": 500.0}
    k = {"Mg": 0.02, "Fe": 0.02}     # wt = k * counts -> 20 / 10 wt
    bm = _wt_bands(counts)
    m = z.index_map(bm, "Mg#", k_factors=k)
    mg = (0.02 * 1000) / ATOMIC_WEIGHTS["Mg"]
    fe = (0.02 * 500) / ATOMIC_WEIGHTS["Fe"]
    assert np.allclose(m, mg / (mg + fe))


def test_an_with_missing_k_is_nan_free_and_bounded():
    bm = _wt_bands({"Ca": 10.0, "Na": 5.0, "K": 1.0})
    m = z.index_map(bm, "An")
    assert np.isfinite(m).all()
    assert ((m > 0) & (m < 1)).all()
    # An + Or + (Ab implied) — Or uses the same denominator
    o = z.index_map(bm, "Or")
    assert (m + o <= 1.0 + 1e-6).all()


def test_index_missing_numerator_is_zero_missing_denominator_nan():
    bm = _wt_bands({"Ca": 10.0, "Na": 5.0})     # no K
    m = z.index_map(bm, "Or")                   # numerator K missing -> 0
    assert np.allclose(m, 0.0)
    bm2 = _wt_bands({"Si": 10.0})               # no Ca/Na/K at all
    m2 = z.index_map(bm2, "An")
    assert np.isnan(m2).all()


def test_index_sub_mask_and_zero_pixels():
    bm = {"Mg": _band([[10.0, 0.0], [10.0, 10.0]]),
          "Fe": _band([[10.0, 0.0], [10.0, 10.0]])}
    mask = np.array([[True, True], [False, True]])
    m = z.index_map(bm, "Mg#", sub_mask=mask)
    assert m[0, 0] == pytest.approx(
        (10 / ATOMIC_WEIGHTS["Mg"]) / (10 / ATOMIC_WEIGHTS["Mg"]
                                       + 10 / ATOMIC_WEIGHTS["Fe"]))
    assert np.isnan(m[0, 1])        # zero signal
    assert np.isnan(m[1, 0])        # masked out


def test_index_unknown_raises():
    with pytest.raises(KeyError):
        z.index_map(_wt_bands({"Mg": 1.0}), "Nope#")


def test_smoothing_stabilises_ratio():
    rng = np.random.default_rng(0)
    base = np.full((32, 32), 100.0)
    bm = {"Mg": _band(base + rng.normal(0, 30, base.shape)),
          "Fe": _band(base + rng.normal(0, 30, base.shape))}
    raw = z.index_map(bm, "Mg#")
    sm = z.index_map(bm, "Mg#", smooth_sigma=2.0)
    assert np.nanstd(sm) < np.nanstd(raw) * 0.5


def test_element_map_percent_and_calibrated():
    bm = _wt_bands({"Si": 30.0, "Mg": 10.0})
    m = z.element_map(bm, "Si")
    assert np.allclose(m, 75.0)
    # calibrated: k reweights the share
    k = {"Si": 1.0, "Mg": 3.0}
    mc = z.element_map(bm, "Si", k_factors=k)
    assert np.allclose(mc, 100.0 * 30.0 / (30.0 + 30.0))


def test_element_map_uncalibrated_target_falls_back_to_raw():
    bm = _wt_bands({"Si": 30.0, "Mg": 10.0})
    k = {"Mg": 3.0}                  # Si has no k -> raw mode for consistency
    m = z.element_map(bm, "Si", k_factors=k)
    assert np.allclose(m, 75.0)


def test_line_profile_linear_gradient():
    g = np.tile(np.arange(50, dtype=float), (50, 1))     # value == x
    dist, vals = z.line_profile(g, (25, 5), (25, 45), n_samples=41)
    assert dist[0] == pytest.approx(0.0)
    assert dist[-1] == pytest.approx(40.0)
    assert vals[0] == pytest.approx(5.0)
    assert vals[-1] == pytest.approx(45.0)
    assert np.allclose(np.diff(vals), 1.0, atol=1e-6)


def test_line_profile_width_averages_across_normal():
    g = np.tile(np.arange(50, dtype=float)[:, None], (1, 50))   # value == y
    # horizontal line at y=25; width 5 averages y in 23..27 -> still 25
    _, vals = z.line_profile(g, (25, 5), (25, 45), n_samples=10, width=5)
    assert np.allclose(vals, 25.0)


def test_line_profile_nan_outside_map():
    g = np.ones((10, 10), dtype=float)
    dist, vals = z.line_profile(g, (5, -5), (5, 5), n_samples=21)
    assert np.isnan(vals[0])         # off-map start
    assert vals[-1] == pytest.approx(1.0)
