# tests/test_pt_region.py
"""Near-minimal-region reduction for the P-T-vs-scale export.

`_pt_region_from_misfit` returns the argmin cell plus the span of all cells
within delta of the minimum; `_pt_scale_rows` unions those regions across
composition variants (and still accepts legacy (P, T) points).
"""
import numpy as np
import pytest

from viewer_thermo_helpers import _pt_region_from_misfit, _pt_scale_rows


def _grid():
    # 3x4 misfit grid: minimum at (p=1, t=2); a shallow valley within 0.02
    # covers t=1..3 at p=1, plus (p=0, t=2).
    m = np.full((3, 4), 1.0)
    m[1, 2] = 0.10
    m[1, 1] = 0.11
    m[1, 3] = 0.115
    m[0, 2] = 0.12
    return m


def test_region_argmin_and_span():
    r = _pt_region_from_misfit(_grid(), (0.2, 0.6), (400.0, 700.0), 3, 4,
                               delta=0.02)
    p_ax = np.linspace(0.2, 0.6, 3)
    t_ax = np.linspace(400.0, 700.0, 4)
    assert r["pt"] == (pytest.approx(p_ax[1]), pytest.approx(t_ax[2]))
    assert r["p_lo"] == pytest.approx(p_ax[0])
    assert r["p_hi"] == pytest.approx(p_ax[1])
    assert r["t_lo"] == pytest.approx(t_ax[1])
    assert r["t_hi"] == pytest.approx(t_ax[3])


def test_region_delta_zero_is_argmin_cell():
    r = _pt_region_from_misfit(_grid(), (0.2, 0.6), (400.0, 700.0), 3, 4,
                               delta=0.0)
    assert r["p_lo"] == r["p_hi"] == r["pt"][0]
    assert r["t_lo"] == r["t_hi"] == r["pt"][1]


def test_region_all_nan_is_none():
    m = np.full((2, 2), np.nan)
    assert _pt_region_from_misfit(m, (0, 1), (0, 1), 2, 2) is None


def test_region_ignores_nan_cells():
    m = _grid()
    m[2, :] = np.nan
    r = _pt_region_from_misfit(m, (0.2, 0.6), (400.0, 700.0), 3, 4)
    assert r is not None
    assert r["p_hi"] <= np.linspace(0.2, 0.6, 3)[1]


def test_scale_rows_union_of_regions():
    lvl = {0: {
        "mean": {"pt": (0.5, 600.0), "p_lo": 0.4, "p_hi": 0.6,
                 "t_lo": 550.0, "t_hi": 650.0},
        "mean+sigma": {"pt": (0.55, 620.0), "p_lo": 0.5, "p_hi": 0.7,
                       "t_lo": 600.0, "t_hi": 700.0},
    }}
    rows = _pt_scale_rows(lvl, roi_h=100.0, roi_w=100.0)
    assert len(rows) == 1
    r = rows[0]
    # best = mean variant's argmin point; envelope = union of the regions
    assert r["T_best_C"] == 600.0 and r["P_best_GPa"] == 0.5
    assert (r["T_lo_C"], r["T_hi_C"]) == (550.0, 700.0)
    assert (r["P_lo_GPa"], r["P_hi_GPa"]) == (0.4, 0.7)


def test_scale_rows_decompose_envelope_and_variant_best_cells():
    import math
    lvl = {0: {
        "mean": {"pt": (0.5, 600.0), "p_lo": 0.4, "p_hi": 0.6,
                 "t_lo": 550.0, "t_hi": 650.0},
        "mean+sigma": {"pt": (0.55, 620.0), "p_lo": 0.5, "p_hi": 0.7,
                       "t_lo": 600.0, "t_hi": 700.0},
        "mean-sigma": {"pt": (0.45, 580.0), "p_lo": 0.3, "p_hi": 0.5,
                       "t_lo": 500.0, "t_hi": 600.0},
    }}
    r = _pt_scale_rows(lvl, roi_h=100.0, roi_w=100.0)[0]
    # inner band = the mean variant's OWN region (valley flatness) ...
    assert (r["T_lo_mean_C"], r["T_hi_mean_C"]) == (550.0, 650.0)
    assert (r["P_lo_mean_GPa"], r["P_hi_mean_GPa"]) == (0.4, 0.6)
    # ... outer band = union over the variants (compositional spread)
    assert (r["T_lo_C"], r["T_hi_C"]) == (500.0, 700.0)
    assert (r["P_lo_GPa"], r["P_hi_GPa"]) == (0.3, 0.7)
    # per-variant best cells for the argmin-hopping markers
    assert (r["T_best_plus_C"], r["P_best_plus_GPa"]) == (620.0, 0.55)
    assert (r["T_best_minus_C"], r["P_best_minus_GPa"]) == (580.0, 0.45)
    # a missing variant leaves NaN, never a crash
    r2 = _pt_scale_rows({0: {"mean": lvl[0]["mean"]}}, 100.0, 100.0)[0]
    assert math.isnan(r2["T_best_plus_C"]) and math.isnan(r2["P_best_minus_GPa"])
    assert (r2["T_lo_mean_C"], r2["T_hi_mean_C"]) == (r2["T_lo_C"], r2["T_hi_C"])


def test_scale_rows_legacy_points_still_work():
    lvl = {0: {"mean": (0.5, 600.0), "mean-sigma": (0.6, 650.0)}}
    rows = _pt_scale_rows(lvl, roi_h=100.0, roi_w=100.0)
    r = rows[0]
    assert r["T_best_C"] == 600.0
    assert (r["T_lo_C"], r["T_hi_C"]) == (600.0, 650.0)
    assert (r["P_lo_GPa"], r["P_hi_GPa"]) == (0.5, 0.6)


def test_scale_rows_mixed_and_mean_absent():
    lvl = {1: {"mean+sigma": {"pt": (0.5, 600.0), "p_lo": 0.45, "p_hi": 0.55,
                              "t_lo": 580.0, "t_hi": 620.0},
               "mean-sigma": (0.7, 700.0)}}
    rows = _pt_scale_rows(lvl, roi_h=100.0, roi_w=100.0)
    r = rows[0]
    assert (r["T_lo_C"], r["T_hi_C"]) == (580.0, 700.0)
    assert (r["P_lo_GPa"], r["P_hi_GPa"]) == (0.45, 0.7)
