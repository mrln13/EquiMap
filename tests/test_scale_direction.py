"""Directional semivariogram + representative-averaging-scale tolerance.

Pure helpers in ``viewer_thermo_helpers`` (pulls qtpy, no display needed):

* ``_semivariogram(direction_deg=)`` keeps only window pairs whose separation
  lies within ``tol_deg`` of the direction. On a field that varies along x
  only, the along-x curve rises with lag while the along-y curve is zero.
* ``_spread_by_level`` / ``_representative_level``: a window size is
  representative when EVERY element's between-window spread is at or below a
  tolerance (absolute per element, or a relative fraction of the mean), and
  the representative level is the deepest level that passes together with
  every coarser one.
"""
import math

import numpy as np
import pytest

pytest.importorskip("qtpy")
from viewer_thermo_helpers import (  # noqa: E402
    _semivariogram, _spread_by_level, _representative_level,
)


def _grid_records(level, n, value_fn, win=10):
    recs = []
    for i in range(n):
        for j in range(n):
            y0, x0 = i * win, j * win
            recs.append({"level": level, "bbox": (y0, y0 + win, x0, x0 + win),
                         "wt_pct": value_fn(i, j)})
    return recs


def _two_el(a):
    return {"A": float(a), "B": float(100.0 - a)}


def test_directional_variogram_separates_x_and_y():
    # Composition changes along x (column index j) only.
    recs = _grid_records(3, 8, lambda i, j: _two_el(10 * j))
    # A narrow sector keeps only the exactly-aligned pairs of a regular grid
    # (the default 22.5 degrees would admit near-diagonal pairs that do change
    # along x).
    vx = _semivariogram(recs, 3, ["A"], n_bins=6, direction_deg=0.0, tol_deg=5.0)
    vy = _semivariogram(recs, 3, ["A"], n_bins=6, direction_deg=90.0, tol_deg=5.0)
    assert vx["direction_deg"] == 0.0 and vy["direction_deg"] == 90.0
    gx = vx["gamma"]["A"]
    gy = vy["gamma"]["A"]
    ok = np.isfinite(gx)
    assert ok.sum() >= 3
    # Along x: gamma grows with lag (first finite bin < last finite bin).
    assert gx[ok][0] < gx[ok][-1]
    # Along y: nothing changes along y, so gamma is 0 in every populated bin.
    assert np.nanmax(gy) == pytest.approx(0.0, abs=1e-12)


def test_directional_variogram_is_modulo_180_and_subset_of_all_pairs():
    recs = _grid_records(2, 6, lambda i, j: _two_el(5 * (i + j)))
    v_all = _semivariogram(recs, 2, ["A"], n_bins=5)
    v0 = _semivariogram(recs, 2, ["A"], n_bins=5, direction_deg=0.0)
    v180 = _semivariogram(recs, 2, ["A"], n_bins=5, direction_deg=180.0)
    # A pair has no orientation: 0 and 180 degrees are the same sector.
    assert np.array_equal(v0["counts"], v180["counts"])
    assert v0["counts"].sum() < v_all["counts"].sum()
    assert v_all["direction_deg"] is None


def test_directional_variogram_empty_sector_returns_empty():
    # Four windows on one row: no pair lies along y.
    recs = [{"level": 1, "bbox": (0, 10, 10 * j, 10 * j + 10), "wt_pct": _two_el(j)}
            for j in range(4)]
    assert _semivariogram(recs, 1, ["A"], direction_deg=90.0, tol_deg=10.0) == {}


def _spread_fixture():
    # Spread halves at every coarser level: L3 8, L2 4, L1 2 (wt%) for A;
    # B is always small.
    return {1: {"A": 2.0, "B": 0.2}, 2: {"A": 4.0, "B": 0.4}, 3: {"A": 8.0, "B": 0.8}}


def test_representative_level_absolute_tolerance():
    best, ratios = _representative_level(_spread_fixture(), {"A": 4.5, "B": 1.0})
    assert best == 2                      # L1 and L2 pass, L3 fails on A
    assert ratios[3]["A"] == pytest.approx(8.0 / 4.5)
    assert ratios[1]["B"] == pytest.approx(0.2)


def test_representative_level_not_reached_and_untested_elements():
    best, ratios = _representative_level(_spread_fixture(), {"A": 1.0})
    assert best is None                   # even the coarsest level fails
    assert "B" not in ratios[1]           # no tolerance for B: not tested
    best2, _ = _representative_level(_spread_fixture(), {"C": 1.0})
    assert best2 is None                  # nothing testable


def test_representative_level_relative_tolerance_needs_means():
    means = {lv: {"A": 50.0, "B": 10.0} for lv in (1, 2, 3)}
    best, ratios = _representative_level(_spread_fixture(), 0.10, means=means)
    # 10 % of 50 = 5 wt% for A: L1 (2) and L2 (4) pass, L3 (8) fails.
    assert best == 2
    assert ratios[2]["A"] == pytest.approx(4.0 / 5.0)
    assert _representative_level(_spread_fixture(), 0.10)[0] is None


def test_representative_level_requires_every_coarser_level_to_pass():
    spread = {1: {"A": 9.0}, 2: {"A": 1.0}, 3: {"A": 1.0}}   # noisy coarse level
    best, _ = _representative_level(spread, {"A": 2.0})
    assert best is None


def test_spread_by_level_matches_sample_std():
    recs = _grid_records(1, 2, lambda i, j: _two_el(10 * (i + j)))   # A = 0,10,10,20
    recs += [{"level": 2, "bbox": (0, 5, 0, 5), "wt_pct": _two_el(1)}]  # 1 window
    sp = _spread_by_level(recs, ["A", "B"], ddof=1)
    assert set(sp) == {1}                                   # L2 has < 2 windows
    assert sp[1]["A"] == pytest.approx(np.std([0, 10, 10, 20], ddof=1))
    assert sp[1]["B"] == pytest.approx(sp[1]["A"])
    sp0 = _spread_by_level(recs, ["A"], ddof=0)
    assert sp0[1]["A"] == pytest.approx(np.std([0, 10, 10, 20], ddof=0))
    assert math.isfinite(sp0[1]["A"])
