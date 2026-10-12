"""Quadtree windowing + net-count fraction helpers (viewer_thermo_helpers)."""
import numpy as np

from viewer_thermo_helpers import (
    _net_count_fractions, _max_useful_quadtree_depth, _quadtree_windows,
    _quadtree_axis_segments, _quadtree_level_dims, _quadtree_sampled_windows,
    _semivariogram, _morans_i, _eligible_variogram_levels,
    _usable_windows_at_level, _pca_perturbed_compositions,
)


def _sum100(d):
    return abs(sum(d.values()) - 100.0) < 1e-6


def test_pca_sweep_single_window_no_spread():
    # One window (e.g. level 0 = whole ROI): no variation → trio collapses.
    w = [{"Si": 50.0, "Mg": 30.0, "Fe": 20.0}]
    mean, plus, minus = _pca_perturbed_compositions(w, ["Si", "Mg", "Fe"])
    assert mean == plus == minus
    assert _sum100(mean)


def test_pca_sweep_recovers_endpoints_and_closes():
    # Two windows that vary along a ratio direction; PC1 = that direction,
    # the ±1σ step reproduces the two windows, and every variant sums to 100.
    els = ["Si", "Mg"]
    w = [{"Si": 60.0, "Mg": 40.0}, {"Si": 40.0, "Mg": 60.0}]
    mean, plus, minus = _pca_perturbed_compositions(w, els)
    assert mean["Si"] == 50.0 and mean["Mg"] == 50.0
    for c in (mean, plus, minus):
        assert _sum100(c)
    # plus/minus are the two original windows (order depends on PC sign).
    got = sorted([round(plus["Si"], 6), round(minus["Si"], 6)])
    assert got == [40.0, 60.0]


def test_pca_sweep_magnitude_is_faithful():
    # The bug being fixed: under marginal-σ + renormalisation, a large vs small
    # spread can give the SAME modelled trio. With PC1 the bracket WIDTH scales
    # with the real spread.
    els = ["Si", "Mg", "Fe"]
    small = [{"Si": 52.0, "Mg": 30.0, "Fe": 18.0},
             {"Si": 48.0, "Mg": 30.0, "Fe": 22.0}]
    large = [{"Si": 65.0, "Mg": 30.0, "Fe": 5.0},
             {"Si": 35.0, "Mg": 30.0, "Fe": 35.0}]

    def _span(windows):
        _, p, m = _pca_perturbed_compositions(windows, els)
        return abs(p["Si"] - m["Si"]) + abs(p["Fe"] - m["Fe"])

    assert _span(large) > 3.0 * _span(small)


def test_pca_sweep_equal_cv_no_longer_degenerate():
    # Equal coefficient-of-variation between windows: the marginal-σ method
    # renormalises μ±σ back to μ exactly (the reported degeneracy). PC1 still
    # produces a real, non-degenerate ratio-changing bracket.
    els = ["Si", "Mg", "Fe"]
    # Closed windows that differ by changing Si:Fe while Mg holds — a genuine
    # ratio change, not a uniform rescaling.
    w = [{"Si": 55.0, "Mg": 30.0, "Fe": 15.0},
         {"Si": 45.0, "Mg": 30.0, "Fe": 25.0}]
    mean, plus, minus = _pca_perturbed_compositions(w, els)
    assert plus != mean and minus != mean
    assert plus["Si"] != minus["Si"]
    for c in (mean, plus, minus):
        assert _sum100(c)
        assert all(v >= 0.0 for v in c.values())


def test_pca_sweep_clips_negatives():
    # A large step that would push an element negative is clipped at 0, then
    # the vector is renormalised — still a valid composition.
    els = ["Si", "Fe"]
    w = [{"Si": 95.0, "Fe": 5.0}, {"Si": 5.0, "Fe": 95.0}]
    mean, plus, minus = _pca_perturbed_compositions(w, els, n_sigma=3.0)
    for c in (mean, plus, minus):
        assert _sum100(c)
        assert all(v >= 0.0 for v in c.values())


def test_max_useful_depth():
    assert _max_useful_quadtree_depth((0, 64, 0, 64), 4) == 4   # 64→32→16→8→4
    assert _max_useful_quadtree_depth((0, 3, 0, 3), 4) == 0     # below floor
    # non-square: limited by the short side
    assert _max_useful_quadtree_depth((0, 64, 0, 8), 4) == 1


def test_quadtree_level_counts_and_tiling():
    bbox = (0, 16, 0, 16)
    levels = _quadtree_windows(bbox, depth=2)
    assert levels[0] == [(0, 16, 0, 16)]
    assert len(levels[1]) == 4
    assert len(levels[2]) == 16
    # Level 1 quadrants tile the parent exactly, no overlap, full coverage.
    cover = np.zeros((16, 16), dtype=int)
    for (y0, y1, x0, x1) in levels[1]:
        cover[y0:y1, x0:x1] += 1
    assert (cover == 1).all()


def test_quadtree_stops_when_degenerate():
    # A 1-px-tall bbox can't be halved vertically → no level 1.
    levels = _quadtree_windows((0, 1, 0, 16), depth=3)
    assert set(levels.keys()) == {0}


def test_quadtree_level_dims_matches_materialised():
    # The analytic per-level (rows*cols, window size) must equal the materialised
    # quadtree wherever no degenerate split occurs (the dialog relies on this to
    # show counts without building 4**k windows).
    bbox = (0, 640, 0, 480)
    for k in range(5):
        rows, cols, (wh, ww) = _quadtree_level_dims(bbox, k)
        full = _quadtree_windows(bbox, depth=k)[k]
        assert rows * cols == len(full)
        wy0, wy1, wx0, wx1 = full[0]
        assert (wh, ww) == (wy1 - wy0, wx1 - wx0)


def test_quadtree_axis_segments_count_and_leaf():
    # 2**level segments while divisible; a too-small segment is kept as a leaf.
    assert len(_quadtree_axis_segments(0, 64, 3)) == 8
    assert _quadtree_axis_segments(0, 64, 3)[0] == (0, 8)
    assert _quadtree_axis_segments(0, 1, 5) == [(0, 1)]   # can't halve → leaf


def test_quadtree_sampled_windows_cap_and_bounds():
    bbox = (0, 800, 0, 600)
    # cap below the total → exactly cap windows, all inside the ROI
    wins = _quadtree_sampled_windows(bbox, 6, cap=50,
                                     rng=np.random.default_rng(0))
    assert len(wins) == 50
    for (y0, y1, x0, x1) in wins:
        assert 0 <= y0 < y1 <= 800 and 0 <= x0 < x1 <= 600
    # cap >= total → every window, identical to the materialised level
    full = _quadtree_windows(bbox, depth=2)[2]
    got = _quadtree_sampled_windows(bbox, 2, cap=10_000)
    assert sorted(got) == sorted(full)
    # deterministic "first N" (no rng) and reproducible random (seeded)
    a = _quadtree_sampled_windows(bbox, 5, cap=20)
    assert a == _quadtree_sampled_windows(bbox, 5, cap=20)
    r1 = _quadtree_sampled_windows(bbox, 5, cap=20, rng=np.random.default_rng(1))
    r2 = _quadtree_sampled_windows(bbox, 5, cap=20, rng=np.random.default_rng(1))
    assert r1 == r2


def test_quadtree_sampled_windows_random_is_spatially_uniform():
    # Regression (2026-10-02): the capped random sample truncated the SORTED
    # unique draws, keeping only the cells nearest the top of the ROI (the two
    # finest levels of the worked example covered 79 % and 57 % of the section).
    # A capped random sample must cover the whole ROI evenly, in both the
    # sparse (draw + dedup) and the dense (permutation) regime, and return
    # exactly ``cap`` distinct windows in scan order.
    bbox = (0, 26784, 0, 49670)
    for level, cap in ((8, 9192), (7, 8192), (6, 3000), (9, 300)):
        wins = _quadtree_sampled_windows(bbox, level, cap=cap,
                                         rng=np.random.default_rng(0))
        assert len(wins) == cap and len(set(wins)) == cap
        assert wins == sorted(wins)
        y0 = np.array([w[0] for w in wins], dtype=float)
        x0 = np.array([w[2] for w in wins], dtype=float)
        tol = 4.0 * 0.5 / np.sqrt(cap)       # 4 sigma of a fair coin share
        assert abs(np.mean(y0 < 26784 / 2) - 0.5) < tol
        assert abs(np.mean(x0 < 49670 / 2) - 0.5) < tol
        assert y0.max() > 0.9 * 26784 and x0.max() > 0.9 * 49670


def test_quadtree_sampled_windows_deep_is_fast_and_bounded():
    # The whole point: a deep level on a huge ROI must NOT materialise 4**level
    # windows. cap stays small and the call returns promptly.
    bbox = (0, 8000, 0, 6000)
    rows, cols, _ = _quadtree_level_dims(bbox, 11)
    assert rows * cols > 4_000_000           # would be catastrophic to build
    wins = _quadtree_sampled_windows(bbox, 11, cap=256,
                                     rng=np.random.default_rng(0))
    assert 0 < len(wins) <= 256
    for (y0, y1, x0, x1) in wins:
        assert 0 <= y0 < y1 <= 8000 and 0 <= x0 < x1 <= 6000


def test_net_count_fractions_formula():
    # net = main - 0.5*(sb_lo + sb_hi); two elements with known totals.
    H, W = 2, 2
    band_maps = {
        "Si": {"main": np.full((H, W), 10.0, np.float32),
               "sb_lo": np.full((H, W), 2.0, np.float32),
               "sb_hi": np.full((H, W), 2.0, np.float32)},   # net = 8 each → 32
        "Fe": {"main": np.full((H, W), 6.0, np.float32),
               "sb_lo": np.zeros((H, W), np.float32),
               "sb_hi": np.zeros((H, W), np.float32)},        # net = 6 each → 24
    }
    frac, n = _net_count_fractions(band_maps)
    assert n == 4
    assert abs(frac["Si"] - 32.0 / 56.0) < 1e-6
    assert abs(frac["Fe"] - 24.0 / 56.0) < 1e-6
    assert abs(sum(frac.values()) - 1.0) < 1e-6


def test_net_count_fractions_subbbox_and_mask():
    H, W = 4, 4
    band_maps = {
        "Si": {"main": np.ones((H, W), np.float32),
               "sb_lo": np.zeros((H, W), np.float32),
               "sb_hi": np.zeros((H, W), np.float32)},
    }
    sub = (0, 2, 0, 2)
    mask = np.array([[True, False], [False, False]])
    frac, n = _net_count_fractions(band_maps, sub_bbox=sub, sub_mask=mask)
    assert n == 1                       # one True pixel in the mask
    assert frac == {"Si": 1.0}          # single element renormalises to 1


def test_net_count_fractions_empty_signal():
    band_maps = {
        "Si": {"main": np.zeros((2, 2), np.float32),
               "sb_lo": np.zeros((2, 2), np.float32),
               "sb_hi": np.zeros((2, 2), np.float32)},
    }
    frac, n = _net_count_fractions(band_maps)
    assert frac == {}


# ---------------------------------------------------------------------------
# Spatial-structure metrics: semivariogram + Moran's I
# ---------------------------------------------------------------------------

def _grid_records(level, n, value_fn, win=10):
    """n×n regular grid of usable windows at `level`; wt_pct = value_fn(i, j)."""
    recs = []
    for i in range(n):
        for j in range(n):
            y0, x0 = i * win, j * win
            recs.append({"level": level, "bbox": (y0, y0 + win, x0, x0 + win),
                         "wt_pct": value_fn(i, j)})
    return recs


def _two_el(a):
    """Composition dict {A:a, B:100-a} (sums to 100)."""
    return {"A": float(a), "B": float(100.0 - a)}


def test_eligible_levels():
    recs = _grid_records(2, 4, lambda i, j: _two_el(50))   # 16 windows at L2
    recs += [{"level": 0, "bbox": (0, 40, 0, 40), "wt_pct": _two_el(50)}]  # 1 at L0
    recs += [{"level": 3, "bbox": (0, 5, 0, 5), "wt_pct": None}]           # skipped
    assert _eligible_variogram_levels(recs, min_windows=4) == [2]


def test_usable_windows_centers():
    recs = _grid_records(1, 2, lambda i, j: _two_el(10 * (i + j)), win=10)
    centers, comps = _usable_windows_at_level(recs, 1)
    assert centers.shape == (4, 2)
    assert len(comps) == 4
    # First window (i=0,j=0) spans (0,10,0,10) → centre (5,5).
    assert tuple(centers[0]) == (5.0, 5.0)


def test_semivariogram_too_few_windows():
    recs = _grid_records(0, 1, lambda i, j: _two_el(50))    # 1 window
    assert _semivariogram(recs, 0, ["A", "B"]) == {}


def test_semivariogram_constant_field_is_zero():
    recs = _grid_records(2, 5, lambda i, j: _two_el(40))    # uniform
    v = _semivariogram(recs, 2, ["A", "B"], n_bins=6)
    assert v["n_windows"] == 25
    finite = np.isfinite(v["gamma_total"])
    assert np.allclose(v["gamma_total"][finite], 0.0)


def test_semivariogram_total_equals_sum_of_elements():
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, 100, size=(6, 6))
    recs = _grid_records(2, 6, lambda i, j: _two_el(vals[i, j]))
    v = _semivariogram(recs, 2, ["A", "B"], n_bins=8)
    summed = v["gamma"]["A"] + v["gamma"]["B"]
    finite = np.isfinite(v["gamma_total"]) & np.isfinite(summed)
    assert np.allclose(v["gamma_total"][finite], summed[finite])


def test_semivariogram_gradient_increases_with_lag():
    # A rises with i+j → distant pairs differ more → γ grows with lag.
    recs = _grid_records(3, 8, lambda i, j: _two_el(min(100, 6 * (i + j))))
    v = _semivariogram(recs, 3, ["A", "B"], n_bins=8)
    g = v["gamma_total"]
    finite = np.where(np.isfinite(g))[0]
    assert finite.size >= 3
    assert g[finite[-1]] > g[finite[0]]


def test_morans_i_gradient_is_positive():
    recs = _grid_records(3, 8, lambda i, j: _two_el(min(100, 6 * (i + j))))
    I = _morans_i(recs, 3, ["A"])["A"]
    assert np.isfinite(I) and I > 0.2              # smooth gradient clusters


def test_morans_i_checkerboard_is_negative():
    recs = _grid_records(3, 8, lambda i, j: _two_el(100 if (i + j) % 2 == 0 else 0))
    I = _morans_i(recs, 3, ["A"])["A"]
    assert np.isfinite(I) and I < 0.0              # neighbours anti-correlate


def test_morans_i_random_is_near_zero():
    rng = np.random.default_rng(42)
    vals = rng.uniform(0, 100, size=(8, 8))
    recs = _grid_records(3, 8, lambda i, j: _two_el(vals[i, j]))
    I = _morans_i(recs, 3, ["A"])["A"]
    assert abs(I) < 0.1                            # no spatial pattern


def test_morans_i_too_few_windows_is_nan():
    recs = _grid_records(0, 1, lambda i, j: _two_el(50))    # 1 window
    assert np.isnan(_morans_i(recs, 0, ["A"])["A"])


def test_morans_i_constant_field_is_nan():
    recs = _grid_records(2, 5, lambda i, j: _two_el(40))    # zero variance
    assert np.isnan(_morans_i(recs, 2, ["A"])["A"])


def test_spatial_metrics_subsample_many_windows():
    # A deep quadtree can produce tens of thousands of windows; the O(N^2)
    # variogram / Moran's I must cap memory by subsampling rather than blow up.
    rng = np.random.default_rng(1)
    n = 200  # 200x200 = 40k windows
    vals = rng.uniform(0, 100, size=(n, n))
    recs = _grid_records(5, n, lambda i, j: _two_el(vals[i, j]))
    assert len(recs) == n * n

    v = _semivariogram(recs, 5, ["A", "B"], n_bins=8, max_points=1500)
    assert v["n_windows"] == n * n          # reports the true total
    assert v["n_used"] == 1500              # but only paired a capped sample
    assert np.isfinite(v["gamma_total"]).any()

    I = _morans_i(recs, 5, ["A"], max_points=1500)["A"]
    assert np.isfinite(I)

    # Subsampling is seeded → reproducible.
    assert _morans_i(recs, 5, ["A"], max_points=1500)["A"] == I
