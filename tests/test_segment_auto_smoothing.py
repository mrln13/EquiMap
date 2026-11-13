"""Count-based smoothing for phase segmentation
(viewer_segment_helpers._auto_smoothing / _min_region_for_sigma).

The Gaussian pools ~4*pi*sigma**2 pixels, so the sigma that pools N counts is
sqrt(N / (4*pi*rate)): it depends on the counts per pixel, not the pixel size.
Opt-in; the dialog's fixed sigma stays the default.
"""
import math

import numpy as np
import pytest

pytest.importorskip("qtpy", reason="viewer_segment_helpers sits beside viewer modules")

from viewer_segment_helpers import (AUTO_SMOOTH_TARGET_COUNTS,  # noqa: E402
                                    _auto_smoothing, _min_region_for_sigma)


def _poisson_maps(rates, shape=(200, 200), seed=0):
    rng = np.random.default_rng(seed)
    return {el: rng.poisson(r, shape).astype(np.float32) for el, r in rates.items()}


def test_sigma_pools_the_target_counts():
    maps = _poisson_maps({"Si": 4.0, "Al": 3.0, "K": 2.0})      # 9 counts per pixel
    res = _auto_smoothing(maps)
    assert res["counts_per_px"] == pytest.approx(9.0, abs=0.5)
    expected = math.sqrt(AUTO_SMOOTH_TARGET_COUNTS / (4 * math.pi * 9.0))
    assert res["sigma"] == pytest.approx(expected, abs=0.1)
    assert 4 * math.pi * res["sigma"] ** 2 * res["counts_per_px"] == pytest.approx(
        AUTO_SMOOTH_TARGET_COUNTS, rel=0.1)


def test_more_counts_less_smoothing():
    low = _auto_smoothing(_poisson_maps({"Si": 2.0, "Al": 2.0}))
    high = _auto_smoothing(_poisson_maps({"Si": 50.0, "Al": 50.0}))
    assert high["sigma"] < low["sigma"]
    # 2x2 binning = 4x the counts per pixel = half the sigma in pixels, the
    # same physical smoothing
    binned = _auto_smoothing(_poisson_maps({"Si": 8.0, "Al": 8.0}))
    assert binned["sigma"] == pytest.approx(low["sigma"] / 2, abs=0.1)


def test_coating_and_void_do_not_count():
    maps = _poisson_maps({"Si": 4.0, "Al": 4.0})
    base = _auto_smoothing(maps)
    with_c = dict(maps, C=np.full((200, 200), 40.0, np.float32))
    assert _auto_smoothing(with_c)["sigma"] == base["sigma"]   # C excluded by default
    padded = {el: np.pad(a, ((0, 200), (0, 0))) for el, a in maps.items()}  # half void
    assert _auto_smoothing(padded)["sigma"] == base["sigma"]


def test_starved_and_rich_maps_are_clipped():
    starved = {"Si": np.zeros((50, 50), np.float32) - 1.0}     # signed nets, all negative
    assert _auto_smoothing(starved, valid_pixels=np.ones((50, 50), bool))["sigma"] == 10.0
    rich = _poisson_maps({"Si": 5000.0})
    assert _auto_smoothing(rich)["sigma"] == 0.5


def test_min_region_follows_sigma():
    assert _min_region_for_sigma(0) == 0
    assert _min_region_for_sigma(2.5) == 27            # the paper used 30 at sigma 2.5
    assert _min_region_for_sigma(3.0) == 39
    assert _min_region_for_sigma(6.0) == 4 * _min_region_for_sigma(3.0) + 1
    res = _auto_smoothing(_poisson_maps({"Si": 9.0}))
    assert res["min_region_size"] == _min_region_for_sigma(res["sigma"])


def test_needs_an_element():
    with pytest.raises(ValueError):
        _auto_smoothing({"C": np.ones((5, 5), np.float32)})
