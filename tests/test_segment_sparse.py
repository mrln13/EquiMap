"""Sparse-element auto-exclusion for phase segmentation
(viewer_segment_helpers._sparse_element_exclusions).

The criterion is smoothing-aware: on low-dose maps the raw zero fraction is
~exp(-rate) (a hidden count-rate cutoff), so an element is excluded only when
it is BOTH raw-sparse AND still count-starved after the Gaussian pooling of
~4*pi*sigma**2 pixels. Regression for the C23S1A failure where Mg/Na/Ca
(sparse but perfectly usable after sigma=3 smoothing) were silently discarded
alongside genuine contamination, collapsing the segmentation onto K-Al-Si
blends.
"""
import numpy as np
import pytest

pytest.importorskip("qtpy", reason="viewer_segment_helpers is Qt-free but "
                    "sits beside viewer modules; skip if env lacks GUI deps")

from viewer_segment_helpers import _sparse_element_exclusions


def _poisson_map(rate, shape=(200, 200), seed=0):
    return np.random.default_rng(seed).poisson(rate, size=shape).astype(np.float32)


def _maps():
    return {
        "Si": _poisson_map(3.0, seed=1),      # dense major element
        "Mg": _poisson_map(0.1, seed=2),      # sparse but usable after smoothing
        "C": _poisson_map(0.001, seed=3),     # contamination-level rate
    }


def _valid(maps):
    return np.ones(next(iter(maps.values())).shape, dtype=bool)


def test_contamination_excluded_sparse_discriminator_kept():
    maps = _maps()
    excluded, report = _sparse_element_exclusions(
        maps, _valid(maps), max_zero_pct=90.0, sigma=3.0)
    assert excluded == ["C"]
    rows = {el: (zf, pooled, drop) for el, zf, pooled, drop in report}
    # Mg: ~90% zeros (exp(-0.1)) but ~11 pooled counts at sigma=3 -> kept.
    assert rows["Mg"][0] > 85.0 and rows["Mg"][1] > 5.0 and not rows["Mg"][2]
    # C: sparse AND count-starved after pooling -> dropped.
    assert rows["C"][1] < 1.0 and rows["C"][2]
    assert not rows["Si"][2]


def test_sigma_zero_reduces_to_plain_zero_fraction_cut():
    # No smoothing -> no pooling rescue: the sparse-but-real element goes too
    # (clustering raw counts on a 90%-zero map genuinely makes noise clusters).
    maps = _maps()
    excluded, _ = _sparse_element_exclusions(
        maps, _valid(maps), max_zero_pct=90.0, sigma=0.0)
    assert set(excluded) == {"Mg", "C"}


def test_threshold_100_excludes_nothing():
    maps = _maps()
    excluded, _ = _sparse_element_exclusions(
        maps, _valid(maps), max_zero_pct=100.0, sigma=3.0)
    assert excluded == []


def test_valid_mask_restricts_statistics():
    # An element dense inside the valid region but absent outside must be
    # judged on the valid region only.
    maps = {"Si": np.zeros((100, 100), dtype=np.float32)}
    maps["Si"][:50, :] = 3.0
    valid = np.zeros((100, 100), dtype=bool)
    valid[:50, :] = True
    excluded, report = _sparse_element_exclusions(
        maps, valid, max_zero_pct=90.0, sigma=3.0)
    assert excluded == []
    assert report[0][2] > 100.0  # pooled counts computed over valid region
