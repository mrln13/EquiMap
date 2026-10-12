# tests/test_segment_reject.py
"""Unclassified-pixel rejection in the GMM segmentation.

Covers:
- ambiguity rejection (max posterior below threshold -> label -1) hits
  midpoint "mixed signal" pixels but not cluster cores;
- atypicality rejection (mixture log-density percentile) hits far outliers
  that posteriors cannot catch;
- rejection off (defaults) reproduces the classic assign-every-pixel result;
- `_fill_unlabeled` fills incidental holes but never fills, nor sources from,
  protected (rejected) pixels;
- `_cluster_dominant_elements` skips negative labels so no "-1 domain" row
  reaches the classification dialog.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from viewer_segment_helpers import (
    _run_clustering,
    _fill_unlabeled,
    _cluster_dominant_elements,
)


def _two_blob_data(n=1000, n_mid=30, n_out=0, seed=0):
    """Two tight, well-separated blobs + optional midpoints and far outliers.

    Returns (X, slices) where slices identify the blob/mid/outlier rows.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal((0.0, 0.0), 0.5, size=(n, 2))
    b = rng.normal((10.0, 10.0), 0.5, size=(n, 2))
    mid = rng.normal((5.0, 5.0), 0.05, size=(n_mid, 2))
    out = rng.normal((40.0, -40.0), 0.5, size=(n_out, 2))
    X = np.vstack([a, b, mid, out]).astype(np.float32)
    sl = {
        "a": slice(0, n),
        "b": slice(n, 2 * n),
        "mid": slice(2 * n, 2 * n + n_mid),
        "out": slice(2 * n + n_mid, 2 * n + n_mid + n_out),
    }
    return X, sl


def test_no_rejection_by_default():
    X, _sl = _two_blob_data()
    labels, k, _bic, _icl, _m = _run_clustering(X, n_clusters=2)
    assert k == 2
    assert (labels >= 0).all()


def test_posterior_rejection_hits_midpoints_not_cores():
    # Two overlapping components (means 4 sigma apart) whose midline is
    # GENUINE ambiguity — a tight midpoint cluster between well-separated
    # blobs would instead be absorbed by one component's covariance and
    # confidently claimed (which is correct GMM behaviour, not a boundary
    # mix). Points on the midline have max posterior ~0.5.
    rng = np.random.default_rng(0)
    a = rng.normal((0.0, 0.0), 1.0, size=(1500, 2))
    b = rng.normal((4.0, 0.0), 1.0, size=(1500, 2))
    mid = rng.normal((2.0, 0.0), 0.05, size=(30, 2))
    X = np.vstack([a, b, mid]).astype(np.float32)
    # 0.7 rather than 0.6: the fitted decision boundary sits within ~0.1 of
    # the true midline, and the wider band keeps the assertion robust to that.
    labels, _k, _bic, _icl, _m = _run_clustering(
        X, n_clusters=2, posterior_min=0.7,
    )
    assert (labels[3000:] == -1).mean() > 0.9
    # Points within 1 sigma of either blob centre are unambiguous.
    core_a = np.linalg.norm(X[:1500] - (0.0, 0.0), axis=1) < 1.0
    core_b = np.linalg.norm(X[1500:3000] - (4.0, 0.0), axis=1) < 1.0
    assert (labels[:1500][core_a] >= 0).all()
    assert (labels[1500:3000][core_b] >= 0).all()
    # Exactly two real clusters remain.
    assert set(labels[labels >= 0]) == {0, 1}


def test_density_rejection_hits_outliers():
    X, sl = _two_blob_data(n_mid=0, n_out=8)
    labels, _k, _bic, _icl, _m = _run_clustering(
        X, n_clusters=2, density_pct=1.0,
    )
    # Far outliers get a confident posterior for their nearest component but
    # a rock-bottom mixture density -> the percentile cut removes them.
    assert (labels[sl["out"]] == -1).all()
    # The vast majority of real pixels survive a 1% cut.
    n_real = sl["b"].stop
    assert (labels[:n_real] >= 0).mean() > 0.97


def test_fill_unlabeled_respects_protection():
    lm = np.array(
        [[0, 0, -1, 1, 1],
         [0, -1, -1, -1, 1],
         [0, 0, -1, 1, 1]], dtype=np.int16)
    protect = np.zeros_like(lm, dtype=bool)
    protect[1, 2] = True
    _fill_unlabeled(lm, protect=protect)
    # Protected pixel stays unclassified; every other hole is filled.
    assert lm[1, 2] == -1
    assert (lm[~protect] >= 0).all()
    # Fills come from the nearest REAL label, never from a protected -1.
    assert lm[1, 1] == 0
    assert lm[1, 3] == 1


def test_fill_unlabeled_no_protection_fills_everything():
    lm = np.array([[0, -1, -1, 1]], dtype=np.int16)
    _fill_unlabeled(lm)
    assert (lm >= 0).all()


def test_fill_unlabeled_all_unlabeled_is_noop():
    lm = np.full((3, 3), -1, dtype=np.int16)
    _fill_unlabeled(lm, protect=np.zeros((3, 3), dtype=bool))
    assert (lm == -1).all()


def test_dominant_elements_skip_negative_labels():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
                  [5.0, 5.0], [5.0, 5.0]])
    labels = np.array([0, 0, 1, 1, -1, -1])
    dom = _cluster_dominant_elements(X, labels, ["A", "B"])
    assert set(dom.keys()) == {0, 1}
    assert dom[0].startswith("A")
    assert dom[1].startswith("B")
