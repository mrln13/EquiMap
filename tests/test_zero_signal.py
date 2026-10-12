"""Zero-signal (no-data) pixel handling in phase segmentation.

Off-sample / unscanned areas hold no measurement, so they must be neither
nearest-neighbour filled from the sample edge nor kept as clustered labels
where smoothing smeared sample counts into the void.
"""
import numpy as np
import pytest

pytest.importorskip("qtpy")

from viewer_segment_helpers import _zero_signal_mask, _fill_unlabeled


def _maps(H=20, W=30):
    a = np.ones((H, W), np.float32)
    b = np.full((H, W), 2.0, np.float32)
    # right third of the map: no signal in ANY element
    a[:, 20:] = 0.0
    b[:, 20:] = 0.0
    # one element zero but the other not -> NOT zero-signal
    a[5, 5] = 0.0
    return {"Si": a, "Fe": b}


def test_zero_signal_mask_all_elements_zero():
    m = _zero_signal_mask(_maps())
    assert m.shape == (20, 30)
    assert m[:, 20:].all()          # the void
    assert not m[:, :20].any()      # sample, incl. the single-element zero
    assert not m[5, 5]


def test_zero_signal_mask_empty_raises():
    with pytest.raises(ValueError):
        _zero_signal_mask({})


def test_zero_region_protected_from_fill_but_holes_still_filled():
    H, W = 20, 30
    labels = np.zeros((H, W), np.int16)
    labels[:, 15:] = 1
    zero = np.zeros((H, W), bool)
    zero[:, 20:] = True             # off-sample void
    labels[zero] = -1
    labels[3, 3] = -1               # incidental hole inside the sample
    _fill_unlabeled(labels, protect=zero)
    assert labels[3, 3] == 0        # hole filled
    assert (labels[:, 20:] == -1).all()   # void stays unclassified


def test_legacy_behaviour_without_protection_fills_the_void():
    H, W = 10, 20
    labels = np.zeros((H, W), np.int16)
    labels[:, 10:] = -1
    _fill_unlabeled(labels, protect=None)
    assert (labels >= 0).all()
