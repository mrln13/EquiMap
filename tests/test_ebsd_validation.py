"""EBSD-vs-EDS confusion matrix (ebsd_validation.py). Qt-free."""
import numpy as np
import pytest

import ebsd_validation as ev


def test_crop_ebsd_to_phase_grid_alignment():
    full = np.zeros((20, 30), dtype=np.uint8)
    full[4:10, 6:16] = 3                      # one indexed phase inside the ROI
    roi = (4, 12, 6, 18)                      # 8 x 12
    out = ev.crop_ebsd_to_phase_grid(full, roi, ds=2, phase_shape=(4, 6))
    assert out is not None and out.shape == (4, 6)
    assert out[0, 0] == 3                     # (4,6) in mosaic coords
    # shape mismatch -> None
    assert ev.crop_ebsd_to_phase_grid(full, roi, 2, (10, 10)) is None


def test_crop_applies_polygon_mask():
    full = np.full((8, 8), 2, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True
    out = ev.crop_ebsd_to_phase_grid(full, (0, 4, 0, 4), 1, (4, 4),
                                     roi_mask_full=mask)
    assert (out[:, :2] == 2).all()
    assert (out[:, 2:] == 0).all()            # masked-out -> unindexed


def _maps():
    """EBSD truth: top half phase 1 (Diopside), bottom half phase 2 (Chlorite).
    EDS map: matches except a 2-row band of the chlorite half called cpx."""
    eb = np.zeros((10, 10), dtype=np.uint8)
    eb[:5] = 1
    eb[5:] = 2
    pm = np.full((10, 10), -1, dtype=np.int16)
    pm[:5] = 0          # cid 0 = "Clinopyroxene"
    pm[5:] = 1          # cid 1 = "Chlorite"
    pm[5:7] = 0         # 20 px misclassified as cpx
    return eb, pm


def test_confusion_counts_and_accuracy():
    eb, pm = _maps()
    res = ev.confusion_matrix(
        eb, pm, {1: "Diopside", 2: "Chlorite"},
        {0: "Clinopyroxene", 1: "Chlorite"})
    assert res.row_labels == ["Diopside", "Chlorite"]
    assert res.col_labels == ["Clinopyroxene", "Chlorite"]
    assert res.counts.tolist() == [[50, 0], [20, 30]]
    assert res.n_both == 100
    # alias matching: Diopside<->Clinopyroxene (cpx group), Chlorite<->Chlorite
    assert res.matched_cols == {0: [0], 1: [1]}
    assert res.accuracy == pytest.approx(0.80)
    assert res.per_row[0]["recall"] == pytest.approx(1.0)
    assert res.per_row[1]["recall"] == pytest.approx(0.60)
    assert res.per_col[0]["precision"] == pytest.approx(50 / 70)
    assert res.per_col[1]["precision"] == pytest.approx(1.0)


def test_confusion_unmatched_phase_counts_as_wrong():
    eb = np.ones((4, 4), dtype=np.uint8)          # all Ilmenite (truth)
    pm = np.zeros((4, 4), dtype=np.int16)         # all called "Olivine"
    res = ev.confusion_matrix(eb, pm, {1: "Ilmenite"}, {0: "Olivine"})
    assert res.accuracy == pytest.approx(0.0)
    assert res.per_row[0]["recall"] is None       # no matching EDS domain at all
    assert res.matched_cols == {}


def test_confusion_one_sided_pixels_excluded():
    eb = np.zeros((4, 4), dtype=np.uint8)
    eb[:2] = 1                                    # indexed top half
    pm = np.full((4, 4), -1, dtype=np.int16)
    pm[:, :2] = 0                                 # EDS domain left half
    res = ev.confusion_matrix(eb, pm, {1: "Olivine"}, {0: "Olivine"})
    assert res.n_both == 4                        # top-left quadrant only
    assert res.n_ebsd_only == 4                   # top-right
    assert res.n_eds_only == 4                    # bottom-left
    assert res.accuracy == pytest.approx(1.0)


def test_confusion_multiple_eds_domains_match_one_phase():
    # two GMM clusters that are both chlorite both count as correct
    eb = np.ones((4, 4), dtype=np.uint8)
    pm = np.zeros((4, 4), dtype=np.int16)
    pm[:, 2:] = 1
    res = ev.confusion_matrix(eb, pm, {1: "Chlorite"},
                              {0: "Chlorite", 1: "Chl"})
    assert res.matched_cols == {0: [0, 1]}
    assert res.accuracy == pytest.approx(1.0)


def test_row_normalised_and_csv_rows():
    eb, pm = _maps()
    res = ev.confusion_matrix(
        eb, pm, {1: "Diopside", 2: "Chlorite"},
        {0: "Clinopyroxene", 1: "Chlorite"})
    rn = res.row_normalised()
    assert rn[1].tolist() == pytest.approx([0.4, 0.6])
    rows = ev.confusion_to_csv_rows(res)
    assert rows[0] == ["EBSD \\ EDS", "Clinopyroxene", "Chlorite", "row total"]
    assert rows[1] == ["Diopside", "50", "0", "50"]
    flat = ["|".join(r) for r in rows]
    assert any(r.startswith("alias-matched accuracy|0.8") for r in flat)


def test_grid_mismatch_raises():
    with pytest.raises(ValueError):
        ev.confusion_matrix(np.zeros((2, 2)), np.zeros((3, 3)), {}, {})
