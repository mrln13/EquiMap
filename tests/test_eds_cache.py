"""eds_cache.roi_from_cache slicing/normalisation + is_valid_for fingerprinting."""
from pathlib import Path

import numpy as np
import pytest

import eds_cache as edsc


def _build_cache(cache_dir: Path, H, W, elements, weight_val=1.0,
                 data_type="oxford_wt_pct", fingerprint="fp0"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    w = np.lib.format.open_memmap(edsc._weight_path(cache_dir), mode="w+",
                                  dtype=np.float32, shape=(H, W))
    w[:] = weight_val
    w.flush()
    patterns = {}
    for el in elements:
        base = (np.arange(H * W).reshape(H, W).astype(np.float32)
                + (10.0 if el == elements[0] else 100.0))
        patterns[el] = base
        for band, arr in (("main", base),
                          ("sb_lo", np.full((H, W), 2.0, np.float32)),
                          ("sb_hi", np.full((H, W), 4.0, np.float32))):
            mm = np.lib.format.open_memmap(edsc._element_path(cache_dir, el, band),
                                           mode="w+", dtype=np.float32, shape=(H, W))
            mm[:] = arr
            mm.flush()
    edsc.save_manifest(cache_dir, {
        "stitch_fingerprint": fingerprint, "elements": list(elements),
        "ds": 1, "H": H, "W": W, "data_type": data_type, "width_eV": 0.0,
    })
    return patterns


def test_roi_mean_divides_by_weight(tmp_path):
    H, W = 6, 8
    pats = _build_cache(tmp_path / "c", H, W, ["Si", "Fe"], weight_val=2.0)
    maps = edsc.roi_from_cache(tmp_path / "c", (0, H, 0, W), ["Si"],
                               blend_mode="mean", progress=False)
    assert np.allclose(maps["Si"]["main"], pats["Si"] / 2.0)
    assert np.allclose(maps["Si"]["sb_lo"], 2.0 / 2.0)


def test_roi_sum_keeps_raw(tmp_path):
    H, W = 6, 8
    pats = _build_cache(tmp_path / "c", H, W, ["Si"], weight_val=2.0)
    maps = edsc.roi_from_cache(tmp_path / "c", (0, H, 0, W), ["Si"],
                               blend_mode="sum", progress=False)
    assert np.allclose(maps["Si"]["main"], pats["Si"])  # no /weight


def test_roi_interior_offset(tmp_path):
    H, W = 10, 12
    pats = _build_cache(tmp_path / "c", H, W, ["Si"], weight_val=1.0)
    y0, y1, x0, x1 = 3, 7, 2, 9
    maps = edsc.roi_from_cache(tmp_path / "c", (y0, y1, x0, x1), ["Si"],
                               blend_mode="mean", progress=False)
    assert maps["Si"]["main"].shape == (y1 - y0, x1 - x0)
    assert np.allclose(maps["Si"]["main"], pats["Si"][y0:y1, x0:x1])


def test_roi_return_weight(tmp_path):
    H, W = 5, 5
    _build_cache(tmp_path / "c", H, W, ["Si"], weight_val=3.0)
    maps, w = edsc.roi_from_cache(tmp_path / "c", (0, H, 0, W), ["Si"],
                                  return_weight=True, progress=False)
    assert w.shape == (H, W)
    assert np.allclose(w, 3.0)


def test_roi_out_ds_decimates(tmp_path):
    # out_ds>1 reads a strided (nearest-pixel) ROI to fit big ROIs in RAM.
    H, W = 12, 16
    pats = _build_cache(tmp_path / "c", H, W, ["Si"], weight_val=1.0)
    ds = 3
    maps = edsc.roi_from_cache(tmp_path / "c", (0, H, 0, W), ["Si"],
                               blend_mode="mean", out_ds=ds, progress=False)
    arr = maps["Si"]["main"]
    assert arr.shape == (len(range(0, H, ds)), len(range(0, W, ds)))
    # Values match the same stride applied directly to the source map.
    assert np.allclose(arr, pats["Si"][::ds, ::ds])


def test_roi_out_ds_with_offset_and_mask(tmp_path):
    H, W = 12, 16
    pats = _build_cache(tmp_path / "c", H, W, ["Si"], weight_val=1.0)
    y0, y1, x0, x1 = 2, 11, 1, 14
    ds = 2
    mask = np.ones((y1 - y0, x1 - x0), dtype=bool)   # full-res ROI mask
    maps = edsc.roi_from_cache(tmp_path / "c", (y0, y1, x0, x1), ["Si"],
                               blend_mode="mean", roi_mask=mask, out_ds=ds,
                               progress=False)
    arr = maps["Si"]["main"]
    expect = pats["Si"][y0:y1, x0:x1][::ds, ::ds]
    assert arr.shape == expect.shape
    assert np.allclose(arr, expect)


def test_roi_out_ds_1_is_unchanged(tmp_path):
    H, W = 8, 8
    pats = _build_cache(tmp_path / "c", H, W, ["Si"])
    a = edsc.roi_from_cache(tmp_path / "c", (0, H, 0, W), ["Si"],
                            blend_mode="mean", out_ds=1, progress=False)
    b = edsc.roi_from_cache(tmp_path / "c", (0, H, 0, W), ["Si"],
                            blend_mode="mean", progress=False)
    assert np.allclose(a["Si"]["main"], b["Si"]["main"])
    assert np.allclose(a["Si"]["main"], pats["Si"])


def test_roi_from_cache_ds_gt1_out_ds(tmp_path):
    # A ds>1 cache stores element maps coarser than the BSE (the two-resolution
    # Oxford montage). ROI coords are BSE-space; roi_from_cache maps them onto
    # the coarser cache (roi_from_cache_ds) and resizes back. out_ds must still
    # decimate the OUTPUT so the large-ROI memory guard works for ds>1 caches.
    H, W = 6, 8                       # coarse element-cache grid
    ds_store = 2
    H_bse, W_bse = H * ds_store, W * ds_store
    _build_cache(tmp_path / "c", H, W, ["Si"], weight_val=1.0)
    m = edsc.load_manifest(tmp_path / "c")
    m.update({"ds": ds_store, "H_bse": H_bse, "W_bse": W_bse})
    edsc.save_manifest(tmp_path / "c", m)

    roi = (0, H_bse, 0, W_bse)
    # out_ds=1 → output at full BSE-ROI resolution
    maps = edsc.roi_from_cache(tmp_path / "c", roi, ["Si"],
                               blend_mode="mean", progress=False)
    assert maps["Si"]["main"].shape == (H_bse, W_bse)
    # out_ds>1 → output decimated (matching roi_from_cache's convention)
    out_ds = 2
    maps2 = edsc.roi_from_cache(tmp_path / "c", roi, ["Si"], blend_mode="mean",
                                out_ds=out_ds, progress=False)
    assert maps2["Si"]["main"].shape == (len(range(0, H_bse, out_ds)),
                                         len(range(0, W_bse, out_ds)))
    # a full-res ROI mask is decimated to the (smaller) output grid, no crash
    mask = np.ones((H_bse, W_bse), dtype=bool)
    maps3 = edsc.roi_from_cache(tmp_path / "c", roi, ["Si"], blend_mode="mean",
                                roi_mask=mask, out_ds=out_ds, progress=False)
    assert maps3["Si"]["main"].shape == maps2["Si"]["main"].shape


def test_per_window_cache_read_matches_inram_slice(tmp_path):
    # The full-res composition-vs-scale path reads each quadtree window straight
    # from the cache. That must give the SAME net-count fractions as slicing the
    # same window out of the full in-RAM ROI maps (the in-memory path).
    from viewer_thermo_helpers import _net_count_fractions
    H, W = 16, 20
    _build_cache(tmp_path / "c", H, W, ["Si", "Fe", "Mg"], weight_val=1.0)
    roi = (2, 14, 3, 18)            # mosaic-coords ROI
    ry0, _, rx0, _ = roi
    full = edsc.roi_from_cache(tmp_path / "c", roi, ["Si", "Fe", "Mg"],
                               blend_mode="mean", progress=False)
    # A sub-window of that ROI, in mosaic coords.
    win = (5, 11, 6, 15)
    wy0, wy1, wx0, wx1 = win
    # In-RAM path: slice the full maps with ROI-local coords.
    sub_local = (wy0 - ry0, wy1 - ry0, wx0 - rx0, wx1 - rx0)
    frac_ram, n_ram = _net_count_fractions(full, sub_bbox=sub_local)
    # Cache path: read just the window, integrate the whole thing.
    bm_win = edsc.roi_from_cache(tmp_path / "c", win, ["Si", "Fe", "Mg"],
                                 blend_mode="mean", progress=False)
    frac_cache, n_cache = _net_count_fractions(bm_win, sub_bbox=None)
    assert n_ram == n_cache == (wy1 - wy0) * (wx1 - wx0)
    assert set(frac_ram) == set(frac_cache)
    for el in frac_ram:
        assert frac_ram[el] == pytest.approx(frac_cache[el])


def test_cache_window_reader_matches_roi_from_cache(tmp_path):
    # CacheWindowReader is the fast per-window path (open memmaps once); it must
    # give the same net-count fractions as roi_from_cache + _net_count_fractions.
    from viewer_thermo_helpers import _net_count_fractions
    H, W = 14, 18
    _build_cache(tmp_path / "c", H, W, ["Si", "Fe", "Mg"], weight_val=2.0)
    reader = edsc.CacheWindowReader(tmp_path / "c", ["Si", "Fe", "Mg"],
                                    blend_mode="mean")
    for win, mask in [((0, H, 0, W), None),
                      ((3, 11, 5, 16), None),
                      ((2, 12, 4, 14),
                       np.tri(10, 10, dtype=bool))]:
        bm = edsc.roi_from_cache(tmp_path / "c", win, ["Si", "Fe", "Mg"],
                                 blend_mode="mean", progress=False)
        frac_ref, _ = _net_count_fractions(bm, sub_bbox=None, sub_mask=mask)
        frac_fast = reader.net_fractions(win, mask)
        assert set(frac_ref) == set(frac_fast)
        for el in frac_ref:
            assert frac_fast[el] == pytest.approx(frac_ref[el])


def test_cache_window_reader_net_sums(tmp_path):
    # net_sums returns the RAW per-element totals (the k-factor fit input) and must
    # match roi_from_cache + _net_count_sums; net_fractions == net_sums renormalised.
    from viewer_thermo_helpers import _net_count_sums
    H, W = 14, 18
    _build_cache(tmp_path / "c", H, W, ["Si", "Fe", "Mg"], weight_val=2.0)
    reader = edsc.CacheWindowReader(tmp_path / "c", ["Si", "Fe", "Mg"],
                                    blend_mode="mean")
    for win, mask in [((0, H, 0, W), None),
                      ((3, 11, 5, 16), None),
                      ((2, 12, 4, 14), np.tri(10, 10, dtype=bool))]:
        bm = edsc.roi_from_cache(tmp_path / "c", win, ["Si", "Fe", "Mg"],
                                 blend_mode="mean", progress=False)
        sums_ref, _ = _net_count_sums(bm, sub_bbox=None, sub_mask=mask)
        sums_fast = reader.net_sums(win, mask)
        assert set(sums_ref) == set(sums_fast)
        for el in sums_ref:
            assert sums_fast[el] == pytest.approx(sums_ref[el])
        # fractions are exactly net_sums renormalised
        s = sum(sums_fast.values())
        if s > 0:
            frac = reader.net_fractions(win, mask)
            for el in frac:
                assert frac[el] == pytest.approx(sums_fast[el] / s)


def test_cache_window_reader_net_sums_subsampled(tmp_path):
    # Subsampled net_sums (step>1) reads step^2 fewer pixels and scales back up to
    # an unbiased estimate of the full-resolution sum — close, not exact. This is
    # the I/O saving that makes whole-mosaic calibration tractable.
    H, W = 48, 48
    _build_cache(tmp_path / "c", H, W, ["Si", "Fe"], weight_val=1.0)
    reader = edsc.CacheWindowReader(tmp_path / "c", ["Si", "Fe"], blend_mode="mean")
    box = (0, H, 0, W)
    full = reader.net_sums(box, step=1)
    sub = reader.net_sums(box, step=3)
    assert set(full) == set(sub)
    for el in full:
        assert sub[el] == pytest.approx(full[el], rel=0.1)
    # coverage fraction is scale-free
    assert reader.coverage_frac(box, step=3) == pytest.approx(1.0)
    # net_fractions still defaults to the exact (step=1) read
    assert reader.net_sums(box) == reader._net_totals(box)


def test_cache_window_reader_rejects_ds_gt_1(tmp_path):
    _build_cache(tmp_path / "c", 6, 6, ["Si"])
    # Mark the manifest as a ds=2 cache → reader must refuse (caller falls back).
    m = edsc.load_manifest(tmp_path / "c")
    m["ds"] = 2
    edsc.save_manifest(tmp_path / "c", m)
    with pytest.raises(ValueError):
        edsc.CacheWindowReader(tmp_path / "c", ["Si"])


def test_is_valid_for_fingerprint(tmp_path):
    csv = tmp_path / "stitch_manifest.csv"
    csv.write_text("row,col\n0,0\n", encoding="utf-8")
    fp = edsc.stitch_fingerprint(csv)
    cache = tmp_path / "eds_cache"
    _build_cache(cache, 4, 4, ["Si"], fingerprint=fp)
    assert edsc.is_valid_for(cache, csv, required_elements=["Si"]) is True
    assert edsc.is_valid_for(cache, csv, required_elements=["Si", "Mn"]) is False
    # Mutating the manifest CSV changes the fingerprint → cache invalid.
    csv.write_text("row,col\n0,1\n", encoding="utf-8")
    assert edsc.is_valid_for(cache, csv, required_elements=["Si"]) is False
