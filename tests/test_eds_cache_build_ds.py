"""build_cache at reduced resolution when the mosaic size is not a multiple of ds.

A tile touching the right/bottom edge used to reach one cache cell past the
floor-divided cache grid and raise "operands could not be broadcast" (seen on
a 5901 px wide mosaic with --eds-cache-ds 2); stitch.py caught it and skipped
the cache silently.
"""
import json

import numpy as np
import pytest

pytest.importorskip("hyperspy")      # eds_cache imports eds_map, which imports HyperSpy
import eds_cache                     # noqa: E402
import eds_map                       # noqa: E402

HEADER = ("row,col,y,x,file,bse_h,bse_w,eds_h,eds_w,scale_y,scale_x,shx,shy,"
          "E0,dE,nE,energy_units,beam_energy,elements,xray_lines\n")
TILES = [  # (row, col, y, x, h, w): mosaic 15 x 19 px, both odd
    (0, 0, 0, 0, 8, 10),
    (0, 1, 0, 9, 8, 10),
    (1, 0, 7, 0, 8, 10),
    (1, 1, 7, 9, 8, 10),
]


def _fake_tile(row, base_dir, elements, bands_def, *args, **kwargs):
    y, x = int(row["y"]), int(row["x"])
    h, w = int(row["bse_h"]), int(row["bse_w"])
    ones = np.ones((h, w), np.float32)
    return {"yB0": y, "yB1": y + h, "xB0": x, "xB1": x + w, "y_end": y + h, "x_end": x + w,
            "ww": ones, "ww_eff": ones, "has_mask": False,
            "patches": {e: {b: ones.copy() for b in ("main", "sb_lo", "sb_hi")} for e in elements}}


@pytest.mark.parametrize("ds", [1, 2, 4])
def test_build_cache_odd_mosaic(tmp_path, monkeypatch, ds):
    monkeypatch.setattr(eds_map, "_process_one_tile", _fake_tile)
    lines = [HEADER] + [f"{r},{c},{y},{x},t_{r}_{c}.bcf,{h},{w},{h},{w},1,1,0,0,0,0.01,2048,keV,,,\n"
                        for r, c, y, x, h, w in TILES]
    man = tmp_path / "stitch_manifest.csv"
    man.write_text("".join(lines))
    eds_cache.build_cache(man, ["Si", "Fe"], ds=ds, progress=False, max_workers=1)
    m = json.loads((tmp_path / "eds_cache" / "manifest.json").read_text())
    assert (m["H_bse"], m["W_bse"]) == (15, 19)
    assert (m["H"], m["W"]) == (15 // ds, 19 // ds)
    w = np.load(tmp_path / "eds_cache" / "weight.npy")
    assert w.shape == (15 // ds, 19 // ds)
    assert w[:, -1].min() > 0 and w[-1, :].min() > 0     # the edge cells are filled
    assert eds_cache.is_valid_for(tmp_path / "eds_cache", man)


@pytest.mark.parametrize("ds", [1, 2])
def test_build_cache_zeroes_pixels_no_tile_wrote(tmp_path, monkeypatch, ds):
    """Pixels no tile covers must read as 0 even when the file system does not
    return zeros for never-written space (a cache built in a OneDrive folder
    had NaN, negative and denormal values in its never-written last row)."""
    monkeypatch.setattr(eds_map, "_process_one_tile", _fake_tile)
    real_zero = eds_cache._zero_unwritten

    def poison_then_zero(arrays, written, **kw):
        gap = np.asarray(arrays[0]) == 0         # no tile wrote these pixels
        for a in arrays:
            a[gap] = np.nan
        return real_zero(arrays, written, **kw)

    monkeypatch.setattr(eds_cache, "_zero_unwritten", poison_then_zero)
    tiles = TILES[:3]                      # bottom-right tile missing: a gap
    lines = [HEADER] + [f"{r},{c},{y},{x},t_{r}_{c}.bcf,{h},{w},{h},{w},1,1,0,0,0,0.01,2048,keV,,,\n"
                        for r, c, y, x, h, w in tiles]
    man = tmp_path / "stitch_manifest.csv"
    man.write_text("".join(lines))
    eds_cache.build_cache(man, ["Si"], ds=ds, progress=False, max_workers=1)
    cd = tmp_path / "eds_cache"
    w = np.load(cd / "weight.npy")
    main = np.load(cd / "Si__main.npy")
    assert np.isfinite(w).all() and (w >= 0).all()
    assert w[-1, -1] == 0 and main[-1, -1] == 0            # inside the missing tile
    assert w[0, 0] > 0
    # The whole BSE extent, past the cells a ds>1 cache holds: padded, not an error.
    maps = eds_cache.roi_from_cache(cd, (0, 15, 0, 19), ["Si"], progress=False)
    assert maps["Si"]["main"].shape == (15, 19)
    assert np.isfinite(maps["Si"]["main"]).all()
