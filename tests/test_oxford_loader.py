"""oxford_loader: .piprj parsing + full synthetic round-trip into the cache."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import eds_cache as edsc
import oxford_loader as ox

tifffile = pytest.importorskip("tifffile")
pd = pytest.importorskip("pandas")


def _write_piprj(path, size, name=3, startx=0, starty=0):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata.json",
                   json.dumps({"version": "1.0.0.0", "projectType": "PhaseIdentifier"}))
        z.writestr("ProjectUnits.json",
                   json.dumps({"Size": size, "Name": name,
                               "StartX": startx, "StartY": starty}))


def _make_dataset(d: Path, piprj_size=None):
    H, W = 8, 10
    for el, val in (("Si", 25.0), ("Fe", 10.0), ("O", 40.0)):
        pd.DataFrame(np.full((H, W), val, np.float32)).to_csv(
            d / f"Montaged Map Data-{el} Wt%.csv", header=False, index=False)
    tifffile.imwrite(d / "BSE image.tif",
                     (np.random.rand(2 * H, 2 * W) * 65535).astype(np.uint16))
    if piprj_size is not None:
        _write_piprj(d / "run.piprj", size=piprj_size, startx=12, starty=34)


def test_parse_piprj_reads_pixel_size(tmp_path):
    _write_piprj(tmp_path / "x.piprj", size=4, startx=5, starty=6)
    m = ox.parse_piprj(tmp_path / "x.piprj", progress=False)
    assert m["pixel_um"] == 4.0
    assert m["pixel_unit_code"] == 3
    assert m["stage_origin_um"] == (5.0, 6.0)
    assert m["project_type"] == "PhaseIdentifier"
    assert m["source"] == "x.piprj"


def test_parse_piprj_bad_file_is_nonfatal(tmp_path):
    bad = tmp_path / "notazip.piprj"
    bad.write_bytes(b"this is not a zip archive")
    m = ox.parse_piprj(bad, progress=False)
    assert m["pixel_um"] is None              # graceful, no exception
    assert m["source"] == "notazip.piprj"


def test_discover_piprj(tmp_path):
    assert ox.discover_piprj(tmp_path) == []
    (tmp_path / "a.piprj").write_bytes(b"PK")
    (tmp_path / "b.piprj").write_bytes(b"PK")
    found = ox.discover_piprj(tmp_path)
    assert [p.name for p in found] == ["a.piprj", "b.piprj"]


def test_load_oxford_autodetects_pixel_um(tmp_path):
    _make_dataset(tmp_path, piprj_size=7)
    out = ox.load_oxford_dataset(tmp_path, force=True, progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["pixel_um"] == 7.0
    assert list(man["stage_origin_um"]) == [12.0, 34.0]
    assert man["pixel_unit_code"] == 3
    assert man["data_type"] == "oxford_wt_pct"
    assert sorted(man["elements"]) == ["Fe", "O", "Si"]


def test_load_oxford_explicit_pixel_um_overrides(tmp_path):
    _make_dataset(tmp_path, piprj_size=7)
    out = ox.load_oxford_dataset(tmp_path, force=True, pixel_um=2.5, progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["pixel_um"] == 2.5


def test_load_oxford_no_piprj(tmp_path):
    _make_dataset(tmp_path, piprj_size=7)
    out = ox.load_oxford_dataset(tmp_path, force=True, use_piprj=False, progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["pixel_um"] is None


def test_load_oxford_coating_default_and_override(tmp_path):
    _make_dataset(tmp_path, piprj_size=7)
    # Default → carbon coat recorded in the manifest.
    out = ox.load_oxford_dataset(tmp_path, force=True, progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["coating_elements"] == ["C"]
    # Explicit Pt coat round-trips (and an empty set = uncoated).
    out = ox.load_oxford_dataset(tmp_path, force=True, coating_elements=("Pt",),
                                 progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["coating_elements"] == ["Pt"]
    out = ox.load_oxford_dataset(tmp_path, force=True, coating_elements=(),
                                 progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["coating_elements"] == []


def test_load_oxford_cache_values_match_source(tmp_path):
    _make_dataset(tmp_path, piprj_size=4)
    out = ox.load_oxford_dataset(tmp_path, force=True, progress=False)
    # Oxford cache: weight=1, sidebands=0 → roi 'main' equals the Wt% map.
    maps = edsc.roi_from_cache(edsc.cache_dir_for(out), (0, 8, 0, 10),
                               ["Si"], blend_mode="mean", progress=False)
    assert np.allclose(maps["Si"]["main"], 25.0)
    assert np.allclose(maps["Si"]["sb_lo"], 0.0)


def test_load_oxford_relocated_cache_dir(tmp_path):
    """--eds-cache-dir: cache lands elsewhere, sidecar link lets the viewer find it."""
    src = tmp_path / "data"; src.mkdir()
    relocated = tmp_path / "ssd_cache"
    _make_dataset(src, piprj_size=4)

    out = ox.load_oxford_dataset(src, force=True, cache_dir=relocated, progress=False)

    # Cache is at the relocated path, not the default one.
    assert (relocated / edsc.CACHE_MANIFEST).exists()
    assert not (edsc.cache_dir_for(out) / edsc.CACHE_MANIFEST).exists()
    # Sidecar link resolves it (this is what the viewer calls).
    assert edsc.resolve_cache_dir(out) == relocated.resolve()
    man = edsc.load_manifest(edsc.resolve_cache_dir(out))
    assert man["data_type"] == "oxford_wt_pct"

    # Idempotent re-run without an explicit cache_dir follows the sidecar
    # and skips the rebuild (manifest mtime unchanged).
    mtime = (relocated / edsc.CACHE_MANIFEST).stat().st_mtime
    ox.load_oxford_dataset(src, progress=False)
    assert (relocated / edsc.CACHE_MANIFEST).stat().st_mtime == mtime

    # Rebuilding at the default location clears the stale sidecar.
    ox.load_oxford_dataset(src, force=True, progress=False)
    assert edsc.resolve_cache_dir(out) == edsc.cache_dir_for(out)


# ---------------------------------------------------------------------------
# Two-resolution montage: BSE kept native, element cache coarser (ds>1)
# ---------------------------------------------------------------------------

def _make_dataset_shaped(d: Path, el_h, el_w, bse_h, bse_w, val=25.0):
    """Element CSVs at (el_h, el_w) + a BSE TIF at (bse_h, bse_w)."""
    for el, v in (("Si", val), ("Fe", 10.0)):
        pd.DataFrame(np.full((el_h, el_w), v, np.float32)).to_csv(
            d / f"Montaged Map Data-{el} Wt%.csv", header=False, index=False)
    rng = np.random.default_rng(0)
    tifffile.imwrite(d / "BSE image.tif",
                     (rng.random((bse_h, bse_w)) * 65535).astype(np.uint16))


def test_load_oxford_two_resolution_native_bse(tmp_path):
    # BSE exactly 3× the element grid → BSE kept native, cache at element grid,
    # ds=3. The detail in the BSE is preserved instead of downsampled away.
    el_h, el_w = 8, 10
    _make_dataset_shaped(tmp_path, el_h, el_w, 3 * el_h, 3 * el_w)
    out = ox.load_oxford_dataset(tmp_path, force=True, use_piprj=False,
                                 progress=False)

    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["ds"] == 3
    assert man["H"] == el_h and man["W"] == el_w
    assert man["H_bse"] == 3 * el_h and man["W_bse"] == 3 * el_w

    # cache memmaps stay at the coarse element grid
    si = np.load(edsc._element_path(edsc.cache_dir_for(out), "Si", "main"))
    assert si.shape == (el_h, el_w)

    # BSE PNG/TIFF written at native resolution
    bse16 = tifffile.imread(out / "stitched_BSE_16bit.tiff")
    assert bse16.shape == (3 * el_h, 3 * el_w)

    # stitch_manifest records the BSE/EDS split + integer scale
    row = pd.read_csv(out / "stitch_manifest.csv").iloc[0]
    assert int(row["bse_h"]) == 3 * el_h and int(row["eds_h"]) == el_h
    assert float(row["scale_y"]) == 3.0 and float(row["scale_x"]) == 3.0

    # ROI read takes BSE-space coords and resizes the coarse cache back up
    maps = edsc.roi_from_cache(edsc.cache_dir_for(out),
                               (0, 3 * el_h, 0, 3 * el_w), ["Si"],
                               blend_mode="mean", progress=False)
    assert maps["Si"]["main"].shape == (3 * el_h, 3 * el_w)
    assert np.allclose(maps["Si"]["main"], 25.0)   # constant map → constant upsample


def test_load_oxford_non_integer_ratio_single_grid(tmp_path):
    # BSE not a clean integer multiple of the element grid → fall back to the
    # single-grid behaviour (BSE downsampled to the element grid, ds=1).
    el_h, el_w = 8, 10
    _make_dataset_shaped(tmp_path, el_h, el_w, 15, 18)   # 15/8, 18/10 → not integer
    out = ox.load_oxford_dataset(tmp_path, force=True, use_piprj=False,
                                 progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["ds"] == 1
    assert man["H"] == el_h and man["W"] == el_w
    assert man["H_bse"] == el_h and man["W_bse"] == el_w
    bse16 = tifffile.imread(out / "stitched_BSE_16bit.tiff")
    assert bse16.shape == (el_h, el_w)   # downsampled to the element grid


def test_load_oxford_anisotropic_ratio_single_grid(tmp_path):
    # Unequal per-axis ratios (2× vs 3×) can't be expressed by a single cache
    # ds → fall back to single grid rather than mis-scaling one axis.
    el_h, el_w = 8, 10
    _make_dataset_shaped(tmp_path, el_h, el_w, 2 * el_h, 3 * el_w)
    out = ox.load_oxford_dataset(tmp_path, force=True, use_piprj=False,
                                 progress=False)
    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["ds"] == 1
    assert man["H_bse"] == el_h and man["W_bse"] == el_w


# ---------------------------------------------------------------------------
# h5oina (Oxford HDF5 EBSD + EDS export)
# ---------------------------------------------------------------------------

def test_parse_phase_name():
    assert ox._parse_phase_name("Diopside   CaMgSi2O6") == ("Diopside", "CaMgSi2O6")
    assert ox._parse_phase_name("Chlorite") == ("Chlorite", None)
    # single-space "Name Formula" with a digit-bearing trailing token
    assert ox._parse_phase_name("Forsterite Mg2SiO4") == ("Forsterite", "Mg2SiO4")
    # a multi-word name with no formula stays intact
    assert ox._parse_phase_name("Ferro actinolite") == ("Ferro actinolite", None)
    assert ox._parse_phase_name("") == ("Unknown", None)


def _make_h5oina(path: Path, H=6, W=8):
    """Write a tiny synthetic .h5oina with EDS + EBSD + FSE groups."""
    h5py = pytest.importorskip("h5py")
    sdt = h5py.string_dtype(encoding="utf-8")
    N = H * W
    with h5py.File(path, "w") as f:
        top = f.create_group("1")
        # --- EDS ---
        eh = top.create_group("EDS/Header")
        eh.create_dataset("X Cells", data=np.array([W], np.int32))
        eh.create_dataset("Y Cells", data=np.array([H], np.int32))
        eh.create_dataset("X Step", data=np.array([6.0], np.float32))
        eh.create_dataset("Beam Voltage", data=np.array([25.0], np.float32))
        wi = top.create_group("EDS/Data/Window Integral")
        wi.create_dataset("Si Kα1", data=np.full(N, 30.0, np.float32))
        wi.create_dataset("Mg Kα1,2", data=np.full(N, 10.0, np.float32))
        wi.create_dataset("O Kα1", data=np.full(N, 5.0, np.float32))
        wi.create_dataset("Zz Kα1", data=np.full(N, 1.0, np.float32))  # not an element
        # --- EBSD ---
        bh = top.create_group("EBSD/Header")
        bh.create_dataset("X Cells", data=np.array([W], np.int32))
        bh.create_dataset("Y Cells", data=np.array([H], np.int32))
        phase = np.zeros(N, np.uint8)
        phase[: N // 2] = 1          # ordinal 1 → Diopside
        phase[N // 2: 3 * N // 4] = 2  # ordinal 2 → Forsterite
        top.create_dataset("EBSD/Data/Phase", data=phase)
        for ordinal, (name, pid, col) in {
            1: ("Diopside  CaMgSi2O6", 47, [200, 100, 50]),
            2: ("Forsterite", 99, [10, 200, 90]),
        }.items():
            pg = top.create_group(f"EBSD/Header/Phases/{ordinal}")
            pg.create_dataset("Phase Name", data=np.array([name], dtype=object), dtype=sdt)
            pg.create_dataset("Phase Id", data=np.array([pid], np.int32))
            pg.create_dataset("Color", data=np.array([col], np.uint8))
        # --- Electron Image / FSE (2x finer grid, 2 channels) ---
        ih = top.create_group("Electron Image/Header")
        ih.create_dataset("X Cells", data=np.array([2 * W], np.int32))
        ih.create_dataset("Y Cells", data=np.array([2 * H], np.int32))
        fse = top.create_group("Electron Image/Data/FSE")
        fse.create_dataset("ch1", data=np.full(4 * N, 100, np.int16))
        fse.create_dataset("ch2", data=np.full(4 * N, 120, np.int16))


def test_discover_h5oina_file_and_dir(tmp_path):
    pytest.importorskip("h5py")
    f = tmp_path / "map.h5oina"
    _make_h5oina(f)
    assert ox.discover_h5oina(f) == f
    assert ox.discover_h5oina(tmp_path) == f
    # subfolder layout (h5oina/ inside the project dir)
    sub = tmp_path / "proj" / "h5oina"
    sub.mkdir(parents=True)
    g = sub / "map.h5oina"
    _make_h5oina(g)
    assert ox.discover_h5oina(tmp_path / "proj") == g
    assert ox.discover_h5oina(tmp_path / "empty" if False else tmp_path / "proj") == g


def test_load_h5oina_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    H, W = 6, 8
    _make_h5oina(tmp_path / "map.h5oina", H=H, W=W)
    out = ox.load_h5oina_dataset(tmp_path, force=True, progress=False)

    man = edsc.load_manifest(edsc.cache_dir_for(out))
    assert man["data_type"] == "oxford_counts"
    assert "H5OINA" in man["source"]
    assert man["provenance"] == "h5oina"
    assert man["pixel_um"] == 6.0
    assert man["beam_voltage_kV"] == 25.0
    # "Zz" is not a real element → skipped; "Mg Kα1,2" parses to Mg.
    assert sorted(man["elements"]) == ["Mg", "O", "Si"]
    assert man["H"] == H and man["W"] == W

    # uncalibrated counts → ratio-only mineral ID (not quantified)
    assert "oxford_counts" not in __import__("mineral_id").QUANTIFIED_TYPES

    # EBSD artifacts
    pm = np.load(out / "ebsd_phase_map.npy")
    assert pm.shape == (H, W) and pm.dtype == np.uint8
    assert set(np.unique(pm).tolist()) == {0, 1, 2}
    legend = json.loads((out / "ebsd_phases.json").read_text(encoding="utf-8"))
    # keyed by the Phase *ordinal* (1, 2), NOT the database Phase Id (47, 99)
    assert legend["1"]["name"] == "Diopside"
    assert legend["1"]["formula"] == "CaMgSi2O6"
    assert legend["2"]["name"] == "Forsterite"
    assert "47" not in legend and "99" not in legend
    assert legend["1"]["phase_id"] == 47

    # cache holds the raw counts (weight=1, sidebands=0)
    maps = edsc.roi_from_cache(edsc.cache_dir_for(out), (0, H, 0, W),
                               ["Si"], blend_mode="mean", progress=False)
    assert np.allclose(maps["Si"]["main"], 30.0)


def test_load_h5oina_idempotent(tmp_path):
    pytest.importorskip("h5py")
    _make_h5oina(tmp_path / "map.h5oina")
    out1 = ox.load_h5oina_dataset(tmp_path, force=True, progress=False)
    # second call without force should hit the valid cache and not raise
    out2 = ox.load_h5oina_dataset(tmp_path, force=False, progress=False)
    assert out1 == out2
    assert edsc.is_valid_for(edsc.cache_dir_for(out2), out2 / "stitch_manifest.csv",
                             required_elements=["Si", "Mg", "O"])


# ---------------------------------------------------------------------------
# Decimal-comma AZtec exports + data-tree file names (2026-10-05)
# ---------------------------------------------------------------------------

def _deccomma_text(arr):
    """Write *arr* the way AZtec does under a decimal-comma locale: comma as
    field delimiter AND decimal mark, up to 5 decimals, trailing zeros trimmed,
    zero as "0", every row ending with a comma."""
    lines = []
    for row in arr:
        toks = []
        for v in row:
            s = f"{float(v):.5f}".rstrip("0").rstrip(".")
            toks.append(s.replace(".", ","))
        lines.append(",".join(toks) + ",")
    return "\n".join(lines) + "\n"


def test_read_element_csv_decimal_comma(tmp_path):
    rng = np.random.default_rng(0)
    arr = np.round(rng.uniform(0, 60, size=(40, 55)), 5)
    arr[rng.random(arr.shape) < 0.5] = 0.0                 # void / absent element
    # the awkward cases: short decimal parts, values below 1, zeros around them
    arr[0, :12] = [17.5341, 0, 0, 0.5, 17.5, 0, 12.125, 0.25, 0, 0.05341, 3.0123, 45.0891]
    arr[1, :6] = [0, 0.5, 0.5, 0, 17.25, 0.125]
    arr[2, :] = 0.0                                        # an all-zero row
    p = tmp_path / "x__EDS Data__Si Wt%.csv"
    p.write_text(_deccomma_text(arr), encoding="utf-8")
    got = ox._read_element_csv(p)
    assert got.shape == arr.shape
    np.testing.assert_allclose(got, arr.astype(np.float32), atol=1e-4)


def test_read_element_csv_plain_files_untouched(tmp_path):
    # dot-decimal floats and integer count maps must take the normal path
    arr = np.array([[0.0, 17.5341, 0.5], [3.25, 0.0, 100.0]], dtype=np.float32)
    p = tmp_path / "a.csv"
    p.write_text("\n".join(",".join(f"{v:g}" for v in r) for r in arr) + "\n")
    np.testing.assert_allclose(ox._read_element_csv(p), arr, atol=1e-5)
    cnt = np.array([[0, 12, 1503], [7, 0, 20415]], dtype=np.float32)
    q = tmp_path / "b.csv"
    q.write_text("\n".join(",".join(str(int(v)) for v in r) for r in cnt) + "\n")
    np.testing.assert_allclose(ox._read_element_csv(q), cnt)
    assert not ox._is_decimal_comma_csv(q.read_text())


def test_discover_inputs_data_tree_names(tmp_path):
    for el in ("Si", "Mg", "O"):
        (tmp_path / f"proj__proj__Area 1__Montaged Data 1__Montaged Map Data__EDS Data__{el} Wt%.csv"
         ).write_text("1.0,2.0\n")
    bse, csvs = ox.discover_inputs(tmp_path)
    assert bse is None and sorted(csvs) == ["Mg", "O", "Si"]


def test_read_element_csv_decimal_comma_exact_integer(tmp_path):
    # "0,1" is 0.1 by the grammar, but a zero followed by an exact integer 1
    # writes the same tokens; the common row width decides (seen once in a
    # 815k-pixel Ti map: ...,0.5545485,0,1,0.5979917,...).
    arr = np.array([[0.5545485, 0.0, 1.0, 0.5979917, 0.0, 3.25],
                    [0.1, 0.0, 0.0, 12.5, 0.0, 0.0],
                    [7.12531, 0.53171, 0.0, 0.0, 2.0, 0.25344]], dtype=np.float32)
    p = tmp_path / "x__EDS Data__Ti Wt%.csv"
    p.write_text(_deccomma_text(arr), encoding="utf-8")
    got = ox._read_element_csv(p)
    assert got.shape == arr.shape
    np.testing.assert_allclose(got, arr, atol=1e-4)
