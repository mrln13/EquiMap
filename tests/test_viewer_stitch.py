"""Stitching from the viewer: argument building, detection, cleanup, presets."""
import os
import time

import pytest

pytest.importorskip("qtpy")
import viewer_stitch as vs   # noqa: E402


def _touch(p, mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def test_find_tiles_recursive_any_case(tmp_path):
    _touch(tmp_path / "job1" / "Mapping_(1,1).bcf")
    _touch(tmp_path / "job2" / "Mapping_(1,2).BCF")
    _touch(tmp_path / "job2" / "Quantified" / "Spectrum_(1,2).spx")
    assert len(vs.find_bcf_tiles(tmp_path)) == 2
    assert vs.needs_stitching(tmp_path)
    _touch(tmp_path / "stitched_BSE_8bit.png")
    assert not vs.needs_stitching(tmp_path)


def test_no_tiles_no_stitching(tmp_path):
    _touch(tmp_path / "notes.txt")
    assert not vs.needs_stitching(tmp_path)


def test_cache_estimate_matches_the_worked_example():
    # C23-S1-A: 2,014 tiles of 1024 x 768, 11 elements -> 168 GB measured
    est = vs.estimate_cache_bytes(2014, 1024 * 768, 11)
    assert 0.85 * 168e9 < est < 1.15 * 168e9
    assert vs.estimate_cache_bytes(2014, ds=2) == pytest.approx(vs.estimate_cache_bytes(2014) / 4)


def test_default_args_read_like_a_typed_command(tmp_path):
    a = vs.build_stitch_args(tmp_path)
    assert a == ["--dir", str(tmp_path), "--eds-preview", "all",
                 "--eds-cache-on-restitch", "rebuild"]


def test_advanced_args(tmp_path):
    a = vs.build_stitch_args(tmp_path, preset="A", cache_ds=2, cache_dir=tmp_path / "c",
                             auto_tune=True, coating="Pt", cache_workers=2)
    for part in (["--preset", "A"], ["--eds-cache-ds", "2"], ["--preflight", "--auto"],
                 ["--eds-cache-dir", str(tmp_path / "c")], ["--coating-elements", "Pt"],
                 ["--eds-cache-workers", "2"]):
        i = a.index(part[0])
        assert a[i:i + len(part)] == part


def test_diagnose_skips_eds_and_no_cache_flag(tmp_path):
    assert "--eds-preview" not in vs.build_stitch_args(tmp_path, diagnose=True)
    a = vs.build_stitch_args(tmp_path, cache=False)
    assert "--no-eds-cache" in a and "--eds-cache-on-restitch" not in a
    a = vs.build_stitch_args(tmp_path, coating="")
    assert a[-1] == "--coating-elements"            # uncoated: flag with no value
    assert "--eds-preview" not in vs.build_stitch_args(tmp_path, cache=False, previews=False)


def test_unfinished_mosaic_removed_old_one_kept(tmp_path):
    started = time.time()
    _touch(tmp_path / "stitched_BSE_8bit.png")                       # written by this run
    _touch(tmp_path / "stitched_BSE_16bit.tiff", mtime=started - 3600)  # from before
    assert vs.remove_unfinished_mosaic(tmp_path, started) == ["stitched_BSE_8bit.png"]
    assert (tmp_path / "stitched_BSE_16bit.tiff").exists()


def test_run_log_keeps_lines_and_final_progress_state(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    dlg = vs._StitchRunDialog(tmp_path, ["--dir", str(tmp_path)])
    dlg.feed("Found 36 .bcf files\n\x1b[33;20mWARNING | Hyperspy | `signal_type='EDS_SEM' not understood\x1b[0m\n")
    dlg.feed("\rEDS cache:   0%| 0/36\rEDS cache:  50%| 18/36")
    assert "50%" in dlg.progress.text()
    dlg.feed("\rEDS cache: 100%| 36/36\n[eds-cache] Skipped (error: boom)\n[composite] Blending...\n")
    lines = dlg.log.toPlainText().splitlines()
    assert lines == ["Found 36 .bcf files", "EDS cache: 100%| 36/36",
                     "[eds-cache] Skipped (error: boom)", "[composite] Blending..."]
    assert dlg.cache_failed


def test_run_log_windows_line_endings_split_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    dlg = vs._StitchRunDialog(tmp_path, ["--dir", str(tmp_path)])
    dlg.feed("usage: stitch.py [-h]\r\n  --dir DIR\r")   # \r\n lines, chunk ends mid-CRLF
    dlg.feed("\nFound 4 .bcf files\r\n")
    assert dlg.log.toPlainText().splitlines() == ["usage: stitch.py [-h]", "  --dir DIR",
                                                  "Found 4 .bcf files"]


def test_preset_descriptions_quote_the_real_values():
    from stitch_presets import PRESET_HELP, resolve_preset
    quoted = {   # preset -> {tunable: value its description states}
        "default": dict(anchor_sigma=25, REFINE_ITERS=2, REFINE_MAX_SHIFT=8, N_SLICES=13),
        "A": dict(VSTEP_ANCHOR_W=0.9, MAX_VERT_X_DRIFT=3, MAX_RIGHT_Y_DRIFT=6,
                  RIGHT_DY_WEIGHT=0, anchor_sigma=40, BAND_FRAC=0.05),
        "B": dict(N_SLICES=15, MAX_STRIP_DRIFT=10, UPSAMPLE=10, PYR_LEVELS=3,
                  REFINE_ITERS=2, REFINE_MAX_SHIFT=8),
        "C": dict(anchor_sigma=220, REFINE_ITERS=7, REFINE_MAX_SHIFT=30),
    }
    for name, facts in quoted.items():
        values = resolve_preset(name)
        for key, v in facts.items():
            assert float(values[key]) == float(v), (name, key)
        assert name == "default" or name in ("A", "B", "C")
        assert PRESET_HELP[name]
    # A has the strongest row-step anchor of all presets, default the strongest pull
    names = ("default", "A", "B", "C", "custom")
    assert max(names, key=lambda n: resolve_preset(n)["VSTEP_ANCHOR_W"]) == "A"
    assert min(names, key=lambda n: resolve_preset(n)["anchor_sigma"]) == "default"


def test_preset_table_marks_differences_from_default():
    rows = {label: (a, b, d) for label, a, b, d in vs.preset_table_rows("A")}
    a, b, d = rows["Pull onto stage positions (px; smaller = stronger)"]
    assert (a, b, d) == ("40", "25", True)
    assert not any(d for _l, _a, _b, d in vs.preset_table_rows("default"))


def test_presets_and_default_is_unchanged():
    stitch = pytest.importorskip("stitch")          # pulls in HyperSpy
    assert tuple(stitch.PRESET_CHOICES) == vs.PRESET_CHOICES
    # "default" = what stitch.py applied before --preset existed: preset C
    # followed by the C23S1A-class overrides (the paper's mosaic).
    stitch.apply_preset("A")
    stitch.apply_preset("default")
    t = stitch.current_tunables()
    assert t == {**stitch._BASE_TUNABLES, **stitch.PRESETS["C"], **stitch._FINE_GRAINED_OVERRIDES}
    expected = dict(REFINE_ITERS=2, REFINE_MAX_SHIFT=8, PYR_LEVELS=4, N_SLICES=13,
                    UPSAMPLE=20, RIGHT_DY_WEIGHT=0.10, VSTEP_ANCHOR_W=0.60,
                    anchor_sigma=25.0, BAND_FRAC=0.08, MAX_STRIP_DRIFT=15,
                    MAX_VERT_X_DRIFT=5, MAX_RIGHT_Y_DRIFT=10, OVERLAP_FRAC=0.10)
    assert {k: t[k] for k in expected} == expected
    stitch.apply_preset("A")
    assert stitch.PYR_LEVELS == 4 and stitch.anchor_sigma == 40.0 and stitch.PRESET == "A"
    stitch.apply_preset("custom")
    assert stitch.current_tunables() == stitch._BASE_TUNABLES
    stitch.apply_preset("default")
    with pytest.raises(ValueError):
        stitch.apply_preset("Z")
