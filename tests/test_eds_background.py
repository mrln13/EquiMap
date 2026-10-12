"""eds_background: line-aware background plan + signed net counts.

The legacy scheme took each element's background as the mean of two
sidebands 225 eV either side of its line. In the Na-Mg-Al-Si-P cluster the
lines are 210-270 eV apart, so a sideband sits on the neighbour's peak
(Al's upper sideband holds Si). These tests pin the plan the rule derives
for the C23S1A element list, that the net is immune to a neighbour's peak
in a sideband, that a linear continuum is recovered exactly under every
mode, the anchor fallback, the signed-sum contract of the helpers, and the
scheme switch.
"""
import numpy as np
import pytest

import eds_background as bg

C23 = ["C", "K", "O", "P", "Al", "Ca", "Fe", "Mg", "Na", "Si", "Ti"]


def test_plan_for_the_silicate_element_list():
    plan = bg.background_plan(C23)
    modes = {e: plan[e]["mode"] for e in C23}
    assert modes["Ti"] == "both" and modes["Fe"] == "both"
    assert modes["K"] == "lo"            # K Kb + Si pile-up in the upper sideband
    assert modes["Ca"] == "hi"           # Si pile-up (3.48 keV) in the lower
    assert modes["P"] == "hi"            # Si Ka + Kb in the lower
    assert modes["Na"] == "lo"           # Mg in the upper
    for e in ("Mg", "Al", "Si"):
        assert modes[e] == "interp"
        assert plan[e]["anchors"] == [("Na", "sb_lo"), ("P", "sb_hi")]
        w = plan[e]["weights"]
        assert abs(sum(w) - 1.0) < 1e-12 and 0.0 < w[1] < 1.0
    assert "Si" in plan["Al"]["dirty"]["sb_hi"] and "Mg" in plan["Al"]["dirty"]["sb_lo"]
    assert "Al" in plan["Si"]["dirty"]["sb_lo"]
    for e in ("O", "C"):
        assert modes[e] == "legacy" and plan[e]["approximate"]
    assert not any(plan[e]["approximate"] for e in C23 if e not in ("O", "C"))
    # interpolation weight grows with the line energy
    assert plan["Na"]["mode"] == "lo"
    assert plan["Mg"]["weights"][1] < plan["Al"]["weights"][1] < plan["Si"]["weights"][1]


def test_plan_without_neighbours_uses_both_sidebands():
    plan = bg.background_plan(["Si", "Fe"])
    assert plan["Si"]["mode"] == "both" and plan["Fe"]["mode"] == "both"
    assert bg.anchor_elements(plan, ["Si"]) == []


def test_anchor_elements_lists_what_a_subset_needs():
    plan = bg.background_plan(C23)
    assert bg.anchor_elements(plan, ["Al", "Si"]) == ["Na", "P"]
    assert bg.anchor_elements(plan, ["Fe", "Ti"]) == []


def _flat_bands(elements, bg_level=1.0, shape=(3, 4)):
    """Every window holds a flat continuum ``bg_level``; peaks added by callers."""
    return {e: {b: np.full(shape, bg_level, np.float32) for b in ("main", "sb_lo", "sb_hi")}
            for e in elements}


def test_neighbour_peak_in_sideband_does_not_touch_the_net():
    bands = _flat_bands(C23, bg_level=1.0)
    bands["Al"]["main"] += 5.0            # the Al peak
    bands["Al"]["sb_hi"] += 40.0          # ...and the Si peak leaking into Al's upper sideband
    legacy = bg.net_count_maps(bands, scheme="sideband")["Al"]
    aware = bg.net_count_maps(bands, scheme="line-aware")["Al"]
    assert np.all(legacy < 0)                        # the old scheme subtracts the neighbour
    assert np.allclose(aware, 5.0)                   # the new one sees the Al peak alone


def _continuum_bands(plan, a, c, peaks):
    """Windows filled with the continuum ``a + c*E`` at their centre, plus a
    peak of known height in each main window."""
    bands = {}
    for e in plan:
        w = plan[e]["windows"]
        bands[e] = {}
        for band in ("main", "sb_lo", "sb_hi"):
            centre = 0.5 * sum(w[band])
            val = a + c * centre + (peaks[e] if band == "main" else 0.0)
            bands[e][band] = np.full((2, 2), val, np.float32)
    return bands


def test_linear_continuum_recovered_exactly_by_two_sided_modes():
    # "both" and "interp" are linear estimates, so a sloped continuum is
    # recovered exactly; one-sided modes ("lo"/"hi") are exact on a flat
    # continuum and carry slope * 225 eV on a sloped one (a per-element
    # constant that the k-factor fit absorbs).
    plan = bg.background_plan(C23)
    peaks = {e: 1.0 + i for i, e in enumerate(C23)}
    nets = bg.net_count_maps(_continuum_bands(plan, 0.4, 0.3, peaks), plan=plan)
    for e in C23:
        mode = plan[e]["mode"]
        if mode in ("both", "interp"):
            assert np.allclose(nets[e], peaks[e], atol=1e-4), (e, mode, nets[e][0, 0])
        elif mode in ("lo", "hi"):
            assert np.allclose(abs(nets[e] - peaks[e]), 0.3 * 0.225, atol=1e-4), (e, mode)
    flat = bg.net_count_maps(_continuum_bands(plan, 0.4, 0.0, peaks), plan=plan)
    for e in C23:
        if not plan[e]["approximate"]:
            assert np.allclose(flat[e], peaks[e], atol=1e-4), (e, plan[e]["mode"])


def test_missing_anchor_falls_back_to_own_sidebands_with_a_note():
    plan = bg.background_plan(C23)
    bands = _flat_bands(["Al", "Si"], bg_level=2.0)   # no Na / P bands available
    bands["Al"]["main"] += 3.0
    notes = []
    nets = bg.net_count_maps(bands, plan=plan, elements=["Al"], notes=notes)
    assert np.allclose(nets["Al"], 3.0)
    assert notes and notes[0].startswith("Al:")


def test_signed_by_default_and_clip_on_request():
    bands = _flat_bands(["Fe"], bg_level=3.0)
    bands["Fe"]["main"] -= 1.0                        # noise: main below its sidebands
    assert np.all(bg.net_count_maps(bands)["Fe"] < 0)
    assert np.all(bg.net_count_maps(bands, clip=True)["Fe"] == 0)


def test_sub_bbox_slices_every_band():
    bands = _flat_bands(["Fe"], bg_level=0.0, shape=(4, 6))
    bands["Fe"]["main"][1:3, 2:5] = 7.0
    net = bg.net_count_maps(bands, sub_bbox=(1, 3, 2, 5))["Fe"]
    assert net.shape == (2, 3) and np.all(net == 7.0)


def test_default_scheme_switch_and_env_free_default():
    assert bg.default_scheme() in bg.SCHEMES
    old = bg.default_scheme()
    try:
        assert bg.set_default_scheme("sideband") == "sideband"
        bands = _flat_bands(["Al", "Si", "Na", "P"], bg_level=1.0)
        bands["Al"]["sb_hi"] += 10.0
        assert np.all(bg.net_count_maps(bands)["Al"] < 0)   # legacy behaviour restored
        with pytest.raises(ValueError):
            bg.set_default_scheme("bogus")
    finally:
        bg.set_default_scheme(old)


def test_describe_plan_mentions_anchors_and_dirty_lines():
    txt = bg.describe_plan(bg.background_plan(C23))
    assert "Al 1.486 keV: interpolated between Na sb_lo" in txt
    assert "sb_hi holds Si" in txt
    assert "APPROXIMATE" in txt


def test_helper_sums_are_signed_and_fractions_clip_at_the_aggregate():
    from viewer_thermo_helpers import _net_count_sums, _net_count_fractions
    bands = _flat_bands(["Fe", "Ti"], bg_level=2.0, shape=(2, 2))
    bands["Fe"]["main"] += 4.0             # net +4 per px
    bands["Ti"]["main"] -= 1.0             # net -1 per px: background-only element
    sums, n = _net_count_sums(bands)
    assert n == 4 and sums["Fe"] == pytest.approx(16.0) and sums["Ti"] == pytest.approx(-4.0)
    frac, _ = _net_count_fractions(bands)
    assert frac == {"Fe": 1.0, "Ti": 0.0}


def test_cache_reader_opens_anchor_bands_and_sums_signed(tmp_path):
    import eds_cache as edsc
    H, W = 4, 5
    cache = tmp_path / "c"; cache.mkdir()
    w = np.lib.format.open_memmap(edsc._weight_path(cache), mode="w+", dtype=np.float32, shape=(H, W))
    w[:] = 1.0; w.flush()
    # continuum 1.0 everywhere; Al main carries a peak of 3; Al's upper sideband
    # carries a big Si leak that the line-aware reader must ignore.
    vals = {e: {b: 1.0 for b in edsc.BANDS} for e in C23}
    vals["Al"]["main"] = 4.0; vals["Al"]["sb_hi"] = 30.0
    for e in C23:
        for b in edsc.BANDS:
            mm = np.lib.format.open_memmap(edsc._element_path(cache, e, b), mode="w+",
                                           dtype=np.float32, shape=(H, W))
            mm[:] = vals[e][b]; mm.flush()
    edsc.save_manifest(cache, {"stitch_fingerprint": "fp", "elements": C23, "ds": 1,
                               "H": H, "W": W, "data_type": "bcf_net_counts",
                               "width_eV": 150.0, "sideband_mult": 1.5})
    reader = edsc.CacheWindowReader(cache, ["Al"])
    assert set(reader._bands) == {"Al", "Na", "P"}          # anchors opened alongside
    assert reader.net_sums((0, H, 0, W)) == {"Al": pytest.approx(3.0 * H * W)}
    legacy = edsc.CacheWindowReader(cache, ["Al"], scheme="sideband")
    assert legacy.net_sums((0, H, 0, W))["Al"] < 0           # the old answer, for the record


def test_smoothing_adds_the_negative_tail_noise_floor_and_replays_legacy_unchanged():
    from viewer_segment_helpers import _smooth_element_maps, _noise_floor
    rng = np.random.default_rng(0)
    signed = rng.normal(0.0, 0.5, (80, 80)).astype(np.float32)   # signal-free noise, half negative
    signed[20:40, 20:40] += 6.0                                    # a grain
    sm = _smooth_element_maps({"Mg": signed}, sigma=2.0)["Mg"]
    assert np.all(sm > 0)                                           # no exact zeros reach the log features
    from scipy.ndimage import gaussian_filter
    ref = gaussian_filter(signed, sigma=2.0)
    floor = _noise_floor(ref)
    assert floor > 0 and np.allclose(sm, np.maximum(ref, 0) + floor, atol=1e-6)
    off = _smooth_element_maps({"Mg": signed}, sigma=2.0, noise_floor=False)["Mg"]
    assert np.any(off == 0) and np.allclose(off, np.maximum(ref, 0), atol=1e-6)
    clipped = np.maximum(signed, 0)                                 # legacy per-pixel clipped nets: no tail
    leg = _smooth_element_maps({"Mg": clipped}, sigma=2.0)["Mg"]
    assert _noise_floor(gaussian_filter(clipped, 2.0)) == 0.0
    assert np.allclose(leg, gaussian_filter(clipped, sigma=2.0), atol=1e-6)
