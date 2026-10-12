"""SPX-anchored EDS calibration (eds_quantify.py).

Pure-function tests: no Qt/napari. Synthetic tiles pair known EDS net-counts
with SPX wt% so the recovered k-factors and calibrated wt% can be checked
against hand-computed values.
"""
import numpy as np
import pytest

import eds_quantify as eq


# ---------------------------------------------------------------------------
# derive_k_factors
# ---------------------------------------------------------------------------

def test_recover_known_k_exact():
    """W = k_true * I exactly over several tiles -> k ~= k_true, r2 ~= 1."""
    k_true = {"Si": 0.020, "Mg": 0.035, "Fe": 0.011}
    counts = [
        {"Si": 1000.0, "Mg": 400.0, "Fe": 900.0},
        {"Si": 2000.0, "Mg": 100.0, "Fe": 1500.0},
        {"Si": 500.0, "Mg": 800.0, "Fe": 200.0},
    ]
    spx = [{el: k_true[el] * c[el] for el in c} for c in counts]

    cal = eq.derive_k_factors(spx, counts)
    assert set(cal.elements) == {"Si", "Mg", "Fe"}
    for el, k in k_true.items():
        assert cal.k_factors[el].k == pytest.approx(k, rel=1e-9)
        assert cal.k_factors[el].n_tiles == 3
        assert cal.k_factors[el].r2 == pytest.approx(1.0, abs=1e-9)


def test_noisy_multi_tile_r2_between_zero_and_one():
    k_true = 0.02
    rng = np.random.default_rng(0)
    counts, spx = [], []
    for _ in range(12):
        I = float(rng.uniform(500, 3000))
        W = k_true * I + float(rng.normal(0, 2.0))
        counts.append({"Si": I})
        spx.append({"Si": max(W, 0.1)})
    cal = eq.derive_k_factors(spx, counts)
    kf = cal.k_factors["Si"]
    assert kf.k == pytest.approx(k_true, rel=0.15)
    assert kf.n_tiles == 12
    assert kf.r2 is not None and 0.0 < kf.r2 < 1.0


def test_single_tile_degenerate():
    cal = eq.derive_k_factors([{"Si": 4.0}], [{"Si": 200.0}])
    kf = cal.k_factors["Si"]
    assert kf.k == pytest.approx(4.0 / 200.0)
    assert kf.n_tiles == 1
    assert kf.r2 is None


def test_missing_element_handling():
    # Si shared; Al only in SPX (no EDS anchor); Fe only in EDS (no SPX ref);
    # O/C present in counts but never calibrated.
    counts = [{"Si": 1000.0, "Fe": 500.0, "O": 9999.0, "C": 50.0},
              {"Si": 2000.0, "Fe": 700.0, "O": 9999.0, "C": 50.0}]
    spx = [{"Si": 20.0, "Al": 5.0}, {"Si": 40.0, "Al": 6.0}]
    cal = eq.derive_k_factors(spx, counts)
    assert cal.elements == ["Si"]
    assert "Al" in cal.spx_only
    assert "Fe" in cal.eds_only
    assert "O" not in cal.k_factors and "C" not in cal.k_factors
    assert "O" not in cal.eds_only and "C" not in cal.eds_only


def test_anion_coating_split_constants():
    assert eq.ANION_ELEMENTS == frozenset({"O"})
    assert eq.DEFAULT_COATING_ELEMENTS == frozenset({"C"})
    assert eq.DROP_ELEMENTS == eq.ANION_ELEMENTS | eq.DEFAULT_COATING_ELEMENTS


def test_drop_set_configurable():
    # C is genuine signal here; Pt is the coating. With the default drop set,
    # C is dropped; with a Pt coating drop set, C is calibrated and Pt dropped.
    counts = [{"Si": 1000.0, "C": 300.0, "Pt": 800.0},
              {"Si": 2000.0, "C": 600.0, "Pt": 800.0}]
    spx = [{"Si": 20.0, "C": 6.0}, {"Si": 40.0, "C": 12.0}]

    # Pt-coated: drop O + Pt → C IS calibrated, Pt never is, O never is.
    cal = eq.derive_k_factors(spx, counts, drop=eq.ANION_ELEMENTS | {"Pt"})
    assert "C" in cal.k_factors
    assert "Pt" not in cal.k_factors and "Pt" not in cal.eds_only
    assert "O" not in cal.k_factors

    # Anion-only drop (uncoated): C is still calibrated.
    cal2 = eq.derive_k_factors(spx, counts, drop=eq.ANION_ELEMENTS)
    assert "C" in cal2.k_factors

    # Default drop set still excludes C (back-compat).
    cal3 = eq.derive_k_factors(spx, counts)
    assert "C" not in cal3.k_factors


def test_empty_inputs():
    cal = eq.derive_k_factors([], [])
    assert cal.k_factors == {}
    assert cal.elements == []


def test_calibration_dict_roundtrip():
    counts = [{"Si": 1000.0, "Mg": 400.0}, {"Si": 2000.0, "Mg": 100.0}]
    spx = [{"Si": 20.0, "Mg": 14.0}, {"Si": 40.0, "Mg": 3.5}]
    cal = eq.derive_k_factors(spx, counts)
    cal2 = eq.Calibration.from_dict(cal.to_dict())
    assert cal2.k() == pytest.approx(cal.k())
    assert cal2.elements == cal.elements
    assert cal2.k_factors["Si"].r2 == pytest.approx(cal.k_factors["Si"].r2)


# ---------------------------------------------------------------------------
# cross_validate_k_factors
# ---------------------------------------------------------------------------

def test_cv_perfect_data_zero_error():
    """Exact W = k*I tiles -> every held-out prediction matches SPX exactly."""
    k_true = {"Si": 0.020, "Mg": 0.035, "Fe": 0.011}
    counts = [
        {"Si": 1000.0, "Mg": 400.0, "Fe": 900.0},
        {"Si": 2000.0, "Mg": 100.0, "Fe": 1500.0},
        {"Si": 500.0, "Mg": 800.0, "Fe": 200.0},
        {"Si": 1500.0, "Mg": 600.0, "Fe": 700.0},
    ]
    spx = [{el: k_true[el] * c[el] for el in c} for c in counts]

    cv = eq.cross_validate_k_factors(spx, counts, method="loo")
    assert cv is not None
    assert cv.method == "loo"
    assert cv.n_tiles == 4
    assert set(cv.elements) == {"Si", "Mg", "Fe"}
    assert cv.rmse == pytest.approx(0.0, abs=1e-9)
    for el, s in cv.per_element.items():
        assert s.n_tiles == 4
        assert s.bias == pytest.approx(0.0, abs=1e-9)
        assert s.rmse == pytest.approx(0.0, abs=1e-9)
    # records carry comparable wt% (both renormalised over the shared set)
    for rec in cv.records:
        assert sum(rec["predicted"].values()) == pytest.approx(100.0)
        assert sum(rec["measured"].values()) == pytest.approx(100.0)


def test_cv_noisy_data_positive_bounded_error():
    """Noisy tiles -> nonzero but bounded RMSE; rmse consistent with residuals."""
    k_true = {"Si": 0.02, "Mg": 0.035}
    rng = np.random.default_rng(1)
    counts, spx = [], []
    for _ in range(10):
        c = {"Si": float(rng.uniform(500, 3000)), "Mg": float(rng.uniform(200, 1500))}
        counts.append(c)
        spx.append({el: max(k_true[el] * v * float(rng.normal(1.0, 0.05)), 0.1)
                    for el, v in c.items()})
    cv = eq.cross_validate_k_factors(spx, counts, method="loo")
    assert cv is not None
    assert cv.n_tiles == 10
    assert 0.0 < cv.rmse < 10.0
    # overall rmse == rms over all per-record residuals
    res = [rec["predicted"][el] - rec["measured"][el]
           for rec in cv.records for el in rec["predicted"]]
    assert cv.rmse == pytest.approx(float(np.sqrt(np.mean(np.square(res)))))


def test_cv_detects_inconsistent_tile():
    """One tile violating the shared k-factors shows up as large residuals."""
    k_true = {"Si": 0.02, "Mg": 0.035}
    counts = [{"Si": 1000.0 + 100 * i, "Mg": 500.0 + 50 * i} for i in range(5)]
    spx = [{el: k_true[el] * c[el] for el in c} for c in counts]
    # corrupt the last tile: Mg quantified 3x higher than its counts imply
    spx[-1]["Mg"] *= 3.0
    cv = eq.cross_validate_k_factors(spx, counts, method="loo")
    assert cv is not None
    bad = max(cv.records, key=lambda r: abs(r["predicted"]["Mg"] - r["measured"]["Mg"]))
    assert bad["tile"] == 4
    assert cv.per_element["Mg"].rmse > 1.0


def test_cv_too_few_tiles_returns_none():
    assert eq.cross_validate_k_factors([{"Si": 4.0}], [{"Si": 200.0}]) is None
    assert eq.cross_validate_k_factors([], []) is None


def test_cv_two_tiles_weak_but_works():
    """LOO with 2 tiles fits each fold on a single tile (k = W/I exactly)."""
    k_true = {"Si": 0.02, "Mg": 0.035}
    counts = [{"Si": 1000.0, "Mg": 400.0}, {"Si": 2000.0, "Mg": 100.0}]
    spx = [{el: k_true[el] * c[el] for el in c} for c in counts]
    cv = eq.cross_validate_k_factors(spx, counts, method="loo")
    assert cv is not None
    assert cv.n_tiles == 2
    assert cv.rmse == pytest.approx(0.0, abs=1e-9)


def test_cv_dict_is_json_friendly():
    import json
    counts = [{"Si": 1000.0, "Mg": 400.0},
              {"Si": 2000.0, "Mg": 100.0},
              {"Si": 700.0, "Mg": 900.0}]
    spx = [{"Si": 0.02 * c["Si"], "Mg": 0.035 * c["Mg"]} for c in counts]
    cv = eq.cross_validate_k_factors(spx, counts, method="loo")
    d = json.loads(json.dumps(cv.to_dict()))
    assert d["n_tiles"] == 3
    assert set(d["per_element"]) == {"Si", "Mg"}
    assert d["method"] == "loo" and "rmse_std" in d and "n_folds" in d


# ---------------------------------------------------------------------------
# filter_calibration_tiles (spurious-tile pre-filter)
# ---------------------------------------------------------------------------

def test_filter_drops_spurious_keeps_good():
    spx = [{"Si": 30.0, "Mg": 10.0},   # 0 good (kept)
           {"Si": 28.0, "Mg": 12.0},   # 1 low coverage -> dropped via valid_fracs
           {},                          # 2 no SPX -> dropped
           {"Si": 26.0, "Mg": 9.0},    # 3 empty counts -> dropped via low_counts
           {"Si": 27.0, "Mg": 11.0}]   # 4 good (kept)
    counts = [{"Si": 1500.0, "Mg": 400.0},
              {"Si": 1400.0, "Mg": 380.0},
              {"Si": 1200.0, "Mg": 300.0},
              {"Si": 0.0, "Mg": 0.0},
              {"Si": 1300.0, "Mg": 350.0}]
    kept_spx, kept_counts, dropped = eq.filter_calibration_tiles(
        spx, counts, valid_fracs=[1.0, 0.2, 1.0, 1.0, 1.0], min_valid_frac=0.5)
    assert len(kept_spx) == len(kept_counts) == 2
    drop_idx = {d["index"] for d in dropped}
    assert drop_idx == {1, 2, 3}
    by_idx = {d["index"]: d["reasons"] for d in dropped}
    assert "low_valid_frac" in by_idx[1]
    assert "no_spx" in by_idx[2]
    assert "low_counts" in by_idx[3]


def test_filter_conservative_defaults_keep_borderline():
    # With no valid_fracs/weights and default thresholds, only truly-empty tiles go.
    spx = [{"Si": 30.0}, {"Si": 5.0}, {"Si": 28.0}]
    counts = [{"Si": 1500.0}, {"Si": 3.0}, {"Si": 0.0}]  # 2nd is low but real, 3rd empty
    kept_spx, kept_counts, dropped = eq.filter_calibration_tiles(spx, counts)
    assert len(kept_spx) == 2            # the borderline low-count tile is kept
    assert {d["index"] for d in dropped} == {2}
    assert dropped[0]["reasons"] == ["low_counts"]


# ---------------------------------------------------------------------------
# configurable cross-validation: closed-form LOO + repeated hold-out
# ---------------------------------------------------------------------------

def _refit_loo_rmse(spx, counts, drop=eq.DROP_ELEMENTS):
    """Reference leave-one-out by *re-fitting* derive_k_factors per held-out tile."""
    res = {}
    for t in range(len(spx)):
        cal = eq.derive_k_factors(spx[:t] + spx[t + 1:],
                                  counts[:t] + counts[t + 1:], drop=drop)
        if not cal.k_factors:
            continue
        pred = eq.apply_calibration_fractions(counts[t], cal.k())
        if not pred:
            continue
        meas = {el: float(v) for el, v in spx[t].items()
                if el in pred and float(v) > 0.0}
        shared = sorted(set(pred) & set(meas))
        if len(shared) < 2:
            continue
        pt = sum(pred[el] for el in shared)
        mt = sum(meas[el] for el in shared)
        pn = {el: 100.0 * pred[el] / pt for el in shared}
        mn = {el: 100.0 * meas[el] / mt for el in shared}
        for el in shared:
            res.setdefault(el, []).append(pn[el] - mn[el])
    return {el: float(np.sqrt(np.mean(np.square(rs)))) for el, rs in res.items()}


def test_cv_loo_closed_form_matches_refit():
    """The closed-form LOO must equal a per-tile re-fit LOO, numerically."""
    k_true = {"Si": 0.02, "Mg": 0.035, "Fe": 0.011}
    rng = np.random.default_rng(7)
    counts, spx = [], []
    for _ in range(8):
        c = {el: float(rng.uniform(400, 3000)) for el in k_true}
        counts.append(c)
        spx.append({el: max(k_true[el] * v * float(rng.normal(1.0, 0.06)), 0.1)
                    for el, v in c.items()})
    cv = eq.cross_validate_k_factors(spx, counts, method="loo")
    ref = _refit_loo_rmse(spx, counts)
    assert cv is not None and cv.method == "loo"
    assert set(cv.per_element) == set(ref)
    for el, rmse in ref.items():
        assert cv.per_element[el].rmse == pytest.approx(rmse, rel=1e-9, abs=1e-9)


def test_cv_holdout_deterministic_and_shaped():
    k_true = {"Si": 0.02, "Mg": 0.035}
    rng = np.random.default_rng(3)
    counts, spx = [], []
    for _ in range(20):
        c = {el: float(rng.uniform(400, 3000)) for el in k_true}
        counts.append(c)
        spx.append({el: max(k_true[el] * v * float(rng.normal(1.0, 0.05)), 0.1)
                    for el, v in c.items()})
    a = eq.cross_validate_k_factors(spx, counts, method="holdout",
                                    holdout_frac=0.2, n_repeats=30, seed=42)
    b = eq.cross_validate_k_factors(spx, counts, method="holdout",
                                    holdout_frac=0.2, n_repeats=30, seed=42)
    assert a is not None and a.method == "holdout"
    assert a.n_folds == 30
    assert a.rmse == pytest.approx(b.rmse)            # seeded determinism
    assert a.rmse_std > 0.0                           # an error bar exists
    assert all(s.rmse_std >= 0.0 for s in a.per_element.values())
    # a different seed gives a (slightly) different estimate
    c = eq.cross_validate_k_factors(spx, counts, method="holdout",
                                    holdout_frac=0.2, n_repeats=30, seed=99)
    assert c.rmse != a.rmse


def test_cv_holdout_small_n_falls_back_to_loo():
    counts = [{"Si": 1000.0, "Mg": 400.0},
              {"Si": 2000.0, "Mg": 100.0},
              {"Si": 700.0, "Mg": 900.0}]
    spx = [{"Si": 0.02 * c["Si"], "Mg": 0.035 * c["Mg"]} for c in counts]
    cv = eq.cross_validate_k_factors(spx, counts, method="holdout")
    assert cv is not None and cv.method == "loo"     # 3 tiles -> auto LOO


# ---------------------------------------------------------------------------
# apply_calibration_fractions
# ---------------------------------------------------------------------------

def test_apply_fractions_exact_wt():
    k = {"Si": 0.02, "Mg": 0.035, "Fe": 0.011}
    # raw net-counts; calibrated weights then renormalise to 100
    net = {"Si": 1000.0, "Mg": 400.0, "Fe": 900.0}
    raw = {el: k[el] * net[el] for el in net}
    tot = sum(raw.values())
    expected = {el: 100.0 * raw[el] / tot for el in net}

    out = eq.apply_calibration_fractions(net, k)
    assert sum(out.values()) == pytest.approx(100.0)
    for el in net:
        assert out[el] == pytest.approx(expected[el])


def test_apply_fractions_scale_invariant():
    """Fractions (sum~1) and raw counts give the same calibrated wt%."""
    k = {"Si": 0.02, "Mg": 0.035}
    net = {"Si": 1000.0, "Mg": 400.0}
    frac = {el: v / sum(net.values()) for el, v in net.items()}
    assert eq.apply_calibration_fractions(net, k) == pytest.approx(
        eq.apply_calibration_fractions(frac, k))


def test_apply_fractions_drops_uncalibrated_and_empty():
    out = eq.apply_calibration_fractions({"Si": 1000.0, "Fe": 500.0}, {"Si": 0.02})
    assert set(out) == {"Si"}
    assert out["Si"] == pytest.approx(100.0)
    assert eq.apply_calibration_fractions({}, {"Si": 0.02}) == {}
    assert eq.apply_calibration_fractions({"Fe": 1.0}, {"Si": 0.02}) == {}


# ---------------------------------------------------------------------------
# apply_calibration_maps
# ---------------------------------------------------------------------------

def _band(main):
    main = np.asarray(main, dtype=np.float32)
    z = np.zeros_like(main)
    return {"main": main, "sb_lo": z, "sb_hi": z}


def test_apply_maps_per_pixel_sums_to_100_where_signal():
    band_maps = {
        "Si": _band([[100.0, 0.0], [50.0, 200.0]]),
        "Mg": _band([[100.0, 0.0], [0.0, 100.0]]),
    }
    k = {"Si": 1.0, "Mg": 2.0}
    out = eq.apply_calibration_maps(band_maps, k)
    assert set(out) == {"Si", "Mg"}
    total = out["Si"] + out["Mg"]
    # pixel (0,0): Si=100*1=100, Mg=100*2=200 -> Si 33.33, Mg 66.67
    assert out["Si"][0, 0] == pytest.approx(100.0 / 3.0, abs=1e-3)
    assert out["Mg"][0, 0] == pytest.approx(200.0 / 3.0, abs=1e-3)
    # pixel (0,1): zero signal -> stays 0
    assert total[0, 1] == pytest.approx(0.0)
    # pixel (1,0): only Si -> 100
    assert out["Si"][1, 0] == pytest.approx(100.0)
    # every non-empty pixel sums to 100
    nz = total > 0
    assert np.allclose(total[nz], 100.0, atol=1e-3)


def test_apply_maps_net_subtraction_and_mask():
    band_maps = {
        "Si": {"main": np.array([[10.0, 10.0]], np.float32),
               "sb_lo": np.array([[4.0, 4.0]], np.float32),
               "sb_hi": np.array([[4.0, 4.0]], np.float32)},
        "Fe": {"main": np.array([[10.0, 10.0]], np.float32),
               "sb_lo": np.zeros((1, 2), np.float32),
               "sb_hi": np.zeros((1, 2), np.float32)},
    }
    # Si net = 10 - 0.5*(8) = 6 ; Fe net = 10
    k = {"Si": 1.0, "Fe": 1.0}
    mask = np.array([[1, 0]], dtype=bool)
    out = eq.apply_calibration_maps(band_maps, k, sub_mask=mask)
    # masked-out pixel (0,1) -> 0
    assert out["Si"][0, 1] == pytest.approx(0.0)
    assert out["Fe"][0, 1] == pytest.approx(0.0)
    # pixel (0,0): Si 6, Fe 10 -> 37.5 / 62.5
    assert out["Si"][0, 0] == pytest.approx(37.5, abs=1e-3)
    assert out["Fe"][0, 0] == pytest.approx(62.5, abs=1e-3)


def test_apply_maps_no_overlap_returns_empty():
    band_maps = {"Si": _band([[1.0]])}
    assert eq.apply_calibration_maps(band_maps, {"Fe": 1.0}) == {}
    assert eq.apply_calibration_maps({}, {"Si": 1.0}) == {}
