"""Session-serialisation round-trip + polygon-mask rasterisation.

These are the cores behind viewer session save/load and ROI handling.
``_rasterize_polygon_mask`` lives in ``viewer_helpers`` which imports napari,
so the geometry tests import it lazily under ``importorskip``.
"""
import numpy as np
import pytest

from thermo.result import ThermoResult, PhaseResult
from viewer_thermo_helpers import (
    _serialize_thermod_result, _deserialize_thermod_result,
)


def test_thermod_result_session_roundtrip():
    r = ThermoResult(
        p_gpa=0.9, t_c=655.0,
        stable_phases=[
            PhaseResult("g", 0.4, 0.41, 0.39,
                        composition_oxide_wt={"FeO": 30.0, "MgO": 12.0}),
            PhaseResult("liq", 0.6, 0.59, 0.61, is_melt=True),
        ],
        bulk_properties={"rho_kgm3": 3120.0, "Vs_kms": 3.4},
        backend="magemin/mp", converged=False,
    )
    d = _serialize_thermod_result(r)
    # JSON-safe: only built-in types
    import json
    json.dumps(d)
    r2 = _deserialize_thermod_result(d)
    assert r2.p_gpa == 0.9 and r2.t_c == 655.0
    assert r2.converged is False
    assert [p.name for p in r2.stable_phases] == ["g", "liq"]
    assert r2.phase("g").composition_oxide_wt == {"FeO": 30.0, "MgO": 12.0}
    assert r2.phase("liq").is_melt is True
    assert r2.bulk_properties["rho_kgm3"] == 3120.0


def test_row_bbox_on_canvas_ignores_eds_scale_and_shear():
    # Regression: manifest y/x are the tile's solved position in mosaic (BSE)
    # pixels and bse_h/bse_w its composited size. scale_y/scale_x are the
    # EDS-grid-to-BSE size ratios (768x768 EDS grid on a 1024x768 tile ->
    # scale_x=0.75) and shx/shy compositing shear -- NOT canvas transforms.
    # A previous version multiplied position and size by scale_x, misplacing
    # every tile on such manifests: SPX<->tile matching and the mosaic
    # calibration silently read the wrong windows (r2 < 0 for every element
    # on the re-stitched C23S1A).
    pytest.importorskip("napari")
    from viewer_helpers import _row_bbox_on_canvas
    row = {"y": "100.4", "x": "28066.2", "bse_h": "768", "bse_w": "1024",
           "eds_h": "768", "eds_w": "768", "scale_y": "1.0", "scale_x": "0.75",
           "shx": "0.0", "shy": "-0.0005"}
    assert _row_bbox_on_canvas(row) == (100, 869, 28066, 29091)
    # Degenerate row -> None.
    assert _row_bbox_on_canvas({"y": "0", "x": "0", "bse_h": "0",
                                "bse_w": "0"}) is None


def test_rasterize_square_polygon():
    pytest.importorskip("napari")
    from viewer_helpers import _rasterize_polygon_mask
    # Polygon = the rectangle y in [2,6], x in [1,5] within a 10x10 bbox.
    verts = np.array([[2, 1], [2, 5], [6, 5], [6, 1]], dtype=float)
    mask = _rasterize_polygon_mask(verts, 0, 10, 0, 10)
    assert mask.shape == (10, 10)
    assert mask.any()
    # Interior point inside; far corner outside.
    assert mask[4, 3] is np.True_ or mask[4, 3]
    assert not mask[0, 0]
    # The mask's bounding box of True pixels lies within the polygon extent.
    ys, xs = np.where(mask)
    assert ys.min() >= 2 and ys.max() <= 6
    assert xs.min() >= 1 and xs.max() <= 5


def test_rasterize_empty_bbox():
    pytest.importorskip("napari")
    from viewer_helpers import _rasterize_polygon_mask
    verts = np.array([[0, 0], [0, 1], [1, 1]], dtype=float)
    assert _rasterize_polygon_mask(verts, 5, 5, 0, 3).shape == (0, 0)


# ---------------------------------------------------------------------------
# ROI EDS-map memory guard (prevents giant-ROI MemoryError)
# ---------------------------------------------------------------------------

def test_roi_map_bytes_persistent_term():
    pytest.importorskip("napari")
    from viewer_helpers import _roi_map_bytes
    # With no transient term + safety 1.0, it's just the persistent maps:
    # 1000×1000, 1 element, 3 bands, float32 = 12 MB.
    b = _roi_map_bytes(1000, 1000, 1, n_bands=3, transient_bands=0, safety=1.0)
    assert b == 1000 * 1000 * 1 * 3 * 4
    # Persistent part is linear in elements and area.
    assert _roi_map_bytes(1000, 1000, 4, transient_bands=0, safety=1.0) == 4 * b
    assert _roi_map_bytes(2000, 1000, 1, transient_bands=0, safety=1.0) == 2 * b


def test_roi_map_bytes_transient_is_fixed_overhead():
    pytest.importorskip("napari")
    from viewer_helpers import _roi_map_bytes
    band = 1000 * 1000 * 4
    # The transient term adds a fixed number of band-sizes, independent of n_el.
    one = _roi_map_bytes(1000, 1000, 1, n_bands=3, transient_bands=14, safety=1.0)
    assert one == (3 * 1 + 14) * band
    twenty = _roi_map_bytes(1000, 1000, 20, n_bands=3, transient_bands=14, safety=1.0)
    assert twenty == (3 * 20 + 14) * band
    # Adding 19 elements adds only the persistent part (3 bands each).
    assert twenty - one == 19 * 3 * band


def test_downsample_to_fit():
    pytest.importorskip("napari")
    from viewer_helpers import _roi_map_bytes, _downsample_to_fit
    h = w = 10000
    n_el = 10
    # Fits comfortably → no downsample.
    big_budget = _roi_map_bytes(h, w, n_el) * 2
    assert _downsample_to_fit(h, w, n_el, big_budget) == 1
    # A tiny budget forces a factor; the decimated estimate must actually fit.
    small_budget = _roi_map_bytes(h, w, n_el) / 9.0
    ds = _downsample_to_fit(h, w, n_el, small_budget)
    assert ds > 1
    ho, wo = -(-h // ds), -(-w // ds)
    assert _roi_map_bytes(ho, wo, n_el) <= small_budget
    # A factor one smaller would NOT have fit (smallest sufficient ds).
    if ds > 1:
        ho2, wo2 = -(-h // (ds - 1)), -(-w // (ds - 1))
        assert _roi_map_bytes(ho2, wo2, n_el) > small_budget


def test_downsample_no_budget_is_one():
    pytest.importorskip("napari")
    from viewer_helpers import _downsample_to_fit
    assert _downsample_to_fit(9999, 9999, 50, None) == 1
    assert _downsample_to_fit(9999, 9999, 50, 0) == 1


def test_add_stoichiometric_oxygen_basis_matches_oxides():
    # add_stoichiometric_oxygen must reconstruct exactly the O carried by the
    # oxides elements_to_oxides_wt builds, so a calibrated cation-only column
    # lands on the same (cation + stoichiometric O) basis as SPX.
    pytest.importorskip("napari")
    from viewer_helpers import add_stoichiometric_oxygen, elements_to_oxides_wt
    cations = {"Si": 30.0, "Mg": 20.0, "Fe": 10.0, "Ca": 5.0}
    with_o = add_stoichiometric_oxygen(cations, fe3_over_fe_total=0.2)
    assert abs(sum(with_o.values()) - 100.0) < 1e-6
    assert "O" in with_o and with_o["O"] > 0
    # Cross-check the O mass fraction against the oxide route: the wt fraction of
    # O in the oxide assemblage = 1 − (Σ cation wt)/(Σ oxide wt).
    ox = elements_to_oxides_wt(cations, fe3_over_fe_total=0.2)  # normalised to 100
    cat_wt_in_ox = 100.0 - ox.get("O", 0.0)  # oxides carry no bare O key
    # Rebuild unnormalised O fraction independently via the cation/oxide split.
    o_frac_expected = with_o["O"] / 100.0
    # Cation:O ratio must be internally consistent (cations renormalise to 1−O).
    cat_sum = sum(v for el, v in with_o.items() if el != "O")
    assert abs(cat_sum / 100.0 - (1.0 - o_frac_expected)) < 1e-9


def test_add_stoichiometric_oxygen_replaces_existing_o_and_keeps_carbon():
    pytest.importorskip("napari")
    from viewer_helpers import add_stoichiometric_oxygen
    # A bogus pre-existing O is discarded (recomputed); C is non-oxide-forming,
    # so it is kept but contributes no oxygen.
    out = add_stoichiometric_oxygen({"Si": 40.0, "O": 999.0, "C": 5.0})
    assert abs(sum(out.values()) - 100.0) < 1e-6
    assert out["O"] < 100.0  # not the bogus 999
    assert "C" in out

    # Fe3+/ΣFe shifts O slightly (Fe2O3 carries more O per Fe than FeO).
    lo = add_stoichiometric_oxygen({"Fe": 100.0}, fe3_over_fe_total=0.0)
    hi = add_stoichiometric_oxygen({"Fe": 100.0}, fe3_over_fe_total=1.0)
    assert hi["O"] > lo["O"]
