# tests/test_unmix.py
"""Mixed-domain NNLS unmixing + modal-proportion redistribution.

Covers `viewer_thermo_helpers._unmix_domain` (areal decomposition of a
sub-pixel mixture domain from mean net-count vectors) and the `unmix=`
extension of `_observed_modes_from_phase_map` (pixel-count redistribution
onto endmember domains for the P-T modes).
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from viewer_thermo_helpers import (
    _unmix_domain,
    _observed_modes_from_phase_map,
)


def test_unmix_recovers_known_fractions():
    means = {
        0: {"Si": 10.0, "K": 0.0},           # endmember A
        1: {"Si": 0.0, "K": 5.0},            # endmember B
        2: {"Si": 6.0, "K": 2.0},            # 0.6 A + 0.4 B
    }
    fr, resid = _unmix_domain(means, source_cid=2, endmember_cids=[0, 1])
    assert fr[0] == pytest.approx(0.6, abs=1e-9)
    assert fr[1] == pytest.approx(0.4, abs=1e-9)
    assert resid == pytest.approx(0.0, abs=1e-9)


def test_unmix_element_alignment_missing_keys():
    # Endmembers with disjoint element sets align over the union (missing=0).
    means = {
        0: {"Si": 8.0},
        1: {"Fe": 4.0, "Mg": 2.0},
        2: {"Si": 4.0, "Fe": 2.0, "Mg": 1.0},   # 0.5 each
    }
    fr, resid = _unmix_domain(means, 2, [0, 1])
    assert fr[0] == pytest.approx(0.5, abs=1e-9)
    assert fr[1] == pytest.approx(0.5, abs=1e-9)
    assert resid == pytest.approx(0.0, abs=1e-9)


def test_unmix_unexplained_signal_shows_in_residual():
    # Source has signal in an element no endmember carries -> residual > 0
    # and the fraction sum stays diagnostic.
    means = {
        0: {"Si": 10.0, "Ca": 0.0},
        1: {"Si": 5.0, "Ca": 8.0},           # source: half A + rogue Ca
    }
    fr, resid = _unmix_domain(means, 1, [0])
    assert fr[0] == pytest.approx(0.5, abs=1e-9)
    assert resid > 0.5                        # the Ca signal is unexplained


def test_unmix_needs_endmembers():
    with pytest.raises(ValueError):
        _unmix_domain({0: {"Si": 1.0}}, 0, [0])   # source filtered out


def test_modes_redistribute_unmixed_domain():
    pm = np.array([[0, 0, 1, 1],
                   [1, 1, 2, 2]])              # 0:2px  1:4px  2:2px
    keys = {0: "a", 1: "b", 2: "c"}
    # Without unmix: a=0.25 b=0.5 c=0.25.
    modes, exc = _observed_modes_from_phase_map(pm, keys)
    assert modes == pytest.approx({"a": 0.25, "b": 0.5, "c": 0.25})
    assert exc == pytest.approx(0.0)
    # Unmix domain 1 into 0 and 2 at 0.5/0.25 (renormalised to 2/3 : 1/3);
    # 'b' disappears, its 4 px split 8/3 to 'a' and 4/3 to 'c'.
    modes, exc = _observed_modes_from_phase_map(
        pm, keys, unmix={1: {0: 0.5, 2: 0.25}})
    assert "b" not in modes
    assert modes["a"] == pytest.approx((2 + 8 / 3) / 8)
    assert modes["c"] == pytest.approx((2 + 4 / 3) / 8)
    assert exc == pytest.approx(0.0)


def test_modes_unmix_string_keys_and_missing_source():
    pm = np.array([[0, 1]])
    keys = {0: "a", 1: "b"}
    # String-keyed fractions (JSON round-trip shape) work; a source cid not
    # present in the map is ignored.
    modes, _ = _observed_modes_from_phase_map(
        pm, keys, unmix={"1": {"0": 1.0}, 7: {0: 1.0}})
    assert modes == pytest.approx({"a": 1.0})


def test_modes_unmix_empty_fractions_is_noop():
    pm = np.array([[0, 1]])
    keys = {0: "a", 1: "b"}
    modes, _ = _observed_modes_from_phase_map(pm, keys, unmix={1: {}})
    assert modes == pytest.approx({"a": 0.5, "b": 0.5})


def test_modes_unmix_to_unkeyed_endmember_goes_excluded():
    pm = np.array([[0, 0, 1, 1]])
    keys = {0: "a"}                            # domain 1's target has no key...
    modes, exc = _observed_modes_from_phase_map(
        pm, keys, unmix={0: {1: 1.0}})         # ...and 0 unmixes fully into 1
    assert modes == {}
    assert exc == pytest.approx(1.0)
