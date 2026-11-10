"""SPX lookup for a BCF tile, in the Bruker layout and the published Yoda layout."""
import pytest

pytest.importorskip("napari")   # viewer_helpers imports napari
from viewer_helpers import _match_spx_for_bcf   # noqa: E402


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


def test_bruker_layout_prefers_quantified(tmp_path):
    job = tmp_path / "Mapping job_2025_09_29"
    bcf = _touch(job / "Mapping_(1,2).bcf")
    _touch(job / "Spectrum_(1,2).spx")                       # raw summed spectrum
    quant = _touch(job / "Quantified" / "Spectrum_(1,2).spx")
    _touch(job / "Quantified" / "Spectrum_(1,3).spx")
    assert _match_spx_for_bcf(bcf) == quant


def test_yoda_layout_uses_sibling_spx_of_the_same_run(tmp_path):
    bcf = _touch(tmp_path / "Run_1" / "BCF" / "Mapping_(1,2).bcf")
    spx = _touch(tmp_path / "Run_1" / "SPX" / "Spectrum_(1,2).spx")
    # the same tile name in another run must not be picked up
    _touch(tmp_path / "Run_2" / "SPX" / "Spectrum_(1,2).spx")
    assert _match_spx_for_bcf(bcf) == spx


def test_sibling_spx_only_for_a_folder_named_bcf(tmp_path):
    bcf = _touch(tmp_path / "Run_1" / "tiles" / "Mapping_(1,2).bcf")
    _touch(tmp_path / "Run_1" / "SPX" / "Spectrum_(1,2).spx")
    assert _match_spx_for_bcf(bcf) is None


def test_no_match(tmp_path):
    bcf = _touch(tmp_path / "Run_1" / "BCF" / "Mapping_(1,2).bcf")
    _touch(tmp_path / "Run_1" / "SPX" / "Spectrum_(4,5).spx")
    assert _match_spx_for_bcf(bcf) is None
