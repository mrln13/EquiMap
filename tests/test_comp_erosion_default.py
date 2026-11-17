"""Interior-only naming is on by default in the phase-classification dialog,
and an explicit 0 (full masks, e.g. a session made before the default) is
restored as off."""
import numpy as np
import pytest


@pytest.fixture
def dialog(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    vd = pytest.importorskip("viewer_dialogs")
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])   # noqa: F841
    labels = np.zeros((20, 20), dtype=np.int16)
    labels[:, 10:] = 1
    dlg = vd._PhaseClassDialog(labels, ["Si", "Al"], {0: "Si", 1: "Al"}, 2,
                               (0.0, 0.0), viewer=None)
    yield vd, dlg
    dlg.deleteLater()


def test_interior_only_is_on_by_default(dialog):
    vd, dlg = dialog
    assert vd.DEFAULT_COMP_EROSION_PX == 3
    assert dlg.erosion_value() == 3


def test_explicit_zero_restores_off_and_values_round_trip(dialog):
    _vd, dlg = dialog
    dlg.set_erosion(0)
    assert dlg.erosion_value() == 0
    dlg.set_erosion(5)
    assert dlg.erosion_value() == 5
