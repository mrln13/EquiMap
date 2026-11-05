# viewer_dialogs.py
"""
Qt dialog classes used by viewer.py:
- `_ThermoRunDialog`        — backend / P-T / composition / workflow options.
- `_CompositionScaleDialog` — quadtree settings + per-level window caps.
- `_PhaseSegDialog`         — segmentation settings (auto-k, normalisation, ...).
- `_PhaseClassDialog`       — post-clustering label-map preview + naming.
- `_CompositionCompareDialog` — EDS vs calibrated-EDS vs SPX ROI composition.

Plus the small `_hline()` factory shared between dialogs.

Moved out of viewer.py as part of the tier-B refactor.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib
from qtpy import QtWidgets, QtCore, QtGui

from viewer_thermo_helpers import (
    _max_useful_quadtree_depth,
    _quadtree_windows,
    _quadtree_level_dims,
)
from viewer_segment_helpers import (
    _label_map_to_rgba,
    _remove_phase_layers,
    _add_phase_cluster_layers,
)


def _hline():
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    return line


# -------------------------
# Thermodynamic modelling dialog
# -------------------------
class _ThermoRunDialog(QtWidgets.QDialog):
    """Options dialog for running a thermodynamic calculation from an ROI composition."""

    def __init__(self, comp_el: Optional[dict], source: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run thermodynamic model")
        self._comp_el = comp_el or {}
        layout = QtWidgets.QVBoxLayout(self)

        # ---- Composition source group ------------------------------------
        comp_group = QtWidgets.QGroupBox("Composition input")
        comp_vl = QtWidgets.QVBoxLayout(comp_group)

        self._rb_roi    = QtWidgets.QRadioButton("ROI composition (from EDS/SPX)")
        self._rb_custom = QtWidgets.QRadioButton("Custom oxide wt%")
        comp_vl.addWidget(self._rb_roi)
        comp_vl.addWidget(self._rb_custom)

        # ROI summary label
        if self._comp_el:
            summary = ", ".join(
                f"{el}={v:.1f}" for el, v in
                sorted(self._comp_el.items(), key=lambda x: -x[1])[:6]
            )
            self._roi_label = QtWidgets.QLabel(f"  {source}: {summary} …")
        else:
            self._roi_label = QtWidgets.QLabel("  (no ROI composition available)")
        self._roi_label.setWordWrap(True)
        comp_vl.addWidget(self._roi_label)

        # Custom oxide wt% input (SiO2:45.4, Al2O3:9.0, …). When an ROI
        # composition exists, pre-fill it (converted to oxide wt%, all Fe as
        # FeO — the Fe³⁺ knob below is applied at run time) so switching to
        # Custom starts from the auto composition instead of a blank line.
        self._custom_edit = QtWidgets.QLineEdit()
        self._custom_edit.setPlaceholderText(
            "e.g. SiO2:45.4, Al2O3:9.0, FeO:8.0, MgO:7.6, CaO:9.4, Na2O:2.7, TiO2:0.95, K2O:1.5"
        )
        if self._comp_el:
            try:
                from thermo.composition import Composition
                ox = Composition.from_element_wt(self._comp_el).to_oxide_wt()
                self._custom_edit.setText(", ".join(
                    f"{k}:{v:.2f}" for k, v in
                    sorted(ox.items(), key=lambda x: -x[1]) if v >= 0.01
                ))
            except Exception:
                pass  # leave the placeholder example
        self._custom_edit.setVisible(False)
        comp_vl.addWidget(self._custom_edit)

        layout.addWidget(comp_group)

        # Wire radio buttons
        if self._comp_el:
            self._rb_roi.setChecked(True)
        else:
            self._rb_roi.setEnabled(False)
            self._rb_custom.setChecked(True)
            self._roi_label.setVisible(False)
            self._custom_edit.setVisible(True)

        def _on_comp_source_changed():
            use_custom = self._rb_custom.isChecked()
            self._roi_label.setVisible(not use_custom)
            self._custom_edit.setVisible(use_custom)

        self._rb_roi.toggled.connect(_on_comp_source_changed)
        self._rb_custom.toggled.connect(_on_comp_source_changed)

        layout.addWidget(_hline())

        form = QtWidgets.QFormLayout()

        # Backend
        self.cmb_backend = QtWidgets.QComboBox()
        self.cmb_backend.addItems(["magemin", "perplex", "pycalphad"])
        form.addRow("Backend:", self.cmb_backend)

        # MAGEMin_C.jl direct - database selector (hidden for other backends)
        self.cmb_magemin_db = QtWidgets.QComboBox()
        for code, desc in [
            ("ig",   "ig - Igneous (Green et al. 2025)"),
            ("igad", "igad - Igneous (Weller et al. 2024)"),
            ("mp",   "mp - Metapelite (White et al. 2014)"),
            ("mb",   "mb - Metabasalt"),
            ("um",   "um - Ultramafic"),
            ("mpe",  "mpe - Metapelite extended"),
            ("mbe",  "mbe - Metabasalt extended"),
            ("mtl",  "mtl - Metatropical limestone"),
        ]:
            self.cmb_magemin_db.addItem(desc, userData=code)
        self._magemin_db_label = QtWidgets.QLabel("Database:")
        form.addRow(self._magemin_db_label, self.cmb_magemin_db)

        # TDB path for pycalphad
        tdb_widget = QtWidgets.QWidget()
        tdb_hl = QtWidgets.QHBoxLayout(tdb_widget)
        tdb_hl.setContentsMargins(0, 0, 0, 0)
        self.tdb_edit = QtWidgets.QLineEdit()
        tdb_browse = QtWidgets.QPushButton("…")
        tdb_browse.setFixedWidth(28)
        tdb_browse.clicked.connect(self._browse_tdb)
        tdb_hl.addWidget(self.tdb_edit)
        tdb_hl.addWidget(tdb_browse)
        self._tdb_row_label = QtWidgets.QLabel("TDB file:")
        form.addRow(self._tdb_row_label, tdb_widget)

        # Perplex dir for perplex backend
        px_widget = QtWidgets.QWidget()
        px_hl = QtWidgets.QHBoxLayout(px_widget)
        px_hl.setContentsMargins(0, 0, 0, 0)
        self.px_dir_edit = QtWidgets.QLineEdit()
        px_browse = QtWidgets.QPushButton("…")
        px_browse.setFixedWidth(28)
        px_browse.clicked.connect(self._browse_px_dir)
        px_hl.addWidget(self.px_dir_edit)
        px_hl.addWidget(px_browse)
        self._px_row_label = QtWidgets.QLabel("Perple_X dir:")
        form.addRow(self._px_row_label, px_widget)

        # Perple_X rock-type preset
        self.cmb_px_preset = QtWidgets.QComboBox()
        _px_presets = [
            ("Metapelite",       "H2O-sat",
             ["Mica(CHA1)", "Bio(TCC)", "Gt(W)", "feldspar", "St(W)",
              "Ctd(W)", "IlHm(A)", "Opx(JH)"]),
            ("Metabasalt",       "H2O-sat",
             ["Cpx(JH)", "Opx(JH)", "Gt(W)", "Pl(JH)", "Amph(DHP)",
              "IlHm(A)", "Bio(TCC)", "Mica(CHA1)"]),
            ("Igneous (mafic)",  "none",
             ["Cpx(JH)", "Opx(JH)", "Pl(JH)", "Gt(W)", "IlHm(A)", "melt(G)"]),
            ("Igneous (felsic)", "none",
             ["feldspar", "Pl(JH)", "Gt(W)", "Bio(TCC)", "Mica(CHA1)",
              "IlHm(A)", "melt(G)"]),
            ("Mantle",           "none",
             ["Ol(HP)", "Opx(JH)", "Cpx(JH)", "Gt(W)", "IlHm(A)"]),
            ("Custom",           None, None),
        ]
        for label, fluid, phases in _px_presets:
            self.cmb_px_preset.addItem(label, userData=(fluid, phases))
        self._px_preset_label = QtWidgets.QLabel("Rock type:")
        form.addRow(self._px_preset_label, self.cmb_px_preset)

        # Perple_X fluid
        self.cmb_px_fluid = QtWidgets.QComboBox()
        for fcode, fdesc in [("H2O-sat", "H\N{SUBSCRIPT TWO}O-saturated"),
                              ("none",    "No free fluid"),
                              ("H2O-CO2","H\N{SUBSCRIPT TWO}O-CO\N{SUBSCRIPT TWO} fluid")]:
            self.cmb_px_fluid.addItem(fdesc, userData=fcode)
        self._px_fluid_label = QtWidgets.QLabel("Fluid:")
        form.addRow(self._px_fluid_label, self.cmb_px_fluid)

        # Perple_X solution phases (editable, populated by preset)
        self.px_phases_edit = QtWidgets.QLineEdit()
        self.px_phases_edit.setPlaceholderText("comma-separated phase names")
        self._px_phases_label = QtWidgets.QLabel("Solution phases:")
        form.addRow(self._px_phases_label, self.px_phases_edit)

        # Wire preset → fluid + phases fields
        def _apply_px_preset(idx):
            fluid, phases = self.cmb_px_preset.itemData(idx)
            if phases is None:   # Custom - leave fields editable, don't overwrite
                return
            fi = self.cmb_px_fluid.findData(fluid)
            if fi >= 0:
                self.cmb_px_fluid.setCurrentIndex(fi)
            self.px_phases_edit.setText(", ".join(phases))

        self.cmb_px_preset.currentIndexChanged.connect(_apply_px_preset)
        _apply_px_preset(0)

        layout.addLayout(form)
        layout.addWidget(_hline())

        # Composition tweaks
        tweak_form = QtWidgets.QFormLayout()
        self.fe3_spin = QtWidgets.QDoubleSpinBox()
        self.fe3_spin.setRange(0.0, 1.0); self.fe3_spin.setSingleStep(0.05); self.fe3_spin.setValue(0.1)
        self.h2o_spin = QtWidgets.QDoubleSpinBox()
        self.h2o_spin.setRange(0.0, 100.0); self.h2o_spin.setDecimals(2); self.h2o_spin.setValue(0.0)
        self.co2_spin = QtWidgets.QDoubleSpinBox()
        self.co2_spin.setRange(0.0, 100.0); self.co2_spin.setDecimals(2); self.co2_spin.setValue(0.0)
        tweak_form.addRow("Fe\N{SUPERSCRIPT THREE}\N{SUPERSCRIPT PLUS SIGN} / Fe_total:", self.fe3_spin)
        tweak_form.addRow("H\N{SUBSCRIPT TWO}O add (wt%):", self.h2o_spin)
        tweak_form.addRow("CO\N{SUBSCRIPT TWO} add (wt%):", self.co2_spin)
        layout.addLayout(tweak_form)
        layout.addWidget(_hline())

        # Calculation mode
        mode_group = QtWidgets.QGroupBox("Calculation mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self.rb_single = QtWidgets.QRadioButton("Single P-T point")
        self.rb_grid   = QtWidgets.QRadioButton("P-T grid (pseudosection)")
        self.rb_single.setChecked(True)
        mode_layout.addWidget(self.rb_single)
        mode_layout.addWidget(self.rb_grid)

        pt_form = QtWidgets.QFormLayout()
        self.p_spin = QtWidgets.QDoubleSpinBox(); self.p_spin.setRange(0.001, 30.0); self.p_spin.setSingleStep(0.1); self.p_spin.setDecimals(3); self.p_spin.setValue(1.0); self.p_spin.setSuffix(" GPa")
        self.t_spin = QtWidgets.QDoubleSpinBox(); self.t_spin.setRange(0.0, 3000.0); self.t_spin.setSingleStep(50.0); self.t_spin.setValue(700.0); self.t_spin.setSuffix(" \N{DEGREE SIGN}C")
        pt_form.addRow("P:", self.p_spin)
        pt_form.addRow("T:", self.t_spin)
        mode_layout.addLayout(pt_form)

        grid_form = QtWidgets.QFormLayout()
        self.p_min_spin = QtWidgets.QDoubleSpinBox(); self.p_min_spin.setRange(0.001,30); self.p_min_spin.setDecimals(3); self.p_min_spin.setValue(0.5); self.p_min_spin.setSuffix(" GPa")
        self.p_max_spin = QtWidgets.QDoubleSpinBox(); self.p_max_spin.setRange(0.001,30); self.p_max_spin.setDecimals(3); self.p_max_spin.setValue(1.5); self.p_max_spin.setSuffix(" GPa")
        self.t_min_spin = QtWidgets.QDoubleSpinBox(); self.t_min_spin.setRange(0,3000); self.t_min_spin.setValue(400.0); self.t_min_spin.setSuffix(" \N{DEGREE SIGN}C")
        self.t_max_spin = QtWidgets.QDoubleSpinBox(); self.t_max_spin.setRange(0,3000); self.t_max_spin.setValue(900.0); self.t_max_spin.setSuffix(" \N{DEGREE SIGN}C")
        self.np_spin = QtWidgets.QSpinBox(); self.np_spin.setRange(2,200); self.np_spin.setValue(20)
        self.nt_spin = QtWidgets.QSpinBox(); self.nt_spin.setRange(2,200); self.nt_spin.setValue(20)
        grid_form.addRow("P min:", self.p_min_spin)
        grid_form.addRow("P max:", self.p_max_spin)
        grid_form.addRow("T min:", self.t_min_spin)
        grid_form.addRow("T max:", self.t_max_spin)
        grid_form.addRow("n(P):", self.np_spin)
        grid_form.addRow("n(T):", self.nt_spin)
        self._grid_widget = QtWidgets.QWidget()
        self._grid_widget.setLayout(grid_form)
        self._grid_widget.setVisible(False)
        mode_layout.addWidget(self._grid_widget)
        layout.addWidget(mode_group)

        self.rb_single.toggled.connect(lambda on: self._update_mode(on))
        self.cmb_backend.currentTextChanged.connect(self._update_backend_widgets)
        self._update_backend_widgets(self.cmb_backend.currentText())

        # Workflow selection
        wf_group = QtWidgets.QGroupBox("Workflow")
        wf_layout = QtWidgets.QVBoxLayout(wf_group)
        self.rb_pseudo   = QtWidgets.QRadioButton("Pseudosection only")
        self.rb_estimate = QtWidgets.QRadioButton("P-T estimate only")
        self.rb_both     = QtWidgets.QRadioButton("Both (pseudosection + estimate overlay)")
        self.rb_pseudo.setChecked(True)
        self._wf_grp = QtWidgets.QButtonGroup(self)
        for rb in (self.rb_pseudo, self.rb_estimate, self.rb_both):
            wf_layout.addWidget(rb)
            self._wf_grp.addButton(rb)
        layout.addWidget(wf_group)

        # Observed assemblage (visible only for estimate workflows)
        self._assem_group = QtWidgets.QGroupBox("Observed assemblage")
        ag_layout = QtWidgets.QVBoxLayout(self._assem_group)
        self.rb_assem_now   = QtWidgets.QRadioButton("Enter now:")
        self.rb_assem_after = QtWidgets.QRadioButton("Select from results after running")
        self.rb_assem_now.setChecked(True)
        self.assem_edit = QtWidgets.QLineEdit()
        self.assem_edit.setPlaceholderText("phase names, comma-separated  e.g. Cpx, Grt, Pl")
        now_row = QtWidgets.QHBoxLayout()
        now_row.addWidget(self.rb_assem_now)
        now_row.addWidget(self.assem_edit)
        ag_layout.addLayout(now_row)
        ag_layout.addWidget(self.rb_assem_after)
        self.rb_assem_now.toggled.connect(self.assem_edit.setEnabled)
        layout.addWidget(self._assem_group)
        self._assem_group.setVisible(False)

        for rb in (self.rb_pseudo, self.rb_estimate, self.rb_both):
            rb.toggled.connect(self._update_workflow_widgets)

        # Buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.resize(420, 680)

    def _hline(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        return line

    def _on_accept(self):
        err = self.validate()
        if err:
            QtWidgets.QMessageBox.warning(self, "Composition error", err)
            return
        self.accept()

    def _update_mode(self, single_on: bool):
        self._grid_widget.setVisible(not single_on)
        self.p_spin.setVisible(single_on)
        self.t_spin.setVisible(single_on)

    def _update_workflow_widgets(self):
        needs_estimate = self.rb_estimate.isChecked() or self.rb_both.isChecked()
        self._assem_group.setVisible(needs_estimate)
        if needs_estimate:
            self.rb_grid.setChecked(True)
            self.rb_single.setEnabled(False)
        else:
            self.rb_single.setEnabled(True)

    def _update_backend_widgets(self, backend: str):
        is_magemin  = backend == "magemin"
        is_calphad  = backend == "pycalphad"
        is_perplex  = backend == "perplex"
        self._magemin_db_label.setVisible(is_magemin)
        self.cmb_magemin_db.setVisible(is_magemin)
        self._tdb_row_label.setVisible(is_calphad)
        self.tdb_edit.parentWidget().setVisible(is_calphad)
        self._px_row_label.setVisible(is_perplex)
        self.px_dir_edit.parentWidget().setVisible(is_perplex)
        self._px_preset_label.setVisible(is_perplex)
        self.cmb_px_preset.setVisible(is_perplex)
        self._px_fluid_label.setVisible(is_perplex)
        self.cmb_px_fluid.setVisible(is_perplex)
        self._px_phases_label.setVisible(is_perplex)
        self.px_phases_edit.setVisible(is_perplex)

    def _browse_tdb(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select TDB file", "", "TDB files (*.tdb);;All files (*)"
        )
        if path:
            self.tdb_edit.setText(path)

    def _browse_px_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Perple_X directory"
        )
        if path:
            self.px_dir_edit.setText(path)

    def get_options(self) -> dict:
        backend = self.cmb_backend.currentText()
        bkw: dict = {}
        if backend == "magemin":
            bkw["db"] = self.cmb_magemin_db.currentData()
            bkw["fe3_fraction"] = float(self.fe3_spin.value())
        elif backend == "pycalphad":
            bkw["tdb_path"] = self.tdb_edit.text().strip()
        elif backend == "perplex":
            px = self.px_dir_edit.text().strip()
            if px:
                bkw["perplex_dir"] = px
            bkw["fe3_fraction"] = float(self.fe3_spin.value())
            bkw["fluid"] = self.cmb_px_fluid.currentData()
            phases_text = self.px_phases_edit.text().strip()
            if phases_text:
                bkw["solution_phases"] = [p.strip() for p in phases_text.split(",") if p.strip()]

        # Resolve effective composition (element wt%)
        if self._rb_custom.isChecked():
            comp_el = self._parse_custom_composition()
        else:
            comp_el = dict(self._comp_el)

        workflow = ("pseudo"   if self.rb_pseudo.isChecked() else
                    "estimate" if self.rb_estimate.isChecked() else "both")
        assem_method = "now" if self.rb_assem_now.isChecked() else "after"
        observed: List[str] = []
        if assem_method == "now":
            observed = [s.strip() for s in self.assem_edit.text().split(",") if s.strip()]

        opts = {
            "backend": backend,
            "backend_kwargs": bkw,
            "comp_el": comp_el,
            "comp_source": "custom" if self._rb_custom.isChecked() else "ROI",
            "fe3": float(self.fe3_spin.value()),
            "h2o": float(self.h2o_spin.value()),
            "co2": float(self.co2_spin.value()),
            "mode": "single" if self.rb_single.isChecked() else "grid",
            "p_gpa": float(self.p_spin.value()),
            "t_c":   float(self.t_spin.value()),
            "p_range": (float(self.p_min_spin.value()), float(self.p_max_spin.value())),
            "t_range": (float(self.t_min_spin.value()), float(self.t_max_spin.value())),
            "n_p": int(self.np_spin.value()),
            "n_t": int(self.nt_spin.value()),
            "workflow": workflow,
            "assem_method": assem_method,
            "observed_phases": observed,
        }
        return opts

    def _parse_custom_composition(self) -> Dict[str, float]:
        """Parse 'SiO2:45.4, Al2O3:9.0, …' - accepts oxide or element names."""
        text = self._custom_edit.text().strip()
        result: Dict[str, float] = {}
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                name, val = token.split(":", 1)
            elif "=" in token:
                name, val = token.split("=", 1)
            else:
                continue
            try:
                result[name.strip()] = float(val.strip())
            except ValueError:
                pass
        return result

    def validate(self) -> Optional[str]:
        """Return an error string if the dialog is in an invalid state, else None."""
        if self._rb_custom.isChecked():
            comp = self._parse_custom_composition()
            if not comp:
                return ("Custom composition is empty or unparseable.\n"
                        "Use the format: SiO2:45.4, Al2O3:9.0, FeO:8.0, …")
        return None


# -------------------------
# Multi-scale composition: settings dialog
# -------------------------

class _CompositionScaleDialog(QtWidgets.QDialog):
    """Settings dialog for the 'Composition vs scale …' tool.

    The maximum quadtree depth and the per-level window count are
    auto-derived from the ROI's pixel dimensions, so the dialog only ever
    offers length scales that actually subdivide for this ROI.
    """

    MIN_WINDOW_PX_EDS = 4
    DEFAULT_DEPTH_HINT = 3
    # Per-level default cap on windows to compute. A deep level has up to
    # 4**level windows; computing every one is both unnecessary (a few hundred
    # already pin the per-scale spread) and ruinously slow. Levels with fewer
    # windows than this are computed in full; the user can raise the cap.
    DEFAULT_MAX_WINDOWS = 256

    def __init__(self,
                 roi_bbox: Tuple[int, int, int, int],
                 has_eds: bool,
                 has_spx_tiles: bool,
                 element_count: int,
                 spx_min_window_px: Optional[int] = None,
                 fullres_cache_available: bool = False,
                 eds_ds: int = 1,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Composition vs length scale — settings")
        self._roi_bbox = tuple(roi_bbox)
        self._spx_min_window_px = (int(spx_min_window_px)
                                    if spx_min_window_px else None)

        # Per-source depth caps. EDS uses the small absolute floor; SPX uses
        # the median tile size (windows below that can't satisfy the 95%
        # tile-overlap rule, so they'd never produce a composition).
        self._eds_max_depth = _max_useful_quadtree_depth(
            self._roi_bbox, min_window_px=self.MIN_WINDOW_PX_EDS
        )
        if self._spx_min_window_px is not None and self._spx_min_window_px > 0:
            self._spx_max_depth = _max_useful_quadtree_depth(
                self._roi_bbox, min_window_px=self._spx_min_window_px
            )
        else:
            self._spx_max_depth = 0
        # SPX is only viable if there is at least one usable subdivision
        spx_viable = has_spx_tiles and self._spx_max_depth >= 1

        if self._eds_max_depth < 1:
            self._eds_max_depth = 1  # caller should have validated

        y0, y1, x0, x1 = self._roi_bbox
        roi_h = y1 - y0
        roi_w = x1 - x0

        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel(
            f"Tile the active ROI ({roi_h} × {roi_w} px) into a quadtree and "
            "compute a composition for the chosen number of windows at each "
            "scale. The spread per scale shows how the bulk composition "
            "depends on length scale."
        ))
        info_bits = [
            f"EDS max depth: <b>{self._eds_max_depth}</b> "
            f"(window short side ≥ {self.MIN_WINDOW_PX_EDS} px)"
        ]
        if has_spx_tiles:
            if spx_viable:
                info_bits.append(
                    f"SPX max depth: <b>{self._spx_max_depth}</b> "
                    f"(window short side ≥ {self._spx_min_window_px} px = "
                    "median SPX tile size)"
                )
            else:
                info_bits.append(
                    f"SPX disabled: ROI is smaller than one SPX tile "
                    f"(~{self._spx_min_window_px} px) - no window could "
                    "contain a tile at 95% overlap."
                )
        layout.addWidget(QtWidgets.QLabel("<i>" + "<br>".join(info_bits) + "</i>"))

        form = QtWidgets.QFormLayout()
        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setToolTip(
            "Number of subdivisions below the full ROI. Depth=3 gives 4 "
            "levels: full, 1/4, 1/16, 1/64. The maximum is constrained by "
            "the ROI size and (for SPX) the tile size."
        )
        form.addRow("Quadtree depth:", self.depth_spin)

        self.cmb_sampling = QtWidgets.QComboBox()
        self.cmb_sampling.addItem("First N (scan order)", "first")
        self.cmb_sampling.addItem("Random N (seed=0, reproducible)", "random")
        self.cmb_sampling.setToolTip(
            "How to pick which windows to compute when 'Max windows' is "
            "below the total at a level. 'First N' = top-left scan order. "
            "'Random N' = uniform sample with a fixed seed."
        )
        form.addRow("Sampling strategy:", self.cmb_sampling)
        layout.addLayout(form)

        # Per-level table
        wins_box = QtWidgets.QGroupBox("Windows per level")
        wins_layout = QtWidgets.QVBoxLayout(wins_box)
        wins_layout.addWidget(QtWidgets.QLabel(
            "<i>Counts and window sizes are derived from the actual ROI "
            "subdivision. 'Max windows' caps the per-level cost; deep levels "
            f"default to a {self.DEFAULT_MAX_WINDOWS}-window sample (raise to "
            "compute more). A few hundred windows already pin the per-scale "
            "spread.</i>"
        ))
        self.tbl_levels = QtWidgets.QTableWidget()
        self.tbl_levels.setColumnCount(4)
        self.tbl_levels.setHorizontalHeaderLabels(
            ["Level", "Window size", "Total windows", "Max to compute"]
        )
        self.tbl_levels.verticalHeader().setVisible(False)
        self.tbl_levels.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.tbl_levels.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        wins_layout.addWidget(self.tbl_levels)
        layout.addWidget(wins_box)

        self.rb_eds = QtWidgets.QRadioButton("EDS net-count fractions (fast)")
        self.rb_spx = QtWidgets.QRadioButton(
            "SPX whole-tile quantification (calibrated)"
        )
        self.rb_eds.setEnabled(has_eds)
        self.rb_spx.setEnabled(spx_viable)
        # Default: EDS if available, otherwise SPX
        if has_eds:
            self.rb_eds.setChecked(True)
        elif spx_viable:
            self.rb_spx.setChecked(True)

        src_box = QtWidgets.QGroupBox("Composition source")
        src_layout = QtWidgets.QVBoxLayout(src_box)
        src_layout.addWidget(self.rb_eds)
        src_layout.addWidget(self.rb_spx)
        if not has_eds:
            src_layout.addWidget(QtWidgets.QLabel(
                "<i>EDS path needs full-resolution band maps - click "
                "'Compute FULL-RES maps for ROI' first.</i>"
            ))
        if has_spx_tiles and not spx_viable:
            src_layout.addWidget(QtWidgets.QLabel(
                f"<i>SPX path disabled: ROI ({roi_h} × {roi_w} px) is too "
                f"small for the median SPX tile (~{self._spx_min_window_px} px).</i>"
            ))

        # Full-resolution opt-in: only meaningful when the EDS maps were
        # downsampled for display (ds>1) and the cache can supply full res.
        # OFF by default — re-reading a huge ROI from disk at every scale is
        # very slow, and the downsampled in-memory maps give statistically
        # identical window averages.
        self.chk_fullres_cache = QtWidgets.QCheckBox(
            "Read EDS at full resolution from cache"
        )
        self.chk_fullres_cache.setChecked(False)
        if fullres_cache_available:
            self.chk_fullres_cache.setText(
                f"Read EDS at full resolution from cache "
                f"(display is 1/{int(eds_ds)} — SLOW: re-reads the ROI at "
                f"every scale)"
            )
            self.chk_fullres_cache.setToolTip(
                "Off (default): use the maps already in memory. They are at "
                f"1/{int(eds_ds)} resolution for this large ROI, which is fine "
                "for window-averaged composition.\n\n"
                "On: read each window's full-resolution data from the cache. "
                "Much slower on large mosaics (re-reads the whole ROI at every "
                "quadtree level) and rarely changes the result."
            )
            src_layout.addWidget(self.chk_fullres_cache)
        else:
            self.chk_fullres_cache.hide()
        layout.addWidget(src_box)

        form2 = QtWidgets.QFormLayout()
        self.min_mask_spin = QtWidgets.QDoubleSpinBox()
        self.min_mask_spin.setRange(0.0, 1.0)
        self.min_mask_spin.setSingleStep(0.05)
        self.min_mask_spin.setDecimals(2)
        self.min_mask_spin.setValue(0.5)
        self.min_mask_spin.setToolTip(
            "Skip a window if less than this fraction of its pixels falls "
            "inside the ROI polygon (relevant for non-rectangular ROIs)."
        )
        form2.addRow("Min mask coverage:", self.min_mask_spin)
        layout.addLayout(form2)

        layout.addWidget(QtWidgets.QLabel(
            f"Element subset: <b>{element_count}</b> elements currently "
            "selected in the panel will be used."
        ))

        # OK/Cancel
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        # Initialise depth bounds + table; react to source toggling and to
        # depth-spinner changes.
        self._on_source_changed()
        self.depth_spin.valueChanged.connect(self._refresh_level_table)
        self.rb_eds.toggled.connect(lambda _: self._on_source_changed())
        self.rb_spx.toggled.connect(lambda _: self._on_source_changed())

        self.resize(440, 560)

    def _on_source_changed(self):
        """Re-clamp the depth spinner whenever the source toggles."""
        if self.rb_spx.isChecked():
            new_max = max(1, self._spx_max_depth)
        else:
            new_max = max(1, self._eds_max_depth)
        self.depth_spin.blockSignals(True)
        try:
            self.depth_spin.setRange(1, new_max)
            cur = self.depth_spin.value()
            if cur > new_max:
                self.depth_spin.setValue(new_max)
            elif cur < 1:
                self.depth_spin.setValue(1)
            elif cur == 0:  # uninitialised
                self.depth_spin.setValue(min(self.DEFAULT_DEPTH_HINT, new_max))
        finally:
            self.depth_spin.blockSignals(False)
        # If we never set a value yet, seed it now
        if self.depth_spin.value() < 1:
            self.depth_spin.setValue(min(self.DEFAULT_DEPTH_HINT, new_max))
        self._refresh_level_table(self.depth_spin.value())

    def _refresh_level_table(self, depth: int):
        """Recompute the quadtree for the active ROI at the given depth and
        repopulate the table from the real per-level window counts and
        pixel sizes. Preserves user-edited caps that still apply.
        """
        prev: Dict[int, int] = {}
        for r in range(self.tbl_levels.rowCount()):
            try:
                item = self.tbl_levels.item(r, 0)
                if item is None:
                    continue
                lvl = int(item.text().lstrip("L"))
                w = self.tbl_levels.cellWidget(r, 3)
                if w is not None:
                    prev[lvl] = int(w.value())
            except Exception:
                continue

        # Per-level counts and window sizes computed analytically (two O(2**k)
        # axis splits) — NOT by materialising the O(4**k) windows, which would
        # freeze the dialog for a deep quadtree on a large ROI.
        sorted_levels = list(range(int(depth) + 1))
        self.tbl_levels.setRowCount(len(sorted_levels))
        for r, k in enumerate(sorted_levels):
            rows_k, cols_k, (wh, ww) = _quadtree_level_dims(self._roi_bbox, k)
            count = rows_k * cols_k
            size_label = f"{wh} × {ww} px" if count else "—"

            self.tbl_levels.setItem(r, 0, QtWidgets.QTableWidgetItem(f"L{k}"))
            self.tbl_levels.setItem(r, 1, QtWidgets.QTableWidgetItem(size_label))
            self.tbl_levels.setItem(r, 2, QtWidgets.QTableWidgetItem(str(count)))
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, max(1, count))
            # Default cap is bounded so deep levels don't compute every window.
            default_cap = min(count, self.DEFAULT_MAX_WINDOWS)
            spin.setValue(min(prev.get(k, default_cap), count))
            if count <= 1:
                spin.setEnabled(False)
            self.tbl_levels.setCellWidget(r, 3, spin)
        self.tbl_levels.resizeColumnsToContents()

    def get_options(self) -> dict:
        # Read per-level caps from the table
        max_windows: Dict[int, int] = {}
        for r in range(self.tbl_levels.rowCount()):
            try:
                item = self.tbl_levels.item(r, 0)
                if item is None:
                    continue
                lvl = int(item.text().lstrip("L"))
                w = self.tbl_levels.cellWidget(r, 3)
                if w is not None:
                    max_windows[lvl] = int(w.value())
            except Exception:
                continue
        return {
            "max_depth": int(self.depth_spin.value()),
            "source": "eds" if self.rb_eds.isChecked() else "spx",
            "min_mask": float(self.min_mask_spin.value()),
            "max_windows": max_windows,
            "sampling": str(self.cmb_sampling.currentData() or "first"),
            "fullres_cache": bool(self.chk_fullres_cache.isChecked()),
            # net-count background the windows are computed with (provenance)
            "background": __import__("eds_background").default_scheme(),
        }


# -------------------------
# Phase segmentation dialogs
# -------------------------

class _PhaseSegDialog(QtWidgets.QDialog):
    """Settings dialog for phase segmentation: data source, n_clusters, normalization."""

    def __init__(
        self,
        has_previews: bool,
        has_roi: bool,
        has_fullres: bool,
        preview_elements: Optional[List[str]] = None,
        fullres_elements: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._preview_elements = sorted(preview_elements or [])
        self._fullres_elements = sorted(fullres_elements or [])
        self.setWindowTitle("Phase segmentation")

        layout = QtWidgets.QVBoxLayout(self)

        # Data source
        src_grp = QtWidgets.QGroupBox("Data source")
        src_lay = QtWidgets.QVBoxLayout(src_grp)
        self.rb_full = QtWidgets.QRadioButton("Full mosaic preview maps")
        self.rb_roi_ds = QtWidgets.QRadioButton("ROI from preview (cropped)")
        self.rb_roi_full = QtWidgets.QRadioButton("ROI full-resolution maps")
        self.rb_full.setEnabled(has_previews)
        self.rb_roi_ds.setEnabled(has_previews and has_roi)
        self.rb_roi_full.setEnabled(has_fullres)
        src_lay.addWidget(self.rb_full)
        src_lay.addWidget(self.rb_roi_ds)
        src_lay.addWidget(self.rb_roi_full)
        if has_fullres:
            self.rb_roi_full.setChecked(True)
        elif has_previews and has_roi:
            self.rb_roi_ds.setChecked(True)
        elif has_previews:
            self.rb_full.setChecked(True)
        layout.addWidget(src_grp)

        # Element selection (hidden behind button)
        self._element_selection: Dict[str, bool] = {}
        el_row = QtWidgets.QHBoxLayout()
        el_row.addWidget(QtWidgets.QLabel("Elements for segmentation:"))
        self._btn_elements = QtWidgets.QPushButton("All elements")
        self._btn_elements.setToolTip(
            "Click to choose which elements are used for segmentation.\n"
            "Uncheck elements to exclude them from clustering."
        )
        self._btn_elements.clicked.connect(self._open_element_dialog)
        el_row.addWidget(self._btn_elements)
        el_row.addStretch()
        layout.addLayout(el_row)

        for rb in (self.rb_full, self.rb_roi_ds, self.rb_roi_full):
            rb.toggled.connect(self._refresh_element_button)
        self._refresh_element_button()

        # Clustering
        clust_grp = QtWidgets.QGroupBox("Clustering")
        clust_lay = QtWidgets.QVBoxLayout(clust_grp)

        auto_row = QtWidgets.QHBoxLayout()
        self.rb_auto = QtWidgets.QRadioButton("Auto-detect (GMM-BIC)")
        self.rb_auto.setChecked(True)
        auto_row.addWidget(self.rb_auto)
        auto_row.addWidget(QtWidgets.QLabel("Max k:"))
        self.spin_kmax = QtWidgets.QSpinBox()
        self.spin_kmax.setRange(3, 30)
        self.spin_kmax.setValue(15)
        auto_row.addWidget(self.spin_kmax)
        auto_row.addWidget(QtWidgets.QLabel("  Min k:"))
        self.spin_mink = QtWidgets.QSpinBox()
        self.spin_mink.setRange(2, 10)
        self.spin_mink.setValue(2)
        self.spin_mink.setToolTip(
            "Auto-k will never select fewer clusters than this.\n"
            "Set to 3+ when you know the sample has more than 2 phases."
        )
        auto_row.addWidget(self.spin_mink)
        auto_row.addStretch()
        clust_lay.addLayout(auto_row)

        manual_row = QtWidgets.QHBoxLayout()
        self.rb_manual = QtWidgets.QRadioButton("Manual")
        manual_row.addWidget(self.rb_manual)
        manual_row.addWidget(QtWidgets.QLabel("Clusters:"))
        self.spin_k = QtWidgets.QSpinBox()
        self.spin_k.setRange(2, 20)
        self.spin_k.setValue(5)
        self.spin_k.setEnabled(False)
        manual_row.addWidget(self.spin_k)
        manual_row.addStretch()
        clust_lay.addLayout(manual_row)

        pca_row = QtWidgets.QHBoxLayout()
        pca_row.addWidget(QtWidgets.QLabel("PCA variance:"))
        self.spin_pca_var = QtWidgets.QDoubleSpinBox()
        self.spin_pca_var.setRange(0.70, 0.99)
        self.spin_pca_var.setValue(0.92)
        self.spin_pca_var.setSingleStep(0.01)
        self.spin_pca_var.setDecimals(2)
        self.spin_pca_var.setToolTip(
            "Cumulative variance threshold for PCA dimensionality reduction.\n"
            "Lower values = fewer PCA components = stronger noise suppression,\n"
            "but may lose subtle phase distinctions.\n"
            "0.90-0.95 works well for most geological samples."
        )
        pca_row.addWidget(self.spin_pca_var)
        pca_row.addStretch()
        clust_lay.addLayout(pca_row)

        self._clust_grp = QtWidgets.QButtonGroup(self)
        self._clust_grp.addButton(self.rb_auto)
        self._clust_grp.addButton(self.rb_manual)
        self.rb_auto.toggled.connect(lambda on: (
            self.spin_kmax.setEnabled(on),
            self.spin_mink.setEnabled(on),
            self.spin_k.setEnabled(not on),
        ))

        layout.addWidget(clust_grp)

        # Preprocessing
        pre_grp = QtWidgets.QGroupBox("Preprocessing")
        pre_lay = QtWidgets.QFormLayout(pre_grp)
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setRange(0.0, 20.0)
        self.spin_sigma.setValue(3.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setToolTip(
            "Gaussian smoothing sigma (pixels). Reduces per-pixel EDS noise.\n"
            "Higher values = smoother phases but less spatial detail.\n"
            "0 = no smoothing (not recommended for raw EDS data)."
        )
        pre_lay.addRow("Spatial smoothing (sigma):", self.spin_sigma)
        self.spin_minsize = QtWidgets.QSpinBox()
        self.spin_minsize.setRange(0, 500)
        self.spin_minsize.setValue(50)
        self.spin_minsize.setToolTip(
            "Minimum cluster region size (pixels). Isolated groups\n"
            "smaller than this are reassigned to neighboring phases.\n"
            "0 = no cleanup."
        )
        pre_lay.addRow("Min. region size (cleanup):", self.spin_minsize)
        self.cmb_norm = QtWidgets.QComboBox()
        self.cmb_norm.addItem("Row-normalize (compositions)", "row")
        self.cmb_norm.addItem("Anscombe √counts (low-count data)", "anscombe")
        self.cmb_norm.setToolTip(
            "Row-normalize: converts each pixel to apparent element fractions "
            "(sum=1).\n"
            "  Best when count rates are reasonable (>20 counts/pixel).\n\n"
            "Anscombe √counts: variance-stabilizing transform for Poisson data.\n"
            "  Better when counts are very low (~6/pixel); preserves absolute\n"
            "  count information instead of normalizing it away."
        )
        pre_lay.addRow("Normalization:", self.cmb_norm)
        self.spin_min_count_pct = QtWidgets.QSpinBox()
        self.spin_min_count_pct.setRange(0, 30)
        self.spin_min_count_pct.setValue(5)
        self.spin_min_count_pct.setSingleStep(5)
        self.spin_min_count_pct.setSuffix(" %")
        self.spin_min_count_pct.setToolTip(
            "Discard pixels whose total EDS count falls below this percentile\n"
            "of all non-zero pixels before clustering.\n"
            "5-15 % removes the noisiest pixels without discarding phase data.\n"
            "0 = no filter."
        )
        pre_lay.addRow("Min. count percentile:", self.spin_min_count_pct)
        self.spin_min_count_ratio = QtWidgets.QDoubleSpinBox()
        self.spin_min_count_ratio.setRange(0.0, 0.50)
        self.spin_min_count_ratio.setValue(0.10)
        self.spin_min_count_ratio.setSingleStep(0.05)
        self.spin_min_count_ratio.setDecimals(2)
        self.spin_min_count_ratio.setToolTip(
            "Exclude pixels whose total EDS count is below this fraction\n"
            "of the median total count.  Targets low-count tiles that have\n"
            "too little signal for reliable phase identification.\n"
            "0.10 = exclude below 10% of median counts. 0 = disabled.\n"
            "Combined with the percentile filter (whichever is stricter wins)."
        )
        pre_lay.addRow("Min. count ratio (vs median):", self.spin_min_count_ratio)
        self.spin_max_zero_pct = QtWidgets.QDoubleSpinBox()
        self.spin_max_zero_pct.setRange(50.0, 100.0)
        self.spin_max_zero_pct.setValue(90.0)
        self.spin_max_zero_pct.setSingleStep(5.0)
        self.spin_max_zero_pct.setDecimals(0)
        self.spin_max_zero_pct.setSuffix(" %")
        self.spin_max_zero_pct.setToolTip(
            "Auto-exclude elements whose raw (pre-smoothing) zero fraction\n"
            "among non-background pixels exceeds this value.\n"
            "Trace/contamination elements (e.g. C, P) are almost always zero\n"
            "but their rare non-zero values create artificial clusters.\n"
            "90% is a safe default. 100% = no exclusion."
        )
        pre_lay.addRow("Sparse element threshold:", self.spin_max_zero_pct)
        self.chk_exclude_overlap = QtWidgets.QCheckBox("Exclude tile overlap zones")
        self.chk_exclude_overlap.setChecked(True)
        self.chk_exclude_overlap.setToolTip(
            "Exclude pixels in tile overlap zones from phase clustering.\n"
            "Overlap blending creates artificial intermediate compositions\n"
            "that the GMM may interpret as separate phases.\n"
            "Excluded pixels are filled in post-clustering by nearest-neighbor."
        )
        pre_lay.addRow(self.chk_exclude_overlap)

        # Unclassified rejection — leave mixed/atypical pixels out instead of
        # forcing every pixel into a phase.
        rej_row = QtWidgets.QHBoxLayout()
        self.chk_reject = QtWidgets.QCheckBox("Leave ambiguous pixels unclassified")
        self.chk_reject.setChecked(False)
        self.chk_reject.setToolTip(
            "Reject pixels whose max GMM posterior falls below the threshold.\n"
            "These are grain-boundary mixed signals that sit genuinely between\n"
            "two phases; forcing them into a cluster creates spurious\n"
            "intermediate domains (duplicate boundary phases, mixed-signal\n"
            "mineral IDs). Rejected pixels stay unclassified (transparent),\n"
            "and are excluded from modal proportions, per-phase composition\n"
            "and mineral ID. Off = classic behaviour (every pixel assigned)."
        )
        rej_row.addWidget(self.chk_reject)
        rej_row.addWidget(QtWidgets.QLabel("  max posterior <"))
        self.spin_reject_post = QtWidgets.QDoubleSpinBox()
        self.spin_reject_post.setRange(0.50, 0.99)
        self.spin_reject_post.setValue(0.60)
        self.spin_reject_post.setSingleStep(0.05)
        self.spin_reject_post.setDecimals(2)
        self.spin_reject_post.setEnabled(False)
        self.spin_reject_post.setToolTip(
            "Ambiguity threshold. 0.60 rejects only pixels the GMM genuinely\n"
            "cannot decide between two components; 0.80+ is aggressive and\n"
            "starts eating into fine-grained phases (their pixels are mostly\n"
            "near boundaries)."
        )
        rej_row.addWidget(self.spin_reject_post)
        rej_row.addStretch()
        pre_lay.addRow(rej_row)
        self.chk_reject.toggled.connect(self.spin_reject_post.setEnabled)

        self.spin_reject_dens = QtWidgets.QDoubleSpinBox()
        self.spin_reject_dens.setRange(0.0, 20.0)
        self.spin_reject_dens.setValue(0.0)
        self.spin_reject_dens.setSingleStep(0.5)
        self.spin_reject_dens.setDecimals(1)
        self.spin_reject_dens.setSuffix(" %")
        self.spin_reject_dens.setToolTip(
            "Additionally reject *atypical* pixels: those whose likelihood\n"
            "under the whole mixture falls in the lowest N percent. Catches\n"
            "outliers that resemble no phase at all (cracks, epoxy, holes,\n"
            "contamination) — posteriors cannot, because they are relative\n"
            "(a far outlier still gets a confident posterior for its nearest\n"
            "component). 0 = disabled; 1-2 % is a sensible starting point."
        )
        pre_lay.addRow("Reject atypical pixels (density):", self.spin_reject_dens)

        self.chk_background = QtWidgets.QCheckBox(
            "Leave zero-signal (no-data) pixels unclassified")
        self.chk_background.setChecked(True)
        self.chk_background.setToolTip(
            "Pixels with no counts in any element (off-sample area, unscanned\n"
            "montage margins) hold no measurement, so they are not a phase:\n"
            "they stay unclassified instead of being nearest-neighbour filled\n"
            "from the sample edge, and edge pixels that only look non-empty\n"
            "because smoothing bled sample counts into the void are cleared\n"
            "too. Judged on the raw (pre-smoothing) maps. Uncheck only to\n"
            "reproduce a recipe recorded before this option existed."
        )
        pre_lay.addRow(self.chk_background)
        layout.addWidget(pre_grp)

        # Buttons
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _refresh_element_button(self):
        """Update _element_selection for currently selected source, then refresh button label."""
        if self.rb_roi_full.isChecked():
            elements = self._fullres_elements
        else:
            elements = self._preview_elements
        new_sel: Dict[str, bool] = {}
        for el in elements:
            new_sel[el] = self._element_selection.get(el, True)
        self._element_selection = new_sel
        self._update_element_button_label()

    def _update_element_button_label(self):
        """Update the element button text to reflect current selection."""
        checked = [el for el, v in self._element_selection.items() if v]
        total = len(self._element_selection)
        if not total:
            self._btn_elements.setText("No elements")
        elif len(checked) == total:
            self._btn_elements.setText(f"All {total} elements ▾")
        else:
            els = ", ".join(checked[:5])
            if len(checked) > 5:
                els += f" +{len(checked) - 5}"
            self._btn_elements.setText(f"{len(checked)}/{total}: {els} ▾")

    def _open_element_dialog(self):
        """Open a small dialog to select/deselect individual elements."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select elements for segmentation")
        lay = QtWidgets.QVBoxLayout(dlg)

        lbl = QtWidgets.QLabel(
            "Check elements to include in segmentation.\n"
            "Unchecked elements are excluded from clustering."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for el, is_checked in self._element_selection.items():
            item = QtWidgets.QListWidgetItem(el)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if is_checked else QtCore.Qt.Unchecked)
            lst.addItem(item)
        lay.addWidget(lst)

        btn_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select all")
        btn_none = QtWidgets.QPushButton("Deselect all")
        btn_all.setMaximumWidth(90)
        btn_none.setMaximumWidth(90)
        btn_all.clicked.connect(
            lambda: [lst.item(i).setCheckState(QtCore.Qt.Checked) for i in range(lst.count())]
        )
        btn_none.clicked.connect(
            lambda: [lst.item(i).setCheckState(QtCore.Qt.Unchecked) for i in range(lst.count())]
        )
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            for i in range(lst.count()):
                el = lst.item(i).text()
                self._element_selection[el] = (
                    lst.item(i).checkState() == QtCore.Qt.Checked
                )
            self._update_element_button_label()

    def get_options(self) -> dict:
        if self.rb_roi_full.isChecked():
            source = "fullres_roi"
        elif self.rb_roi_ds.isChecked():
            source = "preview_roi"
        else:
            source = "preview_full"

        all_els = list(self._element_selection.keys())
        checked_els = [el for el, v in self._element_selection.items() if v]
        elements = checked_els if len(checked_els) < len(all_els) else None

        return {
            "source": source,
            "elements": elements,
            "n_clusters": "auto" if self.rb_auto.isChecked() else self.spin_k.value(),
            "k_max": self.spin_kmax.value(),
            "min_k": self.spin_mink.value(),
            "norm_method": self.cmb_norm.currentData(),
            "smooth_sigma": self.spin_sigma.value(),
            "min_region_size": self.spin_minsize.value(),
            "min_count_pct": self.spin_min_count_pct.value(),
            "min_count_ratio": self.spin_min_count_ratio.value(),
            "max_zero_pct": self.spin_max_zero_pct.value(),
            "pca_variance": self.spin_pca_var.value(),
            "exclude_overlap": self.chk_exclude_overlap.isChecked(),
            "reject_posterior_min": (self.spin_reject_post.value()
                                     if self.chk_reject.isChecked() else 0.0),
            "reject_density_pct": self.spin_reject_dens.value(),
            "protect_background": self.chk_background.isChecked(),
        }


class _PhaseCompositionDialog(QtWidgets.QDialog):
    """All-phases EDS composition: phase×element table + stacked bar chart.

    Reusable both inside the classification dialog and from the main panel, so the
    breakdown stays available after the segmentation window is closed.
    ``compositions`` is ``{cid: {element: fraction}}`` (fractions 0..1), ``npx`` is
    ``{cid: pixel_count}``, ``names`` is ``{cid: label}``, ``units`` is the y-label.
    """

    def __init__(self, compositions, npx, names, units, parent=None):
        super().__init__(parent)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        self.setWindowTitle(f"Composition breakdown — all phases  ({units})")
        self.resize(840, 560)
        lay = QtWidgets.QVBoxLayout(self)

        cids = sorted(compositions.keys())
        totals: Dict[str, float] = {}
        for cid in cids:
            for el, v in compositions[cid].items():
                totals[el] = totals.get(el, 0.0) + float(v)
        elements = [e for e, _ in sorted(totals.items(), key=lambda kv: -kv[1])]

        if not cids or not elements:
            lay.addWidget(QtWidgets.QLabel("No composition data available."))
            b = QtWidgets.QPushButton("Close")
            b.clicked.connect(self.accept)
            lay.addWidget(b)
            return

        labels = [names.get(cid, f"Phase {cid + 1}") for cid in cids]
        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs)

        tbl = QtWidgets.QTableWidget()
        tbl.setColumnCount(2 + len(elements))
        tbl.setHorizontalHeaderLabels(["Phase", "n_px"] + elements)
        tbl.setRowCount(len(cids))
        tbl.verticalHeader().setVisible(False)
        for r, cid in enumerate(cids):
            frac = compositions[cid]
            tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(labels[r]))
            tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{int(npx.get(cid, 0)):,}"))
            for c, el in enumerate(elements):
                v = 100.0 * float(frac.get(el, 0.0))
                tbl.setItem(r, 2 + c, QtWidgets.QTableWidgetItem(f"{v:.2f}" if v > 0 else ""))
        tbl.resizeColumnsToContents()
        tabs.addTab(tbl, "Table")

        fig = plt.figure(figsize=(7, max(2.0, 0.5 * len(cids))))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        cmap = matplotlib.colormaps['tab20']
        y = np.arange(len(cids))
        left = np.zeros(len(cids))
        for j, el in enumerate(elements):
            vals = np.array([100.0 * float(compositions[cid].get(el, 0.0)) for cid in cids])
            ax.barh(y, vals, left=left, color=cmap(j % 20), label=el, edgecolor="none")
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(units)
        ax.set_title("Per-phase composition")
        ax.legend(fontsize=6, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        canvas.draw()
        chart_w = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(chart_w).addWidget(canvas)
        tabs.addTab(chart_w, "Chart")
        self.finished.connect(lambda *_: plt.close(fig))

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)


class _CompositionCompareDialog(QtWidgets.QDialog):
    """Side-by-side comparison of an ROI's composition from several sources.

    ``sources`` is an ordered list of ``(label, {element: value}, available)``;
    each composition is per-element and already in its own units (EDS net-count
    %, calibrated wt%, or SPX wt% — all normalised to ~100, so visually
    comparable). Renders an element×source table (with a Δ column between the
    last two *available* sources, i.e. calibrated−SPX, when ≥2 are present) and
    a grouped bar chart, plus a CSV export. ``notes`` is a list of caveat lines.

    A **"Compare cations only (exclude O, C)"** checkbox renormalises every
    column over its cations only — an O-free, assumption-free basis that
    sidesteps the fact that O is measured intensity in the net-count column,
    stoichiometric in the calibrated/SPX columns.
    """

    _OXYGEN_LIKE = ("O", "C")

    def __init__(self, sources, notes=None, parent=None):
        super().__init__(parent)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        self.setWindowTitle("Compare ROI compositions (EDS / calibrated / SPX)")
        self.resize(720, 600)
        lay = QtWidgets.QVBoxLayout(self)

        present = [(lbl, dict(comp)) for lbl, comp, ok in sources if ok and comp]
        missing = [lbl for lbl, comp, ok in sources if not (ok and comp)]
        self._present = present       # raw (O-bearing) sources, never mutated
        self._view = present          # what the table/chart currently show

        if not present:
            lay.addWidget(QtWidgets.QLabel(
                "No composition sources are available for this ROI."))
            b = QtWidgets.QPushButton("Close")
            b.clicked.connect(self.accept)
            lay.addWidget(b)
            return

        self._cation_cb = QtWidgets.QCheckBox("Compare cations only (exclude O, C)")
        self._cation_cb.setToolTip(
            "Drop O and C from every column and renormalise each to 100, so the "
            "comparison reflects cation proportions only (no oxygen-basis "
            "assumptions).")
        self._cation_cb.toggled.connect(self._render)
        lay.addWidget(self._cation_cb)

        tabs = QtWidgets.QTabWidget()
        lay.addWidget(tabs)
        self._tbl = QtWidgets.QTableWidget()
        self._tbl.verticalHeader().setVisible(False)
        tabs.addTab(self._tbl, "Table")

        self._fig = plt.figure(figsize=(7.2, 4.2))
        self._canvas = FigureCanvas(self._fig)
        self._ax = self._fig.add_subplot(111)
        chart_w = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(chart_w).addWidget(self._canvas)
        tabs.addTab(chart_w, "Chart")
        self.finished.connect(lambda *_: plt.close(self._fig))

        note_lines = list(notes or [])
        if missing:
            note_lines.append("Not available: " + "; ".join(missing))
        if note_lines:
            nl = QtWidgets.QLabel("\n".join("• " + n for n in note_lines))
            nl.setWordWrap(True)
            lay.addWidget(nl)

        btn_row = QtWidgets.QHBoxLayout()
        btn_csv = QtWidgets.QPushButton("Export CSV …")
        btn_csv.clicked.connect(self._export_csv)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_csv)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self._render()

    def _current_sources(self):
        """The sources to display: cation-only (renormalised) when toggled."""
        if not self._cation_cb.isChecked():
            return self._present
        out = []
        for lbl, comp in self._present:
            cat = {el: float(v) for el, v in comp.items()
                   if el not in self._OXYGEN_LIKE and float(v) > 0}
            s = sum(cat.values())
            if s > 0:
                cat = {el: 100.0 * v / s for el, v in cat.items()}
            out.append((lbl, cat))
        return out

    def _ordered_elements(self, present):
        peak: Dict[str, float] = {}
        for _, comp in present:
            for el, v in comp.items():
                peak[el] = max(peak.get(el, 0.0), float(v))
        return [e for e, _ in sorted(peak.items(), key=lambda kv: -kv[1])]

    def _render(self, *_):
        present = self._current_sources()
        self._view = present
        elements = self._ordered_elements(present)
        labels = [lbl for lbl, _ in present]

        # --- table ---
        show_diff = len(present) >= 2
        headers = ["Element"] + labels
        if show_diff:
            headers.append(f"Δ ({labels[-2]}−{labels[-1]})")
        ncol = len(headers)
        tbl = self._tbl
        tbl.clear()
        tbl.setColumnCount(ncol)
        tbl.setHorizontalHeaderLabels(headers)
        tbl.setRowCount(len(elements))
        for r, el in enumerate(elements):
            tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(el))
            for c, (_, comp) in enumerate(present):
                v = float(comp.get(el, 0.0))
                tbl.setItem(r, 1 + c,
                            QtWidgets.QTableWidgetItem(f"{v:.2f}" if v else ""))
            if show_diff:
                a = float(present[-2][1].get(el, 0.0))
                b = float(present[-1][1].get(el, 0.0))
                tbl.setItem(r, ncol - 1, QtWidgets.QTableWidgetItem(
                    f"{a - b:+.2f}" if (a or b) else ""))
        tbl.resizeColumnsToContents()

        # --- chart ---
        ax = self._ax
        ax.clear()
        x = np.arange(len(elements))
        n = len(present)
        width = 0.8 / max(1, n)
        cmap = matplotlib.colormaps['tab10']
        for j, (lbl, comp) in enumerate(present):
            vals = [float(comp.get(el, 0.0)) for el in elements]
            ax.bar(x + (j - (n - 1) / 2.0) * width, vals, width,
                   label=lbl, color=cmap(j % 10))
        ax.set_xticks(x)
        ax.set_xticklabels(elements, fontsize=8)
        ylab = ("cation wt% / net-count % (O,C excluded, ~100 total)"
                if self._cation_cb.isChecked()
                else "value (wt% / net-count %, ~100 total)")
        ax.set_ylabel(ylab)
        ax.set_title("ROI composition by source")
        ax.legend(fontsize=7)
        self._fig.tight_layout()
        self._canvas.draw()

    def _export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export composition comparison", "composition_comparison.csv",
            "CSV files (*.csv)")
        if not path:
            return
        present = self._view
        elements = self._ordered_elements(present)
        labels = [lbl for lbl, _ in present]
        lines = ["element," + ",".join(f'"{l}"' for l in labels)]
        for el in elements:
            vals = ",".join(f"{float(comp.get(el, 0.0)):.4f}"
                            for _, comp in present)
            lines.append(f"{el},{vals}")
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))


class _MapZoomView(QtWidgets.QGraphicsView):
    """Zoom/pan graphics view for `_MapInspectDialog`.

    Mouse wheel zooms about the cursor (clamped); left-drag pans; hover
    positions are emitted in map-pixel coordinates for the readout label.
    """

    zoomed = QtCore.Signal()
    hovered = QtCore.Signal(int, int)   # (row, col) in map px; (-1,-1) off-map

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QColor(40, 40, 43))
        self.min_scale = 0.01
        self.max_scale = 64.0

    def current_scale(self) -> float:
        return float(self.transform().m11())

    def set_scale(self, s: float) -> None:
        s = min(max(float(s), self.min_scale), self.max_scale)
        cur = self.current_scale()
        if cur > 0 and s != cur:
            self.scale(s / cur, s / cur)
            self.zoomed.emit()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        self.set_scale(self.current_scale() * (1.25 if delta > 0 else 0.8))
        ev.accept()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        pos = self.mapToScene(ev.pos())
        self.hovered.emit(int(np.floor(pos.y())), int(np.floor(pos.x())))

    def leaveEvent(self, ev):
        super().leaveEvent(ev)
        self.hovered.emit(-1, -1)


class _MapInspectDialog(QtWidgets.QDialog):
    """Full-screen zoomable inspector for a phase label map.

    Opens maximised; mouse wheel zooms to the cursor (up to 64x — well past
    1:1 map pixels), left-drag pans, hovering names the domain under the
    cursor. When an aligned total-signal underlay is supplied, an opacity
    slider fades the phase colours over it so boundary placement can be
    judged against the raw EDS signal (unclassified pixels are transparent
    and show the underlay directly). Esc closes.
    """

    def __init__(
        self,
        label_map: np.ndarray,
        n_clusters: int,
        names_by_cid: Dict[int, str],
        underlay: Optional[np.ndarray] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Phase map — full-screen inspector")
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowMaximizeButtonHint
                            | QtCore.Qt.WindowMinimizeButtonHint)
        self._label_map = label_map
        self._names = dict(names_by_cid)
        H, W = label_map.shape
        self._shape = (H, W)
        self._fitted_once = False

        # Colour the labels once, full resolution. The RGBA buffer must stay
        # referenced while the QImage exists (QImage wraps, not copies).
        self._rgba = np.ascontiguousarray(_label_map_to_rgba(label_map, n_clusters))
        img = QtGui.QImage(self._rgba.data, W, H, 4 * W,
                           QtGui.QImage.Format_RGBA8888)
        phase_pm = QtGui.QPixmap.fromImage(img)

        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, W, H)
        self._item_under = None
        if underlay is not None and underlay.shape == label_map.shape:
            u = np.asarray(underlay, dtype=np.float32)
            finite = u[np.isfinite(u)]
            if finite.size:
                lo, hi = np.percentile(finite, [1.0, 99.5])
                if hi <= lo:
                    hi = lo + 1.0
                u8 = np.clip((u - lo) / (hi - lo) * 255.0, 0, 255)
                self._under8 = np.ascontiguousarray(
                    np.nan_to_num(u8).astype(np.uint8))
                uimg = QtGui.QImage(self._under8.data, W, H, W,
                                    QtGui.QImage.Format_Grayscale8)
                self._item_under = self._scene.addPixmap(
                    QtGui.QPixmap.fromImage(uimg))
        self._item_phase = self._scene.addPixmap(phase_pm)
        self._item_phase.setZValue(1)

        self._view = _MapZoomView(self._scene, self)
        self._view.zoomed.connect(self._on_zoom_changed)
        self._view.hovered.connect(self._on_hover)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        # ---- control row: fit / 1:1 / zoom readout / opacity / hover name --
        ctrl = QtWidgets.QHBoxLayout()
        btn_fit = QtWidgets.QPushButton("Fit")
        btn_fit.setToolTip("Zoom to fit the whole map in the window.")
        btn_fit.clicked.connect(self._fit)
        ctrl.addWidget(btn_fit)
        btn_11 = QtWidgets.QPushButton("1:1")
        btn_11.setToolTip("One map pixel per screen pixel (full resolution).")
        btn_11.clicked.connect(lambda: self._view.set_scale(1.0))
        ctrl.addWidget(btn_11)
        self._lbl_zoom = QtWidgets.QLabel("")
        self._lbl_zoom.setMinimumWidth(70)
        ctrl.addWidget(self._lbl_zoom)
        if self._item_under is not None:
            ctrl.addSpacing(16)
            ctrl.addWidget(QtWidgets.QLabel("Phase opacity:"))
            self._sld_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self._sld_opacity.setRange(0, 100)
            self._sld_opacity.setValue(70)
            self._sld_opacity.setFixedWidth(140)
            self._sld_opacity.setToolTip(
                "Fade the phase colours over the total-EDS-signal underlay "
                "to judge boundary placement against the raw data.")
            self._sld_opacity.valueChanged.connect(
                lambda v: self._item_phase.setOpacity(v / 100.0))
            ctrl.addWidget(self._sld_opacity)
            self._item_phase.setOpacity(0.7)
        ctrl.addSpacing(16)
        self._lbl_hover = QtWidgets.QLabel("")
        self._lbl_hover.setMinimumWidth(240)
        ctrl.addWidget(self._lbl_hover)
        ctrl.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        ctrl.addWidget(btn_close)
        lay.addLayout(ctrl)

        # ---- legend row: colour chip + name + areal share per domain ------
        counts = np.bincount(label_map[label_map >= 0].ravel(),
                             minlength=n_clusters)
        denom = max(int(counts.sum()) + int((label_map < 0).sum()), 1)
        tab20 = matplotlib.colormaps['tab20']
        legend = QtWidgets.QHBoxLayout()
        legend.setSpacing(12)
        for cid in sorted(self._names):
            if cid < 0 or cid >= n_clusters or counts[cid] == 0:
                continue
            c = tab20(cid % 20)
            chip = QtWidgets.QLabel()
            chip.setFixedSize(12, 12)
            chip.setStyleSheet(
                f"background-color: rgb({int(c[0]*255)},{int(c[1]*255)},"
                f"{int(c[2]*255)}); border: 1px solid #666;")
            legend.addWidget(chip)
            legend.addWidget(QtWidgets.QLabel(
                f"{self._names[cid]}  ({100.0 * counts[cid] / denom:.1f}%)"))
        n_unc = int((label_map < 0).sum())
        if n_unc:
            chip = QtWidgets.QLabel()
            chip.setFixedSize(12, 12)
            chip.setStyleSheet("background-color: rgb(40,40,43); "
                               "border: 1px solid #666;")
            legend.addWidget(chip)
            legend.addWidget(QtWidgets.QLabel(
                f"Unclassified  ({100.0 * n_unc / denom:.1f}%)"))
        legend.addStretch(1)
        lay.addLayout(legend)

        lay.addWidget(self._view, stretch=1)
        self._on_zoom_changed()

    # -- behaviour -------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._fitted_once:
            self._fitted_once = True
            self.showMaximized()
            # Fit after the maximised geometry is in effect.
            QtCore.QTimer.singleShot(0, self._fit)

    def _fit(self):
        self._view.fitInView(self._scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        # Allow zooming a little further out than the fit, and keep the
        # clamp sane for maps smaller than the window.
        self._view.min_scale = min(self._view.current_scale() * 0.5, 1.0)
        self._on_zoom_changed()

    def _on_zoom_changed(self):
        s = self._view.current_scale()
        self._lbl_zoom.setText(f"{100.0 * s:.0f}%")
        # Crisp map pixels once zoomed to (or past) native resolution;
        # smooth minification below it.
        mode = (QtCore.Qt.FastTransformation if s >= 1.0
                else QtCore.Qt.SmoothTransformation)
        self._item_phase.setTransformationMode(mode)
        if self._item_under is not None:
            self._item_under.setTransformationMode(mode)

    def _on_hover(self, r: int, c: int):
        H, W = self._shape
        if 0 <= r < H and 0 <= c < W:
            cid = int(self._label_map[r, c])
            name = ("Unclassified" if cid < 0
                    else self._names.get(cid, f"Phase {cid + 1}"))
            self._lbl_hover.setText(f"y={r}  x={c}   {name}")
        else:
            self._lbl_hover.setText("")


class _PhaseClassDialog(QtWidgets.QDialog):
    """Post-clustering dialog: colored label map + editable cluster name table.

    When ``phase_aligned`` (the phase map shares the full-res band-map grid),
    an *Identify minerals…* button matches each domain's mean composition to the
    ``mineral_id`` library and fills a "Suggested mineral" column (ranked
    candidates + per-candidate confidence %). **Double-click a phase** for its full
    per-element composition (table + bar chart, wt% or net-count %), plus the
    candidate list + structural formula when ID was run; **Composition breakdown…**
    shows all phases side by side (phase×element table + stacked bar chart). A
    ``data_type``-driven banner flags net-counts as approximate vs quantified
    (structural recalc applied), and a Fe³⁺/ΣFe spinbox feeds the oxide split.
    Export adds ``phase_minerals.csv`` + ``phase_composition.csv``. ``scale``
    upsamples a 1/ds phase map (large-ROI guard) to mosaic resolution.
    """

    def __init__(
        self,
        label_map: np.ndarray,
        el_names: List[str],
        dominant: Dict[int, str],
        n_clusters: int,
        translate: Tuple[float, float],
        viewer,
        parent=None,
        bic_scores: Optional[Dict[int, float]] = None,
        icl_scores: Optional[Dict[int, float]] = None,
        auto_method: Optional[str] = None,
        band_maps: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        data_type: str = "bcf_net_counts",
        k_factors: Optional[Dict[str, float]] = None,
        phase_aligned: bool = False,
        scale: Tuple[float, float] = (1.0, 1.0),
        coating_elements: frozenset = frozenset({"C"}),
        seg_params: Optional[dict] = None,
        pixel_um: Optional[float] = None,
    ):
        super().__init__(parent)
        # Mosaic pixel pitch (µm); with ``scale`` it turns the interior-only
        # erosion (phase-map pixels) into a distance. None = unknown.
        self._pixel_um = float(pixel_um) if pixel_um else None
        # Replication recipe of the segmentation behind this phase map (None
        # for the EBSD path / pre-recipe sessions). Written out on export.
        self._seg_params = dict(seg_params) if seg_params else None
        if auto_method == "icl_min":
            auto_tag = "auto-detected, ICL"
        elif auto_method == "icl_knee":
            auto_tag = "auto-detected, ICL-knee"
        elif bic_scores is not None:
            auto_tag = "auto-detected"
        else:
            auto_tag = "manual"
        self.setWindowTitle(f"Phase classification  (k={n_clusters}, {auto_tag})")
        self._label_map = label_map
        self._n_clusters = n_clusters
        self._translate = translate
        self._scale = scale
        self._viewer = viewer
        self._bic_scores = bic_scores
        self._icl_scores = icl_scores
        self._auto_method = auto_method
        # Mineral-identification state (see _identify_minerals).
        self._band_maps = band_maps
        self._data_type = data_type
        # SPX-anchored k-factors: when set, per-domain net fractions are converted
        # to calibrated wt% (eds_quantify) so structural recalc is meaningful.
        self._k_factors = k_factors
        self._phase_aligned = bool(phase_aligned and band_maps)
        self._coating_elements = frozenset(coating_elements)
        self._mineral_matches: Dict[int, list] = {}
        self._domain_comps: Dict[int, tuple] = {}   # cid -> ({el: fraction}, n_px)
        # Manual mineral override: cid -> {name, abbrev, group, thermo_abbrev}.
        # Wins over the auto-ID in mineral_assignments() (auto-ID is a heuristic;
        # the user knows the petrology). Populated by the per-row Override combo.
        self._manual_override: Dict[int, dict] = {}
        self._override_combos: Dict[int, QtWidgets.QComboBox] = {}
        # Phases segmentation missed but the user knows are present (petrography /
        # EBSD / accessories). Each: {key, mode (% or None=presence-only), weight}.
        self._extra_phases: List[dict] = []
        # NNLS unmixing of mixed (sub-pixel mixture) domains:
        # {source_cid: {"fractions": {endmember_cid: f}, "residual": r}}.
        # Affects ONLY the modal proportions fed to the P-T estimate — the
        # phase map, per-domain composition and mineral ID stay untouched.
        self._unmix: Dict[int, dict] = {}
        import mineral_id as _mid_lib
        _seen = set()
        self._lib_minerals: List[dict] = []
        for _e in sorted(_mid_lib.DEFAULT_LIBRARY.values(), key=lambda e: e.name):
            if _e.name in _seen:
                continue
            _seen.add(_e.name)
            self._lib_minerals.append({
                "name": _e.name, "abbrev": _e.abbrev, "group": _e.group,
                "thermo_abbrev": _mid_lib.THERMO_ABBREV.get(
                    _e.abbrev, _e.abbrev.lower())})
        # Al2SiO5 polymorphs as explicit modelling choices (EDS only yields the
        # merged "Als"; the user resolves ky/sil/and by texture — see mineral_id).
        for _pm in getattr(_mid_lib, "MODELLING_POLYMORPHS", []):
            self._lib_minerals.append(dict(_pm))

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )

        main_layout = QtWidgets.QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_w = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self._fig = plt.figure(figsize=(5, 4))
        canvas = FigureCanvas(self._fig)
        toolbar = NavigationToolbar(canvas, self)
        left_lay.addWidget(toolbar)
        left_lay.addWidget(canvas)
        btn_inspect = QtWidgets.QPushButton("Inspect full screen …")
        btn_inspect.setToolTip(
            "Open the phase map maximised with free zoom (mouse wheel, up to "
            "full resolution and beyond) and panning, to judge segmentation "
            "quality. With aligned element maps, an opacity slider fades the "
            "phases over the total EDS signal.")
        btn_inspect.clicked.connect(self._inspect_map_fullscreen)
        left_lay.addWidget(btn_inspect)
        splitter.addWidget(left_w)

        ax = self._fig.add_subplot(111)
        rgba = _label_map_to_rgba(label_map, n_clusters)
        ax.imshow(rgba, interpolation='nearest', aspect='equal')
        ax.set_title("Phase map")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        canvas.draw()

        right_w = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_w)
        right_lay.addWidget(QtWidgets.QLabel("Assign phase names:"))

        # Honest reporting of the unclassified class (GMM-rejected ambiguous/
        # atypical pixels, or EBSD-unindexed pixels). These are not a phase:
        # excluded from modal proportions, composition and mineral ID.
        n_unc = int((label_map < 0).sum())
        if n_unc:
            lbl_unc = QtWidgets.QLabel(
                f"Unclassified: {n_unc:,} px "
                f"({100.0 * n_unc / label_map.size:.1f}% of the map) — "
                "rejected, no-data or unindexed pixels, excluded from "
                "modes, composition and mineral ID."
            )
            lbl_unc.setWordWrap(True)
            lbl_unc.setStyleSheet("color: gray;")
            right_lay.addWidget(lbl_unc)

        tab20 = matplotlib.colormaps['tab20']
        self._clusters = sorted(dominant.keys())
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Dominant elements", "Suggested mineral", "Name", "Pixels",
             "Override", "P-T", "Weight"])
        self._table.setRowCount(len(self._clusters))
        self._table.verticalHeader().setDefaultSectionSize(28)

        for row, cid in enumerate(self._clusters):
            color = tab20(cid % 20)
            r_int, g_int, b_int = int(color[0]*255), int(color[1]*255), int(color[2]*255)

            item_dom = QtWidgets.QTableWidgetItem(dominant.get(cid, ""))
            item_dom.setFlags(item_dom.flags() & ~QtCore.Qt.ItemIsEditable)
            item_dom.setBackground(QtGui.QColor(r_int, g_int, b_int, 60))
            self._table.setItem(row, 0, item_dom)

            item_min = QtWidgets.QTableWidgetItem("—")
            item_min.setFlags(item_min.flags() & ~QtCore.Qt.ItemIsEditable)
            self._table.setItem(row, 1, item_min)

            item_name = QtWidgets.QTableWidgetItem(f"Phase {cid + 1}")
            self._table.setItem(row, 2, item_name)

            n_px = int(np.sum(label_map == cid))
            item_px = QtWidgets.QTableWidgetItem(str(n_px))
            item_px.setFlags(item_px.flags() & ~QtCore.Qt.ItemIsEditable)
            self._table.setItem(row, 3, item_px)

            # Manual override: pick a mineral from the library (or "(auto)").
            combo = QtWidgets.QComboBox()
            combo.addItem("(auto)", None)
            for _m in self._lib_minerals:
                combo.addItem(_m["name"], _m)
            combo.setToolTip("Override the auto-ID for this phase with a mineral "
                             "you pick (sets the assemblage / P-T phase key).")
            combo.currentIndexChanged.connect(
                lambda _idx, c=cid, cb=combo: self._on_override_changed(c, cb))
            self._override_combos[cid] = combo
            self._table.setCellWidget(row, 4, combo)

            # Include-in-P-T checkbox (drop retrograde / relic / disequilibrium
            # domains from the observed assemblage used for P-T estimation).
            item_incl = QtWidgets.QTableWidgetItem()
            item_incl.setFlags((item_incl.flags() | QtCore.Qt.ItemIsUserCheckable)
                               & ~QtCore.Qt.ItemIsEditable)
            item_incl.setCheckState(QtCore.Qt.Checked)
            item_incl.setToolTip("Include this domain in the observed assemblage "
                                 "fed to the P-T estimate (uncheck to drop a "
                                 "retrograde / relic / disequilibrium domain).")
            self._table.setItem(row, 5, item_incl)

            # Per-domain reliability weight for the modal misfit (auto-filled from
            # ID confidence after 'Identify minerals'; editable). 1 = full weight.
            item_w = QtWidgets.QTableWidgetItem("1.00")
            item_w.setToolTip("Reliability weight for this phase in the modal "
                              "misfit (0–1). Lower = a shaky ID counts less. "
                              "Auto-set from ID confidence; edit to override.")
            self._table.setItem(row, 6, item_w)

        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        right_lay.addWidget(self._table)

        # Mineral identification controls + data-quality caveat banner.
        self._caveat_lbl = QtWidgets.QLabel()
        self._caveat_lbl.setWordWrap(True)
        self._caveat_lbl.setStyleSheet("color: #555; font-size: 11px;")
        right_lay.addWidget(self._caveat_lbl)

        id_row = QtWidgets.QHBoxLayout()
        self._btn_identify = QtWidgets.QPushButton("Identify minerals …")
        self._btn_identify.setToolTip(
            "Match each domain's mean composition against a curated mineral library. "
            "Double-click a row for its full element breakdown (and, after ID, the "
            "candidate list + structural formula).")
        id_row.addWidget(self._btn_identify)
        id_row.addWidget(QtWidgets.QLabel("Fe³⁺/ΣFe:"))
        self._fe3_spin = QtWidgets.QDoubleSpinBox()
        self._fe3_spin.setRange(0.0, 1.0)
        self._fe3_spin.setSingleStep(0.05)
        self._fe3_spin.setDecimals(2)
        self._fe3_spin.setValue(0.0)
        self._fe3_spin.setToolTip("Assumed fraction of total Fe as Fe³⁺ (EDS sees total Fe only).")
        id_row.addWidget(self._fe3_spin)
        id_row.addWidget(QtWidgets.QLabel("Coating to ignore:"))
        self._coating_edit = QtWidgets.QLineEdit(
            ", ".join(sorted(self._coating_elements)))
        self._coating_edit.setMaximumWidth(110)
        self._coating_edit.setPlaceholderText("e.g. C, Pt")
        self._coating_edit.setToolTip(
            "Conductive coating / contaminant element(s) excluded before matching "
            "(comma-separated; blank = uncoated). Default C (carbon coat); set to "
            "Pt/Au/… for a different coat. A coating element that is also a "
            "rock-forming cation will be stripped from that mineral's match. "
            "Oxygen is always excluded as the anion.")
        id_row.addWidget(self._coating_edit)
        self._chk_interior = QtWidgets.QCheckBox("Interior only:")
        self._chk_interior.setToolTip(
            "Compute each domain's mean composition from interior pixels only\n"
            "(domain mask eroded by the given number of pixels). Grain-boundary\n"
            "pixels carry mixed signal from the neighbouring phase, which drags\n"
            "the mean toward compositionally intermediate minerals (e.g. a\n"
            "mica-blend domain matching cordierite). Erosion trims that rim.\n"
            "A domain that vanishes entirely at this erosion falls back to its\n"
            "full mask (logged). Modal proportions always use the full mask.\n"
            "NB pixels are phase-map pixels: at 1/ds resolution one pixel is ds\n"
            "mosaic pixels wide. Re-run Identify minerals after changing this.")
        id_row.addWidget(self._chk_interior)
        self._erode_spin = QtWidgets.QSpinBox()
        self._erode_spin.setRange(1, 20)
        self._erode_spin.setValue(3)
        self._erode_spin.setSuffix(" px")
        self._erode_spin.setEnabled(False)
        self._erode_spin.setToolTip(self._chk_interior.toolTip())
        id_row.addWidget(self._erode_spin)
        # The same pixel count means a different distance on every dataset
        # (pixel pitch x decimation), so show it in µm when the pitch is known.
        self._erode_um_label = QtWidgets.QLabel("")
        self._erode_um_label.setToolTip(
            "Erosion distance: phase-map pixels x decimation x mosaic pixel size.")
        id_row.addWidget(self._erode_um_label)
        self._update_erode_um_label()
        self._chk_interior.toggled.connect(self._on_erosion_changed)
        self._erode_spin.valueChanged.connect(self._on_erosion_changed)
        id_row.addStretch()
        right_lay.addLayout(id_row)
        self._btn_identify.clicked.connect(self._identify_minerals)

        # ── P-T inversion: phases segmentation missed + open-assemblage ───────
        # Let the user add phases not picked up by segmentation (presence-only,
        # or with a rough mode%) and forgive minor unmodelled predicted phases.
        miss_box = QtWidgets.QGroupBox("Phases segmentation missed (for P-T estimate)")
        miss_lay = QtWidgets.QVBoxLayout(miss_box)
        add_row = QtWidgets.QHBoxLayout()
        self._extra_combo = QtWidgets.QComboBox()
        self._extra_combo.setEditable(True)
        self._extra_combo.addItem("", None)
        for _m in self._lib_minerals:
            self._extra_combo.addItem(_m["name"], _m.get("thermo_abbrev") or _m["abbrev"])
        self._extra_combo.setToolTip(
            "A phase you know is present (petrography / EBSD / accessory) but "
            "segmentation didn't resolve. Pick from the library or type a name.")
        self._extra_mode = QtWidgets.QLineEdit()
        self._extra_mode.setMaximumWidth(90)
        self._extra_mode.setPlaceholderText("mode % (opt.)")
        self._extra_mode.setToolTip(
            "Optional rough modal abundance (%). Leave blank for a presence-only "
            "constraint (must be present, but no proportion target).")
        btn_add_extra = QtWidgets.QPushButton("Add")
        btn_add_extra.clicked.connect(self._add_extra_phase)
        add_row.addWidget(QtWidgets.QLabel("Phase:"))
        add_row.addWidget(self._extra_combo, 1)
        add_row.addWidget(self._extra_mode)
        add_row.addWidget(btn_add_extra)
        miss_lay.addLayout(add_row)
        self._extra_list = QtWidgets.QListWidget()
        self._extra_list.setMaximumHeight(64)
        self._extra_list.setToolTip("Added phases. Select one and press Delete / "
                                    "the Remove button to drop it.")
        miss_lay.addWidget(self._extra_list)
        rm_row = QtWidgets.QHBoxLayout()
        btn_rm_extra = QtWidgets.QPushButton("Remove selected")
        btn_rm_extra.clicked.connect(self._remove_extra_phase)
        rm_row.addStretch()
        rm_row.addWidget(btn_rm_extra)
        miss_lay.addLayout(rm_row)

        open_row = QtWidgets.QHBoxLayout()
        self._open_check = QtWidgets.QCheckBox("Assemblage may be incomplete — "
                                               "forgive minor unmodelled phases up to")
        self._open_check.setToolTip(
            "Treat the observed assemblage as possibly incomplete: a predicted "
            "phase you didn't segment is not penalised as 'extra' as long as its "
            "mode is below this threshold (an abundant unobserved phase still is).")
        self._open_spin = QtWidgets.QDoubleSpinBox()
        self._open_spin.setRange(0.0, 50.0)
        self._open_spin.setValue(5.0)
        self._open_spin.setSuffix(" %")
        self._open_spin.setMaximumWidth(70)
        self._open_spin.setEnabled(False)
        self._open_check.toggled.connect(self._open_spin.setEnabled)
        open_row.addWidget(self._open_check)
        open_row.addWidget(self._open_spin)
        open_row.addStretch()
        miss_lay.addLayout(open_row)
        right_lay.addWidget(miss_box)

        import mineral_id as _mid
        quantified = self._data_type in _mid.QUANTIFIED_TYPES
        calibrated = self._data_type == "eds_calibrated_wt_pct"
        ds = int(round(self._scale[0])) if self._scale else 1
        ds_note = (f" Domains were segmented at 1/{ds} resolution (fine grains may be "
                   "merged); composition means are still representative.") if ds > 1 else ""
        calib_note = (" Composition is SPX-calibrated EDS (captures the average matrix "
                      "correction, not per-pixel matrix variation).") if calibrated else ""
        if not self._phase_aligned:
            self._caveat_lbl.setText(
                "Mineral ID needs full-resolution ROI maps aligned to the phase map "
                "(segment from the computed full-res ROI). Unavailable for this run.")
            self._btn_identify.setEnabled(False)
            self._fe3_spin.setEnabled(False)
            self._coating_edit.setEnabled(False)
        elif quantified:
            self._caveat_lbl.setText(
                "Quantified wt%: structural-formula recalculation (atoms per formula "
                "unit + site closure) is applied." + calib_note + ds_note)
        else:
            self._caveat_lbl.setText(
                "Net-count intensities are uncalibrated — mineral suggestions are "
                "approximate (cation-ratio match only)." + ds_note)

        splitter.addWidget(right_w)
        splitter.setSizes([460, 380])

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_layer = QtWidgets.QPushButton("Add as napari layer")
        btn_bic = QtWidgets.QPushButton("Show BIC curve")
        btn_bic.setEnabled(bic_scores is not None or icl_scores is not None)
        btn_breakdown = QtWidgets.QPushButton("Composition breakdown …")
        btn_breakdown.setEnabled(bool(self._band_maps))
        btn_breakdown.setToolTip(
            "Full per-element composition for every phase (table + stacked bar chart)."
            if self._band_maps else
            "Needs full-resolution ROI maps — segment from the computed full-res ROI.")
        btn_export = QtWidgets.QPushButton("Export phase map …")
        btn_thermod = QtWidgets.QPushButton("Use for P-T estimate …")
        btn_close = QtWidgets.QPushButton("Close")
        btn_merge = QtWidgets.QPushButton("Merge selected …")
        btn_merge.setToolTip(
            "Merge the selected table rows into one domain (the lowest cluster "
            "id). Rewrites the phase map, so the merged domain gets one row, one "
            "colour, one composition and one legend entry. Use for duplicate "
            "domains of the same phase (e.g. clean vs boundary-mixed quartz). "
            "Cannot be undone except by re-running the segmentation.")
        btn_unmix = QtWidgets.QPushButton("Unmix selected …")
        btn_unmix.setEnabled(bool(self._band_maps))
        btn_unmix.setToolTip(
            "Decompose ONE selected domain that is a sub-pixel MIXTURE of the "
            "other phases (e.g. a fine-grained quartz-mica matrix below the "
            "map resolution) into areal fractions of the other domains, by "
            "non-negative least squares on mean net-count vectors (counts add "
            "linearly over sub-pixel phase fractions). Only the modal "
            "proportions fed to the P-T estimate are redistributed; the phase "
            "map, composition and mineral ID are untouched. Prefer this over "
            "merging a mixed domain into its dominant phase — merging counts "
            "every mixed pixel as that phase and biases the modes."
            if self._band_maps else
            "Needs full-resolution ROI maps — segment from the computed "
            "full-res ROI.")
        btn_row.addWidget(btn_layer)
        btn_row.addWidget(btn_merge)
        btn_row.addWidget(btn_unmix)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_breakdown)
        btn_row.addWidget(btn_bic)
        btn_row.addWidget(btn_thermod)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

        btn_layer.clicked.connect(self._add_napari_layer)
        btn_merge.clicked.connect(self._merge_selected_domains)
        btn_unmix.clicked.connect(self._unmix_selected_domain)
        btn_export.clicked.connect(self._export_phase_map)
        btn_breakdown.clicked.connect(self._show_composition_overview)
        btn_bic.clicked.connect(self._show_bic_curve)
        btn_thermod.clicked.connect(self.accept)
        btn_close.clicked.connect(self.reject)

        self.setMinimumSize(700, 450)
        self.resize(860, 520)

    @property
    def phase_names(self) -> List[str]:
        """Return unique non-empty phase names from the table."""
        names = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 2)
            if item:
                name = item.text().strip()
                if name:
                    names.append(name)
        return list(dict.fromkeys(names))

    def _inspect_map_fullscreen(self):
        """Open the maximised zoom/pan inspector for the current phase map.

        Names come from the table at click time (so edits show); when the
        band maps share the phase-map grid, the summed per-element main
        bands form a total-signal underlay for boundary-quality checks.
        """
        names = {}
        for row, cid in enumerate(self._clusters):
            item = self._table.item(row, 2)
            txt = item.text().strip() if item else ""
            names[cid] = txt or f"Phase {cid + 1}"
        underlay = None
        if self._phase_aligned:
            try:
                tot = None
                for bands in self._band_maps.values():
                    m = np.asarray(bands.get("main"), dtype=np.float32)
                    if m.shape != self._label_map.shape:
                        tot = None
                        break
                    tot = m.copy() if tot is None else tot + m
                underlay = tot
            except Exception as e:
                print(f"[phases] underlay build failed ({e}) — "
                      "inspector opens without it", flush=True)
                underlay = None
        dlg = _MapInspectDialog(self._label_map, self._n_clusters, names,
                                underlay=underlay, parent=self)
        dlg.exec_()

    # ---- Per-domain composition (shared by ID, breakdown, overview, export) -

    def _domain_composition(self, cid: int):
        """Cached ``({element: fraction}, n_px)`` for one cluster, or ``({}, 0)``.

        Fractions sum to 1 across all measured elements (the full EDS signal that
        feeds mineral ID), via ``_net_count_fractions`` over the cluster mask.
        When SPX k-factors are present the net fractions are first calibrated to
        wt% (eds_quantify) and re-expressed as fractions summing to 1, so every
        consumer (×100 → wt%, mineral ID, the all-phases breakdown) sees
        quasi-quantified composition. Requires full-res band maps.
        """
        if not self._band_maps:
            return {}, 0
        if cid not in self._domain_comps:
            from viewer_thermo_helpers import _net_count_fractions
            mask = self._label_map == cid
            erode = self.erosion_value()
            if erode > 0 and mask.any():
                from scipy.ndimage import binary_erosion
                interior = binary_erosion(mask, iterations=erode)
                if interior.any():
                    mask = interior
                else:
                    print(f"[phases] domain {cid}: interior-only erosion "
                          f"({erode} px) left no pixels — using the full mask",
                          flush=True)
            frac, n_px = _net_count_fractions(self._band_maps, sub_mask=mask)
            if frac and self._k_factors:
                import eds_quantify as _eq
                wt = _eq.apply_calibration_fractions(frac, self._k_factors)
                tot = sum(wt.values())
                if tot > 0:
                    frac = {el: v / tot for el, v in wt.items()}
            self._domain_comps[cid] = (frac, n_px)
        return self._domain_comps[cid]

    def _composition_units(self) -> str:
        import mineral_id
        return "wt%" if self._data_type in mineral_id.QUANTIFIED_TYPES else "net-count %"

    # ---- Mineral identification --------------------------------------------

    def _identify_minerals(self):
        """Match each domain's mean composition to the curated mineral library."""
        if not self._phase_aligned or not self._band_maps:
            return
        import mineral_id

        quantified = self._data_type in mineral_id.QUANTIFIED_TYPES
        fe3 = float(self._fe3_spin.value())
        coating = self.coating_value()
        self._mineral_matches = {}
        for row, cid in enumerate(self._clusters):
            frac, n_px = self._domain_composition(cid)
            comp_wt = {k: 100.0 * float(v) for k, v in frac.items()}
            matches = mineral_id.identify(
                comp_wt, quantified=quantified, fe3_fraction=fe3, top_n=5,
                coating_elements=coating)
            self._mineral_matches[cid] = matches
            cell = self._table.item(row, 1)
            cell.setText(self._fmt_suggestion(matches))
            cell.setToolTip(self._tooltip_for(matches, n_px))
            # Default the editable Name to the top mineral, but never clobber a
            # name the user has already typed (only replace the "Phase N" default).
            if matches:
                name_item = self._table.item(row, 2)
                if name_item is not None and name_item.text().strip() in ("", f"Phase {cid + 1}"):
                    name_item.setText(matches[0].name)
            QtWidgets.QApplication.processEvents()
        # Re-running auto-ID must not hide an existing manual override.
        for cid in self._manual_override:
            self._refresh_assignment_display(cid)
        self._prefill_pt_weights()
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)

    def _prefill_pt_weights(self):
        """Set each domain's P-T Weight cell from its top-match ID confidence
        (manual override → 1.0). Leaves un-identified rows at their default."""
        for row, cid in enumerate(self._clusters):
            if self._manual_override.get(cid):
                w = 1.0
            else:
                matches = self._mineral_matches.get(cid) or []
                if not matches:
                    continue
                m = matches[0]
                pct = getattr(m, "confidence_pct", None)
                if pct is not None:
                    w = max(0.1, min(1.0, float(pct) / 100.0))
                else:
                    w = {"high": 1.0, "moderate": 0.7, "low": 0.4,
                         "approximate": 0.5}.get(getattr(m, "confidence", None), 0.6)
            it = self._table.item(row, 6)
            if it is not None:
                it.setText(f"{w:.2f}")

    def _add_extra_phase(self):
        key = self._extra_combo.currentData() or self._extra_combo.currentText()
        key = str(key).strip()
        if not key:
            return
        mode = None
        txt = self._extra_mode.text().strip().rstrip("%").strip()
        if txt:
            try:
                mode = float(txt)
            except ValueError:
                mode = None
        self._extra_phases.append({"key": key, "mode": mode, "weight": 1.0})
        self._extra_mode.clear()
        self._extra_combo.setCurrentIndex(0)
        self._refresh_extra_list()

    def _remove_extra_phase(self):
        rows = sorted((self._extra_list.row(i)
                       for i in self._extra_list.selectedItems()), reverse=True)
        for r in rows:
            if 0 <= r < len(self._extra_phases):
                del self._extra_phases[r]
        self._refresh_extra_list()

    def _refresh_extra_list(self):
        self._extra_list.clear()
        for e in self._extra_phases:
            md = e.get("mode")
            tag = (f"{e['key']}  —  {md:.0f}%" if md
                   else f"{e['key']}  —  present (no mode)")
            self._extra_list.addItem(tag)

    def included_domains(self):
        """Set of cids checked for the P-T estimate (None when all included)."""
        incl, all_in = set(), True
        for row, cid in enumerate(self._clusters):
            it = self._table.item(row, 5)
            if it is not None and it.checkState() == QtCore.Qt.Checked:
                incl.add(int(cid))
            else:
                all_in = False
        return None if all_in else incl

    def domain_weights(self):
        """Per-domain reliability weight from the Weight column (0–1)."""
        out: Dict[int, float] = {}
        for row, cid in enumerate(self._clusters):
            it = self._table.item(row, 6)
            try:
                w = float(it.text()) if it and it.text().strip() else 1.0
            except ValueError:
                w = 1.0
            out[int(cid)] = max(0.0, min(1.0, w))
        return out

    def extra_observed(self):
        """User-added phases segmentation missed: [{key, mode, weight}]."""
        return [dict(e) for e in self._extra_phases]

    def open_assemblage_tol(self):
        """Open-assemblage forgiveness as a mode fraction (0 when unchecked)."""
        return (float(self._open_spin.value()) / 100.0
                if self._open_check.isChecked() else 0.0)

    def set_pt_overrides(self, include=None, weights=None, extra=None, open_tol=0.0):
        """Restore the P-T inversion controls (used when the dialog is re-opened)."""
        if weights:
            for row, cid in enumerate(self._clusters):
                if int(cid) in weights or str(cid) in weights:
                    w = weights.get(int(cid), weights.get(str(cid)))
                    it = self._table.item(row, 6)
                    if it is not None:
                        it.setText(f"{float(w):.2f}")
        if include is not None:
            inc = {int(c) for c in include}
            for row, cid in enumerate(self._clusters):
                it = self._table.item(row, 5)
                if it is not None:
                    it.setCheckState(QtCore.Qt.Checked if int(cid) in inc
                                     else QtCore.Qt.Unchecked)
        if extra:
            self._extra_phases = [dict(e) for e in extra]
            self._refresh_extra_list()
        if open_tol and float(open_tol) > 0:
            self._open_check.setChecked(True)
            self._open_spin.setValue(float(open_tol) * 100.0)

    def _on_override_changed(self, cid: int, combo):
        """Record/clear a manual mineral override for a domain and refresh display."""
        data = combo.currentData()
        prev = self._manual_override.get(cid)
        if data is None:
            self._manual_override.pop(cid, None)
        else:
            self._manual_override[cid] = dict(data)
        self._refresh_assignment_display(cid, prev_override=prev)

    def _refresh_assignment_display(self, cid: int, prev_override: Optional[dict] = None):
        """Update the Suggested-mineral cell and the Name to reflect a manual
        override, or revert to the auto suggestion when cleared.

        The Name follows the override whenever its current text is one the
        dialog itself wrote — the "Phase N" placeholder, the auto-ID top match,
        or a previous override's name — so picking a mineral relabels the phase.
        A hand-typed custom label is never clobbered. Clearing back to "(auto)"
        reverts an override-set name to the auto top match (or the placeholder).
        """
        try:
            row = self._clusters.index(cid)
        except ValueError:
            return
        cell = self._table.item(row, 1)
        name_item = self._table.item(row, 2)
        ov = self._manual_override.get(cid)
        matches = self._mineral_matches.get(cid)
        auto_top = matches[0].name if matches else None
        dialog_written = {"", f"Phase {cid + 1}"}
        if auto_top:
            dialog_written.add(auto_top)
        if prev_override:
            dialog_written.add(prev_override.get("name", ""))
        if ov:
            if cell is not None:
                cell.setText(f"✎ {ov['name']} (manual)")
                cell.setToolTip("Manual override — wins over auto-ID.")
            if name_item is not None and \
                    name_item.text().strip() in dialog_written:
                name_item.setText(ov["name"])
        else:
            if cell is not None:
                cell.setText(self._fmt_suggestion(matches) if matches else "—")
            if name_item is not None and prev_override and \
                    name_item.text().strip() == prev_override.get("name"):
                name_item.setText(auto_top or f"Phase {cid + 1}")

    def set_override(self, cid: int, mineral_name: str):
        """Programmatically select an override by mineral name (e.g. on reopen)."""
        combo = self._override_combos.get(cid)
        if combo is None:
            return
        idx = combo.findText(str(mineral_name))
        if idx >= 0:
            combo.setCurrentIndex(idx)   # triggers _on_override_changed

    @staticmethod
    def _fmt_suggestion(matches) -> str:
        if not matches:
            return "—"
        return " · ".join(f"{m.name} {m.confidence_pct:.0f}%" for m in matches[:2])

    @staticmethod
    def _tooltip_for(matches, n_px: int) -> str:
        if not matches:
            return f"No match ({n_px} px)"
        lines = [f"{n_px} px — double-click for details", ""]
        for m in matches:
            lines.append(f"{m.name}: {m.confidence_pct:.0f}% ({m.confidence})")
        flags = matches[0].flags
        if flags:
            lines += ["", "Caveats: " + ", ".join(flags)]
        return "\n".join(lines)

    def _on_cell_double_clicked(self, row: int, col: int):
        if row < 0 or row >= len(self._clusters):
            return
        if self._band_maps:
            self._show_domain_detail(self._clusters[row])

    def _row_name(self, cid: int) -> str:
        """Current editable Name for a cluster (falls back to the placeholder)."""
        try:
            row = self._clusters.index(cid)
        except ValueError:
            return f"Phase {cid + 1}"
        item = self._table.item(row, 2)
        txt = item.text().strip() if item else ""
        return txt or f"Phase {cid + 1}"

    def _show_domain_detail(self, cid: int):
        """Per-domain popup: full element composition (table + bar chart), plus the
        mineral candidates + structural formula when ``Identify minerals`` was run."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        frac, n_px = self._domain_composition(cid)
        units = self._composition_units()
        items = sorted(((el, 100.0 * v) for el, v in frac.items() if v > 0),
                       key=lambda kv: -kv[1])

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{self._row_name(cid)} — composition  ({n_px:,} px)")
        dlg.resize(520, 560)
        lay = QtWidgets.QVBoxLayout(dlg)

        lay.addWidget(QtWidgets.QLabel(f"<b>EDS composition</b>  ({units})"))
        if items:
            fig = plt.figure(figsize=(5, max(1.6, 0.28 * len(items))))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            els = [e for e, _ in items][::-1]
            vals = [v for _, v in items][::-1]
            ax.barh(range(len(els)), vals, color="#4a78b5")
            ax.set_yticks(range(len(els)))
            ax.set_yticklabels(els, fontsize=8)
            ax.set_xlabel(units)
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v:.1f}", va="center", fontsize=7)
            fig.tight_layout()
            canvas.draw()
            lay.addWidget(canvas)
            dlg.finished.connect(lambda *_: plt.close(fig))

            ctbl = QtWidgets.QTableWidget()
            ctbl.setColumnCount(2)
            ctbl.setHorizontalHeaderLabels(["Element", units])
            ctbl.setRowCount(len(items))
            ctbl.verticalHeader().setVisible(False)
            ctbl.setMaximumHeight(150)
            for r, (el, v) in enumerate(items):
                ctbl.setItem(r, 0, QtWidgets.QTableWidgetItem(el))
                ctbl.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{v:.2f}"))
            ctbl.resizeColumnsToContents()
            ctbl.horizontalHeader().setStretchLastSection(True)
            lay.addWidget(ctbl)
        else:
            lay.addWidget(QtWidgets.QLabel("No EDS signal in this domain."))

        matches = self._mineral_matches.get(cid)
        if matches:
            self._add_mineral_section(lay, matches)

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec_()

    @staticmethod
    def _add_mineral_section(lay, matches):
        """Append the mineral-candidate table + top-hit structural formula to a layout."""
        lay.addWidget(_hline())
        lay.addWidget(QtWidgets.QLabel("<b>Mineral candidates</b>"))
        tbl = QtWidgets.QTableWidget()
        tbl.setColumnCount(4)
        tbl.setHorizontalHeaderLabels(["Mineral", "Confidence", "Band", "Method"])
        tbl.setRowCount(len(matches))
        tbl.verticalHeader().setVisible(False)
        tbl.setMaximumHeight(160)
        for r, m in enumerate(matches):
            tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{m.name} ({m.formula})"))
            tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{m.confidence_pct:.0f}%"))
            tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(m.confidence))
            tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(m.method))
        tbl.resizeColumnsToContents()
        tbl.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(tbl)

        top = matches[0]
        if top.apfu:
            lay.addWidget(QtWidgets.QLabel(
                f"<b>Structural formula — {top.name}</b> (per {top.formula} O-basis)"))
            apfu_txt = "  ".join(f"{el}={v:.2f}" for el, v in sorted(
                top.apfu.items(), key=lambda kv: -kv[1]) if v > 0.005)
            lab = QtWidgets.QLabel("apfu:  " + apfu_txt)
            lab.setWordWrap(True)
            lay.addWidget(lab)
            if top.site_closure:
                clo = "  ".join(f"{s}={c:.2f}" for s, c in top.site_closure.items())
                lay.addWidget(QtWidgets.QLabel("site closure (ideal=1.00):  " + clo))
        if top.indices:
            ix = "  ".join(f"{k}={v:.2f}" for k, v in top.indices.items())
            lay.addWidget(QtWidgets.QLabel("indices:  " + ix))
        if top.flags:
            cav = QtWidgets.QLabel("Caveats: " + ", ".join(top.flags))
            cav.setWordWrap(True)
            cav.setStyleSheet("color: #a0522d;")
            lay.addWidget(cav)

    def _ordered_elements(self):
        """All elements present across domains, sorted by total abundance desc."""
        totals: Dict[str, float] = {}
        for cid in self._clusters:
            frac, _ = self._domain_composition(cid)
            for el, v in frac.items():
                totals[el] = totals.get(el, 0.0) + float(v)
        return [e for e, _ in sorted(totals.items(), key=lambda kv: -kv[1])]

    def domain_compositions(self) -> Dict[int, Dict[str, float]]:
        """``{cid: {element: fraction}}`` for every cluster (empty if no band maps).

        Computed via the cached ``_domain_composition`` so the panel can keep it
        after this dialog closes (and persist it in the session)."""
        if not self._band_maps:
            return {}
        return {cid: dict(self._domain_composition(cid)[0]) for cid in self._clusters}

    def _show_composition_overview(self):
        """All-phases comparison: delegates to the reusable composition dialog."""
        if not self._band_maps:
            return
        comps, npx = {}, {}
        for cid in self._clusters:
            frac, n = self._domain_composition(cid)
            comps[cid], npx[cid] = frac, n
        names = {cid: self._row_name(cid) for cid in self._clusters}
        _PhaseCompositionDialog(
            comps, npx, names, self._composition_units(), parent=self).exec_()

    def mineral_assignments(self) -> Dict[int, dict]:
        """Serialisable per-domain mineral results for export / session.

        A manual override (if set for a domain) replaces the auto-ID: it is
        recorded with ``method="manual"`` and high confidence so it drives the
        assemblage / P-T phase keys, while the auto candidates are kept for
        reference. Domains with only an override (auto-ID never run) are included.
        """
        out: Dict[int, dict] = {}
        cids = set(self._mineral_matches) | set(self._manual_override)
        for cid in cids:
            matches = self._mineral_matches.get(cid) or []
            candidates = [
                {"name": m.name, "abbrev": m.abbrev,
                 "confidence_pct": m.confidence_pct, "confidence": m.confidence}
                for m in matches[:5]]
            ov = self._manual_override.get(cid)
            if ov:
                out[cid] = {
                    "abbrev": ov["abbrev"], "name": ov["name"],
                    "group": ov["group"], "score": 1.0, "method": "manual",
                    "confidence": "high", "confidence_pct": 100.0,
                    "indices": {}, "flags": ["manual_override"],
                    "thermo_abbrev": ov["thermo_abbrev"], "candidates": candidates,
                }
            elif matches:
                top = matches[0]
                out[cid] = {
                    "abbrev": top.abbrev, "name": top.name, "group": top.group,
                    "score": round(float(top.score), 4), "method": top.method,
                    "confidence": top.confidence,
                    "confidence_pct": top.confidence_pct,
                    "indices": {k: round(float(v), 4) for k, v in top.indices.items()},
                    "flags": list(top.flags), "thermo_abbrev": top.thermo_abbrev,
                    "candidates": candidates,
                }
        return out

    def fe3_value(self) -> float:
        return float(self._fe3_spin.value())

    def erosion_value(self) -> int:
        """Interior-only erosion in phase-map pixels (0 = off)."""
        if not getattr(self, "_chk_interior", None):
            return 0
        return int(self._erode_spin.value()) if self._chk_interior.isChecked() else 0

    def set_erosion(self, px: int) -> None:
        """Restore the interior-only setting (session/reopen path)."""
        px = max(0, int(px))
        self._chk_interior.setChecked(px > 0)
        if px > 0:
            self._erode_spin.setValue(px)

    def _on_erosion_changed(self, *_args) -> None:
        """Interior-only toggled/changed: drop cached compositions so every
        consumer recomputes lazily. Like Fe³⁺, an existing Suggested-mineral
        column goes stale until Identify is re-run (spinbox enabled state
        doubles as the visual cue)."""
        self._erode_spin.setEnabled(self._chk_interior.isChecked())
        self._update_erode_um_label()
        self._domain_comps.clear()

    def erosion_um(self) -> Optional[float]:
        """Interior-only erosion as a distance in µm (None if the pixel size
        is unknown; 0 when off)."""
        if not self._pixel_um:
            return None
        return self.erosion_value() * float(self._scale[0]) * self._pixel_um

    def _update_erode_um_label(self) -> None:
        lbl = getattr(self, "_erode_um_label", None)
        if lbl is None:
            return
        if not self._pixel_um:
            lbl.setText("")
            return
        um = int(self._erode_spin.value()) * float(self._scale[0]) * self._pixel_um
        lbl.setText(f"≈ {um:.3g} µm")
        lbl.setEnabled(self._chk_interior.isChecked())

    def _merge_selected_domains(self):
        """Merge the selected table rows into one domain.

        Rewrites ``self._label_map`` IN PLACE (the panel holds the same array,
        so the merge propagates to the stored phase map, session save, modes,
        and the misfit inputs without further plumbing). The lowest selected
        cluster id survives and keeps its name, colour, override, and P-T
        settings; the other rows are removed. Mineral ID is re-run
        automatically when it had been run before.
        """
        rows = sorted({ix.row() for ix in self._table.selectedIndexes()})
        if len(rows) < 2:
            QtWidgets.QMessageBox.information(
                self, "Merge domains",
                "Select two or more rows to merge (click + Ctrl/Shift-click).")
            return
        cids = [self._clusters[r] for r in rows]
        target, merged_away = cids[0], cids[1:]
        name_item = self._table.item(rows[0], 2)
        target_name = (name_item.text().strip()
                       if name_item and name_item.text().strip()
                       else f"Phase {target + 1}")
        ans = QtWidgets.QMessageBox.question(
            self, "Merge domains",
            f"Merge {len(cids)} domains into '{target_name}'?\n\n"
            "The phase map is rewritten; this cannot be undone except by "
            "re-running the segmentation.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if ans != QtWidgets.QMessageBox.Yes:
            return

        self._label_map[np.isin(self._label_map, merged_away)] = target

        # Drop dialog state for the removed domains; the target's composition
        # cache is stale (its mask grew), so drop that too.
        had_matches = bool(self._mineral_matches)
        for cid in merged_away:
            self._clusters.remove(cid)
            self._domain_comps.pop(cid, None)
            self._mineral_matches.pop(cid, None)
            self._manual_override.pop(cid, None)
            self._override_combos.pop(cid, None)
        self._domain_comps.pop(target, None)
        self._mineral_matches.pop(target, None)
        for row in reversed(rows[1:]):
            self._table.removeRow(row)

        # Refresh the target row's pixel count and the phase-map preview.
        n_px = int(np.sum(self._label_map == target))
        px_item = self._table.item(rows[0], 3)
        if px_item is not None:
            px_item.setText(str(n_px))
        ax = self._fig.axes[0]
        ax.clear()
        ax.imshow(_label_map_to_rgba(self._label_map, self._n_clusters),
                  interpolation='nearest', aspect='equal')
        ax.set_title("Phase map")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        self._fig.canvas.draw()

        print(f"[phases] merged domains {merged_away} into {target} "
              f"('{target_name}', {n_px:,} px). A phase-map napari layer added "
              "earlier is now stale — re-add it from this dialog.", flush=True)

        # Unmix entries referencing a merged-away domain (as source or as an
        # endmember) are stale — fractions were fitted against domains that no
        # longer exist. The target's entry is stale too (its mask grew).
        removed = set(merged_away) | {target}
        stale = [c for c, d in self._unmix.items()
                 if c in removed
                 or removed & set((d.get("fractions") or {}).keys())]
        for c in stale:
            self._unmix.pop(c, None)
            self._mark_unmixed_row(c, False)
            print(f"[phases] unmix of domain {c} cleared (merge invalidated "
                  "its endmember set — re-run Unmix if needed)", flush=True)
        if had_matches:
            self._identify_minerals()

    # ---- Sub-pixel mixture unmixing (modal-proportion correction) ----------

    def _domain_mean_counts(self, cid: int, interior: bool = True):
        """Mean *raw* net counts per pixel for one domain: ``({el: mean}, n_px)``.

        The additive basis the NNLS unmix needs (never normalised fractions).
        ``interior`` applies the dialog's interior-only erosion (cleaner
        endmembers); the mixed *source* is always measured on its full mask —
        its whole area is what gets decomposed.
        """
        from viewer_thermo_helpers import _net_count_sums
        mask = self._label_map == cid
        erode = self.erosion_value() if interior else 0
        if erode > 0 and mask.any():
            from scipy.ndimage import binary_erosion
            inner = binary_erosion(mask, iterations=erode)
            if inner.any():
                mask = inner
        sums, n_px = _net_count_sums(self._band_maps, sub_mask=mask)
        if not n_px or not sums:
            return {}, 0
        return {el: v / n_px for el, v in sums.items()}, n_px

    def _mark_unmixed_row(self, cid: int, on: bool) -> None:
        """Visual cue on the Pixels cell that a domain's mode is redistributed."""
        try:
            row = self._clusters.index(cid)
        except ValueError:
            return
        item = self._table.item(row, 3)
        if item is None:
            return
        if on:
            d = self._unmix.get(cid) or {}
            fr = d.get("fractions") or {}
            parts = ", ".join(
                f"{self._row_name(e)} {100.0 * f / max(sum(fr.values()), 1e-9):.0f}%"
                for e, f in sorted(fr.items(), key=lambda kv: -kv[1]) if f > 0)
            item.setBackground(QtGui.QColor(255, 224, 178))
            item.setToolTip(
                "Unmixed: this domain's modal proportion is redistributed onto "
                f"the endmember phases for the P-T estimate ({parts}; NNLS "
                f"residual {d.get('residual', float('nan')):.3f}).")
        else:
            item.setBackground(QtGui.QBrush())
            item.setToolTip("")

    def _unmix_selected_domain(self):
        """Decompose one mixed domain into areal fractions of the others (NNLS).

        For a domain that is a genuine sub-pixel mixture (fine-grained matrix
        below the map resolution), neither rejection nor merging gives honest
        modes: the GMM assigns the mixture its own confident component, and
        merging counts every mixed pixel as the dominant phase. Net counts add
        linearly over sub-pixel phase fractions, so NNLS of the domain's mean
        count vector against the clean domains' mean vectors recovers the
        areal fractions — which then redistribute this domain's pixel count in
        the observed modal proportions (``_observed_modes_from_phase_map``).
        """
        if not self._band_maps:
            return
        rows = sorted({ix.row() for ix in self._table.selectedIndexes()})
        if len(rows) != 1:
            QtWidgets.QMessageBox.information(
                self, "Unmix domain",
                "Select exactly one row: the domain to decompose (the mixed "
                "one).")
            return
        src = self._clusters[rows[0]]
        others = [c for c in self._clusters if c != src]
        if len(others) < 1:
            QtWidgets.QMessageBox.information(
                self, "Unmix domain", "Need at least one other domain to "
                "serve as an endmember.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Unmix '{self._row_name(src)}'")
        lay = QtWidgets.QVBoxLayout(dlg)
        intro = QtWidgets.QLabel(
            f"Decompose <b>{self._row_name(src)}</b> as a linear mixture of "
            "the checked endmember domains (NNLS on mean net-count vectors). "
            "Uncheck any domain that is itself a mixture. The fit quality is "
            "judged by the residual (a few % or less = good) and by the "
            "fraction sum (near 1 = the endmembers explain the signal).")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        lst = QtWidgets.QListWidget()
        lst.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        prev = (self._unmix.get(src) or {}).get("fractions") or {}
        for c in others:
            item = QtWidgets.QListWidgetItem(self._row_name(c))
            item.setData(QtCore.Qt.UserRole, c)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            checked = (c in prev) if prev else True
            item.setCheckState(QtCore.Qt.Checked if checked
                               else QtCore.Qt.Unchecked)
            lst.addItem(item)
        lst.setMaximumHeight(24 * max(3, len(others)))
        lay.addWidget(lst)

        res_lbl = QtWidgets.QLabel("")
        res_lbl.setWordWrap(True)
        lay.addWidget(res_lbl)

        state = {"fractions": None, "residual": None}

        def _compute():
            ecids = [lst.item(i).data(QtCore.Qt.UserRole)
                     for i in range(lst.count())
                     if lst.item(i).checkState() == QtCore.Qt.Checked]
            if not ecids:
                res_lbl.setText("Check at least one endmember.")
                return
            from viewer_thermo_helpers import _unmix_domain
            means = {}
            means[src], n_src = self._domain_mean_counts(src, interior=False)
            for c in ecids:
                means[c], _n = self._domain_mean_counts(c, interior=True)
            try:
                fr, resid = _unmix_domain(means, src, ecids)
            except Exception as exc:
                res_lbl.setText(f"Unmix failed: {exc}")
                return
            state["fractions"] = fr
            state["residual"] = resid
            tot = sum(fr.values())
            lines = [f"{self._row_name(c)}: {100.0 * f:.1f}%"
                     for c, f in sorted(fr.items(), key=lambda kv: -kv[1])]
            res_lbl.setText(
                "Areal fractions (of the mixed domain's area):  "
                + "   ".join(lines)
                + f"<br>fraction sum = {tot:.3f}   relative residual = "
                f"{resid:.4f}   ({n_src:,} px decomposed)"
                + ("<br><i>Warning: fraction sum far from 1 — the checked "
                   "endmembers do not explain this domain's signal.</i>"
                   if not 0.8 <= tot <= 1.2 else ""))
            btn_apply.setEnabled(True)

        btn_row2 = QtWidgets.QHBoxLayout()
        btn_compute = QtWidgets.QPushButton("Compute")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setEnabled(False)
        btn_clear = QtWidgets.QPushButton("Clear unmix")
        btn_clear.setEnabled(src in self._unmix)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_row2.addWidget(btn_compute)
        btn_row2.addWidget(btn_apply)
        btn_row2.addWidget(btn_clear)
        btn_row2.addStretch()
        btn_row2.addWidget(btn_cancel)
        lay.addLayout(btn_row2)

        btn_compute.clicked.connect(_compute)
        btn_cancel.clicked.connect(dlg.reject)

        def _apply():
            self._unmix[src] = {
                "fractions": {int(c): float(f)
                              for c, f in state["fractions"].items()
                              if float(f) > 0},
                "residual": float(state["residual"]),
            }
            self._mark_unmixed_row(src, True)
            print(f"[phases] unmix '{self._row_name(src)}' (cid {src}): "
                  + ", ".join(f"{self._row_name(c)} {100 * f:.1f}%"
                              for c, f in sorted(
                                  self._unmix[src]["fractions"].items(),
                                  key=lambda kv: -kv[1]))
                  + f" (residual {state['residual']:.4f}) — modal proportions "
                  "for the P-T estimate are redistributed accordingly",
                  flush=True)
            dlg.accept()

        def _clear():
            self._unmix.pop(src, None)
            self._mark_unmixed_row(src, False)
            print(f"[phases] unmix of '{self._row_name(src)}' cleared",
                  flush=True)
            dlg.accept()

        btn_apply.clicked.connect(_apply)
        btn_clear.clicked.connect(_clear)
        dlg.exec_()

    def unmix_map(self) -> Dict[int, Dict[int, float]]:
        """``{source_cid: {endmember_cid: fraction}}`` for the modes scorer."""
        return {int(c): dict(d.get("fractions") or {})
                for c, d in self._unmix.items() if d.get("fractions")}

    def unmix_full(self) -> Dict[int, dict]:
        """Full unmix records (fractions + residual) for session/export."""
        return {int(c): {"fractions": {int(e): float(f)
                                       for e, f in (d.get("fractions") or {}).items()},
                         "residual": float(d.get("residual", float("nan")))}
                for c, d in self._unmix.items()}

    def set_unmix(self, unmix: Optional[Dict] = None) -> None:
        """Restore unmix state (session/reopen path). Keys may be strings."""
        self._unmix = {}
        for c, d in (unmix or {}).items():
            fr = {int(e): float(f) for e, f in (d.get("fractions") or {}).items()}
            if not fr or int(c) not in self._clusters:
                continue
            if any(e not in self._clusters for e in fr):
                continue
            self._unmix[int(c)] = {"fractions": fr,
                                   "residual": float(d.get("residual",
                                                           float("nan")))}
            self._mark_unmixed_row(int(c), True)

    def coating_value(self) -> frozenset:
        """Coating/contaminant elements parsed from the edit (blank = uncoated).
        Symbols are normalised to element-case (first letter upper, rest lower)."""
        raw = self._coating_edit.text() if hasattr(self, "_coating_edit") else ""
        out = set()
        for tok in raw.replace(";", ",").split(","):
            t = tok.strip()
            if t:
                out.add(t[:1].upper() + t[1:].lower())
        return frozenset(out)

    def _add_napari_layer(self):
        name_map: Dict[int, str] = {}
        for row, cid in enumerate(self._clusters):
            item = self._table.item(row, 2)
            name_map[cid] = item.text().strip() if item and item.text().strip() else f"Phase {cid + 1}"

        mode_dlg = QtWidgets.QMessageBox(self)
        mode_dlg.setWindowTitle("Add to viewer")
        mode_dlg.setText("How would you like to add the phase map?")
        btn_combined = mode_dlg.addButton("Combined (single layer)", QtWidgets.QMessageBox.AcceptRole)
        btn_separate = mode_dlg.addButton("Per-cluster (one layer each)", QtWidgets.QMessageBox.AcceptRole)
        mode_dlg.addButton(QtWidgets.QMessageBox.Cancel)
        mode_dlg.exec_()
        clicked = mode_dlg.clickedButton()
        if clicked not in (btn_combined, btn_separate):
            return

        _remove_phase_layers(self._viewer)
        ty, tx = self._translate

        if clicked == btn_separate:
            _add_phase_cluster_layers(
                self._viewer, self._label_map, name_map, self._translate,
                scale=self._scale,
            )
        else:
            rgba = _label_map_to_rgba(self._label_map, self._n_clusters)
            self._viewer.add_image(
                rgba, name="Phase segmentation", rgb=True,
                blending='translucent_no_depth', opacity=0.7,
                scale=self._scale, translate=(ty, tx), visible=True,
            )

        QtWidgets.QMessageBox.information(
            self, "Layer added",
            f"Phase segmentation layer(s) added/updated in the viewer."
        )

    def _export_phase_map(self):
        """Export the phase map as a combined image or per-cluster images."""
        import imageio.v2

        name_map: Dict[int, str] = {}
        for row, cid in enumerate(self._clusters):
            item = self._table.item(row, 2)
            name_map[cid] = item.text().strip() if item and item.text().strip() else f"Phase_{cid + 1}"

        mode_dlg = QtWidgets.QMessageBox(self)
        mode_dlg.setWindowTitle("Export phase map")
        mode_dlg.setText("How would you like to export the phase map?")
        btn_combined = mode_dlg.addButton("Combined (single image)", QtWidgets.QMessageBox.AcceptRole)
        btn_separate = mode_dlg.addButton("Per-cluster (one image each)", QtWidgets.QMessageBox.AcceptRole)
        btn_both = mode_dlg.addButton("Both", QtWidgets.QMessageBox.AcceptRole)
        mode_dlg.addButton(QtWidgets.QMessageBox.Cancel)
        mode_dlg.exec_()
        clicked = mode_dlg.clickedButton()
        if clicked not in (btn_combined, btn_separate, btn_both):
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export directory")
        if not out_dir:
            return
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        do_combined = clicked in (btn_combined, btn_both)
        do_separate = clicked in (btn_separate, btn_both)
        saved: List[str] = []

        if do_combined:
            rgba = _label_map_to_rgba(self._label_map, self._n_clusters)
            imageio.v2.imwrite(str(outp / "phase_map_combined.png"), rgba)
            np.save(str(outp / "phase_map_labels.npy"), self._label_map)
            saved.append("phase_map_combined.png")
            saved.append("phase_map_labels.npy")

        if do_separate:
            tab20 = matplotlib.colormaps['tab20']
            H, W = self._label_map.shape
            for cid in self._clusters:
                mask = self._label_map == cid
                if not mask.any():
                    continue
                color_u8 = (np.array(tab20(cid % 20)) * 255).astype(np.uint8)
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                rgba[mask] = color_u8
                safe_name = name_map[cid].replace(" ", "_").replace("/", "-")
                fname = f"phase_{cid}_{safe_name}.png"
                imageio.v2.imwrite(str(outp / fname), rgba)
                saved.append(fname)

        import csv
        legend_path = outp / "phase_legend.csv"
        with open(legend_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cluster_id", "name", "pixel_count"])
            for cid in self._clusters:
                count = int((self._label_map == cid).sum())
                w.writerow([cid, name_map[cid], count])
        saved.append("phase_legend.csv")

        # Mineral identification results (sibling file; leaves phase_legend.csv
        # intact for tooling that depends on its 3-column contract).
        if self._mineral_matches:
            assignments = self.mineral_assignments()
            with open(outp / "phase_minerals.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "cluster_id", "name", "pixel_count",
                    "top1_mineral", "top1_abbrev", "top1_score",
                    "top1_confidence", "top1_method", "Mg#", "An",
                    "top2_mineral", "top2_score", "top3_mineral", "top3_score",
                    "flags",
                ])
                for cid in self._clusters:
                    count = int((self._label_map == cid).sum())
                    a = assignments.get(cid)
                    if not a:
                        w.writerow([cid, name_map[cid], count] + [""] * 12)
                        continue
                    cands = a["candidates"]
                    ix = a["indices"]
                    matches = self._mineral_matches.get(cid) or []
                    def _sc(i):
                        return f"{matches[i].score:.4f}" if i < len(matches) else ""
                    w.writerow([
                        cid, name_map[cid], count,
                        a["name"], a["abbrev"], f"{a['score']:.4f}",
                        a["confidence"], a["method"],
                        f"{ix.get('Mg#', ''):.4f}" if "Mg#" in ix else "",
                        f"{ix.get('An', ''):.4f}" if "An" in ix else "",
                        cands[1]["name"] if len(cands) > 1 else "", _sc(1),
                        cands[2]["name"] if len(cands) > 2 else "", _sc(2),
                        ";".join(a["flags"]),
                    ])
            saved.append("phase_minerals.csv")

        # Full per-element composition for every phase (the EDS signal used for
        # phase ID), in wt% (quantified) or net-count % (Bruker).
        if self._band_maps:
            elements = self._ordered_elements()
            units = self._composition_units()
            with open(outp / "phase_composition.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cluster_id", "name", "pixel_count", "units"] + elements)
                for cid in self._clusters:
                    frac, n_px = self._domain_composition(cid)
                    w.writerow(
                        [cid, name_map[cid], n_px, units]
                        + [f"{100.0 * float(frac.get(el, 0.0)):.4f}" for el in elements])
            saved.append("phase_composition.csv")

        # Segmentation replication recipe as a sidecar (kept out of the CSVs so
        # their column contracts stay parseable by plain readers). The
        # interior-only erosion rides along because it changes the composition
        # values written above (phase_composition.csv pixel_count = the pixels
        # actually averaged; phase_legend.csv keeps full-mask counts for modes).
        erosion = self.erosion_value()
        if self._seg_params or erosion or self._unmix:
            import json as _json
            recipe = dict(self._seg_params or {})
            if erosion:
                recipe["comp_erosion_px"] = int(erosion)
                um = self.erosion_um()
                if um is not None:
                    recipe["comp_erosion_um"] = round(um, 3)
            if self._unmix:
                # Mixture decomposition applied to the modal proportions
                # (source cid -> endmember cid -> areal fraction + residual).
                recipe["unmix"] = {str(c): {
                    "fractions": {str(e): round(float(f), 6)
                                  for e, f in d["fractions"].items()},
                    "residual": round(float(d.get("residual", float("nan"))), 6),
                } for c, d in self._unmix.items()}
            # Outcome stat (not a recipe input): how much of the map the
            # rejection left unclassified — readers of the export should not
            # have to load the label npy to learn the classified coverage.
            n_unc = int((self._label_map < 0).sum())
            if n_unc:
                recipe["unclassified_px"] = n_unc
                recipe["unclassified_frac"] = round(
                    n_unc / self._label_map.size, 6)
            with open(outp / "segmentation_params.json", "w") as f:
                _json.dump(recipe, f, indent=2, default=str)
            saved.append("segmentation_params.json")

        QtWidgets.QMessageBox.information(
            self, "Export complete",
            f"Saved {len(saved)} files to:\n{outp}\n\n" + "\n".join(saved),
        )

    def _show_bic_curve(self):
        if not self._bic_scores and not self._icl_scores:
            return
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("BIC / ICL Curves")
        dlg.resize(520, 380)
        lay = QtWidgets.QVBoxLayout(dlg)

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        canvas = FigureCanvas(fig)
        lay.addWidget(canvas)

        if self._bic_scores:
            ks_bic = sorted(self._bic_scores.keys())
            vals_bic = [self._bic_scores[k] for k in ks_bic]
            ax.plot(ks_bic, vals_bic, 's-', markersize=4, color='#1f77b4',
                    alpha=0.5, label='BIC')

        if self._icl_scores:
            ks_icl = sorted(self._icl_scores.keys())
            vals_icl = [self._icl_scores[k] for k in ks_icl]
            ax.plot(ks_icl, vals_icl, 'o-', markersize=5, color='#e67e22',
                    label='ICL')

        method_label = ""
        if self._auto_method == "icl_min":
            method_label = " (ICL min)"
        elif self._auto_method == "icl_knee":
            method_label = " (ICL knee)"
        ax.axvline(self._n_clusters, color='red', ls='--',
                   label=f'chosen k={self._n_clusters}{method_label}')

        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Score (lower = better)')
        ax.set_title('Model selection — BIC vs ICL')
        ax.legend(fontsize=8)
        fig.tight_layout()
        canvas.draw()

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)

        dlg.finished.connect(lambda _: plt.close(fig))
        dlg.exec_()

    def closeEvent(self, event):
        import matplotlib.pyplot as plt
        plt.close(self._fig)
        super().closeEvent(event)


# -------------------------
# Zoning / traverse dialog
# -------------------------
class _ZoningDialog(QtWidgets.QDialog):
    """Per-pixel index maps (Mg#, An, ...) + core-to-rim traverse profiles.

    Modeless on purpose: the user must be able to draw the traverse line in
    napari while this window is open (use ``show()``, not ``exec_()``; the
    panel keeps a reference). Index maps are computed by ``zoning.index_map``/
    ``element_map`` on the ROI band maps (optionally restricted to one phase
    domain) and added as napari Image layers at mosaic resolution; a traverse
    is sampled from the last line in a "Traverse" Shapes layer and plotted /
    exported as CSV. Distances are mosaic px, plus µm when the pixel size is
    known from the cache manifest.
    """

    TRAVERSE_LAYER = "Traverse"

    def __init__(self, band_maps, k_factors, translate, ds, viewer,
                 roi_mask=None, phase_map=None, phase_names=None,
                 pixel_um=None, quantified=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoning / traverse")
        self.setWindowFlag(QtCore.Qt.Tool, True)   # stay-on-top tool window
        self._band_maps = band_maps
        self._k = dict(k_factors) if k_factors else None
        self._ty, self._tx = float(translate[0]), float(translate[1])
        self._ds = max(1, int(ds))
        self._viewer = viewer
        self._roi_mask = roi_mask
        self._phase_map = phase_map
        self._phase_names = {int(c): str(n) for c, n in (phase_names or {}).items()}
        self._pixel_um = pixel_um
        self._quantified = bool(quantified or k_factors)
        self._map = None          # last computed value map
        self._map_label = ""
        self._profile = None      # (dist_mosaic_px, values, label)

        import zoning as _z
        self._zoning = _z

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )

        lay = QtWidgets.QVBoxLayout(self)

        if not self._quantified:
            cav = QtWidgets.QLabel(
                "⚠ Uncalibrated net counts: maps are intensity ratios — "
                "zoning patterns are real, absolute values are not. Run "
                "Calibrate EDS (SPX k-factors) for true molar ratios.")
            cav.setWordWrap(True)
            cav.setStyleSheet("color: #8a6d00;")
            lay.addWidget(cav)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Map:"))
        self.cmb_kind = QtWidgets.QComboBox()
        for name, (num, den) in _z.INDEX_DEFS.items():
            self.cmb_kind.addItem(
                f"{name}  ({'+'.join(num)})/({'+'.join(den)})", name)
        self.cmb_kind.addItem("Element %", "__element__")
        row.addWidget(self.cmb_kind)
        self.cmb_el = QtWidgets.QComboBox()
        for el in sorted(band_maps):
            self.cmb_el.addItem(el)
        self.cmb_el.setEnabled(False)
        row.addWidget(self.cmb_el)
        row.addStretch()
        lay.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Restrict to phase:"))
        self.cmb_domain = QtWidgets.QComboBox()
        self.cmb_domain.addItem("(whole ROI)", None)
        if self._phase_map is not None:
            for cid in sorted(self._phase_names):
                self.cmb_domain.addItem(self._phase_names[cid], cid)
        else:
            self.cmb_domain.setEnabled(False)
            self.cmb_domain.setToolTip(
                "No ROI-aligned phase map — segment phases to restrict the "
                "map to one domain.")
        row2.addWidget(self.cmb_domain)
        row2.addWidget(QtWidgets.QLabel("Smooth σ:"))
        self.spin_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_sigma.setRange(0.0, 10.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(1.0)
        self.spin_sigma.setToolTip(
            "Gaussian smoothing of the net-count maps BEFORE the ratio "
            "(stabilises per-pixel counting noise).")
        row2.addWidget(self.spin_sigma)
        row2.addStretch()
        lay.addLayout(row2)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_map = QtWidgets.QPushButton("Show map in napari")
        self.btn_line = QtWidgets.QPushButton("Draw traverse (line)")
        self.btn_line.setToolTip(
            "Activates a 'Traverse' Shapes layer in line-drawing mode — draw "
            "the core-to-rim line on the image, then click Sample.")
        self.btn_sample = QtWidgets.QPushButton("Sample traverse")
        btn_row.addWidget(self.btn_map)
        btn_row.addWidget(self.btn_line)
        btn_row.addWidget(self.btn_sample)
        btn_row.addWidget(QtWidgets.QLabel("Avg width (px):"))
        self.spin_width = QtWidgets.QSpinBox()
        self.spin_width.setRange(1, 99)
        self.spin_width.setValue(3)
        btn_row.addWidget(self.spin_width)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._fig = plt.figure(figsize=(6, 3))
        self._canvas = FigureCanvas(self._fig)
        lay.addWidget(NavigationToolbar(self._canvas, self))
        lay.addWidget(self._canvas)

        bottom = QtWidgets.QHBoxLayout()
        self.btn_csv = QtWidgets.QPushButton("Export profile CSV …")
        self.btn_csv.setEnabled(False)
        btn_close = QtWidgets.QPushButton("Close")
        bottom.addWidget(self.btn_csv)
        bottom.addStretch()
        bottom.addWidget(btn_close)
        lay.addLayout(bottom)

        self.cmb_kind.currentIndexChanged.connect(
            lambda _: self.cmb_el.setEnabled(
                self.cmb_kind.currentData() == "__element__"))
        self.btn_map.clicked.connect(self._on_show_map)
        self.btn_line.clicked.connect(self._on_draw_line)
        self.btn_sample.clicked.connect(self._on_sample)
        self.btn_csv.clicked.connect(self._on_csv)
        btn_close.clicked.connect(self.close)
        self.resize(640, 560)

    # ---- map computation -------------------------------------------------
    def _sub_mask(self):
        mask = None
        if self._roi_mask is not None:
            mask = np.asarray(self._roi_mask, dtype=bool)
        cid = self.cmb_domain.currentData()
        if cid is not None and self._phase_map is not None:
            dom = (np.asarray(self._phase_map) == int(cid))
            mask = dom if mask is None else (mask & dom)
        return mask

    def _compute_map(self):
        kind = self.cmb_kind.currentData()
        sigma = float(self.spin_sigma.value())
        mask = self._sub_mask()
        if kind == "__element__":
            el = self.cmb_el.currentText()
            vm = self._zoning.element_map(
                self._band_maps, el, k_factors=self._k,
                smooth_sigma=sigma, sub_mask=mask)
            label = (f"{el} wt%" if (self._k and el in self._k)
                     else f"{el} %")
        else:
            vm = self._zoning.index_map(
                self._band_maps, kind, k_factors=self._k,
                smooth_sigma=sigma, sub_mask=mask)
            label = kind
        cid = self.cmb_domain.currentData()
        if cid is not None:
            label += f" [{self._phase_names.get(int(cid), cid)}]"
        if not self._quantified:
            label += " (uncal.)"
        self._map = vm
        self._map_label = label
        return vm, label

    def _on_show_map(self):
        vm, label = self._compute_map()
        if not np.isfinite(vm).any():
            QtWidgets.QMessageBox.information(
                self, "Empty map", "No pixel has a defined value (missing "
                "elements or empty domain).")
            return
        lo, hi = np.nanpercentile(vm, [2, 98])
        if not np.isfinite(lo) or hi <= lo:
            lo, hi = float(np.nanmin(vm)), float(np.nanmax(vm) or 1.0)
        name = f"Zoning: {label}"
        if name in self._viewer.layers:
            self._viewer.layers.remove(self._viewer.layers[name])
        self._viewer.add_image(
            vm, name=name, colormap="turbo",
            contrast_limits=(float(lo), float(hi)),
            translate=(self._ty, self._tx),
            scale=(float(self._ds), float(self._ds)),
            blending="translucent",
        )

    # ---- traverse --------------------------------------------------------
    def _traverse_layer(self):
        if self.TRAVERSE_LAYER in self._viewer.layers:
            return self._viewer.layers[self.TRAVERSE_LAYER]
        return self._viewer.add_shapes(
            name=self.TRAVERSE_LAYER, shape_type="line",
            edge_color="cyan", edge_width=3,
        )

    def _on_draw_line(self):
        lyr = self._traverse_layer()
        self._viewer.layers.selection.active = lyr
        lyr.mode = "add_line"

    def _on_sample(self):
        if self._map is None:
            self._compute_map()
        lyr = (self._viewer.layers[self.TRAVERSE_LAYER]
               if self.TRAVERSE_LAYER in self._viewer.layers else None)
        if lyr is None or len(lyr.data) == 0:
            QtWidgets.QMessageBox.information(
                self, "No traverse", "Draw a line in the 'Traverse' layer "
                "first (button to the left).")
            return
        pts = np.asarray(lyr.data[-1], dtype=float)   # last drawn shape
        p0_m, p1_m = pts[0], pts[-1]                  # mosaic coords (y, x)
        # mosaic -> ROI-local map coords
        p0 = ((p0_m[0] - self._ty) / self._ds, (p0_m[1] - self._tx) / self._ds)
        p1 = ((p1_m[0] - self._ty) / self._ds, (p1_m[1] - self._tx) / self._ds)
        length_map_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if length_map_px < 2:
            QtWidgets.QMessageBox.information(
                self, "Traverse too short",
                "The line is shorter than 2 map pixels.")
            return
        n = int(min(2000, max(50, np.ceil(length_map_px) + 1)))
        dist, vals = self._zoning.line_profile(
            self._map, p0, p1, n_samples=n,
            width=int(self.spin_width.value()))
        dist_mosaic = dist * self._ds
        self._profile = (dist_mosaic, vals, self._map_label)
        self.btn_csv.setEnabled(True)

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if self._pixel_um:
            ax.plot(dist_mosaic * float(self._pixel_um), vals, "-")
            ax.set_xlabel("distance [µm]")
        else:
            ax.plot(dist_mosaic, vals, "-")
            ax.set_xlabel("distance [mosaic px]")
        ax.set_ylabel(self._map_label)
        n_ok = int(np.isfinite(vals).sum())
        ax.set_title(f"Traverse — {self._map_label}  ({n_ok}/{len(vals)} "
                     f"samples defined)", fontsize=9)
        self._fig.tight_layout()
        self._canvas.draw()

    def _on_csv(self):
        if not self._profile:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export traverse CSV", "traverse_profile.csv",
            "CSV files (*.csv)")
        if not path:
            return
        import csv
        dist, vals, label = self._profile
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            hdr = ["distance_mosaic_px"]
            if self._pixel_um:
                hdr.append("distance_um")
            hdr.append(label)
            w.writerow(hdr)
            for i in range(len(dist)):
                row = [f"{dist[i]:.2f}"]
                if self._pixel_um:
                    row.append(f"{dist[i] * float(self._pixel_um):.2f}")
                row.append("" if not np.isfinite(vals[i]) else f"{vals[i]:.5f}")
                w.writerow(row)

    def closeEvent(self, event):
        import matplotlib.pyplot as plt
        plt.close(self._fig)
        super().closeEvent(event)
