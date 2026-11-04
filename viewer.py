# viewer.py
"""
Interactive viewer for stitched BSE + on-demand EDS elemental maps (ROI rectangle or polygon).

Features:
    - Mosaic BSE display with optional multiscale pyramid
    - Downsampled EDS element preview maps (auto-loaded from eds_preview_ds*/)
    - Full-resolution EDS maps for ROIs (banded extraction with background subtraction)
    - Phase segmentation: Gaussian smoothing → feature normalisation → PCA reduction
      → GMM clustering with automatic k selection via ICL (BIC + classification entropy)
      with knee detection fallback, or manual k; tile overlap exclusion; median-ratio
      low-count filter; morphological cleanup; nearest-neighbor fill for excluded pixels;
      per-cluster or combined napari layers; export (combined/per-cluster/both + legend CSV)
    - Thermodynamic modelling integration (MAGEMin, Perple_X, pycalphad)
    - P-T estimation by comparing segmented/observed phases against computed assemblages
    - Thermod run-record history (last_thermod_runs):
        every single/grid run is appended as a serialisable dict; the
        "Modelling runs …" dialog supports View results / View parameters /
        Locate ROI / Compare selected (phase-field agreement + per-property
        spread heatmaps) / Export bundle / Delete. Records persist in the
        session JSON and re-open without a live backend.
    - Composition vs length scale ("Composition vs scale …"):
        quadtree subdivision of the active ROI with per-window composition
        (EDS net-count or SPX-quantified), per-scale variance plots, and an
        optional thermodynamic sensitivity sweep (μ, μ±σ per level → grid
        runs added to the run history). Per-level cap + sampling strategy
        (first-N or random with fixed seed). SPX path uses the median tile
        short-side as its window floor; EDS path uses 4 px. Max useful
        depth is auto-derived from the ROI size and re-clamps on source
        toggle.
    - Session save/load (preserves ROI maps, compositions, phase maps,
      thermod runs, and ROI geometry)
    - Multi-vendor input: reads both Bruker (built by stitch.py) and
      Oxford AZtec CSV (built by oxford_loader.py) mosaic directories.
      The cache's data_type field is consulted to disable SPX for
      Oxford-tagged caches and to relabel sources in run records.
    - ROIs are auto-clipped to the BSE mosaic bounds; partly-outside
      ROIs are silently clipped, fully-outside ROIs raise a clear
      error instead of trickling down as cryptic shape exceptions.

Usage:
    python viewer.py --dir path/to/mosaic \\
        --bse stitched_BSE_8bit.png \\
        --manifest stitch_manifest.csv \\
        --preview-auto --elements auto \\
        [--session viewer_session.json] \\
        [--debug-manifest] [--debug-spx]

Requirements:
    pip install napari[all] numpy scipy imageio qtpy matplotlib tifffile scikit-learn
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import argparse
import warnings
import uuid
import numpy as np

# Suppress noisy third-party warnings we cannot fix
warnings.filterwarnings(
    "ignore",
    message="Failed to import MELTSdynamic",
    category=UserWarning,
)
from qtpy import QtWidgets, QtCore, QtGui
import napari
import imageio
from matplotlib.path import Path as MplPath
import matplotlib
from PIL import Image
from datetime import datetime

# EDS helpers
from eds_map import (
    read_manifest,
    roi_to_banded_maps,
    roi_to_summed_spectrum_streamed,  # fast, low-memory spectrum export
)

Image.MAX_IMAGE_PIXELS = 2_000_000_000


# -------------------------
# Re-imports from extracted submodules (tier-B refactor).
# Names are re-exported here so EDSControlPanel and main() keep working
# unchanged, and so external `from viewer import _foo` continues to resolve.
# -------------------------
from viewer_helpers import (
    LINE_ENERGIES_KEV, _PT,
    _imread_robust,
    _to_gray, _build_pyramid,
    _find_preview_dir, _auto_load_previews,
    _rasterize_polygon_mask, _rgba_float_from_gray_and_mask,
    _row_bbox_on_canvas, _overlap_area, _coords_from_name,
    _match_spx_for_bcf, _fnum, _parse_spx_quant,
    _write_emsa_msa, _write_perplex_build_keyword,
    elements_to_oxides_wt, add_stoichiometric_oxygen,
    _available_memory_bytes, _roi_map_bytes, _downsample_to_fit,
)
from viewer_thermo_helpers import (
    _make_phase_field_map, _pt_estimate_from_results,
    _phase_matches, _observed_modes_from_phase_map, _modal_misfit_grid,
    _best_fit_pt, _pt_scale_rows, _pt_region_from_misfit,
    _pca_perturbed_compositions,
    _mc_sample_compositions, _mc_aggregate, _mc_sample_modes,
    _net_count_fractions, _net_count_sums, _max_useful_quadtree_depth, _quadtree_windows,
    _quadtree_sampled_windows, _quadtree_level_dims,
    _semivariogram, _morans_i, _eligible_variogram_levels,
    _spread_by_level, _representative_level,
    _serialize_phase_result, _deserialize_phase_result,
    _serialize_thermod_result, _deserialize_thermod_result,
    _format_composition_lines, _thermod_parameters_text,
    _export_thermod_bundle, _assemblage_selector_dialog,
)
from viewer_segment_helpers import (
    _PHASE_LAYER_PREFIX,
    _collect_segment_maps, _smooth_element_maps, _clean_label_map,
    _compute_overlap_mask_from_weight, _compute_overlap_mask_from_manifest,
    _build_feature_matrix, _normalize_features,
    _sparse_element_exclusions,
    _find_knee, _select_k_auto, _run_clustering, _fill_unlabeled,
    _zero_signal_mask,
    _cluster_dominant_elements, _label_map_to_rgba,
    _remove_phase_layers, _add_phase_cluster_layers,
)
from viewer_workers import (
    _EDSWorker, _EDSCacheWorker,
    _ThermoGridWorker, _ThermoSweepWorker, _CompositionScaleWorker,
)
from viewer_dialogs import (
    _hline,
    _ThermoRunDialog, _CompositionScaleDialog,
    _PhaseSegDialog, _PhaseClassDialog, _PhaseCompositionDialog,
    _CompositionCompareDialog, _ZoningDialog,
)


# -------------------------
# Control panel (dock widget)
# -------------------------
class EDSControlPanel(QtWidgets.QWidget):
    def __init__(self,
                 viewer: napari.Viewer,
                 manifest_path: Path,
                 available_elements: List[str],
                 preselected: Optional[List[str]] = None,
                 bse_layer_name: str = 'BSE',
                 debug_manifest: bool = False,
                 debug_spx: bool = False):
        super().__init__()
        self.viewer = viewer
        self.manifest = Path(manifest_path)
        self.bse_layer_name = bse_layer_name
        self.available_elements = list(available_elements)
        self.debug_manifest = debug_manifest
        self.debug_spx = debug_spx

        # Worker thread for async EDS computation
        self._eds_worker: Optional[_EDSWorker] = None
        # Cached manifest rows (parsed once per session to avoid CSV re-reads)
        self._manifest_rows: Optional[List[dict]] = None
        # Caches for the multi-scale SPX path: derived once per session and
        # reused across many sub-window calls in _spx_composition_for_bbox.
        self._tile_bboxes_cache: Optional[List[Tuple[Tuple[int, int, int, int], dict]]] = None
        self._spx_parse_cache: Dict[Path, Tuple[Dict[str, float], float, List[str]]] = {}
        # Worker for the composition-vs-scale tool
        self._comp_scale_worker: Optional["_CompositionScaleWorker"] = None
        # Last composition-vs-scale result (records + summary + opts + ROI), kept
        # so the result window re-opens without re-running the worker.
        self.last_comp_scale: Optional[dict] = None

        # Keep last results for export
        self.last_band_maps: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self.last_band_maps_ds: int = 1   # ROI-map decimation (>1 if downsampled to fit RAM)
        self.last_blend_mode: str = "mean"  # blend used for the last ROI compute
        self.last_weight_map: Optional[np.ndarray] = None
        self.last_roi: Optional[Tuple[int, int, int, int]] = None
        self.last_roi_mask: Optional[np.ndarray] = None
        # Full-res polygon mask kept only when the display maps were downsampled
        # (ds>1), so cache-backed full-res readers (composition / export) can
        # mask at native resolution. None when ds==1 (last_roi_mask is full-res).
        self.last_roi_mask_fullres: Optional[np.ndarray] = None
        self.last_elements: List[str] = []
        self.last_composition: Optional[Dict[str, float]] = None  # net fractions (0..1)
        self.last_spx_composition: Optional[Dict[str, float]] = None  # element wt% from SPX ROI
        # SPX-anchored empirical calibration of the EDS net-count maps. When set
        # for the active ROI, calibrated wt% (eds_quantify) is preferred over raw
        # net-count fractions everywhere downstream (modelling bulk, composition-
        # vs-scale, mineral ID structural recalc). Invalidated on a new compute.
        self.eds_calibration = None                                   # eds_quantify.Calibration | None
        self.eds_calibration_cv = None                                # eds_quantify.CrossValidation | None
        self.last_calibrated_composition: Optional[Dict[str, float]] = None  # ROI wt% (sum=100)
        self._calibration_scope: Optional[str] = None                 # "mosaic" | "roi" | None
        self._calibration_fingerprint: Optional[str] = None           # cache fp when derived (staleness)
        self.last_phase_map: Optional[np.ndarray] = None              # (H,W) int16 label map, -1=invalid
        self.last_phase_names: Optional[Dict[int, str]] = None        # {cluster_id: name}
        self.last_phase_translate: Optional[Tuple[float, float]] = None  # napari (ty, tx)
        self.last_phase_ds: int = 1                                   # phase-map decimation (matches band_maps_ds for fullres)
        self.last_phase_minerals: Dict[int, dict] = {}                # {cid: mineral-ID dict}
        self.last_phase_composition: Dict[int, Dict[str, float]] = {} # {cid: {el: fraction}} per phase
        self.last_phase_fe3: float = 0.0                              # Fe3+/ΣFe used for ID
        self.last_phase_comp_erosion: int = 0                         # interior-only erosion px (0=off)
        # NNLS unmixing of mixed-matrix domains: {src_cid: {"fractions":
        # {end_cid: f}, "residual": r}}. Redistributes the source domain's
        # pixel count over the endmembers in the observed modal proportions
        # (resolution-bias correction); map/composition/ID untouched.
        self.last_phase_unmix: Dict[int, dict] = {}
        # Coating / contaminant element(s) to ignore in mineral ID + calibration.
        # None → resolve from the cache manifest ("coating_elements") or default
        # {"C"}; a frozenset is a session-level override set in the phase dialog.
        self._coating_override: Optional[frozenset] = None
        # EDS pixel pitch (µm/px) session override. None → resolve from the cache
        # manifest ("pixel_um"); a float set here (e.g. in the composition-vs-scale
        # dialog) wins, letting the user supply a pitch the data didn't record.
        self._pixel_um_override: Optional[float] = None
        # P-T inversion overrides set in the phase dialog (relax the
        # exact/complete/equilibrium assumptions of the modal misfit). All
        # default to "no override" so the estimate is unchanged until used.
        self.last_phase_pt_include: Optional[set] = None    # cids to keep (None=all)
        self.last_phase_pt_weights: Optional[Dict[int, float]] = None
        self.last_phase_extra: List[dict] = []              # phases segmentation missed
        self.last_phase_open_tol: float = 0.0               # open-assemblage forgiveness
        self.last_phase_source: str = "gmm"                           # 'gmm' or 'ebsd' (informational)
        # Full recipe of the segmentation that produced last_phase_map (dialog
        # options + seg_ds + final element list + chosen k). Persisted in the
        # session, stamped into exports and thermod run opts so a result can be
        # replicated even when the session file itself was never saved.
        self.last_phase_seg_params: Optional[dict] = None
        # EBSD phase map (Oxford h5oina): full-mosaic uint8 label array + legend
        # {ordinal: {name, formula, color}}. Loaded at startup when present; used
        # as ready-made domains for the active ROI via on_use_ebsd_phases().
        self.ebsd_phase_map: Optional[np.ndarray] = None
        self.ebsd_phase_legend: Optional[Dict[int, dict]] = None
        # Thermodynamic modelling run history (each entry is a serialisable dict
        # so it can be persisted in the session JSON and re-displayed without
        # re-running the calculation). See _record_thermod_run / _open_thermod_run.
        self.last_thermod_runs: List[dict] = []

        layout = QtWidgets.QVBoxLayout(self)

        # Element list
        layout.addWidget(QtWidgets.QLabel('Available elements (from metadata / previews):'))
        self.el_list = QtWidgets.QListWidget()
        self.el_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        pre = set(preselected or [])
        for el in self.available_elements:
            item = QtWidgets.QListWidgetItem(el)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if el in pre else QtCore.Qt.Unchecked)
            self.el_list.addItem(item)
        layout.addWidget(self.el_list)

        # Select/clear all
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton('Select all')
        self.btn_none = QtWidgets.QPushButton('Clear all')
        btn_row.addWidget(self.btn_all); btn_row.addWidget(self.btn_none)
        layout.addLayout(btn_row)
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none.clicked.connect(lambda: self._set_all(False))

        # Options
        self.win_spin = QtWidgets.QSpinBox(); self.win_spin.setRange(50, 2000); self.win_spin.setValue(150)
        self.sum_check = QtWidgets.QCheckBox('Sum counts (instead of mean blend)')
        self.subpix_check = QtWidgets.QCheckBox('Apply subpixel alignment'); self.subpix_check.setChecked(True)
        self.shift_method_combo = QtWidgets.QComboBox()
        self.shift_method_combo.addItem('Bilinear (fast, default)', 'bilinear')
        self.shift_method_combo.addItem('FFT / sinc (slower, exact for bandlimited signals)', 'fft')
        self.shift_method_combo.setCurrentIndex(0)

        form = QtWidgets.QFormLayout()
        form.addRow('Window width (eV):', self.win_spin)
        form.addRow('Subpixel shift method:', self.shift_method_combo)
        layout.addLayout(form)
        layout.addWidget(self.sum_check)
        layout.addWidget(self.subpix_check)

        # EDS cache status (updated by _update_cache_status_label) — tells the
        # user at a glance whether ROI reads use the fast mosaic cache.
        self.cache_status_label = QtWidgets.QLabel("EDS cache: …")
        self.cache_status_label.setWordWrap(True)
        layout.addWidget(self.cache_status_label)

        # Action buttons
        self.btn_compute = QtWidgets.QPushButton('Compute FULL-RES maps for ROI')
        self.btn_export = QtWidgets.QPushButton('Export ROI maps + composition')
        self.btn_export_spec = QtWidgets.QPushButton('Export ROI summed spectrum (MSA)')
        self.btn_spx_comp = QtWidgets.QPushButton('ROI composition from SPX (whole tiles)')
        self.btn_calib_eds = QtWidgets.QPushButton('Calibrate EDS (SPX k-factors) …')
        self.btn_calib_eds.setToolTip(
            'Derive per-element sensitivity (k-) factors by regressing EDS '
            'net-counts against the quantified SPX wt% of the tiles, then apply '
            'them so the uncalibrated EDS maps read as quasi-quantified wt%. '
            'One set of k-factors serves the whole mosaic (they are specific to '
            'the sample and session, not to the instrument): the default fits '
            'them from ALL SPX tiles in the mosaic (no ROI needed), with a '
            '"current ROI only" local-regime option. Bruker only; needs SPX.')
        self.btn_compare_comp = QtWidgets.QPushButton('Compare compositions (EDS / calibrated / SPX) …')
        self.btn_compare_comp.setToolTip(
            'Show the active ROI composition from raw EDS net-counts, '
            'SPX-calibrated EDS, and quantified SPX side by side (table + '
            'grouped bar chart). Derives the calibration on the fly when an SPX '
            'tile lies inside the ROI. Needs computed ROI EDS maps.')
        self.btn_comp_scale = QtWidgets.QPushButton('Composition vs scale …')
        self.btn_comp_scale_reopen = QtWidgets.QPushButton('Results …')
        self.btn_comp_scale_reopen.setToolTip(
            'Re-open the last composition-vs-scale result window without '
            'recomputing it.')
        self.btn_zoning = QtWidgets.QPushButton('Zoning / traverse …')
        self.btn_zoning.setToolTip(
            'Per-pixel index maps (Mg#, An, Or, …) and core-to-rim traverse '
            'profiles from the computed ROI EDS maps, optionally restricted to '
            'one segmented phase domain. Calibrated when SPX k-factors are '
            'active; intensity-ratio otherwise.')
        self.btn_export_perplex = QtWidgets.QPushButton('Export for thermodynamics (Perple_X)')
        self.btn_thermod = QtWidgets.QPushButton('Run thermodynamic model …')
        self.btn_thermod_runs = QtWidgets.QPushButton('Modelling runs …')
        self.btn_segment = QtWidgets.QPushButton('Segment phases …')
        self.btn_phase_comp = QtWidgets.QPushButton('Phase composition …')
        self.btn_phase_comp.setToolTip(
            'Show the full per-element composition of the segmented phases '
            '(table + bar chart). Available after a full-resolution segmentation.')
        self.btn_phase_comp.setEnabled(False)
        self.btn_phase_reopen = QtWidgets.QPushButton('Phase classification …')
        self.btn_phase_reopen.setToolTip(
            'Re-open the phase classification dialog (names, mineral ID, '
            'composition breakdown, P-T estimate) from the stored phase map — '
            'e.g. after loading a session — without re-clustering.')
        self.btn_phase_reopen.setEnabled(False)
        self.btn_clear = QtWidgets.QPushButton('Clear elemental layers')
        layout.addWidget(self.btn_compute)
        layout.addWidget(self.btn_export)
        layout.addWidget(self.btn_export_spec)
        layout.addWidget(self.btn_spx_comp)
        layout.addWidget(self.btn_calib_eds)
        layout.addWidget(self.btn_compare_comp)
        comp_scale_row = QtWidgets.QHBoxLayout()
        comp_scale_row.addWidget(self.btn_comp_scale)
        comp_scale_row.addWidget(self.btn_comp_scale_reopen)
        layout.addLayout(comp_scale_row)
        layout.addWidget(self.btn_zoning)
        layout.addWidget(self.btn_export_perplex)
        thermod_row = QtWidgets.QHBoxLayout()
        thermod_row.addWidget(self.btn_thermod)
        thermod_row.addWidget(self.btn_thermod_runs)
        layout.addLayout(thermod_row)
        self.btn_ebsd = QtWidgets.QPushButton('Use EBSD phases (ROI) …')
        self.btn_ebsd.setToolTip(
            'Use the imported EBSD phase map (Oxford h5oina) as ready-made phase '
            'domains for the active ROI, with crystallographic names. Compute the '
            "ROI's EDS maps first. Mineral ID then runs as a cross-check.")
        self.btn_ebsd.setEnabled(False)
        self.btn_ebsd_conf = QtWidgets.QPushButton('EBSD vs EDS confusion …')
        self.btn_ebsd_conf.setToolTip(
            'Validate the EDS phase map (GMM segmentation + mineral ID) against '
            'the indexed EBSD phase map: per-pixel confusion matrix with '
            'alias-matched accuracy, recall and precision, plus CSV export.')
        self.btn_ebsd_conf.setEnabled(False)
        segment_row = QtWidgets.QHBoxLayout()
        segment_row.addWidget(self.btn_segment)
        segment_row.addWidget(self.btn_phase_comp)
        layout.addLayout(segment_row)
        layout.addWidget(self.btn_phase_reopen)
        ebsd_row = QtWidgets.QHBoxLayout()
        ebsd_row.addWidget(self.btn_ebsd)
        ebsd_row.addWidget(self.btn_ebsd_conf)
        layout.addLayout(ebsd_row)
        layout.addWidget(self.btn_clear)

        # Session save/load
        layout.addWidget(_hline())
        session_row = QtWidgets.QHBoxLayout()
        self.btn_save_session = QtWidgets.QPushButton('Save session …')
        self.btn_load_session = QtWidgets.QPushButton('Load session …')
        session_row.addWidget(self.btn_save_session)
        session_row.addWidget(self.btn_load_session)
        layout.addLayout(session_row)
        layout.addStretch(1)

        self.btn_compute.clicked.connect(self.on_compute)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_export_spec.clicked.connect(self.on_export_spectrum_msa)
        self.btn_spx_comp.clicked.connect(self.on_spx_composition)
        self.btn_calib_eds.clicked.connect(self.on_calibrate_eds)
        self.btn_compare_comp.clicked.connect(self.on_compare_compositions)
        self.btn_comp_scale.clicked.connect(self.on_composition_scale)
        self.btn_comp_scale_reopen.clicked.connect(self.on_reopen_composition_scale)
        self.btn_zoning.clicked.connect(self.on_zoning)
        self.btn_export_perplex.clicked.connect(self.on_export_perplex)
        self.btn_thermod.clicked.connect(self.on_run_thermod)
        self.btn_thermod_runs.clicked.connect(self.on_show_thermod_runs)
        self.btn_segment.clicked.connect(self.on_segment_phases)
        self.btn_phase_comp.clicked.connect(self.on_show_phase_composition)
        self.btn_phase_reopen.clicked.connect(self.on_reopen_phase_classification)
        self.btn_ebsd.clicked.connect(self.on_use_ebsd_phases)
        self.btn_ebsd_conf.clicked.connect(self.on_ebsd_confusion)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_save_session.clicked.connect(self.on_save_session)
        self.btn_load_session.clicked.connect(self.on_load_session)

        # Adopt the EDS cache's window width so a Bruker cache is recognised out
        # of the box: the cache probe requires width_eV to match, and a cache
        # built at a non-default width would otherwise be silently ignored.
        try:
            _st = self._eds_cache_status()
            if (_st.get("exists") and _st.get("data_type") == "bcf_net_counts"
                    and _st.get("width_eV")):
                self.win_spin.setValue(int(round(float(_st["width_eV"]))))
        except Exception:
            pass

        # Grey out controls that don't apply to the freshly-loaded dataset, and
        # show the EDS-cache status line.
        self._refresh_control_availability()

    # ---- UI helpers ----
    def _set_all(self, state: bool):
        for i in range(self.el_list.count()):
            item = self.el_list.item(i)
            item.setCheckState(QtCore.Qt.Checked if state else QtCore.Qt.Unchecked)

    def _selected_elements(self) -> List[str]:
        els: List[str] = []
        for i in range(self.el_list.count()):
            item = self.el_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                els.append(item.text())
        return els

    def _data_bounds(self) -> Optional[Tuple[int, int]]:
        """Return ``(H, W)`` of the underlying mosaic, or None if unknown.

        Sourced from the BSE napari layer. Handles both single-scale and
        multiscale pyramid layers; for multiscale the largest level (index 0)
        is used.
        """
        try:
            lyr = self.viewer.layers[self.bse_layer_name]
        except (KeyError, IndexError):
            return None
        data = lyr.data
        try:
            if isinstance(data, list) and data:
                # multiscale pyramid: data[0] is the largest level
                shape = data[0].shape
            else:
                shape = data.shape
            H = int(shape[0])
            W = int(shape[1])
            if H > 0 and W > 0:
                return H, W
        except Exception:
            pass
        return None

    def _get_roi_bbox_and_mask(self) -> Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]:
        shapes = None
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Shapes) and lyr.name.lower() == 'roi':
                shapes = lyr
                break
        if shapes is None or len(shapes.data) == 0:
            raise RuntimeError("Draw a rectangle or polygon ROI in a Shapes layer named 'ROI'.")

        if getattr(shapes, "selected_data", None):
            idx = max(shapes.selected_data)
        else:
            idx = len(shapes.data) - 1

        verts = np.asarray(shapes.data[idx])  # (N,2) YX
        shape_type = shapes.shape_type[idx] if hasattr(shapes, 'shape_type') else 'rectangle'
        y0 = int(np.floor(verts[:, 0].min())); y1 = int(np.ceil(verts[:, 0].max()))
        x0 = int(np.floor(verts[:, 1].min())); x1 = int(np.ceil(verts[:, 1].max()))

        # Clip to the mosaic bounds so an ROI drawn partly off-data still
        # produces a valid bbox that downstream slicers (eds_cache,
        # roi_to_banded_maps, the quadtree tool) can handle.
        raw_bbox = (y0, y1, x0, x1)
        bounds = self._data_bounds()
        if bounds is not None:
            H, W = bounds
            cy0 = max(0, min(H, y0))
            cy1 = max(0, min(H, y1))
            cx0 = max(0, min(W, x0))
            cx1 = max(0, min(W, x1))
            if cy1 <= cy0 or cx1 <= cx0:
                raise RuntimeError(
                    f"ROI lies entirely outside the data bounds "
                    f"({H}x{W} px). Draw the ROI on the BSE mosaic and retry."
                )
            if (cy0, cy1, cx0, cx1) != raw_bbox:
                print(
                    f"[roi] clipped to data bounds: "
                    f"y=[{y0}:{y1}]→[{cy0}:{cy1}], "
                    f"x=[{x0}:{x1}]→[{cx0}:{cx1}]",
                    flush=True,
                )
            y0, y1, x0, x1 = cy0, cy1, cx0, cx1

        bbox = (y0, y1, x0, x1)
        if str(shape_type).lower() in ('polygon', 'path', 'polygon_lasso', 'polygon pencil'):
            # _rasterize_polygon_mask grids the polygon onto the clipped
            # bbox; polygon points outside that bbox naturally fall out.
            mask = _rasterize_polygon_mask(verts, y0, y1, x0, x1)
            return bbox, mask
        else:
            return bbox, None

    # ---- manifest cache ----
    def _get_manifest_rows(self) -> List[dict]:
        """Return parsed manifest rows, parsing CSV only on first call."""
        if self._manifest_rows is None:
            self._manifest_rows = read_manifest(self.manifest)
        return self._manifest_rows

    # ---- actions ----
    def on_compute(self):
        # Prevent concurrent runs
        if self._eds_worker is not None and self._eds_worker.isRunning():
            return

        try:
            roi, roi_mask = self._get_roi_bbox_and_mask()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'ROI required', str(e))
            return

        elements = self._selected_elements()
        if not elements:
            QtWidgets.QMessageBox.warning(self, 'No elements selected', 'Select at least one element.')
            return

        width_eV   = float(self.win_spin.value())
        blend_mode = 'sum' if self.sum_check.isChecked() else 'mean'
        apply_subpix = self.subpix_check.isChecked()
        shift_method = str(self.shift_method_combo.currentData() or "bilinear")

        kwargs = dict(
            width_eV=width_eV, sideband_mult=1.5,
            apply_subpixel=apply_subpix,
            feather=True, tukey_alpha=0.08,
            blend_mode=blend_mode,
            roi_mask=roi_mask,
            _debug_return_stats=True,
            return_weight=True,
            _rows=self._get_manifest_rows(),
            shift_method=shift_method,
        )

        # Detect the EDS mosaic cache up front: it serves the ROI by memmap
        # slicing (fast, UI stays responsive) and — unlike the per-tile path —
        # can read at reduced resolution, which the memory guard below relies on.
        cache_dir = None
        try:
            import eds_cache as _edsc
            _cd = _edsc.resolve_cache_dir(Path(self.manifest).parent)
            if _edsc.is_valid_for(_cd, Path(self.manifest),
                                  required_elements=elements,
                                  required_width_eV=width_eV):
                cache_dir = _cd
        except Exception as exc:
            print(f"[eds] Cache probe failed ({exc}); will use tile read.")
            cache_dir = None

        # ── Memory pre-flight guard ───────────────────────────────────────
        # A full-res ROI keeps 3 bands × N elements × H × W float32 arrays in
        # RAM; a large ROI used to blow up with a raw MemoryError. Estimate the
        # footprint and, if it exceeds a safe budget, downsample (cache path) or
        # refuse with guidance (tile path) instead of crashing.
        roi_h, roi_w = roi[1] - roi[0], roi[3] - roi[2]
        n_el = len(elements)
        budget = self._eds_mem_budget_bytes()
        ds = _downsample_to_fit(roi_h, roi_w, n_el, budget)
        out_ds = 1
        if ds > 1:
            gb = 1024.0 ** 3
            est_full = _roi_map_bytes(roi_h, roi_w, n_el) / gb
            if cache_dir is not None:
                ho, wo = -(-roi_h // ds), -(-roi_w // ds)
                est_ds = _roi_map_bytes(ho, wo, n_el) / gb
                ans = QtWidgets.QMessageBox.question(
                    self, "Large ROI",
                    f"This ROI ({roi_w}×{roi_h} px, {n_el} element(s)) needs "
                    f"~{est_full:.1f} GB of RAM at full resolution, which "
                    f"exceeds the safe budget (~{budget / gb:.1f} GB).\n\n"
                    f"Compute at 1/{ds} resolution instead "
                    f"(~{est_ds:.1f} GB, {wo}×{ho} px)?\n\n"
                    "Yes = downsample (recommended)    No = cancel",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if ans != QtWidgets.QMessageBox.Yes:
                    return
                out_ds = ds
            else:
                QtWidgets.QMessageBox.warning(
                    self, "ROI too large",
                    f"This ROI ({roi_w}×{roi_h} px, {n_el} element(s)) needs "
                    f"~{est_full:.1f} GB of RAM at full resolution, beyond the "
                    f"safe budget (~{budget / gb:.1f} GB), and no EDS cache is "
                    "available to read it at reduced resolution.\n\n"
                    "Build the EDS cache (stitch.py --eds-preview all) to enable "
                    "automatic downsampling, or select a smaller ROI / fewer "
                    "elements."
                )
                return
        self.last_band_maps_ds = out_ds
        self.last_blend_mode = blend_mode

        # ── Cache fast path ───────────────────────────────────────────────
        if cache_dir is not None:
            pd_prog = QtWidgets.QProgressDialog(
                "Reading EDS ROI from cache…", "Cancel", 0, 0, self
            )
            pd_prog.setWindowTitle("EDS full-resolution maps (cache)")
            pd_prog.setWindowModality(QtCore.Qt.WindowModal)
            pd_prog.setMinimumDuration(0)
            pd_prog.setValue(0)

            worker = _EDSCacheWorker(cache_dir, roi, elements, blend_mode,
                                     roi_mask, out_ds=out_ds, parent=self)
            pd_prog.canceled.connect(worker.cancel)
            def _on_cache_progress(done, total, _pd=pd_prog):
                if _pd.maximum() != total:
                    _pd.setMaximum(total)
                _pd.setValue(done)
            worker.progress.connect(_on_cache_progress)
            worker.finished.connect(
                lambda maps, diag: self._on_eds_done(
                    maps, diag, roi, roi_mask, elements, pd_prog, out_ds)
            )
            worker.error.connect(lambda msg: self._on_eds_error(msg, pd_prog))

            self._eds_worker = worker
            self.btn_compute.setEnabled(False)
            ds_note = f" at 1/{out_ds} res" if out_ds > 1 else ""
            print(f"[eds] ROI served from cache ({len(elements)} elements, "
                  f"{roi_w}x{roi_h} px{ds_note}). Starting background read...")
            worker.start()
            return

        # ── Per-tile path (no cache; only reached when full-res fits) ─────
        from eds_map import _prefilter_rows_for_roi
        n_tiles = max(len(_prefilter_rows_for_roi(self._get_manifest_rows(), roi)), 1)

        pd_prog = QtWidgets.QProgressDialog(
            "Computing EDS maps…", "Cancel", 0, n_tiles, self
        )
        pd_prog.setWindowTitle("EDS full-resolution maps")
        pd_prog.setWindowModality(QtCore.Qt.WindowModal)
        pd_prog.setMinimumDuration(0)
        pd_prog.setValue(0)

        worker = _EDSWorker(self.manifest, roi, elements, kwargs, parent=self)
        pd_prog.canceled.connect(worker.cancel)
        worker.progress.connect(lambda done, total: pd_prog.setValue(done))
        worker.finished.connect(
            lambda maps, diag: self._on_eds_done(
                maps, diag, roi, roi_mask, elements, pd_prog, 1)
        )
        worker.error.connect(lambda msg: self._on_eds_error(msg, pd_prog))

        self._eds_worker = worker
        self.btn_compute.setEnabled(False)
        worker.start()

    @staticmethod
    def _eds_mem_budget_bytes() -> float:
        """Safe RAM budget for an in-memory ROI EDS map.

        ``THERMOD_EDS_ROI_MEM_BUDGET_GB`` overrides; otherwise ~80% of
        currently-available RAM (psutil), falling back to 8 GB. The estimate it
        is compared against (``_roi_map_bytes``) already separates persistent
        maps from a fixed transient term, so this can sit fairly high — raise
        the env var if you routinely compute ROIs larger than this allows.
        """
        import os as _os
        env = _os.environ.get("THERMOD_EDS_ROI_MEM_BUDGET_GB")
        if env:
            try:
                return float(env) * 1024.0 ** 3
            except ValueError:
                pass
        avail = _available_memory_bytes()
        if avail:
            return 0.8 * avail
        return 8.0 * 1024.0 ** 3

    def _on_eds_done(self, band_maps, diag, roi, roi_mask, elements, pd_prog, ds=1):
        """Slot called on main thread when _EDSWorker finishes successfully.
        pd_prog may be None when served from the cache fast path.

        ``ds`` is the resolution-decimation factor the maps were computed at
        (>1 when the ROI was downsampled to fit RAM); napari layers are placed
        with ``scale=(ds, ds)`` so they still overlay at mosaic resolution, and
        a full-res ``roi_mask`` is decimated to match the (smaller) arrays."""
        if pd_prog is not None:
            pd_prog.close()
        self._eds_worker = None
        self.btn_compute.setEnabled(True)
        ds = max(1, int(ds))
        # When downsampled, keep the full-res mask so cache-backed full-res
        # readers (composition / export) can mask at native resolution; the
        # working copy is decimated to match the (smaller) display arrays.
        self.last_roi_mask_fullres = (
            np.asarray(roi_mask) if (roi_mask is not None and ds > 1) else None
        )
        if roi_mask is not None and ds > 1:
            roi_mask = np.asarray(roi_mask)[::ds, ::ds]

        if (not band_maps) or all(v is None or (isinstance(v, dict) and all(arr.size == 0 for arr in v.values()))
                                  for v in band_maps.values()):
            msg = ["No ROI maps were produced."]
            if diag:
                msg.append("\nDiagnostics:")
                for k, v in diag.items():
                    msg.append(f"  {k}: {v}")
            QtWidgets.QMessageBox.warning(self, "Empty result", "\n".join(msg))
            return

        y0, y1, x0, x1 = roi
        net_counts: Dict[str, float] = {}
        # Signed line-aware nets (eds_background); the plan is judged on every
        # element in the ROI dict. Summed signed; the display uses the main window.
        from eds_background import net_count_maps as _bg_nets, describe_plan as _bg_describe, background_plan as _bg_plan
        _nets = _bg_nets(band_maps)
        print("[eds] " + _bg_describe(_bg_plan(list(band_maps))).replace("\n", "\n[eds] "), flush=True)

        for el, bands in band_maps.items():
            main  = bands['main']   # already float32 from roi_to_banded_maps
            net = _nets[el]

            disp = main.copy()
            v = disp[np.isfinite(disp)]
            if v.size:
                lo_p = np.percentile(v, 1.0)
                hi_p = np.percentile(v, 99.0)
                if hi_p > lo_p:
                    disp = np.clip((disp - lo_p) / (hi_p - lo_p), 0, 1)
            else:
                disp[:] = 0.0

            name = f"{el} ROI"
            scale = (ds, ds)   # render downsampled arrays at mosaic resolution
            has_poly_mask = roi_mask is not None
            if has_poly_mask:
                rgba = _rgba_float_from_gray_and_mask(disp, roi_mask, cmap_name='inferno')
                if name in self.viewer.layers and not getattr(self.viewer.layers[name], 'rgb', False):
                    self.viewer.layers.remove(self.viewer.layers[name])
                if name in self.viewer.layers:
                    lyr = self.viewer.layers[name]
                    lyr.data = rgba
                    lyr.scale = scale
                    lyr.translate = (y0, x0)
                    lyr.rgb = True
                    lyr.opacity = 1.0
                    lyr.blending = 'translucent_no_depth'
                    lyr.visible = True
                else:
                    self.viewer.add_image(
                        rgba, name=name, rgb=True,
                        blending='translucent_no_depth', opacity=1.0,
                        scale=scale, translate=(y0, x0), visible=True
                    )
            else:
                if name in self.viewer.layers and getattr(self.viewer.layers[name], 'rgb', False):
                    self.viewer.layers.remove(self.viewer.layers[name])
                if name in self.viewer.layers:
                    lyr = self.viewer.layers[name]
                    lyr.data = disp
                    lyr.scale = scale
                    lyr.translate = (y0, x0)
                    lyr.colormap = 'inferno'
                    lyr.opacity = 1.0
                    lyr.blending = 'opaque'
                    lyr.visible = True
                else:
                    self.viewer.add_image(
                        disp, name=name, colormap='inferno',
                        blending='opaque', opacity=1.0,
                        scale=scale, translate=(y0, x0), visible=True
                    )

            net_counts[el] = float(np.nansum(net))

        total = sum(net_counts.values()) or 1.0
        frac = {el: net_counts[el] / total for el in net_counts}
        self._show_composition({"net_counts": net_counts, "fraction": frac})

        self.last_band_maps = band_maps
        self.last_weight_map = diag.get("weight") if diag else None
        self.last_roi = roi
        self.last_roi_mask = roi_mask
        self.last_elements = elements
        self.last_composition = frac  # keep as fraction (0..1)
        # A fresh ROI compute invalidates an *ROI-scoped* calibration (its
        # k-factors were tied to that ROI's band maps). A *mosaic-wide*
        # calibration serves the whole mosaic and survives — only the
        # ROI bulk it never owned is cleared.
        if self._calibration_scope == "roi":
            self.eds_calibration = None
            self.eds_calibration_cv = None
            self._calibration_scope = None
            self._calibration_fingerprint = None
        self.last_calibrated_composition = None
        # ROI EDS maps now exist (and any prior calibration was cleared) — update
        # which controls are available.
        self._refresh_control_availability()

    def _on_eds_error(self, msg, pd_prog):
        """Slot called on main thread when _EDSWorker raises an exception."""
        pd_prog.close()
        self._eds_worker = None
        self.btn_compute.setEnabled(True)
        if "cancel" not in msg.lower():
            QtWidgets.QMessageBox.critical(self, "EDS computation failed", msg)

    def on_export(self):
        if not self.last_band_maps or not self.last_roi or not self.last_elements:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Compute an ROI first.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export directory")
        if not out_dir:
            return
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        y0, y1, x0, x1 = self.last_roi

        # If the display maps were downsampled to fit RAM (ds>1) but the
        # full-res cache is valid, export at FULL resolution by streaming each
        # element from the cache one at a time (bounded RAM). The combined
        # .npz (which would hold every element at once) is skipped in that mode
        # to avoid re-materialising the whole ROI; per-element TIFFs carry the
        # full-res float32 data.
        ds = self.last_band_maps_ds
        cache_dir = None
        _edsc = None
        # Full-res export needs the full-res polygon mask; if there was a
        # polygon ROI but only the decimated mask survives (e.g. after a session
        # reload), fall back to the in-RAM downsampled maps rather than export
        # full-res data with the wrong (or no) mask.
        _fullres_mask_ok = (self.last_roi_mask is None
                            or self.last_roi_mask_fullres is not None)
        if ds > 1 and _fullres_mask_ok:
            try:
                import eds_cache as _edsc_mod
                _cd = _edsc_mod.resolve_cache_dir(Path(self.manifest).parent)
                if _edsc_mod.is_valid_for(_cd, Path(self.manifest),
                                          required_elements=self.last_elements):
                    cache_dir = _cd
                    _edsc = _edsc_mod
            except Exception as exc:
                print(f"[export] full-res cache unavailable ({exc}); "
                      f"exporting downsampled maps.", flush=True)
                cache_dir = None
            # Full-res export of a downsampled ROI re-reads the whole ROI per
            # element from the cache — very slow/large on a big mosaic. Ask
            # rather than silently do it; default (No) exports the fast 1/N maps.
            if cache_dir is not None:
                ans = QtWidgets.QMessageBox.question(
                    self, "Export resolution",
                    f"These ROI maps were downsampled to 1/{ds} for display.\n\n"
                    "Export at FULL resolution from the cache? On a large mosaic "
                    "this re-reads the whole ROI for every element and can take "
                    "a long time and a lot of disk.\n\n"
                    "Yes = full resolution (slow)\n"
                    "No = downsampled 1/{} maps (fast)".format(ds),
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.No,
                )
                if ans == QtWidgets.QMessageBox.Cancel:
                    return
                if ans != QtWidgets.QMessageBox.Yes:
                    cache_dir = None   # export the in-memory downsampled maps

        export_mask = (self.last_roi_mask_fullres
                       if cache_dir is not None else self.last_roi_mask)
        if export_mask is not None:
            mask = export_mask.astype(np.uint8) * 255
            imageio.v2.imwrite(str(outp / "ROI_mask.png"), mask)
            try:
                import tifffile as tiff
                tiff.imwrite(outp / "ROI_mask.tif", export_mask.astype(np.uint8))
            except Exception:
                pass
            np.save(outp / "ROI_mask.npy", export_mask.astype(bool))

        try:
            import tifffile as tiff
        except Exception:
            tiff = None

        keep_npz = cache_dir is None
        npz_payload: Dict[str, np.ndarray] = {}

        def to_png(img: np.ndarray) -> np.ndarray:
            v = img[np.isfinite(img)]
            if v.size:
                lo, hi = np.percentile(v, [1, 99])
                if hi > lo:
                    img = np.clip((img - lo) / (hi - lo), 0, 1)
            return (img * 255).astype(np.uint8)

        def _bands_for(el):
            """(main, sb_lo, sb_hi) for *el* — full-res from cache when ds>1."""
            if cache_dir is not None:
                bm = _edsc.roi_from_cache(
                    cache_dir, self.last_roi, [el],
                    blend_mode=self.last_blend_mode, roi_mask=None,
                    progress=False,
                )[el]
                return bm['main'], bm['sb_lo'], bm['sb_hi']
            b = self.last_band_maps[el]
            return b['main'], b['sb_lo'], b['sb_hi']

        from eds_background import net_count_maps as _bg_nets
        for el in self.last_elements:
            main, sb_lo, sb_hi = _bands_for(el)   # float32
            _bm = {el: {"main": main, "sb_lo": sb_lo, "sb_hi": sb_hi}}
            if cache_dir is None:
                _bm = self.last_band_maps          # anchors available -> line-aware
            net = _bg_nets(_bm, elements=[el])[el]   # signed; PNG clips for display

            imageio.v2.imwrite(str(outp / f"{el}_ROI_main.png"), to_png(main))
            imageio.v2.imwrite(str(outp / f"{el}_ROI_sb_lo.png"), to_png(sb_lo))
            imageio.v2.imwrite(str(outp / f"{el}_ROI_sb_hi.png"), to_png(sb_hi))
            imageio.v2.imwrite(str(outp / f"{el}_ROI_net.png"),    to_png(net))

            if tiff:
                tiff.imwrite(outp / f"{el}_ROI_main.tif",  main, dtype=np.float32)
                tiff.imwrite(outp / f"{el}_ROI_sb_lo.tif", sb_lo, dtype=np.float32)
                tiff.imwrite(outp / f"{el}_ROI_sb_hi.tif", sb_hi, dtype=np.float32)
                tiff.imwrite(outp / f"{el}_ROI_net.tif",    net,  dtype=np.float32)

            if keep_npz:
                npz_payload[f"{el}_main"]  = main
                npz_payload[f"{el}_sb_lo"] = sb_lo
                npz_payload[f"{el}_sb_hi"] = sb_hi
                npz_payload[f"{el}_net"]   = net

        if keep_npz and npz_payload:
            np.savez_compressed(outp / "ROI_eds_maps.npz", **npz_payload)

        import csv
        with open(outp / "ROI_composition.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Element", "Net fraction"])
            for el in self.last_elements:
                val = float(self.last_composition.get(el, 0.0)) if self.last_composition else 0.0
                w.writerow([el, f"{val:.6f}"])

        extra = ("\n\nExported at FULL resolution from the cache; the combined "
                 ".npz was skipped (per-element TIFFs hold the full-res data)."
                 if cache_dir is not None else "")
        QtWidgets.QMessageBox.information(
            self, "Export complete",
            f"Saved ROI maps and composition to:\n{outp}{extra}")

    def on_export_spectrum_msa(self):
        """Export a fast summed spectrum for the current ROI as EMSA/MAS .msa (Y + calibration)."""
        try:
            roi, roi_mask = self._get_roi_bbox_and_mask()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'ROI required', str(e))
            return

        dlg = QtWidgets.QFileDialog(self)
        dlg.setWindowTitle("Save ROI summed spectrum (MSA)")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilters(["EMSA files (*.msa)", "All files (*)"])
        dlg.setDefaultSuffix("msa")
        if not dlg.exec_():
            return
        out_path = Path(dlg.selectedFiles()[0])

        try:
            spec, (E0, dE, nE, units) = roi_to_summed_spectrum_streamed(
                self.manifest,
                roi,
                roi_mask=roi_mask,
                energy_chunk=16,
                progress=True,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not compute ROI summed spectrum:\n{e}")
            return

        if units and units.lower() == 'ev':
            e0_keV = float(E0) / 1000.0
            dE_keV = float(dE) / 1000.0
        else:
            e0_keV = float(E0)
            dE_keV = float(dE)

        try:
            _write_emsa_msa(out_path, np.asarray(spec, dtype=np.float64), e0_keV, dE_keV)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", f"Could not write MSA:\n{e}")
            return

        QtWidgets.QMessageBox.information(self, "Spectrum exported", f"Saved MSA to:\n{out_path}")

    def _get_tile_bboxes(self) -> List[Tuple[Tuple[int, int, int, int], dict]]:
        """Cached list of (mosaic_bbox, manifest_row) for every tile."""
        if self._tile_bboxes_cache is None:
            rows = self._get_manifest_rows()
            tiles: List[Tuple[Tuple[int, int, int, int], dict]] = []
            for r in rows:
                bb = _row_bbox_on_canvas(r)
                if bb is not None:
                    tiles.append((bb, r))
            self._tile_bboxes_cache = tiles
        return self._tile_bboxes_cache

    def _cache_data_type(self) -> str:
        """Return the EDS cache's ``data_type`` ('bcf_net_counts' /
        'oxford_wt_pct' / …). Falls back to 'bcf_net_counts' when the
        cache or its manifest is missing.
        """
        try:
            import eds_cache as _edsc
            cache_dir = _edsc.resolve_cache_dir(self.manifest.parent)
            data = _edsc.load_manifest(cache_dir)
            if data:
                # Coerce a missing/null data_type to the Bruker default — older
                # caches stored it as JSON null, which would otherwise surface as
                # the string "None" and mis-drive labels / the calibration overlay.
                return str(data.get("data_type") or "bcf_net_counts")
        except Exception:
            pass
        return "bcf_net_counts"

    def _is_oxford_cache(self) -> bool:
        """True for any montaged-Oxford cache (Wt% or uncalibrated counts).

        Both are pre-montaged with no per-tile SPX, so the SPX radio is
        disabled and the source labels switch to an Oxford string.
        """
        return self._cache_data_type() in ("oxford_wt_pct", "oxford_counts")

    def _effective_data_type(self) -> str:
        """Data type seen by *labels / quantified-ness*, not by cache validity.

        When an SPX-anchored calibration is active for the active ROI, Bruker
        net-counts are presented as quantified ``eds_calibrated_wt_pct`` so the
        units read "wt%" and ``mineral_id`` runs structural recalc. The on-disk
        cache still holds net counts, so ``_cache_data_type()`` (which drives
        cache-fingerprint validity) is left untouched.
        """
        if self.eds_calibration is not None and self._cache_data_type() == "bcf_net_counts":
            return "eds_calibrated_wt_pct"
        return self._cache_data_type()

    @staticmethod
    def _oxford_source_label(data_type: str, short: bool = False) -> str:
        """Human label for a montaged-Oxford source given its ``data_type``."""
        if data_type == "oxford_counts":
            return "Oxford counts" if short else "Oxford EDS counts (montaged map)"
        return "Oxford Wt%" if short else "Oxford Wt% (montaged map)"

    def _has_spx_tiles(self) -> bool:
        """True for a Bruker mosaic that carries per-tile SPX quantification.

        Oxford montages are pre-quantified with no per-tile structure, so SPX-
        anchored features (SPX composition, k-factor calibration) don't apply.
        Mirrors the ``has_spx`` proxy used by the composition-vs-scale tool.
        """
        return bool(self._get_tile_bboxes()) and not self._is_oxford_cache()

    def _set_widget_enabled(self, widget, enabled: bool, reason: str = ""):
        """Enable/disable a control and, when disabled, append an explanatory
        line to its tooltip so a greyed-out option says *why* it is unavailable.
        The control's original tooltip is captured once and restored on enable."""
        if not hasattr(self, "_base_tooltips"):
            self._base_tooltips = {}
        key = id(widget)
        if key not in self._base_tooltips:
            self._base_tooltips[key] = widget.toolTip()
        base = self._base_tooltips[key]
        widget.setEnabled(bool(enabled))
        if enabled:
            widget.setToolTip(base)
        else:
            note = f"Unavailable: {reason}" if reason else "Unavailable for this dataset."
            widget.setToolTip(f"{base}\n\n{note}" if base else note)

    def _eds_cache_status(self) -> dict:
        """Return ``eds_cache.status_for`` for this dataset's cache (or a
        minimal ``{'exists': False}`` if the probe can't run)."""
        try:
            import eds_cache as _edsc
            cd = _edsc.resolve_cache_dir(Path(self.manifest).parent)
            return _edsc.status_for(cd, Path(self.manifest))
        except Exception:
            return {"exists": False}

    def _update_cache_status_label(self):
        """Refresh the panel's EDS-cache status line so the user can see whether
        ROI reads use the fast mosaic cache, and why not when they don't."""
        if not hasattr(self, "cache_status_label"):
            return
        st = self._eds_cache_status()
        lbl = self.cache_status_label
        if not st.get("exists"):
            lbl.setText("EDS cache: none — ROIs are read from tiles (slower).")
            lbl.setStyleSheet("color: #b08000;")  # amber
            lbl.setToolTip(
                "No EDS mosaic cache for this dataset. Build one to make ROI "
                "reads fast:\n  stitch.py --dir <dir> --eds-preview all\n"
                "(or use the Tools ▸ Build/refresh EDS cache menu).")
            return
        n_el = len(st.get("elements") or [])
        ds = st.get("ds") or 1
        dt = st.get("data_type") or "bcf_net_counts"
        if dt == "bcf_net_counts":
            kind = f"Bruker net-counts, {st.get('width_eV')} eV window"
        else:
            kind = self._oxford_source_label(dt, short=True)
        if st.get("valid"):
            lbl.setText(f"EDS cache: ready, {n_el} element(s), ds={ds} ({kind}).")
            lbl.setStyleSheet("color: #2e7d32;")  # green
            lbl.setToolTip("ROI reads use the cache (fast memmap slicing).")
        else:
            lbl.setText("EDS cache: ⚠ present but STALE — the stitch manifest "
                        "changed since it was built; rebuild to use the fast path.")
            lbl.setStyleSheet("color: #c62828;")  # red
            lbl.setToolTip(
                "Cache fingerprint no longer matches stitch_manifest.csv. "
                "Rebuild it (stitch.py --eds-preview all, or Tools ▸ Build/"
                "refresh EDS cache); ROIs are read from tiles until then.")

    def _refresh_control_availability(self):
        """Grey out panel controls that don't apply to the loaded dataset/state.

        Two kinds of gating:
          * dataset-type (permanent): SPX composition, SPX k-factor calibration,
            compare-compositions and summed-spectrum export are Bruker-only —
            Oxford montages are pre-quantified wt% with no spectrum cube; the
            EBSD-domains button needs an imported Oxford h5oina phase map.
          * computed-state (transient): exports / calibration / compare need the
            ROI EDS maps to have been computed; phase-composition needs a
            finished full-resolution segmentation.

        Idempotent; called at startup and whenever the relevant state changes.
        ``btn_compute`` / ``btn_comp_scale`` / ``btn_thermod`` etc. apply to every
        dataset (thermod also takes a hand-typed composition) and are left alone.
        """
        if not hasattr(self, "btn_spx_comp"):
            return  # called before the buttons were built
        self._update_cache_status_label()
        is_oxford = self._is_oxford_cache()
        has_spx = self._has_spx_tiles()
        has_maps = bool(self.last_band_maps)
        has_ebsd = self.ebsd_phase_map is not None
        has_phase_comp = bool(self.last_phase_composition)

        self._set_widget_enabled(
            self.btn_export, has_maps,
            "compute the ROI's full-resolution EDS maps first.")
        self._set_widget_enabled(
            self.btn_zoning, has_maps,
            "compute the ROI's full-resolution EDS maps first.")
        self._set_widget_enabled(
            self.btn_export_spec, (not is_oxford),
            "Oxford montages have no spectrum cube to sum (Bruker only).")
        self._set_widget_enabled(
            self.btn_spx_comp, has_spx,
            "needs per-tile SPX quantification (Bruker datasets only).")
        if is_oxford:
            self._set_widget_enabled(
                self.btn_calib_eds, False,
                "Oxford data is already quantified wt% — no SPX calibration needed.")
            self._set_widget_enabled(
                self.btn_compare_comp, False,
                "only one composition source for an Oxford montage — nothing to compare.")
        else:
            # Mosaic calibration needs only per-tile SPX (no ROI maps); the
            # options dialog gates the "current ROI only" choice on computed maps.
            self._set_widget_enabled(
                self.btn_calib_eds, has_spx,
                "needs per-tile SPX (Bruker datasets only).")
            self._set_widget_enabled(
                self.btn_compare_comp, has_maps,
                "compute the ROI's full-resolution EDS maps first.")
        self._set_widget_enabled(
            self.btn_phase_comp, has_phase_comp,
            "available after a full-resolution phase segmentation.")
        self._set_widget_enabled(
            self.btn_comp_scale_reopen, bool(self.last_comp_scale),
            "run 'Composition vs scale …' first; then the result re-opens here.")
        self._set_widget_enabled(
            self.btn_phase_reopen, self.last_phase_map is not None,
            "run 'Segment phases …' or load a session with a segmentation first.")
        self._set_widget_enabled(
            self.btn_ebsd, has_ebsd and has_maps,
            ("compute the ROI's EDS maps first."
             if has_ebsd else "no EBSD phase map in this dataset (Oxford h5oina only)."))
        self._set_widget_enabled(
            self.btn_ebsd_conf, has_ebsd and self.last_phase_map is not None,
            ("segment phases (or load a session with a segmentation) first."
             if has_ebsd else "no EBSD phase map in this dataset (Oxford h5oina only)."))

        # EDS-extraction parameters only affect the Bruker per-tile spectrum
        # integration. Oxford montages are pre-quantified (read straight from the
        # cache), so these settings are no-ops there.
        for w in (self.win_spin, self.shift_method_combo,
                  self.sum_check, self.subpix_check):
            self._set_widget_enabled(
                w, (not is_oxford),
                "applies to Bruker EDS extraction; Oxford data is already quantified wt%.")

    def _spx_for_tile_row(self, row: dict) -> Tuple[Dict[str, float], float, List[str]]:
        """Return (comp, weight, notes) for a manifest row, with per-row caching.

        Returns ({}, 0.0, []) if the SPX is missing or unparseable. Result is
        memoised in ``self._spx_parse_cache`` so the same tile is never
        re-read across many sub-window calls.
        """
        rel = row.get("file") or row.get("path") or ""
        base_dir = self.manifest.parent.resolve()
        bcf_path = (base_dir / rel).resolve()
        if not bcf_path.exists():
            bcf_path = (base_dir / Path(rel).name).resolve()
        if not bcf_path.exists():
            return {}, 0.0, []
        cached = self._spx_parse_cache.get(bcf_path)
        if cached is not None:
            return cached
        spx = _match_spx_for_bcf(bcf_path)
        if spx is None or not spx.exists():
            self._spx_parse_cache[bcf_path] = ({}, 0.0, [])
            return {}, 0.0, []
        comp, weight, notes = _parse_spx_quant(spx)
        self._spx_parse_cache[bcf_path] = (comp, weight, notes)
        return comp, weight, notes

    def _spx_composition_for_bbox(self, bbox: Tuple[int, int, int, int],
                                   min_overlap: float = 0.95
                                   ) -> Optional[Dict[str, float]]:
        """SPX-quantified element wt% (sums to 100) over a rectangular bbox.

        Returns None if no SPX-bearing tile overlaps ``bbox`` by at least
        ``min_overlap`` of the tile area, or if no parseable quantification
        is found.

        Aggregation matches ``on_spx_composition``: weighted by NetIntens.
        Per-tile SPX parses are cached, so repeated calls (e.g. one per
        quadtree window) only pay the disk cost once per tile.
        """
        tiles = self._get_tile_bboxes()
        if not tiles:
            return None

        sum_weights = 0.0
        sum_wt: Dict[str, float] = {}
        num_used = 0
        for bb, r in tiles:
            A_tile = (bb[1] - bb[0]) * (bb[3] - bb[2])
            if A_tile <= 0:
                continue
            A_ov = _overlap_area(bb, bbox)
            if A_ov / A_tile < min_overlap:
                continue
            comp, weight, _notes = self._spx_for_tile_row(r)
            if not comp:
                continue
            num_used += 1
            sum_weights += weight
            for el, wt_pct in comp.items():
                sum_wt[el] = sum_wt.get(el, 0.0) + wt_pct * weight
        if num_used == 0:
            return None
        if sum_weights > 0:
            comp_roi = {el: v / sum_weights for el, v in sum_wt.items()}
        else:
            comp_roi = dict(sum_wt)
        s = sum(comp_roi.values())
        if s > 0:
            return {k: 100.0 * v / s for k, v in comp_roi.items()}
        return comp_roi

    def on_spx_composition(self):
        """Compute ROI composition from quantified SPX for tiles fully inside ROI (≥95% overlap)."""
        rows = self._get_manifest_rows()
        if not rows:
            QtWidgets.QMessageBox.warning(self, "Manifest", "Manifest is empty or unreadable.")
            return

        try:
            roi, _ = self._get_roi_bbox_and_mask()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'ROI required', str(e))
            return

        if self.debug_manifest:
            tiles = self._get_tile_bboxes()
            H = int(max((bb[1] for bb, _ in tiles), default=0))
            W = int(max((bb[3] for bb, _ in tiles), default=0))
            print(f"\n[debug-manifest] tiles={len(tiles)} | Canvas approx HxW: {H} x {W}")

        comp_roi = self._spx_composition_for_bbox(roi, min_overlap=0.95)
        if not comp_roi:
            QtWidgets.QMessageBox.warning(
                self, "SPX composition",
                "No tiles fully inside ROI (≥95% overlap), or none of the "
                "overlapping tiles had usable SPX quantification."
            )
            return

        if self.debug_spx:
            top = sorted(comp_roi.items(), key=lambda kv: -kv[1])[:10]
            print(f"[SPX] ROI: {roi} | top elements:", top)

        # Store for Perple_X export and show dialog
        self.last_spx_composition = dict(comp_roi)

        rows_tbl = sorted(comp_roi.items(), key=lambda kv: -kv[1])
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('ROI composition from SPX (wt%)')
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget()
        table.setRowCount(len(rows_tbl)); table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Element', 'wt%'])
        for i, (el, wt) in enumerate(rows_tbl):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(el))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{wt:.2f}"))
        layout.addWidget(table)
        btn = QtWidgets.QPushButton('OK'); btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.resize(320, 360)
        dlg.exec_()

    # ---- SPX-anchored EDS calibration ---------------------------------------

    def _preferred_roi_composition(self) -> Tuple[Optional[Dict[str, float]], str]:
        """ROI element wt% for modelling, preferring SPX > calibrated-EDS > raw.

        Returns ``(comp_el_wt, source_label)`` or ``(None, "")`` when nothing is
        available. SPX is true matrix-corrected quantification (best when it
        covers the ROI); a calibrated EDS composition beats raw net-count
        fractions (which are intensity-, not mass-proportional).
        """
        if isinstance(self.last_spx_composition, dict) and self.last_spx_composition:
            return dict(self.last_spx_composition), "SPX quantified"
        if isinstance(self.last_calibrated_composition, dict) and self.last_calibrated_composition:
            return dict(self.last_calibrated_composition), "EDS calibrated (SPX k-factors)"
        if isinstance(self.last_composition, dict) and self.last_composition:
            return ({k: 100.0 * float(v) for k, v in self.last_composition.items()},
                    "net-count fractions (approx.)")
        return None, ""

    def _collect_roi_calibration_tiles(
        self,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """Gather (spx_by_tile, counts_by_tile) for tiles >=95% inside the ROI.

        Each pair is a tile's quantified SPX wt% and its integrated EDS
        net-counts over the same (ROI-local, ``//ds``-decimated) bbox — the
        regression input for ``eds_quantify.derive_k_factors``. Returns
        ``([], [])`` when prerequisites are missing (no band maps/ROI, Oxford
        cache, or no SPX-bearing whole tiles). Shared by ``on_calibrate_eds``
        and ``on_compare_compositions``.
        """
        if not self.last_band_maps or not self.last_roi or self._is_oxford_cache():
            return [], []
        tiles = self._get_tile_bboxes()
        if not tiles:
            return [], []
        y0, y1, x0, x1 = self.last_roi
        ds = max(1, int(self.last_band_maps_ds))
        counts_by_tile: List[Dict[str, float]] = []
        spx_by_tile: List[Dict[str, float]] = []
        for bb, row in tiles:
            A_tile = (bb[1] - bb[0]) * (bb[3] - bb[2])
            if A_tile <= 0:
                continue
            if _overlap_area(bb, self.last_roi) / A_tile < 0.95:
                continue
            comp, _weight, _notes = self._spx_for_tile_row(row)
            if not comp:
                continue
            # tile bbox (mosaic) -> ROI-local -> decimated to match band_maps;
            # _net_count_sums clips the sub-bbox to the map shape.
            sub = ((bb[0] - y0) // ds, (bb[1] - y0) // ds,
                   (bb[2] - x0) // ds, (bb[3] - x0) // ds)
            sums, _npix = _net_count_sums(self.last_band_maps, sub_bbox=sub)
            if not sums:
                continue
            counts_by_tile.append(sums)
            spx_by_tile.append(comp)
        return spx_by_tile, counts_by_tile

    def _collect_mosaic_calibration_tiles(self, progress_cb=None, cancel_flag=None,
                                          subset_n=None, subset_seed=0):
        """Gather (spx, counts, weights, valid_fracs) for SPX-bearing tiles.

        One set of k-factors serves the whole mosaic (they are specific to the
        sample and session, not to the instrument), so the headline calibration
        fits them from all SPX tiles across the mosaic — no ROI. ``subset_n``
        reads from a random sample of that many tiles instead (seeded by
        ``subset_seed``) — a few hundred tiles already span the compositional
        range and avoid scanning a huge mosaic. Net-count sums come from the EDS
        cache: memmaps opened once via ``CacheWindowReader`` on the ``ds=1`` fast
        path, else per-tile ``roi_from_cache``. ``weights`` are the per-tile SPX
        intensities and ``valid_fracs`` the tile coverage fraction (cache fast
        path only) — both feed the quality filter. Returns four
        positionally-paired lists; ``([],[],[],[])`` when an EDS cache is absent
        (needed to read net sums cheaply) or for Oxford data.
        """
        if self._is_oxford_cache():
            return [], [], [], []
        tiles = self._get_tile_bboxes()
        if not tiles:
            return [], [], [], []
        import eds_cache as _edsc
        st = self._eds_cache_status()
        if not st.get("exists"):
            return [], [], [], []
        elements = list(st.get("elements") or self.last_elements or [])
        if not elements:
            return [], [], [], []
        cache_dir = _edsc.resolve_cache_dir(Path(self.manifest).parent)
        hw = self._data_bounds()
        if hw is None:
            return [], [], [], []
        H, W = hw
        ds = int(st.get("ds") or 1)
        blend = getattr(self, "last_blend_mode", None) or "mean"

        # Whole-mosaic calibration only needs a per-tile *sum*, not full
        # resolution, so each tile is subsampled to ~``target`` pixels (the sum is
        # scaled back up). Reading every pixel of a billion-pixel cache across
        # thousands of tiles would otherwise saturate the disk and the page cache.
        import os
        target = max(256, int(os.environ.get("THERMOD_CALIB_SAMPLES_PER_TILE", "2000")))

        reader = None
        if st.get("valid") and ds == 1:
            try:
                reader = _edsc.CacheWindowReader(cache_dir, elements, blend_mode=blend)
            except Exception:
                reader = None

        # Optional random subset of tiles (read fewer → faster). Sorted to keep
        # spatial/disk order; a few hundred tiles already constrain the k-factors.
        if subset_n and int(subset_n) < len(tiles):
            rng = np.random.default_rng(int(subset_seed))
            idx = sorted(rng.choice(len(tiles), size=int(subset_n),
                                    replace=False).tolist())
            tiles = [tiles[j] for j in idx]

        spx_l, cnt_l, wt_l, vf_l = [], [], [], []
        n = len(tiles)
        path = "cache fast path" if reader is not None else "per-tile cache read"
        print(f"[calib] collecting SPX + EDS over {n} tiles via {path}; "
              f"subsampling to ~{target} px/tile.", flush=True)
        report_every = max(1, n // 50)
        for i, (bb, row) in enumerate(tiles):
            if cancel_flag is not None and cancel_flag():
                print(f"[calib] cancelled at tile {i}/{n}.", flush=True)
                break
            if progress_cb is not None:
                progress_cb(i, n)
            if (i + 1) % report_every == 0 or (i + 1) == n:
                print(f"[calib]   {i + 1}/{n} tiles scanned, {len(cnt_l)} usable…",
                      flush=True)
            comp, weight, _notes = self._spx_for_tile_row(row)
            if not comp:
                continue
            y0 = max(0, min(int(bb[0]), H)); y1 = max(0, min(int(bb[1]), H))
            x0 = max(0, min(int(bb[2]), W)); x1 = max(0, min(int(bb[3]), W))
            if y1 <= y0 or x1 <= x0:
                continue
            cbb = (y0, y1, x0, x1)
            area = (y1 - y0) * (x1 - x0)
            step = 1
            if area > target:
                step = max(1, min(int(round((area / target) ** 0.5)),
                                  y1 - y0, x1 - x0))
            vf = 1.0
            if reader is not None:
                sums = reader.net_sums(cbb, step=step)
                vf = reader.coverage_frac(cbb, step=step)
            else:
                try:
                    if ds > 1:
                        bm = _edsc.roi_from_cache_ds(cache_dir, cbb, elements,
                                                     blend_mode=blend, progress=False,
                                                     out_ds=step)
                    else:
                        bm = _edsc.roi_from_cache(cache_dir, cbb, elements,
                                                  blend_mode=blend, progress=False,
                                                  out_ds=step)
                except Exception:
                    continue
                sums, _npix = _net_count_sums(bm)
                if step > 1 and sums and _npix > 0:
                    sc = float(area) / float(_npix)   # scale subsample -> full sum
                    sums = {el: v * sc for el, v in sums.items()}
            if not sums:
                continue
            spx_l.append(comp); cnt_l.append(sums)
            wt_l.append(float(weight)); vf_l.append(float(vf))
        if progress_cb is not None:
            progress_cb(n, n)
        print(f"[calib] done: {len(cnt_l)} usable SPX tiles of {n}.", flush=True)
        return spx_l, cnt_l, wt_l, vf_l

    def _run_mosaic_collection(self, subset_n=None, subset_seed=0):
        """Run the whole-mosaic tile collection with a modal progress dialog.

        Synchronous with a ``processEvents`` pump: the read is memmap slicing
        (fast path) so it returns quickly; the modal dialog blocks other input
        and supports Cancel. ``subset_n`` limits the read to a random sample of
        that many tiles. Returns ``(spx, counts, weights, valid_fracs)`` or
        ``(None, None, None, None)`` if cancelled.
        """
        total = len(self._get_tile_bboxes())
        n = max(min(subset_n, total) if subset_n else total, 1)
        msg = (f"Reading SPX + EDS net-counts for {n} tiles…" if subset_n
               else "Reading SPX + EDS net-counts for all tiles…")
        prog = QtWidgets.QProgressDialog(msg, "Cancel", 0, n, self)
        prog.setWindowTitle("Calibrate EDS (mosaic)")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0); prog.setValue(0)

        def _pcb(done, total):
            prog.setMaximum(max(total, 1)); prog.setValue(done)
            QtWidgets.QApplication.processEvents()

        try:
            res = self._collect_mosaic_calibration_tiles(
                progress_cb=_pcb, cancel_flag=prog.wasCanceled,
                subset_n=subset_n, subset_seed=subset_seed)
        finally:
            cancelled = prog.wasCanceled()
            prog.close()
        if cancelled:
            return None, None, None, None
        return res

    def _calibration_options_dialog(self):
        """Modal options for SPX k-factor calibration. Returns a dict or None."""
        has_roi = bool(self.last_band_maps and self.last_roi)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Calibrate EDS — options")
        lay = QtWidgets.QVBoxLayout(dlg)

        lay.addWidget(QtWidgets.QLabel("<b>Tiles to calibrate from</b>"))
        rb_all = QtWidgets.QRadioButton("All SPX tiles in the mosaic (recommended)")
        rb_roi = QtWidgets.QRadioButton("Current ROI only (local regime)")
        rb_all.setChecked(True)
        rb_roi.setEnabled(has_roi)
        if not has_roi:
            rb_roi.setToolTip("Compute the ROI's full-resolution EDS maps first.")
        lay.addWidget(rb_all); lay.addWidget(rb_roi)

        srow = QtWidgets.QHBoxLayout()
        cb_subset = QtWidgets.QCheckBox("Use a random subset of")
        cb_subset.setToolTip(
            "Read and fit from a random sample of this many tiles instead of the "
            "whole mosaic — much faster, and a few hundred tiles already span the "
            "compositional range that constrains the k-factors.")
        sp_subset = QtWidgets.QSpinBox(); sp_subset.setRange(10, 1000000)
        sp_subset.setValue(500); sp_subset.setSingleStep(50)
        sp_sseed = QtWidgets.QSpinBox(); sp_sseed.setRange(0, 99999); sp_sseed.setValue(0)
        srow.addWidget(cb_subset); srow.addWidget(sp_subset)
        srow.addWidget(QtWidgets.QLabel("tiles (seed"))
        srow.addWidget(sp_sseed); srow.addWidget(QtWidgets.QLabel(")"))
        srow.addStretch()
        lay.addLayout(srow)

        lay.addWidget(_hline())
        lay.addWidget(QtWidgets.QLabel("<b>Validation</b>"))
        rb_hold = QtWidgets.QRadioButton("Repeated hold-out (leave a fraction out)")
        rb_loo = QtWidgets.QRadioButton("Leave-one-out (exhaustive, fast)")
        rb_hold.setChecked(True)
        lay.addWidget(rb_hold)
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("    fraction:"))
        sp_frac = QtWidgets.QDoubleSpinBox(); sp_frac.setRange(0.05, 0.5)
        sp_frac.setSingleStep(0.05); sp_frac.setValue(0.2); sp_frac.setDecimals(2)
        frow.addWidget(sp_frac)
        frow.addWidget(QtWidgets.QLabel("repeats:"))
        sp_rep = QtWidgets.QSpinBox(); sp_rep.setRange(2, 500); sp_rep.setValue(50)
        frow.addWidget(sp_rep); frow.addStretch()
        lay.addLayout(frow)
        lay.addWidget(rb_loo)

        lay.addWidget(_hline())
        cb_filter = QtWidgets.QCheckBox(
            "Exclude low-quality tiles (near-empty / holes)")
        cb_filter.setChecked(True)
        lay.addWidget(cb_filter)
        qrow = QtWidgets.QHBoxLayout()
        qrow.addWidget(QtWidgets.QLabel("    min coverage fraction:"))
        sp_vf = QtWidgets.QDoubleSpinBox(); sp_vf.setRange(0.0, 1.0)
        sp_vf.setSingleStep(0.05); sp_vf.setValue(0.5); sp_vf.setDecimals(2)
        qrow.addWidget(sp_vf); qrow.addStretch()
        lay.addLayout(qrow)

        def _sync():
            on = rb_hold.isChecked()
            sp_frac.setEnabled(on); sp_rep.setEnabled(on)
            sp_vf.setEnabled(cb_filter.isChecked())
            allsrc = rb_all.isChecked()
            cb_subset.setEnabled(allsrc)
            sp_subset.setEnabled(allsrc and cb_subset.isChecked())
            sp_sseed.setEnabled(allsrc and cb_subset.isChecked())
        rb_hold.toggled.connect(lambda _: _sync())
        cb_filter.toggled.connect(lambda _: _sync())
        rb_all.toggled.connect(lambda _: _sync())
        cb_subset.toggled.connect(lambda _: _sync())
        _sync()

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return {
            "source": "roi" if rb_roi.isChecked() else "mosaic",
            "method": "holdout" if rb_hold.isChecked() else "loo",
            "holdout_frac": float(sp_frac.value()),
            "n_repeats": int(sp_rep.value()),
            "filter_on": cb_filter.isChecked(),
            "min_valid_frac": float(sp_vf.value()),
            "subset_n": (int(sp_subset.value())
                         if (rb_all.isChecked() and cb_subset.isChecked()) else None),
            "subset_seed": int(sp_sseed.value()),
        }

    def on_calibrate_eds(self):
        """Derive per-element SPX k-factors and calibrate the EDS maps.

        One set of k-factors serves the whole mosaic (they are specific to the
        sample and session, not to the instrument), so the default fits them from
        *all* SPX tiles across the mosaic (no ROI required); a "current ROI only"
        local-regime option remains. A conservative quality filter drops
        near-empty / holey tiles from the fit, and the hold-out validation
        (repeated leave-a-proportion-out by default, or leave-one-out) is shown
        and exportable. Stores the calibration as a runtime overlay so the
        uncalibrated net-count maps read as quasi-quantified wt% downstream.
        """
        import eds_quantify as eq

        if self._is_oxford_cache():
            QtWidgets.QMessageBox.warning(
                self, "Calibrate EDS",
                "SPX k-factor calibration needs per-tile Bruker SPX quantification; "
                "this is a montaged-Oxford dataset (no SPX).")
            return
        if not self._get_tile_bboxes():
            QtWidgets.QMessageBox.warning(
                self, "Calibrate EDS", "No tile manifest available for SPX.")
            return

        opts = self._calibration_options_dialog()
        if opts is None:
            return

        if opts["source"] == "roi":
            if not self.last_band_maps or not self.last_roi:
                QtWidgets.QMessageBox.warning(
                    self, "Calibrate EDS",
                    "Compute the FULL-RES EDS maps for an ROI first, or choose "
                    "'All SPX tiles'.")
                return
            spx_by_tile, counts_by_tile = self._collect_roi_calibration_tiles()
            weights = valid_fracs = None
            scope = "roi"
        else:
            spx_by_tile, counts_by_tile, weights, valid_fracs = \
                self._run_mosaic_collection(subset_n=opts.get("subset_n"),
                                            subset_seed=opts.get("subset_seed", 0))
            if spx_by_tile is None:          # cancelled
                return
            scope = "mosaic"

        if not counts_by_tile:
            QtWidgets.QMessageBox.warning(
                self, "Calibrate EDS",
                "No SPX-bearing tile with usable net-counts was found.\n\n"
                "The 'all SPX tiles' option reads net-counts from the EDS cache — "
                "build one (stitch.py --eds-preview all) if it is missing; the "
                "'ROI only' option needs an SPX tile inside the computed ROI.")
            return

        n_candidates = len(counts_by_tile)
        dropped: list = []
        if opts["filter_on"]:
            spx_by_tile, counts_by_tile, dropped = eq.filter_calibration_tiles(
                spx_by_tile, counts_by_tile, weights=weights,
                valid_fracs=valid_fracs, min_valid_frac=opts["min_valid_frac"])
            if not counts_by_tile:
                QtWidgets.QMessageBox.warning(
                    self, "Calibrate EDS",
                    f"All {n_candidates} candidate tiles were excluded by the "
                    "quality filter; lower the minimum coverage fraction.")
                return

        _drop = eq.ANION_ELEMENTS | self._coating_elements()
        cal = eq.derive_k_factors(spx_by_tile, counts_by_tile, drop=_drop,
                                  background=_bg_scheme())
        try:
            from eds_background import background_plan as _bg_plan, describe_plan as _bg_describe
            _st = self._eds_cache_status()
            _els = list(_st.get("elements") or self.last_elements or [])
            if _els:
                print("[calib] " + _bg_describe(_bg_plan(
                    _els, width_eV=float(_st.get("width_eV") or 150.0))).replace("\n", "\n[calib] "),
                    flush=True)
        except Exception:
            pass
        if not cal.k_factors:
            QtWidgets.QMessageBox.warning(
                self, "Calibrate EDS",
                "Could not anchor any element (no element had both SPX wt% and "
                "EDS net-counts in the same tile).")
            return
        self.eds_calibration = cal

        cv = eq.cross_validate_k_factors(
            spx_by_tile, counts_by_tile, drop=_drop,
            method=opts["method"], holdout_frac=opts["holdout_frac"],
            n_repeats=opts["n_repeats"])
        self.eds_calibration_cv = cv
        self._calibration_scope = scope
        self._calibration_fingerprint = self._eds_cache_status().get("fingerprint")

        if scope == "roi":
            # ROI bulk from the calibrated net fractions (honour polygon).
            frac, _ = _net_count_fractions(self.last_band_maps,
                                           sub_mask=self.last_roi_mask)
            self.last_calibrated_composition = eq.apply_calibration_fractions(
                frac, cal.k())
        # mosaic scope: no single whole-mosaic ROI bulk — leave it to fill
        # lazily when the user computes an ROI.

        self._refresh_control_availability()
        self._show_calibration_dialog(cal, cv, n_dropped=len(dropped),
                                      n_candidates=n_candidates)

    def _export_calibration_cv_csv(self, cv):
        """Write the hold-out validation pairs (long format) for external plotting."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export calibration validation CSV",
            "calibration_validation.csv", "CSV files (*.csv)")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["tile", "fold", "element", "predicted_wt_pct",
                        "measured_wt_pct", "residual_wt_pct"])
            for rec in cv.records:
                fold = rec.get("fold", 0)
                for el in sorted(rec["predicted"]):
                    p = rec["predicted"][el]
                    m = rec["measured"].get(el)
                    if m is None:
                        continue
                    w.writerow([rec["tile"], fold, el, f"{p:.4f}", f"{m:.4f}",
                                f"{p - m:.4f}"])

    def _show_calibration_dialog(self, cal, cv=None, n_dropped=0, n_candidates=None):
        """Show the derived k-factors (k, n_tiles, R²) + resulting calibrated wt%."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('EDS calibration from SPX (k-factors)')
        layout = QtWidgets.QVBoxLayout(dlg)
        scope_txt = ("all SPX tiles in the mosaic" if self._calibration_scope == "mosaic"
                     else "tiles inside the ROI")
        layout.addWidget(QtWidgets.QLabel(
            f"Anchored {len(cal.k_factors)} element(s) over {cal.n_tiles} SPX tile(s) "
            f"({scope_txt}). wt% ≈ k · net-count "
            f"({getattr(cal, 'background', 'sideband')} background; see console for the per-element plan)."))
        if n_dropped and n_candidates:
            lbl_drop = QtWidgets.QLabel(
                f"{n_dropped} of {n_candidates} candidate tiles excluded by the "
                "quality filter (near-empty / low coverage).")
            lbl_drop.setStyleSheet("color: #b08000;")
            layout.addWidget(lbl_drop)

        rows = sorted(cal.k_factors.items(),
                      key=lambda kv: -kv[1].sum_wt)
        table = QtWidgets.QTableWidget()
        table.setRowCount(len(rows)); table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['Element', 'k', 'tiles', 'R²'])
        for i, (el, kf) in enumerate(rows):
            r2 = '—' if kf.r2 is None else f"{kf.r2:.3f}"
            for c, txt in enumerate((el, f"{kf.k:.4g}", str(kf.n_tiles), r2)):
                table.setItem(i, c, QtWidgets.QTableWidgetItem(txt))
        layout.addWidget(table)

        if self.last_calibrated_composition:
            layout.addWidget(QtWidgets.QLabel('Calibrated ROI composition (wt%):'))
            comp_tbl = QtWidgets.QTableWidget()
            crows = sorted(self.last_calibrated_composition.items(), key=lambda kv: -kv[1])
            comp_tbl.setRowCount(len(crows)); comp_tbl.setColumnCount(2)
            comp_tbl.setHorizontalHeaderLabels(['Element', 'wt%'])
            for i, (el, wt) in enumerate(crows):
                comp_tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(el))
                comp_tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{wt:.2f}"))
            layout.addWidget(comp_tbl)

        if cv is not None:
            method = getattr(cv, "method", "loo")
            if method == "holdout":
                hdr = (f"Hold-out validation ({cv.n_folds} folds, "
                       f"{cv.n_tiles} predictions): overall RMSE "
                       f"{cv.rmse:.2f} ± {getattr(cv, 'rmse_std', 0.0):.2f} wt%")
            else:
                hdr = (f"Hold-out validation (leave-one-tile-out, {cv.n_tiles} "
                       f"tiles): overall RMSE {cv.rmse:.2f} wt%")
                if cv.n_tiles < 3:
                    hdr += "  — few tiles, weak evidence"
            layout.addWidget(QtWidgets.QLabel(hdr))
            cv_tbl = QtWidgets.QTableWidget()
            cv_rows = sorted(cv.per_element.items(),
                             key=lambda kv: -kv[1].mean_measured)
            cv_tbl.setRowCount(len(cv_rows)); cv_tbl.setColumnCount(5)
            cv_tbl.setHorizontalHeaderLabels(
                ['Element', 'RMSE wt%', '± σ', 'Bias wt%', 'mean SPX wt%'])
            for i, (el, s) in enumerate(cv_rows):
                vals = (el, f"{s.rmse:.2f}", f"{getattr(s, 'rmse_std', 0.0):.2f}",
                        f"{s.bias:+.2f}", f"{s.mean_measured:.2f}")
                for c, txt in enumerate(vals):
                    cv_tbl.setItem(i, c, QtWidgets.QTableWidgetItem(txt))
            layout.addWidget(cv_tbl)
            btn_cv = QtWidgets.QPushButton('Export validation CSV …')
            btn_cv.clicked.connect(lambda: self._export_calibration_cv_csv(cv))
            layout.addWidget(btn_cv)
        elif cal.n_tiles >= 2:
            layout.addWidget(QtWidgets.QLabel(
                "Hold-out validation unavailable (needs ≥2 tiles with a usable "
                "held-out prediction)."))

        notes = []
        if cal.spx_only:
            notes.append("Not anchored (SPX only / no co-located counts): "
                         + ", ".join(cal.spx_only))
        if cal.eds_only:
            notes.append("Dropped (EDS only / no SPX reference): " + ", ".join(cal.eds_only))
        notes.append("Captures the average matrix correction baked into SPX, not "
                     "per-pixel matrix variation. Calibrated EDS now feeds modelling, "
                     "composition-vs-scale, and structural mineral ID.")
        lbl = QtWidgets.QLabel("\n".join("• " + n for n in notes))
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn = QtWidgets.QPushButton('OK'); btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.resize(420, 680 if cv is not None else 460)
        dlg.exec_()

    def on_compare_compositions(self):
        """Compare the active ROI composition from EDS, calibrated EDS, and SPX.

        Assembles up to three bulk compositions for the current ROI — raw EDS
        net-count % (always), SPX-calibrated EDS wt% (from an existing or
        on-the-fly k-factor fit), and quantified SPX wt% (whole tiles ≥95%
        inside) — and shows them side by side (table + grouped bar chart) so the
        intensity-vs-mass discrepancy and the calibration's correction are
        visible at a glance. Bruker-focused; Oxford caches show only their own
        montaged column.
        """
        import eds_quantify as eq

        if not self.last_band_maps or not self.last_roi:
            QtWidgets.QMessageBox.warning(
                self, "Compare compositions",
                "Compute the FULL-RES EDS maps for an ROI first.")
            return

        frac, _ = _net_count_fractions(self.last_band_maps, sub_mask=self.last_roi_mask)
        if not frac:
            QtWidgets.QMessageBox.warning(
                self, "Compare compositions", "No EDS net-counts in the ROI.")
            return

        is_oxford = self._is_oxford_cache()
        dt = self._cache_data_type()

        # 1. Raw EDS — net-count % (Bruker / oxford_counts) or wt% (oxford_wt_pct).
        if dt == "oxford_wt_pct":
            eds_label = "EDS wt% (montaged)"
        elif dt == "oxford_counts":
            eds_label = "EDS net-count % (montaged)"
        else:
            eds_label = "EDS net-count %"
        eds_comp = {el: 100.0 * float(v) for el, v in frac.items()}

        # 2. Calibrated EDS — apply an existing calibration, else derive inline.
        calib_comp: Dict[str, float] = {}
        notes: List[str] = []
        if not is_oxford:
            cal = self.eds_calibration
            if cal is None:
                spx_by_tile, counts_by_tile = self._collect_roi_calibration_tiles()
                if counts_by_tile:
                    cal = eq.derive_k_factors(
                        spx_by_tile, counts_by_tile,
                        drop=eq.ANION_ELEMENTS | self._coating_elements(),
                        background=_bg_scheme())
                    if cal.k_factors:
                        self.eds_calibration = cal
                        notes.append(
                            f"Calibration derived on the fly from {cal.n_tiles} "
                            "SPX tile(s) and kept for this ROI.")
                    else:
                        cal = None
            if cal is not None and cal.k_factors:
                calib_comp = eq.apply_calibration_fractions(frac, cal.k())
                self.last_calibrated_composition = dict(calib_comp)
                # The k-factor fit excludes O (its soft Kα is a poor mass proxy
                # and SPX derives O by stoichiometry, not as a cation), so the
                # calibrated wt% is cation-only and would sit inflated next to
                # the O-bearing net-count and SPX columns. Reconstruct O the way
                # SPX does — from the standard oxides — to put it on the same
                # basis. (last_calibrated_composition stays cation-only: that is
                # what modelling/mineral-ID renormalise from.)
                fe3 = float(getattr(self, "fe3_over_fe_total", 0.2) or 0.2)
                calib_comp = add_stoichiometric_oxygen(calib_comp, fe3_over_fe_total=fe3)

        # 3. SPX quantified (Bruker only).
        spx_comp: Dict[str, float] = {}
        if not is_oxford:
            spx_comp = self._spx_composition_for_bbox(self.last_roi, min_overlap=0.95) or {}
            if spx_comp:
                self.last_spx_composition = dict(spx_comp)

        sources = [
            (eds_label, eds_comp, True),
            ("EDS calibrated (wt%, O by stoich.)", calib_comp, bool(calib_comp)),
            ("SPX quantified (wt%)", spx_comp, bool(spx_comp)),
        ]

        notes.insert(0,
                     "EDS net-count % is X-ray-intensity normalised (not mass); "
                     "the calibrated column rescales it toward SPX wt%.")
        notes.append(
            "O treatment: the k-factor fit excludes O, so the calibrated column's "
            "O is reconstructed by stoichiometry from its oxides — the same way "
            "SPX derives O. The net-count column's O is measured intensity (not "
            "mass-proportional). Toggle 'cations only' below for an O-free, "
            "assumption-free comparison.")
        if not is_oxford and not calib_comp:
            notes.append(
                "No calibration available — draw an ROI that fully contains an "
                "SPX tile, then click 'Calibrate EDS (SPX k-factors)'.")

        _CompositionCompareDialog(sources, notes=notes, parent=self).exec_()

    # ---- Composition vs scale -----------------------------------------------

    def on_composition_scale(self):
        """Quadtree-based composition variance + optional thermod sweep.

        Pre-conditions: an ROI must be drawn. The EDS source additionally
        requires that ``Compute FULL-RES maps for ROI`` was already run for
        the active ROI (so ``self.last_band_maps`` is populated). The SPX
        source needs a valid manifest.
        """
        # 1. Resolve ROI
        try:
            roi, roi_mask = self._get_roi_bbox_and_mask()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "ROI required", str(e))
            return

        # 2. Decide which sources are available
        has_eds = (
            self.last_band_maps is not None
            and bool(self.last_band_maps)
            and self.last_roi == roi
        )
        # SPX is available if there are any tile bboxes from the manifest.
        # Oxford montaged caches have no SPX (no per-tile spectrum files),
        # so disable the radio outright when the cache is Oxford-tagged.
        has_spx = bool(self._get_tile_bboxes()) and not self._is_oxford_cache()

        if not has_eds and not has_spx:
            QtWidgets.QMessageBox.warning(
                self, "No composition source",
                "Neither EDS band maps nor SPX-bearing tiles are available "
                "for this ROI. Compute full-res ROI maps or check the manifest."
            )
            return

        if not has_eds and self.last_band_maps is not None:
            QtWidgets.QMessageBox.information(
                self, "EDS path skipped",
                "The ROI has changed since the last full-res computation. "
                "Re-run 'Compute FULL-RES maps for ROI' for the current ROI "
                "to use the EDS source."
            )

        # 3. Make sure the ROI is at least subdividable once; otherwise the
        #    quadtree dialog can't offer any useful length scales.
        max_useful = _max_useful_quadtree_depth(roi)
        if max_useful < 1:
            QtWidgets.QMessageBox.warning(
                self, "ROI too small",
                "The ROI is too small to subdivide into a quadtree. Draw a "
                "larger ROI (at least ~8 × 8 px on the short side) and retry."
            )
            return

        # 4. Median SPX tile short-side from the manifest (used by the dialog
        #    to clamp depth so SPX windows can never be smaller than a tile).
        spx_min_window_px: Optional[int] = None
        if has_spx:
            tile_short_sides = []
            for bb, _r in self._get_tile_bboxes():
                h = bb[1] - bb[0]
                w = bb[3] - bb[2]
                if h > 0 and w > 0:
                    tile_short_sides.append(min(h, w))
            if tile_short_sides:
                spx_min_window_px = int(np.median(tile_short_sides))

        # 5. Settings dialog (depth + per-level window cap auto-derived).
        #    Full-res-from-cache is offered (opt-in, OFF by default) only when
        #    the EDS maps were downsampled for display (ds>1) and the cache can
        #    supply full resolution. Default uses the in-memory maps — fast, and
        #    for window-averaged composition the 1/N maps are statistically fine.
        eds_ds = self.last_band_maps_ds if has_eds else 1
        fullres_cache_dir = None
        if has_eds and eds_ds > 1:
            try:
                import eds_cache as _edsc
                _cd = _edsc.resolve_cache_dir(Path(self.manifest).parent)
                if _edsc.is_valid_for(_cd, Path(self.manifest),
                                      required_elements=self.last_elements):
                    fullres_cache_dir = _cd
            except Exception:
                fullres_cache_dir = None

        n_elements = len(self._selected_elements())
        dlg = _CompositionScaleDialog(roi_bbox=roi,
                                       has_eds=has_eds,
                                       has_spx_tiles=has_spx,
                                       element_count=n_elements,
                                       spx_min_window_px=spx_min_window_px,
                                       fullres_cache_available=(fullres_cache_dir is not None),
                                       eds_ds=eds_ds,
                                       parent=self)
        if not dlg.exec_():
            return
        opts = dlg.get_options()
        # Stamp the effective data_type onto opts so downstream code (result
        # dialog labels, sweep records) can distinguish Bruker net-counts from
        # Oxford pre-quantified Wt% / SPX-calibrated EDS without re-reading the
        # cache manifest.
        opts["data_type"] = self._effective_data_type()

        # 6. Build the quadtree windows per level, capped during generation.
        #    A deep quadtree on a large ROI has up to 4**depth windows per level;
        #    _quadtree_sampled_windows emits at most ``cap`` of them directly from
        #    the axis-segment grid, so we never materialise (or process) millions.
        max_windows: Dict[int, int] = opts.get("max_windows", {}) or {}
        sampling = opts.get("sampling", "first")
        rng = np.random.default_rng(0) if sampling == "random" else None
        levels: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for lv in range(int(opts["max_depth"]) + 1):
            cap = max_windows.get(lv)  # None -> all (dialog defaults are bounded)
            wins = _quadtree_sampled_windows(roi, lv, cap=cap, rng=rng)
            if wins:
                levels[lv] = wins
        if not levels:
            QtWidgets.QMessageBox.warning(self, "Empty quadtree",
                                          "ROI is too small to subdivide.")
            return
        n_total = sum(len(v) for v in levels.values())
        if n_total == 0:
            QtWidgets.QMessageBox.warning(self, "No windows",
                                          "Per-level caps reduced the count to zero.")
            return

        # 7. Worker.  EDS path resolution strategy:
        #    - default: slice the in-memory band maps (full-res if the ROI fit
        #      in RAM; the 1/ds display maps otherwise — fast, no disk I/O,
        #      and statistically fine for window-averaged composition).
        #    - opt-in "full resolution from cache" (only offered when ds>1 and a
        #      cache exists): read each window full-res from the cache. Correct
        #      but SLOW on large mosaics (re-reads the ROI at every scale).
        spx_cb = (lambda bbox: self._spx_composition_for_bbox(bbox))
        ds = self.last_band_maps_ds if opts["source"] == "eds" else 1
        use_fullres_cache = (opts["source"] == "eds"
                             and opts.get("fullres_cache")
                             and fullres_cache_dir is not None)
        # When an SPX calibration is active, convert per-window EDS net fractions
        # to calibrated wt% at every quadtree level (None = raw fractions×100).
        k_for_worker = (self.eds_calibration.k()
                        if (opts["source"] == "eds" and self.eds_calibration is not None)
                        else None)

        if use_fullres_cache:
            # Full-res from cache: coordinates + mask stay at native resolution.
            worker = _CompositionScaleWorker(
                source=opts["source"], levels=levels, roi_bbox=roi,
                roi_mask=roi_mask, band_maps=None, spx_callback=spx_cb,
                min_mask=opts["min_mask"], band_maps_ds=1,
                cache_dir=fullres_cache_dir, cache_elements=list(self.last_elements),
                cache_blend_mode=self.last_blend_mode, k_factors=k_for_worker,
                parent=self,
            )
            print(f"[comp-scale] reading full-res windows from cache "
                  f"(display was 1/{ds}); this can be slow on large mosaics.",
                  flush=True)
        else:
            worker_mask = roi_mask
            if ds > 1 and roi_mask is not None:
                worker_mask = np.asarray(roi_mask)[::ds, ::ds]
            worker = _CompositionScaleWorker(
                source=opts["source"],
                levels=levels,
                roi_bbox=roi,
                roi_mask=worker_mask,
                band_maps=self.last_band_maps if opts["source"] == "eds" else None,
                spx_callback=spx_cb,
                min_mask=opts["min_mask"],
                band_maps_ds=ds,
                k_factors=k_for_worker,
                parent=self,
            )

        pd_prog = QtWidgets.QProgressDialog(
            "Computing per-window composition…", "Cancel", 0, n_total, self
        )
        pd_prog.setWindowTitle("Composition vs scale")
        pd_prog.setWindowModality(QtCore.Qt.WindowModal)
        pd_prog.setMinimumDuration(0)
        pd_prog.setValue(0)

        def _on_progress(done, total):
            if pd_prog.maximum() != total:
                pd_prog.setMaximum(total)
            pd_prog.setValue(done)

        # Build a parent label tied to this ROI
        y0, y1, x0, x1 = roi
        parent_label = (
            f"ROI [y={y0}:{y1}, x={x0}:{x1}] "
            f"({(y1 - y0)}×{(x1 - x0)} px)"
        )

        def _on_finished(records, summary):
            pd_prog.close()
            self._comp_scale_worker = None
            usable = sum(1 for r in records if r.get("wt_pct"))
            print(f"[comp-scale] {usable}/{len(records)} windows produced "
                  f"a composition (source={opts['source']}, "
                  f"depth={opts['max_depth']})", flush=True)
            # Keep the result so it re-opens without re-running the worker.
            self.last_comp_scale = {
                "records": records, "summary": summary, "opts": opts,
                "parent_label": parent_label, "roi_bbox": list(roi)}
            self._refresh_control_availability()
            self._show_composition_scale_result(
                records, summary, opts, parent_label, roi,
            )

        def _on_error(msg):
            pd_prog.close()
            self._comp_scale_worker = None
            QtWidgets.QMessageBox.critical(self, "Composition vs scale failed",
                                           msg)

        worker.progress.connect(_on_progress)
        worker.finished_with.connect(_on_finished)
        worker.error.connect(_on_error)
        pd_prog.canceled.connect(worker.cancel)

        self._comp_scale_worker = worker
        worker.start()

    def on_reopen_composition_scale(self):
        """Re-open the last composition-vs-scale result window (no recompute)."""
        cs = self.last_comp_scale
        if not cs or not cs.get("records"):
            QtWidgets.QMessageBox.information(
                self, "No results",
                "Run 'Composition vs scale …' first; the results can then be "
                "re-opened here without recomputing.")
            return
        # summary keys are ints; JSON round-trips them as strings, so coerce.
        summary = {int(k): v for k, v in (cs.get("summary") or {}).items()}
        self._show_composition_scale_result(
            cs["records"], summary, cs["opts"], cs["parent_label"],
            tuple(cs["roi_bbox"]) if cs.get("roi_bbox") else None)

    def on_export_perplex(self):
        """
        Export Perple_X-ready file.
        Options:
          • Composition-only TXT (oxide wt%)
          • BUILD keyword file (batchable into Perple_X BUILD)
        Prefers SPX-derived element wt%, then SPX-calibrated EDS, then raw
        net-count fractions (the last with a warning).
        """
        # Choose composition source (element wt%): SPX > calibrated EDS > raw.
        comp_el, source = self._preferred_roi_composition()
        if source.startswith("net-count"):
            proceed = QtWidgets.QMessageBox.question(
                self, "Use approximate composition?",
                "No SPX or SPX-calibrated composition is available.\n"
                "Use the quick net-count fractions (approximate) as element wt% for oxide conversion?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
            )
            if proceed != QtWidgets.QMessageBox.Yes:
                comp_el = None
        if not comp_el:
            QtWidgets.QMessageBox.information(self, "No composition",
                                              "Compute an ROI and/or SPX-based composition first.")
            return

        # Format picker
        fmt_dialog = QtWidgets.QDialog(self)
        fmt_dialog.setWindowTitle("Perple_X export format")
        vbox = QtWidgets.QVBoxLayout(fmt_dialog)
        info = QtWidgets.QLabel("Choose export format:")
        vbox.addWidget(info)
        cmb = QtWidgets.QComboBox()
        cmb.addItems(["Composition TXT (oxide wt%)", "BUILD keyword file"])
        vbox.addWidget(cmb)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        vbox.addWidget(btns)
        btns.accepted.connect(fmt_dialog.accept);
        btns.rejected.connect(fmt_dialog.reject)
        if not fmt_dialog.exec_():
            return
        build_mode = (cmb.currentIndex() == 1)

        # Common oxide options
        fe3_label = QtWidgets.QLabel("Fe³⁺ / Fe_total (0–1):")
        fe3_input = QtWidgets.QDoubleSpinBox();
        fe3_input.setRange(0.0, 1.0);
        fe3_input.setSingleStep(0.05);
        fe3_input.setValue(0.20)
        h2o_label = QtWidgets.QLabel("H2O (wt% add):")
        h2o_input = QtWidgets.QDoubleSpinBox();
        h2o_input.setRange(0.0, 1000.0);
        h2o_input.setDecimals(3);
        h2o_input.setValue(0.0)
        co2_label = QtWidgets.QLabel("CO2 (wt% add):")
        co2_input = QtWidgets.QDoubleSpinBox();
        co2_input.setRange(0.0, 1000.0);
        co2_input.setDecimals(3);
        co2_input.setValue(0.0)

        # Extra fields for BUILD keyword
        title_edit = QtWidgets.QLineEdit("ROI composition")
        thermo_edit = QtWidgets.QLineEdit("stx21ver.dat")  # change to your local DB filename
        sol_edit = QtWidgets.QLineEdit("solution_model.dat")  # change to your local solution model file
        act_edit = QtWidgets.QLineEdit("none")
        mode_edit = QtWidgets.QComboBox()
        mode_edit.addItems(["td", "tc", "ps", "hpx"])  # typical Perple_X modes; pick what you need

        form = QtWidgets.QFormLayout()
        form.addRow('Fe³⁺/Fe_total:', fe3_input)
        form.addRow('H2O add (wt%):', h2o_input)
        form.addRow('CO2 add (wt%):', co2_input)
        if build_mode:
            form.addRow('Title:', title_edit)
            form.addRow('Thermo DB:', thermo_edit)
            form.addRow('Solution DB:', sol_edit)
            form.addRow('Activity DB:', act_edit)
            form.addRow('Mode:', mode_edit)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Perple_X export options")
        vbox2 = QtWidgets.QVBoxLayout(dlg)
        vbox2.addLayout(form)
        btns2 = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        vbox2.addWidget(btns2)
        btns2.accepted.connect(dlg.accept);
        btns2.rejected.connect(dlg.reject)
        if not dlg.exec_():
            return

        fe3frac = float(fe3_input.value())
        add_h2o = float(h2o_input.value())
        add_co2 = float(co2_input.value())

        ox = elements_to_oxides_wt(comp_el, fe3_over_fe_total=fe3frac, add_h2o=add_h2o, add_co2=add_co2)

        # Save file
        fd = QtWidgets.QFileDialog(self)
        if build_mode:
            fd.setWindowTitle("Save Perple_X BUILD keyword file")
            fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            fd.setNameFilters(["Perple_X BUILD (*.build *.kwd *.txt)", "All files (*)"])
            fd.setDefaultSuffix("build")
        else:
            fd.setWindowTitle("Save Perple_X composition (oxide wt%)")
            fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            fd.setNameFilters(["Perple_X text (*.txt *.comp)", "All files (*)"])
            fd.setDefaultSuffix("txt")
        if not fd.exec_():
            return
        out_path = Path(fd.selectedFiles()[0])

        try:
            if build_mode:
                _write_perplex_build_keyword(
                    out_path,
                    ox,
                    title=title_edit.text().strip() or "ROI composition",
                    thermo_db=thermo_edit.text().strip() or "stx21ver.dat",
                    solution_db=sol_edit.text().strip() or "solution_model.dat",
                    activity_db=act_edit.text().strip() or "none",
                    calc_mode=mode_edit.currentText().strip() or "td",
                )
            else:
                lines = []
                lines.append("# Perple_X composition (wt%) exported from viewer")
                lines.append(f"# Source: {source}")
                lines.append(f"# Fe3+/Fe_total={fe3frac:.3f}, H2O_add={add_h2o:.3f} wt%, CO2_add={add_co2:.3f} wt%")
                order = ["SiO2", "Al2O3", "FeO", "Fe2O3", "MgO", "CaO", "Na2O", "K2O", "TiO2", "MnO", "P2O5", "H2O",
                         "CO2"]
                seen = set()
                for k in order:
                    if k in ox:
                        lines.append(f"{k:<6s} {ox[k]:.4f}")
                        seen.add(k)
                for k, v in sorted(ox.items()):
                    if k not in seen:
                        lines.append(f"{k:<6s} {v:.4f}")
                out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", f"Could not write file:\n{e}")
            return

        QtWidgets.QMessageBox.information(
            self, "Perple_X export",
            f"Saved {'BUILD keyword' if build_mode else 'composition'} to:\n{out_path}\n\n"
            + (
                "Open Perple_X BUILD and load this keyword file." if build_mode else "Paste into Perple_X Build → Composition.")
        )

    # ---- Thermodynamic modelling ----

    def on_run_thermod(self):
        """Run a thermodynamic equilibrium calculation from the current ROI composition."""
        # 1. Resolve ROI composition (if any) to pre-populate the dialog:
        #    SPX > SPX-calibrated EDS > raw net-count fractions.
        comp_el_roi, source = self._preferred_roi_composition()

        # 2. Options dialog — always opens; user can switch to custom composition
        dlg = _ThermoRunDialog(comp_el_roi, source, parent=self)
        if not dlg.exec_():
            return
        opts = dlg.get_options()

        comp_el = opts["comp_el"]
        if not comp_el:
            QtWidgets.QMessageBox.critical(
                self, "No composition",
                "No composition is available. Enter a custom oxide wt% composition."
            )
            return

        # 3. Build Composition
        try:
            from thermo import ThermoCalc, Composition
        except ImportError as exc:
            QtWidgets.QMessageBox.critical(
                self, "thermo package not found",
                f"Could not import the thermo package:\n{exc}\n\n"
                "Make sure the thermo/ directory is in the same folder as viewer.py."
            )
            return

        # "extra" H2O/CO2 are wt% OF THE BULK for runs made from now on; the
        # choice is stored in the run options so a record replays consistently
        # (a record without the key predates 2026-10-05 -> "legacy" arithmetic).
        opts.setdefault("extra_basis", "bulk")
        extra: Optional[Dict[str, float]] = {}
        if opts["h2o"] > 0:
            extra["H2O"] = opts["h2o"]
        if opts["co2"] > 0:
            extra["CO2"] = opts["co2"]

        # Determine if composition is oxide wt% or element wt%
        # Custom input is oxide wt%; ROI comp_el is element wt%
        try:
            if opts["comp_source"] == "custom":
                comp = Composition.from_oxide_wt(comp_el)
            else:
                comp = Composition.from_element_wt(comp_el)
            if extra:
                ox = comp.to_oxide_wt(fe3_fraction=opts["fe3"], extra_oxides=extra, normalize=True,
                                      extra_basis=opts["extra_basis"])
                comp = Composition.from_oxide_wt(ox)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Composition error", str(exc))
            return

        # 4. Instantiate backend
        try:
            calc = ThermoCalc(backend=opts["backend"], **opts["backend_kwargs"])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Backend error", str(exc))
            return

        if not calc.check_backend():
            pkg = {"magemin":   "pip install petthermotools  (also requires Julia + MAGEMin_C.jl)",
                   "pycalphad":  "pip install pycalphad",
                   "perplex":    "install Perple_X and set the Perple_X dir"}.get(opts["backend"], "")
            QtWidgets.QMessageBox.warning(
                self, "Backend not available",
                f"Backend '{opts['backend']}' is not installed.\n"
                + (f"Install it with:  {pkg}" if pkg else "")
            )
            return

        # 5. Run
        try:
            if opts["mode"] == "single":
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    result = calc.run(comp, opts["p_gpa"], opts["t_c"],
                                      **({} if opts["backend"] == "pycalphad"
                                         else {"fe3_fraction": opts["fe3"]}))
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                self._show_thermod_single(result, comp_el, opts["comp_source"], opts=opts)
            else:
                # MAGEMin → subprocess pool (keeps the UI responsive despite
                # the juliacall main-thread constraint); perplex/pycalphad →
                # QThread.  See _dispatch_single_grid.
                self._dispatch_single_grid(
                    comp, opts, comp_el, opts["comp_source"], calc
                )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Thermodynamic calculation failed", str(exc)
            )

    def _on_thermod_grid_done(self, results, opts, comp_el, source, calc, pd_prog):
        pd_prog.close()
        self._thermod_worker = None
        self._show_thermod_grid(results, opts, comp_el, source, calc)

    def _on_thermod_grid_error(self, msg, pd_prog):
        pd_prog.close()
        self._thermod_worker = None
        if "cancelled" not in msg.lower():
            QtWidgets.QMessageBox.critical(self, "Thermodynamic calculation failed", msg)

    # ---- Single P-T grid dispatch (subprocess pool for MAGEMin) ------------

    @staticmethod
    def _backend_is_magemin(opts: dict) -> bool:
        """MAGEMin needs subprocess isolation (juliacall main-thread constraint)."""
        return str(opts.get("backend", "")).lower() == "magemin"

    def _dispatch_single_grid(self, comp, opts, comp_el, source, calc):
        """Run one P-T grid with a progress dialog, keeping the UI responsive.

        MAGEMin runs in a worker subprocess (juliacall cannot run in a
        QThread); perplex/pycalphad use the existing ``_ThermoGridWorker``
        QThread.  If the subprocess pool cannot start (e.g. Julia will not
        initialise in a subprocess) the run falls back to a synchronous
        main-thread call, so there is no regression versus the old behaviour.
        """
        if str(opts.get("backend", "")).lower() == "perplex":
            msg = ("Running P-T grid…\n"
                   "(BUILD → VERTEX → WERAMI — may take several minutes "
                   "for new compositions)")
        else:
            msg = "Running P-T grid…"
        pd_prog = QtWidgets.QProgressDialog(msg, "Cancel", 0, 0, self)
        pd_prog.setWindowTitle("Thermodynamic modelling")
        pd_prog.setWindowModality(QtCore.Qt.WindowModal)
        pd_prog.show()

        if not self._backend_is_magemin(opts):
            worker = _ThermoGridWorker(calc, comp, opts, parent=self)
            pd_prog.canceled.connect(worker.cancel)
            worker.finished.connect(
                lambda results: self._on_thermod_grid_done(
                    results, opts, comp_el, source, calc, pd_prog
                )
            )
            worker.error.connect(lambda msg: self._on_thermod_grid_error(msg, pd_prog))
            worker.start()
            self._thermod_worker = worker
            return

        # MAGEMin → one-job subprocess pool.
        from thermo.result import result_from_portable_dict
        job = {
            "comp": comp,
            "p_range": tuple(opts["p_range"]),
            "t_range": tuple(opts["t_range"]),
            "n_p": int(opts["n_p"]),
            "n_t": int(opts["n_t"]),
        }
        acc: dict = {}
        worker = _ThermoSweepWorker(
            [job], "magemin", opts.get("backend_kwargs", {}),
            max_workers=1, parent=self,
        )

        def _on_grid(idx, dicts):
            acc["results"] = [result_from_portable_dict(d) for d in dicts]

        def _on_grid_failed(idx, msg):
            acc["error"] = msg

        def _on_done():
            pd_prog.close()
            self._thermod_worker = None
            if "error" in acc:
                if "cancel" not in acc["error"].lower():
                    QtWidgets.QMessageBox.critical(
                        self, "Thermodynamic calculation failed", acc["error"]
                    )
                return
            results = acc.get("results")
            if not results:   # cancelled before completing
                return
            self._show_thermod_grid(results, opts, comp_el, source, calc)

        def _on_pool_failed(msg):
            self._thermod_worker = None
            self._run_grid_sync_fallback(comp, opts, comp_el, source, calc, pd_prog, msg)

        worker.grid_done.connect(_on_grid)
        worker.grid_failed.connect(_on_grid_failed)
        worker.finished_all.connect(_on_done)
        worker.pool_failed.connect(_on_pool_failed)
        pd_prog.canceled.connect(worker.cancel)
        worker.start()
        self._thermod_worker = worker

    def _run_grid_sync_fallback(self, comp, opts, comp_el, source, calc, pd_prog, reason):
        """Synchronous main-thread grid run — fallback if the subprocess pool fails."""
        print(f"[thermo] subprocess pool unavailable ({reason}); "
              f"running synchronously on the main thread.", flush=True)
        if pd_prog is not None:
            pd_prog.close()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if hasattr(calc._backend, "_activate"):
                calc._backend._activate()
            results = calc.run_grid(
                comp,
                p_range=opts["p_range"], t_range=opts["t_range"],
                n_p=opts["n_p"], n_t=opts["n_t"],
            )
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Thermodynamic calculation failed", str(exc)
            )
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._show_thermod_grid(results, opts, comp_el, source, calc)

    # ---- Thermod run records ----------------------------------------------

    def _capture_roi_for_run(self) -> dict:
        """Snapshot the ROI tied to the current modelling run (best effort).

        Falls back gracefully if no ROI was ever drawn (custom-composition
        runs). Captures bbox, the napari shape vertices + type so the ROI
        can be relocated/redrawn later (Locate ROI button).
        """
        roi: dict = {"bbox": None, "vertices": None, "shape_type": None,
                     "has_mask": bool(self.last_roi_mask is not None)}
        if self.last_roi is not None:
            roi["bbox"] = list(self.last_roi)

        # Best effort: find the ROI shape whose bbox best matches last_roi
        try:
            for lyr in self.viewer.layers:
                if isinstance(lyr, napari.layers.Shapes) and lyr.name.lower() == 'roi':
                    if len(lyr.data) == 0:
                        break
                    target = self.last_roi
                    best_idx = len(lyr.data) - 1
                    if target is not None:
                        ty0, ty1, tx0, tx1 = target
                        best_d = float("inf")
                        for i, verts in enumerate(lyr.data):
                            v = np.asarray(verts)
                            y0 = int(np.floor(v[:, 0].min()))
                            y1 = int(np.ceil(v[:, 0].max()))
                            x0 = int(np.floor(v[:, 1].min()))
                            x1 = int(np.ceil(v[:, 1].max()))
                            d = abs(y0 - ty0) + abs(y1 - ty1) + abs(x0 - tx0) + abs(x1 - tx1)
                            if d < best_d:
                                best_d = d
                                best_idx = i
                    verts = np.asarray(lyr.data[best_idx])
                    stype = (lyr.shape_type[best_idx]
                             if hasattr(lyr, 'shape_type') else 'rectangle')
                    roi["vertices"] = verts.tolist()
                    roi["shape_type"] = str(stype)
                    break
        except Exception:
            pass
        return roi

    def _record_thermod_run(self, mode: str, opts: dict, comp_el: dict,
                            comp_source: str, result_or_results) -> dict:
        """Create a serialisable run record and append it to history.

        ``result_or_results`` is a single ThermoResult for mode='single' or
        a list of ThermoResult for mode='grid'.
        """
        run_idx = len(self.last_thermod_runs) + 1
        roi_info = self._capture_roi_for_run()
        bbox = roi_info.get("bbox")
        if bbox is not None:
            y0, y1, x0, x1 = bbox
            roi_label = f"ROI #{run_idx} [y={y0}:{y1}, x={x0}:{x1}]"
        else:
            roi_label = f"Run #{run_idx} (no ROI — custom composition)"

        # Strip non-serialisable bits from opts (e.g. tuples already JSON-safe)
        opts_clean = {}
        for k, v in (opts or {}).items():
            if isinstance(v, tuple):
                opts_clean[k] = list(v)
            else:
                opts_clean[k] = v

        run: dict = {
            "id": uuid.uuid4().hex[:8],
            "index": run_idx,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "opts": opts_clean,
            "comp_el": dict(comp_el or {}),
            "comp_source": comp_source,
            "manifest": str(self.manifest),
            "roi_label": roi_label,
            "roi_bbox": list(bbox) if bbox is not None else None,
            "roi_shape": ({"vertices": roi_info["vertices"],
                           "type": roi_info["shape_type"]}
                          if roi_info.get("vertices") is not None else None),
        }
        if mode == "single":
            run["result"] = _serialize_thermod_result(result_or_results)
        else:
            run["results"] = [_serialize_thermod_result(r) for r in result_or_results]

        self.last_thermod_runs.append(run)
        return run

    def _locate_roi_overlay(self, run: dict):
        """Highlight the ROI for *run* on the BSE viewer.

        Adds (or refreshes) a `Modelling ROIs` Shapes layer with this run's
        polygon, then centres the napari camera on it. If no shape was
        captured (custom-composition run), informs the user.
        """
        shape = run.get("roi_shape")
        bbox = run.get("roi_bbox")
        if not shape and not bbox:
            QtWidgets.QMessageBox.information(
                self, "No ROI",
                "This modelling run has no associated ROI (custom composition)."
            )
            return

        # Build vertices: prefer stored shape; otherwise rectangle from bbox
        if shape and shape.get("vertices"):
            verts = np.asarray(shape["vertices"], dtype=float)
            stype = shape.get("type") or "polygon"
        else:
            y0, y1, x0, x1 = bbox
            verts = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)
            stype = "rectangle"

        layer_name = "Modelling ROIs"
        lyr = None
        for L in self.viewer.layers:
            if L.name == layer_name:
                lyr = L
                break
        if lyr is None:
            lyr = self.viewer.add_shapes(
                name=layer_name,
                edge_color='magenta',
                face_color='transparent',
                edge_width=3,
            )

        text = run.get("roi_label", run.get("id", ""))
        # Avoid duplicates: if an entry with same label exists, skip add
        existing_texts = []
        try:
            existing_texts = list(lyr.text.values) if getattr(lyr, "text", None) else []
        except Exception:
            existing_texts = []
        if text not in existing_texts:
            try:
                lyr.add(verts, shape_type=stype)
                # Set/extend text labels to match shape order
                labels = existing_texts + [text]
                lyr.text = {"string": labels, "color": "magenta", "size": 10,
                            "anchor": "upper_left"}
            except Exception:
                # Fallback: just add the shape without text
                try:
                    lyr.add(verts, shape_type=stype)
                except Exception:
                    pass

        # Centre camera on the bbox
        try:
            if bbox is not None:
                y0, y1, x0, x1 = bbox
                self.viewer.camera.center = ((y0 + y1) / 2.0, (x0 + x1) / 2.0)
            lyr.visible = True
        except Exception:
            pass

    def on_show_thermod_runs(self):
        """Open the modelling-runs list dialog."""
        if not self.last_thermod_runs:
            QtWidgets.QMessageBox.information(
                self, "No modelling runs",
                "No thermodynamic modelling runs have been recorded yet.\n\n"
                "Use 'Run thermodynamic model …' to add one, or load a session "
                "that contains saved runs."
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Thermodynamic modelling runs")
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel(
            f"{len(self.last_thermod_runs)} run(s) recorded. "
            "Select one and choose an action."
        ))

        tbl = QtWidgets.QTableWidget()
        cols = ["#", "Time", "Mode", "Backend", "ROI", "Comp source"]
        tbl.setColumnCount(len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.setRowCount(len(self.last_thermod_runs))
        for i, run in enumerate(self.last_thermod_runs):
            opts = run.get("opts", {}) or {}
            tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(str(run.get("index", i + 1))))
            tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(run.get("timestamp", "")))
            tbl.setItem(i, 2, QtWidgets.QTableWidgetItem(run.get("mode", "")))
            tbl.setItem(i, 3, QtWidgets.QTableWidgetItem(opts.get("backend", "")))
            tbl.setItem(i, 4, QtWidgets.QTableWidgetItem(run.get("roi_label", "")))
            tbl.setItem(i, 5, QtWidgets.QTableWidgetItem(run.get("comp_source", "")))
        tbl.resizeColumnsToContents()
        tbl.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(tbl)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_view = QtWidgets.QPushButton("View results")
        btn_params = QtWidgets.QPushButton("View parameters")
        btn_locate = QtWidgets.QPushButton("Locate ROI")
        btn_compare = QtWidgets.QPushButton("Compare selected …")
        btn_export = QtWidgets.QPushButton("Export bundle …")
        btn_delete = QtWidgets.QPushButton("Delete")
        btn_clear_cache = QtWidgets.QPushButton("Clear result cache …")
        btn_clear_cache.setToolTip(
            "Delete memoised P-T grid results. Identical grids are normally "
            "served instantly from this on-disk cache; clear it to force fresh "
            "computation (e.g. after swapping a thermodynamic database)."
        )
        btn_close = QtWidgets.QPushButton("Close")
        for b in (btn_view, btn_params, btn_locate, btn_compare,
                  btn_export, btn_delete):
            btn_row.addWidget(b)
        btn_row.addStretch()
        btn_row.addWidget(btn_clear_cache)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        def _selected() -> Optional[dict]:
            rows = tbl.selectionModel().selectedRows() if tbl.selectionModel() else []
            if not rows:
                QtWidgets.QMessageBox.information(dlg, "No selection",
                                                  "Select a run first.")
                return None
            return self.last_thermod_runs[rows[0].row()]

        def _selected_many() -> List[dict]:
            rows = tbl.selectionModel().selectedRows() if tbl.selectionModel() else []
            return [self.last_thermod_runs[r.row()] for r in rows]

        def _on_view():
            run = _selected()
            if run is None:
                return
            self._open_thermod_run(run)

        def _on_params():
            run = _selected()
            if run is None:
                return
            self._show_run_parameters_dialog(run)

        def _on_locate():
            run = _selected()
            if run is None:
                return
            self._locate_roi_overlay(run)

        def _on_export():
            run = _selected()
            if run is None:
                return
            out = QtWidgets.QFileDialog.getExistingDirectory(
                dlg, "Export bundle to folder", str(self.manifest.parent),
            )
            if not out:
                return
            target = Path(out) / f"thermod_run_{run.get('index', '?')}_{run.get('id', '')}"
            try:
                written = _export_thermod_bundle(run, target, fig=None)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(dlg, "Export failed", str(exc))
                return
            QtWidgets.QMessageBox.information(
                dlg, "Bundle exported",
                f"Wrote:\n  " + "\n  ".join(str(p) for p in written)
            )

        def _on_delete():
            row = tbl.currentRow()
            if row < 0 or row >= len(self.last_thermod_runs):
                return
            run = self.last_thermod_runs[row]
            ans = QtWidgets.QMessageBox.question(
                dlg, "Delete run",
                f"Remove '{run.get('roi_label', run.get('id', '?'))}' from history?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
            del self.last_thermod_runs[row]
            tbl.removeRow(row)

        def _on_compare():
            runs = _selected_many()
            if len(runs) < 2:
                QtWidgets.QMessageBox.information(
                    dlg, "Need 2+ runs",
                    "Hold Ctrl or Shift to select two or more runs."
                )
                return
            grid_runs = [r for r in runs if r.get("mode") == "grid"]
            if len(grid_runs) < 2:
                QtWidgets.QMessageBox.warning(
                    dlg, "Grid runs only",
                    "Compare needs at least two grid (P-T) runs."
                )
                return
            self._show_compare_runs_dialog(grid_runs)

        def _on_clear_cache():
            try:
                from thermo.cache import GridCache
                gc = GridCache()
                s = gc.stats()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(dlg, "Cache error", str(exc))
                return
            mb = s["total_bytes"] / (1024 * 1024)
            ans = QtWidgets.QMessageBox.question(
                dlg, "Clear result cache",
                f"Delete {s['n_entries']} memoised P-T grid(s) "
                f"({mb:.1f} MiB) from:\n{s['dir']}\n\n"
                "This only affects cached computations — your recorded runs "
                "and session are untouched. Identical grids will be recomputed "
                "next time. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
            n = gc.clear()
            QtWidgets.QMessageBox.information(
                dlg, "Cache cleared", f"Removed {n} cached grid(s)."
            )

        btn_view.clicked.connect(_on_view)
        btn_params.clicked.connect(_on_params)
        btn_locate.clicked.connect(_on_locate)
        btn_compare.clicked.connect(_on_compare)
        btn_export.clicked.connect(_on_export)
        btn_delete.clicked.connect(_on_delete)
        btn_clear_cache.clicked.connect(_on_clear_cache)

        # Double-click row → view results
        tbl.doubleClicked.connect(lambda _: _on_view())

        dlg.resize(820, 360)
        dlg.exec_()

    def _show_run_parameters_dialog(self, run: dict):
        """Standalone read-only parameters viewer for a saved run."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(
            f"Modelling parameters — {run.get('roi_label', run.get('id', '?'))}"
        )
        layout = QtWidgets.QVBoxLayout(dlg)
        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFont("Courier New")
        font.setStyleHint(QtGui.QFont.Monospace)
        txt.setFont(font)
        txt.setPlainText(_thermod_parameters_text(run))
        layout.addWidget(txt)
        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Save as text …")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        def _save():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save parameters as text",
                f"thermod_run_{run.get('index', '?')}_{run.get('id', '')}_params.txt",
                "Text files (*.txt);;All files (*)",
            )
            if path:
                Path(path).write_text(txt.toPlainText(), encoding="utf-8")
                QtWidgets.QMessageBox.information(
                    dlg, "Saved", f"Parameters written to:\n{path}"
                )

        btn_export.clicked.connect(_save)
        dlg.resize(640, 480)
        dlg.exec_()

    def _open_thermod_run(self, run: dict):
        """Re-open the result dialog for a saved (or just-recorded) run."""
        try:
            if run.get("mode") == "single":
                result = _deserialize_thermod_result(run.get("result", {}))
                self._show_thermod_single(
                    result, run.get("comp_el", {}), run.get("comp_source", ""),
                    opts=run.get("opts", {}), run=run,
                )
            else:
                results_obj = [_deserialize_thermod_result(d)
                               for d in run.get("results", [])]
                self._show_thermod_grid(
                    results_obj, run.get("opts", {}),
                    run.get("comp_el", {}), run.get("comp_source", ""),
                    calc=None, run=run,
                )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not re-open run", str(exc)
            )

    def _show_compare_runs_dialog(self, grid_runs: List[dict]):
        """Compare 2+ grid runs that share the same P-T grid.

        Tabs:
          - Phase-field agreement: per-cell, fraction of selected runs that
            predict the same dominant assemblage as the first run.
          - Property spread: per shared bulk property, std/range across runs.
          - Summary table: headline numbers per run.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )
        from matplotlib.colors import ListedColormap

        # Validate shared grid
        def _grid_key(r):
            o = r.get("opts", {}) or {}
            pr = tuple(o.get("p_range", ())) if o.get("p_range") else ()
            tr = tuple(o.get("t_range", ())) if o.get("t_range") else ()
            return (pr, tr, o.get("n_p"), o.get("n_t"))

        keys = {_grid_key(r) for r in grid_runs}
        shared_grid = len(keys) == 1 and all(keys.pop()) if len(keys) == 1 else False
        # Re-evaluate: pop consumed the set above; recompute cleanly
        keys = {_grid_key(r) for r in grid_runs}
        shared_grid = len(keys) == 1
        ref_opts = grid_runs[0].get("opts", {}) or {}

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(
            f"Compare {len(grid_runs)} thermodynamic runs"
        )
        layout = QtWidgets.QVBoxLayout(dlg)

        header = QtWidgets.QLabel(
            f"<b>{len(grid_runs)}</b> runs selected.  "
            + (f"Shared P-T grid: P=[{ref_opts.get('p_range', [None,None])[0]:.2f}, "
               f"{ref_opts.get('p_range', [None,None])[1]:.2f}] GPa, "
               f"T=[{ref_opts.get('t_range', [None,None])[0]:.0f}, "
               f"{ref_opts.get('t_range', [None,None])[1]:.0f}] °C, "
               f"{ref_opts.get('n_p')}×{ref_opts.get('n_t')} pts."
               if shared_grid else
               "<span style='color:red'>Grids differ across selection — "
               "agreement & spread maps disabled.</span>")
        )
        layout.addWidget(header)
        layout.addWidget(_hline())

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # ── Tab: per-run summary ──────────────────────────────────────────
        s_w = QtWidgets.QWidget()
        s_layout = QtWidgets.QVBoxLayout(s_w)
        stbl = QtWidgets.QTableWidget()
        # Discover shared bulk-property keys (intersection of first run vs others)
        def _bulk_keys(run):
            results = run.get("results", []) or []
            if not results:
                return set()
            keys = set(results[0].get("bulk_properties", {}) or {})
            for r in results[1:]:
                keys &= set((r.get("bulk_properties") or {}).keys())
            return keys
        common_props = set.intersection(*(_bulk_keys(r) for r in grid_runs)) if grid_runs else set()
        prop_cols = sorted(common_props)
        cols = ["#", "ROI label", "Backend", "n unique assemblages"] + \
               [f"{p} mean" for p in prop_cols]
        stbl.setRowCount(len(grid_runs))
        stbl.setColumnCount(len(cols))
        stbl.setHorizontalHeaderLabels(cols)

        # Pre-compute per-run reshapes for the maps tabs
        def _grid_assemblages(run):
            results_dicts = run.get("results", []) or []
            n_p = (run.get("opts", {}) or {}).get("n_p", 0)
            n_t = (run.get("opts", {}) or {}).get("n_t", 0)
            assem = np.empty((n_p, n_t), dtype=object)
            for i, rd in enumerate(results_dicts):
                phases = frozenset(p.get("name", "")
                                    for p in rd.get("stable_phases", []))
                py = i // n_t
                px = i % n_t
                if py < n_p and px < n_t:
                    assem[py, px] = phases
            return assem, n_p, n_t

        def _grid_property(run, key):
            results_dicts = run.get("results", []) or []
            n_p = (run.get("opts", {}) or {}).get("n_p", 0)
            n_t = (run.get("opts", {}) or {}).get("n_t", 0)
            arr = np.full((n_p, n_t), np.nan, dtype=float)
            for i, rd in enumerate(results_dicts):
                v = (rd.get("bulk_properties") or {}).get(key)
                py = i // n_t
                px = i % n_t
                if v is not None and py < n_p and px < n_t:
                    arr[py, px] = float(v)
            return arr

        for i, run in enumerate(grid_runs):
            assem, _, _ = _grid_assemblages(run) if shared_grid else (None, 0, 0)
            n_unique = len({a for a in (assem.flat if assem is not None else [])
                             if a}) if shared_grid else "—"
            stbl.setItem(i, 0, QtWidgets.QTableWidgetItem(str(run.get("index", i + 1))))
            stbl.setItem(i, 1, QtWidgets.QTableWidgetItem(run.get("roi_label", "")))
            stbl.setItem(i, 2, QtWidgets.QTableWidgetItem((run.get("opts", {}) or {}).get("backend", "")))
            stbl.setItem(i, 3, QtWidgets.QTableWidgetItem(str(n_unique)))
            for j, p in enumerate(prop_cols):
                arr = _grid_property(run, p)
                v = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
                stbl.setItem(i, 4 + j, QtWidgets.QTableWidgetItem(
                    f"{v:.5g}" if np.isfinite(v) else "—"
                ))
        stbl.resizeColumnsToContents()
        stbl.horizontalHeader().setStretchLastSection(True)
        s_layout.addWidget(stbl)
        tabs.addTab(s_w, "Summary")

        # ── Tab: phase-field agreement (only if shared grid) ──────────────
        agree_fig = None
        if shared_grid:
            ag_w = QtWidgets.QWidget()
            ag_layout = QtWidgets.QVBoxLayout(ag_w)
            n_p = ref_opts.get("n_p", 0)
            n_t = ref_opts.get("n_t", 0)
            t_vals = np.linspace(ref_opts.get("t_range", (0, 0))[0],
                                 ref_opts.get("t_range", (0, 0))[1], n_t)
            p_vals = np.linspace(ref_opts.get("p_range", (0, 0))[0],
                                 ref_opts.get("p_range", (0, 0))[1], n_p)
            assemblages = [_grid_assemblages(r)[0] for r in grid_runs]
            ref_assem = assemblages[0]
            n_runs = len(assemblages)
            agree = np.zeros((n_p, n_t), dtype=float)
            for a in assemblages:
                agree += (a == ref_assem).astype(float)
            agree /= float(n_runs)

            agree_fig = plt.figure(figsize=(6.5, 5.0))
            ax = agree_fig.add_subplot(111)
            im = ax.pcolormesh(t_vals, p_vals, agree, cmap='RdYlGn',
                               vmin=0.0, vmax=1.0, shading='auto')
            agree_fig.colorbar(im, ax=ax,
                               label=f"fraction of {n_runs} runs matching run #{grid_runs[0].get('index','?')}")
            ax.set_xlabel("T [°C]")
            ax.set_ylabel("P [GPa]")
            ax.set_title("Phase-field agreement (vs first run)")
            agree_canvas = FigureCanvas(agree_fig)
            ag_layout.addWidget(NavigationToolbar(agree_canvas, dlg))
            ag_layout.addWidget(agree_canvas)
            tabs.addTab(ag_w, "Phase-field agreement")

            # ── Tab: property spread ──────────────────────────────────────
            spread_fig = None
            if prop_cols:
                sp_w = QtWidgets.QWidget()
                sp_layout = QtWidgets.QVBoxLayout(sp_w)
                ctrl_row = QtWidgets.QHBoxLayout()
                ctrl_row.addWidget(QtWidgets.QLabel("Property:"))
                cmb = QtWidgets.QComboBox()
                for p in prop_cols:
                    cmb.addItem(p)
                ctrl_row.addWidget(cmb)
                ctrl_row.addStretch()
                sp_layout.addLayout(ctrl_row)
                spread_fig = plt.figure(figsize=(6.5, 5.0))
                spread_canvas = FigureCanvas(spread_fig)
                sp_layout.addWidget(NavigationToolbar(spread_canvas, dlg))
                sp_layout.addWidget(spread_canvas)

                def _replot_spread():
                    spread_fig.clear()
                    key = cmb.currentText()
                    stack = np.stack([_grid_property(r, key)
                                       for r in grid_runs], axis=0)
                    sd = np.nanstd(stack, axis=0, ddof=0)
                    ax = spread_fig.add_subplot(111)
                    im = ax.pcolormesh(t_vals, p_vals, sd, cmap='magma',
                                       shading='auto')
                    spread_fig.colorbar(im, ax=ax, label=f"σ({key}) across runs")
                    ax.set_xlabel("T [°C]")
                    ax.set_ylabel("P [GPa]")
                    ax.set_title(f"Property spread — {key}")
                    spread_canvas.draw()

                cmb.currentTextChanged.connect(lambda _: _replot_spread())
                _replot_spread()
                tabs.addTab(sp_w, "Property spread")

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export CSV …")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        def _save_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save comparison CSV", "",
                "CSV (*.csv);;All files (*)"
            )
            if not path:
                return
            import pandas as pd
            rows_out = []
            for r in grid_runs:
                base = {
                    "index": r.get("index"),
                    "id": r.get("id"),
                    "roi_label": r.get("roi_label"),
                    "backend": (r.get("opts") or {}).get("backend"),
                }
                for p in prop_cols:
                    arr = _grid_property(r, p)
                    base[f"{p}_mean"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else None
                    base[f"{p}_std"] = float(np.nanstd(arr)) if np.isfinite(arr).any() else None
                rows_out.append(base)
            pd.DataFrame(rows_out).to_csv(path, index=False)
            QtWidgets.QMessageBox.information(dlg, "Saved", f"Wrote:\n{path}")

        btn_export.clicked.connect(_save_csv)
        dlg.setMinimumSize(720, 500)
        dlg.resize(900, 660)
        dlg.exec_()
        # Close any matplotlib figures we opened
        if agree_fig is not None:
            plt.close(agree_fig)

    # ---- Phase segmentation ------------------------------------------------

    def _collect_preview_layers(self) -> List[str]:
        """Return element names for which a '{el} preview' layer exists."""
        result = []
        for el in self.available_elements:
            if f"{el} preview" in self.viewer.layers:
                result.append(el)
        return result

    def on_segment_phases(self):
        """Button handler: segmentation → classification → optional thermod."""
        preview_els = self._collect_preview_layers()
        has_previews = len(preview_els) > 0
        has_fullres = self.last_band_maps is not None

        # Check for a drawn ROI directly from the Shapes layer — this is
        # independent of whether on_compute() has been run.
        drawn_roi = None
        try:
            drawn_roi, _ = self._get_roi_bbox_and_mask()
        except Exception:
            pass
        has_roi = drawn_roi is not None

        if not has_previews and not has_fullres:
            QtWidgets.QMessageBox.warning(
                self, "No data",
                "Load preview layers or compute full-resolution ROI maps first."
            )
            return

        # 1. Settings dialog
        fullres_els = sorted(self.last_band_maps.keys()) if self.last_band_maps else []
        dlg = _PhaseSegDialog(
            has_previews, has_roi, has_fullres,
            preview_elements=preview_els,
            fullres_elements=fullres_els,
            parent=self,
        )
        if not dlg.exec_():
            return
        opts = dlg.get_options()

        # 2. Determine which elements to use and the ROI
        source = opts["source"]
        # Decimation of the maps being segmented. For the full-res ROI source
        # this matches the band-map downsample (>1 when the large-ROI guard
        # decimated the ROI to fit RAM); clustering is resolution-agnostic and
        # the decimated weight map already lines up, so we segment the decimated
        # maps and render the phase map back at mosaic resolution via scale.
        seg_ds = 1
        if source == "fullres_roi":
            seg_ds = max(1, int(self.last_band_maps_ds))
            elements = sorted(self.last_band_maps.keys())
            roi_for_seg = self.last_roi
        else:
            elements = preview_els
            roi_for_seg = drawn_roi  # from Shapes layer, works without on_compute()

        # 3. Warn on large full-mosaic
        if source == "preview_full" and has_previews:
            sample = self.viewer.layers[f"{elements[0]} preview"].data
            est_bytes = sample.shape[0] * sample.shape[1] * len(elements) * 4
            if est_bytes > 500_000_000:
                ans = QtWidgets.QMessageBox.question(
                    self, "Large dataset",
                    f"The feature matrix will be ~{est_bytes // 1_000_000} MB.\n"
                    "This may take a while. Proceed?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if ans != QtWidgets.QMessageBox.Yes:
                    return

        # 4. Collect, build, normalize, cluster — with progress dialog
        pd = QtWidgets.QProgressDialog(
            "Preparing data ...", "Cancel", 0, 100, self
        )
        pd.setWindowTitle("Phase segmentation")
        pd.setWindowModality(QtCore.Qt.WindowModal)
        pd.setMinimumDuration(0)
        pd.setValue(0)
        _cancelled = [False]

        def _on_progress(step, total, msg):
            if pd.wasCanceled():
                _cancelled[0] = True
                raise KeyboardInterrupt("Cancelled by user")
            pct = int(10 + 90 * step / max(total, 1))
            pd.setValue(pct)
            pd.setLabelText(msg)
            QtWidgets.QApplication.processEvents()

        try:
            maps, translate = _collect_segment_maps(
                self.viewer, elements, source,
                roi_for_seg, self.last_band_maps,
            )

            # Manual element filter: applied before everything else so the user
            # has full control over which elements enter the pipeline.
            selected_els = opts.get("elements")
            if selected_els is not None:
                maps = {el: arr for el, arr in maps.items() if el in selected_els}
                if not maps:
                    raise ValueError(
                        "No elements remain after applying the manual element filter.\n"
                        "Select at least two elements in the segmentation dialog."
                    )
                print(f"[segment] Manual element filter: {sorted(maps.keys())}", flush=True)

            # Zero-signal (no-data) pixels: off-sample area / unscanned montage
            # margins. Judged on the RAW maps — smoothing bleeds sample counts
            # a few sigma into the void, so after it the void is no longer
            # zero. When the option is on these pixels are cleared and
            # protected from the nearest-neighbour fill below; otherwise
            # (legacy behaviour, and the fallback for pre-option recipes)
            # they are filled from the sample edge.
            raw_zero = None
            if opts.get("protect_background", False) and maps:
                raw_zero = _zero_signal_mask(maps)
                n_zero = int(raw_zero.sum())
                if n_zero:
                    print(f"[segment] Zero-signal (no-data) pixels: {n_zero:,} "
                          f"({100.0 * n_zero / raw_zero.size:.1f}% of the map) "
                          "— left unclassified", flush=True)
                else:
                    raw_zero = None

            # Auto-exclude sparse elements. Runs BEFORE smoothing on the raw
            # maps, but the criterion is smoothing-aware: an element is dropped
            # only when it is raw-sparse (zero fraction above the threshold)
            # AND still count-starved after the Gaussian pooling of ~4πσ²
            # pixels. A plain zero-fraction cut is a hidden count-rate cutoff
            # on low-dose maps (zero_frac ≈ exp(-rate)) and silently discarded
            # legitimately sparse discriminators like Mg/Na/Ca alongside real
            # contamination (see _sparse_element_exclusions).
            max_zero_pct = opts.get("max_zero_pct", 90.0)
            if max_zero_pct < 100.0 and maps:
                raw_total = np.stack(list(maps.values()), axis=-1).sum(axis=-1)
                valid_pixels = raw_total > 0
                n_valid = int(valid_pixels.sum())
                if n_valid > 0:
                    excluded_els, sparsity_report = _sparse_element_exclusions(
                        maps, valid_pixels,
                        max_zero_pct=max_zero_pct,
                        sigma=opts.get("smooth_sigma", 3.0),
                    )
                    print(f"[segment] Element sparsity check ({n_valid:,} non-background pixels):", flush=True)
                    for el, zf, pooled, drop in sparsity_report:
                        flag = " <-- EXCLUDED" if drop else (
                            "  (sparse, kept: usable after smoothing)"
                            if zf > max_zero_pct else "")
                        print(f"  {el:4s}: {zf:.1f}% near-zero, "
                              f"~{pooled:.1f} counts pooled at sigma{flag}", flush=True)
                    if excluded_els:
                        print(
                            f"[segment] Auto-excluded (>{max_zero_pct:.0f}% sparse "
                            f"AND count-starved after smoothing): {excluded_els}",
                            flush=True,
                        )
                        maps = {el: arr for el, arr in maps.items() if el not in excluded_els}
                    else:
                        print(f"[segment] No elements excluded (threshold {max_zero_pct:.0f}%)", flush=True)

            # Spatial smoothing — reduces per-pixel EDS noise
            sigma = opts.get("smooth_sigma", 3.0)
            if sigma > 0:
                pd.setValue(2)
                pd.setLabelText(f"Smoothing (sigma={sigma:.1f}) ...")
                QtWidgets.QApplication.processEvents()
                maps = _smooth_element_maps(maps, sigma=sigma)

            pd.setValue(4)
            pd.setLabelText("Building feature matrix ...")
            QtWidgets.QApplication.processEvents()

            # Compute tile-overlap mask
            overlap_mask = None
            if opts.get("exclude_overlap", True):
                if source == "fullres_roi" and self.last_weight_map is not None:
                    overlap_mask = _compute_overlap_mask_from_weight(self.last_weight_map)
                else:
                    first_map = next(iter(maps.values()))
                    ds = 1
                    if source != "fullres_roi":
                        first_lyr = self.viewer.layers[f"{elements[0]} preview"]
                        ds = int(round(first_lyr.scale[0]))
                    overlap_mask = _compute_overlap_mask_from_manifest(
                        self._get_manifest_rows(), first_map.shape,
                        roi=roi_for_seg, ds=ds,
                    )

            X, valid, shape, el_names = _build_feature_matrix(
                maps,
                min_count_pct=opts.get("min_count_pct", 0),
                min_count_ratio=opts.get("min_count_ratio", 0.1),
                overlap_mask=overlap_mask,
            )

            pd.setValue(6)
            pd.setLabelText("Normalizing ...")
            QtWidgets.QApplication.processEvents()

            X_norm = _normalize_features(
                X,
                norm_method=opts.get("norm_method", "row"),
            )

            pd.setValue(10)
            pd.setLabelText("Clustering ...")
            QtWidgets.QApplication.processEvents()

            labels_flat, k_chosen, bic_scores, icl_scores, auto_method = _run_clustering(
                X_norm,
                n_clusters=opts["n_clusters"],
                k_max=opts.get("k_max", 12),
                min_k=opts.get("min_k", 2),
                pca_variance=opts.get("pca_variance", 0.92),
                posterior_min=opts.get("reject_posterior_min", 0.0),
                density_pct=opts.get("reject_density_pct", 0.0),
                on_progress=_on_progress,
            )

            label_map = np.full(shape, -1, dtype=np.int16)
            label_map[valid] = labels_flat.astype(np.int16)
            # Pixels the GMM rejected (ambiguous posterior / atypical density)
            # stay unclassified: they were valid inputs that got label -1.
            rejected_mask = valid & (label_map < 0)
            if raw_zero is not None:
                # No-measurement pixels are never a phase, even where edge
                # smearing let them pass the low-count filter into the GMM.
                label_map[raw_zero] = -1
                protect_mask = rejected_mask | raw_zero
            else:
                protect_mask = rejected_mask

            # Post-clustering morphological cleanup
            min_size = opts.get("min_region_size", 50)
            if min_size > 0:
                pd.setValue(98)
                pd.setLabelText("Cleaning up small regions ...")
                QtWidgets.QApplication.processEvents()
                label_map = _clean_label_map(label_map, min_size=min_size)

            # Fill incidental holes (overlap zones, low-count, cleanup) via
            # nearest-neighbor; GMM-rejected and zero-signal pixels are
            # protected so they remain an honest "unclassified" class rather
            # than being silently reassigned to the nearest phase.
            _fill_unlabeled(label_map, protect=protect_mask)
            n_rej = int((label_map < 0).sum())
            if n_rej:
                print(
                    f"[segment] Unclassified: {n_rej:,} px "
                    f"({100.0 * n_rej / label_map.size:.1f}% of the map) — "
                    "excluded from modes, composition and mineral ID",
                    flush=True,
                )

            dominant = _cluster_dominant_elements(X_norm, labels_flat, el_names)
            pd.setValue(100)
        except KeyboardInterrupt:
            pd.close()
            return
        except Exception as exc:
            pd.close()
            if not _cancelled[0]:
                QtWidgets.QMessageBox.critical(
                    self, "Segmentation failed", str(exc)
                )
            return
        finally:
            pd.close()

        if auto_method:
            auto_str = f"auto-detected, {auto_method}"
        elif opts["n_clusters"] == "auto":
            auto_str = "auto-detected"
        else:
            auto_str = "manual"
        ds_note = f", 1/{seg_ds} res" if seg_ds > 1 else ""
        print(
            f"[segment] k={k_chosen} ({auto_str}) from {len(el_names)} elements "
            f"({shape[0]}×{shape[1]} px, source={source}{ds_note})",
            flush=True,
        )

        # Full replication recipe for this segmentation. The clustering is
        # seeded, so these inputs (plus the same cache) reproduce the phase map
        # exactly. seg_ds matters: the memory guard picks it from available RAM,
        # so it can silently differ between machines/runs.
        seg_params = {
            "source": source,
            "seg_ds": int(seg_ds),
            "roi": [int(v) for v in roi_for_seg] if roi_for_seg is not None else None,
            "elements_used": list(el_names),
            "element_filter": (sorted(opts["elements"])
                               if opts.get("elements") else None),
            "n_clusters": opts["n_clusters"],
            "k_chosen": int(k_chosen),
            "k_max": opts.get("k_max"),
            "min_k": opts.get("min_k"),
            "auto_method": auto_method,
            "pca_variance": float(opts.get("pca_variance", 0.92)),
            "smooth_sigma": float(opts.get("smooth_sigma", 3.0)),
            "min_region_size": int(opts.get("min_region_size", 50)),
            "norm_method": opts.get("norm_method", "row"),
            "min_count_pct": float(opts.get("min_count_pct", 0)),
            "min_count_ratio": float(opts.get("min_count_ratio", 0.1)),
            "max_zero_pct": float(opts.get("max_zero_pct", 90.0)),
            "exclude_overlap": bool(opts.get("exclude_overlap", True)),
            "reject_posterior_min": float(opts.get("reject_posterior_min", 0.0)),
            "reject_density_pct": float(opts.get("reject_density_pct", 0.0)),
            "protect_background": bool(opts.get("protect_background", False)),
            # net-count background the features were built from
            # (eds_background.SCHEMES; recipes before 2026-09-16 were "sideband")
            "background": _bg_scheme(),
            # smoothed signed nets get the per-element noise level added
            # (viewer_segment_helpers._smooth_element_maps); 0 for legacy nets
            "noise_floor": _bg_scheme() != "sideband",
        }
        import json as _json
        print("[segment] recipe: " + _json.dumps(seg_params, default=str),
              flush=True)

        # 5-6. Store phase map + open the classification dialog. Mineral ID needs
        # the phase map pixel-aligned with the band maps — true for the full-res
        # ROI source at any decimation (both arrays native or both 1/ds); preview
        # sources are not aligned and lack sidebands, so stay excluded.
        mineral_aligned = (source == "fullres_roi" and self.last_band_maps is not None)
        self._finalize_phase_map(
            label_map, dominant, el_names, k_chosen, translate, seg_ds,
            mineral_aligned=mineral_aligned, phase_source="gmm",
            bic_scores=bic_scores, icl_scores=icl_scores, auto_method=auto_method,
            seg_params=seg_params,
        )

    def _finalize_phase_map(self, label_map, dominant, el_names, n_clusters,
                            translate, seg_ds, mineral_aligned,
                            phase_source="gmm", bic_scores=None,
                            icl_scores=None, auto_method=None,
                            prefill_names=None, seg_params=None):
        """Shared tail for the GMM and EBSD phase-domain paths.

        Stores the phase-map state, opens ``_PhaseClassDialog`` (with the band
        maps when ``mineral_aligned`` so mineral ID / composition breakdown work),
        harvests names / mineral IDs / per-phase composition, and launches the
        thermod P-T flow if the dialog was accepted. ``prefill_names`` seeds the
        editable Name column (e.g. crystallographic names from EBSD); the
        mineral-ID auto-namer never clobbers a pre-filled name.
        """
        self.last_phase_map = label_map
        self.last_phase_translate = translate
        self.last_phase_ds = seg_ds
        self.last_phase_source = phase_source
        # Set unconditionally: the EBSD path passes None, which correctly
        # invalidates any recipe left over from an earlier GMM run.
        self.last_phase_seg_params = seg_params
        prefill_names = prefill_names or {}

        def _default_name(cid):
            return prefill_names.get(cid, f"Phase {cid + 1}")
        self.last_phase_names = {cid: _default_name(cid)
                                 for cid in sorted(dominant.keys())}

        cls_dlg = _PhaseClassDialog(
            label_map, el_names, dominant, n_clusters,
            translate, self.viewer, parent=self,
            bic_scores=bic_scores,
            icl_scores=icl_scores,
            auto_method=auto_method,
            band_maps=self.last_band_maps if mineral_aligned else None,
            data_type=self._effective_data_type(),
            k_factors=(self.eds_calibration.k() if self.eds_calibration else None),
            phase_aligned=mineral_aligned,
            scale=(float(seg_ds), float(seg_ds)),
            coating_elements=self._coating_elements(),
            seg_params=seg_params,
        )
        # Pre-fill the editable Name column (EBSD crystallographic names).
        for row, cid in enumerate(cls_dlg._clusters):
            if cid in prefill_names:
                item = cls_dlg._table.item(row, 2)
                if item is not None:
                    item.setText(str(prefill_names[cid]))
        result = cls_dlg.exec_()

        # Store user-edited names regardless of accept/reject.
        name_map = {}
        for row, cid in enumerate(cls_dlg._clusters):
            item = cls_dlg._table.item(row, 2)
            if item and item.text().strip():
                name_map[cid] = item.text().strip()
        if name_map:
            self.last_phase_names = name_map

        # Harvest mineral-ID results (if the user ran "Identify minerals …").
        self.last_phase_minerals = cls_dlg.mineral_assignments()
        self.last_phase_fe3 = cls_dlg.fe3_value()
        self.last_phase_comp_erosion = cls_dlg.erosion_value()
        self.last_phase_unmix = cls_dlg.unmix_full()
        # A coating corrected in the dialog sticks for later k-factor derivations
        # and dialog re-opens (None → fall back to the manifest/default).
        self._coating_override = cls_dlg.coating_value()
        # Keep the full per-phase composition so it's accessible after the dialog
        # closes (panel "Phase composition …" button) and persists in the session.
        self.last_phase_composition = cls_dlg.domain_compositions()
        self._harvest_pt_overrides(cls_dlg)
        self._refresh_phase_comp_button()

        if result == QtWidgets.QDialog.Accepted:
            # "Use for P-T estimate" was clicked. Prefer the mineral-ID
            # assemblage (thermo keys) over raw display names, and carry the
            # observed modal proportions for the misfit view.
            phase_names = cls_dlg.phase_names
            if phase_names:
                info = self._assemblage_inputs_from_state(
                    self.last_phase_minerals, self.last_phase_names)
                self._launch_thermod_from_phases(
                    info["observed"] or phase_names,
                    observed_modes=info["modes"] or None, misfit_opts=info)
        return cls_dlg

    def _refresh_phase_comp_button(self):
        """Back-compat shim: refresh all dataset-dependent control states
        (the Phase-composition button among them)."""
        self._refresh_control_availability()

    def on_show_phase_composition(self):
        """Re-open the per-phase composition breakdown after the segmentation
        window has been closed (uses the stored composition + phase map)."""
        if not self.last_phase_composition:
            QtWidgets.QMessageBox.information(
                self, "No phase composition",
                "Run a full-resolution phase segmentation first — the per-phase "
                "composition is captured when the classification dialog closes.")
            return
        names = self.last_phase_names or {}
        names = {int(k): v for k, v in names.items()}
        npx = {}
        if self.last_phase_map is not None:
            for cid in self.last_phase_composition:
                npx[cid] = int(np.sum(self.last_phase_map == cid))
        import mineral_id as _mid
        units = "wt%" if self._effective_data_type() in _mid.QUANTIFIED_TYPES else "net-count %"
        _PhaseCompositionDialog(
            self.last_phase_composition, npx,
            {cid: names.get(cid, f"Phase {cid + 1}") for cid in self.last_phase_composition},
            units, parent=self).exec_()

    def on_reopen_phase_classification(self):
        """Re-open the phase classification dialog from the stored segmentation
        state (e.g. after a session load), without re-clustering.

        Restored names / mineral suggestions are pre-filled; mineral ID and the
        composition breakdown are available again when the band maps still share
        the phase map's grid. Stored mineral assignments are only overwritten
        when the user re-runs *Identify minerals*.
        """
        if self.last_phase_map is None:
            QtWidgets.QMessageBox.information(
                self, "No phase map",
                "Run 'Segment phases …' (or load a session that contains a "
                "segmentation) first.")
            return
        label_map = self.last_phase_map
        cids = sorted(int(c) for c in np.unique(label_map) if c >= 0)
        if not cids:
            QtWidgets.QMessageBox.information(
                self, "No phase map", "The stored phase map has no valid domains.")
            return

        # Dominant-elements column: the GMM z-score ranking isn't persisted, so
        # the stored per-phase composition's top-3 is the honest stand-in.
        comp = {int(k): v for k, v in (self.last_phase_composition or {}).items()}
        dominant = {}
        for cid in cids:
            top = sorted((comp.get(cid) or {}).items(), key=lambda kv: -kv[1])[:3]
            dominant[cid] = ", ".join(el for el, _ in top)

        # Mineral ID / composition need the phase map on the band-map grid.
        aligned = False
        if self.last_band_maps:
            sample = next(iter(self.last_band_maps.values()))["main"]
            aligned = (label_map.shape == sample.shape
                       and int(self.last_phase_ds) == int(self.last_band_maps_ds))

        seg_ds = max(1, int(self.last_phase_ds))
        names = {int(k): v for k, v in (self.last_phase_names or {}).items()}
        minerals = {int(k): v for k, v in (self.last_phase_minerals or {}).items()}

        cls_dlg = _PhaseClassDialog(
            label_map, list(self.last_elements or []), dominant, len(cids),
            self.last_phase_translate or (0.0, 0.0), self.viewer, parent=self,
            band_maps=self.last_band_maps if aligned else None,
            data_type=self._effective_data_type(),
            k_factors=(self.eds_calibration.k() if self.eds_calibration else None),
            phase_aligned=aligned,
            scale=(float(seg_ds), float(seg_ds)),
            coating_elements=self._coating_elements(),
            seg_params=self.last_phase_seg_params,
        )
        if self.last_phase_fe3 is not None:
            try:
                cls_dlg._fe3_spin.setValue(float(self.last_phase_fe3))
            except (TypeError, ValueError):
                pass
        for row, cid in enumerate(cls_dlg._clusters):
            if cid in names:
                item = cls_dlg._table.item(row, 2)
                if item is not None:
                    item.setText(str(names[cid]))
            m = minerals.get(cid)
            if m:
                cands = m.get("candidates") or []
                txt = " · ".join(
                    f"{c.get('name')} {float(c.get('confidence_pct') or 0):.0f}%"
                    for c in cands[:3]) or str(m.get("name") or "—")
                item = cls_dlg._table.item(row, 1)
                if item is not None:
                    item.setText(txt)
                # Restore a manual override so it survives the round-trip.
                if m.get("method") == "manual" and m.get("name"):
                    cls_dlg.set_override(cid, m.get("name"))
        # Restore the P-T inversion controls (include / weights / extra / open).
        if hasattr(cls_dlg, "set_pt_overrides"):
            cls_dlg.set_pt_overrides(
                include=self.last_phase_pt_include,
                weights=self.last_phase_pt_weights,
                extra=self.last_phase_extra,
                open_tol=self.last_phase_open_tol)
        if self.last_phase_comp_erosion:
            cls_dlg.set_erosion(self.last_phase_comp_erosion)
        if self.last_phase_unmix and hasattr(cls_dlg, "set_unmix"):
            cls_dlg.set_unmix(self.last_phase_unmix)
        result = cls_dlg.exec_()

        # Harvest conservatively: user edits win, an untouched dialog must not
        # clobber the restored state (mineral_assignments() is {} until the user
        # re-runs Identify).
        name_map = {}
        for row, cid in enumerate(cls_dlg._clusters):
            item = cls_dlg._table.item(row, 2)
            if item and item.text().strip():
                name_map[cid] = item.text().strip()
        if name_map:
            self.last_phase_names = name_map
        new_minerals = cls_dlg.mineral_assignments()
        if new_minerals:
            # Merge: overriding (or re-identifying) only some domains must not drop
            # the other restored assignments. An untouched dialog returns {}.
            self.last_phase_minerals = {**(self.last_phase_minerals or {}),
                                        **new_minerals}
            self.last_phase_fe3 = cls_dlg.fe3_value()
            self._coating_override = cls_dlg.coating_value()
        # A domain merge removes cluster ids from the map — prune restored
        # assignments for ids that no longer exist so they can't leak into the
        # assemblage / session.
        _valid_cids = set(cls_dlg._clusters)
        if self.last_phase_minerals:
            self.last_phase_minerals = {c: m for c, m in self.last_phase_minerals.items()
                                        if c in _valid_cids}
        self.last_phase_comp_erosion = cls_dlg.erosion_value()
        # set_unmix pruned entries whose domains no longer exist; the dialog
        # state (restored + user edits) is authoritative on close.
        self.last_phase_unmix = cls_dlg.unmix_full()
        new_comp = cls_dlg.domain_compositions()
        if new_comp:
            self.last_phase_composition = new_comp
        self._harvest_pt_overrides(cls_dlg)
        self._refresh_control_availability()

        if result == QtWidgets.QDialog.Accepted:
            phase_names = cls_dlg.phase_names
            if phase_names:
                info = self._assemblage_inputs_from_state(
                    self.last_phase_minerals, self.last_phase_names)
                self._launch_thermod_from_phases(
                    info["observed"] or phase_names,
                    observed_modes=info["modes"] or None, misfit_opts=info)

    def on_use_ebsd_phases(self):
        """Use the imported EBSD phase map as ready-made domains for the ROI.

        Crops the full-mosaic EBSD phase map to the active ROI, aligns it to the
        computed EDS band maps (same bbox, same 1/ds decimation), and opens the
        classification dialog with crystallographic names pre-filled — so the
        per-phase composition breakdown works and the EDS ratio mineral ID runs
        as a cross-check of the EBSD truth. GMM segmentation remains available
        for the unindexed area.
        """
        if self.ebsd_phase_map is None:
            QtWidgets.QMessageBox.information(
                self, "No EBSD phases",
                "This dataset has no EBSD phase map. Load an Oxford h5oina export "
                "(it writes ebsd_phase_map.npy alongside the EDS cache).")
            return
        if not self.last_band_maps or not self.last_roi:
            QtWidgets.QMessageBox.information(
                self, "Compute an ROI first",
                "Draw an ROI and click 'Compute EDS maps' first. The EBSD phases "
                "are cropped to that ROI and aligned to the EDS band maps.")
            return

        y0, y1, x0, x1 = self.last_roi
        ds = max(1, int(self.last_band_maps_ds))
        crop = self.ebsd_phase_map[y0:y1, x0:x1]
        # Honour a non-rectangular ROI: zero (unindex) pixels outside the polygon
        # mask. Apply at full resolution (before decimating), so use the full-res
        # mask — last_roi_mask is decimated when ds>1; last_roi_mask_fullres holds
        # the native-res copy in that case.
        roi_mask_full = (self.last_roi_mask_fullres
                         if self.last_roi_mask_fullres is not None
                         else self.last_roi_mask)
        if roi_mask_full is not None and \
                np.asarray(roi_mask_full).shape == crop.shape:
            crop = np.where(np.asarray(roi_mask_full, dtype=bool), crop, 0)
        # Nearest-pixel decimate to match the band-map grid (which the cache
        # produced by the same [::ds] striding on the same bbox).
        lab_full = np.ascontiguousarray(crop[::ds, ::ds])
        sample = next(iter(self.last_band_maps.values()))['main']
        Hb, Wb = sample.shape[:2]
        lab_full = lab_full[:Hb, :Wb]
        if lab_full.shape != (Hb, Wb):
            QtWidgets.QMessageBox.critical(
                self, "EBSD / EDS misalignment",
                f"Decimated EBSD crop {lab_full.shape} does not match the EDS "
                f"band-map grid {(Hb, Wb)}. Recompute the ROI and retry.")
            return

        present = [int(o) for o in np.unique(lab_full).tolist() if int(o) != 0]
        if not present:
            QtWidgets.QMessageBox.information(
                self, "No indexed EBSD phases",
                "The selected ROI contains only unindexed EBSD pixels. Use "
                "'Segment phases …' (GMM on EDS) for this region instead.")
            return

        # Remap EBSD ordinals → contiguous 0-based cluster ids (matches the GMM
        # convention; 0 = unindexed → -1 background).
        remap = {o: i for i, o in enumerate(present)}
        label_map = np.full(lab_full.shape, -1, dtype=np.int16)
        for o, cid in remap.items():
            label_map[lab_full == o] = cid

        legend = self.ebsd_phase_legend or {}
        prefill_names = {}
        for o, cid in remap.items():
            ent = legend.get(o) or legend.get(str(o)) or {}
            prefill_names[cid] = ent.get("name", f"EBSD {o}")

        # Dominant elements per phase from the band maps (for the table column).
        el_names = sorted(self.last_band_maps.keys())
        dominant = {}
        for cid in range(len(present)):
            frac, _ = _net_count_fractions(
                self.last_band_maps, sub_mask=(label_map == cid))
            top = sorted(frac.items(), key=lambda kv: kv[1], reverse=True)[:3]
            dominant[cid] = ", ".join(el for el, _ in top)

        ds_note = f", 1/{ds} res" if ds > 1 else ""
        print(f"[ebsd] using {len(present)} EBSD phase(s) over ROI "
              f"{self.last_roi}{ds_note}: "
              f"{[prefill_names[c] for c in range(len(present))]}", flush=True)

        self._finalize_phase_map(
            label_map, dominant, el_names, len(present),
            (float(y0), float(x0)), ds,
            mineral_aligned=True, phase_source="ebsd",
            prefill_names=prefill_names,
        )

    def on_ebsd_confusion(self):
        """Validate the EDS phase map against the EBSD ground truth.

        Crops the EBSD ordinal map onto the phase-map grid (same alignment as
        'Use EBSD phases') and cross-tabulates per pixel: confusion matrix,
        alias-matched accuracy, per-phase recall/precision, CSV export.
        """
        import ebsd_validation as ev

        if self.ebsd_phase_map is None:
            QtWidgets.QMessageBox.information(
                self, "No EBSD phases",
                "This dataset has no EBSD phase map (Oxford h5oina only).")
            return
        if self.last_phase_map is None or not self.last_roi:
            QtWidgets.QMessageBox.information(
                self, "No phase map",
                "Segment phases for an ROI first (or load a session that "
                "contains a segmentation).")
            return
        # The comparison needs the phase map on the computed-ROI grid: its
        # translate must be the ROI origin (preview-sourced segmentations use
        # a different grid and would silently misalign).
        y0, y1, x0, x1 = self.last_roi
        tr = self.last_phase_translate or (None, None)
        if tr[0] is None or (float(tr[0]), float(tr[1])) != (float(y0), float(x0)):
            QtWidgets.QMessageBox.warning(
                self, "Phase map not ROI-aligned",
                "The stored phase map was not segmented on the computed ROI "
                "grid (preview-sourced segmentation?). Re-run 'Segment phases "
                "…' with the ROI full-resolution source.")
            return

        ds = max(1, int(self.last_phase_ds))
        roi_mask_full = (self.last_roi_mask_fullres
                         if self.last_roi_mask_fullres is not None
                         else self.last_roi_mask)
        eb = ev.crop_ebsd_to_phase_grid(
            self.ebsd_phase_map, self.last_roi, ds,
            self.last_phase_map.shape, roi_mask_full)
        if eb is None:
            QtWidgets.QMessageBox.critical(
                self, "EBSD / EDS misalignment",
                "The decimated EBSD crop does not match the phase-map grid. "
                "Recompute the ROI and re-segment, then retry.")
            return

        legend = self.ebsd_phase_legend or {}
        ebsd_names = {}
        for o in np.unique(eb[eb > 0]):
            ent = legend.get(int(o)) or legend.get(str(int(o))) or {}
            ebsd_names[int(o)] = ent.get("name", f"EBSD {int(o)}")
        eds_names = {int(k): str(v)
                     for k, v in (self.last_phase_names or {}).items()}

        res = ev.confusion_matrix(eb, self.last_phase_map, ebsd_names, eds_names)
        if res.n_both == 0:
            QtWidgets.QMessageBox.information(
                self, "No overlap",
                "No pixel is both EBSD-indexed and inside an EDS phase domain "
                "in this ROI.")
            return
        self._show_confusion_dialog(res)

    def _show_confusion_dialog(self, res):
        """Confusion matrix table (counts + row %) with matched cells shaded."""
        import ebsd_validation as ev

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("EBSD vs EDS confusion matrix")
        layout = QtWidgets.QVBoxLayout(dlg)

        acc = f"{100 * res.accuracy:.1f}%" if res.accuracy is not None else "—"
        hdr = (f"<b>Alias-matched accuracy: {acc}</b> over {res.n_both:,} px "
               f"valid in both maps  ·  EBSD-only {res.n_ebsd_only:,} px, "
               f"EDS-only {res.n_eds_only:,} px")
        lbl = QtWidgets.QLabel(hdr)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        if getattr(self, "last_phase_source", "gmm") == "ebsd":
            warn = QtWidgets.QLabel(
                "⚠ The current phase map was built FROM the EBSD phases — this "
                "comparison is circular. Re-segment with GMM ('Segment phases "
                "…') for an independent validation.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #b71c1c;")
            layout.addWidget(warn)

        tbl = QtWidgets.QTableWidget()
        R, C = res.counts.shape
        tbl.setRowCount(R)
        tbl.setColumnCount(C)
        tbl.setHorizontalHeaderLabels(res.col_labels)
        tbl.setVerticalHeaderLabels(res.row_labels)
        rownorm = res.row_normalised()
        for i in range(R):
            hits = set(res.matched_cols.get(i, []))
            for j in range(C):
                item = QtWidgets.QTableWidgetItem(
                    f"{int(res.counts[i, j]):,}\n({100 * rownorm[i, j]:.1f}%)")
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                if j in hits:
                    # matched (should-be-diagonal) cell: green, darker = fuller
                    g = int(120 + 100 * min(1.0, rownorm[i, j]))
                    item.setBackground(QtGui.QColor(120, g, 120, 110))
                tbl.setItem(i, j, item)
        tbl.resizeColumnsToContents()
        tbl.resizeRowsToContents()
        layout.addWidget(tbl)

        lines = []
        for r in res.per_row:
            rec = ("n/a (no matching EDS domain)" if r["recall"] is None
                   else f"{100 * r['recall']:.1f}%")
            lines.append(f"recall {r['label']}: {rec}  (n={r['n']:,})")
        for c in res.per_col:
            pre = ("n/a (no matching EBSD phase)" if c["precision"] is None
                   else f"{100 * c['precision']:.1f}%")
            lines.append(f"precision {c['label']}: {pre}  (n={c['n']:,})")
        det = QtWidgets.QLabel("\n".join(lines))
        det.setStyleSheet("font-size: 11px; color: #444;")
        layout.addWidget(det)

        btn_row = QtWidgets.QHBoxLayout()
        btn_csv = QtWidgets.QPushButton("Export CSV …")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_csv)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        def _save():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Export confusion matrix CSV",
                "ebsd_confusion.csv", "CSV files (*.csv)")
            if not path:
                return
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(ev.confusion_to_csv_rows(res))
        btn_csv.clicked.connect(_save)

        dlg.resize(min(220 + 130 * C, 1100), 520)
        dlg.exec_()

    def _pixel_um(self) -> Optional[float]:
        """Mosaic pixel pitch in µm.

        Resolution order: session override (set in the composition-vs-scale
        dialog) > cache manifest ``pixel_um`` > None. Not part of the cache
        fingerprint, so an override never invalidates the cache."""
        if self._pixel_um_override is not None:
            return float(self._pixel_um_override)
        try:
            import eds_cache as _ec
            man = _ec.load_manifest(_ec.resolve_cache_dir(self.manifest.parent))
            v = (man or {}).get("pixel_um")
            return float(v) if v else None
        except Exception:
            return None

    def _coating_elements(self) -> frozenset:
        """Coating / contaminant element symbols to ignore in mineral ID and
        SPX calibration. Resolution order: session override (set in the phase
        dialog) > cache manifest ``coating_elements`` > default ``{"C"}``.

        Not part of the cache fingerprint, so changing it never invalidates the
        cache (mirrors :meth:`_pixel_um`)."""
        if self._coating_override is not None:
            return self._coating_override
        try:
            import eds_cache as _ec
            man = _ec.load_manifest(_ec.resolve_cache_dir(self.manifest.parent))
            co = (man or {}).get("coating_elements")
            return frozenset(co) if co is not None else frozenset({"C"})
        except Exception:
            return frozenset({"C"})

    def on_zoning(self):
        """Open the modeless zoning / traverse tool for the computed ROI maps.

        Per-pixel index maps (Mg#, An, …) + line profiles; calibrated when SPX
        k-factors are active, intensity-ratio otherwise. The phase map is
        offered for domain restriction only when it shares the band-map grid.
        """
        if not self.last_band_maps or not self.last_roi:
            QtWidgets.QMessageBox.information(
                self, "Zoning / traverse",
                "Compute the FULL-RES EDS maps for an ROI first.")
            return
        y0, y1, x0, x1 = self.last_roi
        ds = max(1, int(self.last_band_maps_ds))

        # Offer domain restriction only for an ROI-aligned phase map.
        phase_map = None
        phase_names = None
        if self.last_phase_map is not None:
            sample = next(iter(self.last_band_maps.values()))['main']
            tr = self.last_phase_translate or (None, None)
            if (tr[0] is not None
                    and (float(tr[0]), float(tr[1])) == (float(y0), float(x0))
                    and int(self.last_phase_ds) == ds
                    and self.last_phase_map.shape == sample.shape[:2]):
                phase_map = self.last_phase_map
                phase_names = {
                    int(c): str(n)
                    for c, n in (self.last_phase_names or {}).items()}

        quantified = (self._cache_data_type() == "oxford_wt_pct")
        k = self.eds_calibration.k() if self.eds_calibration else None

        # Modeless: the user draws the traverse in napari while it is open.
        if getattr(self, "_zoning_dlg", None) is not None:
            try:
                self._zoning_dlg.close()
            except Exception:
                pass
        self._zoning_dlg = _ZoningDialog(
            self.last_band_maps, k, (float(y0), float(x0)), ds, self.viewer,
            roi_mask=self.last_roi_mask,
            phase_map=phase_map, phase_names=phase_names,
            pixel_um=self._pixel_um(), quantified=quantified, parent=self)
        self._zoning_dlg.show()

    def _harvest_pt_overrides(self, cls_dlg):
        """Capture the phase dialog's P-T inversion controls onto the panel
        (include-set, per-domain weights, extra phases, open-assemblage tol)."""
        for attr, meth in (("last_phase_pt_include", "included_domains"),
                           ("last_phase_pt_weights", "domain_weights"),
                           ("last_phase_extra", "extra_observed"),
                           ("last_phase_open_tol", "open_assemblage_tol")):
            fn = getattr(cls_dlg, meth, None)
            if callable(fn):
                try:
                    setattr(self, attr, fn())
                except Exception:
                    pass

    @staticmethod
    def _confidence_weight(m: dict) -> float:
        """Map a mineral-ID match dict to a 0.1–1.0 reliability weight.

        Prefers the independent ``confidence_pct``; falls back to the coarse
        confidence label. A manual override (confidence "high", pct 100) → 1.0.
        """
        pct = m.get("confidence_pct")
        if pct is not None:
            try:
                return max(0.1, min(1.0, float(pct) / 100.0))
            except (TypeError, ValueError):
                pass
        return {"high": 1.0, "moderate": 0.7, "low": 0.4,
                "approximate": 0.5}.get(m.get("confidence"), 0.6)

    def _assemblage_inputs_from_state(self, minerals_by_cid, names_by_cid):
        """Observed assemblage + modal proportions + misfit-tuning from state.

        Per domain, the observed key is the mineral-ID ``thermo_abbrev`` when
        the top match is at least *moderate* confidence; otherwise the
        user/EBSD display name ("Phase N" placeholders excluded — the
        alias-aware matcher handles full mineral names like "Diopside").

        Honours the optional per-ROI P-T overrides set in the phase dialog and
        stored on the panel: ``last_phase_pt_include`` (cids to keep; None=all),
        ``last_phase_pt_weights`` (per-cid reliability override), ``last_phase_extra``
        (phases segmentation missed — each ``{key, mode%?, weight?}``; mode-less
        entries are presence-only), and ``last_phase_open_tol`` (open-assemblage
        forgiveness).

        Returns a dict: ``observed`` (proportion keys — the strict stability
        assemblage), ``modes`` (proportion modes summing to 1, with any
        user-mode extras folded in), ``excluded`` (unkeyed/excluded fraction),
        ``weights`` (per-key reliability), ``presence_keys`` (present-but-no-mode
        extras), and ``open_other_tol``. Empties when nothing is usable.
        """
        import re
        import mineral_id as _mid
        floor = _mid._CONFIDENCE_RANK.get("moderate", 2)
        include = getattr(self, "last_phase_pt_include", None)        # None = all
        cid_w = getattr(self, "last_phase_pt_weights", None) or {}
        extra = getattr(self, "last_phase_extra", None) or []
        open_tol = float(getattr(self, "last_phase_open_tol", 0.0) or 0.0)
        empty = {"observed": [], "modes": {}, "excluded": 1.0,
                 "weights": {}, "presence_keys": [], "open_other_tol": open_tol}

        key_by_cid: Dict[int, str] = {}
        wt_by_cid: Dict[int, float] = {}
        cids = {int(c) for c in (minerals_by_cid or {})} | \
               {int(c) for c in (names_by_cid or {})}
        for cid in cids:
            if include is not None and cid not in include:
                continue
            m = (minerals_by_cid or {}).get(cid)
            key, cw = None, 1.0
            if m and m.get("thermo_abbrev") and \
                    _mid._CONFIDENCE_RANK.get(m.get("confidence"), 0) >= floor:
                key = str(m["thermo_abbrev"])
                cw = self._confidence_weight(m)
            else:
                nm = str((names_by_cid or {}).get(cid, "")).strip()
                if nm and not re.fullmatch(r"Phase \d+", nm):
                    key = nm                       # display name / EBSD = trusted
            if key:
                key_by_cid[cid] = key
                wt_by_cid[cid] = float(cid_w.get(cid, cid_w.get(str(cid), cw)))

        modes_seg: Dict[str, float] = {}
        excluded = 1.0
        weights: Dict[str, float] = {}
        # Mixed-domain unmixing (dialog "Unmix selected …"): the source
        # domain's pixels are redistributed onto the endmember domains'
        # keys instead of counted under its own.
        unmix = {int(c): (d.get("fractions") or {})
                 for c, d in (getattr(self, "last_phase_unmix", None) or {}).items()
                 if d.get("fractions")}
        if self.last_phase_map is not None and key_by_cid:
            modes_seg, excluded = _observed_modes_from_phase_map(
                self.last_phase_map, key_by_cid, unmix=unmix or None)
            # Per-key reliability = pixel-weighted mean of its domains' weights.
            # Unmixed sources are skipped: their pixels no longer contribute to
            # their own key's mode, so they must not steer its weight either.
            pm = np.asarray(self.last_phase_map)
            cu, cc = np.unique(pm[pm >= 0], return_counts=True)
            px = {int(c): int(n) for c, n in zip(cu, cc)}
            agg: Dict[str, List[float]] = {}
            for cid, key in key_by_cid.items():
                if cid in unmix:
                    continue
                n = max(1, px.get(cid, 0))
                a = agg.setdefault(key, [0.0, 0.0])
                a[0] += wt_by_cid[cid] * n
                a[1] += n
            weights = {k: (s / c if c > 0 else 1.0) for k, (s, c) in agg.items()}

        # Fold in user-added phases segmentation missed.
        extra_modes: Dict[str, float] = {}
        presence: List[str] = []
        for e in extra:
            k = str(e.get("key", "")).strip()
            if not k:
                continue
            ew = float(e.get("weight", 1.0))
            md = e.get("mode")
            if md is None or float(md) <= 0:
                presence.append(k)
            else:
                extra_modes[k] = extra_modes.get(k, 0.0) + float(md) / 100.0
            weights[k] = ew

        m_extra = sum(extra_modes.values())
        if m_extra >= 1.0:                     # user-specified extras dominate
            modes = {k: v / m_extra for k, v in extra_modes.items()}
        else:
            modes = {k: v * (1.0 - m_extra) for k, v in modes_seg.items()}
            for k, v in extra_modes.items():
                modes[k] = modes.get(k, 0.0) + v
        presence = [k for k in presence if k not in modes]

        if not modes and not presence:
            return empty
        return {"observed": sorted(modes), "modes": modes, "excluded": excluded,
                "weights": weights, "presence_keys": presence,
                "open_other_tol": open_tol}

    def _launch_thermod_from_phases(self, phase_names: List[str],
                                    observed_modes: Optional[Dict[str, float]] = None,
                                    misfit_opts: Optional[dict] = None):
        """Open thermod dialog pre-filled with observed phases for P-T estimation.

        ``observed_modes`` (``{observed key: modal fraction}``, from the phase
        map's pixel counts) rides along in the run opts so the grid result
        dialog can offer the **Assemblage misfit** view (observed vs predicted
        modal proportions per P-T cell). ``misfit_opts`` (from
        ``_assemblage_inputs_from_state``) carries the per-key reliability
        weights, presence-only phases, and open-assemblage tolerance — all
        stamped into the run opts so they persist and reach every scorer.
        """
        # Resolve composition: SPX > SPX-calibrated EDS > raw net-count fractions.
        comp_el_roi, source = self._preferred_roi_composition()

        dlg = _ThermoRunDialog(comp_el_roi, source, parent=self)
        # Pre-fill for estimate workflow
        dlg.rb_both.setChecked(True)
        dlg.rb_assem_now.setChecked(True)
        dlg.assem_edit.setText(", ".join(phase_names))
        dlg._update_workflow_widgets()

        if not dlg.exec_():
            return
        opts = dlg.get_options()
        if observed_modes:
            opts["observed_modes"] = {str(k): float(v)
                                      for k, v in observed_modes.items()}
        mo = misfit_opts or {}
        if mo.get("weights"):
            opts["modal_weights"] = {str(k): float(v)
                                     for k, v in mo["weights"].items()}
        if mo.get("presence_keys"):
            opts["presence_keys"] = [str(k) for k in mo["presence_keys"]]
        if mo.get("open_other_tol"):
            opts["open_other_tol"] = float(mo["open_other_tol"])
        # Record how the observed phase map was produced, so the run record
        # alone is enough to replicate the segmentation behind the assemblage.
        if self.last_phase_seg_params:
            opts["segmentation_params"] = dict(self.last_phase_seg_params)
        if getattr(self, "last_phase_unmix", None):
            # The observed_modes above already carry the redistribution; this
            # records HOW they were derived from the phase map.
            opts.setdefault("segmentation_params", {})
            opts["segmentation_params"]["unmix"] = {str(c): {
                "fractions": {str(e): float(f)
                              for e, f in (d.get("fractions") or {}).items()},
                "residual": float(d.get("residual", float("nan"))),
            } for c, d in self.last_phase_unmix.items()}

        # Reuse steps 3-5 from on_run_thermod
        comp_el = opts["comp_el"]
        if not comp_el:
            QtWidgets.QMessageBox.critical(
                self, "No composition",
                "No composition is available. Enter a custom oxide wt% composition."
            )
            return

        try:
            from thermo import ThermoCalc, Composition
        except ImportError as exc:
            QtWidgets.QMessageBox.critical(
                self, "thermo package not found",
                f"Could not import the thermo package:\n{exc}"
            )
            return

        opts.setdefault("extra_basis", "bulk")        # see the ROI run path
        extra: Dict[str, float] = {}
        if opts["h2o"] > 0:
            extra["H2O"] = opts["h2o"]
        if opts["co2"] > 0:
            extra["CO2"] = opts["co2"]

        try:
            if opts["comp_source"] == "custom":
                comp = Composition.from_oxide_wt(comp_el)
            else:
                comp = Composition.from_element_wt(comp_el)
            if extra:
                ox = comp.to_oxide_wt(fe3_fraction=opts["fe3"], extra_oxides=extra, normalize=True,
                                      extra_basis=opts["extra_basis"])
                comp = Composition.from_oxide_wt(ox)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Composition error", str(exc))
            return

        try:
            calc = ThermoCalc(backend=opts["backend"], **opts["backend_kwargs"])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Backend error", str(exc))
            return

        if not calc.check_backend():
            pkg = {"magemin":   "pip install petthermotools  (also requires Julia + MAGEMin_C.jl)",
                   "pycalphad":  "pip install pycalphad",
                   "perplex":    "install Perple_X and set the Perple_X dir"}.get(opts["backend"], "")
            QtWidgets.QMessageBox.warning(
                self, "Backend not available",
                f"Backend '{opts['backend']}' is not installed.\n"
                + (f"Install it with:  {pkg}" if pkg else "")
            )
            return

        try:
            # MAGEMin → subprocess pool; perplex/pycalphad → QThread.
            self._dispatch_single_grid(
                comp, opts, comp_el, opts["comp_source"], calc
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Thermodynamic calculation failed", str(exc)
            )

    # ---- Thermod result display ---------------------------------------------

    def _show_thermod_single(self, result, comp_el: dict, source: str,
                              opts: Optional[dict] = None,
                              run: Optional[dict] = None):
        """Show single-point result in a table dialog with export option.

        ``run`` is a previously recorded run record (from
        ``self.last_thermod_runs``); if None, a new record is created and
        appended automatically.
        """
        if run is None:
            run = self._record_thermod_run("single", opts or {}, comp_el, source, result)

        roi_label = run.get("roi_label", "(no ROI)")
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(
            f"Thermodynamic result — {roi_label}  |  "
            f"P={result.p_gpa:.3f} GPa, T={result.t_c:.0f}°C"
        )
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addWidget(QtWidgets.QLabel(
            f"<b>{roi_label}</b><br>"
            f"Backend: {result.backend}   |   "
            f"{'Converged' if result.converged else 'NOT converged'}   |   "
            f"Composition: {source}"
        ))

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 1: Phases ──────────────────────────────────────────────────
        phases_w = QtWidgets.QWidget()
        phases_layout = QtWidgets.QVBoxLayout(phases_w)
        tbl = QtWidgets.QTableWidget()
        rows = result.stable_phases
        tbl.setRowCount(len(rows))
        tbl.setColumnCount(4)
        tbl.setHorizontalHeaderLabels(["Phase", "mol frac", "wt frac", "vol frac"])
        for i, ph in enumerate(rows):
            tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(ph.name))
            tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{ph.mol_fraction:.4f}"))
            wt  = f"{ph.wt_fraction:.4f}"  if ph.wt_fraction  is not None else "—"
            vol = f"{ph.vol_fraction:.4f}" if ph.vol_fraction is not None else "—"
            tbl.setItem(i, 2, QtWidgets.QTableWidgetItem(wt))
            tbl.setItem(i, 3, QtWidgets.QTableWidgetItem(vol))
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.resizeColumnsToContents()
        phases_layout.addWidget(tbl)
        tabs.addTab(phases_w, "Phases")

        # ── Tab 2: Phase compositions (only shown when data available) ──────
        phases_with_comp = [ph for ph in result.stable_phases if ph.composition_oxide_wt]
        if phases_with_comp:
            comp_w = QtWidgets.QWidget()
            comp_layout = QtWidgets.QVBoxLayout(comp_w)
            all_oxides = sorted({ox for ph in phases_with_comp for ox in ph.composition_oxide_wt})
            comp_tbl = QtWidgets.QTableWidget()
            comp_tbl.setRowCount(len(phases_with_comp))
            comp_tbl.setColumnCount(1 + len(all_oxides))
            comp_tbl.setHorizontalHeaderLabels(["Phase"] + all_oxides)
            for i, ph in enumerate(phases_with_comp):
                comp_tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(ph.name))
                for j, ox in enumerate(all_oxides):
                    val = ph.composition_oxide_wt.get(ox)
                    comp_tbl.setItem(i, j + 1, QtWidgets.QTableWidgetItem(
                        f"{val:.2f}" if val is not None else "—"
                    ))
            comp_tbl.resizeColumnsToContents()
            comp_tbl.horizontalHeader().setStretchLastSection(True)
            comp_layout.addWidget(QtWidgets.QLabel("Oxide wt% per phase:"))
            comp_layout.addWidget(comp_tbl)
            tabs.addTab(comp_w, "Phase compositions")

        # ── Tab 3: Bulk properties ──────────────────────────────────────────
        if result.bulk_properties:
            prop_w = QtWidgets.QWidget()
            prop_layout = QtWidgets.QVBoxLayout(prop_w)
            prop_tbl = QtWidgets.QTableWidget()
            props = list(result.bulk_properties.items())
            prop_tbl.setRowCount(len(props))
            prop_tbl.setColumnCount(2)
            prop_tbl.setHorizontalHeaderLabels(["Property", "Value"])
            for i, (k, v) in enumerate(props):
                prop_tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
                prop_tbl.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{v:.5g}"))
            prop_tbl.resizeColumnsToContents()
            prop_tbl.horizontalHeader().setStretchLastSection(True)
            prop_layout.addWidget(prop_tbl)
            tabs.addTab(prop_w, "Bulk properties")

        # ── Tab: Parameters (read-only) ────────────────────────────────────
        param_w = QtWidgets.QWidget()
        param_layout = QtWidgets.QVBoxLayout(param_w)
        param_text = QtWidgets.QPlainTextEdit()
        param_text.setReadOnly(True)
        param_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        _f = QtGui.QFont("Courier New"); _f.setStyleHint(QtGui.QFont.Monospace)
        param_text.setFont(_f)
        param_text.setPlainText(_thermod_parameters_text(run))
        param_layout.addWidget(param_text)
        tabs.addTab(param_w, "Parameters")

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_csv    = QtWidgets.QPushButton("Export CSV …")
        btn_bundle = QtWidgets.QPushButton("Export bundle …")
        btn_locate = QtWidgets.QPushButton("Locate ROI")
        btn_close  = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_bundle)
        btn_row.addWidget(btn_locate)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        # "Locate ROI" only meaningful if a ROI was associated with the run
        btn_locate.setEnabled(run.get("roi_bbox") is not None)

        def _export_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save result CSV", "", "CSV (*.csv);;All files (*)"
            )
            if path:
                import pandas as pd
                pd.DataFrame([result.to_dict()]).to_csv(path, index=False)
                QtWidgets.QMessageBox.information(dlg, "Saved", f"Result saved to:\n{path}")

        def _export_bundle():
            out = QtWidgets.QFileDialog.getExistingDirectory(
                dlg, "Export bundle to folder", str(self.manifest.parent),
            )
            if not out:
                return
            target = Path(out) / f"thermod_run_{run.get('index', '?')}_{run.get('id', '')}"
            try:
                written = _export_thermod_bundle(run, target, fig=None)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(dlg, "Export failed", str(exc))
                return
            QtWidgets.QMessageBox.information(
                dlg, "Bundle exported",
                f"Wrote:\n  " + "\n  ".join(str(p) for p in written)
            )

        btn_csv.clicked.connect(_export_csv)
        btn_bundle.clicked.connect(_export_bundle)
        btn_locate.clicked.connect(lambda: self._locate_roi_overlay(run))
        dlg.setMinimumSize(560, 400)
        dlg.resize(700, 520)
        dlg.exec_()

    def _show_thermod_grid(self, results, opts: dict, comp_el: dict, source: str,
                            calc, run: Optional[dict] = None):
        """Show P-T grid result: phase field map, property heatmap, melt contours, P-T estimate.

        ``run`` is a previously recorded run record; if None a new record is
        created and appended to ``self.last_thermod_runs``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )
        import matplotlib.cm as mplcm
        from thermo import results_to_dataframe

        df = results_to_dataframe(results)
        n_p, n_t = opts['n_p'], opts['n_t']
        t_vals = np.linspace(opts['t_range'][0], opts['t_range'][1], n_t)
        p_vals = np.linspace(opts['p_range'][0], opts['p_range'][1], n_p)

        # Pre-compute melt fractions (try wt → vol → mol, skip NaN)
        def _phase_frac(ph):
            for v in (ph.wt_fraction, getattr(ph, 'vol_fraction', None), ph.mol_fraction):
                if v is not None and v == v:
                    return v
            return 0.0

        melt_fracs = [sum(_phase_frac(ph) for ph in r.stable_phases if ph.is_melt)
                      for r in results]
        has_melt = max(melt_fracs, default=0.0) > 1e-6

        # All unique phase names (for toggle panel)
        all_phases = sorted({ph.name for r in results for ph in r.stable_phases})

        # ── Dialog layout ──────────────────────────────────────────────────
        # Record this run if not already recorded (e.g. live first display)
        if run is None:
            run = self._record_thermod_run("grid", opts, comp_el, source, results)
        roi_label = run.get("roi_label", "(no ROI)")

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"P-T grid result \u2014 {roi_label}")
        main_layout = QtWidgets.QVBoxLayout(dlg)
        main_layout.addWidget(QtWidgets.QLabel(
            f"<b>{roi_label}</b><br>"
            f"Backend: {opts['backend']}  |  "
            f"P=[{opts['p_range'][0]:.2f}, {opts['p_range'][1]:.2f}] GPa  "
            f"T=[{opts['t_range'][0]:.0f}, {opts['t_range'][1]:.0f}] \u00b0C  "
            f"({n_p}\u00d7{n_t} pts)  |  Composition: {source}"
        ))
        main_layout.addWidget(_hline())

        # Controls row
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("View:"))
        cmb_view = QtWidgets.QComboBox()
        cmb_view.addItem("Phase field map")
        _obs_modes = opts.get("observed_modes") or None
        if _obs_modes or opts.get("presence_keys"):
            cmb_view.addItem("Assemblage misfit")
        _excluded_cols = {'P_GPa', 'T_C', 'backend', 'converged', 'phases', 'n_phases'}
        for col in df.columns:
            if col not in _excluded_cols and df[col].notna().any():
                cmb_view.addItem(f"Property: {col}")
        ctrl_row.addWidget(cmb_view)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QtWidgets.QLabel("Melt contours:"))
        chk_melt = QtWidgets.QCheckBox()
        chk_melt.setChecked(has_melt)
        chk_melt.setEnabled(has_melt)
        ctrl_row.addWidget(chk_melt)
        # Accepted-region tolerance for the misfit view (misfit <= min + δ);
        # shared with the P–T-vs-scale tab/export via self._misfit_region_delta.
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QtWidgets.QLabel("Region tol. δ:"))
        spin_region_delta = QtWidgets.QDoubleSpinBox()
        spin_region_delta.setRange(0.0, 0.5)
        spin_region_delta.setDecimals(3)
        spin_region_delta.setSingleStep(0.005)
        spin_region_delta.setValue(self._region_delta())
        spin_region_delta.setToolTip(
            "Accepted region of the Assemblage-misfit view: all cells within δ "
            "of the minimum misfit (dashed contour). The same δ is used by the "
            "P–T-vs-scale tab and export.")
        ctrl_row.addWidget(spin_region_delta)
        ctrl_row.addStretch()
        main_layout.addLayout(ctrl_row)

        # ── Canvas + phase-toggle panel side by side ───────────────────────
        canvas_and_panel = QtWidgets.QHBoxLayout()
        main_layout.addLayout(canvas_and_panel)

        # Left: toolbar + canvas
        left_w = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_w)
        left_layout.setContentsMargins(0, 0, 0, 0)
        fig = plt.figure(figsize=(6, 5))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, dlg)
        left_layout.addWidget(toolbar)
        left_layout.addWidget(canvas)
        canvas_and_panel.addWidget(left_w, stretch=1)

        # Right: phase toggle panel (only shown in phase field map view)
        phase_panel = QtWidgets.QGroupBox("Highlight phases")
        phase_panel.setMaximumWidth(180)
        phase_panel_layout = QtWidgets.QVBoxLayout(phase_panel)
        phase_scroll_area = QtWidgets.QScrollArea()
        phase_scroll_area.setWidgetResizable(True)
        phase_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        phase_inner = QtWidgets.QWidget()
        phase_inner_layout = QtWidgets.QVBoxLayout(phase_inner)
        phase_inner_layout.setSpacing(2)
        phase_inner_layout.setContentsMargins(4, 4, 4, 4)
        # Fixed overlay colour per phase (stable across redraws) so the coloured
        # checkbox label always matches the shaded region / outline on the plot.
        _overlay_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
            '#ff7f00', '#a65628', '#f781bf', '#999999',
        ]
        phase_color = {nm: _overlay_colors[i % len(_overlay_colors)]
                       for i, nm in enumerate(all_phases)}
        phase_checks: dict = {}
        for name in all_phases:
            chk = QtWidgets.QCheckBox(name)
            chk.setStyleSheet(f"color: {phase_color[name]};")
            phase_checks[name] = chk
            phase_inner_layout.addWidget(chk)
        phase_inner_layout.addStretch()
        phase_scroll_area.setWidget(phase_inner)
        phase_panel_layout.addWidget(phase_scroll_area)
        canvas_and_panel.addWidget(phase_panel)

        main_layout.addWidget(_hline())

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_select = QtWidgets.QPushButton("Select assemblage \u2026")
        btn_select.setEnabled(opts.get("workflow", "pseudo") != "pseudo")
        btn_clear_hl = QtWidgets.QPushButton("Clear highlight")
        btn_clear_hl.setToolTip(
            "Remove the assemblage stability outline drawn by "
            "'Select assemblage \u2026' (or the P-T estimate overlay).")
        btn_clear_hl.setEnabled(opts.get("workflow", "pseudo") != "pseudo")
        btn_mc = QtWidgets.QPushButton("Monte Carlo \u2026")
        btn_mc.setToolTip(
            "Propagate bulk-composition uncertainty: run this grid N times "
            "with the composition perturbed by a chosen \u03c3, then map the "
            "assemblage stability probability, the misfit mean/spread, and "
            "the scatter of the best-fit P-T.")
        btn_params = QtWidgets.QPushButton("Parameters \u2026")
        btn_csv  = QtWidgets.QPushButton("Export CSV \u2026")
        btn_png  = QtWidgets.QPushButton("Save plot \u2026")
        btn_bundle = QtWidgets.QPushButton("Export bundle \u2026")
        btn_locate = QtWidgets.QPushButton("Locate ROI")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_select)
        btn_row.addWidget(btn_clear_hl)
        btn_row.addWidget(btn_mc)
        btn_row.addWidget(btn_params)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_png)
        btn_row.addWidget(btn_bundle)
        btn_row.addWidget(btn_locate)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)
        btn_locate.setEnabled(run.get("roi_bbox") is not None)

        # ── Closure state ──────────────────────────────────────────────────
        _state = {'ax': None}   # mutable ax reference (reset each replot)
        _observed: List[Optional[List[str]]] = [
            opts.get("observed_phases") or None
        ]
        # True once the observed assemblage was set from the Select-assemblage
        # dialog (grid-native phase names → match the grid EXACTLY). The initial
        # value comes from the mineral-ID / EBSD hook (foreign vocabulary →
        # alias-aware matching), so it starts False.
        _observed_exact = [False]
        _no_match_warned = [False]

        # ── Inner drawing functions ────────────────────────────────────────
        tab20 = matplotlib.colormaps['tab20']

        def _draw_phase_fields(ax):
            field_idx, ranked = _make_phase_field_map(results, n_p, n_t)
            n = len(ranked)
            colors = [tab20(i % 20) for i in range(n)]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(np.arange(-0.5, n), cmap.N)
            ax.pcolormesh(t_vals, p_vals, field_idx, cmap=cmap, norm=norm, shading='auto')
            legend_n = min(n, 20)
            patches = []
            for i, (fs, cnt) in enumerate(ranked[:legend_n]):
                label = " + ".join(sorted(fs))
                if len(label) > 45:
                    label = label[:43] + "\u2026"
                patches.append(mpatches.Patch(color=colors[i], label=f"{label}  (n={cnt})"))
            if n > legend_n:
                patches.append(mpatches.Patch(color='lightgray',
                                              label=f"\u2026 {n - legend_n} more assemblages"))
            ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=7, framealpha=0.8, borderaxespad=0)
            fig.subplots_adjust(right=0.60)
            ax.set_title("Phase field map")

        def _draw_phase_overlays(ax):
            T2d, P2d = np.meshgrid(t_vals, p_vals)
            for name, chk in phase_checks.items():
                if not chk.isChecked():
                    continue
                mask = np.array(
                    [any(ph.name == name for ph in r.stable_phases) for r in results],
                    dtype=float
                ).reshape(n_p, n_t)
                if mask.max() < 0.5:
                    continue
                color = phase_color[name]
                # Shade the stable side (translucent fill) so it's unambiguous
                # which side of the boundary line the phase is stable on; the
                # solid line marks the in/out boundary. Colour matches the
                # (coloured) checkbox label.
                ax.contourf(T2d, P2d, mask, levels=[0.5, 1.5],
                            colors=[color], alpha=0.16, zorder=2)
                ax.contour(T2d, P2d, mask, levels=[0.5],
                           colors=[color], linewidths=2.0, linestyles='-', zorder=3)

        def _draw_property_heatmap(ax, prop: str):
            vals = df[prop].values
            expected = n_p * n_t
            if len(vals) != expected:
                # Pad or trim gracefully if the backend returned a different count
                tmp = np.full(expected, np.nan)
                tmp[:min(len(vals), expected)] = vals[:min(len(vals), expected)]
                vals = tmp
            grid = vals.reshape(n_p, n_t)
            im = ax.pcolormesh(t_vals, p_vals, grid, cmap='viridis', shading='auto')
            fig.colorbar(im, ax=ax, label=prop)
            fig.subplots_adjust(right=0.88)
            ax.set_title(f"Property: {prop}")

        def _draw_melt_contours(ax):
            mf = np.array(melt_fracs).reshape(n_p, n_t)
            T2d, P2d = np.meshgrid(t_vals, p_vals)
            levels = [v for v in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9] if v < mf.max()]
            if not levels:
                return
            cs = ax.contour(T2d, P2d, mf, levels=levels,
                            colors='white', linewidths=0.8, linestyles='--')
            ax.clabel(cs, fmt=lambda v: f"{v*100:.0f}%", fontsize=7, inline=True)

        def _draw_estimate_overlay(ax, mask):
            # Translucent red fill marks WHERE the selected assemblage is stable
            # (which side of the boundary); the red line is the boundary itself.
            T2d, P2d = np.meshgrid(t_vals, p_vals)
            m = mask.astype(float)
            ax.contourf(T2d, P2d, m, levels=[0.5, 1.5],
                        colors=['red'], alpha=0.18, zorder=2)
            ax.contour(T2d, P2d, m, levels=[0.5],
                       colors=['red'], linewidths=1.5, zorder=3)

        _misfit_cache: dict = {}

        def _draw_misfit_map(ax):
            # Observed (phase-map pixel counts) vs predicted modal proportions
            # per cell; 0 = predicted modes match the observation exactly.
            if "grid" not in _misfit_cache:
                _misfit_cache["grid"] = _modal_misfit_grid(
                    results, _obs_modes, n_p, n_t,
                    weights=opts.get("modal_weights"),
                    presence_keys=opts.get("presence_keys"),
                    open_other_tol=opts.get("open_other_tol", 0.0))
            mgrid, basis, unmatched = _misfit_cache["grid"]
            im = ax.pcolormesh(t_vals, p_vals, mgrid, cmap='RdYlGn_r',
                               vmin=0.0, vmax=1.0, shading='auto')
            fig.colorbar(im, ax=ax, label="modal misfit (0 = perfect match)")
            fig.subplots_adjust(right=0.88)
            title = "Assemblage misfit"
            if np.isfinite(mgrid).any():
                i, j = np.unravel_index(int(np.nanargmin(mgrid)), mgrid.shape)
                ax.plot(t_vals[j], p_vals[i], marker='*', markersize=16,
                        markeredgecolor='black', markerfacecolor='white', zorder=5)
                title += (f"  —  best fit T={t_vals[j]:.0f}°C, "
                          f"P={p_vals[i]:.2f} GPa "
                          f"(misfit {mgrid[i, j]:.2f}, {basis} basis)")
                # Accepted region (misfit <= min + δ): on a flat valley the
                # best cell is arbitrary; the region is the estimate. Same
                # δ as the P–T-vs-scale export (self._misfit_region_delta).
                delta = self._region_delta()
                reg = _pt_region_from_misfit(mgrid, opts["p_range"],
                                             opts["t_range"], n_p, n_t,
                                             delta=delta)
                if reg is not None:
                    try:
                        ax.contour(t_vals, p_vals, mgrid,
                                   levels=[float(np.nanmin(mgrid)) + delta],
                                   colors='white', linestyles='--',
                                   linewidths=1.3, zorder=4)
                    except Exception:
                        pass
                    title += (f"\naccepted region (misfit ≤ min + {delta:.3g}): "
                              f"T {reg['t_lo']:.0f}–{reg['t_hi']:.0f}°C, "
                              f"P {reg['p_lo']:.2f}–{reg['p_hi']:.2f} GPa")
            else:
                title += "  —  no cell has usable modal fractions"
            ax.set_title(title, fontsize=9)
            obs_txt = ", ".join(f"{k} {100 * v:.0f}%"
                                for k, v in sorted(_obs_modes.items(),
                                                   key=lambda kv: -kv[1]))
            note = "observed: " + obs_txt
            if unmatched:
                note += ("\nnever stable in this grid: "
                         + ", ".join(unmatched))
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7,
                    color='#b71c1c' if unmatched else '#333333',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

        def _replot():
            # Full figure reset — prevents axes shrinkage from colorbar steal
            fig.clear()
            ax = fig.add_subplot(111)
            _state['ax'] = ax

            view = cmb_view.currentText()
            phase_panel.setVisible(view == "Phase field map")

            if view == "Phase field map":
                _draw_phase_fields(ax)
                _draw_phase_overlays(ax)
            elif view == "Assemblage misfit":
                _draw_misfit_map(ax)
            else:
                prop = view[len("Property: "):]
                _draw_property_heatmap(ax, prop)

            if chk_melt.isChecked() and has_melt:
                _draw_melt_contours(ax)

            workflow = opts.get("workflow", "pseudo")
            if workflow != "pseudo" and _observed[0]:
                mask = _pt_estimate_from_results(results, _observed[0], n_p, n_t,
                                                 exact=_observed_exact[0])
                if mask.any():
                    _draw_estimate_overlay(ax, mask)
                elif not _no_match_warned[0]:
                    QtWidgets.QMessageBox.information(
                        dlg, "No P-T match",
                        "No grid cells found where all specified phases are stable."
                    )
                    _no_match_warned[0] = True

            ax.set_xlabel("T [\u00b0C]")
            ax.set_ylabel("P [GPa]")
            canvas.draw()

        # ── Wire controls ──────────────────────────────────────────────────
        cmb_view.currentTextChanged.connect(lambda _: _replot())
        chk_melt.stateChanged.connect(lambda _: _replot())

        def _on_region_delta(v):
            self._misfit_region_delta = float(v)
            if cmb_view.currentText() == "Assemblage misfit":
                _replot()
        spin_region_delta.valueChanged.connect(_on_region_delta)
        for chk in phase_checks.values():
            chk.stateChanged.connect(lambda _: _replot())

        def _open_selector():
            chosen = _assemblage_selector_dialog(results, parent=dlg)
            if chosen is None:
                return
            if not chosen:
                QtWidgets.QMessageBox.warning(dlg, "Nothing selected",
                                              "Select at least one phase.")
                return
            _observed[0] = chosen
            _observed_exact[0] = True   # grid-native names → exact match
            _no_match_warned[0] = False
            _replot()

        btn_select.clicked.connect(_open_selector)

        def _clear_highlight():
            # Reset the assemblage stability outline (Select-assemblage / P-T
            # estimate). Leaves the per-phase 'highlight phases' overlays alone.
            _observed[0] = None
            _observed_exact[0] = False
            _no_match_warned[0] = False
            _replot()

        btn_clear_hl.clicked.connect(_clear_highlight)

        def _open_mc():
            observed = _observed[0]
            if not observed and not _obs_modes:
                QtWidgets.QMessageBox.information(
                    dlg, "Monte Carlo",
                    "Monte Carlo needs something observed to score against: "
                    "run with the P-T estimate workflow (observed phases) or "
                    "launch from the phase dialog so the observed modal "
                    "proportions ride along. 'Select assemblage …' also works.")
                return
            self._run_monte_carlo(opts, comp_el, calc,
                                  observed=observed, modes=_obs_modes,
                                  parent_dlg=dlg)

        btn_mc.clicked.connect(_open_mc)

        def _save_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save grid CSV", "", "CSV (*.csv);;All files (*)"
            )
            if path:
                df.to_csv(path, index=False)
                QtWidgets.QMessageBox.information(dlg, "Saved", f"Grid saved to:\n{path}")

        def _save_png():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save plot", "", "PNG (*.png);;All files (*)"
            )
            if path:
                fig.savefig(path, dpi=150, bbox_inches="tight")
                QtWidgets.QMessageBox.information(dlg, "Saved", f"Plot saved to:\n{path}")

        def _save_bundle():
            out = QtWidgets.QFileDialog.getExistingDirectory(
                dlg, "Export bundle to folder", str(self.manifest.parent),
            )
            if not out:
                return
            target = Path(out) / f"thermod_run_{run.get('index', '?')}_{run.get('id', '')}"
            try:
                written = _export_thermod_bundle(run, target, fig=fig)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(dlg, "Export failed", str(exc))
                return
            QtWidgets.QMessageBox.information(
                dlg, "Bundle exported",
                f"Wrote:\n  " + "\n  ".join(str(p) for p in written)
            )

        btn_csv.clicked.connect(_save_csv)
        btn_png.clicked.connect(_save_png)
        btn_bundle.clicked.connect(_save_bundle)
        btn_locate.clicked.connect(lambda: self._locate_roi_overlay(run))
        btn_params.clicked.connect(lambda: self._show_run_parameters_dialog(run))

        # Initial plot
        _replot()
        dlg.setMinimumSize(700, 520)
        dlg.resize(960, 700)
        dlg.exec_()
        plt.close(fig)

    def _show_composition(self, comp):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('ROI composition (quick estimate)')
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget()
        rows = sorted(comp['fraction'].keys())
        table.setRowCount(len(rows)); table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Element', 'Fraction (net)'])
        for i, el in enumerate(rows):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(el))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{comp['fraction'][el]*100:.2f}%"))
        layout.addWidget(table)
        btn = QtWidgets.QPushButton('OK'); btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.resize(320, 360)
        dlg.exec_()

    # ---- Composition vs scale ----------------------------------------------

    def _show_composition_scale_result(self, records: List[dict], summary: dict,
                                        opts: dict, parent_label: str,
                                        parent_roi_bbox: Tuple[int, int, int, int]):
        """Result dialog: distribution box plots, per-scale stats, all-window table.

        Buttons:
          - Run modelling sweep …  (per-level trio: μ, μ+σ, μ−σ)
          - Export CSV …
          - Locate windows         (Shapes overlay on napari)
          - Close
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )

        levels = sorted(summary.keys())
        # Top elements by full-ROI mean (level 0 if available, else union mean)
        if 0 in summary and summary[0]["mean"]:
            ranking_src = summary[0]["mean"]
        else:
            ranking_src = {}
            for lv in levels:
                for el, v in summary[lv]["mean"].items():
                    ranking_src[el] = max(ranking_src.get(el, 0.0), v)
        all_els = sorted(ranking_src, key=lambda e: -ranking_src[e])
        top_els = [el for el in all_els if ranking_src[el] > 0.0][:8]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Composition vs scale — {parent_label}")
        main_layout = QtWidgets.QVBoxLayout(dlg)

        counts = "  ".join(
            f"L{lv}: {summary[lv]['n_windows']}/{summary[lv]['n_total']}"
            for lv in levels
        )
        dt = opts.get("data_type")
        if dt in ("oxford_wt_pct", "oxford_counts"):
            src_label = self._oxford_source_label(dt)
        elif opts.get("source") == "eds":
            src_label = "EDS net-count"
        else:
            src_label = "SPX-quantified"
        main_layout.addWidget(QtWidgets.QLabel(
            f"<b>{parent_label}</b><br>"
            f"Source: <b>{src_label}</b>  |  "
            f"Levels: {len(levels)}  |  "
            f"Usable windows (per level): {counts}"
        ))
        main_layout.addWidget(_hline())

        # ── Length-scale units ────────────────────────────────────────────
        # Window size per level and the semivariogram lag are lengths (mosaic
        # px). Given an EDS pixel pitch they can be shown in nm/µm/mm instead.
        # The pitch is read from the cache manifest (pixel_um) and is user-
        # overridable here; the override also feeds the zoning tool via
        # _pixel_um(). The window-size column / tick labels / variogram axis all
        # update live when the unit or pitch changes (via _unit_listeners).
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Length unit:"))
        cmb_unit = QtWidgets.QComboBox()
        for _u in ("px", "nm", "µm", "mm"):
            cmb_unit.addItem(_u, _u)
        unit_row.addWidget(cmb_unit)
        unit_row.addSpacing(12)
        unit_row.addWidget(QtWidgets.QLabel("EDS pixel size:"))
        spin_px = QtWidgets.QDoubleSpinBox()
        spin_px.setRange(0.0, 1.0e6)
        spin_px.setDecimals(4)
        spin_px.setSuffix(" µm/px")
        spin_px.setSpecialValueText("unset")     # shown at value 0
        _px0 = self._pixel_um()
        spin_px.setValue(float(_px0) if _px0 else 0.0)
        spin_px.setToolTip(
            "Mosaic pixel pitch in µm. Prefilled from the cache manifest when "
            "recorded; set it here when the data didn't carry one. Used to "
            "convert px ↔ nm/µm/mm here and in the zoning/traverse tool.")
        unit_row.addWidget(spin_px)
        unit_row.addStretch()
        main_layout.addLayout(unit_row)
        cmb_unit.setCurrentIndex(2 if _px0 else 0)   # default µm when known, else px

        # Per-level window short-side length (mosaic px) — the canonical scale.
        level_win_px: Dict[int, float] = {}
        for _lv in levels:
            try:
                _r, _c, (_wh, _ww) = _quadtree_level_dims(parent_roi_bbox, _lv)
                level_win_px[_lv] = float(min(_wh, _ww)) if (_wh and _ww) else 0.0
            except Exception:
                level_win_px[_lv] = 0.0

        def _len_conv():
            """(factor px→unit, unit_label); factor None when a physical unit is
            chosen but no pitch is set (callers fall back to px)."""
            u = cmb_unit.currentData()
            if u == "px":
                return 1.0, "px"
            pu = float(spin_px.value())
            if pu <= 0.0:
                return None, u
            return {"nm": pu * 1000.0, "µm": pu, "mm": pu / 1000.0}[u], u

        def _fmt_size(px_val: float) -> str:
            f, u = _len_conv()
            if f is None:
                f, u = 1.0, "px"
            v = float(px_val) * f
            if u == "px":
                return f"{v:.0f} px"
            if v >= 100:
                return f"{v:.0f} {u}"
            if v >= 10:
                return f"{v:.1f} {u}"
            return f"{v:.3g} {u}"

        _unit_listeners: List = []   # zero-arg callbacks fired on unit/pitch change

        tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(tabs)

        # ── Tab 1: Distributions (box plots per element across levels) ────
        dist_w = QtWidgets.QWidget()
        dist_layout = QtWidgets.QVBoxLayout(dist_w)
        n_top = len(top_els)
        if n_top == 0:
            dist_layout.addWidget(QtWidgets.QLabel(
                "No usable compositions — every window was skipped."
            ))
        else:
            ncols = 2 if n_top <= 4 else 3
            nrows = (n_top + ncols - 1) // ncols
            fig = plt.figure(figsize=(3.0 * ncols, 2.6 * nrows))
            for i, el in enumerate(top_els, start=1):
                ax = fig.add_subplot(nrows, ncols, i)
                data: List[List[float]] = []
                tick_labels: List[str] = []
                for lv in levels:
                    vals = [r["wt_pct"].get(el, 0.0)
                            for r in records
                            if r["level"] == lv and r.get("wt_pct")]
                    if vals:
                        data.append(vals)
                    else:
                        data.append([0.0])
                    tick_labels.append(f"L{lv}")
                # Drawing every outlier point is slow + memory-heavy when a
                # level has many thousands of windows; suppress fliers then.
                _show_fliers = len(records) <= 2000
                ax.boxplot(data, tick_labels=tick_labels,
                           showfliers=_show_fliers, widths=0.6)
                ax.set_title(el, fontsize=10)
                ax.set_ylabel("wt%")
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dlg)
            dist_layout.addWidget(toolbar)
            dist_layout.addWidget(canvas)
            # Window size per level shown once as a compact caption — inlining it
            # on every subplot's x-ticks cluttered the axes. Updates with the unit.
            dist_size_lbl = QtWidgets.QLabel()
            dist_size_lbl.setWordWrap(True)
            dist_size_lbl.setStyleSheet("color: gray; font-size: 8pt;")
            dist_layout.addWidget(dist_size_lbl)

            def _relabel_dist():
                parts = [f"L{lv} = {_fmt_size(level_win_px[lv])}" for lv in levels]
                dist_size_lbl.setText("Window size per level:   "
                                      + "    ".join(parts))
            _relabel_dist()
            _unit_listeners.append(_relabel_dist)
        tabs.addTab(dist_w, "Distributions")

        # ── Tab 2: Per-scale summary table ────────────────────────────────
        summ_w = QtWidgets.QWidget()
        summ_layout = QtWidgets.QVBoxLayout(summ_w)
        if all_els:
            stbl = QtWidgets.QTableWidget()
            stbl.setRowCount(len(levels))
            cols = ["Level", "Window", "n", "n_total"] + [f"{el} mean ± σ"
                                                          for el in all_els]
            stbl.setColumnCount(len(cols))
            stbl.setHorizontalHeaderLabels(cols)
            for i, lv in enumerate(levels):
                row = summary[lv]
                stbl.setItem(i, 0, QtWidgets.QTableWidgetItem(f"L{lv}"))
                stbl.setItem(i, 1, QtWidgets.QTableWidgetItem(_fmt_size(level_win_px[lv])))
                stbl.setItem(i, 2, QtWidgets.QTableWidgetItem(str(row["n_windows"])))
                stbl.setItem(i, 3, QtWidgets.QTableWidgetItem(str(row["n_total"])))
                for j, el in enumerate(all_els):
                    mu = row["mean"].get(el, 0.0)
                    sd = row["std"].get(el, 0.0)
                    stbl.setItem(i, 4 + j, QtWidgets.QTableWidgetItem(
                        f"{mu:6.2f} ± {sd:5.2f}"
                    ))
            stbl.resizeColumnsToContents()
            stbl.horizontalHeader().setStretchLastSection(True)
            summ_layout.addWidget(stbl)

            note = QtWidgets.QLabel(
                "± σ above is the per-element spread <i>between windows</i> at "
                "each scale (marginal standard deviation). The modelling sweep "
                "does <b>not</b> perturb each element by this σ independently — "
                "that would mostly cancel once the bulk is renormalised. Instead "
                "it steps ±1 σ along the dominant axis of joint composition "
                "variation (PC1), so the perturbation changes element ratios and "
                "its size tracks the real heterogeneity.")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray; font-size: 8pt;")
            summ_layout.addWidget(note)

            # ── Representative averaging scale (needs a tolerance) ────────
            # The between-window spread keeps falling as the window grows and
            # never plateaus on a heterogeneous sample, so "where it stops
            # growing" never triggers. A window is called representative when
            # every element's spread is at or below a tolerance: by default the
            # calibration's hold-out error (a placement that moves the bulk by
            # less than the maps resolve cannot be told apart), else a relative
            # fraction of the mean. Plotted as spread / tolerance with a line
            # at 1; the readout names the level and the limiting element.
            spread_lv = _spread_by_level(records, all_els, ddof=1)
            means_lv = {lv: dict(summary[lv]["mean"]) for lv in levels}
            cv = getattr(self, "eds_calibration_cv", None)
            cv_tol = {}
            if cv is not None and getattr(cv, "per_element", None):
                cv_tol = {el: float(st.rmse) for el, st in cv.per_element.items()
                          if el in all_els and np.isfinite(st.rmse) and st.rmse > 0}
            tol_row = QtWidgets.QHBoxLayout()
            tol_row.addWidget(QtWidgets.QLabel("Tolerance:"))
            cmb_tol = QtWidgets.QComboBox()
            if cv_tol:
                cmb_tol.addItem("calibration hold-out error (per element)", "cv")
            cmb_tol.addItem("relative % of the mean", "rel")
            cmb_tol.setToolTip(
                "A window size is representative when the spread between windows "
                "of EVERY element is at or below this tolerance. The calibration's "
                "hold-out RMSE (Calibrate EDS) is the natural choice: heterogeneity "
                "smaller than the maps' own error cannot be resolved. Without a "
                "calibration, a relative fraction of each element's mean.")
            tol_row.addWidget(cmb_tol)
            spin_tol = QtWidgets.QDoubleSpinBox()
            spin_tol.setRange(0.1, 100.0)
            spin_tol.setDecimals(1)
            spin_tol.setSuffix(" %")
            spin_tol.setValue(10.0)
            spin_tol.setEnabled(not cv_tol)
            tol_row.addWidget(spin_tol)
            tol_row.addStretch()
            summ_layout.addLayout(tol_row)
            rep_lbl = QtWidgets.QLabel()
            rep_lbl.setWordWrap(True)
            summ_layout.addWidget(rep_lbl)
            rfig = plt.figure(figsize=(6.0, 2.6))
            rcanvas = FigureCanvas(rfig)
            summ_layout.addWidget(rcanvas)

            def _draw_representative():
                use_cv = cmb_tol.currentData() == "cv"
                spin_tol.setEnabled(not use_cv)
                tol = cv_tol if use_cv else float(spin_tol.value()) / 100.0
                best, ratios = _representative_level(spread_lv, tol, means=means_lv)
                rfig.clear()
                ax = rfig.add_subplot(1, 1, 1)
                f_len, ulab = _len_conv()
                if f_len is None:
                    f_len, ulab = 1.0, "px"
                lvs = [lv for lv in sorted(ratios) if ratios[lv]]
                for el in all_els:
                    xs = [level_win_px[lv] * f_len for lv in lvs if el in ratios[lv]]
                    ys = [ratios[lv][el] for lv in lvs if el in ratios[lv]]
                    if xs:
                        ax.plot(xs, ys, marker="o", ms=3, lw=1.0, label=el)
                ax.axhline(1.0, color="0.4", ls="--", lw=0.9)
                if lvs:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.invert_xaxis()
                ax.set_xlabel(f"window size ({ulab})")
                ax.set_ylabel("spread / tolerance")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, ncol=3)
                rfig.tight_layout()
                rcanvas.draw()
                tol_txt = ("the calibration hold-out error" if use_cv
                           else f"{spin_tol.value():.1f} % of the mean")
                if best is None:
                    first = lvs[0] if lvs else None
                    if first is not None and ratios[first]:
                        el_max = max(ratios[first], key=ratios[first].get)
                        rep_lbl.setText(
                            f"<b>Representative averaging scale: not reached.</b> "
                            f"Even at L{first} ({_fmt_size(level_win_px[first])}) "
                            f"{el_max} spreads {ratios[first][el_max]:.1f}× {tol_txt}; "
                            "the bulk of this region depends on where the window is placed.")
                    else:
                        rep_lbl.setText("Representative averaging scale: no tolerance "
                                        "could be applied (no calibrated elements).")
                else:
                    nxt = [lv for lv in lvs if lv > best]
                    lim = ""
                    if nxt and ratios[nxt[0]]:
                        el_max = max(ratios[nxt[0]], key=ratios[nxt[0]].get)
                        lim = (f" At L{nxt[0]} ({_fmt_size(level_win_px[nxt[0]])}) "
                               f"{el_max} exceeds it ({ratios[nxt[0]][el_max]:.1f}×).")
                    rep_lbl.setText(
                        f"<b>Representative averaging scale: L{best} "
                        f"({_fmt_size(level_win_px[best])}).</b> Down to this window "
                        f"size every element's spread between windows is within "
                        f"{tol_txt}, so a window's bulk no longer depends on its "
                        f"placement.{lim}")

            cmb_tol.currentIndexChanged.connect(lambda _i: _draw_representative())
            spin_tol.valueChanged.connect(lambda _v: _draw_representative())
            _unit_listeners.append(_draw_representative)
            _draw_representative()

            def _relabel_summary():
                for i, lv in enumerate(levels):
                    it = stbl.item(i, 1)
                    if it is not None:
                        it.setText(_fmt_size(level_win_px[lv]))
                stbl.resizeColumnsToContents()
            _unit_listeners.append(_relabel_summary)
        else:
            summ_layout.addWidget(QtWidgets.QLabel("No data."))
        tabs.addTab(summ_w, "Per-scale summary")

        # ── Tab 3: Spatial structure (semivariogram + Moran's I) ──────────
        # Variance (Tab 2) says how much composition spreads at each scale;
        # this tab adds *where*: the semivariogram shows the lag at which
        # composition decorrelates, and Moran's I whether like values cluster.
        struct_w = QtWidgets.QWidget()
        struct_layout = QtWidgets.QVBoxLayout(struct_w)
        eligible = _eligible_variogram_levels(records, min_windows=4)
        vfig = None
        if not all_els or not eligible:
            struct_layout.addWidget(QtWidgets.QLabel(
                "Not enough windows for spatial statistics.\n\n"
                "A semivariogram needs ≥4 usable windows at some quadtree "
                "level — increase the quadtree depth or enlarge the ROI."
            ))
        else:
            vario_curve_els = [el for el in top_els][:6]
            varios = {lv: _semivariogram(records, lv, all_els)
                      for lv in eligible}
            morans = {lv: _morans_i(records, lv, all_els) for lv in eligible}

            struct_layout.addWidget(QtWidgets.QLabel(
                "<b>Semivariogram</b> γ(h): how different two windows' "
                "compositions are, on average, plotted against how far apart "
                "they sit (lag, in the chosen length unit). It usually climbs, "
                "then flattens — the lag where it levels off is the scale beyond "
                "which composition stops being spatially correlated. The "
                "<i>total</i> curve combines all elements into one overall measure. "
                "A layered sample decorrelates faster across the layering than "
                "along it: pick a <i>direction</i> to see the curve for window "
                "pairs lying along that direction only (±22.5°)."
            ))

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Variogram level:"))
            cmb_lv = QtWidgets.QComboBox()
            for lv in eligible:
                cmb_lv.addItem(f"L{lv}  ({varios[lv].get('n_windows', 0)} windows)", lv)
            cmb_lv.setCurrentIndex(len(eligible) - 1)   # deepest = best sampled
            row.addWidget(cmb_lv)
            row.addSpacing(12)
            row.addWidget(QtWidgets.QLabel("Lag bins:"))
            spin_bins = QtWidgets.QSpinBox()
            spin_bins.setRange(3, 50)
            spin_bins.setValue(12)
            spin_bins.setToolTip(
                "Number of lag-distance classes in the semivariogram. More bins "
                "= finer lag resolution but fewer window pairs per bin (noisier "
                "γ); fewer bins = smoother but coarser. Default 12.")
            row.addWidget(spin_bins)
            row.addSpacing(12)
            row.addWidget(QtWidgets.QLabel("Direction:"))
            cmb_dir = QtWidgets.QComboBox()
            for _txt, _deg in (("all directions", None), ("along x (0°)", 0.0),
                               ("45°", 45.0), ("along y (90°)", 90.0),
                               ("135°", 135.0)):
                cmb_dir.addItem(_txt, _deg)
            cmb_dir.setToolTip(
                "Restrict the window pairs to those whose separation lies within "
                "±22.5° of this direction (0° = the mosaic x axis, 90° = the y "
                "axis). Compare along and across a foliation or banding: the "
                "correlation length is usually much shorter across it. The grey "
                "curve in the right panel is the all-directions curve for reference.")
            row.addWidget(cmb_dir)
            row.addStretch()
            struct_layout.addLayout(row)

            vfig = plt.figure(figsize=(8.0, 3.2))
            vcanvas = FigureCanvas(vfig)
            vtoolbar = NavigationToolbar(vcanvas, dlg)
            struct_layout.addWidget(vtoolbar)
            struct_layout.addWidget(vcanvas)

            def _draw_vario(lv: int):
                vfig.clear()
                # Recompute on demand so the lag-bin count and the direction are
                # live-adjustable (cheap: each level is subsampled to max_points
                # before pairing).
                deg = cmb_dir.currentData()
                v = _semivariogram(records, lv, all_els,
                                   n_bins=int(spin_bins.value()),
                                   direction_deg=deg) or {}
                v_all = (_semivariogram(records, lv, all_els,
                                        n_bins=int(spin_bins.value())) or {}
                         if deg is not None else {})
                ax1 = vfig.add_subplot(1, 2, 1)
                ax2 = vfig.add_subplot(1, 2, 2)
                bc = v.get("bin_centers")
                if bc is not None and len(bc):
                    f_len, ulab = _len_conv()
                    if f_len is None:
                        f_len, ulab = 1.0, "px"
                    bcx = np.asarray(bc, dtype=float) * f_len
                    for el in vario_curve_els:
                        ax1.plot(bcx, v["gamma"][el], marker="o", ms=3, lw=1.0,
                                 alpha=0.85, label=el)
                    ax1.set_title("per element", fontsize=10)
                    ax1.set_ylabel("γ(h)  [wt%²]")
                    if vario_curve_els:
                        ax1.legend(fontsize=7, ncol=2)
                    if v_all.get("bin_centers") is not None:
                        ax2.plot(np.asarray(v_all["bin_centers"], dtype=float) * f_len,
                                 v_all["gamma_total"], color="0.6", lw=1.2,
                                 marker="s", ms=3, label="all directions")
                    ax2.plot(bcx, v["gamma_total"], color="k", lw=2.0,
                             marker="s", ms=4,
                             label=(cmb_dir.currentText() if deg is not None else None))
                    ax2.set_title("total (all elements)", fontsize=10)
                    if deg is not None:
                        ax2.legend(fontsize=7)
                    for ax in (ax1, ax2):
                        ax.set_xlabel(f"lag distance ({ulab})")
                        ax.grid(True, alpha=0.3)
                elif deg is not None:
                    ax2.text(0.5, 0.5, "no window pairs in this direction",
                             ha="center", va="center", transform=ax2.transAxes)
                _nw = v.get("n_windows", 0)
                _nu = v.get("n_used", _nw)
                _suf = f", sampled {_nu}" if _nu < _nw else ""
                _dir = f", {cmb_dir.currentText()}" if deg is not None else ""
                vfig.suptitle(f"Semivariogram — L{lv}  "
                              f"({_nw} windows{_suf}{_dir})", fontsize=10)
                vfig.tight_layout()
                vcanvas.draw()

            cmb_lv.currentIndexChanged.connect(
                lambda _i: _draw_vario(cmb_lv.currentData())
            )
            spin_bins.valueChanged.connect(
                lambda _v: _draw_vario(cmb_lv.currentData())
            )
            cmb_dir.currentIndexChanged.connect(
                lambda _i: _draw_vario(cmb_lv.currentData())
            )
            _unit_listeners.append(lambda: _draw_vario(cmb_lv.currentData()))
            _draw_vario(cmb_lv.currentData())

            # Moran's I table: top elements × eligible levels.
            struct_layout.addWidget(QtWidgets.QLabel(
                "<b>Moran's I</b> (inverse-distance): +1 like values cluster, "
                "0 no spatial pattern, −1 neighbours differ."
            ))
            mtbl = QtWidgets.QTableWidget()
            mtbl.setRowCount(len(top_els))
            mcols = ["Element"] + [f"L{lv}" for lv in eligible]
            mtbl.setColumnCount(len(mcols))
            mtbl.setHorizontalHeaderLabels(mcols)
            for i, el in enumerate(top_els):
                mtbl.setItem(i, 0, QtWidgets.QTableWidgetItem(el))
                for j, lv in enumerate(eligible):
                    val = morans[lv].get(el, float("nan"))
                    txt = "—" if not np.isfinite(val) else f"{val:+.2f}"
                    mtbl.setItem(i, 1 + j, QtWidgets.QTableWidgetItem(txt))
            mtbl.resizeColumnsToContents()
            mtbl.horizontalHeader().setStretchLastSection(True)
            mtbl.setMaximumHeight(220)
            struct_layout.addWidget(mtbl)
        tabs.addTab(struct_w, "Spatial structure")

        # ── Tab 4: All windows ────────────────────────────────────────────
        # A deep quadtree over a large mosaic can yield very many windows;
        # populating a QTableWidget with all of them is slow and memory-heavy,
        # so cap the displayed rows (the full set is still in the CSV export).
        WIN_TABLE_CAP = 5000
        wins_w = QtWidgets.QWidget()
        wins_layout = QtWidgets.QVBoxLayout(wins_w)
        if len(records) > WIN_TABLE_CAP:
            wins_layout.addWidget(QtWidgets.QLabel(
                f"<i>Showing the first {WIN_TABLE_CAP:,} of {len(records):,} "
                "windows. Use 'Export CSV …' for the full set.</i>"
            ))
        shown_records = records[:WIN_TABLE_CAP]
        wtbl = QtWidgets.QTableWidget()
        win_cols = (["Level", "y0", "y1", "x0", "x1", "n_pix",
                     "mask_frac", "skipped"] +
                    [f"{el}" for el in all_els])
        wtbl.setRowCount(len(shown_records))
        wtbl.setColumnCount(len(win_cols))
        wtbl.setHorizontalHeaderLabels(win_cols)
        for i, r in enumerate(shown_records):
            y0, y1, x0, x1 = r["bbox"]
            wtbl.setItem(i, 0, QtWidgets.QTableWidgetItem(f"L{r['level']}"))
            wtbl.setItem(i, 1, QtWidgets.QTableWidgetItem(str(y0)))
            wtbl.setItem(i, 2, QtWidgets.QTableWidgetItem(str(y1)))
            wtbl.setItem(i, 3, QtWidgets.QTableWidgetItem(str(x0)))
            wtbl.setItem(i, 4, QtWidgets.QTableWidgetItem(str(x1)))
            wtbl.setItem(i, 5, QtWidgets.QTableWidgetItem(str(r["n_pixels"])))
            wtbl.setItem(i, 6, QtWidgets.QTableWidgetItem(f"{r['mask_frac']:.2f}"))
            wtbl.setItem(i, 7, QtWidgets.QTableWidgetItem(r.get("skipped", "")))
            for j, el in enumerate(all_els):
                v = r.get("wt_pct", {}).get(el)
                wtbl.setItem(i, 8 + j, QtWidgets.QTableWidgetItem(
                    f"{v:.2f}" if v is not None else "—"
                ))
        wtbl.resizeColumnsToContents()
        wtbl.horizontalHeader().setStretchLastSection(True)
        wins_layout.addWidget(wtbl)
        tabs.addTab(wins_w, "Windows")

        # ── P–T vs scale ─────────────────────────────────────────────────
        # Best cell + accepted regions per level, reduced from this ROI's
        # sensitivity-sweep grids in the run history (the same rows the
        # "Export P–T vs scale …" button writes). Two nested bands: the mean
        # variant's OWN accepted region (how flat the misfit valley is on the
        # window-mean bulk) and its union with the ±σ variants (the
        # compositional spread); markers = the ±σ best cells (argmin hopping).
        # Empty, with guidance, until a sweep exists; Refresh after a sweep.
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as _PtCanvas)
        pt_w = QtWidgets.QWidget()
        pt_layout = QtWidgets.QVBoxLayout(pt_w)
        pt_top = QtWidgets.QHBoxLayout()
        pt_top.addWidget(QtWidgets.QLabel("Region tolerance δ (misfit ≤ min + δ):"))
        spin_delta = QtWidgets.QDoubleSpinBox()
        spin_delta.setRange(0.0, 0.5)
        spin_delta.setDecimals(3)
        spin_delta.setSingleStep(0.005)
        spin_delta.setValue(self._region_delta())
        spin_delta.setToolTip("Shared with the grid dialog's misfit view and "
                              "the CSV export.")
        pt_top.addWidget(spin_delta)
        btn_pt_refresh = QtWidgets.QPushButton("Refresh")
        btn_pt_refresh.setToolTip("Re-read the sweep grids from the run history "
                                  "(after 'Run modelling sweep …').")
        pt_top.addWidget(btn_pt_refresh)
        pt_top.addStretch()
        pt_layout.addLayout(pt_top)
        pt_status = QtWidgets.QLabel("")
        pt_status.setWordWrap(True)
        pt_layout.addWidget(pt_status)
        pfig = plt.figure(figsize=(6.4, 3.6))
        pcanvas = _PtCanvas(pfig)
        pt_layout.addWidget(pcanvas, stretch=3)
        ptbl = QtWidgets.QTableWidget()
        ptbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        pt_layout.addWidget(ptbl, stretch=2)
        tabs.addTab(pt_w, "P–T vs scale")

        def _refresh_pt_tab(*_):
            self._misfit_region_delta = float(spin_delta.value())
            rows, problem = self._pt_scale_rows_for(
                parent_label, parent_roi_bbox, self._misfit_region_delta)
            pt_status.setText(problem or (
                f"{len(rows)} scale(s) from the sweep grids in the run history"
                f"  ·  δ = {self._misfit_region_delta:.3g}  ·  bands: dark = "
                "accepted region of the mean composition, light = union with "
                "the ±σ variants; ▲▼ = ±σ best cells"))
            cols = ["Window", "T best", "T accepted (mean)", "T accepted (±σ)",
                    "P best", "P accepted (mean)", "P accepted (±σ)"]
            ptbl.clear()
            ptbl.setColumnCount(len(cols))
            ptbl.setHorizontalHeaderLabels(cols)
            ptbl.setRowCount(len(rows))
            for i, r in enumerate(rows):
                vals = [_fmt_size(r["window_px"]),
                        f'{r["T_best_C"]:.0f} °C',
                        f'{r["T_lo_mean_C"]:.0f}–{r["T_hi_mean_C"]:.0f} °C',
                        f'{r["T_lo_C"]:.0f}–{r["T_hi_C"]:.0f} °C',
                        f'{r["P_best_GPa"]:.2f} GPa',
                        f'{r["P_lo_mean_GPa"]:.2f}–{r["P_hi_mean_GPa"]:.2f} GPa',
                        f'{r["P_lo_GPa"]:.2f}–{r["P_hi_GPa"]:.2f} GPa']
                for j, v in enumerate(vals):
                    ptbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
            ptbl.resizeColumnsToContents()

            pfig.clear()
            ax = pfig.add_subplot(111)
            if rows:
                f, u = _len_conv()
                if f is None:
                    f, u = 1.0, "px"
                col = lambda k: np.array([r[k] for r in rows], dtype=float)
                w = col("window_px") * f
                ax.fill_between(w, col("T_lo_C"), col("T_hi_C"), color="0.55",
                                alpha=0.22, lw=0, label="accepted, mean±σ variants")
                ax.fill_between(w, col("T_lo_mean_C"), col("T_hi_mean_C"),
                                color="0.25", alpha=0.28, lw=0,
                                label="accepted, mean composition")
                for k, mk in (("T_best_plus_C", "^"), ("T_best_minus_C", "v")):
                    ax.plot(w, col(k), mk, ms=3.5, color="0.15", alpha=0.7, lw=0)
                ax.plot(w, col("T_best_C"), "-o", ms=3.5, color="0.15", lw=1.2,
                        label="best cell, mean composition")
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_xlabel(f"window size ({u})")
                ax.set_ylabel("T (°C)")
                pcol = "#1f77b4"
                ax2 = ax.twinx()
                ax2.fill_between(w, col("P_lo_GPa"), col("P_hi_GPa"), color=pcol,
                                 alpha=0.12, lw=0)
                ax2.fill_between(w, col("P_lo_mean_GPa"), col("P_hi_mean_GPa"),
                                 color=pcol, alpha=0.18, lw=0)
                ax2.plot(w, col("P_best_GPa"), "-s", ms=3, color=pcol, lw=1.0)
                ax2.set_ylabel("P (GPa)", color=pcol)
                ax2.tick_params(axis="y", colors=pcol)
                ax.legend(loc="upper left", fontsize=7, frameon=False)
                ax.set_title("P–T estimate vs averaging scale", fontsize=9)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, problem or "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="0.4", wrap=True)
            try:
                pfig.tight_layout()
            except Exception:
                pass
            pcanvas.draw_idle()

        btn_pt_refresh.clicked.connect(_refresh_pt_tab)
        spin_delta.valueChanged.connect(_refresh_pt_tab)
        _unit_listeners.append(_refresh_pt_tab)
        _refresh_pt_tab()

        # Unit / pixel-pitch changes: persist the pitch as a session override
        # (so the zoning tool sees it too) and refresh every length display.
        def _fire_unit_change(*_):
            pu = float(spin_px.value())
            self._pixel_um_override = pu if pu > 0.0 else None
            for _cb in _unit_listeners:
                try:
                    _cb()
                except Exception:
                    pass
        cmb_unit.currentIndexChanged.connect(_fire_unit_change)
        spin_px.valueChanged.connect(_fire_unit_change)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_sweep = QtWidgets.QPushButton("Run modelling sweep …")
        btn_sweep.setToolTip(
            "Run a thermodynamic grid for {μ, μ+σ, μ−σ} per scale and add the "
            "records to the run history. The ±σ pair steps ±1 std along the "
            "dominant axis of between-window composition variation (PC1), so the "
            "bracket reflects the real heterogeneity at that scale.")
        btn_csv = QtWidgets.QPushButton("Export CSV …")
        btn_pt = QtWidgets.QPushButton("Export P–T vs scale …")
        btn_pt.setToolTip(
            "After running the sweep: reduce each scale's pseudosection to a "
            "best-fit P–T (the modal-misfit minimum against the observed "
            "assemblage), with a mean±σ envelope, and export it as a CSV. "
            "Needs a segmented + mineral-identified ROI.")
        btn_locate = QtWidgets.QPushButton("Locate windows")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_sweep)
        btn_row.addWidget(btn_csv)
        btn_row.addWidget(btn_pt)
        btn_row.addWidget(btn_locate)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        usable = any(r.get("wt_pct") for r in records)
        btn_sweep.setEnabled(usable)

        def _save_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Save composition-scale CSV", "",
                "CSV (*.csv);;All files (*)"
            )
            if not path:
                return
            import pandas as pd
            rows_out = []
            for r in records:
                y0, y1, x0, x1 = r["bbox"]
                row = {
                    "level": r["level"],
                    "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                    "n_pixels": r["n_pixels"],
                    "mask_frac": r["mask_frac"],
                    "skipped": r.get("skipped", ""),
                }
                for el in all_els:
                    row[el] = r.get("wt_pct", {}).get(el)
                rows_out.append(row)
            pd.DataFrame(rows_out).to_csv(path, index=False)
            QtWidgets.QMessageBox.information(dlg, "Saved",
                                              f"Wrote:\n{path}")

        btn_csv.clicked.connect(_save_csv)
        btn_locate.clicked.connect(
            lambda: self._locate_composition_windows(records, parent_label)
        )
        btn_sweep.clicked.connect(
            lambda: self._run_composition_sweep(summary, opts, parent_label,
                                                 parent_roi_bbox, dlg,
                                                 records=records)
        )
        btn_pt.clicked.connect(
            lambda: self._export_pt_scale_csv(parent_label, parent_roi_bbox, dlg)
        )

        dlg.setMinimumSize(700, 520)
        dlg.resize(960, 700)
        dlg.exec_()
        if 'fig' in locals():
            plt.close(fig)
        if vfig is not None:
            plt.close(vfig)
        try:
            plt.close(pfig)
        except Exception:
            pass

    def _locate_composition_windows(self, records: List[dict],
                                     parent_label: str):
        """Add a colour-coded Shapes layer showing every quadtree window."""
        if not records:
            QtWidgets.QMessageBox.information(self, "No windows",
                                              "Nothing to draw.")
            return
        layer_name = f"Composition windows — {parent_label}"
        # Remove old instance to avoid duplicates
        for L in list(self.viewer.layers):
            if L.name == layer_name:
                self.viewer.layers.remove(L)

        # One colour per level (cycle through tab10)
        try:
            tab10 = matplotlib.colormaps['tab10']
        except Exception:
            tab10 = None

        levels = sorted({r["level"] for r in records})
        level_color: Dict[int, str] = {}
        for i, lv in enumerate(levels):
            if tab10 is not None:
                rgba = tab10(i % 10)
                level_color[lv] = '#%02x%02x%02x' % (
                    int(255 * rgba[0]), int(255 * rgba[1]), int(255 * rgba[2])
                )
            else:
                level_color[lv] = ['#1f77b4', '#ff7f0e', '#2ca02c',
                                   '#d62728', '#9467bd', '#8c564b',
                                   '#e377c2', '#7f7f7f', '#bcbd22',
                                   '#17becf'][i % 10]

        verts_list = []
        edge_colors = []
        labels = []
        for r in records:
            if not r.get("wt_pct"):
                continue
            y0, y1, x0, x1 = r["bbox"]
            verts_list.append(np.array([[y0, x0], [y0, x1],
                                         [y1, x1], [y1, x0]],
                                        dtype=float))
            edge_colors.append(level_color[r["level"]])
            labels.append(f"L{r['level']}")

        if not verts_list:
            QtWidgets.QMessageBox.information(
                self, "No windows",
                "All windows were skipped — nothing to draw."
            )
            return

        lyr = self.viewer.add_shapes(
            data=verts_list,
            shape_type='rectangle',
            name=layer_name,
            edge_color=edge_colors,
            face_color='transparent',
            edge_width=2,
        )
        try:
            lyr.text = {"string": labels, "color": "white", "size": 9,
                        "anchor": "upper_left"}
        except Exception:
            pass

    def _pt_estimate_from_grid(self, run: dict, observed_modes: Dict[str, float],
                               misfit_opts: Optional[dict] = None):
        """Best-fit (P_GPa, T_C) for one swept grid via the modal-misfit minimum.

        Reduces the recorded pseudosection to a single P-T point: the grid cell
        whose predicted modes best match the observed assemblage (Bray-Curtis
        ``_modal_misfit_grid``; argmin). This is robust to the observed
        assemblage not being exactly stable in any one field (the misfit is
        continuous and always defined). ``misfit_opts`` (from
        ``_assemblage_inputs_from_state``) carries per-key weights, presence-only
        phases, and the open-assemblage tolerance. Returns ``None`` if the grid
        has no usable cell. The records store *serialised* results, so they are
        deserialised here.
        """
        o = run.get("opts") or {}
        try:
            n_p, n_t = int(o["n_p"]), int(o["n_t"])
            results = [_deserialize_thermod_result(d) for d in run["results"]]
        except Exception:
            return None
        mo = misfit_opts or {}
        misfit, _basis, _unmatched = _modal_misfit_grid(
            results, observed_modes, n_p, n_t,
            weights=mo.get("weights"), presence_keys=mo.get("presence_keys"),
            open_other_tol=mo.get("open_other_tol", 0.0))
        return _best_fit_pt(misfit, o["p_range"], o["t_range"], n_p, n_t)

    def _pt_region_from_grid(self, run: dict, observed_modes: Dict[str, float],
                             misfit_opts: Optional[dict] = None,
                             delta: float = 0.02):
        """Near-minimal P-T region for one swept grid (argmin + accepted span).

        Like ``_pt_estimate_from_grid`` but returns the
        ``_pt_region_from_misfit`` dict: the argmin cell plus the extent of
        all cells within ``delta`` of the minimum. The P-T-vs-scale export
        builds its envelope from these regions, because on a flat misfit
        valley the spread of argmin points measures valley-hopping noise
        rather than constraint quality.
        """
        o = run.get("opts") or {}
        try:
            n_p, n_t = int(o["n_p"]), int(o["n_t"])
            results = [_deserialize_thermod_result(d) for d in run["results"]]
        except Exception:
            return None
        mo = misfit_opts or {}
        misfit, _basis, _unmatched = _modal_misfit_grid(
            results, observed_modes, n_p, n_t,
            weights=mo.get("weights"), presence_keys=mo.get("presence_keys"),
            open_other_tol=mo.get("open_other_tol", 0.0))
        return _pt_region_from_misfit(misfit, o["p_range"], o["t_range"],
                                      n_p, n_t, delta=delta)

    def _region_delta(self) -> float:
        """Accepted-region tolerance: cells with misfit <= min + δ (default 0.02).

        One value shared by the grid dialog's misfit view, the Monte Carlo
        misfit-mean view, the P–T-vs-scale tab and the CSV export, so the
        on-screen region, the exported rows and the figure agree.
        """
        return float(getattr(self, "_misfit_region_delta", 0.02))

    def _pt_scale_rows_for(self, parent_label: str,
                           parent_roi_bbox: Tuple[int, int, int, int],
                           delta: Optional[float] = None
                           ) -> Tuple[List[dict], str]:
        """P–T-vs-scale reduction of this ROI's sweep grids: ``(rows, problem)``.

        Per quadtree level, every {mean, mean±σ} sweep grid in the run history
        is scored against the OBSERVED assemblage (modal misfit with the
        phase-dialog weights / presence / open-assemblage settings) and reduced
        to its accepted region (misfit <= min + δ, ``_pt_region_from_grid``);
        ``_pt_scale_rows`` then gives, per level, the mean variant's best cell,
        its own region, the union over the variants, and the ±σ best cells.
        ``problem`` is a user-facing message when there is nothing to reduce.
        """
        delta = self._region_delta() if delta is None else float(delta)
        info = self._assemblage_inputs_from_state(
            self.last_phase_minerals, self.last_phase_names)
        observed_modes = info["modes"]
        if not observed_modes and not info["presence_keys"]:
            return [], ("The P–T estimate matches each scale's pseudosection "
                        "against the OBSERVED mineral assemblage and its modal "
                        "proportions. Segment phases and identify minerals for "
                        "this ROI first.")
        sweep_runs = [r for r in self.last_thermod_runs
                      if r.get("mode") == "grid"
                      and (r.get("opts") or {}).get("sweep_parent_label") == parent_label]
        if not sweep_runs:
            return [], ("No sweep grids for this ROI yet: run 'Run modelling "
                        "sweep …' first (the grids land in the run history).")
        by_level: Dict[int, Dict[str, dict]] = {}
        for r in sweep_runs:
            o = r["opts"]
            lv, rep = o.get("sweep_level"), o.get("sweep_rep")
            if lv is None or rep is None:
                continue
            by_level.setdefault(int(lv), {})[str(rep)] = r
        ry0, ry1, rx0, rx1 = parent_roi_bbox
        roi_h, roi_w = float(ry1 - ry0), float(rx1 - rx0)
        pt_by_level: Dict[int, Dict[str, dict]] = {}
        for lv in sorted(by_level):
            for rep, run in by_level[lv].items():
                # Region (argmin + near-minimal span), not just the argmin:
                # on a flat misfit valley the argmin hops with any
                # perturbation, while the region is stable at coarse scales
                # and widens only when the constraint genuinely degrades.
                pt = self._pt_region_from_grid(run, observed_modes,
                                               misfit_opts=info, delta=delta)
                if pt is not None:
                    pt_by_level.setdefault(lv, {})[rep] = pt
        rows = _pt_scale_rows(pt_by_level, roi_h, roi_w)
        if not rows:
            return [], ("No scale produced a usable P–T estimate (no grid cell "
                        "matched the observed assemblage). Check that the swept "
                        "P–T range covers the expected conditions.")
        return rows, ""

    def _export_pt_scale_csv(self, parent_label: str,
                             parent_roi_bbox: Tuple[int, int, int, int],
                             parent_dlg: QtWidgets.QDialog):
        """Write pt_scale.csv: per-scale best cell + accepted regions.

        Same reduction as the "P–T vs scale" tab (``_pt_scale_rows_for``);
        every column of the rows is written (best cell, union envelope, the
        mean variant's own region, the ±σ best cells). Feeds the manuscript
        scale figure's panel (c) (``figures/fig_scale.py``).
        """
        rows, problem = self._pt_scale_rows_for(parent_label, parent_roi_bbox)
        if problem:
            QtWidgets.QMessageBox.warning(parent_dlg, "P–T vs scale", problem)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent_dlg, "Export P-T vs scale CSV", "pt_scale.csv",
            "CSV files (*.csv)")
        if not path:
            return
        import csv
        cols = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in sorted(rows, key=lambda d: -d["window_px"]):
                w.writerow([("" if (isinstance(r[c], float) and np.isnan(r[c]))
                             else (f"{r[c]:.4f}" if c.startswith("P_")
                                   else f"{r[c]:.1f}"))
                            for c in cols])
        QtWidgets.QMessageBox.information(
            parent_dlg, "Saved",
            f"Wrote {len(rows)} scale(s) (δ = {self._region_delta():.3g}) to:\n{path}")

    def _run_composition_sweep(self, summary: dict, opts: dict,
                                parent_label: str,
                                parent_roi_bbox: Tuple[int, int, int, int],
                                parent_dlg: QtWidgets.QDialog,
                                records: Optional[List[dict]] = None):
        """Run a thermodynamic P-T grid for {μ, μ+σ, μ−σ} per scale.

        Per scale the ``μ±σ`` pair steps ±1 standard deviation along **PC1** of
        that scale's between-window composition spread
        (``_pca_perturbed_compositions``) — a closure-aware, ratio-changing
        direction whose magnitude reflects the real heterogeneity, instead of a
        marginal per-element ``±σ`` that renormalisation would largely cancel.
        Needs the per-window ``records``; without them it falls back to the
        (degenerate) marginal-σ build.

        Each grid is recorded via ``_record_thermod_run`` so it shows up in
        the Modelling-runs dialog and can be compared/exported like any
        other run.
        """
        # Pick a representative composition to seed _ThermoRunDialog: the
        # full-ROI mean (level 0). Convert to element-wt% format expected
        # by _ThermoRunDialog (key=element, value=wt%).
        levels = sorted(summary.keys())
        if not levels:
            QtWidgets.QMessageBox.warning(parent_dlg, "Empty",
                                          "No usable scales.")
            return
        seed = summary[0]["mean"] if 0 in summary else summary[levels[0]]["mean"]
        if not seed or sum(seed.values()) <= 0:
            QtWidgets.QMessageBox.warning(
                parent_dlg, "No seed composition",
                "Cannot derive a representative composition (level 0 has no signal)."
            )
            return

        # Open the existing thermod options dialog with the seed comp.
        # The user's choice of source dictates whether comp is wt% (already)
        # or net-count fractions (multiplied by 100 below).
        seed_for_dialog = dict(seed)  # already wt% in both EDS path (×100 done in worker) and SPX path
        run_dlg = _ThermoRunDialog(seed_for_dialog,
                                   "ROI mean (sweep seed)",
                                   parent=parent_dlg)
        if not run_dlg.exec_():
            return
        run_opts = run_dlg.get_options()
        run_opts.setdefault("extra_basis", "bulk")     # a fresh sweep: extras are wt% of the bulk
        if run_opts.get("mode") != "grid":
            QtWidgets.QMessageBox.warning(
                parent_dlg, "Grid mode required",
                "The sweep needs P-T grid mode to compare runs.\n"
                "Re-open the dialog and pick 'P-T grid (pseudosection)'."
            )
            return

        # Per-window compositions grouped by level (drives the PCA perturbation).
        windows_by_level: Dict[int, List[Dict[str, float]]] = {}
        for r in (records or []):
            wt = r.get("wt_pct")
            if wt:
                windows_by_level.setdefault(r["level"], []).append(wt)

        # Build the (μ, μ+σ, μ−σ) compositions per scale. The ±σ pair steps along
        # PC1 of the between-window spread (closure-aware, magnitude-faithful);
        # falls back to the marginal-σ build only when per-window records are
        # unavailable. All variants are clipped at 0 and renormalised to 100.
        sweep_comps: List[Tuple[int, str, Dict[str, float]]] = []
        for lv in levels:
            # Drop levels with no usable windows
            if summary[lv]["n_windows"] == 0:
                continue
            wins = windows_by_level.get(lv)
            if wins:
                mean_c, plus_c, minus_c = _pca_perturbed_compositions(
                    wins, summary[lv]["elements"], n_sigma=1.0)
                variants = [("mean", mean_c), ("mean+sigma", plus_c),
                            ("mean-sigma", minus_c)]
            else:
                # Fallback: marginal ±σ (no per-window data). Degenerate under
                # renormalisation but preserves behaviour when records is None.
                mean = summary[lv]["mean"]
                std = summary[lv]["std"]
                variants = []
                for rep_name, sign in [("mean", 0.0), ("mean+sigma", +1.0),
                                        ("mean-sigma", -1.0)]:
                    comp = {el: max(mu + sign * std.get(el, 0.0), 0.0)
                            for el, mu in mean.items()}
                    tot = sum(comp.values())
                    comp = ({k: 100.0 * v / tot for k, v in comp.items()}
                            if tot > 0.0 else {})
                    variants.append((rep_name, comp))
            for rep_name, comp in variants:
                if not comp or sum(comp.values()) <= 0.0:
                    continue
                sweep_comps.append((lv, rep_name, comp))

        if not sweep_comps:
            QtWidgets.QMessageBox.warning(parent_dlg, "Nothing to run",
                                          "All sweep compositions were empty.")
            return

        # Confirm.  For MAGEMin the grids run in a process pool (UI stays
        # responsive); the dialog also lets the user pick the worker count.
        n_runs = len(sweep_comps)
        n_p = run_opts["n_p"]; n_t = run_opts["n_t"]
        is_magemin = str(run_opts["backend"]).lower() == "magemin"
        n_workers = 1
        if is_magemin:
            n_workers = self._confirm_sweep_magemin(parent_dlg, n_runs, n_p, n_t)
            if n_workers is None:
                return
        else:
            ans = QtWidgets.QMessageBox.question(
                parent_dlg, "Confirm sweep",
                f"About to run {n_runs} P-T grids ({n_p}×{n_t} pts each, "
                f"backend={run_opts['backend']}).\n\n"
                "The UI will be busy until the sweep finishes. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return

        # Build backend once (to validate availability).  For MAGEMin we do
        # NOT activate Julia here — that happens in the worker subprocesses.
        try:
            from thermo import ThermoCalc
        except ImportError as exc:
            QtWidgets.QMessageBox.critical(
                parent_dlg, "thermo package not found", str(exc)
            )
            return
        try:
            calc = ThermoCalc(backend=run_opts["backend"],
                              **run_opts["backend_kwargs"])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                parent_dlg, "Backend error", str(exc)
            )
            return
        if not calc.check_backend():
            QtWidgets.QMessageBox.warning(
                parent_dlg, "Backend not available",
                f"Backend '{run_opts['backend']}' is not installed."
            )
            return

        if is_magemin:
            self._run_sweep_parallel(
                calc, run_opts, sweep_comps, opts,
                parent_label, parent_roi_bbox, parent_dlg, n_workers,
            )
        else:
            self._run_sweep_serial(
                calc, run_opts, sweep_comps, opts,
                parent_label, parent_roi_bbox, parent_dlg,
            )

    def _confirm_sweep_magemin(self, parent_dlg, n_runs, n_p, n_t):
        """Confirm a MAGEMin sweep and choose the parallel worker count.

        Returns the chosen worker count, or None if the user cancelled.
        """
        import os as _os
        from thermo.parallel import default_worker_count

        cores = _os.cpu_count() or 2
        default_w = default_worker_count(n_runs)
        env = _os.environ.get("THERMOD_SWEEP_WORKERS")
        if env:
            try:
                default_w = max(1, min(int(env), n_runs, cores))
            except ValueError:
                pass

        dlg = QtWidgets.QDialog(parent_dlg)
        dlg.setWindowTitle("Confirm MAGEMin sweep")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(
            f"About to run <b>{n_runs}</b> P-T grids "
            f"({n_p}×{n_t} pts each) with MAGEMin.<br><br>"
            "Each grid runs in a worker subprocess, so the viewer stays "
            "responsive and you can cancel between grids."
        ))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Parallel workers:"))
        spin = QtWidgets.QSpinBox()
        spin.setRange(1, max(1, min(n_runs, cores)))
        spin.setValue(max(1, min(default_w, n_runs, cores)))
        row.addWidget(spin)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(QtWidgets.QLabel(
            "<i>Each worker boots its own Julia + MAGEMin session "
            "(~20 s, ~1–2 GB RAM). More workers = faster, but watch RAM.</i>"
        ))
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if not dlg.exec_():
            return None
        return int(spin.value())

    def _record_sweep_run(self, lv, rep, comp_el, results, run_opts, opts, parent_label):
        """Append one sweep grid to the run history with scale-grouped labels."""
        run_opts_for_this = dict(run_opts)
        run_opts_for_this["sweep_level"] = lv
        run_opts_for_this["sweep_rep"] = rep
        run_opts_for_this["sweep_parent_label"] = parent_label
        dt = opts.get("data_type")
        if dt in ("oxford_wt_pct", "oxford_counts"):
            src_tag = self._oxford_source_label(dt, short=True)
        else:
            src_tag = opts.get("source", "?")
        source_label = f"sweep L{lv} {rep} ({src_tag})"
        run = self._record_thermod_run(
            "grid", run_opts_for_this, comp_el, source_label, results,
        )
        run["roi_label"] = f"{run.get('roi_label', '')}  [{source_label}]"
        return run

    @staticmethod
    def _build_sweep_comp(comp_el, run_opts):
        """Build the Composition for one sweep grid (matches the single-run path)."""
        from thermo import Composition
        comp = Composition.from_element_wt(comp_el)
        extra = {}
        if run_opts.get("h2o", 0) > 0:
            extra["H2O"] = run_opts["h2o"]
        if run_opts.get("co2", 0) > 0:
            extra["CO2"] = run_opts["co2"]
        if extra:
            ox = comp.to_oxide_wt(
                fe3_fraction=run_opts.get("fe3", 0.0),
                extra_oxides=extra, normalize=True,
                extra_basis=run_opts.get("extra_basis", "legacy"),
            )
            comp = Composition.from_oxide_wt(ox)
        return comp

    # ---- Monte Carlo bulk-composition uncertainty ----------------------------

    def _build_mc_comp(self, comp, run_opts):
        """Composition for one MC grid (honours custom-oxide runs + extras)."""
        from thermo import Composition
        if run_opts.get("comp_source") == "custom":
            c = Composition.from_oxide_wt(comp)
        else:
            c = Composition.from_element_wt(comp)
        extra = {}
        if run_opts.get("h2o", 0) > 0:
            extra["H2O"] = run_opts["h2o"]
        if run_opts.get("co2", 0) > 0:
            extra["CO2"] = run_opts["co2"]
        if extra:
            ox = c.to_oxide_wt(fe3_fraction=run_opts.get("fe3", 0.0),
                               extra_oxides=extra, normalize=True,
                               extra_basis=run_opts.get("extra_basis", "legacy"))
            c = Composition.from_oxide_wt(ox)
        return c

    @staticmethod
    def _mc_worker_count() -> int:
        import os as _os
        env = _os.environ.get("THERMOD_SWEEP_WORKERS")
        if env:
            try:
                return max(1, int(env))
            except ValueError:
                pass
        return min(4, max(1, (_os.cpu_count() or 4) // 2))

    def _run_monte_carlo(self, run_opts, comp_el, calc,
                         observed=None, modes=None, parent_dlg=None):
        """Run N perturbed copies of a grid and aggregate uncertainty maps.

        Compositions are sampled by ``_mc_sample_compositions`` (relative σ,
        renormalised, seeded). MAGEMin grids go through the subprocess pool
        (``_ThermoSweepWorker``, serial fallback); perplex/pycalphad run a
        serial in-process loop. All paths are grid-cache-aware, so re-running
        an identical MC is instant. Results are aggregated (not added to the
        run history) and shown by ``_show_mc_result``.
        """
        parent_dlg = parent_dlg or self

        # ---- options ----------------------------------------------------
        opt = QtWidgets.QDialog(parent_dlg)
        opt.setWindowTitle("Monte Carlo settings")
        form = QtWidgets.QFormLayout(opt)
        spin_n = QtWidgets.QSpinBox(); spin_n.setRange(4, 256); spin_n.setValue(16)
        spin_sig = QtWidgets.QDoubleSpinBox()
        spin_sig.setRange(0.5, 50.0); spin_sig.setValue(5.0)
        spin_sig.setSuffix(" %"); spin_sig.setDecimals(1)
        spin_seed = QtWidgets.QSpinBox(); spin_seed.setRange(0, 99999); spin_seed.setValue(0)
        form.addRow("Runs (N):", spin_n)
        form.addRow("Relative σ per component:", spin_sig)
        form.addRow("Seed:", spin_seed)
        note = QtWidgets.QLabel(
            "Each run perturbs every composition component by an independent "
            "Gaussian (σ = the chosen fraction of its value, truncated at 0, "
            "renormalised). Identical settings reuse the grid cache.")
        note.setWordWrap(True)
        form.addRow(note)
        # Observation (mode / mineral-ID) uncertainty — folded into the best-fit
        # P-T cloud by RE-SCORING the same grids, so it adds ~no compute.
        chk_modes = QtWidgets.QCheckBox("Also resample observed modes")
        spin_modesig = QtWidgets.QDoubleSpinBox()
        spin_modesig.setRange(1.0, 80.0); spin_modesig.setValue(20.0)
        spin_modesig.setSuffix(" %"); spin_modesig.setDecimals(0)
        spin_modesig.setEnabled(False)
        chk_modes.toggled.connect(spin_modesig.setEnabled)
        chk_modes.setEnabled(bool(modes))
        chk_modes.setToolTip(
            "Propagate uncertainty in the OBSERVED modal proportions (and hence "
            "mineral-ID errors) into the best-fit P-T scatter. This only re-scores "
            "the grids already computed for the composition runs, so it costs "
            "essentially nothing.")
        form.addRow(chk_modes, spin_modesig)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(opt.accept); bb.rejected.connect(opt.reject)
        form.addRow(bb)
        if not opt.exec_():
            return
        n = int(spin_n.value())
        sigma = float(spin_sig.value()) / 100.0
        seed = int(spin_seed.value())
        mode_samples = None
        if chk_modes.isChecked() and modes:
            mode_samples = _mc_sample_modes(
                modes, max(n, 32), seed=seed,
                rel_sigma=float(spin_modesig.value()) / 100.0)
        # Relaxed-assumption misfit settings (weights / presence / open) ride
        # along in run_opts so MC scores exactly like the rest of the pipeline.
        misfit_kwargs = {
            "weights": run_opts.get("modal_weights"),
            "presence_keys": run_opts.get("presence_keys"),
            "open_other_tol": run_opts.get("open_other_tol", 0.0),
        }

        comps = _mc_sample_compositions(comp_el, sigma, n, seed)
        n_p, n_t = int(run_opts["n_p"]), int(run_opts["n_t"])
        grids: List[Optional[list]] = [None] * n
        mc_meta = {"n": n, "sigma_pct": sigma * 100.0, "seed": seed,
                   "backend": run_opts.get("backend", "?"),
                   "mode_sigma_pct": (float(spin_modesig.value())
                                      if mode_samples else None)}

        if calc is None:
            try:
                from thermo import ThermoCalc
                calc = ThermoCalc(backend=run_opts["backend"],
                                  **(run_opts.get("backend_kwargs") or {}))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    parent_dlg, "Backend error", str(exc))
                return

        def _finish():
            agg = _mc_aggregate(grids, n_p, n_t,
                                observed_phases=observed,
                                observed_modes=modes,
                                misfit_kwargs=misfit_kwargs,
                                mode_samples=mode_samples)
            if agg["n_runs"] == 0:
                QtWidgets.QMessageBox.warning(
                    parent_dlg, "Monte Carlo",
                    "Every perturbed grid failed (no stable assemblage?).")
                return
            self._show_mc_result(agg, mc_meta, run_opts, parent_dlg)

        def _run_serial(indices):
            pd_prog = QtWidgets.QProgressDialog(
                "Running Monte Carlo grids…", "Cancel", 0, len(indices),
                parent_dlg)
            pd_prog.setWindowTitle("Monte Carlo")
            pd_prog.setWindowModality(QtCore.Qt.WindowModal)
            pd_prog.setMinimumDuration(0)
            pd_prog.show()
            if hasattr(calc._backend, "_activate"):
                calc._backend._activate()
            for k, idx in enumerate(indices, start=1):
                QtWidgets.QApplication.processEvents()
                if pd_prog.wasCanceled():
                    break
                pd_prog.setLabelText(f"Grid {k}/{len(indices)}")
                try:
                    grids[idx] = calc.run_grid(
                        self._build_mc_comp(comps[idx], run_opts),
                        p_range=run_opts["p_range"],
                        t_range=run_opts["t_range"],
                        n_p=n_p, n_t=n_t)
                except Exception as exc:
                    print(f"[mc] grid {idx} failed: {exc}", flush=True)
                pd_prog.setValue(k)
            pd_prog.close()
            _finish()

        if run_opts.get("backend") == "magemin":
            from thermo.result import result_from_portable_dict
            jobs = [{
                "comp": self._build_mc_comp(c, run_opts),
                "p_range": tuple(run_opts["p_range"]),
                "t_range": tuple(run_opts["t_range"]),
                "n_p": n_p, "n_t": n_t,
            } for c in comps]
            pd_prog = QtWidgets.QProgressDialog(
                "Running Monte Carlo grids…", "Cancel", 0, n, parent_dlg)
            pd_prog.setWindowTitle("Monte Carlo")
            pd_prog.setWindowModality(QtCore.Qt.WindowModal)
            pd_prog.setMinimumDuration(0)
            pd_prog.setValue(0)
            pd_prog.show()

            worker = _ThermoSweepWorker(
                jobs, "magemin", run_opts.get("backend_kwargs") or {},
                max_workers=self._mc_worker_count(), parent=self)

            def _on_grid(idx, dicts):
                grids[idx] = [result_from_portable_dict(d) for d in dicts]

            def _on_progress(done, total):
                pd_prog.setLabelText(f"Completed {done}/{total} grids")
                pd_prog.setValue(done)

            def _on_done():
                pd_prog.close()
                self._mc_worker = None
                _finish()

            def _on_pool_failed(msg):
                pd_prog.close()
                self._mc_worker = None
                print(f"[mc] parallel pool failed ({msg}); serial fallback.",
                      flush=True)
                _run_serial([i for i in range(n) if grids[i] is None])

            worker.grid_done.connect(_on_grid)
            worker.grid_failed.connect(
                lambda idx, msg: print(f"[mc] grid {idx} failed: {msg}",
                                       flush=True))
            worker.progress.connect(_on_progress)
            worker.finished_all.connect(_on_done)
            worker.pool_failed.connect(_on_pool_failed)
            pd_prog.canceled.connect(worker.cancel)
            worker.start()
            self._mc_worker = worker
        else:
            _run_serial(list(range(n)))

    def _show_mc_result(self, agg, mc_meta, run_opts, parent_dlg):
        """Display aggregated Monte Carlo uncertainty maps."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )

        n_p, n_t = int(run_opts["n_p"]), int(run_opts["n_t"])
        t_vals = np.linspace(run_opts["t_range"][0], run_opts["t_range"][1], n_t)
        p_vals = np.linspace(run_opts["p_range"][0], run_opts["p_range"][1], n_p)

        dlg = QtWidgets.QDialog(parent_dlg)
        dlg.setWindowTitle("Monte Carlo — bulk-composition uncertainty")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel(
            f"<b>{agg['n_runs']} runs</b>"
            + (f" ({agg['n_failed']} failed)" if agg.get("n_failed") else "")
            + f"  ·  σ = {mc_meta['sigma_pct']:.1f}% per component"
            f"  ·  seed {mc_meta['seed']}  ·  {mc_meta['backend']}"))

        views = []
        if "stability_prob" in agg:
            views.append("P(assemblage stable)")
        if "misfit_mean" in agg:
            views += ["Misfit mean", "Misfit std", "Best-fit P-T scatter"]
        cmb = QtWidgets.QComboBox()
        for v in views:
            cmb.addItem(v)
        lay.addWidget(cmb)

        fig = plt.figure(figsize=(6, 5))
        canvas = FigureCanvas(fig)
        lay.addWidget(NavigationToolbar(canvas, dlg))
        lay.addWidget(canvas)

        def _replot():
            fig.clear()
            ax = fig.add_subplot(111)
            view = cmb.currentText()
            if view == "P(assemblage stable)":
                im = ax.pcolormesh(t_vals, p_vals, agg["stability_prob"],
                                   cmap='viridis', vmin=0, vmax=1,
                                   shading='auto')
                fig.colorbar(im, ax=ax, label="P(all observed phases stable)")
                T2d, P2d = np.meshgrid(t_vals, p_vals)
                if 0.0 < float(np.nanmax(agg["stability_prob"])) and \
                        float(np.nanmin(agg["stability_prob"])) < 1.0:
                    ax.contour(T2d, P2d, agg["stability_prob"], levels=[0.5],
                               colors='white', linestyles='--', linewidths=1.2)
                ax.set_title("Assemblage stability probability", fontsize=10)
            elif view == "Misfit mean":
                im = ax.pcolormesh(t_vals, p_vals, agg["misfit_mean"],
                                   cmap='RdYlGn_r', vmin=0, vmax=1,
                                   shading='auto')
                fig.colorbar(im, ax=ax, label="mean modal misfit")
                m = agg["misfit_mean"]
                title = "Modal misfit — mean over runs"
                if np.isfinite(m).any():
                    i, j = np.unravel_index(int(np.nanargmin(m)), m.shape)
                    ax.plot(t_vals[j], p_vals[i], marker='*', markersize=16,
                            markeredgecolor='black', markerfacecolor='white')
                    # Accepted region of the mean misfit (misfit <= min + δ).
                    delta = self._region_delta()
                    reg = _pt_region_from_misfit(m, run_opts["p_range"],
                                                 run_opts["t_range"], n_p, n_t,
                                                 delta=delta)
                    if reg is not None:
                        try:
                            ax.contour(t_vals, p_vals, m,
                                       levels=[float(np.nanmin(m)) + delta],
                                       colors='white', linestyles='--',
                                       linewidths=1.3)
                        except Exception:
                            pass
                        title += (f"\naccepted (≤ min + {delta:.3g}): "
                                  f"T {reg['t_lo']:.0f}–{reg['t_hi']:.0f}°C, "
                                  f"P {reg['p_lo']:.2f}–{reg['p_hi']:.2f} GPa")
                ax.set_title(title, fontsize=10)
            elif view == "Misfit std":
                im = ax.pcolormesh(t_vals, p_vals, agg["misfit_std"],
                                   cmap='magma', shading='auto')
                fig.colorbar(im, ax=ax, label="σ of modal misfit")
                ax.set_title("Modal misfit — spread over runs", fontsize=10)
            else:  # Best-fit P-T scatter
                best = agg.get("best_fits") or []
                from collections import Counter
                cnt = Counter((i, j) for i, j, _ in best)
                if cnt:
                    ii, jj = zip(*cnt.keys())
                    sizes = np.array([cnt[k] for k in cnt.keys()], dtype=float)
                    ax.scatter(t_vals[list(jj)], p_vals[list(ii)],
                               s=40 + 60 * sizes / sizes.max(),
                               c='tab:blue', alpha=0.6, edgecolors='black',
                               label="per-run best fit")
                    med_i = float(np.median([i for i, _, _ in best]))
                    med_j = float(np.median([j for _, j, _ in best]))
                    ax.plot(np.interp(med_j, np.arange(n_t), t_vals),
                            np.interp(med_i, np.arange(n_p), p_vals),
                            marker='*', markersize=18,
                            markeredgecolor='black', markerfacecolor='red',
                            linestyle='none', label="median")
                    ax.legend(fontsize=8)
                ax.set_xlim(t_vals[0], t_vals[-1])
                ax.set_ylim(p_vals[0], p_vals[-1])
                ax.set_title(
                    f"Best-fit P-T under bulk uncertainty ({len(best)} runs)",
                    fontsize=10)
            ax.set_xlabel("T [°C]")
            ax.set_ylabel("P [GPa]")
            canvas.draw()

        cmb.currentTextChanged.connect(lambda _: _replot())

        btn_row = QtWidgets.QHBoxLayout()
        btn_csv = QtWidgets.QPushButton("Export CSV …")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(btn_csv)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        btn_close.clicked.connect(dlg.accept)

        def _save_csv():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                dlg, "Export Monte Carlo CSV", "monte_carlo.csv",
                "CSV files (*.csv)")
            if not path:
                return
            import csv
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["# Monte Carlo", f"n={agg['n_runs']}",
                            f"sigma_pct={mc_meta['sigma_pct']}",
                            f"seed={mc_meta['seed']}"])
                for key in ("stability_prob", "misfit_mean", "misfit_std"):
                    if key not in agg:
                        continue
                    w.writerow([])
                    w.writerow([key] + [f"T={t:.1f}" for t in t_vals])
                    for i in range(n_p):
                        w.writerow([f"P={p_vals[i]:.3f}"]
                                   + [f"{v:.4f}" if np.isfinite(v) else ""
                                      for v in agg[key][i]])
                if agg.get("best_fits"):
                    w.writerow([])
                    w.writerow(["best_fit_T_C", "best_fit_P_GPa", "misfit"])
                    for i, j, m in agg["best_fits"]:
                        w.writerow([f"{t_vals[j]:.1f}", f"{p_vals[i]:.3f}",
                                    f"{m:.4f}"])
        btn_csv.clicked.connect(_save_csv)

        _replot()
        dlg.resize(720, 640)
        dlg.exec_()

    def _run_sweep_parallel(self, calc, run_opts, sweep_comps, opts,
                             parent_label, parent_roi_bbox, parent_dlg, n_workers):
        """Run a MAGEMin sweep across a process pool, recording grids as they finish."""
        from thermo.result import result_from_portable_dict

        n_runs = len(sweep_comps)
        jobs = []
        for (lv, rep, comp_el) in sweep_comps:
            jobs.append({
                "comp": self._build_sweep_comp(comp_el, run_opts),
                "p_range": tuple(run_opts["p_range"]),
                "t_range": tuple(run_opts["t_range"]),
                "n_p": int(run_opts["n_p"]),
                "n_t": int(run_opts["n_t"]),
            })

        pd_prog = QtWidgets.QProgressDialog(
            "Running thermodynamic sweep…", "Cancel", 0, n_runs, parent_dlg
        )
        pd_prog.setWindowTitle("Composition-vs-scale sweep")
        pd_prog.setWindowModality(QtCore.Qt.WindowModal)
        pd_prog.setMinimumDuration(0)
        pd_prog.setValue(0)
        pd_prog.show()

        # _capture_roi_for_run reads self.last_roi; point every sweep record at
        # the parent ROI for the duration of the run, then restore.
        prev_roi = self.last_roi
        prev_roi_mask = self.last_roi_mask
        self.last_roi = parent_roi_bbox

        state = {"done": 0, "completed": set()}

        def _restore():
            self.last_roi = prev_roi
            self.last_roi_mask = prev_roi_mask

        worker = _ThermoSweepWorker(
            jobs, "magemin", run_opts["backend_kwargs"],
            max_workers=n_workers, parent=self,
        )

        def _on_grid(idx, dicts):
            lv, rep, comp_el = sweep_comps[idx]
            results = [result_from_portable_dict(d) for d in dicts]
            self._record_sweep_run(lv, rep, comp_el, results,
                                   run_opts, opts, parent_label)
            state["completed"].add(idx)

        def _on_grid_failed(idx, msg):
            lv, rep, _ = sweep_comps[idx]
            print(f"[sweep] grid L{lv} {rep} failed: {msg}", flush=True)
            state["completed"].add(idx)  # don't re-run a grid that genuinely errored

        def _on_progress(done, total):
            state["done"] = done
            pd_prog.setLabelText(f"Completed {done}/{total} grids")
            pd_prog.setValue(done)

        def _on_done():
            was_cancelled = pd_prog.wasCanceled()
            pd_prog.close()
            _restore()
            self._sweep_worker = None
            n_added = len(state["completed"])
            if was_cancelled:
                QtWidgets.QMessageBox.information(
                    parent_dlg, "Sweep cancelled",
                    f"Cancelled after {n_added}/{n_runs} grids.\n"
                    "Completed runs are kept in the modelling-runs history."
                )
            else:
                QtWidgets.QMessageBox.information(
                    parent_dlg, "Sweep complete",
                    f"Added {n_added} runs to the history. Open "
                    "'Modelling runs …' and select them to compare."
                )

        def _on_pool_failed(msg):
            _restore()
            self._sweep_worker = None
            pd_prog.close()
            # Subprocess pool unusable (e.g. Julia won't init in a subprocess).
            # Fall back to the in-process serial loop for the grids not yet done.
            remaining = [sc for i, sc in enumerate(sweep_comps)
                         if i not in state["completed"]]
            print(f"[sweep] parallel pool failed ({msg}); "
                  f"falling back to serial for {len(remaining)} grid(s).",
                  flush=True)
            if remaining:
                self._run_sweep_serial(
                    calc, run_opts, remaining, opts,
                    parent_label, parent_roi_bbox, parent_dlg,
                )

        worker.grid_done.connect(_on_grid)
        worker.grid_failed.connect(_on_grid_failed)
        worker.progress.connect(_on_progress)
        worker.finished_all.connect(_on_done)
        worker.pool_failed.connect(_on_pool_failed)
        pd_prog.canceled.connect(worker.cancel)
        worker.start()
        self._sweep_worker = worker

    def _run_sweep_serial(self, calc, run_opts, sweep_comps, opts,
                           parent_label, parent_roi_bbox, parent_dlg):
        """Serial in-process sweep (perplex/pycalphad, and the MAGEMin fallback)."""
        n_runs = len(sweep_comps)
        if hasattr(calc._backend, "_activate"):
            calc._backend._activate()

        pd_prog = QtWidgets.QProgressDialog(
            "Running thermodynamic sweep…", "Cancel", 0, n_runs, parent_dlg
        )
        pd_prog.setWindowTitle("Composition-vs-scale sweep")
        pd_prog.setWindowModality(QtCore.Qt.WindowModal)
        pd_prog.setMinimumDuration(0)
        pd_prog.setValue(0)
        pd_prog.show()

        prev_roi = self.last_roi
        prev_roi_mask = self.last_roi_mask
        self.last_roi = parent_roi_bbox

        cancelled = False
        n_ok = 0
        failures: List[str] = []
        k = 0
        for k, (lv, rep, comp_el) in enumerate(sweep_comps, start=1):
            QtWidgets.QApplication.processEvents()
            if pd_prog.wasCanceled():
                cancelled = True
                break
            pd_prog.setLabelText(
                f"Running grid {k}/{n_runs}  —  level L{lv}, {rep}"
            )
            # A single degenerate window (e.g. a non-physical bulk that VERTEX
            # can't stabilise → empty pseudosection, which run_grid now raises
            # on) must NOT abort the whole sweep. Record the failure and carry
            # on so the usable scales still populate the history.
            try:
                comp = self._build_sweep_comp(comp_el, run_opts)
                results = calc.run_grid(
                    comp,
                    p_range=run_opts["p_range"],
                    t_range=run_opts["t_range"],
                    n_p=run_opts["n_p"],
                    n_t=run_opts["n_t"],
                )
            except Exception as exc:
                failures.append(f"L{lv} {rep}: {exc}")
                print(f"[sweep] grid {k}/{n_runs} (L{lv} {rep}) failed: {exc}",
                      flush=True)
                pd_prog.setValue(k)
                continue

            self._record_sweep_run(lv, rep, comp_el, results,
                                   run_opts, opts, parent_label)
            n_ok += 1
            pd_prog.setValue(k)

        pd_prog.close()
        self.last_roi = prev_roi
        self.last_roi_mask = prev_roi_mask

        if cancelled:
            QtWidgets.QMessageBox.information(
                parent_dlg, "Sweep cancelled",
                f"Cancelled after {k - 1}/{n_runs} grids "
                f"({n_ok} succeeded, {len(failures)} failed).\n"
                "Completed runs are kept in the modelling-runs history."
            )
            return

        msg = (f"Added {n_ok} run(s) to the history. Open "
               "'Modelling runs …' and select them to compare.")
        if failures:
            head = "\n".join(f"  • {f}" for f in failures[:8])
            more = "" if len(failures) <= 8 else f"\n  … and {len(failures) - 8} more"
            msg += (f"\n\n{len(failures)} of {n_runs} grids produced no stable "
                    f"assemblage and were skipped (usually a non-physical bulk — "
                    f"e.g. uncalibrated EDS net-counts, or a single-mineral window "
                    f"at fine scales):\n{head}{more}")
        QtWidgets.QMessageBox.information(parent_dlg, "Sweep complete", msg)

    # ---- Session save / load ------------------------------------------------

    def on_save_session(self):
        """Save computed ROI maps, ROI geometry, compositions, and phase maps to disk."""
        has_data = (self.last_band_maps or self.last_composition
                    or self.last_spx_composition or self.last_phase_map is not None
                    or self.last_thermod_runs)
        if not has_data:
            QtWidgets.QMessageBox.information(
                self, "Nothing to save",
                "Compute ROI maps, a composition, a phase segmentation, "
                "or run a thermodynamic model first."
            )
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", str(self.manifest.parent / "viewer_session.json"),
            "Session files (*.json);;All files (*)",
        )
        if not path:
            return
        session_path = Path(path)
        npz_path = session_path.with_suffix(".npz")

        # ── Metadata (JSON) ──────────────────────────────────────────────
        meta: dict = {
            "manifest": str(self.manifest),
            "available_elements": self.available_elements,
            "selected_elements": self._selected_elements(),
            "ui": {
                "window_width_ev": self.win_spin.value(),
                "sum_counts": self.sum_check.isChecked(),
                "apply_subpixel": self.subpix_check.isChecked(),
                "shift_method": str(self.shift_method_combo.currentData() or "bilinear"),
            },
        }

        if self.last_roi is not None:
            meta["roi"] = list(self.last_roi)
        if self.last_band_maps_ds and self.last_band_maps_ds != 1:
            meta["band_maps_ds"] = int(self.last_band_maps_ds)
            meta["blend_mode"] = str(self.last_blend_mode)
        if self.last_elements:
            meta["last_elements"] = self.last_elements
        if self.last_composition:
            meta["last_composition"] = self.last_composition
        if self.last_spx_composition:
            meta["last_spx_composition"] = self.last_spx_composition
        if self.last_calibrated_composition:
            meta["last_calibrated_composition"] = self.last_calibrated_composition
        if self.eds_calibration is not None:
            meta["eds_calibration"] = self.eds_calibration.to_dict()
            meta["calibration_scope"] = self._calibration_scope
            meta["calibration_fingerprint"] = self._calibration_fingerprint
        # Net-count background scheme every composition in this session used.
        meta["background_scheme"] = _bg_scheme()

        # Phase segmentation
        if self.last_phase_map is not None:
            phase_meta: dict = {}
            if self.last_phase_names:
                # JSON keys must be strings
                phase_meta["names"] = {str(k): v for k, v in self.last_phase_names.items()}
            if self.last_phase_translate:
                phase_meta["translate"] = list(self.last_phase_translate)
            if self.last_phase_ds and self.last_phase_ds != 1:
                phase_meta["ds"] = int(self.last_phase_ds)
            if self.last_phase_minerals:
                # JSON keys must be strings
                phase_meta["minerals"] = {str(k): v for k, v in self.last_phase_minerals.items()}
                phase_meta["fe3_fraction"] = float(self.last_phase_fe3)
            if self.last_phase_comp_erosion:
                phase_meta["comp_erosion_px"] = int(self.last_phase_comp_erosion)
            if self.last_phase_unmix:
                phase_meta["unmix"] = {str(c): {
                    "fractions": {str(e): float(f)
                                  for e, f in (d.get("fractions") or {}).items()},
                    "residual": float(d.get("residual", float("nan"))),
                } for c, d in self.last_phase_unmix.items()}
            if self.last_phase_composition:
                phase_meta["composition"] = {
                    str(cid): {el: round(float(v), 6) for el, v in frac.items()}
                    for cid, frac in self.last_phase_composition.items()}
            if self.last_phase_source and self.last_phase_source != "gmm":
                phase_meta["source"] = self.last_phase_source
            if self.last_phase_seg_params:
                phase_meta["seg_params"] = self.last_phase_seg_params
            if self._coating_override is not None:
                # Persist only an explicit override; a None override re-resolves
                # from the cache manifest on load (keeps sessions portable).
                phase_meta["coating_elements"] = sorted(self._coating_override)
            # P-T inversion overrides (only when set, to keep sessions small).
            if self.last_phase_pt_include is not None:
                phase_meta["pt_include"] = sorted(int(c) for c in self.last_phase_pt_include)
            if self.last_phase_pt_weights:
                phase_meta["pt_weights"] = {str(k): float(v)
                                            for k, v in self.last_phase_pt_weights.items()}
            if self.last_phase_extra:
                phase_meta["pt_extra"] = [dict(e) for e in self.last_phase_extra]
            if self.last_phase_open_tol:
                phase_meta["pt_open_tol"] = float(self.last_phase_open_tol)
            meta["phase_segmentation"] = phase_meta

        # Thermodynamic modelling runs (rich enough to re-open the result
        # dialog without recomputing — see _record_thermod_run /
        # _open_thermod_run).
        if self.last_thermod_runs:
            meta["thermod_runs"] = self.last_thermod_runs

        # Last composition-vs-scale result, so it re-opens without recompute.
        if self.last_comp_scale and self.last_comp_scale.get("records"):
            cs = self.last_comp_scale
            meta["comp_scale"] = {
                "records": cs["records"],
                "summary": {str(k): v for k, v in (cs.get("summary") or {}).items()},
                "opts": cs.get("opts"),
                "parent_label": cs.get("parent_label"),
                "roi_bbox": cs.get("roi_bbox"),
            }

        # Save ROI shapes (vertices + type)
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Shapes) and lyr.name.lower() == 'roi':
                shapes_data = []
                for i, verts in enumerate(lyr.data):
                    stype = lyr.shape_type[i] if hasattr(lyr, 'shape_type') else 'rectangle'
                    shapes_data.append({
                        "vertices": np.asarray(verts).tolist(),
                        "type": str(stype),
                    })
                meta["roi_shapes"] = shapes_data
                break

        import json
        session_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # ── Array data (NPZ) ─────────────────────────────────────────────
        arrays: Dict[str, np.ndarray] = {}
        if self.last_roi_mask is not None:
            arrays["roi_mask"] = self.last_roi_mask.astype(np.uint8)
        if self.last_band_maps:
            for el, bands in self.last_band_maps.items():
                for band_key, arr in bands.items():
                    arrays[f"{el}__{band_key}"] = arr.astype(np.float32)
        if self.last_phase_map is not None:
            arrays["phase_map"] = self.last_phase_map.astype(np.int16)
        if arrays:
            np.savez_compressed(npz_path, **arrays)

        saved = [str(session_path)]
        if arrays:
            saved.append(str(npz_path))
        print(f"[session] Saved: {', '.join(saved)}", flush=True)
        QtWidgets.QMessageBox.information(
            self, "Session saved",
            f"Session saved to:\n{session_path}"
            + (f"\nArray data: {npz_path}" if arrays else "")
            + (f"\n\nBand maps: {len(self.last_band_maps)} elements" if self.last_band_maps else "")
        )

    def on_load_session(self):
        """Load a previously saved session: ROI maps, geometry, compositions."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", str(self.manifest.parent),
            "Session files (*.json);;All files (*)",
        )
        if not path:
            return
        self._load_session(Path(path))

    def _load_session(self, session_path: Path):
        """Restore session state from JSON + NPZ files."""
        import json

        if not session_path.exists():
            QtWidgets.QMessageBox.critical(self, "File not found", str(session_path))
            return

        meta = json.loads(session_path.read_text(encoding="utf-8"))

        # Verify manifest matches
        saved_manifest = meta.get("manifest", "")
        if Path(saved_manifest).resolve() != self.manifest.resolve():
            ans = QtWidgets.QMessageBox.question(
                self, "Different dataset",
                f"This session was saved for a different manifest:\n"
                f"  Saved: {saved_manifest}\n"
                f"  Current: {self.manifest}\n\n"
                "Load anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return

        # ── Restore UI state ──────────────────────────────────────────────
        ui = meta.get("ui", {})
        if "window_width_ev" in ui:
            self.win_spin.setValue(ui["window_width_ev"])
        if "sum_counts" in ui:
            self.sum_check.setChecked(ui["sum_counts"])
        if "apply_subpixel" in ui:
            self.subpix_check.setChecked(ui["apply_subpixel"])
        if "shift_method" in ui:
            saved = str(ui["shift_method"])
            for idx in range(self.shift_method_combo.count()):
                if self.shift_method_combo.itemData(idx) == saved:
                    self.shift_method_combo.setCurrentIndex(idx)
                    break

        # Restore element selection
        selected = set(meta.get("selected_elements", []))
        if selected:
            for i in range(self.el_list.count()):
                item = self.el_list.item(i)
                item.setCheckState(
                    QtCore.Qt.Checked if item.text() in selected else QtCore.Qt.Unchecked
                )

        # ── Restore cached state ──────────────────────────────────────────
        if "roi" in meta:
            self.last_roi = tuple(meta["roi"])
        self.last_band_maps_ds = int(meta.get("band_maps_ds", 1) or 1)
        self.last_blend_mode = str(meta.get("blend_mode", "mean"))
        if "last_elements" in meta:
            self.last_elements = meta["last_elements"]
        if "last_composition" in meta:
            self.last_composition = meta["last_composition"]
        if "last_spx_composition" in meta:
            self.last_spx_composition = meta["last_spx_composition"]
        if "last_calibrated_composition" in meta:
            self.last_calibrated_composition = meta["last_calibrated_composition"]
        try:
            from eds_background import set_default_scheme as _set_bg
            _scheme = meta.get("background_scheme")
            if not _scheme:
                _scheme = "sideband"
                print("[session] no background scheme recorded (pre-2026-09 session): "
                      "replaying on the legacy 'sideband' net counts; recompute the ROI "
                      "and recalibrate to move to the line-aware background.", flush=True)
            _set_bg(_scheme)
            print(f"[session] net-count background: {_scheme}", flush=True)
        except Exception as _e:
            print(f"[session] background scheme not restored: {_e}", flush=True)
        if "eds_calibration" in meta:
            try:
                import eds_quantify as _eq
                self.eds_calibration = _eq.Calibration.from_dict(meta["eds_calibration"])
                # Missing scope on an older session -> treat as ROI-scoped (the
                # old behaviour: cleared on the next ROI recompute).
                self._calibration_scope = meta.get("calibration_scope") or "roi"
                self._calibration_fingerprint = meta.get("calibration_fingerprint")
            except Exception:
                self.eds_calibration = None
                self._calibration_scope = None
                self._calibration_fingerprint = None

        # ── Restore ROI shapes ────────────────────────────────────────────
        roi_shapes_data = meta.get("roi_shapes")
        if roi_shapes_data:
            # Find or create Shapes layer
            shapes_lyr = None
            for lyr in self.viewer.layers:
                if isinstance(lyr, napari.layers.Shapes) and lyr.name.lower() == 'roi':
                    shapes_lyr = lyr
                    break
            if shapes_lyr is None:
                shapes_lyr = self.viewer.add_shapes(
                    name='ROI',
                    edge_color='yellow',
                    face_color='transparent',
                    edge_width=2,
                )
            # Add saved shapes
            for s in roi_shapes_data:
                verts = np.array(s["vertices"])
                stype = s.get("type", "rectangle")
                shapes_lyr.add(verts, shape_type=stype)

        # ── Restore band maps from NPZ ────────────────────────────────────
        npz_path = session_path.with_suffix(".npz")
        if npz_path.exists():
            data = np.load(npz_path)

            # ROI mask
            if "roi_mask" in data:
                self.last_roi_mask = data["roi_mask"].astype(bool)

            # Band maps: keys are "{el}__{band}" e.g. "Si__main", "Si__sb_lo"
            band_maps: Dict[str, Dict[str, np.ndarray]] = {}
            for key in data.files:
                if key == "roi_mask":
                    continue
                if "__" in key:
                    el, band = key.split("__", 1)
                    if el not in band_maps:
                        band_maps[el] = {}
                    band_maps[el][band] = data[key].astype(np.float32)
            if band_maps:
                self.last_band_maps = band_maps
                self.last_elements = sorted(band_maps.keys())
            data.close()

            # Re-add computed EDS layers to napari
            if self.last_band_maps and self.last_roi:
                y0, _, x0, _ = self.last_roi
                _ds = max(1, int(self.last_band_maps_ds))
                _scale = (_ds, _ds)
                for el, bands in self.last_band_maps.items():
                    main = bands['main']

                    disp = main.copy().astype(np.float32)
                    v = disp[np.isfinite(disp)]
                    if v.size:
                        lo_p = np.percentile(v, 1.0)
                        hi_p = np.percentile(v, 99.0)
                        if hi_p > lo_p:
                            disp = np.clip((disp - lo_p) / (hi_p - lo_p), 0, 1)

                    name = f"{el} ROI"
                    has_mask = self.last_roi_mask is not None
                    if has_mask:
                        rgba = _rgba_float_from_gray_and_mask(disp, self.last_roi_mask, cmap_name='inferno')
                        if name in self.viewer.layers:
                            self.viewer.layers.remove(self.viewer.layers[name])
                        self.viewer.add_image(
                            rgba, name=name, rgb=True,
                            blending='translucent_no_depth', opacity=1.0,
                            scale=_scale, translate=(y0, x0), visible=True,
                        )
                    else:
                        if name in self.viewer.layers:
                            self.viewer.layers.remove(self.viewer.layers[name])
                        self.viewer.add_image(
                            disp, name=name, colormap='inferno',
                            blending='opaque', opacity=1.0,
                            scale=_scale, translate=(y0, x0), visible=True,
                        )

        # ── Restore phase segmentation ─────────────────────────────────────
        phase_meta = meta.get("phase_segmentation")
        if npz_path.exists():
            # Re-open if needed (data may have been closed above)
            data = np.load(npz_path)
            if "phase_map" in data:
                self.last_phase_map = data["phase_map"].astype(np.int16)
                if phase_meta:
                    names_raw = phase_meta.get("names", {})
                    self.last_phase_names = {int(k): v for k, v in names_raw.items()}
                    translate = phase_meta.get("translate")
                    if translate:
                        self.last_phase_translate = tuple(translate)
                    self.last_phase_ds = int(phase_meta.get("ds", 1) or 1)
                    minerals_raw = phase_meta.get("minerals", {})
                    self.last_phase_minerals = {int(k): v for k, v in minerals_raw.items()}
                    self.last_phase_fe3 = float(phase_meta.get("fe3_fraction", 0.0) or 0.0)
                    self.last_phase_comp_erosion = int(phase_meta.get("comp_erosion_px", 0) or 0)
                    unmix_raw = phase_meta.get("unmix", {}) or {}
                    self.last_phase_unmix = {int(c): {
                        "fractions": {int(e): float(f)
                                      for e, f in (d.get("fractions") or {}).items()},
                        "residual": float(d.get("residual", float("nan"))),
                    } for c, d in unmix_raw.items()}
                    comp_raw = phase_meta.get("composition", {})
                    self.last_phase_composition = {
                        int(cid): {el: float(v) for el, v in frac.items()}
                        for cid, frac in comp_raw.items()}
                    self.last_phase_source = str(phase_meta.get("source", "gmm"))
                    self.last_phase_seg_params = phase_meta.get("seg_params")
                    co = phase_meta.get("coating_elements")
                    if co is not None:
                        self._coating_override = frozenset(co)
                    # P-T inversion overrides (absent in older sessions → defaults).
                    inc = phase_meta.get("pt_include")
                    self.last_phase_pt_include = (
                        {int(c) for c in inc} if inc is not None else None)
                    pw = phase_meta.get("pt_weights")
                    self.last_phase_pt_weights = (
                        {int(k): float(v) for k, v in pw.items()} if pw else None)
                    self.last_phase_extra = list(phase_meta.get("pt_extra", []) or [])
                    self.last_phase_open_tol = float(phase_meta.get("pt_open_tol", 0.0) or 0.0)

                # Add phase map layer to napari (upsampled via scale if the phase
                # map was segmented at 1/ds to fit RAM).
                ty = self.last_phase_translate[0] if self.last_phase_translate else 0.0
                tx = self.last_phase_translate[1] if self.last_phase_translate else 0.0
                names = self.last_phase_names or {}
                _pds = float(max(1, int(self.last_phase_ds)))
                _add_phase_cluster_layers(
                    self.viewer, self.last_phase_map, names, (ty, tx),
                    scale=(_pds, _pds),
                )
            data.close()
        self._refresh_phase_comp_button()

        # ── Restore thermod runs ───────────────────────────────────────────
        runs_raw = meta.get("thermod_runs") or []
        self.last_thermod_runs = []
        for r in runs_raw:
            if isinstance(r, dict):
                self.last_thermod_runs.append(r)

        cs = meta.get("comp_scale")
        if isinstance(cs, dict) and cs.get("records"):
            self.last_comp_scale = {
                "records": cs.get("records") or [],
                "summary": {int(k): v for k, v in (cs.get("summary") or {}).items()},
                "opts": cs.get("opts") or {},
                "parent_label": cs.get("parent_label") or "ROI",
                "roi_bbox": cs.get("roi_bbox"),
            }

        n_maps = len(self.last_band_maps) if self.last_band_maps else 0
        has_phase = self.last_phase_map is not None
        n_runs = len(self.last_thermod_runs)
        print(
            f"[session] Loaded: {session_path.name}"
            f" ({n_maps} element maps"
            f"{', phase segmentation' if has_phase else ''}"
            f"{', ROI shapes restored' if roi_shapes_data else ''}"
            f"{', SPX composition' if self.last_spx_composition else ''}"
            f"{f', {n_runs} thermod run(s)' if n_runs else ''})",
            flush=True,
        )

    def on_clear(self):
        to_remove = [lyr for lyr in self.viewer.layers
                     if isinstance(lyr, napari.layers.Image)
                     and (lyr.name.endswith(' map') or lyr.name.endswith(' ROI')
                          or lyr.name.endswith(' preview')
                          or lyr.name.startswith(_PHASE_LAYER_PREFIX)
                          or lyr.name == "Phase segmentation")]
        for lyr in to_remove:
            self.viewer.layers.remove(lyr)

def _load_ebsd_overlay(viewer, directory, panel) -> None:
    """Load ``ebsd_phase_map.npy`` + ``ebsd_phases.json`` (Oxford h5oina) if present.

    Adds a hidden 'EBSD phases' labels layer (best-effort crystallographic colours)
    and stashes the full-mosaic map + legend on the panel so 'Use EBSD phases (ROI)'
    can use them as ready-made domains. No-op for non-h5oina datasets.
    """
    import json
    npy = Path(directory) / "ebsd_phase_map.npy"
    js = Path(directory) / "ebsd_phases.json"
    if not npy.exists():
        return
    try:
        phase_map = np.load(npy)
    except Exception as exc:
        print(f"[viewer] could not read {npy.name}: {exc}", flush=True)
        return
    legend: Dict[int, dict] = {}
    if js.exists():
        try:
            raw = json.loads(js.read_text(encoding="utf-8"))
            legend = {int(k): v for k, v in raw.items()}
        except Exception as exc:
            print(f"[viewer] could not read {js.name}: {exc}", flush=True)

    panel.ebsd_phase_map = phase_map
    panel.ebsd_phase_legend = legend
    # The EBSD-domains button now has a phase map; let the central refresh decide
    # its state (it also needs a computed ROI before it becomes usable).
    if hasattr(panel, "_refresh_control_availability"):
        panel._refresh_control_availability()

    # Optional coloured overlay. napari deprecated the add_labels(color=) kwarg,
    # so try it and fall back to auto colours on any failure.
    color = None
    try:
        color = {k: [c / 255.0 for c in v.get("color", [128, 128, 128])] + [1.0]
                 for k, v in legend.items() if k != 0}
        color[0] = [0.0, 0.0, 0.0, 0.0]
        color[None] = [0.0, 0.0, 0.0, 0.0]
    except Exception:
        color = None
    try:
        if color is not None:
            viewer.add_labels(phase_map.astype(np.int32), name="EBSD phases",
                              color=color, opacity=0.6, visible=False)
        else:
            raise RuntimeError("no colour map")
    except Exception:
        viewer.add_labels(phase_map.astype(np.int32), name="EBSD phases",
                          opacity=0.6, visible=False)

    named = ", ".join(f"{k}:{v.get('name')}" for k, v in sorted(legend.items())
                      if k != 0)
    print(f"[viewer] loaded EBSD phase map {phase_map.shape} [{named}] — toggle "
          f"the 'EBSD phases' layer; use 'Use EBSD phases (ROI) …' for analysis.",
          flush=True)


# -------------------------
# Main
# -------------------------
def _ensure_qapp():
    """Return a live, properly-configured QApplication.

    The folder-picker / message boxes below may run before ``napari.Viewer()``
    builds the app. Prefer napari's own app getter so high-DPI attributes etc.
    are set the way napari expects (and it's the same instance ``napari.run()``
    will use); fall back to a bare QApplication only if that's unavailable.
    """
    try:
        import napari.qt as _nq
        getter = getattr(_nq, "get_qapp", None) or getattr(_nq, "get_app", None)
        if getter is not None:
            return getter()
    except Exception:
        pass
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _prompt_for_directory(initial=None):
    """Open a folder picker and return the chosen Path (or None if cancelled).

    Used when the viewer is launched without ``--dir`` — the standard path for
    the double-click launcher used by non-expert installs. Built as an explicit
    modal dialog (rather than the static convenience method) and raised to the
    front: a freshly-created QApplication that has not yet run its event loop
    can otherwise silently fail to show the native static dialog on Windows.
    """
    from qtpy.QtWidgets import QFileDialog
    from qtpy.QtCore import Qt
    app = _ensure_qapp()
    app.processEvents()  # let the app settle so the dialog actually paints
    print("[viewer] No --dir given; opening a folder picker. Choose your "
          "dataset folder (it may open behind this window).", flush=True)
    dlg = QFileDialog(
        None,
        "Select dataset folder (stitched mosaic, or an Oxford AZtec / h5oina export)")
    dlg.setFileMode(QFileDialog.Directory)
    dlg.setOption(QFileDialog.ShowDirsOnly, True)
    if initial:
        dlg.setDirectory(str(initial))
    dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)
    dlg.raise_()
    dlg.activateWindow()
    if dlg.exec_():
        selected = dlg.selectedFiles()
        if selected:
            print(f"[viewer] Selected dataset folder: {selected[0]}", flush=True)
            return Path(selected[0])
    print("[viewer] No folder selected — exiting.", flush=True)
    return None


def _error_box(message, title="EquiMap"):
    """Show a friendly modal error instead of dying with a console traceback a
    double-click user would never see."""
    try:
        from qtpy.QtWidgets import QMessageBox
        _ensure_qapp()
        QMessageBox.critical(None, title, message)
    except Exception:
        # Headless / Qt unavailable — fall back to stderr so we still say why.
        print(f"[{title}] {message}", flush=True)


class _CommandRunnerDialog(QtWidgets.QDialog):
    """Run an external command (stitch.py / oxford_loader.py / eds_cache.py) in a
    subprocess via QProcess, streaming its output into a log pane.

    Used by the Tools menu so heavy operations (stitching, cache builds) run in
    isolation — they can't freeze or crash the viewer, and they're cancellable.
    On a clean (exit-code 0) finish, an optional ``on_success`` callback runs.
    """

    def __init__(self, title, argv, cwd, parent=None, on_success=None):
        super().__init__(parent)
        from qtpy.QtCore import QProcess
        self.setWindowTitle(title)
        self.resize(840, 480)
        self._argv = [str(a) for a in argv]
        self._on_success = on_success

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(title))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        try:
            from qtpy.QtGui import QFont
            self.log.setFont(QFont("Consolas", 9))
        except Exception:
            pass
        v.addWidget(self.log, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setEnabled(False)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_close)
        v.addLayout(row)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_close.clicked.connect(self.accept)

        self._append("$ " + " ".join((f'"{a}"' if " " in a else a) for a in self._argv))
        self._append(f"(working dir: {cwd})\n")

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(cwd))
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(
            lambda *_: self._append(f"[process error] {self.proc.errorString()}"))
        self.proc.start(self._argv[0], self._argv[1:])

    def _append(self, text):
        self.log.appendPlainText(str(text).rstrip("\n"))

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        if data.strip():
            self._append(data)

    def _cancel(self):
        from qtpy.QtCore import QProcess
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self._append("\n[cancelled by user]")

    def _finished(self, code, _status):
        self._read()
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        if code == 0:
            self._append("\n✓ Finished successfully.")
            if self._on_success:
                try:
                    self._on_success()
                except Exception as exc:
                    self._append(f"[post-step error] {exc}")
        else:
            self._append(f"\n✗ Exited with code {code}.")

    def closeEvent(self, event):
        from qtpy.QtCore import QProcess
        if self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.proc.waitForFinished(2000)
        super().closeEvent(event)


def _build_tools_menu(viewer, panel, directory):
    """Add a 'Tools' menu (stitch / build EDS cache / import Oxford / cache
    status) to the napari window. Best-effort: wrapped so a napari menubar API
    change can't block viewer startup — the viewer still works without it."""
    import sys
    try:
        win = viewer.window._qt_window
        menubar = win.menuBar()
    except Exception as exc:
        print(f"[viewer] Tools menu unavailable: {exc}", flush=True)
        return
    try:
        from qtpy.QtWidgets import QAction
    except ImportError:
        from qtpy.QtGui import QAction

    repo = Path(__file__).resolve().parent

    def _pick_dir(title, start):
        d = QtWidgets.QFileDialog.getExistingDirectory(win, title, str(start or ""))
        return Path(d) if d else None

    def _run(title, argv, on_success=None):
        _CommandRunnerDialog(title, argv, repo, parent=win, on_success=on_success).exec_()

    def _after(d):
        """Refresh the cache status if the op targeted the loaded dataset;
        otherwise tell the user to reopen EquiMap on the new folder."""
        try:
            same = Path(d).resolve() == Path(directory).resolve()
        except Exception:
            same = False
        if same:
            panel._refresh_control_availability()
        else:
            QtWidgets.QMessageBox.information(
                win, "Done",
                f"Finished for:\n{d}\n\nReopen EquiMap on that folder to view it.")

    def do_stitch():
        d = _pick_dir("Select the BCF tile folder to stitch", directory)
        if d is None:
            return
        build = QtWidgets.QMessageBox.question(
            win, "Stitch mosaic",
            "Also build the full EDS cache now (recommended — makes ROI reads "
            "fast)?\n\nThis adds time and disk but is the usual workflow.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        argv = [sys.executable, "stitch.py", "--dir", str(d)]
        if build == QtWidgets.QMessageBox.Yes:
            argv += ["--eds-preview", "all"]
        _run("Stitching mosaic", argv, on_success=lambda: _after(d))

    def do_build_cache():
        d = _pick_dir("Select the stitched dataset folder", directory)
        if d is None:
            return
        argv = [sys.executable, "stitch.py", "--dir", str(d),
                "--eds-preview", "all", "--eds-cache-on-restitch", "rebuild"]
        _run("Building / refreshing EDS cache", argv, on_success=lambda: _after(d))

    def do_oxford():
        d = _pick_dir("Select the Oxford export folder (CSVs or .h5oina)", directory)
        if d is None:
            return
        argv = [sys.executable, "oxford_loader.py", "--dir", str(d)]
        _run("Importing Oxford dataset", argv, on_success=lambda: _after(d))

    def do_cache_status():
        st = panel._eds_cache_status()
        if not st.get("exists"):
            QtWidgets.QMessageBox.information(
                win, "EDS cache status",
                "No EDS cache for the loaded dataset.\n\n"
                "Build one via Tools ▸ Build / refresh EDS cache.")
            return
        lines = [
            f"State: {'valid ✓' if st.get('valid') else 'STALE ⚠ (stitch manifest changed)'}",
            f"Elements: {', '.join(st.get('elements') or []) or '—'}",
            f"Window width: {st.get('width_eV')} eV",
            f"ds: {st.get('ds')}    size: {st.get('W')}×{st.get('H')}",
            f"Type: {st.get('data_type')}",
            f"Source: {st.get('source')}",
        ]
        box = QtWidgets.QMessageBox(win)
        box.setWindowTitle("EDS cache status")
        box.setText("\n".join(lines))
        defrag = box.addButton("Defragment…", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec_()
        if box.clickedButton() is defrag:
            try:
                import eds_cache as _edsc
                cd = _edsc.resolve_cache_dir(Path(panel.manifest).parent)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(win, "Defragment", f"Cannot locate cache: {exc}")
                return
            _run("Defragmenting EDS cache",
                 [sys.executable, "eds_cache.py", "defragment", str(cd)])

    menu = menubar.addMenu("Tools")
    for text, fn in [
        ("Stitch mosaic from BCF folder…", do_stitch),
        ("Build / refresh EDS cache…", do_build_cache),
        ("Import Oxford dataset…", do_oxford),
        (None, None),
        ("EDS cache status…", do_cache_status),
    ]:
        if text is None:
            menu.addSeparator()
            continue
        act = QAction(text, win)
        act.triggered.connect(lambda checked=False, f=fn: f())
        menu.addAction(act)


def _bg_scheme() -> str:
    """Process-wide net-count background scheme (eds_background)."""
    from eds_background import default_scheme
    return default_scheme()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, default=None,
                    help='Directory with stitched outputs and manifest. If omitted, '
                         'a folder-picker dialog opens (the normal path for the '
                         'double-click launcher).')
    ap.add_argument('--bse', type=str, default='stitched_BSE_8bit.png', help='BSE mosaic image filename')
    ap.add_argument('--manifest', type=str, default='stitch_manifest.csv', help='Tile manifest CSV filename')
    ap.add_argument('--elements', type=str, default='auto',
                    help="Preselect elements: 'auto' (from metadata), 'all', or comma list like 'Si,O,Fe'.")
    ap.add_argument('--preview-auto', action='store_true',
                    help='Auto-load preview layers from eds_preview_ds*/ (hidden, opaque).')
    ap.add_argument('--preview-ds', type=int, default=0, help='Which preview ds folder to load (0=auto smallest ds).')
    ap.add_argument('--no-pyramid', action='store_true',
                    help='Disable multiscale pyramid (not recommended for huge mosaics).')
    ap.add_argument('--debug-manifest', action='store_true', help='Print manifest/bbox debug in console.')
    ap.add_argument('--debug-spx', action='store_true', help='Print SPX overlap/matching debug in console.')
    ap.add_argument('--session', type=str, default=None, metavar='JSON',
                    help='Load a previously saved session (viewer_session.json) on startup.')
    args = ap.parse_args()

    if args.dir:
        directory = Path(args.dir)
    else:
        directory = _prompt_for_directory()
        if directory is None:
            # User cancelled the picker — exit quietly, no traceback.
            raise SystemExit(0)
    if not directory.is_dir():
        _error_box(f"That folder does not exist:\n\n{directory}")
        raise SystemExit(1)

    bse_path = directory / args.bse
    manifest_path = directory / args.manifest

    # If the stitched BSE is missing, this may be a raw (unconverted) Oxford
    # AZtec CSV export. Detect the per-element Wt% CSVs and run the one-shot
    # importer in place — it writes stitched_BSE_8bit.png + stitch_manifest.csv
    # + eds_cache/ that the rest of the viewer reads vendor-agnostically. Bruker
    # dirs have no such CSVs, so discover_inputs raises and we fall through to
    # the original "not found" error.
    if not bse_path.exists():
        import oxford_loader
        h5 = None
        try:
            h5 = oxford_loader.discover_h5oina(directory)
        except Exception:
            h5 = None
        if h5 is not None:
            print(f"[viewer] No stitched BSE in {directory}; detected an Oxford "
                  f"h5oina export ({h5.name}) — running oxford_loader. The first "
                  f"run builds the EDS cache + EBSD phase map and can take a few "
                  f"minutes...", flush=True)
            oxford_loader.load_h5oina_dataset(directory)
        else:
            is_oxford = False
            try:
                oxford_loader.discover_inputs(directory)  # raises if not Oxford
                is_oxford = True
            except Exception:
                is_oxford = False
            if is_oxford:
                print(f"[viewer] No stitched BSE in {directory}; detected an "
                      f"Oxford AZtec CSV export — running oxford_loader. The "
                      f"first run builds the EDS cache and can take a few "
                      f"minutes...", flush=True)
                oxford_loader.load_oxford_dataset(directory)

    if not bse_path.exists():
        _error_box(
            f"No dataset was found in:\n\n{directory}\n\n"
            f"Expected a stitched mosaic ({args.bse}) or an Oxford AZtec / "
            f"h5oina export. Please pick the folder that holds your dataset.")
        raise SystemExit(1)
    if not manifest_path.exists():
        _error_box(f"Dataset is incomplete — manifest not found:\n\n{manifest_path}")
        raise SystemExit(1)

    viewer = napari.Viewer()

    # BSE base layer (multiscale)
    bse = _imread_robust(str(bse_path))
    bse = _to_gray(bse)
    if args.no_pyramid:
        viewer.add_image(bse, name='BSE', colormap='gray', blending='opaque', visible=True)
    else:
        pyramid = _build_pyramid(bse)
        viewer.add_image(pyramid, name='BSE', multiscale=True, colormap='gray', blending='opaque', visible=True)

    # Previews
    preview_loaded: List[str] = []
    if args.preview_auto:
        preview_loaded, _ = _auto_load_previews(viewer, directory, requested_ds=args.preview_ds)

    # Metadata-driven element list
    rows = read_manifest(manifest_path)
    available, seen = [], set()
    for r in rows:
        for el in r.get("elements_list", []):
            if el not in seen and el in LINE_ENERGIES_KEV:
                seen.add(el); available.append(el)
        for ln in r.get("xray_lines_list", []):
            if "_" in ln:
                el = ln.split("_", 1)[0]
                if el not in seen and el in LINE_ENERGIES_KEV:
                    seen.add(el); available.append(el)
    for el in preview_loaded:
        if el not in seen and el in LINE_ENERGIES_KEV:
            seen.add(el); available.append(el)
    # Consult the EDS cache's own element list. The stitch manifest may not carry
    # element metadata (older runs leave the columns blank), and previews are only
    # loaded with --preview-auto. Without this, 'auto' falls back to a generic list
    # that isn't a subset of the cache, so on_compute's cache fast path (which
    # requires the requested elements ⊆ cached elements) is silently rejected —
    # e.g. the large-ROI downsample offer disappears and it asks to rebuild.
    try:
        import eds_cache as _edsc
        _cm = _edsc.load_manifest(_edsc.resolve_cache_dir(directory))
        if _cm:
            for el in _cm.get("elements", []):
                if el not in seen and el in LINE_ENERGIES_KEV:
                    seen.add(el); available.append(el)
    except Exception:
        pass
    if not available:
        fallback = ['C','O','Na','Mg','Al','Si','P','S','Cl','K','Ca','Ti','Cr','Mn','Fe','Ni','Cu','Zn']
        available = [e for e in fallback if e in LINE_ENERGIES_KEV]

    # Determine preselected elements from CLI
    arg_el = (args.elements or 'auto').strip().lower()
    if arg_el in ('all', 'auto'):
        preselected = list(available)
    else:
        want = [s.strip() for s in (args.elements or '').split(',') if s.strip()]
        preselected = [e for e in available if e in want]

    # ROI layer + control panel
    viewer.add_shapes(name='ROI', shape_type='rectangle', edge_color='yellow', face_color='transparent')
    panel = EDSControlPanel(viewer, manifest_path,
                            available_elements=available,
                            preselected=preselected,
                            bse_layer_name='BSE',
                            debug_manifest=args.debug_manifest,
                            debug_spx=args.debug_spx)
    viewer.window.add_dock_widget(panel, area='right')

    # EBSD phase map (Oxford h5oina) — optional overlay + ready-made domains.
    _load_ebsd_overlay(viewer, directory, panel)

    # Tools menu: stitch / build EDS cache / import Oxford / cache status.
    _build_tools_menu(viewer, panel, directory)

    # Restore session if --session was specified
    if args.session:
        session_file = Path(args.session)
        if not session_file.is_absolute():
            session_file = directory / session_file
        if session_file.exists():
            panel._load_session(session_file)
        else:
            print(f"[session] Session file not found: {session_file}")

    napari.run()

if __name__ == '__main__':
    main()
