# viewer_stitch.py
"""Stitch a folder of Bruker tiles from the viewer, without the command line.

When the folder the user opens holds .bcf tiles but no stitched mosaic,
``viewer.main`` offers to stitch it here:

- ``_StitchSettingsDialog``: the basic choices (EDS cache and where it goes,
  with its estimated size against the free disk space; element previews;
  coating element) and an Advanced section with the alignment preset,
  auto-tune, alignment-check-only, prep filter, weight floor and the cache and
  preview options. It always shows the exact ``stitch.py`` command the
  settings produce; stitch.py itself saves that command in
  ``stitch_debug/stitch_command.txt``.
- ``_StitchRunDialog``: runs that command as a subprocess (the same code path
  as the command line) with a live log, elapsed time and Cancel. Mosaic
  files written by a run that did not finish are removed, so a cancelled run
  can never be opened as a dataset; the EDS cache is safe because its
  manifest.json is written last.

The helpers at the top are Qt-free and unit-tested (tests/test_viewer_stitch.py).
"""
from __future__ import annotations

import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STITCH_SCRIPT = Path(__file__).resolve().parent / "stitch.py"
MOSAIC_FILES = ("stitched_BSE_8bit.png", "stitched_BSE_16bit.tiff")

# EDS cache bytes per mosaic pixel, measured on C23-S1-A (11 elements: 168 GB
# for 49,669 x 26,783 px). Each element stores three band sums; scale by the
# element count when it is known.
CACHE_BYTES_PER_PX_11_EL = 126.0
# Fraction of a tile that is new mosaic area (about 9 % overlap on each axis).
NEW_AREA_PER_TILE = 0.83

# The presets, their values and descriptions: the same module stitch.py reads.
from stitch_presets import (PRESET_CHOICES, PRESET_HELP, PRESET_TABLE_KEYS,   # noqa: E402
                            resolve_preset)
PREP_VARIANTS = ("auto", "edge", "wide_hp", "flat", "dog")


def preset_table_rows(name: str):
    """(label, value of ``name``, value of "default", differs) for the dialog table."""
    this, ref = resolve_preset(name), resolve_preset("default")
    rows = []
    for key, label in PRESET_TABLE_KEYS:
        a, b = this[key], ref[key]
        rows.append((label, f"{a:g}", f"{b:g}", float(a) != float(b)))
    return rows


# ---------------------------------------------------------------- Qt-free helpers

def find_bcf_tiles(directory: Path) -> List[Path]:
    """All .bcf files below ``directory`` (any case), as stitch.py finds them."""
    return [p for p in Path(directory).rglob("*")
            if p.is_file() and p.suffix.lower() == ".bcf"]


def needs_stitching(directory: Path) -> bool:
    """True when the folder holds Bruker tiles but no stitched mosaic."""
    directory = Path(directory)
    return (not (directory / MOSAIC_FILES[0]).exists()) and bool(find_bcf_tiles(directory))


def estimate_cache_bytes(n_tiles: int, tile_px: int = 1024 * 768,
                         n_elements: Optional[int] = None, ds: int = 1) -> float:
    """Rough size of the full EDS cache for ``n_tiles`` tiles (bytes)."""
    per_px = CACHE_BYTES_PER_PX_11_EL
    if n_elements:
        per_px *= (3 * n_elements + 1) / 34.0
    mosaic_px = n_tiles * tile_px * NEW_AREA_PER_TILE
    return mosaic_px * per_px / float(max(1, ds) ** 2)


def format_bytes(n: float) -> str:
    for unit, f in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if n >= f:
            return f"{n / f:.1f} {unit}"
    return f"{n / 1e3:.0f} kB"


def free_bytes(path: Path) -> Optional[int]:
    """Free space on the drive of ``path`` (its nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def build_stitch_args(directory: Path, *, cache: bool = True,
                      cache_dir: Optional[Path] = None, previews: bool = True,
                      coating: str = "C", preset: str = "default",
                      auto_tune: bool = False, diagnose: bool = False,
                      prep_variant: str = "auto", weight_floor: float = 0.0,
                      cache_ds: int = 1, shift_method: str = "bilinear",
                      cache_workers: int = 4, flush_every: int = 50,
                      preview_ds: int = 8, preview_width: int = 150) -> List[str]:
    """stitch.py arguments for these settings (defaults left out, so the
    command reads like one typed by hand). Cache and previews come from the
    same --eds-preview pass; a stale cache from an earlier run is rebuilt, since
    a subprocess cannot answer stitch.py's prompt."""
    a = ["--dir", str(directory)]
    if preset != "default":
        a += ["--preset", preset]
    if diagnose:
        a += ["--diagnose"]
    if auto_tune:
        a += ["--preflight", "--auto"]
    if prep_variant != "auto":
        a += ["--prep-variant", prep_variant]
    if weight_floor > 0:
        a += ["--weight-floor", f"{weight_floor:g}"]
    if (cache or previews) and not diagnose:
        a += ["--eds-preview", "all"]
        if preview_ds != 8:
            a += ["--eds-preview-ds", str(preview_ds)]
        if preview_width != 150:
            a += ["--eds-preview-width", str(preview_width)]
        if not cache:
            a += ["--no-eds-cache"]
        else:
            if cache_ds != 1:
                a += ["--eds-cache-ds", str(cache_ds)]
            if cache_dir is not None:
                a += ["--eds-cache-dir", str(cache_dir)]
            if shift_method != "bilinear":
                a += ["--eds-shift-method", shift_method]
            if cache_workers != 4:
                a += ["--eds-cache-workers", str(cache_workers)]
            if flush_every != 50:
                a += ["--eds-cache-flush-every", str(flush_every)]
            a += ["--eds-cache-on-restitch", "rebuild"]
        coat = [c for c in re.split(r"[,\s]+", coating.strip()) if c]
        if coat != ["C"]:
            a += ["--coating-elements"] + coat
    return a


def command_line(args: List[str]) -> str:
    """The command a user would type for these arguments."""
    return " ".join(shlex.quote(x) for x in ["python", "stitch.py"] + list(args))


def remove_unfinished_mosaic(directory: Path, started: float) -> List[str]:
    """Delete mosaic files a run started at ``started`` (time.time()) wrote.

    A run that did not finish may have been stopped while writing them; older
    files (a mosaic from before this run) are left alone.
    """
    removed = []
    for name in MOSAIC_FILES:
        p = Path(directory) / name
        try:
            if p.exists() and p.stat().st_mtime >= started - 1.0:
                p.unlink()
                removed.append(name)
        except OSError:
            pass
    return removed


def probe_tiles(tiles: List[Path]) -> Tuple[Optional[int], Optional[int]]:
    """(pixels per tile, number of mapped elements) from the first tile, or Nones.

    Reads only metadata (lazy HyperSpy load); any failure returns Nones and
    the caller falls back to 1024 x 768 px and 11 elements.
    """
    if not tiles:
        return None, None
    try:
        import warnings
        import hyperspy.api as hs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sigs = hs.load(str(tiles[0]), lazy=True)
        sigs = sigs if isinstance(sigs, list) else [sigs]
        px, n_el = None, None
        for s in sigs:
            if s.axes_manager.signal_dimension == 2 and px is None:   # the BSE image
                w, h = s.axes_manager.signal_shape
                px = int(w) * int(h)
            sample = s.metadata.as_dictionary().get("Sample", {}) or {}
            lines = sample.get("xray_lines") or []
            els = sample.get("elements") or []
            if lines or els:
                n_el = len(set(els) | {ln.split("_")[0] for ln in lines})
        return px, n_el
    except Exception:
        return None, None


# ---------------------------------------------------------------- Qt dialogs

from qtpy import QtCore, QtGui, QtWidgets   # noqa: E402


class _StitchSettingsDialog(QtWidgets.QDialog):
    """Basic + advanced stitch settings for one folder of Bruker tiles."""

    def __init__(self, directory: Path, tiles: List[Path], parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self.setWindowTitle("Stitch Bruker tiles")
        self.setMinimumWidth(640)
        self._tile_px, self._n_el = probe_tiles(tiles)
        self._n_tiles = len(tiles)
        lay = QtWidgets.QVBoxLayout(self)

        jobs = sorted({t.parent for t in tiles})
        where = (f"<b>{self.directory}</b> holds <b>{len(tiles)} Bruker tiles</b>"
                 f" in {len(jobs)} folder{'s' if len(jobs) != 1 else ''}")
        if (self.directory / MOSAIC_FILES[0]).exists():
            text = (f"{where} and already a stitched mosaic. Stitching again replaces "
                    f"the mosaic and its manifest; an EDS cache built for the old "
                    f"positions is rebuilt. A session saved on the old mosaic may no "
                    f"longer line up.")
        else:
            text = (f"{where} but no stitched mosaic. EquiMap can stitch them now: it "
                    f"aligns the tiles, writes the mosaic, and (optionally) builds the "
                    f"EDS cache and element previews. The viewer opens when it is done.")
        intro = QtWidgets.QLabel(text)
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # ---- basic ----
        basic = QtWidgets.QGroupBox("Settings")
        bl = QtWidgets.QGridLayout(basic)
        self.cb_cache = QtWidgets.QCheckBox("Build the EDS cache (fast element maps of any region)")
        self.cb_cache.setChecked(True)
        bl.addWidget(self.cb_cache, 0, 0, 1, 3)
        self.cache_dir_edit = QtWidgets.QLineEdit()
        self.cache_dir_edit.setPlaceholderText(f"{self.directory / 'eds_cache'}  (default)")
        btn_dir = QtWidgets.QPushButton("Store elsewhere …")
        btn_dir.clicked.connect(self._choose_cache_dir)
        bl.addWidget(QtWidgets.QLabel("Cache folder:"), 1, 0)
        bl.addWidget(self.cache_dir_edit, 1, 1)
        bl.addWidget(btn_dir, 1, 2)
        self.size_label = QtWidgets.QLabel()
        self.size_label.setWordWrap(True)
        bl.addWidget(self.size_label, 2, 0, 1, 3)
        self.cb_previews = QtWidgets.QCheckBox("Make element preview maps (shown when the viewer opens)")
        self.cb_previews.setChecked(True)
        bl.addWidget(self.cb_previews, 3, 0, 1, 3)
        bl.addWidget(QtWidgets.QLabel("Coating element(s) to ignore:"), 4, 0)
        self.coating_edit = QtWidgets.QLineEdit("C")
        self.coating_edit.setToolTip("The conductive coat, ignored in calibration and mineral "
                                     "identification: C for carbon, e.g. Pt or Au otherwise; "
                                     "empty for an uncoated sample.")
        bl.addWidget(self.coating_edit, 4, 1)
        lay.addWidget(basic)

        # ---- advanced (collapsed) ----
        self.btn_adv = QtWidgets.QToolButton()
        self.btn_adv.setText("Advanced settings")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_adv.setArrowType(QtCore.Qt.RightArrow)
        lay.addWidget(self.btn_adv)
        self.adv = QtWidgets.QWidget()
        al = QtWidgets.QFormLayout(self.adv)
        self.preset_combo = QtWidgets.QComboBox()
        for p in PRESET_CHOICES:
            self.preset_combo.addItem(p, p)
        self.preset_help = QtWidgets.QLabel()
        self.preset_help.setWordWrap(True)
        al.addRow("Alignment preset:", self.preset_combo)
        al.addRow("", self.preset_help)
        self.preset_table = QtWidgets.QTableWidget(len(PRESET_TABLE_KEYS), 3)
        self.preset_table.setHorizontalHeaderLabels(["Setting", "This preset", "default"])
        self.preset_table.verticalHeader().setVisible(False)
        self.preset_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.preset_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        hh = self.preset_table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.preset_table.setMinimumHeight(
            self.preset_table.verticalHeader().defaultSectionSize() * (len(PRESET_TABLE_KEYS) + 1) + 6)
        self.preset_table.setToolTip("The alignment values this preset sets (from "
                                     "stitch_presets.py); bold where it differs from default.")
        al.addRow("", self.preset_table)
        self.cb_auto = QtWidgets.QCheckBox("Auto-tune the alignment parameters first (slower)")
        al.addRow("", self.cb_auto)
        self.cb_diag = QtWidgets.QCheckBox(
            "Check the alignment only (fast; writes a report in stitch_debug/, no mosaic)")
        al.addRow("", self.cb_diag)
        self.prep_combo = QtWidgets.QComboBox()
        for p in PREP_VARIANTS:
            self.prep_combo.addItem(p, p)
        self.prep_combo.setToolTip("Filter applied before phase correlation; 'auto' calibrates "
                                   "it on a sample of tile pairs.")
        al.addRow("Correlation filter:", self.prep_combo)
        self.wfloor = QtWidgets.QDoubleSpinBox()
        self.wfloor.setRange(0.0, 1.0); self.wfloor.setSingleStep(0.01); self.wfloor.setDecimals(2)
        self.wfloor.setToolTip("Drop tile pairs whose correlation weight is below this from the "
                               "position solve (0 = keep all).")
        al.addRow("Weight floor:", self.wfloor)
        self.cache_ds = QtWidgets.QComboBox()
        for ds, txt in ((1, "full resolution"), (2, "1/2 (a quarter of the size)"),
                        (4, "1/4 (a sixteenth of the size)")):
            self.cache_ds.addItem(txt, ds)
        al.addRow("Cache resolution:", self.cache_ds)
        self.shift_combo = QtWidgets.QComboBox()
        self.shift_combo.addItem("bilinear", "bilinear"); self.shift_combo.addItem("fft", "fft")
        self.shift_combo.setToolTip("Sub-pixel shift of each tile's EDS data; bilinear is about "
                                    "five times faster than fft.")
        al.addRow("Sub-pixel shift:", self.shift_combo)
        self.workers = QtWidgets.QSpinBox(); self.workers.setRange(1, 16); self.workers.setValue(4)
        self.workers.setToolTip("Tiles processed in parallel while building the cache; each "
                                "holds about 1 GB in memory. Lower it on a machine with little RAM.")
        al.addRow("Cache workers:", self.workers)
        self.flush = QtWidgets.QSpinBox(); self.flush.setRange(1, 1000); self.flush.setValue(50)
        self.flush.setToolTip("Write the cache to disk every N tiles; lower values use less memory.")
        al.addRow("Flush every (tiles):", self.flush)
        self.prev_ds = QtWidgets.QSpinBox(); self.prev_ds.setRange(1, 64); self.prev_ds.setValue(8)
        al.addRow("Preview downsampling:", self.prev_ds)
        self.prev_w = QtWidgets.QSpinBox(); self.prev_w.setRange(20, 1000); self.prev_w.setValue(150)
        self.prev_w.setSuffix(" eV")
        al.addRow("Element window width:", self.prev_w)
        note = QtWidgets.QLabel("Other stitch.py options are available on the command line "
                                "(python stitch.py --help).")
        note.setWordWrap(True)
        al.addRow(note)
        self.adv.setVisible(False)
        lay.addWidget(self.adv)

        # ---- command ----
        lay.addWidget(QtWidgets.QLabel("Command (also saved in stitch_debug/stitch_command.txt):"))
        self.cmd = QtWidgets.QPlainTextEdit()
        self.cmd.setReadOnly(True)
        self.cmd.setMaximumHeight(64)
        self.cmd.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self.cmd)

        bb = QtWidgets.QDialogButtonBox()
        self.btn_ok = bb.addButton("Stitch", QtWidgets.QDialogButtonBox.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        for w in (self.cb_cache, self.cb_previews, self.cb_auto, self.cb_diag):
            w.toggled.connect(self._refresh)
        for w in (self.cache_dir_edit, self.coating_edit):
            w.textChanged.connect(self._refresh)
        for w in (self.preset_combo, self.prep_combo, self.cache_ds, self.shift_combo):
            w.currentIndexChanged.connect(self._refresh)
        for w in (self.wfloor, self.workers, self.flush, self.prev_ds, self.prev_w):
            w.valueChanged.connect(self._refresh)
        self.btn_adv.toggled.connect(self._toggle_adv)
        self._refresh()

    def _toggle_adv(self, on):
        self.adv.setVisible(on)
        self.btn_adv.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
        self.adjustSize()

    def _choose_cache_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder for the EDS cache", str(self.directory))
        if d:
            self.cache_dir_edit.setText(d)

    def _cache_dir(self) -> Optional[Path]:
        t = self.cache_dir_edit.text().strip()
        return Path(t) if t else None

    def stitch_args(self) -> List[str]:
        return build_stitch_args(
            self.directory, cache=self.cb_cache.isChecked(), cache_dir=self._cache_dir(),
            previews=self.cb_previews.isChecked(), coating=self.coating_edit.text(),
            preset=self.preset_combo.currentData(), auto_tune=self.cb_auto.isChecked(),
            diagnose=self.cb_diag.isChecked(), prep_variant=self.prep_combo.currentData(),
            weight_floor=self.wfloor.value(), cache_ds=self.cache_ds.currentData(),
            shift_method=self.shift_combo.currentData(), cache_workers=self.workers.value(),
            flush_every=self.flush.value(), preview_ds=self.prev_ds.value(),
            preview_width=self.prev_w.value())

    def diagnose_only(self) -> bool:
        return self.cb_diag.isChecked()

    def _fill_preset_table(self, name: str):
        bold = QtGui.QFont()
        bold.setBold(True)
        for i, (label, val, ref, differs) in enumerate(preset_table_rows(name)):
            for j, text in enumerate((label, val, ref)):
                item = QtWidgets.QTableWidgetItem(text)
                if j > 0:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if differs and j < 2:
                    item.setFont(bold)
                self.preset_table.setItem(i, j, item)

    def _refresh(self, *_):
        name = self.preset_combo.currentData()
        self.preset_help.setText(PRESET_HELP.get(name, ""))
        self._fill_preset_table(name)
        diag = self.cb_diag.isChecked()
        for w in (self.cb_cache, self.cb_previews, self.cache_dir_edit, self.coating_edit):
            w.setEnabled(not diag)
        cache_on = self.cb_cache.isChecked() and not diag
        self.cache_dir_edit.setEnabled(cache_on)
        est = estimate_cache_bytes(self._n_tiles, self._tile_px or 1024 * 768,
                                   self._n_el, int(self.cache_ds.currentData() or 1))
        target = self._cache_dir() or self.directory
        free = free_bytes(target)
        warn = False
        if not cache_on:
            txt = "No cache: element maps are then read from the tiles themselves (slow for large regions)."
        else:
            basis = (f"{self._n_el} elements" if self._n_el else "11 elements assumed")
            txt = f"Estimated cache size: <b>{format_bytes(est)}</b> ({basis})"
            if free is not None:
                txt += f"; free on that drive: {format_bytes(free)}"
                if est > 0.9 * free:
                    warn = True
                    txt += ("<br><span style='color:#c0392b'><b>Not enough space.</b> Store the "
                            "cache on another drive, lower its resolution (Advanced), or "
                            "build it without the cache.</span>")
        self.size_label.setText(txt)
        self.btn_ok.setText("Check alignment" if diag else "Stitch")
        self.btn_ok.setEnabled(not warn)
        self.cmd.setPlainText(command_line(self.stitch_args()))


class _StitchRunDialog(QtWidgets.QDialog):
    """Run stitch.py with a live log; ``succeeded`` tells the caller the outcome."""

    def __init__(self, directory: Path, args: List[str], parent=None):
        super().__init__(parent)
        self.directory = Path(directory)
        self.args = list(args)
        self.succeeded = False
        self.cache_failed = False
        self._started = 0.0
        self._cancelled = False
        self.setWindowTitle("Stitching …")
        self.resize(820, 480)
        lay = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("Starting …")
        lay.addWidget(self.status)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self.log)
        self.progress = QtWidgets.QLabel("")       # the live state of a progress bar
        self.progress.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        lay.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        self.elapsed = QtWidgets.QLabel("0:00")
        row.addWidget(self.elapsed)
        row.addStretch(1)
        self.btn = QtWidgets.QPushButton("Cancel")
        self.btn.clicked.connect(self._cancel_or_close)
        row.addWidget(self.btn)
        lay.addLayout(row)

        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.setWorkingDirectory(str(STITCH_SCRIPT.parent))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._error)
        self._buf = ""
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)

    def start(self):
        self._started = time.time()
        self.log.appendPlainText(command_line(self.args) + "\n")
        self.proc.start(sys.executable, [str(STITCH_SCRIPT)] + self.args)
        self.timer.start(1000)
        self.status.setText("Stitching: aligning the tiles, then the EDS cache and the mosaic.")

    def _tick(self):
        s = int(time.time() - self._started)
        self.elapsed.setText(f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600
                             else f"{s // 60}:{s % 60:02d}")

    def feed(self, data: str):
        """Split subprocess output into log lines.

        A progress bar (tqdm) redraws itself with \\r: its live state goes to
        the line under the log and only its final state (ended by \\n) is
        appended, so the log is append-only.
        """
        # Windows text-mode stdout ends lines with \r\n: a plain line end, not a
        # redraw. A \r at the very end of a chunk may be the first half of one,
        # so it waits for the next chunk.
        self._buf = (self._buf + data).replace("\r\n", "\n")
        while True:
            m = re.search(r"[\r\n]", self._buf)
            if not m or (m.group() == "\r" and m.end() == len(self._buf)):
                break
            line, sep, self._buf = self._buf[:m.start()], m.group(), self._buf[m.end():]
            line = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip()
            if not line.strip() or "signal_type='EDS_SEM' not understood" in line:
                continue
            if sep == "\r":
                self.progress.setText(line.strip()[:160])
                continue
            self.log.appendPlainText(line)
            if "[eds-cache] Skipped" in line:          # stitch.py carries on without it
                self.cache_failed = True
            if re.match(r"\s*(EDS cache|Compositing|\[composite\]|\[eds-|Found \d+)", line):
                self.status.setText(line.strip()[:120])
        if self._buf.strip():                       # a bar mid-redraw, no terminator yet
            self.progress.setText(re.sub(r"\x1b\[[0-9;]*m", "", self._buf).strip()[:160])

    def _read(self):
        self.feed(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace"))

    def _error(self, err):
        if err == QtCore.QProcess.FailedToStart:
            self.log.appendPlainText(f"\nCould not start {sys.executable} {STITCH_SCRIPT}.")

    def _finished(self, code, _status):
        self.timer.stop()
        self._tick()
        self._read()
        if self._buf.strip():
            self.log.appendPlainText(self._buf.strip()); self._buf = ""
        self.progress.setText("")
        diagnose = "--diagnose" in self.args
        ok = (code == 0 and not self._cancelled
              and (diagnose or (self.directory / MOSAIC_FILES[0]).exists()))
        if not ok:
            removed = remove_unfinished_mosaic(self.directory, self._started)
            if removed:
                self.log.appendPlainText(f"\nRemoved unfinished {', '.join(removed)}.")
        self.succeeded = ok
        if self._cancelled:
            self.status.setText("Cancelled. Nothing was opened; start again to stitch.")
        elif ok and diagnose:
            self.status.setText("Alignment check done: see stitch_debug/stitch_diagnose_report.txt.")
        elif ok and self.cache_failed:
            self.status.setText("The mosaic is done, but building the EDS cache failed (see "
                                "'[eds-cache] Skipped' in the log): element maps will be read "
                                "from the tiles, which is slow. Tools > Build / refresh EDS "
                                "cache tries again.")
        elif ok:
            self.status.setText("Done. The viewer opens when you close this window.")
        else:
            self.status.setText(f"Stitching failed (exit code {code}). The log above says where; "
                                f"'python stitch.py --dir … --diagnose' checks the alignment.")
        self.btn.setText("Open the viewer" if ok and not diagnose else "Close")

    def _cancel_or_close(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            if QtWidgets.QMessageBox.question(
                    self, "Cancel stitching",
                    "Stop stitching? The tile alignment cache is kept, so a new run "
                    "starts faster; an unfinished mosaic is removed.") != QtWidgets.QMessageBox.Yes:
                return
            self._cancelled = True
            self.proc.kill()
            return
        self.accept() if self.succeeded else self.reject()

    def closeEvent(self, ev):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            ev.ignore()
            self._cancel_or_close()
        else:
            super().closeEvent(ev)


def offer_stitching(directory: Path, parent=None) -> bool:
    """Ask, configure and run stitching for a tile folder. True = mosaic ready."""
    tiles = find_bcf_tiles(directory)
    dlg = _StitchSettingsDialog(directory, tiles, parent)
    while True:
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return False
        run = _StitchRunDialog(directory, dlg.stitch_args(), parent)
        run.start()
        run.exec_()
        if dlg.diagnose_only():
            continue                      # back to the settings after a check
        if run.succeeded:
            return True
        if QtWidgets.QMessageBox.question(
                parent, "Stitching", "Change the settings and try again?") != QtWidgets.QMessageBox.Yes:
            return False
