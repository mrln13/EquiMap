# oxford_loader.py
"""
Convert an Oxford Instruments AZtec EDS export (BSE TIF + per-element Wt% CSVs)
into a THERMOD-ready mosaic directory.

The Oxford data is already montaged — there is no per-tile structure to stitch.
This loader synthesises the same artefacts ``stitch.py`` would produce for a
Bruker BCF dataset (``stitched_BSE_*.{png,tiff}``, ``stitch_manifest.csv``,
``eds_cache/``), so ``viewer.py`` runs unmodified against an Oxford dataset.

Expected input layout (defaults):

    <src_dir>/
        BSE image.tif                          (or any *BSE*.tif)
        Montaged Map Data-{Element} Wt%.csv    (one per element)

Output layout (same as a Bruker stitch directory):

    <out_dir>/
        stitched_BSE_8bit.png
        stitched_BSE_16bit.tiff
        stitch_manifest.csv
        eds_cache/
            manifest.json                      # data_type = "oxford_wt_pct"
            weight.npy                         # ones
            {El}__main.npy                     # Oxford Wt% map
            {El}__sb_lo.npy                    # zeros
            {El}__sb_hi.npy                    # zeros

Usage:

    python oxford_loader.py --dir D:\\UU\\Kibo
    python oxford_loader.py --dir D:\\UU\\Kibo --resolution bse
    python oxford_loader.py --dir D:\\UU\\Kibo --force
    python oxford_loader.py --dir D:\\UU\\Kibo --pixel-um 4.0
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    import imageio
except ImportError:  # pragma: no cover
    imageio = None  # type: ignore

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None  # type: ignore

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore

import eds_cache as _edsc

# Match "Montaged Map Data-{Element} Wt%.csv" (with or without an underscore
# delimiter and with several possible spellings of the suffix).
_ELEMENT_CSV_RE = re.compile(
    r"Montaged\s+Map\s+Data[-_]\s*(?P<el>[A-Za-z][A-Za-z0-9]?)\s*[Ww]t\s*%?",
    re.IGNORECASE,
)

# Match short Oxford export pattern "{El}_{digits}.csv" (e.g. "Si_301582592023.csv").
# Element symbol is case-sensitive: capital + optional lowercase.
_ELEMENT_CSV_RE_SHORT = re.compile(r"^(?P<el>[A-Z][a-z]?)_\d+$")

# Match the full-path export name AZtec writes when maps are exported from the
# data tree: "<project>__<specimen>__<site>__...__EDS Data__Si Wt%.csv". The
# element is the last token before "Wt%" at the end of the stem.
_ELEMENT_CSV_RE_TAIL = re.compile(r"(?:^|__|[-_\s])(?P<el>[A-Z][a-z]?)\s+[Ww]t\s*%$")

# AZtec writes CSVs with the Windows locale's decimal mark. Under a
# decimal-comma locale the comma is then BOTH the field delimiter and the
# decimal mark ("17,5341,0,0,23,97721,"), so rows are ragged and a plain CSV
# read is wrong. A weight-percent value is either "0" or INT,DEC (trailing
# zeros of DEC trimmed), which makes the text repairable:
#   pass 1 joins INT,DEC where DEC can only be a decimal part (leading zero,
#          four or more digits, or a three-digit number above 100);
#   pass 2 resolves the remaining runs of small non-zero tokens (decimal parts
#          shortened to 1-3 digits) by parity, an odd run taking a preceding
#          "0" as its integer part.
_DECCOMMA_P1 = re.compile(
    r"(?m)(?:(?<=,)|^)(0|[1-9]\d?|100),(0\d+|\d{4,}|[2-9]\d\d|1\d[1-9]|1[1-9]\d)(?=,|$)")
_DECCOMMA_P2 = re.compile(
    r"(?m)(?:(?<=,)|^)(0,)?((?:[1-9]\d{0,2},)*[1-9]\d{0,2})(?=,|$)")
_DECCOMMA_LEADING_ZERO = re.compile(r"(?m)(?:(?<=,)|^)0\d")

# Atomic numbers for synthesising a BSE proxy from element Wt% maps when no
# real BSE TIF is shipped with the dataset.
_Z_BY_ELEMENT = {
    "H": 1, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
    "Te": 52, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69,
    "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76,
    "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
    "Th": 90, "U": 92,
}

# Minimal element → primary X-ray line table for the manifest's xray_lines
# column. Used purely as a hint to viewer.py — the cache itself is
# pre-integrated so no real energy axis is consulted.
_DEFAULT_LINE = {
    "H": "K_a", "Li": "K_a", "Be": "K_a", "B": "K_a", "C": "K_a", "N": "K_a",
    "O": "K_a", "F": "K_a", "Na": "K_a", "Mg": "K_a", "Al": "K_a", "Si": "K_a",
    "P": "K_a", "S": "K_a", "Cl": "K_a", "K": "K_a", "Ca": "K_a", "Sc": "K_a",
    "Ti": "K_a", "V": "K_a", "Cr": "K_a", "Mn": "K_a", "Fe": "K_a", "Co": "K_a",
    "Ni": "K_a", "Cu": "K_a", "Zn": "K_a", "Ga": "K_a", "Ge": "K_a", "As": "K_a",
    "Se": "K_a", "Br": "K_a", "Rb": "K_a", "Sr": "K_a", "Y": "K_a", "Zr": "K_a",
    "Nb": "L_a", "Mo": "L_a", "Ag": "L_a", "Cd": "L_a", "In": "L_a", "Sn": "L_a",
    "Sb": "L_a", "Te": "L_a", "Ba": "L_a", "La": "L_a", "Ce": "L_a", "Pr": "L_a",
    "Nd": "L_a", "Sm": "L_a", "Eu": "L_a", "Gd": "L_a", "Tb": "L_a", "Dy": "L_a",
    "Ho": "L_a", "Er": "L_a", "Tm": "L_a", "Yb": "L_a", "Lu": "L_a", "Hf": "L_a",
    "Ta": "L_a", "W": "L_a", "Re": "L_a", "Os": "L_a", "Ir": "L_a", "Pt": "L_a",
    "Au": "M_a", "Hg": "M_a", "Tl": "M_a", "Pb": "M_a", "Bi": "M_a", "Th": "M_a",
    "U": "M_a",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_inputs(src_dir: Path,
                     bse_tif_glob: str = "*BSE*.tif",
                     element_csv_glob: str = "Montaged Map Data-* Wt%.csv",
                     ) -> Tuple[Optional[Path], Dict[str, Path]]:
    """Locate the BSE TIF and the per-element CSVs.

    Returns ``(bse_tif_path_or_None, {element_name: csv_path})``. The BSE
    TIF is optional — when missing, the loader synthesises a mean-Z proxy
    from the element maps. Element CSVs are required.

    Three filename conventions are recognised:
    1. ``Montaged Map Data-{El} Wt%.csv`` (Oxford AZtec montage exports)
    2. ``{El}_{digits}.csv``             (Oxford AZtec short-name exports)
    3. ``...__EDS Data__{El} Wt%.csv``   (AZtec data-tree exports, full path
       in the name)

    Raises ``ValueError`` if no element CSVs match any pattern.
    """
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    bse_candidates = sorted(src_dir.glob(bse_tif_glob))
    bse_tif = bse_candidates[0] if bse_candidates else None

    def _normalise_el(el: str) -> str:
        return el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()

    # Try the long pattern first (the original Kibo-style export).
    element_csvs: Dict[str, Path] = {}
    for csv_path in sorted(src_dir.glob(element_csv_glob)):
        m = _ELEMENT_CSV_RE.search(csv_path.name)
        if m:
            element_csvs[_normalise_el(m.group("el"))] = csv_path

    # Fall back to the short pattern if nothing matched.
    if not element_csvs:
        for csv_path in sorted(src_dir.glob("*.csv")):
            m = _ELEMENT_CSV_RE_SHORT.match(csv_path.stem)
            if m:
                element_csvs[_normalise_el(m.group("el"))] = csv_path

    # Last: the data-tree export name, element just before a trailing "Wt%".
    if not element_csvs:
        for csv_path in sorted(src_dir.glob("*.csv")):
            m = _ELEMENT_CSV_RE_TAIL.search(csv_path.stem)
            if m:
                element_csvs[_normalise_el(m.group("el"))] = csv_path

    if not element_csvs:
        raise ValueError(
            f"No element Wt% CSVs found in {src_dir}. Tried "
            f"{element_csv_glob!r}, the short '{{El}}_{{digits}}.csv' pattern and "
            f"the '...__{{El}} Wt%.csv' data-tree pattern."
        )

    return bse_tif, element_csvs


def discover_h5oina(src: Path) -> Optional[Path]:
    """Locate the Oxford ``.h5oina`` HDF5 export for *src*.

    *src* may be the ``.h5oina`` file itself, the project directory (the file is
    searched for directly and in an ``h5oina/`` subfolder), or any parent. The
    largest match is returned when several exist (the montaged map is the big
    one). Returns ``None`` when nothing matches.
    """
    src = Path(src)
    if src.is_file() and src.suffix.lower() == ".h5oina":
        return src
    if not src.is_dir():
        return None
    candidates: List[Path] = []
    candidates += sorted(src.glob("*.h5oina"))
    candidates += sorted(src.glob("h5oina/*.h5oina"))
    if not candidates:
        # last resort: one level deeper (e.g. an Area subfolder)
        candidates += sorted(src.glob("*/*.h5oina"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# ---------------------------------------------------------------------------
# .piprj project metadata (pixel size, stage origin)
# ---------------------------------------------------------------------------
# A ``.piprj`` is a ZIP archive of JSON sidecars written by the
# phase-identification tool that exports these Wt% maps (``projectType`` =
# "PhaseIdentifier"). ``ProjectUnits.json`` carries the spatial calibration:
#
#     {"Size": 4, "Name": 3, "StartX": 0, "StartY": 0, ...}
#
# ``Size`` is the pixel pitch (verified: Kibo "…4um…" → Size 4; Granulite → 3);
# ``Name`` is a unit code (observed value 3 == micron); ``StartX``/``StartY``
# are the stage origin. There is NO beam-voltage / kV field in this format.
_PIPRJ_KNOWN_UM_UNIT_CODES = {3}  # observed: 3 == micron (µm)


def discover_piprj(src_dir: Path, glob: str = "*.piprj") -> List[Path]:
    """Return any ``.piprj`` project files in *src_dir* (sorted, may be empty)."""
    return sorted(Path(src_dir).glob(glob))


def parse_piprj(piprj_path: Path, progress: bool = True) -> Dict[str, object]:
    """Extract acquisition metadata from a ``.piprj`` project archive.

    Best effort — never raises on a malformed/foreign archive; returns a dict
    with the keys below (any value may be ``None`` when absent/unreadable):

        pixel_um         pixel pitch in microns (from ``ProjectUnits.Size``)
        pixel_size_raw   the raw ``Size`` value (file's own unit)
        pixel_unit_code  the ``Name`` unit code (3 == micron)
        stage_origin_um  (StartX, StartY) or None
        project_type     ``metadata.json``'s ``projectType`` or None
        source           the ``.piprj`` filename
    """
    import json
    import zipfile

    piprj_path = Path(piprj_path)
    meta: Dict[str, object] = {
        "pixel_um": None, "pixel_size_raw": None, "pixel_unit_code": None,
        "stage_origin_um": None, "project_type": None, "source": piprj_path.name,
    }
    try:
        with zipfile.ZipFile(piprj_path) as z:
            names = set(z.namelist())
            if "ProjectUnits.json" in names:
                pu = json.loads(z.read("ProjectUnits.json").decode("utf-8", "replace"))
                size = pu.get("Size")
                unit = pu.get("Name")
                meta["pixel_unit_code"] = unit
                if size is not None:
                    meta["pixel_size_raw"] = float(size)
                    # Size is the pixel pitch in microns (unit code 3 == µm).
                    meta["pixel_um"] = float(size)
                    if (unit is not None and unit not in _PIPRJ_KNOWN_UM_UNIT_CODES
                            and progress):
                        print(f"[oxford] WARNING: {piprj_path.name} unit code "
                              f"{unit!r} is unrecognised; assuming Size={size} is "
                              "in microns. Override with --pixel-um if wrong.",
                              flush=True)
                sx, sy = pu.get("StartX"), pu.get("StartY")
                if sx is not None and sy is not None:
                    meta["stage_origin_um"] = (float(sx), float(sy))
            if "metadata.json" in names:
                md = json.loads(z.read("metadata.json").decode("utf-8", "replace"))
                meta["project_type"] = md.get("projectType")
    except Exception as exc:
        if progress:
            print(f"[oxford] could not parse {piprj_path.name}: {exc}", flush=True)
    return meta


# ---------------------------------------------------------------------------
# CSV → ndarray
# ---------------------------------------------------------------------------

def _is_decimal_comma_csv(text: str) -> bool:
    """True when *text* is a numeric CSV written with a decimal COMMA.

    Such a file has no "." at all, and either ragged rows (a value with a
    decimal part takes two fields, a zero takes one) or tokens with a leading
    zero ("05341"), which no integer map contains.
    """
    if "." in text:
        return False
    if _DECCOMMA_LEADING_ZERO.search(text):
        return True
    widths = {ln.rstrip(",").count(",") for ln in text.splitlines() if ln.strip()}
    return len(widths) > 1


def _deccomma_run(m) -> str:
    """Pass-2 replacement: pair up a run of short tokens (see ``_DECCOMMA_P2``)."""
    lead, toks = m.group(1) or "", m.group(2).split(",")
    out = []
    if len(toks) % 2 == 1:
        if lead:                          # "0,5" = 0.5 (not a zero and an integer 5)
            out.append("0." + toks[0])
            lead = ""
        else:                             # an exact integer value, kept as is
            out.append(toks[0])
        toks = toks[1:]
    out += [f"{a}.{b}" for a, b in zip(toks[0::2], toks[1::2])]
    return lead + ",".join(out)


def _repair_decimal_comma(text: str) -> str:
    """Rewrite a decimal-comma CSV (see ``_DECCOMMA_P1``) with decimal points.

    One case cannot be decided from the tokens alone: ``0,1`` is 0.1 by the
    grammar, but a zero followed by an exact integer 1 writes the same two
    tokens (about one value in 10**7 is an exact integer, so a large map holds
    a few). Every row of a map has the same width, and a wrong pairing can
    only shorten a row, so a row that comes out ``k`` values short of the
    widest row is re-read with its ``k`` such pairs kept apart; a row that
    cannot be brought to that width this way is left as it is for the caller
    to reject.
    """
    raw_rows = text.splitlines()
    rows = _DECCOMMA_P2.sub(_deccomma_run, _DECCOMMA_P1.sub(r"\1.\2", text)).splitlines()
    rows = [re.sub(r",[ \t]*$", "", r) for r in rows]      # AZtec ends each row with a comma
    widths = [r.count(",") + 1 for r in rows if r.strip()]
    if widths:
        # A wrong pairing can only merge two values into one, never split one,
        # so the widest row is the true map width.
        target = max(widths)
        for i, r in enumerate(rows):
            short = target - (r.count(",") + 1) if r.strip() else 0
            if short <= 0:
                continue
            s1 = _DECCOMMA_P1.sub(r"\1.\2", raw_rows[i])
            cands = [m.span() for m in _DECCOMMA_P2.finditer(s1)
                     if m.group(1) and "," not in m.group(2)]
            if len(cands) != short:
                continue                                  # undecidable: caller rejects the file
            keep = set(cands)
            fixed = _DECCOMMA_P2.sub(
                lambda m: m.group(0) if m.span() in keep else _deccomma_run(m), s1)
            rows[i] = re.sub(r",[ \t]*$", "", fixed)
    return "\n".join(rows) + "\n"


def _read_element_csv(path: Path) -> np.ndarray:
    """Read an Oxford 'Montaged Map Data-{El} Wt%.csv' into float32 (H, W).

    Tolerates a leading header row by detecting non-numeric first row, and a
    file exported under a decimal-comma locale (repaired, then checked to be
    rectangular; see ``_repair_decimal_comma``).
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        pd = None  # type: ignore

    # A file with no "." in its first megabyte is either an integer map or a
    # decimal-comma export; only then is the whole text read and tested.
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        head = fh.read(1 << 20)
    if "." not in head:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        if _is_decimal_comma_csv(text):
            import io
            text = _repair_decimal_comma(text)
            widths = {ln.count(",") + 1 for ln in text.splitlines() if ln.strip()}
            if len(widths) != 1:
                raise ValueError(
                    f"{Path(path).name}: written with a decimal comma (the comma is both "
                    f"the field delimiter and the decimal mark) and could not be "
                    f"repaired unambiguously (row widths {sorted(widths)[:4]} ...). "
                    f"Re-export with '.' as the Windows decimal symbol.")
            if pd is not None:
                arr = pd.read_csv(io.StringIO(text), header=None,
                                  dtype=np.float32).to_numpy(dtype=np.float32)
            else:
                arr = np.loadtxt(io.StringIO(text), delimiter=",", dtype=np.float32, ndmin=2)
            print(f"[oxford] {Path(path).name}: decimal-comma export repaired "
                  f"-> {arr.shape[1]} x {arr.shape[0]} values")
            return np.ascontiguousarray(arr, dtype=np.float32)

    if pd is not None:
        # Try numeric read first; fall back to header=0 if the first line
        # contains text.
        try:
            arr = pd.read_csv(path, header=None, dtype=np.float32).to_numpy(
                dtype=np.float32
            )
        except (ValueError, TypeError):
            arr = pd.read_csv(path, header=0, dtype=np.float32).to_numpy(
                dtype=np.float32
            )
    else:
        # Pure-numpy fallback
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D CSV, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Collapse RGB(A) to grayscale; pass through 2D arrays."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] >= 3:
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(img.dtype)
        return img[..., 0]
    raise ValueError(f"Unexpected BSE image shape: {img.shape}")


def _resize_2d(arr: np.ndarray, target_shape: Tuple[int, int],
                resample: str = "bilinear") -> np.ndarray:
    """Resize a 2D float array to *target_shape* via PIL bilinear."""
    if arr.shape == target_shape:
        return arr
    if Image is None:
        raise RuntimeError("Pillow not installed; cannot resize arrays.")
    mode = {
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "bicubic": Image.BICUBIC,
    }[resample]
    # PIL works best with float32; resize takes (W, H). Pillow infers the "F"
    # (32-bit float) mode from the float32 dtype — passing mode= explicitly is
    # deprecated (removed in Pillow 13).
    pil = Image.fromarray(arr.astype(np.float32, copy=False))
    pil = pil.resize((target_shape[1], target_shape[0]), resample=mode)
    return np.asarray(pil, dtype=np.float32)


# ---------------------------------------------------------------------------
# BSE outputs (PNG 8-bit + TIFF 16-bit)
# ---------------------------------------------------------------------------

def _mean_z_proxy(element_maps: Dict[str, np.ndarray]) -> np.ndarray:
    """Per-pixel mean atomic number from element Wt% maps: Σ(Z·wt%) / Σ(wt%).

    Used as a BSE stand-in when the dataset ships no real BSE TIF. BSE
    intensity tracks mean Z, so this is the closest visual analogue
    available from quantified element maps. Elements without a known Z are
    skipped (no contribution). Background pixels (Σwt = 0) read as 0.
    NaN values are treated as zero contribution so Oxford NaN background
    masks drop out cleanly.
    """
    H, W = next(iter(element_maps.values())).shape
    weighted_sum = np.zeros((H, W), dtype=np.float32)
    total_wt    = np.zeros((H, W), dtype=np.float32)
    for el, arr in element_maps.items():
        z = _Z_BY_ELEMENT.get(el)
        if z is None:
            continue
        a = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
        weighted_sum += float(z) * a
        total_wt    += a
    out = np.zeros_like(weighted_sum)
    np.divide(weighted_sum, total_wt, out=out, where=total_wt > 0)
    return out


def _percentile_uint8(arr: np.ndarray, pct=(1.0, 99.0)) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, pct)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _to_uint16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint16)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint16)
    out = (arr - lo) / (hi - lo)
    return (out * 65535.0 + 0.5).astype(np.uint16)


def _write_bse_outputs(bse_grid: np.ndarray, out_dir: Path,
                        progress: bool = True) -> Tuple[Path, Path]:
    """Write stitched_BSE_8bit.png and stitched_BSE_16bit.tiff."""
    # PNG via PIL directly, not imageio: imageio's pillow plugin can silently
    # fail to register (seen with imageio 2.37 + Pillow 12) and then imwrite
    # produces a file that PIL itself cannot identify. PIL is a hard dep here.
    if Image is None:
        raise RuntimeError("Pillow not installed; cannot write BSE PNG.")
    if tifffile is None:
        raise RuntimeError("tifffile not installed; cannot write BSE TIFF.")
    png_path = out_dir / "stitched_BSE_8bit.png"
    tif_path = out_dir / "stitched_BSE_16bit.tiff"
    if progress:
        print(f"[oxford] writing {png_path.name}", flush=True)
    Image.fromarray(_percentile_uint8(bse_grid)).save(png_path, format="PNG")
    if progress:
        print(f"[oxford] writing {tif_path.name}", flush=True)
    tifffile.imwrite(tif_path, _to_uint16(bse_grid))
    return png_path, tif_path


# ---------------------------------------------------------------------------
# Manifest (single-row CSV the existing pipeline can read)
# ---------------------------------------------------------------------------

def _write_stitch_manifest(out_dir: Path, H: int, W: int,
                            elements: List[str],
                            pixel_um: Optional[float],
                            bse_h: Optional[int] = None,
                            bse_w: Optional[int] = None,
                            scale: float = 1.0) -> Path:
    """Write a minimal one-row ``stitch_manifest.csv`` for the Oxford montage.

    The columns mirror what ``eds_map.write_tile_manifest`` produces so
    ``read_manifest`` parses cleanly and ``viewer.py`` is happy. Per-tile
    fields collapse to a single tile that *is* the whole mosaic.

    ``H``/``W`` are the **EDS (element-map) grid** dimensions. ``bse_h``/``bse_w``
    default to ``H``/``W`` (single-grid montage); pass the native BSE dimensions
    for a two-resolution montage (BSE kept native, element cache coarser) and set
    ``scale`` to the integer BSE:EDS ratio.
    """
    out_csv = out_dir / "stitch_manifest.csv"
    bse_h = int(bse_h) if bse_h is not None else int(H)
    bse_w = int(bse_w) if bse_w is not None else int(W)
    elements_csv = ";".join(elements)
    lines_csv = ";".join(f"{el}_{_DEFAULT_LINE.get(el, 'K_a')}" for el in elements)
    beam_energy = ""
    # Energy axis is a placeholder — the cache values are already integrated.
    E0, dE, nE = 0.0, 0.01, 2048

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "row", "col", "y", "x", "file",
            "bse_h", "bse_w", "eds_h", "eds_w",
            "scale_y", "scale_x", "shx", "shy",
            "E0", "dE", "nE", "energy_units",
            "beam_energy", "elements", "xray_lines",
        ])
        w.writerow([
            0, 0, 0, 0, "oxford_montage",
            bse_h, bse_w, H, W,
            float(scale), float(scale), 0.0, 0.0,
            E0, dE, nE, "keV",
            beam_energy, elements_csv, lines_csv,
        ])
    return out_csv


# ---------------------------------------------------------------------------
# Synthetic cache builder
# ---------------------------------------------------------------------------

def _build_synthetic_cache(out_dir: Path,
                            manifest_csv: Path,
                            element_maps: Dict[str, np.ndarray],
                            H: int, W: int,
                            pixel_um: Optional[float],
                            source_label: str,
                            data_type: str = "oxford_wt_pct",
                            extra_meta: Optional[Dict[str, object]] = None,
                            ds: int = 1,
                            H_bse: Optional[int] = None,
                            W_bse: Optional[int] = None,
                            cache_dir: Optional[Path] = None,
                            coating_elements: Sequence[str] = ("C",),
                            progress: bool = True) -> Path:
    """Populate the EDS cache from in-memory Oxford element arrays.

    ``data_type`` tags the cache so the viewer adapts: ``"oxford_wt_pct"`` for
    quantified Wt% maps (structural mineral ID), ``"oxford_counts"`` for
    uncalibrated window-integral counts from an h5oina export (ratio-only ID).

    ``H``/``W`` are the element-cache grid (where the memmaps are sized).
    ``ds`` is the BSE:cache resolution ratio (>1 for a two-resolution montage
    where the BSE is kept at native resolution); ``H_bse``/``W_bse`` are then the
    native BSE dimensions and the cache read path (``eds_cache.roi_from_cache``)
    maps BSE-space ROIs onto the coarser cache. ``ds=1`` keeps both on one grid.

    ``cache_dir`` relocates the cache (e.g. to an SSD); a sidecar
    ``eds_cache_link.txt`` in *out_dir* is written so the viewer and later runs
    find it, mirroring ``stitch.py --eds-cache-dir``. Default:
    ``<out_dir>/eds_cache/`` (any stale sidecar is cleared).
    """
    if cache_dir is None:
        cache_dir = _edsc.cache_dir_for(out_dir)
        _edsc.clear_cache_link(out_dir)  # fresh build at default location
    else:
        cache_dir = Path(cache_dir).resolve()
        _edsc.write_cache_link(out_dir, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    elements = sorted(element_maps.keys())

    # weight.npy = ones, every pixel contributed by the single "tile".
    weight = np.lib.format.open_memmap(
        _edsc._weight_path(cache_dir),
        mode="w+", dtype=np.float32, shape=(H, W),
    )
    weight[:] = 1.0
    weight.flush()

    # Per-element memmaps.
    for el in elements:
        arr = element_maps[el]
        if arr.shape != (H, W):
            raise ValueError(
                f"Element {el} has shape {arr.shape}, expected ({H}, {W})"
            )
        if progress:
            print(f"[oxford] writing cache memmap for {el} ({H}x{W})", flush=True)
        main = np.lib.format.open_memmap(
            _edsc._element_path(cache_dir, el, "main"),
            mode="w+", dtype=np.float32, shape=(H, W),
        )
        main[:] = arr.astype(np.float32, copy=False)
        main.flush()

        # Sidebands: zero-init memmaps. NTFS sparse-file / VDL semantics
        # mean unwritten pages read as zero — no explicit write needed.
        for band in ("sb_lo", "sb_hi"):
            mm = np.lib.format.open_memmap(
                _edsc._element_path(cache_dir, el, band),
                mode="w+", dtype=np.float32, shape=(H, W),
            )
            mm.flush()
            del mm

    ds = max(1, int(ds))
    H_bse = int(H_bse) if H_bse is not None else int(H)
    W_bse = int(W_bse) if W_bse is not None else int(W)
    manifest = {
        "stitch_fingerprint": _edsc.stitch_fingerprint(manifest_csv),
        "elements": elements,
        "width_eV": 0.0,
        "sideband_mult": 0.0,
        "ds": ds,
        "H": int(H), "W": int(W),
        "H_bse": H_bse, "W_bse": W_bse,
        "E0": 0.0, "dE": 0.01, "nE": 2048,
        "shift_method": "n/a (oxford montage)",
        "data_type": data_type,
        "source": source_label,
        "pixel_um": pixel_um,
        "coating_elements": list(coating_elements),
    }
    if extra_meta:
        # Additive provenance fields (stage origin, unit code, .piprj source).
        # eds_cache.is_valid_for ignores them, so they never affect validation.
        for k, v in extra_meta.items():
            manifest.setdefault(k, v)
    _edsc.save_manifest(cache_dir, manifest)
    return cache_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_oxford_dataset(src_dir: Path,
                         out_dir: Optional[Path] = None,
                         bse_tif_glob: str = "*BSE*.tif",
                         element_csv_glob: str = "Montaged Map Data-* Wt%.csv",
                         target_resolution: str = "elements",
                         force: bool = False,
                         pixel_um: Optional[float] = None,
                         piprj: Optional[Path] = None,
                         use_piprj: bool = True,
                         cache_dir: Optional[Path] = None,
                         coating_elements: Sequence[str] = ("C",),
                         progress: bool = True) -> Path:
    """Convert an Oxford AZtec CSV export into a viewer-ready mosaic dir.

    Parameters
    ----------
    src_dir : Path
        Directory containing the Oxford BSE TIF + element Wt% CSVs.
    out_dir : Path, optional
        Where to write outputs. Defaults to ``src_dir`` (in-place conversion).
    target_resolution : {"elements", "bse"}
        - "elements" (default): downsample BSE to the element-map grid.
        - "bse": upsample element maps to the BSE grid.
    force : bool
        If True, overwrite an existing valid cache.
    pixel_um : float, optional
        Pixel pitch in microns. If None, it is auto-detected from a ``.piprj``
        project file in *src_dir* (see ``use_piprj``); an explicit value here
        always overrides the parsed one. Stored in the cache manifest.
    piprj : Path, optional
        Explicit ``.piprj`` to read metadata from. If None, *src_dir* is
        searched for ``*.piprj``.
    use_piprj : bool
        If True (default), parse a ``.piprj`` for pixel size + stage origin.
    cache_dir : Path, optional
        Relocate the EDS cache (e.g. to an SSD). A sidecar
        ``eds_cache_link.txt`` is written in *out_dir* so the viewer and later
        runs find it automatically. Default: ``<out_dir>/eds_cache/``.

    Returns
    -------
    Path
        The output directory (where ``viewer.py --dir`` should point).
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir) if out_dir else src_dir

    bse_tif, element_csvs = discover_inputs(
        src_dir, bse_tif_glob=bse_tif_glob, element_csv_glob=element_csv_glob,
    )
    if progress:
        print(f"[oxford] src_dir: {src_dir}")
        print(f"[oxford] out_dir: {out_dir}")
        print(f"[oxford] BSE TIF: {bse_tif.name if bse_tif else '(none, will synthesise mean-Z proxy)'}")
        print(f"[oxford] elements ({len(element_csvs)}): "
              f"{', '.join(sorted(element_csvs))}", flush=True)

    # --- .piprj metadata (pixel size, stage origin) ----------------------
    piprj_meta: Dict[str, object] = {}
    if use_piprj:
        piprj_files = [Path(piprj)] if piprj else discover_piprj(src_dir)
        for pf in piprj_files:
            m = parse_piprj(pf, progress=progress)
            if m.get("pixel_um") is not None:
                piprj_meta = m
                if progress:
                    print(f"[oxford] .piprj {pf.name}: pixel_um="
                          f"{m['pixel_um']} (unit code {m.get('pixel_unit_code')}), "
                          f"stage_origin_um={m.get('stage_origin_um')}, "
                          f"projectType={m.get('project_type')}", flush=True)
                break
    if pixel_um is None and piprj_meta.get("pixel_um") is not None:
        pixel_um = float(piprj_meta["pixel_um"])  # type: ignore[arg-type]
        if progress:
            print(f"[oxford] using pixel size {pixel_um} um from "
                  f"{piprj_meta.get('source')}", flush=True)
    elif pixel_um is not None and piprj_meta.get("pixel_um") is not None and progress:
        print(f"[oxford] --pixel-um {pixel_um} overrides .piprj value "
              f"{piprj_meta.get('pixel_um')}", flush=True)

    # Load element maps
    element_maps: Dict[str, np.ndarray] = {}
    el_shape: Optional[Tuple[int, int]] = None
    for el in sorted(element_csvs):
        if progress:
            print(f"[oxford] reading {element_csvs[el].name}", flush=True)
        arr = _read_element_csv(element_csvs[el])
        if el_shape is None:
            el_shape = arr.shape
        elif arr.shape != el_shape:
            raise ValueError(
                f"Element map shape mismatch: {el} is {arr.shape}, "
                f"earlier elements were {el_shape}."
            )
        element_maps[el] = arr
    assert el_shape is not None

    # Load BSE if present. Many Oxford TIFFs use LZW compression — imageio/
    # tifffile need the optional 'imagecodecs' package for that, while PIL
    # handles it natively. Try imageio first, fall back to PIL. If no BSE
    # ships with the dataset, synthesise a mean-Z proxy from the element
    # maps and force target_resolution="elements" (no BSE shape exists).
    el_h, el_w = el_shape
    # Two-resolution montage state: cache_ds is the BSE:cache ratio (1 = both on
    # one grid), (H_bse, W_bse) the native BSE dims written to the PNG/TIFF.
    cache_ds = 1
    H_bse = W_bse = None
    if bse_tif is None:
        if target_resolution == "bse" and progress:
            print("[oxford] target_resolution='bse' requested but no BSE TIF "
                  "found — falling back to 'elements'.", flush=True)
        target_resolution = "elements"
        H, W = el_h, el_w
        bse_grid = _mean_z_proxy(element_maps)
        if progress:
            print(f"[oxford] synthesised mean-Z BSE proxy: {bse_grid.shape}, "
                  f"range [{float(np.nanmin(bse_grid)):.2f}, "
                  f"{float(np.nanmax(bse_grid)):.2f}]", flush=True)
    else:
        bse_raw = None
        if imageio is not None:
            try:
                bse_raw = imageio.v2.imread(bse_tif)
            except Exception as exc:
                if progress:
                    print(f"[oxford] imageio failed on BSE ({exc}); "
                          "falling back to PIL.", flush=True)
                bse_raw = None
        if bse_raw is None:
            if Image is None:
                raise RuntimeError(
                    "Neither imageio nor PIL could read the BSE TIF "
                    f"({bse_tif}). Install 'imagecodecs' for tifffile/imageio, "
                    "or 'Pillow' for the PIL fallback."
                )
            with Image.open(bse_tif) as pil_img:
                bse_raw = np.array(pil_img)
        bse_gray = _to_gray(bse_raw)
        bse_h, bse_w = bse_gray.shape

        if progress:
            print(f"[oxford] BSE shape: {bse_gray.shape} | "
                  f"element shape: {el_shape} | "
                  f"target_resolution: {target_resolution}", flush=True)

        if target_resolution == "elements":
            H, W = el_h, el_w
            # Two-resolution path: when the BSE is an exact integer multiple of
            # the element grid (same factor on both axes), keep the BSE at its
            # native resolution and store the element cache at the coarser grid
            # with ds = that factor — instead of throwing the BSE detail away by
            # downsampling it. eds_cache.roi_from_cache maps BSE-space ROIs onto
            # the coarser cache transparently (the same machinery Bruker ds>1
            # caches use). Falls back to single-grid downsampling otherwise.
            ratio_y = bse_h // el_h if el_h else 0
            ratio_x = bse_w // el_w if el_w else 0
            if (ratio_y >= 2 and ratio_y == ratio_x
                    and ratio_y * el_h == bse_h and ratio_x * el_w == bse_w):
                cache_ds = int(ratio_y)
                H_bse, W_bse = bse_h, bse_w
                bse_grid = bse_gray.astype(np.float32, copy=False)
                # element_maps stay at (H, W); cache is sized at (H, W).
                if progress:
                    print(f"[oxford] two-resolution montage: BSE kept native "
                          f"({bse_h}x{bse_w}), element cache at {el_h}x{el_w} "
                          f"(ds={cache_ds}).", flush=True)
            else:
                bse_grid = _resize_2d(bse_gray.astype(np.float32), (H, W),
                                      resample="bilinear")
                if progress and (bse_h, bse_w) != (el_h, el_w):
                    print(f"[oxford] BSE {bse_h}x{bse_w} is not an exact integer "
                          f"multiple of the element grid {el_h}x{el_w}; "
                          f"downsampling BSE to the element grid (single grid).",
                          flush=True)
        elif target_resolution == "bse":
            H, W = bse_h, bse_w
            bse_grid = bse_gray.astype(np.float32, copy=False)
            for el in list(element_maps.keys()):
                element_maps[el] = _resize_2d(element_maps[el], (H, W),
                                               resample="bilinear")
        else:
            raise ValueError(
                f"Unknown target_resolution: {target_resolution!r}. "
                "Use 'elements' or 'bse'."
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write BSE PNG/TIFF first so the manifest's fingerprint isn't influenced
    # by their presence (the fingerprint is over stitch_manifest.csv only).
    _write_bse_outputs(bse_grid, out_dir, progress=progress)

    # Write stitch_manifest.csv (single-row).
    elements = sorted(element_maps.keys())
    manifest_csv = _write_stitch_manifest(out_dir, H, W, elements,
                                           pixel_um=pixel_um,
                                           bse_h=H_bse, bse_w=W_bse,
                                           scale=float(cache_ds))
    if progress:
        print(f"[oxford] wrote {manifest_csv.name}", flush=True)

    # Idempotency: if a cache already exists and matches this manifest, skip
    # unless --force. An explicit cache_dir wins; otherwise follow any sidecar.
    check_dir = (Path(cache_dir).resolve() if cache_dir is not None
                 else _edsc.resolve_cache_dir(out_dir))
    if not force and _edsc.is_valid_for(check_dir, manifest_csv,
                                         required_elements=elements):
        if progress:
            print(f"[oxford] cache up to date at {check_dir}, skipping rebuild "
                  f"(pass --force to override).", flush=True)
        return out_dir

    # Build the synthetic cache.
    extra_meta: Dict[str, object] = {}
    if piprj_meta:
        extra_meta = {
            "stage_origin_um": piprj_meta.get("stage_origin_um"),
            "pixel_unit_code": piprj_meta.get("pixel_unit_code"),
            "project_type": piprj_meta.get("project_type"),
            "piprj_source": piprj_meta.get("source"),
        }
    _build_synthetic_cache(
        out_dir, manifest_csv, element_maps,
        H=H, W=W, pixel_um=pixel_um,
        source_label="Oxford AZtec CSV export",
        extra_meta=extra_meta,
        ds=cache_ds, H_bse=H_bse, W_bse=W_bse,
        cache_dir=cache_dir,
        coating_elements=coating_elements,
        progress=progress,
    )

    if progress:
        print(f"[oxford] done. Run viewer with:")
        print(f"  python viewer.py --dir \"{out_dir}\"", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# h5oina (Oxford HDF5 export: EBSD + EDS + electron image on one grid)
# ---------------------------------------------------------------------------
# An ``.h5oina`` stores every map as a flat per-pixel array in C-order with X
# varying fastest, so a reshape to ``(Y Cells, X Cells)`` recovers the image.
# EDS lives under ``<top>/EDS/Data/Window Integral/{El line}`` as *window-
# integral counts* (uncalibrated intensities — NOT wt%); EBSD under
# ``<top>/EBSD`` (phase map + named phases); the electron image under
# ``<top>/Electron Image/Data/FSE/*`` (forward-scatter channels).

# Trailing chemical formula on an EBSD phase name, e.g. "Diopside  CaMgSi2O6".
_FORMULA_TOKEN_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def _h5_top_group(f) -> str:
    """Return the top-level group key that holds EDS/EBSD (usually '1')."""
    for k in f.keys():
        try:
            g = f[k]
            if hasattr(g, "keys") and ("EDS" in g or "EBSD" in g):
                return k
        except Exception:
            continue
    # Fall back to the first key.
    return next(iter(f.keys()))


def _h5_scalar(f, path):
    """Read a length-1 dataset as a plain Python scalar (or None if absent)."""
    try:
        v = f[path][()]
    except Exception:
        return None
    arr = np.asarray(v).ravel()
    if arr.size == 0:
        return None
    return arr[0]


def _h5_str(f, path) -> Optional[str]:
    v = _h5_scalar(f, path)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _parse_phase_name(raw: str) -> Tuple[str, Optional[str]]:
    """Split "Diopside   CaMgSi2O6" → ("Diopside", "CaMgSi2O6").

    Returns ``(name, formula_or_None)``. Handles names with no formula
    ("Chlorite") and single-space "Name Formula" separation.
    """
    s = " ".join((raw or "").split())  # collapse whitespace runs
    if not s:
        return "Unknown", None
    toks = s.split(" ")
    if len(toks) >= 2:
        last = toks[-1]
        if any(c.isdigit() for c in last) and _FORMULA_TOKEN_RE.match(last):
            return " ".join(toks[:-1]), last
    return s, None


def _read_h5oina_eds(f, top: str, progress: bool = True,
                     ) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Read EDS window-integral count maps as ``{element: (H, W) float32}``."""
    yc = int(np.asarray(_h5_scalar(f, f"{top}/EDS/Header/Y Cells")).item())
    xc = int(np.asarray(_h5_scalar(f, f"{top}/EDS/Header/X Cells")).item())
    grp = f[f"{top}/EDS/Data/Window Integral"]
    element_maps: Dict[str, np.ndarray] = {}
    for key in grp.keys():
        el = key.split(" ", 1)[0].strip()
        if el not in _Z_BY_ELEMENT:
            if progress:
                print(f"[oxford] h5oina: skipping non-element channel {key!r}",
                      flush=True)
            continue
        if el in element_maps:
            continue
        arr = np.asarray(grp[key][()], dtype=np.float32)
        if arr.size != yc * xc:
            raise ValueError(
                f"EDS channel {key!r} has {arr.size} px, expected {yc}*{xc}="
                f"{yc * xc}.")
        element_maps[el] = np.ascontiguousarray(arr.reshape(yc, xc))
        if progress:
            print(f"[oxford] h5oina: read EDS {el} ({yc}x{xc})", flush=True)
    if not element_maps:
        raise ValueError("No element channels found under EDS/Window Integral.")
    return element_maps, yc, xc


def _read_h5oina_fse(f, top: str, target_hw: Tuple[int, int],
                     progress: bool = True) -> Optional[np.ndarray]:
    """Sum the forward-scatter channels into a virtual-BSE grid (resized to EDS).

    Returns ``None`` when no FSE data is present (caller falls back to mean-Z).
    """
    fse_path = f"{top}/Electron Image/Data/FSE"
    if fse_path not in f:
        return None
    grp = f[fse_path]
    yc = _h5_scalar(f, f"{top}/Electron Image/Header/Y Cells")
    xc = _h5_scalar(f, f"{top}/Electron Image/Header/X Cells")
    if yc is None or xc is None:
        return None
    yc, xc = int(np.asarray(yc).item()), int(np.asarray(xc).item())
    acc: Optional[np.ndarray] = None
    n = 0
    for key in grp.keys():
        ds = grp[key]
        if ds.size != yc * xc:
            continue
        chan = np.asarray(ds[()], dtype=np.float32)
        acc = chan if acc is None else (acc + chan)
        n += 1
    if acc is None or n == 0:
        return None
    grid = acc.reshape(yc, xc)
    if progress:
        print(f"[oxford] h5oina: summed {n} FSE channels ({yc}x{xc}) "
              f"-> resize to {target_hw}", flush=True)
    return _resize_2d(grid, target_hw, resample="bilinear")


def _read_h5oina_ebsd(f, top: str) -> Tuple[Optional[np.ndarray], Dict[int, dict]]:
    """Read the EBSD phase map ``(H, W) uint8`` + ``{id: {name, formula, color}}``.

    Returns ``(None, {})`` when no EBSD group is present.
    """
    if f"{top}/EBSD/Data/Phase" not in f:
        return None, {}
    yc = int(np.asarray(_h5_scalar(f, f"{top}/EBSD/Header/Y Cells")).item())
    xc = int(np.asarray(_h5_scalar(f, f"{top}/EBSD/Header/X Cells")).item())
    phase = np.asarray(f[f"{top}/EBSD/Data/Phase"][()], dtype=np.uint8)
    if phase.size != yc * xc:
        raise ValueError(
            f"EBSD Phase has {phase.size} px, expected {yc}*{xc}={yc * xc}.")
    phase_map = np.ascontiguousarray(phase.reshape(yc, xc))

    # NB: the Phase *data* array holds 1-based ordinals into the Phases group
    # (1, 2, 3, …; 0 = unindexed), NOT the per-phase database "Phase Id". So the
    # legend must be keyed by the group ordinal int(pkey) to line up with the
    # map. The database Phase Id is kept as a reference field only.
    legend: Dict[int, dict] = {0: {"name": "Unindexed", "formula": None,
                                   "color": [0, 0, 0]}}
    phases_path = f"{top}/EBSD/Header/Phases"
    if phases_path in f:
        for pkey in f[phases_path].keys():
            try:
                ordinal = int(pkey)
            except (TypeError, ValueError):
                continue
            base = f"{phases_path}/{pkey}"
            pid = _h5_scalar(f, f"{base}/Phase Id")
            pid = int(np.asarray(pid).item()) if pid is not None else None
            raw_name = _h5_str(f, f"{base}/Phase Name") or f"Phase {ordinal}"
            name, formula = _parse_phase_name(raw_name)
            color = [128, 128, 128]
            try:
                c = np.asarray(f[f"{base}/Color"][()]).ravel()[:3]
                if c.size == 3:
                    color = [int(x) for x in c]
            except Exception:
                pass
            legend[ordinal] = {"name": name, "formula": formula,
                               "color": color, "phase_id": pid}
    return phase_map, legend


def _write_ebsd_artifacts(out_dir: Path, phase_map: np.ndarray,
                          legend: Dict[int, dict], progress: bool = True,
                          ) -> Tuple[Path, Path]:
    """Write ``ebsd_phase_map.npy`` + ``ebsd_phases.json`` into *out_dir*."""
    import json
    npy_path = out_dir / "ebsd_phase_map.npy"
    json_path = out_dir / "ebsd_phases.json"
    np.save(npy_path, phase_map.astype(np.uint8, copy=False))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in legend.items()}, fh, indent=2)
    if progress:
        named = ", ".join(f"{k}:{v['name']}" for k, v in sorted(legend.items())
                          if k != 0)
        print(f"[oxford] h5oina: wrote {npy_path.name} + {json_path.name} "
              f"[{named}]", flush=True)
    return npy_path, json_path


def load_h5oina_dataset(src: Path,
                        out_dir: Optional[Path] = None,
                        force: bool = False,
                        base_layer: str = "fse",
                        cache_dir: Optional[Path] = None,
                        coating_elements: Sequence[str] = ("C",),
                        progress: bool = True) -> Path:
    """Convert an Oxford ``.h5oina`` (EBSD + EDS) export into a viewer-ready dir.

    Writes the same ``eds_cache/`` + ``stitch_manifest.csv`` + BSE PNG/TIFF that
    ``stitch.py`` / ``load_oxford_dataset`` produce — so ``viewer.py`` is
    unchanged — plus ``ebsd_phase_map.npy`` + ``ebsd_phases.json`` carrying the
    indexed EBSD phase map and its crystallographic names.

    Parameters
    ----------
    src : Path
        The ``.h5oina`` file, or a project directory containing one (also
        searched in an ``h5oina/`` subfolder).
    out_dir : Path, optional
        Output directory. Defaults to the ``.h5oina``'s parent's project root:
        if *src* is a dir we use it; if a file, its parent (or its grandparent
        when the file sits in an ``h5oina/`` subfolder).
    base_layer : {"fse", "meanz"}
        Base image: summed forward-scatter channels (default, a real
        virtual-BSE) or a synthesised mean-Z proxy from the element maps.
    cache_dir : Path, optional
        Relocate the EDS cache (e.g. to an SSD); a sidecar
        ``eds_cache_link.txt`` is written in *out_dir* so the viewer finds it.

    The EDS maps are uncalibrated window-integral counts, so the cache is tagged
    ``data_type="oxford_counts"`` → the viewer keeps mineral ID ratio-only.
    """
    if h5py is None:
        raise RuntimeError(
            "h5py is required to read .h5oina files but is not installed. "
            "Install it with: pip install h5py")

    src = Path(src)
    h5_path = discover_h5oina(src)
    if h5_path is None:
        raise FileNotFoundError(f"No .h5oina found at or under {src}")

    if out_dir is None:
        if src.is_dir():
            out_dir = src
        elif h5_path.parent.name.lower() == "h5oina":
            out_dir = h5_path.parent.parent
        else:
            out_dir = h5_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"[oxford] h5oina: {h5_path}")
        print(f"[oxford] out_dir: {out_dir}", flush=True)

    with h5py.File(h5_path, "r") as f:
        top = _h5_top_group(f)

        # --- cheap header reads (no big arrays yet) ----------------------
        yc = int(np.asarray(_h5_scalar(f, f"{top}/EDS/Header/Y Cells")).item())
        xc = int(np.asarray(_h5_scalar(f, f"{top}/EDS/Header/X Cells")).item())
        H, W = yc, xc
        x_step = _h5_scalar(f, f"{top}/EDS/Header/X Step")
        pixel_um = float(np.asarray(x_step).item()) if x_step is not None else None
        kv = _h5_scalar(f, f"{top}/EDS/Header/Beam Voltage")
        beam_kv = float(np.asarray(kv).item()) if kv is not None else None

        element_keys = list(f[f"{top}/EDS/Data/Window Integral"].keys())
        elements = sorted({k.split(" ", 1)[0].strip() for k in element_keys
                           if k.split(" ", 1)[0].strip() in _Z_BY_ELEMENT})
        if progress:
            print(f"[oxford] h5oina: grid {H}x{W}, pixel {pixel_um} um, "
                  f"{beam_kv} kV, elements ({len(elements)}): "
                  f"{', '.join(elements)}", flush=True)

        # --- EBSD artifacts (cheap; always (re)written) ------------------
        phase_map, legend = _read_h5oina_ebsd(f, top)
        if phase_map is not None:
            _write_ebsd_artifacts(out_dir, phase_map, legend, progress=progress)
        elif progress:
            print("[oxford] h5oina: no EBSD phase map present.", flush=True)

        # --- stitch manifest (needed for the cache fingerprint) ----------
        manifest_csv = _write_stitch_manifest(out_dir, H, W, elements,
                                              pixel_um=pixel_um)
        if progress:
            print(f"[oxford] wrote {manifest_csv.name}", flush=True)

        # --- idempotency: skip the heavy element read on a valid cache ---
        check_dir = (Path(cache_dir).resolve() if cache_dir is not None
                     else _edsc.resolve_cache_dir(out_dir))
        if not force and _edsc.is_valid_for(check_dir, manifest_csv,
                                            required_elements=elements):
            if progress:
                print(f"[oxford] cache up to date at {check_dir}, skipping "
                      f"rebuild (pass --force to override).", flush=True)
            return out_dir

        # --- heavy reads + build -----------------------------------------
        element_maps, _, _ = _read_h5oina_eds(f, top, progress=progress)

        bse_grid: Optional[np.ndarray] = None
        if base_layer == "fse":
            try:
                bse_grid = _read_h5oina_fse(f, top, (H, W), progress=progress)
            except Exception as exc:
                if progress:
                    print(f"[oxford] h5oina: FSE base failed ({exc}); using "
                          "mean-Z proxy.", flush=True)
                bse_grid = None
        if bse_grid is None:
            bse_grid = _mean_z_proxy(element_maps)
            if progress:
                print(f"[oxford] h5oina: synthesised mean-Z BSE proxy "
                      f"{bse_grid.shape}", flush=True)

    # h5 file closed; finish writing outputs.
    _write_bse_outputs(bse_grid, out_dir, progress=progress)

    extra_meta: Dict[str, object] = {
        "provenance": "h5oina",
        "h5oina_source": h5_path.name,
        "beam_voltage_kV": beam_kv,
    }
    if phase_map is not None:
        extra_meta["ebsd_phase_map"] = "ebsd_phase_map.npy"
        extra_meta["ebsd_phases"] = "ebsd_phases.json"
    _build_synthetic_cache(
        out_dir, manifest_csv, element_maps,
        H=H, W=W, pixel_um=pixel_um,
        source_label="Oxford H5OINA (EBSD+EDS montage)",
        data_type="oxford_counts",
        extra_meta=extra_meta,
        cache_dir=cache_dir,
        coating_elements=coating_elements,
        progress=progress,
    )

    if progress:
        print(f"[oxford] done. Run viewer with:")
        print(f"  python viewer.py --dir \"{out_dir}\"", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dir", required=True, type=str,
                    help="Source directory with the Oxford CSV export.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output directory (defaults to --dir, i.e. in-place).")
    ap.add_argument("--resolution", choices=("elements", "bse"),
                    default="elements",
                    help="Target mosaic grid: element-map size (default) or "
                         "BSE size (upsamples element maps).")
    ap.add_argument("--bse-glob", default="*BSE*.tif",
                    help="Glob for the BSE TIF (default: '*BSE*.tif').")
    ap.add_argument("--csv-glob", default="Montaged Map Data-* Wt%.csv",
                    help="Glob for the element Wt%% CSVs.")
    ap.add_argument("--pixel-um", type=float, default=None,
                    help="Pixel pitch in microns. Overrides the value parsed "
                         "from a .piprj; if omitted, auto-detected from one.")
    ap.add_argument("--piprj", type=str, default=None,
                    help="Explicit .piprj project file to read metadata from "
                         "(default: auto-discover *.piprj in --dir).")
    ap.add_argument("--no-piprj", action="store_true",
                    help="Don't read pixel size / stage origin from a .piprj.")
    ap.add_argument("--base-layer", choices=("fse", "meanz"), default="fse",
                    help="h5oina base image: summed forward-scatter channels "
                         "(default) or a synthesised mean-Z proxy.")
    ap.add_argument("--eds-cache-dir", type=str, default=None,
                    help="Build the EDS cache in this directory instead of "
                         "<out>/eds_cache/ (e.g. on an SSD). A sidecar "
                         "eds_cache_link.txt is written so the viewer and "
                         "later runs find it automatically.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing valid cache.")
    ap.add_argument("--coating-elements", nargs="*", default=["C"],
                    metavar="EL",
                    help="Conductive coating / contaminant element symbols to "
                         "ignore in mineral ID and calibration (default: C). "
                         "Use e.g. '--coating-elements Pt' for a Pt coat, or "
                         "'--coating-elements' with no value for an uncoated "
                         "sample. Recorded in the cache manifest; the viewer can "
                         "still override it per session.")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    try:
        # Prefer an .h5oina (Oxford HDF5 EBSD+EDS export) when present; fall
        # back to the per-element CSV export otherwise.
        h5 = discover_h5oina(Path(args.dir))
        if h5 is not None:
            load_h5oina_dataset(
                src=Path(args.dir),
                out_dir=Path(args.out) if args.out else None,
                force=args.force,
                base_layer=args.base_layer,
                cache_dir=Path(args.eds_cache_dir) if args.eds_cache_dir else None,
                coating_elements=tuple(args.coating_elements),
                progress=not args.quiet,
            )
            return 0
        load_oxford_dataset(
            src_dir=Path(args.dir),
            out_dir=Path(args.out) if args.out else None,
            bse_tif_glob=args.bse_glob,
            element_csv_glob=args.csv_glob,
            target_resolution=args.resolution,
            force=args.force,
            pixel_um=args.pixel_um,
            piprj=Path(args.piprj) if args.piprj else None,
            use_piprj=not args.no_piprj,
            cache_dir=Path(args.eds_cache_dir) if args.eds_cache_dir else None,
            coating_elements=tuple(args.coating_elements),
            progress=not args.quiet,
        )
        return 0
    except Exception as exc:
        print(f"[oxford] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
