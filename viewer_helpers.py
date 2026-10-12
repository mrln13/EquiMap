# viewer_helpers.py
"""
Low-level helpers for viewer.py: image conversion, pyramid building,
preview auto-loading, ROI rasterisation, manifest tile-bbox math, SPX
quant parsing, EMSA spectrum writing, Perple_X bulk-comp writing,
and element->oxide conversion.

These were previously module-level helpers at the top of viewer.py.
Moved out as part of the tier-B viewer refactor (see project_viewer_features
memory) so each concern lives in its own module.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import re

import numpy as np
import napari
import matplotlib
from matplotlib.path import Path as MplPath
from PIL import Image


# ---- Element line energies (keV). Keep consistent with eds_map.py ----
from eds_background import LINE_ENERGIES_KEV   # canonical table (see eds_background.py)

# periodic table for Z->symbol (for SPX parsing)
_PT = [None,
 'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
 'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
 'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
 'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
 'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
 'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
 'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'
]


# -------------------------
# Robust image read (imageio with PIL fallback)
# -------------------------
def _imread_robust(path: str) -> np.ndarray:
    # imageio's pillow plugin can silently fail to register when imageio and
    # Pillow versions don't line up (seen with imageio 2.37 + Pillow 12) — in
    # that state imread() raises "Could not find a backend" even though Pillow
    # itself reads the file fine. Fall back to PIL directly.
    try:
        import imageio
        return imageio.v2.imread(path)
    except Exception:
        with Image.open(path) as im:
            return np.asarray(im)


# -------------------------
# Multiscale pyramid (avoid GL texture clamp; full-res retained)
# -------------------------
def _to_gray(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).astype(a.dtype)
    return a

def _build_pyramid(img: np.ndarray, max_tex: int = 16384) -> List[np.ndarray]:
    levels = [img]
    h, w = img.shape[:2]
    while h > max_tex or w > max_tex:
        h2, w2 = max(1, h // 2), max(1, w // 2)
        im = Image.fromarray(levels[-1])
        levels.append(np.array(im.resize((w2, h2), resample=Image.BILINEAR)))
        h, w = h2, w2
    return levels

# -------------------------
# Preview auto-loading
# -------------------------
def _find_preview_dir(base_dir: Path, force_ds: int = 0) -> tuple[Optional[Path], int]:
    candidates: List[tuple[int, Path]] = []
    for p in sorted(base_dir.glob("eds_preview_ds*")):
        if not p.is_dir():
            continue
        try:
            ds = int(p.name.split("eds_preview_ds", 1)[1])
            candidates.append((ds, p))
        except Exception:
            pass
    if not candidates:
        return None, 0
    candidates.sort(key=lambda t: t[0])
    if force_ds > 0:
        for ds, d in candidates:
            if ds == force_ds:
                return d, ds
        return None, 0
    return candidates[0][1], candidates[0][0]

def _auto_load_previews(viewer: napari.Viewer, base_dir: Path, requested_ds: int = 0) -> tuple[List[str], int]:
    from imageio import v2 as iio
    prev_dir, ds = _find_preview_dir(base_dir, requested_ds)
    if not prev_dir:
        print("[viewer] No eds_preview_ds*/ found; skipping preview auto-load.")
        return [], 0

    loaded: List[str] = []
    for png in sorted(prev_dir.glob("*.png")):
        el = png.stem
        try:
            arr = iio.imread(str(png))
            arr = _to_gray(arr)
            name = f"{el} preview"
            if name in viewer.layers:
                lyr = viewer.layers[name]
                lyr.data = arr
                lyr.scale = (ds, ds)
                lyr.blending = 'opaque'
                lyr.visible = False
            else:
                viewer.add_image(
                    arr, name=name, colormap='inferno',
                    blending='opaque', scale=(ds, ds), visible=False
                )
            loaded.append(el)
        except Exception as e:
            print(f"[viewer] Failed to load preview {png.name}: {e}")

    if loaded:
        print(f"[viewer] Loaded previews from {prev_dir.name} (ds={ds}): {loaded}")
    return loaded, ds

# -------------------------
# ROI helpers (rect or polygon)
# -------------------------
def _rasterize_polygon_mask(verts_yx: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    H = max(0, y1 - y0)
    W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        return np.zeros((0, 0), dtype=bool)
    poly_xy = np.stack([verts_yx[:, 1] - x0, verts_yx[:, 0] - y0], axis=1)
    path = MplPath(poly_xy)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    mask = path.contains_points(pts).reshape(H, W)
    return mask

def _rgba_float_from_gray_and_mask(gray01: np.ndarray, mask_bool: np.ndarray, cmap_name: str = 'inferno') -> np.ndarray:
    """Colormapped grayscale with transparency mask -> uint8 RGBA.

    Using uint8 instead of float32 cuts GPU memory by 4x per layer.
    """
    g = np.clip(gray01.astype(np.float32), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = (cmap(g) * 255).astype(np.uint8)  # (H, W, 4) uint8
    rgba[..., 3] = np.where(mask_bool, rgba[..., 3], 0)
    return rgba


# -------------------------
# ROI EDS-map memory estimation (guards against giant-ROI MemoryError)
# -------------------------

def _available_memory_bytes() -> Optional[int]:
    """Best-effort available system RAM in bytes via psutil, else None."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _roi_map_bytes(h: int, w: int, n_elements: int, *,
                   n_bands: int = 3, transient_bands: int = 14,
                   safety: float = 1.15, dtype_size: int = 4) -> float:
    """Estimated peak RAM (bytes) to compute + display an ROI's EDS band maps.

    Two parts, both in units of one float32 "band" (``h × w × dtype_size``):

    * *Persistent* — ``n_bands × n_elements`` (each element keeps main, sb_lo,
      sb_hi for the lifetime of the result).
    * *Transient* — ``transient_bands`` band-sizes for the per-element display
      spike that is allocated then freed each loop iteration: the net map, the
      normalised ``disp`` + its finite-value copy, the **float64 colormap
      lookup (~8 bands)**, and the uint8 RGBA layer. This is a per-element
      overhead, so it's modelled as a fixed term rather than scaling with
      ``n_elements`` — which is what made the old flat ``2 × persistent`` model
      far too conservative for many-element (e.g. Oxford 21-element) ROIs.

    ``safety`` (>1) is a small pad for allocator fragmentation / overhead.
    """
    band = float(h) * float(w) * dtype_size
    return float(safety) * (n_bands * n_elements + transient_bands) * band


def _downsample_to_fit(h: int, w: int, n_elements: int,
                       budget_bytes: Optional[float], *,
                       n_bands: int = 3, transient_bands: int = 14,
                       safety: float = 1.15, max_ds: int = 64) -> int:
    """Smallest integer decimation so an ROI's EDS maps fit ``budget_bytes``.

    Returns 1 when it already fits (or no budget is known). The output grid
    after decimation is ``ceil(h/ds) × ceil(w/ds)``.
    """
    if not budget_bytes or budget_bytes <= 0:
        return 1
    ds = 1
    while ds <= max_ds:
        ho = -(-int(h) // ds)   # ceil division
        wo = -(-int(w) // ds)
        if _roi_map_bytes(ho, wo, n_elements, n_bands=n_bands,
                          transient_bands=transient_bands,
                          safety=safety) <= budget_bytes:
            return ds
        ds += 1
    return max_ds

# -------------------------
# Manifest bbox helpers for SPX-ROI composition
# -------------------------
def _row_bbox_on_canvas(row: dict) -> Optional[Tuple[int,int,int,int]]:
    """Tile's bbox on the stitched mosaic canvas, from its manifest row.

    Manifest ``y``/``x`` are the tile's solved position in mosaic (BSE) pixels
    and ``bse_h``/``bse_w`` its composited size, so the canvas footprint is
    simply their sum. ``scale_y``/``scale_x`` are the EDS-grid-to-BSE size
    ratios (e.g. a 768x768 EDS grid on a 1024x768 tile -> scale_x = 0.75) and
    ``shx``/``shy`` the compositing shear coefficients; neither is a canvas
    transform. An earlier version multiplied position and size by the scales
    (a no-op on scale=1 manifests), which misplaced every tile on manifests
    whose EDS grid differs from the BSE grid - silently corrupting the
    SPX<->tile matching and the mosaic calibration (r2 < 0 for every element).
    """
    try:
        y = float(row.get("y", 0.0))
        x = float(row.get("x", 0.0))
        h = float(row.get("bse_h", row.get("eds_h", 0)))
        w = float(row.get("bse_w", row.get("eds_w", 0)))
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        y1 = int(np.ceil(y + h))
        x1 = int(np.ceil(x + w))
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)
    except Exception:
        return None

def _overlap_area(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> int:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    if iy1 <= iy0 or ix1 <= ix0:
        return 0
    return (iy1 - iy0) * (ix1 - ix0)

_COORD_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:\[\s*(-?\d+)\s*\])?\)")
def _coords_from_name(name: str) -> Optional[Tuple[int,int,Optional[int]]]:
    last = None
    for m in _COORD_RE.finditer(name):
        last = m
    if not last:
        return None
    i = int(last.group(1)); j = int(last.group(2)); rep = last.group(3)
    return (i, j, int(rep) if rep is not None else None)

def _match_spx_for_bcf(bcf_path: Path) -> Optional[Path]:
    key = _coords_from_name(bcf_path.name)
    fname = f"Spectrum_({key[0]},{key[1]}).spx" if key else None

    # Collect all candidate SPX files, sorting so subdirectory entries come before
    # same-level ones - this naturally prefers quantified subfolders over raw exports.
    candidates = sorted(
        bcf_path.parent.rglob("Spectrum_(*,*).spx"),
        key=lambda p: (len(p.parts), p.name),  # deeper path first, then name
    )
    # Flip: shorter path = same dir (less preferred), deeper = subdir (more preferred)
    candidates.sort(key=lambda p: (-len(p.parts), p.name))

    for spx in candidates:
        if key:
            k2 = _coords_from_name(spx.name)
            if k2 and k2[:2] == key[:2]:
                return spx
    return None

def _fnum(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", ".").replace("%", "")
    try:
        return float(s)
    except Exception:
        return None

def _parse_spx_quant(spx_path: Path) -> Tuple[Dict[str,float], float, List[str]]:
    import xml.etree.ElementTree as ET
    notes: List[str] = []
    try:
        root = ET.parse(spx_path).getroot()
    except Exception:
        notes.append("xml-parse-failed")
        return {}, 0.0, notes

    results = list(root.iter("Result"))
    if not results:
        notes.append("no-Result-nodes")
        return {}, 0.0, notes

    per_el: Dict[str, float] = {}
    sum_net = 0.0
    any_wt = False
    for r in results:
        z_txt = (r.findtext("Atom") or "").strip()
        try:
            Z = int(z_txt)
        except Exception:
            Z = None
        sym = _PT[Z] if (Z is not None and 0 < Z < len(_PT)) else None

        wt = _fnum(r.findtext("MassPercent"))
        net = _fnum(r.findtext("NetIntens"))

        if wt is not None and sym:
            per_el[sym] = per_el.get(sym, 0.0) + wt
            any_wt = True
        if net is not None and net > 0:
            sum_net += net

    if not any_wt:
        notes.append("no-quant-fields")
        return {}, 0.0, notes

    s = sum(per_el.values())
    if s > 0:
        per_el = {k: 100.0 * v / s for k, v in per_el.items()}
    if sum_net <= 0:
        sum_net = 1.0
        notes.append("no-netintens-weight=1")
    return per_el, float(sum_net), notes

# -------------------------
# EMSA/MAS .MSA writer
# -------------------------
def _write_emsa_msa(out_path: Path, counts: np.ndarray, e0_keV: float, dE_keV: float) -> None:
    """
    Write an EMSA/MAS compatible .msa with Y data and calibration.
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    n = counts.size
    now = datetime.now()
    lines = []
    lines.append("#FORMAT      : EMSA/MAS Spectral Data File")
    lines.append("#VERSION     : 1.0")
    lines.append("#TITLE       : ROI summed spectrum")
    lines.append(f"#DATE        : {now.strftime('%Y-%m-%d')}")
    lines.append(f"#TIME        : {now.strftime('%H:%M:%S')}")
    lines.append("#OWNER       : ")
    lines.append(f"#NPOINTS     : {n}")
    lines.append("#NCOLUMNS    : 1")
    lines.append("#XUNITS      : keV")
    lines.append("#YUNITS      : counts")
    lines.append("#DATATYPE    : Y")
    lines.append("#SIGNALTYPE  : EDS")
    lines.append(f"#OFFSET      : {e0_keV:.9g}")
    lines.append(f"#XPERCHAN    : {dE_keV:.9g}")
    lines.append("#SPECTRUM")
    body = "\n".join(f"{float(v):.6g}" for v in counts)
    lines.append(body)
    lines.append("#ENDOFDATA\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")

# -------------------------
# Perple_X Helpers
#--------------------------
def _write_perplex_build_keyword(
    out_path: Path,
    oxides_wt: Dict[str, float],
    title: str = "ROI composition",
    thermo_db: str = "stx21ver.dat",          # adjust to your local DB (e.g., hp02ver.dat, stx21ver.dat)
    solution_db: str = "solution_model.dat",  # adjust to your local solution model file
    activity_db: str = "none",                # often "none" unless you use one
    calc_mode: str = "td"                     # typical: "td" (T-P section), "tc", "ps", etc.
) -> None:
    """
    Minimal Perple_X BUILD keyword file (plain text).
    You may need to tweak database filenames to match your installation.
    """
    # canonical Perple_X oxide labels - order helps readability (extras go after)
    order = ["SiO2","Al2O3","FeO","Fe2O3","MgO","CaO","Na2O","K2O","TiO2","MnO","P2O5","H2O","CO2"]
    seen = set()
    comps = []
    for k in order:
        if k in oxides_wt:
            comps.append((k, oxides_wt[k]))
            seen.add(k)
    for k, v in sorted(oxides_wt.items()):
        if k not in seen:
            comps.append((k, v))

    # build components list line
    components_list = " ".join(k for k, _ in comps)

    lines = []
    lines.append("# Perple_X BUILD keyword file (generated)")
    lines.append(f"title: {title}")
    lines.append(f"calculation_mode: {calc_mode}")
    lines.append(f"thermodynamic_data_file: {thermo_db}")
    lines.append(f"solution_model_database: {solution_db}")
    lines.append(f"activity_model_database: {activity_db}")
    lines.append("composition_units: wt%")
    lines.append(f"components: {components_list}")
    lines.append("bulk_composition:")
    for k, v in comps:
        lines.append(f"  {k} {v:.4f}")
    lines.append("end")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
# Element->Oxide conversion for Perple_X (wt%)
# -------------------------
# Cation -> (oxide formula, wt-oxide per wt-element). Shared by
# elements_to_oxides_wt and add_stoichiometric_oxygen. Fe is split by valence
# separately (FeO 1.2863 / Fe2O3 1.4297) since its factor depends on Fe3+/ΣFe.
_OXIDE_FACTORS = {  # wt oxide per wt element
    "Si": ("SiO2", 2.1393),
    "Al": ("Al2O3", 1.8895),
    "Mg": ("MgO", 1.6583),
    "Ca": ("CaO", 1.3992),
    "Na": ("Na2O", 1.3480),
    "K":  ("K2O", 1.2046),
    "Ti": ("TiO2", 1.6681),
    "Mn": ("MnO", 1.2912),
    "P":  ("P2O5", 2.2914),
}


def add_stoichiometric_oxygen(elem_wt: Dict[str, float],
                              fe3_over_fe_total: float = 0.2) -> Dict[str, float]:
    """Augment cation wt% with the oxygen carried by their standard oxides.

    The input is element (cation) wt% without oxygen — e.g. an SPX-calibrated
    EDS composition. Each cation's oxide carries a fixed O mass (``wt·(factor−1)``
    using the same ``_OXIDE_FACTORS`` as :func:`elements_to_oxides_wt`, with Fe
    split by valence), so this reconstructs the stoichiometric O exactly the way
    SPX itself derives "O by stoichiometry" — making a cation-only composition
    directly comparable to one that already reports O. Any pre-existing ``O`` in
    the input is replaced; non-oxide elements (e.g. ``C``) are kept as-is and
    contribute no oxygen. The result is renormalized to 100.
    """
    out = {el: float(v) for el, v in elem_wt.items() if el != "O" and float(v) > 0}
    o_mass = 0.0
    Fe = float(out.get("Fe", 0.0))
    if Fe > 0:
        fe3 = Fe * float(fe3_over_fe_total)
        fe2 = Fe - fe3
        o_mass += fe2 * (1.2863 - 1.0) + fe3 * (1.4297 - 1.0)
    for el, (_ox, fac) in _OXIDE_FACTORS.items():
        v = float(out.get(el, 0.0))
        if v > 0:
            o_mass += v * (fac - 1.0)
    if o_mass > 0:
        out["O"] = o_mass
    s = sum(out.values())
    if s > 0:
        out = {k: 100.0 * v / s for k, v in out.items()}
    return out


def elements_to_oxides_wt(elem_wt: Dict[str, float],
                          fe3_over_fe_total: float = 0.2,
                          add_h2o: float = 0.0,
                          add_co2: float = 0.0) -> Dict[str, float]:
    """
    Convert element wt% to oxide wt%; returns dict normalized to 100 wt% (including H2O/CO2).
    """
    f = _OXIDE_FACTORS
    out: Dict[str, float] = {}

    # iron split
    Fe = float(elem_wt.get("Fe", 0.0))
    if Fe > 0:
        fe3 = Fe * float(fe3_over_fe_total)
        fe2 = Fe - fe3
        if fe2 > 0:
            out["FeO"] = out.get("FeO", 0.0) + fe2 * 1.2863
        if fe3 > 0:
            out["Fe2O3"] = out.get("Fe2O3", 0.0) + fe3 * 1.4297

    for el, (ox, fac) in f.items():
        v = float(elem_wt.get(el, 0.0))
        if v > 0:
            out[ox] = out.get(ox, 0.0) + v * fac

    if add_h2o > 0:
        out["H2O"] = out.get("H2O", 0.0) + float(add_h2o)
    if add_co2 > 0:
        out["CO2"] = out.get("CO2", 0.0) + float(add_co2)

    s = sum(out.values())
    if s > 0:
        out = {k: 100.0 * v / s for k, v in out.items()}
    return out
