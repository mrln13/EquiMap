# eds_map.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Optional

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import hyperspy.api as hs
from tqdm import tqdm


# ---------- EDS discovery ----------

def load_eds_signal_lazy(p: Path):
    """Return a lazy HyperSpy EDS signal for a given BCF path, or None if not found."""
    try:
        sigs = hs.load(p, lazy=True)
    except Exception:
        return None

    it = sigs if isinstance(sigs, (list, tuple)) else [sigs]
    for s in it:
        try:
            st = str(s.metadata.Signal.signal_type)
        except Exception:
            st = ""
        if "EDS" in st.upper():
            return s
    return None


# ---------- Metadata helpers ----------

def extract_eds_metadata_from_signal(sig) -> dict:
    """
    Extract beam energy, sample elements/lines, and energy axis from a HyperSpy EDS signal.
    Returns:
      {
        'beam_energy_keV': float|None,
        'elements': [str,...],
        'xray_lines': [str,...],  # e.g. 'Si_Ka'
        'E0': float|None, 'dE': float|None, 'nE': int|None, 'units': str
      }
    """
    out = {
        'beam_energy_keV': None,
        'elements': [],
        'xray_lines': [],
        'E0': None, 'dE': None, 'nE': None, 'units': 'keV',
    }

    md = getattr(sig, 'metadata', None)
    if md is not None:
        # beam energy
        try:
            out['beam_energy_keV'] = float(md.Acquisition_instrument.SEM.beam_energy)
        except Exception:
            try:
                out['beam_energy_keV'] = float(md.get_item('Acquisition_instrument.SEM.beam_energy'))
            except Exception:
                pass
        # sample lists
        try:
            out['elements'] = [str(e) for e in list(md.Sample.elements)]
        except Exception:
            pass
        try:
            out['xray_lines'] = [str(x) for x in list(md.Sample.xray_lines)]
        except Exception:
            pass

    # energy axis
    try:
        ea = sig.axes_manager.signal_axes[0]
        out['E0'] = float(ea.offset)
        out['dE'] = float(ea.scale)
        out['nE'] = int(ea.size)
        out['units'] = str(ea.units) or 'keV'
    except Exception:
        pass
    return out


# ---------- Manifest I/O ----------

def write_tile_manifest(
    base_dir: Path,
    coords_sorted: Iterable[Tuple[int, int]],
    pos: Dict[Tuple[int, int], np.ndarray],
    rc_to_path: Dict[Tuple[int, int], Path],
    tile_shear: Dict[Tuple[int,int], Dict[str, float]],
    th: int,
    tw: int,
) -> Path:
    """
    Write a lightweight manifest mapping mosaic coords → BCF files and EDS metadata.

    Columns:
      row,col,y,x,file,bse_h,bse_w,eds_h,eds_w,scale_y,scale_x,shx,shy,
      E0,dE,nE,energy_units,beam_energy,elements,xray_lines
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    out_csv = base_dir / "stitch_manifest.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "row","col","y","x","file",
            "bse_h","bse_w","eds_h","eds_w",
            "scale_y","scale_x","shx","shy",
            "E0","dE","nE","energy_units",
            "beam_energy","elements","xray_lines",
        ])

        for rc in coords_sorted:
            p = rc_to_path[rc]
            y, x = pos[rc]
            shx = float(tile_shear.get(rc, {}).get("shx", 0.0))
            shy = float(tile_shear.get(rc, {}).get("shy", 0.0))

            eds_h = eds_w = th
            E0 = 0.0; dE = 1.0; nE = 0; units = "keV"
            beam_energy = ""
            elements_csv = ""
            lines_csv = ""

            eds = load_eds_signal_lazy(p)
            if eds is not None:
                # nav shape
                try:
                    nav = eds.axes_manager.navigation_shape
                    if len(nav) >= 2:
                        eds_h, eds_w = int(nav[0]), int(nav[1])
                except Exception:
                    if hasattr(eds, "data") and hasattr(eds.data, "shape") and len(eds.data.shape) >= 2:
                        eds_h, eds_w = int(eds.data.shape[0]), int(eds.data.shape[1])

                meta = extract_eds_metadata_from_signal(eds)
                E0 = meta.get('E0', E0); dE = meta.get('dE', dE); nE = meta.get('nE', nE); units = meta.get('units', units)
                be = meta.get('beam_energy_keV', None)
                if be is not None:
                    beam_energy = f"{be:.6g}"
                els = meta.get('elements') or []
                lns = meta.get('xray_lines') or []
                if els: elements_csv = ';'.join(els)
                if lns: lines_csv = ';'.join(lns)

            # EDS vs BSE scale
            scale_y = eds_h / float(th) if th > 0 else 1.0
            scale_x = eds_w / float(tw) if tw > 0 else 1.0

            w.writerow([
                rc[0], rc[1], y, x, str(p),
                th, tw, eds_h, eds_w,
                scale_y, scale_x, shx, shy,
                E0, dE, nE, units,
                beam_energy, elements_csv, lines_csv
            ])

    print(f"[manifest] Wrote tile manifest → {out_csv}")
    return out_csv


def read_manifest(manifest_path: Path):
    rows = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # numeric casts (tolerant)
            for k in ["row","col","bse_h","bse_w","eds_h","eds_w","nE"]:
                try: row[k] = int(float(row.get(k, "0") or 0))
                except Exception: row[k] = 0
            for k in ["y","x","scale_y","scale_x","shx","shy","E0","dE","beam_energy"]:
                try: row[k] = float(row.get(k, "0") or 0.0)
                except Exception: row[k] = 0.0
            row["file"] = Path(row["file"])
            row["elements_list"] = [s for s in (row.get("elements","") or "").split(";") if s]
            row["xray_lines_list"] = [s for s in (row.get("xray_lines","") or "").split(";") if s]
            rows.append(row)
    return rows


# ---------- EDS ROI → cube ----------

def roi_to_cube(
    manifest_path: Path,
    roi: tuple[int, int, int, int],
    apply_subpixel: bool = True,
    feather: bool = True,
    blend_mode: str = "mean",      # "mean" or "sum"
    tukey_alpha: float = 0.08,
    progress: bool = True,         # <<< NEW: show a progress bar
):
    """
    Assemble an EDS cube (H, W, nE) for ROI in BSE pixel coords.
    Alignment matches the BSE composite:
      1) resample EDS nav grid -> BSE tile size (scale_y/x),
      2) per-tile shear (shx/shy),
      3) sub-pixel shift (fractional y,x).

    Robust path resolution:
      - Each row['file'] is resolved relative to the manifest's folder,
        with a filename-only fallback.

    Progress:
      - If progress=True, shows a tqdm bar over only the tiles intersecting the ROI.
    """
    import hyperspy.api as hs
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from skimage.transform import resize

    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent.resolve()

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty.")

    # --- helper: resolve a file path from manifest robustly
    def _resolve(p):
        p = Path(p)
        if p.is_absolute() and p.exists():
            return p
        cand = (base_dir / p).resolve()
        if cand.exists():
            return cand
        cand2 = (base_dir / p.name).resolve()
        if cand2.exists():
            return cand2
        return None

    # --- ROI dims
    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0); W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    # --- Try to find a common energy axis once (scan all rows, but quick)
    E0 = dE = None; nE = None; units = "keV"
    for r in rows:
        p = _resolve(r["file"])
        if p is None:
            continue
        sig = load_eds_signal_lazy(p)
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if meta.get("nE", 0) > 0:
            E0, dE, nE = meta["E0"], meta["dE"], meta["nE"]
            units = meta.get("units", "keV") or "keV"
            break

    if nE is None or nE <= 0:
        raise RuntimeError(
            "No EDS data found in manifest (no tiles with a valid energy axis could be opened). "
            "If files were moved, ensure paths in stitch_manifest.csv can be resolved from the manifest folder, "
            "or rebuild the manifest with current locations."
        )

    # --- Prefilter: only process tiles whose BSE bbox intersects the ROI
    try:
        roi_rows = _prefilter_rows_for_roi(rows, roi)  # defined elsewhere in eds_map.py
    except NameError:
        # Fallback: cheap inline AABB test if helper isn't available
        roi_rows = []
        for r in rows:
            th = int(r.get("bse_h", 0)); tw = int(r.get("bse_w", 0))
            iy = int(np.floor(float(r.get("y", 0.0)))); ix = int(np.floor(float(r.get("x", 0.0))))
            if (iy + th + 1) <= y0 or iy >= (y1 + 1):
                continue
            if (ix + tw + 1) <= x0 or ix >= (x1 + 1):
                continue
            roi_rows.append(r)

    # --- Accumulators
    accum  = np.zeros((H, W, nE), dtype=np.float32)
    weight = np.zeros((H, W),      dtype=np.float32)

    def tukey_window(h, w, a):
        wy = tukey1d(h, alpha=a).astype(np.float32)
        wx = tukey1d(w, alpha=a).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    # --- Progress bar setup (only tiles we actually process)
    iterator = roi_rows
    pbar = None
    if progress:
        try:
            from tqdm import tqdm  # lazy import
            pbar = tqdm(total=len(roi_rows), desc="ROI EDS cube", unit="tile")
        except Exception:
            pbar = None  # silently disable if tqdm not available

    used_tiles = 0
    try:
        for r in iterator:
            p = _resolve(r["file"])
            if p is None:
                if pbar: pbar.update(1)
                continue

            sig = load_eds_signal_lazy(p)
            if sig is None:
                if pbar: pbar.update(1)
                continue

            meta = extract_eds_metadata_from_signal(sig)
            # require same energy axis as the one we locked above
            if not (meta.get("nE", 0) == nE and
                    abs((meta.get("E0", 0) or 0) - E0) < 1e-6 and
                    abs((meta.get("dE", 0) or 0) - dE) < 1e-9):
                if pbar: pbar.update(1)
                continue

            # Data as (h,w,nE)
            arr = np.asarray(sig.data)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.ndim == 3 and arr.shape[0] == nE:
                arr = np.moveaxis(arr, 0, 2)
            if arr.ndim != 3 or arr.shape[2] != nE:
                if pbar: pbar.update(1)
                continue

            th = int(r["bse_h"]); tw = int(r["bse_w"])
            sy = float(r.get("scale_y", 1.0) or 1.0)
            sx = float(r.get("scale_x", 1.0) or 1.0)
            shx = float(r.get("shx", 0.0) or 0.0)
            shy = float(r.get("shy", 0.0) or 0.0)
            ty  = float(r["y"]);       tx  = float(r["x"])

            # (1) Resample EDS nav -> BSE tile size
            tgt_h = max(1, int(round(th * sy)))
            tgt_w = max(1, int(round(tw * sx)))
            if arr.shape[0] != tgt_h or arr.shape[1] != tgt_w:
                tmp = np.empty((tgt_h, tgt_w, nE), dtype=np.float32)
                for k in range(nE):
                    tmp[..., k] = resize(
                        arr[..., k], (tgt_h, tgt_w), order=1,
                        preserve_range=True, anti_aliasing=False
                    )
                arr = tmp
            if arr.shape[0] != th or arr.shape[1] != tw:
                tmp = np.empty((th, tw, nE), dtype=np.float32)
                for k in range(nE):
                    tmp[..., k] = resize(
                        arr[..., k], (th, tw), order=1,
                        preserve_range=True, anti_aliasing=False
                    )
                arr = tmp

            # (2) Apply tiny per-tile shear
            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                for k in range(nE):
                    arr[..., k] = affine_transform(
                        arr[..., k], Minv, offset=0.0, order=1,
                        mode="nearest", cval=0.0, prefilter=False
                    ).astype(np.float32)

            # (3) Sub-pixel shift
            if apply_subpixel:
                from scipy.ndimage import fourier_shift
                fy = ty - np.floor(ty); fx = tx - np.floor(tx)
                if abs(fy) > 1e-6 or abs(fx) > 1e-6:
                    for k in range(nE):
                        arr[..., k] = np.fft.ifftn(
                            fourier_shift(np.fft.fftn(arr[..., k]), (fy, fx))
                        ).real.astype(np.float32)

            # Intersect with ROI and accumulate
            iy = int(np.floor(ty)); ix = int(np.floor(tx))
            yA0 = max(0, y0 - iy); yA1 = min(th, y1 - iy)
            xA0 = max(0, x0 - ix); xA1 = min(tw, x1 - ix)
            if yA1 <= yA0 or xA1 <= xA0:
                if pbar: pbar.update(1)
                continue

            yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
            xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

            if feather:
                ww_full = tukey_window(th, tw, tukey_alpha)
                ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)
            else:
                ww = np.ones((yA1 - yA0, xA1 - xA0), dtype=np.float32)

            patch = arr[yA0:yA1, xA0:xA1, :]
            for k in range(nE):
                accum[yB0:yB1, xB0:xB1, k] += patch[..., k] * ww
            weight[yB0:yB1, xB0:xB1] += ww
            used_tiles += 1

            if pbar: pbar.update(1)
    finally:
        if pbar:
            pbar.close()

    if used_tiles == 0:
        raise RuntimeError(
            "No EDS tiles intersected the ROI (or none could be opened). "
            "Verify the file paths in the manifest and that your BCFs contain EDS signals."
        )

    # Mean normalization if requested
    if blend_mode != "sum":
        m = weight > 0
        if m.any():
            invw = np.zeros_like(weight, dtype=np.float32)
            invw[m] = 1.0 / weight[m]
            for k in range(nE):
                plane = accum[..., k]
                plane[m] = plane[m] * invw[m]
                accum[..., k] = plane

    return accum, (E0, dE, nE, units)



def extract_eds_roi(
    manifest_path: Path,
    roi: tuple[int, int, int, int],
    out_path: Path,
    apply_subpixel: bool = False,
    feather: bool = True,
    blend_mode: str = "mean",
):
    """Assemble an EDS cube for a mosaic ROI and save it as a HyperSpy .hspy file.
    Thin wrapper over `roi_to_cube` kept for backward-compatibility with stitch.py.
    """
    cube, (E0, dE, nE, units) = roi_to_cube(
        manifest_path,
        roi,
        apply_subpixel=apply_subpixel,
        feather=feather,
        blend_mode=blend_mode,
    )
    import hyperspy.api as hs
    sig = hs.signals.EDSSEMSpectrum(cube)
    sig.axes_manager.navigation_axes[0].name = 'y'
    sig.axes_manager.navigation_axes[1].name = 'x'
    sig.axes_manager.signal_axes[0].name = 'Energy'
    sig.axes_manager.signal_axes[0].units = units
    sig.axes_manager.signal_axes[0].offset = E0
    sig.axes_manager.signal_axes[0].scale = dE
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig.save(out_path)
    print(f"[eds] ROI saved → {out_path}")


# One canonical line table for the whole toolkit (eds_background.py).
from eds_background import LINE_ENERGIES_KEV, sideband_centers as _sideband_centers, net_count_maps as _net_count_maps  # noqa: E402

def _band_indices(E0: float, dE: float, nE: int, E_center_keV: float, width_eV: float) -> Tuple[int,int]:
    half = 0.5*width_eV/1000.0
    lo = max(0, int(np.floor((E_center_keV - half - E0)/dE)))
    hi = min(nE, int(np.ceil((E_center_keV + half - E0)/dE)))
    if hi <= lo:
        hi = min(nE, lo+1)
    return lo, hi



def _prefilter_rows_for_roi(rows: List[dict], roi: Tuple[int,int,int,int]) -> List[dict]:
    """Return only manifest rows that intersect the ROI (fast, no I/O)."""
    y0, y1, x0, x1 = map(int, roi)
    kept = []
    for r in rows:
        th = int(r["bse_h"]); tw = int(r["bse_w"])
        iy = int(np.floor(float(r["y"]))); ix = int(np.floor(float(r["x"])))
        # AABB intersection (with 1 px slack)
        if (iy + th + 1) <= y0 or iy >= (y1 + 1):
            continue
        if (ix + tw + 1) <= x0 or ix >= (x1 + 1):
            continue
        kept.append(r)
    return kept


def roi_to_element_maps(
    manifest_path: Path,
    roi: Tuple[int,int,int,int],
    elements: List[str],
    width_eV: float = 150.0,
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    blend_mode: str = "mean",   # "mean" or "sum"
    progress: bool = True,
    roi_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Memory-lean: returns ONLY per-element 2D maps for the ROI (no HxWxnE cube).
    Applies scale → shear → sub-pixel (same as BSE compositor).
    """
    import hyperspy.api as hs
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from skimage.transform import resize
    from tqdm import tqdm

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty; run stitching first.")

    base_path = Path(manifest_path).parent

    # Determine common energy axis once
    E0 = dE = None; nE = None; units = "keV"
    for r in rows:
        f_path = Path.joinpath(base_path, r["file"])
        sig = load_eds_signal_lazy(f_path)
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if meta.get("nE", 0) > 0:
            E0, dE, nE = meta["E0"], meta["dE"], meta["nE"]
            units = meta.get("energy_units", "keV") or "keV"
            break
    if nE is None:
        raise RuntimeError("No EDS signals found in manifest.")

    # ROI dims
    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0); W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    # Elements + band indices
    elements = [e for e in elements if e in LINE_ENERGIES_KEV]
    if not elements:
        raise ValueError("No valid elements requested.")
    band_idx = {e: _band_indices(E0, dE, nE, LINE_ENERGIES_KEV[e], width_eV) for e in elements}

    # >>> pre-filter intersecting tiles (no I/O)
    roi_rows = _prefilter_rows_for_roi(rows, roi)

    # Accumulators
    accum = {e: np.zeros((H, W), dtype=np.float32) for e in elements}
    weight = np.zeros((H, W), dtype=np.float32)

    def tukey_window(h, w, a):
        wy = tukey1d(h, alpha=a).astype(np.float32)
        wx = tukey1d(w, alpha=a).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    iterator = tqdm(roi_rows, desc="ROI EDS (elements)", unit="tile", total=len(roi_rows)) if progress else roi_rows
    for r in iterator:
        f_path = Path.joinpath(base_path, r["file"])
        sig = load_eds_signal_lazy(f_path)
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if not (meta.get("nE",0) == nE and abs(meta.get("E0",0)-E0) < 1e-6 and abs(meta.get("dE",0)-dE) < 1e-9):
            continue

        th = int(r["bse_h"]); tw = int(r["bse_w"])
        sy = float(r.get("scale_y", 1.0) or 1.0)
        sx = float(r.get("scale_x", 1.0) or 1.0)
        shx = float(r.get("shx", 0.0) or 0.0)
        shy = float(r.get("shy", 0.0) or 0.0)
        ty  = float(r["y"]);       tx  = float(r["x"])

        iy = int(np.floor(ty)); ix = int(np.floor(tx))
        yA0 = max(0, y0 - iy); yA1 = min(th, y1 - iy)
        xA0 = max(0, x0 - ix); xA1 = min(tw, x1 - ix)
        if yA1 <= yA0 or xA1 <= xA0:
            continue

        yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
        xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

        ww_full = tukey_window(th, tw, tukey_alpha) if feather else np.ones((th, tw), np.float32)
        ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)

        fy = ty - np.floor(ty); fx = tx - np.floor(tx)
        use_subpx = apply_subpixel and (abs(fy) > 1e-6 or abs(fx) > 1e-6)
        if use_subpx:
            from scipy.ndimage import fourier_shift

        for el in elements:
            lo, hi = band_idx[el]
            try:
                img = sig.isig[lo:hi].sum(axis='Energy').data
                img = np.asarray(img, dtype=np.float32)
            except Exception:
                continue
            if img.ndim != 2:
                continue

            # scale → shear → subpixel
            tgt_h = max(1, int(round(th*sy))); tgt_w = max(1, int(round(tw*sx)))
            if img.shape != (tgt_h, tgt_w):
                img = resize(img, (tgt_h, tgt_w), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
            if img.shape != (th, tw):
                img = resize(img, (th, tw), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)

            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                img = affine_transform(img, Minv, offset=0.0, order=1, mode='nearest', cval=0.0, prefilter=False).astype(np.float32)

            if use_subpx:
                img = np.fft.ifftn(fourier_shift(np.fft.fftn(img), (fy, fx))).real.astype(np.float32)

            patch = img[yA0:yA1, xA0:xA1]

            if roi_mask is not None:
                mask_sub = roi_mask[yB0:yB1, xB0:xB1]
                if mask_sub.shape != ww.shape:
                    # Should not happen, but guard anyway
                    mh, mw = mask_sub.shape
                    ww = ww[:mh, :mw]
                    patch = patch[:mh, :mw]
                    yB1 = yB0 + mh
                    xB1 = xB0 + mw
                ww_eff = ww * mask_sub.astype(np.float32)
            else:
                ww_eff = ww

            accum[el][yB0:yB1, xB0:xB1] += patch * ww_eff

            # weight once per tile (shared)
            weight[yB0:yB1, xB0:xB1] += ww_eff

    if blend_mode != "sum":
        m = weight > 0
        if m.any():
            invw = np.zeros_like(weight, dtype=np.float32)
            invw[m] = 1.0 / weight[m]
            for el in elements:
                tmp = accum[el]
                tmp[m] = tmp[m] * invw[m]
                accum[el] = tmp

    return accum



def _tukey_window_2d(h: int, w: int, a: float) -> np.ndarray:
    """2-D separable Tukey (cosine-tapered) window."""
    from scipy.signal.windows import tukey as tukey1d
    wy = tukey1d(h, alpha=a).astype(np.float32)
    wx = tukey1d(w, alpha=a).astype(np.float32)
    return np.outer(wy, wx).astype(np.float32)


def _resolve_manifest_path(p_like, base_dir: Path) -> Path:
    """Robustly resolve a path from a manifest row to an existing file if possible."""
    try:
        p = Path(p_like)
    except Exception:
        p = Path(str(p_like))
    if not p.is_absolute():
        p = base_dir / p
    if not p.exists():
        cands = list(base_dir.rglob(p.name))
        if cands:
            return cands[0]
    return p


def _process_one_tile(r, base_dir, elements, bands, E0, dE, nE,
                      roi, feather, tukey_alpha, apply_subpixel,
                      roi_mask, cancel_flag, shift_method: str = "bilinear"):
    """
    Process a single manifest tile row for roi_to_banded_maps.

    Returns a result dict on success, or None if the tile should be skipped.
    Has no shared mutable state — safe to call from multiple threads.

    Result dict keys: yB0, yB1, xB0, xB1, y_end, x_end, ww, ww_eff,
                      has_mask, patches {el: {key: ndarray}}
    """
    from skimage.transform import resize
    from scipy.ndimage import affine_transform

    if cancel_flag is not None and cancel_flag():
        return None

    p = _resolve_manifest_path(r["file"], base_dir)
    sig = load_eds_signal_lazy(p)
    if sig is None:
        return None

    meta = extract_eds_metadata_from_signal(sig)
    if not (meta.get("nE", 0) == nE and
            abs(float(meta.get("E0", 0.0)) - E0) < 1e-6 and
            abs(float(meta.get("dE", 0.0)) - dE) < 1e-9):
        return None

    # geometry from manifest row
    th = int(r["bse_h"]); tw = int(r["bse_w"])
    sy = float(r.get("scale_y", 1.0) or 1.0)
    sx = float(r.get("scale_x", 1.0) or 1.0)
    shx = float(r.get("shx", 0.0) or 0.0)
    shy = float(r.get("shy", 0.0) or 0.0)
    ty = float(r["y"]); tx = float(r["x"])
    y0, y1, x0, x1 = roi

    iy = int(np.floor(ty)); ix = int(np.floor(tx))
    yA0 = max(0, y0 - iy); yA1 = min(th, y1 - iy)
    xA0 = max(0, x0 - ix); xA1 = min(tw, x1 - ix)
    if yA1 <= yA0 or xA1 <= xA0:
        return None

    yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
    xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

    # feathering window (tile-level)
    ww_full = _tukey_window_2d(th, tw, tukey_alpha) if feather else np.ones((th, tw), np.float32)
    ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)

    # subpixel shift params
    fy = ty - np.floor(ty); fx = tx - np.floor(tx)
    use_subpx = apply_subpixel and (abs(fy) > 1e-6 or abs(fx) > 1e-6)

    # compute ww_eff and accumulation bounds once (same for all elements in this tile)
    if roi_mask is not None:
        mask_sub = roi_mask[yB0:yB1, xB0:xB1]
        if mask_sub.shape != ww.shape:
            mh, mw = mask_sub.shape
            ww_use = ww[:mh, :mw]
            y_end = yB0 + mh
            x_end = xB0 + mw
            ww_eff = ww_use * mask_sub.astype(np.float32)
        else:
            ww_use = ww
            y_end, x_end = yB1, xB1
            ww_eff = ww_use * mask_sub.astype(np.float32)
    else:
        ww_eff = ww
        y_end, x_end = yB1, xB1

    # --- Batched energy-axis read: materialise the [min_lo, max_hi) slab once,
    # then sum per-band in numpy. Avoids 3*len(elements) separate dask dispatches
    # (each dispatch costs ~1 s on the Bruker reader). Falls back to the
    # per-band lazy path if the batched read fails.
    band_images: Dict[str, Dict[str, np.ndarray]] = {}
    try:
        all_ranges = [rng for e in elements for rng in bands[e].values()]
        if not all_ranges:
            return None
        min_lo = min(lo for lo, _ in all_ranges)
        max_hi = max(hi for _, hi in all_ranges)
        if max_hi <= min_lo:
            return None

        sub = np.asarray(sig.isig[min_lo:max_hi].data)
        sig_ax_idx = sig.axes_manager.signal_axes[0].index_in_array
        expected_n = max_hi - min_lo
        if sub.ndim < 2 or sub.shape[sig_ax_idx] != expected_n:
            raise ValueError(f"subcube shape {sub.shape} axis {sig_ax_idx} "
                             f"!= {expected_n}; falling back")

        for e in elements:
            band_images[e] = {}
            for key, (lo, hi) in bands[e].items():
                slc = [slice(None)] * sub.ndim
                slc[sig_ax_idx] = slice(lo - min_lo, hi - min_lo)
                img2d = sub[tuple(slc)].sum(axis=sig_ax_idx, dtype=np.float32)
                band_images[e][key] = img2d
        del sub
    except Exception:
        band_images = {}  # sentinel: fall back per-band below

    # --- Phase A: resize + shear per band (still fast; per-band is fine here).
    tgt_h = max(1, int(round(th * sy))); tgt_w = max(1, int(round(tw * sx)))
    shaped: Dict[str, Dict[str, np.ndarray]] = {}
    for e in elements:
        shaped[e] = {}
        for key, (lo, hi) in bands[e].items():
            img = band_images.get(e, {}).get(key)
            if img is None:
                # fallback: one lazy dispatch per band (original slow path)
                try:
                    img = sig.isig[lo:hi].sum(axis='Energy').data
                    img = np.asarray(img, dtype=np.float32)
                except Exception:
                    continue
            if img.ndim != 2:
                continue

            # (1) scale EDS nav grid → BSE tile size
            if img.shape != (tgt_h, tgt_w):
                img = resize(img, (tgt_h, tgt_w), order=1,
                             preserve_range=True, anti_aliasing=False).astype(np.float32)
            if img.shape != (th, tw):
                img = resize(img, (th, tw), order=1,
                             preserve_range=True, anti_aliasing=False).astype(np.float32)

            # (2) tiny per-tile shear
            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                img = affine_transform(
                    img, Minv, offset=0.0, order=1, mode='nearest',
                    cval=0.0, prefilter=False
                ).astype(np.float32)

            shaped[e][key] = img

    # --- Phase B: sub-pixel shift.
    # Bilinear (default): per-slice, ~8x faster than FFT and visually
    # indistinguishable for smooth EDS counts.
    # FFT (sinc): exact for bandlimited signals. Matches how the offsets were
    # originally measured (phase-correlation). Kept as an option for fidelity.
    if use_subpx:
        method = (shift_method or "bilinear").lower()
        if method == "fft":
            from scipy.ndimage import fourier_shift
            key_order = [(e, k) for e in elements for k in bands[e] if k in shaped.get(e, {})]
            if key_order:
                stack = np.stack([shaped[e][k] for e, k in key_order], axis=0).astype(np.float32, copy=False)
                ft = np.fft.fftn(stack, axes=(-2, -1))
                ft = fourier_shift(ft, (0.0, fy, fx))
                stack = np.fft.ifftn(ft, axes=(-2, -1)).real.astype(np.float32, copy=False)
                for (e, k), arr in zip(key_order, stack):
                    shaped[e][k] = arr
        else:  # bilinear
            from scipy.ndimage import shift as _nd_shift
            for e in list(shaped.keys()):
                for k in list(shaped[e].keys()):
                    shaped[e][k] = _nd_shift(
                        shaped[e][k], (fy, fx),
                        order=1, mode="nearest", prefilter=False,
                    ).astype(np.float32, copy=False)

    # --- Phase C: crop + package patches
    patches: Dict[str, Dict[str, np.ndarray]] = {}
    wrote_any = False
    ph = y_end - yB0
    pw = x_end - xB0
    for e in elements:
        patches[e] = {}
        for key, img in shaped.get(e, {}).items():
            patch = img[yA0:yA1, xA0:xA1]
            patches[e][key] = patch[:ph, :pw]
            wrote_any = True

    if not wrote_any:
        return None

    return {
        'yB0': yB0, 'yB1': yB1, 'xB0': xB0, 'xB1': xB1,
        'y_end': y_end, 'x_end': x_end,
        'ww': ww,
        'ww_eff': ww_eff,
        'has_mask': roi_mask is not None,
        'patches': patches,
    }


def roi_to_banded_maps(
    manifest_path: Path,
    roi: Tuple[int, int, int, int],
    elements: List[str],
    width_eV: float = 150.0,
    sideband_mult: float = 1.5,
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    blend_mode: str = "mean",   # "mean" or "sum"
    progress: bool = True,
    roi_mask: Optional[np.ndarray] = None,
    _debug_return_stats: bool = False,
    _cancel_flag=None,       # callable() -> bool; True = abort (used by _EDSWorker)
    _progress_cb=None,       # callable(tiles_done, tiles_total) for Qt progress signal
    max_workers: int = 0,    # 0 = auto (min(n_tiles, cpu_count, 8))
    _rows=None,              # pre-parsed manifest rows; skips read_manifest + energy scan
    return_weight: bool = False,  # include raw weight array in stats dict
    shift_method: str = "bilinear",  # "bilinear" (fast, default) or "fft" (sinc, exact)
) -> Dict[str, Dict[str, np.ndarray]] | Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, int]]:
    """
    Returns three images per element: {'main','sb_lo','sb_hi'} aligned with BSE,
    using only tiles intersecting the ROI. Supports polygon ROI via roi_mask.

    If _debug_return_stats=True, returns (accum, stats_dict).
    stats_dict keys:
      - manifest_rows
      - intersected_tiles
      - opened_eds_tiles
      - energy_axis_matches
      - tiles_accumulated
      - weight (only when return_weight=True): raw accumulation weight array
    """
    base_dir = Path(manifest_path).parent

    # --- manifest rows (use pre-parsed rows when provided to skip CSV I/O) ---
    rows = _rows if _rows is not None else read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty.")

    # --- energy axis: try manifest rows first (no BCF I/O); fall back to loading ---
    E0 = dE = None
    nE = None
    for r in rows:
        try:
            _nE = int(r.get("nE") or 0)
            if _nE and r.get("dE") is not None and r.get("E0") is not None:
                E0 = float(r["E0"]); dE = float(r["dE"]); nE = _nE
                break
        except (ValueError, TypeError):
            continue
    if nE is None:
        for r in rows:
            p = _resolve_manifest_path(r["file"], base_dir)
            sig = load_eds_signal_lazy(p)
            if sig is None:
                continue
            meta = extract_eds_metadata_from_signal(sig)
            if meta.get("nE", 0) and meta.get("dE") is not None and meta.get("E0") is not None:
                E0 = float(meta["E0"]); dE = float(meta["dE"]); nE = int(meta["nE"])
                break
    if nE is None:
        raise RuntimeError("No EDS signals found in manifest (or missing energy axis).")

    # --- ROI dims ---
    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0)
    W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    # --- elements & bands ---
    elements = [e for e in elements if e in LINE_ENERGIES_KEV]
    if not elements:
        raise ValueError("No valid elements requested.")
    bands = {}
    for e in elements:
        Ec = LINE_ENERGIES_KEV[e]
        lo_main, hi_main = _band_indices(E0, dE, nE, Ec, width_eV)
        Ec_lo, Ec_hi = _sideband_centers(Ec, width_eV, sideband_mult)
        lo_lo, hi_lo = _band_indices(E0, dE, nE, Ec_lo, width_eV)
        lo_hi, hi_hi = _band_indices(E0, dE, nE, Ec_hi, width_eV)
        bands[e] = {'main': (lo_main, hi_main), 'sb_lo': (lo_lo, hi_lo), 'sb_hi': (lo_hi, hi_hi)}

    # --- prefilter rows by AABB/ROI ---
    roi_rows = _prefilter_rows_for_roi(rows, roi)

    # --- diagnostics counters ---
    cnt_total_rows = len(rows)
    cnt_intersect = len(roi_rows)
    cnt_accum = 0

    # --- accumulators ---
    accum: Dict[str, Dict[str, np.ndarray]] = {
        e: {k: np.zeros((H, W), dtype=np.float32) for k in ('main', 'sb_lo', 'sb_hi')}
        for e in elements
    }
    weight = np.zeros((H, W), dtype=np.float32)

    # --- parallel tile processing ---
    n_workers = min(
        max_workers if max_workers > 0 else (os.cpu_count() or 4),
        max(len(roi_rows), 1),
        8,
    )
    tiles_done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [
            pool.submit(
                _process_one_tile,
                r, base_dir, elements, bands, E0, dE, nE,
                roi, feather, tukey_alpha, apply_subpixel, roi_mask, _cancel_flag,
                shift_method,
            )
            for r in roi_rows
        ]

        # tqdm for console progress (rate + ETA); _progress_cb drives Qt dialog independently
        if progress:
            from tqdm import tqdm
            futs_iter = tqdm(as_completed(futs), desc="ROI EDS (bands)", unit="tile", total=len(futs))
        else:
            futs_iter = as_completed(futs)

        for fut in futs_iter:
            tiles_done += 1
            if _progress_cb is not None:
                _progress_cb(tiles_done, len(roi_rows))

            result = fut.result()
            if result is None:
                continue

            yB0 = result['yB0']; yB1 = result['yB1']
            xB0 = result['xB0']; xB1 = result['xB1']
            y_end = result['y_end']; x_end = result['x_end']
            ww = result['ww']; ww_eff = result['ww_eff']
            patches = result['patches']

            for e in elements:
                for key in ('main', 'sb_lo', 'sb_hi'):
                    patch = patches.get(e, {}).get(key)
                    if patch is None:
                        continue
                    accum[e][key][yB0:y_end, xB0:x_end] += patch * ww_eff

            if result['has_mask']:
                weight[yB0:y_end, xB0:x_end] += ww_eff
            else:
                weight[yB0:yB1, xB0:xB1] += ww

            cnt_accum += 1

    # Capture raw weight before normalization (for overlap detection)
    weight_raw = weight.copy() if return_weight else None

    # normalize (mean) if requested
    if blend_mode != "sum":
        m = weight > 0
        if m.any():
            invw = np.zeros_like(weight, dtype=np.float32)
            invw[m] = 1.0 / weight[m]
            for e in elements:
                for key in ('main', 'sb_lo', 'sb_hi'):
                    arr = accum[e][key]
                    arr[m] = arr[m] * invw[m]
                    accum[e][key] = arr

    if _debug_return_stats:
        stats = {
            "manifest_rows": cnt_total_rows,
            "intersected_tiles": cnt_intersect,
            "opened_eds_tiles": cnt_accum,
            "energy_axis_matches": cnt_accum,
            "tiles_accumulated": cnt_accum,
        }
        if weight_raw is not None:
            stats["weight"] = weight_raw
        return accum, stats
    else:
        return accum

def roi_to_summed_spectrum(
    manifest_path: Path,
    roi: Tuple[int,int,int,int],
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    progress: bool = True,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[float,float,int,str], Dict]:
    """
    Streamed, alignment-correct, polygon-aware sum spectrum for an ROI.
    Returns:
      spec_1d (nE,), (E0, dE, nE, units), meta dict
    """
    import hyperspy.api as hs
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from skimage.transform import resize
    from tqdm import tqdm

    base_dir = Path(manifest_path).parent

    def _resolve_path(p_like) -> Path:
        try:
            p = Path(p_like)
        except Exception:
            p = Path(str(p_like))
        if not p.is_absolute():
            p = base_dir / p
        if not p.exists():
            cands = list(base_dir.rglob(p.name))
            if cands:
                return cands[0]
        return p

    def tukey_window(h, w, a):
        wy = tukey1d(h, alpha=a).astype(np.float32)
        wx = tukey1d(w, alpha=a).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty.")

    # Energy axis (first usable)
    E0 = dE = None; nE = None; units = "keV"
    for r in rows:
        sig = load_eds_signal_lazy(_resolve_path(r["file"]))
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if meta.get("nE", 0) and meta.get("dE", None) is not None and meta.get("E0", None) is not None:
            E0 = float(meta["E0"]); dE = float(meta["dE"]); nE = int(meta["nE"])
            units = meta.get("energy_units", "keV") or "keV"
            break
    if nE is None:
        raise RuntimeError("No EDS signals found in manifest.")

    y0, y1, x0, x1 = map(int, roi)
    H = max(0, y1 - y0); W = max(0, x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    # Prefilter intersecting tiles
    roi_rows = _prefilter_rows_for_roi(rows, roi)

    # Accumulate summed spectrum with weights
    spec = np.zeros((nE,), dtype=np.float64)
    wsum = 0.0

    iterator = tqdm(roi_rows, desc="ROI EDS (sum spectrum)", unit="tile", total=len(roi_rows)) if progress else roi_rows
    for r in iterator:
        p = _resolve_path(r["file"])
        sig = load_eds_signal_lazy(p)
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        if not (meta.get("nE",0) == nE and
                abs(float(meta.get("E0",0.0)) - E0) < 1e-6 and
                abs(float(meta.get("dE",0.0)) - dE) < 1e-9):
            continue

        # ensure array shape (h,w,nE)
        arr = np.asarray(sig.data)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[0] == nE:
            arr = np.moveaxis(arr, 0, 2)
        if arr.ndim != 3 or arr.shape[2] != nE:
            continue

        th = int(r["bse_h"]); tw = int(r["bse_w"])
        sy = float(r.get("scale_y", 1.0) or 1.0)
        sx = float(r.get("scale_x", 1.0) or 1.0)
        shx = float(r.get("shx", 0.0) or 0.0)
        shy = float(r.get("shy", 0.0) or 0.0)
        ty  = float(r["y"]);  tx  = float(r["x"])

        # resample to BSE size
        tgt_h = max(1, int(round(th*sy))); tgt_w = max(1, int(round(tw*sx)))
        if arr.shape[0] != tgt_h or arr.shape[1] != tgt_w:
            tmp = np.empty((tgt_h, tgt_w, nE), dtype=np.float32)
            for k in range(nE):
                tmp[..., k] = resize(arr[..., k], (tgt_h, tgt_w), order=1, preserve_range=True, anti_aliasing=False)
            arr = tmp
        if arr.shape[0] != th or arr.shape[1] != tw:
            tmp = np.empty((th, tw, nE), dtype=np.float32)
            for k in range(nE):
                tmp[..., k] = resize(arr[..., k], (th, tw), order=1, preserve_range=True, anti_aliasing=False)
            arr = tmp

        # shear
        if abs(shx) > 1e-7 or abs(shy) > 1e-7:
            M = np.array([[1.0, shx],[shy,1.0]], dtype=np.float64)
            Minv = np.linalg.inv(M)
            for k in range(nE):
                arr[..., k] = affine_transform(arr[..., k], Minv, offset=0.0, order=1, mode='nearest', cval=0.0, prefilter=False)

        # subpixel
        if apply_subpixel:
            fy = ty - np.floor(ty); fx = tx - np.floor(tx)
            if abs(fy) > 1e-6 or abs(fx) > 1e-6:
                from scipy.ndimage import fourier_shift
                for k in range(nE):
                    arr[..., k] = np.fft.ifftn(fourier_shift(np.fft.fftn(arr[..., k]), (fy, fx))).real.astype(np.float32)

        # intersect with ROI
        iy = int(np.floor(ty)); ix = int(np.floor(tx))
        yA0 = max(0, y0 - iy); yA1 = min(th, y1 - iy)
        xA0 = max(0, x0 - ix); xA1 = min(tw, x1 - ix)
        if yA1 <= yA0 or xA1 <= xA0:
            continue

        yB0 = yA0 + (iy - y0); yB1 = yB0 + (yA1 - yA0)
        xB0 = xA0 + (ix - x0); xB1 = xB0 + (xA1 - xA0)

        ww_full = tukey_window(th, tw, tukey_alpha) if feather else np.ones((th, tw), np.float32)
        ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)

        if roi_mask is not None:
            mask_sub = roi_mask[yB0:yB1, xB0:xB1].astype(np.float32)
            if mask_sub.shape != ww.shape:
                mh, mw = mask_sub.shape
                ww = ww[:mh, :mw]
                yA1 = yA0 + mh; xA1 = xA0 + mw
            ww *= mask_sub

        # weighted sum over pixels → spectrum
        patch = arr[yA0:yA1, xA0:xA1, :]  # (h,w,nE)
        w = ww[..., None]                 # (h,w,1)
        spec += np.sum(patch * w, axis=(0,1))
        wsum += float(np.sum(ww))

    meta_out = {
        "wsum": wsum,
        "roi": (y0, y1, x0, x1),
        "units": units,
    }
    # If you want mean counts per (weighted) pixel instead of total counts:
    # if wsum > 0: spec = spec / wsum

    return spec.astype(np.float32), (E0, dE, nE, units), meta_out

def write_emsa_msa(
    path: Path,
    spec: np.ndarray,
    E0: float, dE: float, units: str = "keV",
    title: str = "ROI spectrum",
    beam_energy_keV: float | None = None,
    extra_headers: dict | None = None,
):
    """
    Minimal EMSA/MSA exporter.
    spec: (nE,) counts
    axis: E(EV) or keV, per 'units'
    """
    path = Path(path)
    nE = spec.shape[0]
    xunit = "KEV" if units.lower().startswith("k") else "EV"
    # EMSA requires energy axis in eV; if you keep keV, declare KEV and set dE accordingly.
    factor = 1000.0 if xunit == "EV" and units.lower().startswith("k") else 1.0
    with path.open("w", encoding="utf-8") as f:
        w = f.write
        w("#FORMAT      : EMSA/MAS Spectral Data File\n")
        w("#VERSION     : 1.0\n")
        w(f"#TITLE       : {title}\n")
        if beam_energy_keV is not None:
            w(f"#BEAMKV      : {beam_energy_keV:.6g}\n")
        w(f"#XUNITS      : {xunit}\n")
        w("#SIGMA       : COUNTS\n")
        w(f"#NPOINTS     : {nE}\n")
        w(f"#XPERCHAN    : {dE*factor:.9g}\n")
        w(f"#OFFSET      : {E0*factor:.9g}\n")
        if extra_headers:
            for k,v in extra_headers.items():
                w(f"#{k} : {v}\n")
        w("#SPECTRUM    : BEGIN\n")
        for y in spec:
            # EMSA is one value per line; integers are OK but floats are fine
            w(f"{float(y):.6f}\n")
        w("#SPECTRUM    : END\n")


def export_roi_spectrum_msa(
    manifest_path: Path,
    roi: tuple[int,int,int,int],
    out_dir: Path,
    roi_mask: np.ndarray | None = None,
    apply_subpixel: bool = True,
    feather: bool = True,
    tukey_alpha: float = 0.08,
    progress: bool = True,
    title: str = "ROI spectrum",
):
    """
    Assemble the summed ROI spectrum (alignment-consistent), then save .msa and .hspy.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    spec, (E0, dE, nE, units), meta = roi_to_summed_spectrum(
        manifest_path, roi,
        apply_subpixel=apply_subpixel, feather=feather,
        tukey_alpha=tukey_alpha, progress=progress,
        roi_mask=roi_mask,
    )

    # .msa for vendor import
    msa_path = out_dir / "ROI_spectrum.msa"
    # try to include beam energy from first usable row
    rows = read_manifest(manifest_path)
    be = None
    for r in rows:
        if float(r.get("beam_energy", 0.0)) > 0:
            be = float(r["beam_energy"])
            break
    write_emsa_msa(
        msa_path, spec, E0, dE, units=units, title=title,
        beam_energy_keV=be, extra_headers={"ROI": str(tuple(map(int, roi)))}
    )

    # Optional .hspy for auditability
    try:
        import hyperspy.api as hs
        s = hs.signals.EDSSEMSpectrum(spec[np.newaxis, np.newaxis, :].astype(np.float32))
        s.axes_manager.signal_axes[0].name = 'Energy'
        s.axes_manager.signal_axes[0].units = units
        s.axes_manager.signal_axes[0].offset = E0
        s.axes_manager.signal_axes[0].scale = dE
        s.metadata.set_item("General.title", title)
        if be is not None:
            s.metadata.set_item("Acquisition_instrument.SEM.beam_energy", be)
        s.save(out_dir / "ROI_spectrum.hspy")
    except Exception:
        pass

    print(f"[export] Wrote ROI spectrum → {msa_path}")
    return msa_path

def compute_roi_composition_from_tile_csv(
    manifest_path: Path,
    vendor_csv: Path,
    roi: tuple[int,int,int,int],
    roi_mask: np.ndarray | None = None,
    columns_wt: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Combine vendor per-tile quant results into an ROI composition by area-weighted average.

    vendor_csv must include at least: row, col (int), and one column per element wt%.
    columns_wt maps element symbols -> column name in CSV (if None, uses identical names).
    Returns dict: element -> wt_fraction (sums to ~1).
    """
    import csv

    rows = read_manifest(manifest_path)
    # index manifest by (row,col) -> rect
    idx = {}
    for r in rows:
        rc = (int(r["row"]), int(r["col"]))
        th, tw = int(r["bse_h"]), int(r["bse_w"])
        iy, ix = int(np.floor(float(r["y"]))), int(np.floor(float(r["x"])))
        idx[rc] = (iy, iy+th, ix, ix+tw)

    # load vendor CSV
    # expected columns: row,col,<elements...>  (element columns in wt% or fraction; we’ll normalize)
    tiles = []
    with open(vendor_csv, "r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                r = int(row.get("row", row.get("Row", "")))
                c = int(row.get("col", row.get("Col", "")))
            except Exception:
                continue
            tiles.append((r, c, row))

    y0, y1, x0, x1 = map(int, roi)
    H, W = (y1 - y0), (x1 - x0)
    if H <= 0 or W <= 0:
        raise ValueError("ROI must have positive size.")

    # Optional mask
    if roi_mask is not None:
        mask = roi_mask.astype(bool)
        if mask.shape != (H, W):
            raise ValueError("roi_mask shape mismatch with ROI bbox")
    else:
        mask = None

    # accumulate weighted wt%
    elem_sum = {}
    area_sum = 0.0

    def _tile_roi_overlap(rc_rect):
        iy0, iy1, ix0, ix1 = rc_rect
        # intersect with ROI bbox
        yA0 = max(0, y0 - iy0); yA1 = min(iy1 - iy0, y1 - iy0)
        xA0 = max(0, x0 - ix0); xA1 = min(ix1 - ix0, x1 - ix0)
        if yA1 <= yA0 or xA1 <= xA0:
            return 0
        if mask is None:
            return float((yA1 - yA0) * (xA1 - xA0))
        # count masked pixels inside ROI sub-rectangle
        yB0 = yA0 + (iy0 - y0)
        xB0 = xA0 + (ix0 - x0)
        sub = mask[yB0:yB0+(yA1-yA0), xB0:xB0+(xA1-xA0)]
        return float(np.count_nonzero(sub))

    # determine element columns
    # (if columns_wt is None, we use the header names as symbols when they match like 'Si','O',...)
    header_elems = set()
    if tiles:
        sample_row = tiles[0][2]
        for k in sample_row.keys():
            if len(k) <= 2 and k[0].isalpha():  # crude heuristic: 'C','O','Si'...
                header_elems.add(k)
    col_map = columns_wt or {e: e for e in header_elems}

    for r, c, row in tiles:
        rc = (r, c)
        if rc not in idx:
            continue
        pix_area = _tile_roi_overlap(idx[rc])
        if pix_area <= 0:
            continue

        # read tile wt% and normalize to fraction
        local = {}
        total = 0.0
        for el, col in col_map.items():
            try:
                v = float(row.get(col, "0"))
            except Exception:
                v = 0.0
            if v < 0:
                v = 0.0
            local[el] = v
            total += v
        if total <= 0:
            continue
        # convert to fraction
        for el in local:
            local[el] = local[el] / total

        # area-weighted mix
        for el, frac in local.items():
            elem_sum[el] = elem_sum.get(el, 0.0) + frac * pix_area
        area_sum += pix_area

    if area_sum <= 0:
        return {}

    # normalize to 1.0
    for el in elem_sum:
        elem_sum[el] /= area_sum
    # re-normalize (guard small numeric drift)
    s = sum(elem_sum.values())
    if s > 0:
        for el in elem_sum:
            elem_sum[el] /= s
    return elem_sum

def fit_element_calibration(
    manifest_path: Path,
    vendor_csv: Path,
    elements: list[str],
    width_eV: float = 150.0,
    sideband_mult: float = 1.5,
    use_intercept: bool = True,
) -> dict[str, tuple[float, float]]:
    """
    Fit per-element linear calibration: wt% ≈ a * (net counts per tile) + b
    using tiles as calibration samples. Returns dict el -> (a, b).
    """
    import csv
    from sklearn.linear_model import LinearRegression

    rows = read_manifest(manifest_path)
    # map (row,col) -> path
    rc_to_file = {(int(r["row"]), int(r["col"])): r["file"] for r in rows}

    # load vendor wt% per tile
    tile_wt = {}
    with open(vendor_csv, "r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                r = int(row.get("row", row.get("Row", "")))
                c = int(row.get("col", row.get("Col", "")))
            except Exception:
                continue
            tile_wt[(r,c)] = row

    # compute net counts per tile for requested elements (full tile ROI)
    # reuse roi_to_banded_maps over each tile bbox, then sum net counts
    models = {}
    for el in elements:
        X = []  # net counts
        y = []  # vendor wt fraction
        for rc, path in rc_to_file.items():
            if rc not in tile_wt:
                continue
            # tile bbox in BSE coords
            # find the manifest row:
            rline = [r for r in rows if int(r["row"])==rc[0] and int(r["col"])==rc[1]][0]
            iy, ix = int(np.floor(float(rline["y"]))), int(np.floor(float(rline["x"])))
            th, tw = int(rline["bse_h"]), int(rline["bse_w"])
            roi = (iy, iy+th, ix, ix+tw)

            band = roi_to_banded_maps(
                manifest_path, roi, [el],
                width_eV=width_eV, sideband_mult=sideband_mult,
                apply_subpixel=True, feather=True, blend_mode="mean",
                progress=False, roi_mask=None
            )[el]
            net = _net_count_maps({el: band}, width_eV=width_eV, sideband_mult=sideband_mult)[el]
            net_val = max(0.0, float(np.nansum(net)))   # signed sum, clipped at the aggregate

            # vendor wt% (as fraction)
            vrow = tile_wt[rc]
            try:
                wt = float(vrow.get(el, vrow.get(el+"(wt%)", "0")))
            except Exception:
                wt = 0.0
            X.append([net_val])
            y.append(wt/100.0 if wt>1.0 else wt)

        if len(X) >= 3:
            lr = LinearRegression(fit_intercept=use_intercept)
            lr.fit(np.asarray(X), np.asarray(y))
            a = float(lr.coef_[0])
            b = float(lr.intercept_) if use_intercept else 0.0
            models[el] = (a, b)
    return models


def apply_calibration_to_roi(
    band_maps: dict[str, dict[str, np.ndarray]],
    calib: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """
    Convert your ROI net counts maps to estimated wt fractions using per-element linear calibration.
    Returns dict el->fraction normalized to sum=1.
    """
    est = {}
    nets = _net_count_maps(band_maps)
    for el, net in nets.items():
        S = max(0.0, float(np.nansum(net)))   # signed sum, clipped at the aggregate
        if el in calib:
            a, b = calib[el]
            est[el] = max(0.0, a * S + b)
        else:
            est[el] = 0.0
    # normalize to 1
    s = sum(est.values()) or 1.0
    for el in est:
        est[el] /= s
    return est

def roi_to_summed_spectrum_streamed(
    manifest_path: Path,
    roi: tuple[int, int, int, int],
    roi_mask: Optional[np.ndarray] = None,
    energy_chunk: int = 16,
    progress: bool = True,
) -> tuple[np.ndarray, tuple[float, float, int, str]]:
    """
    Ultra-fast, low-memory summed spectrum for an ROI.

    Strategy:
      • Find the single tile intersecting ROI (or pick the best-overlap tile if several).
      • Map ROI bbox to that tile's EDS navigation indices using scale_y/scale_x.
      • (Optional) Rasterize ROI polygon mask to that nav window (nearest).
      • Stream energy in small chunks, do weighted sums, accumulate into final spectrum.

    Returns: (spectrum_float32[nE], (E0, dE, nE, units))
    """
    import numpy as np
    from skimage.transform import resize
    from tqdm import tqdm

    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty; run stitching first.")

    # Resolve file paths relative to manifest
    for r in rows:
        p = Path(r["file"])
        if not p.is_absolute():
            p = base_dir / p
        r["file"] = p

    y0, y1, x0, x1 = map(int, roi)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("ROI must have positive size.")

    # Candidate tiles that intersect ROI (in BSE space)
    cand = []
    for r in rows:
        th, tw = int(r["bse_h"]), int(r["bse_w"])
        iy, ix = int(np.floor(float(r["y"]))), int(np.floor(float(r["x"])))
        Y0 = max(y0, iy);         Y1 = min(y1, iy + th)
        X0 = max(x0, ix);         X1 = min(x1, ix + tw)
        area = max(0, Y1 - Y0) * max(0, X1 - X0)
        if area > 0:
            cand.append((area, r, (Y0, Y1, X0, X1)))
    if not cand:
        raise RuntimeError("No tiles intersect the ROI.")
    # Prefer the tile with the largest intersection
    cand.sort(key=lambda t: t[0], reverse=True)
    r = cand[0][1]
    Y0, Y1, X0, X1 = cand[0][2]

    # Load its EDS signal lazily
    sig = load_eds_signal_lazy(r["file"])
    if sig is None:
        raise RuntimeError("EDS signal not found in the intersecting tile.")

    meta = extract_eds_metadata_from_signal(sig)
    nE = int(meta.get("nE", 0) or 0)
    dE = float(meta.get("dE", 0.0) or 0.0)
    E0 = float(meta.get("E0", 0.0) or 0.0)
    units = meta.get("units", "keV") or "keV"
    if nE <= 0 or dE == 0.0:
        raise RuntimeError("No EDS energy axis in the tile signal.")

    # Map BSE subwindow → EDS nav indices
    th, tw  = int(r["bse_h"]), int(r["bse_w"])
    sy, sx  = float(r.get("scale_y", 1.0) or 1.0), float(r.get("scale_x", 1.0) or 1.0)
    iy, ix  = int(np.floor(float(r["y"]))), int(np.floor(float(r["x"])))

    # BSE-tile local coords of intersection
    yA0 = Y0 - iy; yA1 = Y1 - iy
    xA0 = X0 - ix; xA1 = X1 - ix

    # To EDS nav grid
    eds_y0 = int(np.floor(yA0 * sy))
    eds_y1 = int(np.ceil (yA1 * sy))
    eds_x0 = int(np.floor(xA0 * sx))
    eds_x1 = int(np.ceil (xA1 * sx))
    eds_y0 = max(0, eds_y0); eds_x0 = max(0, eds_x0)

    # Read a small slice to discover nav order/shape
    # We'll try to avoid materializing more than needed.
    test = sig.isig[0:1].inav[eds_y0:eds_y1, eds_x0:eds_x1]
    block = np.asarray(test.data)  # shape could be (h,w,1) or (1,h,w) depending on reader
    # Infer layout
    if block.ndim == 3 and block.shape[-1] == 1:
        nav_h, nav_w = block.shape[0], block.shape[1]
        energy_first = False
    elif block.ndim == 3 and block.shape[0] == 1:
        nav_h, nav_w = block.shape[1], block.shape[2]
        energy_first = True
    else:
        # Fallback: ask axes manager (signal_axes[0] is energy)
        energy_first = (sig.data.ndim == 3 and sig.data.shape[0] == nE)
        if energy_first:
            nav_h, nav_w = sig.data.shape[1], sig.data.shape[2]
            nav_h = min(nav_h, eds_y1 - eds_y0)
            nav_w = min(nav_w, eds_x1 - eds_x0)
        else:
            nav_h, nav_w = eds_y1 - eds_y0, eds_x1 - eds_x0

    if nav_h <= 0 or nav_w <= 0:
        # Extremely tiny ROI that rounds to zero in EDS grid
        return np.zeros((nE,), dtype=np.float32), (E0, dE, nE, units)

    # Prepare (optional) polygon mask resized to EDS nav window
    Wmask = None
    if roi_mask is not None:
        # roi_mask is in full BSE mosaic coords for bbox (y0:y1,x0:x1).
        # Extract the overlapping BSE mask window and resize to EDS nav window.
        mask_sub = roi_mask[(Y0 - y0):(Y0 - y0 + (Y1 - Y0)),
                            (X0 - x0):(X0 - x0 + (X1 - X0))]
        if mask_sub.size == 0:
            # nothing to include
            return np.zeros((nE,), dtype=np.float32), (E0, dE, nE, units)
        Wmask = resize(mask_sub.astype(np.float32),
                       (nav_h, nav_w),
                       order=0, preserve_range=True, anti_aliasing=False).astype(np.float32)
        # Flatten for fast dot later
        Wmask = Wmask.reshape(-1)

    # Stream over energy
    spec = np.zeros((nE,), dtype=np.float64)

    rng = range(0, nE, max(1, int(energy_chunk)))
    if progress:
        rng = tqdm(rng, total=(nE + energy_chunk - 1) // energy_chunk, desc="Summing ROI spectrum", unit="chunk")

    for k0 in rng:
        k1 = min(nE, k0 + energy_chunk)

        # Slice (energy, nav) without reordering entire cube
        if energy_first:
            # data shape ~ (E, H, W)
            sl = sig.isig[k0:k1].inav[eds_y0:eds_y1, eds_x0:eds_x1].data
            blk = np.asarray(sl)  # (k, h, w)
        else:
            # data shape ~ (H, W, E)
            sl = sig.inav[eds_y0:eds_y1, eds_x0:eds_x1].isig[k0:k1].data
            blk = np.asarray(sl)  # (h, w, k)
            blk = np.moveaxis(blk, -1, 0)  # -> (k, h, w)

        k = blk.shape[0]  # chunk size actually loaded
        if k == 0:
            continue

        if Wmask is None:
            # Sum across nav dims directly
            # (k,h,w) -> (k,)
            spec[k0:k0+k] += blk.sum(axis=(1, 2), dtype=np.float64)
        else:
            # Weighted sum per energy: flatten nav to (h*w), dot with Wmask
            h, w = blk.shape[1], blk.shape[2]
            M = blk.reshape(k, h * w)           # (k, hw)
            spec[k0:k0+k] += (M @ Wmask).astype(np.float64)

        # Free the temporary chunk ASAP
        del blk

    return spec.astype(np.float32), (E0, dE, nE, units)
