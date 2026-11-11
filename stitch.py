import numpy as np
import hyperspy.api as hs
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re, csv, math, warnings, hashlib
import matplotlib.pyplot as plt
import argparse

from skimage import exposure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift, gaussian_filter, zoom, affine_transform
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from tqdm import tqdm
from typing import Sequence

from eds_map import write_tile_manifest, extract_eds_roi


warnings.filterwarnings("ignore", category=FutureWarning, module="skimage.registration._phase_cross_correlation")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skimage.registration._phase_cross_correlation")

# =========================
# Tunables
# =========================
#directory = Path("C23S1A_Full")

# The alignment tunables (OVERLAP_FRAC, BAND_FRAC, N_SLICES, UPSAMPLE,
# MAX_STRIP_DRIFT, MAD_Z, REFINE_ITERS, REFINE_MAX_SHIFT, PYR_LEVELS,
# TUKEY_ALPHA, LOW_PCT, HI_PCT, MAX_VERT_X_DRIFT, MAX_RIGHT_Y_DRIFT, SHEAR_CAP,
# SHEAR_ENABLE, RIGHT_DY_WEIGHT, VSTEP_ANCHOR_W, anchor_sigma) are module
# globals set by apply_preset() below. Their base values, the presets A/B/C,
# "default" (C + the C23S1A-class overrides; the paper's mosaic) and what each
# preset does are in stitch_presets.py: one source for this script and the
# viewer's stitching window. Choose with --preset or apply_preset().
from stitch_presets import (BASE_TUNABLES as _BASE_TUNABLES, PRESETS,   # noqa: E402
                            FINE_GRAINED_OVERRIDES as _FINE_GRAINED_OVERRIDES,
                            PRESET_CHOICES, PRESET_HELP, resolve_preset)


def apply_preset(name: str) -> None:
    """Set the alignment tunables of this module to preset ``name``."""
    globals().update(resolve_preset(name))
    globals()["PRESET"] = name


def current_tunables() -> dict:
    """The alignment tunables in effect (for the run record)."""
    return {k: globals()[k] for k in _BASE_TUNABLES}


PRESET = "default"
apply_preset(PRESET)

# Orientation helper:
#   "none"     -> keep as solved
#   "top-left" -> mirror X so minimal (row,col) ends up top-left
#   "auto"     -> mirror if median RIGHT ΔX < 0
ORIENT            = "none"

# Debug
DEBUG = True
SAVE_WORST_OVERLAPS = 8
DEBUG_OUTDIR = Path("stitch_debug")

# Overlay labels
DRAW_LABELS  = False
SHOW_ROWCOL  = True
SHOW_INDEX   = True
LABEL_COLOR  = (255, 0, 0)
LABEL_SIZE   = 18

# =========================
# Stage coordinate helpers
# =========================
def _stage_from_metadata(md):
    """Extract (stage_x, stage_y) from an already-loaded metadata object."""
    if md is None:
        return 0.0, 0.0
    try:
        sx = float(md.Acquisition_instrument.SEM.Stage.X)
        sy = float(md.Acquisition_instrument.SEM.Stage.Y)
        return sx, sy
    except Exception:
        pass
    try:
        sx = float(md.get_item("Acquisition_instrument.SEM.Stage.x"))
        sy = float(md.get_item("Acquisition_instrument.SEM.Stage.y"))
        return sx, sy
    except Exception:
        pass
    return 0.0, 0.0


def _pixel_size_from_metadata(md):
    """Extract pixel size (stage units per pixel) from an already-loaded metadata object."""
    if md is None:
        return None
    try:
        return float(md.Acquisition_instrument.SEM.Detector.BSE.pixel_size)
    except Exception:
        return None


def _bcf_cache_key(p: Path) -> str:
    """Stable hash of (size, mtime_ns). Cheap, stat-only.
    Path is intentionally excluded so the cache is portable across workstations
    where the same dataset is mounted under different drive letters.
    """
    st = p.stat()
    raw = f"{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


def _load_bcf_cached(p: Path, cache_dir: Path | None):
    """Return (bse_float32, sx, sy, ps) for a BCF file, reading from an on-disk
    cache when possible. Cache key is file mtime + size, so any external edit
    to the BCF invalidates the entry transparently.

    When cache_dir is None, caching is disabled (direct hs.load, no write).
    """
    if cache_dir is not None:
        key = _bcf_cache_key(p)
        cf = cache_dir / f"{key}.npz"
        if cf.exists():
            try:
                with np.load(cf, allow_pickle=False) as z:
                    raw = z["bse"].astype(np.float32, copy=False)
                    sx = float(z["sx"]); sy = float(z["sy"])
                    ps_v = float(z["ps"])
                    ps = ps_v if np.isfinite(ps_v) else None
                    return raw, sx, sy, ps
            except Exception as e:
                print(f"[cache] bad entry for {p.name} ({e}); re-reading BCF")

    signals = hs.load(p)
    raw = signals[0].data.astype(np.float32)
    try:
        md = getattr(signals[2], "metadata", None)
    except Exception:
        md = None
    sx, sy = _stage_from_metadata(md)
    ps = _pixel_size_from_metadata(md)

    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_dir / f"{key}.npz",
                bse=raw,
                sx=np.float64(sx),
                sy=np.float64(sy),
                ps=np.float64(ps if ps is not None else float("nan")),
            )
        except Exception as e:
            print(f"[cache] failed to write entry for {p.name}: {e}")

    return raw, sx, sy, ps


def get_stage_coordinates(p: Path):
    """
    Return (stage_x, stage_y) from BCF metadata.

    SEM convention considered here: origin top-right; X increases leftward; Y increases downward.
    """
    try:
        s = hs.load(p)[2]
    except Exception:
        return 0.0, 0.0
    return _stage_from_metadata(getattr(s, "metadata", None))


def get_pixel_size(p: Path):
    """
    Returns pixel size (stage units per pixel), e.g. µm/px, if present; else None.
    """
    try:
        s = hs.load(p)[2]
    except Exception:
        return None
    return _pixel_size_from_metadata(getattr(s, "metadata", None))


# =========================
# Robust quantization from stage → integer grid
# =========================
def _mad(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return np.median(np.abs(x - med))

def _robust_median(v, zcut=2.5):
    v = np.asarray(v, float)
    med = np.median(v)
    mad = _mad(v) + 1e-12
    z = np.abs(v - med) / (1.4826 * mad)
    v2 = v[z < zcut] if np.any(z < zcut) else v
    return float(np.median(v2))

def _robust_row_step(sy):
    sy = np.asarray(sy, float)
    vs = np.sort(sy)
    diffs = np.diff(vs)
    if diffs.size == 0:
        return 1.0
    # Row step is among the *largest* gaps (between horizontal scanlines)
    q = max(0, int(0.7 * diffs.size))
    dy_cand = diffs[q:] if q < diffs.size else diffs
    dy = _robust_median(dy_cand) if dy_cand.size else np.median(diffs)
    if not np.isfinite(dy) or dy <= 0:
        dy = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
    return float(max(dy, 1e-9))

def _robust_col_step_from_rows(sx, r_idx):
    """
    From already-quantized rows (r_idx), estimate global |dx| using within-row diffs.
    """
    sx = np.asarray(sx, float)
    r_idx = np.asarray(r_idx, int)
    diffs = []
    for r in np.unique(r_idx):
        mask = (r_idx == r)
        if np.sum(mask) < 2:
            continue
        row_sx = np.sort(sx[mask])[::-1]  # descending (left->right)
        d = np.abs(np.diff(row_sx))
        if d.size:
            diffs.append(d)
    if not diffs:
        return 1.0
    diffs = np.concatenate(diffs)
    dx = _robust_median(diffs)
    if not np.isfinite(dx) or dx <= 0:
        dx = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    return float(max(dx, 1e-9))

def build_grid_from_stage_quantized(stage_xy):
    """
    Quantize stage coords (sx, sy) into an integer lattice (row, col).

    Steps:
      1) Estimate row spacing dy_stage from the upper-quantile of sy diffs.
      2) Quantize each sy to an integer row: r = round((sy - sy_min)/dy_stage).
         Re-index (compact) rows in ascending order of their median sy.
      3) With rows fixed, estimate column spacing dx_stage from within-row sx diffs.
      4) For each row, set its reference (leftmost) as col 0 and quantize:
             c = round((sx_ref_row - sx)/dx_stage)
         (X increases leftwards, so leftmost has the largest sx.)
      5) Compute row centers (median sy per row) and col centers (median sx per col).

    Returns:
      row_centers, col_centers, r_assign, c_assign, dy_stage, dx_stage
      (Assign arrays correspond 1:1 with input stage_xy order.)
    """
    N = len(stage_xy)
    sx = np.asarray([t[0] for t in stage_xy], float)
    sy = np.asarray([t[1] for t in stage_xy], float)

    # 1) row spacing
    dy_stage = _robust_row_step(sy)
    sy_min = float(np.min(sy))
    r_float = (sy - sy_min) / max(dy_stage, 1e-12)
    r_idx_raw = np.round(r_float).astype(int)

    # Compact rows by ascending median sy per unique raw index
    unique_r = np.unique(r_idx_raw)
    r_groups = {}
    for r in unique_r:
        r_groups[r] = float(np.median(sy[r_idx_raw == r]))
    # sort by center (ascending sy)
    r_sorted = sorted(r_groups.items(), key=lambda kv: kv[1])
    r_map = {old: new for new, (old, _) in enumerate(r_sorted)}
    r_idx = np.array([r_map[r] for r in r_idx_raw], dtype=int)

    # 2) global col spacing from within-row diffs
    dx_stage = _robust_col_step_from_rows(sx, r_idx)

    # 3) quantize cols per row using row-specific leftmost (largest sx) as reference
    c_idx = np.zeros_like(r_idx, dtype=int)
    for r in np.unique(r_idx):
        mask = (r_idx == r)
        sx_row = sx[mask]
        if sx_row.size == 0:
            continue
        sx_ref = float(np.max(sx_row))  # leftmost
        c_idx[mask] = np.round((sx_ref - sx_row) / max(dx_stage, 1e-12)).astype(int)

    # 4) row and col centers
    # rows: median sy per integer row
    row_centers = []
    for r in range(int(np.max(r_idx)) + 1):
        vals = sy[r_idx == r]
        row_centers.append(float(np.median(vals)) if vals.size else (row_centers[-1] if row_centers else sy_min))

    # cols: median sx per integer col (across all rows)
    col_centers = []
    max_c = int(np.max(c_idx)) if c_idx.size else -1
    for c in range(max_c + 1):
        vals = sx[c_idx == c]
        col_centers.append(float(np.median(vals)) if vals.size else (col_centers[-1] if col_centers else float(np.max(sx))))

    return row_centers, col_centers, r_idx, c_idx, dy_stage, dx_stage

def initial_pixel_positions_from_grid(row_idx, col_idx, step_y, step_x):
    """
    Convert integer (row, col) indices into pixel positions (y,x) using provided steps.
    """
    rmin = int(np.min(row_idx)) if len(row_idx) else 0
    cmin = int(np.min(col_idx)) if len(col_idx) else 0

    pos0 = {}
    for r, c in zip(row_idx, col_idx):
        y = (int(r) - rmin) * step_y
        x = (int(c) - cmin) * step_x
        pos0[(int(r), int(c))] = np.array([y, x], dtype=np.float64)
    return pos0

# =========================
# Registration helpers
# =========================
PREP_VARIANT = "edge"  # runtime-selectable; overridden by CLI or auto-calibration

def prep_for_corr(img: np.ndarray, variant: str | None = None) -> np.ndarray:
    """Prep a tile for phase correlation.

    Variants:
      - 'edge':    hist-eq + HP σ=3 + gradient magnitude. Matched for sharp features.
      - 'wide_hp': hist-eq + HP σ=20. Preserves broad texture; suited to diffuse contrast.
      - 'flat':    hist-eq only. Relies on phcorr's native DC invariance.
      - 'dog':     hist-eq + DoG (σ=5 − σ=20). Band-pass centred on 10–30 px period.
    """
    v = variant if variant is not None else PREP_VARIANT
    im = exposure.equalize_hist(img.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    if v == "edge":
        im_hp = im - gaussian_filter(im, sigma=3)
        gy, gx = np.gradient(im_hp.astype(np.float32))
        out = np.hypot(gx, gy)
    elif v == "wide_hp":
        out = im - gaussian_filter(im, sigma=20)
    elif v == "flat":
        out = im
    elif v == "dog":
        out = gaussian_filter(im, sigma=5) - gaussian_filter(im, sigma=20)
    else:
        raise ValueError(f"Unknown prep variant: {v!r}")
    mu = np.mean(out); sd = np.std(out) + 1e-6
    return ((out - mu) / sd).astype(np.float32, copy=False)

def hann2d(h, w):
    wy = np.hanning(max(h, 2))
    wx = np.hanning(max(w, 2))
    return np.outer(wy, wx).astype(np.float32)

def edge_texture_weight(strip):
    gy, gx = np.gradient(strip)
    e = float(np.mean(gy*gy + gx*gx))
    return max(e, 1e-6)

def phcorr(A, B):
    s = phase_cross_correlation(A, B, upsample_factor=UPSAMPLE)
    if hasattr(s[0], "__len__"):
        dy, dx = float(s[0][0]), float(s[0][1])
    else:
        dy, dx = float(s[0]), float(s[1])
    if not np.isfinite(dy): dy = 0.0
    if not np.isfinite(dx): dx = 0.0
    return dy, dx

def median_inliers(vals, zcut=MAD_Z):
    vals = np.asarray(vals, dtype=np.float64)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    z = np.abs(vals - med) / (1.4826 * mad)
    keep = z < zcut
    kept = vals[keep] if np.any(keep) else vals
    return float(np.median(kept)), keep

# ---- robust helpers for med/MAD and residual reweighting ----
def _robust_medmad(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return None, None
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) + 1e-6
    return med, 1.4826 * mad

def reweight_by_residual(pos, pairs, deltas, types, weights, huber_c=3.0):
    """Huber-like downweighting using residuals in the primary dimension."""
    # Collect residuals per type
    res_R, res_D = [], []
    for ((a,b),(dY,dX),t) in zip(pairs, deltas, types):
        ya, xa = pos[a]; yb, xb = pos[b]
        if t == "right":
            res_R.append(abs((xb - xa) - dX))
        else:
            res_D.append(abs((yb - ya) - dY))
    def _scale(v, fallback):
        v = np.asarray(v, float)
        if v.size == 0: return fallback
        mad = np.median(np.abs(v - np.median(v))) + 1e-6
        return max(fallback, 1.4826 * mad)
    sR = _scale(res_R, huber_c)
    sD = _scale(res_D, huber_c)
    # Reweight
    new_w = []
    iR = iD = 0
    for ((a,b),(dY,dX),t,w) in zip(pairs, deltas, types, weights):
        if t == "right":
            r = res_R[iR]; iR += 1; s = sR
        else:
            r = res_D[iD]; iD += 1; s = sD
        z = r / s
        mult = 1.0 if z <= 1.0 else 1.0 / z
        new_w.append(max(1e-6, w * mult))
    return new_w

def filter_pairs_by_nominal(pairs, deltas, types, step_x_nom, step_y_nom,
                            tol_right=0.30, tol_down=0.30,
                            max_dx_down=None, max_dy_right=None):
    """
    Keep only pairs that look consistent with the nominal grid.
    tol_* are relative tolerances (fraction of step).
    max_dx_down / max_dy_right clamp the orthogonal drift gates.
    Returns mask (list of bool).
    """
    keep = []
    if max_dx_down is None:
        max_dx_down = MAX_VERT_X_DRIFT
    if max_dy_right is None:
        max_dy_right = MAX_RIGHT_Y_DRIFT

    for (a, b), (dY, dX), t in zip(pairs, deltas, types):
        if t == "right":
            good_step = abs(dX - step_x_nom) <= tol_right * step_x_nom
            good_ortho = abs(dY) <= max_dy_right
            keep.append(bool(good_step and good_ortho))
        else:  # down
            good_step = abs(dY - step_y_nom) <= tol_down * step_y_nom
            good_ortho = abs(dX) <= max_dx_down
            keep.append(bool(good_step and good_ortho))
    return keep


# =========================
# Band-based neighbor deltas
# =========================
def measure_right_band(A_prep, B_prep, bh, bw, tw, max_drift=None):
    # A left, B right -> vertical band; inputs are already prep_for_corr'd full tiles
    # max_drift: absolute ceiling for per-strip phase-corr shifts. Statistical outliers are
    # still handled by median_inliers; this gate only rejects obvious phcorr failures.
    md = MAX_STRIP_DRIFT if max_drift is None else max_drift
    h, w = A_prep.shape
    ys = np.linspace(0, h - bh, N_SLICES, dtype=int)
    win = hann2d(bh, bw)
    rows = []; dys = []; dxs = []; texW = 0.0
    for y in ys:
        # Slice the prepped tile; multiply (not *=) to avoid mutating the cache.
        As = A_prep[y:y+bh, w-bw:w] * win
        Bs = B_prep[y:y+bh, :bw] * win
        dy, dx = phcorr(As, Bs)
        if abs(dy) <= md and abs(dx) <= md:
            rows.append(y + 0.5*bh)
            dys.append(dy); dxs.append(dx)
            texW += edge_texture_weight(As) + edge_texture_weight(Bs)
    if not dys:
        return None
    dy_med, ky = median_inliers(dys)
    dx_med, kx = median_inliers(dxs)
    inlier_ratio = float(np.mean(ky & kx)) if len(ky)==len(kx) else 0.5

    yv = np.asarray(rows, float)
    dyv = np.asarray(dys,  float)
    dxv = np.asarray(dxs,  float)
    k = (ky & kx) if len(ky)==len(kx) else np.ones_like(dyv, dtype=bool)
    if np.sum(k) >= 3:
        Ay = np.vstack([yv[k], np.ones(int(np.sum(k))) ]).T
        a_y, b_y = np.linalg.lstsq(Ay, dyv[k], rcond=None)[0]
        Ax = np.vstack([yv[k], np.ones(int(np.sum(k))) ]).T
        a_x, b_x = np.linalg.lstsq(Ax, dxv[k], rcond=None)[0]
    else:
        a_y = a_x = 0.0
        b_y = float(np.median(dyv)); b_x = float(np.median(dxv))

    dY = float(np.clip(-b_y, -MAX_RIGHT_Y_DRIFT, MAX_RIGHT_Y_DRIFT))   # clamp ΔY for RIGHT edges
    dX = (tw - bw) - b_x
    base_w = max(1e-3, inlier_ratio)
    weight = base_w * texW / max(len(rows), 1)
    meta = {"type":"right", "a_y":float(a_y), "a_x":float(a_x)}
    return dY, dX, weight, meta

def measure_down_band(A_prep, B_prep, bh, bw, th, max_drift=None):
    # A above, B below -> horizontal band; inputs are already prep_for_corr'd full tiles
    md = MAX_STRIP_DRIFT if max_drift is None else max_drift
    h, w = A_prep.shape
    xs = np.linspace(0, w - bw, N_SLICES, dtype=int)
    win = hann2d(bh, bw)
    cols = []; dys = []; dxs = []; texW = 0.0
    for x in xs:
        # Slice the prepped tile; multiply (not *=) to avoid mutating the cache.
        As = A_prep[h-bh:h, x:x+bw] * win
        Bs = B_prep[:bh,    x:x+bw] * win
        dy, dx = phcorr(As, Bs)
        if abs(dy) <= md and abs(dx) <= md:
            cols.append(x + 0.5*bw)
            dys.append(dy); dxs.append(dx)
            texW += edge_texture_weight(As) + edge_texture_weight(Bs)
    if not dys:
        return None
    dy_med, ky = median_inliers(dys)
    dx_med, kx = median_inliers(dxs)
    inlier_ratio = float(np.mean(ky & kx)) if len(ky)==len(kx) else 0.5

    xv = np.asarray(cols, float)
    dyv = np.asarray(dys,  float)
    dxv = np.asarray(dxs,  float)
    k = (ky & kx) if len(ky)==len(kx) else np.ones_like(dyv, dtype=bool)
    if np.sum(k) >= 3:
        Ay = np.vstack([xv[k], np.ones(int(np.sum(k))) ]).T
        a_y, b_y = np.linalg.lstsq(Ay, dyv[k], rcond=None)[0]
        Ax = np.vstack([xv[k], np.ones(int(np.sum(k))) ]).T
        a_x, b_x = np.linalg.lstsq(Ax, dxv[k], rcond=None)[0]
    else:
        a_y = a_x = 0.0
        b_y = float(np.median(dyv)); b_x = float(np.median(dxv))

    dY = (th - bh) - b_y
    dX = -b_x
    base_w = max(1e-3, inlier_ratio)
    weight = base_w * texW / max(len(cols), 1)
    meta = {"type":"down", "a_y":float(a_y), "a_x":float(a_x)}
    return dY, dX, weight, meta

# =========================
# Overlap refine (multiscale)
# =========================
def phcorr_overlap(A_prep, B_prep, ya, xa, yb, xb, max_shift, levels=None):
    # Resolved at call time, so a preset applied after import (--preset) takes effect.
    if levels is None:
        levels = PYR_LEVELS
    # Inputs are prep_for_corr'd full tiles; slice the overlap directly.
    th, tw = A_prep.shape
    iay, iax = int(np.floor(ya)), int(np.floor(xa))
    iby, ibx = int(np.floor(yb)), int(np.floor(xb))
    y0 = max(iay, iby); x0 = max(iax, ibx)
    y1 = min(iay+th, iby+th); x1 = min(iax+tw, ibx+tw)
    if y1 - y0 < 12 or x1 - x0 < 12:
        # Tiles at current positions barely overlap — phcorr would be meaningless.
        # Sentinel tells refine_pairs_with_overlap to keep the original full-strip
        # measurement rather than endorse the current (possibly bad) position delta.
        return None, None
    Ay0, Ax0 = y0 - iay, x0 - iax
    By0, Bx0 = y0 - iby, x0 - ibx
    # Slice prepped tiles; .astype(float32) ensures contiguous copy for FFT.
    Ap = A_prep[Ay0:y1-iay, Ax0:x1-iax].astype(np.float32, copy=True)
    Bp = B_prep[By0:y1-iby, Bx0:x1-ibx].astype(np.float32, copy=True)

    pyrA = [Ap]; pyrB = [Bp]
    for _ in range(levels-1):
        if min(pyrA[-1].shape) < 24: break
        pyrA.append(zoom(pyrA[-1], 0.5, order=1))
        pyrB.append(zoom(pyrB[-1], 0.5, order=1))

    total_dy = 0.0; total_dx = 0.0
    for lvl in reversed(range(len(pyrA))):
        A_lvl = pyrA[lvl]; B_lvl = pyrB[lvl]
        scale = 1.0 / (2**lvl)
        est = (total_dy*scale, total_dx*scale)
        B_shift = np.fft.ifftn(fourier_shift(np.fft.fftn(B_lvl), est)).real.astype(np.float32)
        dy, dx = phcorr(A_lvl, B_shift)
        total_dy += dy * (2**lvl)
        total_dx += dx * (2**lvl)

    return float(np.clip(total_dy, -max_shift, max_shift)), \
           float(np.clip(total_dx, -max_shift, max_shift))

def refine_pairs_with_overlap(tiles_prepped, pos, pairs, deltas, max_shift=REFINE_MAX_SHIFT):
    new_deltas = []
    for (a, b), (dY, dX) in zip(pairs, deltas):
        A_prep, B_prep = tiles_prepped[a], tiles_prepped[b]
        ya, xa = pos[a]; yb, xb = pos[b]
        dy, dx = phcorr_overlap(A_prep, B_prep, ya, xa, yb, xb, max_shift)
        if dy is None:
            # No overlap at current positions: retain the original (pre-refine)
            # pair measurement. Endorsing current delta would self-confirm a
            # drifted placement.
            new_deltas.append((dY, dX))
        else:
            dY_new = (yb - ya) - dy
            dX_new = (xb - xa) - dx
            new_deltas.append((dY_new, dX_new))
    return new_deltas

# =========================
# Neighbor graph
# =========================
def build_grid_neighbors(coords_sorted, tiles_prepped, th, tw, bh, bw, max_drift=None):
    coords_set = set(coords_sorted)
    pairs, deltas, weights, types, metas = [], [], [], [], []
    for (r, c) in tqdm(coords_sorted, desc="Analyzing neighbors"):
        A_prep = tiles_prepped[(r, c)]

        if (r, c+1) in coords_set:
            B_prep = tiles_prepped[(r, c+1)]
            out = measure_right_band(A_prep, B_prep, bh, bw, tw, max_drift=max_drift)
            if out is not None:
                dY, dX, w, meta = out
                pairs.append(((r, c), (r, c+1)))
                deltas.append((dY, dX))
                weights.append(w)
                types.append("right")
                metas.append(meta)

        if (r+1, c) in coords_set:
            B_prep = tiles_prepped[(r+1, c)]
            out = measure_down_band(A_prep, B_prep, bh, bw, th, max_drift=max_drift)
            if out is not None:
                dY, dX, w, meta = out
                pairs.append(((r, c), (r+1, c)))
                deltas.append((dY, dX))
                weights.append(w)
                types.append("down")
                metas.append(meta)

    return pairs, deltas, weights, types, metas

# =========================
# Constraints
# =========================
def build_constraints(coords_sorted, pairs, deltas, weights, types,
                      tile_h, tile_w,
                      step_y_nom, step_x_nom,
                      max_x_drift=5.0, anchor_sigma=60.0,
                      step_y_anchor=None, curl_w=0.0):
    """
    Build weighted linear system A·u=b for u=[x0,y0,x1,y1,...].
    Anchors are placed on a *nominal grid* that uses (step_y_nom, step_x_nom).
    """
    N = len(coords_sorted)
    idx_map = {rc: i for i, rc in enumerate(coords_sorted)}

    rows, cols, vals, rhs = [], [], [], []
    pair_meta = []

    def add_eq_y(i, j, target, w):
        ww = math.sqrt(max(w, 1e-6))
        rows.extend([len(rhs), len(rhs)])
        cols.extend([2*i+1, 2*j+1])
        vals.extend([-ww, +ww])
        rhs.append(ww*target)

    def add_eq_x(i, j, target, w):
        ww = math.sqrt(max(w, 1e-6))
        rows.extend([len(rhs), len(rhs)])
        cols.extend([2*i+0, 2*j+0])
        vals.extend([-ww, +ww])
        rhs.append(ww*target)

    for ((ra, ca), (rb, cb)), (dY, dX), w, t in zip(pairs, deltas, weights, types):
        ia, ib = idx_map[(ra, ca)], idx_map[(rb, cb)]
        m = {"type": t, "dx_orig": float(dX), "dx_used": float(dX), "dx_clamped": False}

        if t == "right":
            add_eq_x(ia, ib, dX, w)
            add_eq_y(ia, ib, 0.0, w * RIGHT_DY_WEIGHT)
        else:  # down
            dx_used = float(np.clip(dX, -max_x_drift, max_x_drift))
            if abs(dx_used - dX) > 1e-6:
                m["dx_clamped"] = True
            m["dx_used"] = dx_used
            add_eq_y(ia, ib, dY, w)
            add_eq_x(ia, ib, dx_used, w)
            if step_y_anchor is not None:
                add_eq_y(ia, ib, step_y_anchor, w * VSTEP_ANCHOR_W)

        pair_meta.append(m)

    # Fix one tile at origin to remove gauge
    if coords_sorted:
        k0 = idx_map[min(coords_sorted)]
        rows.append(len(rhs)); cols.append(2*k0+1); vals.append(1.0); rhs.append(0.0)  # y=0
        rows.append(len(rhs)); cols.append(2*k0+0); vals.append(1.0); rhs.append(0.0)  # x=0

    # Weak anchors to a nominal grid using provided steps
    rmin = min(r for r, _ in coords_sorted)
    cmin = min(c for _, c in coords_sorted)
    w_anchor = 1.0 / max(anchor_sigma, 1e-6)
    for (r, c) in coords_sorted:
        yn = (r - rmin) * step_y_nom
        xn = (c - cmin) * step_x_nom
        k = idx_map[(r, c)]
        rows.append(len(rhs)); cols.append(2*k+1); vals.append(w_anchor); rhs.append(w_anchor*yn)
        rows.append(len(rhs)); cols.append(2*k+0); vals.append(w_anchor); rhs.append(w_anchor*xn)

    # ---- very light curl constraints on each 2x2 square to prevent staircase drift ----
    # Appended to the same rows/cols/vals/rhs lists; single matrix build below.
    if curl_w > 0 and N > 3:
        S = set(coords_sorted)
        def add_curl_x(iA, iR, iD, iDR, w):
            ww = math.sqrt(max(w, 1e-6))
            rows.extend([len(rhs)]*4)
            cols.extend([2*iDR, 2*iD, 2*iR, 2*iA])
            vals.extend([+ww, -ww, -ww, +ww])
            rhs.append(0.0)
        def add_curl_y(iA, iR, iD, iDR, w):
            ww = math.sqrt(max(w, 1e-6))
            rows.extend([len(rhs)]*4)
            cols.extend([2*iDR+1, 2*iR+1, 2*iD+1, 2*iA+1])
            vals.extend([+ww, -ww, -ww, +ww])
            rhs.append(0.0)
        for (r,c) in coords_sorted:
            a = (r,c); r1 = (r,c+1); d1 = (r+1,c); dr = (r+1,c+1)
            if (r1 in S) and (d1 in S) and (dr in S):
                iA, iR, iD, iDR = idx_map[a], idx_map[r1], idx_map[d1], idx_map[dr]
                add_curl_x(iA, iR, iD, iDR, curl_w)
                add_curl_y(iA, iR, iD, iDR, curl_w)

    A = lil_matrix((len(rhs), 2*N), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        A[r, c] += v
    b = np.asarray(rhs, dtype=np.float64)

    return A.tocsr(), b, idx_map, pair_meta

def solve_positions_with_constraints(A, b, coords_sorted):
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=3000)[0]
    pos = {}
    for i, rc in enumerate(coords_sorted):
        x = sol[2*i + 0]
        y = sol[2*i + 1]
        pos[rc] = np.array([y, x], dtype=np.float64)
    # shift to positive quadrant
    min_y = min(p[0] for p in pos.values()); min_x = min(p[1] for p in pos.values())
    for rc in pos:
        pos[rc] -= np.array([min_y, min_x], dtype=np.float64)
    return pos


def check_monotonicity(pos, coords_sorted, verbose=True):
    """Verify x-positions increase with col within each row, and y with row
    within each col. Returns list of (axis, a_rc, b_rc, delta) violations
    where delta is signed position difference along the expected-monotone axis.
    """
    by_row, by_col = {}, {}
    for (r, c) in coords_sorted:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)
    violations = []
    for r, cs in by_row.items():
        cs.sort()
        for i in range(len(cs) - 1):
            a, b = (r, cs[i]), (r, cs[i + 1])
            dx = pos[b][1] - pos[a][1]
            if dx <= 0:
                violations.append(("x", a, b, float(dx)))
    for c, rs in by_col.items():
        rs.sort()
        for i in range(len(rs) - 1):
            a, b = (rs[i], c), (rs[i + 1], c)
            dy = pos[b][0] - pos[a][0]
            if dy <= 0:
                violations.append(("y", a, b, float(dy)))
    if verbose and violations:
        print(f"[monotonicity] {len(violations)} violation(s):")
        for axis, a, b, delta in violations[:8]:
            print(f"  {axis}: ({int(a[0])},{int(a[1])}) -> "
                  f"({int(b[0])},{int(b[1])})  delta={delta:.1f}")
        if len(violations) > 8:
            print(f"  ... ({len(violations) - 8} more)")
    return violations


def suppress_violation_sources(violations, pairs, pos, deltas, types, weights):
    """For each monotonicity violation, zero the weight of the incident pair
    with the largest LSQR residual. Returns (new_weights_list, notes, count).
    A pair is only suppressed once per call even if it's incident to several
    violations.
    """
    incident = {}
    for i, (a, b) in enumerate(pairs):
        incident.setdefault(a, []).append(i)
        incident.setdefault(b, []).append(i)
    new_w = list(weights)
    suppressed, notes = set(), []
    for axis, a, b, delta in violations:
        cand = set(incident.get(a, [])) | set(incident.get(b, []))
        best_i, best_r = None, 0.0
        for i in cand:
            if i in suppressed:
                continue
            (ra, ca), (rb, cb) = pairs[i]
            dY, dX = deltas[i]
            ya, xa = pos[(ra, ca)]; yb, xb = pos[(rb, cb)]
            r = math.hypot((yb - ya) - dY, (xb - xa) - dX)
            if r > best_r:
                best_r = r
                best_i = i
        if best_i is not None and best_r > 0.0:
            new_w[best_i] = 0.0
            suppressed.add(best_i)
            notes.append(
                f"{axis}@{a}->{b} delta={delta:.1f}: suppressed "
                f"{types[best_i]} {pairs[best_i]} resid={best_r:.1f}"
            )
    return new_w, notes, len(suppressed)


# =========================
# Other helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _rel_to(p: Path, base: Path) -> Path:
    try:
        return p.relative_to(base)
    except Exception:
        return p


def save_debug_csvs(pos, pairs, deltas, weights, types, outdir: Path, pair_meta):
    ensure_dir(outdir)
    with open(outdir / "stitch_debug_positions.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["row","col","y","x"])
        for (r,c), (y,x) in sorted(pos.items()):
            wcsv.writerow([r,c,y,x])

    with open(outdir / "stitch_debug_pairs.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["rA","cA","rB","cB","type","dY_meas","dX_meas","weight","dx_used","dx_clamped","a_y","a_x"])
        for ((a,b),(dY,dX),w,t,m) in zip(pairs, deltas, weights, types, pair_meta):
            wcsv.writerow([
                a[0],a[1],b[0],b[1],t,dY,dX,w,
                m.get("dx_used",None),
                int(m.get("dx_clamped",False)),
                m.get("a_y",0.0), m.get("a_x",0.0)
            ])

def write_diagnose_report(outdir: Path, pos, coords_sorted, pairs, deltas,
                          weights, types, violations, step_x_nom, step_y_nom):
    """Write a short human-readable report summarizing the neighbor graph,
    weight statistics, monotonicity violations, and top-residual pairs.
    Meant for fast preflight-style iteration on parameters before a full run.
    """
    ensure_dir(outdir)

    w_right = [w for w, t in zip(weights, types) if t == "right"]
    w_down  = [w for w, t in zip(weights, types) if t == "down"]

    def _pstat(xs):
        if not xs:
            return "n=0"
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        p = lambda q: xs_sorted[min(n - 1, max(0, int(q * n)))]
        return (f"n={n}  min={xs_sorted[0]:.3f}  median={p(0.5):.3f}  "
                f"p95={p(0.95):.3f}  max={xs_sorted[-1]:.3f}")

    resids = []
    for i, ((a, b), (dY, dX)) in enumerate(zip(pairs, deltas)):
        ya, xa = pos[a]; yb, xb = pos[b]
        r = math.hypot((yb - ya) - dY, (xb - xa) - dX)
        resids.append((r, i))
    resids.sort(reverse=True)

    rows = sorted({rc[0] for rc in coords_sorted})
    cols = sorted({rc[1] for rc in coords_sorted})

    lines = []
    lines.append("Stitch diagnostic report")
    lines.append("=" * 26)
    lines.append(f"Grid: {len(cols)} cols x {len(rows)} rows, {len(coords_sorted)} tiles")
    lines.append(f"Pairs: {len(pairs)} total "
                 f"(right={sum(1 for t in types if t == 'right')}, "
                 f"down={sum(1 for t in types if t == 'down')})")
    lines.append(f"Nominal steps: step_x={step_x_nom:.1f}  step_y={step_y_nom:.1f}")
    lines.append("")
    lines.append("Weights:")
    lines.append(f"  right: {_pstat(w_right)}")
    lines.append(f"  down : {_pstat(w_down)}")
    lines.append("")
    def _fmt_rc(rc):
        return f"({int(rc[0])},{int(rc[1])})"

    lines.append(f"Monotonicity violations: {len(violations)}")
    for axis, a, b, delta in violations[:40]:
        lines.append(f"  {axis}: {_fmt_rc(a)} -> {_fmt_rc(b)}  delta={delta:+.1f}")
    if len(violations) > 40:
        lines.append(f"  ... ({len(violations) - 40} more)")
    lines.append("")
    lines.append("Top-10 residual pairs (|measured - solved|):")
    for r, i in resids[:10]:
        (a, b) = pairs[i]; (dY, dX) = deltas[i]
        ya, xa = pos[a]; yb, xb = pos[b]
        lines.append(f"  {types[i]} {_fmt_rc(a)}->{_fmt_rc(b)}  "
                     f"meas=({dY:+.1f},{dX:+.1f})  "
                     f"solved=({yb - ya:+.1f},{xb - xa:+.1f})  "
                     f"resid={r:.1f}  w={weights[i]:.3f}")

    (outdir / "stitch_diagnose_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def save_debug_plot(tiles_raw, pos, pairs, deltas, weights, types, outpath: Path, pair_meta):
    ensure_dir(outpath.parent)
    coords_sorted = sorted(pos.keys())
    th, tw = next(iter(tiles_raw.values())).shape

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    for rc in coords_sorted:
        y, x = pos[rc]
        ax.add_patch(plt.Rectangle((x,y), tw, th, fill=False, lw=0.8, color='k', alpha=0.6))
        ax.text(x+4, y+12, f"{rc}", fontsize=7)
    for ((a,b), (dY,dX), w, t, m) in zip(pairs, deltas, weights, types, pair_meta):
        ya, xa = pos[a]; yb, xb = pos[b]
        color = (0.0, 0.6, 0.0) if t == "right" else (0.0, 0.2, 0.8)
        if t == "down" and m.get("dx_clamped", False):
            color = (0.5, 0.0, 0.8)
        ax.arrow(xa, ya, xb - xa, yb - ya, head_width=0, head_length=0,
                 length_includes_head=True, color=color, alpha=0.85, lw=1.1)
    ax.invert_yaxis(); ax.axis('equal'); ax.axis('off'); plt.tight_layout()
    plt.savefig(outpath, dpi=220); plt.close()

def save_worst_overlap_crops(tiles_raw, pos, pairs, outdir: Path, topN=8):
    if topN <= 0: return
    ensure_dir(outdir)
    th, tw = next(iter(tiles_raw.values())).shape
    scored = []
    for (a,b) in pairs:
        ya, xa = pos[a]; yb, xb = pos[b]
        iay, iax = int(np.floor(ya)), int(np.floor(xa))
        iby, ibx = int(np.floor(yb)), int(np.floor(xb))
        y0 = max(iay, iby); x0 = max(iax, ibx)
        y1 = min(iay+th, iby+th); x1 = min(iax+tw, ibx+tw)
        area = max(0, y1 - y0) * max(0, x1 - x0)
        scored.append(((a,b), area))
    scored.sort(key=lambda t: t[1])
    for k, ((a,b), area) in enumerate(scored[:topN]):
        A = tiles_raw[a]; B = tiles_raw[b]
        iay, iax = int(np.floor(pos[a][0])), int(np.floor(pos[a][1]))
        iby, ibx = int(np.floor(pos[b][0])), int(np.floor(pos[b][1]))
        y0 = max(iay, iby); x0 = max(iax, ibx)
        y1 = min(iay+th, iby+th); x1 = min(iax+tw, ibx+tw)
        if y1 - y0 < 8 or x1 - x0 < 8: continue
        Ay0, Ax0 = y0 - iay, x0 - iax
        By0, Bx0 = y0 - iby, x0 - ibx
        Apatch = A[Ay0:y1-iay, Ax0:x1-iax]
        Bpatch = B[By0:y1-iby, Bx0:x1-ibx]
        h = min(Apatch.shape[0], Bpatch.shape[0]); w = min(Apatch.shape[1], Bpatch.shape[1])
        Apatch, Bpatch = Apatch[:h,:w], Bpatch[:h,:w]
        a8 = (255*np.clip((Apatch - Apatch.min())/(np.ptp(Apatch)+1e-6),0,1)).astype(np.uint8)
        b8 = (255*np.clip((Bpatch - Bpatch.min())/(np.ptp(Bpatch)+1e-6),0,1)).astype(np.uint8)
        diff = (np.abs(a8.astype(np.int16)-b8.astype(np.int16))).astype(np.uint8)
        rgb = np.stack([a8,b8,diff], axis=-1)
        Image.fromarray(rgb).save(outdir / f"overlap_{k+1}_A({a[0]},{a[1]})_B({b[0]},{b[1]}).png")

LABEL_TILE_LIMIT = 500   # disable dense labeling above this many tiles
LABEL_SKIP = 10          # draw every nth label if too many tiles

def overlay_labels(m16, pos, coords_sorted, th, tw, sparse=None):
    """
    Draw (row,col) and index labels on top of the mosaic image.
    Automatically switches to sparse labeling for large mosaics.
    """
    img = Image.fromarray((m16 >> 8).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", LABEL_SIZE)
    except:
        font = ImageFont.load_default()

    N = len(coords_sorted)
    use_sparse = sparse if sparse is not None else (N > LABEL_TILE_LIMIT)

    for idx, rc in enumerate(coords_sorted):
        y, x = pos[rc]
        iy, ix = int(round(y)), int(round(x))

        # Skip most tiles if sparse labeling is enabled
        if use_sparse and (rc[0] % LABEL_SKIP != 0 or rc[1] % LABEL_SKIP != 0):
            continue

        if SHOW_ROWCOL:
            draw.text((ix+4, iy+4), f"({rc[0]},{rc[1]})", fill=LABEL_COLOR, font=font)
        if SHOW_INDEX:
            draw.text((ix + tw//2, iy + th//2), str(idx),
                      fill=LABEL_COLOR, font=font, anchor="mm")
    return img

# =========================
# Prep-variant calibration
# =========================
def calibrate_prep_variant(
    tiles_rc: dict,
    coords_sorted: list,
    th: int,
    tw: int,
    bh: int,
    bw: int,
    drift_gate: float,
    sample: int = 40,
    rng_seed: int = 0,
) -> tuple[str, dict]:
    """Sample a handful of neighbor pairs, prep them with each variant, and
    pick the variant that maximises median_weight * exp(-orth_std / 10).

    Returns (chosen_variant, report) where report is a dict of per-variant
    (median_weight, orth_std_px, n_pairs, score) tuples. Score is the product
    that was maximised; higher is better.
    """
    variants = ("edge", "wide_hp", "flat", "dog")
    rc_set = set(coords_sorted)

    # Collect candidate right/down pairs from the grid.
    right_pairs = [((r, c), (r, c + 1)) for (r, c) in coords_sorted if (r, c + 1) in rc_set]
    down_pairs  = [((r, c), (r + 1, c)) for (r, c) in coords_sorted if (r + 1, c) in rc_set]

    rng = np.random.default_rng(rng_seed)
    def _sample(pool, n):
        if not pool:
            return []
        n = min(n, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    n_per_dir = max(1, sample // 2)
    chosen_right = _sample(right_pairs, n_per_dir)
    chosen_down  = _sample(down_pairs,  n_per_dir)

    # Unique tiles touched (minimises prep work per variant).
    touched = set()
    for a, b in chosen_right + chosen_down:
        touched.add(a); touched.add(b)
    touched = sorted(touched)

    if not touched:
        print("[prep-calib] no neighbor pairs available; defaulting to 'edge'")
        return "edge", {}

    report = {}
    best_name = None
    best_score = -np.inf
    for v in variants:
        prepped = {rc: prep_for_corr(tiles_rc[rc], variant=v).astype(np.float32, copy=False)
                   for rc in touched}
        weights = []
        orth = []  # orthogonal-axis drift (dY for right-pairs, dX for down-pairs)
        for a, b in chosen_right:
            out = measure_right_band(prepped[a], prepped[b], bh, bw, tw, max_drift=drift_gate)
            if out:
                dY, dX, w, _ = out
                weights.append(float(w))
                orth.append(float(dY))
        for a, b in chosen_down:
            out = measure_down_band(prepped[a], prepped[b], bh, bw, th, max_drift=drift_gate)
            if out:
                dY, dX, w, _ = out
                weights.append(float(w))
                orth.append(float(dX))

        n = len(weights)
        if n == 0:
            report[v] = (0.0, float("nan"), 0, -np.inf)
            continue
        med_w = float(np.median(weights))
        orth_sd = float(np.std(orth)) if orth else float("nan")
        score = med_w * float(np.exp(-orth_sd / 10.0)) if np.isfinite(orth_sd) else med_w
        report[v] = (med_w, orth_sd, n, score)
        if score > best_score:
            best_score = score
            best_name = v
        # Free per-variant prepped dict before next iter.
        del prepped

    if best_name is None:
        return "edge", report

    # Tie-break: if 'edge' is within 10% of best, prefer 'edge' for conservatism.
    edge_score = report.get("edge", (0, 0, 0, -np.inf))[3]
    if edge_score >= 0.9 * best_score and best_name != "edge":
        best_name = "edge"

    return best_name, report


# =========================
# Stitch (stage-initialized grid + all refinements)
# =========================
def stitch(files: list[Path], base_dir: Path):
    # ---- Load tiles, stage coords, and pixel sizes in a single pass
    tiles_raw = {}
    stage_xy = []
    file_list = []
    px_sizes = []
    cache_dir = None if globals().get("_CLI_NO_CACHE", False) else (base_dir / "stitch_cache")
    print(f"[load] Reading tiles... (cache: {'off' if cache_dir is None else cache_dir})")
    for p in tqdm(files, desc="Loading BCF files"):
        p = p.resolve()  # <-- ensure absolute / canonical path
        try:
            raw, sx, sy, ps = _load_bcf_cached(p, cache_dir)
        except Exception as e:
            print(f"[warn] Could not load {p.name}: {e}")
            continue
        tiles_raw[p] = raw
        stage_xy.append((sx, sy))
        file_list.append(p)
        if ps and np.isfinite(ps) and ps > 0:
            px_sizes.append(ps)
    if not tiles_raw:
        raise RuntimeError("No tiles loaded.")

    th, tw = next(iter(tiles_raw.values())).shape
    bh = max(8, int(round(BAND_FRAC * th)))
    bw = max(8, int(round(BAND_FRAC * tw)))

    # ---- Convert median pixel size → px/stage-unit (e.g. µm/px → px/µm)
    px_per_stage = (1.0 / float(np.median(px_sizes))) if px_sizes else None
    # sanity: discard absurd scales (units mixups)
    if px_per_stage is not None and not (1e-4 <= px_per_stage <= 1e4):
        px_per_stage = None

    # ---- Build grid (prevents column collapse) by quantizing stage coords
    row_centers, col_centers, row_assign, col_assign, dy_stage, dx_stage = build_grid_from_stage_quantized(stage_xy)
    print(f"[stage-grid] rows={len(row_centers)} cols={len(col_centers)}")

    # ---- Use pixel size to compute stage→pixel nominal steps if available
    if px_per_stage is not None:
        step_x_nom = float(dx_stage * px_per_stage)
        step_y_nom = float(dy_stage * px_per_stage)
    else:
        step_x_nom = tw * (1.0 - OVERLAP_FRAC)
        step_y_nom = th * (1.0 - OVERLAP_FRAC)
    print(f"[anchors] stage-derived steps (if available) ~ X:{step_x_nom:.2f} px  Y:{step_y_nom:.2f} px")

    # ---- Collect unique (r,c) → representative path
    rc_to_path = {}
    for r, c, p in zip(row_assign, col_assign, file_list):
        if (r, c) not in rc_to_path:
            rc_to_path[(r, c)] = p
        else:
            print(f"[warn] multiple tiles mapped to grid cell ({r}, {c}), "
                  f"keeping first: {rc_to_path[(r, c)].name}, skipping {p.name}")

    tiles_rc = {}
    coords = []
    for rc, p in rc_to_path.items():
        tiles_rc[rc] = tiles_raw[p]
        coords.append(rc)
    coords_sorted = sorted(coords)
    # Free orphaned tiles (those dropped by the "multiple tiles mapped to grid cell" check).
    del tiles_raw

    # ---- Adaptive drift gate: absolute ceiling for per-strip phase-corr shifts.
    # Replaces the hardcoded MAX_STRIP_DRIFT=15 which dropped legitimate data on jittery stages.
    # median_inliers still handles statistical outliers via MAD.
    drift_gate = max(0.25 * max(step_x_nom, step_y_nom), 50.0)
    print(f"[gate] adaptive drift_gate = {drift_gate:.1f} px (was MAX_STRIP_DRIFT={MAX_STRIP_DRIFT})")

    # ---- Prep-variant selection ('auto' calibrates on a small sample).
    cli_variant = globals().get("_CLI_PREP_VARIANT", None)
    if cli_variant == "auto":
        chosen, report = calibrate_prep_variant(
            tiles_rc, coords_sorted, th, tw, bh, bw, drift_gate, sample=40
        )
        print("[prep-calib] per-variant (median_weight, orth_std_px, n, score):")
        for v, stats in report.items():
            mw, osd, n, sc = stats
            print(f"[prep-calib]   {v:8s} med_w={mw:.3f}  orth_std={osd:.2f} px  n={n}  score={sc:.3f}")
        print(f"[prep-calib] selected variant: {chosen!r}")
        globals()["PREP_VARIANT"] = chosen

    # ---- Prep each tile ONCE for phase-correlation; slice from the cache.
    #      prep_for_corr is the dominant hot-spot (used to run ~50×/tile on preset C).
    print(f"[prep] Preparing tiles for phase correlation (variant={PREP_VARIANT!r})...")
    tiles_prepped = {}
    for rc in tqdm(coords_sorted, desc="prep_for_corr"):
        tiles_prepped[rc] = prep_for_corr(tiles_rc[rc]).astype(np.float32, copy=False)

    # ---- Build neighbor graph (RIGHT/DOWN) and measure deltas
    pairs, deltas, weights, types, metas = build_grid_neighbors(
        coords_sorted, tiles_prepped, th, tw, bh, bw, max_drift=drift_gate
    )
    if not pairs:
        raise RuntimeError("No neighbor pairs were found from the grid.")
    print(f"[graph] measured {len(pairs)} neighbor pairs")

    # --- Derive measured nominal steps from data (robust) and adaptive tolerances
    right_dx = [dX for t,(dY,dX) in zip(types, deltas) if t == "right"]
    down_dy  = [dY for t,(dY,dX) in zip(types, deltas) if t == "down"]
    sx_med, sx_scale = _robust_medmad(right_dx)
    sy_med, sy_scale = _robust_medmad(down_dy)
    if sx_med is not None: step_x_nom = sx_med
    if sy_med is not None: step_y_nom = sy_med
    tol_right_init = max(0.25, 3.0 * (sx_scale or 1.0) / max(step_x_nom, 1.0))
    tol_down_init  = max(0.25, 3.0 * (sy_scale or 1.0) / max(step_y_nom, 1.0))

    # --- Gate inconsistent neighbor deltas before solving (adaptive)
    mask = filter_pairs_by_nominal(
        pairs, deltas, types,
        step_x_nom=step_x_nom, step_y_nom=step_y_nom,
        tol_right=tol_right_init, tol_down=tol_down_init,
        max_dx_down=MAX_VERT_X_DRIFT,  # clamps
        max_dy_right=MAX_RIGHT_Y_DRIFT
    )
    if not any(mask):
        print("[gate] WARNING: all pairs rejected by nominal gate; skipping gate for this dataset.")
    else:
        pairs = [p for p, m in zip(pairs, mask) if m]
        deltas = [d for d, m in zip(deltas, mask) if m]
        weights = [w for w, m in zip(weights, mask) if m]
        types = [t for t, m in zip(types, mask) if m]
        metas = [x for x, m in zip(metas, mask) if m]
        print(f"[gate] kept {len(pairs)} pairs out of {len(mask)}")

    # ---- Optional hard weight floor (opt-in via --weight-floor).
    wfloor = float(globals().get("_CLI_WEIGHT_FLOOR", 0.0))
    if wfloor > 0.0 and pairs:
        fmask = [w >= wfloor for w in weights]
        dropped = sum(1 for m in fmask if not m)
        if dropped and any(fmask):
            pairs   = [p for p, m in zip(pairs,   fmask) if m]
            deltas  = [d for d, m in zip(deltas,  fmask) if m]
            weights = [w for w, m in zip(weights, fmask) if m]
            types   = [t for t, m in zip(types,   fmask) if m]
            metas   = [x for x, m in zip(metas,   fmask) if m]
            print(f"[weight-floor] dropped {dropped} pairs below {wfloor:.3f}; kept {len(pairs)}")
        elif not any(fmask):
            print(f"[weight-floor] WARNING: floor {wfloor:.3f} would drop ALL pairs; ignoring.")

    # ---- Solve initial placement with constraints, using measured nominal steps
    print("[constraints] Building linear system...")
    A, b, idx_map, pair_meta = build_constraints(
        coords_sorted, pairs, deltas, weights, types,
        th, tw,
        step_y_nom=step_y_nom, step_x_nom=step_x_nom,
        step_y_anchor=step_y_nom,
        max_x_drift=MAX_VERT_X_DRIFT, anchor_sigma=anchor_sigma,
        curl_w=0.05
    )
    print("[constraints] Solving with LSQR...")
    pos = solve_positions_with_constraints(A, b, coords_sorted)

    # ---- Residual reweighting pass, then resolve once before iterations
    weights = reweight_by_residual(pos, pairs, deltas, types, weights)
    A, b, _, pair_meta = build_constraints(
        coords_sorted, pairs, deltas, weights, types,
        th, tw,
        step_y_nom=step_y_nom, step_x_nom=step_x_nom,
        step_y_anchor=step_y_nom,
        max_x_drift=MAX_VERT_X_DRIFT, anchor_sigma=anchor_sigma,
        curl_w=0.05
    )
    pos = solve_positions_with_constraints(A, b, coords_sorted)

    # ---- Monotonicity sanity check on the initial solve.
    # Cheap post-solve guard: within each row x must increase with col, within
    # each column y must increase with row. Violations flag broken pairs whose
    # residuals dominate the cascade. Attempt one round of targeted pair
    # suppression + resolve; don't spin if the fix didn't stick.
    mono_viol = check_monotonicity(pos, coords_sorted)

    # Diagnose mode: write the report and exit before refinement/composite.
    if globals().get("_CLI_DIAGNOSE", False):
        debug_dir = base_dir / DEBUG_OUTDIR
        ensure_dir(debug_dir)
        save_debug_csvs(pos, pairs, deltas, weights, types, debug_dir, pair_meta)
        write_diagnose_report(
            debug_dir, pos, coords_sorted, pairs, deltas, weights, types,
            mono_viol, step_x_nom, step_y_nom,
        )
        print(f"[diagnose] wrote {debug_dir / 'stitch_diagnose_report.txt'}")
        print("[diagnose] skipping refinement + composite.")
        return

    if mono_viol:
        weights, notes, nsupp = suppress_violation_sources(
            mono_viol, pairs, pos, deltas, types, weights
        )
        if nsupp:
            print(f"[monotonicity] suppressed {nsupp} suspect pair(s):")
            for n in notes[:8]:
                print(f"  {n}")
            A, b, _, pair_meta = build_constraints(
                coords_sorted, pairs, deltas, weights, types,
                th, tw,
                step_y_nom=step_y_nom, step_x_nom=step_x_nom,
                step_y_anchor=step_y_nom,
                max_x_drift=MAX_VERT_X_DRIFT, anchor_sigma=anchor_sigma,
                curl_w=0.05
            )
            pos = solve_positions_with_constraints(A, b, coords_sorted)
            mono_viol2 = check_monotonicity(pos, coords_sorted, verbose=False)
            print(f"[monotonicity] after suppress+resolve: {len(mono_viol2)} violation(s)")

    # ---- Decide whether iter-0 needs permissive settings.
    # If the initial placement's median residual is already tight, skip the permissive gate.
    _resid = []
    for ((ra, ca), (rb, cb)), (dY, dX), t in zip(pairs, deltas, types):
        ya, xa = pos[(ra, ca)]; yb, xb = pos[(rb, cb)]
        _resid.append(abs((yb - ya) - dY))
        _resid.append(abs((xb - xa) - dX))
    r0_median = float(np.median(_resid)) if _resid else float("inf")
    tight_start = r0_median < 0.05 * max(step_x_nom, step_y_nom)
    print(f"[refine] initial median residual = {r0_median:.2f} px ({'tight' if tight_start else 'permissive'} start)")

    # ---- Skip refine entirely when the initial solve is already tight.
    # Observed failure: on a clean initial solve (median residual < 2 px on a
    # ~900 px step), the refine loop can still move individual tiles by
    # 100+ px via overlap-phcorr compounding through the LSQR solve, opening
    # visible seams. When the initial is this good, refine is strictly net-
    # negative. The _i suffix vars are still needed by the final reweight-
    # and-resolve pass, so seed them from the current graph.
    iters_to_run = 0 if tight_start else REFINE_ITERS
    if tight_start:
        print(f"[refine] tight start - skipping {REFINE_ITERS} refine iter(s)")
    pairs_i, deltas_i, weights_i, types_i = pairs, deltas, weights, types
    anch_sigma_iter = anchor_sigma  # used by the post-loop final resolve

    # ---- Refine on overlaps (multiscale), re-solve (repeat)
    REFINE_CONVERGED_PX = 0.1   # max-tile position change to declare convergence
    REFINE_DAMAGE_PX    = 30.0  # any per-iter tile jump this big on a tight-
                                # start run means refine is corrupting positions
    for it in range(iters_to_run):
        print(f"[refine] Iter {it + 1}/{REFINE_ITERS}")
        pos_prev = {rc: pos[rc].copy() for rc in coords_sorted}

        # Permissive early iterations (bigger allowed correction), tighter later iterations.
        # If the initial placement was already tight, skip iter-0's extra permissiveness.
        if it == 0 and not tight_start:
            rmax = max(REFINE_MAX_SHIFT, 40)  # allow bigger fix first
            tol_right, tol_down = 0.35, 0.35
            anch_sigma_iter = max(anchor_sigma, 120.0)  # weaker anchor
        elif it <= 1:
            rmax = max(REFINE_MAX_SHIFT, 25)
            tol_right, tol_down = 0.30, 0.30
            anch_sigma_iter = anchor_sigma
        else:
            rmax = max(int(REFINE_MAX_SHIFT * 0.5), 10)
            tol_right, tol_down = 0.25, 0.25
            anch_sigma_iter = max(40.0, anchor_sigma * 0.75)

        # measure again around current pose (use this iteration's rmax)
        deltas = refine_pairs_with_overlap(tiles_prepped, pos, pairs, deltas, max_shift=rmax)

        # Adaptive tolerances from the current iter's refinement spread.
        # NOTE: step_x_nom / step_y_nom are NOT re-assigned here. The refine
        # deltas are a function of current positions (overlap-phcorr on
        # ambiguous texture endorses wherever the tiles currently sit), so
        # updating step_nom from them causes the anchor to chase the solution
        # and drift cumulates across cols/rows. The initial full-strip
        # measurements set a stable reference at line ~1035; keep it.
        right_dx = [dX for t,(dY,dX) in zip(types, deltas) if t == "right"]
        down_dy  = [dY for t,(dY,dX) in zip(types, deltas) if t == "down"]
        _,        sx_scale = _robust_medmad(right_dx)
        _,        sy_scale = _robust_medmad(down_dy)
        tol_right = max(0.20, min(0.35, 3.0 * (sx_scale or 1.0) / max(step_x_nom, 1.0)))
        tol_down  = max(0.20, min(0.35, 3.0 * (sy_scale or 1.0) / max(step_y_nom, 1.0)))

        # re-gate using the latest deltas (dropping outliers that slipped in)
        mask = filter_pairs_by_nominal(
            pairs, deltas, types,
            step_x_nom=step_x_nom, step_y_nom=step_y_nom,
            tol_right=tol_right, tol_down=tol_down,
            max_dx_down=MAX_VERT_X_DRIFT,
            max_dy_right=MAX_RIGHT_Y_DRIFT
        )
        if any(mask):
            pairs_i = [p for p, m in zip(pairs, mask) if m]
            deltas_i = [d for d, m in zip(deltas, mask) if m]
            weights_i = [w for w, m in zip(weights, mask) if m]
            types_i = [t for t, m in zip(types, mask) if m]
        else:
            # fall back to all pairs if the gate went too hard
            pairs_i, deltas_i, weights_i, types_i = pairs, deltas, weights, types
            print("[gate] iteration gate dropped all pairs; reverting to full set.")

        # build with current (possibly gated) graph + measured steps + curls
        A, b, _, pair_meta = build_constraints(
            coords_sorted, pairs_i, deltas_i, weights_i, types_i,
            th, tw,
            step_y_nom=step_y_nom, step_x_nom=step_x_nom,
            step_y_anchor=step_y_nom,
            max_x_drift=MAX_VERT_X_DRIFT, anchor_sigma=anch_sigma_iter,
            curl_w=0.05
        )
        pos = solve_positions_with_constraints(A, b, coords_sorted)

        # Convergence check: max per-tile displacement vs. start of this iter.
        # Require at least 2 iters so we always get one permissive + one tight pass.
        max_delta = max(float(np.linalg.norm(pos[rc] - pos_prev[rc])) for rc in coords_sorted)
        print(f"[refine] iter {it + 1} max_delta = {max_delta:.3f} px")

        # Damage abort: a big per-iter tile jump on a tight initial solve is
        # always refine corruption, not legitimate correction. Revert to the
        # pre-iter positions and stop iterating.
        if tight_start and max_delta > REFINE_DAMAGE_PX:
            print(f"[refine] iter {it + 1} moved a tile by {max_delta:.1f} px "
                  f"(> {REFINE_DAMAGE_PX} px); reverting and aborting refine")
            pos = pos_prev
            break

        if it >= 1 and max_delta < REFINE_CONVERGED_PX:
            print(f"[refine] Converged after iter {it + 1} (max_delta={max_delta:.3f} px < {REFINE_CONVERGED_PX})")
            break

    # ---- Final residual reweight + resolve (moved out of the per-iter loop).
    # Huber-like reweight effect saturates quickly; one post-loop pass is enough.
    weights_i = reweight_by_residual(pos, pairs_i, deltas_i, types_i, weights_i)
    A, b, _, pair_meta = build_constraints(
        coords_sorted, pairs_i, deltas_i, weights_i, types_i,
        th, tw,
        step_y_nom=step_y_nom, step_x_nom=step_x_nom,
        step_y_anchor=step_y_nom,
        max_x_drift=MAX_VERT_X_DRIFT, anchor_sigma=anch_sigma_iter,
        curl_w=0.05
    )
    pos = solve_positions_with_constraints(A, b, coords_sorted)

    # ---- Post-refine monotonicity check: same guard as the pre-refine one,
    # but positions may have drifted during overlap-phcorr refinement. Attempt
    # targeted suppression + one resolve if violations exist.
    mono_viol_post = check_monotonicity(pos, coords_sorted)
    if mono_viol_post:
        weights_i, notes, nsupp = suppress_violation_sources(
            mono_viol_post, pairs_i, pos, deltas_i, types_i, weights_i
        )
        if nsupp:
            print(f"[monotonicity/post-refine] suppressed {nsupp} suspect pair(s):")
            for n in notes[:8]:
                print(f"  {n}")
            A, b, _, pair_meta = build_constraints(
                coords_sorted, pairs_i, deltas_i, weights_i, types_i,
                th, tw,
                step_y_nom=step_y_nom, step_x_nom=step_x_nom,
                step_y_anchor=step_y_nom,
                max_x_drift=MAX_VERT_X_DRIFT, anchor_sigma=anch_sigma_iter,
                curl_w=0.05
            )
            pos = solve_positions_with_constraints(A, b, coords_sorted)
            mono_viol_post2 = check_monotonicity(pos, coords_sorted, verbose=False)
            print(f"[monotonicity/post-refine] after suppress+resolve: "
                  f"{len(mono_viol_post2)} violation(s)")

    # ---- Refresh per-pair slope fits against the final positions.
    # metas was populated once at the start from pre-refinement strip measurements, so the
    # shear at composite time would otherwise use stale slopes. Positions are NOT re-solved;
    # this only gives fresh slope data for the shear step and the debug CSVs.
    print("[refine] Refreshing slope fits against final positions...")
    pairs_fresh, deltas_fresh, weights_fresh, types_fresh, metas_fresh = build_grid_neighbors(
        coords_sorted, tiles_prepped, th, tw, bh, bw, max_drift=drift_gate
    )
    # Only swap in if we got usable measurements; fall back to stale metas otherwise.
    if pairs_fresh:
        pairs, deltas, weights, types, metas = pairs_fresh, deltas_fresh, weights_fresh, types_fresh, metas_fresh
    else:
        print("[refine] Fresh slope refit produced no pairs; keeping stale metas.")

    # Release prep cache — composite uses raw tiles only.
    del tiles_prepped

    # ---- Tiny per-tile shear from measured slopes
    tile_shear = {rc: {"shx":0.0, "shy":0.0, "n_x":0, "n_y":0} for rc in coords_sorted}
    for ((ra,ca),(rb,cb)), meta in zip(pairs, metas):
        if not meta:
            continue
        if meta.get("type") == "right":
            shy = float(np.clip(meta.get("a_y",0.0), -SHEAR_CAP, SHEAR_CAP))
            tile_shear[(rb,cb)]["shy"] += shy; tile_shear[(rb,cb)]["n_y"] += 1
        elif meta.get("type") == "down":
            shx = float(np.clip(meta.get("a_x",0.0), -SHEAR_CAP, SHEAR_CAP))
            tile_shear[(rb,cb)]["shx"] += shx; tile_shear[(rb,cb)]["n_x"] += 1
    for rc, acc in tile_shear.items():
        if acc["n_y"] > 0: acc["shy"] /= acc["n_y"]
        if acc["n_x"] > 0: acc["shx"] /= acc["n_x"]
        acc["shy"] = float(np.clip(acc["shy"], -SHEAR_CAP, SHEAR_CAP))
        acc["shx"] = float(np.clip(acc["shx"], -SHEAR_CAP, SHEAR_CAP))

    # ---- Orientation control
    def median_right_dx(deltas, types):
        vals = [dx for t,(_,dx) in zip(types, deltas) if t=="right"]
        if not vals: return 0.0
        v = np.asarray(vals, float)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) + 1e-6
        z = np.abs(v - med) / (1.4826*mad)
        v = v[z < 2.5] if np.any(z < 2.5) else v
        return float(np.median(v))

    do_flip = False
    if ORIENT == "top-left":
        do_flip = True
    elif ORIENT == "auto":
        med_dx = median_right_dx(deltas, types)
        do_flip = (med_dx < 0)

    if do_flip:
        max_x = max(p[1] for p in pos.values())
        for rc in list(pos.keys()):
            y, x = pos[rc]
            pos[rc] = np.array([y, max_x - x], dtype=np.float64)

    # ---- Write a manifest for EDS extraction
    # Keep rc_to_path absolute internally, but write relative strings to manifest:
    write_tile_manifest(
        base_dir,
        coords_sorted,
        pos,
        {rc: _rel_to(p, base_dir) for rc, p in rc_to_path.items()},  # relative only for manifest
        tile_shear,
        th, tw
    )

    # ---- EDS low-res previews (metadata-driven)
    if args.eds_preview:
        # Decide which elements
        req = args.eds_preview.strip().lower()
        if req in ("auto", "all"):
            # discover from metadata (fast scan of up to 200 tiles)
            file_paths = [rc_to_path[rc] for rc in coords_sorted]
            elements = discover_elements_from_metadata(file_paths, sample_n=200)
            if not elements:
                print("[eds-preview] No elements discovered in metadata; skipping.")
            else:
                print(f"[eds-preview] Using elements from metadata: {elements}")
        else:
            # explicit comma list
            elements = [s.strip() for s in args.eds_preview.split(",") if s.strip()]
            elements = [e for e in elements if e in EDS_LINE_KEV]
            if not elements:
                print("[eds-preview] No valid elements in list; skipping.")
                elements = []

        if elements:
            cache_enabled = bool(getattr(args, "eds_cache", False))

            # Cache first — the tile-stream pipeline runs here. When on, we'll
            # derive previews from the cache (no second BCF pass).
            if cache_enabled:
                try:
                    _build_or_refresh_eds_cache(
                        base_dir=base_dir,
                        elements=elements,
                        width_eV=float(args.eds_preview_width),
                        ds=int(args.eds_cache_ds),
                        on_restitch=str(args.eds_cache_on_restitch),
                        shift_method=str(args.eds_shift_method),
                        max_workers=int(args.eds_cache_workers),
                        flush_every=int(args.eds_cache_flush_every),
                        cache_dir_override=(str(args.eds_cache_dir).strip() or None),
                        coating_elements=tuple(args.coating_elements),
                    )
                except Exception as e:
                    print(f"[eds-cache] Skipped (error: {e})")

            # Previews: derive from cache if it's valid for the request, else
            # fall back to the standalone tile-stream preview pipeline.
            import eds_cache as _edsc  # local import; function-level usage
            cache_dir_pv = _edsc.resolve_cache_dir(base_dir)
            manifest_csv = base_dir / "stitch_manifest.csv"
            cache_is_good = False
            if cache_enabled and manifest_csv.exists():
                try:
                    cache_is_good = _edsc.is_valid_for(
                        cache_dir_pv, manifest_csv,
                        required_elements=elements,
                        required_width_eV=float(args.eds_preview_width),
                    )
                except Exception:
                    cache_is_good = False

            if cache_is_good:
                try:
                    _edsc.write_preview_pngs(
                        cache_dir_pv, out_dir=base_dir,
                        preview_ds=int(args.eds_preview_ds),
                        elements=elements,
                    )
                except Exception as e:
                    print(f"[eds-preview] Cache->preview failed ({e}); running tile-stream fallback.")
                    try:
                        build_eds_previews_fast(
                            coords_sorted=coords_sorted, pos=pos,
                            rc_to_path=rc_to_path, th=th, tw=tw,
                            preview_elements=elements,
                            width_eV=int(args.eds_preview_width),
                            ds=int(args.eds_preview_ds),
                            out_dir=base_dir,
                            use_feather=bool(args.eds_preview_feather),
                            max_workers=int(args.eds_preview_workers),
                        )
                    except Exception as e2:
                        print(f"[eds-preview] Fallback also failed: {e2}")
            else:
                try:
                    build_eds_previews_fast(
                        coords_sorted=coords_sorted, pos=pos,
                        rc_to_path=rc_to_path, th=th, tw=tw,
                        preview_elements=elements,
                        width_eV=int(args.eds_preview_width),
                        ds=int(args.eds_preview_ds),
                        out_dir=base_dir,
                        use_feather=bool(args.eds_preview_feather),
                        max_workers=int(args.eds_preview_workers),
                    )
                except Exception as e:
                    print(f"[eds-preview] Skipped (error: {e})")


    # ---- Debug outputs
    debug_dir = base_dir / DEBUG_OUTDIR
    if DEBUG:
        ensure_dir(debug_dir)
        save_debug_csvs(pos, pairs, deltas, weights, types, debug_dir, pair_meta)
        save_debug_plot(tiles_rc, pos, pairs, deltas, weights, types,
                        debug_dir / "stitch_debug_positions.png", pair_meta)
        save_worst_overlap_crops(tiles_rc, pos, pairs, debug_dir, topN=SAVE_WORST_OVERLAPS)

    # ---- Composite (subpixel shift + optional tiny shear + Tukey feather)
    H = int(np.ceil(max(p[0] for p in pos.values()) + th))
    W = int(np.ceil(max(p[1] for p in pos.values()) + tw))
    accum  = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    from scipy.signal.windows import tukey as tukey1d
    wy = tukey1d(th, alpha=TUKEY_ALPHA).astype(np.float32)
    wx = tukey1d(tw, alpha=TUKEY_ALPHA).astype(np.float32)
    w_win = np.outer(wy, wx)

    print("[composite] Blending...")
    for rc, raw in tqdm(tiles_rc.items(), desc="Compositing tiles"):
        yx = pos[rc]
        iy, ix = np.floor(yx).astype(int)
        frac = yx - np.floor(yx)

        # Skip the FFT shift when the fractional part is tiny — contributes <0.02 px of blur.
        if max(abs(frac[0]), abs(frac[1])) > 0.02:
            subpix = np.fft.ifftn(fourier_shift(np.fft.fftn(raw), frac)).real.astype(np.float32)
        else:
            subpix = raw

        if SHEAR_ENABLE:
            shx = tile_shear[rc]["shx"]
            shy = tile_shear[rc]["shy"]
            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx],
                              [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                subpix = affine_transform(
                    subpix, Minv, offset=0.0, order=1, mode='nearest', cval=0.0, prefilter=False
                ).astype(np.float32)

        y2, x2 = iy + th, ix + tw
        sy0, sx0 = max(0, -iy), max(0, -ix)
        sy1, sx1 = th - max(0, y2 - H), tw - max(0, x2 - W)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        patch = subpix[sy0:sy1, sx0:sx1]
        ww    = w_win[sy0:sy1, sx0:sx1]
        accum[iy+sy0:iy+sy1, ix+sx0:ix+sx1]  += patch * ww
        weight[iy+sy0:iy+sy1, ix+sx0:ix+sx1] += ww

    mosaic = np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)

    # ---- Save next to data
    ensure_dir(base_dir)
    vals = mosaic[weight > 0]
    lo, hi = np.percentile(vals, [LOW_PCT, HI_PCT]) if vals.size else (0, 1)
    m16 = np.clip((mosaic - lo) / max(hi - lo, 1e-6), 0, 1)
    m16 = (m16 * 65535).astype(np.uint16)

    Image.fromarray(m16).save(base_dir / "stitched_BSE_16bit.tiff")
    Image.fromarray((m16 >> 8).astype(np.uint8)).save(base_dir / "stitched_BSE_8bit.png")

    if DRAW_LABELS:
        overlay_labels(m16, pos, coords_sorted, th, tw).save(base_dir / "stitched_BSE_debug_labels.png")

def preflight_analysis(files: list[Path], base_dir: Path, sample_pairs=30, auto_apply=False):
    """Quick analysis to suggest/auto-tune stitch parameters with plots."""
    print("[preflight] Starting analysis...")

    # --- Load subset of tiles
    tiles_raw = {}
    stage_xy  = []
    px_sizes  = []
    file_list = []

    cache_dir = None if globals().get("_CLI_NO_CACHE", False) else (base_dir / "stitch_cache")
    for p in files[:100]:  # cap for speed
        try:
            raw, sx, sy, ps = _load_bcf_cached(p.resolve(), cache_dir)
        except Exception:
            continue
        if ps and np.isfinite(ps) and ps > 0:
            px_sizes.append(ps)
        tiles_raw[p] = raw
        stage_xy.append((sx, sy))
        file_list.append(p)

    if not tiles_raw:
        print("[preflight] No tiles loaded.")
        return

    th, tw = next(iter(tiles_raw.values())).shape
    bh = max(8, int(round(BAND_FRAC * th)))
    bw = max(8, int(round(BAND_FRAC * tw)))

    # --- Pixel scaling
    px_um = float(np.median(px_sizes)) if px_sizes else None
    px_per_stage = 1.0 / px_um if px_um else None

    if px_per_stage:
        dx_stage = np.median(np.abs(np.diff(sorted([s[0] for s in stage_xy])))) if len(stage_xy) > 1 else tw
        dy_stage = np.median(np.abs(np.diff(sorted([s[1] for s in stage_xy])))) if len(stage_xy) > 1 else th
        step_x_nom = dx_stage * px_per_stage
        step_y_nom = dy_stage * px_per_stage
    else:
        step_x_nom = tw * (1 - OVERLAP_FRAC)
        step_y_nom = th * (1 - OVERLAP_FRAC)

    # --- Measure sample overlaps
    dx_right, dy_down, slice_dx, slice_dy = [], [], [], []
    file_list = list(tiles_raw.keys())
    tile_vals = list(tiles_raw.values())
    n_pairs = min(sample_pairs, len(file_list) - 1)
    # Prep each unique tile once across the sample window.
    prepped_cache = {i: prep_for_corr(tile_vals[i]).astype(np.float32, copy=False)
                     for i in range(n_pairs + 1)}
    for i in range(n_pairs):
        A_prep, B_prep = prepped_cache[i], prepped_cache[i + 1]
        out_r = measure_right_band(A_prep, B_prep, bh, bw, tw)
        if out_r: dY, dX, _, m = out_r; dx_right.append(dX); slice_dy.append(abs(m.get("a_y", 0)))
        out_d = measure_down_band(A_prep, B_prep, bh, bw, th)
        if out_d: dY, dX, _, m = out_d; dy_down.append(dY); slice_dx.append(abs(m.get("a_x", 0)))
    del prepped_cache

    # --- Suggested tunables
    overlap_frac = 1.0 - np.median(dx_right) / tw if dx_right else OVERLAP_FRAC
    step_y_nom   = np.median(dy_down) if dy_down else step_y_nom

    # estimate band fraction from overlap
    band_frac = min(0.25, 0.5 * overlap_frac)

    # raw drift estimates
    max_x_drift  = np.percentile(np.abs(dx_right), 95) if dx_right else MAX_VERT_X_DRIFT
    max_y_drift  = np.percentile(np.abs(dy_down), 95) if dy_down else MAX_RIGHT_Y_DRIFT

    # clamp drift to reasonable fraction of step size
    max_x_drift = min(max_x_drift, 0.1 * step_x_nom)
    max_y_drift = min(max_y_drift, 0.1 * step_y_nom)

    shear_cap    = 3.0 * max(np.median(slice_dx or [0]), np.median(slice_dy or [0]), 1e-6)

    print("\n[preflight] Suggested tunables:")
    print(f"  OVERLAP_FRAC      ~ {overlap_frac:.4f}")
    print(f"  BAND_FRAC         ~ {band_frac:.4f}")
    print(f"  step_x_nom        ~ {step_x_nom:.1f} px")
    print(f"  step_y_nom        ~ {step_y_nom:.1f} px")
    print(f"  MAX_VERT_X_DRIFT  ~ {max_x_drift:.1f} px")
    print(f"  MAX_RIGHT_Y_DRIFT ~ {max_y_drift:.1f} px")
    print(f"  SHEAR_CAP         ~ {shear_cap:.2e}")

    # --- Write report into data directory
    report_path = base_dir / "stitch_preflight_report.txt"
    with open(report_path, "w") as f:
        f.write("Preflight recommended tunables:\n")
        f.write(f"OVERLAP_FRAC      = {overlap_frac:.4f}\n")
        f.write(f"BAND_FRAC         = {band_frac:.4f}\n")
        f.write(f"step_x_nom        = {step_x_nom:.1f}\n")
        f.write(f"step_y_nom        = {step_y_nom:.1f}\n")
        f.write(f"MAX_VERT_X_DRIFT  = {max_x_drift:.1f}\n")
        f.write(f"MAX_RIGHT_Y_DRIFT = {max_y_drift:.1f}\n")
        f.write(f"SHEAR_CAP         = {shear_cap:.2e}\n")
    print(f"[preflight] Report written to {report_path}")

    # --- Diagnostic plots (unchanged)
    # [plots here as before...]

    # --- Auto-apply if requested
    if auto_apply:
        globals()["OVERLAP_FRAC"]     = overlap_frac
        globals()["BAND_FRAC"]        = band_frac
        globals()["MAX_VERT_X_DRIFT"] = max_x_drift
        globals()["MAX_RIGHT_Y_DRIFT"]= max_y_drift
        globals()["SHEAR_CAP"]        = shear_cap
        print("[preflight] Tunables auto-applied for next stitch run.")
        return step_x_nom, step_y_nom
    else:
        return None, None

def discover_elements_from_metadata(file_paths, sample_n=200):
    """
    Scan up to sample_n BCFs, read EDS metadata, and return a sorted list
    of unique elements declared in Sample.elements or inferred from Sample.xray_lines.
    """
    from eds_map import load_eds_signal_lazy, extract_eds_metadata_from_signal
    found = set()
    count = 0
    for p in file_paths:
        if count >= sample_n:
            break
        sig = load_eds_signal_lazy(p)
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        for el in meta.get('elements', []) or []:
            found.add(el)
        for ln in meta.get('xray_lines', []) or []:
            if '_' in ln:
                found.add(ln.split('_', 1)[0])
        count += 1
    # keep only those we have line energies for, and sort nicely
    valid = sorted([e for e in found if e in EDS_LINE_KEV],
                   key=lambda s: (len(s), s))
    return valid


# =========================
# EDS preview builder (downsampled elemental maps)
# =========================
EDS_LINE_KEV = {
    'B':0.183,'C':0.277,'N':0.392,'O':0.525,'F':0.676,'Na':1.041,'Mg':1.254,'Al':1.486,'Si':1.740,
    'P':2.013,'S':2.307,'Cl':2.622,'K':3.312,'Ca':3.691,'Ti':4.511,'V':4.952,'Cr':5.415,'Mn':5.900,'Fe':6.404,
    'Co':6.930,'Ni':7.478,'Cu':8.048,'Zn':8.638,'Ag':2.984,'Cd':3.133,'Sn':3.443,'Sb':3.605,'Te':3.769,
    'Ba':4.466,'W':8.398,'Au':9.713,'Pb':10.551,'U':13.614,
}

def _energy_slice(E0, dE, nE, E_center_keV, width_eV):
    half = 0.5 * width_eV / 1000.0
    lo = max(0, int(np.floor((E_center_keV - half - E0) / dE)))
    hi = min(nE, int(np.ceil((E_center_keV + half - E0) / dE)))
    if hi <= lo: hi = min(nE, lo + 1)
    return lo, hi

def _build_or_refresh_eds_cache(base_dir, elements, width_eV, ds, on_restitch,
                                shift_method="bilinear",
                                max_workers=4, flush_every=50,
                                cache_dir_override=None,
                                coating_elements=("C",)):
    """Build or refresh the EDS mosaic cache after a stitch run.

    Handles the re-stitch decision (prompt/rebuild/keep/discard) by comparing
    the existing cache fingerprint against the freshly written stitch manifest.

    ``cache_dir_override`` (from ``--eds-cache-dir``) relocates the cache to
    an explicit path (e.g. a faster SSD when tiles live on an HDD). A sidecar
    link file at ``<base_dir>/eds_cache_link.txt`` records the location so the
    viewer and future stitch runs find the cache automatically.
    """
    import sys
    import eds_cache

    manifest_csv = Path(base_dir) / "stitch_manifest.csv"
    if not manifest_csv.exists():
        print("[eds-cache] No stitch manifest on disk; cannot build.")
        return

    if cache_dir_override:
        cache_dir = Path(cache_dir_override).resolve()
    else:
        # Follow an existing link file if one is present — otherwise default.
        cache_dir = eds_cache.resolve_cache_dir(base_dir)
    status = eds_cache.status_for(cache_dir, manifest_csv)

    # Case 1: no prior cache — just build.
    if not status["exists"]:
        print(f"[eds-cache] Building fresh cache at {cache_dir} (ds={ds}, elements={elements}, "
              f"shift={shift_method}, workers={max_workers}, flush_every={flush_every}).")
        eds_cache.build_cache(manifest_csv, elements=elements,
                              width_eV=width_eV, ds=ds,
                              shift_method=shift_method,
                              max_workers=max_workers, flush_every=flush_every,
                              cache_dir=cache_dir,
                              coating_elements=coating_elements)
        return

    # Case 2: cache exists and matches current stitch — refresh if elements/width differ.
    if status["valid"]:
        needs_rebuild = False
        cached_els = set(status.get("elements") or [])
        req_els    = set(elements)
        if req_els - cached_els:
            print(f"[eds-cache] Cache is for {sorted(cached_els)}; adding {sorted(req_els - cached_els)}.")
            needs_rebuild = True
        if status.get("width_eV") is not None and abs(float(status["width_eV"]) - float(width_eV)) > 1e-6:
            print(f"[eds-cache] width_eV changed ({status['width_eV']} → {width_eV}); rebuilding.")
            needs_rebuild = True
        if status.get("ds") is not None and int(status["ds"]) != int(ds):
            print(f"[eds-cache] ds changed ({status['ds']} → {ds}); rebuilding.")
            needs_rebuild = True
        if needs_rebuild:
            all_els = sorted(cached_els.union(req_els))
            eds_cache.build_cache(manifest_csv, elements=all_els,
                                  width_eV=width_eV, ds=ds,
                                  shift_method=shift_method,
                                  max_workers=max_workers, flush_every=flush_every,
                                  cache_dir=cache_dir,
                                  coating_elements=coating_elements)
        else:
            print(f"[eds-cache] Existing cache is valid for this stitch — no rebuild needed.")
        return

    # Case 3: cache exists but fingerprint mismatch (re-stitch).
    action = on_restitch
    if action == "prompt":
        if not sys.stdin.isatty():
            print("[eds-cache] Stale cache detected but no interactive TTY — keeping stale cache.")
            action = "keep"
        else:
            print("[eds-cache] An existing EDS cache was built against a previous stitch.")
            print("            The new manifest differs, so the cache is stale.")
            try:
                ans = input("            Action? [r]ebuild / [k]eep stale / [d]iscard [r/K/d]: ").strip().lower()
            except EOFError:
                ans = ""
            action = {"r": "rebuild", "k": "keep", "d": "discard"}.get(ans, "keep")

    if action == "rebuild":
        print(f"[eds-cache] Rebuilding cache at {cache_dir} (ds={ds}, elements={elements}, "
              f"shift={shift_method}, workers={max_workers}, flush_every={flush_every}).")
        eds_cache.build_cache(manifest_csv, elements=elements,
                              width_eV=width_eV, ds=ds,
                              shift_method=shift_method,
                              max_workers=max_workers, flush_every=flush_every,
                              cache_dir=cache_dir,
                              coating_elements=coating_elements)
    elif action == "discard":
        print(f"[eds-cache] Discarding stale cache at {cache_dir}.")
        eds_cache.invalidate(cache_dir)
    else:
        print(f"[eds-cache] Keeping stale cache at {cache_dir} (viewer will ignore it).")


def build_eds_previews_fast(
    coords_sorted,
    pos,
    rc_to_path,
    th, tw,                       # BSE tile size
    preview_elements=('Si','O','Fe'),
    width_eV=150,
    ds=8,
    out_dir=Path('.'),
    use_feather=True,
    max_workers=0,
):
    """
    Memory-lean preview builder:
      • Integrate band → 2D
      • resample to BSE tile size (scale_y/x)
      • apply per-tile shear (shx/shy)
      • apply subpixel shift (fractional y,x)
      • place onto DS canvas with optional feather
    Uses on-disk memmaps for accumulators when the predicted footprint is large.
    """
    import numpy as np
    from skimage.transform import resize
    from scipy.ndimage import affine_transform
    from scipy.signal.windows import tukey as tukey1d
    from eds_map import load_eds_signal_lazy, extract_eds_metadata_from_signal, read_manifest

    manifest_rows = { (int(r['row']), int(r['col'])): r
                      for r in read_manifest(out_dir / "stitch_manifest.csv") }

    # Canvas size (full-res)
    H = int(np.ceil(max(pos[rc][0] for rc in coords_sorted) + th))
    W = int(np.ceil(max(pos[rc][1] for rc in coords_sorted) + tw))
    h_ds, w_ds = max(1, H // ds), max(1, W // ds)

    # Discover energy axis from the first tile with EDS
    E0=dE=None; nE=None
    for rc in coords_sorted:
        eds = load_eds_signal_lazy(rc_to_path[rc])
        if eds is None: continue
        meta = extract_eds_metadata_from_signal(eds)
        if meta.get('nE', 0) > 0:
            E0, dE, nE = meta['E0'], meta['dE'], meta['nE']
            break
    if nE is None:
        print("[eds-preview] No EDS; skipping.")
        return

    # Validate element list against our line table
    els = [e for e in preview_elements if e in EDS_LINE_KEV]
    if not els:
        print("[eds-preview] No valid elements requested.")
        return

    # Helper: energy band
    def eband(E_center_keV):
        half = 0.5 * width_eV / 1000.0
        lo = max(0, int(np.floor((E_center_keV - half - E0)/dE)))
        hi = min(nE, int(np.ceil((E_center_keV + half - E0)/dE)))
        return max(0, lo), max(lo+1, hi)

    # Feather (downsampled)
    def feather(h, w, enable=True):
        if not enable: return np.ones((h, w), dtype=np.float32)
        wy = tukey1d(h, alpha=0.08).astype(np.float32)
        wx = tukey1d(w, alpha=0.08).astype(np.float32)
        return np.outer(wy, wx).astype(np.float32)

    # --- Storage selection: in-RAM vs on-disk (memmap)
    outp = out_dir / f"eds_preview_ds{ds}"
    outp.mkdir(parents=True, exist_ok=True)

    bytes_per_el = h_ds * w_ds * 4  # float32
    total_bytes  = bytes_per_el * (len(els) + 1)  # + weight
    MEMMAP_THRESHOLD = 512 * 1024 * 1024  # 512 MB

    use_memmap = total_bytes >= MEMMAP_THRESHOLD
    mode_str = "memmap" if use_memmap else "RAM"
    print(f"[eds-preview] Canvas {h_ds}×{w_ds}, elements={len(els)}, storage={mode_str}")

    # Allocate accumulators
    if use_memmap:
        weight_path = outp / "_weight.dat"
        weight = np.memmap(weight_path, dtype=np.float32, mode='w+', shape=(h_ds, w_ds))
        previews = {}
        for el in els:
            pth = outp / f"_{el}.dat"
            previews[el] = np.memmap(pth, dtype=np.float32, mode='w+', shape=(h_ds, w_ds))
    else:
        weight  = np.zeros((h_ds, w_ds), dtype=np.float32)
        previews= {el: np.zeros((h_ds, w_ds), dtype=np.float32) for el in els}

    # Process tiles sequentially (keeps memory steady and avoids write races)
    tile_iter = tqdm(coords_sorted, desc="EDS previews (tiles)", unit="tile") if len(coords_sorted) > 25 else coords_sorted

    for rc in tile_iter:
        p = rc_to_path[rc]
        row = manifest_rows.get(rc)
        if row is None:
            continue

        eds = load_eds_signal_lazy(p)
        if eds is None:
            continue
        meta = extract_eds_metadata_from_signal(eds)
        # Require matching energy axis
        if not (meta.get('nE',0) == nE and
                abs(meta.get('E0',0)-E0) < 1e-6 and
                abs(meta.get('dE',0)-dE) < 1e-9):
            continue

        # Transforms/scales (as in BSE)
        sy  = float(row.get('scale_y', 1.0) or 1.0)
        sx  = float(row.get('scale_x', 1.0) or 1.0)
        shx = float(row.get('shx', 0.0) or 0.0)
        shy = float(row.get('shy', 0.0) or 0.0)
        ty  = float(row['y']); tx = float(row['x'])

        # DS placement region for this tile
        iy = int(np.floor(ty)); ix = int(np.floor(tx))
        oy0 = iy // ds; ox0 = ix // ds
        oy1 = min(h_ds, (iy + th + ds - 1)//ds)
        ox1 = min(w_ds, (ix + tw + ds - 1)//ds)
        if oy1 <= oy0 or ox1 <= ox0:
            continue
        ph, pw = oy1 - oy0, ox1 - ox0
        wpatch = feather(ph, pw, enable=use_feather)

        # Precompute shift (subpixel at full-res, then we downsample)
        fy = ty - np.floor(ty); fx = tx - np.floor(tx)
        use_subpx = (abs(fy) > 1e-6 or abs(fx) > 1e-6)

        # For each element: integrate → warp to BSE tile → downsample → accumulate
        # We reuse intermediate buffers to keep allocations tiny.
        tgt_h = max(1, int(round(th * sy)))
        tgt_w = max(1, int(round(tw * sx)))

        for el in els:
            lo, hi = eband(EDS_LINE_KEV[el])

            # Lazy band sum to 2D
            try:
                img = eds.isig[lo:hi].sum(axis='Energy').data
                img = np.asarray(img, dtype=np.float32, order='C')
            except Exception:
                continue
            if img.ndim != 2:
                continue

            # Resize to scaled EDS→BSE, then to exact BSE tile size
            if img.shape != (tgt_h, tgt_w):
                img = resize(img, (tgt_h, tgt_w), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)
            if img.shape != (th, tw):
                img = resize(img, (th, tw), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)

            # Shear (exactly like BSE blend)
            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx],[shy,1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                img = affine_transform(img, Minv, offset=0.0, order=1, mode='nearest', cval=0.0, prefilter=False).astype(np.float32, copy=False)

            # Subpixel shift — skip when fractional part is tiny.
            if use_subpx and (abs(fy) > 0.02 or abs(fx) > 0.02):
                from scipy.ndimage import fourier_shift
                img = np.fft.ifftn(fourier_shift(np.fft.fftn(img), (fy, fx))).real.astype(np.float32, copy=False)

            # Downsample straight to DS patch region
            img_ds = resize(img, (ph, pw), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32, copy=False)

            # Accumulate (memmap or RAM)
            previews[el][oy0:oy1, ox0:ox1] += img_ds * wpatch

        # weight once per tile
        weight[oy0:oy1, ox0:ox1] += wpatch

        # If memmap, flush occasionally to keep OS cache tame
        if use_memmap and (rc[1] % 16 == 0):
            weight.flush()
            for el in els:
                previews[el].flush()

    # --- Normalize (weight) and save PNGs with tiling/chunks to stay RAM-light
    m = weight > 0
    if use_memmap:
        # Read a chunk at a time for percentile/normalization
        # Simple one-pass: compute global 1/99 percentiles from a sampled subset
        # (fast and robust enough for previews).
        import math
        sample_stride = max(1, int(math.sqrt((h_ds * w_ds) // (1024*1024))))  # ~1M sample cap
        mask_sample = m[::sample_stride, ::sample_stride]

        for el in els:
            arr = previews[el]
            sample = arr[::sample_stride, ::sample_stride][mask_sample]
            if sample.size:
                lo, hi = np.percentile(sample, [1, 99])
            else:
                lo, hi = 0.0, 1.0

            # Write 8-bit PNG by blocks
            out_img = np.empty((h_ds, w_ds), dtype=np.uint8)
            block = 2048
            for y0 in range(0, h_ds, block):
                y1 = min(h_ds, y0 + block)
                slab = arr[y0:y1, :]
                ww   = weight[y0:y1, :]
                out  = np.zeros_like(slab, dtype=np.float32)
                mm   = ww > 0
                out[mm] = slab[mm] / ww[mm]
                if hi > lo:
                    out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
                out_img[y0:y1, :] = (out * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(out_img).save(outp / f"{el}.png")

        # Optional: remove temporary .dat files to reclaim disk (comment to cache)
        try:
            (outp / "_weight.dat").unlink(missing_ok=True)
            for el in els:
                (outp / f"_{el}.dat").unlink(missing_ok=True)
        except Exception:
            pass

    else:
        # RAM path: normalize and write
        for el, img in previews.items():
            out = np.zeros_like(img, dtype=np.float32)
            if m.any():
                out[m] = img[m] / weight[m]
                v = out[np.isfinite(out)]
                if v.size:
                    lo, hi = np.percentile(v, [1, 99])
                    if hi > lo:
                        out = np.clip((out - lo)/(hi - lo), 0, 1)
            Image.fromarray((255*out).astype(np.uint8)).save(outp / f"{el}.png")

    print(f"[eds-preview] Saved aligned previews → {outp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Run preflight analysis only")
    parser.add_argument("--auto", action="store_true", help="Auto-apply tunables and run stitching")
    parser.add_argument("--dir", type=str, default="debug_bcf", help="Directory with .bcf files")
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="default",
                        help="Alignment preset (default: 'default', which made the paper's "
                             "mosaic). What each one sets and when to use it: "
                             "stitch_presets.py, or the viewer's stitching window.")
    # NEW:
    parser.add_argument("--eds-preview", type=str, default="",
                        help="Build downsampled EDS preview maps during stitch. "
                             "Values: 'auto' (elements from metadata), 'all' (same as auto), "
                             "or comma list like 'Si,O,Fe'. If empty, previews are skipped.")
    parser.add_argument("--eds-preview-ds", type=int, default=8,
                        help="Downsample factor for EDS previews (default: 8).")
    parser.add_argument("--eds-preview-width", type=int, default=150,
                        help="EDS window width (eV) for previews (default: 150).")
    parser.add_argument("--eds-preview-workers", type=int, default=0,
                        help="Thread workers for preview building (0=off; 4–8 is good on SSDs).")
    parser.add_argument("--eds-preview-feather", type=int, default=0,
                        help="Feather edges in previews (0/1). Off is faster.")
    parser.add_argument("--eds-cache", dest="eds_cache", action="store_true", default=True,
                        help="Build/update the full-resolution EDS mosaic cache at "
                             "<dir>/eds_cache/ (default: on, when --eds-preview is given).")
    parser.add_argument("--no-eds-cache", dest="eds_cache", action="store_false",
                        help="Disable the EDS mosaic cache (only preview PNGs are written).")
    parser.add_argument("--eds-cache-ds", type=int, default=1,
                        help="Downsample factor for the EDS mosaic cache (default 1 = full-res). "
                             "2/4/8 trade resolution for disk space.")
    parser.add_argument("--eds-cache-on-restitch", type=str, default="prompt",
                        choices=["prompt", "rebuild", "keep", "discard"],
                        help="What to do with an existing EDS cache when this stitch run produces "
                             "a different manifest: prompt (ask interactively, default), rebuild, "
                             "keep (marked stale), discard (delete without rebuild).")
    parser.add_argument("--eds-shift-method", type=str, default="bilinear",
                        choices=["bilinear", "fft"],
                        help="Sub-pixel shift interpolation for EDS tile alignment. "
                             "'bilinear' (default) is ~8x faster and visually indistinguishable "
                             "from 'fft' (sinc) for smooth EDS counts. Use 'fft' for strict "
                             "bandlimited fidelity.")
    parser.add_argument("--eds-cache-workers", type=int, default=4,
                        help="Parallel tile workers during EDS cache build (default 4). "
                             "Each worker holds one ~0.8-1.5 GB EDS subcube in RAM while "
                             "processing its tile. Lower to reduce RAM pressure on workstations.")
    parser.add_argument("--eds-cache-flush-every", type=int, default=50,
                        help="Flush cache memmaps to disk every N tiles (default 50). "
                             "Lower values reduce peak working-set size (prevents Windows from "
                             "paging other processes out to the page file during long builds) at "
                             "the cost of more disk I/O. Set very high to disable periodic flush.")
    parser.add_argument("--eds-cache-dir", type=str, default="",
                        help="Store the EDS mosaic cache at this path instead of the default "
                             "<--dir>/eds_cache/. Useful for putting the cache on an SSD when "
                             "the source tiles are on an HDD. A small sidecar file "
                             "<--dir>/eds_cache_link.txt records the location so the viewer and "
                             "subsequent stitch runs find it automatically.")
    parser.add_argument("--coating-elements", nargs="*", default=["C"], metavar="EL",
                        help="Conductive coating / contaminant element symbols to ignore in "
                             "mineral ID and SPX calibration (default: C). Use e.g. "
                             "'--coating-elements Pt' for a Pt coat, or '--coating-elements' "
                             "with no value for an uncoated sample. Recorded in the EDS cache "
                             "manifest; the viewer can still override it per session.")
    parser.add_argument("--prep-variant", type=str, default="auto",
                        choices=["auto", "edge", "wide_hp", "flat", "dog"],
                        help="Phase-correlation prep filter. 'auto' calibrates on a sample "
                             "and picks the best; 'edge' is the classic gradient-magnitude "
                             "prep matched for sharp features.")
    parser.add_argument("--weight-floor", type=float, default=0.0,
                        help="Drop pairs with phase-corr weight below this threshold from "
                             "the LSQR solve. 0 disables (default).")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run only the neighbor-graph build and the initial LSQR solve, "
                             "write a diagnostic report (stitch_diagnose_report.txt + debug CSVs) "
                             "and exit. Skips refinement, shear, and composite.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the on-disk BCF tile cache (stitch_cache/ inside --dir). "
                             "Default is cache enabled: first load writes NPZ files keyed by "
                             "BCF mtime+size; subsequent runs skip hs.load.")

    args = parser.parse_args()

    # Alignment preset first: everything below reads the module tunables.
    apply_preset(args.preset)

    # Apply prep-variant selection before stitch/preflight so prep_for_corr's
    # internal dispatch uses the right filter. The 'auto' path is handled
    # inside stitch() (needs access to the built grid for calibration).
    _CLI_PREP_VARIANT = args.prep_variant
    _CLI_WEIGHT_FLOOR = float(args.weight_floor)
    _CLI_DIAGNOSE = bool(args.diagnose)
    _CLI_NO_CACHE = bool(args.no_cache)
    if args.prep_variant != "auto":
        PREP_VARIANT = args.prep_variant

    directory = Path(args.dir)
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    # Record how this mosaic is being made, whether started from the command
    # line or from the viewer: the exact command, the preset and the alignment
    # tunables it resolved to. Overwritten by the next run.
    try:
        import json as _json
        import shlex as _shlex
        import sys as _sys
        from datetime import datetime as _dt
        _dbg = directory / "stitch_debug"
        _dbg.mkdir(parents=True, exist_ok=True)
        _cmd = " ".join(_shlex.quote(a) for a in ["python", "stitch.py"] + _sys.argv[1:])
        (_dbg / "stitch_command.txt").write_text(
            f"# {_dt.now().isoformat(timespec='seconds')}\n{_cmd}\n\n"
            f"preset: {args.preset}\n"
            f"tunables: {_json.dumps(current_tunables())}\n", encoding="utf-8")
    except OSError as _e:
        print(f"[stitch] could not write stitch_debug/stitch_command.txt: {_e}")

    # Recursive, case-insensitive, files only
    bcf_files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".bcf"]

    print(f"Found {len(bcf_files)} .bcf files (including subdirectories).")
    # Stable-ish sort: by folder, filename, then mtime
    bcf_files.sort(key=lambda p: (p.parent.as_posix(), p.name.lower(), p.stat().st_mtime))

    if args.preflight:
        step_x_nom, step_y_nom = preflight_analysis(bcf_files, directory, auto_apply=args.auto)
        if args.auto:
            stitch(bcf_files, directory)  # runs with adjusted globals
    else:
        stitch(bcf_files, directory)