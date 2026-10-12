import csv
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import hyperspy.api as hs
from scipy.ndimage import affine_transform
from scipy.fft import fftn, ifftn
from scipy.ndimage import fourier_shift
from scipy.signal.windows import tukey as tukey1d

# ---------- metadata helpers ----------
def get_stage_coordinates(p: Path):
    try:
        s = hs.load(p)[2]
    except Exception:
        return 0.0, 0.0
    md = getattr(s, "metadata", None)
    if md is None:
        return 0.0, 0.0
    # try direct attrs
    try:
        return float(md.Acquisition_instrument.SEM.Stage.X), float(md.Acquisition_instrument.SEM.Stage.Y)
    except Exception:
        pass
    # fallback get_item
    try:
        return float(md.get_item("Acquisition_instrument.SEM.Stage.x")), float(md.get_item("Acquisition_instrument.SEM.Stage.y"))
    except Exception:
        return 0.0, 0.0

def _mad(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return np.median(np.abs(x - med))

def _robust_threshold_for_splits(diffs):
    diffs = np.asarray(diffs, float)
    if diffs.size == 0:
        return 0.0
    med = np.median(diffs)
    mad = _mad(diffs) + 1e-12
    thr = med + 3.0 * 1.4826 * mad
    if not np.isfinite(thr) or thr <= 0:
        thr = np.percentile(diffs, 90.0)
    return float(thr)

def _cluster_rows_by_sy(sy_values):
    sy = np.asarray(sy_values, float)
    order = np.argsort(sy)  # ascending (SEM Y grows downward)
    vs = sy[order]
    diffs = np.diff(vs)
    thr = _robust_threshold_for_splits(diffs)
    groups = []
    start = 0
    for i, d in enumerate(diffs):
        if d > thr:
            groups.append(vs[start:i+1])
            start = i + 1
    groups.append(vs[start:])
    centers = [float(np.median(g)) for g in groups]
    centers_arr = np.asarray(centers, float)
    row_assign = np.empty_like(sy, dtype=int)
    for rank, idx in enumerate(order):
        v = vs[rank]
        j = int(np.argmin(np.abs(centers_arr - v)))
        row_assign[idx] = j
    return centers, row_assign

def _estimate_step_x_from_rows(sx_lists_desc):
    diffs = []
    for arr in sx_lists_desc:
        if len(arr) >= 2:
            d = np.diff(arr)
            diffs.append(np.abs(d))
    if not diffs:
        return 1.0
    diffs = np.concatenate(diffs)
    med = np.median(diffs); mad = _mad(diffs) + 1e-12
    z = np.abs(diffs - med) / (1.4826 * mad)
    inliers = diffs[z < 2.5] if np.any(z < 2.5) else diffs
    v = float(np.median(inliers))
    return v if v > 0 else 1.0

def _align_row_to_centers_desc(sx_row_desc, centers_desc, allow_extend, step_x_est):
    centers = list(centers_desc)
    K = len(centers); assign = []; prev_i = -1
    for s in sx_row_desc:
        if K == 0:
            centers.append(float(s)); K = 1; prev_i = 0; assign.append(0); continue
        diffs = np.abs(np.asarray(centers) - s)
        i = int(np.argmin(diffs))
        if i <= prev_i:
            i = prev_i + 1
        if i >= K:
            if allow_extend:
                centers.append(float(s)); K += 1; assign.append(K-1); prev_i = K-1; continue
            else:
                i = K - 1
        assign.append(i); prev_i = i
    return assign, centers

def build_grid_from_stage_robust(stage_xy):
    N = len(stage_xy)
    sx = np.asarray([t[0] for t in stage_xy], float)
    sy = np.asarray([t[1] for t in stage_xy], float)
    row_centers, row_assign = _cluster_rows_by_sy(sy)

    by_row = {}
    for i, r in enumerate(row_assign):
        by_row.setdefault(int(r), []).append(i)

    row_sx_desc = {}
    row_idx_desc = {}
    for r, idxs in by_row.items():
        arr = np.array([sx[i] for i in idxs], float)
        order = np.argsort(arr)[::-1]  # descending X (left→right in image)
        row_sx_desc[r] = arr[order]
        row_idx_desc[r] = [idxs[j] for j in order]

    step_x_est = _estimate_step_x_from_rows(list(row_sx_desc.values()))

    # reference row = most tiles (tie → largest span)
    row_lens = {r: len(v) for r, v in row_sx_desc.items()}
    max_len = max(row_lens.values())
    candidates = [r for r, L in row_lens.items() if L == max_len]
    if len(candidates) > 1:
        spans = {r: (float(np.max(row_sx_desc[r]) - np.min(row_sx_desc[r])) if len(row_sx_desc[r])>1 else 0.0)
                 for r in candidates}
        ref_row = max(spans, key=spans.get)
    else:
        ref_row = candidates[0]

    centers = list(map(float, row_sx_desc[ref_row]))
    col_assign = np.full(N, -1, dtype=int)
    for j, idx in enumerate(row_idx_desc[ref_row]):
        col_assign[idx] = j

    for r, sx_desc in row_sx_desc.items():
        if r == ref_row: continue
        assign, centers = _align_row_to_centers_desc(sx_desc, centers, allow_extend=True, step_x_est=step_x_est)
        for j, idx in enumerate(row_idx_desc[r]):
            col_assign[idx] = assign[j]

    # refine centers
    K = max(col_assign) + 1
    new_centers = []
    for k in range(K):
        vals = sx[col_assign == k]
        new_centers.append(float(np.median(vals)) if vals.size else float(centers[k]))
    centers = new_centers

    # fixed pass
    for r, sx_desc in row_sx_desc.items():
        assign, _ = _align_row_to_centers_desc(sx_desc, centers, allow_extend=False, step_x_est=step_x_est)
        for j, idx in enumerate(row_idx_desc[r]):
            col_assign[idx] = assign[j]

    return row_centers, centers, row_assign, col_assign

# ---------- composite from CSVs ----------
def composite_from_debug(directory: Path,
                         positions_csv: Path = None,
                         pairs_csv: Path = None,
                         tukey_alpha=0.08,
                         shear_enable=True,
                         shear_cap=5e-4,
                         low_pct=0.5,
                         hi_pct=99.5):
    """
    Reconstructs the mosaic using saved debug CSVs.
    Maps (row,col) to actual files by rebuilding the grid from stage metadata.
    """
    if positions_csv is None:
        positions_csv = directory / "stitch_debug" / "stitch_debug_positions.csv"
        if not positions_csv.exists():
            positions_csv = directory / "stitch_debug_positions.csv"
    if pairs_csv is None:
        pairs_csv = directory / "stitch_debug" / "stitch_debug_pairs.csv"
        if not pairs_csv.exists():
            pairs_csv = directory / "stitch_debug_pairs.csv"

    # --- 1) load pixel positions
    pos = {}
    with open(positions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rc = (int(row["row"]), int(row["col"]))
            pos[rc] = np.array([float(row["y"]), float(row["x"])], dtype=np.float32)
    coords_needed = set(pos.keys())
    if not coords_needed:
        raise RuntimeError("No positions in positions CSV.")

    # --- 2) scan BCFs and build (row,col) assignment from stage metadata
    print("[relink] Scanning .bcf files and rebuilding grid from stage metadata...")
    all_files = sorted(directory.rglob("*.bcf"))
    if not all_files:
        raise RuntimeError(f"No .bcf files found under {directory}")

    stage_xy = []
    file_list = []
    for p in tqdm(all_files, desc="Reading stage coords"):
        sx, sy = get_stage_coordinates(p)
        stage_xy.append((sx, sy))
        file_list.append(p)

    row_centers, col_centers, row_assign, col_assign = build_grid_from_stage_robust(stage_xy)
    rc_to_path = {}
    for r, c, p in zip(row_assign, col_assign, file_list):
        rc = (int(r), int(c))
        # only keep files that appear in the positions CSV grid
        if rc in coords_needed and rc not in rc_to_path:
            rc_to_path[rc] = p

    missing = sorted(rc for rc in coords_needed if rc not in rc_to_path)
    if missing:
        miss_str = ", ".join(map(str, missing[:10]))
        raise RuntimeError(f"Could not map {len(missing)} tiles from CSV to files via stage grid. "
                           f"First few missing indices: {miss_str}")

    # --- 3) recover per-tile shear from pairs CSV
    tile_shear = {rc: {"shx": 0.0, "shy": 0.0, "n_x": 0, "n_y": 0} for rc in coords_needed}
    if shear_enable and pairs_csv.exists():
        with open(pairs_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rA, cA = int(row["rA"]), int(row["cA"])
                rB, cB = int(row["rB"]), int(row["cB"])
                t = row["type"].strip().lower()
                a_y = float(row.get("a_y", 0.0))
                a_x = float(row.get("a_x", 0.0))
                if (rB, cB) not in tile_shear:
                    continue
                if t == "right":
                    tile_shear[(rB, cB)]["shy"] += float(np.clip(a_y, -shear_cap, shear_cap))
                    tile_shear[(rB, cB)]["n_y"] += 1
                elif t == "down":
                    tile_shear[(rB, cB)]["shx"] += float(np.clip(a_x, -shear_cap, shear_cap))
                    tile_shear[(rB, cB)]["n_x"] += 1
        for rc, acc in tile_shear.items():
            if acc["n_y"] > 0:
                acc["shy"] = float(np.clip(acc["shy"] / acc["n_y"], -shear_cap, shear_cap))
            if acc["n_x"] > 0:
                acc["shx"] = float(np.clip(acc["shx"] / acc["n_x"], -shear_cap, shear_cap))

    # --- 4) load tiles (only those needed) and composite
    # pick one to get tile size
    first_rc = next(iter(coords_needed))
    th, tw = hs.load(rc_to_path[first_rc])[0].data.shape

    # mosaic canvas (float32 to avoid huge memory)
    H = int(np.ceil(max(p[0] for p in pos.values()) + th))
    W = int(np.ceil(max(p[1] for p in pos.values()) + tw))
    print(f"[composite-only] Canvas = {H} x {W} px")
    accum  = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    wy = tukey1d(th, alpha=tukey_alpha).astype(np.float32)
    wx = tukey1d(tw, alpha=tukey_alpha).astype(np.float32)
    w_win = np.outer(wy, wx)

    print(f"[composite-only] Compositing {len(coords_needed)} tiles…")
    for rc in tqdm(sorted(coords_needed)):
        path = rc_to_path[rc]
        raw = hs.load(path)[0].data.astype(np.float32)

        yx = pos[rc]
        iy, ix = np.floor(yx).astype(int)
        frac = yx - np.floor(yx)

        # subpixel shift
        subpix = ifftn(fourier_shift(fftn(raw), frac)).real.astype(np.float32)

        # optional tiny shear
        if shear_enable:
            shx = tile_shear[rc]["shx"]
            shy = tile_shear[rc]["shy"]
            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx],
                              [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                subpix = affine_transform(
                    subpix, Minv, offset=0.0, order=1, mode="nearest", cval=0.0, prefilter=False
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

    # scale and save
    mosaic = np.divide(accum, weight, out=np.zeros_like(accum), where=weight > 0)
    vals = mosaic[weight > 0]
    lo, hi = np.percentile(vals, [low_pct, hi_pct]) if vals.size else (0, 1)
    m16 = np.clip((mosaic - lo) / max(hi - lo, 1e-6), 0, 1)
    m16 = (m16 * 65535).astype(np.uint16)

    out16 = directory / "stitched_BSE_16bit_from_debug.tiff"
    out8  = directory / "stitched_BSE_8bit_from_debug.png"
    Image.fromarray(m16).save(out16)
    Image.fromarray((m16 >> 8).astype(np.uint8)).save(out8)
    print(f"[composite-only] Wrote:\n  {out16}\n  {out8}")


import argparse
from pathlib import Path

# (Paste the entire composite_from_debug() implementation from above here)

def main():
    parser = argparse.ArgumentParser(description="Composite tiles from saved debug CSVs without recomputing alignment.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Path to the dataset folder containing .bcf files and stitch_debug_positions.csv.")
    parser.add_argument("--positions", type=str, default=None,
                        help="Optional path to stitch_debug_positions.csv (defaults to one found in --dir).")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Optional path to stitch_debug_pairs.csv (for shear correction).")
    parser.add_argument("--no-shear", action="store_true",
                        help="Disable per-tile shear correction.")
    parser.add_argument("--alpha", type=float, default=0.08,
                        help="Tukey window alpha (default: 0.08).")
    parser.add_argument("--low", type=float, default=0.5,
                        help="Lower percentile for intensity normalization (default: 0.5).")
    parser.add_argument("--high", type=float, default=99.5,
                        help="Upper percentile for intensity normalization (default: 99.5).")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    positions_csv = Path(args.positions) if args.positions else None
    pairs_csv = Path(args.pairs) if args.pairs else None

    composite_from_debug(
        directory=data_dir,
        positions_csv=positions_csv,
        pairs_csv=pairs_csv,
        tukey_alpha=args.alpha,
        shear_enable=not args.no_shear,
        low_pct=args.low,
        hi_pct=args.high
    )

if __name__ == "__main__":
    main()
