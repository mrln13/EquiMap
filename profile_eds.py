"""Profile the per-tile EDS pipeline on a subset of real BCF tiles.

Reports:
  (1) cProfile cumulative/tottime ranking — coarse function-level overview
  (2) Hand-instrumented breakdown of per-tile steps — answers
      "is the bottleneck HyperSpy reads, resize, FFT shift, or accumulation?"

Usage:
    python profile_eds.py --dir "D:/UU/C23S1A_Quant" --n 10
    python profile_eds.py --dir "D:/UU/C23S1A_Quant" --n 10 --elements "Si,O,Fe,Al,Ca"
    python profile_eds.py --dir "D:/UU/C23S1A_Quant" --n 10 --width 150

The script does NOT modify source files. It constructs a realistic per-tile
workload (mosaic-wide ROI, 3 bands per element) and runs it synchronously so
profiler output isn't muddied by threads.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from eds_map import (
    read_manifest, load_eds_signal_lazy, extract_eds_metadata_from_signal,
    _resolve_manifest_path, _band_indices, _sideband_centers,
    _tukey_window_2d, LINE_ENERGIES_KEV,
)


def _discover_elements(rows, base_dir, max_probe: int = 20) -> list[str]:
    """Probe a few tiles for metadata elements list."""
    seen: set[str] = set()
    for r in rows[:max_probe]:
        p = _resolve_manifest_path(r["file"], base_dir)
        sig = load_eds_signal_lazy(p)
        if sig is None:
            continue
        meta = extract_eds_metadata_from_signal(sig)
        for el in meta.get("elements", []):
            if el in LINE_ENERGIES_KEV:
                seen.add(el)
        if len(seen) >= 5:
            break
    return sorted(seen)


def _process_one_tile_instrumented(
    r, base_dir, elements, bands, E0, dE, nE, roi, timings: dict
):
    """Replica of eds_map._process_one_tile with per-step timing.

    Matches the batched-read optimisation in eds_map.py (single subcube load,
    numpy-per-band sums). Set env THERMOD_PROFILE_NAIVE=1 to force the old
    per-band lazy reads for comparison.
    """
    import os as _os
    naive = _os.environ.get("THERMOD_PROFILE_NAIVE", "") == "1"

    from skimage.transform import resize
    from scipy.ndimage import affine_transform, fourier_shift

    t0 = time.perf_counter()
    p = _resolve_manifest_path(r["file"], base_dir)
    sig = load_eds_signal_lazy(p)
    timings["bcf_open"] += time.perf_counter() - t0
    if sig is None:
        return None

    t0 = time.perf_counter()
    meta = extract_eds_metadata_from_signal(sig)
    if not (meta.get("nE", 0) == nE and
            abs(float(meta.get("E0", 0.0)) - E0) < 1e-6 and
            abs(float(meta.get("dE", 0.0)) - dE) < 1e-9):
        return None
    timings["energy_check"] += time.perf_counter() - t0

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

    t0 = time.perf_counter()
    ww_full = _tukey_window_2d(th, tw, 0.08)
    ww = ww_full[yA0:yA1, xA0:xA1].astype(np.float32)
    timings["tukey_window"] += time.perf_counter() - t0

    fy = ty - np.floor(ty); fx = tx - np.floor(tx)
    use_subpx = (abs(fy) > 1e-6 or abs(fx) > 1e-6)

    tgt_h = max(1, int(round(th * sy))); tgt_w = max(1, int(round(tw * sx)))

    # Batched band extraction (new fast path)
    band_images: dict = {}
    if not naive:
        t0 = time.perf_counter()
        try:
            all_ranges = [rng for e in elements for rng in bands[e].values()]
            min_lo = min(lo for lo, _ in all_ranges)
            max_hi = max(hi for _, hi in all_ranges)
            sub = np.asarray(sig.isig[min_lo:max_hi].data)
            sig_ax_idx = sig.axes_manager.signal_axes[0].index_in_array
            for e in elements:
                band_images[e] = {}
                for key, (lo, hi) in bands[e].items():
                    slc = [slice(None)] * sub.ndim
                    slc[sig_ax_idx] = slice(lo - min_lo, hi - min_lo)
                    band_images[e][key] = sub[tuple(slc)].sum(
                        axis=sig_ax_idx, dtype=np.float32)
            del sub
        except Exception:
            band_images = {}
        timings["hyperspy_subcube_read"] += time.perf_counter() - t0

    # Phase A: resize + shear per band
    shaped: dict = {}
    wrote_any = False
    for e in elements:
        shaped[e] = {}
        for key, (lo, hi) in bands[e].items():
            img = band_images.get(e, {}).get(key)
            if img is None:
                t0 = time.perf_counter()
                try:
                    img = sig.isig[lo:hi].sum(axis="Energy").data
                    img = np.asarray(img, dtype=np.float32)
                except Exception:
                    continue
                timings["hyperspy_band_read"] += time.perf_counter() - t0
            if img.ndim != 2:
                continue

            t0 = time.perf_counter()
            if img.shape != (tgt_h, tgt_w):
                img = resize(img, (tgt_h, tgt_w), order=1,
                             preserve_range=True, anti_aliasing=False).astype(np.float32)
            if img.shape != (th, tw):
                img = resize(img, (th, tw), order=1,
                             preserve_range=True, anti_aliasing=False).astype(np.float32)
            timings["resize"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            if abs(shx) > 1e-7 or abs(shy) > 1e-7:
                M = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
                Minv = np.linalg.inv(M)
                img = affine_transform(
                    img, Minv, offset=0.0, order=1, mode="nearest",
                    cval=0.0, prefilter=False,
                ).astype(np.float32)
            timings["shear"] += time.perf_counter() - t0

            shaped[e][key] = img
            wrote_any = True

    # Phase B: subpixel shift. Bilinear (fast) or FFT (exact). Env var
    # THERMOD_PROFILE_SHIFT=fft|bilinear overrides; default matches eds_map.py.
    shift_mode = _os.environ.get("THERMOD_PROFILE_SHIFT", "bilinear").lower()
    if use_subpx:
        t0 = time.perf_counter()
        if shift_mode == "fft":
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
        timings["subpixel_shift"] += time.perf_counter() - t0

    # Phase C: crop + mock accumulation
    t0 = time.perf_counter()
    for e in elements:
        for key in shaped.get(e, {}):
            img = shaped[e][key]
            _patch = img[yA0:yA1, xA0:xA1]
            _ = _patch * ww
    timings["crop_mul"] += time.perf_counter() - t0

    return wrote_any


def _print_timings(total_wall: float, timings: dict, n_tiles: int, n_bands_per_tile: int) -> None:
    print("\n=== Per-step breakdown (seconds, averaged over all tile-band ops) ===")
    print(f"{'step':<24} {'total s':>10} {'% of wall':>10} {'per tile ms':>14}")
    # rows that are per-tile vs per-band — keep simple: show total + per-tile.
    # Bands-heavy rows: hyperspy_band_read, resize, shear, subpixel_shift, crop_mul.
    ordered = [
        "bcf_open", "energy_check", "tukey_window",
        "hyperspy_subcube_read", "hyperspy_band_read",
        "resize", "shear", "subpixel_shift", "crop_mul",
    ]
    accounted = 0.0
    for k in ordered:
        v = timings.get(k, 0.0)
        accounted += v
        pct = 100.0 * v / total_wall if total_wall > 0 else 0.0
        per_tile = 1000.0 * v / max(n_tiles, 1)
        print(f"{k:<24} {v:>10.3f} {pct:>9.1f}% {per_tile:>14.1f}")
    other = max(0.0, total_wall - accounted)
    print(f"{'(unaccounted)':<24} {other:>10.3f} {100.0*other/total_wall:>9.1f}%")
    print(f"{'TOTAL wall':<24} {total_wall:>10.3f}   (n_tiles={n_tiles}, bands/tile={n_bands_per_tile})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Stitched directory containing stitch_manifest.csv")
    ap.add_argument("--n", type=int, default=10, help="Number of tiles to profile (default 10)")
    ap.add_argument("--elements", type=str, default="",
                    help='Comma list, e.g. "Si,O,Fe". Empty = auto-discover from metadata.')
    ap.add_argument("--width", type=float, default=150.0, help="Band width (eV) (default 150)")
    ap.add_argument("--sideband-mult", type=float, default=1.5)
    ap.add_argument("--out", type=str, default="eds_profile.txt",
                    help="Path to write profile output (default eds_profile.txt in cwd)")
    args = ap.parse_args()

    base_dir = Path(args.dir).resolve()
    manifest_csv = base_dir / "stitch_manifest.csv"
    if not manifest_csv.exists():
        raise SystemExit(f"No stitch_manifest.csv at {manifest_csv}")
    rows = read_manifest(manifest_csv)
    if not rows:
        raise SystemExit("Manifest is empty.")

    # Energy axis: manifest first, then first readable BCF (matches roi_to_banded_maps)
    E0 = dE = None; nE = None
    for r in rows:
        try:
            _nE = int(r.get("nE") or 0)
            if _nE and r.get("dE") is not None and r.get("E0") is not None:
                E0 = float(r["E0"]); dE = float(r["dE"]); nE = _nE
                break
        except Exception:
            continue
    if nE is None:
        print("[profile] Energy axis missing from manifest; probing BCFs…")
        for r in rows[:50]:
            p = _resolve_manifest_path(r["file"], base_dir)
            sig = load_eds_signal_lazy(p)
            if sig is None:
                continue
            meta = extract_eds_metadata_from_signal(sig)
            if meta.get("nE", 0) and meta.get("dE") is not None and meta.get("E0") is not None:
                E0 = float(meta["E0"]); dE = float(meta["dE"]); nE = int(meta["nE"])
                print(f"[profile] Energy axis found in {p.name}: "
                      f"E0={E0:.4f}, dE={dE:.6f}, nE={nE}")
                break
    if nE is None:
        raise SystemExit("No BCF yielded an energy axis; can't profile EDS pipeline.")

    if args.elements.strip():
        elements = [s.strip() for s in args.elements.split(",") if s.strip()]
    else:
        elements = _discover_elements(rows, base_dir)
    elements = [e for e in elements if e in LINE_ENERGIES_KEV]
    if not elements:
        raise SystemExit("No valid elements to profile.")

    # Bands per element
    bands = {}
    for e in elements:
        Ec = LINE_ENERGIES_KEV[e]
        lo_m, hi_m = _band_indices(E0, dE, nE, Ec, args.width)
        Ec_lo, Ec_hi = _sideband_centers(Ec, args.width, args.sideband_mult)
        lo_l, hi_l = _band_indices(E0, dE, nE, Ec_lo, args.width)
        lo_h, hi_h = _band_indices(E0, dE, nE, Ec_hi, args.width)
        bands[e] = {"main": (lo_m, hi_m), "sb_lo": (lo_l, hi_l), "sb_hi": (lo_h, hi_h)}

    # Mosaic-wide ROI
    H_bse = max(int(r["y"]) + int(r["bse_h"]) for r in rows)
    W_bse = max(int(r["x"]) + int(r["bse_w"]) for r in rows)
    roi = (0, H_bse, 0, W_bse)

    # Deterministic sample across the mosaic (uniform stride)
    n = min(max(1, args.n), len(rows))
    step = max(1, len(rows) // n)
    sample = rows[::step][:n]

    print(f"Profiling {n} tiles from {base_dir}")
    print(f"Elements: {elements}  (bands/tile = {3 * len(elements)})")
    print(f"Energy axis: E0={E0:.4f}, dE={dE:.4f}, nE={nE}")
    print(f"Mosaic extent: {H_bse}x{W_bse} px")
    print()

    # Warm-up: first BCF open triggers any lazy imports
    if sample:
        p = _resolve_manifest_path(sample[0]["file"], base_dir)
        _ = load_eds_signal_lazy(p)

    timings: dict = defaultdict(float)

    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()
    for r in sample:
        _process_one_tile_instrumented(r, base_dir, elements, bands,
                                       E0, dE, nE, roi, timings)
    profiler.disable()
    total_wall = time.perf_counter() - t_start

    _print_timings(total_wall, timings, n_tiles=n, n_bands_per_tile=3 * len(elements))

    # cProfile rankings — top by cumulative, top by tottime
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).strip_dirs()
    s.write("\n\n=== cProfile: top 30 by cumulative time ===\n")
    ps.sort_stats("cumulative").print_stats(30)
    s.write("\n=== cProfile: top 30 by tottime ===\n")
    ps.sort_stats("tottime").print_stats(30)
    text = s.getvalue()
    print(text)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Profiling {n} tiles from {base_dir}\n")
        f.write(f"Elements: {elements}\n")
        f.write(f"Bands/tile: {3 * len(elements)}\n")
        f.write(f"Mosaic: {H_bse}x{W_bse}\n\n")
        f.write("=== Per-step breakdown (seconds) ===\n")
        for k in ("bcf_open", "energy_check", "tukey_window",
                  "hyperspy_subcube_read", "hyperspy_band_read",
                  "resize", "shear", "subpixel_shift", "crop_mul"):
            v = timings.get(k, 0.0)
            f.write(f"{k:<24} {v:>10.3f} s  ({100.0*v/total_wall:5.1f}%)\n")
        f.write(f"TOTAL wall: {total_wall:.3f} s\n\n")
        f.write(text)
    print(f"\nWrote full report to {out}")


if __name__ == "__main__":
    main()
