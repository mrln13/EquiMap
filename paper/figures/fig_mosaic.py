#!/usr/bin/env python
"""Figure: the spectrally-aware EDS mosaic cache -- the engineering headline.

A three-panel composite showing that the EDS spectra survive the stitch and stay
queryable from the mosaic cache:

  (a) the stitched backscattered-electron (BSE) mosaic, with the ROI outlined;
  (b) the real overlapping SEM tiles that cover the ROI, read from the stitch
      manifest (zoomed to the ROI);
  (c) a single-element map for that ROI, sliced straight from the EDS mosaic
      cache -- composition recovered from the stitched product without returning
      to the raw tiles.

With no real inputs a synthetic scene is generated and watermarked
"SYNTHETIC DEMO" so the layout can be reviewed before data exist.

Real-data usage (one command -- panel (a) from the stitch manifest, panel (c)
sliced straight from the EDS cache, exactly the read path the manuscript
describes):
    python fig_mosaic.py --dataset path/to/stitched_dir \
        --roi 9000 9000 12000 12000 --element Fe

  --dataset       stitched dataset dir (stitched_BSE_8bit.png + stitch_manifest.csv
                  + eds_cache/); (a) = real tiles over the ROI, (b) = that BSE,
                  (c) = roi_from_cache(...,[element]).
  --roi x0 y0 x1 y1   ROI box in BSE pixel coordinates (outlined in (b)). The ROI
                  label printed by the viewer is "ROI #N [y=y0:y1, x=x0:x1]".
  --element       element symbol to map (default Fe).
  --pixel-um      physical pixel size (um) -> adds a scalebar to each panel
                  (e.g. 0.5 for a 500 nm/px map).
  --roi-ds        decimate the ROI read by this integer stride (for a huge ROI).
  --repo          EquiMap repo root, to import eds_cache (default: two levels
                  above figures/, i.e. the repo root on this workstation).

A smaller ROI (a few hundred to ~3000 px) gives a cleaner panel (a): fewer,
larger tiles whose ~9% overlap is easy to see. A large ROI still works but packs
in many tiles.

Or supply the two rasters yourself, no repo import needed (REAL when both given;
panel (a) then falls back to a schematic tile grid over the ROI):
    python fig_mosaic.py --bse stitched_BSE_8bit.png \
        --roi 1200 800 2200 1700 --element-map fe_roi.npy --element Fe
  --bse           BSE mosaic image (png/tif); shown faint in (a), full in (b).
  --element-map   2-D array for the ROI: a .npy, or an image (png/tif).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import element_colour  # noqa: E402

TILE_COLOUR = "#1f6feb"


def _load_2d(path, gray=True):
    """Load a .npy array or an image file as a 2-D float array."""
    if os.path.splitext(path)[1].lower() == ".npy":
        a = np.load(path)
    else:
        a = plt.imread(path)
    a = np.asarray(a, dtype=float)
    if gray and a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def _load_bse(path, display_max=2400, crop_box=None, crop_max=1500):
    """Load a (possibly very large) BSE mosaic as a uint8 grayscale array,
    downsampled so the longest side is <= ``display_max`` (it is only a thumbnail
    in the figure). Returns ``(array, (W_full, H_full), crop_or_None)``: the ROI
    box stays in full-resolution pixel coordinates, and when ``crop_box`` =
    (x0,y0,x1,y1) is given a separate, crisp full-resolution (capped) crop is
    returned for the zoomed panel (a). The decompression-bomb guard is disabled
    so a billion-pixel mosaic loads instead of raising."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if os.path.splitext(path)[1].lower() == ".npy":
        a = np.load(path)
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        H, W = a.shape[:2]
        crop = None
        if crop_box is not None:
            cx0, cy0, cx1, cy1 = (int(round(v)) for v in crop_box)
            crop = a[cy0:cy1, cx0:cx1]
            cstep = max(1, int(np.ceil(max(crop.shape[:2]) / crop_max)))
            crop = np.asarray(crop[::cstep, ::cstep])
        step = max(1, int(np.ceil(max(H, W) / display_max)))
        return np.asarray(a[::step, ::step]), (W, H), crop
    img = Image.open(path).convert("L")          # grayscale, uint8 -- no float64
    W, H = img.size
    crop = None
    if crop_box is not None:
        cb = tuple(int(round(v)) for v in crop_box)
        c = img.crop(cb)
        cstep = max(1, int(np.ceil(max(c.size) / crop_max)))
        if cstep > 1:
            c = c.reduce(cstep)
        crop = np.asarray(c)
    step = max(1, int(np.ceil(max(W, H) / display_max)))
    small = img.reduce(step) if step > 1 else img
    return np.asarray(small), (W, H), crop


def _read_tiles(dataset):
    """Real per-tile mosaic bounding boxes (x0, y0, x1, y1) from the stitch
    manifest: each tile's solved top-left position plus its BSE width/height.
    Returns an (N, 4) array, or None if no usable manifest is present."""
    import pandas as pd
    path = os.path.join(dataset, "stitch_manifest.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not {"x", "y", "bse_w", "bse_h"}.issubset(df.columns):
        return None
    x0 = df["x"].to_numpy(float)
    y0 = df["y"].to_numpy(float)
    return np.column_stack([x0, y0, x0 + df["bse_w"].to_numpy(float),
                            y0 + df["bse_h"].to_numpy(float)])


def _schematic_tiles(bbox, ncol=4, nrow=4, overlap=0.1, seed=0):
    """Fallback regular grid of overlapping tile bboxes within ``bbox``
    (x0,y0,x1,y1), for the synthetic demo and the manual two-raster route."""
    bx0, by0, bx1, by1 = bbox
    W, H = bx1 - bx0, by1 - by0
    rng = np.random.default_rng(seed)
    tw = W / (1 + (ncol - 1) * (1 - overlap))
    th = H / (1 + (nrow - 1) * (1 - overlap))
    sx, sy = tw * (1 - overlap), th * (1 - overlap)
    out = []
    for j in range(nrow):
        for i in range(ncol):
            ox = bx0 + i * sx + rng.uniform(-1, 1) * 0.04 * tw
            oy = by0 + j * sy + rng.uniform(-1, 1) * 0.04 * th
            out.append((ox, oy, ox + tw, oy + th))
    return np.array(out, float)


def _tiles_in_roi(tiles, roi, margin_frac=0.0):
    """Subset of tile bboxes intersecting the ROI (grown by ``margin_frac`` of the
    ROI size), and the union bbox of that subset (the crop extent for panel a)."""
    x0, y0, x1, y1 = roi
    mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
    qx0, qy0, qx1, qy1 = x0 - mx, y0 - my, x1 + mx, y1 + my
    m = ((tiles[:, 0] < qx1) & (tiles[:, 2] > qx0) &
         (tiles[:, 1] < qy1) & (tiles[:, 3] > qy0))
    sub = tiles[m]
    if len(sub) == 0:
        return sub, (x0, y0, x1, y1)
    union = (float(sub[:, 0].min()), float(sub[:, 1].min()),
             float(sub[:, 2].max()), float(sub[:, 3].max()))
    return sub, union


def _match_aspect(box, roi):
    """Grow ``box`` (x0,y0,x1,y1), centred, to the width:height aspect of the ROI,
    so panels (b) and (c) share proportions and therefore render at equal height."""
    bx0, by0, bx1, by1 = box
    target = (roi[2] - roi[0]) / (roi[3] - roi[1]) if (roi[3] - roi[1]) else 1.0
    w, h = bx1 - bx0, by1 - by0
    if w / h < target:                  # too tall -> widen
        nw = h * target
        cx = 0.5 * (bx0 + bx1)
        return (cx - nw / 2, by0, cx + nw / 2, by1)
    nh = w / target                     # too wide -> heighten
    cy = 0.5 * (by0 + by1)
    return (bx0, cy - nh / 2, bx1, cy + nh / 2)


def _element_map_from_cache(dataset, repo, roi_xyxy, element, roi_ds):
    """Slice a single-element net-intensity map for the ROI directly from the EDS
    mosaic cache -- the exact read path the manuscript describes. Net intensity
    is the peak band minus the mean of the two background side-bands; the cache
    handles any ``ds>1`` (downsampled / two-resolution) layout internally.
    Returns ``(net_map, clipped_roi)``."""
    import sys as _sys
    from pathlib import Path
    repo = repo or os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    if repo not in _sys.path:
        _sys.path.insert(0, repo)
    try:
        import eds_cache
    except ImportError as exc:
        raise SystemExit(f"could not import eds_cache from repo '{repo}': {exc}\n"
                         "pass --repo <EquiMap repo root>.")
    x0, y0, x1, y1 = map(int, roi_xyxy)
    cache_dir = eds_cache.resolve_cache_dir(Path(dataset))
    man = eds_cache.load_manifest(cache_dir) or {}
    avail = man.get("elements") or []
    if avail and element not in avail:
        raise SystemExit(f"element {element!r} is not in the cache; "
                         f"available: {', '.join(avail)}")
    # The ROI is in BSE pixels. roi_from_cache does NOT clip, so an out-of-bounds
    # box yields an empty slice and a cryptic broadcast error -- clip here and
    # report it clearly instead.
    W = int(man.get("W_bse") or man.get("W") or 0)
    H = int(man.get("H_bse") or man.get("H") or 0)
    if W and H:
        cx0, cx1 = max(0, min(x0, W)), max(0, min(x1, W))
        cy0, cy1 = max(0, min(y0, H)), max(0, min(y1, H))
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            print(f"note: ROI clipped to the {W}x{H} mosaic "
                  f"(x in [0,{W}], y in [0,{H}]): "
                  f"x {x0}:{x1}->{cx0}:{cx1}, y {y0}:{y1}->{cy0}:{cy1}")
        x0, y0, x1, y1 = cx0, cy0, cx1, cy1
    if x1 <= x0 or y1 <= y0:
        raise SystemExit(f"ROI {tuple(roi_xyxy)} is outside the {W}x{H} mosaic; "
                         f"pick x in [0,{W}], y in [0,{H}].")
    banded = eds_cache.roi_from_cache(cache_dir, (y0, y1, x0, x1), [element],
                                      blend_mode="mean", progress=False,
                                      out_ds=max(1, int(roi_ds)))
    # Line-aware background needs the anchor elements' bands as well (the
    # cache reader opens them; roi_from_cache only returns what is asked).
    from eds_background import anchor_elements, background_plan, net_count_maps
    cached = list(eds_cache.load_manifest(cache_dir).get("elements") or [element])
    plan = background_plan(cached)
    extra = [e for e in anchor_elements(plan, [element]) if e in cached and e not in banded]
    if extra:
        banded.update(eds_cache.roi_from_cache(cache_dir, (y0, y1, x0, x1), extra,
                                               blend_mode="mean", progress=False,
                                               out_ds=max(1, int(roi_ds))))
    net = net_count_maps(banded, plan=plan, elements=[element])[element].astype(float)
    return net, (x0, y0, x1, y1)


def _synthetic_scene(seed: int = 0, H: int = 520, W: int = 760):
    """A plausible BSE mosaic + co-registered Fe map for layout preview.

    A few smooth random fields are thresholded into ``minerals``; each gets a
    mean-Z (BSE brightness) and an Fe content, so the BSE mosaic and the Fe map
    share spatial structure exactly as a real dataset would.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    g = (gaussian_filter(rng.standard_normal((H, W)), sigma=18)
         + 0.6 * gaussian_filter(rng.standard_normal((H, W)), sigma=10))
    q = np.quantile(g, [0.35, 0.70, 0.90])
    label = np.digitize(g, q)                      # 0..3: qz, fsp, mafic, oxide
    bse_mean = np.array([60, 110, 150, 225], float)[label]
    fe_mean = np.array([1.0, 3.0, 14.0, 55.0], float)[label]
    bse = np.clip(bse_mean + gaussian_filter(rng.standard_normal((H, W)), 1.5) * 8,
                  0, 255)
    fe = np.clip(fe_mean + gaussian_filter(rng.standard_normal((H, W)), 1.5) * 1.5,
                 0, None)
    roi = (int(W * 0.30), int(H * 0.32), int(W * 0.66), int(H * 0.74))
    x0, y0, x1, y1 = roi
    return bse, fe[y0:y1, x0:x1], roi, "Fe"


def _nice_bar_um(target):
    """Largest 'nice' (1/2/5 x 10^n) length in micrometres not exceeding the
    target (~a quarter of the panel width)."""
    import math
    if target <= 0:
        return 1.0
    p = 10.0 ** math.floor(math.log10(target))
    best = p
    for c in (1, 2, 5, 10):
        if c * p <= target:
            best = c * p
    return best


def _fmt_um(um):
    return f"{um / 1000:g} mm" if um >= 1000 else f"{um:g} " + r"$\mu$m"


def _draw_scalebar(ax, pixel_um, frac=0.28):
    """Add a labelled scalebar (lower-right) sized from the axis's own x-span,
    given the physical pixel size in micrometres."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm
    xl, yl = ax.get_xlim(), ax.get_ylim()
    bar_um = _nice_bar_um(abs(xl[1] - xl[0]) * pixel_um * frac)
    sb = AnchoredSizeBar(ax.transData, bar_um / pixel_um, _fmt_um(bar_um),
                         "lower right", pad=0.3, borderpad=0.4, sep=3,
                         frameon=True, color="white",
                         size_vertical=0.012 * abs(yl[1] - yl[0]),
                         fontproperties=fm.FontProperties(size=7))
    sb.patch.set(facecolor="black", alpha=0.35, edgecolor="none")
    ax.add_artist(sb)


def make_figure(bse, bse_dims, bse_a, extent_a, tiles, crop_a,
                elem_roi, roi, element, out, dpi, synthetic, pixel_um=None):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    })
    W, H = bse_dims                     # full-res mosaic dims; bse may be smaller
    roi_colour = element_colour(element)
    x0, y0, x1, y1 = roi
    aspect_a = (W / H) if H else 1.6
    aspect_p = ((x1 - x0) / (y1 - y0)) if (y1 - y0) else 1.5

    # Explicit layout (units = fraction of figure WIDTH; the figure height is
    # solved so every panel fills its box with no letterboxing). Panel (a) spans
    # exactly the (b)+(c) width, so (b) aligns to (a)'s left edge and (c) plus its
    # colorbar to (a)'s right edge; (b) and (c) are equal and as large as the
    # width allows, with only a small gap between them.
    fig_w = 7.2
    mL = mR = 0.015
    cb_gap, cb_w, cb_lab = 0.012, 0.02, 0.09        # colorbar gap / width / labels
    gap_bc = 0.022                                  # gap between (b) and (c)
    m_top, t_a, v_gap, m_bot = 0.02, 0.05, 0.065, 0.02
    xL = mL
    xR = 1.0 - mR - (cb_gap + cb_w + cb_lab)
    Wp = xR - xL                                    # (a) span == (b)+(c) span
    wB = (Wp - gap_bc) / 2.0
    hA = Wp / aspect_a
    hB = wB / aspect_p
    R = m_top + t_a + hA + v_gap + hB + m_bot       # figure height in width-units
    yA = m_top + t_a
    yBC = yA + hA + v_gap

    fig = plt.figure(figsize=(fig_w, fig_w * R))

    def _ax(x, top, w, h):                          # place by top edge (width units)
        return fig.add_axes([x, (R - top - h) / R, w, h / R])

    axA = _ax(xL, yA, Wp, hA)
    axB = _ax(xL, yBC, wB, hB)
    axC = _ax(xL + wB + gap_bc, yBC, wB, hB)
    cax = fig.add_axes([xR + cb_gap, (R - yBC - hB) / R, cb_w, hB / R])

    # -- (a) stitched BSE mosaic with ROI -------------------------------------
    # interpolation="none" on all three panels: any other setting makes the
    # PDF backend RESAMPLE the image to the savefig dpi; "none" embeds each
    # array at its native resolution and lets the viewer scale it.
    axA.imshow(bse, cmap="gray", vmin=0, vmax=255, extent=(0, W, H, 0),
               aspect="auto", interpolation="none")
    axA.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor=roi_colour, lw=2.4))
    axA.set_xlim(0, W); axA.set_ylim(H, 0)
    axA.set_title("(a) stitched BSE mosaic")
    axA.set_xticks([]); axA.set_yticks([])

    # -- (b) the real overlapping tiles that cover the ROI --------------------
    va, vb = (float(v) for v in np.percentile(bse_a, [1, 99]))  # local contrast
    if vb <= va:
        va, vb = float(np.min(bse_a)), float(np.max(bse_a))
        vb = vb if vb > va else va + 1.0
    axB.imshow(bse_a, cmap="gray", vmin=va, vmax=vb, extent=extent_a,
               aspect="auto", interpolation="none")
    for tx0, ty0, tx1, ty1 in tiles:    # outline only; overlaps show as doubled lines
        axB.add_patch(Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, fill=False,
                                edgecolor=TILE_COLOUR, lw=1.1, alpha=0.9))
    axB.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                            edgecolor=roi_colour, lw=2.2, linestyle=(0, (4, 2))))
    axB.set_xlim(crop_a[0], crop_a[2])
    axB.set_ylim(crop_a[3], crop_a[1])
    axB.set_title("(b) overlapping tiles over ROI")
    axB.set_xticks([]); axB.set_yticks([])

    # -- (c) element map for the ROI, sliced from the cache -------------------
    # Extent = the full-res ROI pixel span, so a scalebar uses the same pixel
    # size as (a)/(b) even when the map was read decimated (--roi-ds).
    # Net counts are signed after background subtraction (noise about zero
    # where the element is absent); clip at zero for display, as the viewer does.
    im = axC.imshow(np.clip(elem_roi, 0, None), cmap="inferno", aspect="auto",
                    vmin=0, extent=(0, x1 - x0, y1 - y0, 0), interpolation="none")
    for sp in axC.spines.values():        # frame the panel in the ROI colour
        sp.set_color(roi_colour); sp.set_linewidth(2.4)
    axC.set_title(f"(c) {element} map from cache (ROI)")
    axC.set_xticks([]); axC.set_yticks([])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"{element} (a.u.)" if synthetic
                 else f"{element} K$\\alpha$ net counts per pixel")
    cb.ax.tick_params(labelsize=7)

    if pixel_um:
        for ax in (axA, axB, axC):
            _draw_scalebar(ax, pixel_um)

    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DEMO", fontsize=34, color="0.85",
                 ha="center", va="center", rotation=18, zorder=5, alpha=0.6)

    fig.savefig(out, dpi=dpi)
    print(f"wrote {out}  (BSE {W}x{H}, ROI {x1 - x0}x{y1 - y0}, {len(tiles)} tiles "
          f"over ROI, element {element}" + ("; SYNTHETIC" if synthetic else "") + ")")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="stitched dataset dir (stitched_BSE_8bit.png + "
                         "stitch_manifest.csv + eds_cache/); needs --roi")
    ap.add_argument("--repo", default=None,
                    help="EquiMap repo root for importing eds_cache "
                         "(default: two levels above figures/)")
    ap.add_argument("--roi-ds", type=int, default=1,
                    help="decimate the ROI element-map read by this stride")
    ap.add_argument("--bse", default=None,
                    help="BSE mosaic image (png/tif)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=None, help="ROI box in BSE pixels")
    ap.add_argument("--element-map", default=None,
                    help="ROI element map (.npy or image)")
    ap.add_argument("--element", default="Fe", help="element symbol (default Fe)")
    ap.add_argument("--pixel-um", type=float, default=None,
                    help="physical pixel size in micrometres; adds a scalebar to "
                         "each panel (e.g. 0.5 for 500 nm/px)")
    ap.add_argument("--out", default=os.path.join(here, "fig_mosaic.pdf"))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    element = args.element
    if args.dataset:
        if not args.roi:
            raise SystemExit("--roi X0 Y0 X1 Y1 is required with --dataset")
        bse_path = args.bse or os.path.join(args.dataset, "stitched_BSE_8bit.png")
        if not os.path.exists(bse_path):
            raise SystemExit(f"BSE not found: {bse_path} (pass --bse explicitly)")
        # Validate ROI + read the (fast) cache slice FIRST, so a bad ROI fails
        # immediately instead of after the slow full-mosaic BSE decode.
        elem_roi, roi = _element_map_from_cache(args.dataset, args.repo, args.roi,
                                                element, args.roi_ds)
        tiles_all = _read_tiles(args.dataset)
        if tiles_all is None or len(tiles_all) == 0:
            print("note: no usable stitch_manifest.csv; panel (a) uses a "
                  "schematic tile grid.")
            tiles, crop_a = _schematic_tiles(roi), roi
        else:
            tiles, crop_a = _tiles_in_roi(tiles_all, roi, margin_frac=0.12)
        crop_a = _match_aspect(crop_a, roi)   # so (b) and (c) end up equal height
        bse, bse_dims, bse_a = _load_bse(bse_path, crop_box=crop_a)
        if bse_a is None:
            bse_a, extent_a = bse, (0, bse_dims[0], bse_dims[1], 0)
        else:                           # matplotlib extent = (left, right, bottom, top)
            extent_a = (crop_a[0], crop_a[2], crop_a[3], crop_a[1])
        synthetic = False
    elif args.bse and args.element_map:
        bse, bse_dims, _ = _load_bse(args.bse)
        elem_roi = _load_2d(args.element_map, gray=True)
        W, H = bse_dims
        roi = tuple(args.roi) if args.roi else (0, 0, W, H)
        tiles, crop_a = _schematic_tiles(roi), roi
        bse_a, extent_a = bse, (0, W, H, 0)
        synthetic = False
    else:
        if args.bse or args.element_map or args.dataset:
            print("note: need --dataset (+ --roi), or BOTH --bse and "
                  "--element-map, for a real figure; using synthetic data.")
        bse, elem_roi, roi, element = _synthetic_scene()
        bse_dims = (bse.shape[1], bse.shape[0])
        tiles_all = _schematic_tiles((0, 0, bse_dims[0], bse_dims[1]),
                                     ncol=8, nrow=6)
        tiles, crop_a = _tiles_in_roi(tiles_all, roi, margin_frac=0.25)
        crop_a = _match_aspect(crop_a, roi)
        bse_a, extent_a = bse, (0, bse_dims[0], bse_dims[1], 0)
        synthetic = True

    make_figure(bse, bse_dims, bse_a, extent_a, tiles, crop_a, elem_roi, roi,
                element, args.out, args.dpi, synthetic, args.pixel_um)


if __name__ == "__main__":
    main()
