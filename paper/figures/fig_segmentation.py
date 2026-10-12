"""Figure: phase segmentation of the worked example (fig:segmentation).

(a) full-section phase-domain map (classification-dialog export, six domains
    incl. the quartz-mica matrix domain that is NNLS-unmixed for the modal
    budget), with the zoom region boxed;
(b) enlargement of a muscovite-nodule cluster, where the epidote cores are
    resolved as their own domain.

The zoom window is picked automatically as the densest epidote region of the
exported label map, so the figure regenerates deterministically from data.

Inputs (this directory; exported from the classification dialog):
  phase_map_labels.npz     integer label map (segmentation grid; -1 = unassigned;
                           compressed, key "labels"); rendered here with a
                           palette keyed by domain NAME (PALETTE), so maps
                           from different runs share colours per mineral
  phase_legend.csv         cluster id -> name
  phase_map_inset.{npz,json}  full-resolution nodule inset (fig_segmentation_inset.py)

Usage:  python fig_segmentation.py [--pixel-um 2.5] [--out fig_segmentation.pdf]
        (--pixel-um is the PHASE-MAP pixel pitch: mosaic pitch x seg_ds,
         e.g. 0.5 um x 5 = 2.5 um for the worked example)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
EPIDOTE_NAME = "Epidote"
ZOOM_PX = 900          # zoom window edge, phase-map pixels (2.25 mm at 2.5 um)

# Colours are keyed by DOMAIN NAME, not by cluster id, so that maps from
# different runs show the same mineral in the same colour. The colours are
# the tab20 entries the manuscript's July figure happened to give each
# phase (cluster ids 0-5 of that run), kept because they read well; new
# domain types take unused tab20 entries.
PALETTE = {
    "biotite": (0.122, 0.467, 0.706), "quartz-mica matrix": (0.682, 0.78, 0.91), "quartz-rich matrix": (0.682, 0.78, 0.91),
    "muscovite": (1.0, 0.498, 0.055), "quartz": (1.0, 0.733, 0.471), "plagioclase": (0.173, 0.627, 0.173), "epidote": (0.596, 0.875, 0.541),
    "sillimanite": (0.839, 0.153, 0.157), "mica-rich matrix": (1.0, 0.596, 0.588), "alkali feldspar": (0.58, 0.404, 0.741),
    "chlorite": (0.773, 0.69, 0.835), "mixed": (0.78, 0.78, 0.78),
}
# Alternative palette (distinct hues per mineral; matrix domains in greys).
PALETTE_ALT = {
    "biotite": (0.15, 0.45, 0.15), "muscovite": (0.20, 0.40, 0.80),
    "quartz": (0.96, 0.92, 0.65), "plagioclase": (0.60, 0.85, 0.60),
    "epidote": (0.85, 0.20, 0.20), "sillimanite": (0.55, 0.10, 0.60),
    "alkali feldspar": (0.95, 0.60, 0.20), "chlorite": (0.10, 0.60, 0.60),
    "quartz-mica matrix": (0.74, 0.74, 0.74), "quartz-rich matrix": (0.74, 0.74, 0.74),
    "mica-rich matrix": (0.45, 0.45, 0.45), "mixed": (0.30, 0.30, 0.30),
}
# Hybrid (default): the manuscript's mineral hues, but the mixed (matrix)
# domains in greys so they read as "not a single phase", which is what the
# unmixing step says about them.
PALETTE_HYBRID = dict(PALETTE)
PALETTE_HYBRID.update({"quartz-mica matrix": (0.74, 0.74, 0.74), "quartz-rich matrix": (0.74, 0.74, 0.74),
                       "mica-rich matrix": (0.45, 0.45, 0.45), "mixed": (0.30, 0.30, 0.30)})
PALETTES = {"hybrid": PALETTE_HYBRID, "manuscript": PALETTE, "alt": PALETTE_ALT}
MATRIX_GREYS = [(0.682, 0.78, 0.91), (1.0, 0.596, 0.588), (0.78, 0.78, 0.78)]


def colour_for(name, matrix_index, palette=None):
    """RGB for a domain name (longest matching palette key wins, so
    'quartz-rich matrix' is not read as 'quartz'); unknown mixed/matrix
    domains take successive spare entries."""
    pal = palette or PALETTE
    n = name.strip().lower()
    best = max((k for k in pal if k in n), key=len, default=None)
    if best is not None:
        return pal[best]
    if "matrix" in n or "mixed" in n:
        return MATRIX_GREYS[min(matrix_index, len(MATRIX_GREYS) - 1)]
    return (1.0, 0.0, 1.0)


def name_colours(names, palette=None):
    """{cluster id: RGB} for an ordered {cid: name} mapping (greys in order)."""
    out, k = {}, 0
    for cid, nm in sorted(names.items()):
        c = colour_for(nm, k, palette)
        if "matrix" in nm.lower() or "mixed" in nm.lower():
            k += 1
        out[int(cid)] = c
    return out


def read_legend(data_dir=None):
    with open((data_dir or HERE) / "phase_legend.csv") as f:
        return list(csv.DictReader(f))


def densest_window(mask, win):
    """Top-left corner of the win x win window with the most True pixels."""
    from scipy.ndimage import uniform_filter
    ds = 4
    m = mask[::ds, ::ds].astype(np.float32)
    dens = uniform_filter(m, size=max(3, win // ds))
    iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
    y0 = int(iy * ds - win / 2)
    x0 = int(ix * ds - win / 2)
    y0 = max(0, min(mask.shape[0] - win, y0))
    x0 = max(0, min(mask.shape[1] - win, x0))
    return y0, x0


def labels_to_rgb(labels, colours):
    """uint8 RGB image of a label map; unassigned (-1) and unknown ids white."""
    rgb = np.full(labels.shape + (3,), 255, dtype=np.uint8)
    for cid in np.unique(labels):
        if cid < 0:
            continue
        c = colours.get(int(cid))
        if c is not None:
            rgb[labels == cid] = tuple(int(round(255 * v)) for v in c)
    return rgb


def scalebar(ax, px_um, frac=0.25):
    x0, x1 = ax.get_xlim()
    span_um = abs(x1 - x0) * px_um
    target = span_um * frac
    nice = min([l for l in (50, 100, 200, 500, 1000, 2000, 5000, 10000)
                if l >= target] or [10000])
    label = f"{nice/1000:g} mm" if nice >= 1000 else f"{nice:g} µm"
    y0, y1 = ax.get_ylim()
    h = abs(y0 - y1)
    w = nice / px_um
    bx = x1 - 0.04 * abs(x1 - x0) - w
    by = max(y0, y1) - 0.07 * h
    ax.add_patch(Rectangle((bx, by), w, 0.018 * h, fc="white", ec="black",
                           lw=0.5, zorder=5))
    ax.text(bx + w / 2, by - 0.02 * h, label, ha="center", va="bottom",
            fontsize=8, zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel-um", type=float, default=None,
                    help="phase-map pixel pitch in µm (mosaic pitch × seg_ds)")
    ap.add_argument("--map-ds", type=int, default=1,
                    help="decimate the overview image (1 = full phase-map "
                         "resolution; nearest-pixel decimation aliases the "
                         "fine matrix speckle, so keep 1 unless the PDF "
                         "must be smaller)")
    ap.add_argument("--zoom-px", type=int, default=ZOOM_PX)
    ap.add_argument("--out", default=str(HERE / "fig_segmentation.pdf"))
    ap.add_argument("--palette", choices=list(PALETTES), default="hybrid",
                    help="colour scheme: 'hybrid' (default: manuscript mineral hues, matrix domains "
                         "in greys), 'manuscript' (the July tab20 assignment) or 'alt' (distinct hues)")
    ap.add_argument("--data-dir", default=str(HERE),
                    help="folder holding phase_map_labels.npz / phase_legend.csv / "
                         "phase_map_inset.* (default: this folder)")
    args = ap.parse_args()
    DATA = Path(args.data_dir)

    legend = read_legend(DATA)
    names = {int(r["cluster_id"]): r["name"].strip() for r in legend}
    palette = PALETTES[args.palette]
    colours = name_colours(names, palette)

    labels = np.load(DATA / "phase_map_labels.npz")["labels"]

    # Panel (b): full-resolution sigma=1 local re-segmentation when the
    # fig_segmentation_inset.py products exist; else fall back to cropping
    # the (coarser) global map around the densest epidote region.
    inset_npz = DATA / "phase_map_inset.npz"
    inset_json = DATA / "phase_map_inset.json"
    if inset_npz.exists() and inset_json.exists():
        import json
        meta = json.loads(inset_json.read_text())
        extra_names = {int(i): nm for i, nm in (meta.get("extra_names") or {}).items()}
        crop_full = np.load(inset_npz)["labels"]
        y0, y1, x0, x1 = meta["window_mosaic_px"]
        seg_ds = 5  # global map grid vs mosaic px (segmentation_params.json)
        zy, zx = y0 // seg_ds, x0 // seg_ds
        zoom_px_global = (y1 - y0) // seg_ds
        zoom_pitch_factor = 1.0   # inset is at mosaic resolution
    else:
        extra_names = {}
        pref = [r for r in legend if r["name"].strip() in (EPIDOTE_NAME, "Sillimanite")]
        ep_cid = int((pref or sorted(legend, key=lambda r: int(r["pixel_count"])))[0]["cluster_id"])
        zy, zx = densest_window(labels == ep_cid, args.zoom_px)
        crop_full = labels[zy:zy + args.zoom_px, zx:zx + args.zoom_px]
        zoom_px_global = args.zoom_px
        zoom_pitch_factor = None  # global-map pitch

    # Panel (a) rendered from the label map with the name-keyed palette (the
    # viewer's phase_map_combined.png is tab20 by cluster id, which differs
    # between runs). Inset-only phases get their own colours too.
    all_colours = dict(colours)
    for i, nm in extra_names.items():
        all_colours[int(i)] = colour_for(nm, 0, palette)
    overview = labels_to_rgb(labels, all_colours)
    if args.map_ds > 1:
        overview = overview[::args.map_ds, ::args.map_ds]

    H, W = labels.shape
    aspect = W / H
    h_in = 2.55
    fig = plt.figure(figsize=(h_in * aspect + h_in + 0.25, h_in + 0.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[aspect, 1.0], wspace=0.04)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # interpolation="none" is load-bearing: with "nearest" (or any real
    # interpolation) matplotlib's PDF backend RESAMPLES the image down to
    # the figure dpi (default 100 -> ~470 px wide), which is why an earlier
    # version looked pixelated regardless of the input resolution. "none"
    # embeds the full-resolution array and lets the viewer scale it.
    ax_a.imshow(overview, interpolation="none", aspect="equal",
                extent=(0, W, H, 0))
    ax_a.add_patch(Rectangle((zx, zy), zoom_px_global, zoom_px_global,
                             fill=False, ec="black", lw=1.6, zorder=4))
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("(a) phase-domain map", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, fc=colours[int(r["cluster_id"])])
               for r in legend]
    labels_txt = [r["name"] for r in legend]
    for i, nm in sorted(extra_names.items()):      # phases resolved only in (b)
        handles.append(Rectangle((0, 0), 1, 1, fc=all_colours[int(i)]))
        labels_txt.append(f"{nm} (b only)")
    ax_a.legend(handles, labels_txt, loc="lower left",
                fontsize=7.5, framealpha=0.85, ncol=2, handlelength=1.0,
                columnspacing=0.9)
    if args.pixel_um:
        scalebar(ax_a, args.pixel_um)

    crop = crop_full[::2, ::2] if crop_full.shape[0] > 3000 else crop_full
    ax_b.imshow(labels_to_rgb(crop, all_colours), interpolation="none",
                aspect="equal", extent=(0, crop.shape[1], crop.shape[0], 0))
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title("(b) muscovite nodules", fontsize=10)
    for side in ax_b.spines.values():
        side.set_linewidth(1.6)
    if args.pixel_um:
        # Inset pitch: mosaic pixels (pixel_um / seg_ds), decimated by the
        # display stride; fallback crop uses the global-map pitch.
        ds_disp = 2 if crop_full.shape[0] > 3000 else 1
        if zoom_pitch_factor is not None:
            pitch = (args.pixel_um / 5.0) * ds_disp
        else:
            pitch = args.pixel_um * ds_disp
        scalebar(ax_b, pitch, frac=0.3)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}  (zoom at y={zy}, x={zx}, {args.zoom_px} px)")


if __name__ == "__main__":
    main()
