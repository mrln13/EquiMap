"""Figures: EBSD-vs-EDS validation on the serpentinite maps
(fig:ebsd + fig:ebsd-conf — two floats; the elsarticle preprint page cannot
hold maps + detail + confusion + caption in one).

fig_ebsd.pdf (fig:ebsd):
(a) EBSD-indexed phase map of the second (larger) serpentinite map;
(b) unsupervised EDS segmentation of the same area (sigma=2, min region 10,
    run independently of the EBSD); locator boxes mark the detail crop.
fig_ebsd_conf.pdf (fig:ebsd-conf):
(a,b) paired high-res detail of an auto-selected crop (dense in ilmenite +
    magnetite grains and containing a diopside-chlorite boundary) — the
    trace oxides are sub-printable at full-map scale, so this is where the
    per-grain agreement and the boundary-pixel disagreement become visible;
(c,d) row-normalised confusion matrices for both maps (EBSD rows = truth,
    EDS domains as columns labelled with their ratio-only auto-ID name,
    reordered so each phase's majority domain sits on the diagonal).

Inputs (this directory, staged by ebsd_confusion_prep.py):
  ebsd_val_serpdi_{1,2}_EBSD.npz   eds_labels (int16) + ebsd ordinals (int8)
  serpdi_{1,2}_EBSD_seg_meta.json  auto-ID names per domain + EBSD legend

The confusion statistics are recomputed here with the toolkit's own
ebsd_validation.confusion_matrix (imported from the repo root), so the
figure and the numbers in the text share one code path.

Usage:  python fig_ebsd.py [--repo <root>] [--out fig_ebsd.pdf]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
TAGS = ("serpdi_1_EBSD", "serpdi_2_EBSD")
SHORT = {"serpdi_1_EBSD": "map 1", "serpdi_2_EBSD": "map 2"}
MAP_TAG = "serpdi_2_EBSD"      # the map shown in panels (a)/(b)
PIXEL_UM = 6.0


def scalebar(ax, px_um, frac=0.2):
    x0, x1 = ax.get_xlim()
    span_um = abs(x1 - x0) * px_um
    target = span_um * frac
    nice = min([l for l in (500, 1000, 2000, 5000, 10000, 20000)
                if l >= target] or [20000])
    label = f"{nice / 1000:g} mm"
    y0, y1 = ax.get_ylim()
    h = abs(y0 - y1)
    w = nice / px_um
    bx = x1 - 0.03 * abs(x1 - x0) - w
    by = max(y0, y1) - 0.10 * h
    ax.add_patch(Rectangle((bx, by), w, 0.025 * h, fc="white", ec="black",
                           lw=0.5, zorder=5))
    ax.text(bx + w / 2, by - 0.03 * h, label, ha="center", va="bottom",
            fontsize=8, zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))


def load(tag):
    z = np.load(HERE / f"ebsd_val_{tag}.npz")
    meta = json.loads((HERE / f"{tag}_seg_meta.json").read_text())
    eds_names = {int(k): v for k, v in meta["eds_names"].items()}
    ebsd_names = {int(k): v["name"] for k, v in meta["ebsd_phases"].items()
                  if int(k) > 0}
    ebsd_colors = {int(k): np.array(v["color"], dtype=float) / 255.0
                   for k, v in meta["ebsd_phases"].items()}
    return z["eds_labels"], z["ebsd"], eds_names, ebsd_names, ebsd_colors


def ebsd_rgb(ebsd, colors):
    rgb = np.zeros(ebsd.shape + (3,), dtype=np.float32)
    for o, c in colors.items():
        if o > 0:
            rgb[ebsd == o] = c[:3]
    return rgb


GREY = np.array([0.55, 0.55, 0.55])


def majority_colors(res, ebsd_colors, eds_names):
    """EDS-domain colour = colour of the EBSD phase it captures.

    Each EBSD phase's pixels concentrate in one domain (the majority
    assignment of the purity analysis); that domain takes the phase's EBSD
    legend colour, so panels (a) and (b) agree visually where the maps
    agree. A domain capturing no phase majority (boundary pixels / the
    unindexed serpentine matrix) is grey. Ties go to the larger phase.
    """
    colors = {cid: GREY for cid in eds_names}
    claimed = {}
    order = np.argsort(-res.counts.sum(axis=1))       # larger phases first
    for i in order:
        cid = res.col_cids[int(np.argmax(res.counts[i]))]
        if cid not in claimed:
            claimed[cid] = res.row_ordinals[i]
    for cid, o in claimed.items():
        colors[cid] = ebsd_colors[o][:3]
    return colors


def eds_rgb(labels, colors):
    rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
    for cid in np.unique(labels):
        if cid >= 0:
            rgb[labels == cid] = colors[int(cid)]
    return rgb


CROP_H, CROP_W = 320, 640          # px; 1.92 x 3.84 mm at 6 um/px


def _box_counts(mask, hc, wc, ys, xs):
    """Pixel counts of `mask` inside each (hc x wc) window at (ys x xs)."""
    integ = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(mask, axis=0), axis=1, out=integ[1:, 1:])
    return (integ[np.ix_(ys + hc, xs + wc)] - integ[np.ix_(ys, xs + wc)]
            - integ[np.ix_(ys + hc, xs)] + integ[np.ix_(ys, xs)])


def select_crop(ebsd, ebsd_names, hc=CROP_H, wc=CROP_W, stride=32):
    """Deterministic zoom-window choice: maximise trace-oxide pixels, require
    both oxides present and a substantial diopside AND chlorite share (so the
    crop contains a major-phase boundary, where the impurity lives)."""
    by_name = {v: k for k, v in ebsd_names.items()}
    H, W = ebsd.shape
    ys = np.arange(0, H - hc + 1, stride)
    xs = np.arange(0, W - wc + 1, stride)
    n_ilm = _box_counts(ebsd == by_name["Ilmenite"], hc, wc, ys, xs)
    n_mag = _box_counts(ebsd == by_name["Magnetite"], hc, wc, ys, xs)
    n_dio = _box_counts(ebsd == by_name["Diopside"], hc, wc, ys, xs)
    n_chl = _box_counts(ebsd == by_name["Chlorite"], hc, wc, ys, xs)
    area = hc * wc
    ok = ((n_dio > 0.10 * area) & (n_chl > 0.10 * area)
          & (n_ilm > 100) & (n_mag > 100))
    score = np.where(ok, n_ilm + n_mag, -1)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    y0, x0 = int(ys[i]), int(xs[j])
    print(f"crop y={y0}:{y0 + hc} x={x0}:{x0 + wc}  "
          f"ilm {n_ilm[i, j]} mag {n_mag[i, j]} px")
    return y0, x0


def locator(ax, y0, x0, hc=CROP_H, wc=CROP_W):
    ax.add_patch(Rectangle((x0, y0), wc, hc, fill=False, ec="black",
                           lw=2.2, zorder=6))
    ax.add_patch(Rectangle((x0, y0), wc, hc, fill=False, ec="white",
                           lw=1.0, zorder=7))


def confusion_order(res):
    """Display order for the EDS-domain columns: each EBSD phase's majority
    domain at its own row's position (so agreement reads as a diagonal and
    both maps show the same phase order), no-majority domains last. The cid
    numbering itself is arbitrary GMM component labelling."""
    order = []
    for i in range(res.counts.shape[0]):
        j = int(np.argmax(res.counts[i]))
        if j not in order:
            order.append(j)
    order += [j for j in range(res.counts.shape[1]) if j not in order]
    return order


def confusion_panel(ax, res, title):
    order = confusion_order(res)
    rn = 100.0 * res.row_normalised()[:, order]
    R, C = rn.shape
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(R):
        for j in range(C):
            v = rn[i, j]
            ax.text(j, i, f"{v:.0f}" if v >= 0.5 else "·",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else "black")
        for j in res.matched_cols.get(i, []):
            ax.add_patch(Rectangle((order.index(j) - 0.5, i - 0.5), 1, 1,
                                   fill=False, ec="#1a7f37", lw=1.8,
                                   zorder=4))
    ax.set_xticks(range(C))
    # Cluster ids are each map's own GMM numbering and mean nothing across
    # maps; show them only when two domains share an auto-ID name.
    names = [res.col_labels[j] for j in order]
    dup = len(set(names)) < len(names)
    ax.set_xticklabels([f"{res.col_labels[j]} ({res.col_cids[j]})" if dup
                        else res.col_labels[j] for j in order], fontsize=7.5,
                       rotation=35, ha="right")
    ax.set_yticks(range(R))
    ax.set_yticklabels(res.row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for side in ax.spines.values():
        side.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent.parent))
    ap.add_argument("--out", default=str(HERE / "fig_ebsd.pdf"))
    ap.add_argument("--out2", default=str(HERE / "fig_ebsd_conf.pdf"))
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from ebsd_validation import confusion_matrix

    results = {}
    for tag in TAGS:
        eds, ebsd, eds_names, ebsd_names, _ = load(tag)
        results[tag] = confusion_matrix(ebsd, eds, ebsd_names, eds_names)

    eds, ebsd, eds_names, ebsd_names, ebsd_colors = load(MAP_TAG)
    dom_colors = majority_colors(results[MAP_TAG], ebsd_colors, eds_names)
    H, W = eds.shape
    aspect = W / H                      # ~2.1 (wide)

    y0, x0 = select_crop(ebsd, ebsd_names)

    # Two separate figures: the elsarticle preprint text block (5.4 x 7.6 in)
    # cannot hold the two full-width maps AND the detail/confusion rows AND a
    # caption on one page, so the full maps go in one float and the detail +
    # confusion panels in a second.
    fig_w = 7.0
    map_h = fig_w / aspect
    fig = plt.figure(figsize=(fig_w, 2 * map_h + 0.7))
    gs = fig.add_gridspec(2, 1, hspace=0.18)

    # interpolation="none" is load-bearing: anything else makes the PDF
    # backend resample the image down to the figure dpi (see fig_segmentation).
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(ebsd_rgb(ebsd, ebsd_colors), interpolation="none",
                aspect="equal", extent=(0, W, H, 0))
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("(a) EBSD-indexed phases", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, fc=ebsd_colors[o][:3])
               for o in sorted(ebsd_names)]
    labels = [ebsd_names[o] for o in sorted(ebsd_names)]
    handles.append(Rectangle((0, 0), 1, 1, fc="black"))
    labels.append("unindexed")
    ax_a.legend(handles, labels, loc="lower left", fontsize=7.5,
                framealpha=0.85, ncol=2, handlelength=1.0, columnspacing=0.9)
    scalebar(ax_a, PIXEL_UM)
    locator(ax_a, y0, x0)

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.imshow(eds_rgb(eds, dom_colors), interpolation="none",
                aspect="equal", extent=(0, W, H, 0))
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title("(b) EDS segmentation (auto-ID names)", fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, fc=dom_colors[cid])
               for cid in sorted(eds_names)]
    dup = len(set(eds_names.values())) < len(eds_names)
    labels = [f"{eds_names[cid]} ({cid})" if dup else eds_names[cid]
              for cid in sorted(eds_names)]
    ax_b.legend(handles, labels, loc="lower left", fontsize=7.5,
                framealpha=0.85, ncol=2, handlelength=1.0, columnspacing=0.9)
    scalebar(ax_b, PIXEL_UM)
    locator(ax_b, y0, x0)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    zoom_h = 1.1
    fig2 = plt.figure(figsize=(fig_w, zoom_h + 2.0 + 0.8))
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[zoom_h, 2.0],
                            hspace=0.28, wspace=0.55)

    sl = np.s_[y0:y0 + CROP_H, x0:x0 + CROP_W]
    ext = (x0, x0 + CROP_W, y0 + CROP_H, y0)
    for col, (img, title) in enumerate(
            ((ebsd_rgb(ebsd[sl], ebsd_colors), "(a) EBSD detail"),
             (eds_rgb(eds[sl], dom_colors), "(b) EDS detail"))):
        ax = fig2.add_subplot(gs2[0, col])
        ax.imshow(img, interpolation="none", aspect="equal", extent=ext)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        scalebar(ax, PIXEL_UM)

    for col, tag in enumerate(TAGS):
        ax = fig2.add_subplot(gs2[1, col])
        res = results[tag]
        confusion_panel(
            ax, res,
            f"({'cd'[col]}) {SHORT[tag]} confusion (row %)")
        print(f"{tag}: accuracy {res.accuracy:.4f}, both {res.n_both:,}")

    fig2.savefig(args.out2, bbox_inches="tight")
    print(f"wrote {args.out2}")


if __name__ == "__main__":
    main()
