# Recommended Default Tunables for `stitch.py`

These defaults are tuned for SEM mosaics with tiles around **1024×768 px** and **5–10% overlap**.

| Parameter | Recommended Default | Rationale                                                                                                                                                                 |
|----------|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`OVERLAP_FRAC`** | `0.08` | Overlap fraction between tiles. Slight overestimation is safer than underestimation, ensures enough area for phase correlation.                                           |
| **`BAND_FRAC`** | `0.05` | Fraction of tile width/height used as strip for cross-correlation. Rule of thumb: `0.5 * OVERLAP_FRAC`, wide enough for texture, narrow enough to ignore central content. |
| **`N_SLICES`** | `9` | Sampling 9 strips gives a good balance between robustness and runtime.                                                                                                    |
| **`UPSAMPLE`** | `15` | Provides ~0.07 px resolution in phase correlation: enough for sub-pixel alignment without overkill.                                                                       |
| **`MAX_STRIP_DRIFT`** | `12` px | Legacy absolute clip; a runtime `drift_gate = max(0.25 × step_nom, 50)` now supersedes it for per-strip phcorr outputs.                                                   |
| **`MAD_Z`** | `2.5` | Robust outlier rejection using median absolute deviation.                                                                                                                 |
| **`REFINE_ITERS`** | `4` | 3–4 refinement passes typically converge; more gives diminishing returns. **Automatically set to 0 when the initial LSQR residual is already tight** (see note below).    |
| **`REFINE_MAX_SHIFT`** | `20` px | Allow up to ~2–3% tile shift correction during refinement: enough for real drift but prevents runaway solutions.                                                          |
| **`PYR_LEVELS`** | `4` | Build multiscale pyramid down to ~64 px for stable refinement.                                                                                                            |
| **`TUKEY_ALPHA`** | `0.08` | Feathering at tile edges to hide seams.                                                                                                                                   |
| **`MAX_VERT_X_DRIFT`** | `5.0` px | Vertical neighbors can shift a few px horizontally; prevents diagonal tearing between rows.                                                                               |
| **`MAX_RIGHT_Y_DRIFT`** | `10.0` px | Horizontal neighbors can shift slightly vertically; prevents staircase distortion.                                                                                        |
| **`SHEAR_CAP`** | `5e-4` | Limits per-tile shear to ~0.5 px across a tile. Enough to correct tiny scan skew without introducing warping.                                                             |
| **`RIGHT_DY_WEIGHT`** | `0.1` | Softly enforces that horizontal neighbors stay level.                                                                                                                     |
| **`VSTEP_ANCHOR_W`** | `0.5` | Weakly anchors vertical step size to nominal value to avoid vertical drift accumulation.                                                                                  |
| **`anchor_sigma`** | `60.0` | Controls how strongly positions are pulled toward stage-derived grid; 60 px is a mild regularization.                                                                     |

---

## Tips for Large Mosaics (≥ 1000 Tiles)

- **Tighten drift tolerances:**
  - `MAX_VERT_X_DRIFT = 3 px`
  - `MAX_RIGHT_Y_DRIFT = 5 px`

- **Stronger stage anchoring:**
  - Decrease `anchor_sigma` to `25–40 px` for fine-grained textures where pair consensus is noisy.

- **Weak-contrast data:**
  - Increase `BAND_FRAC` to `0.06–0.08`.
  - Increase `N_SLICES` to `11–13` to improve robustness.

- **Row-to-row tilt:**
  - Increase `SHEAR_CAP` to `1e-3` to allow tiles to absorb small systematic skew.

---

## Runtime behavior notes

### Tight-start refine skip

If the initial LSQR solve lands at a median pair residual below ~5% of the nominal step (the `tight_start` check), `stitch.py` skips the refine loop entirely. A clean initial solve is easy to damage: overlap-phase-correlation refinement can compound small per-iter corrections through the LSQR solve and move individual tiles by 100+ px, opening visible seams. Logged as:

```
[refine] initial median residual = 1.67 px (tight start)
[refine] tight start - skipping N refine iter(s)
```

A second guard inside the refine loop aborts an iter (reverts positions) if any single tile moved more than 30 px when starting from a tight solve.

### Monotonicity check

After both the initial LSQR and the post-refine LSQR, positions are checked for within-row x-monotonicity and within-column y-monotonicity. Violations trigger targeted pair suppression (the highest-residual incident pair is zero-weighted) followed by one re-solve.

### BCF tile cache

On first run, `stitch.py` writes an NPZ of each BCF's BSE data, stage coords, and pixel size into `<--dir>/stitch_cache/`. The cache key is the BCF's size + mtime (path excluded, so the cache is portable across workstations that mount the same drive under different letters). Subsequent runs of any mode (full stitch, `--diagnose`, `--preflight`) skip the `hs.load` step and can be ~3–10× faster on the load phase. Disable with `--no-cache`.

### EDS mosaic cache

When `--eds-preview` is given, `stitch.py` also builds an **EDS mosaic cache** at `<--dir>/eds_cache/`. This stores per-element, per-band (main/sb_lo/sb_hi) raw weighted sums plus a shared weight mosaic as float32 `.npy` memmaps covering the full BSE-resolution mosaic. A `manifest.json` records the stitch fingerprint (SHA-256 of `stitch_manifest.csv`), element list, window width, shift method, and downsample factor.

Cache is built first, then **preview PNGs are derived from the cache** — no second BCF pass. The preview step becomes a chunked memmap-read + block-average + percentile-clip + PNG write, taking seconds instead of a full tile-stream pipeline. Only when `--no-eds-cache` is set does the standalone `build_eds_previews_fast` pipeline run.

The viewer reads ROIs by slicing the memmaps instead of re-opening every intersecting BCF — ROI extraction drops from minutes to milliseconds once the cache is warm. The cache is automatically used when its fingerprint matches the current stitch, current element list, and current `width_eV`; otherwise the viewer falls back to the per-tile read.

CLI flags:

- `--eds-cache` / `--no-eds-cache` — default **on** whenever `--eds-preview` is given.
- `--eds-cache-ds N` — downsample factor (default 1 = full-res). 2/4/8 trade resolution for disk.
- `--eds-cache-on-restitch {prompt,rebuild,keep,discard}` — what to do with an existing cache when re-stitching produces a new manifest. `prompt` (default, interactive) asks; the others are for scripted runs.
- `--eds-shift-method {bilinear,fft}` — subpixel interpolation for aligning EDS tiles to the BSE grid. Default **bilinear** (~8× faster on this step, ~1.6× end-to-end; visually indistinguishable for smooth EDS counts). Use `fft` for strict sinc interpolation matching the phase-correlation alignment model — slower but exact for bandlimited signals.
- `--eds-cache-workers N` — parallel tile workers during cache build (default **4**). Each worker holds one ~0.8–1.5 GB EDS subcube in RAM while processing its tile, so peak subcube RAM ≈ `N × 1.5 GB`. Lower to reduce RAM pressure on workstations.
- `--eds-cache-flush-every N` — flush cache memmaps to disk every N tiles (default **50**). Periodic flushes turn dirty memmap pages clean so Windows can evict them under memory pressure — prevents a long cache build from paging other processes to the swap/scratch disk and freezing the workstation. Set very high (e.g. 999999) to disable periodic flush if you prefer one big flush at the end.
- `--eds-cache-dir PATH` — store the cache at a custom path instead of `<--dir>/eds_cache/`. Useful when tile BCFs live on an HDD but an SSD is available for the cache. A sidecar file `<--dir>/eds_cache_link.txt` is written with the absolute path; the viewer and later stitch runs follow it automatically, so you only need to pass this flag on the first build. Delete the sidecar (or call `eds_cache.clear_cache_link(base_dir)`) to revert to the default location.

**Memory-constrained builds**: if a cache build is pushing the workstation into swap, drop to `--eds-cache-workers 2 --eds-cache-flush-every 20`. The cache will take longer but RAM peak stays manageable (~3 GB subcubes + periodic flushes ~ every 40 s).

**HDD-stored caches**: ROI reads at full-res are very slow on mechanical disks (observed ~4–10 MB/s on a 168 GB cache). The viewer uses a serial single-worker read strategy by default (parallel reads thrash a single-head HDD). For usable performance, either (a) use `--eds-cache-dir` to put the cache on an SSD, or (b) rebuild at `--eds-cache-ds 4`/`8` so each ROI reads 16–64× less data. A `defragment` utility is provided (`python eds_cache.py defragment <cache_dir>`) but only helps when fragmentation — not raw disk speed — is the bottleneck.

**Disk cost** (≈ H × W × 4 bytes × (3 × n_elements + 1), where H × W is the BSE mosaic):

| ds | 2014-tile mosaic (≈500 Mpx), 10 elements |
|----|------------------------------------------|
| 1  | ~60 GB                                   |
| 2  | ~15 GB                                   |
| 4  | ~4 GB                                    |
| 8  | ~1 GB                                    |

### Prep variant auto-calibration

Phase-correlation prep filter (`edge`/`wide_hp`/`flat`/`dog`) is auto-selected by default via a 40-pair calibration sample scored by `median_weight × exp(-orth_std/10)`. Force a specific variant with `--prep-variant <name>`.

---

## Example Usage

Run with default parameters:

```bash
python stitch.py --dir "Mapping_job_2025_02_05"
```

Diagnose mode — runs the neighbor graph + initial LSQR + monotonicity check and writes `stitch_debug/stitch_diagnose_report.txt` without running refine or composite. Fast iteration for validating parameters:

```bash
python stitch.py --dir "Mapping_job_2025_02_05" --diagnose
```

Preflight parameter auto-tuning:

```bash
python stitch.py --dir "Mapping_job_2025_02_05" --preflight --auto
```

When `--preflight` is used alone, a text file of suggested tunables is written to the input directory. Combined with `--auto`, the suggested values are applied and a full stitch is run.

Force a specific prep variant and drop low-confidence pairs:

```bash
python stitch.py --dir "Mapping_job_2025_02_05" \
    --prep-variant edge \
    --weight-floor 0.05
```

Disable the BCF cache (forces fresh `hs.load` of every tile):

```bash
python stitch.py --dir "Mapping_job_2025_02_05" --no-cache
```

Build the full-resolution EDS mosaic cache alongside preview PNGs:

```bash
python stitch.py --dir "Mapping_job_2025_02_05" \
    --eds-preview all --eds-preview-ds 20 \
    --eds-cache-ds 1
```

Disk-constrained? Use a downsampled cache, or turn it off and keep only previews:

```bash
python stitch.py --dir "Mapping_job_2025_02_05" --eds-preview all --eds-cache-ds 4
python stitch.py --dir "Mapping_job_2025_02_05" --eds-preview all --no-eds-cache
```

Scripted re-stitch — don't prompt, just rebuild the cache:

```bash
python stitch.py --dir "Mapping_job_2025_02_05" --eds-preview all \
    --eds-cache-on-restitch rebuild
```

Use strict FFT/sinc subpixel shift for the cache build (slower, bandlimited-exact):

```bash
python stitch.py --dir "Mapping_job_2025_02_05" --eds-preview all \
    --eds-shift-method fft
```

Relocate the cache to an SSD while source tiles stay on an HDD:

```bash
python stitch.py --dir "D:\UU\MyRun" --eds-preview all \
    --eds-cache-ds 2 --eds-cache-dir "E:\thermod_caches\MyRun"
```

Data already on an SSD, full-res cache in a dedicated SSD folder (typical modern setup):

```bash
python stitch.py --dir "E:\path\to\ssd\data_dir" --eds-preview all \
    --eds-preview-ds 10 --eds-cache-dir "E:\path\to\ssd\cache_folder"
```

Sidecar file `<--dir>/eds_cache_link.txt` will contain the cache path; subsequent runs pick it up automatically without needing `--eds-cache-dir` again. Inspect or manage an existing cache:

```bash
python eds_cache.py status "E:\path\to\ssd\cache_folder"
python eds_cache.py defragment "E:\path\to\ssd\cache_folder"
```

---

## Outputs

Inside the input directory:

- `stitched_BSE_16bit.tiff` — full-resolution mosaic.
- `stitched_BSE_8bit.png` — display copy.
- `stitch_manifest.csv` — per-tile grid position, pixel position, shear, and EDS energy axis parameters (consumed by `viewer.py` and `eds_map.py`).
- `stitch_cache/` — NPZ-cached BCF loads (size+mtime keyed; portable across workstations).
- `eds_cache/` — float32 `.npy` memmaps per element × band + shared weight; plus `manifest.json` fingerprinted against `stitch_manifest.csv`. Read by `viewer.py` for instant ROI extraction. When `--eds-cache-dir` was used, this folder is elsewhere and `eds_cache_link.txt` points at it. `manifest.json` carries `data_type` (`"bcf_net_counts"` from `stitch.py`, `"oxford_wt_pct"` from `oxford_loader.py`) and a `source` provenance string.
- `eds_cache_link.txt` (optional) — one-line absolute path to the actual cache directory when relocated via `--eds-cache-dir`.
- `stitch_debug/` — neighbor graph CSVs, debug plots, worst-overlap crops, and (in `--diagnose`) `stitch_diagnose_report.txt`.

---

## Oxford AZtec CSV datasets

`viewer.py` reads two data sources:

1. **Bruker BCF/SPX tiles** — the original path. Stitched by `stitch.py`, cache built during the EDS preview step.
2. **Oxford Instruments AZtec CSV exports** — already-montaged per-element Wt% CSVs, optionally with a BSE TIF. Imported via a one-shot converter that produces the same on-disk layout the viewer already knows how to consume:

```bash
python oxford_loader.py --dir D:\UU\Kibo          # with BSE TIF
python oxford_loader.py --dir D:\UU\Granulite     # no BSE → mean-Z proxy
python viewer.py        --dir D:\UU\Kibo
```

The Oxford loader auto-detects two element-CSV filename conventions: `Montaged Map Data-{Element} Wt%.csv` (Kibo-style) and `{El}_{digits}.csv` (Granulite-style). Override either with `--csv-glob` if your dataset uses a different pattern. The BSE TIF (`*BSE*.tif`, override with `--bse-glob`) is optional: when missing, the loader synthesises a mean-Z proxy from the element maps (`Σ(Z·wt%) / Σ(wt%)` per pixel) — `--resolution=bse` is unavailable in that mode and silently falls back to `elements`. The loader writes `stitched_BSE_8bit.png` / `stitched_BSE_16bit.tiff` / `stitch_manifest.csv` / `eds_cache/` into the same directory.

Key flags:

- `--resolution {elements,bse}` — target mosaic grid. `elements` (default) downsamples BSE to the element-map size (small cache, fast). `bse` upsamples the element maps to BSE resolution (sharper BSE detail, bigger cache).
- `--force` — rebuild the cache even if the fingerprint matches.
- `--pixel-um` — record pixel size (microns) in the cache manifest for future use.
- `--out <dir>` — write outputs to a different directory than `--dir`.

Inspect what the loader produced:

```bash
python eds_cache.py status D:\UU\Kibo\eds_cache
# Cache: D:\UU\Kibo\eds_cache
#   data_type    : oxford_wt_pct
#   source       : Oxford AZtec CSV export
#   elements     : ['Ag', 'Al', 'Au', ...]
#   shape (H,W)  : (2857, 2835)
#   ...
```

Notes:

- Oxford data is already quantified — the cache stores Wt% directly. Sidebands are zero-filled and weight is ones. The viewer's existing `net = main − 0.5·(sb_lo + sb_hi)` formula reduces to `net = main = Wt%`; composition fractions and modelling sweep both produce correct values because the downstream `Composition.from_element_wt` renormalises internally.
- The SPX-quantification radio in the composition-vs-scale dialog is auto-disabled for Oxford caches (there are no per-tile SPX files).
- LZW-compressed Oxford TIFFs are handled via a PIL fallback — no need to install `imagecodecs`.
- When no BSE TIF is shipped, the synthesised mean-Z proxy uses an atomic-number table covering H–U. Elements with no Z entry contribute nothing; background pixels (Σwt=0) read as 0. The proxy is a visual analogue of true BSE contrast (which tracks mean Z), not a calibrated image.
- Re-running `oxford_loader.py` on an already-converted directory is idempotent: the cache rebuild is skipped when the fingerprint matches (use `--force` to override).

---

## Viewer workflows: composition + modelling

### Composition vs length scale

`Composition vs scale …` (panel button) tiles the active ROI into a quadtree
and reports a composition per window — useful for asking *at what spatial
scale is the rock equilibrated?*.

- **Settings dialog** auto-derives the maximum useful quadtree depth from
  the ROI size. The EDS source uses a 4 px floor; the SPX source uses the
  **median manifest tile short-side** as its floor (below that, no window
  could contain a tile at 95% overlap, so SPX would return nothing). The
  source toggle re-clamps the depth spinner.
- **Per-level table** shows actual window count + pixel size per level and
  exposes a `Max windows` cap per row. Sampling strategy: `First N` (scan
  order) or `Random N` (fixed seed `np.random.default_rng(0)` for
  reproducibility).
- **Result dialog** tabs: Distributions (box plots per top-element across
  levels), Per-scale summary (mean ± σ table), Windows (full per-window
  table). Buttons: `Run modelling sweep …`, `Export CSV …`, `Locate
  windows` (overlays a colour-coded Shapes layer on napari, one colour
  per level).

### Modelling sensitivity sweep

From the composition-vs-scale result dialog, `Run modelling sweep …`
opens the usual thermodynamic options dialog (seeded with the full-ROI
mean composition) and dispatches a sequential batch of P-T grid runs:

```
for each usable quadtree level L:
    for rep in {mean, mean+σ, mean−σ}:
        # negatives clipped to 0, renormalised to 100 wt%
        record = run_grid(comp, P-range, T-range, n_p, n_t)
```

A 4-level default yields up to 12 grids per ROI. Each becomes a record in
the **Modelling runs** history tagged `sweep L{n} {rep} (eds|spx)`.

### Modelling runs history + compare

Every thermodynamic run (single, grid, or sweep entry) is stored on the
panel as a serialisable record and persists in the session JSON. The
**Modelling runs …** button opens a list dialog supporting multi-select
(Ctrl/Shift):

- **View results** — re-opens the original result dialog without
  recomputing.
- **View parameters** — read-only Parameters viewer (backend, P/T grid,
  Fe³⁺/H₂O/CO₂, composition vector, ROI bbox, manifest).
- **Locate ROI** — draws a magenta polygon on the BSE and centres the
  camera.
- **Compare selected …** — needs 2+ grid runs that share the same P-T
  grid. Three tabs: Summary table (per-run unique-assemblage count + mean
  of every shared bulk property), **Phase-field agreement** map (per-cell
  fraction of selected runs whose stable assemblage matches the first
  run — `RdYlGn` heatmap), **Property spread** (per-cell σ of any shared
  bulk property; combo selector). CSV export of the per-run summary.
- **Export bundle …** — folder containing `parameters.json`,
  `summary.txt`, `phases.csv` / `phase_compositions.csv` /
  `bulk_properties.csv` (single) or `grid.csv` + `pseudosection.png`
  (grid).
- **Delete** — removes the run from history (the result dialog/bundle
  is gone too).

Both single and grid result dialogs themselves also expose a Parameters
tab, an Export bundle button, and a Locate-ROI button — so any of these
actions is reachable without first going through the list.
