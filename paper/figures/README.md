# Manuscript figures

**Pixel size (2026-11-06).** The C23-S1-A mosaic pitch is **0.5475 µm/px**, measured from the stage (tiles step 512 µm across / 384 µm down within a mapping job; median of step / stitched offset over all 1301 adjacent intra-job pairs; BCF header DX 0.549, DY 0.554). The figures and text used a nominal 0.5 µm/px until then, which made every physical length 9.5 % too small (section 24.8 × 13.4 → 27.2 × 14.7 mm; quadtree windows 18.2 … 0.071 mm → 20.0 … 0.078 mm; phase-map pitch 2.5 → 2.7375 µm) and could not give the tiles their ~9 % overlap. One constant, `_figstyle.MOSAIC_PIXEL_UM`, now feeds `scale_tolerance.py`, `directional_fit.py` and `banding_orientation.py` (they had derived the pitch from 24.8 mm); `phase_domain_sizes.py` and `nodule_fraction.py` default to the measured pitch (nodule share 0.35 → 0.37 at the same 12.5 µm / 0.5 mm thresholds; Fig 6b minimum 0.076 → 0.073, regions unchanged). Older notes below quote lengths at 0.5 µm/px; multiply by 1.095.

Reproducible figure-generation for the EquiMap manuscript. Each quantitative
figure is a small, self-contained script that turns a CSV exported by the
EquiMap viewer into a publication-ready PDF, so every figure can be regenerated
from data with one command. Raster figures (mosaics, phase maps) are captured
from the viewer and dropped in here as image files.

Run scripts with the project Python environment, e.g.

```
python fig_calibration.py
```

(On this workstation: `C:\Users\4029771\miniforge3\envs\equimap\python.exe`.)
Each script writes its `*.pdf` next to itself and is included from
`../manuscript.tex` via `\includegraphics{figures/<name>}`.

Every script has a **synthetic-data fallback**: if its input CSV is absent it
generates a plausible demonstration dataset and watermarks the figure
`SYNTHETIC DEMO`, so the layout can be reviewed before real data exist. Replace
with real data by exporting the CSV from the viewer and re-running.

## Figure inventory

| Fig | Script / source | Input | Data-ready? |
|-----|-----------------|-------|-------------|
| 1 Pipeline diagram | `../fig_pipeline.tex` (TikZ) | — | **done** |
| 2 Mosaic cache (engineering headline) | `fig_mosaic.py` | BSE image + ROI element-map (synthetic fallback) | **done on real data** |
| 3 Scale analysis | `fig_scale.py` | `composition_scale.csv` (+ `pt_scale.csv` for panel c — re-scored 2026-07-23 with the unmix-corrected modes from the session's sweep grids; **re-written 2026-09-16 by `spx_sweep_score.py --which eds` with 8 extra columns**: `T/P_lo_mean_*`, `T/P_hi_mean_*` = the mean variant's OWN accepted region, `T/P_best_plus/minus_*` = the ±σ best cells; the original 7 columns are unchanged) + **`pt_scale_spx.csv`** (tier-3 cross-check: the same windows L0–L5 on the SPX source, `spx_sweep_run.py` → `spx_windows.csv` + `spx_sweep_comps.json` + `spx_sweep_grids.json` (18 MAGEMin grids, portable dicts) → `spx_sweep_score.py --which spx`; drawn as the red dashed/square series in panel c). Finding: the calibrated-map collapse below 2.3 mm is a calibration artefact (Al between-window SD 1.5–1.6× the SPX one; the Al-poor ±σ variant loses sillimanite entirely); on SPX the union stays 537–721 °C from 4.6 mm down to 0.57 mm | **done on real data** |
| 4 Segmentation (map + nodule zoom) | `fig_segmentation.py` (+ `fig_segmentation_inset.py`) | `phase_map_combined.png` + `phase_map_labels.npz` + `phase_legend.csv` + `segmentation_params.json`; **six domains — the quartz–mica matrix domain (cid 1) is kept separate and NNLS-unmixed (the `unmix` block in `segmentation_params.json`), not merged into quartz**; the inset prep re-segments the nodule window headless at full resolution with `--sigma 1` (window auto-picked as the densest epidote region; local clusters majority-mapped onto the global phase names) and writes `phase_map_inset.{npz,json}` | **done on real data** (no synthetic fallback — born with data) |
| 5 P-T inversion | `fig_pt.py` | `pt_grid.csv` (grid export; **re-exported losslessly 2026-09-15** from session run #3 after the duplicate-phase-instance fix in `ThermoResult.to_dict`: MAGEMin lists a solvus pair as two `fsp` instances, which the old export collapsed onto one column, dropping 5–25 vol% feldspar in 70 K-feldspar-bearing cells; the second instance is now the `*_fsp_2` column and `build_results` maps it back to `fsp`) + `monte_carlo.csv` + `phase_{legend,minerals}.csv` + the `unmix` block of `segmentation_params.json` (applied to the modal budget before scoring) | **done on real data** (no synthetic fallback) |
| 6 P-T reconstruction sensitivity | `fig_reconstruction.py` | same inputs as `fig_pt.py` (incl. unmix); re-scores the SAME grid under three treatments of the pseudomorph nodules (as observed / nodule Ms+Ep → Kfs with matrix Ms kept, `NODULE_MS_FRACTION=0.5` illustrative / full conversion). NB the pre-2026-09-15 panel (c) "escape to 800 °C" was an artefact of the lossy export; lossless: 668 °C / 0.39 GPa, misfit 0.11 | **done on real data** (regenerated 2026-09-15) |
| (Table 1, `tab:pt`) calibrated-bulk Monte Carlo | `mc_calibrated.py` (needs MAGEMin + the July session file `D:/Actief/C23S1A_Quant/viewer_session_22_07_26.json`; grids are served from the result cache on re-runs) | `monte_carlo_calibrated.csv` (same layout as `monte_carlo.csv`, loadable by `fig_pt.load_mc`), `monte_carlo_calibrated_long.csv`, `mc_calibrated_samples.csv`, `mc_calibrated_summary.json`: sweep-L0 calibrated bulk perturbed by the per-element hold-out bias of `calibration_validation.csv` (16 draws, seed 0, Fig-5 grid), scored as `fig_pt.py` | **done on real data** (2026-09-15) |
| 7 Calibration validation | `fig_calibration.py` | `calibration_validation.csv` | **done on real data** (2026-07-22 post-fix export) |
| 8 EBSD-vs-EDS validation | `fig_ebsd.py` (+ `ebsd_confusion_prep.py`) | `ebsd_val_serpdi_{1,2}_EBSD.npz` + `serpdi_{1,2}_EBSD_seg_meta.json` (+ `*_confusion.csv` data record) | **done on real data** (2026-07-24; no synthetic fallback; 2026-10-09: re-run from scratch, label maps byte-identical to the 29 Jul staging; names staged from the full mask (`--erode-px 0`), with the interior-only erosion scan 0-8 px and top-5 candidates recorded in `*_seg_meta.json`; the quantified QuantMap contrast quoted in section 5.2 is in `ebsd_quantified/`) |

## Headless re-run of the worked example (2026-09-16, line-aware background)

The net-count background of the toolkit was corrected on 2026-09-16
(`eds_background.py`: the legacy sideband mean subtracted the neighbouring
Kα peak for Na, Mg, Al, Si and P, and per-pixel clipping hid it; see
`CLAUDE.md`). Every staged worked-example file was therefore regenerated
from the July session's band maps (`viewer_session_22_07_26.npz`, 1/5 grid)
by headless drivers that replay the viewer's own code paths, in this order:

| Step | Script | Writes | Notes |
|------|--------|--------|-------|
| 1 calibration | `calibrate_mosaic.py` | `calibration_validation.csv`, `eds_calibration.json` | all 2014 SPX tiles via `CacheWindowReader`, 50 × 20 % hold-out; Fig 7 input |
| 2 segmentation + classification | `segment_worked_example.py` | `phase_map_labels.npz`, `phase_map_combined.png`, `phase_legend.csv`, `phase_minerals.csv`, `phase_composition.csv`, `segmentation_params.json`, `segmentation_report.json` | recorded recipe on the signed line-aware nets (smoothed, clipped, plus the per-element noise floor of `_smooth_element_maps`; clipping alone left exact zeros that split clean quartz) with `--min-region 30` (the signed nets keep the noise the old per-pixel clip rectified, so at the July cleanup of 15 px the map fragmented; 30 px restores July's region statistics), `--fit-balance density` (the GMM fit sample drawn evenly across feature space, `viewer_segment_helpers._density_balanced_sample`: a uniform pixel sample never gave the 1.5 % sillimanite cores a component at any k, pitch or PCA setting because their edge pixels grade into muscovite; the balanced sample lets the ICL criterion allot one, k = 14), `--n-clusters auto` and `--merge-same-name` (duplicated components merged by their identification, mixed components typed by their dominant unmix endmember first, as the dialog's Merge selected does) → eight domains incl. sillimanite 1.1 % and chlorite 0.7 %, 87.9 % pixel agreement with July; names inherit the July decisions by pixel majority only where the auto-ID is weak (< 75 %) (mixed domains unmixed onto clean ones, disagreeing auto-IDs become recorded manual overrides); `--n-clusters` / `--pca-variance` for diagnostics; the contingency with the previous map is in the report |
| 2b intermediate-pitch segmentation | `segment_fullres.py --ds 2` | `<dataset>/export/phase_map_ds2.npy` (int8, 1 µm pitch, 333 M px, OUTSIDE the repo) + the same staged files as step 2 (`phase_map_labels.npz` = that map sampled onto the 1/5 grid; legend pixel counts are at 1 µm) | the whole section at 1 µm/px (every second native pixel) with the same feature pipeline, fitted on a random block sample (600 × 64² px, seed 42) and predicted tile by tile (2048 px + 48 px margin) from the mosaic cache; sigma 2 map px, min region 190 map px (= 30 px at 1/5), k by ICL up to 15 so the sillimanite cores can take a component. The Fig 4b inset stays a native-pitch re-segmentation of the nodule window (`fig_segmentation_inset.py`). `--ds 1` is the native-pitch variant (1.33 G px, ~70 min) |
| 3 per-phase formulae | `phase_stoichiometry.py` | `phase_stoichiometry.csv` | Section 5.1 table (erosion 0/3/6 px) |
| 4 windows | `eds_windows_run.py` | `composition_scale.csv` | the July quadtree windows (session `comp_scale.records`) re-evaluated; Fig 3a/b |
| 5 calibrated sweep | `eds_sweep_run.py` → `spx_sweep_score.py --which eds` | `eds_sweep_comps.json`, `eds_sweep_grids.json.gz`, `pt_scale.csv` | L0–L8 PC1 variants (`--min-level 6 --max-level 8 --append` for the sub-tile levels), 27 MAGEMin grids; Fig 3c filled bands: the envelope is constant from 18 mm to 0.3 mm and breaks down at 0.14 and 0.07 mm |
| 6 SPX sweep re-score | `spx_sweep_score.py --which spx` | `pt_scale_spx.csv` | grids unchanged (`spx_sweep_grids.json.gz`), re-scored with the new modes |
| 6b per-window inversions | `window_pt_run.py` | `pt_windows.csv`, `pt_windows_grids.json.gz` | Fig 3c: every window at L1–L2 and 12 random windows per level L3–L8 (seed 0) inverted with their OWN calibrated bulk and OWN phase-map modes (unmixed), scored with the section's weights and the sillimanite presence; per-window best cell, accepted region and its overlap with the section's region; 93 MAGEMin grids through the worker pool (`--workers 4`, result-cache aware). `fig_scale.py` draws ONE curve when `pt_windows.csv` exists: the median (with interquartile band) fraction of a window's accepted region that lies inside the section's, versus window size. The overlap declines slowly, within the window-to-window scatter, from the section size to ~0.4 mm (no neighbouring levels differ significantly) and then drops (pooled 3.3–0.42 mm median 0.76 vs 0.2–0.1 mm median 0.52, p = 0.001; 3 of the 24 windows at 0.10 and 0.05 mm lack quartz or feldspar). The shaded interval marks the ONSET of that departure by a persistence rule with no fixed overlap threshold (`_plateau_onset`: the coarsest level from which every finer level's median lies below the lower quartile of the pooled coarser levels; levels with < 8 windows skipped as reference) → between the 0.42 and 0.21 mm windows, i.e. windows comparable to the phase domains (area-weighted median equivalent diameters: plagioclase 150, quartz 290, muscovite 660, biotite 850 µm on the 2.5 µm grid). The earlier 0.5-crossing (0.09 mm) was an arbitrary threshold and is gone. The semivariogram range a ≈ 2.5 mm of panel (b) leaves no mark on (c)/(d): the bulk stops being representative there, but each window's bulk and modes co-vary, so the inversion tolerates the drift; a is therefore marked in (b) only and compared in the caption/text. With per-window data the figure is 2×2: panel (d) shows the 10–90 % spread of the windows' best-fit T (blue, left) and P (red dashed, right) in physical units, the two axes scaled to the same fraction of the grid extent so equal heights are equal fractions of the grid (the lines coincide: the windows drift along the oblique misfit valley); length axes in mm; else the 1×3 layout with the ±σ union bands in (c) |
| 7 Monte Carlo | `mc_spx_rescore.py`, `mc_calibrated.py` | `monte_carlo.csv`, `monte_carlo_summary.json`, `monte_carlo_calibrated*.csv`, `mc_calibrated_summary.json` | SPX draws (10 %, n 16, seed 0) re-scored; calibrated draws on the new L0 bulk and hold-out bias |
| 8 figures | `fig_calibration.py`, `fig_mosaic.py`, `fig_segmentation_inset.py --window 1290 39530`, `fig_segmentation.py --pixel-um 2.7375`, `fig_pt.py`, `fig_reconstruction.py`, `fig_scale.py --pixel-um 0.5475` | the PDFs | `pt_grid.csv` (SPX bulk) is unchanged. `fig_segmentation.py --palette hybrid` (default; manuscript tab20 hues per mineral, matrix domains in greys; `manuscript` and `alt` are the alternatives) colours domains by NAME so old and new maps share colours, and takes `--data-dir` to render another run's exports (used for the old-vs-new comparison). Sillimanite is a global domain (balanced fit) and the native-pitch inset re-segmentation labels its core cluster by the global name. `split_domain.py` (second-stage GMM of one or more domains) and `segment_fullres.py --ds 2` (streamed 1 µm segmentation) remain as diagnostics: neither isolated the cores, which is how the fit-sample cause was found |

Grid dumps are kept gzipped (`*_grids.json.gz`; the scorer reads either
form, the newer file when both exist) so the figures regenerate without Julia. The pre-fix staged files are
backed up outside git in `_scratch/pre_background_fix_staged/` and in the
`pre-background-fix` tag.

**Provenance of the 2026-07-23 staged segmentation files (superseded by the 2026-09-16 re-run above; kept for the record).** The
`phase_map_labels.npz` / `phase_map_combined.png` / `phase_legend.csv` /
`phase_composition.csv` / `phase_minerals.csv` / `segmentation_params.json`
currently staged here come from an exact headless replication of the export
recipe (seeded clustering; verified 0.000% mismatch against the GUI export)
with the two quartz-dominated domains kept separate and the matrix domain
unmixed the way the viewer's **Unmix selected …** button computes it (raw
net-count sums, full masks, all 11 cache elements). Re-doing the
classification in the GUI (re-run segmentation with the recorded recipe, do
NOT merge the quartz domains, apply Unmix to the matrix domain, re-export)
reproduces these files — but note the recipe predates the "Leave zero-signal
(no-data) pixels unclassified" option (2026-07-24, now checked by default):
uncheck it for a byte-identical reproduction (this ROI has 82k zero-signal
px, 0.15%, that the recorded recipe NN-filled).

**Provenance of the EBSD validation files (Figure 8, 2026-07-24).** The staged
`ebsd_val_serpdi_{1,2}_EBSD.npz` / `*_seg_meta.json` / `*_confusion.csv` come
from `ebsd_confusion_prep.py` run on the two h5oina-converted serpentinite
datasets at `D:\data\EBSD\serpdi_{1,2}_EBSD` — the GUI path with
classification-dialog defaults (GMM auto-k, sigma=3, ratio-only mineral ID;
zero-signal off-sample pixels left unclassified per the `protect_background`
dialog default — 26.5% / 24.0% of these maps; never "Use EBSD phases", which
would be circular). `fig_ebsd.py` recomputes
the confusion statistics from the staged arrays with the toolkit's own
`ebsd_validation.confusion_matrix`, so figure and text share one code path.
Headline numbers: majority purity 82.4% / 83.6%, alias-matched auto-ID
accuracy 0.293 / 0.241. GUI reproduction: whole-map ROI → Compute → Segment
phases (defaults) → Identify minerals → "EBSD vs EDS confusion …" → export CSV.

`fig_segmentation.py` takes `--pixel-um` = the **phase-map** pixel pitch
(mosaic pitch × the segmentation's `seg_ds`; worked example: 0.5475 × 5 =
`--pixel-um 2.7375`). `fig_pt.py` recomputes the misfit with the toolkit's own
scorer (imports `viewer_thermo_helpers` from the repo root; `--repo` to
override), with per-domain weights from `phase_minerals.csv` and sillimanite
as a presence-only constraint — edit `PRESENCE` in the script if the observed
assemblage changes.

## Producing `calibration_validation.csv` (Figure 3)

This figure is **not blocked** on the new dataset — it runs on existing Bruker
data with SPX (e.g. `I:\UU\C23S1A_Quant\`):

1. Open the viewer on the Bruker dataset; draw an ROI that contains several
   whole SPX tiles.
2. Click **Calibrate EDS (SPX k-factors)**.
3. In the dialog, click **Export validation CSV …** and save it here as
   `calibration_validation.csv`.
4. Re-run `python fig_calibration.py`. The `SYNTHETIC DEMO` watermark disappears
   and the real result is plotted.
5. Read the printed overall RMSE and per-element values into the §5 text
   (the `\TODO{X}` / `\TODO{Y}` placeholders in `manuscript.tex`).

CSV columns expected: `tile, element, predicted_wt_pct, measured_wt_pct,
residual_wt_pct` (the long-format export written by the Calibrate-EDS dialog).

## Producing the mosaic figure (Figure 2)

This is the "engineering headline" figure (the spectrally-aware EDS mosaic
cache). It is **not blocked** on the new dataset — it runs on any existing
stitched mosaic. Panel (a) shows the **real overlapping tiles** that cover the
ROI (read from `stitch_manifest.csv`) and panel (c) is sliced straight from the
EDS cache by the same `eds_cache.roi_from_cache` the paper describes.

**One-command route (recommended).** Point the script at the stitched dataset
directory (the Bruker dataset you exported the calibration CSV from) and give an
ROI box in BSE pixel coordinates — the viewer prints it as
`ROI #N [y=y0:y1, x=x0:x1]` when you draw one, so `--roi` is `x0 y0 x1 y1`:

```
python fig_mosaic.py --dataset "path\to\stitched_dir" \
    --roi X0 Y0 X1 Y1 --element Fe
```

**The published Figure 2 was generated with** (record kept so the exact ROI
never has to be recovered from the PDF again):

```
python fig_mosaic.py --dataset "D:\Actief\C23S1A_Quant" \
    --roi 34281 23001 36783 25503 --element Al --pixel-um 0.5475 --roi-ds 2
```

It reads the real tile boxes from `<dataset>/stitch_manifest.csv` for panel (a),
loads `<dataset>/stitched_BSE_8bit.png` for panels (a)/(b), and slices the Fe
net-intensity map for that ROI from `<dataset>/eds_cache/` for panel (c). Useful
flags: **`--pixel-um 0.5475`** to add a scalebar to each panel (the C23S1A maps are
0.5475 µm/px, measured; see the pixel-size note at the top of this file);
`--roi-ds 4` to decimate a very large ROI read; `--element Mg` (etc.) for a
different map; `--repo <root>` if `eds_cache` is not importable (it defaults to
the repo root two levels above `figures/`, correct when the manuscript lives
inside the repo).

ROI guidance: pick a region that shows several mineral phases (so panel (c) has
contrast in the chosen element). A **smaller ROI (≈1500–3000 px)** gives the
cleanest panel (a) — a handful of large tiles whose ~9% overlap is obvious; a
large ROI still works but packs in many tiles (a 5000×5000 ROI is covered by
≈56). Valid coordinates are bounded by the mosaic size (the script clips and
reports if you overshoot). First run pays the full-mosaic BSE decode (~30 s).

**Manual route.** If you would rather supply the two rasters yourself: copy
`stitched_BSE_8bit.png` here, export one element's ROI map as a `.npy`/`.tif`
(e.g. **Zoning / traverse …** → pick an element → *Show map in napari* →
*File → Save Selected Layers …*), then:

```
python fig_mosaic.py --bse stitched_BSE_8bit.png \
    --roi X0 Y0 X1 Y1 --element-map fe_roi.npy --element Fe
```

The `SYNTHETIC DEMO` watermark disappears once a real source is supplied. On the
manual route (no `--dataset`) panel (a) falls back to a schematic tile grid over
the ROI, since the stitch manifest is not available.

**Fig 5 additions (2026-10-02, reviewer comment on the inversion).** `fig_pt.py` panel (a) now also draws the field in which every observed phase (proportion keys + presence-only keys) is predicted stable (solid outline) and the melt-in line (dotted; also on panel b), with a small legend, and prints the field's extent, the misfit range inside and outside it, and how many accepted or stable cells contain melt. On the staged grid: 15 stable cells (589-721 C, 0.39-0.76 GPa), misfit 0.073-0.106 inside against >= 0.192 outside, 9 accepted cells all inside the field and melt-free, melt from 695 C (3 of the 15 cells).

**Window-sampling fix (2026-10-02).** The session's two capped levels (L7: 8192 of 16384 windows, L8: 9192 of 65536) came from the viewer's capped random sampler, which truncated the sorted unique draws and so kept only the top of the section (79 % and 57 % of its height). `viewer_thermo_helpers._quadtree_sampled_windows` now shuffles before truncating (regression test in `tests/test_composition_scale.py`), and `eds_windows_run.py` re-draws capped levels uniformly by default (`--keep-session-windows` restores the old list; `--seed`). Re-run after the fix: `eds_windows_run.py` -> `composition_scale.csv`; `window_pt_run.py` (L1-L6 from the grid cache, 24 new grids at L7-L8) -> `pt_windows.csv`; `eds_sweep_run.py --min-level 7 --max-level 8 --append` + `spx_sweep_score.py --which eds` -> `pt_scale.csv`; `fig_scale.py`. Changes: correlation length 2.5 -> 2.7 mm; per-level mean composition flat (Al within 0.15 wt%, Si +0.7 wt% at L8); median overlap at 0.10 / 0.05 mm 0.55 / 0.20 -> 0.49 / 0.58 (no collapse at the finest size); best-fit spread levels off below 0.42 mm (T 129 / 126 / 129 / 103 C, P 0.35 / 0.29 / 0.43 / 0.43 GPa); onset interval unchanged (0.42-0.21 mm). The pre-fix files are kept in `_scratch/background_fix/*_biased_sampler.csv`.

**36 windows per size (2026-10-05).** `window_pt_run.py --n-per-level 36 --workers 4` -> 236 inversions (4 + 16 + 6 x 36). Median overlap 1.00 / 0.85 / 0.76 / 0.76 / 0.72 / 0.61 / 0.54 / 0.18 from 6.7 mm to 52 um; the collapse at 52 um (6 of 36 windows lack two or three phases) is robust, the twelve-window medians at the two finest sizes were not. `fig_scale.py` draws every window as a dot in panel (c). The onset rule returns 0.84-0.42 mm on these data.

**Phase-domain band in Fig 3c/d (2026-10-05).** `phase_domain_sizes.py` (default `--pixel-um 0.5475`) -> `phase_domain_sizes.csv`: connected regions of every domain on the 2.5 um phase map, area-weighted median and 90th-percentile equivalent diameter; the four phases scored as modes span 154 um (plagioclase) to 853 um (biotite), with quartz 291 and muscovite 656. `fig_scale.py` shades that range in panels (c) and (d) (`--domain-sizes`, '' to disable). It replaces the plateau-onset rule, which was derived from the curve itself and moved from 0.42-0.21 mm to 0.84-0.42 mm between 12 and 36 windows per size.

**Window-size convention (2026-10-05).** Quadtree windows keep the ROI's shape (the section is 24.8 x 13.4 mm, so 1.85:1 at every level); a window's size is the SQUARE ROOT OF ITS AREA (18.2 / 9.1 / 4.6 / 2.3 / 1.1 / 0.57 / 0.28 / 0.14 / 0.071 mm for levels 0-8). `fig_scale.py` uses it in every panel; panels (c)/(d) had used the window height (6.7 ... 0.052 mm), a factor 1.36 off panel (a). The text and caption quote the square-root sizes.

**Modelled bulk and water (2026-10-05).** Every modelling script builds its MAGEMin input through `model_bulk.py`: the nine measured elements that are components of the metapelite database (Si, Al, Ca, Mg, Fe, K, Na, Ti, Mn), Fe3+/sum(Fe) = 0.1, and **H2O = 4 wt% of the bulk**. P and the SPX peak-ID artefacts (Sn 1.6, Tl 0.4, Zr 0.25 wt% of the section bulk) are dropped. Before this date the toolkit added the nominal 4 to a cation-normalised oxide total, i.e. 2.1 wt% H2O in the bulk (`Composition.to_oxide_wt(extra_basis="legacy")`); the best cell and accepted region of Fig 5 are identical at 2.1, 3.9 and 7.4 wt%, the melt-bearing end of the stability field is not. `pt_grid_run.py` now computes `pt_grid.csv` headlessly (+ `pt_grid_bulk.json`: bulk, elements passed/dropped, oxides). Full rerun: `_scratch/background_fix/rerun_h2o4/chain.sh` = `pt_grid_run.py`; `mc_spx_rescore.py`, `mc_calibrated.py`, `spx_sweep_run.py`, `eds_sweep_run.py --max-level 8` (concurrently); `spx_sweep_score.py --which both`; `window_pt_run.py --n-per-level 36 --workers 4`; `fig_pt.py`, `fig_reconstruction.py`, `fig_scale.py --pixel-um 0.5475`. The 2.1 wt% outputs are kept in `_scratch/background_fix/h2o_legacy/`.

**Results of the 4 wt% rerun (2026-10-05, all staged files).** Fig 5 grid: best cell 642 °C / 0.62 GPa (misfit 0.073) and the nine-cell accepted region 589–668 °C / 0.39–0.69 GPa are unchanged; the stability field of the five phases is 13 cells (589–695 °C / 0.39–0.69 GPa; was 15), twelve melt-free with misfit 0.07–0.10, one melt-bearing at 0.20, 0.21 or more outside; free fluid (about 7 vol%) in every accepted cell. Reconstructions: 668 / 0.47 (0.087; region 589–668 / 0.39–0.69) and 668 / 0.39 (0.095; region 642–668 / 0.32–0.39). No cell holds muscovite with K-feldspar; with water in excess K-feldspar + sillimanite exists only at or below 0.39 GPa, and above it muscovite goes straight to melt at 695 °C (at 2.1 wt% K-feldspar entered with melt). SPX Monte Carlo: stable in every draw in 10 cells (590–668 °C), in at least half in 15 (589–721 / 0.39–0.76). Calibrated Monte Carlo: every draw in 12 cells (589–695); unperturbed calibrated bulk 642 / 0.62 (0.068), region 589–668 / 0.39–0.69. Per-window inversions (36 per size): median overlap 1.00 / 0.91 / 0.91 / 0.82 / 0.78 / 0.68 / 0.70 / 0.17 at 9.1 … 0.071 mm, lower quartile 0.96 / 0.83 / 0.83 / 0.66 / 0.62 / 0.59 / 0.18 / 0.01, 10–90 % best-fit spread 0 / 0 / 53 / 66 / 105 / 105 / 158 / 250 °C and 0 / 0 / 0.22 / 0.22 / 0.33 / 0.26 / 0.41 / 0.66 GPa, windows missing a phase 2 at 0.14 mm and 6 at 71 µm. Both mean ± σ sweeps (`pt_scale.csv`, `pt_scale_spx.csv`): union envelope 589–668 °C / 0.38–0.67 GPa at every scale (the fine-scale widening of the 2.1 wt% runs is gone; it was water undersaturation of the mica-rich variants). **Stale-dump trap fixed:** `spx_sweep_score.py` preferred `*_grids.json.gz` over a fresher `.json`, so a rerun that wrote only the `.json` was scored from the old grids; the scorer now takes the newer file and both sweep scripts always refresh the `.gz` and delete the raw `.json`. **Sensitivity table (manuscript Table A.2, 2026-10-05).** Input rows: `sens_inputs.py` → `sens_inputs.json` (the SPX bulk at Fe3+/ΣFe 0 / 0.1 / 0.2 / 0.3 and H2O 2 / 4 / 8 wt%, six MAGEMin grids, scored with the adopted modes). Segmentation rows: one `segment_worked_example.py --out-dir <dir>/<tag> --fit-balance density --merge-same-name` run per variant (`--sigma 2.0|3.0`, `--min-region 15|60`, `--n-clusters 10|12`, `--seed 1..5`, and `--seed 1..5 --n-clusters 14`; about 3.5 min each, run one at a time), then `sens_seg_summary.py <dir>` → `sens_segmentation.csv` (modes after unmixing, pixel agreement with the adopted map, best cell and accepted region re-scored on `pt_grid.csv`; no thermodynamic computation). The variant phase maps stay outside git in `_scratch/background_fix/sens_seg2/` (chain script `_scratch/background_fix/rerun_h2o4/chain5.sh`). The driver names a variant's components against the adopted map: a component is a clean mineral only when ONE mineral of the adopted map holds the majority of its pixels, otherwise it is a mixed domain and is unmixed (the earlier rule let matrix-straddling components fall through to a junk automatic match such as "Calcite 5 %"; the adopted map is unaffected, `base` reproduces it exactly). Findings: modes move by at most four points over the fifteen complete variants and the accepted region keeps its extent in all of them; the ICL picks 14 components only for seed 42 (8 for seeds 1, 3, 4, 5; no minimum for seed 2, whose 4-component fallback does not separate the micas); the sillimanite domain appears only at k = 14, and there for four seeds of six.

**Contamination floor inside the EBSD oxide grains (section 5.2, 2026-10-06; reference only, no figure in the paper).** `oxide_si_distance.py` → `oxide_si_distance.png` + `.json`: on both serpdi maps (`D:\data\EBSD\serpdi_{1,2}_EBSD`, EBSD ordinals 3 = ilmenite, 4 = magnetite, 6 µm px, EDS at the 70° EBSD tilt), the apparent share of each cation window (O and C dropped) inside the EBSD-indexed oxide grains against the Euclidean distance from the grain edge, plus the share of windows holding only background (P, S, Cl, K, Y) and the Fe/Ti share outward into the silicate. `oxide_si_pergrain.py` → `oxide_grains_map{1,2}.npz` (not staged): the per-grain interior Si share (grains with ≥ 6 pixels at ≥ 2 px from the edge) and, for map 1, the Si wt% of the AZtec binning-4 QuantMap export (`_scratch/background_fix/serpdi_1_wt_maps.npz`, see the ebsd-quantified-test memory) in bins wholly inside the oxides. Findings: Si about 20 % in the edge pixel, a flat 10–13 % from 12 to 48 µm inward on both maps and both oxides, Mg 9–10 % and Al 4–5 % alongside (Si/Mg ≈ 1.2, chlorite proportions); background-only windows 1.1–1.4 % each; per-grain interior Si 5th–95th percentile 10–15 %, no grain below 3 %, no size dependence; outward Fe 8 % in the first silicate pixel against a 4 % baseline; AZtec quantification leaves 6.7–7.7 wt% Si and 7–8 wt% Mg inside the oxides. The h5oina holds window integrals only (no spectra). The W Mα window of map 2 (1.775 keV, 35 eV from Si Kα) tracks the Si window at ~0.8× everywhere: Si leakage, no W. Reading: a uniform matrix signal reaching every grain (consistent with backscattered-electron re-excitation at the tilt), not window background, lateral mixing or intergrowth.

**Seed robustness of the Fig 4b inset (2026-10-27).** `inset_seed_sweep.py` re-runs
`fig_segmentation_inset.py` (which now takes `--seed` and `--out-dir`; its window
auto-pick follows `--als-name`, so a default run works with the current legend) on
the staged window for seeds 42 and 1-5 at k = 6 and 7, plus seed 5 at k = 8, and
writes `inset_seed_sweep.txt`: the Al2SiO5 cores are recovered with five of six
seeds at k = 6 and with every seed at k >= 7. This is the evidence for the section-4.2
statement that sillimanite presence is taken from petrography and the native-resolution
nodule segmentation, not from the section-wide map (where Table A.2 records it as
seed-dependent). Per-run phase maps go to `_scratch/inset_seeds/` and are not staged.

**Nodule-core formula (section 4.2, 2026-10-27).** `nodule_cores.py` reads the Fig 4b
window from the cache with the staged inset labels and calibration and prints the
calibrated composition and formula (per 5 O, Fe3+/Fe 0.1) of the Al2SiO5 cores at
0, 1.5, 3 and 5 um erosion, with muscovite and plagioclase from the same window as
references (plagioclase reads An17.5), the section-wide domain from
`phase_stoichiometry.csv`, and a pixel-level check of Ca, Fe, Na, P inside the cores.
The text quotes the 3-5 um erosion values. Output is printed only.

**Own-assemblage scoring of the per-window inversions (2026-10-27, co-author comment).** The
published Fig 3c applied the SECTION's sillimanite presence term to every window, although
sillimanite (1.1 % of the map, nodule cores) is absent from 16 of 36 windows at 1.1 mm and from
33-35 of 36 below 0.6 mm; the plateau of 0.78-0.68 between 0.57 and 0.28 mm was supplied by that
term. `window_pt_run.py` now scores every window twice (section's constraint = the original
columns; the window's own assemblage, sillimanite required only where the window's phase map
contains it and `OWN_TOL` = 0.10 open-assemblage tolerance otherwise = the `*_own` columns plus
`sill_px`/`has_sill`), and `--from-grids` re-scores the stored `pt_windows_grids.json.gz` without
MAGEMin (used here; the grids are unchanged). `fig_scale.py` draws both curves in (c) (black /
orange, IQR bands, a four-entry legend in the lower-left corner since 2026-10-27 (direct text labels read as unattached on the wide panel); the dots are the own-assemblage windows,
filled where the window contains sillimanite, open where not) and both best-fit spreads in (d). Own-assemblage medians 1.00 / 0.91 / 0.91 /
0.62 / 0.20 / 0.09 / 0.23 / 0.03 from 9.1 mm to 71 um; the share of the section's region covered
by the window's region stays 0.9-1.0 down to 0.57 mm (windows without sillimanite keep a
four-phase region of about 27 cells against the section's 10).

**Leave-one-session-out calibration test (2026-10-27, co-author comment).** The mosaic was
acquired in three Bruker mapping jobs (2025-09-29: 774 tiles, 09-30: 490, 10-01: 750), each
spread over the whole grid. `calibration_loso.py` loads the tiles exactly as `calibrate_mosaic.py`
does, fits k per session, holds out each session in turn (fit on the other two) and scores the
residuals as `eds_quantify.cross_validate_k_factors` does -> `calibration_loso.json`. Result:
session-wise k (Si-normalised) within 2.3 % of the pooled k; LOSO RMSE 1.79 / 1.35 / 2.23 wt%
(held-out 09-29 / 09-30 / 10-01) against 1.92 for the random hold-out, each equal to that
session's own within-session hold-out, biases unchanged. Beam conditions were identical (15 kV,
same working distance, live time and pulse density). Quoted in section 5.1 as "1.4 to 2.2 wt%".

**Finer grid and region-only reporting (2026-10-27, co-author comment).** One cell of the
20 × 20 grid over 300–800 °C / 0.1–1.5 GPa was 26 °C by 0.074 GPa, so the "642 against 668 °C"
of the reconstructions was a single cell and the nil best-fit spread at coarse window sizes
(Fig 3d) was quantisation. The section-level grid is now 21 × 21 over 550–750 °C / 0.2–1.0 GPa
(10 °C by 0.04 GPa), defined once in `pt_grid_spec.py` and read by `pt_grid_run.py`,
`mc_spx_rescore.py`, `mc_calibrated.py` and `sens_inputs.py`; the suggested 0.2–0.8 GPa clipped
the five-phase field, which closes between 0.80 and 0.86 GPa. Chain: `pt_grid_run.py` ->
`mc_spx_rescore.py` -> `mc_calibrated.py` -> `sens_inputs.py` -> `sens_seg_summary.py` ->
`fig_pt.py` -> `fig_reconstruction.py` (about one hour of MAGEMin; the reconstructions and the
Table A.2 segmentation rows are re-scorings of `pt_grid.csv`). The paper reports accepted regions
only (cells within 0.02 of the minimum): `fig_pt.py` and `fig_reconstruction.py` no longer draw
the best-cell star, Table 1 lists the minimum misfit instead of the best cell, Table A.2 lost its
best-cell column. Results: as observed, minimum 0.069 at 580 °C / 0.40 GPa (a corner of the flat
valley), region 580–680 °C / 0.36–0.72 GPa (38 cells, one with melt); calibrated bulk 0.065, same
region; nodule Ms -> Kfs 0.084, 600–680 / 0.36–0.72; all Ms -> Kfs 0.095, 640–670 / 0.32–0.40;
five-phase field 580–720 / 0.36–0.80 (67 cells, 53 melt-free; misfit 0.07–0.15 in the melt-free
part, 0.19 or more outside); melt-in 670–690 °C; K-feldspar + sillimanite up to 0.40 GPa,
muscovite + K-feldspar in one cell (660 / 0.40); matrix domains merged instead of unmixed:
minimum 0.072, region two cells at 680–690 / 0.60–0.72. SPX Monte Carlo: stable in at least half
the draws over the whole field, in every draw in 32 of the 38 accepted cells; calibrated Monte
Carlo: every draw over 580–700 / 0.36–0.76. Table A.2: Fe3+/ΣFe 0.2 and 0.3 extend the region to
570 °C; 2 wt% H2O gives 550–700 / 0.36–0.76 with 22 of 58 cells without free fluid; all fifteen
complete segmentation rows keep 580–680 / 0.36–0.72 (minima scatter 580–680 °C). The per-window
inversions of Fig 3c stay on the coarse full-range grid (windows that lose a phase land outside
the fine window; the Fig 3 caption says so). Coarse-grid outputs are kept in
`_scratch/background_fix/coarse_grid_2026-10-27/`, the clipped 0.2–0.8 GPa run in
`_scratch/background_fix/fine_grid_2026-10-27/window_0p8/`. Fig 3d (best-fit spread) is undecided.

**Fig 3 panel (d) dropped (2026-10-27, co-author comment 7, user's decision).** The 10–90 % spread
of the windows' best-fit cells was the last place a best cell survived, and on the coarse
per-window grid its nil spread at coarse sizes was quantisation. A replacement was previewed
(width of each window's accepted region, hi − lo + one cell, median and IQR per size, both
scorings; `window_pt_run.py --from-grids` now writes the own-assemblage region bounds
`T_lo_C_own` / `T_hi_C_own` / `P_lo_GPa_own` / `P_hi_GPa_own` to `pt_windows.csv` for it, the
published columns byte-identical): under the section's constraint the width never leaves the
section's own (105 to 132 °C, 0.37 to 0.44 GPa at every size, pinned by the presence term);
under the window's own assemblage it equals the section's down to 1.1 mm and then grows (184 °C /
0.74 GPa at 0.57 mm, about 200 °C / 0.74 GPa at 0.29 to 0.14 mm, 355 °C / 1.03 GPa at 71 µm).
That adds one nuance to panel (c) (small windows give a vaguer answer, not a different one),
which went into one sentence of section 4.4 instead of a panel. `fig_scale.py` now puts (c) full-width under (a) and (b) (a
1 × 3 row was rejected: under two inches per panel at the text width; `GRID_T_RANGE_C` / `GRID_P_RANGE_GPA` removed); the Fig 3 caption and the two
"Fig. 3c, d" references lost the (d). Preview kept in the session scratchpad only.

**Nodule share of the muscovite mode measured (2026-10-28, co-author comment).** The partial
peak-assemblage reconstruction (Fig 6b, Table 1 row 3) used to attribute an illustrative half of
the muscovite mode to the nodules. `nodule_fraction.py` now measures it on `phase_map_labels.npz`
(2.5 µm/px): the nodules are the compact muscovite regions of at least 0.5 mm that survive a
morphological opening with a 12.5 µm radius (flakes thinner than 25 µm removed; opening and
dilation done through the distance transform), the nodule muscovite is every muscovite pixel
within that radius of such a region, and the share is taken over the whole mode, i.e. the clean
muscovite domain plus the muscovite the unmix assigns to the two matrix domains (21 % of the
mode). Result: 42 nodules of 0.5 to 1.9 mm (median 0.7 mm) holding 44 % of the muscovite domain
and **0.35 of the mode** (`nodule_fraction.json`, read by `fig_reconstruction.py`). Sensitivity:
0.25 with a 25 µm opening radius, 0.45 with a 0.3 mm size cut, 0.16 at 1.0 mm; every muscovite
region attached to a sillimanite core (closed over 2 px) gives 0.43 to 0.47; the muscovite within
0.5 / 1.0 mm of a large core (>= 0.2 mm, 24 of them) 0.19 / 0.33. Every definition is below the
old half. P-T consequence (re-scoring the fine grid over the share f): the accepted region is
580–680 °C / 0.36–0.72 GPa for f up to 0.35, 590–680 for 0.40–0.45, 600–680 at 0.50 and only
collapses to the full-conversion region (640–670 / 0.32–0.40) at f >= 0.6; the minimum misfit is
0.08 at the measured share (K-feldspar and muscovite coexist in one cell only, so the reassigned
fraction is unmatched almost everywhere). Section 4.3, 5.3, the Fig 6 caption and Table 1 row 3
(600–680 -> 580–680) were reworded accordingly; the "0.47 GPa coincides with Druguet" wording is
gone (0.47 GPa now lies between the two reconstruction pressure ranges). Diagnostic picture:
`nodule_fraction.png`.

## Section 5.3 reframed as a consistency check (2026-10-28, co-author comment)

The co-author noted that the accepted region covers most of the independent
620-700 C / 0.3-0.74 GPa range and that the same-thin-section estimate
(Rinia 2025) is not independent, so the comparison is a consistency check and
should state what would have counted as a failure. The subsection is now
"Pressure-temperature consistency"; the same-section bound is described as a
Perple_X pseudosection result and no longer counted among the independent
estimates; the text states the failure criterion (an accepted region disjoint
from the 620-700 C span, or a misfit floor of 0.19 or more, the outside-field
level already given in Section 4.3) and that the position test was unlikely
to fail given the widths. `consistency_counterfactuals.py` quantifies the
test's reach by re-scoring `pt_grid.csv` with deliberately wrong
observations (literature box = 99 of 441 cells, 22 %; biotite called chlorite:
floor 0.41, fails; muscovite called cordierite: floor 0.13, six cells at
680-730 C, passes; both matrix domains merged into quartz: floor 0.15, same
region). Those numbers appear only in the Section 5.3 provenance comment, not
in the paper. Net +12 words (7,797).

## Representative averaging scale needs a tolerance; directional variograms (2026-10-28, co-author comment)

The co-author noted that the Section 3.7 criterion ("the window size at which
the spread stops growing") never triggers because the spread grows at every
step in Fig 3a, that the example therefore returned no scale below the section
(at odds with contribution 6), and asked for directional variograms across and
along the banding. Changes: (1) the criterion now takes a tolerance, by default
the calibration's hold-out error (`eds_calibration.json` `holdout.per_element.rmse`):
a window is representative when every element's between-window spread (sample
s.d.) is at or below it. On the section every element passes at the quadrants
(9.1 mm; Si 1.6 vs 4.2 wt%) and Si and Al reach the error at 4.6 mm (1.04 and
1.12 times it), so the representative scale lies between 4.6 and 9.1 mm (four
quadrants only: a rough estimate). A 10 % relative tolerance gives the same.
Fig 3a is now spread / error with one line at 1 (`fig_scale.py --calibration-json`,
default on). (2) `fig_scale.py` panel (b) draws the semivariogram over all pairs
and within +-22.5 deg of the banding (`--banding-deg`, 0 = along the long side,
checked on the phase map: biotite bands and the folded quartz vein run along x)
and of its normal; 0.25 mm bins to 6 mm (`--variogram-max-lag 6000`, 24 bins),
all 9192 windows at the finest level, four major elements standardised and
averaged. (3) The sill is now fixed at 1 (standardised variables). The old rule
took the sill as the maximum of the binned curve, a noise peak (~1.06) with
0.76 mm bins and a 1500-window subsample, so "95 % of the sill" found where the
curve hit its noise ceiling: 2.7 mm. With the fixed sill the 95 % crossing is
1.1 mm over all directions (0.9 mm with all nine elements), 2.4 mm along the
banding, 0.6 mm across it; the curve is long-tailed (92 % at 0.6 mm, 97 % at
1.1 mm, 98 % at 2 mm), so the crossing is quoted as "about 1 mm". The 2.7 mm
figure is gone from the text; the "spread falls as patch size over window
size" sentence was a hand-wave and is gone too. Viewer: `_semivariogram`
takes `direction_deg` (Spatial-structure tab direction selector, grey
all-directions reference curve) and the Per-scale summary tab gained a
tolerance selector (calibration hold-out error when loaded, else relative %),
a spread / tolerance plot and a one-line representative-scale readout
(`_spread_by_level`, `_representative_level`; `tests/test_scale_direction.py`,
270 tests). Text: contribution 6, Section 3.7 (criterion + directional
sentence), Section 4.4 (first paragraph + "three length scales"), Fig 3 caption;
net +58 body words (7,851). Exploratory analysis scripts staged beside this file: scale_tolerance.py
(spread vs tolerance per level, old pt_scale regions, coarse directional
variograms), directional_fit.py (0.95-sill crossing vs exponential fit, all
elements), banding_orientation.py (biotite indicator variograms along x/y +
phase_map_overview.png).
