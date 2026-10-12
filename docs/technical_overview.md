# THERMOD — Technical Overview

This document describes the core aspects of the THERMOD toolkit:

1. [Thermodynamic backends — common approach](#1-thermodynamic-backends--common-approach)
2. [Backend-by-backend differences](#2-backend-by-backend-differences)
3. [High-resolution EDS map construction](#3-high-resolution-eds-map-construction)
4. [EDS mosaic cache](#4-eds-mosaic-cache)
5. [Phase segmentation](#5-phase-segmentation)
6. [Composition vs length scale + thermod run records](#6-composition-vs-length-scale--thermod-run-records)
7. [Oxford AZtec CSV import](#7-oxford-aztec-csv-import)

---

## 1. Thermodynamic backends — common approach

All three backends share a unified interface defined by `AbstractBackend` in
`thermo/backends/base.py`. User code interacts exclusively through the
`ThermoCalc` facade in `thermo/calc.py`, which accepts a backend name string
(`"magemin"`, `"perplex"`, or `"pycalphad"`) and constructs the appropriate
class on demand. Backends are imported lazily so that missing optional
dependencies (Julia, a TDB file, a Perple_X installation) do not break
unrelated code at import time.

### Composition

The universal input type is `Composition` (`thermo/composition.py`). It stores
data internally as mole amounts and can be constructed from oxide wt% or
element wt%, and converted to whichever format a backend requires. Each backend
performs its own conversion internally, so the caller never needs to manage
unit transformations.

```python
Composition.from_oxide_wt({"SiO2": 45.4, "Al2O3": 9.0, "FeO": 8.0, ...})
Composition.from_element_wt({"Fe": 70.0, "Cr": 18.0, "Ni": 12.0})
```

### Results

Every backend returns `ThermoResult` objects (defined in `thermo/result.py`):

```
ThermoResult
  p_gpa, t_c              — conditions
  converged               — bool
  backend                 — name string
  stable_phases           — List[PhaseResult]
  bulk_properties         — dict: "rho_kgm3", "Vp_kms", "Vs_kms", etc.

PhaseResult
  name, mol_fraction, wt_fraction, vol_fraction
  is_melt
  composition_oxide_wt    — per-phase oxide wt% (MAGEMin only)
  endmembers
```

`ThermoCalc.to_dataframe()` converts a list of results to a tidy pandas
DataFrame with one row per P-T point and columns for every bulk property and
per-phase vol/mol fraction. This is the format used by the viewer's
pseudosection grid display.

### Single-point vs grid

`run(composition, p_gpa, t_c)` computes a single equilibrium point.
`run_grid(composition, p_range, t_range, n_p, n_t)` computes a regular P-T
grid and returns a flat list of `ThermoResult` objects. The grid is the basis
for pseudosection visualisation in `viewer.py`.

---

## 2. Backend-by-backend differences

### 2.1 MAGEMin (`MAGEMinDirectBackend`)

**File:** `thermo/backends/magemin.py`

MAGEMin calls `MAGEMin_C.jl` directly through `juliacall`. It is the primary
geological backend, suited to silicate melts and metamorphic systems.

#### Setup

On first use, `petthermotools` is used to activate its managed Julia environment
and load `MAGEMin_C.jl`:

```python
import petthermotools as ptt
ptt.activate_petthermotools_env()      # set up Julia project
jl.seval("using MAGEMin_C")            # load the library
```

#### Execution model

For a grid calculation, `multi_point_minimization()` is called once with
Julia-typed `Float64` arrays for all P-T points simultaneously. Arrays must be
constructed as Julia literals (not passed as NumPy arrays), because NumPy
arrays arrive in Julia as `PyArray{Float64}` and fail dispatch checks:

```python
jl_x = jl.seval(f"Float64[{', '.join(repr(float(v)) for v in x)}]")
```

**Threading constraint**: `multi_point_minimization` uses Julia's `@threads
:static`, which deadlocks if `juliacall` is first initialised from a QThread.
In `viewer.py`, MAGEMin is therefore always activated on the Qt main thread
before any worker is started, and grid calculations run synchronously (not in a
worker thread).

#### Databases supported

`"ig"` (igneous), `"igad"`, `"mp"` (metapelite), `"mb"` (metabasalt),
`"um"`, `"mpe"`, `"mbe"`, `"mtl"`.

#### Output richness

MAGEMin returns the most detailed output of the three backends: per-phase oxide
compositions (`composition_oxide_wt` in `PhaseResult`), seismic velocities
(Vp, Vs), elastic moduli, fO₂, melt/solid/fluid fractions, enthalpy, entropy,
Cp, and thermal expansivity α. Several result fields (`enthalpy`, `entropy`,
`s_cp`, `alpha`) are returned as 1-element Julia vectors and must be
unwrapped with a scalar-coercion helper (`_sc()`).

#### Fe³⁺ handling

The `fe3_fraction` parameter (default 0.1) controls the Fe³⁺/Fe_total ratio
applied when converting `Composition` to the oxide vector expected by MAGEMin.

---

### 2.2 Perple_X (`PerplexBackend`)

**File:** `thermo/backends/perplex.py`

Perple_X is a classical thermodynamic software package. The backend drives it
headlessly through three sequential subprocesses: BUILD, VERTEX, and WERAMI.

#### Pipeline

```
BUILD  →  proj.dat        (problem definition: grid axes, composition, phases)
VERTEX →  proj.plt        (binary solution cache for the entire P-T grid)
WERAMI (×2):
  run 1  →  density (property 2)   →  werami.tab
  run 2  →  phase modes (property 25)  →  werami_phases.tab
```

Each step communicates with its executable via a carefully constructed `stdin`
string. The dialog sequences differ significantly between Perple_X 6.x and
7.x; the backend targets 7.2.2.

#### Caching

The run directory is identified by a 12-character SHA-1 hash of the
composition + dataset + grid parameters. If `proj.plt` already exists and is
non-empty, VERTEX is skipped. If `werami.tab` already exists, that WERAMI run
is also skipped. This allows a composition to be calculated once and then
queried cheaply for different properties.

#### Key implementation details

- **Unit conversions**: P is converted from GPa to bar for BUILD input; T from
  °C to K.
- **fe3_fraction = 0**: `hp633ver.dat` uses FeO, not Fe₂O₃, as a component.
  The backend always passes `fe3_fraction=0.0` when building the composition
  string, regardless of the user setting.
- **Dataset filtering**: each thermodynamic dataset supports only a subset of
  oxides. Invalid oxide names in the BUILD dialog cause `"Try again:"` prompts
  that consume subsequent stdin tokens and corrupt the entire dialog sequence.
  The backend pre-filters the composition against `DATASET_VALID_OXIDES` before
  building the stdin stream.
- **Minimum grid size 5×5**: VERTEX hangs indefinitely on grids smaller than
  approximately 5×5. The option file always clamps `x_nodes` and `y_nodes` to
  `max(n, 5)`.
- **Empty placeholder files**: BUILD creates `proj.plt`, `proj.arf`, etc. as
  0-byte placeholder files on start. Size checks must verify `st_size > 0`, not
  just file existence, before deciding to skip a step.
- **H₂O-saturated fluid**: when `fluid="H2O-sat"`, four additional dialog
  tokens appear in BUILD: the fluid component name, a blank terminator, an EoS
  selection (`"5"` = Holland-Powell H₂O-CO₂, appearing *after* the
  thermodynamic component list), and a y-axis selection prompt (`"2"` for
  P(bar)) that only appears when a fluid is active. A fixed Y(CO₂) = 0 is
  sent to specify pure H₂O.
- **VERTEX output**: VERTEX is run with `subprocess.Popen` and its stdout is
  streamed line-by-line so the user sees live progress in the terminal.
- **WERAMI property 25** (phase modes): this property has two extra prompts
  before the "no more properties" token — cumulative modes (`"n"`) and include
  fluid in aggregate (`"n"`). Missing either causes the dialog to hang.
- **Output format**: WERAMI 7.x tab files start with a `|6.6.6` version tag
  and a structured header (variable names, min/step/count). The parser checks
  for `|` on the first line to select the 7.x parsing path.

#### Single-point `run()`

Implemented as a tiny 1×1 grid (actually 5×5 with a small P and T perturbation
to satisfy the minimum grid size constraint). Returns the first result point.

#### Per-phase output

Phase modes from WERAMI property 25 give vol% per named phase (e.g. `"Gt(W)"`,
`"melt(G)"`). These are parsed into `PhaseResult` objects with
`vol_fraction = val / 100`. Phases with NaN values or vol% ≤ 0 are absent.

---

### 2.3 pycalphad (`PycalphadBackend`)

**File:** `thermo/backends/pycalphad.py`

pycalphad is a Python-native CALPHAD engine suited to alloys, ceramics, and any
system for which a TDB (Thermodynamic Data Base) file exists. It is less
commonly used for geological silicates, which are not distributed in TDB format.

#### Execution model

pycalphad's `equilibrium()` function accepts arrays of T and P conditions and
evaluates all points in a single vectorised call using numba-JIT loops. For a
grid calculation, T and P arrays are passed directly, making the grid much
faster than looping over individual points.

Compositions are passed as element mole fractions (`v.X(El)`). All but the
last element (alphabetically) are specified; pycalphad derives the remainder by
mass balance. The vacant component `"VA"` is added automatically.

Units differ from the geological backends: temperature in K, pressure in Pa
(`p_gpa × 1e9`). The backend handles all conversions.

#### Database

A TDB file must be provided via `tdb_path`. The database is loaded lazily on
first use and cached in `self._db`. Elements in the composition that are not
present in the database are silently excluded.

#### Output

Phase names and mole fractions (`NP`) are extracted from the pycalphad xarray
`Dataset`. Bulk properties such as density or Vp are not currently extracted
(pycalphad does not compute them from first principles without additional
property models).

---

### 2.4 Comparison summary

| Feature | MAGEMin | Perple_X | pycalphad |
|---|---|---|---|
| Primary use case | Geological (igneous, metamorphic) | Geological (metamorphic) | Alloys, ceramics |
| Computation approach | Julia JIT, in-process | External subprocesses | Python/numba, in-process |
| Grid caching | None (fast enough) | SHA-1 run directory | None |
| Bulk properties | Vp, Vs, ρ, moduli, fO₂, H, S, Cp, α | ρ (property 2) | Phase names/fractions only |
| Per-phase compositions | Yes (oxide wt%) | No | No |
| Threading constraint | Must run on main thread | Worker-thread safe | Worker-thread safe |
| Fluid handling | Dissolved via H₂O in composition | H₂O-sat or H₂O-CO₂ mode | Via TDB fluid phases |
| Fe³⁺ control | `fe3_fraction` parameter | Always fe3_fraction=0 (FeO only) | Via TDB definition |
| Database | Built-in (`ig`, `mp`, etc.) | External `.dat` files | External `.tdb` file |
| Single-point speed | Fast (~seconds) | Slow first run; cached thereafter | Moderate |

---

## 3. High-resolution EDS map construction

**File:** `eds_map.py`  
**Primary function:** `roi_to_banded_maps()`

### Overview

EDS maps are not stored as pre-computed images. Instead, they are assembled
on-demand from the raw BCF files by streaming the EDS spectrum data for only
the tiles that intersect the requested Region of Interest (ROI). This avoids
loading the entire dataset into memory.

The key output is a dict of dicts: `{element: {"main", "sb_lo", "sb_hi"}}`,
where each entry is a 2D float32 array in BSE pixel coordinates. The main band
is the net-count intensity map for that element; the two sideband maps (`sb_lo`,
`sb_hi`) are symmetric background windows used for background subtraction.

### Step-by-step pipeline

#### 1. Manifest lookup

A `stitch_manifest.csv` is written during mosaic stitching. It records for each
tile: its grid position (row, col), absolute pixel position in the mosaic
(y, x), BSE tile size (bse_h, bse_w), EDS navigation grid size (eds_h, eds_w),
scale factors (scale_y, scale_x), per-tile shear (shx, shy), and energy axis
parameters (E0, dE, nE).

The energy axis (offset E0, step dE, channel count nE) is read from the
manifest first (no BCF I/O needed) and used to convert from energy in keV to
channel index.

#### 2. ROI prefiltering

Before opening any BCF file, all manifest rows are tested against the ROI bounding
box using axis-aligned bounding box (AABB) intersection. Only tiles that
overlap the ROI are processed. For a typical ROI this reduces the tile count
from hundreds to a handful.

#### 3. Band index calculation

For each requested element, the characteristic X-ray line energy (from
`LINE_ENERGIES_KEV`, a built-in table of Kα energies) is converted to a channel
window:

- **Main band**: `±width_eV/2` around the line energy (default 150 eV full width)
- **Sideband lo**: same width centred at `line_energy − 1.5 × width_eV`
- **Sideband hi**: same width centred at `line_energy + 1.5 × width_eV`

Net counts can then be computed as `main − 0.5 × (sb_lo + sb_hi)` to subtract
the local continuum background.

#### 4. Parallel tile processing

Intersecting tiles are processed in parallel using a `ThreadPoolExecutor` (up
to min(n_tiles, cpu_count, 8) workers). Each tile is processed in isolation
by `_process_one_tile()`, which has no shared mutable state.

For each tile:

**a) Load EDS signal lazily** via HyperSpy (`hs.load(..., lazy=True)`). The
raw data is not materialised until specific channels are accessed.

**b) Energy axis check**: the tile's energy axis parameters must match the
reference axis locked at step 1 (E0 within 1e-6 keV, dE within 1e-9 keV).
Mismatched tiles are skipped.

**c) Batched energy-axis read**: instead of calling HyperSpy's lazy slicer
`sig.isig[lo:hi].sum(axis='Energy').data` once per `(element, band)` pair — 33
times per tile in a typical 11-element run, each paying ~1 s of dask task-graph
dispatch overhead — the pipeline now materialises the union window
`[min_lo, max_hi]` once via `np.asarray(sig.isig[min_lo:max_hi].data)`. This is
a single dask compute per tile. Each band sum is then a plain numpy
`sub[..., lo-min_lo:hi-min_lo].sum(axis=sig_axis_idx, dtype=float32)`. For
2014 tiles this collapses ~66 000 dask dispatches to ~2000. Profiled on a real
dataset the HyperSpy-read step dropped from 365 s to 56 s (10 tiles), driving
a ~3× end-to-end speedup on the per-tile EDS pipeline. A per-band fallback
path is retained if the batched read fails.

**d) Scale resampling**: the EDS navigation grid may have a different pixel
count than the BSE tile (the scale factors `scale_y`, `scale_x` record this
ratio). The extracted 2D image is resampled in two steps using bilinear
interpolation (`skimage.transform.resize`, order=1): first to the EDS nav size,
then to the BSE tile size.

**e) Shear correction**: if the tile has a non-zero shear (`shx`, `shy`), an
affine transform is applied to correct for any mechanical skew in the stage
coordinate system:

```
M = [[1, shx], [shy, 1]]
img_corrected = affine_transform(img, inv(M))
```

**f) Sub-pixel shift**: the tile's global position (y, x) typically has a
fractional component (the sub-pixel offset computed during BSE stitching). This
fractional shift is applied so that EDS data aligns with the BSE composite at
sub-pixel accuracy.

Two methods are supported via `shift_method`:

- **`bilinear`** (default) — `scipy.ndimage.shift(order=1)`. Per-slice bilinear
  interpolation. Fast (~8× faster than FFT on the shift step) and visually
  indistinguishable from FFT for smooth EDS counts. EDS maps are already
  noise-limited, bilinear-resampled from the coarser EDS nav grid, and
  Gaussian-smoothed downstream during phase segmentation — the frequencies that
  FFT/sinc would preserve exactly are precisely the ones that get blurred out
  anyway.
- **`fft`** — `scipy.ndimage.fourier_shift` with a batched 2D FFT over the
  `(n_bands, h, w)` band stack. Sinc interpolation, exact for bandlimited
  signals, matches the interpolation model used when the offsets themselves
  were measured by phase correlation. Kept as an option when strict fidelity
  matters more than throughput.

The choice is exposed as `--eds-shift-method {bilinear,fft}` on `stitch.py`
and as a combo in the viewer's EDS control panel. Session save/load
preserves the viewer selection.

**g) ROI intersection and feathering**: the portion of the tile that falls
within the ROI bounding box is computed. A 2D Tukey (cosine-tapered) window
(default alpha=0.08) is computed for the full tile and the relevant window
patch is extracted. This tapers tile edges to zero, suppressing seam artefacts
at tile boundaries when multiple tiles overlap.

If a polygon ROI mask is provided (when the user drew a polygon shape in napari
rather than a rectangle), it is multiplied into the Tukey window, setting
out-of-polygon pixels to zero weight.

#### 5. Accumulation and normalisation

Results from all threads are accumulated into per-element, per-band float32
arrays of shape (H, W) along with a scalar weight map. After all tiles are
processed:

```
accum[el][band][y, x] += patch_value × tukey_weight
weight[y, x]          += tukey_weight
```

With `blend_mode="mean"` (default), each pixel is divided by its accumulated
weight. Pixels covered by only one tile receive weight ≈ 1.0; pixels in overlap
zones receive the sum of the two tiles' Tukey weights and the mean is
automatically the correct weighted average.

The raw weight array (before normalisation) is retained and returned optionally
as a byproduct. It is used downstream for tile-overlap detection in the phase
segmentation pipeline.

#### 6. Output

The function returns `{element: {"main": arr, "sb_lo": arr, "sb_hi": arr}}`.
All arrays are float32 with shape `(y1−y0, x1−x0)` in BSE pixel units,
aligned with the stitched BSE mosaic. These are what the viewer displays as
coloured EDS layers in napari.

---

## 4. EDS mosaic cache

**File:** `eds_cache.py`
**Entry points:** `build_cache()` (producer) and `roi_from_cache()` (consumer)

### Motivation

The streaming construction in Section 3 is memory-efficient but re-opens every
intersecting BCF on each ROI request. For a typical ROI this is 5–20 tiles and
1–2 minutes of BCF parsing. When iterating on phase-segmentation parameters or
running thermodynamic modelling from many ROIs on the same mosaic, the per-tile
work repeats unnecessarily. The EDS mosaic cache stores the aligned,
feather-blended tile contributions once, over the whole mosaic, so that later
ROI requests are pure memmap slicing (milliseconds).

### Layout

The cache lives under `<stitch_dir>/eds_cache/` and contains:

```
manifest.json              version, stitch fingerprint, elements, width_eV, ds,
                           H, W, E0, dE, nE
weight.npy                 float32, shape (H, W) — shared across all elements
{El}__main.npy             float32, shape (H, W) — raw weighted sum (main band)
{El}__sb_lo.npy            float32, shape (H, W) — sideband low
{El}__sb_hi.npy            float32, shape (H, W) — sideband high
```

All data arrays are NumPy `.npy` files opened with `np.lib.format.open_memmap`
so that reads slice directly off disk without materialising the full mosaic.

### Fingerprint and validity

`manifest.json` stores `stitch_fingerprint`, a SHA-256 hash of the
`stitch_manifest.csv` bytes. The cache is valid only if the fingerprint matches
the current `stitch_manifest.csv`. Any re-stitch — even one that reproduces the
same positions within rounding — changes the CSV and invalidates the cache. In
addition to the fingerprint, a read request also requires that the cache covers
the requested element list and matches `width_eV` (to within 1 µeV).

### Raw sums, not means

The cache stores un-normalised weighted sums alongside a per-pixel weight map,
not pre-normalised mean intensities. This means:

- `blend_mode="sum"` reads: return the raw array directly.
- `blend_mode="mean"` reads: divide by the weight slice at read time
  (`out = sum_slice / weight_slice` at pixels where weight > 0).

This keeps the cache compatible with both modes and matches the semantics of
`roi_to_banded_maps`.

### Build pipeline

`build_cache(manifest_path, elements, width_eV, ds=1, shift_method="bilinear", …)`
runs the same per-tile pipeline as `roi_to_banded_maps` but over *every* tile
in the manifest, writing each tile's contribution into the mosaic-wide memmaps.
The `shift_method` used is recorded in `manifest.json` so the cache is
self-describing:

1. Resolve the energy axis from the manifest (no BCF I/O unless the manifest
   lacks `E0/dE/nE` for every tile).
2. Compute mosaic extent from `y + bse_h`, `x + bse_w` over all rows.
3. Allocate memmapped zero-initialised arrays.
4. For each tile, call `_process_one_tile` (shared with `roi_to_banded_maps`)
   with a mosaic-wide ROI; accumulate patches at their mosaic coordinates.
5. Flush memmaps, write `manifest.json`.

The per-tile loop runs under a `ThreadPoolExecutor`. Parallel writes to
disjoint memmap regions are safe because each tile writes to a distinct
`(yB0:y_end, xB0:x_end)` rectangle.

### Memory management during long builds

A 2014-tile full-res build has four independent memory pressure sources.
Leaving any one of them unmanaged fills tens of GB and freezes the workstation.

- **No zero-initialisation of fresh memmaps.** `allocate_memmaps` creates
  ~31 empty memmap files totalling ~170 GB at full-res and explicitly does
  *not* write zeros across them. A freshly-created memmap on Windows
  NTFS and on Linux reads as zeros from any unwritten page (VDL / sparse-file
  semantics), so an explicit ``mm[:] = 0`` is both redundant and
  catastrophic — it would dirty every page of every file at the start of
  the build, before any tile is processed, forcing Windows to page other
  processes out to the swap file immediately.
- **Completed-Future result accumulation.** Each `_process_one_tile` result
  dict holds ~100 MB of band patches. Submitting all 2014 tasks up front and
  iterating with `as_completed` keeps every Future alive until the batch ends
  — so completed results silently pile up in Future objects. The loop uses a
  **bounded submission window** (`n_workers × 2` pending tasks) to cap the
  number of live result dicts. Tasks are resubmitted one-by-one as earlier
  ones complete, so Futures (and their results) are released incrementally.
- **Subcube RAM per worker** (`--eds-cache-workers`, default 4): each tile
  worker loads one EDS subcube (~0.8–1.5 GB, uint16) covering the union
  energy window. Peak ≈ `workers × 1.5 GB`. Reduced from the older default
  of 8 to keep subcube peak around 3–6 GB.
- **Dirty memmap pages** (`--eds-cache-flush-every`, default 50 tiles):
  `memmap.flush()` on each cache file every N tiles. On Windows this is
  `FlushViewOfFile` (asynchronous mark-for-writeback — doesn't block).
  Once pages are clean, Windows can evict them under memory pressure.
  Without periodic flush, a 60 GB cache accumulates ~60 GB of dirty pages,
  forcing Windows to page other processes to the swap file.
- **Cache downsample** (`--eds-cache-ds`): reduces total cache size by
  `ds²`, directly reducing memory pressure on the memmap side.

Expected peak RAM with defaults (4 workers, flush-every-50): **8–15 GB** on
a 2014-tile × 10-element × ds=1 build.

### Downsampled caches

When `ds > 1`, each tile's patch is block-averaged onto a cache grid that is
`ds` times coarser than the BSE mosaic before accumulation. ROI reads at
`ds > 1` slice the downsampled grid, divide by weight (for mean), then upsample
back to BSE resolution using bilinear interpolation. This preserves the same
caller-visible output shape at any `ds`.

### Preview PNG derivation

When the cache is available, `eds_cache.write_preview_pngs()` replaces the
standalone `build_eds_previews_fast()` pipeline. The function reads each
element's ``{El}__main.npy`` and ``weight.npy`` in row-chunks sized to fit a
~256 MB working set, block-averages (weighted mean: sum(main) / sum(weight)
per block) to the requested preview resolution, percentile-clips (1–99%),
and writes 8-bit PNGs to ``<dir>/eds_preview_ds{ds}/``. Only the ``main``
band is rendered — sidebands exist for background subtraction at ROI time,
not for display. No BCF I/O is required. For a 500 Mpx mosaic at
``preview_ds=20`` this takes seconds per element, compared to minutes for the
tile-stream pipeline. The fallback path (``build_eds_previews_fast``) is only
used when ``--no-eds-cache`` is set or the cache is invalid for the request.

### Cache relocation (`--eds-cache-dir`)

The cache doesn't have to live inside the stitch directory. When
`--eds-cache-dir PATH` is passed to `stitch.py`, `build_cache()` writes both
the cache files at `PATH` and a sidecar pointer `<base_dir>/eds_cache_link.txt`
containing that absolute path. Every later lookup goes through
`resolve_cache_dir(base_dir)`, which:

1. Reads the sidecar if it exists and points at a valid directory — returns that path.
2. Otherwise falls back to the default `<base_dir>/eds_cache/`.

This is the intended way to put the cache on an SSD when source tiles live
on an HDD — the slow medium handles BCF reads (cheap, sequential) while the
SSD serves ROI memmap slices (random-access heavy). Subsequent stitch runs
follow the link automatically; `--eds-cache-dir` is only needed on the first
build (or to relocate). Delete `eds_cache_link.txt` to revert to the default.

### ROI read access pattern

`roi_from_cache` reads in row-chunks (≤500 rows per chunk) and ticks progress
per chunk — so total chunks, and the progress bar, scale with ROI height.
Each slice uses one of two strategies depending on the ROI's column range
relative to the cache's full width:

- **Strided** (ROI width < 25% of cache width). Reads only the ROI columns
  via `mm[y0+cy0:y0+cy1, x0:x1]`. Less total data transferred; good for
  narrow ROIs where per-row seek cost stays bounded.
- **Full-rows** (ROI width ≥ 25% of cache width). Reads full-width rows
  `mm[y0+cy0:y0+cy1, :]` then column-slices in RAM. Sequential-friendly —
  helpful on SSDs (where readahead amplifies) and on well-laid-out HDDs.
  More wasted bandwidth, but usually faster when the column range is wide.

The tqdm postfix announces which mode was chosen (`mode=strided` or
`mode=full-rows`) along with `chunk_rows` and file count.

`max_workers` defaults to **1**. On HDDs, parallel reads across multiple
cache files thrash the physical head; on SSDs, sequential I/O bandwidth is
already the ceiling for single-file reads at this chunk size, so the marginal
benefit of more workers is modest. Callers on SSD can pass `max_workers=4–8`.

### Viewer integration

`viewer.py`'s `on_compute()` resolves the cache dir and consults validity
before starting any per-tile work:

```python
cache_dir = eds_cache.resolve_cache_dir(Path(self.manifest).parent)
if eds_cache.is_valid_for(cache_dir, manifest, elements, width_eV):
    # hand off to _EDSCacheWorker (QThread) with a QProgressDialog so the
    # UI stays responsive. The worker calls roi_from_cache() with a
    # progress_cb that feeds both tqdm and the Qt bar.
```

On a miss the code falls through to `_EDSWorker` (the per-tile pipeline), so
a missing or mismatched cache never blocks ROI computation; it just makes it
slow. The progress dialog's Cancel button is wired through the worker's
`cancel_flag` so long reads on slow storage are interruptible.

### Re-stitch handling

`stitch.py` runs `_build_or_refresh_eds_cache` after writing a new manifest.
The four possible states are:

| State                                    | Action                          |
|------------------------------------------|----------------------------------|
| No cache                                 | Build fresh cache (if elements) |
| Valid cache, same elements + width       | No-op                            |
| Valid cache, new element or new width/ds | Rebuild                         |
| Stale cache (fingerprint mismatch)       | Prompt user, or apply `--eds-cache-on-restitch` (`rebuild`, `keep`, `discard`) |

Interactive prompt is bypassed when stdin is not a TTY — safer for scripted
runs, where the explicit flag is expected.

### Maintenance CLI

The module is executable for quick diagnostics:

```
python eds_cache.py status <cache_dir>       # fingerprint, elements, ds, size
python eds_cache.py defragment <cache_dir>   # rewrite each .npy to a fresh extent
```

`defragment` is useful when an HDD-stored cache accumulated many non-contiguous
extents during the tile-by-tile build (each extent boundary is a seek at read
time). It copies every file to a `.defrag-tmp` and atomically swaps —
requires ~5× the largest single-file size in free disk space, and takes
order-of-disk-sequential-write time (~hours for a 168 GB cache on slow
storage). Only meaningful on HDDs and when fragmentation — not raw disk speed
— is the bottleneck; the physical-media limit can't be worked around.

### Disk footprint

For an H × W × 4-byte float32 layout with three bands per element plus the
shared weight:

```
bytes ≈ H × W × 4 × (3 × n_elements + 1) / ds²
```

At the user's actual 2014-tile C23S1A_Quant mosaic (26783 × 49669 px, 11
elements):

| `ds` | Approx disk |
|------|-------------|
| 1    | ~168 GB     |
| 2    | ~42 GB      |
| 4    | ~10 GB      |
| 8    | ~2.6 GB     |

Full-res on HDD is painful (per-ROI seeks dominate), so the recommended
combinations are either `ds=1` + `--eds-cache-dir` to an SSD, or
`ds=4`/`ds=8` on the HDD. Inspect with `python eds_cache.py status <dir>`.

---

## 5. Phase segmentation

**File:** `viewer.py`  
**Entry point:** `on_segment_phases()` → `_PhaseSegDialog`

Phase segmentation converts per-element EDS intensity maps into a discrete
label map where each pixel is assigned to a mineralogical phase cluster. The
pipeline runs inside a `QProgressDialog` and reports progress at each step.

### Data sources

Three data sources are available from the segmentation dialog:

- **Full preview** — the downsampled EDS preview images loaded at viewer startup
- **ROI from preview** — a cropped region from the preview, no full-resolution
  computation needed
- **ROI full-resolution** — the full-resolution band maps from `roi_to_banded_maps()`

### Pipeline steps

#### Step 1: Net count maps

For each element, the main-band counts minus the sideband average give net
counts (background-subtracted intensity):

```python
net[el] = clip(main − 0.5 × (sb_lo + sb_hi), 0, ∞)
```

#### Step 2: Sparse element exclusion

Elements whose raw zero-fraction exceeds `max_zero_pct` (default 90%) are
automatically excluded before building the feature matrix. These are elements
that are almost entirely absent from the ROI and would contribute only noise.

#### Step 3: Spatial smoothing (`_smooth_element_maps`)

Each element map is smoothed with `scipy.ndimage.gaussian_filter` (default
sigma = 3.0 pixels). This is essential for EDS data: individual-pixel spectra
have very low counts and high Poisson noise. Smoothing averages out per-pixel
noise while preserving phase boundaries at the scale of a few pixels.

Without this step, clustering operates on salt-and-pepper noise rather than
true phase identity.

#### Step 4: Feature matrix construction (`_build_feature_matrix`)

All element maps are cropped to a common shape and stacked into a 2D matrix
of shape `(n_pixels, n_elements)`. Two filters then exclude low-quality pixels:

**Tile overlap exclusion**: pixels in overlap zones (where multiple tiles
contribute) are removed from the clustering. In overlap zones, EDS signals from
two adjacent tiles are averaged, producing intermediate compositions that are
not physically meaningful and would otherwise seed spurious cluster splits.

- For full-resolution ROI maps: determined from the weight array returned by
  `roi_to_banded_maps()` — pixels where weight > 1.05 are overlap pixels.
- For preview maps: determined geometrically from manifest tile positions using
  `_compute_overlap_mask_from_manifest()`.

**Low-count pixel filter**: pixels with very low total counts are excluded. Two
complementary thresholds are applied and the stricter one wins:

- *Percentile filter* (default 5th percentile): removes the lowest few percent
  of all non-zero pixels — catches general background noise.
- *Median-ratio filter* (default 0.10 × median total count): removes any pixel
  whose total count is below 10% of the median — specifically targets entire
  under-exposed tiles rather than individual noisy pixels.

All-zero element columns are also dropped.

#### Step 5: Normalisation (`_normalize_features`)

Two normalisation options:

**Row normalisation** (default): each pixel vector is divided by its total
count, converting absolute intensities to apparent compositions (fractions
summing to 1). Optional log1p compression (`log1p(x × 1000)`) reduces the
dynamic range. This makes the clustering insensitive to tile-to-tile beam
current variations.

**Anscombe transform**: `sqrt(counts)` variance-stabilises Poisson noise so
that variance is approximately 0.25 regardless of count level. Better behaved
at very low counts (< 10 per pixel).

Both paths end with a per-column robust z-score: `(x − median) / MAD`. Using
MAD (median absolute deviation) instead of standard deviation makes the
standardisation insensitive to outlier pixels.

#### Step 6: PCA dimensionality reduction

Principal Component Analysis is fitted on a random subsample of up to 30,000
pixels and then applied to all valid pixels. The number of components retained
is determined by a cumulative explained variance threshold (default 0.92), with
a minimum of 2 components.

**Why PCA before clustering?**

Grain boundary pixels have intermediate compositions between adjacent phases.
In the full element space, these pixels form their own compact region that
provides a genuine log-likelihood boost exceeding the BIC penalty, so BIC-only
selection treats them as a separate phase and keeps adding components. In PCA
space, boundary pixels project along lines between cluster centres and are
already well described by the adjacent phase Gaussians; the marginal gain from
a dedicated "boundary cluster" falls below the penalty threshold.

#### Step 7: Auto-k selection via ICL (`_select_k_auto`)

For each k from 2 to k_max (default 15), a Gaussian Mixture Model is fitted
in the PCA-reduced space (`GaussianMixture`, `n_init=5`, `max_iter=200`,
`reg_covar=1e-4`). Two model selection criteria are computed:

```
BIC(k)  = −2 × log_likelihood + n_params × log(n_pixels)
EN(k)   = −Σ_i Σ_j r_ij × log(r_ij)     [classification entropy]
ICL(k)  = BIC(k) + 2 × EN(k)
```

where `r_ij` is the posterior probability that pixel i belongs to cluster j.

**ICL** (Integrated Classification Likelihood) penalises overlapping clusters.
Grain boundary pixels, which have high posterior uncertainty between adjacent
phases, contribute heavily to classification entropy. Adding a dedicated
boundary cluster lowers BIC (higher likelihood) but raises EN (more overlap),
so ICL detects this as undesirable and finds a genuine minimum at the true
number of phases.

**Entropy numerical note**: the entropy sum uses
`np.where(r > 0, np.log(np.maximum(r, 1e-300)), 0.0)` to avoid `0 × log(0)`
producing NaN on some NumPy versions.

**Primary selection**: argmin(ICL). If the minimum is at k < k_max, that k is
chosen (method `"icl_min"`).

**Fallback**: if ICL is monotonically decreasing (minimum at k_max), knee/elbow
detection is applied. The chord from (2, ICL(2)) to (k_max, ICL(k_max)) is
drawn; the k at maximum perpendicular distance from the chord is selected
(method `"icl_knee"`).

A minimum k floor (`min_k`, default 2) is applied after selection.

**Manual k**: the user can override auto-k with a fixed cluster count, which
skips the sweep and fits a single GMM.

#### Step 8: Final labelling

The chosen GMM (fitted on the 30k subsample) is used to predict labels for all
valid pixels via `best_gmm.predict(X_norm_r)`.

#### Step 9: Morphological cleanup (`_clean_label_map`)

Isolated regions smaller than `min_region_size` pixels are reassigned to the
dominant neighbouring cluster. The procedure is:

1. Label all connected components within each cluster (using
   `scipy.ndimage.label`).
2. Identify components smaller than `min_region_size`.
3. Mark their pixels with a sentinel value (-2).
4. Apply a 3×3 median filter and fill in the sentinel pixels from the filtered
   result.

#### Step 10: Nearest-neighbour fill

Pixels excluded from clustering (tile overlap zones, low-count pixels) receive
no label. These are filled via nearest-neighbour lookup using
`scipy.ndimage.distance_transform_edt`: for each excluded pixel, the label of
the nearest valid labelled pixel is assigned. This ensures the output label map
is dense (no holes).

### Output

The result is an int16 label map of shape (H, W) where each value is a cluster
index (0..k−1) and -1 indicates an unresolvable pixel (should be rare after
nearest-neighbour fill).

This is displayed in napari either as a single combined RGBA layer ("Phase
segmentation") or as individual per-cluster layers (one per phase, toggleable).
All RGBA layers use uint8 instead of float32 to reduce GPU memory by 4×.

### Classification dialog

After segmentation, a `_PhaseClassDialog` presents:

- A colour-coded phase map
- A table with dominant element signatures and pixel counts per cluster
- Editable phase name fields
- Options to export as combined map, per-cluster masks, or both, with a legend
  CSV
- A "Show BIC/ICL curve" button that plots both criteria with the chosen k marked
- A "Use for P-T estimate" button that passes the identified phase names to the
  thermodynamic modelling dialog

---

## 6. Composition vs length scale + thermod run records

This section covers two co-designed viewer features that landed together:
the multi-scale composition tool and the run-record history that supports
both individual runs and sensitivity sweeps. Both live in `viewer.py`.

### 6.1 Motivation

The bulk composition fed into a thermodynamic backend determines every
prediction it makes (stable assemblage, modes, properties, P-T
estimates). In a heterogeneous thin section the *effective* bulk
composition depends on the spatial scale you average over: a small
window may capture a single phase pocket, a large window averages many
domains. The petrologically meaningful question — *at what spatial
scale is the rock effectively equilibrated?* — can only be answered by
looking at how the bulk and its predicted assemblage change with scale.

### 6.2 Quadtree subdivision

The active ROI is subdivided into a quadtree via two module-level
helpers:

- `_max_useful_quadtree_depth(bbox, min_window_px) -> int` — largest
  depth that keeps both window dimensions `≥ min_window_px`.
- `_quadtree_windows(bbox, depth) -> Dict[level, List[bbox]]` —
  recursive integer halving. Degenerate children (where halving would
  produce a window of zero or one pixel on a side) are dropped, so
  non-square ROIs may produce fewer than `4^k` windows at deep levels.

Two depth caps are computed per dialog instance:

- **EDS path**: floor = 4 px (per-pixel statistics are noisy below
  that).
- **SPX path**: floor = **median manifest tile short-side**, because
  the SPX path uses a 95% tile-overlap rule — a window smaller than a
  tile can never contain one. When this floor disqualifies all
  subdivisions (ROI smaller than one tile), the SPX radio in the
  settings dialog is disabled with an explanatory label.

The settings dialog (`_CompositionScaleDialog`) re-clamps the depth
spinner whenever the source toggles, so the user never sees a depth
that wouldn't produce useful windows for the chosen source.

### 6.3 Per-window composition

Two sources are supported, picked per run:

- **EDS net-count fractions** — `_net_count_fractions(band_maps,
  sub_bbox, sub_mask)` computes `net = main − 0.5·(sb_lo + sb_hi)`
  over a ROI-local sub-bbox with optional polygon-mask slice, then
  normalises to element fractions (0..1) which are stored as 0..100
  wt% in the run record. Pure-numpy on the in-memory `last_band_maps`
  — no disk IO after the initial ROI cache fetch.
- **SPX-quantified** — `_spx_composition_for_bbox(bbox,
  min_overlap=0.95)` factored from the original `on_spx_composition`.
  Iterates the manifest tile bboxes, includes any tile whose area is
  ≥95% covered by the window, and aggregates weighted by NetIntens.
  Returns `None` when no SPX-bearing tile is fully inside the window.
  Per-tile SPX parses are memoised in `self._spx_parse_cache`, so 85
  sub-window calls only pay the disk cost once per tile.

### 6.4 Per-level window cap + sampling

For each quadtree level the user can cap the number of windows
computed via a per-row spin box in the settings dialog. The cap is
applied **before** the worker starts (in `on_composition_scale`),
using one of two strategies selected via a combo box:

- **First N** — top-left scan order. Deterministic and easy to
  reason about.
- **Random N** — `np.random.default_rng(0).choice(...)` followed by a
  sort to keep scan-order rendering. The fixed seed makes the
  selection reproducible across runs.

### 6.5 Worker and result dialog

`_CompositionScaleWorker(QThread)` iterates the (possibly subsampled)
levels and emits `progress(done, total)` per window. EDS path: pure
numpy. SPX path: calls back into the panel via a small lambda.

When done, `_show_composition_scale_result` renders:

- **Distributions tab** — matplotlib box plots, one subplot per top-8
  element by full-ROI mean abundance; x-axis = quadtree levels.
- **Per-scale summary tab** — table with mean ± σ per element per
  level, plus n (usable windows) / n_total (offered windows).
- **Windows tab** — full per-window table including any "skipped"
  reason (mask coverage too low, no SPX tile overlap, etc.).
- Action buttons: `Run modelling sweep …`, `Export CSV …`, `Locate
  windows` (overlays a colour-coded Shapes layer on napari with one
  colour per quadtree level).

### 6.6 Modelling sensitivity sweep

`Run modelling sweep …` (from the result dialog) opens
`_ThermoRunDialog` seeded with the level-0 mean composition. The user
picks backend / P-T range / n_p × n_t / volatiles / Fe³⁺ as usual.
On OK, the code iterates **{mean, mean + σ, mean − σ} per usable
quadtree level**, clipping negatives to zero and renormalising each
candidate to 100 wt%. Each grid run is:

1. Converted via `Composition.from_element_wt(comp_el)` (and
   `to_oxide_wt` with H₂O/CO₂ extras if the user supplied any).
2. Run via `calc.run_grid(comp, p_range, t_range, n_p, n_t)`.
3. Stored as a run record (`_record_thermod_run`) with
   `comp_source = "sweep L{n} {rep} (eds|spx)"` and the parent ROI
   bbox/label.

For default depth=3, that's 4 levels × 3 reps = 12 grids per ROI.
Magemin (juliacall) runs sequentially on the main thread by design.
A `QProgressDialog` with Cancel surfaces per-grid progress.

### 6.7 Run records

Every `_show_thermod_single` and `_show_thermod_grid` call (whether
from `on_run_thermod`, the segmentation hand-off, or the sweep)
appends a dict to `self.last_thermod_runs`:

```
id, index, timestamp, mode ("single"|"grid"),
opts (dialog options),
comp_el, comp_source, manifest,
roi_label, roi_bbox, roi_shape ({vertices, type} | None),
result (single)  |  results (list, grid)
```

Round-trip serialisation lives in module-level helpers
(`_serialize_thermod_result`, `_serialize_phase_result` and their
deserialise counterparts) so the dict survives JSON and reconstructs
real `ThermoResult` / `PhaseResult` instances on load.

Result dialogs themselves carry:

- ROI label in the window title and header.
- A **Parameters** tab (single) or button (grid) showing a
  human-readable parameter block (`_thermod_parameters_text`).
- **Export bundle …** — writes a folder via
  `_export_thermod_bundle(run, out_dir, fig=None)` containing
  `parameters.json`, `summary.txt`, plus `phases.csv` +
  `phase_compositions.csv` + `bulk_properties.csv` (single) or
  `grid.csv` + `pseudosection.png` (grid).
- **Locate ROI** — adds (or refreshes) a magenta `Modelling ROIs`
  Shapes layer on napari and centres the camera.

### 6.8 Modelling-runs dialog + Compare

`Modelling runs …` (panel button) opens a table dialog with the run
history. Multi-select is enabled (Ctrl/Shift). Buttons: View
results / View parameters / Locate ROI / Compare selected … /
Export bundle / Delete.

`_show_compare_runs_dialog(grid_runs)` requires two or more grid
runs. Three tabs:

- **Summary** — per-run table: index, ROI label, backend, count of
  unique assemblages, mean of every shared bulk property.
- **Phase-field agreement** — for runs sharing the same P-T grid:
  per-cell fraction of selected runs whose stable-phase set equals
  run-0's. Rendered as an `RdYlGn` heatmap on the P-T grid. Disabled
  when the grids differ.
- **Property spread** — for runs sharing the same P-T grid: per-cell
  `np.nanstd` of any shared bulk property; combo selector switches
  property. Disabled when grids differ.

CSV export dumps mean/std of every shared property per run.

### 6.9 Session persistence

`on_save_session` writes `meta["thermod_runs"]` when the list is
non-empty. `_load_session` restores it. Saved sessions re-open the
result dialogs, Parameters viewer, and Compare dialog without a live
thermo backend, because all data lives in the JSON dict and the
deserialise helpers construct `ThermoResult` objects on demand.

---

## 7. Oxford AZtec CSV import

THERMOD supports two input vendors. Bruker BCF/SPX tiles go through
`stitch.py` (per-tile alignment + composite + cache build). Oxford
AZtec exports come **already montaged** — a single BSE TIF plus one
per-element Wt% CSV — and need a different on-ramp.

### 7.1 What an Oxford export looks like

Inspected on two real datasets (Kibo and Granulite):

- `BSE image.tif` — backscatter electron reference image, often
  LZW-compressed (7619 × 7559 px on Kibo). **Optional** — not all
  exports include it (Granulite ships without one).
- Element Wt% CSVs — one per element, 2D grid of float Wt% values.
  Two filename conventions are observed in the wild:
  - `Montaged Map Data-{Element} Wt%.csv` (Kibo-style, 2857 × 2835 px).
  - `{El}_{digits}.csv` (Granulite-style, 929 × 993 px,
    e.g. `Si_301582592023.csv`).
  Both are bare numeric matrices; NaN at the dataset edges marks the
  background mask.
- `*.piprj` — AZtec project archive (ZIP/XML). Holds pixel size in
  microns, beam voltage, stage origin. Not parsed yet by THERMOD.

There is **no per-tile structure** to stitch and **no spectrum cube**
— the maps are already integrated and quantified.

### 7.2 `oxford_loader.py`

A one-shot importer that converts the Oxford layout into the same
on-disk artefacts `stitch.py` produces, so `viewer.py` runs unmodified
on the converted dataset.

```
python oxford_loader.py --dir D:\UU\Kibo          # with BSE TIF
python oxford_loader.py --dir D:\UU\Granulite     # no BSE — synthesises mean-Z proxy
```

`discover_inputs` tries the Kibo-style glob first
(`Montaged Map Data-* Wt%.csv`) and falls back to the Granulite-style
short pattern (`{El}_{digits}.csv`, matched against `*.csv`). Two-letter
element symbols are normalised to `Xy` capitalisation.

If no BSE TIF matches the glob (`*BSE*.tif` by default), the loader
synthesises a per-pixel mean-Z proxy from the element maps:

```
bse_proxy[i, j] = Σ_el (Z_el · wt%_el[i, j])  /  Σ_el wt%_el[i, j]
```

The atomic-number table `_Z_BY_ELEMENT` in `oxford_loader.py` covers
H–U; elements without a Z entry contribute nothing. NaN values are
treated as zero contribution so the dataset-edge mask drops out
cleanly. BSE intensity tracks mean Z, so this is the closest visual
analogue available from quantified element maps — it is **not** a
calibrated BSE image. When the proxy path is active,
`target_resolution` is forced to `"elements"` (no real BSE shape
exists to upsample to).

writes, in-place:

```
D:\UU\Kibo\
    stitched_BSE_8bit.png            (percentile-clipped uint8)
    stitched_BSE_16bit.tiff          (uncompressed 16-bit)
    stitch_manifest.csv              (single row, mosaic-as-one-tile)
    eds_cache\
        manifest.json                data_type="oxford_wt_pct"
        weight.npy                   ones (every pixel covered once)
        {El}__main.npy               Oxford Wt% map
        {El}__sb_lo.npy              zeros (no sidebands)
        {El}__sb_hi.npy              zeros
```

The BSE and element-map grids have different native resolutions; the
loader resizes them to a single common grid:

- `--resolution elements` (default) — downsample BSE to the element
  size. Smaller cache, faster build.
- `--resolution bse` — upsample element maps to the BSE size. Larger
  cache, retains BSE detail.

Other flags: `--force` rebuilds even when the cache fingerprint
matches; `--pixel-um <um>` records pixel pitch; `--out <dir>` writes
elsewhere; `--bse-glob` / `--csv-glob` override the filename patterns.

LZW-compressed Oxford TIFFs are common; `imageio`/`tifffile` need the
optional `imagecodecs` package to decompress them. The loader has a
PIL fallback that handles LZW natively, so the import works on a
default scientific-Python install.

### 7.3 Cache provenance: `data_type`

`eds_cache/manifest.json` carries two new fields (used by both the
Bruker and Oxford writers):

- `data_type`:
  - `"bcf_net_counts"` — raw band sums; downstream computes
    `net = main − 0.5 · (sb_lo + sb_hi)`.
  - `"oxford_wt_pct"` — `main` is already Wt%; sidebands are zero;
    `net = main`.
- `source` — free-text provenance (e.g. `"Bruker BCF tiles"`,
  `"Oxford AZtec CSV export"`).

`eds_cache.is_valid_for` skips the `width_eV` tolerance check when
`data_type != "bcf_net_counts"` (Oxford caches have no real energy
window — the value is a placeholder). `status_for` and the CLI
`status` subcommand surface both fields.

### 7.4 Why the math survives intact for Oxford data

The viewer's EDS path computes `net = main − 0.5·(sb_lo + sb_hi)` per
pixel; the composition-vs-scale tool sums `net` over a window then
renormalises to fractions.

- For Bruker caches: net counts → element-fraction vector.
- For Oxford caches: `main = Wt%`, sidebands = 0 → `net = Wt%` →
  renormalised wt% fractions.

Both produce a normalised composition that
`Composition.from_element_wt` consumes correctly (the thermo lib
renormalises to 100 wt% internally). Only the *display label* differs
— `"Oxford Wt% (montaged map)"` vs `"EDS net-count"`.

NaN pixels at the dataset edge: the composition helper uses
`np.nansum`, silently treating them as zero — the correct behaviour.

### 7.5 Viewer adaptations for Oxford caches

Two small `viewer.py` changes (`_cache_data_type()` /
`_is_oxford_cache()` read the cache manifest):

1. `on_composition_scale` forces `has_spx = False` when the cache is
   Oxford-tagged. No per-tile SPX files exist for a montaged Oxford
   dataset; the SPX radio would silently fail otherwise.
2. The composition-vs-scale result dialog header and the modelling
   sweep's run-record `comp_source` label say
   `"Oxford Wt% (montaged map)"` rather than `"EDS net-count"`.

### 7.6 ROI clipping

`_get_roi_bbox_and_mask` clips the ROI bbox to the mosaic bounds
(`(H, W)` read from the BSE napari layer via `_data_bounds()` — handles
multiscale pyramids). Partly-outside ROIs are clipped silently with
a `[roi] clipped to data bounds: …` log line; fully-outside ROIs
raise `RuntimeError("ROI lies entirely outside the data bounds …")`
instead of producing the cryptic numpy/shape errors that previously
surfaced when slicing the cache memmap with out-of-range indices.

Polygon ROIs are rasterised onto the clipped bbox grid, so polygon
vertices that extend off-data naturally drop out of the resulting
mask.

This benefits both Bruker and Oxford workflows but was prompted by
Oxford datasets where the user often draws ROIs near the dataset
edge.

### 7.7 Out of scope (deferred)

- **PIPRJ metadata parsing** — auto-extract pixel size in microns,
  beam voltage, stage origin from `.piprj` archives. Currently the
  user passes `--pixel-um` manually.
- **Per-tile Oxford imports** — if a future export ships un-montaged
  tiles, that requires a real stitching pass. Defer until such data
  appears.
- **Two-resolution caches** — keeping BSE at native resolution and
  elements at coarser resolution in the same cache. For now we resize
  to one common grid.

---

## 8. Viewer module layout (tier-B refactor, 2026-05-12)

`viewer.py` was split into six files to bring the top-of-file
content down from 3 060 lines of helpers/dialogs/workers to a single
explicit re-import block. The main `EDSControlPanel` class and
`main()` stayed in `viewer.py`; everything else was extracted
mechanically (cut + paste, no logic changes). Every moved name is
re-exported from `viewer` so `viewer.<symbol>` continues to resolve
for any caller (none observed in tree, but the safety net costs
nothing).

```
viewer.py                  ~3670 lines — EDSControlPanel + main() + re-export block
viewer_helpers.py          ~390  lines — image/manifest/SPX/Perple_X primitives
viewer_thermo_helpers.py   ~390  lines — thermod display helpers, run-record
                                         serialise/deserialise, composition-vs-scale
                                         math, _assemblage_selector_dialog
viewer_segment_helpers.py  ~580  lines — phase-segmentation pipeline
                                         (smooth → features → PCA → ICL sweep →
                                         cleanup; cluster naming + napari layer
                                         constructors)
viewer_workers.py          ~280  lines — 4 QThread workers (_EDSWorker,
                                         _EDSCacheWorker, _ThermoGridWorker,
                                         _CompositionScaleWorker)
viewer_dialogs.py          ~1330 lines — 4 Qt dialogs (_ThermoRunDialog,
                                         _CompositionScaleDialog, _PhaseSegDialog,
                                         _PhaseClassDialog) + _hline()
```

Cross-module dependencies (verified non-circular):

- `viewer_workers` imports `_net_count_fractions` from
  `viewer_thermo_helpers`.
- `viewer_dialogs` imports `_max_useful_quadtree_depth` /
  `_quadtree_windows` from `viewer_thermo_helpers` and
  `_label_map_to_rgba` / `_remove_phase_layers` /
  `_add_phase_cluster_layers` from `viewer_segment_helpers`.
- `viewer.py` imports everything explicitly (no `import *`) and
  trusts the re-export to preserve the historical viewer-namespace
  surface.

Constraints to preserve when editing:

- The QThread workers reference `eds_cache` and `eds_map` lazily
  inside `run()` — keep those imports inside the method bodies so
  module import stays cheap.
- Dialogs do **not** call back into `EDSControlPanel`; the `parent`
  argument is only Qt's window-ownership pointer. New dialogs should
  follow this pattern (return values via `get_options()` / signals).
- `EDSControlPanel` is intentionally not split (tier C would mix-in
  feature-grouped method bundles). The shared `self.last_*` attrs
  make a mixin split require care; defer until you have a concrete
  reason.

A future tier-C split would carve `EDSControlPanel` into mixin classes
per feature group (`EDSPanelMixin_Thermod`, `_CompScale`, `_PhaseSeg`,
`_Session`). Estimated effort 1–2 days, mostly because the `self.last_*`
state is shared across all features and would need a clear ownership
model.
