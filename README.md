# EquiMap

EquiMap is an open Python toolkit that takes raw SEM/EDS map tiles to a
pressure–temperature estimate: it stitches vendor tiles into a large-area
mosaic that keeps its spectra, segments the mosaic into phase domains,
suggests mineral names from a curated library, measures the length scale
over which a bulk composition is representative, and feeds the result to
pluggable thermodynamic backends (MAGEMin, Perple_X, pycalphad). It is
research software developed at the Department of Earth Sciences, Utrecht
University; a methods paper is in preparation for *Computers & Geosciences*.
The working name of the code base is THERMOD, which still appears in some
module and function names.

## What it does

1. **Mosaic stitching** (Bruker BCF tiles): stage-coordinate placement, phase
   cross-correlation, global least-squares refinement, feathered compositing.
2. **Spectrally aware EDS mosaic cache**: per-element band sums over the whole
   mosaic, so any region of interest is read by slicing instead of re-parsing
   tiles. Oxford AZtec exports (per-element wt% CSVs or the `.h5oina` HDF5
   with EBSD, EDS and electron image on one grid) are imported into the same
   cache layout, so everything downstream is vendor-agnostic.
3. **Empirical calibration** of uncalibrated Bruker net-count maps against the
   quantified SPX tiles of the same mosaic (no detector parameters needed),
   with hold-out validation.
4. **Phase segmentation and mineral identification**: Gaussian smoothing,
   PCA, Gaussian-mixture clustering with automatic cluster count (ICL),
   cleanup, and a library-based mineral suggestion per domain (ratio matching
   for all data, structural-formula recalculation for quantified data) with
   a manual override. An EBSD phase map, when present, gives a pixel-level
   confusion matrix against the EDS segmentation.
5. **Composition versus length scale**: quadtree subdivision of a region,
   semivariograms and Moran's I per element, and a thermodynamic sensitivity
   sweep that shows at which scale a bulk composition stops being
   representative.
6. **Thermodynamic modelling**: P–T grids from the segmented assemblage with
   modal-proportion misfit, Monte Carlo propagation of bulk uncertainty, run
   history, and an on-disk grid cache.

All of this is driven from a napari-based viewer (`viewer.py`); the stitching,
import and modelling steps also have command-line entry points.

## Installation

### Windows, no Python experience

The `installer/` folder contains a two-click installer that sets up a private
Miniforge, builds the `equimap` environment from `installer/environment.yml`,
fetches the Julia + MAGEMin engine and puts an EquiMap shortcut on the Desktop.
See [installer/INSTALL.md](installer/INSTALL.md).

### Developers (any platform)

Python 3.11 is required.

```bash
conda env create -f installer/environment.yml
conda activate equimap
```

or, in an existing environment, `pip install -r requirements.txt`. The MAGEMin
backend needs its Julia runtime once:

```bash
python -c "import petthermotools as ptt; ptt.install_MAGEMinCalc()"
```

The Perple_X backend is optional and needs the Perple_X 7.x executables from
ETH Zurich; point `thermo.backends.perplex.PerplexBackend(perplex_dir=...)` at
the install directory.

## Quick start

```bash
# Bruker: stitch a tile directory and build the EDS cache
python stitch.py --dir path/to/tiles --eds-preview all

# Oxford: convert an AZtec export (wt% CSVs or a .h5oina project) in place
python oxford_loader.py --dir path/to/export

# open the viewer on either result (prompts for a folder when --dir is omitted)
python viewer.py --dir path/to/dataset --preview-auto --elements auto

# headless modelling
python -m thermo run --backend magemin --database mp --p 0.5 --t 650 \
    --composition "SiO2:65,Al2O3:16,FeO:6,MgO:3,CaO:2,Na2O:3,K2O:4,H2O:4"
python -m thermo grid --backend magemin --database mp --p-range 0.2 1.0 \
    --t-range 500 800 --n-p 20 --n-t 20 --composition "..." --out grid.csv
```

Inside the viewer the workflow is: draw a region, *Compute* the element maps,
*Segment phases*, *Identify minerals*, then *Use for P–T estimate* or
*Composition vs scale*. Sessions (regions, maps, phase maps, run records) are
saved to JSON + NPZ and restored with `--session`.

## Repository layout

| Path | Contents |
|---|---|
| `stitch.py`, `stitch_CSV.py`, `grid_placement.py` | tile stitching |
| `eds_map.py`, `eds_cache.py`, `eds_background.py`, `eds_quantify.py` | EDS extraction, mosaic cache, line-aware background, SPX-anchored calibration |
| `oxford_loader.py` | Oxford AZtec CSV and `.h5oina` import |
| `viewer.py`, `viewer_*.py` | napari viewer: panel, helpers, segmentation pipeline, workers, dialogs |
| `mineral_id.py`, `ebsd_validation.py`, `zoning.py` | mineral library matching, EBSD confusion matrix, zoning maps and traverses |
| `thermo/` | backend-agnostic thermodynamic modelling package (`python -m thermo`) |
| `thermod.py` | legacy stand-alone Perple_X runner |
| `tests/` | pytest suite (`pytest -q tests`) |
| `installer/` | Windows installer, environment file, launcher |
| `docs/` | technical overview and stitching tunables |
| `paper/figures/` | scripts and staged data that regenerate the figures of the methods paper |

## Tests

```bash
pytest -q tests
```

The suite runs headless on a slim dependency set (no HyperSpy, napari or
Julia; those tests are skipped when the package is absent) and is executed by
GitHub Actions on Ubuntu and Windows.

## Reproducing the paper figures

`paper/figures/README.md` lists every figure, the script that draws it and the
staged data it reads. The scripts run from the repository root environment.
The SEM datasets themselves are not part of this repository; the staged
intermediate products (calibration validation, phase-map labels, P–T grids,
EBSD confusion data) are, except one 77 MB grid archive that exceeds GitHub's
file limit and is regenerated by `window_pt_run.py` or available on request.

## Citing

See [CITATION.cff](CITATION.cff). A methods paper describing EquiMap is in
preparation; until it is published please cite the repository.

## License

GPL-3.0, see [LICENSE](LICENSE). The licence follows from the core
dependencies HyperSpy (GPL-3.0) and MAGEMin (GPL-3.0).
