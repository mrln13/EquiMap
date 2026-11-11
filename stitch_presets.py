# stitch_presets.py
"""Alignment tunables of stitch.py: base values, presets and their descriptions.

One source for stitch.py (which applies them) and the viewer's stitching
window (which shows them), so the values a user is told about are the values
that run. No heavy imports: the viewer reads this without loading HyperSpy.

A preset overrides some of the base values. "default" is what stitch.py has
always applied: preset C followed by the overrides for C23-S1-A-class data;
it produced the paper's mosaic. "custom" is the base values alone.
"""
from __future__ import annotations

from typing import Dict

# ---- base values (edit here; stitch.py reads them) -------------------------
BASE_TUNABLES: Dict[str, object] = dict(
    OVERLAP_FRAC=0.10,      # Expected fractional overlap between adjacent tiles.
    BAND_FRAC=0.08,         # Fraction of tile width/height used as strip for cross-correlation.
    N_SLICES=9,             # Number of slices across the overlap to sample for local drift.
    UPSAMPLE=20,            # Subpixel upsampling factor for phase correlation.
    MAX_STRIP_DRIFT=15,     # Reject slice-level estimates if shift is higher than this (px).
    MAD_Z=2.5,              # Z-score cutoff for rejecting outlier strip shifts.
    REFINE_ITERS=7,         # Number of overlap-refinement phases.
    REFINE_MAX_SHIFT=15,    # Limit for overlap-based correction in refinement stage (px).
    PYR_LEVELS=4,           # Pyramid depth for multi-scale correlation.
    TUKEY_ALPHA=0.08,       # Feather width for Tukey window blending.
    LOW_PCT=0.5,            # Percentile range used for final intensity scaling (low).
    HI_PCT=99.5,            # Percentile range used for final intensity scaling (high).
    MAX_VERT_X_DRIFT=5,     # Allowed |dX| for DOWN neighbours (px).
    MAX_RIGHT_Y_DRIFT=10,   # Allowed |dY| for RIGHT neighbours (px).
    SHEAR_CAP=5e-4,         # Max shear per tile allowed from slope fitting.
    SHEAR_ENABLE=True,      # Toggle shear correction.
    RIGHT_DY_WEIGHT=0.10,   # Weight for softly enforcing RIGHT dY = 0 (prevents staircase artefacts).
    VSTEP_ANCHOR_W=0.50,    # Weak vertical step anchor to stop row drift.
    anchor_sigma=60,        # Pull of tiles towards the stage-derived grid (px); smaller = stronger.
)

PRESETS: Dict[str, Dict[str, object]] = {
    "A": dict(  # Straighter grid
        RIGHT_DY_WEIGHT=0.00, VSTEP_ANCHOR_W=0.90, MAX_VERT_X_DRIFT=3.0,
        MAX_RIGHT_Y_DRIFT=6.0, BAND_FRAC=0.05, N_SLICES=9, UPSAMPLE=15,
        REFINE_ITERS=3, REFINE_MAX_SHIFT=12, PYR_LEVELS=4, anchor_sigma=40.0),
    "B": dict(  # Tighter correlations
        BAND_FRAC=0.08, N_SLICES=15, MAX_STRIP_DRIFT=10, UPSAMPLE=10,
        REFINE_ITERS=2, REFINE_MAX_SHIFT=8, PYR_LEVELS=3, RIGHT_DY_WEIGHT=0.10,
        VSTEP_ANCHOR_W=0.50, anchor_sigma=60.0),
    "C": dict(  # Large local errors / large grids
        REFINE_ITERS=7, REFINE_MAX_SHIFT=30, PYR_LEVELS=4, N_SLICES=13,
        UPSAMPLE=20, RIGHT_DY_WEIGHT=0.05, VSTEP_ANCHOR_W=0.35, anchor_sigma=220.0),
}
# "default" = C + these (C23S1A_Quant-class data: fine-grained BSE, fairly
# uniform texture, well-calibrated stage). Strong anchor + small refine
# budget; stitch.py's tight-start guard skips refinement entirely when the
# initial solve is already clean.
FINE_GRAINED_OVERRIDES: Dict[str, object] = dict(
    REFINE_ITERS=2, REFINE_MAX_SHIFT=8, RIGHT_DY_WEIGHT=0.10,
    VSTEP_ANCHOR_W=0.60, anchor_sigma=25.0)

PRESET_CHOICES = ("default", "A", "B", "C", "custom")

# What each preset does, in terms of the values it sets (see PRESET_TABLE_KEYS),
# and when to use it.
PRESET_HELP: Dict[str, str] = {
    "default": "Pulls the tiles strongly onto their stage positions (anchor 25 px) and "
               "refines little (2 passes, at most 8 px), with 13 correlation slices per "
               "overlap. Tuned on C23-S1-A (fine-grained BSE, well-calibrated stage) and "
               "used for the paper's mosaic. Start here.",
    "A": "Straighter grid: the strongest row-step anchor (0.9) and the tightest "
         "limits on how far neighbouring tiles may shift sideways (3 px) or "
         "vertically (6 px); those hard limits replace the soft row-straightness "
         "term, which is off (0). Pull onto stage positions 40 px; narrower "
         "correlation strips (5 % of the tile). For a mosaic whose rows or columns "
         "come out wavy.",
    "B": "Tighter correlations: more slices per overlap (15), a stricter cut on "
         "outlying slice shifts (10 px), coarser sub-pixel upsampling (10) and a "
         "shallower pyramid (3 levels); little refinement (2 passes, at most 8 px). "
         "For overlaps with clear texture where the default alignment is noisy.",
    "C": "Large grids or large local errors: tiles may move far from their stage "
         "positions (anchor 220 px) and refinement is long (7 passes, up to 30 px). "
         "For big mosaics whose stage positions drift.",
    "custom": "The base values in stitch_presets.py without any preset; for editing "
              "that file by hand.",
}

# The values the stitching window shows per preset, with a plain label.
PRESET_TABLE_KEYS = (
    ("anchor_sigma", "Pull onto stage positions (px; smaller = stronger)"),
    ("VSTEP_ANCHOR_W", "Row-step anchor weight"),
    ("RIGHT_DY_WEIGHT", "Row-straightness weight"),
    ("REFINE_ITERS", "Refinement passes"),
    ("REFINE_MAX_SHIFT", "Largest refinement shift (px)"),
    ("N_SLICES", "Correlation slices per overlap"),
    ("BAND_FRAC", "Correlation strip width (fraction of tile)"),
    ("MAX_STRIP_DRIFT", "Reject slice shifts above (px)"),
    ("UPSAMPLE", "Sub-pixel upsampling"),
    ("PYR_LEVELS", "Pyramid levels"),
    ("MAX_VERT_X_DRIFT", "Largest sideways shift between rows (px)"),
    ("MAX_RIGHT_Y_DRIFT", "Largest vertical shift between columns (px)"),
)


def resolve_preset(name: str) -> Dict[str, object]:
    """All tunables in effect for preset ``name`` (base values plus overrides)."""
    if name not in PRESET_CHOICES:
        raise ValueError(f"unknown preset {name!r}; choose from {PRESET_CHOICES}")
    out = dict(BASE_TUNABLES)
    if name in PRESETS:
        out.update(PRESETS[name])
    elif name == "default":
        out.update(PRESETS["C"])
        out.update(FINE_GRAINED_OVERRIDES)
    return out
