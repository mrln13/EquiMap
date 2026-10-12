# ebsd_validation.py
"""
EBSD-vs-EDS phase-map validation (confusion matrix).

The Oxford h5oina import provides a crystallographically indexed EBSD phase
map — ground truth. The EDS pipeline produces an independent phase map (GMM
segmentation + mineral ID). Cross-tabulating the two per pixel quantifies how
well the EDS-only workflow recovers the true phases: the manuscript's
validation figure, and a per-dataset sanity check in the viewer.

Rows are EBSD phases (truth), columns are EDS domains. Label matching between
the two vocabularies (EBSD "Diopside" vs EDS "Clinopyroxene"/"Cpx"/user-edited
names) reuses the alias-aware phase matcher built for the assemblage→P-T
inversion (``viewer_thermo_helpers._phase_matches``); one EBSD phase may match
several EDS domains (e.g. two GMM clusters that are both chlorite).

Qt-free and napari-free so it unit-tests headless and can drive the
manuscript figure from a script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def crop_ebsd_to_phase_grid(ebsd_map: np.ndarray,
                            roi: Tuple[int, int, int, int],
                            ds: int,
                            phase_shape: Tuple[int, int],
                            roi_mask_full: Optional[np.ndarray] = None,
                            ) -> Optional[np.ndarray]:
    """Crop + decimate the full-mosaic EBSD ordinal map onto the phase-map grid.

    Mirrors the alignment in ``viewer.on_use_ebsd_phases``: crop to the ROI
    bbox, zero (= unindex) pixels outside the full-resolution polygon mask,
    nearest-pixel decimate ``[::ds, ::ds]``, trim to ``phase_shape``. Returns
    None when the shapes don't line up (caller should treat as misalignment).
    """
    y0, y1, x0, x1 = roi
    ds = max(1, int(ds))
    crop = np.asarray(ebsd_map)[y0:y1, x0:x1]
    if roi_mask_full is not None and \
            np.asarray(roi_mask_full).shape == crop.shape:
        crop = np.where(np.asarray(roi_mask_full, dtype=bool), crop, 0)
    dec = np.ascontiguousarray(crop[::ds, ::ds])
    Hb, Wb = phase_shape
    dec = dec[:Hb, :Wb]
    if dec.shape != (Hb, Wb):
        return None
    return dec


@dataclass
class ConfusionResult:
    """Pixel-level contingency between EBSD truth and EDS phase domains."""
    counts: np.ndarray            # (R, C) int64 pixel counts
    row_labels: List[str]         # EBSD phase names (truth)
    col_labels: List[str]         # EDS domain names
    row_ordinals: List[int]       # EBSD ordinals per row
    col_cids: List[int]           # EDS cluster ids per column
    n_both: int                   # pixels indexed by EBSD AND in an EDS domain
    n_ebsd_only: int              # indexed by EBSD, no EDS domain (cid < 0)
    n_eds_only: int               # EDS domain, EBSD unindexed (ordinal 0)
    matched_cols: Dict[int, List[int]] = field(default_factory=dict)  # row idx -> col idxs
    accuracy: Optional[float] = None   # correct / n_both (alias-matched)
    per_row: List[dict] = field(default_factory=list)   # recall per EBSD phase
    per_col: List[dict] = field(default_factory=list)   # precision per EDS domain

    def row_normalised(self) -> np.ndarray:
        """Counts as row fractions (each EBSD phase sums to 1; 0-rows stay 0)."""
        c = self.counts.astype(float)
        sums = c.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(sums > 0, c / sums, 0.0)
        return out


def _match_labels(row_labels: List[str], col_labels: List[str],
                  ) -> Dict[int, List[int]]:
    """Alias-aware row→columns label matching ({row_idx: [col_idx, ...]})."""
    from viewer_thermo_helpers import _phase_matches
    out: Dict[int, List[int]] = {}
    for i, rl in enumerate(row_labels):
        hits = [j for j, cl in enumerate(col_labels)
                if _phase_matches(rl, cl) or _phase_matches(cl, rl)]
        if hits:
            out[i] = hits
    return out


def confusion_matrix(ebsd_crop: np.ndarray,
                     phase_map: np.ndarray,
                     ebsd_names: Dict[int, str],
                     eds_names: Dict[int, str],
                     ) -> ConfusionResult:
    """Cross-tabulate the EBSD ordinal map against the EDS phase label map.

    Both arrays must share a grid (use :func:`crop_ebsd_to_phase_grid`).
    EBSD ordinal 0 = unindexed and EDS cid < 0 = no domain; only pixels valid
    in *both* maps enter the matrix (the one-sided counts are reported).
    ``ebsd_names`` maps ordinals to phase names; ``eds_names`` maps cids to
    domain names. Accuracy counts a pixel correct when its EDS column is one
    of the alias-matched columns for its EBSD row — an EBSD phase with no
    matching EDS domain contributes only wrong pixels (the honest reading:
    the EDS workflow missed that phase).
    """
    eb = np.asarray(ebsd_crop)
    pm = np.asarray(phase_map)
    if eb.shape != pm.shape:
        raise ValueError(f"grid mismatch: EBSD {eb.shape} vs phase map {pm.shape}")

    both = (eb > 0) & (pm >= 0)
    n_both = int(both.sum())
    n_ebsd_only = int(((eb > 0) & (pm < 0)).sum())
    n_eds_only = int(((eb == 0) & (pm >= 0)).sum())

    ords = sorted(int(o) for o in np.unique(eb[eb > 0]))
    cids = sorted(int(c) for c in np.unique(pm[pm >= 0]))
    row_labels = [str((ebsd_names or {}).get(o, f"EBSD {o}")) for o in ords]
    col_labels = [str((eds_names or {}).get(c, f"Phase {c + 1}")) for c in cids]

    counts = np.zeros((len(ords), len(cids)), dtype=np.int64)
    if n_both and ords and cids:
        eb_b = eb[both].ravel()
        pm_b = pm[both].ravel()
        # vectorised bincount over the flattened (row, col) pair index;
        # values -> indices via searchsorted on the sorted uniques
        r = np.searchsorted(np.array(ords), eb_b)
        c = np.searchsorted(np.array(cids), pm_b)
        flat = r * len(cids) + c
        bc = np.bincount(flat, minlength=len(ords) * len(cids))
        counts = bc.reshape(len(ords), len(cids)).astype(np.int64)

    matched = _match_labels(row_labels, col_labels)

    accuracy = None
    per_row: List[dict] = []
    per_col: List[dict] = []
    if n_both:
        correct = 0
        for i in range(len(ords)):
            row_total = int(counts[i].sum())
            hits = matched.get(i, [])
            row_correct = int(counts[i, hits].sum()) if hits else 0
            correct += row_correct
            per_row.append({
                "label": row_labels[i],
                "n": row_total,
                "recall": (row_correct / row_total) if (row_total and hits) else
                          (None if not hits else 0.0),
                "matched": [col_labels[j] for j in hits],
            })
        for j in range(len(cids)):
            col_total = int(counts[:, j].sum())
            rows_hit = [i for i, hs in matched.items() if j in hs]
            col_correct = int(counts[rows_hit, j].sum()) if rows_hit else 0
            per_col.append({
                "label": col_labels[j],
                "n": col_total,
                "precision": (col_correct / col_total) if (col_total and rows_hit)
                             else (None if not rows_hit else 0.0),
                "matched": [row_labels[i] for i in rows_hit],
            })
        accuracy = correct / n_both

    return ConfusionResult(
        counts=counts, row_labels=row_labels, col_labels=col_labels,
        row_ordinals=ords, col_cids=cids,
        n_both=n_both, n_ebsd_only=n_ebsd_only, n_eds_only=n_eds_only,
        matched_cols=matched, accuracy=accuracy,
        per_row=per_row, per_col=per_col,
    )


def confusion_to_csv_rows(res: ConfusionResult) -> List[List[str]]:
    """Long+matrix CSV payload: header row, count rows, then summary lines."""
    rows: List[List[str]] = []
    rows.append(["EBSD \\ EDS"] + res.col_labels + ["row total"])
    for i, rl in enumerate(res.row_labels):
        rows.append([rl] + [str(int(v)) for v in res.counts[i]]
                    + [str(int(res.counts[i].sum()))])
    rows.append(["col total"] + [str(int(v)) for v in res.counts.sum(axis=0)]
                + [str(int(res.counts.sum()))])
    rows.append([])
    rows.append(["pixels both valid", str(res.n_both)])
    rows.append(["EBSD indexed only", str(res.n_ebsd_only)])
    rows.append(["EDS domain only", str(res.n_eds_only)])
    if res.accuracy is not None:
        rows.append(["alias-matched accuracy", f"{res.accuracy:.4f}"])
    for r in res.per_row:
        rows.append(["recall", r["label"],
                     "" if r["recall"] is None else f"{r['recall']:.4f}",
                     "matched: " + (", ".join(r["matched"]) or "(none)")])
    for c in res.per_col:
        rows.append(["precision", c["label"],
                     "" if c["precision"] is None else f"{c['precision']:.4f}",
                     "matched: " + (", ".join(c["matched"]) or "(none)")])
    return rows
