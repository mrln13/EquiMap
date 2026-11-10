# -*- coding: utf-8 -*-
"""Seed robustness of the native-resolution nodule segmentation (Fig 4b).

Runs fig_segmentation_inset.py on the staged window (explicit --window, the
one recorded in phase_map_inset.json) for several random seeds and component
counts, and records whether a local cluster identifies as Al2SiO5 and its
share of the window. Supports the section-4.2 sentence "separates the cores
with five of six seeds, and with all six at seven or more components".

Outputs: per-run phase maps under <repo>/_scratch/inset_seeds/ (not staged)
and inset_seed_sweep.txt beside this script (staged).

Usage:  python inset_seed_sweep.py [--dataset <stitched_dir>]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from _dataset import DATASET_DIR   # dataset folder: env EQUIMAP_DATASET

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUNS = [(s, k) for k in (6, 7) for s in (42, 1, 2, 3, 4, 5)] + [(5, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--out-root", default=str(REPO / "_scratch" / "inset_seeds"))
    args = ap.parse_args()
    meta = json.loads((HERE / "phase_map_inset.json").read_text())
    y0, _y1, x0, _x1 = meta["window_mosaic_px"]
    legend = {int(r.split(",")[0]): r.split(",")[1].strip()
              for r in (HERE / "phase_legend.csv").read_text().splitlines()[1:]}
    sill_global = {i for i, n in legend.items() if n == "Sillimanite"}
    lines = []
    for seed, k in RUNS:
        od = Path(args.out_root) / f"seed_{seed}_k{k}"
        od.mkdir(parents=True, exist_ok=True)
        log = od / "run.log"
        cmd = [sys.executable, str(HERE / "fig_segmentation_inset.py"), "--dataset", args.dataset,
               "--window", str(y0), str(x0), "--seed", str(seed), "--k", str(k), "--out-dir", str(od)]
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        txt = log.read_text(encoding="utf-8", errors="replace")
        if rc != 0:
            lines.append(f"seed {seed:2d} k={k}: FAILED (see {log})")
            print(lines[-1], flush=True)
            continue
        run = json.loads((od / "phase_map_inset.json").read_text())
        lab = np.load(od / "phase_map_inset.npz")["labels"]
        extra = {int(i): n for i, n in run.get("extra_names", {}).items()}
        sill_ids = sill_global | {i for i, n in extra.items() if n == "Sillimanite"}
        n_loc = sum(1 for g in run["cluster_mapping"].values() if int(g) in sill_ids)
        valid = lab >= 0
        frac = float(np.isin(lab, list(sill_ids))[valid].mean()) if valid.any() else float("nan")
        ids = re.search(r"local cluster mineral ID: (.*)", txt)
        lines.append(f"seed {seed:2d} k={k}: Al2SiO5 cluster {'yes' if n_loc else 'no '} "
                     f"({100 * frac:.2f} % of window); IDs {ids.group(1) if ids else '?'}")
        print(lines[-1], flush=True)
    (HERE / "inset_seed_sweep.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
