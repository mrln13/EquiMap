"""Location of the C23-S1-A dataset for the scripts that read it.

Only the scripts that recompute the worked example need the dataset (the
Yoda data package: Bruker tiles, stitch manifest, EDS cache and the July viewer
session); every figure also regenerates from the staged files beside this
module without it.

Set the folder once, for all scripts, with the environment variable
EQUIMAP_DATASET (the folder that holds stitch_manifest.csv, the job folders,
eds_cache/ and viewer_session_22_07_26.json/.npz):

    set EQUIMAP_DATASET=E:\\data\\C23-S1-A          (Windows cmd)
    $env:EQUIMAP_DATASET = "E:\\data\\C23-S1-A"     (PowerShell)
    export EQUIMAP_DATASET=/data/C23-S1-A          (bash)

Scripts with a --dataset / --session-json / --session-npz option still accept
it and it wins over the variable.
"""
import os
from pathlib import Path

DEFAULT_DATASET = r"D:/Actief/C23S1A_Quant"   # the authors' folder

DATASET_DIR = Path(os.environ.get("EQUIMAP_DATASET") or DEFAULT_DATASET)
SESSION_JSON = DATASET_DIR / "viewer_session_22_07_26.json"
SESSION_NPZ = DATASET_DIR / "viewer_session_22_07_26.npz"
EXPORT_DIR = DATASET_DIR / "export"
