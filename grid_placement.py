# stage_grid_preview.py
import numpy as np
import hyperspy.api as hs
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------
# Get stage coordinates
# -------------------------
def get_stage_coordinates(p: Path):
    """
    Return (stage_x, stage_y) from BCF metadata.
    Falls back to (0.0,0.0) if unavailable.
    """
    try:
        s = hs.load(p)[2]
    except Exception:
        return 0.0, 0.0
    md = getattr(s, "metadata", None)
    if md is None:
        return 0.0, 0.0
    try:
        sx = float(md.Acquisition_instrument.SEM.Stage.X)
        sy = float(md.Acquisition_instrument.SEM.Stage.Y)
        return sx, sy
    except Exception:
        pass
    try:
        sx = float(md.get_item("Acquisition_instrument.SEM.Stage.x"))
        sy = float(md.get_item("Acquisition_instrument.SEM.Stage.y"))
        return sx, sy
    except Exception:
        pass
    return 0.0, 0.0

# -------------------------
# Assign tiles to grid
# -------------------------
def assign_grid_from_stage(files):
    stage_coords = {}
    for p in files:
        sx, sy = get_stage_coordinates(p)
        stage_coords[p] = (sx, sy)

    # Flip X axis to match image convention (SEM stage often has X increasing to left)
    xs = np.array([-sx for sx, sy in stage_coords.values()])
    ys = np.array([ sy for sx, sy in stage_coords.values()])

    # Estimate nominal stage steps
    def robust_step(vals):
        diffs = []
        for v in np.unique(vals):
            neigh = np.min(np.abs(vals - v)[np.abs(vals - v) > 1e-6], initial=np.inf)
            if np.isfinite(neigh): diffs.append(neigh)
        return np.median(diffs) if diffs else 1.0

    step_x_stage = robust_step(xs)
    step_y_stage = robust_step(ys)

    print(f"[stage grid] step_x ≈ {step_x_stage:.2f}, step_y ≈ {step_y_stage:.2f}")

    min_x, min_y = xs.min(), ys.min()

    grid = {}
    for p, (sx, sy) in stage_coords.items():
        gx = -sx   # flipped
        gy = sy
        col = int(round((gx - min_x) / step_x_stage))
        row = int(round((gy - min_y) / step_y_stage))
        grid[p] = (row, col)

    return grid

# -------------------------
# Build coarse mosaic
# -------------------------
def make_stage_mosaic(files, outdir="stage_grid_test"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = assign_grid_from_stage(files)

    # Load one tile to get dimensions
    sample = hs.load(files[0])[0].data.astype(np.float32)
    th, tw = sample.shape

    rows = [rc[0] for rc in grid.values()]
    cols = [rc[1] for rc in grid.values()]
    max_r, max_c = max(rows), max(cols)

    H = (max_r+1) * th
    W = (max_c+1) * tw

    mosaic = np.zeros((H, W), dtype=np.float32)

    for p in files:
        r, c = grid[p]
        tile = hs.load(p)[0].data.astype(np.float32)
        y0, x0 = r*th, c*tw
        mosaic[y0:y0+th, x0:x0+tw] = tile

    # Normalize for viewing
    lo, hi = np.percentile(mosaic, [0.5, 99.5])
    m8 = np.clip((mosaic - lo) / max(hi-lo,1e-6), 0, 1)
    m8 = (m8*255).astype(np.uint8)

    Image.fromarray(m8).save(outdir / "stage_grid_preview.png")
    plt.imshow(m8, cmap="gray")
    plt.title("Stage-grid placement (no refinement)")
    plt.show()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    directory = Path("Subset")   # adjust
    bcf_files = list(directory.rglob("*.bcf"))
    print(f"Found {len(bcf_files)} .bcf files.")
    make_stage_mosaic(bcf_files)
