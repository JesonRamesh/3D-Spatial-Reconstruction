#!/usr/bin/env python3
"""
prune_floaters.py — Remove floater Gaussians from scene_aligned.ply.

Three-stage pruning pipeline:
  Step 1 — Opacity filter:    sigmoid(opacity) > OPACITY_THRESH
  Step 2 — Density filter:    ≥ MIN_NEIGHBOURS within radius DENSITY_RADIUS
  Step 3 — Bounding box clip: drop Gaussians outside room inner bbox

Pure numpy + scipy (no plyfile dependency).
Same PLY reader/writer pattern as realign_splat_v4.py.

Input:  outputs/splat_v4/scene_aligned.ply
Output: outputs/splat_v4/scene_pruned.ply
"""

import sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# ── Tunable thresholds ────────────────────────────────────────────────────────
OPACITY_THRESH   = 0.05   # sigmoid(opacity) > this  — 0.30 cuts the semi-transparent haze band
DENSITY_RADIUS   = 0.10   # metres — neighbour search radius
MIN_NEIGHBOURS   = 5      # minimum neighbours within radius to keep
KD_MAX_SAMPLE    = 200_000  # subsample for KD-tree construction

# Bounding box clip (inner room box, tight to avoid wall floaters)
# From PLAN.md: full bbox ≈ X[-6.5,6.5] Y[-6.7,6.3] Z[-6.5,6.9]
BBOX_X = (-5.0,  5.0)
BBOX_Y = (-5.5,  4.5)
BBOX_Z = (-5.0,  5.5)

INPUT_PLY  = "outputs/splat_v4/scene_aligned.ply"
OUTPUT_PLY = "outputs/splat_v4/scene_pruned.ply"

# ──────────────────────────────────────────────────────────────────────────────
# PLY I/O  (pure numpy, no plyfile dependency)
# ──────────────────────────────────────────────────────────────────────────────

def read_ply(path: str):
    """Return (header_lines, dtype, data_array, n_verts)."""
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            raw = f.readline()
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        binary_data = f.read()

    prop_type_map = {
        "float":   np.float32, "float32": np.float32,
        "double":  np.float64, "float64": np.float64,
        "int":     np.int32,   "int32":   np.int32,
        "uint":    np.uint32,  "uint32":  np.uint32,
        "uchar":   np.uint8,   "uint8":   np.uint8,
        "char":    np.int8,    "int8":    np.int8,
        "short":   np.int16,   "int16":   np.int16,
        "ushort":  np.uint16,  "uint16":  np.uint16,
    }
    n_verts = 0
    properties = []
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property":
            properties.append((parts[2], prop_type_map[parts[1]]))

    dtype = np.dtype([(name, t) for name, t in properties])
    data  = np.frombuffer(binary_data, dtype=dtype, count=n_verts).copy()
    return header_lines, dtype, data, n_verts


def write_ply(path: str, header_lines: list, data: np.ndarray):
    """Write binary little-endian PLY, preserving original header structure."""
    path = Path(path)
    header_out = []
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            header_out.append(f"element vertex {len(data)}")
        else:
            header_out.append(line)
    header_str = "\n".join(header_out) + "\n"
    with open(path, "wb") as f:
        f.write(header_str.encode("ascii"))
        f.write(data.tobytes())
    print(f"  Saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("prune_floaters.py — opacity + density + bbox pruning")
    print("=" * 62)

    # ── 0. Load ───────────────────────────────────────────────────────────────
    print(f"\n[0/4] Loading {INPUT_PLY} ...")
    if not Path(INPUT_PLY).exists():
        sys.exit(f"ERROR: {INPUT_PLY} not found. Run realign_splat_v4.py first.")

    header_lines, dtype, data, n = read_ply(INPUT_PLY)
    print(f"      Original: {n:,} Gaussians, {len(dtype.names)} properties")

    xyz = np.stack([
        data['x'].astype(np.float32),
        data['y'].astype(np.float32),
        data['z'].astype(np.float32),
    ], axis=1)  # (N, 3)

    opacity_raw = data['opacity'].astype(np.float64)
    opacity_sig = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid → [0,1]

    # ── 1. Opacity filter ─────────────────────────────────────────────────────
    print(f"\n[1/4] Opacity filter  (sigmoid > threshold)")
    print(f"      Opacity distribution:")
    print(f"        min={opacity_sig.min():.4f}  "
          f"p10={np.percentile(opacity_sig, 10):.4f}  "
          f"p50={np.percentile(opacity_sig, 50):.4f}  "
          f"p90={np.percentile(opacity_sig, 90):.4f}  "
          f"max={opacity_sig.max():.4f}")

    print(f"\n      Survival count at each threshold:")
    for thresh in [0.05, 0.10, 0.15, 0.20]:
        k = (opacity_sig > thresh).sum()
        print(f"        > {thresh:.2f} : {k:>10,}  ({100*k/n:.1f}%)")

    mask_opacity = opacity_sig > OPACITY_THRESH
    n_after_opacity = mask_opacity.sum()
    print(f"\n      ✓ Committed threshold: {OPACITY_THRESH}  →  "
          f"{n_after_opacity:,} kept  ({100*n_after_opacity/n:.1f}%  removed "
          f"{n - n_after_opacity:,})")

    # ── 2. Density filter ─────────────────────────────────────────────────────
    print(f"\n[2/4] Density filter  "
          f"(≥{MIN_NEIGHBOURS} neighbours within r={DENSITY_RADIUS}m)")

    xyz_post_op = xyz[mask_opacity]
    n_post_op   = len(xyz_post_op)
    print(f"      Working on {n_post_op:,} post-opacity Gaussians ...")

    # Build KD-tree on ALL post-opacity points (tree construction is fast;
    # the bottleneck is the query, which we batch).  Using a subsample for
    # tree construction would miss isolated floaters that fall in sparse regions.
    print(f"      Building cKDTree on all {n_post_op:,} pts ...")
    tree = cKDTree(xyz_post_op)

    # Query in batches to keep memory reasonable
    BATCH = 200_000
    counts = np.zeros(n_post_op, dtype=np.int32)
    print(f"      Querying {n_post_op:,} points in batches of {BATCH:,} ...")
    for start in range(0, n_post_op, BATCH):
        end   = min(start + BATCH, n_post_op)
        batch = xyz_post_op[start:end]
        counts[start:end] = tree.query_ball_point(
            batch, r=DENSITY_RADIUS, return_length=True
        )
        pct_done = 100 * end / n_post_op
        print(f"        ... {end:,} / {n_post_op:,}  ({pct_done:.0f}%)")

    mask_density = counts >= MIN_NEIGHBOURS
    n_after_density = mask_density.sum()
    print(f"      ✓ Density filter  →  {n_after_density:,} kept  "
          f"({100*n_after_density/n_post_op:.1f}%  removed "
          f"{n_post_op - n_after_density:,})")

    # Compose masks back to full-array index
    # mask_opacity: (N,)  bool
    # mask_density: (n_post_op,) bool — indexes into post-opacity array
    # Combined mask over original N:
    combined = np.zeros(n, dtype=bool)
    op_indices = np.where(mask_opacity)[0]
    combined[op_indices[mask_density]] = True
    n_combined = combined.sum()
    print(f"      After opacity+density: {n_combined:,} Gaussians")

    # ── 3. Bounding box clip ──────────────────────────────────────────────────
    print(f"\n[3/4] Bounding box clip")
    print(f"      Clip box: "
          f"X{list(BBOX_X)}  Y{list(BBOX_Y)}  Z{list(BBOX_Z)}")

    # Report actual bbox of combined set for reference
    xyz_comb = xyz[combined]
    for i, ax in enumerate("XYZ"):
        print(f"      Current {ax} range: "
              f"[{xyz_comb[:,i].min():.2f}, {xyz_comb[:,i].max():.2f}]")

    mask_bbox = (
        (xyz[:, 0] >= BBOX_X[0]) & (xyz[:, 0] <= BBOX_X[1]) &
        (xyz[:, 1] >= BBOX_Y[0]) & (xyz[:, 1] <= BBOX_Y[1]) &
        (xyz[:, 2] >= BBOX_Z[0]) & (xyz[:, 2] <= BBOX_Z[1])
    )
    final_mask = combined & mask_bbox
    n_final = final_mask.sum()
    n_removed_bbox = n_combined - n_final
    print(f"      ✓ Bbox clip  →  {n_final:,} kept  "
          f"(removed {n_removed_bbox:,} outside-box Gaussians)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  Pruning summary:")
    print(f"    Original:             {n:>10,}")
    print(f"    After opacity  ({OPACITY_THRESH:.2f}): "
          f"{n_after_opacity:>10,}  ({100*(n-n_after_opacity)/n:.1f}% removed)")
    print(f"    After density  ({MIN_NEIGHBOURS:>2}nb): "
          f"{n_combined:>10,}  ({100*(n_after_opacity-n_combined)/n_after_opacity:.1f}% removed)")
    print(f"    After bbox clip:      {n_final:>10,}  "
          f"({100*n_removed_bbox/n_combined:.1f}% removed)")
    print(f"    Total reduction:      {100*(n-n_final)/n:.1f}% of original removed")
    print(f"{'─'*62}")

    # ── 4. Write output ───────────────────────────────────────────────────────
    print(f"\n[4/4] Writing {OUTPUT_PLY} ...")
    out = data[final_mask].copy()
    write_ply(OUTPUT_PLY, header_lines, out)

    print(f"\n✅  Done!")
    print(f"    Input : {INPUT_PLY}  ({Path(INPUT_PLY).stat().st_size/1e6:.0f} MB)")
    print(f"    Output: {OUTPUT_PLY}")
    print(f"    Gaussians: {n:,} → {n_final:,}  "
          f"({100*n_final/n:.1f}% retained)")
    print()
    print("Next step:")
    print("  python3 scripts/convert_to_splat.py \\")
    print("    --input  outputs/splat_v4/scene_pruned.ply \\")
    print("    --output outputs/splat_v4/scene_pruned.splat")


if __name__ == "__main__":
    main()