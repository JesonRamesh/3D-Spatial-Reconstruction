"""
clean_splat_visibility.py — Multi-criterion floater removal for Gaussian splats.

Implements a post-processing approximation of the TIDI-GS paper (arXiv:2601.09291)
using two geometry-aware signals that work on an existing .ply without retraining:

  Signal 1 — Multi-view visibility filter
    Project every Gaussian into every COLMAP camera. Count how many cameras see it
    (inside frustum + above alpha threshold). Real room surfaces are visible from
    many cameras (the room was swept multiple times). Floaters outside the room are
    only visible from 1-3 cameras before exiting the frustum.
    → Remove Gaussians visible in fewer than --min_views cameras.

  Signal 2 — Convex hull crop
    Compute the convex hull of all COLMAP camera positions. Inflate each face outward
    by --hull_padding metres. Remove Gaussians outside the inflated hull.
    This is geometry-aware (follows actual room shape) unlike a bbox crop which cuts
    the bed because the room is not axis-aligned.

  Signal 3 — Alpha threshold (last, conservative)
    Remove near-invisible Gaussians (sigmoid(opacity) < --alpha_min).

Why this beats previous methods:
  - SOR: only uses spatial isolation (single criterion) → fails on semi-dense clusters
  - Alpha threshold: single criterion → removes real geometry in sparse-coverage areas
  - Bbox crop: axis-aligned box → clips bed wall which extends diagonally
  - This script: 3 independent signals, a Gaussian survives if ANY signal protects it

Usage:
  python3 scripts/clean_splat_visibility.py \\
    --input   outputs/splat_v3/scene.ply \\
    --output  outputs/splat_v3/scene_clean.ply \\
    --transforms data/colmap_v3/transforms.json \\
    --min_views 3 \\
    --hull_padding 1.5 \\
    --alpha_min 0.005

  # Then convert to .splat for viewer:
  python3 scripts/convert_to_splat.py \\
    --input  outputs/splat_v3/scene_clean.ply \\
    --output outputs/splat_v3/scene_clean.splat

Tuning guide:
  If bed wall / thin structures missing  → lower --min_views to 2
  If floaters still visible              → raise --min_views to 5
  If corners clipped                     → raise --hull_padding to 2.0
  If external haze remains               → lower --hull_padding to 1.2
  Target Gaussian count: 1.5M - 2.5M (from 3.74M raw)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull


# ── Helpers ───────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_ply(path: Path):
    """Load Gaussian PLY, return vertex record array and position/opacity arrays."""
    print(f"Loading PLY: {path}  ({path.stat().st_size / 1024**2:.1f} MB)")
    ply  = PlyData.read(str(path))
    v    = ply['vertex']
    pos  = np.stack([np.array(v['x']), np.array(v['y']), np.array(v['z'])], axis=1).astype(np.float32)
    ops  = sigmoid(np.array(v['opacity'], dtype=np.float32))
    print(f"  Gaussians: {len(pos):,}")
    return ply, v, pos, ops


def load_cameras(transforms_path: Path):
    """
    Load COLMAP camera poses from nerfstudio transforms.json.
    Returns list of dicts with keys: R (3x3), t (3,), fx, fy, cx, cy, w, h
    These are c2w matrices — we invert to get w2c for projection.
    """
    with open(transforms_path) as f:
        d = json.load(f)

    fx = float(d.get('fl_x', d.get('fx', 800)))
    fy = float(d.get('fl_y', d.get('fy', fx)))
    cx = float(d.get('cx', d.get('w', 1920) / 2))
    cy = float(d.get('cy', d.get('h', 1080) / 2))
    w  = int(d.get('w', 1920))
    h  = int(d.get('h', 1080))

    cameras = []
    for fr in d['frames']:
        c2w = np.array(fr['transform_matrix'], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        # Invert c2w → w2c
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3,  3]
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        cameras.append(dict(R=R_w2c, t=t_w2c,
                            fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h,
                            pos=t_c2w.copy()))   # camera position in world space
    print(f"  Loaded {len(cameras)} cameras from {transforms_path.name}")
    return cameras


# ── Signal 1: Multi-view visibility filter ────────────────────────────────────

def compute_visibility(pos: np.ndarray, ops: np.ndarray,
                       cameras: list, alpha_min: float,
                       batch_size: int = 50000) -> np.ndarray:
    """
    For each Gaussian, count how many cameras it is visible from:
      - Inside the image frustum (projected u,v within [0,w] x [0,h])
      - In front of the camera (depth > 0)
      - Above alpha threshold

    Returns int16 array of view counts per Gaussian.
    """
    n = len(pos)
    vis_count = np.zeros(n, dtype=np.int16)

    # Only consider Gaussians above alpha threshold for visibility
    # (transparent ones are never "visible" in any meaningful sense)
    alpha_mask = ops >= alpha_min

    print(f"  Computing visibility for {alpha_mask.sum():,} Gaussians "
          f"({100*alpha_mask.mean():.1f}% above alpha={alpha_min}) "
          f"across {len(cameras)} cameras...")
    print(f"  (Transparent Gaussians below alpha={alpha_min} will be removed "
          f"directly)")

    # Work in batches to stay within memory
    pos_valid = pos[alpha_mask]   # (M, 3)
    vis_valid = np.zeros(len(pos_valid), dtype=np.int16)

    n_cams = len(cameras)
    report_every = max(1, n_cams // 10)

    for ci, cam in enumerate(cameras):
        if ci % report_every == 0:
            pct = 100 * ci / n_cams
            print(f"    Camera {ci}/{n_cams} ({pct:.0f}%)...")

        R = cam['R'].astype(np.float32)   # (3,3) w2c rotation
        t = cam['t'].astype(np.float32)   # (3,)  w2c translation
        fx, fy = cam['fx'], cam['fy']
        cx, cy = cam['cx'], cam['cy']
        W,  H  = cam['w'],  cam['h']

        # Process in batches to avoid large intermediate arrays
        for start in range(0, len(pos_valid), batch_size):
            end  = min(start + batch_size, len(pos_valid))
            pts  = pos_valid[start:end]           # (B, 3)

            # Transform to camera space: p_cam = R @ p_world + t
            p_cam = pts @ R.T + t                 # (B, 3)

            # Keep only points in front of camera
            depth = p_cam[:, 2]
            front = depth > 0.05

            # Project to image plane
            u = fx * p_cam[:, 0] / np.where(depth > 1e-6, depth, 1e-6) + cx
            v = fy * p_cam[:, 1] / np.where(depth > 1e-6, depth, 1e-6) + cy

            # Inside image bounds (with small margin)
            margin = 10
            in_frame = (front &
                        (u >= margin) & (u < W - margin) &
                        (v >= margin) & (v < H - margin))

            vis_valid[start:end] += in_frame.astype(np.int16)

    # Map back to full array
    vis_count[alpha_mask] = vis_valid
    # Gaussians below alpha get vis_count=0 → will be removed by alpha filter anyway

    return vis_count


# ── Signal 2: Convex hull crop ────────────────────────────────────────────────

def build_inflated_hull(cameras: list, padding: float):
    """
    Build a convex hull from camera positions and inflate each face outward
    by `padding` units. Returns (hull_equations_inflated) — array of shape (F, 4)
    where each row is [nx, ny, nz, d] and a point p is INSIDE if
    hull_equations @ [*p, 1] <= 0 for all faces.
    """
    cam_positions = np.array([c['pos'] for c in cameras], dtype=np.float64)
    print(f"  Building convex hull from {len(cam_positions)} camera positions...")

    hull = ConvexHull(cam_positions)
    equations = hull.equations.copy()   # (F, 4): [nx, ny, nz, d]

    # Inflate: move each face outward by padding.
    # The plane equation is nx*x + ny*y + nz*z + d <= 0 (inside).
    # Moving the plane outward by `padding` means d → d - padding
    # (since the normal points outward, subtracting padding loosens the constraint).
    equations[:, 3] -= padding

    n_faces = len(equations)
    print(f"  Hull has {n_faces} faces, inflated by {padding}m")
    return equations


def apply_hull_filter(pos: np.ndarray, hull_equations: np.ndarray) -> np.ndarray:
    """
    Return boolean mask: True if point is INSIDE the inflated convex hull.
    Point p is inside if hull_equations @ [*p, 1] <= 0 for ALL faces.
    """
    # pos: (N, 3), hull_equations: (F, 4)
    # Extend pos with ones: (N, 4)
    pos_h = np.hstack([pos, np.ones((len(pos), 1), dtype=pos.dtype)])  # (N, 4)

    # scores[i, f] = dot(hull_equations[f], pos_h[i])
    # Process in chunks to manage memory
    chunk = 100000
    inside = np.ones(len(pos), dtype=bool)
    for start in range(0, len(pos), chunk):
        end   = min(start + chunk, len(pos))
        score = pos_h[start:end].astype(np.float64) @ hull_equations.T  # (B, F)
        inside[start:end] = (score <= 0).all(axis=1)

    return inside


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Multi-criterion floater removal: visibility filter + convex hull crop"
    )
    ap.add_argument("--input",      required=True,  help="Input .ply file")
    ap.add_argument("--output",     required=True,  help="Output .ply file")
    ap.add_argument("--transforms", required=True,  help="Path to transforms.json (COLMAP poses)")
    ap.add_argument("--min_views",  type=int,   default=3,
                    help="Remove Gaussians visible in fewer than this many cameras (default 3)")
    ap.add_argument("--hull_padding", type=float, default=1.5,
                    help="Inflate convex hull of cameras by this many metres (default 1.5)")
    ap.add_argument("--alpha_min",  type=float, default=0.005,
                    help="Remove Gaussians with sigmoid(opacity) < this value (default 0.005)")
    ap.add_argument("--skip_visibility", action="store_true",
                    help="Skip visibility filter (faster, hull+alpha only)")
    ap.add_argument("--skip_hull",  action="store_true",
                    help="Skip convex hull filter")
    args = ap.parse_args()

    inp  = Path(args.input)
    out  = Path(args.output)
    trf  = Path(args.transforms)

    if not inp.exists():
        sys.exit(f"[ERROR] Input not found: {inp}")
    if not trf.exists():
        sys.exit(f"[ERROR] transforms.json not found: {trf}")

    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Multi-criterion Gaussian Floater Removal")
    print(f"  Input      : {inp}")
    print(f"  Transforms : {trf}")
    print(f"  min_views  : {args.min_views}")
    print(f"  hull_pad   : {args.hull_padding}m")
    print(f"  alpha_min  : {args.alpha_min}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    ply, v, pos, ops = load_ply(inp)
    cameras = load_cameras(trf)
    n_total = len(pos)

    # Master keep mask — starts as all True
    keep = np.ones(n_total, dtype=bool)

    # ── Signal 3: Alpha threshold (fast, apply first to reduce work) ──────────
    alpha_keep = ops >= args.alpha_min
    removed_alpha = (~alpha_keep).sum()
    keep &= alpha_keep
    print(f"\n[Signal 3] Alpha threshold (>={args.alpha_min}):")
    print(f"  Removed {removed_alpha:,} transparent Gaussians ({100*removed_alpha/n_total:.1f}%)")
    print(f"  Remaining: {keep.sum():,}")

    # ── Signal 2: Convex hull crop ────────────────────────────────────────────
    if not args.skip_hull:
        print(f"\n[Signal 2] Convex hull crop (padding={args.hull_padding}m):")
        hull_eq = build_inflated_hull(cameras, args.hull_padding)
        hull_keep = apply_hull_filter(pos, hull_eq)
        removed_hull = (keep & ~hull_keep).sum()
        keep &= hull_keep
        print(f"  Removed {removed_hull:,} Gaussians outside hull ({100*removed_hull/n_total:.1f}%)")
        print(f"  Remaining: {keep.sum():,}")
    else:
        print("\n[Signal 2] Convex hull crop: SKIPPED")

    # ── Signal 1: Multi-view visibility filter ────────────────────────────────
    if not args.skip_visibility:
        print(f"\n[Signal 1] Visibility filter (min_views={args.min_views}):")
        vis = compute_visibility(pos, ops, cameras, args.alpha_min)

        # Save visibility stats for tuning
        print(f"  Visibility distribution:")
        for threshold in [0, 1, 2, 3, 5, 10, 20]:
            n = (vis >= threshold).sum()
            print(f"    views >= {threshold:2d}: {n:,} ({100*n/n_total:.1f}%)")

        vis_keep = vis >= args.min_views
        # Only remove from the currently-kept set
        removed_vis = (keep & ~vis_keep).sum()
        keep &= vis_keep
        print(f"  Removed {removed_vis:,} low-visibility Gaussians ({100*removed_vis/n_total:.1f}%)")
        print(f"  Remaining: {keep.sum():,}")
    else:
        print("\n[Signal 1] Visibility filter: SKIPPED")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_kept = keep.sum()
    n_removed = n_total - n_kept
    print(f"\n{'='*60}")
    print(f"  Total removed : {n_removed:,} ({100*n_removed/n_total:.1f}%)")
    print(f"  Total kept    : {n_kept:,} ({100*n_kept/n_total:.1f}%)")
    print(f"{'='*60}")

    if n_kept < 500_000:
        print(f"  ⚠️  WARNING: only {n_kept:,} Gaussians kept — scene may look sparse.")
        print(f"     Try lowering --min_views or raising --hull_padding.")

    # ── Write output PLY ──────────────────────────────────────────────────────
    print(f"\nWriting output PLY: {out}")

    # Rebuild vertex element with only kept Gaussians
    kept_data = v.data[keep]
    el = PlyElement.describe(kept_data, 'vertex')
    PlyData([el], text=False).write(str(out))

    sz = out.stat().st_size / 1024**2
    print(f"  Saved {n_kept:,} Gaussians → {out}  ({sz:.1f} MB)")
    print("\n  Next steps:")
    print(f"  python3 scripts/convert_to_splat.py --input {out} --output {out.with_suffix('.splat').with_stem(out.stem)}")
    print(f"  python3 open_viewer.py")
    print("\n  If quality is not right, re-run with different parameters:")
    print(f"  Bed/walls missing  → lower --min_views (try 2)")
    print(f"  Floaters remain    → raise --min_views (try 5) or lower --hull_padding (try 1.2)")


if __name__ == "__main__":
    main()
