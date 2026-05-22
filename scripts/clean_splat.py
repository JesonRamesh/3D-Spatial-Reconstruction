"""
clean_splat.py — Remove floaters using Statistical Outlier Removal (SOR).

How it works:
  For each Gaussian, compute the mean distance to its K nearest neighbours.
  Gaussians whose mean neighbour distance is > (global_mean + std_multiplier * global_std)
  are classified as outliers (floaters) and removed.

  Real surfaces: Gaussians are densely packed → small mean neighbour distance
  Floaters:      Gaussians are isolated in space → large mean neighbour distance

This is the same algorithm used in Open3D's remove_statistical_outlier().

Usage:
  python scripts/clean_splat.py --input outputs/splat_video_v2/scene_full.splat --output outputs/splat_video_v2/scene_clean.splat
  python scripts/clean_splat.py --input outputs/splat_video_v2/scene_full.splat --output outputs/splat_video_v2/scene_clean.splat --k 20 --std_ratio 1.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN = True
except ImportError:
    SKLEARN = False


def sor_filter(positions: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """
    Statistical Outlier Removal.
    Returns boolean mask: True = keep (inlier), False = remove (outlier).
    """
    print(f"  Computing {k}-NN distances for {len(positions):,} points...")

    if SKLEARN:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=-1)
        nn.fit(positions)
        distances, _ = nn.kneighbors(positions)
        mean_dists = distances[:, 1:].mean(axis=1)  # exclude self (distance=0)
    else:
        # Fallback: chunked numpy (slower but no sklearn needed)
        # Process in chunks to avoid OOM
        chunk = 10000
        mean_dists = np.zeros(len(positions), dtype=np.float32)
        for i in range(0, len(positions), chunk):
            batch = positions[i:i+chunk]
            # Compute distances from batch to all points
            diff = positions[np.newaxis, :, :] - batch[:, np.newaxis, :]  # (chunk, N, 3)
            dists = np.sqrt((diff**2).sum(axis=2))  # (chunk, N)
            # Sort and take k nearest (skip self)
            dists.sort(axis=1)
            mean_dists[i:i+chunk] = dists[:, 1:k+1].mean(axis=1)
            if i % 100000 == 0:
                print(f"    {i:,}/{len(positions):,}")

    global_mean = mean_dists.mean()
    global_std  = mean_dists.std()
    threshold   = global_mean + std_ratio * global_std

    print(f"  Mean neighbour dist: {global_mean:.4f}")
    print(f"  Std:                 {global_std:.4f}")
    print(f"  Threshold (mean + {std_ratio}×std): {threshold:.4f}")

    mask = mean_dists <= threshold
    print(f"  Inliers : {mask.sum():,} ({100*mask.mean():.1f}%)")
    print(f"  Outliers: {(~mask).sum():,} ({100*(~mask).mean():.1f}%)")

    return mask


def main():
    ap = argparse.ArgumentParser(description="Remove floaters using Statistical Outlier Removal")
    ap.add_argument("--input",      default="outputs/splat_video_v2/scene_full.splat")
    ap.add_argument("--output",     default="outputs/splat_video_v2/scene_clean.splat")
    ap.add_argument("--k",          type=int,   default=20,
                    help="Number of nearest neighbours to consider (default 20)")
    ap.add_argument("--std_ratio",  type=float, default=2.0,
                    help="Std multiplier for outlier threshold (default 2.0, lower = more aggressive)")
    ap.add_argument("--alpha_min",  type=int,   default=1,
                    help="Pre-filter: only keep splats with alpha >= this (default 1)")
    ap.add_argument("--subsample",  type=int,   default=None,
                    help="Subsample N points for faster KNN (then apply mask to full set). Default: use all.")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        sys.exit(f"[ERROR] Input not found: {inp}")

    print(f"Loading {inp}...")
    raw   = np.fromfile(inp, dtype=np.uint8).reshape(-1, 32)
    total = len(raw)
    print(f"  {total:,} splats  ({inp.stat().st_size/1e6:.0f} MB)")

    # Pre-filter by alpha
    alpha = raw[:, 27]
    pre_mask = alpha >= args.alpha_min
    print(f"  After alpha>={args.alpha_min}: {pre_mask.sum():,} ({100*pre_mask.mean():.1f}%)")

    raw_filtered = raw[pre_mask]
    positions = raw_filtered[:, :12].view(np.float32).reshape(-1, 3).copy()

    # SOR filter
    print(f"\nRunning Statistical Outlier Removal (k={args.k}, std_ratio={args.std_ratio})...")

    if not SKLEARN:
        print("  WARNING: scikit-learn not found. Using slow numpy fallback.")
        print("  Install with: pip install scikit-learn")

    sor_mask = sor_filter(positions, args.k, args.std_ratio)

    # Apply mask
    kept = raw_filtered[sor_mask]
    print(f"\nResult: {len(kept):,} splats kept from {total:,} input ({100*len(kept)/total:.1f}%)")

    out.parent.mkdir(parents=True, exist_ok=True)
    kept.tofile(out)
    print(f"Output: {out}  ({out.stat().st_size/1e6:.0f} MB)")
    print("[DONE]")


if __name__ == "__main__":
    main()
