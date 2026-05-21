"""
prune_splat.py — Remove near-transparent Gaussians from a .splat file.

Rationale: 78% of splats have alpha < 50/255. These are semi-transparent floaters
created during training to fit background noise. Pruning them:
  - Drops file size from ~133MB → ~30MB
  - Reduces sort time in the browser from ~90s → ~10s
  - Removes most foggy/ghosting artifacts
  - Result: 1.3M solid splats instead of 4.35M mostly-transparent ones

Usage:
  python scripts/prune_splat.py
  python scripts/prune_splat.py --input outputs/scene_semantic.splat --alpha_min 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def prune(input_path: Path, output_path: Path, alpha_min: int,
          crop: list | None) -> None:
    raw   = np.fromfile(input_path, dtype=np.uint8).reshape(-1, 32)
    total = len(raw)

    # Stage 1: alpha threshold
    alpha = raw[:, 27]  # byte 27 = alpha
    mask  = alpha >= alpha_min
    print(f"Input      : {total:,} splats  ({input_path.stat().st_size / 1e6:.0f} MB)")
    print(f"After alpha>={alpha_min}: {mask.sum():,} splats  ({100*mask.sum()/total:.1f}%)")

    # Stage 2: spatial crop
    if crop is not None:
        xmin, ymin, zmin, xmax, ymax, zmax = crop
        pos = raw[:, :12].view(np.float32).reshape(-1, 3)
        spatial = (
            (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax) &
            (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax) &
            (pos[:, 2] >= zmin) & (pos[:, 2] <= zmax)
        )
        before_crop = mask.sum()
        mask = mask & spatial
        print(f"After crop : {mask.sum():,} splats  ({100*mask.sum()/total:.1f}%)  "
              f"[removed {before_crop - mask.sum():,} outside bbox]")

    kept = mask.sum()
    print(f"Pruned     : {total - kept:,} splats  ({100*(total-kept)/total:.1f}%)")

    pruned = raw[mask]
    pruned.tofile(output_path)
    out_mb = output_path.stat().st_size / 1e6
    print(f"Output     : {output_path}  ({out_mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Prune floaters from a .splat file")
    ap.add_argument("--input",     default="outputs/scene_semantic.splat")
    ap.add_argument("--output",    default="outputs/scene_semantic_pruned.splat")
    ap.add_argument("--alpha_min", type=int, default=50,
                    help="Keep splats with alpha (0-255) >= this value (default 50)")
    ap.add_argument("--crop",      default=None,
                    help="Spatial crop bbox: xmin,ymin,zmin,xmax,ymax,zmax")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        sys.exit(f"[ERROR] Input not found: {inp}")

    crop = [float(v) for v in args.crop.split(",")] if args.crop else None
    if crop and len(crop) != 6:
        sys.exit("[ERROR] --crop needs exactly 6 values: xmin,ymin,zmin,xmax,ymax,zmax")

    out.parent.mkdir(parents=True, exist_ok=True)
    prune(inp, out, args.alpha_min, crop)
    print("[DONE]")


if __name__ == "__main__":
    main()
