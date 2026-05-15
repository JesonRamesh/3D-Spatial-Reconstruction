#!/usr/bin/env python3
"""
Re-export COLMAP files from existing VGGT output with all fixes applied.

Fixes:
  1. Invert poses from camera-to-world → world-to-camera (COLMAP convention)
  2. Set fx = fy (square pixel constraint) using max-dimension scaling
  3. Regenerate points3D from VGGT depth maps with proper tracks

Usage:
    python scripts/fix_colmap_export.py [--override-focal FOCAL_PX]

Options:
    --override-focal FOCAL_PX   Override VGGT focal estimate with known value.
                                 For iPhone 0.5x ultrawide at 1920px width: ~688
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colmap_utils import rotmat_to_qvec, write_cameras_binary, write_images_binary, write_points3D_binary


def c2w_to_w2c(pose_c2w: np.ndarray) -> np.ndarray:
    """Convert 4x4 camera-to-world -> world-to-camera."""
    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3]  = -R.T @ t
    return w2c


def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      pose_c2w: np.ndarray, conf: np.ndarray = None,
                      conf_threshold: float = 1.5, max_points: int = 5000) -> np.ndarray:
    """Backproject depth map to 3D world points using camera intrinsics and c2w pose.
    
    Returns Nx6 array (x, y, z, r, g, b).
    """
    H, W = depth.shape[:2]
    # Create pixel grid at depth map resolution
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # Scale intrinsics to depth map resolution (depth is at VGGT 518x518 resolution)
    # We need to know the original resolution to scale. For now assume depth is at
    # its own resolution and intrinsics need to be adapted.
    # Since depth is at 518x518 and intrinsics are at orig resolution,
    # we need to scale intrinsics to 518x518.
    # Actually, the depth map corresponds to the squished 518x518 input.
    # We'll compute in camera space and transform to world.

    valid = depth > 0.01
    if conf is not None:
        valid &= conf > conf_threshold

    if valid.sum() == 0:
        return np.zeros((0, 6))

    # Subsample if too many points
    valid_idx = np.where(valid.ravel())[0]
    if len(valid_idx) > max_points:
        rng = np.random.RandomState(42)
        valid_idx = rng.choice(valid_idx, max_points, replace=False)

    u_valid = uu.ravel()[valid_idx]
    v_valid = vv.ravel()[valid_idx]
    z_valid = depth.ravel()[valid_idx]

    # Backproject: (u,v,z) → camera space (using depth-map-resolution intrinsics)
    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)  # Nx4
    pts_world = (pose_c2w @ pts_cam.T).T[:, :3]  # Nx3

    # Use grey colour (we don't have easy access to RGB at depth resolution)
    rgb = np.full((len(pts_world), 3), 128, dtype=np.uint8)

    return np.column_stack([pts_world, rgb])


def main():
    parser = argparse.ArgumentParser(description="Re-export COLMAP with fixes")
    parser.add_argument("--vggt-dir", default="data/vggt_out", help="VGGT output directory")
    parser.add_argument("--output-dir", default=None, help="Output sparse dir (default: {vggt_dir}/sparse_fixed/0)")
    parser.add_argument("--override-focal", type=float, default=None,
                        help="Override focal length in pixels at original resolution")
    parser.add_argument("--resolution", type=int, default=518, help="VGGT inference resolution")
    parser.add_argument("--max-depth-points", type=int, default=100000,
                        help="Max 3D points from depth backprojection")
    args = parser.parse_args()

    vggt_dir = Path(args.vggt_dir)
    output_dir = Path(args.output_dir) if args.output_dir else vggt_dir / "sparse_fixed" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    with open(vggt_dir / "camera_poses.json") as f:
        camera_poses = json.load(f)

    with open(vggt_dir / "vggt_metadata.json") as f:
        metadata = json.load(f)

    orig_w = metadata.get("original_width") or metadata.get("width")
    orig_h = metadata.get("original_height") or metadata.get("height")
    if orig_w is None or orig_h is None:
        # Try to infer from an image
        img_dir = vggt_dir / "images"
        sample = next(img_dir.iterdir())
        from PIL import Image
        im = Image.open(sample)
        orig_w, orig_h = im.size
        print(f"Inferred original resolution from {sample.name}: {orig_w}×{orig_h}")

    print(f"Original resolution: {orig_w}×{orig_h}")
    print(f"VGGT inference resolution: {args.resolution}×{args.resolution}")
    print(f"Number of camera poses: {len(camera_poses)}")

    # ---------------------------------------------------------------
    # FIX 1: Compute correct focal length
    # ---------------------------------------------------------------
    if args.override_focal is not None:
        focal_x = focal_y = args.override_focal
        print(f"\n[FIX 1] Using OVERRIDE focal: fx = fy = {focal_x:.2f}")
    else:
        # Read focal from camera_poses.json intrinsic_3x3 (already uniformly scaled
        # by the fixed scale_intrinsics() in run_vggt.py). Since the fix sets
        # fx = fy = fx_vggt * max(W,H)/518, reading fx_json directly gives the
        # correct single focal value.
        keys_sorted = sorted(camera_poses.keys())
        # camera_poses.json stores intrinsic_3x3 per frame
        sample_key = keys_sorted[0]
        pose_entry = camera_poses[sample_key]
        if isinstance(pose_entry, dict) and "intrinsic_3x3" in pose_entry:
            K = np.array(pose_entry["intrinsic_3x3"])
            focal_x = focal_y = float(K[0, 0])   # fx (== fy after fix)
            print(f"\n[FIX 1] Focal from camera_poses.json intrinsic_3x3: fx=fy={focal_x:.2f}")
        else:
            # Fallback: read from cameras.bin and undo the double-scaling
            for cam_bin_path in [
                vggt_dir / "sparse" / "cameras.bin",
                vggt_dir / "sparse" / "0" / "cameras.bin",
            ]:
                if cam_bin_path.exists():
                    with open(cam_bin_path, "rb") as f:
                        struct.unpack("<Q", f.read(8))
                        struct.unpack("<i", f.read(4))
                        model_id = struct.unpack("<i", f.read(4))[0]
                        struct.unpack("<Q", f.read(8))
                        struct.unpack("<Q", f.read(8))
                        n_params = {1: 4}.get(model_id, 4)
                        params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
                    # cameras.bin may still be double-scaled from an old run;
                    # undo by dividing by max(W,H)/518 to recover corrected focal
                    old_fx  = params[0]
                    rr      = max(orig_w, orig_h) / args.resolution
                    focal_x = focal_y = old_fx / rr   # undo old double-scale
                    print(f"\n[FIX 1] Focal recovered from cameras.bin (undone double-scale): "
                          f"fx=fy={focal_x:.2f}")
                    print(f"  (raw cameras.bin fx={old_fx:.2f} / {rr:.4f})")
                    break
            else:
                # Last resort: use VGGT-typical focal for 518x518 -> max dim
                focal_x = focal_y = 271.6 * max(orig_w, orig_h) / args.resolution
                print(f"\n[FIX 1] Using fallback focal estimate: {focal_x:.2f}")

    cx = orig_w / 2.0
    cy = orig_h / 2.0

    # ---------------------------------------------------------------
    # Write cameras.bin — PINHOLE with fx = fy
    # ---------------------------------------------------------------
    cameras = {
        1: {
            "camera_id": 1,
            "model_id": 1,  # PINHOLE
            "width": orig_w,
            "height": orig_h,
            "params": [focal_x, focal_y, cx, cy],
        }
    }
    write_cameras_binary(cameras, output_dir / "cameras.bin")
    print(f"  → Wrote cameras.bin (PINHOLE, fx=fy={focal_x:.2f}, {orig_w}×{orig_h})")

    # ---------------------------------------------------------------
    # FIX 2: Invert poses c2w → w2c and write images.bin
    # ---------------------------------------------------------------
    print(f"\n[FIX 2] Converting camera-to-world → world-to-camera poses")
    images = {}
    sorted_keys = sorted(camera_poses.keys())
    for idx, frame_name in enumerate(sorted_keys):
        image_id = idx + 1
        pose_c2w = np.array(camera_poses[frame_name])
        pose_w2c = c2w_to_w2c(pose_c2w)

        R_w2c = pose_w2c[:3, :3]
        t_w2c = pose_w2c[:3, 3]
        qvec = rotmat_to_qvec(R_w2c)

        images[image_id] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": t_w2c,
            "camera_id": 1,
            "name": frame_name,
            "xys": np.zeros((0, 2)),
            "point3D_ids": np.array([], dtype=np.int64),
        }

    write_images_binary(images, output_dir / "images.bin")
    print(f"  → Wrote images.bin ({len(images)} images with w2c poses)")

    # ---------------------------------------------------------------
    # FIX 3: Generate points3D with tracks from depth maps
    # ---------------------------------------------------------------
    print(f"\n[FIX 3] Backprojecting depth maps to 3D points with tracks")
    depth_dir = vggt_dir / "depths"
    all_points = []
    points_per_frame = args.max_depth_points // len(sorted_keys)

    # Intrinsics at VGGT depth map resolution (518×518)
    fx_518 = focal_x * args.resolution / orig_w
    fy_518 = focal_y * args.resolution / orig_h
    cx_518 = args.resolution / 2.0
    cy_518 = args.resolution / 2.0

    for idx, frame_name in enumerate(sorted_keys):
        base = frame_name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
        depth_file = depth_dir / f"{base}_depth.npy"
        conf_file = depth_dir / f"{base}_conf.npy"

        if not depth_file.exists():
            continue

        depth = np.load(depth_file)
        conf = np.load(conf_file) if conf_file.exists() else None

        pose_c2w = np.array(camera_poses[frame_name])
        pts = backproject_depth(
            depth, fx_518, fy_518, cx_518, cy_518,
            pose_c2w, conf=conf,
            max_points=points_per_frame,
        )
        if len(pts) > 0:
            # Tag each point with its source image for tracks
            img_ids = np.full(len(pts), idx + 1, dtype=np.int64)
            all_points.append((pts, img_ids))

    if all_points:
        all_pts = np.concatenate([p[0] for p in all_points], axis=0)
        all_img_ids = np.concatenate([p[1] for p in all_points], axis=0)

        # Subsample to max_depth_points
        if len(all_pts) > args.max_depth_points:
            rng = np.random.RandomState(42)
            sel = rng.choice(len(all_pts), args.max_depth_points, replace=False)
            all_pts = all_pts[sel]
            all_img_ids = all_img_ids[sel]

        points3D = {}
        for i in range(len(all_pts)):
            pid = i + 1
            points3D[pid] = {
                "point3D_id": pid,
                "xyz": all_pts[i, :3],
                "rgb": all_pts[i, 3:6].astype(np.uint8),
                "error": 0.0,
                "track": [(int(all_img_ids[i]), 0)],  # (image_id, point2D_idx)
            }
        write_points3D_binary(points3D, output_dir / "points3D.bin")
        print(f"  → Wrote points3D.bin ({len(points3D)} points with single-image tracks)")
    else:
        # Write empty points3D
        write_points3D_binary({}, output_dir / "points3D.bin")
        print("  → Wrote empty points3D.bin (no depth maps found)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FIXED COLMAP OUTPUT: {output_dir}")
    print(f"{'='*60}")
    print(f"  cameras.bin  : PINHOLE, fx=fy={focal_x:.2f}, {orig_w}×{orig_h}")
    print(f"  images.bin   : {len(images)} images with INVERTED (w2c) poses")
    print(f"  points3D.bin : {len(points3D) if all_points else 0} points from depth backprojection")
    print()
    print("To train with fixed data:")
    print(f"  ns-train splatfacto --data {vggt_dir} \\")
    print(f"    --pipeline.datamanager.dataparser.colmap-path {output_dir.relative_to(vggt_dir)}")
    print()
    if args.override_focal is None:
        print("⚠️  Focal length is still VGGT's estimate (~18° FOV for ultrawide lens).")
        print("   For iPhone 0.5x ultrawide, try: --override-focal 688")
        print("   This gives ~120° horizontal FOV which matches the actual lens.")


if __name__ == "__main__":
    main()