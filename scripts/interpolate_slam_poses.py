"""Interpolate MASt3R-SLAM poses from 101 keyframes to all 1282 frames.

Reads the 101 COLMAP-format keyframe poses from images.bin, SLERP-interpolates
rotations and linearly interpolates translations for all frames in frames_v3/,
then writes a new images.bin (and symlinks the images/ directory) so that
gsplat and paint_semantic_pointcloud.py can use all 1282 frames.

Coordinate handling:
  - Interpolation is done in camera-to-world (c2w) space so SLERP is meaningful
  - Frames outside the keyframe range get the nearest boundary pose (no extrapolation)
  - Output written back to COLMAP world-to-camera (w2c) convention

Usage:
  python scripts/interpolate_slam_poses.py            # uses default paths
  python scripts/interpolate_slam_poses.py --dry_run  # print stats, no writes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    write_images_binary,
    rotmat_to_qvec,
    qvec_to_rotmat,
)


def parse_frame_num(name: str) -> int:
    """Extract integer frame number from 'frame_XXXX.jpg'."""
    stem = Path(name).stem  # "frame_0058"
    return int(stem.replace("frame_", ""))


def main():
    ap = argparse.ArgumentParser(description="Interpolate 101 SLAM poses to all 1282 frames.")
    ap.add_argument("--colmap_dir",  default=str(ROOT / "data/mast3r_out_v2/sparse/0"))
    ap.add_argument("--frames_dir",  default=str(ROOT / "data/frames_v3"))
    ap.add_argument("--images_link", default=str(ROOT / "data/mast3r_out_v2/images"),
                    help="Directory of symlinks to frame JPEGs (for gsplat)")
    ap.add_argument("--dry_run",     action="store_true")
    args = ap.parse_args()

    colmap_dir  = Path(args.colmap_dir)
    frames_dir  = Path(args.frames_dir)
    images_link = Path(args.images_link)

    # ── 1. Load existing 101 keyframe poses ────────────────────────────────
    print(f"[...] Reading keyframe poses from {colmap_dir}/images.bin")
    cameras_bin = read_cameras_binary(colmap_dir / "cameras.bin")
    images_bin  = read_images_binary(colmap_dir  / "images.bin")

    # Convert w2c → c2w for interpolation
    # w2c: p_cam = R_w2c @ p_world + t_w2c
    # c2w: R_c2w = R_w2c.T,  t_c2w = -R_w2c.T @ t_w2c  (= camera centre in world)
    kf_frames = []  # list of (frame_num, R_c2w 3x3, t_c2w 3)
    for img in images_bin.values():
        frame_num = parse_frame_num(img["name"])
        R_w2c = qvec_to_rotmat(img["qvec"])
        t_w2c = img["tvec"]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        kf_frames.append((frame_num, R_c2w, t_c2w))

    kf_frames.sort(key=lambda x: x[0])
    kf_nums   = np.array([kf[0] for kf in kf_frames], dtype=np.float64)
    kf_R_c2w  = [kf[1] for kf in kf_frames]
    kf_t_c2w  = np.array([kf[2] for kf in kf_frames])  # (101, 3)

    print(f"[OK] {len(kf_frames)} keyframes  "
          f"frames {int(kf_nums[0])}–{int(kf_nums[-1])}")

    # Build SLERP interpolator (scipy uses xyzw quaternion convention)
    rotations_c2w = Rotation.from_matrix(np.stack(kf_R_c2w, axis=0))
    slerp = Slerp(kf_nums, rotations_c2w)

    # ── 2. Get all frames from frames_v3/ ──────────────────────────────────
    all_jpgs = sorted(frames_dir.glob("frame_*.jpg"))
    all_frame_nums = [parse_frame_num(p.name) for p in all_jpgs]
    print(f"[OK] {len(all_jpgs)} frames in {frames_dir}")

    # ── 3. Interpolate / assign pose for each frame ────────────────────────
    # Build fast keyframe lookup for exact matches
    kf_exact = {kf[0]: (kf[1], kf[2]) for kf in kf_frames}

    min_kf = float(kf_nums[0])
    max_kf = float(kf_nums[-1])

    out_images = []
    cam_id = next(iter(cameras_bin.keys()))  # single camera

    exact_count = 0
    extrap_count = 0

    for img_id, (jpg_path, frame_num) in enumerate(zip(all_jpgs, all_frame_nums), start=1):
        fn = float(frame_num)

        if frame_num in kf_exact:
            # Exact keyframe — use stored pose
            R_c2w, t_c2w = kf_exact[frame_num]
            exact_count += 1
        else:
            # Clamp to keyframe range (no extrapolation)
            fn_clamped = float(np.clip(fn, min_kf, max_kf))
            if fn_clamped != fn:
                extrap_count += 1

            # Interpolate rotation with SLERP
            R_c2w = slerp(fn_clamped).as_matrix()

            # Interpolate translation with linear interp
            # Find bracketing keyframes
            idx = int(np.searchsorted(kf_nums, fn_clamped))
            if idx == 0:
                t_c2w = kf_t_c2w[0].copy()
            elif idx >= len(kf_nums):
                t_c2w = kf_t_c2w[-1].copy()
            else:
                fn0, fn1 = kf_nums[idx - 1], kf_nums[idx]
                alpha = (fn_clamped - fn0) / (fn1 - fn0)
                t_c2w = (1.0 - alpha) * kf_t_c2w[idx - 1] + alpha * kf_t_c2w[idx]

        # Convert c2w → w2c for COLMAP
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        qvec  = rotmat_to_qvec(R_w2c)  # (w, x, y, z)

        out_images.append({
            "image_id":     img_id,
            "qvec":         qvec,
            "tvec":         t_w2c,
            "camera_id":    cam_id,
            "name":         jpg_path.name,
            "point2D_xys":  np.zeros((0, 2)),
            "point3D_ids":  np.array([], dtype=np.int64),
        })

    print(f"\n[OK] Pose summary:")
    print(f"     Exact keyframes  : {exact_count}")
    print(f"     Interpolated     : {len(out_images) - exact_count - extrap_count}")
    print(f"     Clamped (edge)   : {extrap_count}")
    print(f"     Total            : {len(out_images)}")

    if args.dry_run:
        print("\n[dry_run] No files written.")
        return

    # ── 4. Write new images.bin ────────────────────────────────────────────
    out_bin = colmap_dir / "images.bin"
    print(f"\n[...] Writing {len(out_images)} poses → {out_bin}")
    write_images_binary(out_images, out_bin)
    print(f"[OK] images.bin updated  ({out_bin.stat().st_size / 1e6:.1f} MB)")

    # ── 5. Create images/ symlink directory for gsplat ─────────────────────
    # gsplat's dataset loader expects data_dir/images/ containing the JPEGs
    if not args.dry_run:
        images_link.mkdir(parents=True, exist_ok=True)
        linked = 0
        for jpg_path in all_jpgs:
            dst = images_link / jpg_path.name
            if not dst.exists():
                dst.symlink_to(jpg_path.resolve())
                linked += 1
        print(f"[OK] images/ symlinks  : {linked} new  "
              f"(total {len(list(images_link.glob('*.jpg')))})"
              f"  → {images_link}")

    print("\n=== Done ===")
    print(f"images.bin : {len(out_images)} frames  (was 101)")
    print(f"images/    : {images_link}  ({len(all_jpgs)} symlinks)")
    print(f"\nNext: run SAM2 on all {len(all_jpgs)} frames, then:")
    print(f"  python scripts/paint_semantic_pointcloud.py")


if __name__ == "__main__":
    main()
