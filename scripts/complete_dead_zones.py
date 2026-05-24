"""Identify low-confidence dead zones in the voxel grid and inpaint them with LaMa."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label as nd_label

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Identify dead zones and attempt 2-D inpainting."
    )
    p.add_argument(
        "--confidence_map",
        default="outputs/confidence_map.npy",
        help="Path to confidence_map.npy produced by build_confidence_map.py",
    )
    p.add_argument(
        "--confidence_metadata",
        default="outputs/confidence_metadata.json",
        help="Path to confidence_metadata.json",
    )
    p.add_argument(
        "--splat_dir",
        default="outputs/splat_mast3r_v2/",
        help="Splat directory (used to locate camera / config files)",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/dead_zones/",
        help="Directory for all dead-zone outputs",
    )
    p.add_argument(
        "--min_zone_voxels",
        type=int,
        default=200,
        help="Minimum voxels for a cluster to be considered a dead zone",
    )
    p.add_argument(
        "--max_zones",
        type=int,
        default=5,
        help="Maximum number of dead zones to process",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Camera / pose helpers
# ---------------------------------------------------------------------------

def load_cameras_from_splat_dir(splat_dir: str):
    """
    Try to load camera poses + intrinsics from the splat directory.

    Looks for (in order of preference):
      1. cameras.json  – {"frames": [{"file_path":…, "transform_matrix":…,
                                      "fl_x":…, "fl_y":…, "cx":…, "cy":…,
                                      "w":…, "h":…}]}
      2. transforms.json  – same NeRF-style format
      3. room_video.txt   – one pose per line: tx ty tz qx qy qz qw
                            with matching images in keyframes/room_video/

    Returns list of dicts:
      {"position": np.ndarray(3,),   # camera centre in world coords
       "R": np.ndarray(3,3),         # rotation  world-from-camera
       "K": np.ndarray(3,3),         # intrinsic matrix
       "image_path": str,            # absolute path to the keyframe image
       "frame_id": str}
    """
    splat_dir = Path(splat_dir)
    cameras = []

    # ---- attempt 1: cameras.json / transforms.json ----
    for cfile in ["cameras.json", "transforms.json", "transforms_train.json"]:
        cpath = splat_dir / cfile
        if cpath.exists():
            cameras = _parse_nerf_transforms(cpath)
            if cameras:
                print(f"  Loaded {len(cameras)} cameras from {cpath}")
                return cameras

    # ---- attempt 2: VGGT camera_poses.json (dict keyed by filename) ----
    for vggt_path in [
        Path("data/vggt_out/camera_poses.json"),
        Path("data/vggt_out_v2/camera_poses.json"),
        splat_dir.parent / "data" / "vggt_out" / "camera_poses.json",
    ]:
        if vggt_path.exists():
            cameras = _parse_vggt_camera_poses(vggt_path)
            if cameras:
                print(f"  Loaded {len(cameras)} cameras from {vggt_path}")
                return cameras

    # ---- attempt 3: sibling directories (NeRF transforms) ----
    for candidate in [
        splat_dir.parent / "cameras.json",
        splat_dir.parent / "transforms.json",
        Path("data") / "cameras.json",
        Path("data") / "transforms.json",
    ]:
        if candidate.exists():
            cameras = _parse_nerf_transforms(candidate)
            if cameras:
                print(f"  Loaded {len(cameras)} cameras from {candidate}")
                return cameras

    # ---- attempt 4: room_video.txt + keyframes ----
    for pose_path in [
        Path("data/mast3r_out/slam_output/room_video.txt"),
        splat_dir / "room_video.txt",
        Path("data/room_video.txt"),
    ]:
        if pose_path.exists():
            cameras = _parse_room_video_txt(pose_path)
            if cameras:
                print(f"  Loaded {len(cameras)} cameras from {pose_path}")
                return cameras

    print("  WARNING: No camera file found – dead zone projection will be skipped.")
    return []


def _parse_vggt_camera_poses(path: Path):
    """
    Parse VGGT camera_poses.json.

    Format::
        {
          "frame_0001.jpg": {
            "cam_to_world_4x4": [[...], ...],   # 4×4 world-from-camera
            "intrinsic_3x3":   [[...], ...],   # 3×3 K matrix
          },
          ...
        }

    Images are assumed to live alongside in
    ``<path.parent>/images/<frame_name>``.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return []

    # Search for images in multiple candidate directories, in priority order
    img_search_dirs = [
        path.parent / "images",          # data/vggt_out/images/
        Path("data") / "frames",          # data/frames/  (all 511 frames)
        Path("data") / "mast3r_out" / "images",
        path.parent,
    ]

    cameras = []

    for i, (fname, entry) in enumerate(sorted(data.items())):
        c2w = entry.get("cam_to_world_4x4")
        K3  = entry.get("intrinsic_3x3")
        if c2w is None or K3 is None:
            continue

        T = np.array(c2w, dtype=float)
        K = np.array(K3,  dtype=float)

        if T.shape != (4, 4) or K.shape != (3, 3):
            continue

        position = T[:3, 3]
        R        = T[:3, :3]

        # Resolve image path: try each search directory
        img_path = ""
        for img_dir in img_search_dirs:
            candidate = img_dir / fname
            if candidate.exists():
                img_path = str(candidate)
                break

        cameras.append({
            "position":   position,
            "R":          R,
            "K":          K,
            "image_path": img_path,
            "frame_id":   Path(fname).stem,
        })

    return cameras


def _parse_nerf_transforms(path: Path):
    """Parse a NeRF-style transforms.json / cameras.json."""
    with open(path) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return []

    # Global intrinsics (may be overridden per-frame)
    g_fl_x = data.get("fl_x", data.get("focal_length", 0))
    g_fl_y = data.get("fl_y", g_fl_x)
    g_cx   = data.get("cx", data.get("w", 640) / 2)
    g_cy   = data.get("cy", data.get("h", 480) / 2)

    cameras = []
    base = path.parent
    for i, fr in enumerate(frames):
        T = np.array(fr.get("transform_matrix",
                             fr.get("pose", np.eye(4).tolist())), dtype=float)
        if T.shape != (4, 4):
            continue

        fl_x = fr.get("fl_x", g_fl_x) or g_fl_x
        fl_y = fr.get("fl_y", g_fl_y) or g_fl_y
        cx   = fr.get("cx", g_cx) or g_cx
        cy   = fr.get("cy", g_cy) or g_cy

        K = np.array([[fl_x, 0, cx],
                      [0, fl_y, cy],
                      [0,  0,   1]], dtype=float)

        # Camera centre = T[:3, 3]  (world-from-camera convention)
        position = T[:3, 3]
        R        = T[:3, :3]

        fp = fr.get("file_path", fr.get("image_path", ""))
        img_path = str(base / fp) if fp else ""

        cameras.append({
            "position":  position,
            "R":         R,
            "K":         K,
            "image_path": img_path,
            "frame_id":  fr.get("frame_id", str(i)),
        })
    return cameras


def _parse_room_video_txt(path: Path):
    """
    Parse a plain-text pose file produced by MASt3R SLAM.

    Expected format (one line per keyframe):
        frame_id  tx ty tz  qx qy qz qw
    or just:
        tx ty tz  qx qy qz qw

    Images are assumed to live in
        <path.parent>/keyframes/room_video/   (*.png / *.jpg)
    sorted lexicographically.
    """
    from scipy.spatial.transform import Rotation

    # Collect keyframe images
    img_candidates = []
    for img_dir in [
        path.parent / "keyframes" / "room_video",
        path.parent / "keyframes",
        path.parent / "images",
    ]:
        if img_dir.exists():
            imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
            if imgs:
                img_candidates = imgs
                break

    cameras = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for idx, line in enumerate(lines):
        parts = line.split()
        # Accept 7-token or 8-token lines
        if len(parts) == 7:
            frame_id = str(idx)
            tx, ty, tz, qx, qy, qz, qw = map(float, parts)
        elif len(parts) == 8:
            frame_id = parts[0]
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:])
        else:
            continue

        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        position = np.array([tx, ty, tz])

        # Default intrinsics (wide-angle estimate; adjust if known)
        K = np.array([[600,   0, 320],
                      [  0, 600, 240],
                      [  0,   0,   1]], dtype=float)

        img_path = str(img_candidates[idx]) if idx < len(img_candidates) else ""

        cameras.append({
            "position":  position,
            "R":         R,
            "K":         K,
            "image_path": img_path,
            "frame_id":  frame_id,
        })
    return cameras


def closest_camera(centroid_world: np.ndarray, cameras: list):
    """Return the camera dict whose centre is closest to centroid_world."""
    best_cam = None
    best_dist = float("inf")
    for cam in cameras:
        d = np.linalg.norm(cam["position"] - centroid_world)
        if d < best_dist:
            best_dist = d
            best_cam = cam
    return best_cam, best_dist


def project_point(point_world: np.ndarray, cam: dict):
    """
    Project a 3-D world point into pixel coordinates using camera K and pose.

    The pose is stored as world-from-camera (R, t = position), so
        P_cam = R^T (P_world - t)

    Returns (u, v) or None if behind camera.
    """
    R = cam["R"]
    t = cam["position"]
    K = cam["K"]

    P_cam = R.T @ (point_world - t)
    if P_cam[2] <= 0:
        return None  # behind camera

    uv_h = K @ P_cam
    u = uv_h[0] / uv_h[2]
    v = uv_h[1] / uv_h[2]
    return int(round(u)), int(round(v))


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------

# Module-level LaMa singleton — initialised once, reused for all zones
_LAMA_INSTANCE = None
_LAMA_AVAILABLE = None   # None = not yet tested, True/False after first attempt


def run_lama(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Run LaMa inpainting.  Gracefully falls back to a blurred fill if the
    library is not installed or fails (so the rest of the pipeline still works).

    Uses a module-level singleton so the 196 MB model is loaded only once.
    """
    global _LAMA_INSTANCE, _LAMA_AVAILABLE

    # --- first call: attempt to load LaMa ---
    if _LAMA_AVAILABLE is None:
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore
            _LAMA_INSTANCE = SimpleLama()
            _LAMA_AVAILABLE = True
            print("    LaMa model loaded ✓")
        except ImportError:
            print("    simple-lama-inpainting not installed – using fallback inpainting.")
            _LAMA_AVAILABLE = False
        except Exception as exc:
            print(f"    LaMa init failed ({exc}) – using fallback inpainting.")
            _LAMA_AVAILABLE = False

    if not _LAMA_AVAILABLE:
        return _fallback_inpaint(image, mask)

    # --- run inference ---
    # Resize to max 1024px on the long edge for tractable CPU inference;
    # result is resized back to the original dimensions afterwards.
    try:
        orig_size = image.size  # (W, H)
        max_side = 1024
        if max(orig_size) > max_side:
            scale = max_side / max(orig_size)
            new_w = int(orig_size[0] * scale)
            new_h = int(orig_size[1] * scale)
            # Ensure dimensions are multiples of 8 (LaMa requirement)
            new_w = (new_w // 8) * 8
            new_h = (new_h // 8) * 8
            image_r = image.resize((new_w, new_h), Image.LANCZOS)
            mask_r  = mask.resize((new_w, new_h), Image.NEAREST)
            print(f"    Resized {orig_size} → ({new_w}, {new_h}) for LaMa inference")
        else:
            image_r = image
            mask_r  = mask
            orig_size = None  # no resize needed

        result = _LAMA_INSTANCE(image_r, mask_r)
        # SimpleLama may return a numpy array on some versions
        if not isinstance(result, Image.Image):
            result = Image.fromarray(result)

        # Resize result back to original dimensions if we downscaled
        if orig_size is not None:
            result = result.resize(orig_size, Image.LANCZOS)

        return result
    except Exception as exc:
        print(f"    LaMa inference failed ({exc}) – using fallback inpainting.")
        return _fallback_inpaint(image, mask)


def _fallback_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Very simple fallback: replace masked region with the median colour of
    a slightly dilated border of the mask region.  Not pretty but functional.
    """
    from scipy.ndimage import binary_dilation, uniform_filter

    img_arr  = np.array(image).astype(float)
    mask_arr = np.array(mask.convert("L")) > 127  # True = inpaint here

    result = img_arr.copy()
    # Dilated ring for colour sampling
    ring = binary_dilation(mask_arr, iterations=15) & ~mask_arr

    for c in range(img_arr.shape[2]):
        median_val = np.median(img_arr[..., c][ring]) if ring.any() else 128
        result[..., c][mask_arr] = median_val

    # Light blur over the inpainted region for smoothness
    blurred = uniform_filter(result, size=[5, 5, 1])
    result[mask_arr] = blurred[mask_arr]

    return Image.fromarray(result.astype(np.uint8))


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_circle_mask(width: int, height: int, cx: int, cy: int, radius: int = 50) -> Image.Image:
    """
    Create a PIL image (mode 'L') with a filled white circle (255) on black.
    LaMa convention: 255 = inpaint, 0 = keep.
    """
    mask_arr = np.zeros((height, width), dtype=np.uint8)
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask_arr[dist <= radius] = 255
    return Image.fromarray(mask_arr, mode="L")


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def build_summary_figure(zone_images: list, output_path: str):
    """
    zone_images: list of dicts {original, mask, inpainted} (PIL Images).
    """
    n = len(zone_images)
    if n == 0:
        return

    bg_color  = "#1a1a2e"
    text_color = "#e0e0e0"
    accent    = "#4fc3f7"

    fig, axes = plt.subplots(
        n, 3,
        figsize=(15, 5 * n),
        squeeze=False,
        facecolor=bg_color,
    )
    fig.suptitle(
        "RoboScene+ Dead Zone Completion",
        fontsize=18,
        color=accent,
        fontweight="bold",
        y=1.01 if n > 1 else 1.04,
    )

    col_titles = ["Original", "Mask", "Inpainted"]

    for row, zinfo in enumerate(zone_images):
        imgs = [zinfo["original"], zinfo["mask"], zinfo["inpainted"]]
        for col, (ax, img, ctitle) in enumerate(zip(axes[row], imgs, col_titles)):
            ax.set_facecolor(bg_color)
            if img is not None:
                display = img.convert("RGB")
                ax.imshow(display)
            else:
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.text(0.5, 0.5, "N/A", color=text_color, ha="center", va="center",
                        fontsize=14, transform=ax.transAxes)

            if row == 0:
                ax.set_title(ctitle, color=accent, fontsize=13, pad=8)

            ax.set_ylabel(
                f"Zone {zinfo['zone_id']}",
                color=text_color, fontsize=10, rotation=0,
                labelpad=55, va="center",
            )
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_edgecolor("#404060")

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor=bg_color)
    plt.close(fig)
    print(f"  Summary figure → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load confidence map + metadata
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading confidence map from {args.confidence_map} …")
    if not Path(args.confidence_map).exists():
        sys.exit(f"ERROR: {args.confidence_map} not found. "
                 "Run build_confidence_map.py first.")

    conf_map = np.load(args.confidence_map)          # shape (Nx, Ny, Nz)
    print(f"      Map shape: {conf_map.shape}  "
          f"dtype: {conf_map.dtype}  "
          f"range: [{conf_map.min():.3f}, {conf_map.max():.3f}]")

    if not Path(args.confidence_metadata).exists():
        sys.exit(f"ERROR: {args.confidence_metadata} not found.")

    with open(args.confidence_metadata) as f:
        meta = json.load(f)

    voxel_size = float(meta.get("voxel_size", meta.get("voxel_size_m", 0.05)))
    origin_xyz = np.array(meta.get("origin_xyz",
                           meta.get("grid_origin", [0.0, 0.0, 0.0])),
                          dtype=float)
    shape_xyz  = list(conf_map.shape)

    print(f"      voxel_size={voxel_size} m  origin={origin_xyz}  "
          f"shape={shape_xyz}")

    # ------------------------------------------------------------------
    # 2. Find dead zone clusters
    # ------------------------------------------------------------------
    print("\n[2/6] Finding dead zone clusters (confidence < 0.3) …")
    dead_mask = conf_map < 0.3
    labeled, num_features = nd_label(dead_mask)
    print(f"      Raw connected components: {num_features}")

    if num_features == 0:
        print("      No dead zones found – all voxels have confidence ≥ 0.3.")
        _write_empty_report(out_dir, args)
        return

    # Count voxels per label
    label_ids, label_counts = np.unique(labeled[labeled > 0], return_counts=True)
    # Filter by min size
    big_enough = label_counts >= args.min_zone_voxels
    label_ids   = label_ids[big_enough]
    label_counts = label_counts[big_enough]

    # Sort descending
    order = np.argsort(label_counts)[::-1]
    label_ids    = label_ids[order]
    label_counts = label_counts[order]

    num_found     = len(label_ids)
    num_to_process = min(num_found, args.max_zones)

    print(f"      Found {num_found} dead zones ≥ {args.min_zone_voxels} voxels, "
          f"processing top {num_to_process}")

    # ------------------------------------------------------------------
    # 3. Compute zone centroids & volumes
    # ------------------------------------------------------------------
    print("\n[3/6] Computing centroids and volumes …")
    zones = []
    for rank, (lid, count) in enumerate(zip(label_ids[:num_to_process],
                                             label_counts[:num_to_process])):
        vox_indices = np.argwhere(labeled == lid)         # (N, 3) array
        centroid_vox = vox_indices.mean(axis=0)            # fractional index
        centroid_world = origin_xyz + (centroid_vox + 0.5) * voxel_size
        volume_m3 = float(count) * (voxel_size ** 3)

        zones.append({
            "zone_id":        rank,
            "label_id":       int(lid),
            "size_voxels":    int(count),
            "centroid_vox":   centroid_vox.tolist(),
            "centroid_world": centroid_world.tolist(),
            "volume_m3":      round(volume_m3, 6),
            "processed":      False,
            "closest_frame":  None,
        })
        print(f"      Zone {rank}: {count} voxels, "
              f"centroid_world=({centroid_world[0]:.3f}, "
              f"{centroid_world[1]:.3f}, {centroid_world[2]:.3f}), "
              f"volume={volume_m3:.4f} m³")

    # ------------------------------------------------------------------
    # 4. Load cameras
    # ------------------------------------------------------------------
    print("\n[4/6] Loading camera poses …")
    cameras = load_cameras_from_splat_dir(args.splat_dir)
    has_cameras = len(cameras) > 0

    # ------------------------------------------------------------------
    # 5. Inpainting loop
    # ------------------------------------------------------------------
    print("\n[5/6] Running inpainting for each zone …")
    zone_images = []     # for summary figure

    for zinfo in zones:
        i = zinfo["zone_id"]
        centroid_world = np.array(zinfo["centroid_world"])

        # ---- find closest camera ----
        cam = None
        proj_uv = None
        if has_cameras:
            cam, dist = closest_camera(centroid_world, cameras)
            proj_uv = project_point(centroid_world, cam)
            zinfo["closest_frame"] = cam["frame_id"]
            zinfo["dist_to_camera_m"] = round(float(dist), 4)

        # ---- load image ----
        img_pil = None
        if cam is not None and cam.get("image_path") and Path(cam["image_path"]).exists():
            img_pil = Image.open(cam["image_path"]).convert("RGB")
        else:
            # Synthesise a placeholder grey image
            img_pil = Image.fromarray(
                np.full((480, 640, 3), 80, dtype=np.uint8)
            )
            if cam is not None:
                print(f"    Zone {i}: keyframe image not found "
                      f"({cam.get('image_path', 'unknown')}) – using placeholder.")

        W, H = img_pil.size

        # ---- build mask ----
        if proj_uv is not None:
            u, v = proj_uv
            # Clamp to image bounds
            u = max(50, min(W - 50, u))
            v = max(50, min(H - 50, v))
        else:
            # Fallback: centre of image
            u, v = W // 2, H // 2

        mask_pil = make_circle_mask(W, H, u, v, radius=50)

        # ---- inpainting ----
        inpainted_pil = run_lama(img_pil, mask_pil)

        # ---- save outputs ----
        orig_path      = out_dir / f"zone_{i}_original.png"
        mask_path      = out_dir / f"zone_{i}_mask.png"
        inpainted_path = out_dir / f"zone_{i}_inpainted.png"

        img_pil.save(orig_path)
        mask_pil.convert("RGB").save(mask_path)
        inpainted_pil.save(inpainted_path)

        zinfo["processed"] = True

        frame_str = cam["frame_id"] if cam else "N/A"
        print(
            f"    Zone {i}: centroid=({centroid_world[0]:.3f}, "
            f"{centroid_world[1]:.3f}, {centroid_world[2]:.3f}), "
            f"volume={zinfo['volume_m3']:.4f} m³, "
            f"closest_camera={frame_str}, inpainted ✓"
        )

        zone_images.append({
            "zone_id":   i,
            "original":  img_pil,
            "mask":      mask_pil.convert("RGB"),
            "inpainted": inpainted_pil,
        })

    # Also collect info for zones not processed (if num_found > max_zones)
    all_zone_info = []
    for rank, (lid, count) in enumerate(zip(label_ids, label_counts)):
        if rank < num_to_process:
            all_zone_info.append(zones[rank])
        else:
            centroid_vox = np.argwhere(labeled == lid).mean(axis=0)
            centroid_world = origin_xyz + (centroid_vox + 0.5) * voxel_size
            volume_m3 = float(count) * (voxel_size ** 3)
            all_zone_info.append({
                "zone_id":        rank,
                "label_id":       int(lid),
                "size_voxels":    int(count),
                "centroid_world": centroid_world.tolist(),
                "volume_m3":      round(volume_m3, 6),
                "processed":      False,
                "closest_frame":  None,
            })

    # ------------------------------------------------------------------
    # 6. Summary figure
    # ------------------------------------------------------------------
    print("\n[6/6] Generating summary figure …")
    summary_path = str(out_dir / "dead_zone_summary.png")
    if zone_images:
        build_summary_figure(zone_images, summary_path)
    else:
        print("  No zones processed – skipping figure.")

    # ------------------------------------------------------------------
    # Save report JSON
    # ------------------------------------------------------------------
    total_dead_volume = sum(z["volume_m3"] for z in zones)
    report = {
        "num_found":     num_found,
        "num_processed": num_to_process,
        "total_dead_volume_m3": round(total_dead_volume, 6),
        "zones": all_zone_info,
    }
    report_path = out_dir / "dead_zone_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("DEAD ZONE COMPLETION – SUMMARY")
    print("=" * 55)
    print(f"  Dead zones found:      {num_found}")
    print(f"  Dead zones processed:  {num_to_process}")
    print(f"  Total dead volume:     {total_dead_volume:.4f} m³")
    print(f"  Dead zone report:      {report_path}")
    if zone_images:
        print(f"  Summary figure:        {summary_path}")
    print("=" * 55)
    print("\nFuture Work: back-projection of inpainted pixels into new")
    print("3-D Gaussians is not yet implemented (see README).\n")


def _write_empty_report(out_dir: Path, args):
    report = {
        "num_found":     0,
        "num_processed": 0,
        "total_dead_volume_m3": 0.0,
        "zones": [],
    }
    report_path = out_dir / "dead_zone_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDead zone report: {report_path}")
    print("Dead zones found: 0 | Dead zones processed: 0 | "
          "Total dead volume: 0.0 m³")


if __name__ == "__main__":
    main()