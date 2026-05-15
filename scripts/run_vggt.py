#!/usr/bin/env python3
"""
RoboScene+ Session 2: VGGT 3D Reconstruction
==============================================

Runs VGGT (CVPR 2025 Best Paper) on extracted frames to produce:
  - COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
  - Per-frame depth maps as .npy files
  - Camera poses as JSON (4x4 extrinsic matrices)
  - Point cloud as .ply for quick visualisation

Supports CUDA → MPS → CPU automatic device fallback.
MPS fallback env var is set before any torch import.

Usage:
    python scripts/run_vggt.py                              # defaults from config.yaml
    python scripts/run_vggt.py --dry_run                    # first 10 frames only
    python scripts/run_vggt.py --frames_dir data/frames --output_dir data/vggt_out
    python scripts/run_vggt.py --batch_size 10 --device mps
"""

# ── MPS / OpenMP env vars MUST be set before any torch import ──────────
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # allow MPS to use all memory
# Route HF downloads to scratch (large model — don't fill home dir on UCL GPU)
os.environ.setdefault("HF_HOME", "/scratch0/jrameshs/hf_cache")

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── VGGT imports ───────────────────────────────────────────────────────
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

# Add scripts dir to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_utils import write_colmap_reconstruction


# ── Logging ────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("roboscene.vggt")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "vggt.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s"))
    logger.addHandler(fh)

    return logger


# ── Device detection ───────────────────────────────────────────────────

def select_device(requested: str, logger: logging.Logger) -> torch.device:
    """Pick best available device: cuda → mps → cpu."""
    if requested != "auto":
        dev = torch.device(requested)
        logger.info(f"Using requested device: {dev}")
        return dev

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"Auto-detected CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Auto-detected Apple MPS")
    else:
        dev = torch.device("cpu")
        logger.info("Using CPU (no GPU detected)")
    return dev


def select_dtype(device: torch.device) -> torch.dtype:
    """Pick appropriate dtype for the device."""
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if capability >= 8 else torch.float16
    elif device.type == "mps":
        return torch.float16
    return torch.float32


# ── VGGT inference ─────────────────────────────────────────────────────

def load_model(device: torch.device, logger: logging.Logger) -> VGGT:
    """
    Load VGGT-1B weights from HuggingFace.

    VGGT stores weights as model.pt (not pytorch_model.bin), so
    from_pretrained() doesn't work. We use hf_hub_download instead.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Loading VGGT-1B model from HuggingFace (facebook/VGGT-1B)...")
    logger.info(f"  HF cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    t0 = time.time()

    model_path = hf_hub_download(repo_id="facebook/VGGT-1B", filename="model.pt")
    logger.info(f"  Weights: {model_path}")

    model = VGGT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.1f}s")
    return model


def run_inference_batch(
    model: VGGT,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger,
    vggt_resolution: int = 518,
):
    """
    Run VGGT on a batch of images.

    Args:
        images: [S, 3, H, W] tensor in [0, 1], loaded at img_load_resolution
        device: torch device
        dtype: torch dtype for autocast
        vggt_resolution: internal VGGT resolution (always 518)

    Returns:
        extrinsic: [S, 3, 4] numpy  (cam-from-world, OpenCV convention)
        intrinsic: [S, 3, 3] numpy  (at vggt_resolution scale)
        depth_map: [S, H, W, 1] numpy
        depth_conf: [S, H, W] numpy
        world_points: [S, H, W, 3] numpy
    """
    S = images.shape[0]
    logger.debug(f"  Inference on {S} frames, resolution={vggt_resolution}")

    # Resize to VGGT's fixed 518×518
    images_518 = F.interpolate(
        images, size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        # Use autocast for CUDA, manual cast for MPS/CPU
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                images_in = images_518[None].to(device)  # [1, S, 3, H, W]
                agg_tokens, ps_idx = model.aggregator(images_in)
                pose_enc = model.camera_head(agg_tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images_in.shape[-2:]
                )
                depth_map, depth_conf = model.depth_head(agg_tokens, images_in, ps_idx)
        else:
            # MPS / CPU: cast model inputs manually
            images_in = images_518[None].to(device=device, dtype=dtype)
            agg_tokens, ps_idx = model.aggregator(images_in)

            # Camera head — needs float32 for pose decoding
            with torch.autocast(device_type=device.type, enabled=False):
                pose_enc = model.camera_head(agg_tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_in.shape[-2:]
            )

            # Depth head
            with torch.autocast(device_type=device.type, enabled=False):
                depth_map, depth_conf = model.depth_head(agg_tokens, images_in, ps_idx)

    # Convert to numpy
    extrinsic = extrinsic.squeeze(0).cpu().float().numpy()   # [S, 3, 4]
    intrinsic = intrinsic.squeeze(0).cpu().float().numpy()   # [S, 3, 3]
    depth_map = depth_map.squeeze(0).cpu().float().numpy()   # [S, H, W, 1]
    depth_conf = depth_conf.squeeze(0).cpu().float().numpy() # [S, H, W]

    # Unproject to 3D world points
    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    return extrinsic, intrinsic, depth_map, depth_conf, world_points


# ── COLMAP export (feedforward, no BA) ─────────────────────────────────

def export_colmap(
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_conf: np.ndarray,
    world_points: np.ndarray,
    images_tensor: torch.Tensor,
    image_names: list,
    original_coords: np.ndarray,
    output_dir: Path,
    vggt_resolution: int,
    img_load_resolution: int,
    conf_threshold: float,
    max_points: int,
    logger: logging.Logger,
):
    """Export reconstruction to COLMAP sparse format using custom binary writer."""
    import trimesh

    # Get RGB colours for points (resize images to vggt resolution)
    points_rgb = F.interpolate(
        images_tensor,
        size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False,
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # [S, H, W, 3]

    # Adaptive confidence threshold: use percentile if fixed threshold yields 0 points
    actual_threshold = conf_threshold
    n_above = (depth_conf >= conf_threshold).sum()
    if n_above == 0:
        # Fall back to top 50% of confidence values
        actual_threshold = float(np.percentile(depth_conf, 50))
        logger.warning(f"  No points above conf={conf_threshold:.1f}, "
                       f"using adaptive threshold={actual_threshold:.2f}")

    logger.info(f"  Conf threshold: {actual_threshold:.2f}, max points: {max_points}")
    logger.info(f"  Conf range: {depth_conf.min():.2f} – {depth_conf.max():.2f}")

    # Write COLMAP binary files (custom writer, no pycolmap dependency)
    n_points, sparse_dir = write_colmap_reconstruction(
        extrinsics=extrinsic,
        intrinsics=intrinsic,
        image_names=image_names,
        world_points=world_points,
        depth_conf=depth_conf,
        points_rgb=points_rgb,
        output_dir=output_dir,
        original_coords=original_coords,
        vggt_resolution=vggt_resolution,
        conf_threshold=actual_threshold,
        max_points=max_points,
        camera_model="PINHOLE",
    )

    logger.info(f"  COLMAP sparse: {sparse_dir}/ ({n_points} 3D points)")

    # Also save a .ply point cloud for quick visualisation
    conf_mask = depth_conf >= actual_threshold
    true_count = conf_mask.sum()
    if true_count > max_points:
        true_indices = np.flatnonzero(conf_mask.ravel())
        sampled = np.random.choice(true_indices, size=max_points, replace=False)
        new_mask = np.zeros(conf_mask.size, dtype=bool)
        new_mask[sampled] = True
        conf_mask = new_mask.reshape(conf_mask.shape)

    pts3d = world_points[conf_mask]
    pts_rgb_filtered = points_rgb[conf_mask]

    ply_path = sparse_dir / "points.ply"
    trimesh.PointCloud(pts3d, colors=pts_rgb_filtered).export(str(ply_path))
    logger.info(f"  Point cloud: {ply_path} ({len(pts3d)} points)")

    return n_points


# ── Save depth maps and camera poses ───────────────────────────────────

def save_depth_maps(
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    image_names: list,
    output_dir: Path,
    logger: logging.Logger,
):
    """Save per-frame depth maps as .npy files."""
    depths_dir = output_dir / "depths"
    depths_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(image_names):
        stem = Path(name).stem
        depth_path = depths_dir / f"{stem}_depth.npy"
        conf_path = depths_dir / f"{stem}_conf.npy"

        np.save(str(depth_path), depth_map[i].squeeze(-1))  # [H, W]
        np.save(str(conf_path), depth_conf[i])               # [H, W]

        logger.debug(f"  Saved depth: {depth_path.name}")

    logger.info(f"  Depth maps saved to {depths_dir}/ ({len(image_names)} files)")


def save_camera_poses(
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_names: list,
    output_dir: Path,
    logger: logging.Logger,
):
    """
    Save camera extrinsics as JSON.
    Format: {frame_name: {"extrinsic_4x4": [[...]], "intrinsic_3x3": [[...]]}}
    """
    from vggt.utils.geometry import closed_form_inverse_se3

    poses = {}
    for i, name in enumerate(image_names):
        # Build 4×4 from 3×4
        ext_3x4 = extrinsic[i]  # cam-from-world
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :] = ext_3x4

        # Also compute world-from-cam for convenience
        w2c = ext_4x4.copy()
        c2w = closed_form_inverse_se3(ext_4x4[None])[0]

        poses[name] = {
            "extrinsic_4x4": ext_4x4.tolist(),         # cam-from-world
            "world_to_cam_4x4": w2c.tolist(),           # same as above (alias)
            "cam_to_world_4x4": c2w.tolist(),           # inverse
            "intrinsic_3x3": intrinsic[i].tolist(),
        }

    json_path = output_dir / "camera_poses.json"
    with open(json_path, "w") as f:
        json.dump(poses, f, indent=2)
    logger.info(f"  Camera poses saved: {json_path}")


# ── Config loading ─────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except (ImportError, FileNotFoundError):
        return {}


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RoboScene+ Session 2: VGGT 3D Reconstruction"
    )
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="Directory containing extracted frames")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for VGGT results")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Max frames per inference batch (default: 20 for MPS, 30 for CUDA)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device for inference")
    parser.add_argument("--conf_threshold", type=float, default=1.5,
                        help="Depth confidence threshold for COLMAP points (adaptive if too high)")
    parser.add_argument("--max_points", type=int, default=100000,
                        help="Max 3D points for COLMAP reconstruction")
    parser.add_argument("--dry_run", action="store_true",
                        help="Process only the first 10 frames (diagnostic mode)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()

    # Resolve project root
    project_root = Path(__file__).resolve().parent.parent

    # Load config
    cfg = load_config(project_root / args.config)
    paths_cfg = cfg.get("paths", {})
    vggt_cfg = cfg.get("vggt", {})

    # Resolve parameters: CLI > config > defaults
    frames_dir = Path(args.frames_dir or paths_cfg.get("frames_dir", "data/frames"))
    if not frames_dir.is_absolute():
        frames_dir = project_root / frames_dir

    output_dir = Path(args.output_dir or paths_cfg.get("vggt_out", "data/vggt_out"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    device_str = args.device if args.device != "auto" else vggt_cfg.get("device", "auto")

    # Setup
    logger = setup_logging(project_root / "logs")
    logger.info("=" * 62)
    logger.info("  RoboScene+ — VGGT 3D Reconstruction")
    logger.info("=" * 62)

    # Device
    device = select_device(device_str, logger)
    dtype = select_dtype(device)
    logger.info(f"Dtype: {dtype}")

    # Default batch size: smaller for MPS to avoid memory issues
    default_batch = 20 if device.type == "mps" else int(vggt_cfg.get("batch_size", 30))
    batch_size = args.batch_size or default_batch
    logger.info(f"Batch size: {batch_size}")

    # Find frames
    image_paths = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
    if not image_paths:
        # Try PNG too
        image_paths = sorted(glob.glob(str(frames_dir / "frame_*.png")))
    if not image_paths:
        logger.error(f"No frames found in {frames_dir}")
        sys.exit(1)

    # Dry run: limit frames (5 for MPS to fit memory, 10 for CUDA)
    if args.dry_run:
        dry_limit = 5 if device.type == "mps" else 10
        image_paths = image_paths[:dry_limit]
        logger.info(f"🧪 DRY RUN: processing only first {len(image_paths)} frames")
        logger.info(f"  (limited to {dry_limit} for {device.type} memory safety)")

    image_names = [os.path.basename(p) for p in image_paths]
    total_frames = len(image_paths)
    logger.info(f"Frames: {total_frames} images from {frames_dir}")

    # ── Load model ─────────────────────────────────────────────────────
    model = load_model(device, logger)

    # ── Load and preprocess images ─────────────────────────────────────
    vggt_resolution = 518
    img_load_resolution = 1024

    logger.info(f"Loading images at {img_load_resolution}px (VGGT runs at {vggt_resolution}px)...")
    t_load = time.time()

    # Process in batches to avoid OOM during loading too
    all_extrinsics = []
    all_intrinsics = []
    all_depth_maps = []
    all_depth_confs = []
    all_world_points = []
    all_images = []
    all_original_coords = []

    num_batches = (total_frames + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_frames)
        batch_paths = image_paths[start:end]

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: "
                     f"frames {start + 1}–{end} ({len(batch_paths)} frames)")

        # Load images
        t_batch = time.time()
        images, original_coords = load_and_preprocess_images_square(
            batch_paths, img_load_resolution
        )
        images = images.to(device)
        original_coords = original_coords.to(device)

        logger.info(f"  Images loaded: {images.shape} in {time.time() - t_batch:.1f}s")

        # Run inference
        t_inf = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, world_points = \
            run_inference_batch(model, images, device, dtype, logger, vggt_resolution)
        inf_time = time.time() - t_inf
        logger.info(f"  Inference: {inf_time:.1f}s "
                     f"({inf_time / len(batch_paths):.2f}s/frame)")

        # Store results
        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_depth_maps.append(depth_map)
        all_depth_confs.append(depth_conf)
        all_world_points.append(world_points)
        all_images.append(images.cpu())
        all_original_coords.append(original_coords.cpu().numpy())

        # Clear GPU memory between batches
        del images, original_coords
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # Concatenate all batches
    logger.info("Concatenating batch results...")
    extrinsic = np.concatenate(all_extrinsics, axis=0)
    intrinsic = np.concatenate(all_intrinsics, axis=0)
    depth_map = np.concatenate(all_depth_maps, axis=0)
    depth_conf = np.concatenate(all_depth_confs, axis=0)
    world_points = np.concatenate(all_world_points, axis=0)
    images_all = torch.cat(all_images, dim=0)
    original_coords = np.concatenate(all_original_coords, axis=0)

    total_load_time = time.time() - t_load
    logger.info(f"Total inference time: {total_load_time:.1f}s "
                f"({total_load_time / total_frames:.2f}s/frame)")

    # ── Save outputs ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Depth maps
    logger.info("Saving depth maps...")
    save_depth_maps(depth_map, depth_conf, image_names, output_dir, logger)

    # 2. Camera poses JSON
    logger.info("Saving camera poses...")
    save_camera_poses(extrinsic, intrinsic, image_names, output_dir, logger)

    # 3. COLMAP sparse reconstruction
    logger.info("Exporting COLMAP sparse reconstruction...")
    t_colmap = time.time()
    export_colmap(
        extrinsic, intrinsic, depth_conf, world_points,
        images_all, image_names, original_coords,
        output_dir, vggt_resolution, img_load_resolution,
        args.conf_threshold, args.max_points, logger,
    )
    logger.info(f"  COLMAP export: {time.time() - t_colmap:.1f}s")

    # 4. Save reconstruction metadata
    meta = {
        "total_frames": total_frames,
        "image_names": image_names,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "vggt_resolution": vggt_resolution,
        "img_load_resolution": img_load_resolution,
        "conf_threshold": args.conf_threshold,
        "max_points": args.max_points,
        "dry_run": args.dry_run,
        "inference_time_sec": round(total_load_time, 2),
        "depth_shape": list(depth_map.shape),
        "extrinsic_shape": list(extrinsic.shape),
    }
    meta_path = output_dir / "vggt_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────
    sparse_dir = output_dir / "sparse"
    colmap_files = list(sparse_dir.glob("*")) if sparse_dir.exists() else []

    logger.info("")
    logger.info("=" * 62)
    logger.info("  VGGT RECONSTRUCTION COMPLETE")
    logger.info("=" * 62)
    logger.info(f"  Frames processed:    {total_frames}")
    logger.info(f"  Device:              {device} ({dtype})")
    logger.info(f"  Inference time:      {total_load_time:.1f}s total, "
                f"{total_load_time / total_frames:.2f}s/frame")
    logger.info(f"  Depth maps:          {output_dir / 'depths'}/")
    logger.info(f"  Camera poses:        {output_dir / 'camera_poses.json'}")
    logger.info(f"  COLMAP sparse:       {sparse_dir}/")
    logger.info(f"    Files:             {[f.name for f in colmap_files]}")
    logger.info(f"  Metadata:            {meta_path}")
    if args.dry_run:
        logger.info(f"  ⚠️  DRY RUN — only {total_frames} frames processed")
        logger.info(f"  Run without --dry_run for full reconstruction")
    else:
        logger.info(f"  ✅ Ready for Session 3 (Gaussian Splatting)")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()