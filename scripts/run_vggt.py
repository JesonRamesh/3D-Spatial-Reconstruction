"""Run VGGT on extracted frames to produce camera_poses.json and per-frame depth maps."""

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
from PIL import Image as PILImage
from typing import List, Tuple

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


def resize_frames_to_resolution(
    image_paths: list,
    resolution: int,
    logger: logging.Logger,
) -> tuple:
    """
    Load frames with PIL and resize to (resolution, resolution) using LANCZOS.
    Returns:
        images_tensor: [S, 3, resolution, resolution] float32 in [0, 1]
        original_wh:   list of (width, height) tuples — original frame dimensions
    """
    import torchvision.transforms.functional as TVF

    tensors = []
    original_wh = []

    for p in image_paths:
        img = PILImage.open(p).convert("RGB")
        original_wh.append(img.size)  # (width, height) before resize
        img_resized = img.resize((resolution, resolution), PILImage.LANCZOS)
        tensors.append(TVF.to_tensor(img_resized))  # [3, H, W] in [0,1]

    images_tensor = torch.stack(tensors)  # [S, 3, resolution, resolution]
    logger.debug(f"  Resized {len(image_paths)} frames to {resolution}×{resolution} (LANCZOS)")
    return images_tensor, original_wh


def scale_intrinsics(
    intrinsic: np.ndarray,
    original_wh: list,
    resolution: int,
) -> np.ndarray:
    """
    Scale camera intrinsics from VGGT's internal resolution back to the
    original frame dimensions using UNIFORM scaling.

    CRITICAL FIX (was: non-uniform scaling):
    =========================================
    The original code used scale_x = W/res and scale_y = H/res, which
    produces fx != fy for non-square images (e.g. 4032x3024 -> 518x518
    gives scale_x=7.784, scale_y=5.838 -> fx/fy=1.333).

    Physical cameras have square pixels, so fx MUST equal fy.
    VGGT operates on a square-padded image and estimates a single
    symmetric focal length. We must scale it uniformly using the
    LARGER dimension (max(W, H) / resolution) so the focal correctly
    represents the longer axis of the original sensor.

    The principal point (cx, cy) is set to the image centre (W/2, H/2)
    regardless of VGGT's estimate, because cx/cy are unreliable from
    a monocular neural estimator.

    Args:
        intrinsic:   [S, 3, 3] at VGGT resolution (resolution×resolution)
        original_wh: list of S (width, height) tuples
        resolution:  VGGT input resolution (e.g. 518)

    Returns:
        intrinsic_scaled: [S, 3, 3] with fx=fy scaled to original image dims
    """
    intrinsic_scaled = intrinsic.copy()
    for i, (w, h) in enumerate(original_wh):
        # UNIFORM scale: use the max dimension so focal correctly represents
        # the longer sensor axis. This preserves fx == fy (square pixels).
        scale = max(w, h) / resolution
        fx_uniform = intrinsic[i, 0, 0] * scale   # was VGGT-internal fx
        fy_uniform = fx_uniform                     # enforce square pixels
        cx_correct = w / 2.0                        # principal pt at centre
        cy_correct = h / 2.0

        intrinsic_scaled[i, 0, 0] = fx_uniform
        intrinsic_scaled[i, 1, 1] = fy_uniform
        intrinsic_scaled[i, 0, 2] = cx_correct
        intrinsic_scaled[i, 1, 2] = cy_correct
    return intrinsic_scaled


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
        images: [S, 3, resolution, resolution] tensor in [0, 1]
                already resized to vggt_resolution before calling this
        device: torch device
        dtype: torch dtype for autocast
        vggt_resolution: VGGT input resolution (matches images spatial dims)

    Returns:
        extrinsic: [S, 3, 4] numpy  (cam-from-world, OpenCV convention)
        intrinsic: [S, 3, 3] numpy  (at vggt_resolution scale — caller scales)
        depth_map: [S, H, W, 1] numpy
        depth_conf: [S, H, W] numpy
        world_points: [S, H, W, 3] numpy
    """
    S = images.shape[0]
    logger.debug(f"  Inference on {S} frames, resolution={vggt_resolution}")

    # Images are already at vggt_resolution — pass directly
    images_518 = images  # naming kept for minimal diff; shape [S,3,res,res]

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


# ── Pose stitching across batch boundaries ────────────────────────────

def _stitch_batch_poses(
    extrinsic_new: np.ndarray,
    extrinsic_prev: list,
    n_overlap: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Align a new batch of extrinsics into the global coordinate frame
    using shared overlap frames between consecutive batches.

    Problem solved:
    ---------------
    VGGT estimates poses relative to the first frame of each batch
    independently. Without stitching, batch k+1 is in a completely
    different coordinate frame from batch k, causing 180-degree flips
    and disconnected local maps in the reconstruction.

    Method (Umeyama / SVD Procrustes):
    -----------------------------------
    The first n_overlap frames of extrinsic_new are the same physical
    frames as the last n_overlap frames already stored in extrinsic_prev.
    We compute the similarity transform T_align (scale + rotation +
    translation) that maps new-batch camera positions onto the previously
    registered positions, then apply T_align to all frames in the batch.

    Args:
        extrinsic_new  : [B, 3, 4] cam-from-world for the new batch
        extrinsic_prev : list of already-registered [Bi, 3, 4] arrays
        n_overlap      : overlap frames at start of new batch
        logger         : for progress messages

    Returns:
        [B, 3, 4] extrinsics with new batch aligned to global frame
    """
    prev_all     = np.concatenate(extrinsic_prev, axis=0)  # [N_prev, 3, 4]
    prev_overlap = prev_all[-n_overlap:]                   # [n_ov, 3, 4]
    new_overlap  = extrinsic_new[:n_overlap]               # [n_ov, 3, 4]

    def to_4x4(ext34: np.ndarray) -> np.ndarray:
        """[B, 3, 4] -> [B, 4, 4]"""
        B   = ext34.shape[0]
        out = np.tile(np.eye(4), (B, 1, 1))
        out[:, :3, :] = ext34
        return out

    def cam_positions(ext44: np.ndarray) -> np.ndarray:
        """[B, 4, 4] cam-from-world -> [B, 3] world positions = -R^T @ t"""
        R = ext44[:, :3, :3]
        t = ext44[:, :3,  3]
        return np.einsum('bij,bj->bi', R.transpose(0, 2, 1), -t)

    P = to_4x4(prev_overlap)    # [n_ov, 4, 4] registered
    N = to_4x4(new_overlap)     # [n_ov, 4, 4] new (unregistered)

    pos_prev = cam_positions(P)  # [n_ov, 3]
    pos_new  = cam_positions(N)  # [n_ov, 3]

    # SVD-based Procrustes (Umeyama) to find scale, R_align, t_align such that
    # pos_prev ≈ scale * R_align @ pos_new + t_align
    mu_prev = pos_prev.mean(axis=0)
    mu_new  = pos_new.mean(axis=0)
    prev_c  = pos_prev - mu_prev
    new_c   = pos_new  - mu_new

    var_new = float((new_c ** 2).sum()) / n_overlap
    if var_new < 1e-12:
        logger.warning("  Overlap frames zero-variance — skipping stitching for this batch")
        return extrinsic_new

    H             = new_c.T @ prev_c              # [3, 3]
    U, S, Vt      = np.linalg.svd(H)
    d             = np.linalg.det(Vt.T @ U.T)
    D             = np.diag([1., 1., float(np.sign(d))])
    R_align       = Vt.T @ D @ U.T               # [3, 3] rotation
    scale         = float(S.sum()) / var_new / n_overlap
    t_align       = mu_prev - scale * (R_align @ mu_new)   # [3]

    # Log alignment quality
    aligned_check = scale * (R_align @ pos_new.T).T + t_align
    err           = float(np.linalg.norm(aligned_check - pos_prev, axis=1).mean())
    logger.info(f"  Batch stitch: scale={scale:.4f}  mean_pos_err={err:.5f}  "
                f"over {n_overlap} overlap frames")
    if err > 0.5:
        logger.warning(f"  ⚠️  Large stitch error ({err:.3f}) — "
                       f"overlap frames may not share the same physical views")

    # Build the 4x4 alignment transform T and its inverse.
    # T maps old-world -> new-world: p_new = scale * R_align @ p_old + t_align
    T_align          = np.eye(4)
    T_align[:3, :3]  = scale * R_align
    T_align[:3,  3]  = t_align

    T_inv            = np.eye(4)
    T_inv[:3, :3]    = (1.0 / scale) * R_align.T
    T_inv[:3,  3]    = -(1.0 / scale) * (R_align.T @ t_align)

    # Apply: E_aligned = E_new @ T_inv
    # (cam-from-world_aligned = cam-from-world_new @ world_old-from-world_aligned)
    new_all = to_4x4(extrinsic_new)      # [B, 4, 4]
    aligned = new_all @ T_inv[None]      # [B, 4, 4]  (T_inv broadcast)
    return aligned[:, :3, :]             # [B, 3, 4]


# ── Frame similarity sorting ──────────────────────────────────────────

def sort_frames_by_similarity(
    frame_paths: List[str],
) -> Tuple[List[str], List[int]]:
    """
    Reorder *frame_paths* so that visually similar frames are adjacent,
    reducing discontinuities at VGGT batch boundaries.

    Algorithm
    ---------
    1. Load every frame as a 64×64 grayscale thumbnail (cheap I/O).
    2. Compute a normalised 64-bin intensity histogram per thumbnail.
    3. Choose an anchor: the frame whose histogram is closest (L2) to
       the global mean histogram — a 'central' viewpoint that avoids
       starting at an outlier.
    4. Greedy nearest-neighbour traversal (O(N²)) using histogram
       correlation as the similarity metric.  At each step the
       correlation against all unvisited histograms is computed in one
       vectorised matrix multiply, so no Python loop over N frames.

    Parameters
    ----------
    frame_paths : list of str
        Unsorted image file paths.

    Returns
    -------
    sorted_paths : List[str]
        Paths reordered by visual similarity.
    order : List[int]
        Permutation indices: ``sorted_paths[i] == frame_paths[order[i]]``.
    """
    n = len(frame_paths)
    if n == 0:
        return [], []
    if n == 1:
        return list(frame_paths), [0]

    thumb_size = (64, 64)
    n_bins = 64
    print(f"[sort_frames] Loading {n} thumbnails …")

    histograms: List[np.ndarray] = []
    for path in frame_paths:
        try:
            img = PILImage.open(path).convert("L").resize(thumb_size, PILImage.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            hist, _ = np.histogram(arr.ravel(), bins=n_bins, range=(0.0, 256.0))
            total = float(hist.sum())
            histograms.append(hist.astype(np.float32) / total if total > 0
                               else hist.astype(np.float32))
        except Exception as exc:  # noqa: BLE001
            print(f"[sort_frames] Warning: cannot load '{path}': {exc}. "
                  "Using zero histogram.")
            histograms.append(np.zeros(n_bins, dtype=np.float32))

    H = np.stack(histograms)          # (N, 64)

    # Anchor: frame nearest the mean histogram
    mean_hist = H.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(H - mean_hist, axis=1)))

    # Greedy nearest-neighbour using histogram correlation
    # corr(a, b) = dot(a - mean_a, b - mean_b) / (||a - mean_a|| * ||b - mean_b||)
    visited = np.zeros(n, dtype=bool)
    order: List[int] = []
    current = start
    visited[current] = True
    order.append(current)

    H_c = H - H.mean(axis=1, keepdims=True)   # centred histograms (N, 64)

    for step in range(1, n):
        c_vec = H_c[current]                   # (64,)
        num   = H_c @ c_vec                    # (N,) — vectorised dot products
        denom = (np.sqrt((H_c ** 2).sum(axis=1)) *
                 float(np.sqrt((c_vec ** 2).sum())))   # (N,)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 1e-10, num / denom, 0.0)
        corr[visited] = -np.inf               # mask already-visited
        current = int(np.argmax(corr))
        visited[current] = True
        order.append(current)
        if step % 100 == 0:
            print(f"[sort_frames]   {step}/{n} frames ordered")

    sorted_paths = [frame_paths[i] for i in order]
    print(f"[sort_frames] Done — anchor was original index {start} "
          f"({os.path.basename(frame_paths[start])})")
    return sorted_paths, order


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
    parser.add_argument("--resolution", type=int, default=518,
                        help="Resize frames to this square resolution before VGGT "
                             "inference (default: 518, VGGT's training resolution). "
                             "Lower values reduce VRAM at some quality cost.")
    parser.add_argument("--overlap", type=int, default=5,
                        help="Number of frames to overlap between consecutive batches "
                             "for pose stitching (default: 5). Set to 0 to disable. "
                             "Overlap frames are used to compute a rigid alignment "
                             "between adjacent batches, eliminating 180-degree flips "
                             "at batch boundaries.")
    parser.add_argument(
        "--sort_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sort frames by visual similarity before batching to reduce "
             "discontinuities at batch boundaries (default: True). "
             "Use --no-sort_frames to disable.",
    )
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

    # ── Optional similarity-based frame reordering ────────────────────
    # Problem: VGGT processes frames in independent batches. With 511
    # still photos the default filename order may place visually
    # dissimilar shots in consecutive batches, causing 8.8× translation
    # jumps at boundaries. Sorting so that adjacent frames share
    # overlapping viewpoints maximises the signal available to the
    # overlap-based pose stitcher (_stitch_batch_poses).
    if args.sort_frames and not args.dry_run:
        logger.info("[sort_frames] Sorting frames by visual similarity "
                    "(--no-sort_frames to skip) …")
        original_paths = list(image_paths)
        image_paths, order = sort_frames_by_similarity(image_paths)

        # Persist the reordering map for downstream traceability
        frame_order_map = {
            "sorted_paths": image_paths,
            "original_paths": original_paths,
            "sorted_to_original_index": {
                str(si): int(oi) for si, oi in enumerate(order)
            },
            "original_to_sorted_index": {
                str(oi): int(si) for si, oi in enumerate(order)
            },
        }
        frame_order_path = output_dir / "frame_order.json"
        frame_order_path.parent.mkdir(parents=True, exist_ok=True)
        with open(frame_order_path, "w") as fh:
            json.dump(frame_order_map, fh, indent=2)
        logger.info(f"[sort_frames] Reordering map → {frame_order_path}")
    elif args.sort_frames and args.dry_run:
        logger.info("[sort_frames] Skipping similarity sort in dry-run mode.")
    else:
        logger.info("[sort_frames] Frame sorting disabled (--no-sort_frames).")

    image_names = [os.path.basename(p) for p in image_paths]
    total_frames = len(image_paths)
    logger.info(f"Frames: {total_frames} images from {frames_dir}")

    # ── Load model ─────────────────────────────────────────────────────
    model = load_model(device, logger)

    # ── Load and preprocess images ─────────────────────────────────────
    vggt_resolution = args.resolution
    img_load_resolution = vggt_resolution  # we load directly at target resolution
    overlap = args.overlap

    logger.info(f"Input resolution: {vggt_resolution}×{vggt_resolution} "
                f"(VGGT training res=518, current={'✅ native' if vggt_resolution == 518 else '⚠️ rescaled'})")
    logger.info(f"Batch overlap: {overlap} frames "
                f"({'enabled — will stitch poses across boundaries' if overlap > 0 else 'disabled'})")
    t_load = time.time()

    # Process in batches to avoid OOM during loading too
    all_extrinsics = []
    all_intrinsics = []
    all_depth_maps = []
    all_depth_confs = []
    all_world_points = []
    all_images = []
    all_original_coords = []

    # Build batch slice list with optional overlap.
    # Each entry is (start_frame_idx, end_frame_idx, n_overlap_frames_at_start).
    # Overlap frames at the START of a batch are shared with the previous batch
    # so we can compute a rigid alignment to stitch poses globally.
    stride = max(batch_size - overlap, 1) if overlap > 0 else batch_size
    batch_slices = []
    pos = 0
    while pos < total_frames:
        end_pos = min(pos + batch_size, total_frames)
        n_ov = overlap if pos > 0 else 0  # first batch has no overlap frames
        batch_slices.append((pos, end_pos, n_ov))
        pos += stride
        if end_pos == total_frames:
            break
    num_batches = len(batch_slices)
    logger.info(f"Batch plan: {num_batches} batches, stride={stride}, overlap={overlap}")

    for batch_idx, (start, end, n_ov) in enumerate(batch_slices):
        batch_paths = image_paths[start:end]

        logger.info(f"Batch {batch_idx + 1}/{num_batches}: "
                     f"frames {start + 1}–{end} ({len(batch_paths)} frames, "
                     f"{n_ov} overlap at start)")

        # Load and resize images using PIL LANCZOS
        t_batch = time.time()
        images, original_wh_batch = resize_frames_to_resolution(
            batch_paths, vggt_resolution, logger
        )
        images = images.to(device)

        # original_coords in load_and_preprocess_images_square format:
        # [x1, y1, x2, y2, width, height] — here images fill the full square
        # so we build a compatible array from original_wh
        original_coords_batch = np.array([
            [0.0, 0.0, float(vggt_resolution), float(vggt_resolution),
             float(w), float(h)]
            for w, h in original_wh_batch
        ], dtype=np.float32)

        logger.info(f"  Images loaded: {images.shape} in {time.time() - t_batch:.1f}s")

        # Run inference
        t_inf = time.time()
        extrinsic, intrinsic, depth_map, depth_conf, world_points = \
            run_inference_batch(model, images, device, dtype, logger, vggt_resolution)
        inf_time = time.time() - t_inf
        logger.info(f"  Inference: {inf_time:.1f}s "
                     f"({inf_time / len(batch_paths):.2f}s/frame)")

        # Scale intrinsics from vggt_resolution back to original frame dimensions
        intrinsic = scale_intrinsics(intrinsic, original_wh_batch, vggt_resolution)

        # ── Stitch poses using overlap frames ─────────────────────────
        # VGGT estimates poses relative to the first frame of each batch.
        # Consecutive batches are therefore in different coordinate frames.
        # With overlap > 0, the first n_ov frames of this batch are the
        # same physical frames as the last n_ov frames of the previous batch.
        # We compute the rigid transform (SE3) that maps the current batch's
        # frame 0..n_ov-1 poses onto the corresponding already-registered
        # poses from the previous batch, then apply it to all frames in
        # this batch. This globally aligns all batches in one coordinate frame.
        if overlap > 0 and n_ov > 0 and len(all_extrinsics) > 0:
            extrinsic = _stitch_batch_poses(
                extrinsic_new=extrinsic,
                extrinsic_prev=all_extrinsics,
                n_overlap=n_ov,
                logger=logger,
            )

        # Store results (for overlapping batches, drop the overlap frames at the
        # start since they were already stored in the previous batch)
        drop = n_ov  # frames already stored from previous batch
        all_extrinsics.append(extrinsic[drop:])
        all_intrinsics.append(intrinsic[drop:])
        all_depth_maps.append(depth_map[drop:])
        all_depth_confs.append(depth_conf[drop:])
        all_world_points.append(world_points[drop:])
        all_images.append(images[drop:].cpu())
        all_original_coords.append(original_coords_batch[drop:])

        # Clear GPU memory between batches
        del images
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