#!/usr/bin/env python3
"""
paint_semantic_gaussians.py

Colors each Gaussian in the 3DGS splat by its semantic class.

Pipeline
--------
1. Load scene.ply (4.35M Gaussians, binary PLY, float32 properties)
2. For each of the 54 keyframes that have both a COLMAP pose and a
   semantic JSON:
   a. Decode SAM2 masks → label image (H×W uint8, 0=unlabeled, 1..N=class)
   b. Project all Gaussians onto the image plane (vectorized numpy)
   c. For each Gaussian that lands inside a labeled pixel: vote for that class
3. Assign winning class per Gaussian (min_votes threshold)
4. Modify f_dc_0/1/2 → class SH color; zero f_rest_* for labeled Gaussians
   Dim unlabeled Gaussians to 20% brightness
5. Write outputs/scene_semantic.ply + outputs/semantic_stats.json

Usage
-----
    python scripts/paint_semantic_gaussians.py
    python scripts/paint_semantic_gaussians.py --min_votes 1 --max_frames 10
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from colmap_utils import read_cameras_binary, read_images_binary, qvec_to_rotmat

try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[warn] pycocotools not found — will fall back to bbox masks")

# ---------------------------------------------------------------------------
# Semantic class definitions
# ---------------------------------------------------------------------------

# Class index 0 = unlabeled.  Indices 1..N = object classes.
CLASSES = [
    "bed", "desk", "chair", "laptop",
    "monitor", "fan", "lamp", "shelf",
    "door", "window",
]
LABEL_TO_IDX = {lbl: i + 1 for i, lbl in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# RGB colors in [0, 1]  (consistent with existing codebase palette)
CLASS_COLORS_HEX = {
    "bed":     "#FFB6C1",
    "desk":    "#20B2AA",
    "chair":   "#4682B4",
    "laptop":  "#6495ED",
    "monitor": "#2F6F6F",
    "fan":     "#FF6347",
    "lamp":    "#FFD700",
    "shelf":   "#CD853F",
    "door":    "#DEB887",
    "window":  "#87CEEB",
}


def hex_to_rgb01(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


CLASS_RGB = np.zeros((N_CLASSES + 1, 3), dtype=np.float32)  # index 0 = unused
for lbl, idx in LABEL_TO_IDX.items():
    CLASS_RGB[idx] = hex_to_rgb01(CLASS_COLORS_HEX.get(lbl, "#888888"))


# ---------------------------------------------------------------------------
# Coordinate transform helpers
# ---------------------------------------------------------------------------

def _rotation_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix R such that R @ v1 ≈ v2 (both unit vectors)."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(v1, v2))
    if s < 1e-10:
        return np.eye(3) if c > 0 else -np.eye(3)
    axis /= s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def compute_nerfstudio_transform(poses_dict: dict):
    """
    Reconstruct nerfstudio splatfacto's world-normalisation transform from
    the set of COLMAP world-to-camera poses.

    Nerfstudio applies (in order):
      1. Convert COLMAP (OpenCV, Y-down) → OpenGL (Y-up): negate R rows 1,2
      2. Rotate so mean camera-up aligns to [0,1,0]   (auto_orient, method='up')
      3. Centre at mean camera position                (center_method='poses')
      4. Scale so max |camera position component| = 1  (scale_factor)

    The full forward transform is:
        p_ns = scale * R_orient @ (p_colmap − t_center)

    Returns (R_orient, t_center, scale) so the caller can invert it.
    """
    R_list, t_list = zip(*poses_dict.values())   # R_w2c, t_w2c lists

    # Camera centres in COLMAP world space: c = −R_w2c^T @ t_w2c
    cam_centers = np.array([-R.T @ t for R, t in zip(R_list, t_list)])  # (C,3)

    # Camera "up" in OpenGL convention = −(second row of R_w2c)
    # Because: camera Y-axis in world (COLMAP) = R_c2w[:,1] = R_w2c[1,:]
    # nerfstudio negates Y to convert to OpenGL → up = −R_w2c[1,:]
    cam_ups = np.array([-R[1, :] for R in R_list])  # (C,3)

    mean_up = cam_ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)

    R_orient = _rotation_between(mean_up, np.array([0.0, 1.0, 0.0]))

    t_center = cam_centers.mean(axis=0)

    # Camera positions after rotation and centering
    c_centered = (R_orient @ (cam_centers - t_center).T).T  # (C,3)

    scale = 1.0 / float(np.max(np.abs(c_centered)))

    return R_orient, t_center, scale


# ---------------------------------------------------------------------------
# SH color helpers
# ---------------------------------------------------------------------------

SH_C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3) RGB in [0, 1] to SH degree-0 f_dc values."""
    return (rgb - 0.5) / SH_C0


# ---------------------------------------------------------------------------
# PLY I/O — pure numpy, no open3d
# ---------------------------------------------------------------------------

def read_3dgs_ply(path: Path):
    """
    Read a 3DGS PLY (binary little-endian, all float32 properties).
    Returns (data: structured ndarray, props: list[str], header_bytes: bytes).
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    # Find end_header
    eoh = raw.find(b"end_header\n")
    if eoh == -1:
        eoh = raw.find(b"end_header\r\n")
        header_bytes = raw[:eoh + 13]
    else:
        header_bytes = raw[:eoh + 11]

    header_str = header_bytes.decode("ascii", errors="replace")

    n_verts = 0
    props = []
    for line in header_str.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[2])

    dt = np.dtype([(p, "f4") for p in props])
    body = raw[len(header_bytes):]
    data = np.frombuffer(body, dtype=dt).copy()  # writable copy

    return data, props, header_bytes


def write_3dgs_ply(path: Path, data: np.ndarray, header_bytes: bytes):
    """Write modified structured array back with the original header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(data.tobytes())
    print(f"[OK] Written {len(data):,} Gaussians → {path} "
          f"({path.stat().st_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def decode_rle(rle_dict: dict, h: int, w: int) -> np.ndarray | None:
    """Decode COCO RLE to (H, W) uint8. Returns None on failure."""
    if not HAS_COCO or not rle_dict:
        return None
    counts = rle_dict.get("counts")
    size   = rle_dict.get("size", [h, w])
    try:
        if isinstance(counts, list):
            rle = {"counts": counts, "size": size}
            decoded = coco_mask.decode(coco_mask.frPyObjects(rle, size[0], size[1]))
        else:
            decoded = coco_mask.decode({"counts": counts, "size": size})
        return decoded.astype(np.uint8)
    except Exception:
        return None


def bbox_to_mask(bbox, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in bbox)
    x2 = min(x + bw, w); y2 = min(y + bh, h)
    x = max(x, 0);       y = max(y, 0)
    if x2 > x and y2 > y:
        mask[y:y2, x:x2] = 1
    return mask


def build_label_image(sem_data: dict, h: int, w: int,
                      conf_threshold: float = 0.0) -> np.ndarray:
    """
    Build an (H, W) uint8 label image from a semantic JSON dict.
    Pixels are set to the class index (1-based).

    Painting order: large-area objects first, small objects last.
    This ensures compact objects (monitor, lamp) overwrite the broad
    background masks they sit on (desk, shelf), regardless of confidence.
    Within the same area, higher-confidence labels paint last (win).
    """
    label_img = np.zeros((h, w), dtype=np.uint8)

    # Decode all masks first so we can sort by area
    items = []
    for lbl, ann in sem_data.items():
        if lbl not in LABEL_TO_IDX:
            continue
        conf = float(ann.get("confidence", 0.0))
        if conf < conf_threshold:
            continue
        mask = None
        if "mask_rle" in ann and ann["mask_rle"]:
            rle   = ann["mask_rle"]
            rle_h = rle.get("size", [h, w])[0]
            rle_w = rle.get("size", [h, w])[1]
            mask  = decode_rle(rle, rle_h, rle_w)
            # Resize mask to camera resolution if RLE was encoded at different size
            if mask is not None and (rle_h != h or rle_w != w):
                from PIL import Image as _PIL
                mask = np.array(
                    _PIL.fromarray(mask).resize((w, h), _PIL.NEAREST)
                )
                # Also rescale bbox coordinates if present
                if "bbox" in ann and ann["bbox"]:
                    sx, sy = w / rle_w, h / rle_h
                    b = ann["bbox"]
                    ann = dict(ann)  # don't mutate original
                    ann["bbox"] = [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
        if mask is None or mask.sum() == 0:
            if "bbox" in ann and ann["bbox"]:
                mask = bbox_to_mask(ann["bbox"], h, w)
        if mask is not None and mask.sum() > 0:
            items.append((int(mask.sum()), conf, lbl, mask))

    # Paint large objects first; small objects overwrite on overlap.
    # Tie-break by confidence ascending so high-conf wins within same size.
    items.sort(key=lambda t: (-t[0], t[1]))

    for _, _, lbl, mask in items:
        label_img[mask > 0] = LABEL_TO_IDX[lbl]

    return label_img


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paint 3DGS Gaussians with semantic class colors."
    )
    parser.add_argument("--splat_ply",
        default=str(ROOT / "outputs/splat_mast3r_v2/scene.ply"))
    parser.add_argument("--semantic_dir",
        default=str(ROOT / "outputs/semantic"))
    parser.add_argument("--cameras_bin",
        default=str(ROOT / "data/mast3r_out/sparse/0/cameras.bin"))
    parser.add_argument("--images_bin",
        default=str(ROOT / "data/mast3r_out/sparse/0/images.bin"))
    parser.add_argument("--output_ply",
        default=str(ROOT / "outputs/scene_semantic.ply"))
    parser.add_argument("--stats_json",
        default=str(ROOT / "outputs/semantic_stats.json"))
    parser.add_argument("--min_votes", type=int, default=2,
        help="Minimum cross-frame votes to assign a label (default 2)")
    parser.add_argument("--max_frames", type=int, default=None,
        help="Process at most this many frames (for quick testing)")
    parser.add_argument("--conf_threshold", type=float, default=0.30,
        help="Minimum SAM2 confidence to use a mask (default 0.30)")
    parser.add_argument("--dim_factor", type=float, default=0.20,
        help="Brightness multiplier for unlabeled Gaussians (default 0.20)")
    args = parser.parse_args()

    # ── 1. Load COLMAP camera poses ──────────────────────────────────────
    print("[...] Loading COLMAP cameras + poses...")
    cameras = read_cameras_binary(args.cameras_bin)
    cam     = next(iter(cameras.values()))
    params  = cam["params"]
    img_w   = int(cam["width"])
    img_h   = int(cam["height"])
    if len(params) >= 4:
        fx, fy = float(params[0]), float(params[1])
        cx, cy = float(params[2]), float(params[3])
    else:
        fx = fy = float(params[0])
        cx, cy  = float(params[1]), float(params[2])

    print(f"[OK] Camera: {img_w}×{img_h}  "
          f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    colmap_images = read_images_binary(args.images_bin)
    poses = {}  # stem → (R_w2c 3×3, t_w2c 3)
    for img in colmap_images.values():
        stem = Path(img["name"]).stem
        poses[stem] = (
            qvec_to_rotmat(img["qvec"]).astype(np.float64),
            img["tvec"].astype(np.float64),
        )
    print(f"[OK] {len(poses)} COLMAP keyframe poses loaded")

    # ── 1b. Reconstruct nerfstudio normalisation transform ───────────────
    # scene.ply Gaussian XYZs are in nerfstudio's normalised space:
    #   p_ns = scale * R_orient @ (p_colmap − t_center)
    # Invert to get COLMAP world-space coords for projection.
    R_orient, t_center, ns_scale = compute_nerfstudio_transform(poses)
    print(f"[OK] Nerfstudio transform: scale={ns_scale:.4f}  "
          f"t_center=[{t_center[0]:.3f}, {t_center[1]:.3f}, {t_center[2]:.3f}]")
    print(f"     mean_up (pre-orient) = {(-np.array([R for R, _ in poses.values()])[:, 1, :]).mean(axis=0)}")

    # ── 2. Match semantic JSONs to poses ────────────────────────────────
    sem_dir = Path(args.semantic_dir)
    all_jsons = sorted(sem_dir.glob("frame_*.json"))
    matched = [p for p in all_jsons if p.stem in poses]
    if args.max_frames:
        matched = matched[:args.max_frames]
    print(f"[OK] {len(matched)} frames with pose + semantic JSON")
    if not matched:
        sys.exit("[error] No matched frames. Check paths.")

    # ── 3. Load Gaussian splat ───────────────────────────────────────────
    print(f"[...] Loading Gaussian splat: {args.splat_ply}")
    splat_path = Path(args.splat_ply)
    if not splat_path.exists():
        sys.exit(f"[error] splat PLY not found: {splat_path}")

    gaussians, props, header_bytes = read_3dgs_ply(splat_path)
    N = len(gaussians)
    print(f"[OK] Loaded {N:,} Gaussians  ({len(props)} properties)")

    # Extract xyz as float64 for projection precision
    xyz_ns = np.stack([
        gaussians["x"].astype(np.float64),
        gaussians["y"].astype(np.float64),
        gaussians["z"].astype(np.float64),
    ], axis=1)  # (N, 3) — in nerfstudio normalised space

    # ── 3b. Convert Gaussian XYZs to COLMAP world space ─────────────────
    # Inverse of:  p_ns = scale * R_orient @ (p_colmap − t_center)
    # So:          p_colmap = R_orient.T @ (p_ns / scale) + t_center
    xyz = (R_orient.T @ (xyz_ns.T / ns_scale)).T + t_center  # (N, 3)

    xyz_min = xyz.min(axis=0); xyz_max = xyz.max(axis=0)
    print(f"[OK] Gaussians converted to COLMAP space")
    print(f"     X=[{xyz_min[0]:.2f}, {xyz_max[0]:.2f}]  "
          f"Y=[{xyz_min[1]:.2f}, {xyz_max[1]:.2f}]  "
          f"Z=[{xyz_min[2]:.2f}, {xyz_max[2]:.2f}]")

    # Sanity: camera centres should lie within or near the Gaussian cloud
    cam_centers_check = np.array([-R.T @ t for R, t in poses.values()])
    print(f"     Camera centres X=[{cam_centers_check[:,0].min():.2f}, {cam_centers_check[:,0].max():.2f}]  "
          f"Y=[{cam_centers_check[:,1].min():.2f}, {cam_centers_check[:,1].max():.2f}]  "
          f"Z=[{cam_centers_check[:,2].min():.2f}, {cam_centers_check[:,2].max():.2f}]")

    # ── 4. Vote for semantic class per Gaussian ──────────────────────────
    # votes[i, c] = number of frames where Gaussian i projected into class c
    # c=0 means unlabeled (we skip those)
    votes = np.zeros((N, N_CLASSES + 1), dtype=np.int16)

    print(f"[...] Projecting {N:,} Gaussians across {len(matched)} frames...")
    for frame_idx, sem_path in enumerate(matched):
        stem = sem_path.stem
        R_w2c, t_w2c = poses[stem]

        # Load semantic masks → label image
        with open(sem_path) as f:
            sem_data = json.load(f)
        if not sem_data:
            continue
        label_img = build_label_image(sem_data, img_h, img_w,
                                       conf_threshold=args.conf_threshold)
        if label_img.max() == 0:
            continue

        # Project all Gaussians into this camera (fully vectorized)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # p_cam: (N, 3)
            p_cam = (R_w2c @ xyz.T).T + t_w2c

        # Only keep points in front of camera
        in_front = p_cam[:, 2] > 0.05

        # Perspective divide
        z = p_cam[:, 2]
        u = fx * p_cam[:, 0] / z + cx   # (N,)
        v = fy * p_cam[:, 1] / z + cy   # (N,)

        u_i = np.round(u).astype(np.int32)
        v_i = np.round(v).astype(np.int32)

        in_bounds = (in_front
                     & (u_i >= 0) & (u_i < img_w)
                     & (v_i >= 0) & (v_i < img_h))

        valid_idx = np.where(in_bounds)[0]          # Gaussian indices
        u_v = u_i[valid_idx]
        v_v = v_i[valid_idx]

        # Look up label at each projected pixel
        labels_hit = label_img[v_v, u_v]            # (M,) uint8

        # Only vote for labeled pixels (label > 0)
        labeled_mask = labels_hit > 0
        if labeled_mask.sum() == 0:
            continue

        np.add.at(votes,
                  (valid_idx[labeled_mask], labels_hit[labeled_mask]),
                  1)

        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            total_votes = (votes > 0).any(axis=1).sum()
            pct = 100.0 * total_votes / N
            print(f"  [{frame_idx+1:3d}/{len(matched)}] {stem}  "
                  f"labeled so far: {total_votes:,} ({pct:.1f}%)", flush=True)

    print("[OK] Voting complete")

    # ── 5. Assign labels and apply colors ────────────────────────────────
    # For each Gaussian: winning class = argmax over classes 1..N_CLASSES
    total_vote_counts = votes[:, 1:].sum(axis=1)   # (N,) total labeled votes
    winning_class_rel = np.argmax(votes[:, 1:], axis=1) + 1  # 1-indexed, (N,)

    is_labeled = total_vote_counts >= args.min_votes  # (N,) bool
    semantic_class = np.where(is_labeled, winning_class_rel, 0)  # 0=unlabeled

    n_labeled = is_labeled.sum()
    print(f"\n[OK] {n_labeled:,} / {N:,} Gaussians labeled "
          f"({100.0 * n_labeled / N:.1f}%)")

    # Per-class counts
    stats = {"total_gaussians": int(N), "labeled": int(n_labeled),
             "unlabeled": int(N - n_labeled), "classes": {}}
    for lbl, idx in sorted(LABEL_TO_IDX.items(), key=lambda x: x[1]):
        cnt = int((semantic_class == idx).sum())
        stats["classes"][lbl] = {"count": cnt,
                                  "pct": round(100.0 * cnt / N, 2)}
        if cnt > 0:
            print(f"  {lbl:<12} {cnt:>8,}  ({100.0*cnt/N:.2f}%)")

    # ── 6. Modify Gaussian colors in-place ──────────────────────────────
    print("[...] Applying semantic colors...")

    # Identify SH property column indices
    f_dc_cols  = ["f_dc_0", "f_dc_1", "f_dc_2"]
    f_rest_cols = [p for p in props if p.startswith("f_rest_")]

    # Verify all expected columns exist
    for col in f_dc_cols:
        if col not in props:
            sys.exit(f"[error] Expected property '{col}' not found in PLY. "
                     f"Available: {props}")

    # --- Apply to labeled Gaussians ---
    # Build arrays of target f_dc values for each labeled Gaussian
    labeled_indices = np.where(is_labeled)[0]
    target_rgb = CLASS_RGB[semantic_class[labeled_indices]]   # (M, 3) float32
    target_sh  = rgb_to_sh_dc(target_rgb.astype(np.float32)) # (M, 3)

    gaussians["f_dc_0"][labeled_indices] = target_sh[:, 0]
    gaussians["f_dc_1"][labeled_indices] = target_sh[:, 1]
    gaussians["f_dc_2"][labeled_indices] = target_sh[:, 2]

    # Zero f_rest for labeled Gaussians → view-independent flat class color
    for col in f_rest_cols:
        gaussians[col][labeled_indices] = 0.0

    # --- Dim unlabeled Gaussians so labeled objects pop out ---
    unlabeled_indices = np.where(~is_labeled)[0]
    if args.dim_factor < 1.0 and len(unlabeled_indices) > 0:
        for col in f_dc_cols:
            # Dimming in SH space: dc controls base brightness.
            # We push f_dc toward the "0.5 → grey" neutral value.
            # Blend: dc_new = dc_old * dim + 0 * (1 - dim)
            gaussians[col][unlabeled_indices] *= args.dim_factor

    print(f"[OK] Colors applied  "
          f"(labeled {len(labeled_indices):,}, "
          f"dimmed unlabeled {len(unlabeled_indices):,})")

    # ── 7. Write output PLY ──────────────────────────────────────────────
    print(f"[...] Writing {args.output_ply} ...")
    write_3dgs_ply(Path(args.output_ply), gaussians, header_bytes)

    # ── 8. Write stats JSON ──────────────────────────────────────────────
    with open(args.stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Stats → {args.stats_json}")

    print("\n=== Done ===")
    print(f"Semantic splat : {args.output_ply}")
    print(f"Stats          : {args.stats_json}")
    print(f"Labeled        : {n_labeled:,} / {N:,} Gaussians "
          f"({100.0*n_labeled/N:.1f}%)")
    print(f"\nNext: create app/static/viewer.html to view the semantic splat.")


if __name__ == "__main__":
    main()
