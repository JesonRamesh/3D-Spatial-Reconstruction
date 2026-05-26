"""Project SAM2 semantic masks onto the dense MASt3R-SLAM point cloud.

Input:
  --pointcloud   outputs/mast3r_out_v2/dense_pointcloud.ply  (14.9M pts, XYZ+RGB)
  --semantic_dir outputs/semantic_mast3r/                    (101 frame JSONs)
  --cameras_bin  data/mast3r_out_v2/sparse/0/cameras.bin
  --images_bin   data/mast3r_out_v2/sparse/0/images.bin
  --output_ply   outputs/mast3r_out_v2/semantic_pointcloud.ply
  --web_ply      outputs/scene_pointcloud_web.ply            (downsampled for viewer)
  --n_web        2500000

The point cloud is in MASt3R-SLAM world space (Y-down).
The COLMAP poses (from gsplat_colmap_dataset.py) are in the same space.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from colmap_utils import read_cameras_binary, read_images_binary

try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[warn] pycocotools not found — will fall back to bbox masks")

# ---------------------------------------------------------------------------
# Semantic class definitions  (match paint_semantic_gaussians.py)
# ---------------------------------------------------------------------------

CLASSES = [
    "bed", "desk", "chair", "laptop",
    "monitor", "fan", "lamp", "shelf",
    "door", "window",
]
LABEL_TO_IDX = {lbl: i + 1 for i, lbl in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

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

def _hex_to_rgb_u8(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)

# class_rgb_u8[0] = unused, [1..N] = class colors
CLASS_RGB_U8 = np.zeros((N_CLASSES + 1, 3), dtype=np.uint8)
for lbl, idx in LABEL_TO_IDX.items():
    CLASS_RGB_U8[idx] = _hex_to_rgb_u8(CLASS_COLORS_HEX.get(lbl, "#888888"))

# ---------------------------------------------------------------------------
# PLY I/O — pure numpy, handles binary_little_endian XYZ+RGB
# ---------------------------------------------------------------------------

def read_pointcloud_ply(path: Path):
    """Read a binary PLY with float x,y,z and uchar red,green,blue.
    Returns (xyz: float32 N×3, rgb: uint8 N×3).
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    eoh = raw.find(b"end_header\n")
    assert eoh != -1, "end_header not found"
    header = raw[:eoh + 11].decode("ascii", errors="replace")

    n_verts = 0
    props = []
    for line in header.splitlines():
        line = line.strip()
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            props.append((parts[1], parts[2]))  # (type, name)

    # Build dtype
    type_map = {"float": "f4", "uchar": "u1", "double": "f8",
                "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
    dt = np.dtype([(name, type_map.get(typ, "f4")) for typ, name in props])

    body = raw[eoh + 11:]
    data = np.frombuffer(body, dtype=dt, count=n_verts).copy()

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def write_pointcloud_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Write binary little-endian PLY with float XYZ and uchar RGB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    # Interleave xyz (f4) and rgb (u1) as a structured array
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    out = np.empty(n, dtype=dt)
    out["x"] = xyz[:, 0].astype("f4")
    out["y"] = xyz[:, 1].astype("f4")
    out["z"] = xyz[:, 2].astype("f4")
    out["r"] = rgb[:, 0]
    out["g"] = rgb[:, 1]
    out["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(out.tobytes())
    mb = path.stat().st_size / 1e6
    print(f"[OK] Written {n:,} points → {path}  ({mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Mask helpers  (same logic as paint_semantic_gaussians.py)
# ---------------------------------------------------------------------------

def decode_rle(rle_dict: dict, h: int, w: int):
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
                      conf_threshold: float = 0.25,
                      max_mask_coverage: float = 0.30) -> np.ndarray:
    label_img = np.zeros((h, w), dtype=np.uint8)
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
            if mask is not None and (rle_h != h or rle_w != w):
                from PIL import Image as _PIL
                mask = np.array(_PIL.fromarray(mask).resize((w, h), _PIL.NEAREST))
        if mask is None or mask.sum() == 0:
            if "bbox" in ann and ann["bbox"]:
                mask = bbox_to_mask(ann["bbox"], h, w)
        if mask is not None and mask.sum() > 0:
            coverage = mask.sum() / (h * w)
            if coverage > max_mask_coverage:
                continue
            items.append((int(mask.sum()), conf, lbl, mask))

    items.sort(key=lambda t: (-t[0], t[1]))
    for _, _, lbl, mask in items:
        label_img[mask > 0] = LABEL_TO_IDX[lbl]
    return label_img


# ---------------------------------------------------------------------------
# Pose loading from COLMAP binary
# ---------------------------------------------------------------------------

def load_colmap_poses(cameras_bin: Path, images_bin: Path):
    """Return (poses, fx, fy, cx, cy, img_w, img_h).
    poses: dict frame_stem -> (R_w2c 3×3, t_w2c 3)  in COLMAP/OpenCV convention.
    """
    cameras = read_cameras_binary(cameras_bin)
    images  = read_images_binary(images_bin)

    # Use the first camera for intrinsics (single camera model assumed)
    cam = next(iter(cameras.values()))
    W, H = int(cam["width"]), int(cam["height"])
    model = cam["model"]
    params = cam["params"]
    if model in ("PINHOLE",):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in ("SIMPLE_PINHOLE",):
        fx = fy = params[0]; cx, cy = params[1], params[2]
    else:
        fx = fy = params[0]; cx, cy = params[2], params[3]

    poses = {}
    for img in images.values():
        stem = Path(img["name"]).stem
        # Decode with scipy (standard COLMAP convention) to avoid the
        # non-standard .T in colmap_utils.qvec_to_rotmat, which interacts
        # badly with the scipy-written qvecs from interpolate_slam_poses.py.
        # images.bin quaternions are [w, x, y, z]; scipy wants [x, y, z, w].
        q = img["qvec"]
        R_w2c = SciPyRotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t_w2c = img["tvec"]
        poses[stem] = (R_w2c, t_w2c)

    print(f"[OK] {len(poses)} COLMAP poses loaded  "
          f"({W}×{H}  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f})")
    return poses, fx, fy, cx, cy, W, H


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Paint semantic colours onto dense point cloud.")
    ap.add_argument("--pointcloud",   default=str(ROOT / "outputs/mast3r_out_v2/dense_pointcloud.ply"))
    ap.add_argument("--semantic_dir", default=str(ROOT / "outputs/semantic_mast3r"))
    ap.add_argument("--cameras_bin",  default=str(ROOT / "data/mast3r_out_v2/sparse/0/cameras.bin"))
    ap.add_argument("--images_bin",   default=str(ROOT / "data/mast3r_out_v2/sparse/0/images.bin"))
    ap.add_argument("--output_ply",   default=str(ROOT / "outputs/mast3r_out_v2/semantic_pointcloud.ply"))
    ap.add_argument("--web_ply",      default=str(ROOT / "outputs/scene_pointcloud_web.ply"))
    ap.add_argument("--n_web",        type=int, default=2_500_000)
    ap.add_argument("--min_votes",    type=int, default=2,
                    help="Min cross-frame votes to assign a label (default 2)")
    ap.add_argument("--max_frames",   type=int, default=None)
    ap.add_argument("--conf_threshold", type=float, default=0.25)
    ap.add_argument("--dim_factor",   type=float, default=0.30,
                    help="Multiply original RGB by this for unlabeled points (0=black, 1=keep)")
    args = ap.parse_args()

    # ── 1. Load point cloud ─────────────────────────────────────────────
    print(f"[...] Loading point cloud: {args.pointcloud}")
    xyz, rgb_orig = read_pointcloud_ply(Path(args.pointcloud))
    N = len(xyz)
    print(f"[OK] {N:,} points  "
          f"X=[{xyz[:,0].min():.2f},{xyz[:,0].max():.2f}]  "
          f"Y=[{xyz[:,1].min():.2f},{xyz[:,1].max():.2f}]  "
          f"Z=[{xyz[:,2].min():.2f},{xyz[:,2].max():.2f}]")

    # ── 2. Load COLMAP poses ────────────────────────────────────────────
    poses, fx, fy, cx, cy, img_w, img_h = load_colmap_poses(
        Path(args.cameras_bin), Path(args.images_bin)
    )

    # ── 3. Match semantic JSONs to poses ────────────────────────────────
    # Build numeric lookup for fuzzy frame matching (frame_0058 ↔ frame_00058)
    poses_by_num = {}
    for stem, pose in poses.items():
        try:
            poses_by_num[int(stem.replace("frame_", ""))] = (stem, pose)
        except ValueError:
            pass

    sem_dir = Path(args.semantic_dir)
    if not sem_dir.exists():
        sys.exit(f"[error] semantic_dir not found: {sem_dir}\n"
                 "Run scripts/run_semantic.py first (see PLAN.md Step 2).")

    all_jsons = sorted(sem_dir.glob("frame_*.json"))
    matched = []
    matched_poses = {}
    for p in all_jsons:
        try:
            num = int(p.stem.replace("frame_", ""))
        except ValueError:
            continue
        if num in poses_by_num:
            matched.append(p)
            matched_poses[p] = poses_by_num[num][1]  # (R_w2c, t_w2c)

    if args.max_frames:
        matched = matched[:args.max_frames]
    print(f"[OK] {len(matched)} frames matched to poses")
    if not matched:
        sys.exit("[error] No matched frames. Check --semantic_dir and --images_bin.")

    # ── 4. Vote semantic labels per point ───────────────────────────────
    # votes[i, c] = number of frames where point i projected into class c
    print(f"[...] Voting across {len(matched)} frames for {N:,} points ...")
    votes = np.zeros((N, N_CLASSES + 1), dtype=np.int16)

    xyz_f64 = xyz.astype(np.float64)

    for frame_idx, sem_path in enumerate(matched):
        R_w2c, t_w2c = matched_poses[sem_path]

        with open(sem_path) as f:
            sem_data = json.load(f)
        if not sem_data:
            continue

        label_img = build_label_image(sem_data, img_h, img_w,
                                      conf_threshold=args.conf_threshold)
        if label_img.max() == 0:
            continue

        # Project all points into camera (vectorized)
        p_cam = (R_w2c @ xyz_f64.T).T + t_w2c   # (N, 3)

        in_front = p_cam[:, 2] > 0.05
        z = p_cam[:, 2]
        u = fx * p_cam[:, 0] / z + cx
        v = fy * p_cam[:, 1] / z + cy
        u_i = np.round(u).astype(np.int32)
        v_i = np.round(v).astype(np.int32)

        in_bounds = (in_front
                     & (u_i >= 0) & (u_i < img_w)
                     & (v_i >= 0) & (v_i < img_h))

        valid_idx = np.where(in_bounds)[0]
        labels_hit = label_img[v_i[valid_idx], u_i[valid_idx]]

        labeled_mask = labels_hit > 0
        if labeled_mask.sum() == 0:
            continue

        hit_idx    = valid_idx[labeled_mask]
        hit_labels = labels_hit[labeled_mask]
        np.add.at(votes, (hit_idx, hit_labels), 1)

        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            labeled_so_far = (votes[:, 1:].max(axis=1) >= args.min_votes).sum()
            print(f"  [{frame_idx+1:3d}/{len(matched)}] {sem_path.stem}  "
                  f"labeled: {labeled_so_far:,} ({100.*labeled_so_far/N:.1f}%)",
                  flush=True)

    print("[OK] Voting complete")

    # ── 5. Assign semantic colors ────────────────────────────────────────
    total_votes     = votes[:, 1:].sum(axis=1)
    winning_cls_rel = np.argmax(votes[:, 1:], axis=1) + 1  # 1-indexed
    is_labeled      = total_votes >= args.min_votes

    n_labeled = is_labeled.sum()
    print(f"\n[OK] {n_labeled:,} / {N:,} points labeled ({100.*n_labeled/N:.1f}%)")

    semantic_class = np.where(is_labeled, winning_cls_rel, 0)

    # Per-class counts
    stats = {"total_points": int(N), "labeled": int(n_labeled),
             "unlabeled": int(N - n_labeled), "classes": {}}
    for lbl, idx in sorted(LABEL_TO_IDX.items(), key=lambda x: x[1]):
        cnt = int((semantic_class == idx).sum())
        stats["classes"][lbl] = {"count": cnt, "pct": round(100.*cnt/N, 2)}
        if cnt > 0:
            print(f"  {lbl:<12} {cnt:>8,}  ({100.*cnt/N:.2f}%)")

    # Build output RGB
    rgb_out = rgb_orig.copy()

    # Labeled points: assign class color
    labeled_idx = np.where(is_labeled)[0]
    rgb_out[labeled_idx] = CLASS_RGB_U8[semantic_class[labeled_idx]]

    # Unlabeled points: dim the original color
    unlabeled_idx = np.where(~is_labeled)[0]
    if args.dim_factor < 1.0:
        rgb_out[unlabeled_idx] = (
            rgb_orig[unlabeled_idx].astype(np.float32) * args.dim_factor
        ).clip(0, 255).astype(np.uint8)

    # ── 6. Write full semantic point cloud ──────────────────────────────
    print(f"[...] Writing semantic point cloud → {args.output_ply}")
    write_pointcloud_ply(Path(args.output_ply), xyz, rgb_out)

    # ── 7. Write downsampled web PLY ────────────────────────────────────
    web_path = Path(args.web_ply)
    if args.n_web < N:
        step = max(1, N // args.n_web)
        idx_web = np.arange(0, N, step)[:args.n_web]
        print(f"[...] Downsampling {N:,} → {len(idx_web):,} for web viewer...")
        write_pointcloud_ply(web_path, xyz[idx_web], rgb_out[idx_web])
    else:
        write_pointcloud_ply(web_path, xyz, rgb_out)

    # ── 8. Write stats ───────────────────────────────────────────────────
    stats_path = ROOT / "outputs/pointcloud_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Stats → {stats_path}")

    # ── 9. Compute per-class centroids in MASt3R-SLAM world space ────────
    centroids = {}
    for lbl, idx in LABEL_TO_IDX.items():
        mask = semantic_class == idx
        if mask.sum() < 100:
            continue
        pts = xyz[mask]
        # Median is robust against thin outlier clusters
        centroid = np.median(pts, axis=0).tolist()
        centroids[lbl] = {
            "centroid": [round(v, 4) for v in centroid],
            "point_count": int(mask.sum()),
            "color": CLASS_COLORS_HEX.get(lbl, "#888888"),
        }
        print(f"  {lbl:<12} centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")

    centroids_path = ROOT / "outputs/semantic_centroids.json"
    with open(centroids_path, "w") as f:
        json.dump(centroids, f, indent=2)
    print(f"[OK] Centroids → {centroids_path}  ({len(centroids)} objects)")

    print("\n=== Done ===")
    print(f"Full semantic PLY : {args.output_ply}")
    print(f"Web viewer PLY    : {args.web_ply}  ({len(idx_web) if args.n_web < N else N:,} pts)")
    print(f"Labeled           : {n_labeled:,} / {N:,} ({100.*n_labeled/N:.1f}%)")
    print(f"Centroids JSON    : {centroids_path}")


if __name__ == "__main__":
    main()
