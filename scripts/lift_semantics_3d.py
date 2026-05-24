"""Lift 2D semantic masks to 3D by projecting the point cloud into each keyframe."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project scripts importable
sys.path.insert(0, str(Path(__file__).parent))
from colmap_utils import read_cameras_binary, read_images_binary, qvec_to_rotmat

try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    print("[warn] pycocotools not found - falling back to bbox masks")


# ---------------------------------------------------------------------------
# Label colour palette (Session 4)
# ---------------------------------------------------------------------------
LABEL_COLOURS = {
    "floor":        "#8B4513",
    "wall":         "#D3D3D3",
    "ceiling":      "#F5F5DC",
    "door":         "#DEB887",
    "window":       "#87CEEB",
    "table":        "#8FBC8F",
    "chair":        "#4682B4",
    "sofa":         "#9370DB",
    "bed":          "#FFB6C1",
    "desk":         "#20B2AA",
    "cabinet":      "#DAA520",
    "shelf":        "#CD853F",
    "tv":           "#2F4F4F",
    "monitor":      "#2F4F4F",
    "sink":         "#B0C4DE",
    "toilet":       "#FFFACD",
    "bathtub":      "#E0FFFF",
    "refrigerator": "#98FB98",
    "microwave":    "#F0E68C",
    "oven":         "#BC8F8F",
    "lamp":         "#FFD700",
    "plant":        "#228B22",
    "picture":      "#DDA0DD",
    "mirror":       "#C0C0C0",
    "curtain":      "#FF69B4",
    "carpet":       "#8B0000",
    "pillow":       "#FFDAB9",
    "book":         "#6B8E23",
    "box":          "#D2691E",
    "bag":          "#FF8C00",
    "fan":          "#FF6347",
    "laptop":       "#6495ED",
}
DEFAULT_COLOUR = "#888888"


# ---------------------------------------------------------------------------
# PLY reader  (pure numpy - no open3d needed on Python 3.13)
# ---------------------------------------------------------------------------

def read_ply_xyz_rgb(path):
    """
    Read a binary-little-endian PLY with float xyz + uchar rgb.
    Returns (xyz: ndarray (N,3) float64,  rgb: ndarray (N,3) uint8).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")

    with open(path, "rb") as f:
        n_verts = 0
        props   = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        type_map = {
            "float": "f4", "float32": "f4",
            "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1",
            "char": "i1",  "int8": "i1",
            "short": "i2", "int16": "i2",
            "ushort": "u2","uint16": "u2",
            "int": "i4",   "int32": "i4",
            "uint": "u4",  "uint32": "u4",
        }
        dt_fields = [(name, type_map.get(typ, "f4")) for typ, name in props]
        dt  = np.dtype(dt_fields)
        raw = f.read(n_verts * dt.itemsize)

    arr = np.frombuffer(raw, dtype=dt)
    xyz = np.stack([arr["x"].astype(np.float64),
                    arr["y"].astype(np.float64),
                    arr["z"].astype(np.float64)], axis=1)

    rgb_names = [n for _, n in props if n in ("red", "green", "blue")]
    if len(rgb_names) == 3:
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1)
    else:
        rgb = np.zeros((n_verts, 3), dtype=np.uint8)

    return xyz, rgb


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def decode_rle_mask(rle_dict, height, width):
    """Decode COCO RLE to uint8 (H,W). Returns None on failure."""
    if not HAS_COCO:
        return None
    counts = rle_dict.get("counts")
    size   = rle_dict.get("size", [height, width])
    try:
        if isinstance(counts, list):
            rle     = {"counts": counts, "size": size}
            decoded = coco_mask.decode(coco_mask.frPyObjects(rle, size[0], size[1]))
        else:
            decoded = coco_mask.decode({"counts": counts, "size": size})
        return decoded.astype(np.uint8)
    except Exception:
        return None


def bbox_to_mask(bbox, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, bw, bh = (int(v) for v in bbox)
    x2 = min(x + bw, width);  y2 = min(y + bh, height)
    x  = max(x, 0);           y  = max(y, 0)
    if x2 > x and y2 > y:
        mask[y:y2, x:x2] = 1
    return mask


def get_mask(ann, height, width):
    """Return best available mask (RLE preferred, bbox fallback)."""
    mask = None
    if "mask_rle" in ann and ann["mask_rle"]:
        mask = decode_rle_mask(ann["mask_rle"], height, width)
    if mask is None or mask.sum() == 0:
        if "bbox" in ann and ann["bbox"]:
            mask = bbox_to_mask(ann["bbox"], height, width)
    return mask


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def remove_outliers(points, n_std=2.0):
    if len(points) < 4:
        return points
    mean = points.mean(axis=0)
    std  = points.std(axis=0)
    std[std < 1e-6] = 1e-6
    keep = np.all(np.abs(points - mean) <= n_std * std, axis=1)
    out  = points[keep]
    return out if len(out) >= 1 else points


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def get_label_colour(label):
    llow = label.lower()
    for key, col in LABEL_COLOURS.items():
        if key in llow or llow in key:
            return col
    return DEFAULT_COLOUR


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Top-down plot
# ---------------------------------------------------------------------------

def save_topdown_plot(objects_3d, output_path, all_world_pts=None, room_pts=None):
    """X-Z scatter of object centroids with optional room boundary box."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # room boundary from full point cloud extent
    if room_pts is not None and len(room_pts) > 0:
        x_all, z_all = room_pts[:, 0], room_pts[:, 2]
        pad = 0.3
        ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
        ax.set_ylim(z_all.min() - pad, z_all.max() + pad)
        rect = plt.Rectangle(
            (x_all.min(), z_all.min()),
            x_all.max() - x_all.min(),
            z_all.max() - z_all.min(),
            linewidth=1.5, edgecolor="#aaaaaa",
            facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
    elif all_world_pts:
        combined = np.concatenate(list(all_world_pts.values()), axis=0)
        x_all, z_all = combined[:, 0], combined[:, 2]
        pad = 0.3
        ax.set_xlim(x_all.min() - pad, x_all.max() + pad)
        ax.set_ylim(z_all.min() - pad, z_all.max() + pad)

    for label, info in objects_3d.items():
        cx_w, _cy, cz_w = info["centroid_3d"]
        colour = get_label_colour(label)
        rgb    = hex_to_rgb(colour)
        ax.scatter(cx_w, cz_w, s=200, color=rgb, edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(label, (cx_w, cz_w),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=9, fontweight="bold", color="black", zorder=6)

    ax.set_xlabel("X  (metres)", fontsize=12)
    ax.set_ylabel("Z  (metres)", fontsize=12)
    ax.set_title("Object Positions - Top-down view (X-Z plane)", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Top-down plot saved -> {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(objects_3d):
    header = (f"{'Label':<22} {'Centroid (x,y,z)':<32}"
              f" {'Volume m3':>10} {'Frames':>7} {'Points':>8}")
    sep = "-" * len(header)
    print(); print(sep); print(header); print(sep)
    for label, info in sorted(objects_3d.items()):
        cx, cy, cz = info["centroid_3d"]
        cs = f"({cx:+.2f}, {cy:+.2f}, {cz:+.2f})"
        print(f"{label:<22} {cs:<32}"
              f" {info['volume_m3']:>10.4f}"
              f" {info['frames_seen']:>7}"
              f" {info['total_points']:>8}")
    print(sep)
    print(f"  {len(objects_3d)} objects reconstructed")
    print()


# ---------------------------------------------------------------------------
# Core: project point cloud into one keyframe
# ---------------------------------------------------------------------------

def project_frame(xyz_world, sem_data, R_w2c, t_w2c,
                  fx, fy, cx, cy, img_w, img_h):
    """
    Project world-space points into one camera frame and collect
    those that fall inside each label's semantic mask.

    Parameters
    ----------
    xyz_world : (N,3) float64  world-space point cloud
    sem_data  : dict  {label: {bbox, confidence, mask_rle}}
    R_w2c     : (3,3) world-to-camera rotation
    t_w2c     : (3,)  world-to-camera translation
    fx,fy,cx,cy : camera intrinsics
    img_w, img_h : image width / height

    Returns
    -------
    dict {label: (M,3) world-space points that project into mask}
    """
    # transform: p_cam = R @ p_world + t  (broadcast over N points)
    # Cast to float64 first to avoid harmless float32 overflow warnings on Apple Silicon
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_cam = (R_w2c.astype(np.float64) @ xyz_world.astype(np.float64).T).T + t_w2c.astype(np.float64)

    # keep only points in front of camera
    front   = p_cam[:, 2] > 0.05
    p_cam_f = p_cam[front]                        # (M,3)
    idx_f   = np.where(front)[0]                  # original indices

    if len(p_cam_f) == 0:
        return {}

    # perspective projection
    z   = p_cam_f[:, 2]
    u   = fx * p_cam_f[:, 0] / z + cx            # (M,)
    v   = fy * p_cam_f[:, 1] / z + cy            # (M,)

    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)

    # keep only pixels inside image bounds
    in_frame = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)
    u_ib     = u_i[in_frame]
    v_ib     = v_i[in_frame]
    idx_ib   = idx_f[in_frame]                    # back to original indices

    if len(u_ib) == 0:
        return {}

    frame_result = {}
    for label, ann in sem_data.items():
        mask = get_mask(ann, img_h, img_w)
        if mask is None or mask.sum() == 0:
            continue
        in_mask = mask[v_ib, u_ib] > 0
        if in_mask.sum() == 0:
            continue
        frame_result[label] = xyz_world[idx_ib[in_mask]]   # world coords

    return frame_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lift semantic masks to 3D via MASt3R point cloud projection."
    )
    parser.add_argument("--pointcloud",   default="outputs/mast3r_out/room_video.ply",
                        help="MASt3R-SLAM point cloud PLY")
    parser.add_argument("--semantic_dir", default="outputs/semantic/",
                        help="Per-frame semantic JSON files")
    parser.add_argument("--colmap_dir",   default="data/mast3r_out/sparse/0/",
                        help="COLMAP sparse dir (cameras.bin + images.bin)")
    parser.add_argument("--output_file",  default="outputs/objects_3d.json",
                        help="Output JSON")
    parser.add_argument("--max_points",   type=int, default=500_000,
                        help="Subsample point cloud to this many points (default 500k)")
    parser.add_argument("--min_points",   type=int, default=30,
                        help="Min 3D points to keep an object (default 30)")
    args = parser.parse_args()

    semantic_dir = Path(args.semantic_dir)
    colmap_dir   = Path(args.colmap_dir)
    output_file  = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load point cloud
    print(f"[...] Loading point cloud: {args.pointcloud}")
    xyz_all, _ = read_ply_xyz_rgb(args.pointcloud)
    print(f"[OK] Loaded {len(xyz_all):,} points  "
          f"X=[{xyz_all[:,0].min():.2f},{xyz_all[:,0].max():.2f}]  "
          f"Y=[{xyz_all[:,1].min():.2f},{xyz_all[:,1].max():.2f}]  "
          f"Z=[{xyz_all[:,2].min():.2f},{xyz_all[:,2].max():.2f}]")

    if len(xyz_all) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xyz_all), args.max_points, replace=False)
        idx.sort()
        xyz = xyz_all[idx]
        print(f"[OK] Subsampled to {len(xyz):,} points")
    else:
        xyz = xyz_all

    # 2. Load COLMAP intrinsics from cameras.bin
    cameras_bin = colmap_dir / "cameras.bin"
    if not cameras_bin.exists():
        sys.exit(f"[error] cameras.bin not found: {cameras_bin}")

    cameras = read_cameras_binary(str(cameras_bin))
    cam     = next(iter(cameras.values()))
    params  = cam["params"]
    img_w   = int(cam["width"])
    img_h   = int(cam["height"])
    if "SIMPLE" in cam["model"].upper() or len(params) < 4:
        fx = fy = float(params[0])
        cx_k, cy_k = float(params[1]), float(params[2])
    else:
        fx, fy     = float(params[0]), float(params[1])
        cx_k, cy_k = float(params[2]), float(params[3])

    print(f"[OK] Intrinsics: fx={fx:.2f} fy={fy:.2f} "
          f"cx={cx_k:.2f} cy={cy_k:.2f} img={img_w}x{img_h}")

    # 3. Load COLMAP keyframe poses from images.bin
    images_bin = colmap_dir / "images.bin"
    if not images_bin.exists():
        sys.exit(f"[error] images.bin not found: {images_bin}")

    colmap_images = read_images_binary(str(images_bin))
    print(f"[OK] Loaded {len(colmap_images)} COLMAP keyframe poses")

    # Build {frame_stem -> (R_w2c, t_w2c)}
    # COLMAP images.bin stores w2c:  p_cam = R @ p_world + t
    keyframe_poses = {}
    for img_id, img in colmap_images.items():
        stem  = Path(img["name"]).stem
        R_w2c = qvec_to_rotmat(img["qvec"])   # (3,3)
        t_w2c = img["tvec"]                   # (3,)
        keyframe_poses[stem] = (R_w2c, t_w2c)

    # 4. Match semantic JSONs to keyframe poses
    all_sem_jsons = sorted(semantic_dir.glob("frame_*.json"))
    matched = [(p, keyframe_poses[p.stem])
               for p in all_sem_jsons if p.stem in keyframe_poses]
    print(f"[OK] {len(matched)} keyframes with semantic JSON + COLMAP pose")

    if not matched:
        sys.exit("[error] No matching keyframes. "
                 "Check --semantic_dir and --colmap_dir frame names.")

    # 5. Project point cloud into each keyframe
    label_points = {}   # label -> list of (M,3) arrays
    frames_seen  = {}   # label -> set of frame stems

    for i, (sem_path, (R_w2c, t_w2c)) in enumerate(matched):
        frame_stem = sem_path.stem
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(matched)}] {frame_stem}", flush=True)

        with open(sem_path) as f:
            sem_data = json.load(f)
        if not sem_data:
            continue

        frame_result = project_frame(
            xyz_world=xyz,
            sem_data=sem_data,
            R_w2c=R_w2c.astype(np.float64),
            t_w2c=t_w2c.astype(np.float64),
            fx=fx, fy=fy, cx=cx_k, cy=cy_k,
            img_w=img_w, img_h=img_h,
        )

        for label, pts in frame_result.items():
            if label not in label_points:
                label_points[label] = []
                frames_seen[label]  = set()
            label_points[label].append(pts)
            frames_seen[label].add(frame_stem)

    print(f"[OK] Projection done - {len(label_points)} labels found")

    # 6. Aggregate per label: deduplicate, outlier removal, stats
    objects_3d    = {}
    all_world_pts = {}

    for label, pt_list in label_points.items():
        pts = np.concatenate(pt_list, axis=0)
        pts = np.unique(pts, axis=0)          # deduplicate (same point seen from multiple frames)
        pts_clean = remove_outliers(pts, n_std=2.0)

        if len(pts_clean) < args.min_points:
            print(f"  [skip] '{label}' - {len(pts_clean)} pts (min={args.min_points})")
            continue

        all_world_pts[label] = pts_clean
        centroid = pts_clean.mean(axis=0)
        bbox_min = pts_clean.min(axis=0)
        bbox_max = pts_clean.max(axis=0)
        dims     = bbox_max - bbox_min
        volume   = float(dims[0] * dims[1] * dims[2])

        objects_3d[label] = {
            "centroid_3d":               centroid.tolist(),
            "bbox_min":                  bbox_min.tolist(),
            "bbox_max":                  bbox_max.tolist(),
            "volume_m3":                 round(volume, 6),
            "frames_seen":               len(frames_seen[label]),
            "total_points":              int(len(pts_clean)),
            "reconstruction_confidence": None,   # filled in Session 6
            "provenance":                None,   # filled in Session 6
        }

    # 7. Save JSON
    with open(output_file, "w") as f:
        json.dump(objects_3d, f, indent=2)
    print(f"[OK] objects_3d.json -> {output_file}  ({len(objects_3d)} objects)")

    # 8. Top-down plot
    plot_path = output_file.parent / "object_positions_2d.png"
    if objects_3d:
        save_topdown_plot(objects_3d, str(plot_path),
                          all_world_pts=all_world_pts, room_pts=xyz_all)
    else:
        print("[warn] No objects to plot.")

    # 9. Summary table
    print_summary(objects_3d)


if __name__ == "__main__":
    main()
