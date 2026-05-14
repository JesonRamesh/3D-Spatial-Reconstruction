#!/usr/bin/env python3
"""
COLMAP binary file writer.

Writes cameras.bin, images.bin, and points3D.bin in COLMAP's native binary format.
No dependency on pycolmap — works with any version of downstream tools.

Reference: https://colmap.github.io/format.html#binary-format
"""

import struct
import numpy as np
from pathlib import Path

# Camera model IDs (COLMAP convention)
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
}

# Number of params per camera model
CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,  # f, cx, cy
    "PINHOLE": 4,         # fx, fy, cx, cy
    "SIMPLE_RADIAL": 4,   # f, cx, cy, k1
    "RADIAL": 5,          # f, cx, cy, k1, k2
    "OPENCV": 8,          # fx, fy, cx, cy, k1, k2, p1, p2
}


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx = R[0, 0], R[1, 0], R[2, 0]
    Rxy, Ryy, Rzy = R[0, 1], R[1, 1], R[2, 1]
    Rxz, Ryz, Rzz = R[0, 2], R[1, 2], R[2, 2]

    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[:, np.argmax(eigvals)]  # [x, y, z, w]
    if qvec[3] < 0:
        qvec = -qvec
    return np.array([qvec[3], qvec[0], qvec[1], qvec[2]])  # [w, x, y, z]


def write_cameras_binary(cameras: list, path: Path):
    """
    Write cameras.bin.

    cameras: list of dicts with keys:
        camera_id (int), model (str), width (int), height (int), params (np.array)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))  # num cameras
        for cam in cameras:
            model_id = CAMERA_MODEL_IDS[cam["model"]]
            f.write(struct.pack("<I", cam["camera_id"]))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", cam["width"]))
            f.write(struct.pack("<Q", cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def write_images_binary(images: list, path: Path):
    """
    Write images.bin.

    images: list of dicts with keys:
        image_id (int), qvec (np.array[4]), tvec (np.array[3]),
        camera_id (int), name (str),
        point2D_xys (np.array[N,2]), point3D_ids (np.array[N] of int64, -1 for no match)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))  # num images
        for img in images:
            f.write(struct.pack("<I", img["image_id"]))
            # qvec (w, x, y, z)
            for q in img["qvec"]:
                f.write(struct.pack("<d", q))
            # tvec
            for t in img["tvec"]:
                f.write(struct.pack("<d", t))
            # camera_id
            f.write(struct.pack("<I", img["camera_id"]))
            # image name (null-terminated)
            name_bytes = img["name"].encode("utf-8") + b"\x00"
            f.write(name_bytes)
            # 2D points
            xys = img.get("point2D_xys", np.zeros((0, 2)))
            ids = img.get("point3D_ids", np.array([], dtype=np.int64))
            n_pts = len(xys)
            f.write(struct.pack("<Q", n_pts))
            for j in range(n_pts):
                f.write(struct.pack("<d", xys[j, 0]))
                f.write(struct.pack("<d", xys[j, 1]))
                f.write(struct.pack("<q", ids[j]))


def write_points3D_binary(points: list, path: Path):
    """
    Write points3D.bin.

    points: list of dicts with keys:
        point3D_id (int), xyz (np.array[3]), rgb (np.array[3] uint8),
        error (float),
        track: list of (image_id, point2D_idx) tuples
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points:
            f.write(struct.pack("<Q", pt["point3D_id"]))
            for v in pt["xyz"]:
                f.write(struct.pack("<d", v))
            for c in pt["rgb"]:
                f.write(struct.pack("<B", int(c)))
            f.write(struct.pack("<d", pt.get("error", 0.0)))
            track = pt.get("track", [])
            f.write(struct.pack("<Q", len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack("<I", img_id))
                f.write(struct.pack("<I", pt2d_idx))


def write_colmap_reconstruction(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_names: list,
    world_points: np.ndarray,
    depth_conf: np.ndarray,
    points_rgb: np.ndarray,
    output_dir: Path,
    original_coords: np.ndarray = None,
    vggt_resolution: int = 518,
    conf_threshold: float = 3.0,
    max_points: int = 100000,
    camera_model: str = "PINHOLE",
):
    """
    Write a full COLMAP sparse reconstruction from VGGT outputs.

    Args:
        extrinsics: [N, 3, 4] cam-from-world matrices
        intrinsics: [N, 3, 3] intrinsic matrices (at vggt_resolution)
        image_names: list of N image filenames
        world_points: [N, H, W, 3] world coordinates
        depth_conf: [N, H, W] confidence scores
        points_rgb: [N, H, W, 3] uint8 RGB colours
        output_dir: path to write sparse/ directory
        original_coords: [N, 6] from load_and_preprocess_images_square
        vggt_resolution: VGGT internal resolution (518)
        conf_threshold: minimum depth confidence to include a point
        max_points: maximum number of 3D points
        camera_model: COLMAP camera model name
    """
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    N = len(extrinsics)
    num_frames, height, width, _ = world_points.shape

    # ── Cameras ────────────────────────────────────────────────────────
    cameras = []
    for i in range(N):
        K = intrinsics[i]
        if original_coords is not None:
            real_size = original_coords[i, -2:]  # [width, height]
            resize_ratio = max(real_size) / vggt_resolution
            fx = K[0, 0] * resize_ratio
            fy = K[1, 1] * resize_ratio
            cx = real_size[0] / 2
            cy = real_size[1] / 2
            w, h = int(real_size[0]), int(real_size[1])
        else:
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            w, h = vggt_resolution, vggt_resolution

        if camera_model == "PINHOLE":
            params = np.array([fx, fy, cx, cy])
        elif camera_model == "SIMPLE_PINHOLE":
            params = np.array([(fx + fy) / 2, cx, cy])
        else:
            params = np.array([fx, fy, cx, cy])

        cameras.append({
            "camera_id": i + 1,
            "model": camera_model,
            "width": w,
            "height": h,
            "params": params,
        })

    write_cameras_binary(cameras, sparse_dir / "cameras.bin")

    # ── Images (poses) ─────────────────────────────────────────────────
    images = []
    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        qvec = rotmat_to_qvec(R)

        images.append({
            "image_id": i + 1,
            "qvec": qvec,
            "tvec": t,
            "camera_id": i + 1,
            "name": image_names[i],
            "point2D_xys": np.zeros((0, 2)),
            "point3D_ids": np.array([], dtype=np.int64),
        })

    write_images_binary(images, sparse_dir / "images.bin")

    # ── 3D Points ──────────────────────────────────────────────────────
    # Filter by confidence
    conf_mask = depth_conf >= conf_threshold

    # Randomly subsample if too many
    true_count = conf_mask.sum()
    if true_count > max_points:
        true_indices = np.flatnonzero(conf_mask.ravel())
        sampled = np.random.choice(true_indices, size=max_points, replace=False)
        new_mask = np.zeros(conf_mask.size, dtype=bool)
        new_mask[sampled] = True
        conf_mask = new_mask.reshape(conf_mask.shape)

    pts3d = world_points[conf_mask]
    pts_rgb_filtered = points_rgb[conf_mask]
    n_points = len(pts3d)

    points_list = []
    for j in range(n_points):
        points_list.append({
            "point3D_id": j + 1,
            "xyz": pts3d[j],
            "rgb": pts_rgb_filtered[j],
            "error": 0.0,
            "track": [],
        })

    write_points3D_binary(points_list, sparse_dir / "points3D.bin")

    return n_points, sparse_dir