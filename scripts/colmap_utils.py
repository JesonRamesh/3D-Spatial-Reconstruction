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


# ── Binary Readers ─────────────────────────────────────────────────────
# Mirrors the writers below but reads COLMAP binary format back into
# Python dicts. Used by our patched gsplat dataset loader to avoid
# the broken pycolmap.SceneManager dependency.

def read_cameras_binary(path: Path) -> dict:
    """
    Read cameras.bin.

    Returns:
        dict of camera_id -> {
            'camera_id': int, 'model_id': int, 'model': str,
            'width': int, 'height': int, 'params': np.array
        }
    """
    MODEL_ID_TO_NAME = {v: k for k, v in CAMERA_MODEL_IDS.items()}
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_name = MODEL_ID_TO_NAME.get(model_id, "PINHOLE")
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model_name, 4)
            params = np.array(
                struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            )
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model_id": model_id,
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path: Path) -> dict:
    """
    Read images.bin.

    Returns:
        dict of image_id -> {
            'image_id': int, 'qvec': np.array[4] (w,x,y,z),
            'tvec': np.array[3], 'camera_id': int, 'name': str,
            'point2D_xys': np.array[N,2], 'point3D_ids': np.array[N]
        }
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))  # w,x,y,z
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # Read null-terminated name
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)
            # Read 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            xys = np.zeros((num_points2D, 2), dtype=np.float64)
            point3D_ids = np.full(num_points2D, -1, dtype=np.int64)
            for j in range(num_points2D):
                xys[j, 0] = struct.unpack("<d", f.read(8))[0]
                xys[j, 1] = struct.unpack("<d", f.read(8))[0]
                point3D_ids[j] = struct.unpack("<q", f.read(8))[0]
            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "point2D_xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def read_points3d_binary(path: Path) -> dict:
    """
    Read points3D.bin.

    Returns:
        dict of point3D_id -> {
            'point3D_id': int, 'xyz': np.array[3], 'rgb': np.array[3],
            'error': float, 'track': list of (image_id, point2D_idx)
        }
    """
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                img_id = struct.unpack("<I", f.read(4))[0]
                pt2d_idx = struct.unpack("<I", f.read(4))[0]
                track.append((img_id, pt2d_idx))
            points[point3D_id] = {
                "point3D_id": point3D_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix.

    Uses the same convention as COLMAP:
    https://github.com/colmap/colmap/blob/main/src/colmap/util/types.h
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ]).T
    return R


# ── Binary Writers ─────────────────────────────────────────────────────


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