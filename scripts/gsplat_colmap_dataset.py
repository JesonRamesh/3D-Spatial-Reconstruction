"""
Patched COLMAP dataset loader for gsplat's simple_trainer.py.

This replaces gsplat's datasets/colmap.py which depends on
pycolmap.SceneManager (removed in pycolmap 3.x).

Instead, we use our own COLMAP binary readers from colmap_utils.py.
The interface (Parser class, Dataset class) is identical to gsplat's
original so simple_trainer.py works without modification.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

# Add the project scripts/ dir to path so we can import colmap_utils
_this_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.dirname(_this_dir)  # scripts/gsplat_examples -> scripts/
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    qvec_to_rotmat,
)

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser using custom binary readers (no pycolmap dependency)."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # Find COLMAP sparse directory
        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        # Read binary files using our custom readers
        from pathlib import Path
        colmap_path = Path(colmap_dir)
        cameras_dict = read_cameras_binary(colmap_path / "cameras.bin")
        images_dict = read_images_binary(colmap_path / "images.bin")
        points3d_dict = read_points3d_binary(colmap_path / "points3D.bin")

        # Extract extrinsic matrices in world-to-camera format
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        image_names_unsorted = []

        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for image_id in images_dict:
            im = images_dict[image_id]
            qvec = im["qvec"]  # (w, x, y, z)
            tvec = im["tvec"]  # (3,)

            # Convert quaternion to rotation matrix
            rot = qvec_to_rotmat(qvec)
            trans = tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im["camera_id"]
            camera_ids.append(camera_id)
            image_names_unsorted.append(im["name"])

            # Camera intrinsics (only process each camera_id once)
            if camera_id not in Ks_dict:
                cam = cameras_dict[camera_id]
                model = cam["model"]
                params = cam["params"]
                width = cam["width"]
                height = cam["height"]

                # Extract fx, fy, cx, cy based on camera model
                if model == "SIMPLE_PINHOLE":
                    f, cx, cy = params[0], params[1], params[2]
                    fx, fy = f, f
                    dist_params = np.empty(0, dtype=np.float32)
                elif model == "PINHOLE":
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                    dist_params = np.empty(0, dtype=np.float32)
                elif model == "SIMPLE_RADIAL":
                    f, cx, cy = params[0], params[1], params[2]
                    fx, fy = f, f
                    dist_params = np.array(
                        [params[3], 0.0, 0.0, 0.0], dtype=np.float32
                    )
                elif model == "RADIAL":
                    f, cx, cy = params[0], params[1], params[2]
                    fx, fy = f, f
                    dist_params = np.array(
                        [params[3], params[4], 0.0, 0.0], dtype=np.float32
                    )
                elif model == "OPENCV":
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                    dist_params = np.array(
                        [params[4], params[5], params[6], params[7]], dtype=np.float32
                    )
                else:
                    # Default: treat as pinhole
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                    dist_params = np.empty(0, dtype=np.float32)

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                K[:2, :] /= factor

                Ks_dict[camera_id] = K
                params_dict[camera_id] = dist_params
                imsize_dict[camera_id] = (width // factor, height // factor)

        print(
            f"[Parser] {len(images_dict)} images, "
            f"taken by {len(set(camera_ids))} cameras."
        )

        if len(images_dict) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort by image name for reproducibility
        inds = np.argsort(image_names_unsorted)
        image_names = [image_names_unsorted[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images directory
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""

        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)

        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Map COLMAP image names to actual files on disk
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [
            os.path.join(image_dir, colmap_to_image.get(f, f)) for f in image_names
        ]

        # 3D points
        if len(points3d_dict) > 0:
            # Build arrays from points3D dict
            point3D_ids = sorted(points3d_dict.keys())
            point3D_id_to_idx = {pid: idx for idx, pid in enumerate(point3D_ids)}

            points = np.array(
                [points3d_dict[pid]["xyz"] for pid in point3D_ids], dtype=np.float32
            )
            points_err = np.array(
                [points3d_dict[pid]["error"] for pid in point3D_ids], dtype=np.float32
            )
            points_rgb = np.array(
                [points3d_dict[pid]["rgb"] for pid in point3D_ids], dtype=np.uint8
            )

            # Build point_indices: image_name -> [point_idx, ...]
            # Map image_id to image_name
            image_id_to_name = {
                images_dict[iid]["image_id"]: images_dict[iid]["name"]
                for iid in images_dict
            }

            point_indices = dict()
            for pid in point3D_ids:
                pt = points3d_dict[pid]
                for img_id, _ in pt["track"]:
                    if img_id in image_id_to_name:
                        img_name = image_id_to_name[img_id]
                        idx = point3D_id_to_idx[pid]
                        point_indices.setdefault(img_name, []).append(idx)

            point_indices = {
                k: np.array(v).astype(np.int32) for k, v in point_indices.items()
            }
        else:
            # No 3D points (e.g., from VGGT with empty point cloud)
            points = np.zeros((0, 3), dtype=np.float32)
            points_err = np.zeros((0,), dtype=np.float32)
            points_rgb = np.zeros((0, 3), dtype=np.uint8)
            point_indices = dict()

        # Normalize the world space
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            if len(points) > 0:
                points = transform_points(T1, points)

            if len(points) > 0:
                T2 = align_principle_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)
                transform = T2 @ T1
            else:
                transform = T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray]
        self.transform = transform  # np.ndarray, (4, 4)

        # Load one image to check actual size vs COLMAP size
        # (handles datasets where intrinsics correspond to upsampled images)
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height = actual_height / colmap_height
        s_width = actual_width / colmap_width

        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (
                int(width * s_width),
                int(height * s_height),
            )

        # Undistortion maps
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # Scene scale from camera locations
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Number of unique cameras (used by simple_trainer for batch validation)
        self.num_cameras = len(set(camera_ids))


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            # Project points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices.get(image_name, np.array([], dtype=np.int32))

            if len(point_indices) > 0 and len(self.parser.points) > 0:
                points_world = self.parser.points[point_indices]
                points_cam = (
                    worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]
                ).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
                depths = points_cam[:, 2]  # (M,)

                if self.patch_size is not None:
                    points[:, 0] -= x
                    points[:, 1] -= y

                # Filter out points outside the image
                selector = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < image.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < image.shape[0])
                    & (depths > 0)
                )
                points = points[selector]
                depths = depths[selector]
            else:
                points = np.zeros((0, 2), dtype=np.float32)
                depths = np.zeros((0,), dtype=np.float32)

            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data