"""
Patched COLMAP dataset loader for gsplat's simple_trainer.

Replaces pycolmap.SceneManager with our custom binary readers from colmap_utils.py.
This avoids the broken pycolmap dependency on UCL GPU machines.

Drop-in replacement: placed as gsplat_examples/datasets/colmap.py by train_splat.py.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

# Import our custom COLMAP binary readers
# When this file is copied to gsplat_examples/datasets/colmap.py,
# we need to find colmap_utils.py in the scripts/ directory.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.dirname(_this_dir)  # gsplat_examples -> scripts
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
# Also try two levels up (gsplat_examples/datasets -> gsplat_examples -> scripts)
_scripts_dir2 = os.path.dirname(os.path.dirname(_this_dir))
if _scripts_dir2 not in sys.path:
    sys.path.insert(0, _scripts_dir2)

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
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # Find COLMAP sparse directory
        colmap_dir = os.path.join(data_dir, "sparse", "0")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(colmap_dir), (
            f"COLMAP directory {colmap_dir} does not exist."
        )

        # Load binary files using our custom readers
        cameras_bin = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))
        images_bin = read_images_binary(os.path.join(colmap_dir, "images.bin"))
        points3d_bin = read_points3d_binary(os.path.join(colmap_dir, "points3D.bin"))

        # Extract extrinsic matrices in world-to-camera format
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()
        image_names_unsorted = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for img_id in sorted(images_bin.keys()):
            im = images_bin[img_id]
            # Rotation from quaternion
            R = qvec_to_rotmat(im["qvec"])
            t = im["tvec"].reshape(3, 1)
            w2c = np.concatenate([np.concatenate([R, t], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im["camera_id"]
            camera_ids.append(camera_id)
            image_names_unsorted.append(im["name"])

            # Camera intrinsics (only process each camera_id once)
            if camera_id not in Ks_dict:
                cam = cameras_bin[camera_id]
                model = cam["model"]
                params = cam["params"]
                width = cam["width"]
                height = cam["height"]

                # Extract fx, fy, cx, cy based on model
                if model in ("SIMPLE_PINHOLE",):
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                    dist_params = np.empty(0, dtype=np.float32)
                elif model in ("PINHOLE",):
                    fx, fy = params[0], params[1]
                    cx, cy = params[2], params[3]
                    dist_params = np.empty(0, dtype=np.float32)
                elif model in ("SIMPLE_RADIAL",):
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                    dist_params = np.array([params[3], 0.0, 0.0, 0.0], dtype=np.float32)
                elif model in ("RADIAL",):
                    fx = fy = params[0]
                    cx, cy = params[1], params[2]
                    dist_params = np.array([params[3], params[4], 0.0, 0.0], dtype=np.float32)
                elif model in ("OPENCV",):
                    fx, fy = params[0], params[1]
                    cx, cy = params[2], params[3]
                    dist_params = np.array(
                        [params[4], params[5], params[6], params[7]], dtype=np.float32
                    )
                else:
                    # Default: treat as PINHOLE
                    fx, fy = params[0], params[1] if len(params) > 1 else params[0]
                    cx = params[2] if len(params) > 2 else width / 2
                    cy = params[3] if len(params) > 3 else height / 2
                    dist_params = np.empty(0, dtype=np.float32)

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                K[:2, :] /= factor
                Ks_dict[camera_id] = K
                params_dict[camera_id] = dist_params
                imsize_dict[camera_id] = (width // factor, height // factor)

        print(
            f"[Parser] {len(images_bin)} images, "
            f"taken by {len(set(camera_ids))} cameras."
        )

        if len(images_bin) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world
        camtoworlds = np.linalg.inv(w2c_mats)

        # Sort by image name (consistent with original gsplat loader)
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

        # Map between COLMAP image names and actual files on disk
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [
            os.path.join(image_dir, colmap_to_image.get(f, f))
            for f in image_names
        ]

        # 3D points
        if points3d_bin:
            # Build arrays from points3d dict
            point3d_ids_sorted = sorted(points3d_bin.keys())
            points = np.array(
                [points3d_bin[pid]["xyz"] for pid in point3d_ids_sorted],
                dtype=np.float32,
            )
            points_err = np.array(
                [points3d_bin[pid]["error"] for pid in point3d_ids_sorted],
                dtype=np.float32,
            )
            points_rgb = np.array(
                [points3d_bin[pid]["rgb"] for pid in point3d_ids_sorted],
                dtype=np.uint8,
            )

            # Build point3D_id -> index mapping
            point3d_id_to_idx = {
                pid: idx for idx, pid in enumerate(point3d_ids_sorted)
            }

            # Build {image_name -> [point_idx]} from image 2D point observations
            point_indices = dict()
            for img_id in sorted(images_bin.keys()):
                im = images_bin[img_id]
                name = im["name"]
                pt3d_ids = im["point3D_ids"]
                valid = pt3d_ids >= 0
                if valid.any():
                    idxs = [
                        point3d_id_to_idx[pid]
                        for pid in pt3d_ids[valid]
                        if pid in point3d_id_to_idx
                    ]
                    if idxs:
                        point_indices[name] = np.array(idxs, dtype=np.int32)
        else:
            points = np.zeros((0, 3), dtype=np.float32)
            points_err = np.zeros((0,), dtype=np.float32)
            points_rgb = np.zeros((0, 3), dtype=np.uint8)
            point_indices = dict()

        # Normalize the world space
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            # align_principle_axes can fail with degenerate point clouds
            # (e.g. very few points or coplanar points from SLAM)
            try:
                if len(points) > 10:
                    T2 = align_principle_axes(points)
                    camtoworlds = transform_cameras(T2, camtoworlds)
                    points = transform_points(T2, points)
                    transform = T2 @ T1
                else:
                    transform = T1
            except np.linalg.LinAlgError:
                print("[Parser] Warning: align_principle_axes failed (degenerate points). Skipping.")
                transform = T1
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        # Load one image to check actual size vs COLMAP stored size
        # (handles cases where stored intrinsics correspond to upsampled images)
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

        # Scene scale measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


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
        K = self.parser.Ks_dict[camera_id].copy()
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

            if len(point_indices) > 0:
                points_world = self.parser.points[point_indices]
                points_cam = (
                    worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]
                ).T
                points_proj = (K @ points_cam.T).T
                points = points_proj[:, :2] / points_proj[:, 2:3]
                depths = points_cam[:, 2]

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