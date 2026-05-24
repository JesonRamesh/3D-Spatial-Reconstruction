#!/usr/bin/env python3
"""
compute_confidence.py – Session 6: Confidence-Aware Scene Analysis
Novel contribution: voxel-level reconstruction confidence from point density
and camera coverage, producing a navigability map for robotic scene understanding.

Usage:
    python scripts/compute_confidence.py \
        --splat_ply outputs/splat_mast3r_v2/scene.ply \
        --point_cloud outputs/mast3r_out/room_video.ply \
        --poses_file data/vggt_out/camera_poses.json \
        --objects_file outputs/objects_3d.json \
        --output_dir outputs/ \
        --voxel_size 0.05

Pose file formats supported:
  - TUM format (.txt):  timestamp tx ty tz qx qy qz qw  (one pose per line)
  - VGGT format (.json): {"frame_XXXX.jpg": {"cam_to_world_4x4": [[...]], ...}, ...}
"""

import argparse
import json
import os
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.transform import Rotation

# ── Session 4 object colours (kept consistent across sessions) ───────────────
OBJECT_COLOURS = {
    "chair":        "#e74c3c",
    "table":        "#e67e22",
    "desk":         "#f39c12",
    "monitor":      "#3498db",
    "keyboard":     "#2980b9",
    "mouse":        "#1abc9c",
    "laptop":       "#27ae60",
    "book":         "#8e44ad",
    "bottle":       "#16a085",
    "cup":          "#d35400",
    "backpack":     "#c0392b",
    "person":       "#2c3e50",
    "tv":           "#2980b9",
    "couch":        "#7f8c8d",
    "bed":          "#6c5ce7",
    "potted plant": "#00b894",
    "door":         "#636e72",
    "window":       "#74b9ff",
    "default":      "#95a5a6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy PLY reader (no open3d — Python 3.13 incompatible on Mac)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ply_header(f):
    """Read PLY header, return (element_info, header_byte_length, is_binary_little)."""
    header_bytes = b""
    elements = []
    current_el = None
    format_str = "ascii"

    while True:
        line_bytes = f.readline()
        header_bytes += line_bytes
        line = line_bytes.decode("ascii", errors="replace").strip()

        if line.startswith("format"):
            parts = line.split()
            format_str = parts[1]          # ascii / binary_little_endian / binary_big_endian
        elif line.startswith("element"):
            parts = line.split()
            current_el = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current_el)
        elif line.startswith("property") and current_el is not None:
            parts = line.split()
            if parts[1] == "list":
                current_el["properties"].append({
                    "type": "list",
                    "count_type": parts[2],
                    "value_type": parts[3],
                    "name": parts[4],
                })
            else:
                current_el["properties"].append({
                    "type": parts[1],
                    "name": parts[2],
                })
        elif line == "end_header":
            break

    return elements, header_bytes, format_str


_PLY_DTYPE = {
    "float":   np.float32, "float32": np.float32,
    "double":  np.float64, "float64": np.float64,
    "int":     np.int32,   "int32":   np.int32,
    "uint":    np.uint32,  "uint32":  np.uint32,
    "short":   np.int16,   "int16":   np.int16,
    "ushort":  np.uint16,  "uint16":  np.uint16,
    "char":    np.int8,    "int8":    np.int8,
    "uchar":   np.uint8,   "uint8":   np.uint8,
}


def read_ply(filepath):
    """
    Read a PLY file with pure numpy.
    Returns a dict: element_name → numpy structured array.
    Only handles fixed-size (non-list) properties per element.
    """
    filepath = str(filepath)
    with open(filepath, "rb") as f:
        elements_info, header_bytes, fmt = _parse_ply_header(f)
        data_start = f.tell()
        raw = f.read()

    result = {}
    offset = 0

    is_binary = fmt in ("binary_little_endian", "binary_big_endian")
    byteorder = "<" if fmt == "binary_little_endian" else ">"

    for el in elements_info:
        name = el["name"]
        count = el["count"]
        props = el["properties"]

        # Build numpy dtype for this element (skip list properties)
        dt_fields = []
        has_list = False
        for p in props:
            if p["type"] == "list":
                has_list = True
            else:
                np_type = _PLY_DTYPE.get(p["type"])
                if np_type is not None:
                    dt_fields.append((p["name"], np_type))

        if has_list or not is_binary:
            # Fall back to slow path for ASCII or list-containing elements
            result[name] = _read_ply_element_slow(
                raw, offset, el, fmt, byteorder
            )
            # Advance offset by consuming bytes (approximate for binary; exact for ascii)
            offset = _advance_offset(raw, offset, el, fmt, byteorder)
        else:
            # rebuild with correct byte-order prefix
            dt_list = []
            for p in props:
                if p["type"] != "list":
                    np_type = _PLY_DTYPE.get(p["type"])
                    if np_type is not None:
                        dt_list.append((p["name"], np.dtype(np_type).newbyteorder(byteorder)))
            dtype = np.dtype(dt_list)
            nbytes = dtype.itemsize * count
            chunk = raw[offset: offset + nbytes]
            arr = np.frombuffer(chunk, dtype=dtype)
            # Convert to native byte order
            result[name] = arr.astype(arr.dtype.newbyteorder("="))
            offset += nbytes

    return result


def _advance_offset(raw, offset, el, fmt, byteorder):
    """Advance byte offset past an element block (binary only)."""
    count = el["count"]
    props = el["properties"]

    for p in props:
        if p["type"] == "list":
            ct_size = np.dtype(_PLY_DTYPE[p["count_type"]]).itemsize
            vt_size = np.dtype(_PLY_DTYPE[p["value_type"]]).itemsize
            for _ in range(count):
                n_vals = int(np.frombuffer(raw[offset:offset+ct_size],
                                           dtype=np.dtype(_PLY_DTYPE[p["count_type"]]).newbyteorder(byteorder))[0])
                offset += ct_size + n_vals * vt_size
        else:
            np_type = _PLY_DTYPE.get(p["type"])
            if np_type:
                offset += np.dtype(np_type).itemsize * count
    return offset


def _read_ply_element_slow(raw, offset, el, fmt, byteorder):
    """Slow path: ASCII or list-property elements. Returns structured array of non-list props."""
    count = el["count"]
    props = [p for p in el["properties"] if p["type"] != "list"]
    names = [p["name"] for p in props]
    dtype = np.dtype([(p["name"], _PLY_DTYPE.get(p["type"], np.float32)) for p in props])
    rows = np.zeros(count, dtype=dtype)

    if fmt == "ascii":
        text = raw[offset:].decode("ascii", errors="replace")
        lines = text.split("\n")
        for i in range(count):
            vals = lines[i].split()
            for j, p in enumerate(el["properties"]):
                if p["type"] != "list" and p["name"] in names:
                    col_idx = el["properties"].index(p)
                    rows[p["name"]][i] = float(vals[col_idx])
    else:
        # Binary with lists — parse row by row
        for i in range(count):
            col_idx = 0
            for p in el["properties"]:
                if p["type"] == "list":
                    ct = _PLY_DTYPE[p["count_type"]]
                    ct_size = np.dtype(ct).itemsize
                    n_vals = int(np.frombuffer(raw[offset:offset+ct_size],
                                               dtype=np.dtype(ct).newbyteorder(byteorder))[0])
                    offset += ct_size + n_vals * np.dtype(_PLY_DTYPE[p["value_type"]]).itemsize
                else:
                    np_type = _PLY_DTYPE.get(p["type"], np.float32)
                    size = np.dtype(np_type).itemsize
                    val = np.frombuffer(raw[offset:offset+size],
                                        dtype=np.dtype(np_type).newbyteorder(byteorder))[0]
                    if p["name"] in names:
                        rows[p["name"]][i] = val
                    offset += size
        return rows  # offset already advanced

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Voxel grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_voxel_grid(xyz, voxel_size, padding=0.10):
    """
    Compute voxel grid parameters from a point cloud.
    Returns (origin, shape) where origin is the min-corner of the grid.
    """
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    extent = mx - mn
    pad = extent * padding
    origin = mn - pad
    padded_extent = extent + 2 * pad
    shape = np.ceil(padded_extent / voxel_size).astype(int)
    shape = np.maximum(shape, 1)
    return origin, shape


def xyz_to_voxel(xyz, origin, voxel_size):
    """Convert Nx3 xyz coords to Nx3 integer voxel indices (clipped)."""
    return ((xyz - origin) / voxel_size).astype(int)


def voxel_centroids(origin, shape, voxel_size):
    """
    Return (N, 3) array of voxel centroid world-space positions.
    N = shape[0]*shape[1]*shape[2]
    """
    ii = np.arange(shape[0])
    jj = np.arange(shape[1])
    kk = np.arange(shape[2])
    gi, gj, gk = np.meshgrid(ii, jj, kk, indexing="ij")
    centroids = (np.stack([gi, gj, gk], axis=-1) + 0.5) * voxel_size + origin
    return centroids.reshape(-1, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Pose loading
# ─────────────────────────────────────────────────────────────────────────────

def load_poses(poses_file):
    """
    Load keyframe poses — auto-detects format:
      - .txt  → TUM format: timestamp tx ty tz qx qy qz qw
      - .json → VGGT format: {frame: {cam_to_world_4x4: [[...]], ...}}
    Returns list of dicts: {timestamp, position (3,), R_c2w (3,3)}.
    """
    poses_file = str(poses_file)
    if poses_file.endswith(".json"):
        return _load_poses_from_json(poses_file)
    else:
        return _load_poses_from_tum(poses_file)


def _load_poses_from_tum(poses_file):
    """Load poses from TUM-format text file: timestamp tx ty tz qx qy qz qw"""
    poses = []
    with open(poses_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses.append({
                "timestamp": ts,
                "position": np.array([tx, ty, tz]),
                "R_c2w": R,
            })
    return poses


def _load_poses_from_json(poses_file):
    """
    Load poses from VGGT camera_poses.json.
    Format: {"frame_XXXX.jpg": {"cam_to_world_4x4": [[4x4 matrix]], ...}}
    The cam_to_world_4x4 matrix has:
      - R_c2w = top-left 3×3
      - t_c2w = top-right 3×1 (camera position in world)
    """
    with open(poses_file, "r") as f:
        data = json.load(f)

    poses = []

    # Handle nerfstudio transforms.json format (has 'frames' list)
    if 'frames' in data:
        frame_items = []
        for frame in data['frames']:
            fp = frame.get('file_path', '')
            stem = fp.split('/')[-1]
            frame_items.append((stem, {'cam_to_world_4x4': frame['transform_matrix']}))
        frame_items.sort(key=lambda x: x[0])
        data = dict(frame_items)

    for i, (frame_name, frame_data) in enumerate(sorted(data.items())):
        c2w = frame_data.get("cam_to_world_4x4")
        if c2w is None:
            # fallback: try extrinsic (world-to-cam) and invert
            w2c = frame_data.get("world_to_cam_4x4") or frame_data.get("extrinsic_4x4")
            if w2c is None:
                continue
            mat = np.linalg.inv(np.array(w2c, dtype=np.float64))
        else:
            mat = np.array(c2w, dtype=np.float64)  # 4×4

        R_c2w = mat[:3, :3]           # rotation: cam→world
        t_c2w = mat[:3, 3]            # translation: camera position in world

        poses.append({
            "timestamp": float(i),
            "position":  t_c2w,
            "R_c2w":     R_c2w,
        })

    return poses


# ─────────────────────────────────────────────────────────────────────────────
# Confidence computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_point_density(xyz, origin, shape, voxel_size):
    """Count points per voxel, normalise by 95th percentile → [0,1]."""
    density = np.zeros(shape, dtype=np.float32)
    idx = xyz_to_voxel(xyz, origin, voxel_size)

    # Clip indices to valid range
    mask = np.all((idx >= 0) & (idx < shape), axis=1)
    idx = idx[mask]

    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

    p95 = np.percentile(density[density > 0], 95) if density.max() > 0 else 1.0
    density = np.clip(density / max(p95, 1e-6), 0.0, 1.0)
    return density


def compute_camera_coverage(poses, origin, shape, voxel_size,
                             max_dist=3.0, max_cameras=200):
    """
    Camera-centric coverage: for each camera mark voxels within max_dist
    that have a positive dot product with the camera forward vector.

    This is O(C × r³) — far cheaper than O(V × C) for large grids.
    max_cameras: subsample evenly to at most this many cameras.
    """
    counts = np.zeros(shape, dtype=np.float32)

    if not poses:
        return counts

    # Subsample cameras evenly for speed
    if len(poses) > max_cameras:
        step = max(1, len(poses) // max_cameras)
        poses_sub = poses[::step][:max_cameras]
        print(f"    Subsampled {len(poses)} → {len(poses_sub)} cameras")
    else:
        poses_sub = poses

    max_vox_radius = int(np.ceil(max_dist / voxel_size)) + 1
    shape_arr = np.array(shape)

    for cam_idx, pose in enumerate(poses_sub):
        if cam_idx % 20 == 0:
            print(f"    Camera coverage: {cam_idx}/{len(poses_sub)} ...",
                  end="\r", flush=True)

        cam_pos = pose["position"]
        cam_fwd = pose["R_c2w"][:, 2]   # third column = forward in world frame

        # Voxel bounding box around camera
        cam_vox = ((cam_pos - origin) / voxel_size).astype(int)
        lo = np.maximum(cam_vox - max_vox_radius, 0)
        hi = np.minimum(cam_vox + max_vox_radius + 1, shape_arr)

        if np.any(lo >= hi):
            continue

        ni, nj, nk = hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]

        # Centroid offsets from camera (vectorised over bounding box)
        gi = (np.arange(lo[0], hi[0])[:, None, None]
              * np.ones((1, nj, nk), dtype=np.float32))
        gj = (np.ones((ni, 1, nk), dtype=np.float32)
              * np.arange(lo[1], hi[1])[None, :, None])
        gk = (np.ones((ni, nj, 1), dtype=np.float32)
              * np.arange(lo[2], hi[2])[None, None, :])

        cx = (gi + 0.5) * voxel_size + origin[0] - cam_pos[0]
        cy = (gj + 0.5) * voxel_size + origin[1] - cam_pos[1]
        cz = (gk + 0.5) * voxel_size + origin[2] - cam_pos[2]

        dist_sq = cx**2 + cy**2 + cz**2
        dot     = cx * cam_fwd[0] + cy * cam_fwd[1] + cz * cam_fwd[2]

        visible = (dist_sq < max_dist**2) & (dot > 0)
        counts[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += visible

    print(f"    Camera coverage: {len(poses_sub)}/{len(poses_sub)} done.    ")

    max_count = counts.max()
    if max_count > 0:
        counts /= max_count
    return counts


def compute_confidence(point_density, camera_coverage,
                       w_density=0.6, w_coverage=0.4):
    """Weighted combination of density and coverage scores."""
    return w_density * point_density + w_coverage * camera_coverage


# ─────────────────────────────────────────────────────────────────────────────
# Navigability map (THE HERO FIGURE)
# ─────────────────────────────────────────────────────────────────────────────

def _make_rag_cmap():
    """Red-Amber-Green confidence colourmap."""
    colours = [
        (0.0,  "#e74c3c"),   # red   – low
        (0.3,  "#e74c3c"),
        (0.31, "#f39c12"),   # amber – medium
        (0.7,  "#f39c12"),
        (0.71, "#2ecc71"),   # green – high
        (1.0,  "#2ecc71"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "rag",
        [(v, mcolors.to_rgb(c)) for v, c in colours],
        N=256,
    )
    return cmap


def make_navigability_map(confidence_3d, origin, voxel_size,
                          objects, output_path):
    """
    Bird's-eye view: MAX confidence along Y axis.
    X axis → horizontal, Z axis → vertical (depth).
    Objects overlaid as labelled dots with Session 4 colours.
    """
    # Max-project along Y (axis=1) → shape (Nx, Nz)
    bird = confidence_3d.max(axis=1)  # (Nx, Nz)

    cmap = _make_rag_cmap()

    fig_w = max(10, bird.shape[1] / 40)
    fig_h = max(8,  bird.shape[0] / 40)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bg_colour = "#1a1a2e"
    fig.patch.set_facecolor(bg_colour)
    ax.set_facecolor(bg_colour)

    # Extent for imshow: [xmin, xmax, zmin, zmax]
    x_min = origin[0]
    x_max = origin[0] + bird.shape[0] * voxel_size
    z_min = origin[2]
    z_max = origin[2] + bird.shape[1] * voxel_size

    im = ax.imshow(
        bird.T,                          # transpose so X→cols, Z→rows
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        aspect="equal",
        interpolation="bilinear",
        alpha=0.92,
    )

    # ── Overlay object centroids ─────────────────────────────────────────────
    legend_handles = []
    plotted_labels = set()

    for obj in objects:
        label = obj.get("label", "unknown").lower()
        conf  = obj.get("reconstruction_confidence", None)
        prov  = obj.get("provenance", "unknown")

        centroid = None
        if "centroid_3d" in obj:
            c = obj["centroid_3d"]
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                centroid = c
        if centroid is None and "bbox_3d" in obj:
            bb = obj["bbox_3d"]
            if bb and len(bb) >= 6:
                centroid = [
                    (bb[0] + bb[3]) / 2,
                    (bb[1] + bb[4]) / 2,
                    (bb[2] + bb[5]) / 2,
                ]

        if centroid is None:
            continue

        cx, cy, cz = centroid[0], centroid[1], centroid[2]

        colour = OBJECT_COLOURS.get(label, OBJECT_COLOURS["default"])

        # Provenance marker style
        marker_map = {"observed": "o", "sparse": "D", "inferred": "^"}
        marker = marker_map.get(prov, "o")
        edge   = "white" if prov == "observed" else "#aaaaaa"

        sc = ax.scatter(cx, cz, s=120, c=colour, marker=marker,
                        edgecolors=edge, linewidths=1.4, zorder=5)

        # Label text
        display = label.capitalize()
        if conf is not None:
            display += f"\n{conf:.2f}"
        ax.annotate(
            display,
            xy=(cx, cz),
            xytext=(6, 6),
            textcoords="offset points",
            color="white",
            fontsize=7.5,
            fontweight="bold",
            zorder=6,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=colour,
                alpha=0.65,
                edgecolor="none",
            ),
        )

        if label not in plotted_labels:
            plotted_labels.add(label)
            legend_handles.append(
                mpatches.Patch(facecolor=colour, label=label.capitalize())
            )

    # ── Confidence colourbar ─────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Reconstruction Confidence", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Threshold markers on colourbar
    for thresh, label_text, col in [(0.3, "0.3", "#f39c12"), (0.7, "0.7", "#2ecc71")]:
        cbar.ax.axhline(thresh, color=col, linewidth=1.5, linestyle="--", alpha=0.9)
        cbar.ax.text(1.35, thresh, label_text, va="center", ha="left",
                     color=col, fontsize=8, transform=cbar.ax.transAxes)

    # ── Confidence zone legend ───────────────────────────────────────────────
    zone_patches = [
        mpatches.Patch(facecolor="#e74c3c", label="Low  (<0.3)  Dead zone"),
        mpatches.Patch(facecolor="#f39c12", label="Medium (0.3–0.7)  Sparse"),
        mpatches.Patch(facecolor="#2ecc71", label="High  (>0.7)  Well observed"),
    ]

    # Object legend (if any objects plotted)
    all_handles = zone_patches + (
        [plt.Line2D([], [], color="none")] + legend_handles if legend_handles else []
    )

    leg = ax.legend(
        handles=all_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.75,
        facecolor="#0f0f1a",
        edgecolor="#444466",
        labelcolor="white",
        title="Confidence Zones & Objects",
        title_fontsize=9,
    )
    leg.get_title().set_color("white")

    # ── Axes styling ─────────────────────────────────────────────────────────
    ax.set_xlabel("X (metres)", color="white", fontsize=11)
    ax.set_ylabel("Z (metres)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    ax.set_title(
        "RoboScene+ Navigability Map",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # Subtitle with stats
    n_high   = (confidence_3d > 0.7).sum()
    n_medium = ((confidence_3d >= 0.3) & (confidence_3d <= 0.7)).sum()
    n_low    = (confidence_3d < 0.3).sum()
    n_total  = confidence_3d.size
    subtitle = (
        f"Bird's-eye MAX projection  ·  voxel {int(voxel_size*100)}cm  ·  "
        f"High {n_high/n_total*100:.1f}%  "
        f"Medium {n_medium/n_total*100:.1f}%  "
        f"Low {n_low/n_total*100:.1f}%"
    )
    fig.text(0.5, 0.935, subtitle, ha="center", va="center",
             color="#aaaacc", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=bg_colour, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  Navigability map saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Object confidence update
# ─────────────────────────────────────────────────────────────────────────────

def update_object_confidences(objects, confidence_3d, origin, voxel_size, shape):
    """
    For each object compute mean confidence within its bounding box.
    Add reconstruction_confidence and provenance to each object dict.
    Handles both list-of-dicts and dict-of-dicts (keyed by label) formats.
    """
    # Normalise to list of dicts, each guaranteed to have 'label'
    if isinstance(objects, dict):
        obj_list = []
        for label, obj in objects.items():
            obj["label"] = obj.get("label", label)
            obj_list.append(obj)
        objects = obj_list

    for obj in objects:
        # Support both bbox_3d (flat list) and separate bbox_min/bbox_max keys
        bbox = obj.get("bbox_3d")
        if not bbox:
            bmin = obj.get("bbox_min")
            bmax = obj.get("bbox_max")
            if bmin and bmax and len(bmin) >= 3 and len(bmax) >= 3:
                bbox = [bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]]
        if not bbox or len(bbox) < 6:
            obj["reconstruction_confidence"] = 0.0
            obj["provenance"] = "inferred"
            continue

        # bbox may be stored as [min_x,min_y,min_z, max_x,max_y,max_z]
        # or as separate bbox_min / bbox_max keys
        if len(bbox) < 6:
            # try bbox_min / bbox_max
            bmin = obj.get("bbox_min", [])
            bmax = obj.get("bbox_max", [])
            if len(bmin) >= 3 and len(bmax) >= 3:
                bbox = [bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]]
            else:
                obj["reconstruction_confidence"] = 0.0
                obj["provenance"] = "inferred"
                continue
        x0, y0, z0, x1, y1, z1 = bbox[:6]

        # Convert world bbox to voxel indices
        lo = np.floor((np.array([x0, y0, z0]) - origin) / voxel_size).astype(int)
        hi = np.ceil((np.array([x1, y1, z1]) - origin) / voxel_size).astype(int) + 1

        lo = np.clip(lo, 0, shape)
        hi = np.clip(hi, 0, shape)

        if np.any(lo >= hi):
            obj["reconstruction_confidence"] = 0.0
            obj["provenance"] = "inferred"
            continue

        region = confidence_3d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if region.size == 0:
            mean_conf = 0.0
        else:
            mean_conf = float(region.mean())

        obj["reconstruction_confidence"] = round(mean_conf, 4)
        if mean_conf > 0.7:
            obj["provenance"] = "observed"
        elif mean_conf >= 0.3:
            obj["provenance"] = "sparse"
        else:
            obj["provenance"] = "inferred"

    return objects


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian provenance tagging
# ─────────────────────────────────────────────────────────────────────────────

def _ply_scalar_type_str(fmt):
    """Return PLY type string for a given numpy dtype."""
    mapping = {
        "float32": "float",   "float64": "double",
        "int32":   "int",     "uint32":  "uint",
        "int16":   "short",   "uint16":  "ushort",
        "int8":    "char",    "uint8":   "uchar",
    }
    return mapping.get(str(np.dtype(fmt)), "float")


def tag_gaussian_provenance(splat_ply, confidence_3d, origin, voxel_size,
                             shape, output_path):
    """
    Load Gaussian splat PLY, compute per-Gaussian confidence tag,
    and save augmented PLY.

    Tag values:
        0 = observed  (confidence > 0.7)
        1 = sparse    (0.3 ≤ confidence ≤ 0.7)
        2 = inferred  (confidence < 0.3)
    """
    print(f"\n  Loading Gaussian splat: {splat_ply}")
    try:
        ply_data = read_ply(splat_ply)
    except Exception as e:
        print(f"  ⚠  Could not read splat PLY ({e}). Skipping provenance tagging.")
        return

    vertex = ply_data.get("vertex")
    if vertex is None or len(vertex) == 0:
        print("  ⚠  No vertex element found in splat PLY. Skipping.")
        return

    col_names = list(vertex.dtype.names)
    print(f"  Splat columns ({len(col_names)}): {col_names[:10]}{'...' if len(col_names)>10 else ''}")
    print(f"  Number of Gaussians: {len(vertex):,}")

    # Extract XYZ — nerfstudio uses 'x','y','z'
    for xn, yn, zn in [("x","y","z"), ("X","Y","Z")]:
        if xn in col_names and yn in col_names and zn in col_names:
            gx = vertex[xn].astype(np.float64)
            gy = vertex[yn].astype(np.float64)
            gz = vertex[zn].astype(np.float64)
            break
    else:
        print("  ⚠  Could not find x/y/z columns in splat. Skipping tagging.")
        return

    xyz_gauss = np.stack([gx, gy, gz], axis=1)

    # Map each Gaussian to a voxel index
    vox_idx = xyz_to_voxel(xyz_gauss, origin, voxel_size)
    valid   = np.all((vox_idx >= 0) & (vox_idx < np.array(shape)), axis=1)

    tags = np.full(len(vertex), 2, dtype=np.uint8)  # default: inferred
    if valid.any():
        vi = vox_idx[valid]
        conf_vals = confidence_3d[vi[:, 0], vi[:, 1], vi[:, 2]]
        g_tags = np.where(conf_vals > 0.7, 0,
                          np.where(conf_vals >= 0.3, 1, 2)).astype(np.uint8)
        tags[valid] = g_tags

    print(f"  Tags — observed: {(tags==0).sum():,}  "
          f"sparse: {(tags==1).sum():,}  "
          f"inferred: {(tags==2).sum():,}")

    # Write new PLY with confidence_tag appended
    _write_ply_with_tag(splat_ply, vertex, col_names, tags, output_path)
    print(f"  ✓  Tagged splat saved → {output_path}")


def _write_ply_with_tag(original_ply, vertex, col_names, tags, output_path):
    """
    Write a new PLY preserving all original vertex data and adding
    a 'confidence_tag' uchar property.
    """
    n = len(vertex)
    tag_field = "confidence_tag"

    # Build new dtype
    new_fields = [(name, vertex.dtype[name]) for name in col_names]
    new_fields.append((tag_field, np.uint8))
    new_dtype = np.dtype(new_fields)

    new_vertex = np.zeros(n, dtype=new_dtype)
    for name in col_names:
        new_vertex[name] = vertex[name]
    new_vertex[tag_field] = tags

    # Build header
    prop_lines = []
    for name in col_names:
        dt = vertex.dtype[name]
        ply_type = _ply_scalar_type_str(str(dt))
        prop_lines.append(f"property {ply_type} {name}")
    prop_lines.append(f"property uchar {tag_field}")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "\n".join(prop_lines)
        + "\nend_header\n"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        # Write as little-endian
        out = new_vertex.astype(new_vertex.dtype.newbyteorder("<"))
        f.write(out.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(confidence_3d, objects):
    n_total  = confidence_3d.size
    n_high   = int((confidence_3d > 0.7).sum())
    n_medium = int(((confidence_3d >= 0.3) & (confidence_3d <= 0.7)).sum())
    n_low    = int((confidence_3d < 0.3).sum())

    print("\n" + "═" * 60)
    print("  RoboScene+ Confidence Analysis — Summary")
    print("═" * 60)
    print(f"  Total voxels         : {n_total:>10,}")
    print(f"  High confidence >0.7 : {n_high:>10,}  ({n_high/n_total*100:5.1f}%)")
    print(f"  Medium conf 0.3–0.7  : {n_medium:>10,}  ({n_medium/n_total*100:5.1f}%)")
    print(f"  Low conf / dead zones: {n_low:>10,}  ({n_low/n_total*100:5.1f}%)")
    print()

    if objects:
        # Table header
        col_w = [24, 12, 12]
        header = (
            f"  {'Label':<{col_w[0]}}  "
            f"{'Confidence':>{col_w[1]}}  "
            f"{'Provenance':<{col_w[2]}}"
        )
        sep = "  " + "-" * (sum(col_w) + 6)
        print(header)
        print(sep)
        for obj in sorted(objects,
                          key=lambda o: o.get("reconstruction_confidence", 0),
                          reverse=True):
            label = obj.get("label", "?")
            conf  = obj.get("reconstruction_confidence", 0.0)
            prov  = obj.get("provenance", "?")
            print(
                f"  {label:<{col_w[0]}}  "
                f"{conf:>{col_w[1]}.4f}  "
                f"{prov:<{col_w[2]}}"
            )
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute voxel-level reconstruction confidence for RoboScene+."
    )
    p.add_argument("--splat_ply",    default="outputs/splat_mast3r_v2/scene.ply")
    p.add_argument("--point_cloud",  default="outputs/mast3r_out/room_video.ply")
    p.add_argument("--poses_file",   default="data/vggt_out/camera_poses.json")
    p.add_argument("--objects_file", default="outputs/objects_3d.json")
    p.add_argument("--output_dir",   default="outputs/")
    p.add_argument("--voxel_size",   type=float, default=0.05,
                   help="Voxel side length in metres (default 0.05 = 5cm)")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_size = args.voxel_size

    # ── 1. Load MASt3R point cloud ───────────────────────────────────────────
    print(f"\n[1/9] Loading point cloud: {args.point_cloud}")
    pc_data = read_ply(args.point_cloud)
    vertex  = pc_data.get("vertex")
    if vertex is None:
        sys.exit(f"ERROR: no vertex element found in {args.point_cloud}")

    x = vertex["x"].astype(np.float64)
    y = vertex["y"].astype(np.float64)
    z = vertex["z"].astype(np.float64)
    xyz = np.stack([x, y, z], axis=1)
    print(f"  Points loaded: {len(xyz):,}")

    # ── 2. Build voxel grid ──────────────────────────────────────────────────
    print(f"\n[2/9] Building voxel grid (voxel_size={voxel_size}m)")
    origin, shape = build_voxel_grid(xyz, voxel_size, padding=0.10)
    print(f"  Origin (world): {origin.round(3)}")
    print(f"  Grid shape    : {tuple(shape)}  ({np.prod(shape):,} voxels)")

    # ── 3. Camera coverage ───────────────────────────────────────────────────
    print(f"\n[3/9] Loading poses: {args.poses_file}")
    if not Path(args.poses_file).exists():
        print(f"  ⚠  Poses file not found: {args.poses_file}")
        # Try the other common location
        alt_paths = [
            "data/vggt_out/camera_poses.json",
            "data/vggt_out_v2/camera_poses.json",
            "data/mast3r_out/slam_output/room_video.txt",
        ]
        for alt in alt_paths:
            if Path(alt).exists() and alt != args.poses_file:
                print(f"  → Using fallback: {alt}")
                args.poses_file = alt
                break
        else:
            print("  ⚠  No poses file found — camera coverage will be zero.")
            args.poses_file = None

    poses = load_poses(args.poses_file) if args.poses_file else []
    print(f"  Keyframes loaded: {len(poses)}")

    print(f"  Computing camera coverage score …")
    camera_coverage = compute_camera_coverage(
        poses, origin, shape, voxel_size, max_dist=3.0, max_cameras=200
    )
    print(f"  Coverage — min {camera_coverage.min():.3f}  "
          f"max {camera_coverage.max():.3f}  "
          f"mean {camera_coverage.mean():.3f}")

    # ── 4. Point density ─────────────────────────────────────────────────────
    print(f"\n[4/9] Computing point density score …")
    point_density = compute_point_density(xyz, origin, shape, voxel_size)
    print(f"  Density   — min {point_density.min():.3f}  "
          f"max {point_density.max():.3f}  "
          f"mean {point_density.mean():.3f}")

    # ── 5. Combined confidence ───────────────────────────────────────────────
    print(f"\n[5/9] Computing combined confidence (0.6·density + 0.4·coverage) …")
    confidence = compute_confidence(point_density, camera_coverage)

    # ── 6. Save confidence map ───────────────────────────────────────────────
    print(f"\n[6/9] Saving confidence map …")
    conf_npy = output_dir / "confidence_map.npy"
    np.save(conf_npy, confidence)
    print(f"  ✓  confidence_map.npy  → {conf_npy}")

    n_total  = int(confidence.size)
    n_high   = int((confidence > 0.7).sum())
    n_medium = int(((confidence >= 0.3) & (confidence <= 0.7)).sum())
    n_low    = int((confidence < 0.3).sum())

    metadata = {
        "voxel_size":  voxel_size,
        "origin_xyz":  origin.tolist(),
        "shape_xyz":   shape.tolist(),
        "pct_high":    round(n_high   / n_total * 100, 2),
        "pct_medium":  round(n_medium / n_total * 100, 2),
        "pct_low":     round(n_low    / n_total * 100, 2),
    }
    meta_json = output_dir / "confidence_metadata.json"
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓  confidence_metadata.json → {meta_json}")

    # ── 7. Navigability map ──────────────────────────────────────────────────
    print(f"\n[7/9] Generating navigability map (hero figure) …")

    # Load objects for overlay (may not exist yet)
    objects_raw = []
    if Path(args.objects_file).exists():
        with open(args.objects_file) as f:
            objects_raw = json.load(f)
        n_obj = len(objects_raw) if isinstance(objects_raw, list) else len(objects_raw)
        print(f"  Objects loaded for overlay: {n_obj}")
    else:
        print(f"  ⚠  objects file not found ({args.objects_file}), skipping overlay")

    # Normalise to list-of-dicts for rendering (add 'label' key from dict keys)
    if isinstance(objects_raw, dict):
        objects_list = []
        for lbl, obj in objects_raw.items():
            obj["label"] = obj.get("label", lbl)
            objects_list.append(obj)
    else:
        objects_list = objects_raw

    nav_png = output_dir / "navigability_map.png"
    make_navigability_map(confidence, origin, voxel_size, objects_list, nav_png)

    # ── 8. Update objects_3d.json ────────────────────────────────────────────
    print(f"\n[8/9] Updating object confidences …")
    if objects_list:
        objects_list = update_object_confidences(
            objects_list, confidence, origin, voxel_size, shape
        )
        # Write back in same format (dict if it was a dict)
        if isinstance(objects_raw, dict):
            out_data = {obj["label"]: obj for obj in objects_list}
        else:
            out_data = objects_list
        with open(args.objects_file, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"  ✓  {args.objects_file} updated with reconstruction_confidence & provenance")

        # Re-render navigability map with updated provenance
        make_navigability_map(confidence, origin, voxel_size, objects_list, nav_png)
    else:
        objects_list = []
        print("  (no objects to update)")

    # ── 9. Tag Gaussians ─────────────────────────────────────────────────────
    print(f"\n[9/9] Tagging Gaussian splat with provenance …")
    splat_out = output_dir / "scene_confidence_tagged.ply"
    if Path(args.splat_ply).exists():
        tag_gaussian_provenance(
            args.splat_ply, confidence, origin, voxel_size, shape, splat_out
        )
    else:
        print(f"  ⚠  Splat PLY not found at {args.splat_ply}. Skipping.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(confidence, objects_list)


if __name__ == "__main__":
    main()