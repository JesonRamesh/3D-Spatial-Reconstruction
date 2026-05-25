"""
Build the viewer-ready colored point cloud from VGGT dense output.

Input (download from bluestreak after running run_vggt.py --save_world_points):
  data/vggt_out_v3/dense_pointcloud.ply     ← 15–30M filtered points

Outputs (all on Mac, no GPU needed):
  outputs/scene_pointcloud.ply              ← final viewer PLY
  outputs/pointcloud_topdown.png            ← top-down density preview
  outputs/pointcloud_stats.json             ← point counts, bounds, density grid

Usage:
  python scripts/build_point_cloud.py
  python scripts/build_point_cloud.py --input data/vggt_out_v3/dense_pointcloud.ply
  python scripts/build_point_cloud.py --voxel_size 0.008 --max_points 20000000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Logging ────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("roboscene.pointcloud")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)
    return logger


# ── PLY reader ─────────────────────────────────────────────────────────

def read_ply_xyz_rgb(path: Path, logger: logging.Logger):
    """
    Read a binary-little-endian PLY with float xyz + uchar rgb.
    Returns (xyz: float32 (N,3), rgb: uint8 (N,3)).
    """
    logger.info(f"Reading PLY: {path}  ({path.stat().st_size / 1e6:.0f}MB)")
    t0 = time.time()

    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = 0
        props = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (type, name)

        if n_verts == 0:
            raise ValueError(f"PLY has 0 vertices: {path}")

        # Build dtype from properties
        type_map = {
            "float": "<f4", "float32": "<f4",
            "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1",
            "int": "<i4", "uint": "<u4",
            "short": "<i2", "ushort": "<u2",
        }
        dt_fields = []
        for ptype, pname in props:
            np_type = type_map.get(ptype, "<f4")
            dt_fields.append((pname, np_type))

        dt = np.dtype(dt_fields)
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)

    # Extract xyz and rgb
    xyz = np.column_stack([
        data["x"].astype(np.float32),
        data["y"].astype(np.float32),
        data["z"].astype(np.float32),
    ])

    # Try common rgb/color column names
    r_col = next((n for _, n in props if n in ("red", "r", "diffuse_red")), None)
    g_col = next((n for _, n in props if n in ("green", "g", "diffuse_green")), None)
    b_col = next((n for _, n in props if n in ("blue", "b", "diffuse_blue")), None)

    if r_col and g_col and b_col:
        rgb = np.column_stack([
            data[r_col].astype(np.uint8),
            data[g_col].astype(np.uint8),
            data[b_col].astype(np.uint8),
        ])
    else:
        logger.warning("No RGB columns found in PLY — using grey")
        rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

    logger.info(f"  Loaded {len(xyz):,} points in {time.time()-t0:.1f}s")
    return xyz, rgb


# ── PLY writer ─────────────────────────────────────────────────────────

def write_ply_xyz_rgb(pts: np.ndarray, rgb: np.ndarray, path: Path, logger: logging.Logger):
    """Write binary little-endian PLY with float XYZ + uchar RGB."""
    n = len(pts)
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

    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
    ])
    data = np.empty(n, dtype=dt)
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["r"] = rgb[:, 0]
    data["g"] = rgb[:, 1]
    data["b"] = rgb[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())

    size_mb = path.stat().st_size / 1e6
    logger.info(f"  Wrote {n:,} points → {path}  ({size_mb:.0f}MB)")


# ── Voxel downsample ───────────────────────────────────────────────────

def voxel_downsample(pts: np.ndarray, rgb: np.ndarray,
                     voxel_size: float, logger: logging.Logger):
    """Keep one point (first encountered) per voxel cell."""
    logger.info(f"  Voxel downsample at {voxel_size*100:.1f}cm...")
    t0 = time.time()

    vox = np.floor(pts / voxel_size).astype(np.int64)
    vox -= vox.min(axis=0)
    max_dim = (vox.max(axis=0) + 1).astype(np.int64)

    # Guard against overflow (room > ~100m per axis at 5mm voxels = 20000 cells)
    if int(max_dim[0]) * int(max_dim[1]) * int(max_dim[2]) > 2**40:
        logger.warning("  Room too large for linear voxel index — skipping dedup")
        return pts, rgb

    linear = (vox[:, 0] * max_dim[1] * max_dim[2] +
              vox[:, 1] * max_dim[2] +
              vox[:, 2])
    _, unique_idx = np.unique(linear, return_index=True)

    pts_out = pts[unique_idx]
    rgb_out = rgb[unique_idx]
    logger.info(f"  {len(pts):,} → {len(pts_out):,} points in {time.time()-t0:.1f}s")
    return pts_out, rgb_out


# ── Top-down PNG ───────────────────────────────────────────────────────

def save_topdown_png(pts: np.ndarray, rgb: np.ndarray,
                     output_path: Path, logger: logging.Logger,
                     cell_size: float = 0.05, up_axis: int = 1):
    """
    Render a log-normalised top-down density map colored by mean point RGB.
    up_axis: 1 = Y-up (our scene_aligned convention), 2 = Z-up.
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available — skipping top-down PNG")
        return

    logger.info("  Generating top-down PNG...")

    # Project onto ground plane
    if up_axis == 1:
        u_axis, v_axis = 0, 2   # X horizontal, Z depth
    else:
        u_axis, v_axis = 0, 1   # X horizontal, Y depth

    u = pts[:, u_axis]
    v = pts[:, v_axis]

    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()

    W = max(1, int((u_max - u_min) / cell_size) + 1)
    H = max(1, int((v_max - v_min) / cell_size) + 1)

    # Cap grid size to avoid OOM on very large scenes
    if W * H > 4_000_000:
        cell_size *= np.sqrt(W * H / 4_000_000)
        W = max(1, int((u_max - u_min) / cell_size) + 1)
        H = max(1, int((v_max - v_min) / cell_size) + 1)

    u_idx = np.clip(((u - u_min) / cell_size).astype(int), 0, W - 1)
    v_idx = np.clip(((v - v_min) / cell_size).astype(int), 0, H - 1)

    # Accumulate counts and RGB sums
    count_grid = np.zeros((H, W), dtype=np.float32)
    rgb_grid   = np.zeros((H, W, 3), dtype=np.float64)

    np.add.at(count_grid, (v_idx, u_idx), 1)
    for c in range(3):
        np.add.at(rgb_grid[:, :, c], (v_idx, u_idx), rgb[:, c].astype(np.float64))

    # Mean color per cell, log-normalised alpha
    occupied = count_grid > 0
    img = np.zeros((H, W, 4), dtype=np.uint8)
    if occupied.any():
        for c in range(3):
            ch = np.zeros((H, W), dtype=np.float64)
            ch[occupied] = rgb_grid[:, :, c][occupied] / count_grid[occupied]
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        log_cnt = np.log1p(count_grid)
        alpha = log_cnt / log_cnt.max() * 255
        img[:, :, 3] = alpha.astype(np.uint8)

    # 4× upscale for crisp pixel appearance
    scale = 4
    img_big = img.repeat(scale, axis=0).repeat(scale, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.imshow(img_big, origin="lower", interpolation="nearest")
    ax.set_title("RoboScene+ — Top-down Point Cloud View",
                 color="#00d4ff", fontsize=14, pad=10)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RoboScene+ Phase 2: Build viewer-ready point cloud from VGGT dense output"
    )
    parser.add_argument(
        "--input", type=str,
        default="data/vggt_out_v3/dense_pointcloud.ply",
        help="Path to dense_pointcloud.ply from run_vggt.py --save_world_points",
    )
    parser.add_argument(
        "--output_ply", type=str,
        default="outputs/scene_pointcloud.ply",
        help="Output path for final viewer PLY",
    )
    parser.add_argument(
        "--output_png", type=str,
        default="outputs/pointcloud_topdown.png",
        help="Output path for top-down preview PNG",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.005,
        help="Voxel size for dedup in metres (default: 0.005 = 5mm). "
             "Increase to 0.01 to reduce point count.",
    )
    parser.add_argument(
        "--max_points", type=int, default=30_000_000,
        help="Hard cap on output points (default: 30M). "
             "Reduce to 15M if viewer is slow.",
    )
    parser.add_argument(
        "--up_axis", type=int, default=1, choices=[1, 2],
        help="Up axis for top-down view: 1=Y (default, Y-up after alignment), 2=Z",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    logger = setup_logging()

    input_ply  = Path(args.input)
    output_ply = Path(args.output_ply)
    output_png = Path(args.output_png)
    if not input_ply.is_absolute():
        input_ply = project_root / input_ply
    if not output_ply.is_absolute():
        output_ply = project_root / output_ply
    if not output_png.is_absolute():
        output_png = project_root / output_png

    logger.info("=" * 62)
    logger.info("  RoboScene+ — Point Cloud Builder (Phase 2)")
    logger.info("=" * 62)

    if not input_ply.exists():
        logger.error(f"Input PLY not found: {input_ply}")
        logger.error(
            "Run on bluestreak first:\n"
            "  python scripts/run_vggt.py \\\n"
            "    --frames_dir data/frames_v3 \\\n"
            "    --output_dir data/vggt_out_v3 \\\n"
            "    --save_world_points \\\n"
            "    --batch_size 25\n"
            "Then download:\n"
            "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\\n"
            "    jrameshs@bluestreak.cs.ucl.ac.uk:"
            "/scratch0/jrameshs/roboscene-plus/data/vggt_out_v3/dense_pointcloud.ply \\\n"
            "    ~/Downloads/3D-Spatial-Reconstruction/data/vggt_out_v3/"
        )
        sys.exit(1)

    t_start = time.time()

    # Load
    xyz, rgb = read_ply_xyz_rgb(input_ply, logger)
    logger.info(f"  Bounds X: [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
                f"Y: [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
                f"Z: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    # Voxel dedup (in case bluestreak PLY has minor overlap at batch boundaries)
    xyz, rgb = voxel_downsample(xyz, rgb, args.voxel_size, logger)

    # Global cap
    if len(xyz) > args.max_points:
        logger.info(f"  Capping to {args.max_points:,} points...")
        keep = np.random.choice(len(xyz), size=args.max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]

    # Write final PLY
    write_ply_xyz_rgb(xyz, rgb, output_ply, logger)

    # Top-down preview
    save_topdown_png(xyz, rgb, output_png, logger, up_axis=args.up_axis)

    # Stats
    stats = {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "final_point_count": int(len(xyz)),
        "voxel_size_m": args.voxel_size,
        "bounds": {
            "x": [round(float(xyz[:,0].min()), 3), round(float(xyz[:,0].max()), 3)],
            "y": [round(float(xyz[:,1].min()), 3), round(float(xyz[:,1].max()), 3)],
            "z": [round(float(xyz[:,2].min()), 3), round(float(xyz[:,2].max()), 3)],
        },
        "ply_size_mb": round(output_ply.stat().st_size / 1e6, 1),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    stats_path = output_ply.parent / "pointcloud_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  POINT CLOUD BUILT")
    logger.info("=" * 62)
    logger.info(f"  Points:       {len(xyz):,}")
    logger.info(f"  PLY:          {output_ply}  ({stats['ply_size_mb']}MB)")
    logger.info(f"  Top-down PNG: {output_png}")
    logger.info(f"  Stats:        {stats_path}")
    logger.info(f"  Elapsed:      {stats['elapsed_sec']}s")
    logger.info("=" * 62)
    logger.info("")
    logger.info("  Next: update viewer.html SPLAT_CANDIDATES to include")
    logger.info(f"  the point cloud alongside the Gaussian splat.")
    logger.info("  Upload to HF Dataset:")
    logger.info("    huggingface-cli upload JesonRamesh/roboscene-data \\")
    logger.info(f"      {output_ply} scene_pointcloud.ply --repo-type dataset")


if __name__ == "__main__":
    main()
