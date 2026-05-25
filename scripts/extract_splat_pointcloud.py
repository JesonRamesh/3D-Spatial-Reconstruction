"""
Extract a viewer-ready colored point cloud from the Gaussian Splat PLY.

Why this exists:
  dense_pointcloud.ply (from VGGT world_points) is distorted because
  run_vggt.py stitch_batch_poses() aligns the *extrinsic* poses but does NOT
  re-unproject the world_points — each batch's geometry stays in its own local
  coordinate frame. The Gaussian Splat is unaffected (it trains from the stitched
  COLMAP extrinsics), so its Gaussian positions are globally consistent.

  Solution: extract XYZ + color from the Gaussian Splat PLY, which already has
  correct room geometry.

Why walls look patchy without --inflate:
  Gaussian Splat optimiser represents flat surfaces (walls, floor, ceiling) with a
  FEW LARGE Gaussians (scale 5–50cm). In point cloud mode, a 2m wall = 5–20 dots.
  The --inflate flag fixes this: each large Gaussian is sampled into N surface points
  proportional to its physical area, making walls dense and continuous.

Input:
  outputs/splat_v4/scene_semantic_v3.ply   (2.41M Gaussians, Y-up, semantic colors)

Output:
  outputs/scene_pointcloud_web.ply         (viewer-ready, float XYZ + uchar RGB)

Usage:
  python scripts/extract_splat_pointcloud.py                    # basic
  python scripts/extract_splat_pointcloud.py --inflate          # dense walls (recommended)
  python scripts/extract_splat_pointcloud.py --inflate --spacing 0.005 --max_points 6000000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814   # DC coefficient for converting SH → RGB


def read_splat_ply(path: Path):
    """
    Read a nerfstudio/gsplat Gaussian PLY.
    Returns dict of named arrays: xyz, f_dc, opacity_raw, scale, rot.
    """
    print(f"Reading: {path}  ({path.stat().st_size / 1e6:.0f}MB)")
    t0 = time.time()

    with open(path, "rb") as f:
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
                props.append((parts[1], parts[2]))

        if n_verts == 0:
            raise ValueError(f"PLY has 0 vertices: {path}")

        type_map = {
            "float": "<f4", "float32": "<f4",
            "double": "<f8", "float64": "<f8",
            "uchar": "u1", "uint8": "u1",
            "int": "<i4", "uint": "<u4",
        }
        dt_fields = [(name, type_map.get(ptype, "<f4")) for ptype, name in props]
        dt = np.dtype(dt_fields)
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)

    print(f"  Loaded {n_verts:,} Gaussians in {time.time()-t0:.1f}s")

    prop_names = [nm for _, nm in props]
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    f_dc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    opacity_raw = data["opacity"].astype(np.float32)

    # Scale (log-space) and rotation (quaternion w,x,y,z) — needed for inflation
    has_scale = all(f"scale_{i}" in prop_names for i in range(3))
    has_rot   = all(f"rot_{i}" in prop_names for i in range(4))
    scale = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32) if has_scale else None
    rot   = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32) if has_rot else None

    return xyz, f_dc, opacity_raw, scale, rot


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    (N,4) quaternion [w,x,y,z] → (N,3,3) rotation matrices.
    nerfstudio convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z.
    Quaternions are normalised here since nerfstudio PLY output is not guaranteed
    to be unit quaternions — non-unit quats produce non-orthogonal R and wildly
    out-of-bounds point positions after rotation.
    """
    # Normalise to unit quaternions
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)   # avoid division by zero
    q = q / norm

    w = q[:, 0]; x = q[:, 1]; y = q[:, 2]; z = q[:, 3]
    n = len(q)
    R = np.empty((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def inflate_gaussians(
    xyz: np.ndarray,
    f_dc: np.ndarray,
    scale_log: np.ndarray,
    rot_q: np.ndarray,
    spacing: float = 0.007,
    inflate_threshold: float = 0.010,
    batch_size: int = 10_000,
) -> tuple:
    """
    For each Gaussian larger than inflate_threshold, sample multiple surface points
    distributed over the Gaussian's ellipsoid proportional to its physical area.
    Small Gaussians (< inflate_threshold) keep their single centre point.

    This converts a few large wall/floor Gaussians into dense surface samples,
    eliminating black patches where flat surfaces were under-sampled.

    Args:
        xyz:               (N, 3) Gaussian centres
        f_dc:              (N, 3) DC SH coefficients (colour)
        scale_log:         (N, 3) log-scale values (nerfstudio stores log-scale)
        rot_q:             (N, 4) quaternion [w,x,y,z]
        spacing:           target point spacing in metres (default 7mm)
        inflate_threshold: Gaussians with max-scale > this get inflated (default 1cm)
        batch_size:        process this many Gaussians at once to bound peak RAM

    Returns:
        pts:  (M, 3) float32 — inflated point positions
        cols: (M, 3) float32 — corresponding DC SH values (for later RGB conversion)
    """
    t0 = time.time()
    scales = np.exp(scale_log)          # (N, 3) actual scale in scene units
    max_scale = scales.max(axis=1)       # (N,)

    small_mask = max_scale < inflate_threshold
    large_mask = ~small_mask

    n_small = small_mask.sum()
    n_large = large_mask.sum()
    print(f"  Inflate: {n_small:,} small Gaussians (keep centre) + "
          f"{n_large:,} large Gaussians (sample surface)  [spacing={spacing*100:.1f}cm]")

    # ── Small Gaussians: just use their centres ──────────────────────────
    pts_list  = [xyz[small_mask]]
    cols_list = [f_dc[small_mask]]

    # ── Large Gaussians: sample surface points in batches ────────────────
    idxs_large = np.where(large_mask)[0]
    n_total_samples = 0

    for batch_start in range(0, len(idxs_large), batch_size):
        batch_idx = idxs_large[batch_start: batch_start + batch_size]
        b_xyz    = xyz[batch_idx]         # (B, 3)
        b_fdc    = f_dc[batch_idx]        # (B, 3)
        b_scales = scales[batch_idx]      # (B, 3)
        b_rot    = rot_q[batch_idx]       # (B, 4)

        # Sort scales descending so axis-0 is the largest (used as the surface plane)
        b_scales_sorted = np.sort(b_scales, axis=1)[:, ::-1]  # (B, 3) desc
        s0 = b_scales_sorted[:, 0]   # largest scale
        s1 = b_scales_sorted[:, 1]   # second scale
        # s2 = b_scales_sorted[:, 2] # smallest = thin axis (normal direction)

        # Number of samples: fill a 2D grid over the flat face of the ellipsoid
        # n = ceil(2*s0/spacing) × ceil(2*s1/spacing), capped to avoid huge outputs
        n0 = np.clip(np.ceil(2 * s0 / spacing).astype(int), 1, 200)
        n1 = np.clip(np.ceil(2 * s1 / spacing).astype(int), 1, 200)
        n_samples = n0 * n1                # per Gaussian

        R = quat_to_rotmat(b_rot)          # (B, 3, 3)

        batch_pts  = []
        batch_cols = []

        for i in range(len(batch_idx)):
            ns = int(n_samples[i])
            if ns <= 1:
                batch_pts.append(b_xyz[i:i+1])
                batch_cols.append(b_fdc[i:i+1])
                continue

            # Regular grid in local frame over the two major axes, z=0 (flat face)
            u = np.linspace(-s0[i], s0[i], n0[i])
            v = np.linspace(-s1[i], s1[i], n1[i])
            uu, vv = np.meshgrid(u, v)
            # Use the rotation's first two columns (most spread directions)
            # Local coords: (u along axis-0, v along axis-1, 0 along axis-2)
            local = np.column_stack([uu.ravel(), vv.ravel(),
                                     np.zeros(n0[i] * n1[i], dtype=np.float32)])
            # Rotate to world frame
            world = (R[i] @ local.T).T + b_xyz[i]   # (ns, 3)
            color = np.tile(b_fdc[i], (len(world), 1))

            batch_pts.append(world.astype(np.float32))
            batch_cols.append(color.astype(np.float32))
            n_total_samples += len(world)

        if batch_pts:
            pts_list.append(np.concatenate(batch_pts, axis=0))
            cols_list.append(np.concatenate(batch_cols, axis=0))

        if (batch_start // batch_size) % 5 == 0:
            done = min(batch_start + batch_size, len(idxs_large))
            print(f"    {done:,}/{len(idxs_large):,} large Gaussians inflated  "
                  f"({n_total_samples:,} surface samples so far)", end="\r")

    print()
    pts  = np.concatenate(pts_list,  axis=0)
    cols = np.concatenate(cols_list, axis=0)
    print(f"  Inflation complete: {len(pts):,} points in {time.time()-t0:.1f}s")


    return pts, cols


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """Convert SH DC term to uint8 RGB.  rgb = f_dc * SH_C0 + 0.5"""
    rgb_f = f_dc * SH_C0 + 0.5
    return np.clip(rgb_f * 255, 0, 255).astype(np.uint8)


def write_ply_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray, path: Path):
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

    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rec = np.empty(n, dtype=dt)
    rec["x"] = xyz[:, 0]; rec["y"] = xyz[:, 1]; rec["z"] = xyz[:, 2]
    rec["r"] = rgb[:, 0]; rec["g"] = rgb[:, 1]; rec["b"] = rgb[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())
    print(f"  Wrote {n:,} points → {path}  ({path.stat().st_size/1e6:.0f}MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract viewer-ready point cloud from Gaussian Splat PLY"
    )
    parser.add_argument("--input", default="outputs/splat_v4/scene_semantic_v3.ply")
    parser.add_argument("--output", default="outputs/scene_pointcloud_web.ply")
    parser.add_argument("--opacity_min", type=float, default=0.1,
                        help="Minimum post-sigmoid opacity to keep (default 0.1).")
    parser.add_argument("--max_points", type=int, default=5_000_000,
                        help="Hard cap on output points (default 5M)")
    parser.add_argument("--inflate", action="store_true", default=False,
                        help="Sample multiple surface points per large Gaussian to fill "
                             "wall/floor patches. Highly recommended — eliminates black "
                             "holes in flat surfaces.")
    parser.add_argument("--spacing", type=float, default=0.007,
                        help="Target point spacing in metres for inflation (default 7mm). "
                             "Lower = denser walls but more points.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    inp  = Path(args.input) if Path(args.input).is_absolute() else root / args.input
    out  = Path(args.output) if Path(args.output).is_absolute() else root / args.output

    if not inp.exists():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    xyz, f_dc, opacity_raw, scale_log, rot_q = read_splat_ply(inp)

    # Sigmoid opacity
    opacity = 1.0 / (1.0 + np.exp(-opacity_raw))

    # Step 1: spatial crop — P2–P98 of core Gaussians + 30% padding.
    # Excludes background floaters at ±5–6m.
    core = opacity > 0.5
    if core.sum() > 100:
        xc = xyz[core]
        lo = np.percentile(xc, 2, axis=0)
        hi = np.percentile(xc, 98, axis=0)
        pad = (hi - lo) * 0.30
        lo -= pad; hi += pad
        in_box = (
            (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0]) &
            (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1]) &
            (xyz[:, 2] >= lo[2]) & (xyz[:, 2] <= hi[2])
        )
        print(f"  Spatial crop P2-P98+30%: "
              f"X[{lo[0]:.2f},{hi[0]:.2f}] Y[{lo[1]:.2f},{hi[1]:.2f}] Z[{lo[2]:.2f},{hi[2]:.2f}]")
        print(f"  After spatial crop: {in_box.sum():,} / {len(xyz):,} kept")
    else:
        in_box = np.ones(len(xyz), dtype=bool)

    # Step 2: opacity filter
    keep = in_box & (opacity >= args.opacity_min)
    print(f"  After opacity >= {args.opacity_min}: {keep.sum():,} kept")

    xyz       = xyz[keep]
    f_dc      = f_dc[keep]
    scale_log = scale_log[keep] if scale_log is not None else None
    rot_q     = rot_q[keep]    if rot_q    is not None else None

    # Step 3 (optional): inflate large Gaussians into surface samples
    if args.inflate and scale_log is not None and rot_q is not None:
        print("  Inflating large Gaussians into surface points...")
        bbox_lo = xyz.min(axis=0)
        bbox_hi = xyz.max(axis=0)
        xyz, f_dc = inflate_gaussians(
            xyz, f_dc, scale_log, rot_q,
            spacing=args.spacing,
            inflate_threshold=0.020,   # Gaussians > 2cm get inflated (walls/floor/ceiling)
        )
        # Clamp inflated points to the pre-inflation bounding box + 20% margin
        # Non-unit quats or numerical noise can push samples slightly outside
        pad = (bbox_hi - bbox_lo) * 0.20
        in_bbox = np.all((xyz >= bbox_lo - pad) & (xyz <= bbox_hi + pad), axis=1)
        if not in_bbox.all():
            print(f"  Clamped {(~in_bbox).sum():,} out-of-bounds inflated points")
            xyz  = xyz[in_bbox]
            f_dc = f_dc[in_bbox]
    else:
        if args.inflate:
            print("  --inflate requested but scale/rot not available in PLY — skipping")

    # Convert SH DC → RGB
    rgb = sh_dc_to_rgb(f_dc)

    print(f"  Final bounds: X[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
          f"Y[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
          f"Z[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")
    print(f"  Total points before cap: {len(xyz):,}")

    if len(xyz) > args.max_points:
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        xyz = xyz[idx]; rgb = rgb[idx]
        print(f"  Capped to {args.max_points:,} points")

    write_ply_xyz_rgb(xyz, rgb, out)
    print("Done.")


if __name__ == "__main__":
    main()
