"""
Render a fly-through frame sequence DIRECTLY from a Gaussian-splat .ply,
using gsplat's rasterization API -- no nerfstudio checkpoint required.

Why this exists
---------------
The documented "final" splat (outputs/splat_mast3r_v2/scene.ply, 4.35M
Gaussians, MASt3R poses, 60k steps) only survives locally as an exported
.ply -- its nerfstudio run directory was on wiped GPU scratch. nerfstudio's
eval_setup() needs that run dir, so export_frames.py cannot load this splat.
gsplat can rasterize the raw Gaussians straight from the .ply instead.

Design choices (deliberately defensive -- see project history of API mishaps)
-----------------------------------------------------------------------------
* Camera path is derived from the .ply's OWN geometry (an arc orbit around
  the Gaussian centroid). This sidesteps the coordinate-frame mismatch you'd
  hit trying to reuse COLMAP/VGGT poses: we never have the dataparser
  transform that maps those poses into the exported .ply's frame, so we
  build cameras in the .ply frame directly.
* Color defaults to SH degree 0 (the f_dc term only). The 45 f_rest SH
  coefficients have a channel-vs-coeff layout that differs between exporters
  and is a classic footgun; degree 0 gives view-independent color that is
  ~correct and immune to that. Pass --sh_degree 3 to use full SH once
  alignment is confirmed.
* --smoke renders just a few frames spanning the arc so you can eyeball
  alignment before committing to a full 120+ frame render.

This NEVER modifies the input .ply or any existing output -- it only writes
PNGs into --output_dir.

Requirements (already in the ironhide roboscene_env):
    torch, gsplat==1.4.0, numpy, Pillow

Usage (smoke test first):
    python scripts/render_ply.py --ply outputs/splat_mast3r_v2/scene.ply \\
        --output_dir outputs/eval_frames_smoke/ --smoke

    # then the full render once it looks right:
    python scripts/render_ply.py --ply outputs/splat_mast3r_v2/scene.ply \\
        --output_dir outputs/eval_frames/ --num_frames 120
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gsplat import rasterization

SH_C0 = 0.28209479177387814  # 0-th order SH basis constant


# ── PLY loading (pure numpy, matches the codebase's reader pattern) ─────────

def load_gaussian_ply(path: Path):
    """Read a nerfstudio/3DGS Gaussian .ply into raw attribute arrays."""
    with open(path, "rb") as f:
        # Parse ASCII header
        header_lines = []
        while True:
            line = f.readline().decode("latin-1").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertex = None
        prop_names = []
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line.startswith("property float"):
                prop_names.append(line.split()[-1])
        if n_vertex is None:
            raise ValueError("No 'element vertex' in PLY header.")

        # All properties are float32, little-endian binary
        data = np.frombuffer(
            f.read(n_vertex * len(prop_names) * 4), dtype="<f4"
        ).reshape(n_vertex, len(prop_names))

    cols = {name: i for i, name in enumerate(prop_names)}

    def grab(*names):
        return np.stack([data[:, cols[n]] for n in names], axis=1)

    means = grab("x", "y", "z")
    f_dc = grab("f_dc_0", "f_dc_1", "f_dc_2")
    f_rest_names = sorted(
        [n for n in prop_names if n.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    f_rest = grab(*f_rest_names) if f_rest_names else np.zeros((n_vertex, 0))
    opacity = data[:, cols["opacity"]]
    scales = grab("scale_0", "scale_1", "scale_2")
    quats = grab("rot_0", "rot_1", "rot_2", "rot_3")

    return means, f_dc, f_rest, opacity, scales, quats


# ── Camera path derived from the splat's own bounding geometry ─────────────

def look_at_viewmat(eye, target, up):
    """World-to-camera matrix in OpenCV convention (+z forward, +y down)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    true_up = np.cross(forward, right)  # OpenCV: y points down

    R_wc = np.stack([right, true_up, forward], axis=0)  # world->cam rotation
    t = -R_wc @ eye
    viewmat = np.eye(4, dtype=np.float64)
    viewmat[:3, :3] = R_wc
    viewmat[:3, 3] = t
    return viewmat


def build_orbit_path(means, num_frames, up_axis, arc_deg, radius_scale,
                     elevation_scale):
    """
    Build an arc of cameras around the scene centroid, all looking inward.

    Uses robust percentiles so a few stray "floater" Gaussians don't blow up
    the radius. up_axis selects which world axis is vertical (the mast3r_v2
    header declares 'Vertical Axis: z').
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}[up_axis]
    plane_idx = [i for i in range(3) if i != axis_idx]

    lo = np.percentile(means, 2, axis=0)
    hi = np.percentile(means, 98, axis=0)
    centroid = np.median(means, axis=0)
    extent = float(np.linalg.norm(hi - lo))

    radius = radius_scale * extent
    elevation = elevation_scale * extent

    up = np.zeros(3)
    up[axis_idx] = 1.0

    arc = np.deg2rad(arc_deg)
    # Center the arc so it sweeps symmetrically across the front of the scene
    angles = np.linspace(-arc / 2, arc / 2, num_frames)

    viewmats = []
    for a in angles:
        eye = centroid.copy()
        eye[plane_idx[0]] = centroid[plane_idx[0]] + radius * np.sin(a)
        eye[plane_idx[1]] = centroid[plane_idx[1]] + radius * np.cos(a)
        eye[axis_idx] = centroid[axis_idx] + elevation
        viewmats.append(look_at_viewmat(eye, centroid, up))

    return np.stack(viewmats, axis=0), centroid, extent


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", type=Path,
                   default=Path("outputs/splat_mast3r_v2/scene.ply"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs/eval_frames/"))
    p.add_argument("--num_frames", type=int, default=120)
    p.add_argument("--smoke", action="store_true",
                   help="Render only --smoke_frames frames spanning the arc, to "
                        "check alignment cheaply before a full render.")
    p.add_argument("--smoke_frames", type=int, default=6)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--up_axis", choices=["x", "y", "z"], default="z",
                   help="World vertical axis (mast3r_v2 .ply declares z).")
    p.add_argument("--arc_deg", type=float, default=120.0,
                   help="Total angular sweep of the orbit arc.")
    p.add_argument("--radius_scale", type=float, default=0.6,
                   help="Orbit radius as a fraction of scene diagonal.")
    p.add_argument("--elevation_scale", type=float, default=0.1,
                   help="Camera height above centroid as a fraction of diagonal.")
    p.add_argument("--sh_degree", type=int, default=0, choices=[0, 1, 2, 3],
                   help="0 = DC color only (safe default); 3 = full view-dependent SH.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available -- gsplat will be very slow / may fail.")

    if not args.ply.exists():
        raise FileNotFoundError(f"PLY not found: {args.ply}")

    print(f"Loading {args.ply} ...")
    means_np, f_dc, f_rest, opacity_np, scales_np, quats_np = load_gaussian_ply(args.ply)
    n = means_np.shape[0]
    print(f"Loaded {n:,} Gaussians.")

    # Activations: stored values are pre-activation (gsplat/nerfstudio convention)
    means = torch.from_numpy(means_np.astype(np.float32)).to(device)
    quats = torch.from_numpy(quats_np.astype(np.float32)).to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    scales = torch.from_numpy(np.exp(scales_np.astype(np.float32))).to(device)
    opacities = torch.sigmoid(torch.from_numpy(opacity_np.astype(np.float32))).to(device)

    # Colors as SH coefficients [N, K, 3]; K = (sh_degree+1)^2
    k = (args.sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), dtype=np.float32)
    sh[:, 0, :] = f_dc  # DC term
    if args.sh_degree >= 1 and f_rest.shape[1] >= (k - 1) * 3:
        # nerfstudio stores f_rest channel-major: [c0_all_coeffs, c1..., c2...]
        rest = f_rest.reshape(n, 3, -1).transpose(0, 2, 1)  # -> [N, coeff, channel]
        sh[:, 1:k, :] = rest[:, : k - 1, :]
    colors = torch.from_numpy(sh).to(device)

    # Intrinsics from FOV
    fov = np.deg2rad(args.fov_deg)
    fx = fy = (args.width / 2.0) / np.tan(fov / 2.0)
    cx, cy = args.width / 2.0, args.height / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     dtype=torch.float32, device=device)

    num_frames = args.smoke_frames if args.smoke else args.num_frames
    viewmats_np, centroid, extent = build_orbit_path(
        means_np, num_frames, args.up_axis, args.arc_deg,
        args.radius_scale, args.elevation_scale,
    )
    print(f"Scene centroid={np.round(centroid,3)} diagonal={extent:.3f}m")
    print(f"Rendering {num_frames} frames "
          f"({'SMOKE TEST' if args.smoke else 'full'}), "
          f"SH degree {args.sh_degree}, arc {args.arc_deg}deg ...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    viewmats = torch.from_numpy(viewmats_np.astype(np.float32)).to(device)

    for i in range(num_frames):
        with torch.no_grad():
            render, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats[i : i + 1],
                Ks=K[None],
                width=args.width,
                height=args.height,
                sh_degree=args.sh_degree,
                render_mode="RGB",
            )
        img = render[0].clamp(0, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        out = args.output_dir / f"{i + 1:06d}.png"
        Image.fromarray(img).save(out)
        if (i + 1) % 10 == 0 or (i + 1) == num_frames or args.smoke:
            print(f"  rendered {i + 1}/{num_frames} -> {out.name}")

    print(f"Done. {num_frames} frames in {args.output_dir}/")
    if args.smoke:
        print("Smoke test complete -- eyeball these, then rerun without --smoke "
              "(adjust --up_axis/--radius_scale/--arc_deg if the orbit is off).")


if __name__ == "__main__":
    main()
