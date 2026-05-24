"""Convert a 3DGS .ply to the compact .splat binary format used by GaussianSplats3D.js."""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ── PLY reader ────────────────────────────────────────────────────────────────

def _read_ply_header(path: Path):
    """Return (header_text, data_offset, n_vertices, property_names)."""
    lines = []
    offset = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            offset += len(line)
            decoded = line.decode("ascii", errors="replace").strip()
            lines.append(decoded)
            if decoded == "end_header":
                break
    header = "\n".join(lines)
    n_vertices = 0
    props = []
    for line in lines:
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("property float "):
            props.append(line.split()[-1])
    return header, offset, n_vertices, props


def load_ply_gaussians(path: Path):
    """Load the Gaussian PLY, return a structured numpy array."""
    header, data_offset, n_verts, props = _read_ply_header(path)
    print(f"[OK] {n_verts:,} Gaussians  ({len(props)} float properties)")

    dtype = np.dtype([(p, np.float32) for p in props])
    with open(path, "rb") as f:
        f.seek(data_offset)
        data = np.frombuffer(f.read(), dtype=dtype)

    if len(data) != n_verts:
        raise ValueError(f"Expected {n_verts} rows, got {len(data)}")
    return data


# ── Conversion ────────────────────────────────────────────────────────────────

def ply_to_splat(data: np.ndarray) -> bytes:
    n = len(data)

    # Position (pass through)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

    # Scale: exp(log_scale) → actual Gaussian radii
    scale = np.stack([
        np.exp(data["scale_0"]),
        np.exp(data["scale_1"]),
        np.exp(data["scale_2"]),
    ], axis=1).astype(np.float32)

    # Color: inverse-SH DC component → linear → sigmoid → [0,255]
    r = np.clip(sigmoid(data["f_dc_0"] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    g = np.clip(sigmoid(data["f_dc_1"] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    b = np.clip(sigmoid(data["f_dc_2"] * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    a = np.clip(sigmoid(data["opacity"]) * 255, 0, 255).astype(np.uint8)

    # Rotation quaternion: normalize then map [-1,1] → [0,255]
    rot = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1)
    norms = np.linalg.norm(rot, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    rot_n = (rot / norms)
    rot_u8 = np.clip((rot_n + 1.0) * 0.5 * 255, 0, 255).astype(np.uint8)

    # Pack 32 bytes per Gaussian using a uint8 view
    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12]  = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
    buf[:, 24]    = r
    buf[:, 25]    = g
    buf[:, 26]    = b
    buf[:, 27]    = a
    buf[:, 28:32] = rot_u8

    return buf.tobytes()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert 3DGS .ply to .splat (8× smaller)")
    ap.add_argument("--input",  default="outputs/scene_semantic.ply",  help="Input PLY")
    ap.add_argument("--output", default="outputs/scene_semantic.splat", help="Output .splat")
    args = ap.parse_args()

    ply_path   = Path(args.input)
    splat_path = Path(args.output)

    if not ply_path.exists():
        sys.exit(f"[ERROR] Input not found: {ply_path}")

    print(f"[...] Loading {ply_path}  ({ply_path.stat().st_size / 1e9:.2f} GB)")
    data = load_ply_gaussians(ply_path)

    print(f"[...] Converting {len(data):,} Gaussians to .splat format…")
    raw = ply_to_splat(data)

    splat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splat_path, "wb") as f:
        f.write(raw)

    size_mb = len(raw) / 1e6
    orig_mb = ply_path.stat().st_size / 1e6
    print(f"[OK] Written → {splat_path}  ({size_mb:.0f} MB  vs  {orig_mb:.0f} MB PLY,  {orig_mb/size_mb:.1f}× smaller)")


if __name__ == "__main__":
    main()
