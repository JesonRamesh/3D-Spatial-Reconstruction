"""Export a gsplat checkpoint (.pt) to standard 3DGS binary PLY format.

Usage:
    python3 scripts/export_splat_ply.py \
        --ckpt   outputs/splat_v5/ckpts/ckpt_29999_rank0.pt \
        --output outputs/splat_v5/scene.ply
"""

import argparse
import sys
import numpy as np
from pathlib import Path


def load_splats(ckpt_path: Path) -> dict:
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"Checkpoint top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    if isinstance(ckpt, dict) and "splats" in ckpt:
        splats = ckpt["splats"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        splats = ckpt["model_state_dict"]
    else:
        splats = ckpt

    if hasattr(splats, "keys"):
        print(f"Splats keys: {list(splats.keys())}")

    # Convert all tensors to numpy
    out = {}
    for k, v in splats.items():
        if hasattr(v, "numpy"):
            out[k] = v.detach().cpu().numpy()
        elif hasattr(v, "cpu"):
            out[k] = v.cpu().numpy()
        else:
            out[k] = v
    return out


def build_ply(splats: dict) -> tuple[np.ndarray, list]:
    """Convert gsplat parameter dict to standard 3DGS structured array + header."""

    # ── Locate required tensors ────────────────────────────────────────
    # means: (N, 3) — positions
    for key in ("means", "xyz", "positions"):
        if key in splats:
            means = splats[key].reshape(-1, 3).astype(np.float32)
            break
    else:
        raise KeyError(f"No position key found. Available: {list(splats.keys())}")

    N = means.shape[0]
    print(f"  Gaussians: {N:,}")

    # quats: (N, 4) — rotation quaternion
    for key in ("quats", "rotation", "rots", "rot"):
        if key in splats:
            quats = splats[key].reshape(N, 4).astype(np.float32)
            norms = np.linalg.norm(quats, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            quats = quats / norms
            break
    else:
        print("  WARNING: no quaternion key found — using identity rotation")
        quats = np.zeros((N, 4), dtype=np.float32)
        quats[:, 0] = 1.0  # w=1

    # scales: (N, 3) — log scale
    for key in ("scales", "scale", "log_scales"):
        if key in splats:
            scales = splats[key].reshape(N, 3).astype(np.float32)
            break
    else:
        print("  WARNING: no scale key found — using zeros")
        scales = np.zeros((N, 3), dtype=np.float32)

    # opacities: (N,) — pre-sigmoid logit
    for key in ("opacities", "opacity", "alpha"):
        if key in splats:
            op = splats[key].reshape(N).astype(np.float32)
            break
    else:
        print("  WARNING: no opacity key found — using 0.0 logit")
        op = np.zeros(N, dtype=np.float32)

    # SH DC: try sh0 (N,1,3) or f_dc (N,3)
    sh0 = None
    for key in ("sh0", "f_dc", "features_dc", "colors"):
        if key in splats:
            raw = splats[key].astype(np.float32)
            if raw.ndim == 3:
                sh0 = raw.reshape(N, -1, 3)[:, 0, :]   # (N, 3)
            elif raw.ndim == 2:
                sh0 = raw.reshape(N, 3)
            else:
                sh0 = raw.reshape(N, 3)
            break
    if sh0 is None:
        print("  WARNING: no SH DC key found — using grey")
        SH_C0 = 0.28209479177387814
        sh0 = np.full((N, 3), (0.5 - 0.5) / SH_C0, dtype=np.float32)

    # SH rest: optional (N, K, 3) or (N, 3K)
    shN = None
    for key in ("shN", "f_rest", "features_rest", "sh_rest"):
        if key in splats:
            raw = splats[key].astype(np.float32)
            if raw.ndim == 3:
                # (N, K, 3) → flatten per-point: f_rest stores coeff-major
                # Standard 3DGS ordering: for each coeff k, R G B — so reshape to (N, K*3)
                shN = raw.reshape(N, -1)   # (N, K*3) interleaved coeff×channel
            else:
                shN = raw.reshape(N, -1)
            break

    # ── Build structured dtype ─────────────────────────────────────────
    props = [
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("nx", np.float32), ("ny", np.float32), ("nz", np.float32),
        ("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32),
    ]
    n_rest = shN.shape[1] if shN is not None else 0
    for i in range(n_rest):
        props.append((f"f_rest_{i}", np.float32))
    props += [
        ("opacity", np.float32),
        ("scale_0", np.float32), ("scale_1", np.float32), ("scale_2", np.float32),
        ("rot_0", np.float32), ("rot_1", np.float32), ("rot_2", np.float32), ("rot_3", np.float32),
    ]

    dtype = np.dtype(props)
    data = np.zeros(N, dtype=dtype)

    data["x"] = means[:, 0]
    data["y"] = means[:, 1]
    data["z"] = means[:, 2]

    data["f_dc_0"] = sh0[:, 0]
    data["f_dc_1"] = sh0[:, 1]
    data["f_dc_2"] = sh0[:, 2]

    if shN is not None:
        for i in range(n_rest):
            data[f"f_rest_{i}"] = shN[:, i]

    data["opacity"] = op
    data["scale_0"] = scales[:, 0]
    data["scale_1"] = scales[:, 1]
    data["scale_2"] = scales[:, 2]

    # gsplat v1.3.0 stores quats as (w, x, y, z) — same as 3DGS PLY rot order
    data["rot_0"] = quats[:, 0]
    data["rot_1"] = quats[:, 1]
    data["rot_2"] = quats[:, 2]
    data["rot_3"] = quats[:, 3]

    return data, props


def write_ply(path: Path, data: np.ndarray, props: list):
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(data)}",
    ]
    for name, _ in props:
        header_lines.append(f"property float {name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    mb = path.stat().st_size / 1e6
    print(f"  Saved {path}  ({mb:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Export gsplat checkpoint to 3DGS PLY")
    ap.add_argument("--ckpt",   required=True, help="Path to ckpt_XXXXX_rank0.pt")
    ap.add_argument("--output", required=True, help="Output .ply path")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.output)

    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")

    print(f"Loading {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.0f} MB)...")
    splats = load_splats(ckpt_path)

    print("Building PLY data...")
    data, props = build_ply(splats)

    print(f"Writing {out_path}...")
    write_ply(out_path, data, props)

    print(f"\nDone — {len(data):,} Gaussians → {out_path}")
    print(f"\nNext steps:")
    print(f"  python3 scripts/realign_splat_v4.py \\")
    print(f"    --input_ply  {out_path} \\")
    print(f"    --output_ply {out_path.parent}/scene_aligned.ply")


if __name__ == "__main__":
    main()
