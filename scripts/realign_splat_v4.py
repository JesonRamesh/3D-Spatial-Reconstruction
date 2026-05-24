"""Align scene.ply to Y-up with a flat floor using SVD tilt correction."""

import sys
import struct
import numpy as np
from pathlib import Path

INPUT_PLY  = "outputs/splat_v4/scene.ply"
OUTPUT_PLY = "outputs/splat_v4/scene_aligned.ply"

# ──────────────────────────────────────────────────────────────────────────────
# PLY I/O  (pure numpy, no plyfile dependency)
# ──────────────────────────────────────────────────────────────────────────────

def read_ply(path: str):
    """Return (header_lines, dtype, data_array)."""
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            raw = f.readline()
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        binary_data = f.read()

    # Parse dtype from header
    prop_type_map = {
        "float": np.float32, "float32": np.float32,
        "double": np.float64, "float64": np.float64,
        "int": np.int32,   "int32": np.int32,
        "uint": np.uint32, "uint32": np.uint32,
        "uchar": np.uint8, "uint8": np.uint8,
        "char":  np.int8,  "int8":  np.int8,
        "short": np.int16, "int16": np.int16,
        "ushort": np.uint16, "uint16": np.uint16,
    }
    n_verts = 0
    properties = []
    for line in header_lines:
        parts = line.split()
        if parts[0] == "element" and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif parts[0] == "property":
            dtype_str = parts[1]
            name = parts[2]
            properties.append((name, prop_type_map[dtype_str]))

    dtype = np.dtype([(name, t) for name, t in properties])
    data = np.frombuffer(binary_data, dtype=dtype, count=n_verts).copy()
    return header_lines, dtype, data, n_verts


def write_ply(path: str, header_lines: list, data: np.ndarray):
    """Write binary little-endian PLY, preserving original header."""
    path = Path(path)
    # Rebuild header bytes (update element vertex count just in case)
    header_out = []
    for line in header_lines:
        parts = line.split()
        if parts[0] == "element" and parts[1] == "vertex":
            header_out.append(f"element vertex {len(data)}")
        else:
            header_out.append(line)
    header_str = "\n".join(header_out) + "\n"
    with open(path, "wb") as f:
        f.write(header_str.encode("ascii"))
        f.write(data.tobytes())
    print(f"  Saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a ≈ b  (both unit vectors)."""
    a = np.array(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.array(b, dtype=np.float64) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-8:
        # Already aligned (or anti-parallel)
        if c > 0:
            return np.eye(3)
        # 180° flip — pick any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v2 = np.cross(a, perp)
        v2 /= np.linalg.norm(v2)
        K = np.array([[ 0,    -v2[2],  v2[1]],
                      [ v2[2], 0,    -v2[0]],
                      [-v2[1],  v2[0],  0   ]])
        return np.eye(3) + 2.0 * (K @ K)   # Rodrigues for 180°
    kmat = np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - c) / (s ** 2))


def batch_rotate_quats(quats_wxyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Vectorised batch quaternion rotation.
    quats_wxyz : (N, 4)  — columns [w, x, y, z]
    R          : (3, 3)  — world-space rotation to pre-multiply
    Returns    : (N, 4)  — rotated quaternions [w, x, y, z]

    From PLAN.md troubleshooting section.
    """
    w = quats_wxyz[:, 0]
    x = quats_wxyz[:, 1]
    y = quats_wxyz[:, 2]
    z = quats_wxyz[:, 3]

    # Build rotation matrices for each Gaussian
    Rg = np.zeros((len(quats_wxyz), 3, 3), dtype=np.float64)
    Rg[:, 0, 0] = 1 - 2*(y*y + z*z)
    Rg[:, 0, 1] = 2*(x*y - w*z)
    Rg[:, 0, 2] = 2*(x*z + w*y)
    Rg[:, 1, 0] = 2*(x*y + w*z)
    Rg[:, 1, 1] = 1 - 2*(x*x + z*z)
    Rg[:, 1, 2] = 2*(y*z - w*x)
    Rg[:, 2, 0] = 2*(x*z - w*y)
    Rg[:, 2, 1] = 2*(y*z + w*x)
    Rg[:, 2, 2] = 1 - 2*(x*x + y*y)

    # Pre-multiply world rotation:  Rnew = R @ Rg
    Rnew = (R[None, :, :] @ Rg)  # (N, 3, 3)

    # Extract quaternion from rotation matrix (robust 4-case Shepperd method)
    trace = Rnew[:, 0, 0] + Rnew[:, 1, 1] + Rnew[:, 2, 2]

    # Allocate output
    qout = np.zeros_like(quats_wxyz, dtype=np.float64)

    # Case 0: trace > 0
    m0 = trace > 0
    if np.any(m0):
        s = 0.5 / np.sqrt(np.clip(trace[m0] + 1.0, 1e-12, None))
        qout[m0, 0] = 0.25 / s
        qout[m0, 1] = (Rnew[m0, 2, 1] - Rnew[m0, 1, 2]) * s
        qout[m0, 2] = (Rnew[m0, 0, 2] - Rnew[m0, 2, 0]) * s
        qout[m0, 3] = (Rnew[m0, 1, 0] - Rnew[m0, 0, 1]) * s

    # Case 1: R[0,0] largest
    m1 = (~m0) & (Rnew[:, 0, 0] > Rnew[:, 1, 1]) & (Rnew[:, 0, 0] > Rnew[:, 2, 2])
    if np.any(m1):
        s = 2.0 * np.sqrt(np.clip(1.0 + Rnew[m1, 0, 0] - Rnew[m1, 1, 1] - Rnew[m1, 2, 2], 1e-12, None))
        qout[m1, 0] = (Rnew[m1, 2, 1] - Rnew[m1, 1, 2]) / s
        qout[m1, 1] = 0.25 * s
        qout[m1, 2] = (Rnew[m1, 0, 1] + Rnew[m1, 1, 0]) / s
        qout[m1, 3] = (Rnew[m1, 0, 2] + Rnew[m1, 2, 0]) / s

    # Case 2: R[1,1] largest
    m2 = (~m0) & (~m1) & (Rnew[:, 1, 1] > Rnew[:, 2, 2])
    if np.any(m2):
        s = 2.0 * np.sqrt(np.clip(1.0 + Rnew[m2, 1, 1] - Rnew[m2, 0, 0] - Rnew[m2, 2, 2], 1e-12, None))
        qout[m2, 0] = (Rnew[m2, 0, 2] - Rnew[m2, 2, 0]) / s
        qout[m2, 1] = (Rnew[m2, 0, 1] + Rnew[m2, 1, 0]) / s
        qout[m2, 2] = 0.25 * s
        qout[m2, 3] = (Rnew[m2, 1, 2] + Rnew[m2, 2, 1]) / s

    # Case 3: R[2,2] largest
    m3 = (~m0) & (~m1) & (~m2)
    if np.any(m3):
        s = 2.0 * np.sqrt(np.clip(1.0 + Rnew[m3, 2, 2] - Rnew[m3, 0, 0] - Rnew[m3, 1, 1], 1e-12, None))
        qout[m3, 0] = (Rnew[m3, 1, 0] - Rnew[m3, 0, 1]) / s
        qout[m3, 1] = (Rnew[m3, 0, 2] + Rnew[m3, 2, 0]) / s
        qout[m3, 2] = (Rnew[m3, 1, 2] + Rnew[m3, 2, 1]) / s
        qout[m3, 3] = 0.25 * s

    return qout.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("realign_splat_v4.py — Z-up → Y-up + tilt correction")
    print("=" * 60)

    # ── 1. Load PLY ──────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {INPUT_PLY} ...")
    header_lines, dtype, data, n = read_ply(INPUT_PLY)
    print(f"      {n:,} Gaussians, {len(dtype.names)} properties")

    xyz = np.stack([data['x'].astype(np.float64),
                    data['y'].astype(np.float64),
                    data['z'].astype(np.float64)], axis=1)   # (N, 3)

    quats = np.stack([data['rot_0'].astype(np.float64),
                      data['rot_1'].astype(np.float64),
                      data['rot_2'].astype(np.float64),
                      data['rot_3'].astype(np.float64)], axis=1)   # (N,4) wxyz

    # ── 2. Step A: Z-up → Y-up  (-90° around X axis) ────────────────────────
    # nerfstudio exports with Z as vertical axis ("Vertical Axis: z" in header)
    # Viewer expects Y-up.  Rotation: X→X, Y→Z, Z→-Y  (i.e. Z becomes up=Y)
    # Matrix form: rotate -90° around X  =>  [[1,0,0],[0,0,1],[0,-1,0]]
    print("\n[2/6] Step A: Z-up → Y-up (rotate -90° around X) ...")
    R_zup_to_yup = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)

    xyz_a = (R_zup_to_yup @ xyz.T).T
    quats_a = batch_rotate_quats(quats, R_zup_to_yup)

    print(f"      Bbox after Z→Y rotation:")
    for i, ax in enumerate("XYZ"):
        print(f"        {ax}: [{xyz_a[:,i].min():.3f}, {xyz_a[:,i].max():.3f}]")

    # ── 3. Step B: fit floor plane to lowest 1% hi-opacity Gaussians ─────────
    print("\n[3/6] Step B: fitting residual floor tilt ...")
    opacity_raw = data['opacity'].astype(np.float64)
    high_opacity = 1.0 / (1.0 + np.exp(-opacity_raw)) > 0.5
    print(f"      High-opacity Gaussians: {high_opacity.sum():,} / {n:,}")

    # Use lowest 1% by Y after Z→Y rotation  (tightest, purest floor sample)
    y_vals_a = xyz_a[:, 1]
    y_floor_thresh = np.percentile(y_vals_a[high_opacity], 1.0)
    floor_mask = high_opacity & (y_vals_a <= y_floor_thresh)
    print(f"      Floor Gaussians (Y ≤ {y_floor_thresh:.4f}): {floor_mask.sum():,}")

    floor_pts = xyz_a[floor_mask]
    centroid = floor_pts.mean(axis=0)
    centred  = floor_pts - centroid
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    floor_normal = Vt[-1].copy()
    floor_normal /= np.linalg.norm(floor_normal)
    if floor_normal[1] > 0:
        floor_normal = -floor_normal

    tilt_deg = np.degrees(np.arccos(np.clip(
        float(np.dot(floor_normal, [0.0, -1.0, 0.0])), -1.0, 1.0)))
    print(f"      Floor normal after Z→Y: [{floor_normal[0]:+.4f}, {floor_normal[1]:+.4f}, {floor_normal[2]:+.4f}]")
    print(f"      Residual tilt: {tilt_deg:.2f}°")

    # ── 4. Step B rotation: floor_normal → [0,-1,0] ──────────────────────────
    print("\n[4/6] Computing tilt correction rotation ...")
    TARGET = np.array([0.0, -1.0, 0.0])
    R_tilt = rotation_between(floor_normal, TARGET)
    check = R_tilt @ floor_normal
    print(f"      Tilt R @ normal = [{check[0]:+.4f}, {check[1]:+.4f}, {check[2]:+.4f}]")

    # ── 5. Apply tilt correction ─────────────────────────────────────────────
    print("\n[5/6] Applying tilt correction ...")
    xyz_b = (R_tilt @ xyz_a.T).T
    quats_b = batch_rotate_quats(quats_a.astype(np.float64), R_tilt)

    print(f"      Final bbox:")
    for i, ax in enumerate("XYZ"):
        print(f"        {ax}: [{xyz_b[:,i].min():.3f}, {xyz_b[:,i].max():.3f}]")

    # Scene centroid (hi-opacity) — useful for HOME_LOOK
    cx = xyz_b[high_opacity, 0].mean()
    cy = xyz_b[high_opacity, 1].mean()
    cz = xyz_b[high_opacity, 2].mean()
    print(f"      Scene centroid (hi-opacity): [{cx:.4f}, {cy:.4f}, {cz:.4f}]")

    # Suggested HOME_POS: near centroid, shifted back, at eye level
    eye_y = np.percentile(xyz_b[high_opacity, 1], 30)  # ~lower third = eye level
    print(f"      Suggested eye-level Y (30th pct): {eye_y:.4f}")
    print(f"      Suggested HOME_POS : [{cx:.3f}, {eye_y:.3f}, {cz + 1.5:.3f}]")
    print(f"      Suggested HOME_LOOK: [{cx:.3f}, {cy:.3f}, {cz:.3f}]")

    # ── 6. Build output array and save ───────────────────────────────────────
    print(f"\n[6/6] Writing {OUTPUT_PLY} ...")
    out = data.copy()
    out['x'] = xyz_b[:, 0].astype(np.float32)
    out['y'] = xyz_b[:, 1].astype(np.float32)
    out['z'] = xyz_b[:, 2].astype(np.float32)
    out['rot_0'] = quats_b[:, 0].astype(np.float32)
    out['rot_1'] = quats_b[:, 1].astype(np.float32)
    out['rot_2'] = quats_b[:, 2].astype(np.float32)
    out['rot_3'] = quats_b[:, 3].astype(np.float32)

    write_ply(OUTPUT_PLY, header_lines, out)

    # Compose total rotation for reference
    R_total = R_tilt @ R_zup_to_yup
    print(f"\n✅ Done!")
    print(f"   Input : {INPUT_PLY}")
    print(f"   Output: {OUTPUT_PLY}")
    print(f"   Step A: Z-up → Y-up (-90° around X)")
    print(f"   Step B: Residual tilt correction ({tilt_deg:.1f}°)")
    print(f"   Total rotation matrix:\n{R_total}")
    print("\nNext steps:")
    print("  1. python3 scripts/convert_to_splat.py \\")
    print("       --input outputs/splat_v4/scene_aligned.ply \\")
    print("       --output outputs/splat_v4/scene_aligned.splat")
    print("  2. Update index.html HOME_POS/HOME_LOOK with suggested values above")
    print("  3. Restart viewer and verify orbit")


if __name__ == "__main__":
    main()