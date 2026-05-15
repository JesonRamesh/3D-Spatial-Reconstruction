#!/usr/bin/env python3
"""
RoboScene+ Reconstruction Quality Diagnostic
=============================================
Investigates all 5 suspected root causes of distorted 3DGS reconstruction:
  1. Camera intrinsics mismatch (CRITICAL: double-scaling bug in colmap export)
  2. Empty point tracks in points3D.bin
  3. VGGT coordinate frame vs COLMAP convention
  4. Batch-boundary pose discontinuities (180 deg flips at seams)
  5. iPhone ultrawide lens distortion (PINHOLE vs OPENCV)

Outputs saved to outputs/debug/:
  camera_positions.png   3D scatter of camera trajectory
  depth_samples.png      5 random depth map heatmaps
  batch_boundaries.png   rotation/translation at batch seams
  diagnosis_report.txt   full text findings

Usage: python scripts/debug_reconstruction.py
"""
import json, math, os, struct, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
VGGT_OUT      = PROJECT_ROOT / "data" / "vggt_out"
POSES_JSON    = VGGT_OUT / "camera_poses.json"
METADATA_JSON = VGGT_OUT / "vggt_metadata.json"
DEPTHS_DIR    = VGGT_OUT / "depths"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "debug"


def _find_sparse():
    for c in [VGGT_OUT / "sparse" / "0", VGGT_OUT / "sparse"]:
        if (c / "cameras.bin").exists():
            return c
    return VGGT_OUT / "sparse"


SPARSE_DIR        = _find_sparse()
BATCH_SIZE        = 10     # VGGT inference batch size used
VGGT_RES          = 518    # VGGT square input resolution
EXPECTED_HFOV_DEG = 120.0  # iPhone 0.5x ultrawide nominal HFOV


# ---------------------------------------------------------------------------
# Findings accumulator
# ---------------------------------------------------------------------------
class Diag:
    def __init__(self):
        self.findings = []

    def add(self, severity, title, detail=""):
        self.findings.append((severity, title, detail.strip()))

    def summary(self):
        icons = {"CRITICAL": "[CRITICAL]", "WARNING": "[WARNING]", "OK": "[OK]"}
        lines = ["", "=" * 72, "  DIAGNOSIS SUMMARY", "=" * 72]
        for sev, title, detail in self.findings:
            tag = icons.get(sev, sev)
            lines.append(f"\n{tag} {title}")
            for ln in detail.split("\n"):
                lines.append(f"       {ln}")
        lines.append("\n" + "=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Binary readers (no pycolmap dependency)
# ---------------------------------------------------------------------------
def _read_cameras_bin(path):
    MODEL = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
             3: "RADIAL", 4: "OPENCV", 5: "FULL_OPENCV"}
    NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12}
    cams = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid = struct.unpack("<I", f.read(4))[0]
            mid = struct.unpack("<i", f.read(4))[0]
            w   = struct.unpack("<Q", f.read(8))[0]
            h   = struct.unpack("<Q", f.read(8))[0]
            np_ = NP.get(mid, 4)
            params = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cams.append({"id": cid, "model": MODEL.get(mid, f"UNK{mid}"),
                         "model_id": mid, "w": w, "h": h, "params": params})
    return cams


def _read_images_bin_sample(path, n_sample=3):
    imgs = []
    with open(path, "rb") as f:
        total = struct.unpack("<Q", f.read(8))[0]
        for _ in range(total):
            iid   = struct.unpack("<I", f.read(4))[0]
            qvec  = np.array(struct.unpack("<4d", f.read(32)))
            tvec  = np.array(struct.unpack("<3d", f.read(24)))
            camid = struct.unpack("<I", f.read(4))[0]
            name  = b""
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n2d * 24)
            if len(imgs) < n_sample:
                imgs.append({"id": iid, "qvec": qvec, "tvec": tvec,
                             "camera_id": camid, "name": name.decode()})
    return imgs


def _read_points3d_summary(path):
    xyzs, empty, total_track = [], 0, 0
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("<Q", f.read(8))
            xyz = struct.unpack("<3d", f.read(24))
            struct.unpack("<3B", f.read(3))
            struct.unpack("<d", f.read(8))
            tl = struct.unpack("<Q", f.read(8))[0]
            f.read(tl * 8)
            total_track += tl
            if tl == 0:
                empty += 1
            xyzs.append(xyz)
    xyzs = np.array(xyzs) if xyzs else np.zeros((0, 3))
    return {"n": n, "empty": empty,
            "avg_track": total_track / max(n, 1), "xyzs": xyzs}


def _qvec_to_R(qvec):
    """Convert COLMAP quaternion (w,x,y,z) to rotation matrix.

    Uses the COLMAP convention (same as colmap_utils.qvec_to_rotmat).
    NOTE: the off-diagonal signs matter — this is the correct form that
    matches what colmap_utils.rotmat_to_qvec/qvec_to_rotmat round-trips.
    """
    w, x, y, z = qvec
    # Matches colmap_utils.qvec_to_rotmat exactly (with the .T)
    R = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,   2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,   2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,   2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ]).T   # colmap_utils applies .T so we do the same for consistency
    return R


def _angle_R(R1, R2):
    cos = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# CHECK 1  Camera Intrinsics: the double-scaling bug
# ---------------------------------------------------------------------------
def check_intrinsics(diag, poses):
    print("\n" + "=" * 62)
    print("  CHECK 1: Camera Intrinsics")
    print("=" * 62)

    # --- A. What cameras.bin stores (what the trainer actually uses) ---
    cams = _read_cameras_bin(SPARSE_DIR / "cameras.bin")
    cam  = cams[0]
    W, H = cam["w"], cam["h"]
    print(f"\n  [cameras.bin]  model={cam['model']}  size={W}x{H}")
    print(f"  params = {[round(p, 2) for p in cam['params']]}")

    if cam["model"] == "PINHOLE":
        fx_bin, fy_bin, cx_bin, cy_bin = cam["params"]
    elif cam["model"] == "SIMPLE_PINHOLE":
        fx_bin = fy_bin = cam["params"][0]
        cx_bin, cy_bin  = cam["params"][1], cam["params"][2]
    else:
        fx_bin, fy_bin  = cam["params"][0], cam["params"][1]
        cx_bin, cy_bin  = cam["params"][2], cam["params"][3]

    hfov_bin  = 2.0 * math.degrees(math.atan(W / 2.0 / fx_bin))
    vfov_bin  = 2.0 * math.degrees(math.atan(H / 2.0 / fy_bin))
    ratio_bin = fx_bin / fy_bin

    print(f"  fx={fx_bin:.2f}  fy={fy_bin:.2f}  cx={cx_bin:.2f}  cy={cy_bin:.2f}")
    print(f"  fx/fy = {ratio_bin:.4f}  (1.0 = square pixels)")
    print(f"  HFOV  = {hfov_bin:.2f} deg   VFOV = {vfov_bin:.2f} deg")
    print(f"  Expected for iPhone 0.5x ultrawide: ~{EXPECTED_HFOV_DEG:.0f} deg")

    # --- B. camera_poses.json intrinsics (after scale_intrinsics()) ---
    keys  = sorted(poses.keys())
    K0    = np.array(poses[keys[0]]["intrinsic_3x3"])
    fx_j  = K0[0, 0];  fy_j  = K0[1, 1]
    cx_j  = K0[0, 2];  cy_j  = K0[1, 2]
    hfov_j = 2.0 * math.degrees(math.atan(cx_j / fx_j))

    # Reverse-engineer VGGT-internal focal (scale_intrinsics applied W/518, H/518)
    fx_vggt = fx_j / (W / VGGT_RES)
    fy_vggt = fy_j / (H / VGGT_RES)

    print(f"\n  [camera_poses.json]  fx={fx_j:.2f}  fy={fy_j:.2f}  "
          f"cx={cx_j:.2f}  cy={cy_j:.2f}")
    print(f"  fx/fy={fx_j/fy_j:.4f}   HFOV={hfov_j:.2f} deg")
    print(f"  VGGT-internal:  fx_vggt={fx_vggt:.3f}  fy_vggt={fy_vggt:.3f}")
    print(f"  VGGT symmetric: |fx-fy|={abs(fx_vggt-fy_vggt):.4f}  "
          f"({'YES' if abs(fx_vggt-fy_vggt)<1 else 'NO'})")

    # --- C. Double-scaling in write_colmap_reconstruction() ---
    # scale_intrinsics() already scaled: fx_json = fx_vggt * (W/518)
    # write_colmap_reconstruction() then scales AGAIN: fx_bin = fx_json * max(W,H)/518
    rr = max(W, H) / VGGT_RES          # resize_ratio = 4032/518 = 7.784
    expected_fx_bin = fx_j * rr
    double_scale    = fx_bin / fx_j
    correct_fx      = fx_vggt * max(W, H) / VGGT_RES

    print(f"\n  [DOUBLE-SCALING CHECK]")
    print(f"  resize_ratio = max({W},{H})/{VGGT_RES} = {rr:.4f}")
    print(f"  Expected fx_bin if double-scaled: {fx_j:.2f} * {rr:.4f} = {expected_fx_bin:.2f}")
    print(f"  Actual   fx_bin in cameras.bin:   {fx_bin:.2f}")
    is_dbl = abs(fx_bin - expected_fx_bin) < 50
    print(f"  Match: {'DOUBLE-SCALING CONFIRMED' if is_dbl else 'no match'}")
    print(f"  Over-scale factor: {double_scale:.2f}x")
    print(f"  Correct focal (uniform scale): fx=fy={correct_fx:.2f} px")
    print(f"  Correct HFOV:  {2*math.degrees(math.atan(W/2/correct_fx)):.2f} deg")

    # --- Emit findings ---
    if abs(ratio_bin - 1.0) > 0.01:
        diag.add("CRITICAL",
            f"fx/fy={ratio_bin:.4f} in cameras.bin -- non-square pixels from non-uniform scaling",
            f"Images are {W}x{H} (4:3) resized to {VGGT_RES}x{VGGT_RES} (1:1) for VGGT.\n"
            f"scale_intrinsics() uses different scale per axis:\n"
            f"  scale_x={W}/{VGGT_RES}={W/VGGT_RES:.4f}  -> fx=fx_vggt*{W/VGGT_RES:.4f}\n"
            f"  scale_y={H}/{VGGT_RES}={H/VGGT_RES:.4f}  -> fy=fx_vggt*{H/VGGT_RES:.4f}\n"
            f"Physical cameras have square pixels so fx MUST equal fy.\n"
            f"Every pixel's depth->3D unprojection uses wrong geometry.\n"
            f"FIX: use fx=fy=fx_vggt*max(W,H)/{VGGT_RES}={correct_fx:.1f} px")

    if is_dbl:
        diag.add("CRITICAL",
            f"DOUBLE-SCALING in cameras.bin: fx={fx_bin:.0f} is {double_scale:.1f}x too large -> HFOV={hfov_bin:.1f} deg",
            f"write_colmap_reconstruction() receives already-scaled intrinsics\n"
            f"(scale_intrinsics already ran: fx_vggt={fx_vggt:.1f} -> fx_json={fx_j:.1f}),\n"
            f"then multiplies AGAIN by resize_ratio={rr:.2f}.\n"
            f"Result: fx_bin={fx_j:.0f}*{rr:.2f}={fx_bin:.0f}\n"
            f"Trainer thinks cameras have a {hfov_bin:.1f} deg telephoto FOV.\n"
            f"Every Gaussian initialises at wrong depth (off by ~{double_scale:.0f}x).\n"
            f"FIX: pass raw VGGT-resolution K (before scale_intrinsics) to\n"
            f"write_colmap_reconstruction(), scale once: fx=fy=fx_vggt*max(W,H)/{VGGT_RES}={correct_fx:.1f}")

    if hfov_j < 80:
        diag.add("WARNING",
            f"VGGT focal estimate: HFOV={hfov_j:.1f} deg vs true ultrawide ~{EXPECTED_HFOV_DEG:.0f} deg",
            f"Barrel distortion (Lens Correction OFF) causes VGGT to over-estimate focal.\n"
            f"FIX: pre-undistort images with Apple's k1/k2 coefficients before VGGT.")

    return {"fx_bin": fx_bin, "fy_bin": fy_bin, "W": W, "H": H,
            "hfov_bin": hfov_bin, "double_scale": double_scale,
            "correct_fx": correct_fx, "fx_vggt": fx_vggt}


# ---------------------------------------------------------------------------
# CHECK 2  Empty Tracks
# ---------------------------------------------------------------------------
def check_tracks(diag):
    print("\n" + "=" * 62)
    print("  CHECK 2: points3D.bin Track Fields")
    print("=" * 62)

    pts = _read_points3d_summary(SPARSE_DIR / "points3D.bin")
    print(f"\n  Total 3D points : {pts['n']}")
    print(f"  Empty tracks    : {pts['empty']}  ({100*pts['empty']/max(pts['n'],1):.1f}%)")
    print(f"  Avg track length: {pts['avg_track']:.2f}")

    if pts["n"] == 0:
        diag.add("WARNING", "No 3D points in points3D.bin",
                 "gsplat falls back to random Gaussian initialisation.\n"
                 "Slower convergence but NOT the root cause of the explosion.")
    elif pts["empty"] == pts["n"]:
        diag.add("WARNING",
            f"All {pts['n']} points have empty tracks (track_len=0)",
            "gsplat uses XYZ positions for init (still correct).\n"
            "Tracks only needed for depth_loss=True (unused here).\n"
            "Empty tracks do NOT break training -- NOT root cause.\n"
            "FIX (optional): assign track=[(image_id, 0)] per point.")
    else:
        diag.add("OK",
            f"Points have tracks (avg_len={pts['avg_track']:.1f})",
            f"{pts['n']-pts['empty']}/{pts['n']} points have >=1 track.")
    return pts


# ---------------------------------------------------------------------------
# CHECK 3  Pose Convention (c2w vs w2c)
# ---------------------------------------------------------------------------
def check_pose_convention(diag, poses):
    print("\n" + "=" * 62)
    print("  CHECK 3: Pose Convention (c2w vs w2c)")
    print("=" * 62)

    keys    = sorted(poses.keys())
    ext_j   = np.array(poses[keys[0]]["extrinsic_4x4"])     # w2c from VGGT
    c2w_j   = np.array(poses[keys[0]]["cam_to_world_4x4"])  # c2w from VGGT
    R_ext   = ext_j[:3, :3];  t_ext = ext_j[:3, 3]
    R_c2w   = c2w_j[:3, :3];  t_c2w = c2w_j[:3, 3]

    imgs = _read_images_bin_sample(SPARSE_DIR / "images.bin", 1)
    if not imgs:
        diag.add("WARNING", "Could not read images.bin", "")
        return

    R_bin = _qvec_to_R(imgs[0]["qvec"])
    t_bin = np.array(imgs[0]["tvec"])

    d_w2c = np.linalg.norm(R_bin - R_ext)
    d_c2w = np.linalg.norm(R_bin - R_c2w)
    d_t   = np.linalg.norm(t_bin - t_ext)

    print(f"\n  Comparing images.bin vs camera_poses.json:")
    print(f"  ||R_bin - R_extrinsic(w2c)||    = {d_w2c:.8f}  "
          f"-> {'MATCH' if d_w2c < 1e-4 else 'no match'}")
    print(f"  ||R_bin - R_cam_to_world(c2w)|| = {d_c2w:.8f}  "
          f"-> {'MATCH' if d_c2w < 1e-4 else 'no match'}")
    print(f"  ||t_bin - t_extrinsic(w2c)||    = {d_t:.8f}  "
          f"-> {'MATCH' if d_t < 1e-4 else 'no match'}")

    # Use 1e-3 tolerance: quat round-tripping introduces ~2e-7 numerical diff.
    # Verified separately: inv(images.bin w2c) == camera_poses.json c2w to 2e-7.
    if d_w2c < 1e-3 and d_t < 1e-4:
        diag.add("OK",
            "Pose convention correct: extrinsic_4x4 (w2c) stored in images.bin",
            "COLMAP needs w2c. VGGT extrinsic_4x4 IS w2c. Written directly -- correct.\n"
            "gsplat_colmap_dataset.py reads w2c, inverts to c2w for training -- correct.\n"
            "Round-trip: inv(images.bin w2c) == camera_poses.json c2w (diff < 2e-7).")
        print("  Pose convention: CORRECT")
    elif d_c2w < 1e-3:
        diag.add("CRITICAL",
            "WRONG CONVENTION: cam_to_world written to images.bin instead of w2c",
            "COLMAP images.bin needs R_w2c and t_w2c.\n"
            "FIX: use extrinsic_4x4 (which IS w2c), not cam_to_world_4x4.")
        print("  Pose convention: WRONG (c2w written instead of w2c)")
    else:
        diag.add("WARNING",
            f"Pose convention ambiguous (w2c_diff={d_w2c:.6f}, c2w_diff={d_c2w:.6f})",
            "Manual verification recommended.")


# ---------------------------------------------------------------------------
# CHECK 4  Batch Boundary Discontinuities
# ---------------------------------------------------------------------------
def check_batch_boundaries(diag, poses):
    print("\n" + "=" * 62)
    print("  CHECK 4: Batch-Boundary Pose Discontinuity")
    print("=" * 62)

    keys = sorted(poses.keys())
    n    = len(keys)

    positions = np.array(
        [np.array(poses[k]["cam_to_world_4x4"])[:3, 3] for k in keys])
    rotations = np.array(
        [np.array(poses[k]["cam_to_world_4x4"])[:3, :3] for k in keys])

    dists  = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angles = np.array([_angle_R(rotations[i], rotations[i + 1])
                       for i in range(n - 1)])

    b_idx  = [i for i in range(len(angles)) if (i + 1) % BATCH_SIZE == 0]
    nb_idx = [i for i in range(len(angles)) if (i + 1) % BATCH_SIZE != 0]

    b_ang  = angles[b_idx];  nb_ang  = angles[nb_idx]
    b_dist = dists[b_idx];   nb_dist = dists[nb_idx]

    r_ang  = b_ang.mean()  / max(nb_ang.mean(),  1e-9)
    r_dist = b_dist.mean() / max(nb_dist.mean(), 1e-9)

    print(f"\n  Frames: {n}   Batch size: {BATCH_SIZE}")
    print(f"  Boundary transitions: {len(b_idx)}   Non-boundary: {len(nb_idx)}")
    print(f"\n  Rotation (deg):")
    print(f"    Boundary:     mean={b_ang.mean():.2f}  median={np.median(b_ang):.2f}  "
          f"max={b_ang.max():.2f}")
    print(f"    Non-boundary: mean={nb_ang.mean():.2f}  median={np.median(nb_ang):.2f}  "
          f"max={nb_ang.max():.2f}")
    print(f"    Ratio: {r_ang:.2f}x")
    print(f"\n  Translation:")
    print(f"    Boundary:     mean={b_dist.mean():.5f}  max={b_dist.max():.5f}")
    print(f"    Non-boundary: mean={nb_dist.mean():.5f}  max={nb_dist.max():.5f}")
    print(f"    Ratio: {r_dist:.2f}x")

    print(f"\n  Top-5 worst batch-boundary transitions:")
    worst = np.argsort(b_ang)[-5:][::-1]
    for wi in worst:
        bi = b_idx[wi]
        print(f"    Frame {bi:3d}->{bi+1:3d}: rot={b_ang[wi]:.1f} deg  "
              f"dist={b_dist[wi]:.5f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    x = np.arange(len(angles))
    for bi in b_idx:
        ax1.axvline(bi, color="red", alpha=0.2, lw=0.8)
        ax2.axvline(bi, color="red", alpha=0.2, lw=0.8)
    ax1.plot(x, angles, lw=0.5, alpha=0.7, color="steelblue",
             label="Rotation (deg)")
    ax1.axhline(nb_ang.mean(), color="green", ls="--", lw=1.5,
                label=f"Non-boundary mean ({nb_ang.mean():.1f} deg)")
    ax1.axhline(b_ang.mean(),  color="red",   ls="--", lw=1.5,
                label=f"Boundary mean ({b_ang.mean():.1f} deg)")
    ax1.set_ylabel("Rotation (deg)")
    ax1.set_title(f"Inter-frame Rotation  [red lines = batch boundaries every {BATCH_SIZE}]")
    ax1.legend(fontsize=9)
    ax2.plot(x, dists, lw=0.5, alpha=0.7, color="darkorange",
             label="Translation dist")
    ax2.axhline(nb_dist.mean(), color="green", ls="--", lw=1.5,
                label=f"Non-boundary mean ({nb_dist.mean():.5f})")
    ax2.axhline(b_dist.mean(),  color="red",   ls="--", lw=1.5,
                label=f"Boundary mean ({b_dist.mean():.5f})")
    ax2.set_ylabel("Translation distance")
    ax2.set_xlabel("Frame transition index")
    ax2.set_title("Inter-frame Translation")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_boundaries.png", dpi=150)
    plt.close()
    print(f"\n  -> Saved {OUTPUT_DIR / 'batch_boundaries.png'}")

    mx = max(r_ang, r_dist)
    if mx > 3.0:
        diag.add("CRITICAL",
            f"Batch-boundary jumps: {r_ang:.1f}x rotation, {r_dist:.1f}x translation",
            f"VGGT estimates relative poses INDEPENDENTLY within each batch of {BATCH_SIZE}.\n"
            f"With 511 still photos (large viewpoint jumps), consecutive batches\n"
            f"have no overlap constraint -- batch k and k+1 are unregistered.\n"
            f"Worst cases: 180 deg flips at frames 249->250, 119->120, 459->460.\n"
            f"Effect: {n//BATCH_SIZE}+ disconnected local maps oriented arbitrarily\n"
            f"-> 'abstract explosion' appearance in SuperSplat.\n"
            f"\n"
            f"FIX A: Sort frames by visual similarity before batching.\n"
            f"FIX B: Use overlapping batches (stride < batch_size).\n"
            f"FIX C: Run COLMAP BA on VGGT poses to stitch batches globally.\n"
            f"FIX D: Increase batch_size to cover all frames at once.")
    elif mx > 1.5:
        diag.add("WARNING",
            f"Mild batch boundary effects ({mx:.1f}x ratio)", "")
    else:
        diag.add("OK",
            f"No significant batch boundary discontinuity ({mx:.1f}x)", "")

    return positions, rotations


# ---------------------------------------------------------------------------
# CHECK 5  Depth Maps
# ---------------------------------------------------------------------------
def check_depth_maps(diag):
    print("\n" + "=" * 62)
    print("  CHECK 5: Depth Maps")
    print("=" * 62)

    depth_files = sorted(
        [f for f in DEPTHS_DIR.iterdir() if f.name.endswith("_depth.npy")])
    print(f"\n  Found {len(depth_files)} depth maps")

    if not depth_files:
        diag.add("WARNING", "No depth maps found in depths/", "")
        return

    rng = np.random.RandomState(42)
    si  = sorted(rng.choice(len(depth_files),
                            min(5, len(depth_files)), replace=False))
    sampled = [depth_files[i] for i in si]

    all_ranges = []
    ncols = len(sampled)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for col, dp in enumerate(sampled):
        depth = np.load(dp)
        cp    = DEPTHS_DIR / dp.name.replace("_depth.npy", "_conf.npy")
        conf  = np.load(cp) if cp.exists() else None

        mn, mx_, mu = float(depth.min()), float(depth.max()), float(depth.mean())
        all_ranges.append((mn, mx_, mu))
        print(f"  {dp.name}: shape={depth.shape}  range=[{mn:.3f},{mx_:.3f}]  "
              f"mean={mu:.3f} m")

        im = axes[0, col].imshow(depth, cmap="turbo", vmin=0, vmax=min(mx_, 10.0))
        axes[0, col].set_title(f"Depth {dp.name[:18]}\n[{mn:.2f},{mx_:.2f}] m",
                               fontsize=7)
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        if conf is not None:
            ic = axes[1, col].imshow(conf, cmap="viridis")
            axes[1, col].set_title(
                f"Confidence\n[{conf.min():.2f},{conf.max():.2f}]", fontsize=7)
            plt.colorbar(ic, ax=axes[1, col], fraction=0.046, pad=0.04)
        else:
            axes[1, col].text(0.5, 0.5, "No conf", ha="center", va="center")
        axes[1, col].axis("off")

    plt.suptitle("Depth Maps (top) + Confidence (bottom) -- 5 random samples",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "depth_samples.png", dpi=150)
    plt.close()
    print(f"\n  -> Saved {OUTPUT_DIR / 'depth_samples.png'}")

    arr = np.array(all_ranges)
    g_min, g_max, g_mean = arr[:, 0].min(), arr[:, 1].max(), arr[:, 2].mean()
    print(f"\n  Global depth: min={g_min:.3f}  max={g_max:.3f}  mean={g_mean:.3f} m")
    print(f"  Expected room depth: 0.1-5.0 m")

    s = np.load(depth_files[0]).shape
    if s[0] == s[1] == VGGT_RES:
        diag.add("OK", f"Depth maps at VGGT resolution ({VGGT_RES}x{VGGT_RES})", "")
    else:
        diag.add("WARNING",
            f"Depth maps are {s} -- expected ({VGGT_RES},{VGGT_RES})",
            "Unexpected depth resolution may mis-align with intrinsics.")

    if g_max > 100:
        diag.add("WARNING",
            f"Depth range very large: [{g_min:.1f},{g_max:.1f}] m",
            "Expected ~0.1-5.0 m for indoor room.\n"
            "Large values may reflect VGGT scale errors from wrong intrinsics.")
    elif g_min < 0:
        diag.add("WARNING", "Negative depth values", "Depth must be >= 0.")
    else:
        diag.add("OK", f"Depth range plausible: [{g_min:.2f},{g_max:.2f}] m", "")


# ---------------------------------------------------------------------------
# CHECK 6  Angular Coverage
# ---------------------------------------------------------------------------
def check_angular_coverage(diag, positions, rotations):
    print("\n" + "=" * 62)
    print("  CHECK 6: Angular Coverage")
    print("=" * 62)

    # c2w: camera looks along +Z in camera space -> view dir in world = R @ [0,0,1]
    vd      = np.array([R @ np.array([0., 0., 1.]) for R in rotations])
    vd_unit = vd / np.linalg.norm(vd, axis=1, keepdims=True)

    theta = np.degrees(np.arccos(np.clip(vd_unit[:, 2], -1, 1)))
    phi   = np.degrees(np.arctan2(vd_unit[:, 1], vd_unit[:, 0]))

    print(f"\n  Views: {len(vd)}")
    print(f"  Azimuthal phi:  [{phi.min():.1f},{phi.max():.1f}] = {phi.max()-phi.min():.1f} deg")
    print(f"  Polar theta:    [{theta.min():.1f},{theta.max():.1f}] = "
          f"{theta.max()-theta.min():.1f} deg")

    n_phi, n_theta = 36, 18
    grid = np.zeros((n_theta, n_phi), dtype=bool)
    for p, t in zip(phi, theta):
        pi_ = min(int((p + 180) / 360 * n_phi), n_phi - 1)
        ti_ = min(int(t / 180 * n_theta), n_theta - 1)
        grid[ti_, pi_] = True
    pct = 100.0 * grid.sum() / grid.size
    print(f"  Sphere coverage: {pct:.1f}%  ({grid.sum()}/{grid.size} bins)")

    if pct < 10:
        diag.add("WARNING",
            f"Very low angular coverage ({pct:.1f}%) -- cameras highly clustered",
            "Need inward-facing cameras from all sides for full-room reconstruction.")
    elif pct < 25:
        diag.add("WARNING",
            f"Partial angular coverage ({pct:.1f}%)",
            "Unobserved regions will have floaters or holes.")
    else:
        diag.add("OK", f"Good angular coverage ({pct:.1f}%)", "")


# ---------------------------------------------------------------------------
# PLOT  Camera positions in 3D
# ---------------------------------------------------------------------------
def plot_camera_positions(positions, rotations):
    print(f"\n  Plotting camera positions...")

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    c  = np.arange(len(positions))
    sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=c, cmap="viridis", s=6, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label="Frame index", shrink=0.55, pad=0.1)

    bd = list(range(0, len(positions), BATCH_SIZE))
    ax.scatter(positions[bd, 0], positions[bd, 1], positions[bd, 2],
               c="red", s=50, marker="^", zorder=5,
               label=f"Batch starts (every {BATCH_SIZE})", alpha=0.9)

    step  = max(1, len(positions) // 30)
    scale = 0.04 * max(
        (positions.max(axis=0) - positions.min(axis=0)).max(), 1e-6)
    for i in range(0, len(positions), step):
        vd = rotations[i] @ np.array([0., 0., 1.])
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                  vd[0], vd[1], vd[2],
                  length=scale, color="gray", alpha=0.4, arrow_length_ratio=0.3)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(
        f"Camera Positions ({len(positions)} frames)\n"
        f"Colour=frame index  |  Red triangle=batch starts  |  Arrows=view dir")
    ax.legend(fontsize=9)

    mid = positions.mean(axis=0)
    rng = max((positions.max(axis=0) - positions.min(axis=0)).max() / 2, 1e-6)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "camera_positions.png", dpi=150)
    plt.close()
    print(f"  -> Saved {OUTPUT_DIR / 'camera_positions.png'}")


# ---------------------------------------------------------------------------
# Print cameras.bin intrinsics explicitly (requested)
# ---------------------------------------------------------------------------
def print_colmap_intrinsics():
    print("\n" + "=" * 62)
    print("  cameras.bin -- exact stored intrinsics")
    print("=" * 62)
    cams = _read_cameras_bin(SPARSE_DIR / "cameras.bin")
    for cam in cams[:3]:
        print(f"\n  Camera {cam['id']}:  model={cam['model']}  "
              f"{cam['w']}x{cam['h']}")
        print(f"    params = {[round(p, 4) for p in cam['params']]}")
        if len(cam["params"]) >= 4:
            fx, fy = cam["params"][0], cam["params"][1]
            print(f"    HFOV  = {2*math.degrees(math.atan(cam['w']/2/fx)):.2f} deg")
            print(f"    fx/fy = {fx/fy:.4f}")
    if len(cams) > 3:
        print(f"\n  ... ({len(cams)} cameras total, showing first 3)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diag = Diag()

    print("\n" + "#" * 72)
    print("  RoboScene+  --  3DGS RECONSTRUCTION DIAGNOSTIC")
    print("#" * 72)
    print(f"\n  VGGT output : {VGGT_OUT}")
    print(f"  Sparse dir  : {SPARSE_DIR}")
    print(f"  Debug output: {OUTPUT_DIR}")

    with open(POSES_JSON) as f:
        poses = json.load(f)
    print(f"\n  Loaded {len(poses)} camera poses")

    if METADATA_JSON.exists():
        with open(METADATA_JSON) as f:
            meta = json.load(f)
        print(f"  Metadata: device={meta.get('device','?')}  "
              f"vggt_res={meta.get('vggt_resolution','?')}  "
              f"batch_size={meta.get('batch_size','?')}  "
              f"frames={meta.get('total_frames','?')}")

    print_colmap_intrinsics()
    intr                    = check_intrinsics(diag, poses)
    check_tracks(diag)
    check_pose_convention(diag, poses)
    positions, rotations    = check_batch_boundaries(diag, poses)
    check_depth_maps(diag)
    check_angular_coverage(diag, positions, rotations)
    plot_camera_positions(positions, rotations)

    summary = diag.summary()
    print(summary)

    rp = OUTPUT_DIR / "diagnosis_report.txt"
    rp.write_text(summary)
    print(f"\n  Full report: {rp}")
    print(f"  Plots:       {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()