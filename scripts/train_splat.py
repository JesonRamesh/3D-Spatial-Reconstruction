#!/usr/bin/env python3
"""
RoboScene+ Session 3: Gaussian Splatting Training
===================================================

Trains a 3D Gaussian Splat from VGGT's COLMAP-format output using
gsplat's simple_trainer.

Workflow:
  1. Log nvidia-smi GPU info at start
  2. Launch gsplat simple_trainer as a subprocess
  3. After training: locate output .ply, copy to outputs/splat/scene.ply
  4. Parse training log for final PSNR

Usage:
    python scripts/train_splat.py
    python scripts/train_splat.py --colmap_dir data/vggt_out/ --output_dir outputs/splat/
    python scripts/train_splat.py --iterations 30000
    python scripts/train_splat.py --colmap_dir /scratch0/jrameshs/roboscene-plus/data/vggt_out/
"""

import argparse
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── Logging ────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure console + file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("roboscene.splat")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "splat_train.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-7s │ %(message)s"))
    logger.addHandler(fh)

    return logger


# ── GPU diagnostics ────────────────────────────────────────────────────

def log_gpu_info(logger: logging.Logger) -> None:
    """Log nvidia-smi output at start for diagnostics."""
    logger.info("Querying GPU info (nvidia-smi)...")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                logger.info(f"  GPU │ {line}")
        else:
            logger.warning(f"  nvidia-smi returned code {result.returncode}")
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        logger.warning("  nvidia-smi not found — no NVIDIA GPU detected")
    except subprocess.TimeoutExpired:
        logger.warning("  nvidia-smi timed out")


# ── Validate COLMAP directory ──────────────────────────────────────────

def validate_colmap_dir(colmap_dir: Path, logger: logging.Logger) -> bool:
    """Check that the COLMAP sparse directory has required files."""
    sparse_dir = colmap_dir / "sparse"
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]

    if not sparse_dir.exists():
        logger.error(f"Sparse directory not found: {sparse_dir}")
        logger.error("Run VGGT first (Session 2) to generate COLMAP output.")
        return False

    missing = [f for f in required_files if not (sparse_dir / f).exists()]
    if missing:
        logger.error(f"Missing COLMAP files in {sparse_dir}: {missing}")
        return False

    # Check for images directory (gsplat needs the frames)
    images_dir = colmap_dir / "images"
    if not images_dir.exists():
        # gsplat simple_trainer expects images/ alongside sparse/
        # VGGT outputs frames to data/frames/, so we may need to symlink
        logger.warning(f"No images/ directory at {images_dir}")
        logger.info("Will attempt to create symlink from frames directory...")
        return True

    num_images = len(list(images_dir.glob("*")))
    logger.info(f"Found {num_images} images in {images_dir}")
    return True


def ensure_images_dir(colmap_dir: Path, frames_dir: Path, logger: logging.Logger) -> bool:
    """
    gsplat simple_trainer expects an images/ subdirectory inside data_dir.
    If it doesn't exist, create a symlink or copy from the frames directory.
    """
    images_dir = colmap_dir / "images"

    if images_dir.exists():
        num = len(list(images_dir.glob("*")))
        if num > 0:
            logger.info(f"images/ directory already exists with {num} files")
            return True

    # Try to find frames
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        logger.error("Cannot create images/ directory for gsplat.")
        return False

    frame_files = sorted(glob.glob(str(frames_dir / "frame_*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(str(frames_dir / "frame_*.png")))
    if not frame_files:
        logger.error(f"No frame_*.jpg or frame_*.png files in {frames_dir}")
        return False

    logger.info(f"Creating images/ symlink: {images_dir} → {frames_dir.resolve()}")
    try:
        # Prefer symlink (saves disk space)
        if images_dir.exists() or images_dir.is_symlink():
            images_dir.unlink() if images_dir.is_symlink() else shutil.rmtree(images_dir)
        images_dir.symlink_to(frames_dir.resolve())
        logger.info(f"  Symlinked {len(frame_files)} frames")
    except OSError:
        # Fall back to copy if symlink fails (e.g., cross-device)
        logger.info("  Symlink failed, copying frames instead...")
        images_dir.mkdir(parents=True, exist_ok=True)
        for fp in frame_files:
            shutil.copy2(fp, images_dir / Path(fp).name)
        logger.info(f"  Copied {len(frame_files)} frames to {images_dir}")

    return True


# ── Run gsplat training ────────────────────────────────────────────────

def run_gsplat_training(
    colmap_dir: Path,
    output_dir: Path,
    iterations: int,
    logger: logging.Logger,
) -> tuple:
    """
    Launch gsplat simple_trainer as a subprocess.

    Returns:
        (success: bool, log_output: str)
    """
    cmd = [
        sys.executable, "-m", "gsplat.scripts.simple_trainer",
        "default",
        "--data_dir", str(colmap_dir),
        "--result_dir", str(output_dir),
        "--max_steps", str(iterations),
        "--data_factor", "1",
    ]

    logger.info("=" * 62)
    logger.info("  Launching gsplat simple_trainer")
    logger.info("=" * 62)
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info(f"  Data dir:    {colmap_dir}")
    logger.info(f"  Result dir:  {output_dir}")
    logger.info(f"  Iterations:  {iterations}")
    logger.info(f"  Data factor: 1")
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    full_output = []

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Stream output in real-time
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                logger.info(f"  gsplat │ {line}")
                full_output.append(line)

        process.wait()
        elapsed = time.time() - t0

        if process.returncode == 0:
            logger.info("")
            logger.info(f"  gsplat training completed in {elapsed:.1f}s "
                        f"({elapsed / 60:.1f} min)")
            return True, "\n".join(full_output)
        else:
            logger.error(f"  gsplat exited with code {process.returncode}")
            return False, "\n".join(full_output)

    except FileNotFoundError:
        logger.error("Failed to launch gsplat. Is gsplat installed?")
        logger.error("  Install: pip install gsplat==1.3.0")
        return False, ""
    except Exception as e:
        logger.error(f"Unexpected error running gsplat: {e}")
        return False, ""


# ── Extract PSNR from training log ─────────────────────────────────────

def extract_final_psnr(log_output: str, logger: logging.Logger) -> float | None:
    """
    Parse gsplat training output for the final PSNR value.

    gsplat simple_trainer typically prints lines like:
        Step XXXXX: PSNR = XX.XX
    or similar patterns. We search for the last PSNR value.
    """
    psnr_values = []

    # Try multiple patterns that gsplat might use
    patterns = [
        r"[Pp][Ss][Nn][Rr]\s*[=:]\s*([\d.]+)",       # PSNR = 25.43 or psnr: 25.43
        r"[Pp][Ss][Nn][Rr]\s+([\d.]+)",                # PSNR 25.43
        r"\"psnr\"\s*:\s*([\d.]+)",                     # "psnr": 25.43 (JSON)
    ]

    for pattern in patterns:
        matches = re.findall(pattern, log_output)
        for m in matches:
            try:
                val = float(m)
                if 5.0 < val < 60.0:  # sanity check: valid PSNR range
                    psnr_values.append(val)
            except ValueError:
                continue

    if psnr_values:
        final_psnr = psnr_values[-1]
        logger.info(f"  Final PSNR: {final_psnr:.2f} dB")

        # Also report range if multiple values found
        if len(psnr_values) > 1:
            logger.info(f"  PSNR progression: {psnr_values[0]:.2f} → {final_psnr:.2f} dB "
                        f"({len(psnr_values)} measurements)")
        return final_psnr
    else:
        logger.warning("  Could not parse PSNR from training output")
        return None


# ── Find and copy output .ply ──────────────────────────────────────────

def find_and_copy_ply(
    output_dir: Path,
    logger: logging.Logger,
) -> Path | None:
    """
    Find the trained .ply file in gsplat's output directory and copy
    to outputs/splat/scene.ply for downstream use.

    gsplat simple_trainer saves .ply files in various subdirectory
    structures depending on version. We search recursively.
    """
    # Search for .ply files in the output directory
    ply_files = sorted(output_dir.rglob("*.ply"), key=lambda p: p.stat().st_mtime)

    if not ply_files:
        logger.error(f"No .ply files found in {output_dir}")
        logger.error("Training may have failed or output format changed.")
        return None

    # Use the most recently modified .ply
    source_ply = ply_files[-1]
    logger.info(f"  Found .ply: {source_ply}")
    logger.info(f"  Size: {source_ply.stat().st_size / (1024 * 1024):.1f} MB")

    if len(ply_files) > 1:
        logger.info(f"  ({len(ply_files)} .ply files found, using most recent)")
        for p in ply_files:
            logger.debug(f"    {p} ({p.stat().st_size / (1024 * 1024):.1f} MB)")

    # Copy to canonical location
    dest_ply = output_dir / "scene.ply"
    if source_ply.resolve() != dest_ply.resolve():
        shutil.copy2(source_ply, dest_ply)
        logger.info(f"  Copied to: {dest_ply}")
    else:
        logger.info(f"  Already at canonical path: {dest_ply}")

    return dest_ply


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RoboScene+ Session 3: Gaussian Splatting Training"
    )
    parser.add_argument("--colmap_dir", type=str, default="data/vggt_out",
                        help="COLMAP output directory from VGGT (default: data/vggt_out/)")
    parser.add_argument("--output_dir", type=str, default="outputs/splat",
                        help="Output directory for trained splat (default: outputs/splat/)")
    parser.add_argument("--iterations", type=int, default=15000,
                        help="Number of training iterations (default: 15000)")
    parser.add_argument("--frames_dir", type=str, default="data/frames",
                        help="Frames directory (used to create images/ symlink if needed)")
    args = parser.parse_args()

    # Resolve project root
    project_root = Path(__file__).resolve().parent.parent

    # Resolve paths (support both relative and absolute)
    colmap_dir = Path(args.colmap_dir)
    if not colmap_dir.is_absolute():
        colmap_dir = project_root / colmap_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_absolute():
        frames_dir = project_root / frames_dir

    # Setup logging
    logger = setup_logging(project_root / "logs")
    logger.info("=" * 62)
    logger.info("  RoboScene+ — Gaussian Splatting Training (gsplat)")
    logger.info("=" * 62)
    logger.info(f"  COLMAP dir:   {colmap_dir}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Iterations:   {args.iterations}")
    logger.info(f"  Frames dir:   {frames_dir}")
    logger.info("")

    # ── Step 1: Log GPU info ───────────────────────────────────────────
    log_gpu_info(logger)
    logger.info("")

    # ── Step 2: Validate inputs ────────────────────────────────────────
    if not validate_colmap_dir(colmap_dir, logger):
        logger.error("COLMAP directory validation failed. Aborting.")
        sys.exit(1)

    # Ensure images/ directory exists for gsplat
    if not ensure_images_dir(colmap_dir, frames_dir, logger):
        logger.error("Could not set up images/ directory. Aborting.")
        sys.exit(1)
    logger.info("")

    # ── Step 3: Run gsplat training ────────────────────────────────────
    success, log_output = run_gsplat_training(
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        iterations=args.iterations,
        logger=logger,
    )

    if not success:
        logger.error("")
        logger.error("=" * 62)
        logger.error("  GAUSSIAN SPLATTING TRAINING FAILED")
        logger.error("=" * 62)
        logger.error("  Check the log output above for errors.")
        logger.error("  Common issues:")
        logger.error("    - gsplat not installed: pip install gsplat==1.3.0")
        logger.error("    - COLMAP files malformed: re-run VGGT (Session 2)")
        logger.error("    - CUDA OOM: reduce --iterations or check GPU memory")
        sys.exit(1)

    # ── Step 4: Extract PSNR ───────────────────────────────────────────
    logger.info("")
    logger.info("Extracting PSNR from training log...")
    final_psnr = extract_final_psnr(log_output, logger)

    # ── Step 5: Find and copy output .ply ──────────────────────────────
    logger.info("")
    logger.info("Locating output .ply file...")
    scene_ply = find_and_copy_ply(output_dir, logger)

    # ── Step 6: Save training metadata ─────────────────────────────────
    import json
    meta = {
        "colmap_dir": str(colmap_dir),
        "output_dir": str(output_dir),
        "iterations": args.iterations,
        "final_psnr": final_psnr,
        "scene_ply": str(scene_ply) if scene_ply else None,
        "scene_ply_size_mb": round(scene_ply.stat().st_size / (1024 * 1024), 2)
            if scene_ply else None,
        "training_success": success,
    }
    meta_path = output_dir / "splat_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata saved: {meta_path}")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 62)
    logger.info("  GAUSSIAN SPLATTING TRAINING COMPLETE")
    logger.info("=" * 62)
    logger.info(f"  Iterations:    {args.iterations}")
    if final_psnr is not None:
        logger.info(f"  Final PSNR:    {final_psnr:.2f} dB")
    else:
        logger.info(f"  Final PSNR:    (could not parse)")
    if scene_ply:
        logger.info(f"  Scene PLY:     {scene_ply}")
        logger.info(f"  PLY size:      {scene_ply.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        logger.info(f"  Scene PLY:     NOT FOUND — check output directory")
    logger.info(f"  Metadata:      {meta_path}")
    logger.info(f"")
    logger.info(f"  ✅ Ready for Session 4 (Grounded SAM2 Semantics)")
    logger.info(f"")
    logger.info(f"  Download to Mac:")
    logger.info(f"    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\")
    logger.info(f"      jrameshs@bluestreak.cs.ucl.ac.uk:{output_dir}/ \\")
    logger.info(f"      ~/3D-Spatial-Reconstruction/outputs/splat/")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()