"""Train a 3D Gaussian Splat from COLMAP data using nerfstudio splatfacto."""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────

# Pinned to v1.3.0 for reproducibility (simpler deps than main branch)
GSPLAT_TAG = "v1.3.0"
GSPLAT_RAW_BASE = f"https://raw.githubusercontent.com/nerfstudio-project/gsplat/{GSPLAT_TAG}/examples"

# Files needed from gsplat examples to run simple_trainer.py
# NOTE: datasets/colmap.py is NOT downloaded -- we use our own patched version
# that replaces pycolmap.SceneManager with our custom binary readers.
GSPLAT_FILES_DOWNLOAD = [
    "simple_trainer.py",
    "utils.py",
    "datasets/normalize.py",
    "datasets/traj.py",
]

# All files that should be present (including our patched colmap.py)
GSPLAT_FILES_ALL = [
    "simple_trainer.py",
    "utils.py",
    "datasets/colmap.py",
    "datasets/normalize.py",
    "datasets/traj.py",
]


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
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "splat_train.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
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
                logger.info(f"  GPU | {line}")
        else:
            logger.warning(f"  nvidia-smi returned code {result.returncode}")
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        logger.warning("  nvidia-smi not found -- no NVIDIA GPU detected")
    except subprocess.TimeoutExpired:
        logger.warning("  nvidia-smi timed out")


# ── Download gsplat examples ───────────────────────────────────────────

def ensure_gsplat_trainer(project_root: Path, logger: logging.Logger) -> Path:
    """
    Download gsplat's official simple_trainer.py and supporting files
    from the pinned tag. Cached in scripts/gsplat_examples/.

    Returns the path to simple_trainer.py.
    """
    examples_dir = project_root / "scripts" / "gsplat_examples"
    trainer_path = examples_dir / "simple_trainer.py"

    # Check if already set up
    all_present = all((examples_dir / f).exists() for f in GSPLAT_FILES_ALL)
    if all_present:
        logger.info(f"gsplat examples already cached at {examples_dir}")
        return trainer_path

    logger.info(f"Downloading gsplat examples ({GSPLAT_TAG}) to {examples_dir}...")

    for rel_path in GSPLAT_FILES_DOWNLOAD:
        url = f"{GSPLAT_RAW_BASE}/{rel_path}"
        dest = examples_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Downloading {rel_path}...")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            logger.error(f"  Failed to download {url}: {e}")
            raise

    # Copy our patched datasets/colmap.py (replaces pycolmap with our readers)
    patched_colmap = project_root / "scripts" / "gsplat_colmap_dataset.py"
    dest_colmap = examples_dir / "datasets" / "colmap.py"
    dest_colmap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(patched_colmap, dest_colmap)
    logger.info(f"  Installed patched datasets/colmap.py (no pycolmap dependency)")

    # Create __init__.py for datasets package
    datasets_init = examples_dir / "datasets" / "__init__.py"
    if not datasets_init.exists():
        datasets_init.write_text("")

    logger.info(f"  All files set up successfully ({len(GSPLAT_FILES_ALL)} total)")
    return trainer_path


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

    logger.info(f"Creating images/ symlink: {images_dir} -> {frames_dir.resolve()}")
    try:
        # Prefer symlink (saves disk space)
        if images_dir.exists() or images_dir.is_symlink():
            if images_dir.is_symlink():
                images_dir.unlink()
            else:
                shutil.rmtree(images_dir)
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
    trainer_script: Path,
    colmap_dir: Path,
    output_dir: Path,
    iterations: int,
    logger: logging.Logger,
    opacity_reg: float = 0.0,
    scale_reg: float = 0.0,
    prune_opa: float = 0.005,
) -> tuple:
    """
    Launch gsplat simple_trainer.py as a subprocess using sys.executable.

    The script is run with cwd set to its parent directory so that
    relative imports (datasets.colmap, utils) resolve correctly.

    Returns:
        (success: bool, log_output: str)
    """
    cmd = [
        sys.executable, str(trainer_script),
        "default",
        "--data_dir", str(colmap_dir),
        "--result_dir", str(output_dir),
        "--max_steps", str(iterations),
        "--data_factor", "1",
        "--prune_opa", str(prune_opa),
    ]
    # Regularizers: only pass when non-zero (older gsplat builds may not have them)
    if opacity_reg > 0:
        cmd += ["--opacity_reg", str(opacity_reg)]
    if scale_reg > 0:
        cmd += ["--scale_reg", str(scale_reg)]

    logger.info("=" * 62)
    logger.info("  Launching gsplat simple_trainer")
    logger.info("=" * 62)
    logger.info(f"  Python:      {sys.executable}")
    logger.info(f"  Script:      {trainer_script}")
    logger.info(f"  CWD:         {trainer_script.parent}")
    logger.info(f"  Command:     {' '.join(cmd)}")
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
            cwd=str(trainer_script.parent),  # so relative imports work
        )

        # Stream output in real-time
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                logger.info(f"  gsplat | {line}")
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
        logger.error(f"Failed to launch Python: {sys.executable}")
        logger.error("  Check that your venv is properly activated.")
        return False, ""
    except Exception as e:
        logger.error(f"Unexpected error running gsplat: {e}")
        return False, ""


# ── Extract PSNR from training log ─────────────────────────────────────

def extract_final_psnr(log_output: str, logger: logging.Logger):
    """
    Parse gsplat training output for the final PSNR value.

    gsplat simple_trainer typically prints lines like:
        Step XXXXX: PSNR = XX.XX
    or similar patterns. We search for the last PSNR value.

    Returns:
        float or None
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
            logger.info(f"  PSNR progression: {psnr_values[0]:.2f} -> {final_psnr:.2f} dB "
                        f"({len(psnr_values)} measurements)")
        return final_psnr
    else:
        logger.warning("  Could not parse PSNR from training output")
        return None


# ── Find and copy output .ply ──────────────────────────────────────────

def find_and_copy_ply(output_dir: Path, logger: logging.Logger):
    """
    Find the trained .ply file in gsplat's output directory and copy
    to outputs/splat/scene.ply for downstream use.

    gsplat simple_trainer saves .ply files in various subdirectory
    structures depending on version. We search recursively.

    Returns:
        Path or None
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
    parser.add_argument("--opacity_reg", type=float, default=0.0,
                        help="Opacity regularization weight (default: 0.0). "
                             "Use 0.01 to suppress floater Gaussians during training.")
    parser.add_argument("--scale_reg", type=float, default=0.0,
                        help="Scale regularization weight (default: 0.0). "
                             "Use 0.001 to penalize oversized/elongated Gaussians.")
    parser.add_argument("--prune_opa", type=float, default=0.005,
                        help="Opacity threshold for pruning Gaussians (default: 0.005). "
                             "Increase to 0.01 for more aggressive pruning.")
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
    logger.info("  RoboScene+ -- Gaussian Splatting Training (gsplat)")
    logger.info("=" * 62)
    logger.info(f"  Python:       {sys.executable}")
    logger.info(f"  COLMAP dir:   {colmap_dir}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Iterations:   {args.iterations}")
    logger.info(f"  Frames dir:   {frames_dir}")
    logger.info("")

    # ── Step 1: Log GPU info ───────────────────────────────────────────
    log_gpu_info(logger)
    logger.info("")

    # ── Step 2: Download gsplat trainer if needed ──────────────────────
    logger.info("Ensuring gsplat trainer script is available...")
    try:
        trainer_script = ensure_gsplat_trainer(project_root, logger)
    except Exception as e:
        logger.error(f"Failed to download gsplat trainer: {e}")
        logger.error("Check internet connectivity or manually place files in")
        logger.error(f"  {project_root / 'scripts' / 'gsplat_examples'}/")
        sys.exit(1)
    logger.info("")

    # ── Step 3: Validate inputs ────────────────────────────────────────
    if not validate_colmap_dir(colmap_dir, logger):
        logger.error("COLMAP directory validation failed. Aborting.")
        sys.exit(1)

    # Ensure images/ directory exists for gsplat
    if not ensure_images_dir(colmap_dir, frames_dir, logger):
        logger.error("Could not set up images/ directory. Aborting.")
        sys.exit(1)
    logger.info("")

    # ── Step 4: Run gsplat training ────────────────────────────────────
    success, log_output = run_gsplat_training(
        trainer_script=trainer_script,
        colmap_dir=colmap_dir,
        output_dir=output_dir,
        iterations=args.iterations,
        logger=logger,
        opacity_reg=args.opacity_reg,
        scale_reg=args.scale_reg,
        prune_opa=args.prune_opa,
    )

    if not success:
        logger.error("")
        logger.error("=" * 62)
        logger.error("  GAUSSIAN SPLATTING TRAINING FAILED")
        logger.error("=" * 62)
        logger.error("  Check the log output above for errors.")
        logger.error("  Common issues:")
        logger.error("    - gsplat not installed: pip install gsplat")
        logger.error("    - Missing deps: pip install tyro viser nerfview==0.0.2")
        logger.error("      torchmetrics[image] tensorboard")
        logger.error("    - COLMAP files malformed: re-run VGGT (Session 2)")
        logger.error("    - CUDA OOM: reduce --iterations or check GPU memory")
        sys.exit(1)

    # ── Step 5: Extract PSNR ───────────────────────────────────────────
    logger.info("")
    logger.info("Extracting PSNR from training log...")
    final_psnr = extract_final_psnr(log_output, logger)

    # ── Step 6: Find and copy output .ply ──────────────────────────────
    logger.info("")
    logger.info("Locating output .ply file...")
    scene_ply = find_and_copy_ply(output_dir, logger)

    # ── Step 7: Save training metadata ─────────────────────────────────
    meta = {
        "colmap_dir": str(colmap_dir),
        "output_dir": str(output_dir),
        "iterations": args.iterations,
        "final_psnr": final_psnr,
        "scene_ply": str(scene_ply) if scene_ply else None,
        "scene_ply_size_mb": round(scene_ply.stat().st_size / (1024 * 1024), 2)
            if scene_ply else None,
        "training_success": success,
        "gsplat_tag": GSPLAT_TAG,
        "python_executable": sys.executable,
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
        logger.info("  Final PSNR:    (could not parse)")
    if scene_ply:
        logger.info(f"  Scene PLY:     {scene_ply}")
        logger.info(f"  PLY size:      {scene_ply.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        logger.info("  Scene PLY:     NOT FOUND -- check output directory")
    logger.info(f"  Metadata:      {meta_path}")
    logger.info("")
    logger.info("  Ready for Session 4 (Grounded SAM2 Semantics)")
    logger.info("")
    logger.info("  Download to Mac:")
    logger.info("    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\")
    logger.info(f"      jrameshs@bluestreak.cs.ucl.ac.uk:{output_dir}/ \\")
    logger.info("      ~/3D-Spatial-Reconstruction/outputs/splat/")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()