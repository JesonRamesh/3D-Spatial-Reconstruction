"""Extract frames from a room video at a configurable FPS using ffmpeg."""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("roboscene.extract")
    logger.setLevel(logging.DEBUG)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_dir / "extraction.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s"
    ))
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> str:
    """Verify ffmpeg is installed and return its path."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, check=True,
        )
        version_line = result.stdout.split("\n")[0]
        return version_line
    except FileNotFoundError:
        print(
            "ERROR: ffmpeg not found on PATH.\n"
            "Install it:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  conda:  conda install -c conda-forge ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg returned an error: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def probe_video(video_path: Path) -> dict:
    """Use ffprobe to get video metadata."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffprobe failed: {e}") from e


def get_video_info(video_path: Path) -> dict:
    """Extract key video properties."""
    probe = probe_video(video_path)

    # Find the video stream
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise RuntimeError(f"No video stream found in {video_path}")

    # Parse FPS (can be a fraction like "59.94/1")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    # Duration: try stream first, then format
    duration = float(video_stream.get(
        "duration",
        probe.get("format", {}).get("duration", 0)
    ))

    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0:
        total_frames = int(fps * duration)

    return {
        "path": str(video_path),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": round(fps, 2),
        "duration_sec": round(duration, 2),
        "total_frames": total_frames,
        "codec": video_stream.get("codec_name", "unknown"),
        "pix_fmt": video_stream.get("pix_fmt", "unknown"),
    }


def extract_frames_ffmpeg(
    video_path: Path,
    output_dir: Path,
    fps: float,
    quality: int,
    logger: logging.Logger,
) -> int:
    """
    Extract frames from video using ffmpeg at the specified FPS rate.

    Returns the number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output pattern: frame_0001.jpg, frame_0002.jpg, ...
    output_pattern = str(output_dir / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-qmin", "1",
        "-q:v", str(max(1, min(31, (100 - quality) * 31 // 100 + 1))),
        "-start_number", "1",
        output_pattern,
        "-y",  # overwrite existing files
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg stderr:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

    # Log ffmpeg output at debug level
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-5:]:
            logger.debug(f"ffmpeg: {line.strip()}")

    # Count extracted frames
    extracted = sorted(output_dir.glob("frame_*.jpg"))
    return len(extracted)


# ---------------------------------------------------------------------------
# Post-extraction analysis (using OpenCV if available)
# ---------------------------------------------------------------------------

def compute_frame_sharpness(frame_path: Path) -> float:
    """Compute Laplacian variance as a sharpness score."""
    try:
        import cv2
        img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except ImportError:
        return -1.0  # OpenCV not available


def analyse_extracted_frames(output_dir: Path, logger: logging.Logger) -> list:
    """Compute per-frame metadata for all extracted frames."""
    frames = sorted(output_dir.glob("frame_*.jpg"))
    records = []

    for i, fpath in enumerate(frames):
        sharpness = compute_frame_sharpness(fpath)
        stat = fpath.stat()

        record = {
            "filename": fpath.name,
            "frame_index": i + 1,
            "sharpness": round(sharpness, 2),
            "file_size_bytes": stat.st_size,
        }
        records.append(record)

        if sharpness >= 0:
            logger.debug(
                f"  {fpath.name}: sharpness={sharpness:.1f}, "
                f"size={stat.st_size / 1024:.0f}KB"
            )

    return records


# ---------------------------------------------------------------------------
# Blur filter
# ---------------------------------------------------------------------------

def filter_blurry_frames(
    output_dir: Path,
    frame_records: list,
    drop_bottom_pct: float,
    logger: logging.Logger,
) -> tuple:
    """
    Delete the bottom drop_bottom_pct% of frames ranked by Laplacian sharpness,
    then renumber the survivors sequentially (frame_0001.jpg, frame_0002.jpg, …).

    Returns (surviving_records, filter_report) where filter_report is a dict
    suitable for inclusion in the metadata JSON.

    Skips filtering entirely if no sharpness scores are available (OpenCV missing).
    """
    scored = [r for r in frame_records if r["sharpness"] >= 0]
    if not scored:
        logger.warning(
            "Blur filter skipped — OpenCV not available. "
            "Install opencv-python to enable sharpness filtering."
        )
        return frame_records, {"skipped": True, "reason": "opencv_unavailable"}

    total = len(scored)
    n_drop = max(0, int(total * drop_bottom_pct / 100.0))

    if n_drop == 0:
        logger.info(f"Blur filter: drop_bottom_pct={drop_bottom_pct}% → 0 frames to drop")
        return frame_records, {"skipped": True, "reason": "nothing_to_drop"}

    # Sort by ascending sharpness to find the threshold
    by_sharpness = sorted(scored, key=lambda r: r["sharpness"])
    threshold = by_sharpness[n_drop - 1]["sharpness"]
    to_drop = {r["filename"] for r in by_sharpness[:n_drop]}

    logger.info(
        f"Blur filter: dropping bottom {drop_bottom_pct}% "
        f"({n_drop}/{total} frames, sharpness < {threshold:.1f})"
    )

    # Delete blurry frames
    actually_dropped = 0
    for fname in to_drop:
        p = output_dir / fname
        if p.exists():
            p.unlink()
            actually_dropped += 1
            logger.debug(f"  dropped: {fname}")

    # Renumber survivors in sorted order
    survivors = sorted(output_dir.glob("frame_*.jpg"))
    for new_idx, p in enumerate(survivors, start=1):
        new_name = output_dir / f"frame_{new_idx:04d}.jpg"
        if p.name != new_name.name:
            p.rename(new_name)

    # Rebuild records for survivors (sharpness already computed, just remap filenames)
    surviving_names = {f"frame_{i:04d}.jpg" for i in range(1, len(survivors) + 1)}
    orig_by_name = {r["filename"]: r for r in frame_records}

    # Map old filenames → new sequential names in the order they were sorted
    old_survivors_sorted = sorted(
        [r for r in frame_records if r["filename"] not in to_drop],
        key=lambda r: r["filename"],
    )
    surviving_records = []
    for new_idx, r in enumerate(old_survivors_sorted, start=1):
        new_r = dict(r)
        new_r["filename"] = f"frame_{new_idx:04d}.jpg"
        new_r["frame_index"] = new_idx
        surviving_records.append(new_r)

    kept = len(surviving_records)
    logger.info(
        f"Blur filter complete: {actually_dropped} dropped, "
        f"{kept} sharp frames kept, renumbered frame_0001..frame_{kept:04d}.jpg"
    )

    filter_report = {
        "skipped": False,
        "drop_bottom_pct": drop_bottom_pct,
        "frames_before": total,
        "frames_dropped": actually_dropped,
        "frames_kept": kept,
        "sharpness_threshold": round(threshold, 2),
    }
    return surviving_records, filter_report


# ---------------------------------------------------------------------------
# Metadata output
# ---------------------------------------------------------------------------

def save_metadata(
    video_info: dict,
    frame_records: list,
    fps: float,
    elapsed_sec: float,
    output_dir: Path,
    logger: logging.Logger,
    blur_filter_report: dict = None,
) -> None:
    """Save extraction metadata as JSON and CSV."""
    import csv

    sharpness_vals = [r["sharpness"] for r in frame_records if r["sharpness"] >= 0]

    report = {
        "video": video_info,
        "extraction": {
            "fps": fps,
            "extracted_frames": len(frame_records),
            "elapsed_sec": round(elapsed_sec, 2),
        },
        "quality_stats": {
            "sharpness_min": round(min(sharpness_vals), 2) if sharpness_vals else None,
            "sharpness_max": round(max(sharpness_vals), 2) if sharpness_vals else None,
            "sharpness_mean": round(
                sum(sharpness_vals) / len(sharpness_vals), 2
            ) if sharpness_vals else None,
        },
        "blur_filter": blur_filter_report or {"skipped": True, "reason": "not_requested"},
        "frames": frame_records,
    }

    # JSON
    json_path = output_dir / "extraction_metadata.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Metadata JSON: {json_path}")

    # CSV
    csv_path = output_dir / "frames_summary.csv"
    if frame_records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_records[0].keys())
            writer.writeheader()
            writer.writerows(frame_records)
        logger.info(f"Metadata CSV:  {csv_path}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML config if available."""
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fall back to defaults
        return {}
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RoboScene+ Session 1: Extract frames from video using ffmpeg"
    )
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Path to input video file (default: from config.yaml)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for frames (default: data/frames)"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Frames per second to extract (default: 1.0)"
    )
    parser.add_argument(
        "--quality", type=int, default=None,
        help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--filter_blur_bottom_pct", type=float, default=0.0,
        help=(
            "Delete the bottom N%% of frames ranked by Laplacian sharpness before "
            "saving. Survivors are renumbered sequentially. 0 = no filtering (default). "
            "Recommended: 20 (matches Spatiality v2 approach). Requires opencv-python."
        ),
    )
    args = parser.parse_args()

    # Resolve project root (parent of scripts/)
    project_root = Path(__file__).resolve().parent.parent

    # Load config
    config_path = project_root / args.config
    cfg = load_config(config_path)
    extraction_cfg = cfg.get("extraction", {})
    paths_cfg = cfg.get("paths", {})

    # Resolve parameters: CLI > config.yaml > defaults
    video_path = Path(args.video_path or paths_cfg.get("video", "room.MOV"))
    if not video_path.is_absolute():
        video_path = project_root / video_path

    output_dir = Path(args.output_dir or paths_cfg.get("frames_dir", "data/frames"))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    fps = args.fps or extraction_cfg.get("fps", 1.0)
    quality = args.quality or extraction_cfg.get("quality", 95)

    # Setup logging
    logger = setup_logging(project_root / "logs")

    logger.info("=" * 62)
    logger.info("  RoboScene+ — Frame Extraction (Session 1)")
    logger.info("=" * 62)

    # Check ffmpeg
    ffmpeg_version = check_ffmpeg()
    logger.info(f"ffmpeg: {ffmpeg_version}")

    # Check video exists
    if not video_path.is_file():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Get video info
    video_info = get_video_info(video_path)
    logger.info(f"Video:      {video_info['path']}")
    logger.info(f"  Codec:    {video_info['codec']}  |  {video_info['pix_fmt']}")
    logger.info(f"  Size:     {video_info['width']}×{video_info['height']}")
    logger.info(f"  FPS:      {video_info['fps']}")
    logger.info(f"  Duration: {video_info['duration_sec']}s  "
                f"({video_info['total_frames']} total frames)")
    logger.info(f"Extracting at: {fps} fps  →  "
                f"~{int(video_info['duration_sec'] * fps)} expected frames")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Quality:    {quality}")
    if args.filter_blur_bottom_pct > 0:
        logger.info(f"Blur filter: will drop bottom {args.filter_blur_bottom_pct}% by Laplacian sharpness")

    # Extract frames
    t0 = time.time()
    num_frames = extract_frames_ffmpeg(video_path, output_dir, fps, quality, logger)
    elapsed = time.time() - t0

    if num_frames == 0:
        logger.error("No frames extracted! Check video file and ffmpeg installation.")
        sys.exit(1)

    logger.info(f"Extraction complete: {num_frames} frames in {elapsed:.1f}s")

    # Analyse frames (sharpness scoring)
    logger.info("Analysing frame quality...")
    frame_records = analyse_extracted_frames(output_dir, logger)

    # Apply blur filter if requested
    blur_report = None
    if args.filter_blur_bottom_pct > 0:
        logger.info(f"Applying blur filter (drop bottom {args.filter_blur_bottom_pct}%)...")
        frame_records, blur_report = filter_blurry_frames(
            output_dir, frame_records, args.filter_blur_bottom_pct, logger
        )

    # Save metadata (reflects post-filter state)
    save_metadata(video_info, frame_records, fps, elapsed, output_dir, logger, blur_report)

    # Print summary
    sharpness_vals = [r["sharpness"] for r in frame_records if r["sharpness"] >= 0]
    total_size_mb = sum(r["file_size_bytes"] for r in frame_records) / (1024 * 1024)
    final_count = len(frame_records)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("=" * 62)
    logger.info(f"  Frames extracted:   {num_frames}")
    if blur_report and not blur_report.get("skipped"):
        logger.info(f"  Blurry dropped:     {blur_report['frames_dropped']} "
                    f"(sharpness < {blur_report['sharpness_threshold']})")
        logger.info(f"  Sharp frames kept:  {blur_report['frames_kept']}")
    logger.info(f"  Output directory:   {output_dir}")
    logger.info(f"  Total size:         {total_size_mb:.1f} MB")
    if sharpness_vals:
        logger.info(f"  Sharpness range:    {min(sharpness_vals):.1f} – "
                     f"{max(sharpness_vals):.1f}")
        logger.info(f"  Mean sharpness:     "
                     f"{sum(sharpness_vals) / len(sharpness_vals):.1f}")
    logger.info(f"  Time elapsed:       {elapsed:.1f}s")
    logger.info("=" * 62)

    # Validate final frame count (300–1000 is ideal for VGGT with 5fps input)
    if 300 <= final_count <= 1000:
        logger.info(f"\n  ✅ {final_count} frames — good range for VGGT")
        logger.info("  ✅ Ready for VGGT reconstruction")
    elif final_count < 100:
        logger.warning(f"\n  ⚠️  Only {final_count} frames — consider increasing --fps")
    elif final_count < 300:
        logger.warning(f"\n  ⚠️  {final_count} frames — workable but more is better; "
                       f"consider increasing --fps")
    else:
        logger.info(f"\n  ✅ {final_count} frames extracted")


if __name__ == "__main__":
    main()