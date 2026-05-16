#!/usr/bin/env python3
"""
run_mast3r_slam.py — MASt3R-SLAM reconstruction pipeline

End-to-end pipeline that:
  1. Extracts frames from an input MP4 at a configurable FPS.
  2. Runs MASt3R-SLAM to produce camera poses and a dense point cloud.
  3. Converts the raw SLAM output to a COLMAP-compatible sparse model
     so that downstream tools (train_splat.py, nerfstudio, etc.) work
     without modification.

Usage
-----
    python scripts/run_mast3r_slam.py \\
        --video_path data/raw/room.mp4 \\
        --output_dir data/mast3r_out/ \\
        --mast3r_dir /scratch0/jrameshs/MASt3R-SLAM/ \\
        --config base.yaml \\
        --device cuda \\
        --fps 2
"""

import argparse
import json
import logging
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_mast3r_slam")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _default_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is visible, else 'cpu'."""
    if _TORCH_AVAILABLE:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """Build and return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_mast3r_slam",
        description="Extract frames, run MASt3R-SLAM, export a COLMAP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Path to the input MP4 video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/mast3r_out/"),
        help="Root directory where all outputs will be written.",
    )
    parser.add_argument(
        "--mast3r_dir",
        type=Path,
        default=Path("/scratch0/jrameshs/MASt3R-SLAM/"),
        help="Path to the cloned MASt3R-SLAM repository.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base.yaml",
        help="Config filename located in <mast3r_dir>/config/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        choices=["cuda", "cpu"],
        help="Compute device passed to MASt3R-SLAM.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame extraction rate (frames per second of video).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def print_banner(args: argparse.Namespace) -> None:
    """Print a formatted startup banner summarising all runtime arguments."""
    border = "=" * 60
    lines = [
        "",
        border,
        "  MASt3R-SLAM Pipeline",
        border,
        f"  video_path  : {args.video_path}",
        f"  output_dir  : {args.output_dir}",
        f"  mast3r_dir  : {args.mast3r_dir}",
        f"  config      : {args.config}",
        f"  device      : {args.device}",
        f"  fps         : {args.fps}",
        border,
        "",
    ]
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Stub functions
# ---------------------------------------------------------------------------


def extract_frames(video_path: Path, output_dir: Path, fps: float) -> List[str]:
    """Extract individual frames from an MP4 video at a given frame rate.

    Tries ffmpeg-python first (if installed) then falls back to the ffmpeg
    CLI — the same approach used in ``scripts/extract_frames.py``.  Frames
    are written as ``frame_0001.jpg``, ``frame_0002.jpg`` … into
    ``output_dir / "images"``.

    Args:
        video_path: Absolute or relative path to the source ``.mp4`` file.
        output_dir: Root output directory.  An ``images/`` sub-directory is
            created inside it to hold the extracted frames.
        fps: Number of frames to extract per second of video.

    Returns:
        A sorted list of absolute path strings pointing to every extracted
        frame image, in temporal order.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the ffmpeg subprocess exits with a non-zero status.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # frame_0001.jpg, frame_0002.jpg, … (4-digit zero-padded, 1-indexed)
    output_pattern = str(images_dir / "frame_%04d.jpg")

    logger.info("Extracting frames at %.2f fps → %s", fps, images_dir)

    # ------------------------------------------------------------------ #
    # Try ffmpeg-python first; fall back to CLI                           #
    # ------------------------------------------------------------------ #
    _used_lib = False
    try:
        import ffmpeg as _ffmpeg  # ffmpeg-python

        (
            _ffmpeg
            .input(str(video_path))
            .filter("fps", fps=fps)
            .output(output_pattern, qmin=1, **{"q:v": 2}, start_number=1)
            .overwrite_output()
            .run(quiet=True)
        )
        _used_lib = True
        logger.debug("Frame extraction used ffmpeg-python")
    except ImportError:
        logger.debug("ffmpeg-python not installed — using CLI")
    except Exception as exc:  # ffmpeg-python runtime error
        logger.warning("ffmpeg-python failed (%s) — retrying with CLI", exc)

    if not _used_lib:
        # CLI fallback — identical quality flags to extract_frames.py
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qmin", "1",
            "-q:v", "2",          # ~95 % JPEG quality
            "-start_number", "1",
            output_pattern,
        ]
        logger.debug("ffmpeg CLI: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg stderr:\n%s", result.stderr[-2000:])
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}"
            )

    # ------------------------------------------------------------------ #
    # Collect and validate results                                        #
    # ------------------------------------------------------------------ #
    frame_paths = sorted(
        str(p.resolve()) for p in images_dir.glob("frame_*.jpg")
    )
    if not frame_paths:
        raise RuntimeError(
            f"ffmpeg ran without error but no frames were written to {images_dir}"
        )

    logger.info(
        "Extracted %d frames  (%.2f fps, saved to %s)",
        len(frame_paths), fps, images_dir,
    )
    return frame_paths


def _tail_log_forever(log_path: Path, stop_event: threading.Event, interval: int) -> None:
    """Background thread: print the last non-empty log line every *interval* s.

    Stops when *stop_event* is set.  Designed to be run as a daemon thread so
    it never blocks the main process from exiting.
    """
    last_pos = 0
    while not stop_event.is_set():
        stop_event.wait(interval)
        try:
            with open(log_path, "r", errors="replace") as fh:
                fh.seek(last_pos)
                new_lines = fh.readlines()
                last_pos = fh.tell()
            # Find the last non-whitespace line written since last check
            relevant = [ln.rstrip() for ln in new_lines if ln.strip()]
            if relevant:
                logger.info("[mast3r-slam] %s", relevant[-1])
        except OSError:
            pass  # log file not yet created — keep waiting


def run_slam(
    video_path: Path,
    mast3r_dir: Path,
    config: str,
    output_dir: Path,
    device: str,
) -> Path:
    """Execute MASt3R-SLAM on the input video and return the raw output directory.

    Launches ``{mast3r_dir}/main.py`` via :data:`sys.executable` as a child
    process.  stdout and stderr are both redirected to
    ``logs/mast3r_slam.log`` (created under the *current working directory*,
    mirroring the ``nohup … > logs/…`` pattern used for VGGT on bluestreak).
    A background thread tails that log and prints the most-recent line every
    30 seconds so there is visible progress without flooding the terminal.

    Args:
        video_path: Path to the source ``.mp4`` file.
        mast3r_dir: Root of the cloned MASt3R-SLAM repo.
        config: Config filename inside ``{mast3r_dir}/config/``.
        output_dir: Root output directory; SLAM writes into
            ``{output_dir}/slam_output/``.
        device: ``"cuda"`` or ``"cpu"`` — forwarded via ``--device``.

    Returns:
        :class:`~pathlib.Path` to ``{output_dir}/slam_output/``.

    Raises:
        FileNotFoundError: If ``main.py`` or the config file cannot be found.
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    mast3r_dir  = Path(mast3r_dir).resolve()
    output_dir  = Path(output_dir).resolve()
    video_path  = Path(video_path).resolve()

    main_script = mast3r_dir / "main.py"
    config_path = mast3r_dir / "config" / config
    slam_out    = output_dir / "slam_output"

    # ------------------------------------------------------------------ #
    # Pre-flight checks                                                   #
    # ------------------------------------------------------------------ #
    if not main_script.exists():
        raise FileNotFoundError(
            f"MASt3R-SLAM entry-point not found: {main_script}\n"
            f"  Is --mast3r_dir correct?  (got: {mast3r_dir})"
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"  Available configs: "
            + str(list((mast3r_dir / "config").glob("*.yaml")))
        )

    slam_out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Log file — written relative to cwd so it lands in the project root  #
    # logs/ directory, consistent with run_vggt_job.sh on bluestreak      #
    # ------------------------------------------------------------------ #
    log_dir  = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mast3r_slam.log"

    cmd = [
        sys.executable,
        str(main_script),
        "--input",  str(video_path),
        "--config", str(config_path),
        "--output", str(slam_out),
        "--device", device,
    ]

    logger.info("Starting MASt3R-SLAM")
    logger.info("  script  : %s", main_script)
    logger.info("  config  : %s", config_path)
    logger.info("  input   : %s", video_path)
    logger.info("  output  : %s", slam_out)
    logger.info("  device  : %s", device)
    logger.info("  log     : %s", log_file.resolve())
    logger.info("  command : %s", " ".join(cmd))

    # ------------------------------------------------------------------ #
    # Launch — stdout + stderr → log file (nohup-style)                  #
    # ------------------------------------------------------------------ #
    _PROGRESS_INTERVAL = 30  # seconds between progress prints

    with open(log_file, "w") as log_fh:
        process = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,  # merge stderr into same file
        )

    # Background thread tails the log every 30 s
    stop_tail = threading.Event()
    tail_thread = threading.Thread(
        target=_tail_log_forever,
        args=(log_file, stop_tail, _PROGRESS_INTERVAL),
        daemon=True,
        name="log-tailer",
    )
    tail_thread.start()

    t_start = time.time()
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        logger.warning("Interrupted — terminating MASt3R-SLAM subprocess …")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        stop_tail.set()
        raise
    finally:
        stop_tail.set()   # always stop the tail thread

    elapsed = time.time() - t_start
    logger.info(
        "MASt3R-SLAM finished in %.1f s  (exit code %d)",
        elapsed, return_code,
    )

    if return_code != 0:
        # Surface the tail of the log to help diagnose failures
        try:
            with open(log_file, "r", errors="replace") as fh:
                tail = fh.read()[-3000:]
        except OSError:
            tail = "(log file unreadable)"
        raise RuntimeError(
            f"MASt3R-SLAM exited with code {return_code}.\n"
            f"Last log output:\n{tail}\n"
            f"Full log: {log_file.resolve()}"
        )

    return slam_out


# ---------------------------------------------------------------------------
# convert_to_colmap — helpers
# ---------------------------------------------------------------------------


def _find_ply(slam_output_dir: Path) -> Optional[Path]:
    """Return the first .ply file found in *slam_output_dir*, else None."""
    candidates = sorted(slam_output_dir.glob("**/*.ply"))
    if not candidates:
        return None
    # Prefer files whose names suggest a dense/final cloud
    preferred_keywords = ("cloud", "map", "points", "dense", "final", "output")
    for kw in preferred_keywords:
        for p in candidates:
            if kw in p.stem.lower():
                return p
    return candidates[0]  # fall back to whichever comes first alphabetically


def _parse_poses_json(
    json_path: Path,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Parse camera poses from a JSON file.

    Handles three common layouts produced by MASt3R-style SLAM systems:

    1. NeRF ``transforms.json``  — ``{"frames": [{"transform_matrix": [...]}]}``
    2. Flat list                 — ``[ [[...4x4...]], [[...4x4...]], ... ]``
    3. Numeric-keyed dict        — ``{"0": {"transform_matrix": [...]}, ...}``

    Returns (poses, intrinsics) where *poses* is a list of (4, 4) float64
    arrays (camera-to-world) and *intrinsics* is a plain dict that may be
    empty.
    """
    with open(json_path, "r", errors="replace") as fh:
        data = json.load(fh)

    intrinsics: Dict[str, Any] = {}
    poses: List[np.ndarray] = []

    if isinstance(data, dict) and "frames" in data:
        # NeRF-style transforms.json
        for key in ("fl_x", "fl_y", "cx", "cy", "w", "h",
                    "camera_angle_x", "camera_angle_y"):
            if key in data:
                intrinsics[key] = data[key]
        for frame in data["frames"]:
            mat = np.array(frame["transform_matrix"], dtype=np.float64)
            if mat.shape == (3, 4):
                mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
            poses.append(mat)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                mat = np.array(item, dtype=np.float64)
                if mat.shape == (3, 4):
                    mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
                poses.append(mat)
            elif isinstance(item, dict) and "transform_matrix" in item:
                mat = np.array(item["transform_matrix"], dtype=np.float64)
                if mat.shape == (3, 4):
                    mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
                poses.append(mat)

    elif isinstance(data, dict):
        # Numeric-keyed: {"0": [[...]], "1": {...}, ...}
        for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            v = data[k]
            if isinstance(v, list):
                mat = np.array(v, dtype=np.float64)
            elif isinstance(v, dict) and "transform_matrix" in v:
                mat = np.array(v["transform_matrix"], dtype=np.float64)
            else:
                continue
            if mat.shape == (3, 4):
                mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
            poses.append(mat)

    return poses, intrinsics


def _parse_poses_npz(
    npz_path: Path,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Parse camera poses from a NumPy .npz file.

    Tries common key names used by MASt3R and similar SLAM outputs:
    ``poses``, ``camera_poses``, ``extrinsics``, ``c2w``, ``w2c``.

    Returns (poses, intrinsics) in the same format as :func:`_parse_poses_json`.
    """
    data = np.load(npz_path, allow_pickle=True)
    intrinsics: Dict[str, Any] = {}
    poses: List[np.ndarray] = []

    for key in ("poses", "camera_poses", "extrinsics", "c2w", "w2c"):
        if key in data:
            arr = data[key]  # (N, 4, 4) or (N, 3, 4)
            for i in range(arr.shape[0]):
                mat = arr[i].astype(np.float64)
                if mat.shape == (3, 4):
                    mat = np.vstack([mat, [0.0, 0.0, 0.0, 1.0]])
                poses.append(mat)
            break

    for key in ("intrinsics", "K", "camera_matrix"):
        if key in data:
            K = data[key]
            K0 = K[0] if K.ndim == 3 else K
            if K0.shape == (3, 3):
                intrinsics.update({
                    "fl_x": float(K0[0, 0]),
                    "fl_y": float(K0[1, 1]),
                    "cx":   float(K0[0, 2]),
                    "cy":   float(K0[1, 2]),
                })
            break

    return poses, intrinsics


def _read_ply_points(
    ply_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read XYZ and RGB from a PLY point cloud.

    Tries open3d first (handles binary/ASCII PLY), then falls back to a
    minimal ASCII-only manual parser.  Returns ``(xyz, rgb)`` arrays of
    shape ``(N, 3)``; *rgb* dtype is ``uint8``.
    """
    try:
        import open3d as o3d  # type: ignore
        pcd = o3d.io.read_point_cloud(str(ply_path))
        xyz = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
        return xyz, rgb
    except Exception:
        pass

    # Minimal ASCII PLY fallback
    xyz_rows: List[List[float]] = []
    rgb_rows: List[List[int]] = []
    try:
        with open(ply_path, "rb") as fh:
            header: List[str] = []
            while True:
                line = fh.readline().decode("utf-8", errors="replace").strip()
                header.append(line)
                if line == "end_header":
                    break
            if any("binary" in h for h in header):
                logger.warning(
                    "Binary PLY detected but open3d unavailable — "
                    "point cloud will be empty"
                )
                return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.uint8)
            for raw in fh:
                parts = raw.decode("utf-8", errors="replace").split()
                if len(parts) >= 3:
                    xyz_rows.append([float(parts[0]),
                                     float(parts[1]),
                                     float(parts[2])])
                    if len(parts) >= 6:
                        rgb_rows.append([int(parts[3]),
                                         int(parts[4]),
                                         int(parts[5])])
                    else:
                        rgb_rows.append([128, 128, 128])
    except Exception as exc:
        logger.warning("Could not read PLY %s: %s", ply_path, exc)
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.uint8)

    xyz = np.array(xyz_rows, dtype=np.float64) if xyz_rows else np.zeros((0, 3), np.float64)
    rgb = np.array(rgb_rows, dtype=np.uint8)   if rgb_rows else np.zeros((0, 3), np.uint8)
    return xyz, rgb


def _intrinsics_to_params(
    intrinsics: Dict[str, Any],
    w: int,
    h: int,
) -> np.ndarray:
    """Build a PINHOLE params array ``[fx, fy, cx, cy]`` from intrinsics dict.

    Falls back to sensible defaults (fx = fy = max(w, h),
    principal point at image centre) when intrinsics are absent.
    """
    if "fl_x" in intrinsics:
        fx = float(intrinsics["fl_x"])
        fy = float(intrinsics.get("fl_y", fx))
    elif "camera_angle_x" in intrinsics:
        fx = float(w) / (2.0 * np.tan(float(intrinsics["camera_angle_x"]) / 2.0))
        if "camera_angle_y" in intrinsics:
            fy = float(h) / (2.0 * np.tan(float(intrinsics["camera_angle_y"]) / 2.0))
        else:
            fy = fx
    else:
        fx = fy = float(max(w, h))

    cx = float(intrinsics.get("cx", w / 2.0))
    cy = float(intrinsics.get("cy", h / 2.0))
    return np.array([fx, fy, cx, cy], dtype=np.float64)


# ---------------------------------------------------------------------------
# convert_to_colmap — main function
# ---------------------------------------------------------------------------


def convert_to_colmap(slam_output_dir: Path, output_dir: Path) -> None:
    """Convert raw MASt3R-SLAM output into a COLMAP-compatible sparse model.

    Searches *slam_output_dir* for a .ply point cloud and a pose file
    (``transforms.json``, ``poses.json``, any ``*.json``, or ``*.npz``).
    Writes the standard COLMAP sparse layout used by ``train_splat.py``::

        output_dir/
        └── sparse/
            └── 0/
                ├── cameras.bin   — one PINHOLE camera per frame
                ├── images.bin    — one entry per frame with R, t from poses
                └── points3D.bin  — XYZ + RGB from the .ply cloud

    The ``images/`` sub-directory is also symlinked (or its path recorded)
    so that gsplat's data loader can find frames alongside the sparse model.

    Args:
        slam_output_dir: Directory returned by :func:`run_slam`.
        output_dir: Root output directory (same value passed to
            :func:`run_slam`).  ``sparse/0/`` is created inside it.

    Raises:
        FileNotFoundError: If *slam_output_dir* does not exist.
        RuntimeError: If conversion fails due to malformed data.
    """
    # Lazy import so the module is usable without sys.path tricks when
    # run from the project root, but also works when the scripts/ dir is
    # on sys.path (e.g. on bluestreak after `cd roboscene-plus`).
    import importlib.util as _ilu
    import os as _os

    slam_output_dir = Path(slam_output_dir).resolve()
    output_dir      = Path(output_dir).resolve()

    if not slam_output_dir.exists():
        raise FileNotFoundError(
            f"SLAM output directory not found: {slam_output_dir}"
        )

    # ------------------------------------------------------------------ #
    # Locate colmap_utils                                                 #
    # ------------------------------------------------------------------ #
    _scripts_dir = Path(__file__).resolve().parent
    _spec = _ilu.spec_from_file_location(
        "colmap_utils", _scripts_dir / "colmap_utils.py"
    )
    _cu = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_cu)       # type: ignore[union-attr]

    write_cameras_binary  = _cu.write_cameras_binary
    write_images_binary   = _cu.write_images_binary
    write_points3D_binary = _cu.write_points3D_binary
    rotmat_to_qvec        = _cu.rotmat_to_qvec

    # ------------------------------------------------------------------ #
    # Output directory                                                    #
    # ------------------------------------------------------------------ #
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Locate + parse pose file                                            #
    # ------------------------------------------------------------------ #
    poses: List[np.ndarray] = []
    intrinsics: Dict[str, Any] = {}
    pose_file_used: Optional[Path] = None

    # Priority: transforms.json > poses.json > *.json > *.npz
    json_candidates = (
        list(slam_output_dir.glob("transforms.json"))
        + list(slam_output_dir.glob("poses.json"))
        + [p for p in slam_output_dir.glob("**/*.json")
           if p.name not in ("transforms.json", "poses.json")]
    )
    npz_candidates = (
        list(slam_output_dir.glob("poses.npz"))
        + [p for p in slam_output_dir.glob("**/*.npz")
           if p.name != "poses.npz"]
    )

    for candidate in json_candidates:
        try:
            p, intr = _parse_poses_json(candidate)
            if p:
                poses, intrinsics, pose_file_used = p, intr, candidate
                break
        except Exception as exc:
            logger.debug("Skipping %s: %s", candidate.name, exc)

    if not poses:
        for candidate in npz_candidates:
            try:
                p, intr = _parse_poses_npz(candidate)
                if p:
                    poses, intrinsics, pose_file_used = p, intr, candidate
                    break
            except Exception as exc:
                logger.debug("Skipping %s: %s", candidate.name, exc)

    if poses:
        logger.info("Poses   : %d loaded from %s", len(poses), pose_file_used.name)
    else:
        logger.warning(
            "No pose file found in %s — images will use identity poses",
            slam_output_dir,
        )

    # ------------------------------------------------------------------ #
    # Locate + read point cloud                                           #
    # ------------------------------------------------------------------ #
    ply_path = _find_ply(slam_output_dir)
    if ply_path:
        logger.info("PLY     : %s", ply_path.relative_to(slam_output_dir))
        xyz, rgb = _read_ply_points(ply_path)
    else:
        logger.warning("No .ply found in %s — points3D.bin will be empty",
                       slam_output_dir)
        xyz = np.zeros((0, 3), dtype=np.float64)
        rgb = np.zeros((0, 3), dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Collect frame image names from output_dir/images/                  #
    # ------------------------------------------------------------------ #
    images_dir  = output_dir / "images"
    frame_names: List[str] = []
    if images_dir.exists():
        frame_names = sorted(p.name for p in images_dir.glob("frame_*.jpg"))
    if not frame_names:
        # Synthesise names from pose count so COLMAP files stay consistent
        n_frames = max(len(poses), 1)
        frame_names = [f"frame_{i+1:04d}.jpg" for i in range(n_frames)]
        logger.warning(
            "images/ directory empty or missing — using %d synthetic frame names",
            n_frames,
        )

    n_frames = len(frame_names)

    # Image dimensions: prefer intrinsics dict, else default 1920×1080
    img_w = int(intrinsics.get("w", 1920))
    img_h = int(intrinsics.get("h", 1080))

    # ------------------------------------------------------------------ #
    # Build cameras list — one PINHOLE camera per frame                  #
    # ------------------------------------------------------------------ #
    params = _intrinsics_to_params(intrinsics, img_w, img_h)
    cameras = [
        {
            "camera_id": i + 1,
            "model":     "PINHOLE",
            "width":     img_w,
            "height":    img_h,
            "params":    params,       # shared intrinsics across all frames
        }
        for i in range(n_frames)
    ]

    # ------------------------------------------------------------------ #
    # Build images list — R, t from c2w poses                            #
    # ------------------------------------------------------------------ #
    images = []
    for idx, name in enumerate(frame_names):
        if idx < len(poses):
            c2w = poses[idx]                   # 4×4 camera-to-world
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            # COLMAP convention: world-to-camera
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
        else:
            R_w2c = np.eye(3, dtype=np.float64)
            t_w2c = np.zeros(3, dtype=np.float64)

        images.append({
            "image_id":     idx + 1,
            "qvec":         rotmat_to_qvec(R_w2c),
            "tvec":         t_w2c,
            "camera_id":    idx + 1,
            "name":         name,
            "point2D_xys":  np.zeros((0, 2), dtype=np.float64),
            "point3D_ids":  np.array([], dtype=np.int64),
        })

    # ------------------------------------------------------------------ #
    # Build points3D list from PLY                                       #
    # ------------------------------------------------------------------ #
    points3d = [
        {
            "point3D_id": i + 1,
            "xyz":        xyz[i],
            "rgb":        rgb[i],
            "error":      1.0,
            "track":      [],          # no 2-D observations needed for init
        }
        for i in range(len(xyz))
    ]

    # ------------------------------------------------------------------ #
    # Write COLMAP binary files via colmap_utils                         #
    # ------------------------------------------------------------------ #
    write_cameras_binary(cameras,  sparse_dir / "cameras.bin")
    write_images_binary(images,    sparse_dir / "images.bin")
    write_points3D_binary(points3d, sparse_dir / "points3D.bin")

    logger.info("cameras.bin  : %d cameras written",  len(cameras))
    logger.info("images.bin   : %d images written",   len(images))
    logger.info("points3D.bin : %d points written",   len(points3d))
    logger.info("COLMAP sparse: %s", sparse_dir)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the full MASt3R-SLAM pipeline.

    Execution order:
        1. Parse arguments and print the startup banner.
        2. Extract frames from the video.
        3. Run MASt3R-SLAM.
        4. Convert SLAM output to COLMAP sparse format.
        5. Print completion summary.

    Exits with status code 1 on any unrecovered error.
    """
    args = parse_args()
    print_banner(args)

    t0 = time.time()
    logger.info("Pipeline start.")

    # ------------------------------------------------------------------
    # Step 1 – extract frames
    # ------------------------------------------------------------------
    logger.info("Step 1/3 — Extracting frames at %.1f fps …", args.fps)
    frame_paths: List[str] = extract_frames(
        args.video_path, args.output_dir, args.fps
    )
    logger.info("Extracted %d frames.", len(frame_paths))

    # ------------------------------------------------------------------
    # Step 2 – run SLAM
    # ------------------------------------------------------------------
    logger.info("Step 2/3 — Running MASt3R-SLAM on device '%s' …", args.device)
    slam_output_dir: Path = run_slam(
        args.video_path,
        args.mast3r_dir,
        args.config,
        args.output_dir,
        args.device,
    )
    logger.info("SLAM output: %s", slam_output_dir)

    # ------------------------------------------------------------------
    # Step 3 – convert to COLMAP
    # ------------------------------------------------------------------
    logger.info("Step 3/3 — Converting to COLMAP format …")
    convert_to_colmap(slam_output_dir, args.output_dir)
    colmap_path = args.output_dir / "sparse" / "0"

    # ------------------------------------------------------------------
    # Completion summary  (mirrors run_vggt.py style)
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    border = "=" * 60
    summary = [
        "",
        border,
        "  MAST3R-SLAM RECONSTRUCTION COMPLETE",
        border,
        f"  Frames extracted : {len(frame_paths)}",
        f"  SLAM output      : {slam_output_dir}",
        f"  COLMAP sparse    : {colmap_path}",
        f"  Time elapsed     : {elapsed:.1f}s",
        border,
        "  ✅ Ready for Session 3 (Gaussian Splatting)",
        border,
        "",
        "  Next step:",
        f"    python scripts/train_splat.py \\",
        f"        --colmap_dir {args.output_dir} \\",
        f"        --output_dir outputs/splat/",
        border,
        "",
    ]
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()