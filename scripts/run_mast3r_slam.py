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
    parser.add_argument(
        "--skip_frames",
        action="store_true",
        default=False,
        help="Skip frame extraction; assume images already exist at output_dir/images/.",
    )
    parser.add_argument(
        "--skip_slam",
        action="store_true",
        default=False,
        help="Skip SLAM; assume slam output already exists at output_dir/slam_output/.",
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
        "--dataset", str(video_path),
        "--config",  str(config_path),
        "--save-as", str(slam_out),
        "--no-viz",
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


def _read_pose_txt(
    txt_path: Path,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Parse a MASt3R-SLAM trajectory text file.

    Each non-comment, non-empty line has the format::

        timestamp  tx ty tz  qx qy qz qw

    Returns a list of ``(timestamp, t_c2w, q_xyzw)`` tuples sorted by
    timestamp, where *t_c2w* is a ``(3,)`` float64 translation vector and
    *q_xyzw* is a ``(4,)`` float64 quaternion in ``[qx, qy, qz, qw]`` order
    as expected by ``scipy.spatial.transform.Rotation.from_quat``.
    """
    entries: List[Tuple[float, np.ndarray, np.ndarray]] = []
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts     = float(parts[0])
            t_c2w  = np.array([float(parts[1]),
                               float(parts[2]),
                               float(parts[3])], dtype=np.float64)
            q_xyzw = np.array([float(parts[4]),
                               float(parts[5]),
                               float(parts[6]),
                               float(parts[7])], dtype=np.float64)
            entries.append((ts, t_c2w, q_xyzw))
    entries.sort(key=lambda e: e[0])
    return entries


def _read_image_size(png_path: Path) -> Tuple[int, int]:
    """Return ``(width, height)`` of a PNG by reading the IHDR chunk.

    Works without Pillow/cv2.  Falls back to ``(1920, 1080)`` on any error.
    """
    try:
        with open(png_path, "rb") as fh:
            fh.read(8)           # PNG signature
            fh.read(4)           # IHDR chunk length
            fh.read(4)           # b'IHDR'
            w = int.from_bytes(fh.read(4), "big")
            h = int.from_bytes(fh.read(4), "big")
        return w, h
    except Exception:
        return 1920, 1080


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


# ---------------------------------------------------------------------------
# convert_to_colmap — main function
# ---------------------------------------------------------------------------


def convert_to_colmap(slam_output_dir: Path, output_dir: Path) -> None:
    """Convert MASt3R-SLAM output into a COLMAP-compatible sparse model.

    Expected MASt3R-SLAM output layout inside *slam_output_dir*::

        slam_output/
        ├── room_video.txt                 — trajectory (one line per keyframe)
        │                                    ``timestamp tx ty tz qx qy qz qw``
        ├── room_video.ply                 — dense point cloud
        └── keyframes/
            └── room_video/
                └── <timestamp>.png        — keyframe images

    Produces::

        output_dir/
        ├── images/
        │   ├── frame_0001.jpg             — keyframes renamed + converted
        │   └── …
        └── sparse/
            └── 0/
                ├── cameras.bin            — single shared PINHOLE camera
                ├── images.bin             — one entry per keyframe
                └── points3D.bin           — full dense point cloud

    Args:
        slam_output_dir: Path returned by :func:`run_slam`
            (``output_dir/slam_output/``).
        output_dir: Root output directory; ``images/`` and ``sparse/0/``
            are written here.

    Raises:
        FileNotFoundError: If *slam_output_dir* does not exist or no
            trajectory ``.txt`` is found.
        RuntimeError: If no keyframe PNGs are found.
    """
    from scipy.spatial.transform import Rotation
    import importlib.util as _ilu
    import shutil

    slam_output_dir = Path(slam_output_dir).resolve()
    output_dir      = Path(output_dir).resolve()

    if not slam_output_dir.exists():
        raise FileNotFoundError(
            f"SLAM output directory not found: {slam_output_dir}"
        )

    # ------------------------------------------------------------------ #
    # Load colmap_utils via importlib (works regardless of sys.path)     #
    # ------------------------------------------------------------------ #
    _scripts_dir = Path(__file__).resolve().parent
    _spec = _ilu.spec_from_file_location(
        "colmap_utils", _scripts_dir / "colmap_utils.py"
    )
    _cu = _ilu.module_from_spec(_spec)   # type: ignore[arg-type]
    _spec.loader.exec_module(_cu)        # type: ignore[union-attr]

    write_cameras_binary  = _cu.write_cameras_binary
    write_images_binary   = _cu.write_images_binary
    write_points3D_binary = _cu.write_points3D_binary
    rotmat_to_qvec        = _cu.rotmat_to_qvec

    # ------------------------------------------------------------------ #
    # 1. Locate the trajectory .txt file                                  #
    # ------------------------------------------------------------------ #
    txt_candidates = sorted(slam_output_dir.glob("*.txt"))
    if not txt_candidates:
        raise FileNotFoundError(
            f"No trajectory .txt found in {slam_output_dir}\n"
            f"  Expected: slam_output/<video_stem>.txt"
        )
    txt_path   = txt_candidates[0]       # typically only one .txt
    video_stem = txt_path.stem           # e.g. "room_video"
    logger.info("Trajectory : %s", txt_path.name)

    pose_entries = _read_pose_txt(txt_path)
    if not pose_entries:
        raise RuntimeError(f"Trajectory file is empty or unparseable: {txt_path}")
    logger.info("Poses      : %d keyframes", len(pose_entries))

    # Build a timestamp-string → (t_c2w, q_xyzw) lookup for exact matching;
    # also keep a sorted float array for nearest-neighbour fallback.
    ts_to_pose: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        str(ts): (t, q) for ts, t, q in pose_entries
    }
    ts_sorted = np.array([e[0] for e in pose_entries], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 2. Locate keyframe PNGs                                             #
    # ------------------------------------------------------------------ #
    # Primary layout: keyframes/<video_stem>/<timestamp>.png
    kf_dir = slam_output_dir / "keyframes" / video_stem
    if not kf_dir.exists():
        # Fallback: first sub-directory under keyframes/
        kf_root      = slam_output_dir / "keyframes"
        kf_subdirs   = [p for p in kf_root.iterdir() if p.is_dir()] if kf_root.exists() else []
        kf_dir       = kf_subdirs[0] if kf_subdirs else kf_root

    kf_pngs = sorted(kf_dir.glob("*.png"), key=lambda p: float(p.stem))
    if not kf_pngs:
        raise RuntimeError(
            f"No keyframe PNGs found in {kf_dir}\n"
            f"  Expected: keyframes/{video_stem}/<timestamp>.png"
        )
    logger.info("Keyframes  : %d PNGs in %s", len(kf_pngs), kf_dir)

    # ------------------------------------------------------------------ #
    # 3. Read image size from the first keyframe PNG                     #
    # ------------------------------------------------------------------ #
    img_w, img_h = _read_image_size(kf_pngs[0])
    logger.info("Image size : %d × %d", img_w, img_h)

    # iPhone 1× lens prior: fx = fy = max(w, h) * 1.2
    fx = fy = float(max(img_w, img_h)) * 1.2
    cx = img_w / 2.0
    cy = img_h / 2.0
    logger.info("Intrinsics : fx=fy=%.1f  cx=%.1f  cy=%.1f", fx, cx, cy)

    # ------------------------------------------------------------------ #
    # 4. Copy + rename keyframe PNGs → output_dir/images/frame_NNNN.jpg #
    # ------------------------------------------------------------------ #
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Match each PNG to its pose (exact timestamp string, then nearest-neighbour)
    matched: List[Tuple[Path, np.ndarray, np.ndarray]] = []
    for png in kf_pngs:
        ts_str = png.stem
        if ts_str in ts_to_pose:
            t_c2w, q_xyzw = ts_to_pose[ts_str]
        else:
            ts_val        = float(ts_str)
            nn_idx        = int(np.argmin(np.abs(ts_sorted - ts_val)))
            _, t_c2w, q_xyzw = pose_entries[nn_idx]
            logger.debug("Timestamp %s → nearest %.6f", ts_str, pose_entries[nn_idx][0])
        matched.append((png, t_c2w, q_xyzw))

    for i, (png, _, _) in enumerate(matched):
        dst = images_out / f"frame_{i + 1:04d}.jpg"
        try:
            from PIL import Image as _PIL
            _PIL.open(png).convert("RGB").save(dst, quality=95)
        except ImportError:
            # No Pillow: copy PNG bytes with .jpg extension (still readable
            # by most loaders; file is actually PNG data)
            shutil.copy2(png, dst)
    logger.info("Copied %d keyframes → %s", len(matched), images_out)

    # ------------------------------------------------------------------ #
    # 5. cameras.bin — one shared PINHOLE camera per frame               #
    # ------------------------------------------------------------------ #
    n_frames      = len(matched)
    camera_params = np.array([fx, fy, cx, cy], dtype=np.float64)
    cameras = [
        {
            "camera_id": i + 1,
            "model":     "PINHOLE",
            "width":     img_w,
            "height":    img_h,
            "params":    camera_params,
        }
        for i in range(n_frames)
    ]

    # ------------------------------------------------------------------ #
    # 6. images.bin — c2w (t, q) → w2c (R, t) via scipy Rotation        #
    # ------------------------------------------------------------------ #
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    images_colmap = []
    for idx, (_, t_c2w, q_xyzw) in enumerate(matched):
        # scipy convention: [qx, qy, qz, qw]
        R_c2w = Rotation.from_quat(q_xyzw).as_matrix()  # (3, 3)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        qvec  = rotmat_to_qvec(R_w2c)                    # [qw, qx, qy, qz]
        images_colmap.append({
            "image_id":    idx + 1,
            "qvec":        qvec,
            "tvec":        t_w2c,
            "camera_id":   idx + 1,
            "name":        f"frame_{idx + 1:04d}.jpg",
            "point2D_xys": np.zeros((0, 2), dtype=np.float64),
            "point3D_ids": np.array([], dtype=np.int64),
        })

    # ------------------------------------------------------------------ #
    # 7. points3D.bin — load .ply with open3d                            #
    # ------------------------------------------------------------------ #
    ply_candidates = sorted(slam_output_dir.glob("*.ply"))
    xyz = np.zeros((0, 3), dtype=np.float64)
    rgb = np.zeros((0, 3), dtype=np.uint8)
    if ply_candidates:
        ply_path = ply_candidates[0]
        logger.info("Point cloud: %s", ply_path.name)
        xyz, rgb = _read_ply_points(ply_path)
    else:
        logger.warning("No .ply found in %s — points3D.bin will be empty",
                       slam_output_dir)

    points3d = [
        {
            "point3D_id": i + 1,
            "xyz":        xyz[i],
            "rgb":        rgb[i],
            "error":      1.0,
            "track":      [],   # empty tracks — identical to VGGT pipeline
        }
        for i in range(len(xyz))
    ]

    # ------------------------------------------------------------------ #
    # 8. Write COLMAP binary files via colmap_utils                      #
    # ------------------------------------------------------------------ #
    write_cameras_binary(cameras,        sparse_dir / "cameras.bin")
    write_images_binary(images_colmap,   sparse_dir / "images.bin")
    write_points3D_binary(points3d,      sparse_dir / "points3D.bin")

    logger.info("cameras.bin  : %d cameras written",  len(cameras))
    logger.info("images.bin   : %d images written",   len(images_colmap))
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
    if args.skip_frames:
        images_dir = args.output_dir / "images"
        frame_paths = sorted(str(p) for p in images_dir.glob("frame_*.jpg"))
        logger.info(
            "Skipping frame extraction — using existing images at %s (%d frames)",
            images_dir, len(frame_paths),
        )
    else:
        logger.info("Step 1/3 — Extracting frames at %.1f fps …", args.fps)
        frame_paths = extract_frames(args.video_path, args.output_dir, args.fps)
        logger.info("Extracted %d frames.", len(frame_paths))

    # ------------------------------------------------------------------
    # Step 2 – run SLAM
    # ------------------------------------------------------------------
    slam_output_dir: Path = Path(args.output_dir) / "slam_output"
    if args.skip_slam:
        logger.info(
            "Skipping SLAM — using existing output at %s",
            slam_output_dir,
        )
    else:
        logger.info("Step 2/3 — Running MASt3R-SLAM on device '%s' …", args.device)
        slam_output_dir = run_slam(
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