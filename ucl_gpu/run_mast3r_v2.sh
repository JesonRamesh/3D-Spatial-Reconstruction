#!/bin/bash
# =============================================================================
# RoboScene+ — MASt3R-SLAM v2 + Splat v6 Pipeline (UCL bluestreak)
#
# PURPOSE:
#   Run MASt3R-SLAM on 1282 sharp frames from room_video_v2.MOV (data/frames_v3).
#   MASt3R-SLAM is globally consistent with no batch stitching — the root cause
#   of the distorted point cloud from VGGT world_points.
#
#   Produces:
#     outputs/mast3r_out_v2/dense_pointcloud.ply  — dense per-pixel 3D points
#     data/mast3r_out_v2/sparse/0/                — COLMAP format for splat
#     outputs/splat_v6/scene_aligned.splat         — retrained Gaussian Splat
#
# USAGE (SSH-disconnect safe background run):
#   nohup bash ucl_gpu/run_mast3r_v2.sh > logs/mast3r_v2.log 2>&1 &
#   tail -f logs/mast3r_v2.log
#
# PRE-REQUISITES (do on Mac before running):
#   1. Upload frames_v3 if not already on bluestreak:
#      scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#        ~/Downloads/3D-Spatial-Reconstruction/data/frames_v3/ \
#        jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/
#
#   2. SSH + bash + activate env:
#      bash
#      source /opt/Python/Python-3.11.5_Setup.csh
#      source /scratch0/jrameshs/roboscene_env/bin/activate
#
# EXPECTED RUNTIME: ~2-3 hours (SLAM ~40min, training ~2h)
# SCRATCH USAGE: ~15GB (MASt3R-SLAM repo + checkpoints + outputs)
#
# DOWNLOAD OUTPUTS (run on Mac after job completes):
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#     "jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/mast3r_out_v2/dense_pointcloud.ply" \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/mast3r_out_v2/
#
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_v6/ \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v6/
#
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/mast3r_out_v2/ \
#     ~/Downloads/3D-Spatial-Reconstruction/data/mast3r_out_v2/
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — MASt3R-SLAM v2 + Splat v6"
echo "  $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────────
echo "[1/8] Setting up environment..."

if [ -d /opt/Python/Python-3.11.5/bin ]; then
    export PATH="/opt/Python/Python-3.11.5/bin:$PATH"
elif [ -d /opt/Python/Python-3.11/bin ]; then
    export PATH="/opt/Python/Python-3.11/bin:$PATH"
fi
echo "  Python: $(python3 --version)"

VENV="/scratch0/jrameshs/roboscene_env"
if [ ! -d "$VENV" ]; then
    echo "  ERROR: venv not found at $VENV. Rebuild it first."
    echo "    python3 -m venv $VENV"
    echo "    source $VENV/bin/activate"
    echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi
source "$VENV/bin/activate"
echo "  venv: $VENV ✓"

export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"
export TORCH_EXTENSIONS_DIR="/scratch0/jrameshs/torch_extensions"
export TMPDIR="/scratch0/jrameshs/tmp"
export XDG_CACHE_HOME="/scratch0/jrameshs/cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$XDG_CACHE_HOME"

echo ""
echo "  GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
python3 -c "import torch; print(f'  CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU\"}')"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "  ERROR: Project not found at $PROJECT_DIR"
    echo "    git clone https://github.com/JesonRamesh/3D-Spatial-Reconstruction.git $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"
mkdir -p logs outputs/mast3r_out_v2 data/mast3r_out_v2

FRAMES_DIR="data/frames_v3"
FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo ""
echo "  frames_v3: $FRAME_COUNT frames"
if [ "$FRAME_COUNT" -lt 100 ]; then
    echo "  ERROR: Expected ~1282 frames. Upload from Mac:"
    echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "      ~/Downloads/3D-Spatial-Reconstruction/data/frames_v3/ \\"
    echo "      jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/"
    exit 1
fi

# ── Install MASt3R-SLAM ───────────────────────────────────────────────────
MAST3R_DIR="/scratch0/jrameshs/MASt3R-SLAM"
echo ""
echo "[2/8] Checking MASt3R-SLAM installation..."

if [ ! -d "$MAST3R_DIR" ]; then
    echo "  Not found — cloning MASt3R-SLAM..."
    cd /scratch0/jrameshs
    git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git
    cd "$MAST3R_DIR"
    echo "  Installing Python dependencies..."
    pip install -e . --no-build-isolation -q
    echo "  ✓ MASt3R-SLAM installed"
else
    echo "  Found at $MAST3R_DIR"
    # Ensure it's installed in venv
    cd "$MAST3R_DIR"
    pip install -e . --no-build-isolation -q 2>/dev/null || true
fi

# Download checkpoints if missing
CKPT_DIR="$MAST3R_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

MAST3R_CKPT="$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ ! -f "$MAST3R_CKPT" ]; then
    echo "  Downloading MASt3R checkpoint (~750MB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
    filename='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
    cache_dir='$CKPT_DIR/hf_cache'
)
shutil.copy(path, '$MAST3R_CKPT')
print('  ✓ MASt3R checkpoint downloaded')
" || {
    echo "  Trying wget fallback..."
    wget -q -O "$MAST3R_CKPT" \
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
        || echo "  WARNING: Could not download MASt3R checkpoint — SLAM may fail"
}
else
    echo "  MASt3R checkpoint: ✓"
fi

cd "$PROJECT_DIR"

# ── Run MASt3R-SLAM ───────────────────────────────────────────────────────
SLAM_WORKSPACE="/scratch0/jrameshs/mast3r_slam_v2_workspace"
echo ""
echo "[3/8] Running MASt3R-SLAM on frames_v3..."
echo "  Input:     $PROJECT_DIR/$FRAMES_DIR ($FRAME_COUNT frames)"
echo "  Workspace: $SLAM_WORKSPACE"
echo "  Started:   $(date)"
echo ""

mkdir -p "$SLAM_WORKSPACE"

# MASt3R-SLAM uses Python API or demo.py. Use Python API for programmatic control.
python3 - <<'PYEOF'
import sys, os, shutil, json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/scratch0/jrameshs/MASt3R-SLAM')

project_dir   = '/scratch0/jrameshs/roboscene-plus'
frames_dir    = f'{project_dir}/data/frames_v3'
workspace_dir = '/scratch0/jrameshs/mast3r_slam_v2_workspace'
output_dir    = f'{project_dir}/outputs/mast3r_out_v2'
colmap_dir    = f'{project_dir}/data/mast3r_out_v2'

os.makedirs(workspace_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{colmap_dir}/sparse/0', exist_ok=True)

frames = sorted(Path(frames_dir).glob('frame_*.jpg'))
print(f'  Found {len(frames)} frames')

# Try to import and run MASt3R-SLAM
try:
    from mast3r_slam.dataloader import ImageFolderDataset
    from mast3r_slam.slam import MASt3rSLAM

    dataset = ImageFolderDataset(frames_dir)
    slam = MASt3rSLAM(
        checkpoint='/scratch0/jrameshs/MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth',
        workspace=workspace_dir,
        device='cuda'
    )
    slam.run(dataset)
    slam.save(workspace_dir)
    print('  ✓ MASt3R-SLAM complete (Python API)')

except ImportError as e:
    print(f'  Python API not available ({e}) — using demo.py subprocess')
    import subprocess
    demo_script = '/scratch0/jrameshs/MASt3R-SLAM/demo.py'
    result = subprocess.run([
        'python3', demo_script,
        '--dataset', 'ImageFolder', frames_dir,
        '--output_dir', workspace_dir,
        '--device', 'cuda',
        '--silent'
    ], cwd='/scratch0/jrameshs/MASt3R-SLAM', check=True)
    print('  ✓ MASt3R-SLAM complete (demo.py)')

except Exception as e:
    print(f'  ERROR running MASt3R-SLAM: {e}')
    raise

print(f'  Workspace contents:')
for p in sorted(Path(workspace_dir).iterdir()):
    size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    print(f'    {p.name:<40} {size/1e6:.1f} MB')

PYEOF

echo ""
echo "  MASt3R-SLAM finished: $(date)"

# ── Extract dense PLY + COLMAP from SLAM workspace ────────────────────────
echo ""
echo "[4/8] Extracting dense PLY and COLMAP format..."

python3 - <<'PYEOF'
import sys, os, json, shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, '/scratch0/jrameshs/MASt3R-SLAM')
sys.path.insert(0, '/scratch0/jrameshs/roboscene-plus')

workspace_dir = Path('/scratch0/jrameshs/mast3r_slam_v2_workspace')
output_dir    = Path('/scratch0/jrameshs/roboscene-plus/outputs/mast3r_out_v2')
colmap_dir    = Path('/scratch0/jrameshs/roboscene-plus/data/mast3r_out_v2')
frames_dir    = Path('/scratch0/jrameshs/roboscene-plus/data/frames_v3')

colmap_sparse = colmap_dir / 'sparse' / '0'
colmap_sparse.mkdir(parents=True, exist_ok=True)
(colmap_dir / 'images').mkdir(parents=True, exist_ok=True)

# ── Find dense PLY in workspace ──────────────────────────────────────────
ply_candidates = list(workspace_dir.rglob('*.ply'))
print(f'  Found PLY files in workspace: {[str(p) for p in ply_candidates]}')

dense_ply = None
for name in ['global_point_cloud.ply', 'pointcloud.ply', 'reconstruction.ply', 'map.ply']:
    candidate = workspace_dir / name
    if candidate.exists():
        dense_ply = candidate
        break
if dense_ply is None and ply_candidates:
    # Take the largest PLY (likely the dense reconstruction)
    dense_ply = max(ply_candidates, key=lambda p: p.stat().st_size)

if dense_ply and dense_ply.exists():
    dst = output_dir / 'dense_pointcloud.ply'
    shutil.copy(dense_ply, dst)
    print(f'  ✓ Dense PLY: {dst} ({dst.stat().st_size/1e6:.1f} MB)')
else:
    print('  WARNING: No dense PLY found in workspace')
    print('  Workspace contents:')
    for p in workspace_dir.rglob('*'):
        print(f'    {p}')

# ── Load camera poses from workspace ─────────────────────────────────────
# MASt3R-SLAM saves poses in various formats depending on version
poses = {}

# Try cameras.json
for pose_file in ['cameras.json', 'trajectory.json', 'poses.json']:
    pf = workspace_dir / pose_file
    if pf.exists():
        with open(pf) as f:
            data = json.load(f)
        print(f'  Found pose file: {pose_file} ({len(data)} entries)')
        poses = data
        break

# Try npz/npy
if not poses:
    for npz_file in workspace_dir.rglob('*.npz'):
        data = np.load(npz_file)
        print(f'  NPZ keys in {npz_file.name}: {list(data.keys())}')

# Try the MASt3R-SLAM Python save format
if not poses:
    try:
        from mast3r_slam.config import load_config
        slam_state_file = workspace_dir / 'slam_state.pkl'
        if slam_state_file.exists():
            import pickle
            with open(slam_state_file, 'rb') as f:
                state = pickle.load(f)
            print(f'  SLAM state keys: {list(state.keys()) if isinstance(state, dict) else type(state)}')
    except Exception as e:
        print(f'  Could not load slam state: {e}')

# ── Build COLMAP-format output ────────────────────────────────────────────
# Use scripts/colmap_utils.py to write binary COLMAP files
from scripts.colmap_utils import write_cameras_binary, write_images_binary, write_points3d_binary

# Get image dimensions from first frame
from PIL import Image
sample_img = next(frames_dir.glob('frame_*.jpg'))
with Image.open(sample_img) as img:
    W, H = img.size
print(f'  Image dimensions: {W}×{H}')

# Default intrinsics — assume ~70° horizontal FOV for phone camera
fx = W * 1.2   # reasonable approximation; SLAM may have better estimates
fy = fx
cx = W / 2.0
cy = H / 2.0

frames = sorted(frames_dir.glob('frame_*.jpg'))

# If we have pose data from SLAM, use it; otherwise write a minimal stub
# so the downstream splat training can work
if poses:
    print(f'  Building COLMAP from {len(poses)} SLAM poses...')
    cameras = {1: {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [fx, fy, cx, cy]}}

    images_data = {}
    for i, (name, pose) in enumerate(poses.items(), 1):
        # pose is cam_to_world 4x4 — invert to world_to_cam
        T = np.array(pose['cam_to_world']) if isinstance(pose, dict) and 'cam_to_world' in pose else np.array(pose)
        T_inv = np.linalg.inv(T)
        R = T_inv[:3, :3]
        t = T_inv[:3, 3]
        # Quaternion from R
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()  # xyzw
        qw, qx, qy, qz = q[3], q[0], q[1], q[2]

        img_name = Path(name).name if '/' in str(name) else f'frame_{i:04d}.jpg'
        images_data[i] = {
            'id': i, 'qvec': [qw, qx, qy, qz], 'tvec': t.tolist(),
            'camera_id': 1, 'name': img_name, 'xys': [], 'point3D_ids': []
        }
        # Copy keyframe image
        src = frames_dir / img_name
        if src.exists():
            shutil.copy(src, colmap_dir / 'images' / img_name)

    write_cameras_binary(cameras, str(colmap_sparse / 'cameras.bin'))
    write_images_binary(images_data, str(colmap_sparse / 'images.bin'))
    write_points3d_binary({}, str(colmap_sparse / 'points3D.bin'))
    print(f'  ✓ COLMAP sparse written: {colmap_sparse}')

    # Save camera_poses.json (same format as VGGT output, for compatibility)
    cam_poses_out = {}
    for name, pose in poses.items():
        img_name = Path(name).name if '/' in str(name) else name
        cam_poses_out[img_name] = {
            'cam_to_world_4x4': pose['cam_to_world'] if isinstance(pose, dict) and 'cam_to_world' in pose else pose,
            'intrinsic_3x3': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        }
    with open(output_dir / 'camera_poses.json', 'w') as f:
        json.dump(cam_poses_out, f, indent=2)
    print(f'  ✓ camera_poses.json: {len(cam_poses_out)} cameras')

else:
    print('  WARNING: No pose data extracted from SLAM workspace.')
    print('  Check workspace manually and re-run conversion if needed.')
    print('  Workspace files:')
    for p in workspace_dir.iterdir():
        print(f'    {p.name}')

PYEOF

echo ""
echo "  Extraction done: $(date)"
echo "  Dense PLY: $(du -h outputs/mast3r_out_v2/dense_pointcloud.ply 2>/dev/null || echo 'NOT FOUND')"
echo "  COLMAP:    $(ls data/mast3r_out_v2/sparse/0/ 2>/dev/null)"

# ── Install gsplat deps ───────────────────────────────────────────────────
echo ""
echo "[5/8] Checking gsplat and splat training dependencies..."
pip install -q gsplat 2>/dev/null || true
pip install -q tyro viser "nerfview==0.0.2" "torchmetrics[image]" tensorboard \
    imageio "numpy<2.0.0" scikit-learn tqdm opencv-python Pillow pyyaml scipy 2>/dev/null || true
python3 -c "import gsplat; print('  gsplat', gsplat.__version__)"

# ── Train Splat v6 ────────────────────────────────────────────────────────
OUT_DIR="outputs/splat_v6"
COLMAP_INPUT="data/mast3r_out_v2"
IMAGES_INPUT="data/mast3r_out_v2/images"
mkdir -p "$OUT_DIR"

echo ""
echo "[6/8] Training Splat v6 on MASt3R-SLAM v2 output (30K steps)..."

# Check if COLMAP is usable
if [ ! -f "$COLMAP_INPUT/sparse/0/cameras.bin" ]; then
    echo "  ERROR: COLMAP not found at $COLMAP_INPUT/sparse/0/"
    echo "  MASt3R-SLAM conversion may have failed. Check Step 4 logs."
    echo "  Skipping splat training."
else
    KEYFRAME_COUNT=$(ls "$IMAGES_INPUT"/*.jpg 2>/dev/null | wc -l)
    echo "  COLMAP dir:  $COLMAP_INPUT/sparse/0/ ✓"
    echo "  Keyframes:   $KEYFRAME_COUNT"
    echo "  Output dir:  $OUT_DIR"
    echo "  Started:     $(date)"

    python3 scripts/train_splat.py \
        --colmap_dir "$COLMAP_INPUT" \
        --output_dir "$OUT_DIR" \
        --frames_dir "$IMAGES_INPUT" \
        --iterations 30000 \
        --opacity_reg 0.0 \
        --scale_reg 0.0 \
        --prune_opa 0.002

    echo "  Training done: $(date)"
fi

# ── Export PLY from checkpoint if needed ─────────────────────────────────
echo ""
echo "[7/8] Exporting PLY..."
if [ ! -f "$OUT_DIR/scene.ply" ]; then
    CKPT=$(ls "$OUT_DIR/ckpts/"ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -1)
    if [ -n "$CKPT" ]; then
        echo "  Exporting from checkpoint: $CKPT"
        python3 scripts/export_splat_ply.py --ckpt "$CKPT" --output "$OUT_DIR/scene.ply"
    else
        echo "  No checkpoint found — training may have failed."
    fi
else
    echo "  scene.ply already exists"
fi

if [ -f "$OUT_DIR/scene.ply" ]; then
    echo "  scene.ply: $(du -h $OUT_DIR/scene.ply | cut -f1)"

    echo ""
    echo "  Aligning Y-up..."
    python3 scripts/realign_splat_v4.py \
        --input_ply  "$OUT_DIR/scene.ply" \
        --output_ply "$OUT_DIR/scene_aligned.ply"

    echo "  Pruning floaters..."
    python3 scripts/prune_floaters.py \
        --input_ply  "$OUT_DIR/scene_aligned.ply" \
        --output_ply "$OUT_DIR/scene_pruned.ply"

    echo "  Converting to .splat..."
    python3 scripts/convert_to_splat.py \
        --input  "$OUT_DIR/scene_aligned.ply" \
        --output "$OUT_DIR/scene_aligned.splat"

    python3 scripts/convert_to_splat.py \
        --input  "$OUT_DIR/scene_pruned.ply" \
        --output "$OUT_DIR/scene_pruned.splat"
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  MASt3R-SLAM v2 + Splat v6 COMPLETE"
echo "  $(date)"
echo ""
echo "  Key outputs:"
du -h outputs/mast3r_out_v2/dense_pointcloud.ply 2>/dev/null && true
ls -lh "$OUT_DIR"/*.ply "$OUT_DIR"/*.splat 2>/dev/null | awk '{print "  "$NF, $5}' && true
echo ""
echo "  Download to Mac (run on Mac terminal):"
echo ""
echo "  # Dense point cloud (for viewer):"
echo "  mkdir -p ~/Downloads/3D-Spatial-Reconstruction/outputs/mast3r_out_v2"
echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/outputs/mast3r_out_v2/dense_pointcloud.ply \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/mast3r_out_v2/"
echo ""
echo "  # Splat v6:"
echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/outputs/splat_v6/ \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v6/"
echo ""
echo "  # COLMAP (for semantic painting):"
echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/data/mast3r_out_v2/ \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/data/mast3r_out_v2/"
echo "========================================"
