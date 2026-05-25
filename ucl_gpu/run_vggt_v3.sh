#!/bin/bash
# =============================================================================
# RoboScene+ — VGGT v3 Inference Job Script (UCL bluestreak)
# =============================================================================
# Run on bluestreak after SSH + bash + activating env.
#
# Usage (foreground — watch output):
#   bash ucl_gpu/run_vggt_v3.sh
#
# Usage (background — SSH-disconnect safe):
#   nohup bash ucl_gpu/run_vggt_v3.sh > logs/vggt_v3.log 2>&1 &
#   tail -f logs/vggt_v3.log
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — VGGT v3 (1282 sharp frames)"
echo "  $(date)"
echo "========================================"

# ── Environment setup ──────────────────────────────────────────────────
echo "[1/5] Setting up environment..."

# Python 3.11 — the .csh script doesn't work in bash, set PATH directly
if [ -d /opt/Python/Python-3.11.5/bin ]; then
    export PATH="/opt/Python/Python-3.11.5/bin:$PATH"
    echo "  ✓ Added Python 3.11.5 to PATH"
elif [ -d /opt/Python/Python-3.11/bin ]; then
    export PATH="/opt/Python/Python-3.11/bin:$PATH"
    echo "  ✓ Added Python 3.11 to PATH"
else
    echo "  ⚠️  /opt/Python/Python-3.11* not found — using system python3"
fi
echo "  Python: $(python3 --version)"

# Activate venv on scratch
VENV="/scratch0/jrameshs/roboscene_env"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
    echo "  ✓ Activated venv: $VENV"
else
    echo "  ✗ ERROR: venv not found at $VENV"
    echo "    Rebuild with:"
    echo "      python3 -m venv $VENV"
    echo "      source $VENV/bin/activate"
    echo "      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo "      pip install huggingface_hub trimesh pillow numpy tqdm pyyaml"
    echo "      pip install git+https://github.com/facebookresearch/vggt.git"
    exit 1
fi

# pip cache on scratch (don't fill 10GB home)
export PIP_CACHE_DIR=/scratch0/jrameshs/pip_cache
export HF_HOME=/scratch0/jrameshs/hf_cache

# ── GPU check ──────────────────────────────────────────────────────────
echo ""
echo "[2/5] GPU check:"
nvidia-smi 2>/dev/null || echo "  ⚠️  nvidia-smi not available"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
echo ""

# ── Project directory ──────────────────────────────────────────────────
PROJECT="/scratch0/jrameshs/roboscene-plus"
if [ ! -d "$PROJECT" ]; then
    echo "  ✗ ERROR: Project not found at $PROJECT"
    echo "    Clone with:"
    echo "      unset SSH_ASKPASS && unset DISPLAY"
    echo "      git clone https://github.com/JesonRamesh/3D-Spatial-Reconstruction.git $PROJECT"
    exit 1
fi

cd "$PROJECT"
echo "[3/5] Working directory: $(pwd)"

# ── Verify frames ─────────────────────────────────────────────────────
FRAMES_DIR="data/frames_v3"
FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "  Frames in $FRAMES_DIR: $FRAME_COUNT"

if [ "$FRAME_COUNT" -lt 100 ]; then
    echo "  ✗ ERROR: Expected ~1282 frames, found $FRAME_COUNT"
    echo "    Upload from Mac with:"
    echo "      scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "        ~/3D-Spatial-Reconstruction/data/frames_v3/ \\"
    echo "        jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/"
    exit 1
fi

# ── Create output + log dirs ──────────────────────────────────────────
mkdir -p data/vggt_out_v3
mkdir -p logs

# ── Run VGGT ──────────────────────────────────────────────────────────
echo ""
echo "[4/5] Starting VGGT inference..."
echo "  Frames: $FRAME_COUNT"
echo "  Output: data/vggt_out_v3/"
echo "  Batch size: 25"
echo "  Overlap: 5 frames"
echo "  --save_world_points: YES (dense PLY)"
echo ""
echo "  Estimated time: ~15-25 min for 1282 frames on RTX 4070 Ti"
echo "  Started at: $(date)"
echo ""

python scripts/run_vggt.py \
    --frames_dir data/frames_v3 \
    --output_dir data/vggt_out_v3 \
    --save_world_points \
    --batch_size 25 \
    --overlap 5 \
    --device auto

echo ""
echo "[5/5] ════════════════════════════════════════════════════════════"
echo "  VGGT v3 COMPLETE — $(date)"
echo "  Output: data/vggt_out_v3/"
echo ""
echo "  Key files:"
echo "    data/vggt_out_v3/dense_pointcloud.ply  ← DOWNLOAD THIS"
echo "    data/vggt_out_v3/camera_poses.json"
echo "    data/vggt_out_v3/sparse/"
echo "    data/vggt_out_v3/depths/"
echo ""
echo "  Download PLY to Mac:"
echo "    scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "      jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/vggt_out_v3/dense_pointcloud.ply \\"
echo "      ~/Downloads/3D-Spatial-Reconstruction/data/vggt_out_v3/"
echo "  ════════════════════════════════════════════════════════════════"