#!/bin/bash
# =============================================================================
# RoboScene+ — Gaussian Splatting Training Job Script (UCL GPU)
# =============================================================================
# Usage:
#   nohup bash ucl_gpu/run_splat_job.sh > logs/splat_train.log 2>&1 &
#   tail -f logs/splat_train.log
#
# Run this on a UCL CS GPU machine (bluestreak / cream / vanilla).
# Uses nohup so it survives SSH disconnection.
#
# Prerequisites:
#   - VGGT output exists at data/vggt_out/sparse/ (Session 2 complete)
#   - gsplat installed: pip install gsplat==1.3.0
#   - roboscene_env activated on /scratch0
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — Gaussian Splatting Training"
echo "  $(date)"
echo "========================================"

# ── Environment setup ──────────────────────────────────────────────────

# Switch to bash-compatible Python 3.11
if [ -f /opt/Python/Python-3.11.5_Setup.csh ]; then
    echo "Setting up Python 3.11..."
    source /opt/Python/Python-3.11.5_Setup.csh
fi

# Activate the roboscene venv on scratch
VENV_PATH="/scratch0/jrameshs/roboscene_env"
if [ -d "$VENV_PATH" ]; then
    echo "Activating roboscene_env from scratch..."
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: venv not found at $VENV_PATH"
    echo "Create it first: python3 -m venv $VENV_PATH"
    exit 1
fi

# Set PYTHONPATH for VGGT (needed by gsplat data loader if it uses VGGT utils)
export PYTHONPATH="/scratch0/jrameshs/vggt:$PYTHONPATH"

# pip cache on scratch (don't fill home)
export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"

# HuggingFace cache on scratch
export HF_HOME="/scratch0/jrameshs/hf_cache"

# Restrict to 1 GPU (etiquette on shared machines)
if [ -f /usr/local/cuda/CUDA_VISIBILITY.csh ]; then
    source /usr/local/cuda/CUDA_VISIBILITY.csh
fi

# ── Show GPU info ──────────────────────────────────────────────────────

echo ""
echo "GPU Info:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo ""

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# ── Set project paths ─────────────────────────────────────────────────

# Try scratch working copy first, fall back to script location
PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
if [ ! -d "$PROJECT_DIR" ]; then
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Ensure log directory exists
mkdir -p logs

# ── Verify VGGT output exists ─────────────────────────────────────────

COLMAP_DIR="data/vggt_out"
if [ ! -f "$COLMAP_DIR/sparse/cameras.bin" ]; then
    echo "ERROR: COLMAP files not found at $COLMAP_DIR/sparse/"
    echo "Run VGGT first (Session 2): nohup bash ucl_gpu/run_vggt_job.sh > logs/vggt_job.log 2>&1 &"
    exit 1
fi
echo "COLMAP sparse files found at $COLMAP_DIR/sparse/"

# ── Check gsplat is installed ─────────────────────────────────────────

if ! python -c "import gsplat" 2>/dev/null; then
    echo "gsplat not installed. Installing gsplat..."
    pip install gsplat
fi
echo "gsplat version: $(python -c 'import gsplat; print(gsplat.__version__)' 2>/dev/null || echo 'unknown')"

# Install extra deps needed by gsplat's example simple_trainer.py
echo "Checking simple_trainer dependencies..."
pip install -q tyro viser "nerfview==0.0.2" "torchmetrics[image]" tensorboard \
    imageio "numpy<2.0.0" scikit-learn tqdm opencv-python Pillow pyyaml scipy 2>/dev/null
echo ""

# ── Run Gaussian Splatting training ────────────────────────────────────

OUTPUT_DIR="outputs/splat"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting Gaussian Splatting training..."
echo "  COLMAP dir:  $COLMAP_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Iterations:  15000"
echo ""

python scripts/train_splat.py \
    --colmap_dir "$COLMAP_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --frames_dir "data/frames" \
    --iterations 15000

# ── Verify output ─────────────────────────────────────────────────────

echo ""
if [ -f "$OUTPUT_DIR/scene.ply" ]; then
    PLY_SIZE=$(du -h "$OUTPUT_DIR/scene.ply" | cut -f1)
    echo "✅ scene.ply created: $OUTPUT_DIR/scene.ply ($PLY_SIZE)"
else
    echo "⚠️  scene.ply not found at $OUTPUT_DIR/scene.ply"
    echo "   Check training log for errors."
fi

echo ""
echo "========================================"
echo "  Gaussian Splatting Job Complete — $(date)"
echo "  Output: $OUTPUT_DIR/"
echo ""
echo "  Download to Mac (run on Mac terminal):"
echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "      jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/$OUTPUT_DIR/ \\"
echo "      ~/3D-Spatial-Reconstruction/outputs/splat/"
echo ""
echo "  Next: Session 4 (Grounded SAM2 Semantics)"
echo "========================================"