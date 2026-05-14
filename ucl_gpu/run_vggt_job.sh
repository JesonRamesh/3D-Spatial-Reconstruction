#!/bin/bash
# =============================================================================
# RoboScene+ — VGGT Inference Job Script (UCL GPU)
# =============================================================================
# Usage:
#   nohup bash ucl_gpu/run_vggt_job.sh > logs/vggt_job.log 2>&1 &
#
# Run this on a UCL CS GPU machine (cream/vanilla/booked workstation).
# Uses nohup so it survives SSH disconnection.
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — VGGT Inference Job"
echo "  $(date)"
echo "========================================"

# Environment setup
if [ -f /opt/Python/Python-3.11.5_Setup.csh ]; then
    echo "Setting up Python 3.11..."
    source /opt/Python/Python-3.11.5_Setup.csh
fi

# Activate venv
if [ -d ~/roboscene_env ]; then
    echo "Activating roboscene_env..."
    source ~/roboscene_env/bin/activate
fi

# Restrict to 1 GPU (etiquette on shared machines)
if [ -f /usr/local/cuda/CUDA_VISIBILITY.csh ]; then
    source /usr/local/cuda/CUDA_VISIBILITY.csh
fi

# Show GPU info
echo ""
echo "GPU Info:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Set project paths
PROJECT_DIR="${HOME}/roboscene-plus"
if [ ! -d "$PROJECT_DIR" ]; then
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Use scratch for output (fast I/O, 1TB)
SCRATCH="/scratch0/${USER}"
mkdir -p "${SCRATCH}/vggt_out"

# Run VGGT
echo ""
echo "Starting VGGT inference..."
python scripts/run_vggt.py \
    --frames_dir data/frames \
    --output_dir "${SCRATCH}/vggt_out" \
    --batch_size 30 \
    --device auto

# Copy results to home (permanent storage)
echo ""
echo "Copying results to home directory..."
mkdir -p data/vggt_out
cp -r "${SCRATCH}/vggt_out/"* data/vggt_out/

echo ""
echo "========================================"
echo "  VGGT Job Complete — $(date)"
echo "  Output: data/vggt_out/"
echo "  Remember to scp to your Mac!"
echo "========================================"