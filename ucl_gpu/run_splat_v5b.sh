#!/bin/bash
# =============================================================================
# RoboScene+ — Splat v5b (no regularizers, better Gaussian count)
# Same as v5 but with opacity_reg=0 and scale_reg=0 to allow full densification.
#
# Usage:
#   nohup bash ucl_gpu/run_splat_v5b.sh > logs/splat_v5b.log 2>&1 &
#   tail -f logs/splat_v5b.log
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — Splat v5b (no reg)"
echo "  $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────
if [ -d /opt/Python/Python-3.11.5/bin ]; then
    export PATH="/opt/Python/Python-3.11.5/bin:$PATH"
elif [ -d /opt/Python/Python-3.11/bin ]; then
    export PATH="/opt/Python/Python-3.11/bin:$PATH"
fi

VENV="/scratch0/jrameshs/roboscene_env"
source "$VENV/bin/activate"

export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/splat_v5b

COLMAP_DIR="data/vggt_out_v3"
FRAMES_DIR="data/frames_v3"
OUT_DIR="outputs/splat_v5b"

# ── Verify inputs ─────────────────────────────────────────────────────
echo "[1/6] Verifying inputs..."
if [ ! -f "$COLMAP_DIR/sparse/cameras.bin" ]; then
    echo "ERROR: COLMAP not found at $COLMAP_DIR/sparse/"
    exit 1
fi
FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "  COLMAP: $COLMAP_DIR/sparse/ OK"
echo "  Frames: $FRAME_COUNT"

# ── Train ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Training splat v5b (30K steps, no regularizers)..."
python3 scripts/train_splat.py \
    --colmap_dir "$COLMAP_DIR" \
    --output_dir "$OUT_DIR" \
    --frames_dir "$FRAMES_DIR" \
    --iterations 30000 \
    --opacity_reg 0.0 \
    --scale_reg 0.0 \
    --prune_opa 0.002

# ── Export PLY from checkpoint if needed ──────────────────────────────
if [ ! -f "$OUT_DIR/scene.ply" ]; then
    echo "[3/6] Exporting PLY from checkpoint..."
    CKPT=$(ls "$OUT_DIR/ckpts/"ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -1)
    python3 scripts/export_splat_ply.py --ckpt "$CKPT" --output "$OUT_DIR/scene.ply"
else
    echo "[3/6] scene.ply already exists"
fi
echo "  scene.ply: $(du -h $OUT_DIR/scene.ply | cut -f1)"

# ── Align ─────────────────────────────────────────────────────────────
echo ""
echo "[4/6] Aligning Y-up..."
python3 scripts/realign_splat_v4.py \
    --input_ply  "$OUT_DIR/scene.ply" \
    --output_ply "$OUT_DIR/scene_aligned.ply"
echo "  scene_aligned.ply: $(du -h $OUT_DIR/scene_aligned.ply | cut -f1)"

# ── Prune ─────────────────────────────────────────────────────────────
echo ""
echo "[5/6] Pruning floaters..."
python3 scripts/prune_floaters.py \
    --input_ply  "$OUT_DIR/scene_aligned.ply" \
    --output_ply "$OUT_DIR/scene_pruned.ply"
echo "  scene_pruned.ply: $(du -h $OUT_DIR/scene_pruned.ply | cut -f1)"

# ── Convert ───────────────────────────────────────────────────────────
echo ""
echo "[6/6] Converting to .splat..."
python3 scripts/convert_to_splat.py \
    --input  "$OUT_DIR/scene_aligned.ply" \
    --output "$OUT_DIR/scene_aligned.splat"
echo "  scene_aligned.splat: $(du -h $OUT_DIR/scene_aligned.splat | cut -f1)"

python3 scripts/convert_to_splat.py \
    --input  "$OUT_DIR/scene_pruned.ply" \
    --output "$OUT_DIR/scene_pruned.splat"
echo "  scene_pruned.splat: $(du -h $OUT_DIR/scene_pruned.splat | cut -f1)"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Splat v5b complete — $(date)"
ls -lh "$OUT_DIR"/*.ply "$OUT_DIR"/*.splat 2>/dev/null | awk '{print "  "$NF, $5}'
echo ""
echo "  Download to Mac:"
echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "      jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/$OUT_DIR/ \\"
echo "      ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v5b/"
echo "========================================"
