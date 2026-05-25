#!/bin/bash
# =============================================================================
# RoboScene+ — Full Splat v5 Pipeline (UCL bluestreak)
#
# Trains a new Gaussian Splat on the 1282 sharp frames from frames_v3,
# using COLMAP produced by VGGT v3. Includes floater regularizers.
# After training: prunes → aligns Y-up → converts to .splat → semantic paint.
#
# Usage (background, SSH-disconnect safe):
#   nohup bash ucl_gpu/run_splat_v5.sh > logs/splat_v5.log 2>&1 &
#   tail -f logs/splat_v5.log
#
# Expected runtime: ~3-4 hours on RTX 4070 Ti SUPER
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — Splat v5 Full Pipeline"
echo "  $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────
echo "[1/8] Setting up environment..."
if [ -d /opt/Python/Python-3.11.5/bin ]; then
    export PATH="/opt/Python/Python-3.11.5/bin:$PATH"
elif [ -d /opt/Python/Python-3.11/bin ]; then
    export PATH="/opt/Python/Python-3.11/bin:$PATH"
fi
echo "  Python: $(python3 --version)"

VENV="/scratch0/jrameshs/roboscene_env"
source "$VENV/bin/activate"
echo "  venv: $VENV"

export PYTHONPATH="/scratch0/jrameshs/vggt:$PYTHONPATH"
export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"

# CRITICAL: Redirect ALL caches to scratch (home dir is only 10GB)
export TORCH_EXTENSIONS_DIR="/scratch0/jrameshs/torch_extensions"
export TMPDIR="/scratch0/jrameshs/tmp"
export XDG_CACHE_HOME="/scratch0/jrameshs/cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$XDG_CACHE_HOME"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/splat_v5

# ── Verify inputs ─────────────────────────────────────────────────────
echo ""
echo "[2/8] Verifying inputs..."

COLMAP_DIR="data/vggt_out_v3"
FRAMES_DIR="data/frames_v3"
OUT_DIR="outputs/splat_v5"

if [ ! -f "$COLMAP_DIR/sparse/cameras.bin" ]; then
    echo "ERROR: COLMAP not found at $COLMAP_DIR/sparse/"
    echo "       Run run_vggt_v3.sh first, then re-run this script."
    exit 1
fi

FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "  COLMAP: $COLMAP_DIR/sparse/ ✓"
echo "  Frames: $FRAME_COUNT frames in $FRAMES_DIR"

# ── Install gsplat deps ────────────────────────────────────────────────
echo ""
echo "[3/8] Checking gsplat dependencies..."
pip install -q gsplat 2>/dev/null || true
pip install -q tyro viser "nerfview==0.0.2" "torchmetrics[image]" tensorboard \
    imageio "numpy<2.0.0" scikit-learn tqdm opencv-python Pillow pyyaml scipy 2>/dev/null || true
python3 -c "import gsplat; print('  gsplat', gsplat.__version__)"

# ── Train splat v5 ────────────────────────────────────────────────────
echo ""
echo "[4/8] Training Gaussian Splat v5 (30K steps, floater regularizers)..."
echo "  COLMAP dir:  $COLMAP_DIR"
echo "  Output dir:  $OUT_DIR"
echo "  opacity_reg: 0.01  scale_reg: 0.001  prune_opa: 0.005"
echo ""

python3 scripts/train_splat.py \
    --colmap_dir "$COLMAP_DIR" \
    --output_dir "$OUT_DIR" \
    --frames_dir "$FRAMES_DIR" \
    --iterations 30000 \
    --opacity_reg 0.01 \
    --scale_reg   0.001 \
    --prune_opa   0.005

if [ ! -f "$OUT_DIR/scene.ply" ]; then
    echo "  scene.ply not found — exporting from checkpoint..."
    CKPT=$(ls "$OUT_DIR/ckpts/"ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint in $OUT_DIR/ckpts/ — training failed."
        exit 1
    fi
    echo "  Checkpoint: $CKPT"
    python3 scripts/export_splat_ply.py --ckpt "$CKPT" --output "$OUT_DIR/scene.ply"
fi

if [ ! -f "$OUT_DIR/scene.ply" ]; then
    echo "ERROR: scene.ply still missing after export. Check logs."
    exit 1
fi
echo "  ✓ scene.ply: $(du -h $OUT_DIR/scene.ply | cut -f1)"

# ── Prune floaters ────────────────────────────────────────────────────
echo ""
echo "[5/8] Aligning Z-up → Y-up (needed before pruning to get correct floor)..."
python3 scripts/realign_splat_v4.py \
    --input_ply  "$OUT_DIR/scene.ply" \
    --output_ply "$OUT_DIR/scene_aligned.ply"
echo "  ✓ scene_aligned.ply: $(du -h $OUT_DIR/scene_aligned.ply | cut -f1)"

echo ""
echo "[6/8] Pruning floaters..."
python3 scripts/prune_floaters.py \
    --input_ply  "$OUT_DIR/scene_aligned.ply" \
    --output_ply "$OUT_DIR/scene_pruned.ply"
echo "  ✓ scene_pruned.ply: $(du -h $OUT_DIR/scene_pruned.ply | cut -f1)"

# ── Convert to .splat ─────────────────────────────────────────────────
echo ""
echo "[7/8] Converting to .splat format..."
python3 scripts/convert_to_splat.py \
    --input  "$OUT_DIR/scene_aligned.ply" \
    --output "$OUT_DIR/scene_aligned.splat"
echo "  ✓ scene_aligned.splat: $(du -h $OUT_DIR/scene_aligned.splat | cut -f1)"

python3 scripts/convert_to_splat.py \
    --input  "$OUT_DIR/scene_pruned.ply" \
    --output "$OUT_DIR/scene_pruned.splat"
echo "  ✓ scene_pruned.splat: $(du -h $OUT_DIR/scene_pruned.splat | cut -f1)"

# ── Semantic painting (only if semantic_v3 exists) ────────────────────
echo ""
echo "[8/8] Semantic painting..."
SEMANTIC_DIR="outputs/semantic_v3"
if [ -d "$SEMANTIC_DIR" ] && [ "$(ls $SEMANTIC_DIR/*.json 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "  Found semantic_v3 — painting Gaussians..."
    python3 scripts/paint_semantic_gaussians.py \
        --splat_ply    "$OUT_DIR/scene_aligned.ply" \
        --semantic_dir "$SEMANTIC_DIR" \
        --cameras_bin  "data/mast3r_out/sparse/0/cameras.bin" \
        --images_bin   "data/mast3r_out/sparse/0/images.bin" \
        --output_ply   "$OUT_DIR/scene_semantic.ply"
    python3 scripts/convert_to_splat.py \
        --input  "$OUT_DIR/scene_semantic.ply" \
        --output "$OUT_DIR/scene_semantic.splat"
    echo "  ✓ scene_semantic.splat done"
else
    echo "  semantic_v3 not ready — skipping (run run_semantic_v3.sh first)"
    echo "  Re-run step 8 manually once semantic_v3 is available."
fi

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Splat v5 pipeline complete — $(date)"
echo "  Output files:"
ls -lh "$OUT_DIR"/*.ply "$OUT_DIR"/*.splat 2>/dev/null | awk '{print "    "$NF, $5}'
echo ""
echo "  Download to Mac (run on Mac terminal):"
echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "      jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/$OUT_DIR/ \\"
echo "      ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v5/"
echo "========================================"
