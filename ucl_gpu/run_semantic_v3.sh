#!/bin/bash
# =============================================================================
# RoboScene+ — Semantic Segmentation v3 (UCL bluestreak)
#
# Runs Grounded SAM2 on all 1282 sharp frames from frames_v3.
# Output: outputs/semantic_v3/frame_XXXX.json (per-frame mask JSONs)
#
# Must be run AFTER run_splat_v5.sh trains the splat (semantic_v3 is used
# by run_splat_v5.sh step 8 for Gaussian painting).
#
# Usage (background, SSH-disconnect safe):
#   nohup bash ucl_gpu/run_semantic_v3.sh > logs/semantic_v3.log 2>&1 &
#   tail -f logs/semantic_v3.log
#
# Expected runtime: ~3-5 hours on RTX 4070 Ti SUPER (1282 frames)
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — Semantic v3 (1282 frames)"
echo "  $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────
echo "[1/6] Setting up environment..."
if [ -d /opt/Python/Python-3.11.5/bin ]; then
    export PATH="/opt/Python/Python-3.11.5/bin:$PATH"
elif [ -d /opt/Python/Python-3.11/bin ]; then
    export PATH="/opt/Python/Python-3.11/bin:$PATH"
fi
echo "  Python: $(python3 --version)"

VENV="/scratch0/jrameshs/roboscene_env"
source "$VENV/bin/activate"
echo "  venv: $VENV"

export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# CRITICAL: Redirect ALL caches to scratch (home dir is only 10GB)
export TORCH_EXTENSIONS_DIR="/scratch0/jrameshs/torch_extensions"
export TMPDIR="/scratch0/jrameshs/tmp"
export XDG_CACHE_HOME="/scratch0/jrameshs/cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$XDG_CACHE_HOME"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/semantic_v3 outputs/semantic_v3/debug

# ── Verify inputs ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Verifying inputs..."

FRAMES_DIR="data/frames_v3"
FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.jpg 2>/dev/null | wc -l)
echo "  Frames: $FRAME_COUNT in $FRAMES_DIR"

if [ "$FRAME_COUNT" -lt 100 ]; then
    echo "ERROR: Expected ~1282 frames in $FRAMES_DIR, found $FRAME_COUNT"
    echo "       Upload from Mac:"
    echo "         scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "           ~/Downloads/3D-Spatial-Reconstruction/data/frames_v3/ \\"
    echo "           jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/"
    exit 1
fi

# ── Install deps ──────────────────────────────────────────────────────
echo ""
echo "[3/6] Checking segmentation dependencies..."

# GroundingDINO — often not pre-installed in roboscene_env
python3 -c "import groundingdino" 2>/dev/null || {
    echo "  Installing GroundingDINO..."
    pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git 2>/dev/null || true
}

# SAM2
python3 -c "import sam2" 2>/dev/null || {
    echo "  Installing SAM2..."
    pip install -q git+https://github.com/facebookresearch/sam2.git 2>/dev/null || true
}

pip install -q pycocotools huggingface_hub tqdm 2>/dev/null || true

python3 -c "import groundingdino; print('  GroundingDINO: OK')" 2>/dev/null || echo "  ⚠️ GroundingDINO may not be installed — check logs"
python3 -c "import sam2; print('  SAM2: OK')" 2>/dev/null || echo "  ⚠️ SAM2 may not be installed — check logs"

# ── Check for existing outputs (resumable) ────────────────────────────
echo ""
echo "[4/6] Checking existing outputs..."
DONE_COUNT=$(ls outputs/semantic_v3/frame_*.json 2>/dev/null | wc -l)
echo "  Already done: $DONE_COUNT / $FRAME_COUNT frames"
if [ "$DONE_COUNT" -gt 0 ]; then
    echo "  --skip_existing enabled — will resume from where we left off"
fi

# ── Run segmentation ──────────────────────────────────────────────────
echo ""
echo "[5/6] Running Grounded SAM2 on $FRAME_COUNT frames..."
echo "  Output:     outputs/semantic_v3/"
echo "  Labels:     bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor"
echo "  Confidence: 0.30 (box threshold)"
echo "  Device:     cuda"
echo "  Started at: $(date)"
echo ""

python3 scripts/run_semantic.py \
    --frames_dir  "$FRAMES_DIR" \
    --output_dir  "outputs/semantic_v3" \
    --labels      "bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor" \
    --device      cuda \
    --confidence  0.30 \
    --skip_existing

EXIT_CODE=$?

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "[6/6] ════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    JSON_COUNT=$(ls outputs/semantic_v3/frame_*.json 2>/dev/null | wc -l)
    echo "  Semantic v3 COMPLETE — $(date)"
    echo "  JSON files: $JSON_COUNT / $FRAME_COUNT frames"
    echo ""
    echo "  Next step: run_splat_v5.sh step 8 (Gaussian painting) will"
    echo "  automatically pick up outputs/semantic_v3/ once it exists"
    echo "  with >100 JSON files."
    echo ""
    echo "  Or paint manually:"
    echo "    python3 scripts/paint_semantic_gaussians.py \\"
    echo "      --splat_ply    outputs/splat_v5/scene_aligned.ply \\"
    echo "      --semantic_dir outputs/semantic_v3 \\"
    echo "      --cameras_bin  data/vggt_out_v3/sparse/cameras.bin \\"
    echo "      --images_bin   data/vggt_out_v3/sparse/images.bin \\"
    echo "      --output_ply   outputs/splat_v5/scene_semantic.ply"
    echo ""
    echo "  Download to Mac:"
    echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "      jrameshs@bluestreak.cs.ucl.ac.uk:$(pwd)/outputs/semantic_v3/ \\"
    echo "      ~/Downloads/3D-Spatial-Reconstruction/outputs/"
else
    echo "  ✗ ERROR: run_semantic.py exited with code $EXIT_CODE"
    echo "    Check logs/semantic_v3.log for details"
fi
echo "  ════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
