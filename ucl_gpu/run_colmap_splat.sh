#!/bin/bash
# =============================================================================
# RoboScene+ — COLMAP + Nerfstudio Splatfacto on raw_images/ (511 phone photos)
# =============================================================================
#
# PURPOSE:
#   Retrain the Gaussian Splat using all 511 high-res phone photos (data/raw_images/).
#   The current splat was trained on only 54 MASt3R-SLAM keyframes → wall floaters.
#   COLMAP on 511 photos should give 200-350 usable views → 4-6× better coverage.
#
# PRE-REQUISITES (do these on Mac BEFORE running this script):
#   1. Upload raw_images/ to bluestreak:
#      scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#        ~/3D-Spatial-Reconstruction/data/raw_images/ \
#        jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/
#
#   2. SSH to bluestreak and run:
#      bash && source /opt/Python/Python-3.11.5_Setup.csh
#      source /scratch0/jrameshs/roboscene_env/bin/activate
#      cd /scratch0/jrameshs/roboscene-plus
#      nohup bash ucl_gpu/run_colmap_splat.sh > logs/colmap_splat.log 2>&1 &
#      tail -f logs/colmap_splat.log
#
# EXPECTED WALL TIME: 3-5 hours (COLMAP matching ~60min, training ~2h at 30k steps)
# OUTPUT: outputs/splat_raw_v1/scene.ply  (download this to Mac)
#
# DOWNLOAD (run on Mac after job completes):
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_raw_v1/ \
#     ~/3D-Spatial-Reconstruction/outputs/splat_raw_v1/
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — COLMAP + Splatfacto"
echo "  Starting: $(date)"
echo "========================================"

# ── Environment ────────────────────────────────────────────────────────────
source /opt/Python/Python-3.11.5_Setup.csh 2>/dev/null || true
source /scratch0/jrameshs/roboscene_env/bin/activate

export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/splat_raw_v1

echo "Working dir: $(pwd)"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')"
echo ""

# ── Install nerfstudio if needed ───────────────────────────────────────────
if ! python -c "import nerfstudio" 2>/dev/null; then
    echo "[1/5] Installing nerfstudio..."
    pip install -q nerfstudio
else
    echo "[1/5] nerfstudio already installed: $(python -c 'import nerfstudio; print(nerfstudio.__version__)')"
fi

# ── Verify input images ────────────────────────────────────────────────────
IMG_DIR="data/raw_images"
if [ ! -d "$IMG_DIR" ]; then
    echo "ERROR: $IMG_DIR not found."
    echo "Upload it first:"
    echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk ~/3D-Spatial-Reconstruction/data/raw_images/ \\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/"
    exit 1
fi

N_IMGS=$(ls "$IMG_DIR"/*.JPG "$IMG_DIR"/*.jpg 2>/dev/null | wc -l)
echo "[2/5] Found $N_IMGS images in $IMG_DIR"
echo "      Expected ~511. If 0, check upload completed."
if [ "$N_IMGS" -lt 50 ]; then
    echo "ERROR: Too few images ($N_IMGS). Expected 511."
    exit 1
fi

# ── Run ns-process-data (COLMAP SfM) ──────────────────────────────────────
COLMAP_OUT="data/colmap_raw"

if [ -f "$COLMAP_OUT/transforms.json" ]; then
    echo "[3/5] COLMAP already done ($COLMAP_OUT/transforms.json exists). Skipping."
else
    echo "[3/5] Running ns-process-data (COLMAP feature extraction + matching + SfM)..."
    echo "      This takes 30-90 minutes for 511 images."
    echo "      Started: $(date)"

    ns-process-data images \
        --data "$IMG_DIR" \
        --output-dir "$COLMAP_OUT" \
        --num-downscales 2 \
        --matching-method exhaustive \
        --verbose

    echo "      COLMAP done: $(date)"

    # Check how many cameras were registered
    if [ -f "$COLMAP_OUT/transforms.json" ]; then
        N_REG=$(python3 -c "
import json
with open('$COLMAP_OUT/transforms.json') as f:
    d = json.load(f)
print(len(d.get('frames', [])))
")
        echo "      Registered $N_REG cameras"
        if [ "$N_REG" -lt 80 ]; then
            echo "WARNING: Only $N_REG cameras registered (expected 200+)."
            echo "  This means COLMAP struggled with the images (e.g. too dark, blurry, low overlap)."
            echo "  Will continue training but quality may be limited."
        fi
    fi
fi

# ── Train splatfacto ──────────────────────────────────────────────────────
OUTPUT_PLY="outputs/splat_raw_v1/scene.ply"

if [ -f "$OUTPUT_PLY" ]; then
    echo "[4/5] scene.ply already exists. Skipping training."
else
    echo "[4/5] Starting ns-train splatfacto (30000 steps)..."
    echo "      Expected training time: 1.5-2.5 hours"
    echo "      Started: $(date)"

    ns-train splatfacto \
        --data "$COLMAP_OUT" \
        --output-dir "outputs/splat_raw_v1" \
        --max-num-iterations 30000 \
        --pipeline.model.cull-alpha-thresh 0.005 \
        --pipeline.model.densify-grad-thresh 0.0002 \
        --viewer.quit-on-train-completion True

    echo "      Training done: $(date)"

    # nerfstudio saves to a timestamped subdirectory — find and copy the .ply
    LATEST_PLY=$(find "outputs/splat_raw_v1" -name "*.ply" -newer "$COLMAP_OUT/transforms.json" \
        2>/dev/null | sort | tail -1)

    if [ -n "$LATEST_PLY" ] && [ "$LATEST_PLY" != "$OUTPUT_PLY" ]; then
        cp "$LATEST_PLY" "$OUTPUT_PLY"
        echo "      Copied $(basename $LATEST_PLY) → $OUTPUT_PLY"
    fi
fi

# ── Verify output ─────────────────────────────────────────────────────────
if [ -f "$OUTPUT_PLY" ]; then
    SIZE=$(du -h "$OUTPUT_PLY" | cut -f1)
    echo "[5/5] ✅ scene.ply created: $OUTPUT_PLY ($SIZE)"
else
    echo "[5/5] ⚠️  scene.ply not found — check logs for errors"
    # List what was actually produced
    find outputs/splat_raw_v1 -name "*.ply" 2>/dev/null && true
    exit 1
fi

echo ""
echo "========================================"
echo "  Job Complete: $(date)"
echo ""
echo "  Rendered quality check (compare rendered vs gt):"
echo "    ls outputs/splat_raw_v1/"
echo ""
echo "  Download to Mac:"
echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "      jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/splat_raw_v1/ \\"
echo "      ~/3D-Spatial-Reconstruction/outputs/splat_raw_v1/"
echo ""
echo "  Also download COLMAP poses (needed for semantic painting):"
echo "    scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "      jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/colmap_raw/ \\"
echo "      ~/3D-Spatial-Reconstruction/data/colmap_raw/"
echo "========================================"
