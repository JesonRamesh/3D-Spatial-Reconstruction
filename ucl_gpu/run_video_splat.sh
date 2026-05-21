#!/bin/bash
# =============================================================================
# RoboScene+ — Plan A: Video → COLMAP → splatfacto
# =============================================================================
#
# WHY THIS SCRIPT:
#   The previous splat used 54 MASt3R-SLAM keyframes from the video.
#   54 frames × 73° FOV = barely covers the room once → wall floaters + distortion.
#   This script extracts 316 frames (2fps) from the SAME video, giving
#   COLMAP ~6× more views and much better wall/ceiling coverage.
#
# PRE-REQUISITES (run on Mac BEFORE this script):
#   1. Upload the video to bluestreak:
#      scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#        ~/3D-Spatial-Reconstruction/data/raw/room_video.MOV \
#        jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/raw/
#
#   2. SSH to bluestreak and run:
#      bash
#      source /opt/Python/Python-3.11.5_Setup.csh
#      source /scratch0/jrameshs/roboscene_env/bin/activate
#      cd /scratch0/jrameshs/roboscene-plus
#      git pull
#      nohup bash ucl_gpu/run_video_splat.sh > logs/video_splat.log 2>&1 &
#      tail -f logs/video_splat.log
#
# EXPECTED WALL TIME: ~3 hours total
#   - Frame extraction:  < 5 min
#   - COLMAP matching:   60-90 min (exhaustive on 316 frames)
#   - splatfacto 30k:    1.5-2 h
#
# OUTPUT: outputs/splat_video_v1/scene.ply
#
# DOWNLOAD (run on Mac after job completes):
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_video_v1/ \
#     ~/3D-Spatial-Reconstruction/outputs/splat_video_v1/
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_video/ \
#     ~/3D-Spatial-Reconstruction/data/colmap_video/
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — Video COLMAP Splatfacto"
echo "  Starting: $(date)"
echo "========================================"

# ── Environment ────────────────────────────────────────────────────────────────
source /opt/Python/Python-3.11.5_Setup.csh 2>/dev/null || true
source /scratch0/jrameshs/roboscene_env/bin/activate

export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/splat_video_v1

echo "Working dir: $(pwd)"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')"
echo ""

# ── Install nerfstudio if needed ───────────────────────────────────────────────
if ! python -c "import nerfstudio" 2>/dev/null; then
    echo "[1/5] Installing nerfstudio..."
    pip install -q nerfstudio
else
    echo "[1/5] nerfstudio: $(python -c 'import nerfstudio; print(nerfstudio.__version__)')"
fi

# ── Locate video ───────────────────────────────────────────────────────────────
VIDEO="data/raw/room_video.MOV"
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: $VIDEO not found."
    echo ""
    echo "Upload it first (run on Mac):"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "    ~/3D-Spatial-Reconstruction/data/raw/room_video.MOV \\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/raw/"
    exit 1
fi

VIDEO_SIZE=$(du -h "$VIDEO" | cut -f1)
echo "[2/5] Video found: $VIDEO ($VIDEO_SIZE)"

# ── Extract frames at 2 fps ────────────────────────────────────────────────────
FRAMES_DIR="data/video_frames"

if [ -d "$FRAMES_DIR" ] && [ "$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)" -ge 200 ]; then
    N_EXISTING=$(ls "$FRAMES_DIR"/*.jpg | wc -l)
    echo "[3/5] Frames already extracted ($N_EXISTING JPGs). Skipping."
else
    echo "[3/5] Extracting frames at 2fps from video…"
    mkdir -p "$FRAMES_DIR"

    ffmpeg -y -i "$VIDEO" \
        -vf "fps=2,scale=1920:1080" \
        -q:v 2 \
        "$FRAMES_DIR/frame_%04d.jpg" \
        2>&1 | tail -5

    N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg | wc -l)
    echo "      Extracted $N_FRAMES frames."

    if [ "$N_FRAMES" -lt 100 ]; then
        echo "ERROR: Only $N_FRAMES frames extracted. Check video file."
        exit 1
    fi
fi

N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg | wc -l)
echo "      Using $N_FRAMES frames (2fps × 158s)"

# ── Run ns-process-data (COLMAP SfM) ──────────────────────────────────────────
COLMAP_OUT="data/colmap_video"

if [ -f "$COLMAP_OUT/transforms.json" ]; then
    N_REG=$(python3 -c "
import json
with open('$COLMAP_OUT/transforms.json') as f:
    d = json.load(f)
print(len(d.get('frames', [])))
" 2>/dev/null || echo "?")
    echo "[4/5] COLMAP already done — $N_REG cameras registered. Skipping."
else
    echo "[4/5] Running COLMAP SfM on $N_FRAMES video frames…"
    echo "      FOV: ~73° (main 1x lens) — ideal for COLMAP."
    echo "      Matching method: exhaustive (best for <500 images)"
    echo "      This takes 60-90 minutes."
    echo "      Started: $(date)"

    ns-process-data images \
        --data "$FRAMES_DIR" \
        --output-dir "$COLMAP_OUT" \
        --num-downscales 2 \
        --matching-method exhaustive \
        --verbose

    echo "      COLMAP done: $(date)"

    # Quality check
    if [ -f "$COLMAP_OUT/transforms.json" ]; then
        N_REG=$(python3 -c "
import json
with open('$COLMAP_OUT/transforms.json') as f:
    d = json.load(f)
print(len(d.get('frames', [])))
")
        PCT=$(python3 -c "print(round($N_REG * 100 / $N_FRAMES))")
        echo ""
        echo "  ┌─ COLMAP Quality Report ──────────────────────────┐"
        echo "  │  Input frames  : $N_FRAMES"
        echo "  │  Registered    : $N_REG  (${PCT}%)"
        if [ "$N_REG" -ge 200 ]; then
            echo "  │  Quality       : ✅ EXCELLENT  (200+ views)"
        elif [ "$N_REG" -ge 100 ]; then
            echo "  │  Quality       : ✅ GOOD  (100+ views, proceed)"
        elif [ "$N_REG" -ge 50 ]; then
            echo "  │  Quality       : ⚠️  MARGINAL  (50-100 views)"
            echo "  │  → Check for motion blur in extracted frames"
        else
            echo "  │  Quality       : ❌ POOR  (<50 views)"
            echo "  │  → STOP — investigate before wasting GPU time"
            echo "  │  → Check data/video_frames/ for blur/dark frames"
            echo "  └─────────────────────────────────────────────────┘"
            exit 1
        fi
        echo "  └─────────────────────────────────────────────────┘"
        echo ""
    else
        echo "ERROR: transforms.json not created. COLMAP failed."
        echo "  Check logs above for COLMAP error messages."
        exit 1
    fi
fi

# ── Train splatfacto ──────────────────────────────────────────────────────────
OUTPUT_PLY="outputs/splat_video_v1/scene.ply"

if [ -f "$OUTPUT_PLY" ]; then
    echo "[5/5] scene.ply already exists. Skipping training."
else
    echo "[5/5] Training splatfacto (30,000 steps)…"
    echo "      Expected: 1.5-2 hours"
    echo "      Started: $(date)"

    ns-train splatfacto \
        --data "$COLMAP_OUT" \
        --output-dir "outputs/splat_video_v1" \
        --max-num-iterations 30000 \
        --pipeline.model.cull-alpha-thresh 0.005 \
        --pipeline.model.densify-grad-thresh 0.0002 \
        --viewer.quit-on-train-completion True

    echo "      Training done: $(date)"

    # Find and copy the output PLY
    LATEST_PLY=$(find "outputs/splat_video_v1" -name "*.ply" 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST_PLY" ] && [ "$LATEST_PLY" != "$OUTPUT_PLY" ]; then
        cp "$LATEST_PLY" "$OUTPUT_PLY"
        echo "      Copied $(basename $LATEST_PLY) → $OUTPUT_PLY"
    fi
fi

# ── Verify and report ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Job Complete: $(date)"
echo "========================================"
echo ""

if [ -f "$OUTPUT_PLY" ]; then
    SIZE=$(du -h "$OUTPUT_PLY" | cut -f1)
    echo "  ✅ scene.ply: $OUTPUT_PLY ($SIZE)"
else
    echo "  ⚠️  scene.ply not found — check training log above"
    find outputs/splat_video_v1 -name "*.ply" 2>/dev/null
fi

echo ""
echo "  Download to Mac (run from Mac terminal):"
echo ""
echo "  # Splat (for viewer)"
echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/splat_video_v1/ \\"
echo "    ~/3D-Spatial-Reconstruction/outputs/splat_video_v1/"
echo ""
echo "  # COLMAP poses (for semantic reprojection)"
echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/colmap_video/ \\"
echo "    ~/3D-Spatial-Reconstruction/data/colmap_video/"
echo ""
echo "  After downloading, on Mac:"
echo "  1. python scripts/convert_to_splat.py --input outputs/splat_video_v1/scene.ply"
echo "  2. python scripts/prune_splat.py --input outputs/splat_video_v1/scene.splat"
echo "  3. python scripts/paint_semantic_gaussians.py --cameras_bin data/colmap_video/sparse/0/cameras.bin --images_bin data/colmap_video/sparse/0/images.bin"
echo "  4. python open_viewer.py"
echo "========================================"
