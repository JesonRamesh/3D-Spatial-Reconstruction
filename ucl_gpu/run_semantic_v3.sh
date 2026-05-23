#!/bin/bash
# =============================================================================
# RoboScene+ — Semantic segmentation on video_frames_v2 using Grounded SAM2
# =============================================================================
#
# Runs Grounded DINO + SAM2 on the v2 video frames (5-digit names)
# then paints the splat_v3 Gaussians with semantic labels using colmap_v3 poses.
#
# PRE-REQUISITES:
#   - data/video_frames_v2/ must exist (frame_00001.jpg ... frame_00640.jpg)
#   - data/colmap_v3/sparse/0/ must exist (cameras.bin, images.bin)
#   - outputs/splat_v3/scene.ply must exist
#
# USAGE (on bluestreak):
#   bash
#   source /opt/Python/Python-3.11.5_Setup.csh
#   source /scratch0/jrameshs/roboscene_env/bin/activate
#   cd /scratch0/jrameshs/roboscene-plus
#   git pull
#   mkdir -p logs
#   nohup bash ucl_gpu/run_semantic_v3.sh > logs/semantic_v3.log 2>&1 &
#   tail -f logs/semantic_v3.log
#
# OUTPUTS:
#   outputs/semantic_v3/frame_00001.json ... (5-digit, one per frame)
#   outputs/scene_semantic_v3.ply           (painted Gaussians)
#   outputs/objects_3d_v3.json              (3D object centroids)
#   outputs/scene_graph_v3.json             (scene graph)
#
# DOWNLOAD (run on Mac after complete):
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/semantic_v3/ \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/semantic_v3/
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/scene_semantic_v3.ply \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/scene_semantic_v3.ply
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/objects_3d_v3.json \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/objects_3d_v3.json
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/scene_graph_v3.json \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/scene_graph_v3.json
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — Semantic Pipeline v3"
echo "  Starting: $(date)"
echo "========================================"

# ── Environment ────────────────────────────────────────────────────────────
source /opt/Python/Python-3.11.5_Setup.csh 2>/dev/null || true
source /scratch0/jrameshs/roboscene_env/bin/activate
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
export HF_HOME="/scratch0/jrameshs/hf_cache"
export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export PYTORCH_ENABLE_MPS_FALLBACK=1

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/semantic_v3

echo "Working dir: $(pwd)"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')"
echo ""

# ── Check prerequisites ────────────────────────────────────────────────────
FRAMES_DIR="data/video_frames_v2"
COLMAP_DIR="data/colmap_v3/sparse/0"
SPLAT_PLY="outputs/splat_v3/scene.ply"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "ERROR: $FRAMES_DIR not found. Run run_video_splat_v3.sh first."
    exit 1
fi
if [ ! -f "$COLMAP_DIR/cameras.bin" ]; then
    echo "ERROR: $COLMAP_DIR/cameras.bin not found."
    exit 1
fi
if [ ! -f "$SPLAT_PLY" ]; then
    echo "ERROR: $SPLAT_PLY not found. Run run_video_splat_v3.sh first."
    exit 1
fi

N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
echo "[CHECK] Frames: $N_FRAMES in $FRAMES_DIR"
echo "[CHECK] COLMAP: $COLMAP_DIR"
echo "[CHECK] Splat:  $SPLAT_PLY"
echo ""

# ── Install dependencies ───────────────────────────────────────────────────
echo "[1/4] Checking dependencies..."
python3 -c "import groundingdino" 2>/dev/null || {
    echo "      Installing GroundingDINO..."
    pip install -q groundingdino-py 2>/dev/null || \
    pip install -q git+https://github.com/IDEA-Research/GroundingDINO.git
}
python3 -c "import sam2" 2>/dev/null || {
    echo "      Installing SAM2..."
    pip install -q git+https://github.com/facebookresearch/sam2.git
}
python3 -c "import pycocotools" 2>/dev/null || pip install -q pycocotools
echo "      Dependencies OK"

# ── Step 1: Run Grounded SAM2 on v2 frames ────────────────────────────────
SEMAN_OUT="outputs/semantic_v3"
N_DONE=$(ls "$SEMAN_OUT"/*.json 2>/dev/null | grep -v debug | wc -l)

if [ "$N_DONE" -ge 400 ]; then
    echo "[2/4] Semantic JSONs already exist ($N_DONE files). Skipping SAM2."
else
    echo "[2/4] Running Grounded SAM2 on $N_FRAMES frames..."
    echo "      Output dir: $SEMAN_OUT"
    echo "      Expected: ~2-3 hours on GPU"
    echo "      Started: $(date)"

    python3 scripts/run_semantic.py \
        --frames_dir   "$FRAMES_DIR" \
        --output_dir   "$SEMAN_OUT" \
        --labels       "bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor" \
        --device       cuda \
        --weights_dir  /scratch0/jrameshs/gdino_weights \
        --debug_frames 5

    N_DONE=$(ls "$SEMAN_OUT"/*.json 2>/dev/null | grep -v debug | wc -l)
    echo "      Done: $N_DONE semantic JSONs created"
    echo "      Finished: $(date)"
fi

# ── Step 2: Paint Gaussians with semantic labels ──────────────────────────
SEMAN_PLY="outputs/scene_semantic_v3.ply"

if [ -f "$SEMAN_PLY" ]; then
    echo "[3/4] $SEMAN_PLY already exists. Skipping paint step."
else
    echo "[3/4] Painting Gaussians with semantic labels..."
    echo "      Using colmap_v3 poses + semantic_v3 JSONs"
    echo "      Started: $(date)"

    python3 scripts/paint_semantic_gaussians.py \
        --splat_ply      "$SPLAT_PLY" \
        --semantic_dir   "$SEMAN_OUT" \
        --cameras_bin    "$COLMAP_DIR/cameras.bin" \
        --images_bin     "$COLMAP_DIR/images.bin" \
        --output_ply     "$SEMAN_PLY" \
        --conf_threshold 0.20 \
        --min_votes      2

    echo "      Done: $SEMAN_PLY"
    echo "      Finished: $(date)"
fi

# ── Step 3: Lift semantics to 3D object centroids ─────────────────────────
OBJECTS_JSON="outputs/objects_3d_v3.json"

if [ -f "$OBJECTS_JSON" ]; then
    echo "[4/4] $OBJECTS_JSON already exists. Skipping lift step."
else
    echo "[4/4] Lifting semantics to 3D object centroids..."

    python3 scripts/lift_semantics_3d.py \
        --semantic_dir   "$SEMAN_OUT" \
        --cameras_bin    "$COLMAP_DIR/cameras.bin" \
        --images_bin     "$COLMAP_DIR/images.bin" \
        --output_dir     outputs/ \
        --output_suffix  "_v3"

    echo "      Done: $OBJECTS_JSON"
fi

# ── Final report ───────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Semantic Pipeline v3 Complete"
echo "  $(date)"
echo "========================================"
echo ""
echo "  Outputs:"
[ -d "$SEMAN_OUT" ]   && echo "  ✅ $SEMAN_OUT ($(ls $SEMAN_OUT/*.json 2>/dev/null | grep -v debug | wc -l) JSONs)"
[ -f "$SEMAN_PLY" ]   && echo "  ✅ $SEMAN_PLY ($(du -h $SEMAN_PLY | cut -f1))"
[ -f "$OBJECTS_JSON" ] && echo "  ✅ $OBJECTS_JSON"
echo ""
echo "  Download to Mac:"
echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/semantic_v3/ \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/semantic_v3/"
echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/scene_semantic_v3.ply \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/scene_semantic_v3.ply"
echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/objects_3d_v3.json \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/objects_3d_v3.json"
echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/scene_graph_v3.json \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/scene_graph_v3.json"
echo "========================================"