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

# ── Run COLMAP SfM directly (bypasses nerfstudio wrapper bug) ────────────────
COLMAP_OUT="data/colmap_video"
COLMAP_DB="$COLMAP_OUT/colmap.db"
COLMAP_SPARSE="$COLMAP_OUT/sparse"

# Make sure COLMAP from conda env is on PATH
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH

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

    mkdir -p "$COLMAP_SPARSE/0"

    # Step 1: Feature extraction
    echo "      [COLMAP 1/3] Extracting features..."
    colmap feature_extractor \
        --database_path "$COLMAP_DB" \
        --image_path "$FRAMES_DIR" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --SiftExtraction.use_gpu 0

    # Step 2: Exhaustive matching
    echo "      [COLMAP 2/3] Matching features (exhaustive)..."
    colmap exhaustive_matcher \
        --database_path "$COLMAP_DB" \
        --SiftMatching.use_gpu 0

    # Step 3: Sparse reconstruction
    echo "      [COLMAP 3/3] Running bundle adjustment..."
    colmap mapper \
        --database_path "$COLMAP_DB" \
        --image_path "$FRAMES_DIR" \
        --output_path "$COLMAP_SPARSE"

    echo "      COLMAP done: $(date)"

    # Convert COLMAP output to nerfstudio transforms.json using pycolmap
    echo "      Converting to nerfstudio format..."
    python3 -c "
import json, struct, numpy as np
from pathlib import Path
import pycolmap

sparse_dir = Path('$COLMAP_SPARSE/0')
recon = pycolmap.Reconstruction(str(sparse_dir))

cameras = recon.cameras
images = recon.images

frames = []
for img_id, img in images.items():
    cam = cameras[img.camera_id]
    # Get rotation and translation
    R = img.rotation_matrix()
    t = np.array(img.tvec)
    # Build c2w matrix (nerfstudio convention)
    c2w = np.eye(4)
    c2w[:3,:3] = R.T
    c2w[:3, 3] = -R.T @ t
    # Flip axes to match nerfstudio convention
    c2w[:,1] *= -1
    c2w[:,2] *= -1
    frames.append({
        'file_path': 'images/' + img.name,
        'transform_matrix': c2w.tolist()
    })

# Get intrinsics from first camera
cam = list(cameras.values())[0]
params = cam.params
transforms = {
    'fl_x': float(params[0]),
    'fl_y': float(params[0]),
    'cx': float(params[1]),
    'cy': float(params[2]),
    'w': cam.width,
    'h': cam.height,
    'camera_model': 'OPENCV',
    'frames': frames
}

out = Path('$COLMAP_OUT/transforms.json')
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(transforms, f, indent=2)
print(f'Wrote {len(frames)} frames to {out}')
"

    # Copy images dir into colmap_out for nerfstudio
    if [ ! -d "$COLMAP_OUT/images" ]; then
        ln -s "$(pwd)/$FRAMES_DIR" "$COLMAP_OUT/images"
    fi

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

# ── Generate point cloud PLY from COLMAP for initialisation ───────────────────
POINTS_PLY="$COLMAP_SPARSE/0/points3D.ply"

if [ -f "$POINTS_PLY" ]; then
    echo "[4b/5] Point cloud PLY already exists. Skipping."
else
    echo "[4b/5] Converting COLMAP points to PLY for Gaussian initialisation..."
    python3 -c "
import pycolmap, numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

recon = pycolmap.Reconstruction('$COLMAP_SPARSE/0')
pts = np.array([[p.xyz[0], p.xyz[1], p.xyz[2]] for p in recon.points3D.values()])
rgb = np.array([[int(p.color[0]), int(p.color[1]), int(p.color[2])] for p in recon.points3D.values()])
vertex = np.zeros(len(pts), dtype=[
    ('x','f4'),('y','f4'),('z','f4'),
    ('red','u1'),('green','u1'),('blue','u1')
])
vertex['x'],vertex['y'],vertex['z'] = pts[:,0],pts[:,1],pts[:,2]
vertex['red'],vertex['green'],vertex['blue'] = rgb[:,0],rgb[:,1],rgb[:,2]
PlyData([PlyElement.describe(vertex,'vertex')]).write('$POINTS_PLY')
print(f'Saved {len(pts)} seed points to $POINTS_PLY')
"
fi

# ── Train splatfacto v2 (with point cloud init + scale regularisation) ────────
OUTPUT_DIR="outputs/splat_video_v2"
OUTPUT_PLY="$OUTPUT_DIR/scene.ply"

if [ -f "$OUTPUT_PLY" ]; then
    echo "[5/5] scene.ply already exists. Skipping training."
else
    echo "[5/5] Training splatfacto v2 (30,000 steps, point cloud init)…"
    echo "      Key improvements over v1:"
    echo "        - Initialised from 64K COLMAP points (not random)"
    echo "        - Scale regularisation stops Gaussians growing outside room"
    echo "        - Tighter cull threshold removes floaters during training"
    echo "      Expected: 1.5-2 hours"
    echo "      Started: $(date)"

    ns-train splatfacto \
        --data "$COLMAP_OUT" \
        --output-dir "$OUTPUT_DIR" \
        --max-num-iterations 30000 \
        --pipeline.model.cull-alpha-thresh 0.005 \
        --pipeline.model.densify-grad-thresh 0.0002 \
        --pipeline.model.use-scale-regularization True \
        --pipeline.model.max-gauss-ratio 10.0 \
        --viewer.quit-on-train-completion True

    echo "      Training done: $(date)"

    # Find and copy the output PLY
    LATEST_PLY=$(find "$OUTPUT_DIR" -name "*.ply" 2>/dev/null | grep -v points3D | sort | tail -1)
    if [ -n "$LATEST_PLY" ]; then
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
    find outputs/splat_video_v2 -name "*.ply" 2>/dev/null
fi

echo ""
echo "  Download to Mac (run from Mac terminal):"
echo ""
echo "  # Splat v2 (cleaner, point-cloud initialised)"
echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/splat_video_v2/scene.ply \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_video_v2/scene.ply"
echo ""
echo "  # COLMAP sparse (for semantic reprojection)"
echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/colmap_video/sparse/ \\"
echo "    ~/Downloads/3D-Spatial-Reconstruction/data/colmap_video/sparse/"
echo ""
echo "  After downloading, on Mac:"
echo "  1. python scripts/convert_to_splat.py --input outputs/splat_video_v2/scene.ply --output outputs/splat_video_v2/scene.splat"
echo "  2. python scripts/prune_splat.py --input outputs/splat_video_v2/scene.splat --output outputs/splat_video_v2/scene_pruned.splat --alpha_min 80 --crop -3.5,-1.5,-3.0,3.5,1.5,3.5"
echo "  3. python open_viewer.py"
echo "========================================"
