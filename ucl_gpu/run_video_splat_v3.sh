#!/bin/bash
# =============================================================================
# RoboScene+ — v3: room_video_v2.MOV → COLMAP v3 → splatfacto v3
# =============================================================================
#
# WHY v3:
#   v2 used room_video.MOV (158s, 304/317 frames registered).
#   v3 uses room_video_v2.MOV (320s, better coverage: floor, corners, objects).
#   Key training improvements:
#     - densify-grad-thresh 0.0001  (was 0.0002) → 2× more aggressive densification
#     - densify-until-iter 50000    (was ~15000)  → keeps densifying longer
#     - 60K steps                  (was 40K)     → more convergence time
#   Expected: ~480 frames, 400+ registered, near-complete room coverage.
#
# PRE-REQUISITES (run on Mac BEFORE this script):
#   1. Upload the new video to bluestreak:
#      scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#        ~/Downloads/3D-Spatial-Reconstruction/data/raw/room_video_v2.MOV \
#        jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/raw/room_video_v2.MOV
#
#   2. SSH to bluestreak and run:
#      bash
#      source /opt/Python/Python-3.11.5_Setup.csh
#      export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
#      source /scratch0/jrameshs/roboscene_env/bin/activate
#      cd /scratch0/jrameshs/roboscene-plus
#      git pull
#      mkdir -p logs
#      nohup bash ucl_gpu/run_video_splat_v3.sh > logs/splat_v3.log 2>&1 &
#      tail -f logs/splat_v3.log
#
# EXPECTED WALL TIME: ~4-5 hours total
#   - Frame extraction:  < 5 min
#   - COLMAP matching:   90-120 min (exhaustive on ~480 frames)
#   - splatfacto 60k:    2.5-3 h
#
# OUTPUTS:
#   outputs/splat_v3/scene.ply          ← main output (raw PLY)
#   data/colmap_v3/transforms.json      ← camera poses for semantics
#   data/colmap_v3/sparse/0/            ← COLMAP binaries
#
# DOWNLOAD (run on Mac after job completes):
#   mkdir -p ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v3
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_v3/scene.ply \
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v3/scene.ply
#
#   mkdir -p ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/sparse/0
#   scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
#     "jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_v3/sparse/0/" \
#     ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/sparse/0/
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_v3/transforms.json \
#     ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/transforms.json
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — v3 Pipeline"
echo "  Video: room_video_v2.MOV"
echo "  Starting: $(date)"
echo "========================================"

# ── Environment ────────────────────────────────────────────────────────────────
source /opt/Python/Python-3.11.5_Setup.csh 2>/dev/null || true
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
source /scratch0/jrameshs/roboscene_env/bin/activate

export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"
mkdir -p logs outputs/splat_v3 data/colmap_v3/sparse/0

echo "Working dir: $(pwd)"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')"
echo ""

# ── Check nerfstudio ───────────────────────────────────────────────────────────
if ! python -c "import nerfstudio" 2>/dev/null; then
    echo "[1/6] Installing nerfstudio..."
    pip install -q nerfstudio
else
    echo "[1/6] nerfstudio: $(python -c 'import nerfstudio; print(nerfstudio.__version__)')"
fi

# ── Locate video ───────────────────────────────────────────────────────────────
VIDEO="data/raw/room_video_v2.MOV"
if [ ! -f "$VIDEO" ]; then
    echo "ERROR: $VIDEO not found on bluestreak."
    echo ""
    echo "Upload it first (run on Mac):"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "    ~/Downloads/3D-Spatial-Reconstruction/data/raw/room_video_v2.MOV \\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/raw/room_video_v2.MOV"
    exit 1
fi

VIDEO_SIZE=$(du -h "$VIDEO" | cut -f1)
echo "[2/6] Video found: $VIDEO ($VIDEO_SIZE)"

# ── Extract frames at 2 fps ────────────────────────────────────────────────────
FRAMES_DIR="data/video_frames_v2"

if [ -d "$FRAMES_DIR" ] && [ "$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)" -ge 200 ]; then
    N_EXISTING=$(ls "$FRAMES_DIR"/*.jpg | wc -l)
    echo "[3/6] Frames already extracted ($N_EXISTING JPGs in $FRAMES_DIR). Skipping."
else
    echo "[3/6] Extracting frames at 2fps from room_video_v2.MOV..."
    mkdir -p "$FRAMES_DIR"

    ffmpeg -y -i "$VIDEO" \
        -vf "fps=2,scale=1920:1080" \
        -q:v 2 \
        "$FRAMES_DIR/frame_%05d.jpg" \
        2>&1 | tail -5

    N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg | wc -l)
    echo "      Extracted $N_FRAMES frames."

    if [ "$N_FRAMES" -lt 100 ]; then
        echo "ERROR: Only $N_FRAMES frames extracted. Check video file."
        exit 1
    fi
fi

N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg | wc -l)
echo "      Using $N_FRAMES frames (2fps from 320s video)"

# ── Run COLMAP SfM ─────────────────────────────────────────────────────────────
COLMAP_OUT="data/colmap_v3"
COLMAP_DB="$COLMAP_OUT/colmap.db"
COLMAP_SPARSE="$COLMAP_OUT/sparse"

if [ -f "$COLMAP_OUT/transforms.json" ]; then
    N_REG=$(python3 -c "
import json
with open('$COLMAP_OUT/transforms.json') as f:
    d = json.load(f)
print(len(d.get('frames', [])))
" 2>/dev/null || echo "?")
    echo "[4/6] COLMAP already done — $N_REG cameras registered. Skipping."
else
    echo "[4/6] Running COLMAP SfM on $N_FRAMES video frames..."
    echo "      Method: exhaustive matching (best for <600 frames)"
    echo "      Started: $(date)"

    # Wipe any stale DB from a previous failed run
    rm -f "$COLMAP_DB"
    mkdir -p "$COLMAP_SPARSE/0"

    # Step 1: Feature extraction
    echo "      [COLMAP 1/3] Extracting SIFT features..."
    colmap feature_extractor \
        --database_path "$COLMAP_DB" \
        --image_path "$FRAMES_DIR" \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model SIMPLE_RADIAL \
        --SiftExtraction.use_gpu 0

    # Step 2: Sequential matching (correct for video — finds neighbours by filename order)
    echo "      [COLMAP 2/3] Sequential feature matching (video-optimised)..."
    colmap sequential_matcher \
        --database_path "$COLMAP_DB" \
        --SequentialMatching.overlap 10 \
        --SequentialMatching.loop_detection 1 \
        --SequentialMatching.vocab_tree_path /scratch0/jrameshs/colmap_env/share/colmap/vocab_tree_flickr100K_words32K.bin \
        --SiftMatching.use_gpu 0 || \
    colmap sequential_matcher \
        --database_path "$COLMAP_DB" \
        --SequentialMatching.overlap 10 \
        --SiftMatching.use_gpu 0

    # Step 3: Sparse reconstruction
    echo "      [COLMAP 3/3] Bundle adjustment / mapper..."
    colmap mapper \
        --database_path "$COLMAP_DB" \
        --image_path "$FRAMES_DIR" \
        --output_path "$COLMAP_SPARSE"

    echo "      COLMAP done: $(date)"

    # Convert to nerfstudio transforms.json
    echo "      Converting to nerfstudio transforms.json..."
    python3 -c "
import json, numpy as np
from pathlib import Path
import pycolmap

sparse_dir = Path('$COLMAP_SPARSE/0')
recon = pycolmap.Reconstruction(str(sparse_dir))
cameras = recon.cameras
images = recon.images

def get_Rt(img):
    # Support both old and new pycolmap APIs
    try:
        # pycolmap >= 0.6
        pose = img.cam_from_world
        if callable(pose):
            pose = pose()
        R = np.array(pose.rotation.matrix())
        t = np.array(pose.translation)
    except AttributeError:
        try:
            # pycolmap 0.4-0.5
            R = np.array(img.rotation_matrix())
            t = np.array(img.tvec)
        except AttributeError:
            # fallback: read qvec/tvec directly
            qw,qx,qy,qz = img.qvec
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
            ])
            t = np.array(img.tvec)
    return R, t

frames = []
for img_id, img in images.items():
    R, t = get_Rt(img)
    c2w = np.eye(4)
    c2w[:3,:3] = R.T
    c2w[:3, 3] = -R.T @ t
    # nerfstudio convention: flip Y and Z
    c2w[:,1] *= -1
    c2w[:,2] *= -1
    frames.append({
        'file_path': 'images/' + img.name,
        'transform_matrix': c2w.tolist()
    })

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
with open(out, 'w') as f:
    json.dump(transforms, f, indent=2)
print(f'Wrote {len(frames)} frames to {out}')
"

    # Symlink images dir for nerfstudio
    if [ ! -e "$COLMAP_OUT/images" ]; then
        ln -s "$(pwd)/$FRAMES_DIR" "$COLMAP_OUT/images"
        echo "      Symlinked images: $COLMAP_OUT/images -> $FRAMES_DIR"
    fi

    # Quality report
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
        if [ "$N_REG" -ge 300 ]; then
            echo "  │  Quality       : ✅ EXCELLENT  (300+ views)"
        elif [ "$N_REG" -ge 150 ]; then
            echo "  │  Quality       : ✅ GOOD  (150+ views, proceed)"
        elif [ "$N_REG" -ge 50 ]; then
            echo "  │  Quality       : ⚠️  MARGINAL  (50-150 views)"
            echo "  │  → Proceeding but quality may be limited"
        else
            echo "  │  Quality       : ❌ POOR  (<50 views)"
            echo "  │  → STOP — investigate before wasting GPU time"
            echo "  └─────────────────────────────────────────────────┘"
            exit 1
        fi
        echo "  └─────────────────────────────────────────────────┘"
        echo ""
    else
        echo "ERROR: transforms.json not created. COLMAP failed."
        exit 1
    fi
fi

# ── Generate point cloud PLY for Gaussian initialisation ──────────────────────
POINTS_PLY="$COLMAP_SPARSE/0/points3D.ply"

if [ -f "$POINTS_PLY" ]; then
    echo "[4b/6] Point cloud PLY already exists. Skipping."
else
    echo "[4b/6] Exporting COLMAP point cloud to PLY for Gaussian seed..."
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
print(f'Saved {len(pts):,} seed points → $POINTS_PLY')
"
fi

# ── Train splatfacto v3 (60K steps, aggressive densification) ────────────────
OUTPUT_DIR="outputs/splat_v3"
OUTPUT_PLY="$OUTPUT_DIR/scene.ply"

if [ -f "$OUTPUT_PLY" ]; then
    echo "[5/6] scene.ply already exists at $OUTPUT_PLY. Skipping training."
else
    echo "[5/6] Training splatfacto v3 (60,000 steps)..."
    echo "      Improvements over v2:"
    echo "        - densify-grad-thresh 0.0001  (2x more aggressive vs v2's 0.0002)"
    echo "        - densify-until-iter 50000    (keeps densifying 3x longer)"
    echo "        - 60K steps                   (vs 40K in v2)"
    echo "        - ~480 input frames           (vs 304 in v2)"
    echo "      Expected: 2.5-3 hours"
    echo "      Started: $(date)"

    ns-train splatfacto \
        --data "$COLMAP_OUT" \
        --output-dir "$OUTPUT_DIR" \
        --max-num-iterations 60000 \
        --pipeline.model.cull-alpha-thresh 0.005 \
        --pipeline.model.densify-grad-thresh 0.0001 \
        --pipeline.model.use-scale-regularization True \
        --pipeline.model.max-gauss-ratio 10.0 \
        --pipeline.model.densify-until-iter 50000 \
        --viewer.quit-on-train-completion True

    echo "      Training done: $(date)"
fi

# ── Export PLY from nerfstudio checkpoint ─────────────────────────────────────
if [ -f "$OUTPUT_PLY" ]; then
    echo "[6/6] scene.ply already present. Skipping export."
else
    echo "[6/6] Exporting Gaussians to PLY from nerfstudio checkpoint..."
    python3 -c "
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np, glob
from plyfile import PlyData, PlyElement

# Find latest config
configs = sorted(glob.glob('outputs/splat_v3/*/splatfacto/*/config.yml'))
if not configs:
    configs = sorted(glob.glob('outputs/splat_v3/splatfacto/*/config.yml'))
if not configs:
    raise FileNotFoundError('No splatfacto config.yml found under outputs/splat_v3/')
config = configs[-1]
print(f'Loading config: {config}')

_, pipeline, _, _ = eval_setup(Path(config))
m = pipeline.model
n = len(m.means)
print(f'Gaussians in model: {n:,}')

means = m.means.detach().cpu().numpy()
fdc   = m.features_dc.detach().cpu().numpy()
ops   = m.opacities.detach().cpu().numpy()
scs   = m.scales.detach().cpu().numpy()
qts   = m.quats.detach().cpu().numpy()

dtype = [
    ('x','f4'),('y','f4'),('z','f4'),
    ('nx','f4'),('ny','f4'),('nz','f4'),
    ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
    ('opacity','f4'),
    ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
    ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
]
vertex = np.zeros(n, dtype=dtype)
vertex['x'], vertex['y'], vertex['z']           = means[:,0], means[:,1], means[:,2]
vertex['nx'], vertex['ny'], vertex['nz']         = 0, 0, 0
vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'] = fdc[:,0], fdc[:,1], fdc[:,2]
vertex['opacity']                               = ops[:,0]
vertex['scale_0'], vertex['scale_1'], vertex['scale_2'] = scs[:,0], scs[:,1], scs[:,2]
vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3'] = qts[:,0], qts[:,1], qts[:,2], qts[:,3]

out = Path('$OUTPUT_PLY')
out.parent.mkdir(parents=True, exist_ok=True)
PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(out))
sz = out.stat().st_size / 1024 / 1024
print(f'Saved {n:,} Gaussians → {out}  ({sz:.1f} MB)')
"
fi

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  v3 Pipeline Complete: $(date)"
echo "========================================"
echo ""

if [ -f "$OUTPUT_PLY" ]; then
    SIZE=$(du -h "$OUTPUT_PLY" | cut -f1)
    echo "  ✅ scene.ply: $OUTPUT_PLY ($SIZE)"
    echo ""
    echo "  Download to Mac:"
    echo ""
    echo "  mkdir -p ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v3"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/splat_v3/scene.ply \\"
    echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v3/scene.ply"
    echo ""
    echo "  mkdir -p ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/sparse/0"
    echo "  scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "    \"jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/colmap_v3/sparse/0/\" \\"
    echo "    ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/sparse/0/"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/colmap_v3/transforms.json \\"
    echo "    ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/transforms.json"
    echo ""
    echo "  Then on Mac:"
    echo "  1. python3 scripts/convert_to_splat.py --input outputs/splat_v3/scene.ply --output outputs/splat_v3/scene.splat"
    echo "  2. python3 open_viewer.py"
else
    echo "  ⚠️  scene.ply not found — check training log"
    echo "     Looking for any PLY files..."
    find outputs/splat_v3 -name "*.ply" 2>/dev/null || echo "     None found."
fi

echo "========================================"