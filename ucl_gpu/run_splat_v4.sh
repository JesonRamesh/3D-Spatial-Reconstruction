#!/bin/bash
# =============================================================================
# RoboScene+ — v4: Y-UP training (fixes viewer orbit)
# =============================================================================
#
# WHY v4:
#   v3 produced scene.ply in an arbitrary orientation (nerfstudio default).
#   Post-hoc rotation via rotate_splat.py failed — the floor normal could not
#   be reliably found from the Gaussian cloud (nearly cubic room + floaters).
#
#   v4 uses ns-process-data --orientation-method up --auto-scale-poses which
#   computes the mean camera up-vector from ALL 539 COLMAP poses and rotates
#   the entire world so that vector aligns to [0,1,0] BEFORE training.
#   The resulting scene.ply is natively Y-up — OrbitControls works correctly
#   with no post-processing. This is the canonical nerfstudio solution.
#
# INPUTS (already on Bluestreak from v3 run):
#   data/video_frames_v2/          ← 641 frames @ 2fps (already extracted)
#   data/colmap_v3/sparse/0/       ← COLMAP sparse (539/641 registered)
#
# OUTPUTS:
#   data/colmap_v4/transforms.json  ← Y-up camera poses (new)
#   outputs/splat_v4/scene.ply      ← natively Y-up Gaussians
#
# HOW TO RUN (on Bluestreak):
#   ssh -J jrameshs@knuckles.cs.ucl.ac.uk jrameshs@bluestreak.cs.ucl.ac.uk
#   bash
#   source /opt/Python/Python-3.11.5_Setup.csh
#   export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
#   source /scratch0/jrameshs/roboscene_env/bin/activate
#   cd /scratch0/jrameshs/roboscene-plus
#   git pull
#   mkdir -p logs
#   nohup bash ucl_gpu/run_splat_v4.sh > logs/splat_v4.log 2>&1 &
#   tail -f logs/splat_v4.log
#
# EXPECTED WALL TIME: ~3-4 hours
#   - ns-process-data (pose alignment):  5-10 min (COLMAP already done)
#   - splatfacto 60K steps:              2.5-3 h
#   - PLY export:                        5-10 min
#
# DOWNLOAD (run on Mac after job completes):
#   mkdir -p ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v4
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \\
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_v4/scene.ply \\
#     ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v4/scene.ply
#
#   mkdir -p ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v4
#   scp -J jrameshs@knuckles.cs.ucl.ac.uk \\
#     jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_v4/transforms.json \\
#     ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v4/transforms.json
# =============================================================================

set -e

echo "========================================"
echo "  RoboScene+ — v4 Y-UP Training"
echo "  Starting: $(date)"
echo "========================================"
echo "  KEY: --orientation-method up guarantees Y-up output PLY"
echo "       No post-hoc rotation needed. OrbitControls will work."
echo ""

# ── Environment ────────────────────────────────────────────────────────────────
source /opt/Python/Python-3.11.5_Setup.csh 2>/dev/null || true
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
source /scratch0/jrameshs/roboscene_env/bin/activate
export PIP_CACHE_DIR="/scratch0/jrameshs/pip_cache"
export HF_HOME="/scratch0/jrameshs/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="/scratch0/jrameshs/roboscene-plus"
cd "$PROJECT_DIR"

FRAMES_DIR="data/video_frames_v2"
COLMAP_V3="data/colmap_v3"
COLMAP_V4="data/colmap_v4"
OUTPUT_DIR="outputs/splat_v4"
OUTPUT_PLY="$OUTPUT_DIR/scene.ply"

mkdir -p "$COLMAP_V4" "$OUTPUT_DIR" logs

# ── Sanity checks ──────────────────────────────────────────────────────────────
echo "[check] GPU status:"
python3 -c "import torch; print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'); print('  VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')"
echo ""

N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "[check] Input frames : $N_FRAMES  (in $FRAMES_DIR)"
if [ "$N_FRAMES" -lt 200 ]; then
    echo "ERROR: Expected 600+ frames in $FRAMES_DIR. Run v3 script first to extract frames."
    exit 1
fi

if [ ! -f "$COLMAP_V3/sparse/0/cameras.bin" ]; then
    echo "ERROR: COLMAP v3 sparse not found at $COLMAP_V3/sparse/0/"
    echo "       Run run_video_splat_v3.sh first (or ensure v3 data is present)."
    exit 1
fi

N_V3=$(python3 -c "import json; d=json.load(open('$COLMAP_V3/transforms.json')); print(len(d['frames']))" 2>/dev/null || echo "?")
echo "[check] COLMAP v3    : $N_V3 registered cameras"
echo "[check] nerfstudio   : $(python3 -c 'import nerfstudio; print(nerfstudio.__version__)' 2>/dev/null || echo 'not found')"
echo ""

# ── Step 1: ns-process-data with --orientation-method up ──────────────────────
# This is the KEY step. It:
#   1. Reads the existing COLMAP sparse reconstruction
#   2. Computes mean camera up-vector from all poses
#   3. Rotates the entire world so that vector = [0,1,0]  ← THE ORBIT FIX
#   4. Centers and scales poses
#   5. Writes data/colmap_v4/transforms.json with Y-up poses
#
# Crucially: we pass --skip-colmap because COLMAP is already done in v3.
# We reuse data/colmap_v3/sparse/0/ to save 90+ minutes.

if [ -f "$COLMAP_V4/transforms.json" ]; then
    N_V4=$(python3 -c "import json; d=json.load(open('$COLMAP_V4/transforms.json')); print(len(d['frames']))" 2>/dev/null || echo "?")
    echo "[1/3] ns-process-data already done — $N_V4 Y-up poses. Skipping."
else
    echo "[1/3] Running ns-process-data with --orientation-method up..."
    echo "      This rotates the scene so camera mean-up = [0,1,0]"
    echo "      Input : $COLMAP_V3/sparse/0/  (existing COLMAP)"
    echo "      Output: $COLMAP_V4/transforms.json"
    echo "      Started: $(date)"

    # Create images symlink that ns-process-data expects
    if [ ! -e "$COLMAP_V4/images" ]; then
        ln -sf "$(pwd)/$FRAMES_DIR" "$COLMAP_V4/images"
    fi

    # Copy COLMAP sparse from v3 so ns-process-data can read it
    mkdir -p "$COLMAP_V4/colmap/sparse/0"
    cp "$COLMAP_V3/sparse/0/cameras.bin"  "$COLMAP_V4/colmap/sparse/0/"
    cp "$COLMAP_V3/sparse/0/images.bin"   "$COLMAP_V4/colmap/sparse/0/"
    cp "$COLMAP_V3/sparse/0/points3D.bin" "$COLMAP_V4/colmap/sparse/0/"

    ns-process-data images \
        --data "$FRAMES_DIR" \
        --output-dir "$COLMAP_V4" \
        --skip-colmap \
        --colmap-model-path "$COLMAP_V4/colmap/sparse/0" \
        --orientation-method up \
        --auto-scale-poses \
        --matching-method exhaustive \
        --no-gpu

    echo "      ns-process-data done: $(date)"

    # Verify Y-up alignment
    python3 - <<'PYEOF'
import json, numpy as np
with open('data/colmap_v4/transforms.json') as f:
    tfm = json.load(f)
frames = tfm['frames']
c2ws = [np.array(f['transform_matrix']) for f in frames]
ups = [c2w[:3,1] for c2w in c2ws]
mean_up = np.mean(ups, axis=0)
mean_up /= np.linalg.norm(mean_up)
dot = abs(np.dot(mean_up, [0,1,0]))
print(f'  Frames: {len(frames)}')
print(f'  Mean camera UP: {mean_up}')
print(f'  Alignment with [0,1,0]: {dot:.4f}  (target > 0.95)')
if dot > 0.95:
    print('  ✅ Y-UP ALIGNMENT CONFIRMED — orbit will work correctly')
else:
    print(f'  ⚠️  Alignment weaker than expected (dot={dot:.3f})')
    print('     Orbit may still be slightly off — but should be much better than v3')
PYEOF
fi

# ── Step 2: Train splatfacto v4 ───────────────────────────────────────────────
if [ -f "$OUTPUT_PLY" ]; then
    N_G=$(python3 -c "
from plyfile import PlyData
n = len(PlyData.read('$OUTPUT_PLY')['vertex'])
print(f'{n:,}')
" 2>/dev/null || echo "?")
    echo "[2/3] scene.ply already exists ($N_G Gaussians). Skipping training."
else
    echo "[2/3] Training splatfacto v4 (60,000 steps, Y-up data)..."
    echo "      Input: $COLMAP_V4/transforms.json"
    echo "      Output: $OUTPUT_DIR"
    echo "      Started: $(date)"

    ns-train splatfacto \
        --data "$COLMAP_V4" \
        --output-dir "$OUTPUT_DIR" \
        --max-num-iterations 60000 \
        --pipeline.model.cull-alpha-thresh 0.005 \
        --pipeline.model.densify-grad-thresh 0.0002 \
        --pipeline.model.use-scale-regularization True \
        --pipeline.model.max-gauss-ratio 10.0 \
        --viewer.quit-on-train-completion True

    echo "      Training done: $(date)"
fi

# ── Step 3: Export PLY ────────────────────────────────────────────────────────
if [ -f "$OUTPUT_PLY" ] && [ $(python3 -c "from plyfile import PlyData; print(len(PlyData.read('$OUTPUT_PLY')['vertex']))" 2>/dev/null || echo 0) -gt 1000000 ]; then
    echo "[3/3] scene.ply already exported with sufficient Gaussians. Skipping."
else
    echo "[3/3] Exporting Gaussians to PLY..."

    # Find the latest splatfacto config
    CONFIG=$(find "$OUTPUT_DIR" -name "config.yml" | sort | tail -1)
    if [ -z "$CONFIG" ]; then
        echo "ERROR: No config.yml found under $OUTPUT_DIR"
        find "$OUTPUT_DIR" -name "*.yml" 2>/dev/null || true
        exit 1
    fi
    echo "      Config: $CONFIG"

    ns-export gaussian-splat \
        --load-config "$CONFIG" \
        --output-dir "$OUTPUT_DIR"

    # ns-export writes to output-dir/splat.ply — rename to scene.ply
    if [ -f "$OUTPUT_DIR/splat.ply" ] && [ ! -f "$OUTPUT_PLY" ]; then
        mv "$OUTPUT_DIR/splat.ply" "$OUTPUT_PLY"
        echo "      Renamed splat.ply → scene.ply"
    fi

    # Fallback: extract from checkpoint directly
    if [ ! -f "$OUTPUT_PLY" ]; then
        echo "      ns-export didn't produce splat.ply — extracting from checkpoint..."
        python3 - <<PYEOF
import torch
_orig = torch.load
torch.load = lambda *a,**kw: _orig(*a,**{**kw,'weights_only':False})

import glob
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np
from plyfile import PlyData, PlyElement

configs = sorted(glob.glob('$OUTPUT_DIR/*/splatfacto/*/config.yml'))
if not configs:
    configs = sorted(glob.glob('$OUTPUT_DIR/splatfacto/*/config.yml'))
assert configs, 'No config.yml found'
_, pipeline, _, _ = eval_setup(Path(configs[-1]))
m = pipeline.model
n = len(m.means)
print(f'Gaussians: {n:,}')

means = m.means.detach().cpu().numpy()
fdc   = m.features_dc.detach().cpu().numpy().reshape(n,-1)
frest = m.features_rest.detach().cpu().numpy().reshape(n,-1) if hasattr(m,'features_rest') else np.zeros((n,45))
ops   = m.opacities.detach().cpu().numpy().reshape(n,1)
scs   = m.scales.detach().cpu().numpy()
qts   = m.quats.detach().cpu().numpy()

n_rest = frest.shape[1]
dtype = ([('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4')]
       + [('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4')]
       + [(f'f_rest_{i}','f4') for i in range(n_rest)]
       + [('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
          ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')])
vertex = np.zeros(n, dtype=dtype)
vertex['x'],vertex['y'],vertex['z'] = means[:,0],means[:,1],means[:,2]
vertex['f_dc_0'],vertex['f_dc_1'],vertex['f_dc_2'] = fdc[:,0],fdc[:,1],fdc[:,2]
for i in range(n_rest):
    vertex[f'f_rest_{i}'] = frest[:,i]
vertex['opacity'] = ops[:,0]
vertex['scale_0'],vertex['scale_1'],vertex['scale_2'] = scs[:,0],scs[:,1],scs[:,2]
vertex['rot_0'],vertex['rot_1'],vertex['rot_2'],vertex['rot_3'] = qts[:,0],qts[:,1],qts[:,2],qts[:,3]

out = Path('$OUTPUT_PLY')
PlyData([PlyElement.describe(vertex,'vertex')]).write(str(out))
print(f'Saved {n:,} Gaussians -> {out}  ({out.stat().st_size/1e6:.0f} MB)')
PYEOF
    fi
fi

# ── Verify Y-up in output PLY ─────────────────────────────────────────────────
if [ -f "$OUTPUT_PLY" ]; then
    python3 - <<'PYEOF'
import numpy as np
from plyfile import PlyData
ply = PlyData.read('outputs/splat_v4/scene.ply')
v = ply['vertex']
rng = np.random.default_rng(42)
idx = rng.choice(len(v['x']), min(100000,len(v['x'])), replace=False)
xyz = np.stack([np.array(v['x'])[idx], np.array(v['y'])[idx], np.array(v['z'])[idx]], axis=1)
ranges = xyz.max(axis=0) - xyz.min(axis=0)
print(f'  scene.ply Gaussian count: {len(v["x"]):,}')
print(f'  X range: {xyz[:,0].min():.2f} to {xyz[:,0].max():.2f}  ({ranges[0]:.2f}m)')
print(f'  Y range: {xyz[:,1].min():.2f} to {xyz[:,1].max():.2f}  ({ranges[1]:.2f}m)')
print(f'  Z range: {xyz[:,2].min():.2f} to {xyz[:,2].max():.2f}  ({ranges[2]:.2f}m)')
height_axis = np.argmin(ranges)
names = ['X','Y','Z']
print(f'  Smallest-range axis: {names[height_axis]} ({ranges[height_axis]:.2f}m) — should be Y for Y-up scene')
if height_axis == 1 and ranges[1] < ranges[0] * 0.8 and ranges[1] < ranges[2] * 0.8:
    print('  ✅ Y IS THE HEIGHT AXIS — scene is correctly Y-up')
else:
    print('  ⚠️  Y is not clearly the smallest axis — check orbit carefully in viewer')
PYEOF
fi

# ── Final report ──────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  v4 Pipeline Complete: $(date)"
echo "========================================"

if [ -f "$OUTPUT_PLY" ]; then
    SIZE=$(du -h "$OUTPUT_PLY" | cut -f1)
    echo "  ✅ outputs/splat_v4/scene.ply  ($SIZE)"
    echo "  ✅ data/colmap_v4/transforms.json"
    echo ""
    echo "  === DOWNLOAD TO MAC ==="
    echo "  mkdir -p ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v4"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\\\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/outputs/splat_v4/scene.ply \\\\"
    echo "    ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v4/scene.ply"
    echo ""
    echo "  mkdir -p ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v4"
    echo "  scp -J jrameshs@knuckles.cs.ucl.ac.uk \\\\"
    echo "    jrameshs@bluestreak.cs.ucl.ac.uk:${PROJECT_DIR}/data/colmap_v4/transforms.json \\\\"
    echo "    ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v4/transforms.json"
    echo ""
    echo "  === NEXT STEPS ON MAC ==="
    echo "  1. python3 scripts/convert_to_splat.py --input outputs/splat_v4/scene.ply --output outputs/splat_v4/scene.splat"
    echo "  2. python3 open_viewer.py  # verify orbit"
    echo "  3. python3 scripts/paint_semantic_gaussians.py --splat_ply outputs/splat_v4/scene.ply ..."
else
    echo "  ❌ scene.ply not found — check logs/splat_v4.log"
fi

echo "========================================"
