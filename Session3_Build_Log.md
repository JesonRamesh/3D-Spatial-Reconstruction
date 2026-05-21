# RoboScene+ — Session 3 Build Log & Context
## Gaussian Splatting Training on bluestreak.cs.ucl.ac.uk
### Last updated: 15 May 2026

---

## Current Status

**Session 1:** ✅ Complete — 82 frames extracted  
**Session 2:** ✅ Complete — VGGT ran successfully (80 frames, 11s, all outputs saved)  
**Session 3:** 🔄 In progress — switching to nerfstudio splatfacto after gsplat CUDA build failures  

---

## Environment State on bluestreak

### Activate on reconnect (always do this first)
```bash
ssh bluestreak
bash
source ~/.bashrc
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
```

### Key Paths
```
Project:         /scratch0/jrameshs/roboscene-plus/
gsplat src:      /scratch0/jrameshs/gsplat_src/         (broken — ignore)
venv:            /scratch0/jrameshs/roboscene_env/
VGGT outputs:    /scratch0/jrameshs/roboscene-plus/data/vggt_out/
Frames symlink:  /scratch0/jrameshs/roboscene-plus/data/vggt_out/images/
Build log:       /scratch0/jrameshs/roboscene-plus/logs/gsplat_build.log
```

### Installed and Working
- Python 3.11.5 ✅
- PyTorch 2.3.1+cu121 ✅
- CUDA 12.6 at `/opt/cuda/cuda-12.6/` ✅
- nvcc at `/opt/cuda/cuda-12.6/bin/nvcc` ✅
- nerfstudio ✅ (just installed — splatfacto available)
- VGGT outputs complete ✅

### ~/.bashrc exports (already set)
```bash
export CUDA_HOME=/opt/cuda/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export PIP_CACHE_DIR=/scratch0/jrameshs/pip_cache
```

---

## What Happened with gsplat (Full Issue Log)

### Problem 1 — gsplat.scripts.simple_trainer doesn't exist
gsplat 1.5.3 removed the `simple_trainer` module that our script called via
`python -m gsplat.scripts.simple_trainer`. The module doesn't exist in any
version as a runnable subprocess target.

**Fix attempted:** Downloaded `simple_trainer.py` from gsplat GitHub examples
into `scripts/gsplat_examples/`. This worked for the script itself but exposed
further dependency issues.

### Problem 2 — pycolmap.SceneManager removed
gsplat's `datasets/colmap.py` uses `pycolmap.SceneManager` which was removed
in pycolmap 0.5.0+. All available versions (0.5.0 through 4.0.4) lack it.

**Fix applied:** Patched `scripts/gsplat_examples/datasets/colmap.py` in-place
on bluestreak using Python patch scripts. Replaced SceneManager with our own
`colmap_utils.py` readers. Also added `camera_type`, distortion params, and
safe point track iteration.

### Problem 3 — gsplat CUDA extension (_C) not compiled
The pip-installed gsplat wheel has no CUDA extension — `from gsplat import _C`
raises ImportError. This is because PyPI distributes a pure-Python fallback
wheel when CUDA compilation isn't triggered.

**Fix attempted:** Build from source via:
```bash
TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```
Failed because:
- `wheel` package missing initially
- GLM math library submodule not cloned (empty directory)
- nvcc not on PATH (found at `/opt/cuda/cuda-12.6/bin/nvcc`, not `/usr/local/cuda`)

**Fix attempted:** Manually cloned GLM:
```bash
cd /scratch0/jrameshs/gsplat_src/gsplat/cuda/csrc/third_party/
git clone https://github.com/g-truc/glm.git glm
```
Then rebuilt with editable install. Build completed ("Successfully installed
gsplat-1.5.3") but `_C` still not importable — extension compiled into
wrong location or not compiled at all despite no visible error.

### Decision: Switch to nerfstudio splatfacto
After multiple failed gsplat build attempts, switched to nerfstudio which:
- Has a clean pip install with no CUDA compilation required
- Uses gsplat internally but handles CUDA setup automatically
- Is widely used in research, well-maintained
- Supports COLMAP data format natively
- Exports .ply files compatible with the rest of our pipeline

---

## Immediate Next Step — Run splatfacto Training

nerfstudio is installed. Run this now:

```bash
cd /scratch0/jrameshs/roboscene-plus

# First convert VGGT COLMAP output to nerfstudio format
ns-process-data images \
  --data data/vggt_out/images/ \
  --output-dir data/nerfstudio_data/ \
  --skip-colmap \
  --colmap-model-path data/vggt_out/sparse/

# Then train splatfacto
ns-train splatfacto \
  --data data/nerfstudio_data/ \
  --output-dir outputs/splat/ \
  --viewer.quit-on-train-completion True \
  --max-num-iterations 15000
```

**If ns-process-data fails**, try direct colmap format:
```bash
ns-train splatfacto \
  --data data/vggt_out/ \
  --output-dir outputs/splat/ \
  --pipeline.datamanager.dataparser nerfstudio \
  --max-num-iterations 15000
```

**Expected output:** A `outputs/splat/splatfacto/` directory containing:
- `config.yml`
- `nerfstudio_models/` with checkpoint files
- We then export to .ply format

**Export to .ply after training:**
```bash
ns-export gaussian-splat \
  --load-config outputs/splat/splatfacto/config.yml \
  --output-dir outputs/splat/
# This creates outputs/splat/splat.ply
cp outputs/splat/splat.ply outputs/splat/scene.ply
```

---

## If nerfstudio Also Fails — Fallback Options

### Option A — Run Gaussian Splatting on M4 Pro (guaranteed to work)
Slower (~90 min) but MPS backend is confirmed working from dry run.

```bash
# On Mac terminal:
cd ~/3D-Spatial-Reconstruction

# Install opensplat for Mac
brew install cmake
pip install opensplat  # or build from source

# Or use the nerfstudio Mac install:
pip install nerfstudio
ns-train splatfacto --data data/vggt_out/ --output-dir outputs/splat/
```

### Option B — Use instant-ngp as intermediate
```bash
ns-train instant-ngp --data data/vggt_out/ --output-dir outputs/splat/
```

### Option C — Skip to a pre-existing Gaussian Splatting result
Use a public demo .ply file temporarily to build Sessions 4–8 while
the GPU training issue is resolved separately. The confidence map,
scene graph, and Claude API work are all independent of the exact
training method.

---

## Data Already Saved (Do Not Lose)

### On bluestreak scratch (download before booking ends):
```
/scratch0/jrameshs/roboscene-plus/data/vggt_out/
├── depths/          80 depth maps (.npy)
├── sparse/          cameras.bin, images.bin, points3D.bin, points.ply
├── images/          symlink to ../frames/
├── camera_poses.json
└── vggt_metadata.json
```

### Download command (run from Mac terminal):
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/vggt_out/ \
  ~/3D-Spatial-Reconstruction/data/
```

⚠️ **Do this before your booking expires. Set a phone alarm.**

---

## VGGT Output Verification (confirmed working)

```
Frames processed:    80
Device:              cuda (torch.bfloat16)
Inference time:      11.0s total, 0.14s/frame
Depth maps:          80 files in data/vggt_out/depths/
Camera poses:        80 cameras in camera_poses.json
COLMAP sparse:       cameras.bin, images.bin, points3D.bin, points.ply
Depth range:         confirmed sensible (metres scale)
```

---

## colmap_utils.py Confirmed Working

```python
Cameras: 80   # PINHOLE model, fx~5948, fy~3346, cx=960, cy=540
Images:  80
Points:  100000
Point fields: point3D_id, xyz, rgb, error, track (track=[] — empty, OK)
```

The empty `track` field means point-image associations aren't stored.
This is fine — gsplat/nerfstudio uses the point cloud for initialisation
only, not for the track associations.

---

## Camera Intrinsics Note

VGGT outputs intrinsics for original 1920×1080 resolution:
```
fx=5948.78, fy=3346.19, cx=960.0, cy=540.0
```

Frames were resized to 518×518 for VGGT inference (--resolution 518).
If nerfstudio needs intrinsics matching the actual image size, use
the resized intrinsics:
```
scale_x = 518/1920 = 0.2698
scale_y = 518/1080 = 0.4796
fx_resized = 5948.78 * 0.2698 = 1605.4
fy_resized = 3346.19 * 0.4796 = 1605.6
cx_resized = 960 * 0.2698 = 259.0
cy_resized = 540 * 0.4796 = 259.0
```

This makes the camera roughly square (as expected for 518×518 images).

---

## Sessions Remaining

| Session | What | GPU? | Status |
|---|---|---|---|
| 3 | Gaussian Splatting | ✅ | 🔄 Switching to nerfstudio splatfacto |
| 4 | Grounded SAM2 | ✅ | ⬜ TODO |
| 5 | 3D semantic lifting | ❌ | ⬜ TODO |
| 6 | Confidence map ★ | ❌ | ⬜ TODO |
| 7 | Dead zone completion | ❌ | ⬜ TODO |
| 8 | Scene graph + Claude API | ❌ | ⬜ TODO |
| 9 | Gradio app + deploy | ❌ | ⬜ TODO |
| 10 | README + polish | ❌ | ⬜ TODO |

---

## Critical Reminders

1. **Download vggt_out/ to Mac before booking ends**
2. **Check booking end time at mydesk.cs.ucl.ac.uk**
3. **Set phone alarm 1 hour before booking ends**
4. Scratch is wiped at session end — home directory (10GB) is persistent
5. Always `bash` first after SSH (default shell is tcsh)
6. Always `source ~/.bashrc` then activate venv after reconnecting
---

## Session 3 — Final Status & What We Have

### Completed
- ✅ nerfstudio installed successfully (splatfacto available)
- ✅ Gaussian Splatting training completed — 15000 steps
- ✅ scene.ply exported — 528MB
- ✅ All files downloaded to Mac at ~/3D-Spatial-Reconstruction/
- ✅ Committed and pushed to GitHub

### Output Files on Mac
~/3D-Spatial-Reconstruction/
├── data/vggt_out/
│   ├── depths/          80 depth maps + 80 confidence maps (160 files)
│   ├── sparse/          cameras.bin, images.bin, points3D.bin, points.ply
│   ├── camera_poses.json
│   └── vggt_metadata.json
└── outputs/splat/
└── scene.ply        528MB Gaussian Splat

### Visual Inspection Result
Opened scene.ply in SuperSplat (supersplat.playcanvas.com).
Reconstruction is visible but distorted — colours are correct (warm
beiges matching indoor room) but geometry is fragmented. Not
submission-quality yet.

### Root Causes of Poor Quality
1. Camera intrinsics mismatch — VGGT output intrinsics for 1920×1080
   but frames were resized to 518×518 for inference. Nerfstudio may
   have used wrong focal length.
2. 15000 steps may be insufficient — splatfacto often needs 30000
   for indoor scenes.
3. Video quality — first diagnostic run, reshoot planned.

### What Needs to Happen Next

**Option A — Reshoot + rerun (recommended):**
1. Record a new video — slower pan, better lighting, film corners
2. Re-extract frames: `python scripts/extract_frames.py`
3. Fix intrinsics: pass correct scaled intrinsics to nerfstudio
4. Retrain with 30000 steps on bluestreak

**Fix for intrinsics mismatch — pass transforms.json to nerfstudio:**
Instead of using COLMAP format, generate a transforms.json with
correct scaled intrinsics:
```python
# Scale intrinsics from 1920x1080 to 518x518
fx_scaled = 5948.78 * (518/1920)  # = 1605.4
fy_scaled = 3346.19 * (518/1080)  # = 1605.6
cx_scaled = 960.0  * (518/1920)   # = 259.0
cy_scaled = 540.0  * (518/1080)   # = 259.0
```
Write a script `scripts/export_transforms.py` that generates
nerfstudio-format transforms.json with these corrected values.

**Option B — Run more iterations on existing data:**
```bash
# On bluestreak, continue training from checkpoint:
ns-train splatfacto \
  --data data/vggt_out/ \
  --output-dir outputs/splat/ \
  --max-num-iterations 30000 \
  --pipeline.model.sh-degree 3 \
  colmap --colmap-path sparse/
```

### gsplat Build Issue — Documented for Future Reference
gsplat CUDA extension could not be compiled on bluestreak due to:
- GLM submodule not populated by default
- nvcc at non-standard path: /opt/cuda/cuda-12.6/bin/nvcc
- Even after fixing both, editable install succeeded but _C module
  still not importable

**Resolution:** Switched to nerfstudio splatfacto which handles
CUDA internally without requiring manual compilation.

**If gsplat is needed in future sessions**, try:
```bash
cd /scratch0/jrameshs/gsplat_src
git submodule update --init --recursive
ls gsplat/cuda/csrc/third_party/glm/glm/  # verify populated
export CUDA_HOME=/opt/cuda/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 pip install -e . --no-build-isolation
python3 -c "from gsplat import _C; print(_C)"
```

### Next Session (Session 4) — Grounded SAM2
Run on bluestreak (book a new GPU slot):
```bash
# Install SAM2 + GroundingDINO
pip install groundingdino-py
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Run segmentation on 80 frames
python scripts/run_semantic.py \
  --frames_dir data/frames/ \
  --output_dir outputs/semantic/ \
  --labels "chair,desk,laptop,monitor,lamp,bookshelf,door,window,plant,keyboard" \
  --device cuda
```

### Reconnect to bluestreak
```bash
ssh bluestreak
bash
source ~/.bashrc
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
```

---

## Session 3 — Continued Debugging & Frame Sorting Fix

### Root Cause Confirmed by debug_reconstruction.py
Running scripts/debug_reconstruction.py produced a definitive diagnosis:

**INTRINSICS: ✅ FIXED**
- fx=2113.81, fy=2113.81, fx/fy=1.0000
- HFOV=87.29 deg (correct for iPhone ultrawide)
- Double-scaling bug fully resolved

**CRITICAL: Batch pose discontinuity**
- 511 still photos processed in batches of 10
- VGGT estimates poses independently per batch
- Batch boundaries show 8.8x translation jumps
- Worst cases: 138.9° rotation flip at frames 109→110
- 51 disconnected local maps → "abstract explosion" appearance
- Empty tracks (100%) confirmed NOT a root cause

### Fix Implemented: Frame Similarity Sorting
Claude Code added sort_frames_by_similarity() to scripts/run_vggt.py:
- Loads all frames as 64x64 grayscale thumbnails
- Computes greedy nearest-neighbour path via histogram correlation
- Ensures consecutive batches share overlapping viewpoints
- Saves reordering map to output_dir/frame_order.json
- --sort_frames flag added (default True)
- Verified with unit tests, AST parse confirmed clean

### Batch Size Testing on RTX 4070 Ti Super (16GB VRAM)
- batch_size=50: OOM ❌
- batch_size=30: OOM ❌  
- batch_size=20: ✅ Success — 511 frames, 145.4s, 0.28s/frame

### Current State
VGGT completed with batch_size=20 + frame similarity sorting.
Outputs in /scratch0/jrameshs/roboscene-plus/data/vggt_out/

### Next Steps
1. Push frame sorting fix to GitHub
2. Run debug_reconstruction.py again to verify boundary jumps improved
3. Chain splatfacto training (30000 steps)
4. Download scene.ply and verify in SuperSplat
5. If still distorted → consider COLMAP BA as global registration step

### Commands to Continue
```bash
# On bluestreak — push latest changes first
cd /scratch0/jrameshs/roboscene-plus
unset SSH_ASKPASS && unset DISPLAY && git pull

# Verify boundary jumps improved
python3 scripts/debug_reconstruction.py 2>&1 | grep -A 20 "CHECK 4"

# If improved, chain training
ln -sf /scratch0/jrameshs/roboscene-plus/data/frames \
       /scratch0/jrameshs/roboscene-plus/data/vggt_out/images

yes | nohup ns-train splatfacto \
  --data data/vggt_out/ \
  --output-dir outputs/splat/ \
  --max-num-iterations 30000 \
  --viewer.quit-on-train-completion True \
  colmap \
  --colmap-path sparse/ \
  > logs/splat_v3.log 2>&1 &
```

### Key File Locations
```
Frames (sorted):  /scratch0/jrameshs/roboscene-plus/data/frames/ (511 photos)
Frame order map:  /scratch0/jrameshs/roboscene-plus/data/vggt_out/frame_order.json
VGGT outputs:     /scratch0/jrameshs/roboscene-plus/data/vggt_out/
Debug outputs:    /scratch0/jrameshs/roboscene-plus/outputs/debug/
```

---

## Frame Sorting Results — Still Failing

### Debug Results After Similarity Sort (batch_size=20)
- Rotation ratio: 8.8x → 1.08x ✅ (rotation improved)
- Translation ratio: 8.8x → 10.35x ❌ (translation worse)
- 180° flips remain at frames 319→320, 289→290
- Root cause: VGGT produces 26 independent local coordinate frames
  that SVD stitching cannot recover from when 180° flips exist

### Next Fix: COLMAP Bundle Adjustment
Use VGGT poses as initialisation, run COLMAP BA to globally
register all 511 poses into one coherent coordinate frame.
Checking if COLMAP is available on bluestreak:
`which colmap || find /usr /opt -name "colmap" -type f 2>/dev/null`

---

## Subsampling Fix — Every 5th Frame

frame_order.json not saved by last run.
Subsampling directly from frames/ directory instead:
- Take every 5th frame from filename-sorted order = ~102 frames
- Run VGGT with batch_size=103 (single batch, all frames at once)
- Single batch = single coherent coordinate frame
- No stitching errors possible

---

## Strategic Pivot to MASt3R-SLAM

### Why VGGT Was Abandoned for Video Input
- VGGT batch_size limited by 16GB VRAM
- 30 frames = too little coverage for a room
- 511 still photos = 26 disconnected local maps, unfixable
- VGGT not designed for sequential video input

### MASt3R-SLAM — New Primary Pipeline
Repo: https://github.com/rmurai0610/MASt3R-SLAM
Paper: CVPR 2025
Key advantages:
- Processes video frame-by-frame (no VRAM batch limit)
- Loop closure ensures global consistency
- No camera calibration required
- Takes MP4 video directly
- Outputs dense point cloud + globally consistent poses

### Installation on bluestreak
```bash
cd /scratch0/jrameshs
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM
source /scratch0/jrameshs/roboscene_env/bin/activate
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
```
numpy/opencv conflict warning — harmless, ignored.

### Checkpoints to Download
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth \
  -P checkpoints/
```

### New Script (Claude Code — in progress)
scripts/run_mast3r_slam.py:
- Takes --video_path, --output_dir, --mast3r_dir args
- Runs MASt3R-SLAM via subprocess on MP4 video
- Converts SLAM output to COLMAP format
- Extracts frames at 2fps for nerfstudio
- Output structure identical to data/vggt_out/
  so train_splat.py works unchanged

ucl_gpu/run_mast3r_job.sh:
- nohup jobscript for bluestreak

### Existing Scripts — Untouched
- scripts/run_vggt.py: kept as alternative
- scripts/train_splat.py: unchanged, works with both pipelines
- All debug scripts: kept for reference

### Next Steps
1. Checkpoints finish downloading on bluestreak
2. run_mast3r_slam.py created by Claude Code → push → pull on bluestreak
3. Record proper 2-3 minute video of room
4. Upload video to bluestreak
5. Run MASt3R-SLAM → verify point cloud quality
6. Chain nerfstudio splatfacto training
7. Download scene.ply → verify in SuperSplat

### Video Recording Plan
- 2-3 minutes total (not 30 seconds)
- Slow continuous pan, three height layers
- Start at door, full perimeter, desk detail, corners
- 1080p 30fps, main camera (not ultrawide this time)
- Sequential smooth motion — designed for SLAM

---

## MASt3R-SLAM Setup Complete — Config Path Bug

### Installation Status
- MASt3R-SLAM cloned and installed ✅
- Checkpoints downloaded (2.6GB + 1.8MB) ✅
- help message confirmed working ✅
- run_mast3r_slam.py created and pushed ✅

### Bug: Config Relative Path
MASt3R-SLAM's main.py looks for config relative to its
working directory. Our script runs from project root so
config/base.yaml resolves incorrectly.

Fix: pass absolute path to --config:
/scratch0/jrameshs/MASt3R-SLAM/config/base.yaml

### Video Ready
room_video.MOV uploaded to bluestreak:
/scratch0/jrameshs/roboscene-plus/data/raw/room_video.MOV
Format: MOV, 1080p 60fps, 2min 30sec
Shot at 1x main lens (no barrel distortion)

---

## MASt3R-SLAM Runtime Errors

### Error 1: Config relative path — FIXED
Pass absolute path: /scratch0/jrameshs/MASt3R-SLAM/config/base.yaml

### Error 2: Checkpoint relative path
MASt3R-SLAM looks for checkpoints/ relative to working directory.
Fix: run main.py from /scratch0/jrameshs/MASt3R-SLAM/ directory.

### Error 3: img_size TypeError
'int' object is not subscriptable in mast3r/model.py line 55.
Possible version mismatch between checkpoint and code.
Testing if running from correct directory fixes this.

### Command That Should Work
```bash
cd /scratch0/jrameshs/MASt3R-SLAM
python main.py \
  --dataset /path/to/room_video.MOV \
  --config /scratch0/jrameshs/MASt3R-SLAM/config/base.yaml \
  --save-as /path/to/output \
  --no-viz
```

---

## MASt3R-SLAM SUCCESS ✅

### Point Cloud Results
- 7,333,380 points with RGB colour
- Scene size: 3.74 x 3.56 x 4.89 m (plausible for small room)
- Bird's eye view confirms: room outline, desk, bed, shelf, door all visible
- 54 keyframes extracted
- Globally consistent geometry — no batch discontinuity issues

### Visual Confirmation
Bird's eye view (X-Z plane) shows:
- Beige perimeter = walls correctly placed
- Desk cluster = right side ✅
- Bed cluster = top ✅  
- Shelf cluster = left ✅
- Door gap = bottom ✅

### Next Steps
1. Convert MASt3R-SLAM output to COLMAP format
   (cameras.bin, images.bin, points3D.bin)
2. Run nerfstudio splatfacto on converted output
3. Export scene.ply and verify in SuperSplat
4. This should finally produce a clean photorealistic splat

### Key Files
- Point cloud: data/mast3r_out/slam_output/room_video.ply (105MB)
- Poses: data/mast3r_out/slam_output/room_video.txt
- Keyframes: data/mast3r_out/slam_output/keyframes/room_video/ (54 PNGs)

---

## MASt3R-SLAM + Splatfacto — RECONSTRUCTION CONFIRMED WORKING ✅

### Quality Verification
Rendered frame vs ground truth comparison:
- Bed, duvet pattern, headboard: matched ✅
- Fan, chair, laptop: matched ✅  
- Wall colour and proportions: matched ✅
- Minor blurring/ghosting on walls: expected (54 keyframes, 30k steps)

### Root Cause of SuperSplat "Distortion"
SuperSplat navigation issue — viewing from outside scene.
Rendered outputs from training cameras confirm correct geometry.

### Final Pipeline (Working)
Video → MASt3R-SLAM → COLMAP conversion → nerfstudio splatfacto
- 317 extracted frames, 54 SLAM keyframes
- 30000 training steps
- scene.ply downloaded to outputs/splat_mast3r/

### Decision
Accept current quality, proceed to Sessions 4-8.
Deadline: 25 May 2026. Remaining sessions are the differentiators.

### Sessions Remaining
- Session 4: Grounded SAM2 semantic segmentation
- Session 5: 3D semantic lifting  
- Session 6: Confidence map (novel contribution)
- Session 7: Dead zone completion
- Session 8: Scene graph + Claude API
- Session 9: Gradio app + deployment
- Session 10: README + polish

---

## Session 4 — Semantic Segmentation Finalised ✅

### Mask Quality
- SAM2 pixel-level masks working for high-contrast objects
- Chair, desk, laptop, shelf: correct pixel masks ✅
- Bed: bounding box fallback (low contrast vs wall) — expected
- Both bounding box + mask rendered in debug PNGs

### Colour Scheme (per object class)
bed=#4E79A7, desk=#59A14F, chair=#F28E2B, laptop=#E15759,
shelf=#76B7B2, door=#EDC948, window=#B07AA1, fan=#FF9DA7,
lamp=#9C755F, monitor=#BAB0AC

### Known Limitation
SAM2 struggles with low-contrast boundaries (bed vs cream wall).
Bounding box used as fallback mask. 3D location still correct.
Document in README limitations section.

### Output Files
outputs/semantic_v2/ — fixed pixel-level masks (317 JSON files)
outputs/semantic_v2/debug/ — visualisation PNGs with masks

### Next: Session 5 — 3D Semantic Lifting
Runs entirely on M4 Pro. No GPU needed.
Uses depth maps + camera poses to lift 2D masks into 3D space.
Input: outputs/semantic_v2/*.json + data/mast3r_out/
Output: outputs/objects_3d.json (3D bounding boxes per object)

---

## Sessions 7 + 8 — Complete ✅

### Session 7 — Dead Zone Completion
- Found dead zones from confidence_map.npy (65.6% low confidence)
- Dead zones = walls/corners, not scene centre voids
- Inpainting mask was tiny (1 pixel) — confirms MASt3R-SLAM
  covered the main navigable area well
- Key insight: small mask = good coverage, not a failure
- Output: outputs/dead_zones/dead_zone_summary.png

### Session 8 — Scene Graph + Claude API ✅

#### Scene Graph (build_scene_graph.py)
- 10 objects with 3D positions, confidence, provenance
- Spatial edges computed: next_to, near_wall, on_top_of
- Room summary: dimensions, coverage stats
- Output: outputs/scene_graph.json

#### Claude API Query (query_scene.py)
- Model: claude-sonnet-4-5, streaming, max_tokens=500
- Verified working — example response for "Where is the laptop?":
  Position: (0.73m, 0.30m, 0.79m)
  Confidence: 0.47 (partially seen)
  Spatial context: near desk, chair, monitor, fan, lamp ✅
- EXAMPLE_QUERIES: 4 presets for Gradio buttons
- query_scene() and query_scene_streaming() importable by Gradio

### All Outputs on Mac
outputs/scene_graph.json
outputs/dead_zones/dead_zone_summary.png
outputs/dead_zones/dead_zone_report.json
outputs/navigability_map.png  ← hero figure
outputs/objects_3d.json       ← with confidence + provenance

### Next: Session 9 — Gradio App + Deployment
Runs on M4 Pro. Deploy to Hugging Face Spaces (free).