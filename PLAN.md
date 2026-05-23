# RoboScene+ — Full Quality Plan
### Updated: 2026-05-23 | Deadline: 2026-05-25 (2 days)

---

## Where We Are Right Now

| What | Status | Notes |
|---|---|---|
| Video v1 | ✅ | 158s, 73.7° HFOV, 1920×1080, 2fps → 317 frames |
| Video v2 | ✅ | 320s, 1920×1080, 60fps, room_video_v2.MOV — better coverage |
| COLMAP v2 | ✅ | 304/317 registered, 64K points |
| COLMAP v3 | ✅ | 539 frames registered (84%), 103K points — from video v2 |
| Splat v1 | ✅ | 30K steps, random init — superseded |
| Splat v2 | ✅ | 40K steps, point cloud init — full room visible, some floaters |
| Splat v3 | ✅ | 60K steps, 539 views, 3.74M Gaussians — **current best base** |
| Splat v3 clean variants | ❌ | hull+visibility filter cuts bed wall — failed approach |
| Semantic painting (Mac) | ⚠️ | Ran locally with v1 COLMAP poses → labels in wrong positions |
| Semantic painting correct | ⬜ | Need to re-run with colmap_v3 poses on bluestreak |
| GroundingDINO on bluestreak | ❌ | BERT/transformers incompatibility blocks SAM2 pipeline |
| Camera navigation | ⚠️ | v3 coordinate system differs — rotation axis wrong |
| HF Spaces | ⬜ | Not started |
| README | ⬜ | Not started |

**Current viewer**: loads `splat_v3/scene_semantic.splat` (wrong labels)
**Best clean splat**: `splat_v3/scene.splat` (full 3.74M Gaussians, no cleaning)
**Best previous splat**: `splat_video_v2/scene_full.splat` (2.28M, correct camera)

---

## Root Cause Analysis — Why Patches Exist

The patches in the reconstruction have 3 distinct causes:

### Cause 1 — Insufficient Video Coverage (Partially Fixed)
Video v1 was shot at 2fps sweeping around the room. Several areas were not covered:
- **Floor centre**: camera never pointed straight down at the floor
- **Bed wall corners**: camera moved too fast
- **Ceiling**: camera rarely pointed upward

Video v2 (room_video_v2.MOV, 320s) adds ceiling passes and better corner coverage.
COLMAP v3 registered 539/640 frames (84%) vs 304/317 (96%) for v1 — more total
views but some motion blur frames dropped.

### Cause 2 — Single-Criterion Pruning Always Fails (Root Cause of Cleaning Failures)
All previous cleaning attempts used a single criterion:
- Alpha threshold → removes real geometry in thin-coverage areas
- SOR (spatial outlier removal) → floaters are semi-dense clusters, not isolated
- Bbox crop → room is not axis-aligned, crops the bed

**Research finding (TIDI-GS, arXiv:2601.09291)**: Floaters can only be reliably
identified by combining 4 simultaneous weak signals:
1. Low multi-view visibility (seen from <3 cameras)
2. Low opacity
3. Low learned importance score
4. High spatial isolation

A Gaussian is only a floater if ALL 4 signals agree. This prevents false positives
on real geometry (bed wall, thin structures) that share 1-2 floater characteristics.

**Our post-processing approximation** (implementable now without retraining):
- Signal 1: visibility filter — project Gaussians into all COLMAP cameras, count views
- Signal 2+3: alpha threshold (weak, used only after visibility filter)
- Signal 4: convex hull crop of camera positions (not bbox — preserves bed)

### Cause 3 — v3 Coordinate System Changed
Video v2 started from a different position/direction than v1. COLMAP chose a
different canonical orientation. Nerfstudio's scene normalisation scales differently.
Result: v3 splat is rotated relative to v2, breaking the viewer camera constants.

**Fix**: Recalibrate viewer HOME_POS/HOME_LOOK/cameraUp from colmap_v3/transforms.json.
Analysis shows: cameraUp=[0,1,0], HOME_POS=[0.30, 0.31, -0.56], but OrbitControls
rotation axis still wrong — needs screenSpacePanning=false fix in GS library controls.

---

## Current Status

| What | Status | Notes |
|---|---|---|
| Splat v3 PLY | ✅ Done | 539 views, 3.74M Gaussians, 242MB — outputs/splat_v3/scene.ply |
| Splat v3 .splat | ✅ Done | 114MB — outputs/splat_v3/scene.splat |
| COLMAP v3 poses | ✅ Done | data/colmap_v3/transforms.json, sparse/0/ |
| Splat viewer loads v3 | ✅ | SPLAT_CANDIDATES updated, v3 first |
| v3 visual quality | ⚠️ | Floaters present, patches on objects vs v2 |
| Camera navigation | ⚠️ | Eye-level position correct, rotation axis wrong |
| Floater removal | ⬜ | Visibility filter + convex hull crop — next step |
| Semantic labels | ⬜ | Needs re-run with colmap_v3 poses |
| Scene graph + Claude API | ✅ Done | query_scene.py working |
| HF Spaces | ⬜ | After splat cleaning done |
| README | ⬜ | Last step |

**Current best splat**: `outputs/splat_v3/scene.splat` (114MB, 3.74M Gaussians)
**Fallback**: `outputs/splat_video_v2/scene_full.splat` (70MB, 2.28M Gaussians) — full room visible, camera works correctly.

---

## Cleaning Journey — What We Tried

| Attempt | Method | Result | Issue |
|---|---|---|---|
| v1 scene_pruned.splat | alpha>=50 | 928K Gaussians | Bed wall missing |
| v2 scene_pruned.splat | alpha>=80 + crop -3.5,-1.5,-3.0,3.5,1.5,3.5 | 767K Gaussians | Bed still cropped |
| v2 scene_full.splat | alpha>=1, no crop | 2.28M Gaussians | ✅ Full room visible, some floaters |
| v2 scene_clean.splat | SOR k=20 std=2.0 | 2.14M Gaussians | Only 6% removed, floaters remain |
| v2 scene_clean_aggressive | SOR k=20 std=1.0 | 2.06M Gaussians | Only 10% removed |
| v2 scene_final.splat | SOR+crop tight bbox | 1.21M Gaussians | Bed wall missing again |
| v2 scene_room.splat | camera bbox + 1.5m pad | 1.8M Gaussians | Bed still clipped |
| v3 scene.splat | 60K steps, 539 views | 3.74M Gaussians | ⚠️ More floaters, camera broken |

**Root cause of all failures**: Single-criterion pruning cannot distinguish floaters
from real geometry. Floaters are semi-dense clusters with similar alpha/scale to
real surfaces. Need multi-criterion approach (TIDI-GS research, 2026).

**Best current result**: `scene_full.splat` — full room visible, 70MB, some external floaters acceptable for deadline.

**The 4 blocking issues** are documented below with root causes and fixes. The new splat (304 views) is trained and downloaded. Now we fix quality, navigation, and semantics.

---

## The 4 Problems — Root Causes & Fixes

### Problem 1 — Object Labels in Wrong Positions
**Root cause:** `objects_3d.json` was computed using old VGGT/MASt3R poses (different coordinate system). The new splat uses COLMAP video poses. The centroids in objects_3d.json do not match where the Gaussians actually are.

**Fix:** Re-run `paint_semantic_gaussians.py` and `lift_semantics_3d.py` using the new COLMAP poses from `data/colmap_video/transforms.json`. Also fix the filename mismatch: semantic JSONs use `frame_0001.json` (4 digits) but COLMAP images are `frame_00001.jpg` (5 digits).

---

### Problem 2 — Camera Navigation Locked to Top-Down
**Root cause:** The viewer's `cameraUp`, `initialCameraPosition` and `initialCameraLookAt` are set for the old splat's coordinate system. The new COLMAP splat has a different orientation. The intro orbit sweeps a fixed arc that doesn't match the new scene centre.

**Fix:** Recalibrate the viewer's camera constants to match the new COLMAP coordinate system. Find correct home position by sampling the transforms.json camera positions.

---

### Problem 3 — Floaters Outside Room + Low Object Opacity
**Root cause (floaters):** Current prune threshold (alpha ≥ 50) keeps 928K Gaussians but many are outside the room (nerfstudio bounding sphere haze, ceiling fog). Need tighter prune + world-space crop to room bbox.

**Root cause (low opacity):** nerfstudio's splatfacto had **no point cloud initialisation** — the training log said "Warning: load_3D_points set to true but no point cloud found". The model started from random Gaussians and didn't fully converge for thin/small objects in 30k steps.

**Fix A — immediate:** Tighten prune threshold to alpha ≥ 80 and add world-space crop to remove Gaussians outside the room bounding box.

**Fix B — better (re-train on bluestreak):** Re-train with COLMAP point cloud as initialisation. The colmap.db has 64K 3D points — these give the model a correct starting geometry so opacity builds up faster.

---

### Problem 4 — Floor Gap in Centre of Room
**Root cause:** The video never pointed at the floor centre. At 2fps the camera always pointed at walls/objects. COLMAP registered 304/317 frames but none had floor-centre coverage → no Gaussians initialised there → blank patch.

**Fix (short term):** Accept the gap — document in README as a known data limitation.

**Fix (proper, optional):** Record a supplementary 30-second video pointing the camera down at the floor and slow-panning across it. Merge these frames into the COLMAP reconstruction and retrain.

---

## Full Quality Plan — Updated

```
Phase A ── Re-record video (better coverage)          ✅ DONE (room_video_v2.MOV)
    │
Phase B ── Upload + retrain on bluestreak             ✅ DONE (splat_v3, 539 views, 60K steps)
    │
Phase C ── Multi-criterion floater removal            ⬜ IN PROGRESS
    │         visibility filter + convex hull crop
    │         script: scripts/clean_splat_visibility.py
    │
Phase D ── Fix viewer camera for v3 coordinate system ⬜ NEXT
    │         screenSpacePanning fix in OrbitControls
    │
Phase E ── Re-run semantics with colmap_v3 poses      ⬜ After C+D
    │
Phase F ── Deploy to HF Spaces                        ⬜
    │
Phase G ── README + Polish                            ⬜
```

---

## Current Blocker — GroundingDINO / Transformers Incompatibility on Bluestreak

### Status: ⚠️ BLOCKED

### Symptom
```
AttributeError: 'BertModel' object has no attribute 'get_head_mask'
  File groundingdino/models/GroundingDINO/bertwarper.py line 29
  self.get_head_mask = bert_model.get_head_mask
```

### Root Cause Analysis
GroundingDINO's `BertModelWarper` tries to grab `get_head_mask` as an instance
attribute from `BertModel`. In older transformers (<=4.35) this was a method on
the instance. In newer transformers (>=4.36) it moved to the base class and is
no longer directly accessible as an attribute in the same way.

Three compounding issues:
1. **Wrong env**: `colmap_env` (Python 3.14) has a broken groundingdino install
   that keeps getting imported instead of `roboscene_env` (Python 3.11)
2. **Transformers version mismatch**: the transformers version in roboscene_env
   may be too new for the groundingdino version installed
3. **torch CUDA mismatch**: needed torch reinstall for CUDA 12.6 → cu121
   (already fixed — torch 2.5.1+cu121 now works)

### Things Already Tried
| Attempt | Result |
|---|---|
| `pip install groundingdino-py` | Installs into colmap_env (Python 3.14) |
| `pip install transformers==4.38.2 --force-reinstall` | Still same error |
| `git clone + pip install -e .` from source | Same error |
| Reinstall torch for cu121 | ✅ Fixed torch CUDA issue |
| Use explicit `/roboscene_env/bin/python3` in shell script | Still loads colmap_env gdino |

### Fixes to Try Next
1. **Check actual transformers version installed** in roboscene_env:
   ```bash
   /scratch0/jrameshs/roboscene_env/bin/pip show transformers | grep Version
   ```
   If >4.35, downgrade: `pip install "transformers==4.33.0" --force-reinstall`

2. **Patch bertwarper.py directly** (most reliable fix):
   ```bash
   GDINO_PATH=$(/scratch0/jrameshs/roboscene_env/bin/python3 -c \
     "import groundingdino; print(groundingdino.__file__)")
   BERTWARPER=$(dirname $GDINO_PATH)/models/GroundingDINO/bertwarper.py
   # Line 29: change attribute grab to method bind
   sed -i 's/self.get_head_mask = bert_model.get_head_mask/self.get_head_mask = bert_model.get_head_mask if hasattr(bert_model, "get_head_mask") else bert_model.base_model.get_head_mask/' $BERTWARPER
   ```

3. **Skip GroundingDINO entirely** — use existing semantic JSONs from v1
   (317 frames, 4-digit names) and just re-run paint + lift with colmap_v3 poses.
   The semantic JSONs cover the same room objects. Frame name mismatch (4→5 digit)
   can be handled by a mapping script.

4. **Use OWL-ViT instead of GroundingDINO** — alternative open-vocabulary detector
   that doesn't have the BERT compatibility issue.

### Recommended Next Step
Option 3 (use existing semantic JSONs) is the fastest path to unblocking:
- We already have 317 semantic JSONs in `outputs/semantic/`
- We need to map `frame_0001` → nearest `frame_00001` in colmap_v3
- Run paint_semantic_gaussians.py with colmap_v3 poses
- This gives semantic labels on splat_v3 within 30 min, no GPU needed

---

## Phase C — Multi-Criterion Floater Removal (1-2h, Mac)

### Why previous methods failed
SOR, alpha threshold, and bbox crop all use a single criterion.
Floaters are semi-dense clusters with similar characteristics to real surfaces.
The TIDI-GS paper (arXiv:2601.09291) proves you need 4 simultaneous signals.

### Our post-processing approximation of TIDI-GS
We cannot run the full TIDI framework without retraining, but we can approximate
the two most powerful signals as post-processing steps:

**Signal 1 — Multi-view visibility filter** (most powerful)
Project every Gaussian into every COLMAP camera. Count how many cameras it is
visible in (inside frustum + alpha > threshold). Real surfaces are visible from
many cameras (the room was swept 2-3 times). Floaters outside the room are only
visible from 1-2 cameras before exiting the frustum.
- Remove Gaussians visible in fewer than N_min views (start with N_min=3)
- This is geometry-aware: bed wall Gaussians are seen from many views → safe

**Signal 2 — Convex hull crop** (replaces failed bbox crop)
The bbox crop failed because the room is not axis-aligned. The convex hull of
camera positions follows the actual room shape. Any Gaussian outside the
camera convex hull + 1.5m padding cannot be real room geometry (cameras were
always inside the room).
- Compute scipy.spatial.ConvexHull of all COLMAP camera positions
- Inflate each face outward by 1.5m
- Remove Gaussians outside inflated hull
- Preserves the bed (cameras walked near the bed → inside hull)

**Signal 3 — Alpha threshold** (last, conservative)
- Remove Gaussians with opacity < 0.005 after sigmoid (near-invisible)

### Step C.1 — Build and run the cleaning script
```bash
python3 scripts/clean_splat_visibility.py \
  --input   outputs/splat_v3/scene.ply \
  --output  outputs/splat_v3/scene_clean.ply \
  --transforms data/colmap_v3/transforms.json \
  --min_views 3 \
  --hull_padding 1.5 \
  --alpha_min 0.005
```

### Step C.2 — Convert and preview
```bash
python3 scripts/convert_to_splat.py \
  --input  outputs/splat_v3/scene_clean.ply \
  --output outputs/splat_v3/scene_clean.splat
python3 open_viewer.py
```

### Step C.3 — Tune if needed
If bed wall is missing → lower --min_views to 2
If floaters remain → raise --min_views to 5, tighten --hull_padding to 1.2
Target: 1.5-2.5M Gaussians (vs 3.74M raw)

---

## Phase D — Fix Viewer Camera for v3 (30 min, Mac)

### What is known
- cameraUp = [0, 1, 0] ✅ confirmed from floor camera analysis
- HOME_POS = [0.30, 0.31, -0.56] ✅ actual camera position from transforms.json
- HOME_LOOK = [0, 0, 0] ✅ scene centre in normalised space
- Problem: OrbitControls rotation axis is wrong (rotates around screen-Z not world-Y)

### Root cause
The GS library's built-in controls initialise OrbitControls with a default up vector
that doesn't match our cameraUp setting. Need to force:
  ctrl.object.up.set(0, 1, 0)
  ctrl.screenSpacePanning = false
  ctrl.update()
after controls initialise (async — needs setTimeout).

### Fix already applied
fixControls() function added to index.html, called at 0ms, 500ms, 1500ms.
If still broken: inspect viewer.controls in console after page loads to find
the correct OrbitControls handle.

---

## Original Full Quality Plan — In Order

```
Phase A ── Re-record video (better coverage)          ✅ DONE
    │
Phase B ── Upload + retrain on bluestreak             ✅ DONE
    │
Phase C ── Download + clean splat on Mac              (30 min, Mac)
    │         scene_full → final best result
    │
Phase D ── Re-run Grounded SAM2 semantics             (1h, Mac/bluestreak)
    │         with new COLMAP poses, fix 4→5 digit names
    │
Phase E ── Deploy to HF Spaces                        (1h, Mac)
    │
Phase F ── README + Polish                            (1h, Mac)
```

**Time budget**: 3h waiting (GPU) + 4h active work = fits in deadline.

---

## Phase A — Re-record Video for Full Coverage (20 min)

This is the single most impactful thing you can do. Better input = better splat.

### What to record
Record 4 passes in sequence, saving as one continuous video:

**Pass 1 — Perimeter eye-level** (already have, re-record better):
- Start at door, walk slowly around the entire room perimeter
- Camera at eye level, pointing slightly downward (30° below horizontal)
- Slow continuous pan — 1 step per second
- Duration: ~90 seconds

**Pass 2 — Floor coverage** (NEW — fixes floor patches):
- Point camera straight DOWN at the floor
- Walk a grid pattern across the floor: left→right, step forward, right→left
- Cover every 50cm of floor area
- Duration: ~60 seconds

**Pass 3 — Corner coverage** (NEW — fixes corner patches):
- Stand in each corner, point camera at the corner junction (wall+wall+floor)
- Slowly pan from floor to ceiling in each corner
- 4 corners × 10 seconds = ~40 seconds

**Pass 4 — Object close-ups** (NEW — improves object detail):
- Walk slowly past desk, shelf, bed with camera 50cm from objects
- This gives nerfstudio high-resolution texture detail
- Duration: ~40 seconds

**Total**: ~4 minutes, 1080p 30fps, main 1x lens (NOT ultrawide)
**Result**: ~480 frames at 2fps → near-complete room coverage

### Recording tips
- Keep phone steady — use both hands or a tripod
- Move SLOWLY — faster than 0.3m/s causes motion blur
- Good lighting — turn on all room lights
- Keep the same exposure throughout — don't let auto-exposure flicker

---

## Phase B — Upload + Retrain on Bluestreak (3h)

### Step B.1 — Upload new video
```bash
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  ~/Downloads/room_video_v2.MOV \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/raw/room_video_v2.MOV
```

### Step B.2 — SSH to bluestreak and initialise
```bash
ssh -J jrameshs@knuckles.cs.ucl.ac.uk jrameshs@bluestreak.cs.ucl.ac.uk
bash
source /opt/Python/Python-3.11.5_Setup.csh
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
git pull
```

### Step B.3 — Extract frames from new video
```bash
mkdir -p data/video_frames_v2
ffmpeg -y -i data/raw/room_video_v2.MOV \
    -vf "fps=2,scale=1920:1080" \
    -q:v 2 \
    data/video_frames_v2/frame_%05d.jpg
ls data/video_frames_v2/ | wc -l  # expect ~480
```

### Step B.4 — Run COLMAP on new frames
```bash
mkdir -p data/colmap_v3/sparse
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH

# Feature extraction
colmap feature_extractor \
    --database_path data/colmap_v3/colmap.db \
    --image_path data/video_frames_v2 \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 0

# Matching
colmap exhaustive_matcher \
    --database_path data/colmap_v3/colmap.db \
    --SiftMatching.use_gpu 0

# Reconstruction
colmap mapper \
    --database_path data/colmap_v3/colmap.db \
    --image_path data/video_frames_v2 \
    --output_path data/colmap_v3/sparse
```
Expect: 400+ cameras registered (>80% of frames).

### Step B.5 — Convert to nerfstudio format
```bash
/scratch0/jrameshs/roboscene_env/bin/python3 -c "
import json, numpy as np, pycolmap
from pathlib import Path

recon = pycolmap.Reconstruction('data/colmap_v3/sparse/0')
cameras = recon.cameras
images = recon.images

frames = []
for img_id, img in images.items():
    cam = cameras[img.camera_id]
    pose = img.cam_from_world()
    R = pose.rotation.matrix()
    t = np.array(pose.translation)
    c2w = np.eye(4)
    c2w[:3,:3] = R.T
    c2w[:3, 3] = -R.T @ t
    c2w[:,1] *= -1
    c2w[:,2] *= -1
    frames.append({'file_path': 'images/' + img.name, 'transform_matrix': c2w.tolist()})

cam = list(cameras.values())[0]
params = cam.params
transforms = {'fl_x': float(params[0]), 'fl_y': float(params[0]),
    'cx': float(params[1]), 'cy': float(params[2]),
    'w': cam.width, 'h': cam.height, 'camera_model': 'OPENCV', 'frames': frames}

with open('data/colmap_v3/transforms.json', 'w') as f:
    json.dump(transforms, f, indent=2)
print(f'Wrote {len(frames)} frames')
"

# Symlink images
ln -sf /scratch0/jrameshs/roboscene-plus/data/video_frames_v2 \
       /scratch0/jrameshs/roboscene-plus/data/colmap_v3/images
```

### Step B.6 — Generate point cloud PLY
```bash
/scratch0/jrameshs/roboscene_env/bin/python3 -c "
import pycolmap, numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path

recon = pycolmap.Reconstruction('data/colmap_v3/sparse/0')
pts = np.array([[p.xyz[0], p.xyz[1], p.xyz[2]] for p in recon.points3D.values()])
rgb = np.array([[int(p.color[0]), int(p.color[1]), int(p.color[2])] for p in recon.points3D.values()])
vertex = np.zeros(len(pts), dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
vertex['x'],vertex['y'],vertex['z'] = pts[:,0],pts[:,1],pts[:,2]
vertex['red'],vertex['green'],vertex['blue'] = rgb[:,0],rgb[:,1],rgb[:,2]
PlyData([PlyElement.describe(vertex,'vertex')]).write('data/colmap_v3/sparse/0/points3D.ply')
print('Saved', len(pts), 'seed points')
"
```

### Step B.7 — Train splatfacto v3 (60K steps)
```bash
nohup ns-train splatfacto \
    --data data/colmap_v3 \
    --output-dir outputs/splat_v3 \
    --max-num-iterations 60000 \
    --pipeline.model.cull-alpha-thresh 0.005 \
    --pipeline.model.densify-grad-thresh 0.0001 \
    --pipeline.model.use-scale-regularization True \
    --pipeline.model.max-gauss-ratio 10.0 \
    --pipeline.model.densify-until-iter 50000 \
    --viewer.quit-on-train-completion True \
    > logs/splat_v3.log 2>&1 &
tail -f logs/splat_v3.log
```

Key improvements vs v2:
- `densify-grad-thresh 0.0001` (was 0.0002) → 2× more aggressive densification → fills patches
- `densify-until-iter 50000` (was default ~15000) → keeps densifying for longer
- 60K steps vs 40K → more time to converge thin-coverage areas
- 480 frames vs 304 → 58% more views → better coverage

### Step B.8 — Export PLY
```bash
/scratch0/jrameshs/roboscene_env/bin/python3 -c "
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
import numpy as np
from plyfile import PlyData, PlyElement

# Find config
import glob
config = sorted(glob.glob('outputs/splat_v3/colmap_v3/splatfacto/*/config.yml'))[-1]
print('Config:', config)
_, pipeline, _, _ = eval_setup(Path(config))
m = pipeline.model
n = len(m.means)
means = m.means.detach().cpu().numpy()
fdc = m.features_dc.detach().cpu().numpy()
ops = m.opacities.detach().cpu().numpy()
scs = m.scales.detach().cpu().numpy()
qts = m.quats.detach().cpu().numpy()
vertex = np.zeros(n, dtype=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')])
vertex['x'],vertex['y'],vertex['z']=means[:,0],means[:,1],means[:,2]
vertex['f_dc_0'],vertex['f_dc_1'],vertex['f_dc_2']=fdc[:,0],fdc[:,1],fdc[:,2]
vertex['opacity']=ops[:,0]
vertex['scale_0'],vertex['scale_1'],vertex['scale_2']=scs[:,0],scs[:,1],scs[:,2]
vertex['rot_0'],vertex['rot_1'],vertex['rot_2'],vertex['rot_3']=qts[:,0],qts[:,1],qts[:,2],qts[:,3]
out = Path('outputs/splat_v3/scene.ply')
PlyData([PlyElement.describe(vertex,'vertex')]).write(str(out))
print(f'Saved {n:,} Gaussians → {out}  ({out.stat().st_size/1024/1024:.1f} MB)')
"
```

### Step B.9 — Download
```bash
# On Mac:
mkdir -p ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v3
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_v3/scene.ply \
  ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_v3/scene.ply

mkdir -p ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/sparse/0
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  "jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_v3/sparse/0/" \
  ~/Downloads/3D-Spatial-Reconstruction/data/colmap_v3/sparse/0/
```

---

## Phase C — Convert + Clean Splat on Mac (30 min)

### Step C.1 — Convert to .splat
```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 scripts/convert_to_splat.py \
  --input  outputs/splat_v3/scene.ply \
  --output outputs/splat_v3/scene.splat
```

### Step C.2 — Load full splat first to verify quality
Add `outputs/splat_v3/scene.splat` as first candidate in `app/static/index.html`.
Launch viewer: `python3 open_viewer.py`
Check: are patches gone? Is the bed wall visible? Are objects clear?

### Step C.3 — Apply density-based crop only if needed
If floaters are still visible, apply the density crop:
```bash
python3 scripts/prune_splat.py \
  --input  outputs/splat_v3/scene.splat \
  --output outputs/splat_v3/scene_pruned.splat \
  --alpha_min 1
```
**Do NOT use --crop** unless you have verified the exact bbox from the new splat's
Gaussian density. Re-run `scripts/analyse_splat.py` first.

---

## Phase D — Semantic Segmentation with New Poses (1h)

### Why semantics need re-running
1. `objects_3d.json` centroids are in old VGGT coordinate system
2. Semantic JSONs use 4-digit names (`frame_0001.json`) but COLMAP uses 5-digit (`frame_00001.jpg`)
3. After Phase B we'll have new COLMAP poses (colmap_v3) that are more accurate

### Step D.1 — Fix frame name mapping in paint_semantic_gaussians.py
The script needs to map `frame_0001` → `frame_00001` when looking up semantic JSONs.
Add this to `paint_semantic_gaussians.py`:
```python
# Normalise frame name: try both 4-digit and 5-digit
def find_semantic_json(semantic_dir, image_name):
    stem = Path(image_name).stem  # e.g. frame_00001
    # Try exact match first
    p = semantic_dir / f"{stem}.json"
    if p.exists(): return p
    # Try 4-digit version (strip leading zero)
    stem4 = re.sub(r'frame_(\d+)', lambda m: f'frame_{int(m.group(1)):04d}', stem)
    p = semantic_dir / f"{stem4}.json"
    if p.exists(): return p
    return None
```

### Step D.2 — Re-run semantic painting
```bash
python3 scripts/paint_semantic_gaussians.py \
  --splat_ply    outputs/splat_v3/scene.ply \
  --semantic_dir outputs/semantic \
  --cameras_bin  data/colmap_v3/sparse/0/cameras.bin \
  --images_bin   data/colmap_v3/sparse/0/images.bin \
  --output_ply   outputs/scene_semantic_v3.ply \
  --conf_threshold 0.20 \
  --min_votes 2
```
Expected: 40-60% of Gaussians labeled (vs 14.5% before).

### Step D.3 — Re-lift semantics to 3D
```bash
python3 scripts/lift_semantics_3d.py \
  --semantic_dir outputs/semantic \
  --cameras_bin  data/colmap_v3/sparse/0/cameras.bin \
  --images_bin   data/colmap_v3/sparse/0/images.bin \
  --output_dir   outputs/
```

### Step D.4 — Rebuild scene graph
```bash
python3 scripts/build_scene_graph.py
```

### Step D.5 — Convert semantic PLY
```bash
python3 scripts/convert_to_splat.py \
  --input  outputs/scene_semantic_v3.ply \
  --output outputs/scene_semantic_v3.splat

python3 scripts/prune_splat.py \
  --input  outputs/scene_semantic_v3.splat \
  --output outputs/scene_semantic_v3_pruned.splat \
  --alpha_min 1
```

---

## Phase E — Deploy to HuggingFace Spaces (1h)

### Step E.1 — Upload splat to HF Dataset
```bash
pip3 install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
  path_or_fileobj='outputs/scene_semantic_v3_pruned.splat',
  path_in_repo='scene_semantic_pruned.splat',
  repo_id='JesonRamesh/roboscene-data',
  repo_type='dataset',
)
print('Done')
"
```

### Step E.2 — Update app.py for HF Spaces
- Set `SPLAT_URL` from HF Dataset URL
- Set `ANTHROPIC_API_KEY` as HF Space secret

### Step E.3 — Push
```bash
git add app/ outputs/scene_graph.json outputs/objects_3d.json
git commit -m 'v3 splat + semantic pipeline'
git push
```

---

## Phase F — README + Polish (1h)

### Sections
1. Demo screenshot/GIF of the 3D viewer
2. What it does (robot scene understanding framing)
3. Quick start: `python3 open_viewer.py`
4. Pipeline: Video → COLMAP → Splatfacto → SAM2 → Scene Graph → Claude
5. Novel contribution: confidence-aware Gaussian tagging
6. Limitations: floor gap (if still present), semantic coverage %

### Final checks
```
[ ] Viewer loads in Chrome, Firefox, Safari
[ ] Camera starts inside room at eye level
[ ] WASD navigation works
[ ] Object labels on correct objects
[ ] All 4 robot query examples work
[ ] HF Space loads publicly
[ ] GitHub repo public
```

---

## Timeline

| Day | Task | Where | Active time |
|---|---|---|---|
| **Today (22 May)** | Re-record video (Phase A) | Phone | 20 min |
| **Today (22 May)** | Upload + launch retrain (Phase B) | bluestreak | 30 min active + 3h waiting |
| **22 May evening** | Download v3 + convert + preview (Phase C) | Mac | 30 min |
| **23 May** | Re-run semantics (Phase D) | Mac | 1h |
| **23 May** | Deploy to HF Spaces (Phase E) | Mac | 1h |
| **24 May** | README + final polish (Phase F) | Mac | 2h |
| **25 May** | Buffer / submit | — | — |

---

## If You Cannot Re-record Video (Fallback)

If re-recording is not possible, use `scene_full.splat` as the final splat.
It shows the full room with some floaters — acceptable for submission.
Focus time on Phases D-F instead.

Document in README: *"Patches in the floor centre and some corners are due to
video coverage gaps during data collection. A dedicated floor-coverage pass
would eliminate these."*

---

## Quick Reference

```bash
# Launch viewer
python3 open_viewer.py

# Analyse splat quality
python3 scripts/analyse_splat.py

# Convert + prune
python3 scripts/convert_to_splat.py --input <file.ply> --output <file.splat>
python3 scripts/prune_splat.py --input <file.splat> --output <file_pruned.splat> --alpha_min 1

# SOR floater removal
python3 scripts/clean_splat.py --input <file.splat> --output <file_clean.splat> --k 20 --std_ratio 1.5

# Re-run semantics
python3 scripts/paint_semantic_gaussians.py --cameras_bin data/colmap_v3/sparse/0/cameras.bin --images_bin data/colmap_v3/sparse/0/images.bin
python3 scripts/lift_semantics_3d.py
python3 scripts/build_scene_graph.py

# Query scene
python3 scripts/query_scene.py

# Bluestreak reconnect
ssh -J jrameshs@knuckles.cs.ucl.ac.uk jrameshs@bluestreak.cs.ucl.ac.uk
bash && source /opt/Python/Python-3.11.5_Setup.csh
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
```

---

## Phase 2D — Better Floater Removal (Do This Next, 45 min)

The challenge: floaters form semi-dense clusters outside the room so SOR
cannot distinguish them. The correct approach is **voxel-based density filtering**
combined with a **loose spatial crop that preserves the bed**.

### Step 2D.1 — Find the true room extent from COLMAP camera positions
The COLMAP cameras define where the photographer stood — always inside the room.
The room extent = camera bbox + 1.5m padding on each side:
```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 -c "
import json, numpy as np
with open('data/colmap_video/transforms.json') as f:
    d = json.load(f)
pos = np.array([f['transform_matrix'] for f in d['frames']])[:,:3,3]
print('Camera bbox:')
print('  X:', pos[:,0].min().round(2), 'to', pos[:,0].max().round(2))
print('  Y:', pos[:,1].min().round(2), 'to', pos[:,1].max().round(2))
print('  Z:', pos[:,2].min().round(2), 'to', pos[:,2].max().round(2))
pad = 1.5
print('Suggested crop (camera bbox + 1.5m padding):')
print(f'  {pos[:,0].min()-pad:.2f},{pos[:,1].min()-pad:.2f},{pos[:,2].min()-pad:.2f},{pos[:,0].max()+pad:.2f},{pos[:,1].max()+pad:.2f},{pos[:,2].max()+pad:.2f}')
"
```

### Step 2D.2 — Apply loose crop based on camera bbox
Use the padding bbox from Step 2D.1 instead of a manually guessed bbox:
```bash
python3 scripts/prune_splat.py \
  --input  outputs/splat_video_v2/scene_full.splat \
  --output outputs/splat_video_v2/scene_room.splat \
  --alpha_min 1 \
  --crop="<xmin>,<ymin>,<zmin>,<xmax>,<ymax>,<zmax>"
```

### Step 2D.3 — Two-pass SOR on the cropped result
After cropping to room, floaters are now truly isolated → SOR works much better:
```bash
python3 scripts/clean_splat.py \
  --input  outputs/splat_video_v2/scene_room.splat \
  --output outputs/splat_video_v2/scene_room_clean.splat \
  --k 20 --std_ratio 1.5
```

### Step 2D.4 — Preview and compare
Update SPLAT_CANDIDATES in index.html to load scene_room_clean.splat first.
Check: full bed visible + less fog outside room.

---

## Phase 2B — Fix Viewer Camera & Navigation (30 min)

### Step 2B.1 — Find the correct home camera position
Sample the new COLMAP transforms.json to find where cameras actually are:
```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 -c "
import json, numpy as np
with open('data/colmap_video/transforms.json') as f:
    d = json.load(f)
positions = np.array([fr['transform_matrix'] for fr in d['frames']])[:, :3, 3]
print('Scene centre:', positions.mean(axis=0).round(3))
print('Scene range X:', positions[:,0].min().round(2), 'to', positions[:,0].max().round(2))
print('Scene range Y:', positions[:,1].min().round(2), 'to', positions[:,1].max().round(2))
print('Scene range Z:', positions[:,2].min().round(2), 'to', positions[:,2].max().round(2))
"
```

### Step 2B.2 — Update viewer constants
Edit `app/static/index.html` — update `HOME_POS`, `HOME_LOOK`, `cameraUp`,
and `INTRO_CTR`/`INTRO_R` based on the output above.

### Step 2B.3 — Trial and error in browser console
While viewer is running, use the browser console:
```javascript
// Check current camera position
viewer.camera.position
viewer.controls.target
// Manually fly to a test position
flyTo([x, y, z], [tx, ty, tz])
```

---

## Phase 2C — Tighten Prune & Crop Floaters (20 min)

### Step 2C.1 — Re-prune with higher alpha threshold
```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 scripts/prune_splat.py \
  --input  outputs/splat_video_v1/scene.splat \
  --output outputs/splat_video_v1/scene_pruned_80.splat \
  --alpha-threshold 80
```

### Step 2C.2 — Check prune_splat.py supports --alpha-threshold flag
If not, we add it. Check with:
```bash
python3 scripts/prune_splat.py --help
```

### Step 2C.3 — Update viewer to load new pruned file
Add `outputs/splat_video_v1/scene_pruned_80.splat` to `SPLAT_CANDIDATES` at top of list.

---

## Phase 2D — Re-train with Point Cloud Init (Optional, Bluestreak)

Only do this if Phase 2C still leaves objects looking transparent/sparse.

### On bluestreak:
```bash
# Convert COLMAP DB points to PLY
python3 -c "
import pycolmap, numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
recon = pycolmap.Reconstruction('data/colmap_video/sparse/0')
pts = np.array([[p.xyz[0], p.xyz[1], p.xyz[2]] for p in recon.points3D.values()])
rgb = np.array([[int(p.color[0]), int(p.color[1]), int(p.color[2])] for p in recon.points3D.values()])
vertex = np.zeros(len(pts), dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
vertex['x'],vertex['y'],vertex['z'] = pts[:,0],pts[:,1],pts[:,2]
vertex['red'],vertex['green'],vertex['blue'] = rgb[:,0],rgb[:,1],rgb[:,2]
PlyData([PlyElement.describe(vertex,'vertex')]).write('data/colmap_video/sparse/0/points3D.ply')
print('Saved', len(pts), 'points')
"

# Re-train with point cloud initialisation
nohup ns-train splatfacto \
  --data data/colmap_video \
  --output-dir outputs/splat_video_v2 \
  --max-num-iterations 30000 \
  --pipeline.model.random-init False \
  --viewer.quit-on-train-completion True \
  > logs/splat_v2.log 2>&1 &
tail -f logs/splat_v2.log
```

---

## Phase 3 — Re-run Semantics with New COLMAP Poses (1h)

### Key issues to fix
1. Semantic JSONs: `frame_0001.json` (4-digit) but COLMAP images: `frame_00001.jpg` (5-digit)
2. `objects_3d.json` centroids are in old VGGT coordinate system
3. Need COLMAP sparse binaries (`cameras.bin`, `images.bin`) — not downloaded yet

### Step 3.1 — Download COLMAP sparse binaries from bluestreak
```bash
mkdir -p ~/Downloads/3D-Spatial-Reconstruction/data/colmap_video/sparse/0
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  "jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_video/sparse/0/cameras.bin" \
  ~/Downloads/3D-Spatial-Reconstruction/data/colmap_video/sparse/0/
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  "jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_video/sparse/0/images.bin" \
  ~/Downloads/3D-Spatial-Reconstruction/data/colmap_video/sparse/0/
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  "jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_video/sparse/0/points3D.bin" \
  ~/Downloads/3D-Spatial-Reconstruction/data/colmap_video/sparse/0/
```

### Step 3.2 — Re-paint semantic Gaussians
```bash
python3 scripts/paint_semantic_gaussians.py \
  --splat_ply    outputs/splat_video_v1/scene.ply \
  --semantic_dir outputs/semantic \
  --cameras_bin  data/colmap_video/sparse/0/cameras.bin \
  --images_bin   data/colmap_video/sparse/0/images.bin \
  --output_ply   outputs/scene_semantic_v2.ply \
  --conf_threshold 0.20 \
  --min_votes 2
```
Note: `paint_semantic_gaussians.py` needs a fix for the 4→5 digit frame name mapping.

### Step 3.3 — Re-lift semantics to 3D
```bash
python3 scripts/lift_semantics_3d.py \
  --semantic_dir outputs/semantic \
  --cameras_bin  data/colmap_video/sparse/0/cameras.bin \
  --images_bin   data/colmap_video/sparse/0/images.bin \
  --output_dir   outputs/
```

### Step 3.4 — Rebuild scene graph
```bash
python3 scripts/build_scene_graph.py
```

### Step 3.5 — Convert + prune semantic PLY
```bash
python3 scripts/convert_to_splat.py \
  --input  outputs/scene_semantic_v2.ply \
  --output outputs/scene_semantic_v2.splat

python3 scripts/prune_splat.py \
  --input  outputs/scene_semantic_v2.splat \
  --output outputs/scene_semantic_v2_pruned.splat
```

---

## Phase 4 — Deploy to HuggingFace Spaces (1-2h)

### Step 4.1 — Upload splat to HF Dataset
```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
  path_or_fileobj='outputs/scene_semantic_v2_pruned.splat',
  path_in_repo='scene_semantic_pruned.splat',
  repo_id='JesonRamesh/roboscene-data',
  repo_type='dataset',
)
print('Done')
"
```

### Step 4.2 — Update app.py for HF Spaces
Edit `app/app.py` to load splat from HF Dataset URL.
Set `SPLAT_URL` from env var `HF_SPLAT_URL` (set as HF Space secret).

### Step 4.3 — Push to HF Spaces
```bash
git add app/ outputs/scene_graph.json outputs/objects_3d.json
git commit -m "Retrained splat + fixed semantics"
git push
```

---

## Phase 5 — README + Polish (1h)

### Step 5.1 — Write README.md
1. Demo screenshot of the 3D viewer
2. What it does (robot framing, one paragraph)
3. Quick start: `python open_viewer.py` in 3 commands
4. Pipeline diagram
5. Novel contribution: confidence-aware Gaussian tagging
6. Limitations: floor gap, 14%→50% semantic labeling, bbox inaccuracy

### Step 5.2 — Final checks
```
[ ] Viewer loads in Chrome, Firefox, Safari
[ ] Camera starts inside room, can navigate freely in all directions
[ ] Object labels appear on correct objects
[ ] All 4 robot query examples work
[ ] ANTHROPIC_API_KEY set as HF Space secret
[ ] HF Space loads publicly without login
[ ] GitHub repo is public
```

---

## Timeline

| Day | Task | Time |
|---|---|---|
| **Today (21 May)** | Fix viewer camera + prune floaters (2B+2C) | 1h |
| **22 May** | Download COLMAP sparse + re-run semantics (Phase 3) | 1-2h |
| **23 May** | HF Spaces deployment (Phase 4) | 2h |
| **24 May** | README + final testing (Phase 5) | 2h |
| **25 May** | Buffer / submit | — |

---

## Known Limitations (Document in README)

1. **Floor gap** — centre of floor has no Gaussians (video never pointed there). Fix requires re-recording.
2. **Semantic coverage** — 14.5% labeled with old poses, expected 30-50% after Phase 3 re-run.
3. **Object bbox accuracy** — bounding boxes are approximate (depth-lifted from 2D masks).
4. **No real-time updates** — scene graph is static, built at training time.

---

## Quick Reference — Key Commands

```bash
# Launch viewer
python3 open_viewer.py

# Convert + prune new splat
python3 scripts/convert_to_splat.py --input outputs/splat_video_v1/scene.ply --output outputs/splat_video_v1/scene.splat
python3 scripts/prune_splat.py --input outputs/splat_video_v1/scene.splat --output outputs/splat_video_v1/scene_pruned.splat

# Re-run semantics (needs COLMAP sparse binaries)
python3 scripts/paint_semantic_gaussians.py --cameras_bin data/colmap_video/sparse/0/cameras.bin --images_bin data/colmap_video/sparse/0/images.bin
python3 scripts/lift_semantics_3d.py
python3 scripts/build_scene_graph.py

# Query scene
python3 scripts/query_scene.py
```

---

## If You Need to Re-train on Bluestreak

```bash
# Reconnect
ssh -J jrameshs@knuckles.cs.ucl.ac.uk jrameshs@bluestreak.cs.ucl.ac.uk
bash
source /opt/Python/Python-3.11.5_Setup.csh
export PATH=/scratch0/jrameshs/colmap_env/bin:$PATH
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
git pull
```

---

## Phase 1 — GPU Retrain ✅ COMPLETE
### (Archived — kept for reference)

**Status**: ✅ Complete. 304/317 frames registered, 3.1M Gaussians, scene.ply downloaded.
**Results**: Room structure clearly visible. 4 quality issues identified above (Problems 1-4).

### Step 1.1 — Book bluestreak
Go to https://mydesk.cs.ucl.ac.uk and book a 4-hour slot.
Set a phone alarm for 30 min before the slot ends so you don't lose your outputs.

### Step 1.2 — Upload the video (run on Mac, ~5 min)
```bash
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  ~/Downloads/3D-Spatial-Reconstruction/data/raw/room_video.MOV \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/raw/
```
The video is 444 MB. This takes ~5 min on UCL network.

### Step 1.3 — SSH to bluestreak and run the job
```bash
# Connect
ssh bluestreak

# Switch to bash + activate Python (always do these two first)
bash
source /opt/Python/Python-3.11.5_Setup.csh
source /scratch0/jrameshs/roboscene_env/bin/activate

# Pull latest code (includes the new run_video_splat.sh)
cd /scratch0/jrameshs/roboscene-plus
unset SSH_ASKPASS && unset DISPLAY
git pull

# Install nerfstudio if not already there
pip install -q nerfstudio

# Launch the job in the background (safe if SSH disconnects)
mkdir -p logs
nohup bash ucl_gpu/run_video_splat.sh > logs/video_splat.log 2>&1 &

# Watch the log
tail -f logs/video_splat.log
```

### Step 1.4 — What to watch for in the log

The script has 5 stages. Here's what good output looks like:

```
[1/5] nerfstudio: 1.1.3
[2/5] Video found: data/raw/room_video.MOV (444M)
[3/5] Extracted 316 frames.
[4/5] Running COLMAP SfM on 316 video frames...
      ┌─ COLMAP Quality Report ──────────────────────────┐
      │  Input frames  : 316
      │  Registered    : 248  (78%)           ← 200+ is EXCELLENT
      │  Quality       : ✅ EXCELLENT          ← if you see this, proceed
      └─────────────────────────────────────────────────┘
[5/5] Training splatfacto (30,000 steps)...
      ✅ scene.ply created: outputs/splat_video_v1/scene.ply
```

If COLMAP registers < 50 cameras, the script stops and tells you — don't continue wasting GPU time.

### Step 1.5 — Download outputs (run on Mac, ~10 min)
```bash
# Splat file
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat_video_v1/ \
  ~/Downloads/3D-Spatial-Reconstruction/outputs/splat_video_v1/

# COLMAP poses (needed for semantic reprojection)
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/colmap_video/ \
  ~/Downloads/3D-Spatial-Reconstruction/data/colmap_video/
```

**✅ Check**: `ls outputs/splat_video_v1/scene.ply` should exist and be 200-600 MB.

---

## Phase 2 — Convert & Preview (Mac, ~10 min)

After downloading, quickly verify the splat looks good before doing anything else.

### Step 2.1 — Convert to fast .splat format
```bash
cd ~/Downloads/3D-Spatial-Reconstruction

python scripts/convert_to_splat.py \
  --input outputs/splat_video_v1/scene.ply \
  --output outputs/splat_video_v1/scene.splat
```

### Step 2.2 — Prune floaters (removes ~70% of transparent Gaussians)
```bash
python scripts/prune_splat.py \
  --input  outputs/splat_video_v1/scene.splat \
  --output outputs/splat_video_v1/scene_pruned.splat
```

### Step 2.3 — Preview in the viewer
```bash
python open_viewer.py
```
Then open: `http://localhost:8080/app/static/index.html`

Change the viewer's splat URL temporarily to load the new splat:
- Open browser dev tools → Console
- Type: `loadScene('/outputs/splat_video_v1/scene_pruned.splat')`

Or just update `SPLAT_CANDIDATES[0]` in `app/static/index.html` temporarily to `/outputs/splat_video_v1/scene_pruned.splat`.

**What you should see**: A clear room with distinct walls, floor, furniture. No fog. Objects in correct positions.

---

## Phase 3 — Fix Semantics (Mac, ~45 min)

Now that we have correct camera poses from the new COLMAP run, we re-run the semantic pipeline so object colors and positions are accurate.

### Step 3.1 — Re-paint semantic Gaussians using new poses
```bash
python scripts/paint_semantic_gaussians.py \
  --splat_ply    outputs/splat_video_v1/scene.ply \
  --semantic_dir outputs/semantic \
  --cameras_bin  data/colmap_video/sparse/0/cameras.bin \
  --images_bin   data/colmap_video/sparse/0/images.bin \
  --output_ply   outputs/scene_semantic.ply \
  --conf_threshold 0.20 \
  --min_votes 2
```
Expected: ~30-50% of Gaussians labeled (up from 14.5% with 54 frames).

### Step 3.2 — Convert semantic PLY to pruned .splat
```bash
python scripts/convert_to_splat.py \
  --input  outputs/scene_semantic.ply \
  --output outputs/scene_semantic.splat

python scripts/prune_splat.py \
  --input  outputs/scene_semantic.splat \
  --output outputs/scene_semantic_pruned.splat
```

### Step 3.3 — Re-lift semantics to 3D (fixes object positions)
The old `objects_3d.json` used VGGT poses (wrong coordinate system). Re-run with new poses:
```bash
python scripts/lift_semantics_3d.py \
  --semantic_dir outputs/semantic \
  --cameras_bin  data/colmap_video/sparse/0/cameras.bin \
  --images_bin   data/colmap_video/sparse/0/images.bin \
  --output_dir   outputs/
```

### Step 3.4 — Rebuild scene graph
```bash
python scripts/build_scene_graph.py
```

### Step 3.5 — Test everything together
```bash
python open_viewer.py
```
Open the viewer. You should see:
- Semantic colors on objects (bed = pink, desk = teal, chair = blue, etc.)
- Clicking an object in the sidebar → camera flies to correct location
- Info card shows reasonable confidence and volume values

---

## Phase 4 — Deploy to HuggingFace Spaces (1-2h)

### Step 4.1 — Upload large PLY files to HF Dataset
The splat files are too large for the Spaces repo. Upload to HF Dataset via git-lfs:
```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
# Upload pruned semantic splat (fast-loading, ~40MB)
api.upload_file(
  path_or_fileobj='outputs/scene_semantic_pruned.splat',
  path_in_repo='scene_semantic_pruned.splat',
  repo_id='JesonRamesh/roboscene-data',
  repo_type='dataset',
)
print('Done')
"
```

### Step 4.2 — Update app.py for HF Spaces
On HF Spaces, port 8082 (background file server) is not accessible from the browser.
The viewer needs to load files via Gradio's `/file=` route or the HF Dataset URL.

Edit `app/app.py`:
- Set `SPLAT_URL` from env var `HF_SPLAT_URL` (set as HF Space secret)
- The viewer HTML should accept the URL directly

### Step 4.3 — Push to HF Spaces
```bash
# Commit + push app to HF Spaces
cd ~/Downloads/3D-Spatial-Reconstruction
git add app/ outputs/scene_graph.json outputs/objects_3d.json outputs/navigability_map.png
git commit -m "Add retrained splat + semantic viewer"
git push

# Or use the HF CLI
huggingface-cli repo create roboscene-plus --type space --sdk gradio
```

---

## Phase 5 — README + Polish (1h, last step)

### Step 5.1 — Write README.md
Key sections (in order of importance):
1. **Demo GIF / screenshot** — record a screen capture of the 3D viewer with tour mode
2. **What it does** — one paragraph, robot framing
3. **Quick start** — `python open_viewer.py` in 3 commands
4. **Pipeline diagram** — the `app/assets/pipeline_diagram.svg` file
5. **Novel contribution** — confidence-aware Gaussian tagging section
6. **Design choices** — why VGGT, why COLMAP, why Claude API
7. **Limitations** — honest: 14% → 50% semantic labeling, bbox inaccuracy

### Step 5.2 — Final checks
```
[ ] Open viewer in Chrome, Firefox, Safari — does it load?
[ ] Run all 4 example queries in Robot Query tab — do they answer?
[ ] Does ANTHROPIC_API_KEY work on HF Spaces (set as secret)?
[ ] Does HF Space load publicly without login?
[ ] Is GitHub repo public?
```

---

## Timeline to Deadline

| Day | Task | Machine | Time |
|---|---|---|---|
| **Today (21 May)** | Book bluestreak + upload video + launch GPU job | Mac → bluestreak | 30 min active + 3h waiting |
| **Today/Tomorrow** | Download GPU outputs + convert + preview splat | Mac | 15 min |
| **22 May** | Re-run semantic pipeline (Steps 3.1–3.5) | Mac | 1h |
| **23 May** | HF Spaces deployment | Mac | 2h |
| **24 May** | README + final testing | Mac | 2h |
| **25 May** | Buffer / submit | — | — |

---

## Quick Reference — Key Commands

```bash
# Launch the viewer (always works locally)
python open_viewer.py

# Convert + prune a new splat
python scripts/convert_to_splat.py --input <file.ply> --output <file.splat>
python scripts/prune_splat.py --input <file.splat> --output <file_pruned.splat>

# Repaint semantics (after new COLMAP)
python scripts/paint_semantic_gaussians.py \
  --cameras_bin data/colmap_video/sparse/0/cameras.bin \
  --images_bin  data/colmap_video/sparse/0/images.bin

# Rebuild everything downstream
python scripts/lift_semantics_3d.py
python scripts/build_scene_graph.py

# Query the scene
python scripts/query_scene.py
```

---

## If COLMAP Fails (Fallback Plan)

If the COLMAP step in `run_video_splat.sh` registers < 50 cameras:

**Option B — Use existing VGGT poses (no new upload needed)**
```bash
# On bluestreak (git pull already done):
bash ucl_gpu/run_colmap_splat.sh
```
This runs COLMAP on the 511 telephoto photos. The 14° FOV is harder for COLMAP but
511 images compensates. Expected: 150-250 cameras registered.

**Why Plan A is better**: The video's 73° FOV gives much more parallax and easier
feature matching. Plan B is a fallback if the video has unexpected issues.

---

## Technical Reference (Keep for Debugging)

### Why the old splat was bad
- MASt3R-SLAM: 54 keyframes from video → 54 × 73° ≈ 2 passes around the room
- 3DGS needs 150+ views for indoor rooms → floaters fill unobserved areas
- Telephoto photos (14° FOV): VGGT computed 511 poses but narrow FOV means less parallax

### Coordinate systems in this project
- **Splat coordinate system**: set by nerfstudio's scene normalisation during training
  - Nerfstudio applies: `p_ns = scale × R_orient × (p_colmap − t_center)`
  - `scale≈0.98`, `t_center≈[-0.33,-0.14,0.35]`
  - `paint_semantic_gaussians.py` already handles this inversion correctly
- **objects_3d.json**: in VGGT world frame (WRONG after retrain — needs re-running)
- **After retrain**: all data will be in the new COLMAP video frame coordinate system

### SH colour encoding (for paint_semantic_gaussians.py)
```python
SH_C0 = 0.28209479177387814
f_dc = (rgb_0_to_1 - 0.5) / SH_C0   # converts RGB to DC SH coefficient
```

### COLMAP binary format (for read_images_binary)
```python
# images.bin → R_w2c, t_w2c per frame
# cameras.bin → fx, fy, cx, cy
# Projection: p_cam = R_w2c @ p_world + t_w2c
#             u = fx * p_cam[0]/p_cam[2] + cx
```
