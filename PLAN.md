# RoboScene+ — Updated Plan
### Updated: 2026-05-21 | Deadline: 2026-05-25 (4 days)

---

## Current Status

| What | Status | Notes |
|---|---|---|
| 3D Gaussian Splat (new) | ✅ Done | 304 views, 3.1M Gaussians, 202MB PLY |
| Splat viewer | ✅ Loads | New splat visible, room structure correct |
| Splat quality | ⚠️ Needs fixing | Floaters outside room, floor gap, low opacity |
| Camera navigation | ⚠️ Broken | Top-down locked, cannot freely navigate |
| Semantic labels | ⚠️ Wrong coords | objects_3d.json uses old VGGT coordinate system |
| Semantic painting | ⬜ Not re-run | Needs re-running with new COLMAP poses |
| Scene graph + Claude API | ✅ Done | query_scene.py working |
| HF Spaces | ⬜ Not started | After splat is fixed |
| README | ⬜ Not started | Last step |

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

## Updated Plan — In Order

```
Phase 2B ── Fix viewer camera + navigation     (30 min, Mac)
    │
Phase 2C ── Tighten prune + crop floaters      (20 min, Mac)
    │
Phase 2D ── Re-train with point cloud init     (3h, bluestreak) ← optional but recommended
    │
Phase 3  ── Re-run semantics w/ new poses      (1h, Mac)
    │
Phase 4  ── Deploy to HF Spaces               (1-2h, Mac)
    │
Phase 5  ── README + Polish                    (1h, Mac)
```

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
