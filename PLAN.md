# RoboScene+ — Step-by-Step Guide to Completion
### Updated: 2026-05-21 | Deadline: 2026-05-25 (4 days)

---

## Where We Are Right Now

| What | Status | Notes |
|---|---|---|
| 3D Gaussian Splat | ⚠️ Distorted | Only 54 training views — needs GPU retrain |
| Standalone viewer | ✅ Done | `python open_viewer.py` launches it |
| Semantic painting | ⚠️ Partial | 14.5% labeled — will improve after retrain |
| Object positions | ⚠️ Wrong coord system | Will be fixed after retrain |
| Scene graph + Claude API | ✅ Done | query_scene.py working |
| HF Spaces | ⬜ Not started | After splat is fixed |
| README | ⬜ Not started | Last step |

**The single blocking issue**: The splat looks distorted because MASt3R-SLAM only used 54 video frames. Everything else is working. Once we retrain with 316 frames, the whole pipeline falls into place.

---

## The Full Plan — In Order

```
Phase 1 ── GPU Retrain        (today, ~3h on bluestreak)
    │
Phase 2 ── Mac Post-Processing  (after GPU, ~1h on Mac)
    │
Phase 3 ── Fix Semantics        (~1h on Mac)
    │
Phase 4 ── Deploy to HF Spaces  (~1h)
    │
Phase 5 ── README + Polish      (~1h)
```

---

## Phase 1 — GPU Retrain (Do This Today)

**Why**: The current splat used 54 keyframes. We're retraining with 316 video frames (6× more).
**Where**: UCL bluestreak GPU
**Time**: ~3 hours total (most of it waiting)

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
