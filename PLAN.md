# RoboScene+ — Plan
### Updated: 2026-05-23 | Deadline: 2026-05-25 (2 days)

---

## Current State

| What | Status | Notes |
|---|---|---|
| Video v2 | ✅ | room_video_v2.MOV, 320s, 1920x1080 |
| COLMAP v3 | ✅ | 539/641 frames registered, 103K points |
| Splat v3 | ✅ | 60K steps, 3.74M Gaussians — current best |
| Viewer | ✅ | Loads splat_v3/scene.splat, eye-level camera |
| Camera rotation | ⚠️ | Position correct, orbit axis wrong (screen-Z not world-Y) |
| Floater removal | ❌ | All post-processing cuts bed wall — fundamentally limited |
| SAM2 on bluestreak | ❌ | GroundingDINO BERT error blocks semantic pipeline |
| Semantic labels | ❌ | Ran with wrong poses (v1) → labels in wrong 3D positions |
| HF Spaces | ⬜ | Not started |
| README | ⬜ | Not started |

**Active splat**: outputs/splat_v3/scene.splat (114MB, 3.74M Gaussians)
**Fallback**: outputs/splat_video_v2/scene_full.splat (70MB, 2.28M, camera works)

---

## What Failed and Why

| Attempt | Why It Failed |
|---|---|
| Alpha threshold prune | Removes real geometry in sparse-coverage areas |
| SOR (k=20, std=1.0-2.0) | Floaters are semi-dense clusters, not isolated points |
| Bbox crop | Room not axis-aligned — clips bed wall |
| Convex hull + visibility filter | Floaters have same visibility as real geometry (mean 35 views each) |
| Semantic paint with v1 poses | v1 COLMAP coordinate system != v3 splat coordinate system |
| SAM2 on bluestreak | GroundingDINO: AttributeError: BertModel has no attribute get_head_mask |

Key insight (TIDI-GS paper, arXiv:2601.09291): Floaters can only be reliably
removed during training using 4 simultaneous signals (visibility, opacity, learned
importance, gradient EMA). Post-processing on a finished splat cannot replicate this.
The correct fix is semantic segmentation — keep only Gaussians inside SAM2 object masks.

---

## Next 3 Actions (in order)

### Action 1 — Fix GroundingDINO on bluestreak

A patch script is ready: ucl_gpu/fix_gdino.sh
It patches bertwarper.py line 29 directly — BERT API changed in newer transformers.

On bluestreak:
```bash
bash
source /opt/Python/Python-3.11.5_Setup.csh
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
git pull
bash ucl_gpu/fix_gdino.sh
```
Expected output: "GroundingDINO OK" and "GPU: NVIDIA GeForce RTX 4070 Ti SUPER"

If it works, immediately launch SAM2:
```bash
nohup bash ucl_gpu/run_semantic_v3.sh > logs/semantic_v3.log 2>&1 &
echo "PID: $!"
tail -f logs/semantic_v3.log
```
This runs Grounded SAM2 on 641 v2 frames (~2-3h), paints splat_v3 with
colmap_v3 poses (correct coordinate system), then lifts to 3D centroids.

Download when done (on Mac):
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/semantic_v3/ \
  ~/Downloads/3D-Spatial-Reconstruction/outputs/semantic_v3/
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/scene_semantic_v3.ply \
  ~/Downloads/3D-Spatial-Reconstruction/outputs/scene_semantic_v3.ply
scp -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/objects_3d_v3.json \
  ~/Downloads/3D-Spatial-Reconstruction/outputs/objects_3d_v3.json
```

---

### Action 2 — Fix viewer camera rotation axis

What is wrong: OrbitControls orbits around screen-Z instead of world-Y.
cameraUp=[0,1,0] is set correctly, HOME_POS=[0.30, 0.31, -0.56] is correct,
but the GS library OrbitControls is not picking up the up vector.

Diagnostic: open http://localhost:8080/app/static/index.html, then in browser
console (Cmd+Option+J):
```javascript
Object.keys(viewer)
viewer.controls
viewer.camera.up
```
Share what viewer.controls prints — this identifies the correct property to patch.

Fix already attempted in index.html (fixControls() at 0ms/500ms/1500ms):
```javascript
cam.up.set(0, 1, 0)
ctrl.object.up.set(0, 1, 0)
ctrl.screenSpacePanning = false
ctrl.update()
```
ctrl.object may not be the right handle — console output will reveal correct name.

---

### Action 3 — Rebuild semantic splat with correct poses + full segmentation

Once Action 1 produces outputs/semantic_v3/ JSONs (5-digit, matching colmap_v3),
re-run painting with correct poses:
```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 scripts/paint_semantic_gaussians.py \
  --splat_ply      outputs/splat_v3/scene.ply \
  --semantic_dir   outputs/semantic_v3 \
  --cameras_bin    data/colmap_v3/sparse/0/cameras.bin \
  --images_bin     data/colmap_v3/sparse/0/images.bin \
  --output_ply     outputs/splat_v3/scene_semantic_v3.ply \
  --conf_threshold 0.20 \
  --min_votes      2
```

For full object segmentation (objects highlighted, not just labelled):
Modify paint_semantic_gaussians.py:
- Labelled Gaussians: full opacity, bright class RGB colour
- Unlabelled Gaussians: dim_factor=0.0 (remove) or neutral grey
Then update SPLAT_CANDIDATES in app/static/index.html to load the new splat first.

---

## After Actions 1-3

4. HF Spaces: upload splat to HF Dataset, update app/app.py with URL,
   set ANTHROPIC_API_KEY as HF Space secret, push via git push
5. README: demo screenshot, pipeline diagram, quick-start, limitations

---

## Key File Locations

| File | Location |
|---|---|
| Best splat PLY | outputs/splat_v3/scene.ply |
| Best splat (viewer) | outputs/splat_v3/scene.splat |
| COLMAP v3 transforms | data/colmap_v3/transforms.json |
| COLMAP v3 sparse | data/colmap_v3/sparse/0/ |
| v1 semantic JSONs (wrong) | outputs/semantic/frame_0001.json (4-digit) |
| v3 semantic JSONs (pending) | outputs/semantic_v3/frame_00001.json (5-digit) |
| GroundingDINO patch | ucl_gpu/fix_gdino.sh |
| SAM2 job script | ucl_gpu/run_semantic_v3.sh |
| Floater removal script | scripts/clean_splat_visibility.py |
| Semantic paint script | scripts/paint_semantic_gaussians.py |
| Viewer HTML | app/static/index.html |

---

## Bluestreak Quick Connect

```bash
ssh -J jrameshs@knuckles.cs.ucl.ac.uk jrameshs@bluestreak.cs.ucl.ac.uk
bash
source /opt/Python/Python-3.11.5_Setup.csh
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
```

## Mac Quick Start

```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 open_viewer.py
# open http://localhost:8080/app/static/index.html
```
