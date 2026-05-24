# RoboScene+ — Active Plan
### Last updated: 2026-05-24 (Session 10 — Semantics + Viewer cleanup) | Deadline: 2026-05-25

---

## Current State ✅

```
room_video_v2.MOV (641 frames)
  → COLMAP v4 (539 frames, transforms.json)
  → nerfstudio splatfacto 60K steps (RTX 4070 Ti, bluestreak)
  → outputs/splat_v4/scene.ply                (2.41M Gaussians, 571MB, Z-up)
  → outputs/splat_v4/scene_aligned.ply        (2.41M Gaussians, 598MB, Y-up ✅)
  → outputs/splat_v4/scene_aligned.splat      (74MB)
  → outputs/splat_v4/scene_pruned.ply         (1.22M Gaussians, 302MB ✅)
  → outputs/splat_v4/scene_pruned.splat       (39MB)
  → outputs/splat_v4/scene_semantic.ply       (2.41M Gaussians, 598MB, 35% tint ✅)
  → outputs/splat_v4/scene_semantic.splat     (77MB — PRIMARY in viewer ✅)
  → outputs/splat_v4/semantic_class.npy       (exact per-Gaussian class uint8 labels)
  → outputs/splat_v4/highlights/              (10 x per-object full-color splats)
  → outputs/objects_3d_yup.json              (Y-up centroids + confidence)
  → outputs/semantic_v2/                     (641 SAM2 JSONs from v2 frames ✅)
```

**Viewer:** `python3 open_viewer.py` → http://localhost:8080/app/static/index.html  
**Splat loaded:** `outputs/splat_v4/scene_aligned.splat` (74MB, clean appearance, normal opacity)  
**Sidebar:** 10 detected objects — color swatch + name + frames seen (SAM2 + GroundingDINO)  
**HOME_POS:** `[0.261, -0.553, 1.238]` — eye-level, pulled back from room  
**HOME_LOOK:** `[0.261, -0.412, -0.262]` — scene centroid (hi-opacity)

---

## Session 7 Implementation Log (2026-05-24) — Floor Alignment ✅

### Problem
The orbit was wrong in all previous splat versions:
- `splat_v3/scene_yup.ply` — used mean camera-up vector → rotation made it worse
- `splat_v4/scene.ply` — nerfstudio `--orientation-method up` only partial fix (17° off)
- First attempt at `realign_splat_v4.py` — used lowest 10% Y Gaussians for floor SVD
  → floaters contaminated the sample → floor normal was `[0.224, -0.954, 0.199]`
  → rotation_between() introduced a roll → floor became a vertical wall ❌

### Root Cause Diagnosed
```
nerfstudio PLY header: "comment Vertical Axis: z"  ← Z-up, NOT Y-up
Viewer expects Y-up.
Fix = two-step rotation:
  Step A: -90° around X axis (Z-up → Y-up)  [[1,0,0],[0,0,1],[0,-1,0]]
  Step B: 2.73° residual tilt correction (fit to lowest 1% hi-opacity Gaussians)
```

### Floor SVD lesson learned
At 10% percentile threshold, singular values are `[339, 298, 222]` — nearly equal,
meaning floaters are spread in all 3 axes and contaminate the floor normal.
At **1% threshold**, singular values drop to `[228, 192, 90]` — clear separation,
floor normal `[-0.007, -0.999, -0.047]` — nearly perfect -Y. Use 1% always.

### Script: `scripts/realign_splat_v4.py`
- Input: `outputs/splat_v4/scene.ply`
- Step A: apply `R_zup = [[1,0,0],[0,0,1],[0,-1,0]]` to all positions + quaternions
- Step B: SVD on lowest 1% hi-opacity Gaussians (by Y after Step A) → residual tilt R
- Vectorised batch quaternion rotation (Shepperd 4-case, no Python loop)
- Output: `outputs/splat_v4/scene_aligned.ply` (598MB)
- Converted: `outputs/splat_v4/scene_aligned.splat` (74MB)

### Result
- Floor is flat and horizontal ✅
- Orbit is lazy-Susan around vertical Y axis ✅  
- HOME_POS inside room at eye level ✅
- **Remaining problem: floaters throughout the scene** (see next session below)

---

## ⚠️ NEXT PROBLEM: Floaters

The scene has many semi-transparent floater Gaussians spread throughout the room
(artefacts from splatfacto training on a phone video with limited coverage).
They appear as ghostly patches in the air that obscure the geometry.

**Effect:** The room looks foggy/noisy — hard to read the furniture clearly.
**Root cause:** Low-opacity Gaussians that survived the training pruning threshold
(`cull-alpha-thresh 0.005`). These are real outputs, not a coordinate bug.

---

## ✅ DONE Session 8 — Prune Floaters

### Result
`scripts/prune_floaters.py` written and run. Produces `scene_pruned.ply`.

**Pruning pipeline:**
- Step 1 — Opacity filter: `sigmoid(opacity) > 0.30`
- Step 2 — Density filter: ≥5 neighbours within r=0.1m (cKDTree on full set)
- Step 3 — Bbox clip: X[-5,5] Y[-5.5,4.5] Z[-5,5.5] (scene fits within, 0 clipped)

**Results:**
```
Original:         2,410,031 Gaussians  (598 MB PLY / 74 MB splat)
After opacity:    1,255,751  (47.9% removed — haze band opacity 0.0–0.30)
After density:    1,215,812  (3.2% additional removed)
After bbox:       1,215,812  (0 clipped — scene within bounds)
Final:            1,215,812 Gaussians  (302 MB PLY / 39 MB splat)
```

**Key insight — opacity distribution is bimodal:**
- p05=0.14, p10=0.19, p20=0.49 — haze band
- p30=0.976, p50=1.0, p70+=1.0 — opaque core (70%+ fully saturated)
- Threshold 0.30 cuts the semi-transparent layer cleanly

**Viewer updated:** `SPLAT_CANDIDATES[0]` → `scene_pruned.splat` ✅

### Original Goal
Write and run `scripts/prune_floaters.py` to remove low-opacity and spatially
isolated Gaussians from `scene_aligned.ply`, producing a clean `scene_pruned.ply`.

### Script spec

```
Inputs:  outputs/splat_v4/scene_aligned.ply
Outputs: outputs/splat_v4/scene_pruned.ply
         outputs/splat_v4/scene_pruned.splat
```

**Step 1 — Opacity filter:**
Keep only Gaussians where `sigmoid(opacity) > OPACITY_THRESH` (try 0.1 first).
Print how many survive at thresholds 0.05, 0.10, 0.15, 0.20 before committing.

**Step 2 — Spatial density filter (remove isolated floaters):**
For surviving Gaussians, compute local neighbourhood density:
- Subsample to max 200K points for KD-tree construction (speed)
- For each Gaussian, count neighbours within radius `r = 0.1m`
- Keep Gaussians with at least `min_neighbours = 5` neighbours
- This removes lone floaters far from any surface

**Step 3 — Bounding box clip:**
Drop Gaussians outside the room bounding box with 10% padding.
Room bbox from hi-opacity Gaussians: roughly X=[-6.5, 6.5], Y=[-6.7, 6.3], Z=[-6.5, 6.9]
Clip to X=[-5, 5], Y=[-5.5, 4.5], Z=[-5, 5.5] (tighter inner box, avoids wall floaters).

**Step 4 — Convert and verify:**
```bash
python3 scripts/convert_to_splat.py \
  --input outputs/splat_v4/scene_pruned.ply \
  --output outputs/splat_v4/scene_pruned.splat
```

**Step 5 — Update viewer:**
Change `SPLAT_CANDIDATES[0]` in `app/static/index.html` to `scene_pruned.splat`.
Verify the room looks cleaner with the floaters gone.

### Print targets
```
Original:       2,410,031 Gaussians
After opacity:  ~1,200,000 (target: keep ~50%)
After density:  ~900,000   (target: remove ~25% of remaining)
After bbox:     ~800,000   (target: final clean scene)
File size:      ~25MB .splat  (vs 74MB now)
```

### Tune if needed
If the scene looks over-pruned (missing walls/furniture):
- Lower `OPACITY_THRESH` to 0.05
- Lower `min_neighbours` to 3
- Widen the bbox clip

If floaters still visible:
- Raise `OPACITY_THRESH` to 0.15
- Raise `min_neighbours` to 10
- Tighten the bbox clip

---

---

## Session 9 Investigation — Camera-Based Pruning (2026-05-24)

### Goal
Attempt stronger floater removal using distance-to-camera filter.

### Key Finding: Camera→PLY Transform Is Broken
- `applied_scale_yup = 0.1043` is wrong for placing cameras in PLY space
- Camera Y normalised: [-0.58, 0.38]; × (1/0.1043) = [-5.55, 3.64] — outside room bounds
- Empirical Y-scale needed: ~3.27 (not 9.59). PLY and camera normalisation used **different scales**.
- `applied_transform_yup` R matrix is a ~45° arbitrary rotation — all transform attempts failed
- Result: dist-to-camera filter unreliable; cameras placed incorrectly in PLY space

### Sparse Point Cloud Anchor — Most Promising Alternative
Files available:
- `data/colmap_v3/sparse/0/0/points3D.ply` — 103K COLMAP pts (raw Z-up COLMAP space)
- `data/vggt_out/sparse/points.ply` — 100K VGGT pts (unknown space)
- `data/vggt_out_v2/sparse/points.ply` — 100K VGGT pts v2

**Key unknown**: which sparse cloud is in the same coordinate space as `scene_aligned.ply`?
Need to check XYZ ranges: scene_aligned hi-opacity bounds ≈ X[-6,6] Y[-2,1] Z[-6,7].

### Next Step: Sparse Anchor Pruner
1. Check ranges of all 3 sparse clouds vs scene_aligned bounds (quick python script)
2. Build KD-tree from whichever matches PLY space
3. Keep Gaussian if dist_to_nearest_anchor < 0.5m OR (height ok AND opacity>0.01 AND scale<0.3)
4. Expected: removes 15-25% additional floaters vs current scene_pruned.ply

### Fallback: Statistical Outlier Removal (no camera/transform needed)
- For each Gaussian, count neighbours within r=0.15m
- Remove if count < 5 (isolated floater)
- Already partially done in session 8 density filter — but tighter radius may help
- Risk: may remove thin features (curtain edges, chair legs)

---

## ✅ Session 10 — Semantic Painting (2026-05-24)

### What was done
1. **Re-ran SAM2 semantics on v2 frames** (bluestreak): 641 frames from `data/video_frames_v2/`
   - 226s at 2.8fps on RTX 4070 Ti → `outputs/semantic_v2/frame_00001.json` … `frame_00641.json`
   - Frame naming fixed: v2 uses 5-digit zero-padding matching `transforms.json`
   - Old `outputs/semantic/` used 4-digit naming from v1 video — wrong frames entirely

2. **Fixed frame matching bug** in `paint_semantic_gaussians.py`:
   - Old code: exact stem match → 0 frames matched
   - Fix: numeric matching `int(stem.replace('frame_',''))` → 539/539 matched

3. **Ran semantic painting** on `scene_aligned.ply` (full 2.41M Gaussians):
   - `--tint_strength 0.35 --dim_factor 1.0` — 65% original + 35% class color
   - 1,878,707 / 2,410,031 Gaussians labeled (78.0%)
   - `f_rest` SH preserved → photorealistic tinted appearance
   - `semantic_class.npy` saved for exact per-Gaussian class lookup

4. **Generated per-object highlight splats** (`scripts/gen_highlight_splats.py`):
   - 10 x full-color object splats in `outputs/splat_v4/highlights/`

5. **Viewer updated**:
   - Primary splat → `scene_semantic.splat` (77MB)
   - Sidebar: color dot + name + confidence % badge per object
   - `splatAlphaRemovalThreshold: 1` → denser point cloud
   - `objects_3d_yup.json` with Y-up centroids

### Semantic label coverage
```
bed 31.9%  door 15.0%  desk 7.0%  chair 6.9%  shelf 5.3%
laptop 5.1%  window 2.2%  monitor 1.7%  fan 1.5%  lamp 1.3%
unlabeled 22.0%
```

---

## Remaining TODO

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Prune floaters | ✅ Done | `scene_pruned.splat` (39MB, 1.22M Gaussians) |
| 2 | Semantic painting | ✅ Done | `scene_semantic.splat` generated (78% labeled) — not shown in viewer (too noisy) |
| 3 | Viewer | ✅ Done | Clean splat with sidebar legend: color swatch + object name + frame count |
| 4 | Fix object centroids | 🔧 Needed | Monitor + window centroids wrong; recompute from `semantic_class.npy` cluster means |
| 5 | Fix object volumes | 🔧 Needed | Run `fix_volumes.py` → accurate bboxes + `objects_3d_v4.json` |
| 6 | HF Spaces deploy | ⏳ Pending | Upload `scene_aligned.splat` + `objects_3d_yup.json` to HF Dataset, push static Space |
| 7 | README.md | ⏳ Pending | Demo link, pipeline diagram, SAM2 detection novel contribution |

---

## Key File Locations

```
outputs/splat_v4/
  scene.ply                ← original from bluestreak (Z-up)
  scene_aligned.ply        ← Y-up aligned (current working version)
  scene_aligned.splat      ← loaded in viewer now
  scene_semantic.ply       ← semantic painted (Z-up coords — stale, redo after pruning)

scripts/
  realign_splat_v4.py      ← Z-up→Y-up + floor tilt fix ✅
  convert_to_splat.py      ← PLY→.splat converter ✅
  paint_semantic_gaussians.py  ← semantic painting (needs re-run on pruned PLY)
  fix_volumes.py           ← DBSCAN bbox cleanup

app/static/index.html      ← viewer (HOME_POS/HOME_LOOK set, scene_aligned.splat loaded)
open_viewer.py             ← local dev server (port 8080)
data/colmap_v4/transforms.json  ← camera poses (Y-up, used for training)
```

---

## Bluestreak Quick Connect (if GPU needed)

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

---