# RoboScene+ — Active Plan
### Last updated: 2026-05-24 (Session 11 — Viewer Polish + HF Deploy next) | Deadline: 2026-05-25

---

## Current State ✅

```
room_video_v2.MOV (641 frames)
  → COLMAP v4 (539 frames, transforms.json)
  → nerfstudio splatfacto 60K steps (RTX 4070 Ti, bluestreak)
  → outputs/splat_v4/scene.ply                (2.41M Gaussians, 571MB, Z-up)
  → outputs/splat_v4/scene_aligned.ply        (2.41M Gaussians, 598MB, Y-up ✅)
  → outputs/splat_v4/scene_aligned.splat      (74MB — PRIMARY in viewer ✅)
  → outputs/splat_v4/scene_pruned.ply         (1.22M Gaussians, 302MB ✅)
  → outputs/splat_v4/scene_pruned.splat       (39MB)
  → outputs/splat_v4/scene_semantic.ply       (2.41M Gaussians, 598MB, 35% tint)
  → outputs/splat_v4/scene_semantic.splat     (77MB)
  → outputs/splat_v4/semantic_class.npy       (exact per-Gaussian class uint8 labels)
  → outputs/splat_v4/highlights/              (per-object full-color splats)
  → outputs/objects_3d_yup.json              (Y-up centroids + confidence)
  → outputs/semantic_v2/                     (641 SAM2 JSONs from v2 frames ✅)
```

**Viewer:** `python3 open_viewer.py` → http://localhost:8080/app/static/index.html
**Splat loaded:** `outputs/splat_v4/scene_aligned.splat` (74MB, clean appearance)
**HOME_POS:** `[0.261, -0.447, 0.114]` — inside room at eye level
**HOME_LOOK:** `[0.261, -0.412, -0.262]` — scene centroid

**Confirmed objects (5):** bed, laptop, fan, chair, shelf
**False positives removed:** monitor, desk, door, window, lamp

---

## ✅ Session 11 — Viewer Polish (2026-05-24)

### Futuristic UI overhaul
- **Color scheme:** electric cyan `#00d4ff` + neon purple `#b039ff` (replaced muted `#7F77DD`)
- **Neon panel glow:** animated `box-shadow` pulse on topbar and sidebar (5–6s breathing cycle)
- **Animated edges:** sliding cyan→purple gradient on topbar bottom edge and sidebar right edge
- **Animated topbar background:** slow 14s gradient shift across dark-blue hues
- **Scanlines:** subtle horizontal scanlines on sidebar, topbar, and loading overlay (≤3% opacity)
- **Object dot glow:** each confirmed object's color swatch glows with its semantic color
- **Buttons:** neon glow on hover, box-shadow active states
- **Screen-edge vignette:** radial gradient giving a subtle cyan halo at viewport corners

### Loading screen upgrade
- Glitch animation on `⬡` glyph and "RoboScene+" title (CSS clip-path + skew + color split)
- Progress bar: shimmer gradient (cyan→purple→cyan) instead of solid fill
- Scanline sweep overlay during loading

### Onboarding overlay
- Appears after loading completes; scene rotates visibly behind it (translucent dark)
- Shows `⬡ RoboScene+ / Scroll to enter the scene ↓` with floating + pulsing arrow
- Dismissed after 2 scroll wheel ticks or any click; fades out over 1.1s

### HUD panels
- Info card + camera coords panel: top neon strip (::before), bottom-right corner bracket (::after)
- Scrollbar neon cyan thumb

### Tour navigation — root cause fixed
**Bug:** `objects_3d_yup.json` centroids were in camera-normalisation space (~10× smaller
scale than PLY viewer space). Tour was flying to completely wrong positions.

**Fix:** Hardcoded `TOUR_STOPS` computed from actual Gaussian cluster means read directly
from `semantic_class.npy` + `scene_aligned.ply`. Camera positions then captured manually
in the viewer for each object.

**Second bug:** `tickAnim` called `anim.onDone?.()` before `anim = null`, so any `flyTo`
inside the callback was immediately overwritten. Fixed by nulling `anim` first.

**Tour order:** Bed → Laptop → Fan → Chair → Shelf (hardcoded `TOUR_ORDER`)

### Exact tour stops (manually captured)
```javascript
bed:    pos:[-0.067, -0.415, -0.053]  look:[ 0.105, -0.412, -0.163]
laptop: pos:[ 0.587, -0.342, -0.057]  look:[ 0.765, -0.341,  0.290]
fan:    pos:[ 0.626, -0.386,  0.020]  look:[ 0.929, -0.341,  0.261]
chair:  pos:[ 0.596, -0.360, -0.555]  look:[ 0.608, -0.412, -0.323]
shelf:  pos:[ 0.378, -0.308, -0.183]  look:[ 0.204, -0.412, -0.306]
```

### Other changes
- Intro orbit removed — scene opens directly at HOME_POS
- Sidebar shows only 5 confirmed objects; false positives hidden
- Clicking a sidebar row flies camera to that object's exact position
- Tour dots updated to match 5-object tour

---

## ✅ Session 10 — Semantic Painting (2026-05-24)

1. Re-ran SAM2 semantics on v2 frames (641 frames, bluestreak RTX 4070 Ti)
2. Fixed frame matching bug in `paint_semantic_gaussians.py` (numeric stem matching)
3. Semantic painting on `scene_aligned.ply`: 78% Gaussians labeled, 35% tint strength
4. Generated per-object highlight splats (`scripts/gen_highlight_splats.py`)
5. `semantic_class.npy` saved — exact per-Gaussian class labels (used for centroid diagnosis)

---

## ✅ Session 8 — Prune Floaters (2026-05-24)

`scripts/prune_floaters.py`: opacity filter (>0.30) + density filter (≥5 neighbours, r=0.1m)
Result: 2.41M → 1.22M Gaussians (302MB PLY / 39MB splat)

---

## ✅ Session 7 — Floor Alignment (2026-05-24)

`scripts/realign_splat_v4.py`: Z-up → Y-up (−90° around X) + 2.73° SVD tilt correction
Key lesson: use lowest 1% hi-opacity Gaussians for SVD (not 10% — floaters contaminate)

---

## 🚀 NEXT: Hugging Face Spaces Deployment

### Goal
Host the viewer publicly at `https://huggingface.co/spaces/JesonRamesh/roboscene-plus`
so anyone can open it in a browser without running a local server.

### What needs to go up

| File | Size | Where |
|---|---|---|
| `outputs/splat_v4/scene_aligned.splat` | 74MB | HF Dataset |
| `outputs/objects_3d_yup.json` | ~5KB | HF Dataset or inline |
| `app/static/index.html` | ~50KB | HF Space (static) |

### Step-by-step plan

**Step 1 — Upload splat to HF Dataset**
```bash
# Install HF CLI if needed
pip install huggingface_hub

# Login
huggingface-cli login  # paste HF_TOKEN

# Upload the splat file
huggingface-cli upload JesonRamesh/roboscene-data \
  outputs/splat_v4/scene_aligned.splat \
  scene_aligned.splat \
  --repo-type dataset

# Upload objects JSON
huggingface-cli upload JesonRamesh/roboscene-data \
  outputs/objects_3d_yup.json \
  objects_3d_yup.json \
  --repo-type dataset
```

**Step 2 — Get the raw file URLs**
After upload, files are served at:
```
https://huggingface.co/datasets/JesonRamesh/roboscene-data/resolve/main/scene_aligned.splat
https://huggingface.co/datasets/JesonRamesh/roboscene-data/resolve/main/objects_3d_yup.json
```

**Step 3 — Update index.html for HF deployment**
Change `SPLAT_CANDIDATES[0]` and `OBJECTS_URL` to the HF Dataset raw URLs.
The viewer is pure static HTML — no server needed.

**Step 4 — Create static HF Space**
```bash
# Clone the Space repo
git clone https://huggingface.co/spaces/JesonRamesh/roboscene-plus
cd roboscene-plus

# Copy viewer
cp ~/Downloads/3D-Spatial-Reconstruction/app/static/index.html index.html

# HF static Spaces need a README.md with metadata
cat > README.md << 'EOF'
---
title: RoboScene+
emoji: ⬡
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---
EOF

git add . && git commit -m "deploy: futuristic 3D semantic viewer"
git push
```

**Step 5 — Verify**
Open `https://huggingface.co/spaces/JesonRamesh/roboscene-plus` in a browser.
The splat should load from the HF Dataset CDN. Test Tour button and sidebar clicks.

### CORS note
HF Dataset files are served with permissive CORS headers — `fetch()` from the Space
to the Dataset will work without a proxy.

### Fallback if splat is too large
74MB may be slow on HF CDN. If load time > 15s, switch to `scene_pruned.splat` (39MB).
Update `SPLAT_CANDIDATES[0]` accordingly.

---

## Remaining TODO

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Floor alignment | ✅ Done | `scene_aligned.splat` (74MB, Y-up) |
| 2 | Prune floaters | ✅ Done | `scene_pruned.splat` (39MB, 1.22M Gaussians) |
| 3 | Semantic painting | ✅ Done | `semantic_class.npy` + per-object highlights |
| 4 | Viewer UI | ✅ Done | Futuristic neon UI, onboarding overlay, fixed tour |
| 5 | Tour navigation | ✅ Done | 5 manually-captured stops, correct coordinate space |
| 6 | HF Spaces deploy | 🚀 Next | Upload splat to HF Dataset, push static Space |
| 7 | README.md | ⏳ Pending | Demo link, pipeline diagram, novel contribution write-up |

---

## Key File Locations

```
outputs/splat_v4/
  scene_aligned.splat      ← PRIMARY — loaded in viewer (74MB)
  scene_pruned.splat       ← fallback if 74MB too slow for HF (39MB)
  semantic_class.npy       ← per-Gaussian class labels (uint8, 2.41M entries)

outputs/
  objects_3d_yup.json      ← object metadata (frames_seen, confidence)

app/static/index.html      ← complete self-contained viewer
open_viewer.py             ← local dev server (port 8080)

scripts/
  realign_splat_v4.py      ← Z-up→Y-up alignment
  prune_floaters.py        ← opacity + density pruning
  paint_semantic_gaussians.py  ← semantic tinting
  gen_highlight_splats.py  ← per-object splat generation
  convert_to_splat.py      ← PLY→.splat converter
```

---

## Mac Quick Start

```bash
cd ~/Downloads/3D-Spatial-Reconstruction
python3 open_viewer.py
# open http://localhost:8080/app/static/index.html
```

## Bluestreak Quick Connect (if GPU needed)

```bash
ssh -J jrameshs@knuckles.cs.ucl.ac.uk jrameshs@bluestreak.cs.ucl.ac.uk
bash
source /opt/Python/Python-3.11.5_Setup.csh
source /scratch0/jrameshs/roboscene_env/bin/activate
cd /scratch0/jrameshs/roboscene-plus
```

---
