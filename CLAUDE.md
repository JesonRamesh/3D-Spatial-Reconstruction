# CLAUDE.md — RoboScene+ Project Guide

> This file lives in the root of your project repo.
> Claude Code reads it automatically at the start of every session.
> Keep it updated as the project evolves.

---

## Project Identity

**Name:** RoboScene+
**Goal:** Video → VGGT → 3D Gaussian Splatting → Semantic Scene Graph → Confidence-Aware Robot Spatial Memory
**Submission deadline:** 25 May 2026
**Target:** Humanoid (thehumanoid.ai) internship challenge
**Developer:** Jeson Ramesh Selvakumar, UCL MEng Robotics & AI, Year 2

---

## What This Project Does (Read This First Every Session)

RoboScene+ takes a short phone video of an indoor room and produces:

1. A photorealistic 3D Gaussian Splat reconstruction (via VGGT + gsplat)
2. Semantic labels on every object in 3D (via Grounded SAM2 + depth unprojection)
3. A **confidence map** — the novel contribution — which identifies which regions
   of the scene were well-observed vs guessed vs never seen (the "dead zone" problem)
4. A language-queryable scene graph: "Where is the laptop?" → 3D coordinates + confidence score
5. A deployed Gradio web demo on Hugging Face Spaces

The unique angle: we tag every Gaussian as `observed`, `sparse`, or `inferred`.
No other student project does this. It solves a documented open problem in 3DGS robotics.

---

## Repository Structure

```
roboscene-plus/
├── CLAUDE.md                  ← YOU ARE HERE (always read first)
├── README.md                  ← Public-facing project README
├── config.yaml                ← All paths and hyperparameters
├── requirements.txt           ← Python dependencies
├── environment.yml            ← Conda environment spec
│
├── data/
│   ├── raw/                   ← Original video file (room.mp4)
│   ├── frames/                ← Extracted frames (frame_0001.jpg ...)
│   └── vggt_out/              ← VGGT outputs (COLMAP format + depth maps)
│
├── scripts/                   ← One script per pipeline stage
│   ├── 01_extract_frames.py
│   ├── 02_run_vggt.py
│   ├── 03_train_splat.py
│   ├── 04_run_semantic.py
│   ├── 05_lift_semantics_3d.py
│   ├── 06_compute_confidence.py
│   ├── 07_complete_dead_zones.py
│   ├── 08_build_scene_graph.py
│   └── 09_query_scene.py
│
├── outputs/
│   ├── splat/                 ← Trained Gaussian Splat (.ply, .splat)
│   ├── semantic/              ← Per-frame semantic masks (JSON + PNG)
│   ├── objects_3d.json        ← 3D bounding boxes per object
│   ├── confidence_map.npy     ← 3D voxel confidence grid
│   ├── navigability_map.png   ← Bird's-eye confidence visualisation
│   └── scene_graph.json       ← Final robot-queryable scene graph
│
├── ucl_gpu/                   ← UCL CS GPU job scripts
│   ├── setup_env.sh           ← First-time environment setup on cream/vanilla
│   ├── run_vggt_job.sh        ← VGGT inference job
│   ├── run_splat_job.sh       ← Gaussian Splatting training job
│   └── run_semantic_job.sh    ← Grounded SAM2 segmentation job
│
├── app/
│   ├── app.py                 ← Gradio demo application
│   ├── requirements.txt       ← App-specific deps
│   └── assets/
│       ├── pipeline_diagram.svg
│       └── banner.png
│
└── notebooks/
    └── debug_visualise.ipynb  ← Quick visual checks at each stage
```

---

## Technology Stack

| Component | Tool | Runs on |
|---|---|---|
| Frame extraction | ffmpeg-python | M4 Pro |
| 3D reconstruction | VGGT (CVPR 2025 Best Paper, Meta) | UCL GPU (cream/vanilla) |
| Gaussian Splatting | gsplat (nerfstudio) | UCL GPU |
| Semantic segmentation | Grounded SAM2 | UCL GPU |
| 3D lifting | Custom numpy/open3d | M4 Pro |
| Confidence map | Custom numpy | M4 Pro |
| Dead zone inpainting | LaMa (simple-lama-inpainting) | M4 Pro (CPU) |
| Scene graph | Custom Python | M4 Pro |
| Query interface | Anthropic Claude API (claude-sonnet-4-5) | API |
| Demo UI | Gradio + gsplat.js | Hugging Face Spaces |

---

## UCL CS GPU Access (No Supervisor Required)

### Your Three Options

**Option A — Remote Workstation (BEST for this project)**
- Machines: RTX 4070 Ti SUPER, RTX 4090, RTX PRO 6000 (x2)
- Booking: https://mydesk.cs.ucl.ac.uk (CS account login)
- Max session: 3 days per booking, one booking at a time
- Access from home: SSH tunnel via knuckles.cs.ucl.ac.uk
- **Use this for VGGT + Gaussian Splatting training runs**

SSH into booked or timeshare machine (correct syntax — use -l flag):
```bash
# The correct UCL CS format (confirmed pattern):
ssh -l YOUR_CS_USERNAME -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk MACHINE_NAME.cs.ucl.ac.uk

# Examples:
ssh -l YOUR_CS_USERNAME -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk sentinel.cs.ucl.ac.uk
ssh -l YOUR_CS_USERNAME -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk cream.cs.ucl.ac.uk
```

Add to ~/.ssh/config to avoid typing this every time:
```bash
Host ucl-jump
  HostName knuckles.cs.ucl.ac.uk
  User YOUR_CS_USERNAME

Host ucl-cream
  HostName cream.cs.ucl.ac.uk
  User YOUR_CS_USERNAME
  ProxyJump ucl-jump

Host ucl-gpu
  HostName sentinel.cs.ucl.ac.uk   # update to whichever machine you book
  User YOUR_CS_USERNAME
  ProxyJump ucl-jump

# Then just type: ssh ucl-cream  or  ssh ucl-gpu
```

For Guacamole (browser-based remote desktop for booked workstations):
```bash
# SSH tunnel from Mac (if not on UCL network):
ssh -L 8081:sentinel.cs.ucl.ac.uk:8443 YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk
# Then visit https://localhost:8081/guacamole in browser
```

**Option B — Timeshare Machines (good for overnight jobs)**
- Machines: cream / vanilla — each has 4x Quadro RTX 6000, 375GB RAM
- Access: SSH via knuckles gateway, no booking needed, 24/7
- Etiquette: only use 1 GPU at a time (`source /usr/local/cuda/CUDA_VISIBILITY.csh`)
- Risk: shared, other users present — don't run interactive foreground jobs

```bash
# Correct syntax:
ssh -l YOUR_CS_USERNAME -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk cream.cs.ucl.ac.uk

# Set up Python 3.11:
source /opt/Python/Python-3.11.5_Setup.csh

# Restrict to 1 GPU:
source /usr/local/cuda/CUDA_VISIBILITY.csh
```

**Option C — Lab Machines (on-campus only)**
- Lab 1.05: 24 machines, 128GB RAM, RTX 3090 — best lab GPU
- Lab 1.21: 30 machines, 32GB RAM, RTX 4070 Super Ti
- SSH remotely or use in person
- Risk: machines reboot Monday/Thursday evenings and randomly

### GPU Decision Matrix

| Task | Time on M4 Pro | Time on UCL GPU | Use |
|---|---|---|---|
| VGGT inference (60 frames) | ~15 min | ~3 min (RTX 4090) | UCL GPU |
| Gaussian Splatting training | ~90 min | ~20 min (RTX 6000) | UCL GPU |
| Grounded SAM2 (90 frames) | ~30 min | ~8 min | UCL GPU |
| Confidence map (numpy) | ~5 min | same | M4 Pro |
| LaMa inpainting (CPU) | ~2 min/patch | same | M4 Pro |
| Scene graph build | <1 min | same | M4 Pro |
| Claude API queries | instant | same | M4 Pro |
| Gradio app dev | — | — | M4 Pro |

**Rule of thumb:** Anything touching CUDA models → UCL GPU. Everything else → M4 Pro.

### First-Time Setup on UCL GPU

```bash
# 1. Get CS account at: https://tsg.cs.ucl.ac.uk/apply-for-an-account/
# 2. SSH in (see above)
# 3. Run this once:

source /opt/Python/Python-3.11.5_Setup.csh
pip install --user conda  # if conda not available, use venv
python3 -m venv ~/roboscene_env
source ~/roboscene_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install gsplat open3d numpy tqdm pillow huggingface_hub
pip install git+https://github.com/facebookresearch/segment-anything-2.git
# Clone VGGT
git clone https://github.com/facebookresearch/vggt.git ~/vggt
cd ~/vggt && pip install -r requirements.txt

# 4. Clone your project
git clone https://github.com/JesonRamesh/roboscene-plus.git ~/roboscene-plus
```

### Storage Architecture — Critical to Understand

```
UCL Home Directory (~/)  = 10GB  — PERMANENT across sessions
UCL Session Scratch      = 1TB   — WIPED when session ends (/scratch0/YOUR_CS_USERNAME/)
Your Mac                 = permanent — final destination for all outputs
HuggingFace Dataset      = free, permanent — for .splat and JSON files
```

**The golden rule:** Run training on /scratch0/ (fast, 1TB). Copy outputs to your Mac before logging off. Your home dir (10GB) can store code and the conda env, but NOT large model outputs.

### Transferring Files — Correct Commands

```bash
# Upload frames TO UCL GPU scratch (start of Sessions 2 and 4):
scp -r -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk \
  ./data/frames/ \
  YOUR_CS_USERNAME@sentinel.cs.ucl.ac.uk:/scratch0/YOUR_CS_USERNAME/frames/

# Download VGGT outputs (do BEFORE session ends — Session 2):
scp -r -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk \
  YOUR_CS_USERNAME@sentinel.cs.ucl.ac.uk:/scratch0/YOUR_CS_USERNAME/vggt_out/ \
  ./data/

# Download scene.ply — THE most critical file (Session 3):
scp -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk \
  YOUR_CS_USERNAME@sentinel.cs.ucl.ac.uk:/scratch0/YOUR_CS_USERNAME/outputs/scene.ply \
  ./outputs/splat/

# Download semantic masks (Session 4):
scp -r -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk \
  YOUR_CS_USERNAME@sentinel.cs.ucl.ac.uk:/scratch0/YOUR_CS_USERNAME/semantic/ \
  ./outputs/

# Use rsync for large folders (resumable if your connection drops):
rsync -avz -e "ssh -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk" \
  YOUR_CS_USERNAME@sentinel.cs.ucl.ac.uk:/scratch0/YOUR_CS_USERNAME/outputs/ \
  ./outputs/

# Replace 'sentinel' with the actual machine name you booked,
# or 'cream'/'vanilla' for timeshare machines.
```

### End-of-Session Checklist (GPU sessions only)
```bash
# In your GPU SSH terminal — copy scratch to home before closing:
cp -r /scratch0/$USER/vggt_out/ ~/roboscene-plus/data/      # Session 2
cp -r /scratch0/$USER/outputs/ ~/roboscene-plus/outputs/     # Session 3, 4

# In a NEW terminal on your Mac — scp from home to Mac:
scp -r -J YOUR_CS_USERNAME@knuckles.cs.ucl.ac.uk \
  YOUR_CS_USERNAME@machine.cs.ucl.ac.uk:~/roboscene-plus/outputs/ \
  ./outputs/

# ONLY THEN close the session or disconnect.
```

---

## Environment Variables (Required)

Create a `.env` file in project root (never commit this):

```bash
ANTHROPIC_API_KEY=sk-ant-...     # For scene graph query interface
HF_TOKEN=hf_...                   # For Hugging Face deployment
```

Load with: `export $(cat .env | xargs)` before running scripts.

---

## Pipeline Stages — Status Tracking

Update this as you complete each stage:

| Stage | Script | Status | Output |
|---|---|---|---|
| 1. Extract frames | extract_frames.py | ✅ DONE | data/frames/ (80 frames) |
| 2. VGGT reconstruction | run_vggt.py | ✅ DONE (dry run MPS verified) | data/vggt_out/ |
| 3. Gaussian Splatting | train_splat.py | ⬜ TODO | outputs/splat/scene.ply |
| 4. Grounded SAM2 | run_semantic.py | ⬜ TODO | outputs/semantic/ |
| 5. 3D semantic lifting | lift_semantics_3d.py | ⬜ TODO | outputs/objects_3d.json |
| 6. Confidence map | compute_confidence.py | ⬜ TODO | outputs/confidence_map.npy |
| 7. Dead zone completion | complete_dead_zones.py | ⬜ TODO | outputs/splat/scene_tagged.ply |
| 8. Scene graph | build_scene_graph.py | ⬜ TODO | outputs/scene_graph.json |
| 9. Query interface | query_scene.py | ⬜ TODO | scripts/query_scene.py |
| 10. Gradio app | app/app.py | ⬜ TODO | Deployed on HF Spaces |

Mark stages: ⬜ TODO → 🔄 IN PROGRESS → ✅ DONE → ❌ BLOCKED (with reason)

---

## Claude Code Session Rules

### How to start every session
```bash
# In your project root:
claude  # launches Claude Code

# First message to Claude Code every session:
"Read CLAUDE.md and the current state of the codebase.
Tell me what's done and what we're building today."
```

### Session discipline
- One pipeline stage per session. Don't context-switch mid-session.
- If a script fails: paste the full error + the script content into the same session. Don't start a new session.
- End every session by updating the Status column above and committing: `git add -A && git commit -m "Session N: [what was built]"`
- Never let Claude Code overwrite a working script without reading it first. Say "read scripts/XX_name.py first, then improve it to also..."

### What Claude Code is good at in this project
- Writing boilerplate (argparse, logging, path handling)
- Writing the integration glue between models (dtype conversions, coordinate frame transforms)
- Debugging stack traces — paste error + relevant code, it will fix it
- Writing the Gradio UI (tell it exactly what tabs and components you want)
- Writing the jobscripts for UCL GPU
- Generating the pipeline SVG diagram for the README

### What to do yourself (don't delegate to Claude Code)
- Filming the room
- Making the booking at mydesk.cs.ucl.ac.uk
- Uploading files to UCL GPU / HuggingFace
- Checking that rendered outputs actually look correct (open the .ply in a viewer)
- Recording the demo video

---

## Session-by-Session Plan

### Session 1 — Scaffold + Frame Extraction (Day 1, ~2 hours)
**Goal:** Working repo structure + frames extracted from video
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 1: project scaffold and
frame extraction. Create the full directory structure from the Repository
Structure section. Then write scripts/01_extract_frames.py that takes
--video_path and --output_dir and --fps (default 1.0) arguments, uses
ffmpeg-python to extract frames as JPG files named frame_0001.jpg etc,
prints a summary of how many frames were extracted and the total
duration. Add error handling for missing ffmpeg. Also create config.yaml
with all paths defaulting to the structure in CLAUDE.md, and a
requirements.txt with all dependencies. Use Python 3.11, pathlib
throughout, argparse for all scripts.
```

**Expected output after this session:**
- `scripts/01_extract_frames.py` — runs without error on your video
- `config.yaml` — all paths defined
- `requirements.txt` — all deps listed
- `data/frames/` — ~60–90 JPG files
- Commit: "Session 1: scaffold and frame extraction"

---

### Session 2 — VGGT Integration (Day 2, ~3 hours)
**Goal:** VGGT running on your frames, COLMAP-format output ready for gsplat
**Run on:** UCL GPU (book a Remote Workstation for this session)

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 2: VGGT reconstruction.
Write scripts/02_run_vggt.py that:
1. Clones or imports VGGT from ~/vggt (already installed on this machine).
   Load the model from HuggingFace: facebook/VGGT-1B.
2. Accepts --frames_dir, --output_dir, --batch_size (default 30) args.
3. Loads all JPG frames from frames_dir, sorts them by name.
4. Runs VGGT inference in batches of batch_size frames to avoid OOM.
5. Exports results in COLMAP sparse format to output_dir/sparse/:
   cameras.bin, images.bin, points3D.bin — use VGGT's built-in
   export_to_colmap() function (check vggt/utils/colmap.py).
6. Also saves: depth maps as output_dir/depths/frame_XXXX_depth.npy,
   camera extrinsics as output_dir/camera_poses.json (dict of
   {frame_name: 4x4 extrinsic matrix as list}).
7. Prints timing for inference and export steps.
8. Detects CUDA vs MPS vs CPU automatically.
Also write ucl_gpu/run_vggt_job.sh — a shell script that activates the
venv, runs this script with the right paths, and uses nohup to keep
running if SSH disconnects.
```

**Expected output after this session:**
- `data/vggt_out/sparse/cameras.bin, images.bin, points3D.bin`
- `data/vggt_out/depths/` — one .npy per frame
- `data/vggt_out/camera_poses.json`
- Commit: "Session 2: VGGT reconstruction pipeline"

---

### Session 3 — Gaussian Splatting Training (Day 2–3, runs overnight)
**Goal:** Trained .ply file, interactive web viewer working
**Run on:** UCL GPU

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 3: Gaussian Splatting
training. Write scripts/03_train_splat.py that:
1. Takes --colmap_dir (the vggt_out/ folder), --output_dir, --iterations
   (default 15000) args.
2. Calls gsplat's simple_trainer via subprocess with the correct args:
   python -m gsplat.scripts.simple_trainer default
   --data_dir {colmap_dir} --result_dir {output_dir}
   --max_steps {iterations} --data_factor 1
3. After training completes, finds the output .ply file and copies it to
   outputs/splat/scene.ply.
4. Converts scene.ply to scene.splat format using the ply_converter from
   gsplat (for web viewer compatibility).
5. Prints the final PSNR metric from the training log.
Also write ucl_gpu/run_splat_job.sh — nohup shell script that runs
training with output redirected to logs/splat_train.log.
The script should log nvidia-smi output at the start.
```

**Expected output after this session:**
- `outputs/splat/scene.ply` — the trained Gaussian Splat (~100–500MB)
- `outputs/splat/scene.splat` — web-compatible format
- A rendered test image confirming the splat looks correct
- Commit: "Session 3: Gaussian Splatting training complete"

---

### Session 4 — Grounded SAM2 Semantic Segmentation (Day 3, ~3 hours)
**Goal:** Per-frame semantic masks for all objects
**Run on:** UCL GPU

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 4: semantic segmentation
with Grounded SAM2. Write scripts/04_run_semantic.py that:
1. Takes --frames_dir, --output_dir, --labels (comma-separated string,
   e.g. "chair,desk,laptop,monitor,lamp,bookshelf,door,window,plant"),
   --device (cuda/mps/cpu).
2. Loads GroundingDINO (use groundingdino-py package) for open-vocab
   detection. Box threshold 0.3, text threshold 0.25.
3. Loads SAM2 for mask refinement (use sam2 package, model
   sam2.1_hiera_large).
4. For each frame: run GroundingDINO to get boxes for each label, then
   run SAM2 to get precise masks. If multiple boxes per label, keep the
   highest-confidence one.
5. Saves per-frame JSON to output_dir/frame_XXXX.json:
   {label: {bbox: [x1,y1,x2,y2], confidence: float, mask_rle: <RLE>}}
   Use pycocotools for RLE encoding of masks.
6. Saves visual debug PNGs to output_dir/debug/ with coloured mask
   overlays and label text.
7. At the end, prints a summary: which labels were found in what
   percentage of frames.
Handle the MPS SAM2 bicubic upsampling fix: set env var
PYTORCH_ENABLE_MPS_FALLBACK=1 before any torch import.
Also write ucl_gpu/run_semantic_job.sh.
```

**Expected output after this session:**
- `outputs/semantic/frame_XXXX.json` for each frame
- `outputs/semantic/debug/*.png` — visual confirmation masks look right
- Commit: "Session 4: Grounded SAM2 semantic segmentation"

---

### Session 5 — 3D Semantic Lifting (Day 4, ~2 hours)
**Goal:** 3D bounding boxes and centroids for every detected object
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 5: 3D semantic lifting.
Write scripts/05_lift_semantics_3d.py that:
1. Loads: semantic JSONs from outputs/semantic/, depth maps from
   data/vggt_out/depths/ (as .npy), camera_poses.json from
   data/vggt_out/, and camera intrinsics from the COLMAP cameras.bin
   (parse with read_cameras_binary from the colmap utils).
2. For each (frame, label): decode the RLE mask, get all masked pixel
   coordinates (u, v). For each pixel: depth_val = depth_map[v, u].
   Unproject to 3D: point_camera = depth_val * inv(K) @ [u, v, 1].T.
   Transform to world: point_world = R @ point_camera + t (from extrinsic).
3. Accumulate all world-space 3D points per label across all frames into
   per-object point clouds.
4. For each object point cloud:
   a. Remove outliers: open3d statistical_outlier_removal
      (nb_neighbors=20, std_ratio=2.0).
   b. Compute axis-aligned bounding box: min/max xyz.
   c. Compute centroid (mean of inlier points).
   d. Compute approximate volume in m³.
   e. Count frames_seen and total_points.
5. Save outputs/objects_3d.json:
   {label: {centroid_3d: [x,y,z], bbox_min: [x,y,z], bbox_max: [x,y,z],
    volume_m3: float, num_points: int, frames_seen: int,
    reconstruction_confidence: null}}  # filled in Session 6
6. Also save a top-down 2D scatter plot (matplotlib) of all object
   centroids for visual verification — outputs/object_positions_2d.png.
```

**Expected output after this session:**
- `outputs/objects_3d.json` with 3D positions for all detected objects
- `outputs/object_positions_2d.png` — sanity check that positions look right
- Commit: "Session 5: 3D semantic lifting and object localisation"

---

### Session 6 — Confidence Map (Day 5, ~3 hours) ★ THE NOVEL FEATURE
**Goal:** Voxel confidence grid + navigability map + tagged Gaussian provenance
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 6: the confidence map —
our novel contribution. This is the feature that differentiates
RoboScene+ from every other submission.

Write scripts/06_compute_confidence.py that:
1. Load scene.ply using open3d. Extract Gaussian positions (xyz) and
   opacities. Note: 3DGS .ply files store Gaussians as points with
   attributes. The opacity column in gsplat output is named 'opacity'
   and is pre-sigmoid (apply sigmoid to get 0–1 value).
2. Create a 3D voxel grid:
   - Compute scene bounding box from Gaussian positions + 10% padding.
   - Voxel size = 0.05m (configurable via config.yaml).
   - Shape: (Nx, Ny, Nz) = ceil(scene_extent / voxel_size).
3. For each Gaussian: find its voxel index, add its opacity to
   voxel_opacity_sum[ix, iy, iz] and increment voxel_count[ix, iy, iz].
4. gaussian_density = voxel_opacity_sum / (voxel_count + 1e-6).
   Normalise to 0–1 by dividing by 95th percentile.
5. Camera coverage: load camera_poses.json. For each voxel centroid,
   compute a coverage score = number of cameras within 3m that have
   positive dot product between (voxel - camera_pos) and camera_forward.
   Normalise to 0–1 by dividing by max coverage count.
6. confidence = 0.6 * gaussian_density + 0.4 * camera_coverage.
7. Save as outputs/confidence_map.npy. Save metadata JSON:
   {voxel_size, origin_xyz, shape_xyz, min_confidence, max_confidence,
    pct_high, pct_medium, pct_low}.
8. Generate outputs/navigability_map.png:
   - Take max confidence along Y axis (bird's-eye view).
   - Red = 0–0.3, amber = 0.3–0.7, green = 0.7–1.0.
   - Overlay object centroids as labelled dots (from objects_3d.json).
   - Add a confidence legend.
   - This PNG is a KEY figure for the README and demo.
9. Update objects_3d.json: for each object, compute its
   reconstruction_confidence as the mean confidence of voxels within
   its bounding box. Classify provenance:
   >0.7 = "observed", 0.3–0.7 = "sparse", <0.3 = "inferred".

Also write a separate function tag_gaussian_provenance(scene_ply_path,
confidence_map, output_path) that loads the .ply, adds a 'provenance'
scalar attribute to each Gaussian based on the confidence of its voxel
(0=observed, 1=sparse, 2=inferred), and saves as
outputs/splat/scene_confidence_tagged.ply. This is the novelty: a
confidence-tagged Gaussian Splat file.
```

**Expected output after this session:**
- `outputs/confidence_map.npy`
- `outputs/navigability_map.png` — the hero figure
- `outputs/splat/scene_confidence_tagged.ply`
- `outputs/objects_3d.json` updated with confidence + provenance
- Commit: "Session 6: confidence-aware scene analysis — novel contribution"

---

### Session 7 — Dead Zone Completion (Day 6, ~2 hours)
**Goal:** Inpainted dead zones, LaMa running, scene complete
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 7: dead zone completion.
Write scripts/07_complete_dead_zones.py that:
1. Load confidence_map.npy and metadata JSON.
2. Find dead zone voxel clusters: threshold confidence < 0.3, use
   scipy.ndimage.label to find connected clusters. Filter to clusters
   with > 100 voxels (ignore tiny gaps). Print how many dead zones found.
3. For each dead zone cluster (limit to 5 largest for MVP):
   a. Compute cluster centroid in world coordinates.
   b. Synthesise a virtual camera: position 1.5m from centroid looking
      toward it. Camera intrinsics: use same K as training cameras.
   c. Render this virtual view from scene.ply using gsplat's Python
      renderer API (from gsplat import rasterization).
   d. The rendered image will look blurry/artefact-filled in the dead
      zone. That blurry region IS the inpainting mask (pixels with
      alpha < 0.5 in the rendering).
   e. Run LaMa inpainting: from simple_lama_inpainting import SimpleLama.
      lama = SimpleLama(); result = lama(rendered_image, mask).
   f. Save inpainted render as outputs/dead_zones/dz_{i}_inpainted.png.
4. Print a summary: N dead zones found, N processed, total volume of
   dead zone voxels in m³.
5. Generate outputs/dead_zone_report.json:
   {num_dead_zones, total_dead_volume_m3, zones: [{centroid, volume,
    size_voxels, processed: bool}]}
NOTE: For MVP, we do not back-project inpainted pixels into new
Gaussians (that's complex). Instead, we document this as the next step
in the README Future Work section. The key deliverable is the confidence
map and the report — not perfect inpainting.
```

**Expected output after this session:**
- `outputs/dead_zones/dz_X_inpainted.png` for each processed dead zone
- `outputs/dead_zone_report.json`
- Commit: "Session 7: dead zone identification and completion"

---

### Session 8 — Scene Graph + Claude API (Day 7, ~2 hours)
**Goal:** Queryable scene graph, working chat interface
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 8: scene graph and
Claude API query interface.

Part A — Build scripts/08_build_scene_graph.py:
1. Load objects_3d.json (with confidence and provenance fields).
2. Build scene graph nodes: each object becomes a node with fields:
   {id, label, position_3d, bbox_min, bbox_max, volume_m3,
    reconstruction_confidence, provenance, frames_seen}.
3. Compute spatial relation edges automatically:
   - "on_top_of": obj A centroid Y > obj B bbox_max Y AND
     obj A centroid XZ is within obj B bbox XZ ± 0.15m
   - "next_to": centroid distance < 0.7m (but not on_top_of)
   - "near_wall": centroid within 0.25m of scene bounding box face
   - "between": A's XZ centroid lies between B and C's XZ centroids
     (within 0.3m of the line BC)
4. Add room summary node:
   {room_dimensions_m, num_objects, num_low_confidence_objects,
    num_inferred_objects, navigability_coverage_pct}
5. Save outputs/scene_graph.json. Also print a human-readable
   summary of the room.

Part B — Write scripts/09_query_scene.py:
1. Load ANTHROPIC_API_KEY from environment.
2. Load scene_graph.json.
3. System prompt (embed verbatim in the script):
   "You are a robot spatial reasoning assistant for a humanoid robot.
   You have access to the 3D reconstruction of an indoor room.
   Here is the complete scene graph:
   {scene_graph_json}
   When answering navigation questions:
   - Always give 3D coordinates as (X.Xm, Y.Ym, Z.Zm) from room origin
   - Always cite the reconstruction_confidence score for the target object
     and explain what it means (>0.7 = well-observed, 0.3-0.7 = partially
     seen, <0.3 = inferred/uncertain)
   - For path questions, list waypoints and their confidence
   - Be concise but precise — you are feeding a robot navigation planner"
4. Interactive query loop: read from stdin, call claude-sonnet-4-5,
   stream the response to stdout. Also expose as a function
   query_scene(question: str) -> str for the Gradio app.
5. Include 4 example queries as a list EXAMPLE_QUERIES in the module
   that the Gradio app will use as preset buttons.
```

**Expected output after this session:**
- `outputs/scene_graph.json`
- `scripts/09_query_scene.py` — interactive CLI works
- Running `python scripts/09_query_scene.py` and typing "where is the desk?" returns a useful answer
- Commit: "Session 8: scene graph and Claude API query interface"

---

### Session 9 — Gradio App + Deployment (Day 8, ~3 hours)
**Goal:** Professional deployed demo on Hugging Face Spaces
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md. Today we are building Session 9: the Gradio demo app.

Write app/app.py as a professional Gradio demo. Requirements:

THEME: Dark theme using gr.themes.Base() with these CSS overrides:
- Background: #0f0f1a
- Card surfaces: #1a1a2e  
- Accent purple: #7F77DD
- Text: #e0e0e0
- Success green, amber warning, red danger for confidence badges

LAYOUT: gr.Blocks with a header banner, then 4 tabs:

TAB 1 "🔍 Scene Explorer":
- Left column: gr.HTML embedding gsplat.js viewer as iframe loading
  scene.splat from HuggingFace dataset URL (load from env var SPLAT_URL)
- Right column: gr.Image showing navigability_map.png, gr.Markdown
  showing room summary from scene_graph.json

TAB 2 "📊 Object Map":
- gr.Dataframe showing all objects with columns:
  Object | Position | Confidence | Provenance | Size (m³)
- Confidence column rendered as coloured HTML badges:
  green pill for >0.7, amber for 0.3–0.7, red for <0.3
- gr.Plot showing a 2D scatter plot of object positions (bird's-eye)
  coloured by confidence

TAB 3 "🤖 Robot Query":
- gr.Markdown: "Ask a question about the scene — where objects are,
  navigation paths, confidence levels"
- Row of gr.Button for each item in EXAMPLE_QUERIES from query_scene.py
  (clicking fills the text box)
- gr.Textbox for user input
- gr.Button "Ask Robot" that calls query_scene() and streams response
- gr.Textbox (interactive=False) showing the streaming Claude response
- gr.Markdown showing the raw scene graph JSON in a collapsible section

TAB 4 "⚙️ Pipeline":
- gr.Image showing app/assets/pipeline_diagram.svg
- gr.Markdown with brief explanation of each stage

LOADING: Load all JSON outputs at app startup, not on-demand.
Handle missing files gracefully with informative error messages.

Also write app/requirements.txt and a README for the HF Space.
Include instructions for setting ANTHROPIC_API_KEY as a HF Space secret.
```

**Expected output after this session:**
- `app/app.py` running locally at localhost:7860
- `app/requirements.txt`
- Looks professional — not default Gradio
- Commit: "Session 9: Gradio demo app"

---

### Session 10 — README + Polish (Day 10, ~2 hours)
**Goal:** Submission-ready GitHub repo
**Run on:** M4 Pro

Prompt to use:
```
Read CLAUDE.md and the full codebase. Today we are building Session 10:
the final README and project polish.

1. Generate app/assets/pipeline_diagram.svg: a clean horizontal flow
   diagram showing 7 stages with icons, colours matching our purple
   accent (#7F77DD for reconstruction stages, teal for semantic stages,
   amber for confidence stage, coral for output stage). Each stage box:
   stage name + tool name underneath. Arrows between stages. Clean,
   professional, suitable for a portfolio README.

2. Write the complete README.md following this structure exactly:
   - Hero: title + tagline + 3 badges (HF Space, Python 3.11, CVPR 2025)
   - Demo video embed (placeholder: [Demo Video](link))
   - Problem statement (3 sentences, robot framing)
   - Pipeline diagram image
   - Quick Start section (git clone + pip install + one command)
   - Key Output: navigability_map.png with caption
   - Design Choices section (this is the most important section):
     * VGGT over COLMAP: why, with one sentence from the paper
     * gsplat over nerfstudio: CUDA efficiency, no FlashGS (requires NVIDIA)
     * Open-vocabulary over fixed-class: why this matters for deployment
     * Confidence map: THE NOVEL CONTRIBUTION — explain the dead zone problem,
       cite that robotics surveys identify this as an open problem (success
       rates below 50% for multi-stage tasks), explain our solution
     * Claude API: mirrors KinetIQ's VLM reasoning architecture
   - Limitations (honest, shows maturity)
   - Future Work (multi-visit change detection, dynamic objects)
   - Citation section

3. Run a final check: make sure all imports in all scripts are correct,
   all output paths are consistent with CLAUDE.md, and the repo README
   is accurate.
```

**Expected output after this session:**
- `README.md` — complete, professional, submission-ready
- `app/assets/pipeline_diagram.svg`
- Final commit: "Session 10: README and project polish — submission ready"

---

## Common Errors and Fixes

**SAM2 on MPS — bicubic upsampling crash:**
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch  # must import AFTER setting env var
```

**VGGT OOM on GPU (too many frames at once):**
Use --batch_size 30. If still OOM, reduce to 15.
On MPS (M4 Pro), max ~5 frames per batch at 518×518. Use --batch_size 5.

**VGGT confidence threshold too high (0 points exported):**
The script uses adaptive thresholding: if conf_threshold yields 0 points,
it falls back to the 50th percentile. Default is now 1.5 (was 5.0).
On MPS dry run, conf range was 1.0–2.14.

**pycolmap version incompatibility (v3.13+ / v4.0):**
We use a custom COLMAP binary writer (`scripts/colmap_utils.py`) instead
of pycolmap. This avoids all version issues with the Frame/Rig API changes.

**gsplat import error on UCL GPU:**
```bash
pip install gsplat==1.3.0  # pin the version
```

**SCP through jump host:**
```bash
scp -J username@knuckles.cs.ucl.ac.uk username@cream.cs.ucl.ac.uk:~/path ./local/
```

**UCL GPU scratch wiped — lost outputs:**
Always `cp -r /scratch0/$USER/ ~/roboscene-plus/outputs/` before session ends.

**Claude API rate limit:**
Use streaming (stream=True) and add a 0.5s delay between API calls.

**open3d not rendering on remote machine:**
Use `open3d.visualization.draw_plotly()` instead of `draw()` — works headlessly.

---

## Design Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| COLMAP export | Custom binary writer (not pycolmap) | pycolmap 3.13+/4.0 broke Image API; our writer is zero-dependency |
| Pose estimation | VGGT (not COLMAP) | CVPR 2025 Best Paper, 1-second inference, no feature matching failures |
| Gaussian Splatting | gsplat (not nerfstudio, not FlashGS) | CUDA-native, well-maintained, FlashGS requires NVIDIA CUDA only |
| Semantic segmentation | Grounded SAM2 (not fixed-class) | Open vocabulary, aligns with Humanoid's VLM architecture |
| Confidence metric | Opacity density × camera coverage | Computationally cheap, interpretable, doesn't require ground truth |
| Query interface | Claude API (not local LLM) | Mirrors KinetIQ System 2's VLM reasoning architecture |
| Deployment | HF Spaces + gsplat.js | Free, permanent URL, WebGL renders .splat natively in browser |
| GPU (UCL) | CS Remote Workstation or cream/vanilla | No supervisor required, CS account sufficient |

---

## Key URLs

- UCL CS GPU booking: https://mydesk.cs.ucl.ac.uk
- UCL CS account apply: https://tsg.cs.ucl.ac.uk/apply-for-an-account/
- UCL CS SSH gateway: knuckles.cs.ucl.ac.uk (taught students)
- Timeshare machines: cream.cs.ucl.ac.uk, vanilla.cs.ucl.ac.uk
- VGGT repo: https://github.com/facebookresearch/vggt
- gsplat docs: https://docs.gsplat.studio
- Grounded SAM2: https://github.com/IDEA-Research/Grounded-SAM-2
- HF Space: https://huggingface.co/spaces/JesonRamesh/roboscene-plus (update when live)
- GitHub repo: https://github.com/JesonRamesh/3D-Spatial-Reconstruction

---

## Submission Checklist

- [ ] GitHub repo is public
- [ ] README has demo video, pipeline diagram, design choices
- [ ] HF Space is live and accessible without login
- [ ] Claude API key is set as HF Space secret (not in code)
- [ ] All example queries work in the deployed app
- [ ] navigability_map.png is in the README
- [ ] Design choices section explicitly names the confidence map as novel
- [ ] Limitations section is honest
- [ ] Git history shows incremental development (10 commits minimum)
- [ ] Repo URL ready to paste into Humanoid application form

---

## Progress Notes

### Session 1 (completed)
- Built full project scaffold matching CLAUDE.md structure
- `scripts/extract_frames.py`: ffmpeg-based extraction at 1fps → 80 frames
- All config, requirements, .gitignore, README in place

### Session 2 (completed — dry run on MPS)
- `scripts/run_vggt.py`: Full VGGT pipeline with CUDA→MPS→CPU fallback
- `scripts/colmap_utils.py`: Custom COLMAP binary writer (avoids pycolmap API breakage)
- `ucl_gpu/run_vggt_job.sh`: GPU job script for UCL machines
- Verified on M4 Pro MPS: 5 frames processed in 28s, 95K 3D points exported
- Outputs: cameras.bin, images.bin, points3D.bin, depth maps, camera_poses.json, points.ply
- Key finding: MPS max batch ~5 frames (6GB model + images = OOM at 10)
- Key finding: Depth confidence range is 1.0–2.14 on MPS, so threshold lowered to 1.5

### Next Steps
- **Run full VGGT inference** on UCL GPU (all 80 frames, ~4 min on RTX 4090)
  - Command: `python scripts/run_vggt.py` (no --dry_run)
  - Upload frames first: `scp -r data/frames/ user@machine:/scratch0/user/frames/`
  - Download outputs: `scp -r user@machine:/scratch0/user/vggt_out/ ./data/`
- **Session 3**: Write `scripts/train_splat.py` for Gaussian Splatting with gsplat