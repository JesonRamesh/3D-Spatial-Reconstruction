# CLAUDE.md — RoboScene+ Project Guide

> This file lives in the root of your project repo.
> Claude Code reads it automatically at the start of every session.
> Keep it updated as the project evolves.

---

## Project Identity

**Name:** RoboScene+
**GitHub:** https://github.com/JesonRamesh/3D-Spatial-Reconstruction
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
3D-Spatial-Reconstruction/          ← repo root (Mac: ~/3D-Spatial-Reconstruction)
├── CLAUDE.md                       ← YOU ARE HERE (always read first)
├── README.md
├── config.yaml
├── requirements.txt
│
├── data/
│   ├── raw/                        ← Original video file (room.MOV)
│   ├── frames/                     ← 82 extracted frames ✅
│   ├── mast3r_out/                 ← MASt3R-SLAM COLMAP output ✅
│   │   └── sparse/0/              ← cameras.bin images.bin points3D.bin
│   └── vggt_out/                   ← VGGT outputs (511 poses + depth maps) ✅
│       ├── camera_poses.json
│       └── sparse/
│
├── scripts/
│   ├── extract_frames.py           ← ✅ DONE
│   ├── run_vggt.py                 ← ✅ DONE
│   ├── colmap_utils.py             ← ✅ DONE (custom COLMAP binary writer)
│   ├── run_mast3r_slam.py          ← ✅ DONE (Session 3)
│   ├── train_splat.py              ← ✅ DONE (Session 3, nerfstudio splatfacto)
│   ├── run_semantic.py             ← ✅ DONE (Session 4, Grounded SAM2)
│   ├── lift_semantics_3d.py        ← ✅ DONE (Session 5)
│   ├── compute_confidence.py       ← ✅ DONE (Session 6 ★ novel contribution)
│   ├── complete_dead_zones.py      ← ✅ DONE (Session 7)
│   ├── build_scene_graph.py        ← ⬜ TODO Session 8
│   └── query_scene.py              ← ⬜ TODO Session 8
│
├── outputs/
│   ├── splat_mast3r_v2/            ← FINAL Gaussian Splat ✅
│   │   └── scene.ply              ← 4.35M Gaussians, 60k steps
│   ├── mast3r_out/
│   │   └── room_video.ply         ← MASt3R point cloud (7.3M pts) ✅
│   ├── semantic/                   ← 317 per-frame JSON masks ✅
│   ├── objects_3d.json             ← 10 objects with bbox + confidence ✅
│   ├── object_positions_2d.png     ← top-down scatter plot ✅
│   ├── confidence_map.npy          ← 90×86×118 voxel grid ✅
│   ├── confidence_metadata.json    ← grid params + zone percentages ✅
│   ├── navigability_map.png        ← HERO FIGURE for README ✅
│   ├── scene_confidence_tagged.ply ← Gaussians tagged 0/1/2 ✅
│   ├── dead_zones/                 ← Dead zone images + report ✅
│   ├── dead_zone_report.json
│   ├── dead_zone_summary.png
│   ├── zone_0_original.png
│   ├── zone_0_mask.png
│   └── zone_0_inpainted.png
└── scene_graph.json            ← ⬜ TODO Session 8
│
├── ucl_gpu/
│   ├── run_vggt_job.sh             ← ✅ DONE
│   ├── run_splat_job.sh            ← ✅ DONE
│   └── run_semantic_job.sh         ← ✅ DONE
│
├── app/
│   ├── app.py                      ← Gradio demo
│   ├── requirements.txt
│   └── assets/
│       ├── pipeline_diagram.svg
│       └── banner.png
│
└── notebooks/
    └── debug_visualise.ipynb
```

---

## Technology Stack

| Component | Tool | Runs on |
|---|---|---|
| Frame extraction | ffmpeg-python | M4 Pro |
| 3D reconstruction | VGGT (CVPR 2025 Best Paper) | UCL bluestreak |
| Gaussian Splatting | gsplat | UCL bluestreak |
| Semantic segmentation | Grounded SAM2 | UCL bluestreak |
| 3D lifting | Custom numpy/open3d | M4 Pro |
| Confidence map | Custom numpy | M4 Pro |
| Dead zone inpainting | LaMa (simple-lama-inpainting) | M4 Pro (CPU) |
| Scene graph | Custom Python | M4 Pro |
| Query interface | Anthropic Claude API (claude-sonnet-4-5) | API |
| Demo UI | Gradio + gsplat.js | Hugging Face Spaces |

---

## UCL CS GPU — Confirmed Working Setup

### Machine
- **Host:** bluestreak.cs.ucl.ac.uk
- **GPU:** NVIDIA RTX 4070 Ti SUPER — 16376MB VRAM
- **CUDA:** 12.6 driver / PyTorch cu121 (compatible)
- **OS:** Rocky Linux 9.7
- **CS username:** jrameshs
- **Booking:** https://mydesk.cs.ucl.ac.uk (needs UCL VPN if off-campus)

### SSH Config (~/.ssh/config on Mac — already configured)
```
Host bluestreak
  HostName bluestreak.cs.ucl.ac.uk
  User jrameshs
  ProxyJump jrameshs@knuckles.cs.ucl.ac.uk
  ServerAliveInterval 60
  ServerAliveCountMax 10

Host knuckles
  HostName knuckles.cs.ucl.ac.uk
  User jrameshs
```

Connect with: `ssh bluestreak`

### ⚠️ CRITICAL: Always Do These Two Things First After SSH
```bash
# 1. Switch from tcsh to bash (default shell is tcsh — % prompt breaks everything)
bash

# 2. Activate Python + venv
source /opt/Python/Python-3.11.5_Setup.csh
source /scratch0/jrameshs/roboscene_env/bin/activate

# Verify GPU is accessible:
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4070 Ti SUPER
```

### Environment Layout on bluestreak
```
/scratch0/jrameshs/               ← 1TB scratch (WIPED AT SESSION END)
├── roboscene_env/                ← Python venv (PyTorch 2.5.1+cu121) ✅
├── pip_cache/                    ← pip cache ✅
├── vggt/                         ← VGGT repo ✅
└── roboscene-plus/               ← Working copy of project
    └── data/frames/              ← 82 frames uploaded ✅

/cs/student/ug/2024/jrameshs/    ← 10GB home (persistent across bookings)
```

### ⚠️ STORAGE RULES
- **Scratch = 1TB but WIPED when booking ends**
- **Home = 10GB persistent but DO NOT install packages here** (fills up instantly)
- Set phone alarm 1 hour before booking ends → download outputs to Mac
- PyTorch alone = ~7GB → always goes to scratch

### pip — Always Set Cache Before Installing
```bash
export PIP_CACHE_DIR=/scratch0/jrameshs/pip_cache
# Already added to ~/.bashrc — verify with: echo $PIP_CACHE_DIR
```

### File Transfers (always run from Mac terminal, NOT from bluestreak)

Upload frames to bluestreak:
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  ~/3D-Spatial-Reconstruction/data/frames/ \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/
```

Download VGGT outputs after Session 2:
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/data/vggt_out/ \
  ~/3D-Spatial-Reconstruction/data/
```

Download splat after Session 3:
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/splat/ \
  ~/3D-Spatial-Reconstruction/outputs/
```

Download semantic outputs after Session 4:
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/semantic/ \
  ~/3D-Spatial-Reconstruction/outputs/
```

### Running Jobs in Background (SSH-disconnect safe)
```bash
nohup python scripts/run_vggt.py [args] > logs/vggt.log 2>&1 &
tail -f logs/vggt.log    # watch live
jobs -l                  # check still running
```

### Pulling Latest Code onto bluestreak
```bash
cd /scratch0/jrameshs/roboscene-plus
unset SSH_ASKPASS && unset DISPLAY
git pull
# If prompted: Username=JesonRamesh, Password=GitHub personal access token
```

### Timeshare Alternative (no booking needed)
```bash
ssh -l jrameshs -J jrameshs@knuckles.cs.ucl.ac.uk cream.cs.ucl.ac.uk
bash
source /opt/Python/Python-3.11.5_Setup.csh
source /usr/local/cuda/CUDA_VISIBILITY.csh   # restrict to 1 GPU (etiquette)
# Recreate venv on cream's scratch if first time
```

---

## Session Progress

| Session | What | GPU? | Status |
|---|---|---|---|
| 1 | Scaffold + frame extraction | ❌ | ✅ DONE — 82 frames extracted |
| 2 | VGGT reconstruction | ✅ | ✅ DONE — 511 poses, depth maps in data/vggt_out/ |
| 3 | Gaussian Splatting | ✅ | ✅ DONE — MASt3R-SLAM + nerfstudio splatfacto 60k steps |
| 4 | Grounded SAM2 | ✅ | ✅ DONE — 317 frames, 10 objects, outputs/semantic/ |
| 5 | 3D semantic lifting | ❌ | ✅ DONE (needs bbox fix — see PLAN.md §4 Day 4) |
| 6 | Confidence map ★ | ❌ | ✅ DONE — confidence_map.npy + tagged splat |
| 7 | Dead zone completion | ❌ | ✅ DONE — outputs/dead_zones/ (LaMa inpainting) |
| 8 | Scene graph + Claude API | ❌ | ✅ DONE — outputs/scene_graph.json + query_scene.py working |
| A | Semantic Gaussian painting | ❌ | ✅ DONE — outputs/scene_semantic.ply (628K/4.35M labeled) |
| B | 3D viewer HTML | ❌ | ✅ DONE — app/static/viewer.html (GaussianSplats3D.js) |
| C | Gradio app overhaul | ❌ | ✅ DONE — Tab 1 = 3D viewer (background file server) |
| V | Viewer fix + splat pruning | ❌ | ✅ DONE — 41MB pruned splat, no double-download |
| G | Re-train on raw_images/ (GPU) | ✅ | ⬜ TODO — ucl_gpu/run_colmap_splat.sh |
| D | Fix object bounding boxes | ❌ | ⬜ TODO — re-run lift_semantics_3d.py with tighter filters |
| E | HF Spaces deployment | ❌ | ⬜ TODO |
| F | README + polish | ❌ | ⬜ TODO |

---

## Session 7 — COMPLETE ✅

### Script: scripts/complete_dead_zones.py
**Pipeline:** confidence_map.npy → dead zone detection → closest keyframe → LaMa inpainting → summary figure

**Key design decisions:**
- Dead zone threshold: `confidence < 0.3` (matches low-confidence band from Session 6)
- Connected components: `scipy.ndimage.label()` on dead mask
- Camera source: `data/vggt_out/camera_poses.json` — dict keyed by `frame_XXXX.jpg` with `cam_to_world_4x4` + `intrinsic_3x3`
- Image source: `data/frames/` (all 511 frames matched to 511 poses)
- LaMa resize: images downscaled to max 1024px (multiples of 8) before inference, resized back after
- LaMa singleton: model loaded once at module level, reused across zones
- Fallback: median-fill + blur if `simple-lama-inpainting` unavailable or errors
- LaMa install workaround: `pip3 install /tmp/simple-lama-inpainting` (cloned from GitHub; PyPI build broken on Python 3.13 due to `setuptools.__version__` KeyError + Pillow<10 constraint)
- SSL fix required: `/Applications/Python\ 3.13/Install\ Certificates.command`

**Results on room scene:**
- Dead zones found: 1 large cluster (598,709 voxels = 74.84 m³) — the entire unobserved interior
- Closest camera: frame_0304 (0.81 m from centroid)
- Projection: centroid world=(-0.427, 0.128, 1.134) → pixel (u, v) via K + R^T(P-t)
- Inpainting: LaMa `big-lama.pt` (196 MB), resized 4032×3024 → 1024×768

**Output files:**
```
outputs/dead_zones/
├── dead_zone_report.json       ← {num_found:1, num_processed:1, total_dead_volume_m3:74.84, zones:[...]}
├── dead_zone_summary.png       ← dark-bg 3-col grid: Original | Mask | Inpainted
├── zone_0_original.png         ← frame_0304 keyframe (4032×3024)
├── zone_0_mask.png             ← circle mask radius=50px at projected centroid
└── zone_0_inpainted.png        ← LaMa result (4032×3024, 8.5MB vs 13MB original)
```

**Run command:**
```bash
python3 scripts/complete_dead_zones.py \
  --confidence_map outputs/confidence_map.npy \
  --confidence_metadata outputs/confidence_metadata.json \
  --splat_dir outputs/splat_mast3r_v2/ \
  --output_dir outputs/dead_zones/ \
  --min_zone_voxels 200 \
  --max_zones 5
```

---

## ⚠️ IMPORTANT: Project Direction Change (Read PLAN.md)

The project goal is a **navigable 3D semantic viewer**, not a 2D confidence map.
See `PLAN.md` for root cause analysis and the corrected 7-day plan.

## Immediate Next Action — Session A: Semantic Gaussian Painting (Mac, no GPU)

All inputs ready locally:
- `outputs/splat_mast3r_v2/scene.ply` — 4.35M Gaussians ✅
- `outputs/semantic/frame_XXXX.json` — 317 semantic masks ✅
- `data/mast3r_out/sparse/0/cameras.bin` + `images.bin` — COLMAP poses ✅

Write and run `scripts/paint_semantic_gaussians.py` (see PLAN.md §4 Day 1–2).
Output: `outputs/scene_semantic.ply` — Gaussians colored by semantic class.

Then:
1. Create `app/static/viewer.html` using GaussianSplats3D.js
2. Overhaul `app/app.py` — Tab 1 = interactive 3D viewer
3. Fix object bounding boxes (lift_semantics_3d.py, n_std=1.5)
4. Deploy to HF Spaces

---

## Known Issues and Fixes

**tcsh vs bash:** Always run `bash` first. tcsh breaks venv activate + redirects.

**Quota exceeded (Errno 122):**
```bash
pip cache purge
export PIP_CACHE_DIR=/scratch0/jrameshs/pip_cache
# venv must be at /scratch0/jrameshs/roboscene_env (not ~/roboscene_env)
```

**GitHub clone GUI askpass error on bluestreak:**
```bash
unset SSH_ASKPASS && unset DISPLAY
git clone https://JesonRamesh:YOUR_TOKEN@github.com/JesonRamesh/3D-Spatial-Reconstruction.git /scratch0/jrameshs/roboscene-plus
```

**SAM2 on MPS bicubic crash:**
```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch   # AFTER env var
```

**VGGT batch size:**
- bluestreak (16GB VRAM): --batch_size 30
- M4 Pro MPS: --batch_size 5

**VGGT confidence threshold:**
Default 1.5. Script auto-adapts to 50th percentile if threshold yields 0 points.

**pycolmap incompatibility:**
Use scripts/colmap_utils.py (custom writer) — already implemented, zero deps.

**gsplat install:**
```bash
pip install gsplat==1.3.0   # pin if latest breaks
```

**numpy/scipy conflict:**
```bash
pip install "numpy>=1.26.4"
```

**open3d headless on remote:**
```python
open3d.visualization.draw_plotly([geometry])   # not draw()
```

---

## Design Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| COLMAP export | Custom colmap_utils.py | pycolmap 3.13+/4.0 broke Image API |
| Pose estimation | VGGT not COLMAP | CVPR 2025 Best Paper, 1-pass inference |
| Gaussian Splatting | gsplat not FlashGS/nerfstudio | FlashGS = NVIDIA CUDA only, won't run on Mac |
| Semantics | Grounded SAM2 open-vocab | Mirrors Humanoid KinetIQ VLM architecture |
| Confidence metric | Point density × camera coverage | Cheap, interpretable, no GT needed — uses MASt3R point cloud + VGGT poses |
| Point cloud source | MASt3R-SLAM (outputs/mast3r_out/room_video.ply) | 7.3M points, globally consistent |
| Pose source for coverage | VGGT camera_poses.json (511 frames) | cam_to_world_4x4 matrices, auto-detected by load_poses() |
| Camera coverage algorithm | Camera-centric O(C×r³) not O(V×C) | 200× faster; subsample to 200 cameras |
| Query interface | Claude API | Mirrors KinetIQ System 2 VLM reasoning |
| Deployment | HF Spaces + gsplat.js | Free, permanent URL, WebGL splat viewer |
| GPU venv location | /scratch0 not ~ | Home = 10GB; PyTorch alone = 7GB |
| Shell on bluestreak | bash not tcsh | tcsh breaks standard shell commands |

---

## Session Prompts for Claude Code

### Session 3 — Gaussian Splatting
```
Read CLAUDE.md. Session 3: write scripts/train_splat.py that:
1. Takes --colmap_dir (data/vggt_out/), --output_dir (outputs/splat/),
   --iterations (default 15000) args.
2. Calls gsplat simple_trainer via subprocess:
   python -m gsplat.scripts.simple_trainer default
   --data_dir {colmap_dir} --result_dir {output_dir}
   --max_steps {iterations} --data_factor 1
3. After training: find output .ply, copy to outputs/splat/scene.ply.
4. Print final PSNR from training log. Log nvidia-smi at start.
Also write ucl_gpu/run_splat_job.sh: nohup shell script that activates
/scratch0/jrameshs/roboscene_env, sets PYTHONPATH for vggt,
runs train_splat.py, logs to logs/splat_train.log.
```

### Session 4 — Grounded SAM2
```
Read CLAUDE.md. Session 4: write scripts/run_semantic.py that:
1. Takes --frames_dir, --output_dir, --labels (comma-separated string),
   --device (cuda/mps/cpu) args.
2. Loads Grounded SAM2 (groundingdino-py + sam2).
   Box threshold 0.3, text threshold 0.25. SAM2: sam2.1_hiera_large.
3. Per frame: GroundingDINO detects boxes per label → SAM2 refines masks.
   Keep highest-confidence box per label.
4. Save output_dir/frame_XXXX.json:
   {label: {bbox, confidence, mask_rle}} (pycocotools RLE encoding).
5. Save debug PNGs to output_dir/debug/ with coloured overlays.
6. Print: which labels found in what % of frames.
Handle MPS: set PYTORCH_ENABLE_MPS_FALLBACK=1 before torch import.
Write ucl_gpu/run_semantic_job.sh.
```

### Session 5 — 3D Semantic Lifting
```
Read CLAUDE.md. Session 5: write scripts/lift_semantics_3d.py that:
1. Loads semantic JSONs, depth maps (.npy), camera_poses.json,
   camera intrinsics from colmap cameras.bin (use colmap_utils.py).
2. Per (frame, label): decode RLE mask, get masked pixels (u,v).
   Per pixel: depth = depth_map[v,u].
   point_cam = depth * inv(K) @ [u,v,1].T
   point_world = R @ point_cam + t
3. Accumulate 3D points per label across all frames.
4. Per object: open3d outlier removal (nb=20, std=2.0),
   compute bbox, centroid, volume_m3.
5. Save outputs/objects_3d.json with centroid_3d, bbox_min,
   bbox_max, volume_m3, num_points, frames_seen,
   reconstruction_confidence (null — filled Session 6).
6. Save outputs/object_positions_2d.png — top-down matplotlib scatter.
```

### Session 6 — Confidence Map ★ NOVEL FEATURE
```
Read CLAUDE.md. Session 6: write scripts/compute_confidence.py.
This is our novel contribution — confidence-aware scene analysis.

1. Load outputs/splat/scene.ply. Extract Gaussian xyz + opacities
   (apply sigmoid — gsplat stores pre-sigmoid values in 'opacity' column).
2. Create 3D voxel grid at 0.05m resolution over scene bbox + 10% padding.
3. Per voxel: sum Gaussian opacities → gaussian_density (normalise by 95th pct).
4. Load camera_poses.json. Per voxel: count cameras within 3m with
   positive dot product to voxel direction → camera_coverage (normalise).
5. confidence = 0.6 * gaussian_density + 0.4 * camera_coverage.
6. Save outputs/confidence_map.npy + metadata JSON.
7. Generate outputs/navigability_map.png:
   - Max confidence along Y axis (bird's-eye view)
   - Red=0–0.3, amber=0.3–0.7, green=0.7–1.0
   - Overlay object centroids as labelled dots
   - Add confidence legend
   THIS IS THE HERO FIGURE FOR THE README.
8. Update outputs/objects_3d.json: add reconstruction_confidence
   (mean voxel confidence in bbox) and provenance per object
   (>0.7=observed, 0.3–0.7=sparse, <0.3=inferred).
9. Function tag_gaussian_provenance(): adds provenance int attribute
   (0/1/2) to each Gaussian, saves scene_confidence_tagged.ply.
```

### Session 7 — Dead Zone Completion
```
Read CLAUDE.md. Session 7: write scripts/complete_dead_zones.py:
1. Load confidence_map.npy. Find dead zones: threshold < 0.3,
   scipy.ndimage.label, filter > 100 voxels. Print count found.
2. For 5 largest clusters:
   a. Compute centroid in world coords.
   b. Synthesise virtual camera: 1.5m from centroid, looking at it.
   c. Render view using gsplat Python renderer (from gsplat import rasterization).
   d. Create mask: pixels with alpha < 0.5.
   e. Run LaMa: from simple_lama_inpainting import SimpleLama.
      result = SimpleLama()(rendered_image, mask).
   f. Save outputs/dead_zones/dz_{i}_inpainted.png.
3. Save outputs/dead_zone_report.json:
   {num_dead_zones, total_dead_volume_m3, zones:[...]}
NOTE: Back-projection into new Gaussians = Future Work in README.
```

### Session A — Semantic Gaussian Painting (NEW — replaces planned Session 8 visual output)
```
Read CLAUDE.md and PLAN.md. Session A: write scripts/paint_semantic_gaussians.py.

Goal: color every Gaussian in outputs/splat_mast3r_v2/scene.ply by its semantic
class so the 3D viewer shows a highlighted bedroom.

1. Load scene.ply using a pure-numpy PLY reader (same pattern as lift_semantics_3d.py).
   Extract: x, y, z, f_dc_0, f_dc_1, f_dc_2 for all 4.35M Gaussians.
   Keep ALL other property columns unchanged.

2. Load COLMAP cameras.bin + images.bin from data/mast3r_out/sparse/0/
   using colmap_utils.read_cameras_binary() and read_images_binary().
   Build map: image_name (e.g. "frame_0001.jpg") → (R_w2c, t_w2c, K)
   where K = [[fx,0,cx],[0,fy,cy],[0,0,1]].

3. Define class color map (consistent with existing codebase):
   {bed:#FFB6C1, desk:#20B2AA, chair:#4682B4, laptop:#6495ED,
    monitor:#2F4F4F, fan:#FF6347, lamp:#FFD700, shelf:#CD853F,
    door:#DEB887, window:#87CEEB}
   Build label→index mapping, index→RGB array.

4. Allocate votes array: shape (N_gaussians, n_classes+1) int32 (+1 for unlabeled).

5. For each frame in outputs/semantic/ with a matching camera pose:
   a. Load JSON → build semantic label image (H×W uint8, 0=unlabeled, 1..n=class index)
      Using all object masks, later masks overwrite earlier (same as debug PNG logic).
      Decode RLE masks using pycocotools.
   b. Project all Gaussians vectorized (no Python loop over Gaussians):
      p_cam = (R_w2c @ P.T).T + t_w2c   # N×3
      in_front = p_cam[:,2] > 0.01
      u = fx * p_cam[:,0]/p_cam[:,2] + cx
      v = fy * p_cam[:,1]/p_cam[:,2] + cy
      in_img = in_front & (u>=0)&(u<W)&(v>=0)&(v<H)
   c. For Gaussians in_img: look up label from semantic_image[v_int, u_int].
      Vectorize with np.add.at(votes, (valid_idxs, labels[valid_idxs]), 1).

6. Assign semantic class: cls = argmax(votes, axis=1) where votes.max > 0.

7. Apply semantic colors:
   SH_C0 = 0.28209479177387814
   For labeled Gaussians: f_dc = (class_rgb - 0.5) / SH_C0
   Zero out f_rest_* columns for labeled Gaussians (view-independent color).
   For unlabeled Gaussians: multiply f_dc by 0.25 (grey them out).

8. Write outputs/scene_semantic.ply in same binary format as input.
   Also write outputs/semantic_stats.json: per-class Gaussian count + %.

Args: --splat_ply, --semantic_dir, --cameras_bin, --images_bin, --output_ply,
      --max_frames (default all), --min_votes (default 2).

Print: frames processed, % Gaussians labeled, per-class counts.
```

### Session B — 3D Viewer + App Overhaul (NEW)
```
Read CLAUDE.md and PLAN.md. Session B: create the 3D viewer and overhaul app/app.py.

Part A — app/static/viewer.html:
Create a minimal HTML file using GaussianSplats3D.js from CDN:
  https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.min.js

The HTML must:
1. Fill the entire iframe viewport with a dark background (#0f0f1a).
2. Load a .ply file whose URL is passed as a query parameter:
   ?ply=<url_encoded_path>
   Default to ./scene_semantic.ply if no parameter.
3. Use GaussianSplats3D.Viewer with cameraUp=[0,-1,0],
   initialCameraPosition=[0,0.5,3], selfDrivenMode=true.
4. Show a top-left legend panel with the 10 semantic class colors.
5. Show bottom-right text: "WASD + drag to navigate | Scroll to zoom".
6. No external fonts or heavy dependencies beyond the three.js importmap.

Part B — app/app.py overhaul:
Tab 1 "🔍 3D Scene Viewer" (PRIMARY):
  - gr.HTML of an <iframe> pointing to viewer.html served from app/static/
  - Two gr.Button side by side: "Semantic View" and "Appearance View"
    clicking switches the iframe src URL between scene_semantic.ply and scene.ply
  - Below: object legend (gr.Markdown) showing class colors + names
  - Below: compact room summary (keep existing build_room_summary())

Tab 2 "📊 Object Map": Keep existing dataframe and dead zone section.

Tab 3 "🤖 Robot Query": Keep unchanged.

Tab 4 "⚙️ Pipeline": Keep unchanged.

For the iframe src: Gradio serves files in app/ directory when passed to
  allowed_paths in app.launch(). Use:
  allowed_paths=[str(Path(__file__).parent), str(ROOT/"outputs")]

The iframe src should be:
  "/gradio_api/file=app/static/viewer.html?ply=/gradio_api/file=outputs/scene_semantic.ply"
  (or equivalent Gradio static file URL format for the installed version).

Fallback: if scene_semantic.ply not found, load scene.ply (appearance only).
```

### Session C — Fix Object Bounding Boxes (NEW)
```
Read CLAUDE.md and PLAN.md. Session C: fix lift_semantics_3d.py and regenerate outputs.

Changes to scripts/lift_semantics_3d.py:
1. Change remove_outliers: use n_std=1.5 (was 2.0).
   For objects with >50k points use n_std=1.2.

2. Add volume sanity filter: after outlier removal, if computed volume > 3.0 m³
   AND label not in {"bed", "sofa", "wardrobe", "couch"}:
   trim to points within 85th-percentile radius from centroid, recompute.

3. Add confidence filter: only lift pixels where the frame JSON has confidence > 0.35
   for the detected object (already stored in semantic JSON per label).

4. For structural classes {"window", "door", "wall"}: skip bounding box, use
   centroid ± 0.25m as placeholder bbox. These cannot be accurately bounded.

Run: python scripts/lift_semantics_3d.py (uses default paths from CLAUDE.md).
Verify: desk volume ~0.3–1.0 m³, monitor ~0.05–0.2 m³, window volume flagged as structural.

Then run: python scripts/build_scene_graph.py to update outputs/scene_graph.json.
```

### Session 8 — Scene Graph + Claude API
```
Read CLAUDE.md. Session 8:

Part A — scripts/build_scene_graph.py:
1. Load objects_3d.json (with confidence + provenance from Session 6).
2. Nodes: {id, label, position_3d, bbox_min, bbox_max, volume_m3,
   reconstruction_confidence, provenance, frames_seen}.
3. Edges (compute automatically):
   - on_top_of: A centroid Y > B bbox_max Y AND A XZ within B XZ ±0.15m
   - next_to: distance < 0.7m (not on_top_of)
   - near_wall: centroid within 0.25m of scene bbox face
   - between: A XZ lies between B and C XZ centroids ±0.3m
4. Room summary node: dimensions, counts, coverage_pct.
5. Save outputs/scene_graph.json.

Part B — scripts/query_scene.py:
1. Load ANTHROPIC_API_KEY from env. Load scene_graph.json.
2. System prompt (embed verbatim):
   "You are a robot spatial reasoning assistant for a humanoid robot.
   You have access to the 3D reconstruction of an indoor room.
   Scene graph: {scene_graph_json}
   Always include: 3D coords as (X.Xm, Y.Ym, Z.Zm), confidence score
   explained, waypoint confidence for path questions.
   Be concise — you are feeding a robot planner."
3. Interactive query loop + query_scene(question)->str function.
4. EXAMPLE_QUERIES list of 4 presets for Gradio buttons.
Use claude-sonnet-4-5, streaming, max_tokens=400.
```

### Session 9 — Gradio App + Deployment
```
Read CLAUDE.md. Session 9: write app/app.py.

THEME: gr.themes.Base(), custom CSS: bg #0f0f1a, cards #1a1a2e,
accent #7F77DD. Confidence badges: green>0.7, amber 0.3–0.7, red<0.3.

4 TABS:
Tab 1 "🔍 Scene Explorer":
  gr.HTML iframe: gsplat.js viewer loading scene.splat from SPLAT_URL env var
  gr.Image: navigability_map.png
  gr.Markdown: room summary from scene_graph.json

Tab 2 "📊 Object Map":
  gr.Dataframe: Object|Position|Confidence|Provenance|Size(m³) with HTML badges
  gr.Plot: 2D bird's-eye scatter coloured by confidence

Tab 3 "🤖 Robot Query":
  gr.Button row for EXAMPLE_QUERIES (fills textbox on click)
  gr.Textbox input + gr.Button "Ask Robot"
  Streaming Claude API response
  Collapsible raw scene graph JSON

Tab 4 "⚙️ Pipeline":
  gr.Image: pipeline_diagram.svg
  gr.Markdown: stage explanations

Load all JSONs at startup. Handle missing files gracefully.
Also write app/requirements.txt and HF Space README.
ANTHROPIC_API_KEY loaded from env (set as HF Space secret).
```

### Session 10 — README + Polish
```
Read CLAUDE.md and full codebase. Session 10: final polish.

1. Generate app/assets/pipeline_diagram.svg: horizontal 7-stage flow.
   Colors: purple #7F77DD=reconstruction, teal=semantic,
   amber=confidence, coral=output. Stage name + tool name per box.

2. Write complete README.md:
   - Hero: title + tagline + badges (HF Space, Python 3.11, CVPR 2025)
   - Demo video embed
   - Problem statement (3 sentences, robot framing)
   - Pipeline diagram
   - Quick start: clone + pip install + one command to run demo
   - navigability_map.png as hero figure with caption
   - Design Choices (most important — this wins the challenge):
     * VGGT over COLMAP
     * gsplat over FlashGS (NVIDIA-only) / nerfstudio
     * Open vocab over fixed-class
     * Confidence map as THE NOVEL CONTRIBUTION
       (cite robotics survey: <50% success rate for multi-stage tasks)
     * Claude API mirrors KinetIQ VLM architecture
   - Limitations (honest)
   - Future Work (multi-visit change detection, back-projection)

3. Final check: all imports correct, all paths match CLAUDE.md.
```

---

## Environment Variables

On Mac, create `.env` in project root (never commit):
```bash
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```
Load: `export $(cat .env | xargs)`

On bluestreak: `export ANTHROPIC_API_KEY=sk-ant-...`

---

## Key URLs

- GitHub: https://github.com/JesonRamesh/3D-Spatial-Reconstruction
- UCL GPU booking: https://mydesk.cs.ucl.ac.uk
- UCL SSH gateway: knuckles.cs.ucl.ac.uk
- Timeshare: cream.cs.ucl.ac.uk, vanilla.cs.ucl.ac.uk
- VGGT: https://github.com/facebookresearch/vggt
- gsplat: https://docs.gsplat.studio
- Grounded SAM2: https://github.com/IDEA-Research/Grounded-SAM-2
- HF Space: https://huggingface.co/spaces/JesonRamesh/roboscene-plus
- HF Dataset: https://huggingface.co/datasets/JesonRamesh/roboscene-data

---

## Submission Checklist

- [ ] GitHub repo is public
- [ ] README has demo video, pipeline diagram, design choices
- [ ] HF Space live and accessible without login
- [ ] ANTHROPIC_API_KEY set as HF Space secret (not in code)
- [ ] All 4 example queries work in deployed app
- [ ] navigability_map.png embedded in README
- [ ] Design choices names confidence map as novel contribution
- [ ] Limitations section present and honest
- [ ] Git history has 10+ meaningful commits
- [ ] Repo URL ready to paste into Humanoid application form

## Session 6 — COMPLETE ✅

### Script: scripts/compute_confidence.py
**Novel contribution:** Voxel-level reconstruction confidence = 0.6 × point_density + 0.4 × camera_coverage

**Key design decisions:**
- Point cloud source: `outputs/mast3r_out/room_video.ply` (7.33M points)
- Pose source: `data/vggt_out/camera_poses.json` — auto-detected `.json` format
  - `_load_poses_from_json()` reads `cam_to_world_4x4` matrices directly
  - Falls back to TUM `.txt` format if given a text file
- Voxel grid: 90×86×118 at 5cm resolution (913,320 voxels)
- Camera coverage: camera-centric loop O(C×r³), subsampled to 200 cameras
- Pure-numpy PLY reader (no open3d — Python 3.13 incompatible)
- Handles `objects_3d.json` as dict keyed by label (not list)
- bbox from `bbox_min`/`bbox_max` keys (not flat `bbox_3d`)

**Output files:**
```
outputs/confidence_map.npy          ← 3D float32 array (90, 86, 118)
outputs/confidence_metadata.json    ← origin, shape, zone percentages
outputs/navigability_map.png        ← HERO FIGURE: bird's-eye RAG map
outputs/scene_confidence_tagged.ply ← 4.35M Gaussians + confidence_tag (0/1/2)
outputs/objects_3d.json             ← updated: reconstruction_confidence + provenance
```

**Confidence results:**
- High (>0.7): 0.4% — tightly triangulated regions
- Medium (0.3–0.7): 34.1% — all main objects fall here (monitor 0.53, fan 0.49 …)
- Low (<0.3): 65.6% — walls/corners/dead zones

**Object provenance:**
- Sparse: monitor, fan, lamp, laptop, chair, desk, bed
- Inferred: window, shelf, door (near room boundaries)

**Run command:**
```bash
python scripts/compute_confidence.py \
  --splat_ply outputs/splat_mast3r_v2/scene.ply \
  --point_cloud outputs/mast3r_out/room_video.ply \
  --poses_file data/vggt_out/camera_poses.json \
  --objects_file outputs/objects_3d.json \
  --output_dir outputs/ \
  --voxel_size 0.05
```

---

## Session 5 — COMPLETE ✅

### Script: scripts/lift_semantics_3d.py
**Pipeline:** Semantic JSON masks + VGGT depth maps + camera poses → 3D object bounding boxes

**Output:** `outputs/objects_3d.json` — 10 objects with centroid_3d, bbox_min, bbox_max, volume_m3, frames_seen
**Hero figure:** `outputs/object_positions_2d.png` — top-down matplotlib scatter

---

## Session 4 — COMPLETE ✅

### Script: scripts/run_semantic.py
**Pipeline:** GroundingDINO (SwinT_OGC) detection → SAM2 (sam2.1_hiera_large) masks → pycocotools RLE JSON + debug PNGs

**Key design decisions:**
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set before any torch import (MPS safety)
- Single-caption multi-label query: `"bed . desk . chair . …"` for efficiency
- Weights auto-downloaded via `huggingface_hub`; local `weights/` takes priority
  - GDino: `ShilongLiu/GroundingDINO` → `groundingdino_swint_ogc.pth`
  - SAM2: `facebook/sam2.1-hiera-large` → `sam2.1_hiera_large.pt`
- Phrase→label matching: exact → substring → token fallback
- Best-confidence detection per label per frame (no duplicates)
- SAM2 box-mask fallback if SAM2 inference fails on a frame
- `--skip_existing` flag for resumable runs on bluestreak

**Output format:**
```json
{
  "chair": {
    "bbox": [142.3, 88.1, 410.7, 512.0],
    "confidence": 0.7231,
    "mask_rle": {"size": [720, 1280], "counts": "..."}
  }
}
```

**Job script:** `ucl_gpu/run_semantic_job.sh`
- 6 h wall-time, 1 GPU (RTX 4070 Ti SUPER), 32 GB RAM
- Venv: `/scratch0/jrameshs/roboscene_env`
- All params overridable: `bash ucl_gpu/run_semantic_job.sh` (uses defaults)
- Env vars: `FRAMES_DIR`, `OUTPUT_DIR`, `LABELS`, `CONFIDENCE`, `BATCH_SIZE`

**Run on bluestreak:**
```bash
cd /scratch0/jrameshs/roboscene-plus
git pull
mkdir -p logs outputs/semantic outputs/semantic/debug

# Install deps (first time only)
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/sam2.git
pip install pycocotools huggingface_hub tqdm colorama

# Run (foreground — watch progress bar)
python scripts/run_semantic.py \
  --frames_dir data/mast3r_out/images \
  --output_dir outputs/semantic \
  --device cuda

# OR background (SSH-disconnect safe)
nohup python scripts/run_semantic.py \
  --frames_dir data/mast3r_out/images \
  --output_dir outputs/semantic \
  --device cuda \
  > logs/semantic.log 2>&1 &
tail -f logs/semantic.log
```

**Download outputs to Mac:**
```bash
scp -r -J jrameshs@knuckles.cs.ucl.ac.uk \
  jrameshs@bluestreak.cs.ucl.ac.uk:/scratch0/jrameshs/roboscene-plus/outputs/semantic/ \
  ~/3D-Spatial-Reconstruction/outputs/
```

---

## Session 3 — COMPLETE ✅

### Final Pipeline
Video (room_video.MOV, 2.5min, 1080p 60fps, 1x lens)
→ MASt3R-SLAM (CVPR 2025, 54 keyframes, globally consistent)
→ COLMAP conversion (317 frames, sparse/0/)
→ nerfstudio splatfacto (60000 steps)
→ scene.ply (outputs/splat_mast3r_v2/scene.ply)

### Quality Verification
- Rendered frame matches ground truth: bed, headboard, fan, 
  chair, laptop all correctly reconstructed ✅
- Bird's eye point cloud shows correct room layout ✅
- 60k steps marginally better than 30k — use v2 as final

### Key Output Files (on Mac)
outputs/splat_mast3r_v2/scene.ply  ← FINAL SPLAT
outputs/mast3r_out/room_video.ply  ← point cloud
data/mast3r_out/sparse/0/          ← COLMAP format
data/mast3r_out/images/            ← 317 keyframes

### Known Limitations (document in README)
- Wall ghosting from limited keyframes (54)
- Not photorealistic from novel views outside training set
- SuperSplat looks distorted from outside — navigate inside with W key

### Bluestreak Paths
/scratch0/jrameshs/roboscene-plus/outputs/splat_mast3r_v2/splat.ply
/scratch0/jrameshs/roboscene-plus/data/mast3r_out/

### MASt3R-SLAM Install Location
/scratch0/jrameshs/MASt3R-SLAM/
Checkpoints: /scratch0/jrameshs/MASt3R-SLAM/checkpoints/