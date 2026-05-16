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
│   ├── raw/                        ← Original video file
│   ├── frames/                     ← 82 extracted frames ✅
│   └── vggt_out/                   ← VGGT outputs (COLMAP format + depth maps)
│
├── scripts/
│   ├── extract_frames.py           ← ✅ DONE
│   ├── run_vggt.py                 ← ✅ DONE (dry run verified on MPS)
│   ├── colmap_utils.py             ← ✅ DONE (custom COLMAP binary writer)
│   ├── train_splat.py              ← ⬜ TODO Session 3
│   ├── run_semantic.py             ← ⬜ TODO Session 4
│   ├── lift_semantics_3d.py        ← ⬜ TODO Session 5
│   ├── compute_confidence.py       ← ⬜ TODO Session 6
│   ├── complete_dead_zones.py      ← ⬜ TODO Session 7
│   ├── build_scene_graph.py        ← ⬜ TODO Session 8
│   └── query_scene.py              ← ⬜ TODO Session 8
│
├── outputs/
│   ├── splat/                      ← Trained Gaussian Splat (.ply, .splat)
│   ├── semantic/                   ← Per-frame semantic masks
│   ├── objects_3d.json
│   ├── confidence_map.npy
│   ├── navigability_map.png        ← HERO FIGURE for README
│   └── scene_graph.json
│
├── ucl_gpu/
│   ├── run_vggt_job.sh             ← ✅ DONE
│   ├── run_splat_job.sh            ← ⬜ TODO
│   └── run_semantic_job.sh         ← ⬜ TODO
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
| 2 | VGGT reconstruction | ✅ | 🔄 Script done + MPS dry run verified. **Run full on bluestreak now.** |
| 3 | Gaussian Splatting | ✅ | ⬜ Write train_splat.py → run on bluestreak |
| 4 | Grounded SAM2 | ✅ | ✅ DONE — scripts/run_semantic.py + ucl_gpu/run_semantic_job.sh |
| 5 | 3D semantic lifting | ❌ | ⬜ TODO |
| 6 | Confidence map ★ | ❌ | ⬜ TODO |
| 7 | Dead zone completion | ❌ | ⬜ TODO |
| 8 | Scene graph + Claude API | ❌ | ⬜ TODO |
| 9 | Gradio app + deploy | ❌ | ⬜ TODO |
| 10 | README + polish | ❌ | ⬜ TODO |

---

## Immediate Next Action — Run VGGT Full on bluestreak

82 frames are already at `/scratch0/jrameshs/roboscene-plus/data/frames/`

```bash
# On bluestreak (after bash + activate):
cd /scratch0/jrameshs/roboscene-plus
git pull
mkdir -p data/vggt_out/depths data/vggt_out/sparse logs
export PYTHONPATH=/scratch0/jrameshs/vggt:$PYTHONPATH

nohup python scripts/run_vggt.py \
  --frames_dir data/frames/ \
  --output_dir data/vggt_out/ \
  --batch_size 30 \
  > logs/vggt_full.log 2>&1 &

tail -f logs/vggt_full.log
```

Expected: 5–10 min. Verify when done:
```bash
ls data/vggt_out/depths/ | wc -l          # should be 82
ls data/vggt_out/sparse/                  # cameras.bin images.bin points3D.bin
python3 -c "
import json, numpy as np, glob
poses = json.load(open('data/vggt_out/camera_poses.json'))
print(f'{len(poses)} camera poses')
d = np.load(glob.glob('data/vggt_out/depths/*.npy')[0])
print(f'Depth shape: {d.shape}, range: {d.min():.2f}–{d.max():.2f}m')
"
```

Then immediately chain Session 3 (Gaussian Splatting) in the same booking.

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
| Confidence metric | Opacity density × camera coverage | Cheap, interpretable, no GT needed |
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