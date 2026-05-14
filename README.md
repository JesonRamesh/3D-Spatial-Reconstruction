# RoboScene+ 🤖🏠

**Video → 3D Gaussian Splatting → Semantic Scene Graph → Confidence-Aware Robot Spatial Memory**

Transform a simple phone video of an indoor room into an interactive 3D scene with AI-powered semantic understanding and confidence-aware spatial reasoning.

## Pipeline Overview

| Session | Stage | Script | Status |
|---------|-------|--------|--------|
| 1 | Frame Extraction | `scripts/extract_frames.py` | ✅ DONE |
| 2 | VGGT Reconstruction | `scripts/run_vggt.py` | ⬜ TODO |
| 3 | Gaussian Splatting | `scripts/train_splat.py` | ⬜ TODO |
| 4 | Semantic Segmentation | `scripts/run_semantic.py` | ⬜ TODO |
| 5 | 3D Semantic Lifting | `scripts/lift_semantics_3d.py` | ⬜ TODO |
| 6 | Confidence Map | `scripts/compute_confidence.py` | ⬜ TODO |
| 7 | Dead Zone Completion | `scripts/complete_dead_zones.py` | ⬜ TODO |
| 8 | Scene Graph | `scripts/build_scene_graph.py` | ⬜ TODO |
| 9 | Query Interface | `scripts/query_scene.py` | ⬜ TODO |
| 10 | Gradio App | `app/app.py` | ⬜ TODO |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract frames from video (Session 1)
python scripts/01_extract_frames.py --video_path room.MOV --fps 1.0
```

## Project Structure

```
roboscene-plus/
├── CLAUDE.md                  # AI assistant context
├── README.md                  # This file
├── config.yaml                # All paths and hyperparameters
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/                   # Original video
│   ├── frames/                # Extracted frames
│   └── vggt_out/              # VGGT outputs
├── scripts/                   # One script per pipeline stage
│   └── extract_frames.py      # ✅
├── outputs/                   # Final outputs
├── models/                    # Model weights
├── logs/                      # Execution logs
├── ucl_gpu/                   # UCL GPU job scripts
├── app/                       # Gradio demo
└── notebooks/                 # Debug notebooks
```

## Requirements

- Python 3.11+
- ffmpeg (for frame extraction)
- OpenCV (for quality analysis)