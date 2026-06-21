"""
Export a fly-through frame sequence from a trained nerfstudio splatfacto checkpoint.

Renders a smooth camera path interpolated through the training camera poses
(no manual path authoring needed) and saves the result as a sequentially
numbered PNG sequence for use in an external evaluation tool.

Requires the FULL nerfstudio run directory (config.yml + nerfstudio_models/*.ckpt +
dataparser_transforms.json) -- a standalone scene.ply export is NOT sufficient,
since it carries no camera/pose/transform metadata.

Confirmed locally available run:
  outputs/splat/vggt_out/splatfacto/2026-05-15_140827/config.yml
  (VGGT-posed, 15,000 steps -- NOT the documented "final" MASt3R 60k-step splat,
   whose source run directory only survives as an exported .ply locally).

Must be run on a CUDA machine (e.g. UCL bluestreak) -- gsplat's rasterizer
backend used by splatfacto has no Metal/MPS kernels, so this cannot render
on the Mac M4 Pro.

Requirements (install into the bluestreak roboscene_env venv):
    nerfstudio==1.1.5
    torch>=2.0.0          (already present in roboscene_env)
    torchvision>=0.15.0   (already present in roboscene_env)
    gsplat==1.3.0          (already present in roboscene_env)
    Pillow

Usage:
    python export_frames.py \\
        --config_path outputs/splat/vggt_out/splatfacto/2026-05-15_140827/config.yml \\
        --output_dir frames/ \\
        --num_frames 120
"""

import argparse
from pathlib import Path

import torch

# nerfstudio 1.1.5 checkpoints were saved before PyTorch's torch.load default
# flipped to weights_only=True; without this patch eval_setup() fails to
# unpickle the saved TrainerConfig. Same workaround already used in
# ucl_gpu/run_splat_v4.sh when extracting Gaussians from a checkpoint.
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

import numpy as np
from PIL import Image

from nerfstudio.cameras.camera_paths import get_interpolated_camera_path
from nerfstudio.utils.eval_utils import eval_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("outputs/splat/vggt_out/splatfacto/2026-05-15_140827/config.yml"),
        help="Path to the trained run's config.yml (full run dir required, not a .ply).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("frames/"),
        help="Directory to write numbered PNG frames into.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=120,
        help="Number of frames to render along the interpolated path (100-150 recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config_path.exists():
        raise FileNotFoundError(
            f"config.yml not found at {args.config_path}. "
            "A standalone scene.ply is not enough -- this script needs the full "
            "nerfstudio run directory (config.yml + nerfstudio_models/*.ckpt + "
            "dataparser_transforms.json)."
        )

    print(f"Loading pipeline from {args.config_path} ...")
    _, pipeline, _, _ = eval_setup(args.config_path)
    pipeline.eval()

    train_cameras = pipeline.datamanager.train_dataparser_outputs.cameras
    print(f"Loaded {train_cameras.size} training camera poses.")

    print(f"Interpolating a {args.num_frames}-frame fly-through path ...")
    render_cameras = get_interpolated_camera_path(
        cameras=train_cameras,
        steps=args.num_frames,
        order_poses=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = pipeline.device
    for i in range(render_cameras.size):
        camera = render_cameras[i : i + 1].to(device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        rgb = outputs["rgb"].detach().cpu().numpy()
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_uint8 = (rgb * 255.0).astype(np.uint8)

        frame_path = args.output_dir / f"{i + 1:06d}.png"
        Image.fromarray(rgb_uint8).save(frame_path)

        if (i + 1) % 10 == 0 or (i + 1) == render_cameras.size:
            print(f"  rendered {i + 1}/{render_cameras.size} -> {frame_path.name}")

    print(f"Done. {render_cameras.size} frames written to {args.output_dir}/")


if __name__ == "__main__":
    main()
