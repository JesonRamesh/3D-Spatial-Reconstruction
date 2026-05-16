"""
run_semantic.py – Grounded SAM2 semantic segmentation on keyframe images.

Pipeline
--------
1. Load GroundingDINO (SwinT_OGC) for open-vocabulary bounding-box detection.
2. Load SAM2 (sam2.1_hiera_large) for precise pixel-level masks.
3. For every frame: detect boxes per label → refine with SAM2 → save JSON + debug PNG.

Output layout
-------------
outputs/semantic/
    frame_0000.json          # {label: {bbox, confidence, mask_rle}}
    frame_0001.json
    ...
    debug/
        frame_0000_debug.png
        ...

Usage
-----
python scripts/run_semantic.py \
    --frames_dir  data/mast3r_out/images \
    --output_dir  outputs/semantic \
    --labels      "bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor" \
    --device      auto \
    --batch_size  10 \
    --confidence  0.3
"""

# ---------------------------------------------------------------------------
# Environment tweaks BEFORE any torch import so MPS fallback is honoured
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional coloured console output
# ---------------------------------------------------------------------------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _C = True
except ImportError:
    class _Dummy:
        def __getattr__(self, _): return ""
    Fore = Style = _Dummy()
    _C = False


def info(msg: str):  print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL}  {msg}")
def ok(msg: str):    print(f"{Fore.GREEN}[OK]{Style.RESET_ALL}    {msg}")
def warn(msg: str):  print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL}  {msg}")
def err(msg: str):   print(f"{Fore.RED}[ERR]{Style.RESET_ALL}   {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Colour palette for debug overlays (one per label, cycling)
# ---------------------------------------------------------------------------
PALETTE = [
    (255,  82,  82),   # red
    ( 82, 182, 255),   # blue
    ( 82, 255, 140),   # green
    (255, 210,  82),   # yellow
    (210,  82, 255),   # purple
    (255, 140,  82),   # orange
    ( 82, 255, 220),   # cyan
    (255,  82, 190),   # pink
    (180, 255,  82),   # lime
    (255, 255, 140),   # cream
]


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device(preference: str) -> str:
    import torch
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_GDINO_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
_GDINO_WEIGHTS_FILENAME = "groundingdino_swint_ogc.pth"


def download_gdino_weights(weights_dir: str) -> str:
    """
    Download GroundingDINO SwinT_OGC weights from the official GitHub release
    to <weights_dir>/groundingdino_swint_ogc.pth.
    Skips download if the file already exists.

    Returns the absolute path to the weights file.
    """
    dest_dir  = Path(weights_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _GDINO_WEIGHTS_FILENAME

    if dest_path.exists():
        info(f"GroundingDINO weights already cached: {dest_path}")
        return str(dest_path)

    info(f"Downloading GroundingDINO weights → {dest_path}")
    info(f"  URL: {_GDINO_WEIGHTS_URL}")

    # Progress callback — prints MB downloaded
    _last: Dict = {"mb": 0}
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        mb = downloaded / 1024 ** 2
        total_mb = total_size / 1024 ** 2 if total_size > 0 else "?"
        if mb - _last["mb"] >= 50 or block_num == 0:
            print(f"  ... {mb:.0f} / {total_mb} MB", flush=True)
            _last["mb"] = mb

    try:
        urllib.request.urlretrieve(_GDINO_WEIGHTS_URL, str(dest_path), _progress)
    except Exception as exc:
        # Remove partial file on failure
        if dest_path.exists():
            dest_path.unlink()
        err(f"Download failed: {exc}\n"
            f"Download manually from:\n  {_GDINO_WEIGHTS_URL}\n"
            f"and save to  {dest_path}")
        raise SystemExit(1) from exc

    ok(f"GroundingDINO weights downloaded ({dest_path.stat().st_size / 1024**2:.0f} MB)")
    return str(dest_path)


def load_grounding_dino(device: str, weights_dir: str):
    """
    Load GroundingDINO (SwinT_OGC) using the groundingdino package.
    Config is resolved from inside the installed package.
    Weights are downloaded via download_gdino_weights() if not already present.
    """
    try:
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        err("groundingdino not found.  Install with:\n"
            "  pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        raise SystemExit(1) from exc

    # Config — shipped inside the groundingdino package
    import groundingdino
    pkg_dir     = Path(groundingdino.__file__).parent
    config_path = pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py"
    if not config_path.exists():
        candidates = list(pkg_dir.rglob("GroundingDINO_SwinT_OGC.py"))
        if candidates:
            config_path = candidates[0]
        else:
            err(f"Could not locate GroundingDINO_SwinT_OGC.py inside {pkg_dir}")
            raise SystemExit(1)

    # Weights — download to weights_dir if needed
    weights_path = download_gdino_weights(weights_dir)

    info(f"Loading GroundingDINO  (config: {config_path.name})")
    info(f"  weights: {weights_path}")
    model = load_model(str(config_path), weights_path, device=device)
    model.eval()
    return model


def load_sam2(device: str):
    """
    Load SAM2 (sam2.1_hiera_large) via HuggingFace hub using
    SAM2ImagePredictor.from_pretrained() — no local config files needed.
    HF_HOME is set to /scratch0/jrameshs/hf_cache on bluestreak so the
    download lands on scratch (1TB) rather than the 10GB home quota.
    """
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        err("sam2 not found.  Install with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git")
        raise SystemExit(1) from exc

    # Redirect HF cache to scratch so we don't fill the 10GB home quota.
    # Use existing env var if already set (e.g. by the job script), otherwise
    # default to the bluestreak scratch path.
    os.environ.setdefault("HF_HOME", "/scratch0/jrameshs/hf_cache")
    info(f"HF_HOME={os.environ['HF_HOME']}")

    info("Loading SAM2 sam2.1-hiera-large via HuggingFace hub …")
    try:
        predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-large",
            device=device,
        )
    except Exception as exc:
        err(f"SAM2 from_pretrained() failed: {exc}\n"
            "Ensure sam2 is installed from source:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git")
        raise SystemExit(1) from exc

    ok("SAM2 loaded.")
    return predictor


# ---------------------------------------------------------------------------
# GroundingDINO inference helpers
# ---------------------------------------------------------------------------
def gdino_predict(
    model,
    image_pil: Image.Image,
    labels: List[str],
    confidence_threshold: float,
    device: str,
) -> Dict[str, List[Dict]]:
    """
    Run GroundingDINO for all labels on one PIL image.

    Returns
    -------
    dict  label -> list of {"bbox": [x1,y1,x2,y2], "confidence": float}
          Bounding boxes are in absolute pixel coordinates.
    """
    import torch
    from groundingdino.util.inference import predict as gdino_predict_fn
    from groundingdino.util import box_ops
    import torchvision.transforms as T

    # GroundingDINO expects a normalised float tensor
    transform = T.Compose([
        T.Resize((800, 1333), max_size=1333),   # closest to training res
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image_pil.convert("RGB")).to(device)

    W, H = image_pil.size
    results: Dict[str, List[Dict]] = {lbl: [] for lbl in labels}

    # Build caption: "label1 . label2 . label3" (no trailing dot, space-dot-space
    # separator is the format GroundingDINO was trained on).
    caption = " . ".join(labels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            boxes, logits, phrases = gdino_predict_fn(
                model=model,
                image=image_tensor,
                caption=caption,
                box_threshold=0.25,
                text_threshold=0.20,
                device=device,
            )

    if boxes is None or len(boxes) == 0:
        return results

    # boxes are cx,cy,w,h in [0,1] — convert to absolute xyxy
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
        [W, H, W, H], dtype=torch.float32, device=boxes.device
    )
    boxes_xyxy = boxes_xyxy.cpu().numpy()
    confidences = logits.cpu().numpy()
    phrases_list = phrases  # list of str

    # Assign each detection to the closest matching label
    for box, conf, phrase in zip(boxes_xyxy, confidences, phrases_list):
        phrase_clean = phrase.strip().lower()
        # Find best matching label (simple substring / exact match)
        matched_label = _match_phrase_to_label(phrase_clean, labels)
        if matched_label is None:
            continue
        x1, y1, x2, y2 = box.tolist()
        results[matched_label].append({
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": float(conf),
        })

    return results


def _match_phrase_to_label(phrase: str, labels: List[str]) -> Optional[str]:
    """
    Match a detected phrase to a query label.
    Preference order: exact → label-in-phrase → phrase-in-label → edit-distance.
    """
    phrase = phrase.lower().strip()
    labels_lower = [l.lower() for l in labels]

    # Exact match
    if phrase in labels_lower:
        return labels[labels_lower.index(phrase)]

    # Label word contained in detected phrase
    for orig, lbl in zip(labels, labels_lower):
        if lbl in phrase:
            return orig

    # Detected phrase contained in label
    for orig, lbl in zip(labels, labels_lower):
        if phrase in lbl:
            return orig

    # Fallback: first token of phrase matches a label token
    tokens = phrase.split()
    for token in tokens:
        for orig, lbl in zip(labels, labels_lower):
            if token == lbl or token in lbl.split():
                return orig

    return None


# ---------------------------------------------------------------------------
# SAM2 inference helper
# ---------------------------------------------------------------------------
def sam2_predict(
    predictor,
    image_np: np.ndarray,
    boxes_xyxy: np.ndarray,
) -> np.ndarray:
    """
    Run SAM2 for a set of bounding-box prompts on one image.

    Parameters
    ----------
    predictor   : SAM2ImagePredictor
    image_np    : uint8 HxWx3 numpy array
    boxes_xyxy  : Nx4 float array in absolute pixel coords

    Returns
    -------
    masks : boolean NxHxW array (one mask per box)
    """
    import torch

    predictor.set_image(image_np)

    if len(boxes_xyxy) == 0:
        return np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool)

    input_boxes = torch.from_numpy(boxes_xyxy).float()
    # SAM2 expects (N,4) on the same device as the model
    device = next(iter(predictor.model.parameters())).device
    input_boxes = input_boxes.to(device)

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    # masks: (N, 1, H, W) or (N, H, W) depending on SAM2 version
    if masks.ndim == 4:
        masks = masks[:, 0]          # → (N, H, W)
    return masks.cpu().numpy().astype(bool)


# ---------------------------------------------------------------------------
# RLE encoding via pycocotools
# ---------------------------------------------------------------------------
def mask_to_rle(mask: np.ndarray) -> Dict:
    """
    Encode a boolean HxW mask to COCO-style RLE.
    Returns a dict {"size": [H,W], "counts": <bytes-decoded str>}.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        err("pycocotools not found.  Install with:  pip install pycocotools")
        raise SystemExit(1) from exc

    # pycocotools expects Fortran-order uint8
    mask_uint8 = np.asfortranarray(mask.astype(np.uint8))
    rle = coco_mask.encode(mask_uint8)
    # 'counts' is bytes – decode for JSON serialisability
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------
def draw_debug_overlay(
    image_pil: Image.Image,
    detections: Dict[str, Dict],
    label_to_colour: Dict[str, Tuple[int, int, int]],
) -> Image.Image:
    """
    Draw coloured mask + bounding box + label text on a copy of image_pil.
    detections: {label: {"bbox":..., "confidence":..., "mask_np": np.ndarray}}
    """
    from pycocotools import mask as coco_mask

    composite = image_pil.convert("RGBA").copy()
    overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a small font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()

    for label, det in detections.items():
        colour = label_to_colour.get(label, (200, 200, 200))
        r, g, b = colour

        # Decode mask for drawing
        mask_np = det.get("mask_np")
        if mask_np is not None and mask_np.any():
            # Semi-transparent filled mask
            mask_img = Image.fromarray((mask_np * 100).astype(np.uint8), mode="L")
            colour_layer = Image.new("RGBA", composite.size, (r, g, b, 120))
            overlay = Image.composite(colour_layer, overlay, mask_img)
            draw = ImageDraw.Draw(overlay)  # re-bind after composite

        # Bounding box
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 230), width=2)

        # Label text background
        label_text = f"{label} {det['confidence']:.2f}"
        bbox_text = draw.textbbox((x1, y1 - 16), label_text, font=font)
        draw.rectangle(bbox_text, fill=(r, g, b, 200))
        draw.text((x1, y1 - 16), label_text, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(composite, overlay)
    return result.convert("RGB")


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------
def process_frame(
    frame_path: Path,
    labels: List[str],
    gdino_model,
    sam2_predictor,
    confidence: float,
    device: str,
    out_json_path: Path,
    out_debug_path: Optional[Path],
    label_to_colour: Dict[str, Tuple[int, int, int]],
) -> Dict[str, bool]:
    """
    Process one frame: detect → mask → save JSON + debug PNG.

    Returns dict {label: True/False} indicating which labels were found.
    """
    image_pil = Image.open(frame_path).convert("RGB")
    image_np  = np.array(image_pil)

    # --- Detection ---
    raw_detections = gdino_predict(
        gdino_model, image_pil, labels, confidence, device
    )

    # Pick the highest-confidence detection per label (avoid duplicate entries)
    best_per_label: Dict[str, Dict] = {}
    for label, dets in raw_detections.items():
        if not dets:
            continue
        best = max(dets, key=lambda d: d["confidence"])
        best_per_label[label] = best

    if not best_per_label:
        out_json_path.write_text(json.dumps({}))
        if out_debug_path:
            image_pil.save(str(out_debug_path))
        return {lbl: False for lbl in labels}

    # --- SAM2 masks ---
    ordered_labels  = list(best_per_label.keys())
    boxes_array     = np.array([best_per_label[l]["bbox"] for l in ordered_labels],
                               dtype=np.float32)

    try:
        masks = sam2_predict(sam2_predictor, image_np, boxes_array)
    except Exception as exc:
        warn(f"SAM2 failed on {frame_path.name}: {exc} — falling back to box masks")
        H, W = image_np.shape[:2]
        masks = np.zeros((len(ordered_labels), H, W), dtype=bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes_array):
            masks[i,
                  max(0, int(y1)):min(H, int(y2)),
                  max(0, int(x1)):min(W, int(x2))] = True

    # --- Build output JSON + debug dict ---
    frame_json:  Dict[str, Dict] = {}
    debug_dets:  Dict[str, Dict] = {}
    found:       Dict[str, bool] = {lbl: False for lbl in labels}

    for i, label in enumerate(ordered_labels):
        mask_np = masks[i] if i < len(masks) else None
        det     = best_per_label[label]
        rle     = mask_to_rle(mask_np) if mask_np is not None else None

        frame_json[label] = {
            "bbox":       det["bbox"],
            "confidence": round(det["confidence"], 4),
            "mask_rle":   rle,
        }
        debug_dets[label] = {
            "bbox":       det["bbox"],
            "confidence": det["confidence"],
            "mask_np":    mask_np,
        }
        found[label] = True

    # --- Save JSON ---
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(frame_json, f, indent=2)

    # --- Save debug PNG ---
    if out_debug_path is not None:
        out_debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_img = draw_debug_overlay(image_pil, debug_dets, label_to_colour)
        debug_img.save(str(out_debug_path))

    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded SAM2 semantic segmentation on keyframe images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames_dir", type=Path,
        default=Path("data/mast3r_out/images"),
        help="Directory containing input JPG/PNG keyframes.",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("outputs/semantic"),
        help="Root output directory (JSON masks + debug PNGs).",
    )
    parser.add_argument(
        "--labels", type=str,
        default="bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor",
        help="Comma-separated list of object labels to detect.",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (auto = pick best available).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Number of frames to process before logging a progress checkpoint.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Detection confidence threshold for GroundingDINO.",
    )
    parser.add_argument(
        "--no_debug", action="store_true",
        help="Skip saving debug PNG overlays (faster).",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip frames whose JSON output already exists.",
    )
    parser.add_argument(
        "--weights_dir", type=str,
        default="/scratch0/jrameshs/gdino_weights",
        help="Directory where GroundingDINO weights are cached / downloaded to.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Resolve device
    # -----------------------------------------------------------------------
    import torch  # noqa: imported after MPS env-var is set
    device = pick_device(args.device)
    ok(f"Using device: {device}")
    if device == "mps":
        warn("MPS device selected – PYTORCH_ENABLE_MPS_FALLBACK=1 is set.")

    # -----------------------------------------------------------------------
    # Parse labels
    # -----------------------------------------------------------------------
    labels: List[str] = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not labels:
        err("--labels is empty.  Provide at least one label.")
        raise SystemExit(1)
    info(f"Labels ({len(labels)}): {', '.join(labels)}")

    label_to_colour = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    # -----------------------------------------------------------------------
    # Collect frames
    # -----------------------------------------------------------------------
    frames_dir = args.frames_dir
    if not frames_dir.exists():
        err(f"frames_dir does not exist: {frames_dir}")
        raise SystemExit(1)

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    frame_paths: List[Path] = sorted(
        p for p in frames_dir.iterdir() if p.suffix in exts
    )
    if not frame_paths:
        err(f"No image files found in {frames_dir}")
        raise SystemExit(1)
    info(f"Found {len(frame_paths)} frames in {frames_dir}")

    # -----------------------------------------------------------------------
    # Prepare output directories
    # -----------------------------------------------------------------------
    out_dir   = args.output_dir
    debug_dir = out_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    info("Loading models…")
    t0 = time.time()
    gdino_model    = load_grounding_dino(device, args.weights_dir)
    sam2_predictor = load_sam2(device)
    ok(f"Models loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # Process frames
    # -----------------------------------------------------------------------
    # label_found_count[label] = number of frames where it was detected
    label_found_count: Dict[str, int] = {lbl: 0 for lbl in labels}
    skipped = 0

    t_start = time.time()

    pbar = tqdm(frame_paths, desc="Segmenting", unit="frame",
                dynamic_ncols=True, colour="cyan")

    for idx, frame_path in enumerate(pbar):
        # Derive output filenames – strip extension, use stem
        stem = frame_path.stem          # e.g. "frame_0000" or "00042"
        # Normalise to frame_XXXX format using sorted index if stem is numeric
        if stem.isdigit():
            out_stem = f"frame_{int(stem):04d}"
        else:
            out_stem = stem

        out_json  = out_dir / f"{out_stem}.json"
        out_debug = (debug_dir / f"{out_stem}_debug.png") if not args.no_debug else None

        if args.skip_existing and out_json.exists():
            skipped += 1
            pbar.set_postfix(status="skipped")
            continue

        try:
            found = process_frame(
                frame_path     = frame_path,
                labels         = labels,
                gdino_model    = gdino_model,
                sam2_predictor = sam2_predictor,
                confidence     = args.confidence,
                device         = device,
                out_json_path  = out_json,
                out_debug_path = out_debug,
                label_to_colour= label_to_colour,
            )
        except Exception as exc:
            warn(f"Frame {frame_path.name} failed: {exc}")
            found = {lbl: False for lbl in labels}

        for lbl, was_found in found.items():
            if was_found:
                label_found_count[lbl] += 1

        # Batch checkpoint log
        if (idx + 1) % args.batch_size == 0:
            elapsed = time.time() - t_start
            fps     = (idx + 1 - skipped) / max(elapsed, 1e-6)
            pbar.set_postfix(fps=f"{fps:.1f}", elapsed=f"{elapsed:.0f}s")

    pbar.close()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_processed = len(frame_paths) - skipped
    elapsed_total   = time.time() - t_start

    print()
    print("=" * 60)
    print(f"  Semantic segmentation complete")
    print(f"  Frames processed : {total_processed}  (skipped: {skipped})")
    print(f"  Total time       : {elapsed_total:.1f}s  "
          f"({total_processed / max(elapsed_total,1e-6):.1f} fps)")
    print(f"  Output JSON dir  : {out_dir}")
    if not args.no_debug:
        print(f"  Debug PNG dir    : {debug_dir}")
    print()
    print(f"  {'Label':<16}  {'Frames found':>13}  {'Coverage':>9}")
    print(f"  {'-'*16}  {'-'*13}  {'-'*9}")

    n_total = len(frame_paths)
    for lbl in labels:
        n     = label_found_count[lbl]
        pct   = 100.0 * n / max(n_total, 1)
        bar   = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {lbl:<16}  {n:>6}/{n_total:<6}  {pct:>6.1f}%  {bar}")

    print("=" * 60)
    ok("Done.")


if __name__ == "__main__":
    main()