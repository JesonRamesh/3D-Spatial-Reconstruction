"""
run_semantic.py – Grounded SAM2 semantic segmentation on keyframe images.

Usage
-----
python scripts/run_semantic.py \
    --frames_dir  data/mast3r_out/images \
    --output_dir  outputs/semantic \
    --labels      "bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor" \
    --device      cuda \
    --weights_dir /scratch0/jrameshs/gdino_weights
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = [
    (255,  82,  82), (82, 182, 255), (82, 255, 140), (255, 210,  82),
    (210,  82, 255), (255, 140,  82), (82, 255, 220), (255,  82, 190),
    (180, 255,  82), (255, 255, 140),
]


# ---------------------------------------------------------------------------
# GDino weight download
# ---------------------------------------------------------------------------
GDINO_URL  = ("https://github.com/IDEA-Research/GroundingDINO/releases/download/"
              "v0.1.0-alpha/groundingdino_swint_ogc.pth")
GDINO_FILE = "groundingdino_swint_ogc.pth"


def ensure_gdino_weights(weights_dir):
    dest = Path(weights_dir) / GDINO_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[INFO]  GDino weights cached: {dest}")
        return str(dest)
    print(f"[INFO]  Downloading GDino weights -> {dest}")
    last = {"mb": 0}
    def _cb(n, bs, total):
        mb = n * bs / 1024 ** 2
        if mb - last["mb"] >= 50:
            print(f"  {mb:.0f} / {total / 1024**2:.0f} MB", flush=True)
            last["mb"] = mb
    urllib.request.urlretrieve(GDINO_URL, str(dest), _cb)
    print(f"[OK]    Downloaded {dest.stat().st_size / 1024**2:.0f} MB")
    return str(dest)


# ---------------------------------------------------------------------------
# RLE encoding
# ---------------------------------------------------------------------------
def mask_to_rle(mask):
    from pycocotools import mask as M
    rle = M.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------
def save_debug(image_pil, detections, colours, path):
    comp    = image_pil.convert("RGBA").copy()
    overlay = Image.new("RGBA", comp.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for label, d in detections.items():
        r, g, b = colours.get(label, (200, 200, 200))
        mask_np = d.get("mask_np")
        if mask_np is not None and mask_np.any():
            ml = Image.fromarray((mask_np * 100).astype(np.uint8), "L")
            cl = Image.new("RGBA", comp.size, (r, g, b, 120))
            overlay = Image.composite(cl, overlay, ml)
            draw    = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 230), width=2)
        txt = f"{label} {d['confidence']:.2f}"
        tb  = draw.textbbox((x1, y1 - 16), txt, font=font)
        draw.rectangle(tb, fill=(r, g, b, 200))
        draw.text((x1, y1 - 16), txt, fill=(255, 255, 255, 255), font=font)
    Image.alpha_composite(comp, overlay).convert("RGB").save(str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--frames_dir",    type=Path,  default=Path("data/mast3r_out/images"))
    p.add_argument("--output_dir",    type=Path,  default=Path("outputs/semantic"))
    p.add_argument("--labels",        type=str,   default="bed,desk,chair,laptop,shelf,door,window,fan,lamp,monitor")
    p.add_argument("--device",        type=str,   default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--batch_size",    type=int,   default=10)
    p.add_argument("--confidence",    type=float, default=0.25,  help="Unused — thresholds fixed at 0.25/0.20.")
    p.add_argument("--no_debug",      action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--weights_dir",   type=str,   default="/scratch0/jrameshs/gdino_weights")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args    = parse_args()
    labels  = [l.strip() for l in args.labels.split(",") if l.strip()]
    colours = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    # collect frames
    exts   = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    frames = sorted(p for p in args.frames_dir.iterdir() if p.suffix in exts)
    if not frames:
        raise SystemExit(f"No images found in {args.frames_dir}")
    print(f"[INFO]  {len(frames)} frames  |  labels: {', '.join(labels)}")

    # output dirs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug"
    if not args.no_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # ── load GroundingDINO — exact working pattern ───────────────────────────
    import groundingdino
    from groundingdino.util.inference import load_model, load_image, predict as gdino_predict

    config  = os.path.join(os.path.dirname(groundingdino.__file__),
                           "config", "GroundingDINO_SwinT_OGC.py")
    weights = ensure_gdino_weights(args.weights_dir)
    print(f"[INFO]  Loading GroundingDINO  config={config}")
    gdino = load_model(config, weights, device=args.device)
    gdino.eval()
    print("[OK]    GroundingDINO ready.")

    # ── load SAM2 ────────────────────────────────────────────────────────────
    os.environ.setdefault("HF_HOME", "/scratch0/jrameshs/hf_cache")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("[INFO]  Loading SAM2 sam2.1-hiera-large ...")
    sam2 = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large",
                                              device=args.device)
    print("[OK]    SAM2 ready.")

    # caption: "bed . desk . chair . ..."
    caption = " . ".join(labels)
    print(f"[INFO]  Caption: \"{caption}\"")

    # ── per-frame loop ───────────────────────────────────────────────────────
    found_count = {lbl: 0 for lbl in labels}
    t0 = time.time()

    for fp in tqdm(frames, desc="Segmenting", unit="frame", colour="cyan"):
        stem     = f"frame_{int(fp.stem):04d}" if fp.stem.isdigit() else fp.stem
        out_json = args.output_dir / f"{stem}.json"
        out_png  = debug_dir / f"{stem}_debug.png" if not args.no_debug else None

        if args.skip_existing and out_json.exists():
            continue

        # detection
        try:
            image_source, image_tensor = load_image(str(fp))
        except Exception as e:
            tqdm.write(f"[WARN]  load_image failed {fp.name}: {e}")
            out_json.write_text("{}")
            continue

        boxes, logits, phrases = gdino_predict(
            model          = gdino,
            image          = image_tensor,
            caption        = caption,
            box_threshold  = 0.25,
            text_threshold = 0.20,
            device         = args.device,
        )

        if boxes is None or len(boxes) == 0:
            out_json.write_text("{}")
            if out_png:
                Image.open(fp).save(str(out_png))
            continue

        # boxes to absolute xyxy
        import torch
        from groundingdino.util import box_ops
        H, W     = image_source.shape[:2]
        xyxy     = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor(
                       [W, H, W, H], dtype=torch.float32)
        boxes_np = xyxy.cpu().numpy()
        confs_np = logits.cpu().numpy()

        # match phrases to labels, keep best per label
        best = {}
        for i, (phrase, conf) in enumerate(zip(phrases, confs_np)):
            phrase = phrase.strip().lower()
            matched = None
            for lbl in labels:
                if lbl in phrase or phrase in lbl:
                    matched = lbl
                    break
            if matched is None:
                for lbl in labels:
                    if any(tok in lbl for tok in phrase.split()):
                        matched = lbl
                        break
            if matched and (matched not in best or conf > best[matched]["confidence"]):
                x1, y1, x2, y2 = boxes_np[i].tolist()
                best[matched] = {
                    "bbox":       [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": float(conf),
                }

        if not best:
            out_json.write_text("{}")
            if out_png:
                Image.open(fp).save(str(out_png))
            continue

        # SAM2 masks
        img_np    = np.array(Image.open(fp).convert("RGB"))
        sel_boxes = np.array([best[l]["bbox"] for l in best], dtype=np.float32)
        try:
            import torch
            sam2.set_image(img_np)
            with torch.no_grad():
                masks, _, _ = sam2.predict(
                    point_coords     = None,
                    point_labels     = None,
                    box              = torch.from_numpy(sel_boxes).to(args.device),
                    multimask_output = False,
                )
            if masks.ndim == 4:
                masks = masks[:, 0]
            masks_bool = masks.cpu().numpy().astype(bool)
        except Exception as e:
            tqdm.write(f"[WARN]  SAM2 failed {fp.name}: {e} -- box fallback")
            masks_bool = np.zeros((len(best), H, W), dtype=bool)
            for i, (x1, y1, x2, y2) in enumerate(sel_boxes):
                masks_bool[i, max(0, int(y1)):min(H, int(y2)),
                              max(0, int(x1)):min(W, int(x2))] = True

        # save JSON + debug PNG
        frame_data = {}
        debug_dets = {}
        image_pil  = Image.open(fp).convert("RGB")
        for i, lbl in enumerate(best):
            mask_np = masks_bool[i] if i < len(masks_bool) else None
            frame_data[lbl] = {
                "bbox":       best[lbl]["bbox"],
                "confidence": round(best[lbl]["confidence"], 4),
                "mask_rle":   mask_to_rle(mask_np) if mask_np is not None else None,
            }
            debug_dets[lbl] = {**best[lbl], "mask_np": mask_np}
            found_count[lbl] += 1

        out_json.write_text(json.dumps(frame_data, indent=2))
        if out_png:
            save_debug(image_pil, debug_dets, colours, out_png)

    # ── summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    n = len(frames)
    print(f"\n{'='*55}")
    print(f"  Done  {n} frames in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} fps)")
    print(f"  {'Label':<16} {'Found':>8} {'Coverage':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*10}")
    for lbl in labels:
        c   = found_count[lbl]
        pct = 100.0 * c / max(n, 1)
        bar = chr(9608) * int(pct // 5) + chr(9617) * (20 - int(pct // 5))
        print(f"  {lbl:<16} {c:>5}/{n:<3} {pct:>6.1f}%  {bar}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()