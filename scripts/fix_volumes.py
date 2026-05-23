#!/usr/bin/env python3
"""
fix_volumes.py — Recompute object bboxes using DBSCAN to exclude floater Gaussians.
Run after paint_semantic_gaussians.py has produced a corrected semantic PLY.

Usage:
    python scripts/fix_volumes.py
    python scripts/fix_volumes.py --semantic_ply outputs/splat_v3/scene_semantic_v3.ply
    python scripts/fix_volumes.py --apply   # overwrite objects_3d_v3.json in-place
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Try to import sklearn; give a helpful message if missing ─────────────────
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit(
        "[error] scikit-learn not installed.\n"
        "  pip install scikit-learn\n"
    )

# ── Try to import plyfile; give a helpful message if missing ─────────────────
try:
    from plyfile import PlyData
except ImportError:
    sys.exit(
        "[error] plyfile not installed.\n"
        "  pip install plyfile\n"
    )


# ---------------------------------------------------------------------------
# Class label mapping — must match paint_semantic_gaussians.py
# ---------------------------------------------------------------------------
LABEL_MAP = {
    0:  'background',
    1:  'bed',
    2:  'desk',
    3:  'chair',
    4:  'laptop',
    5:  'monitor',
    6:  'fan',
    7:  'lamp',
    8:  'shelf',
    9:  'door',
    10: 'window',
}

DBSCAN_PARAMS = {
    'bed':     {'eps': 0.15, 'min_samples': 200},
    'door':    {'eps': 0.15, 'min_samples': 100},
    'desk':    {'eps': 0.15, 'min_samples': 100},
    'chair':   {'eps': 0.12, 'min_samples': 50 },
    'laptop':  {'eps': 0.15, 'min_samples': 30 },
    'monitor': {'eps': 0.08, 'min_samples': 5  },
    'fan':     {'eps': 0.15, 'min_samples': 10 },
    'lamp':    {'eps': 0.15, 'min_samples': 10 },
    'shelf':   {'eps': 0.15, 'min_samples': 80 },
    'window':  {'eps': 0.15, 'min_samples': 10 },
}


def clean_bbox(positions: np.ndarray, eps: float = 0.3, min_samples: int = 50,
               max_points: int = 50_000):
    """Return bbox and centroid of the largest DBSCAN cluster, or None."""
    if len(positions) < min_samples:
        return None
    # Subsample for DBSCAN to avoid OOM on large classes
    if len(positions) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(positions), max_points, replace=False)
        sample = positions[idx]
    else:
        sample = positions
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(sample)
    labels = db.labels_
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        return None
    largest_label = unique[np.argmax(counts)]
    core = sample[labels == largest_label]
    dims = core.max(axis=0) - core.min(axis=0)
    n_noise   = int(np.sum(labels == -1))
    n_other   = int(np.sum((labels >= 0) & (labels != largest_label)))
    return {
        'min':               core.min(axis=0).tolist(),
        'max':               core.max(axis=0).tolist(),
        'centroid':          core.mean(axis=0).tolist(),
        'volume':            float(np.prod(np.clip(dims, 0.01, None))),
        'n_gaussians':       int(len(positions)),  # total labeled, not just sample
        'n_floaters_removed': n_noise + n_other,
    }


def load_semantic_ply(path: Path):
    """
    Load a semantic PLY and return (xyz, labels).

    paint_semantic_gaussians.py stores the label as the dominant f_dc color.
    It doesn't write an explicit label field — so we recover the label by
    finding the nearest CLASS_RGB color for each Gaussian's f_dc values.

    Returns:
        xyz    (N, 3) float32
        labels (N,)  int  — 0 = unlabeled, 1..10 = class index
    """
    print(f"[...] Loading {path} ...")
    ply = PlyData.read(str(path))
    verts = ply['vertex']
    print(f"[OK]  {len(verts):,} Gaussians")
    print(f"      Fields: {list(verts.data.dtype.names[:8])} ...")

    xyz = np.stack([
        np.array(verts['x'], dtype=np.float32),
        np.array(verts['y'], dtype=np.float32),
        np.array(verts['z'], dtype=np.float32),
    ], axis=1)

    # Check if there's an explicit label field
    field_names = verts.data.dtype.names
    for candidate in ('label', 'semantic_label', 'f_label', 'class_idx'):
        if candidate in field_names:
            print(f"[OK]  Using explicit label field: '{candidate}'")
            return xyz, np.array(verts[candidate], dtype=np.int32)

    # No explicit label field — recover from f_dc SH coefficients directly.
    # paint_semantic_gaussians.py sets f_dc = (rgb - 0.5) / SH_C0.
    # We match in f_dc space (avoids the sigmoid saturation problem).
    print("[...] No explicit label field found — recovering labels from f_dc values")
    SH_C0 = 0.28209479177387814

    # Reconstruct class SH-DC palette (must match paint_semantic_gaussians.py)
    CLASS_COLORS_HEX = {
        'bed':     '#FFB6C1', 'desk':    '#20B2AA', 'chair':   '#4682B4',
        'laptop':  '#6495ED', 'monitor': '#2F6F6F', 'fan':     '#FF6347',
        'lamp':    '#FFD700', 'shelf':   '#CD853F', 'door':    '#DEB887',
        'window':  '#87CEEB',
    }
    CLASSES = ['bed', 'desk', 'chair', 'laptop', 'monitor',
               'fan', 'lamp', 'shelf', 'door', 'window']

    # Convert target RGB → expected f_dc = (rgb - 0.5) / SH_C0
    class_sh = np.zeros((len(CLASSES) + 1, 3), dtype=np.float32)  # 0 = unlabeled
    for i, cls in enumerate(CLASSES):
        h = CLASS_COLORS_HEX[cls].lstrip('#')
        rgb_val = np.array([int(h[j:j+2], 16) / 255.0 for j in (0, 2, 4)], dtype=np.float32)
        class_sh[i + 1] = (rgb_val - 0.5) / SH_C0

    f_dc = np.stack([
        np.array(verts['f_dc_0'], dtype=np.float32),
        np.array(verts['f_dc_1'], dtype=np.float32),
        np.array(verts['f_dc_2'], dtype=np.float32),
    ], axis=1)  # (N, 3)

    # Match each Gaussian's f_dc to nearest class SH vector
    diffs = f_dc[:, None, :] - class_sh[None, :, :]  # (N, K+1, 3)
    dists = np.linalg.norm(diffs, axis=2)             # (N, K+1)
    best      = np.argmin(dists, axis=1)              # (N,)  0..K
    best_dist = dists[np.arange(len(dists)), best]

    # SH_C0 ~ 0.282, so RGB diff 0.05 → SH diff ~0.18. Use 0.25 as threshold.
    SH_THRESH = 0.25
    labels = np.where((best > 0) & (best_dist < SH_THRESH), best, 0)

    n_labeled = (labels > 0).sum()
    print(f"[OK]  Color-recovered labels: {n_labeled:,} / {len(labels):,} "
          f"({100.0*n_labeled/len(labels):.1f}%)")
    return xyz, labels


def main():
    ap = argparse.ArgumentParser(
        description="Recompute object volumes by DBSCAN-cleaning floater Gaussians."
    )
    ap.add_argument('--semantic_ply',
        default=str(ROOT / 'outputs/splat_v3/scene_semantic_v3.ply'),
        help='Semantic PLY produced by paint_semantic_gaussians.py')
    ap.add_argument('--objects_json',
        default=str(ROOT / 'outputs/objects_3d_v3.json'),
        help='Existing objects JSON (centroids / metadata source)')
    ap.add_argument('--output_json',
        default=str(ROOT / 'outputs/objects_3d_v3_fixed.json'),
        help='Output path for fixed JSON')
    ap.add_argument('--apply', action='store_true',
        help='Overwrite --objects_json in-place after saving --output_json')
    args = ap.parse_args()

    ply_path  = Path(args.semantic_ply)
    json_path = Path(args.objects_json)
    out_path  = Path(args.output_json)

    if not ply_path.exists():
        sys.exit(f"[error] Semantic PLY not found: {ply_path}\n"
                 "  Run paint_semantic_gaussians.py first.")

    xyz, labels = load_semantic_ply(ply_path)

    with open(json_path) as f:
        objects = json.load(f)

    print()
    print(f"{'Class':<12} {'Raw Gaussians':>15} {'Old Vol (m³)':>13} "
          f"{'New Vol (m³)':>13} {'Floaters removed':>18}")
    print("-" * 73)

    for label_idx, class_name in LABEL_MAP.items():
        if class_name == 'background':
            continue

        mask = labels == label_idx
        pts  = xyz[mask]
        n_raw = len(pts)
        old_vol = objects.get(class_name, {}).get('volume_m3', '—')
        old_vol_str = f"{old_vol:.2f}" if isinstance(old_vol, float) else str(old_vol)

        if n_raw < 10:
            print(f"  {class_name:<12} {n_raw:>13,}   {'(too few, skip)':>13}")
            continue

        params = DBSCAN_PARAMS.get(class_name, {'eps': 0.3, 'min_samples': 50})
        result = clean_bbox(pts, **params)

        if result is None:
            print(f"  {class_name:<12} {n_raw:>13,}   {old_vol_str:>13}   "
                  f"{'(no cluster found)':>13}")
            continue

        print(f"  {class_name:<12} {n_raw:>13,}   {old_vol_str:>13}   "
              f"{result['volume']:>13.2f}   {result['n_floaters_removed']:>16,}")

        if class_name not in objects:
            objects[class_name] = {}
        objects[class_name].update({
            'bbox_min':    result['min'],
            'bbox_max':    result['max'],
            'centroid_3d': result['centroid'],
            'volume_m3':   result['volume'],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(objects, f, indent=2)
    print(f"\n[OK] Saved → {out_path}")

    if args.apply:
        import shutil
        shutil.copy(out_path, json_path)
        print(f"[OK] Applied → {json_path} (overwritten)")
    else:
        print(f"\nTo apply: cp {out_path} {json_path}")
        print(f"Or re-run with: --apply")


if __name__ == '__main__':
    main()