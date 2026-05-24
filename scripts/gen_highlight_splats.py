"""Generate per-object highlight .splat files from scene_semantic.ply."""
import numpy as np
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SH_C0 = 0.28209479177387814

CLASS_COLORS_HEX = {
    "bed":     "#FFB6C1",
    "desk":    "#20B2AA",
    "chair":   "#4682B4",
    "laptop":  "#6495ED",
    "monitor": "#2F6F6F",
    "fan":     "#FF6347",
    "lamp":    "#FFD700",
    "shelf":   "#CD853F",
    "door":    "#DEB887",
    "window":  "#87CEEB",
}

def hex_to_rgb01(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def read_ply(path):
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line.decode('utf-8', 'replace').rstrip())
            if header_lines[-1] == 'end_header':
                break
        n_verts = 0; props = []; in_vert = False
        for hl in header_lines:
            t = hl.strip()
            if t.startswith('element vertex'):
                n_verts = int(t.split()[-1]); in_vert = True
            elif t.startswith('element') and in_vert:
                in_vert = False
            elif t.startswith('property') and in_vert:
                parts = t.split(); props.append((parts[1], parts[2]))
            elif t == 'end_header':
                break
        type_map = {
            'float': np.float32, 'float32': np.float32, 'double': np.float64,
            'int': np.int32, 'uint': np.uint32, 'uchar': np.uint8, 'char': np.int8,
            'short': np.int16, 'ushort': np.uint16, 'int32': np.int32,
            'uint32': np.uint32, 'float64': np.float64,
        }
        dtype = np.dtype([(name, type_map.get(t, np.float32)) for t, name in props])
        data = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype).copy()
    return data, n_verts

def write_splat(path, data_subset, rgb_override=None):
    d = data_subset
    N = len(d)
    if N == 0:
        return

    xyz   = np.column_stack([d['x'], d['y'], d['z']]).astype(np.float32)
    scale = np.column_stack([d['scale_0'], d['scale_1'], d['scale_2']]).astype(np.float32)
    ops   = sigmoid(d['opacity'].astype(np.float32))
    rots  = np.column_stack([d['rot_0'], d['rot_1'], d['rot_2'], d['rot_3']]).astype(np.float32)

    if rgb_override is not None:
        r_u8 = np.full(N, int(rgb_override[0] * 255), dtype=np.uint8)
        g_u8 = np.full(N, int(rgb_override[1] * 255), dtype=np.uint8)
        b_u8 = np.full(N, int(rgb_override[2] * 255), dtype=np.uint8)
    else:
        r_u8 = (np.clip(d['f_dc_0'] * SH_C0 + 0.5, 0, 1) * 255).astype(np.uint8)
        g_u8 = (np.clip(d['f_dc_1'] * SH_C0 + 0.5, 0, 1) * 255).astype(np.uint8)
        b_u8 = (np.clip(d['f_dc_2'] * SH_C0 + 0.5, 0, 1) * 255).astype(np.uint8)

    a_u8 = np.full(N, 255, dtype=np.uint8)

    scale_exp = np.exp(scale)
    rot_n = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-12)
    rot_u8 = np.clip((rot_n + 1.0) * 0.5 * 255, 0, 255).astype(np.uint8)

    order = np.argsort(-ops)
    buf = np.zeros((N, 32), dtype=np.uint8)
    buf[:, 0:12]  = xyz[order].view(np.uint8).reshape(N, 12)
    buf[:, 12:24] = scale_exp[order].view(np.uint8).reshape(N, 12)
    buf[:, 24] = r_u8[order]; buf[:, 25] = g_u8[order]
    buf[:, 26] = b_u8[order]; buf[:, 27] = a_u8[order]
    buf[:, 28] = rot_u8[order, 0]; buf[:, 29] = rot_u8[order, 1]
    buf[:, 30] = rot_u8[order, 2]; buf[:, 31] = rot_u8[order, 3]

    with open(path, 'wb') as f:
        f.write(buf.tobytes())

# Class index mapping must match paint_semantic_gaussians.py LABEL_TO_IDX
# index 0 = unlabeled, 1..N = classes in this order:
CLASS_ORDER = [
    "bed", "desk", "chair", "laptop",
    "monitor", "fan", "lamp", "shelf",
    "door", "window",
]
IDX_TO_CLASS = {i+1: c for i, c in enumerate(CLASS_ORDER)}

def main():
    ply_path   = ROOT / 'outputs/splat_v4/scene_aligned.ply'  # original appearance
    class_npy  = ROOT / 'outputs/splat_v4/semantic_class.npy'
    out_dir    = ROOT / 'outputs/splat_v4/highlights'
    out_dir.mkdir(exist_ok=True)

    if not class_npy.exists():
        print(f"[error] {class_npy} not found.")
        print("Run paint_semantic_gaussians.py first (it saves semantic_class.npy).")
        return

    print(f"Loading class map from {class_npy} ...")
    semantic_class = np.load(str(class_npy))  # uint8, 0=unlabeled, 1..N=class
    print(f"  {len(semantic_class):,} entries")

    print(f"Reading original PLY {ply_path} ...")
    data, N = read_ply(ply_path)
    print(f"  {N:,} Gaussians")

    assert len(semantic_class) == N, f"Class map length {len(semantic_class)} != PLY count {N}"

    print(f"\nGenerating per-object highlight splats -> {out_dir}")
    manifest = {}
    for idx, cls in IDX_TO_CLASS.items():
        mask = semantic_class == idx
        n = mask.sum()
        if n == 0:
            print(f"  {cls:10s}: 0 Gaussians — skipping")
            continue
        rgb_full  = hex_to_rgb01(CLASS_COLORS_HEX[cls])
        out_path  = out_dir / f'{cls}.splat'
        write_splat(out_path, data[mask], rgb_override=rgb_full)
        sz = os.path.getsize(out_path) / 1024
        manifest[cls] = f'/outputs/splat_v4/highlights/{cls}.splat'
        print(f"  {cls:10s}: {n:>8,} Gaussians -> {cls}.splat ({sz:.0f} KB)")

    # Write manifest JSON for the viewer
    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {manifest_path}")
    print("Done.")

if __name__ == '__main__':
    main()