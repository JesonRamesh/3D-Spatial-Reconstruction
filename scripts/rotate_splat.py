#!/usr/bin/env python3
import numpy as np
from plyfile import PlyData, PlyElement

INPUT_PLY  = 'outputs/splat_v3/scene.ply'
OUTPUT_PLY = 'outputs/splat_v3/scene_yup.ply'

WORLD_UP = np.array([0.06378696878537453, -0.833419284810258, -0.153478649498545])


def rotation_between(a, b):
    a = np.array(a, dtype=float) / np.linalg.norm(a)
    b = np.array(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    kmat = np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])
    return np.eye(3) + kmat + kmat @ kmat * ((1.0 - c) / (s ** 2))


def quat_to_rotmat(w, x, y, z):
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),      2*(x*z+w*y)    ],
        [2*(x*y+w*z),    1-2*(x*x+z*z),    2*(y*z-w*x)    ],
        [2*(x*z-w*y),    2*(y*z+w*x),      1-2*(x*x+y*y)  ],
    ])


def rotmat_to_quat(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def batch_rotate_quats(quats_wxyz, R):
    w, x, y, z = quats_wxyz[:,0], quats_wxyz[:,1], quats_wxyz[:,2], quats_wxyz[:,3]
    Rg = np.zeros((len(quats_wxyz), 3, 3))
    Rg[:,0,0]=1-2*(y*y+z*z); Rg[:,0,1]=2*(x*y-w*z); Rg[:,0,2]=2*(x*z+w*y)
    Rg[:,1,0]=2*(x*y+w*z);   Rg[:,1,1]=1-2*(x*x+z*z); Rg[:,1,2]=2*(y*z-w*x)
    Rg[:,2,0]=2*(x*z-w*y);   Rg[:,2,1]=2*(y*z+w*x);   Rg[:,2,2]=1-2*(x*x+y*y)
    Rnew = R[None,:,:] @ Rg
    trace = Rnew[:,0,0] + Rnew[:,1,1] + Rnew[:,2,2]
    s = 0.5 / np.sqrt(np.clip(trace + 1.0, 1e-8, None))
    return np.stack([0.25/s,
                     (Rnew[:,2,1]-Rnew[:,1,2])*s,
                     (Rnew[:,0,2]-Rnew[:,2,0])*s,
                     (Rnew[:,1,0]-Rnew[:,0,1])*s], axis=1)


def main():
    print(f"Loading {INPUT_PLY}...")
    ply = PlyData.read(INPUT_PLY)
    verts = ply['vertex']
    n = len(verts)
    print(f"  {n:,} Gaussians")

    R = rotation_between(WORLD_UP, [0.0, 1.0, 0.0])
    print(f"Rotation matrix:\n{R}")

    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1)
    xyz_rot = (R @ xyz.T).T
    print("Positions rotated.")

    quats = np.stack([verts['rot_0'], verts['rot_1'],
                      verts['rot_2'], verts['rot_3']], axis=1)
    print("Rotating quaternions (vectorised)...")
    quats_rot = batch_rotate_quats(quats.astype(np.float64), R)
    print("Quaternions rotated.")

    dtype = verts.data.dtype
    out = np.zeros(n, dtype=dtype)
    for name in dtype.names:
        out[name] = verts[name]

    out['x'] = xyz_rot[:, 0].astype(np.float32)
    out['y'] = xyz_rot[:, 1].astype(np.float32)
    out['z'] = xyz_rot[:, 2].astype(np.float32)
    out['rot_0'] = quats_rot[:, 0].astype(np.float32)
    out['rot_1'] = quats_rot[:, 1].astype(np.float32)
    out['rot_2'] = quats_rot[:, 2].astype(np.float32)
    out['rot_3'] = quats_rot[:, 3].astype(np.float32)

    el = PlyElement.describe(out, 'vertex')
    PlyData([el], text=False).write(OUTPUT_PLY)
    print(f"Saved → {OUTPUT_PLY}")


if __name__ == '__main__':
    main()
