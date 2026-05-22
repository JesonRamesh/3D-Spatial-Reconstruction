import numpy as np
from plyfile import PlyData

ply = PlyData.read('outputs/splat_video_v2/scene.ply')
v = ply['vertex'].data
print('Fields:', v.dtype.names)

scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)
opacities = v['opacity']
sig = lambda x: 1/(1+np.exp(-x))
ops = sig(opacities)

print('\nOpacity distribution (after sigmoid):')
for t in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]:
    print(f'  op>={t}: {(ops>=t).sum():,} ({100*(ops>=t).mean():.1f}%)')

max_scale = np.exp(scales).max(axis=1)
print('\nMax scale percentiles:', np.percentile(max_scale, [25,50,75,90,95,99]).round(4))
print('Max scale histogram:')
for lo, hi in [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.5),(0.5,1.0),(1.0,5.0),(5.0,100)]:
    n = ((max_scale>=lo) & (max_scale<hi)).sum()
    print(f'  {lo:.2f}-{hi:.2f}: {n:,} ({100*n/len(max_scale):.1f}%)')

pos = np.stack([v['x'], v['y'], v['z']], axis=1)
print('\nPosition range:')
print('  X:', pos[:,0].min().round(2), 'to', pos[:,0].max().round(2))
print('  Y:', pos[:,1].min().round(2), 'to', pos[:,1].max().round(2))
print('  Z:', pos[:,2].min().round(2), 'to', pos[:,2].max().round(2))

# Check if floaters correlate with large scale
large_scale = max_scale > 0.5
low_op = ops < 0.1
print(f'\nLarge scale (>0.5m) Gaussians: {large_scale.sum():,}')
print(f'Large scale AND low opacity: {(large_scale & low_op).sum():,}')
print(f'Large scale AND high opacity: {(large_scale & ~low_op).sum():,}')