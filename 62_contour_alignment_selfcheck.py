# -*- coding: utf-8 -*-
"""自检：contour(extent=...) 与 imshow(extent=...) 是否上下翻转（matplotlib 3.5.1 会）。
判据：描边像素到填充边界的平均距离。对齐正确 ≈ 3 px；翻转 ≈ 35 px。
对应修法见 61/56 脚本内的 contour_xy()。
"""

# ---------------------------------------------------------------------------
# Working-copy root. Set the environment variable PAS_PROJECT_ROOT to the root of
# your own copy of the project (the folder that contains the imaging data, the
# clinical table and the mask folders). No personal path is hard-coded here.
import os as _os
PROJECT_ROOT = _os.environ.get('PAS_PROJECT_ROOT')
if not PROJECT_ROOT:
    raise SystemExit(
        'PAS_PROJECT_ROOT is not set. Point it at the root of your working copy, e.g.\n'
        '  Windows   : set PAS_PROJECT_ROOT=D:\\my_project\n'
        '  macOS/Linux: export PAS_PROJECT_ROOT=/path/to/my_project')
# ---------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""决定性判据：红色描边像素 → 白色填充边界 的平均距离（像素）。
对齐正确时 ≈1–2 px；上下翻转时会是几十 px。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

N = 60
m = np.zeros((N, N))
m[6:26, 12:52] = 1.0
m[40:48, 44:56] = 1.0
EXT = [0, N, 0, N]


def render(fn, mode):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.imshow(m, cmap='gray', vmin=0, vmax=1, extent=EXT, origin='upper', interpolation='nearest')
    if mode == 'extent':
        ax.contour(m, levels=[0.5], colors=['red'], linewidths=2.0, extent=EXT)
    else:
        xs = np.linspace(EXT[0], EXT[1], m.shape[1])
        ys = np.linspace(EXT[3], EXT[2], m.shape[0])
        ax.contour(xs, ys, m, levels=[0.5], colors=['red'], linewidths=2.0)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(fn, dpi=120); plt.close(fig)
    return np.asarray(Image.open(fn).convert('RGB')).astype(int)


for mode in ('extent', 'xy'):
    a = render(r'C:\Users\Administrator\AppData\Local\hermes\workspace\_ctv_%s.png' % mode, mode)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 170) & (g < 100) & (b < 100)
    white = (r > 235) & (g > 235) & (b > 235)
    # 去掉最外 3px，避免坐标轴框线
    red[:3, :] = False; red[-3:, :] = False; red[:, :3] = False; red[:, -3:] = False
    white[:3, :] = False; white[-3:, :] = False; white[:, :3] = False; white[:, -3:] = False
    bnd = white & ~ndimage.binary_erosion(white, np.ones((3, 3)))
    if not red.any() or not bnd.any():
        print(mode, '无像素，跳过'); continue
    d = ndimage.distance_transform_edt(~bnd)          # 每个像素到最近边界的距离
    dv = d[red]
    print('%-7s 描边 %5d px ｜ 填充 %6d px ｜ 边界 %5d px ｜ 描边→边界距离 中位 %6.1f px  均值 %6.1f px  P90 %6.1f px'
          % (mode, int(red.sum()), int(white.sum()), int(bnd.sum()),
             float(np.median(dv)), float(dv.mean()), float(np.percentile(dv, 90))))
print()
print('判据：中位距离 ≈1–3 px ⇒ 描边贴住填充边界（正确）；≥20 px ⇒ 明显错位/翻转')
