# -*- coding: utf-8 -*-
"""57 · ROI 信号一致性 QC（全 202 例）

指标：ROI 内均值 / 体内均值（体内 = 图像强度 > 40 百分位的体素）。
胎盘在 T2WI 上偏亮，故该比值过低（掩膜落在相对暗的组织上）提示 ROI 可能未勾在胎盘上。
输出：探索性分析/08_ROI信号一致性QC.csv
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

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

B = PROJECT_ROOT
IMG = os.path.join(B, r'使用的数据\影像数据\images')
MSK = os.path.join(B, r'使用的数据\影像数据\masks')
OUT = os.path.join(B, '探索性分析', '08_ROI信号一致性QC.csv')
S26 = np.ones((3, 3, 3), bool)

rows = []
if os.path.exists(OUT) and os.path.getsize(OUT) > 100:
    rows = pd.read_csv(OUT, encoding='utf-8-sig').to_dict('records')
done = {r['fid'] for r in rows}
fids = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(MSK, '*.nii.gz')))
print('掩膜 %d 例，已完成 %d' % (len(fids), len(done)), flush=True)
for i, fid in enumerate(fids):
    if fid in done:
        continue
    try:
        im = nib.load(os.path.join(IMG, fid + '.nii.gz'))
        mm = nib.load(os.path.join(MSK, fid + '.nii.gz'))
        S = np.asanyarray(im.dataobj).astype(np.float32)
        M0 = np.asanyarray(mm.dataobj) > 0.5
        lab, n = ndimage.label(M0, structure=S26)
        szs = np.array(ndimage.sum(M0, lab, range(1, n + 1)))
        M = ndimage.binary_fill_holes(lab == (int(np.argmax(szs)) + 1))
        body = S > np.percentile(S, 40)
        bm = float(S[body].mean()) or 1e-6
        rm = float(S[M].mean())
        # ROI 内部异质性（与 resid 类特征同源：ROI 内强度的稳健离散度）
        x = S[M]; mad = float(np.median(np.abs(x - np.median(x)))) or 1.0
        rows.append(dict(fid=fid, roi_mean=round(rm, 2), body_mean=round(bm, 2),
                         ratio=round(rm / bm, 4), n_comp=int(n),
                         largest_frac=round(float(szs.max() / szs.sum()), 4),
                         roi_mad_rel=round(mad / (abs(float(np.median(x))) or 1.0), 5)))
    except Exception as e:
        print('  [失败] %s: %s' % (fid, e), flush=True)
    if (i + 1) % 25 == 0:
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
        print('  %d/%d' % (i + 1, len(fids)), flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False, encoding='utf-8-sig')
v = R['ratio'].values
print('\nn=%d  ROI/体内 均值比：中位 %.2f  IQR %.2f–%.2f  范围 %.2f–%.2f'
      % (len(v), np.median(v), np.percentile(v, 25), np.percentile(v, 75), v.min(), v.max()))
q1 = float(np.percentile(v, 25))
poor = R[R['ratio'] < q1]
print('低于 25 百分位（%.2f）的例数：%d' % (q1, len(poor)))
for fid in ('CASE_ID_1', 'CASE_ID_2'):
    r = R[R['fid'] == fid]
    if len(r):
        pct = float((v < r['ratio'].iloc[0]).mean() * 100)
        print('  %s  比值 %.2f  百分位 %.0f%%' % (fid, r['ratio'].iloc[0], pct))
print('wrote', OUT, os.path.getsize(OUT))
