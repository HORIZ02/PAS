# -*- coding: utf-8 -*-
"""65 · 全队列 K=3 生境组成（L/M/H 体素数、体积占比、各生境信号均值）
输出：结果_生境\\15_生境组成_K3.csv
用途：B4 多病例 montage 的选例与标注（使标注与画面一致）
★ 必须 strip 文件编号（10 例带前后空格，不 strip 会静默丢例）
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

import os, sys, time
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to

B = PROJECT_ROOT
DAT = os.path.join(B, '使用的数据', '影像数据')
OUT = os.path.join(B, r'结果_生境\15_生境组成_K3.csv')
LAB = ['L', 'M', 'H']
assert os.path.isdir(os.path.dirname(OUT))

E = pd.read_excel(os.path.join(B, r'使用的数据\临床数据\IPP_PAS.xlsx'))
E['fid'] = E['文件编号'].astype(str).str.strip()            # ★ strip
F = pd.read_csv(os.path.join(B, r'探索性分析\02_影像形态与信号特征.csv'), encoding='utf-8-sig')
M = E[['fid', 'Groups']].merge(F[['fid', 'dark_frac_2', 'resid_sd_rel', 'volume_cm3']], on='fid')
assert len(M) == 202, ('合并丢例', len(M))
print('队列 %d 例' % len(M), flush=True)

rows, t0 = [], time.time()
for i, r in M.iterrows():
    fid = r['fid']
    o = nib.load(os.path.join(DAT, 'masks', 'habitat_k3', 'habitat_L_%s.nii.gz' % fid))
    Hm = {}
    for L in LAB:
        oo = nib.load(os.path.join(DAT, 'masks', 'habitat_k3', 'habitat_%s_%s.nii.gz' % (L, fid)))
        Hm[L] = np.asarray(oo.dataobj) > 0.5
    U = np.logical_or.reduce([Hm[L] for L in LAB])
    assert int(sum(Hm[L].astype(np.uint8) for L in LAB).max()) == 1, ('生境重叠', fid)
    img = nib.load(os.path.join(DAT, 'images', fid + '.nii.gz'))
    S = np.asarray(resample_from_to(img, nib.Nifti1Image(
        np.zeros(o.shape, np.uint8), o.affine), order=1).dataobj, dtype=float)
    v = S[U]
    med = float(np.median(v)); sd = 1.4826 * (float(np.median(np.abs(v - med))) or 1.0)
    d = dict(fid=fid, g='PAS' if r['Groups'] == 1 else 'IPP',
             n_roi=int(U.sum()), vol_cm3=round(float(r['volume_cm3']), 3),
             dark_frac_2=round(float(r['dark_frac_2']), 5),
             resid_sd_rel=round(float(r['resid_sd_rel']), 5))
    for L in LAB:
        n = int(Hm[L].sum())
        d['n_' + L] = n
        d['frac_' + L] = round(n / U.sum(), 5)
        d['meanZ_' + L] = round(float(np.nanmean((S[Hm[L]] - med) / sd)), 3)
    rows.append(d)
    if (len(rows) % 50) == 0:
        print('  %d/%d  %.0fs' % (len(rows), len(M), time.time() - t0), flush=True)

D = pd.DataFrame(rows)
assert len(D) == 202
for _, r in D.iterrows():
    assert r['n_L'] + r['n_M'] + r['n_H'] == r['n_roi'], ('体素和≠ROI', r['fid'])   # ★ 整数精确校验
    assert abs(r['frac_L'] + r['frac_M'] + r['frac_H'] - 1.0) < 1e-4, ('占比和不为1', r['fid'])
    assert r['meanZ_L'] < r['meanZ_M'] < r['meanZ_H'], ('生境均值未升序', r['fid'])
D.to_csv(OUT, index=False, encoding='utf-8-sig')
print('wrote', OUT, os.path.getsize(OUT), '｜用时 %.0fs' % (time.time() - t0))
print()
print('frac_L 中位：PAS %.4f ｜ IPP %.4f ｜比值 %.2f'
      % (D[D.g == 'PAS']['frac_L'].median(), D[D.g == 'IPP']['frac_L'].median(),
         D[D.g == 'PAS']['frac_L'].median() / max(D[D.g == 'IPP']['frac_L'].median(), 1e-9)))
print('frac_L 与 dark_frac_2 的 Spearman ρ = %.3f'
      % D[['frac_L', 'dark_frac_2']].corr(method='spearman').iloc[0, 1])
from scipy import stats
u = stats.mannwhitneyu(D[D.g == 'PAS']['frac_L'], D[D.g == 'IPP']['frac_L'])
print('frac_L 组间 Mann-Whitney P = %.2e（U=%.0f）' % (u.pvalue, u.statistic))
print('frac_L 各生境均值(z)中位：L %.2f  M %.2f  H %.2f'
      % (D['meanZ_L'].median(), D['meanZ_M'].median(), D['meanZ_H'].median()))
