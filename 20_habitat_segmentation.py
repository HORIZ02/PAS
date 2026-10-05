# -*- coding: utf-8 -*-
"""P0 上游 · 生境分割（空间生境影像组学）
掩膜清洗 → 裁剪 → 面内重采样 1.0mm → ROI 内稳健标准化 → K-means(K=2/3/4) → 生境 mask 落盘

输出:
  {RAW}/使用的数据/影像数据/masks/habitat_k{k}/habitat_{L}_{fid}.nii.gz
  {RAW}/方法设计/生境QC.csv               每例: 各生境体积/平均信号/K-means切点/一致性校验

约定（方案 v2 §2.1）:
  - 掩膜清洗: 最大连通分量 + 填洞（94/202 例有多分量）
  - 面内重采样到 1.0 mm（线性）；层间保持原始 sz，不插值（避免"制造" z 分辨率）
  - 强度归一化: ROI 内 robust z-score（中位/MAD）→ 跨例可比
  - 生境命名: 按生境平均信号排序 → L(低)/M(中)/H(高)；K=4 → L/ML/MH/H
  - 校验: 各生境互斥且并集 == 清洗后 ROI（逐例断言）

用法: python 20_生境分割_P0.py [limit]
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

import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import contextlib, threadpoolctl
threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()

import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
from sklearn.cluster import KMeans

RAW = PROJECT_ROOT
IMG = os.path.join(RAW, r'使用的数据\影像数据\images')
MSK = os.path.join(RAW, r'使用的数据\影像数据\masks')
OUTBASE = MSK
QC = os.path.join(RAW, r'方法设计\生境QC.csv')
TARGET_MM = 1.0
KS = [2, 3, 4]
LABELS = {2: ['L', 'H'], 3: ['L', 'M', 'H'], 4: ['L', 'ML', 'MH', 'H']}
N_FIT = 20000          # K-means 拟合用体素上限（全量体素再统一指派）
S26 = np.ones((3, 3, 3), bool)
SEED = 42


def clean_mask(M0):
    lab, n = ndimage.label(M0, structure=S26)
    if n == 0:
        return None, 0
    sz = np.atleast_1d(ndimage.sum(M0, lab, range(1, n + 1)))
    return ndimage.binary_fill_holes(lab == (int(np.argmax(sz)) + 1)), int(n)


def resample_inplane(S, M, sp, target=TARGET_MM):
    """只重采样面内两轴；第 3 轴（层间）保持原状。
    返回 (S2, M2, sp_new, scale) —— scale 用于重建 affine。"""
    f = (sp[0] / target, sp[1] / target, 1.0)
    if abs(f[0] - 1) < 1e-3 and abs(f[1] - 1) < 1e-3:
        return S, M, sp, (1.0, 1.0, 1.0)
    S2 = zoom(S, f, order=1, prefilter=False)
    M2 = zoom(M.astype(np.float32), f, order=0, prefilter=False) > 0.5
    return S2, M2, (target, target, sp[2]), f


def crop_affine(A, mn, zoom_f):
    """裁剪偏移 mn（原体素单位）+ 重采样后的新 affine。
    zoom 因子 f = sp_old/sp_new ⇒ 新索引 i' ↔ 原索引 mn + i'*f
    ⇒ 新列 = 原列 / f；平移量仍以"原体素"为单位，故不缩放。"""
    A = np.asarray(A, float)
    A2 = A.copy()
    for c in range(3):
        A2[:3, c] = A[:3, c] / float(zoom_f[c])
    A2[:3, 3] = (A[:3, 0] * mn[0] + A[:3, 1] * mn[1] + A[:3, 2] * mn[2] + A[:3, 3])
    return A2


def one_case(fid, ks=KS):
    im, mm = nib.load(os.path.join(IMG, fid + '.nii.gz')), nib.load(os.path.join(MSK, fid + '.nii.gz'))
    sp0 = tuple(float(s) for s in im.header.get_zooms()[:3])
    S = np.asanyarray(im.dataobj).astype(np.float32)
    M, ncomp = clean_mask(np.asanyarray(mm.dataobj) > 0.5)
    if M is None or M.sum() < 500:
        return None, f'{fid}: 清洗后体素不足'
    # 裁剪（±3 体素）后再重采样，省算力
    cs = np.argwhere(M)
    mn = np.clip(cs.min(0) - 3, 0, None); mx = cs.max(0) + 4
    sl = tuple(slice(int(mn[i]), int(mx[i])) for i in range(3))
    S, M, sp, scale = resample_inplane(S[sl], M[sl], sp0)
    if M.sum() < 500:
        return None, f'{fid}: 重采样后体素不足'
    # 重建 affine（否则头里体素尺寸仍是原始值 → 下游表面积/体积/距离全错）
    A2 = crop_affine(im.affine, mn, scale)  # scale 即 zoom 因子 f
    _zz = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), A2).header.get_zooms()[:3]
    assert abs(_zz[0] - sp[0]) < 1e-3 and abs(_zz[1] - sp[1]) < 1e-3, \
        (f'{fid} affine 体素尺寸不符', _zz, sp)
    # ROI 内稳健标准化
    x = S[M]
    med = float(np.median(x)); mad = float(np.median(np.abs(x - np.median(x)))) or 1.0
    z = (x - med) / (1.4826 * mad)
    Zc = np.zeros(S.shape, np.float32); Zc[M] = z
    # K-means 拟合（子样本）→ 全量指派
    idx = np.arange(len(z))
    if len(z) > N_FIT:
        rng = np.random.RandomState(SEED); idx = rng.choice(idx, N_FIT, replace=False)
    Zf = z[idx].reshape(-1, 1)
    rec = {'fid': fid, 'n_vox_clean': int(M.sum()), 'n_comp_orig': ncomp,
           'sp_x': round(sp[0], 4), 'sp_y': round(sp[1], 4), 'sz': round(sp[2], 4),
           'vol_clean_cm3': round(float(M.sum()) * np.prod(sp) / 1000, 2)}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Zf)
        lab_all = km.predict(Zc[M].reshape(-1, 1))
        means = np.array([z[lab_all == c].mean() if (lab_all == c).any() else np.inf
                          for c in range(k)])
        order = np.argsort(means)                       # 按平均信号升序 → L..H
        names = LABELS[k]
        vsum = 0
        for pos, c in enumerate(order):
            Lname = names[pos]
            Hm = np.zeros(S.shape, bool); Hm[M] = (lab_all == c)
            vsum += int(Hm.sum())
            vol = float(Hm.sum()) * np.prod(sp) / 1000
            rec[f'k{k}_{Lname}_vol_cm3'] = round(vol, 2)
            rec[f'k{k}_{Lname}_mean_z'] = round(float(means[c]), 3)
            rec[f'k{k}_{Lname}_share'] = round(float(Hm.sum()) / int(M.sum()), 4)
            # 落盘（.nii.gz 强制）
            d = os.path.join(OUTBASE, f'habitat_k{k}')
            os.makedirs(d, exist_ok=True)
            nib.save(nib.Nifti1Image(Hm.astype(np.uint8), A2),
                     os.path.join(d, f'habitat_{Lname}_{fid}.nii.gz'))
        # 校验：并集 == 清洗后 ROI
        assert vsum == int(M.sum()), f'{fid} K={k} 生境并集 {vsum} != ROI {int(M.sum())}'
        # 一致性：切点单调
        cut = np.sort(means)
        rec[f'k{k}_cuts'] = '|'.join('%.3f' % c for c in cut)
    return rec, None


def main():
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = pd.read_excel(os.path.join(RAW, r'使用的数据\临床数据\IPP_PAS.xlsx'))
    df['文件编号'] = df['文件编号'].astype(str).str.strip()
    fids = df['文件编号'].tolist()
    if lim: fids = fids[:lim]
    rows, done = [], set()
    if os.path.exists(QC) and os.path.getsize(QC) > 100:
        p = pd.read_csv(QC); rows = p.to_dict('records'); done = set(p['fid'].astype(str))
        print(f'续跑 {len(done)} 例', flush=True)
    t0 = time.time()
    for i, fid in enumerate(fids):
        if fid in done: continue
        try:
            rec, err = one_case(fid)
        except Exception as e:
            import traceback
            print(f'  [失败] {fid}: {e}', flush=True)
            if i < 2: traceback.print_exc()
            continue
        if rec is None:
            print(f'  [跳过] {err}', flush=True); continue
        rows.append(rec); done.add(fid)
        pd.DataFrame(rows).to_csv(QC, index=False, encoding='utf-8-sig')
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f'  {i+1}/{len(fids)}  {el:.0f}s  余 ~{el/max(i+1,1)*(len(fids)-i-1):.0f}s', flush=True)
    R = pd.DataFrame(rows); R.to_csv(QC, index=False, encoding='utf-8-sig')
    print(f'\n完成 {len(R)} 例 -> {QC}', flush=True)
    for k in KS:
        cols = [c for c in R.columns if c.startswith(f'k{k}_') and c.endswith('_share')]
        sh = R[cols].mean()
        print(f'  K={k} 各生境平均占比: ' +
              ', '.join(f'{c.split("_")[1]}={v:.3f}' for c, v in sh.items()), flush=True)


if __name__ == '__main__':
    main()
