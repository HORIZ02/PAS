# -*- coding: utf-8 -*-
"""03_生境空间特征提取.py — 前置胎盘项目适配版
改编自 research/habitat-spatial-omics/templates/spatial_features_extract.py

与本项目相关的适配点（相对模板）：
  1. mask 布局：**平铺**，文件命名 habitat_{L}.nii.gz 放在 masks/habitat_k{k}/ 下
     （模板原本是 {MASK_ROOT}/{Case}/habitat_{L}.nii.gz 的每例子目录布局）
  2. 单模态：MODALITIES=[('T2','t2')]（原项目为 CT+PET / DWI+DCE 双模态）
  3. 特征计数式：单模态 = 11K + 3·C(K,2) → K=2:25 / K=3:42 / K=4:62

用法:
    set PY=D:\\python\\Anaconda3\\python.exe
    %PY% 03_生境空间特征提取.py 3          # K=3
    %PY% 03_生境空间特征提取.py 2 3 4      # 多 K 依次跑

输出:
    {PROJECT_DIR}/features/habitat_spatial_features_k{k}.csv
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

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage import measure
from skimage.measure import euler_number

warnings.filterwarnings('ignore')

# ============================ CONFIG ★★★ ============================
PROJECT_DIR = PROJECT_ROOT                 # ★ 项目根
MASK_DIR_TEMPLATE = '使用的数据/影像数据/masks/habitat_k{k}'   # ★ 生境 mask 目录（相对 PROJECT_DIR，平铺）
MASK_FILE = 'habitat_{L}_{fid}.nii.gz'               # ★ 文件名模板
MODALITIES = [('T2', 't2')]                          # ★ 单模态：(输出tag, 目录内前缀)
OUT_DIR = 'features'                                 # ★ 输出目录
LABELS = {2: ['L', 'H'], 3: ['L', 'M', 'H'], 4: ['L', 'ML', 'MH', 'H']}   # ★ 值必须是 list

SHAPE_FEATS = ['n_voxels', 'n_components', 'largest_frac', 'euler', 'surface',
               'sphericity', 'fractal_dim', 'centroid_dist', 'radial_mean',
               'radial_sd', 'edge_dist_mean']
INT_FEATS = ('n_voxels', 'n_components', 'euler')
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

# 病例名单来源：临床表「文件编号」列（202 例，已与 images/masks 1:1 对齐）
CASE_LIST_XLSX = '使用的数据/临床数据/IPP_PAS.xlsx'
CASE_LIST_COL = '文件编号'
# ====================================================================


def surface_area(mask, spacing):
    """marching_cubes 表面积 (mm^2)。
    不要用体素面计数: 6-邻域会系统性低估表面积，导致球度 > 1（物理不可能）。"""
    if not mask.any():
        return 0.0
    try:
        padded = np.pad(mask.astype(np.float32), 1)
        v, f, _, _ = measure.marching_cubes(padded, level=0.5, spacing=spacing)
        return float(measure.mesh_surface_area(v, f))
    except Exception:
        return np.nan


def fractal_dim(mask):
    """box-counting 分形维数 (box = 2,4,8,16)。"""
    if not mask.any():
        return np.nan
    coords = np.argwhere(mask)
    mn, mx = coords.min(0), coords.max(0) + 1
    crop = mask[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    counts, sizes = [], []
    for box in [2, 4, 8, 16]:
        if any(s < box for s in crop.shape):
            continue
        d = [crop.shape[i] // box * box for i in range(3)]
        if min(d) == 0:
            continue
        sub = crop[:d[0], :d[1], :d[2]]
        counts.append(sub.reshape(d[0] // box, box, d[1] // box,
                                  box, d[2] // box, box).max(axis=(1, 3, 5)).sum())
        sizes.append(box)
    if len(counts) < 2:
        return np.nan
    return float(np.polyfit(np.log(1.0 / np.array(sizes, float)),
                            np.log(np.array(counts, float)), 1)[0])


def load_case_masks(mask_dir, case, labs, prefix):
    """平铺布局: {mask_dir}/habitat_{L}_{case}.nii.gz"""
    masks, spacing = {}, None
    for L in labs:
        fp = os.path.join(mask_dir, MASK_FILE.format(L=L, fid=case))
        if not os.path.exists(fp):
            fp = fp[:-3]                       # 兼容未压缩 .nii
        if not os.path.exists(fp):
            return None, None
        im = nib.load(fp)
        masks[L] = np.asanyarray(im.dataobj) > 0.5
        if spacing is None:
            spacing = tuple(float(s) for s in im.header.get_zooms()[:3])
    return masks, spacing


def process_case(case, modality, mask_dir, k):
    tag = modality.lower()
    labs = LABELS[k]
    masks, spacing = load_case_masks(mask_dir, case, labs, tag)
    if masks is None:
        return None

    # --- 统一 bbox 裁剪（基于 L|M|H 联合包围盒, ±2 padding）---
    tumor = np.zeros_like(masks[labs[0]], dtype=bool)
    for L in labs:
        tumor |= masks[L]
    nv_t = int(tumor.sum())
    if nv_t == 0:
        return None
    cs = np.argwhere(tumor)
    mn = np.clip(cs.min(0) - 2, 0, None).astype(int)
    mx = (cs.max(0) + 3).astype(int)
    sl = tuple(slice(int(mn[i]), int(mx[i])) for i in range(3))
    M = {L: masks[L][sl] for L in labs}
    Tc = np.zeros_like(M[labs[0]], dtype=bool)
    for L in labs:
        Tc |= M[L]

    vox_vol = spacing[0] * spacing[1] * spacing[2]
    R_eq = (3 * nv_t * vox_vol / (4 * np.pi)) ** (1 / 3)
    tcent = (np.argwhere(Tc) * np.array(spacing)).mean(axis=0)

    # 边界距离图: bbox 内算一次，所有生境复用（必须带 sampling → mm）
    boundary = Tc & ~ndimage.binary_erosion(Tc, structure=STRUCT26)
    dist_map = (ndimage.distance_transform_edt(~boundary, sampling=spacing)
                if boundary.any() else None)

    rec = {'patient': case, 'modality': modality}
    surf = {}
    for L in labs:
        msk = M[L]
        nv = int(msk.sum())
        if nv == 0:
            for kk in SHAPE_FEATS:
                rec[f'{tag}_{L}_{kk}'] = 0 if kk in INT_FEATS else np.nan
            surf[L] = 0.0
            continue
        lab, ncomp = ndimage.label(msk, structure=STRUCT26)
        sizes = ndimage.sum(msk, lab, range(1, ncomp + 1))
        eul = float(euler_number(msk, connectivity=3))
        sa = surface_area(msk, spacing)
        surf[L] = sa if sa == sa else 0.0
        Rh = (3 * nv * vox_vol / (4 * np.pi)) ** (1 / 3)
        sph = (4 * np.pi * Rh ** 2) / sa if (sa and sa > 0) else np.nan
        fd = fractal_dim(msk)
        hc = (np.argwhere(msk) * np.array(spacing)).mean(axis=0)
        cdist = float(np.linalg.norm(hc - tcent) / R_eq) if R_eq > 0 else np.nan
        d = np.linalg.norm((np.argwhere(msk) * np.array(spacing)) - tcent, axis=1)
        rmean = float(d.mean() / R_eq) if R_eq > 0 else np.nan
        rsd = float(d.std(ddof=1) / R_eq) if (len(d) > 1 and R_eq > 0) else np.nan
        edm = float(dist_map[msk].mean()) if dist_map is not None else np.nan
        rec.update({
            f'{tag}_{L}_n_voxels': nv,
            f'{tag}_{L}_n_components': int(ncomp),
            f'{tag}_{L}_largest_frac': round(float(sizes.max()) / nv, 4),
            f'{tag}_{L}_euler': eul,
            f'{tag}_{L}_surface': round(sa, 3) if sa == sa else np.nan,
            f'{tag}_{L}_sphericity': round(sph, 4) if sph == sph else np.nan,
            f'{tag}_{L}_fractal_dim': round(fd, 4) if fd == fd else np.nan,
            f'{tag}_{L}_centroid_dist': round(cdist, 4) if cdist == cdist else np.nan,
            f'{tag}_{L}_radial_mean': round(rmean, 4) if rmean == rmean else np.nan,
            f'{tag}_{L}_radial_sd': round(rsd, 4) if rsd == rsd else np.nan,
            f'{tag}_{L}_edge_dist_mean': round(edm, 4) if edm == edm else np.nan,
        })

    # --- 生境两两交互（对数 = C(K,2)）---
    sp2 = sorted(spacing)
    for A, B in itertools.combinations(labs, 2):
        A_dil = ndimage.binary_dilation(M[A], structure=STRUCT26)
        iface = int(np.sum(A_dil & M[B])) * sp2[0] * sp2[1]
        avg_s = (surf.get(A, 0) + surf.get(B, 0)) / 2
        rec[f'{tag}_{A}_{B}_interface'] = round(iface, 3)
        rec[f'{tag}_{A}_{B}_interface_norm'] = (round(iface / avg_s, 4)
                                                if avg_s > 0 else np.nan)
        if M[A].any() and M[B].any():
            ca = (np.argwhere(M[A]) * np.array(spacing)).mean(axis=0)
            cb = (np.argwhere(M[B]) * np.array(spacing)).mean(axis=0)
            rec[f'{tag}_{A}_{B}_centroid_dist'] = (round(float(np.linalg.norm(ca - cb)) / R_eq, 4)
                                                   if R_eq > 0 else np.nan)
        else:
            rec[f'{tag}_{A}_{B}_centroid_dist'] = np.nan
    return rec


def main():
    ks = [int(x) for x in sys.argv[1:]] or [3]
    xl = pd.read_excel(os.path.join(PROJECT_DIR, CASE_LIST_XLSX))
    # ★ 10 例 文件编号 带前后空格 → 不 strip 会静默丢例
    cases = sorted(xl[CASE_LIST_COL].astype(str).str.strip().tolist())
    print('病例名单: %d 例' % len(cases))

    for k in ks:
        print('=' * 60)
        print(f'K={k} 生境空间结构特征')
        print('=' * 60)
        mask_dir = os.path.join(PROJECT_DIR, MASK_DIR_TEMPLATE.format(k=k))
        if not os.path.isdir(mask_dir):
            print(f'  [跳过] 无目录: {mask_dir}（需先跑上游聚类）')
            continue
        out = os.path.join(PROJECT_DIR, OUT_DIR, f'habitat_spatial_features_k{k}.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)

        rows, done = [], set()
        if os.path.exists(out):
            prev = pd.read_csv(out)
            rows = prev.to_dict('records')
            done = set(zip(prev['patient'], prev['modality']))
            print(f'  已完成 {len(done)} 条, 续跑')

        for modality, _pref in MODALITIES:
            for i, case in enumerate(cases):
                if (case, modality) in done:
                    continue
                try:
                    rec = process_case(case, modality, mask_dir, k)
                except Exception as e:
                    print(f'    [{case}/{modality}] 失败: {e}')
                    continue
                if rec is None:
                    print(f'    [{case}/{modality}] 无有效 mask')
                    continue
                rows.append(rec)
                done.add((case, modality))
                pd.DataFrame(rows).to_csv(out, index=False)     # 每例增量写盘
                if (i + 1) % 50 == 0:
                    print(f'    进度 {i + 1}/{len(cases)}', flush=True)
        D = pd.DataFrame(rows)
        D.to_csv(out, index=False)

        # 覆盖率断言：每例都应有记录（否则就是文件缺失被静默跳过）
        n_miss = len(cases) * len(MODALITIES) - len(D)
        if n_miss:
            have = set(zip(D['patient'].astype(str), D['modality'])) if len(D) else set()
            missing = [c for c in cases
                       if (c, MODALITIES[0][0]) not in have][:8]
            msg = (f'K={k} 缺 {n_miss} 条记录；前几例缺失: {missing}')
            if os.environ.get('IPP_ALLOW_PARTIAL') == '1':
                print('  [警告·部分覆盖] ' + msg, flush=True)
            else:
                raise AssertionError(msg + '（冒烟测试可设 IPP_ALLOW_PARTIAL=1）')

        # 特征数自检：单模态 11K + 3·C(K,2)
        n_feat = len([c for c in D.columns if c not in ('patient', 'modality')])
        expect = 11 * k + 3 * (k * (k - 1) // 2)
        print(f'  完成: {len(D)} 行, {n_feat} 特征列 (期望 {expect}) -> {out}')
        assert n_feat == expect, ('特征数不符！', n_feat, expect)


if __name__ == '__main__':
    main()
