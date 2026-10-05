# -*- coding: utf-8 -*-
"""方向选择的关键判定：
A) T1 到底有多少是"胎盘大"？—— 用去尺寸特征池（|ρ(体积)|<0.5）单独建模
B) T2 的预测力有多少只是"在重新识别 PAS"？—— Groups 作基线 + PAS/IPP 组内建模
C) 组内（PAS 内 / IPP 内）影像还有没有预测力
输出: 结果_生境/08_方向判定.csv
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

import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import contextlib, threadpoolctl
threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = PROJECT_ROOT
EXP = os.path.join(RAW, '探索性分析')
OUT = os.path.join(RAW, '结果_生境')
SEEDS = [42] + list(range(19))
PRENATAL = ['Age', 'GDS', 'NCS', 'GDM', 'NDC', 'AFI', 'UABF']


def load_all():
    cl = pd.read_excel(os.path.join(RAW, r'使用的数据\临床数据\IPP_PAS.xlsx'))
    cl['文件编号'] = cl['文件编号'].astype(str).str.strip()
    E = pd.read_csv(os.path.join(EXP, '02_影像形态与信号特征.csv'))
    E['fid'] = E['fid'].astype(str).str.strip()
    H = pd.read_csv(os.path.join(RAW, 'features', 'habitat_spatial_features_k2.csv'))
    H['fid'] = H['patient'].astype(str).str.strip()
    H = H[[c for c in H.columns if c.startswith('t2_') or c == 'fid']]
    M = cl.merge(E, left_on='文件编号', right_on='fid', how='inner')
    M = M.merge(H, on='fid', how='inner', suffixes=('', '_h'))
    assert len(M) == len(cl) == len(E) == len(H), ('合并丢例', len(M), len(cl), len(E), len(H))
    return M


EXPFEAT = ['volume_cm3', 'surface_cm2', 'sphericity', 'elongation', 'flatness', 'solidity',
           'max_thick_mm', 'mean_thick_mm', 'thick_sd_mm', 'thickness_cv', 'cc_extent_mm',
           'largest_frac', 'n_slices', 'vol_per_slice_cm3', 'nvox',
           'roi_skew', 'roi_entropy', 'roi_iqr_rel', 'roi_p5_rel', 'roi_p95_rel', 'roi_range_rel',
           'dark_frac_1', 'dark_frac_15', 'dark_frac_2', 'dark_frac_25', 'dark_ncomp',
           'band_ncomp', 'resid_sd_rel', 'resid_skew']
# 尺寸代理（与体积高度共线）
SIZE_PROXY = ['volume_cm3', 'nvox', 'surface_cm2', 'vol_per_slice_cm3', 'max_thick_mm',
              'mean_thick_mm', 'thick_sd_mm', 'cc_extent_mm', 'n_slices', 'largest_frac']


def oof(Xp, Xi, y, k, seeds=SEEDS):
    aucs = []
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=sd); oof = np.zeros(len(y))
        for tr, te in skf.split(Xp, y):
            sp_ = StandardScaler().fit(Xp[tr]); Zt = sp_.transform(Xp[tr]); Ze = sp_.transform(Xp[te])
            if Xi.shape[1] == 0:
                Xtr, Xte = Zt, Ze
            else:
                a = np.array([abs(roc_auc_score(y[tr], Xi[tr, j]) - .5)
                              if np.isfinite(Xi[tr, j]).all() else -1 for j in range(Xi.shape[1])])
                kk = min(k, Xi.shape[1]); sel = np.argsort(-a)[:kk]
                si = StandardScaler().fit(Xi[tr][:, sel])
                Xtr = np.hstack([Zt, si.transform(Xi[tr][:, sel])])
                Xte = np.hstack([Ze, si.transform(Xi[te][:, sel])])
            oof[te] = LogisticRegression(C=1.0, max_iter=4000).fit(Xtr, y[tr])\
                                                           .predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), np.array(aucs)


def main():
    M = load_all()
    print(f'n = {len(M)}\n', flush=True)
    y1 = M['Groups'].astype(int).values
    y2 = M['Prognosis'].astype(int).values
    Xp = np.nan_to_num(M[PRENATAL].apply(pd.to_numeric, errors='coerce').astype(float).values, nan=0.0)
    Xall = np.nan_to_num(M[EXPFEAT].astype(float).values, nan=0.0)
    HAB = [c for c in M.columns if c.startswith('t2_')]
    Xh = np.nan_to_num(M[HAB].astype(float).values, nan=0.0)
    # K=2 的 A/B/C/D 精确拆分（严格池 = B+C+D，不含形态副本 A）
    LAB = ['L', 'H']
    HB = [f't2_{L}_{b}' for L in LAB for b in ('n_components', 'largest_frac', 'euler')]
    HC = [f't2_{L}_{b}' for L in LAB for b in ('centroid_dist', 'radial_mean', 'radial_sd', 'edge_dist_mean')]
    HA = [f't2_{L}_{b}' for L in LAB for b in ('n_voxels', 'surface', 'sphericity', 'fractal_dim')]
    HD = [f't2_L_H_{b}' for b in ('interface', 'interface_norm', 'centroid_dist')]
    assert len(HA) + len(HB) + len(HC) + len(HD) == len(HAB) == 25, (len(HA), len(HB), len(HC), len(HD), len(HAB))
    Hs = [c for c in (HB + HC + HD) if c in M.columns]
    Xhs = np.nan_to_num(M[Hs].astype(float).values, nan=0.0)
    print(f'生境K2 拆分：A {len(HA)} | B {len(HB)} | C {len(HC)} | D {len(HD)} | 严格池 {len(Hs)} | 全池 {len(HAB)}\n',
          flush=True)

    # 尺寸相关性 → 划"去尺寸池"
    vol = M['volume_cm3'].astype(float).values
    rho = {c: abs(float(spearmanr(M[c].astype(float).values, vol, nan_policy='omit').correlation))
           for c in EXPFEAT}
    free = [c for c in EXPFEAT if rho[c] < 0.5]
    sized = [c for c in EXPFEAT if rho[c] >= 0.5]
    print('=== 特征池按"与体积共线"划分 ===')
    print(f'  尺寸相关 (|rho|>=0.5) {len(sized)}: {sized}')
    print(f'  尺寸无关 (|rho|<0.5)  {len(free)}: {free}\n', flush=True)
    Xf = np.nan_to_num(M[free].astype(float).values, nan=0.0)
    Xs = np.nan_to_num(M[sized].astype(float).values, nan=0.0)

    rows = []

    def rec(tag, tn, arm, subset, a, s, extra=''):
        rows.append(dict(判定组=tag, 任务=tn, 子集=subset, 臂=arm, AUC=round(a, 4), SD=round(s, 4), 注=extra))
        print(f'  {subset:12s} {arm:34s} {a:.4f} ± {s:.4f}  {extra}', flush=True)

    # ---------- A) T1 有多少是"胎盘大" ----------
    print('=== A) T1 PAS vs IPP：尺寸成分到底占多少 ===', flush=True)
    b, bs, _ = oof(Xp, np.zeros((len(y1), 0)), y1, 0)
    rec('A', 'T1_PAS', '基线·产前临床7', '全队列', b, bs)
    for nm, X in (('影像·全池', Xall), ('影像·尺寸相关池', Xs), ('影像·尺寸无关池', Xf),
                  ('生境K2·严格池', Xhs), ('生境K2·全池', Xh), ('生境K2·形态A', np.nan_to_num(M[HA].astype(float).values, nan=0.0))):
        for k in (5, 8):
            if X.shape[1] < k: continue
            a, s, _ = oof(Xp, X, y1, k)
            rec('A', 'T1_PAS', f'{nm} top{k}', '全队列', a, s, f'(池宽 {X.shape[1]})')
    for nm, X in (('影像·尺寸无关池', Xf), ('影像·尺寸相关池', Xs), ('影像·全池', Xall)):
        a, s, _ = oof(np.zeros((len(y1), 1)), X, y1, 5)
        rec('A', 'T1_PAS', f'仅{nm} top5', '全队列', a, s)

    # ---------- B) T2 有多少只是"重新识别 PAS" ----------
    print('\n=== B) T2 预后：影像增量是否超越 PAS 状态 ===', flush=True)
    G = M['Groups'].astype(float).values.reshape(-1, 1)
    b2, bs2, b2seed = oof(Xp, np.zeros((len(y2), 0)), y2, 0)
    rec('B', 'T2_预后', '基线·产前临床7', '全队列', b2, bs2)
    g, gs, gseed = oof(G, np.zeros((len(y2), 0)), y2, 0)
    rec('B', 'T2_预后', '仅 Groups(PAS状态)', '全队列', g, gs)
    Xgp = np.hstack([G, Xp])
    bg, bgs, bgseed = oof(Xgp, np.zeros((len(y2), 0)), y2, 0)
    rec('B', 'T2_预后', 'Groups + 产前临床7', '全队列', bg, bgs)
    for nm, X in (('影像·全池', Xall), ('影像·尺寸无关池', Xf), ('生境K2·严格池', Xhs),
                  ('生境K2·全池', Xh)):
        a, s, aseed = oof(Xgp, X, y2, 5)
        rec('B', 'T2_预后', f'Groups+临床7+{nm} top5', '全队列', a, s)
        d = aseed - bgseed
        print(f'      → 配对 vs (Groups+临床7): Δ{d.mean():+.4f}  大者胜 {int((d>0).sum())}/20  '
              f'P={wilcoxon(d).pvalue:.3g}', flush=True)
    a, s, _ = oof(np.zeros((len(y2), 1)), Xf, y2, 5)
    rec('B', 'T2_预后', '仅影像·尺寸无关池 top5', '全队列', a, s)

    # ---------- C) 组内 ----------
    print('\n=== C) 组内（PAS 内 / IPP 内）影像还有没有预测力 ===', flush=True)
    for gname, gv in (('PAS组内', 1), ('IPP组内', 0)):
        m = (y1 == gv)
        yy = y2[m]
        npos = int(yy.sum())
        Xp_, Xa_, Xf_ = Xp[m], Xall[m], Xf[m]
        b, bs, _ = oof(Xp_, np.zeros((len(yy), 0)), yy, 0)
        rec('C', 'T2_预后', '基线·产前临床7', f'{gname}(n={m.sum()}, 不良{npos})', b, bs)
        for k in (3, 5):
            a, s, _ = oof(Xp_, Xa_, yy, k)
            rec('C', 'T2_预后', f'+影像全池 top{k}', f'{gname}(n={m.sum()}, 不良{npos})', a, s)
        a, s, _ = oof(np.zeros((len(yy), 1)), Xf_, yy, 5)
        rec('C', 'T2_预后', '仅影像·尺寸无关池 top5', f'{gname}(n={m.sum()}, 不良{npos})', a, s)
        a, s, _ = oof(np.zeros((len(yy), 1)), Xa_, yy, 5)
        rec('C', 'T2_预后', '仅影像·全池 top5', f'{gname}(n={m.sum()}, 不良{npos})', a, s)

    # ---------- D) 连续结局：组内是否还相关 ----------
    print('\n=== D) 连续结局 log(PHH)：组内相关 ===', flush=True)
    phh = np.log1p(pd.to_numeric(M['PHH'], errors='coerce').astype(float).values)
    for c in ['dark_ncomp', 'resid_sd_rel', 'dark_frac_2', 'roi_skew', 'flatness', 'solidity',
              'volume_cm3', 't2_L_n_components' if 't2_L_n_components' in M.columns else 'nvox']:
        if c not in M.columns: continue
        v = M[c].astype(float).values
        r_all = spearmanr(v, phh, nan_policy='omit').correlation
        r_pas = spearmanr(v[y1 == 1], phh[y1 == 1], nan_policy='omit').correlation
        r_ipp = spearmanr(v[y1 == 0], phh[y1 == 0], nan_policy='omit').correlation
        print(f'  {c:22s} 全队列 {r_all:+.3f} | PAS内 {r_pas:+.3f} | IPP内 {r_ipp:+.3f}', flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, '08_方向判定.csv'), index=False, encoding='utf-8-sig')
    print(f'\n-> {os.path.join(OUT, "08_方向判定.csv")}')


if __name__ == '__main__':
    main()
