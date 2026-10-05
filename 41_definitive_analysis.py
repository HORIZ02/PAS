# -*- coding: utf-8 -*-
"""T1 定稿分析：可报告口径
  * 病例级 5 折 × 20 种子 → OOF（点估计）
  * Bootstrap 1000×（患者层重采样）→ 95% CI  ← 口径=OOF 的 Bootstrap，不是折间波动
  * DeLong 检验（AUC 专用）→ 模型间比较
  * 校准（斜率/截距 + Brier）+ DCA 净获益
输出: PROJECT_ROOT\\结果_生境\\09_T1定稿.csv 及逐臂明细
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

import os, itertools
import numpy as np, pandas as pd
import contextlib, threadpoolctl
threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

RAW = PROJECT_ROOT
EXP = os.path.join(RAW, '探索性分析')
OUT = os.path.join(RAW, '结果_生境')
CLIN = os.path.join(RAW, r'使用的数据\临床数据\IPP_PAS.xlsx')
SEEDS = [42] + list(range(19))
PRENATAL = ['Age', 'GDS', 'NCS', 'GDM', 'NDC', 'AFI', 'UABF']
EXPFEAT = ['volume_cm3', 'surface_cm2', 'sphericity', 'elongation', 'flatness', 'solidity',
           'max_thick_mm', 'mean_thick_mm', 'thick_sd_mm', 'thickness_cv', 'cc_extent_mm',
           'largest_frac', 'n_slices', 'vol_per_slice_cm3', 'nvox',
           'roi_skew', 'roi_entropy', 'roi_iqr_rel', 'roi_p5_rel', 'roi_p95_rel', 'roi_range_rel',
           'dark_frac_1', 'dark_frac_15', 'dark_frac_2', 'dark_frac_25', 'dark_ncomp',
           'band_ncomp', 'resid_sd_rel', 'resid_skew']


def load():
    cl = pd.read_excel(CLIN); cl['文件编号'] = cl['文件编号'].astype(str).str.strip()
    E = pd.read_csv(os.path.join(EXP, '02_影像形态与信号特征.csv')); E['fid'] = E['fid'].astype(str).str.strip()
    M = cl.merge(E, left_on='文件编号', right_on='fid', how='inner')
    assert len(M) == len(cl) == len(E), ('合并丢例', len(M), len(cl), len(E))
    return M


def oof_topk(Xp, Xi, y, k, seeds=SEEDS, ret_scores=False):
    aucs, sc42 = [], None
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=sd); oof = np.zeros(len(y))
        for tr, te in skf.split(Xp, y):
            a = StandardScaler().fit(Xp[tr]); Zt = a.transform(Xp[tr]); Ze = a.transform(Xp[te])
            if Xi.shape[1] == 0:
                Xtr, Xte = Zt, Ze
            else:
                s = np.array([abs(roc_auc_score(y[tr], Xi[tr, j]) - .5) for j in range(Xi.shape[1])])
                sel = np.argsort(-s)[:min(k, Xi.shape[1])]
                b = StandardScaler().fit(Xi[tr][:, sel])
                Xtr = np.hstack([Zt, b.transform(Xi[tr][:, sel])])
                Xte = np.hstack([Ze, b.transform(Xi[te][:, sel])])
            oof[te] = LogisticRegression(C=1.0, max_iter=4000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y, oof))
        if sd == 42:
            sc42 = oof.copy()
    out = (float(np.mean(aucs)), float(np.std(aucs, ddof=1)))
    return (out, sc42) if ret_scores else out


# ---------------- DeLong ----------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, p1, p2):
    """Sun & Xu (2014) 的 DeLong 实现。
    ★ AUC 必须用**合并样本**的中间秩 tz 计算（用组内秩 tx 会得到恒为 0 的 AUC——
      因为组内秩和恒等于 m(m+1)/2，本项目实测过：四组比较全部 ΔAUC=0.0000、P=1）。"""
    y = np.asarray(y).astype(int); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    k = 2
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r, pr in enumerate((p1, p2)):
        tx[r] = _midrank(pr[pos]); ty[r] = _midrank(pr[neg]); tz[r] = _midrank(pr)
    aucs = (tz[:, pos].sum(1) / m - (m + 1) / 2) / n          # ★ 用 tz, 不是 tx
    for r, pr in enumerate((p1, p2)):                          # ★ 自检: 与 sklearn 对拍
        ref = roc_auc_score(y, pr)
        assert abs(aucs[r] - ref) < 1e-9, ('DeLong AUC 与 sklearn 不符', aucs[r], ref)
    v01 = (tz[:, pos] - tx) / n
    v10 = 1 - (tz[:, neg] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    L = np.array([[1, -1]])
    return aucs, float(L @ S @ L.T), float(np.sqrt(L @ S @ L.T))


def boot_ci(y, s, n=1000, seed=7):
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], s[b]))
    a = np.array(a)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def calib(y, s):
    lg = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lg, y)
    return float(m.coef_[0][0]), float(m.intercept_[0]), float(brier_score_loss(y, s))


def nb(y, s, pt):
    return s >= pt


def dca(y, s, thresholds):
    out = []
    for pt in thresholds:
        f = nb(y, s, pt)
        tp = int(((f) & (y == 1)).sum()); fp = int(((f) & (y == 0)).sum())
        nn = len(y)
        out.append(dict(阈值=pt, 净获益=tp / nn - fp / nn * pt / (1 - pt)))
    return out


def main():
    M = load(); y = M['Groups'].astype(int).values
    Xp = np.nan_to_num(M[PRENATAL].apply(pd.to_numeric, errors='coerce').astype(float).values, nan=0.0)
    Xa = np.nan_to_num(M[EXPFEAT].astype(float).values, nan=0.0)
    vol = M['volume_cm3'].astype(float).values
    free = [c for c in EXPFEAT if abs(spearmanr(M[c].astype(float).values, vol,
                                                nan_policy='omit').correlation) < 0.5]
    Xf = np.nan_to_num(M[free].astype(float).values, nan=0.0)
    print(f'n={len(y)}  尺寸无关池 {len(free)} 个特征: {free}\n', flush=True)

    ARMS = [('产前临床7', Xp, np.zeros((len(y), 0))),
            ('仅影像 top5', np.zeros((len(y), 1)), Xa),
            ('临床7+影像 top5', Xp, Xa),
            ('临床7+影像 top12', Xp, Xa),
            ('仅影像·尺寸无关 top5', np.zeros((len(y), 1)), Xf),
            ('临床7+尺寸无关 top8', Xp, Xf)]
    res, scores = [], {}
    for name, A, B in ARMS:
        k = 12 if 'top12' in name else (8 if 'top8' in name else 5)
        if A.shape[1] == 0 or '仅影像' in name:
            k = 5
        (mu, sd), s = oof_topk(A, B, y, k, ret_scores=True)
        lo, hi = boot_ci(y, s)
        sl, ic, br = calib(y, s)
        scores[name] = s
        res.append(dict(模型=name, k=k, OOF_AUC=round(mu, 4), 跨种子SD=round(sd, 4),
                        CI95_low=round(lo, 4), CI95_high=round(hi, 4),
                        校准斜率=round(sl, 3), 校准截距=round(ic, 3), Brier=round(br, 4)))
        print(f'  {name:22s} AUC {mu:.4f} ± {sd:.4f}  95%CI {lo:.4f}–{hi:.4f}  '
              f'校准斜率 {sl:.3f} 截距 {ic:.3f}  Brier {br:.4f}', flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, '09_T1定稿_性能.csv'), index=False, encoding='utf-8-sig')

    # DeLong
    cmp_pairs = [('仅影像 top5', '产前临床7'), ('临床7+影像 top5', '产前临床7'),
                 ('临床7+影像 top12', '临床7+影像 top5'),
                 ('仅影像·尺寸无关 top5', '仅影像 top5')]
    rows = []
    for a, b in cmp_pairs:
        aucs, v, se = delong(y, scores[a], scores[b])
        z = (aucs[0] - aucs[1]) / se if se > 0 else 0.0
        p = 2 * (1 - norm.cdf(abs(z)))
        rows.append(dict(比较=f'{a} vs {b}', AUC_a=round(aucs[0], 4), AUC_b=round(aucs[1], 4),
                         ΔAUC=round(aucs[0] - aucs[1], 4), z=round(z, 3),
                         P=('%.3g' % p), 口径='DeLong（主种子 42 的 OOF）'))
        print(f'  DeLong {a} vs {b}: Δ{aucs[0]-aucs[1]:+.4f}  z={z:.3f}  P={p:.3g}', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, '10_T1定稿_DeLong.csv'),
                              index=False, encoding='utf-8-sig')

    # DCA
    th = [i / 100 for i in range(1, 80)]
    D = []
    for name in ('产前临床7', '仅影像 top5', '临床7+影像 top5'):
        for r in dca(y, scores[name], th):
            D.append(dict(模型=name, **r))
    pd.DataFrame(D).to_csv(os.path.join(OUT, '11_T1定稿_DCA.csv'), index=False, encoding='utf-8-sig')
    print('\n→ 09_T1定稿_性能.csv / 10_T1定稿_DeLong.csv / 11_T1定稿_DCA.csv')


if __name__ == '__main__':
    main()
