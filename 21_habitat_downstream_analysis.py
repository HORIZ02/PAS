# -*- coding: utf-8 -*-
"""P2/P3 下游分析 · 生境空间影像组学
1) sz(层厚)依赖筛查      —— 各向异性体素会使 surface/sphericity/fractal_dim 随 sz 漂移
2) 冗余谱(控制①)        —— 与"常规几何/形状"特征的最大 |ρ|
3) 单变量关联            —— T1(PAS) / T2(预后)
4) 增量检验              —— 产前临床 vs +影像块（折内 top-k，无选择泄漏）
5) 匹配 k 对照(控制②)   —— A形态副本 vs 严格池(B+C+D)，同 k
输出: {RAW}/结果_生境/*.csv
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

import os, sys, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import contextlib, threadpoolctl
threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RAW = PROJECT_ROOT
EXP = os.path.join(RAW, '探索性分析')
OUT = os.path.join(RAW, '结果_生境')
os.makedirs(OUT, exist_ok=True)
CLIN = os.path.join(RAW, r'使用的数据\临床数据\IPP_PAS.xlsx')
PRENATAL = ['Age', 'GDS', 'NCS', 'GDM', 'NDC', 'AFI', 'UABF']
# 常规几何/形状对照块（来自探索性扫描；PyRadiomics 未跑时的代理）
CONV = ['volume_cm3', 'surface_cm2', 'sphericity', 'elongation', 'flatness', 'solidity',
        'max_thick_mm', 'mean_thick_mm', 'thick_sd_mm', 'thickness_cv', 'cc_extent_mm',
        'largest_frac', 'n_slices', 'vol_per_slice_cm3']
TAX = {
    'A_形态非空间': ['n_voxels', 'surface', 'sphericity', 'fractal_dim'],
    'B_拓扑碎片化': ['n_components', 'largest_frac', 'euler'],
    'C_空间分布': ['centroid_dist', 'radial_mean', 'radial_sd', 'edge_dist_mean'],
}
SEEDS = [42] + list(range(19))


def load(K):
    fp = os.path.join(RAW, 'features', f'habitat_spatial_features_k{K}.csv')
    if not os.path.exists(fp):
        return None
    d = pd.read_csv(fp)
    cl = pd.read_excel(CLIN); cl['文件编号'] = cl['文件编号'].astype(str).str.strip()
    d = d.rename(columns={'patient': 'fid'}); d['fid'] = d['fid'].astype(str).str.strip()
    M = cl.merge(d, left_on='文件编号', right_on='fid', how='inner')
    assert len(M) == len(cl) == len(d), (f'K={K} 合并丢例', len(M), len(cl), len(d))
    return M


def cols_by_class(M, K):
    """按 11 特征分类学拆块。必须精确，否则严格池成分错、且会重复计入。"""
    labs = [L for L in ('L', 'ML', 'M', 'MH', 'H') if f't2_{L}_n_voxels' in M.columns]
    out = {}
    for cls, bases in TAX.items():
        cs = []
        for L in labs:
            for b in bases:
                c = f't2_{L}_{b}'
                if c in M.columns:
                    cs.append(c)
        out[cls] = cs
    # D = 生境间关系：每对生境的 interface / interface_norm / centroid_dist
    D = []
    for a, b in itertools.combinations(labs, 2):
        for s in ('interface', 'interface_norm', 'centroid_dist'):
            c = f't2_{a}_{b}_{s}'
            if c in M.columns:
                D.append(c)
    out['D_生境间关系'] = D
    # 自检：无重复、总数与提取表一致
    allf = [c for v in out.values() for c in v]
    assert len(allf) == len(set(allf)), ('分类有重复', len(allf), len(set(allf)))
    n_data = len([c for c in M.columns if c.startswith('t2_')])
    assert len(allf) == n_data, ('分类未覆盖全部特征', len(allf), n_data)
    assert len(allf) == 11 * K + 3 * (K * (K - 1) // 2), (len(allf), K)
    return out


def sz_screen(M, feats):
    sz = M['sz'].astype(float).values if 'sz' in M.columns else None
    if sz is None:
        d = pd.read_csv(os.path.join(EXP, '02_影像形态与信号特征.csv'))
        d['fid'] = d['fid'].astype(str).str.strip()
        sz = d.set_index('fid')['sz'].reindex(M['fid']).values
    rows = []
    for c in feats:
        v = M[c].astype(float).values
        r = spearmanr(v, sz, nan_policy='omit').correlation
        rows.append(dict(特征=c, rho_sz=round(float(r), 3), abs_rho=abs(float(r))))
    return pd.DataFrame(rows).sort_values('abs_rho', ascending=False)


def univariate(M, feats):
    y1 = M['Groups'].astype(int).values; y2 = M['Prognosis'].astype(int).values
    rows = []
    for c in feats:
        v = M[c].astype(float).values
        ok = np.isfinite(v)
        if ok.sum() < 150 or len(np.unique(v[ok])) < 3:
            continue
        vv = np.nan_to_num(v, nan=np.nanmedian(v))
        for tn, y in (('T1_PAS', y1), ('T2_预后', y2)):
            a = roc_auc_score(y, vv)
            p = mannwhitneyu(v[y == 0], v[y == 1], nan_policy='omit').pvalue
            rows.append(dict(任务=tn, 特征=c, AUC=round(a, 3), AUCd=round(max(a, 1 - a), 3),
                             P=('%.3g' % p)))
    return pd.DataFrame(rows)


def oof_topk(Xp, Xi, y, k, seeds=SEEDS):
    """折内 top-k（严格防泄漏）。返回 (mean, sd, 逐种子 AUC 列表)
    同一 seed 下所有臂用同一份折划分 ⇒ 跨臂可做配对比较。"""
    aucs = []
    for sd in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=sd); oof = np.zeros(len(y))
        for tr, te in skf.split(Xp, y):
            sp_ = StandardScaler().fit(Xp[tr]); Zt = sp_.transform(Xp[tr]); Ze = sp_.transform(Xp[te])
            if Xi.shape[1] == 0:
                Xtr, Xte = Zt, Ze
            else:
                a = np.array([abs(roc_auc_score(y[tr], Xi[tr, j]) - .5) if
                              np.isfinite(Xi[tr, j]).all() else -1 for j in range(Xi.shape[1])])
                kk = min(k, Xi.shape[1])
                sel = np.argsort(-a)[:kk]
                si = StandardScaler().fit(Xi[tr][:, sel])
                Xtr = np.hstack([Zt, si.transform(Xi[tr][:, sel])])
                Xte = np.hstack([Ze, si.transform(Xi[te][:, sel])])
            oof[te] = LogisticRegression(C=1.0, max_iter=4000).fit(Xtr, y[tr])\
                                                           .predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)), np.array(aucs)


def paired(a, b, label, out):
    """同划分下的配对比较：a - b 逐种子"""
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    try:
        p = wilcoxon(d).pvalue
    except Exception:
        p = np.nan
    out.append(dict(比较=label, mean_dAUC=round(float(d.mean()), 4),
                    sd=round(float(d.std(ddof=1)), 4),
                    胜出种子数=f'{int((d>0).sum())}/{len(d)}', P=('%.3g' % p) if p == p else ''))
    return d


def main(K):
    M = load(K)
    if M is None:
        print(f'K={K}: 特征文件不存在，跳过'); return
    CLS = cols_by_class(M, K)
    ALL = [c for v in CLS.values() for c in v]
    STRICT = CLS['B_拓扑碎片化'] + CLS['C_空间分布'] + CLS['D_生境间关系']
    FORM = CLS['A_形态非空间']
    print(f'\n{"="*84}\nK={K}   总特征 {len(ALL)}（A {len(FORM)} | B {len(CLS["B_拓扑碎片化"])} | '
          f'C {len(CLS["C_空间分布"])} | D {len(CLS["D_生境间关系"])}）严格池 {len(STRICT)}')
    print(f'n = {len(M)}  各生境平均占比: ' + ', '.join(
        f'{L}={M[f"t2_{L}_n_voxels"].sum()/M[[f"t2_{x}_n_voxels" for x in ("L","M","H") if f"t2_{x}_n_voxels" in M.columns]].sum().sum()*100:.1f}%'
        for L in ('L', 'M', 'H') if f't2_{L}_n_voxels' in M.columns))

    # 1) sz 依赖
    S = sz_screen(M, ALL)
    S.to_csv(os.path.join(OUT, f'K{K}_01_sz依赖筛查.csv'), index=False, encoding='utf-8-sig')
    print(f'\n[1] sz(层厚) 依赖最强的 10 个特征（|ρ| 降序）')
    print(S.head(10).to_string(index=False))
    hi = S[S.abs_rho >= 0.5]
    print(f'    |ρ(sz)| ≥ 0.5 的特征数: {len(hi)}/{len(ALL)}  → 这些特征跨层厚不可比，需标注')

    # 2) 冗余谱
    D2 = pd.read_csv(os.path.join(EXP, '02_影像形态与信号特征.csv'))
    D2['fid'] = D2['fid'].astype(str).str.strip()
    C2 = D2.set_index('fid')[CONV].reindex(M['fid'].values)
    rows = []
    for c in ALL:
        v = M[c].astype(float).values
        mx, arg = 0.0, ''
        for cc in CONV:
            u = C2[cc].values.astype(float)
            r = spearmanr(v, u, nan_policy='omit').correlation
            if np.isfinite(r) and abs(r) > mx:
                mx, arg = abs(r), cc
        rows.append(dict(特征=c, max_abs_rho_常规=round(mx, 3), 最相关常规特征=arg,
                         类别='冗余' if mx >= 0.9 else ('中间' if mx >= 0.7 else '新颖')))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f'K{K}_02_冗余谱.csv'), index=False, encoding='utf-8-sig')
    print(f'\n[2] 冗余谱（与常规几何/形状的 max|ρ|）')
    print(R['类别'].value_counts().to_string())
    print(f'    严格池中新颖( max|ρ|<0.7 )特征数: '
          f'{len(R[(R.类别=="新颖") & (R.特征.isin(STRICT))])}/{len(STRICT)}')

    # 3) 单变量
    U = univariate(M, ALL)
    U.to_csv(os.path.join(OUT, f'K{K}_03_单变量关联.csv'), index=False, encoding='utf-8-sig')
    for tn in ('T1_PAS', 'T2_预后'):
        u = U[U.任务 == tn].sort_values('AUCd', ascending=False).head(10)
        print(f'\n[3] {tn} 生境空间特征 Top10')
        print(u[['特征', 'AUCd', 'P']].to_string(index=False))

    # 4/5) 增量 + 匹配 k
    y1 = M['Groups'].astype(int).values; y2 = M['Prognosis'].astype(int).values
    Xp = np.nan_to_num(M[PRENATAL].apply(pd.to_numeric, errors='coerce').astype(float).values,
                       nan=0.0)
    Xs = np.nan_to_num(M[STRICT].astype(float).values, nan=0.0)
    Xf = np.nan_to_num(M[FORM].astype(float).values, nan=0.0)
    res, pr = [], []
    for tn, y in (('T1_PAS', y1), ('T2_预后', y2)):
        b, bs, bseed = oof_topk(Xp, np.zeros((len(y), 0)), y, 0)
        print(f'\n[4] {tn}  基线(产前临床7)  AUC {b:.4f} ± {bs:.4f}')
        res.append(dict(任务=tn, 臂='基线·产前临床7', k_加=0, AUC=round(b, 4), SD=round(bs, 4), dAUC=0.0))
        for k in (3, 5, 8):
            a, s, aseed = oof_topk(Xp, Xs, y, k)
            c_, s2, cseed = oof_topk(Xp, Xf, y, k)
            res.append(dict(任务=tn, 臂='+严格池(B+C+D)', k_加=k, AUC=round(a, 4), SD=round(s, 4),
                            dAUC=round(a - b, 4)))
            res.append(dict(任务=tn, 臂='+形态副本(A)', k_加=k, AUC=round(c_, 4), SD=round(s2, 4),
                            dAUC=round(c_ - b, 4)))
            print(f'    +严格池 top{k}: {a:.4f} ± {s:.4f} (Δ{a-b:+.4f})   |   '
                  f'+形态副本A top{k}: {c_:.4f} ± {s2:.4f} (Δ{c_-b:+.4f})')
            paired(aseed, bseed, f'{tn} 严格池top{k} vs 基线', pr)
            paired(cseed, bseed, f'{tn} 形态副本A top{k} vs 基线', pr)
            paired(aseed, cseed, f'{tn} 严格池top{k} vs 形态副本A top{k}', pr)
        ao, so, aseed5 = oof_topk(np.zeros((len(y), 1)), Xs, y, 5)
        print(f'    仅严格池 top5: {ao:.4f} ± {so:.4f}')
        res.append(dict(任务=tn, 臂='仅严格池', k_加=5, AUC=round(ao, 4), SD=round(so, 4),
                        dAUC=round(ao - b, 4)))
        paired(aseed5, bseed, f'{tn} 仅严格池top5 vs 基线', pr)
    pd.DataFrame(res).to_csv(os.path.join(OUT, f'K{K}_04_增量与匹配k.csv'),
                             index=False, encoding='utf-8-sig')
    P = pd.DataFrame(pr)
    P.to_csv(os.path.join(OUT, f'K{K}_05_配对比较.csv'), index=False, encoding='utf-8-sig')
    print(f'\n[5] 配对比较（同一划分逐种子，Wilcoxon）')
    print(P.to_string(index=False))


if __name__ == '__main__':
    for K in ([int(x) for x in sys.argv[1:]] or [3]):
        main(K)
