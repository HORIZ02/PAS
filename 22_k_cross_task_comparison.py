# -*- coding: utf-8 -*-
"""K 值跨任务比较：在同一 k、同一折划分下，逐种子配对检验 K=2 vs K=3 vs K=4
回答"K 取 2 是否最佳"——必须按任务分开，并用配对检验而非均值差。
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
from scipy.stats import wilcoxon, spearmanr

RAW = PROJECT_ROOT
SRC = open(os.path.join(RAW, r'方法设计\脚本\21_生境下游分析.py'), encoding='utf-8').read()
exec(SRC.split('def main(')[0])          # 复用 load / cols_by_class / oof_topk / paired / 常量


def main():
    DATA = {}
    for K in (2, 3, 4):
        M = load(K)
        CLS = cols_by_class(M, K)
        DATA[K] = dict(M=M, CLS=CLS,
                       S=CLS['B_拓扑碎片化'] + CLS['C_空间分布'] + CLS['D_生境间关系'],
                       A=CLS['A_形态非空间'])

    # 关键前提：三个 K 的病例顺序必须完全一致，否则折划分不同、配对无效
    order0 = DATA[2]['M']['fid'].astype(str).tolist()
    for K in (2, 3, 4):
        assert DATA[K]['M']['fid'].astype(str).tolist() == order0, f'K={K} 病例顺序不一致'
    print(f'前提校验通过：三个 K 的病例顺序一致（{len(order0)} 例），折划分因此共享\n')

    KS = [2, 3, 4]
    res, pr = [], []
    for tn in ('T1_PAS', 'T2_预后'):
        y = DATA[2]['M']['Groups' if tn == 'T1_PAS' else 'Prognosis'].astype(int).values
        Xp = np.nan_to_num(DATA[2]['M'][PRENATAL].apply(pd.to_numeric, errors='coerce')
                           .astype(float).values, nan=0.0)
        b, bs, bseed = oof_topk(Xp, np.zeros((len(y), 0)), y, 0)
        print(f'{"="*78}\n{tn}   基线(产前临床7) OOF AUC {b:.4f} ± {bs:.4f}')
        for arm, key in (('严格池(B+C+D)', 'S'), ('形态副本(A)', 'A')):
            for k in (3, 5, 8):
                seeds = {}
                print(f'\n  {arm}  @ 加 {k} 个特征')
                for K in KS:
                    Xi = np.nan_to_num(DATA[K]['M'][DATA[K][key]].astype(float).values, nan=0.0)
                    kk = min(k, Xi.shape[1])
                    if kk < k:
                        print(f'    K={K}: 池宽 {Xi.shape[1]} < k={k}，跳过（块宽铁律）')
                        continue
                    a, s, sd = oof_topk(Xp, Xi, y, k)
                    seeds[K] = sd
                    var = '池宽 %d' % Xi.shape[1]
                    print(f'    K={K}: {a:.4f} ± {s:.4f}   (Δ vs 基线 {a-b:+.4f}, {var})')
                    res.append(dict(任务=tn, 臂=arm, k=k, K=K, AUC=round(a, 4),
                                    SD=round(s, 4), dAUC基线=round(a - b, 4), 池宽=Xi.shape[1]))
                for Ka, Kb in ((2, 3), (2, 4), (3, 4)):
                    if Ka in seeds and Kb in seeds:
                        d = seeds[Ka] - seeds[Kb]
                        try:
                            p = wilcoxon(d).pvalue
                        except Exception:
                            p = np.nan
                        lab = f'{tn} {arm}@k={k}  K={Ka} vs K={Kb}'
                        pr.append(dict(比较=lab, mean_dAUC=round(float(d.mean()), 4),
                                       sd=round(float(d.std(ddof=1)), 4),
                                       K大者胜=f'{int((d<0).sum())}/{len(d)}',
                                       P=('%.3g' % p) if p == p else ''))
        # 仅严格池
        seeds = {}
        print(f'\n  仅严格池 @ 加 5')
        for K in KS:
            Xi = np.nan_to_num(DATA[K]['M'][DATA[K]['S']].astype(float).values, nan=0.0)
            a, s, sd = oof_topk(np.zeros((len(y), 1)), Xi, y, 5)
            seeds[K] = sd
            print(f'    K={K}: {a:.4f} ± {s:.4f}   (Δ vs 基线 {a-b:+.4f})')
            res.append(dict(任务=tn, 臂='仅严格池', k=5, K=K, AUC=round(a, 4),
                            SD=round(s, 4), dAUC基线=round(a - b, 4)))
        for Ka, Kb in ((2, 3), (2, 4), (3, 4)):
            d = seeds[Ka] - seeds[Kb]
            try:
                p = wilcoxon(d).pvalue
            except Exception:
                p = np.nan
            pr.append(dict(比较=f'{tn} 仅严格池@5  K={Ka} vs K={Kb}',
                           mean_dAUC=round(float(d.mean()), 4), sd=round(float(d.std(ddof=1)), 4),
                           K大者胜=f'{int((d<0).sum())}/{len(d)}', P=('%.3g' % p) if p == p else ''))

    OUT = os.path.join(RAW, '结果_生境')
    pd.DataFrame(res).to_csv(os.path.join(OUT, '06_K值跨任务比较.csv'), index=False,
                             encoding='utf-8-sig')
    P = pd.DataFrame(pr)
    P.to_csv(os.path.join(OUT, '07_K值配对检验.csv'), index=False, encoding='utf-8-sig')
    print(f'\n{"="*78}\n配对检验汇总（mean_dAUC = K小 − K大；「K大者胜」= K大者更高的种子数/20）')
    print(P.to_string(index=False))


if __name__ == '__main__':
    main()
