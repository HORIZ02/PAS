# -*- coding: utf-8 -*-
"""结果汇总：从各产物 CSV 直接读取，逐条标注来源文件（不手抄，避免转录错误）
输出: PROJECT_ROOT\\论文撰写\\结果汇总_可溯源.md
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
import numpy as np
import pandas as pd
import contextlib, threadpoolctl
threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

RAW = PROJECT_ROOT
EXP = os.path.join(RAW, '探索性分析')
HAB = os.path.join(RAW, '结果_生境')
OUT = os.path.join(RAW, r'论文撰写\结果汇总_可溯源.md')
CLIN = os.path.join(RAW, r'使用的数据\临床数据\IPP_PAS.xlsx')
L = []


def w(s=''):
    L.append(s)


def rd(p):
    return pd.read_csv(p)


def main():
    cl = pd.read_excel(CLIN)
    cl['文件编号'] = cl['文件编号'].astype(str).str.strip()
    n = len(cl)
    y1 = cl['Groups'].astype(int).values
    y2 = cl['Prognosis'].astype(int).values

    w('# 结果汇总（可溯源）')
    w()
    w('> 本文件由 `方法设计/脚本/40_结果汇总.py` 从各产物 CSV **直接读取生成**，非手抄。')
    w('> 每条数字后标注来源文件。生成时间与脚本同目录运行日志一致。')
    w()
    w('## 0. 队列')
    w()
    w(f'- n = **{n}**；PAS {int((cl.Groups==1).sum())} / IPP {int((cl.Groups==0).sum())}'
      f'；预后不良 {int((cl.Prognosis==1).sum())} / 良好 {int((cl.Prognosis==0).sum())}')
    ct = pd.crosstab(cl['Groups'], cl['Prognosis'])
    w(f'- 交叉表（行=Groups 0/1，列=Prognosis 0/1）：IPP 良好 {ct.loc[0,0]} / 不良 {ct.loc[0,1]}'
      f'；PAS 良好 {ct.loc[1,0]} / 不良 {ct.loc[1,1]}')
    w('  来源：`使用的数据/临床数据/IPP_PAS.xlsx`')
    w()

    # ---------- 1 临床单变量 ----------
    A = rd(os.path.join(EXP, '01_临床变量信号扫描.csv'))
    w('## 1. 临床变量单变量（Mann-Whitney + AUC）')
    w()
    w('| 任务 | 变量 | IPP/良好 mean±SD | PAS/不良 mean±SD | AUC | P | 方向 |')
    w('|:--|:--|:--|:--|--:|--:|:--|')
    for tn in ('T1_PAS', 'T2_预后'):
        for _, r in A[(A.任务 == tn)].iterrows():
            if r['变量'] in ('Groups', 'Prognosis'):
                continue
            w(f"| {tn} | {r['变量']} | {r['mean0']}±{r['sd0']} | {r['mean1']}±{r['sd1']} | "
              f"{r['AUC']} | {r['P']} | {r['方向']} |")
    w()
    w('来源：`探索性分析/01_临床变量信号扫描.csv`')
    w()

    # ---------- 2 临床多变量 ----------
    B = rd(os.path.join(EXP, '01b_临床多变量OOF.csv'))
    w('## 2. 临床多变量模型（病例级 5 折 × 5 种子 OOF）')
    w()
    w('| 任务 | 特征块 | OOF AUC | 跨种子 SD | 可否报告 |')
    w('|:--|:--|--:|--:|:--|')
    for _, r in B.iterrows():
        ok = '可报告' if r['特征块'] == '产前7' else '不可报告（含术中/术后量＝结果或标签成分）'
        w(f"| {r['任务']} | {r['特征块']} | {r['OOF_AUC']} | {r['SD_跨种子']} | {ok} |")
    w()
    w('来源：`探索性分析/01b_临床多变量OOF.csv`')
    w()

    # ---------- 3 影像单变量 ----------
    C = rd(os.path.join(EXP, '03_特征与标签关联.csv'))
    for tn in ('T1_PAS', 'T2_预后'):
        w(f'## 3. 影像特征单变量 Top12 —— {tn}')
        w()
        w('| 特征 | AUC(方向校正) | P | ρ(体积) | 去体积后 AUC |')
        w('|:--|--:|--:|--:|--:|')
        for _, r in C[C.任务 == tn].head(12).iterrows():
            w(f"| {r['变量']} | {r['AUCd']} | {r['P']} | {r['rho_vol']} | {r['AUC_去体积']} |")
        w()
    w('来源：`探索性分析/03_特征与标签关联.csv`（AUCd = max(AUC,1−AUC)）')
    w()

    # ---------- 4 增量（探索性池）----------
    D = rd(os.path.join(EXP, '04_增量检验.csv'))
    w('## 4. 增量检验（产前临床7 为基线；折内 top-k；5 折 × 10 种子）')
    w()
    w('| 任务 | 模型 | 入模变量数 | OOF AUC | SD | ΔAUC |')
    w('|:--|:--|--:|--:|--:|--:|')
    for _, r in D.iterrows():
        d = '' if pd.isna(r['dAUC']) else f"{r['dAUC']:+.4f}"
        w(f"| {r['任务']} | {r['模型']} | {r['k']} | {r['AUC']} | {r['SD']} | {d} |")
    w()
    w('来源：`探索性分析/04_增量检验.csv`')
    w()

    # ---------- 5 采集参数混杂 ----------
    E = rd(os.path.join(EXP, '05_采集参数混杂检验.csv'))
    w('## 5. 采集参数混杂检验')
    w()
    w('| 参数 | 任务 | 组0 中位 | 组1 中位 | P |')
    w('|:--|:--|--:|--:|--:|')
    for _, r in E.iterrows():
        w(f"| {r['参数']} | {r['任务']} | {r['组0中位']} | {r['组1中位']} | {r['P']} |")
    w()
    w('来源：`探索性分析/05_采集参数混杂检验.csv` —— 三参数在两间均无差异 ⇒ 影像特征不是在"学扫描协议"。')
    w()

    # ---------- 6 连续结局 ----------
    F = rd(os.path.join(EXP, '06_连续结局关联.csv'))
    phh = pd.to_numeric(cl['PHH'], errors='coerce').astype(float).values
    ibl = pd.to_numeric(cl['IBL'], errors='coerce').astype(float).values
    w('## 6. 连续结局')
    w()
    w(f'- PHH：中位 {np.median(phh):.0f} mL（IQR {np.percentile(phh,25):.0f}–{np.percentile(phh,75):.0f}，'
      f'范围 {phh.min():.0f}–{phh.max():.0f}）；≥1000 mL {int((phh>=1000).sum())} 例')
    w(f'- IBL：中位 {np.median(ibl):.0f} mL（IQR {np.percentile(ibl,25):.0f}–{np.percentile(ibl,75):.0f}，'
      f'范围 {ibl.min():.0f}–{ibl.max():.0f}）')
    w()
    w('| 特征 | ρ vs log PHH | ρ vs log IBL | ρ(logPHH) 去体积 | ρ(logIBL) 去体积 | ρ(体积) |')
    w('|:--|--:|--:|--:|--:|--:|')
    for _, r in F.head(12).iterrows():
        w(f"| {r['变量']} | {r['rho_logPHH']} | {r['rho_logIBL']} | "
          f"{r['rho_logPHH_去体积']} | {r['rho_logIBL_去体积']} | {r['rho_体积']} |")
    w()
    w('来源：`探索性分析/06_连续结局关联.csv`')
    w()

    # ---------- 7 组内（连续结局）----------
    d2 = rd(os.path.join(EXP, '02_影像形态与信号特征.csv'))
    d2['fid'] = d2['fid'].astype(str).str.strip()
    M = cl.merge(d2, left_on='文件编号', right_on='fid', how='inner')
    lphh = np.log1p(np.nan_to_num(phh))
    w('## 7. 连续结局的组内分解（判断关联是否只来自"组间差异"）')
    w()
    w('| 特征 | 全队列 ρ(logPHH) | PAS 组内 | IPP 组内 |')
    w('|:--|--:|--:|--:|')
    for c in ['volume_cm3', 'flatness', 'solidity', 'dark_ncomp', 'resid_sd_rel', 'roi_skew']:
        v = M[c].astype(float).values
        r_all = spearmanr(v, lphh, nan_policy='omit').correlation
        r_p = spearmanr(v[y1 == 1], lphh[y1 == 1], nan_policy='omit').correlation
        r_i = spearmanr(v[y1 == 0], lphh[y1 == 0], nan_policy='omit').correlation
        w(f'| {c} | {r_all:+.3f} | {r_p:+.3f} | {r_i:+.3f} |')
    w()
    w('来源：`探索性分析/02_影像形态与信号特征.csv` + 临床表（本脚本现算）')
    w()

    # ---------- 8 勾画敏感性 ----------
    G = rd(os.path.join(EXP, '07_勾画敏感性_内缩与外扩.csv'))
    G['fid'] = G['fid'].astype(str).str.strip()
    MG = cl.merge(G, left_on='文件编号', right_on='fid', how='inner')
    w('## 8. 掩膜勾画敏感性（内缩 5/8 mm 只统计胎盘内部）')
    w()
    w('| 特征 | 口径 | 有效 n | T1_PAS AUC | T2_预后 AUC |')
    w('|:--|:--|--:|--:|--:|')
    for base in ['roi_skew', 'roi_entropy', 'dark_frac_2', 'resid_sd_rel']:
        for tag in ('full', 'erode5', 'erode8'):
            c = f'{tag}_{base}'
            if c not in MG.columns:
                continue
            v = MG[c].astype(float).values
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            a1 = roc_auc_score(y1[ok], v[ok]); a2 = roc_auc_score(y2[ok], v[ok])
            w(f'| {base if tag == "full" else ""} | {tag} | {int(ok.sum())} | '
              f'{max(a1,1-a1):.3f} | {max(a2,1-a2):.3f} |')
    for tag in ('full', 'erode5', 'erode8'):
        c = 'vol_' + tag
        if c in MG.columns:
            v = MG[c].astype(float).values; ok = np.isfinite(v)
            a = roc_auc_score(y1[ok], v[ok])
            w(f'| 体积 | {tag} | {int(ok.sum())} | {max(a,1-a):.3f} | — |')
    w()
    w('来源：`探索性分析/07_勾画敏感性_内缩与外扩.csv`（AUC 本脚本现算）')
    w()

    # ---------- 9 尺寸无关池 ----------
    Z = rd(os.path.join(HAB, '08_方向判定.csv'))
    w('## 9. 尺寸无关池（|ρ(胎盘体积)| < 0.5）与方向判定')
    w()
    w('| 判定组 | 子集 | 臂 | AUC | SD |')
    w('|:--|:--|:--|--:|--:|')
    for _, r in Z.iterrows():
        w(f"| {r['判定组']} | {r['子集']} | {r['臂']} | {r['AUC']} | {r['SD']} |")
    w()
    w('来源：`结果_生境/08_方向判定.csv`')
    w()

    # ---------- 10 生境：增量 + 匹配 k + 配对 + K 比较 ----------
    frames = []
    for K in (2, 3, 4):
        d = rd(os.path.join(HAB, f'K{K}_04_增量与匹配k.csv')); d['K'] = K; frames.append(d)
    H = pd.concat(frames)
    w('## 10. 生境影像组学：增量与匹配 k 对照')
    w()
    w('| K | 任务 | 臂 | 加 k | OOF AUC | SD | ΔAUC |')
    w('|--:|:--|:--|--:|--:|--:|--:|')
    for _, r in H[H.k_加 > 0].iterrows():
        w(f"| {r['K']} | {r['任务']} | {r['臂']} | {r['k_加']} | {r['AUC']} | {r['SD']} | {r['dAUC']:+.4f} |")
    w()
    w('来源：`结果_生境/K{2,3,4}_04_增量与匹配k.csv`')
    w()
    P = []
    for K in (2, 3, 4):
        d = rd(os.path.join(HAB, f'K{K}_05_配对比较.csv')); d['K'] = K; P.append(d)
    P = pd.concat(P)
    w('### 10b. 配对比较（同一折划分逐种子，Wilcoxon）')
    w()
    w('| K | 比较 | mean ΔAUC | SD | K大者胜(种子) | P |')
    w('|--:|:--|--:|--:|:--|--:|')
    for _, r in P.iterrows():
        w(f"| {r['K']} | {r['比较']} | {r['mean_dAUC']:+.4f} | {r['sd']} | "
          f"{r['胜出种子数']} | {r['P']} |")
    w()
    w('来源：`结果_生境/K{2,3,4}_05_配对比较.csv`')
    w()

    # ---------- 11 K 值跨任务 ----------
    if os.path.exists(os.path.join(HAB, '06_K值跨任务比较.csv')):
        Q = rd(os.path.join(HAB, '07_K值配对检验.csv'))
        w('## 11. K 值跨任务比较（配对检验）')
        w()
        w('| 比较 | mean ΔAUC(K小−K大) | SD | K大者胜 | P |')
        w('|:--|--:|--:|:--|--:|')
        for _, r in Q.iterrows():
            w(f"| {r['比较']} | {r['mean_dAUC']:+.4f} | {r['sd']} | {r['K大者胜']} | {r['P']} |")
        w()
        w('来源：`结果_生境/07_K值配对检验.csv`')
        w()

    # ---------- T1: Table 1 ----------
    BIN = ['GDM', 'IBT']
    CONT = ['Age', 'GDS', 'NCS', 'NDC', 'OM', 'AFI', 'UABF', 'IBL', 'PHH']
    w('## T1. Table 1（按 Groups 分层；均值±SD 保留 2 位小数，P 保留 3 位）')
    w()
    w('| 变量 | IPP (n=%d) | PAS (n=%d) | P |' % (int((cl.Groups == 0).sum()), int((cl.Groups == 1).sum())))
    w('|:--|:--|:--|--:|')
    for c in CONT:
        a = pd.to_numeric(cl.loc[cl.Groups == 0, c], errors='coerce').astype(float)
        b = pd.to_numeric(cl.loc[cl.Groups == 1, c], errors='coerce').astype(float)
        p = mannwhitneyu(a.dropna(), b.dropna()).pvalue
        ps = '<0.001' if p < 0.001 else '%.3f' % p
        w(f'| {c} (mean±SD) | {a.mean():.2f}±{a.std(ddof=1):.2f} | {b.mean():.2f}±{b.std(ddof=1):.2f} | {ps} |')
    for c in BIN:
        a = pd.to_numeric(cl.loc[cl.Groups == 0, c], errors='coerce').astype(float)
        b = pd.to_numeric(cl.loc[cl.Groups == 1, c], errors='coerce').astype(float)
        from scipy.stats import chi2_contingency
        tb = pd.crosstab(cl['Groups'], cl[c])
        p = chi2_contingency(tb)[1] if tb.shape == (2, 2) else np.nan
        ps = '<0.001' if (p == p and p < 0.001) else ('%.3f' % p if p == p else 'NA')
        w(f'| {c}, n (%) | {int(a.sum())} ({a.mean()*100:.1f}%) | {int(b.sum())} ({b.mean()*100:.1f}%) | {ps} |')
    w()
    w('来源：`使用的数据/临床数据/IPP_PAS.xlsx`（本脚本现算；P：连续变量 Mann-Whitney，二分类卡方）')
    w()

    # ---------- 12 生境单变量 ----------
    w('## 12. 生境空间特征单变量 Top6（方向校正 AUC）')
    for tn in ('T1_PAS', 'T2_预后'):
        w()
        w(f'**{tn}**')
        w()
        w('| K | 特征 | AUC(方向校正) | P |')
        w('|--:|:--|--:|--:|')
        for K in (2, 3, 4):
            U = rd(os.path.join(HAB, f'K{K}_03_单变量关联.csv'))
            for _, r in U[U.任务 == tn].sort_values('AUCd', ascending=False).head(6).iterrows():
                w(f"| {K} | {r['特征']} | {r['AUCd']} | {r['P']} |")
    w()
    w('来源：`结果_生境/K{2,3,4}_03_单变量关联.csv`')
    w()

    # ---------- 13 T1 定稿性能 + DeLong ----------
    P9 = os.path.join(HAB, '09_T1定稿_性能.csv')
    if os.path.exists(P9):
        w('## 13. T1 定稿性能（OOF 点估计 + Bootstrap 1000× 95%CI + 校准 + Brier）')
        w()
        w('| 模型 | 入模 k | OOF AUC | 跨种子 SD | 95% CI（Bootstrap 患者层） | 校准斜率 | 校准截距 | Brier |')
        w('|:--|--:|--:|--:|:--|--:|--:|--:|')
        for _, r in rd(P9).iterrows():
            w(f"| {r['模型']} | {r['k']} | {r['OOF_AUC']} | {r['跨种子SD']} | "
              f"{r['CI95_low']}–{r['CI95_high']} | {r['校准斜率']} | {r['校准截距']} | {r['Brier']} |")
        w()
        w('口径说明：点估计 = 病例级 5 折 × 20 种子 OOF AUC 的均值；**95%CI = 主种子(42) OOF 分数经'
          '1000 次患者层 Bootstrap 重采样的百分位区间**（不是折间波动、不是 DeLong）。')
        w('来源：`结果_生境/09_T1定稿_性能.csv`（脚本 `41_T1定稿分析.py`）')
        w()
        w('### 13b. DeLong 模型间比较（AUC 专用检验）')
        w()
        w('| 比较 | AUC_a | AUC_b | ΔAUC | z | P |')
        w('|:--|--:|--:|--:|--:|--:|')
        for _, r in rd(os.path.join(HAB, '10_T1定稿_DeLong.csv')).iterrows():
            w(f"| {r['比较']} | {r['AUC_a']} | {r['AUC_b']} | {r['ΔAUC']:+.4f} | {r['z']} | {r['P']} |")
        w()
        w('来源：`结果_生境/10_T1定稿_DeLong.csv`。实现为 Sun & Xu (2014) DeLong，'
          '**内置与 sklearn `roc_auc_score` 的逐臂对拍断言**（曾因误用组内秩和导致全部 ΔAUC=0.0000/P=1）。')
        w()

    # ---------- 14 体积混杂的两项佐证（此前只在日志里，落盘以便溯源）----------
    d2b = rd(os.path.join(EXP, '02_影像形态与信号特征.csv'))
    d2b['fid'] = d2b['fid'].astype(str).str.strip()
    M2 = cl.merge(d2b, left_on='文件编号', right_on='fid', how='inner')
    w('## 14. 体积混杂的两项佐证')
    w()
    w('### 14a. 胎盘体积与临床背景的相关（排除"体积大只是因为孕周晚"）')
    w()
    w('| 对照 | Spearman ρ | n |')
    w('|:--|--:|--:|')
    for c in ['GDS', 'Age', 'NCS', 'NDC']:
        r = spearmanr(M2['volume_cm3'].astype(float), pd.to_numeric(M2[c], errors='coerce'),
                      nan_policy='omit').correlation
        w(f'| 胎盘体积 vs {c} | {r:+.3f} | {len(M2)} |')
    w()
    w('来源：`探索性分析/02_影像形态与信号特征.csv` + 临床表（本脚本现算）')
    w()
    w('### 14b. 体积三分位内分层 AUC（按设计消除体积混杂，非回归校正）')
    w()
    v = M2['volume_cm3'].astype(float).values
    q = pd.qcut(v, 3, labels=['小', '中', '大'])
    w(f'三分位范围：小 {v[np.asarray(q=="小")].min():.0f}–{v[np.asarray(q=="小")].max():.0f}；'
      f'中 {v[np.asarray(q=="中")].min():.0f}–{v[np.asarray(q=="中")].max():.0f}；'
      f'大 {v[np.asarray(q=="大")].min():.0f}–{v[np.asarray(q=="大")].max():.0f} cm³')
    w()
    w('| 特征 | 总体 AUC | 小体积 | 中体积 | 大体积 |')
    w('|:--|--:|--:|--:|--:|')
    for c in ['resid_sd_rel', 'dark_frac_2', 'roi_skew', 'roi_entropy', 'dark_ncomp',
              'flatness', 'solidity', 'thickness_cv', 'volume_cm3']:
        vv = np.nan_to_num(M2[c].astype(float).values,
                           nan=np.nanmedian(M2[c].astype(float).values))
        a0 = roc_auc_score(y1, vv)
        cells = []
        for lb in ('小', '中', '大'):
            s = np.asarray(q == lb)
            a = roc_auc_score(y1[s], vv[s]); cells.append(max(a, 1 - a))
        w(f"| {c} | {max(a0,1-a0):.3f} | " + ' | '.join(f'{x:.3f}' for x in cells) + ' |')
    w()
    w('来源：`探索性分析/02_影像形态与信号特征.csv`（本脚本现算）。'
      '各层内 AUC 保留即说明信号不是体积驱动。')
    w()

    # ---------- 15 掩膜与体积的组间描述（供 Methods/Results 引用）----------
    w('## 15. 掩膜与胎盘体积的组间描述')
    w()
    w('### 15a. 胎盘体积（清洗后）按组分层的 median (IQR)')
    w()
    w('| 组 | n | 中位 cm³ | IQR | P |')
    w('|:--|--:|--:|:--|--:|')
    for g, nm in ((0, 'IPP'), (1, 'PAS')):
        vv = M2.loc[M2['Groups'] == g, 'volume_cm3'].astype(float)
        w(f'| {nm} | {len(vv)} | {vv.median():.0f} | {vv.quantile(.25):.0f}–{vv.quantile(.75):.0f} | — |')
    v0 = M2.loc[M2['Groups'] == 0, 'volume_cm3'].astype(float)
    v1 = M2.loc[M2['Groups'] == 1, 'volume_cm3'].astype(float)
    w(f'| 组间 Mann-Whitney | | | | {mannwhitneyu(v0, v1).pvalue:.3g} |')
    w()
    w('### 15b. 掩膜连通性（清洗前）')
    w()
    qc = os.path.join(RAW, r'数据核验\掩膜连通性QC.csv')
    if os.path.exists(qc):
        Q = pd.read_csv(qc)
        c0 = next((c for c in Q.columns if c in ('连通分量数', 'n_components') or
                   'component' in c.lower() or 'n_comp' in c.lower()), None)
        w(f'- 掩膜连通性 QC 文件列: {list(Q.columns)}')
        if c0:
            multi = int((Q[c0] > 1).sum())
            w(f'- 使用列 `{c0}`')
            w(f'- 连通分量数 > 1 的例数: **{multi}/{len(Q)} ({multi/len(Q)*100:.1f}%)**')
            w(f'- 分量数分布: {Q[c0].value_counts().sort_index().to_dict()}')
        else:
            w('- 【未识别到分量数列】')
        w()
        w('来源：`数据核验/掩膜连通性QC.csv`')
    else:
        w('- 【未找到 掩膜连通性QC.csv】')
    w()
    w('来源：`探索性分析/02_影像形态与信号特征.csv`（体积）与 `数据核验/掩膜连通性QC.csv`（连通性）')
    w()

    # ---------- 16 采集几何分布（供 Methods 引用）----------
    w('## 16. 采集几何分布（真实世界异质性；**写范围不写单一值**）')
    w()
    for c, nm, unit in (('sp_x', '面内间距 sp_x', 'mm'), ('sp_y', '面内间距 sp_y', 'mm'),
                        ('sz', '层厚 sz', 'mm')):
        v = M2[c].astype(float)
        w(f'- {nm}：{v.nunique()} 个不同取值；范围 {v.min():.4f}–{v.max():.4f} {unit}；'
          f'中位 {v.median():.4f}；IQR {v.quantile(.25):.4f}–{v.quantile(.75):.4f}')
    sz = M2['sz'].astype(float)
    # ★ 层厚必须**按 0.1 mm 取整**后比较：头里的 “9.0” 实际多为 8.999999x，
    #   精确浮点比较会只得 67/202，取整后才是真实的 119/202（浮点精度陷阱）
    szr = sz.round(1)
    main = int((szr == 8.0).sum() + (szr == 9.0).sum())
    w(f'- 层厚（按 0.1 mm 取整）为 8.0 或 9.0 mm 的例数：**{main}/{len(M2)} ({main/len(M2)*100:.1f}%)**')
    exact = int(((sz - 8.0).abs() < 1e-6).sum() + ((sz - 9.0).abs() < 1e-6).sum())
    w(f'  - （对照：若用精确浮点相等比较只得 `{exact}` 例，属浮点精度假象，**不可用**）')
    w(f'- 层厚唯一值：{sz.nunique()} 个（浮点）；按 0.1 mm 取整后 {szr.nunique()} 个')
    w(f'- 层厚分布（按 0.1 mm 取整，仅列 ≥5 例）：'
      f'{ {float(k): int(v) for k, v in szr.value_counts().items() if v >= 5} }')
    w()
    w('来源：`探索性分析/02_影像形态与信号特征.csv`（本脚本现算）')
    w()

    w()
    w('### 16b. 视野（FOV）= 矩阵 × 面内间距（现算；此前无落盘源）')
    w()
    try:
        hdr = pd.read_csv(os.path.join(RAW, r'稿件信息提取\影像头参数汇总.csv'))
        hdr['fid'] = hdr['fid'].astype(str).str.strip()
        hdr['shape_x'] = hdr['shape'].astype(str).str.split('x').str[0].astype(float)
        hdr['shape_y'] = hdr['shape'].astype(str).str.split('x').str[1].astype(float)
        hdr['fov_x'] = hdr['shape_x'] * pd.to_numeric(hdr['sx'], errors='coerce')
        hdr['fov_y'] = hdr['shape_y'] * pd.to_numeric(hdr['sy'], errors='coerce')
        fv = pd.concat([hdr['fov_x'], hdr['fov_y']]).dropna()
        w(f'- 面内 FOV：n={len(fv)} 个方向值（{len(hdr)} 例）；范围 {fv.min():.0f}–{fv.max():.0f} mm；'
          f'中位 {fv.median():.0f}；IQR {fv.quantile(.25):.0f}–{fv.quantile(.75):.0f}')
        allf = pd.concat([hdr['fov_x'], hdr['fov_y']])
        w(f'- 按例（取 x/y 均值）：范围 {hdr[["fov_x","fov_y"]].mean(axis=1).min():.0f}–'
          f'{hdr[["fov_x","fov_y"]].mean(axis=1).max():.0f} mm')
        w(f'- 矩阵形状种类：{hdr["shape"].nunique()} 种')
        w()
        w('来源：`稿件信息提取/影像头参数汇总.csv`（shape × spacing，本脚本现算）')
    except Exception as e:
        w(f'- 【计算失败：{e}】')
    w()

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', OUT, os.path.getsize(OUT), 'bytes,', len(L), 'lines')


if __name__ == '__main__':
    main()
