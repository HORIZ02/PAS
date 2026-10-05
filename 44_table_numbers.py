# -*- coding: utf-8 -*-
"""补齐 Table 1 / Table 4 的两个待确认项，并把结果落盘供溯源。
  1) Table 4 分层：各层 n、层内 PAS/IPP 事件数（分层定义 = pd.qcut(volume_cm3, 3)）
  2) Table 1 偏态变量（GDS / IBL / PHH）的 median (IQR)，分组建模与整体
输出: 结果_生境\\13_表1表4补数.csv
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
import importlib.util

import numpy as np
import pandas as pd
import contextlib
import threadpoolctl

threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()

from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = PROJECT_ROOT
OUT = os.path.join(PROJ, '结果_生境')

_spec = importlib.util.spec_from_file_location('m41', os.path.join(HERE, '41_T1定稿分析.py'))
m41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m41)

M = m41.load()
y = M['Groups'].astype(int).values
vol = M['volume_cm3'].astype(float).values

rows = []

# ---------------- 1) 体积三分位分层 ----------------
q = pd.qcut(vol, 3, labels=['Small', 'Intermediate', 'Large'])
lab_cn = {'Small': '小', 'Intermediate': '中', 'Large': '大'}
print('体积三分位（pd.qcut(volume_cm3, 3)）')
for lab in ['Small', 'Intermediate', 'Large']:
    m = np.asarray(q == lab)
    n = int(m.sum())
    n_pas = int(y[m].sum())
    n_ipp = n - n_pas
    print('  %-13s n=%3d  PAS=%3d  IPP=%3d  体积 %.0f–%.0f cm3'
          % (lab, n, n_pas, n_ipp, vol[m].min(), vol[m].max()))
    rows.append(dict(项目='Table4分层', 变量=lab, 数值=f'n={n}', 备注='分母'))
    rows.append(dict(项目='Table4分层', 变量=lab, 数值=f'PAS={n_pas}', 备注='层内事件数(阳性)'))
    rows.append(dict(项目='Table4分层', 变量=lab, 数值=f'IPP={n_ipp}', 备注='层内非事件数'))
    rows.append(dict(项目='Table4分层', 变量=lab,
                     数值='%.0f–%.0f' % (vol[m].min(), vol[m].max()),
                     备注='体积范围 cm3'))
assert sum(np.asarray(q == l).sum() for l in ['Small', 'Intermediate', 'Large']) == len(y)

# 反向验证：分层结果必须复现 Table 4 已登记的数值
print('\n分层复现自检（与 Table 4 对照）')
CHECK = {'flatness': (0.793, 0.702, 0.606), 'solidity': (0.718, 0.612, 0.515),
         'volume_cm3': (0.720, 0.568, 0.661)}
for feat, exp in CHECK.items():
    x = M[feat].astype(float).values
    got = []
    for lab in ['Small', 'Intermediate', 'Large']:
        m = np.asarray(q == lab)
        a = roc_auc_score(y[m], x[m])
        got.append(round(max(a, 1 - a), 3))
    assert got == list(exp), ('分层未复现 Table 4', feat, got, exp)
    print('  %-12s %s  = Table 4 %s' % (feat, got, list(exp)))

# ---------------- 2) 偏态变量的 median (IQR) ----------------
print('\n偏态变量 median (IQR)')
for v in ['GDS', 'IBL', 'PHH']:
    x = pd.to_numeric(M[v], errors='coerce').astype(float)
    for grp, name in [(0, 'IPP'), (1, 'PAS'), (None, 'Overall')]:
        s = x if grp is None else x[y == grp]
        med = s.median()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        print('  %-5s %-8s median %.1f (IQR %.1f–%.1f)' % (v, name, med, q1, q3))
        rows.append(dict(项目='Table1偏态', 变量='%s_%s' % (v, name),
                         数值='%.1f' % med, 备注='median (IQR %.1f-%.1f)' % (q1, q3)))

fp = os.path.join(OUT, '13_表1表4补数.csv')
pd.DataFrame(rows).to_csv(fp, index=False, encoding='utf-8-sig')
print('\nwrote', fp, os.path.getsize(fp))
