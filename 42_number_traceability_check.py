# -*- coding: utf-8 -*-
"""数字溯源核对：把初稿里的数字抓出来，逐个在 结果汇总_可溯源.md 里查
查不到的输出为 MISS，需人工判断（可能是年份/百分比/引用编号等非结果数字）
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
import re

RAW = os.path.join(PROJECT_ROOT, '论文撰写')
draft = open(os.path.join(RAW, '初稿_v1.md'), encoding='utf-8').read()
src = open(os.path.join(RAW, '结果汇总_可溯源.md'), encoding='utf-8').read()

# 抓「像结果」的数字：3 位小数、≥2 位小数、或 3 位以上整数
pat = re.compile(r'(?<![\w.])(\d+\.\d{2,4}|\d{3,5}(?![\w.]))')
nums = []
for m in pat.finditer(draft):
    n = m.group(1)
    ctx = draft[max(0, m.start() - 60):m.end() + 40].replace('\n', ' ')
    nums.append((n, ctx))

seen, miss, ok = set(), [], 0
for n, ctx in nums:
    if n in seen:
        continue
    seen.add(n)
    # 允许 0.918 与 0.9176 互查；也允许去掉末尾 0
    cands = {n, n.rstrip('0').rstrip('.')}
    if '.' in n and len(n.split('.')[1]) == 3:
        cands.add(n[:-1])          # 0.933 -> 0.93
    if any(c in src for c in cands):
        ok += 1
    else:
        miss.append((n, ctx))

print(f'初稿中不同数字 {len(seen)} 个；在汇总文件中查到 {ok} 个；查不到 {len(miss)} 个\n')
for n, ctx in miss:
    print(f'  MISS {n:>10s}   …{ctx}…')
