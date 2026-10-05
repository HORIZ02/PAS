# -*- coding: utf-8 -*-
"""生境 K 可辨识性探针（子集）：单模态 T2WI，两种体素特征空间
   A) 仅信号强度   B) 强度 + 多尺度局部纹理
   判据：silhouette / GMM-BIC / Gap(1-SE 规则) / ARI 复现性
   规则：最优落在搜索区间端点 ⇒ 该判据不可仲裁
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

import os, warnings, itertools, contextlib
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
# Windows + sklearn1.0.2 + 新版 threadpoolctl：get_config() 返回 None → 崩溃
# 必须在 import sklearn 之前打补丁
import threadpoolctl
threadpoolctl.threadpool_info = lambda *a, **k: []
threadpoolctl.threadpool_limits = lambda *a, **k: contextlib.nullcontext()
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE = PROJECT_ROOT
IMG = BASE + r"\使用的数据\影像数据\images"; MSK = BASE + r"\使用的数据\影像数据\masks"
xl = pd.read_excel(BASE + r"\使用的数据\临床数据\IPP_PAS.xlsx"); xl["fid"] = xl["文件编号"].astype(str).str.strip()
N_CASE, NSUB, KS, NB = 16, 4000, [1, 2, 3, 4, 5, 6], 8
rng = np.random.default_rng(0)

fids = sorted(f[:-7] for f in os.listdir(MSK) if f.endswith(".nii.gz"))
# 分层抽 24 例（PAS/IPP 各半）
sel = (xl[xl.Groups == 1].sample(N_CASE // 2, random_state=0)["fid"].tolist() +
       xl[xl.Groups == 0].sample(N_CASE - N_CASE // 2, random_state=0)["fid"].tolist())
sel = [f for f in sel if f in fids]

def voxel_features(fid):
    im = nib.load(os.path.join(IMG, fid + ".nii.gz")); mk = nib.load(os.path.join(MSK, fid + ".nii.gz"))
    img = np.asanyarray(im.dataobj).astype(np.float32); m = np.asanyarray(mk.dataobj) > 0
    sp = tuple(float(z) for z in im.header.get_zooms()[:3])
    # bbox 裁切
    cs = np.argwhere(m); mn = np.clip(cs.min(0) - 2, 0, None); mx = cs.max(0) + 3
    sl = tuple(slice(int(mn[i]), int(mx[i])) for i in range(3))
    img, m = img[sl], m[sl]
    # 面内重采样到 1.0 mm（层间不动）
    f = sp[0] / 1.0
    img = ndi.zoom(img, (f, f, 1.0), order=1); m = ndi.zoom(m.astype(np.uint8), (f, f, 1.0), order=0) > 0
    # 强度归一化（ROI 内 robust z-score）
    v = img[m]; med, iqr = np.median(v), np.percentile(v, 75) - np.percentile(v, 25)
    img = (img - med) / max(iqr, 1e-6)
    # 多尺度局部纹理
    l3 = ndi.uniform_filter(img, 3); l3s = np.sqrt(np.maximum(ndi.uniform_filter(img * img, 3) - l3 * l3, 0))
    l7 = ndi.uniform_filter(img, 7); l7s = np.sqrt(np.maximum(ndi.uniform_filter(img * img, 7) - l7 * l7, 0))
    idx = np.argwhere(m)
    A = img[m][:, None]                                       # 通道集 A：强度
    B = np.stack([img[m], l3s[m], l7s[m]], 1)                 # 通道集 B：强度+纹理
    return A, B, len(idx)

def kcrit(X, ks, nb=NB, seed=0):
    """返回 silhouette(2..), BIC(1..), gap(1..) 及 ARI 复现性"""
    n = len(X)
    out = {}
    two = X if X.shape[1] == 1 else X
    # silhouette（K>=2）
    sil = {}
    for k in ks:
        if k < 2: continue
        km = KMeans(k, n_init=10, random_state=seed).fit(X)
        sil[k] = float(silhouette_score(X, km.labels_)) if len(set(km.labels_)) > 1 else np.nan
    out["sil"] = sil
    # GMM BIC
    bic = {}
    for k in ks:
        try:
            g = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=seed).fit(X)
            bic[k] = float(g.bic(X))
        except Exception:
            bic[k] = np.nan
    out["bic"] = bic
    # Gap（uniform 参考分布，1-SE 规则）
    def wk(x, k, s):
        return KMeans(k, n_init=10, random_state=s).fit(x).inertia_
    logs = {}
    for k in ks:
        lk = [np.log(wk(X, k, seed))]
        for b in range(nb):
            ref = np.random.default_rng(1000 + b).uniform(X.min(0), X.max(0), size=X.shape)
            lk.append(np.log(wk(ref, k, seed)))
        lk = np.array(lk); logs[k] = (lk.mean(), lk.std(ddof=1))
    gap = {k: logs[k][0] - logs[k][0] for k in ks}   # placeholder
    gap = {k: (logs[k][0] - logs[k][0]) for k in ks}
    # 正确 gap：Gap(k) = E*[log Wk] - log Wk ; 存 E* 与 s_k
    Ek = {k: logs[k][0] for k in ks}; Sk = {k: logs[k][1] for k in ks}
    Wk = {k: np.log(wk(X, k, seed)) for k in ks}
    gap = {k: Ek[k] - Wk[k] for k in ks}
    # 1-SE 规则：最小的 k 满足 Gap(k) >= Gap(k+1) - s(k+1)
    kk = sorted(ks); k1se = kk[-1]
    for i in range(len(kk) - 1):
        k, k2 = kk[i], kk[i + 1]
        if gap[k] >= gap[k2] - Sk[k2]:
            k1se = k; break
    # ARI 复现性（两次不同种子）
    ari = {}
    for k in ks:
        if k < 2: continue
        l1 = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        l2 = KMeans(k, n_init=10, random_state=7).fit_predict(X)
        ari[k] = float(adjusted_rand_score(l1, l2))
    return sil, bic, gap, k1se, ari

rows = []
for i, fid in enumerate(sel):
    A, B, nvox = voxel_features(fid)
    for tag, X in [("A_强度", A), ("B_强度+纹理", B)]:
        sub = X if len(X) <= NSUB else X[rng.choice(len(X), NSUB, replace=False)]
        sil, bic, gap, k1se, ari = kcrit(sub, KS)
        k_sil = min(sil, key=lambda k: sil[k]) if sil else None           # 轮廓越大越好 → max
        k_sil = max(sil, key=lambda k: sil[k]) if sil else None
        k_bic = min(bic, key=lambda k: bic[k]) if bic else None
        kgap = max(gap, key=lambda k: gap[k])
        rows.append(dict(fid=fid, ch=tag, nvox=nvox,
                         k_sil=k_sil, sil={k: round(v, 3) for k, v in sil.items()},
                         k_bic=k_bic, k_gap_argmax=kgap, k_gap_1se=k1se,
                         gap={k: round(v, 3) for k, v in gap.items()}, ari_k3=round(ari.get(3, np.nan), 3)))
    print("  %2d/%d %s nvox=%d" % (i + 1, len(sel), fid, nvox), flush=True)

D = pd.DataFrame(rows)
D.to_csv(BASE + r"\方法设计\K可辨识性探针.csv", index=False, encoding="utf-8-sig")
print("\n=== K 判据汇总（%d 例 × 2 通道集）===" % len(sel))
for tag in ["A_强度", "B_强度+纹理"]:
    s = D[D.ch == tag]
    print("\n[%s]" % tag)
    print("  silhouette 最优 K 分布:", s.k_sil.value_counts().sort_index().to_dict(), " → 众数", s.k_sil.mode().tolist())
    print("  GMM BIC   最优 K 分布:", s.k_bic.value_counts().sort_index().to_dict(), " → 众数", s.k_bic.mode().tolist())
    print("  Gap argmax    K 分布:", s.k_gap_argmax.value_counts().sort_index().to_dict())
    print("  Gap 1-SE      K 分布:", s.k_gap_1se.value_counts().sort_index().to_dict(), " → 众数", s.k_gap_1se.mode().tolist())
    print("  K=3 复现性 ARI: 中位 %.3f (min %.3f)" % (s.ari_k3.median(), s.ari_k3.min()))
    gm = s.gap.apply(lambda d: d[3]).mean()
    print("  平均 Gap: " + ", ".join("K%d=%.3f" % (k, s.gap.apply(lambda d, k=k: d[k]).mean()) for k in KS))
