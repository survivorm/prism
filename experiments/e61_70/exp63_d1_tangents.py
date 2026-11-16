#!/usr/bin/env python
"""E-63 D-1: кросс-слойная геометрия БЕЗ residual-ловушки.
(a) cos(v_L(t), v_L'(t)) по t; (b) corr по k кривых Δ_L vs Δ_L' per-doc;
(c) контроль чужим doc; (d) preferred direction cos(v_L(t), mean_t v_L).
v — тангенсы curve_u; общий сдвиг направлений НЕ путаем: (a) за минусом (d)."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, '.')
from exp62_batt_lib import curve_u, K  # noqa: E402

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
names = sorted(meta, key=lambda n: (["wiki", "math", "code", "chat"].index(meta[n]["dom"]),
                                    int(n.rsplit("_", 1)[-1])))
LAY = [1, 5, 11, 17, 23]
Vs, Dks = {}, {}
for nm in names:
    for L in LAY:
        V = curve_u(z[f"{nm}_L{L}_u"], z[f"{nm}_L{L}_hn"])
        Vs[(nm, L)] = V
        Dks[(nm, L)] = np.array([float(np.median(np.einsum("ij,ij->i", V[:-k], V[k:])))
                                 for k in K])
res = {"a_preferred": {}, "a_pair": {}, "b_profile_corr": {}, "c_foreign": {}}
for L in LAY:
    res["a_preferred"][f"L{L}"] = {}
    for nm in names:
        V = Vs[(nm, L)]
        m = V.mean(0); m /= max(np.linalg.norm(m), 1e-12)
        res["a_preferred"][f"L{L}"][nm[:4]] = float(np.mean(V @ m))
for La, Lb in ((1, 23), (5, 23), (11, 23), (17, 23), (11, 17)):
    ka = f"L{La}_L{Lb}"
    a, dp, cc, cf = [], [], [], []
    for i, nm in enumerate(names):
        A, B = Vs[(nm, La)], Vs[(nm, Lb)]
        n_ = min(len(A), len(B))
        A, B = A[:n_], B[:n_]
        mA = A.mean(0); mA /= max(np.linalg.norm(mA), 1e-12)
        mB = B.mean(0); mB /= max(np.linalg.norm(mB), 1e-12)
        cosAB = A @ B.T
        np.fill_diagonal(cosAB[np.arange(0, min(n_, n_)), np.arange(0, min(n_, n_))] if False else np.eye(n_, dtype=bool), np.nan)
        a.append(float(np.nanmean(cosAB)))
        dp.append(float((A @ mA) @ (B @ mB).T / max(n_, 1)))  # произведение проекций
        da, db = Dks[(nm, La)], Dks[(nm, Lb)]
        cc.append(float(np.corrcoef(da, db)[0, 1]))
        nb = names[(i + 7) % len(names)]
        db2 = Dks[(nb, Lb)]
        cf.append(float(np.corrcoef(da, db2)[0, 1]))
    res["a_pair"][ka] = round(float(np.mean(a)), 3)
    res["d_preferred_prod"][ka] = round(float(np.mean(dp)), 3) if "d_preferred_prod" in res else None
    res.setdefault("d_preferred_prod", {})[ka] = round(float(np.mean(dp)), 3)
    res["b_profile_corr"][ka] = round(float(np.mean(cc)), 3)
    res["c_foreign"][ka] = round(float(np.mean(cf)), 3)
    print(f"{ka}: cos(v,v')={res['a_pair'][ka]:+.3f} предпр.= {res['d_preferred_prod'][ka]:+.3f} "
          f"corr(Δk)={res['b_profile_corr'][ka]:+.3f} чужой={res['c_foreign'][ka]:+.3f}", flush=True)
os.makedirs("../results/e63", exist_ok=True)
json.dump(res, open("../results/e63/d1_tangent.json", "w"))
