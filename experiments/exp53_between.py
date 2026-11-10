#!/usr/bin/env python
"""E-53 v2 (ревью E53_review): tests 1-2 c усилением.
T1: pooled PCA С per-doc центрированием (честная верхняя оценка внутр-доковой
    размерности) + per-doc.
T2: v_t с сохранением выравнивания (None-пропуски пар корректно), lag-кривая
    cos(v_{t+k},v_t) k∈{1,2,3,4,8,16}, распределение cs (квартили, доля<0),
    SNR-фильтр ||v||>0.02 + Спирman(min||v||, cos), нули: shuffle и sign-permute
    (нетривиальный), per-doc значения в JSON.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

z = np.load("/data/prism_kernels/exp47b/exp47b_gen_kinematics.npz", allow_pickle=True)
docs = json.load(open("../results/exp47b_gen_kinematics.json"))["docs"]
rng = np.random.default_rng(53)
T = 96
LAGS = (1, 2, 3, 4, 8, 16)

per_doc_c90, per_doc_in, null_sh, null_sp = [], [], [], []
lag_med = {k: [] for k in LAGS}
cs_all, normed_pairs = [], []
pooled_centered = []
for d in docs:
    k = f"{d['dom']}_{d['si']}"
    u = z[k + "_u"].astype(np.float64)[:T]
    hn = np.asarray(z[k].item()["hnorm"], dtype=float)[:T]
    h = u * hn[:, None]
    U = h / np.linalg.norm(h, axis=1, keepdims=True)
    X = U - U.mean(0)
    pooled_centered.append(X)
    _, S2, _ = np.linalg.svd(X, full_matrices=False)
    ev = S2 ** 2 / (S2 ** 2).sum()
    per_doc_c90.append(int(np.searchsorted(np.cumsum(ev), 0.9) + 1))
    # v с фиксированным индексным совпадением (ревью 3.1)
    V, nv = [], []
    for t in range(len(U) - 1):
        a = U[t + 1] - float(U[t + 1] @ U[t]) * U[t]
        n = float(np.linalg.norm(a))
        V.append(a / n if n > 1e-9 else None)
        nv.append(n)
    def cser(kk):
        return [float(V[t + kk] @ V[t]) for t in range(len(V) - kk)
                if V[t] is not None and V[t + kk] is not None]
    cs = cser(1)
    cs_all.extend(cs)
    for kk in LAGS:
        s = cser(kk)
        if s:
            lag_med[kk].append(float(np.median(s)))
    per_doc_in.append(float(np.median(cs)) if cs else None)
    # SNR: (min нормы пары, cos) — по парам lag1
    for t in range(len(V) - 1):
        if V[t] is not None and V[t + 1] is not None:
            normed_pairs.append((min(nv[t], nv[t + 1]), float(V[t + 1] @ V[t])))
    # нули: shuffle кадров V (порядок), sign-permute
    Vs = np.array([x for x in V if x is not None])
    idx = rng.permutation(len(Vs))
    w = Vs[idx]
    null_sh.append(np.median([float(w[i + 1] @ w[i]) for i in range(len(w) - 1)]))
    sg = rng.choice([-1.0, 1.0], size=len(Vs))
    null_sp.append(np.median([sg[i + 1] * sg[i] * float(Vs[i + 1] @ Vs[i]) for i in range(len(Vs) - 1)]))

Pc = np.vstack(pooled_centered)
_, S2, _ = np.linalg.svd(Pc, full_matrices=False)
ev = S2 ** 2 / (S2 ** 2).sum(); cum = np.cumsum(ev)
cs = np.array(cs_all)
nrm = np.array(normed_pairs)
def spear(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
big = nrm[:, 0] > 0.02
out = {
    "T1_pooled_perdoc_centered": {"dims_50pct": int(np.searchsorted(cum, .5) + 1),
                                  "dims_90pct": int(np.searchsorted(cum, .9) + 1),
                                  "ev1": float(ev[0])},
    "T1_perdoc_dims90": {"med": int(np.median(per_doc_c90)), "p90": int(np.percentile(per_doc_c90, 90))},
    "T2_cs": {"med": float(np.median(cs)), "q25": float(np.quantile(cs, .25)),
              "q75": float(np.quantile(cs, .75)), "frac_neg": float((cs < 0).mean())},
    "T2_lag_curve": {str(k): float(np.median(v)) for k, v in lag_med.items()},
    "T2_nulls": {"shuffle": float(np.median(null_sh)), "sign_permute": float(np.median(null_sp))},
    "T2_snr": {"spearman_minorm_vs_cos": spear(nrm[:, 0], nrm[:, 1]),
               "med_cos_norm_gt_0.02": float(np.median(nrm[big, 1])),
               "frac_norm_gt_0.02": float(big.mean())},
    "per_doc_inertia": per_doc_in,
}
for k in LAGS:
    v = np.array([x for x in lag_med[k] if x is not None])
    idx = rng.integers(0, len(v), (3000, len(v)))
    lo, hi = np.quantile(np.median(v[idx], 1), [.025, .975])
    out["T2_lag_curve"][str(k)] = [out["T2_lag_curve"][str(k)], round(lo, 4), round(hi, 4)]
print(json.dumps(out, indent=1, ensure_ascii=False)[:1500])
json.dump(out, open("../results/exp53_between.json", "w"), indent=1)
