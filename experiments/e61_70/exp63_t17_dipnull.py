#!/usr/bin/env python
"""E-63 T-17: null под ОЦЕНЩИКОМ E-60c. Число −0.136 (V-касательные на сфере:
Vo[t]=U[t+1]−(U[t+1]·U[t])U[t], Δ(k)=median cos(V̂_t,V̂_{t+k})) мерялось НЕ тем
оценщиком, что null-батарея (у той −0.072). Здесь: тот же оценщик E-60c на тех же
47b-путях и на null'ах, построенных из них (iid/blk8/16/32/AR). Если null догоняет
−0.136 — «законы о дипах» (S2 симплектика, S1 renewal, S4 mean-reversion) объясняют
свойство V-оценщика; если нет — у real есть реальный добавок."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

ZP = "/data/prism_kernels/exp47b/exp47b_gen_kinematics.npz"
docs = json.load(open("../results/exp47b_gen_kinematics.json"))["docs"]
z = np.load(ZP, allow_pickle=True)
KM = 24


def U47(d):
    u = z[f"{d['dom']}_{d['si']}_u"].astype(np.float64)
    hn = np.asarray(z[d['dom'] + '_' + str(d['si'])].item()["hnorm"], dtype=np.float64)
    T = min(len(u), len(hn))
    U = u[:T] * hn[:T, None]
    return U / np.linalg.norm(U[:T], axis=1, keepdims=True)


def dcurve(U):
    """оценщик E-60c: Vo[t]=U[t+1]−(U[t+1]·U[t])U[t]... (формула derive: проекция на
    U[t], не U[t+1]) → Δ(k)=median cos на unit-строках."""
    Vo = np.stack([U[t + 1] - float(U[t + 1] @ U[t]) * U[t] for t in range(len(U) - 1)])
    ok = np.linalg.norm(Vo, axis=1) > 1e-6
    V = Vo[ok]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i", V[:-k], V[k:])))
                     for k in range(1, KM + 1)])


def null_iid(U, seed):
    rng = np.random.default_rng(10000 + seed)
    T, D = U.shape
    delta = float(np.median(np.linalg.norm(np.diff(U, axis=0), axis=1)))
    out = np.empty_like(U)
    x = rng.normal(size=D); x /= np.linalg.norm(x); out[0] = x
    for t in range(1, T):
        g = rng.normal(size=D); g -= x * float(g @ x)
        x = x + delta * g / np.linalg.norm(g); x /= np.linalg.norm(x); out[t] = x
    return out


def null_block(U, seed, blk):
    rng = np.random.default_rng(30000 + seed)
    T = len(U); nb = T // blk
    idx = rng.permutation(nb)
    out = np.concatenate([U[b * blk:(b + 1) * blk] for b in idx], 0)
    if len(out) < T:
        out = np.vstack([out, U[nb * blk:]])
    return out


import math
def null_ar(U, seed, phi):
    rng = np.random.default_rng(20000 + seed)
    T, D = U.shape
    delta = float(np.median(np.linalg.norm(np.diff(U, axis=0), axis=1)))
    out = np.empty_like(U); x = U[0].copy()
    d = rng.normal(size=D); d /= np.linalg.norm(d); out[0] = x
    for t in range(1, T):
        g = rng.normal(size=D); g /= np.linalg.norm(g)
        d = phi * d + math.sqrt(max(1e-9, 1 - phi * phi)) * g; d /= np.linalg.norm(d)
        x = x + delta * d; x /= np.linalg.norm(x); out[t] = x
    return out


gens = {"real": lambda U, i: U,
        "iid": null_iid,
        "blk8": lambda U, i: null_block(U, i, 8),
        "blk16": lambda U, i: null_block(U, i, 16),
        "blk32": lambda U, i: null_block(U, i, 32),
        "ar0.8": lambda U, i: null_ar(U, i, 0.8)}
res = {}
for tag, g in gens.items():
    per = {dm: [] for dm in ("chat", "code", "math", "wiki")}
    for i, d in enumerate(docs):
        try:
            U = U47(d)
        except KeyError:
            continue
        per[d["dom"]].append(dcurve(g(U, i)))
    curves = {dm: np.median(np.array(v), 0).tolist() for dm, v in per.items() if len(v)}
    allc = np.array([c for v in per.values() for c in v])
    res[tag] = {"d1_pooled_med": float(np.median(allc[:, 0])),
                "d1_pooled_iqr": [float(np.percentile(allc[:, 0], 25)),
                                  float(np.percentile(allc[:, 0], 75))],
                "per_dom_d1": {dm: round(float(np.median(np.array(v)[:, 0])), 3) for dm, v in per.items() if len(v)},
                "curve_d1_8": [round(float(x), 3) for x in (np.median(allc, 0)[:8])]}
    print(f"{tag:>6}: Δ(1) pooled={res[tag]['d1_pooled_med']:+.3f} "
          f"IQR={np.round(res[tag]['d1_pooled_iqr'],3).tolist()} per-dom={res[tag]['per_dom_d1']}",
          flush=True)
    json.dump(res, open("../results/e63/t17_dipnull.json", "w"))
