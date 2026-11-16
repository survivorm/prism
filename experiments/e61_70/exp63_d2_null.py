#!/usr/bin/env python
"""E-63 D-2: null-батарея — глобальный негативный контроль метода E-60/E-62.
Три null на КАЖДЫЙ реальный doc (L23, та же длина/норма шага):
 N1 iid-блуждание: U[t+1]=norm(U[t]+δ·ξ⟂), δ=медиана реального шага;
 N2 AR(1)-направления φ∈{0.3,0.6,0.8}: d_t=norm(φ d_{t-1}+√(1-φ²)ξ);
 N3 block-shuffle(8) реального пути.
Батарея (те же σ-бутстрап/победители/clip/strict/Δ(1)) + Δ(k)-кривые.
Гейт (по ревью r5): решение по Δ(1) (null∈[−0.02,0.02]), osc-составу и clip
в СРАВНЕНИИ с null-распределениями, не по абсолютному F1."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, '.')
from exp62_batt_lib import run, K  # noqa: E402

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
L = "23"
z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
names = sorted(meta, key=lambda n: (["wiki", "math", "code", "chat"].index(meta[n]["dom"]),
                                    int(n.rsplit("_", 1)[-1])))
dom_of = {n: meta[n]["dom"] for n in names}


def U_of(nm):
    u = np.asarray(z[f"{nm}_L{L}_u"], dtype=np.float64)
    U = u * np.asarray(z[f"{nm}_L{L}_hn"], dtype=np.float32)[:, None][:, None][:, 0]
    U = u * np.asarray(z[f"{nm}_L{L}_hn"], dtype=np.float64)[:, None]
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    return U


def unit_rows(U):
    return U, np.ones(len(U))


def null_iid(U, seed):
    rng = np.random.default_rng(10000 + seed)
    T, D = U.shape
    delta = float(np.median(np.linalg.norm(np.diff(U, axis=0), axis=1)))
    out = np.empty_like(U)
    x = rng.normal(size=D); x /= np.linalg.norm(x)
    out[0] = x
    for t in range(1, T):
        g = rng.normal(size=D)
        g -= x * float(g @ x)
        x = x + delta * g / np.linalg.norm(g)
        x /= np.linalg.norm(x)
        out[t] = x
    return out


def null_ar(U, seed, phi):
    rng = np.random.default_rng(20000 + seed)
    T, D = U.shape
    delta = float(np.median(np.linalg.norm(np.diff(U, axis=0), axis=1)))
    out = np.empty_like(U); x = U[0].copy()
    d = rng.normal(size=D); d /= np.linalg.norm(d)
    out[0] = x
    for t in range(1, T):
        g = rng.normal(size=D); g /= np.linalg.norm(g)
        d = phi * d + math.sqrt(max(1e-9, 1 - phi * phi)) * g
        d /= np.linalg.norm(d)
        x = x + delta * d
        x -= float(x @ out[0]) * 0   # без привязки
        x /= np.linalg.norm(x)
        out[t] = x
    return out


def null_block(U, seed, blk=8):
    rng = np.random.default_rng(30000 + seed)
    T = len(U)
    nb = T // blk
    idx = rng.permutation(nb)
    out = np.concatenate([U[b * blk:(b + 1) * blk] for b in idx], 0)
    if len(out) < T:
        out = np.vstack([out, U[nb * blk:]])
    return out


import math  # noqa: E402
os.makedirs("../results/e63", exist_ok=True)
res = {}
for tag, gen in (("iid", lambda U, i: null_iid(U, i)),
                 ("ar0.3", lambda U, i: null_ar(U, i, 0.3)),
                 ("ar0.6", lambda U, i: null_ar(U, i, 0.6)),
                 ("ar0.8", lambda U, i: null_ar(U, i, 0.8)),
                 ("blk8", lambda U, i: null_block(U, i))):
    traj = {f"n{i:02d}": (gen(U_of(nm), i), None) for i, nm in enumerate(names)}
    traj = {k: (v[0], np.ones(len(v[0]))) for k, v in traj.items()}
    dom2 = {f"n{i:02d}": dom_of[nm] for i, nm in enumerate(names)}
    r = run(traj, dom2, verbose=False)
    res[tag] = {k: r[k] for k in ("F1", "n_osc", "n", "clip", "n_strict", "dip1")}
    print(f"NULL {tag:>6}: F1={r['F1']} osc={r['n_osc']}/64 clip={r['clip']} "
          f"strict={r['n_strict']} Δ(1)={r['dip1']:+.3f}", flush=True)
    json.dump(res, open("../results/e63/d2_nulls.json", "w"), ensure_ascii=False)
print("REAL L23 (справка): F1=0.913 osc=45/64 clip=19 strict=17 Δ(1)=-0.072")
