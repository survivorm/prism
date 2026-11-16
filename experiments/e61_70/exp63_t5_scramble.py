#!/usr/bin/env python
"""E-63 T-5: PHASE-SCRAMBLE null — корректный дискриминатор «глобальные часы vs
локальный узор» (blk-shuffle слишком слаб: он не трогает фазу).
Для каждого doc: центрируем путь, SVD→своя 2D-плоскость (p1,p2), координаты z_t
вращаем на угол θ(t), остаток (⊥) не трогаем. θ — случайный φ_b на блок b.
Варианты: hard (θ=const в блоке, форма внутри блока точна, разрывы на границах)
tri   (треугольная рампа 0→φ_b→0, путь непрерывен, сдвиг максимума в центре блока).
Если после scramble R̄strict падает к iid-уровню — фаза была ГЛОБАЛЬНО согласована;
если держится — фазовой глобальности и не было (локальный узор)."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, '.')
from exp62_batt_lib import run  # noqa: E402

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
L = "23"
z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
names = sorted(meta, key=lambda n: (["wiki", "math", "code", "chat"].index(meta[n]["dom"]),
                                    int(n.rsplit("_", 1)[-1])))
dom_of = {n: meta[n]["dom"] for n in names}


def U_of(nm):
    u = np.asarray(z[f"{nm}_L{L}_u"], dtype=np.float64)
    U = u * np.asarray(z[f"{nm}_L{L}_hn"], dtype=np.float64)[:, None]
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    return U


def scramble(U, seed, blk, mode):
    rng = np.random.default_rng(seed)
    T = len(U)
    m = U.mean(0)
    X = U - m
    # своя плоскость пути: SVD на X (top-2)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:2]                        # 2×D
    zc = X @ P.T                      # T×2 координаты в плоскости
    r = X - zc @ P                    # ортогональный остаток
    nb = (T + blk - 1) // blk
    phi = rng.uniform(0, 2 * np.pi, nb)
    th = np.zeros(T)
    for b in range(nb):
        s, e = b * blk, min((b + 1) * blk, T)
        Lb = e - s
        if mode == "hard":
            th[s:e] = phi[b]
        else:  # tri: 0→φ→0 по блоку (непрерывно)
            h = Lb // 2
            th[s:s + h] = phi[b] * np.linspace(0, 1, h, endpoint=False)
            th[s + h:e] = phi[b] * np.linspace(1, 0, e - s - h)
    c, s_ = np.cos(th), np.sin(th)
    z0, z1 = zc[:, 0], zc[:, 1]
    zr = np.stack([c * z0 - s_ * z1, s_ * z0 + c * z1], 1)
    out = m + zr @ P + r
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


res = {}
for blk in (8, 16, 32):
    for mode in ("hard", "tri"):
        tag = f"sc{blk}{mode[0]}"
        traj = {nm: (scramble(U_of(nm), 70000 + hash((nm, tag)) % 10000, blk, mode),
                     np.ones(len(U_of(nm)))) for nm in names}
        r = run(traj, dom_of, verbose=False)
        res[tag] = {k: r[k] for k in ("F1", "n_osc", "n", "clip", "n_strict", "dip1",
                                      "phase", "winners")}
        print(f"SCRAMBLE {tag}: osc={r['n_osc']}/64 clip={r['clip']} strict={r['n_strict']} "
              f"Δ(1)={r['dip1']:+.3f}", flush=True)
        for dm, ph in (r.get("phase") or {}).items():
            print(f"   {dm:<5} ω*={ph[0]} R̄all={ph[1]} R̄strict={ph[2]} n={ph[3]}", flush=True)
        json.dump(res, open("../results/e63/t5_scramble.json", "w"), ensure_ascii=False)
print("real: R̄strict code 0.98 math 0.89 chat 0.97 wiki 0.01 | iid: 0.27–0.66")
