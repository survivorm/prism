#!/usr/bin/env python
"""АУДИТ T-5: что реально сделал sc8h.
(1) геометрия: инварианты расстояний внутри блока, Δ(k) real vs scrambled;
(2) где живёт фаза: доля дисперсии в SVD-top2 vs угол с плоскостью фазового фита (ω* домена);
(3) ЧЕСТНЫЙ R̄: спроецировать scrambled на ПЛОСКОСТЬ И ИСТОЧНИК ω, найденные на РЕАЛЬНОМ
    пути (без ре-фита) — если R̄ низкий, фазы перемешаны, и выживание R̄strict в run()
    = эффект ре-фита/селекта (зигзаг воссоздаёт «когерентность»)."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, '.')

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
L = "23"
z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
DOM = {"chat": 0.50, "code": 0.94, "math": 1.28, "wiki": 0.60}
names = {}
for nm in meta:
    names.setdefault(meta[nm]["dom"], [])
    names[meta[nm]["dom"]].append(nm)
for d in names:
    names[d].sort(key=lambda n: int(n.rsplit("_", 1)[-1]))


def U_of(nm):
    u = np.asarray(z[f"{nm}_L{L}_u"], dtype=np.float64)
    U = u * np.asarray(z[f"{nm}_L{L}_hn"], dtype=np.float64)[:, None]
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)


def phase_stats(U, om):
    """Подогнать R·e^{iωt} плоскость (a,b) и фазу: линейная регрессия на [1,cos,sin]."""
    T = len(U)
    t = np.arange(T)
    G = np.stack([np.ones(T), np.cos(om * t), np.sin(om * t)], 1)
    coef = np.linalg.lstsq(G, U, rcond=None)[0]        # 3×D
    Xc = U - G @ coef                                  # остаток
    A, B = coef[1], coef[2]                            # D-векторы плоскости
    # ортонормируем (a,b)
    a = A / max(np.linalg.norm(A), 1e-12)
    b = B - (B @ a) * a
    b = b / max(np.linalg.norm(b), 1e-12)
    ph = np.arctan2(U @ b, U @ a) - om * t
    Rbar = float(np.abs(np.mean(np.exp(1j * ph))))
    amp = float(np.hypot(np.linalg.norm(A), np.linalg.norm(B)) / np.sqrt(2))
    var_plane = float((np.linalg.norm(A) ** 2 + np.linalg.norm(B) ** 2) /
                      ((U - U.mean(0)) ** 2).sum())
    return dict(Rbar=Rbar, amp=amp, var_plane=var_plane,
                plane=np.stack([a, b]))


def sc8h(U, seed, blk=8):
    rng = np.random.default_rng(seed)
    T = len(U)
    m = U.mean(0)
    X = U - m
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = Vt[:2]
    zc = X @ P.T
    r = X - zc @ P
    nb = (T + blk - 1) // blk
    phi = rng.uniform(0, 2 * np.pi, nb)
    th = np.zeros(T)
    for b in range(nb):
        s, e = b * blk, min((b + 1) * blk, T)
        th[s:e] = phi[b]
    c, s_ = np.cos(th), np.sin(th)
    z0, z1 = zc[:, 0], zc[:, 1]
    zr = np.stack([c * z0 - s_ * z1, s_ * z0 + c * z1], 1)
    out = m + zr @ P + r
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def dcurve(U, KM=24):
    Vo = np.stack([U[t + 1] - float(U[t + 1] @ U[t]) * U[t] for t in range(len(U) - 1)])
    ok = np.linalg.norm(Vo, axis=1) > 1e-6
    V = Vo[ok]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i", V[:-k], V[k:])))
                     for k in range(1, KM + 1)])


out = {}
for dm, om in DOM.items():
    rows = []
    for nm in names[dm][:6]:
        U = U_of(nm)
        S = sc8h(U, 70000 + hash((nm, "sc8h")) % 10000, 8, )
        real = phase_stats(U, om)
        scr = phase_stats(S, om)
        # R̄ scrambled В ПЛОСКОСТИ РЕАЛА:
        a, b = real["plane"]
        t = np.arange(len(S))
        phs = np.arctan2(S @ b, S @ a) - om * t
        Rscr_in_realplane = float(np.abs(np.mean(np.exp(1j * phs))))
        # угол плоскостей (2D↔2D): мин сингулярное число [a;b]×[a';b']
        M = np.stack([a, b]) @ np.stack([scr["plane"][0], scr["plane"][1]]).T
        sv = np.linalg.svd(M, compute_uv=False)
        d1r, d1s = dcurve(U)[0], dcurve(S)[0]
        # инвариантность расстояний внутри блоков:
        Bm = np.linalg.norm(U[1:8] - U[0:7], axis=1).round(6)
        Bs = np.linalg.norm(S[1:8] - S[0:7], axis=1).round(6)
        rows.append(dict(nm=nm, R_real=round(real["Rbar"], 3), R_scr=round(scr["Rbar"], 3),
                         R_scr_realplane=round(Rscr_in_realplane, 3),
                         var_plane=round(real["var_plane"], 3),
                         plane_minSV=round(float(sv.min()), 3),
                         d1_real=round(float(d1r), 3), d1_scr=round(float(d1s), 3),
                         d24_real=round(float(dcurve(U)[23]), 3),
                         dist_intra_ok=bool(np.allclose(Bm, Bs))))
    out[dm] = rows
    for r in rows:
        print(f"{r['nm']:<9} R̄ real={r['R_real']} scr(refit)={r['R_scr']} "
              f"scr(плоск.реала)={r['R_scr_realplane']} var⊥пл={r['var_plane']:.2f} "
              f"minSV={r['plane_minSV']:.2f} Δ1 r/s={r['d1_real']:+.2f}/{r['d1_scr']:+.2f} "
              f"Δ24 r={r['d24_real']:+.2f} inv={r['dist_intra_ok']}", flush=True)
json.dump(out, open("../results/e63/t5_audit.json", "w"))
