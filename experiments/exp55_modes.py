#!/usr/bin/env python
"""E-55: спектр мод Δ-кривой. Стадия -1 (контроли C1-C7) + стадия 0 (real 14 лагов).
ТЗ: docs/E55.md (аддендум R1-R3). САМОАУДИТ-инварианты: одна и та же функция Δ-оценщика для
synthetic и real (AGENTS.md №1); генераторы = собственные моды, не внешняя накачка (R1);
фазовый ток — прямая статистика, порог из C2/C3 (R2)."""
import os, sys, json, math, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E

D, T = E.D, E.T
LAGS14 = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32)
RNG = np.random.default_rng(0)

# ---------- генератор мод (R1: собственные, не накачка) ----------

def derive2(U):
    return E.derive(dict(name="s", dom="syn", si=0, U=U))

def simulate_modes(n_docs, seed, modes, floor=0.004, sig=0.018, jitter_phase=False, amp_scale=None):
    """modes:
    ('r',   r, sd, dir)                — вещественная AR(1) вдоль dir
    ('c',   r, omega, sd, w)           — AR(2)-комплекс СКАЛЯРО вдоль w (собств. осциллятор)
    ('rot', r, omega, sd, (u1,u2))     — вращение rR(ω) в 2-plane
    ('x',  omega, amp, w)              — ВНЕШНЯЯ накачка; ОДНА 'x' на контроль (ph0 общий на doc — ревью E55п4)
    amp_scale: список коэф. длины n_docs — множится sd/amp ОСЦИЛЛЯТОРНЫХ мод ('c'/'rot'/'x');
    'r' (β-канал) НЕ масштабируется (ревью v12п1: иначе amp_scale убивает β, target≠fit)."""
    rng = np.random.default_rng(seed)
    if amp_scale is None:
        amp_scale = [1.0] * n_docs
    docs = []
    for i in range(n_docs):
        ph0 = rng.uniform(0, 2 * math.pi) if jitter_phase else 0.0
        ascale = float(amp_scale[i])
        U = np.zeros((T, D), np.float32)
        x = rng.standard_normal(D); U[0] = x / np.linalg.norm(x)
        eta = (rng.standard_normal(D) * (floor / math.sqrt(D))).astype(np.float32)
        st = {}
        for t in range(T - 1):
            v = eta.astype(np.float32).copy()
            v += rng.standard_normal(D).astype(np.float32) * sig
            for j, m in enumerate(modes):
                kind = m[0]
                if kind == 'r':
                    _, r, sd, dv = m
                    a = r * st.get(j, 0.0) + rng.normal(0, sd)          # β-канал: amp_scale НЕ трогает (v12п1)
                    st[j] = a; v += a * dv
                elif kind == 'c':
                    _, r, om, sd, wv = m
                    s1, s2 = st.get(j, (0.0, 0.0))
                    s0 = 2 * r * math.cos(om) * s1 - r * r * s2 + rng.normal(0, sd * ascale)  # амплитуда AR(2) ∝Driving noise → sd*ascale
                    st[j] = (s0, s1)
                    v += s0 * wv
                elif kind == 'rot':
                    _, r, om, sd, (u1, u2) = m
                    z = st.get(j, np.zeros(2, np.float32))
                    z = r * np.array([[math.cos(om), -math.sin(om)],
                                      [math.sin(om), math.cos(om)]], np.float32) @ z \
                        + rng.standard_normal(2).astype(np.float32) * (sd * ascale)
                    st[j] = z
                    v += z[0] * u1 + z[1] * u2
                elif kind == 'x':
                    _, om, amp, wv = m
                    v += (amp * ascale * math.cos(om * t + ph0)) * wv
            nx = U[t] + v
            U[t + 1] = nx / np.linalg.norm(nx)
        docs.append(derive2(U))
    return docs

# ---------- Δ-оценщик: ОДИН для real и synthetic ----------

def lag14(docs, lags=LAGS14):
    out = {}
    for k in lags:
        real, nl = [], []
        for d in docs:
            Vn = d["V"][d["ok"]]
            Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
            if len(Vn) <= k:
                continue
            real.append(np.median([float(Vn[i + k] @ Vn[i]) for i in range(len(Vn) - k)]))
            W = E.shuffle_seq(Vn)
            nl.append(np.median([float(W[i + k] @ W[i]) for i in range(len(W) - k)]))
        real = np.array(real); nl = np.array(nl)
        dl = E.bci((real - nl).tolist())
        out[k] = dict(med=float(np.median(real)), delta=dl, null=float(np.median(nl)))
    return out

# ---------- фазовый ток (R2) ----------

def phase_current(docs):
    """R2-статистика, глобальная (без выбора плоскости — при Sa≈0 выбор плоскости шумит,
    self-audit): J_glob = ‖Sa‖_F / ‖Sm‖_F от lag-1 ковариации; + J в топ-плоскости."""
    C1 = np.zeros((D, D), np.float64); n = 0
    for d in docs:
        idx = np.nonzero(d["ok"][1:] & d["ok"][:-1])[0]
        P = d["V"][idx].astype(np.float64); N = d["V"][idx + 1].astype(np.float64)
        P = P - P.mean(0, keepdims=True); N = N - N.mean(0, keepdims=True)   # ревью v13п2: per-doc центрирование = cov1 (η уходит в сим.часть)
        C1 += P.T @ N; n += len(idx)
    C1 /= max(n, 1)
    Sm = (C1 + C1.T) / 2; Sa = (C1 - C1.T) / 2
    j_glob = float(np.linalg.norm(Sa) / max(np.linalg.norm(Sm), 1e-12))
    if j_glob < 0.05:                        # ревью E55п3/6: Sa≈0 ⇒ плоскость не определена, j_plane=0 семантически
        return j_glob, 0.0
    w, Q = np.linalg.eigh(1j * Sa)
    iv = int(np.argmax(w))
    v = Q[:, iv]                             # плоскость = span(Re v, Im v) главного собственного вектора 1j·Sa
    vr, vi = v.real, v.imag
    nrm_r, nrm_i = np.linalg.norm(vr), np.linalg.norm(vi)
    if min(nrm_r, nrm_i) < 1e-10:            # self-audit: Sa скалярно ⇒ Im-направление вырождено
        return j_glob, 0.0
    u = np.stack([vr / nrm_r, vi / nrm_i])   # (2,D) — правка ревью: был vstack → (4,D)
    A = u @ C1 @ u.T
    den = max(abs(A[0, 1] + A[1, 0]) / 2 + math.sqrt(max(abs(A[0, 0] * A[1, 1]), 0.0)), 1e-12)
    j_plane = float(abs(A[0, 1] - A[1, 0]) / den)
    return j_glob, j_plane

def j_paired(docs):
    """Ревью v13/v14: J с PAIRED shuffle-бейслайном. Плоскость = топ-2 СИММЕТРИЧНОЙ части
    C1 (сигнал там; плоскость из Sa переобучается на антисим-шум → null j=60, а перезапуск
    eigh на shuffled давал Sa≈0→случайную плоскость→Jsh→∞). Та же плоскость к real и
    shuffled C1. Возвращает (j_glob, j_plane, j_plane_shuf, j_excess)."""
    def c1_of(ds, shuffle=False):
        C = np.zeros((D, D), np.float64); n = 0
        for d in ds:
            idx = np.nonzero(d["ok"][1:] & d["ok"][:-1])[0]
            P = d["V"][idx].astype(np.float64); N = d["V"][idx + 1].astype(np.float64)
            if shuffle:
                P = E.shuffle_seq(P)
            P = P - P.mean(0, keepdims=True); N = N - N.mean(0, keepdims=True)
            C += P.T @ N; n += len(idx)
        return C / max(n, 1)
    C1r = c1_of(docs)
    Sm = (C1r + C1r.T) / 2
    jg = float(np.linalg.norm((C1r - C1r.T) / 2) / max(np.linalg.norm(Sm), 1e-12))
    # плоскость = топ-2 СИММЕТРИЧНОЙ части (сигнал там); антисимметрию меряем в ней.
    # Ревью v13: плоскость из Sa переобучается на антисим-шум → null j=60.
    def jplane(C1, U):
        A = U @ C1 @ U.T
        den = math.sqrt(max(abs(A[0, 0] * A[1, 1]), 1e-18))
        return float(abs(A[0, 1] - A[1, 0]) / den)
    if jg < 0.02:
        return jg, 0.0, 0.0, 0.0
    _, Qsm = np.linalg.eigh(Sm)
    U = np.stack([Qsm[:, -1], Qsm[:, -2]])          # топ-2 Sm
    C1s = c1_of(docs, shuffle=True)
    j, js = jplane(C1r, U), jplane(C1s, U)
    return jg, j, js, j - js

# ---------- спектральный фит: Kung/Prony по взвешенной Δ (6.10) ----------

def kung_fit(delta, sigma, lags=None, pmax=3):
    """delta: массив ρ(k), k=1..K ПОСЛЕДОВАТЕЛЬНЫХ лагов (Hankel y[i+j] требует шага 1 —
    review v15 critical: разреженные LAGS14 сдвигали ось и делали r/ω артефактами).
    σ — веса. Возвращает выбор по BIC: список мод + obj."""
    if lags is not None and tuple(lags) != tuple(range(1, len(delta) + 1)):
        raise ValueError("kung_fit: нужны последовательные лаги 1..K (Hankel), получил нешаговую сетку")
    y = np.asarray(delta, float)
    w = np.asarray(sigma, float)
    K = len(y)
    best = None
    for p in range(0, pmax + 1):
        modes, obj = [], float(sum((y / w) ** 2))   # мёртвый `amps` убран (ревью v13 мелочь)
        if p > 0:
            L = max(2, (K + 1) // 2); M = K + 1 - L
            if L * M < p:
                continue
            H = np.empty((L, M))   # Hankel от ρ(1..K); веса входят через WLS амплитуд и obj
            for i in range(L):
                for j in range(M):
                    H[i, j] = y[i + j]
            U_, s_, Vt = np.linalg.svd(H, full_matrices=False)
            s_[s_ < 1e-10 * s_[0]] = 0
            p_use = min(p, int((s_ > 1e-8 * s_[0]).sum()))
            Us = U_[:, :p_use] * s_[:p_use]
            try:
                Z = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0]
                z = np.linalg.eigvals(Z)
            except np.linalg.LinAlgError:
                continue
            z = z[np.abs(z) < 0.999]
            z = z[np.argsort(-np.abs(z))][:p_use]
            # базис {r^k cos, r^k sin} + константы; WLS амплитуд
            cols = []
            for zi in z:
                r_, om_ = abs(zi), abs(np.angle(zi))
                if r_ < 1e-3:
                    continue
                kk = np.arange(1, K + 1)
                if om_ < 1e-3:
                    cols.append((("r", float(r_), 0.0), r_ ** kk))
                else:
                    cols.append((("c", float(r_), float(om_)), r_ ** kk * np.cos(om_ * kk)))
                    cols.append((("c", float(r_), float(om_)), r_ ** kk * np.sin(om_ * kk)))
            if not cols:
                continue
            Amat = np.stack([c[1] for c in cols], 1) / w[:, None]
            b = y / w
            coef, *_ = np.linalg.lstsq(Amat, b, rcond=None)
            fit = np.stack([c[1] for c in cols], 1) @ coef
            obj = float((((fit - y) / w) ** 2).sum())
            seen, modes = set(), []
            for (key, _col) in cols:
                key = (key[0], round(key[1], 4), round(key[2], 4))
                seen.add(key)
            k_par = sum(2 if m[0] == "r" else 4 for m in list(seen))   # r,c | r,ω,2 амплитуды
        else:
            seen, k_par = set(), 0
        aic = 2 * k_par + obj
        bic = k_par * math.log(K) + obj
        cand = dict(p=p, obj=obj, aic=aic, bic=bic,
                    modes=sorted([list(m) for m in seen]))
        if best is None or cand["bic"] < best["bic"]:
            best = cand
    return best

# ---------- контроли C1-C7 ----------

def unit(v):
    return v / np.linalg.norm(v)

def controls(n_docs=24, seed=11):
    rd = np.random.default_rng(5)
    w1 = unit(rd.standard_normal(D)); w2 = unit(rd.standard_normal(D))
    u1, u2 = w1, unit(w2 - (w2 @ w1) * w1)
    b_m = ('r', 0.15, 0.09, w1)                    # общий β-канал
    om8 = 2 * math.pi / 8
    C = dict(
        C1=[b_m],
        C2=[b_m, ('c', 0.92, om8, 0.05, w2)],
        C3=[b_m, ('rot', 0.92, om8, 0.05, (u1, u2))],
        C4=[b_m, ('c', 0.92, om8, 0.04, w2), ('c', 0.95, 2 * math.pi / 16, 0.03, unit(rd.standard_normal(D)))],
        C5=[],
        C6=[b_m, ('c', 0.92, om8, 0.04, w2), ('r', 0.93, 0.05, unit(rd.standard_normal(D)))],
        C7=[b_m, ('x', om8, 0.06, w2)],       # ВНИМАНИЕ: фаза-джиттер НЕ гасит Δ(k) (ревью v12); честный C7″ — в exp55_sbi.py
    )
    res = {}
    for name, modes in C.items():
        d = simulate_modes(n_docs, seed, modes, jitter_phase=(name == "C7"))
        c = lag14(d)
        delta = [c[k]["delta"][0] for k in LAGS14]
        sigm = [max((c[k]["delta"][2] - c[k]["delta"][1]) / 3.92, 1e-3) for k in LAGS14]
        fit = kung_fit(delta, sigm)
        jg, jp = phase_current(d)
        res[name] = dict(curve={str(k): round(c[k]["delta"][0], 4) for k in LAGS14},
                         sigma={str(k): round(s, 4) for k, s in zip(LAGS14, sigm)},
                         fit=fit, J_glob=jg, J_plane=jp, n_docs=n_docs)
        print(name, "J_glob=%.4f J_plane=%.4f" % (jg, jp), "fit:", fit, flush=True)
    return res

if __name__ == "__main__":
    t0 = time.time()
    res = controls()
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                     "results", "exp55_stage-1.json"), "w"), indent=1, default=float)
    print(f"done {time.time()-t0:.0f}s -> results/exp55_stage-1.json", flush=True)
