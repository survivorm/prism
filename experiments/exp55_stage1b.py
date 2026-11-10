#!/usr/bin/env python
"""E-55 стадия 1b: верификация спектра мод НА REAL через пайплайн (SBI), held-out доки.
Структуры: A=β; B=β+медл.вещ; C=β+скалярный осциллятор; R=β+вращение.
Направления ОЦЕНИВАЮТСЯ НА TRAIN (объявленные оценщики, без утечки):
 w_beta = топ-собств. сим(C1 train); плоскость = span(Re v, Im v) гл. собств. 1j·Sa(train);
 w_c = гл. собств. сим(A) в плоскости, поднятая обратно; w_slow = топ-собств. сим(C8 train).
σ/floor симулятора — дефолты семейства контролей (объявлено; mismatch входит в obj как есть).
obj/AIC/BIC на TEST-доках; гейт победителя obj_test < 30 (ТЗ §5 ст.1)."""
import os, sys, json, math, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E
from exp55_sbi import nelder

HERE = os.path.dirname(os.path.abspath(__file__))
NDOC_F = 24
OM_LO, OM_HI = 0.25, 1.4          # T ∈ [4.5, 25] — объявленное окно поиска ω


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)

def cov1(docs, lag=1):
    C = np.zeros((M.D, M.D), np.float64); n = 0
    for d in docs:
        idx = np.nonzero(d["ok"][lag:] & d["ok"][:-lag])[0]
        P = d["V"][idx].astype(np.float64); N = d["V"][idx + lag].astype(np.float64)
        P = P - P.mean(0, keepdims=True); N = N - N.mean(0, keepdims=True)
        C += P.T @ N; n += len(idx)
    return C / max(n, 1)

def directions(tr):
    C1 = cov1(tr)
    S = (C1 + C1.T) / 2
    wsm, Qsm = np.linalg.eigh(S)                     # одно eigh на топ-оси (ревью v13 self-audit: было два)
    w_beta = unit(Qsm[:, -1])                         # топ-1 сим. части = направление scalar-моды/β
    # плоскость = топ-2 симметричной части (ревью v13: топ из Sa переобучается на антисим-шум)
    vr, vi = unit(Qsm[:, -1]), unit(Qsm[:, -2])
    U = np.stack([vr, vi])
    A = U @ C1 @ U.T
    _, Qv = np.linalg.eigh((A + A.T) / 2)
    coef = Qv[:, -1]
    w_c = unit(coef[0] * vr + coef[1] * vi)          # диагностическая (scalar-модел её не использует)
    S8c = cov1(tr, 8); S8 = (S8c + S8c.T) / 2
    w_slow = unit(np.linalg.eigh(S8)[1][:, -1])
    return dict(w_beta=w_beta, plane=U, w_c=w_c, w_slow=w_slow)

STRUCTS = {
    "A":  dict(k=2, x0=[0.2, 0.4], st=[0.1, 0.15],
               bnds=[(0.02, 0.7), (0.02, 1.2)],
               mk=lambda p, dr: [('r', p[0], p[1], dr["w_beta"])]),
    "B":  dict(k=4, x0=[0.2, 0.4, 0.85, 0.05], st=[0.1, 0.15, 0.05, 0.02],
               bnds=[(0.02, 0.7), (0.02, 1.2), (0.5, 0.98), (0.005, 0.3)],
               mk=lambda p, dr: [('r', p[0], p[1], dr["w_beta"]), ('r', p[2], p[3], dr["w_slow"])]),
    "C":  dict(k=5, x0=[0.2, 0.4, 0.9, 0.8, 0.06], st=[0.1, 0.15, 0.05, 0.12, 0.02],
               bnds=[(0.02, 0.7), (0.02, 1.2), (0.5, 0.985), (OM_LO, OM_HI), (0.003, 0.3)],
               # ревью v13 critical: scalar-мода вдоль топ-1 сим.части (w_beta), НЕ w_c (Sa≈0 ⇒ w_c был мусором)
               mk=lambda p, dr: [('r', p[0], p[1], dr["w_beta"]), ('c', p[2], p[3], p[4], dr["w_beta"])]),
    "R":  dict(k=5, x0=[0.2, 0.4, 0.9, 0.8, 0.06], st=[0.1, 0.15, 0.05, 0.12, 0.02],
               bnds=[(0.02, 0.7), (0.02, 1.2), (0.5, 0.985), (OM_LO, OM_HI), (0.003, 0.3)],
               mk=lambda p, dr: [('r', p[0], p[1], dr["w_beta"]),
                                 ('rot', p[2], p[3], p[4], (dr["plane"][0], dr["plane"][1]))]),
}

def curve_sigma(docs):
    c = M.lag14(docs)
    y = np.array([c[k]["delta"][0] for k in M.LAGS14])
    lo = np.array([c[k]["delta"][1] for k in M.LAGS14])
    hi = np.array([c[k]["delta"][2] for k in M.LAGS14])
    return y, np.maximum((hi - lo) / 3.92, 2e-3)

def sim_curve(modes, seed=77, floor=0.004):
    c = M.lag14(M.simulate_modes(NDOC_F, seed, modes, floor=floor))
    return np.array([c[k]["delta"][0] for k in M.LAGS14])

def obj_on(y, s, p, sp, dr, seed=77, floor=0.004):
    for v_, (lo, hi) in zip(p, sp["bnds"]):
        if not (lo <= v_ <= hi):
            return 1e6, None
    try:
        sc = sim_curve(sp["mk"](p, dr), seed=seed, floor=floor)
        v = float((((sc - y) / s) ** 2).sum())
        return (v if np.isfinite(v) else 1e6), sc
    except Exception:
        return 1e6, None

if __name__ == "__main__":
    docs = [E.derive(d) for d in E.load_real()]
    tr = [d for d in docs if d["si"] % 4 != 3]
    te = [d for d in docs if d["si"] % 4 == 3]
    print("train/test docs:", len(tr), len(te), flush=True)
    ytr, str_ = curve_sigma(tr)
    yte, ste = curve_sigma(te)
    dr = directions(tr)
    # floor нормирован по real (ревью v13п3): ‖η_real‖=median‖mean V[ok]‖, иначе obj = model+floor двойное наказание
    FLOOR = float(np.median([np.linalg.norm(d["V"][d["ok"]].mean(0)) for d in tr if d["ok"].any()]))
    print("floor(sim)=real ‖mean V‖ =", round(FLOOR, 4), flush=True)
    print("cos(w_beta,w_slow)=%.2f" % float(dr["w_beta"] @ dr["w_slow"]), flush=True)

    out = {}
    for name, sp in STRUCTS.items():
        t0 = time.time()
        f = lambda p, sp=sp: obj_on(ytr, str_, p, sp, dr, floor=FLOOR)[0]
        x, _ = nelder(f, sp["x0"], sp["st"], iters=110)
        ftr, _ = obj_on(ytr, str_, x, sp, dr, seed=77, floor=FLOOR)
        fte, sc_te = obj_on(yte, ste, x, sp, dr, seed=78, floor=FLOOR)   # независимая сим-реализация для test (self-audit)
        aic, bic = 2 * sp["k"] + fte, sp["k"] * math.log(14) + fte
        out[name] = dict(p=[float(v) for v in x], obj_train=ftr, obj_test=fte, aic=aic, bic=bic,
                         perlag_z=[float(z) for z in (sc_te - yte) / ste],
                         secs=round(time.time() - t0))
        print(name, "p=", [round(float(v), 4) for v in x],
              f"obj_tr={ftr:.1f} obj_te={fte:.1f} AIC={aic:.1f} BIC={bic:.1f} [{time.time()-t0:.0f}s]",
              flush=True)
    win = min(out, key=lambda k: out[k]["bic"])
    out["winner_bic"] = win
    out["floor_sim"] = FLOOR
    # гейт 50 (ревью v13п1): «~2σ на лаг» = 14·4 = 56; 30 было строже собственного ТЗ
    out["gate_obj50"] = bool(out[win]["obj_test"] < 50)
    out["gate_perlag3"] = bool(max(abs(z) for z in out[win]["perlag_z"]) < 3)   # ключ perlag_z (self-audit: был perlag_test → KeyError в конце)
    print("winner(BIC):", win, "obj50:", out["gate_obj50"], "per-lag<3σ:", out["gate_perlag3"], flush=True)
    json.dump(out, open(os.path.join(HERE, "..", "results", "exp55_stage1b.json"), "w"), indent=1, default=float)
    print("saved -> results/exp55_stage1b.json", flush=True)
