#!/usr/bin/env python
"""E-62 battery (однопроходная): shared-exp 14лаг, состав osc/exp, clip, strict (CI Δχ²≤1),
фазы при своей ω*_strict. Калибровка t1: на L23-npz обязана воспроизвести
Qwen-базлайн (osc~45/64, clip~14, strict~15, shared-exp≈0.913@(0.16,0.97))."""
import math
import numpy as np

K = np.arange(1, 25)
NL = 14


def curve_u(u_rows, hn):
    U = np.asarray(u_rows, dtype=np.float64) * np.asarray(hn, dtype=np.float64)[:, None]
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    T = U.shape[0]
    Vo = np.zeros((T - 1, U.shape[1]))
    for t in range(T - 1):
        Vo[t] = U[t + 1] - float(U[t + 1] @ U[t]) * U[t]
    ok = np.linalg.norm(Vo, axis=1) > 1e-6
    V = Vo[ok]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


def run(traj, dom_of, seed=11, w_step=0.02, dom_need=3,
        dom_order=("wiki", "math", "code", "chat"), verbose=True):
    """traj: {name: (u_rows, hnorm)}. Грид и порядок обхода выровнены с exp60b
    (WG 0.02, RGX→0.9996, rng(11), обход в порядке пула exp47b wiki→chat).
    Возвращает словарь фактов (NaN-safe скаляры)."""
    WG = np.round(np.arange(0.10, 2.001, w_step), 3)
    RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
    RGX = np.round(np.concatenate([RG, np.arange(0.992, 0.99965, 0.001)]), 4)
    RG2 = RG[::2]
    ONES = np.ones(len(K))
    rng = np.random.default_rng(seed)

    def wls(y, w, cols):
        G = np.stack(cols, 1)
        co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        return float(np.sum(w * (y - G @ co) ** 2)), co

    pairs2 = [(a, b) for a in RG2 for b in RG2[RG2 > a + 0.02]]
    Ys, SDs, win, om_r, profs = {}, {}, {}, {}, {}
    ordm = {d: i for i, d in enumerate(dom_order)}
    names = sorted(traj, key=lambda n: (ordm.get(dom_of[n], 9), int(n.rsplit("_", 1)[-1])))
    for i, nm in enumerate(names):
        V = curve_u(*traj[nm])
        y, sg = [], []
        for k in K:
            P = np.einsum("ij,ij->i", V[:-k], V[k:])
            y.append(float(np.median(P)))
            sg.append(max(float(np.median(P[rng.integers(0, len(P), (200, len(P)))], axis=1).std()), 2e-3))
        y = np.array(y); sd = np.array(sg); w = 1.0 / sd ** 2
        tot = float(np.sum(w * y ** 2))
        Ys[nm] = y; SDs[nm] = sd
        best = None
        prof = np.zeros(len(WG))
        for j, om in enumerate(WG):
            c, s = np.cos(om * K), np.sin(om * K)
            fits = [wls(y, w, [r_ ** K * c, r_ ** K * s, ONES])[0] for r_ in RGX]
            prof[j] = min(fits)
            m = int(np.argmin(fits))
            if best is None or prof[j] < best[0]:
                best = (prof[j], float(om), float(RGX[m]), j)
        be = None
        for a, b in pairs2:
            sse, _ = wls(y, w, [a ** K, b ** K, ONES])
            if be is None or sse < be[0]:
                be = (sse, float(a), float(b))
        win[nm] = "osc" if best[0] < be[0] else "exp"
        om_r[nm] = (best[1], best[2], float(1 - best[0] / tot))
        profs[nm] = prof                 # сырой взвешенный χ²-профиль (ревью #4: без ×tot)
        if verbose and i % 16 == 15:
            print(f"  ...{i+1}", flush=True)
    # strict
    strict = {}
    for nm in names:
        if win[nm] != "osc":
            continue
        dch = profs[nm] - profs[nm].min()   # честный Δχ² (не Δχ²·SST)
        p = int(np.argmin(dch))
        l_ = p
        while l_ > 0 and dch[l_ - 1] <= 1.0:
            l_ -= 1
        r_ = p
        while r_ < len(dch) - 1 and dch[r_ + 1] <= 1.0:
            r_ += 1
        ok = bool(p > 0 and p < len(WG) - 1 and om_r[nm][1] < 0.989 and (WG[r_] - WG[l_]) < 0.6)
        strict[nm] = ok
    # доменные ω* (по strict) и фазы всех доков домена
    doms = sorted({dom_of[n] for n in names})
    real = {dm: np.mean([Ys[n] for n in names if dom_of[n] == dm], 0) for dm in doms}
    Kl = K[:NL]; ONl = np.ones(NL)
    RG2b = RG2

    def shared_exp(tgt):
        be = None
        for a in RG2b:
            for b in RG2b[RG2b > a + 0.02]:
                G = np.stack([a ** Kl, b ** Kl, ONl], 1)
                sse = tt = 0.0
                for dm in doms:
                    x = tgt[dm][:NL]
                    co = np.linalg.lstsq(G, x, rcond=None)[0]
                    sse += float(np.sum((x - G @ co) ** 2)); tt += float(np.sum(x ** 2))
                if be is None or sse < be[0]:
                    be = (sse, tt, a, b)
        return dict(R2=round(1 - be[0] / be[1], 3), r1=float(be[2]), r2=float(be[3]))

    F1 = shared_exp(real)
    phase = {}
    for dm in doms:
        sel = [n for n in names if dom_of[n] == dm and strict.get(n, False)]
        if len(sel) < min(3, dom_need):
            phase[dm] = None
            continue
        pr = np.zeros(len(WG))
        for n in sel:
            w = 1.0 / SDs[n] ** 2
            tot = float(np.sum(w * Ys[n] ** 2))
            pr += (tot - profs[n]) / tot          # сумма взвешенных R² по strict-докам
        star = float(WG[int(np.argmax(pr))])
        ps = []
        psS = []
        for n in [m for m in names if dom_of[m] == dm]:
            w = 1.0 / SDs[n] ** 2
            c, s = np.cos(star * K), np.sin(star * K)
            fits = [wls(Ys[n], w, [r ** K * c, r ** K * s, ONES]) for r in RGX]
            cf = min(fits, key=lambda t: t[0])[1]
            p_ = math.atan2(cf[1], cf[0])
            ps.append(p_)
            if strict.get(n, False):
                psS.append(p_)
        phase[dm] = dict(omega=star,
                         Rbar=round(float(abs(np.exp(1j * np.array(ps)).mean())), 2),
                         Rbar_strict=round(float(abs(np.exp(1j * np.array(psS)).mean())), 2)
                         if psS else None,
                         n_strict=len(sel), n_all=len(ps),
                         strict_names=[m for m in sel])
    return dict(
        F1=F1,
        n_osc=sum(1 for v in win.values() if v == "osc"),
        n=len(names),
        clip=sum(1 for nm in names if win[nm] == "osc" and om_r[nm][1] >= 0.989),
        n_strict=sum(1 for v in strict.values() if v),
        phase={k: (None if v is None else (v["omega"], v["Rbar"], v["Rbar_strict"],
                                           v["n_strict"], v["strict_names"]))
               for k, v in phase.items()},
        dip1=round(float(np.mean([real[dm][0] for dm in doms])), 3),  # Δ(1) (dip на lag 1 = отрицательный)
        winners=win, om_r={k: list(v) for k, v in om_r.items()},
    )
