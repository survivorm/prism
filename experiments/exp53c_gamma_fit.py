#!/usr/bin/env python
"""E-53c v2: фит (β, γ, φ) по Δ-кривой — вопрос ревью v9 §5/v10: закрывает ли slow-drift
знак-flip на лагах 3–8. Правки v10: β фитится (фикс. β навязывал Δ(2)=β², доминант в obj);
β_w=0 (random w — численный knob, не физика); веса W[k] из CI артефакта; A,β0 из JSON;
диагностика границ и сходимости. Пункт v10-(2) отвергнут: Δ = raw − shuffle-null floor-инвариантна
(E⟨W_i,W_{i+1}⟩=‖μ‖² для shuffle-пар), 45× разрыв null сокращается в Δ по построению."""
import os, sys, json, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.argv = ['x']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp53b_servo.json")
_j = json.load(open(ART))
_real_e = _j["delta_curve"]
real = {int(k): e["delta"][0] for k, e in _real_e.items()}
W = {int(k): max((e["delta"][2] - e["delta"][1]) / 3.92, 1e-3) for k, e in _real_e.items()}  # σ лага из ПОЛНОЙ ширины CI (ревью v11: был σ/2)
LAGS_F = [1, 2, 3, 4, 8, 16]
assert set(LAGS_F) <= set(E.LAGS)
A = float(_j["stage1"]["a"])                    # из JSON (ревью v10: не hardcode)
B0 = float(-_j["stage1"]["b"])
BND = dict(b=(0.0, 0.5), g=(0.0, 0.6), p=(0.5, 0.995))

def sim_delta(beta, gam, phi, seed=7, nd=24):
    d = E.simulate(nd, alpha=A, beta=beta, gamma=gam, m_phi=phi, seed=seed)
    c = E.lag_curve(d)
    return {k: c[k]["delta"][0] for k in LAGS_F}, c      # та же статистика, что и real-таргеты

def obj(p):
    beta, gam, phi = p
    if not (BND["b"][0] <= beta <= BND["b"][1] and BND["g"][0] <= gam <= BND["g"][1]
            and BND["p"][0] <= phi <= BND["p"][1]):
        return 1e6
    try:                                                    # ревью v11: экстремальные параметры могут уронить sim/lag_curve
        sd, _ = sim_delta(beta, gam, phi)
    except Exception:
        return 1e6
    s = sum(((sd[k] - real[k]) / W[k]) ** 2 for k in LAGS_F)
    return float(s) if np.isfinite(s) else 1e6

def obj_split(sd):
    e = sum(((sd[k] - real[k]) / W[k]) ** 2 for k in (1, 2))
    l = sum(((sd[k] - real[k]) / W[k]) ** 2 for k in (3, 4, 8, 16))   # ревью v12: измеримый early/late
    return float(e), float(l)

def nelder(f, x0, step, iters=160):
    n = len(x0); pts = [np.array(x0, float)]
    for i in range(n):
        q = np.array(x0, float); q[i] += step[i]; pts.append(q)
    vals = [f(q) for q in pts]; neval = n + 1
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        cen = np.mean(pts[:-1], 0)
        xr = cen + (cen - pts[-1]); fr = f(xr); neval += 1
        if fr < vals[0]:
            xe = cen + 2.0 * (cen - pts[-1]); fe = f(xe); neval += 1
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc); neval += 1
            if fc < vals[-1]: pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i]); neval += 1
    o = np.argsort(vals)
    spread = float(np.max(np.abs(pts[o[1]] - pts[o[0]])))          # сходимость: разброс simplex
    return pts[o[0]], vals[o[0]], neval, spread

if __name__ == "__main__":
    print("real targets:", {k: round(v, 4) for k, v in real.items()}, flush=True)
    print("W (σ лага):", {k: round(v, 4) for k, v in W.items()}, "A=", round(A, 4), flush=True)
    t0 = time.time()
    x, f, nev, sp = nelder(obj, [B0, 0.05, 0.93], [0.05, 0.05, 0.04])
    beta, gam, phi = x
    edges = []
    for nm, v_, key in (("beta", x[0], "b"), ("gamma", x[1], "g"), ("phi", x[2], "p")):
        lo, hi = BND[key]
        if v_ < lo + 0.02 * (hi - lo) or v_ > hi - 0.02 * (hi - lo):
            edges.append(nm)
        print(f"  {nm}: d_lo={v_-lo:.4f} d_hi={hi-v_:.4f}", flush=True)          # ревью v11: расстояние до границы
    print(f"fitted: beta={beta:.4f} (stage1 {B0:.4f})  gamma={gam:.4f}  phi={phi:.4f}", flush=True)
    rob = {}
    # note (ревью v11): Δ=floor-инвариантна только в пределе: ‖μ‖²_sim≈0.001 vs real≈0.06 —
    # остаток разницы mean-структуры живёт в остатках фита, это не только динамика.
    print(f"obj={f:.2f} evals={nev} simplex_spread={sp:.4f} boundary_hits={edges or 'none'} ({time.time()-t0:.0f}s)", flush=True)
    sd, _ = sim_delta(beta, gam, phi)
    o_e, o_l = obj_split(sd)
    print("lag  sim      real    sim/σ   real", flush=True)
    print(f"obj_early={o_e:.1f} obj_late={o_l:.1f}", flush=True)      # (1)+(2) ревью v12
    for k in LAGS_F:
        print(f"{k:>3} {sd[k]:+.4f}  {real[k]:+.4f}  {(sd[k]-real[k])/W[k]:+6.2f}", flush=True)
    for sdz in (101, 202):                                          # робастность: w и шум другие
        sd2, _ = sim_delta(beta, gam, phi, seed=sdz)
        print(f"seed{sdz}:", {k: round(sd2[k], 4) for k in LAGS_F}, flush=True)
        rob[f"sim_curve_seed{sdz}"] = {str(k): float(sd2[k]) for k in LAGS_F}
    json.dump({"beta": float(beta), "gamma": float(gam), "m_phi": float(phi), "obj": float(f),
               "evals": nev, "spread": sp, "boundary_hits": edges, "robust": rob,
               "boundary_dist": {nm: {"d_lo": float(v_ - lo), "d_hi": float(hi - v_)}
                                 for nm, v_, key in (("beta", beta, "b"), ("gamma", gam, "g"), ("phi", phi, "p"))
                                 for lo, hi in [BND[key]]},   # ревью v12 косметика
               "obj_early": o_e, "obj_late": o_l,
               "sim_curve": {str(k): float(sd[k]) for k in LAGS_F},
               "real": {str(k): float(real[k]) for k in LAGS_F},
               "W_sigma": {str(k): float(W[k]) for k in LAGS_F}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                "exp53c_gammafit.json"), "w"), indent=1)
