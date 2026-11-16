#!/usr/bin/env python
"""E-59c (docs/E59c.md, пред-рег до прогона): одна ω или разброс?
RG→0.98; Cochran Q по strict-ident winners; SBC: 100 синтетик на док из фитированных
параметров + Gaussian σ(лаг), рефит ω при r=r̂ — калибрует Q против допущения метода.
Дополнительно: рефит r_clip-доков при r∈{0.90,0.95,0.98}; контроль ω̂(fix r̂) vs ω̂(free)."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
K = np.arange(1, 25)
n = len(K)
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.981, 0.02), 3)
ONES = np.ones(n)
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]

def unit(doc):
    V = doc["V"][doc["ok"]]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

def curve_and_sigma(Vn, B=200):
    y, sg = [], []
    for k in K:
        P = np.einsum("ij,ij->i", Vn[:-k], Vn[k:])
        y.append(float(np.median(P)))
        sg.append(max(float(np.median(P[rng.integers(0, len(P), (B, len(P)))], axis=1).std()), 2e-3))
    return np.array(y), np.array(sg)

def wls(y, w, cols):
    G = np.stack(cols, 1)
    A = G * w[:, None] ** .5
    coef = np.linalg.lstsq(A, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ coef) ** 2)), coef

def osc_profile(y, w, r):
    rk = r ** K
    return np.array([wls(y, w, [rk * np.cos(om * K), rk * np.sin(om * K), ONES])[0] for om in WG])

def walk_ci(dch):
    p = int(np.argmin(dch))
    l_ = p
    while l_ > 0 and dch[l_ - 1] <= 1.0:
        l_ -= 1
    r_ = p
    while r_ < len(dch) - 1 and dch[r_ + 1] <= 1.0:
        r_ += 1
    return WG[l_], WG[r_], bool(l_ == 0 or r_ == len(dch) - 1), p

out = []
for d in docs:
    Vn = unit(d)
    if len(Vn) <= 24:
        continue
    y, sd = curve_and_sigma(Vn)
    w = 1.0 / sd ** 2
    tot = float(np.sum(w * y ** 2))
    be = None
    for a in RG:
        for b in RG[RG > a + 0.02]:
            sse, _ = wls(y, w, [a ** K, b ** K, ONES])
            if be is None or sse < be[0]:
                be = (sse, a, b)
    chi2 = np.empty((len(WG), len(RG)))
    for j, om in enumerate(WG):
        c, s = np.cos(om * K), np.sin(om * K)
        for m, r in enumerate(RG):
            rk = r ** K
            chi2[j, m] = wls(y, w, [rk * c, rk * s, ONES])[0]
    jm, mm = np.unravel_index(np.argmin(chi2), chi2.shape)
    om_hat, r_hat = WG[jm], RG[mm]
    lo_, hi_, clip, _ = walk_ci(chi2[:, mm] - chi2[jm, mm])   # профиль ω при свободном r
    _, coef = wls(y, w, [(r_hat ** K) * np.cos(om_hat * K), (r_hat ** K) * np.sin(om_hat * K), ONES])
    hr = {k: float(np.median(np.einsum("ij,ij->i", Vn[:-k], Vn[k:]))) for k in range(3, 17)}
    out.append(dict(name=d["name"], dom=d["dom"], y_real=y.tolist(),
                    osc_wins=bool(chi2[jm, mm] < be[0]),   # BIC: обе модели 5 пар ⇒ по SSE
                    omega=float(om_hat), r_osc=float(r_hat), R2_osc=1 - chi2[jm, mm] / tot,
                    ci_w=float(hi_ - lo_), ci_clip=clip, r_clip=bool(r_hat >= RG[-1] - 1e-9),
                    hump=int(max(hr, key=hr.get)),
                    sd=sd.tolist(), coef=coef.tolist(),
                    identifiable=bool(not clip and r_hat < RG[-1] - 1e-9 and hi_ - lo_ < 0.6)))
    if len(out) % 16 == 0:
        print(f"...{len(out)}", flush=True)

win = [o for o in out if o["osc_wins"]]
ident = [o for o in win if o["identifiable"]]
rclip = [o for o in win if o["r_clip"]]
print(f"winners {len(win)}/64 | strict {len(ident)} | r_clip {len(rclip)}", flush=True)

def cochran(omg, sdv):
    w_ = 1.0 / np.asarray(sdv) ** 2
    mbar = float(np.sum(w_ * np.asarray(omg)) / np.sum(w_))
    return float(np.sum(w_ * (np.asarray(omg) - mbar) ** 2)), mbar

verdict_info = {}
world = "вердикт не выносится (strict<12)"
if len(ident) >= 12:
    omg = [o["omega"] for o in ident]
    sdv = [max(o["ci_w"] / 2, 0.02) for o in ident]
    Q, mbar = cochran(omg, sdv)
    # SBC: 100 синтетик/док из фитированных параметров, рефит ω при r=r̂
    NS = 100
    om_sim = np.empty((NS, len(ident)))
    for i, o in enumerate(ident):
        rk = o["r_osc"] ** K
        c, s = np.cos(o["omega"] * K), np.sin(o["omega"] * K)
        mu = np.stack([rk * c, rk * s, ONES], 1) @ np.array(o["coef"])
        sd = np.array(o["sd"])
        for j in range(NS):
            ys = mu + rng.standard_normal(n) * sd
            prof = osc_profile(ys, 1.0 / sd ** 2, o["r_osc"])
            om_sim[j, i] = WG[int(np.argmin(prof))]
    Q_sim = np.array([cochran(om_sim[j], sdv)[0] for j in range(NS)])
    pQ = float(np.mean(Q_sim >= Q))
    # --- самоаудит v2: исходный SBC проверял НЕ тот null (собственные ω̂ доку — тривиально).
    # Правильный тест «одной ω»: sd_i^boot из разброса ω̂_sim вокруг ω̂_i + null ВСЕХ доков
    # при общей ω₀ (перефит coef при (ω₀, r̂_i)). Оба дешёвые, данные уже есть.
    sd_boot = np.maximum(om_sim.std(axis=0, ddof=1), 0.02)
    wcal = 1.0 / sd_boot ** 2
    mbar_c = float(np.sum(wcal * np.array(omg)) / np.sum(wcal))
    Q_cal = float(np.sum(wcal * (np.array(omg) - mbar_c) ** 2))
    # null: общая ω₀ = inv-var среднее; 100 сим с общими ω₀, свои r̂ и coef(ω₀,r̂_i)
    om_null = np.empty((NS, len(ident)))
    mus = []
    for i, o in enumerate(ident):
        rk0 = o["r_osc"] ** K
        c0, s0 = np.cos(mbar_c * K), np.sin(mbar_c * K)
        G0 = np.stack([rk0 * c0, rk0 * s0, ONES], 1)
        _, cf = wls(np.array(o["y_real"]), 1.0 / np.array(o["sd"]) ** 2, [rk0 * c0, rk0 * s0, ONES])
        mus.append((G0, cf, np.array(o["sd"])))
    for j in range(NS):
        for i, (G0, cf, sd) in enumerate(mus):
            ys = G0 @ cf + rng.standard_normal(n) * sd
            prof = osc_profile(ys, 1.0 / sd ** 2, ident[i]["r_osc"])
            om_null[j, i] = WG[int(np.argmin(prof))]
    Q_null = np.array([cochran(om_null[j], sd_boot.tolist())[0] for j in range(NS)])
    p_shared = float(np.mean(Q_null >= Q_cal))
    # контроль метода на РЕАЛЬНЫХ данных: профиль ω при r=r̂ фикс против свободного фита
    ctrl = []
    for o in ident:
        prof = osc_profile(np.array(o["y_real"]), 1.0 / np.array(o["sd"]) ** 2, o["r_osc"])
        ctrl.append(abs(WG[int(np.argmin(prof))] - o["omega"]) > 0.02 + 1e-9)
    verdict_info = dict(Q=Q, df=len(ident) - 1, omega_bar=mbar, p_SBC=pQ,
                        Q_sim_med=float(np.median(Q_sim)), control_disagree_frac=float(np.mean(ctrl)),
                        Q_cal=Q_cal, omega_bar_cal=mbar_c, p_shared_omega=p_shared,
                        Q_null_med=float(np.median(Q_null)), sd_boot=sd_boot.tolist())
    print(f"CAL: Q_cal={Q_cal:.1f}/{len(ident)-1} ω̄={mbar_c:.2f} Q_null_med={np.median(Q_null):.1f} "
          f"p_shared={p_shared:.3f}", flush=True)
    if np.mean(ctrl) > 0.2:
        world = "МЕТОД НЕСТАБИЛЕН (ω̂(fix r̂) расходится с ω̂(free) у >20%) — вердикт не выносится"
    elif p_shared > 0.05:
        world = f"D: ОДНА ω≈{mbar_c:.2f} в пределах восстановимого шума; вариативны затухание/амплитуда"
    else:
        world = "C: реальный разброс ω (Q_cal превышает null общей ω₀)"
    print(f"Q={Q:.1f}/{len(ident)-1} ω̄={mbar:.2f} p_SBC(own-ω, инф.=нет)={pQ:.3f} ctrl_bad={np.mean(ctrl):.2f}", flush=True)

# r_clip рефиты: куда уходит ω при фиксированном r за пределами прежнего grid
rclip_res = {}
for rf in (0.90, 0.95, 0.98):
    om_at_r, n_bad = [], 0
    for o in rclip:
        sd = np.array(o["sd"])
        prof = osc_profile(np.array(o["y_real"]), 1.0 / sd ** 2, rf)
        lo2, hi2, cl2, p2 = walk_ci(prof - prof.min())
        om_at_r.append(float(WG[p2]))
        n_bad += int(cl2)
    if om_at_r:
        rclip_res[str(rf)] = dict(med=float(np.median(om_at_r)), p25=float(np.percentile(om_at_r, 25)),
                                  p75=float(np.percentile(om_at_r, 75)), clip_frac=n_bad / len(om_at_r))
print("r_clip refit ω̂(r):", {k: round(v["med"], 2) for k, v in rclip_res.items()}, flush=True)

# spearman по strict
rho = None
if len(ident) >= 8:
    A_ = np.array([o["omega"] for o in ident]); B_ = np.array([o["hump"] for o in ident])
    rho = float(np.corrcoef(np.argsort(np.argsort(A_)), np.argsort(np.argsort(B_)))[0, 1])

def gmm_bic(x, k):
    if len(x) < 4 * k:
        return np.inf
    mu = np.linspace(x.min(), x.max(), k + 2)[1:-1] if k > 1 else np.array([x.mean()])
    sdv = np.full(k, x.std() / max(k, 1) + 1e-6); pk = np.full(k, 1.0 / k); ll = -np.inf
    for _ in range(150):
        L = -np.log(2 * np.pi * sdv ** 2) / 2 - ((x[:, None] - mu) / sdv) ** 2 / 2 + np.log(pk)
        m = L.max(1, keepdims=True)
        r = np.exp(L - m); r /= r.sum(1, keepdims=True)
        pk = r.mean(0); mu = (r * x[:, None]).sum(0) / r.sum(0)
        sdv = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / r.sum(0)) + 1e-6
        ll = float(np.sum(m[:, 0] + np.log(np.exp(L - m).sum(1))))
    return -2 * ll + (2 * k + k - 1) * math.log(len(x))

gm = {}
if len(ident) >= 12:
    _w = np.array([o["omega"] for o in ident])
    b1, b2 = gmm_bic(_w, 1), gmm_bic(_w, 2)
    gm = dict(bic1=float(b1), bic2=float(b2), bimodal=bool((b1 - b2) >= 6))
# пред-рег E59c.md: p≤0.05 ∧ GMM-бимодально → B (раньше в коде терялось — самоаудит)
if verdict_info and verdict_info["p_shared_omega"] <= 0.05 and gm.get("bimodal"):
    world = "B: два типа документов (значимый разброс + бимодальный ω̂)"

res = dict(n_win=len(win), n_ident=len(ident), n_rclip=len(rclip), world=world,
           ident_params=verdict_info, ident_omega=[o["omega"] for o in ident],
           ident_r=[o["r_osc"] for o in ident], rclip_refit=rclip_res,
           spearman_ident=rho, gmm_ident=gm,
           docs=[{k: v for k, v in o.items() if k not in ("sd", "coef", "y_real")} for o in out])
print("МИР:", world, "| spearman(strict):", rho, "| gmm_ident:", gm, flush=True)
def sane(o):
    if isinstance(o, dict):  return {k: sane(v) for k, v in o.items()}
    if isinstance(o, list):  return [sane(v) for v in o]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):  return None
    return o
json.dump(sane(res), open(os.path.join(HERE, "..", "results", "exp59c.json"), "w"), indent=1, allow_nan=False)
