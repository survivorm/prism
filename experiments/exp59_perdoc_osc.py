#!/usr/bin/env python
"""E-59: per-doc EXP (две релаксации+пол) против OSC (демпфированный косинус+пол), обе 5 пар.
Миры A/B/C по пред-рег E59.md. ω̂ — с документов, где OSC побеждает по BIC; бимодальность —
GMM k=2 vs k=1 (ΔBIC≥6, веса ≥0.25).
Веса (review v16): w=1/SE(median_домена)² — АБСОЛЮТНЫЙ масштаб произволен (lstsq от него
не зависит), относительные веса по лагам корректны; per-doc гетероскедастичность НЕ учтена
(будет per-doc bootstrap-σ в E-59b). Данные — raw медиана cos, веса от delta: модель
предсказывает raw, относительный шум delta≈raw — несогласованность осознанная (limitations).
~30 c CPU."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
K = np.arange(1, 15)
n = len(K)
docs = [E.derive(d) for d in E.load_real()]
doms = sorted({d["dom"] for d in docs})

# σ домена по лагам (веса для per-doc фита)
SIG = {}
for dom in doms:
    cur = M.lag14([d for d in docs if d["dom"] == dom], lags=tuple(K))
    SIG[dom] = np.array([max((cur[k]["delta"][2] - cur[k]["delta"][1]) / 3.92, 2e-3) for k in K])

def perdoc_curve(doc):
    Vn = doc["V"][doc["ok"]]
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
    if len(Vn) <= 14:
        return None
    return np.array([float(np.median([float(Vn[i + k] @ Vn[i]) for i in range(len(Vn) - k)])) for k in K])

def fit_lin(y, w, cols):
    G = np.stack(cols, 1) * w[:, None] ** .5
    AB = np.linalg.lstsq(G, y * w ** .5, rcond=None)[0]
    sse = float(np.sum(w * (y - np.stack(cols, 1) @ AB) ** 2))
    tot = float(np.sum(w * y ** 2))
    return sse, 1 - sse / max(tot, 1e-30)

RG = np.arange(0.02, 0.995, 0.02)
WG = np.arange(0.05, 3.10, 0.02)
out = []
for d in docs:
    y = perdoc_curve(d)
    if y is None:
        continue
    w = 1.0 / SIG[d["dom"]] ** 2
    be, bw = None, None
    for a in RG:
        for b in RG[RG > a + 0.02]:
            sse, r2 = fit_lin(y, w, [a ** K, b ** K, np.ones(n)])
            if be is None or sse < be[0]:
                be = (sse, r2, a, b)
    bo = None
    for om in WG:
        c, s = np.cos(om * K), np.sin(om * K)
        for r in RG:
            rk = r ** K
            sse, r2 = fit_lin(y, w, [rk * c, rk * s, np.ones(n)])
            if bo is None or sse < bo[0]:
                bo = (sse, r2, om, r)
    bic_e = n * math.log(be[0] / n) + 5 * math.log(n)
    bic_o = n * math.log(bo[0] / n) + 5 * math.log(n)
    out.append(dict(name=d["name"], dom=d["dom"], R2_exp=be[1], R2_osc=bo[1],
                    r=[be[2], be[3]], omega=bo[2], r_osc=bo[3],
                    osc_wins=bool(bic_o < bic_e), dBIC=bic_e - bic_o))

med_osc = float(np.median([o["R2_osc"] for o in out]))
med_exp = float(np.median([o["R2_exp"] for o in out]))
win = [o for o in out if o["osc_wins"]]
W = np.array([o["omega"] for o in win]) if win else np.array([])

def gmm_bic(x, k):
    if len(x) < 4 * k:
        return np.inf, 0.0
    mu = np.linspace(x.min(), x.max(), k + 2)[1:-1] if k > 1 else np.array([x.mean()])
    sd = np.full(k, x.std() / max(k, 1) + 1e-6); pk = np.full(k, 1.0 / k)
    for _ in range(120):
        if k == 1:
            L = -np.log(2 * np.pi * sd[0] ** 2) / 2 - ((x - mu[0]) / sd[0]) ** 2 / 2
            ll = float(np.sum(L))
            break
        L = -np.log(2 * np.pi * sd ** 2) / 2 - ((x[:, None] - mu) / sd) ** 2 / 2 + np.log(pk)
        m = L.max(1, keepdims=True)
        r = np.exp(L - m); r /= r.sum(1, keepdims=True)
        pk = r.mean(0)
        mu = (r * x[:, None]).sum(0) / r.sum(0)
        sd = np.sqrt(((r * (x[:, None] - mu) ** 2).sum(0)) / r.sum(0)) + 1e-6
        ll = float(np.sum(m[:, 0] + np.log(np.exp(L - m).sum(1))))
    p = 2 * k + (k - 1)
    return -2 * ll + p * math.log(len(x)), float(pk.min())

bi = {"k": "unimodal"}
bimodal = False
bimodal_lo = False          # чувствительность: только разрешимые ω ≤ 2.0 (T ≥ 3.1)
Wlo = W[W <= 2.0] if len(W) else W
if len(W) >= 12:
    b1, _ = gmm_bic(W, 1)
    b2, wmin = gmm_bic(W, 2)
    bimodal = (b1 - b2) >= 6.0 and wmin >= 0.25
    bi = dict(bic1=float(b1), bic2=float(b2), min_mix_weight=wmin)
if len(Wlo) >= 12:
    b1l, _ = gmm_bic(Wlo, 1)
    b2l, wminl = gmm_bic(Wlo, 2)
    bimodal_lo = (b1l - b2l) >= 6.0 and wminl >= 0.25
    bi["lo"] = dict(n=int(len(Wlo)), bic1=float(b1l), bic2=float(b2l), min_mix_weight=wminl)

if med_osc < 0.5:
    world = "A: hump = артефакт argmax, E-58 обобщается на per-doc"
elif med_osc >= 0.7 and len(W) >= 12:
    if bimodal and not bimodal_lo:
        world = "C: континуум частот; бимодальность только на неразрешимых ω>2 (Найквист-артефакт)"
    elif bimodal and bimodal_lo:
        world = "B: два типа документов (бимодальный ω̂ на разрешимых частотах)"
    else:
        world = "C: континуум частот (унимодальный широкий ω̂)"
elif med_osc >= 0.7:
    world = "C? (R²osc≥0.7, но OSC-побед <12 — GMM не тестировался, унимодальность НЕ утверждается)"
else:
    world = "неопределённо (0.5≤R²osc<0.7)"

res = dict(med_R2_exp=med_exp, med_R2_osc=med_osc,
           osc_win_frac=len(win) / len(out), omega_win=dict(
               n=len(W), med=float(np.median(W)) if len(W) else None,
               p25=float(np.percentile(W, 25)) if len(W) else None,
               p75=float(np.percentile(W, 75)) if len(W) else None),
           gmm=bi, bimodal=bool(bimodal), bimodal_omega_le_2=bool(bimodal_lo), world=world, docs=out)
def sane(o):
    if isinstance(o, dict):  return {k: sane(v) for k, v in o.items()}
    if isinstance(o, list):  return [sane(v) for v in o]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):  return None
    return o
json.dump(sane(res), open(os.path.join(HERE, "..", "results", "exp59_perdoc_osc.json"), "w"),
          indent=1, allow_nan=False)
print(f"med R2: exp={med_exp:.3f} osc={med_osc:.3f} | OSC побеждает {len(win)}/{len(out)} | "
      f"ω̂ med={np.median(W) if len(W) else float('nan'):.2f} [{np.percentile(W,25) if len(W) else float('nan'):.2f},"
      f"{np.percentile(W,75) if len(W) else float('nan'):.2f}] gmm={bi}", flush=True)
print("МИР:", world, flush=True)
