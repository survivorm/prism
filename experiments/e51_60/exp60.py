#!/usr/bin/env python
"""E-60 (docs/E60.md + аддендум): fix-ω sweep — общая ω или per-doc.
T1: сетка ω×r, per-doc R²; ω* = argmax среднего; Δ_doc = R²(ω*_doc)−R²(ω*) — мир по strict-osc.
T2: r* при ω*. T3: синтез per-doc осцилляторов при ω* против shared-exponential на тех же
подмножествах. Фаза: R̄=|mean e^{iφ}|. Guard: cond(design√w). Инварианты assert'ами."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
K = np.arange(1, 25)
n = len(K)
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
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

def wls_fit(y, w, cols):
    G = np.stack(cols, 1)
    A = G * w[:, None] ** .5
    coef = np.linalg.lstsq(A, y * w ** .5, rcond=None)[0]
    sse = float(np.sum(w * (y - G @ coef) ** 2))
    return sse, coef

R2 = np.zeros((len(docs), len(WG), len(RG)))
R2e = np.zeros((len(docs),))
Ys, SDs, CONDB = [], [], []
for di, d in enumerate(docs):
    Vn = unit(d)
    y, sd = curve_and_sigma(Vn)
    w = 1.0 / sd ** 2
    tot = float(np.sum(w * y ** 2))
    for j, om in enumerate(WG):
        c, s = np.cos(om * K), np.sin(om * K)
        for m, r in enumerate(RG):
            rk = r ** K
            sse, _ = wls_fit(y, w, [rk * c, rk * s, ONES])
            R2[di, j, m] = 1 - sse / tot
    best = None
    for a in RG:
        for b in RG[RG > a + 0.02]:
            sse, _ = wls_fit(y, w, [a ** K, b ** K, ONES])
            if best is None or sse < best[0]:
                best = (sse, a, b)
    R2e[di] = 1 - best[0] / tot
    jm, mm = np.unravel_index(np.argmax(R2[di]), R2[di].shape)
    rk = RG[mm] ** K
    _, cf = wls_fit(y, w, [rk * np.cos(WG[jm] * K), rk * np.sin(WG[jm] * K), ONES])
    G = np.stack([rk * np.cos(WG[jm] * K), rk * np.sin(WG[jm] * K), ONES], 1) * w[:, None] ** .5
    CONDB.append(bool(np.linalg.cond(G) > 1e8))
    Ys.append(y); SDs.append(sd)
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)

# свободные оценки и строгость (из той же сетки; CI — Δχ²≤1 по профилю max_r с шагом 1/tot)
free, strict, osc_win = [], [], []
for di, d in enumerate(docs):
    jm, mm = np.unravel_index(np.argmax(R2[di]), R2[di].shape)
    tot = float(np.sum((1 / SDs[di] ** 2) * Ys[di] ** 2))
    prof = R2[di].max(1)                       # max по r при каждом ω
    dch = tot * (prof.max() - prof)
    p = int(np.argmin(dch))
    l_ = p
    while l_ > 0 and dch[l_ - 1] <= 1.0:
        l_ -= 1
    r_ = p
    while r_ < len(dch) - 1 and dch[r_ + 1] <= 1.0:
        r_ += 1
    w_ = dict(name=d["name"], dom=d["dom"], om=float(WG[jm]), r=float(RG[mm]),
              R2=float(R2[di].max()), ci_w=float(WG[r_] - WG[l_]),
              clip=bool(l_ == 0 or r_ == len(dch) - 1), cond_bad=CONDB[di])
    osc_win.append(bool(R2[di].max() > R2e[di]))
    strict.append(bool(osc_win[-1] and not w_["clip"] and w_["r"] < 0.989
                       and WG[0] < WG[jm] < WG[-1] and w_["ci_w"] < 0.6))
    free.append(w_)
print(f"winners {sum(osc_win)}/64 | strict {sum(strict)} | cond_bad {sum(CONDB)}", flush=True)

# T1: ω* — ГЛАВНАЯ по strict-множеству (дыра аддендумa v1: среднее по всем 64 = E-56
# contamination, ω*=1.84 ушла в шум не-осцилляторов). По всем 64 — диагностик.
R2_mean_all = R2.max(2).mean(0)
R2_mean_str = R2.max(2)[np.array(strict)].mean(0)
star = int(np.argmax(R2_mean_str))
om_star = float(WG[star])
om_star_all = float(WG[int(np.argmax(R2_mean_all))])
# инвариант: R²_free ≥ R²(ω*) (та же сетка) — assert по каждому доку
per = []
for di in range(len(docs)):
    prof_star = R2[di, star].max()
    best_doc = R2[di].max()
    jd, md = np.unravel_index(np.argmax(R2[di]), R2[di].shape)
    assert best_doc >= prof_star - 1e-12, f"нарушение инварианта на {docs[di]['name']}"
    per.append(dict(name=docs[di]["name"], dom=docs[di]["dom"], delta=float(best_doc - prof_star),
                    om_doc=float(WG[jd]), r_at_star=float(RG[int(np.argmax(R2[di, star]))]),
                    strict=bool(strict[di])))
S = [p for p in per if p["strict"]]
dlt = np.array([p["delta"] for p in S]); omS = np.array([p["om_doc"] for p in S])
med, p90 = float(np.median(dlt)), float(np.percentile(dlt, 90))
iqr = float(np.percentile(omS, 75) - np.percentile(omS, 25))
if med > 0.15:
    world = "H-ω REFUTED (median Δ>0.15): одна частота невозможна"
elif med < 0.02 and p90 < 0.05 and iqr < 0.4:
    world = f"D: общая ω≈{om_star:.2f}" + (" (остро, IQR<0.2)" if iqr < 0.2 else "")
elif med > 0.05 or p90 > 0.15 or iqr > 0.8:
    world = "B': per-doc ω"
else:
    world = "D-mixed (частичное разделение)"

# T2: r при ω*
rstar_vals = np.array([p["r_at_star"] for p in per])
t2 = dict(clip_frac_all=float(np.mean(rstar_vals >= 0.989)),
          clip_frac_strict=float(np.mean([p["r_at_star"] >= 0.989 for p in S])))

# T3: синтез при ω* на strict-подмножествах vs shared-exponential (E-58 метод, без C)
doms = sorted({p["dom"] for p in S})
synth, real = {}, {}
phi_bar = {}
for dm in doms:
    ps = [p for p in S if p["dom"] == dm]
    acc = np.zeros(n)
    ph = []
    # параметры возьмём из сетки: r*, coef при (ω*, r*)
    for p in ps:
        di = next(i for i in range(len(docs)) if docs[i]["name"] == p["name"])
        w_ = 1.0 / SDs[di] ** 2
        rm = p["r_at_star"]
        rk = rm ** K
        c, s = np.cos(om_star * K), np.sin(om_star * K)
        sse, cf = wls_fit(Ys[di], w_, [rk * c, rk * s, ONES])
        acc += (cf[0] * rk * c + cf[1] * rk * s + cf[2])
        ph.append(math.atan2(cf[1], cf[0]))
    synth[dm] = acc / max(len(ps), 1)
    real[dm] = np.mean([Ys[i] for i in range(len(docs))
                        if docs[i]["name"] in {x["name"] for x in ps}], axis=0)
    e = np.exp(1j * np.array(ph))
    phi_bar[dm] = float(abs(e.mean())) if len(ph) else None

def shared_exp_r2(target):
    Xs = [target[dm] for dm in doms]
    best = None
    for a in RG[::2]:
        for b in RG[RG > a + 0.02][::2]:
            G = np.stack([a ** K, b ** K], 1)
            sse = tot = 0.0
            for x in Xs:
                co = np.linalg.lstsq(G, x, rcond=None)[0]
                sse += float(np.sum((x - G @ co) ** 2)); tot += float(np.sum(x ** 2))
            if best is None or sse < best[0]:
                best = (sse, tot, a, b)
    return 1 - best[0] / best[1], float(best[2]), float(best[3])

r2s, r1s, r2sr = shared_exp_r2(synth)
r2r, _, _ = shared_exp_r2(real)
t3 = dict(R2_synth=float(r2s), R2_real_subset=float(r2r), r=[r1s, r2sr],
          verdict=("E-58/E-59 совместимы (synth объясняется экспонентами)" if r2s > 0.85
                   else "synth НЕ сводится к экспонентам — структура сверх de-phasing" if r2s < 0.5
                   else "промежуточно"))

res = dict(omega_star=om_star, omega_star_all64=om_star_all, world=world,
           delta=dict(med=med, p90=p90, iqr_om=iqr, n_strict=len(S)),
           t2=t2, t3=t3, phase_R_bar=phi_bar, R2mean_curve=[float(x) for x in R2_mean_str],
           R2mean_curve_all64=[float(x) for x in R2_mean_all],
           per_doc=per)
print(f"ω*={om_star:.2f} | strict Δ: med={med:.3f} p90={p90:.3f} IQR(ω*)={iqr:.2f} n={len(S)}", flush=True)
print("T2 clip@ω*:", {k: round(v, 2) for k, v in t2.items()},
      "| T3:", t3["verdict"], f"R2s={t3['R2_synth']:.3f} R2r={t3['R2_real_subset']:.3f}", flush=True)
print("фазы R̄:", {k: round(v, 2) for k, v in phi_bar.items()}, flush=True)
print("МИР:", world, flush=True)

def sane(o):
    if isinstance(o, dict):  return {k: sane(v) for k, v in o.items()}
    if isinstance(o, list):  return [sane(v) for v in o]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):  return None
    return o
json.dump(sane(res), open(os.path.join(HERE, "..", "results", "exp60.json"), "w"), indent=1, allow_nan=False)
