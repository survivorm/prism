#!/usr/bin/env python
"""E-60e: проверка трёх подозрений против E-60c.
P1 сеточная асимметрия: exp-победы считались на RG2 (шаг 0.02), osc — на RGX (0.005..0.01).
   Пересбор победителя при СИММЕТРИЧНЫХ сетках; доля флипов; как двигаются (r1,r2) JOINT.
P2 шиповое доминирование: доля x(1)² в SST (14 лагов) для real/synth — объясняет ли k=1
   выбор вырожденной пары (0.02,0.05) в JOINT-synth.
P3 локализация остатка: лаговый профиль |z| остатка real−fit: если энергия на k≤3 —
   остаток = недостающий шип у osc-победителей (misspecification), не баг."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E
K = np.arange(1, 25); n = 24; NL = 14
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
RGX = np.round(np.concatenate([RG, np.arange(0.992, 0.9996, 0.001)]), 4)
RG2 = RG[::2]
ONES = np.ones(n)
DOMS = ['chat', 'code', 'math', 'wiki']
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]

def unit(doc):
    V = doc["V"][doc["ok"]]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

def cs(Vn, B=200):
    y, sg = [], []
    for k in K:
        P = np.einsum("ij,ij->i", Vn[:-k], Vn[k:])
        y.append(float(np.median(P)))
        sg.append(max(float(np.median(P[rng.integers(0, len(P), (B, len(P)))], axis=1).std()), 2e-3))
    return np.array(y), np.array(sg)

def wls(y, w, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ co) ** 2)), co

pairs2 = [(a, b) for a in RG2 for b in RG2[RG2 > a + 0.02]]
pairsX = [(a, b) for a in RGX for b in RGX[RGX > a + 0.02]]
win60c, winSym = {}, {}
fit60c, fitSym, sseO = {}, {}, {}
Y = {}
for di, doc in enumerate(docs):
    nm = doc["name"]
    y, sd = cs(unit(doc)); w = 1.0 / sd ** 2
    Y[nm] = (y, sd)
    bo = None
    for om in WG:
        c, s = np.cos(om * K), np.sin(om * K)
        for r_ in RGX:
            rk = r_ ** K
            sse, cf = wls(y, w, [rk * c, rk * s, ONES])
            if bo is None or sse < bo[0]:
                bo = (sse, cf[0] * rk * c + cf[1] * rk * s + cf[2])
    sseO[nm] = bo[0]
    b2 = None
    for a, b in pairs2:
        sse, cf = wls(y, w, [a ** K, b ** K, ONES])
        if b2 is None or sse < b2[0]:
            b2 = (sse, cf[0] * a ** K + cf[1] * b ** K + cf[2])
    bx = None
    for a, b in pairsX:
        sse, cf = wls(y, w, [a ** K, b ** K, ONES])
        if bx is None or sse < bx[0]:
            bx = (sse, cf[0] * a ** K + cf[1] * b ** K + cf[2])
    fit60c[nm] = bo[1] if bo[0] < b2[0] else b2[1]
    fitSym[nm] = bo[1] if bo[0] < bx[0] else bx[1]
    win60c[nm] = "osc" if bo[0] < b2[0] else "exp"
    winSym[nm] = "osc" if bo[0] < bx[0] else "exp"
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)

flips = [nm for nm in win60c if win60c[nm] != winSym[nm]]
print(f"P1 флипы победителя (60c-сетки → симметричные): {len(flips)}/64", flips[:12], flush=True)

def joint(tgt):
    best = None
    Kl = K[:NL]
    for a in RG2:
        for b in RG2[RG2 > a + 0.02]:
            G = np.stack([a ** Kl, b ** Kl, ONES[:NL]], 1)
            sse = tot = 0.0
            for dm in DOMS:
                x = tgt[dm][:NL]
                co = np.linalg.lstsq(G, x, rcond=None)[0]
                sse += float(np.sum((x - G @ co) ** 2)); tot += float(np.sum(x ** 2))
            if best is None or sse < best[0]:
                best = (sse, tot, a, b)
    return dict(R2=round(1 - best[0] / best[1], 3), r1=float(best[2]), r2=float(best[3]))

def means(fits):
    out = {}
    for dm in DOMS:
        nmz = [d["name"] for d in docs if d["dom"] == dm]
        out[dm] = (np.mean([Y[x][0] for x in nmz], 0), np.mean([fits[x] for x in nmz], 0),
                   np.mean([Y[x][1] for x in nmz], 0))
    return out

mS = means(fitSym); mC = means(fit60c)
print("P1 JOINT-synth на 60c-сетах:", joint({dm: mC[dm][1] for dm in DOMS}),
      "| на симметричных:", joint({dm: mS[dm][1] for dm in DOMS}),
      "| real:", joint({dm: mS[dm][0] for dm in DOMS}), flush=True)
# P2 шип
for tag, cur in (("real", {dm: mS[dm][0] for dm in DOMS}),
                 ("synthSym", {dm: mS[dm][1] for dm in DOMS})):
    spike = float(np.mean([cur[dm][0] ** 2 / np.sum(cur[dm][:NL] ** 2) for dm in DOMS]))
    print(f"P2 {tag}: доля x(1)² в SST(14) по доменам ≈ {spike:.2f}", flush=True)
# P3 лаговый профиль остатка (симметричные победители)
print("P3 профиль |z| остатка (лаги 1-8 | 9-16 | 17-24; доля энергии остатка k≤3):", flush=True)
for dm in DOMS:
    ym, fm, sdm = mS[dm]
    res = ym - fm
    z = np.abs(res) / (sdm / 4.0)
    e3 = float(np.sum(res[:3] ** 2)) / float(np.sum(res ** 2))
    print(f"  {dm:<5} |z|1-8: {' '.join(f'{v:4.0f}' for v in z[:8])} | max z9-16={z[8:16].max():.1f}"
          f" | max z17-24={z[16:].max():.1f} | энергия k≤3: {e3:.2f}", flush=True)
json.dump(dict(flips=flips, winners_60c=win60c, winners_sym=winSym,
               joint_60c=joint({dm: mC[dm][1] for dm in DOMS}),
               joint_sym=joint({dm: mS[dm][1] for dm in DOMS}),
               joint_real=joint({dm: mS[dm][0] for dm in DOMS})),
          open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60e.json"), "w"),
          indent=1, allow_nan=False)
