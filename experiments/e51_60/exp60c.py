#!/usr/bin/env python
"""E-60c: двухвременность доменного среднего =emergent или =смесь популяций?
Синтез: каждый док — своей лучшей моделью (osc на RGX×WG vs 2-exp), per-domain mean
из 16; shared_exp (с C, (r1,r2), критерий |r1-r2|) на synth и на real, по доменам и joint, 14 лагов."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E
K = np.arange(1, 25); n = 24; NLAG = 14; Kl = K[:NLAG]
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
RGX = np.round(np.concatenate([RG, np.arange(0.992, 0.9996, 0.001)]), 4)
RG2 = RG[::2]
ONES = np.ones(n)
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

synth = {dm: np.zeros(n) for dm in ['chat', 'code', 'math', 'wiki']}
real = {dm: np.zeros(n) for dm in ['chat', 'code', 'math', 'wiki']}
rcount = {dm: 0 for dm in real}
model_used = {}
for di, doc in enumerate(docs):
    dm = doc["dom"]
    y, sd = cs(unit(doc)); w = 1.0 / sd ** 2
    bo = None
    for om in WG:
        c, s = np.cos(om * K), np.sin(om * K)
        for r_ in RGX:
            rk = r_ ** K
            sse, cf = wls(y, w, [rk * c, rk * s, ONES])
            if bo is None or sse < bo[0]:
                bo = (sse, cf[0] * rk * c + cf[1] * rk * s + cf[2])
    be = None
    for a in RG2:
        for b in RG2[RG2 > a + 0.02]:
            sse, cf = wls(y, w, [a ** K, b ** K, ONES])
            if be is None or sse < be[0]:
                be = (sse, cf[0] * a ** K + cf[1] * b ** K + cf[2])
    win = "osc" if bo[0] < be[0] else "exp"
    synth[dm] += bo[1] if win == "osc" else be[1]
    real[dm] += y; rcount[dm] += 1
    model_used[doc["name"]] = win
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)
for dm in real:
    synth[dm] /= rcount[dm]; real[dm] /= rcount[dm]
print("модели:", {m: sum(1 for v in model_used.values() if v == m) for m in ("osc", "exp")},
      flush=True)

def shared_exp_r2(tgt, withC=True):
    best = None
    for a in RG[::2]:
        for b in RG[RG > a + 0.02][::2]:
            cols = [a ** Kl, b ** Kl] + ([ONES[:NLAG]] if withC else [])
            G = np.stack(cols, 1)
            sse = tot = 0.0
            for dm in real:
                x = tgt[dm][:NLAG]
                co = np.linalg.lstsq(G, x, rcond=None)[0]
                sse += float(np.sum((x - G @ co) ** 2)); tot += float(np.sum(x ** 2))
            if best is None or sse < best[0]:
                best = (sse, tot, a, b)
    return dict(R2=round(1 - best[0] / best[1], 3), r1=float(best[2]), r2=float(best[3]),
                sep=round(abs(best[2] - best[3]), 2))

out = {}
for dm in real:
    t1 = {d: (synth[dm] if d == dm else np.zeros(n)) for d in real}
    out[dm] = dict(synth=shared_exp_r2(synth), real=shared_exp_r2(real)) if False else None
# per-domain по отдельности (каждый домен сам по себе, shared (r1,r2) внутри домена trivially)
def shared_one(curve):
    best = None
    for a in RG[::2]:
        for b in RG[RG > a + 0.02][::2]:
            G = np.stack([a ** Kl, b ** Kl, ONES[:NLAG]], 1)
            x = curve[:NLAG]
            co = np.linalg.lstsq(G, x, rcond=None)[0]
            sse = float(np.sum((x - G @ co) ** 2)); tot = float(np.sum(x ** 2))
            if best is None or sse < best[0]:
                best = (sse, tot, a, b)
    return dict(R2=round(1 - best[0] / best[1], 3), r1=float(best[2]), r2=float(best[3]),
                sep=round(abs(best[2] - best[3]), 2))
for dm in real:
    out[dm] = dict(synth=shared_one(synth[dm]), real=shared_one(real[dm]))
out["JOINT"] = dict(synth=shared_exp_r2(synth), real=shared_exp_r2(real))
print(json.dumps(out, indent=1), flush=True)
json.dump(dict(models=model_used, res=out),
          open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60c.json"), "w"),
          indent=1, allow_nan=False)
