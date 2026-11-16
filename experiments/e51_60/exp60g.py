#!/usr/bin/env python
"""E-60g: реакция на step-back ревью. A' (с исправленным критерием): two-exp+C на остаток
ПЛЮС автокорреляция остатка (зигзаг = не-гладкий → ни одна эксп-модель не опишет, и низкий
R² не будет доказательством emergent). C-пилот (главное): per-position Δ(k): bimodality
(BIC 2-comp vs 1) и доля fast/plateau позиций по доменам."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E
K = np.arange(1, 25); n = 24
ONES = np.ones(n)
DOMS = ['chat', 'code', 'math', 'wiki']
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]
PR = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60d.json")))["params"]

def unit(doc):
    V = doc["V"][doc["ok"]]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

def wls(y, w, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ co) ** 2)), co

def ols(y, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G, y, rcond=None)[0]
    return float(np.sum((y - G @ co) ** 2))

def em12(x, iters=200):
    """1D 2-компонентный EM; BIC против 1-компоненты."""
    x = np.asarray(x, float)
    N = len(x)
    if N < 30:
        return None
    ll1 = np.sum(-0.5 * ((x - x.mean()) / x.std()) ** 2 - np.log(x.std()) - 0.5 * np.log(2 * np.pi))
    bic1 = -2 * ll1 + 2 * math.log(N)
    mu = np.array([np.quantile(x, .25), np.quantile(x, .75)]); sd = np.array([x.std() / 2] * 2)
    pi = np.array([.5, .5])
    for _ in range(iters):
        L = np.stack([np.log(pi[j]) - 0.5 * ((x - mu[j]) / max(sd[j], 1e-6)) ** 2 - np.log(max(sd[j], 1e-6))
                      for j in range(2)])
        m = L.max(0); r = np.exp(L - m); r /= r.sum(0)
        pi = r.mean(1)
        mu = (r * x).sum(1) / r.sum(1)
        sd = np.sqrt(((r * (x - mu[:, None]) ** 2).sum(1)) / r.sum(1))
    ll2 = np.sum(m + np.log(np.exp(L[0] - m) + np.exp(L[1] - m)))
    bic2 = -2 * ll2 + 5 * math.log(N)
    sep = abs(mu[0] - mu[1]) / max(math.sqrt(pi[0] * sd[0] ** 2 + pi[1] * sd[1] ** 2), 1e-9)
    return dict(better2=bool(bic2 < bic1), sep=round(float(sep), 2), w_minor=round(float(min(pi)), 2))

res_acc = {dm: [] for dm in DOMS}; sd_acc = {dm: [] for dm in DOMS}
bimod = {dm: [] for dm in DOMS}; frac = {dm: [] for dm in DOMS}
for di, doc in enumerate(docs):
    nm = doc["name"]; dm = doc["dom"]
    Vn = unit(doc)
    y = np.array([float(np.median(np.einsum("ij,ij->i", Vn[:-k], Vn[k:]))) for k in K])
    P = np.einsum("ij,ij->i", Vn[:-8], Vn[8:])
    b = em12(P)
    if b:
        bimod[dm].append(b)
    s4 = np.einsum("ij,ij->i", Vn[:-4], Vn[4:])[:len(Vn) - 20]
    s20 = np.einsum("ij,ij->i", Vn[:-20], Vn[20:])
    fast = float(np.mean(s4 < np.quantile(s4, .33)))
    plateau = float(np.mean((s4 > np.quantile(s4, .6)) & (s20 > np.quantile(s20, .6))))
    frac[dm].append((fast, plateau))
    # остаток (для A')
    p = PR.get(nm)
    sd = None
    if p is not None:
        sgs = []
        for k in K:
            Pk = np.einsum("ij,ij->i", Vn[:-k], Vn[k:])
            sgs.append(max(float(np.median(Pk[rng.integers(0, len(Pk), (200, len(Pk)))], axis=1).std()), 2e-3))
        sd = np.array(sgs); w = 1.0 / sd ** 2
        if p["m"] == "osc":
            om, r = p["om"], p["r"]; rk = r ** K
            _, cf = wls(y, w, [rk * np.cos(om * K), rk * np.sin(om * K), ONES])
            fit = cf[0] * rk * np.cos(om * K) + cf[1] * rk * np.sin(om * K) + cf[2]
        else:
            a, b2 = p["a"], p["b"]
            _, cf = wls(y, w, [a ** K, b2 ** K, ONES])
            fit = cf[0] * a ** K + cf[1] * b2 ** K + cf[2]
        res_acc[dm].append(y - fit); sd_acc[dm].append(sd)
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)
print("A' остаток: two-exp+C R² и лаг-1 автокорреляция (зигзаг):", flush=True)
out = {"A": {}, "C": {}}
RG2 = np.round(np.arange(0.02, 0.991, 0.02), 3)
for dm in DOMS:
    r = np.mean(res_acc[dm], 0); sdm = np.mean(sd_acc[dm], 0) / 4
    s0 = float(np.sum((r - r.mean()) ** 2))
    be = None
    for a in RG2:
        for b in RG2[RG2 > a + 0.02]:
            sse = ols(r, [a ** K, b ** K, ONES])
            if be is None or sse < be:
                be = sse
    ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    zz = float(np.mean(np.abs(np.diff(np.abs(r) / (sdm / 4)))))
    out["A"][dm] = dict(R2_2exp=round(1 - be / s0, 2), ac1=round(ac, 2))
    print(f"  {dm:<5} R2(2exp+C)={out['A'][dm]['R2_2exp']:5.2f}  ac(лаг1)={ac:+.2f}", flush=True)
print("C-пилот: per-position bimodality лага-8 и доли режимов:", flush=True)
for dm in DOMS:
    bs = bimod[dm]
    nb = sum(b["better2"] for b in bs)
    med_sep = float(np.median([b["sep"] for b in bs]))
    fa = np.mean([x[0] for x in frac[dm]]); pl = np.mean([x[1] for x in frac[dm]])
    out["C"][dm] = dict(bimod_docs=f"{nb}/{len(bs)}", med_sep=round(med_sep, 2),
                        fast_share=round(float(fa), 2), plateau_share=round(float(pl), 2))
    print(f"  {dm:<5} bimodal docs: {nb}/{len(bs)}  med sep={med_sep:.2f}  fast={fa:.2f} plateau={pl:.2f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60g.json"), "w"),
          indent=1, allow_nan=False)
