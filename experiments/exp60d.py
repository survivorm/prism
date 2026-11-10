#!/usr/bin/env python
"""E-60d (post-review follow-ups): (0) emergent-vs-smoothing: доменный средний остаток
real−synth против шума; (1) R̄ по всем 16 доменам при своей ω*; (2) состав популяции
(exp-winners их (a,b); osc-winners r̂<0.2 = «осциллятор-предел-затухания») — классы или
континуум; (3) chat sep детально. Всё на RGX/WG, победитель по SSE (равнопараметрично)."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E
K = np.arange(1, 25); n = 24
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
RGX = np.round(np.concatenate([RG, np.arange(0.992, 0.9996, 0.001)]), 4)
RG2 = RG[::2]
ONES = np.ones(n)
DOMS = ['chat', 'code', 'math', 'wiki']
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]
b60 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60b.json")))
om_own = {dm: b60["B"][f"{dm}_strict"]["omega"] for dm in DOMS}

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

fits = {}; params = {}
Y = {d["name"]: None for d in docs}
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
                bo = (sse, cf[0] * rk * c + cf[1] * rk * s + cf[2], om, r_)
    be = None
    for a in RG2:
        for b in RG2[RG2 > a + 0.02]:
            sse, cf = wls(y, w, [a ** K, b ** K, ONES])
            if be is None or sse < be[0]:
                be = (sse, cf[0] * a ** K + cf[1] * b ** K + cf[2], float(a), float(b))
    if bo[0] < be[0]:
        fits[nm] = bo[1]; params[nm] = dict(m="osc", om=bo[2], r=bo[3], dom=doc["dom"])
    else:
        fits[nm] = be[1]; params[nm] = dict(m="exp", a=be[2], b=be[3], dom=doc["dom"])
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)

# (0) остаток
print("(0) Emergent-vs-smoothing: residual = mean(real) − mean(best-fit) по домену", flush=True)
res_tab = {}
for dm in DOMS:
    nmz = [d["name"] for d in docs if d["dom"] == dm]
    resid = np.mean([Y[x][0] for x in nmz], 0) - np.mean([fits[x] for x in nmz], 0)
    sd_m = np.mean([[Y[x][1] for x in nmz]], axis=(0, 1)) / math.sqrt(len(nmz))
    z = np.abs(resid) / np.maximum(sd_m, 1e-4)
    res_tab[dm] = dict(max_z=round(float(z.max()), 1), n_lag_z2=int((z > 2).sum()),
                       resid_med=round(float(np.median(resid)), 4))
    print(f"  {dm:<5} max|z|={res_tab[dm]['max_z']:5.1f}  лагов |z|>2: {res_tab[dm]['n_lag_z2']}/24"
          f"  med resid={res_tab[dm]['resid_med']:+.4f}", flush=True)

# (1) R̄ по всем докам при своей ω*
print("(1) R̄ (все 16) при ω*_own:", flush=True)
ph = {}
for dm in DOMS:
    om_ = om_own[dm]; ps = []
    for doc in docs:
        if doc["dom"] != dm:
            continue
        y, sd = Y[doc["name"]]; w = 1.0 / sd ** 2
        c, s = np.cos(om_ * K), np.sin(om_ * K)
        cand = [wls(y, w, [r ** K * c, r ** K * s, ONES]) for r in RGX]
        cf = min(cand, key=lambda t: t[0])[1]
        ps.append(math.atan2(cf[1], cf[0]))
    ph[dm] = round(float(abs(np.exp(1j * np.array(ps)).mean())), 2)
print("  ", ph, flush=True)

# (2) состав
print("(2) состав: exp-winners (a,b); osc r̂<0.2 доля", flush=True)
comp = {}
for dm in DOMS:
    ps = [p for p in params.values() if p["dom"] == dm]
    osc = [p for p in ps if p["m"] == "osc"]
    exp = [p for p in ps if p["m"] == "exp"]
    comp[dm] = dict(n_osc=len(osc), n_exp=len(exp),
                    osc_r_le02=int(sum(p["r"] <= 0.2 for p in osc)),
                    osc_r_ge099=int(sum(p["r"] >= 0.989 for p in osc)),
                    osc_om_med=round(float(np.median([p["om"] for p in osc])), 2) if osc else None,
                    exp_a_med=round(float(np.median([p["a"] for p in exp])), 2) if exp else None,
                    exp_b_med=round(float(np.median([p["b"] for p in exp])), 2) if exp else None)
print("  ", json.dumps(comp), flush=True)
# эксп-победы math: их (a,b) — может ω просто вне сетки/низкая амплитуда
mx = {k: (round(v["a"], 2), round(v["b"], 2)) for k, v in params.items()
      if v["dom"] == "math" and v["m"] == "exp"}
print("   math exp-winners (a,b):", mx, flush=True)
json.dump(dict(residual=res_tab, phase_all=ph, composition=comp,
               math_exp=mx, params={k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                        for kk, vv in v.items()} for k, v in params.items()}),
          open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60d.json"), "w"),
          indent=1, allow_nan=False)
