#!/usr/bin/env python
"""E-60h: СТРОГИЙ C — per-position бимодальность лага-8 против двух известных ловушек:
(i) N-инфляция коррелированных позиций → тест на прореженных (stride 8) позициях;
(ii) EM-BIC режет скошенные unimodal → параметрический null: skew-normal с теми же
моментами, 200 симуляций, эмпирический p; плюс split-half сходимость кластеров
(настоящий критерий «режимы», а не «тест любит данные»)."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E
from scipy.stats import skewnorm
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]
DOMS = ['chat', 'code', 'math', 'wiki']

def unit(doc):
    V = doc["V"][doc["ok"]]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

def em2(x, seed=0):
    x = np.asarray(x, float)
    N = len(x)
    r = np.random.default_rng(seed)
    mu = np.array([x.min() + .25 * (x.max() - x.min()), x.min() + .75 * (x.max() - x.min())])
    sd = np.array([(x.max() - x.min()) / 4] * 2); pi = np.array([.5, .5])
    for _ in range(300):
        L = np.stack([np.log(pi[j] + 1e-12) - .5 * ((x - mu[j]) / max(sd[j], 1e-9)) ** 2
                      - np.log(max(sd[j], 1e-9)) for j in range(2)])
        m = L.max(0); r_ = np.exp(L - m); r_ /= r_.sum(0)
        pi = r_.mean(1)
        mu = (r_ * x).sum(1) / r_.sum(1)
        sd = np.sqrt((r_ * (x - mu[:, None]) ** 2).sum(1) / r_.sum(1))
    ll2 = np.sum(m + np.log(np.exp(L[0] - m) + np.exp(L[1] - m)))
    g1 = float(((x - x.mean()) ** 3).mean() / x.std() ** 3)
    ll1 = np.sum(skewnorm.logpdf(x, g1 if abs(g1) > 1e-3 else 0.0, x.mean(), x.std()))
    bic1 = -2 * ll1 + 3 * math.log(N)      # skew-normal: 3 параметра — честный unimodal-null
    bic2 = -2 * ll2 + 5 * math.log(N)
    sep = abs(mu[0] - mu[1]) / max(math.sqrt(pi[0] * sd[0] ** 2 + pi[1] * sd[1] ** 2), 1e-9)
    return bic2 - bic1, sep, mu, sd, pi

out = {dm: [] for dm in DOMS}
for di, doc in enumerate(docs):
    Vn = unit(doc)
    P = np.einsum("ij,ij->i", Vn[:-8], Vn[8:])
    Pt = P[::8]                                  # прореживание: N≈46
    dbic, sep, *_ = em2(Pt, seed=di)
    # skew-normal null с моментами Pt: p = доля симуляций с dbic ≤ наблюдённого
    g1 = float(((Pt - Pt.mean()) ** 3).mean() / Pt.std() ** 3)
    sims = 0; n_sim = 200
    worse = 0
    for s in range(n_sim):
        x = skewnorm.rvs(g1 if abs(g1) > 1e-3 else 0.0, Pt.mean(), Pt.std(), size=len(Pt),
                         random_state=rng.integers(1 << 30))
        if em2(x, seed=s)[0] <= dbic:
            worse += 1
    p_sim = (worse + 1) / (n_sim + 1)
    # split-half: режимы должны воспроизводиться на независимых половинах
    idx = rng.permutation(len(Pt))
    h1, h2 = Pt[idx[::2]], Pt[idx[1::2]]
    _, _, mu1, sd1, pi1 = em2(h1, seed=di)
    _, _, mu2, sd2, pi2 = em2(h2, seed=di + 1)
    def assign(x, mu, sd, pi):
        L = np.stack([np.log(pi[j]) - .5 * ((x - mu[j]) / max(sd[j], 1e-9)) ** 2 - np.log(sd[j])
                      for j in range(2)])
        return L.argmax(0)
    a1, a2 = assign(Pt, mu1, sd1, pi1), assign(Pt, mu2, sd2, pi2)
    agr = float(max((a1 == a2).mean(), 1 - (a1 == a2).mean()))  # с поправкой на пермутацию меток
    out[doc["dom"]].append(dict(dbic=round(float(dbic), 1), sep=round(float(sep), 2),
                                p_sim=p_sim, agree=round(agr, 2), n=len(Pt)))
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)
print(f"{'dom':<5} p_sim<0.05  sep(med)  agree(med)  n_bimod+agree")
res = {}
for dm in DOMS:
    o = out[dm]
    nb = sum(1 for r in o if r["p_sim"] < 0.05)
    na = sum(1 for r in o if r["p_sim"] < 0.05 and r["agree"] >= 0.8)
    res[dm] = dict(p05=f"{nb}/16", sep=round(float(np.median([r['sep'] for r in o])), 2),
                   agree=round(float(np.median([r['agree'] for r in o])), 2), strict=f"{na}/16")
    print(f"{dm:<5} {res[dm]['p05']:>9} {res[dm]['sep']:9.2f} {res[dm]['agree']:10.2f} {res[dm]['strict']:>12}",
          flush=True)
json.dump(dict(per_doc=out, summary=res),
          open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60h.json"), "w"),
          indent=1, allow_nan=False)
