#!/usr/bin/env python
"""E-60b (revew follow-ups): A) r-grid→0.9995 для clip-доков: r-boundary или ω-boundary;
B) per-domain ω* (strict и all), фазы при своей ω*, бут CI; C) T3 С INTERCEPT (sensitivity
и для E-58: не экспоненты ли травят floor)."""
import os, sys, json, math, collections
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp53b_servo as E

K = np.arange(1, 25); n = 24
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
RGX = np.round(np.concatenate([RG, np.arange(0.992, 0.9996, 0.001)]), 4)
ONES = np.ones(n)
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]
d60 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60.json")))
om_star = d60["omega_star"]
per60 = {p["name"]: p for p in d60["per_doc"]}

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

Ys, SDs = {}, {}
for doc in docs:
    y, sd = cs(unit(doc))
    Ys[doc["name"]] = y; SDs[doc["name"]] = sd

# --- A: clip-доки на расширенной r-сетке
clip_docs = [nm for nm, p in per60.items() if p["r_at_star"] >= 0.989]
A_res = {}
print(f"A) расширенный r-grid для {len(clip_docs)} clip-доков", flush=True)
for nm in clip_docs:
    y, sd = Ys[nm], SDs[nm]
    w = 1.0 / sd ** 2
    tot = float(np.sum(w * y ** 2))
    best = None
    for om in WG:
        c, s = np.cos(om * K), np.sin(om * K)
        for r_ in RGX:
            rk = r_ ** K
            sse, _ = wls(y, w, [rk * c, rk * s, ONES])
            if best is None or sse < best[0]:
                best = (sse, om, r_)
    # и при ω*=0.6 отдельно
    c, s = np.cos(om_star * K), np.sin(om_star * K)
    bx = None
    for r_ in RGX:
        rk = r_ ** K
        sse, _ = wls(y, w, [rk * c, rk * s, ONES])
        if bx is None or sse < bx[0]:
            bx = (sse, r_)
    A_res[nm] = dict(om_new=float(best[1]), r_new=float(best[2]),
                     interior=bool(0.021 < best[2] < 0.999),
                     r_at_star_new=float(bx[1]),
                     interior_at_star=bool(bx[1] < 0.999))
    print(f"  {nm:<12} ω:{per60[nm]['om_doc']:.2f}→{best[1]:.2f} r:{best[2]:.3f} "
          f"{'ИНТЕРЬЕР' if A_res[nm]['interior'] else 'r>0.999'} | r@ω*={bx[1]:.3f}"
          f"{'(инт)' if A_res[nm]['interior_at_star'] else '(клип)'}", flush=True)
n_int = sum(v["interior"] for v in A_res.values())
n_ints = sum(v["interior_at_star"] for v in A_res.values())
print(f"A) итог: interior при free ω: {n_int}/{len(clip_docs)}; при ω*=0.6: {n_ints}/{len(clip_docs)}", flush=True)

# --- B: per-domain ω* (review v20 фик #1: RGX, не RG — иначе clip-байас наследуется)
R2d = collections.defaultdict(list)
R2dS = collections.defaultdict(list)
R2dRG = collections.defaultdict(list)   # diag: старая сетка для сравнения
grid = {}
for doc in docs:
    nm = doc["name"]; y, sd = Ys[nm], SDs[nm]
    w = 1.0 / sd ** 2
    tot = float(np.sum(w * y ** 2))
    prof = np.zeros(len(WG)); profG = np.zeros(len(WG))
    for j, om in enumerate(WG):
        c, s = np.cos(om * K), np.sin(om * K)
        fits = [wls(y, w, [r_ ** K * c, r_ ** K * s, ONES])[0] for r_ in RGX]
        prof[j] = 1 - min(fits) / tot
        profG[j] = 1 - min(fits[:len(RG)]) / tot
    grid[nm] = prof
    R2d[doc["dom"]].append(prof); R2dRG[doc["dom"]].append(profG)
    if per60[nm]["strict"]:
        R2dS[doc["dom"]].append(prof)
B = {}
rngb = np.random.default_rng(7)
for dm in ['chat', 'code', 'math', 'wiki']:
    for tag, src in (("all", R2d), ("strict", R2dS)):
        M = np.array(src[dm])
        if len(M) < 3:
            continue
        mean = M.mean(0)
        star = int(np.argmax(mean))
        om_d = float(WG[star])
        bs = []
        for _ in range(2000):
            bs.append(int(np.argmax(M[rngb.integers(0, len(M), len(M))].mean(0))))
        bs = np.array(bs)
        lo, hi = float(np.percentile(WG[bs], 2.5)), float(np.percentile(WG[bs], 97.5))
        B[f"{dm}_{tag}"] = dict(omega=float(om_d), ci=[lo, hi], n=len(M))
# diag: что было бы на RG (квант. эффекта фикса #1)
Bdiag = {}
for dm in ['chat', 'code', 'math', 'wiki']:
    M = np.array(R2dRG[dm])
    Bdiag[dm] = float(WG[int(np.argmax(M.mean(0)))])
print("B) per-domain ω*:", {k: (v['omega'], [round(c, 2) for c in v['ci']], v['n'])
      for k, v in B.items()}, flush=True)
print("B-diag ω*(all) на старой RG:", {k: round(v, 2) for k, v in Bdiag.items()}, flush=True)

# фазы при СВОЕЙ доменной ω* (strict-доки)
ph = {}
for dm in ['chat', 'code', 'math', 'wiki']:
    if f"{dm}_strict" not in B:
        continue
    om_d = B[f"{dm}_strict"]["omega"]
    ps = []
    for doc in docs:
        nm = doc["name"]
        if doc["dom"] != dm or not per60[nm]["strict"]:
            continue
        y, sd = Ys[nm], SDs[nm]
        w = 1.0 / sd ** 2
        c, s = np.cos(om_d * K), np.sin(om_d * K)
        cand = [(wls(y, w, [r ** K * c, r ** K * s, ONES]), r) for r in RG]
        fit, r_ = min(cand, key=lambda t: t[0][0])
        cf = fit[1]
        ps.append(math.atan2(cf[1], cf[0]))
    e = np.exp(1j * np.array(ps))
    ph[dm] = float(abs(e.mean())) if len(ps) else None
print("B) R̄ при своей доменной ω*:", {k: round(v, 2) for k, v in ph.items()}, flush=True)

# --- C: T3 с intercept (и E-58 sensitivity)
def shared_exp_r2(target, withC, nlag=24):
    Xs = [target[dm][:nlag] for dm in ['chat', 'code', 'math', 'wiki']]
    Kl = K[:nlag]
    best = None
    for a in RG[::2]:
        for b in RG[RG > a + 0.02][::2]:
            cols = [a ** Kl, b ** Kl] + ([ONES[:nlag]] if withC else [])
            G = np.stack(cols, 1)
            sse = tot = 0.0
            for x in Xs:
                co = np.linalg.lstsq(G, x, rcond=None)[0]
                sse += float(np.sum((x - G @ co) ** 2)); tot += float(np.sum(x ** 2))
            if best is None or sse < best[0]:
                best = (sse, tot, a, b)
    return dict(R2=round(1 - best[0] / best[1], 3), r1=float(best[2]), r2=float(best[3]))

def synth_at(om_key):
    """синтез strict-доков; om_key='star' (ω*=0.6) или 'free' (ω_doc)."""
    out = {}
    for dm in ['chat', 'code', 'math', 'wiki']:
        acc = np.zeros(n); cnt = 0
        for doc in docs:
            nm = doc["name"]
            if doc["dom"] != dm or not per60[nm]["strict"]:
                continue
            om_ = om_star if om_key == "star" else None
            y, sd = Ys[nm], SDs[nm]
            w = 1.0 / sd ** 2
            if om_ is None:   # фикс #2: free ω на RGX (не per60 с RG)
                bfree = None
                for om in WG:
                    cc, ss = np.cos(om * K), np.sin(om * K)
                    for r_ in RGX:
                        sse, cf_ = wls(y, w, [r_ ** K * cc, r_ ** K * ss, ONES])
                        if bfree is None or sse < bfree[0]:
                            bfree = (sse, om, r_, cf_)
                acc += G_eval(bfree[3], bfree[1], bfree[2]); cnt += 1
                continue
            c, s = np.cos(om_ * K), np.sin(om_ * K)
            cand = [(wls(y, w, [r ** K * c, r ** K * s, ONES]), r) for r in RG]
            fit, r_ = min(cand, key=lambda t: t[0][0])
            cf = fit[1]
            rk = r_ ** K
            acc += cf[0] * rk * c + cf[1] * rk * s + cf[2]; cnt += 1
        out[dm] = acc / max(cnt, 1)
    return out

def G_eval(cf, om_, r_):
    rk = r_ ** K
    return cf[0] * rk * np.cos(om_ * K) + cf[1] * rk * np.sin(om_ * K) + cf[2]

real_sub = {dm: np.mean([Ys[doc["name"]] for doc in docs
              if doc["dom"] == dm and per60[doc["name"]]["strict"]], axis=0)
            for dm in ['chat', 'code', 'math', 'wiki']}
real_all = {dm: np.mean([Ys[doc["name"]] for doc in docs if doc["dom"] == dm], axis=0)
            for dm in ['chat', 'code', 'math', 'wiki']}
C = {}
for tag, tgt in (("synth@0.6", synth_at("star")), ("synth_free", synth_at("free")),
                 ("real_subset", real_sub), ("real_all16", real_all)):
    C[tag + "_24lag"] = shared_exp_r2(tgt, True)
    C[tag + "_14lag"] = shared_exp_r2(tgt, True, nlag=14)
C["real_all16_24lag_noC"] = shared_exp_r2(real_all, False)
C["real_all16_14lag_noC"] = shared_exp_r2(real_all, False, nlag=14)
print("C) shared-exp R2 с (r1,r2):", json.dumps(C), flush=True)

out = dict(A=A_res, A_summary=dict(interior_free=n_int, interior_at_star=n_ints,
                                   n_clip=len(clip_docs)),
           B=B, phase_at_own=ph, C=C)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60b.json"), "w"),
          indent=1, allow_nan=False)
