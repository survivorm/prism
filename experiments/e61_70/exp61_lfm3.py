#!/usr/bin/env python
"""E-61: переносимость ключевых фактов E-58/E-60 на LFM3 (64 дока, 24 лага).
Факт-за-фактом против Qwen-базлайна: (1) доменное среднее two-timescale (r1,r2,C,14лаг);
(2) состав osc/exp; (3) strict-ident; (4) R̄ при ω*_strict own; (5) clip r≥0.989."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
K = np.arange(1, 25); n = 24; NL = 14
WG = np.round(np.arange(0.10, 2.001, 0.04), 3)
RG = np.round(np.arange(0.02, 0.991, 0.01), 3)
RGX = np.round(np.concatenate([RG, np.arange(0.992, 0.996, 0.002)]), 4)
RG2 = RG[::2]
ONES = np.ones(n)
DOMS = ['chat', 'code', 'math', 'wiki']
rng = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))
docs = json.load(open(os.path.join(HERE, "..", "results", "exp48b_lfm3_gen_kinematics.json")))["docs"]
z = np.load(os.path.join(HERE, "..", "results", "exp48b_lfm3_gen_kinematics.npz"), allow_pickle=True)

def key(d):
    return f"{d['dom']}_{d['si']}"

def wls(y, w, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ co) ** 2)), co

pairs2 = [(a, b) for a in RG2 for b in RG2[RG2 > a + 0.02]]
Ys, SDs, win, om_r = {}, {}, {}, {}
for di, d in enumerate(docs):
    nm = key(d)
    u = z[nm + "_u"].astype(np.float64)[:384]
    hn = np.asarray(z[nm].item()["hnorm"], dtype=float)[:384]
    U = u * hn[:, None]
    U /= np.linalg.norm(U, axis=1, keepdims=True)          # ровно exp53b.load_real
    T = U.shape[0]
    Vo = np.zeros((T - 1, U.shape[1]))
    for t in range(T - 1):
        Vo[t] = U[t + 1] - float(U[t + 1] @ U[t]) * U[t]
    okm = np.linalg.norm(Vo, axis=1) > 1e-6                 # ровно exp53b.derive
    V = Vo[okm]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    y, sg = [], []
    for k in K:
        P = np.einsum("ij,ij->i", V[:-k], V[k:])
        y.append(float(np.median(P)))
        sg.append(max(float(np.median(P[rng.integers(0, len(P), (200, len(P)))], axis=1).std()), 2e-3))
    y = np.array(y); sd = np.array(sg); w = 1.0 / sd ** 2
    Ys[nm] = y; SDs[nm] = sd
    tot = float(np.sum(w * y ** 2))
    best = None
    for om in WG:
        c, s = np.cos(om * K), np.sin(om * K)
        for r_ in RGX:
            rk = r_ ** K
            sse, _ = wls(y, w, [rk * c, rk * s, ONES])
            if best is None or sse < best[0]:
                best = (sse, om, r_)
    be = None
    for a, b in pairs2:
        sse, _ = wls(y, w, [a ** K, b ** K, ONES])
        if be is None or sse < be[0]:
            be = (sse, a, b)
    win[nm] = "osc" if best[0] < be[0] else "exp"
    om_r[nm] = (float(best[1]), float(best[2]), float(1 - best[0] / tot), d["dom"])
    if di % 16 == 15:
        print(f"...{di+1}", flush=True)

real = {dm: np.mean([Ys[key(d)] for d in docs if d["dom"] == dm], 0) for dm in DOMS}

def shared_exp(tgt, nlag=NL):
    Kl = K[:nlag]; be = None
    for a in RG2:
        for b in RG2[RG2 > a + 0.02]:
            G = np.stack([a ** Kl, b ** Kl, ONES[:nlag]], 1)
            sse = tot = 0.0
            for dm in DOMS:
                x = tgt[dm][:nlag]
                co = np.linalg.lstsq(G, x, rcond=None)[0]
                sse += float(np.sum((x - G @ co) ** 2)); tot += float(np.sum(x ** 2))
            if be is None or sse < be[0]:
                be = (sse, tot, a, b)
    return dict(R2=round(1 - be[0] / be[1], 3), r1=float(be[2]), r2=float(be[3]))

F1 = shared_exp(real)
n_osc = sum(1 for v in win.values() if v == "osc")
clip = sum(1 for nm, v in om_r.items() if v[1] >= 0.989 and win[nm] == "osc")
strict = {}
for d in docs:
    nm = key(d)
    if win[nm] != "osc":
        continue
    y, sd = Ys[nm], SDs[nm]; w = 1.0 / sd ** 2
    tot = float(np.sum(w * y ** 2))
    prof = []
    for om in WG:
        c, s = np.cos(om * K), np.sin(om * K)
        prof.append(min(wls(y, w, [r ** K * c, r ** K * s, ONES])[0] for r in RGX))
    prof = np.array(prof); p = int(np.argmin(prof))
    dch = (prof - prof.min()) * tot
    l_ = p
    while l_ > 0 and dch[l_ - 1] <= 1.0:
        l_ -= 1
    r_ = p
    while r_ < len(dch) - 1 and dch[r_ + 1] <= 1.0:
        r_ += 1
    om_, r_om, _, dm = om_r[nm]
    strict[nm] = dict(dom=dm, om=om_, r=r_om,
                      ok=bool(p > 0 and p < len(WG) - 1 and r_om < 0.989 and (WG[r_] - WG[l_]) < 0.6))
ns_ok = sum(1 for v in strict.values() if v["ok"])
out_ph = {}
for dm in DOMS:
    sel = [nm for nm, v in strict.items() if v["dom"] == dm and v["ok"]]
    if len(sel) < 3:
        out_ph[dm] = None
        continue
    prof_dm = np.zeros(len(WG))
    for nm in sel:
        y, sd = Ys[nm], SDs[nm]; w = 1.0 / sd ** 2
        tot = float(np.sum(w * y ** 2))
        for j, om in enumerate(WG):
            c, s = np.cos(om * K), np.sin(om * K)
            prof_dm[j] += (tot - min(wls(y, w, [r ** K * c, r ** K * s, ONES])[0] for r in RGX)) / tot
    star = float(WG[int(np.argmax(prof_dm))])
    ps = []
    for d in docs:
        if d["dom"] != dm:
            continue
        nm = key(d)
        y, sd = Ys[nm], SDs[nm]; w = 1.0 / sd ** 2
        c, s = np.cos(star * K), np.sin(star * K)
        fit, r = min((wls(y, w, [r ** K * c, r ** K * s, ONES]), r) for r in RGX)
        ps.append(math.atan2(fit[1][1], fit[1][0]))
    out_ph[dm] = dict(omega=star, Rbar=round(float(abs(np.exp(1j * np.array(ps)).mean())), 2),
                      n_strict=len(sel))
print("LFM3 доменное среднее (14лаг):", F1, flush=True)
print(f"LFM3: osc {n_osc}/64 | clip r≥0.989: {clip} | strict: {ns_ok}", flush=True)
print("LFM3 ω*_own, R̄, n_strict:",
      {k: v and (v['omega'], v['Rbar'], v['n_strict']) for k, v in out_ph.items()}, flush=True)
print("Qwen: (0.913, r=0.16/0.97) | osc 45/64 | clip 14 | strict 15 | R̄ .79/.58/.78/.46", flush=True)
json.dump(dict(shared_exp_real=F1, n_osc=n_osc, clip=clip, n_strict=ns_ok, phase=out_ph,
               winners=win, om_r={k: list(v) for k, v in om_r.items()}),
          open(os.path.join(HERE, "..", "results", "exp61_lfm3.json"), "w"), indent=1, allow_nan=False)
