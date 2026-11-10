#!/usr/bin/env python
"""E-60f: природа доменного остатка real−mean(best-fit): когерентная осцилляция (выживший
дефазинг) или третья экспонента? Для каждого домена: (i) лаговый профиль остатка;
(ii) фит остатка одной экспонентой+C против косинуса с ω*_own+C (BIC равный 3p); (iii) доля
объяснения. Решает (a) дефазинг vs (c) третья компонента."""
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
d60d = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60d.json")))
b60 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60b.json")))
om_own = {dm: b60["B"][f"{dm}_strict"]["omega"] for dm in DOMS}
PR = d60d["params"]

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

def ols(y, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G, y, rcond=None)[0]
    sse = float(np.sum((y - G @ co) ** 2))
    return sse, co

acc = {dm: [[], []] for dm in DOMS}   # [resid list, sd list]
for doc in docs:
    nm = doc["name"]
    if nm not in PR:
        continue
    y, sd = cs(unit(doc))
    p = PR[nm]
    if p["m"] == "osc":
        w = 1.0 / sd ** 2
        om, r = p["om"], p["r"]
        rk = r ** K
        G = np.stack([rk * np.cos(om * K), rk * np.sin(om * K), ONES], 1)
        co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        fit = G @ co
    else:
        w = 1.0 / sd ** 2
        a, b = p["a"], p["b"]
        G = np.stack([a ** K, b ** K, ONES], 1)
        co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        fit = G @ co
    acc[doc["dom"]][0].append(y - fit)
    acc[doc["dom"]][1].append(sd)
print(f"{'dom':<5} R2_exp+C  R2_osc+C  peak_ω  мощность osc-компоненты остатка vs эксп.", flush=True)
out = {}
for dm in DOMS:
    rs = np.array(acc[dm][0]); sd = np.mean(acc[dm][1], 0)
    res = rs.mean(0)
    # фиты остатка (веса σ_mean)
    w = 1.0 / (sd / math.sqrt(len(rs))) ** 2
    om = om_own[dm]
    sseE, _ = ols(res, [0.5 ** K, ONES])   # сетку r не ищем — сравним равные 3p: эксп(0.5 фикс) честно? нет:
    # сделаем мини-грид для обеих моделей, равные 3p
    sseE = min(ols(res, [r ** K, ONES])[0] for r in np.round(np.arange(0.5, 0.99, 0.02), 3))
    sseO = min(ols(res, [r ** K * np.cos(om * K), r ** K * np.sin(om * K), ONES])[0]
               for r in np.round(np.arange(0.5, 0.99, 0.02), 3))
    sse0 = float(np.sum((res - res.mean()) ** 2))
    # периодграмма остатка: мощность на ω* против максимума прочих
    fr = np.fft.rfftfreq(48) * 2 * np.pi
    sig = np.concatenate([res, res[::-1]])  # чётное продолжение
    pw = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    pw[0] = 0
    ip = int(np.argmin(np.abs(fr - om)))
    others = np.delete(pw[1:], ip - 1 if ip > 0 else 0)
    out[dm] = dict(R2exp=round(1 - sseE / sse0, 2), R2osc=round(1 - sseO / sse0, 2),
                   osc_beats_exp=bool(sseO < sseE), peak_at_om=bool(int(np.argmax(pw[1:]) + 1) == ip),
                   ratio_pk_om=float(round(pw[ip] / max(np.max(others), 1e-12), 2)))
    print(f"{dm:<5} {out[dm]['R2exp']:8.2f} {out[dm]['R2osc']:8.2f}   "
          f"osc{'>' if out[dm]['osc_beats_exp'] else '<'}exp  пик на ω*: {out[dm]['peak_at_om']}"
          f"  power(ω*)/max проч = {out[dm]['ratio_pk_om']}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "exp60f.json"), "w"),
          indent=1, allow_nan=False)
