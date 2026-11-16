#!/usr/bin/env python
"""E-57: per-domain анализ по предзарегистрированным критериям (docs/E57.md).
SHAPE (cos по k∈3..16), SHIFT (bootstrap-CI медиан hump_pos), AMBER — итог-вердикт."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
docs = [E.derive(d) for d in E.load_real()]
doms = sorted({d["dom"] for d in docs})
KS = tuple(range(3, 17))

def perdoc(doc):
    Vn = doc["V"][doc["ok"]]
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
    if len(Vn) <= 16:
        return None
    r = {k: float(np.median([float(Vn[i + k] @ Vn[i]) for i in range(len(Vn) - k)])) for k in KS}
    return float(np.median([float(Vn[i + 1] @ Vn[i]) for i in range(len(Vn) - 1)])), r

res = {"rules": "SHAPE: все 6 cos>0.90; SHIFT: пара с непересекающимися CI и разн. медиан>=3; иначе AMBER"}
sh = {}
res["domain"] = {}
for dom in doms:
    dd = [d for d in docs if d["dom"] == dom]
    cur = M.lag14(dd)
    curK = M.lag14(dd, lags=tuple(range(1, 15)))       # kung: последовательные 1..14 (Hankel-гейт)
    curE = M.lag14(dd, lags=KS)                      # форма на плотной сетке 3..16
    y = [curK[k]["delta"][0] for k in range(1, 15)]
    s = [max((curK[k]["delta"][2] - curK[k]["delta"][1]) / 3.92, 2e-3) for k in range(1, 15)]
    sh[dom] = np.array([curE[k]["delta"][0] for k in KS]); sh[dom] /= max(np.linalg.norm(sh[dom]), 1e-12)
    pr = [perdoc(d) for d in dd]
    pr = [(a, r) for a, r in pr if r is not None]
    hps = np.array([max(r, key=r.get) for _, r in pr])
    boot = hps[rng.integers(0, len(hps), (2000, len(hps)))]
    med = np.median(boot, axis=1)
    jg, j, js, jx = M.j_paired(dd)
    res["domain"][dom] = dict(
        kung=M.kung_fit(y, s, lags=tuple(range(1, 15))),
        hump_med=float(np.median(hps)),
        hump_ci=[float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
        no_dip_frac=float(np.mean([a >= -0.05 for a, _ in pr])),
        j=dict(j_glob=jg, j=j, j_shuf=js, J_excess=jx))
    print(dom, {k: res["domain"][dom][k] for k in ("hump_med", "hump_ci", "no_dip_frac")}, flush=True)

cos = {}
for i, a in enumerate(doms):
    for b in doms[i + 1:]:
        cos[f"{a}-{b}"] = float(sh[a] @ sh[b])
res["shape_cos"] = cos
pairs = []
for i, a in enumerate(doms):
    for b in doms[i + 1:]:
        ca, cb = res["domain"][a]["hump_ci"], res["domain"][b]["hump_ci"]
        dm = abs(res["domain"][a]["hump_med"] - res["domain"][b]["hump_med"])
        pairs.append((a, b, dm, ca[1] < cb[0] or cb[1] < ca[0]))
shift = [p for p in pairs if p[2] >= 3 and p[3]]
res["shift_pairs"] = [f"{a}-{b} d={d} sep={sep}" for a, b, d, sep in pairs if d >= 3]
# приоритет при конфликте зафиксирован (review v15): SHIFT главнее SHAPE, но их пересечение
# — отдельный исход SHIFTED_SAME_SHAPE (форма та же, пик сдвинут)
if shift and all(c > 0.90 for c in cos.values()):
    v = "SHIFTED_SAME_SHAPE"
elif shift:
    v = "DIFFERENT morphologies"
elif all(c > 0.90 for c in cos.values()):
    v = "SAME mechanism, different amplitudes"
else:
    v = "AMBER: one mechanism with floating parameters"
res["verdict"] = v
print("cos:", {k: round(x, 3) for k, x in cos.items()}, flush=True)
print("SHIFT пары:", res["shift_pairs"], flush=True)
print("ВЕРДИКТ:", v, flush=True)
json.dump(res, open(os.path.join(HERE, "..", "results", "exp57_perdomain.json"), "w"), indent=1, default=float)
