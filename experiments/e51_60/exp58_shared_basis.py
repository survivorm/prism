#!/usr/bin/env python
"""E-58: shared-basis тест (docs/E58.md, правила зафиксированы ДО фита).
Т1: WLS joint: общие r1,r2 + per-domain A,B (линейные — решаются точно на каждой сетке r).
   против per-domain полных (A,B,r1,r2). Т2: базис из math → у wiki только A,B.
   Т3: wiki r=0.031 фикс, одна амплитуда."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
K = np.arange(1, 15)
docs = [E.derive(d) for d in E.load_real()]
doms = sorted({d["dom"] for d in docs})
Y, W = {}, {}
for dom in doms:
    dd = [d for d in docs if d["dom"] == dom]
    cur = M.lag14(dd, lags=tuple(K))
    y = np.array([cur[k]["delta"][0] for k in K])
    s = np.array([max((cur[k]["delta"][2] - cur[k]["delta"][1]) / 3.92, 2e-3) for k in K])
    Y[dom], W[dom] = y, 1.0 / s ** 2

def solve_AB(y, w, r1, r2):
    G = np.stack([r1 ** K, r2 ** K], 1)
    AB = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    res = y - G @ AB
    sse = float(np.sum(w * res ** 2))
    tot = float(np.sum(w * y ** 2))
    return AB, sse, 1 - sse / max(tot, 1e-30)

# --- сетка по (r1<r2); A,B линейно точны ---
grid = np.arange(0.01, 0.995, 0.005)
best_shared, best_pd = None, {}
for a in grid:
    for b in grid[grid > a + 0.02]:
        tot = 0.0
        r2d = {}
        for dom in doms:
            AB, sse, _ = solve_AB(Y[dom], W[dom], a, b)
            tot += sse
            r2d[dom] = (AB, sse)
        if best_shared is None or tot < best_shared[0]:
            best_shared = (tot, a, b, r2d)   # r2d — свежий dict на каждой итерации b, ссылка безопасна
for dom in doms:
    bb = None
    for a in grid:
        for b in grid[grid > a + 0.02]:
            AB, sse, r2 = solve_AB(Y[dom], W[dom], a, b)
            if bb is None or sse < bb[0]:
                bb = (sse, a, b, AB, r2)
    best_pd[dom] = bb

n = 4 * len(K)
sse_sh = best_shared[0]
sse_pd = sum(v[0] for v in best_pd.values())
bic_sh = n * np.log(sse_sh / n) + 10 * np.log(n)
bic_pd = n * np.log(sse_pd / n) + 16 * np.log(n)
r2_sh = {d: solve_AB(Y[d], W[d], best_shared[1], best_shared[2])[2] for d in doms}
rat = {d: float(best_shared[3][d][0][1] / best_shared[3][d][0][0]) for d in doms}

# --- Т2: базис из math per-domain фита ---
r1m, r2m = best_pd["math"][1], best_pd["math"][2]
ABw, ssew, r2w2 = solve_AB(Y["wiki"], W["wiki"], r1m, r2m)
# --- Т3: wiki r=0.031 фикс, одна мода ---
G1 = (0.031 ** K)[:, None]
a1 = np.linalg.lstsq(G1 * W["wiki"][:, None] ** .5, Y["wiki"] * W["wiki"] ** .5, rcond=None)[0]
r3 = 1 - float(np.sum(W["wiki"] * (Y["wiki"] - G1[:, 0] * a1[0]) ** 2)) / float(np.sum(W["wiki"] * Y["wiki"] ** 2))

Rw = r2_sh["wiki"]
ext = abs(rat["wiki"]) > 3 or abs(rat["wiki"]) < 0.5   # пред-рег E58.md: обычные веса 0.5–3 (review v16)
if Rw > 0.85 and not ext:
    v = "wiki в общем базисе, обычные веса — E-57 DIFFERENT ложен"
elif Rw > 0.85 and ext:
    v = "shared basis + extreme weights (сценарий пользователя)"
elif Rw >= 0.5:
    v = "промежуточный: тот же базис, сдвинутые параметры"
else:
    v = "wiki действительно другой механизм"

out = dict(shared=dict(r1=float(best_shared[1]), r2=float(best_shared[2]),
                       AB={d: [float(x) for x in best_shared[3][d][0]] for d in doms},
                       ratio_BA=rat, R2=r2_sh, sse=sse_sh),
           per_domain={d: dict(r=[float(best_pd[d][1]), float(best_pd[d][2])],
                               AB=[float(x) for x in best_pd[d][3]], R2=float(best_pd[d][4]))
                       for d in doms},
           bic=dict(shared=float(bic_sh), per_domain=float(bic_pd), prefer="shared" if bic_sh < bic_pd else "per_domain"),
           t2_math_basis=dict(r=[float(r1m), float(r2m)], AB=[float(x) for x in ABw], R2=float(r2w2)),
           t3_wiki_r031=dict(R2=float(r3)),
           verdict=v)
print(json.dumps({k: out[k] for k in ("shared", "per_domain", "bic", "t2_math_basis", "t3_wiki_r031")},
                 indent=1, ensure_ascii=False), flush=True)
print("ВЕРДИКТ:", v, flush=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "exp58_shared_basis.json"), "w"), indent=1)
