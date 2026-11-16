#!/usr/bin/env python
"""E-63 D-0 (долг E-62 r4): тот ли осциллятор на L11/L17, что на L23.
R̄ вычисляется при ФИКСИРОВАННЫХ ω*(L23) на наборе strict-доков L23
(strict_names из layers_qwen_L23.part) для каждого слоя. Если R̄strict@ω*_L23
на L11 ≈ своему локальному R̄strict (0.86) — тот же осциллятор; если << — сосед."""
import json, math, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, '.')
from exp62_batt_lib import curve_u, K  # noqa: E402

ONES = np.ones(len(K))
RGX = np.round(np.concatenate([np.arange(0.02, 0.991, 0.01),
                               np.arange(0.992, 0.99965, 0.001)]), 4)
NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
W = {"chat": 0.5, "code": 0.94, "math": 1.28, "wiki": 0.6}
S23 = json.load(open("../results/e62/layers_qwen_L23.part"))["L23"]["phase"]
STRICT = {dm: v[4] for dm, v in S23.items() if v}


def wls(y, w, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ co) ** 2)), co


z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
out = {}
for L in (1, 5, 11, 17, 23):
    row = {}
    for dm, om in W.items():
        rng = np.random.default_rng(11)   # одна и та же последовательность на слой
        ps = {}
        names = [n for n in sorted(meta) if meta[n]["dom"] == dm]
        names = [n for n in names]        # dom-major, si-asc (sorted == dom_si-asc here)
        order = sorted(names, key=lambda n: int(n.rsplit("_", 1)[-1]))
        for n in order:
            V = curve_u(z[f"{n}_L{L}_u"], z[f"{n}_L{L}_hn"])
            y = []; sg = []
            for k in K:
                P = np.einsum("ij,ij->i", V[:-k], V[k:])
                y.append(float(np.median(P)))
                sg.append(max(float(np.median(
                    P[rng.integers(0, len(P), (200, len(P)))], axis=1).std()), 2e-3))
            y = np.array(y); w = 1 / np.array(sg) ** 2
            c, s = np.cos(om * K), np.sin(om * K)
            cand = [wls(y, w, [r ** K * c, r ** K * s, ONES]) for r in RGX]
            cf = cand[int(np.argmin([q[0] for q in cand]))][1]
            ps[n] = math.atan2(cf[1], cf[0])
        sel = [dname for dname in STRICT.get(dm, []) if dname in ps]
        allp = list(ps.values())
        row[dm] = dict(
            R_all=round(float(abs(np.exp(1j * np.array(allp)).mean())), 2),
            R_strict23=round(float(abs(np.exp(1j * np.array([ps[dname] for dname in sel])).mean())), 2)
                       if sel else None, nsel=len(sel))
    out[f"L{L}"] = row
    print(f"L{L:>2} @ω*(L23): " + "  ".join(
        f"{dm}: all={row[dm]['R_all']} strict@L23={row[dm]['R_strict23']}(n={row[dm]['nsel']})"
        for dm in W), flush=True)
json.dump(out, open("../results/e63/d0_strict_at_L23omega.json", "w"), ensure_ascii=False)
