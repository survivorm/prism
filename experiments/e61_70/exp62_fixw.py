import json, math, sys
import numpy as np
sys.path.insert(0, '.')
from exp62_batt_lib import curve_u, K
NL = len(K); ONES = np.ones(NL)
z = np.load('/data/prism_kernels/exp62/gen_layers_qwen.npz', allow_pickle=True)
meta = json.load(open('/data/prism_kernels/exp62/gen_layers_qwen_meta.json'))
names = sorted(meta); dom = {n: meta[n]['dom'] for n in names}
RGX = np.round(np.concatenate([np.arange(0.02, 0.991, 0.01),
                               np.arange(0.992, 0.99965, 0.001)]), 4)
rng = np.random.default_rng(11)
W = {'chat': 0.5, 'code': 0.94, 'math': 1.28, 'wiki': 0.6}   # ω*_strict c L23


def wls(y, w, cols):
    G = np.stack(cols, 1)
    co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ co) ** 2)), co


out = {}
for L in (1, 5, 11, 17, 23):
    Rb = {dm: [] for dm in W}
    for n in names:
        V = curve_u(z[f'{n}_L{L}_u'], z[f'{n}_L{L}_hn'])
        y = []; sg = []
        for k in K:
            P = np.einsum('ij,ij->i', V[:-k], V[k:])
            y.append(float(np.median(P)))
            sg.append(max(float(np.median(P[rng.integers(0, len(P), (200, len(P)))],
                                          axis=1).std()), 2e-3))
        y = np.array(y); w = 1 / np.array(sg) ** 2
        for dm in W:
            c, s = np.cos(W[dm] * K), np.sin(W[dm] * K)
            cands = [wls(y, w, [r ** K * c, r ** K * s, ONES]) for r in RGX]
            cf = cands[int(np.argmin([q[0] for q in cands]))][1]
            Rb[dm].append(math.atan2(cf[1], cf[0]))
    out[f"L{L}"] = {dm: round(float(abs(np.exp(1j * np.array(Rb[dm])).mean())), 2)
                    for dm in W}
    print(f"L{L:>2} R̄@ω*(L23): " + " ".join(f"{dm}={out[f'L{L}'][dm]:.2f}"
                                            for dm in W), flush=True)
json.dump(out, open('../results/e62/phase_at_L23omega.json', 'w'))
