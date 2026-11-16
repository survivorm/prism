#!/usr/bin/env python
"""E-62x: (A) тест r2=1 — двухвременка или «dip+floor» (ревью E-62, приоритет 1);
(B) робастность фазового R̄ к семени σ-бутстрапа (самоаудит: L23-контроль fixw
не сошёлся с E-60d — 0.57 vs 0.79). Источники: Qwen L{1,5,11,17,23}, LFM3 L23, 4B."""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, '.')
from exp62_batt_lib import curve_u, K  # noqa: E402

NL = 14
KL = K[:NL]
ONES = np.ones(NL)
RG2 = np.round(np.arange(0.02, 0.981, 0.02), 3)


def fits(tgt, dm_order):
    """tgt: {dm: y} по 14 лагам. Возвращает (F1_2exp, F1_1exp, F1_const, r-оценки)."""
    best = {"2": None, "1": None, "0": None}
    tt_tot = sum(float(np.sum(tgt[dm][:NL] ** 2)) for dm in dm_order)
    for a in RG2:
        Ga2 = a ** KL
        for b in RG2[RG2 > a + 0.02]:
            G = np.stack([Ga2, b ** KL, ONES], 1)
            sse = tt = 0.0
            for dm in dm_order:
                x = tgt[dm][:NL]
                co = np.linalg.lstsq(G, x, rcond=None)[0]
                sse += float(np.sum((x - G @ co) ** 2)); tt += float(np.sum(x ** 2))
            if best["2"] is None or sse < best["2"][0]:
                best["2"] = (sse, tt, a, b)
        G1 = np.stack([Ga2, ONES], 1)
        sse = tt = 0.0
        for dm in dm_order:
            x = tgt[dm][:NL]
            co = np.linalg.lstsq(G1, x, rcond=None)[0]
            sse += float(np.sum((x - G1 @ co) ** 2))
        if best["1"] is None or sse < best["1"][0]:
            best["1"] = (sse, tt_tot, a)
    x0 = np.concatenate([tgt[dm][:NL] for dm in dm_order])
    mu = float(x0.mean())
    best["0"] = (float(np.sum((x0 - mu) ** 2)), tt_tot, None)
    return (round(1 - best["2"][0] / tt_tot, 3), best["2"][2], best["2"][3],
            round(1 - best["1"][0] / tt_tot, 3), best["1"][2],
            round(1 - best["0"][0] / best["0"][1], 3))


def curves(traj, dom_of, order, seed=11):
    rng = np.random.default_rng(seed)
    Y = {}
    for nm in order:
        V = curve_u(*traj[nm])
        y = []
        for k in K:
            P = np.einsum("ij,ij->i", V[:-k], V[k:])
            y.append(float(np.median(P)))
            _ = float(np.median(P[rng.integers(0, len(P), (200, len(P)))], axis=1).std())
        Y[nm] = np.array(y)
    doms = [d for d in ("wiki", "math", "code", "chat") if d in set(dom_of.values())]
    real = {dm: np.mean([Y[n] for n in order if dom_of[n] == dm], 0) for dm in doms}
    return Y, real, doms


def load_qwen():
    z = np.load("/data/prism_kernels/exp62/gen_layers_qwen.npz", allow_pickle=True)
    meta = json.load(open("/data/prism_kernels/exp62/gen_layers_qwen_meta.json"))
    dom_order = ("wiki", "math", "code", "chat")
    order = sorted(meta, key=lambda n: (dom_order.index(meta[n]["dom"]),
                                        int(n.rsplit("_", 1)[-1])))
    dom_of = {n: meta[n]["dom"] for n in order}
    return z, meta, order, dom_of, "L{L}"


def main():
    out = {}
    z, meta, order, dom_of, pat = load_qwen()
    dom_order = ("wiki", "math", "code", "chat")
    for L in (1, 5, 11, 17, 23):
        traj = {n: (z[f"{n}_L{L}_u"], z[f"{n}_L{L}_hn"]) for n in order}
        Y, real, _ = curves(traj, dom_of, order)
        f2, a2, b2, f1_, a1, f0 = fits(real, dom_order)
        out[f"qwen_L{L}"] = dict(F2=f2, r2=[a2, b2], F1=f1_, r1=a1, F0=f0)
        print(f"Qwen L{L:>2}: F(2exp)={f2} @(r={a2},{b2})  F(r2=1)={f1_} @(r={a1})"
              f"  F(const)={f0}  Δ={round(f2 - f1_, 3)}", flush=True)
    # LFM3 (exp48b npz)
    z2 = np.load("../results/exp48b_lfm3_gen_kinematics.npz", allow_pickle=True)
    docs = json.load(open("../results/exp48b_lfm3_gen_kinematics.json"))["docs"]
    order2 = [f"{d['dom']}_{d['si']}" for d in docs]
    dom2 = {f"{d['dom']}_{d['si']}": d["dom"] for d in docs}
    traj2 = {n: (z2[n + "_u"], z2[n].item()["hnorm"]) for n in order2}
    Y, real, _ = curves(traj2, dom2, order2)
    f2, a2, b2, f1_, a1, f0 = fits(real, ("wiki", "math", "code", "chat"))
    out["lfm3_L23"] = dict(F2=f2, r2=[a2, b2], F1=f1_, r1=a1, F0=f0)
    print(f"LFM3 L23: F(2exp)={f2} @(r={a2},{b2})  F(r2=1)={f1_} @(r={a1})  "
          f"F(const)={f0}  Δ={round(f2 - f1_, 3)}", flush=True)
    # 4B
    z3 = np.load("../results/e52_4b/exp47b_gen_kinematics.npz", allow_pickle=True)
    docs3 = json.load(open("../results/e52_4b/exp47b_gen_kinematics.json"))["docs"]
    order3 = [f"{d['dom']}_{d['si']}" for d in docs3]
    dom3 = {f"{d['dom']}_{d['si']}": d["dom"] for d in docs3}
    traj3 = {n: (z3[n + "_u"], z3[n].item()["hnorm"]) for n in order3}
    Y, real, _ = curves(traj3, dom3, order3)
    f2, a2, b2, f1_, a1, f0 = fits(real, ("wiki", "math", "code", "chat"))
    out["qwen_4b"] = dict(F2=f2, r2=[a2, b2], F1=f1_, r1=a1, F0=f0)
    print(f"4B: F(2exp)={f2} @(r={a2},{b2})  F(r2=1)={f1_} @(r={a1})  "
          f"F(const)={f0}  Δ={round(f2 - f1_, 3)}", flush=True)
    json.dump(out, open("../results/e62/r2test.json", "w"), ensure_ascii=False)
    # (B) seed-робастность фазы на Qwen L23
    import math
    RGX = np.round(np.concatenate([np.arange(0.02, 0.991, 0.01),
                                   np.arange(0.992, 0.99965, 0.001)]), 4)
    W = {"chat": 0.5, "code": 0.94, "math": 1.28, "wiki": 0.6}

    def wls(y, w, cols):
        G = np.stack(cols, 1)
        co = np.linalg.lstsq(G * w[:, None] ** .5, y * w ** .5, rcond=None)[0]
        return float(np.sum(w * (y - G @ co) ** 2)), co

    for seed in (11, 7, 2024):
        for wmode in ("w", "unw"):
            rng = np.random.default_rng(seed)
            Rb = {dm: [] for dm in W}
            for n in order:
                V = curve_u(z[f"{n}_L23_u"], z[f"{n}_L23_hn"])
                y = []; sg = []
                for k in K:
                    P = np.einsum("ij,ij->i", V[:-k], V[k:])
                    y.append(float(np.median(P)))
                    sg.append(max(float(np.median(
                        P[rng.integers(0, len(P), (200, len(P)))], axis=1).std()), 2e-3))
                y = np.array(y)
                w = 1 / np.array(sg) ** 2 if wmode == "w" else np.ones_like(y)
                for dm in W:
                    c, s = np.cos(W[dm] * K), np.sin(W[dm] * K)
                    cand = [wls(y, w, [r ** K * c, r ** K * s, ONES])
                            for r in RGX]
                    cf = cand[int(np.argmin([q[0] for q in cand]))][1]
                    Rb[dm].append(math.atan2(cf[1], cf[0]))
            row = {dm: round(float(abs(np.exp(1j * np.array(Rb[dm])).mean())), 2)
                   for dm in W}
            print(f"seed={seed} {wmode}: R̄@ω*_L23 {row}", flush=True)


main()
