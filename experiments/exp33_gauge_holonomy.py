"""E-33 (preflight). Gauge-голономия: транспортируемый gauge-вектор vs случайный
той же нормы. Только exp27b-кэш, без модели.

Gauge-направление (cheap-реализация E-12a, без SVD):
  u_g(t) = J_t J_t^T 1 / ||.||   (левый, R^K, токенный)
  v_g(t) = J_t^T u_g(t) / ||.||  (правый, R^D)
P(p)u = p o u - p(p.u)  (= (diag(p) - p p^T) u), нормализуем после.

НОГА ПОЗИЦИИ (T_t^(p)): u'(t) = norm(P(p_t) u_g(t)), compare с u_g(t+1);
  цепочка L=8. Baseline: N_SHUF случайных единичных u, тот же транспорт.
  One-sided permutation: p = frac(csh >= med(cs)) по медиане real.
НОГА ГЛУБИНЫ (прокси, НЕ T^h — D_l в кэше нет): cos(v_g^a(t), v_g^b(t))
  для пар слоёв (6,12),(12,18),(18,22),(22,23) vs случайные пары R^D.
Sentinel: P128_RAW sum < 0.5 — позиция исключается, если попадает в [t..t+L].
"""
import argparse, gc, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp34_p1_coupling as E34
import exp28_leaf_structure as E28

SEED = 42
LAYERS = [6, 12, 18, 22, 23]
L_CHAIN = 8
N_SHUF = 128
DEPTHS = [(6, 12), (12, 18), (18, 22), (22, 23)]
SENT = E34.SENTINEL_SUM
K = 128
D = 1024


def gauge_vec(J):
    """J [K,D] -> (u_g [K], v_g [D]) or None."""
    ones = torch.ones(J.shape[0], dtype=J.dtype)  # K
    Ju = J @ (J.T @ ones)
    n = Ju.norm()
    if n < 1e-12:
        return None
    ug = Ju / n
    vg = J.T @ ug
    ng = vg.norm()
    if ng < 1e-12:
        return None
    return ug, vg / ng


def Pp(p, u):
    """(diag(p) - p p^T) u, normalized. None если вырожден."""
    out = p * u - p * (p * u).sum()
    n = out.norm()
    if n < 1e-12:
        return None
    return out / n


def pos_leg(J, P, sent, rng):
    """One-step + chain L=8 transport vs shuffled. J [S,K,D], P [S,K]."""
    S = J.shape[0]
    ugs = []
    for t in range(S):
        g = gauge_vec(J[t])
        ugs.append(g[0] if g else None)
    out = {}
    for L in (1, L_CHAIN):
        cs, csh = [], []
        for t in range(S - L):
            if ugs[t] is None or ugs[t + L] is None:
                continue
            if sent[t:t + L + 1].any():
                continue
            cur = ugs[t].clone()
            ok = True
            for s in range(L):
                cur = Pp(P[t + s], cur)
                if cur is None:
                    ok = False
                    break
            if not ok:
                continue
            cs.append(float((cur * ugs[t + L]).sum()))
            R = torch.tensor(rng.standard_normal((N_SHUF, P.shape[1])), dtype=torch.float32)
            R = R / R.norm(dim=1, keepdim=True)
            for s in range(L):
                p = P[t + s]
                R = p * R - (R * p).sum(1, keepdim=True) * p
                R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-30)
            csh.extend((R @ ugs[t + L]).tolist())
        if len(cs) < 10:
            out[f"chain{L}"] = None
            continue
        cs, csh = np.array(cs), np.array(csh)
        # null: u_g(t+L) — случайная единица, indep от транспорта -> real cos ~ один shuffled cos
        out[f"chain{L}"] = {
            "cos_real": float(np.median(cs)), "cos_shuf": float(np.median(csh)),
            "p": float((csh >= np.median(cs)).mean()), "n": len(cs),
        }
    return out


def depth_leg(Ja, Jb, sent, rng):
    """cos(v_g^a, v_g^b) same t vs random pairs in R^D. sent: исключение sentinel."""
    S = min(Ja.shape[0], Jb.shape[0])
    cs = []
    for t in range(S):
        if sent[t]:
            continue
        ga, gb = gauge_vec(Ja[t]), gauge_vec(Jb[t])
        if not ga or not gb:
            continue
        cs.append(float((ga[1] * gb[1]).sum()))
    if len(cs) < 10:
        return None
    ra = rng.standard_normal((N_SHUF, D))
    rb = rng.standard_normal((N_SHUF, D))
    ra /= np.linalg.norm(ra, axis=1, keepdims=True)
    rb /= np.linalg.norm(rb, axis=1, keepdims=True)
    csh = np.einsum("ij,ij->i", ra, rb)
    cs = np.array(cs)
    return {"cos_real": float(np.median(cs)), "cos_shuf": float(np.median(csh)),
            "p": float((csh >= np.median(cs)).mean()), "n": len(cs)}


def med_iqr(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None
    return [float(np.median(a)), float(np.percentile(a, 75) - np.percentile(a, 25))]


def load_doc(ck, dom, si, S):
    J = torch.stack([ck["JR"][(dom, si, t)] for t in range(S)]).float()
    P = torch.stack([ck["P128"][(dom, si, t)] for t in range(S)]).float()
    Pr = torch.stack([ck["P128_RAW"][(dom, si, t)] for t in range(S)]).float()
    sent = (Pr.sum(-1) < SENT).numpy()
    return J, P, sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/data/prism_kernels/exp27b/results")
    ap.add_argument("--out", default="results/exp33_gauge_holonomy.json")
    ap.add_argument("--sample-cap", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        sc(); return
    torch.set_num_threads(1)
    E28._pin_p_cores()
    rng = np.random.default_rng(SEED)
    out = {"n_docs": 0, "L_chain": L_CHAIN, "n_shuf": N_SHUF,
           "sentinel": SENT, "layers": {}, "depth": {}}
    ck = E28.load_layer(a.ckpt_dir, a.layers[0])
    S = ck["S"]
    docs = sorted([(dom, si) for dom in ck["domains"] for si in range(ck["n_samples"])])
    if a.sample_cap:
        docs = docs[:min(a.sample_cap, len(docs))]
    del ck; gc.collect()
    out["n_docs"] = len(docs)
    # position leg per layer
    for l in a.layers:
        ck = E28.load_layer(a.ckpt_dir, l)
        st1, st8, sents = [], [], []
        for (dom, si) in docs:
            J, P, sent = load_doc(ck, dom, si, S)
            sents.append(float(sent.mean()))
            r = pos_leg(J, P, sent, rng)
            if r["chain1"]:
                st1.append((r["chain1"]["cos_real"], r["chain1"]["p"]))
            if r["chain8"]:
                st8.append((r["chain8"]["cos_real"], r["chain8"]["p"]))
            del J, P
            print(f"  L{l} {dom}/{si}", flush=True)
        del ck; gc.collect()
        out["layers"][str(l)] = {
            "pos1": {"cos_real": med_iqr([s[0] for s in st1]),
                     "p": med_iqr([s[1] for s in st1]), "n_docs": len(st1)},
            "pos8": {"cos_real": med_iqr([s[0] for s in st8]),
                     "p": med_iqr([s[1] for s in st8]), "n_docs": len(st8)},
            "sent_frac": float(np.mean(sents)) if sents else None,
        }
    print("pos done", flush=True)
    # depth leg (прокси)
    for la, lb in DEPTHS:
        cka, ckb = E28.load_layer(a.ckpt_dir, la), E28.load_layer(a.ckpt_dir, lb)
        st = []
        for (dom, si) in docs:
            Ja, Pa, sa = load_doc(cka, dom, si, S)
            Jb, Pb, sb = load_doc(ckb, dom, si, S)
            r = depth_leg(Ja, Jb, sa | sb, rng)
            if r:
                st.append((r["cos_real"], r["p"], r["n"]))
            del Ja, Jb, Pa, Pb
            gc.collect()
        del cka, ckb; gc.collect()
        out["depth"][f"{la}-{lb}"] = {
            "cos_real": med_iqr([s[0] for s in st]),
            "p": med_iqr([s[1] for s in st]),
            "n_docs": len(st),
            "note": "coherence-прокси, НЕ транспорт T^h (D_l в кэше нет)",
        }
    ps = {l: out["layers"][str(l)]["pos1"]["p"][0] for l in a.layers
          if out["layers"][str(l)]["pos1"]["p"] is not None}
    if not ps:
        out["verdict"] = ("нет данных: P(p) вырожден на всех слоях (sentinel/degenerate) — "
                          "preflight неинформативен, смотреть depth-прокси")
    else:
        # двухсторонний: p~0.5 = тривиально; p<=0.05 = сигнал; p>=0.95 = anti-signal
        trig = all(0.05 < p < 0.95 for p in ps.values())
        sig = [l for l, p in ps.items() if p <= 0.05]
        anti = [l for l, p in ps.items() if p >= 0.95]
        out["p_values_pos1"] = {str(l): p for l, p in ps.items()}
        out["verdict"] = ("gauge-связность тривиальна (транспорт ≈ случайный), E-33 закрыт как контроль"
                          if trig else
                          f"транспорт отличим от случайного (signal layers={sig}, anti-signal layers={anti}) — разворачивать полный вариант 1 (плакетка)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, allow_nan=False)
    print("saved", a.out)


def sc():
    torch.manual_seed(0); np.random.seed(0)
    # gauge_vec: rank-1 J = u v^T -> u_g = u/|u|
    u = np.random.default_rng(0).standard_normal(8).astype(np.float32)
    u /= np.linalg.norm(u)
    v = np.random.default_rng(1).standard_normal(16).astype(np.float32)
    v /= np.linalg.norm(v)
    J = torch.tensor(np.outer(u, v))
    gu = gauge_vec(J)
    assert gu is not None
    assert abs(float((gu[0] * torch.tensor(u)).sum())) > 0.99
    # P(p)u formula vs (diag(p)-pp^T)u
    p = torch.exp(torch.randn(8)); p = p / p.sum()
    x = torch.arange(8, dtype=torch.float32) + 1.0  # не в ker(P)=span{1}
    a1 = Pp(p, x)
    Pmat = torch.diag(p) - p.unsqueeze(1) * p.unsqueeze(0)
    a2 = Pmat @ x
    a2 = a2 / a2.norm()
    assert abs(float((a1 * a2).sum())) > 0.999
    # P(p) 1 = 0 (ker = span{1})
    e = torch.ones(8)
    assert Pp(p, e) is None
    # sentinel mask
    sent = np.array([0, 0, 1, 0, 0], dtype=bool)
    assert sent[0:3].any() and not sent[3:5].any()
    # uniform p: Pp сохраняет направление (P(1/K) = I/K на 1^⊥)
    Ks = 32
    pu = torch.full((Ks,), 1.0 / Ks)
    u0 = torch.randn(Ks)
    u0 = u0 - u0.mean()
    u0 = u0 / u0.norm()
    assert abs(float((Pp(pu, u0) * u0).sum())) > 0.99
    # pos_leg на синтетике, сигнал: const u_g, uniform p -> cos ≈ 1, p ≈ 0
    rng = np.random.default_rng(0)
    S_ = 24  # L=8: t ∈ [0, S-8) = 16 позиций, минимум 10
    vs = torch.randn(S_, 8)
    u_ = torch.randn(Ks); u_ = u_ / u_.norm()
    Jc = torch.stack([torch.outer(u_, vs[t]) for t in range(S_)])
    Pc = torch.stack([pu] * S_)
    sent0 = np.zeros(S_, dtype=bool)
    r = pos_leg(Jc, Pc, sent0, rng)
    assert r["chain1"] is not None and r["chain1"]["cos_real"] > 0.95, r["chain1"]
    assert r["chain1"]["p"] < 0.05, r["chain1"]["p"]
    assert r["chain8"] is not None and r["chain8"]["cos_real"] > 0.99, r["chain8"]
    # null: случайные u_g(t) -> cos ≈ 0, p ≈ 0.5
    us = torch.randn(S_, Ks)
    us = us / us.norm(dim=1, keepdim=True)
    vs = torch.randn(S_, 8)
    Jn = torch.stack([torch.outer(us[t], vs[t]) for t in range(S_)])
    r2 = pos_leg(Jn, Pc, sent0, rng)
    assert r2["chain1"] is not None
    assert 0.2 < r2["chain1"]["p"] < 0.8, r2["chain1"]["p"]
    assert abs(r2["chain1"]["cos_real"]) < 0.3
    print("selfcheck OK")


if __name__ == "__main__":
    main()
