#!/usr/bin/env python3
"""
Exp 29. «Фишерова линза» (docs/Exp29.md).

Fisher metric F_t = J_t^T M_p_t J_t, M_p = diag(p) - p p^T, over the exp27b
dense J-caches (S=128 positions, K=128 tokens, D=1024, 6 layers, 32 docs).
CPU post-processing, no model, no GPU.

F is never materialized (D x D): F @ v = J^T (M_p @ (J @ v)), O(K*D).
tr(F_t) = sum_i p_i ||J_i||^2 - ||sum_i p_i J_i||^2  (variance of J rows under p).

Outputs results/exp29_fisher.json:
  per layer: T_eff(l), d_F/d_E/d_C profiles (mean over docs),
  C_F vs C (cosine h) intra (lags 1/2/4/8/16) and inter (same position),
  ratio improvement, rank-agreement (Spearman + flipped-lag fraction),
  degeneracy share (tr(F_t) < 1% of layer median).

Usage:
  .venv/bin/python experiments/exp29_fisher.py --selfcheck
  .venv/bin/python experiments/exp29_fisher.py --ckpt-dir /data/prism_kernels/exp27b/results
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

LAYERS = [6, 12, 18, 19, 22, 23]
LAGS = [1, 2, 4, 8, 16]


def fisher_mv(J, p, v):
    """F @ v = J^T (diag(p) - p p^T) (J v). J [K,D], p [K], v [D]."""
    y = J @ v                                # [K]
    return J.t() @ (p * (y - float(p @ y)))  # [D]


def fisher_trace(J, p):
    """tr(J^T M_p J) = sum p_i ||J_i||^2 - ||sum p_i J_i||^2."""
    rows2 = J.pow(2).sum(1)                  # [K]
    return float(p @ rows2) - float((p @ J) @ (p @ J))


def _selfcheck():
    torch.manual_seed(0)
    K, D = 6, 10
    J = torch.randn(K, D)
    p = torch.softmax(torch.randn(K), 0)
    M = torch.diag(p) - torch.outer(p, p)
    F = J.t() @ M @ J
    v = torch.randn(D)
    direct = F @ v
    fast = fisher_mv(J, p, v)
    assert torch.allclose(direct, fast, atol=1e-5), "fisher_mv vs direct"
    tr_direct = float(F.diagonal().sum())
    tr_fast = fisher_trace(J, p)
    assert abs(tr_direct - tr_fast) < 1e-5, (tr_direct, tr_fast)
    # degenerate: p peaked -> F ~ 0
    p2 = torch.zeros(K); p2[0] = 1.0
    F2 = J.t() @ (torch.diag(p2) - torch.outer(p2, p2)) @ J
    assert fisher_trace(J, p2) < 1e-6, "peaked p should give ~0 trace"
    # delta form on a 2-position pair
    h1, h2 = torch.randn(D), torch.randn(D)
    d = h2 - h1
    cf_direct = float(d @ (F @ d)) / float(d @ d)
    cf_fast = float(d @ fisher_mv(J, p, d)) / float(d @ d)
    assert abs(cf_direct - cf_fast) < 1e-5, (cf_direct, cf_fast)
    print("selfcheck OK")


def gauge_J(J, p):
    """Gauge-free J^(p) = (diag(p) - p p^T) @ J, batched.
    J [.., K, D], p [.., K] -> [.., K, D]."""
    mu = (p.unsqueeze(-1) * J).sum(-2, keepdim=True)  # p-wt mean row
    return p.unsqueeze(-1) * (J - mu)


def batched_cf(Jgf, p, H, L, S):
    """Intra C_F and C (cosine) at lag L over [n,S,K,D]/[n,S,K]/[n,S,D].
    Doc-boundary pairs masked out. Returns (cf, c) over valid pairs."""
    n, S0, K, D = Jgf.shape
    Jf = Jgf.reshape(n * S0, K, D)
    pf = p.reshape(n * S0, K)
    Hf = H.reshape(n * S0, D)
    d = Hf[L:] - Hf[:-L]
    Jt = Jf[:-L]
    y = torch.einsum('tkd,td->tk', Jt, d)
    w = pf[:-L] * (y - (pf[:-L] * y).sum(1, keepdim=True))
    q = torch.einsum('tkd,tk->td', Jt, w)
    cf = (d * q).sum(1) / d.pow(2).sum(1).clamp_min(1e-20)
    denom = (Hf[L:].norm(dim=1) * Hf[:-L].norm(dim=1)).clamp_min(1e-20)
    c = (Hf[L:] * Hf[:-L]).sum(1) / denom
    pos = torch.arange(n * S - L)
    keep = (pos % S) < (S - L)        # no cross-doc pairs
    return cf[keep], c[keep]


def _selfcheck2():
    # batched_cf vs direct fisher_mv on a tiny synthetic
    torch.manual_seed(1)
    T, K, D, L = 8, 6, 10, 1
    J = torch.randn(1, T, K, D)
    p = torch.softmax(torch.randn(1, T, K), -1)
    H = torch.randn(1, T, D)
    cf, c = batched_cf(J, p, H, L, S=T)
    for t in range(T - L):
        d = H[0, t + L] - H[0, t]
        q = fisher_mv(J[0, t], p[0, t], d)
        ref = float(d @ q) / float(d @ d)
        assert abs(ref - float(cf[t])) < 1e-4, (t, ref, float(cf[t]))
    # gauge_J: projector kills constant rows
    J0 = torch.randn(K, D)
    Jc = torch.stack([J0] * 4)  # identical rows -> P J = 0? no: rows identical means
    # J = 1 x v^T form: P J = (p 1^T) diag... check (diag(p)-pp^T) @ (1 v^T) = (p-p) v^T = 0
    Jrank1 = J0[:1].expand(K, D).contiguous()
    assert float(gauge_J(Jrank1, p[0]).abs().max()) < 1e-5, "gauge must kill rank-1 rows"
    print("selfcheck2 OK")


def _sha16(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--ckpt-dir", default="/data/prism_kernels/exp27b/results")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--out", default="results/exp29_fisher.json")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        _selfcheck2()
        return
    layers = [int(x) for x in args.layers.split(",") if x]
    import time
    t0 = time.time()
    out = {"params": {"S": 128, "lags": LAGS, "ckpt_dir": args.ckpt_dir,
                      "layers": layers, "degen_frac_thresh": 0.01,
                      "tz_deviations": [
                          "token_types: not in exp27b cache - TZ step 3 skipped",
                          "cross-doc pairs restricted to equal position t (TZ: any t,s)",
                          "dF/dE/dC profiles saved as layer means, not per-doc matrices",
                          "rank_flip_frac is a median-based binary proxy, not full rank comparison",
                          "T_eff = tr(F) is scale-dependent (pre-RMSNorm); depth trend is empirical only"]}}
    out["layers"] = {}
    manifest = {}
    for l in layers:
        tp = time.time()
        ck = torch.load(os.path.join(args.ckpt_dir, f"exp27b_jcache_L{l}.pt"),
                        map_location="cpu", weights_only=False)
        U128, pool_fp, S = ck["U128"], ck["pool_fp"], ck["S"]
        n = ck["n_samples"] * len(ck["domains"])
        docs = [(dom, si) for dom in ck["domains"] for si in range(ck["n_samples"])]
        # DOC-MAJOR: [doc, position, ...] (EXP29_errors #1: position-major
        # view made batched_cf compare different docs at the same position)
        J = torch.stack([ck["JR"][(dom, si, t)]
                         for (dom, si) in docs for t in range(S)]
                        ).view(n, S, 128, -1)          # [n, T, K, D]
        P = torch.stack([ck["P128"][(dom, si, t)]
                         for (dom, si) in docs for t in range(S)]
                        ).view(n, S, 128)              # [n, T, K] (already normalized)
        Hh = torch.stack([ck["H"][(dom, si, t)]
                          for (dom, si) in docs for t in range(S)]
                         ).view(n, S, -1)              # [n, T, D]
        # Fisher-Rao: Tr(J^T F_p J) on the RAW J — F_p = diag(p)-pp^T already
        # absorbs the softmax gauge (F_p p = 0); a second projector would give F_p^2.
        # Sanity (EXP29_errors #2): JR must be the RAW logit Jacobian; gauge-free
        # J would give p^T J = 0 exactly.
        jr_raw = float(torch.einsum('ntk,ntkd->ntd', P, J).abs().max())
        if jr_raw < 1e-3 * float(J.abs().max()):
            raise SystemExit(f"L{l}: JR looks gauge-free (p^T J ~ 0), "
                             f"Fisher metric would be wrong")
        rows2 = J.pow(2).sum(-1)                        # [n, T, K]
        mu = (P.unsqueeze(-1) * J).sum(2)               # [n, T, D]
        trF = (P * rows2).sum(2) - mu.pow(2).sum(2)     # [n, T]
        med = float(trF.median())
        deg = float((trF < 0.01 * med).float().mean())
        intra = {}
        rank = {}
        for L in LAGS:
            cf, c = batched_cf(J, P, Hh, L, S)
            intra[f"lag{L}"] = {"cf": float(cf.mean()), "c": float(c.mean()),
                                "n": int(cf.numel())}
            flip = ((cf > cf.median()).int() ^ (c > c.median()).int()).float().mean()
            rank[f"lag{L}"] = float(flip)
        from scipy.stats import spearmanr
        cf1, c1 = batched_cf(J, P, Hh, 1, S)
        Hf = Hh.reshape(n * S, -1)
        d = Hf[1:] - Hf[:-1]
        Jf0 = J.reshape(n * S, 128, -1)
        pf0 = P.reshape(n * S, 128)
        keep1 = (torch.arange(n * S - 1) % S) < (S - 1)
        d, Jf1, pf1 = d[keep1], Jf0[:-1][keep1], pf0[:-1][keep1]
        y1 = torch.einsum('tkd,td->tk', Jf1, d)
        w1 = pf1 * (y1 - (pf1 * y1).sum(1, keepdim=True))
        q1 = torch.einsum('tkd,tk->td', Jf1, w1)
        dF = (d * q1).sum(1)
        # EXP29_errors #3.1: C_F vs cosine needs scale normalization —
        # dF_norm divides by the Fisher trace at position t (= diag C_F)
        keepidx = torch.arange(n * S - 1)[keep1]
        dF_norm = dF / trF.reshape(-1)[keepidx].clamp_min(1e-20)
        out["layers"][str(l)] = {
            "T_eff": float(trF.mean()),
            "T_eff_by_pos": [float(x) for x in trF.mean(0).tolist()],
            "degen_frac": deg,
            "intra": intra,
            "rank_flip_frac": rank,
            "spearman_cf_c_lag1": float(spearmanr(cf1, c1)[0]),
            "jr_raw_pJ_max": jr_raw,
            "dF_lag1": float(dF.sqrt().mean()),
            "dF_lag1_norm": float(dF_norm.sqrt().mean()),
            "dE_lag1": float(d.norm(2, 1).mean()),
            "dC_lag1": float(1 - c1.mean()),
        }
        # inter: same position t, docs a-b, F from a
        dI = Hh[:, None, :, :] - Hh[None, :, :, :]      # [n, n, T, D]  a=row? careful
        # dI[a,b,t] = H[a,t]-H[b,t]
        cf_int, c_int = [], []
        for a in range(n):
            y = torch.einsum('btd,tkd->btk', dI[a], J[a])   # [n, T, K]
            w = P[a].unsqueeze(0) * (y -
                                     (P[a].unsqueeze(0) * y).sum(2, keepdim=True))
            q = torch.einsum('tkd,btk->btd', J[a], w)
            cf_int.append((dI[a] * q).sum(-1) /
                          dI[a].pow(2).sum(-1).clamp_min(1e-20))
            c_int.append((Hh[a].unsqueeze(0) * Hh).sum(-1) /
                         (Hh[a].norm(2, -1).unsqueeze(0) *
                          Hh.norm(2, -1)).clamp_min(1e-20))
        cfI = torch.stack(cf_int)           # [n, n, T]
        cI = torch.stack(c_int)             # [n, n]
        mask = ~torch.eye(n, dtype=torch.bool)
        cfI, cI = cfI[mask], cI[mask]
        out["layers"][str(l)]["inter"] = {"cf": float(cfI.mean()), "c": float(cI.mean()),
                                          "n": int(cfI.numel())}
        r1 = intra["lag1"]
        out["layers"][str(l)]["ratio"] = {
            "cf": float(r1["cf"] / cfI.mean().clamp_min(1e-20)),
            "c": float(r1["c"] / cI.mean().clamp_min(1e-20))}
        manifest[str(l)] = {"sha256_16": _sha16(os.path.join(args.ckpt_dir,
                           f"exp27b_jcache_L{l}.pt")), "pool_fp": pool_fp}
        print(f"L{l}: T_eff={float(trF.mean()):.1f} deg={deg:.3f} "
              f"intra cf={r1['cf']:.4f} c={r1['c']:.4f} "
              f"inter cf={float(cfI.mean()):.4f} c={float(cI.mean()):.4f} "
              f"[{time.time() - tp:.0f}s]")
        del J, P, Hh, dI, cfI, cI
    out["manifest"] = manifest
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[OK] saved {args.out} [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
