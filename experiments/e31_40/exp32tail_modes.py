"""E-32-tail — token-space localization of the second order (docs/E32t.md).

Question (TЗ §3.1): is the Hessian cross-talk b_off = b − diag(b) localized
in the SAME tail ESG modes (u_k, eigenvectors of ESG = JR·JRᵀ) as the first
order off²(r) (E-30b: L22 68.6% / L18 34.3% in r ∈ [119,128])?

Primary metric (token-space cumulative, TЗ §3.2):
    C_b(r) = ‖P_r b_off P_r‖_F² / ‖b_off‖_F²,  P_r = Σ_{k≤r} u_k u_kᵀ
via the Gram trick:  M_b[k,k'] = (u_kᵀ b_off u_k')²,
C_b(r) = diag(cumsum(cumsum(M_b))) / M_b.sum().
Sign-invariant (u_k → −u_k), O(K³) per position.

b = a2 EXACTLY (TЗ normalization): the cubic fit is dz = a1·e + a2·e² + a3·e³
with absolute e and unit injection Jn_i = JR_i/‖JR_i‖, so
    a2_ji = ½(Jᵀe_i)ᵀ H_j (Jᵀe_i) / (ESG)_ii,
which is the TЗ definition of b_ji. No renormalization needed.

Control in the same script (cheap, no extra forwards): per-position off²(r)
cumulative from the SAME cached JR (exact E-30b algebra: gauge-free
w_k = u_k − (u_kᵀP)P, M = (WᵀW)²·λ_k λ_k', diag from P·ESG_r·P) — so
Spearman(C_b, C_off²) is computed PER DOC (128 modes, 128 points each) and
aggregated, and my per-doc off2_cum reproduces E-30b on the same corpus
(aggregated curve must match results/exp30_full_spectrum.json).

NOT done (TЗ §3.2 secondary): the hidden-space sweep b^(r) needs the
bilinear form ½ uᵀ H_j v at arbitrary directions — a2 stores only the
diagonal samples Q_j(Jn_i); computing it needs O(K²) extra injections.
Marked in output meta; token-space C_b is the primary object anyway.

Outcomes (TЗ §3.5): (T1) inheritance — Spearman > 0.9, overlap top-5 ≥ 3,
tail_share_b ≈ first-order tail share; (T2) independence — Spearman < 0.5,
overlap ≤ 1; (T3) diffuse — C_b(r) near-linear, tail share < 30%.

Pipeline = E-32r.2/3 (16-pt relative ε sweep 5%..70% of ‖h_t‖, cubic fit
without const, float32 Δz / float64 fit) — a2 is recomputed, NOT read from
exp32r2_hessian.json (that file stores only the norm nb_median, not b).

Aggregation (E-34 protocol, as E-30b): per-doc median over valid (non-
sentinel) positions → median/IQR over 32 docs.

Layers: pilot L18, L22 (TЗ §3.7); `--layers 6 12 18 19 22 23` for the full
profile if the pilot lands (T1)/(T2).

CPU-safe (12 threads, CHUNK=4096, ~30-60 min for the 2-layer pilot).
Self-contained: no prism.* imports.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

S = 128
D = 1024
T_FULL = 128
EPS_REL = np.round(np.logspace(np.log10(0.05), np.log10(0.70), 16), 6)
NEPS = len(EPS_REL)
MODEL = 'Qwen/Qwen3.5-0.8B'
CHUNK = 4096
OUT = 'results/exp32tail_hessian_modes.json'
JCACHE = {
    6: '/data/stage/jc_a/exp27b_jcache_L6.pt',
    12: '/data/stage/jc_a/exp27b_jcache_L12.pt',
    18: '/data/stage/jc_a/exp27b_jcache_L18.pt',
    19: '/data/stage/jc_b/exp27b_jcache_L19.pt',
    22: '/data/stage/jc_b/exp27b_jcache_L22.pt',
    23: '/data/stage/jc_b/exp27b_jcache_L23.pt',
}
E30B = 'results/exp30_full_spectrum.json'
PILOT = (18, 22)


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------
def cb_curve(b_off, U):
    """C_b(r), r=1..K, via Gram trick. U [K,K] columns = modes u_k
    (descending λ). b_off [K,K]. Returns [K] in (0, 1], C_b(K)=1."""
    Bm = U.T @ b_off @ U            # [K,K] — mode coefficients
    M = Bm * Bm                     # (u_kᵀ b_off u_k')² — sign-invariant
    tot = M.sum()
    if tot < 1e-30:
        return np.zeros(M.shape[0])
    cum = np.cumsum(np.cumsum(M, axis=1), axis=0)
    return np.diag(cum) / tot


def off2_cum(P, U, lam):
    """E-30b algebra EXACTLY as experiments/exp30_mode_decomposition.py
    (decompose_batched), one position: off²(r)/off²(K), r=1..K.
    P [K] token distribution (P128), U [K,K] modes (desc), lam [K] (desc)."""
    pe = U.T @ P                                     # (u_kᵀ P)_k
    W = P[:, None] * (U - pe[None, :])               # [tok, mod] w_k[i]
    G = (W.T @ W) ** 2 * np.outer(lam, lam)          # λ_k λ_k' (w_k·w_k')²
    nA2 = np.diag(np.cumsum(np.cumsum(G, axis=1), axis=0))   # ‖A_r‖²
    Dc = np.cumsum(lam[None, :] * (W ** 2), axis=1)  # [tok, mod]
    ndA2 = (Dc * Dc).sum(0)                          # ‖diag A_r‖² (E-30 proxy)
    off2 = nA2 - ndA2
    offK = off2[-1]
    if offK < 1e-30:
        return np.zeros(S)
    return off2 / offK


def spearman(x, y):
    """Spearman = Pearson over ranks. Centering in corrcoef removes the rank
    constant shift; NO index-dependent subtraction (that corrupts the value)."""
    n = len(x)
    if n < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_cubic(eps, y):
    """y = a*e + b*e² + c*e³ (без const); возврат a1,a2,a3,pC,resid.
    eps [NEPS], y [NEPS] -> scalars."""
    A = np.stack([eps, eps ** 2, eps ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = pred - y
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    pC = 1.0 - ss_res / max(ss_tot, 1e-30)
    return coef, pC, resid


def fit_all(eps_abs, dz):
    """Векторизованный cubic fit, ТОЖДЕСТВЕННО exp32r2_hessian.fit_all
    (float64). eps_abs [T,NEPS] = EPS_REL[:,None]*norms[None,:],
    dz [T,K,K,NEPS] -> a1,a2,a3 [T,K,K]."""
    e = eps_abs.astype(np.float64)
    Z = dz.astype(np.float64)
    nrm = e[:, 0] / EPS_REL[0]                                   # [T]
    er = np.asarray(EPS_REL, dtype=np.float64)
    A0 = np.stack([er, er ** 2, er ** 3], -1)                    # [NEPS,3]
    inv0 = np.linalg.inv(A0.T @ A0)
    Atb = np.einsum('eu,tije->tuij', A0, Z)                      # [T,3,K,K]
    coef_rel = np.einsum('pq,tqij->tpij', inv0, Atb)
    n1 = 1.0 / nrm
    a1 = coef_rel[:, 0] * n1[:, None, None]
    a2 = coef_rel[:, 1] * (n1[:, None, None] ** 2)
    a3 = coef_rel[:, 2] * (n1[:, None, None] ** 3)
    return a1, a2, a3


def load_decoder(U128):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation='eager').eval()
    W = model.lm_head.weight.detach().float().numpy()
    g = (1 + model.model.norm.weight.detach().float()).numpy()
    eps_norm = float(model.model.norm.eps)
    W_U = W[U128]
    del model
    g_t = torch.from_numpy(g.astype(np.float32))
    W_U_t = torch.from_numpy(W_U.astype(np.float32)).T

    def forward(hs):  # hs [n, D] f32 -> logits [n, 128]
        x = torch.from_numpy(hs.astype(np.float32))
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps_norm)
        x = (x * g_t).float()
        with torch.no_grad():
            lg = x @ W_U_t
        return lg.numpy()

    return forward, W_U, g, eps_norm


def analytic_J(W_U, g, h):
    """Jacobian of the decoder tail at h (RMSNorm + lm_head on U128 slice).
    Same formula as exp32r2_hessian / exp32r3_1: sqrt(D)/r · (M − (M u)uᵀ),
    M = W_U·g. Returns [K, D] (rows = token directions)."""
    r = np.linalg.norm(h)
    u = h / r
    M = W_U * g[None, :]
    return math.sqrt(D) / r * (M - (M @ u)[:, None] * u[None, :])


# --------------------------------------------------------------------------
# per doc
# --------------------------------------------------------------------------
def doc_layer(forward, W_U, g, ck, dom_si):
    """One doc × one layer: injections -> fit -> a2 [T,S,S]; per-position
    C_b(t) [T,S] (non-sentinel), off2_cum(t) [T,S] (all, E-30b control).
    Returns per-doc: cb_med [S], off2_med [S], per-doc spearman, n_valid."""
    (dom, si) = dom_si
    Js, Ps, Raws, Hs = [], [], [], []
    for t in range(T_FULL):
        k = (dom, si, t)
        Js.append(ck['JR'][k].numpy())
        Ps.append(ck['P128'][k].numpy())
        Raws.append(ck['P128_RAW'][k].numpy())
        Hs.append(ck['H'][k].numpy())
    JR = np.stack(Js).astype(np.float64)
    P128 = np.stack(Ps).astype(np.float64)
    RAW = np.stack(Raws).astype(np.float64)
    H = np.stack(Hs).astype(np.float64)
    T = JR.shape[0]
    del Js, Ps, Raws, Hs

    sent = RAW.sum(-1) < 0.5
    norms = np.linalg.norm(H, axis=1)
    eps_abs = np.array(EPS_REL)[:, None] * norms[None, :]      # [NEPS, T]

    Jn = JR / np.maximum(np.linalg.norm(JR, axis=-1)[:, :, None], 1e-12)
    H32 = H.astype(np.float32)
    Jn32 = Jn.astype(np.float32)
    scaled = eps_abs.T.astype(np.float32)[:, None, :, None] * Jn32[:, :, None, :]
    h_inj = (H32[:, None, None, :] + scaled).reshape(-1, D)
    del scaled, H32, Jn32, Jn
    base_all = forward(H)
    K = S
    dz32 = np.empty((T, K, K, NEPS), dtype=np.float32)
    for t_ in range(T):
        hc = h_inj[t_ * K * NEPS:(t_ + 1) * K * NEPS]
        out = [forward(hc[c0:c0 + CHUNK]) for c0 in range(0, hc.shape[0], CHUNK)]
        L = np.concatenate(out).reshape(K, NEPS, S).transpose(0, 2, 1)
        dz32[t_] = L - base_all[t_][None, :, None]
    del h_inj, base_all
    dz_all = np.ascontiguousarray(dz32).astype(np.float64)
    del dz32

    _, a2, _ = fit_all(eps_abs.T, dz_all)                    # b = a2 [T,S,S]
    del dz_all

    # ESG eigendecomposition per position (E-30 convention: desc, u_1 = top).
    # Primary (TЗ): U_full from full-path JR; control (self-consistent): U_tail
    # from the ANALYTIC decoder-tail Jacobian at the same H (~ms, no forwards).
    cb_t = np.zeros((T, S))
    off2_t = np.zeros((T, S))
    cb_tail_t = np.zeros((T, S))
    off2_tail_t = np.zeros((T, S))
    for t_ in range(T):
        ESG = JR[t_] @ JR[t_].T
        lam, U = np.linalg.eigh(ESG)
        lam = lam[::-1]
        U = U[:, ::-1]
        b_off = a2[t_] - np.diag(np.diag(a2[t_]))
        cb_t[t_] = cb_curve(b_off, U)
        off2_t[t_] = off2_cum(P128[t_], U, lam)
        # self-consistent tail basis: J_tail = analytic decoder Jacobian at H[t_]
        Jt = analytic_J(W_U, g, H[t_])                       # [K, D]
        ESG_t = Jt @ Jt.T
        lam_t, U_t = np.linalg.eigh(ESG_t)
        lam_t = lam_t[::-1]
        U_t = U_t[:, ::-1]
        cb_tail_t[t_] = cb_curve(b_off, U_t)
        off2_tail_t[t_] = off2_cum(P128[t_], U_t, lam_t)
    del a2, JR

    v = ~sent
    if v.sum() == 0:
        v = np.ones(T, dtype=bool)
    cb_med = np.median(cb_t[v], axis=0)
    off2_med = np.median(off2_t[v], axis=0)
    cb_tail_med = np.median(cb_tail_t[v], axis=0)
    off2_tail_med = np.median(off2_tail_t[v], axis=0)
    # E-30b protocol match: E-30b's sentinel test ran on the NORMALIZED P128
    # (sum == 1.0 everywhere -> no positions excluded), so its reference
    # curve aggregates ALL 128 positions. Primary C_b comparison uses the
    # non-sentinel curves (TЗ §3.3); the all-positions curve is control-only.
    off2_med_all = np.median(off2_t, axis=0)
    # per-doc Spearman over INCREMENTS (cumulative curves are degenerate:
    # C_b strictly increasing -> pointwise ranks 1..K -> Spearman ≡ 1)
    sp = spearman(np.diff(cb_med), np.diff(off2_med))
    sp_tail = spearman(np.diff(cb_tail_med), np.diff(off2_tail_med))
    # per-position spearman distribution (extra robustness)
    sp_pos = []
    for t_ in np.where(v)[0]:
        s = spearman(np.diff(cb_t[t_]), np.diff(off2_t[t_]))
        if np.isfinite(s):
            sp_pos.append(s)
    return {
        'cb_med': cb_med.tolist(),
        'off2_med': off2_med.tolist(),
        'off2_med_all': off2_med_all.tolist(),
        'cb_tail_med': cb_tail_med.tolist(),
        'off2_tail_med': off2_tail_med.tolist(),
        'spearman_doc_mean': sp,
        'spearman_tail_doc_mean': sp_tail,
        'spearman_pos_med': float(np.median(sp_pos)) if sp_pos else None,
        'spearman_pos_q1': float(np.percentile(sp_pos, 25)) if sp_pos else None,
        'spearman_pos_q3': float(np.percentile(sp_pos, 75)) if sp_pos else None,
        'n_valid': int(v.sum()),
        'sentinel_fraction': float(sent.mean()),
    }


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
def _iqr(arr):
    return [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))]


def _block(cb_docs, off2_docs, off2_all_docs=None):
    """Aggregate one (C_b, off2) pair: curves, weights, top-5, tail, spearman."""
    cb = np.array(cb_docs)
    off2 = np.array(off2_docs)
    cb_med = np.median(cb, axis=0)
    cb_q = np.percentile(cb, [25, 75], axis=0)
    off2_med = np.median(off2, axis=0)
    off2_q = np.percentile(off2, [25, 75], axis=0)
    w_b = np.diff(cb_med, prepend=0.0)                        # [128], ΔC_b
    w_b[0] = cb_med[0]
    top5_b = np.argsort(w_b)[::-1][:5] + 1
    w_off = np.diff(off2_med, prepend=0.0)
    w_off[0] = off2_med[0]
    top5_off = np.argsort(w_off)[::-1][:5] + 1
    overlap = len(set(top5_b) & set(top5_off))
    tail_share_b = float(1.0 - cb_med[118])                   # tail = r∈[119,128]
    tail_share_off = float(1.0 - off2_med[118])
    # C_b is strictly increasing (sum of squares) -> pointwise ranks are
    # trivially 1..K -> Spearman(C_b, C_off2) degenerates to 1 (C_off2 is
    # non-monotone, E-30b neg_frac). The METRIC is on INCREMENTS (TЗ intent:
    # "which modes carry the weight"), TЗ §3.1 line 57 compared curves, but
    # that comparison is degenerate; increments are the non-degenerate object.
    dcb = np.diff(cb_med)
    doff = np.diff(off2_med)
    sp_agg = spearman(dcb, doff)
    sp_agg_cum = spearman(cb_med, off2_med)                   # degenerate, for audit
    cb_inc = np.diff(cb, axis=1)                              # [32,127]
    neg_frac = np.zeros(S)
    neg_frac[1:] = (cb_inc < 0).mean(0)
    out = {
        'C_b_r': [float(x) for x in cb_med],
        'C_b_q1': [float(x) for x in cb_q[0]],
        'C_b_q3': [float(x) for x in cb_q[1]],
        'C_off2_r': [float(x) for x in off2_med],
        'C_off2_q1': [float(x) for x in off2_q[0]],
        'C_off2_q3': [float(x) for x in off2_q[1]],
        'w_b': [float(x) for x in w_b],
        'top5_b_modes': [int(x) for x in top5_b],
        'top5_off2_modes': [int(x) for x in top5_off],
        'overlap_top5': int(overlap),
        'tail_share_b': tail_share_b,
        'tail_share_off2': tail_share_off,
        'spearman_agg': sp_agg,
        'spearman_agg_cumulative': sp_agg_cum,
        'neg_frac_per_r_b': [float(x) for x in neg_frac],
        'delta_C_b': [float(x) for x in dcb],
        'delta_C_off2': [float(x) for x in doff],
    }
    if off2_all_docs is not None:
        out['off2_all_med'] = np.median(
            np.array(off2_all_docs), axis=0).tolist()
    return out


def aggregate_layer(docs):
    full = _block([d['cb_med'] for d in docs], [d['off2_med'] for d in docs],
                  [d['off2_med_all'] for d in docs])
    tail = _block([d['cb_tail_med'] for d in docs], [d['off2_tail_med'] for d in docs])
    sp_docs = [d['spearman_doc_mean'] for d in docs if d['spearman_doc_mean'] is not None
               and np.isfinite(d['spearman_doc_mean'])]
    sp_tail_docs = [d['spearman_tail_doc_mean'] for d in docs
                    if d['spearman_tail_doc_mean'] is not None
                    and np.isfinite(d['spearman_tail_doc_mean'])]
    sp_pos = [d['spearman_pos_med'] for d in docs if d['spearman_pos_med'] is not None]
    out = {
        'full': full,                                         # TЗ object: b in full-path ESG basis
        'tail': tail,                                         # control: b in tail-ESG basis (self-consistent)
        'spearman_docs_med': float(np.median(sp_docs)) if sp_docs else None,
        'spearman_docs_q1': float(np.percentile(sp_docs, 25)) if sp_docs else None,
        'spearman_docs_q3': float(np.percentile(sp_docs, 75)) if sp_docs else None,
        'spearman_tail_docs_med': float(np.median(sp_tail_docs)) if sp_tail_docs else None,
        'spearman_pos_med': float(np.median(sp_pos)) if sp_pos else None,
        'basis_agreement': full['spearman_agg'] > 0 and tail['spearman_agg'] > 0,
        'n_docs': len(docs),
    }
    return out


def verdict(block, e30b_ref):
    """Apply T1/T2/T3 to ONE basis block (dict from _block)."""
    sp = block['spearman_agg']
    ov = block['overlap_top5']
    ts = block['tail_share_b']
    if e30b_ref is not None and len(e30b_ref) > S:
        ref_ts = 1.0 - float(np.asarray(e30b_ref)[118])       # tail = r∈[119,128]
    else:
        ref_ts = block['tail_share_off2']
    w = np.array(block['w_b'])
    if sp > 0.9 and ov >= 3 and abs(ts - ref_ts) < 0.25:
        return 'T1_inheritance'
    if sp < 0.5 and ov <= 1:
        return 'T2_independence'
    # T3: no mode dominates (max increment < 2× uniform 1/K) and no tail share
    if ts < 0.30 and w.max() < 2.0 / S:
        return 'T3_diffuse'
    return 'mixed'


# --------------------------------------------------------------------------
# selfcheck
# --------------------------------------------------------------------------
def selfcheck():
    rng = np.random.default_rng(7)
    K = S
    U, _ = np.linalg.qr(rng.standard_normal((K, K)))
    b = rng.standard_normal((K, K))
    b_off = b - np.diag(np.diag(b))

    # 1. Gram trick == brute force ‖P_r b_off P_r‖²/‖b_off‖² at r ∈ {1,5,32,64,128}
    C = cb_curve(b_off, U)
    tot = (b_off ** 2).sum()
    for r in (1, 5, 32, 64, 128):
        P = U[:, :r] @ U[:, :r].T
        br = P @ b_off @ P
        assert abs(C[r - 1] - (br ** 2).sum() / tot) < 1e-9, f'gram r={r}'
    # 2. C(K) = 1, C non-decreasing, C(1) ≥ 0
    assert abs(C[-1] - 1.0) < 1e-12 and (np.diff(C) >= -1e-12).all()
    # 3. sign invariance
    Us = U * np.array([1, -1, 1, -1, 1] + [1] * (K - 5))
    assert np.allclose(cb_curve(b_off, Us), C, atol=1e-12)
    # 4. zero b_off -> zeros
    assert np.allclose(cb_curve(np.zeros((K, K)), U), 0)
    # 5. off2_cum: synthetic sanity — well-defined in [0,1], ends at 1;
    #    and for a rank-1 P (one-hot token) the gauge-free part vanishes
    #    (w_k = P∘(u_k−pe_k·P) with P=one-hot_i: w_k = e_i (u_k[i]−u_k[i]) = 0)
    Pp = rng.dirichlet(np.ones(K))
    lam = np.sort(rng.standard_normal(K))[::-1].clip(1e-6)
    o2 = off2_cum(Pp, U, lam)
    assert abs(o2[-1] - 1.0) < 1e-12 and (o2 >= -1e-12).all()
    Pone = np.zeros(K); Pone[3] = 1.0
    assert np.allclose(off2_cum(Pone, U, lam), 0.0, atol=1e-12)
    # 6. spearman sanity: identity, anti, monotone invariance
    x = rng.standard_normal(50)
    assert abs(spearman(x, x) - 1.0) < 1e-9
    assert abs(spearman(x, -x) + 1.0) < 1e-9
    assert abs(spearman(x, x ** 3) - 1.0) < 1e-9
    # 7. fit_all reproduces coefficients (absolute eps = er*n, n random per t)
    er = np.asarray(EPS_REL, dtype=np.float64)
    t_, k_ = 2, 3
    a1t, a2t, a3t = (rng.standard_normal((k_, k_)) for _ in range(3))
    nrm = rng.uniform(0.5, 2.0, t_)
    en = er[None, None, None, :] * nrm[:, None, None, None]     # [T,1,1,NEPS] = abs e
    dz_t = a1t[:, :, None] * en + a2t[:, :, None] * en ** 2 \
        + a3t[:, :, None] * en ** 3                          # [T,k,k,NEPS]
    eps_abs = er[None, :] * nrm[:, None]
    a1, a2, a3 = fit_all(eps_abs, dz_t)
    assert np.allclose(a2, a2t, atol=1e-9) and np.allclose(a1, a1t, atol=1e-9)
    print('selfcheck: 7/7 OK')


# --------------------------------------------------------------------------
def main(layers=None, out=None):
    global OUT
    torch.set_num_threads(12)
    layers = list(layers) if layers else list(PILOT)
    out = out or OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    e30b = {}
    if os.path.isfile(E30B):
        e30b_raw = json.load(open(E30B))
        for l in layers:
            if str(l) in e30b_raw.get('layers', {}):
                e30b[l] = e30b_raw['layers'][str(l)]['off2_cum_med']

    res = {'meta': {
        'experiment': 'E-32-tail (docs/E32t.md) — token-space localization of b',
        'model': MODEL, 'layers': layers, 'T': T_FULL,
        'fit': 'dz = a1*e + a2*e² + a3*e³ (без const), b = a2 (TЗ normalization)',
        'OBJECT_HONESTY': ('b = Hessian of the DECODER-TAIL readout (RMSNorm+lm_head at '
                           'layer l, E-32r.2 pipeline) along unit injections Jn = JR/‖JR‖; '
                           'U/off² from FULL-PATH ESG = JR·JRᵀ (E-30b). TЗ says "b уже '
                           'посчитан" = E-32r.2, i.e. exactly this tail object. For l=23 '
                           'full≡tail (cos=1.0000, E-32r.3.1); cos(JR,J_tail): L6 0.105, '
                           'L12 0.237, L18 0.598, L19 0.642, L22 0.789 — so C_b at L18/L22 '
                           'is "tail-Hessian in full-path token modes", a defined mixed '
                           'object, NOT the full-path Hessian.'),
        'C_b': '‖P_r b_off P_r‖_F²/‖b_off‖_F², Gram trick, u_k desc ESG eigenvectors',
        'off2_cum': 'per-doc E-30b algebra recomputed from the same jcache (control)',
        'aggregation': 'E-34: per-doc median over non-sentinel t -> median/IQR over docs',
        'spearman': ('Pearson over ranks (corrcoef) of the INCREMENTS ΔC_b vs ΔC_off2 '
                     '(TЗ line 57 says "curves" but C_b is strictly increasing so '
                     'curve-Spearman degenerates to 1.0; increments are the intended, '
                     'non-degenerate object; cumulative version kept as '
                     'spearman_agg_cumulative for audit); per-doc over doc-median curves'),
        'neg_frac_per_r_b': 'TRIVIALLY 0 (C_b is a cumulative sum of squares) — informational only',
        'hidden_space_sweep': 'NOT computed (needs bilinear H probes, O(K²) injections) — TЗ secondary',
        'e30b_ref': {str(l): len(e30b.get(l, [])) > 0 for l in layers},
    }, 'layers': {}, 'per_doc': {}}
    t0 = time.time()
    for l in layers:
        ck = torch.load(JCACHE[l], map_location='cpu', weights_only=False)
        U128 = list(ck['U128'])
        forward, W_U, g, _ = load_decoder(U128)
        # keys are (dom, si, t); group by (dom, si)
        doms = sorted({(k[0], k[1]) for k in ck['JR']})
        agg_docs = []
        for di, ds in enumerate(doms):
            d = doc_layer(forward, W_U, g, ck, ds)
            res['per_doc'][f'{l}|{ds[0]}/{ds[1]}'] = d
            agg_docs.append(d)
            print(f'  L{l} {ds[0]}/{ds[1]}: n_valid={d["n_valid"]} '
                  f'sp_full={d["spearman_doc_mean"]:.3f} sp_tail={d["spearman_tail_doc_mean"]:.3f}',
                  flush=True)
        agg = aggregate_layer(agg_docs)
        # control: my aggregated off2 curve (full basis) vs E-30b (same corpus)
        if l in e30b:
            ref = np.array(e30b[l])
            # protocol-matched control: E-30b aggregated over ALL 128 positions
            # (its sentinel test ran on normalized P128, sum == 1 -> vacuous)
            agg['off2_vs_e30b_maxabs'] = float(np.max(np.abs(
                np.array(agg['full']['off2_all_med']) - ref)))
            agg['off2_vs_e30b_spearman'] = spearman(
                agg['full']['off2_all_med'], ref)
        agg['verdict_full'] = verdict(agg['full'], e30b.get(l))
        agg['verdict_tail'] = verdict(agg['tail'], e30b.get(l))
        res['layers'][str(l)] = agg
        F, Tb = agg['full'], agg['tail']
        print(f'L{l} full: {agg["verdict_full"]} sp={F["spearman_agg"]:.3f} '
              f'overlap={F["overlap_top5"]} tail_b={F["tail_share_b"]:.3f} top5={F["top5_b_modes"]}',
              flush=True)
        print(f'L{l} tail: {agg["verdict_tail"]} sp={Tb["spearman_agg"]:.3f} '
              f'overlap={Tb["overlap_top5"]} tail_b={Tb["tail_share_b"]:.3f} top5={Tb["top5_b_modes"]}',
              flush=True)
    res['meta']['runtime_s'] = int(time.time() - t0)
    json.dump(res, open(out, 'w'), indent=1)
    print(f'saved {out} in {res["meta"]["runtime_s"]}s')


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
        sys.exit(0)
    args = sys.argv[1:]
    layers = None
    if '--layers' in args:
        i = args.index('--layers')
        layers = [int(x) for x in args[i + 1:] if x.isdigit()]
    main(layers)
