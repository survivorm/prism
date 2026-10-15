"""E-32r.3 — Hessian of the decoder readout: layer profile.

Supersedes E-32r.2 on the shared layers (6, 12, 18, 22, 23) and adds L19:
  * LAYERS = (6, 12, 18, 19, 22, 23) — ALL layers of the E-27b dense cache
    (L19 verified present: 32 docs x 128 t; assert in main);
  * full T=128 on EVERY layer (E-32r.2 used T=64 on non-L23);
  * N=32 docs (the entire E-27b dense corpus — there is nothing else);
  * slope_ratio: E-32r.2 formula UNCHANGED (|dz(e0)|/e0 over |dz(e1)|/e1 —
    linear-regime check, 1.0 = linear; identical for cross-version comparison);
  * quad_frac: NEW metric (no E-32r.2 counterpart) — (s1 - s2)/|s2| with
    s1=||a1||, s2=||a1 + 2*(eps0*norms)*a2||: relative quadratic contribution
    at the smallest eps (0 = quadratic term negligible);
  * d3_fixP: p0 normalized to sum 1 (softmax-Jacobian identity requires it);
  * float32 storage for the injection residuals (dz32, no float64 copy),
    fit in float64 (upcast inside fit_all) — per review point 4;
  * CHUNK = 4096 (E-32r.2: 512) — Python-overhead is the bottleneck, not FLOP;
  * GPU forward (.to(dev)) — E-32r.2 ran on 1-thread CPU (6142 s); the same
    pointwise decoder-tail matmul is seconds on a T4.

Pipeline otherwise UNCHANGED from E-32r.2 (docs/Exp32.md + docs/Exp32r_errors.md):
  16-point relative eps sweep (5%..70% of ||h_t||), cubic fit
  dz = a1*e + a2*e^2 + a3*e^3 (no const, absolute eps), per-doc median ->
  median/IQR over 32 docs, bootstrap CI (seed 12345), d3 top-r sweep,
  Q1/Q4 |dp| stratification, OLS+HC3, sentinel masks, E-35 identity check.

Review point 3 (docs/E32r_suggestions.md + reviewer 2026-07-22):
  a_vs_ESG_cos is a FIT-QUALITY sanity check (chain rule a1 = J e, exact for
  any J when the cubic fit is good), expected ~1 on every layer with small
  resid_frac. It is NOT a depth metric. The depth metrics of this profile are
  s_l(l), d3_raw(1,l), d3_gf(1,l) (plus d3_fixP(1,l) as diagnostics only on
  mid layers, per TЗ §9: J = M_K*P_u is exact only at L23).

Self-contained: no prism.* / exp28 imports (kernel inlines this file alone).
Kaggle T4: model Qwen/Qwen3.5-0.8B (decoder tail only), cache from the
prism-exp27b-dense-profile kernel output at /kaggle/working/results.
"""
import math
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from scipy.stats import t as tdist

S = 128
D = 1024
LAYERS = (6, 12, 18, 19, 22, 23)  # sorted: monotone depth profile
T_FULL = 128
EPS_REL = np.round(np.logspace(np.log10(0.05), np.log10(0.70), 16), 6)
NEPS = len(EPS_REL)
EPS_CANON = 0.15
R_SWEEP = (1, 2, 4, 8, 16)
BOOT_N = 64
PERM_N = 1000
MODEL = 'Qwen/Qwen3.5-0.8B'
OUT = 'results/exp32r3_layer_profile.json'  # separate file: E-32r.2 kept
CHUNK = 4096  # review point 2/6: T4-safe (16 MB activations, 2 MB output)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------
# math helpers (verbatim from E-32r.2)
# --------------------------------------------------------------------------
def offdiag_norm(M):
    return float(math.sqrt(max((M * M).sum() - (np.diag(M) ** 2).sum(), 0.0)))


def gauge_free_rows(J, p):
    """P(p) = diag(p) - pp^T in token space: J [K,D] rows=tokens."""
    return J * p[:, None] - (p @ J)[None, :] * p[:, None]


def top_r_right(J, r):
    """Top-r right singular vectors [r, D] (rows = directions in D-space)."""
    U, _, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[:r, :]


def _right_angle(V1, V2):
    """Largest principal angle (rad): max(acos(svdvals(V1 @ V2.T))) — same
    estimator as exp33r delta_svd."""
    sv = np.linalg.svdvals(V1 @ V2.T)
    return float(math.acos(float(np.clip(sv.max(), 0.0, 0.999999999999))))


def d3_top_r(Jseq, Pseq, r, mode, p0=None):
    """d3 = per-pair top-r largest principal angle (rad), mode raw/gf/fixP
    (p0 — canonical, passed EXPLICITLY). Returns [T-1]."""
    out = []
    for t in range(len(Jseq) - 1):
        A = Jseq[t]
        B = Jseq[t + 1]
        if mode == 'gf':
            A = gauge_free_rows(A, Pseq[t])
            B = gauge_free_rows(B, Pseq[t + 1])
        elif mode == 'fixP':
            A = gauge_free_rows(A, p0)
            B = gauge_free_rows(B, p0)
        V1 = top_r_right(A, r)
        V2 = top_r_right(B, r)
        out.append(_right_angle(V1, V2))  # single acos
    return np.array(out)


def ols_hc3(x, y, rng=None, n_perm=PERM_N):
    """OLS y ~ alpha*x + beta, HC3 standard errors; permutation p for alpha."""
    n = len(x)
    if n < 5:
        return {'alpha': None, 'beta': None, 'alpha_se': None,
                'alpha_p': None, 'perm_p': None}
    X = np.stack([np.ones(n), x], axis=1)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    resid = y - X @ beta_hat
    H = X @ XtX_inv @ X.T
    lev = np.diag(H)
    hc3 = np.sqrt(np.maximum(np.diag(XtX_inv) * float(
        (resid ** 2 / np.maximum(1 - lev, 1e-12) ** 2).sum()), 0.0))
    tstat = beta_hat[1] / hc3[1] if hc3[1] > 0 else 0.0
    alpha_p = float(2 * tdist.sf(abs(tstat), n - 2))
    if rng is None:
        perm_p = None
    else:
        obs = abs(tstat)
        cnt = 0
        for _ in range(n_perm):
            yp = y[rng.permutation(n)]
            bp = np.linalg.solve(X.T @ X, X.T @ yp)
            rp = yp - X @ bp
            h3 = np.sqrt(np.maximum(np.diag(XtX_inv) * float(
                (rp ** 2 / np.maximum(1 - lev, 1e-12) ** 2).sum()), 0.0))
            if h3[1] > 0 and abs(bp[1] / h3[1]) >= obs:
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)
    return {'alpha': float(beta_hat[1]), 'beta': float(beta_hat[0]),
            'alpha_se': float(hc3[1]), 'alpha_p': alpha_p, 'perm_p': perm_p}


def fit_cubic(eps, y):
    """y = a*e + b*e^2 + c*e^3 (no const); returns a1,a2,a3,pC,resid."""
    e = np.asarray(eps, dtype=np.float64)
    Z = np.asarray(y, dtype=np.float64)
    A = np.stack([e, e ** 2, e ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
    res = Z - A @ coef
    dof = Z.size - A.shape[1]
    s2 = float(res @ res) / dof
    se = math.sqrt(max(s2 * np.linalg.inv(A.T @ A)[2, 2], 0.0))
    tstat = coef[2] / se if se > 0 else 0.0
    return float(coef[0]), float(coef[1]), float(coef[2]), \
        float(2 * tdist.sf(abs(tstat), dof)), res


def fit_all(eps_abs, dz):
    """Vectorized cubic fit. eps_abs [T,NEPS], dz [T,K,K,NEPS] (float32 ok).

    Fit in float64 (upcast of the float32 residuals) — review point 4.
    Returns a1,a2,a3,pC [T,K,K] (float32), resid [T,K,K,NEPS] (float64).
    """
    e = eps_abs.astype(np.float64)
    Z = dz.astype(np.float64) if dz.dtype != np.float64 else dz  # float32->64
    nrm = e[:, 0] / EPS_REL[0]                                   # [T]
    er = np.asarray(EPS_REL, dtype=np.float64)
    A0 = np.stack([er, er ** 2, er ** 3], -1)                    # [NEPS,3]
    inv0 = np.linalg.inv(A0.T @ A0)
    Atb = np.einsum('eu,tije->tuij', A0, Z)                      # [T,3,K,K]
    coef_rel = np.einsum('pq,tqij->tpij', inv0, Atb)
    n1 = 1.0 / nrm
    a1 = (coef_rel[:, 0] * n1[:, None, None]).astype(np.float32)
    a2 = (coef_rel[:, 1] * (n1[:, None, None] ** 2)).astype(np.float32)
    a3 = (coef_rel[:, 2] * (n1[:, None, None] ** 3)).astype(np.float32)
    resid = Z - (a1[:, :, :, None] * e[:, None, None, :]
                 + a2[:, :, :, None] * e[:, None, None, :] ** 2
                 + a3[:, :, :, None] * e[:, None, None, :] ** 3)
    s2v = (resid ** 2).sum(-1) / max(NEPS - 3, 1)
    se_rel = np.sqrt(np.maximum(s2v * inv0[2, 2], 0.0))
    tstat = np.zeros_like(coef_rel[:, 2])
    np.divide(coef_rel[:, 2], se_rel, out=tstat, where=se_rel > 0)
    pC = (2 * tdist.sf(np.abs(tstat), NEPS - 3)).astype(np.float32)
    return a1, a2, a3, pC, resid


def pick_t0(P128, sent):
    """Canonical p0 position: non-sentinel position with p closest to median."""
    ok = ~sent
    if not ok.any():
        ok = np.ones_like(sent)
    med = np.median(P128[ok], 0)
    sel = int(np.argmin(np.linalg.norm(P128[ok] - med[None, :], axis=1)))
    return int(np.where(ok)[0][sel])


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# --------------------------------------------------------------------------
# model (decoder tail on device — review point 2: .to(dev)/.cpu())
# --------------------------------------------------------------------------
def load_decoder(U128, model_dir):
    """RMSNorm + g + lm_head[U128] as float32 torch tensors on DEVICE.

    Returns (forward, W_U, g, eps_norm); forward: [n, D] f32 -> [n, K] f32.
    """
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1)  # CPU fallback only; GPU path ignores this
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation='eager').eval()
    W = model.lm_head.weight.detach().float().numpy()
    g = (1 + model.model.norm.weight.detach().float()).numpy()
    eps_norm = float(model.model.norm.eps)
    W_U = W[U128]
    del model
    g_t = torch.from_numpy(g.astype(np.float32)).to(DEVICE)
    W_U_t = torch.from_numpy(W_U.astype(np.float32).T).to(DEVICE)

    @torch.no_grad()
    def forward(hs):  # hs [n, D] f32 numpy -> logits [n, K] f32 numpy
        x = torch.from_numpy(hs.astype(np.float32)).to(DEVICE)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps_norm)
        x = x * g_t
        lg = x @ W_U_t
        return lg.cpu().numpy()

    return forward, W_U, g, eps_norm


def analytic_J(W_U, g, h):
    r = np.linalg.norm(h)
    u = h / r
    M = W_U * g[None, :]
    return math.sqrt(D) / r * (M - (M @ u)[:, None] * u[None, :])


# --------------------------------------------------------------------------
def doc_layer_metrics(forward, ck, dom_si, T):
    """One doc x one layer: all injections, fit, all TЗ metrics. T=128.

    float32 storage for dz32 (review point 4: NO float64 copy of dz32);
    fit_all upcasts to float64 internally. CHUNK=4096 GPU forward.
    """
    (dom, si) = dom_si
    Js, Ps, Raws, Hs = [], [], [], []
    for t in range(T):
        k = (dom, si, t)
        Js.append(ck['JR'][k].numpy())
        Ps.append(ck['P128'][k].numpy())
        Raws.append(ck['P128_RAW'][k].numpy())
        Hs.append(ck['H'][k].numpy())
    JR = np.stack(Js).astype(np.float64)
    P128 = np.stack(Ps).astype(np.float64)
    RAW = np.stack(Raws).astype(np.float64)
    H = np.stack(Hs).astype(np.float64)

    sent = RAW.sum(-1) < 0.5
    vp = ~sent[:-1] & ~sent[1:]
    norms = np.linalg.norm(H, axis=1)
    eps_abs = np.array(EPS_REL)[:, None] * norms[None, :]  # [NEPS, T]

    # --- injections: h'[t,i,e] = H[t] + eps_abs[e,t] * Jn[t,i] ---
    Jn = JR / np.maximum(np.linalg.norm(JR, axis=-1)[..., None], 1e-12)
    H32 = H.astype(np.float32)
    Jn32 = Jn.astype(np.float32)
    scaled = eps_abs.T.astype(np.float32)[:, None, :, None] * Jn32[:, :, None, :]
    h_inj = (H32[:, None, None, :] + scaled).reshape(-1, D)
    del scaled, Jn32, Jn
    # base_all: one forward over T positions
    outs = []
    for c0 in range(0, T, CHUNK):
        outs.append(forward(H32[c0:c0 + CHUNK]))
    base_all = np.concatenate(outs)
    del H32
    K = S  # vocab slice (logits); T — positions (Fix v2: stride/reshape by K)
    dz32 = np.empty((T, K, K, NEPS), dtype=np.float32)
    for t_ in range(T):
        hc = h_inj[t_ * K * NEPS:(t_ + 1) * K * NEPS]
        out = []
        for c0 in range(0, hc.shape[0], CHUNK):
            out.append(forward(hc[c0:c0 + CHUNK]))
        L = np.concatenate(out).reshape(K, NEPS, S).transpose(0, 2, 1)
        dz32[t_] = L - base_all[t_][None, :, None]
    del h_inj, base_all

    a1, a2, a3, pC, resid = fit_all(eps_abs.T, dz32)  # float64 fit inside
    A1 = a1 + 2.0 * a2 * (EPS_CANON * norms)[:, None, None]

    # --- s_t / slope / residual (TЗ «Математика», raw normalization) ---
    ESG_all = np.einsum('std,sjd->stj', JR, JR)
    ESGd = np.diagonal(ESG_all, axis1=1, axis2=2)
    off_esg = np.sqrt(np.maximum((ESG_all ** 2).sum((-1, -2)) - (ESGd ** 2).sum(-1), 0.0))
    nb = np.array([offdiag_norm(a2[t]) for t in range(T)])
    s_t = EPS_CANON * norms * nb / np.maximum(off_esg, 1e-12)
    # slope_ratio: E-32r.2 formula UNCHANGED (comparability): ratio of
    # |dz/eps| at the two smallest eps; 1.0 = linear regime.
    e0v = EPS_REL[0] * norms
    e1v = EPS_REL[1] * norms
    slope0 = np.abs(dz32[..., 0]) / np.maximum(e0v[:, None, None], 1e-12)
    slope1 = np.abs(dz32[..., 1]) / np.maximum(e1v[:, None, None], 1e-12)
    slope = slope0 / np.maximum(slope1, 1e-12)
    slope_ratio_med = float(np.median(slope))
    # quad_frac: NEW metric (no E-32r.2 counterpart) — relative quadratic
    # contribution at eps0. Absolute-eps reconstruction:
    # eps_abs = EPS_REL[0] * ||h_t|| (norms).
    c1 = a1 + 2.0 * a2 * (EPS_REL[0] * norms)[:, None, None]
    s1 = np.linalg.norm(a1, axis=-1)
    s2 = np.linalg.norm(c1, axis=-1)
    quad = (s1 - s2) / np.maximum(np.abs(s2), 1e-12)
    quad_frac_med = float(np.median(quad))
    c_sig = float((pC[~sent] < 0.05).mean())

    # Fix (d): a1 vs ESG (Frobenius cosine) — fit sanity check, ~1 expected
    amed = np.median(a1, 0)
    ESG_med = np.median(ESG_all, 0)
    ESGn = ESG_med / np.maximum(
        np.sqrt(np.maximum(np.diag(ESG_med), 0.0))[None, :], 1e-12)
    a_vs_esg = float((amed * ESGn).sum() / max(
        np.linalg.norm(amed) * np.linalg.norm(ESGn), 1e-12))

    resid_frac_t = np.zeros((T, NEPS))  # (T, NEPS), Fix v3 — E-32r.2 semantics:
    # ||resid_f64|| / ||dz_measured||; float32 norm is a denominator, no
    # catastrophic cancellation, so float32 storage loses nothing here.
    for e in range(NEPS):
        num = np.linalg.norm(resid[:, :, :, e], axis=(1, 2))
        den = np.linalg.norm(dz32[:, :, :, e], axis=(1, 2))
        resid_frac_t[:, e] = num / np.maximum(den, 1e-12)
    rfe_all = resid_frac_t
    del dz32, resid

    # --- p0 canonical + d3 (top-r, sweep); p0 NORMALIZED (review point 6) ---
    t0 = pick_t0(P128, sent)
    p0 = P128[t0] / np.maximum(P128[t0].sum(), 1e-30)
    d3 = {m: {r: d3_top_r(JR, P128, r, m, p0) for r in R_SWEEP}
          for m in ('raw', 'gf', 'fixP')}

    # --- Q1/Q4 by |dp| over valid pairs (review #2: n>=16/group) ---
    dp = np.linalg.norm(np.diff(P128, axis=0), axis=1)
    dpv = dp[vp]
    q1 = q4 = None
    mQ1 = mQ4 = None
    bQ1, bQ4 = None, None
    db_rel = None
    if len(dpv) >= 32:
        q1, q4 = np.percentile(dpv, [25, 75])
        Q1 = np.where((dp <= q1) & vp)[0]
        Q4 = np.where((dp >= q4) & vp)[0]
        if len(Q1) >= 16 and len(Q4) >= 16:
            mQ1 = np.median(a2[Q1], 0)
            mQ4 = np.median(a2[Q4], 0)
            bQ1, bQ4 = float(np.linalg.norm(mQ1)), float(np.linalg.norm(mQ4))
            b_mean = float(np.linalg.norm(np.median(a2[:-1][vp], 0)))
            db_rel = float(np.linalg.norm(mQ4 - mQ1) / max(b_mean, 1e-12))

    # --- OLS+HC3: nb(t) ~ alpha*|dp(t)| + beta, valid pairs ---
    tidx = np.where(vp)[0]
    ols = ols_hc3(dp[tidx], nb[tidx]) if len(tidx) >= 5 else None

    # --- 4 cross-checks (TЗ): rho(nb, dSVD_raw/fixP/gf/r=1, |dp|) ---
    def spearman(x, y):
        n = len(x)
        if n < 3:
            return float('nan')
        rx = np.argsort(np.argsort(x)) - (np.arange(n) + 1) / (n + 1)
        ry = np.argsort(np.argsort(y)) - (np.arange(n) + 1) / (n + 1)
        return float(np.corrcoef(rx, ry)[0, 1])

    rho = {
        'rho_nb_dSVD_raw1': spearman(nb[tidx], d3['raw'][1][tidx]),
        'rho_nb_dSVD_fixP1': spearman(nb[tidx], d3['fixP'][1][tidx]),
        'rho_nb_dSVD_gf1': spearman(nb[tidx], d3['gf'][1][tidx]),
        'rho_nb_dp': spearman(nb[tidx], dp[tidx]),
    }

    out = {
        't0': int(t0),
        's_l_median': float(np.median(s_t)),
        's_l_iqr': [float(np.percentile(s_t, 25)), float(np.percentile(s_t, 75))],
        'slope_ratio_median': slope_ratio_med,
        'slope_ratio_iqr': [float(np.percentile(slope, 25)),
                            float(np.percentile(slope, 75))],
        'quad_frac_median': quad_frac_med,
        'quad_frac_iqr': [float(np.percentile(quad, 25)),
                          float(np.percentile(quad, 75))],
        'A1_median': float(np.median(A1)),
        'nb_median': float(np.median(nb[~sent])),
        'frac_c_sig': c_sig,
        'resid_frac_by_eps_med': [float(np.median(resid_frac_t[:, e]))
                                  for e in range(NEPS)],
        'resid_frac_by_eps_max': [float(np.max(resid_frac_t[:, e]))
                                  for e in range(NEPS)],
        'sentinel_fraction': float(sent.mean()),
        'n_valid_pairs': int(vp.sum()),
        'a_vs_ESG_cos': a_vs_esg,
        'd3_by_r': {m: {str(r): [float(np.median(d3[m][r][vp])),
                                [float(np.percentile(d3[m][r][vp], 25)),
                                 float(np.percentile(d3[m][r][vp], 75))]]
                        for r in R_SWEEP} for m in ('raw', 'gf', 'fixP')},
        'Q1_n': int(len(np.where((dp <= (q1 if q1 is not None else 0)) & vp)[0]))
                if q1 is not None else 0,
        'Q4_n': int(len(np.where((dp >= (q4 if q4 is not None else 0)) & vp)[0]))
                if q4 is not None else 0,
        'bQ1_fro': bQ1,
        'bQ4_fro': bQ4,
        'delta_b_rel': db_rel,
        'ols': ols,
        'cross_checks': rho,
    }
    return _clean(out), {'dp_v': dp[vp], 'nb_v': nb[:-1][vp]}


def _pooled_ols(cache, keys):
    """Pooled OLS+HC3: all valid pairs of all docs of one layer."""
    x_all = [cache[k]['dp_v'] for k in keys]
    y_all = [cache[k]['nb_v'] for k in keys]
    if sum(len(a) for a in x_all) < 5:
        return None
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    return ols_hc3(x, y, rng=np.random.default_rng(99), n_perm=PERM_N)


def main(results_dir=None, layers=None, cache=None, model=None,
         doc_cap=None, out=None, selfcheck=False):
    global S, D, NEPS, EPS_REL  # selfcheck temporarily changes globals
    # NOTE: no argparse here — the Kaggle runner cell calls main(**kwargs)
    # and Kaggle's sys.argv contains IPython flags that would crash
    # parse_args() (SystemExit: 2). CLI-only parsing lives in main_cli(),
    # invoked only by the __main__ block (stripped by build_kernel).
    layers = list(layers) if layers else list(LAYERS)
    results_dir = results_dir or os.path.join(os.path.dirname(os.getcwd()),
                                              'results')
    if not os.path.isdir(results_dir):
        results_dir = 'results'
    if out is None:
        out = os.path.join(results_dir, 'exp32r3_layer_profile.json')
    import types
    a = types.SimpleNamespace(out=out, layers=layers, cache=cache, model=model,
                              doc_cap=doc_cap, selfcheck=selfcheck)

    # cache/model resolution: args -> /kaggle/input (kernel sources) ->
    # /kaggle/working -> local paths
    cache_dir = a.cache
    if cache_dir is None and os.path.isdir('/kaggle/input'):
        for slug in sorted(os.listdir('/kaggle/input')):
            p = os.path.join('/kaggle/input', slug, 'results')
            if os.path.isfile(os.path.join(p, f'exp27b_jcache_L{LAYERS[0]}.pt')):
                cache_dir = p
                print(f'cache from kernel source: {p}', flush=True)
                break
    if cache_dir is None:
        for cand in ('/kaggle/working/results',
                     os.path.expanduser('~/work/prism/results'),
                     '/data/prism_kernels/exp27b/results'):
            if os.path.isfile(os.path.join(cand, f'exp27b_jcache_L{LAYERS[0]}.pt')):
                cache_dir = cand
                break
    model_dir = a.model
    if model_dir is None and os.path.isdir('/kaggle/input'):
        for slug in sorted(os.listdir('/kaggle/input')):
            p = os.path.join('/kaggle/input', slug, 'qwen35-08b')
            if os.path.isdir(p):
                model_dir = p
                print(f'model from kernel source: {p}', flush=True)
                break
    if model_dir is None:
        for cand in (os.path.join(
            '/kaggle/working', 'qwen35-08b'),
            os.path.expanduser('~/.cache/hub/qwen35-08b')):
            if os.path.isdir(cand):
                model_dir = cand
                break
        if model_dir is None:
            model_dir = MODEL  # HF download
    print(f'cache: {cache_dir}', flush=True)
    print(f'model: {model_dir}', flush=True)

    if a.selfcheck:
        rng = np.random.default_rng(0)
        # (1) fit_cubic 16-point decode
        y = 1.0 * EPS_REL + 2.0 * EPS_REL ** 2 + 0.5 * EPS_REL ** 3 + \
            1e-6 * rng.standard_normal(NEPS)
        a1f, a2f, a3f, pc, _ = fit_cubic(EPS_REL, y)
        assert abs(a1f - 1) < 0.05 and abs(a2f - 2) < 0.05 and abs(a3f - 0.5) < 0.05, \
            f'fit16 fail {a1f} {a2f} {a3f}'
        # (2) fit_all == pairwise, incl. float32 input (review point 4)
        rng3 = np.random.default_rng(2)
        Sa, Ta = 3, 4
        eabs = rng3.uniform(10, 30, (Sa, 1)) * np.asarray(EPS_REL)
        dz = rng3.standard_normal((Sa, Ta, Ta, NEPS)).astype(np.float32)
        a1v, a2v, a3v, pCv, _ = fit_all(eabs, dz)
        for s_ in range(Sa):
            for i_ in range(Ta):
                for j_ in range(Ta):
                    aa, ab, ac, pcc, _ = fit_cubic(eabs[s_], dz[s_, i_, j_])
                    assert abs(a1v[s_, i_, j_] - aa) < 1e-3 and abs(a2v[s_, i_, j_] - ab) < 1e-3
        # (3) OLS+HC3: signal alpha + p-value + permutation
        rngx = np.random.default_rng(4)
        x = rngx.uniform(0, 1, 40)
        yx = 3.0 * x + 1.0 + 0.05 * rngx.standard_normal(40)
        o = ols_hc3(x, yx, rng=rngx, n_perm=200)
        assert abs(o['alpha'] - 3.0) < 0.3 and o['alpha_p'] < 0.001, f'ols fail {o}'
        assert o['perm_p'] < 0.05
        # (4) d3_top_r NUMERIC (Fix e): identical -> 0, orthogonal -> pi/2
        J1 = np.zeros((8, 6)); J1[0, 0] = 1.0
        J2 = np.zeros((8, 6)); J2[0, 1] = 2.0
        Jseq = np.stack([J1, J1, J2])
        Pc = np.full((3, 8), 1.0 / 8.0)
        d = d3_top_r(Jseq, Pc, 1, 'raw')
        assert d.shape == (2,)
        assert abs(d[0]) < 1e-4, f'id angle {d[0]}'
        assert abs(d[1] - math.pi / 2) < 1e-6, f'orth angle {d[1]}'
        dfix = d3_top_r(Jseq, Pc, 1, 'fixP', np.full(8, 1.0 / 8.0))
        assert np.all(np.isfinite(dfix)) and abs(dfix[0]) < 1e-4, f'fixP id {dfix}'
        rngt = np.random.default_rng(11)
        Jfr = np.stack([rngt.standard_normal((8, 6)), rngt.standard_normal((8, 6))])
        Pseq1h = np.zeros((2, 8)); Pseq1h[:, 0] = 1.0
        d_implicit = d3_top_r(Jfr, Pseq1h, 1, 'fixP', Pseq1h[0])
        d_explicit = d3_top_r(Jfr, Pseq1h, 1, 'fixP', np.full(8, 1.0 / 8.0))
        assert abs(d_implicit[0] - d_explicit[0]) > 1e-6, \
            f'p0 not respected {d_implicit} vs {d_explicit}'
        # (5) quad_frac formula, [T,K,K] shapes: quadratic-free -> quad == 0
        # exactly; linear-free (a1=0) -> quad == -1 (s1=0, s2>0)
        Tz = np.array([10.0, 20.0, 30.0])
        a1z = np.zeros((3, 4, 4)); a2z = np.zeros((3, 4, 4))
        c1z = a1z + 2.0 * a2z * (EPS_REL[0] * Tz)[:, None, None]
        assert c1z.shape == (3, 4, 4), c1z.shape
        assert np.allclose(
            (np.linalg.norm(a1z, axis=-1) - np.linalg.norm(c1z, axis=-1))
            / np.maximum(np.abs(np.linalg.norm(c1z, axis=-1)), 1e-12), 0.0)
        a1q = np.zeros((3, 4, 4)); a2q = np.ones((3, 4, 4))
        c1q = a1q + 2.0 * a2q * (EPS_REL[0] * Tz)[:, None, None]
        assert np.allclose(
            (np.linalg.norm(a1q, axis=-1) - np.linalg.norm(c1q, axis=-1))
            / np.maximum(np.abs(np.linalg.norm(c1q, axis=-1)), 1e-12), -1.0)
        # (6) pick_t0 canonical determinism
        rng2 = np.random.default_rng(1)
        Pc2 = rng2.random((20, 128))
        Pc2 = Pc2 / Pc2.sum(-1, keepdims=True)
        sentc = np.zeros(20, bool)
        sentc[3] = sentc[7] = True
        assert pick_t0(Pc2, sentc) == pick_t0(Pc2.copy(), sentc.copy())
        assert pick_t0(Pc2, sentc) not in (3, 7)
        # (7) doc_layer_metrics T=K fake-forward (linear -> slope 0, resid 0)
        S0, D0, NEPS0, EPS0 = S, D, NEPS, EPS_REL
        try:
            S, D, NEPS = 4, 16, 5
            EPS_REL = np.array([0.05, 0.1, 0.2, 0.4, 0.7])
            rng7 = np.random.default_rng(77)
            Wf = rng7.standard_normal((16, S))

            class _Arr:
                def __init__(self, aa):
                    self.a = aa
                def numpy(self):
                    return self.a
            keys7 = [(0, 0, t) for t in range(2)]
            ck7 = {
                'JR': {k: _Arr(rng7.standard_normal((S, 16))) for k in keys7},
                'P128': {k: _Arr(rng7.random(S)) for k in keys7},
                'P128_RAW': {k: _Arr(rng7.random(S) * 10) for k in keys7},
                'H': {k: _Arr(rng7.standard_normal(16)) for k in keys7},
            }
            res7, _ = doc_layer_metrics(lambda hs: hs @ Wf, ck7, (0, 0), 2)
            assert len(res7['resid_frac_by_eps_med']) == NEPS
            assert max(res7['resid_frac_by_eps_med']) < 1e-3, \
                f'T=K resid {res7["resid_frac_by_eps_med"]}'
            assert res7['a_vs_ESG_cos'] is not None
            assert 'd3_by_r' in res7 and math.isfinite(res7['s_l_median'])
        finally:
            S, D, NEPS, EPS_REL = S0, D0, NEPS0, EPS0
        print('selfcheck OK', flush=True)
        return

    if cache_dir is None:
        print('[FAIL] no E-27b dense cache found; pass --cache')
        return

    t0 = time.time()

    # --- cache load + L19 sanity assert (review point 1) ---
    ck = {}
    for l in a.layers:
        p = os.path.join(cache_dir, f'exp27b_jcache_L{l}.pt')
        ck[l] = torch.load(p, map_location='cpu', weights_only=False)
        docs_l = {(k[0], k[1]) for k in ck[l]['JR'].keys()}
        ts_l = {k[2] for k in ck[l]['JR'].keys()}
        assert len(docs_l) == 32, f'L{l}: expected 32 docs, got {len(docs_l)}'
        assert ts_l == set(range(T_FULL)), \
            f'L{l}: expected t=0..{T_FULL - 1}, got {len(ts_l)} distinct'
    U128 = np.array(ck[a.layers[0]]['U128'], dtype=np.int64)
    for l in a.layers[1:]:
        assert np.array_equal(np.array(ck[l]['U128'], dtype=np.int64), U128), \
            f'L{l}: U128 mismatch'
    # P128_RAW: identical across layers (decoder-tail p128, layer-independent)
    # — the L19 file predates the RAW addition; fill from the first layer
    # that has it.
    raw_src = None
    for l in a.layers:
        if ck[l].get('P128_RAW'):
            raw_src = l
            break
    if raw_src is None:
        print('[FAIL] P128_RAW missing in every layer cache')
        return
    for l in a.layers:
        if not ck[l].get('P128_RAW'):
            ck[l]['P128_RAW'] = ck[raw_src]['P128_RAW']
            print(f'L{l}: P128_RAW filled from L{raw_src} ' \
                  f'(identical across layers)', flush=True)
        assert set(ck[l]['P128_RAW'].keys()) == set(ck[l]['JR'].keys()), \
            f'L{l}: P128_RAW keys mismatch'
    docs = sorted({(k[0], k[1]) for k in ck[a.layers[0]]['JR'].keys()})
    if a.doc_cap:
        docs = docs[:a.doc_cap]
    print(f'docs: {len(docs)}, layers: {a.layers}, eps: {NEPS} pts, '
          f'T={T_FULL} all layers, device={DEVICE}', flush=True)

    forward, W_U, g, eps_norm = load_decoder(U128, model_dir)

    # E-35 identity (1 position, 1 doc, L23 — desync guard); skipped when
    # L23 not in --layers (it validates L23 analytics, not the pipeline)
    if 23 in ck:
        d23 = docs[0]
        h0 = np.stack([ck[23]['H'][(d23[0], d23[1], 0)].numpy()]).reshape(D)
        J0 = ck[23]['JR'][(d23[0], d23[1], 0)].numpy()
        J23a = analytic_J(W_U, g, h0)
        cos = float((J23a * J0).sum() / (np.linalg.norm(J23a) * np.linalg.norm(J0)))
        print(f'E-35 identity cos = {cos:.6f}', flush=True)
        assert abs(abs(cos) - 1) < 1e-4, 'E-35 identity fail'
    else:
        print('E-35 identity skipped: L23 not in --layers', flush=True)

    per_doc_layer = {}
    cache = {}
    # --- checkpoint (self-sourced kernel: prev /kaggle/working is mounted
    # read-only at /kaggle/input/<slug>/ — resume from there if present) ---
    ckpt_path = os.path.join(results_dir, 'exp32r3_checkpoint.pt')
    for cand in ([ckpt_path] + [
            os.path.join('/kaggle/input', s, 'results', 'exp32r3_checkpoint.pt')
            for s in sorted(os.listdir('/kaggle/input'))] if os.path.isdir('/kaggle/input') else []):
        if os.path.isfile(cand):
            c = torch.load(cand, map_location='cpu', weights_only=False)
            per_doc_layer = c.get('per_doc_layer', {})
            cache = c.get('cache', {})
            print(f'checkpoint {cand}: {len(per_doc_layer)} (doc,layer) pairs resumed',
                  flush=True)
            break

    def save_ckpt():
        torch.save({'per_doc_layer': per_doc_layer, 'cache': cache}, ckpt_path)

    first = True
    for li, l in enumerate(a.layers):
        for d in docs:
            key = f'{l}|{d[0]}/{d[1]}'
            if key in per_doc_layer:
                continue
            t1 = time.time()
            per_doc_layer[key], cache[key] = \
                doc_layer_metrics(forward, ck[l], d, T_FULL)
            save_ckpt()  # per-pair checkpoint: T4 wipe at any point is safe
            if first:
                try:
                    import psutil
                    rss = psutil.Process().memory_info().rss / 2 ** 30
                    print(f'  first (doc, layer) {time.time() - t1:.1f}s '
                          f'RSS={rss:.2f} GB', flush=True)
                except Exception:
                    print(f'  first (doc, layer) {time.time() - t1:.1f}s',
                          flush=True)
                first = False
            print(f'  L{l} T={T_FULL} {d}: {time.time() - t1:.1f}s', flush=True)
    del ck

    # --- aggregation E-34: per-doc median -> median/IQR over docs ---
    def _agg(vals):
        vals = np.array([v for v in vals if v is not None])
        if len(vals) == 0:
            return None
        return float(np.median(vals)), [float(np.percentile(vals, 25)),
                                        float(np.percentile(vals, 75))]

    layers_out = {}
    boot_rng = np.random.default_rng(12345)
    for l in a.layers:
        keys = [f'{l}|{d[0]}/{d[1]}' for d in docs]
        missing = [k for k in keys if k not in per_doc_layer]
        if missing:
            sys.exit(f'L{l}: {len(missing)} (doc,layer) pairs missing from '
                     f'checkpoint (e.g. --doc-cap smaller than the run). '
                     f'Re-run with the original --doc-cap or delete the checkpoint.')
        V = [per_doc_layer[k] for k in keys]
        sl = _agg([v['s_l_median'] for v in V])
        sr = _agg([v['slope_ratio_median'] for v in V])
        qf = _agg([v['quad_frac_median'] for v in V])
        rfe = np.array([[v['resid_frac_by_eps_med'][e] for e in range(NEPS)]
                        for v in V])
        d3fix = {r: np.array([v['d3_by_r']['fixP'][str(r)][0] for v in V])
                 for r in R_SWEEP}
        d3raw = {r: np.array([v['d3_by_r']['raw'][str(r)][0] for v in V])
                 for r in R_SWEEP}
        d3gf = {r: np.array([v['d3_by_r']['gf'][str(r)][0] for v in V])
                for r in R_SWEEP}
        db = np.array([v['delta_b_rel'] for v in V
                       if v['delta_b_rel'] is not None])
        alphas = np.array([v['ols']['alpha'] for v in V
                           if v['ols'] and v['ols']['alpha'] is not None])

        def _boot(arr, stat='median'):
            idx = boot_rng.integers(0, len(arr), (BOOT_N, len(arr)))
            s = np.median(arr[idx], axis=1) if stat == 'median' \
                else arr[idx].mean(axis=1)
            return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
        sl_arr = np.array([v['s_l_median'] for v in V])
        d3b = {'fixP': {r: _boot(d3fix[r]) for r in R_SWEEP},
               'raw': {r: _boot(d3raw[r]) for r in R_SWEEP},
               'gf': {r: _boot(d3gf[r]) for r in R_SWEEP}}

        d3_by_r_out = {}
        for m in ('raw', 'gf', 'fixP'):
            arrm = d3fix if m == 'fixP' else (d3raw if m == 'raw' else d3gf)
            d3_by_r_out[m] = {}
            for r in R_SWEEP:
                arr = arrm[r]
                d3_by_r_out[m][str(r)] = {
                    'median': float(np.median(arr)),
                    'iqr': [float(np.percentile(arr, 25)),
                            float(np.percentile(arr, 75))],
                    'boot_ci': d3b[m][r]}

        layers_out[str(l)] = {
            's_l': {'median': sl[0], 'iqr': sl[1], 'boot_ci': _boot(sl_arr)},
            'a_vs_ESG_cos': _agg([v['a_vs_ESG_cos'] for v in V]),
            'slope_ratio': {'median': sr[0], 'iqr': sr[1]},
            'quad_frac': {'median': qf[0], 'iqr': qf[1]},
            'A1_median': _agg([v['A1_median'] for v in V]),
            'nb_median': _agg([v['nb_median'] for v in V]),
            'frac_c_sig': _agg([v['frac_c_sig'] for v in V]),
            'resid_frac_by_eps': {f'{int(round(EPS_REL[e] * 100))}pct':
                                  {'median': float(np.median(rfe[:, e])),
                                   'max_over_docs': float(np.max(rfe[:, e]))}
                                  for e in range(NEPS)},
            'd3_by_r': d3_by_r_out,
            'stratification': {
                'bQ1_fro': _agg([v['bQ1_fro'] for v in V]),
                'bQ4_fro': _agg([v['bQ4_fro'] for v in V]),
                'delta_b_rel': ({'median': float(np.median(db)),
                                 'iqr': [float(np.percentile(db, 25)),
                                         float(np.percentile(db, 75))],
                                 'boot_ci': _boot(db)} if len(db) else None),
                'ols': {'alpha': [float(np.median(alphas)),
                                  [float(np.percentile(alphas, 25)),
                                   float(np.percentile(alphas, 75))]]
                        if len(alphas) else None,
                        'pooled': _pooled_ols(cache, keys)},
                'n_docs_with_strata': int(len(db)),
            },
            'cross_checks': {k: _agg([v['cross_checks'][k] for v in V])
                             for k in ('rho_nb_dSVD_raw1', 'rho_nb_dSVD_fixP1',
                                       'rho_nb_dSVD_gf1', 'rho_nb_dp')},
            'sentinel_frac': _agg([v['sentinel_fraction'] for v in V]),
        }
        print(f'L{l}: s_l={sl[0]:.4f} slope={sr[0]:.4f} quad={qf[0]:.4f} '
              f'd3r1 raw/gf/fixP = '
              f'{np.median(d3raw[1]):.3f}/{np.median(d3gf[1]):.3f}/'
              f'{np.median(d3fix[1]):.3f} '
              f'd_b={np.median(db) if len(db) else None}', flush=True)

    report = {
        'meta': {
            'experiment': ('E-32r.3 (layer profile; supersedes E-32r.2 on '
                           'shared layers: normalized p0, L19 added, '
                           'T=128 all layers, GPU, float32 dz / float64 fit, '
                           'new quad_frac metric)'),
            'model': model_dir,
            'cache': cache_dir,
            'layers': list(a.layers),
            'n_docs': len(docs),
            'T': {str(l): T_FULL for l in a.layers},
            'eps_rel': [float(e) for e in EPS_REL],
            'eps_canon': EPS_CANON,
            'fit': 'dz = a*e + b*e^2 + c*e^3 (no const), eps absolute in h',
            's_l': 'eps_canon * ||off(b_t)||_F / ||off(ESG_raw_t)||_F (RAW norm)',
            'slope_ratio': ('E-32r.2 formula UNCHANGED: median of '
                            '|dz(e0)|/e0 / (|dz(e1)|/e1); 1.0 = linear regime'),
            'quad_frac': ('NEW (no E-32r.2 counterpart): (s1-s2)/|s2|, '
                          's1=||a1||, s2=||a1+2*(eps0*||h_t||)*a2||; '
                          '0 = quadratic term negligible at eps0'),
            'a_vs_ESG_cos': ('FIT-QUALITY sanity check (chain rule a1 = J e, '
                             'exact for any J): ~1 expected on every layer '
                             'with small resid_frac; NOT a depth metric'),
            'aggregation': 'E-34: per-doc median -> median/IQR over docs',
            'sentinel': 'P128_RAW.sum < 0.5, endpoints-only valid pairs; '
                        'sentinel_frac = share raw rows, nb_median over ~sent',
            'p0': 'canonical t0 position, NORMALIZED to sum 1',
            'd3': 'top-r right singular direction angle (cos-sim, exp33r)',
            'd3_fixP_note': ('diagnostics only on mid layers (TЗ §9: '
                             'J = M_K*P_u exact only at L23)'),
            'bootstrap': f'{BOOT_N} iterations over docs (seed=12345)',
            'device': DEVICE,
        },
        'layers': layers_out,
        'verdicts': {},
        'per_doc': {k: v for k, v in per_doc_layer.items()},
    }
    # --- verdicts ---
    sl_med = [layers_out[str(l)]['s_l']['median'] for l in a.layers]
    lin = {}
    for l in a.layers:
        vals = [layers_out[str(l)]['resid_frac_by_eps'][
            f'{int(round(EPS_REL[e] * 100))}pct']['median'] for e in range(NEPS)]
        lin[str(l)] = bool(np.median(vals).max() < 0.01)
    s_ok = all(v is not None and v < 0.3 for v in sl_med)
    report['verdicts'] = {
        's_l_lt_0.3_all_layers': bool(s_ok),
        's_l_medians': {str(l): sl_med[i] for i, l in enumerate(a.layers)},
        'resid_frac_lt_1pct_per_layer': lin,
        'linear_verdict': bool(s_ok and all(lin.values())),
        'depth_metrics': 's_l(l), d3_raw(1,l), d3_gf(1,l) — profile over '
                         'L6..L23; d3_fixP(1,l) diagnostics on mid layers',
        'note': ('A/B/C resolved via d3 sweep + stratification + cross-checks '
                 '(see layers.*)'),
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(_clean(report), f, indent=1, ensure_ascii=False,
                  allow_nan=False)
    print(f'WROTE {a.out}  total {time.time() - t0:.0f}s', flush=True)


def main_cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--layers', type=int, nargs='+', default=list(LAYERS))
    ap.add_argument('--cache', default=None, help='E-27b dense cache dir')
    ap.add_argument('--model', default=None, help='decoder-tail model dir')
    ap.add_argument('--doc-cap', type=int, default=None)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    main(out=a.out, layers=a.layers, cache=a.cache, model=a.model,
         doc_cap=a.doc_cap, selfcheck=a.selfcheck)


if __name__ == '__main__':
    try:
        main_cli()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
