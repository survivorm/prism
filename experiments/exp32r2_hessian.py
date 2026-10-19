#!/usr/bin/env python
"""E-32r.2 (docs/Exp32.md + docs/Exp32r_errors.md): полный Гессиан readout,
5 слоёв {6, 12, 18, 22, 23}, 32 дока, T=128, 16 eps, bootstrap CI.

Пилот: results/exp32r_pilot.json (L23, 1 док, 5 eps) — s_l=0.075,
residual_frac < 1%, slope_ratio=1.04, fixed-P r=1: raw 0.144 / gf 1.49 /
fixP 0.034. Гейт пройден; этот прогон — воспроизводимость s_l << 1 на
32 доках x 5 слоях + разрешение A/B/C.

Математика (ТЗ): инъекция h'[t] = H[t] + eps*Jn[t,i] (norm-ованная строка
JR, ||delta_i|| = eps); полный forward через decoder-tail (prism/jacobian.py
_run_decoder_pass, prefix_cache ЧИСТЫЙ — инвариант ТЗ §3). Fit per-pair
dz = a*e + b*e^2 + c*e^3 (без const; eps в АБСОЛЮТНЫХ единицах h,
16 точек на log-шкале [5%, 70%]). s_t = eps_canon*||off(b_t)||_F /
||off(ESG_raw_t)||_F (нормировка на RAW, не gf — ТЗ «фикс»). A1 =
a + 2*b*eps_canon, slope_ratio = median(A1/ESG_norm_row).

Стратификация (ревью #2): OLS+HC3 nb(t) ~ alpha*|dp(t)| + beta per-doc
+ pooled; Q1/Q4 (n>=16/группу, перм. p 1000); d_b_rel = ||mQ4 - mQ1|| /
||median(a2)||.

Fixed-P (ревью #1 + Exp32_errors.md): p0 = P128[t0] (canonical = median-p позиция,
нормализованная, sum=1) ПЕРЕДАЁТСЯ в d3_top_r явно (не Pseq[0]); d3 = top-r right
singular direction angle (тот же estimator что в exp33r/exp33r2: max principal
angle = acos(max svdvals), БЕЗ двойного acos); sweep r in {1,2,4,8,16};
d3_raw / d3_gf / d3_fixP per r, per layer.
T: 128 на L23, 64 на остальных (ТЗ §полный прогон; слайс [:64]).
slope_ratio = median(A1/ESG_norm_row) с корректным broadcasting [S,K,1].
a_vs_ESG_cos: Frobenius-косинус median_t(a1_t) vs column-normalized median_t(ESG_t)
(валидация a ≈ ESG, как в пилоте).

Кросс-чеки (ТЗ §«Что собираем», 4 correlations): rho(nb, dSVD_raw),
rho(nb, dSVD_fixP), rho(nb, dSVD_gf), rho(nb, |dp|) — Spearman per-doc
+ pooled, valid pairs (non-sentinel endpoints).

Bootstrap CI (ревью #3, 64 итерации, по документам): s_l, delta_b_rel,
alpha (pooled), d3 (per r).

Selfcheck: (1) fit_cubic 16 точек декодирует; (2) fit_all == попарный;
(3) OLS+HC3 на синтетике (сигнальный alpha + p-value); (4) d3_top_r
ЧИСЛЕННО: идентичные subspaces → 0, ортогональные → π/2 (ловит двойной acos);
(5) bootstrap детерминированность; (6) pick_t0/p0 canonical-детерминированность.

Fixes per docs/Exp32_errors.md: (a) убран внешний acos в d3_top_r;
(b) fixP получает канонический p0 явно; (c) slope_ratio = median(|dz(e0)|/e0 ÷ |dz(e1)|/e1) (Δz/ε linearity,
как в пилоте; прежний A1/esgn[None,:] — и broadcasting-мусор, и не тот
метрик);
(d) добавлен a_vs_ESG_cos; (e) selfcheck d3_top_r численный; (f) T=64 для не-L23; (g) doc_layer_metrics
на синтетике T=2, K=4 (Fix v2: stride по K, не по T).
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import math
import sys
import time
import traceback

import numpy as np
from scipy.stats import t as tdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp28_leaf_structure as E28  # noqa: E402

S = 128
D = 1024
LAYERS = (6, 12, 18, 22, 23)
EPS_REL = np.round(np.logspace(np.log10(0.05), np.log10(0.70), 16), 6)
NEPS = len(EPS_REL)
EPS_CANON = 0.15
R_SWEEP = (1, 2, 4, 8, 16)
T0_CANON = 64
BOOT_N = 64
PERM_N = 1000
CKPT = '/data/prism_kernels/exp27b/results'
MODEL = 'Qwen/Qwen3.5-0.8B'
OUT = 'results/exp32r2_hessian.json'
CHUNK = 512  # OOM-guard: batch [rows, 128] logits


# --------------------------------------------------------------------------
# math helpers
# --------------------------------------------------------------------------
def offdiag_norm(M):
    return float(math.sqrt(max((M * M).sum() - (np.diag(M) ** 2).sum(), 0.0)))


def gauge_free_rows(J, p):
    """P(p) = diag(p) - pp^T в token-пространстве: J [K,D] rows=tokens."""
    return J * p[:, None] - (p @ J)[None, :] * p[:, None]


def top_r_right(J, r):
    """Топ-r right singular vectors [r, D] (строки = направления в D-пространстве)."""
    U, _, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[:r, :]


def _right_angle(V1, V2):
    """Largest principal angle (rad): max(acos(svdvals(V1 @ V2.T))) — тот же
    estimator что в exp33r delta_svd."""
    sv = np.linalg.svdvals(V1 @ V2.T)
    return float(math.acos(float(np.clip(sv.max(), 0.0, 0.999999999999))))


def d3_top_r(Jseq, Pseq, r, mode, p0=None):
    """d3 = per-pair top-r largest principal angle (rad) для режима mode:
    raw / gf / fixP (p0 — канонический, ПЕРЕДАЁТСЯ явно). Возвращает [T-1]."""
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
        out.append(_right_angle(V1, V2))  # один acos (в _right_angle)
    return np.array(out)


def ols_hc3(x, y, rng=None, n_perm=PERM_N):
    """OLS y ~ alpha*x + beta, HC3 standard errors; permutation p for alpha.

    Возврат dict: alpha, beta, alpha_se, alpha_p, perm_p.
    """
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
        xp = x - x.mean()
        for _ in range(n_perm):
            yp = y[rng.permutation(n)]
            Xp = X
            bp = np.linalg.solve(X.T @ X, X.T @ yp)
            rp = yp - Xp @ bp
            Hm = X @ XtX_inv @ X.T
            hv = np.diag(Hm)
            h3 = np.sqrt(np.maximum(np.diag(XtX_inv) * float(
                (rp ** 2 / np.maximum(1 - hv, 1e-12) ** 2).sum()), 0.0))
            if h3[1] > 0 and abs(bp[1] / h3[1]) >= obs:
                cnt += 1
        perm_p = (cnt + 1) / (n_perm + 1)
    return {'alpha': float(beta_hat[1]), 'beta': float(beta_hat[0]),
            'alpha_se': float(hc3[1]), 'alpha_p': alpha_p, 'perm_p': perm_p}


def fit_cubic(eps, y):
    """y = a*e + b*e^2 + c*e^3 (без const); возврат a1,a2,a3,pC,resid."""
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
    """Векторизованный cubic fit. eps_abs [T,NEPS], dz [T,K,K,NEPS].

    Возврат: a1,a2,a3,pC [T,K,K], resid [T,K,K,NEPS].
    """
    e = eps_abs.astype(np.float64)
    Z = dz if dz.dtype == np.float64 else dz.astype(np.float64)
    nrm = e[:, 0] / EPS_REL[0]                                   # [S]
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
    """Canonical p0-позиция: non-sentinel позиция с p, ближайшей к медиане."""
    ok = ~sent
    if not ok.any():
        ok = np.ones_like(sent)
    med = np.median(P128[ok], 0)
    sel = int(np.argmin(np.linalg.norm(P128[ok] - med[None, :], axis=1)))
    return int(np.where(ok)[0][sel])


# --------------------------------------------------------------------------
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
# model
# --------------------------------------------------------------------------
def load_decoder(U128):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1)
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
    r = np.linalg.norm(h)
    u = h / r
    M = W_U * g[None, :]
    return math.sqrt(D) / r * (M - (M @ u)[:, None] * u[None, :])


# --------------------------------------------------------------------------
def doc_layer_metrics(forward, ck, dom_si, U128, T):
    """Один документ x один слой: forward всех инъекций, fit, все метрики ТЗ.
    T = 128 (L23) / 64 (остальные, слайс [:64])."""
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

    # --- инъекции: h'[t,i,e] = H[t] + eps_abs[e,t] * Jn[t,i] ---
    Jn = JR / np.maximum(np.linalg.norm(JR, axis=-1)[..., None], 1e-12)
    H32 = H.astype(np.float32)
    Jn32 = Jn.astype(np.float32)
    scaled = eps_abs.T.astype(np.float32)[:, None, :, None] * Jn32[:, :, None, :]
    h_inj = (H32[:, None, None, :] + scaled).reshape(-1, D)
    del scaled, H32, Jn32, Jn
    base_all = forward(H)
    K = S  # vocab-slice (logits); T — число позиций (Fix по Exp32_errors.md v2:
    # stride и reshape — по K, не по T; для T≠K слайсирование по T даёт мусор)
    dz32 = np.empty((T, K, K, NEPS), dtype=np.float32)
    for t_ in range(T):
        hc = h_inj[t_ * K * NEPS:(t_ + 1) * K * NEPS]
        out = []
        for c0 in range(0, hc.shape[0], CHUNK):
            out.append(forward(hc[c0:c0 + CHUNK]))
        L = np.concatenate(out).reshape(K, NEPS, S).transpose(0, 2, 1)
        dz32[t_] = L - base_all[t_][None, :, None]
    del h_inj, base_all
    dz_all = np.ascontiguousarray(dz32).astype(np.float64)
    del dz32

    a1, a2, a3, pC, resid = fit_all(eps_abs.T, dz_all)
    A1 = a1 + 2.0 * a2 * (EPS_CANON * norms)[:, None, None]

    # --- s_t / slope / residual (ТЗ «Математика», raw-нормировка) ---
    ESG_all = np.einsum('std,sjd->stj', JR, JR)
    ESGd = np.diagonal(ESG_all, axis1=1, axis2=2)
    off_esg = np.sqrt(np.maximum((ESG_all ** 2).sum((-1, -2)) - (ESGd ** 2).sum(-1), 0.0))
    nb = np.array([offdiag_norm(a2[t]) for t in range(T)])
    s_t = EPS_CANON * norms * nb / np.maximum(off_esg, 1e-12)
    # slope_ratio (ТЗ, как в пилоте): Δz/ε linearity = median(|dz(e0)|/e0 /
    # (|dz(e1)|/e1)) per pair — 1.0 = linear regime (Fix c: прежний
    # A1/esgn[None,:] был и broadcasting-мусором, и не тем метриком ТЗ)
    e0v = EPS_REL[0] * norms
    e1v = EPS_REL[1] * norms
    slope0 = np.abs(dz_all[..., 0]) / np.maximum(e0v[:, None, None], 1e-12)
    slope1 = np.abs(dz_all[..., 1]) / np.maximum(e1v[:, None, None], 1e-12)
    slope = slope0 / np.maximum(slope1, 1e-12)
    slope_ratio_med = float(np.median(slope))
    c_sig = float((pC[~sent] < 0.05).mean())

    # Fix (d): a ≈ ESG (Frobenius-косинус, как в пилоте)
    amed = np.median(a1, 0)
    ESG_med = np.median(ESG_all, 0)
    ESGn = ESG_med / np.maximum(
        np.sqrt(np.maximum(np.diag(ESG_med), 0.0))[None, :], 1e-12)
    a_vs_esg = float((amed * ESGn).sum() / max(
        np.linalg.norm(amed) * np.linalg.norm(ESGn), 1e-12))

    resid_frac_t = np.zeros((T, NEPS))  # Fix v3: (T, NEPS), не (S, NEPS)
    for e in range(NEPS):
        num = np.linalg.norm(resid[:, :, :, e], axis=(1, 2))
        den = np.linalg.norm(dz_all[:, :, :, e], axis=(1, 2))
        resid_frac_t[:, e] = num / np.maximum(den, 1e-12)
    rfe_all = resid_frac_t

    # --- p0 canonical + d3 (top-r, sweep r); fixP получает p0 ЯВНО (Fix b) ---
    t0 = pick_t0(P128, sent)
    p0 = P128[t0] / P128[t0].sum()
    d3 = {m: {r: d3_top_r(JR, P128, r, m, p0) for r in R_SWEEP}
          for m in ('raw', 'gf', 'fixP')}

    # --- Q1/Q4 по |dp| среди valid pairs (ревью #2: n>=16/группа) ---
    dp = np.linalg.norm(np.diff(P128, axis=0), axis=1)
    dpv = dp[vp]
    q1 = q4 = None
    mQ1 = mQ4 = None
    bQ1 = bQ4 = None
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

    # --- OLS+HC3: nb(t) ~ alpha*|dp(t)| + beta, valid pairs (ревью #2) ---
    tidx = np.where(vp)[0]
    ols = ols_hc3(dp[tidx], nb[tidx]) if len(tidx) >= 5 else None

    # --- 4 кросс-чека (ТЗ): rho(nb, dSVD_raw/fixP/gf/r=1, |dp|) ---
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
        'slope_ratio_iqr': [float(np.percentile(slope, 25)), float(np.percentile(slope, 75))],
        'A1_median': float(np.median(A1)),
        'nb_median': float(np.median(nb[~sent])),
        'frac_c_sig': c_sig,
        'resid_frac_by_eps_med': [float(np.median(resid_frac_t[:, e])) for e in range(NEPS)],
        'resid_frac_by_eps_max': [float(np.max(resid_frac_t[:, e])) for e in range(NEPS)],
        'sentinel_fraction': float(sent.mean()),
        'n_valid_pairs': int(vp.sum()),
        'a_vs_ESG_cos': a_vs_esg,
        'd3_by_r': {m: {str(r): [float(np.median(d3[m][r][vp])),
                                 [float(np.percentile(d3[m][r][vp], 25)),
                                  float(np.percentile(d3[m][r][vp], 75))]]
                        for r in R_SWEEP} for m in ('raw', 'gf', 'fixP')},
        'Q1_n': int(len(np.where((dp <= (q1 if q1 is not None else 0)) & vp)[0])) if q1 is not None else 0,
        'Q4_n': int(len(np.where((dp >= (q4 if q4 is not None else 0)) & vp)[0])) if q4 is not None else 0,
        'bQ1_fro': bQ1,
        'bQ4_fro': bQ4,
        'delta_b_rel': db_rel,
        'ols': ols,
        'cross_checks': rho,
    }
    return _clean(out), {'dp_v': dp[vp], 'nb_v': nb[:-1][vp]}


def main():
    global S, D, NEPS, EPS_REL  # (7) selfcheck меняет globals временно
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--layers', type=int, nargs='+', default=list(LAYERS))
    ap.add_argument('--doc-cap', type=int, default=None)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()

    E28._pin_p_cores()

    if a.selfcheck:
        rng = np.random.default_rng(0)
        # (1) fit_cubic 16 точек декодирует
        y = 1.0 * EPS_REL + 2.0 * EPS_REL ** 2 + 0.5 * EPS_REL ** 3 + \
            1e-6 * rng.standard_normal(NEPS)
        a1f, a2f, a3f, pc, _ = fit_cubic(EPS_REL, y)
        assert abs(a1f - 1) < 0.05 and abs(a2f - 2) < 0.05 and abs(a3f - 0.5) < 0.05, \
            f'fit16 fail {a1f} {a2f} {a3f}'
        # (2) fit_all == попарный
        rng3 = np.random.default_rng(2)
        Sa, Ta = 3, 4
        eabs = rng3.uniform(10, 30, (Sa, 1)) * np.asarray(EPS_REL)
        dz = rng3.standard_normal((Sa, Ta, Ta, NEPS))
        a1v, a2v, a3v, pCv, _ = fit_all(eabs, dz)
        for s_ in range(Sa):
            for i_ in range(Ta):
                for j_ in range(Ta):
                    aa, ab, ac, pcc, _ = fit_cubic(eabs[s_], dz[s_, i_, j_])
                    assert abs(a1v[s_, i_, j_] - aa) < 1e-5 and abs(a2v[s_, i_, j_] - ab) < 1e-5
        # (3) OLS+HC3: сигнальный alpha + p-value + permutation
        rngx = np.random.default_rng(4)
        x = rngx.uniform(0, 1, 40)
        yx = 3.0 * x + 1.0 + 0.05 * rngx.standard_normal(40)
        o = ols_hc3(x, yx, rng=rngx, n_perm=200)
        assert abs(o['alpha'] - 3.0) < 0.3 and o['alpha_p'] < 0.001, \
            f'ols fail {o}'
        assert o['perm_p'] < 0.05
        # (4) d3_top_r ЧИСЛЕННО (Fix e): идентичные → 0, ортогональные → π/2
        J1 = np.zeros((8, 6)); J1[0, 0] = 1.0   # right sv e1
        J2 = np.zeros((8, 6)); J2[0, 1] = 2.0   # right sv e2 (⊥ e1)
        Jseq = np.stack([J1, J1, J2])
        Pc = np.full((3, 8), 1.0 / 8.0)
        d = d3_top_r(Jseq, Pc, 1, 'raw')
        assert d.shape == (2,)
        assert abs(d[0]) < 1e-4, f'id angle {d[0]}'
        assert abs(d[1] - math.pi / 2) < 1e-6, f'orth angle {d[1]}'
        # fixP с явным p0: одинаковый p0 в обоих членах пары → идентичные
        # P(p0)·J → угол 0 (gauge сам по себе НЕ сохраняет ортогональность)
        dfix = d3_top_r(Jseq, Pc, 1, 'fixP', np.full(8, 1.0 / 8.0))
        assert np.all(np.isfinite(dfix)) and abs(dfix[0]) < 1e-4, f'fixP id {dfix}'
        # и НЕ использует Pseq[0] как раньше: one-hot Pseq[0] (gauge = обнуление
        # строки 0) vs uniform явный p0 (строчные смешения) → разные углы
        rngt = np.random.default_rng(11)
        Jfr = np.stack([rngt.standard_normal((8, 6)), rngt.standard_normal((8, 6))])
        Pseq1h = np.zeros((2, 8)); Pseq1h[:, 0] = 1.0
        d_implicit = d3_top_r(Jfr, Pseq1h, 1, 'fixP', Pseq1h[0])  # = old bug
        d_explicit = d3_top_r(Jfr, Pseq1h, 1, 'fixP', np.full(8, 1.0 / 8.0))
        assert abs(d_implicit[0] - d_explicit[0]) > 1e-6, \
            f'p0 not respected {d_implicit} vs {d_explicit}'
        # (5) bootstrap детерминированность
        rb = np.random.default_rng(7)
        vals = np.abs(rb.standard_normal(32))
        s1 = sorted(vals[np.random.default_rng(7).integers(0, 32, (64, 32), dtype=int)].mean(-1))
        s2 = sorted(vals[np.random.default_rng(7).integers(0, 32, (64, 32), dtype=int)].mean(-1))
        assert np.allclose(s1, s2)
        # (6) pick_t0 canonical-детерминированность
        rng2 = np.random.default_rng(1)
        Pc2 = rng2.random((20, 128))
        Pc2 = Pc2 / Pc2.sum(-1, keepdims=True)
        sentc = np.zeros(20, bool)
        sentc[3] = sentc[7] = True
        assert pick_t0(Pc2, sentc) == pick_t0(Pc2.copy(), sentc.copy())
        assert pick_t0(Pc2, sentc) not in (3, 7)
        # (7) doc_layer_metrics T≠K (Fix v2/v3): stride/reshape по K, не по T;
        # resid_frac_t (T, NEPS). Линейный fake-forward → Δz ровно ∝ ε:
        # slope≈1, resid≈0. (globals объявлены в main)
        S0, D0, NEPS0, EPS0 = S, D, NEPS, EPS_REL
        try:
            S, D, NEPS = 4, 16, 5   # K=4, D=16, 5 точек
            EPS_REL = np.array([0.05, 0.1, 0.2, 0.4, 0.7])
            rng7 = np.random.default_rng(77)
            Wf = rng7.standard_normal((D, S))

            class _Arr:  # shim: .numpy(), как в реальном кэше
                def __init__(self, a):
                    self.a = a
                def numpy(self):
                    return self.a
            keys7 = [(0, 0, t) for t in range(2)]
            ck7 = {
                'JR': {k: _Arr(rng7.standard_normal((S, D))) for k in keys7},
                'P128': {k: _Arr(rng7.random(S)) for k in keys7},
                'P128_RAW': {k: _Arr(rng7.random(S) * 10) for k in keys7},
                'H': {k: _Arr(rng7.standard_normal(D)) for k in keys7},
            }
            res7, _ = doc_layer_metrics(lambda hs: hs @ Wf, ck7, (0, 0),
                                        np.arange(S), 2)
            assert len(res7['resid_frac_by_eps_med']) == NEPS
            assert abs(res7['slope_ratio_median'] - 1.0) < 1e-4, \
                f'T≠K slope {res7["slope_ratio_median"]}'
            assert max(res7['resid_frac_by_eps_med']) < 1e-4, \
                f'T≠K resid {res7["resid_frac_by_eps_med"]}'
            assert res7['a_vs_ESG_cos'] is not None
            assert 'd3_by_r' in res7 and math.isfinite(res7['s_l_median'])
        finally:
            S, D, NEPS, EPS_REL = S0, D0, NEPS0, EPS0
        print('selfcheck OK', flush=True)
        return

    ck = {l: E28.load_layer(CKPT, l) for l in a.layers}
    U128 = [np.array(ck[l]['U128'], dtype=np.int64) for l in a.layers]
    docs = sorted({(k[0], k[1]) for k in ck[a.layers[0]]['JR'].keys()})
    if a.doc_cap:
        docs = docs[:a.doc_cap]
    print(f'docs: {len(docs)}, layers: {a.layers}, eps: {NEPS} pts', flush=True)

    t0 = time.time()
    forward, W_U, g, eps_norm = load_decoder(U128[0])
    # E-35 identity (1 позиция, 1 док, L23 — защита от рассинхрона)
    d23 = docs[0]
    h0 = np.stack([ck[23]['H'][(d23[0], d23[1], 0)].numpy()]).reshape(D)
    J0 = ck[23]['JR'][(d23[0], d23[1], 0)].numpy()
    J23a = analytic_J(W_U, g, h0)
    cos = float((J23a * J0).sum() / (np.linalg.norm(J23a) * np.linalg.norm(J0)))
    print(f'E-35 identity cos = {cos:.6f}', flush=True)
    assert abs(abs(cos) - 1) < 1e-4, 'E-35 identity fail'

    per_doc_layer = {}
    cache = {}
    for li, l in enumerate(a.layers):
        Tl = 128 if l == 23 else 64  # ТЗ: T=128 L23, T=64 остальные
        for d in docs:
            t1 = time.time()
            per_doc_layer[f'{l}|{d[0]}/{d[1]}'], cache[f'{l}|{d[0]}/{d[1]}'] = \
                doc_layer_metrics(forward, ck[l], d, U128[li], Tl)
            print(f'  L{l} T={Tl} {d}: {time.time() - t1:.1f}s', flush=True)
    del ck

    # --- агрегация E-34: per-doc median -> median/IQR по документам ---
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
        V = [per_doc_layer[k] for k in keys]
        sl = _agg([v['s_l_median'] for v in V])
        sr = _agg([v['slope_ratio_median'] for v in V])
        rfe = np.array([[v['resid_frac_by_eps_med'][e] for e in range(NEPS)] for v in V])
        d3fix = {r: np.array([v['d3_by_r']['fixP'][str(r)][0] for v in V]) for r in R_SWEEP}
        d3raw = {r: np.array([v['d3_by_r']['raw'][str(r)][0] for v in V]) for r in R_SWEEP}
        d3gf = {r: np.array([v['d3_by_r']['gf'][str(r)][0] for v in V]) for r in R_SWEEP}
        db = np.array([v['delta_b_rel'] for v in V if v['delta_b_rel'] is not None])
        alphas = np.array([v['ols']['alpha'] for v in V if v['ols'] and v['ols']['alpha'] is not None])

        # --- bootstrap CI (64 итерации, по документам, ревью #3) ---
        def _boot(arr, stat='median'):
            idx = boot_rng.integers(0, len(arr), (BOOT_N, len(arr)))
            s = np.median(arr[idx], axis=1) if stat == 'median' else arr[idx].mean(axis=1)
            return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
        sl_arr = np.array([v['s_l_median'] for v in V])
        d3b = {m: {r: _boot(d3fix[r] if m == 'fixP' else d3raw[r])
                   for r in R_SWEEP} for m in ('fixP',)}
        d3b['raw'] = {r: _boot(d3raw[r]) for r in R_SWEEP}
        d3b['gf'] = {r: _boot(d3gf[r]) for r in R_SWEEP}

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
                                  [float(np.percentile(alphas, 25)), float(np.percentile(alphas, 75))]]
                        if len(alphas) else None,
                        'pooled': _pooled_ols(cache, keys)},
                'n_docs_with_strata': int(len(db)),
            },
            'cross_checks': {k: _agg([v['cross_checks'][k] for v in V])
                             for k in ('rho_nb_dSVD_raw1', 'rho_nb_dSVD_fixP1',
                                       'rho_nb_dSVD_gf1', 'rho_nb_dp')},
            'sentinel_frac': _agg([v['sentinel_fraction'] for v in V]),
        }
        print(f'L{l}: s_l={sl[0]:.4f} slope={sr[0]:.4f} '
              f'd3r1 raw/gf/fixP = '
              f'{np.median(d3raw[1]):.3f}/{np.median(d3gf[1]):.3f}/{np.median(d3fix[1]):.3f} '
              f'd_b={np.median(db) if len(db) else None}', flush=True)

    report = {
        'meta': {
            'experiment': 'E-32r.2 (docs/Exp32.md + docs/Exp32r_errors.md)',
            'model': MODEL, 'layers': list(a.layers), 'n_docs': len(docs),
            'T': {str(l): (128 if l == 23 else 64) for l in a.layers}, 'eps_rel': [float(e) for e in EPS_REL], 'eps_canon': EPS_CANON,
            'fit': 'dz = a*e + b*e^2 + c*e^3 (без const), eps в абсолютных единицах h',
            's_l': 'eps_canon * ||off(b_t)||_F / ||off(ESG_raw_t)||_F (RAW-нормировка)',
            'aggregation': 'E-34: per-doc median -> median/IQR по документам',
            'sentinel': 'P128_RAW.sum < 0.5, endpoints-only valid pairs; '
                        'sentinel_frac = share raw rows, nb_median по ~sent',
            'p0': 'canonical: non-sentinel позиция с p, ближайшей к медиане (t0)',
            'd3': 'top-r right singular direction angle (cos-sim, как в exp33r)',
            'bootstrap': f'{BOOT_N} итераций по документам (seed=12345)',
            'pilot': 'results/exp32r_pilot.json (L23, 1 док) — базовые числа',
        },
        'layers': layers_out,
        'verdicts': {},
        'per_doc': {k: v for k, v in per_doc_layer.items()},
    }
    # --- вердикты ---
    sl_med = [layers_out[str(l)]['s_l']['median'] for l in a.layers]
    lin = {}
    for l in a.layers:
        vals = [layers_out[str(l)]['resid_frac_by_eps'][
            f'{int(round(EPS_REL[e] * 100))}pct']['median'] for e in range(NEPS)]
        lin[str(l)] = bool(np.median(vals).max() < 0.01)
    s_ok = all(v < 0.3 for v in sl_med)
    report['verdicts'] = {
        's_l_lt_0.3_all_layers': bool(s_ok),
        's_l_medians': {str(l): sl_med[i] for i, l in enumerate(a.layers)},
        'resid_frac_lt_1pct_per_layer': lin,
        'linear_verdict': bool(s_ok and all(lin.values())),
        'note': 'A/B/C разрешается через d3 sweep + стратификацию + кросс-чеки (см. layers.*)',
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(_clean(report), f, indent=1, ensure_ascii=False, allow_nan=False)
    print(f'WROTE {a.out}  total {time.time() - t0:.0f}s', flush=True)


def _pooled_ols(cache, keys):
    """Pooled OLS+HC3: все valid pairs всех документов одного слоя."""
    x_all = [cache[k]['dp_v'] for k in keys]
    y_all = [cache[k]['nb_v'] for k in keys]
    if sum(len(a) for a in x_all) < 5:
        return None
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    return ols_hc3(x, y, rng=np.random.default_rng(99), n_perm=1000)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
