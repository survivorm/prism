#!/usr/bin/env python
"""E-32r-пилот (docs/Exp32.md): кривизна readout L23 vs gauge-структура E-33r.

Дизайн (per-position, L23, T = U128 = 128, 1 документ wiki/0):
  h_t = H[L23][t] из кэша (pre-norm, E-35: H[l] = hidden_states[l+1][0]).
  Decoder tail L23 pointwise: logits = lm_head(U128)(RMSNorm(h) * g), g = 1 + norm.weight,
  RMSNorm = x * rsqrt(mean(x^2) + eps_norm), eps_norm = model.model.norm.eps.
  Инъекция d_i = J_t[i]/||J_t[i]|| (из кэша JR), h' = h_t + eps_abs * d_i,
  eps_abs = eps_rel * ||h_t||, eps_rel in {5,15,30,50,70}% (0.15 = canon).
  Batching: 128 x fwd(640) по t (OOM-guard: единый batch 81920 строк убивал процесс).
  Fit (векторизованный, cubic, в относительных eps): dz = a*eps + b*eps^2 + c*eps^3 (без const),
  A1 = a + 2 b eps_canon (экстраполяция на canon 15%); pC = p-value кубического члена c.
  s_t = 0.15*||h_t|| * ||off(b_t)||_F / ||off(ESG_t)||_F (ESG_raw per position, ТЗ «нормировка на raw»).
  Fixed-P: Jfix_t = P(p0)*J_t (P в token-пространстве), p0 = медиана p_t (E-33r.2 A),
  d3_*: top-1 right, max principal angle, pair (t, t+1), медианы по valid-парам (vp).
  Cross-checks per position: ||b(t)|| vs d3_raw / d3_gf / d3_fixP / |dp| / dS (H).
  Q1/Q4-стратификация по |dp| среди non-sentinel-пар (n~32 — только диагностика).
  slope_ratio — diagnostic: 1.0 = линейный режим; > 1.2 — нелинейность.

Selfcheck: (1) точное воспроизведение J из аналитического J = M_K c (I-u u^T)
  против кэша JR (E-35 identity); (2) finite-diff J_fd vs JR (subsample);
  (3) fit-декод синтетического cubic (без const); (4) gauge_free_rows = (diag(p)-pp^T)J;
  (5) векторизованный fit == попарный fit_cubic.
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
LAYER = 23
EPS_REL = [0.05, 0.15, 0.30, 0.50, 0.70]
EPS_CANON = 0.15
CKPT = '/data/prism_kernels/exp27b/results'
MODEL = 'Qwen/Qwen3.5-0.8B'
OUT = 'results/exp32r_pilot_L23.json'


# --------------------------------------------------------------------------
# math helpers
# --------------------------------------------------------------------------
def offdiag_norm(M):
    return float(math.sqrt(max((M * M).sum() - (np.diag(M) ** 2).sum(), 0.0)))


def gauge_free_rows(J, p):
    """P(p) = diag(p) − pp^T в token-пространстве: J [K,D] rows=tokens, p [K]."""
    return J * p[:, None] - (p @ J)[None, :] * p[:, None]


def top1_right(J):
    JJt = J @ J.T
    return np.linalg.eigh(JJt)[1][:, -1]


def right_top1(J):
    """top-1 right singular vector = J^T u / ||..||, u — top-1 left."""
    u = top1_right(J)
    v = J.T @ u
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def angle_max(a, b):
    c = abs(float(a @ b))
    return math.acos(min(1.0, c))


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - (np.arange(n) + 1) / (n + 1)
    ry = ry - (np.arange(n) + 1) / (n + 1)
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_cubic(eps, y):
    """1-D: y = a*eps + b*eps^2 + c*eps^3 (без const, ТЗ) по 5 точкам, p-значение c (t, dof=3)."""
    e = np.asarray(eps, dtype=np.float64)
    Z = np.asarray(y, dtype=np.float64)
    A = np.stack([e, e ** 2, e ** 3], axis=1)
    coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
    res = Z - A @ coef
    dof = Z.size - A.shape[1]
    XtXi = np.linalg.inv(A.T @ A)
    s2 = float(res @ res) / dof
    se = math.sqrt(max(s2 * XtXi[2, 2], 0.0))
    tstat = coef[2] / se if se > 0 else 0.0
    p_c = 2 * tdist.sf(abs(tstat), dof)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(p_c), res


def fit_all(eps_abs, dz):
    """Векторизованный cubic fit dz = a*e + b*e^2 + c*e^3 (без const, ТЗ).

    eps_abs [S,5], dz [S,T,T,5] -> A1(a), B(b), C(c), pC (p c) [S,T,T], resid [S,T,T,5].
    """
    assert dz.ndim == 4 and dz.shape[0] == eps_abs.shape[0] and dz.shape[-1] == eps_abs.shape[1], \
        f'dz {dz.shape}, eps {eps_abs.shape}'
    e = eps_abs.astype(np.float64)
    Z = dz if dz.dtype == np.float64 else dz.astype(np.float64)  # [S,T,T,5], без копии
    # Фит в относительных eps: eps_abs[s,e] = n_s * er[e] => одна константная инверсия 3x3
    nrm = e[:, 0] / EPS_REL[0]                              # [S]
    er = np.asarray(EPS_REL, dtype=np.float64)
    A0 = np.stack([er, er ** 2, er ** 3], -1)               # [5,3]
    inv0 = np.linalg.inv(A0.T @ A0)                         # [3,3]
    Atb = np.einsum('eu,sije->suij', A0, Z)                 # [S,3,T,T]
    coef_rel = np.einsum('pq,sqij->spij', inv0, Atb)        # [S,3,T,T]
    n1 = 1.0 / nrm
    a1 = (coef_rel[:, 0] * n1[:, None, None]).astype(np.float32)
    a2 = (coef_rel[:, 1] * (n1[:, None, None] ** 2)).astype(np.float32)
    a3 = (coef_rel[:, 2] * (n1[:, None, None] ** 3)).astype(np.float32)
    resid = Z - (a1[:, :, :, None] * e[:, None, None, :]
                 + a2[:, :, :, None] * e[:, None, None, :] ** 2
                 + a3[:, :, :, None] * e[:, None, None, :] ** 3)
    # p-value c: t = c_rel/sqrt(s2*inv0_cc) — nrm сокращается; dof = 5 - 3 = 2
    sse = (resid ** 2).sum(-1)                          # [S,T,T]
    s2v = sse / 2.0
    se_rel = np.sqrt(np.maximum(s2v * inv0[2, 2], 0.0))
    tstat = np.zeros_like(coef_rel[:, 2])
    np.divide(coef_rel[:, 2], se_rel, out=tstat, where=se_rel > 0)
    pC = (2 * tdist.sf(np.abs(tstat), 2)).astype(np.float32)
    return a1, a2, a3, pC, resid


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def load_decoder(U128):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation='eager').eval()
    W = model.lm_head.weight.detach().float().numpy()  # [vocab, D]
    g = (1 + model.model.norm.weight.detach().float()).numpy()  # [D]
    eps_norm = float(model.model.norm.eps)
    W_U = W[U128]  # [128, D]
    del model
    g_t = torch.from_numpy(g.astype(np.float32))
    W_U_t = torch.from_numpy(W_U.astype(np.float32)).T

    def forward(hs):  # hs [n, D] f32 -> logits [n, 128]
        x = torch.from_numpy(hs.astype(np.float32))
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps_norm)  # rms до g (Qwen3)
        x = (x * g_t).float()
        with torch.no_grad():
            lg = x @ W_U_t
        return lg.numpy()

    return forward, W_U, g, eps_norm


def analytic_J(W_U, g, h):
    """J = M_K (sqrt(D)/r_t) (I - u u^T), M_K = W_U diag(g). E-35 identity."""
    r = np.linalg.norm(h)
    u = h / r
    M = W_U * g[None, :]
    c = math.sqrt(D) / r
    return c * (M - (M @ u)[:, None] * u[None, :])


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--doc', default='wiki/0')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()

    E28._pin_p_cores()
    ck = E28.load_layer(CKPT, LAYER)
    (dom, si) = a.doc.split('/')
    sidx = int(si)
    Js, Ps, Raws, Hs = [], [], [], []
    for t in range(S):
        k = (dom, sidx, t)
        Js.append(ck['JR'][k].numpy())
        Ps.append(ck['P128'][k].numpy())
        Raws.append(ck['P128_RAW'][k].numpy())
        Hs.append(ck['H'][k].numpy())
    JR = np.stack(Js).astype(np.float64)   # [S, T, D] rows=tokens
    P128 = np.stack(Ps).astype(np.float64)  # [S, T]
    RAW = np.stack(Raws).astype(np.float64)  # [S, T]
    H = np.stack(Hs).astype(np.float64)      # [S, D]
    U128 = np.array(ck['U128'], dtype=np.int64)
    del ck
    assert JR.shape == (S, S, D) and P128.shape == (S, S) and H.shape == (S, D)
    T = S  # пилот: T = U128 (128), не 16 — см. meta.deviation

    sent = RAW.sum(-1) < 0.5
    # valid pair (t, t+1): оба конца non-sentinel (E-33r endpoints-only)
    vp = ~sent[:-1] & ~sent[1:]

    # --- forward ---
    t0 = time.time()
    fwd, W_U, g, eps_norm = load_decoder(U128)

    if a.selfcheck:
        # (1) analytic J vs cache at t=0
        h0 = H[0]
        J_ana = analytic_J(W_U, g, h0)
        cos = float((J_ana * JR[0]).sum() / (np.linalg.norm(J_ana) * np.linalg.norm(JR[0])))
        print(f'selfcheck (1) analytic J vs JR[0]: cos = {cos:.6f}', flush=True)
        assert abs(abs(cos) - 1) < 1e-4, 'E-35 identity fail (sign? abs)'
        # (1b) fwd(h0) == W_U @ RMSNorm(h0) (защита от рассинхрона eps_norm)
        y0 = h0 / math.sqrt(float((h0 ** 2).mean()) + eps_norm) * g
        assert np.allclose(W_U @ y0, fwd(h0[None, :])[0], atol=1e-3), 'fwd/RMSNorm desync'
        # (2) finite-diff J_fd vs JR[0] (subsample по h-координатам)
        dJ = []
        for i in np.arange(0, D, 32):
            e = np.zeros(D)
            e[i] = 1e-3
            dJ.append((fwd((h0 + e)[None, :])[0] - fwd((h0 - e)[None, :])[0]) / 2e-3)
        J_fd = np.stack(dJ).T  # [128, 32]: столбцы = ∂logits/∂h_i
        ref = JR[0][:, np.arange(0, D, 32)]
        cos2 = float((J_fd * ref).sum() / (np.linalg.norm(J_fd) * np.linalg.norm(ref)))
        print(f'selfcheck (2) finite-diff J vs JR[0]: cos = {cos2:.6f}', flush=True)
        assert abs(abs(cos2) - 1) < 1e-3, 'finite-diff fail'
        # (3) cubic fit (без const) decodes coefficients
        rng = np.random.default_rng(0)
        eps = np.array(EPS_REL)
        y = 1.0 * eps + 2.0 * eps ** 2 + 0.5 * eps ** 3 + 1e-6 * rng.standard_normal(5)
        a1f, a2f, a3f, pc, _ = fit_cubic(eps, y)
        assert abs(a1f - 1) < 0.05 and abs(a2f - 2) < 0.05 and abs(a3f - 0.5) < 0.05, \
            f'fit fail {a1f} {a2f} {a3f}'
        # (4) gauge_free_rows == (diag(p) − pp^T) J; uniform killed
        rng2 = np.random.default_rng(1)
        Jc = rng2.standard_normal((8, 6))
        pc_ = rng2.random(8)
        pc_ = pc_ / pc_.sum()
        assert np.allclose(gauge_free_rows(Jc, pc_), (np.diag(pc_) - np.outer(pc_, pc_)) @ Jc, atol=1e-12)
        assert np.allclose(gauge_free_rows(np.ones((8, 6)), pc_), 0.0, atol=1e-12)
        # (5) vectorized fit == per-triplet fit_cubic (f32-каст => tol 1e-5)
        rng3 = np.random.default_rng(2)
        Sa, Ta = 3, 4
        eabs = rng3.uniform(10, 30, (Sa, 1)) * np.asarray(EPS_REL)  # n_s*er: структура fit_all
        dz = rng3.standard_normal((Sa, Ta, Ta, 5))
        a1v, a2v, a3v, pCv, _ = fit_all(eabs, dz)
        for s_ in range(Sa):
            for i_ in range(Ta):
                for j_ in range(Ta):
                    a1, a2, a3, pc, _ = fit_cubic(eabs[s_], dz[s_, i_, j_])
                    assert abs(a1v[s_, i_, j_] - a1) < 1e-5
                    assert abs(a2v[s_, i_, j_] - a2) < 1e-5
                    assert abs(a3v[s_, i_, j_] - a3) < 1e-5
        print('selfcheck OK', flush=True)
        return

    # --- directions + injections ---
    Jn = JR / np.maximum(np.linalg.norm(JR, axis=-1)[..., None], 1e-12)  # [S, T, D] unit rows
    norms = np.linalg.norm(H, axis=1)  # [S]
    eps_abs = np.array(EPS_REL)[:, None] * norms[None, :]  # [5, S]
    # h_inj[t, i, e] = H[t] + eps_abs[e, t] * Jn[t, i]; f32, пик ~0.7 ГБ
    H32 = H.astype(np.float32)
    Jn32 = Jn.astype(np.float32)
    scaled = eps_abs.T.astype(np.float32)[:, None, :, None] * Jn32[:, :, None, :]
    h_inj = (H32[:, None, None, :] + scaled).reshape(-1, D)
    del scaled, H32, Jn32, Jn
    n_inj = h_inj.shape[0]
    print(f'forward: {n_inj} injections, 128 blocks x 640 rows', flush=True)
    t1 = time.time()
    base_all = fwd(H)  # [S, T], один раз
    dz32 = np.empty((S, T, T, len(EPS_REL)), dtype=np.float32)  # [t, i, j, e]
    for t_ in range(S):
        hc = h_inj[t_ * T * len(EPS_REL):(t_ + 1) * T * len(EPS_REL)]
        L = fwd(hc).reshape(T, len(EPS_REL), T).transpose(0, 2, 1)  # [i, j, e]
        dz32[t_] = L - base_all[t_][None, :, None]
    del h_inj, base_all
    dz_all = np.ascontiguousarray(dz32).astype(np.float64)  # [t, i, j, e] = layout fit_all
    del dz32
    print(f'forward done {time.time() - t1:.1f}s', flush=True)

    # --- per-position fits (vectorized) ---
    t1 = time.time()
    a1, a2, a3, pC, resid = fit_all(eps_abs.T, dz_all)  # [S,T,T] each, resid [S,T,T,5]
    # ТЗ: A1 = a1 + 2 a2 eps_canon, eps_canon_t = 0.15 * ||h_t||
    A1 = a1 + 2.0 * a2 * (EPS_CANON * norms)[:, None, None]
    print(f'fits done {time.time() - t1:.1f}s', flush=True)

    # ESG_raw per position
    ESG_all = np.einsum('std,sjd->stj', JR, JR)  # [S, T, T], один раз
    ESG_med = np.median(ESG_all, 0)  # для валидации a1
    ESGd = np.diagonal(ESG_all, axis1=1, axis2=2)  # [S, T]
    off_esg = np.sqrt(np.maximum((ESG_all ** 2).sum((-1, -2)) - (ESGd ** 2).sum(-1), 0.0))

    nb = np.array([offdiag_norm(a2[t]) for t in range(S)])  # b-матрица (cubic)
    s_t = EPS_CANON * norms * nb / np.maximum(off_esg, 1e-12)  # per position (ТЗ raw)
    c_sig = float((pC[~sent] < 0.05).mean())  # p кубического c, non-sentinel t
    # per-eps residual frac (диагностика); resid [S,T,T,5] из fit_all
    resid_frac = {}
    for e in range(len(EPS_REL)):
        resid_frac[f'{int(EPS_REL[e] * 100)}pct'] = float(
            np.linalg.norm(resid[:, :, :, e]) / max(float(np.linalg.norm(dz_all[:, :, :, e])), 1e-12))

    # --- fixed-P control (E-33r.2 A analog) ---
    p0 = np.median(P128, axis=0)
    p0 = p0 / p0.sum()
    # P(p0) = diag(p0) - p0 p0^T в token-пространстве: JR [S,T,D], p0 [T]
    Jr_p = JR * p0[None, :, None]
    Jfix = Jr_p - Jr_p.sum(1)[:, None, :] * p0[None, :, None]  # [S, T, D]
    d3_raw, d3_gf, d3_fix = [], [], []
    for t in range(S - 1):
        d3_raw.append(angle_max(right_top1(JR[t]), right_top1(JR[t + 1])))
        d3_gf.append(angle_max(right_top1(gauge_free_rows(JR[t], P128[t])),
                               right_top1(gauge_free_rows(JR[t + 1], P128[t + 1]))))
        d3_fix.append(angle_max(right_top1(Jfix[t]), right_top1(Jfix[t + 1])))
    d3_raw, d3_gf, d3_fix = (np.array(x) for x in (d3_raw, d3_gf, d3_fix))

    # --- cross-checks ---
    dp = np.linalg.norm(np.diff(P128, axis=0), axis=1)  # [S-1]
    dS = np.arccos(np.clip((H[:-1] * H[1:]).sum(-1) / (norms[:-1] * norms[1:]), -1, 1))
    # per-position series aligned on valid pairs: nb[t] (b at t) vs d3_*[t], dp[t], dS[t]
    tidx = np.where(vp)[0]
    nb_pair = nb[tidx]
    rho_raw = spearman(nb_pair, d3_raw[tidx])
    rho_gf = spearman(nb_pair, d3_gf[tidx])
    rho_fix = spearman(nb_pair, d3_fix[tidx])
    rho_dp = spearman(nb_pair, dp[tidx])
    rho_dS = spearman(nb_pair, dS[tidx])

    # --- Q1/Q4 stratification (diagnostic, non-sentinel pairs — п.13 ревью) ---
    dpv = dp[vp]
    q = np.percentile(dpv, [25, 75])
    Q1 = np.where((dp <= q[0]) & vp)[0]
    Q4 = np.where((dp >= q[1]) & vp)[0]
    med_a2 = np.median(a2, 0)  # [T, T], один раз (b_mean и d_b)
    mQ1 = np.median(a2[Q1], 0) if len(Q1) else None
    mQ4 = np.median(a2[Q4], 0) if len(Q4) else None
    bQ1 = None if mQ1 is None else float(np.linalg.norm(mQ1))  # Frobenius
    bQ4 = None if mQ4 is None else float(np.linalg.norm(mQ4))  # Frobenius
    b_mean = np.linalg.norm(med_a2)  # deviation: медиана по t, ТЗ подразумевает среднее
    d_b = float(np.linalg.norm(mQ4 - mQ1) / max(b_mean, 1e-12)) \
        if (mQ1 is not None and mQ4 is not None) else None

    # --- mode diagnostics: linearity of |dz| at small eps (diagnostic) ---
    dze1 = np.abs(dz_all[:, :, :, 1])
    slope_ratio = float(np.median(np.abs(dz_all[:, :, :, 0]) / np.maximum(dze1, 1e-3)
                                  / (EPS_REL[0] / EPS_REL[1])))
    frac_dz_small = float((dze1 < 1e-6).mean())  # доля пар, где |dz|(15%) шумовой уровень

    report = {
        'meta': {
            'experiment': 'E-32r-пилот (docs/Exp32.md)',
            'model': MODEL, 'layer': LAYER, 'doc': a.doc, 'eps_norm': eps_norm,
            'T': T, 'n_eps': len(EPS_REL), 'eps_rel': EPS_REL, 'eps_canon': EPS_CANON,
            'sentinel_fraction': float(sent.mean()),
            'deviation': ('T = U128 = 128, не 16: L23 decoder tail pointwise, '
                          'стоимость нулевая; native P(p_t) и кросс-чеки E-33r '
                          '(128-токенное JR). Пилот на 1 док — CI нет. '
                          'b_mean = медиана b по t (не среднее, ТЗ ||b_mean||_F).'),
            'fit': ('absolute eps = eps_rel * ||h_t||; dz = a eps + b eps^2 + c eps^3 (без const, ТЗ); '
                     'A1 = a + 2 b eps_canon (canon 15%); pC = p c'),
        },
        's_l_median': float(np.median(s_t)),
        's_l_iqr': [float(np.percentile(s_t, 25)), float(np.percentile(s_t, 75))],
        'c_significant_frac': c_sig,
        'residual_frac': resid_frac,
        'slope_ratio_linearity': {'value': slope_ratio, 'expected_linear': 1.0,
                                   'frac_dz_eps2_lt_1e-6': frac_dz_small},
        'fixed_p': {
            'd3_raw_med': float(np.median(d3_raw[vp])),
            'd3_gf_med': float(np.median(d3_gf[vp])),
            'd3_fixP_med': float(np.median(d3_fix[vp])),
            'n_valid_pairs': int(vp.sum()),
        },
        'cross_checks': {
            'rho_nb_d3_raw': rho_raw,
            'rho_nb_d3_gf': rho_gf,
            'rho_nb_d3_fixP': rho_fix,
            'rho_nb_dp': rho_dp,
            'rho_nb_dS': rho_dS,
            'n': int(len(tidx)),
        },
        'stratification': {
            'Q1_n': int(len(Q1)), 'Q4_n': int(len(Q4)),
            'bQ1_fro': None if bQ1 is None else float(bQ1),
            'bQ4_fro': None if bQ4 is None else float(bQ4),
            'delta_b_rel': d_b,
        },
        'validation': {
            'a_vs_ESG_cos': None,  # filled below
            'off_esg_F_med': float(np.median(off_esg)),
            'A1_median': float(np.median(A1)),
            'A1_vs_a1_med_ratio': float(np.median(A1 / np.maximum(np.abs(a1), 1e-12))),
        },
        'per_doc': {a.doc: {
            'nb_median': float(np.median(nb)),
            's_median': float(np.median(s_t)),
        }},
    }
    # validation: a vs normalized ESG (Frobenius косинус; нормировка по столбцу i)
    amed = np.median(a1, 0)  # ТЗ: «a vs normalized ESG» — линейный коэффициент a1
    ESGn = ESG_med / np.maximum(np.sqrt(np.diag(ESG_med))[None, :], 1e-12)
    report['validation']['a_vs_ESG_cos'] = float(
        (amed * ESGn).sum() / max(np.linalg.norm(amed) * np.linalg.norm(ESGn), 1e-12))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: report[k] for k in
                      ('s_l_median', 'c_significant_frac', 'residual_frac',
                       'fixed_p', 'cross_checks', 'stratification',
                       'validation')}, indent=1, ensure_ascii=False), flush=True)
    print(f'WROTE {a.out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
