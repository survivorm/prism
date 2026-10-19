#!/usr/bin/env python3
"""Exp 35 preflight — регрессионный контроль: геометрия vs выходное распределение.

docs/Exp35.md, секция E-35-preflight. Per-position регрессия, L23, per документ:

    dSVD(t) = alpha * d_S(u_t, u_{t+1}) + beta * |dp_t| + eps

  dSVD(t) — максимальный главный угол топ-r правых подпространств J_t, J_{t+1}
    (r=16 основной; r in {8,32} — устойчивость), считается из кэша exp27b
    (batched SVD + Vh-префиксы, E28.right_subspaces) — per-position Δ_SVD в
    results/exp28_leaf_structure.json нет.
  d_S = arccos(<u_t, u_{t+1}>), u = H_cache / ||H_cache||.
  |dp_t| = ||p_{t+1} - p_t||_2, p = P128 normalized.
  Per-doc OLS + HC3 robust SE, VIF, частичная корреляция с d_S (при контроле
  |dp|), R^2, runs test остатков, Breusch-Pagan.
  Исключения: sentinel (сумма P128_RAW < 0.5, E34.SENTINEL_SUM; P128_RAW в кэше,
  ~50.7% позиций) и SVD-degen (sigma_r < DEGEN_REL*||J||_F, E28) — пара
  (t, t+1) валидна, если оба конца валидны.
  d-u чек: d_S на границах (MAD-детектор E-28) vs вне границ, стратифицированный
  пермутационный тест (перестановка меток «граница» внутри каждого документа,
  B пермутаций).
  Ось 2: готовая атрибуция из results/exp28_semantic.json — артефакт 2 = она же.
  Контроль E-34: corr(rho_raw, |Δp|) из results/exp34_p1_coupling.json.
  Контроль E-32 (||b(t)||, Hessian): неприменим — E-32 не выполнен.

statsmodels нет — OLS/HC3/VIF/partial/runs/BP на numpy+scipy.

Артефакты: results/exp35_preflight_regression.json,
           results/exp35_preflight_axis2_attribution.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import t as tdist
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp28_leaf_structure as E28
import exp34_p1_coupling as E34

LAYER = 23
R_MAIN = 16
B_PERM = 2000
MIN_VALID = 20
SENTINEL_SUM = E34.SENTINEL_SUM
DEGEN_REL = E28.DEGEN_REL
CKPT_DIR = '/data/prism_kernels/exp27b/results'


def _tcdf(z, df):
    return tdist.cdf(z, df)


def dS(d):
    return float(np.arccos(np.clip(d, -1.0, 1.0)))


def valid_pairs(sent, deg):
    """Маска пар (t, t+1): оба конца не sentinel и не degenerate."""
    v = ~(sent | deg)
    return v[:-1] & v[1:]


def ols_hc3(x1, x2, y):
    """OLS y ~ a + b*x1 + c*x2, HC3 robust SE. b — d_S, c — |dp|."""
    n = len(y)
    X = np.column_stack([np.ones(n), x1, x2])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    h = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    w = (resid / (1.0 - h)) ** 2
    V = XtX_inv @ (X.T @ (X * w[:, None])) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    tval = beta / se
    pval = 2.0 * (1.0 - _tcdf(np.abs(tval), n - 3))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float('nan')
    return {'a': beta[0], 'b': beta[1], 'c': beta[2],
            'se_b': se[1], 'se_c': se[2], 'p_b': pval[1], 'p_c': pval[2],
            'r2': r2, 'resid': resid, 'X': X}


def _r2_on(a, b):
    X = np.column_stack([np.ones(len(a)), b])
    r = np.linalg.lstsq(X, a, rcond=None)[0]
    e = a - X @ r
    sst = ((a - a.mean()) ** 2).sum()
    return 1.0 - float(e @ e) / sst if sst > 0 else 0.0


def vif(x1, x2):
    r1, r2 = _r2_on(x1, x2), _r2_on(x2, x1)
    return 1.0 / (1.0 - r1), 1.0 / (1.0 - r2)


def _resid_on(z, w):
    X = np.column_stack([np.ones(len(z)), w])
    r = np.linalg.lstsq(X, z, rcond=None)[0]
    return z - X @ r


def partial_corr_ds(x1, x2, y):
    """Частичная корреляция y с x1 (d_S) при контроле x2 (|dp|)."""
    e_y, e_1 = _resid_on(y, x2), _resid_on(x1, x2)
    den = np.sqrt((e_y @ e_y) * (e_1 @ e_1))
    r = float(e_y @ e_1 / den) if den > 0 else float('nan')
    n = len(y)
    if abs(r) >= 1.0 or n < 5:
        return r, float('nan')
    tval = r * np.sqrt((n - 3) / (1 - r * r))
    return r, float(2.0 * (1.0 - _tcdf(abs(tval), n - 3)))


def runs_test(resid):
    """Wald-Wolfowitz runs test на знаках остатков."""
    s = np.sign(resid)
    s = s[s != 0]
    n1, n2 = int((s == 1).sum()), int((s == -1).sum())
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    runs = 1 + int((s[1:] != s[:-1]).sum())
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    if var <= 0:
        return float('nan'), float('nan')
    z = (runs - mu) / np.sqrt(var)
    return float(z), float(2.0 * (1.0 - ndtr(abs(z))))


def breusch_pagan(X, resid):
    """BP: LM = n*R^2 регрессии resid^2 на X."""
    z = resid ** 2
    r = np.linalg.lstsq(X, z, rcond=None)[0]
    e = z - X @ r
    sst = ((z - z.mean()) ** 2).sum()
    r2 = 1.0 - float(e @ e) / sst if sst > 0 else 0.0
    lm = len(resid) * r2
    return float(lm), float(chi2.sf(lm, X.shape[1] - 1))


def boundary_mask(d1):
    """MAD-детектор границ E-28: d1 > median + max(2*MAD, 1e-6)."""
    dn = d1[~np.isnan(d1)]
    dm = float(np.median(dn))
    mad = float(np.median(np.abs(dn - dm)))
    return d1 > dm + max(2.0 * mad, 1e-6)


def doc_metrics(J, H, P, Pr, subs, r, keep=False):
    """Все preflight-метрики одного документа для фиксированного r.
    J [T,K,D], H [T,D], P [T,K] (norm), Pr [T,K] (raw). subs — из
    E28.right_subspaces(J, [r]). keep=True — возвращает и per-position
    массивы (для пермутационного теста)."""
    T = J.shape[0]
    V = subs[r][0]
    jnorm = J.norm(dim=(1, 2))
    deg = subs[r][1] < DEGEN_REL * jnorm.clamp_min(1e-30)
    sent = (Pr.sum(-1) < SENTINEL_SUM).numpy()
    vv = valid_pairs(sent, deg.numpy()).astype(bool)
    M1 = V[:-1].transpose(-1, -2) @ V[1:]
    d1 = torch.acos(torch.linalg.svdvals(M1).clamp(max=1.0 - 1e-12)).max(-1).values.numpy()
    hn = H.norm(dim=1).numpy()
    u = H.numpy() / np.clip(hn, 1e-30, None)[:, None]
    ds = np.arccos(np.clip((u[:-1] * u[1:]).sum(-1), -1.0, 1.0))
    dp = np.linalg.norm(P[1:] - P[:-1], axis=1)
    boundary = boundary_mask(d1)
    row = {'n_valid': int(vv.sum()), 'd1_med': float(np.median(d1[vv])) if vv.any() else float('nan')}
    y, xs, xp = d1[vv], ds[vv], dp[vv]
    if len(y) < MIN_VALID:
        row['ok'] = False
        return row
    fit = ols_hc3(xs, xp, y)
    v1, v2 = vif(xs, xp)
    pc, pc_p = partial_corr_ds(xs, xp, y)
    rz, rp = runs_test(fit['resid'])
    blm, bp = breusch_pagan(fit['X'], fit['resid'])
    bb = boundary & vv
    row.update({'ok': True,
                'alpha': fit['b'], 'beta': fit['c'],
                'alpha_se': fit['se_b'], 'beta_se': fit['se_c'],
                'alpha_p': fit['p_b'], 'beta_p': fit['p_c'],
                'vif_ds': v1, 'vif_dp': v2,
                'partial_corr_ds': pc, 'partial_corr_ds_p': pc_p,
                'r2': fit['r2'], 'runs_z': rz, 'runs_p': rp,
                'bp_lm': blm, 'bp_p': bp,
                'ds_boundary': float(ds[bb].mean()) if bb.any() else float('nan'),
                'ds_rest': float(ds[vv & ~boundary].mean()) if (vv & ~boundary).any() else float('nan'),
                'n_boundary': int(bb.sum())})
    if keep:
        row['_ds'], row['_lab'] = ds[bb], (vv & ~boundary)
    return row


def ds_perm_check(stores, b_perm, seed):
    """Стратифицированный пермутационный тест d_S границ vs вне.
    stores: [(ds_boundary_values, ds_rest_values) per doc]. Statistic:
    mean(boundary pool) - mean(rest pool). Метки «граница» переставляются
    внутри каждого документа (пул из всех его валидных пар)."""
    g = np.random.default_rng(seed)
    obs = (np.mean([v for db, _ in stores for v in db]) -
           np.mean([v for _, dr in stores for v in dr]))
    cnt = 0
    for _ in range(b_perm):
        b_pool, r_pool = [], []
        for db, dr in stores:
            allv = np.concatenate([np.asarray(db, float), np.asarray(dr, float)])
            lab = np.concatenate([np.ones(len(db), bool), np.zeros(len(dr), bool)])
            lab = g.permutation(lab)
            b_pool.append(allv[lab]); r_pool.append(allv[~lab])
        stat = np.concatenate(b_pool).mean() - np.concatenate(r_pool).mean()
        cnt += stat >= obs if obs >= 0 else stat <= obs
    p = (cnt + 1) / (b_perm + 1)
    return {'obs_diff': float(obs),
            'ds_boundary_pool': float(np.mean([v for db, _ in stores for v in db])),
            'ds_rest_pool': float(np.mean([v for _, dr in stores for v in dr])),
            'b_perm': b_perm, 'perm_p': float(p),
            'method': 'stratified permutation, boundary labels shuffled within each document'}


def verdict(m):
    """Критерии остановки ТЗ: frac_alpha_sig / frac_beta_sig > 0.5 = значим."""
    a_sig = m['frac_alpha_sig'] > 0.5
    b_sig = m['frac_beta_sig'] > 0.5
    high_vif = m['vif_ds_med'] > 5
    if a_sig and not b_sig:
        return ('alpha significant, beta not -> geometry is genuine; '
                'E-35-full as planned (M-pinning, r-sweep)')
    if b_sig and not a_sig:
        return ('beta significant, alpha not -> dSVD is a |dp| proxy; '
                'STOP E-35-full, redefine')
    if a_sig and b_sig:
        if high_vif:
            return ('both significant, VIF > 5 -> multicollinear; '
                    'decide by partial corr (partial_corr_ds_med)')
        return 'both significant, VIF < 5 -> mixed; E-35-full with |dp| covariate'
    return ('neither significant -> regression explains nothing; '
            'look for other predictors (e.g. M_K structure)')


def _clean(o):
    """NaN/Inf -> None, numpy-типы -> python, чтобы json.dump(allow_nan=False) не падал."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(description='E-35 preflight: dSVD ~ a*d_S + b*|dp| (CPU)')
    ap.add_argument('--ckpt-dir', default=CKPT_DIR)
    ap.add_argument('--out-reg', default='results/exp35_preflight_regression.json')
    ap.add_argument('--out-axis2', default='results/exp35_preflight_axis2_attribution.json')
    ap.add_argument('--r', type=int, default=R_MAIN)
    ap.add_argument('--r-sweep', default='8,16,32')
    ap.add_argument('--b-perm', type=int, default=B_PERM)
    ap.add_argument('--sample-cap', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    E28._pin_p_cores()
    torch.manual_seed(a.seed)
    ck = E28.load_layer(a.ckpt_dir, LAYER)
    D, S, n, domains = ck['D'], ck['S'], ck['n_samples'], ck['domains']
    assert 'P128_RAW' in ck, 'P128_RAW not in cache'
    rs = [int(x) for x in a.r_sweep.split(',')]
    assert a.r in rs
    per_doc, per_r, stores = {}, {r: [] for r in rs}, []
    for dom in domains:
        for si in range(n if a.sample_cap == 0 else min(n, a.sample_cap)):
            J = torch.stack([ck['JR'][(dom, si, t)] for t in range(S)])
            H = torch.stack([ck['H'][(dom, si, t)] for t in range(S)])
            P = torch.stack([ck['P128'][(dom, si, t)] for t in range(S)])
            Pr = torch.stack([ck['P128_RAW'][(dom, si, t)] for t in range(S)])
            subs = E28.right_subspaces(J, rs)
            rows = {}
            for r in rs:
                row = doc_metrics(J, H, P, Pr, subs, r, keep=(r == a.r))
                if row.get('ok') and r == a.r:
                    stores.append((row.pop('_ds'), row.pop('_lab')))
                    # конвенция [T,D,r]: Δ_SVD должен быть O(0.1-0.6), не π/2
                    assert row['d1_med'] < 1.0, f'd1_med={row["d1_med"]} — правая конвенция подпространства?'
                rows[str(r)] = row
                if row.get('ok'):
                    per_r[r].append(row)
            per_doc[f'{dom}/{si}'] = rows
            del J, H, P, Pr
    agg = {}
    for r in rs:
        rows = per_r[r]
        if not rows:
            agg[str(r)] = {'n_docs': 0}
            continue
        agg[str(r)] = {
            'n_docs': len(rows),
            'alpha_med': float(np.median([x['alpha'] for x in rows])),
            'beta_med': float(np.median([x['beta'] for x in rows])),
            'alpha_p_med': float(np.median([x['alpha_p'] for x in rows])),
            'beta_p_med': float(np.median([x['beta_p'] for x in rows])),
            'frac_alpha_sig': float(np.mean([x['alpha_p'] < 0.05 for x in rows])),
            'frac_beta_sig': float(np.mean([x['beta_p'] < 0.05 for x in rows])),
            'partial_corr_ds_med': float(np.median([x['partial_corr_ds'] for x in rows])),
            'r2_med': float(np.median([x['r2'] for x in rows])),
            'vif_ds_med': float(np.median([x['vif_ds'] for x in rows])),
            'vif_dp_med': float(np.median([x['vif_dp'] for x in rows])),
        }
    report = {
        'experiment': 'E-35-preflight',
        'layer': LAYER, 'r_main': a.r, 'r_sweep': rs,
        'n_docs': len(per_doc), 'S': S, 'D': D,
        'min_valid_pairs': MIN_VALID,
        'sentinel': f'P128_RAW sum < {SENTINEL_SUM} (~50.7% of positions)',
        'degen': f'sigma_r < {DEGEN_REL}*||J||_F',
        'aggregates': agg,
        'per_doc': per_doc,
        'ds_boundary_check': ds_perm_check(stores, a.b_perm, a.seed),
        'e34_control': None,
        'e32_control': 'not applicable — Hessian (E-32) not computed',
        'source': 'experiments/exp35_preflight.py',
    }
    p34 = 'results/exp34_p1_coupling.json'
    if os.path.exists(p34):
        with open(p34) as f:
            d34 = json.load(f)
        report['e34_control'] = {
            'corr_rho_raw_dP_layer_median': d34['layers'][str(LAYER)]['correlation_rho_dP']['median'],
            'source': p34}
    m = agg[str(a.r)]
    du = report['ds_boundary_check']
    report['verdict'] = verdict(m)
    os.makedirs(os.path.dirname(a.out_reg) or '.', exist_ok=True)
    with open(a.out_reg, 'w') as f:
        json.dump(_clean(report), f, indent=2, allow_nan=False)
    with open('results/exp28_semantic.json') as f:
        axis2 = json.load(f)['axis2_token_attribution']
    with open(a.out_axis2, 'w') as f:
        json.dump({'axis2_token_attribution': axis2,
                   'source': 'results/exp28_semantic.json (computed in exp28_semantic.py)',
                   'note': 'attribution pre-computed by the exp28 semantic run; copied here per TZ artifact list'},
                  f, indent=2, allow_nan=False)
    print(f"E-35 preflight L{LAYER} r={a.r}: n_docs={m['n_docs']}, "
          f"alpha_med={m['alpha_med']:.4f} (p_med={m['alpha_p_med']:.3g}, sig={m['frac_alpha_sig']:.2f}), "
          f"beta_med={m['beta_med']:.3g} (p_med={m['beta_p_med']:.3g}, sig={m['frac_beta_sig']:.2f}), "
          f"partial(ds)={m['partial_corr_ds_med']:.3f}, VIF(ds)={m['vif_ds_med']:.1f}, "
          f"R2={m['r2_med']:.2f}, ds_bound/rest={du['ds_boundary_pool']:.4f}/{du['ds_rest_pool']:.4f} (p={du['perm_p']:.3g})")
    print(f"artifacts: {a.out_reg}, {a.out_axis2}")


def selfcheck():
    g = np.random.default_rng(0)
    # dS: known angle
    u = np.array([1.0, 0.0, 0.0]); v = np.array([np.cos(0.5), np.sin(0.5), 0.0])
    assert abs(dS(u @ v) - 0.5) < 1e-12
    # OLS/HC3: known coefficients
    n = 400
    x1, x2 = g.normal(size=n), g.normal(size=n)
    y = 0.5 * x1 + 2.0 * x2 + g.normal(scale=0.1, size=n)
    fit = ols_hc3(x1, x2, y)
    assert abs(fit['b'] - 0.5) < 0.05 and abs(fit['c'] - 2.0) < 0.1, (fit['b'], fit['c'])
    assert fit['p_b'] < 0.01 and fit['p_c'] < 0.01
    # HC3-SE: при noise sigma=0.1, x~N(0,1), n=400 -> se ~ sigma/sqrt(n) = 0.005
    assert 0.003 < fit['se_b'] < 0.01 and 0.003 < fit['se_c'] < 0.01, (fit['se_b'], fit['se_c'])
    assert 0.9 < fit['r2'] <= 1.0
    # VIF: collinear -> >5, independent -> ~1
    v1, v2 = vif(x2 + g.normal(scale=0.01, size=n), x2)
    assert v1 > 5 and v2 > 5, (v1, v2)
    v1b, _ = vif(x1, x2)
    assert v1b < 1.1, v1b
    # partial corr: with control, recovers x1 signal; p < 0.01
    pc, pc_p = partial_corr_ds(x1, x2, y)
    assert pc > 0.3 and pc_p < 0.01, (pc, pc_p)
    pc2, _ = partial_corr_ds(x1, x2, x1 + g.normal(scale=0.05, size=n))
    assert abs(pc2 - 1.0) < 0.05, pc2
    # runs: iid -> not rejected; autocorrelated -> rejected
    _, rp = runs_test(g.normal(size=300))
    assert 0 < rp < 1
    ar = np.zeros(300)
    for i in range(1, 300):
        ar[i] = 0.9 * ar[i - 1] + g.normal(scale=0.1)
    _, rp2 = runs_test(ar)
    assert rp2 < 0.05, rp2
    # BP: homoskedastic -> not rejected; heteroskedastic -> rejected
    X = fit['X']
    _, bp = breusch_pagan(X, fit['resid'])
    assert bp > 0.05, bp
    # монотонная гетероскедастичность (симметричная |x1| видима BP через линейный X хуже)
    scale = 1.0 + 20.0 * (x1 - x1.min()) / (x1.max() - x1.min())
    het = 0.5 * x1 + 2.0 * x2 + scale * g.normal(size=n) * 0.1
    res_h = het - X @ np.linalg.lstsq(X, het, rcond=None)[0]
    _, bp2 = breusch_pagan(X, res_h)
    assert bp2 < 0.01, bp2
    # valid_pairs
    sent = np.array([True, False, False, True])
    deg = np.array([False, False, True, False])
    assert valid_pairs(sent, deg).tolist() == [False, False, False]
    # permutation test: separated boundary signal -> small p
    stores = [(np.full(10, 0.5) + g.normal(scale=0.01, size=10),
               np.full(10, 0.1) + g.normal(scale=0.01, size=10)) for _ in range(4)]
    res = ds_perm_check(stores, 200, 1)
    assert res['perm_p'] < 0.05, res
    # permutation test: no signal -> p not small
    stores0 = [(g.normal(size=10), g.normal(size=10)) for _ in range(4)]
    res0 = ds_perm_check(stores0, 200, 2)
    assert res0['perm_p'] > 0.05, res0
    # verdict: four TZ stop-criteria branches
    mv = lambda aa, bb, vif: {'frac_alpha_sig': aa, 'frac_beta_sig': bb, 'vif_ds_med': vif}
    assert 'as planned' in verdict(mv(0.9, 0.1, 2))
    assert 'STOP' in verdict(mv(0.1, 0.9, 2))
    assert 'multicollinear' in verdict(mv(0.9, 0.9, 6))
    assert 'covariate' in verdict(mv(0.9, 0.9, 2))
    assert 'other predictors' in verdict(mv(0.1, 0.1, 2))
    # _clean: NaN -> None recursively
    c = _clean({'a': float('nan'), 'b': [1.0, float('inf')], 'c': 2.0})
    assert c == {'a': None, 'b': [1.0, None], 'c': 2.0}, c
    print('selfcheck OK')


if __name__ == '__main__':
    main()
