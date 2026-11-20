#!/usr/bin/env python3
"""E-32r.3/r.2 — disk-only реанализ HC3-SE: store не перепускается.

Зачем. В `exp32r3_layer_profile.py`/`exp32r2_hessian.py` «HC3» считался как
    se_j = sqrt( diag(XtX^-1)_jj * sum_i w_i ),  w_i = e_i^2/(1-h_i)^2
т.е. один скаляр на все параметры (верно только для интерсепта). Правильный
сэндвич: V = (XtX)^-1 X^T diag(w) X (XtX)^-1. Направление ошибки — SE завышен
(в 20-70x по синтетике из audit-записи), значит alpha_p завышен, и отклонение
гипотезы (B) position-dependent могло быть артефактом SE.

Что делает. Берёт `results/exp32r3_checkpoint.pt` (raw пары `dp_v`/`nb_v` по
документам — ровно те входы, из которых собран `pooled` OLS), для каждого слоя:
  1) воспроизводит опубликованный SE/alpha_p багованной формулой (проверка,
     что реконструкция входа верна: должно совпасть со store до ~1e-9);
  2) считает honest HC3 (сэндвич), HC1, классический SE;
  3) считает honest permutation-p на статистике t с honest SE;
  4) выносит решение по прежнему правилу порога (двусторонний p<0.05).

Правило решения НЕ меняется задним числом: тот же регрессор, тот же порог,
тот же двусторонний тест; меняется только реализация SE (багфикс) — по
AGENTS.md п.10 это новая ревизия машины + реанализ, оба вердикта публикуются.

Запуск: python experiments/e31_40/exp32r3_hc3_reanalysis.py
Артефакт: results/exp32r3_hc3_reanalysis.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import t as tdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CKPT = os.path.join(ROOT, 'results', 'exp32r3_checkpoint.pt')
STORE = os.path.join(ROOT, 'results', 'exp32r3_layer_profile.json')
OUT = os.path.join(ROOT, 'results', 'exp32r3_hc3_reanalysis.json')

MACHINE = 'E32-HC3-reanalysis'
MACHINE_SUBREV = 'r1'
N_PERM = 10_000
ALPHA_LEVEL = 0.05


def design(x: np.ndarray) -> dict:
    """X = [1, x]; возвращает (XtX)^-1, hat-диагональ, w-знаменатель."""
    n = len(x)
    X = np.stack([np.ones(n), x], axis=1)
    XtX_inv = np.linalg.inv(X.T @ X)
    lev = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    return {'X': X, 'XtX_inv': XtX_inv, 'lev': lev, 'n': n, 'x': x}


def fit(D: dict, y: np.ndarray) -> dict:
    """Точечные оценки + четыре версии SE для наклона alpha (beta_hat[1])."""
    X, XtX_inv, lev, n = D['X'], D['XtX_inv'], D['lev'], D['n']
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    w = (resid ** 2) / np.maximum(1.0 - lev, 1e-12) ** 2

    # --- багованная формула (как в проде): скаляр на все параметры
    se_bug = np.sqrt(np.maximum(np.diag(XtX_inv) * float(w.sum()), 0.0))

    # --- honest sandwich: V = (XtX)^-1 X^T diag(w) X (XtX)^-1
    meat = (X * w[:, None]).T @ X
    V_hc3 = XtX_inv @ meat @ XtX_inv
    se_hc3 = float(np.sqrt(max(V_hc3[1, 1], 0.0)))

    # --- HC1 (конечная поправка n/(n-2)), для полноты картины
    V_hc1 = V_hc3 * (n / max(n - 2, 1))
    se_hc1 = float(np.sqrt(max(V_hc1[1, 1], 0.0)))

    # --- классический homoskedastic SE: sigma^2 (XtX)^-1
    sigma2 = float(resid @ resid) / max(n - 2, 1)
    se_class = float(np.sqrt(max(sigma2 * XtX_inv[1, 1], 0.0)))

    def t_p(se):
        if not se > 0:
            return None, None
        tstat = float(beta[1] / se)
        return tstat, float(2 * tdist.sf(abs(tstat), n - 2))

    t_bug, p_bug = t_p(float(se_bug[1]))
    t_hc3, p_hc3 = t_p(se_hc3)
    t_hc1, p_hc1 = t_p(se_hc1)
    t_cls, p_cls = t_p(se_class)

    # effect size (AGENTS.md п.6): значимость != величина эффекта;
    # вклад позиционного плеча в отклик при среднем |dp|
    mean_x = float(np.mean(D['x']))
    effect_share = (float(beta[1]) * mean_x / float(beta[0])
                    if abs(float(beta[0])) > 1e-12 else None)
    return {
        'n': n, 'alpha': float(beta[1]), 'beta': float(beta[0]),
        'mean_abs_dp': mean_x, 'alpha_x_mean_dp': float(beta[1]) * mean_x,
        'alpha_x_mean_dp_over_intercept': effect_share,
        'alpha_se_buggy': float(se_bug[1]), 'alpha_t_buggy': t_bug,
        'alpha_p_buggy': p_bug,
        'alpha_se_hc3': se_hc3, 'alpha_t_hc3': t_hc3, 'alpha_p_hc3': p_hc3,
        'alpha_se_hc1': se_hc1, 'alpha_p_hc1': p_hc1,
        'alpha_se_classical': se_class, 'alpha_p_classical': p_cls,
        'se_ratio_honest_over_buggy': (se_hc3 / float(se_bug[1])
                                       if float(se_bug[1]) > 0 else None),
    }


def perm_p_honest(D: dict, y: np.ndarray, obs_abs_t: float,
                  rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Permutation-p на |t| с честным (сэндвич) SE: статистика инвариантна."""
    X, XtX_inv, lev, n = D['X'], D['XtX_inv'], D['lev'], D['n']
    cnt = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(n)]
        bp = XtX_inv @ (X.T @ yp)
        rp = yp - X @ bp
        w = (rp ** 2) / np.maximum(1.0 - lev, 1e-12) ** 2
        V = XtX_inv @ ((X * w[:, None]).T @ X) @ XtX_inv
        s = float(np.sqrt(max(V[1, 1], 0.0)))
        if s > 0 and abs(bp[1] / s) >= obs_abs_t:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main() -> int:
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    cache = ck['cache']
    store = json.load(open(STORE, encoding='utf-8'))

    by_layer: dict[int, list[str]] = {}
    for k in cache:
        by_layer.setdefault(int(k.split('|')[0]), []).append(k)

    rng = np.random.default_rng(99)  # тот же сид, что в проде
    out_layers, repro_err = {}, []
    for layer in sorted(by_layer):
        keys = sorted(by_layer[layer])
        x = np.concatenate([np.asarray(cache[k]['dp_v'], float) for k in keys])
        y = np.concatenate([np.asarray(cache[k]['nb_v'], float) for k in keys])
        D = design(x)
        r = fit(D, y)
        r['perm_p_honest_hc3'] = perm_p_honest(
            D, y, abs(r['alpha_t_hc3']) if r['alpha_t_hc3'] is not None else 0.0,
            rng)

        pub = (store['layers'][str(layer)]['stratification']['ols']
               .get('pooled') or {})
        # воспроизведение store: alpha, buggy SE, buggy p
        for mine, theirs, key in (
                (r['alpha'], pub.get('alpha'), 'alpha'),
                (r['alpha_se_buggy'], pub.get('alpha_se'), 'alpha_se'),
                (r['alpha_p_buggy'], pub.get('alpha_p'), 'alpha_p')):
            if theirs is None or mine is None:
                repro_err.append({'layer': layer, 'field': key,
                                  'status': 'missing'})
            else:
                denom = max(abs(theirs), 1e-12)
                repro_err.append({'layer': layer, 'field': key,
                                  'mine': mine, 'store': theirs,
                                  'rel_err': abs(mine - theirs) / denom})

        # правило не меняется: двусторонний p < 0.05
        r['layer'] = layer
        r['n_docs'] = len(keys)
        r['decision_buggy'] = ('B_rejected' if (r['alpha_p_buggy'] is not None
                                                and r['alpha_p_buggy'] >= ALPHA_LEVEL)
                               else 'B_accepted')
        r['decision_hc3'] = ('B_rejected' if (r['alpha_p_hc3'] is not None
                                              and r['alpha_p_hc3'] >= ALPHA_LEVEL)
                             else 'B_accepted')
        r['decision_perm_hc3'] = ('B_rejected'
                                  if r['perm_p_honest_hc3'] >= ALPHA_LEVEL
                                  else 'B_accepted')
        out_layers[str(layer)] = r

    worst = max((abs(e['rel_err']) for e in repro_err if 'rel_err' in e),
                default=0.0)
    changed = [l for l, r in out_layers.items()
               if r['decision_buggy'] != r['decision_hc3']]
    both_perm = [l for l, r in out_layers.items()
                 if r['decision_hc3'] == r['decision_perm_hc3']]

    res = {
        'machine': MACHINE,
        'machine_subrevision': MACHINE_SUBREV,
        'experiment': ('E-32r.2/r.3/r.3.1 — disk-only пересчёт HC3-SE для '
                       'pooled OLS alpha (регрессор |dp|, отклик nb)'),
        'store': {'checkpoint': os.path.relpath(CKPT, ROOT),
                  'layer_profile': os.path.relpath(STORE, ROOT),
                  'repassthrough': False,
                  'note': ('store не перепускается: входы регрессии лежат в '
                           'checkpoint (dp_v/nb_v по документам), слой/порядок '
                           'ключей воспроизводит _pooled_ols из прода')},
        'rule': {'model': 'nb ~ alpha*|dp| + beta (OLS)',
                 'se_variants': ['buggy_scalar_diag*sum(w)', 'hc3_sandwich',
                                 'hc1_sandwich_n/(n-2)', 'classical'],
                 'decision_rule': 'two-sided p < 0.05 => (B) accepted',
                 'alpha_level': ALPHA_LEVEL, 'n_perm': N_PERM,
                 'perm_rng_seed': 99},
        'reproduction_of_store': {
            'max_rel_err': worst,
            'ok': bool(worst < 1e-6),
            'per_field': repro_err},
        'layers': out_layers,
        'verdict': {
            'layers_where_decision_flips': changed,
            'n_flipped': len(changed),
            'hc3_matches_perm': both_perm,
            'n_hc3_matches_perm': len(both_perm),
            'summary': None,  # заполняется ниже
        },
    }

    sig_bug = [l for l, r in out_layers.items() if r['decision_buggy'] == 'B_accepted']
    sig_hc3 = [l for l, r in out_layers.items() if r['decision_hc3'] == 'B_accepted']
    if len(sig_hc3) == len(out_layers) and len(sig_bug) == 0:
        summary = ('honest HC3 делает alpha значимым на всех слоях '
                   '(%d/%d) там, где багованная формула не давала значимости '
                   'ни на одном (0/%d): опубликованное «(B) отклонён» — '
                   'артефакт SE, (B) принимается. buggy-вердикт публикуется '
                   'как опровергнутый.' % (len(sig_hc3), len(out_layers),
                                           len(out_layers)))
    elif len(sig_hc3) > len(sig_bug):
        summary = ('honest HC3 делает alpha значимым на %d слоях из %d '
                   'против %d у багованной формулы: (B) не отклоняется на '
                   'всех слоях, вердикт переоткрывается.'
                   % (len(sig_hc3), len(out_layers), len(sig_bug)))
    else:
        summary = ('honest HC3 не меняет картину: alpha значим на %d слоях '
                   'из %d (у багованной %d).' % (len(sig_hc3), len(out_layers),
                                                 len(sig_bug)))
    res['verdict']['summary'] = summary

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({'repro_max_rel_err': worst,
                      'repro_ok': res['reproduction_of_store']['ok'],
                      'verdict': res['verdict']}, ensure_ascii=False,
                     indent=1))
    print('\nlayer  n   alpha   se_bug   se_hc3   ratio  p_bug   p_hc3   '
          'perm_p  dec_bug->dec_hc3')
    for l, r in out_layers.items():
        print('%5s %4d %8.3f %8.3f %8.4f %7.1f %7.3f %8.2e %8.4f  %s -> %s'
              % (l, r['n'], r['alpha'], r['alpha_se_buggy'],
                 r['alpha_se_hc3'], r['se_ratio_honest_over_buggy'],
                 r['alpha_p_buggy'], r['alpha_p_hc3'],
                 r['perm_p_honest_hc3'], r['decision_buggy'],
                 r['decision_hc3']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
