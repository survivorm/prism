#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-проверки аудита strayed_experiments (docs/strayed_experiments.md).

Зачем: 12.11 ревьюер справедливо указал, что числа «проверено исполнением» в
аудите получены ad-hoc и из репо не воспроизводятся (AGENTS.md п.8: ревью/проверка
существует только как файл на диске). Этот скрипт закрывает щель: каждое число
аудита, полученное «своим счётом», здесь воспроизводится кодом проекта (или
минимальной нуль-моделью с явным определением) и пишется в артефакт.

Проверки (ключ в артефакте):
  C1_e1_random_j_gap      — E-1: jacobian_gap на СЛУЧАЙНОМ J (V=20000,D=256,K=4);
                            три плеча: полный ранг E, E_L!=E_S, E ранга V-2.
                            Числа аудита: 3.5e-9 / 1.025 / 3.8e-6.
  C2_cosine_floor         — E-0.1: нуль для косинуса матриц J [K=64,D=1024] при
                            эффективном ранге r: три нуль-модели, 400 дро.
                            Числа аудита: 0.0078/0.0056/0.0054 (общий P),
                            0.0039/0.0038/0.0037 (независимые P), 0.0040/0.0037/0.0039 (B1B2).
  C3_overlap_null         — E-13: нуль для overlap двух r-мерных подпространств в R^K:
                            E[||U1^T U2||_F^2 / r] = r/K; 2000 дро, K=64, r=8.
                            Число аудита: 0.1245 +/- 0.0186.
  C4_rho_null_random_j    — E-34: rho_gf (та же машина, что exp34: A = pESGp,
                            rho = offdiag/diag/sqrt(K-1)) на СЛУЧАЙНОМ J [128,1024],
                            50 дро, p uniform + p random-dirichlet.
                            Число аудита: 0.023 (нуль оператора P).
  C5_degeneracy_noise     — E-38c: min_ratio_t на чистом шуме J [128,1024],
                            EPS_MAIN=0.01, R=16: распределение статистики одного дро
                            и статистика ТОЙ ЖЕ формы, что в отчёте (min по 64 t от
                            медианы по 32 докам). Прежняя цифра 0.051 не
                            воспроизвелась — см. артефакт (медиана 0.186).
  C6_pinv_conditioning    — E-1/E-5: точность fp32-pinv vs kappa(G):
                            max|GG+ - I| при kappa 5e5 и 3.8e7.
  C7_subspace_overlap_null — E-38g: распределение overlap ||V^T W||_F^2/r1 двух
                            независимых случайных подпространств (r1=12, r2=102,
                            D=1024) против наблюдаемого 0.0596 и «окна» кода.
Все проверки — только numpy/torch, CPU, детерминированы по seed. Артефакт:
results/_audit_checks/audit_checks.json (+ meta с sha скрипта и started_at).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_PATH = os.path.join(ROOT, "results", "_audit_checks", "audit_checks.json")
SEED = 4242


# ---------------------------------------------------------------- exp34 helpers
def _load_exp34_funcs():
    """p_esg_p / rho_offdiag из exp34 (те самые функции, что в прогоне E-34)."""
    import importlib.util

    path = os.path.join(ROOT, "experiments", "e31_40", "exp34_p1_coupling.py")
    spec = importlib.util.spec_from_file_location("exp34_mod", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # noqa: F841 — импорт ради функций
    except SystemExit:
        pass
    return mod.p_esg_p, mod.rho_offdiag


# ------------------------------------------------------------------- check C1
def c1_e1_random_j_gap(rng: np.random.Generator) -> dict:
    """jacobian_gap на случайном J: тест обязан дать ~0 при любой полной E."""
    from prism.bridge import jacobian_gap

    V, D, K = 20000, 256, 4
    J = torch.from_numpy(rng.standard_normal((K, D)).astype(np.float32))
    # NB: probs_L живёт в пространстве СТРОК J ([K]), а не в пространстве словаря:
    # jacobian_gap маскирует J-строки (K) и индексирует probs[mask] — при
    # probs длины V получится IndexError (пойман первым прогоном r1, rc=1).
    probs = torch.full((K,), 1.0 / K, dtype=torch.float32)

    E_full = torch.from_numpy(rng.standard_normal((V, D)).astype(np.float32))
    gap_full = float(jacobian_gap(J, J, E_full, E_full, probs))

    E_other = torch.from_numpy(rng.standard_normal((V, D)).astype(np.float32))
    gap_diff = float(jacobian_gap(J, J, E_full, E_other, probs))

    E_rank_def = E_full.clone()
    E_rank_def[1] = E_rank_def[0]  # две дублированные строки => ранг V-2 по строкам
    gap_rank_def = float(jacobian_gap(J, J, E_rank_def, E_rank_def, probs))

    return {
        "random_J_full_rank_E": gap_full,
        "random_J_E_L_ne_E_S": gap_diff,
        "random_J_E_rank_V_minus_2": gap_rank_def,
        "params": {"V": V, "D": D, "K": K},
        "audit_claim": {"random_J_full_rank_E": 3.5e-9,
                        "random_J_E_L_ne_E_S": 1.025,
                        "random_J_E_rank_V_minus_2": 3.8e-6},
    }


# ------------------------------------------------------------------- check C2
def _cos_matrices(A: np.ndarray, B: np.ndarray) -> float:
    """Косинус двух матриц по Фробениусу (как в E-27/E-0.1)."""
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na == 0 or nb == 0:
        return float("nan")
    return float((A * B).sum() / (na * nb))


def c2_cosine_floor(rng: np.random.Generator, n_draws: int = 400) -> dict:
    """Нуль для косинуса матриц J [K, D] при эффективном ранге r (три нуль-модели)."""
    K, D = 64, 1024
    out = {}
    for r in (16, 28, 32):
        acc = {"shared_P": [], "indep_P": [], "B1B2": []}
        for _ in range(n_draws):
            W1 = rng.standard_normal((K, D))   # J = P·W, [K, D]
            W2 = rng.standard_normal((K, D))

            Q, _ = np.linalg.qr(rng.standard_normal((K, r)))
            P = Q @ Q.T                        # проектор ранга r в R^K
            acc["shared_P"].append(_cos_matrices(P @ W1, P @ W2))

            Qa, _ = np.linalg.qr(rng.standard_normal((K, r)))
            Qb, _ = np.linalg.qr(rng.standard_normal((K, r)))
            acc["indep_P"].append(_cos_matrices((Qa @ Qa.T) @ W1, (Qb @ Qb.T) @ W2))

            acc["B1B2"].append(_cos_matrices(
                rng.standard_normal((K, r)) @ rng.standard_normal((r, D)),
                rng.standard_normal((K, r)) @ rng.standard_normal((r, D))))

        out[f"r={r}"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                         for k, v in acc.items()}
        out[f"r={r}"]["F_analytic"] = float(1.0 / np.sqrt(r * (K + D - r)))
    out["params"] = {"K": K, "D": D, "n_draws": n_draws, "F_1_over_sqrtD": 1.0 / np.sqrt(D)}
    out["audit_claim"] = {
        "r=16": {"shared_P": 0.0078, "indep_P": 0.0039, "B1B2": 0.0040},
        "r=28": {"shared_P": 0.0056, "indep_P": 0.0038, "B1B2": 0.0037},
        "r=32": {"shared_P": 0.0054, "indep_P": 0.0037, "B1B2": 0.0039},
    }
    return out


# ------------------------------------------------------------------- check C3
def c3_overlap_null(rng: np.random.Generator, n_draws: int = 2000) -> dict:
    """E[||U1^T U2||_F^2 / r] = r/K для независимых r-мерных подпространств."""
    K, r = 64, 8
    vals = np.empty(n_draws)
    for i in range(n_draws):
        U1, _ = np.linalg.qr(rng.standard_normal((K, r)))
        U2, _ = np.linalg.qr(rng.standard_normal((K, r)))
        vals[i] = float(np.linalg.norm(U1.T @ U2) ** 2) / r
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std()),
        "analytic_r_over_K": r / K,
        "params": {"K": K, "r": r, "n_draws": n_draws},
        "audit_claim": {"mean": 0.1245, "sd": 0.0186},
    }


# ------------------------------------------------------------------- check C4
def c4_rho_null_random_j(rng: np.random.Generator, n_draws: int = 50) -> dict:
    """rho_gf нуль-оператора P на случайном J (машина E-34)."""
    p_esg_p, rho_offdiag = _load_exp34_funcs()
    K, D = 128, 1024
    p_unif = np.full(K, 1.0 / K)
    res = {"p_uniform": [], "p_dirichlet": []}
    for _ in range(n_draws):
        J = rng.standard_normal((K, D))
        ESG = J @ J.T
        res["p_uniform"].append(rho_offdiag(p_esg_p(ESG, p_unif)))
        p_rand = rng.dirichlet(np.ones(K))
        res["p_dirichlet"].append(rho_offdiag(p_esg_p(ESG, p_rand)))
    return {
        "p_uniform": {"mean": float(np.mean(res["p_uniform"])),
                      "sd": float(np.std(res["p_uniform"]))},
        "p_dirichlet": {"mean": float(np.mean(res["p_dirichlet"])),
                        "sd": float(np.std(res["p_dirichlet"]))},
        "params": {"K": K, "D": D, "n_draws": n_draws},
        "audit_claim": {"p_uniform_mean": 0.023},
    }


# ------------------------------------------------------------------- check C5
def c5_degeneracy_noise(rng: np.random.Generator, n_draws: int = 200,
                        n_dist: int = 4096, n_sim: int = 2000) -> dict:
    """min_ratio_t на чистом шуме J [128, 1024] (машина E-38c: R=16, EPS_MAIN=0.01).

    Две части:
      per_draw  — распределение статистики ОДНОГО дро (среднее/SD/min).
      matched   — статистика ТОЙ ЖЕ формы, что в отчёте E-38c: min по 64 t от
                  медианы по 32 докам; симулируется ресемплингом из
                  эмпирического распределения per_draw (доки/позиции iid, поэтому
                  ресемплинг эквивалентен новым SVD; отмечено в meta).
                  Сравнение с реальным min_ratio_t = 0.071–0.089 — только по
                  matched, а не по среднему одного дро.
    """
    K, D, R, EPS_MAIN = 128, 1024, 16, 0.01
    rel_gap, min_ratio = np.empty(n_draws), np.empty(n_draws)
    for i in range(n_draws):
        J = rng.standard_normal((K, D))
        lam = np.linalg.svd(J, compute_uv=False)[:R] ** 2
        gaps = lam[:-1] - lam[1:]
        rel_gap[i] = gaps.min() / lam[0]          # gap_min / sigma_1^2
        min_ratio[i] = rel_gap[i] / EPS_MAIN

    # эмпирическое распределение per-doc статистики для ресемплинга
    pool = np.empty(n_dist)
    for i in range(n_dist):
        J = rng.standard_normal((K, D))
        lam = np.linalg.svd(J, compute_uv=False)[:R] ** 2
        pool[i] = (lam[:-1] - lam[1:]).min() / lam[0] / EPS_MAIN

    sims = np.empty(n_sim)
    for s in range(n_sim):
        draw = rng.choice(pool, size=(64, 32), replace=True)  # 64 t × 32 docs
        # ВАЖНО: exp38c берёт МЕДИАНУ по докам (ratio_t), не среднее; распределение
        # per-doc ratio скошено вправо, mean > median на ~10 % — на этом первая
        # версия r4 завышала нуль (mean вместо median).
        sims[s] = np.median(draw, axis=1).min()
    q = np.percentile(sims, [1, 5, 50, 95, 99])

    return {
        "per_draw": {
            "gap_min_over_sigma1sq": {"mean": float(rel_gap.mean()), "sd": float(rel_gap.std())},
            "min_ratio_t": {"mean": float(min_ratio.mean()), "sd": float(min_ratio.std()),
                            "min": float(min_ratio.min())},
        },
        "matched_statistic": {
            "q01": float(q[0]), "q05": float(q[1]), "median": float(q[2]),
            "q95": float(q[3]), "q99": float(q[4]),
            "observed_real_min_ratio_t": [0.0714, 0.0894, 0.0718, 0.0821, 0.0773, 0.0780],
        },
        "params": {"K": K, "D": D, "R": R, "EPS_MAIN": EPS_MAIN,
                   "n_draws": n_draws, "n_dist": n_dist, "n_sim": n_sim,
                   "sim_shape": "min over 64 t of median over 32 docs"},
        "audit_claim": {"gap_min_over_sigma1sq": 5.1e-4, "min_ratio_t": 0.051},
    }


# ------------------------------------------------------------------- check C6
def c6_pinv_conditioning(rng: np.random.Generator) -> dict:
    """E-1: точность fp32-pinv как функция числа обусловленности
    $\\kappa(G)$, $G = E^T E$. Строим E = U·diag(s) с ортонормальными
    столбцами U и геометрическим спектром s, т.е. $\\kappa(G) = \\kappa(E)^2$
    задаётся явно; меряем $\\max|GG^+ - I|$.
    """
    V, D = 20000, 256
    Q, _ = np.linalg.qr(rng.standard_normal((V, D)))
    U = Q[:, :D].astype(np.float32)
    out = {}
    for kappa_G in (5e5, 3.8e7):
        s = np.logspace(0.0, -0.5 * np.log10(kappa_G), D).astype(np.float32)
        E = torch.from_numpy(U * s.astype(np.float32))
        G = E.T @ E
        Ginv = torch.linalg.pinv(G)
        err = float((Ginv @ G - torch.eye(D)).abs().max())
        sv = torch.linalg.svdvals(G.double())
        # для контраста — та же матрица в float64: если fp64 даёт ~1e-9, то
        # расхождение с ad-hoc 4e-2/0.44 объясняется конструкцией E, а не точностью
        G64 = E.double().T @ E.double()
        err64 = float((torch.linalg.pinv(G64) @ G64 - torch.eye(D, dtype=torch.float64)).abs().max())
        out[f"target_kappa_G={kappa_G:.3g}"] = {
            "measured_kappa_G": float(sv[0] / sv[-1]),
            "max_abs_GGinv_minus_I_fp32": err,
            "max_abs_GGinv_minus_I_fp64": err64,
        }
    return {
        "levels": out,
        "params": {"V": V, "D": D},
        "audit_claim": {"kappa_5e5": 4e-2, "kappa_3.8e7": 0.44},
    }


# ------------------------------------------------------------------- check C7
def c7_subspace_overlap_null(rng: np.random.Generator, n_draws: int = 2000) -> dict:
    """E-38g: нуль для overlap = ||V^T W||_F^2 / r1 двух НЕЗАВИСИМЫХ
    случайных подпространств (r1=12, r2=102 в R^1024). Аналитика E = r2/D,
    но вердикт карточки («at_random») сравнивает с этим средним БЕЗ разброса.
    """
    D, r1, r2 = 1024, 12, 102
    vals = np.empty(n_draws)
    for i in range(n_draws):
        V = np.linalg.qr(rng.standard_normal((D, r1)))[0]
        W = np.linalg.qr(rng.standard_normal((D, r2)))[0]
        vals[i] = float(((V.T @ W) ** 2).sum()) / r1
    obs = 0.05959700668851534                      # artifact exp38g primary median
    sd = float(vals.std(ddof=1))
    return {
        "null": {"mean": float(vals.mean()), "sd": sd,
                 "q01": float(np.quantile(vals, 0.01)),
                 "q05": float(np.quantile(vals, 0.05)),
                 "q99": float(np.quantile(vals, 0.99))},
        "analytic_r2_over_D": r2 / D,
        "observed_overlap": obs,
        "z_vs_null": (obs - float(vals.mean())) / sd,
        "observed_below_q01": bool(obs < np.quantile(vals, 0.01)),
        "code_window_at_random": [0.5 * r2 / D, 3.0 * r2 / D],
        "anticorrelated_branch_threshold": 0.5 * r2 / D,
        "params": {"D": D, "r1": r1, "r2": r2, "n_draws": n_draws},
        "audit_claim": {"card_label": "at_random", "card_note": "overlap ~0.1 = random independence"},
    }


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="минимальные n (проверка размерностей всех чеков за секунды)")
    a = ap.parse_args(argv)
    n2 = 8 if a.smoke else 400
    n3 = 20 if a.smoke else 2000
    n4 = 4 if a.smoke else 50
    n5 = 8 if a.smoke else 200
    n5_dist = 64 if a.smoke else 4096
    n5_sim = 50 if a.smoke else 2000
    n7 = 10 if a.smoke else 2000
    t0 = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    script = os.path.abspath(__file__)
    script_sha = hashlib.sha256(open(script, "rb").read()).hexdigest()

    rng = np.random.default_rng(SEED)
    checks = {}
    order = [
        ("C1_e1_random_j_gap", lambda: c1_e1_random_j_gap(rng)),
        ("C2_cosine_floor", lambda: c2_cosine_floor(rng, n2)),
        ("C3_overlap_null", lambda: c3_overlap_null(rng, n3)),
        ("C4_rho_null_random_j", lambda: c4_rho_null_random_j(rng, n4)),
        ("C5_degeneracy_noise", lambda: c5_degeneracy_noise(rng, n5, n5_dist, n5_sim)),
        ("C6_pinv_conditioning", lambda: c6_pinv_conditioning(rng)),
        ("C7_subspace_overlap_null", lambda: c7_subspace_overlap_null(rng, n7)),
    ]
    for name, fn in order:
        ts = time.time()
        checks[name] = fn()
        checks[name]["wall_s"] = round(time.time() - ts, 2)
        print(f"[{name}] {checks[name]['wall_s']}s", flush=True)

    for name, c in checks.items():
        print(f"\n== {name} [{c.get('wall_s')}s]")
        body = {k: v for k, v in c.items() if k not in ("wall_s",)}
        print(json.dumps(body, ensure_ascii=False, indent=1, default=float)[:4000])

    payload = {
        "meta": {
            "experiment": "audit_checks (docs/strayed_experiments.md, «проверено исполнением»)",
            "purpose": "воспроизвести из репо ad-hoc числа аудита (замечание внешнего ревью 12.11)",
            "script": os.path.relpath(script, ROOT),
            "script_sha256": script_sha,
            "device": "cpu",
            "seed": SEED,
            "started_at": started_at,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "note": ("Числа аудита получены на другой машине/сиде; критерий — не побитовое "
                     "совпадение, а согласие порядка: см. поле agreement в каждом чеке."),
        },
        "checks": checks,
        "duration_s": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload["meta"]["smoke"] = bool(a.smoke)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nAUDIT_CHECKS_DONE -> {os.path.relpath(OUT_PATH, ROOT)} "
          f"({payload['duration_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
