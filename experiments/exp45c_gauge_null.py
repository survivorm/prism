#!/usr/bin/env python
"""E-45c: нулевой контроль конвейера ущерба на ТОЧНОЙ калибровке считывания.

Зачем. `--selftest` в E-45 сверяет разреженный путь ущерба с плотным — но если обе реализации
ошибаются одинаково (одна и та же путаница знаков или оснований), самопроверка этого не видит.
Здесь правка, чей ответ известен АНАЛИТИЧЕСКИ и равен нулю.

Считывание имеет точную свободу W → W + 1·cᵗ (добавить один и тот же вектор c ко всем строкам):
логиты получают слагаемое c·ĥ, ОДИНАКОВОЕ для всех строк словаря, softmax по строке инвариантен ⇒
любое поведение (KL, TV, top-1, PPL) не меняется РОВНО, для любого входа, без линейизации.
Якобиан в репозитории — якобиан лог-вероятностей,

    J_t = cs_t · (M − u_t (Mᵗ u_t)ᵗ),   M = W ∘ g,   u_t = p_t (ренормированная, Σ_k p_k = 1),

поэтому при M → M + 1cᵗ добавка равна cs_t·(1cᵗ − (u_t·1)cᵗ) = cs_t·(1 − 1)cᵗ = 0: не меняются
ни J, ни лист, ни κ. Ядро отображения δW ↦ δJ — это в точности калибровка, так что «невидимое
разрушение листа» калибровкой невозможно: лист от неё не двигается вообще (в отличие от «сырого»
якобиана логитов, где та же добавка оставляет нетронутым только softmax). Это алгебра, а не
эмпирика; скрипт проверяет, что численные реализации ведут себя ровно так.

Что обязано сойтись нулём (иначе конвейер E-45 врёт):
  1. дрейф ‖ΔJ‖/‖J‖           → 0     (проверка J_of)
  2. κ(r=16) до/после         → равно (проверка links κ↔J)
  3. KL, TV, смена top-1      → 0     (проверка dense_metrics — тем же кодом считается ущерб
                                        всех правок E-45)
Контроль НЕ-калибровкой того же масштаба (одна строка W): ответ обязан быть большим, иначе нули
свыше получены молчанием метрик, а не инвариантностью.

Запуск:  cd experiments && ../.venv/bin/python exp45c_gauge_null.py   (~5 минут CPU)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp28_leaf_structure as E28                       # noqa: E402
import exp45_damage_vs_structure as E45                   # noqa: E402
from _e38_cache import jcache_path                        # noqa: E402

S, R, D = E45.S, E45.R, 1024
N_SPEC_DOCS = 8                 # κ считается по 8 документам (сравнение корректно: база та же)
METRIC_STRIDE = 4               # метрики ущерба по 1/4 позиций: полный [m,V] matmul на 4096
                                # позициях стоит ~100 с правка, нулевому контролю хватает 1024
# Допуски НЕ нулевые: аналитический ответ 0, но остаток даёт плавающая точка — в J вклад
# калибровки сокращается как (1 − Σ_k p_k)·c, а P128 в кэше нормирована с ошибкой ~1e-7.
TOL_DRIFT = 1e-5
TOL_KL = 1e-10
TOL_TV = 1e-5
TOL_KAPPA_REL = 1e-6
MIN_CONTROL_KL = 1e-4
MIN_CONTROL_DRIFT = 1e-3
OUT = os.path.join(_HERE, "..", "results", "exp45c_gauge_null.json")


def kappa_part(J: np.ndarray) -> float:
    """κ(r=16) по первым N_SPEC_DOCS документам (тот же E28.document_profile, что и в E-45)."""
    ks = []
    for i in range(0, N_SPEC_DOCS * S, S):
        _d1, _dm, k2, _nd = E28.document_profile(
            torch.from_numpy(J[i:i + S].astype(np.float32)), [1, R, 32], r_main=R)
        k2 = np.asarray(k2, dtype=np.float64)
        ks.append(k2[~np.isnan(k2)])
    return float(np.median(np.concatenate(ks)))


def main():
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 8)
    E45.log("loading jcache L23 ...")
    ck = torch.load(jcache_path(E45.L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy") else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    Ha = torch.from_numpy(Hst).float()
    del ck
    E45.log("loading model (fp32) ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        E45.MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    torch.set_grad_enabled(False)
    W = model.lm_head.weight.detach().float().clone()
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    del model

    J_of = E45.make_j_fn(Hst, g_np)
    J_base = J_of(W[U128].numpy().astype(np.float64))
    nrm_min = float(np.linalg.norm(J_base.reshape(len(J_base), -1), axis=1).min())
    kap_base = kappa_part(J_base)
    sigW = float(W.std())
    E45.log(f"база: κ(r=16, {N_SPEC_DOCS} доков) = {kap_base:.8f}, мин ‖J_t‖ = {nrm_min:.3e}, "
            f"σ_W = {sigW:.3e}")

    # насколько P128 в кэше вообще нормирована: это масштаб ожидаемого численного остатка
    rng = np.random.default_rng(0)
    rows = []
    specs = [("gauge_rand_a0.05", "rand", 0.05), ("gauge_rand_a0.15", "rand", 0.15),
             ("gauge_rand_a0.50", "rand", 0.50),
             ("gauge_hmean_dir", "hmean", 0.15), ("gauge_first_row", "first", 0.15)]
    for name, kind, alpha in specs:
        if kind == "rand":
            c = torch.from_numpy(rng.standard_normal(D).astype(np.float32))
        elif kind == "hmean":
            c = torch.from_numpy((Hst.mean(0) * g_np).astype(np.float32))
        else:
            c = W[0].clone()
        c = c / c.norm().clamp_min(1e-20) * (alpha * sigW * np.sqrt(D))
        W_e = W + c[None, :] * torch.ones(W.shape[0], 1)      # 1_V ⊗ cᵗ
        J1 = J_of(W_e[U128].numpy().astype(np.float64))
        drift = float(np.linalg.norm((J1 - J_base).reshape(len(J1), -1), axis=1).max() / nrm_min)
        m = E45.dense_metrics(Ha[::METRIC_STRIDE], W_e, W)
        kap = kappa_part(J1)
        row = dict(name=name, alpha=alpha, c_norm=float(c.norm()), drift_J_max=drift,
                   kappa=kap, kappa_rel_dev=abs(kap - kap_base) / max(kap_base, 1e-30),
                   kl_be_max=float(m["kl_be"].max()), kl_be_mean=float(m["kl_be"].mean()),
                   tv_max=float(m["tv"].max()), flip=float(m["flip"].mean()))
        rows.append(row)
        E45.log(f"  {name:>18s} ‖c‖={row['c_norm']:.3e}: drift_J = {drift:.2e}  "
                f"κ = {kap:.8f} (отн. {row['kappa_rel_dev']:.2e})  "
                f"KL_max = {row['kl_be_max']:.2e}  TV_max = {row['tv_max']:.2e}  "
                f"top1→ = {row['flip']:.4f}")
        del W_e, J1, m

    # контроль: правка одного ряда того же масштаба — обязана быть заметной
    kcol = int(U128[7])
    W_c = W.clone()
    W_c[kcol] = W_c[kcol] + 0.15 * sigW * torch.from_numpy(rng.standard_normal(D).astype(np.float32))
    Jc = J_of(W_c[U128].numpy().astype(np.float64))
    d_ctl = float(np.linalg.norm((Jc - J_base).reshape(len(Jc), -1), axis=1).max() / nrm_min)
    m_c = E45.dense_metrics(Ha[::METRIC_STRIDE], W_c, W)
    ctl = dict(name=f"control_1row_U128idx7_drift{d_ctl:.3f}", drift_J_max=d_ctl,
               kappa=kappa_part(Jc), kl_be_max=float(m_c["kl_be"].max()),
               tv_max=float(m_c["tv"].max()), flip=float(m_c["flip"].mean()))
    E45.log(f"  {'control_1row':>18s}: drift_J = {ctl['drift_J_max']:.2e}  "
            f"κ = {ctl['kappa']:.8f}  KL_max = {ctl['kl_be_max']:.2e}  "
            f"top1→ = {ctl['flip']:.4f}")

    ok = (max(r["drift_J_max"] for r in rows) < TOL_DRIFT
          and max(r["kl_be_max"] for r in rows) < TOL_KL
          and max(r["tv_max"] for r in rows) < TOL_TV
          and max(r["kappa_rel_dev"] for r in rows) < TOL_KAPPA_REL
          and ctl["drift_J_max"] > MIN_CONTROL_DRIFT and ctl["kl_be_max"] > MIN_CONTROL_KL)
    res = dict(experiment="E-45c: gauge-null контроль конвейера ущерба", L=E45.L, r=R,
               kappa_base=kap_base, rows=rows, control=ctl,
               metric_stride=METRIC_STRIDE,
               tol=dict(drift=TOL_DRIFT, kl=TOL_KL, tv=TOL_TV, kappa_rel=TOL_KAPPA_REL),
               passed=bool(ok),
               wall_sec=round(time.time() - t0, 1))
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    E45.log("\nВЕРДИКТ: " + ("калибровочный нуль сошёлся (J, κ, KL, TV, top-1 неподвижны в пределах "
                             f"{TOL_DRIFT}/{TOL_KL}), контроль заметен ⇒ dense_metrics и J_of "
                             "корректны на правке с известным ответом"
                             if ok else
                             "КАЛИБРОВОЧНЫЙ НУЛЬ НЕ СОШЁЛСЯ — искать ошибку в J_of/dense_metrics, "
                             "числа E-45 под подозрением"))
    E45.log(f"  → {OUT}, wall {res['wall_sec']} с")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
