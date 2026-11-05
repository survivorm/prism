#!/usr/bin/env python
"""E-45c: нулевой контроль конвейера ущерба на ТОЧНОЙ калибровке считывания.

Зачем. `--selftest` в E-45 сверяет разреженный путь ущерба с плотным — но если обе реализации
ошибаются одинаково (одна и та же путаница знаков или оснований), самопроверка этого не видит.
Здесь правка, чей ответ известен АНАЛИТИЧЕСКИ.

Считывание имеет точную свободу W → W + 1·cᵗ (добавить один и тот же вектор c ко всем строкам
словаря): логиты получают слагаемое c·ĥ, ОДИНАКОВОЕ для всех строк, softmax по строке словаря
инвариантен ⇒ любое поведение (KL, TV, top-1, PPL) не меняется РОВНО, для любого входа, без
линеаризации. Это алгебра; скрипт проверяет, что численные реализации ведут себя так же.

Про Якобиан — важное уточнение, найденное этим же скриптом (первый запуск 2026-11-05 дал на
gauge-правке дрейф J 6.8 %, а не нуль). J в репозитории (E45.make_j_fn, тот же объект, на котором
κ всего семейства E-28/E-33r/E-39/E-45) —

    J_t = cs_t · (M − (M ḣ_t) ḣ_tᵗ),   M = W_U ∘ g,   ḣ_t = h_t/‖h_t‖,

то есть якобиан ЛОГИТОВ по направлению RMSNorm (спроецировано только радиальное направление ḣ,
никого не спроецировано по оси словаря). Ранг-1 добавка от калибровки

    ΔJ_t = cs_t · 1_K ⊗ (d − (d·ḣ_t) ḣ_t),   d = g ∘ c,

не обнуляется: ноль дал бы левый множитель (I − 1pᵗ), то есть якобиан лог-вероятностей. Значит
корректные утверждения нуль-контроля такие, и они проверяются здесь по отдельности:
  1. поведение (KL, TV, top-1) → нуль с подом float32 (под измеряется отдельно: ту же разность
     логитов считаем в fp64 — если там ~1e-15, значит остаток есть округление, а не ошибка);
  2. J ЛОГИТОВ от калибровки ДВИГАЕТСЯ — и обязан двигаться ровно предсказанным ранг-1 членом
     (сверяем остаток ‖ΔJ_факт − ΔJ_предсказ‖/‖ΔJ_факт‖); это не баг, а область применимости:
     κ(логит-J) — величина калибровочно-НЕинвариантная;
  3. левая проекция Q = I − 1pᵗ (p = P128, renormalized, Σ_k p_k = 1) калибровку убивает:
     дрейф QJ → 0 и κ(QJ) не двигается. То есть калибровочно-инвариантна κ именно gauged-free
     проекции — того объекта, который в статье и называется снятым по калибровке.
Контроль НЕ-калибровкой того же масштаба (одна строка W): ответ обязан быть большим, иначе нули
получены молчанием метрик, а не инвариантностью.

Запуск:  cd experiments && ../.venv/bin/python exp45c_gauge_null.py   (~10 минут CPU)
Память. Всё, что смотрит в J, идёт чанками (см. CHUNK): первый запуск держал базовый якобиан,
якобиан правки и их разность целиком — 13 ГБ, и процесс умер без traceback.
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
# Допуски НЕ нулевые: аналитический ответ 0, но остаток даёт плавающая точка — в QJ остаток
# идёт по ошибке нормировки P128 (~1e-7), в поведении — по округлению fp32-логитов.
TOL_QJ_DRIFT = 1e-4             # дрейф QJ (gauged-free проекция) в абсолютных единицах
TOL_QJ_OVER_J = 1e-3            # …и относительно дрейфа сырого J: подавление минимум в 1000×
TOL_KL = 1e-4                   # KL/TV на fp32-подлоге: под измерен ~2e-06
TOL_TV = 1e-4
TOL_KAPPA_REL = 1e-4            # κ(QJ): пол не машинный, а float32 — document_profile внутри
                                # кастует J в fp32 (замер: 2.7e-05 при подавлении в 3e7 раз),
                                # и это на два порядка меньше самой мелкой информативной
                                # разницы κ в таблице E-45 (0.16 %) и её полосы (0.58 %)
TOL_PRED_REL = 1e-8             # остаток против предсказанного ранг-1 члена
TOL_FP64_KL = 1e-10             # KL, пересчитанный в fp64: доказательство, что под = округление
MIN_CONTROL_KL = 1e-4
MIN_CONTROL_DRIFT = 1e-3
MIN_GAUGE_J_DRIFT = 1e-6        # сырой J обязан ДВИГАТЬСЯ, иначе проверка п.2 пустая
OUT = os.path.join(_HERE, "..", "results", "exp45c_gauge_null.json")

# Полный стек [4096,128,1024] fp64 = 4.3 ГБ; базовый якобиан + правленный + их разность —
# 13 ГБ, на этом пике запуск умер без traceback (2026-11-05, swap забит). Поэтому всё, что
# смотрит на J, идёт чанками: держим не больше двух блоков по CHUNK позиций.
CHUNK = 512                                  # [CHUNK,K,D] fp64 = 0.54 ГБ


def j_chunk(Hst, g, rows, c0, c1) -> np.ndarray:
    return E45.make_j_fn(Hst[c0:c1], g)(rows)


def left_proj(Jc: np.ndarray, Pc: np.ndarray) -> np.ndarray:
    """QJ при Q = I − 1pᵗ: (QJ)_k = J_k − Σ_j p_j J_j — одно и то же число вычитается из всех
    строк словаря. Q1 = 0 ⇒ калибровочную добавку 1⊗v убивает точно."""
    w = np.einsum("tkd,tk->td", Jc, Pc)
    return Jc - w[:, None, :]


def gauge_dJ_pred(Hst, g, c, c0, c1) -> np.ndarray:
    """ΔJ_t = cs_t·1_K ⊗ (d − (d·ḣ_t)ḣ_t), d = g∘c — точная добавка от W → W + 1cᵗ."""
    Hs = Hst[c0:c1]
    d = g * c
    u = Hs / np.linalg.norm(Hs, axis=1, keepdims=True)
    cs = E45.SQD / np.linalg.norm(Hs, axis=1)
    v = d[None, :] - np.einsum("td,d->t", u, d)[:, None] * u          # [n,D]
    return cs[:, None, None] * v[:, None, :]                           # broadcast по K


def drift_of(Hst, g, Pst, base_rows, rows, proj: bool):
    """(max, median) отношения ‖ΔJ_t‖/‖J_t‖ по позициям; proj=True — для QJ вместо J."""
    r, pred_res_num, pred_res_den = [], 0.0, 0.0
    for c0 in range(0, len(Hst), CHUNK):
        c1 = min(c0 + CHUNK, len(Hst))
        J0 = j_chunk(Hst, g, base_rows, c0, c1)
        J1 = j_chunk(Hst, g, rows, c0, c1)
        if proj:
            P = Pst[c0:c1]
            D0 = left_proj(J0, P)
            D1 = left_proj(J1, P)
        else:
            D0, D1 = J0, J1
        n0 = np.linalg.norm(D0.reshape(c1 - c0, -1), axis=1)
        r.append(np.linalg.norm((D1 - D0).reshape(c1 - c0, -1), axis=1) / n0)
        del J0, J1, D0, D1, n0
    r = np.concatenate(r)
    return float(r.max()), float(np.median(r))


def kappa_chunk(Hst, g, rows, proj: bool, Pst=None) -> float:
    """κ(r=16) поблочно по документу (блок [128,K,D] fp64 = 134 МБ); proj=True — κ от QJ."""
    ks = []
    for d0 in range(N_SPEC_DOCS):
        c0, c1 = d0 * S, (d0 + 1) * S
        J = j_chunk(Hst, g, rows, c0, c1)
        if proj:
            J = left_proj(J, Pst[c0:c1])
        _d1, _dm, k2, _nd = E28.document_profile(
            torch.from_numpy(J.astype(np.float32)), [1, R, 32], r_main=R)
        k2 = np.asarray(k2, dtype=np.float64)
        ks.append(k2[~np.isnan(k2)])
        del J
    return float(np.median(np.concatenate(ks)))


def norm_rows(Hst, g, rows) -> np.ndarray:
    """‖J_t‖_F по позициям, чанками."""
    out = np.empty(len(Hst))
    for c0 in range(0, len(Hst), CHUNK):
        c1 = min(c0 + CHUNK, len(Hst))
        Jc = j_chunk(Hst, g, rows, c0, c1)
        out[c0:c1] = np.linalg.norm(Jc.reshape(c1 - c0, -1), axis=1)
        del Jc
    return out


def main():
    t0 = time.time()
    torch.set_num_threads(len(os.sched_getaffinity(0)))   # не os.cpu_count(): на этой коробке
    # он возвращает 64 при 12 доступных ядрах, и OpenMP-потоки задыхаются на барьерах
    E45.log("loading jcache L23 ...")
    ck = torch.load(jcache_path(E45.L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy") else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    Pst = np.stack([ck["P128"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    Ha = torch.from_numpy(Hst).float()
    del ck
    dev_p = np.abs(Pst.sum(1) - 1.0).max()
    E45.log(f"P128: Σp − 1 ≤ {dev_p:.2e} (это под ожидаемого остатка QJ)")

    E45.log("loading model (fp32) ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        E45.MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    torch.set_grad_enabled(False)
    W = model.lm_head.weight.detach().float().clone()
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    del model

    rows_base = W[U128].numpy().astype(np.float64)
    nrm0 = norm_rows(Hst, g_np, rows_base)
    kap_base = kappa_chunk(Hst, g_np, rows_base, False)
    kap_base_q = kappa_chunk(Hst, g_np, rows_base, True, Pst)
    sigW = float(W.std())
    E45.log(f"база: κ(J) = {kap_base:.8f}, κ(QJ) = {kap_base_q:.8f} (r=16, {N_SPEC_DOCS} доков), "
            f"мин ‖J_t‖ = {nrm0.min():.3e}, σ_W = {sigW:.3e}")

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
        cn = c.numpy().astype(np.float64)
        cvec = torch.from_numpy(cn.astype(np.float32))
        W_e = W + cvec[None, :]                             # 1_V ⊗ cᵗ (без temps на целый [V,D])
        # Для J и κ берём правку в fp64, без округления W: иначе предсказанный ранг-1 член
        # расходился бы с фактом на уровне fp32-округления, и проверка п.2 меряла бы не то.
        rows_e = rows_base + cn[None, :]
        cn = c.numpy().astype(np.float64)
        cvec = torch.from_numpy(cn.astype(np.float32))

        # п.2: сырой J движется, и ровно предсказанным ранг-1 членом
        num = den = 0.0
        for c0 in range(0, len(Hst), CHUNK):
            c1 = min(c0 + CHUNK, len(Hst))
            J0 = j_chunk(Hst, g_np, rows_base, c0, c1)
            J1 = j_chunk(Hst, g_np, rows_e, c0, c1)
            dd = J1 - J0
            pred = gauge_dJ_pred(Hst, g_np, cn, c0, c1)
            num += float(np.einsum("tkd,tkd->", dd - pred, dd - pred))
            den += float(np.einsum("tkd,tkd->", dd, dd))
            del J0, J1, dd, pred
        pred_rel = float(np.sqrt(num / max(den, 1e-300)))
        dJ_max, dJ_med = drift_of(Hst, g_np, Pst, rows_base, rows_e, False)
        dQ_max, dQ_med = drift_of(Hst, g_np, Pst, rows_base, rows_e, True)
        kap_q = kappa_chunk(Hst, g_np, rows_e, True, Pst)
        m = E45.dense_metrics(Ha[::METRIC_STRIDE], W_e, W)
        kap_raw = kappa_chunk(Hst, g_np, rows_e, False)
        row = dict(name=name, alpha=alpha, c_norm=float(c.norm()),
                   drift_J_max=dJ_max, drift_J_med=dJ_med,
                   drift_QJ_max=dQ_max, drift_QJ_med=dQ_med,
                   drift_QJ_over_J=dQ_max / max(dJ_max, 1e-300),
                   pred_rank1_rel_resid=pred_rel,
                   kappa=kap_raw, kappa_rel_dev=abs(kap_raw - kap_base) / max(kap_base, 1e-30),
                   kappa_QJ=kap_q, kappa_QJ_base=kap_base_q,
                   kappa_QJ_rel_dev=abs(kap_q - kap_base_q) / max(kap_base_q, 1e-30),
                   kl_be_max=float(m["kl_be"].max()), kl_be_mean=float(m["kl_be"].mean()),
                   tv_max=float(m["tv"].max()), flip=float(m["flip"].mean()))
        rows.append(row)
        E45.log(f"  {name:>18s} ‖c‖={row['c_norm']:.3e}: drift J = {dJ_med:.2e} (max {dJ_max:.2e})"
                f"  против ранг-1 предсказ. остаток {pred_rel:.2e}")
        E45.log(f"  {'':>18s} drift QJ = {dQ_med:.2e}  κ(J) = {kap_raw:.8f} "
                f"(отн. {row['kappa_rel_dev']:.2e})  κ(QJ) = {kap_q:.8f} "
                f"(отн. {row['kappa_QJ_rel_dev']:.2e})  KL_max = {row['kl_be_max']:.2e}  "
                f"TV_max = {row['tv_max']:.2e}  top1→ = {row['flip']:.4f}")
        del W_e, m

    # п.1: тот же KL в fp64 — если там ~1e-15, значит fp32-остаток есть округление логитов
    n64, vstep = 64, 8
    H64 = torch.from_numpy(Hst[:n64 * vstep:vstep])
    cd64 = torch.from_numpy(cn)
    z32a = (H64.float() @ W.T).double()
    z32b = (H64.float() @ (W + cvec[None, :]).T).double()
    Wd = W.double()
    z64a = H64 @ Wd.T
    z64b = H64 @ (Wd + cd64[None, :]).T
    del Wd
    shift64 = H64 @ cd64[:, None]                            # [n,1] — точная добавка логитов
    fp = dict(n_pos=n64, vocab_stride=vstep,
              fp32_max_abs_dev=float((z32b - z32a - shift64.float().double()).abs().max()),
              fp64_max_abs_dev=float((z64b - z64a - shift64).abs().max()),
              shift_scale=float(shift64.abs().max()))
    E45.log(f"  {'под логитов':>18s}: fp32 max|Δz − c·ĥ| = {fp['fp32_max_abs_dev']:.2e}, "
            f"fp64 = {fp['fp64_max_abs_dev']:.2e} при масштабе добавки {fp['shift_scale']:.2e}")

    # контроль: правка одного ряда того же масштаба — обязана быть заметной
    kcol = int(U128[7])
    W_c = W.clone()
    W_c[kcol] = W_c[kcol] + 0.15 * sigW * torch.from_numpy(rng.standard_normal(D).astype(np.float32))
    rows_c = W_c[U128].numpy().astype(np.float64)
    dc_max, dc_med = drift_of(Hst, g_np, Pst, rows_base, rows_c, False)
    m_c = E45.dense_metrics(Ha[::METRIC_STRIDE], W_c, W)
    ctl = dict(name=f"control_1row_U128idx7_drift{dc_med:.3f}", drift_J_max=dc_max,
               drift_J_med=dc_med, kappa=kappa_chunk(Hst, g_np, rows_c, False),
               kl_be_max=float(m_c["kl_be"].max()),
               tv_max=float(m_c["tv"].max()), flip=float(m_c["flip"].mean()))
    E45.log(f"  {'control_1row':>18s}: drift_J = {dc_med:.2e}  κ = {ctl['kappa']:.8f}  "
            f"KL_max = {ctl['kl_be_max']:.2e}  top1→ = {ctl['flip']:.4f}")

    ok_beh = (max(r["kl_be_max"] for r in rows) < TOL_KL
              and max(r["tv_max"] for r in rows) < TOL_TV
              and fp["fp64_max_abs_dev"] < TOL_FP64_KL)
    ok_raw = (min(r["drift_J_med"] for r in rows) > MIN_GAUGE_J_DRIFT
              and max(r["pred_rank1_rel_resid"] for r in rows) < TOL_PRED_REL)
    ok_proj = (max(r["drift_QJ_max"] for r in rows) < TOL_QJ_DRIFT
               and max(r["drift_QJ_over_J"] for r in rows) < TOL_QJ_OVER_J
               and max(r["kappa_QJ_rel_dev"] for r in rows) < TOL_KAPPA_REL)
    ok_ctl = ctl["drift_J_med"] > MIN_CONTROL_DRIFT and ctl["kl_be_max"] > MIN_CONTROL_KL
    ok = ok_beh and ok_raw and ok_proj and ok_ctl
    res = dict(experiment="E-45c: gauge-null контроль конвейера ущерба", L=E45.L, r=R,
               kappa_base=kap_base, kappa_QJ_base=kap_base_q, p128_norm_dev=float(dev_p),
               rows=rows, control=ctl, logits_floor=fp, metric_stride=METRIC_STRIDE,
               tol=dict(qj_drift=TOL_QJ_DRIFT, kl=TOL_KL, tv=TOL_TV, kappa_rel=TOL_KAPPA_REL,
                        pred_rel=TOL_PRED_REL, fp64_kl=TOL_FP64_KL),
               checks=dict(behaviour=ok_beh, raw_J_rank1=ok_raw, projected_J=ok_proj,
                           control_visible=ok_ctl),
               passed=bool(ok), wall_sec=round(time.time() - t0, 1))
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    E45.log("\nВЕРДИКТ:")
    E45.log("  поведение (KL/TV/top-1) invariant: " + ("да" if ok_beh else "НЕТ")
            + " — с учётом того, что fp64-под логитов "
            f"{fp['fp64_max_abs_dev']:.1e} против fp32 {fp['fp32_max_abs_dev']:.1e}")
    E45.log("  сырой J логитов движется ровно ранг-1 членом (κ(J) НЕ калибровочно-инвариантна): "
            + ("да" if ok_raw else "НЕТ"))
    E45.log("  QJ = (I−1pᵗ)J и κ(QJ) invariant: " + ("да" if ok_proj else "НЕТ"))
    E45.log("  контроль одной строкой заметен: " + ("да" if ok_ctl else "НЕТ"))
    E45.log(f"  → {OUT}, wall {res['wall_sec']} с")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
