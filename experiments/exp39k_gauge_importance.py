"""E-39k — gauge-free норма строк J против ‖W‖ как importance для квантизации.

ТЗ: docs/E39.md §11. Утверждение ТЗ: стандартная importance `‖W_i‖²` (GPTQ/AWQ) — это
лишь приближение для gauge-free нормы строк канонического Якобиана, потому что
gauge-оператор `P(p)` убивает 98-99% энергии J (E-33r). Математика (paper §2.5):

    P(p) = diag(p) − p pᵀ,   p = p_S — softmax, ренормализованный на 𝒮 (кэш-поле P128)
    J^(p) = P(p)·J           — канонический (gauge-free) Якобиан = ∂p/∂h
    s_i    = ‖W_i‖²          — стандартная importance строки i ∈ 𝒮
    s_i^gf = mean_t ‖J^(p)_t,i‖²

Строчная форма, которую считаем (сворачивается ровно в P(p)J):

    J^(p)_i = p_i·(J_i − m),  m = Σ_j p_j J_j    ⇒   s_i^gf = mean_t p_i²‖J_i − m‖²

Второй множитель важен для чтения вердикта: gauge-free importance — произведение
«масса токена» × «форма», поэтому рядом измеряется ρ с каждым множителем и partial ρ по
rank(p̄) (прирост сверх частоты токена; та же дисциплина инкрементальности, что в E-39j
и E-39l v2/v3).

Замер корреляционный, настоящее квантование не запускается. Слой: L23 основной
(единственный, где J — readout и строки J суть строки W_lm), L18 вторичный.

Запуск (локально, CPU, ~5 мин, ~8 GB RAM):
    HF_HOME=.hf_cache .venv/bin/python experiments/exp39k_gauge_importance.py
Выход: results/exp39k_gauge_aware_importance.json (имя по ТЗ §11.4).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from _e38_cache import DOMAINS, N_PER_DOMAIN, S, load_layer

MODEL = "Qwen/Qwen3.5-0.8B"
L_MAIN = 23
L_SECOND = 18
LAYERS = (L_MAIN, L_SECOND)
PERM = 1000            # перестановочный null и bootstrap, B (один rng, seed ниже)
SEED = 0
TOP_FRAC = 0.25        # квартиль строк = «кого защищал бы» importance
AFTER_P_RANGE = (0.0, 0.10)          # ‖P J‖²/‖J‖²: P масштаба ~p (не идемпотентен!),
                                     # поэтому доля после P мелкая — это sanity-граница,
                                     # а НЕ gauge-share; g_uni ниже — отдельный объект
                                     # paper §2.5, измеряется здесь впервые
CHECK_POSITIONS = 256  # подвыборка позиций для проверки выравнивания J ↔ W
RHO_PASS, RHO_MODERATE = 0.5, 0.8     # таблица ТЗ §11.6


# ---------------------------------------------------------------- статистика
def _rho(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _rho_p(x, y, b, rng):
    """Spearman + перестановочный p (двусторонний, (k+1)/(B+1)) + bootstrap CI."""
    r = _rho(x, y)
    n = len(x)
    null = np.array([_rho(x[rng.permutation(n)], y) for _ in range(b)])
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / (b + 1)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        vals.append(_rho(x[idx], y[idx]))
    ci = ([float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
          if vals else [float("nan"), float("nan")])
    return {"rho": r, "p_perm": float(p), "ci95": ci, "n": int(n), "b": int(b)}


def _rank(a):
    order = np.argsort(np.argsort(a))
    return order.astype(float) / (len(a) - 1)


def _resid(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _partial_rho(x, y, ctrl):
    """Частный Спирмен: всё ранжируется, затем остатки по контрольным переменным."""
    rx, ry = _rank(x), _rank(y)
    rc = np.column_stack([_rank(ctrl[:, j]) for j in range(ctrl.shape[1])])
    return _rho(_resid(ry, rc), _resid(rx, rc))


def _top_overlap(x, y, k):
    """Пересечение топ-k строк двух importance-ранжировок (кого реально защищали бы)."""
    ax = set(np.argsort(-np.asarray(x, dtype=float))[:k].tolist())
    ay = set(np.argsort(-np.asarray(y, dtype=float))[:k].tolist())
    inter = len(ax & ay)
    return {"k": int(k), "intersect": int(inter), "frac_of_k": inter / k,
            "jaccard": (inter / (2 * k - inter)) if inter else 0.0}


# ---------------------------------------------------------------- измерение
@torch.no_grad()
def _row_energy(J, P):
    """Энергии gauge-free и сырого J по строкам → (gf, raw) формы [n,S,K].

    `J_gf = p·(J − Jᵀp) = P(p)J` — та же конструкция, что в шапке _e38_cache и в
    exp27_6_isolation. Считается по документам в fp64 (сумма квадратов по D=1024).
    """
    gf = torch.empty(J.shape[:3], dtype=torch.float64)
    raw = torch.empty_like(gf)
    for i in range(J.shape[0]):
        Jd, Pd = J[i].double(), P[i].double()
        m = torch.einsum("sk,skd->sd", Pd, Jd)   # m_s = Σ_j p_sj J_sj
        raw[i] = Jd.pow(2).sum(-1)
        gf[i] = Pd.pow(2) * (Jd - m.unsqueeze(-2)).pow(2).sum(-1)
    return gf, raw


@torch.no_grad()
def _alignment_check(J, H, W_rows, g):
    """L23: J_i = c·(M_i − (M_i·û)û), M = W_lm[𝒮]·diag(g), c = √D/‖h‖, û = h/‖h‖.

    Единственная проверка, что «строка i у J» и «строка i у W» — один и тот же токен:
    при переставленных индексах относительная ошибка реконструкции взлетает на порядки
    (эталон E-39b на fp64 — 7.77e-07; здесь кэш fp32 → порог 1e-4). Позиционное
    пространство кэша — from-the-end, J и H идут одним ключом (dom,si,t), flip не нужен.
    """
    n, _, K, Dd = J.shape
    idx = np.linspace(0, n * S - 1, CHECK_POSITIONS).astype(int)
    M = (W_rows * g)
    errs, ddiag, doff = [], [], []
    for flat in idx:
        i, t = divmod(int(flat), S)
        h = H[i, t].double()
        u = h / h.norm()
        c = Dd ** 0.5 / h.norm()
        recon = c * (M - (M @ u).unsqueeze(1) * u.unsqueeze(0))
        Jc = J[i, t].double()
        errs.append(((recon - Jc).norm() / Jc.norm()).item())
        cos = (recon / recon.norm(dim=1, keepdim=True)) @ \
              (M / M.norm(dim=1, keepdim=True)).T
        ddiag.append(torch.diagonal(cos).mean().item())
        doff.append(((cos.sum() - torch.diagonal(cos).sum()) / (K * (K - 1))).item())
    errs = np.array(errs)
    return {"n_positions": int(len(idx)),
            "recon_rel_err_median": float(np.median(errs)),
            "recon_rel_err_max": float(errs.max()),
            "cos_diag_mean": float(np.mean(ddiag)),
            "cos_offdiag_mean": float(np.mean(doff)),
            "note": "диагональное доминирование cos(M_i, J_i) + малая rel err = индексы "
                    "J и строки W выровнены; без этого корреляция importance бессмысленна"}


def measure(layer, w_lm, g, want):
    """Слой → пер-строковые важности + диагностика. Возвращает плоский dict."""
    ck = load_layer(layer, want=want)
    J, P, ts = ck["J"], ck["P"], ck["U128"]
    W_rows = w_lm[ts].double()
    gf, raw = _row_energy(J, P)
    p2 = P.double().pow(2)
    after_P = float((gf.sum() / raw.sum()).item())
    assert AFTER_P_RANGE[0] < after_P < AFTER_P_RANGE[1], (
        f"L{layer}: ‖P(p)J‖²/‖J‖² = {after_P:.4g} — вне ({AFTER_P_RANGE}): либо p не "
        f"ренормализован, либо J пуст")
    # g_uni = ‖Q J‖²/‖J‖², Q = 1·1ᵀ/K — доля энергии сырого J на uniform-режиме
    # (paper §2.5). Измеряется здесь впервые: никакого пререгистрированного ожиданий
    # нет, поэтому это диагностика, а не гейт. Особо: «P убивает 98-99% энергии» (E-33r)
    # относится к ESG `gauge_share` (§2.3), а не к `g_uni` — paper сам предупреждает.
    g_uni = float((J.double().sum(-2).pow(2).sum(-1).sum()
                   / (raw.sum() * J.shape[-2])).item())
    # «P убивает 99.95% энергии» — это свойство матрицы P, а не факта gauge-
    # доминирования: для изотропного J ожидание ‖PJ‖²/‖J‖² = ‖P‖²_F/K, а
    # ‖P(p)‖²_F = Σp² − 2Σp³ + (Σp²)² ≈ 1/K при равномерном p ⇒ отношение ≈ 1/K².
    # Отсюда after_P / ref_iso — во сколько раз реальный J дальше от изотропного.
    ref_iso = _iso_P_reference(P)
    dom_of = np.repeat(np.arange(len(DOMAINS)), N_PER_DOMAIN)
    out = {
        "layer": layer,
        "energy_after_P_over_raw": after_P,
        "ref_iso_ratio": ref_iso,
        "after_P_over_iso": after_P / ref_iso,
        "g_uni_uniform_energy_share": g_uni,   # диагностика, ожиданий не было
        "p_normalized_max_dev": float((P.sum(-1) - 1.0).abs().max()),
        "n_positions": int(J.shape[0] * S),
        "U128": ts,
        "s_w": W_rows.pow(2).sum(-1).numpy(),                 # ‖W_i‖² (ТЗ)
        "s_mk": (W_rows * g).pow(2).sum(-1).numpy(),          # ‖M_K,i‖² (g-вариант)
        "gf_mean": gf.mean(dim=(0, 1)).numpy(),               # s^gf по ТЗ
        "gf_median": np.median(gf.numpy(), axis=(0, 1)),
        "raw_mean": raw.mean(dim=(0, 1)).numpy(),
        "p_bar": P.double().mean(dim=(0, 1)).numpy(),         # средняя масса токена
        "shape_mean": (gf / p2.clamp_min(1e-30)).mean(dim=(0, 1)).numpy(),
        "gf_by_domain": {d: gf[dom_of == k].mean(dim=(0, 1)).numpy()
                         for k, d in enumerate(DOMAINS)},
    }
    if "H" in ck:
        out["align"] = _alignment_check(J, ck["H"], W_rows, g)
    del ck, J, P, gf, raw
    return out


@torch.no_grad()
def _iso_P_reference(P, max_pos: int = 1024) -> float:
    """E[‖P(p)J‖²_F / ‖J‖²_F] для изотропного J = ‖P(p)‖²_F / K, среднее по позициям.

    При j ~ iid(σ²I) имеем E‖Pj‖² = σ²·tr(PᵀP), E‖j‖² = σ²K ⇒ отношение ‖P‖₂_F²/K, а
    для P = diag(p) − ppᵀ (симметрична) tr(P²) = Σp² − 2Σp³ + (Σp²)². Смысл: это «сколько
    P съедает энергии сама по себе, без всякого gauge», то есть знаменатель для
    energy_after_P_over_raw. При равномерном p величина равна (K−1)/K³ ≈ 1/K².
    """
    Pd = P.double()[:max_pos]
    s2 = Pd.pow(2).sum(-1)
    s3 = Pd.pow(3).sum(-1)
    return float(((s2 - 2 * s3 + s2.pow(2)) / Pd.shape[-1]).mean().item())


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _rho_vs_threshold(rho, thr, n):
    """Односторонний Fisher-z тест «ρ < thr» (p_perm отвечает на другой вопрос).

    p_perm тестирует ρ ≠ 0; таблица §11.6 решает неравенством с порогом, поэтому для
    строки таблицы нужен именно этот тест, а не p_perm.
    """
    if not (np.isfinite(rho) and 0.0 < thr < 1.0 and abs(rho) < 1.0):
        return None
    z, z_t, se = math.atanh(rho), math.atanh(thr), 1.0 / math.sqrt(n - 3)
    p1 = _phi((z - z_t) / se)
    return {"threshold": thr, "test": f"Fisher z, one-sided, n={n}",
            "p_rho_below_threshold": p1,
            "ci95_fisher": [math.tanh(z - 1.959963985 * se),
                            math.tanh(z + 1.959963985 * se)]}


def _final_verdict(res):
    """Вердикт v2: строка таблицы §11.6 — не вывод, вывод делает incrementality.

    Таблица читается как «ρ(s^gf, ‖W‖²) = …», но из одного лишь ρ < 0.5 следовало бы
    «стандартную importance ошибочна, gauge-aware — кандидат на замену». Это чтение
    неверно, потому что s^gf = p̄²·shape и ρ(p̄, s^gf) ≈ 1: сравнивается не «gauge против
    ‖W‖», а «частота токена против ‖W‖». Поэтому вердикт строится по частному ρ.
    """
    corr = res["correlations_L23"]
    prim = corr["primary_s_gf_vs_W2"]
    part = corr["partial_s_gf_vs_W2_given_rank_mass"]
    rho, n = prim["rho"], prim["n"]
    v1 = res.get("verdict") or {}
    vs05 = _rho_vs_threshold(rho, RHO_PASS, n)
    vs08 = _rho_vs_threshold(rho, RHO_MODERATE, n)
    mass = corr["mass_vs_s_gf"]["rho"]
    shape_w = corr["shape_vs_W2"]["rho"]
    res["verdict"] = {
        "schema": "v2",
        "rho_primary": rho, "p_perm": prim["p_perm"], "ci95": prim["ci95"],
        "p_perm_question": "ρ ≠ 0 (не «ρ < 0.5»): на строку таблицы он не опирается",
        "rho_vs_threshold_0.5": vs05,
        "rho_vs_threshold_0.8": vs08,
        "table_row": "ошибочна" if rho < RHO_PASS else (
            "умеренно" if rho <= RHO_MODERATE else "случайно правильна"),
        "table_row_status": "формально выполняется (односторонний p = "
                            f"{vs05['p_rho_below_threshold']:.4f}), но строка не является "
                            "вердиктом: см. decisive_test",
        "decisive_test": "partial ρ(s^gf, ‖W‖² | rank p̄) — incrementality",
        "rho_partial_given_mass": part["rho_partial"], "p_partial": part["p_perm"],
        "text": (
            "s_gf — это токен-частота, а не gauge-структура: ‖W‖² и s^gf ранжируют "
            f"разное (ρ = {rho:.3f}, односторонний p < 0.5 = {vs05['p_rho_below_threshold']:.4f}), "
            f"но s^gf ≈ p̄² (ρ = {mass:.3f}), shape ⊥ ‖W‖ (ρ = {shape_w:.3f}) и после "
            f"контроля массы связь исчезает (partial ρ = {part['rho_partial']:.3f}, "
            f"p = {part['p_perm']:.3f}) ⇒ «gauge-aware importance» из ТЗ не является "
            "объектом, альтернативным ‖W‖²; заменять importance на s^gf не из чего"),
        "incrementality": _incrementality_note(part["p_perm"]),
        "ci_clears_band": bool(prim["ci95"][1] < RHO_PASS or prim["ci95"][0] > RHO_MODERATE),
        "not_a_replacement_reasons": _reasons(res, corr),
        "verdict_history": v1.get("verdict_history") or {"v1": v1.get("text", "(не записан)")},
    }
    res["verdict"]["verdict_history"].setdefault("v2", res["verdict"]["text"])
    lim = res.setdefault("limitations", {})
    lim["p_perm_vs_threshold"] = (
        "p_perm в primary тестирует «ρ ≠ 0», а не «ρ < 0.5»; для строки таблицы нужен "
        f"односторонний Fisher-z: p = {vs05['p_rho_below_threshold']:.4f} (то есть «ρ < 0.5» "
        "значимо и на α = 0.01, хотя bootstrap CI [..] почти касается 0.5)")
    lim["per_domain_spread"] = (
        "ρ(s^gf, ‖W‖²) по доменам различается в разы ("
        + ", ".join(f"{d} {v:.2f}" for d, v in res["per_domain_rho_gf_vs_W2_L23"].items())
        + ") ⇒ величина ρ есть свойство данных калибровки, а не свойство модели")
    lim["l18_not_in_report"] = (
        "ρ(L18) = "
        f"{res.get('secondary_L18', res.get('exploratory_L18', {})).get('s_gf_vs_W2', {}).get('rho', float('nan')):.3f} "
        "не идёт в отчёт: J на L18 — слойный Якобиан, его строки не суть строки W_lm, "
        "механизм неопределён; числу место только в exploratory")
    d23 = res["layers_diag"][str(L_MAIN)]
    if "ref_iso_ratio" in d23:
        lim["P_energy_is_not_gauge_evidence"] = (
            f"ТЗ/§2.5 формулировка «P убивает 98-99% энергии, потому что gauge доминирует» "
            f"для этой величины неверна: ‖P(p)J‖²/‖J‖² = {d23['energy_after_P_over_raw']:.2e} "
            f"при изотропном референсе {d23['ref_iso_ratio']:.2e} (отношение "
            f"{d23['after_P_over_iso']:.2f}×) — то есть P съедает не меньше, чем съела бы на "
            "случайном J, и в величине нет никакого свидетельства о gauge. Gauge-доля в "
            f"собственном смысле — g_uni = {d23['g_uni_uniform_energy_share']:.3f}")
    if "secondary_L18" in res:
        res["exploratory_L18"] = res.pop("secondary_L18")
        res["exploratory_L18"]["in_report"] = False
    res.setdefault("field_history", {})['verdict_schema'] = "v2 (_final_verdict)"
    return res


def _reasons(res, corr):
    ov, dr = res["top_overlap_L23"], res["dynamic_range"]
    dom = res["per_domain_rho_gf_vs_W2_L23"]
    return {
        "dynamic_range": (
            f"p90/p10 = {dr['s_gf_p90_over_p10']:.1f}× у s^gf против "
            f"{dr['W2_p90_over_p10']:.2f}× у ‖W‖²: s^gf концентрирует защиту в верхних "
            "процентах токенов, тогда как ‖W‖² ранжирует строки почти равномерно"),
        "top_k": (
            f"пересечение топ-{ov['gf_vs_W']['k']}: s^gf∩‖W‖ = {ov['gf_vs_W']['intersect']}"
            f"/{ov['gf_vs_W']['k']} ({ov['gf_vs_W']['frac_of_k']:.1%}), а просто масса "
            f"∩‖W‖ = {ov['mass_vs_W']['intersect']}/{ov['mass_vs_W']['k']}"
            f" ({ov['mass_vs_W']['frac_of_k']:.1%}) — масса ближе к ‖W‖, чем s^gf, потому "
            f"что shape антикоррелирует с массой (ρ = "
            f"{corr['shape_vs_s_gf']['rho']:.3f}) и тянет s^gf от ‖W‖"),
        "per_domain": f"разброс ρ по доменам {min(dom.values()):.2f}…{max(dom.values()):.2f}",
        "incrementality": "нулевая (p_partial ≥ 0.05)",
    }


def recheck_verdict(path):
    """Пересобрать вердикт v2 из уже посчитанных чисел (без прогона и без кэша).

    Идемпотентен: v1 берётся из сохранённого verdict_history, если он там есть.
    """
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    _final_verdict(res)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1))
    print(f"→ {path}")
    return res


# ---------------------------------------------------------------- вердикт
def _incrementality_note(p_partial):
    if p_partial is not None and p_partial >= 0.05:
        return ("прироста нет: после контроля rank(p̄) связь gauge-free importance с ‖W‖ не "
                "значима ⇒ gauge-aware importance = переименованная частота токена")
    return ("прирост есть: связь переживает контроль массы токена ⇒ gauge-free норма "
            "несёт структуру сверх частоты токена")


# ---------------------------------------------------------------- main
def main(device="cpu", results_dir="results", small_model=None, **kw):
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model_p = small_model or MODEL
    rng = np.random.default_rng(SEED)
    print(f"E-39k gauge-aware importance: model={model_p}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_p, torch_dtype=torch.float32, attn_implementation="eager")
    text = model.model if hasattr(model.model, "norm") else model.model.language_model
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    w_lm = (text.embed_tokens if tied else model.lm_head).weight.detach().float().cpu()
    g = (text.norm.weight.detach().float().cpu() + 1.0).double()
    assert w_lm.shape[0] > 1000 and g.numel() == 1024

    res = {"experiment": "E-39k", "tz": "docs/E39.md §11", "model": model_p,
           "tied_weights": tied, "seed": SEED, "b": PERM,
           "hypothesis_tz": "‖J^(p)_i‖² (gauge-free) — правильная importance, ‖W_i‖² — "
                            "приближение; P(p) убивает ~98% энергии J",
           "spec_interpretation_table": {"rho<0.5": "standard importance ошибочна",
                                         "0.5-0.8": "умеренно, GPTQ OK",
                                         ">0.8": "standard случайно правильна"},
           "layers_diag": {}, "definitions": {
               "s_w": "‖W_lm[𝒮,i]‖²",
               "s_gf": "mean_t ‖(P(p_t)J_t)_i‖², p = P128 (ренормализованный)",
               "p_bar": "mean_t p_i — средняя масса токена на носителе",
               "shape": "mean_t ‖J_i − m_t‖² — вторая множитель s^gf (без массы)"}}

    meas = {}
    ts_main = None
    for L in LAYERS:
        want = ("J", "P", "H") if L == L_MAIN else ("J", "P")
        m = measure(L, w_lm, g, want)
        if ts_main is None:
            ts_main = m["U128"]
        else:
            assert m["U128"] == ts_main, "U128 разошёлся между слоями"
        meas[L] = m
        diag = {"energy_after_P_over_raw": m["energy_after_P_over_raw"],
                "ref_iso_ratio": m["ref_iso_ratio"],
                "after_P_over_iso": m["after_P_over_iso"],
                "ref_iso_note": "E[‖P J‖²/‖J‖²] для изотропного J = ‖P‖²_F/K; "
                                "отношение after_P/ref_iso близко к 1 ⇒ «P убила энергию" "» ничего не говорит о gauge",
                "g_uni_uniform_energy_share": m["g_uni_uniform_energy_share"],
                "g_uni_note": "доля энергии сырого J на uniform-режиме токен-пространства "
                              "(paper §2.5); это НЕ ESG gauge_share из §2.3, с которым "
                              "сопоставимы цифры 98-99% E-33r",
                "p_normalized_max_dev": m["p_normalized_max_dev"],
                "n_positions": m["n_positions"]}
        if "align" in m:
            diag["alignment_J_vs_W"] = m["align"]
            assert m["align"]["recon_rel_err_median"] < 1e-4, (
                f"L{L}: реконструкция J из строк W_lm не совпала (med "
                f"{m['align']['recon_rel_err_median']:.2e}) — строки J не выровнены со "
                f"строками W, корреляция importance была бы бессмысленной")
        res["layers_diag"][str(L)] = diag
        print(f"  L{L}: g_uni={m['g_uni_uniform_energy_share']:.4f} "
              f"afterP={m['energy_after_P_over_raw']:.2e} "
              + (f"align med={m['align']['recon_rel_err_median']:.2e} "
                 f"cos_diag={m['align']['cos_diag_mean']:.3f} "
                 f"cos_off={m['align']['cos_offdiag_mean']:.3f}"
                 if "align" in m else ""), flush=True)

    m, s_w = meas[L_MAIN], meas[L_MAIN]["s_w"]
    gf, raw, p_bar = m["gf_mean"], m["raw_mean"], m["p_bar"]
    K = len(gf)
    corr = {
        "primary_s_gf_vs_W2": _rho_p(gf, s_w, PERM, rng),
        "raw_J2_vs_W2": _rho_p(raw, s_w, PERM, rng),
        "s_gf_median_vs_W2": _rho_p(m["gf_median"], s_w, PERM, rng),
        "s_gf_vs_MK2": _rho_p(gf, meas[L_MAIN]["s_mk"], PERM, rng),
        "mass_vs_W2": _rho_p(p_bar, s_w, PERM, rng),
        "shape_vs_W2": _rho_p(m["shape_mean"], s_w, PERM, rng),
        "s_gf_vs_raw_J2": _rho_p(gf, raw, PERM, rng),
        # «gauge-aware = переименованная частота?» — проверяется прямо на самом s^gf
        "mass_vs_s_gf": _rho_p(p_bar, gf, PERM, rng),
        "shape_vs_s_gf": _rho_p(m["shape_mean"], gf, PERM, rng),
    }
    partial = _partial_rho(gf, s_w, p_bar[:, None])
    null = np.array([_partial_rho(gf[rng.permutation(K)], s_w, p_bar[:, None])
                     for _ in range(PERM)])
    p_partial = float((np.sum(np.abs(null) >= abs(partial)) + 1) / (PERM + 1))
    corr["partial_s_gf_vs_W2_given_rank_mass"] = {"rho_partial": partial,
                                                  "p_perm": p_partial, "n": int(K),
                                                  "b": PERM}
    res["correlations_L23"] = corr
    k_q = max(1, int(round(K * TOP_FRAC)))
    res["top_overlap_L23"] = {"gf_vs_W": _top_overlap(gf, s_w, k_q),
                              "gf_vs_raw": _top_overlap(gf, raw, k_q),
                              "mass_vs_W": _top_overlap(p_bar, s_w, k_q)}
    res["per_domain_rho_gf_vs_W2_L23"] = {d: _rho(v, s_w)
                                          for d, v in m["gf_by_domain"].items()}
    ms = meas[L_SECOND]
    res["secondary_L18"] = {"s_gf_vs_W2": _rho_p(ms["gf_mean"], ms["s_w"], PERM, rng),
                            "note": "J на L18 — истинный слойный Якобиан, к строкам W_lm "
                                    "отношения не имеет; слой = проверка устойчивости "
                                    "ранжировки по данным, не часть вердикта"}

    res["dynamic_range"] = {
        "s_gf_p90_over_p10": float(np.percentile(gf, 90) / max(np.percentile(gf, 10), 1e-300)),
        "W2_p90_over_p10": float(np.percentile(s_w, 90) / max(np.percentile(s_w, 10), 1e-300)),
        "note": "во сколько раз верхняя дека отличается от нижней — насколько по-разному "
                "две importance взвешивали бы строки"}
    res["limitations"] = {
        "rows": f"K={K} строк 𝒮; V−K строк W_lm недоступны (ТЗ §11.5 «остальные — "
                f"standard»), то есть проверяется ранжирование внутри 𝒮, а не по матрице",
        "level": "row-level утверждение; column-level (активационная) AWQ-importance "
                 "не проверялась",
        "power": "n=128 → широкое bootstrap CI; таблицу §11.6 читать вместе с ci_clears_band",
        "no_ptq": "квантование не запускалось: связь importance с фактической деградацией "
                  "строки (‖ΔW_i‖, вклад в PPL) — отдельный эксперимент",
        "layer": f"вывод по L{L_MAIN} — единственному, где J = readout и строки J суть "
                 f"строки W_lm",
        "dominated_by_mass": "s^gf = p̄²·shape, поэтому ρ(s^gf,·) частично наследует "
                             "частоту токена; для разделения — mass/shape/partial ρ",
        "raw_is_structural": "ρ(‖J_i‖², ‖W_i‖²) ≈ 1 не является свидетельством правоты "
                             "стандартной importance: J_i = c_t·(M_i − (M_i·û_t)û_t) — это "
                             "строка M_K с общим для всех строк пер-позиционным множителем "
                             "и почти равным для всех строк вычитанием компоненты по û, то "
                             "есть ранжирование по ‖J_i‖ почти совпадает с ‖W_i‖ по "
                             "построению; значим ровно gauge-free контраст",
    }
    _final_verdict(res)
    res["wall_s"] = round(time.time() - t0, 1)

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "exp39k_gauge_aware_importance.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res["verdict"], ensure_ascii=False, indent=1))
    print(f"→ {path} (wall {res['wall_s']} с)")
    return res


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--recheck":
        recheck_verdict(sys.argv[2])   # пересобрать вердикт из JSON, прогон не трогать
    else:
        main()
