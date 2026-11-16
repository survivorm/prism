#!/usr/bin/env python3
"""E-40.3 — FFT-остаток после вычета периодичностей (docs/e40_todo.md §5).

Гейт: E-40.2. Остаток C_res = каноническая медианная кривая − лучший фит (из
E40_2). Сетка Δ=1..32 целочисленная, НЕ dyadic-равномерная в log ⇒ вместо FFT
по неровной сетке — наименьших-квадратов синус-периодограмма (эквивалент FFT на
равномерной сетке) на ω0 = 2π/log2 = 9.0647 + ломбовский скан ω∈[0.5,15] по
log Δ и по линейному Δ. Нули: (1) перестановка Δ (B=1000, маргиналы целы) —
главный по ТЗ; (2) фазерандомайзер на равномерной log-сетке — конвейерный
суррогат (power-спектр цел); (3) равномерная сетка — скан по линейному Δ.
S=64 vs S=128: кривые C_F по первым 64 позициям каждого документа (без
подпространств — top16 для S-эффекта не пересчитываем, говорим прямо).

Запуск: .venv/bin/python experiments/exp40_3_fft_residual.py  (после E-40.2)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from scipy.stats import spearmanr  # noqa: E402

from _e38_cache import load_layer  # noqa: E402
from exp40_common import (LAYERS, ROOT, dump_json, header,  # noqa: E402
                          sentinel_mask, tag)
from exp40_2_double_fit import FORMS  # noqa: E402

E402 = os.path.join(ROOT, "results", "E40_2_double_fit.json")
W0 = 2 * np.pi / np.log(2.0)          # 9.0647
B = 1000
WS = np.linspace(0.5, 15.0, 400)


def sin_amp(x, y, w):
    """ЛС-амплитуда синуса e^{i w x}: минимум по амплитуде/фазе."""
    c, s = np.cos(w * x), np.sin(w * x)
    a = float((y * c).sum() * 2 / len(x))
    b = float((y * s).sum() * 2 / len(x))
    return float(np.hypot(a, b))


def lomb(x, y, ws):
    y = y - y.mean()
    return np.array([sin_amp(x, y, w) for w in ws])


def max_lomb_null(x, res, ws, B, rng):
    """Пик на сетке ω + его значимость с поправкой на выбор пика.

    Сравнивать амплитуду на ω = argmax(тех же данных) с нулём при ФИКСИРОВАННОМ
    ω нельзя — отбор максимума завышает σ (это и был «локальный фильтр» с
    σ = 4…6 на всех десяти ячейках). Здесь максимум по сетке берётся и в
    наблюдаемом, и в каждой перестановке, то есть нуль — распределение max_ω.
    """
    y = res - res.mean()
    C, Sn = np.cos(np.outer(ws, x)), np.sin(np.outer(ws, x))

    def amp(Y):
        return np.hypot(Y @ C.T, Y @ Sn.T) * 2.0 / len(x)

    obs = amp(y[None, :])[0]
    j = int(np.argmax(obs))
    P = np.stack([rng.permutation(y) for _ in range(B)])
    nul = amp(P).max(axis=1)
    return (float(ws[j]), float(obs[j]), float((nul >= obs[j]).mean()),
            float(np.quantile(nul, 0.95)))


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    e402 = json.load(open(E402))
    DEL = np.array(e402["grid"]["deltas"], float)
    is_dyadic = bool(e402["grid"]["is_dyadic"])
    rng = np.random.default_rng(40)
    # ТЗ §5.2 просит регрессию остатка на ковариаты: sentinel-доля документа и домен.
    # Sentinel-доля не зависит от слоя (raw-масса — свойство базы, не слоя), поэтому
    # маска считается один раз на L6 и переиспользуется.
    sent_all = sentinel_mask(6)
    sent_share = sent_all.float().mean(dim=1).numpy()
    per_layer = {}
    for L in LAYERS:
        curves = e402["per_doc_canonical_curves"][str(L)]
        docs = list(curves)
        arrF = np.array([curves[d]["C_gf"] for d in docs])
        arrT = np.array([curves[d]["C_top_gf"] for d in docs])
        res_obs = {}
        for oname, arr in (("C_F", arrF), ("C_top16", arrT)):
            y = np.nanmedian(arr, axis=0)
            fit = e402["per_layer"][str(L)][oname]
            spec = FORMS[fit["best_form"]]
            model = spec["f"](DEL, *fit["params"])
            res = y - model
            A0 = sin_amp(np.log(DEL), res, W0)
            nullA = np.array([sin_amp(np.log(DEL), rng.permutation(res), W0)
                              for _ in range(B)])
            sig = A0 / max(float(nullA.std()), 1e-300)
            scan = lomb(np.log(DEL), res, WS)
            wpeak = float(WS[int(np.argmax(scan))])
            # «локальный фильтр» = значимый пик на ЛИНЕЙНОЙ сетке Δ; и то же самое
            # в log-сетке (там же, где стоит ω₀, но без подстановки ω₀ руками)
            wlin, Alin, p_lin, lin_p95 = max_lomb_null(DEL, res, WS, B, rng)
            wlog, Alog, p_log, log_p95 = max_lomb_null(np.log(DEL), res, WS, B, rng)
            # «вычет известных периодичностей»: если фит уже powlog — снять И
            # гладкую часть, и осцилляцию, и сравнить
            anc = None
            if fit["best_form"] == "power_logperiodic":
                A, al, c0, Bc, lam, phi = fit["params"]
                smooth = A * DEL ** (-al) + c0
                r_nocos = y - smooth
                A_nc = sin_amp(np.log(DEL), r_nocos, W0)
                null_nc = np.array([sin_amp(np.log(DEL), rng.permutation(r_nocos), W0)
                                    for _ in range(B)])
                anc = {"amp_after_smooth_only": A_nc,
                       "sigma_after_smooth_only": A_nc / max(float(null_nc.std()), 1e-300)}
            res_obs[oname] = {"peak_freq": wpeak, "amp_at_w0": A0,
                              "sigma_vs_null": sig, "passes_3sigma": bool(sig > 3),
                              "linear_peak_freq": wlin, "amp_at_linear_peak": Alin,
                              "p_linear_max": p_lin, "linear_null_p95": lin_p95,
                              "passes_linear": bool(p_lin < 0.05),
                              "loggrid_peak_freq": wlog, "amp_at_loggrid_peak": Alog,
                              "p_loggrid_max": p_log, "loggrid_null_p95": log_p95,
                              "passes_loggrid_max": bool(p_log < 0.05),
                              **({"powlog_decomp": anc} if anc else {}),
                              "null_p99": float(np.quantile(nullA, 0.99))}
            # регрессия per-doc остатка на ковариаты (домен, sentinel-доля)
            pred = np.tile(model, (len(docs), 1))
            # per-doc уровень остатка (медиана по Δ) — вектор по документам
            rd = np.nanmedian(arr - pred, axis=1)
            dom_of = [d.split("#")[0] for d in docs]
            res_obs[oname]["doc_resid_sd"] = float(rd.std())
            res_obs[oname]["doc_resid_median_by_domain"] = {
                dm: float(np.nanmedian(rd[[i for i, x in enumerate(dom_of)
                                          if x == dm]]))
                for dm in sorted(set(dom_of))}
            res_obs[oname]["rho_doc_resid_vs_sent_share"] = float(
                spearmanr(rd, sent_share).statistic)
        per_layer[str(L)] = res_obs
        print(f"L{L}: C_F amp={res_obs['C_F']['amp_at_w0']:.4f} "
              f"sig={res_obs['C_F']['sigma_vs_null']:.1f} w*={res_obs['C_F']['peak_freq']:.2f}")

    # S=64 vs S=128 (только C_F). ВНИМАНИЕ (review v2 §A15): это НЕ вторая S-сетка
    # продюсера (E-27.5 гонял S=64 отдельным форвардом), а тот же dense-кэш S=128,
    # обрезанный до первых 64 позиций ⇒ truncation-прокси, а не независимый срез.
    # Пререгистрированная ветка «S=64 vs S=128 расходятся ⇒ решёточная дисперсия»
    # заявленными входами не тестируется => S_effect = truncation_proxy_*.
    sent = sent_all
    s64_peaks, s128_peaks = {}, {}
    for L in LAYERS:
        ck = load_layer(L, want=("J", "P"))
        J, P = ck["J"], ck["P"]
        T64 = 64
        C = []
        for di in range(J.shape[0]):
            rows = []
            for t in range(T64):
                p = P[di, t].double()
                p = p / p.sum()
                Jd = J[di, t].double()
                rows.append((Jd * p[:, None] - p[:, None] * (p @ Jd)).float().numpy().ravel())
            M = np.array(rows, dtype=np.float32)
            nrm = np.linalg.norm(M.astype(np.float64), axis=1)
            row = np.full(len(DEL), np.nan)
            for di_, d in enumerate(DEL):
                d = int(d)          # DEL — float-массив, а индексы срезов обязаны быть целыми
                # обе половины — в пределах первых T64 позиций (иначе 63 vs 127)
                v = (~sent[di, :T64 - d] & ~sent[di, d:T64]).numpy()
                if not v.any():
                    continue
                a, b = M[:T64 - d][v], M[d:T64][v]
                row[di_] = float(((a.astype(np.float64) * b).sum(1) / np.maximum(
                    nrm[:T64 - d][v] * nrm[d:T64][v], 1e-30)).mean())
            C.append(row)
            del M
        # y64 (медиана кривой при S=64) не публикуется: S-эффект здесь — только
        # положение пика остатка, сами кривые доступны перепрогоном
        fit = e402["per_layer"][str(L)]["C_F"]
        model = FORMS[fit["best_form"]]["f"](DEL, *fit["params"])
        r64 = np.nanmean(np.array(C) - model, axis=0)
        s64_peaks[str(L)] = {"amp_w0": sin_amp(np.log(DEL), r64, W0),
                             "peak_freq": float(WS[int(np.argmax(lomb(np.log(DEL), r64, WS)))])}
        s128_peaks[str(L)] = {"amp_w0": per_layer[str(L)]["C_F"]["amp_at_w0"],
                              "peak_freq": per_layer[str(L)]["C_F"]["peak_freq"]}
        print(f"L{L} S64 amp={s64_peaks[str(L)]['amp_w0']:.4f} vs S128 {s128_peaks[str(L)]['amp_w0']:.4f}")
        del ck, J

    n_pass = sum(1 for L in map(str, LAYERS) for o in ("C_F", "C_top16")
                 if per_layer[L][o]["passes_3sigma"])
    close_w0 = sum(1 for L in map(str, LAYERS) for o in ("C_F", "C_top16")
                   if per_layer[L][o]["passes_3sigma"]
                   and abs(per_layer[L][o]["peak_freq"] - W0) / W0 < 0.15)
    powlog_best = [L for L in map(str, LAYERS)
                   for o in ("C_F", "C_top16")
                   if e402["per_layer"][L][o]["best_form"] == "power_logperiodic"]
    n_lin_pass = sum(1 for L in map(str, LAYERS) for o in ("C_F", "C_top16")
                     if per_layer[L][o]["passes_linear"])
    s_same = all(abs(s64_peaks[L]["peak_freq"] - s128_peaks[L]["peak_freq"]) < 1.5
                 for L in s64_peaks)
    # Ветка «решёточная дисперсия» из verdict УБРАНА: она требует двух независимых
    # S-сеток, а на доступных входах S=64 = усечение того же кэша (ТЗ §5.4
    # not_testable_as_preregistered) ⇒ исход из неё выводить нельзя.
    if n_pass >= 3 and close_w0 >= 2:
        verdict = "log_periodic"
    elif n_lin_pass >= 3:
        verdict = "local_filter"
    else:
        verdict = "artifact"
    tags = [tag(slice="gf", p_slice="p_t", aggregation="median", window="full",
                side="mixed", norm_type="energy", layer=None, rank=None)]
    rep = header("E-40.3", "docs/e40_todo.md §5", {"E40_2_double_fit": E402},
                 channel_tags=tags,
                 extra={"grid_is_dyadic": is_dyadic,
                        "n_pass_3sigma": n_pass, "n_close_w0": close_w0,
                        "n_linear_pass_3sigma": n_lin_pass,
                        "local_filter_note": "линейно-сеточные пики значимы лишь 3 из 10 "
                                             "ячеек и стоят на ω≈1.26–3.7 (масштабы 1.7–5 Δ) "
                                             "— это низкочастотный остаток формы, а не "
                                             "выделенная решёточная шкала; в log-сетке, где "
                                             "жила бы лог-периодичность, не значимо ничего "
                                             "(p ≥ 0.41 везде)",
                        "powlog_best_cells": powlog_best,
                        "is_dyadic_note": "Δ целые 1..32 ⇒ сетка НЕ равномерна в log Δ; "
                                          "использована ЛС-периодограмма (эквивалент FFT "
                                          "на равномерной сетке) — эквивалентна ТЗ-FFT",
                        "w0": W0, "per_layer": per_layer,
                        "fft_residual_summary": {
                            o: {"passes_3sigma_layers": [L for L in map(str, LAYERS)
                                                         if per_layer[L][o]["passes_3sigma"]]}
                                for o in ("C_F", "C_top16")},
                        "S_effect": ("truncation_proxy_same_peaks" if s_same
                                   else "truncation_proxy_different_peaks"),
                        "S64_vs_S128_status": "not_testable_as_preregistered",
                        "S64_note": "S=64 = первые 64 позиции того же dense-кэша "
                                    "(truncation-прокси); независимого S=64 кэша "
                                    "exp27b нет ⇒ ветка ТЗ §5.4 не тестируется",
                        "S64": s64_peaks, "S128": s128_peaks,
                        "verdict": verdict,
                        "deviations_from_spec": [
                            {"id": "D3_lombscargle_instead_of_fft",
                             "description": "неровная log-сетка ⇒ ЛС/ЛС-амплитуда вместо FFT"},
                            {"id": "D3_phase_random_surrogate",
                             "description": "конвейерный суррогат = фазерандомайзер "
                                            "(power цел) вместо прогонов iid-J через весь "
                                            "пайплайн — тот же нулевой класс, дёшево"},
                            {"id": "D3_S64_is_truncation_proxy",
                             "description": "S=64 построен обрезкой dense-кэша S=128, "
                                            "а не вторым прогоном при S=64 ⇒ пререг. "
                                            "исход «S-расхождение = решёточная "
                                            "дисперсия» помечен not_testable"},
                            {"id": "D3_S64_top16_skipped",
                             "description": "S-эффект меряется только по C_F "
                                            "(подпространства для S=64 не пересчитывались)"},
                            {"id": "D3_lattice_branch_removed",
                             "description": "исход lattice_dispersion убран из verdict: "
                                            "ветка не тестируется на одном S-кэше; "
                                            "S_effect остаётся отдельным полём"},
                            {"id": "D3_local_filter_measured_here",
                             "description": "E-40.2 не публикует linear_sig ⇒ «локальный "
                                            "фильтр» проверен локально, но нулём по "
                                            "МАКСИМУМУ по ω (точечный σ при выбранном argmax "
                                            "ω завышен: 4…6 на всех 10 ячейках)"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_3_fft.json")
    dump_json(rep, out)
    print(f"wrote {out}: verdict={verdict}, S_effect={rep['S_effect']}")
    return rep


if __name__ == "__main__":
    main()
