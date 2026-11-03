#!/usr/bin/env python3
"""E-40.8 — синтез 6×6 (docs/e40_todo.md §10, формулы заморожены в (l1)).

Носитель первичный: per-(doc,layer), n=32×5=160 (фиксация (l1): слоёв 5).
z-скор по слою внутри статистики; NaN — попарное исключение; корреляции Spearman.
Ранг: перестановочный нуль (метки независимо по переменным), λ_i выше 95-го
перцентиля нуля; PR=(Σλ)²/Σλ²; bootstrap по документам (B=1000).
S03 per-doc: остаток log C − log(S02-фит), log-uniform 64 точки, ДПФ-бин
1 цикл/октаву (Δ∈[1,32]=5 октав ⇒ бин k=5) — из кривых E40_2.

Запуск: .venv/bin/python experiments/exp40_8_synthesis.py  (после 0.2–0.7)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from exp40_common import ROOT, dump_json, header  # noqa: E402

STATS = ["S02_xi_exp", "S03_amp_oct", "S04_tail_gap", "S05_radial_share",
         "S06_R_can", "S07_G"]
LAY = [6, 12, 18, 22, 23]
B = 1000


def s03_from_curve(curve, fit_params):
    """ДПФ-бин 1 цикл/октаву остатка log C − log fit (log-uniform 64 точки)."""
    d = np.arange(1, 33, dtype=float)
    ok = np.isfinite(curve) & (np.array(curve) > 0)
    if ok.sum() < 12:
        return None
    x = np.log(np.log(2.0) * np.linspace(0, 5, 64))       # log Δ, 5 октав
    y = np.interp(x, np.log(d[ok]), np.log(np.array(curve, float)[ok]))
    A, ell, c0 = fit_params
    model = A * np.exp(-np.exp(x) / ell) + c0
    if not np.all(model > 0):
        return None
    res = y - np.log(model)
    sp = np.fft.rfft(res)
    return float(2 * np.abs(sp[5]) / 64)                   # k=5 = 1 цикл/октаву


def corr6(M):
    """Spearman 6×6 с попарным исключением NaN."""
    p = M.shape[1]
    C = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            v = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
            if v.sum() >= 20:
                C[i, j] = C[j, i] = spearmanr(M[v, i], M[v, j]).statistic
    return C


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    e402 = json.load(open(os.path.join(results_dir, "E40_2_double_fit.json")))
    e404 = json.load(open(os.path.join(results_dir, "E40_4_intersection.json")))
    e405 = json.load(open(os.path.join(results_dir, "E40_5_kinetic.json")))
    e406 = json.load(open(os.path.join(results_dir, "E40_6_R_profile.json")))
    e407 = json.load(open(os.path.join(results_dir, "E40_7_G_fixation.json")))

    rows, rowid = [], []
    for L in LAY:
        s02 = e402["per_doc_S02_xi_exp"][str(L)]
        curves = e402["per_doc_canonical_curves"][str(L)]
        fitp = e402["per_layer"][str(L)]["C_F"]["params"] \
            if e402["per_layer"][str(L)]["C_F"]["best_form"] == "exp" else None
        s04 = e404["per_doc_S04_tail_gap"].get(str(L), {})
        s05 = e405.get("per_doc_S05_radial_share", {}).get(str(L), {})
        s06 = e406["per_doc_S06_R_can"][str(L)]
        s07 = e407["per_doc_S07_G"][str(L)]
        for d in s02:
            s3 = None
            if fitp is not None and d in curves:
                s3 = s03_from_curve(curves[d]["C_gf"], fitp)
            rows.append([s02[d], s3, s04.get(d), s05.get(d), s06.get(d), s07.get(d)])
            rowid.append({"layer": L, "doc": d})
        # разъехавшиеся конвенции ключей документов уже один раз молча обнулили
        # целую статистику (S04 писала «dom/i», все остальные — «dom#i») и тем самым
        # сдвинули PR/ранг ⇒ каждый источник обязан целиком ложиться на носитель
        for nm, s in (("S04", s04), ("S05", s05), ("S06", s06), ("S07", s07)):
            miss = sorted(set(s) - set(s02))
            if miss:
                raise RuntimeError(
                    f"E-40.8 L{L}: {nm}: {len(miss)} документов не найдены "
                    f"в носителе, пример {miss[:2]} vs {sorted(s02)[:2]} — "
                    f"разъехались ключи, статистика молча стала NaN")
    M0 = np.array(rows, float)
    # z-скор по слою
    Mz = M0.copy()
    for L in LAY:
        idx = [i for i, r in enumerate(rowid) if r["layer"] == L]
        block = M0[idx]
        mu = np.nanmean(block, axis=0)
        sd = np.nanstd(block, axis=0)
        sd[sd == 0] = 1.0
        Mz[idx] = (block - mu) / sd
    C = corr6(Mz)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    evp = np.clip(ev, 0, None)
    pr = float(evp.sum() ** 2 / max((evp ** 2).sum(), 1e-300))

    rng = np.random.default_rng(40)
    null_ev = np.zeros((B, 6))
    for b in range(B):
        Pm = np.column_stack([rng.permutation(Mz[:, j]) for j in range(6)])
        null_ev[b] = np.sort(np.linalg.eigvalsh(corr6(Pm)))[::-1]
    thr = np.quantile(null_ev, 0.95, axis=0)
    rank = int(np.sum(ev > thr))

    docs_all = sorted({r["doc"] for r in rowid})
    boots = []
    by_doc = {d: [i for i, r in enumerate(rowid) if r["doc"] == d] for d in docs_all}
    for _ in range(B):
        # doc-level бутстрап: документ тянется целиком (все его строки-слои)
        idx = []
        for d in rng.choice(np.array(docs_all), len(docs_all)):
            idx += by_doc[d]
        Cb = corr6(Mz[idx])
        evb = np.sort(np.linalg.eigvalsh(Cb))[::-1]
        evpb = np.clip(evb, 0, None)
        boots.append([int(np.sum(evb > thr)),
                      float(evpb.sum() ** 2 / max((evpb ** 2).sum(), 1e-300))])
    boots = np.array(boots, float)
    ci_rank = [float(np.quantile(boots[:, 0], 0.025)), float(np.quantile(boots[:, 0], 0.975))]
    ci_pr = [float(np.quantile(boots[:, 1], 0.025)), float(np.quantile(boots[:, 1], 0.975))]

    if rank >= 5 or pr >= 5:
        rule, fw = "R1", "dead"
    elif rank <= 1 and pr <= 2.5:
        rule, fw = "R2", "alive_weak"
    else:
        rule, fw = "R3", "indeterminate"
    C1, C2 = [0, 1, 2], [3, 4, 5]
    inter = float(np.mean([abs(C[i, j]) for i in C1 for j in C2]))
    tags_note = ("все шесть: натуральные каналы, срез/агрегация — см. артефакты "
                 "источников; z-скор по слою")

    rep = header("E-40.8", "docs/e40_todo.md §10",
                 {"E40_2": "results/E40_2_double_fit.json",
                  "E40_4": "results/E40_4_intersection.json",
                  "E40_5": "results/E40_5_kinetic.json",
                  "E40_6": "results/E40_6_R_profile.json",
                  "E40_7": "results/E40_7_G_fixation.json"},
                 channel_tags=[{"note": tags_note}],
                 extra={"carrier_primary": "per_(doc,layer)", "n_primary": int(len(M0)),
                        "n_nan_cells": int(np.isnan(M0).sum()),
                        "matrix_6x6": C.tolist(), "stats_order": STATS,
                        "eigenvalues": ev.tolist(),
                        "permutation_null_thresholds_p95": thr.tolist(),
                        "rank_estimate": {"rank": rank, "PR": pr,
                                          "bootstrap_CI_rank": ci_rank,
                                          "bootstrap_CI_PR": ci_pr},
                        "mp_brackets_spec": {"lambda_plus_n32_p6": 2.0535,
                                             "lambda_plus_n160_p6": 1.4248,
                                             "note": "первичный носитель n=160 "
                                                     "(фиксация (l1)); числа ТЗ §10.2, "
                                                     "источник истины — Монте-Карло"},
                        "clusters": {"C1": STATS[:3], "C2": STATS[3:], "C3": [], "C4": []},
                        "intra_cluster_mean_abs_rho": {"C1": float(np.mean(
                            [abs(C[i, j]) for i in range(3) for j in range(3) if i < j])),
                            "C2": float(np.mean([abs(C[i, j]) for i in C2 for j in C2 if i < j]))},
                        "inter_cluster_mean_abs_rho": inter,
                        "intra_note": "внутрикластерные связи дефинициональны (ТЗ §10.3)",
                        "R1_R2_R3_outcome": rule,
                        "framework_single_object": fw,
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_8_synthesis.json")
    dump_json(rep, out)
    print(f"wrote {out}: rank={rank} CI={ci_rank} PR={pr:.2f} CI={ci_pr} rule={rule}")
    return rep


if __name__ == "__main__":
    main()
