#!/usr/bin/env python
"""E-45-verdicts: механическое чтение таблицы E-45 по пререгистрационным критериям.

Смысл скрипта — не «красивый отчёт», а убрать место, где вывод можно согнуть под историю после
того, как числа уже увидены. Критерии взяты из PLAN.md («Пререгистрация чтения результатов»,
пункты 1–6) и применяются к results/exp45_damage_vs_structure.json как есть; каждый пункт печатает
вердикт и числа, на которых он стоит. Ничего не пересчитывается, кроме отношений и bootstrap'а.

Запуск:  cd experiments && ../.venv/bin/python exp45_verdicts.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "exp45_damage_vs_structure.json")
KL_MATCH_TOL = 0.10          # п.1: «одинаковый ущерб» = KL различается ≤ 10 %
KAPPA_RATIO_GAP = 1.5        # п.1: «разный лист» = κ/κ₀ различается ≥ 1.5×
INVISIBLE_KAPPA = 0.50       # п.2: лист разрушен, если κ/κ₀ < 0.5
INVISIBLE_KL = 1e-3          #      …при KL_B < 1e-3
INVISIBLE_FLIP = 0.01        #      …и смене top-1 < 1 %
DENSE_VS_SPARSE = 2.0        # п.3: «дрейф J — плохой предиктор», если ущерб различается ≥ 2×
KAPPA_NULL_BAND = 1e-4       # п.5: |Δκ/κ| < 1e-4 — изменения нет
GAP_MIN = 0.01               # спектральная валидность разреза r


def log(*a):
    print(*a, flush=True)


def main():
    with open(RES) as f:
        d = json.load(f)
    ed = {r["name"]: r for r in d["edits"]}
    others = [r for r in d["edits"] if r["name"] != "base"]
    # iid null'ы того же семейства (E45_SEEDNULL) — не «правки», а линейка разброса одного
    # draw'а: в критериях 1–5 и в регрессии их быть не должно, они читаются в п.7.
    fam = [r for r in d["edits"] if r["name"].startswith("seednull_")]
    others = [r for r in others if not r["name"].startswith("seednull_")]

    log("== таблица (подряд, без выделения «удачных») ==")
    hdr = (f"{'правка':>18s} {'дрейф':>7s} {'κ/κ₀':>7s} {'KL_A':>9s} {'KL_B':>9s} "
           f"{'TV_B':>7s} {'top1→':>7s} {'ΔPPL %':>8s} {'gap16':>7s} {'%gap<1':>7s}")
    log(hdr)
    for r in d["edits"]:
        dp = 0.0 if r["name"] == "base" else d["ppl_delta_pct"][r["name"]]
        log(f"{r['name']:>18s} {r['drift_med']:7.4f} {r['ratio_A']:7.3f} "
            f"{r['kl_be_A']:9.5f} {r['kl_be_B']:9.5f} {r['tv_B']:7.4f} {r['flip_B']:7.4f} "
            f"{dp:8.2f} {r['spec_gap16_med']:7.4f} {100 * r['spec_frac_gap_lt_1pct']:7.1f}")

    # --- валидность разреза: без зазора на r κ не определён ---
    bad = [r["name"] for r in d["edits"] if r["spec_gap16_med"] < GAP_MIN
           or r["spec_frac_gap_lt_1pct"] > 0.05]
    log(f"\n== п.0 валидность разреза r=16 (gap16 ≥ {GAP_MIN} и <5 % позиций без зазора) ==")
    log(f"  правок с невалидным разрезом: {bad if bad else 'нет'}")

    # --- п.1 пара с согласованным ущербом и разным листом ---
    log("\n== п.1 несёт ли κ информацию об ущербе сверх величины возмущения ==")
    pairs = []
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            if min(a["kl_be_B"], b["kl_be_B"]) <= 0:
                continue
            rel_kl = abs(a["kl_be_B"] / b["kl_be_B"] - 1.0)
            rel_k = max(a["ratio_A"] / b["ratio_A"], b["ratio_A"] / a["ratio_A"])
            if rel_kl <= KL_MATCH_TOL and rel_k >= KAPPA_RATIO_GAP:
                pairs.append((a["name"], b["name"], rel_kl, a["ratio_A"], b["ratio_A"]))
    if pairs:
        log(f"  ЕСТЬ: {len(pairs)} пар с KL_B в пределах {100 * KL_MATCH_TOL:.0f} % и "
            f"различием κ ≥ {KAPPA_RATIO_GAP}× ⇒ κ не метрика качества, а метрика способа правки")
        for a, b, rk, ka, kb in pairs[:6]:
            log(f"    {a} (κ/κ₀={ka:.3f}) ↔ {b} (κ/κ₀={kb:.3f}), |ΔKL|/KL = {rk:.3f}")
    else:
        log("  НЕТ таких пар ⇒ в этом наборе κ идёт за ущербом; дальше смотреть п.3 и п.5")

    # --- п.2 невидимое разрушение листа ---
    log("\n== п.2 разрушим ли лист невидимо для распределения ==")
    inv = [r for r in others if r["ratio_A"] < INVISIBLE_KAPPA
           and r["kl_be_B"] < INVISIBLE_KL and r["flip_B"] < INVISIBLE_FLIP]
    weakest = min(others, key=lambda r: r["kl_be_B"] / max(1e-12, (1 - r["ratio_A"])))
    log(f"  минимальный KL_B среди «сильно уплощающих» (κ/κ₀ < {INVISIBLE_KAPPA}): "
        f"{min([r['kl_be_B'] for r in others if r['ratio_A'] < INVISIBLE_KAPPA] or [float('nan')]):.2e}")
    log(f"  кандидат с наименьшей ценой на единицу уплощения: {weakest['name']} "
        f"(KL_B={weakest['kl_be_B']:.2e}, κ/κ₀={weakest['ratio_A']:.3f}, "
        f"top1→{weakest['flip_B']:.4f})")
    log("  ВЕРДИКТ: " + ("лист разрушается незаметно ⇒ аномалия листа НЕ свидетельство подмены"
                         if inv else
                         "невидимого разрушения в наборе нет; порог обнаружимости = "
                         f"min KL_B = {min(r['kl_be_B'] for r in others):.2e} при "
                         "соответствующем κ/κ₀"))

    # --- п.3 сколько строк шевелится ---
    log("\n== п.3 плотные против разреженных при одинаковом дрейфе J ==")
    for r in others:
        if r["name"].startswith("rand_full"):
            twin = "rand_K_" + r["name"].split("_")[-1]
            if twin in ed:
                ratio = (max(r["kl_be_B"], ed[twin]["kl_be_B"])
                         / max(min(r["kl_be_B"], ed[twin]["kl_be_B"]), 1e-12))
                log(f"  {r['name']} (KL_B={r['kl_be_B']:.5f}) vs {twin} "
                    f"(KL_B={ed[twin]['kl_be_B']:.5f}) → ×{ratio:.2f}")
                log(f"    вывод: {'дрейф J — плохой предиктор ущерба' if ratio >= DENSE_VS_SPARSE else 'дрейф J работает как предиктор'}")

    # --- п.4 генерализация уплощения ---
    log("\n== п.4 обучено на корпусе A, проверено на C4 (B) ==")
    for r in others:
        if r["name"].startswith("flat"):
            rel = r["kl_be_B"] / max(r["kl_be_A"], 1e-12)
            verdict = ("функциональное изменение" if 0.5 < rel < 2 else
                       "переобучено под корпус A" if rel >= 2 else
                       "BUG: KL_B < KL_A — искать ошибку в масках")
            log(f"  {r['name']:>16s}: KL_A={r['kl_be_A']:.5f} KL_B={r['kl_be_B']:.5f} "
                f"(B/A = {rel:.2f}) → {verdict}")

    # --- п.5 порог «изменения нет» ---
    log("\n== п.5 правки, где κ не изменился (|Δκ/κ| < %.0e) ==" % KAPPA_NULL_BAND)
    still = [r["name"] for r in others if abs(r["ratio_A"] - 1) < KAPPA_NULL_BAND]
    log(f"  {still if still else 'нет'}")
    near = [(r["name"], abs(r["ratio_A"] - 1)) for r in others
            if KAPPA_NULL_BAND <= abs(r["ratio_A"] - 1) < 0.02]
    log(f"  пограничные (<2 %): {near if near else 'нет'}")

    # --- п.7 разрешающая способность κ (iid seed null, если уже посчитан) ---
    log("\n== п.7 с какими различиями κ и ущерба нельзя спорить: разброс одного draw'а ==")
    sn_path = os.path.join(os.path.dirname(RES), "exp45_kappa_seed_null.json")
    if os.path.exists(sn_path):
        with open(sn_path) as f:
            sn = json.load(f)
        lb = sn["level_boot"]["L23"]
        log(f"  несогласованный CI уровня κ (док-бутстрап, exp28 per-doc): "
            f"±{lb['ci32_rel_pct_n32']:.1f} % при 32 документах, ±{lb['ci32_rel_pct_n8']:.1f} % "
            f"при 8; rel SD между документами {lb['rel_sd_docs_pct']:.1f} %")
    if not fam:
        log("  iid-строк в таблице нет — догнать отдельным проходом: "
            "E45_SEEDNULL=8 ../.venv/bin/python exp45_damage_vs_structure.py --resume")
    else:
        def spread(key, pct=True):
            v = np.array([r[key] for r in fam], dtype=float)
            lo, hi = v.min(), v.max()
            hr = 100 * (hi - lo) / 2 if pct else (hi - lo) / 2
            return lo, hi, hr, v
        for key, lbl in (("ratio_A", "κ/κ₀"), ("spec_gap16_med", "gap16"),
                         ("spec_frac_gap_lt_1pct", "доля позиций с зазором < 1 %"),
                         ("kl_be_A", "KL A"), ("kl_be_B", "KL B"), ("flip_B", "смена top-1")):
            lo, hi, hr, v = spread(key, pct=(key == "ratio_A"))
            log(f"  {len(fam)} draw'ов при дрейфе {fam[0]['drift_med']:.3f}: {lbl} ∈ "
                f"[{lo:.5g}, {hi:.5g}], медиана {np.median(v):.5g}, «разброс одного draw'а» "
                f"±{hr:.4g}{' %' if key == 'ratio_A' else ''}")
        hr_k = spread("ratio_A")[2]
        blind = [(r["name"], f"{100 * abs(r['ratio_A'] - 1):.2f} %") for r in others
                 if 100 * abs(r["ratio_A"] - 1) <= hr_k]
        log(f"  правки, сдвиг κ которых ≤ разброса одного draw'а: {blind if blind else 'нет'}")
        for key, lbl in (("kl_be_B", "KL B"), ("spec_gap16_med", "gap16")):
            v = np.array([r[key] for r in fam], dtype=float)
            out_ = [(r["name"], f"{r[key]:.5g}") for r in others
                    if not (v.min() <= r[key] <= v.max())]
            log(f"  правки вне полосы iid-семейства по {lbl}: {out_ if out_ else 'нет'}")
        log("  ⇒ знаку Δκ и величине gap16 верить нельзя, если они внутри этой полосы; "
            "заявлять различие можно только выходя за неё")

    # --- сводная регрессия: что объясняет ущерб ---
    log("\n== что объясняет KL_B: величина возмущения или лист ==")
    x = np.array([r["drift_med"] for r in others])
    k = np.array([r["ratio_A"] for r in others])
    y = np.array([r["kl_be_B"] for r in others])
    def spear(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    log(f"  Spearman(дрейф, KL_B) = {spear(x, y):+.3f}   "
        f"Spearman(κ/κ₀, KL_B) = {spear(k, y):+.3f}   n = {len(y)} правок")
    log("  Внимание: это не причинный вывод и не частичная корреляция — при n≈12 и двух режимах")
    log("  правок (плотные/разреженные) маргинальная корреляция обязана читаться вместе с п.1–3.")


if __name__ == "__main__":
    main()
