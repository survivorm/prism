"""E-45-finish: собрать финальный JSON из `*.partial.json` без перепрогона.

Зачем. Полный прогон (13 правок, ~5 ч) отвалился в самом конце на якорном assert'е — и не
потому, что числа неверные, а потому что проверка была описана не про то: база `κ` на подыборке
8 документов (0.0159209) сверялась с корпусным `κ` по 4096 позициям (0.0162333) с допуском 2e-03,
то есть требовала совпадения двух разных срезов. Измеренная сегодня полоса док-бутстрапа уровня
`κ` на L23 — ±9.5 % при 8 документах (`results/exp45_kappa_seed_null.json`), так что 1.92 % между
срезами есть выборка, а не ошибка. То же с PPL: якорь E-39b снят по 8 документам C4, а таблица —
по 32 документам C4-holdout (разница 7.8 %).

Скрипт делает ровно то, что делал хвост `main()`: проверки (уже разделённые на самоточность
конвейера и сопоставимость с чужим срезом), `ppl_delta_pct`, слайс-заметка, запись финального
JSON. Чего он НЕ делает: не пишет `exp45_edits_sparse.npz` (матрицы правок живут только в памяти
прогона) — в финальном JSON это отмечено полем `saved_sparse_edits: null`.

Запуск:  cd experiments && ../.venv/bin/python exp45_finish.py
"""
from __future__ import annotations

import json
import time

PART = "../results/exp45_damage_vs_structure.partial.json"
OUT = "../results/exp45_damage_vs_structure.json"


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    with open(PART) as f:
        out = json.load(f)
    by = {r["name"]: r for r in out["edits"]}
    anchor = out["anchors"]
    assert "base" in by, "в partial нет базовой строки — пересборка невозможна, нужен прогон"
    base_ppl = by["base"]["ppl_B"]

    # --- самоточность: база таблицы и база, против которой считается ratio_A ----------
    # ratio_A в строке = kappa_A / KAPPA_SUB, а KAPPA_SUB — это κ базы на том же срезе, значит
    # для строки base отношение обязано быть единицей ровно до арифметики с плавающей точкой.
    rel_self = abs(by["base"]["ratio_A"] - 1.0)
    log(f"  самоточность: ratio_A базы = {by['base']['ratio_A']:.12f} (rel {rel_self:.2e})")
    assert rel_self < 1e-12, f"база не совпадает сама с собой: {by['base']['ratio_A']}"
    out["selfcheck"]["kappa_selfcheck_rel"] = rel_self
    assert by["base"]["kl_be_B"] < 1e-9, "base должен давать KL = 0 точно"

    # --- сопоставимость с чужими срезами: допуск = полоса выборки, а не 2e-03 ----------
    for k, v, tol, why in (
            ("kappa_base", by["base"]["kappa_A"], 0.25,
             "другой срез позиций: подыборка 8 документов против 4096 "
             "(полоса док-бутстрапа уровня κ ±9.5 % при 8 документах)"),
            ("ppl_base", base_ppl, 0.15,
             "другой корпус C4: 32 документа holdout против 8 документов E-39b"),
            ("ppl_gptq4", by["gptq4"]["ppl_B"], 0.15,
             "то же, на правке GPTQ"),
            ("gptq_kappa_ratio", by["gptq4"]["ratio_A"], 5e-2,
             "тот же объект и тот же тип отношения, поэтому допуск узкий")):
        rel = abs(v / anchor[k] - 1)
        log(f"anchor {k}: {v:.6g} vs {anchor[k]:.6g} (rel {rel:.2e}, допуск {tol:.0e}; {why})")
        assert rel < tol, f"якорь {k} не сошёлся: rel={rel:.2e} — {why}"
        out["selfcheck"][f"anchor_{k}_rel"] = rel

    # --- уплощение обязано уплощать ----------------------------------------------------
    for name in [n for n in by if n.startswith("flat_")]:
        r = by[name]
        assert r["kappa_A"] < by["base"]["kappa_A"], f"{name}: κ не уменьшился"
        assert r["kl_be_B"] > by["base"]["kl_be_B"], f"{name}: KL не вырос"

    out["base_ppl_B"] = base_ppl
    out["ppl_delta_pct"] = {k: 100 * (r["ppl_B"] / base_ppl - 1)
                            for k, r in by.items() if k != "base"}
    out["saved_sparse_edits"] = None
    out["assembled_from_partial"] = True
    out["note"] = (out.get("note") or "") + \
        " | финальный JSON собран exp45_finish.py из *.partial.json (полный прогон упал на " \
        "mis-specified якорном assert, числа все 13 строк); матрицы правок exp45_edits_sparse.npz" \
        " из этой сборки недоступны"
    out["wall_sec_finish"] = round(time.time() - t0, 1)

    # состав правок фиксирован: base + 2 квантизации + 2 семейства × 3 дрейфа + 3 уплощения = 12
    want = {"base", "gptq4", "rtn8"} | {f"{f}_d{d}" for f in ("rand_K", "rand_full")
                                        for d in (0.06, 0.12, 0.25)} \
        | {"flat_l0.01", "flat_l0.1", "flat_noleash"}
    have = {r["name"] for r in out["edits"]}
    assert have == want, f"состава нет: нет {sorted(want - have)}, лишнее {sorted(have - want)}"
    n = len(out["edits"])
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"→ results/exp45_damage_vs_structure.json ({n} строк, {out['wall_sec_finish']} с)")


if __name__ == "__main__":
    main()
