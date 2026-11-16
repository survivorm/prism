#!/usr/bin/env python
"""E-45: проверить, что таблицы-записи E-45 в `docs/ref-exp.md` совпадают с артефактами.

Зачем. Таблица §4 два раза переписывалась в документ руками из `*.partial.json` (прогон падал на
якорях до записи финального JSON), колонка `κ(gf)` в §7 добавлялась скриптом. Числа в статье живут
из документа, а истина — в JSON; без проверки расхождение заметят читатели, а не мы.

Что делает. Парсит markdown-таблицы §4 и §7, пересчитывает каждую ячейку из
`results/exp45_damage_vs_structure.json` и `results/exp45f_gauge_decomp.json` и требует совпадения
в пределах половины последней печатной цифры (плюс запас ×3 на округление посередине). Любое
расхождение — падение с именем правки, колонкой и обоими значениями.

Запуск:  cd experiments && ../.venv/bin/python exp45_table_check.py
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 05.11: документ свёрнут в git history (`07c27df`), таблицы-записи перенесены в ref-exp
DOC = os.path.join(HERE, "..", "docs", "ref-exp.md")
if len(sys.argv) > 1:                      # негативная половина проверки: свой файл
    DOC = sys.argv[1]
D45 = os.path.join(HERE, "..", "results", "exp45_damage_vs_structure.json")
D45F = os.path.join(HERE, "..", "results", "exp45f_gauge_decomp.json")

# имена строк документа → ключи артефакта
NAME4 = {"base": "base", "GPTQ-4bit": "gptq4", "RTN-8bit": "rtn8",
         "rand K d0.06": "rand_K_d0.06", "rand full d0.06": "rand_full_d0.06",
         "rand K d0.12": "rand_K_d0.12", "rand full d0.12": "rand_full_d0.12",
         "rand K d0.25": "rand_K_d0.25", "rand full d0.25": "rand_full_d0.25",
         "flat_l0.01": "flat_l0.01", "flat_l0.1": "flat_l0.1", "flat_noleash": "flat_noleash"}


def num(s: str) -> float:
    """Число из ячейки: снимает `**`, `+`, `%`, пробелы, `†` и всё, что в скобках."""
    s = re.sub(r"\(.*?\)", "", s.replace("**", "").replace("`", ""))
    s = s.replace("%", "").replace(",", ".").replace("\u2212", "-").replace("+", "").strip()
    return float(s)


def rows_of(doc_text: str, header: str) -> list[list[str]]:
    """Ячейки строк markdown-таблицы, идущей сразу за строкой, содержащей `header`."""
    lines = doc_text.split("\n")
    i = next(k for k, l in enumerate(lines) if header in l and l.startswith("|"))
    out = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        out.append([c.strip() for c in l.strip().strip("|").split("|")])
    return out


CHECKED: list[str] = []


def chk(tag: str, got: float, want: float, dec: int, rel_floor: float = 1e-9) -> list[str]:
    """1.5 пол-ulp последней печатной цифры: переживает двойное округление, но не подмену
    строки или колонки (проверено негативной половиной: сдвиг последней цифры таблицы §7 падает).
    """
    CHECKED.append(tag)
    tol = max(1.5 * 0.5 * 10 ** (-dec), abs(want) * rel_floor)
    if abs(got - want) > tol:
        return [f"{tag}: в документе {want}, в артефакте {got:.6g} (допуск {tol:g})"]
    return []


def main() -> int:
    doc = open(DOC, encoding="utf-8").read()
    d = json.load(open(D45, encoding="utf-8"))
    ed = {r["name"]: r for r in d["edits"]}
    kb = ed["base"]["kappa_A"]
    gf = {r["name"]: r for r in json.load(open(D45F, encoding="utf-8"))["rows"]}
    kbf = json.load(open(D45F, encoding="utf-8"))["kappa_base"]
    bad: list[str] = []

    # --- §4: дрейф, κ/κ₀, κ(gf)/κ₀, KL A, KL B, TV B, top-1→, ΔPPL B, gap16 (доля) ---
    for r in rows_of(doc, "| правка | дрейф"):
        nm = r[0].replace("**", "").strip()
        if nm not in NAME4:
            bad.append(f"§4: строка «{nm}» не найдена в артефакте")
            continue
        k, e = NAME4[nm], ed[NAME4[nm]]
        bad += chk(f"§4 {nm} дрейф", e["drift_med"], num(r[1]), 3, 5e-3)
        bad += chk(f"§4 {nm} κ/κ₀", e["ratio_A"], num(r[2]), 4, 5e-4)
        if k in gf:
            bad += chk(f"§4 {nm} κ(gf)/κ₀", 1 + gf[k]["dkap_gf"] / kbf, num(r[3]), 4, 5e-4)
        bad += chk(f"§4 {nm} KL A", e["kl_be_A"], num(r[4]), 5, 5e-4)
        bad += chk(f"§4 {nm} KL B", e["kl_be_B"], num(r[5]), 5, 5e-4)
        bad += chk(f"§4 {nm} TV B", e["tv_B"], num(r[6]), 4, 5e-4)
        bad += chk(f"§4 {nm} top-1→", 100 * e["flip_B"], num(r[7]), 1, 5e-3)
        if nm != "base":
            bad += chk(f"§4 {nm} ΔPPL B", d["ppl_delta_pct"][k], num(r[8]), 2, 5e-3)
        bad += chk(f"§4 {nm} gap16", e["spec_gap16_med"], num(r[9]), 4, 5e-4)
        frac = re.search(r"\(([^)]*?)\)", r[9])
        bad += chk(f"§4 {nm} доля gap<1 %", 100 * e["spec_frac_gap_lt_1pct"],
                   num(frac.group(1)) if frac else float("nan"), 1, 5e-3)

    # --- §7: ‖c‖, s, Δκ полный / gauge / gf / отношение ---
    for r in rows_of(doc, "| правка | ‖c‖"):
        nm = r[0].replace("**", "").strip()
        if "…" in nm or nm not in gf:
            continue                       # строки-диапазоны (seednull_s0…s7) сравниваются ниже
        e = gf[nm]
        bad += chk(f"§7 {nm} ‖c‖", e["c_over_sigWsqD"], num(r[1]), 4, 5e-4)
        bad += chk(f"§7 {nm} s", e["s_norm"], num(r[2]), 4, 5e-4)
        bad += chk(f"§7 {nm} Δκ", 100 * e["dkap_full"] / kbf, num(r[3]), 2, 5e-3)
        bad += chk(f"§7 {nm} Δκ gauge", 100 * e["dkap_gauge"] / kbf, num(r[4]), 2, 5e-3)
        bad += chk(f"§7 {nm} Δκ gf", 100 * e["dkap_gf"] / kbf, num(r[5]), 2, 5e-3)
        bad += chk(f"§7 {nm} gauge/full", e["dkap_gauge_over_full"], num(r[6]), 2, 5e-3)

    # строка-диапазон: концы обязаны быть настоящими минимумами/максимумами полосы
    rng = next(r for r in rows_of(doc, "| правка | ‖c‖") if "…" in r[0])
    sn = [gf[n] for n in gf if n.startswith("seednull_")]
    lo, hi = (num(x) for x in rng[2].replace("–", " ").split())
    got = sorted(g["s_norm"] for g in sn)
    bad += chk("§7 seednull s min", got[0], lo, 4, 1e-6)
    bad += chk("§7 seednull s max", got[-1], hi, 4, 1e-6)

    # --- самопроверки самого чекера: база обязана читать, κ базы из §4 = κ базы артефакта ---
    assert abs(kb - kbf) / kb < 0.05, f"κ базы в двух артефактах разошёлся: {kb} против {kbf}"
    if len([r for r in rows_of(doc, "| правка | дрейф") if r[0].replace("**", "") in NAME4]) != 12:
        bad.append("§4: найдено не 12 строк правок — таблица перестроена, чекер надо обновить")

    print(f"проверено ячеек: {len(CHECKED)} (все строки §4 и §7, включая диапазонную), "
          f"расхождений: {len(bad)}")
    for b in bad:
        print("  РАСХОЖДЕНИЕ:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
