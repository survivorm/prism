#!/usr/bin/env python3
"""split_paper.py — шаг 1 ТЗ v1.1: механический разрез paper.md.

Срезает (дословно) три ref-файла из docs/paper.md и проверяет,
что множество ID в baseline ⊆ union(срезанных файлов).

paper.md в этом шаге НЕ трогается (перезапись — шаг 2).
"""
import json
import re
import sys

PAPER = "docs/paper.md"
BASELINE = "docs/baseline_ids.json"

HDR_EXP = """# ref-exp.md — Карточки экспериментов и журнал

Полные карточки (название, цель, протокол, числа, вывод) по каждому эксперименту `E-*`.
Канонический текст, вырезан из paper.md v4.0 (2026-10-11).

**Порядок чтения:**
1. Протокол и гид (статистический каркас) и гид по чтению экспериментов — ниже (быв. §13/§14).
2. Карточки по блокам 4.1–4.7 (соответствие блоков — таблица в paper.md §4;
   внутри файла карточки в исходном порядке по темам: поле J → identity/аблации →
   bridge → чувствительность → gauge/анатомия → Гессиан → causal lever).
3. Индекс экспериментов (Прил. B) + реестр гипотез H1–H8.

Основной текст (paper.md) ссылается на `E-*`-ID и не повторяет числа.

---

"""

HDR_MATH = """# ref-math.md — Математика: доказательства, индекс символов, глоссарий

Определения, леммы (`[L-*]`, `[M-*]`, `[D-holonomy*]`), полные доказательства,
полный индекс символов и глоссарий. Канонический текст, вырезан из paper.md v4.0.

paper.md §2 и карточки ref-exp.md **ссылаются** на определения здесь — не переопределяют.

---

"""

HDR_IMPL = """# ref-impl.md — Реализация: листинги и инфраструктура

Листинги C.1–C.9 (код) и инфраструктурные детали: вычисление Якобиана (быв. §9),
кэш, данные, 𝒮, layer mapping, загрузка моделей (быв. §12). Канонический текст,
вырезан из paper.md v4.0.

---

"""

# (file, header, [(start, end), ...]) 1-based inclusive line ranges
PLAN = [
    ("docs/ref-exp.md", HDR_EXP, [(495, 1561), (1768, 1840)]),
    ("docs/ref-math.md", HDR_MATH, [(71, 134), (1723, 1766), (2269, 2343)]),
    ("docs/ref-impl.md", HDR_IMPL, [(400, 469), (472, 494), (1843, 2267)]),
]


def extract_ids(text: str) -> dict:
    return {
        "E": set(re.findall(r"\bE-(?:\d+(?:\.\d+)?[a-d]?(?:\s*Semantika)?)", text)),
        "L": set(re.findall(r"\[L-[a-z\-]+\]", text)),
        "H": set(re.findall(r"\bH[1-8]\b", text)),
        "C": set(re.findall(r"\bC\.\d\b", text)),
        "D": set(re.findall(r"\[D-[a-z\-]+\]", text)),
        "M": set(re.findall(r"\[M-\d\]", text)),
        "anchors": set(re.findall(r"\b[DE]\.\d\b", text)),
    }


def main() -> int:
    lines = open(PAPER, encoding="utf-8").readlines()
    total = len(lines)
    print(f"paper.md: {total} lines")

    moved: dict = {}
    for path, hdr, ranges in PLAN:
        body = []
        for start, end in ranges:
            if not (1 <= start <= end <= total):
                print(f"ERROR: range {start}-{end} out of bounds")
                return 1
            body.append("".join(lines[start - 1 : end]))
        body = "".join(body)
        with open(path, "w", encoding="utf-8") as f:
            f.write(hdr + body)
        ids = extract_ids(body)
        n = sum(len(v) for v in ids.values())
        moved[path] = ids
        print(f"{path}: ranges={ranges} ids={n}")

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    # остаток paper.md (строки, не вынесенные в ref-файлы) остаётся в новом paper.md
    moved_lines = set()
    for _, _, ranges in PLAN:
        for start, end in ranges:
            moved_lines.update(range(start, end + 1))
    remainder = "".join(l for i, l in enumerate(lines, 1) if i not in moved_lines)
    rem_ids = extract_ids(remainder)
    union = {}
    for key in baseline:
        union[key] = rem_ids[key]
        for ids in moved.values():
            union[key] |= ids[key]
    missing = {k: sorted(set(b) - union[k]) for k, b in baseline.items() if set(b) - union[k]}
    if missing:
        print("ERROR: ids lost in split:", json.dumps(missing, ensure_ascii=False, indent=2))
        return 1
    print("ID check OK: baseline ⊆ union(ref-*, paper-remainder)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
