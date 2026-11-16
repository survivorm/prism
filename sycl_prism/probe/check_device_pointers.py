#!/usr/bin/env python3
"""Линтер device-кода: в ядро не должны попадать ХОСТОВЫЕ указатели.

ЗАЧЕМ (найдено на карте, 16.11.2026):
    dcurve и perm_lags захватывали в ядро указатель на ХОСТ-массив лагов:
        const int64_t* pks = ks;     // ks — параметр функции (хост!)
    На реальной GPU ядро читает host-память => k = мусор => T-k = мусор =>
    циклы пишут ЗА ПРЕДЕЛЫ буфера => порча кучи ("malloc(): unaligned tcache
    chunk detected") и нулевые результаты (rel=1.0).

    CPU-шим этого НЕ ВИДИТ: у него один адресное пространство, и хост-указатель
    «работает». Это третий уже найденный класс, который видит только реальный
    SYCL-рантайм.

ПРАВИЛО: внутри тела ядра (h.parallel_for(...)) не должно встречаться имени
    параметра enclosing-функции, если этот параметр — УКАЗАТЕЛЬ.
    Скалярные параметры (int64_t T, int stat) безопасны: они копируются
    по значению.

Как работает: грубый, но достаточный разбор C++:
    1) находим функции и их списки параметров (указатель = в типе есть '*');
    2) находим тела ядер (h.parallel_for ... { ... } с балансировкой скобок);
    3) ищем в теле ядра вхождения имён-параметров-указателей.

Возвращает 0, если нарушений нет, и 1 с перечислением — иначе.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "src" / "prism_sycl.cpp", ROOT / "probe" / "probe_sycl_min.cpp"]

# Сигнатура функции: "имя( тип имя, тип имя )" — может переноситься строками.
FUNC_RE = re.compile(r"(?:^|\n)[ \t]*(?:template\s*<[^>]*>\s*)?"
                     r"(?:[\w:<>,\*&\s]+?)\b(\w+)\s*\(([^;{]*?)\)\s*(?:const\s*)?\{",
                     re.S)


def split_params(param_text: str) -> list[tuple[str, str]]:
    """-> [(тип, имя)] для одного списка параметров (без учёта вложенных скобок)."""
    out = []
    depth = 0
    cur = ""
    for ch in param_text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    res = []
    for p in out:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"(\w+)\s*(\[\s*\d*\s*\])?$", p)
        if not m:
            continue
        name = m.group(1)
        ptype = p[: m.start(1)]
        res.append((ptype, name))
    return res


def find_kernels(text: str) -> list[tuple[int, int]]:
    """-> [(start, end) оф тел ядер] по h.parallel_for(...) { ... }."""
    spans = []
    for m in re.finditer(r"h\.parallel_for\s*\(", text):
        i = text.index("{", m.end())
        depth = 0
        j = i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        spans.append((i, j))
    return spans


def body_span(text: str, open_brace: int) -> tuple[int, int]:
    """-> (start, end) тела функции по балансировке скобок."""
    depth = 0
    j = open_brace
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return open_brace, j
        j += 1
    return open_brace, len(text)


def strip_comments(text: str) -> str:
    """Заменяет содержимое комментариев пробелами (номера строк сохраняются).

    Без этого ловились бы ЛОЖНЫЕ срабатывания: в комментарии
    «U[t] = normalize(u[t]*hn[t])» параметры u и hn встречаются как u[t]/hn[t].
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def check(path: Path) -> list[str]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    kernels = find_kernels(text)
    for fm in FUNC_RE.finditer(text):
        fname = fm.group(1)
        params = split_params(fm.group(2))
        ptr_params = {name for ptype, name in params if "*" in ptype}
        if not ptr_params:
            continue
        ob = text.index("{", fm.end() - 1)
        b0, b1 = body_span(text, ob)
        body = text[b0:b1]

        # (1) ЛОКАЛЬНЫЙ УКАЗАТЕЛЬ, ИНИЦИАЛИЗИРОВАННЫЙ ПАРАМЕТРОМ:
        #     const int64_t* pks = ks;   — это и есть баг (хост-массив в ядре).
        for m in re.finditer(
                r"(?:^|\n)[ \t]*(?:const\s+)?[\w:]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*;",
                body):
            alias, src = m.group(1), m.group(2)
            if src in ptr_params:
                line = text[: b0 + m.start()].count("\n") + 1
                problems.append(
                    f"{path.relative_to(ROOT)}:{line}: функция {fname}(): "
                    f"указатель '{alias}' инициализирован параметром '{src}' ⇒ "
                    f"это ХОСТовый указатель (ядро прочитает мусор ⇒ OOB ⇒ порча кучи)")

        # (2) ПРЯМОЕ использование параметра-указателя внутри тела ядра.
        for ks, ke in kernels:
            if not (b0 <= ks <= b1):
                continue
            kbody = text[ks:ke]
            for name in sorted(ptr_params):
                if re.search(r"\b" + re.escape(name) + r"\b\s*[\[\.\-\+]", kbody):
                    line = text[:ks].count("\n") + 1
                    problems.append(
                        f"{path.relative_to(ROOT)}:{line}: функция {fname}(): "
                        f"параметр-указатель '{name}' используется В ЯДРЕ ⇒ "
                        f"хост-память читается с устройства")
    return problems


def main() -> int:
    all_problems: list[str] = []
    for src in SOURCES:
        if not src.exists():
            print(f"!! нет файла {src}")
            return 1
        all_problems += check(src)
    if all_problems:
        print("НАРУШЕНИЯ (хостовые указатели в ядрах):")
        for p in all_problems:
            print("  " + p)
        print("⇒ скопируйте массив на устройство (Dev<T> + q.memcpy), как это сделано "
              "для hn в vpath.")
        return 1
    print("OK: в ядрах нет хостовых указателей (все массивы скопированы на устройство)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
