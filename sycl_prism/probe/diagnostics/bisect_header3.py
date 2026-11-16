#!/usr/bin/env python3
"""Бисекция №3: подставляем В ТОЧНОСТИ вырезанный из header'а текст пробы
в реконструкцию, где всё идентично, и смотрим, что ломается.

Шаги:
  A) реконструкция с эталонной пробой            -> 1
  B) реконструкция с ТОЧНЫМ текстом из header'а   -> 1 (после исправления опечатки)

СОСТОЯНИЕ (16.11.2026, после правки): опечатка `fence_scope` вместо
`fence_space` исправлена ⇒ все варианты дают 1. Скрипт оставлен как
ВОСПРОИЗВОДИТЕЛЬ дефекта: если вернуть опечатку в src/prism_sycl_compat.h,
вариант B снова даст 0 ⇒ это и есть проверка самого дефекта.

ЧТО БЫЛО: вариант B давал 0 при дословно одинаковом с A тексте ⇒ две
семантически одинаковые пробы с РАЗНЫМИ ответами ⇒ дефект.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # корень дерева sycl_prism
SHIM = ROOT / "probe" / "sycl_cpu"
HDR = ROOT / "src" / "prism_sycl_compat.h"
TMP = Path("/tmp/bs3")
TMP.mkdir(parents=True, exist_ok=True)

htext = HDR.read_text(encoding="utf-8")

# --- точный текст пробы из header'а ---
i = htext.find("/* Форма C-value (DPC++ 2026)")
j = htext.find("/* Форма C-type (SYCL 2020)")
EXACT_PROBE = htext[i:j]

# --- точный текст alias group_type_of из header'а ---
a = htext.find("/* Тип группы, который возвращает it.get_group()")
b = htext.find("/* РАЗЛИЧЕНИЕ API: в DPC++ 2026")
EXACT_ALIAS = htext[a:b]

# --- точный текст блока D из header'а ---
d0 = htext.find("/* Форма D (СТАНДАРТ SYCL 2020)")
EXACT_D = htext[d0:i]

# --- точный текст can_barrier_group_fence ---
k0 = htext.find("/* Форма C-type (SYCL 2020)")
k1 = htext.find("/* Диагностика: доступна ли форма БЕЗ fence")
EXACT_CTYPE = htext[k0:k1]

# --- точный текст fence_space_is_type + has_fence_scope_value ---
f0 = htext.find("/* РАЗЛИЧЕНИЕ API: в DPC++ 2026")
EXACT_FENCE = htext[f0:d0]

# эталонная (заведомо рабочая) проба
REF_PROBE = """
template <class It, class = void>
struct can_barrier_scope_value : std::false_type {};
template <class It>
struct can_barrier_scope_value<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};
"""


def run(tag: str, body: str) -> str:
    hdr = TMP / f"h_{tag}.h"
    hdr.write_text(body, encoding="utf-8")
    src = TMP / f"p_{tag}.cpp"
    src.write_text(
        "#include <sycl/sycl.hpp>\n"
        f'#include "{hdr}"\n'
        "#include <cstdio>\n"
        "#include <type_traits>\n"
        "#include <utility>\n"
        "int main() {\n"
        "  using It = sycl::nd_item<1>;\n"
        '  std::printf("%d\\n", (int)prism_sycl::can_barrier_scope_value<It>::value);\n'
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    binp = TMP / f"b_{tag}"
    r = subprocess.run(["g++", "-std=c++20", f"-I{SHIM}", str(src), "-o", str(binp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        errs = [line for line in r.stderr.splitlines() if "error" in line]
        return "ERR: " + (errs[0] if errs else "?")
    out = subprocess.run([str(binp)], capture_output=True, text=True)
    return out.stdout.strip() or "EMPTY"


def wrap(alias: str, blocks: str, probe: str) -> str:
    return ("#pragma once\n#include <sycl/sycl.hpp>\n#include <type_traits>\n"
            "#include <utility>\nnamespace prism_sycl {\n" + alias + blocks + probe + "\n}\n")


# A) эталонная проба + alias из header'а
print("A) alias из header'а + ЭТАЛОННАЯ проба      :",
      run("A", wrap(EXACT_ALIAS, "", REF_PROBE)), "(ожидаем 1)")

# B) точная проба из header'а
print("B) alias из header'а + ТОЧНАЯ проба header'а:",
      run("B", wrap(EXACT_ALIAS, "", EXACT_PROBE)), "(ожидаем 1: опечатка fence_scope исправлена ⇒ проба находит форму)")

# C) точная проба, но БЕЗ alias из header'а (свой alias)
MY_ALIAS = """
template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;
"""
print("C) свой alias      + ТОЧНАЯ проба header'а:",
      run("C", wrap(MY_ALIAS, "", EXACT_PROBE)))

# D) точная проба + точный alias, но с блоком fence_space_is_type перед ней
print("D) alias+fence-блоки + ТОЧНАЯ проба       :",
      run("D", wrap(EXACT_ALIAS, EXACT_FENCE, EXACT_PROBE)))

# E) точная проба + блок D
print("E) alias+блок D    + ТОЧНАЯ проба        :",
      run("E", wrap(EXACT_ALIAS, EXACT_D, EXACT_PROBE)))

# F) точная проба + блок C-type
print("F) alias+блок C-type+ ТОЧНАЯ проба       :",
      run("F", wrap(EXACT_ALIAS, EXACT_CTYPE, EXACT_PROBE)))

print()
print("--- ТЕКСТ эталонной пробы ---")
print(REF_PROBE)
print("--- ТЕКСТ ТОЧНОЙ пробы из header'а ---")
print(EXACT_PROBE)
