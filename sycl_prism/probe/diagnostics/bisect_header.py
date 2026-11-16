#!/usr/bin/env python3
"""Автоматическая бисекция: почему can_barrier_scope_value в header давал 0,
а идентичная проба в другом месте — 1.

СОСТОЯНИЕ (16.11.2026): дефект исправлен (опечатка `fence_scope` вместо
`fence_space`) ⇒ база даёт 1. Скрипт оставлен как воспроизводитель: если
вернуть опечатку в src/prism_sycl_compat.h, база снова даст 0.

Метод: берём src/prism_sycl_compat.h, вырезаем по одному блоку, компилируем
и печатаем значение can_barrier_scope_value<It>::value.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # корень дерева sycl_prism
SRC = ROOT / "src" / "prism_sycl_compat.h"
SHIM = ROOT / "probe" / "sycl_cpu"
TMP = Path("/tmp/bs")
TMP.mkdir(parents=True, exist_ok=True)

text = SRC.read_text(encoding="utf-8")


def probe_source(header_path: Path, tag: str) -> Path:
    src = TMP / f"probe_{tag}.cpp"
    src.write_text(
        "#include <sycl/sycl.hpp>\n"
        f'#include "{header_path}"\n'
        "#include <cstdio>\n"
        "int main() {\n"
        "  using It = sycl::nd_item<1>;\n"
        "  std::printf(\"%d\\n\", (int)prism_sycl::can_barrier_scope_value<It>::value);\n"
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    return src


def compile_and_run(header_text: str, tag: str) -> str:
    hdr = TMP / f"h_{tag}.h"
    hdr.write_text(header_text, encoding="utf-8")
    src = probe_source(hdr, tag)
    binp = TMP / f"bin_{tag}"
    r = subprocess.run(
        ["g++", "-std=c++20", f"-I{SHIM}", str(src), "-o", str(binp)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return "COMPILE-ERROR: " + (r.stderr.strip().splitlines() or ["?"])[0]
    out = subprocess.run([str(binp)], capture_output=True, text=True)
    return out.stdout.strip() or "RUN-EMPTY"


# --- границы блоков по тексту ---
BLOCKS = {
    "fence_scope_is_type": (
        "/* РАЗЛИЧЕНИЕ API: в DPC++ 2026",
        "/* Есть ли у группы член fence_space вообще (как значение). */",
    ),
    "has_fence_scope_value": (
        "/* Есть ли у группы член fence_space вообще (как значение). */",
        "/* Форма D (СТАНДАРТ SYCL 2020)",
    ),
    "can_barrier_memory_scope": (
        "/* Форма D (СТАНДАРТ SYCL 2020)",
        "/* Форма C-value (DPC++ 2026)",
    ),
    "can_barrier_scope_value": (
        "/* Форма C-value (DPC++ 2026)",
        "/* Форма C-type (SYCL 2020)",
    ),
    "can_barrier_group_fence": (
        "/* Форма C-type (SYCL 2020)",
        "/* Диагностика: доступна ли форма БЕЗ fence",
    ),
    "can_barrier_group_only": (
        "/* Диагностика: доступна ли форма БЕЗ fence",
        "/* Переносимый барьер внутри ядра",
    ),
    "barrier_fn": (
        "/* Переносимый барьер внутри ядра",
        "/* Какая форма барьера выбрана",
    ),
    "barrier_form_name": (
        "/* Какая форма барьера выбрана",
        None,  # до конца файла
    ),
}


def cut(text: str, name: str) -> str:
    start_marker, end_marker = BLOCKS[name]
    i = text.find(start_marker)
    if i < 0:
        return text
    if end_marker is None:
        return text[:i]
    j = text.find(end_marker, i + 1)
    if j < 0:
        return text
    return text[:i] + text[j:]


print("БАЗА (без изменений):", compile_and_run(text, "base"),
      "(ожидаем 1: опечатка fence_scope исправлена)")
print()

for name in BLOCKS:
    if name in ("can_barrier_scope_value",):
        continue  # вырезать саму пробу нельзя — нужен результат
    variant = cut(text, name)
    if variant == text:
        print(f"вырезать {name:28s}: МАРКЕР НЕ НАЙДЕН")
        continue
    print(f"вырезать {name:28s}:", compile_and_run(variant, name))
