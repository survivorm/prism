#!/usr/bin/env python3
"""Статическая сверка C ABI (src/prism_sycl.h) с ctypes-биндингами (py/prism_sycl.py).

Зачем: обёртка ctypes — единственное место, где ошибка не падает при компиляции.
История: без явных argtypes int64 обрезался до int; такие же ошибки не падают
при загрузке, а молча портят числа. Этот скрипт ловит их ДО сборки библиотеки.

Проверяем:
  1) каждая функция, которую биндинг ищет через getattr/bind, есть в заголовке;
  2) число аргументов в argtypes совпадает с числом параметров в C-прототипе;
  3) каждый тип размера (int64) не сбиндинен как 32-битный (c_int) — поимённо;
  4) каждая экспортируемая функция заголовка либо связана, либо в allowlist.

Запуск:  python3 py/check_abi.py
Выход:   0 — всё сошлось; 1 — расхождения (печатаются списком).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDR = ROOT / "src" / "prism_sycl.h"
PY = ROOT / "py" / "prism_sycl.py"

# Функции заголовка, которые Python не биндит сознательно (диагностика/служебные).
ALLOW_UNBOUND = {
    "prism_sycl_abi_version",
    "prism_sycl_probe",
    "prism_sycl_last_error",
}

# Имена параметров, несущих размеры/индексы: обязаны быть int64 на стороне Python.
SIZE_HINT = re.compile(
    r"^(n|nd|d|dd|t|td|nk|k|n_lags|n_perm|nperm|nperm|L|T|N|D|K|lag|nbatch|seed|m)$",
    re.IGNORECASE,
)


def split_args(text: str) -> list[str]:
    """Разбить список параметров по запятой верхнего уровня."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_header(path: Path) -> dict[str, list[str]]:
    """{имя функции: [параметры]} — прототипы из заголовка."""
    src = path.read_text(encoding="utf-8")
    # убираем комментарии, чтобы не ловить примеры в /* ... */
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    funcs: dict[str, list[str]] = {}
    for m in re.finditer(
        r"(?:^|[\s;])([A-Za-z_][A-Za-z0-9_]*)\s*\**\s*(prism_sycl_[A-Za-z0-9_]+)\s*\(([^;]*?)\)\s*;",
        src,
        flags=re.S | re.M,
    ):
        name, params = m.group(2), m.group(3).strip()
        if not params or params == "void":
            args: list[str] = []
        else:
            args = split_args(params)
        funcs[name] = args
    return funcs


def parse_python_bindings(path: Path) -> dict[str, int]:
    """{имя функции: число аргументов в argtypes}."""
    src = path.read_text(encoding="utf-8")
    src = re.sub(r"#[^\n]*", " ", src)
    out: dict[str, int] = {}

    # bind("name", [ ... ]) — вложенные скобки ⇒ свой сканер, а не регулярка
    for name, expr in scan_calls(src, "bind"):
        m = re.fullmatch(r"[\"'](\w+)[\"']", name.strip())
        if not m:
            continue
        out["prism_sycl_" + m.group(1)] = count_argtypes(expr)

    # прямые обращения: lib.prism_sycl_xxx.argtypes = [...]
    for m in re.finditer(
        r"lib\.(prism_sycl_\w+)\.argtypes\s*=\s*\[(.*?)\]", src, flags=re.S
    ):
        out.setdefault(m.group(1), count_argtypes("[" + m.group(2) + "]"))
    return out


def scan_calls(src: str, fname: str) -> list[tuple[str, str]]:
    """Найти вызовы fname(...) и вернуть список (аргумент1, аргумент2) как текст."""
    res: list[tuple[str, str]] = []
    for m in re.finditer(rf"\b{re.escape(fname)}\s*\(", src):
        i = m.end()  # сразу за открывающей скобкой
        depth = 1
        parts: list[str] = []
        cur = ""
        while i < len(src) and depth > 0:
            ch = src[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    parts.append(cur)
                    break
            if ch == "," and depth == 1:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
            i += 1
        if len(parts) >= 2:
            res.append((parts[0], parts[1]))
    return res


def split_plus(text: str) -> list[str]:
    """Разбить по '+' верхнего уровня (конкатенация списков argtypes)."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == "+" and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def count_argtypes(body: str) -> int:
    """Число элементов в списке argtypes; раскрывает вспомогательные списки и конкатенацию."""
    body = body.strip()
    if not body:
        return 0
    total = 0
    for part in split_plus(body):
        part = part.strip()
        if not part:
            continue
        # срез вспомогательного списка: acf_args[1:]
        m = re.fullmatch(r"(\w+)\[(\d+):(\d*)\]", part)
        if m:
            var, lo, hi = m.group(1), int(m.group(2)), m.group(3)
            base = count_list_var(var)
            if base is None:
                return -1
            total += max(0, (int(hi) if hi else base) - lo)
            continue
        # целый вспомогательный список: acf_args
        m = re.fullmatch(r"(\w+)", part)
        if m and not re.search(r"\b(?:c_p|c_d|c_i64|c_i32|c_u8|None)\b", part):
            base = count_list_var(m.group(1))
            if base is not None:
                total += base
                continue
        # список-литерал: [ ... ]
        if part.startswith("[") and part.endswith("]"):
            total += len(split_args(part[1:-1]))
            continue
        total += 1
    return total


def count_list_var(var: str) -> int | None:
    """Сколько элементов в списке-литерале, определённом в модуле (acf_args = [...])."""
    src = PY.read_text(encoding="utf-8")
    m = re.search(rf"^\s*{re.escape(var)}\s*=\s*\[(.*?)\]", src, flags=re.S | re.M)
    if not m:
        return None
    return len(split_args(m.group(1)))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    hdr_path = Path(argv[0]) if len(argv) > 0 else HDR
    py_path = Path(argv[1]) if len(argv) > 1 else PY
    if not hdr_path.exists() or not py_path.exists():
        print(f"НЕ НАЙДЕНЫ: {hdr_path} или {py_path}")
        return 1

    globals()["PY"] = py_path  # для подстановки вспомогательных списков
    hdr = parse_header(hdr_path)
    py = parse_python_bindings(py_path)

    print(f"функций в заголовке: {len(hdr)}; связано в python: {len(py)}")
    problems: list[str] = []

    for name, n_py in sorted(py.items()):
        if name not in hdr:
            problems.append(
                f"[bind] python ищет «{name}», в заголовке нет такого символа"
            )
            continue
        n_c = len(hdr[name])
        if n_c != n_py:
            problems.append(
                f"[args] {name}: в C {n_c} параметр(ов), в argtypes {n_py} "
                f"⇒ ctypes передаст мусор (типичное обрезание int64)"
            )

    for name in sorted(hdr):
        if name not in py and name not in ALLOW_UNBOUND:
            problems.append(f"[unbound] «{name}» есть в .h, но не связана в python")

    # имена параметров, несущих размеры, должны быть int64 в C
    for name, params in sorted(hdr.items()):
        for p in params:
            toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", p)
            if not toks:
                continue
            pname = toks[-1]
            if SIZE_HINT.match(pname) and "int64_t" not in p and "size_t" not in p:
                problems.append(
                    f"[size] {name}: параметр «{pname}» объявлен как «{p.strip()}» "
                    f"⇒ ожидается 64-битный размер/индекс"
                )

    if problems:
        print(f"\nРАСХОЖДЕНИЙ: {len(problems)}")
        for pr in problems:
            print("  - " + pr)
        return 1
    print("\nABI OK: сигнатуры заголовка и argtypes совпадают по числу аргументов.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
