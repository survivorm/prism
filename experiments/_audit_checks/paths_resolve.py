#!/usr/bin/env python
"""Аудит: пути, собранные от `__file__`, существуют ли после реорга experiments/ (d6ab76b).

Класс дефекта (найден на E-45b/E-46): скрипт собирает путь как
`os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "X")`.
Пока файл лежал в `experiments/`, это вело в `results/` репозитория; после переноса
в `experiments/e41_50/` тот же код ведёт в `experiments/results/`, которого нет.
Скрипт при этом не падает при импорте и не отличается ничем в git-blame: он просто
перестаёт воспроизводиться, а docstring продолжает обещать рабочую команду.

Что делает проверка: для каждого .py в experiments/ находит литералы
`dirname(abspath(__file__))` + список строковых сегментов и проверяет, существует ли
собранный путь. Плюс отдельно ищет `"..", ".."`-варианты. Не исполняет код.

Запуск: .venv/bin/python experiments/_audit_checks/paths_resolve.py
Артефакт: results/_audit_checks/paths_resolve.json
"""
from __future__ import annotations

import ast
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
OUT = os.path.join(ROOT, "results", "_audit_checks", "paths_resolve.json")

MARKERS = ("__file__",)


def _segments(node):
    """Список сегментов из str-констант внутри узла (в порядке появления)."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.append((sub.lineno, sub.value))
    out.sort()
    return [v for _l, v in out]


def _base_depth(node):
    """Сколько раз от каталога файла поднялись: dirname(dirname(abspath(__file__))) → 2.

    Первая версия этой проверки считала ровно один dirname и объявляла битым
    experiments/at8/test_at8_2_verdict.py, где стоит ТРИ dirname (каталог файла →
    experiments → корень). Считать надо глубину, иначе проверка врёт (её собственный
    самоаудит: false positive на at8).
    """
    src = ast.unparse(node)
    if "__file__" not in src:
        return 0
    return src.count("dirname") + src.count(".parent")


def scan(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = ast.unparse(node.func)
        if not (fname.endswith("os.path.join") or fname.endswith("join") or fname.endswith("Path")):
            continue
        args = node.args
        if not args:
            continue
        depth = _base_depth(args[0])
        if depth < 1:
            continue
        segs = []
        for a in args[1:]:
            segs.extend(_segments(a))
        if not segs:
            continue
        base = os.path.dirname(path)
        for _ in range(depth - 1):
            base = os.path.dirname(base)
        cand = os.path.join(base, *segs)
        found.append({"line": node.lineno, "depth": depth,
                      "expr": ast.unparse(node)[:200],
                      "resolved": os.path.relpath(cand, ROOT),
                      "exists": os.path.exists(cand), "segments": segs})
    return found


def main():
    res = {"checked": 0, "with_file_paths": 0, "broken": [], "broken_unique_targets": {}}
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "experiments")):
        if "__pycache__" in dirpath:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            res["checked"] += 1
            hits = scan(p)
            if not hits:
                continue
            res["with_file_paths"] += 1
            bad = [h for h in hits if not h["exists"]]
            if bad:
                rel = os.path.relpath(p, ROOT)
                res["broken"].append({"file": rel, "paths": bad})
                for h in bad:
                    tgt = h["resolved"]
                    full = os.path.join(ROOT, tgt)
                    # два разных класса: каталог не существует (перенос файла сломал путь)
                    # vs каталог есть, файла нет (артефакт не положен/не закоммичен)
                    h["kind"] = ("dir_missing" if not os.path.isdir(os.path.dirname(full))
                                 else "file_missing")
                    res["broken_unique_targets"].setdefault(tgt, []).append(
                        f"{rel}:{h['line']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"проверено .py: {res['checked']}, из них собирают пути от __file__: "
          f"{res['with_file_paths']}")
    print(f"файлов с НЕсуществующим путём: {len(res['broken'])}")
    kinds = {}
    for b in res["broken"]:
        for h in b["paths"]:
            kinds.setdefault(h["kind"], []).append(f"{b['file']}:{h['line']}")
    for k, v in sorted(kinds.items()):
        print(f"  класс {k}: {len(v)} ссылок в {len(set(x.split(':')[0] for x in v))} файлах")
    for tgt, users in sorted(res["broken_unique_targets"].items()):
        print(f"  {tgt}  ←  {len(users)} ссылок ({users[0]}{' …' if len(users) > 1 else ''})")
    print("сохранено:", OUT)


if __name__ == "__main__":
    main()
