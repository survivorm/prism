"""G2: путь, собранный от `__file__`, обязан указывать в существующее место.

Зачем (класс 6 аудита, `docs/strayed_experiments.md`): коммит `d6ab76b` («Reorg experiments/ по
сериям») перенёс скрипты на уровень глубже, а пути в них собраны от `__file__`; после реорга
`experiments/e41_50/` + `../results` = несуществующий `experiments/results/`, поэтому скрипты
E-45b/E-46 не воспроизводят собственные артефакты, а docstring'и продолжают обещать рабочую
команду. Ни один тест этого не ловил.

Метод: AST-интерпретация без исполнения кода. `__file__` подставляется как абсолютный
символический корень `SENT/<путь файла от корня репо>`, поэтому `dirname`/`abspath`/`join`/
`Path(...).parent`/`/`-оператор считают **настоящую** глубину (прежняя версия с одним компонентом
`"\x00FILE"` теряла вложенный `dirname`: `os.path.dirname("\x00FILE") == ""` — нашёл внешний
ревьюер r2, воспроизведя импортом).

Классы (по ТЗ §4):
  DIR_MISSING  — каталог-родитель цели не существует ⇒ RC=1 (объективное последствие реорга);
  FILE_MISSING — родитель есть, файла нет ⇒ печатается как INFO, RC не меняет (скрипт вправе
                 писать артефакт, которого ещё нет в git).

Allowlist (`tools/gate_allowlist.txt`, gate=G2): путь файла + `links=<K>`; если фактическое число
`DIR_MISSING`-ссылок файла ≠ K — RC=1 (иначе одна запись легализует и завтрашние поломки).

Границы: `getattr`, f-строки, пути из JSON/env/CLI-аргументов не разрешаются — такие ссылки
отбрасываются, а не угадываются. Это нижняя граница класса, а не его размер.
"""
from __future__ import annotations

import ast
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SENT = "/\x00REPO"
GATE_ID = "G2"
AREAS = ("experiments",)          # область: скрипты экспериментов (дизайн-тесты сужают/меняют)
SKIP_DIRS = {".git", "__pycache__", "build", ".venv", "node_modules"}


def load_allowlist(path=None):
    path = path or os.path.join(REPO, "tools", "gate_allowlist.txt")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise SystemExit(f"gate_allowlist.txt: строка без табуляции: {line!r}")
            if parts[0].strip() != GATE_ID:
                continue
            links = None
            for extra in parts[2:]:
                if extra.strip().startswith("links="):
                    links = int(extra.split("=", 1)[1])
            out[parts[1].strip()] = links
    return out


def ev(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr in ("parent", "parents"):
        base = ev(node.value, env)
        return None if base is None else os.path.dirname(base)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = ev(node.left, env), ev(node.right, env)
        return None if (left is None or right is None) else os.path.join(left, right)
    if isinstance(node, ast.Call):
        fname = ast.unparse(node.func)
        args = [ev(a, env) for a in node.args]
        if any(a is None for a in args):
            return None
        if fname.endswith("dirname") and len(args) == 1:
            return os.path.dirname(args[0])
        if fname.endswith("abspath") and len(args) == 1:
            return args[0]
        if fname.endswith("join") and args:
            return os.path.join(*args)
        if (fname.endswith("resolve") or fname.endswith("absolute")) and not args:
            return ev(node.func.value, env)
        if fname.endswith("Path") and len(args) == 1:
            return args[0]
    return None


def build_env(tree, seed):
    env = dict(seed)
    for _ in range(3):                    # имена могут быть объявлены ниже использования
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                val = ev(node.value, env)
                if val is not None:
                    env[tgt.id] = val
    return env


def scan_file(path):
    """Список ссылок файла: {line, resolved, kind, expr}."""
    rel = os.path.relpath(path, REPO)
    if os.path.abspath(path) == os.path.abspath(__file__):
        return []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return []
    env = build_env(tree, {"__file__": os.path.join(SENT, rel)})
    refs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fname = ast.unparse(node.func)
        if not (fname.endswith("join") or fname.endswith("Path")):
            continue
        args = [ev(a, env) for a in node.args]
        if any(a is None for a in args) or not any(SENT in a for a in args):
            continue
        raw = os.path.normpath(os.path.join(*args))
        if not (raw == SENT or raw.startswith(SENT + "/")):
            continue                       # ушло выше корня репо
        target = os.path.normpath(os.path.join(REPO, raw[len(SENT) + 1:] or "."))
        if os.path.exists(target):
            continue
        parent_missing = not os.path.isdir(os.path.dirname(target))
        refs.append({"line": node.lineno,
                     "resolved": os.path.relpath(target, REPO),
                     "kind": "dir_missing" if parent_missing else "file_missing",
                     "expr": ast.unparse(node)[:160]})
    return refs


def scan_repo():
    found = {}
    for area in AREAS:
        for dirpath, dirs, names in os.walk(os.path.join(REPO, area)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(names):
                if not name.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), REPO)
                refs = scan_file(os.path.join(dirpath, name))
                if refs:
                    found[rel] = refs
    return found


def main():
    t0 = time.time()
    allowed = load_allowlist()
    found = scan_repo()
    dir_links = sum(1 for refs in found.values() for r in refs if r["kind"] == "dir_missing")
    file_links = sum(1 for refs in found.values() for r in refs if r["kind"] == "file_missing")
    covered = 0
    bad_rc = []
    for rel in sorted(found):
        refs = found[rel]
        long_refs = [r for r in refs if r["kind"] == "dir_missing"]
        for r in long_refs:
            print(f"DIR_MISSING {rel}:{r['line']} -> {r['resolved']}")
        for r in refs:
            if r["kind"] == "file_missing":
                print(f"INFO FILE_MISSING {rel}:{r['line']} -> {r['resolved']}")
        if rel in allowed:
            covered += len(long_refs)
            want = allowed[rel]
            if want is not None and want != len(long_refs):
                print(f"ALLOWLIST_STALE {rel}  (links={want}, фактически {len(long_refs)})")
                bad_rc.append(rel)
    blocked = [rel for rel in found
               if rel not in allowed and any(r["kind"] == "dir_missing" for r in found[rel])]
    for rel in sorted(set(allowed) - set(found)):
        print(f"ALLOWLIST_UNUSED {rel}  (G2)")
    print(f"G2: files_with_refs={len(found)} dir_missing_links={dir_links} "
          f"dir_missing_files={sum(1 for rel in found if any(r['kind'] == 'dir_missing' for r in found[rel]))} "
          f"file_missing_links={file_links} allowlisted_links={covered} "
          f"allowlist_entries={len(allowed)} t={time.time() - t0:.1f}s")
    return 1 if (blocked or bad_rc) else 0


if __name__ == "__main__":
    raise SystemExit(main())
