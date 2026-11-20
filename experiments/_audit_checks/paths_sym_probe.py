"""Символический скан путей от `__file__` — АЛИАС-подкласс, невидимый `paths_resolve.py`.

Зачем: `paths_resolve.py` берёт `args[0]` вызова `join(...)` и считает в нём `dirname`;
форма `HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")`
даёт `depth = 0` и **молча пропускается**. Основатель класса 6 (E-45b/E-46) — ровно такая форма
(`exp45b_analysis.py:28–29`), поэтому слепота принципиальна, а не косметическая.

Как: AST-интерпретация выражений, встречающих `__file__`, в символическую строку, где каталог
файла обозначен `FILE`; поддерживаются `dirname`, `abspath`, `join`, `Path(...)`, `.parent`,
`/`-оператор, константы и имена (среда собирается за 3 прохода — имена могут быть объявлены
ниже использования). Путь разрешается от реального каталога файла и проверяется `os.path.exists`.

Границы (читать буквально):
  * это **дополнение**, а не замена: inline-форма покрывается `paths_resolve.py`, здесь она
    тоже видна, но `checked`/`with_file_paths` считаются только по файлам с именованным алиасом;
  * среда имён плоская (одна на файл): локальная переменная с тем же именем, что модульный
    алиас, перезапишет его — это может дать как ложный плюс, так и пропуск;
  * динамические пути (`getattr`, f-строки, значения из JSON) не разрешаются — они отбрасываются;
  * `file_missing` тут означает «каталог есть, артефакта нет», то есть артефакт не коммитился,
    а не сломанный путь.
"""
from __future__ import annotations
import ast
import json
import os

# Символический корень репо. Раньше здесь был один компонент (\x00FILE), и вложенный
# `dirname` терялся: os.path.dirname("\x00FILE") == "" — файлы вида
# `_ROOT = dirname(dirname(abspath(__file__)))` выпадали из скана целиком (нашёл ревьюер r2).
# Теперь `__file__` подставляется как <SENT>/<путь файла от корня репо>, поэтому dirname/join
# работают по настоящей глубине, а разрешение = срезать SENT.
SENT = "/\x00REPO"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "_audit_checks", "paths_sym_scan.json")


def ev(node, env):
    """Символическое значение выражения или None, если выражение не разрешается."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return env["__file__"]                     # подставляется scan_file (см. SENT)
        return env.get(node.id)
    if isinstance(node, ast.Attribute):
        if node.attr in ("parent", "parents"):
            v = ev(node.value, env)
            return None if v is None else os.path.dirname(v)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = ev(node.left, env), ev(node.right, env)
        if left is None or right is None:
            return None
        return os.path.join(left, right)
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
        if fname.endswith("pardir") or fname.endswith("curdir"):
            return None
        if (fname.endswith("resolve") or fname.endswith("absolute")) and not args:
            return ev(node.func.value, env)          # Path(__file__).resolve()
        if fname.endswith("Path") and len(args) == 1:
            return args[0]
    return None


def build_env(tree, seed=None):
    env = dict(seed or {})
    for _ in range(3):                                # имена могут быть объявлены ниже
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            v = ev(node.value, env)
            if v is not None:
                env[tgt.id] = v
    return env


def scan_file(path):
    rel = os.path.relpath(path, ROOT)
    if os.path.abspath(path) == os.path.abspath(__file__):
        return None, []                               # не проверять себя: собственные OUT/SENT-литералы
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except SyntaxError:
        return None, []
    seed = {"__file__": os.path.join(SENT, rel)}
    env = build_env(tree, seed)
    if not any(SENT in v for v in env.values()):
        return None, []                               # файл без алиаса от __file__ — не наш подкласс
    out = []
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
            continue                              # ушло выше корня репо — не разрешаем
        cand = os.path.normpath(os.path.join(ROOT, raw[len(SENT) + 1:] or "."))
        is_dir = os.path.isdir(os.path.dirname(cand))
        out.append({"file": rel, "line": node.lineno,
                    "resolved": os.path.relpath(cand, ROOT),
                    "exists": os.path.exists(cand),
                    "kind": "dir_missing" if not is_dir else "file_missing",
                    "expr": ast.unparse(node)[:160]})
    return rel, out


def main():
    refs, files = [], 0
    for dirpath, _dirs, names in os.walk(os.path.join(ROOT, "experiments")):
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            rel, got = scan_file(os.path.join(dirpath, name))
            if rel is None:
                continue
            files += 1
            refs.extend(got)
    seen, uniq = set(), []
    for r in refs:
        key = (r["file"], r["line"], r["resolved"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    bad = [r for r in uniq if not r["exists"]]
    by_kind = {}
    for r in bad:
        by_kind.setdefault(r["kind"], []).append(r)
    res = {
        "alias_files": files,
        "refs": len(uniq),
        "broken_refs": len(bad),
        "broken_files": len({r["file"] for r in bad}),
        "by_kind": {k: {"refs": len(v), "files": len({r["file"] for r in v})}
                    for k, v in by_kind.items()},
        "broken": sorted(bad, key=lambda r: (r["kind"], r["file"], r["line"])),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"файлов с алиасом от __file__: {files}, ссылок: {len(uniq)}, "
          f"битых: {len(bad)} в {res['broken_files']} файлах")
    for k, v in sorted(res["by_kind"].items()):
        print(f"  {k}: {v['refs']} ссылок в {v['files']} файлах")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
