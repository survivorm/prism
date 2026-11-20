"""G1: файл с тестовым именем обязан собирать хотя бы один тест.

Зачем (класс 6 аудита, `docs/strayed_experiments.md`): в репозитории 400+ скриптов и 11 тестов;
`experiments/at0/at0_7_tests.py` вообще не матчится дефолтным `python_files=test_*.py`, а
`experiments/e31_40/test_exp39f_math.py` содержит `def main()` вместо тестов — оба выглядят
аппаратом проверки и не являются им. Гейт обходит файлы ПО ИМЕНИ (не через сбор каталога),
поэтому видит оба.

Область: `experiments/` + `tools/`. `tests/` намеренно вне: там 363 теста и 26–39 с, это работа
существующей батареи, а не гейта (ТЗ §3.1).

Правило: `pytest <file> --collect-only -q`
  RC=0 и собрано ≥ 1 теста   → OK
  RC=5 (`no tests collected`) → DEAD_TEST
  RC=2 (`collection error`)   → DEAD_TEST
Файлы из allowlist пропускаются, но печатаются как `dead_allowlisted`.
Импорт torch/numpy при сборе разрешён (pytest импортирует модуль): бюджет 60 с на гейт.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AREAS = ("experiments", "tools")
GATE_ID = "G1"
TIMEOUT_S = 120


def test_named_files():
    for area in AREAS:
        for dirpath, dirs, names in os.walk(os.path.join(REPO, area)):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "build"}]
            for name in sorted(names):
                if name.endswith(".py") and (name.startswith("test_") or name.endswith("_tests.py")):
                    yield os.path.relpath(os.path.join(dirpath, name), REPO)


def load_allowlist(path=None):
    """[{gate, path, links, reason}] — путь и links; у G1 links не используется."""
    path = path or os.path.join(REPO, "tools", "gate_allowlist.txt")
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise SystemExit(f"gate_allowlist.txt: строка без табуляции: {line!r}")
            gate, rel = parts[0].strip(), parts[1].strip()
            meta, reason = {}, ""
            for extra in parts[2:]:
                if "=" in extra and not reason:
                    k, v = extra.split("=", 1)
                    meta[k.strip()] = v.strip()
                else:
                    reason = (reason + " " + extra).strip()
            entries.append({"gate": gate, "path": rel, "links": meta.get("links"), "reason": reason})
    return entries


def collect_one(rel):
    """(rc, n_collected) для одного пути."""
    cmd = [sys.executable, "-m", "pytest", rel, "--collect-only", "-q", "-p", "no:cacheprovider"]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return 124, 0
    out = p.stdout + p.stderr
    m = re.search(r"(\d+) tests? collected", out)
    return p.returncode, int(m.group(1)) if m else 0


def main():
    t0 = time.time()
    allowed = {e["path"] for e in load_allowlist() if e["gate"] == GATE_ID}
    files = list(test_named_files())
    dead, dead_allowed, collected = [], [], 0
    for rel in files:
        rc, n = collect_one(rel)
        if rc == 0 and n >= 1:
            collected += n
        elif rel in allowed:
            dead_allowed.append(rel)
        else:
            dead.append((rel, rc, n))
    for rel, rc, n in dead:
        reason = "no tests collected" if rc == 5 else ("collection error" if rc == 2 else f"rc={rc}")
        print(f"DEAD_TEST {rel}  ({reason})")
    for rel in dead_allowed:
        print(f"DEAD_TEST {rel}  (allowlisted)")
    seen = set(dead_allowed) | {r for r, _, _ in dead}
    unused = sorted(allowed - seen)
    for rel in unused:
        print(f"ALLOWLIST_UNUSED {rel}  (G1)")
    print(f"G1: test_named={len(files)} collected={collected} dead={len(dead)} "
          f"dead_allowlisted={len(dead_allowed)} allowlist_unused={len(unused)} "
          f"t={time.time() - t0:.1f}s")
    return 1 if dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
