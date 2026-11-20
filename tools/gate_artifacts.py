"""G6: артефакт обязан описывать сам себя.

Зачем (классы 3/4 аудита, `docs/strayed_experiments.md`): в `results/` лежит артефакт с
оборванной записью, закоммиченный в git (`results/AT2e/at2e_smoke_r1.json`), и 5 файлов с
нефинитными float; состояния `{"a": NaN}` в git JSON не должно появляться вовсе.

Канон выборки (ТЗ §3.3): `results/**/*.json` + `results/**/*.json.gz`, **без**
`results/**/store/**` (адресные блобы кэша: 425.8 МБ, у них нет `meta.n_rows` и нет смысла в
`isfinite` — их судьба в G7 итерации 3). `.jsonl` и `*.json.tar.gz` — вне scope (line-delimited и
архивы, а не документы); широкий glob `*.json*` дал бы 18 файлов, 17 из которых — ложный INVALID.

Проверки (RC=1 при любом попадании):
  1. файл не парсится (`json.load` / `gzip` + `json.load`);
  2. в разобранной структуре есть `float`, для которого `math.isfinite` ложно;
  3. `meta.n_rows` есть и не равен `len(rows)` / `len(positions)` / `len(per_position)`.

Сводка печатает фактическое число файлов — эталонов в тексте нет (иначе строка устаревает в день
внедрения).
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GATE_ID = "G6"


def load_allowlist(path=None):
    path = path or os.path.join(REPO, "tools", "gate_allowlist.txt")
    out = set()
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
            if parts[0].strip() == GATE_ID:
                out.add(parts[1].strip())
    return out


def canon():
    files = []
    for pattern in ("results/**/*.json", "results/**/*.json.gz"):
        files.extend(glob.glob(os.path.join(REPO, pattern), recursive=True))
    return sorted({f for f in files if "/store/" not in f.replace(os.sep, "/")})


def nonfinite_values(obj, acc):
    if isinstance(obj, float):
        if not math.isfinite(obj):
            acc[0] += 1
    elif isinstance(obj, dict):
        for v in obj.values():
            nonfinite_values(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            nonfinite_values(v, acc)


def nrows_mismatch(obj):
    if not isinstance(obj, dict):
        return None
    meta = obj.get("meta")
    if not isinstance(meta, dict) or "n_rows" not in meta:
        return None
    want = meta["n_rows"]
    for key in ("rows", "positions", "per_position"):
        seq = obj.get(key)
        if isinstance(seq, (list, dict)) and len(seq) != want:
            return f"meta.n_rows={want} != len({key})={len(seq)}"
    return None


def main():
    t0 = time.time()
    allowed = load_allowlist()
    files = canon()
    total_mb = sum(os.path.getsize(f) for f in files) / 1e6
    invalid, nonfinite_files, nrows_bad, allowed_hits = [], [], [], []
    for path in files:
        rel = os.path.relpath(path, REPO)
        try:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:                       # noqa: BLE001 — важен факт, не тип
            if rel in allowed:
                allowed_hits.append(rel)
            invalid.append((rel, type(exc).__name__))
            continue
        acc = [0]
        nonfinite_values(data, acc)
        if acc[0]:
            if rel in allowed:
                allowed_hits.append(rel)
            nonfinite_files.append((rel, acc[0]))
        bad = nrows_mismatch(data)
        if bad:
            nrows_bad.append((rel, bad))
    for rel, exc in invalid:
        print(f"INVALID {rel}  ({exc})")
    for rel, n in nonfinite_files:
        print(f"NONFINITE {rel}  ({n} значений)")
    for rel, why in nrows_bad:
        print(f"NROWS {rel}  ({why})")
    unused = sorted(allowed - set(allowed_hits))
    for rel in unused:
        print(f"ALLOWLIST_UNUSED {rel}  (G6)")
    failures = [rel for rel, _ in invalid if rel not in allowed]
    failures += [rel for rel, _ in nonfinite_files if rel not in allowed]
    failures += [rel for rel, _ in nrows_bad]
    print(f"G6: canon_files={len(files)} canon_mb={total_mb:.1f} invalid={len(invalid)} "
          f"nonfinite_files={len(nonfinite_files)} nrows_mismatch={len(nrows_bad)} "
          f"allowlisted={len(allowed_hits)} allowlist_unused={len(unused)} "
          f"t={time.time() - t0:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
