#!/usr/bin/env python3
"""baseline_ids.py — выгрузка всех стабильных ID из paper.md (ТЗ v1.1, шаг 0).

Номера секций НЕ входят (нестабильны). Сохраняет baseline_ids.json.
"""
import json
import re
import sys


def main(path: str) -> int:
    text = open(path, encoding="utf-8").read()
    ids: dict[str, set[str]] = {
        "E": set(re.findall(r"\bE-(?:\d+(?:\.\d+)?[a-d]?(?:\s*Semantika)?)", text)),
        "L": set(re.findall(r"\[L-[a-z\-]+\]", text)),
        "H": set(re.findall(r"\bH[1-8]\b", text)),
        "C": set(re.findall(r"\bC\.\d\b", text)),
        "D": set(re.findall(r"\[D-[a-z\-]+\]", text)),
        "M": set(re.findall(r"\[M-\d\]", text)),
        "anchors": set(re.findall(r"\b[DE]\.\d\b", text)),
    }
    out = {k: sorted(v) for k, v in ids.items()}
    total = sum(len(v) for v in out.values())
    with open("baseline_ids.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"{total} unique ids written to baseline_ids.json")
    for k, v in out.items():
        print(f"  {k} ({len(v)}): {', '.join(v[:12])}{'...' if len(v) > 12 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/paper.md"))
