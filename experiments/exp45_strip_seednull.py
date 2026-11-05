"""Вырезать строки seednull_* из partial, чтобы --resume пересчитал их заново.

Нужно потому, что σ первого iid-прохода калибровался неверно (дрейф вышел 0.0855 вместо 0.12),
а механизм --resume пропускает всё, что уже лежит в partial["edits"] по имени.
"""
import json
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                 "exp45_damage_vs_structure.partial.json")
d = json.load(open(P))
n0 = len(d["edits"])
d["edits"] = [e for e in d["edits"] if not e["name"].startswith("seednull_")]
with open(P, "w") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print(f"вырезано строк seednull_*: {n0 - len(d['edits'])}, осталось {len(d['edits'])}")
