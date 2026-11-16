# -*- coding: utf-8 -*-
"""Exec-тест resume_plan (AGENTS п.2: designed-сценарий — exec точного фрагмента).

Минимальный store (без модели): wiki_0 — полный (ctrl + 4 дозы), wiki_1 — только ctrl,
wiki_2..7 — пустой. Проверяет решение «что пересчитывать» на ТОМ же фрагменте,
который вызывает main() (at8_2_analysis.resume_plan).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from at8_2_analysis import N_DOSE, resume_plan

names = [f"wiki_{i}" for i in range(8)]
res = {
    "wiki_0__ctrl": dict(h23_sha="a" * 16),
    "wiki_0__n2": dict(), "wiki_0__n4": dict(), "wiki_0__n8": dict(), "wiki_0__n16": dict(),
    "wiki_1__ctrl": dict(h23_sha="b" * 16),
}
plan = resume_plan(names, N_DOSE, res)

# wiki_0: всё в store → 0 пересчётов; wiki_1: ctrl в store, 4 дозы — пересчёт;
# wiki_2..7: ctrl + 4 дозы — пересчёт.
assert plan["ctrl_resume"]["wiki_0"] is True
assert plan["ctrl_resume"]["wiki_1"] is True
assert all(plan["ctrl_resume"][f"wiki_{i}"] is False for i in range(2, 8))
assert all(plan["dose_resume"][("wiki_0", n)] is True for n in N_DOSE)
assert all(plan["dose_resume"][("wiki_1", n)] is False for n in N_DOSE)
assert all(plan["dose_resume"][("wiki_5", n)] is False for n in N_DOSE)
# 6 записей в store (2 ctrl + 4 дозы) → n_resume=6
assert plan["n_resume"] == 6, plan["n_resume"]
# fresh run: пустой store → всё пересчитывается
p0 = resume_plan(names, N_DOSE, {})
assert p0["n_resume"] == 0 and all(v is False for v in p0["ctrl_resume"].values()) \
    and all(v is False for v in p0["dose_resume"].values())

print("RESUME_TEST_PASS: resume_plan на синтетическом store + fresh run (exec точного фрагмента)")
