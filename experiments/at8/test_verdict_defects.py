# -*- coding: utf-8 -*-
"""AT8_2 (T8.2) — дефекты машины вердиктов (внешнее ревью, docs/AT8_2_review.md §E)
и их фиксация машиной v3.

Файл предложен ревьюером (дисклеймер: «числа выведены аналитически/вручную, не
прогоном»); механические дефекты оригинала исправлены (не выполнялся): сигнатура
mk() (keywords), _fit(lr, n_dose) — второй аргумент обязателен. Тесты фиксируют:
- механизмы, оставшиеся в коде (A2: ветка DIFFUSION жива для hand-crafted оси с
  нулевой медианой; из classify_cell недостижима — лемма + популяция v3);
- зафиксированное поведение v3 (A1: гейт по recovered-парам; A3: ratio=None;
  B6: no_effect из оси B; C1: статус по rounded pert0).
Запуск: .venv/bin/python -m pytest experiments/at8/test_verdict_defects.py
"""
import json
import os
import sys

import numpy as np, pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AT8_2 import (classify_cell, _ratio_lr, _fit, compute_verdict,
                   axis_B, N_DOSE, EPS_MIN)


def mk(doc, n, W, T, ik, ck, dip_global, dip_control_whole):
    return classify_cell(doc, n, W, T, dip_global, dip_control_whole, ck, ik)


def test_A2_diffusion_machinery_from_null_at_small_dose():
    """A2 (механизм, документирован): нулевая медиана на малой дозе -> ratio=inf;
    4-точечный fit проходит пороги b>0 и r2>=0.6 — машина СПОСОБНА напечатать
    DIFFUSION из «нет эффекта на дозе 2» (контрпример ревью: b=3.8174, r2=0.6214,
    порог 0.6 пройден с зазором 0.021)."""
    lr = {2: 0.0, 4: 39.0, 8: 63.0, 16: 64.0}
    assert _ratio_lr(lr) == float("inf")
    b, a, r2 = _fit(lr, N_DOSE)
    assert b == 3.8174 and r2 == 0.6214
    assert b > 0 and r2 >= 0.6


def test_A2b_diffusion_branch_alive_for_handcrafted_axis():
    """A2 (ветка жива): hand-crafted ось с нулевой медианой + клетки, закрывающие
    все гейты v3 (>=6 recovered-пар, majority < 0.5, ne_share <= 0.5) — машина
    печатает DIFFUSION. Из classify_cell такое недостижимо (test_A2c лемма +
    v3-популяция исключает no_effect-нули из медианы) — защита: популяция + гейты."""
    lr = {2: 0.0, 4: 39.0, 8: 63.0, 16: 64.0}
    cells = {}
    for k in range(24):
        doc, n = f"d{k % 6}", [2, 4, 8, 16][k // 6]
        cells[f"{doc}|n{n}|W64"] = dict(doc=doc, n=n, W=64, l_rec=64,
                                        rec_status="recovered", pert0=0.02,
                                        D_max=64, to_global=False,
                                        residual_ctrl=0.02, residual_global=0.03)
    b, a, r2 = _fit(lr, N_DOSE)
    axA = {"W64": dict(lrec=lr, ratio=float("inf"), ratio_inf=True, b=b, r2=r2)}
    v, _, diag = compute_verdict(axA, {"frac_to_global": 0.1}, cells)
    assert v == "DIFFUSION" and diag["n_rec_pairs"] == 24


def test_A2c_lemma_no_zero_recovery_from_cells():
    """Лемма ревью (A1/A2): recovered с l_rec=0 невозможен (ветка входит при
    pert0>=eps>0, и pert0 <= 0.25*pert0 ложно); значит ВСЕ нулевые l_rec — это
    no_effect. Проверено на реальном store (v2-артефакт) + синтетике."""
    claim = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "results", "AT8",
        "AT8_2_RECOVERY_v2.json")))
    for c in claim["cells"].values():
        if c.get("rec_status") in ("recovered", "not_recovered"):
            assert c["l_rec"] > 0, c
        if c.get("l_rec") == 0:
            assert c["rec_status"] == "no_effect", c
    # синтетика: pert0 ровно на пороге и выше — recovered всегда l_rec >= 1
    T, W = 256, 64
    D = T - T // 2 - W
    for ik0 in (-0.210001, -0.22, -0.24):
        ck = [-0.2] * (D + 1)
        ik = [ik0] + [-0.2] * D
        c = classify_cell("d", 8, W, T, -0.2, -0.2, ck, ik)
        if c.get("rec_status") == "recovered":
            assert c["l_rec"] >= 1, c


def test_A3_missing_dose16_is_none_not_zero():
    """A3 (фикс): отсутствующая доза -> ratio=None (не 0.0) — отсутствие
    измерения не является положительным свидетельством упругости."""
    assert _ratio_lr({2: 0.0, 4: 39.0, 8: 63.0, 16: None}) is None
    assert _ratio_lr({2: None, 4: 39.0, 8: 63.0, 16: 64.0}) is None


def test_A3_pipeline_missing_dose_not_resolved():
    """A3 (pipeline): ось с l16=None -> ratio=None -> NOT_RESOLVED
    («недостаточно доз»), даже если остальные условия выглядят упругими."""
    cells = {}
    for k in range(24):
        doc, n = f"d{k % 6}", [2, 4, 8][k // 8]
        cells[f"{doc}|n{n}|W64"] = dict(doc=doc, n=n, W=64, l_rec=5,
                                        rec_status="recovered", D_max=64,
                                        to_global=False)
    axA = {"W64": dict(lrec={2: 5.0, 4: 5.0, 8: 5.0, 16: None},
                       ratio=None, ratio_inf=False, b=0.0, a=5.0, r2=None)}
    v, det, _ = compute_verdict(axA, {"frac_to_global": 0.1}, cells)
    assert v == "NOT_RESOLVED" and "недостаточно доз" in det, (v, det)


def test_B1_recovered_cell_cannot_vote_to_global():
    """B1 (механизм, частично): восстановившаяся клетка с крошечным res_ctrl
    голосует to_global=0 — ось B получает структурные нули от recoveries.
    (Полное тождество «Всегда 0» неверно — 3/29 recovered на store проголосовали
    1 при малом pert0: см. AT8_2_review_r1_answers.md, спор по B1.)"""
    T, W = 256, 64
    D = T - T // 2 - W
    ck = [-0.160] * (D + 1)
    ik = [-0.160] * (D + 1)
    ik[0] = -0.140                               # pert0 = 0.020 >= eps
    ik[D] = -0.162                               # res_ctrl = 0.002 <= 0.25*pert0
    c = mk("d", 8, W, T, ik, ck, -0.050, -0.160)
    assert c["rec_status"] == "recovered" and c["to_global"] is False


def test_B6_no_effect_cell_votes_but_excluded_from_axis_B():
    """B6: no_effect-клетка голосует to_global=1 на основании дальнего отклонения
    (больше pert0!) — внутренне несогласованно; фикс v3: такие клетки исключены
    из популяции оси B (axis_B), legacy-чтение — diagnostic."""
    T, W = 256, 64
    D = T - T // 2 - W
    ck = [-0.160] * (D + 1)
    ik = [-0.160] * (D + 1)
    ik[0] = -0.165                               # pert0 = 0.005 < eps -> no_effect
    ik[D] = -0.100                               # res_ctrl = 0.060 (больше pert0!)
    c = mk("d", 8, W, T, ik, ck, -0.090, -0.160)
    assert c["rec_status"] == "no_effect" and c["to_global"] is True
    cells = {"a": c,
             "b": dict(doc="b", n=8, W=16, D_max=112, rec_status="recovered",
                       l_rec=3, to_global=False)}
    axB = axis_B(cells)
    assert axB["frac_to_global"] == 0.0          # no_effect исключён (B6)
    assert axB["frac_to_global_legacy"] == 0.5   # старое чтение — diagnostic


def test_C1_status_decided_on_rounded_pert0():
    """C1 (фикс): решение и публикация на одной точности. pert0 = 0.0099996
    -> публикуется 0.01 -> статус решается по 0.01 (НЕ no_effect). v1/v2 решали
    по сырому 0.0099996 < 0.01 -> артефакт противоречил сам себе."""
    T, W = 256, 64
    D = T - T // 2 - W
    ck = [-0.160] * (D + 1)
    ik = [-0.160] * (D + 1)
    ik[0] = -0.1500004
    c = mk("d", 8, W, T, ik, ck, -0.16, -0.16)
    assert c["pert0"] == 0.01 and c["rec_status"] == "recovered" \
        and c["l_rec"] == 1, c


def test_A1_gate_counts_only_recovered_pairs():
    """A1 (фикс): гейт «мало пар» считает (doc,n) с rec_status=='recovered' —
    цензурированные (l_rec=D_max — провал) не засчитываются восстановлением.
    15 not_recovered + 5 recovered на W64: v1/v2 читали l_rec>0 (20 пар, гейт
    проходил -> ELASTIC); v3: 5 < 6 -> NOT_RESOLVED."""
    cells = {}
    for k in range(20):
        doc, n = f"d{k % 5}", [2, 4, 8, 16][k // 5]
        cells[f"{doc}|n{n}|W64"] = dict(doc=doc, n=n, W=64, l_rec=64,
                                        rec_status="not_recovered" if k < 15
                                        else "recovered")
    lr = {2: 64.0, 4: 64.0, 8: 64.0, 16: 64.0}
    axA = {"W64": dict(lrec=lr, ratio=1.0, ratio_inf=False, b=0.0, r2=0.0)}
    v, det, diag = compute_verdict(axA, {"frac_to_global": 0.1}, cells)
    assert diag["n_rec_pairs"] == 5
    assert v == "NOT_RESOLVED" and "мало восстановленных пар" in det, (v, det)


def test_A4_majority_gate_on_perturbed_population():
    """A4 (фикс): majority-гейт — по возмущённой популяции {pert0>=eps} (no_effect
    не могут быть not_recovered по построению). Режим реального store: 55 not_rec
    + 40 recovered + 21 no_effect: старое чтение 55/116=0.474 < 0.5 (промах 3 п.п.
    — вердикт ELASTIC держался на нём); v3: 55/95=0.579 > 0.5 -> NOT_RESOLVED."""
    cells = {}
    i = 0
    for st in ["not_recovered"] * 55 + ["recovered"] * 40 + ["no_effect"] * 21:
        doc, n, W = f"d{i % 12}", N_DOSE[i % 4], (8, 16, 32, 64)[i % 4]
        cells[f"{doc}|n{n}|W{W}|{i}"] = dict(
            doc=doc, n=n, W=W, D_max=64, l_rec=64 if st == "not_recovered"
            else (0 if st == "no_effect" else 5), rec_status=st)
        i += 1
    axA = {"W64": dict(lrec={2: 5.0, 4: 5.0, 8: 5.0, 16: 5.0}, ratio=1.0,
                       ratio_inf=False, b=0.0, a=5.0, r2=0.0)}
    v, det, diag = compute_verdict(axA, {"frac_to_global": 0.1}, cells)
    # B3 (v3): W=8 исключён из популяции гейта: 41/71 = 0.5775 (старое чтение 55/116)
    assert diag["nr_share_perturbed"] == 0.5775
    assert diag["notrecovered_share_full"] == 0.4741
    assert v == "NOT_RESOLVED" and "большинство not_recovered" in det, (v, det)


def test_no_effect_below_resolution_outcome():
    """NO_EFFECT (ревью §5): большинство клеток оси — no_effect, но восстановленных
    пар достаточно (>=6) и majority < 0.5 — машина выражает фактическое состояние
    (effect below resolution), которого v1/v2 не умели (только NOT_RESOLVED)."""
    cells = {}
    for di in range(6):
        for j, n in enumerate(N_DOSE):
            rec = di < 3 and n in (2, 4, 8)
            cells[f"wiki_{di}|n{n}|W64"] = dict(
                doc=f"wiki_{di}", n=n, W=64, D_max=64,
                l_rec=5 if rec else 0,
                rec_status="recovered" if rec else "no_effect")
    axA = {"W64": dict(lrec={2: 5.0, 4: 5.0, 8: 5.0, 16: None}, ratio=None,
                       ratio_inf=False, b=None, a=None, r2=None)}
    v, det, diag = compute_verdict(axA, {"frac_to_global": None}, cells)
    assert v == "NO_EFFECT" and "ниже разрешения" in det, (v, det)
    assert diag["ne_share_axis"] == 0.625 and diag["n_rec_pairs"] == 9


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
