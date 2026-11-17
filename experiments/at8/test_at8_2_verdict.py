# -*- coding: utf-8 -*-
"""Exec-тест машины v3 (AGENTS п.2: designed-сценарии — exec точного фрагмента).

Синтетические клетки (без модели) покрывают: DIFFUSION, DILUTION, ELASTIC,
NO_EFFECT (below_resolution), NOT_RESOLVED (три гейта машины v3: «мало
восстановленных пар» по rec_status=='recovered' — A1; majority по возмущённой
популяции — A4; ratio=None — A3) + воспроизведение вердикта на реальном store
через v2-артефакт (машина v3: NOT_RESOLVED — P0-проверки ревью).

Ветка «мало пар» на старых чтениях (l_rec>0 / no_effect-доля) — в
test_verdict_defects.py (дефекты v1/v2 + механизмы v3).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AT8_2 import (axis_A, axis_B, compute_verdict, N_DOSE, W_AXIS)


def mk(doc, n, W, D_max, l_rec, rec_status, to_global=None):
    c = dict(doc=doc, n=n, W=W, D_max=D_max, rec_status=rec_status, l_rec=l_rec,
             W8_noisy_flag=False)
    if to_global is not None:
        c["to_global"] = to_global
    return c


def cells8(nmap_w64, to_global_all):
    """8 доков × 4 дозы; w64: {n: l_rec} (D_max=64, recovered); остальные W — l_rec=5."""
    out = {}
    for di in range(8):
        for n in N_DOSE:
            for W in (8, 16, 32, 64):
                lr = nmap_w64[n] if W == 64 else 5
                out[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr,
                                                  "recovered", to_global_all)
    return out


# 1) ELASTIC: lrec W64 ≈ константа, to_global мало (машина v3: возмущённая популяция)
cells = cells8({2: 10, 4: 12, 8: 11, 16: 13}, False)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "ELASTIC", (v, det)
assert diag["n_pos"] == 4 and diag["n_rec_pairs"] == 32  # 8 доков × 4 дозы
assert diag["notrecovered_share_full"] == 0.0

# 2) DIFFUSION: lrec растёт с дозой, ratio>=4, r2>=0.6
cells = cells8({2: 2, 4: 4, 8: 8, 16: 16}, False)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "DIFFUSION", (v, det, axA["W64"])

# 3) DILUTION: to_global > 0.6 (ось A — умеренная, не DIFFUSION)
cells = cells8({2: 10, 4: 12, 8: 11, 16: 13}, True)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "DILUTION", (v, det)

# 4) NOT_RESOLVED гейт 1 (A1): мало ВОССТАНОВЛЕННЫХ пар (doc,n) на W=64.
#    5 восстановленных пар (5 < 6), остальные — not_recovered (не no_effect —
#    NO_EFFECT-ветка не срабатывает), f_tg=None (to_global не задан).
cells = {}
for di in range(5):
    for n in N_DOSE:
        rec = (di == 0) or (di == 1 and n == 2)
        lr = 10 if rec else 18
        st = "recovered" if rec else "not_recovered"
        for W in (8, 16, 32, 64):
            cells[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 18 if W == 64 else 64,
                                               lr if W == 64 else 5, st)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "NOT_RESOLVED" and "мало восстановленных пар" in det, (v, det, diag)
assert diag["n_rec_pairs"] == 5
# граница: ровно 6 пар → гейт НЕ срабатывает (<6 строгое)
cells6 = {}
for di in range(5):
    for n in N_DOSE:
        rec = (di == 0) or (di == 1 and n in (2, 4))
        lr = 10 if rec else 18
        st = "recovered" if rec else "not_recovered"
        for W in (8, 16, 32, 64):
            cells6[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 18 if W == 64 else 64,
                                                lr if W == 64 else 5, st)
axA, axB = axis_A(cells6), axis_B(cells6)
v, det, diag = compute_verdict(axA, axB, cells6)
assert diag["n_rec_pairs"] == 6 and "мало восстановленных пар" not in det, (v, det, diag)

# 4b) NO_EFFECT (ревью §5): большинство клеток оси — no_effect (below_resolution)
#     при достаточном числе восстановленных пар (9 >= 6 — гейт 1 не срабатывает).
cells = {}
for di in range(6):
    for j, n in enumerate(N_DOSE):
        rec = di < 3 and n in (2, 4, 8)
        st = "recovered" if rec else "no_effect"
        for W in (8, 16, 32, 64):
            cells[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64,
                                               5 if rec else 0, st)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "NO_EFFECT" and "ниже разрешения" in det, (v, det, diag)
assert diag["ne_share_axis"] == 0.625 and diag["n_rec_pairs"] == 9

# 5) NOT_RESOLVED гейт 2 (A4): большинство not_recovered по ВОЗМУЩЁННОЙ популяции
#    (no_effect в знаменателе нет — v1/v2 читали по всем 116, промах 3 п.п.)
cells = {}
for di in range(8):
    for n in N_DOSE:
        for W in (8, 16, 32, 64):
            st = "not_recovered" if di < 5 else "recovered"
            lr = 64 if st == "not_recovered" else 5
            cells[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "NOT_RESOLVED" and "большинство not_recovered" in det, (v, det, diag)
assert diag["nr_share_perturbed"] == 0.625 and diag["notrecovered_share_full"] == 0.625
# при 50/50 — НЕ большинство (строго больше)
cells2 = {}
for di in range(8):
    for n in N_DOSE:
        for W in (8, 16, 32, 64):
            st = "not_recovered" if di < 4 else "recovered"
            lr = 64 if st == "not_recovered" else 5
            cells2[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells2), axis_B(cells2)
v, det, diag = compute_verdict(axA, axB, cells2)
assert diag["nr_share_perturbed"] == 0.5
assert "большинство not_recovered" not in det, (v, det)

# 5b) гейт 2 vs старое чтение (режим реального store, точные per-W счётчики
#     committed store): 55 not_rec + 29 recovered + 32 no_effect = 116.
#     По всем (v1/v2): 55/116 = 0.474 < 0.5 — промах 3 п.п.; по возмущённым с
#     W=8 исключён (B3): 38/58 = 0.655 > 0.5 — срабатывает.
W8 = ["not_recovered"] * 17 + ["recovered"] * 9 + ["no_effect"] * 6
W16 = ["not_recovered"] * 11 + ["recovered"] * 9 + ["no_effect"] * 12
W32 = ["not_recovered"] * 12 + ["recovered"] * 7 + ["no_effect"] * 13
W64 = ["not_recovered"] * 15 + ["recovered"] * 4 + ["no_effect"] * 1
statuses = W8 + W16 + W32 + W64
cells3 = {}
for i, st in enumerate(statuses):
    W = (8, 16, 32, 64)[i // 32]
    cells3[f"d{i % 12}|n{N_DOSE[i % 4]}|W{W}|{i}"] = dict(
        doc=f"d{i % 12}", n=N_DOSE[i % 4], W=W, D_max=64,
        l_rec=64 if st == "not_recovered" else (0 if st == "no_effect" else 5),
        rec_status=st)
axA, axB = axis_A(cells3), axis_B(cells3)
v, det, diag = compute_verdict(axA, axB, cells3)
# v3 (B3): возмущённые без W=8: (11+9+12+7+15+4) = 58, not_rec = 38 → 0.655 > 0.5
assert v == "NOT_RESOLVED" and "большинство not_recovered" in det, (v, det, diag)
assert diag["nr_share_perturbed"] == 0.6552 and diag["notrecovered_share_full"] == 0.4741, diag

# 6) реальное распределение r1/v2 (из v2-артефакта): машина v3 → NOT_RESOLVED
#    (P0-проверки ревью: #recovered(W64)=4 < 6; возмущённые без W=8: 38/58 = 0.655 > 0.5).
#    Машина v2 на тех же клетках — ELASTIC (артефакт v2, sha 215900e4): гейты
#    v1/v2 считали l_rec>0 (19 пар) и все 116 (0.474 < 0.5) — см. test_verdict_defects.
import json
claim = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "AT8", "AT8_2_RECOVERY_v2.json")))
axA = axis_A(claim["cells"])
axB = axis_B(claim["cells"])
v, det, diag = compute_verdict(axA, axB, claim["cells"])
assert v == "NOT_RESOLVED" and claim["verdict"]["value"] == "ELASTIC", (v, claim["verdict"])
assert diag["n_rec_pairs"] == 4 and diag["nr_share_perturbed"] == 0.6552, diag  # B3: без W=8: 38/58
assert axA["W64"]["ratio"] == 1.0333  # возмущённая популяция (no_effect исключены)
assert axB["frac_to_global"] == 0.3276 and axB["frac_to_global_legacy"] == 0.2328, axB
# W64-только, корректная популяция — выше порога DILUTION (ревью §2/§6.3):
w64 = [c for c in claim["cells"].values() if c["W"] == 64
       and c.get("rec_status") in ("recovered", "not_recovered")]
assert round(sum(c["to_global"] for c in w64) / len(w64), 4) == 0.6842

print("VERDICT_TEST_PASS: машина v3 (ELASTIC/DIFFUSION/DILUTION/NO_EFFECT; гейты "
      "A1/A4/A3, границы; реальный store → NOT_RESOLVED: n_rec_pairs=4, 38/58 (без "
      "W8, B3), frac 0.3276 legacy 0.2328, W64-only 0.6842)")
