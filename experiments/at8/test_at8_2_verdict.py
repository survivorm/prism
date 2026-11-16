# -*- coding: utf-8 -*-
"""Exec-тест осей/вердикта/гейтов (AGENTS п.2: designed-сценарии — exec точного фрагмента).

Синтетические клетки (без модели) покрывают: DIFFUSION, DILUTION, NOT_RESOLVED
(оба гейта: <6 пар (doc,n) с ℓ_rec>0 на W64; большинство not_recovered по полной
популяции), ELASTIC. Функции — те же, что в прогоне и v2-реанализе.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from at8_2_analysis import axis_A, axis_B, compute_verdict, N_DOSE, W_AXIS


def mk(doc, n, W, D_max, l_rec, rec_status, to_global=None):
    c = dict(doc=doc, n=n, W=W, D_max=D_max, rec_status=rec_status, l_rec=l_rec,
             W8_noisy_flag=False)
    if to_global is not None:
        c["to_global"] = to_global
    return c


def cells8(nmap_w64, to_global_all):
    """8 доков × 4 дозы; w64: {n: l_rec} (recovered, D_max=64); остальные W — l_rec=5."""
    out = {}
    for di in range(8):
        for n in N_DOSE:
            for W in (8, 16, 32, 64):
                if W == 64:
                    lr = nmap_w64[n]
                else:
                    lr = 5
                out[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr,
                                                  "recovered", to_global_all)
    return out


# 1) ELASTIC: lrec W64 ≈ константа, to_global мало
cells = cells8({2: 10, 4: 12, 8: 11, 16: 13}, False)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "ELASTIC", (v, det)
assert diag["n_pos"] == 4 and diag["n_meas_doc_n"] == 32  # 8 доков × 4 дозы
assert diag["notrecovered_share_full"] == 0.0

# 2) DIFFUSION: lrec растёт с дозой, ratio≥4, r2≥0.6
cells = cells8({2: 2, 4: 4, 8: 8, 16: 16}, False)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "DIFFUSION", (v, det, axA["W64"])

# 3) DILUTION: to_global > 0.6 (ось A — умеренная, не DIFFUSION)
cells = cells8({2: 10, 4: 12, 8: 11, 16: 13}, True)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "DILUTION", (v, det)

# 4) NOT_RESOLVED гейт 1: <6 пар (doc,n) с ℓ_rec>0 на W=64
#    1 док × 4 дозы = 4 пары < 6; остальные 7 доков — l_rec=0 (no_effect)
cells = {}
for di in range(8):
    for n in N_DOSE:
        lr = 10 if di == 0 else 0
        st = "recovered" if lr > 0 else "no_effect"
        for W in (8, 16, 32, 64):
            cells[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells), axis_B(cells)
v, det, diag = compute_verdict(axA, axB, cells)
assert v == "NOT_RESOLVED" and "мало пар" in det, (v, det, diag)
assert diag["n_meas_doc_n"] == 4  # 1 док × 4 дозы на W64
# граница: ровно 6 пар → гейт НЕ срабатывает (<6 строгое)
cells6 = {}
for di in range(8):
    for n in N_DOSE:
        lr = 10 if di < 6 else 0
        st = "recovered" if lr > 0 else "no_effect"
        for W in (8, 16, 32, 64):
            cells6[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells6), axis_B(cells6)
v, det, diag = compute_verdict(axA, axB, cells6)
assert diag["n_meas_doc_n"] == 24 and "мало пар" not in det, (v, det, diag)
# точная граница 5/6 (r2 N-2): ровно 5 пар → срабатывает, ровно 6 пар → не срабатывает
# 5 пар на W64: 1 док × 4 дозы + 1 доза у второго дока
cells5 = {}
for di in range(8):
    for n in N_DOSE:
        lr = 10 if (di == 0) or (di == 1 and n == 2) else 0
        st = "recovered" if lr > 0 else "no_effect"
        for W in (8, 16, 32, 64):
            cells5[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells5), axis_B(cells5)
v, det, diag = compute_verdict(axA, axB, cells5)
assert diag["n_meas_doc_n"] == 5 and "мало пар" in det, (v, det, diag)
# ровно 6 пар: 1 док × 4 дозы + 2 дозы у второго дока
cells6b = {}
for di in range(8):
    for n in N_DOSE:
        lr = 10 if (di == 0) or (di == 1 and n in (2, 4)) else 0
        st = "recovered" if lr > 0 else "no_effect"
        for W in (8, 16, 32, 64):
            cells6b[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells6b), axis_B(cells6b)
v, det, diag = compute_verdict(axA, axB, cells6b)
assert diag["n_meas_doc_n"] == 6 and "мало пар" not in det, (v, det, diag)

# 5) NOT_RESOLVED гейт 2: большинство not_recovered по полной популяции
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
assert diag["notrecovered_share_full"] == 0.625
# при 50/50 — НЕ большинство (строго больше): 4 дока not_rec, 4 дока rec → 0.5
cells2 = {}
for di in range(8):
    for n in N_DOSE:
        for W in (8, 16, 32, 64):
            st = "not_recovered" if di < 4 else "recovered"
            lr = 64 if st == "not_recovered" else 5
            cells2[f"wiki_{di}|n{n}|W{W}"] = mk(f"wiki_{di}", n, W, 64, lr, st, False)
axA, axB = axis_A(cells2), axis_B(cells2)
v, det, diag = compute_verdict(axA, axB, cells2)
assert diag["notrecovered_share_full"] == 0.5
assert "большинство not_recovered" not in det, (v, det)

# 6) реальное распределение r1 (из v2-реанализа): ELASTIC, n_pos=4, 55/116
import json
claim = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "AT8", "AT8_2_RECOVERY_v2.json")))
axA = axis_A(claim["cells"])
axB = axis_B(claim["cells"])
v, det, diag = compute_verdict(axA, axB, claim["cells"])
assert v == "ELASTIC" and claim["verdict"]["value"] == "ELASTIC", (v, claim["verdict"])
assert diag == {k: claim["diagnostics"][k] for k in diag}, (diag, claim["diagnostics"])
assert axA["W64"]["ratio"] == claim["axis_A"]["W64"]["ratio"] == 1.0333
assert axB["frac_to_global"] == claim["axis_B"]["frac_to_global"] == 0.2328

print("VERDICT_TEST_PASS: DIFFUSION/ELASTIC/DILUTION/NOT_RESOLVED (оба гейта, граница 0.5) "
      "+ воспроизведение v2-реанализа из артефакта (ratio 1.0333, frac 0.2328)")
