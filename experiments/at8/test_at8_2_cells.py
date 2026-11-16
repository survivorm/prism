# -*- coding: utf-8 -*-
"""Exec-тест classify_cell (AGENTS п.2: designed-сценарии веток — exec точного фрагмента).

Ветка not_finite в r1-прогоне не срабатывала (finite-модель → finite dip) —
протестирована здесь на синтетических NaN/None-кривых (тот же фрагмент,
который использует build_cells_from_store).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from at8_2_analysis import classify_cell

# 1) not_finite: NaN в dip-кривой → status=not_finite, нет l_rec/pert0
c = classify_cell("wiki_0", 2, 64, 256, -0.2, -0.2,
                  ck=[-0.2] * 65, ik=[-0.2, float("nan"), -0.2] + [-0.2] * 62)
assert c["status"] == "not_finite", c
assert "l_rec" not in c and "pert0" not in c and "to_global" not in c

# 2) not_finite: None в кривой (окно не влезло по d, но D_max>=0)
c2 = classify_cell("wiki_0", 2, 64, 256, -0.2, -0.2,
                   ck=[-0.2] * 65, ik=[-0.2, None] + [-0.2] * 63)
assert c2["status"] == "not_finite", c2

# 3) not_measurable: ck/ik = None (D_max < 0)
c3 = classify_cell("wiki_5", 2, 64, 125, -0.2, -0.2, ck=None, ik=None)
assert c3["status"] == "not_measurable", c3

# 4) no_effect: pert0 < eps_min
c4 = classify_cell("wiki_1", 4, 64, 256, -0.2, -0.2,
                   ck=[-0.2] * 65, ik=[-0.205] + [-0.2] * 64)
assert c4["rec_status"] == "no_effect" and c4["l_rec"] == 0, c4
assert abs(c4["pert0"] - 0.005) < 1e-9

# 5) recovered: pert0 >= eps_min, спад ≤ 25% к d=3 (pert[2]=0.011 > 0.01=25%·pert0)
c5 = classify_cell("wiki_1", 8, 64, 256, -0.21, -0.2,
                   ck=[-0.2] * 65, ik=[-0.24, -0.22, -0.211, -0.2005] + [-0.2] * 61)
assert c5["rec_status"] == "recovered" and c5["l_rec"] == 3, c5

# 6) not_recovered: pert > 25%·pert0 до D_max
c6 = classify_cell("wiki_0", 8, 64, 256, -0.21, -0.2,
                   ck=[-0.2] * 65, ik=[-0.24] + [-0.239] * 64)
assert c6["rec_status"] == "not_recovered" and c6["l_rec"] == 64, c6

# 7) D_max < 8: l_rec/rec_status не отчитаны, rec_measurable=False
c7 = classify_cell("wiki_4", 2, 64, 132, -0.21, -0.2,
                   ck=[-0.2] * 3, ik=[-0.24, -0.239, -0.239])
assert c7["D_max"] == 2 and c7["rec_measurable"] is False, c7
assert "l_rec" not in c7 and "rec_status" not in c7 and "to_global" not in c7

# 8) to_global: residual_global < residual_ctrl И |dip_global - dip_control_whole| > 25%·pert0
c8 = classify_cell("wiki_1", 8, 64, 256, -0.25, -0.2,
                   ck=[-0.2] * 65, ik=[-0.24] + [-0.245] * 64)
# pert0=0.04; D_max=64: ik[64]=-0.245, ck[64]=-0.2 → res_ctrl=0.045;
# res_glob=|-0.245-(-0.25)|=0.005 < 0.045; |-0.25-(-0.2)|=0.05 > 0.01 → to_global=True
assert c8["to_global"] is True, c8
# и наоборот: окно ушло к контролю → to_global=False
c9 = classify_cell("wiki_1", 8, 64, 256, -0.25, -0.2,
                   ck=[-0.2] * 65, ik=[-0.24] + [-0.2001] * 64)
assert c9["to_global"] is False, c9

print("CLASSIFY_TEST_PASS: ветки not_finite/not_measurable/no_effect/recovered/"
      "not_recovered/rec_measurable/to_global (exec точного фрагмента)")
