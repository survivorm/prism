# -*- coding: utf-8 -*-
"""AT8_2 (T8.2) «Оконное восстановление + доза» — ОДИН скрипт (user 17.11: «3 скрипта
для одного эксперимента — многовато, сведи в 1»). Слияние: AT8_2_recovery.py +
AT8_2_v2_reanalysis.py + at8_2_analysis.py (все три удалены; логика — вербати +
машина v3).

Режимы:
  python AT8_2.py               — прогон (40 форвардов + 2 S8-check, CPU; ./run_one.sh)
  python AT8_2.py --reanalysis  — disk-only реанализ v3 из committed store (без модели)
  python AT8_2.py --calibrate   — калибровка машины на синтетических store (P0.5 ревью)

Машина v3 (внешнее ревью docs/AT8_2_review.md, 17.11; P0-проверки подтвердили дефекты
на committed store — см. docs/AT8_2_review_r1_answers.md):
- A1: гейт «мало пар» считает (doc,n) с rec_status=='recovered' на W=w_axis —
  цензурированные (ℓ_rec=D_max — провал восстановления) не засчитываются
  восстановлением (r1/v2: 19 пар с l_rec>0 проходили гейт при 4 восстановленных).
- A2: no_effect-нули (pert0<eps — «возмущения не видно») исключены из основной
  медианы и из ratio; отдельный исход NO_EFFECT (большинство no_effect на оси —
  below_resolution, paper §6 п.25: not_testable, не refuted).
- A3: отсутствующая доза (l2/l16 None) → ratio=None → NOT_RESOLVED, не 0.0
  (отсутствие измерения — не положительное свидетельство упругости).
- A4: majority-гейт — по возмущённой популяции {pert0>=eps_min, D_max>=8}
  (no_effect не могут быть not_recovered по построению — они топили знаменатель:
  55/116=0.474 промах мимо 0.5; 38/58=0.655 — срабатывает, без W=8 (B3)). Старое чтение —
  diagnostic notrecovered_share_full.
- B3: W8_noisy_flag — рабочая (W=8 исключена из оси B и гейтов, как из оси A —
  MINOR-4): самые шумные клетки (медиана ~6 косинус-пар) не в оси, несущей вердикт.
- B6: no_effect-клетки исключены из оси B (нет ограничения на pert[D_max] —
  голосуют «dilution» на дальнем отклонении, превышающем начальное).
- C1: статус решается на той же точности, что публикуется (round(pert0,6) ДО
  сравнения с eps_min). C2: строгие сравнения остатков — с tol 1e-9
  (асимметрия округления dip_global/dip_control_whole). C6: ratio=inf → None +
  флаг ratio_inf (JSON-строка "inf" не парсится обратно). C9: SS_tot=0 → r2=None.
  C10: assert длин dip-кривых (длинная кривая не обрезается молча). C11: G14-координаты
  (object/stat/layer) в каждой клетке. C8: sys.modules['fla']=None (False даёт
  «успешный» импорт) + attn_impl в setup. C4: store несёт отпечаток конфигурации
  (__config__); resume чужого/старого store — STOP.
- S8 (C3/§6.1 ревью): остаётся in-process регресс-тестом индексации (ТЗ-объявлено);
  честный кросс-чек E-71B-конвенции = AST-сплайс из experiments/e71_80/exp71_soliton.py
  + его sha — P1-перепрогон (тут недоступен disk-only).

Механика dip — пин e64_gpu.py (гейт S7). Запуск прогона: ./run_one.sh at8_2 ...
(OMP=6, nice-10; TRACKIO_DIR экспортирует run_one.sh — AGENTS п.6).
trackio: run at8_2_recovery_r3 (AGENTS: обязателен для CPU-прогонов).
"""
import ast
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/at8/ → repo

# =====================================================================
# ЧАСТЬ 1 — МАШИНА (анализный модуль, disk-safe, только numpy)
# =====================================================================

N_DOSE = [2, 4, 8, 16]
W_SET = [8, 16, 32, 64]
W_AXIS = 64
W_NOISY = 8           # MINOR-4/B3: самое шумное окно (медиана ~6 косинус-пар)
EPS_MIN = 0.01
THR = 0.25
D_MIN_MEAS = 8
TOL_RES = 1e-9        # C2: tol строгих сравнений остатков (асимметрия округления)

# C11 (G14): координаты объекта в каждой клетке — константы прогона
OBJECT_META = dict(object="h_raw", stat="median_of_cos", layer=23,
                   frame="L23 full-attention, dip = median lag-1 cos, e64_gpu pin")


def classify_cell(nm, n, W, T, dip_global, dip_control_whole, ck, ik,
                  eps_min=EPS_MIN, thr=THR, d_min_meas=D_MIN_MEAS,
                  object_meta=OBJECT_META):
    """Одна ячейка (doc,n,W).

    v3 (ревью AT8_2_review.md): C10 — assert длин кривых; C1 — статус по rounded
    pert0 (решение и публикация на одной точности); C2 — сравнение остатков с tol;
    C11 — G14-координаты; B3 — W8_noisy_flag (исключение W=8 из оси B — в axis_B).
    """
    D_max = T - T // 2 - W
    cell = dict(doc=nm, n=n, W=W, D_max=D_max,
                dip_global=round(dip_global, 6),
                dip_control_whole=round(dip_control_whole, 6),
                W8_noisy_flag=(W == W_NOISY))
    if ck is not None and ik is not None:
        assert len(ik) == D_max + 1 == len(ck), \
            (f"C10: длины кривых {nm} n={n} W={W}: len(ik)={len(ik)}, "
             f"len(ck)={len(ck)}, D_max+1={D_max + 1}")
    if ck is None or ik is None:
        cell.update(status="not_measurable")
        return cell
    if any(v is None for v in ik + ck) or \
       any(not np.isfinite(v) for v in ik + ck):
        cell.update(status="not_finite")
        return cell
    cell.update(dip_inj_d0=round(ik[0], 6), dip_ctrl_d0=round(ck[0], 6))
    pert = [abs(ik[d] - ck[d]) for d in range(D_max + 1)]
    pert0 = round(pert[0], 6)          # C1: статус решается на публикуемой точности
    cell["pert0"] = pert0
    if pert0 < eps_min:
        cell.update(l_rec=0, rec_status="no_effect")
    else:
        found = None
        for d in range(D_max + 1):
            if max(pert[d:]) <= thr * pert0:
                found = d
                break
        if found is None:
            cell.update(l_rec=D_max, rec_status="not_recovered")
        else:
            cell.update(l_rec=found, rec_status="recovered")
    if D_max < d_min_meas:
        cell["rec_measurable"] = False
        cell.pop("l_rec", None)
        cell.pop("rec_status", None)
    # dilution (только при D_max ≥ d_min_meas), TZ §4: одна документная шкала
    # (B2-ограничение в силе: допуск thr*pert0 — оконный масштаб; правка — P1 с
    # per-position рядами и δ_min; здесь — tol C2 + эксклюзии B3/B6 в axis_B)
    if D_max >= d_min_meas and "rec_status" in cell:
        res_ctrl = abs(ik[D_max] - ck[D_max])
        res_glob = abs(ik[D_max] - dip_global)
        cell["residual_ctrl"] = round(res_ctrl, 6)
        cell["residual_global"] = round(res_glob, 6)
        cell["to_global"] = bool(res_glob + TOL_RES < res_ctrl and
                                 abs(dip_global - dip_control_whole) > thr * pert0)
    cell.update(object_meta)
    return cell


def build_cells_from_store(res, names, n_dose=N_DOSE, w_set=W_SET,
                           eps_min=EPS_MIN, thr=THR, d_min_meas=D_MIN_MEAS):
    """Ячейки из store (двухпроходные dip-кривые) — порядок как в r1-прогоне."""
    cells = {}
    for nm in names:
        ctrl = res[f"{nm}__ctrl"]
        T = ctrl["T"]
        dip_control_whole = ctrl["dip_control_whole"]
        for n in n_dose:
            inj = res[f"{nm}__n{n}"]
            dip_global = inj["dip_global"]
            for W in w_set:
                ck = ctrl["dip_ctrl"][str(W)]
                ik = inj["dip_inj"][str(W)]
                cell = classify_cell(nm, n, W, T, dip_global, dip_control_whole, ck, ik,
                                     eps_min=eps_min, thr=thr, d_min_meas=d_min_meas)
                cells[f"{nm}|n{n}|W{W}"] = cell
    return cells


def _perturbed(status):
    """v3 (A2/A4): возмущённые клетки — pert0 >= eps_min (recovered + not_recovered).
    no_effect (ℓ_rec=0 — «возмущения не видно») исключены из основной медианы и
    ratio: лемма ревью — recovered с l_rec=0 невозможен (ветка входит при
    pert0>=eps>0, и pert0 <= 0.25*pert0 ложно), значит нулевые ℓ_rec — это всегда
    no_effect, и нулевая медиана — не «мгновенное восстановление»."""
    return status in ("recovered", "not_recovered")


def lrec_medians(cells, n, w, censored):
    """Медиана ℓ_rec по докам для дозы n при W=w.

    censored=False (ОСНОВНАЯ, TZ §5 + ревью A2): ВОЗМУЩЁННЫЕ клетки (recovered +
    not_recovered; not_recovered, ℓ_rec=D_max — цензурированное значение, входит в
    медиану; no_effect исключены — A2). censored=True (вторичный, справочно):
    только recovered/no_effect — чтение r1 («censored excluded»).
    """
    vals = [c["l_rec"] for c in cells.values()
            if c["n"] == n and c["W"] == w and "l_rec" in c
            and (_perturbed(c.get("rec_status")) if not censored
                 else c.get("rec_status") in ("recovered", "no_effect"))]
    return round(float(np.median(vals)), 3) if vals else None


def _ratio_lr(lr):
    """l16/l2. v3 (A3): отсутствующая доза (None) → None («недостаточно доз»),
    НЕ 0.0 — отсутствие измерения не является свидетельством упругости."""
    l2, l16 = lr.get(2), lr.get(16)
    if l2 is None or l16 is None:
        return None
    if l2 == 0 and l16 > 0:
        return float("inf")
    if l2 == 0 and l16 == 0:
        return 0.0
    return round(l16 / l2, 4)


def _fit(lr, n_dose):
    ns_arr = np.array(n_dose, dtype=float)
    lr_arr = np.array([lr[n] if lr[n] is not None else np.nan for n in n_dose], dtype=float)
    mask = ~np.isnan(lr_arr)
    if mask.sum() < 3:
        return None, None, None
    b, a = np.polyfit(ns_arr[mask], lr_arr[mask], 1)
    pred = a + b * ns_arr[mask]
    ss_res = float(((lr_arr[mask] - pred) ** 2).sum())
    ss_tot = float(((lr_arr[mask] - lr_arr[mask].mean()) ** 2).sum())
    # C9: SS_tot=0 — r2 не определён (None), 0.0 читалось бы как «фита нет»
    r2 = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None
    return round(float(b), 4), round(float(a), 4), r2


def axis_A(cells, n_dose=N_DOSE, w_set=W_SET, w_axis=W_AXIS):
    """Ось A — ℓ_rec(n) по медиане доков. Основная популяция — возмущённые
    (TZ §5 + v3 A2); censored_excluded — вторичный (справочно, явно помечен).
    Конвенция «медиана доков» (TZ v1.2): сначала редукция к уровню дока (для W=64 —
    одно значение на док; для combined — медиана по W внутри дока), потом медиана по докам."""
    def both(w):
        full = {n: lrec_medians(cells, n, w, censored=False) for n in n_dose}
        cens = {n: lrec_medians(cells, n, w, censored=True) for n in n_dose}
        b, a, r2 = _fit(full, n_dose)
        ratio = _ratio_lr(full)
        return dict(lrec=full, lrec_censored_excluded=cens,
                    ratio=ratio if ratio != float("inf") else None,
                    ratio_inf=(ratio == float("inf")),   # C6: inf не сериализуется
                    b=b, a=a, r2=r2)
    w64 = both(w_axis)
    full = {n: lrec_medians_allw(cells, n, censored=False) for n in n_dose}
    cens = {n: lrec_medians_allw(cells, n, censored=True) for n in n_dose}
    b, a, r2 = _fit(full, n_dose)
    ratio = _ratio_lr(full)
    combined = dict(lrec=full, lrec_censored_excluded=cens,
                    ratio=ratio if ratio != float("inf") else None,
                    ratio_inf=(ratio == float("inf")),
                    b=b, a=a, r2=r2,
                    note="combined (справочно): медиана по W внутри дока, потом по докам (TZ v1.2)")
    return dict(W_axis=w_axis,
                W64=dict(ratio=w64["ratio"], ratio_inf=w64["ratio_inf"], b=w64["b"],
                         a=w64["a"], r2=w64["r2"], lrec=w64["lrec"],
                         lrec_censored_excluded=w64["lrec_censored_excluded"]),
                combined_secondary=combined)


def lrec_medians_allw(cells, n, censored):
    """Медиана ℓ_rec по докам (объединённо по W) — справочно (TZ §5).
    Популяция — как в lrec_medians (v3: основную составляют возмущённые)."""
    by_doc = {}
    for c in cells.values():
        if c["n"] == n and "l_rec" in c and \
           (_perturbed(c.get("rec_status")) if not censored
            else c.get("rec_status") in ("recovered", "no_effect")):
            by_doc.setdefault(c["doc"], []).append(c["l_rec"])
    if not by_doc:
        return None
    per_doc = [float(np.median(v)) for v in by_doc.values()]
    return round(float(np.median(per_doc)), 3)


def axis_B(cells, w_noisy=W_NOISY):
    """Ось B — доля to_global (TZ §5).

    v3 (ревью B3/B6): популяция — ВОЗМУЩЁННЫЕ клетки (no_effect исключены: нет
    ограничения на pert[D_max], такие клетки голосуют «dilution» на дальнем
    отклонении, B6) и W != w_noisy (W8_noisy_flag рабочая: самое шумное окно —
    6-парная медиана против ~400-парной документной — не в оси, несущей вердикт,
    B3). Старое чтение (все измеримые, все W) — diagnostic frac_to_global_legacy.
    """
    legacy = [c["to_global"] for c in cells.values() if "to_global" in c]
    cor = [c["to_global"] for c in cells.values() if "to_global" in c
           and c.get("rec_status") in ("recovered", "not_recovered")
           and c["W"] != w_noisy]
    return dict(frac_to_global=round(float(np.mean(cor)), 4) if cor else None,
                n_to_global_pairs=len(cor),
                frac_to_global_legacy=round(float(np.mean(legacy)), 4) if legacy else None,
                n_legacy_pairs=len(legacy))


def compute_verdict(axis_A, axis_B, cells, n_dose=N_DOSE, w_axis=W_AXIS):
    """Вердикт (машина v3, ревью AT8_2_review.md).

    Приоритет: NO_EFFECT (below_resolution) > гейты NOT_RESOLVED > DIFFUSION >
    ELASTIC > DILUTION > NOT_RESOLVED (fallback).

    v3-гейты (явные популяции):
    - NO_EFFECT: большинство клеток W=w_axis — no_effect (возмущение ниже
      разрешения оценщика — paper §6 п.25: not_testable, не refuted);
    - «мало восстановлений»: (doc,n) с rec_status=='recovered' на W=w_axis < 6
      (A1: цензурированный провал — не восстановление; r1/v2 читали l_rec>0 —
      19 пар проходили гейт при 4 восстановленных);
    - «большинство not_recovered»: по ВОЗМУЩЁННОЙ популяции {pert0>=eps_min,
      D_max>=8} (A4: no_effect топили знаменатель — 55/116=0.474 промах,
      38/58=0.655 срабатывает, без W=8 (B3); старое чтение — diagnostic notrecovered_share_full;
    - ratio=None (A3: отсутствующая доза) → NOT_RESOLVED.
    """
    lrec_w64 = axis_A["W64"]["lrec"]
    ratio = axis_A["W64"]["ratio"]
    ratio_inf = axis_A["W64"].get("ratio_inf", False)
    b = axis_A["W64"]["b"]
    r2 = axis_A["W64"]["r2"]
    f_tg = axis_B["frac_to_global"]

    rec_pairs = {(c["doc"], c["n"]) for c in cells.values()
                 if c["W"] == w_axis and c.get("rec_status") == "recovered"}
    n_rec_pairs = len(rec_pairs)
    n_pos = len({n for _, n in rec_pairs})          # A1: дозы с восстановившимися клетками
    # B3 (v3): W_NOISY исключён и из majority-гейта (самые шумные клетки — 6-парная
    # медиана — не в популяции, несущей вердикт)
    perturbed = [c for c in cells.values()
                 if _perturbed(c.get("rec_status")) and c["W"] != W_NOISY]
    n_notrec = sum(1 for c in perturbed if c["rec_status"] == "not_recovered")
    n_perturbed = len(perturbed)
    with_status = [c for c in cells.values() if "rec_status" in c]
    n_notrec_full = sum(1 for c in with_status if c["rec_status"] == "not_recovered")
    n_noef = sum(1 for c in cells.values() if c.get("rec_status") == "no_effect")
    axis_cells = [c for c in with_status if c["W"] == w_axis]
    ne_share_axis = (sum(1 for c in axis_cells if c["rec_status"] == "no_effect")
                     / len(axis_cells)) if axis_cells else None
    n_w64_status = [c for c in with_status if c["W"] == w_axis]
    w64_notrec_share = (round(sum(1 for c in n_w64_status
                                  if c["rec_status"] == "not_recovered")
                              / len(n_w64_status), 4) if n_w64_status else None)

    diag = dict(n_rec_pairs=n_rec_pairs, n_pos=n_pos,
                n_notrec_perturbed=n_notrec, n_perturbed=n_perturbed,
                nr_share_perturbed=round(n_notrec / n_perturbed, 4) if n_perturbed else None,
                n_no_effect=n_noef,
                ne_share_axis=round(ne_share_axis, 4) if ne_share_axis is not None else None,
                notrecovered_share_full=round(n_notrec_full / len(with_status), 4)
                if with_status else None,
                axis_W64_notrecovered_share=w64_notrec_share)

    # NO_EFFECT (ревью §5): возмущение ниже разрешения оценщика
    if ne_share_axis is not None and ne_share_axis > 0.5:
        return ("NO_EFFECT",
                f"большинство клеток W={w_axis} — no_effect ({ne_share_axis:.2f} > 0.5): "
                f"возмущение ниже разрешения оценщика — not_testable, не refuted "
                f"(paper §6 п.25)", diag)

    r = ratio if not ratio_inf else None
    cond_diff = (r is not None and r >= 4) or ratio_inf
    verdict, vdetail = "NOT_RESOLVED", ""
    if cond_diff and b is not None and b > 0 and r2 is not None and r2 >= 0.6:
        verdict = "DIFFUSION"
        vdetail = f"ℓ_rec растёт с дозой (ratio={'inf' if ratio_inf else ratio}, b={b}, r2={r2}, W={w_axis})"
    elif (r is not None and r <= 2) and f_tg is not None and f_tg <= 0.4 and n_pos >= 2:
        verdict = "ELASTIC"
        vdetail = (f"ℓ_rec слабо зависит от n (ratio={ratio}, W={w_axis}, "
                   f"возмущённая популяция), окно к контролю (frac_to_global={f_tg})")
    elif f_tg is not None and f_tg > 0.6:
        verdict = "DILUTION"
        vdetail = f"окно ≈ глобальное разбавление (frac_to_global={f_tg})"
    else:
        vdetail = (f"ratio={ratio} b={b} r2={r2} frac_to_global={f_tg} "
                   f"n_rec_pairs={n_rec_pairs} n_pos={n_pos} "
                   f"nr_pert={n_notrec}/{n_perturbed}")
    # гейты NOT_RESOLVED (v3: явные популяции, ревью A1/A3/A4)
    if ratio is None and not ratio_inf:
        verdict = "NOT_RESOLVED"
        vdetail += " (недостаточно доз: ratio=None — отсутствие измерения не " \
                   "свидетельство, A3)"
    if n_rec_pairs < 6:
        verdict = "NOT_RESOLVED"
        vdetail += (f" (мало восстановленных пар (doc,n) на W={w_axis}: "
                    f"{n_rec_pairs} < 6; провал (цензурированный) не засчитывается "
                    f"восстановлением — A1)")
    if n_perturbed and n_notrec > n_perturbed / 2:
        verdict = "NOT_RESOLVED"
        vdetail += (f" (большинство not_recovered по возмущённой популяции: "
                    f"{n_notrec}/{n_perturbed} — A4)")
    return verdict, vdetail, diag


def resume_plan(names, n_dose, res):
    """Решение «что пересчитывать» при resume по ключам (TZ §6 S2) — чистая функция.

    res — существующий store ({} при fresh run). Возвращает:
    - ctrl_resume: {nm: bool} — контроль в store (пересчёт не нужен);
    - dose_resume: {(nm, n): bool} — доза в store;
    - n_resume: число переиспользуемых записей (для лога).
    Считаем только КЛЮЧИ: целостность содержимого store при resume — отпечаток
    конфигурации store["__config__"] (C4, v3) + sha-верификация manifest'ом.
    """
    ctrl_resume = {nm: f"{nm}__ctrl" in res for nm in names}
    dose_resume = {(nm, n): f"{nm}__n{n}" in res
                   for nm in names for n in n_dose}
    n_resume = sum(ctrl_resume.values()) + sum(dose_resume.values())
    return dict(ctrl_resume=ctrl_resume, dose_resume=dose_resume, n_resume=n_resume)


def cells_equal(c1, c2, allow_extra=frozenset()):
    """Равенство ячеек (r1-claim vs пересчёт): все ОБЩИЕ ключи; ключи из
    allow_extra (v3: G14-provenance object/stat/layer/frame, нет в r1-claim) —
    не расхождение."""
    keys = (set(c1) | set(c2)) - set(allow_extra)
    for k in sorted(keys):
        if k not in c1 or k not in c2:
            return False, f"key {k}: в r1={k in c1}, в v2={k in c2}"
        if c1[k] != c2[k]:
            return False, f"key {k}: {c1[k]} != {c2[k]}"
    return True, ""


def store_config_fp(model, layer, phrase_ids, n_dose, w_set, w_axis,
                    eps_min, thr, d_min_meas, e64_sha, pool_fp, revision):
    """C4 (v3): отпечаток конфигурации прогона — store несёт его, resume чужого
    store (другая фраза/модель/слой/пороги) — STOP."""
    return dict(model=model, layer=layer, phrase_ids=phrase_ids, n_dose=n_dose,
                W_set=w_set, W_axis=w_axis, eps_min=eps_min, thr=thr,
                d_min_meas=d_min_meas, e64_gpu_sha=e64_sha, pool_fp=pool_fp,
                revision=revision)


# =====================================================================
# ЧАСТЬ 2 — ПРОГОН (40 форвардов + 2 S8-check; CPU; verbatim из
# AT8_2_recovery.py + C4/C8 + S8-честность + REVISION r3)
# =====================================================================

MODEL = "/data/Qwen/Qwen3.5-0.8B"
LAYER = 23
PHRASE = " The class will be held at the park."          # 9 токенов (E-71B)
PHRASE_IDS = [561, 523, 668, 381, 5467, 506, 279, 5926, 13]   # TZ §3 (add_special_tokens=False)
POOL_PIN = "data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"
POOL_FP = "144c722df3046f2d"
E64_PATH = "experiments/kaggle_e64gpu/e64_gpu.py"
E64_SHA = "8c517e18f801fa877dfde9265ed5ce18df1e7d3af705194fc01ff6b150cf7633"

OUT_DIR = os.path.join(_ROOT, "results", "AT8")
CLAIM = os.path.join(OUT_DIR, "AT8_2_RECOVERY.json")
STORE = os.path.join(OUT_DIR, "AT8_2_RECOVERY_store.json")
MANIFEST = os.path.join(OUT_DIR, "AT8_2_RECOVERY_manifest.json")
RUNLOG = os.path.join(_ROOT, "logs", "AT8", "AT8_2_run.log")
REVISION = "r3"


def logline(s):
    print(s, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- dip-механика из e64_gpu.py (гейт S7) ----
def load_dip_machinery():
    src = open(os.path.join(_ROOT, E64_PATH)).read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    assert sha == E64_SHA, f"S7 fail: e64_gpu.py sha {sha[:16]} != {E64_SHA[:16]}"
    tree = ast.parse(src)
    fns = {n.name: ast.get_source_segment(src, n)
           for n in tree.body if isinstance(n, ast.FunctionDef)}
    ns = {"np": np}
    for w in ["vpath_units", "dcurve", "curves_series"]:
        exec(fns[w], ns)
    return ns["curves_series"], sha


# ---- документы из пина (plain-путь load_calibration_data, как E-71B) ----
def load_docs():
    meta = json.load(open(os.path.join(_ROOT, POOL_PIN)))
    assert meta["fp"] == POOL_FP, f"pool_fp mismatch: {meta['fp']} != {POOL_FP}"
    import torch
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL)
    docs, names = [], []
    for i, t in enumerate(meta["texts"]):
        tok = tk(t, max_length=256, truncation=True, return_tensors="pt")
        if tok["input_ids"].shape[1] >= 64:
            docs.append(tok["input_ids"][0].tolist()[:256])
            names.append(f"wiki_{i}")
        if len(docs) >= 8:
            break
    return docs, names, tk


def docs_sha256(docs, names):
    h = hashlib.sha256()
    for nm, d in zip(names, docs):
        h.update(nm.encode())
        h.update(np.array(d, dtype=np.int64).tobytes())
    return h.hexdigest()


# ---- forward: h23 (layer 23) + nll ----
def measure(m, ids, torch):
    caps = {}
    hook = m.model.layers[LAYER].register_forward_hook(
        lambda mod, i, o, l=LAYER: caps.__setitem__(l, o.detach()))
    t = torch.tensor([ids])
    try:
        with torch.no_grad():
            o = m(input_ids=t, use_cache=False)
    finally:
        hook.remove()   # NIT-3 (r2): снятие в try/finally — при краше forward хук не утекает
    nll = float(torch.nn.functional.cross_entropy(o.logits[0, :-1].float(), t[0, 1:]))
    h23 = caps[LAYER][0, :len(ids)].to(torch.float32).numpy()
    return nll, h23


def win_dips(curves_series, h23, start, W):
    """dip окна h23[start:start+W] (None, если не влезает)."""
    if start + W > len(h23):
        return None
    return float(curves_series(h23[start:start + W])["dip"])


# ---- trackio (AGENTS: обязателен для CPU-прогонов) ----
def init_trackio(t0, script_path):
    # AGENTS п.6: TRACKIO_DIR задаёт run_one.sh — скрипт его не переопределяет.
    # Fail-loud (r2 NIT-5): прямой запуск без run_one.sh — не писать live-точки в
    # дефолтный HF-cache (сценарий AT2/AT3) — стоп с явным сообщением.
    if not os.environ.get("TRACKIO_DIR"):
        raise SystemExit(
            "trackio: TRACKIO_DIR не задан — запускать через ./run_one.sh "
            "(он экспортирует TRACKIO_DIR=$PRISM_DIR/.trackio, AGENTS п.6); "
            "прямой запуск остановлен (иначе live-точки ушли бы в HF-cache)")
    try:
        import trackio
    except ImportError as e:
        raise SystemExit(f"trackio: модуль не установлен — прогон остановлен "
                         f"(AGENTS: trackio обязателен для CPU-прогонов): {e}")
    tr_cfg = {
        "started_at": datetime.now().isoformat(),
        "revision": REVISION,
        "script_sha": sha256_file(os.path.abspath(script_path)),
        "device": "CPU", "model": MODEL, "layer": LAYER,
        "n_dose": N_DOSE, "W_set": W_SET, "W_axis": W_AXIS,
        "use_cache": False, "log": os.path.abspath(RUNLOG),
        "seed": "n/a (deterministic forward, no sampling)",   # AGENTS п.5 (r2 MINOR-2)
        "e64_gpu_sha": E64_SHA[:16], "pool_fp": POOL_FP,
    }
    tr = trackio.init(project="prism", name=f"at8_2_recovery_{REVISION}", group="AT8", config=tr_cfg)
    tr.log({"started": 1.0}, step=0)
    logline(f"trackio: run at8_2_recovery_{REVISION} init ok")
    done = [False]
    import atexit
    def _on_exit():
        if not done[0]:
            try:
                tr.log({"finished_at": datetime.now().isoformat(), "status": "crash"},
                       step=int(time.time() - t0))
                tr.finish()
            except Exception:
                pass
    atexit.register(_on_exit)
    return tr, done


def tstage(tr, t0, name):
    """AGENTS п.4: крупные этапы — точка {"stage": ..., "stage_s": wall-t}."""
    tr.log({"stage": name, "stage_s": round(time.time() - t0, 1)})
    logline(f"stage: {name} ({time.time() - t0:.1f}s)")


def run_main():
    """Прогон (40 форвардов + 2 S8-check) — verbatim из AT8_2_recovery.py +
    C4 (store config fp + resume-проверка), C8 (fla=None + attn_impl)."""
    import torch
    # torch-fallback (без fla/triton) — референсный путь серии (AT8_1). v3 C8:
    # None — корректная заглушка (import даёт ImportError); False давал «успешный»
    # импорт с возвратом False (детектор transformers мог пройти мимо).
    sys.modules["fla"] = None
    sys.modules["fla.ops"] = None
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    t0 = time.time()
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    # run-log (трекаемый артефакт, TZ §9): tee stdout → logs/AT8/AT8_2_run.log
    _logf = open(RUNLOG, "w")
    class _Tee:
        def __init__(self, a, b): self.a, self.b = a, b
        def write(self, s):
            self.a.write(s); self.b.write(s); self.b.flush()
        def flush(self):
            self.a.flush(); self.b.flush()
    sys.stdout = _Tee(sys.__stdout__, _logf)
    logline(f"AT8_2 (T8.2) start {datetime.now(timezone.utc).isoformat()} "
            f"py={sys.version.split()[0]} torch={torch.__version__} (CPU, windowed recovery, "
            f"rev={REVISION}, machine v3)")

    tr, tr_done = init_trackio(t0, os.path.abspath(__file__))

    curves_series, e64_sha = load_dip_machinery()
    logline(f"S7: e64_gpu.py sha {e64_sha[:16]} PASS")

    # ---- S1: config ----
    cfg = AutoConfig.from_pretrained(MODEL)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None)
    sub = getattr(cfg, "text_config", None)
    if n_layers is None and sub is not None:   # qwen3_5: вложенный конфиг
        n_layers = getattr(sub, "num_hidden_layers", n_layers)
        hidden = getattr(sub, "hidden_size", hidden)
        vocab = getattr(sub, "vocab_size", vocab)
    s1 = dict(num_hidden_layers=n_layers, hidden_size=hidden,
              vocab_size=vocab, model_type=getattr(cfg, "model_type", None),
              dtype="fp32", eval=True)
    assert n_layers == 24 and hidden == 1024, f"S1 fail: {s1}"
    logline(f"S1: config {s1} PASS")

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    attn_impl = getattr(m.config, "_attn_implementation", "unknown")  # C8: какой керн-путь реально
    logline(f"C8: attn_impl={attn_impl} (fla заглушен None — CPU-fallback)")
    tstage(tr, t0, "model_loaded")

    # ---- S3: доки ----
    docs, names, tk = load_docs()
    dsha = docs_sha256(docs, names)
    lens = [len(d) for d in docs]
    logline(f"S3: {len(docs)} docs lens={lens} docs_sha256={dsha[:16]} pool_fp={POOL_FP}")
    phrase_ids = PHRASE_IDS
    pcheck = tk(PHRASE, add_special_tokens=False)["input_ids"]
    assert pcheck == phrase_ids, f"phrase-токены не совпадают: {pcheck} != {phrase_ids}"
    logline(f"phrase: {len(phrase_ids)} токенов {phrase_ids} (сверено с токенизатором)")
    tstage(tr, t0, "docs_loaded")

    # ---- два прохода: forward → dip-кривые по d (скаляры) → store ----
    res = {}
    if os.path.exists(STORE):
        res = json.load(open(STORE))
        # C4 (v3): store несёт отпечаток конфигурации; чужой/старый store — STOP
        cfg_fp = store_config_fp(MODEL, LAYER, phrase_ids, N_DOSE, W_SET, W_AXIS,
                                 EPS_MIN, THR, D_MIN_MEAS, e64_sha, POOL_FP, REVISION)
        if "__config__" not in res:
            raise SystemExit("C4: store без отпечатка конфигурации (__config__) — это "
                             "frozen r1-store: перепуск поверх frozen-артефакта запрещён; "
                             "сначала заархивировать/удалить store")
        if res["__config__"] != cfg_fp:
            raise SystemExit(f"C4: store-конфигурация != текущая: "
                             f"{json.dumps(res['__config__'], ensure_ascii=False)} != {cfg_fp} — "
                             f"resume чужого store остановлен")
        logline(f"C4: store-конфигурация совпадает ({res['__config__']['revision']})")
    # r2 (AGENTS п.2): решение «что пересчитывать» — чистая функция resume_plan
    # (протестирована exec-тестом experiments/at8/test_at8_2_resume.py)
    plan = resume_plan(names, N_DOSE, res)
    if res:
        logline(f"resume: в store {len(res)} записей, план resume: {plan['n_resume']} "
                f"(ctrl={sum(plan['ctrl_resume'].values())}, "
                f"dose={sum(plan['dose_resume'].values())})")
    for di, ids in enumerate(docs):
        nm = names[di]
        T = len(ids)
        cut = T // 2
        # контроль: один forward на док (одна статистика на обеих сторонах)
        if not plan["ctrl_resume"][nm]:
            nll_c, h0 = measure(m, ids, torch)
            csha = sha256_np(h0)
            dip_ctrl_curve = {}
            dip_global_c = float(curves_series(h0)["dip"])
            for W in W_SET:
                D_max = T - cut - W
                if D_max < 0:
                    dip_ctrl_curve[str(W)] = None
                    continue
                dip_ctrl_curve[str(W)] = [
                    win_dips(curves_series, h0, cut + d, W) for d in range(D_max + 1)]
            res[f"{nm}__ctrl"] = dict(T=T, cut=cut, nll=round(nll_c, 6),
                                      dip_ctrl=dip_ctrl_curve,
                                      dip_control_whole=round(dip_global_c, 6),
                                      h23_sha=csha)
            logline(f"  {nm} ctrl: T={T} cut={cut} nll={nll_c:.4f} "
                    f"dip_control_whole={dip_global_c:+.4f}")
            json.dump(res, open(STORE, "w"), default=float)
            del h0  # h23 не храним (двухпроходность) — dip-кривые уже в store

        for n in N_DOSE:
            if plan["dose_resume"][(nm, n)]:
                continue
            L_n = len(phrase_ids) * n
            inj = ids[:cut] + phrase_ids * n + ids[cut:]
            nll_i, h1 = measure(m, inj, torch)
            dip_global = float(curves_series(h1)["dip"])
            dip_inj_curve = {}
            for W in W_SET:
                D_max = T - cut - W
                if D_max < 0:
                    dip_inj_curve[str(W)] = None
                    continue
                dip_inj_curve[str(W)] = [
                    win_dips(curves_series, h1, cut + L_n + d, W) for d in range(D_max + 1)]
            res[f"{nm}__n{n}"] = dict(T=T, cut=cut, L_n=L_n, nll=round(nll_i, 6),
                                      dip_inj=dip_inj_curve,
                                      dip_global=round(dip_global, 6))
            logline(f"  {nm} n={n}: L_n={L_n} len(inj)={len(inj)} nll={nll_i:.4f} "
                    f"dip_global={dip_global:+.4f}")
            tr.log({"doc": di + 1, "n": n, "nll": nll_i,
                    "dip_global": dip_global}, step=int(time.time() - t0))
            json.dump(res, open(STORE, "w"), default=float)
            del h1
    tstage(tr, t0, "forwards_done")

    # C4 (v3): отпечаток конфигурации — в store (первый раз / при совпадении — переписать
    # тем же значением идемпотентно)
    res["__config__"] = store_config_fp(MODEL, LAYER, phrase_ids, N_DOSE, W_SET, W_AXIS,
                                        EPS_MIN, THR, D_MIN_MEAS, e64_sha, POOL_FP, REVISION)
    json.dump(res, open(STORE, "w"), default=float)

    # ---- S2: полнота ----
    expected = set()
    for nm in names:
        expected.add(f"{nm}__ctrl")
        for n in N_DOSE:
            expected.add(f"{nm}__n{n}")
    have = {k for k in res.keys() if k != "__config__"}
    assert have == expected, f"S2 fail: missing={expected - have} extra={have - expected}"
    logline(f"S2: полнота {len(have)}/{len(expected)} PASS")

    # ---- S2b: одна статистика контроля на обеих сторонах (р1 MINOR-3) ----
    # контроль считается один раз на док и переиспользуется по всем дозам:
    # все записи доз ссылаются на тот же h0 (в анализе — одна dip_ctrl-кривая
    # на (doc,W) для всех n). control_sha256 — h23_sha по doc из store.
    s2b = {nm: res[f"{nm}__ctrl"]["h23_sha"] for nm in names}
    n_dose_reuse = len(names) * len(N_DOSE)
    logline(f"S2b: control_sha256 (1 на док, reuse={n_dose_reuse} доз): "
            + " ".join(f"{k}={v[:8]}" for k, v in s2b.items()))

    # ---- S4: nll finite ----
    s4_ok = all(np.isfinite(res[k]["nll"]) for k in have)
    logline(f"S4: nll finite в {sum(1 for k in have if np.isfinite(res[k]['nll']))}/{len(have)} "
            f"{'PASS' if s4_ok else 'FAIL'}")
    assert s4_ok, "S4 fail: nll не finite"

    # ---- анализ: ℓ_rec, dilution, оси, вердикт — МАШИНА v3 (часть 1) ----
    tstage(tr, t0, "analysis")
    cells = build_cells_from_store(res, names)
    meas = [c for c in cells.values() if "dip_inj_d0" in c and c.get("status") is None]
    logline(f"окна: {len(cells)} пар (doc,n,W); measurable={len(meas)}")

    # ---- S5: dip finite ----
    s5_ok = all(np.isfinite(c["dip_inj_d0"]) and np.isfinite(c["dip_ctrl_d0"]) and
                np.isfinite(c["dip_global"]) and np.isfinite(c["dip_control_whole"])
                for c in meas)
    logline(f"S5: dip finite в {len(meas)} measurable окнах {'PASS' if s5_ok else 'FAIL'}")
    assert s5_ok, "S5 fail"

    axA = axis_A(cells)
    axB = axis_B(cells)
    verdict, vdetail, diag = compute_verdict(axA, axB, cells)
    logline(f"ось A (W={W_AXIS}, возмущённая популяция, v3): ratio={axA['W64']['ratio']} "
            f"b={axA['W64']['b']} r2={axA['W64']['r2']} lrec={axA['W64']['lrec']}")
    logline(f"  censored_excluded (вторичный): lrec={axA['W64']['lrec_censored_excluded']}")
    logline(f"ось B (v3: возмущённые, W!=8): frac_to_global={axB['frac_to_global']} "
            f"(n={axB['n_to_global_pairs']}; legacy={axB['frac_to_global_legacy']})")

    # ---- вердикт (машина v3: NO_EFFECT > гейты > DIFFUSION > ELASTIC > DILUTION) ----
    match = (verdict == "ELASTIC")
    logline(f"VERDICT: {verdict} (match_pre_reg={match}): {vdetail}")
    logline(f"diag: {diag}")

    # ---- S8: in-process самопроверка механики E-71B (р1 MAJOR-2) ----
    # Честность (C3/ревью §6.1): это регресс-тест ИНДЕКСАЦИИ (два пути одного кода на
    # одних срезах — bit-идентичность по построению), НЕ нуль и НЕ кросс-чек E-71B-
    # конвенции. Честный кросс-чек: AST-сплайс оконного замера из
    # experiments/e71_80/exp71_soliton.py (G1′ разрешает, файл в git) + его sha —
    # P1-перепрогон (disk-only недоступен).
    assert "wiki_1" in names, "S8: wiki_1 отсутствует в пуле (фиксированный пул, ТЗ §2)"
    di1 = names.index("wiki_1")
    ids1 = docs[di1]
    T1 = len(ids1); cut1 = T1 // 2
    L8 = len(phrase_ids) * 8
    c8 = res[f"wiki_1__ctrl"]
    i8 = res[f"wiki_1__n8"]
    dip_post_e71b = i8["dip_inj"]["64"][0]        # общий путь: win_dips(h1, cut1+L8, 64)
    dip_ctrl_e71b = c8["dip_ctrl"]["64"][0]       # общий путь: win_dips(h0, cut1, 64)
    # независимый пересчёт по определению E-71B на тех же h23 (один forward):
    n0chk, h0chk = measure(m, ids1, torch)
    n1chk, h1chk = measure(m, ids1[:cut1] + phrase_ids * 8 + ids1[cut1:], torch)
    cw_e71 = float(curves_series(h1chk[cut1 + L8:cut1 + L8 + 64])["dip"])
    kw_e71 = float(curves_series(h0chk[cut1:cut1 + 64])["dip"])
    dev_post = abs(cw_e71 - dip_post_e71b)
    dev_ctrl = abs(kw_e71 - dip_ctrl_e71b)
    s8 = dict(doc="wiki_1", n=8, W=64, d=0,
              dip_post_window_store=round(dip_post_e71b, 6),
              dip_control_window_store=round(dip_ctrl_e71b, 6),
              dip_post_window_recompute=round(cw_e71, 6),
              dip_control_window_recompute=round(kw_e71, 6),
              dev_post=dev_post, dev_ctrl=dev_ctrl,
              nll_recompute=(round(n0chk, 6), round(n1chk, 6)),
              tol=1e-12,
              ok=bool(dev_post <= 1e-12 and dev_ctrl <= 1e-12),
              note=("регресс-тест индексации (C3: тавтология для E-71B-конвенции — "
                    "обе руки в одной конвенции); честный кросс-чек — AST-сплайс "
                    "exp71_soliton.py + sha, P1-перепрогон"))
    logline(f"S8: E-71B in-process wiki_1 n=8 W=64: dev_post={dev_post:.2e} "
            f"dev_ctrl={dev_ctrl:.2e} pass={s8['ok']}")
    assert s8["ok"], "S8 fail: обобщение сломало механику E-71B"

    # ---- артефакты ----
    tstage(tr, t0, "artifact")
    claim = dict(
        experiment="AT8_2", cell="T8.2 windowed recovery + dose",
        tz_version="v1.3 (17.11, машина v3 по внешнему ревью AT8_2_review.md)",
        date=datetime.now(timezone.utc).isoformat(),
        setup=dict(model=MODEL, layer=LAYER, phrase=PHRASE, phrase_ids=phrase_ids,
                   L_n={f"n{n}": len(phrase_ids) * n for n in N_DOSE},
                   W_set=W_SET, W_axis=W_AXIS, n_dose=N_DOSE, use_cache=False,
                   threads="run_one.sh OMP/MKL/OPENBLAS=6", device="cpu",
                   attn_impl=attn_impl,          # C8: какой керн-путь реально исполнялся
                   e64_gpu_sha=e64_sha, trackio_run=f"at8_2_recovery_{REVISION}"),
        docs=dict(names=names, lens=lens, docs_sha256=dsha, pool_fp=POOL_FP,
                  control_sha256=s2b),
        gates=dict(S1=s1, S2="pass", S2b=dict(control_sha256=s2b,
                   n_dose_reuse=n_dose_reuse, note="контроль — 1 forward на док, "
                   "переиспользуется по всем дозам (одна статистика на обеих сторонах)"),
                   S3=dict(docs_sha256=dsha, pool_fp=POOL_FP),
                   S4="pass" if s4_ok else "fail",
                   S5="pass" if s5_ok else "fail",
                   S7=dict(e64_gpu_sha=e64_sha, ok=True), S8=s8,
                   forwards_note="42 форварда = 40 по сетке (8 доков × 5) + 2 в S8 "
                                 "(ревью §6.7/§D: расхождение с «40» объяснено)"),
        prereg=dict(eps_min=EPS_MIN, thr=THR, D_max_rule="T - cut - W",
                    d_min_meas=D_MIN_MEAS,
                    verdict_thresholds=dict(
                        diffusion="ratio>=4 & b>0 & r2>=0.6 (W=64, возмущённая популяция)",
                        elastic="ratio<=2 & frac_to_global<=0.4 & n_pos>=2 (W=64; n_pos — "
                                "дозы с восстановившимися клетками, машина v3)",
                        dilution="frac_to_global>0.6 (возмущённые, W!=8, машина v3)",
                        no_effect="большинство клеток W=64 — no_effect (below_resolution, "
                                  "paper §6 п.25)",
                        not_resolved="ratio=None (отсутствующая доза) ИЛИ <6 восстановленных "
                                     "пар (doc,n) на W=64 ИЛИ большинство not_recovered по "
                                     "возмущённой популяции (машина v3, ревью A1/A3/A4)"),
                    note=("р1 MINOR-2 (ограничение мощности) + v3-ограничения (ревью): "
                          "ось A неинформативна в маловозмущённом режиме; W64-ось — 5 самых "
                          "длинных доков (селекция по длине, T>=143 — ревью §6.5); ℓ_rec у потолка "
                          "D_max (suffix-max при пороге ниже шумового пола); ℓ_rec при W64 не "
                          "разрешает восстановление короче ~W/2 (W-сглаживание); доза в "
                          "токенах (L_n=9n) ≠ доза в натах (D(n) насыщается: per-token 4.2→"
                          "0.61 nat) — чтение «доза» ограничено; ось B смещена (B1/B2): "
                          "recovered-клетки вносят в неё нули, допуск оконного масштаба. "
                          "Полные правки — P1-перепрогон (per-position ряды, δ_min, "
                          "filler-контроль, 4 слоя)")),
        lrec_by_n=dict(W64=axA["W64"]["lrec"],
                       W64_censored_excluded=axA["W64"]["lrec_censored_excluded"],
                       combined_secondary=axA["combined_secondary"]["lrec"],
                       combined_secondary_censored_excluded=
                       axA["combined_secondary"]["lrec_censored_excluded"]),
        axis_A=axA, axis_B=axB,
        diagnostics=diag,
        cells=cells,
        verdict=dict(value=verdict, match_pre_reg=match, detail=vdetail),
        summary=(f"T8.2 (r{REVISION[1:]}, machine v3): {verdict}. {vdetail}. "
                 f"42 форварда (40 сетка + 2 S8), wall {time.time()-t0:.0f}s."),
        wall_s=round(time.time() - t0, 1),
    )

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating,)):
            o = float(o)
        if isinstance(o, (np.integer,)):
            o = int(o)
        if isinstance(o, (np.bool_,)):
            o = bool(o)
        if isinstance(o, float) and not np.isfinite(o):
            o = "inf" if o == float("inf") else "-inf"
        return o
    claim = _san(claim)
    json.dump(claim, open(CLAIM, "w"), indent=1, ensure_ascii=False)
    csha = sha256_file(CLAIM)
    man = dict(claim_file_sha256=csha, store_file_sha256=sha256_file(STORE),
               e64_gpu_sha=e64_sha, docs_sha256=dsha, pool_fp=POOL_FP,
               model=MODEL, date=datetime.now(timezone.utc).isoformat(),
               verdict=verdict, wall_s=claim["wall_s"])
    json.dump(man, open(MANIFEST, "w"), indent=1, ensure_ascii=False)
    logline(f"written: {CLAIM} (sha {csha[:16]}), {MANIFEST}")

    tr.log({"finished_at": datetime.now(timezone.utc).isoformat(), "status": "ok",
            "verdict": verdict, "match_pre_reg": match,
            "claim_sha": csha[:16], "wall_sec": round(time.time() - t0, 1),
            "log_md5": md5_file(RUNLOG)},          # AGENTS п.5 (r2 MINOR-2)
           step=int(time.time() - t0))
    tr.finish()
    tr_done[0] = True
    logline(f"trackio: finish ok (status=ok, verdict={verdict})")
    logline(f"AT8_2_DONE verdict={verdict} match_pre_reg={match} (wall {time.time()-t0:.0f}s)")


# =====================================================================
# ЧАСТЬ 3 — DISK-ONLY РЕАНАЛИЗ v3 (машина v3 поверх committed r1-store;
# verbatim из AT8_2_v2_reanalysis.py + C1 machine_sha + C5 gates/prereg-source
# + allow_extra G14-ключи)
# =====================================================================

R1_CLAIM = CLAIM
R1_MANIFEST = MANIFEST
R1_STORE = STORE
V3_CLAIM = os.path.join(OUT_DIR, "AT8_2_RECOVERY_v3.json")
V3_MANIFEST = os.path.join(OUT_DIR, "AT8_2_v3_manifest.json")
V3_RUNLOG = os.path.join(_ROOT, "logs", "AT8", "AT8_2_v3_run.log")
G14_EXTRA = frozenset({"object", "stat", "layer", "frame"})


def doc_bootstrap(cells, n_resamples=2000, seed=0):
    """P0.2 (ревью §6.2): 116 клеток ≠ 116 независимых наблюдений (общий h0,
    вложенность W/n) — биномиальный CI невалиден; доковый bootstrap (доки — единица
    ресемплинга). Статистики: медианы ℓ_rec по дозам на оси A (W=w_axis,
    возмущённые), ratio, frac_to_global (ось B). 95% CI = перцентили 2.5/97.5.
    Честность: при 5 доках CI широкий — это и есть ответ на «нет CI»."""
    rng = np.random.default_rng(seed)
    docs_a = sorted({c["doc"] for c in cells.values()
                     if c["W"] == W_AXIS and _perturbed(c.get("rec_status"))
                     and "l_rec" in c})
    docs_b = sorted({c["doc"] for c in cells.values()
                     if "to_global" in c and _perturbed(c.get("rec_status"))
                     and c["W"] != W_NOISY})
    if not docs_a or not docs_b:
        return dict(n_resamples=n_resamples, seed=seed,
                    ratio_ci95=None, frac_ci95=None, lrec_ci95_by_n=None,
                    note="недостаточно доков для bootstrap")
    ia = {d: i for i, d in enumerate(docs_a)}
    ib = {d: i for i, d in enumerate(docs_b)}
    ratios, fracs, lrecs = [], [], {n: [] for n in N_DOSE}
    for _ in range(n_resamples):
        sel_a = set(rng.choice(len(docs_a), len(docs_a), replace=True).tolist())
        sel_b = set(rng.choice(len(docs_b), len(docs_b), replace=True).tolist())
        sub_a = [c for c in cells.values() if c["W"] == W_AXIS
                 and _perturbed(c.get("rec_status")) and "l_rec" in c
                 and ia[c["doc"]] in sel_a]
        sub_b = [c for c in cells.values() if "to_global" in c
                 and _perturbed(c.get("rec_status")) and c["W"] != W_NOISY
                 and ib[c["doc"]] in sel_b]
        if sub_a:
            lr = {n: (float(np.median([c["l_rec"] for c in sub_a if c["n"] == n]))
                      if any(c["n"] == n for c in sub_a) else None)
                  for n in N_DOSE}
            ratios.append(_ratio_lr(lr))
            for n in N_DOSE:
                if any(c["n"] == n for c in sub_a):
                    lrecs[n].append(lr[n])
        if sub_b:
            fracs.append(float(np.mean([c["to_global"] for c in sub_b])))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4),
                    round(float(np.percentile(v, 97.5)), 4)] if v else None
    return dict(n_resamples=n_resamples, seed=seed,
                docs_axis_A=docs_a, docs_axis_B=docs_b,
                ratio_ci95=ci([r for r in ratios if r is not None]),
                frac_ci95=ci(fracs),
                lrec_ci95_by_n={n: ci(v) for n, v in lrecs.items()})


def reanalysis_main():
    t0 = time.time()
    os.makedirs(os.path.dirname(V3_RUNLOG), exist_ok=True)
    _logf = open(V3_RUNLOG, "w")

    def logline3(s):
        print(s, flush=True)
        _logf.write(s + "\n")
        _logf.flush()

    logline3(f"AT8_2 r3 (T8.2) disk-only reanalysis start "
             f"{datetime.now(timezone.utc).isoformat()} py={sys.version.split()[0]} "
             f"np={np.__version__} (machine v3, без модели/перепрогона)")

    # ---- 1. sha-верификация ----
    man = json.load(open(R1_MANIFEST))
    r1_claim = json.load(open(R1_CLAIM))
    store_sha = sha256_file(R1_STORE)
    r1_claim_sha = sha256_file(R1_CLAIM)
    assert man["store_file_sha256"] == store_sha, \
        f"STOP: store sha {store_sha[:16]} != manifest {man['store_file_sha256'][:16]}"
    assert man["claim_file_sha256"] == r1_claim_sha, \
        f"STOP: r1-claim sha {r1_claim_sha[:16]} != manifest {man['claim_file_sha256'][:16]}"
    logline3(f"sha-верификация: store {store_sha[:16]} / r1-claim {r1_claim_sha[:16]} "
             f"— совпадает с manifest (r1 прогон не тронут)")

    names = r1_claim["docs"]["names"]
    store = json.load(open(R1_STORE))

    # ---- 2. пересчёт ячеек машиной v3 + сверка с r1 ----
    cells = build_cells_from_store(store, names, n_dose=N_DOSE, w_set=W_SET,
                                   eps_min=EPS_MIN, thr=THR, d_min_meas=D_MIN_MEAS)
    n_diff, n_g14 = 0, 0
    for k, c1 in r1_claim["cells"].items():
        ok, why = cells_equal(c1, cells.get(k, {}), allow_extra=G14_EXTRA)
        if not ok:
            n_diff += 1
            logline3(f"  RASXOD {k}: {why}")
        elif (set(cells.get(k, {})) - set(c1)) & G14_EXTRA:
            n_g14 += 1
    extra = set(cells) - set(r1_claim["cells"])
    assert n_diff == 0 and not extra, \
        f"STOP: расхождения r1-claim vs пересчёт машиной v3: {n_diff} ячеек, extra={extra}"
    n_meas = sum(1 for c in cells.values() if "dip_inj_d0" in c and c.get("status") is None)
    logline3(f"ячейки: {len(cells)} (measurable={n_meas}) — пересчёт == r1-claim "
             f"(0 расхождений; +G14-provenance ключи в {n_g14} клетках — документировано, C11)")

    # ---- 3. оси + вердикт (машина v3) ----
    axA = axis_A(cells, n_dose=N_DOSE, w_set=W_SET, w_axis=W_AXIS)
    axB = axis_B(cells)
    verdict, vdetail, diag = compute_verdict(axA, axB, cells, n_dose=N_DOSE, w_axis=W_AXIS)
    match = (verdict == "ELASTIC")
    logline3(f"ось A (W={W_AXIS}, возмущённая популяция, v3): "
             f"lrec={axA['W64']['lrec']} ratio={axA['W64']['ratio']} "
             f"b={axA['W64']['b']} r2={axA['W64']['r2']}")
    logline3(f"  censored_excluded (вторичный): lrec={axA['W64']['lrec_censored_excluded']}")
    logline3(f"ось B (v3): frac_to_global={axB['frac_to_global']} "
             f"(n={axB['n_to_global_pairs']}; legacy pooled={axB['frac_to_global_legacy']})")
    logline3(f"diag: {diag}")
    logline3(f"VERDICT: {verdict} (match_pre_reg={match}): {vdetail}")

    # ---- 3b. доковый bootstrap (P0.2 ревью: 116 клеток ≠ 116 независимых —
    # общий h0, вложенность W/n; биномиальный CI невалиден) ----
    boot = doc_bootstrap(cells)
    logline3(f"bootstrap (доковый, {boot['n_resamples']} resamples, seed {boot['seed']}): "
             f"ratio_ci95={boot['ratio_ci95']} frac_ci95={boot['frac_ci95']} "
             f"lrec_ci95_by_n={boot['lrec_ci95_by_n']}")

    # ---- 4. артефакт ----
    claim = dict(
        experiment="AT8_2", cell="T8.2 windowed recovery + dose",
        revision="v3 (disk-only reanalysis, machine v3; r1-claim — снапшот прогона)",
        tz_version="v1.3 (17.11, по внешнему ревью docs/AT8_2_review.md: машина v3)",
        date=datetime.now(timezone.utc).isoformat(),
        machine_sha256=sha256_file(os.path.abspath(__file__)),   # C1 (v3): sha машины
        base_run=dict(
            r1_claim_file_sha256=r1_claim_sha, store_file_sha256=store_sha,
            r1_run_name="at8_2_recovery_r1", r1_wall_s=r1_claim["wall_s"],
            note=("прогон не тронут: store/claim r1 в исходном виде; v3 — пересчёт "
                  "агрегатов (оси/вердикт) из того же store машиной v3")),
        setup=r1_claim["setup"],
        docs=r1_claim["docs"],
        # C5 (v3): gates/prereg — дословно из r1_claim (зафиксированы ДО r1-прогона;
        # реанализ не меняет пре-рег — текст промахов v1/v2-машины остаётся пре-регом)
        gates=r1_claim["gates"],
        prereg=r1_claim["prereg"],
        gates_prereg_source="r1_claim (пре-рег фиксирован до r1-прогона; "
                            "r2/v3 — пересчёт машиной, не новая пре-рег — C5)",
        v3_fixes=dict(
            A1=("гейт «мало пар» — по (doc,n) с rec_status=='recovered' на W=64 "
                "(цензурированный провал — не восстановление); на store: 4 < 6 → "
                "NOT_RESOLVED (r1/v2 читали l_rec>0: 19 пар, гейт проходил)"),
            A2=("no_effect-нули исключены из основной медианы/ratio; исход NO_EFFECT "
                "(below_resolution, paper §6 п.25); на store: ne_share_axis=0.05 — "
                "не срабатывает"),
            A3=("ratio=None при отсутствующей дозе → NOT_RESOLVED (не 0.0); на store: "
                "не активно"),
            A4=("majority-гейт — по возмущённой популяции {pert0>=eps_min} без W=8: 38/58=0.655 "
                ">0.5 → NOT_RESOLVED (старое чтение 55/116=0.474 — diagnostic)"),
            B3B6=("ось B: возмущённые клетки, W!=8 (W8_noisy_flag рабочая: W=8 "
                  "исключена и из majority-гейта; no_effect исключены) — frac 0.3276 "
                  "(legacy pooled 0.2328); W64-только 0.684; frac по W монотонно растёт "
                  "0.000/0.125/0.3125/0.650 — подпись смещения B2/B3 подтверждена на "
                  "store (ревью §2 направленная проверка)"),
            C1C2C6C9C10C11=("hygiene: статус по rounded pert0; tol 1e-9 в сравнении "
                           "остатков; ratio inf→None+флаг; SS_tot=0→r2=None; assert "
                           "длин кривых; G14-координаты в клетках"),
        ),
        lrec_by_n=dict(W64=axA["W64"]["lrec"],
                       W64_censored_excluded=axA["W64"]["lrec_censored_excluded"],
                       combined_secondary=axA["combined_secondary"]["lrec"],
                       combined_secondary_censored_excluded=
                       axA["combined_secondary"]["lrec_censored_excluded"]),
        axis_A=axA, axis_B=axB,
        bootstrap=boot,   # P0.2: доковый bootstrap (доки как единица ресемплинга)
        diagnostics=diag,
        cells=cells,
        verdict=dict(value=verdict, match_pre_reg=match, detail=vdetail),
        summary=(f"T8.2 v3 (disk-only, machine v3): {verdict}. {vdetail}. П0-проверки "
                 f"ревью подтверждены на store: #recovered(W64)=4, no_effect=32, "
                 f"nr-по-возмущённым 38/58 (без W=8, B3), frac по W 0.000/0.125/0.3125/0.650, "
                 f"D(n)=75.8/80.1/84.4/88.3 nat (формула ревью: D(n) = CE(inj)·(T+L_n−1) − "
                 f"CE(ctrl)·(T−1); насыщается: per-token 4.21→0.61 nat). Честный статус "
                 f"r1/v2 — NOT_RESOLVED/below_resolution, не ELASTIC. 40 форвардов + 2 S8 "
                 f"(r1-прогон, wall {r1_claim['wall_s']:.0f}s); v3 — disk-only, "
                 f"{time.time()-t0:.1f}s."),
        wall_s=round(time.time() - t0, 1),
    )

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating,)):
            o = float(o)
        if isinstance(o, (np.integer,)):
            o = int(o)
        if isinstance(o, (np.bool_,)):
            o = bool(o)
        if isinstance(o, float) and not np.isfinite(o):
            o = "inf" if o == float("inf") else "-inf"
        return o
    claim = _san(claim)
    json.dump(claim, open(V3_CLAIM, "w"), indent=1, ensure_ascii=False)
    csha = sha256_file(V3_CLAIM)
    man_v3 = dict(claim_file_sha256=csha, store_file_sha256=store_sha,
                  r1_claim_file_sha256=r1_claim_sha, r1_manifest_file_sha256=
                  sha256_file(R1_MANIFEST),
                  machine_sha256=claim["machine_sha256"],
                  e64_gpu_sha=r1_claim["setup"]["e64_gpu_sha"],
                  docs_sha256=r1_claim["docs"]["docs_sha256"],
                  pool_fp=r1_claim["docs"]["pool_fp"],
                  model=r1_claim["setup"]["model"],
                  date=datetime.now(timezone.utc).isoformat(),
                  verdict=verdict, wall_s=claim["wall_s"])
    json.dump(man_v3, open(V3_MANIFEST, "w"), indent=1, ensure_ascii=False)
    logline3(f"written: {V3_CLAIM} (sha {csha[:16]}), {V3_MANIFEST}")
    logline3(f"AT8_2_V3_DONE verdict={verdict} match_pre_reg={match} "
             f"(wall {time.time()-t0:.1f}s)")
    _logf.close()


# =====================================================================
# ЧАСТЬ 4 — КАЛИБРОВКА МАШИНЫ (P0.5 ревью: синтетические store с известным
# ответом; confusion matrix машина v2 [frozen, git] vs v3)
# =====================================================================

V2_GIT_REF = ("c73d8f33687daae28a1747eef95dd0d80a9fe40b:"
              "experiments/at8/at8_2_analysis.py")   # frozen машина v2 (pinned-коммит c73d8f3;
                                                    # "HEAD" не годится: после коммита слияния
                                                    # файла в HEAD не будет)
CALIB_OUT = os.path.join(OUT_DIR, "AT8_2_machine_calibration.json")


def _load_v2_machine(tmpdir):
    """Frozen машина v2 — точный код из git (pinned-коммит c73d8f3)."""
    import subprocess
    src = subprocess.run(["git", "show", V2_GIT_REF], cwd=_ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert hashlib.sha256(src.encode()).hexdigest() == \
        "3c8212e86f585387f1c2a3238cb45b3e109e00d11cf89bd1a5070d206ae8f056", \
        "frozen v2-машина: sha не совпадает (pinned c73d8f3)"
    path = os.path.join(tmpdir, "at8_2_analysis_v2_frozen.py")
    with open(path, "w") as f:
        f.write(src)
    sys.path.insert(0, tmpdir)
    import importlib
    mod = importlib.import_module("at8_2_analysis_v2_frozen")
    sys.path.pop(0)
    return mod


def synth_cells(mod, scenario, seed=0, n_docs=8, T=256, sigma=None,
                A_el=0.10, A_shift=0.015, lam_el=6.0, lam_diff=4.0):
    """Синтетические клетки с известным ответом (ревью §7 P0.5).

    Модель: dip-кривые ck[d] = -0.2 + e_d (e_d — сглаженный шум, AR(1) phi=0.8),
    ik[d] = ck[d] + s_n(d) + f_d (f_d — независимый AR(1)-шум, тот же sigma).
    Сценарии (истина) и sigma (намеренно разные — режим шума под истину сценария):
      null       — s=0 (эффекта нет); sigma=0.02;
      elasticity — s(d)=A_el·exp(-d/lam_el), lam_el=6 — быстрое восстановление,
                   доза-независимо → ELASTIC; sigma=0.005 (хвостовый шум ниже
                   порога 0.25·pert0 — восстановление разрешено);
      shift      — s(d)=A_shift (постоянный сдвиг — восстановления НЕТ) →
                   NOT_RESOLVED; sigma=0.02 (режим store: хвостовый шум сопоставим
                   со сдвигом);
      diffusion  — s(d)=A_el·exp(-d/(lam_diff·n)) — длина восстановления растёт с
                   дозой → DIFFUSION; sigma=0.005 (как elasticity).
    """
    if sigma is None:
        sigma = {"null": 0.02, "elasticity": 0.005, "shift": 0.02,
                 "diffusion": 0.005}[scenario]
    rng = np.random.default_rng(seed)
    doc_off = {i: float(rng.normal(0, 0.01)) for i in range(n_docs)}
    cells = {}
    for i in range(n_docs):
        base = -0.2 + doc_off[i]
        for n in N_DOSE:
            for W in W_SET:
                D = T - T // 2 - W
                # AR(1)-шум (сглаженный): станц. std ≈ sigma
                phi = 0.8
                e = np.zeros(D + 1)
                f = np.zeros(D + 1)
                for d in range(1, D + 1):
                    e[d] = phi * e[d - 1] + rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
                    f[d] = phi * f[d - 1] + rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
                ck = base + e
                if scenario == "null":
                    s = np.zeros(D + 1)
                elif scenario == "elasticity":
                    s = A_el * np.exp(-np.arange(D + 1) / lam_el)
                elif scenario == "shift":
                    s = np.full(D + 1, A_shift)
                elif scenario == "diffusion":
                    s = A_el * np.exp(-np.arange(D + 1) / (lam_diff * n))
                else:
                    raise ValueError(scenario)
                ik = ck + s + f
                dip_cw = base + float(rng.normal(0, sigma / 3))
                dip_g = base + ({"null": 0.0, "elasticity": 0.004,
                                 "shift": 0.02, "diffusion": 0.006}[scenario]) \
                    + float(rng.normal(0, sigma / 3))
                cells[f"doc{i}|n{n}|W{W}"] = mod.classify_cell(
                    f"doc{i}", n, W, T, dip_g, dip_cw, ck.tolist(), ik.tolist())
    return cells


def run_calibration():
    """Confusion matrix: 4 сценария × 2 машины (v2 frozen / v3). Артефакт JSON."""
    import tempfile
    t0 = time.time()
    v2 = _load_v2_machine(tempfile.mkdtemp(prefix="at82v2_"))
    out = dict(
        experiment="AT8_2", what="калибровка машины на синтетических store "
                                 "(ревью §7 P0.5: нуль / истинная упругость / вечный "
                                 "сдвиг / диффузия)",
        date=datetime.now(timezone.utc).isoformat(),
        machine_v2_sha256="3c8212e86f585387f1c2a3238cb45b3e109e00d11cf89bd1a5070d206ae8f056",
        machine_v3_sha256=sha256_file(os.path.abspath(__file__)),
        scenarios={},
    )
    truth = {"null": "NOT_RESOLVED/NO_EFFECT (эффекта нет)",
             "elasticity": "ELASTIC (быстрое восстановление)",
             "shift": "NOT_RESOLVED (вечный сдвиг — восстановления нет)",
             "diffusion": "DIFFUSION (длина растёт с дозой)"}
    for sc in ("null", "elasticity", "shift", "diffusion"):
        row = {}
        for name, mod in (("v2_frozen", v2), ("v3", sys.modules[__name__])):
            cells = synth_cells(mod, sc)
            axA = mod.axis_A(cells)
            axB = mod.axis_B(cells)
            v, det, diag = mod.compute_verdict(axA, axB, cells)
            row[name] = dict(verdict=v, detail=det,
                             lrec_w64=axA["W64"]["lrec"],
                             ratio=axA["W64"]["ratio"],
                             frac=diag and axB["frac_to_global"],
                             diag=diag)
        out["scenarios"][sc] = dict(truth=truth[sc], **row)
    out["scenarios"]["shift"]["degeneracy_note"] = (
        "ревью §1: условие ratio<=2 ВЫПОЛНЕНО при полном невосстановлении "
        "(ratio=1.0 в этом сценарии); спасает только majority-гейт (share 0.93 "
        "здесь). На реальном store share=0.474 < 0.5 — гейт промахнулся → v2-"
        "артефакт = ELASTIC; v3-гейты (A1/A4) дают NOT_RESOLVED disk-only — см. "
        "AT8_2_RECOVERY_v3.json")
    json.dump(out, open(CALIB_OUT, "w"), indent=1, ensure_ascii=False)
    for sc, row in out["scenarios"].items():
        print(f"{sc:12s} truth={row['truth']}")
        for m in ("v2_frozen", "v3"):
            print(f"   {m:10s}: {row[m]['verdict']:<14s} lrec={row[m]['lrec_w64']} "
                  f"ratio={row[m]['ratio']} frac={row[m]['frac']}")
    print(f"CALIB_DONE: {CALIB_OUT} (wall {time.time()-t0:.1f}s)")
    return out


if __name__ == "__main__":
    if "--reanalysis" in sys.argv:
        reanalysis_main()
    elif "--calibrate" in sys.argv:
        run_calibration()
    else:
        run_main()
