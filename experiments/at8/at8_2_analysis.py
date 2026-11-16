# -*- coding: utf-8 -*-
"""AT8_2 (T8.2) — общий анализный модуль (disk-safe, только numpy).

Единый фрагмент для прогона (AT8_2_recovery.py) и disk-only-реанализа
(AT8_2_v2_reanalysis.py) — AGENTS п.2: exec точного фрагмента, не копии.

r2-фиксы (внешнее ревью AT8_2, docs/dynamic/AT8_2_results_review.md):
- MAJOR-1: ось A — ОСНОВНАЯ популяция = ВСЕ измеримые (doc,W) с ℓ_rec (TZ §5:
  «по измеримым (doc,W)»; not_recovered имеют ℓ_rec=D_max — цензурированное
  значение, входит в медиану). Цензурированное-исключение (recovered+no_effect
  только) — вторичный вариант с явной меткой censored_excluded.
- MINOR-1: NOT_RESOLVED-гейт «большинство not_recovered» — по полной измеримой
  популяции (все W, D_max≥8 — та же популяция, что у оси B); «мало пар (doc,n)
  с ℓ_rec>0» — по основной оси (W=64). Чтение по W64-оси (75% not_recovered) —
  diagnostic в артефакте, не гейт (см. TZ v1.2 §5).
"""
import numpy as np

N_DOSE = [2, 4, 8, 16]
W_SET = [8, 16, 32, 64]
W_AXIS = 64
EPS_MIN = 0.01
THR = 0.25
D_MIN_MEAS = 8


def classify_cell(nm, n, W, T, dip_global, dip_control_whole, ck, ik,
                  eps_min=EPS_MIN, thr=THR, d_min_meas=D_MIN_MEAS):
    """Одна ячейка (doc,n,W) — точно как в r1-прогоне (сверено reвьюем: 0 расхождений).

    ck/ik — dip-кривые по d (None, если D_max<0) из store.
    """
    D_max = T - T // 2 - W
    cell = dict(doc=nm, n=n, W=W, D_max=D_max,
                dip_global=round(dip_global, 6),
                dip_control_whole=round(dip_control_whole, 6),
                W8_noisy_flag=(W == 8))
    if ck is None or ik is None:
        cell.update(status="not_measurable")
        return cell
    if any(v is None for v in ik + ck) or \
       any(not np.isfinite(v) for v in ik + ck):
        cell.update(status="not_finite")
        return cell
    cell.update(dip_inj_d0=round(ik[0], 6), dip_ctrl_d0=round(ck[0], 6))
    pert = [abs(ik[d] - ck[d]) for d in range(D_max + 1)]
    pert0 = pert[0]
    cell["pert0"] = round(pert0, 6)
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
    if D_max >= d_min_meas and "rec_status" in cell:
        res_ctrl = abs(ik[D_max] - ck[D_max])
        res_glob = abs(ik[D_max] - dip_global)
        cell["residual_ctrl"] = round(res_ctrl, 6)
        cell["residual_global"] = round(res_glob, 6)
        cell["to_global"] = bool(res_glob < res_ctrl and
                                 abs(dip_global - dip_control_whole) > thr * pert0)
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


def lrec_medians(cells, n, w, censored):
    """Медиана ℓ_rec по докам для дозы n при W=w.

    censored=False (ОСНОВНАЯ, TZ §5): все измеримые (doc,W) с ℓ_rec —
    включая not_recovered (ℓ_rec=D_max — цензурированное значение).
    censored=True (вторичный, справочно): только recovered/no_effect
    (not_recovered исключены — «censored excluded»).
    """
    vals = [c["l_rec"] for c in cells.values()
            if c["n"] == n and c["W"] == w and "l_rec" in c
            and (not censored or c.get("rec_status") in ("recovered", "no_effect"))]
    return round(float(np.median(vals)), 3) if vals else None


def _ratio_lr(lr):
    r = None
    l2, l16 = lr.get(2), lr.get(16)
    if l2 not in (None, 0) and l16 is not None:
        r = round(l16 / l2, 4)
    elif l2 == 0 and l16 is not None and l16 > 0:
        r = float("inf")
    elif l2 == 0 and (l16 in (None, 0)):
        r = 0.0
    return r


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
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return round(float(b), 4), round(float(a), 4), round(r2, 4)


def axis_A(cells, n_dose=N_DOSE, w_set=W_SET, w_axis=W_AXIS):
    """Ось A — ℓ_rec(n) по медиане доков. Основная популяция — полная (TZ §5, r2 MAJOR-1);
    censored_excluded — вторичный (справочно, явно помечен).
    Конвенция «медиана доков» (TZ v1.2): сначала редукция к уровню дока (для W=64 —
    одно значение на док; для combined — медиана по W внутри дока), потом медиана по докам."""
    def both(w):
        full = {n: lrec_medians(cells, n, w, censored=False) for n in n_dose}
        cens = {n: lrec_medians(cells, n, w, censored=True) for n in n_dose}
        b, a, r2 = _fit(full, n_dose)
        return dict(lrec=full, lrec_censored_excluded=cens,
                    ratio=_ratio_lr(full), b=b, a=a, r2=r2)
    w64 = both(w_axis)
    full = {n: lrec_medians_allw(cells, n, censored=False) for n in n_dose}
    cens = {n: lrec_medians_allw(cells, n, censored=True) for n in n_dose}
    b, a, r2 = _fit(full, n_dose)
    combined = dict(lrec=full, lrec_censored_excluded=cens,
                    ratio=_ratio_lr(full), b=b, a=a, r2=r2,
                    note="combined (справочно): медиана по W внутри дока, потом по докам (TZ v1.2)")
    return dict(W_axis=w_axis,
                W64=dict(ratio=w64["ratio"], b=w64["b"], a=w64["a"], r2=w64["r2"],
                         lrec=w64["lrec"],
                         lrec_censored_excluded=w64["lrec_censored_excluded"]),
                combined_secondary=combined)


def lrec_medians_allw(cells, n, censored):
    """Медиана ℓ_rec по докам (объединённо по W) — справочно (TZ §5)."""
    by_doc = {}
    for c in cells.values():
        if c["n"] == n and "l_rec" in c and \
           (not censored or c.get("rec_status") in ("recovered", "no_effect")):
            by_doc.setdefault(c["doc"], []).append(c["l_rec"])
    if not by_doc:
        return None
    per_doc = [float(np.median(v)) for v in by_doc.values()]
    return round(float(np.median(per_doc)), 3)


def axis_B(cells):
    """Ось B — доля to_global среди измеримых с D_max≥8 (TZ §5)."""
    tg = [c["to_global"] for c in cells.values() if "to_global" in c]
    return dict(frac_to_global=round(float(np.mean(tg)), 4) if tg else None,
                n_to_global_pairs=len(tg))


def compute_verdict(axis_A, axis_B, cells, n_dose=N_DOSE, w_axis=W_AXIS):
    """Вердикт (приоритет DIFFUSION > ELASTIC > DILUTION > NOT_RESOLVED, TZ §5).

    r2 MINOR-1: гейты NOT_RESOLVED по явным популяциям:
    - «мало пар (doc,n) с ℓ_rec>0» — по основной оси (W=w_axis);
    - «большинство not_recovered» — по полной измеримой популяции (все W,
      D_max≥8 — та же популяция, что у оси B). Чтение по W64-оси — diagnostic
      (axis_W64_notrecovered_share), не гейт (TZ v1.2 §5).
    Доп. консервативный гейт n_pos≥2 (в r1-прогоне, зафиксирован в TZ v1.2).
    """
    lrec_w64 = axis_A["W64"]["lrec"]
    ratio = axis_A["W64"]["ratio"]
    b = axis_A["W64"]["b"]
    r2 = axis_A["W64"]["r2"]
    f_tg = axis_B["frac_to_global"]

    n_pos = sum(1 for n in n_dose if (lrec_w64.get(n) or 0) > 0)
    n_meas_doc_n = len({(c["doc"], c["n"]) for c in cells.values()
                        if c["W"] == w_axis and c.get("l_rec", 0) > 0})
    with_status = [c for c in cells.values() if "rec_status" in c]
    n_notrec = sum(1 for c in with_status if c["rec_status"] == "not_recovered")
    n_w64_status = [c for c in with_status if c["W"] == w_axis]
    w64_notrec_share = (round(sum(1 for c in n_w64_status
                                  if c["rec_status"] == "not_recovered")
                              / len(n_w64_status), 4) if n_w64_status else None)

    diag = dict(n_pos=n_pos, n_meas_doc_n=n_meas_doc_n, n_notrec=n_notrec,
                n_meas_total=len(with_status),
                notrecovered_share_full=round(n_notrec / len(with_status), 4)
                if with_status else None,
                axis_W64_notrecovered_share=w64_notrec_share)

    r = ratio if ratio != float("inf") else None
    r_is_inf = (ratio == float("inf"))
    cond_diff = (r is not None and r >= 4) or r_is_inf
    verdict, vdetail = "NOT_RESOLVED", ""
    if cond_diff and b is not None and b > 0 and r2 is not None and r2 >= 0.6:
        verdict = "DIFFUSION"
        vdetail = f"ℓ_rec растёт с дозой (ratio={ratio}, b={b}, r2={r2}, W={w_axis})"
    elif (r is not None and r <= 2) and f_tg is not None and f_tg <= 0.4 and n_pos >= 2:
        verdict = "ELASTIC"
        vdetail = (f"ℓ_rec слабо зависит от n (ratio={ratio}, W={w_axis}, "
                   f"полная популяция), окно к контролю (frac_to_global={f_tg})")
    elif f_tg is not None and f_tg > 0.6:
        verdict = "DILUTION"
        vdetail = f"окно ≈ глобальное разбавление (frac_to_global={f_tg})"
    else:
        vdetail = (f"ratio={ratio} b={b} r2={r2} frac_to_global={f_tg} "
                   f"n_pos={n_pos} n_meas_doc_n={n_meas_doc_n} "
                   f"notrec={n_notrec}/{len(with_status)}")
    # гейты NOT_RESOLVED (r2 MINOR-1: явные популяции)
    if n_meas_doc_n < 6:
        verdict = "NOT_RESOLVED"
        vdetail += f" (мало пар (doc,n) с ℓ_rec>0 на W={w_axis}: {n_meas_doc_n} < 6)"
    if with_status and n_notrec > len(with_status) / 2:
        verdict = "NOT_RESOLVED"
        vdetail += f" (большинство not_recovered: {n_notrec}/{len(with_status)} по полной популяции)"
    return verdict, vdetail, diag


def resume_plan(names, n_dose, res):
    """Решение «что пересчитывать» при resume по ключам (TZ §6 S2) — чистая функция.

    res — существующий store ({} при fresh run). Возвращает:
    - ctrl_resume: {nm: bool} — контроль в store (пересчёт не нужен);
    - dose_resume: {(nm, n): bool} — доза в store;
    - n_resume: число переиспользуемых записей (для лога).
    Считаем только КЛЮЧИ: целостность содержимого store при resume — sha-верификация
    manifest'ом (h23_sha в записи пересчёту не подвергается — зафиксировано в review).
    """
    ctrl_resume = {nm: f"{nm}__ctrl" in res for nm in names}
    dose_resume = {(nm, n): f"{nm}__n{n}" in res
                   for nm in names for n in n_dose}
    n_resume = sum(ctrl_resume.values()) + sum(dose_resume.values())
    return dict(ctrl_resume=ctrl_resume, dose_resume=dose_resume, n_resume=n_resume)


def cells_equal(c1, c2):
    """Побитовое равенство ячеек (r1-claim vs пересчёт) — все общие ключи."""
    keys = set(c1) | set(c2)
    for k in sorted(keys):
        if k not in c1 or k not in c2:
            return False, f"key {k}: в r1={k in c1}, в v2={k in c2}"
        if c1[k] != c2[k]:
            return False, f"key {k}: {c1[k]} != {c2[k]}"
    return True, ""
