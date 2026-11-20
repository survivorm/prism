#!/usr/bin/env python3
"""AT2e: программная сводка артефакта (TZ r6, ревью r6: «сводку прогона
генерировать функцией, не прозой»; ревью r7: выигрыши подъёма — из функции,
N2 — единый критерий verify["mode"], явные скобки, метка состояния gnorm).

Читает at2e_s1_r*.json и выдаёт (stdout + JSON):
  1. run_summary  — счётчики вердиктов, docs_found, pos_spec-n, глобальный
     вердикт + контрфактуалы (что не хватает до каждой глобальной метки);
  2. linear_gate_decomp — по каждому NOT_TESTABLE: КАКОЙ под-гейт упал
     (единый критерий: verify["mode"] ∈ {normal, denom_below_floor}),
     κ-статистики по всем k (сырые s_k — БЕЗ шумовой классификации: S_floor,k
     в артефакте r3 отсутствует — measure_floor сохраняет только q(k=0);
     самокалибруемый запас — только в r4, TZ r6 п.4), linear_ok_pre_U1
     (пересчёт ЛИНЕЙНОГО гейта L1+L2 без κ-конъюнкции; не «клетка прошла бы»);
  3. ascent_table — 16 клеток: t/t'-подъёмы (converged, stop, gnorm — на
     состоянии стопа / ПОСЛЕДНИЙ ИЗМЕРЕННЫЙ для не-сошедшихся), gain_t /
     gain_tprime (λ(v*)/λ(v0) — собственный выигрыш подъёма), gain_1st_step,
     needed_tprime = gap/GAP_MARGIN (требуемый множитель: >1 — подъём нужен;
     <1 — уже pass с запасом), pencil/λ(v*) (Rayleigh-Ritz-сертификат),
     1d gap, ratio_tprime_to_tstar = 1/gap = λ(v*_{t′};t′)/λ(v*;t′)
     (точно из артефакта; р7 M1: имя/комментарий раньше были инвертированы),
     stop_iter_tprime (r3: выведено — при conv: n_traj+1, иначе None);
  4. run_summary: global_under_r6_tree (машинная метка r3 по новому дереву —
     р8: иначе N1 в новом месте), gaps_to_labels.blockers, и
     tprime_fix_monotonicity (формальное свойство: t'-фикс монотонен —
     только добавляет кандидатов; r3 — нижняя граница кандидатного множества);
  5. r9: recovery_fallback_check (N1: fallback-правило 100 калибровано под
     первичный a_lo=3e-4; recovery a_lo=1e-4 ⇒ margin/9 — кто бы умер),
     kappa_scope_counterfactual (р9 §4: пересчёт вердиктов по M2-окну из
     того же артефакта), tprime_gain_margin (мин полный gain_tprime / макс
     needed_tprime — 2.997 из r7 как машина-поле);
  6. r10 R2/R3: ВСЕ диагностические циклы (mono, recovery_fallback_check,
     kappa_scope_counterfactual, tprime_gain_margin, ascent_not_converged_n,
     tprime_early_stop_n, per_cell, linear_gate_decomp, ascent_table) —
     primary-only (cells_p): r3 no-op, r4 — recovery-записи не попадают в
     поля с primary-семантикой; n_recovery_cells + global_recovery —
     машинная видимость фильтра. unresolved_k → flagged_k (R3: «unresolved»
     — классификация, которой нет без S_floor; статус целей — верхняя
     граница, не утверждение).

Использование:
    python3 experiments/at2/at2e_artifact_summary.py <путь к артефакту.json>
"""
import json
import sys

GAP_MARGIN = 0.3          # TZ §3.11: 1d pass при gap = λ(v*;t')/λ(v*_t';t') <= 0.3
A_HI_REL = 1e-3           # AT2e.py L97: a_hi = 1e-3*||h_t|| (для t': a_p = 1e-3*||h_t'||)
L1_MAX = 0.10             # TZ §3.5: L1 <= 0.10
L2_MIN, L2_MAX = 1.9, 2.1  # TZ §3.5: 1.9 <= L2 <= 2.1
KAPPA_ABS = 0.5           # TZ §3.5: flag при |2κ-q|/max(q,floor) > 0.5
FLOOR_FACTOR = 10         # TZ §3.9: denom-гейты 10*floor
KAPPA_WINDOW = {1, 2, 4, 8}   # M2 (р8): K1∪K2 — только эти k блокируют
RECOVERY_ALO2 = (1e-4 / 3e-4) ** 2  # r9 N1: (a_lo_recovery/a_lo_primary)² = 1/9
FALLBACK_TOL = 100.0      # TZ r5.2 F3: median q_t(a_lo)/floor < 100 → NOT_TESTABLE
TAU_STRS = ("0.001", "0.01", "0.05")  # str(t) для t in TAUS=(1e-3, 1e-2, 5e-2)


def _lam_k1(qd, a, eps0, tau=1e-2):
    """λ^{K1}(v) = 2*qbar_K1/(q0 + a^2*tau*eps0) — тот же объект, что в ascent()
    (K1 = (1,2) — AT2e.py L106; ключи qd — str(k))."""
    K1 = (1, 2)  # AT2e.py: K1 = (1, 2), K2 = (4, 8)
    qd = {int(k): v for k, v in qd.items()}
    return float(2.0 * sum(qd[k] for k in K1) / len(K1)
                 / (qd[0] + a * a * tau * eps0))


def _stop_info(a):
    """(gnorm, n_iters, converged, gnorm_at). gnorm_at — состояние, в котором
    измерен ‖g‖: ранний стоп — состояние стопа (v0 при стопе на i=1);
    8 итераций — ПОСЛЕДНИЙ ИЗМЕРЕННЫЙ ‖g(v_{n-1})‖ (на v* градиента нет —
    финальный проход FD-only), т.е. power-лимит, не «на v*»."""
    if a["ascent_converged"]:
        it = a["early_stop_iter"]
        g = a["v0_recheck"]["gnorm"] if it == 1 else a["traj"][it - 2]["gnorm"]
        return g, it, True, "v_stop"
    last = [r for r in a["traj"] if "gnorm" in r]
    return last[-1]["gnorm"], len(a["traj"]), False, "v_last_measured"


def _verdict_m2(c):
    """Зеркало cell_verdict (AT2e.py L1025-1057) с M2 k-окном (р8/р9): unresolved
    κ блокирует только при k ∈ {1,2,4,8} = K1∪K2; вне окон — report-only.
    Всё остальное идентично (тот же приоритет, те же гейты). Используется
    ТОЛЬКО как контрфактуал (р9 §4) — для проверки: на r3 данных должен
    воспроизвести вердикты всех клеток, кроме c12 (NOT_TESTABLE → PG)."""
    v = c["verify"]
    lin = v["linear"]
    kfl = lin.get("kappa_flags_hi") or lin.get("kappa_flags") or {}
    unres = {int(k) for k, f in kfl.items() if f}
    kappa_ok = not (unres & KAPPA_WINDOW)
    if v["mode"] == "normal":
        l12 = (lin["L1"] is not None and lin["L1"] <= L1_MAX
               and L2_MIN <= lin["L2"] <= L2_MAX)
    else:
        l12 = (lin["L1_num"] is not None and lin["L1_num"] <= L1_MAX
               and L2_MIN <= lin["L2_num"] <= L2_MAX)
    lin_ok = bool(l12 and kappa_ok)
    s1d = c.get("stage1d")
    if (not lin_ok or not v["denom_gate_cell"] or not v["companion_ok"]
            or v.get("fallback_grid_triggered")
            or (s1d and s1d["test_guard"])):
        return "NOT_TESTABLE"
    rank1_k1 = sum(1 for tau in TAU_STRS if c["rank"]["K1"][tau] == 1)
    rank1_k2 = sum(1 for tau in TAU_STRS if c["rank"]["K2"][tau] == 1)
    rho = v["rho"]
    signature = bool((rho is not None and c["rho_ref_p90"] is not None
                      and rho > c["rho_ref_p90"])
                     or (not v["denom_gate_vstar"]))
    is_1d = bool(c.get("pos_spec") is not None and rank1_k1 >= 2 and signature)
    if is_1d:
        if not c["pos_spec"]["screen_ok"]:
            return "POSITION_GENERIC"  # early-exit, 1d не гоняется
        if s1d["gate"]["pass"]:
            return "FOUND" if rank1_k2 >= 2 else "SHORT_HORIZON_ONLY"
        return "POSITION_GENERIC"
    if (rank1_k1 >= 2 and v["denom_gate_vstar"]
            and c["rho_rank"] is not None
            and c["rho_ref_p10"] <= c["rho_rank"] <= c["rho_ref_p90"]):
        return "REDUNDANT"
    return "NOT_FOUND"


def run_summary(d):
    cells = d["cells"]
    # r9 N3: обе tree-оценки — только по ПЕРВИЧНОЙ сетке; recovery-клетки
    # (amplitude_grid: recovery, r4) учитываются своим первичным исходом.
    # r3: no-op (поля нет) — r4: исключение recovery-записей из счётчиков.
    cells_p = [c for c in cells if c.get("amplitude_grid", "primary") == "primary"]
    n_recovery = len(cells) - len(cells_p)  # r10 R2: фильтр видим машинно
    vs = [c["verdict"] for c in cells_p]
    found = [c for c in cells_p if c["verdict"] == "FOUND"]
    sho = [c for c in cells_p if c["verdict"] == "SHORT_HORIZON_ONLY"]
    docs_found = sorted({c["doc"] for c in found + sho})
    n_found_sho = len(found) + len(sho)
    n_pg = sum(1 for v in vs if v == "POSITION_GENERIC")
    n_red = sum(1 for v in vs if v == "REDUNDANT")
    n_nt = sum(1 for v in vs if v == "NOT_TESTABLE")
    n_nf = sum(1 for v in vs if v == "NOT_FOUND")
    n_ok8 = sum(1 for c in cells_p
                if c["verify"]["denom_gate_cell"] and c["verify"]["companion_ok"]
                and c["verdict"] != "NOT_TESTABLE")
    n_ps = sum(1 for c in cells_p if c.get("pos_spec"))
    # контрфактуалы до меток §5 (+ blockers — р8: где стоит 0, а есть блокер)
    def _blockers():
        return [f"n_found_sho={n_found_sho}≠0"] if n_found_sho > 0 else []
    gaps = {
        "SCRATCHPAD_CANDIDATE": {
            "n_found_sho": max(0, 4 - n_found_sho),
            "n_found": max(0, 2 - len(found)),
            "docs": max(0, 2 - len(docs_found)),
        },
        "POSITION_GENERIC_DOMINANT": {"n_found_sho": max(0, 1 - n_found_sho),
                                      "n_pg": max(0, 8 - n_pg),
                                      "blockers": _blockers()},
        "REDUNDANT_DOMINANT": {"n_found_sho": max(0, 1 - n_found_sho),
                               "n_red": max(0, 8 - n_red),
                               "blockers": _blockers()},
        "NOT_FOUND_IN_SUBSPACE": {"n_found_sho": max(0, 1 - n_found_sho),
                                  "n_ok8": max(0, 8 - n_ok8),
                                  "blockers": _blockers()},
    }
    # р6-дерево (р7 п.3): + SUB_THRESHOLD_FOUND (n_found_sho>=1, n_found>=1,
    # docs>=1) перед старыми ветками; r3 по нему — вычисляется машинно,
    # не из прозы (р8: иначе N1 в новом месте)
    if n_found_sho >= 4 and len(found) >= 2 and len(docs_found) >= 2:
        g_r6 = "SCRATCHPAD_CANDIDATE"
    elif n_found_sho >= 1 and len(found) >= 1 and len(docs_found) >= 1:
        g_r6 = "SUB_THRESHOLD_FOUND"
    elif n_found_sho == 0 and n_pg >= 8:
        g_r6 = "POSITION_GENERIC_DOMINANT"
    elif n_found_sho == 0 and n_red >= 8:
        g_r6 = "REDUNDANT_DOMINANT"
    elif n_found_sho == 0 and n_ok8 >= 8 and n_pg < 8 and n_red < 8:
        g_r6 = "NOT_FOUND_IN_SUBSPACE"
    else:
        g_r6 = "NOT_RESOLVED"
    # т'-фикс монотонен (р8 §3): больше поиска в t' ⇒ λ(v*_t')↑ ⇒ gap↓ ⇒
    # pass легче. Кандидаты, у которых t'-подъём полный (траектория
    # не изменится после фикса) ИЛИ уже pass с gain=1.0 — робастны;
    # PG-клетки с gain=1.0 — их gap = ВЕРХНЯЯ граница (могут pass).
    # t-фикс НЕ монотонен по v*, но на r3 данных трогает только
    # early-stop клетки (все blocked или rank1_K1=0) — тоже только добавляет.
    mono = {"robust_found_sho": [], "pg_upper_bound_gap": [],
            "t_ascent_early_stop_cells": []}
    for c in cells_p:  # r10 R2: primary-only
        s1d = c.get("stage1d")
        a = c["ascent"]
        if a["ascent_converged"] and len(a["traj"]) == 0:
            mono["t_ascent_early_stop_cells"].append(f"c{c['cell_id']}")
        if s1d is None:
            continue
        atp = s1d["ascent_tprime"]
        tp_early = bool(atp["ascent_converged"] and len(atp["traj"]) == 0)
        gap = s1d["gap"]
        if c["verdict"] in ("FOUND", "SHORT_HORIZON_ONLY") and \
                (not tp_early or (gap is not None and gap < GAP_MARGIN)):
            mono["robust_found_sho"].append(f"c{c['cell_id']}")
        elif c["verdict"] == "POSITION_GENERIC" and tp_early:
            mono["pg_upper_bound_gap"].append(f"c{c['cell_id']}")
    mono["property"] = ("t'-фикс монотонен (больше поиска ⇒ gap падает ⇒ "
                        "pass легче): текущие FOUND/SHO робастны, PG с "
                        "gain=1.0 имеют gap-верхнюю границу — r3 = нижняя "
                        "граница кандидатного множества по t'-оси")
    # r9 N1: recovery-цели = клетки с linear-fail (κ или L1/L2) на mode
    # normal; below_floor/denom-провалы (c3/c11) в recovery не входят (р7).
    # M2: внеоконный κ больше не блокирует — c12 перестаёт быть целью.
    rfc = {}
    for c in cells_p:  # r10 R2: recovery-цели — первичные записи
        v = c["verify"]
        lin = v["linear"]
        if lin["ok"] or v["mode"] == "denom_below_floor":
            continue
        kfl = lin.get("kappa_flags_hi") or lin.get("kappa_flags") or {}
        flagged = {int(k) for k, f in kfl.items() if f}
        in_win = bool(flagged & KAPPA_WINDOW)
        l12 = (lin["L1"] is not None and lin["L1"] <= L1_MAX
               and L2_MIN <= lin["L2"] <= L2_MAX)
        m = c["b0_qt_alo_median"] / c["floor"]["used"]
        rfc[f"c{c['cell_id']}"] = {
            "fail_reason": "L1L2" if not l12 else "kappa",
            # r10 R3: flagged_k — НЕ «unresolved»: без S_floor (r3-артефакт)
            # флаг ≠ «нерешиемый»; в r4 S_floor решит часть флагов, но клетка
            # останется recovery-целью по genuine-нелинейности — множество
            # целей не меняется. Статус — верхняя граница, не утверждение.
            "flagged_k": sorted(flagged),
            "target_under_old_rule": True,
            "target_under_new_rule_M2": bool(in_win or not l12),
            "target_under_new_rule_M2_is_upper_bound": True,
            "primary_margin_median_qt_alo_over_floor": m,
            "recovery_margin_at_a_lo_1e-4": m * RECOVERY_ALO2,
            "would_trigger_fallback_100_at_recovery":
                bool(m * RECOVERY_ALO2 < FALLBACK_TOL),
        }
    # р9 §4: контрфактуал M2 — пересчёт вердиктов из того же артефакта;
    # сверка: _verdict_m2 должен воспроизвести вердикт каждой клетки, кроме
    # тех, что реально меняются (r3: только c12)
    ks_changes = {}
    for c in cells_p:  # r10 R2: primary-only
        v_m2 = _verdict_m2(c)
        if v_m2 != c["verdict"]:
            ks_changes[f"c{c['cell_id']}"] = {"old": c["verdict"], "new": v_m2}
    # счётчики под M2-правилом (р9 §4: n_pg 4→5, n_nt 7→6, headline не меняет)
    ks_m2 = [_verdict_m2(c) for c in cells_p]
    ks_counts = {
        "n_pg": sum(1 for v in ks_m2 if v == "POSITION_GENERIC"),
        "n_nt": sum(1 for v in ks_m2 if v == "NOT_TESTABLE"),
        "n_found": sum(1 for v in ks_m2 if v == "FOUND"),
        "n_sho": sum(1 for v in ks_m2 if v == "SHORT_HORIZON_ONLY"),
    }
    # р9 п.5: tprime_gain_margin = min(полный gain_tprime) / max(needed)
    full_gains, needed = [], []  # (doc, cell_id, value) — r10 нитка 3: docs
    for c in cells_p:  # r10 R2: primary-only
        s1d = c.get("stage1d")
        if s1d is None:
            continue
        atp = s1d["ascent_tprime"]
        a_p = A_HI_REL * c["hnorm_prime"]
        lam_v0_tp = max(_lam_k1(b, a_p, c["eps0"])
                        for b in s1d["probe_b0_tprime"])
        needed.append((c["doc"], c["cell_id"], s1d["gap"] / GAP_MARGIN))
        if not (atp["ascent_converged"] and len(atp["traj"]) == 0):
            full_gains.append((c["doc"], c["cell_id"],
                               atp["lambda_vstar_tprime"] / lam_v0_tp))
    min_full = min(v for _, _, v in full_gains)
    max_needed = max(v for _, _, v in needed)
    tpm = min_full / max_needed
    return {"per_cell": {f"c{c['cell_id']}": f"{c['doc']} L{c['L']} t{c['t']}={c['verdict']}"
                         for c in cells_p},  # r10 R2: primary-only (r4: ключи-дублики)
            "n_recovery_cells": n_recovery,
            "global": d["verdicts"]["global"], "final": d["verdicts"]["final"],
            # r9 N3: recovery-исход — отдельный exploratory-счёт (r4)
            "global_recovery": d["verdicts"].get("global_recovery"),
            "global_under_r6_tree": g_r6,
            "n_found": len(found), "n_sho": len(sho), "n_found_sho": n_found_sho,
            "docs_found": docs_found, "n_pg": n_pg, "n_red": n_red,
            "n_not_testable": n_nt, "n_not_found": n_nf, "n_ok8": n_ok8,
            "pos_spec_n": n_ps,
            "gaps_to_labels": gaps,
            "tprime_fix_monotonicity": mono,
            "ascent_not_converged_n": sum(
                1 for c in cells_p if not c["ascent"]["ascent_converged"]),  # r10 R2
            # r3: точные early_stop_iter t'-подъёма не в артефакте (N3);
            # conv=True И n_traj==0 — единственный наблюдаемый след стопа
            "tprime_early_stop_n": sum(
                1 for c in cells_p if c.get("stage1d")  # r10 R2
                and c["stage1d"]["ascent_tprime"].get("ascent_converged")
                and len(c["stage1d"]["ascent_tprime"]["traj"]) == 0),
            # ---- r9 N1: recovery-сетка × fallback-правило ----
            # Fallback (r5 F3: median q_t(a_lo)/floor < 100) калиброван под
            # первичный a_lo = 3e-4·||h||; recovery-сетка {1e-4, 3e-4} имеет
            # a_lo втрое меньше ⇒ q ∝ a² ⇒ margin /9. Правило на recovery-сетке
            # убивает клетки с первичным margin < 900 (р3: c6 252→28 и
            # c13 399→44 — умирают; c12 926→103 — 2.8% запаса). TZ r6 фикс:
            # правило — только первичная сетка; на recovery — только
            # 10·floor denom-гейт. Поле — машинная видимость (р9 п.4).
            "recovery_fallback_check": rfc,
            # ---- р9 §4: контрфактуал k-скоупа как ФАКТ ----
            "kappa_scope_counterfactual": {
                "rule": "M2: k ∈ {1,2,4,8} unresolved -> NOT_TESTABLE; "
                        "вне окон -> report-only",
                "changes": ks_changes,
                "unchanged_n": len(cells) - len(ks_changes),
                "counts_under_M2": ks_counts,
            },
            # ---- р9 п.5: tprime_gain_margin (2.997 из r7 §2.2 как поле) ----
            "tprime_gain_margin": tpm,
            "tprime_gain_margin_components": {
                "min_full_gain_tprime": min_full,
                # r10 нитка 3: экстраполяция wiki видима машинно, не только
                # в прозе (оговорка M4 из TZ)
                "min_full_gain_tprime_cells": [
                    f"c{ci}" for d_, ci, v in full_gains if v == min_full],
                "min_full_gain_tprime_docs": sorted(
                    {d_ for d_, _, v in full_gains if v == min_full}),
                "max_needed_tprime": max_needed,
                "max_needed_tprime_cells": [
                    f"c{ci}" for d_, ci, v in needed if v == max_needed],
                "max_needed_tprime_docs": sorted(
                    {d_ for d_, _, v in needed if v == max_needed}),
            }}


def linear_gate_decomp(d):
    """По каждому NOT_TESTABLE (primary, r10 R2): какой под-гейт упал
    (единый критерий: verify["mode"]) + κ-статистики по всем k (сырые)."""
    out = {}
    for c in d["cells"]:
        if c.get("amplitude_grid", "primary") != "primary":  # r10 R2
            continue
        if c["verdict"] != "NOT_TESTABLE":
            continue
        v = c["verify"]
        lin = v["linear"]
        mode = v["mode"]  # "normal" | "denom_below_floor" — ЕДИНСТВЕННЫЙ критерий
        kfl = lin.get("kappa_flags_hi") or lin.get("kappa_flags") or {}
        fl_k = sorted(int(k) for k, f in kfl.items() if f)
        sub = {}
        if mode == "normal":
            l1, l2 = lin.get("L1"), lin.get("L2")
            sub["L1"] = {"val": l1, "ok": l1 is not None and l1 <= L1_MAX}
            sub["L2"] = {"val": l2,
                         "ok": l2 is not None and L2_MIN <= l2 <= L2_MAX}
        elif mode == "denom_below_floor":
            l1, l2 = lin.get("L1_num"), lin.get("L2_num")
            sub["L1_num"] = {"val": l1, "ok": l1 is not None and l1 <= L1_MAX}
            sub["L2_num"] = {"val": l2,
                             "ok": l2 is not None and L2_MIN <= l2 <= L2_MAX}
        else:
            raise ValueError(f"неизвестный verify.mode: {mode!r} (c{c['cell_id']})")
        sub["kappa"] = {"flagged_k": fl_k, "ok": not fl_k}
        sub["denom_gate_vstar"] = v["denom_gate_vstar"]
        sub["denom_gate_cell"] = v["denom_gate_cell"]
        sub["companion_ok"] = v["companion_ok"]
        sub["fallback_grid_triggered"] = v["fallback_grid_triggered"]
        # N2(a): явные скобки; сработавший fallback ВИДИМ в failed (он и есть
        # причина NOT_TESTABLE для c3/c11 по TZ r5.2: fallback-правило -> NOT_TESTABLE)
        failed = []
        for k, x in sub.items():
            if k == "fallback_grid_triggered":
                inc = bool(x)
            else:
                inc = (isinstance(x, dict) and not x["ok"]) \
                      or (isinstance(x, bool) and not x)
            if inc:
                failed.append(k)
        sub["failed"] = sorted(failed)
        # κ-статистики по всем k (при a_hi): s_k = |2κ_k - q_k|/max(q_k, floor) —
        # СЫРЫЕ значения. Шумовая классификация (шумовой/решительный) НЕ делается:
        # S_floor,k отсутствует в артефакте r3 (measure_floor хранит только q(k=0));
        # самокалибруемый шумовой запас — r4 (TZ r6 п.4).
        per = v["per_amp"]["a_hi"]
        fq = c["floor"]["used"]
        stats = {str(k): round(float(abs(2.0 * per["kappa"][str(k)]
                                        - per["q"][str(k)]) / max(per["q"][str(k)], fq)), 4)
                 for k in range(9)}
        # linear_ok_pre_U1: пересчёт ЛИНЕЙНОГО гейта (L1+L2 без κ) — тем же
        # критерием (mode); клетка всё равно может пасть по denom/fallback
        if mode == "normal":
            pre_u1 = bool(l1 <= L1_MAX and L2_MIN <= l2 <= L2_MAX)
        else:
            pre_u1 = bool(l1 <= L1_MAX and L2_MIN <= l2 <= L2_MAX)
        out[f"c{c['cell_id']}"] = {
            "cell": f"{c['doc']} L{c['L']} t{c['t']}", "mode": mode,
            "sub_gates": sub, "kappa_stats_per_k": stats,
            # р8: pre-U1 = пересчёт ЛИНЕЙНОГО гейта (L1+L2 без κ) — клетка всё
            # равно может пасть по denom/fallback; имя без «прошла бы»
            "linear_ok_pre_U1": pre_u1,
            "linear_ok": lin["ok"]}
    return out


def ascent_table(d):
    """16 клеток (primary, r10 R2): подъёмы (t и t') + выигрыши + pencil-
    сертификат + 1d."""
    rows = []
    for c in d["cells"]:
        if c.get("amplitude_grid", "primary") != "primary":  # r10 R2
            continue
        a = c["ascent"]
        g, it, conv, g_at = _stop_info(a)
        a2 = c["a_hi"] ** 2
        # t-подъём: v0 = argmax B0 при t (v0_lambda_K1 из артефакта)
        gain_t = a["lambda_vstar"] / a["v0_lambda_K1"]
        lam_1st_t = a["traj"][0]["lambda_K1"] if a["traj"] else None
        s1d = c.get("stage1d")
        if s1d is None:
            row_tp = {"present": False, "gap_1d": None, "near_threshold": None}
        else:
            atp = s1d["ascent_tprime"]
            n = len(atp["traj"])
            tp_conv = bool(atp["ascent_converged"])
            tp_early = bool(tp_conv and n == 0)  # r3: единственный наблюдаемый след
            trj = [r for r in atp["traj"] if "gnorm" in r]
            tp_g = trj[-1]["gnorm"] if trj else None
            # собственный выигрыш t'-подъёма: v0 = argmax B0 при t' (из
            # probe_b0_tprime — в артефакте r3); амплитуда — a_p = 1e-3*||h_t'||
            # (AT2e.py A_HI_REL=1e-3, hnorm_prime), НЕ a_hi при t (проверено:
            # max_j λ(B0_j;t') == λ(v*_t') по 16 цифр для v*=v0-клеток);
            # r4: v0_lambda_K1 будет сохранён
            a_p = A_HI_REL * c["hnorm_prime"]
            lam_v0_tp = max(_lam_k1(b, a_p, c["eps0"])
                            for b in s1d["probe_b0_tprime"])
            lam_vs_tp = atp["lambda_vstar_tprime"]
            gap = s1d["gap"]
            row_tp = {
                "present": True,
                "converged": tp_conv, "early_stop": tp_early, "n_traj": n,
                # r3: gnorm t'-стопа НЕ в артефакте (N3) — только в логе;
                # для 8-итераций — последний измеренный ‖g(v_{n-1})‖
                "gnorm_last_measured": tp_g,
                "gnorm_at": "v_stop(log-only)" if tp_early else (
                    "v_last_measured" if trj else None),
                "lambda_v0_tprime": lam_v0_tp,
                "lambda_vstar_tprime": lam_vs_tp,
                "gain_tprime": lam_vs_tp / lam_v0_tp,
                "gain_1st_step_tprime": (atp["traj"][0]["lambda_K1"] / lam_v0_tp
                                         if atp["traj"] else None),
                # r8 M1: 1/gap = λ(v*_{t′};t′)/λ(v*;t′) — tprime over tstar
                # (ранее имя/комментарий были инвертированы)
                "ratio_tprime_to_tstar": (1.0 / gap if gap else None),
                # требуемый множитель: во сколько раз λ(v*_t';t') должен
                # превзойти λ(v*;t') для 1d-pass; >1 — нужен рост, <1 — pass
                # с запасом (c15: 0.513 = уже pass, c0: 0.064 = pass с запасом)
                "needed_tprime": (gap / GAP_MARGIN if gap else None),
                # r8: выведено бесплатно (N3-часть): conv=True ⇒ записей
                # i−1 ⇒ stop_iter = n_traj+1 (0→1, 7→8); не-conv → None
                "stop_iter_tprime": (n + 1 if tp_conv else None),
                "gap_1d": gap,
                "near_threshold": s1d.get("near_threshold_flag"),
            }
        rows.append({
            "cell": f"c{c['cell_id']} {c['doc']} L{c['L']} t{c['t']} t'={c['t_prime']}",
            "verdict": c["verdict"],
            "t_ascent": {"converged": conv, "stop_iter": it, "gnorm": g,
                         "gnorm_at": g_at,
                         "a_hi": c["a_hi"], "a_hi2": a2,
                         "gnorm_over_a2": g / a2,
                         "abs_thr_over_a2": 1e-8 / a2},
            "lambda_v0_t": a["v0_lambda_K1"],
            "lambda_vstar_t": a["lambda_vstar"],
            "gain_t": gain_t,
            "gain_1st_step_t": (lam_1st_t / a["v0_lambda_K1"]
                                if lam_1st_t is not None else None),
            "pencil_over_lambda_vstar": float(c["pencil_report"] / a["lambda_vstar"]),
            "tprime_ascent": row_tp,
        })
    return rows


def main(path):
    d = json.load(open(path))
    res = {"artifact": path,
           "run_summary": run_summary(d),
           "linear_gate_decomp": linear_gate_decomp(d),
           "ascent_table": ascent_table(d)}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    base = path.rsplit(".", 1)[0] + "_summary"
    with open(base + ".json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"-> {base}.json", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
