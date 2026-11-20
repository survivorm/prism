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
     только добавляет кандидатов; r3 — нижняя граница кандидатного множества).

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


def run_summary(d):
    cells = d["cells"]
    vs = [c["verdict"] for c in cells]
    found = [c for c in cells if c["verdict"] == "FOUND"]
    sho = [c for c in cells if c["verdict"] == "SHORT_HORIZON_ONLY"]
    docs_found = sorted({c["doc"] for c in found + sho})
    n_found_sho = len(found) + len(sho)
    n_pg = sum(1 for v in vs if v == "POSITION_GENERIC")
    n_red = sum(1 for v in vs if v == "REDUNDANT")
    n_nt = sum(1 for v in vs if v == "NOT_TESTABLE")
    n_nf = sum(1 for v in vs if v == "NOT_FOUND")
    n_ok8 = sum(1 for c in cells
                if c["verify"]["denom_gate_cell"] and c["verify"]["companion_ok"]
                and c["verdict"] != "NOT_TESTABLE")
    n_ps = sum(1 for c in cells if c.get("pos_spec"))
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
    for c in cells:
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
    return {"per_cell": {f"c{c['cell_id']}": f"{c['doc']} L{c['L']} t{c['t']}={c['verdict']}"
                         for c in cells},
            "n_found": len(found), "n_sho": len(sho), "n_found_sho": n_found_sho,
            "docs_found": docs_found, "n_pg": n_pg, "n_red": n_red,
            "n_not_testable": n_nt, "n_not_found": n_nf, "n_ok8": n_ok8,
            "pos_spec_n": n_ps,
            "global": d["verdicts"]["global"], "final": d["verdicts"]["final"],
            "global_under_r6_tree": g_r6,
            "gaps_to_labels": gaps,
            "tprime_fix_monotonicity": mono,
            "ascent_not_converged_n": sum(
                1 for c in cells if not c["ascent"]["ascent_converged"]),
            # r3: точные early_stop_iter t'-подъёма не в артефакте (N3);
            # conv=True И n_traj==0 — единственный наблюдаемый след стопа
            "tprime_early_stop_n": sum(
                1 for c in cells if c.get("stage1d")
                and c["stage1d"]["ascent_tprime"].get("ascent_converged")
                and len(c["stage1d"]["ascent_tprime"]["traj"]) == 0)}


def linear_gate_decomp(d):
    """По каждому NOT_TESTABLE: какой под-гейт упал (единый критерий:
    verify["mode"]) + κ-статистики по всем k (сырые)."""
    out = {}
    for c in d["cells"]:
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
    """16 клеток: подъёмы (t и t') + выигрыши + pencil-сертификат + 1d."""
    rows = []
    for c in d["cells"]:
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
