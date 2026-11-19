#!/usr/bin/env python3
"""AT2e: программная сводка артефакта (TZ r6, ревью r6: «сводку прогона
генерировать функцией, не прозой»).

Читает at2e_s1_r*.json и выдаёт (stdout + JSON):
  1. run_summary  — счётчики вердиктов, docs_found, pos_spec-n, глобальный
     вердикт + контрфактуалы (что не хватает до каждой глобальной метки);
  2. linear_gate_decomp — по каждому NOT_TESTABLE: КАКОЙ под-гейт упал
     (mode/denom_vstar/denom_cell/companion/L1/L2/kappa(f,k)/fallback);
  3. ascent_table — 16 клеток: t-подъём (conv, stop_iter, gnorm,
     gnorm/a_hi^2 vs порог 1e-8/a_hi^2), pencil/lambda(v*) (Rayleigh-Ritz
     сертификат), t'-подъём (iter, gnorm), 1d gap;
  4. c1_c9_kappa  — κ-статистики по всем k + pre-U1 контрфактуал
     (прошли бы без κ-конъюнкции).

Использование:
    python3 experiments/at2/at2e_artifact_summary.py <путь к артефакту.json>
"""
import json
import sys

GAP_MARGIN = 0.3


def _stop_info(a):
    """(gnorm на итерации останова, итерация, converged)."""
    if a["ascent_converged"]:
        it = a["early_stop_iter"]
        g = a["v0_recheck"]["gnorm"] if it == 1 else a["traj"][it - 2]["gnorm"]
        return g, it, True
    last = [r for r in a["traj"] if "gnorm" in r]
    return last[-1]["gnorm"], len(last), False


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
    # контрфактуалы до меток §5
    gaps = {
        "SCRATCHPAD_CANDIDATE": {
            "n_found_sho": max(0, 4 - n_found_sho),
            "n_found": max(0, 2 - len(found)),
            "docs": max(0, 2 - len(docs_found)),
        },
        "POSITION_GENERIC_DOMINANT": {"n_found_sho": max(0, 1 - n_found_sho),
                                      "n_pg": max(0, 8 - n_pg)},
        "REDUNDANT_DOMINANT": {"n_found_sho": max(0, 1 - n_found_sho),
                               "n_red": max(0, 8 - n_red)},
        "NOT_FOUND_IN_SUBSPACE": {"n_found_sho": max(0, 1 - n_found_sho),
                                  "n_ok8": max(0, 8 - n_ok8)},
    }
    return {"per_cell": {f"c{c['cell_id']}": f"{c['doc']} L{c['L']} t{c['t']}={c['verdict']}"
                         for c in cells},
            "n_found": len(found), "n_sho": len(sho), "n_found_sho": n_found_sho,
            "docs_found": docs_found, "n_pg": n_pg, "n_red": n_red,
            "n_not_testable": n_nt, "n_not_found": n_nf, "n_ok8": n_ok8,
            "pos_spec_n": n_ps,
            "global": d["verdicts"]["global"], "final": d["verdicts"]["final"],
            "gaps_to_labels": gaps,
            "ascent_not_converged_n": sum(
                1 for c in cells if not c["ascent"]["ascent_converged"]),
            "tprime_early_stop_n": sum(
                1 for c in cells if c.get("stage1d")
                and c["stage1d"]["ascent_tprime"].get("ascent_converged")
                and len(c["stage1d"]["ascent_tprime"]["traj"]) == 0)}


def linear_gate_decomp(d):
    """По каждому NOT_TESTABLE: какой под-гейт упал + κ-статистики по всем k."""
    out = {}
    for c in d["cells"]:
        if c["verdict"] != "NOT_TESTABLE":
            continue
        v = c["verify"]
        lin = v["linear"]
        kfl = lin.get("kappa_flags") or lin.get("kappa_flags_hi") or {}
        fl_k = {k for k, f in kfl.items() if f}
        sub = {}
        if v["mode"] == "normal":
            sub["L1"] = {"val": lin.get("L1"), "ok": lin.get("L1") is not None
                         and lin["L1"] <= 0.10}
            sub["L2"] = {"val": lin.get("L2"), "ok": lin.get("L2") is not None
                         and 1.9 <= lin["L2"] <= 2.1}
        else:
            sub["L1_num"] = {"val": lin.get("L1_num"), "ok": lin.get("L1_num") is not None
                             and lin["L1_num"] <= 0.10}
            sub["L2_num"] = {"val": lin.get("L2_num"), "ok": lin.get("L2_num") is not None
                             and 1.9 <= lin["L2_num"] <= 2.1}
        sub["kappa"] = {"flagged_k": sorted(int(k) for k in fl_k),
                        "ok": not fl_k}
        sub["denom_gate_vstar"] = v["denom_gate_vstar"]
        sub["denom_gate_cell"] = v["denom_gate_cell"]
        sub["companion_ok"] = v["companion_ok"]
        sub["fallback_grid_triggered"] = v["fallback_grid_triggered"]
        sub["failed"] = sorted(k for k, x in sub.items()
                               if (isinstance(x, dict) and not x["ok"])
                               or (isinstance(x, bool) and not x)
                               and k not in ("fallback_grid_triggered",))
        # κ-статистики по всем k (при a_hi): |2κ-q|/max(q,floor)
        per = v["per_amp"]["a_hi"]
        fq = c["floor"]["used"]
        stats = {str(k): float(abs(2.0 * per["kappa"][str(k)] - per["q"][str(k)])
                               / max(per["q"][str(k)], fq)) for k in range(9)}
        # pre-U1 контрфактуал: linear без κ-конъюнкции
        if "L1" in lin:
            pre_u1 = bool(lin["L1"] <= 0.10 and 1.9 <= lin["L2"] <= 2.1)
        else:
            pre_u1 = bool(lin["L1_num"] <= 0.10 and 1.9 <= lin["L2_num"] <= 2.1)
        out[f"c{c['cell_id']}"] = {
            "cell": f"{c['doc']} L{c['L']} t{c['t']}", "mode": v["mode"],
            "sub_gates": sub, "kappa_stats_per_k": {k: round(s, 4)
                                                    for k, s in stats.items()},
            "pre_U1_counterfactual": pre_u1,
            "linear_ok": lin["ok"]}
    return out


def ascent_table(d):
    """16 клеток: подъёмы + pencil-сертификат + t'-ранний-стоп + 1d."""
    rows = []
    for c in d["cells"]:
        a = c["ascent"]
        g, it, conv = _stop_info(a)
        a2 = c["a_hi"] ** 2
        s1d = c.get("stage1d")
        if s1d is None:
            tp_n, tp_g, tp_conv = None, None, None
            gap = None
        else:
            atp = s1d["ascent_tprime"]
            tp_n = len(atp["traj"])
            trj = [r for r in atp["traj"] if "gnorm" in r]
            tp_conv = atp.get("ascent_converged")
            # v0_recheck/early_stop_iter в ascent_tprime не записаны: ранний стоп =
            # conv=True И n_traj==0 (тогда gnorm < 1e-8, точное значение не в артефакте)
            tp_early = bool(tp_conv and tp_n == 0)
            tp_g = trj[0]["gnorm"] if trj else None
            gap = s1d["gap"]
        rows.append({
            "cell": f"c{c['cell_id']} {c['doc']} L{c['L']} t{c['t']} t'={c['t_prime']}",
            "verdict": c["verdict"],
            "t_ascent": {"converged": conv, "stop_iter": it, "gnorm": g,
                         "a_hi": c["a_hi"], "a_hi2": a2,
                         "gnorm_over_a2": g / a2,
                         "abs_thr_over_a2": 1e-8 / a2},
            "pencil_over_lambda_vstar": float(c["pencil_report"] / a["lambda_vstar"]),
            "tprime_ascent": {"n_traj": tp_n, "converged": tp_conv,
                              "early_stop": tp_early, "gnorm_first": tp_g},
            "gap_1d": gap,
            "near_threshold": s1d.get("near_threshold_flag") if s1d else None,
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
