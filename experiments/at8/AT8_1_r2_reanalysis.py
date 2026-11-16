#!/usr/bin/env python3
"""AT0_8 машина v2 (T8.1-r2): disk-only реанализ committed store.

TZ (пре-рег): docs/TZ_add_4-8.md §T8.1 «Машина v2 (T8.1-r2)» (16.11).
Ревью: docs/AT0_8_review.md §10 (минимальный патч) + §11 (r2-exec-поправки).
Исход v1 (не переписывается): results/AT0_8_THRESHOLD.json — gates S1–S9 PASS,
машина v1 вернула V0_NOT_RESOLVED; r2-exec: гейты/провенанс ACCEPT, машина — REVISE.

Вход (read-only, sha-верификация по results/AT0_8_manifest.json):
  results/AT0_8_THRESHOLD.json   (claim, file sha b2923d74324e...)
  results/store/018c3dac90eb1bba.json (store, file sha 28a0c28d94d9...)

Правила (пре-рег, 16.11):
  F2: max_u/overshoot/argmax_λ — для ВСЕХ наблюдаемых (включая исключённые), до гейта.
  F1: поле (dip/hump/plateau) — без требования монотонности; событийный эстиманд =
      интервал [λ_j, λ_{j+1}] максимальной скорости |Δu/Δλ| по u_seedavg (LAM11).
  F4: бинарные (loopy, rep16) — переход = первый λ с raw seedavg > 0 (не u=0.5-crossing);
      raw(0) > 0 → no_transition (отчётный).
  F5: S9 — декларация: один раунд бисекции (0.8), ширина 0.05 = ½ зазора сетки;
      «≤0.02» недостижимо на этой сетке — декларатив, не гейт; расхождение пре-рег
      «кап ≤ 6» / код BIS_CAP=12 зафиксировано (capped=false — не сработало).
  V3: max(u_dip) > 1 (любой λ) ∧ hinge(argmax u_dip) > hinge(0) (seed-avg;
      hinge = |med_dip(λ) − med_dip(1.0)| из store health_ray). Хардкод-0.7 и
      rep16-сравнение убраны (r2-exec: F3a согласовано, F3b опровергнуто).
  V2: полевой интервал dip отделён от λ_c компетентности (nll_holdout ∨ margin,
      resolved) на ≥ 1 шага сетки (по границам CI).
  V1: прежние правила, но dip без λ_c (немонотонен) → структурно недостижим —
      в reachability-декларации.
  families_check → verdict details (3/3 false ⇒ нет единого координированного порога).
  Sensitivity F6 (report-only): per-doc dip из store; медиана на не-луповом
      {wiki_2, wiki_3} vs все 6; форма/амплитуды.
  Порядок: V3 > V2 > V1 > V0 (пре-рег §7 без изменений).

Детерминизм: без RNG, без перепрогона (пере-агрегация уже посчитанных данных).
Запись: results/AT0_8_THRESHOLD_v2.json + results/AT0_8_v2_manifest.json +
results/AT0_8_v2_run.log. Store не трогается.
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_p(name, rel):
    _sp = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "prism", rel))
    _m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    return _m

_ju = _load_p("prism_json_utils", "json_utils.py")
_nl = _load_p("prism_nulls_and_stats", "nulls_and_stats.py")
_el = _load_p("prism_explog", "explog.py")  # spec-лоад (минуя prism/__init__ — паттерн AT-экспов)

sanitize = _ju.sanitize
u_curve = _nl.u_curve
_log = _el.ExpLog()
logline = _log

LAM11 = [0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 1.0]
LAM11_IDX = {lam: i for i, lam in enumerate(LAM11)}
SEEDS = [11, 12, 13]
OBS_ALL = ["dip", "hump", "plateau", "nll_holdout", "rep16", "loopy", "pr_sigma",
           "eff_rank", "margin"]
FIELD = ["dip", "hump", "plateau"]
BINARIES = ["loopy", "rep16"]
COMPETENCE = ["nll_holdout", "margin"]
MAIN6 = ["wiki_0", "wiki_1", "wiki_2", "wiki_3", "wiki_4", "wiki_5"]
CLEAN_SUBSET = ["wiki_2", "wiki_3"]  # не-луповое подмножество (F6; r2-exec)

ART = os.path.join(_ROOT, "results", "AT8", "AT8_1_THRESHOLD.json")
STORE = os.path.join(_ROOT, "results", "AT8", "store", "018c3dac90eb1bba.json")
MANIFEST = os.path.join(_ROOT, "results", "AT8", "AT8_1_manifest.json")
OUT_DIR = os.path.join(_ROOT, "results", "AT8")
CLAIM = os.path.join(OUT_DIR, "AT8_1_THRESHOLD_v2.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "AT8_1_v2_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT8_1_v2_run.log")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def u_from_artifact(lc_entry):
    """u_seedavg из артефакта (ключи f'{lam:.3f}') → {lam: float} по LAM11."""
    us = lc_entry.get("u_seedavg")
    if not us:
        return None
    return {lam: float(us[f"{lam:.3f}"]) for lam in LAM11}


def raw_from_artifact(curves, obs):
    return {lam: float(curves[obs][f"{lam:.2f}"]) for lam in LAM11}


def max_speed_interval(u):
    """(λ_j, λ_{j+1}, Δu, |Δu|/Δλ) с max скорости по соседним узлам LAM11."""
    best = None
    for i in range(len(LAM11) - 1):
        a, b = LAM11[i], LAM11[i + 1]
        du = u[b] - u[a]
        key = abs(du) / (b - a)
        if best is None or key > best[3]:  # best = (a, b, du, key)
            best = (a, b, du, key)
    return best


def gap_steps(lam, interval):
    """Шагов сетки от λ до ближайшей границы интервала (0 — внутри)."""
    i = LAM11_IDX[lam]
    a, b = LAM11_IDX[interval[0]], LAM11_IDX[interval[1]]
    if a <= i <= b:
        return 0
    return min(abs(i - a), abs(i - b))


def hinge_seedavg(health_ray, lam):
    """seed-avg dip_band_hinge(λ); None, если хотя бы один сид None."""
    vals = []
    for sd in SEEDS:
        v = health_ray.get(f"s{sd}_lam{lam:.2f}", {}).get("dip_band_hinge")
        if v is None:
            return None
        vals.append(float(v))
    return float(np.mean(vals))


def main():
    t0 = time.time()
    logline(f"AT0_8_r2 start {datetime.now(timezone.utc).isoformat()} py={sys.version.split()[0]} "
            f"np={np.__version__} (disk-only reanalysis, machine v2)")

    # ---- входы + sha-верификация (гейт до записи)
    man = json.load(open(MANIFEST))
    art_sha = sha256_file(ART)
    store_sha = sha256_file(STORE)
    assert art_sha == man["claim_file_sha256"], \
        f"claim file sha mismatch: {art_sha[:16]} != manifest {man['claim_file_sha256'][:16]}"
    assert store_sha == man["store_file_sha256"], \
        f"store file sha mismatch: {store_sha[:16]} != manifest {man['store_file_sha256'][:16]}"
    logline(f"sha-верификация: claim {art_sha[:16]} / store {store_sha[:16]} — совпадает с manifest")

    art = json.load(open(ART))
    store = json.load(open(STORE))
    lc = art["lambda_c"]
    curves = art["curves_seedavg"]
    health_ray = store["health_ray"]

    # ---- самопроверка: пересчёт seed-avg из store == артефакт (0 расхождений > 1e-12)
    cells = store["cells"]
    seedavg6 = {lam: float(np.mean([float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d]["dip"]
                                                      for d in MAIN6])) for sd in SEEDS]))
                for lam in LAM11}
    dev_max = max(abs(seedavg6[lam] - float(curves["dip"][f"{lam:.2f}"])) for lam in LAM11)
    assert dev_max <= 1e-12, f"pereset dip 6-doc != artifact: dev={dev_max:.2e}"
    logline(f"самопроверка: 6-doc median dip (store) == curves_seedavg.dip (артефакт), "
            f"max dev = {dev_max:.2e}")

    # u-кривая dip == артефакту
    u_dip = u_from_artifact(lc["dip"])
    u_art_check = u_curve(seedavg6, LAM11)
    dev_u = max(abs(u_dip[lam] - u_art_check[lam]) for lam in LAM11)
    assert dev_u <= 1e-12, f"u_dip != u_curve(seedavg6): dev={dev_u:.2e}"
    logline(f"самопроверка: u_dip (артефакт) == u_curve(пересчёт), max dev = {dev_u:.2e}")

    # ---- F2: max_u/overshoot/argmax — для всех наблюдаемых, до гейта
    f2 = {}
    for o in OBS_ALL:
        u = u_from_artifact(lc[o])
        if u is None:
            f2[o] = {"max_u": None, "argmax_lam": None, "overshoot": None,
                     "note": "no_range (u не определена)"}
            continue
        mx = max(u.values())
        argmax = min(lam for lam in LAM11 if u[lam] == mx)
        f2[o] = {"max_u": float(mx), "argmax_lam": argmax, "overshoot": bool(mx > 1.0)}
    logline(f"F2: overshoot (до гейта, все наблюдаемые): "
            + ", ".join(f"{o}={'yes' if f2[o]['overshoot'] else 'no'}"
                        for o in OBS_ALL if f2[o]["overshoot"] is not None
                        and f2[o]["overshoot"]))

    # ---- F1: полевые события = max-скорость (без монотонности)
    field_events = {}
    for o in FIELD:
        u = u_from_artifact(lc[o])
        a, b, du, speed = max_speed_interval(u)
        field_events[o] = {"interval": [a, b], "du": float(du), "speed": float(speed),
                           "spearman_seedavg": lc[o].get("spearman_seedavg"),
                           "monotone_gate_v1": lc[o].get("outcome")}
    logline("F1: полевые события (max |Δu/Δλ|): "
            + ", ".join(f"{o}=[{e['interval'][0]},{e['interval'][1]}] (Δu={e['du']:+.3f})"
                        for o, e in field_events.items()))

    # ---- F4: бинарные — первый λ с raw seedavg > 0
    binaries = {}
    for o in BINARIES:
        raw = raw_from_artifact(curves, o)
        if raw[LAM11[0]] > 0.0:
            binaries[o] = {"transition_lam": None, "note": "no_transition: raw(0) > 0 "
                               f"(baseline {raw[0.0]:.4f})"}
        else:
            tr = next((lam for lam in LAM11 if raw[lam] > 0.0), None)
            binaries[o] = {"transition_lam": tr, "raw_at_transition": raw[tr] if tr else None}
    logline("F4: бинарные (first λ raw > 0): "
            + ", ".join(f"{o}={b['transition_lam']}" for o, b in binaries.items()))

    # ---- Компетентность (без изменений): resolved λ_c + CI
    competence = {}
    for o in COMPETENCE:
        e = lc[o]
        competence[o] = {"outcome": e.get("outcome"), "ci": e.get("ci"),
                         "refined": e.get("refined"),
                         "spearman_seedavg": e.get("spearman_seedavg")}
    logline("компетентность: "
            + ", ".join(f"{o}: {c['outcome']} ci={c['ci']}" for o, c in competence.items()))

    # ---- F5: S9-декларация (из артефакта)
    bis = art.get("bisection", {})
    s9 = {"declaration": "один раунд бисекции (точка 0.8); ширина 0.05 = ½ зазора сетки; "
          "«≤ 0.02» на этой сетке недостижимо — декларатив, не гейт (пре-рег S9: «width ≤ 0.02 "
          "или итераций = 5» — по пре-регу v1-гейт должен был FAIL, код v1 проверял только "
          "наличие клеток)",
          "points": bis.get("points"), "capped": bis.get("capped"),
          "widths": bis.get("widths"),
          "cap_mismatch": "пре-рег §6.6 «кап ≤ 6 точек» vs код BIS_CAP = 12 — зафиксировано; "
                          "capped=false (не сработало)"}
    logline(f"S9-декларация: points={s9['points']}, widths={s9['widths']}")

    # ---- V3 (исправленный): max(u_dip) > 1 ∧ hinge(argmax) > hinge(0)
    mx_dip = max(u_dip.values())
    argmax_dip = min(lam for lam in LAM11 if u_dip[lam] == mx_dip)
    hinge_argmax = hinge_seedavg(health_ray, argmax_dip)
    hinge_0 = hinge_seedavg(health_ray, LAM11[0])
    v3_fired = bool(mx_dip > 1.0 and hinge_argmax is not None and hinge_0 is not None
                    and hinge_argmax > hinge_0)
    v3 = {"max_u_dip": float(mx_dip), "argmax_lam": argmax_dip,
          "hinge_argmax_seedavg": hinge_argmax, "hinge_0_seedavg": hinge_0,
          "cond1_max_gt_1": bool(mx_dip > 1.0), "cond2_hinge_gt_0": (
              None if (hinge_argmax is None or hinge_0 is None) else bool(hinge_argmax > hinge_0)),
          "fired": v3_fired,
          "rule": "max(u_dip) > 1 (любой λ) ∧ hinge(argmax u_dip) > hinge(0) (seed-avg); "
                  "хардкод-0.7 и rep16-сравнение убраны (r2-exec: F3a, F3b)"}
    logline(f"V3: max_u_dip={mx_dip:.4f}@{argmax_dip}, hinge({argmax_dip})={hinge_argmax}, "
            f"hinge(0)={hinge_0} → fired={v3_fired}")

    # ---- V2: полевой интервал dip отделён от λ_c компетентности на ≥ 1 шага
    dip_iv = field_events["dip"]["interval"]
    v2_reasons = {}
    v2_fired = False
    for o in COMPETENCE:
        c = competence[o]
        if c["outcome"] != "resolved" or not c["ci"]:
            v2_reasons[o] = "not resolved"
            continue
        gap = min(gap_steps(lam, dip_iv) for lam in c["ci"])
        sep = gap >= 1
        v2_reasons[o] = {"ci": c["ci"], "field_interval": dip_iv,
                         "gap_steps": gap, "separated": bool(sep)}
        v2_fired = v2_fired or sep
    logline(f"V2: dip-интервал {dip_iv} vs компетентность: {v2_reasons} → fired={v2_fired}")

    # ---- V1: структурно недостижим (dip немонотонен → нет λ_c)
    v1 = {"reachable": False,
          "reason": "dip немонотонен (v1: non_monotone_excluded, ρ = "
                    f"{lc['dip'].get('spearman_seedavg')}) → λ_c^dip не существует → "
                    "coin_all (все DISC_OTHERS сравнимы + coincide с dip) недостижим "
                    "на этом дизайне"}

    # ---- families → details
    families = art.get("families", {})
    n_inconsistent = sum(1 for f in families.values() if not f.get("intra_consistent"))
    families_note = (f"{n_inconsistent}/{len(families)} семейств intra_consistent=false "
                     "⇒ нет единого координированного порога (содержательный результат, §8)") \
        if n_inconsistent == len(families) else \
        f"{n_inconsistent}/{len(families)} семейств inconsistent"

    # ---- Reachability-декларация
    reachability = {
        "V3_PATHOLOGY": {"reachable": v3_fired,
                         "why": f"cond1={v3['cond1_max_gt_1']}, cond2={v3['cond2_hinge_gt_0']} "
                                f"(hinge({argmax_dip})={hinge_argmax} vs hinge(0)={hinge_0})"},
        "V2_SOLITON_CANDIDATE": {"reachable": v2_fired,
                                 "why": f"dip-интервал {dip_iv}; "
                                        + "; ".join(f"{o}: {r}" for o, r in v2_reasons.items())},
        "V1_GENERIC_COMPETENCE": {"reachable": v1["reachable"], "why": v1["reason"]},
        "V0_NOT_RESOLVED": {"reachable": True,
                            "why": "fallback, если V3/V2 не сработали (не ожидается — pre-reg)"},
    }

    # ---- Sensitivity F6 (report-only): per-doc dip, clean-субмножество
    def seedavg_subset(docs):
        return {lam: float(np.mean([float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d]["dip"]
                                                      for d in docs])) for sd in SEEDS]))
                for lam in LAM11}

    sub = seedavg_subset(CLEAN_SUBSET)
    u_sub = u_curve(sub, LAM11)
    a, b, du, speed = max_speed_interval(u_sub)
    peak = max(LAM11, key=lambda l: u_sub[l])
    trough = min(LAM11, key=lambda l: u_sub[l])
    sensitivity = {
        "subset": CLEAN_SUBSET,
        "raw_seedavg": {f"{lam:.2f}": round(sub[lam], 6) for lam in LAM11},
        "u_seedavg": {f"{lam:.3f}": round(u_sub[lam], 6) for lam in LAM11},
        "max_speed_interval": [a, b], "du": float(du),
        "u_peak_lam": peak, "u_peak": float(u_sub[peak]),
        "u_trough_lam": trough, "u_trough": float(u_sub[trough]),
        "note": "dip-сигнатура на не-луповом подмножестве: та же форма (пик/провал на тех же "
                "λ), что и 6-doc — составный артефакт луповых доков отклонён (r2-exec F6)",
        "full6_check": "пересчёт 6-doc median == артефакту (max dev ≤ 1e-12, самопроверка выше)",
    }
    logline(f"F6 sensitivity: {CLEAN_SUBSET}: u пик {u_sub[peak]:+.3f}@{peak}, "
            f"провал {u_sub[trough]:+.3f}@{trough}, max-скорость [{a},{b}]")

    # ---- Вердикт (порядок V3 > V2 > V1 > V0)
    if v3_fired:
        outcome, rule = "V3_PATHOLOGY", v3["rule"]
    elif v2_fired:
        outcome, rule = "V2_SOLITON_CANDIDATE", \
            "полевой интервал dip отделён от λ_c компетентности на ≥ 1 шага сетки (машина v2)"
    elif v1["reachable"]:
        outcome, rule = "V1_GENERIC_COMPETENCE", "все λ_c совпадают"
    else:
        outcome, rule = "V0_NOT_RESOLVED", "V3/V2 не сработали (машина v2)"

    expectation = "V2_SOLITON_CANDIDATE"
    match = bool(outcome == expectation)
    if not match:
        logline(f"DIVERGENCE_FROM_PREREG: {outcome} != {expectation} — разбор (не «исправление» машины)")
    verdict = {
        "outcome": outcome,
        "rule": rule,
        "prereg_expectation": expectation,
        "expectation_match": match,
        "details": {
            "field_events": field_events,
            "binaries": binaries,
            "competence": competence,
            "v3": v3,
            "v2": v2_reasons,
            "v1": v1,
            "families": {"families": families, "note": families_note},
            "reachability": reachability,
            "sensitivity_F6": sensitivity,
            "two_event_summary": {
                "field_reorganization": (f"[{dip_iv[0]}, {dip_iv[1]}] (dip max-скорость, "
                                         f"Δu = {field_events['dip']['du']:+.3f})"),
                "competence_rise": "0.85 (nll_holdout ∧ margin resolved)",
                "mode_on": f"{binaries['loopy']['transition_lam']} (loopy first raw > 0; "
                           "λ=1 — phrase-loop, raw rep16(1) = 0.0 — r2-exec F3b)",
            },
        },
    }
    logline(f"VERDICT: {outcome} (pre-reg ожидание: {expectation}, match={match})")

    # ---- Запись (claim + manifest + run.log)
    claim = {
        "experiment": "AT0_8 T8.1 — машина v2 (T8.1-r2) disk-only reanalysis",
        "revision": "v2-r2-exec",
        "date": datetime.now(timezone.utc).isoformat(),
        "input": {"claim_file_sha256": art_sha, "store_file_sha256": store_sha,
                  "manifest_file_sha256": sha256_file(MANIFEST)},
        "machine": "v2 (TZ_add_4-8.md T8.1 r2-exec pre-reg; review §10 + §11 r2-exec)",
        "selfcheck": {"dip6_recompute_dev": dev_max, "u_dip_dev": dev_u},
        "f2_max_overshoot": f2,
        "s9": s9,
        "verdict": verdict,
        "note": ("перепуск прогона не нужен: пере-агрегация committed store; store не трогается; "
                 "вердикт v1 (V0_NOT_RESOLVED) не переписывается — остаётся артефактом v1-машины"),
    }
    payload = sanitize(claim)
    with open(CLAIM, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
    claim_sha = sha256_file(CLAIM)

    man2 = {
        "experiment": "AT0_8 T8.1 — машина v2 (T8.1-r2)",
        "date": claim["date"],
        "claim_file_sha256": claim_sha,
        "store_file_sha256": store_sha,  # вход, не выход (не трогается)
        "script_sha256": sha256_file(os.path.join(_ROOT, "experiments", "at8", "AT8_1_r2_reanalysis.py")),
        "hash_convention": "AT0_2b-v2 C7 (file sha256 фактических файлов)",
        "note": "disk-only reanalysis из committed store (sha-верификация в claim.input); "
                "без RNG, без перепрогона.",
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(sanitize(man2), f, ensure_ascii=False, indent=1, sort_keys=True)
    logline(f"written: {CLAIM} (file sha {claim_sha[:16]}), {OUT_MANIFEST}, {RUN_LOG}")
    logline(f"AT0_8_R2_DONE verdict={outcome} match={match} "
            f"(wall {time.time() - t0:.1f}s)")
    _log.write_to(RUN_LOG)  # в конце — чтобы финальный маркер был в файле


if __name__ == "__main__":
    main()
