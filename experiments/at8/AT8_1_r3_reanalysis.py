#!/usr/bin/env python3
"""AT8_1 машина v3 (T8.1-r3): disk-only реанализ committed store.

TZ (пре-рег): docs/TZ_add_4-8.md §T8.1 «Машина v3 (T8.1-r3)» (16.11, раунд 2 ревью).
Ревью (основание): docs/AT8_1_review.md (sha 56edfeba2028e03732a8544f163306ef29f9ce0b6ce2412b1226e7d01da92e9e) —
минимальный патч §5 пункты 1–5 + §6 «что прислать»; вердикт ревью: «дисциплина v2 — ACCEPT;
вердикт V2_SOLITON_CANDIDATE — REVISE».

Вход (read-only, sha-верификация по results/AT8/AT8_1_manifest.json):
  results/AT8/AT8_1_THRESHOLD.json   (claim v1, file sha b2923d74324e...)
  results/AT8/store/018c3dac90eb1bba.json (store, file sha 28a0c28d94d9...)

Артефакты v1/v2 НЕ переписываются: AT8_1_THRESHOLD.json (машина v1 = V0_NOT_RESOLVED),
AT8_1_THRESHOLD_v2.json (машина v2 = V2_SOLITON_CANDIDATE) — остаются артефактами своих машин.

Правила (пре-рег, 16.11):
  1. V3 (знаковый детектор, O2-алгебра): cond1 = min_λ dip_seedavg(λ) < dip_seedavg(1.0);
     корроборация вырожденности в той же λ-клетке λ* = argmin — минимум одно:
     C1: loopy_seedavg(λ*) > 0; C2: eff_rank(λ*) < min(eff_rank(0), eff_rank(1));
     C3: nll_holdout(λ*) > max(nll_holdout(0), nll_holdout(1)).
     V3 fired = cond1 ∧ (C1 ∨ C2 ∨ C3). max_u_dip/argmax — описательные (наследие v2).
     Hinge-условие v2 убрано (hinge(λ)/hinge(0) = |u(λ)−1| — эталонную подпись E-71
     (u≈1.19 → 0.19 ≪ 1) не ловило).
  2. V2 (таблица трёх локаторов, F1): L1 — первое 2-точечное пересечение u=0.5 (по возрастанию λ);
     L2 — onset укручения: первая клетка со скоростью > 2× медианы скоростей всех 10 клеток;
     L3 — max-скорость (v2-правило; tie — меньший λ, множество тиевых клеток отчётно).
     Таблица 3 × {dip, nll_holdout, margin}.
     Отделение: gap = мин. число шагов сетки между граничными узлами двух locator-интервалов;
     общая граница/клетка ⇒ gap = 0. V2 fired только если отделение держится на ВСЕХ трёх
     локаторах; расхождение между локаторами / неопределённый локатор ⇒ V2 not_resolved.
  3. Per-seed (F2): per-seed L3-клетки dip (3 сида); span = (max − min) индексов клеток
     (левые узлы, в шагах сетки); span > 1 ⇒ полевой интервал unresolved (отчётное;
     на данных: клетки {5, 6, 5} → span 1 ⇒ не срабатывает). Per-seed L1 — отчётно.
  4. families v3 (F3): member информативен, если u определена И (для бинарных — есть
     переход; no_transition — пороговой информации нет). mode: rep16 no_transition
     (raw(0) = 0.8824 > 0 — F4-правило, baseline-луп) ⇒ исключён; остаётся loopy (0.85)
     ⇒ одно информативное ⇒ vacuous по D7′. field/leaf — по v1 intra_consistent.
  5. S9 (F4): декларация v2 переносится без изменений.
  6. Raw-отчёт (F5 + §6): dip(0), dip(1), Δ = dip(1)−dip(0), min/max dip (с λ; tie — меньший λ,
     множество аргминимума/аргмаксимума отчётно) в сырых единицах; полные per-seed dip-кривые
     (3×11) + per-seed cond1/λ*/C2 (медиана по 6 докам)/C3 (медиана holdout nll); per-seed C1
     нечислится (в store нет per-seed loopy — только seed-avg в claim v1) — в note.
  7. Порядок вердикта: V3 > V2 > V1 > V0. V1 структурно недостижим (dip немонотонен,
     v1-артефакт) — reachability-декларация. F2-описательный и F6-sensitivity — наследие v2.

Детерминизм: без RNG, без перепрогона (пере-агрегация committed store).
Запись: results/AT8/AT8_1_THRESHOLD_v3.json + results/AT8/AT8_1_v3_manifest.json +
results/AT8/AT8_1_v3_run.log. Store не трогается.
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
LOCATOR_OBS = ["dip", "nll_holdout", "margin"]
MAIN6 = ["wiki_0", "wiki_1", "wiki_2", "wiki_3", "wiki_4", "wiki_5"]
CLEAN_SUBSET = ["wiki_2", "wiki_3"]  # не-луповое подмножество (F6; наследие v2)

ART = os.path.join(_ROOT, "results", "AT8", "AT8_1_THRESHOLD.json")
STORE = os.path.join(_ROOT, "results", "AT8", "store", "018c3dac90eb1bba.json")
MANIFEST = os.path.join(_ROOT, "results", "AT8", "AT8_1_manifest.json")
OUT_DIR = os.path.join(_ROOT, "results", "AT8")
CLAIM = os.path.join(OUT_DIR, "AT8_1_THRESHOLD_v3.json")
OUT_MANIFEST = os.path.join(OUT_DIR, "AT8_1_v3_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT8_1_v3_run.log")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def u_from_artifact(lc_entry):
    """u_seedavg из артефакта (ключи f'{lam:.3f}') → {lam: float} по LAM11; None — no_range."""
    us = lc_entry.get("u_seedavg")
    if not us:
        return None
    return {lam: float(us[f"{lam:.3f}"]) for lam in LAM11}


def raw_from_artifact(curves, obs):
    return {lam: float(curves[obs][f"{lam:.2f}"]) for lam in LAM11}


def interval_speeds(u):
    """[(a, b, |Δu|/Δλ)] по 10 интервалам LAM11; None, если u None."""
    if u is None:
        return None
    out = []
    for i in range(len(LAM11) - 1):
        a, b = LAM11[i], LAM11[i + 1]
        out.append((a, b, abs(u[b] - u[a]) / (b - a)))
    return out


def l1_u05_crossing(u):
    """L1: первое 2-точечное пересечение u=0.5 (первое по λ); None — нет пересечения."""
    if u is None:
        return None
    for i in range(len(LAM11) - 1):
        a, b = LAM11[i], LAM11[i + 1]
        ua, ub = u[a], u[b]
        if ua != ub and (ua - 0.5) * (ub - 0.5) <= 0.0:
            return [a, b]
    return None


def l2_onset_steeper(u):
    """L2: первая клетка со скоростью > 2× медианы скоростей всех 10 клеток; None — нет."""
    sp = interval_speeds(u)
    if sp is None:
        return None
    med = float(np.median([s[2] for s in sp]))
    for a, b, s in sp:
        if s > 2.0 * med:
            return [a, b]
    return None


def l3_max_speed(u):
    """L3: клетка max скорости (tie — меньший λ, детерминированно); None — нет."""
    sp = interval_speeds(u)
    if sp is None:
        return None
    best = sp[0]
    for t in sp[1:]:
        if t[2] > best[2]:
            best = t
    return [best[0], best[1]]


def l3_tie_cells(u):
    """Множество клеток с max-скоростью (1 — нет tie). (MINOR-3 отчёт)"""
    sp = interval_speeds(u)
    if sp is None:
        return None
    mx = max(s[2] for s in sp)
    return [[a, b] for a, b, s in sp if s == mx]


def gap_steps(lam, interval):
    """Шагов сетки от узла λ до ближайшего граничного узла интервала (0 — внутри)."""
    i = LAM11_IDX[lam]
    a, b = LAM11_IDX[interval[0]], LAM11_IDX[interval[1]]
    if a <= i <= b:
        return 0
    return min(abs(i - a), abs(i - b))


def separation(dip_iv, comp_iv):
    """Отделение dip-интервала от competence-интервала (v3-конвенция, TZ MINOR-1):
    gap = мин. число шагов сетки между граничными узлами двух интервалов; общая граница
    или общая клетка ⇒ gap = 0 (не отделено)."""
    gap = min(gap_steps(x, comp_iv) for x in dip_iv)
    return {"gap": int(gap), "separated": bool(gap >= 1)}


def main():
    t0 = time.time()
    logline(f"AT8_1_r3 start {datetime.now(timezone.utc).isoformat()} py={sys.version.split()[0]} "
            f"np={np.__version__} (disk-only reanalysis, machine v3, раунд 2 ревью)")

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
    cells = store["cells"]

    # ---- самопроверки: пересчёт из store == артефакт
    def seedavg6_docs(obs):
        return {lam: float(np.mean([float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d][obs]
                                                      for d in MAIN6])) for sd in SEEDS]))
                for lam in LAM11}

    def seedavg_holdout():
        return {lam: float(np.mean([float(np.median([v["nll"] for v in
                                                      cells[f"s{sd}_lam{lam:.2f}"]["holdout"].values()]))
                                    for sd in SEEDS])) for lam in LAM11}

    dev = {}
    for obs in ("dip", "eff_rank"):
        r = seedavg6_docs(obs)
        dev[obs] = max(abs(r[lam] - float(curves[obs][f"{lam:.2f}"])) for lam in LAM11)
    dev["nll_holdout"] = max(abs(seedavg_holdout()[lam] - float(curves["nll_holdout"][f"{lam:.2f}"]))
                                 for lam in LAM11)
    for k, v in dev.items():
        assert v <= 1e-12, f"recompute {k} != artifact: dev={v:.2e}"
    logline(f"самопроверки: dip/eff_rank/nll_holdout (пересчёт из store) == артефакту, "
            f"max dev = {max(dev.values()):.2e}")

    u_dip = u_from_artifact(lc["dip"])
    u_art_check = u_curve({lam: float(curves["dip"][f"{lam:.2f}"]) for lam in LAM11}, LAM11)
    dev_u = max(abs(u_dip[lam] - u_art_check[lam]) for lam in LAM11)
    assert dev_u <= 1e-12, f"u_dip != u_curve(raw): dev={dev_u:.2e}"
    logline(f"самопроверка: u_dip (артефакт) == u_curve(сырой), max dev = {dev_u:.2e}")

    # ---- F2-описательный (наследие v2): max_u/overshoot/argmax — все наблюдаемые, до гейта
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
    logline(f"F2 (описательный): overshoot: "
            + ", ".join(f"{o} (max_u={f2[o]['max_u']:.3f}@{f2[o]['argmax_lam']})"
                        for o in OBS_ALL if f2[o]["overshoot"] is True))

    # ---- Raw-отчёт (F5 + §6 «что прислать»)
    dip_raw = raw_from_artifact(curves, "dip")
    d0, d1 = dip_raw[0.0], dip_raw[1.0]
    lmin = min(LAM11, key=lambda l: (dip_raw[l], LAM11_IDX[l]))  # tie — меньший λ (MINOR-4)
    lmax = max(LAM11, key=lambda l: (dip_raw[l], -LAM11_IDX[l]))
    vmin, vmax = dip_raw[lmin], dip_raw[lmax]
    raw_report = {
        "dip_seedavg": {f"{lam:.2f}": round(dip_raw[lam], 6) for lam in LAM11},
        "dip_0": d0, "dip_1": d1, "delta_dip1_minus_dip0": d1 - d0,
        "min_dip": {"value": vmin, "lam": lmin,
                    "argmin_set": [l for l in LAM11 if dip_raw[l] == vmin]},
        "max_dip": {"value": vmax, "lam": lmax,
                    "argmax_set": [l for l in LAM11 if dip_raw[l] == vmax]},
        "sign_note": "Δ > 0: dip(1) ВЫШЕ dip(0) (меньше коллапса на обученном конце); "
                     "отрицательный overshoot = min dip ГЛУБЖЕ dip(1)",
        "per_seed": {},
    }
    for sd in SEEDS:
        ds = {lam: float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d]["dip"]
                                    for d in MAIN6])) for lam in LAM11}
        er_s = {lam: float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d]["eff_rank"]
                                      for d in MAIN6])) for lam in LAM11}
        nllh_s = {lam: float(np.median([v["nll"] for v in
                                        cells[f"s{sd}_lam{lam:.2f}"]["holdout"].values()]))
                  for lam in LAM11}
        lmin_s = min(LAM11, key=lambda l: (ds[l], LAM11_IDX[l]))
        cond1_s = ds[lmin_s] < ds[1.0]
        c2_s = er_s[lmin_s] < min(er_s[0.0], er_s[1.0])
        c3_s = nllh_s[lmin_s] > max(nllh_s[0.0], nllh_s[1.0])
        raw_report["per_seed"][f"s{sd}"] = {
            "dip_curve": {f"{lam:.2f}": round(ds[lam], 6) for lam in LAM11},
            "dip_1": ds[1.0], "min_dip": ds[lmin_s], "lam_star": lmin_s,
            "cond1_min_lt_dip1": bool(cond1_s),
            "c2_eff_rank": {"er_star": er_s[lmin_s], "er_0": er_s[0.0], "er_1": er_s[1.0],
                            "c2": bool(c2_s)},
            "c3_nll_holdout": {"nll_star": nllh_s[lmin_s],
                               "endpoints": [nllh_s[0.0], nllh_s[1.0]], "c3": bool(c3_s)},
            "note": "per-seed C2 — медиана по 6 докам (seedavg-конвенция); per-seed loopy в store "
                    "отсутствует (только seed-avg в claim v1) — C1 нечислится per-seed (в note); "
                    "verdict по seed-avg (пре-рег); per-seed — отчётное",
        }
    logline(f"raw-отчёт: dip(0)={d0:+.5f} dip(1)={d1:+.5f} Δ={d1 - d0:+.5f} "
            f"min={dip_raw[lmin]:+.5f}@{lmin} max={dip_raw[lmax]:+.5f}@{lmax}")

    # ---- V3 (знаковый детектор + корроборация)
    er_raw = raw_from_artifact(curves, "eff_rank")
    nllh_raw = raw_from_artifact(curves, "nll_holdout")
    loopy_raw = raw_from_artifact(curves, "loopy")
    cond1 = bool(dip_raw[lmin] < d1)
    c1 = bool(loopy_raw[lmin] > 0.0)
    c2 = bool(er_raw[lmin] < min(er_raw[0.0], er_raw[1.0]))
    c3 = bool(nllh_raw[lmin] > max(nllh_raw[0.0], nllh_raw[1.0]))
    v3_fired = bool(cond1 and (c1 or c2 or c3))
    mx_dip = max(u_dip.values())
    argmax_dip = min(lam for lam in LAM11 if u_dip[lam] == mx_dip)
    v3 = {
        "cond1_min_lt_dip1": cond1,
        "lam_star": lmin,
        "min_dip": dip_raw[lmin], "dip_1": d1,
        "corroboration": {
            "C1_loopy": {"value_at_lstar": loopy_raw[lmin], "c1": c1},
            "C2_eff_rank": {"value_at_lstar": er_raw[lmin],
                            "endpoints": [er_raw[0.0], er_raw[1.0]], "c2": c2},
            "C3_nll_holdout": {"value_at_lstar": nllh_raw[lmin],
                               "endpoints": [nllh_raw[0.0], nllh_raw[1.0]], "c3": c3},
        },
        "fired": v3_fired,
        "descriptive_max_u": {"max_u_dip": float(mx_dip), "argmax_lam": argmax_dip,
                              "note": "описательная величина (ревью O2: max_u — не детектор)"},
        "rule": "cond1 = min_λ dip_seedavg < dip_seedavg(1.0); корроборация в λ*-клетке: "
                "C1 ∨ C2 ∨ C3; hinge-условие v2 убрано (O2-алгебра: hinge-отношение = |u−1|, "
                "эталонную подпись E-71 не ловило)",
    }
    logline(f"V3: cond1={cond1} (min {dip_raw[lmin]:+.5f}@{lmin} vs dip(1)={d1:+.5f}); "
            f"C1={c1} C2={c2} C3={c3} → fired={v3_fired}")

    # ---- V2 (таблица трёх локаторов)
    locator_table = {}
    for o in LOCATOR_OBS:
        u = u_from_artifact(lc[o])
        locator_table[o] = {"L1": l1_u05_crossing(u), "L2": l2_onset_steeper(u),
                            "L3": l3_max_speed(u), "L3_tie_cells": l3_tie_cells(u)}
    logline("локаторы (dip/nll/margin × L1/L2/L3): "
            + ", ".join(f"{o}: L1={locator_table[o]['L1']} L2={locator_table[o]['L2']} "
                        f"L3={locator_table[o]['L3']}" for o in LOCATOR_OBS))

    v2 = {}
    for L in ("L1", "L2", "L3"):
        dip_iv = locator_table["dip"][L]
        row = {"dip_interval": dip_iv, "per_competence": {}}
        if dip_iv is None:
            row["status"] = "dip_locator_undefined"
            v2[L] = row
            continue
        seps = []
        for c in COMPETENCE:
            civ = locator_table[c][L]
            if civ is None:
                row["per_competence"][c] = {"status": "competence_locator_undefined"}
                seps.append(None)
            else:
                sep = separation(dip_iv, civ)
                row["per_competence"][c] = {"competence_interval": civ, **sep}
                seps.append(sep["separated"])
        if all(s is True for s in seps):
            row["status"] = "separated"
        elif any(s is False for s in seps):
            row["status"] = "not_separated"
        else:
            row["status"] = "not_resolved"
        v2[L] = row
    v2_statuses = {L: v2[L]["status"] for L in ("L1", "L2", "L3")}
    v2_fired = all(s == "separated" for s in v2_statuses.values())
    if v2_fired:
        v2_status = "separated_on_all_locators"
    elif any(s == "separated" for s in v2_statuses.values()):
        v2_status = "verdict_differs_between_locators"  # пункт F1: ⇒ not_resolved, не V2
    else:
        v2_status = "not_resolved"
    v2_report = {"locator_table": locator_table, "per_locator": v2,
                 "statuses": v2_statuses, "overall_status": v2_status, "fired": v2_fired}
    logline(f"V2: статусы {v2_statuses} → {v2_status} (fired={v2_fired})")

    # ---- Per-seed полевой интервал (F2-патч 3)
    per_seed_field = {}
    for sd in SEEDS:
        ds = {lam: float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d]["dip"]
                                    for d in MAIN6])) for lam in LAM11}
        u_s = u_curve(ds, LAM11)
        per_seed_field[f"s{sd}"] = {"L1": l1_u05_crossing(u_s), "L3": l3_max_speed(u_s),
                                    "L3_tie_cells": l3_tie_cells(u_s)}
    l3_cells = [per_seed_field[f"s{sd}"]["L3"] for sd in SEEDS]
    if all(iv is not None for iv in l3_cells):
        # TZ MINOR-2: span = (max − min) индексов клеток (левые узлы) в шагах сетки
        lefts = [LAM11_IDX[iv[0]] for iv in l3_cells]
        span_steps = max(lefts) - min(lefts)
        field_unresolved = span_steps > 1
    else:
        span_steps, field_unresolved = None, True
    per_seed_report = {"per_seed": per_seed_field, "l3_cells": l3_cells,
                       "span_steps": span_steps, "unresolved": field_unresolved,
                       "rule": "span = (max − min) индексов per-seed L3-клеток (левые узлы) в шагах "
                               "сетки; span > 1 ⇒ полевой интервал unresolved (F2)"}
    logline(f"per-seed L3: {l3_cells} span={span_steps} → unresolved={field_unresolved}")

    # ---- families v3 (F3-патч 4)
    def has_transition(o):
        r = raw_from_artifact(curves, o)
        if r[0.0] > 0.0:
            return False  # no_transition (F4-правило v2: переход уже при λ=0)
        return any(r[lam] > 0.0 for lam in LAM11[1:])

    families_v3 = {}
    for fam, f in art.get("families", {}).items():
        members = f["members"]
        informative = [m for m in members
                       if u_from_artifact(lc[m]) is not None
                       and (m not in BINARIES or has_transition(m))]
        excluded = [m for m in members if m not in informative]
        if len(informative) < 2:
            status = "vacuous"
        else:
            status = "consistent" if f.get("intra_consistent") else "inconsistent"
        families_v3[fam] = {"members": members, "informative": informative,
                            "excluded": excluded,
                            "intra_consistent_v1": f.get("intra_consistent"), "status": status}
    n_inconsistent = sum(1 for f in families_v3.values() if f["status"] == "inconsistent")
    n_vacuous = sum(1 for f in families_v3.values() if f["status"] == "vacuous")
    families_note = (f"{n_inconsistent} измеримых семейства inconsistent, {n_vacuous} vacuous "
                     "⇒ нет единого координированного порога (mode: rep16 no_transition — "
                     "raw(0) = 0.8824 > 0, F4-правило; остаётся loopy@0.85 — D7′)")
    logline(f"families v3: " + ", ".join(f"{k}={v['status']}" for k, v in families_v3.items()))

    # ---- S9 (патч 5: декларация v2 без изменений)
    bis = art.get("bisection", {})
    s9 = {"declaration": "один раунд бисекции (точка 0.8); ширина 0.05 = ½ зазора сетки; "
          "«≤ 0.02» на этой сетке недостижимо — декларатив, не гейт (перенос декларации v2, "
          "патч 5)",
          "points": bis.get("points"), "capped": bis.get("capped"),
          "widths": bis.get("widths"),
          "cap_mismatch": "пре-рег §6.6 «кап ≤ 6 точек» vs код BIS_CAP = 12 — зафиксировано; "
                          "capped=false (не сработало)"}

    # ---- V1: структурно недостижим (наследие v2)
    v1 = {"reachable": False,
          "reason": "dip немонотонен (v1-артефакт: non_monotone_excluded, ρ = "
                    f"{lc['dip'].get('spearman_seedavg')}) → λ_c^dip не существует → "
                    "coin_all (все DISC_OTHERS сравнимы + coincide с dip) недостижим "
                    "на этом дизайне"}

    # ---- Reachability-декларация
    reachability = {
        "V3_PATHOLOGY": {"reachable": v3_fired,
                         "why": f"cond1={cond1}, C1={c1}, C2={c2}, C3={c3} "
                                f"(λ*={lmin}: dip {dip_raw[lmin]:+.5f}, er {er_raw[lmin]:.2f}, "
                                f"loopy {loopy_raw[lmin]:.3f}, nll {nllh_raw[lmin]:.3f})"},
        "V2_SOLITON_CANDIDATE": {"reachable": v2_fired,
                                 "why": f"статусы локаторов {v2_statuses} → {v2_status}"},
        "V1_GENERIC_COMPETENCE": {"reachable": v1["reachable"], "why": v1["reason"]},
        "V0_NOT_RESOLVED": {"reachable": True,
                            "why": "fallback, если V3/V2 не сработали"},
    }

    # ---- Sensitivity F6 (report-only, наследие v2)
    def seedavg_subset(docs):
        return {lam: float(np.mean([float(np.median([cells[f"s{sd}_lam{lam:.2f}"]["docs"][d]["dip"]
                                                      for d in docs])) for sd in SEEDS]))
                for lam in LAM11}

    sub = seedavg_subset(CLEAN_SUBSET)
    u_sub = u_curve(sub, LAM11)
    sp_sub = interval_speeds(u_sub)
    best = sp_sub[0]
    for t in sp_sub[1:]:
        if t[2] > best[2]:
            best = t
    peak = max(LAM11, key=lambda l: u_sub[l])
    trough = min(LAM11, key=lambda l: u_sub[l])
    sensitivity = {
        "subset": CLEAN_SUBSET,
        "raw_seedavg": {f"{lam:.2f}": round(sub[lam], 6) for lam in LAM11},
        "u_seedavg": {f"{lam:.3f}": round(u_sub[lam], 6) for lam in LAM11},
        "max_speed_interval": [best[0], best[1]],
        "u_peak_lam": peak, "u_peak": float(u_sub[peak]),
        "u_trough_lam": trough, "u_trough": float(u_sub[trough]),
        "note": "dip-сигнатура на не-луповом подмножестве (наследие v2 F6, report-only)",
    }

    # ---- Вердикт (порядок V3 > V2 > V1 > V0)
    if v3_fired:
        outcome, rule = "V3_PATHOLOGY", v3["rule"]
    elif v2_fired:
        outcome, rule = "V2_SOLITON_CANDIDATE", "отделение dip от компетентности на всех трёх локаторах"
    elif v1["reachable"]:
        outcome, rule = "V1_GENERIC_COMPETENCE", "все λ_c совпадают"
    else:
        outcome, rule = "V0_NOT_RESOLVED", "V3/V2 не сработали (машина v3)"

    expectation = "V3_PATHOLOGY"
    match = bool(outcome == expectation)
    if not match:
        logline(f"DIVERGENCE_FROM_PREREG: {outcome} != {expectation} — разбор (не «исправление» машины)")
    verdict = {
        "outcome": outcome,
        "rule": rule,
        "prereg_expectation": expectation,
        "expectation_match": match,
        "v2_divergence_note": "вердикт v2 (V2_SOLITON_CANDIDATE) — артефакт v2-машины, не "
                              "переписывается; причины смены: (a) знаковый V3-детектор (O2) "
                              "срабатывает на тех же данных; (b) V2 при трёх локаторах — "
                              "расхождение между локаторами (F1)",
        "details": {
            "v3": v3,
            "v2": v2_report,
            "per_seed_field": per_seed_report,
            "raw_report": raw_report,
            "families_v3": {"families": families_v3, "note": families_note},
            "s9": s9,
            "v1": v1,
            "f2_descriptive": f2,
            "sensitivity_F6": sensitivity,
            "reachability": reachability,
            "two_event_summary": {
                "field_event": f"отрицательный overshoot dip: min {dip_raw[lmin]:+.4f}@{lmin} "
                               f"(глубже обученного dip(1)={d1:+.4f}) с rank-collapse "
                               f"eff_rank {er_raw[lmin]:.2f} < trained {er_raw[1.0]:.2f}",
                "competence_rise": "0.85 (nll_holdout ∧ margin resolved, v1-артефакт)",
                "mode_on": "0.85 (loopy first raw > 0; rep16 no_transition — raw(0) > 0)",
            },
        },
    }
    logline(f"VERDICT: {outcome} (pre-reg ожидание: {expectation}, match={match})")

    # ---- Запись (claim + manifest + run.log)
    claim = {
        "experiment": "AT8_1 T8.1 — машина v3 (T8.1-r3) disk-only reanalysis",
        "revision": "v3 (раунд 2 ревью, docs/AT8_1_review.md)",
        "date": datetime.now(timezone.utc).isoformat(),
        "review_sha256": "56edfeba2028e03732a8544f163306ef29f9ce0b6ce2412b1226e7d01da92e9e",
        "input": {"claim_v1_file_sha256": art_sha, "store_file_sha256": store_sha,
                  "manifest_file_sha256": sha256_file(MANIFEST)},
        "machine": "v3 (TZ_add_4-8.md §T8.1 «Машина v3 (T8.1-r3)»; ревью §5 патч 1–5 + §6)",
        "selfcheck": {"recompute_dev": dev, "u_dip_dev": dev_u},
        "verdict": verdict,
        "note": "перепуск прогона не нужен: пере-агрегация committed store; store не трогается; "
                "артефакты v1 (V0_NOT_RESOLVED) и v2 (V2_SOLITON_CANDIDATE) не переписываются",
    }
    payload = sanitize(claim)
    with open(CLAIM, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
    claim_sha = sha256_file(CLAIM)

    man2 = {
        "experiment": "AT8_1 T8.1 — машина v3 (T8.1-r3)",
        "date": claim["date"],
        "claim_file_sha256": claim_sha,
        "store_file_sha256": store_sha,  # вход, не выход (не трогается)
        "script_sha256": sha256_file(os.path.join(_ROOT, "experiments", "at8", "AT8_1_r3_reanalysis.py")),
        "hash_convention": "AT0_2b-v2 C7 (file sha256 фактических файлов)",
        "note": "disk-only reanalysis из committed store (sha-верификация в claim.input); "
                "без RNG, без перепрогона.",
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(sanitize(man2), f, ensure_ascii=False, indent=1, sort_keys=True)
    logline(f"written: {CLAIM} (file sha {claim_sha[:16]}), {OUT_MANIFEST}, {RUN_LOG}")
    logline(f"AT8_1_R3_DONE verdict={outcome} match={match} (wall {time.time() - t0:.1f}s)")
    _log.write_to(RUN_LOG)  # в конце — чтобы финальный маркер был в файле


if __name__ == "__main__":
    main()
