#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AT0_4 — одноразовый rebuild артефакта (exec-review r5).

NOT REUSABLE: expectations pinned to store e852a4bd88c92649
(full sha e852a4bd88c9264954a294f7a194825c85ab34a95dceb3c1fd0f6402e6e4a877).
Другой store — скрипт не запустится (ассерты).

БЕЗ пере-форвардов: данные всех 112 клеток в store e852a4bd… (rebuild r4).
Вся логика — функции модуля experiments/at0/AT0_4_mix.py (единый источник).

Действия (r5: «ACCEPT как own-doc r/DC pilot, wording/consistency fixes»):
 1. c0-sensitivity (r5 C1.3/C5, НОВЫЙ блок r_decomposition.trained_vs_random.sensitivity):
    all_8 / no_period1_docs (без w0,w4) / gnd_filtered_pairs (per-instance G_nd ok;
    w5 исключён целиком, w2/w7 — по одной паре) / no_periodic_docs (без w5,w7) +
    pairwise_note (48 ≠ 48 независимых, r5 C1.2) + status (post-hoc, descriptive, r5 C1.1).
 2. P2 (r5 C2): cell_relative_note (data-driven: S11 0.1736, S12 0.1627, S13 0.1027) +
    scale_note: «trunk_carries» = trunk_carries_absolute_scale, «same» ≠ zero-effect.
    Вердикты same ×3 + cell_verdict trunk_carries — не меняются (пре-рег-строки фикс).
 3. P1 (r5 C3): band_meaning — бэнд НЕ детектор когерентного дрейфа (drift_smooth_walk
    r_med≈192 вне бэнда), на данных structurally unreachable; P1 — исторический
    пре-рег, не содержательный тест. Verdict vacuous ×3 — не меняется.
 4. doc_order_provenance (r5 C4): per-doc note — own_w5 (period-13, пары не live: G_nd
    not-ok во всех 6 инстансах), own_w7 (period-16, |Δr| < 0.1 во всех live-парах);
    interp «≈93%/≈94%» по max_share (было «≥93%»); note_key.
 5. meta (r5 C5/C7): doc_stratification (period_1: w0,w4; periodic: w5,w7 — own_w7
    ревью не учло; clean: w1,w2,w3,w6) + mix_note: «mix treatment fixed; no
    permutation-seed sensitivity».
 6. verdict.summary: P2-reading (абсолютный масштаб, НЕ zero-effect, 0.10–0.17) +
    c0-summary (descriptive, sensitivity 8/8, 7/7, 6/6, 4/4).
    pretest_results — бит-неизменен (ни один вердикт не изменился).
 7. provenance.aggregation_rebuild_r5 + sha-цепочка store → claim → manifest.

ВАЖНО: старый store (e852a4bd88c92649) НЕ УДАЛЯЕТСЯ — untracked в git; удаление —
на этапе коммита, по команде юзера (как 4c0a1db2 в r4).

Ассерты (до записи):
  * store meta (кроме mix_note/doc_stratification) / gates / per_doc / profiles /
    token_provenance / verdict.pretest_results — БИТ-неизменны; scope — бит-неизменен;
  * P4/P3/P5 — БИТ-неизменны (build_p4/build_p3_p5 не тронуты r5-правками);
  * P1: verdict vacuous ×3, r/r_max bit-идентичны, synthetic_null bit-идентичен;
  * P2: verdicts same ×3 + cell_verdict trunk_carries, per_doc bit-идентичны,
    n_docs 7/7/5;
  * c0-тест: per-instance 8/8 × 6 (p=0.0078); per_doc c0_ratio — независимый пересчёт;
  * sensitivity: независимый пересчёт из store.per_doc + G_nd (медианы/знаки/состав);
  * changed == ровно ожидаемый набор ключей.

venv: transformers сломан параллельной сессией — модуль импортируется с
transformers-стубом (rebuild data-only, форвардов нет); decode — через `tokenizers`.
Лог: results/AT0/AT0_4_rebuild_r5.log.
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
OLD_STORE = os.path.join(STORE_DIR, "e852a4bd88c92649.json")
OLD_STORE_SHA = "e852a4bd88c9264954a294f7a194825c85ab34a95dceb3c1fd0f6402e6e4a877"
CLAIM_JSON = os.path.join(OUT_DIR, "AT0_4_MIX.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_4_manifest.json")
REBUILD_LOG = os.path.join(OUT_DIR, "AT0_4_rebuild_r5.log")
SELF_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

LOG = []


def logline(msg):
    LOG.append(msg)
    print(msg, flush=True)


# ── transformers-стуб (venv сломан; rebuild data-only) ─────────────────────────
try:
    import transformers  # noqa: F401
    STUBBED = False
except Exception as e:
    stub = types.ModuleType("transformers")
    stub.__version__ = "stub(venv-broken)"
    stub.AutoModelForCausalLM = object
    stub.AutoConfig = object
    sys.modules["transformers"] = stub
    STUBBED = True
    logline(f"transformers STUB (venv broken: {type(e).__name__}: {str(e).splitlines()[0]}) "
            "— data-only rebuild, форвардов нет")

# ── импорт модуля (единый источник логики) ─────────────────────────────────────
_spec = importlib.util.spec_from_file_location("at0", os.path.join(ROOT, "experiments", "at0", "AT0_4_mix.py"))
at0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at0)

jclean, sanitize = at0.jclean, at0.sanitize
sha256_file = at0.sha256_file

logline(f"AT0_4 REBUILD r5 (exec-review r5: c0-sensitivity, P2/P1 wording, "
        f"w5/w7 notes, stratification) {time.strftime('%Y-%m-%d %H:%M:%S')}")
logline(f"rebuild script sha256: {SELF_SHA}")
logline(f"experiment script sha256: {sha256_file(os.path.join(ROOT, "experiments", "at0", "AT0_4_mix.py"))} "
        f"(METRICS_VERSION={at0.METRICS_VERSION})")
assert sha256_file(OLD_STORE) == OLD_STORE_SHA, "старый store не тот — pinned sha mismatch"

store = json.load(open(OLD_STORE, encoding="utf-8"))
store_orig = copy.deepcopy(store)
pd = store["per_doc"]
seeds = store["meta"]["seeds"]
docs_run = at0.DOCS
assert sorted(pd["R0_trained_real"].keys()) == docs_run, "docs mismatch"
logline(f"docs={len(docs_run)} seeds={seeds} store-per_doc groups={len(pd)}")

# cells-минимум из store (build_p*/r_decomposition читают r/f_dc/c0/r_max/degenerate)
cells = {}
for k, docs in pd.items():
    c, i, mo = k.rsplit("_", 2)
    for nm, m in docs.items():
        cells[(c, i, mo, nm)] = {"r": m.get("r"), "f_dc": m.get("f_dc"),
                                 "c0": m.get("c0"), "r_max": m.get("r_max"),
                                 "degenerate": m.get("degenerate", False)}

gnd = store["gates"]["G_nd"]["per_instance"]
inst_not_testable = store["gates"]["G_nd"]["not_testable_instances"]
divergent = store["gates"]["G_nd"]["divergent_instances"]
logline("G_nd not-ok: " + ", ".join(
    f"{inst}.{nm}" for inst in sorted(gnd) for nm, v in sorted(gnd[inst].items()) if not v.get("ok")))

# ── token_provenance (пересчёт → bit-identity; нужен для rdec/dop/strat) ───────
z = np.load(at0.OWN_NPZ, allow_pickle=True)
toks = {nm: np.asarray(z[f"{nm}_tok"]).ravel() for nm in docs_run}
tp = at0.token_provenance(toks)
assert tp == store["token_provenance"], "token_provenance changed (tokens?)"
period = {nm: tp[nm]["min_period_90pct"] for nm in docs_run}
assert period == {"own_w0": 1, "own_w4": 1, "own_w5": 13, "own_w7": 16,
                  "own_w1": None, "own_w2": None, "own_w3": None, "own_w6": None}, \
    f"period-карта изменилась: {period}"
logline("token_provenance: bit-идентичен; period-карта: w0/w4=1, w5=13, w7=16, "
        "w1/w2/w3/w6=None")

# ── 1. P4 (БЕЗ изменений r5 — бит-идентичность) ────────────────────────────────
p4_new = at0.build_p4(cells, docs_run, seeds, inst_not_testable, divergent, gnd=gnd)
p4_old = store["prereg"]["P4_random_order_insensitive"]
assert p4_new == p4_old, "P4 изменился, хотя build_p4 r5 не трогал"
logline("P4: БИТ-НЕИЗМЕНЕН (build_p4 r5 не тронут); null ×6, outliers 2/6+6/6")

# ── 2. P3/P5 (БЕЗ изменений r5 — бит-идентичность) ─────────────────────────────
p3_new, p5_new = at0.build_p3_p5(cells, docs_run)
assert p3_new == store["prereg"]["P3_R0_real_vs_mix"]
assert p5_new == store["prereg"]["P5_R0_real_vs_mix_fdc"]
logline("P3/P5: БИТ-НЕИЗМЕНЕНЫ (build_p3_p5 r5 не тронут)")

# ── 3. P1 (band_meaning обновлён; verdict vacuous ×3 — не меняется) ────────────
t0 = time.time()
null_rep = at0.synthetic_null_r()
logline(f"synthetic_null_r: {time.time() - t0:.1f}s (B={null_rep['B']}, seed={null_rep['seed']})")
assert null_rep["drift_smooth_walk"]["band_frac_0.75_1.5"] < 0.01
assert null_rep["drift_smooth_walk"]["r"]["median"] > 100
p1_new = at0.build_p1(cells, docs_run, seeds, inst_not_testable, divergent, null_rep, gnd=gnd)
p1_old = store["prereg"]["P1_R2_real_orthogonal_walk"]
assert p1_new["synthetic_null"] == p1_old["synthetic_null"]
for inst in [f"S{S}" for S in seeds]:
    assert p1_new[inst]["verdict"] == p1_old[inst]["verdict"] == "vacuous"
    assert p1_new[inst]["per_doc"] == p1_old[inst]["per_doc"], f"P1 per_doc {inst} изменился"
    assert p1_new[inst]["c0_stats"] == p1_old[inst]["c0_stats"]
assert "НЕ детектор когерентного дрейфа" in p1_new["band_meaning"]
assert "исторический пре-рег, не содержательный тест" in p1_new["band_meaning"]
assert p1_new["band_note"] == p1_old["band_note"], "band_note не должен меняться"
logline("P1: vacuous ×3 НЕ ИЗМЕНИЛСЯ (per_doc/c0_stats bit-идентичны); band_meaning "
        "+ r5 C3 (не детектор дрейфа; исторический пре-рег)")

# ── 4. P2 (cell_relative_note + scale_note; verdicts — не меняются) ────────────
p2_new = at0.build_p2(cells, docs_run, seeds, inst_not_testable, divergent, gnd=gnd)
p2_old = store["prereg"]["P2_R1_R2_trunk_vs_embed"]
for inst in [f"S{S}" for S in seeds]:
    assert p2_new[inst]["verdict"] == p2_old[inst]["verdict"] == "same"
    assert p2_new[inst]["per_doc"] == p2_old[inst]["per_doc"], f"P2 per_doc {inst} изменился"
    assert p2_new[inst]["n_docs"] == p2_old[inst]["n_docs"]
assert p2_new["cell_verdict"] == p2_old["cell_verdict"] == "trunk_carries"
assert p2_new["cell_note"] == p2_old["cell_note"]
assert "trunk_carries_absolute_scale" in p2_new["cell_relative_note"]
assert "0.1736" in p2_new["cell_relative_note"] and "0.1627" in p2_new["cell_relative_note"] \
    and "0.1027" in p2_new["cell_relative_note"], p2_new["cell_relative_note"]
assert "0.10–0.17" in p2_new["scale_note"] and "НЕ zero-effect" in p2_new["scale_note"]
assert p2_new["scale_note"] != p2_old["scale_note"], "scale_note не обновился?"
logline("P2: same ×3 + trunk_carries НЕ ИЗМЕНИЛИСЬ; + cell_relative_note "
        "(S11 0.1736, S12 0.1627, S13 0.1027) + scale_note (0.10–0.17, НЕ zero-effect)")

# ── 5. r_decomposition (sensitivity/pairwise_note/status; ядро — не меняется) ──
rdec = at0.build_r_decomposition(cells, docs_run, seeds, token_prov=tp, gnd=gnd)
t2r = rdec["trained_vs_random"]
t2r_old = store["r_decomposition"]["trained_vs_random"]
assert rdec["identity"] == store["r_decomposition"]["identity"]
assert rdec["per_cell"] == store["r_decomposition"]["per_cell"], "per_cell изменился?"
assert t2r["per_doc"] == t2r_old["per_doc"], "per_doc c0 изменился?"
assert t2r["per_instance"] == t2r_old["per_instance"]
assert t2r["median_ratio"] == t2r_old["median_ratio"]
assert t2r["boot_ci95"] == t2r_old["boot_ci95"]
assert t2r["n_pairs"] == 8 and t2r["n_positive"] == 8
assert t2r["pairwise"] == t2r_old["pairwise"] == {"n": 48, "n_positive": 48,
                                                   "median": t2r_old["pairwise"]["median"]}
for k, v in t2r["per_instance"].items():
    assert v["n"] == 8 and v["n_positive"] == 8 and v["sign_test_p"] == 0.0078125, f"{k}: {v}"
# независимый пересчёт per_doc c0_ratio из store.per_doc
for nm in docs_run:
    tr = pd["R0_trained_real"][nm]["c0"]
    rs = [pd[f"{c}_S{S}_real"][nm]["c0"] for c in ("R1", "R2") for S in seeds]
    exp_ratio = tr / float(np.median(rs))
    assert abs(t2r["per_doc"][nm]["c0_ratio"] - exp_ratio) < 1e-15, f"c0_ratio {nm}"
    exp_cos = 1.0 - pd["R0_trained_real"][nm]["r"] * tr / 2.0
    assert abs(t2r["per_doc"][nm]["endpoint_cos_trained"] - exp_cos) < 1e-15
# р5-новое: status / pairwise_note / sensitivity
assert "descriptive" in t2r["status"] and "НЕ pre-registered" in t2r["status"]
assert "НЕ 48 независимых" in t2r["pairwise_note"]
sens = t2r["sensitivity"]

def _expect_sens(doc_list, pair_ok=None, name=""):
    """Независимый пересчёт subset-статистики из store.per_doc + gnd."""
    per_doc_ratios, excluded, rw_s = [], [], []
    for nm in doc_list:
        tr = pd["R0_trained_real"][nm]["c0"]
        rr = []
        for S in seeds:
            for c in ("R1", "R2"):
                m = pd[f"{c}_S{S}_real"][nm]
                if m.get("c0") is None:
                    continue
                if pair_ok is not None and not pair_ok(nm, f"{c}_S{S}"):
                    continue
                rr.append(m["c0"])
        if not rr:
            excluded.append(nm)
            continue
        per_doc_ratios.append(tr / float(np.median(rr)))
        rw_s += [tr / x for x in rr]
    return per_doc_ratios, excluded, rw_s

def _check_sens(name, doc_list, pair_ok=None):
    per_doc_ratios, excluded, rw_s = _expect_sens(doc_list, pair_ok)
    s = sens[name]
    assert s["excluded_docs"] == excluded, f"sens {name}: excluded {s['excluded_docs']} vs {excluded}"
    assert s["n_docs"] == len(per_doc_ratios)
    assert abs(s["median_ratio"] - float(np.median(per_doc_ratios))) < 1e-15, f"sens {name} med"
    assert s["n_positive"] == int(sum(1 for x in per_doc_ratios if x > 1.0))
    assert s["pairwise"]["n"] == len(rw_s)
    assert s["pairwise"]["n_positive"] == int(sum(1 for x in rw_s if x > 1.0))
    assert abs(s["pairwise"]["median"] - float(np.median(rw_s))) < 1e-15, f"sens {name} pw med"
    return s

check_docs = {nm: period[nm] for nm in docs_run}
_s1 = _check_sens("all_8", list(docs_run))
assert _s1["n_positive"] == 8 and _s1["pairwise"]["n_positive"] == 48
_no_p1 = [nm for nm in docs_run if check_docs[nm] != 1]
_s2 = _check_sens("no_period1_docs", _no_p1)
assert _s2["n_docs"] == 6 and _s2["n_positive"] == 6 and _s2["pairwise"]["n"] == 36 \
    and _s2["pairwise"]["n_positive"] == 36, _s2
gnd_ok = lambda nm, k: gnd[k][nm]["ok"]
_s3 = _check_sens("gnd_filtered_pairs", list(docs_run), pair_ok=gnd_ok)
assert _s3["excluded_docs"] == ["own_w5"], f"w5 должен исключён целиком: {_s3['excluded_docs']}"
assert _s3["n_docs"] == 7 and _s3["n_positive"] == 7
assert _s3["pairwise"]["n"] == 40 and _s3["pairwise"]["n_positive"] == 40, _s3
_no_per = [nm for nm in docs_run if check_docs[nm] is None]
_s4 = _check_sens("no_periodic_docs", _no_per)
assert _s4["n_docs"] == 4 and _s4["n_positive"] == 4 and _s4["pairwise"]["n"] == 24 \
    and _s4["pairwise"]["n_positive"] == 24, _s4
# частичные пары: w2 (R2_S13 not-ok), w7 (R1_S13 not-ok) — по одной паре
partial_expect = {"own_w2": 5, "own_w7": 5}
for nm, n_exp in partial_expect.items():
    n_ok = sum(1 for S in seeds for c in ("R1", "R2") if gnd[f"{c}_S{S}"][nm]["ok"])
    assert n_ok == n_exp, f"{nm}: gnd-ok {n_ok} != {n_exp}"
assert "sensitivity" in t2r["summary_line"] and "post-hoc, descriptive" in t2r["summary_line"]
logline(f"sensitivity: all_8 {sens['all_8']['n_positive']}/8 (pw 48/48, med "
        f"{sens['all_8']['median_ratio']:.4f}); no_period1 "
        f"{sens['no_period1_docs']['n_positive']}/6 (med {sens['no_period1_docs']['median_ratio']:.4f}); "
        f"gnd_filtered {sens['gnd_filtered_pairs']['n_positive']}/7 (pw 40/40, med "
        f"{sens['gnd_filtered_pairs']['median_ratio']:.4f}, w5 исключён); no_periodic "
        f"{sens['no_periodic_docs']['n_positive']}/4 (med {sens['no_periodic_docs']['median_ratio']:.4f}) — "
        "ВСЕ > 1; независимый пересчёт сошёлся")

# ── 6. doc_order_provenance (r5 C4: notes + interp) ───────────────────────────
dop = at0.build_doc_order_provenance(cells, docs_run, seeds, p4_new, tp)
dop_old = store["doc_order_provenance"]
for nm in docs_run:
    old, new = dop_old["docs"][nm], dop["docs"][nm]
    for k in ("order_flag", "order_flag_count", "min_period_90pct", "max_share", "n_uniq",
              "c0_trained", "c0_random_median", "r_trained_real", "max_abs_diff_r"):
        assert new[k] == old[k], f"dop {nm}.{k} изменился"
assert dop["docs"]["own_w0"]["interp"] == "period-1 token loop (single-token mass ≈93%) — вырожденный low-motion вход"
assert dop["docs"]["own_w4"]["interp"] == "period-1 token loop (single-token mass ≈94%) — вырожденный low-motion вход"
w5n, w7n = dop["docs"]["own_w5"]["note"], dop["docs"]["own_w7"]["note"]
assert "period-13" in w5n and "G_nd not-ok во всех random-инстансах" in w5n, w5n
assert "period-16" in w7n and "0.0017" in w7n, w7n
assert "note_key" in dop and "order_flag ≠ token-периодичность" in dop["note_key"]
assert dop["note"] == dop_old["note"], "doc-level note не должен меняться"
logline("doc_order_provenance: w0 ≈93% / w4 ≈94%; w5 note (period-13, не live: G_nd); "
        "w7 note (period-16, max |Δr| 0.0017 < 0.1); note_key; doc-level note неизменен")

# ── 7. summary + pretest_results ──────────────────────────────────────────────
summary_new = at0.build_summary(p1_new, p2_new["cell_verdict"], p3_new, p5_new, p4_new,
                                extra_note=t2r["summary_line"])
summary_old = store["verdict"]["summary"]
for frag in ("structurally unreachable", "при n=8 p<0.05 только при 8/8",
             "period-1 token loop", "trained/random median ratio", "2-DoF",
             "trunk_carries", "own_w4 (6/6)", "own_w0 (2/6)", "descriptive",
             "trunk_carries = абсолютный масштаб", "0.10–0.17", "sensitivity"):
    assert frag in summary_new, f"summary lacks: {frag}"
pretest_new = {
    "P1": {k: v["verdict"] for k, v in p1_new.items()
           if isinstance(v, dict) and "verdict" in v},
    "P2": {"cell_verdict": p2_new["cell_verdict"],
           **{k: v["verdict"] for k, v in p2_new.items()
              if isinstance(v, dict) and "verdict" in v}},
    "P3": p3_new["verdict"], "P5": p5_new["verdict"],
    "P4": {k: v["verdict"] for k, v in p4_new.items()
           if isinstance(v, dict) and "verdict" in v},
    "P4_token_statistics_flags": p4_new["n_token_statistics_flags"],
}
pretest_old = store["verdict"]["pretest_results"]
assert pretest_new == pretest_old, f"pretest_results changed: {pretest_new} vs {pretest_old}"
logline("pretest_results: БИТ-НЕИЗМЕНЕН (ни один вердикт не изменился)")
logline(f"summary:\n  old: {summary_old}\n  new: {summary_new}")

# ── 8. применение + provenance ───────────────────────────────────────────────
assert store["meta"]["scope"] == store_orig["meta"]["scope"], "scope не должен меняться"
store["meta"]["mix_note"] = (
    "mix = одна фиксированная перестановка на док (seed 7) — mix treatment fixed; "
    "no permutation-seed sensitivity (новых форвардов нет); эффект mix может зависеть "
    "от конкретной перестановки (r3 C6.4, r5 C7)")
store["meta"]["doc_stratification"] = {
    "period_1_low_motion": [nm for nm in docs_run if period[nm] == 1],
    "periodic_non_low_motion": [nm for nm in docs_run if period[nm] not in (None, 1)],
    "clean_no_periodic": [nm for nm in docs_run if period[nm] is None],
    "note": ("стратификация по token_provenance.min_period_90pct + c0 (r5 C5); "
             "own_w7 — тоже периодический (period-16, rep16≥0.9) — в clean-наборе "
             "ревью r5 его не учтено (р5-ответка C5); c0-headline стратифицирован в "
             "r_decomposition.trained_vs_random.sensitivity")}
assert store["meta"]["doc_stratification"]["period_1_low_motion"] == ["own_w0", "own_w4"]
assert store["meta"]["doc_stratification"]["periodic_non_low_motion"] == ["own_w5", "own_w7"]
assert store["meta"]["doc_stratification"]["clean_no_periodic"] == ["own_w1", "own_w2", "own_w3", "own_w6"]
store["prereg"]["P1_R2_real_orthogonal_walk"] = p1_new
store["prereg"]["P2_R1_R2_trunk_vs_embed"] = p2_new
store["prereg"]["P3_R0_real_vs_mix"] = p3_new
store["prereg"]["P4_random_order_insensitive"] = p4_new
store["prereg"]["P5_R0_real_vs_mix_fdc"] = p5_new
store["r_decomposition"] = rdec
store["doc_order_provenance"] = dop
store["verdict"]["summary"] = summary_new
store["provenance"]["aggregation_rebuild_r5"] = {
    "script_sha256": SELF_SHA,
    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "reason": ("exec-review r5 (docs/AT0_4_EXEC_REVIEW.md): C1 c0-sensitivity "
               "(all_8/no_period1/gnd_filtered/no_periodic) + pairwise_note (48 ≠ 48 "
               "независимых) + status (post-hoc, descriptive); C2 P2 cell_relative_note "
               "(S11 0.1736, S12 0.1627, S13 0.1027) + scale_note (trunk_carries = "
               "trunk_carries_absolute_scale, НЕ zero-effect; пре-рег-вердикты не "
               "пересматриваются); C3 P1 band_meaning (бэнд не детектор когерентного "
               "дрейфа; исторический пре-рег); C4 doc_order_provenance per-doc notes "
               "(w5 period-13 не live: G_nd; w7 period-16 ниже порога) + interp ≈93%/≈94%; "
               "C5 meta.doc_stratification (own_w7 period-16 — ревью считало clean; "
               "коррекция в р5-ответке); C7 mix_note (fixed, no permutation-seed "
               "sensitivity); C6 — sha-цепочка подтверждена (см. р5-ответку, файлы в "
               "репо, не пустые). БЕЗ пере-форвардов — данные клеток из store "
               "e852a4bd88c92649…; ни один вердикт pretest_results не изменился"),
    "replaced_store_sha256": OLD_STORE_SHA,
    "replaced_store_name": "e852a4bd88c92649.json",
    "old_store_retained": True,
    "old_store_note": ("e852a4bd88c92649 НЕ удалён — untracked в git (как 4c0a1db2 в "
                       "r4); удаление — на этапе коммита, по команде юзера"),
    "script_diff_chain": (store["meta"]["script_sha256"]
                          + " (r4-модуль, store.meta) → "
                          + sha256_file(os.path.join(ROOT, "experiments", "at0", "AT0_4_mix.py"))
                          + " (r5-фиксы, текущий) → " + SELF_SHA + " (этот rebuild). "
                            "Более ранние звенья — в provenance.aggregation_rebuild_r4."),
    "s5_note": store["provenance"]["aggregation_rebuild_r4"]["s5_note"],
    "venv_note": "transformers в venv сломан параллельной сессией (huggingface-hub 2.4.0) — "
                 "rebuild с transformers-стубом; decode — через tokenizers (локальный tokenizer.json)",
    "forward_window": store["provenance"]["aggregation_rebuild"]["forward_window"],
}

# финальная сверка: изменены ТОЛЬКО ожидаемые ключи (по UNION ключей)
changed = []
for k in set(store_orig) | set(store):
    if k == "prereg":
        for kk in set(store_orig["prereg"]) | set(store["prereg"]):
            if store_orig["prereg"].get(kk) != store["prereg"].get(kk):
                changed.append(f"prereg.{kk}")
    elif k == "verdict":
        for kk in set(store_orig["verdict"]) | set(store["verdict"]):
            if store_orig["verdict"].get(kk) != store["verdict"].get(kk):
                changed.append(f"verdict.{kk}")
    elif k == "meta":
        for kk in set(store_orig["meta"]) | set(store["meta"]):
            if store_orig["meta"].get(kk) != store["meta"].get(kk):
                changed.append(f"meta.{kk}")
    elif store_orig.get(k) != store.get(k):
        changed.append(k)
expected = sorted(["prereg.P1_R2_real_orthogonal_walk", "prereg.P2_R1_R2_trunk_vs_embed",
                   "verdict.summary", "provenance",
                   "meta.mix_note", "meta.doc_stratification",
                   "r_decomposition", "doc_order_provenance"])
assert sorted(changed) == expected, f"unexpected changes: {sorted(changed)} vs {expected}"
logline(f"store diff: только {sorted(changed)}")
assert store["token_provenance"] == store_orig["token_provenance"]
assert store["per_doc"] == store_orig["per_doc"] and store["gates"] == store_orig["gates"]
assert store["profiles"] == store_orig["profiles"]
assert store["meta"]["scope"] == store_orig["meta"]["scope"]
assert store["verdict"]["pretest_results"] == store_orig["verdict"]["pretest_results"]

# ── 9. sha-цепочка (старый store НЕ удаляется) ────────────────────────────────
store_j = jclean(store)
blob = json.dumps(store_j, ensure_ascii=False, sort_keys=True)
store_sha = hashlib.sha256(blob.encode()).hexdigest()
assert store_sha != OLD_STORE_SHA, "store sha unchanged — rebuild не применился?"
store_path = os.path.join(STORE_DIR, store_sha[:16] + ".json")
with open(store_path, "w") as f:
    f.write(blob)
logline(f"store: {store_path} (sha {store_sha[:16]})")

claim = {"meta": store["meta"], "gates": store["gates"], "per_doc": store["per_doc"],
         "prereg": store["prereg"], "verdict": store["verdict"],
         "token_provenance": store["token_provenance"],
         "token_provenance_meta": store["token_provenance_meta"],
         "r_decomposition": rdec, "doc_order_provenance": dop,
         "store": {"path": os.path.relpath(store_path, OUT_DIR), "sha256": store_sha}}
claim_j = jclean(claim)
claim_blob = json.dumps(claim_j, ensure_ascii=False, sort_keys=True)
with open(CLAIM_JSON, "w") as f:
    f.write(claim_blob)
logline(f"claim: {CLAIM_JSON}")

man = {"manifest": {"claim_sha256": hashlib.sha256(claim_blob.encode()).hexdigest(),
                    "store_sha256": store_sha,
                    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "aggregation_rebuild": True,
                    "aggregation_rebuild_r3": True,
                    "aggregation_rebuild_r4": True,
                    "aggregation_rebuild_r5": True,
                    "retired_stores": ["f1cc7265dd40c874.json (deleted in rebuild r3; in git 61483f3)",
                                       "4c0a1db2e7984621.json (deleted in rebuild r4 commit)",
                                       "e852a4bd88c92649.json (retained on disk; delete at commit)"]}}
with open(MAN_JSON, "w") as f:
    json.dump(sanitize(man), f, indent=1)
logline(f"manifest: {MAN_JSON}")

# ── самосверка записанного ───────────────────────────────────────────────────
assert hashlib.sha256(open(store_path, "rb").read()).hexdigest() == store_sha, "store file sha"
assert json.load(open(CLAIM_JSON, encoding="utf-8"))["store"]["sha256"] == store_sha, "claim.store.sha256"
m2 = json.load(open(MAN_JSON, encoding="utf-8"))["manifest"]
assert m2["claim_sha256"] == hashlib.sha256(open(CLAIM_JSON, "rb").read()).hexdigest(), "manifest.claim_sha256"
assert m2["store_sha256"] == store_sha, "manifest.store_sha256"
assert os.path.exists(OLD_STORE), "старый store должен быть сохранён (untracked!)"
logline("sha-цепочка: store==имя, claim.store==store, manifest.claim==claim, manifest.store==store")
logline(f"старый store СОХРАНЁН (untracked, удаление — в коммите): {os.path.basename(OLD_STORE)}")
logline("AT0_4 REBUILD r5 DONE")

with open(REBUILD_LOG, "w") as f:
    f.write("\n".join(LOG) + "\n")
print(f"\nrun log: {REBUILD_LOG}")
