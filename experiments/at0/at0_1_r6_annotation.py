#!/usr/bin/env python3
"""r6-cosmetic правка AT0_1 (docs/AT0_1_review.md, «Первоочередное» 1–5 + мелкое #6).
Только текстовые аннотации + детерминированный пересчёт tpu_dev — чисел прогона НЕТ.
Формат идентичен генерации (AT0_1_accounting.py:1182-1189): claim — компактный
json.dumps(jclean, ensure_ascii=False); manifest — indent=1; manifest = claim + manifest-блок.
"""
import hashlib
import importlib.util
import json
import os

ROOT = "/home/survivor/work/prism"
spec = importlib.util.spec_from_file_location("pf", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)
jclean = pf.jclean

CLAIM_P = os.path.join(ROOT, "results/AT0/T0_ACCOUNTING.json")
MAN_P = os.path.join(ROOT, "results/AT0/AT0_1_manifest.json")
STORE_SHA = "23281c64cc418ff1fedad2a5013c4bf7c8b7dd7680ac8d4fa7cf746c62fd1732"
BASE_RUN_SHA = "d4a4b10135e8b63f361f6fe75475cff7d44b6810e9248d03a8bdf96b5b73d94a"

claim = json.load(open(CLAIM_P, encoding="utf-8"))
man = json.load(open(MAN_P, encoding="utf-8"))

# --- пре-проверки ---
assert {k: v for k, v in man.items() if k != "manifest"} == claim, "manifest != claim+block до правки"
assert man["manifest"]["store_sha256"] == STORE_SHA
assert man["manifest"]["claim_sha256"] == BASE_RUN_SHA
assert hashlib.sha256(open(CLAIM_P, "rb").read()).hexdigest() == BASE_RUN_SHA

# --- review #2: P4 direction ---
p4 = claim["prereg"]["4_plateau_vs_fDC"]
p4["h_raw"]["direction"] = "real_at_fdc"
p4["h_raw"]["direction_note"] = (
    "r6 (docs/AT0_1_review.md, #2): «refuted» = pair-block null не объясняет real (p_perm = 0.005), "
    "не «плато отвергнуто»; real (est = −9.7e-5, CI содержит 0) на ~19× ближе к f_dc, чем сдвиг null "
    "(E = −1.85e-3) — плато на уровне f_dc")
p4["du_unit"]["direction"] = "real_at_fdc"
p4["du_unit"]["direction_note"] = (
    "r6 (docs/AT0_1_review.md, #2): «refuted» = null не объясняет real; est = −9.1e-5 (CI [−2.1e-4, −1.2e-5], "
    "ниже 0 с точностью CI) на ~20× меньше по величине сдвига null (E = −1.86e-3) — real практически на уровне f_dc")
p4["v_tangent"]["direction"] = "real_above_fdc"
p4["v_tangent"]["direction_note"] = (
    "r6 (docs/AT0_1_review.md, #2): est = +1.94e-3, CI [+6.9e-4, +3.4e-3] — плато ВЫШЕ f_dc "
    "(null E = −2.23e-3); здесь «refuted» = null не объясняет real, и real не на f_dc")

# --- review #4: MA(1) applicability (v_tangent) ---
p4["v_tangent"]["null_analytic_ma1"]["applicability"] = (
    "MA(1) неприменима к v_tangent (r6, docs/AT0_1_review.md, #4): f_dc ≈ 0.08 (DC-dominant, "
    "f_dc_median = 0.0797); z = (s_doc + f_dc)/SE = mean_band/SE — blow-up (median_z = 381, "
    "n_abs_z_ge_3 = 51/51) — by-product DC-доминирования в числителе, не «огромная структура»; "
    "описательно только (вердикт — по первичному каналу, r5 D2)")

# --- review #3: t0_headline_note (T_none_full) ---
claim["aggregates"]["t0_headline_note"] += (
    " r6 (docs/AT0_1_review.md, #3): T_none_full = (r−1)/2 − (N−1)·f_dc — РАЗНОСТЬ двух больших "
    "слагаемых (Gram-тождество, G8: max |sum_rho_full − (r−1)/2| = 1.07e-14; пересчёт r6 — bit-eq "
    "медианы −12.868956825537097 по 53 докам nondegen), НЕ «excess по всем лагам»; для v_tangent "
    "(r_median ≈ 24.9, f_dc_median ≈ 0.08) ≈ −12.87 — знак ПРОТИВОПоложен band mean P4 (+0.00194), "
    "не «плато провалено»")

# --- review #5: G0a tpu_dev (post-hoc пересчёт) ---
g0a = claim["gates"]["G0a_tpu_hn_field"]
g0a["max_rel_dev_reconstruction"] = 3.7939126356469717e-16
g0a["max_rel_dev_reconstruction_detail"] = {
    "definition": ("max по TPU-докам (own_w0..15, cor_w0..15) × L ∈ {1,5,11,17,23} × row t "
                   "‖U·hn − h_raw‖₂/‖h_raw‖₂ (r6, docs/AT0_1_review.md, #5); "
                   "U = u/max(‖u‖,1e-12), u = h_raw/max(‖h_raw‖,1e-12), hn = ‖h_raw‖ — зеркало "
                   "load_source + cell-loop (experiments/AT0_1_accounting.py:185-189, 275-276)"),
    "worst": {"doc": "own_w6", "L": 5},
    "max_by_source": {"P1_own": 3.7939126356469717e-16, "P2_corpus": 2.651087330108574e-16},
    "max_by_layer": {"L1": 2.6175831752051945e-16, "L5": 3.7939126356469717e-16,
                     "L11": 2.6682856128897777e-16, "L17": 2.6607990136516047e-16,
                     "L23": 2.5493718517941864e-16},
    "self_audit": ("bit-eq воспроизведение отчётных G0a evidence_own_w0 (row_norm_median, hn_median, "
                   "hn_vs_column_norms_max_rel, row_norm_vs_hn_corr)"),
    "inputs_sha256": {
        "results/e64/tpu_v8/P1_own.npz": "783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198",
        "results/e64/tpu_v8/P2_corpus.npz": "5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125",
    },
    "provenance": ("post-hoc детерминированный пересчёт r6 (2026-11-13) — не перепуск эксперимента; "
                   "числа прогона не менялись (experiments/at0_1_tpu_dev_recompute.py)"),
}

# --- review #6: G8 n_l1_absent ---
claim["gates"]["G8_gram"]["n_l1_absent"] = 2

# --- meta.r6_annotation ---
claim["meta"]["r6_annotation"] = {
    "date": "2026-11-13",
    "review": "docs/AT0_1_review.md",
    "base_run": ("file sha256 d4a4b10135e8b63f361f6fe75475cff7d44b6810e9248d03a8bdf96b5b73d94a "
                 "(пин TZ_add_0-3.md v1.2, D4); числа прогона и store 23281c64cc418ff1 не менялись"),
    "changes": [
        "prereg.4_plateau_vs_fDC.{h_raw,du_unit,v_tangent}: + direction, direction_note (ревью #2)",
        "prereg.4_plateau_vs_fDC.v_tangent.null_analytic_ma1: + applicability (ревью #4)",
        "aggregates.t0_headline_note: + T_none_full — разность, не excess (ревью #3)",
        "gates.G0a_tpu_hn_field: + max_rel_dev_reconstruction(+detail) — post-hoc пересчёт (ревью #5)",
        "gates.G8_gram: + n_l1_absent = 2 (ревью #6)",
    ],
    "invariant": ("числа прогона не менялись; manifest = claim + manifest-блок; добавленные величины — "
                  "детерминированные пересчёты из записанных входов (sha256) либо явные счёты из артефакта"),
}

# --- запись claim ---
claim_bytes = json.dumps(jclean(claim), ensure_ascii=False).encode("utf-8")
with open(CLAIM_P, "wb") as f:
    f.write(claim_bytes)
claim_sha = hashlib.sha256(claim_bytes).hexdigest()

# --- manifest = claim + manifest-блок ---
man2 = {k: claim[k] for k in ("meta", "gates", "prereg", "aggregates", "store")}
man2["manifest"] = {"store_sha256": STORE_SHA, "claim_sha256": claim_sha, "written": True}
man_bytes = json.dumps(jclean(man2), ensure_ascii=False, indent=1).encode("utf-8")
with open(MAN_P, "wb") as f:
    f.write(man_bytes)
man_sha = hashlib.sha256(man_bytes).hexdigest()

# --- пост-проверки ---
c2 = json.loads(claim_bytes)
m2 = json.loads(man_bytes)
assert {k: v for k, v in m2.items() if k != "manifest"} == c2
assert m2["manifest"]["claim_sha256"] == claim_sha
assert c2["prereg"]["4_plateau_vs_fDC"]["h_raw"]["direction"] == "real_at_fdc"
assert c2["prereg"]["4_plateau_vs_fDC"]["v_tangent"]["direction"] == "real_above_fdc"
assert c2["gates"]["G0a_tpu_hn_field"]["max_rel_dev_reconstruction"] == 3.7939126356469717e-16

print("claim_sha  =", claim_sha)
print("manifest_sha =", man_sha)
print("store_sha  =", STORE_SHA)
print("OK: invariant manifest=claim+block; all asserts passed")
