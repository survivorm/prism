#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AT0_4 — одноразовый rebuild артефакта (exec-review P2/P3, docs/AT0_4_EXEC_REVIEW.md).

БЕЗ пере-форвардов: данные всех 112 клеток уже в store/71d3b5d868ed00c7.json.
Действия:
  1. P4-блок: per_doc + per_doc_outliers + outlier_note (build_p4 из экспериментного
     модуля — единый источник логики).
  2. verdict.summary: build_summary (P5 — «mirror P3 по G3», P4 — per-doc исключения).
  3. provenance.aggregation_rebuild: sha rebuild-скрипта, дата, причина, окно форвардов.
  4. Новая sha-цепочка: store (content-addressed, новый файл) → claim → manifest.

Ассерты (before write):
  * медианы/вердикты P4 bit-идентичны существующему блоку;
  * per_doc_outliers сверены НЕЗАВИСИМЫМ пересчётом из store.per_doc;
  * summary: префикс до «f_DC» = префиксу старого summary, суффикс — точный;
  * всё остальное в store (meta/gates/per_doc/profiles/pretest_results) неизменно.

Лог: results/AT0/AT0_4_rebuild.log. Старый store-файл удаляется (содержимое —
строгое подмножество нового).
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
OLD_STORE = os.path.join(STORE_DIR, "71d3b5d868ed00c7.json")
CLAIM_JSON = os.path.join(OUT_DIR, "AT0_4_MIX.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_4_manifest.json")
REBUILD_LOG = os.path.join(OUT_DIR, "AT0_4_rebuild.log")
SELF_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()

LOG = []


def logline(msg):
    LOG.append(msg)
    print(msg, flush=True)


# ── импорт экспериментного модуля (единый источник: build_p4/build_summary/jclean) ──
_spec = importlib.util.spec_from_file_location("at0_4_mix", os.path.join(ROOT, "experiments", "AT0_4_mix.py"))
at0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at0)

jclean, sanitize = at0.jclean, at0.sanitize
sha256_file = at0.sha256_file

logline(f"AT0_4 REBUILD (exec-review P2/P3) {time.strftime('%Y-%m-%d %H:%M:%S')}")
logline(f"rebuild script sha256: {SELF_SHA}")
logline(f"experiment script sha256: {sha256_file(os.path.join(ROOT, 'experiments', 'AT0_4_mix.py'))}")

store = json.load(open(OLD_STORE, encoding="utf-8"))
store_orig = json.loads(json.dumps(store))  # deep copy для финальной сверки
pd = store["per_doc"]
seeds = store["meta"]["seeds"]
docs_run = sorted({nm for v in pd.values() for nm in v})
logline(f"docs={len(docs_run)} seeds={seeds} store-per_doc groups={len(pd)}")

# cells-минимум из store (build_p4 читает только .get("degenerate") и ["r"])
cells = {}
for k, docs in pd.items():
    c, i, mo = k.rsplit("_", 2)
    for nm, m in docs.items():
        cells[(c, i, mo, nm)] = {"r": m.get("r"), "degenerate": m.get("degenerate", False)}

gnd = store["gates"]["G_nd"]
inst_not_testable = gnd["not_testable_instances"]
divergent = gnd["divergent_instances"]

# ── 1. P4-блок (единая логика из модуля) ─────────────────────────────────────────
p4_new = at0.build_p4(cells, docs_run, seeds, inst_not_testable, divergent)
p4_old = store["prereg"]["P4_random_order_insensitive"]

for k, v in p4_old.items():
    if k == "n_token_statistics_flags":
        continue
    if "median_abs_diff" not in v:
        assert "median_abs_diff" in p4_new[k], f"shape mismatch {k}"
        assert p4_new[k]["verdict"] == v["verdict"], f"verdict mismatch {k}"
        continue
    assert p4_new[k]["median_abs_diff"] == v["median_abs_diff"], f"median mismatch {k}"
    assert p4_new[k]["verdict"] == v["verdict"], f"verdict mismatch {k}"
assert p4_new["n_token_statistics_flags"] == p4_old["n_token_statistics_flags"]
logline(f"P4: 6/6 медиан и вердиктов bit-идентичны старому блоку (flags={p4_new['n_token_statistics_flags']})")

# независимый пересчёт per_doc_outliers из store (не через build_p4)
live_keys = [k for k, v in p4_new.items() if isinstance(v, dict) and "per_doc" in v]
out = {}
for k in live_keys:
    c, i, _ = k.split(" ")[0], k.split(" ")[1], None
    for nm in docs_run:
        rr = pd[f"{c}_{i}_real"][nm]["r"]
        rm = pd[f"{c}_{i}_mix"][nm]["r"]
        if abs(rr - rm) >= 0.1:
            out[nm] = out.get(nm, 0) + 1
expect_out = {nm: f"{c}/{len(live_keys)}" for nm, c in sorted(out.items())}
assert p4_new["per_doc_outliers"] == expect_out, f"outliers mismatch: {p4_new['per_doc_outliers']} vs {expect_out}"
assert expect_out == {"own_w0": "2/6", "own_w4": "6/6"}, f"outlier values unexpected: {expect_out}"
logline(f"P4 outliers (независимый пересчёт): {expect_out}")

# ── 2. summary (единая логика из модуля) ─────────────────────────────────────────
p1 = store["prereg"]["P1_R2_real_orthogonal_walk"]
p2_cell = store["prereg"]["P2_R1_R2_trunk_vs_embed"]["cell_verdict"]
p3 = store["prereg"]["P3_R0_real_vs_mix"]
p5 = store["prereg"]["P5_R0_real_vs_mix_fdc"]
summary_new = at0.build_summary(p1, p2_cell, p3, p5, p4_new)
summary_old = store["verdict"]["summary"]
prefix_old = summary_old.split("f_DC")[0]
assert summary_new.startswith(prefix_old), f"summary prefix changed:\nold: {prefix_old}\nnew: {summary_new.split('f_DC')[0]}"
suffix_new = ("random order-invariant: 6/6 null (медианный тест; per-doc исключения: "
              "own_w0 (2/6), own_w4 (6/6)), 0 token-statistics флаг(ов), 0 not_testable")
assert summary_new.endswith(suffix_new), f"summary suffix mismatch:\n{summary_new}"
logline("summary: префикс (до f_DC) неизменён; суффикс = медианный тест + per-doc исключения + P5 mirror")

# ── 3. provenance ─────────────────────────────────────────────────────────────────
store["provenance"]["aggregation_rebuild"] = {
    "script_sha256": SELF_SHA,
    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "reason": ("exec-review P2 (P4 per_doc + per-doc outlier note) и P3 (P5 mirror-пометка "
               "в summary); без пере-форвардов — данные клеток из store 71d3b5d868ed00c7"),
    "forward_window": ("2026-11-14T04:20:00..05:11:00 (112 forward, прогон 168465→kill→resume 181249); "
                       "05:14 resume-проход 313192 — replay из checkpoint, форвардов нет; "
                       "артефакты v1: 05:19:54"),
}

# ── 4. sha-цепочка ───────────────────────────────────────────────────────────────
store["prereg"]["P4_random_order_insensitive"] = p4_new
store["verdict"]["summary"] = summary_new

# финальная сверка: изменены ТОЛЬКО P4/summary/provenance
changed = []
for k in store_orig:
    if k == "prereg":
        for kk in store_orig["prereg"]:
            if store_orig["prereg"][kk] != store["prereg"][kk]:
                changed.append(f"prereg.{kk}")
    elif store_orig[k] != store[k]:
        changed.append(k)
assert sorted(changed) == ["prereg.P4_random_order_insensitive", "provenance", "verdict"], \
    f"unexpected changes: {changed}"
assert store_orig["verdict"]["pretest_results"] == store["verdict"]["pretest_results"], "pretest_results changed"
logline(f"store diff: только {changed}; в verdict — только summary (pretest_results неизменён)")

store_j = jclean(store)
blob = json.dumps(store_j, ensure_ascii=False, sort_keys=True)
store_sha = hashlib.sha256(blob.encode()).hexdigest()
store_path = os.path.join(STORE_DIR, store_sha[:16] + ".json")
assert store_sha[:16] != "71d3b5d868ed00c7", "store sha unchanged — rebuild не применился?"
with open(store_path, "w") as f:
    f.write(blob)
logline(f"store: {store_path} (sha {store_sha[:16]})")

claim = {"meta": store["meta"], "gates": store["gates"], "per_doc": store["per_doc"],
         "prereg": store["prereg"], "verdict": store["verdict"],
         "store": {"path": os.path.relpath(store_path, OUT_DIR), "sha256": store_sha}}
claim_j = jclean(claim)
claim_blob = json.dumps(claim_j, ensure_ascii=False, sort_keys=True)
with open(CLAIM_JSON, "w") as f:
    f.write(claim_blob)
logline(f"claim: {CLAIM_JSON}")

man = {"manifest": {"claim_sha256": hashlib.sha256(claim_blob.encode()).hexdigest(),
                    "store_sha256": store_sha,
                    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "aggregation_rebuild": True}}
with open(MAN_JSON, "w") as f:
    json.dump(sanitize(man), f, indent=1)
logline(f"manifest: {MAN_JSON}")

# ── самосверка записанного ──────────────────────────────────────────────────────
assert hashlib.sha256(open(store_path, "rb").read()).hexdigest() == store_sha, "store file sha"
assert json.load(open(CLAIM_JSON, encoding="utf-8"))["store"]["sha256"] == store_sha, "claim.store.sha256"
m2 = json.load(open(MAN_JSON, encoding="utf-8"))["manifest"]
assert m2["claim_sha256"] == hashlib.sha256(open(CLAIM_JSON, "rb").read()).hexdigest(), "manifest.claim_sha256"
assert m2["store_sha256"] == store_sha, "manifest.store_sha256"
logline("sha-цепочка: store==имя, claim.store==store, manifest.claim==claim, manifest.store==store")

os.remove(OLD_STORE)
logline(f"старый store удалён: {os.path.basename(OLD_STORE)}")
logline("AT0_4 REBUILD DONE")

with open(REBUILD_LOG, "w") as f:
    f.write("\n".join(LOG) + "\n")
print(f"\nrun log: {REBUILD_LOG}")
