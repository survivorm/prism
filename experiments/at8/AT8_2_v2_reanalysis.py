# -*- coding: utf-8 -*-
"""AT8_2 (T8.2) — disk-only реанализ v2 (машина v2 по прецеденту AT8_1_v2).

Причина (внешнее ревью AT8_2, docs/dynamic/AT8_2_results_review.md, sha 6acfdd64…):
- MAJOR-1: медианы оси A в r1-claim исчислены по цензурированной популяции
  (recovered/no_effect) — против TZ §5 («по измеримым (doc,W)»). Fикс: основная
  популяция = все измеримые с ℓ_rec (not_recovered, ℓ_rec=D_max — цензурированное
  значение, входит); censored-excluded — вторичный с явной меткой.
- MINOR-1: NOT_RESOLVED-гейт «большинство not_recovered» не был реализован;
  реализован по полной измеримой популяции + уточнение TZ v1.2.

БЕЗ модели и без перепрогона (store на месте):
1. sha-верификация store/manifest/r1-claim (fail-loud);
2. пересчёт ЯЧЕЕК из store общим фрагментом at8_2_analysis.build_cells_from_store
   (тот же код, что в прогоне) — ПОБЕВОВОЕ сверение с r1-claim (любое расхождение
   = STOP, RC≠0);
3. пересчёт осей A/B и вердикта с r2-фиксами (at8_2_analysis);
4. артефакт: AT8_2_RECOVERY_v2.json + AT8_2_v2_manifest.json + AT8_2_v2_run.log
   (DB trackio backfill — re-run trackio_migration.py, источник истины — лог).

Запуск: /opt/miniconda3/bin/python3.12 experiments/at8/AT8_2_v2_reanalysis.py
"""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from at8_2_analysis import (N_DOSE, W_SET, W_AXIS, EPS_MIN, THR, D_MIN_MEAS,
                            build_cells_from_store, axis_A, axis_B,
                            compute_verdict, cells_equal)

OUT_DIR = os.path.join(_ROOT, "results", "AT8")
STORE = os.path.join(OUT_DIR, "AT8_2_RECOVERY_store.json")
R1_CLAIM = os.path.join(OUT_DIR, "AT8_2_RECOVERY.json")
R1_MANIFEST = os.path.join(OUT_DIR, "AT8_2_RECOVERY_manifest.json")
CLAIM = os.path.join(OUT_DIR, "AT8_2_RECOVERY_v2.json")
MANIFEST = os.path.join(OUT_DIR, "AT8_2_v2_manifest.json")
RUNLOG = os.path.join(_ROOT, "logs", "AT8", "AT8_2_v2_run.log")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    _logf = open(RUNLOG, "w")

    def logline(s):
        print(s, flush=True)
        _logf.write(s + "\n")
        _logf.flush()

    logline(f"AT8_2 r2 (T8.2) disk-only reanalysis start "
            f"{datetime.now(timezone.utc).isoformat()} py={sys.version.split()[0]} "
            f"np={np.__version__} (machine v2, без модели/перепрогона)")

    # ---- 1. sha-верификация ----
    man = json.load(open(R1_MANIFEST))
    r1_claim = json.load(open(R1_CLAIM))
    store_sha = sha256_file(STORE)
    r1_claim_sha = sha256_file(R1_CLAIM)
    assert man["store_file_sha256"] == store_sha, \
        f"STOP: store sha {store_sha[:16]} != manifest {man['store_file_sha256'][:16]}"
    assert man["claim_file_sha256"] == r1_claim_sha, \
        f"STOP: r1-claim sha {r1_claim_sha[:16]} != manifest {man['claim_file_sha256'][:16]}"
    logline(f"sha-верификация: store {store_sha[:16]} / r1-claim {r1_claim_sha[:16]} "
            f"— совпадает с manifest (r1 прогон не тронут)")

    names = r1_claim["docs"]["names"]
    store = json.load(open(STORE))

    # ---- 2. пересчёт ячеек общим фрагментом + побитовая сверка с r1 ----
    cells = build_cells_from_store(store, names, n_dose=N_DOSE, w_set=W_SET,
                                   eps_min=EPS_MIN, thr=THR, d_min_meas=D_MIN_MEAS)
    n_diff = 0
    for k, c1 in r1_claim["cells"].items():
        ok, why = cells_equal(c1, cells.get(k, {}))
        if not ok:
            n_diff += 1
            logline(f"  RASXOD {k}: {why}")
    extra = set(cells) - set(r1_claim["cells"])
    assert n_diff == 0 and not extra, \
        f"STOP: расхождения r1-claim vs пересчёт: {n_diff} ячеек, extra={extra}"
    n_meas = sum(1 for c in cells.values() if "dip_inj_d0" in c and c.get("status") is None)
    logline(f"ячейки: {len(cells)} (measurable={n_meas}) — пересчёт == r1-claim "
            f"(0 расхождений; общий фрагмент at8_2_analysis воспроизводит r1-числа)")

    # ---- 3. оси + вердикт (r2-фиксы) ----
    axA = axis_A(cells, n_dose=N_DOSE, w_set=W_SET, w_axis=W_AXIS)
    axB = axis_B(cells)
    verdict, vdetail, diag = compute_verdict(axA, axB, cells, n_dose=N_DOSE, w_axis=W_AXIS)
    match = (verdict == "ELASTIC")
    logline(f"ось A (W={W_AXIS}, ПОЛНАЯ популяция, TZ §5): "
            f"lrec={axA['W64']['lrec']} ratio={axA['W64']['ratio']} "
            f"b={axA['W64']['b']} r2={axA['W64']['r2']}")
    logline(f"  censored_excluded (вторичный): lrec={axA['W64']['lrec_censored_excluded']} "
            f"(r1-claim был по этой популяции — MAJOR-1)")
    logline(f"ось B: frac_to_global={axB['frac_to_global']} (n={axB['n_to_global_pairs']})")
    logline(f"diag: {diag}")
    logline(f"VERDICT: {verdict} (match_pre_reg={match}): {vdetail}")

    # ---- 4. артефакт ----
    claim = dict(
        experiment="AT8_2", cell="T8.2 windowed recovery + dose",
        revision="v2 (disk-only reanalysis, machine v2; r1-claim — снапшот прогона)",
        tz_version="v1.2 (16.11, по results-review r1: MAJOR-1 + MINOR-1)",
        date=datetime.now(timezone.utc).isoformat(),
        base_run=dict(
            r1_claim_file_sha256=r1_claim_sha, store_file_sha256=store_sha,
            r1_run_name="at8_2_recovery_r1", r1_wall_s=r1_claim["wall_s"],
            note=("прогон не тронут: store/claim r1 в исходном виде; v2 — пересчёт "
                  "агрегатов (оси/вердикт) из того же store общим фрагментом")),
        setup=r1_claim["setup"],
        docs=r1_claim["docs"],
        gates=r1_claim["gates"],
        prereg=r1_claim["prereg"],
        r2_fixes=dict(
            MAJOR_1=("основная популяция оси A = все измеримые (doc,W) с ℓ_rec (TZ §5); "
                     "not_recovered (ℓ_rec=D_max) — цензурированное значение, входит в "
                     "медиану; censored_excluded (recovered+no_effect) — вторичный, "
                     "явно помечен"),
            MINOR_1=("NOT_RESOLVED-гейты по явным популяциям: «<6 пар (doc,n) с ℓ_rec>0» "
                     "— по W=64; «большинство not_recovered» — по полной измеримой "
                     "популяции (все W, D_max>=8); чтение по W64-оси (75% not_recovered) "
                     "— diagnostic (axis_W64_notrecovered_share), не гейт (TZ v1.2 §5)"),
        ),
        lrec_by_n=dict(W64=axA["W64"]["lrec"],
                       W64_censored_excluded=axA["W64"]["lrec_censored_excluded"],
                       combined_secondary=axA["combined_secondary"]["lrec"],
                       combined_secondary_censored_excluded=
                       axA["combined_secondary"]["lrec_censored_excluded"]),
        axis_A=axA, axis_B=axB,
        diagnostics=diag,
        cells=cells,
        verdict=dict(value=verdict, match_pre_reg=match, detail=vdetail),
        summary=(f"T8.2 v2 (disk-only): {verdict}. {vdetail}. Ось A — полная "
                 f"популяция (TZ §5). 40 форвардов + 2 S8-check (r1-прогон, wall "
                 f"{r1_claim['wall_s']:.0f}s); v2 — disk-only, {time.time()-t0:.1f}s."),
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
    man_v2 = dict(claim_file_sha256=csha, store_file_sha256=store_sha,
                  r1_claim_file_sha256=r1_claim_sha, r1_manifest_file_sha256=
                  sha256_file(R1_MANIFEST),
                  e64_gpu_sha=r1_claim["setup"]["e64_gpu_sha"],
                  docs_sha256=r1_claim["docs"]["docs_sha256"],
                  pool_fp=r1_claim["docs"]["pool_fp"],
                  model=r1_claim["setup"]["model"],
                  date=datetime.now(timezone.utc).isoformat(),
                  verdict=verdict, wall_s=claim["wall_s"])
    json.dump(man_v2, open(MANIFEST, "w"), indent=1, ensure_ascii=False)
    logline(f"written: {CLAIM} (sha {csha[:16]}), {MANIFEST}")
    logline(f"AT8_2_V2_DONE verdict={verdict} match_pre_reg={match} "
            f"(wall {time.time()-t0:.1f}s)")
    _logf.close()


if __name__ == "__main__":
    main()
