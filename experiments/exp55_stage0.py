#!/usr/bin/env python
"""E-55 стадия 0: real Δ-кривая на 14 лагах + гейт совпадения с 6-лаг артефактом E-53b,
per-domain таблица, гипотеза Kung (1a — гипотезы, НЕ вердикт), scale-чек ритма (loops).
Ограничение (фиксирую честно): полный контроль «ритм текста» требует декодированного
сгенерированного текста exp47b (в артефакте только sha хэши) — отдельная микро-задача."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "results", "exp53b_servo.json")

docs = [E.derive(d) for d in E.load_real()]
print("docs:", len(docs), flush=True)
cur = M.lag14(docs)
old = json.load(open(OLD))["delta_curve"]

ok, bad = [], []
for k in (1, 2, 3, 4, 8, 16):
    dm = abs(cur[k]["med"] - old[str(k)]["median"])
    # только med — RNG-независимая инвариантность (ревью v12п3: delta зависит от порядка shuffle-вытяг)
    (ok if dm < 5e-4 else bad).append((k, round(dm, 5)))
print("гейт совпадения (med) с E-53b:", "PASS" if not bad else f"FAIL {bad}", flush=True)

out = {"curve": {str(k): cur[k] for k in cur}, "gate_vs_e53b": "PASS" if not bad else str(bad)}
# kung на ПОСЛЕДОВАТЕЛЬНЫХ 1..14 (review v15 critical: Hankel требует шаг 1; прежний вызов
# на разреженных LAGS14 сдвигал ось → r slow mode и «osc» были артефактами)
curK = M.lag14(docs, lags=tuple(range(1, 15)))
y = [curK[k]["delta"][0] for k in range(1, 15)]
s = [max((curK[k]["delta"][2] - curK[k]["delta"][1]) / 3.92, 2e-3) for k in range(1, 15)]
out["sigma"] = {str(k): si for k, si in zip(M.LAGS14,
              [max((cur[k]["delta"][2] - cur[k]["delta"][1]) / 3.92, 2e-3) for k in M.LAGS14])}
out["kung_hypothesis"] = M.kung_fit(y, s, lags=tuple(range(1, 15)))   # 1a (гейт сетки включён)
print("Kung-гипотеза:", out["kung_hypothesis"], flush=True)

doms = {}
for d in docs:
    doms.setdefault(d["dom"], []).append(d)
out["per_domain"] = {}
for name, dd in sorted(doms.items()):
    c = M.lag14(dd)
    out["per_domain"][name] = {"n": len(dd), "delta": {str(k): round(c[k]["delta"][0], 4) for k in (1, 2, 3, 4, 8, 16)},
                               "pos_hump": max((k for k in (2, 3, 4, 5, 6, 7, 8)), key=lambda k: c[k]["delta"][0])}
    print("dom", name, out["per_domain"][name], flush=True)

try:
    jd = json.load(open(os.path.join(HERE, "..", "results", "exp47b_gen_kinematics.json")))["docs"]
    lp = [np.median([l[2] for l in d["loops"]]) if d["loops"] else float("nan") for d in jd]
    out["rhythm_scale_check"] = {"median_loop_tokens": float(np.nanmedian(lp)),
                                 "note": "окно T=384 шага (D,T из exp53b); медианный цикл ~378 токенов ≈"
                                         " соизмеримо с длиной окна → период цикла порядка всего окна,"
                                         " а не k=3..8: ритм-артефакт циклов не может давать горб 3–12"
                                         " напрямую (вывод прежний, арифметика исправлена review v15)" +
                                         "; полный контроль требует декодированного текста (sha в артефакте)"}
    print("scale-чек:", out["rhythm_scale_check"], flush=True)
except Exception as ex:
    out["rhythm_scale_check"] = {"error": str(ex)}

J = M.phase_current(docs)
jp = M.j_paired(docs)
out["J_real"] = dict(j_glob=J[0], j_plane=J[1],
                     j_paired=dict(j_glob=jp[0], j=jp[1], j_shuf=jp[2], J_excess=jp[3]),
                     note="j_plane из phase_current — legacy/артефакт (плоскость из Sa, review v15/E-56);"
                           " решение по j_paired против собственного shuffled-бейслайна")
print("J real:", J, "| j_paired:", jp, flush=True)

json.dump(out, open(os.path.join(HERE, "..", "results", "exp55_stage0.json"), "w"), indent=1, default=float)
print("saved -> results/exp55_stage0.json", flush=True)
