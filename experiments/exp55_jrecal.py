#!/usr/bin/env python
"""E-55 v14: пост-hoc пересчёт J-гейта через j_paired (paired shuffle-бейслайн).
J — свойство цели, не фита ⇒ независимо от nelder-прогона, детерминирован по SEED_T.

Ревью v14:
- порог НЕ «1.0 из головы»: j_C3 калибруется на seed'ах 13..22 (≠ тестового 11, без циркулярности),
  thr = percentile(cal, 5); дополнит. условие отделения от шума: j_C3 > 4·j_C2 и j_C2 < 0.25;
- перезапись gate_J в exp55_sbi_gate.json с полной провенанс-меткой (gate_J_original/reason)
  и пересчётом gate_all из сохранённых булевых гейтов + нового gate_J.
Пересчитываются J только для C1,C2,C3,C5,C7 (single-mode; C4/C6 — multi-mode, вне объёма)."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp55_sbi as S

# те же контрольные конфигурации, что в exp55_sbi (target-доки, SEED_T)
OM8 = S.OM8
tg_docs = {
    "C1": S.target([S.BETA])[2],
    "C2": S.target([S.BETA, ('c', 0.92, OM8, 0.08, S.w2)])[2],
    "C3": S.target([S.BETA, ('rot', 0.92, OM8, 0.08, (S.u1, S.u2))])[2],
    "C5": S.target([])[2],
    "C7": S.target([S.BETA, ('c', 0.92, OM8, 0.08, S.w2)],
                   amp_scale=[1.0 if i % 2 == 0 else 0.0 for i in range(S.NDOC)])[2],
}
res = {}
for name, dd in tg_docs.items():
    jg, j, js, jx = M.j_paired(dd)
    res[name] = dict(j_glob=jg, j=j, j_shuf=js, J_excess=jx)
    print(name, f"j_glob={jg:.3f} j={j:.3f} j_shuf={js:.3f} ex={jx:+.3f}", flush=True)

# калибровка ожидания C3 на seed'ах, ОТЛИЧНЫХ от тестового SEED_T=11 (иначе порог циркулярен);
# n=10 и percentile(5) вместо mean−2σ при n=6 — тот же смысл, но устойчивее к шуму σ (review v14 #2)
cal = []
st_seed = S.SEED_T
try:
    for s in range(13, 23):
        S.SEED_T = s
        _, _, dd = S.target([S.BETA, ('rot', 0.92, OM8, 0.08, (S.u1, S.u2))])
        cal.append(M.j_paired(dd)[1])
finally:
    S.SEED_T = st_seed   # review v14: восстановление даже при exception в калибровке
cal = np.array(cal)
thr = float(np.percentile(cal, 5))
res["calib"] = dict(seeds=list(range(13, 23)), j=cal.tolist(), thr=thr)
print(f"калибровка C3 j на seed 13..22: mean={cal.mean():.3f} sd={cal.std(ddof=1):.3f} "
      f"p5(thr)={thr:.3f} min={cal.min():.3f}", flush=True)

# гейт: сигнал на тестовом seed выше калиброванного ожидания И отделён от шума; C2 сам тихий
gateJ = bool(res["C3"]["j"] > max(thr, 4 * res["C2"]["j"]) and res["C2"]["j"] < 0.25)
res["gate_J"] = gateJ
print("gate_J:", gateJ, {k: round(res[k]["j"], 3) for k in res if k.startswith("C")}, flush=True)

gp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exp55_sbi_gate.json")
if os.path.exists(gp):
    d = json.load(open(gp))
    for k in ("C1", "C2", "C3", "C5", "C7"):
        if k in d:
            d[k].update({kk: res[k][kk] for kk in ("j_glob", "j", "j_shuf", "J_excess")})
    if "gate_J_original" not in d:                 # фиксируем ОДИН раз: на повторных прогонах
        d["gate_J_original"] = d.get("gate_J")      # jrecal оригинал из sbi не затирается (review v14)
        # миграция v13→v14: файл без gate_J_owner написан старой sbi (гейт по Jex) —
        # честно помечаем источник, не делая вид, что оригинала не было
        if "gate_J_owner" not in d:
            d["gate_J_original_src"] = "pre-v14 sbi (Jex-based gate)"
    d["gate_J"] = gateJ
    d["gate_J_recalc_reason"] = ("phase_current j_plane: null-biased (plane fit to antisym noise); "
                                 "replaced by j_paired (plane from sym part, paired shuffle baseline), "
                                 "threshold = p5 of seeds 13-22")
    d["jrecal_script"] = "exp55_jrecal.py"
    d["j_note"] = ("гейт использует j против калиброванного thr и уровня C2; J_excess/j_shuf — "
                   "диагностика: на чистом C1/C2 знаменатель (сим-диагонали shuffled) мал → j_shuf "
                   "раздувается, J_excess там отрицателен и в решение не входит. Ограничение "
                   "j_paired: чувствителен к top-2 Sm — rotation plane ⟂ β может не попадать в "
                   "них (C3' перпендикуляр не тестировался). Следствие: если stage1b даст "
                   "'R wins, J low' — это может быть НЕ отсутствие вращения, а слепота j_paired "
                   "к плоскости вне top-2 Sm; тогда прогнать C3' (u ⟂ w1) как контроль. "
                   "C7: факт obj=19.1 против потолка 20 — pass впритык (дизайн C7, не pipeline). "
                   "j null-контролей (C1/C5) — это УРОВЕНЬ ШУМА в плоскости, не сигнал (факт: "
                   "C5=0.023, C1=0.071); C3 сравнивается с калиброванным thr и 4*C2, не с C5")
    # gate_all = пер-контрольные passed (вложенные!) + верхнеуровневые булевы гейты + НОВЫЙ gate_J;
    # gate_J_original ИСКЛЮЧЁН (иначе старый False обнулял бы смысл пересчёта — review v14 #1)
    n_per = sum(1 for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
                if isinstance(d.get(k), dict) and isinstance(d[k].get("passed"), bool))
    assert n_per == 7, f"jrecal: {n_per}/7 per-control passed в sbi_gate.json — сначала прогони exp55_sbi"
    bools = [v for k, v in d.items() if isinstance(v, bool)
             and k not in ("gate_all", "gate_J_original")]
    bools += [d[k]["passed"] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
              if isinstance(d.get(k), dict) and isinstance(d[k].get("passed"), bool)]
    d["gate_all"] = all(bools)
    json.dump(d, open(gp, "w"), indent=1, default=float)
    print("обновлён gate_J (с провенансом) в", gp, flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "results", "exp55_jrecal.json"), "w"), indent=1, default=float)
