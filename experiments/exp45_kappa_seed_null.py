"""E-45d: разброс κ по iid-правкам одного семейства (seed null) — сколько значит «κ не сдвинулся».

Зачем. В таблице E-45 у `rand_K_d*` κ/κ₀ < 1, у `rand_full_d*` κ/κ₀ > 1, и на этом хотелось
построить вывод «правки всей матрицы поднимают κ, правки по 128 строкам U128 — опускают».
Это ловушка с одним draw на семейство: якобиан L23 (readout) есть функция ТОЛЬКО 128 строк
`U128` (`make_j_fn` строит `J` из `W[U128]`), поэтому `rand_full` отличается от `rand_K`
в геометрическом канале только тем, какая именно гауссова матрица легла на эти 128 строк,
а величина подобрана одной и той же линейной калибровкой под тот же дрейф. Значит разница
знаков может быть разбросом одной выборки.

Метрика. Восемь iid-правок строк `U128` при согласованном медианном дрейфе J = 0.12 (тот же
класс возмущений, что и оба семейства таблицы), для каждой — κ(r=16) на той же подыборке
документов, что и вся таблица, плюс попарный бутстрап по документам от Δκ (тот же срез
позиций, база против правки). Выход — разброс κ/κ₀ внутри семейства и разрешающая способность
попарной оценки: всё, что меньше этого разброса, интерпретировать нельзя.

Дополнительно, без единой новой операции над моделью: из `results/exp28_leaf_structure.json`
(там лежат per-doc κ по 32 документам) считается несогласованный док-бутстрап уровня κ —
порог, ниже которого протоколы с разными наборами документов вообще неразличимы.

Запуск:  cd experiments && nice -n 10 ../.venv/bin/python exp45_kappa_seed_null.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _e38_cache import jcache_path                    # noqa: E402
import exp45_damage_vs_structure as E45               # noqa: E402

L, S, R = E45.L, E45.S, E45.R
N_BOOT = 2000
SEED0 = 100               # не пересекается с 0 и 11 из таблицы
# Режим по умолчанию — только CI УРОВНЯ κ (numpy по уже закоммиченному exp28-артефакту, секунды,
# модель не нужна). Разброс внутри iid-семейства с тех пор меряется СТРОКАМИ ТаБЛИЦЫ
# (`E45_SEEDNULL=8 … --resume`): там те же iid-правки, но со всеми метриками сразу, поэтому
# дублировать их отдельным процессом смысла нет. Попарный CI на Δκ остаётся опцией --paired N:
# он требует J правки, которого в таблице нет (там только сводная медиана κ).
PAIRED = 0
for a in sys.argv[1:]:
    if a.startswith("--paired"):
        PAIRED = int(a.split("=")[1]) if "=" in a else 4
KBASE_REL = 2e-3          # база обязана сойтись с базой таблицы (детерминировано)


def log(*a):
    print(*a, flush=True)


def kappa_docs(J, s_len=S):
    """κ(r) как E45.kappa_of, но с сохранением per-doc медиан (для попарного бутстрапа)."""
    ks = []
    for i in range(0, len(J), s_len):
        Jd = torch.from_numpy(J[i:i + s_len].astype(np.float32))
        _, _, k2, _ = E45.E28.document_profile(Jd, [1, R, 32], r_main=R)
        k2 = np.asarray(k2, dtype=np.float64)
        ks.append(float(np.median(k2[~np.isnan(k2)])))
    return np.array(ks)


def main():
    t0 = time.time()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    torch.set_grad_enabled(False)
    out = {"experiment": "E-45d", "L": L, "r": R, "n_paired": PAIRED,
           "edits": [], "level_boot": {}}

    # --------- несогласованный док-бутстрап уровня κ из уже закоммиченного артефакта ---------
    p28 = os.path.join(os.path.dirname(_HERE), "results", "exp28_leaf_structure.json")
    rng = np.random.default_rng(0)
    with open(p28) as f:
        d28 = json.load(f)
    for Lk in ("6", "12", "18", "22", "23"):
        km = np.array([x["kappa_mean"] for x in d28["layers"][Lk]["docs"]])
        rec = {"kappa_mean": float(km.mean()),
               "rel_sd_docs_pct": float(100 * km.std(ddof=1) / km.mean())}
        for n in (32, 8):
            b = np.array([km[rng.integers(0, len(km), n)].mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(b, [2.5, 97.5])
            rec[f"ci32_rel_pct_n{n}"] = float(100 * (hi - lo) / 2 / km.mean())
        out["level_boot"][f"L{Lk}"] = rec
        log(f"L{Lk}: κ={km.mean():.6f} rel SD между документами {rec['rel_sd_docs_pct']:.1f} %, "
            f"несогласованный CI уровня ±{rec['ci32_rel_pct_n32']:.1f} % (32 дока) / "
            f"±{rec['ci32_rel_pct_n8']:.1f} % (8 документов)")

    if PAIRED == 0:
        out["wall_sec"] = round(time.time() - t0, 1)
        out["note"] = ("разброс iid-семейства и полосы по метрикам — в таблице E-45, строки "
                       "seednull_s*; здесь только CI уровня κ")
        with open(os.path.join(os.path.dirname(_HERE), "results",
                               "exp45_kappa_seed_null.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        log("→ results/exp45_kappa_seed_null.json (только уровень; --paired N для Δκ)")
        return

    # --------- iid null на том же срезе, что и таблица E-45 (попарный Δκ, --paired N) ---------
    ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy") else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    del ck
    spec_pos = np.concatenate([np.arange(i * S, (i + 1) * S) for i in E45.SPEC_DOCS])
    log(f"срез геометрии: {len(E45.SPEC_DOCS)} документов × {S} = {len(spec_pos)} позиций")

    from transformers import AutoModelForCausalLM
    log("loading model (fp32) ...")
    # эталон базы — base-строка таблицы E-45 (та же подыборка, то же построение J)
    tab = os.path.join(os.path.dirname(_HERE), "results", "exp45_damage_vs_structure.json")
    if not os.path.exists(tab):
        tab = tab.replace(".json", ".partial.json")
    with open(tab) as f:
        dj = json.load(f)
    base_row = next(e for e in dj["edits"] if e["name"] == "base")
    KAPPA_TABLE_BASE = base_row["kappa_A"]
    model = AutoModelForCausalLM.from_pretrained(
        E45.MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    W = model.lm_head.weight.detach().float().clone()
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    WU0 = W[U128].numpy().astype(np.float64)
    J_of = E45.make_j_fn(Hst[spec_pos], g_np)
    del model
    J0 = J_of(WU0)
    kd0 = kappa_docs(J0)
    kap0 = float(np.median(kd0))
    log(f"  база κ = {kap0:.6f}")
    assert abs(kap0 / KAPPA_TABLE_BASE - 1) < KBASE_REL, (
        f"база {kap0:.6f} не сходится с базой таблицы {KAPPA_TABLE_BASE:.6f} — "
        "срез или построение J разошлись с exp45_damage_vs_structure")

    for seed in range(SEED0, SEED0 + PAIRED):
        G = np.random.default_rng(seed).standard_normal(WU0.shape) * float(WU0.std())
        lin = E45.drift_med(J0, J_of(G))[0]          # дрейф линеен по σ → одна калибровка
        s_ = 0.12 / lin
        J1 = J0 + s_ * J_of(G)
        dmed = E45.drift_med(J0, J1)[0]
        kd1 = kappa_docs(J1)
        del J1
        d = kd1 - kd0
        rng_b = np.random.default_rng(seed + 7)
        b = np.array([d[rng_b.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(b, [2.5, 97.5])
        rec = {"seed": seed, "drift_med": float(dmed),
               "kappa": float(np.median(kd1)), "ratio": float(np.median(kd1) / kap0),
               "d_kappa_pooled": float(np.median(d)),
               "d_kappa_mean": float(d.mean()),
               "paired_ci_rel_pct": [float(100 * lo / kap0), float(100 * hi / kap0)],
               "doc_kappa_base": [float(x) for x in kd0],
               "doc_kappa_edit": [float(x) for x in kd1]}
        out["edits"].append(rec)
        log(f"  seed {seed}: drift {rec['drift_med']:.4f}  κ {rec['kappa']:.6f} "
            f"(×{rec['ratio']:.4f})  попарный CI Δκ = [{rec['paired_ci_rel_pct'][0]:+.2f}, "
            f"{rec['paired_ci_rel_pct'][1]:+.2f}] %  [{time.time() - t0:.0f} с]")

    rr = np.array([e["ratio"] for e in out["edits"]])
    out["family_spread"] = {"min": float(rr.min()), "max": float(rr.max()),
                            "sd_pct": float(100 * rr.std(ddof=1)),
                            "half_range_pct": float(100 * (rr.max() - rr.min()) / 2)}
    ci = np.array([e["paired_ci_rel_pct"] for e in out["edits"]])
    out["paired_resolution_rel_pct"] = float(np.median((ci[:, 1] - ci[:, 0]) / 2))
    out["wall_sec"] = round(time.time() - t0, 1)
    log(f"разброс κ/κ₀ внутри семейства: [{rr.min():.4f}, {rr.max():.4f}] "
        f"(±{out['family_spread']['half_range_pct']:.2f} %), медианный попарный CI "
        f"Δκ = ±{out['paired_resolution_rel_pct']:.2f} %, wall {out['wall_sec']} с")

    rd = os.path.join(os.path.dirname(_HERE), "results")
    with open(os.path.join(rd, "exp45_kappa_seed_null.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"→ {rd}/exp45_kappa_seed_null.json")


if __name__ == "__main__":
    main()
