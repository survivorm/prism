"""E-45e: где на L23 вообще есть зазор — скан `r` на базовом якобиане.

Зачем. Пререгистрационный критерий 0 («`gap16 < 0.01` или >5 % позиций без зазора ⇒ разрез
неустойчив, `Δκ` на этой правке невалиден») срабатывает уже на БАЗЕ: `gap16` в медиане 0.00875,
и 98.9 % позиций имеют зазор < 1 %. До этого скана оставался неотвеченным вопрос «а есть ли
устойчивый разрез вообще, или r=16 просто неудачная отметка»: без него «разрез вырожден» нельзя
отличить от «разрез вырожден везде, и выбирать не из чего».

Метод. Базовый `J` (он же `W_lm[U128]` с rmsnorm-поправкой) на той же подыборке 8 документов,
сингулярные числа — из `eigh` грамма `J Jᵗ` в fp64: это в ~12 раз быстрее пакетного SVD
(замер: 0.86 с против 10.0 с на документ) и даёт те же `σ` с точностью до округления извлечения
корня. Самопроверка обязательна: медиана `σ16/σ1` обязана сойтись со значением из таблицы E-45
(`spec_s16_over_s1_med`), иначе быстрый путь врёт и числа в скане недействительны.

Отчёт по каждому `r` от 1 до R_MAX: медианный зазор `(σ_r − σ_{r+1})/σ_1`, доля позиций с зазором
≥ 1 %, и то же для ЭНЕРГЕТИЧЕСкого зазора `(σ_r² − σ_{r+1}²)/σ_1²` — критерий «разрез можно
провести» законен только относительно той величины, по которой разрез выбирают.

Запуск:  cd experiments && ../.venv/bin/python exp45_gap_scan.py
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

L, S = E45.L, E45.S
R_MAX = 40
ANCHOR_REL = 1e-4          # медиана σ16/σ1 против строки base в таблице E-45
N_BOOT = 2000


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.set_grad_enabled(False)

    tab = os.path.join(os.path.dirname(_HERE), "results", "exp45_damage_vs_structure.json")
    if not os.path.exists(tab):
        tab = tab.replace(".json", ".partial.json")
    with open(tab) as f:
        dj = json.load(f)
    base = next(e for e in dj["edits"] if e["name"] == "base")

    ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy") else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    del ck
    spec_pos = np.concatenate([np.arange(i * S, (i + 1) * S) for i in E45.SPEC_DOCS])

    from transformers import AutoModelForCausalLM
    log("loading model (fp32) ...")
    model = AutoModelForCausalLM.from_pretrained(
        E45.MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    W = model.lm_head.weight.detach().float().clone()
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    WU0 = W[U128].numpy().astype(np.float64)
    del model
    J = E45.make_j_fn(Hst[spec_pos], g_np)(WU0)              # [T,K,D] fp64
    log(f"J базы: {J.shape}, {time.time() - t0:.0f} с")

    # σ через спектр грамма K×K (K=128 ≪ D): eigh дешёвый, порядок собственный
    s_all = []
    for i in range(0, len(J), S):
        Jd = J[i:i + S]
        Gm = torch.from_numpy(np.matmul(Jd, np.swapaxes(Jd, -1, -2)))
        ev = torch.linalg.eigvalsh(Gm).clamp_min(0.0).sqrt()   # по возрастанию
        s_all.append(ev.flip(-1).numpy())                       # [S,K] по убыванию
        del Gm, ev
    sv = np.concatenate(s_all)                                  # [T,K]
    log(f"  сингулярные числа: {sv.shape}, {time.time() - t0:.0f} с")

    s1 = sv[:, 0:1]
    chk = float(np.median(sv[:, E45.R - 1] / sv[:, 0]))
    log(f"  самопроверка: медиана σ{E45.R}/σ1 = {chk:.9f}, в таблице "
        f"{base['spec_s16_over_s1_med']:.9f} (rel {abs(chk / base['spec_s16_over_s1_med'] - 1):.2e})")
    assert abs(chk / base["spec_s16_over_s1_med"] - 1) < ANCHOR_REL, (
        "быстрый путь (eigh грамма) не сходится с пакетным SVD таблицы — скан недействителен")

    rows = []
    for r in range(1, R_MAX + 1):
        gap = (sv[:, r - 1] - sv[:, r]) / sv[:, 0]
        gape = (sv[:, r - 1] ** 2 - sv[:, r] ** 2) / sv[:, 0] ** 2
        rng = np.random.default_rng(r)
        bg = np.array([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(300)])
        rows.append(dict(r=r,
                         gap_med=float(np.median(gap)), gap_mean=float(gap.mean()),
                         frac_ge_1pct=float((gap >= 0.01).mean()),
                         frac_ge_01pct=float((gap >= 0.001).mean()),
                         gap_se_boot=float(bg.std(ddof=1)),
                         gapE_med=float(np.median(gape)),
                         fracE_ge_1pct=float((gape >= 0.01).mean()),
                         s_over_s1=float(np.median(sv[:, r - 1] / s1[:, 0]))))
        log(f"  r={r:>2d}: зазор σ медиана {rows[-1]['gap_med']:.5f} "
            f"(≥1 % на {100 * rows[-1]['frac_ge_1pct']:.1f} % позиций, ≥0.1 % на "
            f"{100 * rows[-1]['frac_ge_01pct']:.1f} %), энерг. зазор медиана "
            f"{rows[-1]['gapE_med']:.5f} (≥1 % на {100 * rows[-1]['fracE_ge_1pct']:.1f} %), "
            f"σ_r/σ_1 = {rows[-1]['s_over_s1']:.4f}")

    ok1 = [x["r"] for x in rows if x["frac_ge_1pct"] >= 0.5]
    ok01 = [x["r"] for x in rows if x["frac_ge_01pct"] >= 0.5]
    oke = [x["r"] for x in rows if x["fracE_ge_1pct"] >= 0.5]
    log(f"  r с зазором ≥1 % хотя бы на половине позиций: {ok1 if ok1 else 'НЕТ'}")
    log(f"  r с зазором ≥0.1 % хотя бы на половине позиций: {ok01 if ok01 else 'НЕТ'}")
    log(f"  r с ЭНЕРГЕТИЧЕСКИМ зазором ≥1 % на половине позиций: {oke if oke else 'НЕТ'}")

    out = dict(experiment="E-45e", L=L, r_max=R_MAX, n_positions=int(sv.shape[0]),
               n_docs=len(E45.SPEC_DOCS), selfcheck_s16_over_s1=dict(
                   scan=chk, table=base["spec_s16_over_s1_med"]),
               by_r=rows, r_ok_1pct=ok1, r_ok_01pct=ok01, r_ok_energy_1pct=oke,
               wall_sec=round(time.time() - t0, 1))
    with open(os.path.join(os.path.dirname(_HERE), "results", "exp45_gap_scan.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"→ results/exp45_gap_scan.json, wall {out['wall_sec']} с")


if __name__ == "__main__":
    sys.exit(main())
