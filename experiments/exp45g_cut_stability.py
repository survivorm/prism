#!/usr/bin/env python
"""E-45g: что вообще значит «зазор на r мал» — измеряем ПЛАВУЧЕСТЬ разреза, а не порог на глаз.

Провоцирующий факт. В E-45b доля позиций с `gap_r16 < 0.01` вышла 99.7 % (медиана зазора 0.0087,
процентиля 0.0078…0.0097 — то есть зазор не «у части позиций плохой», а весь распределён вокруг
0.009). При таком самопроверочном пороге анализ пуст: все корреляции `n = 0`. Значит порог 0.01
в пререгистрации был не измерением, а догадкой, и на нём нельзя ни «валидировать» разрез, ни
«опровергать» его.

Что вместо порога. Разрез r информативен ровно настолько, насколько правое подпространство `V_r`
непрерывно по возмущению `J`: если собственная рамка плавает при шуме `ε‖J‖`, то и `κ(r)`, и
`Δ_SVD(r)` — числа над шумом, а не над листом. Меряем напрямую:

  * `P(σ16 < σ17)` — доля позиций, где порядок на разрезе МЕНЯЕТСЯ при шуме `ε`;
  * `max главный угол` между `V16(J)` и `V16(J')` там, где порядок сохранён;
  * `Δκ(r=16)` от того же шума.

Сетка `ε` захватывает и машинный пол (fp32 ≈ 1.2e-07), и `document_profile`, который кастует `J`
в fp32 внутри, и фактические дрейфы правок таблицы E-45 (0.06 / 0.12 / 0.25). Последний столбец —
ответ на вопрос «сравнимы ли κ по строкам таблицы»: если при `ε = 0.12` рамка уже переставлена,
то `κ` разных правок — числа о РАЗНЫХ подпространствах, и сравнивать их можно только с этой
оговоркой.

J — аналитический readout-якобиан слоя 23 (`E45.make_j_fn`), позиции — кэш exp27b (8 документов,
как в колонке κ таблицы; распределение зазора на свежем тексте E-45b и на кэше совпало: 0.0087
против 0.0088, поэтому замер плавучести на кэше законен).

Запуск:  cd experiments && ../.venv/bin/python exp45g_cut_stability.py    (~5 минут CPU)
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

import exp28_leaf_structure as E28                       # noqa: E402
import exp45_damage_vs_structure as E45                  # noqa: E402
from _e38_cache import jcache_path                       # noqa: E402

S, R, D, K_TOK = E45.S, E45.R, 1024, E45.K_TOK
SPEC_DOCS = E45.SPEC_DOCS
SP_POS = np.concatenate([np.arange(i * S, (i + 1) * S) for i in SPEC_DOCS])
EPS = (0.0, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 0.06, 0.12, 0.25)
CHUNK = S    # ОДИН документ на вызов document_profile: он сравнивает позиции t и t+1 внутри
             # переданного стека, поэтому чанк, кратный 2·S, добавляет фантомную пару через
             # границу документов (первый прогон дал −0.16 % при ε=0 вместо нуля именно так)
OUT = os.path.join(_HERE, "..", "results", "exp45g_cut_stability.json")
log = E45.log


def sv_and_V(J: np.ndarray):
    """σ [T,K] по убыв. и V_r [T,D,r] — правые сингулярные векторы (fp64 → fp32 внутри SVD)."""
    U, s, Vh = torch.linalg.svd(torch.from_numpy(J), full_matrices=False)
    return s.numpy(), Vh[:, :R].transpose(1, 2).contiguous().numpy()


def max_angle(A, B):
    """Максимальный главный угол между [T,D,r] × [T,D,r] (по позициям, максимум по T)."""
    M = np.einsum("tdr,tde->tre", A, B)
    sv = np.linalg.svd(M, compute_uv=False)
    return float(np.acos(np.clip(sv.max(axis=1), -1.0, 1.0)).max())


def main():
    t0 = time.time()
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    log("loading jcache L23 ...")
    ck = torch.load(jcache_path(E45.L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy") else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    del ck
    Hs = Hst[SP_POS]
    log("loading model (fp32) ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        E45.MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    torch.set_grad_enabled(False)
    W = model.lm_head.weight.detach().float().clone()
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    WU0 = W[U128].numpy().astype(np.float64)
    del model

    J_of = E45.make_j_fn(Hs, g_np)
    J0 = J_of(WU0)                                   # [1024,K,D] fp64 = 1.07 ГБ
    nrm = np.linalg.norm(J0.reshape(len(J0), -1), axis=1)[:, None, None]
    s0, V0 = sv_and_V(J0)
    gap0 = (s0[:, R - 1] - s0[:, R]) / s0[:, 0]
    log(f"зазор на r={R}: медиана {np.median(gap0):.5f}, процентиля 1/99 "
        f"{np.percentile(gap0, 1):.5f}/{np.percentile(gap0, 99):.5f} "
        f"(порог валидности E-45b был 0.01 → невалидно {np.mean(gap0 < 0.01) * 100:.1f} %)")
    kap0 = np.concatenate([   # по документу, см. комментарий у CHUNK
        np.asarray(E28.document_profile(torch.from_numpy(J0[i:i + S].astype(np.float32)),
                                       [1, R, 32], r_main=R)[2], dtype=np.float64)
        for i in range(0, len(J0), S)])
    kap0 = float(np.nanmedian(kap0[kap0 == kap0]))
    log(f"κ базы = {kap0:.6f}")

    rng = np.random.default_rng(0)
    rows = []
    for eps in EPS:
        t1 = time.time()
        swap = 0
        ang = []
        kap = []
        for c0 in range(0, len(J0), CHUNK):
            c1 = min(c0 + CHUNK, len(J0))
            Jp = J0[c0:c1] if eps == 0.0 else J0[c0:c1] + eps * nrm[c0:c1] * rng.standard_normal(
                J0[c0:c1].shape)
            sp, Vp = sv_and_V(Jp)
            swap += int((sp[:, R - 1] < sp[:, R]).sum())
            keep = sp[:, R - 1] >= sp[:, R]
            if keep.any():
                ang.append(max_angle(V0[c0:c1][keep], Vp[keep]))
            k2 = E28.document_profile(torch.from_numpy(Jp.astype(np.float32)), [1, R, 32],
                                      r_main=R)[2]
            kap.append(np.asarray(k2, dtype=np.float64))
            del Jp, sp, Vp
        kap = np.concatenate([k[~np.isnan(k)] for k in kap])
        row = dict(eps=eps, frac_swap_order=swap / len(J0),
                   max_angle_rad=max(ang) if ang else float("nan"),
                   max_angle_deg=float(np.degrees(max(ang))) if ang else float("nan"),
                   kappa_med=float(np.nanmedian(kap)),
                   kappa_rel_dev=float(np.nanmedian(kap) / kap0 - 1.0),
                   n_defined=int(len(kap)))
        rows.append(row)
        log(f"  ε={eps:<7g}: порядок на разрезе меняется у {row['frac_swap_order'] * 100:5.1f} % "
            f"позиций, max угол V₁₆ = {row['max_angle_deg']:6.2f}°, "
            f"κ = {row['kappa_med']:.6f} ({row['kappa_rel_dev'] * 100:+.2f} %) "
            f"[{time.time() - t1:.0f} с]")

    eps32 = 1.2e-7
    fp32_row = min(rows[1:], key=lambda r: abs(r["eps"] - eps32))
    verdict = dict(
        gap_med=float(np.median(gap0)),
        gap_pct={q: float(np.percentile(gap0, q)) for q in (1, 10, 50, 90, 99)},
        frac_below_0p01=float(np.mean(gap0 < 0.01)),
        eps_swap_onset=float(next((r["eps"] for r in rows if r["frac_swap_order"] > 0.01),
                                 float("inf"))),
        eps_angle_1deg=float(next((r["eps"] for r in rows if r["max_angle_deg"] > 1.0),
                                 float("inf"))),
        fp32_level=fp32_row["eps"], fp32_angle_deg=fp32_row["max_angle_deg"],
        kappa_drift_at_table_drifts={str(e): next(r["kappa_rel_dev"] for r in rows
                                                  if r["eps"] == e) for e in (0.06, 0.12, 0.25)},
        # ЧИТАТЬ ОСТОРОЖНО. Этот шум — возмущение САМОГО J, изотропное по [K,D], то есть
        # ВНЕ образа отображения W_U → J. Правки таблицы (квантизация, iid по строкам U128,
        # уплощение) живут ВНУТРИ образа: ΔJ = J(ΔW) структурирован (та же проекция на ḣ_t,
        # тот же множитель g). Поэтому ε-скан — верхняя оценка чувствительности к
        # произвольному возмущению J, а не «что сделала бы с κ правка такого же дрейфа»:
        # у rand_K_d0.25 (тоже iid, но по строкам W) κ = ×1.05, а изотропный шум при ε = 0.25
        # даёт ×342.
        out_of_image_note=(
            "шум вне образа W_U→J: ε=1e-3 вращает V₁₆ на несколько градусов и поднимает κ на "
            "порядок, ε≥0.01 насыщает κ на ~5.4; при этом порядок σ₁₆>σ₁₇ НЕ ломается даже "
            "при ε=0.25, то есть «зазор мал» ≠ «разрез переставляется»"),
        # контроль детерминизма: при ε ≤ fp32-пола рамка и κ обязаны не двигаться вовсе
        kappa_reproducible_at_fp32=bool(fp32_row["max_angle_deg"] < 0.1
                                        and abs(fp32_row["kappa_rel_dev"]) < 1e-3))
    log("\n== вердикт ==")
    log(f"  порог «gap ≥ 0.01» отвергает {verdict['frac_below_0p01'] * 100:.1f} % позиций при "
        f"машинном поле ε≈{eps32:.1e}: это не критерий, а произвол")
    log(f"  порядок σ₁₆>σ₁₇ начинает ломаться с ε ≈ {verdict['eps_swap_onset']:g}; "
        f"угол V₁₆ > 1° с ε ≈ {verdict['eps_angle_1deg']:g}")
    log(f"  при fp32-поле ({fp32_row['eps']:g}) угол = {fp32_row['max_angle_deg']:.2f}°")
    log(f"  сдвиг κ на дрейфах таблицы: {verdict['kappa_drift_at_table_drifts']} — но это"
        f" шум ВНЕ образа W_U→J; у iid-правок таблицы того же дрейфа κ = ×1.05")
    assert verdict["kappa_reproducible_at_fp32"], (
        f"при fp32-поле рамка (угол {fp32_row['max_angle_deg']:.3f}°) или κ "
        f"({fp32_row['kappa_rel_dev']:.1e}) шевелятся — тогда по-position κ невоспроизводим,"
        f" а порог валидности надо ставить по этому полу, а не по 0.01")
    json.dump(dict(experiment="E-45g", L=E45.L, r=R, spec_docs=list(SPEC_DOCS),
                   n_pos=int(len(SP_POS)), kappa_base=kap0, rows=rows, verdict=verdict,
                   wall_sec=round(time.time() - t0, 1)),
              open(OUT, "w"), ensure_ascii=False, indent=1)
    log(f"→ {OUT}, wall {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
