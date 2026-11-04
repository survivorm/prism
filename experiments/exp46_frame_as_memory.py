#!/usr/bin/env python
"""E-46: лист как носитель — сколько предсказательной информации переживает проекцию на лист.

Вопрос пользователя («что нам эта структура может дать, пока не ясно»): если топ-r правые
подпространства якобиана V_r,t — не побочный продукт, а то, что остаточный поток тащит вперёд,
то скрытое состояние позиции, сжатое в лист позиции НА ШАГ РАНЬШЕ, должно сохранять предсказание
сильно лучше, чем при сжатии в любое другое подпространство той же размерности.

Формулировка по индексам (важно, тут легко съехать на шаг). Для позиции q (её состояние h_q и её
предсказание y_{q+1}) берём раму позиции f = q − lag и сравниваем

    p(·|h_q)      — истинное распределение модели,
    p(·|Π_f h_q)  — то же, но h_q ограничен рамой позиции f,  Π_f = V_f V_fᵗ.

Варианты Π (все размерности r=16, кроме mean_dir):
  leaf_lag{Δ}  Δ ∈ {0,1,2,4,8}: рама Δ шагов назад. Δ=0 — потолок «своей» рамы, Δ=1 — основной
               тест, лестница Δ — предсказание связности: если лист медленно вращающаяся рама,
               KL обязан расти с Δ (то же предсказание, что стоит за κ).
  haar         случайное r-подпространство (E28.haar_subspace) — тривиальный пол.
  pca_doc      топ-r главных направлений облака h по документу (центированного) — консервативный
               конкурент: искусственно лишён «массивного» среднего направления.
  pca_doc_raw  то же без центрирования — честный конкурент линейного метода (имеет право занять
               среднее направление); ГЛАВНЫЙ КОНКУРЕНТ: если он не хуже листа, лист — это
               периметровка главных направлений, и слово «носитель» не используется.
  pca_win      топ-r главных направлений окна [q−WIN, q−1] (без текущего h, иначе проекция
               тривиальна) — конкурент «адаптивные локальные направления без всякого якобиана».
  mean_dir     направление среднего h по документу, r=1 — контроль «а может, всё в среднем».

Метрики на вариант: KL(p_true‖p_proj), ΔNLL реализовавшегося токена (nll_proj − nll_true),
пересечение top-5, энтропия спроецированного распределения, И доля энергии ‖Πh‖²/‖h‖² плюс
парная метрика при восстановлении нормы (суффикс _nr): логиты линейны по h, поэтому α·Πh даёт
α·logits(Πh) второй перемножения не требует. Без _nr сравнение смещено в пользу вариантов,
случайно захвативших больше энергии, — вывод читается по обоим наборам чисел. Всё сохраняется попозиционно, сводки
и bootstrap — в отдельном скрипте анализа, чтобы числа нельзя было пересобрать под вывод.

Критерий зафиксирован ДО чисел (PLAN.md, п. 7): лист побеждает, если KL(leaf_lag1) ниже минимума
из {haar, pca_doc, pca_doc_raw, pca_win} с документным bootstrap-CI и лестница Δ монотонна. Если
хотя бы один PCA-конкурент не хуже — вывод «носителя нет», и он пишется так же отчётливо.

Вход: results/exp45b_leaf_table.npz (+ .json.gz за идентификаторами целей), пересоздаётся
exp45b_leaf_table.py.

Запуск:  cd experiments && ../.venv/bin/python exp46_frame_as_memory.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp28_leaf_structure as E28      # noqa: E402  (haar_subspace)

MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
TAB = "../results/exp45b_leaf_table.npz"
META = "../results/exp45b_leaf_table.json.gz"
LAGS = (0, 1, 2, 4, 8)
WIN = 16                                # окно локального PCA (позиций)
TOL = 1e-3                              # fp16-хранение V: допуск на idempotency и след


def log(*a):
    print(*a, flush=True)


def restricted(Hq: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Π h по позициям: B [S,D,r] (столбцы ортонормированы), Hq [S,D] → [S,D].

    Считаем B (Bᵗ h) — это численно устойчивее явного [D,D]-проектора и не материализует его.
    """
    c = torch.einsum("sdr,sd->sr", B, Hq)
    return torch.einsum("sdr,sr->sd", B, c)


def main():
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 8)
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM

    with gzip.open(META, "rt") as f:
        blob = json.load(f)
    meta, tab_rows = blob["meta"], blob["rows"]
    S, R, DOMS, NPD = meta["S"], meta["r_main"], meta["domains"], meta["n_per_domain"]
    # идентификатор следующего токена по (doc, t) — нужен для ΔNLL
    nxt = np.zeros((len(DOMS) * NPD, S), dtype=np.int64)
    has_t = np.zeros((len(DOMS) * NPD, S), dtype=np.float32)
    for i, row in enumerate(tab_rows):
        di = DOMS.index(row["domain"]) * NPD + row["doc"]
        nxt[di, row["t"]] = row["nxt"]
        has_t[di, row["t"]] = row["has_target"]

    z = np.load(TAB)
    V = torch.from_numpy(z["V"]).float()              # [N_doc, S, D, r]
    H = torch.from_numpy(z["H"]).float()              # [N_doc, S, D]
    n_doc, S_chk, D, r = V.shape
    assert (S_chk, r) == (S, R), f"форма npz {tuple(V.shape)} не совпала с meta {(S, R)}"
    assert n_doc == len(DOMS) * NPD, f"{n_doc} доков, ожидалось {len(DOMS) * NPD}"
    log(f"таблица E-45b: {n_doc} доков × S={S} × r={r}")

    # ---- самопроверка: V из fp16 обязан давать ортонормированные столбцы ----
    sel_d = torch.randint(0, n_doc, (64,), generator=torch.Generator().manual_seed(0))
    sel_t = torch.randint(0, S, (64,), generator=torch.Generator().manual_seed(1))
    Vs = V[sel_d, sel_t]                               # [64, D, r]
    P = Vs @ Vs.transpose(1, 2)
    idem = float((P @ P - P).abs().max())
    trdev = float((torch.diagonal(P, dim1=1, dim2=2).sum(1) - r).abs().max())
    orth = float((Vs.transpose(1, 2) @ Vs - torch.eye(r)).abs().max())
    log(f"проекторы: ‖P²−P‖max = {idem:.2e}, |tr P − r|max = {trdev:.2e}, "
        f"‖VᵗV−I‖max = {orth:.2e}")
    assert max(idem, trdev, orth) < TOL, "V из npz не ортонормирован — таблица испорчена"

    log("loading model (fp32, нужен только lm_head) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    W = model.lm_head.weight.detach().float()          # [V, D]
    del model

    rows = []
    for di in range(n_doc):
        dom = DOMS[di // NPD]
        Hd, Vd = H[di], V[di]                          # [S,D], [S,D,r]
        mu = Hd.mean(0)
        mu_hat = (mu / mu.norm().clamp_min(1e-12))[:, None]           # [D,1]
        kl_med = {}
        Hc = Hd - mu                                    # центрированное облако
        U_pca, _, _ = torch.linalg.svd(Hc, full_matrices=False)
        pca_doc = U_pca[:, :r]                          # первое направление ≠ «среднее»
        # Нецентрированный PCA — честный конкурент линейного метода: он имеет полное право
        # занять «массивное» среднее направление, а лист его не занимает по построению.
        # Отставать лист обязан от обоих, иначе вывод «носителя нет».
        U_raw, _, _ = torch.linalg.svd(Hd, full_matrices=False)
        pca_doc_raw = U_raw[:, :r]

        variants = {}
        for lag in LAGS:
            src = torch.arange(S) - lag
            ok = src >= 0
            B = torch.zeros(S, D, r)
            B[ok] = Vd[src[ok]]
            variants[f"leaf_lag{lag}"] = (B, ok)
        variants["pca_doc"] = (pca_doc.expand(S, D, r).contiguous(),
                               torch.ones(S, dtype=torch.bool))
        variants["pca_doc_raw"] = (pca_doc_raw.expand(S, D, r).contiguous(),
                                   torch.ones(S, dtype=torch.bool))
        variants["mean_dir"] = (mu_hat.expand(S, D, 1).contiguous(),
                                torch.ones(S, dtype=torch.bool))
        Vw = torch.zeros(S, D, r)
        okw = torch.zeros(S, dtype=torch.bool)
        for t in range(WIN, S):                        # окно БЕЗ текущего h
            Hw = Hd[t - WIN:t] - Hd[t - WIN:t].mean(0)
            Uw, _, _ = torch.linalg.svd(Hw, full_matrices=False)
            Vw[t] = Uw[:, :r]
            okw[t] = True
        variants["pca_win"] = (Vw, okw)
        gen = torch.Generator().manual_seed(1000 + di)
        variants["haar"] = (torch.stack([E28.haar_subspace(D, r, "cpu", generator=gen)
                                         for _ in range(S)]),
                            torch.ones(S, dtype=torch.bool))

        lp_true = torch.log_softmax(Hd @ W.T, 1)
        top_true = lp_true.topk(5, 1).indices
        ntgt = torch.from_numpy(nxt[di])[:, None]
        nll_true = -lp_true.gather(1, ntgt)[:, 0]
        h_norm2 = (Hd ** 2).sum(1)
        for name, (B, ok) in variants.items():
            hp = restricted(Hd, B)
            p_norm2 = (hp ** 2).sum(1)
            # Какая доля энергии h попала в подпространство. Без этого сравнение смещено:
            # считывание линейно по норме h, и вариант, случайно захвативший больше энергии,
            # выглядит «лучше» по KL по причине, к раме не имеющей отношения.
            energy = (p_norm2 / h_norm2).clamp(1e-12, None)
            lg_proj = hp @ W.T
            del hp
            lp = torch.log_softmax(lg_proj, 1)
            # …и нормо-восстановленный вариант: logits(α·Πh) = α·logits(Πh), то есть вторая
            # перемножение матриц не нужна — только масштабирование уже посчитанных логитов.
            lp_nr = torch.log_softmax(lg_proj * (h_norm2.sqrt() / p_norm2.sqrt()).clamp_min(
                1e-6)[:, None], 1)
            del lg_proj
            kl = (torch.exp(lp_true) * (lp_true - lp)).sum(1).clamp_min(0.0)
            kl_nr = (torch.exp(lp_true) * (lp_true - lp_nr)).sum(1).clamp_min(0.0)
            top_p = lp.topk(5, 1).indices
            inter = (top_p[:, :, None] == top_true[:, None, :]).any(2).sum(1).float() / 5.0
            nll_p = -lp.gather(1, ntgt)[:, 0]
            nll_p_nr = -lp_nr.gather(1, ntgt)[:, 0]
            ent = -(torch.exp(lp) * lp).sum(1)
            for t in range(S):
                if not bool(ok[t]):
                    continue
                rows.append(dict(domain=dom, doc=di, t=t, variant=name, r=int(B.shape[-1]),
                                 kl=float(kl[t]), top5=float(inter[t]), ent=float(ent[t]),
                                 nll_true=float(nll_true[t]), nll_proj=float(nll_p[t]),
                                 kl_nr=float(kl_nr[t]), nll_proj_nr=float(nll_p_nr[t]),
                                 energy=float(energy[t]),
                                 has_target=float(has_t[di, t])))
            kl_med[name] = float(kl[ok].median())
            del lp, lp_nr, kl, kl_nr, top_p, inter, nll_p, nll_p_nr, ent, energy
        order = [f"leaf_lag{l}" for l in LAGS] + ["pca_doc", "pca_doc_raw", "pca_win",
                                                  "haar", "mean_dir"]
        log(f"  док {di} ({dom}) готов, медианный KL: "
            + "  ".join(f"{k}={kl_med[k]:.2e}" for k in order)
            + f"   t={time.time() - t0:.0f} с")

    out = dict(experiment="E-46",
               meta_meta={k: meta[k] for k in ("S", "r_main", "domains", "n_per_domain",
                                               "ppl_table", "kappa_median_r16")},
               lags=list(LAGS), win=WIN,
               projector_check=dict(idem=idem, trdev=trdev, orth=orth),
               rows=rows, wall_sec=time.time() - t0)
    with gzip.open("../results/exp46_frame_as_memory.json.gz", "wt") as f:
        json.dump(out, f, ensure_ascii=False)
    log(f"готово за {time.time() - t0:.0f} с → results/exp46_frame_as_memory.json.gz")


if __name__ == "__main__":
    main()
