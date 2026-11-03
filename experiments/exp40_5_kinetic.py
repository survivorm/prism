#!/usr/bin/env python3
"""E-40.5 — шестиканальная кинетическая декомпозиция (docs/e40_todo.md §7).

Кэш exp27b, пары соседних позиций (Δ=1), r=16, транспортная рецептура §7.1
целиком в prism.transport/kinetic. Каналы — энергии. Точное замыкание — 5×5
Gram channel_energy_matrix = ‖ΔJ‖² (tol 1e-10, ассерт на каждой паре).
Ожидаем (прereg): L6–L22 linearization_valid=False (отчёт только E_radial +
E_off_manifold), L23 valid + rotation_unique=False; E_off/‖ΔJ‖² ≈ 0.5 на L6–L22.

(5e) левый порог: нулевой ансамбль — Haar-конъюгация фактической M на подвыборке
64 пар, B=1000 общих Haar-матриц (перепозиционный MC запрещён бюджетом — D-нота).

Запуск: .venv/bin/python experiments/exp40_5_kinetic.py   (десятки минут, CPU).
"""
from __future__ import annotations

# BLAS-треды пинятся ДО импорта numpy/torch: на матрицах [128,1024] 12 тредов
# МЕДЛЕННЕЕ одного в 30 раз (замер 2026-11-03: SVD 38 мс @1 тред, 142 мс @4,
# 1162 мс @12), а два таких процесса на 12 ядрах кладут машину (load average
# 57, прогоны стояли 3 ч 25 мин). Числа от числа тредов не зависят.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from _e38_cache import load_layer  # noqa: E402
from exp40_common import DOMAINS, N_PER_DOMAIN, S, LAYERS, dump_json, header, sentinel_mask, tag  # noqa: E402
from prism.kinetic import channel_energy_matrix, kinetic_decomposition  # noqa: E402
from prism.p_algebra import P_matrix  # noqa: E402

ROOT = os.path.join(HERE, "..")
R_TOP = 16
B_MC = 1000
N_PAIRS_NULL = 64
GRAM_TOL = 1e-10


def haar_ensemble(K: int, B: int, seed: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((B, K, K))
    for b in range(B):
        q, _ = np.linalg.qr(rng.standard_normal((K, K)))
        out[b] = q
    return out


def left_channel_null(M: np.ndarray, Sb: np.ndarray, Os: np.ndarray) -> np.ndarray:
    """E_left^null_b = ‖O_b M O_bᵀ Σ̄‖² при сохранённом спектре M (B значений)."""
    vals = np.empty(Os.shape[0])
    for b in range(Os.shape[0]):
        Mp = Os[b] @ M @ Os[b].T          # спектр M сохранён, база случайная
        vals[b] = float(np.square(Mp @ Sb).sum())
    return vals


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    per_layer: dict[str, dict] = {}
    per_doc_s05: dict[str, dict[str, float]] = {}   # {layer: {doc: radial_share}}
    dev_extra = {}
    for L in LAYERS:
        tl = time.time()
        ck = load_layer(L, want=("J",))
        Jf = ck["J"]                                  # [N,S,K,D] float32
        sent = sentinel_mask(L)
        n_pairs = 0
        ch_sum = dict(E_radial=0.0, E_right=0.0, E_left=0.0, E_cross=0.0,
                      E_cross_abs=0.0, E_resid=0.0, E_off_manifold=0.0,
                      norm_dJ_sq=0.0, eps3_correction=0.0)
        # ‖ΔJ‖² только по парам с вычисленным транспортом: `eps3_correction`
        # определён на них же, поэтому его знаменатель — эта сумма, а не E_tot
        # (иначе «≤ 1e-2» из чек-листа §12.8 измерялось бы долями валидных пар).
        dJ_sq_valid = 0.0
        pairs_skipped = 0
        flags_count = dict(linearization_valid=0, rotation_unique=0,
                           order_cross_U=0, order_cross_V=0, unstable_transport=0)
        gram_worst = 0.0
        gram_n = 0
        pairs_doc: dict[str, list] = {}
        null_pairs = []                               # (M, Σ̄, E_left, E_right)
        ch_mat_rows = []                              # энергии пар для 5×5 анализа
        for d_of, dom in enumerate(DOMAINS):
            for si in range(N_PER_DOMAIN):
                di = d_of * N_PER_DOMAIN + si   # плоский индекс документа
                # (кэш лежит домен-мажорно: (wiki,0..7),(math,0..7),…); раньше
                # сюда попадал индекс ДОМЕНА 0..3, и все 32 «документа» были
                # четырьмя первыми wiki-документами, повторёнными по 8 раз
                doc = f"{dom}#{si}"
                rad_num, den = 0.0, 0.0
                # SVD позиции считается ОДИН раз: каждая J участвует в двух
                # соседних парах, и обе функции ядра считали бы его заново
                # (4 полных SVD на пару вместо 1). Побитово то же: вход тот же
                # fp64-массив, тот же np.linalg.svd(full_matrices=False).
                svs: dict[int, tuple] = {}

                def svd_at(t: int) -> tuple:
                    got = svs.get(t)
                    if got is None:
                        got = np.linalg.svd(Jf[di, t].double().numpy(),
                                            full_matrices=False)
                        svs[t] = got
                        if len(svs) > 8:              # скользящее окно памяти
                            svs.pop(min(svs))
                    return got
                for t in range(S - 1):
                    if bool(sent[di, t] and sent[di, t + 1]):
                        pairs_skipped += 1
                        continue                      # обе sentinel — пара вне игры
                    Jt = Jf[di, t].double().numpy()
                    Jt1 = Jf[di, t + 1].double().numpy()
                    pair = (svd_at(t), svd_at(t + 1))
                    kd = kinetic_decomposition(Jt, Jt1, r=R_TOP, svds=pair)
                    dJ = Jt1 - Jt
                    norm_dJ_sq = float((dJ ** 2).sum())
                    try:
                        # на линеаризационно-невалидных парах transport_log законно
                        # отказывает (главная ветвь) — Gram только для вычислимых пар
                        G = channel_energy_matrix(Jt, Jt1, r=R_TOP, svds=pair)
                    except ValueError:
                        G = None
                    if G is not None:
                        gram = abs(G.sum() - norm_dJ_sq) / max(norm_dJ_sq, 1e-300)
                        gram_worst = max(gram_worst, gram)
                        gram_n += 1
                        assert gram < GRAM_TOL, \
                            f"5×5 Gram не замкнут на L{L} {doc} t{t}: {gram:.2e}"
                    for k in ch_sum:
                        v = kd[k] if k != "E_cross_abs" else kd["E_cross_abs"]
                        if v is not None:
                            ch_sum[k] += v
                    for fl in flags_count:
                        flags_count[fl] += bool(kd["flags"][fl])
                    n_pairs += 1
                    er, eo = kd["E_radial"], kd["E_off_manifold"]
                    rad_num += er
                    den += er + eo
                    if kd["E_left"] is not None:
                        dJ_sq_valid += float(kd["norm_dJ_sq"])
                        # 5 энергий диагонали Gram (radial,left,right,resid,off);
                        # cross — внедиагональный элемент, у него свой пункт (5d)
                        ch_mat_rows.append([kd["E_radial"], kd["E_left"], kd["E_right"],
                                            kd["E_resid"], eo])
                    if len(null_pairs) < N_PAIRS_NULL and kd["E_left"] is not None:
                        # нулевой ансамбль (5e) — Haar-конъюгация ФАКТИЧЕСКОЙ M
                        # при фактическом Σ̄: обе величины берутся из
                        # разложения. Прежняя пересборка здесь же не работала
                        # (align_signs_and_columns возвращает 3 значения, а не 2
                        # — блок падал на первом слое с вычисленным E_left) и
                        # теряла перестановку σ (урок M2).
                        null_pairs.append((kd["M_transport"], kd["sigma_bar"],
                                           kd["E_left"], kd["E_right"]))
                pairs_doc[doc] = (rad_num, den)
        per_doc_s05[str(L)] = {d: a / b for d, (a, b) in pairs_doc.items()}
        E_tot = ch_sum["norm_dJ_sq"]
        lin = flags_count["linearization_valid"]
        reportable = lin == n_pairs

        # (5e) левый порог на слое, если левый канал вообще вычислен
        left_block = None
        if null_pairs:
            Os = haar_ensemble(R_TOP, B_MC)
            q95s, ratios = [], []
            for (Mx, Sb, e_left, e_right) in null_pairs:
                vals = left_channel_null(Mx, Sb, Os)
                q95 = float(np.quantile(vals, 0.95))
                below = e_left < max(0.05 * e_right, q95)
                q95s.append(q95)
                ratios.append(below)
            left_block = {"n_sampled_pairs": len(null_pairs), "B": B_MC,
                          "unstable_left_frac_sampled": float(np.mean(ratios)),
                          "note": "MC на подвыборке пар (перепозиционный запрещён "
                                  "бюджетом); ансамбль — Haar-конъюгация фактической M"}

        # 5×5 по энергиям: ранг против перестановочного нуля + MP + PR + ρ̂
        rank_block = None
        if ch_mat_rows and reportable:
            X = np.asarray(ch_mat_rows)
            Xs = (X - X.mean(0)) / X.std(0)
            C = np.corrcoef(Xs.T)
            k = Xs.shape[1]
            ev = np.linalg.eigvalsh(C)[::-1]
            rng = np.random.default_rng(40)
            ev_null = []
            for _ in range(200):
                Y = np.column_stack([rng.permutation(Xs[:, j]) for j in range(k)])
                ev_null.append(np.linalg.eigvalsh(np.corrcoef(Y.T))[::-1])
            thr = np.quantile(np.asarray(ev_null), 0.95, axis=0)
            rho_hat = float((np.abs(C) - np.eye(k)).sum() / (k * (k - 1)))
            rank_block = {"rank": int((ev > thr).sum()),
                          "null_eig_p95": thr.tolist(),
                          "PR": float(ev.sum() ** 2 / (ev ** 2).sum()),
                          "rho_hat": rho_hat, "eigenvalues": ev.tolist(),
                          "B_perm": 200}

        per_layer[str(L)] = {
            "n_pairs": n_pairs,
            "sent_pair_excluded_frac": pairs_skipped / (n_pairs + pairs_skipped),
            "channels_sums": ch_sum,
            "E_off_over_dJ": ch_sum["E_off_manifold"] / E_tot,
            "E_radial_over_dJ": ch_sum["E_radial"] / E_tot,
            # чек-лист §12.8: eps3/‖ΔJ‖² ≤ 1e-2 на линеаризационно-валидных слоях
            "eps3_over_dJ_valid": (ch_sum["eps3_correction"] / dJ_sq_valid
                                   if dJ_sq_valid else None),
            "E_cross_signed_sum": ch_sum["E_cross"], "E_cross_abs_sum": ch_sum["E_cross_abs"],
            "flags_frac": {k: v / n_pairs for k, v in flags_count.items()},
            "linearization_valid_all": reportable,
            "gram_pairs_computed": gram_n,
            "gram_closure_worst_rel": gram_worst if gram_n else None,
            "rank_block_energy_5x5": rank_block,
            "left_channel_threshold": left_block,
            "wall_sec": round(time.time() - tl, 1),
        }
        print(f"L{L}: pairs={n_pairs} lin_valid={lin} E_off/‖ΔJ‖²="
              f"{ch_sum['E_off_manifold']/E_tot:.3f} gram_worst={gram_worst:.1e} "
              f"({time.time() - tl:.0f} с)", flush=True)
        assert flags_count["unstable_transport"] / n_pairs <= 0.05, \
            f"unstable_transport > 5% на L{L} — стоп по §14.4"
        assert gram_worst < GRAM_TOL
        if reportable:
            e3 = per_layer[str(L)]["eps3_over_dJ_valid"]
            assert e3 is not None and abs(e3) <= 1e-2, \
                f"L{L}: eps3/‖ΔJ‖²={e3} > 1e-2 на валидном слое — линеаризация не работает"
        del ck, Jf

    off_ok = {L: per_layer[L]["E_off_over_dJ"] for L in map(str, LAYERS)}
    # каналы — суммы энергий по парам / Σ‖ΔJ‖² = отношение сумм (ratio_of_means
    # по пер-позиционным долям тождественен ему; ядро его отвергает — Q3 audit4)
    tag_list = [tag(slice="gf", p_slice="p_t", aggregation="ratio_of_means",
                    window="full", side="mixed", norm_type="energy",
                    layer=None, rank=R_TOP)]
    rep = header("E-40.5", "docs/e40_todo.md §7",
                 {"exp27b_jcache_L6": "/data/prism_kernels/exp27b/results/exp27b_jcache_L6.pt"},
                 channel_tags=tag_list,
                 extra={"per_layer": per_layer,
                        "per_doc_S05_radial_share": per_doc_s05,
                        "pair_rule": "Δ=1 соседние кэш-позиции; пара исключена лишь если "
                                     "ОБЕ позиции sentinel (иначе остаётся — срез all)",
                        "pre_registered_off_manifold": {
                            "expected": "≈0.5 на L6–L22", "observed": off_ok,
                            "confirmed_L6_L22": all(v >= 0.3 for k, v in off_ok.items()
                                                    if k != "23")},
                        "deviations_from_spec": [
                            {"id": "D5g_gram_transport_defined_only",
                             "description": "5×5 Gram = ‖ΔJ‖² проверяется только на парах, "
                                            "где transport_log определим (главная ветвь); "
                                            "на L6–L22 это пустое множество по пререг. "
                                            "невалидности линеаризации — «все слои» из ТЗ "
                                            "физически недостижимо"},
                            {"id": "D5e_null_subsample",
                             "description": "Q_95(E_left^null) по B=1000 общим Haar-"
                                            "конъюгациям на 64 парах (перепозиционный "
                                            "MC ~1e5 GFLOP/позицию нереализуем)"},
                            {"id": "D5x_rank_L23_only",
                             "description": "5×5 корреляционный ранг считается лишь на "
                                            "слоях со всеми валидными каналами (L23)"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_5_kinetic.json")
    dump_json(rep, out)
    print(f"wrote {out}")
    return rep


if __name__ == "__main__":
    main()
