#!/usr/bin/env python3
"""E-40.1 — T4a: разрешение E6 сеткой 8 ячеек + g_uni-чекпоинт + identity-asserts (a3).

docs/e40_todo.md §3.2/§3.1(a3). Тождество frac_K = 1 − 2c + s² (обратная форма
c = (1 + s² − frac_K)/2, патч v2) позипозиционное при общем знаменателе ‖J_t‖²;
линейные агрегации (ratio_of_means, mean c s²=mean O1_t) обязаны замыкаться,
median — нет (негативный контроль). p_0-ячейки: P(p̄) с p̄ = средним p по позициям.

g_uni-чекпоинт по всем 5 слоям против ДВУХ закоммиченных артефактов
(exp39_supp_three_objects.json и exp39k_gauge_aware_importance.json).
T4b: смешанная опубликованная тройка (c — rom дополнения, frac_K — медиана E-37)
⇒ s²_implied; ожидаем красным, знак фиксируем как есть (assert «все < 0» из
патча v2 не навязываем — docs/E40_math_core_answers.md §8).

Запуск: .venv/bin/python experiments/exp40_1_t4a.py   (минуты, ноль форвардов).
"""
from __future__ import annotations

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
from exp40_common import LAYERS, dump_json, header, sentinel_mask, tag  # noqa: E402
from prism.p_algebra import P_matrix  # noqa: E402

ROOT = os.path.join(HERE, "..")
SUPP = os.path.join(ROOT, "results", "exp39_supp_three_objects.json")
E39K = os.path.join(ROOT, "results", "exp39k_gauge_aware_importance.json")
CLOSED_TOL = 1e-12


def triple_from_chunk(J: torch.Tensor, p: torch.Tensor) -> dict:
    """Поозиционные c_t, s²_t, frac_K,t для пачки J [B,K,D] и p [B,K] (fp64)."""
    m = torch.einsum("bk,bkd->bd", p, J)                    # p-взвешенное среднее
    d = J - m[:, None, :]
    pd = p[..., None] * d                                   # (P(p)J)_i
    a = pd.pow(2).sum(dim=(-1, -2))                         # ‖PJ‖²  (O3)
    cross = torch.einsum("bk,bkd,bkd->b", p, J, d)          # ⟨J, PJ⟩
    e = J.pow(2).sum(dim=(-1, -2))
    K = J - pd
    k2_direct = K.pow(2).sum(dim=(-1, -2))
    k2 = e - 2 * cross + a
    # тождество ‖K‖² = ‖J‖² − 2⟨J,PJ⟩ + ‖PJ‖² против прямого вычитания
    err = float(((k2 - k2_direct).abs() / e).max())
    assert err < 1e-9, f"тождество K не замкнулось: rel {err:.3e}"
    gf1 = pd.sum(dim=1)                                     # (P(p)J)ᵀ1 = Σ_i p_i d_i
    gf1_rel = float((gf1.norm(dim=-1) / J.sum(dim=1).norm(dim=-1).clamp_min(1e-30)).max())
    return {"c_t": cross / e, "s2_t": a / e, "frac_t": k2 / e,
            "a": a, "cross": cross, "k2": k2, "e": e,
            "gf1_rel_max": gf1_rel}


def main(results_dir: str = "results") -> dict:
    t0 = time.time()
    supp = json.load(open(SUPP))
    e39k = json.load(open(E39K))

    per_layer: dict[str, dict] = {}
    a3 = {"O2_eq_O3_rel": {}, "g_uni_gf_zero_rel": {}, "iso_inline_eq_closed_rel": {}}
    for L in LAYERS:
        ck = load_layer(L, want=("J", "P"))
        J_all = ck["J"].double().reshape(-1, *ck["J"].shape[-2:])   # [n_pos,K,D]
        P_all = ck["P"].double().reshape(-1, ck["P"].shape[-1])     # [n_pos,K]
        # Σp=1 в кэше — fp32-точность (±1e-8); ренормируем в fp64, иначе точное
        # тождество Σ_i p_i d_i ≡ 0 упирается в округление исходной нормировки
        P_all = P_all / P_all.sum(dim=-1, keepdim=True)
        sent = sentinel_mask(L).reshape(-1)
        alive = ~sent
        n_pos, K, D = J_all.shape
        p_bar = P_all.mean(dim=0)                                   # p_0 = фикс. среднее p
        # (a3.3) iso inline vs closed form — p не зависит от слоя, проверка на месте
        Pm = torch.as_tensor(P_matrix(p_bar)).double()
        iso_inline = float(Pm.pow(2).sum() / K)
        s2p = float(p_bar.pow(2).sum()); s3p = float(p_bar.pow(3).sum())
        iso_closed = (s2p - 2 * s3p + s2p ** 2) / K        # closed form — это ‖P‖²_F
        a3["iso_inline_eq_closed_rel"][str(L)] = abs(iso_inline - iso_closed) / iso_closed
        assert a3["iso_inline_eq_closed_rel"][str(L)] < 1e-9, "iso inline != closed"

        acc = {pv: {sl: {"a": 0.0, "cross": 0.0, "k2": 0.0, "e": 0.0, "n": 0}
                    for sl in ("all", "alive")} for pv in ("p_t", "p_0")}
        ratios: dict[str, dict[str, list]] = {pv: {k: [] for k in ("c_t", "s2_t", "frac_t")}
                                              for pv in ("p_t", "p_0")}
        gf_rel_max = 0.0
        g_num = 0.0      # Σ_t ‖J_tᵀ 1‖²  (1ᵀG1 для G = Σ_t J_t J_tᵀ, не квадрат суммы)
        E_tot = 0.0
        CH = 64
        for i0 in range(0, n_pos, CH):
            Jc = J_all[i0:i0 + CH]
            Pc = P_all[i0:i0 + CH]
            sc = sent[i0:i0 + CH]
            for pv, p in (("p_t", Pc), ("p_0", p_bar.expand_as(Pc))):
                r = triple_from_chunk(Jc, p.clone())
                gf_rel_max = max(gf_rel_max, r["gf1_rel_max"])
                for k in ("c_t", "s2_t", "frac_t"):
                    ratios[pv][k].append(r[k].numpy())
                if pv == "p_t":
                    g_num += float(Jc.sum(dim=1).pow(2).sum())     # Σ_t ‖Σ_i J_i‖²
                    E_tot += float(r["e"].sum())
                for sl, mask in (("all", np.ones(len(Jc), bool)), ("alive", ~sc.numpy())):
                    A = acc[pv][sl]
                    for f in ("a", "cross", "k2", "e"):
                        A[f] += float(r[f][mask].sum())
                    A["n"] += int(mask.sum())
        a3["g_uni_gf_zero_rel"][str(L)] = gf_rel_max
        assert gf_rel_max < 1e-12, f"g_uni(J^gf) не ноль: {gf_rel_max:.3e} на L{L}"

        # (a3.1) O2 ≡ O3: явный P-матричный умножитель против построчной формы
        o23 = []
        for pos in torch.linspace(0, n_pos - 1, 8).long():
            Jp = J_all[pos]; pp = P_all[pos]
            o3 = float((torch.as_tensor(P_matrix(pp)).double() @ Jp).pow(2).sum())
            o2 = float(triple_from_chunk(Jp[None], pp[None])["a"][0])
            o23.append(abs(o2 - o3) / max(o3, 1e-300))
        a3["O2_eq_O3_rel"][str(L)] = max(o23)
        assert max(o23) < 1e-12, f"O2 != O3 на L{L}: {max(o23):.3e}"

        g_uni_recomputed = g_num / (K * E_tot)
        ratios = {pv: {k: np.concatenate(v) for k, v in d.items()}
                  for pv, d in ratios.items()}

        cells = {}
        for pv in ("p_t", "p_0"):
            for sl in ("all", "alive"):
                A = acc[pv][sl]
                c_rom = A["cross"] / A["e"]; s2_rom = A["a"] / A["e"]
                fr_rom = A["k2"] / A["e"]
                res_rom = fr_rom - (1 - 2 * c_rom + s2_rom)
                rt = ratios[pv]
                # mean по той же маске, что и rom-накопление
                sel = (np.ones(len(rt["c_t"]), bool) if sl == "all"
                       else (~sent).numpy())
                c_m = float(rt["c_t"][sel].mean()); s2_m = float(rt["s2_t"][sel].mean())
                fr_m = float(rt["frac_t"][sel].mean())
                res_mean = fr_m - (1 - 2 * c_m + s2_m)
                cells[f"{pv}_ratio_of_means_{sl}"] = {
                    "c": c_rom, "s": s2_rom ** 0.5, "frac_K": fr_rom,
                    "residual": res_rom, "closed": abs(res_rom) < CLOSED_TOL}
                cells[f"{pv}_mean_{sl}"] = {
                    "c": c_m, "s": s2_m ** 0.5, "frac_K": fr_m,
                    "residual": res_mean, "closed": abs(res_mean) < CLOSED_TOL,
                    "s2_definition": "mean_t(O1_t)"}
                c_md = float(np.median(rt["c_t"][sel])); s2_md = float(np.median(rt["s2_t"][sel]))
                fr_md = float(np.median(rt["frac_t"][sel]))
                res_med = fr_md - (1 - 2 * c_md + s2_md)
                cells[f"{pv}_median_{sl}"] = {
                    "c": c_md, "s": s2_md ** 0.5, "frac_K": fr_md,
                    "residual": res_med, "closed": abs(res_med) < CLOSED_TOL,
                    "reason": None if abs(res_med) < CLOSED_TOL
                    else "median ≠ ratio of sums (ожидаемый негативный контроль)"}
        pos_cells = [k for k in cells if "median" not in k]
        assert all(cells[k]["closed"] for k in pos_cells), \
            f"L{L}: линейная ячейка не замкнулась: " + \
            str({k: cells[k]["residual"] for k in pos_cells if not cells[k]["closed"]})
        neg = [k for k in cells if "median" in k]
        assert any(not cells[k]["closed"] for k in neg), \
            f"L{L}: негативный контроль median замкнулся — сломался контроль"
        per_layer[str(L)] = {
            "cells": cells,
            "g_uni_recomputed_corpus": g_uni_recomputed,
            "g_uni_cached_supp": supp["layers"][str(L)]["g_uni_corpus"],
            "g_uni_cached_e39k": (e39k["layers_diag"].get(str(L), {}).get(
                "g_uni_uniform_energy_share")),
        }
        per_layer[str(L)]["g_uni_matches"] = bool(
            abs(g_uni_recomputed - supp["layers"][str(L)]["g_uni_corpus"]) < 1e-6)
        print(f"L{L}: cells closed, g_uni {g_uni_recomputed:.6f} vs "
              f"{supp['layers'][str(L)]['g_uni_corpus']:.6f}, gf1_rel {gf_rel_max:.2e}")
        del ck, J_all, P_all

    # g_uni-чекпоинт по 5 слоям — суммарный вердикт
    g_ok = all(v["g_uni_matches"] for v in per_layer.values())
    e39k_over = {L: (per_layer[L]["g_uni_cached_e39k"] is not None
                     and abs(per_layer[L]["g_uni_recomputed_corpus"]
                             - per_layer[L]["g_uni_cached_e39k"]) < 1e-6)
                 for L in ("18", "23")}

    # T4b: опубликованная смешанная тройка (c — rom дополнения alive, frac — медиана E-37)
    s2_implied = {}
    for L in LAYERS:
        kb = supp["layers"][str(L)]["K_from_I_minus_P"]
        s2_implied[str(L)] = (kb["e37_frac_reference"] - 1
                              + 2 * kb["alive_non_sentinel"]["c_cross"])
    t4b_all_negative = all(v < 0 for v in s2_implied.values())

    tag_list = [tag(slice="gf", p_slice="p_t", aggregation=a_, window="full", side="right",
                    norm_type="energy", layer=None, rank=None)
                for a_ in ("ratio_of_means", "mean", "median")]
    rep = header("E-40.1-T4a", "docs/e40_todo.md §3.2",
                 {"exp39_supp_three_objects": SUPP,
                  "exp39k_gauge_aware_importance": E39K,
                  "exp27b_jcache_L6": os.path.join("/data/prism_kernels/exp27b/results",
                                                   "exp27b_jcache_L6.pt")},
                 channel_tags=tag_list,
                 extra={"per_layer": per_layer,
                        "a3_identity_asserts": a3,
                        "g_uni_checkpoint": {
                            "layers": LAYERS, "cached": "см. per_layer.g_uni_cached_*",
                            "recomputed": "см. per_layer.g_uni_recomputed_corpus",
                            "matches": bool(g_ok),
                            "matches_exp39k_overlap": e39k_over,
                            "cached_source": ["exp39_supp_three_objects.json (g_uni_corpus)",
                                              "exp39k_gauge_aware_importance.json (L18,L23)"]},
                        "T4b_published_mixed": {"s2_values": s2_implied,
                                                "all_negative": bool(t4b_all_negative),
                                                "note": "c — rom (alive), frac_K — медиана E-37; "
                                                        "смешение агрегаций ⇒ невязка; "
                                                        "красный = ожидаемо (ТЗ §3.2 T4b)"},
                        "resolved_cell": "p_t_ratio_of_means_all",
                        "E6_status": "resolved",
                        "deviations_from_spec": [
                            {"id": "D7_T4b", "description":
                             "published table mixed aggregation (c=rom, frac=E-37 median) — "
                             "s²_implied отрицателен; тождество не обязано замыкаться между "
                             "агрегациями"},
                            {"id": "D7_p0", "description":
                             "p_0-ячейки построены на p̄=mean_t p (в кэше p_0 как фиксированная "
                             "позиция не хранится); p̄ — единственная каноническая фиксация p, "
                             "доступная из данных"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_1_T4a.json")
    dump_json(rep, out)
    print(f"wrote {out} (E6_status={rep['E6_status']}, t4b_all_negative={t4b_all_negative})")
    return rep


if __name__ == "__main__":
    main(results_dir=os.path.join(ROOT, "results"))
