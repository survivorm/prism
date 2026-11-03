#!/usr/bin/env python3
"""E-40.7 — фиксация G(l) = 1 − Δ_fixP/Δ_raw (docs/e40_todo.md §9).

Слойный уровень E-33r (strict_diag, Δ=1, r=16) даёт ТОЛЬКО Δ_raw
(поля `E_fixed_p` нет ни в одном артефакте): G(l) считается из per-doc пересчёта ниже — «G —
побочный выход E-40.1» (та же фиксация p̄). Плюс независимый per-(doc,layer)
пересчёт из кэша (он же и есть источник Δ_fixP): max principal angle правых топ-16 подпространств соседних
позиций, raw и P(p̄)J, p̄ = корпусное среднее P-строка (один и тот же для всех
слоёв — P в кэше слойно-инвариантен). Корреляции G(l) с каноническими
C_F(1)/C_top16(1) (из E40_2) и κ(l) (exp38d).

Замечание аудита: |Δp| и энтропия — функции одного и того же слойно-инвариантного
p ⇒ переслойная корреляция G(l) с ними вырождена (константный ковариат); её не
считаем, говорим прямо.

Запуск: .venv/bin/python experiments/exp40_7_G_fixation.py  (после E-40.2)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from _e38_cache import load_layer  # noqa: E402
from exp40_common import DOMAINS, LAYERS, N_PER_DOMAIN, ROOT, S, dump_json, header, tag  # noqa: E402

STRICT = os.path.join(ROOT, "results", "exp33r_gauge_anatomy_strict_diag.json")
E31 = os.path.join(ROOT, "results", "exp31_correlation_function.json")
E38D = os.path.join(ROOT, "results", "exp38d_holonomy_spectrum.json")
E402 = os.path.join(ROOT, "results", "E40_2_double_fit.json")
R_TOP = 16


def vtop(J, r=R_TOP):
    G = torch.bmm(J, J.transpose(1, 2))
    w, u = torch.linalg.eigh(G)
    wr, Ur = w[..., -r:], u[..., -r:]
    V = torch.bmm(J.transpose(1, 2), Ur)
    return (V / wr.unsqueeze(1).clamp_min(1e-30).sqrt()).contiguous()


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    strict = json.load(open(STRICT))
    e33r_raw = {str(L): float(strict["layers"][str(L)]["delta"]["1"]["raw"]["med"])
                for L in LAYERS}
    assert abs(e33r_raw["23"] - 0.289) < 0.02, "якорь Δ_raw(L23) уехал"   # E-33r
    per_layer = {}          # заполняется после per-doc цикла: G из одного среза

    # корпусное среднее p (слойно-инвариантно, берём L6)
    ck = load_layer(6, want=("P",))
    p_bar = ck["P"].double().reshape(-1, ck["P"].shape[-1]).mean(0)
    p_bar = p_bar / p_bar.sum()
    del ck

    per_doc = {}
    for L in LAYERS:
        tl = time.time()
        ck = load_layer(L, want=("J", "P"))
        J = ck["J"]
        docs = {}
        for d_of, dom in enumerate(DOMAINS):
            for si in range(N_PER_DOMAIN):
                di = d_of * N_PER_DOMAIN + si   # плоский индекс документа
                # (кэш лежит домен-мажорно: (wiki,0..7),(math,0..7),…); раньше
                # сюда попадал индекс ДОМЕНА 0..3, и все 32 «документа» были
                # четырьмя первыми wiki-документами, повторёнными по 8 раз
                d = f"{dom}#{si}"
                Jd = J[di]
                Jfix = torch.empty_like(Jd)
                for t in range(S):
                    Jfix[t] = (Jd[t].double() * p_bar[:, None]
                               - p_bar[:, None] * (p_bar @ Jd[t].double())).float()
                with torch.no_grad():
                    Vr, Vf = vtop(Jd), vtop(Jfix)
                    A, B = Vr[:-1], Vr[1:]
                    M = A.transpose(-1, -2) @ B
                    ang_r = torch.acos(torch.linalg.svdvals(M).clamp(max=1 - 1e-12)).max(-1)
                    A, B = Vf[:-1], Vf[1:]
                    M = A.transpose(-1, -2) @ B
                    ang_f = torch.acos(torch.linalg.svdvals(M).clamp(max=1 - 1e-12)).max(-1)
                dr, df = float(ang_r.values.mean()), float(ang_f.values.mean())
                docs[d] = {"delta_raw": dr, "delta_fixP": df, "G": 1.0 - df / dr}
                del Vr, Vf
                Jd = Jfix = None
        per_doc[str(L)] = docs
        print(f"L{L} per-doc G med={np.median([v['G'] for v in docs.values()]):.3f} "
              f"({time.time()-tl:.0f}s)")
        del ck, J

    # Слойный уровень из ТОГО ЖЕ пер-документного оценивателя, что и знаменатель:
    # отношение медиан по одному срезу (правило агрегации, e40_todo §1.1 / E6).
    for L in LAYERS:
        dr = float(np.median([v["delta_raw"] for v in per_doc[str(L)].values()]))
        df = float(np.median([v["delta_fixP"] for v in per_doc[str(L)].values()]))
        if df > dr * (1 + 1e-9):
            raise AssertionError(
                f"L{L}: фиксация p̄ УВЕЛИЧИЛА угловую разность ({df:.4f} > {dr:.4f}); "
                "фиксированный срез не может быть сырым — искать баг в p_bar/Jfix")
        per_layer[str(L)] = {
            "delta_raw": dr, "delta_fixP": df, "G": 1.0 - df / dr,
            "e33r_delta_raw": e33r_raw[str(L)],
            "rel_dev_own_vs_e33r_raw": abs(dr - e33r_raw[str(L)]) / e33r_raw[str(L)],
            "label": "observational × fixP × top16 right × angle (медиана по 32 документам)",
        }
        print(f"L{L}: Δ_raw {dr:.4f} (E-33r {e33r_raw[str(L)]:.4f}), "
              f"Δ_fixP {df:.4f} → G={1.0 - df / dr:.3f}")

    # корреляции с каноническими C(1) и κ
    e402 = json.load(open(E402))
    e38d = json.load(open(E38D))
    cf1, ct1, kap = {}, {}, {}
    for L in LAYERS:
        arr = np.array([v["C_gf"] for v in e402["per_doc_canonical_curves"][str(L)].values()])
        arrt = np.array([v["C_top_gf"] for v in e402["per_doc_canonical_curves"][str(L)].values()])
        cf1[L], ct1[L] = float(np.nanmedian(arr[:, 0])), float(np.nanmedian(arrt[:, 0]))
        kap[L] = e38d["metrics"][str(L)]["16"]["kappa_median"]
    g = np.array([per_layer[str(L)]["G"] for L in LAYERS])
    corrs = {}
    for nm, xs in (("C_F_canon_delta1", [cf1[L] for L in LAYERS]),
                   ("C_top16_canon_delta1", [ct1[L] for L in LAYERS]),
                   ("kappa_exp38d", [kap[L] for L in LAYERS])):
        r = spearmanr(g, xs)
        corrs[nm] = {"spearman": float(r.statistic), "p": float(r.pvalue), "n": len(LAYERS)}
    gdoc = np.array([v["G"] for L in LAYERS for v in per_doc[str(L)].values()])

    monotone = bool(all(x >= y for x, y in zip(g, g[1:]))
                    or all(x <= y for x, y in zip(g, g[1:])))
    max_g = float(g.max())
    if monotone and max_g < 0.2:
        verdict = "G_small_monotone"
    elif max_g > 0.5:
        verdict = "G_large"
    else:
        verdict = "ambiguous"
    # p̄ (один на все позиции) в словаре ТЗ называется p_slice="p_0" (см. E-40.1
    # manifest: «слойные med'ы strict помечаются p_slice=p_0»), срез — fixP.
    tags = [tag(slice="fixP", p_slice="p_0", aggregation="median", window="top",
                side="right", norm_type="norm", layer=None, rank=R_TOP,
                channel="observational")]
    rep = header("E-40.7", "docs/e40_todo.md §9",
                 {"exp33r_gauge_anatomy_strict_diag": STRICT,
                  "E40_2_double_fit": E402, "exp38d_holonomy_spectrum": E38D},
                 channel_tags=tags,
                 extra={"per_layer": per_layer, "per_doc": per_doc,
                        "per_doc_S07_G": {str(L): {d: v["G"] for d, v in per_doc[str(L)].items()}
                                          for L in LAYERS},
                        "profile": monotone and "monotone" or "non_monotone",
                        "max_G": max_g, "verdict": verdict,
                        "correlations_G_l": corrs,
                        "per_doc_G_stats": {"median": float(np.median(gdoc)),
                                            "p10": float(np.quantile(gdoc, 0.1)),
                                            "p90": float(np.quantile(gdoc, 0.9))},
                        "deviations_from_spec": [
                            {"id": "D7_degenerate_covariates",
                             "description": "переслойная корреляция G(l) с |Δp| и энтропией "
                                            "не вычисляется: p слойно-инвариантен в кэше ⇒ "
                                            "ковариаты константны по l (стрельба по нулям); "
                                            "частотный partial-out дан per-doc"},
                            {"id": "D7_no_fixed_p_artifact",
                             "description": "пререгистрация «слойный Δ_fixP — побочный выход "
                                            "E-40.1/E-33r» неисполнима: поля E_fixed_p нет ни "
                                            "в exp33r_gauge_anatomy.json, ни в strict_diag, ни "
                                            "в E40_1_audit.json (проверено по всем трём); G(l) "
                                            "построен целиком из собственного per-doc "
                                            "пересчёта, E-33r оставлен якорем по Δ_raw "
                                            "(rel_dev в per_layer)"},
                            {"id": "D7_pbar_corpus",
                             "description": "p̄ = корпусное среднее P-строка (renorm), "
                                            "один на все слои — согласовано с E-33r fix-p"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_7_G_fixation.json")
    dump_json(rep, out)
    print(f"wrote {out}: verdict={verdict}, max_G={max_g:.3f}")
    return rep


if __name__ == "__main__":
    main()
