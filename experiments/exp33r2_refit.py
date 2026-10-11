#!/usr/bin/env python3
"""E-33r.2 refit A/E: p0 = НОРМИРОВАННАЯ медиана (Σp=1) + assert.

Причина (ревью 2026-07-07): ненормированная медиана по строкам симплекса
имеет Σ ≈ 0.006, и P(c·p) = c·diag(p) − c²·ppᵀ ≈ c·diag(p) — крошечный
диагональный скалинг, не gauge-оператор. Старые dgp/E_fixed_p мерили
«насколько малое диагональное возмущение сдвигает подпространство», а не
«возвращает ли фиксация P структуру».

Диагностики C/D/F/B не зависят от p0 — не трогаем. Пересчитываем только
dgp_* (A/E), обновляем layers[23].diag.E_fixed_p / E_fixed_p_r + verdict.
Вход: results/exp33r_gauge_anatomy.json (per_doc из полного прогона).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch

torch.set_num_threads(1)   # не забирать все ядра

import exp33r_gauge_anatomy as E33R
from exp33r_gauge_anatomy import agg_positions

CKPT_DIR = "/data/prism_kernels/exp27b/results"
L23 = 23
R_MAIN = 16
DIAG_R = [1, 2, 4, 8, 16, 32]
DELTA_GRID = [1, 4, 8, 16]
JSON = "results/exp33r_gauge_anatomy.json"
CKPT = "results/exp33r2_refit.ckpt.json"


def dgp_for_doc(Jdoc, Pdoc, sent):
    """A/E с нормированным p0: dgp_{dt} (r=16) + dgp_{dt}_r{rt} для rt∈{1,2,8,16}.
    Возвращает dict {name: [T-dt] (None=NaN)}."""
    T = Jdoc.shape[0]
    sent_t = torch.as_tensor(sent, dtype=torch.bool)
    alive = ~sent_t
    p0 = Pdoc[alive].median(dim=0).values if alive.any() else Pdoc.mean(dim=0)
    p0 = p0 / p0.sum().clamp_min(1e-30)
    assert abs(float(p0.sum()) - 1.0) < 1e-5, "p0 must be normalized"
    Sfp, sfp, Vhfp = torch.linalg.svd(
        E33R.p_apply(Jdoc, p0.unsqueeze(0).expand(T, -1)), full_matrices=False)
    out = {}
    for dt in DELTA_GRID:
        mask = (~(sent_t[:-dt] | sent_t[dt:])).numpy()
        # dgp_{dt} — r = R_MAIN
        r = R_MAIN
        M = Vhfp[:-dt, :r] @ Vhfp[dt:, :r].transpose(-1, -2)
        sv = torch.linalg.svdvals(M).clamp(max=1.0 - 1e-12)
        flat = (sfp[:-dt, r - 1] / sfp[:-dt, 0].clamp_min(1e-30) < E33R.FLAT_REL).numpy()
        a = torch.acos(sv).max(-1).values.numpy()
        out[f"dgp_{dt}"] = [None if (not m or f) else float(x)
                            for m, f, x in zip(mask, flat, a)]
        # dgp_{dt}_r{rt}
        for rt in (1, 2, 8, 16):
            r = rt
            M = Vhfp[:-dt, :r] @ Vhfp[dt:, :r].transpose(-1, -2)
            sv = torch.linalg.svdvals(M).clamp(max=1.0 - 1e-12)
            flat = (sfp[:-dt, r - 1] / sfp[:-dt, 0].clamp_min(1e-30) < E33R.FLAT_REL).numpy()
            a = torch.acos(sv).max(-1).values.numpy()
            out[f"dgp_{dt}_r{rt}"] = [None if (not m or f) else float(x)
                                      for m, f, x in zip(mask, flat, a)]
    return out


def main():
    t0 = time.time()
    res = json.load(open(JSON))
    ck = E33R.load_layer(CKPT_DIR, L23)
    domains = sorted({k[0] for k in ck["JR"]})
    docs = [(d, si) for d in domains for si in range(ck["n_samples"])]
    n_docs = res["layers"][str(L23)]["n_docs"]
    assert len(docs) == n_docs, f"len(docs)={len(docs)} vs n_docs={n_docs}"

    if os.path.exists(CKPT):
        done = {int(k): v for k, v in json.load(open(CKPT)).items()}
        print(f"resume: {len(done)}/{len(docs)}")
    else:
        done = {}

    for i, (d, si) in enumerate(docs):
        if i in done:
            continue
        Jdoc, Pdoc, sent = E33R.load_doc(ck, d, si)
        done[i] = dgp_for_doc(Jdoc, Pdoc, sent)
        del Jdoc, Pdoc, sent
        json.dump({str(k): v for k, v in done.items()}, open(CKPT, "w"))
        if (i + 1) % 4 == 0:
            print(f"  doc {i+1}/{len(docs)} t={time.time()-t0:.0f}s", flush=True)

    # --- update layers[23].diag ---
    diag = res["layers"][str(L23)]["diag"]
    rts = [rt for rt in DIAG_R if str(rt) in diag.get("C_raw", {})]
    diag["E_fixed_p"] = {
        str(rt): {str(dt): agg_positions(
            [np.asarray(done[i][f"dgp_{dt}"], dtype=float) for i in range(len(docs))])
            for dt in DELTA_GRID} for rt in (16, 32) if rt in rts}
    diag["E_fixed_p_r"] = {
        str(rt): {str(dt): agg_positions(
            [np.asarray(done[i][f"dgp_{dt}_r{rt}"], dtype=float) for i in range(len(docs))])
            for dt in DELTA_GRID} for rt in (1, 2, 8, 16)}

    res["meta"]["refit_note"] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "E-33r.2 A/E (dgp, E_fixed_p, E_fixed_p_r) пересчитаны с "
                "нормированным p0 (Σp=1). Старые значения использовали "
                "ненормированную медиану (Σ≈0.006): P ≈ diag — не gauge-оператор. "
                "C/D/F/B не зависят от p0 — не тронуты. Также зафиксировано "
                "doc↔code расхождение: Exp33r2.md «p_0 = аргмакс p_max», код — "
                "медиана; публикуемая версия = медиана + нормировка.",
        "refit_timing_sec": round(time.time() - t0, 1),
    }

    with open(JSON, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, allow_nan=True)
    os.remove(CKPT)
    print(f"done in {time.time()-t0:.0f}s; JSON updated")

    # --- MD: обновить A/E таблицу ---
    md = open("docs/Exp33r2.md").read()
    l23 = res["layers"][str(L23)]["diag"]

    def val(e):
        return e.get("med") if e else None
    new1 = val(l23["E_fixed_p_r"]["16"]["1"])
    new16 = val(l23["E_fixed_p_r"]["16"]["16"])
    # замена старой строки Δ=1 в A/E таблице
    old1 = "0.252"
    old16 = "0.338"
    if new1 is not None:
        md = md.replace(f"| 1 | 1.461 | 1.456 | 1.447 | 1.455 | 1.499 | 0.289 | **{old1}** |",
                        f"| 1 | 1.461 | 1.456 | 1.447 | 1.455 | 1.499 | 0.289 | **{new1:.3f}** (refit) |")
        md = md.replace(f"| 16 | 1.567 | 1.565 | 1.568 | 1.570 | 1.566 | 0.464 | {old16} |",
                        f"| 16 | 1.567 | 1.565 | 1.568 | 1.570 | 1.566 | 0.464 | {new16:.3f} (refit) |")
    # заголовок: median, не argmax
    md = md.replace("### Диагностика A/E — fixed-P (p_0 = аргмакс p_max среди non-sentinel)",
                    "### Диагностика A/E — fixed-P (p_0 = нормированная медиана p_t по non-sentinel, Σp=1)\n\n"
                    "*(до 2026-07-07 заголовок гласил «аргмакс p_max» — расхождение doc↔code:\n"
                    "код всегда использовал медиану; также медиана была ненормированной — см. refit_note)*")
    open("docs/Exp33r2.md", "w").write(md)
    print("MD updated")


if __name__ == "__main__":
    main()
