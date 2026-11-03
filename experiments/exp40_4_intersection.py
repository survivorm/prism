#!/usr/bin/env python3
"""E-40.4 — пересечение кривых r_cross(l) (docs/e40_todo.md §6).

Вход: exp32tail_hessian_modes.json — per-doc кривые b (C_b) и off² на ОДНОЙ
согласованной базисной сетке мод (basis_agreement в том же файле; сверка с ESG
из exp30 уже выполнена в E-32tail как off2_vs_e30b_*). Нуль: независимая
перестановка значений ВНУТРИ каждой кривой (не между кривыми — разные базы).
Пререгистрированная ставка: r_cross(L22) ≈ 60.

r_cross: пересечение накопленных долей Δ(r)=cum_b(r)−cum_off²(r): b топ-концентрирована,
off² хвост-концентрирована ⇒ Δ идёт от 0 вверх, и r_cross — первый ноль Δ.

Запуск: .venv/bin/python experiments/exp40_4_intersection.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from exp40_common import ROOT, dump_json, header, tag  # noqa: E402

SRC = os.path.join(ROOT, "results", "exp32tail_hessian_modes.json")
B = 1000
TAIL_R = slice(118, 128)      # r∈[119,128] (0-индекс 118..127)
TOP_K = slice(0, 6)           # k∈[1,6]
BET = 60


def cumshare(x):
    x = np.maximum(np.asarray(x, float), 0.0)
    s = x.sum()
    return np.cumsum(x) / s if s > 0 else np.zeros_like(x)


def r_cross_of(cb, co):
    """Первый ноль Δ(r)=cum_b−cum_off² с линейной интерполяцией; None если нет."""
    d = cumshare(cb) - cumshare(co)
    peak = float(d.max())
    below = np.where(d[1:] < 0)[0]
    if len(below) == 0:
        return None, peak
    i = below[0] + 1
    d0, d1 = d[i - 1], d[i]
    rc = (i - 1) + d0 / (d0 - d1) if d0 > d1 else float(i)
    return float(rc), peak


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    src = json.load(open(SRC))
    per_layer, s04 = {}, {}
    rng = np.random.default_rng(40)
    for L in ("18", "22"):
        docs = {k.split("|", 1)[1]: v for k, v in src["per_doc"].items()
                if k.startswith(L + "|")}
        cb = np.array([v["cb_med"] for v in docs.values()])
        co = np.array([v["off2_med"] for v in docs.values()])
        med_cb, med_co = np.nanmedian(cb, axis=0), np.nanmedian(co, axis=0)
        rc, peak = r_cross_of(med_cb, med_co)
        # нуль: независимые перестановки внутри кривых
        null_peaks = []
        for _ in range(B):
            np_, op_ = rng.permutation(med_cb), rng.permutation(med_co)
            null_peaks.append(float((cumshare(np_) - cumshare(op_)).max()))
        null_peaks = np.array(null_peaks)
        p_onesided = float((null_peaks >= peak).mean())
        p_two = min(1.0, 2 * p_onesided)
        # bootstrap по документам
        boots = []
        for _ in range(B):
            idx = rng.integers(0, cb.shape[0], cb.shape[0])
            rcb, _ = r_cross_of(np.nanmedian(cb[idx], axis=0),
                                np.nanmedian(co[idx], axis=0))
            boots.append(rcb if rcb is not None else 129.0)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        stable = (rc is not None) and p_two < 0.05
        per_layer[L] = {
            "r_cross": rc, "delta_peak": peak,
            "null_peak_p95": float(np.quantile(null_peaks, 0.95)),
            "p_two_sided": p_two, "n_docs": len(docs),
            "bootstrap_CI95": [float(lo), float(hi)],
            "frac_boot_no_crossing": float(np.mean(np.array(boots) > 128.5)),
            "stable": bool(stable),
            "label": "natural × canonical × off²-tail/b_off-top × both",
        }
        s04[L] = {d.replace("/", "#"): float(np.asarray(v["off2_med"])[TAIL_R].sum() /
                           max(np.asarray(v["off2_med"]).sum(), 1e-300)
                           - np.asarray(v["cb_med"])[TOP_K].sum() /
                           max(np.asarray(v["cb_med"]).sum(), 1e-300))
                  for d, v in docs.items()}
        print(f"L{L}: r_cross={rc} peak={peak:.3f} p={p_two:.4f} CI=[{lo:.1f},{hi:.1f}]")

    # lineage реконструкции ставки: ESG-off² (exp30) против b (exp32tail) — РАЗНЫЕ
    # базы; считаем только чтобы показать, откуда взялось «≈60», в вердикт не входит
    e30 = json.load(open(os.path.join(ROOT, "results", "exp30_full_spectrum.json")))
    bet_lineage = {}
    for L in ("18", "22"):
        docs = {k.split("|", 1)[1]: v for k, v in src["per_doc"].items()
                if k.startswith(L + "|")}
        cb_med = np.nanmedian([v["cb_med"] for v in docs.values()], axis=0)
        off_esg = np.maximum(np.array(e30["layers"][L]["delta_off2_med"]), 0.0)
        rc_mix, peak_mix = r_cross_of(cb_med, off_esg)
        bet_lineage[L] = {"r_cross_mixed_bases": rc_mix, "delta_peak": peak_mix,
                          "note": "bases NOT aligned — lineage only, not an object"}

    L22 = per_layer["22"]
    rc22 = L22["r_cross"]
    if not L22["stable"]:
        verdict_bet = "inconclusive"
    elif rc22 is not None and L22["bootstrap_CI95"][0] <= BET <= L22["bootstrap_CI95"][1]:
        verdict_bet = "hit"
    else:
        verdict_bet = "miss"
    all_stable = any(v["stable"] for v in per_layer.values())
    tags = [tag(slice="gf", p_slice="p_t", aggregation="median", window="tail",
                side="mixed", norm_type="energy", layer=22, rank=None)]
    rep = header("E-40.4", "docs/e40_todo.md §6", {"exp32tail_hessian_modes": SRC},
                 channel_tags=tags,
                 extra={"per_layer": per_layer,
                        "per_doc_S04_tail_gap": s04,
                        "bet_lineage_mixed_bases": bet_lineage,
                        "prediction_check": {"predicted": BET, "observed": rc22,
                                             "verdict": verdict_bet},
                        "H4_strong_status": "alive" if all_stable else "dead",
                        "deviations_from_spec": [
                            {"id": "D4_single_basis",
                             "description": "off² берётся с СОГЛАСОВАННОЙ гессайновой базы "
                                            "(exp32tail full), а не сырой ESG-база из exp30: "
                                            "пересечение кривых на разных базах не объект; "
                                            "сверка off²(E-32tail)↔ESG(E-30) уже сделана в "
                                            "E-32tail (off2_vs_e30b_*)"},
                            {"id": "D4_layers",
                             "description": "b_off существует только на L18/L22 — «все "
                                            "доступные слои» = {18,22}"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_4_intersection.json")
    dump_json(rep, out)
    print(f"wrote {out}: bet={verdict_bet}, H4_strong={rep['H4_strong_status']}")
    return rep


if __name__ == "__main__":
    main()
