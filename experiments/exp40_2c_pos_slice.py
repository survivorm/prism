#!/usr/bin/env python3
"""E-40.2/A4 · закрытие p-среза: `r_eff` пары-матрицы по позициям (tier-0, 0 форвардов).

Зачем. Аудит A4 (`results/E40_0_gate.json`) зафиксировал: все спектры E-27.5/E-27.6 сняты с
одной позиции `POS = 0` (`exp27.5_pair_matrix.py:28`, `exp27_6_isolation.py:38`; ключ кэша
`(d, si, l, S, POS)`). Важно: в названии среза `p0` буква `p` — это **позиция**, а не порог
вероятности (в коде нет ни `p > 0`, ни top-K-фильтра). Открытым оставался вопрос, насколько
`r_eff` = exp(энтропия спектра матрицы попарных косинусов) зависит от этой позиции.

Ответ здесь: тот же оценщик (`_gram` + entropic PR из `exp27_6_isolation.py`, перенесены
дословно), тот же корпус 32 документов, но срез меняется по `t`. Если `r_eff(t)` плоский —
«24.4–31.6» не есть артефакт позиции 0; если едет — диапазон надо цитировать с позицией.

Совпадение с E-27.6 по абсолютным числам не ожидается (у нас S=128, там S=64; при S=64-подсрезе
даём отдельную строку). Проверяется другое: форма зависимости от позиции.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _e38_cache as e38  # noqa: E402

LAYERS = (6, 12, 18, 22, 23)
TS = (0, 8, 16, 32, 48, 64, 96, 127)   # позиции в индексе кэша (t=0 ↔ последний токен окна)
KS = (128, 64)                          # 64 = верхняя половина базиса U128, ближе к E-27.5 (K=64)


def gram_r_eff(J: torch.Tensor, P: torch.Tensor, K: int) -> dict:
    """Тот же оценщик, что `_gram` + `t6_eigengap` в `exp27_6_isolation.py` (перенос дословный).

    J: [n, S, K, D], P: [n, S, K] — берётся позиция t (срез делается вызывающим кодом).
    """
    J = J[:, :K].double()
    P = P[:, :K].double()
    Jg = torch.einsum("ik,ikd->id", P, J)              # J^T p
    Jb = P[:, :, None] * (J - Jg[:, None, :])          # gauge-free
    E = torch.einsum("ikd,ikd->i", Jb, Jb)
    Kf = torch.einsum("ikd,jkd->ij", Jb, Jb) / (E[:, None] * E[None, :]).clamp_min(1e-30).sqrt()
    Gb = torch.einsum("nkd,nld->nkl", Jb, Jb)
    gk = Gb.flatten(1).norm(dim=1)
    Kg = (Gb.flatten(1) @ Gb.flatten(1).T) / (gk[:, None] * gk[None, :]).clamp_min(1e-30)
    out = {}
    for name, Km in (("K_F", Kf), ("K_G", Kg)):
        Ks = (Km + Km.T) / 2
        ev = np.sort(np.linalg.eigvalsh(Ks.numpy()))[::-1]
        ev = np.clip(ev, 1e-12, None)
        p = ev / ev.sum()
        out[name] = float(np.exp(-(p * np.log(p)).sum()))
    # own Σp² на том же срезе — нужно, чтобы сопоставлять с полями E-40.2
    out["sum_p2"] = float((P.float().pow(2).sum(-1)).mean())
    # энтропия p по позициям — независимая проверка направления индекса t: у первого токена
    # контекста нет, p должна быть самой размытой (максимум энтропии)
    out["p_entropy"] = float(-(P * P.clamp_min(1e-12).log()).sum(-1).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    rd = Path(args.results_dir)
    t0 = time.time()

    out: dict = {
        "experiment": "E-40.2/A4 p-slice: r_eff матрицы попарных косинусов по позициям",
        "tz": "docs/e40_todo.md §12 (A4) · docs/E40_tier0.md A4",
        "observational": True, "new_forwards": 0,
        "inputs": {"cache": "exp27b (только чтение)", "layers": list(LAYERS), "positions": list(TS),
                   "K": list(KS), "docs": 32},
        "estimator": "`_gram` + entropic PR из exp27_6_isolation.py, перенесены дословно; "
                     "там срез S=64/K=128/POS=0, здесь позиция меняется",
        "per_layer": {},
    }

    for L in LAYERS:
        d = e38.load_layer(L, want=("J", "P"))            # [32,128,K,*], [32,128,K]
        J, P = d["J"], d["P"]
        assert J.shape[1] == 128, f"L{L}:unexpected shape {tuple(J.shape)}"
        rows = {}
        for t in TS:
            per_k = {}
            for K in KS:
                r = gram_r_eff(J[:, t], P[:, t], K)
                per_k[f"K{K}"] = {"r_eff_KF": r["K_F"], "r_eff_KG": r["K_G"], "sum_p2": r["sum_p2"],
                                  "p_entropy": r["p_entropy"]}
            rows[str(t)] = per_k
        # сводка: размах r_eff по позициям при K=128
        v = [rows[str(t)]["K128"]["r_eff_KF"] for t in TS]
        v64 = [rows[str(t)]["K64"]["r_eff_KF"] for t in TS]
        out["per_layer"][str(L)] = {
            "by_position": rows,
            "r_eff_KF_span_K128": {"min": min(v), "max": max(v), "rel_span": (max(v) - min(v)) / float(np.mean(v)),
                                   "t_at_min": TS[int(np.argmin(v))], "t_at_max": TS[int(np.argmax(v))]},
            "r_eff_KF_span_K64": {"min": min(v64), "max": max(v64), "rel_span": (max(v64) - min(v64)) / float(np.mean(v64))},
            "p_entropy_by_t": {str(t): rows[str(t)]["K128"]["p_entropy"] for t in TS},
            "sum_p2_by_t": {str(t): rows[str(t)]["K128"]["sum_p2"] for t in TS},
            "first_vs_last_ratio_K128": float(v[-1] / v[0]),
        }
        print(f"L{L}: r_eff(K=128) по t " + " ".join(f"{x:.1f}" for x in v), flush=True)
        print("     r_eff(K=64)  по t " + " ".join(f"{x:.1f}" for x in v64), flush=True)
        print("     H(p)          " + " ".join(f"{rows[str(t)]['K128']['p_entropy']:.2f}" for t in TS), flush=True)

    # вердикт: плоский срез или нет
    spans = [out["per_layer"][str(L)]["r_eff_KF_span_K128"]["rel_span"] for L in LAYERS]
    spans64 = [out["per_layer"][str(L)]["r_eff_KF_span_K64"]["rel_span"] for L in LAYERS]
    out["verdict"] = {
        "max_rel_span_K128": max(spans), "max_rel_span_K64": max(spans64),
        "slice_flat": bool(max(spans) < 0.15),
        "текст": "slicing-чувствительность `r_eff` оценена на том же оценщике и корпусе, что E-27.6; "
                 "при `slice_flat = true` диапазон 24.4–31.6 нельзя объяснить позицией 0 (позиция меняется — "
                 "число стоит), при false — цитировать только вместе с позицией",
    }
    out["wall_sec"] = round(time.time() - t0, 1)
    assert all(math.isfinite(x) for L in LAYERS for x in
               (out["per_layer"][str(L)]["first_vs_last_ratio_K128"],)), "r_eff не конечный"
    path = rd / "E40_2c_pos_slice.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {path}: slice_flat={out['verdict']['slice_flat']} "
          f"max_span K128={max(spans):.3f} K64={max(spans64):.3f} wall {out['wall_sec']} с")


if __name__ == "__main__":
    main()
