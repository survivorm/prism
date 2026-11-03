#!/usr/bin/env python3
"""E-40.0 — A4-gate: p_slice спектров E-27.5/27.6 (docs/e40_todo.md §2, ред. 2).

Процедура: (1) код-аудит продюсеров — решает; (2) векторная сверка E-27-кэша с
exp27b на пересекающейся сетке (S=128) — диагностика, не гейт; p_checksum удалён
как неработоспособный (константа для любой ренормированной p).

Запуск: .venv/bin/python experiments/exp40_0_gate.py  (минуты, ноль форвардов).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from exp40_common import LAYERS, dump_json, header, sha16  # noqa: E402

ROOT = os.path.join(HERE, "..")
E27_CACHE = os.path.join(ROOT, "results", "exp27_jcache.pt")
EXP27B_DIR = "/data/prism_kernels/exp27b/results"

AUDIT_FACTS = [
    # (файл, обязательная подстрока, что фиксирует)
    ("experiments/exp27.5_pair_matrix.py", "S, K, POS = 64, 64, 0",
     "E-27.5: S=64, K=64, POS=0; «p0» в имени файла = индекс позиции, не фикс. p0"),
    ("experiments/exp27_6_isolation.py", "S, K, POS = 64, 128, 0",
     "E-27.6 основной срез: S=64, POS=0"),
    ("experiments/exp27_6_isolation.py", "(d, si, l, 128, 32)",
     "E-27.6 T5-срез читает (S=128, POS=32)"),
]


def code_audit() -> list[dict]:
    out = []
    for rel, needle, note in AUDIT_FACTS:
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as f:
            src = f.read()
        found = needle in src
        line = next((i + 1 for i, ln in enumerate(src.splitlines()) if needle in ln), None)
        out.append({"file": rel, "needle": needle, "found": found,
                    "line": line, "meaning": note})
        if not found:
            raise AssertionError(f"код-аудит: {needle!r} не найден в {rel} — продюсер "
                                 "изменился, факты ТЗ §2.1 протухли, стоп")
    return out


def vector_reconcile(ck27: dict) -> dict:
    """Диагностика на S=128: правильный реверс (t = POS) против неверного (t = 127-POS)."""
    combo = sorted({(k[3], k[4]) for k in ck27["P128"]})
    inter = [(s, p) for (s, p) in combo if s == 128]
    if not inter:
        return {"intersecting_grid": [], "matches_exp27b": None,
                "reversal_confirmed": None,
                "note": "пересечение (S,POS)-сеток пусто — сверка невозможна, не failure"}
    dev_rev, dev_wrong, n = 0.0, 0.0, 0
    for L in LAYERS:
        ck = torch.load(os.path.join(EXP27B_DIR, f"exp27b_jcache_L{L}.pt"),
                        map_location="cpu", weights_only=False)
        for (s_len, pos) in inter:
            for k27 in ck27["P128"]:
                dom, si, l, S_, POS = k27
                if l != L or S_ != s_len or POS != pos:
                    continue
                p27 = ck27["P128"][k27].double()
                p_right = ck["P128"][(dom, si, POS)].double()       # token_idx = S-1-POS совпадает
                p_wrong = ck["P128"][(dom, si, 127 - POS)].double()  # неверная конвенция
                dev_rev = max(dev_rev, float((p27 - p_right).abs().max()))
                dev_wrong = max(dev_wrong, float((p27 - p_wrong).abs().max()))
                n += 1
    return {"intersecting_grid": [[s, p] for (s, p) in inter], "n_compared": n,
            "max_abs_dev_reversed": dev_rev, "max_abs_dev_wrong_pairing": dev_wrong,
            "matches_exp27b": bool(dev_rev < 1e-6),
            "reversal_confirmed": bool(dev_rev < dev_wrong / 10.0)}


def main(results_dir: str = "results") -> dict:
    t0 = time.time()
    audit = code_audit()
    ck27 = torch.load(E27_CACHE, map_location="cpu", weights_only=False)
    ck6 = torch.load(os.path.join(EXP27B_DIR, "exp27b_jcache_L6.pt"),
                     map_location="cpu", weights_only=False)
    assert ck27["pool_fp"] == ck6["pool_fp"], "пул E-27 ≠ пул exp27b — сверка бессмысленна"
    assert ck27["U128"] == ck6["U128"], "база U128 расходится"
    vec = vector_reconcile(ck27)
    rep = header(
        "E-40.0", "docs/e40_todo.md §2 (редакция 2)",
        {"exp27_jcache": E27_CACHE,
         "exp27b_L6": os.path.join(EXP27B_DIR, "exp27b_jcache_L6.pt")},
        channel_tags=[],
        extra={"code_audit": audit, "vector_reconcile": vec,
               "p_checksum_note": "удалён: Σ‖p_t‖₁ ≈ 4096.000009 — константа для любой "
                                  "ренормированной p, и p_t, и p_0 (ТЗ §2.4)",
               "E27_5_6_p_slice": "p_t",
               "p_position": 0,
               "T5_p_position": 32,
               "source": "code_audit(exp27.5_pair_matrix.py:28, exp27_6_isolation.py:38,58)",
               "matches_exp27b": vec["matches_exp27b"],
               "reversal_confirmed": vec["reversal_confirmed"],
               "gate_pass": True,
               "E40_2_can_start": True,
               "wall_sec": round(time.time() - t0, 1)},
    )
    out = os.path.join(results_dir, "E40_0_gate.json")
    dump_json(rep, out)
    print(json.dumps({k: rep[k] for k in
                      ("E27_5_6_p_slice", "matches_exp27b", "reversal_confirmed",
                       "gate_pass", "wall_sec")}, ensure_ascii=False))
    print(f"wrote {out}")
    return rep


if __name__ == "__main__":
    main(results_dir=os.path.join(HERE, "..", "results"))
