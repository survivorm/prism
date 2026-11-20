#!/usr/bin/env python
"""Аудит E-45b: выдерживает ли блочный нуль свои предположения.

Карточка E-45b (`docs/ref-exp.md:2255`) отчитывает «из 54 пар лист→поведение при |z|>3
оказалось 26» против нуля «перестановка БЛОКАМИ по 8 позиций внутри документа». У нуля
одно несущее предположение: блоку длины 8 достаточно, чтобы убить дальнюю связь.
Если у целевой переменной автокорреляция длиннее блока, пермутация оставляет связь
внутри переставленной серии, нулевое распределение ρ сужается, и z раздувается.

Проверяется:
  A. автокорреляция каждой цели по лагам 1..24 внутри документов (длина связи);
  B. те же 54 пары при BLOCK = 8/16/32/64 — сколько |z|>3 остаётся;
  C. калибровка самого нуля: распределение z перестановок (должно быть ~N(0,1));
  D. вклад «вырожденных» целей: copy16 = 0 у 80 % позиций, in5 = 1 у 73 %.

Запуск: cd experiments && ../.venv/bin/python _audit_checks/e45b_blocknull.py
Артефакт: results/_audit_checks/e45b_blocknull.json
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, os.pardir, "e41_50"))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, EXP)
import exp45b_analysis as A  # noqa: E402

RES = os.path.join(ROOT, "results", "exp45b_leaf_table.json.gz")
OUT = os.path.join(ROOT, "results", "_audit_checks", "e45b_blocknull.json")
BLOCKS = (8, 16, 32, 64)
N_PERM = 1000


def main():
    with gzip.open(RES, "rt") as f:
        d = json.load(f)
    rows = d["rows"]
    ht = np.array([r["has_target"] for r in rows], dtype=bool)
    doc_of = np.array([f"{r['domain']}#{r['doc']}" for r in rows])

    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    gap = col("gap_r16")
    valid = ht & np.isfinite(gap) & (gap >= A.GAP_MIN)

    out = {"experiment": "E-45b", "audit": "block-null assumptions",
           "source_artifact": "results/exp45b_leaf_table.json.gz",
           "block_original": A.BLOCK, "n_perm": N_PERM}

    # --- D: вырожденность целей
    deg = {}
    for tg in A.TARGETS:
        y = A.shift_next(rows, tg)
        m = valid & np.isfinite(y)
        v = y[m]
        vals, cnt = np.unique(v, return_counts=True)
        deg[tg] = {"n": int(m.sum()),
                   "frac_mode": float(cnt.max() / len(v)),
                   "mode_value": float(vals[cnt.argmax()]),
                   "n_unique": int(len(vals))}
    out["degeneracy"] = deg

    # --- A: автокорреляция целей внутри документов по лагам
    acf = {}
    for tg in A.TARGETS:
        y = A.shift_next(rows, tg)
        prof = []
        for lag in range(1, 25):
            rs = []
            for doc in np.unique(doc_of):
                ii = np.flatnonzero(doc_of == doc)
                if len(ii) <= lag + 10:
                    continue
                a, b = y[ii[:-lag]], y[ii[lag:]]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
                    continue
                rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
            prof.append(float(np.median(rs)) if rs else float("nan"))
        # длина связи: последний лаг, где |acf| ещё > 0.1
        acf[tg] = {"acf_lag": prof,
                   "cross_lag_0.1": int(max([k + 1 for k, v in enumerate(prof)
                                             if abs(v) > 0.1], default=0))}
    out["acf"] = acf

    # --- B/C: те же пары при разных BLOCK
    def run(block):
        A.BLOCK = block
        rng = np.random.default_rng(A.SEED)
        res = []
        for f_ in A.FEATS:
            x = col(f_)
            for tg in A.TARGETS:
                y = A.shift_next(rows, tg)
                m2 = valid & np.isfinite(x) & np.isfinite(y)
                c, z, _ci, n = A.block_perm_corr(x, y, m2, doc_of, rng)
                res.append({"feat": f_, "target": tg, "rho": c, "z": z, "n": n})
        sig = [r for r in res if np.isfinite(r["z"]) and abs(r["z"]) > 3]
        zs = np.array([r["z"] for r in res if np.isfinite(r["z"])])
        return res, sig, zs

    per_block = {}
    raw = {}
    for blk in BLOCKS:
        res, sig, zs = run(blk)
        raw[blk] = res
        per_block[str(blk)] = {
            "n_pairs": len(res),
            "n_abs_z_gt3": len(sig),
            "n_abs_z_gt3_null_only": int(sum(1 for r in sig if r["feat"] in ("h_norm", "j_norm")
                                             and r["target"] in ("in5", "copy16"))),
            "max_abs_z": float(np.nanmax(np.abs(zs))),
            "sig_pairs": sorted([f"{r['feat']}->{r['target']}" for r in sig]),
        }
        print(f"BLOCK={blk:3d}: |z|>3 у {len(sig)} из {len(res)} пар, max|z| = {np.nanmax(np.abs(zs)):.1f}")
    out["per_block"] = per_block

    # --- C: насколько нуль-распределение ρ симметрично/центрировано для самой сильной пары
    A.BLOCK = 8
    strongest = max(raw[8], key=lambda r: abs(r["z"]) if np.isfinite(r["z"]) else -1)
    cal = {}
    for blk in BLOCKS:
        A.BLOCK = blk
        rng = np.random.default_rng(A.SEED)
        x = col(strongest["feat"])
        y = A.shift_next(rows, strongest["target"])
        m2 = valid & np.isfinite(x) & np.isfinite(y)
        idx, pos = A.masked_blocks(m2, doc_of)
        rx = np.argsort(np.argsort(x[idx])).astype(np.float64)
        ry = np.argsort(np.argsort(y[idx])).astype(np.float64)
        null = np.empty(N_PERM)
        for p in range(N_PERM):
            order = rng.permutation(len(pos))
            perm = np.concatenate([pos[k] for k in order])
            null[p] = np.corrcoef(rx[perm], ry)[0, 1]
        cal[str(blk)] = {"pair": f"{strongest['feat']}->{strongest['target']}",
                         "null_mean": float(null.mean()), "null_sd": float(null.std()),
                         "null_p2.5": float(np.percentile(null, 2.5)),
                         "null_p97.5": float(np.percentile(null, 97.5)),
                         "frac_abs_null_gt_rho": float(np.mean(np.abs(null) >= abs(strongest["rho"])))}
    out["null_calibration"] = cal

    # --- блок-нуль против полного нуля (перестановка по всем документам = верхняя граница)
    per_block["_note"] = ("если |z|>3 held at BLOCK=8 but disappears at larger block, "
                          "the reported z measures the block length, not the coupling")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("\n=== D. вырожденность целей")
    for tg, v in deg.items():
        print(f"  {tg:>7s}: n={v['n']:5d} доля моды={v['frac_mode']*100:5.1f} % "
              f"(значение {v['mode_value']:g}), уникальных {v['n_unique']}")
    print("\n=== A. автокорреляция целей внутри документов (медиана по 32 документам)")
    for tg, v in acf.items():
        print(f"  {tg:>7s}: связь до лага {v['cross_lag_0.1']:2d} "
              f"(acf1 {v['acf_lag'][0]:+.3f} … acf8 {v['acf_lag'][7]:+.3f} "
              f"… acf16 {v['acf_lag'][15]:+.3f})")
    print("\n=== C. калибровка нуля для самой сильной пары "
          f"({cal['8']['pair']})")
    for blk, v in cal.items():
        print(f"  BLOCK={blk:>3s}: null mean {v['null_mean']:+.4f} sd {v['null_sd']:.4f} "
              f"p2.5..p97.5 [{v['null_p2.5']:+.3f}, {v['null_p97.5']:+.3f}] "
              f"доля |null| ≥ |ρ| = {v['frac_abs_null_gt_rho']*100:.1f} %")
    print("\nсохранено:", OUT)


if __name__ == "__main__":
    main()
