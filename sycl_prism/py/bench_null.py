"""Бенчмарк null-слоя — главного CPU-горба «остатка».

Что меряем: один блок перестановочного null'а AT0_1 (60 сидов × 1 (док, объект))
и экстраполяция на весь null-слой (53 дока × 3 объекта).

Протокол идентичен серии (AT0_1_accounting._perm_stat): RNG остаётся на хосте,
на GPU уезжает арифметика ⇒ null-распределение остаётся тем же.

Запуск:
    python3 bench_null.py                # N=383, D=1024 (масштаб серии)
    python3 bench_null.py --N 65535      # профиль «32k-игла»
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prism_sycl as ps
import prism_sycl_ref as ref

NNULL = 60          # ТЗ §0.4 (у ревьюера 60)
POOL_DOCS = 53
OBJECTS = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=383)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=NNULL)
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    N, D = args.N, args.D
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, D)).astype(np.float32)
    f = float(ref.f_dc(x.astype(np.float64)))
    seeds = [(s + 1) * 1009 + 0 for s in range(args.seeds)]

    print(f"=== null-слой: N={N} D={D} сидов={len(seeds)} ===")
    print("backend:", ps.backend())

    # numpy-эталон (как сейчас в AT0_1)
    t0 = time.perf_counter()
    for s in seeds:
        ref.perm_stat(x.astype(np.float64), f, s)
    t_np = (time.perf_counter() - t0) / args.n
    print(f"numpy (fp64)      : {t_np*1e3:9.1f} мс на {len(seeds)} сидов "
          f"({t_np/len(seeds)*1e3:.1f} мс/сид)")

    if ps.available() and not ps._is_reference():
        t0 = time.perf_counter()
        for _ in range(args.n):
            got = ps.permutation_null(x, f, seeds)
        t_gpu = (time.perf_counter() - t0) / args.n
        print(f"sycl (iGPU)       : {t_gpu*1e3:9.1f} мс на {len(seeds)} сидов "
              f"({t_gpu/len(seeds)*1e3:.1f} мс/сид)")
        print(f"ускорение         : ×{t_np/max(t_gpu,1e-12):.2f}")
        # контроль: значения совпадают?
        want = np.array([ref.perm_stat(x.astype(np.float64), f, s) for s in seeds])
        gotv = np.array([got["band"], got["e3"], got["e4"]]).T
        d = float(np.max(np.abs(gotv - want) / max(np.max(np.abs(want)), 1e-12)))
        print(f"паритет значений  : rel={d:.3e}")
        per_block = t_gpu
    else:
        per_block = t_np

    print()
    for label, mult in (("один (док, объект)", 1), ("весь null-слой (53×3)", POOL_DOCS * OBJECTS)):
        t = per_block * mult
        print(f"{label:28s}: {t:8.2f} с  ({t/60:.1f} мин)"
              + (f"   [на 20 свипов: {t*20/60:.0f} мин]" if mult == 1 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
