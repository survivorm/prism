"""Бенчмарк прибора: SYCL (iGPU) ↔ numpy-эталон ↔ torch CPU (если есть).

Цель — не «показать, что GPU быстрее», а найти ТОЧКУ ПЕРЕЛОМА: при каких
размерах/статистиках выигрыш реальный, а при каких пересылка данных съедает
всё (для iGPU «device memory» — это системная RAM, но копирование всё равно
идёт через драйвер).

Запуск:
    python3 bench.py
    python3 bench.py --T 32768 --D 1024      # профиль «32k-игла»

Печатает: время на один (док, слой) для каждой стадии, сумму на типовой прогон
(96 доков × 5 слоёв) и оценку, что это даёт на серии.
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


def timeit(fn, n=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=384)
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--K", type=int, default=96)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    T, D, K = args.T, args.D, args.K
    rng = np.random.default_rng(3)
    u = rng.standard_normal((T, D)).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x = rng.standard_normal((T, D)).astype(np.float32)
    ks = list(range(1, min(K, T - 1) + 1))

    print(f"=== T={T} D={D} K={len(ks)} ===")
    print("backend:", ps.backend())
    if ps.available() and not ps._is_reference():
        print("устройство:", ps.device_info()["name"], "|", ps.device_info()["backend"])

    t_ref_mean = timeit(lambda: ref.dcurve(u.astype(np.float64), ks, "mean"), args.n)
    t_ref_med = timeit(lambda: ref.dcurve(u.astype(np.float64), ks, "median"), args.n)
    t_ref_acf = timeit(lambda: ref.acf_moments(x.astype(np.float64), len(ks)), args.n)
    print(f"numpy(f64)  dcurve mean={t_ref_mean*1e3:8.2f} ms | median={t_ref_med*1e3:8.2f} ms | acf={t_ref_acf*1e3:8.2f} ms")

    if ps.available() and not ps._is_reference():
        t_mean = timeit(lambda: ps.dcurve(u, ks, "mean"), args.n)
        t_med = timeit(lambda: ps.dcurve(u, ks, "median"), args.n)
        t_acf = timeit(lambda: ps.acf_moments(x, len(ks)), args.n)
        print(f"sycl(iGPU)  dcurve mean={t_mean*1e3:8.2f} ms | median={t_med*1e3:8.2f} ms | acf={t_acf*1e3:8.2f} ms")
        print(f"speedup     mean×{t_ref_mean/max(t_mean,1e-9):6.2f}  median×{t_ref_med/max(t_med,1e-9):6.2f}  acf×{t_ref_acf/max(t_acf,1e-9):6.2f}")
        per_doc_layer = t_mean + t_med + t_acf
    else:
        per_doc_layer = t_ref_mean + t_ref_med + t_ref_acf

    # типовой прогон AT0: 96 доков × 5 слоёв × 3 объекта
    cells = 96 * 5 * 3
    total = per_doc_layer * cells
    print(f"\nтиповой прогон (96 доков × 5 слоёв × 3 объекта = {cells} клеток): {total:.2f} с")
    print(f"серия на 20 прогонов: {total*20/60:.1f} мин")

    try:
        import torch
        print("\ntorch:", torch.__version__)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print("torch.xpu доступен:", torch.xpu.get_device_name(0))
    except Exception as e:  # noqa: BLE001
        print("\ntorch недоступен:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
