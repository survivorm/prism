# sycl_prism — SYCL-библиотека линейной алгебры (iGPU)

Прибор для переноса тяжёлых операций prism (AT2c/E-45) на iGPU (Intel Iris Xe,
Level-Zero). C ABI (`src/prism_sycl.cpp`) + python-обёртка на ctypes
(`py/prism_sycl.py`): numpy-входы/выходы, torch-тензор конвертируется прозрачно.

## Quickstart

```python
import numpy as np
import prism_sycl as ps          # без устройства/GPU — ps.available() == False,
                                 # вызовы падут с RuntimeError (PS_ERR_NODEV: «нет SYCL-устройства — <причина>») — не тайный сбой
print(ps.gemm(A32, B32))         # C = A@B, f32, row-major (A [M,K], B [K,N], M≤65536, K≤4096, N≤262144)
lam, V, off = ps.eigjac(A32[None, :, :])   # полный спектр симм. батча [b,n,n] (n≤256):
                                           # vals f64 ПО ВОЗРАСТАНИЮ, vecs f32, offdiag f64 (честная сходимость)
print(np.abs(lam - np.linalg.eigvalsh(A32)).max())   # сверка с numpy (f64-эталон; fp32-шум входа ~1e-3)
```

Сборка: `cd build && make` (icpx -fsycl). Устройство — iGPU: `LD_LIBRARY_PATH`
обязан включать `/opt/intel/oneapi/2026.0/lib` (иначе «No device of requested
type available»). Тесты: `python -m pytest py/test_new_ops.py -q` (без GPU —
автоскип, DEVICE_OK-guard; wide-формы gemm ~1–2 мин на iGPU).

## Операции (контракты — docstrings, полная спека — docs/TZ_sycl_new_ops.md v5.3)

| вызов | выход | что |
|---|---|---|
| `ps.gemm(A, B)` | np f32 [M,N] | C = A@B; hang-check чанкинг по N ВНУТРИ (бюджет 32 GFLOP/launch); детерминизм бит-в-бит |
| `ps.eigjac(A, max_sweeps=12, tol=1e-5, curve=None)` | (vals f64 [b,n], vecs f32 [b,n,n], offdiag f64 [b]) | cyclic Jacobi, полный спектр, n≤256, batch≤65536; vals по возрастанию; curve — per-sweep offdiag матрицы 0 (буфер ≥ max_sweeps) |
| `ps.eigtop(A, r, max_iter)` / `ps.eigbot(A, r, max_iter)` | (vals f64 [b,r], vecs f32 [b,n,r], resid f64 [b]) | power/inverse iteration + двойной MGS, n≤1024, r≤32 (n=1024: r≤7 — local-memory бюджет); λ_j = Rayleigh колонки j (порядок колонок — динамика MGS, не гарантируется: обычно top по убыванию / bot по возрастанию, сравнение — после сортировки); resid — подпространственный остаток (честный отчёт, не критерий стопа) |
| `ps.svd_top(A, r)` | (U f32 [m,r], S f64 [r], Vh f32 [r,n]) | top-r SVD через gemm+eigjac по малой стороне, m/n≤1024; конвенция torch (S descending); вырожденный ранг — σ=0 и V_i=0 явно |

Ошибки — громкие: `PS_ERR_ARG` (лимиты; формы), `PS_ERR_LOCAL` (local-memory
бюджет), `PS_ERR_NODEV` (нет устройства) → python-исключения с сообщением
(форма, причина), не тихий мусор. NaN во входе → NaN в resid/offdiag (сигнал,
не ноль).

Диспетчер для экспериментов: `prism/backend.py` + `PRISM_BACKEND=cpu|sycl`
(дефолт cpu) — один вызов `B.matmul/B.eigh/B.svd_top`, режим в env (core:
`tests/test_backend.py`).
