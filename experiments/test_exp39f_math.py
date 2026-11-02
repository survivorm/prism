"""Мини-проверка математики E-39f против эталона E-39b (малые размеры, CPU, ~1 с).

Ломается, если разъедутся сборки J, определение κ или аналитический drift:
  1. `_J_from_u(M, û, c)` == `exp39b._recompute_J(M, H)`;
  2. `median(_kappa_per_position(J))` == `exp39b._kappa_from_J(J)` (то же число, что
     в гейтах E-39b/E-38d);
  3. `_drift_dMK` (аналитически из ΔM_K) == ‖J₁−J₀‖_F/‖J₀‖_F по позициям.

Запуск: .venv/bin/python experiments/test_exp39f_math.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exp39b_ptq_perturbation as B  # noqa: E402
import exp39f_full_tied as F  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    n, S, K, Dd, r = 3, 12, 16, 64, 8
    M0 = torch.randn(K, Dd)
    M1 = M0 + 0.1 * torch.randn(K, Dd)
    H = torch.randn(n, S, Dd)

    U, c = F._hidden_to_u(H)
    J0 = F._J_from_u(M0, U, c)
    d1 = (J0 - B._recompute_J(M0, H)).abs().max().item()
    assert d1 < 1e-5, f"_J_from_u vs _recompute_J: max |Δ| = {d1:.2e}"

    kp = F._kappa_per_position(J0, r, dev)
    kb = B._kappa_from_J(J0, r, dev)
    assert kp.shape == (n, S - 2), kp.shape
    assert abs(float(np.median(kp)) - kb) < 1e-9, (
        f"κ: median(_kappa_per_position) = {np.median(kp):.10f} vs "
        f"_kappa_from_J = {kb:.10f}")

    J1 = F._J_from_u(M1, U, c)
    ref = (torch.linalg.norm(J1 - J0, ord="fro", dim=[2, 3])
           / torch.linalg.norm(J0, ord="fro", dim=[2, 3])).numpy()
    d3 = np.abs(F._drift_dMK(M0, M1, U, c) - ref).max()
    assert d3 < 1e-6, f"_drift_dMK vs явная разность J: max |Δ| = {d3:.2e}"

    beta_f, nf = F._beta_sigma1(M0, U)
    beta_b, nb = B._beta_sigma1(M0, H)
    assert abs(beta_f - beta_b) < 1e-6 and nf == nb, (
        f"β/σ₁: {beta_f} vs {beta_b}")

    # _M_K_rows (без fp64-копии всей матрицы) == build_M_K на полной матрице
    gg = torch.randn(Dd)
    ts = list(range(7, 7 + K))
    Wbig = torch.randn(K + 20, Dd)
    d4 = (F._M_K_rows(Wbig, ts, gg) - F.build_M_K(Wbig, ts, gg)).abs().max().item()
    assert d4 < 1e-5, f"_M_K_rows vs build_M_K: max |Δ| = {d4:.2e}"

    print(f"OK: J Δ={d1:.2e}  κ={np.median(kp):.6f}(≡E-39b)  "
          f"drift Δ={d3:.2e}  β/σ₁={beta_f:.4f}  M_K Δ={d4:.2e}")


if __name__ == "__main__":
    main()
