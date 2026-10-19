"""Selfcheck for prism/holonomy.py (TZ part 2, §3.6).

15 checks: procrustes_align (identity, rotation, sign invariance,
residual formula, non-orthonormal → ValueError), holonomy_accumulate
(trivial, constant rotation open/closed, closed vs open),
holonomy_along_path (constant, linear rotation), holonomy_along_2d_loop
(small epsilon → κ≈0, trivial measure, basis checks, n_theta), and the
fundamental gauge-invariance property (3.6.15).
"""

import math

import pytest
import torch

from prism.holonomy import (
    procrustes_align,
    holonomy_accumulate,
    holonomy_along_path,
    holonomy_along_2d_loop,
)


def _qr(A: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(A)
    return Q


# --- 1-5: procrustes_align ---------------------------------------------------


def test_procrustes_identity():
    U = torch.eye(8)[:, :4]
    R, res = procrustes_align(U, U)
    assert torch.allclose(R, torch.eye(4), atol=1e-6)
    assert abs(res) < 1e-10


def test_procrustes_rotation():
    U = _qr(torch.randn(8, 4))
    Q, _ = torch.linalg.qr(torch.randn(4, 4))
    R, res = procrustes_align(U, U @ Q)
    # R must equal Q up to a column-sign gauge ambiguity of the SVD:
    # the subspace U @ Q is reached from U by exactly Q. For a generic
    # Q the polar factor is unique (no degenerate singular values).
    assert torch.allclose(R, Q, atol=1e-5)
    # residual: fp32 input (U^T U = I to ~1e-7) → ~1e-6 noise, not 0 exactly
    assert abs(res) < 1e-5


def test_procrustes_sign_invariance():
    """Subspaces, not bases: a column sign flip changes R (to the sign
    flip) but residual stays ≈ 0 — the distance is between subspaces."""
    U1 = _qr(torch.randn(8, 4))
    U2 = U1.clone()
    U2[:, 0] = -U2[:, 0]
    R_same, res_same = procrustes_align(U1, U1)
    R_flip, res_flip = procrustes_align(U1, U2)
    assert not torch.allclose(R_flip, R_same, atol=1e-3)  # R differs (≈ diag(-1,1,1,1))
    assert abs(res_flip) < 1e-5                            # ≈ 0 (fp32 input noise ~1e-6)


def test_procrustes_residual_formula():
    U1 = _qr(torch.randn(8, 4))
    U2 = _qr(torch.randn(8, 4))
    R, res = procrustes_align(U1, U2)
    direct = (U1 @ R - U2).norm().pow(2).item()
    assert abs(res - direct) < 1e-6


def test_procrustes_non_orthonormal_raises():
    U1 = _qr(torch.randn(8, 4))
    U2 = U1.clone()
    U2[:, 0] *= 3.0  # breaks orthonormality
    with pytest.raises(ValueError, match="orthonormal"):
        procrustes_align(U1, U2)
    with pytest.raises(ValueError, match="2-D"):
        procrustes_align(torch.randn(8, 4, 2), U1)
    with pytest.raises(ValueError, match="shape mismatch"):
        procrustes_align(U1, _qr(torch.randn(8, 3)))


# --- 6-8: holonomy_accumulate -------------------------------------------------


def test_accumulate_trivial():
    U = _qr(torch.randn(8, 4))
    kappa, per_step = holonomy_accumulate([U, U, U])
    assert kappa < 1e-8
    # per-step: fp32 input (U^T U = I to ~1e-7) → ~1e-6 noise floor
    assert all(r < 1e-6 for r in per_step)
    assert len(per_step) == 3  # closed loop: N steps


def test_accumulate_constant_rotation():
    """U_i = U @ Q^i: pure group orbit.

    OPEN:  κ ≈ ‖Q^{N-1} − I‖ (accumulated transport).
    CLOSED: κ ≈ 0 — the closing step R_{N-1,0} is the EXACT inverse of
    the accumulated product (group-orbit property; TZ §3.6.7's
    'closed κ ≈ ‖Q^n − I‖' is refuted numerically — see docstring of
    holonomy_accumulate). This is the correct semantics.
    """
    U = _qr(torch.randn(8, 4))
    Q, _ = torch.linalg.qr(torch.randn(4, 4))
    n = 4
    U_list = [U]
    cur = torch.eye(4)
    for _ in range(n - 1):
        cur = cur @ Q
        U_list.append(U @ cur)
    kappa_open, _ = holonomy_accumulate(U_list, closed_loop=False)
    kappa_closed, _ = holonomy_accumulate(U_list, closed_loop=True)

    Qm = torch.eye(4)
    for _ in range(n - 1):
        Qm = Qm @ Q
    assert abs(kappa_open - (Qm - torch.eye(4)).norm().item()) < 1e-5
    assert kappa_closed < 1e-6  # group orbit closes exactly


def test_accumulate_closed_vs_open_differ():
    """Genuine GEOMETRIC path (left action: the subspace itself moves in
    ambient space — residuals > 0, best-approximation transitions do not
    telescope): κ(closed) ≠ κ(open).

    Contrast: any sequence of the SAME subspace in different bases
    (right action U @ Q_i, a pure gauge orbit) has κ(closed) ≈ 0 —
    see test_accumulate_constant_rotation.
    """
    U = _qr(torch.randn(8, 4))
    A = torch.randn(8, 8)
    A = (A - A.T) / torch.norm(A, p="fro")  # ambient-space rotation, O(1) angle
    U_list = [torch.matrix_exp(A * i) @ U for i in range(3)]
    k_closed, _ = holonomy_accumulate(U_list, closed_loop=True)
    k_open, _ = holonomy_accumulate(U_list, closed_loop=False)
    assert k_open > 0.1     # the subspace genuinely moved
    assert k_closed > 0.05  # the closing step is not an exact inverse
    assert abs(k_closed - k_open) > 1e-3


def test_accumulate_too_short_raises():
    with pytest.raises(ValueError):
        holonomy_accumulate([torch.eye(4)])


# --- 9-10: holonomy_along_path -------------------------------------------------


def test_along_path_constant():
    U = _qr(torch.randn(8, 4))
    V = _qr(torch.randn(4, 6))
    out = holonomy_along_path(lambda th: (U, V), [0.0, 0.5, 1.0, 2.0])
    assert out["kappa_U"] < 1e-8
    assert out["kappa_V"] < 1e-8
    assert out["thetas"] == [0.0, 0.5, 1.0, 2.0]


def test_along_path_linear_rotation():
    """path(θ) = U @ exp(θA): pure group orbit (A antisymmetric).

    OPEN over θ ∈ [0, 2π): κ ≈ ‖exp(2πA) − I‖.
    CLOSED: κ ≈ 0 (group-orbit property — the closing step is the exact
    inverse; the TZ §3.6.10 closed expectation is refuted, see
    holonomy_accumulate docstring).
    """
    U = _qr(torch.randn(8, 4))
    V = _qr(torch.randn(4, 6))
    A = torch.randn(4, 4)
    A = A - A.T  # antisymmetric → exp(θA) orthogonal
    thetas = [2.0 * math.pi * i / 16 for i in range(16)]  # no 2π endpoint

    def path(th):
        return U @ torch.matrix_exp(th * A), V

    out_open = holonomy_along_path(path, thetas, closed_loop=False)
    # 16 points → 15 steps of Δθ = 2π/16: R_total = exp(15·Δθ·A)
    expected_open = (torch.matrix_exp(15.0 / 16.0 * 2.0 * math.pi * A) - torch.eye(4)).norm().item()
    assert abs(out_open["kappa_U"] - expected_open) < 1e-4

    out_closed = holonomy_along_path(path, thetas, closed_loop=True)
    assert out_closed["kappa_U"] < 1e-5  # orbit closes exactly
    assert out_closed["kappa_V"] < 1e-8  # V constant


def test_along_path_bad_fn_raises():
    with pytest.raises(TypeError, match="path_fn"):
        holonomy_along_path(lambda th: torch.eye(4), [0.0, 1.0])
    with pytest.raises(ValueError, match="thetas"):
        holonomy_along_path(lambda th: (torch.eye(4)[:, :2], torch.eye(4)[:, :2]), [0.0])


# --- 11-14: holonomy_along_2d_loop ----------------------------------------------


def _trivial_measure(U, V):
    return lambda h: (U, V)


def test_2d_loop_small_epsilon():
    D = 16
    base = torch.randn(D)
    Qb = _qr(torch.randn(D, 2))
    b1 = Qb[:, 0].clone()
    b2 = Qb[:, 1].clone()
    U = _qr(torch.randn(8, 4))
    V = _qr(torch.randn(4, 6))
    A = torch.randn(D, 4)

    def measure(h):
        # subspace varies slightly with h (non-trivial but ε-scaled)
        Q = _qr(A + 0.01 * h @ A)
        return Q, V

    out = holonomy_along_2d_loop(base, b1, b2, epsilon=1e-10, n_theta=8,
                                 measure_fn=measure)
    assert out["kappa_U"] < 1e-4  # ~0: the loop is infinitesimal
    assert len(out["h_perturbed"]) == 8


def test_2d_loop_trivial_measure():
    D = 16
    base = torch.randn(D)
    Q = _qr(torch.randn(D, 2))
    b1, b2 = Q[:, 0], Q[:, 1]
    U = _qr(torch.randn(8, 4))
    V = _qr(torch.randn(4, 6))
    out = holonomy_along_2d_loop(base, b1, b2, epsilon=0.1, n_theta=8,
                                 measure_fn=_trivial_measure(U, V))
    assert out["kappa_U"] < 1e-8
    assert out["kappa_V"] < 1e-8


def test_2d_loop_basis_checks():
    D = 8
    base = torch.randn(D)
    b1 = torch.randn(D)
    b1 = b1 / b1.norm()
    b2 = torch.nn.functional.normalize(torch.randn(D, 1)).squeeze()
    m = _trivial_measure(_qr(torch.randn(8, 4)), _qr(torch.randn(4, 6)))
    with pytest.raises(ValueError, match="orthogonal"):
        holonomy_along_2d_loop(base, b1, b1, 0.1, 8, m)
    with pytest.raises(ValueError, match="unit"):
        holonomy_along_2d_loop(base, 2.0 * b1, b2, 0.1, 8, m)


def test_2d_loop_errors():
    D = 8
    base = torch.randn(D)
    Q = _qr(torch.randn(D, 2))
    b1, b2 = Q[:, 0], Q[:, 1]
    m = _trivial_measure(_qr(torch.randn(8, 4)), _qr(torch.randn(4, 6)))
    with pytest.raises(ValueError, match="epsilon"):
        holonomy_along_2d_loop(base, b1, b2, 0.0, 8, m)
    with pytest.raises(ValueError, match="n_theta"):
        holonomy_along_2d_loop(base, b1, b2, 0.1, 3, m)


# --- 15: gauge invariance (fundamental) -----------------------------------------


def test_gauge_invariance():
    """U_i → U_i @ Q_i (orthogonal Q_i) must leave κ unchanged — the
    defining property of holonomy in subspace space."""
    torch.manual_seed(0)
    K, r, n = 12, 4, 6
    U_list = [_qr(torch.randn(K, r)) for _ in range(n)]
    Q_list = [_qr(torch.randn(r, r)) for _ in range(n)]
    U_gauged = [U @ Q for U, Q in zip(U_list, Q_list)]

    k1, _ = holonomy_accumulate(U_list)
    k2, _ = holonomy_accumulate(U_gauged)
    assert abs(k1 - k2) < 1e-6


def test_gauge_invariance_2d_loop():
    D = 12
    torch.manual_seed(1)
    base = torch.randn(D)
    Qb = _qr(torch.randn(D, 2))
    b1, b2 = Qb[:, 0], Qb[:, 1]
    U = _qr(torch.randn(8, 4))
    V = _qr(torch.randn(4, 6))
    Qr = _qr(torch.randn(4, 4))

    m_plain = _trivial_measure(U, V)
    m_gauged = _trivial_measure(U @ Qr, V @ Qr)
    o1 = holonomy_along_2d_loop(base, b1, b2, 0.1, 8, m_plain)
    o2 = holonomy_along_2d_loop(base, b1, b2, 0.1, 8, m_gauged)
    assert abs(o1["kappa_U"] - o2["kappa_U"]) < 1e-6
    assert abs(o1["kappa_V"] - o2["kappa_V"]) < 1e-6
