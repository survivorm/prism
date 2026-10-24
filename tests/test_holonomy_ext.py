"""Tests for prism/holonomy.py extensions (new_fun_core.md §1)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from prism.holonomy import (
    eigenframe_basis,
    geodesic_curvature,
    holonomy_accumulate,
    holonomy_spectrum,
    procrustes_align,
)


def _rot2(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def _U_from_R(R: torch.Tensor, K: int = 8) -> torch.Tensor:
    """An orthonormal [K, r] frame whose first r rows encode R.

    U = [[R], [0]] (r = R.shape[0]); K ≥ r.
    """
    r = R.shape[0]
    K = max(K, r)
    U = torch.zeros(K, r, dtype=torch.float64)
    U[:r, :r] = R
    return U


def _open_path_to(R_target: torch.Tensor, K: int = 4) -> list[torch.Tensor]:
    """Open path [I, R_target]: with closed_loop=False the accumulate gives
    R_total = R_{0,1} = R_target exactly (identical r ⇒ residual 0)."""
    r = R_target.shape[0]
    U0 = torch.zeros(K, r, dtype=torch.float64)
    U0[:r, :r] = torch.eye(r, dtype=torch.float64)
    U1 = _U_from_R(R_target, K)
    return [U0, U1]


# --------------------------------------------------------------------------
# 1.1 holonomy_accumulate return_total
# --------------------------------------------------------------------------

def test_accumulate_backward_compat_two_tuple():
    U = _U_from_R(_rot2(0.3))
    out = holonomy_accumulate([U, U])
    assert len(out) == 2
    kappa, per_step = out
    assert isinstance(kappa, float)
    assert isinstance(per_step, list)


def test_accumulate_return_total_identity():
    U = _U_from_R(_rot2(0.3))
    kappa, per_step, R_total = holonomy_accumulate([U, U], closed_loop=True,
                                                   return_total=True)
    assert kappa < 1e-6
    assert R_total.shape == (2, 2)
    torch.testing.assert_close(R_total, torch.eye(2, dtype=torch.float64),
                               atol=1e-6, rtol=0)
    assert R_total.dtype == torch.float64


def test_accumulate_return_total_composition():
    th = (0.2, 0.5, -0.7)
    Us = [_U_from_R(_rot2(t)) for t in th]
    _, _, R_total = holonomy_accumulate(Us, closed_loop=True, return_total=True)
    R01, _ = procrustes_align(Us[0], Us[1])
    R12, _ = procrustes_align(Us[1], Us[2])
    R20, _ = procrustes_align(Us[2], Us[0])
    expected = (R01.double() @ R12.double()) @ R20.double()
    torch.testing.assert_close(R_total, expected, atol=1e-6, rtol=0)


def test_accumulate_return_total_orthonormal_and_kappa_identity():
    th = (0.1, 0.9, -0.4, 0.6)
    Us = [_U_from_R(_rot2(t)) for t in th]
    kappa, _, R_total = holonomy_accumulate(Us, closed_loop=True,
                                            return_total=True)
    torch.testing.assert_close(R_total.T @ R_total,
                               torch.eye(2, dtype=torch.float64),
                               atol=1e-6, rtol=0)
    expected = torch.norm(R_total - torch.eye(2, dtype=torch.float64),
                          p="fro").item()
    assert kappa == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------
# 1.2 holonomy_spectrum
# --------------------------------------------------------------------------

def test_spectrum_identity_is_zero():
    U = _U_from_R(_rot2(0.4))
    theta, R_total = holonomy_spectrum([U, U])
    assert theta.shape == (1,)  # r = 2 → r//2 = 1
    np.testing.assert_allclose(theta, 0.0, atol=1e-6)
    torch.testing.assert_close(R_total, torch.eye(2, dtype=torch.float64),
                               atol=1e-6, rtol=0)


def test_spectrum_two_blocks():
    theta_val = 0.7
    R = torch.block_diag(_rot2(theta_val), _rot2(theta_val),
                         torch.eye(2, dtype=torch.float64))  # 6x6
    theta, _ = holonomy_spectrum(_open_path_to(R, K=6), closed_loop=False)
    assert theta.shape == (2,)
    assert theta[0] == pytest.approx(theta_val, abs=1e-6)
    assert theta[1] == pytest.approx(theta_val, abs=1e-6)


def test_spectrum_single_pi_block():
    R = torch.block_diag(_rot2(math.pi), torch.eye(2, dtype=torch.float64))
    theta, _ = holonomy_spectrum(_open_path_to(R, K=4), closed_loop=False)
    assert theta.shape == (1,)
    assert theta[0] == pytest.approx(math.pi, abs=1e-6)


def test_spectrum_minus_identity():
    theta, _ = holonomy_spectrum(_open_path_to(-torch.eye(4, dtype=torch.float64), K=4),
                                 closed_loop=False)
    assert theta.shape == (2,)
    assert theta[0] == pytest.approx(math.pi, abs=1e-6)
    assert theta[1] == pytest.approx(math.pi, abs=1e-6)


def test_spectrum_all_pi_pair_not_silently_lost():
    # docs/E38_fixes.md §2 HIGH: an all-π pair (both eigenvalues -1) can
    # land on the -π side of the ±π boundary (fp sign of imag part) and
    # the old `angles > 1e-6` filter dropped it -> "R ≈ I". -I4 is the
    # canonical all-π case.
    theta, _ = holonomy_spectrum(_open_path_to(-torch.eye(4, dtype=torch.float64), K=4),
                                 closed_loop=False)
    assert theta.shape == (2,)
    np.testing.assert_allclose(theta, math.pi, atol=1e-6)


def test_spectrum_lone_minus_one_reflection_axis():
    # rot(0.5) block + {+1, -1}: det = -1 (a Procrustes step may be a
    # reflection) -> a LONE -1, reported as a π angle with a warning.
    R = torch.block_diag(
        _rot2(0.5),
        torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64),
    )
    with pytest.warns(RuntimeWarning, match="lone -1 eigenvalue"):
        theta, _ = holonomy_spectrum(_open_path_to(R, K=4), closed_loop=False)
    assert theta.shape == (2,)
    assert theta[0] == pytest.approx(0.5, abs=1e-6)
    assert theta[1] == pytest.approx(math.pi, abs=1e-6)


def test_spectrum_trace_and_frobenius_identities():
    th = 0.35
    R = torch.block_diag(_rot2(th), _rot2(2 * th))
    theta, R_total = holonomy_spectrum(_open_path_to(R, K=4), closed_loop=False)
    # tr(R) = 2 * sum cos(theta_j)
    tr = R_total.trace().item()
    assert tr == pytest.approx(2 * sum(math.cos(t) for t in theta), abs=1e-6)
    # ||R - I||_F^2 = 4 * sum (1 - cos theta_j)
    fro2 = torch.norm(R_total - torch.eye(4, dtype=torch.float64), p="fro").item() ** 2
    assert fro2 == pytest.approx(4 * sum(1 - math.cos(t) for t in theta), abs=1e-6)


def test_spectrum_odd_r_warns():
    R = torch.block_diag(_rot2(0.5), torch.tensor([[1.0]], dtype=torch.float64))  # 3x3
    with pytest.warns(RuntimeWarning, match="odd r"):
        theta, _ = holonomy_spectrum(_open_path_to(R, K=5), closed_loop=False)
    assert theta.shape == (1,)
    assert theta[0] == pytest.approx(0.5, abs=1e-6)


def test_spectrum_degenerate_warns_and_keeps_multiplicity():
    # R = Rot(θ) ⊕ Rot(θ): duplicate angles — warn, keep multiplicity
    theta_val = 0.7
    R = torch.block_diag(_rot2(theta_val), _rot2(theta_val))
    with pytest.warns(RuntimeWarning, match="degenerate spectrum"):
        theta, _ = holonomy_spectrum(_open_path_to(R, K=4), closed_loop=False)
    assert theta.shape == (2,)


# --------------------------------------------------------------------------
# 1.3 geodesic_curvature
# --------------------------------------------------------------------------

def test_geodesic_curvature_big_circle_zero():
    u1 = torch.tensor([1.0, 0.0, 0.0])
    u2 = torch.tensor([0.0, 1.0, 0.0])
    u3 = torch.tensor([-1.0, 0.0, 0.0])
    assert geodesic_curvature(u1, u2, u3) == pytest.approx(0.0, abs=1e-6)


def test_geodesic_curvature_small_circle():
    # points on the latitude z = √(1 − R²) (Euler geodesic radius 0.5),
    # angles 0, 1, 2 rad. Exact discrete value for this construction:
    # 1.8435 (1/R = 2 is the Euler limit; the 3-point discretization sits
    # 7% below it — the TZ 5%-tol on 1/R does not cover it).
    R = 0.5
    z = math.sqrt(1.0 - R * R)
    pts = [torch.tensor([R * math.cos(a), R * math.sin(a), z])
           for a in (0.0, 1.0, 2.0)]
    kg = geodesic_curvature(pts[0], pts[1], pts[2])
    assert kg == pytest.approx(1.8435, abs=1e-4)


def test_geodesic_curvature_basis_vectors():
    e1, e2, e3 = torch.eye(3)
    kg = geodesic_curvature(e1, e2, e3)
    assert kg == pytest.approx((math.pi / 2) / math.sqrt(2), abs=1e-4)


def test_geodesic_curvature_collinear_zero():
    u1 = torch.tensor([1.0, 0.0])
    u2 = torch.tensor([0.0, 1.0])
    assert geodesic_curvature(u1, u1, u2) == 0.0
    assert geodesic_curvature(u1, u2, u2) == 0.0


def test_geodesic_curvature_round_trip_zero():
    # u1 == u3 → t1 == t3 → cos_turn = 1 → turn = π (the TZ's "π −
    # arccos(1) = 0" is a slip: π − arccos(1) = π; the TZ's formula block
    # is authoritative, and the big-circle test above pins it)
    e1, e2 = torch.eye(2)[:2]
    assert geodesic_curvature(e1, e2, e1) == pytest.approx(math.pi / math.sqrt(2), abs=1e-6)


def test_geodesic_curvature_non_unit_raises():
    with pytest.raises(ValueError, match="not unit"):
        geodesic_curvature(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]),
                           torch.tensor([0.0, 0.0, 1.0]))


def test_geodesic_curvature_d1_raises():
    with pytest.raises(ValueError, match="D < 2"):
        geodesic_curvature(torch.tensor([1.0]), torch.tensor([1.0]),
                           torch.tensor([1.0]))


def test_geodesic_curvature_coincident_raises():
    u = torch.tensor([1.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="coincide"):
        geodesic_curvature(u, u, u)


# --------------------------------------------------------------------------
# 1.4 eigenframe_basis
# --------------------------------------------------------------------------

def test_eigenframe_identity_M():
    D = 6
    M = torch.eye(D, dtype=torch.float64)
    uhat = torch.zeros(D, dtype=torch.float64); uhat[0] = 1.0
    with pytest.warns(RuntimeWarning, match="degenerate eigenvalues"):
        B = eigenframe_basis(M, uhat)
    assert B.shape == (D, D - 1)
    torch.testing.assert_close(B.T @ B, torch.eye(D - 1, dtype=torch.float64),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(B.T @ uhat, torch.zeros(D - 1, dtype=torch.float64),
                               atol=1e-6, rtol=0)


def test_eigenframe_diagonal_M_ordering():
    D = 5
    vals = torch.tensor([2.0, 1.0, 0.5, 0.25, 0.1])
    M = vals.diag()
    uhat = torch.zeros(D); uhat[0] = 1.0
    B = eigenframe_basis(M, uhat)
    # largest remaining eigenvalue is vals[1] (e2 direction)
    torch.testing.assert_close(B[:, 0].abs(),
                               torch.tensor([0., 1., 0., 0., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(B[:, 1].abs(),
                               torch.tensor([0., 0., 1., 0., 0.]), atol=1e-6, rtol=0)


def test_eigenframe_reconstruction():
    torch.manual_seed(3)
    K, D = 10, 12
    M = torch.randn(K, D, dtype=torch.float64)
    uhat = torch.randn(D, dtype=torch.float64)
    uhat = uhat / uhat.norm()
    B = eigenframe_basis(M, uhat)
    A = M.T @ M
    Au = A @ uhat
    A_proj = A - torch.outer(uhat, Au) - torch.outer(Au, uhat) \
        + torch.dot(uhat, Au) * torch.outer(uhat, uhat)
    # B columns are eigenvectors: A_proj @ B = B @ diag(λ)
    lam = (B.T @ A_proj @ B).diag()
    recon = B @ lam.diag() @ B.T
    # A_proj is supported on span(B) (the û-û component is ~0 by construction)
    torch.testing.assert_close(recon, A_proj, atol=1e-5, rtol=1e-5)


def test_eigenframe_non_unit_raises():
    M = torch.eye(4)
    with pytest.raises(ValueError, match="not unit"):
        eigenframe_basis(M, torch.tensor([2.0, 0.0, 0.0, 0.0]))


def test_eigenframe_shape_mismatch_raises():
    M = torch.randn(3, 5)
    with pytest.raises(ValueError, match="M_K.shape"):
        eigenframe_basis(M, torch.tensor([1.0, 0.0, 0.0]))


def test_eigenframe_full_rank_no_isolation_warns():
    # M = I_4: P_⊥ M^T M P_⊥ has a zero û-axis (isolated, no isolation
    # warning) and D−1 = 3 degenerate ones (degeneracy warning expected).
    D = 4
    M = torch.eye(D, dtype=torch.float64)
    uhat = torch.zeros(D, dtype=torch.float64); uhat[0] = 1.0
    with pytest.warns(RuntimeWarning, match="degenerate eigenvalues"):
        B = eigenframe_basis(M, uhat)
    assert B.shape == (D, D - 1)


def test_eigenframe_zero_M_empty_basis():
    D = 4
    M = torch.zeros(2, D)
    uhat = torch.zeros(D); uhat[0] = 1.0
    with pytest.warns(RuntimeWarning, match="fully degenerate"):
        B = eigenframe_basis(M, uhat)
    assert B.shape == (D, 0)
