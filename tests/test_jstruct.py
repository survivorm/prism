"""Tests for prism/jstruct.py (new_fun_core.md §3)."""
from __future__ import annotations

import math

import pytest
import torch

from prism.jstruct import build_M_K, decompose_w_vs_uhat, v_pred_from_JT


# --------------------------------------------------------------------------
# 3.1 build_M_K
# --------------------------------------------------------------------------

def test_build_M_K_basic():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    M = build_M_K(W, [0, 2], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(M, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_build_M_K_scaled_g():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    M = build_M_K(W, [0, 2], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(M, torch.tensor([[2.0, 6.0], [10.0, 18.0]]))


def test_build_M_K_duplicates_allowed():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    M = build_M_K(W, [0, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(M[0], M[1])
    torch.testing.assert_close(M[0], W[0])


def test_build_M_K_index_out_of_range():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError, match="token_set\\[0\\] = 5"):
        build_M_K(W, [5], torch.tensor([1.0, 1.0]))
    with pytest.raises(ValueError, match="token_set\\[1\\] = -1"):
        build_M_K(W, [0, -1], torch.tensor([1.0, 1.0]))


def test_build_M_K_empty_token_set():
    W = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="empty"):
        build_M_K(W, [], torch.tensor([1.0, 1.0]))


def test_build_M_K_shape_mismatch():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="W_lm.shape\\[1\\]"):
        build_M_K(W, [0], torch.tensor([1.0]))


def test_build_M_K_nan_g_raises():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="non-finite g at index 1"):
        build_M_K(W, [0], torch.tensor([1.0, float("nan")]))


# --------------------------------------------------------------------------
# 3.2 decompose_w_vs_uhat
# --------------------------------------------------------------------------

def test_decompose_full_split():
    u = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([0.0, 1.0, 0.0])
    w = 3.0 * u + 4.0 * v
    alpha, beta, v_perp = decompose_w_vs_uhat(w, u)
    assert alpha == pytest.approx(3.0, abs=1e-6)
    assert beta == pytest.approx(4.0, abs=1e-6)
    torch.testing.assert_close(v_perp, v, atol=1e-6, rtol=0)


def test_decompose_parallel():
    u = torch.tensor([0.0, 1.0])
    alpha, beta, v_perp = decompose_w_vs_uhat(torch.tensor([0.0, 1.0]), u)
    assert alpha == pytest.approx(1.0, abs=1e-6)
    assert beta == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(v_perp, torch.zeros(2))


def test_decompose_zero():
    u = torch.tensor([1.0, 0.0])
    alpha, beta, v_perp = decompose_w_vs_uhat(torch.zeros(2), u)
    assert alpha == 0.0
    assert beta == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(v_perp, torch.zeros(2))


def test_decompose_perpendicular():
    u = torch.tensor([1.0, 0.0])
    w = torch.tensor([0.0, 5.0])
    alpha, beta, v_perp = decompose_w_vs_uhat(w, u)
    assert alpha == pytest.approx(0.0, abs=1e-6)
    assert beta == pytest.approx(5.0, abs=1e-6)
    torch.testing.assert_close(v_perp, w / 5.0, atol=1e-6, rtol=0)


def test_decompose_non_unit_uhat_raises():
    with pytest.raises(ValueError, match="not unit"):
        decompose_w_vs_uhat(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))


def test_decompose_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        decompose_w_vs_uhat(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0]))


def test_decompose_nan_w_raises():
    with pytest.raises(ValueError, match="non-finite w at index 1"):
        decompose_w_vs_uhat(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 0.0]))


# --------------------------------------------------------------------------
# 3.3 v_pred_from_JT
# --------------------------------------------------------------------------

def test_v_pred_identity_J():
    J = torch.eye(4)
    v = v_pred_from_JT(J)
    expected = torch.full((4,), 1.0 / math.sqrt(4.0))
    torch.testing.assert_close(v, expected, atol=1e-10, rtol=0)


def test_v_pred_random_direction():
    torch.manual_seed(7)
    J = torch.randn(5, 5, dtype=torch.float64)
    v = v_pred_from_JT(J.double())
    assert float(v.norm()) == pytest.approx(1.0, abs=1e-10)
    expected = J.T @ torch.full((5,), 1.0 / math.sqrt(5.0), dtype=torch.float64)
    cos = float(torch.dot(v, expected / expected.norm()))
    assert cos == pytest.approx(1.0, abs=1e-10)


def test_v_pred_perp_to_uhat_M1_property():
    # J = M_K (I − ûû^T): columns of J lie in û^⊥, so J^T u ∈ û^⊥
    torch.manual_seed(11)
    D = 6
    M_K = torch.randn(4, D, dtype=torch.float64)
    uhat = torch.randn(D, dtype=torch.float64)
    uhat = uhat / uhat.norm()
    P = torch.eye(D, dtype=torch.float64) - torch.outer(uhat, uhat)
    J = M_K @ P
    v = v_pred_from_JT(J)
    assert float(torch.dot(v, uhat).abs()) < 1e-6  # [M-1](2) on synthetic data


def test_v_pred_zero_J_raises():
    J = torch.zeros(3, 4)
    with pytest.raises(RuntimeError, match="M-1"):
        v_pred_from_JT(J)


def test_v_pred_uniform_bad_shape():
    J = torch.eye(3)
    with pytest.raises(ValueError, match="uniform shape"):
        v_pred_from_JT(J, uniform=torch.ones(4))


def test_v_pred_uniform_zero_norm_raises():
    J = torch.eye(3)
    with pytest.raises(ValueError, match="zero norm"):
        v_pred_from_JT(J, uniform=torch.zeros(3))


def test_v_pred_uniform_normalized_silently():
    J = torch.eye(4)
    v = v_pred_from_JT(J, uniform=torch.ones(4))  # not unit; normalized inside
    expected = torch.full((4,), 1.0 / math.sqrt(4.0))
    torch.testing.assert_close(v, expected, atol=1e-10, rtol=0)


def test_v_pred_J_not_2d_raises():
    with pytest.raises(ValueError, match="2-D"):
        v_pred_from_JT(torch.ones(5))
