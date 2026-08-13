"""Tests for bridge module."""

import torch

from prism.bridge import jacobian_bridge


def test_identity_loss_is_zero():
    """Identical Jacobians with orthogonal embedding -> loss ~ 0."""
    # Orthogonal E so J @ E^T @ E = J (projection is identity)
    D = 8
    J = torch.randn(4, D)
    E = torch.eye(D)  # E^T @ E = I
    p = torch.ones(4) / 4
    loss = jacobian_bridge(J, J, E, E, p)
    assert loss < 0.001


def test_antiparallel_loss_is_one():
    """Anti-parallel Jacobians → loss ≈ 1."""
    J = torch.randn(4, 8)
    E = torch.randn(10, 8)
    p = torch.ones(4) / 4
    loss = jacobian_bridge(J, -J, E, E, p)
    assert loss > 0.99


def test_loss_in_range():
    """Loss always >= 0 and <= 2 (cosine dissimilarity bound)."""
    J_L = torch.randn(4, 16)
    J_S = torch.randn(4, 8)
    E_L = torch.randn(100, 16)
    E_S = torch.randn(100, 8)
    p = torch.rand(4)
    p = p / p.sum()
    loss = jacobian_bridge(J_L, J_S, E_L, E_S, p)
    assert -0.001 <= loss <= 2.001


def test_decayed_gradients_masked():
    """Rows with norm < threshold are excluded."""
    J_L = torch.randn(4, 8)
    J_S = torch.zeros(4, 8)  # all decayed
    E_L = torch.randn(10, 8)
    E_S = torch.randn(10, 8)
    p = torch.ones(4) / 4
    loss = jacobian_bridge(J_L, J_S, E_L, E_S, p)
    # All rows masked → max loss
    assert loss >= 0.9


def test_different_dimensions():
    """Bridge works when D_L != D_S."""
    J_L = torch.randn(6, 256)
    J_S = torch.randn(6, 64)
    E_L = torch.randn(1000, 256)
    E_S = torch.randn(1000, 64)
    p = torch.ones(6) / 6
    loss = jacobian_bridge(J_L, J_S, E_L, E_S, p)
    assert -0.001 <= loss <= 2.001
