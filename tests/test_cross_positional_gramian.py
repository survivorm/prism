"""Tests for cross_positional_gramian (v6.0, docs/changed_functions.md §3.4)."""

import pytest

import torch

from prism.esg import compute_esg, cross_positional_gramian


def test_identity_with_esg_at_t_eq_t():
    """Δ_U(t, t) is spectrally equal to compute_esg([J_t]) for N=1."""
    torch.manual_seed(0)
    K, D = 8, 16
    J = torch.randn(K, D)
    G = cross_positional_gramian(J, J)
    ev, _ = compute_esg([J], n_components=K)
    ref = torch.linalg.eigvalsh(G).flip(0)  # eigh ascending -> descending
    assert torch.allclose(ev, ref, atol=1e-6)


def test_symmetry_cross_position():
    """Δ_U(t, t').T == Δ_U(t', t)."""
    torch.manual_seed(1)
    J1, J2 = torch.randn(8, 16), torch.randn(8, 16)
    assert torch.allclose(
        cross_positional_gramian(J1, J2).T, cross_positional_gramian(J2, J1)
    )
    # and the self-gramian is symmetric
    G = cross_positional_gramian(J1, J1)
    assert torch.allclose(G, G.T, atol=1e-6)


def test_shape_device_dtype_validation():
    """_validate rejects shape / device mismatch (duplicating esg tests, per TZ)."""
    J = torch.randn(8, 16)
    with pytest.raises(ValueError):
        cross_positional_gramian(J, torch.randn(9, 16))
    with pytest.raises(ValueError):
        cross_positional_gramian(J, torch.randn(8, 16, device="meta"))


def test_float32_output():
    """Output is float32 regardless of input dtype (stability for eigvalsh)."""
    J = torch.randn(8, 16, dtype=torch.float64)
    G = cross_positional_gramian(J, J)
    assert G.dtype == torch.float32
    if torch.cuda.is_available():
        J16 = torch.randn(8, 16, dtype=torch.float16, device="cuda")
        assert cross_positional_gramian(J16, J16).dtype == torch.float32
