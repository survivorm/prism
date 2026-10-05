"""Tests for ESG module (pooling semantics, validation, per-sample mode)."""

import pytest

import torch

from prism.esg import compute_esg, compute_esg_per_sample


def test_esg_single_jacobian_matches_gramian_eigh():
    torch.manual_seed(0)
    J = torch.randn(16, 8)
    ev, evec = compute_esg([J], n_components=16)
    G = J @ J.T
    ref = torch.linalg.eigvalsh(G).flip(0)
    assert torch.allclose(ev, ref, atol=1e-5)
    # eigenvectors span the right space
    assert evec.shape == (16, 16)


def test_esg_pool_is_mean_of_gramians():
    torch.manual_seed(1)
    Js = [torch.randn(12, 8) for _ in range(3)]
    ev, _ = compute_esg(Js, n_components=12)
    G = sum(J.float() @ J.float().T for J in Js) / 3
    ref = torch.linalg.eigvalsh(G).flip(0)
    assert torch.allclose(ev, ref, atol=1e-5)


def test_esg_per_sample_equals_independent_pools():
    torch.manual_seed(2)
    samples = [[torch.randn(12, 8) for _ in range(3)] for _ in range(4)]
    evs, evecs = compute_esg_per_sample(samples, n_components=8)
    assert len(evs) == 4
    for i, Js in enumerate(samples):
        ev_ref, _ = compute_esg(Js, n_components=8)
        assert torch.allclose(evs[i], ev_ref)
    # no cross-sample mixing: per-sample eigenvalues differ across samples
    assert not torch.allclose(evs[0], evs[1])


def test_esg_rejects_empty_list():
    with pytest.raises(ValueError):
        compute_esg([])
    with pytest.raises(ValueError):
        compute_esg_per_sample([])


def test_esg_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_esg([torch.randn(12, 8), torch.randn(10, 8)])


def test_esg_rejects_device_mismatch():
    with pytest.raises(ValueError):
        compute_esg([torch.randn(12, 8), torch.randn(12, 8, device="meta")])
