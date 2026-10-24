"""Tests for prism/spectral.py extensions (new_fun_core.md §2.1)."""
from __future__ import annotations

import pytest
import torch

from prism.spectral import eigengap_min


def test_eigengap_min_basic_well_separated():
    s = torch.tensor([100.0, 81.0, 64.0, 49.0])  # σ² for σ=[10,9,8,7]
    gap, ok = eigengap_min(s, eps_frac=0.01)
    assert gap == pytest.approx(15.0)
    assert ok is True  # threshold = 1.0, 15 >= 1


def test_eigengap_min_large_eps_not_well_separated():
    s = torch.tensor([100.0, 81.0, 64.0, 49.0])
    gap, ok = eigengap_min(s, eps_frac=0.2)
    assert gap == pytest.approx(15.0)
    assert ok is False  # threshold = 20.0


def test_eigengap_min_boundary_ge_true():
    # exact equality gap_min == threshold → well_separated (the deliberate >=)
    s = torch.tensor([10.0, 9.0, 8.0])
    gap, ok = eigengap_min(s, eps_frac=0.1)
    assert gap == pytest.approx(1.0)
    assert ok is True


def test_eigengap_min_boundary_ge_false():
    s = torch.tensor([10.0, 9.0, 8.0])
    gap, ok = eigengap_min(s, eps_frac=0.101)  # threshold = 1.01
    assert gap == pytest.approx(1.0)
    assert ok is False


def test_eigengap_min_unsorted_input():
    # input order must not matter
    a = eigengap_min(torch.tensor([49.0, 64.0, 100.0, 81.0]), eps_frac=0.01)
    b = eigengap_min(torch.tensor([100.0, 81.0, 64.0, 49.0]), eps_frac=0.01)
    assert a == b


def test_eigengap_min_tiny_gap():
    s = torch.tensor([100.02, 100.0, 99.98], dtype=torch.float64)
    gap, ok = eigengap_min(s, eps_frac=0.01)
    assert gap == pytest.approx(0.02, abs=1e-6)
    assert ok is False  # threshold ≈ 1.0002


def test_eigengap_min_duplicate_eigenvalues():
    s = torch.tensor([100.0, 25.0, 25.0])
    gap, ok = eigengap_min(s, eps_frac=0.01)
    assert gap == pytest.approx(0.0)
    assert ok is False


def test_eigengap_min_k1_raises():
    with pytest.raises(ValueError, match="k >= 2"):
        eigengap_min(torch.tensor([10.0]))


def test_eigengap_min_negative_raises():
    with pytest.raises(ValueError, match="negative eigenvalue at index 1"):
        eigengap_min(torch.tensor([10.0, -1.0, 5.0]))


def test_eigengap_min_eps_frac_bounds_raise():
    with pytest.raises(ValueError, match="eps_frac"):
        eigengap_min(torch.tensor([10.0, 9.0]), eps_frac=0.0)
    with pytest.raises(ValueError, match="eps_frac"):
        eigengap_min(torch.tensor([10.0, 9.0]), eps_frac=1.0)


def test_eigengap_min_nan_raises():
    with pytest.raises(ValueError, match="non-finite value at index 1"):
        eigengap_min(torch.tensor([10.0, float("nan"), 5.0]))


def test_eigengap_min_all_zero_warns():
    with pytest.warns(RuntimeWarning, match="all eigenvalues are zero"):
        gap, ok = eigengap_min(torch.tensor([0.0, 0.0, 0.0]))
    assert gap == 0.0
    assert ok is False
