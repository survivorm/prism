"""Tests for prism/spectral.py (docs/new_code_1.md §1.6 selfcheck).

Deviation from TZ §1.6.7: the TZ asserts
    gauge_share(spectrum=[1]*K, eigenvectors=eye(K), uniform=1/√K) ≈ 1.0
which is mathematically wrong: with eigenvectors = standard basis and a
UNIFORM vector, each (u_k·uniform)² = 1/K, so the weighted sum is
Σ_k (1/K)·λ_k / Σλ = 1/K, not 1.0. The value 1.0 is reached when the
GRAMIAN itself has uniform as its dominant eigenvector (tested separately
in test_gauge_share_uniform_is_dominant). The direct-formula cross-check
from the TZ IS implemented (test_gauge_share_matches_direct_formula).
"""

import math

import pytest
import torch

from prism.esg import compute_esg
from prism.spectral import (
    entropy_effective_rank,
    eigengap,
    gauge_share,
    gauge_share_top1,
    participation_ratio,
    snr_e14,
    snr_top1_vs_rest,
    stable_rank,
    spectrum_from_gramian,
    top_r_svd,
)


# ---------------------------------------------------------------------------
# SVD primitives
# ---------------------------------------------------------------------------


def test_top_r_svd_matches_full_svd_and_low_residual():
    """TZ §1.6.1: truncated SVD agrees with torch.linalg.svd up to sign,
    and ‖J − USV‖_F / ‖J‖_F is small (rank-r approximation)."""
    torch.manual_seed(0)
    J = torch.randn(8, 16, dtype=torch.float64)
    r = 4
    U, S, V = top_r_svd(J, r)
    assert U.shape == (8, r) and S.shape == (r,) and V.shape == (r, 16)
    assert torch.all(S[:-1] >= S[1:])
    Uf, Sf, Vf = torch.linalg.svd(J, full_matrices=False)
    # top-r components agree up to sign
    for i in range(r):
        assert math.isclose(abs(float((U[:, i] @ Uf[:, i]))), 1.0, abs_tol=1e-10)
        assert math.isclose(abs(float((V[i] @ Vf[i]))), 1.0, abs_tol=1e-10)
        assert math.isclose(float(S[i]), float(Sf[i]), rel_tol=1e-10)
    # relative residual of the rank-r approximation
    resid = torch.linalg.norm(J - U @ torch.diag(S) @ V)
    assert resid / torch.linalg.norm(J) < 0.5


def test_top_r_svd_dtype_and_validation():
    J16 = torch.randn(6, 12, dtype=torch.float16)
    U, S, V = top_r_svd(J16, 3)
    assert U.dtype == torch.float32 and S.dtype == torch.float32
    with pytest.raises(ValueError, match="out of range"):
        top_r_svd(torch.randn(4, 4), r=5)
    with pytest.raises(ValueError, match="2-D"):
        top_r_svd(torch.randn(4, 4, 4), r=2)
    with pytest.raises(ValueError, match="out of range"):
        top_r_svd(torch.randn(4, 4), r=0)


def test_spectrum_from_gramian_matches_svd_squared():
    """TZ §1.6.2: eigvalsh(J Jᵀ) == sorted svd(J).S²."""
    torch.manual_seed(1)
    J = torch.randn(10, 16)
    G = J @ J.T
    spec = spectrum_from_gramian(G)
    ref = torch.sort(torch.linalg.svd(J, full_matrices=False).S**2, descending=True).values
    assert torch.allclose(spec, ref, atol=1e-5)
    # ascending
    spec_a = spectrum_from_gramian(G, ascending=True)
    assert torch.allclose(spec_a, ref.flip(0), atol=1e-5)


def test_spectrum_from_gramian_validation():
    with pytest.raises(ValueError):
        spectrum_from_gramian(torch.randn(4, 5))  # not square
    G = torch.randn(4, 4)
    with pytest.raises(ValueError, match="symmetric"):
        spectrum_from_gramian(G)  # not symmetric
    with pytest.raises(ValueError):
        spectrum_from_gramian(torch.randn(4))  # not 2-D


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def test_eigengap_sorted_and_unsorted():
    """TZ §1.6.3."""
    assert eigengap(torch.tensor([4.0, 2.0, 1.0])) == pytest.approx(2.0)
    assert eigengap(torch.tensor([1.0, 2.0, 4.0])) == pytest.approx(2.0)
    assert eigengap(torch.tensor([1.0, 4.0, 2.0])) == pytest.approx(2.0)


def test_entropy_effective_rank():
    """TZ §1.6.4: uniform rank-4 → 4; rank-1 → 1."""
    assert entropy_effective_rank(torch.tensor([0.25] * 4)) == pytest.approx(4.0, abs=1e-6)
    assert entropy_effective_rank(torch.tensor([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
    # negatives dropped
    assert entropy_effective_rank(torch.tensor([1.0, -1.0, 0.0])) == pytest.approx(1.0, abs=1e-6)
    with pytest.warns(RuntimeWarning):
        assert entropy_effective_rank(torch.tensor([-1.0, -2.0])) == 0.0
    with pytest.raises(ValueError):
        entropy_effective_rank(torch.tensor([]))


def test_stable_rank_vs_participation_ratio():
    """TZ §1.6.5: for (4,1,1,1,1) — stable_rank = 2.0, PR = (Σλ)²/Σλ² = 64/20 = 3.2.

    TZ typo: it says "64/18 ≈ 3.555", but Σλ² = 16+1+1+1+1 = 20, not 18;
    64/20 = 3.2 is the correct value (the distinction from stable_rank = 2.0
    stands either way).
    """
    s = torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])
    assert stable_rank(s) == pytest.approx(2.0)
    assert participation_ratio(s) == pytest.approx(64.0 / 20.0)
    # unsorted input — same
    assert stable_rank(torch.tensor([1.0, 4.0, 1.0, 1.0, 1.0])) == pytest.approx(2.0)
    with pytest.warns(RuntimeWarning):
        assert stable_rank(torch.zeros(3)) == 0.0
    assert participation_ratio(torch.zeros(3)) == 0.0


def test_snr_top1_vs_rest():
    """TZ §1.6.6: [10, 1, 1] → 10 / mean(1, 1) = 10."""
    assert snr_top1_vs_rest(torch.tensor([10.0, 1.0, 1.0])) == pytest.approx(10.0)
    assert snr_top1_vs_rest(torch.tensor([1.0, 10.0, 1.0])) == pytest.approx(10.0)
    assert snr_top1_vs_rest(torch.tensor([5.0, 0.0, 0.0])) == float("inf")
    with pytest.raises(ValueError):
        snr_top1_vs_rest(torch.tensor([5.0]))
    with pytest.raises(ValueError):
        snr_top1_vs_rest(torch.tensor([-1.0, -2.0]))


def test_snr_e14_reproduces_exp14_formula():
    """snr_e14 == ‖mean J‖²/max(1e-12, mean‖J‖²−‖mean J‖²) (exp14 _snr_sample)."""
    torch.manual_seed(2)
    Js = [torch.randn(8, 16) for _ in range(5)]
    M = torch.stack(Js).mean(dim=0)
    mean_sq = sum(J.pow(2).sum().item() for J in Js) / len(Js)
    var = mean_sq - M.pow(2).sum().item()
    ref = M.pow(2).sum().item() / max(1e-12, var)
    assert snr_e14(Js) == pytest.approx(ref, rel=1e-6)  # float32 sum order
    with pytest.raises(ValueError):
        snr_e14([torch.randn(4, 4)])


def test_gauge_share_matches_direct_formula():
    """TZ §1.6.7 (direct-formula half): random orthogonal eigenvectors."""
    torch.manual_seed(3)
    K = 8
    # random orthogonal Q (columns = eigenvectors)
    Q, _ = torch.linalg.qr(torch.randn(K, K))
    lam = torch.rand(K) + 0.5
    uniform = torch.ones(K) / math.sqrt(K)
    got = gauge_share(lam, Q)
    direct = sum(lam[k].item() * (Q[:, k] @ uniform) ** 2 for k in range(K)) / lam.sum().item()
    assert got == pytest.approx(direct, rel=1e-6)
    # TZ literal case (corrected expectation: 1/K, not 1.0 — see module docstring)
    got_eyeq = gauge_share(torch.ones(K), torch.eye(K))
    assert got_eyeq == pytest.approx(1.0 / K, rel=1e-6)


def test_gauge_share_uniform_is_dominant():
    """If the Gramian's top eigenvector IS uniform, gauge_share ≈ 1."""
    K = 8
    u = torch.ones(K) / math.sqrt(K)
    G = 10.0 * torch.outer(u, u)  # rank-1, top eigenvector = uniform
    lam = spectrum_from_gramian(G)
    eig, evec = torch.linalg.eigh(G)
    idx = eig.argsort(descending=True)
    assert gauge_share(lam, evec[:, idx]) == pytest.approx(1.0, abs=1e-6)


def test_gauge_share_validation():
    with pytest.raises(ValueError):
        gauge_share(torch.tensor([1.0, 2.0]), torch.eye(3))  # size mismatch
    with pytest.warns(RuntimeWarning):
        assert gauge_share(torch.zeros(3), torch.eye(3)) == 0.0
    # custom uniform is L2-normalized internally
    lam = torch.tensor([2.0, 1.0])
    v = torch.eye(2)
    assert gauge_share(lam, v, uniform=torch.ones(2)) == pytest.approx(
        gauge_share(lam, v, uniform=torch.ones(2) / math.sqrt(2))
    )


def test_gauge_share_with_esg_convention():
    """Integration: esg returns ROW eigenvectors — pass eig.T (columns)."""
    torch.manual_seed(4)
    Js = [torch.randn(16, 32) for _ in range(6)]
    ev, eig = compute_esg(Js, n_components=16)
    # direct check on the pooled Gramian (esg convention: rows = eigenvectors)
    G = sum(J.float() @ J.float().T for J in Js) / len(Js)
    u = torch.ones(16) / 4.0
    direct = sum(ev[k].item() * (eig[k] @ u) ** 2 for k in range(16)) / ev.sum().item()
    assert gauge_share(ev, eig.T) == pytest.approx(direct, rel=1e-5)
    assert gauge_share_top1(ev, eig.T) == pytest.approx(
        ev[0].item() * (eig[0] @ u) ** 2 / ev.sum().item(), rel=1e-5
    )


def test_gauge_share_top1():
    """TZ §1.6.8: U=eye, spectrum=[1,0,…,0] → 1/K."""
    K = 8
    lam = torch.zeros(K)
    lam[0] = 1.0
    assert gauge_share_top1(lam, torch.eye(K)) == pytest.approx(1.0 / K, rel=1e-6)


def test_nan_handling():
    """TZ §1.6.10: NaN in spectrum → ValueError (catches corrupt cache J)."""
    for fn in (eigengap, entropy_effective_rank, stable_rank,
               participation_ratio, snr_top1_vs_rest):
        with pytest.raises(ValueError, match="non-finite"):
            fn(torch.tensor([1.0, float("nan"), 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        gauge_share(torch.tensor([1.0, float("inf")]), torch.eye(2))
