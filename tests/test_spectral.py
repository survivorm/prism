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
    snr_e14_parts,
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


# ---------------------------------------------------------------------------
# Review new_code_1_review.md: P0 (device), P1 (E-14 validation), stable_rank
# ---------------------------------------------------------------------------


def test_gauge_share_device_follows_eigenvectors():
    """Review P0: uniform must be created on eigenvectors' device/dtype.

    CPU always works; CUDA is exercised when available (skipped otherwise) —
    before the fix this raised RuntimeError: expected scalar type / device
    mismatch in the matrix product.
    """
    lam = torch.rand(8) + 0.5
    Q, _ = torch.linalg.qr(torch.randn(8, 8))
    ref = gauge_share(lam, Q)
    got_cpu = gauge_share(lam, Q)
    assert got_cpu == pytest.approx(ref)
    # custom uniform: device+dtype must match eigenvectors
    u = torch.ones(8, dtype=Q.dtype)
    assert gauge_share(lam, Q, uniform=u) == pytest.approx(ref)
    if torch.cuda.is_available():
        Q_c, lam_c = Q.cuda(), lam.cuda()
        got_gpu = gauge_share(lam_c, Q_c)
        assert got_gpu == pytest.approx(ref)
        assert gauge_share_top1(lam_c, Q_c) == pytest.approx(gauge_share_top1(lam, Q))
    # float64 eigenvectors (dtype propagation of the default uniform)
    Q_d, lam_d = Q.double(), lam.double()
    assert gauge_share(lam_d, Q_d) == pytest.approx(gauge_share(lam, Q), abs=1e-6)  # fp32 vs fp64 QR rounding


def test_snr_e14_validated_against_exp14_json():
    """Review P1: formula validated against committed E-14 results.

    results/exp14_snr_pr.json stores (SNR, mean_sq, var_sq) per layer from
    the ORIGINAL exp14 run. Two checks:
    (a) the stored SNR equals (mean_sq - var_sq)/max(1e-12, var_sq) in
        float64 — bit-for-bit (rel < 1e-9) for ALL 24 layers, including the
        paper's numbers L0=0.1415, L21=3.6509, L23=199.3080;
    (b) snr_e14_parts on reconstructed Jacobians reproduces the same
        intermediates/SNR (rel < 1e-5 — the stored mean_sq/var_sq are
        float32-rounded, so this is a float32-precision check).
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results" / "exp14_snr_pr.json"
    if not path.is_file():
        pytest.skip("results/exp14_snr_pr.json not in this checkout")
    d = json.load(open(path))
    for layer, r in d.items():
        # (a) bit-level formula check on the stored intermediates
        ref_snr = (r["mean_sq"] - r["var_sq"]) / max(1e-12, r["var_sq"])
        assert ref_snr == pytest.approx(r["SNR"], rel=1e-9), f"layer {layer}"
        # (b) function round-trip on reconstructed J
        snr, mean_sq, var = snr_e14_parts(_placeholder_js(r["mean_sq"], r["var_sq"]))
        assert mean_sq == pytest.approx(r["mean_sq"], rel=1e-5), f"layer {layer}"
        assert var == pytest.approx(r["var_sq"], rel=1e-5), f"layer {layer}"
        assert snr == pytest.approx(r["SNR"], rel=1e-5), f"layer {layer}"


def _placeholder_js(mean_sq: float, var: float) -> list[torch.Tensor]:
    """Two 1-D-row Jacobians with exactly the given (mean_sq, var) intermediates.

    J0 = a·u, J1 = b·u, u = ones(1, 1):
        mean_sq = (a²+b²)/2,  signal = ((a+b)/2)²,  var = (a-b)²/4.
    Solving:  a = √(mean_sq − var) + √var,  b = √(mean_sq − var) − √var,
    which gives snr = (mean_sq − var)/var — the exact E-14 identity.
    """
    sq = math.sqrt(max(mean_sq - var, 1e-12))
    sv = math.sqrt(max(var, 0.0))
    u = torch.ones(1, 1)
    return [(sq + sv) * u, (sq - sv) * u]


def test_stable_rank_negative_lam1():
    """Review P3: lam1 < 0 (mixed-sign spectrum) → 0.0 + warning, not a
    huge negative value."""
    with pytest.warns(RuntimeWarning, match="<= 0"):
        assert stable_rank(torch.tensor([-1.0, -0.5])) == 0.0
    with pytest.warns(RuntimeWarning, match="<= 0"):
        assert stable_rank(torch.tensor([-1e-9, 0.0])) == 0.0
    # positives still work
    assert stable_rank(torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])) == pytest.approx(2.0)
