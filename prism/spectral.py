"""Spectral metrics over already-computed Jacobians / Gramians.

Pure functions: no model, no forward passes, no state. All metrics take a
spectrum tensor (``[K]``) or a matrix and return scalars/tensors, so the same
``eigengap`` works for ``Δ_U(t, t')``, ESG Gramians, and arbitrary
symmetric matrices (paper §6, §8.5, §8.9).

Relationship to ``esg.py``:
- ``esg._spectral_decomposition(G, n_components)`` (eigh, K×K) returns
  top-n eigenvalues + eigenvectors (ROWS = eigenvectors, esg convention).
  Not duplicated here.
- ``spectral.spectrum_from_gramian(G)`` returns the FULL eigenvalue list
  only (no vectors) — ``eigvalsh``, ~2× faster than ``eigh``. Use it when
  only the spectrum is needed (``entropy_effective_rank`` etc.).
- ``spectral.top_r_svd(J, r)`` (SVD, K×D) returns ``(U_r, S_r, V_r)`` for
  holonomy / top-r projections. SVD on K×D is a different operation from
  eigh on the K×K Gramian.

Sign convention: singular/eigenvector signs are LAPACK-version-dependent.
Consumers (holonomy, ε) must use sign-invariant quantities (``|cos|``,
Procrustes residuals).
"""

from __future__ import annotations

import math
import warnings

import torch

__all__ = [
    "top_r_svd",
    "spectrum_from_gramian",
    "eigengap",
    "eigengap_min",
    "entropy_effective_rank",
    "stable_rank",
    "participation_ratio",
    "snr_top1_vs_rest",
    "snr_e14",
    "snr_e14_parts",
    "gauge_share",
    "gauge_share_top1",
]

_LOW_PREC = (torch.float16, torch.bfloat16)


def _check_1d_finite(spectrum: torch.Tensor, name: str = "spectrum") -> None:
    """1-D tensor, no NaN/Inf (index of the first offender in the message)."""
    if spectrum.dim() != 1:
        raise ValueError(f"{name}: expected 1-D tensor, got {spectrum.dim()}-D")
    if spectrum.numel() == 0:
        raise ValueError(f"{name}: empty tensor")
    bad = torch.nonzero(~torch.isfinite(spectrum), as_tuple=False).flatten()
    if bad.numel() > 0:
        raise ValueError(f"{name}: non-finite value at index {int(bad[0])}")


# ---------------------------------------------------------------------------
# SVD primitives
# ---------------------------------------------------------------------------


def top_r_svd(
    J: torch.Tensor, r: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated SVD: ``J ≈ U_r @ diag(S_r) @ V_r``.

    Args:
        J: [K, D] Jacobian (float32/float64; fp16/bf16 is cast to float32
           for SVD numerical stability).
        r: 1 ≤ r ≤ min(K, D).

    Returns:
        (U_r [K, r], S_r [r] descending, V_r [r, D]) with the same dtype
        (post-cast) and device as the input.

    Sign caveat: signs of U_r / V_r depend on the LAPACK version — use
    |cos| / Procrustes residuals, never raw sign comparisons.
    """
    if J.dim() != 2:
        raise ValueError(f"top_r_svd: expected 2-D J, got {J.dim()}-D")
    if not torch.isfinite(J).all():
        bad = int((~torch.isfinite(J)).nonzero(as_tuple=False).flatten()[0])
        raise ValueError(
            f"top_r_svd: J contains NaN/Inf (first at flat index {bad}); "
            "refuse to propagate a corrupted cache silently"
        )
    K, D = J.shape
    if not (1 <= r <= min(K, D)):
        raise ValueError(
            f"top_r_svd: r={r} out of range [1, min(K, D)] = [1, {min(K, D)}]"
        )
    if J.dtype in _LOW_PREC:
        J = J.float()
    U, S, V = torch.linalg.svd(J, full_matrices=False)
    return U[:, :r].contiguous(), S[:r].contiguous(), V[:r, :].contiguous()


def spectrum_from_gramian(G: torch.Tensor, ascending: bool = False) -> torch.Tensor:
    """All eigenvalues of a symmetric Gramian, sorted.

    ``eigvalsh`` (no eigenvectors) computed in float64 for stability on
    ill-conditioned ``Δ_U``; the result is cast back to the input dtype.

    Args:
        G: [K, K] symmetric (PSD NOT required: ``Δ_U(t, t')`` for
           ``t ≠ t'`` is symmetric but can have negative eigenvalues).
        ascending: False (default) = descending λ₁ ≥ λ₂ ≥ … (paper §6
           convention); True = ascending (numerical checks).

    Raises:
        ValueError: non-2-D, non-square, or asymmetric (``atol=1e-6``)
           input — no silent symmetrization, that would hide Δ_U bugs.
    """
    if G.dim() != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"spectrum_from_gramian: expected square 2-D, got {tuple(G.shape)}")
    if not torch.allclose(G, G.T, atol=1e-6):
        raise ValueError("spectrum_from_gramian: G is not symmetric (no silent symmetrization)")
    lam = torch.linalg.eigvalsh(G.double())
    lam = torch.sort(lam, descending=not ascending).values
    return lam.to(G.dtype)


# ---------------------------------------------------------------------------
# Scalar spectral metrics (take a spectrum, not a matrix)
# ---------------------------------------------------------------------------


def eigengap(spectrum: torch.Tensor) -> float:
    """λ₁ − λ₂ (two largest eigenvalues; paper §6).

    Input order does not matter (sorted internally). Negative eigenvalues
    are NOT dropped: for a proper eigengap ``Δ_U`` should be PSD, but the
    function does not enforce it.
    """
    _check_1d_finite(spectrum)
    if spectrum.numel() < 2:
        raise ValueError(f"eigengap: need K >= 2, got {spectrum.numel()}")
    s = torch.sort(spectrum, descending=True).values
    return float(s[0] - s[1])


def eigengap_min(
    spectrum: torch.Tensor,      # [k] eigenvalues (NOT singular values), descending not required
    eps_frac: float = 0.01,      # threshold fraction of λ_1 = max(spectrum)
) -> tuple[float, bool]:
    """Pre-flight [M-4]: min consecutive eigenvalue gap inside top-k.

    Returns (gap_min, well_separated), well_separated = gap_min >= eps_frac
    * spectrum[0] (after sorting descending; so spectrum[0] is λ_1 = max).

    SEMANTICS (docs/E38_fixes.md §1.1): ``well_separated`` is a
    mathematically neutral predicate — the spectrum is well separated
    (no near-degenerate pair). It is NOT the [M-4] verdict: per [M-4]
    ``min_t gap_min < ε`` (i.e. NOT well_separated) is when the non-Abelian
    formalism is applicable. Callers decide the mapping; E-38c reports
    ``fraction_well_separated`` and drives verdicts from
    ``min_t_gap_min_over_threshold`` directly.

    NOT min(eigengap(...)): eigengap returns only the FIRST gap λ₁−λ₂;
    this returns the MINIMUM over ALL consecutive gaps λ_i − λ_{i+1}.
    Same input convention as eigengap/stable_rank/participation_ratio
    (eigenvalues, i.e. σ²), so it can follow spectrum_from_gramian
    directly without a manual sqrt(). Do NOT pass singular values here.

    The ``>=`` (not ``>``) is deliberate: an exact equality
    ``gap_min == threshold`` is classified as well_separated (boundary =
    "acceptable, not degenerate", per [M-4]).

    Args:
        spectrum: [k] eigenvalues (≥ 0), any order, k ≥ 2.
        eps_frac: threshold as a fraction of λ_1; must lie in (0, 1).

    Returns:
        (gap_min, well_separated) as Python (float, bool).

    Warnings:
        RuntimeWarning: spectrum all zero → returns (0.0, False).

    Errors:
        ValueError: k < 2, NaN/Inf (index reported), negative eigenvalue
            (index reported), eps_frac outside (0, 1).
    """
    _check_1d_finite(spectrum, "spectrum")
    if spectrum.numel() < 2:
        raise ValueError(f"eigengap_min: need k >= 2, got {spectrum.numel()}")
    neg = torch.nonzero(spectrum < 0, as_tuple=False).flatten()
    if neg.numel() > 0:
        raise ValueError(
            f"eigengap_min: negative eigenvalue at index {int(neg[0])} "
            f"(spectrum must be PSD eigenvalues)"
        )
    if not (0.0 < eps_frac < 1.0):
        raise ValueError(f"eigengap_min: eps_frac must be in (0, 1), got {eps_frac}")
    s = torch.sort(spectrum, descending=True).values
    if s[0] == 0:
        warnings.warn("eigengap_min: all eigenvalues are zero", RuntimeWarning)
        return 0.0, False
    # Gaps computed in float64: in float32 (eps ≈ 1.2e-7) a gap near
    # eps_frac * λ_1 with eps_frac ≤ 0.01 would sit at the precision floor
    # (absolute error ~1e-7·λ_1). Same policy as spectrum_from_gramian,
    # which already casts to double internally (API symmetry).
    gaps = (s[:-1] - s[1:]).double()
    gap_min = float(gaps.min())
    threshold = float(eps_frac * s[0])
    return gap_min, gap_min >= threshold


def entropy_effective_rank(spectrum: torch.Tensor) -> float:
    """r_eff = exp(−Σ p_k log p_k), p_k = λ_k / Σλ (paper §6, §15; E-14, E-27.6).

    Non-positive eigenvalues are DROPPED (p_k = 0); if all λ ≤ 0 the
    function returns 0.0 with a RuntimeWarning. For ``Δ_U(t, t')`` with
    t ≠ t' (mixed spectrum) r_eff is computed on the positive part only —
    semantically correct (negative "energy" has no meaning) but the value
    can be smaller than naively expected.
    """
    _check_1d_finite(spectrum)
    pos = spectrum[spectrum > 0]
    if pos.numel() == 0:
        warnings.warn(
            "entropy_effective_rank: all eigenvalues <= 0, returning 0.0",
            RuntimeWarning,
        )
        return 0.0
    p = pos / pos.sum()
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy))


def stable_rank(spectrum: torch.Tensor) -> float:
    """Σλ / λ₁ (paper §6).

    "How far the spectrum is from being concentrated in one mode."
    NOT a synonym for participation_ratio: for λ = (4,1,1,1,1),
    stable_rank = 8/4 = 2.0 while PR = 64/20 = 3.2 (TZ said 64/18 ≈ 3.555 — typo, Σλ² = 20).

    Negative eigenvalues are not dropped (ratio definition). λ₁ ≤ 0
    (all zero, or a mixed-sign spectrum where the largest is negative)
    → 0.0 with a RuntimeWarning.
    """
    _check_1d_finite(spectrum)
    s = torch.sort(spectrum, descending=True).values
    lam1 = s[0]
    if lam1 <= 0:
        warnings.warn(
            f"stable_rank: λ₁ = {float(lam1):.3e} <= 0, returning 0.0", RuntimeWarning
        )
        return 0.0
    return float(s.sum() / lam1)


def participation_ratio(spectrum: torch.Tensor) -> float:
    """PR = (Σλ)² / Σλ² ∈ [1, K] (paper §6, §17; E-14).

    Defined on all eigenvalues (negatives not dropped). Σλ² = 0 → 0.0.
    Originally ``_participation_ratio`` in experiments/exp14_snr_pr.py.
    """
    _check_1d_finite(spectrum)
    s2 = spectrum.pow(2).sum()
    if s2 == 0:
        return 0.0
    return float(spectrum.sum().pow(2) / s2)


def snr_top1_vs_rest(spectrum: torch.Tensor) -> float:
    """λ₁ / mean(λ₂..λ_K), λ₁ = largest eigenvalue.

    DEFINITION UNDER REVIEW (TZ docs/new_code_1.md §1.3): the paper §6
    mentions a "leave-one-out SNR" without an exact formula, and the
    original E-14 SNR is sample-based, not spectral (see :func:`snr_e14`).
    This spectral proxy is provided for E-ε; if the original E-14 formula
    turns out to be the one used downstream, prefer ``snr_e14`` and record
    which one is used in ``meta.json``.

    mean(λ_{-1}) = 0 → ``inf``. K = 1 → ValueError. Negatives are not
    dropped. No positive λ → ValueError.
    """
    _check_1d_finite(spectrum)
    if spectrum.numel() < 2:
        raise ValueError(f"snr_top1_vs_rest: need K >= 2, got {spectrum.numel()}")
    s = torch.sort(spectrum, descending=True).values
    lam1 = s[0]
    if lam1 <= 0:
        raise ValueError("snr_top1_vs_rest: no positive eigenvalue (λ₁ <= 0)")
    rest = s[1:].mean()
    if rest == 0:
        return float("inf")
    return float(lam1 / rest)


def snr_e14_parts(
    jacobians: list[torch.Tensor],
) -> tuple[float, float, float]:
    """Original E-14 sample-based SNR with intermediates
    (experiments/exp14_snr_pr.py::``_snr_sample`` — MOVED here, not copied).

    Returns ``(snr, mean_sq, var)`` where
    ``mean_sq = mean‖J‖²_F``, ``var = mean‖J‖²_F − ‖mean J‖²_F`` (per the
    original E-14 definition), and ``snr = ‖mean J‖²_F / max(1e-12, var)``.
    Measures cross-position CONSISTENCY of J, not a spectral property — do
    not confuse with :func:`snr_top1_vs_rest`.

    VALIDATED against results/exp14_snr_pr.json (committed): for all 24
    layers the formula ``(mean_sq − var_sq)/var_sq`` reproduces the stored
    ``SNR`` bit-for-bit (L0 = 0.1415, L21 = 3.6509, L23 = 199.3080 — the
    paper's numbers). See tests/test_spectral.py::test_snr_e14_validated_against_exp14_json.
    """
    if len(jacobians) < 2:
        raise ValueError(f"snr_e14_parts: need >= 2 Jacobians, got {len(jacobians)}")
    J = torch.stack([j.float() for j in jacobians])
    M = J.mean(dim=0)
    signal = M.pow(2).sum().item()
    mean_sq = J.pow(2).sum(dim=(1, 2)).mean().item()
    var = mean_sq - signal
    return float(signal / max(1e-12, var)), float(mean_sq), float(var)


def snr_e14(jacobians: list[torch.Tensor]) -> float:
    """Original E-14 sample-based SNR, scalar form (``snr_e14_parts()[0]``)."""
    return snr_e14_parts(jacobians)[0]


def gauge_share(
    spectrum: torch.Tensor,
    eigenvectors: torch.Tensor,
    uniform: torch.Tensor | None = None,
) -> float:
    """Σ_k (u_kᵀ uniform)² λ_k / Σ_k λ_k (paper §6, "gauge_share ESG-version").

    Fraction of ``JJ^T`` energy lying on the uniform direction in
    token-space.

    Args:
        spectrum: [K] eigenvalues of the Gramian (full spectrum, matching
            ``eigenvectors``).
        eigenvectors: [K, K] with COLUMNS = u_k, the k-th column
            corresponding to ``spectrum[k]``. NOTE: ``esg`` returns
            eigenvectors with ROWS = eigenvectors — pass ``eig.T``.
        uniform: [K] direction; L2-normalized internally. Default
            ``1/√K · ones(K)``.

    The spectrum/eigenvectors pair must cover ALL modes (not top-r) — the
    ``eigenvectors.shape == (K, K)`` check (K = len(spectrum)) enforces this:
    a top-r spectrum paired with a full K×K matrix fails the shape check.
    For top-r use :func:`gauge_share_top1` (or pass the full trace
    explicitly).

    Distinct from ``g_uni`` (paper §8): ``g_uni = ‖QJ‖²_F / ‖J‖²_F`` is
    the energy share of the RAW J; this is the share of ``JJ^T``.

    Σλ = 0 → 0.0 with a RuntimeWarning.
    """
    _check_1d_finite(spectrum)
    K = spectrum.numel()
    if eigenvectors.shape != (K, K):
        raise ValueError(
            f"gauge_share: eigenvectors {tuple(eigenvectors.shape)} "
            f"!= (K, K) = ({K}, {K})"
        )
    if uniform is None:
        uniform = torch.full((K,), 1.0 / math.sqrt(K), device=eigenvectors.device, dtype=eigenvectors.dtype)
    uniform = uniform / uniform.norm()
    coeffs = (eigenvectors.T @ uniform) ** 2  # [K]: (u_k · uniform)^2
    total = spectrum.sum()
    if total == 0:
        warnings.warn("gauge_share: Σλ = 0, returning 0.0", RuntimeWarning)
        return 0.0
    return float((coeffs * spectrum).sum() / total)


def gauge_share_top1(
    spectrum: torch.Tensor, eigenvectors: torch.Tensor
) -> float:
    """λ₁ · (u₁ᵀ uniform)² / Σλ — top-1 proxy (paper §6; E-10a ``gauge_share_0``).

    The value reported in E-10a as ``gauge_share_0`` IS this function
    (top mode only, uniform = 1/√K). ``spectrum[0]`` must be the largest
    eigenvalue (descending convention) and ``eigenvectors[:, 0]`` the
    corresponding eigenvector (columns convention; pass ``eig.T`` from esg).

    Σλ = 0 → 0.0 with a RuntimeWarning.
    """
    _check_1d_finite(spectrum)
    K = spectrum.numel()
    if eigenvectors.shape != (K, K):
        raise ValueError(
            f"gauge_share_top1: eigenvectors {tuple(eigenvectors.shape)} "
            f"!= (K, K) = ({K}, {K})"
        )
    uniform = torch.full((K,), 1.0 / math.sqrt(K), device=eigenvectors.device, dtype=eigenvectors.dtype)
    c1 = (eigenvectors[:, 0] @ uniform) ** 2
    total = spectrum.sum()
    if total == 0:
        warnings.warn("gauge_share_top1: Σλ = 0, returning 0.0", RuntimeWarning)
        return 0.0
    return float(spectrum[0] * c1 / total)
