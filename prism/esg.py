"""Empirical Sensitivity Gramian (ESG) computation.

Pooling semantics (paper v4.0, §6 vs §15 E-0.1):
- `compute_esg` pools Jacobians of ONE trajectory (positions of a single
  document). Intra-sample coherence C_a is 8-41x the noise floor F (E-27.6),
  so pooling positions of one sample is valid.
- Cross-SAMPLE pooling of Jacobians is INVALID (E-0.1: C_b ≈ 0.003-0.05,
  near-orthogonal): different samples almost-orthogonal J's average to noise.
  Aggregate across samples ONLY at the scalar-metric level (lambda, PR, SNR,
  r_eff), never at the matrix level. Use `compute_esg_per_sample`.

**Cross-positional Gramian (`cross_positional_gramian`).** `Δ_U(t, t') = J_t J_{t'}^T`
— token correlations between positions. For `t = t'` it coincides with the
per-position ESG. Used in `E-ε` (spectral Goodhart) and as the baseline for
`E-T2'` (holonomy). Do not confuse with `compute_esg`, which pools along a
trajectory.
"""

import torch


def _validate(jacobians: list[torch.Tensor]) -> None:
    if not jacobians:
        raise ValueError("jacobians: empty list")
    ref = jacobians[0]
    for J in jacobians[1:]:
        if J.shape != ref.shape:
            raise ValueError(f"jacobians: shape mismatch {tuple(J.shape)} vs {tuple(ref.shape)}")
        if J.device != ref.device:
            raise ValueError(f"jacobians: device mismatch {J.device} vs {ref.device}")


def compute_esg(
    jacobians: list[torch.Tensor],
    n_components: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ESG from a list of Jacobian matrices.

    Pooling is INTRA-TRAJECTORY: all J_i must be positions of ONE sample
    (document). Cross-sample pooling is invalid — see module docstring /
    paper §15 (E-0.1); aggregate across samples at the scalar-metric level.

    Args:
        jacobians: list of N Jacobian matrices (one trajectory), each [K, D].
        n_components: number of top eigenvalues/vectors to return.

    Returns:
        eigenvalues: [min(n_components, K)] sorted descending (K < n_components
        possible in degenerate cases; in practice K = |𝒮| ≥ 32).
        eigenvectors: [min(n_components, K), K] corresponding eigenvectors.
    """
    _validate(jacobians)
    G = _build_gramian(jacobians)
    eigenvalues, eigenvectors = _spectral_decomposition(G, n_components)
    return eigenvalues, eigenvectors


def compute_esg_per_sample(
    sample_jacobians: list[list[torch.Tensor]],
    n_components: int = 32,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Sample-local ESG: one Gramian per sample, NO cross-sample pooling.

    Args:
        sample_jacobians: list over samples; each entry is a list of the
            sample's per-position Jacobians [K, D].
        n_components: number of top eigenvalues/vectors to return.

    Returns:
        eigenvalues: list [n_samples] of [n_components].
        eigenvectors: list [n_samples] of [n_components, K].

    Cross-sample aggregation happens on scalars (lambda, PR, SNR, r_eff),
    never on the matrices (E-0.1: cross-sample J's are near-orthogonal).
    """
    if not sample_jacobians:
        raise ValueError("sample_jacobians: empty list")
    eigenvalues = []
    eigenvectors = []
    for Js in sample_jacobians:
        ev, evec = compute_esg(Js, n_components=n_components)
        eigenvalues.append(ev)
        eigenvectors.append(evec)
    return eigenvalues, eigenvectors


def cross_positional_gramian(
    J_t: torch.Tensor,
    J_tp: torch.Tensor,
) -> torch.Tensor:
    """Cross-positional Gramian Δ_U(t, t') = J_t @ J_tp.T.

    Args:
        J_t:  [K, D] gauge-free Jacobian at position t.
        J_tp: [K, D] gauge-free Jacobian at position t'.

    Returns:
        [K, K] Gramian (symmetric iff t == t').

    Note:
        - NOT a pooling operation: each call uses exactly two positions.
          See module docstring for intra-trajectory pooling (compute_esg)
          and cross-sample pooling validity (E-0.1).
        - When t == t', result equals the inner-product part of
          compute_esg([J_t]) for N=1: compute_esg([J_t])[0][0] ==
          eigenvalue_0(Δ_U(t, t)). Checked in selfcheck.
        - Caller must ensure J_t, J_tp come from the same model / edit /
          token_set; the function does not check semantic compatibility,
          only shape / device / dtype (via _validate).
    """
    _validate([J_t, J_tp])
    return J_t.float() @ J_tp.float().T


def _build_gramian(jacobians: list[torch.Tensor]) -> torch.Tensor:
    """G = (1/N) * sum_i (J_i @ J_i^T)  (intra-trajectory pool)."""
    U = jacobians[0].shape[0]
    device = jacobians[0].device
    G = torch.zeros(U, U, device=device, dtype=torch.float)

    for J in jacobians:
        G += J.float() @ J.float().T
    G /= len(jacobians)
    return G


def _spectral_decomposition(
    G: torch.Tensor, n_components: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigen decomposition, sorted descending."""
    eigenvalues, eigenvectors = torch.linalg.eigh(G)

    # Sort descending
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components].T

    return eigenvalues, eigenvectors
