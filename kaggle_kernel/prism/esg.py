"""Empirical Sensitivity Gramian (ESG) computation."""

import torch


def compute_esg(
    jacobians: list[torch.Tensor],
    n_components: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ESG from a list of Jacobian matrices.

    Args:
        jacobians: list of N Jacobian matrices, each [|U|, D].
        n_components: number of top eigenvalues/vectors to return.

    Returns:
        eigenvalues: [n_components] sorted descending.
        eigenvectors: [n_components, |U|] corresponding eigenvectors.
    """
    G = _build_gramian(jacobians)
    eigenvalues, eigenvectors = _spectral_decomposition(G, n_components)
    return eigenvalues, eigenvectors


def _build_gramian(jacobians: list[torch.Tensor]) -> torch.Tensor:
    """G = (1/N) * sum_i (J_i @ J_i^T)."""
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
