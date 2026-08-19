"""Jacobian Bridge: projection through vocabulary space + loss computation."""

import hashlib
import os

import torch

_gramian_cache: dict[str, torch.Tensor] = {}
_GRAMIAN_DIR = os.path.join("results", ".gramians")


def jacobian_bridge(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Compute Jacobian bridge loss between large and small models.

    The bridge chain: J_L @ E_L^T @ E_S @ (E_S^T @ E_S)^{-1}.
    The inverse Gramian (E_S^T @ E_S)^{-1} corrects for non-orthogonal
    embeddings, so identity (J_L == J_S, E_L == E_S) yields loss ~0.
    """
    G_inv = _inverse_gramian(E_S)
    J_L_proj = _project_to_vocab(J_L, E_L)
    J_S_target = _project_to_small(J_L_proj, E_S, G_inv)
    mask = _valid_mask(J_S, J_S_target, threshold)

    if mask.sum() == 0:
        return 1.0

    return _weighted_cosine_loss(J_S, J_S_target, probs_L, mask)


def _project_to_vocab(J_L: torch.Tensor, E_L: torch.Tensor) -> torch.Tensor:
    """Project J_L [|U|, D_L] through E_L^T -> [|U|, V]."""
    return J_L.float() @ E_L.float().T


def _project_to_small(
    J_L_proj: torch.Tensor, E_S: torch.Tensor, G_inv: torch.Tensor
) -> torch.Tensor:
    """Project [|U|, V] through E_S @ G_inv -> [|U|, D_S].

    G_inv = (E_S^T @ E_S)^{-1} corrects non-orthogonal embeddings.
    """
    return J_L_proj @ E_S.float() @ G_inv


def _inverse_gramian(E: torch.Tensor) -> torch.Tensor:
    """Compute (E^T @ E)^{-1}, cached on disk per embedding hash."""
    key = _gramian_key(E)
    if key not in _gramian_cache:
        path = os.path.join(_GRAMIAN_DIR, f"{key}.pt")
        if os.path.exists(path):
            _gramian_cache[key] = torch.load(path, weights_only=True)
        else:
            _gramian_cache[key] = _compute_inverse_gramian(E)
            os.makedirs(_GRAMIAN_DIR, exist_ok=True)
            torch.save(_gramian_cache[key], path)
    return _gramian_cache[key].to(E.device)


def _gramian_key(E: torch.Tensor) -> str:
    """Stable hash of an embedding matrix for disk caching.

    Uses first/last rows as representative content — hashing the full
    248k×1024 matrix would be prohibitively expensive.
    """
    header = f"{tuple(E.shape)}|{str(E.device)}".encode()
    data = header + E[0].cpu().numpy().tobytes() + E[-1].cpu().numpy().tobytes()
    h = hashlib.sha256(data).hexdigest()[:16]
    return f"g_{h}"


def _compute_inverse_gramian(E: torch.Tensor) -> torch.Tensor:
    """Compute (E^T @ E)^{-1} via pseudoinverse for numerical stability."""
    G = E.float().T @ E.float()
    return torch.linalg.pinv(G)


def _valid_mask(
    J_S: torch.Tensor, J_target: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Mask rows where either norm < threshold."""
    n1 = J_S.float().norm(dim=1)
    n2 = J_target.float().norm(dim=1)
    return (n1 > threshold) & (n2 > threshold)


def _weighted_cosine_loss(
    J_S: torch.Tensor,
    J_target: torch.Tensor,
    probs: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Weighted cosine dissimilarity loss."""
    cos_sim = torch.nn.functional.cosine_similarity(
        J_S.float()[mask], J_target[mask], dim=1
    )
    w = probs[mask]
    w = w / (w.sum() + 1e-8)
    return (w * (1.0 - cos_sim)).sum().item()
