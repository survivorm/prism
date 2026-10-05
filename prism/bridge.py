"""Jacobian Bridge: projection through vocabulary space + loss computation."""

import hashlib
import os

import torch

_gramian_cache: dict[str, torch.Tensor] = {}
_GRAMIAN_DIR = os.path.join("results", ".gramians")


def jacobian_gap(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    threshold: float = 0.0,
) -> float:
    """Compute the Jacobian gap (alignment loss, LOWER = BETTER) between large and small models.

    Terminology (v4.0): gap = weighted (1 - cos), lower=better; bridge = weighted
    cos, higher=better; gap = 1 - bridge at equal weights. Callers reporting
    "bridge" (E-16a/b) invert: bridge = 1 - jacobian_gap(...). The legacy
    `jacobian_bridge` alias (which silently returned the GAP direction despite
    its name) was removed after being identified as an inversion trap — the
    inversion lives in the experiment kernels (kaggle_kernel/kernel-exp1*/...py),
    not here.

    The bridge chain: J_L @ E_L^T @ E_S @ (E_S^T @ E_S)^{-1}.
    The inverse Gramian (E_S^T @ E_S)^{-1} (disk-cached, see _inverse_gramian)
    corrects for non-orthogonal embeddings, so identity (J_L == J_S,
    E_L == E_S) yields gap ~0.

    threshold: row mask on BOTH norms, (n1 > threshold) & (n2 > threshold).
    Default 0.0 = sentinel-only mode: only exactly-zero rows are masked
    (no decay filtering). Historical default was 1e-6; the paper's
    relative quantile tau_l = 0.01 * median_i ||J_S[i]|| (E-6 calibration,
    ~1e-4 at L18) can be passed explicitly to enable decay masking.
    """
    G_inv = _inverse_gramian(E_S)
    J_L_proj = _project_to_vocab(J_L, E_L)
    J_S_target = _project_to_small(J_L_proj, E_S, G_inv)
    mask = _valid_mask(J_S, J_S_target, threshold)

    if mask.sum() == 0:
        return 1.0

    return _weighted_cosine_loss(J_S, J_S_target, probs_L, mask)


def _project_to_vocab(J_L: torch.Tensor, E_L: torch.Tensor) -> torch.Tensor:
    """Project J_L [K, D_L] through E_L^T -> [K, V]."""
    return J_L.float() @ E_L.float().T


def _project_to_small(
    J_L_proj: torch.Tensor, E_S: torch.Tensor, G_inv: torch.Tensor
) -> torch.Tensor:
    """Project [K, V] through E_S @ G_inv -> [K, D_S].

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

    Hashes shape/device header + first, last, and 10 deterministic mid-rows
    (uniformly spaced, seeded from row count so the sample is stable across
    runs but covers the interior). First/last rows alone would collide for
    matrices that share boundary rows (e.g. PAD/UNK) but differ internally.
    Hashing the full 248k×1024 matrix would be prohibitively expensive.
    """
    V = E.shape[0]
    idx = [0, V - 1]
    if V > 4:
        step = (V - 1) // 10
        idx += sorted({(i * step) % (V - 1) + 1 for i in range(1, 10)})
    header = f"{tuple(E.shape)}|{str(E.device)}".encode()
    data = header + E[0].cpu().numpy().tobytes() + E[-1].cpu().numpy().tobytes()
    for i in sorted(set(idx)):
        data += E[i].cpu().numpy().tobytes()
    h = hashlib.sha256(data).hexdigest()[:16]
    return f"g_{h}"


def _compute_inverse_gramian(E: torch.Tensor) -> torch.Tensor:
    """Compute (E^T @ E)^{-1} via pseudoinverse for numerical stability."""
    G = E.float().T @ E.float()
    return torch.linalg.pinv(G)


def _valid_mask(
    J_S: torch.Tensor, J_target: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Mask rows where either norm <= threshold (sentinel rows).

    threshold=0.0 (default): only exactly-zero rows are masked.
    threshold>0: decay masking (absolute threshold; the paper's relative
    tau_l = 0.01*median is applied by the caller, E-6 calibration).
    """
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
