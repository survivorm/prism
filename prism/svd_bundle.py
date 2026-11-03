"""SVD bundle: one place to cut a Jacobian into ``U Σ Vᵀ`` + metadata (ТЗ E40_math_core §II.4, M4).

Consolidates three private copies of principal angles that existed in
``experiments/`` (exp28 ``principal_angles_max``, exp31 ``principal_angle_max``,
exp39a ``_principal_angles_cos``) and attaches the ChannelTag the ТЗ demands.
``slice`` is renamed ``slice_name`` (``slice`` is a python builtin and shadows
it inside the function — deliberate deviation, ТЗ §II.4).
"""

from __future__ import annotations

import numpy as np

from .transport import _check_frame

__all__ = ["svd_bundle", "principal_angles", "effective_rank"]


def effective_rank(Sigma: np.ndarray, threshold: float = 1e-2) -> int:
    """Number of singular values with ``σ_i ≥ threshold · σ_1`` (thresholded PR,
    the ``r_eff^U(τ)`` convention of E-39a — NOT the entropic r_eff of the ESG
    spectrum; always subscript in artifacts, E-40.1 audit A4)."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.ndim != 1 or Sigma.size == 0:
        raise ValueError(f"Sigma must be 1-D, got {Sigma.shape}")
    if not (0 < threshold <= 1):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if Sigma[0] <= 0:
        # A zero spectrum has NO directions above threshold. With σ₁ = 0 the
        # comparison `σ >= threshold·σ₁` is `0 >= 0` and would return the full
        # dimension — a rank-0 Jacobian reported as full rank (audit4 Q1).
        return 0
    return int((Sigma >= threshold * Sigma[0]).sum())


def principal_angles(V1: np.ndarray, V2: np.ndarray, r: int | None = None) -> np.ndarray:
    """Cosines of principal angles between column spaces of ``V1``, ``V2``
    (``[d, r]`` frames): ``σ(eigvals of V1ᵀV2)``, descending. Frames must have
    orthonormal columns (``_check_frame``) — on non-orthonormal inputs the SVD
    values are not angle cosines and clipping to [0,1] would pass the wrong
    object off as a valid one (audit M10). Supersedes the per-experiment
    copies (same math)."""
    V1 = _check_frame(V1, "V1")
    V2 = _check_frame(V2, "V2")
    if V1.shape[0] != V2.shape[0]:
        raise ValueError(f"frames must be [d, r] on a common d, got {V1.shape}, {V2.shape}")
    if r is not None:
        V1, V2 = V1[:, :r], V2[:, :r]
    s = np.linalg.svd(V1.T @ V2, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def svd_bundle(J: np.ndarray, rank: int, side: str = "right",
               slice_name: str = "raw", tag=None) -> dict:
    """Top-``rank`` SVD bundle of a single ``J [K, D]`` with bookkeeping.

    Returns dict with ``U [K, r]``, ``Sigma [r]``, ``V [D, r]`` (columns are
    singular vectors, ``J ≈ U diag(Σ) Vᵀ``), ``energy`` — captured Frobenius
    energy fraction, ``rank_effective`` — thresholded effective rank of the
    FULL spectrum, ``stable`` — False when the top-r block has a σ gap below
    ``1e-2 σ₁`` anywhere (rotation ambiguity, E40_ans §5.2). CONVENTION WARNING
    (audit M9): this gap is in σ units (the E-38c ``gap_min`` convention);
    ``kinetic.rotation_unique`` uses the Davis–Kahan σ² form
    (``min|σᵢ²−σⱼ²| > eps·σ₁²``) — on a spectrum like [1.0, 0.991] the two
    disagree by construction. Quote the convention with the flag.
    ``tag`` — the
    ChannelTag passed in (ТЗ: a number without a passport is invalid).

    ``side`` is metadata for the consumer (which axis of the transport the
    bundle addresses); it does not change the SVD.
    """
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"J must be [K, D], got {J.shape}")
    if side not in ("left", "right", "mixed"):
        raise ValueError(f"side must be left/right/mixed, got {side!r}")
    K, D = J.shape
    if not (0 < rank <= min(K, D)):
        raise ValueError(f"rank must be in (0, min(K,D)={min(K, D)}], got {rank}")
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Vr = Vt[:rank].T
    gaps = S[:rank - 1] / S[0] - S[1:rank] / S[0] if rank > 1 else np.array([1.0])
    return {
        "U": U[:, :rank],
        "Sigma": S[:rank],
        "V": Vr,
        "Sigma_full": S,
        "energy": float((S[:rank] ** 2).sum() / (S ** 2).sum()),
        "rank_effective": effective_rank(S),
        "stable": bool(np.abs(gaps).min() > 1e-2),
        "side": side,
        "slice": slice_name,
        "tag": tag,
    }
