"""Procrustes transport between orthonormal frames (ТЗ E40_math_core §II.5, M5).

numpy-side companion to ``prism.holonomy`` (which is torch and does the
layer-loop geometry); this module provides the PRECONDITIONS the ТЗ makes
mandatory before any ``log Q``:

``align_signs_and_columns`` before polar: ``Q_U = polar(U_tᵀ U_{t+1})`` is
meaningful only when the columns are sign-consistent and matched. A sign flip
leaves ``−1`` on the diagonal → principal ``log`` complex; a permutation at
``σ_j ≈ σ_{j+1}`` makes ``log`` branch-ambiguous (ТЗ §I.8).

Flags (all reported into the artifact, ТЗ §I.8 pt 4):
- ``sign_flips``   — columns flipped to make ``u_j(t)ᵀu_j(t+1) ≥ 0``;
- ``order_cross``  — Hungarian matching ≠ rank order (Σ reorders between t's);
- ``unstable_transport`` — spectrum of the polar factor close to ``−1`` or
  ``‖log Q‖_F`` beyond ``angle_max``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import logm
from scipy.optimize import linear_sum_assignment

__all__ = [
    "polar_factor", "align_signs_and_columns", "procrustes_transport",
    "transport_log", "holonomy", "holonomy_angles", "kappa",
]


def _check_frame(U: np.ndarray, name: str) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"{name} must be [d, r] with orthonormal columns, got {U.shape}")
    r = U.shape[1]
    gram_err = float(np.abs(U.T @ U - np.eye(r)).max())
    if gram_err > 1e-6:
        raise ValueError(f"{name}: columns not orthonormal (max |UᵀU−I| = {gram_err:.2e})")
    return U


def polar_factor(M: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix ``Q = U Vᵀ`` from ``M = U S Vᵀ`` (∈ O(r))."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"polar_factor wants a square matrix, got {M.shape}")
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def align_signs_and_columns(U0: np.ndarray, U1: np.ndarray,
                            tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, dict]:
    """Match columns of ``U1`` to ``U0`` (Hungarian on ``|U0ᵀU1|``), then fix
    signs so the matched diagonal overlaps are ``≥ 0``.

    Returns ``(U0, U1_aligned, flags)`` with ``flags["perm"]`` — the column
    matching (``U1a[:, j]`` came from ``U1[:, perm[j]]``), so consumers can
    carry per-column quantities (σ!) along with the frame (M2 fix). 
    Matching by ``|U0ᵀU1|`` — NOT by rank ``σ`` — is the ТЗ requirement;
    disagreement with σ-order is reported as ``order_cross`` (a fact about
    Σ reordering, not an error).
    """
    U0 = _check_frame(U0, "U0")
    U1 = _check_frame(U1, "U1")
    if U0.shape != U1.shape:
        raise ValueError(f"frame shape mismatch {U0.shape} vs {U1.shape}")
    S = np.abs(U0.T @ U1)
    _, perm = linear_sum_assignment(-S)
    flags = {
        "order_cross": bool(np.any(perm != np.arange(perm.size))),
        "sign_flips": [],
        "perm": [int(k) for k in perm],
    }
    U1a = U1[:, perm]
    diag = np.einsum("ij,ij->j", U0, U1a)             # u_j(t)ᵀ u_j(t+1)
    if (np.abs(diag) < tol).any():
        raise ValueError(
            "align_signs_and_columns: a matched pair is ~orthogonal "
            f"(min |diag| = {np.abs(diag).min():.2e}) — frames too far apart, "
            "column identity is lost; refuse to guess signs.")
    signs = np.sign(diag)
    flags["sign_flips"] = [int(j) for j in np.where(signs < 0)[0]]
    U1a = U1a * signs
    return U0, U1a, flags


def procrustes_transport(V0: np.ndarray, V1: np.ndarray) -> np.ndarray:
    """``Q = polar(V0ᵀ V1)`` — the optimal rotation ``O(r)`` between frames."""
    V0 = _check_frame(V0, "V0")
    V1 = _check_frame(V1, "V1")
    return polar_factor(V0.T @ V1)


def transport_log(Q: np.ndarray, angle_max: float = math.pi - 1e-3) -> np.ndarray:
    """Principal matrix logarithm of an orthogonal ``Q``; refuses (ValueError)
    when the spectrum touches ``−1`` (log undefined on the principal branch)."""
    Q = np.asarray(Q, dtype=np.float64)
    ang = np.abs(np.angle(np.linalg.eigvals(Q)))
    if ang.max() > angle_max:
        raise ValueError(
            f"transport_log: eigenangle {ang.max():.6f} rad at/over the branch "
            "cut (−1): log Q undefined on the principal branch (unstable_transport)")
    L = logm(Q)
    if float(np.abs(L.imag).max()) > 1e-8:
        raise ValueError(f"transport_log: complex log (max |Im| = {np.abs(L.imag).max():.2e})")
    return L.real


def holonomy(V0: np.ndarray, V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Loop holonomy ``R01 · R12 · R02⁻¹`` for three consecutive frames
    (each ``Rij = polar(Viᵀ Vj)``, columns aligned first)."""
    V0a, V1a, _ = align_signs_and_columns(V0, V1)
    _, V2a, _ = align_signs_and_columns(V0a, V2)      # align all to V0 gauge
    R01 = procrustes_transport(V0a, V1a)
    R12 = procrustes_transport(V1a, V2a)
    R02 = procrustes_transport(V0a, V2a)
    return R01 @ R12 @ R02.T                          # R02⁻¹ = R02ᵀ (orthogonal)


def holonomy_angles(H: np.ndarray) -> np.ndarray:
    """Rotation angles ``θ_j = arg eig(H)`` in ``[0, π]`` (± pairs)."""
    H = np.asarray(H, dtype=np.float64)
    return np.sort(np.abs(np.angle(np.linalg.eigvals(H))))


def kappa(H: np.ndarray) -> float:
    """``κ(H) = ‖log H‖_F`` — invariant under orthogonal similarity
    ``H → Oᵀ H O`` (the M0 registry asserts exactly this)."""
    return float(np.linalg.norm(transport_log(H), "fro"))
