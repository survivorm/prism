"""Recipes on top of the kernel — NOT core (ТЗ E40_math_core §II.11).

Promotion rule: a recipe enters the core only when a second consumer appears
(after E-41). Do not call ``R_can`` an order parameter before it passes
criteria (a′)∧(a)∧(b)∧(c) (ТЗ §V).
"""

from __future__ import annotations

import numpy as np

from .operator_checks import apply_P

__all__ = ["R_can", "R_raw", "G_l", "r_cross"]


def _lam_share(G: np.ndarray) -> float:
    """λ_max(G)/tr(G) for a symmetric PSD ``G``."""
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square, got {G.shape}")
    tr = float(np.trace(G))
    if tr <= 0:
        raise ValueError("λ-share: tr G ≤ 0")
    return float(np.linalg.eigvalsh(G)[-1] / tr)


def R_raw(J: np.ndarray) -> float:
    """``λ_max(JJᵀ)/tr(JJᵀ)`` — top-mode share of the RAW Jacobian."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(f"J must be [K, D], got {J.shape}")
    return _lam_share(J @ J.T)


def R_can(J: np.ndarray, p: np.ndarray) -> float:
    """``λ_max(E_K)/tr(E_K)`` with ``E_K = J^(p)(J^(p))ᵀ`` on the canonical
    slice ``J^(p) = P(p) J`` (ТЗ E40_math_core §I «Определение», paper §2.11.1,
    e40_todo §8.1). The slice must be named in the ChannelTag — R-values from
    different slices are not comparable (slice-locality, ТЗ §I.2).

    NOT ``(I − P)J``: that complement is the E-37 ``K`` object (frac_K) and is
    ≈ J in norm (P kills 98–99 % of it), which made an ``R_can`` built on it
    indistinguishable from ``R_raw`` (audit_core_math_v1 M1, fixed 2026-11-03)."""
    J = np.asarray(J, dtype=np.float64)
    Kmat = apply_P(J, p)
    return _lam_share(Kmat @ Kmat.T)


def G_l(delta_raw: float, delta_fixP: float) -> float:
    """``G(l) = 1 − Δ_fixP/Δ_raw`` — gauge-reduction of a transport distance.
    Takes the two ALREADY-computed distances (aggregation of Δ is the caller's
    contract; state it in the tag)."""
    if delta_raw <= 0:
        raise ValueError("G_l: Δ_raw must be > 0")
    return 1.0 - float(delta_fixP) / float(delta_raw)


def r_cross(curve1: np.ndarray, curve2: np.ndarray) -> float | None:
    """First crossing index of two curves on a common grid (linear
    interpolation); ``None`` if they never cross."""
    c1 = np.asarray(curve1, dtype=np.float64)
    c2 = np.asarray(curve2, dtype=np.float64)
    if c1.shape != c2.shape or c1.ndim != 1:
        raise ValueError("curves must be equal-length 1-D")
    d = c1 - c2
    sign = np.sign(d)
    for i in range(1, d.size):
        if sign[i] == 0:
            return float(i)
        if sign[i] != sign[i - 1] and sign[i - 1] != 0:
            t = d[i - 1] / (d[i - 1] - d[i])
            return float(i - 1 + t)
    return None
