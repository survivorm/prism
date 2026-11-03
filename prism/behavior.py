"""Behavioral endpoints — GATED module (ТЗ E40_math_core §II.10, M11).

Gate: tier-0 (``E40_*.py`` carriers, and the kernel modules) must NOT import
this module — behavior lives with E-42/E-43/E-44 (interventional/behavioral
tiers). A pytest (``tests/test_behavior_gate.py``) enforces it: CI is
pytest-only, so a CI import-check is a test that greps sources.

Boundary conditions baked into the API (ТЗ §III.9/§III.10):
- ``b_mag`` is ``‖Δz‖²`` — the SQUARE, ``‖·‖₂`` (first power) is not offered;
- ``b_lin`` is the general ``ε²‖Jv‖²``; ``ε²σ₁²`` is the corollary at ``v = v₁``
  (asserted in tests), never a separate definition;
- ``KL_lin`` carries the Fisher factor ``F = diag(p) − ppᵀ = P(p)``, not ``I``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

__all__ = ["b_mag", "b_lin", "b_res", "lq_coupling", "b_qual", "KL_lin"]


def b_mag(delta_z: np.ndarray) -> float:
    """Magnitude endpoint ``‖Δz‖²`` (logit-space energy of the perturbation).

    NOTE (E44 §5.1.1): ``Δz`` is ``J_{t→t+k} v`` with ``v`` chosen by the
    experimenter, so ``ρ(g_F, b_mag)`` correlates two operator-derived objects
    and needs BOTH nulls (operator + content-matched) before interpretation.
    """
    dz = np.asarray(delta_z, dtype=np.float64)
    return float((dz ** 2).sum())


def b_lin(J: np.ndarray, eps: float, v: np.ndarray) -> float:
    """Linear response ``ε² ‖J v‖²`` for a unit-norm direction ``v``.

    ``v`` is normalized here; passing ``v = v₁`` (top right singular vector)
    yields ``ε² σ₁²`` — the test asserts exactly that identity.
    """
    J = np.asarray(J, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nv <= 0:
        raise ValueError("v must be nonzero")
    Jv = J @ (v / nv)
    return float(eps) ** 2 * float((Jv ** 2).sum())


def b_res(b_mag_value: float, b_lin_value: float) -> float:
    """Residual response ``b_mag − b_lin`` (signed).

    Order (патч v2 §1.1): ``‖Δz‖² = ε²‖Jv‖² + ε³⟨Jv, H[v,v]⟩ + O(ε⁴)``, so
    ``b_res = O(ε³)`` — NOT ``O(ε⁴)``: the cubic term is the first nonzero
    remainder unless ``⟨Jv, H[v,v]⟩ = 0`` (a checkable condition, not an
    identity). Reduction control therefore tests ``b_res/(ε²‖Jv‖²) → 0``
    (``o(ε²)``), never an ``O(ε⁴)`` rate — an honest ε³ term must not be read
    as a reduction bug. The coefficient is :func:`lq_coupling`.
    """
    return float(b_mag_value) - float(b_lin_value)


def lq_coupling(Jv: np.ndarray, Hvv: np.ndarray) -> float:
    """Linear-quadratic coupling ``⟨Jv, H[v,v]⟩`` (signed) — the ε³ coefficient
    of ``b_res`` and a mandatory artifact field for E-44 (патч v2 §1.1.2).

    ``Jv = J v`` and ``Hvv = H[v,v]`` (Hessian bilinear form applied to ``v``),
    both already carrying the same ``v``; ``v`` normalization is the caller's
    convention (unit norm assumed). This is a measurable object, not noise.
    """
    a = np.asarray(Jv, dtype=np.float64).ravel()
    b = np.asarray(Hvv, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"lq_coupling: shape {a.shape} vs {b.shape}")
    return float(a @ b)


def b_qual(logits_base: np.ndarray, logits_perturbed: np.ndarray) -> float:
    """Qualitative endpoint: rank correlation of logits (corr_f family,
    Pearson-on-ranks = Spearman). Interventional channel only — forbidden in
    the natural channel (paper §6 item 21)."""
    a = np.asarray(logits_base, dtype=np.float64).ravel()
    b = np.asarray(logits_perturbed, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"logit shapes differ {a.shape} vs {b.shape}")
    return float(spearmanr(a, b).statistic)


def KL_lin(J: np.ndarray, p: np.ndarray, eps: float, v: np.ndarray) -> float:
    """Second-order KL of the softmax under ``h → h + ε v``:
    ``½ ε² (Jv)ᵀ P(p) (Jv)`` with the Fisher–Rao metric ``P(p)`` (ТЗ §III.10:
    the factor is ``diag(p) − ppᵀ``, not ``I``)."""
    J = np.asarray(J, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if p.ndim != 1 or J.shape[-2] != p.size:
        raise ValueError(f"p [K] must match J token axis {J.shape}")
    if abs(p.sum() - 1.0) > 1e-8:
        raise ValueError("KL_lin: p must be normalized (Σp = 1)")
    Jv = J @ (v / np.linalg.norm(v))
    # xᵀPx = Σ_i p_i x_i² − (Σ_i p_i x_i)² = Var_p(x) — closed form, P never materialized
    return 0.5 * float(eps) ** 2 * (float(np.dot(p, Jv ** 2)) - float(np.dot(p, Jv)) ** 2)
