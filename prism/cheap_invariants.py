"""Cheap triple invariants on ``P(p)`` and ``J`` (ТЗ E40_math_core §II.8, M10).

Re-exports the triple from :mod:`prism.operator_checks` (single code path —
do NOT reimplement, E6 was exactly a two-code-path problem) and adds the
isotropic predictor of ``c``:

- ``c_pred(p) = tr P / K`` — the isotropic predictor of ``c``: for isotropic
  ``J`` (iid entries or orthonormal rows) ``c = tr P / K`` holds exactly
  (matches the E-40 doc ``A``: ``A = c/(trP/K)``, ``A = 1 ⟺ isotropic``).
- ``A = c / c_pred`` — measured cross-term over its isotropic value.

``frac_K``: the ТЗ staged it ``blocked_on_E6``; E6 was closed the same day
(the published table mixed slices; the JSON triple was self-consistent:
``c/s = 2.4418e-3/0.018772 = 0.13008 = cos_J_PJ`` to 5 digits), so it is a
normal core member with one rule: numerator, denominator and the identity
must be quoted on ONE aggregation and ONE position slice (paper §6 item 23).
"""

from __future__ import annotations

import numpy as np

from .operator_checks import c, cos_J_PJ, frac_K, s  # re-exports, one path
from .p_algebra import trace_P

__all__ = ["c", "s", "cos_J_PJ", "frac_K", "c_pred", "A_alignment"]


def c_pred(p: np.ndarray) -> float:
    """Isotropic predictor ``tr P / K`` ∈ (0, 1/K].

    For isotropic ``J`` (iid entries or orthonormal rows)
    ``c = ⟨J, PJ⟩/‖J‖² = tr P / K`` exactly (E-40 doc:
    ``A := K·tr(PG)/(trP·trG) = c/(trP/K)``, ``A = 1 ⟺ isotropic``).
    The former formula ``trP/√(K·trP²)`` was the spectral flatness of
    ``P`` — an object of ``p`` alone, not a predictor of ``c`` (70–330×
    off on real p; docs/prism_core_audit.md P1-1).
    """
    p = np.asarray(p, dtype=np.float64)
    return trace_P(p) / p.size


def A_alignment(J: np.ndarray, p: np.ndarray) -> float:
    """``A = c / c_pred`` — measured cross-term over its isotropic value
    (``A = 1 ⟺ isotropic J``; the E-40 doc ``A``)."""
    return float(c(J, p) / c_pred(p))
