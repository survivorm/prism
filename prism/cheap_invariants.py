"""Cheap triple invariants on ``P(p)`` and ``J`` (ТЗ E40_math_core §II.8, M10).

Re-exports the triple from :mod:`prism.operator_checks` (single code path —
do NOT reimplement, E6 was exactly a two-code-path problem) and adds the
isotropic predictor of ``c``:

- ``c_pred(p) = tr P / sqrt(K · tr P²)`` — the alignment coefficient of the
  operator's own spectrum (1 iff P ∝ I); compares a measured ``c`` against
  what the OPERATOR alone suggests, before reading anything off the data.
- ``A = c / c_pred`` — data-vs-operator alignment ratio.

``frac_K``: the ТЗ staged it ``blocked_on_E6``; E6 was closed the same day
(the published table mixed slices; the JSON triple was self-consistent:
``c/s = 2.4418e-3/0.018772 = 0.13008 = cos_J_PJ`` to 5 digits), so it is a
normal core member with one rule: numerator, denominator and the identity
must be quoted on ONE aggregation and ONE position slice (paper §6 item 23).
"""

from __future__ import annotations

import numpy as np

from .operator_checks import c, cos_J_PJ, frac_K, s  # re-exports, one path
from .p_algebra import trace_P, trace_P2

__all__ = ["c", "s", "cos_J_PJ", "frac_K", "c_pred", "A_alignment"]


def c_pred(p: np.ndarray) -> float:
    """Isotropic predictor ``tr P / sqrt(K · tr P²)`` ∈ (0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    K = p.size
    return trace_P(p) / float(np.sqrt(K * trace_P2(p)))


def A_alignment(J: np.ndarray, p: np.ndarray) -> float:
    """``A = c / c_pred`` — measured cross-term over its operator-level value."""
    return float(c(J, p) / c_pred(p))
