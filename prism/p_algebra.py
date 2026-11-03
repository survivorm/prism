"""Algebra of the softmax-gauge operator ``P(p) = diag(p) − p pᵀ`` (ТЗ E40_math_core §I.1, §II.3, M3).

``P`` plays three roles at once (paper §2.5): Jacobian of softmax, Fisher–Rao
metric in logit coordinates (``JᵀPJ``), and gauge annihilator (``P·1 = 0``).

Not to be confused with ``prism.jacobian.softmax_jacobian(J, p)`` which APPLIES
``P(p)`` to a Jacobian (torch, never materializes ``P``). This module is the
numpy algebra OF the operator itself. Naming follows the physics object, so
the ТЗ's ``softmax_jacobian(p) -> [K, K]`` lives here as :func:`P_matrix`.

Closed forms (exact, no eigendecomposition):
- ``tr P = Σp − Σp²``            (= ``1 − Σp²`` for normalized p)
- ``‖P‖²_F = tr P² = Σp² − 2Σp³ + (Σp²)²``
- ``P`` is symmetric PSD (``xᵀPx = Var_p(x)``), ``rank = K−1`` for full-support
  p, and ``P`` is NOT a projector: ``P² = P`` is impossible for interior p
  (eigenvalues are < 1; at the vertex ``p = e_i`` deg ``P = 0`` IS idempotent —
  the boundary case, ТЗ §I.1 wording corrected, see docs/E40_math_core_answers.md).
- eigenvalue bound ``λ_max ≤ p_max`` (from ``P ⪯ diag p``), NOT
  ``p_max(1−p_max)`` which fails by 2× at the two-point limit (ТЗ §I.1 claim,
  refuted numerically in the answers file).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "P_matrix", "trace_P", "trace_P2", "isotropic_reference",
    "g_uni", "gauge_share", "secular_residual", "secular_roots",
]


def _check_p(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError(f"p must be a 1-D array of length ≥ 2, got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("p has non-finite entries")
    if (p < 0).any():
        raise ValueError("p has negative entries (not a probability vector)")
    # Допуск нормировки — по точности входа, не фиксированный 1e-8: float32-строка,
    # нормированная на 1, несёт |Σp−1| до ~K·eps32 (на кэше exp27b замер: медиана
    # 4.0e-08, p99 1.7e-07, max 2.8e-07 при K=128), поэтому фиксированный 1e-8
    # отвергал 88 % реальных позиций (docs/audit3_math.md п. 2Б + самопроверка).
    # Обрезанную строку (P128_RAW, Σp ≈ 0.49) этот допуск по-прежнему ловит.
    tol = max(1e-8, p.size * np.finfo(np.float32).eps)
    if abs(p.sum() - 1.0) > tol:
        raise ValueError(
            f"p must be normalized (Σp=1); got Σp={p.sum():.12g} (tol {tol:.1e}, "
            f"= K·eps32 for float32-sourced p). Truncated/raw rows (P128_RAW) must "
            "be renormalized upstream and tagged p_mode in the ChannelTag (ТЗ §II.1).")
    return p


def P_matrix(p: np.ndarray) -> np.ndarray:
    """``P(p) = diag(p) − p pᵀ`` as a dense ``[K, K]`` matrix."""
    p = _check_p(p)
    return np.diag(p) - np.outer(p, p)


def trace_P(p: np.ndarray) -> float:
    """``tr P = Σp − Σp²`` (= ``1 − ‖p‖₂²`` for normalized p)."""
    p = _check_p(p)
    return float(p.sum() - np.dot(p, p))


def trace_P2(p: np.ndarray) -> float:
    """``tr P² = ‖P‖²_F = Σp² − 2Σp³ + (Σp²)²`` (closed form)."""
    p = _check_p(p)
    s2, s3 = float(np.dot(p, p)), float(np.dot(p, p * p))
    return s2 - 2.0 * s3 + s2 * s2


def isotropic_reference(p: np.ndarray) -> float:
    """``iso = ‖P‖²_F / K`` — what ``‖PJ‖²_F/‖J‖²_F` would be for isotropic J."""
    p = _check_p(p)
    return trace_P2(p) / p.size


def g_uni(J: np.ndarray) -> float:
    """Uniform-direction energy share ``‖Jᵀ1‖² / (K ‖J‖²_F)`` (paper §2.5).

    Corpus-aggregated: accepts ``[K, D]`` or ``[n, K, D]``; sums numerator and
    denominator over everything, ONE aggregate (never a mean of per-position
    ratios — aggregation rule, paper §6 item 23). Identical by construction to
    ``gauge_share`` of the Gramian ``G = Σ_t J_t J_tᵀ`` (identity O2≡O3,
    max measured discrepancy 7.7e-16). Independent of p — p is not an argument.
    """
    J = np.asarray(J, dtype=np.float64)
    if J.ndim == 2:
        J = J[None]
    if J.ndim != 3:
        raise ValueError(f"J must be [K, D] or [n, K, D], got {J.shape}")
    K = J.shape[1]
    col_sums = J.sum(axis=1)                    # [n, D]: Σ_k J_k
    num = float((col_sums ** 2).sum())          # Σ_t ‖Σ_k J_k‖²
    den = float((J ** 2).sum())
    if den <= 0:
        raise ValueError("g_uni: ‖J‖_F = 0")
    return num / (K * den)


def gauge_share(G: np.ndarray) -> float:
    """ESG gauge share ``1ᵀG1 / (K tr G)`` on the Gramian ``G`` (``[K, K]``).

    Equals ``Σ_k (u_kᵀ1)² λ_k / Σ_k λ_k`` = ``spectral.gauge_share(spectrum,
    eigenvectors)`` on the same ``G`` (identity O2≡O3), but needs no
    eigendecomposition. Not a duplicate of ``prism.spectral.gauge_share``:
    different input (Gramian, not spectrum+vectors), numpy not torch.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be square [K, K], got {G.shape}")
    tr = float(np.trace(G))
    if tr <= 0:
        raise ValueError("gauge_share: tr G ≤ 0")
    K = G.shape[0]
    ones = np.ones(K)
    return float(ones @ G @ ones) / (K * tr)


def secular_residual(p: np.ndarray, lam: float) -> float:
    """``Σ_i p_i²/(p_i − λ) − 1``; zero exactly at the eigenvalues of ``P``
    strictly between the poles (secular equation, ТЗ §I.1)."""
    p = _check_p(p)
    d = p - lam
    if np.abs(d).min() < 1e-14:
        raise ValueError("lam coincides with a pole p_i")
    return float(np.dot(p * p, 1.0 / d) - 1.0)


def secular_roots(p: np.ndarray) -> np.ndarray:
    """The ``K−1`` nonzero eigenvalues of ``P(p)`` via root search between the
    poles of the secular equation (structure-preserving, no ``eigvalsh``).

    Assumes distinct ``p_i`` (poles separate the roots: exactly one root in
    each gap between consecutive sorted ``p_i``; f increases strictly from
    ``−∞`` at the lower pole to ``+∞`` at the upper one). Falls back loudly if
    p has duplicates within 1e-12 — the secular form degenerates there
    (deflated block), and ``np.linalg.eigvalsh(P_matrix(p))`` is the right tool.

    ПОТОЛОК (измерено на реальном срезе softmax, exp27b, K=128, 1580 позиций):
    на 11.4 % позиций полюса неразрешимы в fp64 и функция отказывается — это
    честный предел метода бисекции, а не баг; для них — ``eigvalsh``. На
    остальных 88.6 % корни сходятся с ``eigvalsh(P_matrix(p))`` до 8.7e-12
    (тезис `docs/audit3_math.md` п. 2Б: падать на реальных данных действительно
    будет, но по причине разрешения полюсов, а не «одинаковых softmax-значений»;
    вторая половина тезиса — «заменить на eigvalsh» — не принята, см.
    `docs/audits_answers.md` §9).
    Абсолютный порог здесь сознательный: относительный («от p_max») ломает
    бисекцию, когда зазор оказывается меньше ulp верхнего полюса (проверено —
    даёт 100 % отказов «корень совпал с полюсом»).
    """
    p = np.sort(_check_p(p))[::-1]                    # descending p_1 > p_2 > …
    K = p.size
    if len(np.unique(np.round(p, 12))) < K:
        raise ValueError("secular_roots: duplicate p_i — use eigvalsh instead")
    roots = np.empty(K - 1)
    for i in range(K - 1):                             # one root in (p_{i+1}, p_i)
        lo, hi = p[i + 1], p[i]
        span = hi - lo
        if span < 1e-12:
            raise ValueError("secular_roots: adjacent p_i closer than 1e-12 "
                             "— use np.linalg.eigvalsh instead")
        off = min(span * 0.01, max(span * 1e-9, 1e-13))
        a, b = lo + off, hi - off
        fa, fb = secular_residual(p, a), secular_residual(p, b)
        nudge = 0
        while not (fa < 0.0 < fb):                     # pull endpoints inward
            nudge += 1
            if nudge > 40:
                raise RuntimeError(f"secular_roots: no bracket in ({lo}, {hi})")
            if fa >= 0:
                a = 0.5 * (a + lo)
                fa = secular_residual(p, a)
            if fb <= 0:
                b = 0.5 * (b + hi)
                fb = secular_residual(p, b)
        for _ in range(200):                           # bisection
            m = 0.5 * (a + b)
            fm = secular_residual(p, m)
            if fm < 0:
                a, fa = m, fm
            else:
                b, fb = m, fm
            if b - a < 1e-14 * max(1.0, abs(m)):
                break
        roots[i] = 0.5 * (a + b)
    return roots
