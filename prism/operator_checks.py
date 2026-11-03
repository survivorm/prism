"""Operator-fraction checks and the two-null reduction control call-signature.

Implements ТЗ E40_math_core §II.7 (M7). Everything here answers one question:
how much of ``X`` does an operator ``O`` eat, and is that number information
about the DATA or just the norm of the OPERATOR (paper §6 items 19/22/23).

Aggregation rule (paper §6 item 23, E6 lesson): for stacks of positions the
corpus aggregate sums numerator and denominator over the SAME slice in ONE
ratio. Per-position medians/means are different objects — the identity
``frac_K = 1 − 2c + s²`` closes exactly for the aggregates and for each
position separately, but NOT for a median of one quantity combined with the
aggregate of another (this exact mix produced the phantom ``s² < 0`` of E6).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "operator_fraction", "isotropic_reference", "apply_P",
    "c", "s", "O1", "cos_J_PJ", "frac_K",
]


def _check_OX(O: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    O = np.asarray(O, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if O.ndim != 2 or O.shape[0] != O.shape[1]:
        raise ValueError(f"O must be square 2-D, got {O.shape}")
    if X.ndim < 2 or X.shape[-2] != O.shape[1]:
        raise ValueError(
            f"X must have its second-to-last axis = O.shape[1]={O.shape[1]}, got {X.shape}")
    return O, X


def operator_fraction(O: np.ndarray, X: np.ndarray, norm_type: str = "energy") -> float:
    """``‖OX‖²_F/‖X‖²_F`` (``norm_type="energy"``) or ``‖OX‖_F/‖X‖_F`` (``"norm"``).

    ``norm_type`` is identity-relevant (norm vs energy differ by a square —
    paper §6 item 19); the returned value is meaningless without a ChannelTag.
    """
    O, X = _check_OX(O, X)
    OX = np.tensordot(O, X, axes=([1], [-2]))          # [..., m, q]
    num = float((OX ** 2).sum())
    den = float((X ** 2).sum())
    if den <= 0:
        raise ValueError("operator_fraction: ‖X‖ = 0")
    return num / den if norm_type == "energy" else float(np.sqrt(num / den))


def isotropic_reference(O: np.ndarray) -> float:
    """``E‖OX‖²/‖X‖²`` over isotropic ``X`` = ``‖O‖²_F / dim_in``.

    Any claim «O eats f·(reference)» must be quoted against this AND against a
    content-matched null (paper §6 item 23); the operator alone already eats
    ``1 − conc`` of a random input (E-39 supplement: conc_iid ≈ 0.115).
    """
    O = np.asarray(O, dtype=np.float64)
    if O.ndim != 2 or O.shape[0] != O.shape[1]:
        raise ValueError(f"O must be square 2-D, got {O.shape}")
    return float((O ** 2).sum() / O.shape[1])


def apply_P(J: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``P(p) J`` along the token axis without materializing P (torch-free).

    Accepts ``J [K, D]`` + ``p [K]``, or stacks ``J [n, K, D]`` with ``p [K]``
    or per-position ``p [n, K]``.
    """
    J = np.asarray(J, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    K = J.shape[-2]
    if p.shape == (K,):
        pw = p[:, None]                                # broadcast over the D axis
    elif p.shape == J.shape[:-1]:
        pw = p[..., None]
    else:
        raise ValueError(f"p shape {p.shape} incompatible with J {J.shape}")
    m = (pw * J).sum(axis=-2, keepdims=True)           # p-weighted row mean
    return pw * (J - m)                                # (PJ)_i = p_i (J_i − m)


def _split(J: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    PJ = apply_P(J, p)
    return PJ, p


def _check_p_slice(p: np.ndarray, p_slice: str) -> None:
    """Label guard (патч v2 §1.6.2): ``p_t`` = per-position, ``p_0`` = fixP.
    They are DIFFERENT operators ``P(p)``, so the label must be explicit — and
    ``p_0`` is validated: stacked rows must actually be one fixed p."""
    if p_slice not in ("p_t", "p_0"):
        raise ValueError(f"p_slice={p_slice!r} not in p_t|p_0 (stop condition)")
    if p_slice == "p_0" and p.ndim >= 2:
        dev = float(np.abs(p - p.reshape(-1, p.shape[-1])[0]).max())
        if dev > 1e-12:
            raise ValueError(
                f"p_slice='p_0' but p varies per position (max dev {dev:.3g}); "
                "this is per-position p_t — mislabelling mixes operators (E6 class)")


def _agg(num: float | np.ndarray, den: float | np.ndarray, aggregation: str,
         root: bool = False):
    """One aggregation over one slice — never mix them across quantities (E6)."""
    if aggregation == "ratio_of_means":
        val = float(num.sum() / den.sum())
        return val ** 0.5 if root else val
    r = num / den
    if aggregation == "mean":
        return float((np.sqrt(r) if root else r).mean())
    if aggregation == "median":
        return float(np.median(np.sqrt(r) if root else r))
    if aggregation == "per_position":   # RAW energy ratios (root is a property of the
        return r                        # aggregate; frac = 1 − 2c + O1 closes positionally)
    raise ValueError(
        f"aggregation={aggregation!r} not in ratio_of_means|mean|median|per_position "
        "(energy_weighted of per-position ratios with \u2016J_t\u2016\u00b2 weights "
        "equals ratio_of_means \u2014 use it)")


def c(J: np.ndarray, p: np.ndarray, aggregation: str = "ratio_of_means",
      p_slice: str = "p_t") -> np.ndarray | float:
    """``c = ⟨J, PJ⟩_F / ‖J‖²_F`` — cross term of ``‖(I−P)J‖²`` (PSD of P
    licenses ``c ≥ 0`` and NOTHING about alignment: measured cos(J, PJ) is
    only 0.064–0.108, i.e. PJ is nearly ORTHOGONAL to J, not parallel to it).

    ``p_slice`` labels which ``P(p)`` produced the number (per-position ``p_t``
    vs fixed ``p_0``); the position slice (all/alive) is whatever you passed —
    report it with the number (tag, §1.6.5)."""
    _check_p_slice(p, p_slice)
    PJ, _ = _split(J, p)
    num = (PJ * J).sum(axis=(-2, -1))
    den = (J * J).sum(axis=(-2, -1))
    return _agg(num, den, aggregation)


def s(J: np.ndarray, p: np.ndarray, aggregation: str = "ratio_of_means",
      p_slice: str = "p_t") -> np.ndarray | float:
    """``s = ‖PJ‖_F / ‖J‖_F`` — the NORM ratio (E-33r's «98–99 %» object)."""
    _check_p_slice(p, p_slice)
    PJ = apply_P(J, p)
    num = (PJ * PJ).sum(axis=(-2, -1))
    den = (J * J).sum(axis=(-2, -1))
    return _agg(num, den, aggregation, root=True)


def O1(J: np.ndarray, p: np.ndarray, aggregation: str = "ratio_of_means",
       p_slice: str = "p_t") -> np.ndarray | float:
    """``O1 = ‖PJ‖²_F / ‖J‖²_F = s²`` — ENERGY kill fraction. Test O1==s² is a
    tautology guard on the aggregation, keep it asserted (M0 registry)."""
    _check_p_slice(p, p_slice)
    PJ = apply_P(J, p)
    num = (PJ * PJ).sum(axis=(-2, -1))
    den = (J * J).sum(axis=(-2, -1))
    return _agg(num, den, aggregation)


def cos_J_PJ(J: np.ndarray, p: np.ndarray, aggregation: str = "ratio_of_means",
             p_slice: str = "p_t") -> np.ndarray | float:
    """``cos(J, PJ) = c/s`` — a DIFFERENT number from ``c`` (ТЗ §III.3):
    measured 0.064–0.108 while c is 0.0005–0.003."""
    return c(J, p, aggregation, p_slice) / s(J, p, aggregation, p_slice)


def frac_K(J: np.ndarray, p: np.ndarray, aggregation: str = "ratio_of_means",
           p_slice: str = "p_t") -> np.ndarray | float:
    """``frac_K = ‖(I−P)J‖²_F / ‖J‖²_F`` — ENERGY, not norm (E-37 metric).

    Identity ``frac_K = 1 − 2c + s²`` holds per position and for the corpus
    aggregate computed on one slice; combining it with per-position medians is
    what produced the phantom E6 (closed 2026-11-03, no longer blocked).
    """
    _check_p_slice(p, p_slice)
    PJ = apply_P(J, p)
    Kmat = J - PJ
    num = (Kmat * Kmat).sum(axis=(-2, -1))
    den = (J * J).sum(axis=(-2, -1))
    return _agg(num, den, aggregation)
