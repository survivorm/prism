"""p_algebra unit checks — ТЗ E40_math_core §IV items 1, 2, 3, 7 + the corrected
spectral claims (docs/E40_math_core_answers.md §2–3)."""

import numpy as np
import pytest

from prism.operator_checks import apply_P
from prism.p_algebra import (
    P_matrix, g_uni, gauge_share, isotropic_reference, secular_residual,
    secular_roots, trace_P, trace_P2,
)
from prism.spectral import gauge_share as gauge_share_torch
import torch


def _p(rng, K=32):
    x = rng.standard_normal(K) * 2.0
    return np.exp(x) / np.exp(x).sum()


def test_O2_eq_O3_g_uni_equals_esg_gauge_share():
    """§IV.1: g_uni(J) == ESG gauge_share, rel < 1e-12 (identity, not a fit)."""
    rng = np.random.default_rng(3)
    J = rng.standard_normal((8, 32, 16))
    G = sum(Jt @ Jt.T for Jt in J)
    got, want = g_uni(J), gauge_share(G)
    assert abs(got - want) < 1e-12 + 1e-10 * abs(want)
    # cross-check against the torch spectrum/eigenvector route in spectral.py
    ev, evec = np.linalg.eigh(G)                    # ascending
    idx = np.argsort(ev)[::-1]
    gs_t = gauge_share_torch(torch.tensor(ev[idx]), torch.tensor(evec[:, idx]))
    assert float(gs_t) == pytest.approx(want, rel=1e-9)


def test_g_uni_gf_zero():
    """§IV.2: g_uni(P(p)J) ≡ 0 (algebraic: P·1 = 0, not empirical)."""
    rng = np.random.default_rng(4)
    p, J = _p(rng), rng.standard_normal((6, 32, 16))
    assert g_uni(np.stack([apply_P(Jt, p) for Jt in J])) < 1e-12


def test_iso_inline_eq_closed():
    """§IV.3: ‖P‖²_F/K inline == closed form, rel < 1e-9."""
    rng = np.random.default_rng(5)
    for K in (4, 17, 128):
        p = _p(rng, K)
        inline = float((P_matrix(p) ** 2).sum()) / K
        assert abs(inline - isotropic_reference(p)) < 1e-9 * inline


def test_p_tilde_sum_branches():
    """§IV.7: Σp̃ ∈ {1, 1 − mass} — renormalized row vs raw (P128_RAW) row."""
    rng = np.random.default_rng(6)
    p, mass = _p(rng), 0.412
    raw = p * mass
    assert abs(raw.sum() - mass) < 1e-12
    assert abs((raw / raw.sum()).sum() - 1.0) < 1e-12
    with pytest.raises(ValueError, match="normalized"):
        P_matrix(raw)                                # sub-stochastic rows rejected


def test_trace_identities_and_psd():
    rng = np.random.default_rng(7)
    p = _p(rng, 40)
    P = P_matrix(p)
    assert abs(float(np.trace(P)) - trace_P(p)) < 1e-12
    assert abs(float(np.trace(P @ P)) - trace_P2(p)) < 1e-9
    assert np.linalg.eigvalsh(P).min() >= -1e-12
    assert np.linalg.matrix_rank(P) == len(p) - 1


def test_lambda_max_bound_corrected():
    """The ТЗ claim λmax ≤ p_max(1−p_max) is wrong (2× at the two-point limit);
    the correct bound from P ⪯ diag(p) is λmax ≤ p_max."""
    rng = np.random.default_rng(8)
    for K in (2, 8, 128):
        for _ in range(20):
            p = _p(rng, K)
            lam = np.linalg.eigvalsh(P_matrix(p))
            assert lam[-1] <= p.max() + 1e-12
        if K == 2:                                   # exact two-point limit
            p = np.array([0.73, 0.27])
            lam = np.linalg.eigvalsh(P_matrix(p))
            assert lam[-1] == pytest.approx(2 * 0.73 * 0.27, rel=1e-12)


def test_not_projector_interior_and_vertex():
    rng = np.random.default_rng(9)
    K = 16
    pu = np.full(K, 1.0 / K)
    Pu = P_matrix(pu)
    assert np.abs(Pu @ Pu - Pu / K).max() < 1e-12    # P² = (1/K)P ≠ P
    vertex = np.zeros(K); vertex[3] = 1.0
    Pv = P_matrix(vertex)                            # boundary: P = 0 IS idempotent
    assert np.abs(Pv).max() < 1e-15


def test_secular_roots_match_eigvalsh():
    rng = np.random.default_rng(10)
    p = _p(rng, 24) + rng.uniform(1e-3, 1e-2, 24)
    p /= p.sum()
    lam = secular_roots(p)
    ev = np.sort(np.linalg.eigvalsh(P_matrix(p)))[::-1]
    assert np.abs(lam - ev[:23]).max() < 1e-8
    for x in lam:
        assert abs(secular_residual(p, x)) < 1e-4


def test_p_validation_trust_boundary():
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError):
        P_matrix(np.array([0.5, 0.9]))               # Σp ≠ 1
    with pytest.raises(ValueError):
        P_matrix(np.array([0.5, -0.5]))              # negative
    with pytest.raises(ValueError):
        P_matrix(np.full(1, 1.0))                    # K < 2
