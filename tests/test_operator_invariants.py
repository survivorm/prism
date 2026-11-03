"""operator_checks / cheap_invariants / behavior endpoints — ТЗ §IV.4 (now a
PASSING test: E6 closed) + boundary conditions §III.3/§III.9/§III.10."""

import numpy as np
import pytest

from prism.behavior import KL_lin, b_lin, b_mag, b_res
from prism.cheap_invariants import A_alignment, c_pred
from prism.operator_checks import O1, apply_P, c, cos_J_PJ, frac_K, operator_fraction, s
from prism.p_algebra import P_matrix


@pytest.fixture()
def data():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(32) * 2.0
    p = np.exp(x) / np.exp(x).sum()
    J = rng.standard_normal((8, 32, 16))
    return p, J


def test_triple_identity_one_aggregation(data):
    """§IV.4 — the E6 triple. The identity was NEVER wrong on one aggregation;
    the phantom s²<0 came from mixing an alive-slice c with an all-positions
    median frac. Closed 2026-11-03; test locks the ONE-AGGREGATION semantics."""
    p, J = data
    got = frac_K(J, p)
    want = 1.0 - 2.0 * c(J, p) + s(J, p) ** 2
    assert abs(got - want) < 1e-12
    assert abs(cos_J_PJ(J, p) - c(J, p) / s(J, p)) < 1e-12   # cos ≠ c (§III.3)
    # and per-position the identity also closes, position by position.
    # NOTE: per_position returns RAW energy ratios (sqrt is a property of the
    # aggregate), so s enters as s, not s**2 — the square is already taken.
    assert np.abs(frac_K(J, p, aggregation="per_position")
                  - (1 - 2 * c(J, p, aggregation="per_position")
                     + s(J, p, aggregation="per_position"))).max() < 1e-10


def test_O1_eq_s2_and_energy_vs_norm(data):
    p, J = data
    assert abs(O1(J, p) - s(J, p) ** 2) < 1e-12     # §III: norm vs energy = square
    assert O1(J, p) < s(J, p)                        # small fractions: E < ‖·‖


def test_apply_P_matches_dense_product(data):
    p, J = data
    Pd = P_matrix(p)
    assert np.abs(apply_P(J, p) - np.stack([Pd @ Jt for Jt in J])).max() < 1e-10


def test_operator_fraction_matches_iso_on_iid(data):
    """iid-null consistency: ‖PX‖²/‖X‖² on large isotropic X → ‖P‖²_F/K."""
    p, _ = data
    rng = np.random.default_rng(1)
    Pd = P_matrix(p)
    X = rng.standard_normal((4000, 32, 64))
    got = operator_fraction(Pd, X)
    iso = float((Pd ** 2).sum() / 32)
    assert got == pytest.approx(iso, rel=0.02)


def test_c_pred_and_A(data):
    p, J = data
    cp = c_pred(p)
    assert 0.0 < cp <= 1.0
    assert A_alignment(J, p) == pytest.approx(c(J, p) / cp, rel=1e-12)


def test_b_mag_is_square_and_b_res(data):
    _, J = data
    dz = J[0, :, 0]
    assert b_mag(dz) == pytest.approx(float(dz @ dz), rel=1e-14)   # square, not ‖·‖₂
    assert abs(b_res(b_mag(dz), 2.0) - (b_mag(dz) - 2.0)) < 1e-15


def test_b_lin_corollary_v1(data):
    _, J = data
    Jm = J[0]
    _, S, Vt = np.linalg.svd(Jm, full_matrices=False)
    assert b_lin(Jm, 0.05, Vt[0]) == pytest.approx(0.05 ** 2 * S[0] ** 2, rel=1e-9)


def test_KL_lin_fisher_factor_not_identity(data):
    """§III.10: KL_lin uses F = diag(p) − ppᵀ, so it must equal ½ε²Var_p(Jv)
    and NOT ½ε²‖Jv‖² (bare) — and vanish for Jv ∝ 1 (pure gauge)."""
    p, J = data
    Jm = J[0]
    rng = np.random.default_rng(2)
    v = rng.standard_normal(Jm.shape[1])
    Jv = Jm @ (v / np.linalg.norm(v))
    manual = 0.5 * 0.03 ** 2 * (float(np.dot(p, Jv ** 2)) - float(np.dot(p, Jv)) ** 2)
    assert KL_lin(Jm, p, 0.03, v) == pytest.approx(manual, rel=1e-12)
    f = np.arange(1.0, Jm.shape[1] + 1.0)                   # J[i, :] = f ⇒ Jv ∝ 1
    Jflat = np.tile(f, (Jm.shape[0], 1))
    assert KL_lin(Jflat, p, 0.03, v) == pytest.approx(0.0, abs=1e-10)
