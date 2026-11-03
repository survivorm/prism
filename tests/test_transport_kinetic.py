"""transport / kinetic — ТЗ §IV.5 (synthetic transport with known log Q),
r=1 control, E_resid/E_off independence (E40_ans §2/§3/§6)."""

import numpy as np
import pytest
from scipy.linalg import expm

from prism.invariants import _kin_pair  # synthetic-pair fixture builder
from prism.kinetic import channel_energy_matrix, kinetic_decomposition
from prism.transport import (
    align_signs_and_columns, holonomy, kappa, polar_factor, transport_log,
)


def _rot(rng, r, angle):
    M = rng.standard_normal((r, r))
    A = M - M.T
    mu = np.abs(np.linalg.eigvalsh(1j * A)).max()
    return expm(angle / mu * A)                       # exact max angle


def test_polar_log_roundtrip_known_Q():
    """§IV.5: build Q = exp(M_known), embed as polar(U0ᵀU1), reconstruct log."""
    rng = np.random.default_rng(20)
    r = 5
    M = rng.standard_normal((r, r)) - np.eye(r)
    M = M - M.T                                        # skew, angles < π
    Q = expm(0.4 * M / np.abs(np.linalg.eigvalsh(1j * M)).max())
    A = rng.standard_normal((20, r))
    U0, _ = np.linalg.qr(A)
    U1 = np.linalg.qr(U0 @ Q)[0]
    U0a, U1a, _ = align_signs_and_columns(U0, U1)      # gate first (QR flips signs)
    Qg = polar_factor(U0a.T @ U1a)
    assert np.abs(Qg - Q).max() < 1e-10
    assert np.abs(transport_log(Qg) - 0.4 * M / np.abs(np.linalg.eigvalsh(1j * M)).max()).max() < 1e-8


def test_align_signs_and_columns_recovers_flips_and_perm():
    rng = np.random.default_rng(21)
    U0, _ = np.linalg.qr(rng.standard_normal((15, 5)))
    perm = np.array([2, 0, 4, 1, 3])
    signs = np.array([1, -1, 1, -1, 1])
    U1 = U0[:, perm] * signs
    U0a, U1a, flags = align_signs_and_columns(U0, U1)
    assert np.abs(U1a - U0).max() < 1e-12              # frames identified again
    # flips are reported in the MATCHED (U0-gauge) coordinates: perm maps
    # signs[1], signs[3] onto positions perm[1]=0, perm[3]=1
    assert flags["order_cross"] and flags["sign_flips"] == [0, 1]
    Q = polar_factor(U0a.T @ U1a)
    assert np.abs(Q - np.eye(5)).max() < 1e-10         # transport is now log-safe


def test_align_refuses_orthogonal_pair():
    rng = np.random.default_rng(22)
    U0, _ = np.linalg.qr(rng.standard_normal((8, 4)))
    # column 0 replaced by a span(U0)⊥ vector: orthonormal, one matched pair
    # has zero overlap — column identity is genuinely lost
    U0perp, _ = np.linalg.qr(np.eye(8) - U0 @ U0.T)
    U1 = U0.copy()
    U1[:, 0] = U0perp[:, 0]
    with pytest.raises(ValueError, match="orthogonal"):
        align_signs_and_columns(U0, U1)


def test_transport_log_refuses_branch_cut():
    with pytest.raises(ValueError, match="branch"):
        transport_log(np.diag([1.0, -1.0, 1.0]))     # eig = −1 ⇒ log undefined


def test_r1_cross_zero_and_no_angular():
    rng = np.random.default_rng(23)
    Jt, J1 = _kin_pair(rng)
    out = kinetic_decomposition(Jt, J1, r=1)
    assert out["E_cross"] == pytest.approx(0.0, abs=1e-12)
    assert out["E_left"] < 1e-24 and out["E_right"] < 1e-24
    assert "E_radial" in out["channels_reportable"]


def test_six_channel_closure_and_independence():
    rng = np.random.default_rng(24)
    closures, offs = [], []
    for _ in range(6):
        Jt, J1 = _kin_pair(rng)
        o = kinetic_decomposition(Jt, J1, r=4)
        assert o["flags"]["linearization_valid"]
        dJ2 = float(((J1 - Jt) ** 2).sum())
        six = sum(o[k] for k in ("E_radial", "E_right", "E_left", "E_cross",
                                 "E_resid", "E_off_manifold"))
        closures.append(abs(dJ2 - six) / dJ2)
        offs.append(o["E_off_manifold"])
        # E_off > 0 comes from the injected out-of-frame rank-1 drift: if
        # E_off were a conservation leftover it would be 0 by construction.
    assert max(closures) < 1e-2                        # E40_ans §4 ε_tol
    assert all(e > 0 for e in offs)


def test_channel_energy_matrix_closes_exactly():
    rng = np.random.default_rng(25)
    Jt, J1 = _kin_pair(rng)
    G = channel_energy_matrix(Jt, J1, r=4)
    dJ2 = float(((J1 - Jt) ** 2).sum())
    assert G.sum() == pytest.approx(dJ2, rel=1e-12)   # accounting, not linearization
    assert G.shape == (5, 5)
    assert np.abs(G - G.T).max() < 1e-12


def test_holonomy_and_kappa():
    rng = np.random.default_rng(26)
    V0, _ = np.linalg.qr(rng.standard_normal((18, 4)))
    V1 = V0 @ _rot(rng, 4, 0.2)
    V2 = V1 @ _rot(rng, 4, 0.25)
    H = holonomy(V0, V1, V2)
    assert np.abs(H @ H.T - np.eye(4)).max() < 1e-10
    O, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    assert kappa(O.T @ H @ O) == pytest.approx(kappa(H), abs=1e-10)


def test_invalid_regime_reports_only_safe_channels():
    """Δ_SVD near π/2 ⇒ linearization_valid=False, reportable = {radial, off}
    (the pre-registered L6–L22 behaviour; E40_ans §5.3/§9.6)."""
    rng = np.random.default_rng(27)
    Jt, _ = _kin_pair(rng)
    U, S, Wt = np.linalg.svd(Jt, full_matrices=False)
    # push the left frame almost out of span: U' mixes U0r with U-perp so the
    # max principal angle exceeds π/2 − 0.3 ≈ 1.27
    Umix, _ = np.linalg.qr(U[:, :4] + 4.0 * U[:, 4:8])
    J1 = Umix @ np.diag(S[:4]) @ Wt[:4]
    out = kinetic_decomposition(Jt, J1, r=4)
    assert not out["flags"]["linearization_valid"]
    assert out["channels_reportable"] == ["E_radial", "E_off_manifold"]
