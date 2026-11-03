"""M0 assert-hub: registry of cross-module identities (ТЗ E40_math_core §II.2).

Run the whole registry after ANY kernel change; a status change is an event
for the artifact manifest. Every entry must be FALSIFIABLE — e.g.
``channels_sum`` only counts when ``E_resid``/``E_off_manifold`` are computed
independently (norms of matrix differences), never as conservation leftovers
(ТЗ §III.8, E40_ans §2/§7).

Statuses: ``passing`` | ``failing`` | ``blocked``. The ``triple_identity``
entry was ``blocked_on_E6`` in the ТЗ; E6 was closed 2026-11-03 (the published
table mixed slices; data were self-consistent), so it is ``passing`` here —
the check enforces the LESSON: the identity is asserted on ONE aggregation,
not across median/mean mixes (paper §6 item 23).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .kinetic import kinetic_decomposition
from .operator_checks import O1, apply_P, c, cos_J_PJ, frac_K, s
from .tag import make_tag
from .p_algebra import (
    P_matrix, g_uni, gauge_share, isotropic_reference, secular_residual,
    secular_roots, trace_P, trace_P2,
)
from .transport import holonomy

__all__ = ["INVARIANTS", "Invariant", "run_invariants"]


@dataclass
class Invariant:
    identity: str
    modules: tuple[str, ...]
    check: Callable[[np.random.Generator], None]     # raises AssertionError
    status: str = "passing"
    note: str = ""


def _fixtures(rng):
    K, D, n = 32, 16, 8
    x = rng.standard_normal((K,)) * 2.0
    p = np.exp(x) / np.exp(x).sum()
    J = rng.standard_normal((n, K, D))
    return K, D, p, J


def _chk_O2_eq_O3(rng):
    K, D, p, J = _fixtures(rng)
    g = g_uni(J)
    G = sum(Jt @ Jt.T for Jt in J)
    gs = gauge_share(G)
    assert abs(g - gs) <= 1e-12 + 1e-10 * abs(gs), f"O2≡O3: {g!r} vs {gs!r}"


def _chk_g_uni_gf_zero(rng):
    K, D, p, J = _fixtures(rng)
    g = g_uni(np.stack([apply_P(Jt, p) for Jt in J]))
    assert g < 1e-12, f"g_uni(J^gf) = {g:g} ≠ 0"


def _chk_P_properties(rng):
    K, D, p, J = _fixtures(rng)
    P = P_matrix(p)
    assert np.abs(P @ np.ones(K)).max() < 1e-12, "P·1 ≠ 0"
    assert np.linalg.eigvalsh(P).min() >= -1e-12, "P not PSD"
    # non-projector, falsifiable at interior p: uniform p gives P² = (1/K)P.
    # (Vertex p = e_i ⇒ P = 0 IS idempotent — the boundary excluded here.)
    pu = np.full(K, 1.0 / K)
    Pu = P_matrix(pu)
    assert np.abs(Pu @ Pu - Pu).max() > 1e-6, "P looks like a projector at uniform p"
    assert np.abs(Pu @ Pu - Pu / K).max() < 1e-12, "P² ≠ (1/K)P at uniform p"


def _chk_iso_inline_eq_closed(rng):
    K, D, p, J = _fixtures(rng)
    inline = float((P_matrix(p) ** 2).sum()) / K
    closed = isotropic_reference(p)
    assert abs(inline - closed) <= 1e-9 * closed, f"iso {inline} vs {closed}"


def _chk_p_tilde_sum(rng):
    K, D, p, J = _fixtures(rng)
    mass = float(rng.uniform(0.05, 0.95))              # U128 coverage of a position
    raw = p * mass                                     # sub-stochastic row (P128_RAW)
    tilde = raw / raw.sum()                            # renormalized (P128)
    assert abs(raw.sum() - mass) < 1e-12
    assert abs(tilde.sum() - 1.0) < 1e-12, "Σp̃ ∉ {{1, 1 − mass}}"


def _chk_secular(rng):
    K, D, p, J = _fixtures(rng)
    p = p + rng.uniform(1e-3, 1e-2, size=K)            # force distinct p_i
    p = p / p.sum()
    lam = secular_roots(p)
    ev = np.sort(np.linalg.eigvalsh(P_matrix(p)))[::-1]
    for x in lam:
        # f′ ~ Σp²/(p−λ)² explodes near close poles; 1e-4 is the honest
        # tolerance for f AT the bisection root — the real identity check is
        # the eigvalsh match below.
        assert abs(secular_residual(p, x)) < 1e-4, f"residual {x}: {secular_residual(p, x)}"
    assert np.abs(lam - ev[: K - 1]).max() < 1e-8, "secular roots ≠ eigvalsh"


def _chk_trace_identities(rng):
    K, D, p, J = _fixtures(rng)
    P = P_matrix(p)
    assert abs(float(np.trace(P)) - trace_P(p)) < 1e-12
    assert abs(float(np.trace(P @ P)) - trace_P2(p)) < 1e-9


from scipy.linalg import expm


def _rot(rng, r, angle):
    M = rng.standard_normal((r, r))
    A = M - M.T
    mu = np.abs(np.linalg.eigvalsh(1j * A)).max()      # skew eigenvalue scale
    return expm(angle / mu * A)                        # max rotation angle = angle


def _frames(rng, r=4, d=24, angle=0.15):
    A = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(A)
    return Q, Q @ _rot(rng, r, angle)


def _chk_H_orthogonal(rng):
    V0, V1 = _frames(rng)
    V2 = V1 @ _rot(rng, 4, 0.2)
    H = holonomy(V0, V1, V2)
    assert np.abs(H @ H.T - np.eye(H.shape[0])).max() < 1e-10, "holonomy not orthogonal"


def _blkrot(theta: float) -> np.ndarray:
    """2×2 rotation by θ — a paired-spectrum building block with det = +1."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def _chk_kappa_curv_algebra(rng):
    """κ_curv² = 2r − 2 tr H — UNCONDITIONAL for any H ∈ O(r) (патч v2 §1.4).
    Small-angle holonomy H and large-angle H both pass — no spectral guard."""
    V0, V1 = _frames(rng)
    V2 = V1 @ _rot(rng, V1.shape[1], 2.1)
    for H in (holonomy(V0, V1, V2), np.eye(2)):        # large-angle + H = I (κ=0)
        r = H.shape[0]
        lhs = float(np.linalg.norm(H - np.eye(r), "fro") ** 2)
        rhs = 2.0 * r - 2.0 * float(np.trace(H))
        assert abs(lhs - rhs) < 1e-10 * max(1.0, rhs), f"κ_curv algebra: {lhs} vs {rhs}"


def _chk_kappa_curv_spectral(rng):
    """κ_curv² = 4Σ(1−cosθⱼ) — CONDITIONAL: needs paired spectrum
    (r even, det H = +1, no eigenvalues ±1). Guarded precondition (патч v2 §1.4)."""
    thetas = rng.uniform(0.05, 2.5, size=3)            # strictly inside (0, π)
    H = np.zeros((6, 6))
    for j, th in enumerate(thetas):
        H[2 * j:2 * j + 2, 2 * j:2 * j + 2] = _blkrot(th)
    r = H.shape[0]
    assert r % 2 == 0 and abs(np.linalg.det(H) - 1.0) < 1e-10
    ev = np.linalg.eigvals(H)
    assert np.min(np.abs(np.abs(ev) - 1.0)) < 1e-9      # all on unit circle
    assert np.min(np.abs(ev - 1.0)) > 1e-6 and np.min(np.abs(ev + 1.0)) > 1e-6
    lhs = float(np.linalg.norm(H - np.eye(r), "fro") ** 2)
    rhs = 4.0 * float(np.sum(1.0 - np.cos(thetas)))
    assert abs(lhs - rhs) < 1e-10 * max(1.0, rhs)
    # negative control: the identity FAILS for H with a −1 eigenvalue
    Hbad = np.diag([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) @ H
    bad = float(np.linalg.norm(Hbad - np.eye(6), "fro") ** 2) - 4.0 * float(
        np.sum(1.0 - np.cos(np.concatenate([thetas, [0.0]]))))
    assert abs(bad) > 1e-6, "spectral identity must NOT close through θ=0/π"


def _chk_kappa_cond_invariance(rng):
    """κ_cond(OᵀGO) = κ_cond(G) for orthogonal O (патч v2 §1.4). Tautological
    for exact arithmetic (σ(OᵀGO)=σ(G)); guards our cond usage, not the math."""
    A = rng.standard_normal((8, 8))
    G = A @ A.T + 1e-3 * np.eye(8)
    O, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    assert abs(np.linalg.cond(O.T @ G @ O) - np.linalg.cond(G)) < 1e-8


def _kin_pair(rng, K=12, D=24, r=4, angle=0.12):
    U0, S0, W0t = np.linalg.svd(rng.standard_normal((K, D)), full_matrices=False)
    U0r, V0, S0r = U0[:, :r], W0t[:r].T, S0[:r]      # J = U0r diag(S0r) V0ᵀ
    S1 = S0r * np.exp(rng.normal(0, 0.002, r))       # radial part ≪ angular, so
    #                                     eps3/‖ΔJ‖² ~ O(θ³) is inside the 1e-2
    #                                     calibration band of E40_ans §4
    Jt = U0r @ np.diag(S0r) @ V0.T
    # deterministic rank-1 drift OUT of span(U0)⊗span(V0) (next null directions),
    # so E_off_manifold is nonzero but closes exactly (off ⊥ frame channels):
    up, vp, sp = U0[:, r], W0t[r], float(S0r[-1]) / 3.0   # below the r-th σ
    J1 = ((U0r @ _rot(rng, r, angle)) @ np.diag(S1) @ (V0 @ _rot(rng, r, angle)).T
          + sp * np.outer(up, vp))
    return Jt, J1


def _chk_channels_sum(rng):
    Jt, J1 = _kin_pair(rng)
    dJ2 = float(((J1 - Jt) ** 2).sum())
    out = kinetic_decomposition(Jt, J1, r=4)
    six = (out["E_radial"] + out["E_right"] + out["E_left"] + out["E_cross"]
           + out["E_resid"] + out["E_off_manifold"])
    assert out["flags"]["linearization_valid"], "fixture must be in the valid regime"
    assert abs(dJ2 - six) / dJ2 < 1e-2, f"six-channel closure: {out['closure_rel']:.2e}"
    # non-vacuity: E_resid is an independent matrix difference, and the Gram
    # accounting sums to ‖ΔJ‖² exactly (channel_energy_matrix does that).
    assert out["E_resid"] > 0, "E_resid == 0 exactly? suspect conservation form"


def _chk_r1_cross_zero(rng):
    Jt, J1 = _kin_pair(rng, r=4)
    out = kinetic_decomposition(Jt, J1, r=1)
    assert out["E_cross"] is not None and abs(out["E_cross"]) < 1e-12, \
        f"r=1 cross = {out['E_cross']}"
    assert out["E_right"] < 1e-24 and out["E_left"] < 1e-24, "r=1 angular energies ≠ 0"


def _chk_O1_eq_s2(rng):
    K, D, p, J = _fixtures(rng)
    assert abs(O1(J, p) - s(J, p) ** 2) < 1e-12, "O1 ≠ s²"


def _chk_b_lin_r1(rng):
    # lazy import: the behavioral module is GATED out of tier-0 kernel imports
    # (tests/test_nulls_tag_gate.py::test_behavior_gate_no_import_in_kernel_and_tier0);
    # the hub may execute it without importing
    # it into the kernel namespace at load time.
    from .behavior import b_lin
    K, D, p, J = _fixtures(rng)
    Jm = J[0]
    _, S, Vt = np.linalg.svd(Jm, full_matrices=False)
    v1 = Vt[0]
    eps = 0.037
    got = b_lin(Jm, eps, v1)
    want = eps ** 2 * S[0] ** 2
    assert abs(got - want) <= 1e-9 * want, f"b_lin(r=1): {got} vs {want}"


def _chk_triple_identity(rng):
    K, D, p, J = _fixtures(rng)
    got, want = frac_K(J, p), 1.0 - 2.0 * c(J, p) + s(J, p) ** 2
    assert abs(got - want) < 1e-12, f"frac_K identity: {got!r} vs {want!r}"
    # the cos is the SAME aggregation: cos = c/s on aggregates
    assert abs(cos_J_PJ(J, p) - c(J, p) / s(J, p)) < 1e-12


def _chk_same_slice_aggregation(rng):
    """ТЗ §II.2 (ревизия 2): slice+aggregation identity-relevant in the tag; the
    triple identity must NOT close across mixed aggregations (test IV.6)."""
    t = make_tag(protocol="natural", slice="raw", p_mode="truncated_U128",
                 p_slice="p_t", norm_type="energy")
    assert t.aggregation == "ratio_of_means", "default must be the identity-closing one"
    assert t.slice == "raw"
    try:
        make_tag(protocol="natural", slice="raw", p_mode="truncated_U128",
                 norm_type="energy", aggregation=None)
    except ValueError:
        pass
    else:
        raise AssertionError("aggregation=None must be rejected (identity-relevant)")
    # патч v2 §1.6: p_mode has no "other"; triple reporting needs p_slice.
    try:
        make_tag(protocol="natural", slice="raw", p_mode="other",
                 norm_type="energy")
    except ValueError:
        pass
    else:
        raise AssertionError("p_mode='other' must be a stop condition, not a tag")
    from .tag import require_triple_tag
    try:
        require_triple_tag(make_tag(protocol="natural", slice="raw",
                                    p_mode="truncated_U128", norm_type="energy"))
    except ValueError:
        pass
    else:
        raise AssertionError("triple number without p_slice must be rejected")
    require_triple_tag(t)
    # negative half (IV.6): mixing median c/s with aggregated frac_K breaks the identity
    K, D, p, J = _fixtures(rng)
    mixed = 1.0 - 2.0 * c(J, p, "median") + s(J, p, "median") ** 2
    assert abs(frac_K(J, p) - mixed) > 1e-9, "mixed aggregations must NOT close the identity"


def _chk_r_can_definition(rng):
    """R_can must be the λ-share of the CANONICAL slice E_K = (PJ)(PJ)ᵀ
    (ТЗ §8.1, paper §2.11.1) — not of (I−P)J, which is ≈ raw J (audit M1)."""
    from .profiles import R_can
    K, D, p, J = _fixtures(rng)
    Jm = J[0]
    PJ = apply_P(Jm, p)
    G = PJ @ PJ.T
    want = float(np.linalg.eigvalsh(G)[-1] / np.trace(G))
    got = R_can(Jm, p)
    assert abs(got - want) < 1e-12, f"R_can: {got} vs {want}"
    # negative half: the (I−P)J slice is a DIFFERENT object and must not agree
    Kc = Jm - PJ
    Gc = Kc @ Kc.T
    other = float(np.linalg.eigvalsh(Gc)[-1] / np.trace(Gc))
    assert abs(got - other) > 1e-6, "R_can collapsed onto the (I−P)J slice again"


def _chk_agg_closure_matrix(rng):
    """Documented closure matrix (audit M3): the triple identity closes for
    ratio_of_means and per-position, NOT for mean (Jensen on s) — the kernel
    must behave as its docstrings claim. The two closures are written with
    DIFFERENT members on purpose: ``per_position`` deliberately ignores ``root``
    and returns RAW energy ratios (``s(..., "per_position") == O1(...)``), so
    positionally the identity is ``frac_t = 1 − 2c_t + O1_t``, while pooled it is
    ``frac = 1 − 2c + s²`` with s the root of the pooled ratio."""
    K, D, p, J = _fixtures(rng)
    got = frac_K(J, p, "ratio_of_means")
    want = 1.0 - 2.0 * c(J, p, "ratio_of_means") + s(J, p, "ratio_of_means") ** 2
    err = float(np.abs(got - want).max())
    assert err < 1e-12, f"ratio_of_means must close, err={err}"
    got = frac_K(J, p, "per_position")
    want = (1.0 - 2.0 * c(J, p, "per_position")
            + O1(J, p, "per_position"))
    err = float(np.abs(got - want).max())
    assert err < 1e-12, f"per_position must close, err={err}"
    assert float(np.abs(s(J, p, "per_position")
                       - O1(J, p, "per_position")).max()) < 1e-15, \
        "per_position must stay root-free (raw energy ratios)"
    # mean: s(mean) is the mean of the per-position ROOT — Jensen gap is real
    assert s(J, p, "mean") ** 2 <= O1(J, p, "mean") + 1e-15, "Jensen direction flipped"
    assert abs(frac_K(J, p, "mean") - (1.0 - 2.0 * c(J, p, "mean")
                                       + s(J, p, "mean") ** 2)) > 1e-12, \
        "mean closure must NOT hold silently (it would mask the Jensen gap)"


def _chk_mp_edge_labelled(rng):
    """MP edges under the XᵀX/n convention, pinned with n (ТЗ §III.11; the
    retracted 1.4425/1.45 pair must not creep back — audit M8)."""
    from .nulls_and_stats import mp_upper_edge
    assert abs(mp_upper_edge(6, 32) - 2.053525) < 1e-6
    assert abs(mp_upper_edge(8, 32) - 2.250000) < 1e-6
    assert abs(mp_upper_edge(6, 192) - 1.384803) < 1e-6
    assert abs(mp_upper_edge(8, 192) - 1.449915) < 1e-6


def _chk_svd_bundle_contract(rng):
    """`svd_bundle` величин фазы (E-40.2/0.4 их отчитывают), а их в реестре не
    было — пробел M13. Пиним: бандл реконструирует J, `energy` = доля топ-r, нулевой
    спектр имеет ранг 0 (Q1), `principal_angles` требует ортонормированный фрейм
    (M10), и `stable` — σ-форма, а НЕ σ²-форма `kinetic.rotation_unique` (M9)."""
    from .svd_bundle import effective_rank, principal_angles, svd_bundle
    K, D, p, J = _fixtures(rng)
    Jm = J[0]
    b = svd_bundle(Jm, rank=4)
    full = svd_bundle(Jm, rank=min(K, D))            # ранг = полная размерность ⇒ J точно
    rec = (full["U"] * full["Sigma"]) @ full["V"].T
    assert np.linalg.norm(Jm - rec) / np.linalg.norm(Jm) < 1e-10, "bundle must rebuild J"
    assert b["energy"] < 1.0 - 1e-9, "top-r energy share of a full-rank J must be < 1"
    s_full = np.linalg.svd(Jm, compute_uv=False)
    want = float((b["Sigma"] ** 2).sum() / (s_full ** 2).sum())
    assert abs(b["energy"] - want) < 1e-12, "energy must be the top-r σ² share"
    assert effective_rank(np.zeros(5)) == 0, "zero spectrum must report rank 0"
    V = full["V"]
    ang = principal_angles(V, V)
    assert np.all(ang > 1 - 1e-9), "principal_angles(V, V) must be ~1"
    try:
        principal_angles(V * 2.0, V)                 # non-orthonormal frame
        raise AssertionError("principal_angles accepted a non-orthonormal frame")
    except ValueError:
        pass
    # M9: на [1.0, 0.991] конвенции σ и σ² расходятся по построению — это пинит,
    # что `stable` остаётся σ-формой (E-38c gap_min), а не тихой σ²-формой.
    Jn = np.zeros((8, 8))
    Jn[0, 0], Jn[1, 1] = 1.0, 0.991
    st = svd_bundle(Jn, rank=2)["stable"]
    sigma_form = abs(1.0 - 0.991) > 1e-2                        # False
    sigma2_form = abs(1.0 - 0.991 ** 2) / 1.0 > 1e-2            # True
    assert st == sigma_form and sigma_form != sigma2_form, \
        "svd_bundle.stable must stay the σ-form (M9 conventions must not merge)"


INVARIANTS: dict[str, Invariant] = {
    "O2_eq_O3": Invariant("g_uni(J) == gauge_share(G=ΣJJᵀ)", ("p_algebra",),
                          _chk_O2_eq_O3),
    "g_uni_gf_zero": Invariant("g_uni(P(p)J) ≡ 0", ("p_algebra", "operator_checks"),
                               _chk_g_uni_gf_zero),
    "P_properties": Invariant("P·1=0, P⪰0, P²≠P (interior p; vertex P=0 excluded)",
                              ("p_algebra",), _chk_P_properties),
    "iso_inline_eq_closed": Invariant("iso inline == iso closed form", ("p_algebra",),
                                      _chk_iso_inline_eq_closed),
    "p_tilde_sum": Invariant("Σp̃ ∈ {1, 1 − mass}", ("p_algebra", "tag"), _chk_p_tilde_sum),
    "secular_equation": Invariant("Σ pᵢ²/(pᵢ−λ) = 1 at λ ∈ eig(P)∖{0}", ("p_algebra",),
                                  _chk_secular),
    "trace_identities": Invariant("tr P = 1−Σp²; tr P² = Σp²−2Σp³+(Σp²)²", ("p_algebra",),
                                  _chk_trace_identities),
    "H_orthogonal": Invariant("H Hᵀ = I", ("transport",), _chk_H_orthogonal),
    "kappa_curv_algebra": Invariant("κ_curv² = 2r − 2trH (любая H ∈ O(r))",
                                    ("transport",), _chk_kappa_curv_algebra),
    "kappa_curv_spectral": Invariant("κ_curv² = 4Σ(1−cosθ) (парный спектр: r чётн., det +1, ±1 нет)",
                                     ("transport",), _chk_kappa_curv_spectral,
                                     note="патч v2 §1.4: условное; precondition guard + "
                                          "negative control θ=0/π"),
    "kappa_cond_invariance": Invariant("κ_cond(OᵀGO) = κ_cond(G), O орт.",
                                       ("p_algebra", "spectral"), _chk_kappa_cond_invariance,
                                       note="vacuous-by-invariance check kept per патч v2; "
                                            "κ(OᵀHO)=κ(H) запись удалена по его же требованию"),
    "channels_sum": Invariant("Σ6 channels + eps3 = ‖ΔJ‖², E_resid/E_off independent",
                              ("kinetic",), _chk_channels_sum,
                              note="6 channels per docs/E40_ans.md, not the ТЗ's 5"),
    "r1_cross_zero": Invariant("r=1 ⇒ E_cross = 0", ("kinetic",), _chk_r1_cross_zero),
    "O1_eq_s2": Invariant("O1 ≡ s² on one aggregation", ("operator_checks",), _chk_O1_eq_s2),
    "b_lin_r1": Invariant("b_lin(v=v₁) = ε²σ₁²", ("behavior",), _chk_b_lin_r1),
    "triple_identity": Invariant("frac_K = 1 − 2c + s² (one aggregation, one slice)",
                                 ("cheap_invariants", "operator_checks"),
                                 _chk_triple_identity,
                                 note="was blocked_on_E6; E6 closed 2026-11-03"),
    "same_slice_aggregation": Invariant("tag carries slice+aggregation; mixed aggregations "
                                        "must not close the triple identity (IV.6)",
                                        ("tag", "cheap_invariants"),
                                        _chk_same_slice_aggregation,
                                        note="NEW ТЗ §II.2 ревизия 2"),
    "R_can_definition": Invariant("R_can = λshare((PJ)(PJ)ᵀ), ≠ λshare(((I−P)J)…) ",
                                  ("profiles", "operator_checks"), _chk_r_can_definition,
                                  note="NEW audit_core_math_v1 M1 (critical): wrong slice "
                                       "made R_can ≈ R_raw and E-40.6 blind"),
    "agg_closure_matrix": Invariant("triple closes for rom/per_position, Jensen gap for "
                                    "mean is real and documented", ("operator_checks",),
                                    _chk_agg_closure_matrix,
                                    note="NEW audit M3"),
    "mp_edge_labelled": Invariant("MP edges pinned with explicit n (2.0535/2.25/1.3848/1.4499)",
                                  ("nulls_and_stats",), _chk_mp_edge_labelled,
                                  note="NEW audit M8"),
    "svd_bundle_contract": Invariant("svd_bundle rebuilds J; energy = top-r σ² share; "
                                     "rank(0)=0; principal_angles needs an orthonormal "
                                     "frame; stable is the σ-form",
                                     ("svd_bundle",), _chk_svd_bundle_contract,
                                     note="NEW audit_core_math_v1 M13 (closes the last "
                                          "phase-magnitude module without registry cover)"),
}


def run_invariants(seed: int = 0) -> dict[str, dict]:
    """Run every registered identity on synthetic fixtures; a declared
    ``passing`` entry that raises is a FAILURE of the hub (returned ok=False,
    the pytest driver asserts all-ok)."""
    out = {}
    for name, inv in INVARIANTS.items():
        rng = np.random.default_rng(seed + sum(map(ord, name)))
        try:
            inv.check(rng)
            out[name] = {"ok": True, "detail": "ok", "declared": inv.status}
        except AssertionError as e:
            out[name] = {"ok": False, "detail": str(e), "declared": inv.status}
    return out
