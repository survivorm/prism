"""Kinetic SVD decomposition with SIX channels (ТЗ E40_math_core §I.7/§II.6,
M6; full specification in docs/E40_ans.md — that file supersedes the ТЗ
wherever they differ).

Linearization recipe (E40_ans §1): with aligned frames,

    ΔJ_lin = ΔΣ + M Σ̄ − Σ̄ N,   M = log polar(U_tᵀ U_{t+1}),  N likewise,
    Σ̄ = (Σ_t + Σ_{t+1})/2.

Channels (ENERGIES, never shares):
    E_radial = ‖ΔΣ‖², E_right = ‖Σ̄N‖², E_left = ‖MΣ̄‖²,
    E_cross  = 2 tr(Σ̄ M Σ̄ N)  (SIGNED; report signed and abs),
    E_resid  = ‖U_tᵀ ΔJ V_t − ΔJ_lin‖²_F            (independent, matrix diff),
    E_off_manifold = ‖ΔJ − U_t U_tᵀ ΔJ V_t V_tᵀ‖²_F (independent; mandatory at
                                                    r ≪ D, E40_ans §3),
    eps3_correction = ‖ΔJ‖² − Σ6                    (diagnostic, NOT a channel).

Validity flags (E40_ans §5): ``linearization_valid`` = max principal angle
< π/2 − eps_angle (else log is off the principal branch); ``rotation_unique``
= min |σᵢ²−σⱼ²| > eps_gap·σ₁² (else left/right/cross are basis-ambiguous
inside near-degenerate blocks). On the real cache E-40.5 PRE-REGISTERS:
L6–L22 invalid (Δ_SVD ≈ 1.4–1.55), L23 valid-but-ambiguous (0.266, near
degeneracy) — only E_radial and E_off_manifold are interpretation-safe
anywhere. Getting ``linearization_valid = false`` on L6–L22 is the expected
result, not a failure.
"""

from __future__ import annotations

import math

import numpy as np

from .svd_bundle import principal_angles
from .transport import align_signs_and_columns, polar_factor, transport_log

__all__ = ["kinetic_decomposition", "channel_energy_matrix"]

_EPS = 1e-12


def _fsq(A: np.ndarray) -> float:
    return float((A ** 2).sum())


def _check_svd(triple, J: np.ndarray, name: str):
    """Validate a caller-supplied ``np.linalg.svd`` triple against ``J`` — a
    stale or swapped ``svds`` must fail loud, not silently renumber channels.
    Shape check alone cannot see a swap of same-shaped inputs, so the triple is
    checked by reconstruction (one matmul, ~10× cheaper than the SVD)."""
    U, S, W = triple
    U = np.asarray(U, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    K, D = J.shape
    if U.shape != (K, min(K, D)) or S.shape != (min(K, D),) or W.shape != (min(K, D), D):
        raise ValueError(f"{name}: SVD shapes {U.shape}/{S.shape}/{W.shape} "
                         f"do not match [K,D]=[{K},{D}] (full_matrices=False)")
    rec = float(np.linalg.norm((U * S) @ W - J)
                / max(np.linalg.norm(J), 1e-300))
    if rec > 1e-8:
        raise ValueError(f"{name}: U diag(S) Vᵀ ≠ J (rel {rec:.2e}) — stale or "
                         "swapped svds (swapping two same-shaped triples is "
                         "otherwise undetectable)")
    return U, S, W


def _min_gap_ratio(S: np.ndarray) -> float:
    if S.size < 2:
        return 1.0
    s2 = S ** 2
    d = s2[:-1] - s2[1:]
    return float(d.min() / s2[0]) if d.size else 1.0


def kinetic_decomposition(J_t: np.ndarray, J_t1: np.ndarray, r: int = 16,
                          eps_angle: float = 0.3, eps_gap: float = 1e-2,
                          svds: tuple | None = None) -> dict:
    """Six-channel kinetic accounting between consecutive Jacobians.

    ``J_t``, ``J_t1``: ``[K, D]`` in a common transport frame (the E-40.5
    pipeline transports before calling; nothing here is model-aware).
    Returns every field listed in E40_ans §7/§8; when a flag invalidates a
    channel the number is still computed (if real) for diagnostics, but the
    flags tell the report what may be interpreted.

    ``svds=((U0,S0,W0),(U1,S1,W1))``: precomputed ``np.linalg.svd(...,
    full_matrices=False)`` of the two fp64 inputs, for callers that sweep a
    chain of positions where each ``J`` appears in two consecutive pairs — a
    full [K,D] SVD is the whole cost of this function. Bit-identical to the
    internal call when the cached triple came from the same fp64 array.
    """
    J_t = np.asarray(J_t, dtype=np.float64)
    J_t1 = np.asarray(J_t1, dtype=np.float64)
    if J_t.shape != J_t1.shape or J_t.ndim != 2:
        raise ValueError(f"J_t, J_t1 must be matching [K, D], got {J_t.shape}, {J_t.shape}")
    K, D = J_t.shape
    if not (0 < r <= min(K, D)):
        raise ValueError(f"r must be in (0, {min(K, D)}], got {r}")

    if svds is None:
        U0, S0, W0 = np.linalg.svd(J_t, full_matrices=False)
        U1, S1, W1 = np.linalg.svd(J_t1, full_matrices=False)
    else:
        (U0, S0, W0) = _check_svd(svds[0], J_t, "svds[0]")
        (U1, S1, W1) = _check_svd(svds[1], J_t1, "svds[1]")
    V0, V1 = W0[:r].T, W1[:r].T                       # [D, r]
    U0a = U0[:, :r]
    U1a = U1[:, :r]

    U0a, U1a, fU = align_signs_and_columns(U0a, U1a)
    V0, V1, fV = align_signs_and_columns(V0, V1)
    order_cross = bool(fU["order_cross"] or fV["order_cross"])
    # M2 fix (audit_core_math_v1): σ must travel WITH the matched columns —
    # column j of the aligned t+1 frame is S1[perm[j]], not S1[j]. σ follows
    # the U-side perm; a U/V perm disagreement means the singular triplets
    # themselves split (near-degenerate block) and is flagged.
    permU, permV = fU["perm"], fV["perm"]
    sigma_perm_mismatch = bool(permU != permV)

    dJ = J_t1 - J_t
    Sig_t = np.diag(S0[:r])
    Sig_t1 = np.diag(np.asarray(S1[:r])[permU])
    dSig = Sig_t1 - Sig_t
    Sb = 0.5 * (Sig_t + Sig_t1)

    # validity A: rotation size (E40_ans §5.1 — MAX principal angle threshold)
    cos_pa = principal_angles(U0a, U1a)
    max_angle = float(np.arccos(np.clip(cos_pa.min(), -1.0, 1.0)))
    linearization_valid = bool(max_angle < math.pi / 2 - eps_angle)
    # validity B: σ-separation inside the top-r block (both t's must separate)
    gap_ratio = min(_min_gap_ratio(S0[:r]), _min_gap_ratio(S1[:r]))
    rotation_unique = bool(gap_ratio > eps_gap)

    M = N = None
    unstable_transport = False
    if linearization_valid:
        try:
            M = transport_log(polar_factor(U0a.T @ U1a))
            N = transport_log(polar_factor(V0.T @ V1))
        except ValueError:
            # Either side may fail (U-transport is computed first): the linearization is
            # rejected as a whole, so BOTH must go to None — leaving `M` set while `N` is
            # None makes the `M is not None` branch below compute `Sb @ None` (TypeError on
            # a 0-dim object array) instead of reporting the rejected cell.
            M = N = None
            unstable_transport = True
    # r = 1: after alignment Q = {+1}, log = 0 — exact, no try needed
    if r == 1:
        M = np.zeros((1, 1))
        N = np.zeros((1, 1))
        linearization_valid = True

    B = U0a.T @ dJ @ V0                                # coords of ΔJ in the t-frame
    out: dict = {"r": r, "max_principal_angle": max_angle,
                 "min_sigma_gap_ratio": gap_ratio,
                 "flags": {"linearization_valid": linearization_valid,
                           "rotation_unique": rotation_unique,
                           "order_cross_U": fU["order_cross"],
                           "order_cross_V": fV["order_cross"],
                           "unstable_transport": unstable_transport,
                           "sign_flips_U": fU["sign_flips"],
                           "sign_flips_V": fV["sign_flips"],
                           "perm_U": permU,
                           "perm_V": permV,
                           "sigma_perm_mismatch": sigma_perm_mismatch}}
    dJ_proj = U0a @ (U0a.T @ dJ @ V0) @ V0.T
    E_off = _fsq(dJ - dJ_proj)
    E_radial = _fsq(dSig)

    if M is not None:
        dJ_lin = dSig + M @ Sb - Sb @ N
        E_left = _fsq(M @ Sb)
        E_right = _fsq(Sb @ N)
        E_cross = float(2.0 * np.trace(Sb @ M @ Sb @ N))   # signed (E40_ans §1)
        E_resid = _fsq(B - dJ_lin)
    else:
        dJ_lin = E_left = E_right = E_cross = E_resid = None

    six_sum = (E_radial + E_off + E_resid + E_left + E_right + E_cross
               if dJ_lin is not None else None)
    out.update({
        "E_radial": E_radial,
        "E_right": E_right,
        "E_left": E_left,
        "E_cross": E_cross,
        "E_cross_abs": abs(E_cross) if E_cross is not None else None,
        "E_resid": E_resid,
        "E_off_manifold": E_off,
        "norm_dJ_sq": _fsq(dJ),
        "eps3_correction": (_fsq(dJ) - six_sum) if six_sum is not None else None,
        "diag": {"normM_fro": float(np.linalg.norm(M, "fro")) if M is not None else None,
                 "normN_fro": float(np.linalg.norm(N, "fro")) if N is not None else None},
        # The transport objects themselves. Consumers that build a NULL over the
        # left channel must perturb the ACTUAL M at the ACTUAL Σ̄ — re-deriving
        # them in the script duplicates the pipeline and silently loses the
        # σ↔column permutation (the M2 lesson), so they are exposed here.
        "M_transport": None if M is None else np.array(M),
        "sigma_bar": np.array(Sb),
    })
    # reportable set (E40_ans §5.3/§8)
    if not linearization_valid:
        out["channels_reportable"] = ["E_radial", "E_off_manifold"]
    elif not rotation_unique:
        out["channels_reportable"] = ["E_radial", "E_off_manifold", "E_resid"]
        out["ambiguous"] = ["E_right", "E_left", "E_cross"]
    else:
        out["channels_reportable"] = ["E_radial", "E_right", "E_left", "E_cross",
                                      "E_resid", "E_off_manifold"]
    if six_sum is not None:
        # non-vacuous closure: eps3 is a DIAGNOSTIC residual, not a channel —
        # the assertion bounds it, it never redefines E_resid (E40_ans §4).
        out["closure_rel"] = abs(_fsq(dJ) - six_sum) / max(_fsq(dJ), _EPS)
    if sigma_perm_mismatch and "ambiguous" not in out:
        out["ambiguous"] = []
    if sigma_perm_mismatch:
        out["ambiguous"].append("E_radial")
        if "E_radial" in out["channels_reportable"]:
            out["channels_reportable"].remove("E_radial")
    return out


def channel_energy_matrix(J_t: np.ndarray, J_t1: np.ndarray, r: int = 16,
                          svds: tuple | None = None) -> np.ndarray:
    """5×5 Gram of the vector components ``A1..A5`` with
    ``ΔJ = A1 + A2 + A3 + A4 + A5``:
    ``radial`` = ΔΣ embedded, ``left`` = U M Σ̄ Vᵀ, ``right`` = −U Σ̄ N Vᵀ,
    ``resid`` = U (B − ΔJ_lin) Vᵀ, ``off`` = ΔJ − ΔJ_proj.
    ``sum(matrix)`` == ``‖ΔJ‖²`` EXACTLY (accounting identity, all cross terms
    visible off-diagonal); the diagonal holds channel energies,
    ``2·[left, right]`` is ``E_cross``. Energies, not shares (shares are
    compositional, rank ≤ 4 by construction — ТЗ §I.7).

    Slice caveat (патч v2 §3.3): this 5×5 includes ``off``, so it closes
    ``‖ΔJ‖²``. Restricting the Gram to the four in-frame vectors
    {radial, left, right, resid} closes only ``‖ΔJ_proj‖²`` — and on real
    exp27b data (r = 16 ≪ D = 1024, Δ_SVD ≈ π/2 on L6–L22)
    ``E_off ≈ ‖ΔJ‖²/2``, i.e. "4×4 closes ‖ΔJ‖²" is FALSE there and TRUE on
    synthetic rank(ΔJ) ≤ r. Full closure always needs Σ6 + eps3.
    """
    J_t = np.asarray(J_t, dtype=np.float64)
    J_t1 = np.asarray(J_t1, dtype=np.float64)
    K, D = J_t.shape
    if svds is None:
        U0, S0, W0 = np.linalg.svd(J_t, full_matrices=False)
        U1, S1, W1 = np.linalg.svd(J_t1, full_matrices=False)
    else:
        (U0, S0, W0) = _check_svd(svds[0], J_t, "svds[0]")
        (U1, S1, W1) = _check_svd(svds[1], J_t1, "svds[1]")
    U0a, S0r, V0 = U0[:, :r], S0[:r], W0[:r].T
    U1a, _, fU = align_signs_and_columns(U0a, U1[:, :r])
    V1 = align_signs_and_columns(V0, W1[:r].T)[1]
    S1r = S1[:r][fU["perm"]]                      # σ follows the frame (M2)
    dJ = J_t1 - J_t
    M = transport_log(polar_factor(U0a.T @ U1a))
    N = transport_log(polar_factor(V0.T @ V1))
    dSig = np.diag(S1r - S0r)
    Sb = 0.5 * np.diag(S0r + S1r)
    B = U0a.T @ dJ @ V0
    dJ_lin = dSig + M @ Sb - Sb @ N
    embed = lambda X: U0a @ X @ V0.T                    # noqa: E731
    A1 = embed(dSig)
    A2 = embed(M @ Sb)
    A3 = embed(-Sb @ N)
    A4 = embed(B - dJ_lin)
    A5 = dJ - embed(B)
    comps = [A1, A2, A3, A4, A5]
    G = np.array([[float((Ai * Aj).sum()) for Aj in comps] for Ai in comps])
    return G
