"""Module 3: holonomy — pure geometric primitives (TZ part 2, docs/new_functions_2.md).

Procrustes-based holonomy (κ = ‖R_total − I‖_F) in **subspace space**
(Grassmannian Gr(r, K) for left U_r, Gr(r, D_S) for right V_r).

No forward passes, no model dependency: only tensor math over already-
computed subspaces. Concrete paths (W₀→W₁₀₀→W₀) and loops (V₁,V₂ vs
PC₁,PC₂) live in experiments/exp_T2.py — they use the primitives here.

Gauge invariance: replacing U_i with U_i @ Q_i (any orthogonal Q_i, a
basis change inside the subspace) leaves every κ unchanged. This is a
fundamental property; violating it is a bug (selfcheck 3.6.15).

Related: paper.md §8.6, §8.7.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Sequence

import numpy as np
import torch

__all__ = [
    "procrustes_align",
    "holonomy_accumulate",
    "holonomy_spectrum",
    "geodesic_curvature",
    "eigenframe_basis",
    "holonomy_along_path",
    "holonomy_along_2d_loop",
]


def _validate_orthonormal(U: torch.Tensor, tol: float) -> None:
    """‖U^T U − I‖_F < tol, else ValueError.

    Silent work on non-orthonormal input gives a wrong R (the SVD polar
    factor solves the orthonormal problem); fail loudly.
    """
    dev = U.device
    gram = U.T @ U
    err = torch.norm(gram - torch.eye(gram.shape[0], device=dev, dtype=gram.dtype),
                     p="fro")
    if not torch.isfinite(err) or err.item() >= tol:
        raise ValueError(
            f"input not orthonormal: ‖U^T U − I‖_F = {err.item():.3e} (tol {tol:.0e})"
        )


def _check_loop_basis(b1: torch.Tensor, b2: torch.Tensor, tol: float = 1e-4) -> None:
    """‖b1‖ ≈ 1, ‖b2‖ ≈ 1, |b1·b2| < tol — else ValueError.

    A non-orthogonal loop pair makes κ uninterpretable as holonomy
    (the loop is not a coordinate chart on the sphere).
    """
    n1, n2 = torch.norm(b1).item(), torch.norm(b2).item()
    dot = torch.dot(b1, b2).abs().item()
    if abs(n1 - 1.0) > tol or abs(n2 - 1.0) > tol:
        raise ValueError(f"loop basis not unit: ‖b1‖={n1:.6f}, ‖b2‖={n2:.6f}")
    if dot >= tol:
        raise ValueError(f"loop basis not orthogonal: |b1·b2| = {dot:.3e} (tol {tol:.0e})")


def procrustes_align(
    U1: torch.Tensor, U2: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Procrustes alignment of two orthonormal subspaces (TZ §3.3).

    Args:
        U1, U2: [K, r] with U^T U ≈ I_r, same shape.

    Returns:
        (R, residual_sq): R is [r, r] orthogonal, minimizing
        ‖U1 @ R − U2‖_F; residual_sq = ‖U1 R − U2‖_F² as a Python float.

    Semantics (subspaces, not bases): R is **invariant** to a joint sign
    flip of columns of U1 and U2, and residual_sq is invariant to any
    column sign changes — the distance is between the subspaces span(U1)
    and span(U2). If U1 and U2 span the same subspace in different
    bases, residual_sq ≈ 0 even though R ≠ I (R is that basis change).

    Algorithm: M = U2^T U1 [r, r]; SVD M = P Σ Q^T; R = Q P^T (polar
    factor, O(r) — no det-fix; with sign-gauged bases the SO(r)
    projection flips a column and breaks H = I for identical subspaces).
    SVD computed in float64 for stability; R cast back to U1.dtype.

    Degenerate SVD (s_i = s_j, i ≠ j): R is not unique; this function
    returns one valid minimizer. residual_sq is always well-defined:
    ‖U1 R − U2‖_F² = 2r − 2·tr(M^T R) = 2r − 2·Σs exactly (the clamp to
    0 guards only against ~1e-16 floating noise).

    Ill-conditioned (subspaces nearly orthogonal, all s_i ≈ 0): R is
    numerically unstable; residual_sq → 2r (the maximum) and remains
    meaningful. Consumers should not rely on R in that regime.

    Errors:
        ValueError: shape mismatch, U1.dim() != 2, non-orthonormal input
            (tol 1e-4 for float32, 1e-6 for float64).
    """
    if U1.dim() != 2:
        raise ValueError(f"U1 must be 2-D [K, r], got {U1.dim()}-D")
    if U1.shape != U2.shape:
        raise ValueError(f"shape mismatch: {tuple(U1.shape)} vs {tuple(U2.shape)}")
    if U1.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"U1 dtype must be float32/float64, got {U1.dtype}")

    tol = 1e-6 if U1.dtype == torch.float64 else 1e-4
    _validate_orthonormal(U1, tol)
    _validate_orthonormal(U2, tol)

    # SVD in float64 for stability (TZ §3.4); result cast back to U1.dtype.
    M = (U2.T @ U1).double()
    P, s, Qt = torch.linalg.svd(M, full_matrices=False)
    R64 = Qt.T @ P.T
    R = R64.to(U1.dtype)

    r = U1.shape[1]
    residual = (2.0 * r - 2.0 * s.sum()).clamp(min=0.0).item()
    return R, residual


def holonomy_accumulate(
    U_list: Sequence[torch.Tensor],
    closed_loop: bool = True,
    return_total: bool = False,
) -> tuple[float, list[float]] | tuple[float, list[float], torch.Tensor]:
    """Accumulate Procrustes rotation along a (closed) cycle (TZ §3.3).

    Args:
        U_list: list of [K, r] orthonormal, len ≥ 2.
        closed_loop: True — include the final R_{N-1,0} step (a real
            holonomy); False — open path: R_total = R_{0,1} ... R_{N-2,N-1}
            (measures accumulated parallel transport, not holonomy in the
            strict sense).
        return_total: False (default, backward-compatible) — return
            (kappa, per_step_residuals). True — additionally return
            R_total [r, r] float64 (the accumulated product, already
            computed internally) for spectral analysis (see
            :func:`holonomy_spectrum`), i.e. (kappa, per_step, R_total).

    Returns:
        (kappa, per_step_residuals[, R_total]): kappa = ‖R_total − I_r‖_F
        as a Python float; per_step_residuals — the residual_sq of each
        step in cycle order (len = N−1 open, N closed). Diagnostic: large
        kappa with all small per-step residuals → composition problem;
        small kappa with one large per-step → one bad step. R_total is
        the float64 [r, r] product (only when return_total=True).

    Semantics: R_{i,i+1} satisfies U_{i+1} ≈ U_i @ R_{i,i+1} (left
    multiplication in cycle order: R_total = R_{0,1} @ R_{1,2} @ ...).
    R_total ≈ I means the cycle is trivial (flat bundle).

    len == 2 + closed_loop: the pair is traversed forward and backward;
    the inverse Procrustes is the exact inverse, so κ ≈ 0 always.

    dtype: R_total accumulated in float64; kappa returned as float.

    Gauge invariant: U_i → U_i @ Q_i (orthogonal Q_i) leaves κ unchanged.

    Group-orbit property: if the path is a pure orbit U_i = U₀ @ g_i
    (e.g. U @ Q^i, or U @ exp(θᵢA)), the CLOSING step is the exact
    inverse of the accumulated product, so with closed_loop=True
    **κ = 0 always**. Non-zero κ requires the subspace path to fail to
    return to its start (genuine curvature in Grassmannian space) —
    the open-path κ for such an orbit is ‖g_end − I‖ (e.g. ‖Q^{N−1}−I‖
    or ‖exp(2πA)−I‖). This is the correct semantics: a loop in
    parameter space that the subspace bundle maps to a closed loop in
    subspace space has trivial holonomy.

    Errors:
        ValueError: len(U_list) < 2.
    """
    n = len(U_list)
    if n < 2:
        raise ValueError(f"need ≥ 2 subspaces, got {n}")

    r = U_list[0].shape[1]
    R_total = torch.eye(r, dtype=torch.float64)
    per_step: list[float] = []

    steps = range(n) if closed_loop else range(n - 1)
    for i in steps:
        a, b = U_list[i], U_list[(i + 1) % n]
        R_step, res = procrustes_align(a, b)
        R_total = R_total @ R_step.double()
        per_step.append(res)

    kappa = torch.norm(R_total - torch.eye(r, dtype=torch.float64), p="fro").item()
    if return_total:
        return kappa, per_step, R_total
    return kappa, per_step


def holonomy_spectrum(
    U_list: Sequence[torch.Tensor],
    closed_loop: bool = True,
) -> tuple[np.ndarray, torch.Tensor]:
    """Spectral angles of R_total; convenience wrapper over
    ``holonomy_accumulate(..., return_total=True)``.

    NOT a substitute for the scalar κ (``holonomy_accumulate()[0]``):
    κ collapses R_total to one number; this returns the full angle set
    used in paper [M-2] / §6.

    Args:
        U_list: list of [K, r] orthonormal, len ≥ 2 (see
            :func:`holonomy_accumulate`).
        closed_loop: see :func:`holonomy_accumulate`.

    Returns:
        (theta, R_total): theta — np.ndarray [r//2] of rotation angles
        in (0, π], sorted ascending (one per conjugate pair
        {e^{iθ}, e^{−iθ}} of R_total; +1-axes dropped; odd r leaves a
        trivial +1 axis); R_total — [r, r] float64.

    Uses ``torch.linalg.eigvals`` (complex) — ``spectrum_from_gramian`` /
    ``eigvalsh`` CANNOT be used: R_total is orthogonal, not symmetric.

    Warnings:
        RuntimeWarning: odd r (trivial +1 axis); eigenvalues off the
        unit circle (|λ| < 1 − 1e-6); duplicate angles (degenerate
        spectrum, basis ambiguity).

    Errors:
        RuntimeError: R_total not orthogonal (‖R^T R − I‖_F > 1e-4).
    """
    _, _, R_total = holonomy_accumulate(U_list, closed_loop=closed_loop,
                                        return_total=True)
    r = R_total.shape[0]
    orth_err = torch.norm(R_total.T @ R_total - torch.eye(r, dtype=torch.float64),
                          p="fro").item()
    if orth_err > 1e-4:
        raise RuntimeError(
            f"holonomy_spectrum: R_total not orthogonal: "
            f"‖R^T R − I‖_F = {orth_err:.3e} (tol 1e-4)"
        )

    eigvals = torch.linalg.eigvals(R_total)
    if torch.any(eigvals.abs() < 1 - 1e-6):
        off = float((1.0 - eigvals.abs()).max().item())
        warnings.warn(
            f"holonomy_spectrum: eigenvalue off the unit circle by {off:.2e} "
            "(numerical instability)",
            RuntimeWarning,
        )

    # One angle per conjugate pair {e^{iθ}, e^{-iθ}}: for θ < π the pair
    # contributes exactly ONE positive angle (+θ; −θ is filtered out), so
    # mult = counts. At θ = π both eigenvalues are −1 and both angles are
    # +π — counts is doubled there, so halve only the π bin (−I₄ → θ =
    # [π, π], not [π, π, π, π]).
    angles = torch.angle(eigvals)  # (-π, π]
    positive = angles[angles > 1e-6]
    rounded = positive.round(decimals=8)
    uniques, counts = torch.unique(rounded, return_counts=True)
    pi_mask = uniques > math.pi - 1e-6
    mult = torch.where(pi_mask, counts // 2, counts)
    # mult 0: a π-pair split across the ±π boundary (one side negative) —
    # numerical noise; treat as one pair
    if (mult == 0).any():
        warnings.warn(
            "holonomy_spectrum: a conjugate pair split across the ±π "
            "boundary (numerical noise) — merged into one",
            RuntimeWarning,
        )
        mult = mult.clamp(min=1)
    # mult > 1: the same angle appears in several pairs — the spectrum is
    # degenerate and the basis inside that block is ambiguous, but the
    # angle set (with multiplicities) is well-defined and REQUIRED for
    # tr(R) = 2 Σ_j cos θ_j — keep the multiplicity
    if (mult > 1).any():
        warnings.warn(
            "holonomy_spectrum: degenerate spectrum (repeated angles) — "
            "basis is ambiguous inside the degenerate block; multiplicities "
            "kept",
            RuntimeWarning,
        )
    # repeat_interleave is unimplemented for float64 on CPU — expand in Python
    expanded_vals = [float(u) for u, m in zip(uniques.tolist(), mult.tolist())
                     for _ in range(int(m))]
    expanded = torch.tensor(sorted(expanded_vals), dtype=torch.float64)
    expected = r // 2
    if r % 2 == 1:
        warnings.warn(
            f"holonomy_spectrum: odd r = {r}, trivial +1 axis present",
            RuntimeWarning,
        )
    elif expanded.numel() != expected:
        if positive.numel() == 0:
            # R_total ≈ I: all angles are +1 — report r//2 zeros
            expanded = torch.zeros(expected, dtype=torch.float64)
        else:
            warnings.warn(
                f"holonomy_spectrum: expected {expected} angle pairs, "
                f"got {expanded.numel()}",
                RuntimeWarning,
            )
    return expanded.numpy().astype(np.float64), R_total


def geodesic_curvature(
    u1: torch.Tensor, u2: torch.Tensor, u3: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Discrete geodesic curvature κ_g ∈ R≥0 at u2, from three unit vectors.

    Local Abelian invariant of the û_t curve on S^{D-1} (lemma [M-3],
    corrected formulation) — for comparison with κ (holonomy). NOT the
    triangle-inequality defect.

    Args:
        u1, u2, u3: [D] unit vectors (‖·‖ = 1, checked).
        eps: collinearity tolerance for the tangent norms.

    Returns:
        turn / arc, where turn = π − arccos(<t1, t3>/(‖t1‖‖t3‖)) is the
        exterior angle between the tangents at u2 (toward u1 / u3) and
        arc = (‖u1 − u2‖ + ‖u3 − u2‖) / 2 (chord lengths ≈ geodesic
        lengths for small steps). Collinear (degenerate) triple → 0.0.

    Errors:
        ValueError: non-unit input (index and value reported), D < 2.
        RuntimeError: all three points coincide (arc < eps).
    """
    for i, u in enumerate((u1, u2, u3)):
        u = u.reshape(-1)
        n = torch.norm(u).item()
        if abs(n - 1.0) > 1e-4:
            raise ValueError(
                f"geodesic_curvature: u{i + 1} not unit: ‖u‖ = {n:.6f}"
            )
        if u.shape[0] < 2:
            raise ValueError(
                f"geodesic_curvature: D < 2 (got D = {u.shape[0]}) — "
                "cannot build tangents"
            )
    u1, u2, u3 = u1.reshape(-1), u2.reshape(-1), u3.reshape(-1)

    arc = (torch.norm(u1 - u2).item() + torch.norm(u3 - u2).item()) / 2
    if arc < eps:
        raise RuntimeError(
            "geodesic_curvature: all three points coincide (arc < eps) — "
            "not informative"
        )

    t1 = u1 - torch.dot(u1, u2) * u2   # tangent at u2, toward u1
    t3 = u3 - torch.dot(u3, u2) * u2   # tangent at u2, toward u3
    n1, n3 = float(torch.norm(t1)), float(torch.norm(t3))
    if n1 < eps or n3 < eps:
        return 0.0                      # u1 or u3 collinear with u2
    cos_turn = float(torch.clip(torch.dot(t1, t3) / (n1 * n3), -1.0, 1.0))
    turn = math.pi - math.acos(cos_turn)
    return turn / arc


def eigenframe_basis(
    M_K: torch.Tensor,   # [K, D]
    uhat: torch.Tensor,  # [D] unit
) -> torch.Tensor:
    """Canonical basis B [D, D-1] orthonormal, B^T uhat = 0, ordered by
    decreasing eigenvalue of P_⊥ M_K^T M_K P_⊥.

    This is a CONVENTION per [M-5]; the basis is data-determined up to
    permutation and sign, but the decomposition of H_{012} using it is
    fixed once chosen.

    NOT a substitute for spectral.top_r_svd: the operator is the symmetric
    P_⊥ M_K^T M_K P_⊥ (D×D, in uhat^⊥), NOT J (general K×D). top_r_svd
    returns right singular vectors of J; this returns eigenvectors of the
    gauge-projected M_K Gramian. Consumers needing right singular vectors
    of J must call top_r_svd directly.

    Warnings:
        RuntimeWarning: the trivial û-axis is not isolated (|λ_min| >
        1e-8, M_K^T M_K non-degenerate on û) — the smallest-λ vector is
        dropped anyway; degenerate eigenvalues (basis ambiguity inside
        a degenerate block).

    Errors:
        ValueError: uhat not unit, M_K.shape[1] != D, D < 2.
    """
    if M_K.dim() != 2:
        raise ValueError(f"eigenframe_basis: M_K must be 2-D, got {M_K.dim()}-D")
    uhat = uhat.reshape(-1)
    K, D = M_K.shape
    if uhat.shape[0] != D:
        raise ValueError(
            f"eigenframe_basis: M_K.shape[1] = {D} != uhat dim = {uhat.shape[0]}"
        )
    if D < 2:
        raise ValueError(f"eigenframe_basis: D < 2 (got D = {D})")
    n = torch.norm(uhat).item()
    if abs(n - 1.0) > 1e-4:
        raise ValueError(f"eigenframe_basis: uhat not unit: ‖uhat‖ = {n:.6f}")

    A = M_K.T @ M_K                              # [D, D] symmetric PSD
    Au = A @ uhat
    A_proj = A - torch.outer(uhat, Au) - torch.outer(Au, uhat) \
        + torch.dot(uhat, Au) * torch.outer(uhat, uhat)   # P_⊥ A P_⊥

    eigvals, eigvecs = torch.linalg.eigh(A_proj)   # ascending
    mask = eigvals.abs() > 1e-8
    if mask.sum() == D:
        # no ~0 eigenvalue: trivial û-axis not isolated — drop the smallest
        warnings.warn(
            "eigenframe_basis: trivial û-axis not isolated "
            "(|λ_min| > 1e-8, M_K^T M_K non-degenerate on û) — "
            "dropping the smallest-λ vector",
            RuntimeWarning,
        )
        eigvals_pos, eigvecs_pos = eigvals[1:], eigvecs[:, 1:]
    else:
        eigvals_pos, eigvecs_pos = eigvals[mask], eigvecs[:, mask]
    # degenerate spectrum check (consecutive gaps below 1e-8 of the scale)
    if eigvals_pos.numel() > 1:
        scale = eigvals_pos.max().abs()
        if scale > 0 and ((eigvals_pos[1:] - eigvals_pos[:-1]).abs() < 1e-8 * scale).any():
            warnings.warn(
                "eigenframe_basis: degenerate eigenvalues — basis is "
                "ambiguous inside the degenerate block",
                RuntimeWarning,
            )
    order = torch.argsort(eigvals_pos, descending=True)
    if eigvals_pos.numel() == 0:
        # fully degenerate (e.g. M_K = 0): no non-trivial axis — return [D, 0]
        warnings.warn(
            "eigenframe_basis: spectrum fully degenerate — returning an "
            "empty basis [D, 0]",
            RuntimeWarning,
        )
        return eigvecs[:, :0].contiguous()
    B = eigvecs_pos[:, order]
    B = B / B.norm(dim=0, keepdim=True)
    return B.contiguous()


def holonomy_along_path(
    path_fn: Callable[[float], tuple[torch.Tensor, torch.Tensor]],
    thetas: Sequence[float],
    closed_loop: bool = True,
) -> dict:
    """Holonomy of a family of subspaces along a 1-D parameter (TZ §3.3).

    Args:
        path_fn: θ → (U_r, V_r), U_r: [K, r], V_r: [r, D_S]. ALL model
            logic (weight setup, forward, SVD, top-r selection) lives in
            path_fn — this function does not know about models and does
            not cache.
        thetas: sequence of θ, len ≥ 2. With closed_loop=True the last
            θ must NOT duplicate the first (holonomy_accumulate adds the
            closing step R_{N-1,0} itself; a duplicate would just add a
            trivial Procrustes I and double-count a measurement).
        closed_loop: see holonomy_accumulate.

    Returns:
        dict with keys: kappa_U, kappa_V, per_step_U, per_step_V, thetas
        (thetas as a list for JSON-friendliness).

    Use case: parameter-space holonomy, cycle W₀→W₁₀₀→W₀ —
    path_fn(θ) = compute top-r U, V at interpolated lm_head weights.
    Note the group-orbit property (holonomy_accumulate): if the
    subspaces form a pure orbit U(θ) = U₀·g(θ), a closed loop has
    κ = 0; non-zero κ means the subspaces do NOT return to their start
    at the end of the parameter cycle — that is the signal the
    experiment is looking for.

    Fully deterministic given path_fn and thetas.

    Errors:
        ValueError: len(thetas) < 2.
        TypeError: path_fn does not return a (U, V) pair of tensors.
    """
    thetas = list(thetas)
    if len(thetas) < 2:
        raise ValueError(f"need ≥ 2 thetas, got {len(thetas)}")

    U_list: list[torch.Tensor] = []
    V_list: list[torch.Tensor] = []
    for th in thetas:
        out = path_fn(th)
        if not (isinstance(out, tuple) and len(out) == 2
                and isinstance(out[0], torch.Tensor) and isinstance(out[1], torch.Tensor)):
            raise TypeError(
                f"path_fn must return (U_r [K, r], V_r [r, D_S]) tensors, "
                f"got {type(out).__name__}"
            )
        U_list.append(out[0])
        V_list.append(out[1])

    kappa_U, per_U = holonomy_accumulate(U_list, closed_loop=closed_loop)
    kappa_V, per_V = holonomy_accumulate(V_list, closed_loop=closed_loop)
    return {
        "kappa_U": kappa_U,
        "kappa_V": kappa_V,
        "per_step_U": per_U,
        "per_step_V": per_V,
        "thetas": thetas,
    }


def holonomy_along_2d_loop(
    base: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    epsilon: float,
    n_theta: int,
    measure_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    """Holonomy around a small 2-D loop (ε·(cos θ·b1 + sin θ·b2)) (TZ §3.3).

    The loop h(θ) = base + ε(cos θ·b1 + sin θ·b2), θ = 0, 2π/n, ...,
    2π(n−1)/n (no 2π endpoint — holonomy_accumulate closes it), probes
    the local curvature of the subspace bundle around `base`.

    Args:
        base: [D] / [1, D] / [1, 1, D] base point (normalized to [D]).
        b1, b2: [D] orthonormal directions (‖·‖ = 1, |b1·b2| < 1e-4 —
            checked; a non-orthogonal pair makes κ uninterpretable).
            For 'sens': top-2 right singular vectors of J(base); for
            'canon': top-2 PCs of h_batch (wrappers in exp_T2.py).
        epsilon: loop radius, must be > 0 and ≲ 0.2·‖base‖ (a larger
            loop wanders far from the operating point and measure_fn may
            return degenerate subspaces).
        n_theta: loop discretization, ≥ 4 (a 3-point 'loop' is a
            triangle — crude; 16 recommended).
        measure_fn: h → (U_r, V_r). All model logic lives there.

    Returns:
        dict: kappa_U, kappa_V, per_step_U, per_step_V, h_perturbed
        (list of [D] loop points, for diagnostics).

    Errors:
        ValueError: epsilon ≤ 0; n_theta < 4; b1/b2 not orthonormal;
            base not squeezable to [D].
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if n_theta < 4:
        raise ValueError(f"n_theta must be ≥ 4, got {n_theta}")

    base = base.reshape(-1)
    b1 = b1.reshape(-1)
    b2 = b2.reshape(-1)
    if b1.shape != b2.shape or b1.shape != base.shape:
        raise ValueError(
            f"shape mismatch: base {tuple(base.shape)}, b1 {tuple(b1.shape)}, "
            f"b2 {tuple(b2.shape)}"
        )
    _check_loop_basis(b1, b2)

    U_list: list[torch.Tensor] = []
    V_list: list[torch.Tensor] = []
    h_perturbed: list[torch.Tensor] = []
    for i in range(n_theta):
        th = 2.0 * math.pi * i / n_theta
        h = base + epsilon * (math.cos(th) * b1 + math.sin(th) * b2)
        out = measure_fn(h)
        if not (isinstance(out, tuple) and len(out) == 2
                and isinstance(out[0], torch.Tensor) and isinstance(out[1], torch.Tensor)):
            raise TypeError(
                f"measure_fn must return (U_r, V_r) tensors, got {type(out).__name__}"
            )
        U_list.append(out[0])
        V_list.append(out[1])
        h_perturbed.append(h)

    kappa_U, per_U = holonomy_accumulate(U_list, closed_loop=True)
    kappa_V, per_V = holonomy_accumulate(V_list, closed_loop=True)
    return {
        "kappa_U": kappa_U,
        "kappa_V": kappa_V,
        "per_step_U": per_U,
        "per_step_V": per_V,
        "h_perturbed": h_perturbed,
    }
