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
from typing import Callable, Sequence

import torch

__all__ = [
    "procrustes_align",
    "holonomy_accumulate",
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
    if abs(n1 - 1.0) > 1e-3 or abs(n2 - 1.0) > 1e-3:
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
) -> tuple[float, list[float]]:
    """Accumulate Procrustes rotation along a (closed) cycle (TZ §3.3).

    Args:
        U_list: list of [K, r] orthonormal, len ≥ 2.
        closed_loop: True — include the final R_{N-1,0} step (a real
            holonomy); False — open path: R_total = R_{0,1} ... R_{N-2,N-1}
            (measures accumulated parallel transport, not holonomy in the
            strict sense).

    Returns:
        (kappa, per_step_residuals): kappa = ‖R_total − I_r‖_F as a
        Python float; per_step_residuals — the residual_sq of each step
        in cycle order (len = N−1 open, N closed). Diagnostic: large
        kappa with all small per-step residuals → composition problem;
        small kappa with one large per-step → one bad step.

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
    return kappa, per_step


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
