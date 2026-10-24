"""Module: structural primitives for the L23 gauge story (new_fun_core.md §3).

Thin tensor helpers formalizing existing quantities (M_K, û_t^⊥,
w = α û + β v_⊥) — no model, no forward passes, no caching (W_lm changes
under ``apply_edit``; callers own any cache invalidation).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

__all__ = [
    "build_M_K",
    "decompose_w_vs_uhat",
    "v_pred_from_JT",
]


def build_M_K(
    W_lm: torch.Tensor,     # [V, D]
    token_set: Sequence[int],
    g: torch.Tensor,        # [D] — RMSNorm weight (1 + w, NOT raw w)
) -> torch.Tensor:         # [K, D]
    """M_K = W_lm[S] · diag(g) ∈ R^{K×D} (lemma [M-1], fundamental for L23).

    Args:
        W_lm: [V, D] lm_head weight rows.
        token_set: token row indices (duplicates allowed — the caller is
            responsible for uniqueness).
        g: [D] RMSNorm weight, i.e. (1 + w), broadcast over the K rows.

    Returns:
        M_K [K, D].

    NOT cached inside: W_lm mutates under ``apply_edit`` — a ``data_ptr``-keyed
    cache would be unsafe. If an experiment needs caching, the caller owns
    invalidation.

    Errors:
        ValueError: empty token_set, index out of [0, V) (first offender
            reported), W_lm.shape[1] != g.shape[0], NaN/Inf in g.
    """
    if len(token_set) == 0:
        raise ValueError("build_M_K: token_set is empty")
    if W_lm.dim() != 2:
        raise ValueError(f"build_M_K: W_lm must be 2-D, got {W_lm.dim()}-D")
    if g.dim() != 1 or W_lm.shape[1] != g.shape[0]:
        raise ValueError(
            f"build_M_K: W_lm.shape[1] = {W_lm.shape[1]} != g dim = {g.shape[0] if g.dim() else 0}"
        )
    V = W_lm.shape[0]
    for i, idx in enumerate(token_set):
        if idx < 0 or idx >= V:
            raise ValueError(
                f"build_M_K: token_set[{i}] = {idx} out of [0, V) = [0, {V})"
            )
    if not torch.isfinite(g).all():
        bad = int(torch.nonzero(~torch.isfinite(g), as_tuple=False).flatten()[0])
        raise ValueError(f"build_M_K: non-finite g at index {bad}")

    ts = torch.tensor(token_set, device=W_lm.device, dtype=torch.long)
    # Cast g to W_lm's dtype: an fp64 g would silently promote the result
    # to fp64 (torch type promotion), surprising the caller (docs/E38_fixes.md §5).
    return W_lm[ts] * g.to(W_lm.dtype).unsqueeze(0)


def decompose_w_vs_uhat(
    w: torch.Tensor,     # [D]
    uhat: torch.Tensor,  # [D] unit
) -> tuple[float, float, torch.Tensor]:
    """Rank-1 split w = α·uhat + β·v_perp (paper [M-1], step 1).

    Returns (α: float, β: float ≥ 0, v_perp: [D] unit if β > 0 else zeros).

    NOT a generic projection (cf. topology.project_to_subspace): this
    returns a rank-1 coefficient + residual norm + unit residual direction,
    working on a single vector [D]. project_to_subspace returns
    h @ basis.T for [N, D] inputs and does not compute a residual.

    Errors:
        ValueError: uhat not unit (‖·‖ reported), w.shape != uhat.shape,
            NaN/Inf in w.
    """
    w = w.reshape(-1)
    uhat = uhat.reshape(-1)
    if w.shape != uhat.shape:
        raise ValueError(
            f"decompose_w_vs_uhat: shape mismatch {tuple(w.shape)} vs {tuple(uhat.shape)}"
        )
    n = torch.norm(uhat).item()
    if abs(n - 1.0) > 1e-4:
        raise ValueError(f"decompose_w_vs_uhat: uhat not unit: ‖uhat‖ = {n:.6f}")
    if not torch.isfinite(w).all():
        bad = int(torch.nonzero(~torch.isfinite(w), as_tuple=False).flatten()[0])
        raise ValueError(f"decompose_w_vs_uhat: non-finite w at index {bad}")

    alpha = float(torch.dot(w, uhat))
    r = w - alpha * uhat
    beta = float(r.norm())
    if beta > 1e-12:
        v_perp = r / beta
    else:
        v_perp = torch.zeros_like(w)
    return alpha, beta, v_perp


def v_pred_from_JT(
    J: torch.Tensor,                  # [K, D]
    uniform: torch.Tensor | None = None,  # [K], default 1/√K
    eps: float = 1e-10,
) -> torch.Tensor:                    # [D] unit
    """Normalized transpose projection v_pred = J^T·u / ‖J^T·u‖, u = 1/√K
    (paper [M-1], step 2). No J^+ — a pseudo-inverse would emphasize the
    tail, not the gauge.

    NOT a generic projection: transpose-side of J, returns a UNIT vector,
    raises on ‖J^T u‖ ≈ 0 (lemma [M-1] degeneracy). Cf.
    topology.project_to_subspace (h @ basis.T, [N, D] → [N, r]).

    A non-unit ``uniform`` is normalized silently (documented behavior).

    Errors:
        ValueError: J.dim() != 2, K == 0, uniform.shape != (K,),
            ‖uniform‖ == 0.
        RuntimeError: ‖J^T u‖ < eps ([M-1] degeneracy, β ≈ 0).
    """
    if J.dim() != 2:
        raise ValueError(f"v_pred_from_JT: J must be 2-D, got {J.dim()}-D")
    K, D = J.shape
    if K == 0:
        raise ValueError("v_pred_from_JT: K == 0")
    if uniform is None:
        uniform = torch.full((K,), 1.0 / math.sqrt(K), device=J.device, dtype=J.dtype)
    else:
        if uniform.shape != (K,):
            raise ValueError(
                f"v_pred_from_JT: uniform shape {tuple(uniform.shape)} != (K,) = ({K},)"
            )
        n = uniform.norm()
        if not (n > 0) or float(n) < 1e-30:
            raise ValueError("v_pred_from_JT: uniform has (near-)zero norm")
        uniform = uniform / n
    x = J.T @ uniform
    nx = float(x.norm())
    if nx < eps:
        raise RuntimeError(
            f"v_pred_from_JT: ‖J^T u‖ = {nx:.2e} < {eps:.0e}; "
            "this is the [M-1] degeneracy (β≈0); see lemma"
        )
    return x / nx
