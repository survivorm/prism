"""Analytical last-layer Jacobian (prism/jacobian.py `_last_layer_jacobian`).

RED evidence: the OLD formula (LayerNorm-style, evaluated at the norm
OUTPUT) has ~0.87 relative error vs autograd on Qwen3.5's final norm,
which is Qwen3_5RMSNorm (no mean subtraction). The new formula
(RMSNorm family, evaluated at the norm INPUT) matches autograd.
"""

import torch

from prism.jacobian import _last_layer_jacobian


class RMSNormLikeQwen35(torch.nn.Module):
    """Same forward as transformers.models.qwen3_5 Qwen3_5RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        ) * (1.0 + self.weight.float())
        return out.type_as(x)


def _autograd_j(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Reference J = dy/dx via per-row autograd (x: [1, D])."""
    D = x.shape[-1]
    xg = x.detach().clone().requires_grad_(True)
    y = norm(xg).flatten()
    rows = [
        torch.autograd.grad(y[i], xg, retain_graph=(i < D - 1))[0][0] for i in range(D)
    ]
    return torch.stack(rows)


def _rel_err(J: torch.Tensor, ref: torch.Tensor) -> float:
    return (J - ref).norm().item() / ref.norm().item()


def _old_formula_at_norm_output(h_norm_row: torch.Tensor) -> torch.Tensor:
    """The pre-fix formula: LayerNorm stats on the NORM OUTPUT x."""
    x = h_norm_row
    D = x.shape[-1]
    mean = x.mean(dim=-1, keepdim=True)
    std = (x.var(dim=-1, keepdim=True) + 1e-5).sqrt()
    x_norm = (x - mean) / std
    J = (
        torch.eye(D) - torch.ones(D, D) / D - x_norm[:, None] @ x_norm[None, :] / D
    ) / std.squeeze(0)
    return J


def test_old_formula_has_large_rel_error():
    """RED: old formula (LayerNorm stats on norm output) is wrong for
    Qwen3_5RMSNorm — relative error must be far from zero."""
    torch.manual_seed(0)
    D = 64
    norm = RMSNormLikeQwen35(D)
    norm.weight.data = 0.1 * torch.randn(D)
    h = 1.0 + 0.5 * torch.randn(1, D)
    ref = _autograd_j(norm, h)
    old = _old_formula_at_norm_output(norm(h).squeeze(0))
    err = _rel_err(old, ref)
    print(f"old-formula rel err = {err:.4f}")
    assert err > 0.1, "old formula unexpectedly accurate — bug was not real?"


def test_rmsnorm_weighted_matches_autograd():
    """GREEN: new RMSNorm formula matches autograd to float precision."""
    torch.manual_seed(0)
    D = 64
    norm = RMSNormLikeQwen35(D)
    norm.weight.data = 0.3 * torch.randn(D)
    h = 1.0 + 0.5 * torch.randn(1, D)
    ref = _autograd_j(norm, h)
    new = _last_layer_jacobian(h, norm)
    err = _rel_err(new, ref)
    print(f"new-rmsnorm rel err = {err:.2e}")
    assert err < 1e-5, f"RMSNorm analytical J wrong: rel err {err}"


def test_rmsnorm_unweighted_matches_autograd():
    """GREEN: unweighted RMSNorm (weight=None) — the fallback branch."""

    class PlainRMS(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.float() * torch.rsqrt(
                x.float().pow(2).mean(-1, keepdim=True) + 1e-6
            )

    torch.manual_seed(1)
    D = 32
    norm = PlainRMS()
    h = 1.0 + 0.5 * torch.randn(1, D)
    ref = _autograd_j(norm, h)
    new = _last_layer_jacobian(h, norm)
    err = _rel_err(new, ref)
    print(f"new-rmsnorm-unweighted rel err = {err:.2e}")
    assert err < 1e-5, f"unweighted RMSNorm analytical J wrong: rel err {err}"


def test_layernorm_matches_autograd():
    """GREEN: nn.LayerNorm branch (kept for other model families)."""
    torch.manual_seed(2)
    D = 32
    norm = torch.nn.LayerNorm(D, eps=1e-6)
    h = 1.0 + 0.5 * torch.randn(1, D)
    ref = _autograd_j(norm, h)
    new = _last_layer_jacobian(h, norm)
    err = _rel_err(new, ref)
    print(f"new-layernorm rel err = {err:.2e}")
    assert err < 1e-5, f"LayerNorm analytical J wrong: rel err {err}"
