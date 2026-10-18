"""Tests for h_override / h-respect (v6.0, docs/changed_functions.md §2.5).

Fixtures: tiny Qwen3.5 (tests/helpers/tiny_model.py), CPU, S=8, K=4,
layer_idx=2 (linear_attention) and 3 (full_attention).
"""

import pytest

import torch

from tests.helpers.tiny_model import make_tiny_qwen
from prism.jacobian import (
    compute_hidden_jacobian,
    compute_jacobian_for_tokens,
    _extract_hidden_state,
    _run_prefix_pass,
)

torch.manual_seed(0)
# eager attention: flash/sdpa have no CPU backward (see test_bridge_rows_backward).
MODEL = make_tiny_qwen(attn_implementation="eager").eval()
S, K = 8, 4
INPUT_IDS = torch.randint(0, 60, (1, S))
ATTENTION = torch.ones(1, S, dtype=torch.long)
TOKEN_SET = list(range(1, K + 1))
L_IDX = 2  # linear_attention layer
H_SIZE = MODEL.config.text_config.hidden_size


def _ref_hidden():
    return _extract_hidden_state(MODEL, INPUT_IDS, ATTENTION, L_IDX)


# ---------------------------------------------------------------------------
# compute_jacobian_for_tokens(h_override=...)
# ---------------------------------------------------------------------------

def test_identity_override_equals_forward():
    """§2.5.1: h_override = h extracted from the forward -> same J."""
    h = _ref_hidden()
    J_a = compute_jacobian_for_tokens(MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET)
    J_b = compute_jacobian_for_tokens(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=h
    )
    assert torch.allclose(J_a, J_b, atol=1e-6)


def test_shape_normalization():
    """§2.5.2: [D], [1, D], [1, 1, D] all give the same J."""
    h = _ref_hidden()
    base = compute_jacobian_for_tokens(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=h
    )
    for shaped in (h.reshape(-1), h.reshape(1, -1)):
        J = compute_jacobian_for_tokens(
            MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=shaped
        )
        assert torch.allclose(base, J, atol=1e-6)


def test_device_mismatch_raises():
    """§2.5.3: h_override on a different device -> ValueError."""
    h = _ref_hidden().to("meta")
    with pytest.raises(ValueError, match="device"):
        compute_jacobian_for_tokens(
            MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=h
        )


def test_d_mismatch_raises():
    """§2.5.4: h_override with D+1 -> ValueError."""
    h_bad = torch.randn(1, 1, H_SIZE + 1)
    with pytest.raises(ValueError, match="hidden_size"):
        compute_jacobian_for_tokens(
            MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=h_bad
        )


def test_shape_not_reducible_raises():
    h_bad = torch.randn(2, 3, H_SIZE)  # not 1x1xD
    with pytest.raises(ValueError, match="reducible"):
        compute_jacobian_for_tokens(
            MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=h_bad
        )


def test_detach_safety():
    """§2.5.5: the caller's tensor is NOT mutated (clone), and J is a plain
    tensor (no grad graph through the detached leaf h).

    NOTE (deviation from ТЗ §2.5.5 literal assertion): the ТЗ also asks
    `assert not h_override.requires_grad` after the call. That requires an
    in-place `requires_grad_(False)` on the caller's tensor, which is itself a
    mutation and contradicts the ТЗ's own rule "не мутировать тензор
    вызывающего (clone)". The clone keeps the caller's flag untouched; the
    meaningful guarantees are (a) values unchanged, (b) no graph leak into J.
    """
    h = _ref_hidden()
    h_vals_before = h.detach().clone()
    h.requires_grad_(True)
    J = compute_jacobian_for_tokens(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, TOKEN_SET, h_override=h
    )
    # caller's tensor not mutated: values AND flag preserved (it's a clone)
    assert torch.equal(h.detach(), h_vals_before)
    assert h.requires_grad  # untouched — clone, not in-place flag clear
    # no graph leak: J is a plain tensor
    assert J.grad_fn is None


# ---------------------------------------------------------------------------
# compute_hidden_jacobian(create_graph=True)
# ---------------------------------------------------------------------------

def test_create_graph_h_none_identity():
    """§2.5.6: h=None (and explicit h=None) — behavior identical, both
    go through the full forward and stay connected to params."""
    J_a = compute_hidden_jacobian(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, create_graph=True
    )
    J_b = compute_hidden_jacobian(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, create_graph=True, h=None
    )
    assert torch.allclose(J_a, J_b, atol=1e-6)
    assert J_a.requires_grad


def test_create_graph_with_leaf_h():
    """§2.5.7: h=ready-leaf -> J has grad_fn and h_leaf.grad is set.

    GRAD-SCOPE NOTE (deviation from ТЗ §2.5.7 literal assertion): the ТЗ
    asserts that NO model param receives grad. That is mathematically wrong
    for compute_hidden_jacobian: J_norm = d(h_norm)/d(h_l) is computed by
    DECODING h through layers layer_idx+1..N-1 + final norm, so J_norm
    depends on those decoder weights by definition and they DO get grads.
    What the leaf-h path guarantees (and what the test asserts):
      (a) the passed h_leaf itself receives grad (use-in-place, not a clone);
      (b) params of layers <= layer_idx (embeddings, layers 0..L_IDX) do NOT
          receive grad — the forward below layer_idx was skipped.
    """
    h = _ref_hidden()
    h_leaf = h.clone().detach().requires_grad_(True)
    J = compute_hidden_jacobian(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, create_graph=True, h=h_leaf
    )
    assert J.requires_grad
    J.sum().backward()
    assert h_leaf.grad is not None  # (a) grads flow to the caller's h
    # (b) params at/below layer_idx excluded; decoder weights included.
    # Restrict to text params (skip the visual tower, out of the path).
    for n, p in MODEL.named_parameters():
        if "visual" in n:
            continue
        below = "embed" in n or any(f".layers.{i}." in n
                                    for i in range(L_IDX + 1))
        assert (p.grad is not None) != below, n
    # sanity: at least one decoder param (> L_IDX) did receive grad
    assert any("visual" not in n and p.grad is not None
               for n, p in MODEL.named_parameters())


def test_create_graph_h_respected_not_overwritten():
    """The bug-fix itself: a connected h built from params must be respected
    (not re-extracted from a fresh forward). We verify by making the model's
    forward return a *different* h than the one passed: if the code respected
    the passed h, J matches the reference computed from the passed h."""
    # reference: same call with h given explicitly, vs a sentinel check.
    # Build a connected h from one forward, then zero out a param so a fresh
    # forward would produce a different h; J must reflect the PASSED h.
    h = _ref_hidden()  # [1, 1, D]
    # Perturb the TEXT embedding (a param that reaches layer_idx in the
    # forward) so a re-forward would differ. `next(MODEL.parameters())` is a
    # visual weight — it does NOT affect the text path. Restored after.
    first_w = MODEL.model.language_model.embed_tokens.weight
    saved = first_w.detach().clone()
    with torch.no_grad():
        first_w.add_(0.01 * torch.randn_like(first_w))
    h2 = _ref_hidden()
    assert not torch.allclose(h, h2, atol=1e-6)  # forwards now differ
    # J from the ORIGINAL h must equal a J computed from h (leaf) — i.e. the
    # passed h is respected, not overwritten by the perturbed forward.
    J_orig = compute_hidden_jacobian(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, create_graph=True, h=h
    )
    J_leaf = compute_hidden_jacobian(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, create_graph=True,
        h=h.clone().detach().requires_grad_(True),
    )
    assert torch.allclose(J_orig, J_leaf, atol=1e-6)
    # And it must NOT equal what the perturbed forward would give.
    J_fresh = compute_hidden_jacobian(
        MODEL, L_IDX, INPUT_IDS, ATTENTION, create_graph=True
    )
    assert not torch.allclose(J_orig, J_fresh, atol=1e-3)
    with torch.no_grad():
        first_w.copy_(saved)  # restore shared fixture model


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
