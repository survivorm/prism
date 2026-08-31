"""Matrix-free J_S_p (v23 A1): per-row autograd.grad, no dense [D,D].

CONTRACT (matches exp16 stage0 target + exp17 dense bridge): p is the
RE-NORMALIZED softmax over the K-subset (p_s = softmax(full_logits[token_set])),
W_rows = W[token_set] [K, D]. The matrix-free path must equal the dense
path (J_norm = [D,D] -> W_rows@J_norm -> softmax gauge) to rel<1e-3, because
J_i = d(h_norm @ W_rows[i])/dh is the i-th row of W_rows@J_norm and
Jp_i = p_i*(J_i - J_bar) is exactly the softmax gauge form.

The dense path is the v22 OOM (D=1024 retained 2nd-order row-graphs); the
matrix-free path keeps only K=128 (+1 J_bar). Tests are pure-math (no model):
h_norm is a nonlinear, h-connected function (LN), mirroring the production
decoder-norm graph.
"""

import torch

from prism.jacobian import (
    _compute_jacobian_rows_full,
    _compute_jacobian_rows_matrixfree,
    compute_hidden_jacobian,
    compute_hidden_jacobian_rows,
    softmax_jacobian,
)
from tests.helpers.tiny_model import full_attn_layers, make_tiny_qwen


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item() / b.norm().item()


def _synthetic_hnorm(D: int):
    """Nonlinear h_norm = LayerNorm(h) — a grad_fn that DEPENDS on h, so
    create_graph=True builds a 2nd-order graph (a linear h@M has a
    constant Jacobian and no 2nd-order nodes)."""
    torch.manual_seed(0)
    h = torch.randn(1, 1, D, requires_grad=True)
    h_norm = torch.nn.functional.layer_norm(h[:, 0, :], (D,), eps=1e-5)
    return h, h_norm


def test_matrixfree_matches_dense():
    D, V, K = 64, 200, 32
    h, h_norm = _synthetic_hnorm(D)
    torch.manual_seed(1)
    W_full = torch.randn(V, D)
    token_set = list(range(K))
    W_rows = W_full[token_set]  # [K, D]
    # K-subset gauge: re-normalized softmax over the selected rows
    p = torch.softmax(torch.randn(V)[token_set], dim=-1)  # [K]

    # dense reference: [D,D] J_norm -> per-token rows -> softmax gauge
    J_norm = _compute_jacobian_rows_full(h_norm, h, create_graph=True)  # [D,D]
    J_S = W_rows @ J_norm  # [K,D]
    J_bar = (p @ W_rows) @ J_norm  # [D] = sum_k p_k J_k
    J_S_p_ref = p[:, None] * (J_S - J_bar[None, :])

    # matrix-free: per-row, no [D,D]
    J_S_p_mf = _compute_jacobian_rows_matrixfree(
        h_norm, h, W_rows, p, token_set, create_graph=True
    )

    rel = _rel_err(J_S_p_mf, J_S_p_ref)
    assert rel < 1e-3, f"matrix-free vs dense rel={rel:.3e}"


def test_matrixfree_shape_and_dtype():
    D, V, K = 32, 100, 16
    h, h_norm = _synthetic_hnorm(D)
    W_rows = torch.randn(V, D)
    p = torch.softmax(torch.randn(V), dim=-1)
    out = _compute_jacobian_rows_matrixfree(
        h_norm, h, W_rows, p, list(range(K)), create_graph=True
    )
    assert out.shape == (K, D)
    assert out.dtype == h.dtype


def test_matrixfree_graph_reaches_h():
    # create_graph=True: J_S_p must carry a grad_fn (so a downstream
    # backward reaches h / o_proj / lm_head).
    D, V, K = 32, 100, 16
    h, h_norm = _synthetic_hnorm(D)
    W_rows = torch.randn(V, D)
    p = torch.softmax(torch.randn(V), dim=-1)
    out = _compute_jacobian_rows_matrixfree(
        h_norm, h, W_rows, p, list(range(K)), create_graph=True
    )
    assert out.grad_fn is not None, "matrix-free J_S_p must be differentiable"


def test_matrixfree_chunked_matches_full():
    # A4: chunked (chunk_size=8) == full (chunk_size=K) — the chunking must
    # be a memory optimization, not a value change.
    D, V, K = 64, 200, 32
    h, h_norm = _synthetic_hnorm(D)
    torch.manual_seed(2)
    W_rows = torch.randn(V, D)[list(range(K))]  # [K, D]
    p = torch.softmax(torch.randn(K), dim=-1)  # [K] re-normalized
    token_set = list(range(K))

    full = _compute_jacobian_rows_matrixfree(
        h_norm, h, W_rows, p, token_set, create_graph=True, chunk_size=K
    )
    chunked = _compute_jacobian_rows_matrixfree(
        h_norm, h, W_rows, p, token_set, create_graph=True, chunk_size=8
    )
    assert _rel_err(chunked, full) < 1e-6, "chunked != full"


def test_matrixfree_model_level_matches_dense():
    # End-to-end on a real (tiny, nested) Qwen3_5: prefix-cache + decoder +
    # norm + lm_head. matrix-free J_S_p must equal dense J_norm -> rows ->
    # gauge. Validates the _lm_module/_n_layers compat path (nested
    # ForConditionalGeneration: model.model.language_model).
    m = make_tiny_qwen().eval()
    l0 = full_attn_layers(m)[0]  # a non-final full-attn layer
    ids = torch.tensor([[3, 7, 11, 15, 19, 5, 9, 2]])
    am = torch.ones_like(ids)
    K = 16
    token_set = list(range(K))

    J_norm = compute_hidden_jacobian(m, l0, ids, am, create_graph=True)
    logits = m(input_ids=ids, attention_mask=am).logits
    W = m.lm_head.weight
    p = torch.softmax(logits[0, -1].float()[token_set], dim=-1)  # [K] re-norm
    W_rows = W[token_set]  # [K, D]

    # dense reference = exp17 bridge gauge: softmax_jacobian(W[token_set]@J_norm, p)
    ref = p[:, None] * (W_rows @ J_norm - (p @ W_rows) @ J_norm[None, :])
    out = compute_hidden_jacobian_rows(
        m, l0, ids, am, token_set, p, W_rows,
        create_graph=True, chunk_size=4
    )
    rel = _rel_err(out, ref)
    assert rel < 1e-3, f"model-level matrix-free vs dense rel={rel:.3e}"
    assert out.grad_fn is not None
    # gauge identity: also equal to the dense softmax_jacobian directly
    ref2 = softmax_jacobian(W_rows @ J_norm, p)
    assert _rel_err(out, ref2) < 1e-3


def test_softmax_jacobian_gauge_identity():
    # Known answer: with J = I (d logit_i/dh = e_i), J_bar_p = p,
    # Jp_i = p_i*(e_i - p).
    D = 4
    J = torch.eye(D)
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = softmax_jacobian(J, p)
    expected = p[:, None] * (torch.eye(D) - p[None, :])  # row i = p_i*(e_i - p)
    assert _rel_err(out, expected) < 1e-6
