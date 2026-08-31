"""v23 A4: bridge_rows_backward chunked-backward correctness.

Mirrors exp17's EXACT wiring (docs #16): the KL is one `m.model(...)`
forward (logits -> kl + p); the bridge h is a SEPARATE `m.model(...)`
forward (inside bridge_rows_backward/_forward_h_norm). Because the two are
separate forwards, the KL graph never shares nodes with the bridge h graph
and can backprop on its own. This is what lets the chunked path work.

Checks:
  * param grads (A_lh, B_lh, A_attn, B_attn, lm_head) from the chunked
    per-chunk backward == a single dense one-shot backward,
  * the returned detached J_S_p rows match the dense gauge,
  * the scalar bridge loss matches,
  * chunk_size invariance (C=8 vs C=K).
"""

import torch

from prism.jacobian import (
    _lm_module,
    bridge_rows_backward,
    compute_hidden_jacobian,
)
from tests.helpers.tiny_model import full_attn_layers, make_tiny_qwen

K = 16  # <= tiny vocab (64)
IDS = torch.tensor([[3, 7, 11, 15, 19, 5, 9, 2]])
AM = torch.ones_like(IDS)
TOKEN_SET = list(range(K))
KL_W = 0.1


def _tiny_with_lora(seed: int):
    """Tiny Qwen3_5 (eager, CPU) + LoRA on lm_head + o_proj, EXACTLY like
    exp17: A_lh=[r,D], B_lh=[V,r] (B@A=[V,D]=W), o_proj forward patched."""
    torch.manual_seed(seed)
    m = make_tiny_qwen(n_layers=4, attn_implementation="eager").eval()
    l0 = full_attn_layers(m)[0]
    o_proj = _lm_module(m).layers[l0].self_attn.o_proj
    V, D = m.lm_head.weight.shape
    r = 4
    torch.manual_seed(7)
    A_lh = torch.nn.Parameter(torch.randn(r, D) * 0.02, requires_grad=True)
    B_lh = torch.nn.Parameter(torch.zeros(r, V).T, requires_grad=True)  # [V,r]
    B_attn = torch.nn.Parameter(
        torch.randn(o_proj.out_features, r) * 0.02, requires_grad=True
    )
    A_attn = torch.nn.Parameter(
        torch.randn(r, o_proj.in_features) * 0.02, requires_grad=True
    )
    orig_fwd = o_proj.forward

    def patched(x, _b=B_attn, _a=A_attn, _f=orig_fwd):
        return _f(x) + (_b @ _a @ x.transpose(-1, -2)).transpose(-1, -2)

    o_proj.forward = patched
    return m, l0, A_lh, B_lh, A_attn, B_attn


def _kl_and_p(m, B_lh, A_lh):
    """exp17's KL forward: one m.model forward -> edited logits -> kl + p.
    Separate from the bridge h forward (which bridge_rows_backward does)."""
    W_eff = m.lm_head.weight + B_lh @ A_lh  # [V, D] live
    out = m.model(input_ids=IDS, attention_mask=AM)
    hidden = out.last_hidden_state[:, -1, :]
    b = m.lm_head.bias if m.lm_head.bias is not None else 0
    logits_ed = hidden @ W_eff.T + b
    p_ed = torch.softmax(logits_ed.float(), dim=-1)
    kl = (p_ed * torch.log(p_ed.clamp_min(1e-12) / (0.1 + 1e-12))).mean()
    p = torch.softmax(logits_ed[0][TOKEN_SET].float(), dim=-1)  # re-norm K-subset
    return W_eff, kl, p


def test_bridge_rows_backward_grads_match_dense():
    torch.manual_seed(21)
    J_tgt = torch.randn(K, 64)  # 64 = tiny hidden (default)

    # ---- reference: dense one-shot (KL + dense bridge), single backward ---
    m, l0, A_lh, B_lh, A_attn, B_attn = _tiny_with_lora(123)
    W_eff, kl, p = _kl_and_p(m, B_lh, A_lh)
    W_rows = W_eff[TOKEN_SET]
    J_norm = compute_hidden_jacobian(m, l0, IDS, AM, create_graph=True)
    J_S = W_rows @ J_norm            # [K, D]  (J_i = W_rows[i] @ J_norm)
    J_bar = (p @ W_rows) @ J_norm    # [D]     (the softmax-gauge mean row)
    J_S_p = p[:, None] * (J_S - J_bar)  # [K, D]  row_i = p_i*(J_i - J_bar)
    br_ref = (1 - torch.nn.functional.cosine_similarity(J_S_p, J_tgt, 1)).mean()
    (br_ref + KL_W * kl).backward()  # single one-shot
    g_ref = {n: getattr(x, "grad").clone() for n, x in [
        ("A_lh", A_lh), ("B_lh", B_lh), ("A_attn", A_attn), ("B_attn", B_attn)]}
    g_ref["lm"] = m.lm_head.weight.grad.clone()

    # ---- candidate: chunked bridge_rows_backward --------------------------
    m2, l1, A_lh2, B_lh2, A_attn2, B_attn2 = _tiny_with_lora(123)
    W_eff2, kl2, p2 = _kl_and_p(m2, B_lh2, A_lh2)
    out, br2 = bridge_rows_backward(
        m2, l1, IDS, AM, TOKEN_SET, p2, W_eff2[TOKEN_SET], J_tgt,
        extra_loss=KL_W * kl2, chunk_size=8,
    )
    g_cand = {n: getattr(x, "grad").clone() for n, x in [
        ("A_lh", A_lh2), ("B_lh", B_lh2), ("A_attn", A_attn2),
        ("B_attn", B_attn2)]}
    g_cand["lm"] = m2.lm_head.weight.grad.clone()

    # 1. parameter gradients match
    for n in ("A_lh", "B_lh", "A_attn", "B_attn", "lm"):
        a, c = g_ref[n], g_cand[n]
        assert a is not None and c is not None, f"{n} missing grad"
        rel = (a - c).norm().item() / max(a.norm().item(), 1e-12)
        assert rel < 1e-4, f"{n} grad rel err {rel:.3e}"

    # 2. returned rows match the dense gauge:
    # row_i = p_i * (W_rows[i] @ J_norm - p @ W_rows @ J_norm)
    J_norm2 = compute_hidden_jacobian(m2, l1, IDS, AM, create_graph=False)
    Wm = W_eff2[TOKEN_SET]
    J_ref = torch.stack([p2[i] * (Wm[i] @ J_norm2 - p2 @ Wm @ J_norm2) for i in range(K)])
    rel = (out.float() - J_ref.float()).norm().item() / max(J_ref.norm().item(), 1e-12)
    assert rel < 1e-3, f"J_S_p rows rel err {rel:.3e}"

    # 3. scalar bridge loss matches
    br_ref2 = (1 - torch.nn.functional.cosine_similarity(J_ref, J_tgt, 1)).mean()
    assert abs(br2 - br_ref2.item()) < 1e-5, f"bridge loss {br2} vs {br_ref2}"

    # 4. A_lh is DEAD at init (B_lh=0 -> W_eff=W; standard LoRA). Both the
    # dense reference and the chunked candidate give dL/dA_lh == 0 here, so
    # the grad-match above (check 1) already covers it. No >0 expectation.
    print("PASS: bridge_rows_backward grads match dense one-shot")


def test_bridge_rows_chunk_size_invariance():
    torch.manual_seed(22)
    J_tgt = torch.randn(K, 64)

    def run(chunk: int):
        mm, l, A, B, Aa, Ba = _tiny_with_lora(123)
        W_eff, kl, p = _kl_and_p(mm, B, A)
        return bridge_rows_backward(
            mm, l, IDS, AM, TOKEN_SET, p, W_eff[TOKEN_SET], J_tgt,
            extra_loss=KL_W * kl, chunk_size=chunk,
        )

    out8, br8 = run(8)
    outK, brK = run(K)
    assert (out8 - outK).norm().item() < 1e-5, "rows differ across chunk size"
    assert abs(br8 - brK) < 1e-6, f"bridge loss {br8} vs {brK}"
    print("PASS: bridge_rows_backward chunk-size invariant (C=8 vs C=K)")


if __name__ == "__main__":
    test_bridge_rows_backward_grads_match_dense()
    test_bridge_rows_chunk_size_invariance()
    print("ALL PASS")
