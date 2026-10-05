"""Numerical validation of prism math.

1. Last-layer analytical J_norm vs autograd truth.
2. _run_decoder_to_norm reproduces the true final norm hidden state.
3. compute_jacobian_for_tokens vs independent finite differences.
4. softmax_jacobian gauge invariance.
5. Identity bridge loss ~ 0 on real embeddings.
6. J_logit = W[token_set] @ J_norm chain (last layer).

Run:  .venv/bin/python scripts/verify_math.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

torch.set_num_threads(8)
torch.manual_seed(0)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    return cond


def main():
    from transformers import AutoModelForCausalLM

    import prism.jacobian as PJ
    from prism.jacobian import (
        compute_hidden_jacobian,
        compute_jacobian_for_tokens,
        softmax_jacobian,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    text_model = model.model
    D = text_model.config.hidden_size
    n_layers = text_model.config.num_hidden_layers
    print(f"D={D}, n_layers={n_layers}")

    ids = torch.randint(0, 1000, (1, 32))
    attn = torch.ones_like(ids)
    S = ids.shape[1]

    # ---------- 1. Last-layer norm Jacobian: analytical vs autograd ---------
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        h = outputs.hidden_states[n_layers][:, -1:, :].clone()

    h_leaf = h.clone().detach().requires_grad_(True)
    y = text_model.norm(h_leaf).flatten()
    J_ref = torch.zeros(D, D)
    for i in range(D):
        g = torch.autograd.grad(y[i], h_leaf, retain_graph=(i < D - 1), allow_unused=True)[0]
        if g is not None:
            J_ref[i] = g[0, 0, :]

    J_ana = compute_hidden_jacobian(model, n_layers - 1, ids, attn)
    rel = (J_ana - J_ref).abs().max().item() / J_ref.abs().max().item()
    check("last-layer analytical J_norm == autograd J_norm", rel < 1e-4, f"rel_err={rel:.3e}")

    # Correct RMSNorm formula on the INPUT h (validates the proposed fix)
    x = h[0, 0].detach()
    norm_mod = text_model.norm
    with torch.no_grad():
        eps_n = norm_mod.eps
        xt = x * torch.rsqrt(x.pow(2).mean(-1) + eps_n)
        rms = torch.sqrt(x.pow(2).mean(-1) + eps_n)
        J_corr = (torch.eye(D) - xt.unsqueeze(1) @ xt.unsqueeze(0) / D) / rms
        J_corr = J_corr * (1.0 + norm_mod.weight).unsqueeze(1)
    err_c = (J_corr - J_ref).abs().max().item()
    print(f"  info: correct-RMSNorm-formula max_err vs autograd = {err_c:.3e}")

    # ---------- 2. _run_decoder_to_norm == true final norm state ------------
    l = 6
    with torch.no_grad():
        prefix_cache = PJ._run_prefix_pass(model, ids, attn)
        h_ext = PJ._extract_hidden_state(model, ids, attn, l)
        h_n = h_ext.to(next(model.parameters()).dtype).detach().clone()
        h_norm_code = PJ._run_decoder_to_norm(text_model, prefix_cache, h_n, S, l)

        true_out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        with torch.no_grad():
            h_full = true_out.hidden_states[n_layers][:, -1:, :]  # input to final norm
            true_h_norm = text_model.norm(h_full)

    err2 = (h_norm_code - true_h_norm).abs().max().item()
    scale = true_h_norm.abs().max().item()
    check("_run_decoder_to_norm == true final norm state", err2 / scale < 1e-5,
          f"max_err={err2:.2e} (rel {err2/scale:.2e})")

    # ---------- 3. compute_jacobian_for_tokens vs finite differences --------
    token_set = [101, 770, 2996, 3155]
    J_code = compute_jacobian_for_tokens(model, l, ids, attn, token_set).clone()

    with torch.no_grad():
        prefix_cache = PJ._run_prefix_pass(model, ids, attn)
        h_base = PJ._extract_hidden_state(model, ids, attn, l)
        h_base = h_base.to(next(model.parameters()).dtype).detach().clone()

        def decode_logits(h):
            h2 = h.detach().clone()
            return PJ._run_decoder_pass(model, prefix_cache, h2, S, l)[:, -1, token_set]

        logits0 = decode_logits(h_base)
        eps = 1e-4
        J_fd = torch.zeros(len(token_set), D)
        for j in range(D):
            dp = torch.zeros_like(h_base)
            dp[0, 0, j] = eps
            dl = decode_logits(h_base + dp) - decode_logits(h_base - dp)
            J_fd[:, j] = dl / (2 * eps)

    Jc = J_code.float()
    Jf = J_fd.float()
    rel3 = (Jc - Jf).abs().max().item() / Jf.abs().max().item()
    check("compute_jacobian_for_tokens == finite differences", rel3 < 1e-3,
          f"rel_err={rel3:.3e}")

    # ---------- 4. softmax_jacobian gauge invariance ------------------------
    p = torch.softmax(torch.randn(10), dim=0)
    J = torch.randn(10, D)
    J1 = softmax_jacobian(J, p)
    J2 = softmax_jacobian(J + 0.37, p)
    check("softmax_jacobian gauge-invariant to uniform logit shift",
          bool(torch.allclose(J1, J2, atol=1e-6)),
          f"max diff={(J1 - J2).abs().max().item():.3e}")
    Jc1 = softmax_jacobian(torch.ones(10, D) * 3.0, p)  # constant J -> zero
    check("softmax_jacobian kills constant J", bool(torch.allclose(Jc1, 0, atol=1e-7)))

    # ---------- 5. Identity bridge on real embeddings (tie W=E) -------------
    from prism.bridge import jacobian_gap

    E = model.model.embed_tokens.weight.detach().float()
    torch.manual_seed(1)
    J_true = torch.randn(16, D)
    loss_id = jacobian_gap(J_true, J_true, E, E, torch.softmax(torch.randn(16), 0))
    check("identity bridge loss ~ 0 with real E (W=E)", loss_id < 1e-5, f"loss={loss_id:.2e}")

    # ---------- 6. J_logit = W[token_set] @ J_ref chain (last layer) --------
    W_eff = model.lm_head.weight.detach()
    logits_leaf = h.clone().detach().requires_grad_(True)
    logits = model.lm_head(text_model.norm(logits_leaf)[0, -1, :])[0, token_set]
    J_logit_num = torch.zeros(len(token_set), D)
    for i in range(len(token_set)):
        g = torch.autograd.grad(logits[i], logits_leaf,
                                retain_graph=(i < len(token_set) - 1))[0]
        J_logit_num[i] = g[0, 0, :]
    J_logit_ana = W_eff[token_set] @ J_ref
    rel6 = (J_logit_ana - J_logit_num).abs().max().item() / J_logit_num.abs().max().item()
    check("J_logit = W[token_set] @ J_ref chain (last layer)", rel6 < 1e-4, f"rel_err={rel6:.3e}")

    print("\ndone")


if __name__ == "__main__":
    main()
