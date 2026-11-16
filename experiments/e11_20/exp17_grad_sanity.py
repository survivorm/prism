"""Sanity: bridge grad flows to lm_head AND o_proj LoRA."""

import torch

from prism.jacobian import compute_hidden_jacobian, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


def main():
    torch.manual_seed(42)
    device = "cpu"
    torch.set_num_threads(4)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    # DON'T call model.eval() — keeps grad graph alive through frozen layers

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    l_S = 6
    r = 2
    V, D = model.lm_head.weight.shape
    token_set = list(range(4))

    # LoRA params
    A_lh = torch.nn.Parameter(torch.randn(r, D, device=device) * 0.02)
    B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device).T)

    # Hook on attention layer
    layer_mod = model.model.layers[l_S]
    attn_mod = (
        layer_mod.self_attn
        if hasattr(layer_mod, "self_attn")
        else layer_mod.linear_attn
    )

    # Find o_proj shape
    D_l = D  # fallback
    for name, param in attn_mod.named_parameters():
        if "o_proj" in name:
            _, D_l = param.shape
            break

    A_attn = torch.nn.Parameter(torch.randn(r, D_l, device=device) * 0.02)
    B_attn = torch.nn.Parameter(torch.zeros(r, D_l, device=device).T)

    _b = B_attn
    _a = A_attn

    def _hook(mod, inp, out):
        delta = (
            (_b @ _a @ out[0].transpose(-1, -2)).transpose(-1, -2)
            if isinstance(out, tuple)
            else (_b @ _a @ out.transpose(-1, -2)).transpose(-1, -2)
        )
        if isinstance(out, tuple):
            return (out[0] + delta,) + out[1:]
        return out + delta

    attn_mod.register_forward_hook(_hook)

    # Fake target
    J_tgt_p = torch.randn(len(token_set), D, device=device)
    J_tgt_p = J_tgt_p / J_tgt_p.norm(dim=1, keepdim=True)
    p_L = torch.softmax(torch.randn(len(token_set), device=device), dim=0)

    # Sample
    ids = torch.randint(0, 1000, (1, 32), device=device)
    attn = torch.ones_like(ids)

    # Check h WITHOUT no_grad — does hook create grad graph?
    outs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    h_check = outs.hidden_states[l_S + 1][:, -1:, :]
    print(f"h.grad_fn: {h_check.grad_fn}")
    print(f"A_attn.requires_grad: {A_attn.requires_grad}")
    print(f"B_attn.requires_grad: {B_attn.requires_grad}")

    # Forward
    W_eff = model.lm_head.weight + B_lh @ A_lh
    J_norm = compute_hidden_jacobian(
        model, l_S, ids, attn, create_graph=True, verbose=False
    )
    J_S = W_eff[token_set] @ J_norm
    J_S_p = softmax_jacobian(J_S, p_L)
    cos = torch.nn.functional.cosine_similarity(J_S_p.float(), J_tgt_p.float(), dim=1)
    loss = (1.0 - cos).mean()

    loss.backward()

    # Check
    print(
        f"A_lh.grad norm: {A_lh.grad.norm().item():.6e}"
        if A_lh.grad is not None
        else "A_lh.grad: None"
    )
    print(
        f"B_lh.grad norm: {B_lh.grad.norm().item():.6e}"
        if B_lh.grad is not None
        else "B_lh.grad: None"
    )
    print(
        f"A_attn.grad norm: {A_attn.grad.norm().item():.6e}"
        if A_attn.grad is not None
        else "A_attn.grad: None"
    )
    print(
        f"B_attn.grad norm: {B_attn.grad.norm().item():.6e}"
        if B_attn.grad is not None
        else "B_attn.grad: None"
    )

    ok = True
    for name, g in [
        ("A_lh", A_lh.grad),
        ("B_lh", B_lh.grad),
        ("A_attn", A_attn.grad),
        ("B_attn", B_attn.grad),
    ]:
        if g is None or g.norm().item() < 1e-8:
            print(f"FAIL: {name} has no gradient!")
            ok = False
        else:
            print(f"PASS: {name} grad = {g.norm().item():.6e}")

    if ok:
        print("\nALL PASS: bridge gradient flows to ALL LoRA params (lm_head + o_proj)")


if __name__ == "__main__":
    main()
