"""Sanity: compute_hidden_jacobian under the four caller regimes Kaggle hits.

Memory-bounded model (0.8B); the row loop runs the full D=1024 backward
passes (identical autograd path to production).

Scenarios (each must produce a finite J with no crash):
  S0: caller OUTSIDE no_grad, model FROZEN, general path
      (exp17 training J_norm: leaf-h fallback must keep the graph alive)
  S1: caller under torch.no_grad, general path (exp17 eval loop crash site)
  S2: fully frozen model, create_graph=True, general path
      (exp17 pair with linear_attention layer: no LoRA hook -> h built
       under enable_grad, leaf-h fallback, row loop needs grad mode ON)
  S3: control — plain call, unfrozen model, general path (worked before too)

Exit 0 = all pass. This is the RED/GREEN check for the enable_grad +
detach-fallback fix in prism/jacobian.py.
"""

import sys
import torch

sys.path.insert(0, ".")
from prism.jacobian import compute_hidden_jacobian  # noqa: E402


def _last_attention_layer(m):
    """Index of the last full-attention layer (linear_attn layers have no o_proj)."""
    types = m.config.layer_types
    for i in range(len(types) - 1, -1, -1):
        if types[i] != "linear_attention":
            return i
    raise ValueError("no full-attention layers")


MODEL = "Qwen/Qwen3.5-0.8B"
K = 16
INPUT = (
    "The transport map pushes the small model hidden state toward the "
    "large model readout direction, layer by layer."
)


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    m.eval()
    m.tokenizer = AutoTokenizer.from_pretrained(MODEL)
    return m


def mtokenize(m, text):
    return m.tokenizer(text, return_tensors="pt")["input_ids"]


def mattn(m, ids):
    return torch.ones_like(ids)


def run_case(name, m, ids, l_S, attn_s, token_set, mode, wrap_nograd, freeze):
    if freeze:
        for p in m.parameters():
            p.requires_grad_(False)
    kw = dict(create_graph=(mode == "create_graph"))
    if not freeze and l_S == _last_attention_layer(m):
        kw["last_layer"] = True  # analytical path, no autograd
    try:
        if wrap_nograd:
            with torch.no_grad():
                J = compute_hidden_jacobian(m, l_S, ids, attn_s, **kw)
        else:
            J = compute_hidden_jacobian(m, l_S, ids, attn_s, **kw)
        finite = torch.isfinite(J).all().item()
        ok = finite and J.abs().sum().item() > 0
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}: J={tuple(J.shape)} "
            f"finite={finite} nonzero={J.abs().sum().item() > 0}"
        )
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[CRASH] {name}: {type(e).__name__}: {e}")
        return False


def main():
    m = load()
    ids = mtokenize(m, INPUT)
    attn_s = mattn(m, ids)
    token_set = torch.topk(m(ids).logits[0, -1], K).indices.sort().values.tolist()
    # full-attn layer for general path, linear-attn (no o_proj) for S2
    l_full = _last_attention_layer(m) - 1  # second-to-last full layer
    l_lin = 21  # known linear_attention (no o_proj) — the 28_21 crash
    ok = run_case(
        "S0 plain + FROZEN + general path",
        m,
        ids,
        l_full,
        attn_s,
        token_set,
        "general",
        False,
        True,
    )
    m = load()
    ok = ok and run_case(
        "S1 no_grad + general path",
        m,
        ids,
        l_full,
        attn_s,
        token_set,
        "general",
        True,
        False,
    )
    m2 = load()
    ok = ok and run_case(
        "S2 frozen + create_graph (no hook)",
        m2,
        ids,
        l_lin,
        attn_s,
        token_set,
        "create_graph",
        False,
        True,
    )
    m3 = load()
    ok = ok and run_case(
        "S3 control plain general path",
        m3,
        ids,
        l_full,
        attn_s,
        token_set,
        "general",
        False,
        False,
    )
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
