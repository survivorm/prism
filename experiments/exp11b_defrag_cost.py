"""Experiment 11B: Cost measurement — raw vs defrag.

Measure:
  - T_raw: time for compute_jacobian_for_tokens (|U| backward passes)
  - T_defrag(r): time for batched J_r computation (r backward, single autograd call)
  - Projection cost: |U|*D*V vs r*D*V
  - Storage: |U|*D vs r*D

Layers {6, 12, 18}, |U|=256, r in {8, 16, 32}.
"""

import argparse
import json
import os
import time

import torch
from transformers import DynamicCache  # type: ignore[import-not-found]

from prism.calibration import load_calibration_data
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)


def compute_jacobian_batched(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    basis_rows: torch.Tensor,  # [r, K] — basis in token space
    token_set: list[int],
) -> torch.Tensor:
    """Compute J_r = basis @ J^(p) with r batched backward passes.

    basis_rows has shape [r, K]. Each row defines a linear combination of
    logit gradients. Single autograd call with retain_graph computes all r rows.

    Returns J_r: [r, D].
    """
    from transformers.cache_utils import (  # type: ignore[import-not-found]
        DynamicLayer,
        LinearAttentionLayer,
    )
    from transformers.masking_utils import (  # type: ignore[import-not-found]
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    S = input_ids.shape[1]
    model.eval()

    # Prefix pass
    with torch.no_grad():
        out = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
            output_hidden_states=False,
        )
    prefix_cache = out.past_key_values

    # Full pass → hidden state
    with torch.no_grad():
        full = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    h = (
        full.hidden_states[layer_idx + 1][:, -1:, :]
        .float()
        .detach()
        .requires_grad_(True)
    )

    # Decoder pass
    cache_clone = DynamicCache()
    for idx in range(len(prefix_cache.layers)):
        lyr = prefix_cache.layers[idx]
        if isinstance(lyr, DynamicLayer):
            nk = [k.clone() if k is not None else None for k in lyr.keys]  # type: ignore[union-attr]
            nv = [v.clone() if v is not None else None for v in lyr.values]  # type: ignore[union-attr]
            cache_clone.layers.append(DynamicLayer(keys=nk, values=nv))
        else:
            nc = {k: v.clone() for k, v in lyr.conv_states.items()}
            nr = {k: v.clone() for k, v in lyr.recurrent_states.items()}
            cache_clone.layers.append(
                LinearAttentionLayer(conv_states=nc, recurrent_states=nr)
            )

    text_model = model.model
    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
    pos_emb = text_model.rotary_emb(h, pos[1:])

    dummy = h.new_zeros(1, 1, text_model.config.hidden_size)
    mk = {
        "config": text_model.config,
        "inputs_embeds": dummy,
        "attention_mask": None,
        "past_key_values": cache_clone,
        "position_ids": pos[0],
    }
    masks = {
        "full_attention": create_causal_mask(**mk),
        "linear_attention": create_recurrent_attention_mask(**mk),
    }

    x = h
    n_layers = model.config.num_hidden_layers
    for i in range(layer_idx + 1, n_layers):
        lt = model.config.layer_types[i]
        x = text_model.layers[i](
            x,
            position_embeddings=pos_emb,
            attention_mask=masks[lt],
            position_ids=pos[0],
            past_key_values=cache_clone,
        )

    x = text_model.norm(x)
    logits = model.lm_head(x[:, -1, :])
    logits_subset = logits[0, token_set]  # [K]

    # Single backward with matrix grad_outputs: [r, K] → [r, D]
    r = basis_rows.shape[0]
    J_r = torch.zeros(r, h.shape[-1], device=h.device, dtype=h.dtype)

    for ri in range(r):
        g_out = basis_rows[ri]  # [K]
        grad = torch.autograd.grad(
            outputs=logits_subset,
            inputs=h,
            grad_outputs=g_out,
            retain_graph=(ri < r - 1),
            create_graph=False,
        )[0]
        J_r[ri] = grad[0, 0, :]

    return J_r


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=2,
    K=256,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    torch.manual_seed(42)

    print(f"Loading {n_samples} samples...")
    samples = load_calibration_data(model, max_length, min_length, n_samples)

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    selected = [6, 12, 18]
    rs = [8, 16, 32]
    D = m.config.hidden_size
    V = m.config.vocab_size

    # Token set from first sample
    s0 = samples[0]
    ids0 = s0["input_ids"].to(m.device)
    attn0 = s0["attention_mask"].to(m.device)
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = lg0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)
    K_actual = len(token_set)

    all_results = {}

    for layer in selected:
        print(f"\nLayer {layer}:")
        layer_res = {}

        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]

            # === RAW ===
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_raw = time.perf_counter() - t0

            # Softmax probs for gauge-free
            with torch.no_grad():
                lg = m(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            p_tokens = torch.softmax(lg[0, token_set], dim=-1)
            J_p = softmax_jacobian(J, p_tokens)

            # Defrag basis: eigenvectors of G^(p) = J_p @ J_p.T
            G_p = J_p @ J_p.T
            ev, eig = torch.linalg.eigh(G_p)
            idx = ev.argsort(descending=True)
            U_all = eig[:, idx]  # [K, K]

            # === DEFRAG R ===
            for r in rs:
                U_r = U_all[:, :r]  # [K, r]
                basis_rows = U_r.T  # [r, K]

                t0 = time.perf_counter()
                J_r = compute_jacobian_batched(
                    m, layer, ids, attn, basis_rows, token_set
                )
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t_defrag = time.perf_counter() - t0

                # Reconstruction fidelity
                J_tilde = U_r @ J_r  # [K, D]
                recon_fidelity = (J_tilde - J_p).norm() / (J_p.norm() + 1e-10)

                speedup = t_raw / (t_defrag + 1e-9)

                layer_res.setdefault(r, []).append(
                    {
                        "sample": si,
                        "t_raw_ms": t_raw * 1000,
                        "t_defrag_ms": t_defrag * 1000,
                        "speedup": speedup,
                        "recon_fidelity_rel": recon_fidelity.item(),
                    }
                )

                print(
                    f"  sample {si}: r={r}, raw={t_raw * 1000:.1f}ms, defrag={t_defrag * 1000:.1f}ms, speedup={speedup:.1f}x, recon={recon_fidelity.item():.4f}"
                )

            layer_res["raw_time_ms"] = t_raw * 1000

        # Projection cost estimate
        layer_res["projection_raw_FLOPs"] = K_actual * D * V * 2
        for r in rs:
            layer_res[f"projection_dr{r}_FLOPs"] = r * D * V * 2
        layer_res["storage_raw_elements"] = K_actual * D
        for r in rs:
            layer_res[f"storage_dr{r}_elements"] = r * D

        all_results[layer] = layer_res

    # Offline basis cost (one-time per model)
    print(f"\nOffline basis cost: SVD of {K_actual}x{K_actual} matrix")
    G_test = torch.randn(K_actual, K_actual)
    t0 = time.perf_counter()
    torch.linalg.eigh(G_test)
    t_offline = time.perf_counter() - t0
    all_results["_offline_basis_ms"] = t_offline * 1000

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length}
    with open(os.path.join(results_dir, "exp11b_defrag_cost.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp11b_defrag_cost.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 11B: Defrag cost")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
