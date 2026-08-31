"""Experiment 12A: Perturbation test via sensitivity map.

For layers {6,12,18}:
  - v_g: right singular vector of J whose left vector ≈ uniform (common mode)
  - v_d: top right singular vector of J^(p) (differential direction)

Perturb h by ±ε·v at same norm. Measure KL(p_orig ‖ p_pert).
Prediction: KL_g ≪ KL_d (orders of magnitude).
"""

import argparse
import json
import os

import torch
from scipy.stats import entropy

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    ps = (p + eps) / (p + eps).sum()
    qs = (q + eps) / (q + eps).sum()
    return float(entropy(ps.detach().numpy(), qs.detach().numpy()))


def run_full_decoder(
    model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int
) -> torch.Tensor:
    """Run h through remaining layers, return probs."""
    from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
    from transformers.masking_utils import (  # type: ignore[import-not-found]
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    from transformers import DynamicCache

    cache_clone = DynamicCache()
    for idx in range(len(prefix_cache.layers)):
        lyr = prefix_cache.layers[idx]
        if isinstance(lyr, DynamicLayer):
            nk = [k.clone() if k is not None else None for k in lyr.keys]
            nv = [v.clone() if v is not None else None for v in lyr.values]
            cache_clone.layers.append(DynamicLayer(keys=nk, values=nv))
        else:
            nc = {k: v.clone() for k, v in lyr.conv_states.items()}
            nr = {k: v.clone() for k, v in lyr.recurrent_states.items()}
            cache_clone.layers.append(
                LinearAttentionLayer(conv_states=nc, recurrent_states=nr)
            )

    tm = model.model
    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
    pos_emb = tm.rotary_emb(h, pos[1:])

    dummy = h.new_zeros(1, 1, tm.config.hidden_size)
    mk = {
        "config": tm.config,
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
    for i in range(layer_idx + 1, model.config.num_hidden_layers):
        x = tm.layers[i](
            x,
            position_embeddings=pos_emb,
            attention_mask=masks[model.config.layer_types[i]],
            position_ids=pos[0],
            past_key_values=cache_clone,
        )

    x = tm.norm(x)
    logits = model.lm_head(x[:, -1, :])
    return torch.softmax(logits[0], dim=-1)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    K=64,
    max_length=64,
    min_length=32,
    eps_list=None,
    results_dir="results",
):
    torch.set_num_threads(4)
    torch.manual_seed(42)

    if eps_list is None:
        eps_list = [1.0]

    samples = load_calibration_data(model, max_length, min_length, n_samples)

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    selected = [6, 12, 18]

    s0 = samples[0]
    ids0 = s0["input_ids"].to(m.device)
    attn0 = s0["attention_mask"].to(m.device)
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = lg0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    all_results = {}

    for eps_val in eps_list:
        eps_key = f"eps_{eps_val}"
        print(f"\n{'='*60}")
        print(f"Epsilon = {eps_val}")
        print(f"{'='*60}")
        eps_results = {}

        for layer in selected:
            print(f"\nLayer {layer}:")
            layer_results = []

            for si, sample in enumerate(samples):
                ids = sample["input_ids"]
                attn = sample["attention_mask"]
                S = ids.shape[1]

                J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)

                # Forward for baseline
                with torch.no_grad():
                    out = m(
                        input_ids=ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                prefix_cache = out.past_key_values
                h_base = out.hidden_states[layer + 1][:, -1:, :]  # [1,1,D]
                p_orig = torch.softmax(m.lm_head(m.model.norm(h_base))[:, -1, :][0], dim=-1)

                # p for J^(p)
                p_tokens = torch.softmax(out.logits[0, -1, token_set], dim=-1)
                J_p = softmax_jacobian(J, p_tokens)

                # SVD of J → find gauge direction (left vector ≈ uniform)
                U_j, S_j, Vt_j = torch.linalg.svd(J.float(), full_matrices=False)
                ones = torch.ones(J.shape[0]) / torch.sqrt(torch.tensor(float(J.shape[0])))
                uniformity = torch.abs(U_j.T @ ones)  # [min(K,D)]
                gauge_idx = uniformity.argmax()
                v_g = Vt_j[gauge_idx]

                # Top right singular vector of J^(p)
                _, _, Vt_p = torch.linalg.svd(J_p.float(), full_matrices=False)
                v_d = Vt_p[0]

                # Normalize
                v_g = v_g / v_g.norm()
                v_d = v_d / v_d.norm()

                results_g = {}
                results_d = {}

                for sign in [-1, +1]:
                    delta_g = sign * eps_val * v_g
                    h_g = h_base + delta_g
                    p_g = run_full_decoder(m, prefix_cache, h_g, S, layer)
                    results_g[sign] = kl_div(p_orig, p_g)

                    delta_d = sign * eps_val * v_d
                    h_d = h_base + delta_d
                    p_d = run_full_decoder(m, prefix_cache, h_d, S, layer)
                    results_d[sign] = kl_div(p_orig, p_d)

                entry = {
                    "layer": layer,
                    "sample": si,
                    "uniformity_vg": uniformity[gauge_idx].item(),
                    "KL_g_minus": results_g[-1],
                    "KL_g_plus": results_g[+1],
                    "KL_d_minus": results_d[-1],
                    "KL_d_plus": results_d[+1],
                }
                layer_results.append(entry)
                print(
                    f"  sample {si}: KL_g={[round(v, 4) for v in results_g.values()]}, KL_d={[round(v, 4) for v in results_d.values()]}"
                )

            eps_results[layer] = layer_results

            avg_klg = sum(r["KL_g_plus"] for r in layer_results) / len(layer_results)
            avg_kld = sum(r["KL_d_plus"] for r in layer_results) / len(layer_results)
            print(
                f"  avg: KL_g+={avg_klg:.4f}, KL_d+={avg_kld:.4f}, ratio={avg_kld / (avg_klg + 1e-8):.1f}x"
            )

        all_results[eps_key] = eps_results

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {"model": model, "n_samples": n_samples, "K": K, "max_length": max_length, "min_length": min_length, "eps_list": eps_list}
    with open(os.path.join(results_dir, "exp12a_perturbation_test.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp12a_perturbation_test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 12A: Perturbation test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--eps-list", default="1.0",
                        help="Comma-separated epsilon values, e.g. 0.01,0.1,0.5,1.0")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    args.eps_list = [float(x) for x in args.eps_list.split(",")]
    main(**vars(args))
