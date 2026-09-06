"""Experiment 4: Sensitivity to K (token set size) — cross-model (FIXED).

v1 was BROKEN: it used the self-bridge `jacobian_gap(J, J, E, E)` on a
single (0.8B) model, so 4B was never loaded and every value was the identity
case (~0.0) — that measured nothing about K. v2 mirrors exp4_u: load BOTH
4B (large) + 0.8B (small), build the token set from the 4B top-K final logits,
and compute the cross-bridge `jacobian_gap(J_L, J_S, E_L, E_S, probs)`.

Question: does L_bridge(4B->0.8B) stabilize as K grows? (n_samples>=4, K->256.)
"""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_gap
from prism.jacobian import compute_jacobian_for_tokens


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    max_length=64,
    min_length=32,
    K_values=(4, 8, 16, 32, 64, 128, 256),
    results_dir="results",
):
    print(f"Loading {n_samples} calibration samples (from {large_model})...")
    samples = load_calibration_data(
        large_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    print(f"Loading {large_model}...")
    model_L = AutoModelForCausalLM.from_pretrained(
        large_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_L.eval()

    print(f"Loading {small_model}...")
    model_S = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_S.eval()

    n_L = model_L.config.num_hidden_layers
    n_S = model_S.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_L, n_S)

    step = max(1, len(mapping) // 5)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs (4B->0.8B)")

    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    results = {}
    for K in K_values:
        layer_losses = {}
        for l_L, l_S in sampled:
            ids = samples[0]["input_ids"].to(model_L.device)
            attn = samples[0]["attention_mask"].to(model_L.device)

            with torch.no_grad():
                logits_L = model_L(
                    input_ids=ids, attention_mask=attn
                ).logits[:, -1, :]
            token_set = sorted(logits_L.topk(K, dim=-1).indices[0].tolist())
            with torch.no_grad():
                probs = torch.softmax(logits_L[0, token_set], dim=-1)

            J_L = compute_jacobian_for_tokens(
                model_L, l_L, ids, attn, token_set
            )
            J_S = compute_jacobian_for_tokens(
                model_S, l_S, ids, attn, token_set
            )
            loss = jacobian_gap(J_L, J_S, E_L, E_S, probs)
            layer_losses[f"{l_L}->{l_S}"] = loss

        avg = sum(layer_losses.values()) / max(len(layer_losses), 1)
        results[f"K={K}"] = layer_losses
        print(f"K={K}: avg={avg:.6f}")

    os.makedirs(results_dir, exist_ok=True)
    results["params"] = {
        "large_model": large_model,
        "small_model": small_model,
        "n_samples": n_samples,
        "max_length": max_length,
        "min_length": min_length,
        "K_values": list(K_values),
    }
    with open(os.path.join(results_dir, "exp4_k_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {results_dir}/exp4_k_sensitivity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: K sensitivity (cross-model)")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(
        large_model=args.large_model,
        small_model=args.small_model,
        n_samples=args.n_samples,
        max_length=args.max_length,
        min_length=args.min_length,
        results_dir=args.results_dir,
    )
