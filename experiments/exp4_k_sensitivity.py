"""Experiment 4: Sensitivity to K (token set size)."""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    build_global_token_set,
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.jacobian import compute_jacobian_for_tokens
from prism.bridge import jacobian_bridge


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    max_length=64,
    min_length=32,
    K_values=(4, 8, 16, 32),
    results_dir="results",
):
    torch.set_num_threads(4)
    samples = load_calibration_data(
        model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_obj.eval()
    n_layers = model_obj.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_layers, n_layers)
    E = extract_embeddings(model_obj)

    results = {}

    for K in K_values:
        token_set = build_global_token_set(model_obj, samples, K=K)
        layer_losses = {}

        for l_L in list(mapping.keys())[:5]:
            ids = samples[0]["input_ids"].to(model_obj.device)
            attn = samples[0]["attention_mask"].to(model_obj.device)
            J = compute_jacobian_for_tokens(model_obj, l_L, ids, attn, token_set)

            with torch.no_grad():
                logits = model_obj(input_ids=ids).logits[:, -1, :]
                probs = torch.softmax(logits[0, token_set], dim=-1)

            loss = jacobian_bridge(J, J, E, E, probs)
            layer_losses[f"layer_{l_L}"] = loss

        results[f"K={K}"] = layer_losses
        print(
            f"K={K}: avg={sum(layer_losses.values()) / max(len(layer_losses), 1):.8f}"
        )

    os.makedirs(results_dir, exist_ok=True)
    results["params"] = {model: model, n_samples: n_samples, max_length: max_length, min_length: min_length, K_values: K_values}
    with open(os.path.join(results_dir, "exp4_k_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: K sensitivity")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(
        model=args.model,
        n_samples=args.n_samples,
        max_length=args.max_length,
        min_length=args.min_length,
        results_dir=args.results_dir,
    )
