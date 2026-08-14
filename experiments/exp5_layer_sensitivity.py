"""Experiment 5: Layer-wise sensitivity — which layers matter most."""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    build_global_token_set,
    load_calibration_data,
)
from prism.jacobian import compute_jacobian_for_tokens


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    K=8,
    max_length=64,
    min_length=32,
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
    token_set = build_global_token_set(model_obj, samples, K=K)

    results = {}
    for layer in range(n_layers):
        norms = []
        for sample in samples:
            ids = sample["input_ids"].to(model_obj.device)
            attn = sample["attention_mask"].to(model_obj.device)
            J = compute_jacobian_for_tokens(model_obj, layer, ids, attn, token_set)
            norms.append(J.norm().item())

        avg_norm = sum(norms) / len(norms)
        results[f"layer_{layer}"] = avg_norm
        print(f"  Layer {layer}: J.norm = {avg_norm:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp5_layer_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5: Layer sensitivity")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
