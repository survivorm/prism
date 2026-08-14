"""Experiment 4: Sensitivity to |U| (token set size) — cross-model.

Run 4B→0.8B bridge with |U| = 8, 16, 32, 64, 128.
Does L_bridge stabilize at some |U|?
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
from prism.bridge import jacobian_bridge
from prism.jacobian import compute_jacobian_for_tokens


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)

    print(f"Loading {n_samples} calibration samples...")
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

    # Sample a few layers
    step = max(1, len(mapping) // 5)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs")

    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    sample = samples[0]
    ids = sample["input_ids"]
    attn = sample["attention_mask"]

    u_sizes = [8, 16, 32, 64, 128, 256, 512]
    results = {}

    for u_size in u_sizes:
        with torch.no_grad():
            logits_L = model_L(
                input_ids=ids.to(model_L.device), attention_mask=attn.to(model_L.device)
            ).logits[:, -1, :]
        top_k = logits_L.topk(u_size, dim=-1).indices[0].tolist()
        token_set = sorted(top_k)

        with torch.no_grad():
            probs = torch.softmax(logits_L[0, token_set], dim=-1)

        layer_losses = []
        for l_L, l_S in sampled:
            J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
            J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)
            loss = jacobian_bridge(J_L, J_S, E_L, E_S, probs)
            layer_losses.append({"layer": f"{l_L}->{l_S}", "loss": loss})

        avg = sum(r["loss"] for r in layer_losses) / len(layer_losses)
        results[str(u_size)] = {
            "per_layer": layer_losses,
            "avg": avg,
        }
        print(f"|U|={u_size}: avg={avg:.4f}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp4_u_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {results_dir}/exp4_u_sensitivity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: |U| sensitivity (cross-model)")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
