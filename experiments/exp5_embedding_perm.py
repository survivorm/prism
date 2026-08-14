"""Experiment 5: Embedding permutation test.

Randomly permute rows of E_L. If bridge breaks → semantics matter (good).
If bridge survives → bridge ignores semantics (bad).
"""

import argparse
import json
import os

import torch

from prism.calibration import extract_embeddings, load_calibration_data
from prism.bridge import jacobian_bridge
from prism.jacobian import compute_jacobian_for_tokens


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=2,
    K=8,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)

    from transformers import AutoModelForCausalLM

    samples = load_calibration_data(
        model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    print(f"Loading {model}...")
    m = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    m.eval()

    E = extract_embeddings(m)

    sample = samples[0]
    ids = sample["input_ids"].to(m.device)
    attn = sample["attention_mask"].to(m.device)

    with torch.no_grad():
        logits = m(input_ids=ids, attention_mask=attn).logits[:, -1, :]
    top_k = logits.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    with torch.no_grad():
        probs = torch.softmax(logits[0, token_set], dim=-1)

    E = extract_embeddings(m)
    results = {}

    # Baseline: normal embeddings
    for layer in [0, 12, 23]:
        J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
        loss = jacobian_bridge(J, J, E, E, probs)
        results[f"baseline_layer_{layer}"] = loss
        print(f"baseline layer {layer}: {loss:.2e}")

    # Permuted embeddings
    V = E.shape[0]
    perm = torch.randperm(V)
    E_perm = E[perm]

    for layer in [0, 12, 23]:
        J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
        loss = jacobian_bridge(J, J, E_perm, E, probs)
        results[f"perm_layer_{layer}"] = loss
        print(f"permuted layer {layer}: {loss:.2e}")

    # Inverse permuted (E permuted as both L and S)
    for layer in [0, 12, 23]:
        J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
        loss = jacobian_bridge(J, J, E_perm, E_perm, probs)
        results[f"inv_perm_layer_{layer}"] = loss
        print(f"inv_perm layer {layer}: {loss:.2e}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp5_embedding_perm.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {results_dir}/exp5_embedding_perm.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5: Embedding permutation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
