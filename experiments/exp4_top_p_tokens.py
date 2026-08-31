"""Experiment 4 (variant): Token-set sensitivity with top-p vs uniform vs random-negative.

Compare bridge loss using three token selection strategies:
  1. uniform: current top-K from softmax (baseline)
  2. top-p: tokens covering cumulative probability p=0.92
  3. random: uniformly random tokens from vocabulary (low-probability negatives)

This addresses reviewer concern about uniform token-set bias.
"""

import argparse
import json
import os
import random

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    build_global_token_set,
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_bridge
from prism.jacobian import compute_jacobian_for_tokens


def build_top_p_token_set(
    model, calibration_samples, p_target=0.92, seed=42
):
    """Build token set covering cumulative probability p_target."""
    token_counter = {}
    model.eval()

    for sample in calibration_samples:
        ids = sample["input_ids"].to(model.device)
        attn = sample["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        probs = torch.softmax(logits[0], dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum <= p_target
        selected = sorted_indices[mask].tolist()
        for t in selected:
            token_counter[t] = token_counter.get(t, 0) + 1

    U = [tid for tid, _ in sorted(token_counter.items(), key=lambda x: -x[1])]
    return sorted(U[:min(len(U), 512)])  # cap at 512


def build_random_token_set(model, K=128, seed=42):
    """Build token set of uniformly random tokens (low-probability negatives)."""
    vocab_size = model.config.vocab_size
    rng = random.Random(seed)
    # Exclude special tokens (first ~10)
    candidates = list(range(10, vocab_size))
    U = sorted(rng.sample(candidates, min(K, len(candidates))))
    return U


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
        print("WARNING: Running on CPU — will be very slow for 4B model")

    samples = load_calibration_data(
        small_model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )
    print(f"Loaded {len(samples)} samples")

    print(f"Loading {large_model}...")
    dtype_L = torch.float16 if torch.cuda.is_available() else torch.float32
    model_L = AutoModelForCausalLM.from_pretrained(
        large_model, torch_dtype=dtype_L, attn_implementation="eager"
    )
    model_L.eval()

    print(f"Loading {small_model}...")
    model_S = AutoModelForCausalLM.from_pretrained(
        small_model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model_S.eval()

    n_L = model_L.config.num_hidden_layers
    n_S = model_S.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_L, n_S)

    # Sample ~5 layer pairs for speed
    step = max(1, len(mapping) // 5)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs")

    EL = extract_embeddings(model_L)
    ES = extract_embeddings(model_S)

    # Build three token sets
    print("\nBuilding token sets...")

    # 1. Uniform (current baseline: top-K from softmax)
    ts_uniform = build_global_token_set(model_L, samples, K=64)
    print(f"  uniform (top-64): {len(ts_uniform)} tokens")

    # 2. Top-p=0.92
    ts_topp = build_top_p_token_set(model_L, samples, p_target=0.92)
    print(f"  top-p=0.92: {len(ts_topp)} tokens")

    # 3. Random
    ts_random = build_random_token_set(model_L, K=64)
    print(f"  random: {len(ts_random)} tokens")

    token_sets = {
        "uniform_64": ts_uniform,
        "top_p_92": ts_topp,
        "random_64": ts_random,
    }

    results = {}

    # Use first sample for bridge computation
    sample = samples[0]
    for name, token_set in token_sets.items():
        print(f"\n=== Token set: {name} ({len(token_set)} tokens) ===")

        ids_L = sample["input_ids"].to(model_L.device)
        attn_L = sample["attention_mask"].to(model_L.device)
        with torch.no_grad():
            logits_L = model_L(input_ids=ids_L, attention_mask=attn_L).logits[:, -1, :]
        probs = torch.softmax(logits_L[0, token_set], dim=-1)

        layer_losses = {}
        for l_L, l_S in sampled:
            ids_L_d = sample["input_ids"].to(model_L.device)
            attn_L_d = sample["attention_mask"].to(model_L.device)
            J_L = compute_jacobian_for_tokens(model_L, l_L, ids_L_d, attn_L_d, token_set)
            ids_S = sample["input_ids"].to(model_S.device)
            attn_S = sample["attention_mask"].to(model_S.device)
            J_S = compute_jacobian_for_tokens(model_S, l_S, ids_S, attn_S, token_set)

            loss = jacobian_bridge(J_L, J_S, EL, ES, probs)
            layer_losses[f"{l_L}->{l_S}"] = loss
            print(f"  {l_L}->{l_S}: loss={loss:.6f}")

        avg = sum(layer_losses.values()) / len(layer_losses)
        results[name] = {"per_layer": layer_losses, "avg": avg}

    # Save
    os.makedirs(results_dir, exist_ok=True)
    output = {
        **results,
        "params": {
            "large_model": large_model,
            "small_model": small_model,
            "n_samples": n_samples,
            "max_length": max_length,
            "min_length": min_length,
        },
    }
    with open(os.path.join(results_dir, "exp4_top_p_tokens.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_dir}/exp4_top_p_tokens.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: Token-set sensitivity (top-p, random)")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
