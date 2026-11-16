"""Experiment 10c: Top-K vs Top-P coverage comparison.

Problem: top-K gives unstable coverage across samples.
Solution: top-p guarantees coverage by definition.

Reports:
  - For each K: coverage per sample (shows variability)
  - For each p: required K per sample (shows adaptation)
  - Demonstrates why top-p is preferred
"""

import argparse
import json
import os

import numpy as np
import torch

from prism.calibration import load_calibration_data


def compute_topk_coverage(p_cumsum, K):
    """Coverage of top-K tokens."""
    return p_cumsum[K - 1].item()


def compute_topk_tokens_needed(p_cumsum, p_target):
    """Minimum K to reach target cumulative probability."""
    idx = (p_cumsum >= p_target).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return len(p_cumsum)
    return int(idx[0].item()) + 1


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)

    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    from transformers import AutoModelForCausalLM

    print(f"Loading {model}...")
    m = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    m.eval()

    k_values = [32, 64, 128, 256]
    p_targets = [0.90, 0.95, 0.99]

    results = {"topK": {}, "topP": {}}

    for si, sample in enumerate(samples):
        ids = sample["input_ids"].to(m.device)
        attn = sample["attention_mask"].to(m.device)

        with torch.no_grad():
            logits = m(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        p = torch.softmax(logits[0], dim=-1)  # [V]

        # Sort descending and compute cumulative
        p_sorted, _ = torch.sort(p, descending=True)
        p_cumsum = torch.cumsum(p_sorted, dim=0)

        # --- Top-K: fixed K, variable coverage ---
        results["topK"][f"sample_{si}"] = {}
        for K in k_values:
            coverage = compute_topk_coverage(p_cumsum, K)
            results["topK"][f"sample_{si}"][str(K)] = round(coverage, 4)
            print(f"  sample {si+1}, K={K}: coverage={coverage:.4f}")

        # --- Top-P: fixed coverage target, variable K ---
        results["topP"][f"sample_{si}"] = {}
        for pt in p_targets:
            K_needed = compute_topk_tokens_needed(p_cumsum, pt)
            actual_cov = p_cumsum[K_needed - 1].item()
            key = str(pt).replace("0.", "p")
            results["topP"][f"sample_{si}"][key] = {
                "K_needed": K_needed,
                "actual_coverage": round(actual_cov, 4),
            }
            print(f"  sample {si+1}, p={pt}: K={K_needed}, coverage={actual_cov:.4f}")

    # --- Summary statistics ---
    summary_topK = {}
    for K in k_values:
        vals = [results["topK"][f"sample_{i}"][str(K)] for i in range(n_samples)]
        summary_topK[str(K)] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
    results["summary_topK"] = summary_topK

    summary_topP = {}
    for pt in p_targets:
        key = str(pt).replace("0.", "p")
        ks = [results["topP"][f"sample_{i}"][key]["K_needed"] for i in range(n_samples)]
        summary_topP[key] = {
            "mean_K": round(float(np.mean(ks)), 1),
            "min_K": min(ks),
            "max_K": max(ks),
            "std_K": round(float(np.std(ks)), 1),
        }
    results["summary_topP"] = summary_topP

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp10c_coverage.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {results_dir}/exp10c_coverage.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 10c: Top-K vs Top-P coverage")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
