"""Experiment: ESG (Empirical Sensitivity Gramian) visualization.

Compute ESG for selected layers of Qwen3.5-0.8B, visualize eigenvalue spectra
and top eigenvectors.
"""

import argparse
import json
import os

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.calibration import load_calibration_data
from prism.esg import compute_esg
from prism.jacobian import compute_jacobian_for_tokens


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    K=64,
    max_length=64,
    min_length=32,
    n_components=32,
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

    n_layers = m.config.num_hidden_layers
    selected_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Selected layers: {selected_layers}")

    # Build token set from first sample
    sample0 = samples[0]
    ids0 = sample0["input_ids"].to(m.device)
    attn0 = sample0["attention_mask"].to(m.device)
    with torch.no_grad():
        logits = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = logits.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    print("Computing ESG per layer...")
    all_results = {}

    for layer in selected_layers:
        jacobians = []
        for sample in samples:
            ids = sample["input_ids"]
            attn = sample["attention_mask"]
            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
            jacobians.append(J)
            print(f"  layer {layer}, sample {len(jacobians)}/{len(samples)}")

        eigenvalues, eigenvectors = compute_esg(jacobians, n_components=n_components)

        all_results[layer] = {
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors.tolist(),
        }
        top3 = [f"{v.item():.1f}" for v in eigenvalues[:3]]
        print(f"  layer {layer}: top3 eigenvalues = {top3}")

    # Save JSON
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "esg_data.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot eigenvalue spectra
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in selected_layers:
        evs = all_results[layer]["eigenvalues"]
        ax.plot(range(len(evs)), evs, marker="o", label=f"Layer {layer}")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("ESG Eigenvalue Spectra")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "esg_eigenvalues.png"), dpi=100)
    plt.close()

    # Plot top eigenvector heatmap
    fig, axes = plt.subplots(
        1, len(selected_layers), figsize=(5 * len(selected_layers), 4)
    )
    if len(selected_layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, selected_layers):
        evs = torch.tensor(all_results[layer]["eigenvectors"])
        ax.imshow(evs[:8, :32].T, aspect="auto", cmap="RdBu")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvector")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "esg_eigenvectors.png"), dpi=100)
    plt.close()

    print(
        f"Results saved to {results_dir}/esg_data.json, esg_eigenvalues.png, esg_eigenvectors.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG visualization")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
