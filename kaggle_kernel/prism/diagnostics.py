"""Diagnostics: per-layer bridge loss heatmaps and baselines."""

import json
import os
from typing import Any

import matplotlib
import torch

from prism.bridge import jacobian_bridge
from prism.calibration import (
    extract_embeddings,
    uniform_layer_mapping,
)
from prism.jacobian import compute_jacobian_for_tokens

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_layer_diagnostics(
    model_large,
    model_small,
    token_set,
    layer_mapping,
    samples,
    results_dir="results",
    exp_name="exp2_heatmap",
) -> dict[str, float]:
    """Compute bridge loss for each layer pair and save results."""
    E_L, _ = extract_embeddings(model_large)
    E_S, _ = extract_embeddings(model_small)
    results: dict[str, float] = {}

    for l_L, l_S in layer_mapping.items():
        losses = []
        for sample in samples:
            ids = sample["input_ids"].to(model_large.device)
            mask = sample["attention_mask"].to(model_large.device)

            J_L = compute_jacobian_for_tokens(model_large, l_L, ids, mask, token_set)
            J_S = compute_jacobian_for_tokens(model_small, l_S, ids, mask, token_set)

            with torch.no_grad():
                logits_L = model_large(input_ids=ids).logits[:, -1, :]
                probs = torch.softmax(logits_L[0, token_set], dim=-1)

            loss = jacobian_bridge(J_L, J_S, E_L, E_S, probs)
            losses.append(loss)

        avg_loss = sum(losses) / len(losses)
        results[f"{l_L}->{l_S}"] = avg_loss

    _save_results(results, exp_name, results_dir)
    return results


def run_identity_test(
    model,
    token_set,
    samples,
    n_layers: int,
    results_dir="results",
) -> dict[str, float]:
    """Identity test: same model as large and small."""
    mapping = uniform_layer_mapping(n_layers, n_layers)
    results = run_layer_diagnostics(
        model, model, token_set, mapping, samples, results_dir, "exp1_identity"
    )
    return results


def run_heatmap(
    model_large,
    model_small,
    token_set,
    layer_mapping,
    samples,
    results_dir="results",
) -> dict[str, float]:
    """Run cross-model heatmap (8B -> 0.8B)."""
    results = run_layer_diagnostics(
        model_large,
        model_small,
        token_set,
        layer_mapping,
        samples,
        results_dir,
        "exp2_heatmap",
    )
    _plot_heatmap(results, layer_mapping, results_dir)
    return results


def _save_results(results: dict[str, float], exp_name: str, results_dir: str) -> None:
    """Save results to JSON."""
    # ponytail: fail-fast is intentional for experiments
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{exp_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _plot_heatmap(
    results: dict[str, Any],
    layer_mapping: dict[int, int],
    results_dir: str,
) -> None:
    """Save heatmap as PNG."""
    l_L_vals = sorted(layer_mapping.keys())
    l_S_vals = [layer_mapping[li] for li in l_L_vals]
    losses = [
        results.get(f"{li}->{layer_mapping[li]}", float("nan")) for li in l_L_vals
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow([losses], aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(l_L_vals)))
    ax.set_xticklabels(
        [f"{li}->{si}" for li, si in zip(l_L_vals, l_S_vals, strict=True)]
    )
    ax.set_yticks([0])
    ax.set_yticklabels(["L_bridge"])
    fig.colorbar(im, ax=ax)
    ax.set_title("Jacobian Bridge Loss (8B -> 0.8B)")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp2_heatmap.png"))
    plt.close(fig)
