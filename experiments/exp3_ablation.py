"""Experiment 3: Ablation — without Gramian correction.

Compare identity loss WITH pinv Gramian (should be ~0)
vs WITHOUT (should be ~0.5-0.8 for real embeddings).
"""

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
from prism.bridge import (
    _project_to_vocab,
    _valid_mask,
    _weighted_cosine_loss,
)
from prism.bridge import jacobian_bridge


def jacobian_bridge_raw(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Bridge WITHOUT Gramian correction — raw J @ E^T @ E."""
    J_L_proj = _project_to_vocab(J_L, E_L)
    J_S_target = J_L_proj @ E_S.float()
    mask = _valid_mask(J_S, J_S_target, threshold)
    if mask.sum() == 0:
        return 1.0
    return _weighted_cosine_loss(J_S, J_S_target, probs_L, mask)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    K=8,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    print("Loading calibration data...")
    samples = load_calibration_data(
        model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )
    print(f"Loaded {len(samples)} samples")

    print("Loading model...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_obj.eval()
    n_layers = model_obj.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_layers, n_layers)

    token_set = build_global_token_set(model_obj, samples, K=K)
    E = extract_embeddings(model_obj)

    results_corrected = {}
    results_raw = {}

    print("Running ablation (corrected vs raw)...")

    for l_L, l_S in mapping.items():
        losses_c, losses_r = [], []
        for sample in samples:
            ids = sample["input_ids"].to(model_obj.device)
            attn = sample["attention_mask"].to(model_obj.device)
            J = compute_jacobian_for_tokens(model_obj, l_L, ids, attn, token_set)

            with torch.no_grad():
                logits = model_obj(input_ids=ids).logits[:, -1, :]
                probs = torch.softmax(logits[0, token_set], dim=-1)

            l_c = jacobian_bridge(J, J, E, E, probs)
            l_r = jacobian_bridge_raw(J, J, E, E, probs)
            losses_c.append(l_c)
            losses_r.append(l_r)

        results_corrected[f"{l_L}->{l_S}"] = sum(losses_c) / len(losses_c)
        results_raw[f"{l_L}->{l_S}"] = sum(losses_r) / len(losses_r)

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp3_ablation.json"), "w") as f:
        json.dump(
            {
                "with_correction": results_corrected,
                "without_correction": results_raw,
            },
            f,
            indent=2,
        )

    print(
        f"\nMean corrected:  {sum(results_corrected.values()) / len(results_corrected):.8f}"
    )
    print(f"Mean without:    {sum(results_raw.values()) / len(results_raw):.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Ablation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
