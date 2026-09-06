"""Condition number of Gramian G_S + Tikhonov λ-sweep.

Computes κ(E_S^T @ E_S) per sample/token_set and tests robustness of
jacobian_gap under Ridge regularization:
  G_inv(λ) = (G_S + λI)^{-1}

Sweep λ ∈ {1e-8, 1e-6, 1e-4, 1e-2, 1.0}.
Identity bridge should be ~0 for all λ (since pinv already handles conditioning).
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
from prism.bridge import (
    jacobian_gap,
    _project_to_vocab,
    _valid_mask,
    _weighted_cosine_loss,
)
from prism.jacobian import compute_jacobian_for_tokens


def _ridge_inverse_gramian(E: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute (E^T @ E + λI)^{-1} via direct solve."""
    G = E.float().T @ E.float()
    I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
    return torch.linalg.inv(G + lam * I)


def jacobian_gap_ridge(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    lam: float,
    threshold: float = 1e-6,
) -> float:
    """Bridge with Ridge regularized Gramian inverse."""
    G_inv = _ridge_inverse_gramian(E_S, lam)
    J_L_proj = _project_to_vocab(J_L, E_L)
    J_S_target = J_L_proj @ E_S.float() @ G_inv
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
    lams="1e-8,1e-6,1e-4,1e-2,1.0",
    results_dir="results",
):
    torch.set_num_threads(4)
    lams = [float(x) for x in lams.split(",")]

    samples = load_calibration_data(
        model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )
    print(f"Loaded {len(samples)} samples")

    model_obj = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model_obj.eval()
    n_layers = model_obj.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_layers, n_layers)

    token_set = build_global_token_set(model_obj, samples, K=K)
    E = extract_embeddings(model_obj)

    # Compute condition number of Gramian G = E^T @ E
    G = E.float().T @ E.float()
    cond = torch.linalg.cond(G).item()
    print(f"\nCondition number κ(E^T @ E) = {cond:.2e}")

    # Eigenvalue stats of G
    evals = torch.linalg.eigvalsh(G)
    print(f"  min eigenvalue: {evals.min().item():.2e}")
    print(f"  max eigenvalue: {evals.max().item():.2e}")
    print(f"  rank(G) (eps=1e-6): {(evals.abs() > 1e-6).sum().item()} / {evals.shape[0]}")

    # Identity bridge with different λ
    lam_results = {}
    sample = samples[0]
    ids = sample["input_ids"].to(model_obj.device)
    attn = sample["attention_mask"].to(model_obj.device)
    with torch.no_grad():
        logits = model_obj(input_ids=ids, attention_mask=attn).logits[:, -1, :]
    probs = torch.softmax(logits[0, token_set], dim=-1)

    # Sample a few layers
    step = max(1, len(mapping) // 5)
    sampled = list(mapping.items())[::step]

    for lam in lams:
        losses = []
        for l_L, l_S in sampled:
            J = compute_jacobian_for_tokens(model_obj, l_L, ids, attn, token_set)
            if lam == 0.0:
                loss = jacobian_gap(J, J, E, E, probs)
            else:
                loss = jacobian_gap_ridge(J, J, E, E, probs, lam)
            losses.append(loss)
        avg = sum(losses) / len(losses)
        lam_results[f"lam_{lam}"] = {
            "per_layer": {f"{l_L}->{l_S}": loss for (l_L, l_S), loss in zip(sampled, losses)},
            "avg": avg,
        }
        print(f"  λ={lam:.0e}: identity_loss={avg:.6e}")

    # Save
    os.makedirs(results_dir, exist_ok=True)
    output = {
        "condition_number": cond,
        "eigenvalue_stats": {
            "min": evals.min().item(),
            "max": evals.max().item(),
            "rank_above_1e-6": int((evals.abs() > 1e-6).sum().item()),
        },
        "ridge_sweep": lam_results,
        "params": {
            "model": model,
            "n_samples": n_samples,
            "K": K,
            "max_length": max_length,
            "min_length": min_length,
            "lams": lams,
        },
    }
    with open(os.path.join(results_dir, "exp6b_condition_number.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {results_dir}/exp6b_condition_number.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6b: Condition number + Ridge λ-sweep")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--lams", default="1e-8,1e-6,1e-4,1e-2,1.0")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
