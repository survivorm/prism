"""Experiment 8: SVD ablation — truncated projection vs full."""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    build_global_token_set,
    extract_embeddings,
    load_calibration_data,
)
from prism.jacobian import compute_jacobian_for_tokens


def jacobian_bridge_truncated_svd(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    n_components: int,
    threshold: float = 1e-6,
) -> float:
    """Bridge with truncated SVD of the Gramian (top-k components).

    Instead of full G_inv = (E^T @ E)^{-1}, use SVD of G:
    G = V @ diag(s^2) @ Vt, truncated G_inv_k = V_k @ diag(1/s_k^2) @ Vt_k.
    """
    G = E_S.float().T @ E_S.float()  # [D, D]
    eigenvalues, eigenvectors = torch.linalg.eigh(G)

    # eigh returns ascending, take top-k
    k = min(n_components, len(eigenvalues))
    idx = torch.arange(len(eigenvalues) - k, len(eigenvalues))
    eig_k = eigenvalues[idx]
    V_k = eigenvectors[:, idx]  # [D, k]

    # Truncated inverse: V_k @ diag(1/s_k^2) @ Vt_k
    D_inv_k = 1.0 / (eig_k + 1e-12)
    G_inv_k = (V_k * D_inv_k) @ V_k.T  # [D, D]

    J_proj = J_L.float() @ E_L.float().T  # [|U|, V]
    J_target = J_proj @ E_S.float() @ G_inv_k

    from prism.bridge import _valid_mask, _weighted_cosine_loss

    mask = _valid_mask(J_S, J_target, threshold)
    if mask.sum() == 0:
        return 1.0
    return _weighted_cosine_loss(J_S, J_target, probs_L, mask)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    K=8,
    max_length=64,
    min_length=32,
    n_components=(16, 64, 256, 512, 1024),
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
    token_set = build_global_token_set(model_obj, samples, K=K)
    E = extract_embeddings(model_obj)

    results = {}

    for nc in n_components:
        layer_losses = {}
        ids = samples[0]["input_ids"].to(model_obj.device)
        attn = samples[0]["attention_mask"].to(model_obj.device)
        J = compute_jacobian_for_tokens(model_obj, 0, ids, attn, token_set)

        with torch.no_grad():
            logits = model_obj(input_ids=ids).logits[:, -1, :]
            probs = torch.softmax(logits[0, token_set], dim=-1)

        loss = jacobian_bridge_truncated_svd(J, J, E, E, probs, nc)
        layer_losses["layer_0"] = loss
        results[f"n_components={nc}"] = layer_losses
        print(f"n_components={nc}: loss={loss:.8f}")

    os.makedirs(results_dir, exist_ok=True)
    results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length, n_components: n_components}
    with open(os.path.join(results_dir, "exp8_svd_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 8: SVD ablation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(
        model=args.model,
        n_samples=args.n_samples,
        K=args.K,
        max_length=args.max_length,
        min_length=args.min_length,
        results_dir=args.results_dir,
    )
