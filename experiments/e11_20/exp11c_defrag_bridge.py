"""Experiment 11C: Does defrag preserve bridge signal?

4B→0.8B, bridge curve vs depth. Compare:
  - raw-J (from exp4),
  - softmax-J (from exp10b),
  - defrag-r at r=16 (reconstructed J^(p) → jacobian_gap).

Metric: Spearman between curves; monotonicity (early > late).
"""

import argparse
import json
import os

import torch
from scipy.stats import spearmanr

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_gap
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    torch.manual_seed(42)

    print(f"Loading {n_samples} samples...")
    samples = load_calibration_data(
        large_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

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
    step = max(1, len(mapping) // 7)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs")

    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    sample = samples[0]
    ids = sample["input_ids"]
    attn = sample["attention_mask"]

    with torch.no_grad():
        logits_L = model_L(
            input_ids=ids.to(model_L.device),
            attention_mask=attn.to(model_L.device),
        ).logits[:, -1, :]
        logits_S = model_S(
            input_ids=ids.to(model_S.device),
            attention_mask=attn.to(model_S.device),
        ).logits[:, -1, :]

    u_size = 256
    top_k = logits_L.topk(u_size, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    with torch.no_grad():
        p_L = torch.softmax(logits_L[0, token_set], dim=-1)
        p_S = torch.softmax(logits_S[0, token_set], dim=-1)

    rs = [8, 16, 32]
    results = {
        "raw": {"per_layer": [], "avg": 0},
        "softmax": {"per_layer": [], "avg": 0},
    }
    for r in rs:
        results[f"defrag_{r}"] = {"per_layer": [], "avg": 0}

    for l_L, l_S in sampled:
        J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
        J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)

        # Raw
        loss_raw = jacobian_gap(J_L, J_S, E_L, E_S, p_L)

        # Softmax gauge-free
        J_L_p = softmax_jacobian(J_L, p_L)
        J_S_p = softmax_jacobian(J_S, p_S)
        loss_soft = jacobian_gap(J_L_p, J_S_p, E_L, E_S, p_L)

        # Defrag reconstruction
        for r in rs:
            # Build basis from J_L_p and J_S_p separately, then reconstruct
            G_L = J_L_p @ J_L_p.T
            _, eig_L = torch.linalg.eigh(G_L)
            idx_L = eig_L[0].argsort(descending=True)
            U_L = eig_L[:, idx_L][:, :r]  # [K, r]

            G_S = J_S_p @ J_S_p.T
            _, eig_S = torch.linalg.eigh(G_S)
            idx_S = eig_S[0].argsort(descending=True)
            U_S = eig_S[:, idx_S][:, :r]

            # Reconstructed J^(p) for each side
            J_L_tilde = U_L @ (U_L.T @ J_L_p)  # [K, D]
            J_S_tilde = U_S @ (U_S.T @ J_S_p)  # [K, D]

            loss_dr = jacobian_gap(J_L_tilde, J_S_tilde, E_L, E_S, p_L)
            results[f"defrag_{r}"]["per_layer"].append(
                {"layer": f"{l_L}->{l_S}",  "gap": loss_dr}
            )

        results["raw"]["per_layer"].append({"layer": f"{l_L}->{l_S}",  "gap": loss_raw})
        results["softmax"]["per_layer"].append(
            {"layer": f"{l_L}->{l_S}",  "gap": loss_soft}
        )

        print(
            f"  {l_L}->{l_S}: raw={loss_raw:.4f}, soft={loss_soft:.4f}, "
            + ", ".join(
                f"dr{r}={results[f'defrag_{r}']['per_layer'][-1]['loss']:.4f}"
                for r in rs
            )
        )

    # Compute averages
    for key in results:
        results[key]["avg"] = sum(v["loss"] for v in results[key]["per_layer"]) / len(
            results[key]["per_layer"]
        )

    # Spearman correlations
    raw_losses = [v["loss"] for v in results["raw"]["per_layer"]]
    print("\nSpearman correlations (vs raw):")
    for variant in ["softmax"] + [f"defrag_{r}" for r in rs]:
        var_losses = [v["loss"] for v in results[variant]["per_layer"]]
        rho, _ = spearmanr(raw_losses, var_losses)
        print(f"  raw vs {variant}: ρ = {rho:.4f}")
        results[variant]["spearman_vs_raw"] = rho

    # Monotonicity check (should decrease with depth)
    for variant in ["raw", "softmax"] + [f"defrag_{r}" for r in rs]:
        losses = [v["loss"] for v in results[variant]["per_layer"]]
        mon_rho, _ = spearmanr(range(len(losses)), losses)
        print(f"  {variant} monotonicity (depth vs loss): ρ = {mon_rho:.4f}")
        results[variant]["monotonicity"] = mon_rho

    os.makedirs(results_dir, exist_ok=True)
    results["params"] = {
        "large_model": large_model,
        "small_model": small_model,
        "n_samples": n_samples,
        "max_length": max_length,
        "min_length": min_length,
    }
    with open(os.path.join(results_dir, "exp11c_defrag_bridge.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp11c_defrag_bridge.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 11C: Defrag bridge signal")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
