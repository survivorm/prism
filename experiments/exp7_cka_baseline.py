"""Experiment 7: CKA (Centered Kernel Alignment) baseline between hidden states.

Computes CKA between hidden states of 4B and 0.8B models across multiple samples,
then compares with Jacobian gap (1-cos) on the same layer pairs.
"""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_gap
from prism.jacobian import compute_jacobian_for_tokens


def cka_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """Linear CKA between two sets of vectors [N, D].

    CKA(X,Y) = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    where HSIC(X,Y) = ||center(K_XY)||_F^2
    """
    # Linear kernels
    K_XX = X @ X.T  # [N, N]
    K_YY = Y @ Y.T
    K_XY = X @ Y.T

    # Center kernels
    N = X.shape[0]
    one_N = torch.ones(N, N, device=X.device) / N
    center = torch.eye(N, device=X.device) - one_N

    K_XX_c = center @ K_XX @ center
    K_YY_c = center @ K_YY @ center
    K_XY_c = center @ K_XY @ center

    # HSIC norms
    hsic_xx = K_XX_c.pow(2).sum()
    hsic_yy = K_YY_c.pow(2).sum()
    hsic_xy = K_XY_c.pow(2).sum()

    denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-12))
    return (hsic_xy / denom).item()


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=5,
    K=8,
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
    print(f"Loaded {len(samples)} samples")

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

    # Sample layers (every N-th) to keep runtime manageable
    step = max(1, len(mapping) // 10)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs")

    # Collect hidden states for CKA across all samples
    print("Collecting hidden states for CKA...")
    cka_data = {}

    for l_L, l_S in sampled:
        h_L_list: list[torch.Tensor] = []
        h_S_list: list[torch.Tensor] = []

        for sample in samples:
            ids = sample["input_ids"]
            attn = sample["attention_mask"]

            with torch.no_grad():
                out_L = model_L(
                    input_ids=ids.to(model_L.device),
                    attention_mask=attn.to(model_L.device),
                    output_hidden_states=True,
                )
                out_S = model_S(
                    input_ids=ids.to(model_S.device),
                    attention_mask=attn.to(model_S.device),
                    output_hidden_states=True,
                )

            # Last token hidden state per sample
            h_L_list.append(out_L.hidden_states[l_L + 1][:, -1, :])
            h_S_list.append(out_S.hidden_states[l_S + 1][:, -1, :])

        H_L = torch.cat(h_L_list, dim=0)  # [N, D_L]
        H_S = torch.cat(h_S_list, dim=0)  # [N, D_S]

        cka = cka_similarity(H_L.float(), H_S.float())
        cka_data[f"{l_L}->{l_S}"] = cka
        print(f"  {l_L}->{l_S}: CKA={cka:.6f}")

    # Jacobian gap on same layers (single sample)
    print("\nComputing Jacobian gap...")
    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    sample = samples[0]
    ids = sample["input_ids"].to(model_L.device)
    attn = sample["attention_mask"].to(model_L.device)

    with torch.no_grad():
        logits_L = model_L(input_ids=ids, attention_mask=attn).logits[:, -1, :]
    top_k = logits_L.topk(K, dim=-1).indices[0].tolist()
    probs = torch.softmax(logits_L[0, top_k], dim=-1)
    token_set = sorted(top_k)

    gap_data = {}

    for l_L, l_S in sampled:
        J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
        J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)
        loss = jacobian_gap(J_L, J_S, E_L, E_S, probs)
        gap_data[f"{l_L}->{l_S}"] = loss
        print(f"  {l_L}->{l_S}  : gap={loss:.6f}")

    # Save results + correlation
    os.makedirs(results_dir, exist_ok=True)
    import scipy

    cka_vals = list(cka_data.values())
    gap_vals = list(gap_data.values())
    corr = scipy.stats.spearmanr(cka_vals, gap_vals).correlation

    output = {
        "cka": cka_data,
        "gap": gap_data,
        "spearman_corr": corr,
        "n_samples": len(samples),
        "params": {
            "large_model": large_model,
            "small_model": small_model,
            "n_samples": n_samples,
            "K": K,
            "max_length": max_length,
            "min_length": min_length,
        },
    }
    with open(os.path.join(results_dir, "exp7_cka_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSpearman correlation (CKA vs gap): {corr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 7: CKA baseline")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
