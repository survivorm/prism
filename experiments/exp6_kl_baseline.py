"""Experiment 6: KL-divergence baseline vs Jacobian bridge.

For each layer pair, compute:
1. Bridge loss: J_bridge(l_L → l_S)
2. KL divergence between projected logit distributions from each layer

Correlation analysis: if bridge correlates with KL → bridge captures nothing
beyond logits. If not → bridge captures something structural beyond logits.
"""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_gap
from prism.jacobian import compute_jacobian_for_tokens


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) from logit vectors."""
    p = torch.nn.functional.log_softmax(p_logits, dim=-1)
    q = torch.nn.functional.softmax(q_logits, dim=-1)
    return (p * q).sum().item()


def logits_from_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    """Project hidden state through norm + lm_head to get logits."""
    text_model = model.model
    h_norm = text_model.norm(hidden.float())
    return model.lm_head(h_norm)[:, -1, :]


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=3,
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

    # Collect KL and bridge for each layer pair across samples
    kl_all: dict[str, list[float]] = {}
    bridge_all: dict[str, list[float]] = {}

    # Precompute token set on a single sample
    sample0 = samples[0]
    ids0 = sample0["input_ids"].to(model_L.device)
    attn0 = sample0["attention_mask"].to(model_L.device)
    with torch.no_grad():
        logits_L = model_L(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = logits_L.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    print("Computing KL + bridge per layer...")

    for l_L, l_S in sampled:
        kl_pair = []
        bridge_pair = []

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

            h_L = out_L.hidden_states[l_L + 1]
            h_S = out_S.hidden_states[l_S + 1]

            # KL between projected logits
            logits_L = logits_from_hidden(model_L, h_L)
            logits_S = logits_from_hidden(model_S, h_S)
            kl = kl_divergence(logits_L[0], logits_S[0])
            kl_pair.append(kl)

            # Bridge loss
            J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
            J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)

            with torch.no_grad():
                probs = torch.softmax(logits_L[0, token_set], dim=-1)

            loss = jacobian_gap(J_L, J_S, E_L, E_S, probs)
            bridge_pair.append(loss)

        kl_all[f"{l_L}->{l_S}"] = kl_pair
        bridge_all[f"{l_L}->{l_S}"] = bridge_pair
        print(
            f"  {l_L}->{l_S}: KL={sum(kl_pair) / len(kl_pair):.4f}, bridge={sum(bridge_pair) / len(bridge_pair):.4f}"
        )

    # Correlation analysis
    kl_means = [sum(v) / len(v) for v in kl_all.values()]
    gap_means = [sum(v) / len(v) for v in gap_all.values()]

    import scipy

    corr = scipy.stats.spearmanr(kl_means, gap_means).correlation

    os.makedirs(results_dir, exist_ok=True)
    output = {
        "kl_per_layer": kl_all,
        "gap_per_layer": bridge_all,
        "kl_means": kl_means,
        "gap_means": bridge_means,
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
    with open(os.path.join(results_dir, "exp6_kl_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSpearman correlation (KL vs bridge): {corr:.4f}")
    if abs(corr) < 0.5:
        print("→ Bridge captures something beyond logits (low correlation)")
    else:
        print("→ Bridge correlates with KL (may be redundant)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6: KL baseline")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
