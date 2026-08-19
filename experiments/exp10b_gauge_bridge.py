"""Experiment 10b: Gauge-free bridge 4B→0.8B.

Compare bridge curves on J_raw, J_centered, J_soft.
Same setup as exp4 (|U|=128, sampled layer pairs).
"""

import argparse
import json
import os

import torch

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_bridge
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
    uniform_centered,
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

    print(f"Loading {n_samples} calibration samples...")
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

    mapping = uniform_layer_mapping(
        model_L.config.num_hidden_layers, model_S.config.num_hidden_layers
    )
    step = max(1, len(mapping) // 6)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs")

    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    sample = samples[0]
    ids = sample["input_ids"]
    attn = sample["attention_mask"]

    with torch.no_grad():
        logits_L = model_L(
            input_ids=ids.to(model_L.device), attention_mask=attn.to(model_L.device)
        ).logits[:, -1, :]

    u_size = 128
    top_k = logits_L.topk(u_size, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    with torch.no_grad():
        p_L = torch.softmax(logits_L[0, token_set], dim=-1)
        logits_S = model_S(
            input_ids=ids.to(model_S.device), attention_mask=attn.to(model_S.device)
        ).logits[:, -1, :]
        p_S = torch.softmax(logits_S[0, token_set], dim=-1)

    variants = {"raw": [], "centered": [], "softmax": []}

    for l_L, l_S in sampled:
        J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
        J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)

        # Raw
        loss_raw = jacobian_bridge(J_L, J_S, E_L, E_S, p_L)

        # Centered
        loss_cent = jacobian_bridge(
            uniform_centered(J_L), uniform_centered(J_S), E_L, E_S, p_L
        )

        # Softmax-weighted
        loss_soft = jacobian_bridge(
            softmax_jacobian(J_L, p_L),
            softmax_jacobian(J_S, p_S),
            E_L,
            E_S,
            p_L,
        )

        key = f"{l_L}->{l_S}"
        variants["raw"].append({"layer": key, "loss": loss_raw})
        variants["centered"].append({"layer": key, "loss": loss_cent})
        variants["softmax"].append({"layer": key, "loss": loss_soft})

        print(
            f"  {key}: raw={loss_raw:.4f}, cent={loss_cent:.4f}, soft={loss_soft:.4f}"
        )

    results = {
        "raw": {
            "per_layer": variants["raw"],
            "avg": sum(v["loss"] for v in variants["raw"]) / len(variants["raw"]),
        },
        "centered": {
            "per_layer": variants["centered"],
            "avg": sum(v["loss"] for v in variants["centered"])
            / len(variants["centered"]),
        },
        "softmax": {
            "per_layer": variants["softmax"],
            "avg": sum(v["loss"] for v in variants["softmax"])
            / len(variants["softmax"]),
        },
    }

    os.makedirs(results_dir, exist_ok=True)
    results["params"] = {large_model: large_model, small_model: small_model, n_samples: n_samples, max_length: max_length, min_length: min_length}
    with open(os.path.join(results_dir, "exp10b_gauge_bridge.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {results_dir}/exp10b_gauge_bridge.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 10b: Gauge-free bridge")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
