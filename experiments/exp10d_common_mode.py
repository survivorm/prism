"""Experiment 10d: Common mode as meta-signal (optional).

Correlate common-mode coordinate c_t = v_0^T h_l with entropy H(p_t)
across positions/samples of calibration data, per layer.
"""

import argparse
import json
import os

import torch
from scipy.stats import spearmanr

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

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    print(f"Loading {model}...")
    m = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    m.eval()

    n_layers = m.config.num_hidden_layers
    selected = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Selected layers: {selected}")

    # Build token set
    sample0 = samples[0]
    ids0 = sample0["input_ids"].to(m.device)
    attn0 = sample0["attention_mask"].to(m.device)
    with torch.no_grad():
        logits0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = logits0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    # Compute ESG to get v_0 for each layer
    esg_cache: dict[int, torch.Tensor] = {}
    for layer in selected:
        js = []
        for sample in samples:
            J = compute_jacobian_for_tokens(
                m, layer, sample["input_ids"], sample["attention_mask"], token_set
            )
            js.append(J)
            print(f"  ESG layer {layer}, sample {len(js)}/{len(samples)}")
        _, v_raw = compute_esg(js, n_components)
        esg_cache[layer] = v_raw[0]  # top eigenvector [K]

    all_results = {}

    for layer in selected:
        v0 = esg_cache[layer]  # [K]
        c_values = []
        h_values = []

        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]
            S = ids.shape[1]

            # Forward pass with hidden states
            with torch.no_grad():
                out = m(input_ids=ids, attention_mask=attn, output_hidden_states=True)
                hidden = out.hidden_states[layer + 1]  # [1, S, D]

            # Project hidden states through v0
            # J[h] @ v0 gives common-mode coordinate
            # But we need J for each position — compute Jacobian at each position
            # Simplification: use last-token Jacobian only per sample
            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
            # v0^T @ J: common-mode direction in hidden space [D]
            c_t = (v0 @ J).norm().item()  # magnitude of common-mode component

            # Entropy of probability distribution over token_set
            with torch.no_grad():
                lg = out.logits[:, -1, :]
            p = torch.softmax(lg[0, token_set], dim=-1)
            H = -(p * torch.log(p + 1e-10)).sum().item()

            c_values.append(c_t)
            h_values.append(H)

        if len(c_values) >= 3:
            corr, _ = spearmanr(c_values, h_values)
        else:
            corr = float("nan")

        all_results[layer] = {
            "c_values": c_values,
            "h_values": h_values,
            "spearman_corr": corr,
        }
        print(f"  layer {layer}: corr={corr:.3f}, c={c_values}, H={h_values}")

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length, n_components: n_components}
    with open(os.path.join(results_dir, "exp10d_common_mode.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {results_dir}/exp10d_common_mode.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 10d: Common mode as meta-signal")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
