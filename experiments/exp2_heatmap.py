"""Experiment 2: Heatmap — neighbor windows around proportional vs. end mapping.

For each anchor layer in the large model, compute bridge loss over a local
window of small layers around two candidates: proportional p(l) and end-mapping
e(l) = l - 8. Position of local minimum reveals:
  - min at e(l) → end-mapping hypothesis (tail confound);
  - min at p(l) → functional correspondence;
  - flat and high → weak signal.
"""

import argparse
import json
import os

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_bridge
from prism.jacobian import compute_jacobian_for_tokens


# Anchor layers in large model and their local windows in small model
ANCHOR_WINDOWS = {
    0: [0, 1, 2],
    8: [0, 3, 6],
    16: [8, 10, 12],
    24: [16, 17, 18],
    30: [21, 22, 23],
}


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=2,
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
    from transformers import AutoModelForCausalLM

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
    print(f"Large: {n_L} layers, Small: {n_S} layers")
    print(f"Proportional mapping sample: {list(mapping.items())[:5]}")

    # Build anchor -> {window, proportional, end_mapping}
    anchors = []
    for l_L, window in ANCHOR_WINDOWS.items():
        prop = mapping.get(l_L, None)
        end_map = max(0, l_L - 8)
        if prop is not None:
            anchors.append(
                {
                    "large_layer": l_L,
                    "window": window,
                    "proportional": prop,
                    "end_mapping": min(end_map, n_S - 1),
                }
            )

    print(f"Testing {len(anchors)} anchor layers")

    # Precompute token set and embeddings
    sample0 = samples[0]
    ids0 = sample0["input_ids"].to(model_L.device)
    attn0 = sample0["attention_mask"].to(model_L.device)
    with torch.no_grad():
        logits_L = model_L(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = logits_L.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    print("Computing bridge loss per window...")

    results = []
    for anchor in anchors:
        l_L = anchor["large_layer"]
        window = anchor["window"]
        row = {
            "large_layer": l_L,
            "losses": {},
            "proportional": anchor["proportional"],
            "end_mapping": anchor["end_mapping"],
        }

        for l_S in window:
            losses = []
            for sample in samples:
                ids = sample["input_ids"]
                attn = sample["attention_mask"]

                J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
                J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)

                with torch.no_grad():
                    out_L = model_L(
                        input_ids=ids.to(model_L.device),
                        attention_mask=attn.to(model_L.device),
                    )
                    probs = torch.softmax(out_L.logits[0, -1, token_set], dim=-1)

                loss = jacobian_bridge(J_L, J_S, E_L, E_S, probs)
                losses.append(loss)

            avg = sum(losses) / len(losses)
            row["losses"][f"small_{l_S}"] = avg
            print(f"  {l_L} -> {l_S}: {avg:.4f}")

        row["min_at"] = min(row["losses"], key=row["losses"].get)
        results.append(row)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "exp2_heatmap.json")
    all_results = {
        "params": {large_model: large_model, small_model: small_model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length},
        "rows": results,
    }
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot: scatter of bridge loss per (small_layer, large_layer)
    fig, ax = plt.subplots(figsize=(10, 5))
    xs, ys, zss = [], [], []
    for row in results:
        l_L = row["large_layer"]
        for l_S_str, val in row["losses"].items():
            l_S = int(l_S_str.removeprefix("small_"))
            xs.append(l_S)
            ys.append(l_L)
            zss.append(val)

    sc = ax.scatter(xs, ys, c=zss, cmap="YlOrRd", s=200, edgecolors="k")
    for x, y, lab in zip(xs, ys, zss, strict=True):
        ax.text(x, y, f"{lab:.2f}", ha="center", va="center", fontsize=7, color="k")

    fig.colorbar(sc, ax=ax, label="Bridge Loss")
    ax.set_xlabel("Small model layer")
    ax.set_ylabel("Large model layer")
    ax.set_title("Neighbor Windows: Bridge Loss")
    ax.set_xticks(range(n_S))
    ax.set_yticks(range(n_L, -1, -2))
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp2_heatmap.png"), dpi=100)
    plt.close()

    print(f"\nResults saved to {result_path} and {results_dir}/exp2_heatmap.png")
    print("\nMinima analysis:")
    for row in results:
        l_L = row["large_layer"]
        min_at = row["min_at"]
        prop = row["proportional"]
        end = row["end_mapping"]
        status = (
            "PROPORTIONAL"
            if min_at == f"small_{prop}"
            else "END"
            if min_at == f"small_{end}"
            else "MIDDLE"
        )
        print(f"  L{l_L}: min at {min_at} (prop={prop}, end={end}) -> {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: Heatmap with neighbor windows")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
