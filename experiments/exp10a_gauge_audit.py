"""Experiment 10a: Gauge audit of ESG.

Compare raw, uniform-centered, and softmax-weighted Gramians.
Metrics: uniformity_0 (cos(v_0, 1/sqrt(K))), gauge_share_0, trace, top-5 eigenvalues.
"""

import argparse
import json
import os

import torch

from prism.calibration import load_calibration_data
from prism.esg import compute_esg
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
    uniform_centered,
)


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

    # Build token set from first sample
    sample0 = samples[0]
    ids0 = sample0["input_ids"].to(m.device)
    attn0 = sample0["attention_mask"].to(m.device)
    with torch.no_grad():
        logits0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = logits0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    all_results = {}

    for layer in selected:
        raw_js: list[torch.Tensor] = []
        cent_js: list[torch.Tensor] = []
        soft_js: list[torch.Tensor] = []

        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]
            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)

            # Softmax probabilities for this sample
            with torch.no_grad():
                lg = m(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            p = torch.softmax(lg[0, token_set], dim=-1)

            raw_js.append(J)
            cent_js.append(uniform_centered(J))
            soft_js.append(softmax_jacobian(J, p))
            print(f"  layer {layer}, sample {si + 1}/{n_samples}")

        # Three Gramians
        ev_raw, v_raw = compute_esg(raw_js, n_components)
        ev_cent, _ = compute_esg(cent_js, n_components)
        ev_soft, _ = compute_esg(soft_js, n_components)

        K_val = len(token_set)
        ones = torch.ones(K_val) / torch.sqrt(torch.tensor(float(K_val)))
        uniformity_0 = torch.abs((v_raw[0] @ ones)).item()
        gauge_share_0 = 1.0 - (ev_cent[0] / ev_raw[0]).item()

        all_results[layer] = {
            "uniformity_0": uniformity_0,
            "gauge_share_0": gauge_share_0,
            "raw": {
                "top5": ev_raw[:5].tolist(),
                "trace": ev_raw.sum().item(),
            },
            "centered": {
                "top5": ev_cent[:5].tolist(),
                "trace": ev_cent.sum().item(),
            },
            "softmax": {
                "top5": ev_soft[:5].tolist(),
                "trace": ev_soft.sum().item(),
            },
        }

        print(
            f"  layer {layer}: "
            f"unif={uniformity_0:.3f}, gauge_share={gauge_share_0:.3f}, "
            f"raw_top3=[{ev_raw[:3].tolist()}], "
            f"cent_top3=[{ev_cent[:3].tolist()}], "
            f"soft_top3=[{ev_soft[:3].tolist()}]"
        )

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length, n_components: n_components}
    with open(os.path.join(results_dir, "exp10a_gauge_audit.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {results_dir}/exp10a_gauge_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 10a: Gauge audit")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
