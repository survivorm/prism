"""Experiment 10c: Coverage of top-K for soft metric.

Compute what fraction of p-mass falls into U at |U| = 64/128/256.
Also compute effective rank: exp(H(p)) for the probability on U.
"""

import argparse
import json
import os

import torch

from prism.calibration import load_calibration_data


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    max_length=64,
    min_length=32,
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

    u_sizes = [32, 64, 128, 256]
    all_results = {}

    for si, sample in enumerate(samples):
        ids = sample["input_ids"].to(m.device)
        attn = sample["attention_mask"].to(m.device)

        with torch.no_grad():
            logits = m(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        p = torch.softmax(logits[0], dim=-1)  # [V]

        for u in u_sizes:
            top_k = logits[0].topk(u, dim=-1).indices.tolist()
            coverage = p[top_k].sum().item()

            # Effective rank of the distribution restricted to top-K
            p_top = p[top_k]
            p_top_norm = p_top / (p_top.sum() + 1e-10)
            entropy = -(p_top_norm * torch.log(p_top_norm + 1e-10)).sum().item()
            eff_rank = float(torch.exp(torch.tensor(entropy)))

            print(
                f"  sample {si + 1}, |U|={u}: coverage={coverage:.4f}, eff_rank={eff_rank:.1f}"
            )

            key = f"sample_{si}"
            if key not in all_results:
                all_results[key] = {}
            all_results[key][str(u)] = {
                "coverage": coverage,
                "eff_rank": eff_rank,
            }

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, max_length: max_length, min_length: min_length}
    with open(os.path.join(results_dir, "exp10c_coverage.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {results_dir}/exp10c_coverage.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 10c: Coverage")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
