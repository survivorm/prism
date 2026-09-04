"""Experiment 6 (L18-SNR headline): SNR & Participation Ratio vs spectrum depth r.

This is the paper's "headline" exp6 (source `results/exp6_l18_snr.json`), which
was LOST from the repo (git/results/kernels all checked) — ρ=+2.94 was therefore
unverifiable. Mandatory re-run (audit GPU-queue item 6).

v2 spec (from audit): L18-only SNR (plus L12/L13 for context), r ∈ {4,8,16,32,64},
N=8, K=256 (token set), 0.8B only.

Method mirrors exp14 (SNR/PR on the ESG spectrum) but the *spectrum depth r* is
swept: for each r we take the top-r eigenvalues of the full (n_tokens×n_tokens)
ESG matrix and report SNR(r) and PR(r). PR is monotone non-decreasing in r; SNR
is r-invariant (mean-Jac statistic) — reported so the headline curve is honest
about which quantity actually moves with r.
"""

import argparse
import json
import os

import torch

from prism.calibration import (
    get_layer_types,
    load_calibration_data,
)
from prism.esg import compute_esg
from prism.jacobian import compute_jacobian_for_tokens


def _participation_ratio(eigenvalues):
    """PR = (sum lambda)^2 / sum lambda^2, effective rank in [1, len]."""
    s = eigenvalues.sum()
    return (s * s / eigenvalues.pow(2).sum()).item()


def _snr_sample(jacobians):
    """SNR = ||mean J||_F^2 / max(eps, mean||J||^2_F - ||mean J||^2_F)."""
    M = torch.stack(jacobians).mean(dim=0)
    mean_sq = sum(J.pow(2).sum().item() for J in jacobians) / len(jacobians)
    var = mean_sq - M.pow(2).sum().item()
    return M.pow(2).sum().item() / max(1e-12, var), mean_sq, var


def _gauge_share_top_r(eigenvectors, eigenvalues, r, device=None):
    """gauge_share over top-r modes: u^T G u / trace(G_top_r), u = 1/sqrt(r)."""
    dev = device or eigenvectors.device
    ones = torch.ones(r, device=dev) / torch.sqrt(torch.tensor(float(r), device=dev))
    uGu = sum(
        eigenvalues[i].item() * (eigenvectors[i] @ ones).pow(2).item()
        for i in range(min(r, len(eigenvalues)))
    )
    return uGu / max(eigenvalues[:r].sum().item(), 1e-12)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=8,
    K=256,
    max_length=64,
    min_length=32,
    device="auto",
    results_dir="results",
    layers=(18,),
    r_values=(4, 8, 16, 32, 64),
):
    torch.manual_seed(42)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    print(f"Loading {model}...")
    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    m.eval()

    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )

    n_layers = m.config.num_hidden_layers
    layer_types = get_layer_types(m)

    # Token set: top-K final-logit tokens of the first sample (K=256 headline).
    ids0, attn0 = (
        samples[0]["input_ids"].to(m.device),
        samples[0]["attention_mask"].to(m.device),
    )
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    token_set = sorted(lg0.topk(K, dim=-1).indices[0].tolist())
    print(f"Token set size |U| = {len(token_set)}")

    per_layer = {}
    for layer in layers:
        js = []
        for si, s in enumerate(samples):
            J = compute_jacobian_for_tokens(
                m,
                layer,
                s["input_ids"].to(m.device),
                s["attention_mask"].to(m.device),
                token_set,
            )
            js.append(J)
            print(f"  L{layer} sample {si + 1}/{n_samples}")

        n_tok = len(token_set)
        ev, eig = compute_esg(js, n_components=n_tok)  # full spectrum

        # SNR is r-invariant (mean-Jac statistic) — compute once.
        snr, mean_sq, var_sq = (
            _snr_sample(js) if len(js) > 1 else (1e30, 0.0, 0.0)
        )

        by_r = {}
        for r in r_values:
            rr = min(r, n_tok)
            top_r = ev[:rr]
            by_r[str(r)] = {
                "SNR": snr,
                "PR": _participation_ratio(top_r),
                "gauge_share": _gauge_share_top_r(eig, ev, rr, device=ev.device),
                "trace_top_r": top_r.sum().item(),
            }

        per_layer[str(layer)] = {
            "layer_type": layer_types[layer],
            "last_layer": layer == n_layers - 1,
            "n_samples": n_samples,
            "K": K,
            "snr_mean_sq": mean_sq,
            "snr_var_sq": var_sq,
            "by_r": by_r,
        }
        print(
            f"L{layer} ({layer_types[layer]}): "
            + "  ".join(f"r={r} PR={by_r[str(r)]['PR']:.1f}" for r in r_values)
        )

    all_results = {
        "layers": per_layer,
        "params": {
            model: model,
            "n_samples": n_samples,
            "K": K,
            "max_length": max_length,
            "min_length": min_length,
            "device": device,
            "layers": list(layers),
            "r_values": list(r_values),
        },
    }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "exp6_l18_snr.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {json_path}")

    # Headline summary
    for layer in layers:
        d = per_layer[str(layer)]
        print(
            f"L{layer}: SNR={d['by_r'][str(r_values[0])]['SNR']:.2f} "
            f"PR(4→{r_values[-1]})="
            f"{d['by_r'][str(r_values[0])]['PR']:.1f}→{d['by_r'][str(r_values[-1])]['PR']:.1f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 6 (L18): SNR & PR vs spectrum depth r")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--layers", default="18", help="Comma-separated layer indices")
    parser.add_argument("--r-values", default="4,8,16,32,64")
    args = parser.parse_args()
    main(
        model=args.model,
        n_samples=args.n_samples,
        K=args.K,
        max_length=args.max_length,
        min_length=args.min_length,
        device=args.device,
        results_dir=args.results_dir,
        layers=tuple(int(x) for x in args.layers.split(",")),
        r_values=tuple(int(x) for x in args.r_values.split(",")),
    )
