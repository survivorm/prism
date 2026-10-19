"""Experiment 14: SNR and Participation Ratio per layer.

Per-layer quantification of ESG spectrum structure (effective dimensionality)
and Jacobian consistency across samples (SNR), linked to layer types (GDN vs full).

FIXES (v2):
  - n_samples=8 (was 4)
  - n_components=K (full spectrum, no truncation-bias)
  - Exact gauge_share: u^T G u / trace(G); proxy (ev[0]/trace) kept for compat
  - SNR: added mean_sq, var_sq, leave-one-out SNR per sample
  - Layer 23 flagged as last_layer (trivial readout Jacobian)
"""

import argparse
import json
import os

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.calibration import (
    get_layer_types,
    load_calibration_data,
)
from prism.esg import compute_esg
from prism.jacobian import compute_jacobian_for_tokens
from prism.spectral import participation_ratio, snr_e14_parts


def _snr_leave_one_out(jacobians):
    """Per-sample leave-one-out SNR."""
    N = len(jacobians)
    stacked = torch.stack(jacobians)
    total = stacked.sum(dim=0)
    result = []
    for i in range(N):
        loo_mean = (total - stacked[i]) / (N - 1)
        loo_js = [stacked[j] for j in range(N) if j != i]
        loo_mean_sq = sum(J.pow(2).sum().item() for J in loo_js) / (N - 1)
        loo_var = loo_mean_sq - loo_mean.pow(2).sum().item()
        loo_snr = loo_mean.pow(2).sum().item() / max(1e-12, loo_var)
        result.append(float(loo_snr))
    return result


def _gauge_share_exact(eigenvectors, eigenvalues, K, device=None):
    """Exact gauge_share: u^T G u / trace(G), where u = 1/sqrt(K)."""
    ones = torch.ones(K, device=device or eigenvectors.device) / torch.sqrt(torch.tensor(float(K), device=device or eigenvectors.device))
    uGu = sum(
        eigenvalues[i].item() * (eigenvectors[i] @ ones).pow(2).item()
        for i in range(len(eigenvalues))
    )
    return uGu / max(eigenvalues.sum().item(), 1e-12)


def _uniformity(eigenvectors, K, device=None):
    """cos(v_0, 1/sqrt(K))."""
    ones = torch.ones(K, device=device or eigenvectors.device) / torch.sqrt(torch.tensor(float(K), device=device or eigenvectors.device))
    return torch.abs(eigenvectors[0] @ ones).item()


def _plot_results(all_res, out_path):
    """Plot PR and SNR vs depth, colored by layer type."""
    layers = sorted(all_res.keys(), key=int)
    pr_vals = [all_res[li]["PR"] for li in layers]
    snr_vals = [all_res[li]["SNR"] for li in layers]
    types = [all_res[li]["layer_type"] for li in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2196F3" if t == "full_attention" else "#FF9800" for t in types]
    ax1.scatter(layers, pr_vals, c=colors, s=50, zorder=1)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Participation Ratio")
    ax1.set_yscale("log")
    ax1.set_title("PR by layer")
    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor="#2196F3", label="full_attention"),
        Patch(facecolor="#FF9800", label="linear_attention"),
    ]
    ax1.legend(handles=legend)

    ax2.scatter(layers, snr_vals, c=colors, s=50, zorder=1)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("SNR (log)")
    ax2.set_yscale("log")
    ax2.set_title("SNR by layer")
    ax2.legend(handles=legend)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close()


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=8,
    K=64,
    max_length=64,
    min_length=32,
    device="auto",
    results_dir="results",
    sample_layers=None,
):
    torch.manual_seed(42)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}")

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
        model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    m.eval()

    n_layers = m.config.num_hidden_layers
    layer_types = get_layer_types(m)
    n_components = K  # full spectrum, no truncation bias

    # Token set from first sample
    ids0, attn0 = (
        samples[0]["input_ids"].to(m.device),
        samples[0]["attention_mask"].to(m.device),
    )
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    token_set = sorted(lg0.topk(K, dim=-1).indices[0].tolist())

    all_results = {}
    if sample_layers:
        layers_to_run = sorted(sample_layers)
    else:
        layers_to_run = list(range(n_layers))
    print(f"Computing metrics for {len(layers_to_run)} layers...")

    for layer in layers_to_run:
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
            print(f"  layer {layer}, sample {si + 1}/{n_samples}")

        # ESG spectrum
        ev, eig = compute_esg(js, n_components=n_components)
        pr = participation_ratio(ev)
        gauge_share_proxy = ev[0].item() / ev.sum().item()
        gauge_share_exact = _gauge_share_exact(eig, ev, len(token_set), device=ev.device)
        unif = _uniformity(eig, len(token_set), device=ev.device)
        snr, mean_sq, var_sq = snr_e14_parts(js) if len(js) > 1 else (1e30, 0.0, 0.0)
        snr_loo = _snr_leave_one_out(js) if len(js) > 2 else []
        trace = ev.sum().item()

        all_results[str(layer)] = {
            "PR": pr,
            "gauge_share": gauge_share_exact,
            "gauge_share_proxy": gauge_share_proxy,
            "uniformity": unif,
            "SNR": snr,
            "snr_per_sample": snr_loo,
            "mean_sq": mean_sq,
            "var_sq": var_sq,
            "trace": trace,
            "layer_type": layer_types[layer],
            "last_layer": layer == n_layers - 1,
        }

    # Save
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "exp14_snr_pr.json")
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length, device: device, sample_layers: sample_layers}
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    png_path = os.path.join(results_dir, "exp14_snr_pr.png")
    _plot_results(all_results, png_path)

    # Summary table
    print("\n=== SNR/PR per layer ===")
    print(
        f"{'Layer':>6} {'Type':<16} {'PR':>8} {'g_exact':>8} {'g_proxy':>8} "
        f"{'unif':>7} {'SNR':>10} {'trace':>12}"
    )
    for layer_idx in layers_to_run:
        r = all_results[str(layer_idx)]
        print(
            f"{layer_idx:>6} {r['layer_type']:<16} {r['PR']:>8.1f} "
            f"{r['gauge_share']:>8.3f} {r['gauge_share_proxy']:>8.3f} "
            f"{r['uniformity']:>7.3f} {r['SNR']:>10.2f} {r['trace']:>12.1f}"
        )

    print(f"\nSaved to {json_path}, {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 14: SNR and Participation Ratio")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--sample-layers",
        default=None,
        help="Comma-separated layer indices (default: all)",
    )
    args = parser.parse_args()
    sl = args.sample_layers
    if sl is not None:
        args.sample_layers = [int(x) for x in sl.split(",")]
    main(**vars(args))
