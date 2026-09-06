"""Experiment 12C: Bridge original ↔ quantized.

Softmax bridge between original model and each quantization scheme.
Prediction: uniform scheme raises the bridge (damaged differential directions),
map-based scheme does not.
"""

import argparse
import json
import os

import torch

from prism.calibration import extract_embeddings, load_calibration_data
from prism.bridge import jacobian_gap
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)


def quantize_vector(
    x: torch.Tensor,
    directions: torch.Tensor,
    bits_per_dir: list[int],
    n_bits_rest: int,
) -> torch.Tensor:
    """Quantize x along directions with specified bits."""
    assert directions.shape[1] == len(bits_per_dir)
    coeffs = directions.T @ x

    max_val = x.abs().max()
    q_coeffs = []
    for c, bits in zip(coeffs, bits_per_dir, strict=False):
        step = 2 * max_val / (2**bits)
        q = torch.round(c / step) * step
        q = torch.clamp(q, -max_val, max_val)
        q_coeffs.append(q)

    recon = directions @ torch.stack(q_coeffs)
    residual = x - recon
    step_r = 2 * residual.abs().max().clamp(min=1e-6) / (2**n_bits_rest)
    q_residual = torch.round(residual / step_r) * step_r
    return recon + q_residual


def uniform_quantize(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    max_val = x.abs().max()
    step = 2 * max_val / (2**n_bits)
    q = torch.round(x / step) * step
    return torch.clamp(q, -max_val, max_val)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    K=64,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    torch.manual_seed(42)

    samples = load_calibration_data(model, max_length, min_length, n_samples)

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    selected = [6, 12, 18]
    E = extract_embeddings(m)

    s0 = samples[0]
    ids0 = s0["input_ids"].to(m.device)
    attn0 = s0["attention_mask"].to(m.device)
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = lg0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    n_diff = 5
    bits_gauge = 2
    bits_diff = 8
    n_bits_uniform = 4
    n_bits_rest = 6

    all_results = {}

    for layer in selected:
        print(f"\nLayer {layer}:")
        bridges_orig_map = []
        bridges_orig_uniform = []
        bridges_map_uniform = []

        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]

            # Jacobian of original
            J_orig = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
            p_tokens = torch.softmax(
                m(input_ids=ids, attention_mask=attn).logits[0, -1, token_set], dim=-1
            )

            # J^(p) for original
            J_orig_p = softmax_jacobian(J_orig, p_tokens)

            # Build sensitivity basis
            _, _, Vt_p = torch.linalg.svd(J_orig_p.float(), full_matrices=False)
            diff_dirs = Vt_p.T[:, :n_diff]

            gauge_dir = J_orig.sum(dim=0)
            gauge_dir = gauge_dir / gauge_dir.norm()

            basis = torch.stack(
                [gauge_dir] + [diff_dirs[:, i] for i in range(n_diff)], dim=1
            )
            Q, _ = torch.linalg.qr(basis)

            bits = [bits_gauge] + [bits_diff] * n_diff

            # Quantize J^(p) rows
            J_map_p = J_orig_p.clone()
            J_uniform_p = J_orig_p.clone()

            for i in range(J_orig_p.shape[0]):
                row = J_orig_p[i]
                row_map = quantize_vector(row, Q, bits, n_bits_rest)
                row_uniform = uniform_quantize(row, n_bits_uniform)
                J_map_p[i] = row_map
                J_uniform_p[i] = row_uniform

            # Bridge: orig vs map-quantized
            bridge_orig_map = jacobian_gap(J_orig_p, J_map_p, E, E, p_tokens)
            bridge_orig_uniform = jacobian_gap(J_orig_p, J_uniform_p, E, E, p_tokens)
            bridge_map_uniform = jacobian_gap(J_map_p, J_uniform_p, E, E, p_tokens)

            bridges_orig_map.append(bridge_orig_map)
            bridges_orig_uniform.append(bridge_orig_uniform)
            bridges_map_uniform.append(bridge_map_uniform)

            print(
                f"  sample {si}: orig↔map={bridge_orig_map:.4f}, "
                f"orig↔uniform={bridge_orig_uniform:.4f}, map↔uniform={bridge_map_uniform:.4f}"
            )

        all_results[layer] = {
            "orig_vs_map": {
                "avg": sum(bridges_orig_map) / len(bridges_orig_map),
                "values": bridges_orig_map,
            },
            "orig_vs_uniform": {
                "avg": sum(bridges_orig_uniform) / len(bridges_orig_uniform),
                "values": bridges_orig_uniform,
            },
            "map_vs_uniform": {
                "avg": sum(bridges_map_uniform) / len(bridges_map_uniform),
                "values": bridges_map_uniform,
            },
        }

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length}
    with open(os.path.join(results_dir, "exp12c_bridge_quantized.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp12c_bridge_quantized.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 12C: Bridge original vs quantized"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
