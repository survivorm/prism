"""Experiment 12B: Quantization by sensitivity map.

Simulate directed quantization of last hidden state:
  - gauge component: 2-3 bits
  - differential components: 8 bits
  - vs uniform scheme with same budget

Metrics: PPL, KL, top-1.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from prism.calibration import load_calibration_data
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)


def quantize_vector(
    x: torch.Tensor,
    directions: torch.Tensor,  # [D, n_dirs] orthonormal basis columns
    bits_per_dir: list[int],  # length n_dirs
    n_bits_rest: int,
) -> torch.Tensor:
    """Quantize x along given directions with specified bits.

    Each direction is quantized to its own bit-depth (step = range / 2^bits).
    Residual (orthogonal complement) gets n_bits_rest.
    """
    assert directions.shape[1] == len(bits_per_dir)

    # Project onto each direction
    coeffs = directions.T @ x  # [n_dirs]

    max_val = x.abs().max()
    q_coeffs = []
    for c, bits in zip(coeffs, bits_per_dir):
        step = 2 * max_val / (2**bits)
        q = torch.round(c / step) * step
        q = torch.clamp(q, -max_val, max_val)
        q_coeffs.append(q)

    # Reconstruct projected part
    recon = directions @ torch.stack(q_coeffs)

    # Residual = x - recon, quantize with uniform bits
    residual = x - recon
    step_r = 2 * residual.abs().max().clamp(min=1e-6) / (2**n_bits_rest)
    q_residual = torch.round(residual / step_r) * step_r

    return recon + q_residual


def uniform_quantize(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Uniform quantization: every dimension gets n_bits."""
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

    # Token set
    s0 = samples[0]
    ids0 = s0["input_ids"].to(m.device)
    attn0 = s0["attention_mask"].to(m.device)
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = lg0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    # Quantization configs
    # Gauge: 2 bits, differential: 8 bits (top-5 of J^(p))
    n_diff = 5
    bits_gauge = 2
    bits_diff = 8

    # Uniform: same total budget → n_bits = log2(D * range) / log2(D)
    # Gauge takes 1 dim, diff takes n_diff dims, rest = D - 1 - n_diff
    # Budget: 1*2 + n_diff*8 + (D-1-n_diff)*n_bits_rest ≈ D * n_uniform
    # n_uniform ≈ (2 + 8*n_diff) / D ≈ 0.05 — too low, unrealistic
    # More realistic: uniform 4 bits across all dims
    n_bits_uniform = 4

    all_results = {}

    for layer in selected:
        print(f"\nLayer {layer}:")
        layer_results = {}

        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]

            # Forward pass
            with torch.no_grad():
                out = m(input_ids=ids, attention_mask=attn, output_hidden_states=True)

            h = out.hidden_states[layer + 1][:, -1, :]  # [1, D]
            h_norm = m.model.norm(h)  # [1, D]
            z_orig = m.lm_head(h_norm)[0]  # [V]
            p_orig = F.softmax(z_orig[0], dim=-1)

            # Build sensitivity map
            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
            p_tokens = torch.softmax(out.logits[0, -1, token_set], dim=-1)
            J_p = softmax_jacobian(J, p_tokens)

            # SVD of J_p → differential directions
            _, _, Vt_p = torch.linalg.svd(J_p.float(), full_matrices=False)
            diff_dirs = Vt_p.T[:, :n_diff]  # [D, n_diff]

            # Gauge direction: J^T @ 1 (common mode)
            gauge_dir = J.sum(dim=0)  # [D]
            gauge_dir = gauge_dir / gauge_dir.norm()

            # Full basis: gauge + differential + rest (orthonormalized)
            basis = torch.stack(
                [gauge_dir] + [diff_dirs[:, i] for i in range(n_diff)], dim=1
            )  # [D, n_diff+1]

            # Orthonormalize (QR)
            Q, _ = torch.linalg.qr(basis)
            directions = Q  # [D, n_diff+1]

            bits = [bits_gauge] + [bits_diff] * n_diff
            n_bits_rest = 6  # reasonable for remaining dims

            # Quantized hidden state
            h_vec = h[0, :]  # [D]

            h_q_map = quantize_vector(h_vec, directions, bits, n_bits_rest)
            h_q_uniform = uniform_quantize(h_vec, n_bits_uniform)

            # Forward with quantized hidden state
            h_q_map_norm = m.model.norm(h_q_map.unsqueeze(0))
            z_q_map = m.lm_head(h_q_map_norm)[0]
            p_q_map = F.softmax(z_q_map, dim=-1)

            h_q_uniform_norm = m.model.norm(h_q_uniform.unsqueeze(0))
            z_q_uniform = m.lm_head(h_q_uniform_norm)[0]
            p_q_uniform = F.softmax(z_q_uniform, dim=-1)

            # Metrics
            kl_map = float(
                F.kl_div(p_q_map.log() + 1e-8, p_orig, reduction="batchmean")
            )
            kl_uniform = float(
                F.kl_div(p_q_uniform.log() + 1e-8, p_orig, reduction="batchmean")
            )

            # PPL (per-token)
            ppl_map = float(
                torch.exp(F.cross_entropy(z_q_map, ids[0, -1]).unsqueeze(0))
            )
            ppl_uniform = float(
                torch.exp(F.cross_entropy(z_q_uniform, ids[0, -1]).unsqueeze(0))
            )
            ppl_orig = float(
                torch.exp(F.cross_entropy(z_orig, ids[0, -1]).unsqueeze(0))
            )

            # Top-1 agreement
            top1_map = int(p_orig.argmax() == p_q_map.argmax())
            top1_uniform = int(p_orig.argmax() == p_q_uniform.argmax())

            entry = {
                "sample": si,
                "kl_map": kl_map,
                "kl_uniform": kl_uniform,
                "ppl_orig": ppl_orig,
                "ppl_map": ppl_map,
                "ppl_uniform": ppl_uniform,
                "top1_map": top1_map,
                "top1_uniform": top1_uniform,
                "delta_h_map": (h_q_map - h_vec).norm().item(),
                "delta_h_uniform": (h_q_uniform - h_vec).norm().item(),
            }

            layer_results.setdefault(str(si), []).append(entry)
            print(
                f"  sample {si}: KL_map={kl_map:.4f}, KL_unif={kl_uniform:.4f}, top1_map={top1_map}, top1_unif={top1_uniform}"
            )

        # Average
        all_entries = [e for v in layer_results.values() for e in v]
        layer_results["_avg"] = {
            "kl_map": sum(e["kl_map"] for e in all_entries) / len(all_entries),
            "kl_uniform": sum(e["kl_uniform"] for e in all_entries) / len(all_entries),
            "ppl_map": sum(e["ppl_map"] for e in all_entries) / len(all_entries),
            "ppl_uniform": sum(e["ppl_uniform"] for e in all_entries)
            / len(all_entries),
        }

        all_results[layer] = layer_results

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length}
    with open(os.path.join(results_dir, "exp12b_quantization_map.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp12b_quantization_map.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 12B: Quantization by sensitivity map"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
