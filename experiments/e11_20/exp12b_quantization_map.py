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
    _run_prefix_pass,
    last_position_logits,
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

    coeffs = directions.T @ x  # [n_dirs]

    max_val = x.abs().max()
    q_coeffs = []
    for c, bits in zip(coeffs, bits_per_dir):
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
    """Uniform quantization: every dimension gets n_bits."""
    max_val = x.abs().max()
    step = 2 * max_val / (2**n_bits)
    q = torch.round(x / step) * step
    return torch.clamp(q, -max_val, max_val)


def run_full_decoder(
    model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int,
    probs: bool = True,
):
    """Истинный хвост модели: слои layer+1..n, norm, head — путь prism
    (_run_decoder_pass), согласованный с compute_jacobian_for_tokens. h —
    состояние ПРОМЕЖУТОЧНОГО слоя, срезанный lm_head(norm(h)) — не выход
    модели (аудит 2026-11-05, баг 1: ppl_orig = 3.7e6). Баг 2 того же дня:
    прежняя копия клала h поверх кэша полного forward (0..S-1) — двойной
    учёт последнего токена; prefix_cache обязан быть из _run_prefix_pass.
    """
    with torch.no_grad():
        logits = last_position_logits(model, h, S, layer_idx, prefix_cache)
    return torch.softmax(logits[0].float(), dim=-1) if probs else logits[0].float()


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
                out = m(input_ids=ids, attention_mask=attn, output_hidden_states=True,
                        use_cache=True)
            prefix_cache = _run_prefix_pass(m, ids, attn)  # 0..S-2 (аудит 2026-11-05)

            h = out.hidden_states[layer + 1][:, -1:, :]  # [1, 1, D] (форма для хвоста)
            # ЯКОРЬ — истинный хвост модели (см. run_full_decoder), не lm_head(norm(h)):
            # на слоях 6/12/18 срезанный ярлык давал ppl_orig ≈ 3.7e6 в артефакте.
            z_orig = run_full_decoder(m, prefix_cache, h, ids.shape[1], layer,
                                      probs=False)  # [V]
            p_orig = F.softmax(z_orig, dim=-1)
            # fail-loud: хвост при δ=0 обязан совпасть с полным форвардом
            _d_anchor = (z_orig - out.logits[0, -1]).abs().max().item()
            assert _d_anchor < 1e-2, f"E-12b anchor tail vs full forward: {_d_anchor:.2e}"

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
            h_vec = h[0, 0, :]  # [D]

            h_q_map = quantize_vector(h_vec, directions, bits, n_bits_rest)
            h_q_uniform = uniform_quantize(h_vec, n_bits_uniform)
            # Диагностика климпера (аудит 2026-11-05): квантизатор режет сетку
            # ±‖x‖_∞ во ВСЕХ направлениях, а легитимный коэффициент |<v,x>| ≤ ‖x‖₂
            # ≫ ‖x‖_∞ — 8-битная «защита» дифф-направлений может съедаться климпом,
            # а не битностью. Пишем |c| и долю климпованных в артефакт.
            _coeffs = directions.T @ h_vec
            _mv = float(h_vec.abs().max())
            clamp_diag = {
                "max_abs_x": _mv,
                "coeff_abs": [float(abs(c)) for c in _coeffs],
                "clamp_frac": [float(abs(c) > _mv) for c in _coeffs],
            }

            # Оба квантованных состояния — через ИСТИННЫЙ хвост (тот же
            # run_full_decoder, что и база: сравнение корректно только на
            # одной и той же функции выхода)
            z_q_map = run_full_decoder(m, prefix_cache, h_q_map.view(1, 1, -1),
                                       ids.shape[1], layer, probs=False)
            p_q_map = F.softmax(z_q_map, dim=-1)

            z_q_uniform = run_full_decoder(m, prefix_cache, h_q_uniform.view(1, 1, -1),
                                           ids.shape[1], layer, probs=False)
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
                "clamp_diag": clamp_diag,
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
    all_results["params"] = {"model": model, "n_samples": n_samples, "K": K, "max_length": max_length, "min_length": min_length}
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
