"""Experiment 16: Readout-only pull (no training loop on other layers).

Edit **only** `lm_head` (down_proj of last layer): W' = W + ΔW, starting ΔW=0.

Loss = bridge(J_tgt_p, J_S_p(θ)) + 0.1 · KL(p_θ ‖ p_orig)
After each step: trust-region projection:
    if ‖ΔW‖_F > α·‖W‖_F: ΔW *= α·‖W‖_F / ‖ΔW‖_F
Sweep over α values. Battery metrics at each α.
"""

import argparse
import json
import os
import time

import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


def _alpha_sweep(device: str) -> list[float]:
    if device.startswith("cuda"):
        return [0.01, 0.03, 0.1, 0.3, 1.0]
    return [0.03, 0.1, 0.3]


def _gd_steps(device: str) -> int:
    return 100 if device.startswith("cuda") else 40


def main(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
):
    torch.manual_seed(42)
    start = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}")

    os.makedirs(results_dir, exist_ok=True)

    # -- Load small model -------------------------------------------------
    print(f"Loading small model: {small_model}")
    mS = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    mS.eval()

    n_S = mS.config.num_hidden_layers
    last_layer = n_S - 1
    print(f"Small model: {n_S} layers, last={last_layer}")

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if not device.startswith("cuda") else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    # -- Resolve target cache location ------------------------------------
    # Fallback chain: local dir -> kernel_sources -> error
    possible_dirs = [
        targets_dir,
        os.path.join("/kaggle/input", "prism-exp16-stage0-targets"),
        os.path.join(
            "/kaggle/input", "prism-exp16-stage0-targets", "results", "targets"
        ),
    ]
    for d in possible_dirs:
        if os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d)):
            targets_dir = d
            print(f"Found target cache at {d}")
            break

    # -- Load target cache ------------------------------------------------
    cache_files = sorted(
        f
        for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(
            f"No target cache found in any of: {possible_dirs}. Run Stage 0 first."
        )

    # Load all cache files, group by pair
    cache_by_pair: dict[str, list[dict]] = {}
    for cf in cache_files:
        data = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        pair = data["pair"]
        cache_by_pair.setdefault(pair, []).append(data)

    # Prefer last-layer pair; fall back to any pair
    target_pair = None
    for p, c in cache_by_pair.items():
        _, ls = p.split("_")
        if int(ls) == last_layer:
            target_pair = p
            break
    if target_pair is None:
        target_pair = list(cache_by_pair.keys())[0]
    targets = cache_by_pair[target_pair]
    token_set = targets[0]["token_set"]
    K = len(token_set)
    print(f"Using pair {target_pair}, {len(targets)} cached samples, K={K}")

    # -- Original lm_head weights -----------------------------------------
    lm_head = mS.lm_head
    W_orig = lm_head.weight.data.clone()
    W_norm = W_orig.norm().item()
    print(f"lm_head W: {W_orig.shape}, ‖W‖_F = {W_norm:.4f}")

    # -- Pre-compute original logits on all samples (for KL leash) ---------
    orig_logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for s in samples[:n_samples]:
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            logits = mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            orig_logits_cache.append(logits)
    print("Cached original logits for KL leash")

    # -- Alpha sweep ------------------------------------------------------
    alphas = _alpha_sweep(device)
    steps = _gd_steps(device)
    lr = 1e-3
    kl_weight = 0.1

    per_alpha_results = {}

    for alpha in alphas:
        print(f"\n{'=' * 60}")
        print(f"α = {alpha}")
        print(f"{'=' * 60}")

        # Reset lm_head to original
        lm_head.weight.data.copy_(W_orig)

        optimizer = torch.optim.Adam([lm_head.weight], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            total_loss = 0.0

            for si in range(min(n_samples, len(targets))):
                s = samples[si]
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                tgt = targets[si]
                J_tgt_p = tgt["J_tgt_p"].to(device)

                # Forward: compute J_S on edited model
                J_S = compute_jacobian_for_tokens(mS, last_layer, ids, attn, token_set)
                p_L = tgt["p"].to(device)
                J_S_p = softmax_jacobian(J_S, p_L)

                # Bridge loss: 1 - cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    J_S_p.float(), J_tgt_p.float(), dim=1
                )
                bridge_loss = (1.0 - cos_sim).mean()

                # KL leash: KL(p_ed ‖ p_orig)
                with torch.no_grad():
                    p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
                p_ed = torch.softmax(
                    mS(input_ids=ids, attention_mask=attn).logits[:, -1, :].float(),
                    dim=-1,
                )
                kl_loss = (p_ed * torch.log(p_ed / (p_orig + 1e-12))).mean()

                loss = bridge_loss + kl_weight * kl_loss
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            # Trust-region projection
            delta_W = lm_head.weight.data - W_orig
            delta_norm = delta_W.norm().item()
            max_norm = alpha * W_norm
            if delta_norm > max_norm and delta_norm > 1e-8:
                scale = max_norm / delta_norm
                lm_head.weight.data.copy_(W_orig + delta_W * scale)

            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"  step {step + 1}/{steps}: loss={total_loss.item():.4f}, "
                    f"‖ΔW‖_F={delta_W.norm().item():.4f}"
                )

        # -- Evaluate at this α -------------------------------------------
        delta_W_final = lm_head.weight.data - W_orig
        print(f"  Final ‖ΔW‖_F = {delta_W_final.norm().item():.4f}")

        # Bridge on cached targets
        bridge_vals = []
        for si in range(min(n_samples, len(targets))):
            s = samples[si]
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            tgt = targets[si]
            J_tgt_p = tgt["J_tgt_p"].to(device)
            J_S_ed = compute_jacobian_for_tokens(mS, last_layer, ids, attn, token_set)
            p_L = tgt["p"].to(device)
            J_S_ed_p = softmax_jacobian(J_S_ed, p_L)
            cos = (
                torch.nn.functional.cosine_similarity(
                    J_S_ed_p.float(), J_tgt_p.float(), dim=1
                )
                .mean()
                .item()
            )
            bridge_vals.append(cos)

        # KL to original
        kl_sum = 0.0
        for si in range(min(n_samples, len(orig_logits_cache))):
            s = samples[si]
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            logits_ed = mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            p_ed = torch.softmax(logits_ed.float(), dim=-1)
            p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
            kl = (p_ed * torch.log(p_ed / (p_orig + 1e-12))).mean().item()
            kl_sum += kl
        kl_avg = kl_sum / min(n_samples, len(orig_logits_cache))

        battery = {
            "alpha": alpha,
            "bridge_mean": sum(bridge_vals) / len(bridge_vals) if bridge_vals else 0,
            "KL_ed_orig": kl_avg,
            "delta_W_norm": delta_W_final.norm().item(),
            "delta_W_frac": delta_W_final.norm().item() / W_norm,
        }
        per_alpha_results[str(alpha)] = battery

        print(
            f"  α={alpha}: bridge={battery['bridge_mean']:.4f}, "
            f"KL={battery['KL_ed_orig']:.4f}, ‖ΔW‖/‖W‖={battery['delta_W_frac']:.4f}"
        )

    # -- Determine α* (elasticity limit) ----------------------------------
    alpha_star = alphas[0]
    prev_bridge = float("inf")
    for alpha in alphas:
        b = per_alpha_results[str(alpha)]["bridge_mean"]
        kl = per_alpha_results[str(alpha)]["KL_ed_orig"]
        if b < prev_bridge and kl < 0.5:
            alpha_star = alpha
            prev_bridge = b
        else:
            break

    print(f"\nα* (elasticity limit) = {alpha_star}")

    # -- Save results ------------------------------------------------------
    result = {
        "description": "Readout-only pull: gradient descent on lm_head only, minimizing bridge loss (cosine to transported large-model Jacobian) + KL leash to original predictions. Trust-region projection constrains ‖ΔW‖_F ≤ α·‖W‖_F. Sweeps alpha values to find elasticity limit alpha* where bridge improvement stalls. Measures bridge_mean (cosine similarity), KL_ed_orig, and ‖ΔW‖/‖W‖ at each alpha.",
        "per_alpha": per_alpha_results,
        "alpha_star": alpha_star,
        "target_pair": target_pair,
        "n_layers_S": n_S,
        "last_layer": last_layer,
        "K": K,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp16_readout_pull.json")
    result["params"] = {
        "small_model": small_model,
        "device": device,
        "max_length": max_length,
        "min_length": min_length,
        "targets_dir": targets_dir,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 16: Readout-only pull")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
