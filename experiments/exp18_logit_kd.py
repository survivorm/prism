"""Experiment 18: Control — logit-KD with the same budget.

Same LoRA configuration and steps as Exp17, but loss = KL(p_θ ‖ p_L)
on prefixes (no bridge).  Compare Δbridge per unit ΔKL_ed_orig
of Exp17 vs Exp18.

Заголовок-сценарий: «механизм, а не ответы».
"""

import argparse
import json
import os
import time

import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


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
        small_model, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    mS.eval()

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if device == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model, max_length=max_length, min_length=min_length, n_samples=n_samples,
    )

    # -- Load target cache ------------------------------------------------
    # Fallback chain: local dir -> kernel_sources -> error
    possible_dirs = [targets_dir,
                     os.path.join("/kaggle/input", "prism-exp16-stage0-targets"),
                     os.path.join("/kaggle/input", "prism-exp16-stage0-targets", "results", "targets")]
    for d in possible_dirs:
        if os.path.isdir(d) and any(f.endswith(".pt") for f in os.listdir(d)):
            targets_dir = d
            print(f"Found target cache at {d}")
            break

    cache_files = sorted(
        f for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(f"No target cache found in any of: {possible_dirs}. Run Stage 0 first.")

    cache_by_pair: dict[str, list[dict]] = {}
    for cf in cache_files:
        data = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        pair = data["pair"]
        cache_by_pair.setdefault(pair, []).append(data)

    # Same order as Exp17 late→early
    all_pairs = sorted(cache_by_pair.keys(), key=lambda p: int(p.split("_")[1]))
    late_to_early = list(reversed(all_pairs))
    if device == "cpu":
        wanted = [23, 15, 7]
        late_to_early = [p for p in late_to_early if int(p.split("_")[1]) in wanted]
        if not late_to_early:
            late_to_early = all_pairs[:7]

    steps = 100 if device == "cuda" else 40
    r = 4
    lr = 1e-3

    per_layer_results = []

    # Cache original logits
    orig_logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for s in samples[:n_samples]:
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            orig_logits_cache.append(
                mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            )

    # -- Iterate pairs late→early -----------------------------------------
    for pair_key in late_to_early:
        targets = cache_by_pair[pair_key]
        l_S = int(pair_key.split("_")[1])
        token_set = targets[0]["token_set"]
        print(f"\n  Pair {pair_key} (l_S={l_S})")

        # Freeze all params
        for p in mS.parameters():
            p.requires_grad = False

        # LoRA on lm_head only (same as Exp17 for fair comparison)
        V, D = mS.lm_head.weight.shape
        A_lh = torch.nn.Parameter(torch.randn(D, r, device=device) * 0.02)
        B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device))

        trainable = [A_lh, B_lh]
        for p in trainable:
            p.requires_grad = True

        optimizer = torch.optim.Adam(trainable, lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            total_loss = 0.0

            for si in range(min(n_samples, len(targets))):
                s = samples[si]
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)

                # Apply LoRA
                lm_orig = mS.lm_head.weight.data.clone()
                mS.lm_head.weight.data = lm_orig + (B_lh @ A_lh).T

                try:
                    # Logit-KD: KL(p_ed ‖ p_L) on token_set subset
                    tgt = targets[si]
                    p_L_sub = tgt["p"].to(device)  # large model probs on token_set

                    logits_ed = mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
                    p_ed_sub = torch.softmax(logits_ed[0, token_set].float(), dim=-1)

                    kd_loss = (p_ed_sub * torch.log(p_ed_sub / (p_L_sub + 1e-12))).mean()
                    total_loss += kd_loss
                except Exception as e:
                    print(f"    Step {step} sample {si} error: {e}")
                finally:
                    mS.lm_head.weight.data.copy_(lm_orig)

            total_loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                print(f"    step {step+1}/{steps}: loss={total_loss.item():.4f}")

        # Apply LoRA permanently
        delta_lm = (B_lh @ A_lh).T
        mS.lm_head.weight.data += delta_lm

        # Evaluate bridge (even though we trained with KD!)
        bridge_vals = []
        for si in range(min(n_samples, len(targets))):
            s = samples[si]
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            tgt = targets[si]
            J_tgt_p = tgt["J_tgt_p"].to(device)
            J_S_ed = compute_jacobian_for_tokens(mS, l_S, ids, attn, token_set)
            p_L = tgt["p"].to(device)
            J_S_ed_p = softmax_jacobian(J_S_ed, p_L)
            cos = torch.nn.functional.cosine_similarity(
                J_S_ed_p.float(), J_tgt_p.float(), dim=1
            ).mean().item()
            bridge_vals.append(cos)

        # KL to original
        kl_sum = 0.0
        for si in range(min(n_samples, len(orig_logits_cache))):
            s = samples[si]
            ids = s["input_ids"].to(device)
            attn_s = s["attention_mask"].to(device)
            logits_ed = mS(input_ids=ids, attention_mask=attn_s).logits[:, -1, :]
            p_ed = torch.softmax(logits_ed.float(), dim=-1)
            p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
            kl = (p_ed * torch.log(p_ed / (p_orig + 1e-12))).mean().item()
            kl_sum += kl
        kl_avg = kl_sum / min(n_samples, len(orig_logits_cache))

        layer_result = {
            "pair": pair_key,
            "bridge_mean": sum(bridge_vals) / len(bridge_vals) if bridge_vals else 0,
            "KL_ed_orig": kl_avg,
        }
        per_layer_results.append(layer_result)
        print(f"    bridge={layer_result['bridge_mean']:.4f}, KL={kl_avg:.4f}")

        # Update orig logits for next layer
        with torch.no_grad():
            for si, s in enumerate(samples[:n_samples]):
                ids = s["input_ids"].to(device)
                attn_s = s["attention_mask"].to(device)
                orig_logits_cache[si] = mS(input_ids=ids, attention_mask=attn_s).logits[:, -1, :]

        del A_lh, B_lh

    # -- Compare with Exp17 if available ----------------------------------
    exp17_path = os.path.join(results_dir, "exp17_lora_distill.json")
    comparison = None
    if os.path.exists(exp17_path):
        with open(exp17_path) as f:
            exp17_data = json.load(f)
        exp17_order = exp17_data.get("per_order", {}).get("late_to_early", {})
        exp17_layers = exp17_order.get("per_layer", [])
        if exp17_layers and per_layer_results:
            bridge_17 = exp17_layers[-1]["bridge_mean"] - exp17_layers[0]["bridge_mean"]
            kl_17 = exp17_layers[-1]["KL_ed_orig"] - exp17_layers[0]["KL_ed_orig"]
            bridge_18 = per_layer_results[-1]["bridge_mean"] - per_layer_results[0]["bridge_mean"]
            kl_18 = per_layer_results[-1]["KL_ed_orig"] - per_layer_results[0]["KL_ed_orig"]
            ratio_17 = bridge_17 / kl_17 if abs(kl_17) > 1e-8 else float("inf")
            ratio_18 = bridge_18 / kl_18 if abs(kl_18) > 1e-8 else float("inf")
            comparison = {
                "delta_bridge_17": bridge_17,
                "delta_KL_17": kl_17,
                "delta_bridge_18": bridge_18,
                "delta_KL_18": kl_18,
                "ratio_17_bridge_per_KL": ratio_17,
                "ratio_18_bridge_per_KL": ratio_18,
            }
            print("\nComparison Exp17 vs Exp18:")
            print(f"  Exp17 (bridge): Δbridge={bridge_17:.4f}, ΔKL={kl_17:.4f}, ratio={ratio_17:.4f}")
            print(f"  Exp18 (KD):     Δbridge={bridge_18:.4f}, ΔKL={kl_18:.4f}, ratio={ratio_18:.4f}")

    # -- Save --------------------------------------------------------------
    result = {
        "description": "Control experiment for Exp17: same LoRA configuration (r=4 on lm_head) but loss is KL divergence between edited-small logits and large-model logits on token_set, instead of bridge loss. Purpose: compare delta-bridge per unit delta-KL between bridge-guided (Exp17) and logit-KD (Exp18) to test whether bridge guidance is more efficient at aligning sensitivity.",
        "per_layer": per_layer_results,
        "pairs": late_to_early,
        "r": r,
        "steps": steps,
        "comparison_with_exp17": comparison,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp18_logit_kd.json")
    result["params"] = {"small_model": small_model, "device": device, "max_length": max_length, "min_length": min_length, "targets_dir": targets_dir}
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 18: Logit-KD control")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))