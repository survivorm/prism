"""Experiment 17: Per-layer LoRA distillation of the bridge.

LoRA r=4 on `lm_head` of each layer from the pair sequence.  For layer pair l:
  loss = bridge(J_tgt_p, J_S_p(θ)) + 0.1 · KL(p_θ ‖ p_orig)
  100/40 steps; freeze; recompute tail Jacobians for next layer.
Two orders (GPU), only late→early (CPU).

CPU config: pairs [23, 15, 7] mapped to small model layers.
"""

import argparse
import json
import os
import time

import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


def _layer_order(pairs: list[str], device: str) -> dict[str, list[str]]:
    late_to_early = sorted(pairs, key=lambda p: int(p.split("_")[1]), reverse=True)
    if device == "cpu":
        # CPU subset: only late→early, layers [23, 15, 7] or closest available
        wanted = [23, 15, 7]
        subset = [p for p in late_to_early if int(p.split("_")[1]) in wanted]
        if not subset:
            subset = late_to_early[:7]
        return {"late_to_early": subset}
    return {"late_to_early": late_to_early, "early_to_late": list(reversed(late_to_early))}


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
    mS_template = small_model  # for re-loading fresh copies
    n_S = 24  # Qwen3.5-0.8B has 24 layers

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if device == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model, max_length=max_length, min_length=min_length, n_samples=n_samples,
    )

    # -- Load target cache ------------------------------------------------
    cache_files = sorted(
        f for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(f"No target cache in {targets_dir}. Run Stage 0 first.")

    cache_by_pair: dict[str, list[dict]] = {}
    for cf in cache_files:
        data = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        pair = data["pair"]
        cache_by_pair.setdefault(pair, []).append(data)

    all_pairs = sorted(cache_by_pair.keys(), key=lambda p: int(p.split("_")[1]))
    orders = _layer_order(all_pairs, device)

    steps = 100 if device == "cuda" else 40
    r = 4
    lr = 1e-3
    kl_weight = 0.1

    per_order_results = {}

    for order_name, pair_seq in orders.items():
        print(f"\n{'='*60}")
        print(f"Order: {order_name} ({len(pair_seq)} pairs)")
        print(f"{'='*60}")

        # Load fresh model for this order
        mS = AutoModelForCausalLM.from_pretrained(
            small_model, torch_dtype=torch.float32, attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
        mS.eval()

        # Cache original logits (updated after each layer)
        orig_logits_cache: list[torch.Tensor] = []
        with torch.no_grad():
            for s in samples[:n_samples]:
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                orig_logits_cache.append(
                    mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
                )

        per_layer_results = []

        for pair_key in pair_seq:
            targets = cache_by_pair[pair_key]
            l_L_str, l_S_str = pair_key.split("_")
            l_S = int(l_S_str)
            token_set = targets[0]["token_set"]
            print(f"\n  Pair {pair_key} (l_S={l_S})")

            # Freeze all params
            for p in mS.parameters():
                p.requires_grad = False

            # LoRA on lm_head: ΔW = (B @ A).T, A: [D,r], B: [r,V]
            V, D = mS.lm_head.weight.shape
            A_lh = torch.nn.Parameter(torch.randn(D, r, device=device) * 0.02)
            B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device))

            # LoRA on target layer's attention output projection
            layer_mod = mS.model.layers[l_S]
            attn_mod = layer_mod.self_attn if hasattr(layer_mod, "self_attn") else layer_mod.linear_attn

            # Find o_proj weight
            o_proj_name = None
            for name, _ in attn_mod.named_parameters():
                if "o_proj" in name:
                    o_proj_name = name
                    break

            A_attn = None
            B_attn = None
            if o_proj_name:
                o_proj_weight = getattr(attn_mod, o_proj_name)
                V_l, D_l = o_proj_weight.shape
                A_attn = torch.nn.Parameter(torch.randn(D_l, r, device=device) * 0.02)
                B_attn = torch.nn.Parameter(torch.zeros(r, V_l, device=device))

            trainable = [A_lh, B_lh]
            if A_attn is not None:
                trainable.extend([A_attn, B_attn])
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
                    tgt = targets[si]
                    J_tgt_p = tgt["J_tgt_p"].to(device)

                    # Apply LoRA to lm_head for Jacobian computation
                    lm_orig = mS.lm_head.weight.data.clone()
                    delta_lm = (B_lh @ A_lh).T
                    mS.lm_head.weight.data = lm_orig + delta_lm

                    # Also apply LoRA to attention o_proj
                    if A_attn is not None and o_proj_name:
                        attn_orig = getattr(attn_mod, o_proj_name).clone()
                        delta_attn = B_attn @ A_attn
                        setattr(attn_mod, o_proj_name, attn_orig + delta_attn)

                    try:
                        J_S = compute_jacobian_for_tokens(mS, l_S, ids, attn, token_set)
                        p_L = tgt["p"].to(device)
                        J_S_p = softmax_jacobian(J_S, p_L)

                        cos_sim = torch.nn.functional.cosine_similarity(
                            J_S_p.float(), J_tgt_p.float(), dim=1
                        )
                        bridge_loss = (1.0 - cos_sim).mean()

                        # KL leash: need forward with LoRA applied
                        with torch.no_grad():
                            p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
                        logits_ed = mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
                        p_ed = torch.softmax(logits_ed.float(), dim=-1)
                        kl_loss = (p_ed * torch.log(p_ed / (p_orig + 1e-12))).mean()

                        total_loss += bridge_loss + kl_weight * kl_loss
                    except Exception as e:
                        print(f"    Step {step} sample {si} error: {e}")
                    finally:
                        # Restore weights
                        mS.lm_head.weight.data.copy_(lm_orig)
                        if A_attn is not None and o_proj_name:
                            setattr(attn_mod, o_proj_name, attn_orig)

                total_loss.backward()
                optimizer.step()

                if (step + 1) % 10 == 0 or step == 0:
                    print(f"    step {step+1}/{steps}: loss={total_loss.item():.4f}")

            # Freeze: apply LoRA permanently to lm_head
            delta_lm_final = (B_lh @ A_lh).T
            mS.lm_head.weight.data += delta_lm_final

            # Evaluate bridge
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

            # Update orig logits for next layer (cumulative edits)
            with torch.no_grad():
                for si, s in enumerate(samples[:n_samples]):
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    orig_logits_cache[si] = mS(input_ids=ids, attention_mask=attn_s).logits[:, -1, :]

            # Clean up
            del A_lh, B_lh
            if A_attn is not None:
                del A_attn, B_attn

        per_order_results[order_name] = {"pairs": pair_seq, "per_layer": per_layer_results}

    # -- Save --------------------------------------------------------------
    result = {
        "per_order": per_order_results,
        "r": r,
        "steps": steps,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp17_lora_distill.json")
    result["params"] = {"small_model": small_model, "max_length": max_length, "min_length": min_length}
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 17: Per-layer LoRA distillation")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))