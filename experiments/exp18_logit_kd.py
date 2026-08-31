"""Experiment 18: Control — logit-KD with the same budget.

Same LoRA configuration and steps as Exp17, but loss = KL(p_θ ‖ p_L)
on prefixes (no bridge).  Compare Δbridge per unit ΔKL_ed_orig
of Exp17 vs Exp18.

Заголовок-сценарий: «механизм, а не ответы».
"""

import argparse
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.calibration import load_calibration_data
from prism.cache_io import find_targets_dir
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian


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
    ).to(device)  # type: ignore[arg-type]
    mS.eval()

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if device == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )
    _orig_ids = [s["input_ids"].view(-1) for s in samples]

    # -- Load target cache ------------------------------------------------
    # FUSE-safe, layout-independent (prism.cache_io): preferred dir, else
    # listdir walk of /kaggle/input — no hardcoded slug/subpath.
    targets_dir = find_targets_dir(targets_dir)
    print(f"Using target cache at {targets_dir}")

    cache_files = sorted(
        f
        for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(
            f"No target cache .pt files found in {targets_dir}. Run Stage 0 first."
        )

    cache_by_pair: dict[str, list[dict]] = {}
    for cf in cache_files:
        data = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        pair = data["pair"]
        m = re.search(r"_(\d+)\.pt$", cf)
        data["_si"] = int(m.group(1)) if m else -1
        cache_by_pair.setdefault(pair, []).append(data)
    for pair in cache_by_pair:
        cache_by_pair[pair].sort(key=lambda t: t["_si"])

    # Same order as Exp17 late→early
    all_pairs = sorted(cache_by_pair.keys(), key=lambda p: int(p.split("_")[1]))
    late_to_early = list(reversed(all_pairs))
    if device == "cpu":
        wanted = [23, 15, 7]
        late_to_early = [p for p in late_to_early if int(p.split("_")[1]) in wanted]
        if not late_to_early:
            late_to_early = all_pairs[:7]

    n_pairs = len(late_to_early)
    print(f"Pairs ({n_pairs}, late→early): {late_to_early}")

    steps = 100 if device.startswith("cuda") else 40
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
        n_use = min(n_samples, len(targets))
        print(f"\n  Pair {pair_key} (l_S={l_S}, {n_use} targets)")

        # Freeze all params
        for p in mS.parameters():
            p.requires_grad = False

        # LoRA on lm_head only (same convention as Exp17 for fair comparison)
        # A_lh [r, D], B_lh [V, r], delta = B_lh @ A_lh [V, D]
        V, D = mS.lm_head.weight.shape
        A_lh = torch.nn.Parameter(torch.randn(r, D, device=device) * 0.02)
        B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device).T)  # [V, r]

        trainable = [A_lh, B_lh]
        for p in trainable:
            p.requires_grad = True

        optimizer = torch.optim.Adam(trainable, lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            total_loss_val = 0.0
            nan_hit = False

            for si in range(n_use):
                s = samples[si]
                tgt = targets[si]
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)

                # Logit-KD: KL(p_ed ‖ p_L) on token_set subset.
                # Target pairing: skip bridge-of-wrong-sample; KL leash below
                # (eval) uses only the small model.
                if tgt.get("_si") != si:
                    continue

                # W_eff inside the sample loop: fresh graph per backward.
                # (W_eff shared across samples would die on the 2nd
                # backward() — no retain_graph on Kaggle 16GB.)
                W_eff = mS.lm_head.weight + B_lh @ A_lh  # [V, D]
                lm_b = mS.lm_head.bias if mS.lm_head.bias is not None else 0
                # KD target: fp32 p_L_full restricted to token_set, renormalized.
                # (tgt["p"] is the bf16 gauge weight for the Jacobian — rounded,
                # and its mass over token_set is renormalized to 1, so it is not
                # the true restricted distribution; p_L_full is the exact one.)
                p_L_full = tgt["p_L_full"].to(device).float()
                p_L_sub = p_L_full[
                    torch.as_tensor(token_set, dtype=torch.long, device=device)
                ]
                p_L_sub = p_L_sub / p_L_sub.sum()
                out = mS.model(input_ids=ids, attention_mask=attn)
                hidden = out.last_hidden_state[:, -1, :]  # [1, D]
                logits_ed = hidden @ W_eff.T + lm_b  # [1, V]
                p_ed_sub = torch.softmax(logits_ed[0, token_set].float(), dim=-1)
                kd_loss = (
                    p_ed_sub * torch.log(p_ed_sub.clamp_min(1e-12) / (p_L_sub + 1e-12))
                ).mean()
                # NaN guard: one NaN step poisons all subsequent steps of the pair
                if not torch.isfinite(kd_loss):
                    print(f"  [pair] sample {si}: NaN KD loss, stopping pair")
                    nan_hit = True
                    break
                kd_loss.backward()
                total_loss_val += kd_loss.item()

            if nan_hit:
                break

            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                print(f"    step {step + 1}/{steps}: loss={total_loss_val:.4f}")

        # -- Apply LoRA permanently (same as Exp17: W += B@A, no in-place
        # .data swap — the swap breaks the autograd graph of W_eff).
        delta_lm = B_lh @ A_lh  # [V, D]
        with torch.no_grad():
            mS.lm_head.weight.data += delta_lm

        # Evaluate bridge (even though we trained with KD!)
        bridge_vals = []
        with torch.no_grad():
            for si in range(n_use):
                s = samples[si]
                tgt = targets[si]
                if tgt.get("_si") != si:
                    continue
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                J_tgt_p = tgt["J_tgt_p"].to(device)
                J_S_ed = compute_jacobian_for_tokens(mS, l_S, ids, attn, token_set)
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

        # -- Cache original logits for the LAST layer ---------------------
        # Computed after every pair; a per-pair snapshot taken before the
        # pair would be a layer mismatch: KL(p_ed(layer l) || p_orig(l_prev)).
        kl_pairs = []
        with torch.no_grad():
            for si in range(n_use):
                if targets[si].get("input_ids") is None:
                    continue  # pre-v2 cache: no pairing field
                tids = targets[si]["input_ids"].view(-1)
                match = next(
                    (
                        si2
                        for si2 in range(n_samples)
                        if samples[si2]["input_ids"].shape[0] == tids.shape[0]
                        and bool((samples[si2]["input_ids"].view(-1) == tids).all())
                    ),
                    None,
                )
                if match is not None:
                    kl_pairs.append((si, match))
        kl_sum = 0.0
        with torch.no_grad():
            for si, si2 in kl_pairs:
                s = samples[si2]
                ids = s["input_ids"].to(device)
                attn_s = s["attention_mask"].to(device)
                kl = F.kl_div(
                    F.log_softmax(orig_logits_cache[si2].float(), dim=-1),
                    F.log_softmax(
                        mS(input_ids=ids, attention_mask=attn_s)
                        .logits[:, -1, :]
                        .float(),
                        dim=-1,
                    ),
                    log_target=True,  # both args are log-probs: NaN without it
                    reduction="batchmean",
                )
                kl_sum += kl.item()
        kl_avg = kl_sum / max(1, len(kl_pairs))

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
                orig_logits_cache[si] = mS(input_ids=ids, attention_mask=attn_s).logits[
                    :, -1, :
                ]

        del A_lh, B_lh

    # -- Compare with Exp17 if available ----------------------------------
    # Per-pair, aligned on pair_key — the v1 last-vs-first difference assumed
    # both experiments walked the same pair list, which the shallow-pair
    # fallback in 17/exp16r broke (and which stage0 v2 eliminates).
    exp17_path = os.path.join(results_dir, "exp17_lora_distill.json")
    comparison = None
    if os.path.exists(exp17_path):
        with open(exp17_path) as f:
            exp17_data = json.load(f)
        per_layer_17 = exp17_data.get("per_layer") or exp17_data.get(
            "per_order", {}
        ).get("late_to_early", {}).get("per_layer", [])
        if per_layer_17 and per_layer_results:
            p17 = {d["pair"]: d for d in per_layer_17 if "pair" in d}
            shared = [p for p in late_to_early if p in p17]
            if shared:
                # Cumulative sum over shared pairs: sum(kl_i - kl_{i-1}) = kl_last - kl_first
                bridge_17 = sum(float(p17[p]["bridge_mean"]) for p in shared) - float(
                    p17[shared[0]]["bridge_mean"]
                )
                kl_17 = float(p17[shared[-1]]["KL_ed_orig"]) - float(
                    p17[shared[0]]["KL_ed_orig"]
                )
                k18 = {d["pair"]: d for d in per_layer_results}
                bridge_18 = sum(float(k18[p]["bridge_mean"]) for p in shared) - float(
                    k18[shared[0]]["bridge_mean"]
                )
                kl_18 = float(k18[shared[-1]]["KL_ed_orig"]) - float(
                    k18[shared[0]]["KL_ed_orig"]
                )
                ratio_17 = bridge_17 / kl_17 if abs(kl_17) > 1e-8 else float("inf")
                ratio_18 = bridge_18 / kl_18 if abs(kl_18) > 1e-8 else float("inf")
                comparison = {
                    "shared_pairs": shared,
                    "delta_bridge_17": bridge_17,
                    "delta_KL_17": kl_17,
                    "delta_bridge_18": bridge_18,
                    "delta_KL_18": kl_18,
                    "ratio_17_bridge_per_KL": ratio_17,
                    "ratio_18_bridge_per_KL": ratio_18,
                }
                print("\nComparison Exp17 vs Exp18 (shared pairs):")
                print(
                    f"  Exp17 (bridge): Δbridge={bridge_17:.4f}, ΔKL={kl_17:.4f}, ratio={ratio_17:.4f}"
                )
                print(
                    f"  Exp18 (KD):     Δbridge={bridge_18:.4f}, ΔKL={kl_18:.4f}, ratio={ratio_18:.4f}"
                )
            else:
                print("\nExp17 JSON found but no shared pair keys — comparison skipped")

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
    parser = argparse.ArgumentParser(description="Exp 18: Logit-KD control")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
