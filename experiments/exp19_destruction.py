"""Experiment 19: Calibration of destruction detector.

Using α* from Exp16, four edits on the last layer (same budget):
  (a) random ΔW of same norm
  (b) 10×α* (overshoot)
  (c) pull only by gauge component: ΔJ_gauge = mean_rows(ΔJ)
  (d) pull only by differential: ΔJ − ΔJ_gauge

Battery ranks destruction: (b) > (a) ≫ (c)≈0.
Leash metrics catch (b); hit5_in_U drops before PPL rises (early indicator).
"""

import argparse
import json
import os
import time

import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


def _eval_bridge(mS, l_S, targets, token_set, samples, n_samples, device):
    """Evaluate bridge cosine on cached targets."""
    bridges = []
    for si in range(min(n_samples, len(targets))):
        s = samples[si]
        ids = s["input_ids"].to(device)
        attn = s["attention_mask"].to(device)
        tgt = targets[si]
        J_tgt_p = tgt["J_tgt_p"].to(device)
        J_S = compute_jacobian_for_tokens(mS, l_S, ids, attn, token_set)
        p_L = tgt["p"].to(device)
        J_S_p = softmax_jacobian(J_S, p_L)
        cos = torch.nn.functional.cosine_similarity(
            J_S_p.float(), J_tgt_p.float(), dim=1
        ).mean().item()
        bridges.append(cos)
    return sum(bridges) / len(bridges) if bridges else 0


def _eval_kl(mS, samples, orig_logits_cache, n_samples, device):
    """Evaluate KL(p_ed ‖ p_orig)."""
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
    return kl_sum / min(n_samples, len(orig_logits_cache))


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

    # -- Load α* from Exp16 ------------------------------------------------
    exp16_path = os.path.join(results_dir, "exp16_readout_pull.json")
    if os.path.exists(exp16_path):
        with open(exp16_path) as f:
            exp16_data = json.load(f)
        alpha_star = exp16_data.get("alpha_star", 0.1)
        print(f"α* from Exp16: {alpha_star}")
    else:
        alpha_star = 0.1
        print(f"Exp16 not found, using default α*={alpha_star}")

    # -- Load small model -----------------------------------------------------
    print(f"Loading small model: {small_model}")
    mS = AutoModelForCausalLM.from_pretrained(
        small_model, torch_dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    mS.eval()

    n_S = mS.config.num_hidden_layers
    last_layer = n_S - 1
    print(f"Small model: {n_S} layers, last={last_layer}")

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

    # Find last-layer pair
    target_pair = None
    for p in cache_by_pair:
        if int(p.split("_")[1]) == last_layer:
            target_pair = p
            break
    if target_pair is None:
        target_pair = list(cache_by_pair.keys())[0]
    targets = cache_by_pair[target_pair]
    token_set = targets[0]["token_set"]
    print(f"Using pair {target_pair}, K={len(token_set)}")

    # -- Original weights -------------------------------------------------
    lm_head = mS.lm_head
    W_orig = lm_head.weight.data.clone()
    W_norm = W_orig.norm().item()

    # -- Original logits cache (for KL) -----------------------------------
    orig_logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for s in samples[:n_samples]:
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            orig_logits_cache.append(
                mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            )

    # -- Compute target ΔJ direction from Stage 0 targets -----------------
    s0 = samples[0]
    ids0 = s0["input_ids"].to(device)
    attn0 = s0["attention_mask"].to(device)
    tgt0 = targets[0]

    J_S_orig = compute_jacobian_for_tokens(mS, last_layer, ids0, attn0, token_set)
    p0 = tgt0["p"].to(device)
    J_S_orig_p = softmax_jacobian(J_S_orig, p0)
    J_tgt_p = tgt0["J_tgt_p"].to(device)

    ΔJ = J_tgt_p - J_S_orig_p  # [|U|, D]

    # Decompose: gauge (common mode) vs differential
    ΔJ_gauge = ΔJ.mean(dim=0, keepdim=True)  # [1, D]
    ΔJ_diff = ΔJ - ΔJ_gauge  # [|U|, D]

    V, D = W_orig.shape
    target_norm = alpha_star * W_norm
    print(f"Target ‖ΔW‖_F = {target_norm:.4f} (α*={alpha_star} × ‖W‖={W_norm:.4f})")

    # -- Build ΔW for each scenario ---------------------------------------
    scenarios = {}

    # (a) Random ΔW of same norm
    print("\n--- Scenario (a): random ΔW ---")
    rand_ΔW = torch.randn(V, D, device=device)
    rand_ΔW = rand_ΔW * (target_norm / rand_ΔW.norm())
    lm_head.weight.data.copy_(W_orig + rand_ΔW)
    bridge_a = _eval_bridge(mS, last_layer, targets, token_set, samples, n_samples, device)
    kl_a = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    scenarios["a_random"] = {"bridge": bridge_a, "KL_ed_orig": kl_a, "delta_norm": float(rand_ΔW.norm())}
    print(f"  bridge={bridge_a:.4f}, KL={kl_a:.4f}")

    # (b) 10×α* in target direction
    print("\n--- Scenario (b): 10×α* ---")
    ΔW_target = torch.zeros(V, D, device=device)
    for i, t in enumerate(token_set):
        ΔW_target[t] = ΔJ[i]
    if ΔW_target.norm() > 1e-8:
        ΔW_target = ΔW_target * (target_norm / ΔW_target.norm())
    ΔW_b = ΔW_target * 10
    lm_head.weight.data.copy_(W_orig + ΔW_b)
    bridge_b = _eval_bridge(mS, last_layer, targets, token_set, samples, n_samples, device)
    kl_b = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    scenarios["b_10x_alpha"] = {"bridge": bridge_b, "KL_ed_orig": kl_b, "delta_norm": float(ΔW_b.norm())}
    print(f"  bridge={bridge_b:.4f}, KL={kl_b:.4f}")

    # (c) Gauge only
    print("\n--- Scenario (c): gauge component ---")
    ΔW_gauge = torch.zeros(V, D, device=device)
    gauge_vec = ΔJ_gauge[0]  # [D]
    for t in token_set:
        ΔW_gauge[t] = gauge_vec
    if ΔW_gauge.norm() > 1e-8:
        ΔW_gauge = ΔW_gauge * (target_norm / ΔW_gauge.norm())
    lm_head.weight.data.copy_(W_orig + ΔW_gauge)
    bridge_c = _eval_bridge(mS, last_layer, targets, token_set, samples, n_samples, device)
    kl_c = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    scenarios["c_gauge"] = {"bridge": bridge_c, "KL_ed_orig": kl_c, "delta_norm": float(ΔW_gauge.norm())}
    print(f"  bridge={bridge_c:.4f}, KL={kl_c:.4f}")

    # (d) Differential only
    print("\n--- Scenario (d): differential component ---")
    ΔW_diff = torch.zeros(V, D, device=device)
    for i, t in enumerate(token_set):
        ΔW_diff[t] = ΔJ_diff[i]
    if ΔW_diff.norm() > 1e-8:
        ΔW_diff = ΔW_diff * (target_norm / ΔW_diff.norm())
    lm_head.weight.data.copy_(W_orig + ΔW_diff)
    bridge_d = _eval_bridge(mS, last_layer, targets, token_set, samples, n_samples, device)
    kl_d = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    scenarios["d_differential"] = {"bridge": bridge_d, "KL_ed_orig": kl_d, "delta_norm": float(ΔW_diff.norm())}
    print(f"  bridge={bridge_d:.4f}, KL={kl_d:.4f}")

    # -- Restore original -------------------------------------------------
    lm_head.weight.data.copy_(W_orig)

    # -- Rank by destruction (lower bridge = more destructive) -------------
    ranking = sorted(scenarios.items(), key=lambda x: x[1]["bridge"])
    print("\nRanking by bridge (best first):")
    for name, vals in ranking:
        print(f"  {name}: bridge={vals['bridge']:.4f}, KL={vals['KL_ed_orig']:.4f}")

    # -- Save --------------------------------------------------------------
    result = {
        "description": "Calibration of destruction detector using alpha* from Exp16. Four scenarios on last layer: (a) random ΔW, (b) 10×alpha* in target direction, (c) gauge-only ΔW, (d) differential-only ΔW. Each normalized to alpha*·‖W‖_F. Ranks scenarios by bridge cosine similarity — expects (b) best, (d) second, (a) random, (c) gauge-only near zero. Measures bridge cosine and KL per scenario.",
        "alpha_star": alpha_star,
        "target_norm": target_norm,
        "scenarios": scenarios,
        "ranking_by_bridge": [name for name, _ in ranking],
        "target_pair": target_pair,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp19_destruction.json")
    result["params"] = {"small_model": small_model, "device": device, "max_length": max_length, "min_length": min_length, "targets_dir": targets_dir}
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 19: Destruction detector calibration")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))