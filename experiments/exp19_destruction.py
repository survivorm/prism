"""Experiment 19: Dose sweep — bridge response curve via lm_head edit (v6, 2026-09-13).

Maps the FULL dose-response: bridge vs ‖ΔW‖_F (absolute Frobenius norm of
the lm_head edit) across ×10 steps and 4 directions.

Doses: 0.1, 1, 10, 100, 500, 1000 (absolute ‖ΔW‖_F)
Directions:
  target    — toward 4B target (ΔJ direction)
  anti      — away from 4B target (−ΔJ)
  gauge     — common-mode row only (ΔJ_gauge)
  random    — random direction

Single pair: 31_23 (l_S=23, zero remaining decoder layers → TPU-safe,
one XLA graph reused for all scenarios since ΔW applies to lm_head only).

Per direction: stop increasing dose once bridge < 0.05 (total collapse —
further doses are meaningless).

Output: results/exp19_destruction.json with full grid + saved ΔW .pt
files for inference inspection (all doses, both 0.8B and 4B compatible).

Cost: 6 doses × 4 dirs × (1 eval_bridge + 1 eval_kl) = ~48 evals on the
smallest possible graph (last-layer, no decoder). ~30–60 s total on TPU.
"""

import argparse
import gc
import json
import os
import re
import time

import torch
# --- XLA / TPU support (no-op on GPU/CPU; active when device == "xla") ---
try:
    import torch_xla.core.xla_model as _xm
    _XLA = True
except Exception:
    _xm = None
    _XLA = False

# (exp19 is closed-form: no optimizer step, no xm.step() needed on TPU.)

from prism.calibration import load_calibration_data
from prism.cache_io import find_targets_dir
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

# resolve_model is provided by build_kernel.py runner cell on Kaggle
resolve_model = lambda hf_id: hf_id  # noqa: E731  # stub for local type checking

DOSES = [0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
DIRECTIONS = ["target", "anti", "gauge", "random"]
COLLAPSE_THRESHOLD = 0.05  # stop this direction at bridge < this


def _eval_bridge(
    mS, l_S, targets, token_set, samples, n_samples, device, pairing_ok
):
    """Bridge cosine on cached targets, input_ids-paired samples only."""
    bridges = []
    for si in range(min(n_samples, len(targets))):
        if not pairing_ok[si]:
            continue
        tgt = targets[si]
        s = samples[si]
        ids = s["input_ids"].to(device)
        attn = s["attention_mask"].to(device)
        J_S = compute_jacobian_for_tokens(mS, l_S, ids, attn, token_set)
        p_L = tgt["p"].to(device)
        J_S_p = softmax_jacobian(J_S, p_L)
        cos = (
            torch.nn.functional.cosine_similarity(
                J_S_p.float(), tgt["J_tgt_p"].to(device).float(), dim=1
            )
            .mean()
            .item()
        )
        bridges.append(cos)
    return sum(bridges) / len(bridges) if bridges else 0.0


def _eval_kl(mS, samples, orig_logits_cache, n_samples, device):
    """Evaluate KL(p_ed ‖ p_orig)."""
    kl_sum = 0.0
    for si in range(min(n_samples, len(orig_logits_cache))):
        s = samples[si]
        ids = s["input_ids"].to(device)
        attn = s["attention_mask"].to(device)
        with torch.no_grad():
            logits_ed = mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        p_ed = torch.softmax(logits_ed.float(), dim=-1)
        p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
        kl = (p_ed * torch.log(p_ed.clamp_min(1e-12) / (p_orig + 1e-12))).mean().item()
        kl_sum += kl
    return kl_sum / min(n_samples, len(orig_logits_cache))


def main(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
    dose: float | None = None,
    base_bridge: float | None = None,
):
    """Dose sweep.

    Args:
        dose: unused in v6 (fixed DOSES list). Kept for API compat.
        base_bridge: unused in v6 (self-measured baseline). Kept for API compat.
    """
    torch.manual_seed(42)
    start = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "xla":
        assert _XLA, "torch_xla not installed"
        device = _xm.xla_device()
        torch.set_num_threads(1)
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}")
    os.makedirs(results_dir, exist_ok=True)

    # -- Load small model -------------------------------------------------
    model_path = small_model
    from contextlib import suppress

    with suppress(NameError):
        model_path = resolve_model(small_model)  # noqa: F821
    print(f"Loading small model: {model_path}")
    mS = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)  # type: ignore[arg-type]
    mS.eval()

    n_S = mS.config.num_hidden_layers
    last_layer = n_S - 1
    print(f"Small model: {n_S} layers, last={last_layer}")

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if str(device) == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    # -- Load target cache ------------------------------------------------
    targets_dir = find_targets_dir(targets_dir)
    print(f"Using target cache at {targets_dir}")

    cache_files = sorted(
        f
        for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(
            f"No target cache .pt files found at {targets_dir}. Run Stage 0 first."
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

    # Last-layer pair
    target_pair = None
    for p in sorted(cache_by_pair):
        _, ls = p.split("_")
        if int(ls) == last_layer:
            target_pair = p
            break
    if target_pair is None:
        raise SystemExit(
            f"FATAL: no pair with l_S={last_layer} in {targets_dir} "
            f"(have: {sorted(cache_by_pair)}). Re-run stage 0 with the "
            f"last-layer pair included."
        )
    targets = cache_by_pair[target_pair]
    token_set = targets[0]["token_set"]
    print(f"Using pair {target_pair}, K={len(token_set)}")

    # -- input_ids pairing -------------------------------------------------
    pairing_ok: list[bool] = []
    for si in range(min(n_samples, len(targets))):
        t = targets[si]
        ids = samples[si]["input_ids"]
        ok = (
            t.get("input_ids") is not None
            and t["input_ids"].shape[0] == ids.shape[0]
            and bool((t["input_ids"].view(-1) == ids.view(-1)).all())
        )
        pairing_ok.append(ok)
    print(f"Pairing verified: {sum(pairing_ok)}/{len(pairing_ok)} samples")

    # -- Original weights / logits ------------------------------------------
    lm_head = mS.lm_head
    W_orig = lm_head.weight.data.clone()

    orig_logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for s in samples[:n_samples]:
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            orig_logits_cache.append(
                mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            )

    # -- Baseline: no edit -------------------------------------------------
    print("\n--- Baseline (ΔW = 0) ---")
    base_bridge = _eval_bridge(
        mS, last_layer, targets, token_set, samples, n_samples, device, pairing_ok
    )
    base_kl = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    print(f"  bridge={base_bridge:.4f}, KL={base_kl:.6f}")

    # -- Target ΔJ direction (sample 0, paired) -----------------------------
    s0 = samples[0]
    ids0 = s0["input_ids"].to(device)
    attn0 = s0["attention_mask"].to(device)
    tgt0 = targets[0]
    J_S_orig = compute_jacobian_for_tokens(mS, last_layer, ids0, attn0, token_set)
    p0 = tgt0["p"].to(device)
    J_S_orig_p = softmax_jacobian(J_S_orig, p0)
    J_tgt_p = tgt0["J_tgt_p"].to(device)
    delta_J = J_tgt_p - J_S_orig_p  # [K, D]
    delta_J_gauge = delta_J.mean(dim=0, keepdim=True)  # [1, D]
    print(f"‖ΔJ‖={delta_J.norm():.4f}, ‖ΔJ_gauge‖={delta_J_gauge.norm():.4f}")

    def build_dW(direction: str, scale: float) -> torch.Tensor:
        """Build ΔW [V, D] for a direction at given ‖·‖_F = scale."""
        V, D = W_orig.shape
        dW = torch.zeros(V, D, device=device)
        if direction == "target":
            rows = delta_J
        elif direction == "anti":
            rows = -delta_J
        elif direction == "gauge":
            rows = delta_J_gauge.expand(len(token_set), -1)
        elif direction == "random":
            rows = torch.randn(len(token_set), D, device=device)
        else:
            raise ValueError(f"unknown direction {direction}")
        for i, t in enumerate(token_set):
            dW[t] = rows[i]
        n = dW.norm()
        if n > 1e-8:
            dW *= scale / n
        return dW

    # -- Dose sweep ---------------------------------------------------------
    # Per-scenario checkpointing for TPU 2h-kill safety.
    all_scenarios: dict[str, dict] = {}
    collapsed: list[str] = []

    def save_partial() -> None:
        part = {
            "status": "partial" if len(all_scenarios) < len(DOSES) * len(DIRECTIONS) else "complete",
            "completed": sorted(all_scenarios.keys()),
            "baseline_bridge": base_bridge,
            "baseline_KL": base_kl,
            "scenarios": all_scenarios,
            "collapsed": collapsed,
            "target_pair": target_pair,
            "last_layer": last_layer,
            "K": len(token_set),
            "device": str(device),
        }
        with open(os.path.join(results_dir, "exp19_destruction.json"), "w") as f:
            json.dump(part, f, indent=2)

    save_partial()  # baseline committed

    for direction in DIRECTIONS:
        dir_collapsed = False
        for dose in DOSES:
            key = f"{direction}_{dose:g}"
            print(f"\n--- {key} (‖ΔW‖={dose}) ---")
            dW = build_dW(direction, dose)
            lm_head.weight.data.copy_(W_orig + dW)

            b = _eval_bridge(
                mS, last_layer, targets, token_set, samples, n_samples, device, pairing_ok
            )
            k = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
            all_scenarios[key] = {
                "bridge": b,
                "KL_ed_orig": k,
                "delta_bridge": b - base_bridge,
                "delta_KL": k - base_kl,
                "delta_norm": float(dW.norm()),
                "direction": direction,
                "dose": dose,
            }
            print(f"  bridge={b:.4f} (Δ={b - base_bridge:+.4f}), "
                  f"KL={k:.6f} (Δ={k - base_kl:+.2e}), ‖ΔW‖={dW.norm():.4f}")

            # Save ΔW for inference inspection
            pt_path = os.path.join(
                results_dir, f"exp19_dW_{key}.pt"
            )
            torch.save({
                "delta_W": dW.cpu(),
                "W_orig_norm": float(W_orig.norm()),
                "direction": direction,
                "dose": dose,
                "bridge": b,
                "base_bridge": base_bridge,
                "pair": target_pair,
            }, pt_path)

            save_partial()

            # Early stop: collapse
            if b < COLLAPSE_THRESHOLD:
                collapsed.append(key)
                dir_collapsed = True
                print(f"  COLLAPSE: bridge {b:.4f} < {COLLAPSE_THRESHOLD}, "
                      f"stopping {direction}")
                break

        if not dir_collapsed:
            print(f"  {direction}: no collapse across all doses")

    # Restore
    lm_head.weight.data.copy_(W_orig)

    # -- Summary ------------------------------------------------------------
    print("\n=== Summary ===")
    for direction in DIRECTIONS:
        print(f"  {direction}:")
        for dose in DOSES:
            key = f"{direction}_{dose:g}"
            if key in all_scenarios:
                sc = all_scenarios[key]
                print(f"    ‖ΔW‖={dose:>6g}: bridge={sc['bridge']:.4f} "
                      f"(Δ={sc['delta_bridge']:+.4f}), KL={sc['KL_ed_orig']:.4f}")
            else:
                print(f"    ‖ΔW‖={dose:>6g}: (skipped — collapsed earlier)")

    result = {
        "description": (
            "Dose sweep v6: bridge vs ‖ΔW‖_F (0.1/1/10/100/500/1000) × "
            "4 directions (target/anti/gauge/random), single pair 31_23, "
            "early-stop on collapse (bridge < 0.05). ΔW saved as .pt per "
            "scenario for inference inspection."
        ),
        "status": "complete",
        "doses": DOSES,
        "directions": DIRECTIONS,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "collapsed": collapsed,
        "baseline_bridge": base_bridge,
        "baseline_KL": base_kl,
        "scenarios": all_scenarios,
        "target_pair": target_pair,
        "pairing": {
            "mode": "input_ids",
            "verified": int(sum(pairing_ok)),
            "total": int(len(pairing_ok)),
        },
        "n_layers_S": n_S,
        "last_layer": last_layer,
        "K": len(token_set),
        "W_norm": float(W_orig.norm()),
        "device": str(device),
        "elapsed_seconds": time.time() - start,
        "params": {
            "small_model": small_model,
            "device": str(device),
            "max_length": max_length,
            "min_length": min_length,
            "targets_dir": targets_dir,
        },
    }
    path = os.path.join(results_dir, "exp19_destruction.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 19: dose sweep (v6)")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dose", type=float, default=None,
                        help="(unused in v6, kept for compat)")
    parser.add_argument("--base-bridge", type=float, default=None,
                        help="(unused in v6, kept for compat)")
    args = parser.parse_args()
    main(**vars(args))
