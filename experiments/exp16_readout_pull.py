"""Experiment 16: Readout-only pull — DOSE-RESPONSE curve (v2, 2026-08-31).

Edit **only** `lm_head`: W' = W + ΔW(θ), one unclamped Adam run.

Why v2 (the v1 design was voided, results/voided/): v1 swept α as a
one-sided trust-region cap (‖ΔW‖ ≤ α·‖W‖). At α ≥ 0.1 the cap never binds
(the 100-step Adam trajectory stays at ‖ΔW‖/‖W‖ ≈ 4.5%), so α carried no
information — three alphas produced bit-identical weights, and "α* = 0.01"
was an artifact of comparing trajectories of different lengths. A cap is a
budget, not a dose. v2 measures what the paper needs: the dose–response of
the readout pull on ONE unclamped run.

Dose = ‖ΔW‖_F (reported as α = ‖ΔW‖/‖W‖). Every `eval_every` steps we
record (dose, bridge, KL(ed‖orig)) → one curve instead of five
incomparable points.

α* (the knee) = the largest step t with Δbridge ≥ tol, where
Δbridge(t) = bridge(t) − bridge(t − eval_every). A pull that keeps buying
bridge at any cost has no knee and is reported as null (α* = None) —
honest, not a first-slope artifact.

Loss = mean over bridge-paired samples of (1 − cos(J_S_p, J_tgt_p))
     + kl_weight · KL(p_ed ‖ p_orig).  Target pair = last-layer pair
(31_23 in the current cache; asserted — a shallow target is a null target).

Cost: 1 forward + 1 J_norm rebuild (decoder-tail, h reuse) + backward per
sample per step. ~3 s/step on T4 → 100 steps ≈ 5 min.
"""

import argparse
import gc
import json
import os
import re
import time

import torch
import torch.nn.functional as F

from prism.calibration import load_calibration_data
from prism.cache_io import find_targets_dir
from prism.jacobian import compute_hidden_jacobian, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

# resolve_model is provided by build_kernel.py runner cell on Kaggle
resolve_model = lambda hf_id: hf_id  # noqa: E731  # stub for local type checking


def _gd_steps(device: str) -> int:
    return 100 if str(device).startswith(("cuda", "xla")) else 40


def _eval_every(device: str) -> int:
    return 10 if str(device).startswith(("cuda", "xla")) else 10


def main(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
    lr: float = 1e-3,
    kl_weight: float = 0.1,
    knee_tol: float = 0.001,
    samples_from_targets: bool = False,
):
    torch.manual_seed(42)
    if str(device).startswith(("cuda", "xla")):
        gc.collect()
        torch.cuda.empty_cache()
    start = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # -- Resolve target cache location ------------------------------------
    # FUSE-safe, layout-independent: preferred dir, else walk /kaggle/input
    # (prism.cache_io — isdir lies on the mount; listdir-only, any slug/depth).
    targets_dir = find_targets_dir(targets_dir)
    print(f"Using target cache at {targets_dir}")

    # -- Load target cache ------------------------------------------------
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
    for v in cache_by_pair.values():
        v.sort(key=lambda t: t["_si"])

    if samples_from_targets:
        # OFFLINE mode (no HF network, e.g. local smoke): the cached
        # input_ids ARE the calibration samples (stage 0 was built from
        # them). One synthetic sample per sample index, sharing the
        # target pair's ids (pairing trivially exact; offline smoke's
        # only purpose is mechanics, not science).
        t0 = list(cache_by_pair.values())[-1][0]
        n_samples = min(2 if not str(device).startswith(("cuda", "xla")) else 4,
                        len(list(cache_by_pair.values())[-1]))
        samples = [
            {
                "input_ids": t0["input_ids"].clone(),
                "attention_mask": torch.ones_like(t0["input_ids"]),
            }
            for _ in range(n_samples)
        ]
        print(f"  (offline: {n_samples} samples from target cache input_ids)")
    else:
        n_samples = 2 if not str(device).startswith(("cuda", "xla")) else 4
        print(f"Loading {n_samples} calibration samples...")
        samples = load_calibration_data(
            small_model,
            max_length=max_length,
            min_length=min_length,
            n_samples=n_samples,
        )

    # The target pair MUST be the last-layer pair: this experiment measures
    # the last-layer readout Jacobian (J_S = W[token_set] @ J_norm,
    # J_norm = ∂h_norm/∂h_last). A shallow target is a null target — matched
    # bridge ~0, indistinguishable from a random rotation. The v1 silent
    # fallback to 0_0 is exactly the bug that voided the v1 results: assert.
    target_pair = None
    for p in sorted(cache_by_pair):
        _, ls = p.split("_")
        if int(ls) == last_layer:
            target_pair = p
            break
    if target_pair is None:
        raise ValueError(
            f"No pair with l_S={last_layer} in {targets_dir} "
            f"(have: {sorted(cache_by_pair)}). Re-run stage 0 with the "
            f"last-layer pair included before this experiment."
        )
    targets = cache_by_pair[target_pair]
    token_set = targets[0]["token_set"]
    K = len(token_set)
    print(f"Using pair {target_pair}, {len(targets)} cached samples, K={K}")

    # -- Sample<->target pairing via cached input_ids -----------------------
    # A mismatch means the reloaded sample differs from the one the target
    # was computed on: that sample's bridge would be a random target, so its
    # bridge term is skipped (KL is unaffected — it uses the sample's own
    # orig logits). No silent position-based pairing.
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
        if not ok:
            print(
                f"  WARNING: sample {si} <-> target {target_pair}_{si} "
                f"input_ids mismatch — its bridge term is skipped"
            )
    print(f"Pairing verified: {sum(pairing_ok)}/{len(pairing_ok)} samples")

    # -- Original lm_head weights -----------------------------------------
    lm_head = mS.lm_head
    W_orig = lm_head.weight.data.clone()
    W_norm = W_orig.norm().item()

    # -- Pre-compute original logits on all samples (for KL leash) ---------
    orig_logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for s in samples[:n_samples]:
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            logits = mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            orig_logits_cache.append(logits)
    print("Cached original logits for KL leash")

    # -- Baseline battery (ΔW = 0) — the null reference --------------------
    def battery() -> dict[str, float]:
        bridge_vals: list[float] = []
        with torch.no_grad():
            for si in range(min(n_samples, len(targets))):
                if not pairing_ok[si]:
                    continue
                tgt = targets[si]
                s = samples[si]
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                out = mS(
                    input_ids=ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=False,
                )
                h_s = out.hidden_states[last_layer + 1]
                J_norm = compute_hidden_jacobian(
                    mS, last_layer, ids, attn, h=h_s
                )
                J_S_ed = lm_head.weight[token_set] @ J_norm
                p_L = tgt["p"].to(device)
                J_S_ed_p = softmax_jacobian(J_S_ed, p_L)
                cos = (
                    torch.nn.functional.cosine_similarity(
                        J_S_ed_p.float(), tgt["J_tgt_p"].to(device).float(), dim=1
                    )
                    .mean()
                    .item()
                )
                bridge_vals.append(cos)
        kl_sum = 0.0
        with torch.no_grad():
            for si in range(min(n_samples, len(orig_logits_cache))):
                s = samples[si]
                ids = s["input_ids"].to(device)
                attn_s = s["attention_mask"].to(device)
                logits_ed = mS(input_ids=ids, attention_mask=attn_s).logits[
                    :, -1, :
                ]
                kl = F.kl_div(
                    F.log_softmax(orig_logits_cache[si].float(), dim=-1),
                    F.log_softmax(logits_ed.float(), dim=-1),
                    log_target=True,  # NaN without it (Kaggle run 2026-08-23)
                    reduction="batchmean",
                )
                kl_sum += kl.item()
        return {
            "bridge": sum(bridge_vals) / len(bridge_vals) if bridge_vals else 0.0,
            "KL": kl_sum / min(n_samples, len(orig_logits_cache)),
        }

    baseline = battery()
    print(
        f"Baseline (ΔW=0): bridge={baseline['bridge']:.4f}, "
        f"KL={baseline['KL']:.6f}"
    )

    # -- Single unclamped Adam run with per-eval dose recording ------------
    # The old trust-region buffer block is GONE on purpose: no cap, so
    # ‖ΔW‖ is a real, unbounded dose. (No allocator churn either: the
    # in-place sub() buffer existed only to apply the clamp.)
    steps = _gd_steps(device)
    every = _eval_every(device)
    optimizer = torch.optim.Adam([lm_head.weight], lr=lr)

    curve: list[dict] = [
        {"step": 0, "dose": 0.0, **baseline}
    ]

    t0 = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        total_loss_val = 0.0

        for si in range(min(n_samples, len(targets))):
            s = samples[si]
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            tgt = targets[si]

            # KL leash: true KL(p_ed || p_orig) over the full vocab.
            # log_target=True is MANDATORY: without it PyTorch computes
            # target*(log(target)-input) with target already a log-prob
            # (<=0) -> log(negative) = NaN from step 1.
            out = mS(
                input_ids=ids,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            kl_loss = F.kl_div(
                F.log_softmax(orig_logits_cache[si].float(), dim=-1),
                F.log_softmax(out.logits.float(), dim=-1),
                log_target=True,
                reduction="batchmean",
            )

            if pairing_ok[si]:
                # Fresh per-sample J_norm, differentiable through lm_head.
                h_s = out.hidden_states[last_layer + 1]
                J_norm = compute_hidden_jacobian(
                    mS, last_layer, ids, attn, h=h_s
                )
                J_S = lm_head.weight[token_set] @ J_norm  # [K, D]
                p_L = tgt["p"].to(device)
                J_S_p = softmax_jacobian(J_S, p_L)
                cos_sim = torch.nn.functional.cosine_similarity(
                    J_S_p.float(), tgt["J_tgt_p"].to(device).float(), dim=1
                )
                loss = (1.0 - cos_sim).mean() + kl_weight * kl_loss
            else:
                loss = kl_weight * kl_loss
            total_loss_val += loss.item()
            loss.backward()

        optimizer.step()

        if (step + 1) % 5 == 0:
            print(
                f"  step {step + 1}/{steps}: loss={total_loss_val:.4f} "
                f"({(time.time() - t0) / ((step + 1) // 5 + 1):.1f}s/eval-5)"
            )

        if (step + 1) % every == 0 or step + 1 == steps:
            dose = (lm_head.weight.data - W_orig).norm().item()
            b = battery()
            curve.append(
                {"step": step + 1, "dose": dose, **b}
            )
            print(
                f"  [eval] step {step + 1}: dose ‖ΔW‖={dose:.4f} "
                f"(α={dose / W_norm:.4f}), bridge={b['bridge']:.4f}, "
                f"KL={b['KL']:.6f}"
            )

    del optimizer
    gc.collect()
    if str(device).startswith(("cuda", "xla")):
        torch.cuda.empty_cache()

    # -- α* (the knee): largest step t with Δbridge ≥ knee_tol -------------
    # Δbridge(t) = bridge(t) − bridge(t − eval_every) on the same curve —
    # a genuine dose-derivative, unlike v1's cross-α differences (which
    # compared trajectories of different lengths). If every step improves
    # past tol, no knee exists: report None (honest null), not the first
    # slope.
    alpha_star: float | None = None
    knee_step: int | None = None
    for i in range(1, len(curve)):
        db = curve[i]["bridge"] - curve[i - 1]["bridge"]
        if db >= knee_tol:
            alpha_star = curve[i]["dose"] / W_norm
            knee_step = curve[i]["step"]
    if alpha_star is None:
        print(f"\nα* = null (no knee: bridge kept improving ≥ {knee_tol} "
              f"per eval window to the end)")
    else:
        print(f"\nα* (knee) = {alpha_star:.4f} at step {knee_step}")

    # -- Save results ------------------------------------------------------
    result = {
        "description": (
            "Readout-only pull as a DOSE-RESPONSE curve (v2). One unclamped "
            "Adam run on lm_head: loss = (1-cos(J_S_p, J_tgt_p)) per "
            "input_ids-paired sample + 0.1·KL(p_ed‖p_orig). Every 10 steps: "
            "dose ‖ΔW‖_F (α = ‖ΔW‖/‖W‖), bridge (fresh J_norm via h-reuse), "
            "true KL over the full vocab. Baseline at ΔW=0 included. α* = "
            "largest step where the per-window bridge gain ≥ knee_tol "
            "(None if no knee). v1 (alpha sweep as trust-region caps) was "
            "VOID: caps ≥0.1 never bind, α was not a dose, and its 'α*=0.01' "
            "was a first-slope artifact. v2 fixes the dose definition. "
            "Target pair is the last-layer pair (asserted)."
        ),
        "curve": curve,
        "baseline": baseline,
        "alpha_star": alpha_star,
        "knee_step": knee_step,
        "knee_tol": knee_tol,
        "lr": lr,
        "kl_weight": kl_weight,
        "steps": steps,
        "eval_every": every,
        "W_norm": W_norm,
        "target_pair": target_pair,
        "pairing": {
            "mode": "input_ids",
            "verified": int(sum(pairing_ok)),
            "total": int(len(pairing_ok)),
        },
        "n_layers_S": n_S,
        "last_layer": last_layer,
        "K": K,
        "device": device,
        "elapsed_seconds": time.time() - start,
        "params": {
            "small_model": small_model,
            "device": device,
            "max_length": max_length,
            "min_length": min_length,
            "targets_dir": targets_dir,
        },
    }
    path = os.path.join(results_dir, "exp16_readout_pull.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 16: Readout-only pull (dose curve)")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--knee-tol", type=float, default=0.001)
    parser.add_argument("--samples-from-targets", action="store_true",
                        help="offline: use cached input_ids as calibration samples")
    args = parser.parse_args()
    main(**vars(args))
