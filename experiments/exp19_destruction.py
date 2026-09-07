"""Experiment 19: Destruction scenarios at a controlled dose (v2, 2026-08-31).

Same experiment, correct scaffolding. v1 (results/voided/) is void for
three independent reasons:
  1. target pair fell back to 0_0 — the GDN kl_only pair, a NULL target
     (cross-layer bridge ≈ 0 by construction) → every "bridge" value was
     noise on noise;
  2. the dose came from exp16 v1's α*=0.01 — a first-slope artifact of
     voided data, not a real knee;
  3. no no-edit baseline — with a null target you cannot even tell which
     scenario did the least damage.

v2 (this file):
  - target pair = LAST-layer pair (asserted, like exp16);
  - input_ids pairing (mismatched sample → its bridge term skipped,
    like exp16 — no silent position-based pairing);
  - dose ‖ΔW‖ = the knee from the exp16 v2 DOSE CURVE
    (results/exp16_readout_pull.json: curve[knee_step].dose);
    if exp16 v2 has not run yet: FATAL (no more silent default dose);
  - scenarios, all at the SAME dose ‖ΔW‖ (Frobenius):
      no_edit   — ΔW = 0 (THE baseline; everything is a delta from it);
      a_random  — random ΔW;
      b_target  — pull direction ΔJ (per-token rows), 1× dose;
      c_10x     — pull direction, 10× dose (overshoot);
      d_gauge   — gauge component of ΔJ only (common-mode rows);
      e_diff    — differential component of ΔJ only;
  - reported per scenario: bridge, KL(ed‖orig), Δbridge = bridge −
    bridge(no_edit), ΔKL = KL − KL(no_edit). The paper reads the DELTAS.
  - sanity: baseline bridge must match exp16's step-0 curve point
    (±0.05) — if not, the detector isn't calibrated; the run aborts
    with FATAL instead of publishing inconsistent numbers.
"""

import argparse
import json
import os
import re
import contextlib
import time

import torch
# --- XLA / TPU support (no-op on GPU/CPU; active when device == "xla") ---
try:
    import torch_xla.core.xla_model as _xm
    _XLA = True
except Exception:
    _xm = None
    _XLA = False


@contextlib.contextmanager
def _step_cm():
    """One optimizer step: xm.step() commits the XLA graph on TPU;
    nullcontext on GPU/CPU (existing behavior unchanged)."""
    if _XLA and str(_DEVICE).startswith("xla"):
        with _xm.step():
            yield
    else:
        with contextlib.nullcontext():
            yield


from prism.calibration import load_calibration_data
from prism.cache_io import find_targets_dir
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

# resolve_model is provided by build_kernel.py runner cell on Kaggle
resolve_model = lambda hf_id: hf_id  # noqa: E731  # stub for local type checking


def _find_exp16_knee(results_dir: str) -> tuple[float, float]:
    """Read the exp16 v2 dose curve. Returns (knee_dose, baseline_bridge).

    FATAL if missing: a destruction experiment without a measured dose
    is the exact v1 bug (silent default). No fallback."""
    cands = [os.path.join(results_dir, "exp16_readout_pull.json")]
    if os.path.exists("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            base = os.path.join("/kaggle/input", d)
            cands.append(os.path.join(base, "exp16_readout_pull.json"))
            cands.append(os.path.join(base, "results", "exp16_readout_pull.json"))
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        knee = d.get("knee_step")
        curve = d.get("curve")
        if knee is None or not curve:
            continue  # v1-format or incomplete file — not usable
        dose = None
        for pt in curve:
            if pt.get("step") == knee:
                dose = pt["dose"]
                break
        if dose is not None:
            base_bridge = next(
                (pt["bridge"] for pt in curve if pt["step"] == 0), 0.0
            )
            print(f"exp16 v2 curve: knee dose ‖ΔW‖={dose:.4f}, "
                  f"baseline bridge={base_bridge:.4f} ({os.path.basename(os.path.dirname(p))})")
            return dose, base_bridge
    raise SystemExit(
        "FATAL: exp16 v2 dose curve not found (need results/exp16_readout_pull.json "
        "with 'curve' and 'knee_step'). Run exp16 first — a destruction experiment "
        "without a measured knee dose is the v1 bug."
    )


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
        # p_ed clamped: 0 * log(0/...) = NaN otherwise
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
):
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

    # -- Dose from the exp16 v2 dose curve (FATAL if absent) ----------------
    knee_dose, exp16_base_bridge = _find_exp16_knee(results_dir)

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
    n_samples = 2 if device == "cpu" else 4
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

    # Last-layer pair — ASSERTED (v1's silent fallback to 0_0 was the null
    # target that voided v1).
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

    # -- input_ids pairing (v1 had none — position-based = broken) ---------
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
            print(f"  WARNING: sample {si} <-> target {target_pair}_{si} "
                  f"input_ids mismatch — its bridge term is skipped")
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

    # -- Baseline: no edit (THE reference) ---------------------------------
    print("\n--- Scenario: no_edit (baseline) ---")
    base_bridge = _eval_bridge(
        mS, last_layer, targets, token_set, samples, n_samples, device, pairing_ok
    )
    base_kl = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    print(f"  bridge={base_bridge:.4f}, KL={base_kl:.6f}")

    # Sanity: the detector must reproduce exp16's step-0 bridge.
    if abs(base_bridge - exp16_base_bridge) > 0.05:
        raise SystemExit(
            f"FATAL: baseline bridge {base_bridge:.4f} vs exp16 step-0 "
            f"{exp16_base_bridge:.4f} (Δ>0.05). Detector not calibrated — "
            f"aborting instead of publishing inconsistent numbers."
        )
    print(f"  matches exp16 step-0 bridge ({exp16_base_bridge:.4f}) — calibrated")

    # -- Target ΔJ direction (sample 0, paired) -----------------------------
    s0 = samples[0]
    ids0 = s0["input_ids"].to(device)
    attn0 = s0["attention_mask"].to(device)
    tgt0 = targets[0]
    J_S_orig = compute_jacobian_for_tokens(mS, last_layer, ids0, attn0, token_set)
    p0 = tgt0["p"].to(device)
    J_S_orig_p = softmax_jacobian(J_S_orig, p0)
    J_tgt_p = tgt0["J_tgt_p"].to(device)
    ΔJ = J_tgt_p - J_S_orig_p  # [K, D]
    ΔJ_gauge = ΔJ.mean(dim=0, keepdim=True)  # [1, D]
    ΔJ_diff = ΔJ - ΔJ_gauge  # [K, D]
    print(f"‖ΔJ‖={ΔJ.norm():.4f}, ‖ΔJ_gauge‖={ΔJ_gauge.norm():.4f}, "
          f"‖ΔJ_diff‖={ΔJ_diff.norm():.4f}")

    def build_ΔW(rows: torch.Tensor, scale: float) -> torch.Tensor:
        """Rows [K, D] → sparse ΔW [V, D] at tokens, scaled to ‖·‖=scale."""
        V, D = W_orig.shape
        ΔW = torch.zeros(V, D, device=device)
        for i, t in enumerate(token_set):
            ΔW[t] = rows[i]
        n = ΔW.norm()
        if n > 1e-8:
            ΔW *= scale / n
        return ΔW

    # -- Scenarios (all at the knee dose; 10× for the overshoot) -----------
    print(f"\nDose ‖ΔW‖ = {knee_dose:.4f} (knee of the exp16 v2 dose curve)")
    scenarios = {"no_edit": {"bridge": base_bridge, "KL_ed_orig": base_kl,
                             "delta_norm": 0.0}}

    names = [
        ("a_random", lambda: torch.randn_like(W_orig)),
        ("b_target_1x", lambda: ΔJ),
        ("c_target_10x", None),  # built below (10× dose)
        ("d_gauge", lambda: ΔJ_gauge),
        ("e_differential", lambda: ΔJ_diff),
    ]
    for name, mk in names:
        if mk is None:
            continue
        print(f"\n--- Scenario ({name}) ---")
        ΔW = build_ΔW(mk(), knee_dose)
        lm_head.weight.data.copy_(W_orig + ΔW)
        b = _eval_bridge(
            mS, last_layer, targets, token_set, samples, n_samples, device, pairing_ok
        )
        k = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
        scenarios[name] = {"bridge": b, "KL_ed_orig": k,
                           "delta_norm": float(ΔW.norm())}
        print(f"  bridge={b:.4f} (Δ={b - base_bridge:+.4f}), "
              f"KL={k:.6f} (Δ={k - base_kl:+.2e}), ‖ΔW‖={ΔW.norm():.4f}")

    print("\n--- Scenario (c_target_10x) ---")
    ΔW10 = build_ΔW(ΔJ, 10 * knee_dose)
    lm_head.weight.data.copy_(W_orig + ΔW10)
    b = _eval_bridge(
        mS, last_layer, targets, token_set, samples, n_samples, device, pairing_ok
    )
    k = _eval_kl(mS, samples, orig_logits_cache, n_samples, device)
    scenarios["c_target_10x"] = {"bridge": b, "KL_ed_orig": k,
                                 "delta_norm": float(ΔW10.norm())}
    print(f"  bridge={b:.4f} (Δ={b - base_bridge:+.4f}), "
          f"KL={k:.6f} (Δ={k - base_kl:+.2e}), ‖ΔW‖={ΔW10.norm():.4f}")

    lm_head.weight.data.copy_(W_orig)

    # -- Report: deltas from the no-edit baseline --------------------------
    for name, sc in scenarios.items():
        sc["delta_bridge"] = sc["bridge"] - base_bridge
        sc["delta_KL"] = sc["KL_ed_orig"] - base_kl
    ranking = sorted(
        (n for n in scenarios if n != "no_edit"),
        key=lambda n: scenarios[n]["delta_bridge"],
    )
    print("\nRanking by bridge damage (most → least):")
    for n in ranking:
        print(f"  {n}: Δbridge={scenarios[n]['delta_bridge']:+.4f}, "
              f"ΔKL={scenarios[n]['delta_KL']:+.2e}")

    result = {
        "description": (
            "Destruction scenarios at a controlled dose (v2). Dose ‖ΔW‖ = "
            "the knee of the exp16 v2 dose curve (FATAL if absent — no "
            "silent default). Scenarios, all at equal dose except the 10× "
            "overshoot: random ΔW; pull direction ΔJ (per-token rows); "
            "10× overshoot; gauge component of ΔJ; differential component "
            "of ΔJ. Plus THE no-edit baseline. Target pair = last-layer "
            "pair (asserted; v1's silent fallback to 0_0 was a null "
            "target that voided v1). input_ids pairing (v1 had none). "
            "All quantities are reported as deltas from no_edit; the "
            "baseline bridge must match exp16's step-0 curve point ±0.05 "
            "or the run aborts (detector calibration check)."
        ),
        "dose_norm": knee_dose,
        "exp16_baseline_bridge": exp16_base_bridge,
        "baseline_bridge": base_bridge,
        "baseline_KL": base_kl,
        "scenarios": scenarios,
        "ranking_by_damage": ranking,
        "target_pair": target_pair,
        "pairing": {
            "mode": "input_ids",
            "verified": int(sum(pairing_ok)),
            "total": int(len(pairing_ok)),
        },
        "n_layers_S": n_S,
        "last_layer": last_layer,
        "K": len(token_set),
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
    path = os.path.join(results_dir, "exp19_destruction.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 19: destruction scenarios (v2)")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
