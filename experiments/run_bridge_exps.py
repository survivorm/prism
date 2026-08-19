"""Runner: Execute bridge experiments in order (§9 of exps_bridge.md).

Order: Stage0 → 16 → 19 → 17 → 18

Produces final comparison table: scenario × battery metrics + α* + Δbridge/ΔKL ratio.
"""

import argparse
import json
import os
import time


def run(name: str, module: str, args: list[str], results_dir: str):
    """Run a single experiment by importing and calling main()."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(f"experiments.{module}")
        # Build namespace with defaults
        ns = {
            "device": "auto",
            "max_length": 64,
            "min_length": 32,
            "targets_dir": os.path.join(results_dir, "targets"),
            "results_dir": results_dir,
        }
        # Parse extra args
        for i in range(0, len(args) - 1, 2):
            key = args[i].lstrip("-").replace("-", "_")
            ns[key] = args[i + 1]
        mod.main(**ns)
        print(f"\n✓ {name} completed in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        print(f"\n✗ {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_table(results_dir: str):
    """Print final comparison table from all experiment results."""
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    data = {}
    for fname in ["exp16_readout_pull.json", "exp17_lora_distill.json",
                   "exp18_logit_kd.json", "exp19_destruction.json"]:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data[fname] = json.load(f)

    # Exp16: α* and per-alpha results
    if "exp16_readout_pull.json" in data:
        d = data["exp16_readout_pull.json"]
        print("\nExp16 — Readout-only pull")
        print(f"  α* (elasticity limit): {d['alpha_star']}")
        print(f"  {'Alpha':>8} {'Bridge':>10} {'KL_ed_orig':>12} {'‖ΔW‖/‖W‖':>12}")
        for alpha, vals in d.get("per_alpha", {}).items():
            print(f"  {alpha:>8} {vals.get('bridge_mean', 0):>10.4f} "
                  f"{vals.get('KL_ed_orig', 0):>12.4f} "
                  f"{vals.get('delta_W_frac', 0):>12.4f}")

    # Exp17: per-layer LoRA
    if "exp17_lora_distill.json" in data:
        d = data["exp17_lora_distill.json"]
        print("\nExp17 — Per-layer LoRA distillation")
        for order_name, order_data in d.get("per_order", {}).items():
            print(f"  Order: {order_name}")
            print(f"  {'Pair':>12} {'Bridge':>10} {'KL_ed_orig':>12}")
            for layer in order_data.get("per_layer", []):
                print(f"  {layer.get('pair', '?'):>12} "
                      f"{layer.get('bridge_mean', 0):>10.4f} "
                      f"{layer.get('KL_ed_orig', 0):>12.4f}")

    # Exp18: logit-KD control
    if "exp18_logit_kd.json" in data:
        d = data["exp18_logit_kd.json"]
        print("\nExp18 — Logit-KD control")
        print(f"  {'Pair':>12} {'Bridge':>10} {'KL_ed_orig':>12}")
        for layer in d.get("per_layer", []):
            print(f"  {layer.get('pair', '?'):>12} "
                  f"{layer.get('bridge_mean', 0):>10.4f} "
                  f"{layer.get('KL_ed_orig', 0):>12.4f}")
        comp = d.get("comparison_with_exp17")
        if comp:
            print("\n  Δbridge/ΔKL ratio:")
            print(f"    Exp17 (bridge): {comp.get('ratio_17_bridge_per_KL', 'N/A'):.4f}")
            print(f"    Exp18 (KD):     {comp.get('ratio_18_bridge_per_KL', 'N/A'):.4f}")

    # Exp19: destruction detector
    if "exp19_destruction.json" in data:
        d = data["exp19_destruction.json"]
        print("\nExp19 — Destruction detector calibration")
        print(f"  α* = {d.get('alpha_star', '?')}")
        print(f"  {'Scenario':<25} {'Bridge':>10} {'KL_ed_orig':>12} {'‖ΔW‖_F':>12}")
        for name, vals in d.get("scenarios", {}).items():
            print(f"  {name:<25} {vals.get('bridge', 0):>10.4f} "
                  f"{vals.get('KL_ed_orig', 0):>12.4f} "
                  f"{vals.get('delta_norm', 0):>12.4f}")
        ranking = d.get("ranking_by_bridge", [])
        print(f"\n  Ranking (best bridge first): {' → '.join(ranking)}")

    print(f"\n{'='*70}")
    print("Interpretation guide:")
    print("  (1) pull improves bridge AND behavior → sensitivity distillation as new KD")
    print("  (2) bridge drops, behavior flat → problem is in representations, not readout")
    print("  (3) breaks at small α → readout tightly coupled to representations")
    print(f"{'='*70}")


def main(
    results_dir: str = "results",
    device: str = "auto",
    run_stage0: bool = True,
    run_exp16: bool = True,
    run_exp17: bool = True,
    run_exp18: bool = True,
    run_exp19: bool = True,
    max_length: int = 64,
    min_length: int = 32,
):
    torch_manual = __import__("torch")
    torch_manual.manual_seed(42)
    if device == "cpu":
        torch_manual.set_num_threads(4)

    start = time.time()
    common = [
        "--device", device,
        "--max-length", str(max_length),
        "--min-length", str(min_length),
    ]

    if run_stage0:
        if not run("Stage 0", "exp16_stage0_targets", common, results_dir):
            print("Stage 0 failed — cannot proceed.")
            return

    if run_exp16:
        run("Exp16", "exp16_readout_pull", common, results_dir)

    if run_exp19:
        run("Exp19", "exp19_destruction", common, results_dir)

    if run_exp17:
        run("Exp17", "exp17_lora_distill", common, results_dir)

    if run_exp18:
        run("Exp18", "exp18_logit_kd", common, results_dir)

    print_final_table(results_dir)

    elapsed = time.time() - start
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bridge experiments (Stage0→16→19→17→18)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-stage0", action="store_true", help="Skip Stage 0")
    parser.add_argument("--skip-exp16", action="store_true")
    parser.add_argument("--skip-exp17", action="store_true")
    parser.add_argument("--skip-exp18", action="store_true")
    parser.add_argument("--skip-exp19", action="store_true")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    args = parser.parse_args()

    main(
        results_dir=args.results_dir,
        device=args.device,
        run_stage0=not args.skip_stage0,
        run_exp16=not args.skip_exp16,
        run_exp17=not args.skip_exp17,
        run_exp18=not args.skip_exp18,
        run_exp19=not args.skip_exp19,
        max_length=args.max_length,
        min_length=args.min_length,
    )
