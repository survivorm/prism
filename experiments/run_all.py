"""Run all experiments.

Usage:
    uv run python -m experiments.run_all
    uv run python -m experiments.run_all --only exp1
    uv run python -m experiments.run_all --skip exp2,exp6
"""

import argparse
import sys

_EXP_MODULES = [
    "exp1_identity",
    "exp2_heatmap",
    "exp3_ablation",
    "exp4_u_sensitivity",
    "exp5_embedding_perm",
    "exp6_kl_baseline",
    "exp7_cka_baseline",
    "exp8_svd_ablation",
    "exp9_esg",
]


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--only", default=None, help="Run only these (comma-separated)")
    parser.add_argument("--skip", default=None, help="Skip these (comma-separated)")
    args = parser.parse_args()

    skip = [s.strip() for s in args.skip.split(",")] if args.skip else []

    if args.only:
        targets = [t.strip() for t in args.only.split(",")]
    else:
        targets = [e for e in _EXP_MODULES if e not in skip]

    ok = True
    for name in targets:
        module_name = f"experiments.{name}"
        print(f"\n{'=' * 60}")
        print(f"Running {name}...")
        print(f"{'=' * 60}")
        try:
            mod = __import__(module_name, fromlist=[name])
            mod.main()
        except Exception as e:
            print(f"  FAILED: {e}")
            ok = False

    print(f"\n{'=' * 60}")
    if ok:
        print("All experiments completed successfully")
    else:
        print("Some experiments failed — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
