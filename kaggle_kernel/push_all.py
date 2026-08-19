#!/usr/bin/env python3
"""Push one or more kernels to Kaggle.

Usage:
    uv run python3 push_all.py              # push nothing (default)
    uv run python3 push_all.py --exp exp13  # push kernel-exp13
    uv run python3 push_all.py --exp exp13 exp15  # push both
    uv run python3 push_all.py --all        # push all
"""

import argparse
import subprocess
import sys
import os

KAGGLE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(KAGGLE_DIR)

KERNELS = {
    "exp13": {
        "dir": "kernel-exp13",
        "slug": "prism-exp13-domain-esg",
        "id": "voroninmichael/prism-exp13-domain-esg",
        "title": "Prism: Domain-specific ESG",
    },
    "exp14": {
        "dir": "kernel-exp14",
        "slug": "prism-exp14-snr-pr",
        "id": "voroninmichael/prism-exp14-snr-pr",
        "title": "Prism: SNR & Participation Ratio",
    },
    "exp15": {
        "dir": "kernel-exp15",
        "slug": "prism-exp15-gdn-anchors",
        "id": "voroninmichael/prism-exp15-gdn-anchors",
        "title": "Prism: GDN Anchor Shifts",
    },
    "exp6": {
        "dir": "kernel-exp6",
        "slug": "prism-kl-baseline",
        "id": "voroninmichael/prism-kl-baseline",
        "title": "Prism: KL Baseline",
    },
    "exp16_stage0": {
        "dir": "kernel-exp16",
        "slug": "prism-exp16-stage0-targets",
        "id": "voroninmichael/prism-exp16-stage0-targets",
        "title": "Prism: Stage0 Targets",
        "exp_file": "experiments/exp16_stage0_targets.py",
    },
    "exp16_readout": {
        "dir": "kernel-exp16r",
        "slug": "prism-exp16-readout-pull",
        "id": "voroninmichael/prism-exp16-readout-pull",
        "title": "Prism: Readout Pull",
        "exp_file": "experiments/exp16_readout_pull.py",
        "sources": ["voroninmichael/prism-exp16-stage0-targets"],
    },
    "exp17": {
        "dir": "kernel-exp17",
        "slug": "prism-exp17-lora-distill",
        "id": "voroninmichael/prism-exp17-lora-distill",
        "title": "Prism: LoRA Distillation",
        "sources": ["voroninmichael/prism-exp16-stage0-targets"],
    },
    "exp18": {
        "dir": "kernel-exp18",
        "slug": "prism-exp18-logit-kd",
        "id": "voroninmichael/prism-exp18-logit-kd",
        "title": "Prism: Logit KD Control",
        "sources": ["voroninmichael/prism-exp16-stage0-targets"],
    },
    "exp19": {
        "dir": "kernel-exp19",
        "slug": "prism-exp19-destruction",
        "id": "voroninmichael/prism-exp19-destruction",
        "title": "Prism: Destruction Detector",
        "sources": [
            "voroninmichael/prism-exp16-stage0-targets",
            "voroninmichael/prism-exp16-readout-pull",
        ],
    },
}


def build(exp_name: str):
    meta = KERNELS[exp_name]  # type: ignore[index]
    meta_dict: dict = meta  # type: ignore[assignment]

    # Resolve exp file
    if "exp_file" in meta_dict:
        exp_path = os.path.join(KAGGLE_DIR, "..", "experiments", meta_dict["exp_file"])
    else:
        import glob as _glob

        exp_file_path = os.path.join(
            KAGGLE_DIR, "..", "experiments", f"{exp_name}_*.py"
        )
        matches = _glob.glob(exp_file_path)
        if not matches:
            print(f"Error: no file matches {exp_file_path}", flush=True)
            return
        exp_path = matches[0]

    kernel_dir = os.path.join(KAGGLE_DIR, meta_dict["dir"])  # type: ignore[index]
    subprocess.run(
        [
            sys.executable,
            os.path.join(KAGGLE_DIR, "build_kernel.py"),
            "--exp",
            exp_path,
            "--out",
            kernel_dir,
            "--id",
            meta_dict["id"],  # type: ignore[index]
            "--title",
            meta_dict["title"],  # type: ignore[index]
        ]
        + ["--sources"]
        + (meta_dict.get("sources", [])),  # type: ignore[attr-defined]
        check=True,
    )


def push(exp_name: str):
    meta = KERNELS[exp_name]  # type: ignore[index]
    kernel_dir = meta["dir"]  # type: ignore[index]
    subprocess.run(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            os.path.join(KAGGLE_DIR, kernel_dir),
        ],
        check=False,  # kaggle CLI is noisy
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=[])
    parser.add_argument("--all", action="store_true", help="Build and push all kernels")
    args = parser.parse_args()

    if args.all:
        targets = list(KERNELS.keys())
    else:
        targets = args.exp

    if not targets:
        print("No targets specified. Use --exp exp13 or --all")
        return

    for name in targets:
        if name not in KERNELS:
            print(f"Unknown kernel: {name}. Available: {list(KERNELS.keys())}")
            continue
        print(f"\n=== Building + pushing {name} ===")
        build(name)
        push(name)


if __name__ == "__main__":
    main()
