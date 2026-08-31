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
        "sources": ["voroninmichael/prism-exp13-domain-esg"],
    },
    "exp14": {
        "dir": "kernel-exp14",
        "slug": "prism-exp14-snr-pr",
        "id": "voroninmichael/prism-exp14-snr-pr",
        "title": "Prism: SNR & Participation Ratio",
        "sources": ["voroninmichael/prism-exp14-snr-pr"],
    },
    "exp15": {
        "dir": "kernel-exp15",
        "slug": "prism-exp15-gdn-anchors",
        "id": "voroninmichael/prism-exp15-gdn-anchors",
        "title": "Prism: GDN Anchor Shifts",
        "sources": ["voroninmichael/prism-exp15-gdn-anchors"],
    },
    "exp6": {
        "dir": "kernel-exp6",
        "slug": "prism-kl-baseline",
        "id": "voroninmichael/prism-kl-baseline",
        "title": "Prism: KL Baseline",
        "sources": ["voroninmichael/prism-kl-baseline"],
    },
    "exp16_stage0": {
        "dir": "kernel-exp16",
        # Canonical slug on Kaggle (title-derived). The old slug
        # 'prism-exp16-stage0-targets' is a stale handle: `kernels pull`->404,
        # not in `kaggle kernels list`, rejected as a kernel source by the
        # server. Only the legacy `kernels output` path still resolves it.
        "slug": "exp16-stage-0-target-jacobian-cache",
        "id": "voroninmichael/exp16-stage-0-target-jacobian-cache",
        "title": "Exp16: Stage 0 - Target Jacobian Cache",
        "exp_file": "experiments/exp16_stage0_targets.py",
        # Producer: no kernel source (models via HF, targets computed fresh).
        "sources": [],
    },
    "exp16_readout": {
        "dir": "kernel-exp16r",
        "slug": "prism-readout-pull",
        "id": "voroninmichael/prism-readout-pull",
        "title": "Prism: Readout Pull",
        "exp_file": "experiments/exp16_readout_pull.py",
        "sources": ["voroninmichael/exp16-stage-0-target-jacobian-cache"],
        # Dataset fallback: targets also as a dataset (kernel-outputs mount
        # of the stage0 kernel has been unreliable — new slug never mounted).
        "datasets": ["voroninmichael/prism-exp16-targets"],
    },
    "exp17": {
        "dir": "kernel-exp17",
        # Live kernel slug: title 'Prism: LoRA Distillation' resolves to
        # prism-lora-distillation (kaggle derives the handle from the title).
        # The old prism-exp17-lora-distill id 409-conflicts (title != id).
        "slug": "prism-lora-distillation",
        "id": "voroninmichael/prism-lora-distillation",
        "title": "Prism: LoRA Distillation",
        # Explicit: glob exp17_*.py now also matches exp17_grad_sanity.py.
        "exp_file": "experiments/exp17_lora_distill.py",
        "sources": ["voroninmichael/exp16-stage-0-target-jacobian-cache"],
        "datasets": ["voroninmichael/prism-exp16-targets"],
    },
    "exp16_4_3": {
        # B2 mini-kernel: exp17 v23 on a single pair (4,3) + finite-diff
        # spot check. 0.8B student (fp32 J chain); 4B targets come from
        # the stage0 cache (4B never loaded).
        "dir": "kernel-exp16_4_3",
        # Actual Kaggle slug is title-derived: prism-exp16-4-3-v23-single-pair-fd-check
        # (registry id must match it or re-push 409-conflicts, like exp17 did).
        "slug": "prism-exp16-4-3-v23-single-pair-fd-check",
        "id": "voroninmichael/prism-exp16-4-3-v23-single-pair-fd-check",
        "title": "Prism: Exp16 4_3 (v23 single pair + FD check)",
        "exp_file": "experiments/exp17_lora_distill.py",
        "sources": ["voroninmichael/exp16-stage-0-target-jacobian-cache"],
        "datasets": ["voroninmichael/prism-exp16-targets"],
        "runner_extra": (
            "if 'pairs' in params: kwargs['pairs'] = '4_3'\n"
            "if 'steps' in params: kwargs['steps'] = 300\n"
            "if 'fd_check' in params: kwargs['fd_check'] = 0  # v23.2b: FD off-kernel (OOM-killed T4); run locally\n"
        ),
    },
    "exp18": {
        "dir": "kernel-exp18",
        "slug": "prism-exp18-logit-kd",
        "id": "voroninmichael/prism-exp18-logit-kd",
        "title": "Prism: Logit KD Control",
        "exp_file": "experiments/exp18_logit_kd.py",
        "sources": ["voroninmichael/exp16-stage-0-target-jacobian-cache"],
        "datasets": ["voroninmichael/prism-exp16-targets"],
    },
    "exp19": {
        "dir": "kernel-exp19",
        "slug": "prism-exp19-destruction",
        "id": "voroninmichael/prism-exp19-destruction",
        "title": "Prism: Destruction Detector",
        "sources": [
            "voroninmichael/exp16-stage-0-target-jacobian-cache",
            "voroninmichael/prism-readout-pull",
        ],
        "datasets": ["voroninmichael/prism-exp16-targets"],
    },
}


def build(exp_name: str):
    meta = KERNELS[exp_name]  # type: ignore[index]
    meta_dict: dict = meta  # type: ignore[assignment]

    # Resolve exp file
    if "exp_file" in meta_dict:
        exp_path = os.path.join(KAGGLE_DIR, "..", meta_dict["exp_file"])
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
    cmd = [
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
    cmd += ["--sources"] + meta_dict.get("sources", [])  # type: ignore[arg-type]
    cmd += ["--datasets"] + meta_dict.get("datasets", [])  # type: ignore[arg-type]
    if "runner_extra" in meta_dict:
        cmd += ["--runner-extra", meta_dict["runner_extra"]]  # type: ignore[index]
    subprocess.run(cmd, check=True)


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
