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
    # --- Mandatory / optional GPU re-runs (audit GPU-queue items 1-6) ---
    # runner_extra = Python kwargs lines injected into runner_<exp>(...) by
    # build_kernel (it auto-passes model/device/results). NOT CLI flags.
    # id/slug are PROVISIONAL for never-pushed kernels: after the first
    # successful push kaggle derives the canonical slug from the title —
    # update id/slug to match (re-push 409-conflicts otherwise, see exp17).
    "exp6_l18_snr": {
        "dir": "kernel-exp6l18",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp6-l18-snr",
        "title": "Prism: exp6-l18-snr",
        "exp_file": "experiments/exp6_l18_snr.py",
        "runner_extra": "kwargs['n_samples'] = 8\nkwargs['K'] = 256\n",
        "note": "MANDATORY: data LOST (results/exp6_l18_snr.json absent from "
                "repo; rho=+2.94 unverifiable). L18 SNR/PR vs r, N=8, K=256, ~2h T4.",
    },
    "exp2_heatmap": {
        "dir": "kernel-exp2",
        "slug": "prism-exp2-heatmap",
        "id": "voroninmichael/prism-exp2-heatmap",
        "title": "Prism: exp2-heatmap",
        "exp_file": "experiments/exp2_heatmap.py",
        "runner_extra": "kwargs['n_samples'] = 8\nkwargs['K'] = 64\n",
        "note": "optional: full 32x24 heatmap, n=8, K=64 (was thin n=3/K=8).",
    },
    "exp4_k_sensitivity": {
        "dir": "kernel-exp4k",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp4-k-sensitivity",
        "title": "Prism: exp4-k-sensitivity",
        "exp_file": "experiments/exp4_k_sensitivity.py",
        "runner_extra": "kwargs['n_samples'] = 4\n",
        "note": "fixed: cross-bridge 4B->0.8B (v1 broken self-bridge) + K->256, n=4.",
    },
    "exp11c_defrag_bridge": {
        "dir": "kernel-exp11c",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp11c-defrag-bridge",
        "title": "Prism: exp11c-defrag-bridge",
        "exp_file": "experiments/exp11c_defrag_bridge.py",
        "runner_extra": "kwargs['n_samples'] = 8\n",
        "note": "n=8 (was n=1): bridge degradation vs defrag r.",
    },
    "exp10b_gauge_bridge": {
        "dir": "kernel-exp10b",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp10b-gauge-bridge",
        "title": "Prism: exp10b-gauge-bridge",
        "exp_file": "experiments/exp10b_gauge_bridge.py",
        "runner_extra": "kwargs['n_samples'] = 8\n",
        "note": "n=8 (was n=1): gauge/bridge.",
    },
    "exp4_top_p_tokens": {
        "dir": "kernel-exp4topp",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp4-top-p-tokens",
        "title": "Prism: exp4-top-p-tokens",
        "exp_file": "experiments/exp4_top_p_tokens.py",
        "runner_extra": "kwargs['n_samples'] = 4\n",
        "note": "optional: token-set-bias claim (never ran).",
    },
    "exp13": {
        "dir": "kernel-exp13",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp13-domain-esg",
        "title": "Prism: Domain-specific ESG",
        "sources": ["voroninmichael/prism-exp13-domain-esg"],
    },
    "exp14": {
        "dir": "kernel-exp14",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp14-snr-pr",
        "title": "Prism: SNR & Participation Ratio",
        "sources": ["voroninmichael/prism-exp14-snr-pr"],
    },
    "exp15": {
        "dir": "kernel-exp15",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp15-gdn-anchors",
        "title": "Prism: GDN Anchor Shifts",
        "sources": ["voroninmichael/prism-exp15-gdn-anchors"],
    },
    "exp6": {
        "dir": "kernel-exp6",
        "slug": "exp4-top-p-tokens",
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
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/exp16-stage-0-target-jacobian-cache",
        "title": "Exp16: Stage 0 - Target Jacobian Cache",
        "exp_file": "experiments/exp16_stage0_targets.py",
        # Producer: no kernel source (models via HF, targets computed fresh).
        "sources": [],
    },
    "exp16_readout": {
        "dir": "kernel-exp16r",
        "slug": "exp4-top-p-tokens",
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
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-lora-distillation",
        "title": "Prism: LoRA Distillation",
        # Explicit: glob exp17_*.py now also matches exp17_grad_sanity.py.
        "exp_file": "experiments/exp17_lora_distill.py",
        "sources": ["voroninmichael/exp16-stage-0-target-jacobian-cache"],
        "datasets": ["voroninmichael/prism-exp16-targets"],
        # Pending push (audit): pairs 4_3, 1000 steps (~9.2h, fits one 12h
        # quota); default order (1st + 2nd HVP). 300-step both-orders anchor
        # JSON already committed.
        "runner_extra": (
            "kwargs['pairs'] = '4_3'\n"
            "kwargs['steps'] = 1000\n"
        ),
    },
    "exp16_4_3": {
        # B2 mini-kernel: exp17 v23 on a single pair (4,3) + finite-diff
        # spot check. 0.8B student (fp32 J chain); 4B targets come from
        # the stage0 cache (4B never loaded).
        "dir": "kernel-exp16_4_3",
        # Actual Kaggle slug is title-derived: prism-exp16-4-3-v23-single-pair-fd-check
        # (registry id must match it or re-push 409-conflicts, like exp17 did).
        "slug": "exp4-top-p-tokens",
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
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp18-logit-kd",
        "title": "Prism: Logit KD Control",
        "exp_file": "experiments/exp18_logit_kd.py",
        "sources": ["voroninmichael/exp16-stage-0-target-jacobian-cache"],
        "datasets": ["voroninmichael/prism-exp16-targets"],
    },
    "exp01": {
        "dir": "kernel-exp01",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp01-coherence",
        "title": "Prism: exp01-coherence",
        "exp_file": "experiments/exp01_coherence.py",
        "runner_extra": "kwargs['n_samples'] = 8\nkwargs['K'] = 64\n",
        "note": "Exp 0.1 coherence length: C(delta) curves, 4 domains x 4 layers {6,12,18,22}, n=8/K=64. Kill-switch for ESG/Exp 0.2. GPU after quota reset (~09-14).",
    },
    "exp27": {
        "dir": "kernel-exp27",
        "slug": "exp4-top-p-tokens",
        "id": "voroninmichael/prism-exp27-noise-floor",
        "title": "Prism: exp27-noise-floor",
        "exp_file": "experiments/exp27_noise_floor.py",
        "runner_extra": "",
        "note": "Exp 27: noise-floor calibration (r_eff -> F) + S/K/position sweeps, layers {6,12,18,19,22,23} (19,23 full-attention), 256-token pool, nested U128. ~5.4k J, ~2.5-3h T4.",
    },
    "exp27b": {
        "dir": "kernel-exp27b",
        "slug": "exp27b-dense-profile",
        "id": "voroninmichael/prism-exp27b-dense-profile",
        "title": "Prism: exp27b-dense-profile",
        "exp_file": "experiments/exp27b_dense.py",
        "note": "Exp 27b: dense J-profile cache S=128 t=0..127, 24,576 J, per-layer ckpt files. Same exp27 pools/model; feeds exp28 post + future dense analysis.",
        # FULL: n_samples=8 x 4 domains x 128 t x 6 layers = 24,576 J (no smoke overrides).
        # ~3.3h T4 IF fla triton kernels work (v9: 0.48 s/J); ~90h at torch fallback — v10 FLA probe decides.
    },
    "exp19": {
        "dir": "kernel-exp19",
        "slug": "exp4-top-p-tokens",
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
        re_ = meta_dict["runner_extra"]
        if isinstance(re_, (list, tuple)):
            re_ = "".join(f"{x}\n" for x in re_)
        cmd += ["--runner-extra", re_]
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
