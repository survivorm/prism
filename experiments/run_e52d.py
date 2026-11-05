#!/usr/bin/env python3
"""E-52d runner: full-model RTN трассы (тест A/B честный: меняется и u)."""
import os, subprocess, sys
EXP = "/home/survivor/work/prism/experiments"
P = "/home/survivor/work/prism/.venv/bin/python"
JOBS = [  # (model_dir, tag, bits)
    ("/data/Qwen/Qwen3.5-0.8B", "q08_int4", 4),
    ("/data/Qwen/Qwen3.5-0.8B", "q08_int2", 2),
    ("/data/Qwen/Qwen3.5-4B",  "q4_int2",  2),
]
for md, tag, bits in JOBS:
    rd = f"../results/r52d_{tag}"
    os.makedirs(rd, exist_ok=True)
    with open(f"/tmp/e52d_{tag}_trace.log", "w") as lg:
        subprocess.run([P, "-u", "exp47b_gen_kinematics.py", "--model", md,
                        "--n-docs", "4", "--max-new", "160", "--rtn-bits", str(bits),
                        "--results-dir", rd], cwd=EXP, stdout=lg, stderr=subprocess.STDOUT)
    if not os.path.exists(rd + "/exp47b_gen_kinematics.npz"):
        print(tag, "trace FAILED", flush=True); continue
    # τ базовая (квантованная модель её же головой — npz g_norm) + RTN-поверх-RTN уже не нужен
    with open(f"/tmp/e52d_{tag}_tau.log", "w") as lg:
        subprocess.run([P, "-u", "exp52a_perturb_axis.py", "--model", "qwen", "--n-docs", "16",
                        "--families", "noise", "--model-dir", md,
                        "--npz", rd + "/exp47b_gen_kinematics.npz",
                        "--docs-json", rd + "/exp47b_gen_kinematics.json",
                        "--out", f"../results/exp52a_fullrtn_{tag}.json"],
                       cwd=EXP, stdout=lg, stderr=subprocess.STDOUT)
    print(tag, "done", flush=True)
print("CHAIN_DONE")
