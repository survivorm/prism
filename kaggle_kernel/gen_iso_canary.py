#!/usr/bin/env python3
"""Isolation canary: why does Qwen3.5 GDN backward hang under torch_xla on TPU v5e-8?

Each probe runs in a watchdog-bounded subprocess (os._exit on timeout) so one
hang never eats the slot. We vary XLA_FLAGS + train/eval to find a config that
compiles the backward, and capture PT_XLA_DEBUG 'Op(s) not lowered' to name the
offending op.
"""
import os, sys, time, threading

TPU = "Qwen/Qwen3.5-0.8B"

PROBES = [
    ("A_baseline_train", {"XLA_FLAGS": ""}, 1),
    ("B_train_no_hlo_opt", {"XLA_FLAGS": "--xla_optimize_hlo=false"}, 1),
    ("C_train_stablehlo", {"XLA_FLAGS": "--xla_use_stablehlo=True --xla_optimize_hlo=false"}, 1),
    ("D_train_noopt_smaller_graph", {"XLA_FLAGS": "--xla_optimize_hlo=false --xla_num_replication_threads=1"}, 1),
    ("E_eval_bwd", {"XLA_FLAGS": ""}, 0),
    ("F_train_dump_hlo", {"XLA_FLAGS": "--xla_optimize_hlo=false --xla_dump_to=/tmp/hlo"}, 1),
]

MAX = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROBES)


def run_probe(name, env_extra, train):
    env = dict(os.environ)
    env.update(env_extra)
    for k, v in env_extra.items():
        print(f"[{name}] {k}={v!r}", flush=True)
    env["PJRT_DEVICE"] = "TPU"
    # fresh interpreter: re-exec self with this probe selected
    args = [sys.executable, os.path.abspath(__file__), "--probe", name, str(train), "run"]
    p = __import__("subprocess").run(args, env=env)
    return p.returncode


def probe_body(name, train):
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = xm.xla_device()
    m = AutoModelForCausalLM.from_pretrained(TPU, torch_dtype=torch.float32).to(dev)
    tok = AutoTokenizer.from_pretrained(TPU)
    inp = tok(["the quick brown fox jumps over the lazy dog."], return_tensors="pt").to(dev)

    if train:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)
    else:
        m.eval()

    out = m(**inp)
    xm.mark_step()
    print(f"[{name}] fwd OK, training={train}", flush=True)
    t0 = time.time()
    out.logits.sum().backward()
    xm.mark_step()
    print(f"[{name}] BWD OK in {time.time()-t0:.0f}s", flush=True)


def watchdog(fn, secs):
    done = False
    def w():
        nonlocal done
        fn()
        done = True
        print(f"[{NAME}] PROBE COMPLETE", flush=True)
    t = threading.Thread(target=w, daemon=True)
    t.start()
    for _ in range(secs):
        if done:
            return
        time.sleep(1)
    print(f"[{NAME}] HANG ({secs}s) — stuck after last marker", flush=True)
    os._exit(99)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "menu"
    if mode == "--probe":
        name, train = sys.argv[2], sys.argv[3] == "1"
        global NAME
        NAME = name
        # debug level only on the dump probe to keep others fast
        env2 = {"PT_XLA_DEBUG_LEVEL": "2"} if "dump" in name else {}
        env = dict(os.environ); env.update(env2)
        os.environ.update(env2)
        watchdog(lambda: probe_body(name, train), 540)
    else:
        # menu: run probes A.. in order, each its own interpreter
        for i, (name, env_extra, train) in enumerate(PROBES[:MAX]):
            print(f"\n===== PROBE {i}: {name} =====", flush=True)
            run_probe(name, env_extra, train)
        print("\nALL PROBES DONE", flush=True)
