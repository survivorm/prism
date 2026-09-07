import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
# name any op XLA cannot lower in backward (printed to stderr by torch_xla debug)
os.environ["PT_XLA_DEBUG_LEVEL"] = "2"
import time, threading
T0 = time.time()
def ts(): return f"[{time.time()-T0:5.0f}s]"
def log(*a): print(ts(), *a, flush=True)

TIMEOUT = 600  # hard bound for the WHOLE run (fwd compile + bwd compiles)

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = xm.xla_device()
log("device:", dev)
M = "Qwen/Qwen3.5-0.8B"
m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).to(dev)
tok = AutoTokenizer.from_pretrained(M)
inp = tok(["the quick brown fox jumps over the lazy dog."], return_tensors="pt").to(dev)

for p in m.parameters():
    p.requires_grad_(True)


def bwd(train, tag):
    (m.train() if train else m.eval())
    out = m(**inp)
    xm.mark_step()
    log(f"{tag}: fwd OK (train={train})")
    t0 = time.time()
    out.logits.sum().backward()
    xm.mark_step()
    log(f"{tag}: BWD OK in {time.time()-t0:.0f}s")


def seq():
    # THE prism question: frozen-model (eval) Jacobian backward — no dropout, small graph.
    log("PROBE 1/2: EVAL backward (frozen-model Jacobian = what prism 16/17/18/19 need)")
    bwd(train=False, tag="EVAL")
    # completeness: training-mode backward (adds dropout -> larger reverse graph)
    log("PROBE 2/2: TRAIN backward (dropout path)")
    bwd(train=True, tag="TRAIN")
    log("RESULT: BOTH BACKWARDS COMPILED — TPU is viable for prism causal experiments")


def run(seq, TIMEOUT):
    done = False
    def w():
        nonlocal done
        try:
            seq()
        finally:
            done = True
    t = threading.Thread(target=w, daemon=True)
    t.start()
    for _ in range(TIMEOUT):
        if done:
            break
        time.sleep(1)
    else:
        print(f"\n=== HANG: stuck {TIMEOUT}s after last marker (see which PROBE n printed fwd OK) ===", flush=True)
        print("=== torch_xla HLO compile stalls on that backward graph ===", flush=True)
        os._exit(99)


run(seq, TIMEOUT)
