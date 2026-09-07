import os
os.environ["PJRT_DEVICE"] = "TPU"
import time, threading
T0 = time.time()
def ts(): return f"[{time.time()-T0:5.0f}s]"
def log(*a): print(ts(), *a, flush=True)

TIMEOUT = 600  # hard bound: a deadlock must not eat the whole TPU slot

# --- P1: TPU attach ---
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer
log("P1 attaching TPU...")
dev = xm.xla_device()
log("P1 device:", dev)
xm.mark_step()

M = "Qwen/Qwen3.5-0.8B"
log("loading", M, "(HLO graph compile on 1st forward)")
m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32).to(dev)
tok = AutoTokenizer.from_pretrained(M)
inp = tok(["the quick brown fox jumps over the lazy dog."], return_tensors="pt").to(dev)

out = m(**inp)
log("fwd1 OK, last logit=", round(float(out.logits[0,-1,-1]), 4))
xm.mark_step()

# --- fwd2 + backward under a watchdog ---
DONE = False
def fwd2_bwd():
    global DONE
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    out2 = m(**inp)
    log("fwd2(train,grad) OK")
    xm.mark_step()
    out2.logits.sum().backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    log("BWD OK, grad norm sum=", round(gn, 4))
    xm.mark_step()
    DONE = True
    print("BWD DONE", flush=True)

t = threading.Thread(target=fwd2_bwd, daemon=True)
t.start()
for _ in range(TIMEOUT):
    if DONE:
        break
    time.sleep(1)

if DONE:
    print("\n=== VERDICT: TPU OK — Qwen3.5 GDN fwd+bwd compile+run on v5e-8. "
          "exp17/18/19 can run on TPU (separate TPU quota). ===", flush=True)
else:
    print(f"\n=== VERDICT: HANG — stuck after the last marker above (> {TIMEOUT}s). "
          "XLA HLO compile of GDN fwd2/backward stalls on v5e-8.", flush=True)
    print("=== => TPU is a dead end for any backward (all causal exps 16/17/18/19). "
          "Fall back to GPU (30h/wk). ===", flush=True)
    os._exit(99)
