#!/usr/bin/env python3
"""
Generate + push a standalone TPU canary kernel (v5e-8, Kaggle free tier).

The earlier canary failed with "TPU initialization failed: No ... device found"
because it used the OLD notebook metadata (machine_shape=v3-8, phased out).
This one mirrors the proven-working kaggle-tpu-lab launcher: a SCRIPT kernel
with enable_tpu="true" and NO machine_shape (Kaggle defaults to v5e-8).

Probes, each wrapped so a failure reports WHERE it broke:
  P1  TPU device attach       (xm.xla_device)  -> metadata/quota issue
  P2  Qwen3.5-0.8B fwd + bwd  (GDN compile + autograd)  -> the real risk
  P4  Qwen3.5-4B fwd only     (GDN at scale; exp17 runs 4B frozen)
P1+P2 passing => TPU viable for exp17 LoRA (separate TPU quota).
"""
import json
import os
import subprocess
import sys

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canary-tpu")
os.makedirs(KDIR, exist_ok=True)

meta = {
    "id": "voroninmichael/prism-canary-tpu",
    "title": "prism-canary-tpu",
    "code_file": "canary_tpu.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "true",
    "enable_internet": "true",
}
with open(os.path.join(KDIR, "kernel-metadata.json"), "w") as f:
    json.dump(meta, f, indent=1)

script = r'''import os
os.environ["PJRT_DEVICE"] = "TPU"
import time
T0 = time.time()
def ts(): return f"[{time.time()-T0:5.0f}s]"

print("\n=== [P1] TPU device attach ===", flush=True)
try:
    import torch
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
    print(ts(), "device:", dev, flush=True)
    xm.mark_step()
    P1 = True
except Exception as e:
    import traceback; traceback.print_exc()
    P1 = False
    print("\n>>> VERDICT: FAIL at P1 (TPU attach) — metadata/quota, NOT GDN.", flush=True)
    raise SystemExit(0)

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = xm.xla_device()

def probe(tag, model, do_bwd):
    print(f"\n=== {tag} ===", flush=True)
    try:
        m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32).to(dev)
        m.eval()
        tok = AutoTokenizer.from_pretrained(model)
        print(ts(), f"  loaded: {model}  params={sum(p.numel() for p in m.parameters())/1e6:.0f}M", flush=True)
        inp = tok(["the quick brown fox jumps over the lazy dog."], return_tensors="pt").to(dev)
        out = m(**inp)               # 1st forward = XLA HLO compile (slow)
        print(ts(), f"  forward OK, last logit={out.logits[0,-1,-1].item():.4f}", flush=True)
        if do_bwd:
            m.train()
            for p in m.parameters():
                p.requires_grad_(True)
            out = m(**inp)
            out.logits.sum().backward()
            gx = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
            print(ts(), f"  backward OK, grad norm sum={gx:.4f}", flush=True)
        xm.mark_step()
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n>>> VERDICT: FAIL at {tag} ({e}).", flush=True)
        return False

p2 = probe("P2  Qwen3.5-0.8B forward+backward (GDN compile+autograd)", "Qwen/Qwen3.5-0.8B", do_bwd=True)
p4 = probe("P4  Qwen3.5-4B forward (GDN at scale)", "Qwen/Qwen3.5-4B", do_bwd=False)

print("\n" + "=" * 44, flush=True)
print("CANARY VERDICT:", flush=True)
print(f"  P1 TPU attach     : {'PASS' if P1 else 'FAIL'}", flush=True)
print(f"  P2 0.8B fwd+bwd   : {'PASS' if p2 else 'FAIL'}", flush=True)
print(f"  P4 4B  fwd        : {'PASS' if p4 else 'FAIL'}", flush=True)
if p2:
    print("  => TPU OK for Qwen3.5 GDN. exp17 LoRA can run on TPU.", flush=True)
else:
    print("  => TPU dead end for Qwen3.5 GDN. Fall back to GPU.", flush=True)
print("=" * 44, flush=True)
'''
with open(os.path.join(KDIR, "canary_tpu.py"), "w") as f:
    f.write(script)

print(f"canary-tpu/ ready: {sorted(os.listdir(KDIR))}")
print(f"metadata: {json.dumps(meta)}")

if "--push" in sys.argv:
    r = subprocess.run(["kaggle", "kernels", "push", "-p", KDIR],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(r.stderr[-800:])
