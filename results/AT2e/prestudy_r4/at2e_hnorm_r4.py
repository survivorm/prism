"""Canonical ‖h_t‖ measurement (r4, N-new-2): (L11,L12) x (t32,t96) x 4 docs, ONE forward per doc.
Convention: L = output of model.model.layers[L] = hidden_states[L+1] (hidden_states[0] = embed).
r1 table (at2e_l11_hnorm.py) used hidden_states[L] = output of layer L-1 — off by one.
Also cross-checks the forward-hook convention (should match to ~1e-6).
"""
import sys; sys.modules["fla"] = None
import os
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
import torch, numpy as np
from transformers import AutoModelForCausalLM
torch.set_num_threads(4)
model = AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32,
                                            attn_implementation="eager")
model.eval()
pool = torch.load("/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt", map_location="cpu",
                  weights_only=False)
out = {}
for dom in ("wiki", "math", "code", "chat"):
    toks = np.asarray(pool["ids"][dom][0], dtype=np.int64)
    ids = torch.as_tensor(toks[None])
    hooks = {}
    def make_hook(L):
        def h(m, inp, o):
            hooks[L] = o.detach()
        return h
    h11 = model.model.layers[11].register_forward_hook(make_hook(11))
    h12 = model.model.layers[12].register_forward_hook(make_hook(12))
    with torch.no_grad():
        hs = model(ids, attention_mask=torch.ones((1, 128), dtype=torch.long),
                   output_hidden_states=True).hidden_states
    row = {}
    for L in (11, 12):
        h = hs[L + 1][0]            # canonical: output of layer L
        hh = hooks[L][0]            # hook cross-check
        row[f"L{L}"] = {
            "t32": float(h[32].norm()), "t96": float(h[96].norm()),
            "hook_t32": float(hh[32].norm()), "hook_t96": float(hh[96].norm()),
            "hook_max_dev": float((h - hh).abs().max()),
        }
    out[dom] = row
    h11.remove(); h12.remove()
    print(f"{dom}/0: " + "  ".join(
        f"L{L}: t32={row[f'L{L}']['t32']:.4f} t96={row[f'L{L}']['t96']:.4f} "
        f"hook_dev={row[f'L{L}']['hook_max_dev']:.2e}" for L in (11, 12)), flush=True)
print("CANON_HNORM_DONE")
