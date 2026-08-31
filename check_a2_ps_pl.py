"""One-shot A2 gate: is p_S ~= p_L per pair? (0.8B fp32 vs 4B targets from cache)

For every results/targets/*.pt:
  p_S = softmax(0.8B logits[0, -1, token_set])   (same gauge as stage0)
  p_L = p_L_full[token_set] renormalized         (4B, from cache, fp32)
Report cosine + max|delta| per pair.
"""

import glob
import os

import torch
from transformers import AutoModelForCausalLM

MODEL = "/data/Qwen/Qwen3.5-0.8B"
device = os.environ.get("A2_DEVICE", "cpu")

mS = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, attn_implementation="eager"
)
mS = mS.to(device)  # type: ignore[arg-type]
mS.eval()  # type: ignore[union-attr]

files = sorted(glob.glob("results/targets/*.pt"))
assert files, "no targets cache"

for f in files:
    d = torch.load(f, weights_only=True)
    l_S, l_L = d["l_S"], d["l_L"]
    ids = d["input_ids"].to(device)
    attn = torch.ones_like(ids)
    token_set = d["token_set"]
    ts = torch.tensor(token_set, device=device)
    with torch.no_grad():
        logits = mS(input_ids=ids, attention_mask=attn).logits[0, -1]
    p_S = torch.softmax(logits[ts], dim=0).float()
    p_L = d["p_L_full"].float()[ts]
    p_L = p_L / p_L.sum()

    cos = torch.nn.functional.cosine_similarity(p_S, p_L, dim=0).item()
    maxd = (p_S - p_L).abs().max().item()
    print(f"{os.path.basename(f)}  p_S vs p_L: cos={cos:.6f}  max|d|={maxd:.5f}")
