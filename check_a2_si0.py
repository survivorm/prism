"""A2 follow-up: is the 4B flat distribution on si=0 real (not a bf16 artifact)?
Run 4B fp16 (CPU) on that sample, report top-5 probs. Also 0.8B top-5 for contrast.
"""

import torch
from transformers import AutoModelForCausalLM

device = "cpu"
tok = torch.load("results/targets/4_3_0.pt", weights_only=True)["input_ids"].to(device)
attn = torch.ones_like(tok)


def top5(m):
    with torch.no_grad():
        lg = m(input_ids=tok, attention_mask=attn).logits[0, -1].float()
    p = torch.softmax(lg, dim=0)
    top = p.topk(5).values.tolist()
    return top, (p > 0.01).sum().item()


m08 = AutoModelForCausalLM.from_pretrained(
    "/data/Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16
).to(device)  # type: ignore[arg-type]
m08.eval()
top, n_gt1 = top5(m08)
print(f"0.8B fp16: top5={[f'{x:.4f}' for x in top]}  n(p>1%)={n_gt1}")
del m08
torch.cuda.empty_cache()

m4 = AutoModelForCausalLM.from_pretrained(
    "/data/Qwen/Qwen3.5-4B", torch_dtype=torch.float16
).to(device)  # type: ignore[arg-type]
m4.eval()
top, n_gt1 = top5(m4)
print(f"4B   fp16: top5={[f'{x:.4f}' for x in top]}  n(p>1%)={n_gt1}")
