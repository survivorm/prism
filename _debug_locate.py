import torch
from transformers import AutoModelForCausalLM
import prism.jacobian as PJ

torch.set_num_threads(8)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32)
model.eval()
tm = model.model
ids = (torch.arange(32) % 100000).view(1, -1)
attn = torch.ones(1, 32, dtype=torch.long)
S = ids.shape[1]
n_layers = model.config.num_hidden_layers

# ONE full forward, capturing per-layer outputs via hooks
layer_out = {}
hooks = [tm.layers[li].register_forward_hook(lambda m, i, o, li=li: layer_out.__setitem__(li, o[:, -1, :].clone())) for li in range(n_layers)]
with torch.no_grad():
    o = model(ids, attention_mask=attn, output_hidden_states=True)
for h in hooks: h.remove()
true_logits = o.logits[0, -1].clone()

# Native incremental decode (all 24 layers, token 31)
with torch.no_grad():
    prefix = model(ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    native = model(ids[:, -1:], attention_mask=attn, past_key_values=prefix.past_key_values, use_cache=False)
print("native logits err:", (native.logits[0, -1] - true_logits).abs().max().item())

# Prism decoder pass for layer_idx=5, capturing its per-layer outputs
L = 5
cache = PJ._run_prefix_pass(model, ids, attn)
h = layer_out[L].clone().detach().view(1, 1, -1).requires_grad_(True)
prism_layer_out = {}
hooks2 = [tm.layers[li].register_forward_hook(lambda m, i, o, li=li: prism_layer_out.__setitem__(li, o[:, -1, :].clone())) for li in range(L + 1, n_layers)]
with torch.enable_grad():
    dl = PJ._run_decoder_pass(model, cache, h, S, L)
for hh in hooks2: hh.remove()
print(f"prism logits (L={L}) err:", (dl[0, -1] - true_logits).abs().max().item())
print("\nPer-layer output diffs (prism vs full-forward), token 31:")
for li in range(L + 1, n_layers):
    d = (prism_layer_out[li] - layer_out[li]).abs().max().item()
    rel = d / layer_out[li].abs().max().item()
    flag = "  <-- DIVERGES" if rel > 1e-4 else ""
    print(f"  layer {li:2d}: abs={d:.3e} rel={rel:.3e}{flag}")
