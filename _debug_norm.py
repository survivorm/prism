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

with torch.no_grad():
    o = model(ids, attention_mask=attn)
true_logits = o.logits[0, -1].clone()

# Re-implement decoder pass manually with prints (L=5)
L = 5
with torch.no_grad():
    o2 = model(ids, attention_mask=attn, output_hidden_states=True)
# capture layer outputs
layer_out = {}
hooks = [tm.layers[li].register_forward_hook(lambda m, i, oo, li=li: layer_out.__setitem__(li, oo[:, -1, :].clone())) for li in range(tm.config.num_hidden_layers)]
with torch.no_grad():
    o3 = model(ids, attention_mask=attn)
for h in hooks: h.remove()

# Reference: lm_head(norm(layer23out))
with torch.no_grad():
    ref_normed = tm.norm(layer_out[23].view(1, 1, -1))
    ref_logits = model.lm_head(ref_normed[:, -1, :])
print("ref (lm_head(norm(layer23))) err:", (ref_logits[0] - true_logits).abs().max().item())

# Now manual decoder pass
cache = PJ._run_prefix_pass(model, ids, attn)
h = layer_out[L].clone().detach().view(1, 1, -1)
cache_clone = PJ._deep_clone_cache(cache)
pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
text_position_ids = pos[0]; rope_position_ids = pos[1:]
position_embeddings = tm.rotary_emb(h, rope_position_ids)
x = h
n_layers = model.config.num_hidden_layers
start = L + 1
print("n_layers (model.config):", n_layers, " tm.config.num_hidden_layers:", tm.config.num_hidden_layers)
masks = PJ._build_causal_masks(tm, text_position_ids, cache_clone, x)
with torch.no_grad():
    for i in range(start, n_layers):
        lt = model.config.layer_types[i]
        x = tm.layers[i](x, position_embeddings=position_embeddings, attention_mask=masks[lt], position_ids=text_position_ids, past_key_values=cache_clone)
x_pre_norm = x[:, -1, :].clone()
print("x_pre_norm vs layer23(full):", (x_pre_norm - layer_out[23]).abs().max().item())
x_normed = tm.norm(x)
print("x_normed vs ref_normed:", (x_normed[:, -1, :] - ref_normed[:, -1, :]).abs().max().item())
logits = model.lm_head(x_normed[:, -1, :])
print("manual decoder logits err:", (logits[0] - true_logits).abs().max().item())

# Compare with actual _run_decoder_pass
h2 = layer_out[L].clone().detach().view(1, 1, -1).requires_grad_(True)
with torch.enable_grad():
    dl = PJ._run_decoder_pass(model, cache, h2, S, L)
print("_run_decoder_pass logits err:", (dl[0, -1] - true_logits).abs().max().item())
