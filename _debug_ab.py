"""A/B/C test of prism's decoder loop with different mask setups."""
import os, copy
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
torch.set_num_threads(8)
torch.manual_seed(0)
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

def build_masks(tm, cc, text_pos, x, D, attn_mode, inputs_embeds):
    mk = dict(config=tm.config, inputs_embeds=inputs_embeds, attention_mask=attn_mode,
              past_key_values=cc, position_ids=text_pos)
    return {"full_attention": create_causal_mask(**mk),
            "linear_attention": create_recurrent_attention_mask(**mk)}

def run_pass(tm, model, cache, h_l, S, l, attn_mode, rebuild_per_layer, D):
    cc = copy.deepcopy(cache)
    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long)
    text_pos = pos[0]; rope_pos = pos[1:]
    pe = tm.rotary_emb(h_l, rope_pos)
    n_layers = model.config.num_hidden_layers
    x = h_l
    with torch.no_grad():
        if not rebuild_per_layer:
            masks = build_masks(tm, cc, text_pos, x, D, attn_mode, x)
        for i in range(l + 1, n_layers):
            lt = model.config.layer_types[i]
            if rebuild_per_layer:
                masks = build_masks(tm, cc, text_pos, x, D, attn_mode, x)
            x = tm.layers[i](x, position_embeddings=pe, attention_mask=masks[lt],
                             position_ids=text_pos, past_key_values=cc)
        return x[:, 0, :]

def main():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tm = model.model
    D = tm.config.hidden_size
    n_layers = tm.config.num_hidden_layers

    ids = torch.randint(0, 1000, (1, 32))
    attn = torch.ones_like(ids)
    S = ids.shape[1]
    l = 6

    with torch.no_grad():
        out = model(ids, attention_mask=attn, output_hidden_states=True)
        pout = model(ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    true_final = out.hidden_states[n_layers][:, -1, :]
    h_l = out.hidden_states[l + 1][:, -1:, :]
    scale = true_final.abs().max().item()

    def rel(final):
        return (final - true_final).abs().max().item() / scale

    # A: prism's exact setup (attn None, dummy-ish via x, no per-layer rebuild)
    fa = run_pass(tm, model, pout.past_key_values, h_l, S, l, None, False, D)
    print(f"A prism-style (attn=None, build once):        rel={rel(fa):.4e}")

    # B: attn=[1,S] full mask, build once
    fb = run_pass(tm, model, pout.past_key_values, h_l, S, l, attn, False, D)
    print(f"B attn=[1,S], build once:                     rel={rel(fb):.4e}")

    # C: attn=None, rebuild mask per layer (so each sees current cache len)
    fc = run_pass(tm, model, pout.past_key_values, h_l, S, l, None, True, D)
    print(f"C attn=None, rebuild per layer:               rel={rel(fc):.4e}")

    # D: attn=[1,S], rebuild per layer
    fd = run_pass(tm, model, pout.past_key_values, h_l, S, l, attn, True, D)
    print(f"D attn=[1,S], rebuild per layer:              rel={rel(fd):.4e}")

if __name__ == "__main__":
    main()
