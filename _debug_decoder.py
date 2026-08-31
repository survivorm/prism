"""Independent, self-contained check of the decoder-pass reconstruction.

Does NOT import prism.jacobian internals for the comparison; only uses the
model directly. Confirms whether running layers l+1..end on h(l) with the
prefix cache reproduces the true final norm hidden state.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

torch.set_num_threads(8)
torch.manual_seed(0)


def main():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tm = model.model
    D = tm.config.hidden_size
    n_layers = tm.config.num_hidden_layers
    print("D", D, "n_layers", n_layers)
    print("layer_types[:12]", tm.config.layer_types[:12])

    ids = torch.randint(0, 1000, (1, 32))
    attn = torch.ones_like(ids)
    S = ids.shape[1]
    l = 6

    # ---- true full pass ------------------------------------------------
    with torch.no_grad():
        out = model(ids, attention_mask=attn, output_hidden_states=True)
    true_final = out.hidden_states[n_layers][:, -1, :]          # [D]
    true_normed = tm.norm(true_final)                            # [D]
    true_layer_out = [hs[:, -1, :] for hs in out.hidden_states]  # list[n_layers+1]

    # ---- prefix cache ---------------------------------------------------
    with torch.no_grad():
        pout = model(ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    cache = pout.past_key_values
    print("cache type", type(cache))
    print("cache layers", len(cache.layers))
    for i, lay in enumerate(cache.layers[:6]):
        print("  layer", i, type(lay).__name__)

    # ---- h at layer l from true pass ------------------------------------
    with torch.no_grad():
        h_l = out.hidden_states[l + 1][:, -1:, :]  # [1,1,D]

    # ---- manual decoder pass (independent reimplementation) -------------
    def clone_cache(cache):
        import copy
        return copy.deepcopy(cache)

    with torch.no_grad():
        cc = clone_cache(cache)
        pos = torch.full((4, 1, 1), S - 1, dtype=torch.long)
        text_pos = pos[0]
        rope_pos = pos[1:]
        pe = tm.rotary_emb(h_l, rope_pos)

        from transformers.masking_utils import (
            create_causal_mask, create_recurrent_attention_mask)

        def build_masks(x):
            dummy = x.new_zeros(1, 1, D)
            kw = dict(config=tm.config, inputs_embeds=dummy, attention_mask=None,
                      past_key_values=cc, position_ids=text_pos)
            return {"full_attention": create_causal_mask(**kw),
                    "linear_attention": create_recurrent_attention_mask(**kw)}

        x = h_l
        masks = build_masks(x)
        layer_outs = [x[:, 0, :].clone()]
        for i in range(l + 1, n_layers):
            lt = tm.config.layer_types[i]
            x = tm.layers[i](x, position_embeddings=pe, attention_mask=masks[lt],
                             position_ids=text_pos, past_key_values=cc)
            layer_outs.append(x[:, 0, :].clone())
        manual_final = x[:, 0, :]
        manual_normed = tm.norm(manual_final)

    print("\n== final hidden (before norm) ==")
    print("max abs diff", (manual_final - true_final).abs().max().item())
    print("rel", (manual_final - true_final).abs().max().item() / true_final.abs().max().item())

    print("\n== per-layer hidden states (last token) ==")
    # true_layer_out[i+1] = output of layer i
    for i in range(l, n_layers):
        d = (layer_outs[i - l] - true_layer_out[i + 1]).abs().max().item()
        s = true_layer_out[i + 1].abs().max().item()
        lt = tm.config.layer_types[i]
        print(f"  layer {i:2d} ({lt:16s}) maxdiff={d:.3e} rel={d / max(s,1e-9):.3e}")

    print("\n== normed final ==")
    print("max abs diff", (manual_normed - true_normed).abs().max().item())


if __name__ == "__main__":
    main()
