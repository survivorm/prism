"""Focused probe: why does only the LAST layer (full_attention) diverge?"""
import os, copy
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
torch.set_num_threads(8)
torch.manual_seed(0)

def main():
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tm = model.model
    D = tm.config.hidden_size
    n_layers_cfg = tm.config.num_hidden_layers
    n_layers_obj = len(tm.layers)
    print("n_layers_cfg", n_layers_cfg, "n_layers_obj", n_layers_obj)
    print("block types:", [(i, tm.layers[i].block_type) for i in range(n_layers_obj)])

    ids = torch.randint(0, 1000, (1, 32))
    attn = torch.ones_like(ids)
    S = ids.shape[1]

    with torch.no_grad():
        out = model(ids, attention_mask=attn, output_hidden_states=True)
        pout = model(ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
    cache = pout.past_key_values
    true_final = out.hidden_states[n_layers_obj][:, -1, :]

    print("\ncache.get_seq_length() ==", cache.get_seq_length())
    print("len(cache.layers) ==", len(cache.layers))
    for i in [7, 11, 15, 19, 23]:
        lay = cache.layers[i]
        if hasattr(lay, "keys") and lay.keys is not None:
            print(f"  layer {i} ({tm.layers[i].block_type}): key len = {lay.keys[0].shape if isinstance(lay.keys[0], torch.Tensor) else [k.shape for k in lay.keys]}")

    # Now step through the decoder pass and report, right before EACH full_attention layer,
    # the cache key-length that layer sees and the mask shape.
    with torch.no_grad():
        cc = copy.deepcopy(cache)
        h_l = out.hidden_states[7][:, -1:, :]   # output of layer 6
        pos = torch.full((4, 1, 1), S - 1, dtype=torch.long)
        text_pos = pos[0]; rope_pos = pos[1:]
        pe = tm.rotary_emb(h_l, rope_pos)
        dummy = h_l.new_zeros(1, 1, D)
        # build masks exactly as prism does (once, up front)
        mk = dict(config=tm.config, inputs_embeds=dummy, attention_mask=None,
                  past_key_values=cc, position_ids=text_pos)
        mask_full = create_causal_mask(**mk)
        mask_rec = create_recurrent_attention_mask(**mk)
        masks = {"full_attention": mask_full, "linear_attention": mask_rec}

        x = h_l
        for i in range(7, n_layers_obj):
            lt = tm.layers[i].block_type
            if lt == "full_attention":
                lay = cc.layers[i]
                klen = None
                if hasattr(lay, "keys") and lay.keys is not None and len(lay.keys):
                    kk = lay.keys[0]
                    klen = kk.shape[-1] if isinstance(kk, torch.Tensor) else [k.shape[-1] for k in kk]
                print(f"BEFORE layer {i}: own-cache key len = {klen}")
            x = tm.layers[i](x, position_embeddings=pe,
                             attention_mask=masks[lt],
                             position_ids=text_pos, past_key_values=cc)

        manual_final = x[:, 0, :]
        print("\nmanual final vs true final:",
              (manual_final - true_final).abs().max().item(),
              (manual_final - true_final).abs().max().item() / true_final.abs().max().item())

if __name__ == "__main__":
    main()
