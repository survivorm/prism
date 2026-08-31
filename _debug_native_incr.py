"""Does the model's OWN incremental decoding (prefix + last token) reproduce truth?"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
torch.set_num_threads(8)
torch.manual_seed(0)

def main():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32,
        attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    tm = model.model
    n_layers = tm.config.num_hidden_layers

    ids = torch.randint(0, 1000, (1, 32))
    attn = torch.ones_like(ids)
    S = ids.shape[1]

    with torch.no_grad():
        out = model(ids, attention_mask=attn, output_hidden_states=True)
        true_final = out.hidden_states[n_layers][:, -1, :]

        # prefix pass
        pout = model(ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True)
        cache = pout.past_key_values

        # incremental: feed ONLY the last token, using model's own forward
        attn_incr = torch.ones((1, S), dtype=attn.dtype)
        incr = model(ids[:, -1:], attention_mask=attn_incr, past_key_values=cache,
                     output_hidden_states=True)
    incr_final = incr.hidden_states[n_layers][:, -1, :]

    d = (incr_final - true_final).abs().max().item()
    s = true_final.abs().max().item()
    print(f"\nNATIVE incremental final vs true: maxdiff={d:.4e} rel={d/s:.4e}")

    # also check logits match
    with torch.no_grad():
        out2 = model(ids, attention_mask=attn)
        incr2 = model(ids[:, -1:], attention_mask=torch.ones((1, S)),
                      past_key_values=model(ids[:, :-1], attention_mask=attn[:, :-1], use_cache=True).past_key_values)
    print("logits last-token native incr vs full:",
          (incr2.logits[:, -1] - out2.logits[:, -1]).abs().max().item())

if __name__ == "__main__":
    main()
