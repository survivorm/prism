import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import prism.jacobian as PJ

torch.manual_seed(0)
torch.set_num_threads(8)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32)
model.eval()
tm = model.model.language_model if hasattr(model.model, "language_model") else model.model
text_model = tm

ids = torch.arange(32) % 100000  # arbitrary token ids, batch 1
ids = ids.view(1, -1).to(model.device)

def native_incremental():
    prefix = model(ids[:, :-1], attention_mask=torch.ones(1, 31, dtype=torch.long, device=ids.device), use_cache=True)
    out = model(ids[:, -1:], attention_mask=torch.ones(1, 32, dtype=torch.long, device=ids.device),
                past_key_values=prefix.past_key_values, use_cache=False)
    return out

# Hook: capture inputs (first arg = hidden_states) and output of specific layers
def make_capture(layer, store, tag):
    def pre(m, args, kwargs):
        store[(tag, "in")] = args[0].detach().clone() if len(args) > 0 else kwargs.get("hidden_states").detach().clone()
    def post(m, args, out):
        store[(tag, "out")] = out.detach().clone()
    m.register_forward_pre_hook(pre)
    m.register_forward_hook(post)

for target_l in (19, 23):
    print(f"\n=== Layer {target_l} ===")
    store = {}
    make_capture(text_model.layers[target_l], store, "n")
    out = native_incremental()
    native_out = store[("n", "out")].clone()
    native_in = store[("n", "in")].clone()

    # Now prism's loop
    store2 = {}
    make_capture(text_model.layers[target_l], store2, "p")
    # prefix pass via prism
    prefix_cache = PJ._run_prefix_pass(model, ids)
    S = ids.shape[1]  # 32
    # true layer (target_l - 1) output as input to layer target_l
    full = model(ids, attention_mask=torch.ones(1, S, dtype=torch.long, device=ids.device), output_hidden_states=True)
    l = target_l  # layer idx to start at
    h = full.hidden_states[l][:, -1, :].clone()  # output of layer l-1, input to layer l
    # run prism decoder from layer target_l
    res = PJ._run_decoder_to_norm(model, prefix_cache, h, S, target_l)
    prism_out = store2[("p", "out")].clone()
    prism_in = store2[("p", "in")].clone()
    text_model.layers[target_l]._forward_pre_hooks.clear()
    text_model.layers[target_l]._forward_hooks.clear()

    din = (native_in - prism_in).abs().max().item()
    dout = (native_out - prism_out).abs().max().item()
    print(f"  input diff:  {din:.3e} (native norm={native_in.norm().item():.4f})")
    print(f"  output diff: {dout:.3e} (native norm={native_out.norm().item():.4f})")
