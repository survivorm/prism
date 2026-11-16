# E-65b W-QUANT (CPU): НАСТОЯЩЕЕ весовое квантование (W8A16/W4A16/W4g32/W3) — GGUF-семантика
# (асимметричные per-group scale/zero, де-квант в fp32-компьют). Две оси: Δdip (ритм) × Δnll.
import ast, json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e65"; os.makedirs(OUT,exist_ok=True)
LAYERS=[1,11,23]; NDOC=12; TOKMAX=256
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); want=["vpath_units","dcurve","curves_series"]
fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
missing=[w for w in want if w not in fns]; assert not missing, missing
ns={"np":np}
for w in want: exec(fns[w],ns)
curves_series=ns["curves_series"]
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
docs=[d["input_ids"][0].tolist()[:TOKMAX] for d in pool[:NDOC]]
FN=f"{OUT}/QUANT_HEALTH_w.json"
res={"cells":{}}
if os.path.exists(FN): res.update(json.load(open(FN)))
def save(): json.dump(res,open(FN,"w"),default=float)
def wquant(m,bits,group):
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod,torch.nn.Linear):
                W=mod.weight.data.float(); O,I=W.shape
                g=min(group,I)
                if I%g: g=I
                Wg=W.view(O,I//g,g)
                mn=Wg.amin(-1,keepdim=True); mx=Wg.amax(-1,keepdim=True)
                lvl=float(2**bits-1)
                s=((mx-mn)/lvl).clamp_min(1e-8)
                q=((Wg-mn)/s).round().clamp(0,lvl)
                mod.weight.data=((q*s+mn).view(O,I)).to(mod.weight.dtype)
    return m
def run_cond(tag,mk):
    if tag in res["cells"]: print("skip",tag,flush=True); return
    t0=time.time(); m=mk().eval()
    caps={}
    hs=[m.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in LAYERS]
    per={}
    for di,ids in enumerate(docs):
        try:
            t=torch.tensor([ids]); caps.clear()
            with torch.no_grad(): o=m(input_ids=t,use_cache=False)
            nll=float(torch.nn.functional.cross_entropy(o.logits[0,:-1].float(),t[0,1:]))
            row={}
            for L in LAYERS:
                H=caps[L][0,:len(ids)].to(torch.float32).numpy()
                c=curves_series(H); row[f"L{L}"]=dict(dip=c["dip"],hump=c["hump"],plateau=c["plateau"])
            per[f"wiki_{di}"]=dict(nll=nll,**row)
            print(f"[{time.time()-t0:6.0f}s] {tag} wiki_{di} nll {nll:.3f} dipL23 {row['L23']['dip']:+.3f}",flush=True)
            res["cells"][tag]=per; save()
        except Exception as e:
            print("!!",tag,di,e,flush=True); res.setdefault("errors",[]).append(f"{tag}/{di}:{e}"); save()
    for h in hs: h.remove()
    del m; import gc; gc.collect()
B="Qwen/Qwen3.5-0.8B"
for tag,bits,g in [("w8_g128",8,128),("w4_g128",4,128),("w4_g32",4,32),("w3_g128",3,128)]:
    run_cond(tag,lambda bits=bits,g=g: wquant(AutoModelForCausalLM.from_pretrained(B,dtype=torch.float32),bits,g))
print("E65B DONE",flush=True)
