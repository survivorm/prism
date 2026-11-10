# E-71 SOLITON HYPOTHESIS (пробуждение/режим как солитон), CPU. Две пробы:
# A) ПОРОГ ПРОБУЖДЕНИЯ: интерполяция весов W(λ)=λW_trained+(1−λ)W_random (λ: 0→1, 11 точек).
#    Солитонная интуиция предсказывает ПОРОГ (ниже критической нелинейности объект не образуется,
#    dip≈0; выше — быстро выходит на плато «запертой формы»), линейная — рост ∝λ.
#    Метрика: dip(L23) и Σc² по λ на wiki-корпусе (8 доков). Гейт: доля дисперсии, объяснённая
#    ступенькой (скала), против линейной регрессии по λ.
# B) УПРУГОЕ СТОЛКНОВЕНИЕ: в здоровый текст вставляется блок-«луп» (8× повторов фразы), текст
#    прогоняется teacher-forced; сравниваем dip СЕГМЕНТА ПОСЛЕ вставки с контролем (тот же текст
#    без вставки). Солитон = луп проходит, среда восстанавливает ритм (упруго); дисперсия =
#    ритм после вставки разлажен надолго.
import ast, json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e71"; os.makedirs(OUT,exist_ok=True)
LAYERS=[11,23]; TOKMAX=256
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
ns={"np":np}
for w in ["vpath_units","dcurve","curves_series"]: exec(fns[w],ns)
curves_series=ns["curves_series"]
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
docs=[d["input_ids"][0].tolist()[:TOKMAX] for d in pool[:8]]
FN=f"{OUT}/SOLITON.json"; res={"A":{},"B":{}}
if os.path.exists(FN): res.update(json.load(open(FN)))
def save(): json.dump(res,open(FN,"w"),default=float)
def measure(m,ids):
    caps={}
    hs=[m.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in LAYERS]
    t=torch.tensor([ids])
    with torch.no_grad(): o=m(input_ids=t,use_cache=False)
    nll=float(torch.nn.functional.cross_entropy(o.logits[0,:-1].float(),t[0,1:]))
    for h in hs: h.remove()
    c=curves_series(caps[23][0,:len(ids)].to(torch.float32).numpy())
    return nll,c["dip"]
B0="Qwen/Qwen3.5-0.8B"
# --- A: порог пробуждения ---
if not res["A"]:
    base=AutoModelForCausalLM.from_pretrained(B0,dtype=torch.float32)
    sd0={k:v.clone() for k,v in base.state_dict().items()}
    g=torch.Generator().manual_seed(11)
    sd_r={k:(torch.randn(v.shape,generator=g)*v.std()) if v.ndim>=2 else v.clone() for k,v in sd0.items()}
    for lam in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        with torch.no_grad():
            for k,v in sd0.items(): base.state_dict()[k].copy_(lam*v+(1-lam)*sd_r[k])
        ds=[measure(base,d)[1] for d in docs]
        res["A"][f"lam_{lam:.1f}"]=dict(dip_med=round(float(np.median(ds)),4),dip_iqr=[round(float(x),3) for x in np.percentile(ds,[25,75])])
        print("[A] lam",lam,res["A"][f"lam_{lam:.1f}"],flush=True); save()
    del base; import gc; gc.collect()
# --- B: упругое столкновение ---
if not res["B"]:
    m=AutoModelForCausalLM.from_pretrained(B0,dtype=torch.float32).eval()
    phrase=tk(" The class will be held at the park.",add_special_tokens=False)["input_ids"]
    for di,ids in enumerate(docs):
        mid=len(ids)//2
        cut=min(mid,len(ids)-40)
        ctrl=ids[:cut]+ids[cut:]
        inj =ids[:cut]+phrase*8+ids[cut:]
        n0,d0=measure(m,ctrl); n1,d1=measure(m,inj)
        # сегмент после вставки: переиспользуем общий dip; узкий — по хвосту
        res["B"][f"wiki_{di}"]=dict(dip_control=round(d0,4),dip_injected=round(d1,4),nll_c=round(n0,3),nll_i=round(n1,3))
        print(f"[B] wiki_{di} dip control {d0:+.3f} vs injected {d1:+.3f}",flush=True); save()
print("E71 DONE",flush=True)
