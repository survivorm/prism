# E-68 READER HARNESS (CPU, kv-кэш): «привязь читателя» к писателю (пре-рег §27 U-B).
# Гипотеза: раз чтение сильнее письма (corpus dip глубже), писателя можно вести сигналом читателя:
#   greedy:  x_t = argmax p(x|ctx)
#   harness: x_t = argmax_{x∈top-k(p)} READER(x), READER(x) = max_y log p(y | ctx+x)  («наскок чтения»)
# Метрики на 8 wiki-доках × 64 шага: flip-rate vs greedy, p_same/rep16/лоопность, dip(h-пути)
# против полос corpus(−0.167) и own-greedy(−0.111); средняя логита выбранного токена.
# Побочное для U-C: сохраняем матрицу оценок S[step][k] (для «инвертированного чтения»).
import ast, json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(12)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e68"; os.makedirs(OUT,exist_ok=True)
NDOC=int(os.environ.get("E68_NDOC","8")); NSTEP=64; TOPK=8
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); want=["vpath_units","dcurve","curves_series"]
fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
missing=[w for w in want if w not in fns]; assert not missing, missing
ns={"np":np}
for w in want: exec(fns[w],ns)
curves_series=ns["curves_series"]

tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.float32).eval()
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
docs=[d["input_ids"][0].tolist()[:64] for d in pool[:NDOC]]
caps={}
h=m.model.layers[23].register_forward_hook(lambda mod,i,o: caps.__setitem__(23,o.detach()))
res={"meta":dict(NDOC=NDOC,NSTEP=NSTEP,TOPK=TOPK),"docs":{}}
def save(): json.dump(res,open(f"{OUT}/HARNESS.json","w"),default=float)

def step_fwd(ids_t,cache):
    o=m(input_ids=ids_t,past_key_values=cache,use_cache=True)
    return o.logits[0,-1].float(), o.past_key_values

def gen(ids0, mode):
    from transformers import DynamicCache
    cache=DynamicCache(config=m.config)
    o=m(input_ids=torch.tensor([ids0]),past_key_values=cache,use_cache=True)
    lg=o.logits[0,-1].float(); cache=o.past_key_values
    toks=[]; hpath=[]; flips=0; S_rows=[]; chosen_logits=[]
    for st in range(NSTEP):
        caps.clear()
        top=torch.topk(lg,TOPK)
        ids_t=torch.tensor([[top.indices[0]]])
        lg2,cache2=step_fwd(ids_t,cache)           # greedy-шаг (он же кандидат №0)
        hs=caps.get(23,[None])
        hs=caps[23][0,-1].numpy().astype(np.float32) if 23 in caps else None
        if mode=="greedy":
            nxt=int(top.indices[0]); row=[None]*TOPK
        else:
            scores=[]
            cache=cache2  # базовый кэш уже включает greedy-кандидата — но нам нужны независимые пробы:
            for j in range(TOPK):
                cid=int(top.indices[j])
                lgc,_=step_fwd(torch.tensor([[cid]]),cache)
                scores.append(float(lgc.max()))
            S_rows.append(scores)
            nxt=int(top.indices[int(np.argmax(scores))])
        chosen_logits.append(float(lg[nxt]))
        if mode=="harness":
            if nxt!=int(top.indices[0]): flips+=1
            lg,_=step_fwd(torch.tensor([[nxt]]),cache)  # перекидываем кэш на выбранного
        else:
            lg=lg2
        toks.append(nxt)
        if hs is not None: hpath.append(hs)
    Hp=np.stack(hpath) if hpath else np.zeros((2,2))
    return dict(toks=toks,flip_rate=flips/NSTEP,p_same=float(np.mean([toks[i]==toks[i-1] for i in range(1,len(toks))])),
        rep16=float(np.mean([toks[i]==toks[i-16] for i in range(16,len(toks))])) if len(toks)>20 else None,
        mean_logit=float(np.mean(chosen_logits)),S=S_rows,**{k:v for k,v in curves_series(Hp).items() if k!="curve"})

t0=time.time()
for di,ids0 in enumerate(docs):
    try:
        g=gen(ids0,"greedy"); r=gen(ids0,"harness")
        res["docs"][f"wiki_{di}"]=dict(greedy=g,harness=r)
        print(f"[{time.time()-t0:6.0f}s] wiki_{di} flip {r['flip_rate']:.2f} "
              f"dip greedy {g['dip']:+.3f} -> harness {r['dip']:+.3f} "
              f"p_same {g['p_same']:.2f}->{r['p_same']:.2f}",flush=True)
        save()
    except Exception as e:
        print("!!",di,e,flush=True); res.setdefault("errors",[]).append(f"{di}:{e}"); save()
import numpy as _np
dg=[d["greedy"]["dip"] for d in res["docs"].values()]; dh=[d["harness"]["dip"] for d in res["docs"].values()]
res["verdict"]=dict(dip_greedy=round(float(_np.median(dg)),3),dip_harness=round(float(_np.median(dh)),3),
    flip_med=float(_np.median([d["harness"]["flip_rate"] for d in res["docs"].values()])),
    psame_g=round(float(_np.median([d["greedy"]["p_same"] for d in res["docs"].values()])),3),
    psame_h=round(float(_np.median([d["harness"]["p_same"] for d in res["docs"].values()])),3),
    bands=dict(corpus=-0.167,own=-0.111),
    n=len(res["docs"]))
save(); print(json.dumps(res["verdict"],indent=1),flush=True); print("E68 DONE",flush=True)
