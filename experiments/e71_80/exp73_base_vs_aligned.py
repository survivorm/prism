# E-73 BASE vs ALIGNED (CPU): меняет ли выравнивание читающий субстрат?
# Два modelo одного размера (Qwen3.5-0.8B-Base против выровненной Qwen3.5-0.8B), один пиновый
# корпус teacher-forced (шаблон НЕ нужен — чистое сравнение полей), попарно по документам.
# Метрики: nll, dip/hump/plateau L11/L23. Вердикт-гейт (пре-рег): сдвиг поля, если |Δdip_med|
# > σ_doc (междоковая σ base) и ≥7/16 попарных сдвигов одного знака; иначе — «выравнивание не
# трогает чтение». Второй класс текста (math) — контроль генерализации вывода.
import ast, json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e73"; os.makedirs(OUT,exist_ok=True)
LAYERS=[11,23]; TOKMAX=256
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
ns={"np":np}
for w in ["vpath_units","dcurve","curves_series"]: exec(fns[w],ns)
curves_series=ns["curves_series"]
pools={}
for dm in ["wiki","math"]:
    pools[dm]=[d["input_ids"][0].tolist()[:TOKMAX] for d in
               load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,
                                     n_samples=16 if dm=="wiki" else 8,domain=dm)]
FN=f"{OUT}/BASE_VS_ALIGNED.json"
res={"pool":"Hub-поток того же протокола, что E-65 (pool_fp в логе)"}
if os.path.exists(FN): res.update(json.load(open(FN)))
def save(): json.dump(res,open(FN,"w"),default=float)
def run(tag,mid):
    if tag in res: print("skip",tag,flush=True); return
    m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.float32).eval()
    caps={}
    hs=[m.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in LAYERS]
    out={}
    for dm,docs in pools.items():
        t0=time.time(); rows={}
        for di,ids in enumerate(docs):
            t=torch.tensor([ids]); caps.clear()
            with torch.no_grad(): o=m(input_ids=t,use_cache=False)
            nll=float(torch.nn.functional.cross_entropy(o.logits[0,:-1].float(),t[0,1:]))
            row={"nll":nll}
            for L in LAYERS:
                c=curves_series(caps[L][0,:len(ids)].to(torch.float32).numpy())
                row[f"L{L}"]=dict(dip=c["dip"],hump=c["hump"],plateau=c["plateau"])
            rows[dm+"_"+str(di)]=row
            print(f"[{tag}/{dm}] {di} nll {nll:.2f} dipL23 {row['L23']['dip']:+.3f} [{time.time()-t0:.0f}s]",flush=True)
            out[dm]=rows; res[tag]=out; save()
    for h in hs: h.remove()
    del m; import gc; gc.collect()
run("base","Qwen/Qwen3.5-0.8B-Base")
run("aligned","Qwen/Qwen3.5-0.8B")
if "base" in res and "aligned" in res:
    ver={}
    for dm in pools:
        ks=list(res["base"][dm])
        dd=[res["aligned"][dm][k]["L23"]["dip"]-res["base"][dm][k]["L23"]["dip"] for k in ks]
        sd=float(np.std([res["base"][dm][k]["L23"]["dip"] for k in ks]))
        sgn=int(np.sum(np.sign(dd)!=np.sign(dd[0])))
        ver[dm]=dict(d_dip_med=round(float(np.median(dd)),4),sigma_doc=round(sd,4),
            shifted=bool(abs(float(np.median(dd)))>sd), same_sign=int(len(dd)-sgn),n=len(dd),
            d_nll_med=round(float(np.median([res["aligned"][dm][k]["nll"]-res["base"][dm][k]["nll"] for k in ks])),3))
    res["verdict"]=ver; save()
    print(json.dumps(ver,indent=1),flush=True)
print("E73 DONE",flush=True)
