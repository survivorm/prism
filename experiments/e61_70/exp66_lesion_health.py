# E-66 LESION HEALTH (CPU): шкала «здоровья» на синтетических поражениях fp32 0.8B.
# Идея: если отпечаток поля — метрика здоровья, она обязана (а) мониторить известные деградации
# и (б) быть ДЕШЁВЫМ целевым функционалом починки без датасетов.
# Клетки (wiki corpus × 12 доков, teacher-forced; оценщик — ast-дубликат ядра):
#  L-0 baseline (= E-65 fp32, не пересчитываем)
#  L-1 шум: w *= (1+N(0,sigma)) по всем Linear, sigma ∈ {0.005, 0.01, 0.02, 0.05}
#  L-2 ранг: SVD-усечение mlp down/up_proj до r ∈ {8,16,32} (низкоранговое «отупение»)
#  L-3 ампутация: пропуск слоёв 12–17 (грубая хирургия; через hook-идентичность)
# Пре-рег вердикты:
#  S-1 (монотонность): |dip| падает с ростом поражения монотонно у обеих метрик? кто раньше?
#  S-2 (чувствительность): при |Δpplx| < 0.1 (невидимо для лосса) отпечаток уже уходит за ±0.02?
#  S-3 (психотерапия-минимум): для L-2 найти минимальный ранг r*, при котором отпечаток
#      восстановлен (>= fp32−0.02), сравнить с pplx-выбором ранга — датасет-фри критерий починки.
# Артефакт: results/e66/LESION_HEALTH.json — инкрементально по документу.
import ast, json, time, os, copy
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(12)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e66"; os.makedirs(OUT,exist_ok=True)
LAYERS=[1,11,23]; NDOC=int(os.environ.get("E66_NDOC","12")); TOKMAX=256
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
res={"cells":{}}
def save(): json.dump(res,open(f"{OUT}/LESION_HEALTH.json","w"),default=float)
BASE="Qwen/Qwen3.5-0.8B"
def mk_noise(sig,seed=7):
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32)
    g=torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim>=2: p.mul_(1+torch.randn(p.shape,generator=g)*sig)
    return m
def mk_svd(rank):
    from torch.linalg import svd
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32)
    with torch.no_grad():
        for lyr in m.model.layers:
            for nm in ["mlp.down_proj","mlp.up_proj","mlp.gate_proj"]:
                W=dict(lyr.mlp.named_parameters())[nm] if False else getattr(lyr.mlp,nm).weight
                U,S,Vh=svd(W.data.float(),full_matrices=False)
                S2=S.clone(); S2[rank:]=0
                W.data=(U@torch.diag(S2)@Vh).to(W.dtype)
    return m
def mk_amputate(lo=12,hi=18):
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32)
    for l in range(lo,hi):
        m.model.layers[l].register_forward_hook(lambda mod,i,o: i[0])  # identity skip
    return m
def run_cond(tag,mk):
    t0=time.time(); m=mk(); m.eval()
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
                if not np.isfinite(H).all(): row[f"L{L}"]=dict(dip=float("nan"),hump=float("nan"),plateau=float("nan")); continue
                c=curves_series(H); row[f"L{L}"]=dict(dip=c["dip"],hump=c["hump"],plateau=c["plateau"])
            per[f"wiki_{di}"]=dict(nll=nll,**row)
            print(f"[{time.time()-t0:6.0f}s] {tag} wiki_{di} nll {nll:.3f} dipL23 {row['L23']['dip']:+.3f}",flush=True)
            res["cells"][tag]=per; save()
        except Exception as e:
            print(f"!! {tag} doc{di}: {e}",flush=True); res.setdefault("errors",[]).append(f"{tag}/{di}:{e}"); save()
    for h in hs: h.remove()
    del m; import gc; gc.collect()
for tag,mk in [("noise_0.005",lambda:mk_noise(0.005)),("noise_0.01",lambda:mk_noise(0.01)),
               ("noise_0.02",lambda:mk_noise(0.02)),("noise_0.05",lambda:mk_noise(0.05)),
               ("svd_8",lambda:mk_svd(8)),("svd_16",lambda:mk_svd(16)),("svd_32",lambda:mk_svd(32)),
               ("amput_12_17",lambda:mk_amputate())]:
    if tag in res["cells"]: continue
    run_cond(tag,mk)
print("E66 DONE",flush=True)
