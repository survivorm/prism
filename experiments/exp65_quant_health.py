# E-65 QUANT HEALTH (CPU-бэкенд): отпечаток здоровья при квантовании весов.
# Клетки: wiki-corpus × {fp32, bf16, int8-динам} × 12 доков; teacher-forced forward.
# Оценщик поля — ДОСЛОВНО из e64-ядра (ast-экстракция vpath_units/dcurve/curves_series).
# Пре-рег:
#  H-q1: есть условие, где pplx почти цел, а поле уехало за ±0.02 от fp32 => отпечаток
#        чувствительнее pplx (самостоятельная метрика).
#  H-q2: Δfingerprint коррелирует с Δpplx по докам => метрика = датасет-фри прокси pplx.
#  Отрицание: int8 == fp32 в пределах doc-IQR => ритм неуязвим к инт8 (тоже результат).
#  Бонус: локальный fp32 против Kaggle-GPU fp32 на том же корпусе (pool_fp сверяется) —
#         третья аппаратная эпоха кросс-стекового null'а.
import ast, json, time, sys
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(12)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data

OUT="results/e65"; import os; os.makedirs(OUT,exist_ok=True)
LAYERS=[1,11,23]; NDOC=12; TOKMAX=256
# --- дословный оценщик из e64-ядра ---
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); want=["vpath_units","dcurve","curves_series"]
fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
missing=[w for w in want if w not in fns]
assert not missing, f"no funcs {missing}"
ns={"np":np}
for w in want: exec(fns[w],ns)
curves_series=ns["curves_series"]

tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
docs=[d["input_ids"][0].tolist()[:TOKMAX] for d in pool[:NDOC]]
res={"pool_note":"load_calibration_data wiki n16 (см. pool_fp в логе)","layers":LAYERS,"conds":{}}
def save(): json.dump(res,open(f"{OUT}/QUANT_HEALTH.json","w"),default=float)

def run_cond(tag,mk):
    t0=time.time(); m=mk(); m.eval()
    caps={}
    hs=[m.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in LAYERS]
    per={}
    for di,ids in enumerate(docs):
        try:
            t=torch.tensor([ids])
            caps.clear()
            with torch.no_grad():
                o=m(input_ids=t,use_cache=False)
            nll=float(torch.nn.functional.cross_entropy(o.logits[0,:-1].float(),t[0,1:]))
            row={}
            for L in LAYERS:
                H=caps[L][0,:len(ids)].to(torch.float32).numpy()
                c=curves_series(H)
                row[f"L{L}"]=dict(dip=c["dip"],hump=c["hump"],plateau=c["plateau"])
            per[f"wiki_{di}"]=dict(nll=nll,**row)
            print(f"[{time.time()-t0:6.0f}s] {tag} wiki_{di} nll {nll:.3f} dipL23 {row['L23']['dip']:+.3f}",flush=True)
            res["conds"][tag]=per; save()
        except Exception as e:
            print(f"!! {tag} doc{di}: {e}",flush=True); res.setdefault("errors",[]).append(f"{tag}/{di}:{e}"); save()
    for h in hs: h.remove()
    del m; import gc, torch as T; gc.collect(); T.cuda.empty_cache() if T.cuda.is_available() else None

fp32=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.float32)
run_cond("fp32",lambda: fp32)  # одна и та же загрузка; bf16/int8 — свежие копии
del fp32
run_cond("bf16",lambda: AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.bfloat16))
def mkq():
    mm=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B",dtype=torch.float32)
    return torch.ao.quantization.quantize_dynamic(mm,{torch.nn.Linear},dtype=torch.qint8)
run_cond("int8d",mkq)

# --- вердикты ---
import math
from scipy.stats import spearmanr
base=res["conds"].get("fp32",{})
ver={}
for tag in ["bf16","int8d"]:
    c=res["conds"].get(tag,{})
    ks=[k for k in c if k in base]
    if not ks: continue
    dd=[c[k]["L23"]["dip"]-base[k]["L23"]["dip"] for k in ks]
    dp=[c[k]["nll"]-base[k]["nll"] for k in ks]
    rho=float(spearmanr(dd,dp).statistic) if len(ks)>3 else None
    ver[tag]=dict(n=len(ks),d_dip_med=float(np.median(dd)),d_dip_iqr=[float(x) for x in np.percentile(dd,[25,75])],
        d_nll_med=float(np.median(dp)),spearman=round(rho,3) if rho is not None else None,
        moved=abs(float(np.median(dd)))>0.02)
res["verdicts"]=ver; save()
print(json.dumps(ver,indent=1),flush=True)
print("E65 DONE",flush=True)
