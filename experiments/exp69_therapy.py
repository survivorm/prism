# E-69 THERAPY (CPU): первая «психотерапия» — лечение повреждённой модели с ДВУМЯ осями в логах.
# Пре-рег §29/§30: поражение SVD-8 (карта: R −4σ, P +11σ). Терапия = LoRA-r8 на attention-Linear,
# цель = обычный CE на том же wiki-корпусе (мы НЕ оптимизируем dip — он независимый контрольный
# канал: если после CE-терапии ритм сам вернётся в норму — двухосевой чек-ап валидируется как
# мониторинг восстановления; если ритм НЕ вернётся при нормальном nll — каналы независимы и
# «психотерапии» нужен специальный лосс по R-каналу).
# Arтефакт: results/e69/THERAPY.json — trajectory по эпохам (nll, dip L23).
# FIX 11.11 (audit2 B6/D3/D10): (1) healthy_anchor СЧИТАЕТСЯ этим же скриптом на тех же 8 доках
#   тем же evalm (был зашит [3.19,-0.174] — медиана по 12 докам E-65 против среднего по 8: Б8 в коде);
#   (2) .eval(); (3) rep16 НЕ СЧИТАЕТСЯ (докстринг обещал, код не считал) — заявка «ритм вылечился,
#   не лупы убраны» НЕ ЗАКРЫТА альтернативой «убрали лупы» — см. audit2 B6; (4) nll здесь — train-nll
#   на тех же 8 доках обучения; holdout E-69c (math/code/chat) НЕ ИМЕЕТ КОДА в репо (audit2 D4) — в очереди.
import ast, json, os, time, math
import numpy as np, torch
torch.set_num_threads(10)
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e69"; os.makedirs(OUT,exist_ok=True)
LAYERS=[1,11,23]; NDOC=8; TOKMAX=256; EPOCHS=4; R=8; ALPHA=16.0; LR=1e-4
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); want=["vpath_units","dcurve","curves_series"]
fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
missing=[w for w in want if w not in fns]; assert not missing
ns={"np":np}
for w in want: exec(fns[w],ns); 
curves_series=ns["curves_series"]
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
docs=[d["input_ids"][0].tolist()[:TOKMAX] for d in pool[:NDOC]]
B="Qwen/Qwen3.5-0.8B"
def svd8(m):
    from torch.linalg import svd
    with torch.no_grad():
        for lyr in m.model.layers:
            for nm in ["down_proj","up_proj","gate_proj"]:
                W=getattr(lyr.mlp,nm).weight
                U,S,Vh=svd(W.data.float(),full_matrices=False)
                S2=S.clone(); S2[8:]=0
                W.data=(U@torch.diag(S2)@Vh).to(W.dtype)
    return m
class LoRA(nn.Module):
    def __init__(s,lin,r,al):
        super().__init__(); s.lin=lin; s.r=r
        s.A=nn.Parameter(torch.zeros(r,lin.in_features)); s.B=nn.Parameter(torch.zeros(lin.out_features,r))
        nn.init.normal_(s.A,std=0.02); s.scale=al/r
    def forward(s,x):
        return s.lin(x)+ (x@s.A.T@s.B.T)*s.scale
def attach(m,r=R,al=ALPHA):
    n=0
    for lyr in m.model.layers:
        for nm in ["q_proj","k_proj","v_proj","o_proj"]:
            lin=getattr(lyr.self_attn,nm)
            setattr(lyr.self_attn,nm,LoRA(lin,r,al)); n+=1
    return n
res={"meta":dict(R=R,ALPHA=ALPHA,LR=LR,EPOCHS=EPOCHS,NDOC=NDOC),"traj":[]}
FN=f"{OUT}/THERAPY.json"
if os.path.exists(FN): res.update(json.load(open(FN)))
def save(): json.dump(res,open(FN,"w"),default=float)
def evalm(m,grad=False):
    caps={}
    hs=[m.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in LAYERS]
    rows=[]
    for ids in docs:
        t=torch.tensor([ids]); caps.clear()
        with torch.no_grad() if not grad else torch.enable_grad():
            o=m(input_ids=t,use_cache=False)
            nll=torch.nn.functional.cross_entropy(o.logits[0,:-1].float(),t[0,1:])
        c=curves_series(caps[23][0,:len(ids)].to(torch.float32).numpy())
        rows.append((float(nll),c["dip"]))
    for h in hs: h.remove()
    return float(np.mean([r[0] for r in rows])),float(np.median([r[1] for r in rows]))
m=AutoModelForCausalLM.from_pretrained(B,dtype=torch.float32).eval()  # FIX audit2 D10
anchor_nll,anchor_dip=evalm(m)  # FIX audit2 D3: якорь = те же 8 доков, тот же evalm, ДО поражения
res["healthy_anchor_measured"]=[round(float(anchor_nll),3),round(float(anchor_dip),4)]
m=svd8(m)
n=attach(m)
for p in m.parameters(): p.requires_grad_(False)
for mod in m.modules():
    if isinstance(mod,LoRA): mod.A.requires_grad_(True); mod.B.requires_grad_(True)
opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=LR)
n0,d0=evalm(m)
res["traj"].append(dict(epoch=-1,nll=round(n0,3),dip=round(d0,4),note="SVD8 поражен"))
print(f"[t0] SVD8: nll {n0:.3f} dip {d0:+.4f} (lora n={n})",flush=True); save()
t0=time.time()
for ep in range(EPOCHS):
    tot=0.0
    for di,ids in enumerate(docs):
        t=torch.tensor([ids])
        o=m(input_ids=t,use_cache=False)
        loss=torch.nn.functional.cross_entropy(o.logits[0,:-1].float(),t[0,1:])
        opt.zero_grad(); loss.backward(); opt.step()
        tot+=float(loss)
        if di%4==0: print(f"  ep{ep} doc{di} ce {float(loss):.3f} [{time.time()-t0:.0f}s]",flush=True)
    nv,dv=evalm(m)
    res["traj"].append(dict(epoch=ep,ce_mean=round(tot/len(docs),3),nll=round(nv,3),dip=round(dv,4)))
    print(f"[ep{ep}] CE {tot/len(docs):.3f} | eval nll {nv:.3f} | dip {dv:+.4f}",flush=True); save()
res["verdict"]=dict(start=[n0,round(d0,4)],end=[res["traj"][-1]["nll"],res["traj"][-1]["dip"]],
    # FIX audit2 D3: якорь из замера выше (те же 8 доков), не зашитая медиана-12
    healthy_anchor=[anchor_nll,anchor_dip],
    p_recovered=bool(abs(res["traj"][-1]["nll"]-anchor_nll)<0.4),
    r_recovered=bool(abs(res["traj"][-1]["dip"]-anchor_dip)<0.04),
    caveat="rep16 не считается (audit2 B6): «лечение» не отделено от «убрали лупы»; nll — train на 8 доках")
save(); print(json.dumps(res["verdict"],indent=1),flush=True); print("E69 DONE",flush=True)
