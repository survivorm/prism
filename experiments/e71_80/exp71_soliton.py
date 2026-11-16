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
# FIX 11.11 (audit2 B2/B3/D1/D2/D10): nll не выбрасывается по λ; sd_r — ВСЕ параметры случайные
# (1-D RMSNorm/bias тоже — λ=0 совпадает с R-null, было «обученные нормировки + случайные матрицы»);
# сида 3 (был 1: 11); resume — проверка полноты ключей, не truthiness; гейт «ступенька vs линейная»
# реализован в коде (res["A"]["gate"]); B: окно ПОСЛЕ вставки vs то же окно контроля
# (раньше ctrl=ids[:cut]+ids[cut:]==ids и сравнивались ГЛОБАЛЬНЫЕ медианы разной длины —
# «упругость» смешивалась с разбавлением); .eval() на обеих моделях (раньше держалось на
# dropout=0 Qwen3).
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
    h=caps[23][0,:len(ids)].to(torch.float32).numpy()
    return nll,curves_series(h),h  # FIX audit2 B3: h возвращаем для оконных замеров
B0="Qwen/Qwen3.5-0.8B"
LAM=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
SEEDS=[11,12,13]  # FIX audit2 B2: один сид => позиция порога сид-зависима, полоса не была измерена
res.setdefault("A",{}); res.setdefault("B",{})
def save(): json.dump(res,open(FN,"w"),default=float)
# --- A: порог пробуждения (полнота: все seed×lam, не truthiness) ---
AEXP={f"s{sd}_lam_{lam:.1f}" for sd in SEEDS for lam in LAM}
if not AEXP <= set(res["A"].keys()):
    base=AutoModelForCausalLM.from_pretrained(B0,dtype=torch.float32).eval()  # FIX audit2 D10
    sd0={k:v.clone() for k,v in base.state_dict().items()}
    for sd in SEEDS:
        g=torch.Generator().manual_seed(sd)
        # FIX audit2 B2: ВСЕ параметры случайные (включая 1-D) — λ=0 совпадает с R-null
        sd_r={k:torch.randn(v.shape,generator=g)*v.std() for k,v in sd0.items()}
        for lam in LAM:
            with torch.no_grad():
                for k,v in sd0.items(): base.state_dict()[k].copy_(lam*v+(1-lam)*sd_r[k])
            rows=[measure(base,d) for d in docs]
            dip=[r[1]["dip"] for r in rows]; nll=[r[0] for r in rows]
            res["A"][f"s{sd}_lam_{lam:.1f}"]=dict(
                dip_med=round(float(np.median(dip)),4),
                dip_iqr=[round(float(x),3) for x in np.percentile(dip,[25,75])],
                nll_med=round(float(np.median(nll)),3))  # FIX audit2 B2: nll не выбрасывается
            print("[A] seed",sd,"lam",lam,res["A"][f"s{sd}_lam_{lam:.1f}"],flush=True); save()
    # FIX audit2 D2: гейт «ступенька против линейной» — в коде, не прозой
    L=np.array(LAM); D={sd:np.array([res["A"][f"s{sd}_lam_{lam:.1f}"]["dip_med"] for lam in LAM]) for sd in SEEDS}
    y=np.concatenate([D[sd] for sd in D])
    lin=np.polyfit(L,y,1)
    r2lin=float(1-((y-(lin[1]*L+lin[0]))**2).sum()/((y-y.mean())**2).sum())
    best=(None,-1.0)
    for th in LAM[:-1]:
        lo=y[L<th].mean() if (L<th).any() else y.mean(); hi=y[L>=th].mean() if (L>=th).any() else y.mean()
        step=np.where(L>=th,hi,lo)
        rr=float(1-((y-step)**2).sum()/((y-y.mean())**2).sum())
        if rr>best[1]: best=(th,rr)
    res["A"]["gate"]=dict(r2_linear=round(r2lin,4),r2_step_best=round(best[1],4),step_at=best[0],
        linear_pred_lam05=round(float(np.polyval(lin,0.5)),4),
        measured_lam05=round(float(np.median([D[sd][5] for sd in D])),4),n_seeds=len(SEEDS),
        note="×17 теперь пересчитывается из артефакта: |linear_pred_lam05|/max(|measured_lam05|,1e-6)")
    del base; import gc; gc.collect()
# --- B: упругое столкновение (окно после вставки + контроль усечения) ---
BEXP={f"wiki_{di}" for di in range(len(docs))}
if not BEXP <= set(res["B"].keys()):
    m=AutoModelForCausalLM.from_pretrained(B0,dtype=torch.float32).eval()
    phrase=tk(" The class will be held at the park.",add_special_tokens=False)["input_ids"]
    L8=len(phrase)*8; W=64
    for di,ids in enumerate(docs):
        mid=len(ids)//2
        cut=min(mid,len(ids)-40)
        inj =ids[:cut]+phrase*8+ids[cut:]
        n0,c0,h0=measure(m,ids)                       # контроль: полный документ, без вставки
        n1,c1,h1=measure(m,inj)                       # inject
        cw=curves_series(h1[cut+L8:cut+L8+W])         # FIX audit2 B3: окно ПОСЛЕ вставки
        kw=curves_series(h0[cut:cut+W])               # то же окно в контроле
        # (контроль «то же усечение без инъекции» не существует: inject-документ ДЛИННЕЕ,
        #  original measure мерил его целиком — truncation в D1 не подтверждена; ядро контраста — cw vs kw)
        res["B"][f"wiki_{di}"]=dict(
            dip_control=round(c0["dip"],4),dip_injected=round(c1["dip"],4),
            dip_post_window=round(cw["dip"],4),dip_control_window=round(kw["dip"],4),
            nll_c=round(n0,3),nll_i=round(n1,3))
        print(f"[B] wiki_{di} post-window {cw['dip']:+.3f} vs ctrl-window {kw['dip']:+.3f} (global {c1['dip']:+.3f} vs {c0['dip']:+.3f})",flush=True); save()
print("E71 DONE",flush=True)
