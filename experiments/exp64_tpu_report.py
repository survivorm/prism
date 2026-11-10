#!/usr/bin/env python3
"""exp64_tpu_report.py — вердикты по TPU-артефактам (всё пре-регистрируется ЗДЕСЬ, до вскрытия).

Критерии:
 RNULL  : полоса random = медиана±полуинтервал dips по 3×8 докам; trained dip ВНЕ полосы и |Δ|>=0.02 => «выучено»
 MATRIX : dip(own) vs dip(corpus): |разница|<=0.02 => текстодостаточная; own|dip|−corpus|dip| >=0.02 => петлевая добавка
 P5     : plateau(shift±48) < 0.5*base_plateau => сопряжение (A5); >=0.8 => канал (P3); иначе — смешанно
 LADDER : тренд dip(T) 1.2->2.0; наклон с бутстрапом по докам;rep16(T), lookup(T)
 MARGIN : spearman(margin_mean, |dip|) и (p1_mean,|dip|) по докам T=0; partial с lookup
 LL     : lead-lag ρ(τ)=median_t cos(v_L(t),v_L'(t+τ)); асимметрия = max|ρ(τ)-ρ(-τ)| по τ<=12; порог 0.05
 DEG    : loopy если curve[k>=2].max()>=0.95; состав невырожденных per домен (wiki из P1)
"""
import numpy as np, json, os, sys, itertools, glob
D=sys.argv[1] if len(sys.argv)>1 else "results/e64/tpu"
def jload(p):
    try: return json.load(open(os.path.join(D,p)))
    except Exception: return None
def md(x): return None if x is None or len(x)==0 else float(np.median(x))
def band(x):
    x=np.array(x,float); 
    return dict(med=float(np.median(x)),lo=float(np.percentile(x,25)),hi=float(np.percentile(x,75)),
                mn=float(x.min()),mx=float(x.max()),n=len(x))
def sp(a,b):
    a,b=np.array(a,float),np.array(b,float); m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<4 or len(set(a[m]))<2 or len(set(b[m]))<2: return None
    ra=np.argsort(np.argsort(a[m])); rb=np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra,rb)[0,1])
R={}
m1=jload("P1_own_meta.json"); m3=jload("P3_rnull_meta.json"); m2=jload("P2_corpus_meta.json")
# ---- P0 gate
st=jload("STATUS.json")
R["P0"]=st.get("P0") if st else None
# ---- R-null полоса
if m1 and m3:
    tr=[v["curves"]["23"]["dip"] for k,v in m1.items() if k.startswith("own_w")]
    rn=[v["curves"]["23"]["dip"] for k,v in m3.items() if k.startswith("rnull")]
    rb=band(rn); tmed=md(tr)
    inside = rb["mn"]-1e-12 <= tmed <= rb["mx"]+1e-12
    R["RNULL"]=dict(trained=dict(med=tmed,n=len(tr)),random=rb,
        learned=bool((not inside) and abs(tmed-rb["med"])>=0.02))
    # горб/плато
    for stat in ["hump","plateau"]:
        a=[v["curves"]["23"][stat] for k,v in m1.items() if k.startswith("own_w")]
        b=[v["curves"]["23"][stat] for k,v in m3.items() if k.startswith("rnull")]
        R[f"RNULL_{stat}"]=dict(trained=md(a),random=band(b))
# ---- матрица чтения own/corpus
if m1 and m2:
    a=[v["curves"]["23"]["dip"] for k,v in m2.items() if k.startswith("cor_w")]
    t=[v["curves"]["23"]["dip"] for k,v in m1.items() if k.startswith("own_w")]
    da,dt=md(a),md(t)
    R["MATRIX"]=dict(dip_corpus=da,dip_own=dt,diff=dt-da,
        verdict=("текстодостаточная" if abs(dt-da)<=0.02 else
                 ("петлевая добавка" if abs(dt)-abs(da)>=0.02 else "иное")))
# ---- P5 сдвиг
p5=jload("P5_shift.json")
if p5:
    v=[]; 
    for k,d in p5.items():
        b=abs(d["base"]["plateau"]); s=abs(d["shift"]["48"]["plateau"]); s2=abs(d["shift"]["-48"]["plateau"])
        v.append(np.median([s,s2])/max(b,1e-9))
    ratio=float(np.mean(v))
    R["P5_shift"]=dict(ratio=ratio,per_doc=v,
        verdict=("сопряжение(A5)" if ratio<0.5 else ("канал(P3)" if ratio>=0.8 else "смешанно")))
    if "mix" in next(iter(p5.values())):
        R["P5_mix0"]=md([d["mix"]["0.0"]["plateau"] for d in p5.values()])
# ---- T-лестница
lad={}
for T in ["1.2","1.5","2.0"]:
    m=jload(f"P4_T{T}_meta.json")
    if m:
        lad[T]=dict(dip=md([v["curves"]["23"]["dip"] for k,v in m.items()]),
            hump=md([v["curves"]["23"]["hump"] for k,v in m.items()]),
            plateau=md([v["curves"]["23"]["plateau"] for k,v in m.items()]),
            rep16=md([v["rep16"] for k,v in m.items()]),
            lookup=md([v["lookup"] for k,v in m.items()]),n=len(m))
if lad: R["LADDER"]=lad
# ---- degenerate состав (wiki T=0)
if m1:
    lp=[bool(max((v["curves"]["23"]["curve"].get(str(k),0) for k in range(2,25)))>=0.95) for k,v in m1.items() if k.startswith("own_w")]
    R["DEG_wiki"]=dict(loopy=int(sum(lp)),n=len(lp))
# ---- MARGIN (из npz P1)
p=os.path.join(D,"P1_own.npz")
if m1 and os.path.exists(p):
    z=np.load(p)
    names=[k[:-3] for k in z.files if k.endswith("_mg")]
    mg=[float(z[f"{n}_mg"].mean()) for n in names]
    p1=[float(z[f"{n}_p1"].mean()) for n in names]
    dip=[m1[n]["curves"]["23"]["dip"] for n in names]
    lk=[m1[n]["lookup"][0] for n in names]
    R["MARGIN"]=dict(sp_margin_dip=sp(mg,dip),sp_p1_dip=sp(p1,dip),sp_p1_lookup=sp(p1,lk),
        sp_margin_lookup=sp(mg,lk),n=len(names))
# ---- lead-lag из P1 npz (по 6 докам)
if os.path.exists(p):
    z=np.load(p); layers=sorted({int(k.split("_L")[1]) for k in z.files if "_L" in k and z[k].ndim==2})
    if len(layers)>=3:
        L1,L2=min(layers),max(layers)
        A=z[f"own_w0_L{L1}"].astype(np.float32); B=z[f"own_w0_L{L2}"].astype(np.float32)
        def vn(X): 
            Dd=X[1:]-X[:-1]; return Dd/np.maximum(np.linalg.norm(Dd,axis=1,keepdims=True),1e-9)
        PA,PB=vn(A),vn(B); rho={}
        for tau in range(-12,13):
            if tau>=0: c=np.mean(np.sum(PA[:len(PB)-tau or None]*PB[tau:],axis=1)) if tau<len(PB)-2 else np.nan
            else:     c=np.mean(np.sum(PA[-tau:]*PB[:len(PA)+tau],axis=1))
            rho[tau]=float(c)
        asym=max(abs(rho[t]-rho[-t]) for t in range(1,13))
        R["LL"]=dict(rho={str(k):round(v,3) for k,v in rho.items()},asym=asym,
            verdict="направленное время" if asym>0.05 else "симметрично")
print(json.dumps(R,indent=1,ensure_ascii=False))
json.dump(R,open(os.path.join(D,"JUDGMENTS.json"),"w"),indent=1,ensure_ascii=False)
