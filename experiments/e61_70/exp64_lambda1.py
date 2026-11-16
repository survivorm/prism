#!/usr/bin/env python3
"""exp64_lambda1.py — λ₁-тест (И-2): lifted-оператор на vec обученной vs случайные сида.
M в PCA-базе r: V≈Z C; z_{t+1}=A z_t; спектр A. Прогноз: у trained есть λ с Re<0 и |arg|≈π
(period-2), у random — нет. Дип-привязка: Δ(1) ≈ (z-автокорреляция) — сверяем знак/величину.
Запуск локально после выгрузки P1/P3 npz с TPU."""
import numpy as np, json, sys, glob, os

def lift_spec(V, r=48):
    """V [T,D] -> спектр A (r×r) шаг-оператора в PCA-базе r; + реконструкция dip."""
    Vc=V-V.mean(0,keepdims=True)
    U,S,Vt=np.linalg.svd(Vc,full_matrices=False)
    Z=U[:,:r]*S[:r]                      # [T,r]
    A=np.linalg.lstsq(Z[:-1],Z[1:],rcond=None)[0]   # [r,r]
    ev=np.linalg.eigvals(A)
    order=np.argsort(-np.abs(ev))
    ev=ev[order]
    # dip по оценщику: cos(v_t,v_{t+1})
    D=V[1:]-V[:-1]; Dn=D/np.maximum(np.linalg.norm(D,axis=1,keepdims=True),1e-9)
    dip=float(np.mean(np.sum(Dn[:-1]*Dn[1:],axis=1))) if len(Dn)>2 else float('nan')
    return dict(dip=dip,
        top_abs=[complex(e) for e in ev[:6]],
        min_re=float(np.min(ev.real)),
        n_neg_re=int((ev.real<0).sum()),
        arg_near_pi=float(np.max(np.abs(np.angle(ev))>2.6)),
        spec_abs=[float(abs(e)) for e in ev[:6]],
        spec_re=[float(e.real) for e in ev[:6]],
        spec_im=[float(e.imag) for e in ev[:6]])

def med(rows,k): 
    v=[r[k] for r in rows if r.get(k) is not None]
    return float(np.median(v)) if v else None

def collect(npz,keys_prefix,L):
    z=np.load(npz); out=[]
    for k in z:
        if k.startswith(keys_prefix) and k.endswith(f"_L{L}"):
            out.append((k,z[k].astype(np.float32)))
    return out

if __name__=="__main__":
    d=sys.argv[1] if len(sys.argv)>1 else "results/e64/tpu"
    L=int(sys.argv[2]) if len(sys.argv)>2 else 23
    res={}
    for name,pref,src in [("trained","own_w","P1_own.npz"),("rnull","rnull_s","P3_rnull.npz")]:
        p=os.path.join(d,src)
        if not os.path.exists(p): print("skip",p); continue
        rows=[]
        for k,V in collect(p,pref,L):
            rows.append(dict(doc=k,**lift_spec(V)))
            print(name,k,"dip %.4f"%rows[-1]["dip"],"minRe %.3f"%rows[-1]["min_re"],flush=True)
        res[name]=dict(n=len(rows),dip=med(rows,"dip"),min_re=med(rows,"min_re"),
                       neg_re=med(rows,"n_neg_re"),has_pi_mode=float(np.mean([r["arg_near_pi"] for r in rows])),
                       docs=rows)
    if "trained" in res and "rnull" in res:
        t,rr=res["trained"],res["rnull"]
        res["verdict"]=dict(
            dip_gap=(t["dip"] or 0)-(rr["dip"] or 0),
            lambda1_claim=bool(t["min_re"] is not None and rr["min_re"] is not None
                               and t["min_re"]<-0.05 and rr["min_re"]>t["min_re"]+0.05),
            note="λ₁<0 глубже у trained на ≥0.05 и период-2 моды нет у random => И-2 жива")
    out=os.path.join(d,"LAMBDA1.json")
    def clean(o):
        if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
        if isinstance(o,list): return [clean(v) for v in o]
        if isinstance(o,complex): return [round(o.real,4),round(o.imag,4)]
        if isinstance(o,(np.floating,float)): return None if np.isnan(o) else round(float(o),5)
        if isinstance(o,(np.integer,int)): return int(o)
        return o
    json.dump(clean(res),open(out,"w"),indent=1)
    print("saved",out)
