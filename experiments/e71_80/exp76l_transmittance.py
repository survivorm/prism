#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76L ПРОВОДАЧИМОСТЬ ЗЕРКАЛА (собственный ответ на «readout режет 99.9%») и
# векторный тест обратного рассеяния (v2 после скалярного null E-76k).
# J зеркала (L23out→logits) явный: J_t = (1/rms_t)·W_U·diag(γ)·(I−ĥ_t ĥ_tᵀ).
#  τ_t = ||J_t Δ_t||/||Δ_t||; f_t = τ_t²/λ_max(J_tᵀJ_t) ∈ [0,1] — какая доля
#  «максимально возможной для этого зеркала» проводимости ушла с этим шагом.
#  1−f — «сколько порезали». (rms опущен И в τ, и в λmax — сокращается в f.)
# Тест v2: corr(e2(t), 1−f(t−k)), k=0,1,2; own vs корпус (там зеркало никуда не
# feedback-ит — контроль механики). ПРЕ-РЕГ: рассеяние живо, если в own ∃k:
# CI∌0 и корпус строее слабее; иначе — «показалось» (фиксируется положительно).
import os; os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, torch, json
from pathlib import Path
torch.set_grad_enabled(False)
ROOT=Path(__file__).resolve().parent.parent; D=ROOT/"results"/"e64"/"tpu"
R=16; rng=np.random.default_rng(63)
from transformers import AutoModelForCausalLM
mm=AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B",dtype=torch.float32,
                                        attn_implementation="eager").eval()
W=mm.model.lm_head.weight.detach().numpy().astype(np.float32)
gam=mm.model.norm.weight.detach().numpy().astype(np.float32)
del mm
z=np.load(D/"P1_own.npz"); zc=np.load(D/"P2_corpus.npz"); sfx="_L23"
def raw(zz): return [zz[k].astype(np.float64) for k in sorted(zz.files) if k.endswith(sfx)]
def unitc(Hr):
    H=Hr-Hr.mean(0); return H/np.linalg.norm(H,axis=1,keepdims=True)
def rows(zz): return [unitc(Hr) for Hr in raw(zz)]
Call=np.cov(np.concatenate(rows(z)+rows(zc),0).T);_,V=np.linalg.eigh(Call);B=V[:,-R:]
def u2_of(Ucell):
    Xs=[];Ys=[]
    for U in Ucell: A=U@B; Xs.append(A[:-1]);Ys.append(A[1:])
    Sxy=sum(x.T@y for x,y in zip(Xs,Ys));Sxx=sum(x.T@x for x in Xs)
    M=np.linalg.solve(Sxx.T,Sxy.T).T;w,Vv=np.linalg.eigh(.5*(M+M.T))
    return Vv[:,np.argsort(w)[:2]]
def Dfwd(x,r):  return gam*(x-np.dot(r,x)*r)            # D = diag(γ)(I−ĥĥᵀ)
def Dadg(x,r):  g=gam*x; return g-np.dot(r,g)*r         # Dᵀ = (I−ĥĥᵀ)diag(γ)
def lmax2(Hr,rng):
    """λ_max(JᵀJ) power-итерацией (J=W D), на 8 ĥ документа, усреднение."""
    lam=0.0
    for hi in range(0,min(8,len(Hr)),4):
        h=Hr[hi]; r=h/np.linalg.norm(h)
        v=rng.standard_normal(h.shape[0]).astype(np.float64); v/=np.linalg.norm(v)
        for _ in range(6):
            y=Dfwd(v,r); y=W@y; y=Dadg(W.T@y,r)          # JᵀJ v
            nv=np.linalg.norm(y)
            if nv<1e-30: break
            v=y/nv
        y=Dfwd(v,r); y=W@y
        lam=max(lam,float(y@y))                          # ||Jv||² = λ при сходимости
    return max(lam,1e-12)
def series(zz,plane):
    Fs=[];E1=[];E2=[]
    for Hr in raw(zz):
        Un=unitc(Hr); Dl=Hr[1:]-Hr[:-1]
        lmx=lmax2(Hr,rng)
        fs=np.empty(len(Dl))
        for t in range(len(Dl)):
            r=Hr[t]/np.linalg.norm(Hr[t]); d=Dl[t]
            Jd=W@Dfwd(d,r)
            fs[t]=float(Jd@Jd)/(float(d@d)*lmx)
        A=Un@B; DA=A[1:]-A[:-1]; n2=np.sum(DA*DA,axis=1)+1e-12
        E2.append((DA@plane[:,1])**2/n2); E1.append((DA@plane[:,0])**2/n2)
        Fs.append(np.clip(fs,0,None))
    return Fs,E1,E2
def corrboot(F,E,ks):
    out={}
    for k in ks:
        per=[]
        for f,e in zip(F,E):
            Tm=min(len(f),len(e)); m0=max(8,k)
            per.append((1.0-f[m0-k:Tm-k], e[m0:Tm]))
        def tt(ii):
            a=np.concatenate([per[i][0] for i in ii]); b=np.concatenate([per[i][1] for i in ii])
            return float(np.corrcoef(a,b)[0,1])
        r0=tt(range(len(per)))
        bs=[tt(rng.integers(0,len(per),len(per))) for _ in range(200)]
        out[f"k{k}"]=[round(r0,3),[round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)]]
    return out
Po=u2_of(rows(z)); Pc=u2_of(rows(zc))
fs_o,e1_o,e2_o=series(z,Po); fs_c,e1_c,e2_c=series(zc,Pc)
q=np.percentile(np.concatenate(fs_o),[5,50,95]); qc=np.percentile(np.concatenate(fs_c),[5,50,95])
print(f"f=τ²/λmax: own med {q[1]:.3f} [p5 {q[0]:.3f} p95 {q[2]:.3f}] | cor med {qc[1]:.3f} — «порезали» ≈{(1-q[1])*100:.1f}% макс.возможного")
o2=corrboot(fs_o,e2_o,[0,1,2]); c2=corrboot(fs_c,e2_c,[0,1,2])
print("e2 vs (1−f): own",o2,"\n             cor",c2,flush=True)
o1=corrboot(fs_o,e1_o,[0,1]); c1=corrboot(fs_c,e1_c,[0,1])
print("e1 vs (1−f): own",o1,"\n             cor",c1,flush=True)
(D/"EIGEN_L.json").write_text(json.dumps(dict(f_own=q.tolist(),f_cor=qc.tolist(),
    e2_own=o2,e2_cor=c2,e1_own=o1,e1_cor=c1),ensure_ascii=False,indent=1))
print("E76L DONE")
