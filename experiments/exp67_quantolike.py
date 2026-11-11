# E-67 QUANTOLOBIE (офлайн на P2_corpus.npz): структурные тесты «квантоподобия» времени.
# Пре-рег §27 U-A. Тесты:
#  T1 g2-когерентность по «фазовым» модам lift-оператора (комплексные собственные пары):
#     g2 = <|I_t I_{t+1}>|/(<I_t><I_{t+1}>); классика когерентного поля g2>=1, анти-группировка<1.
#  T2 CHSH-контекстуальность на последовательных бинаризациях h_t,h_{t+1} (4 направления):
#     |S|>2 => нет совместного колмогоровского распределения (КОНТЕКСТУАЛЬНОСТЬ, не «квантовый мозг»).
#  T3 мнимая мощность спектра lift-оператора: доля |Im λ| в ||λ|| — «вращательность» обновления.
# Null'ы: те же тесты на P3_rnull (3 сида) — если random такие же числа, эффект не выучен.
# FIX 11.11 (audit2 H3/H3b/H3c, S5):
#  T2 CHSH МЁРТВ: все 4 наблюдаемые — детерминированные функции одной пары (h_t,h_{t+1}),
#  поточечно |S|=2 тождественно => тест не может дать ничего, кроме "classical". Оставлен в коде
#  с вердиктом "degenerate".
#  T1 g2 = 1 + Cov_1(I) (I нормирована на mean 1) — это переименованный дип (анти-персистентность
#  интенсивности), не оптический g2. Вердикт помечен.
#  rdocs: k[:-3] при суффиксе _L23 (4 символа) => KeyError на random-ветке; исправлено на k[:-4].
#  добавлены import os и os.makedirs("results/e67") — раньше скрипт падал ПОСЛЕ печати trained-чисел.
import os
import numpy as np, json
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # FIX audit4 S5: пути от ROOT, не от cwd
zT=np.load(os.path.join(ROOT,"results/e64/tpu/P2_corpus.npz"))  # corpus-скан: h [n,H] по слоям (реальный текст!)
zR=np.load(os.path.join(ROOT,"results/e64/tpu/P3_rnull.npz"))
m2=json.load(open(os.path.join(ROOT,"results/e64/tpu/P2_corpus_meta.json")))
docsT=[k for k in m2 if k.startswith("cor_w")]
rng=np.random.default_rng(0)
def lift(H,rank=64,ridge=1e-3):
    V=H[1:]-H[:-1]; V=V-V.mean(0)
    _,_,Vt=np.linalg.svd(V,full_matrices=False); r=min(rank,len(Vt))
    X=V@Vt[:r].T; A,B=X[:-1],X[1:]
    G=A.T@A+ridge*np.trace(A.T@A)/r*np.eye(r)
    M=np.linalg.solve(G,A.T@B); ev,Vc=np.linalg.eig(M)
    return X,ev,Vc
def g2_phase(X,ev,Vc,k=8):
    out=[]
    for j in np.argsort(-np.abs(ev.imag))[:k]:
        vr=Vc[:,j]; a,b=vr.real,vr.imag
        I=((X@a)**2+(X@b)**2)
        I=I/np.mean(I)
        g=float(np.mean(I[:-1]*I[1:])/1.0)  # I нормирована: g2=E[I_t I_t+1]
        out.append(g)
    return float(np.median(out)) if out else np.nan
def chsh(X,ev,n_dirs=12):
    # бинаризации по случайным плоскостям, observables на t и t+1
    D=X-X.mean(0); D/= (np.linalg.norm(D,axis=1,keepdims=True)+1e-9)
    P=np.linalg.qr(rng.normal(size=(X.shape[1],4)))[0][:,:4]  # 4 направления
    sgn=lambda M: np.where(M>=0,1,-1)
    A=[sgn(D[:-1]@P[:,i]) for i in range(2)]; B=[sgn(D[1:]@P[:,i+2]) for i in range(2)]
    E=lambda u,v: float(np.mean(u*v))
    S=E(A[0],B[0])+E(A[0],B[1])+E(A[1],B[0])-E(A[1],B[1])
    return abs(S)
def imag_power(ev):
    p=np.abs(ev)**2
    return float(np.sum((ev.imag)**2)/np.sum(p)) if p.sum()>0 else np.nan
def run(z,nms,tag):
    rows=[]
    for nm in nms:
        H=z[f"{nm}_L23"].astype(np.float32)
        X,ev,Vc=lift(H)
        rows.append(dict(nm=nm,g2=g2_phase(X,ev,Vc),chsh=chsh(X,ev),impow=imag_power(ev)))
    med=lambda key: float(np.nanmedian([r[key] for r in rows]))
    return dict(tag=tag,n=len(rows),g2=round(med("g2"),3),chsh=round(med("chsh"),3),
                impow=round(med("impow"),3),per=[{k:round(v,3) if isinstance(v,float) else v for k,v in r.items()} for r in rows])
t=run(zT,docsT,"trained_corpus")
rdocs=[k[:-4] for k in zR.files if k.endswith("_L23")]  # FIX audit2 H3c: было k[:-3]
r=run(zR,rdocs[:24],"random3sid")
res=dict(trained=t,random=r,
 verdict_chsh="DEGENERATE TEST (audit2 H3): ко-измеренные наблюдаемые => |S|≤2 тождественно, значение = "+str(t["chsh"]),
 verdict_g2=("анти-персистентность интенсивности (g2 = 1+Cov_1, переименованный дип — audit2 H3b)" if t["g2"]<1 else "кисп. когерентность (>=1)"),
 learned=dict(chsh=t["chsh"]-r["chsh"],g2=t["g2"]-r["g2"],impow=t["impow"]-r["impow"]))
print(json.dumps({k:v for k,v in res.items() if k not in("trained","random")},indent=1))
os.makedirs(os.path.join(ROOT,"results/e67"),exist_ok=True)  # FIX audit2 H3c
json.dump(res,open(os.path.join(ROOT,"results/e67/QUANTOLIKE.json"),"w"),indent=1)
