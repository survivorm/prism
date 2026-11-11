#!/usr/bin/env python3
"""exp64_readmatrix.py — сводка матрицы чтения (own vs corpus vs rnull-клетки) + gap×lookup.
Вход: P1_own_meta.json, P2_corpus_meta.json (TPU). Профили: нормированные кривые k=1..48;
own-vs-xdoc контроль — по P1 между документами; xdoc_cos_mean = СРЕДНЯЯ КОСИНУСНАЯ ПОХОЖЕСТЬ кривых own (НЕ «gap»: v1-докстринг обещал
«own-профиль минус средний чужий», код считает cos — audit2 D5). FIX 11.11 (audit2 B4/D6):
авто-вердикт СНЯТ — порядок corpus(−0.167)>crossdom(−0.141)>own(−0.111) совпадает ни с одной
пре-рег сигнатурой; «читатель > писатель» неотделим от «естественный > вырожденный» (own 63% лупов)
до E-72/E-73.
Сигнатуры линз: corpus≈own (текст), own>corpus (петля), оба≫null (выученность)."""
import numpy as np, json, os, sys
d=sys.argv[1] if len(sys.argv)>1 else "results/e64/tpu"
L=str(23) if len(sys.argv)<3 else sys.argv[2]
def load(p):
    return json.load(open(os.path.join(d,p))) if os.path.exists(os.path.join(d,p)) else None
m1=load("P1_own_meta.json"); m2=load("P2_corpus_meta.json")
def curves(meta,pref):
    ks=[str(k) for k in range(1,49)]
    out={}
    for k,v in meta.items():
        if not k.startswith(pref): continue
        c=v["curves"].get(L) or v["curves"].get(str(23))
        if c: out[k]=np.array([c.get(kk,np.nan) for kk in ks],dtype=float)
    return out
def summarize(name,C):
    if not C: return None
    A=np.stack(list(C.values()))
    ks=np.arange(1,49)
    return dict(n=len(C),
        dip=float(np.nanmedian(A[:,0])),
        hump_k=[int(ks[np.nanargmax(r[1:8])]+1) for r in A][:8],
        hump=float(np.nanmedian([np.nanmax(r[1:8]) for r in A])),
        plateau=float(np.nanmedian(np.nanmean(A[:,8:24],axis=1))))
res={}
if m1:
    own=curves(m1,"own_w"); res["own"]=summarize("own",own)
    # профильный own−xdoc gap: cos(кривая_i, средняя чужая)
    if len(own)>=4:
        A=np.stack(list(own.values())); An=A/np.linalg.norm(A,axis=1,keepdims=True)
        S=An@An.T; np.fill_diagonal(S,np.nan)
        gaps=[S[i][~np.isnan(S[i])].mean() for i in range(len(A))]
        res["xdoc_cos_mean"]=float(np.nanmean(gaps))
        lk=[m1[k].get("lookup",[None,None])[0] for k in own]
        g=list(gaps)
        rho=float(np.corrcoef(np.array(lk),np.array(g))[0,1]) if len(set(lk))>2 else None
        res["gap_x_lookup_pearson"]=rho
if m2:
    cor=curves(m2,"cor_w"); res["corpus"]=summarize("corpus",cor)
if m1 and m2:
    o=res["own"]; c=res["corpus"]
    res["lens_read"]=dict(dip_own=o["dip"],dip_corpus=c["dip"],
        # FIX audit2 B4/D6: авто-вердикт снят (числа — в артефакт, вердикт — человеком после E-72/E-73)
        verdict="NO-AUTO-VERDICT (audit2 B4): порядок ни с одной пре-рег сигнатурой не совпадает; naturalness-конфаунд открыт до E-72/E-73")
print(json.dumps(res,indent=1,ensure_ascii=False))
json.dump(res,open(os.path.join(d,"READMATRIX.json"),"w"),indent=1,ensure_ascii=False)
