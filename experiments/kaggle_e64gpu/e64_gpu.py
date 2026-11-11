#!/usr/bin/env python3
"""Восстановлен 11.11 из __pycache__/e64_gpu.cpython-312.pyc (git-инцидент §42 снёс исходник;
pyc 08.11 15:24, 34968 B) по диссекции байткода (python3.12 dis). Восстановлены ТРИ функции,
которые тянут все 8 потребителей (exp65/65b/66/68/69/71/73/75, want=["vpath_units","dcurve","curves_series"]):
  vpath_units, dcurve, curves_series.
Остальной исходник ядра (log/find/sha/prompt_ids/_mk/mark/fwd_buf/run_doc/gen_corpus/
lookup_cov/save_npz/a5/ma/stat) не восстанавливается — потребители его не используют;
при запросе посторонней функции AST-сплайс упадёт (assert not missing) — fail-loud по design.
ПИНЫ: sha256 pyc и кода функций записаны в results/e64/ESTIMATOR_PIN.json (генератор:
experiments/_pin_estimator.py). curves_series — ВАРИАНТ С ПИНАМИ (адаптивный kk=min(96, len(P)//3),
nan-гарды при коротких рядах), НЕ совпадает с b4_reshot.py (фикс. kk=1..96 без гардов)."""
import numpy as np

def dcurve(P,kk):
    """сериийный оценщик (exp64_b4_battery/A5): median cos по t, без средних."""
    Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    return np.array([float(np.median(np.einsum("ij,ij->i",Pn[:-k],Pn[k:]))) for k in kk])

def vpath_units(u):
    """сериийный Vpath: ортогонализованная составляющая шага между единичными состояниями."""
    Un=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
    Vs=[]
    for t in range(len(Un)-1):
        a=float(Un[t]@Un[t+1]); vv=Un[t+1]-a*Un[t]
        if np.linalg.norm(vv)>1e-6: Vs.append(vv/np.linalg.norm(vv))
    return np.array(Vs)

def curves_series(Vh):
    """Vh [T,H] сырые h -> {dip,hump,plateau,curve{}} в ТОЧНОСТИ батареи B4 (median, ортогональн.шаг)."""
    Vh=Vh.astype(np.float64)
    u=Vh/np.maximum(np.linalg.norm(Vh,axis=1,keepdims=True),1e-12)
    P=vpath_units(u)
    kk=np.arange(1,min(96,max(2,len(P)//3))+1)
    c=dcurve(P,kk)
    return dict(dip=float(c[0]),hump=float(c[1:8].max()) if len(c)>8 else float("nan"),
                plateau=float(np.median(c[8:])) if len(c)>9 else float("nan"),
                curve={str(int(k)):float(x) for k,x in zip(kk,c)})
