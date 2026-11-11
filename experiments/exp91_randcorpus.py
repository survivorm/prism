#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9-2 (audit2 H2, r2): клетки (trained|full-random|half-random) × (corpus|shuffled).
Конфаунд «dip выучен»: прежний R-null = случайные веса × СВОЙ МУСОР (вырожденный вход)
→ dip=0 по построению оценщика. Разделяем: (случайные|обученные) × (корпус|перемешанный).
half-random = обученные embeddings + final norm (model.norm) + layer-нормы
(input_layernorm/post_attention_layernorm) + lm_head (tied); ВСЕ трансформер-блоки —
случайные (full/linear-attn, mlp, вкл. внутренний norm linear_attn — он часть блока);
visual/mtp — случайные (в текстовом forward не участвуют — записано в артефакт).
Вердикт-правило (M9): half-random × corpus: |dip|≤0.02 → «выучено» живёт;
dip<−0.02 → метрика меряет вход. full-random × corpus dip<−0.02 — отдельный сильный
флаг токен-статистики (не вердикт).
Оценщик ОДИН И ТОТ ЖЕ код-путь для всех клеток (curve_u + cos lag-1, L23, fp32,
output_hidden_states[23]; trained пересчитан заново, не npz).
SANITY-пины ПЕРВЫМИ (до доверия random-клеткам, печать в лог):
 (a) смоук 1 док×32 токена: finiteness hidden у full-random (оба сида);
 (b) регресс-чек trained×corpus = −0.0722±0.002 (M1) — если мимо, STOP (RC=2);
 (c) ESTIMATOR_PIN.json загружен; sha256 используемых функций (curve_u, dip_path) — в артефакт.
Пул = канонический пул пина: LADDER_V2.json meta.nonloopy ∩ nonwiki (n=35) —
ровно тот, на котором получен −0.0722 (M1: pooled nonwiki, ladder v2b).
Сид init (M9): torch.manual_seed(S) ПЕРЕД from_config, S∈{11,12}, значения в артефакт;
блоки half-random побайто идентичны блокам full-random того же сида (тот же init-поток,
обученный сабсет поверх).
Shuffled-последовательности ОДНИ для всех клеток (rng(7), по докам) — парное сравнение (M2).
"""
import hashlib, inspect, json, os, sys
from datetime import datetime
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"4")
import numpy as np, torch, transformers
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp62_batt_lib import curve_u
from transformers import AutoModelForCausalLM, AutoConfig
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ="/data/prism_kernels/exp62/gen_layers_qwen.npz"; L=23; BASE="/data/Qwen/Qwen3.5-0.8B"
SEEDS=[11,12]; PIN_DIP=-0.0722; PIN_TOL=0.002
PINFILE=ROOT+"/results/e64/ESTIMATOR_PIN.json"
OUT=ROOT+"/results/e92/R92_RANDCORPUS.json"; MAN=ROOT+"/results/ARTIFACTS_MANIFEST.json"
import gc
z=np.load(NPZ,allow_pickle=True)
meta=json.load(open(NPZ.replace(".npz","_meta.json")))
dom={n:meta[n]["dom"] for n in meta}
# пул = канонический пул пина (LADDER_V2: nonloopy ∩ nonwiki), не весь nonwiki
lad=json.load(open(ROOT+"/results/e90/LADDER_V2.json"))
names=[n for n in sorted(lad["meta"]["nonloopy"]) if dom.get(n)!="wiki"]
if not names:
    print("FATAL: пустой пул (LADDER_V2 meta.nonloopy ∩ nonwiki) — STOP (RC=5)",flush=True); sys.exit(5)

def dip_path(model,toks):
    with torch.no_grad():
        o=model(input_ids=torch.tensor([toks]),output_hidden_states=True)
    h=o.hidden_states[L][0,:len(toks)].double().numpy()
    hn=np.linalg.norm(h,axis=1,keepdims=True); u=h/np.maximum(hn,1e-12)
    P=curve_u(u,hn.ravel()); Pn=P/np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-12)
    c1=np.einsum("ij,ij->i",Pn[:-1],Pn[1:])
    return float(np.median(c1))

docs=[]
for nm in names:
    toks=np.asarray(z[f"{nm}_tok"]).ravel().astype(int).tolist()
    if len(toks)>=64: docs.append((nm,toks[:256]))
print(f"docs={len(docs)}",flush=True)
# shuffled-последовательности общие для всех клеток (парное сравнение, M2)
rngl=np.random.default_rng(7)
docs2=[(nm,toks,list(map(int,rngl.permutation(toks)))) for nm,toks in docs]

def cells(model,ds):
    out={"corpus":[],"shuffled":[]}
    for nm,toks,sh in ds:
        out["corpus"].append(dip_path(model,toks))
        out["shuffled"].append(dip_path(model,sh))
    return out

def fn_sha(fn):
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()

def sanitize(o):
    if isinstance(o,dict): return {k:sanitize(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [sanitize(v) for v in o]
    if isinstance(o,np.floating): o=float(o)
    elif isinstance(o,np.integer): return int(o)
    if isinstance(o,float) and (np.isnan(o) or np.isinf(o)): return None
    return o

sanity={}
# (c) ESTIMATOR_PIN: загрузка + sha256 используемых функций → в артефакт
ep_raw=open(PINFILE,"rb").read()
est_pin=json.loads(ep_raw.decode("utf-8"))
sanity["estimator_pin"]=dict(
    file_sha256_16=hashlib.sha256(ep_raw).hexdigest()[:16],
    pyc_sha256=est_pin.get("pyc_sha256"),
    pyc_bytecode_match=bool(est_pin.get("bytecode_match")),
    restored_fns=list(est_pin.get("restored_fns",[])),
    fns_used_sha256=dict(curve_u=fn_sha(curve_u), dip_path=fn_sha(dip_path)))
print("SANITY(c) ESTIMATOR_PIN loaded: pyc_match=%s | fns_used sha256: curve_u=%s… dip_path=%s…"
      % (sanity["estimator_pin"]["pyc_bytecode_match"],
         sanity["estimator_pin"]["fns_used_sha256"]["curve_u"][:12],
         sanity["estimator_pin"]["fns_used_sha256"]["dip_path"][:12]),flush=True)

res={}
mt=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float32).eval(); mt.requires_grad_(False)
res["trained"]=cells(mt,docs2); print("trained готов",flush=True)
# (b) регресс-пин ДО доверия random-клеткам
tv=float(np.median(res["trained"]["corpus"]))
t_ok=abs(tv-PIN_DIP)<=PIN_TOL
print(f"SANITY(b) PIN trained×corpus dip={tv:.4f} (target {PIN_DIP}±{PIN_TOL}) -> {'PASS' if t_ok else 'FAIL'}",flush=True)
if not t_ok:
    for nm,v in zip([d[0] for d in docs2],res["trained"]["corpus"]):
        print(f"  per-doc {nm}: {v:.4f}",flush=True)
    try:
        lad=json.load(open(ROOT+"/results/e90/LADDER_V2.json"))
        pool35={n for n in lad["meta"]["nonloopy"] if dom.get(n)!="wiki"}
        sub=[v for nm,v in zip([d[0] for d in docs2],res["trained"]["corpus"]) if nm in pool35]
        print(f"DIAG: LADDER pool35 (nonloopy∩nonwiki, n={len(sub)}) median = {float(np.median(sub)):.4f}",flush=True)
    except Exception as e:
        print(f"DIAG: re-aggregation по LADDER-пулу не удалась: {e}",flush=True)
    print("PIN FAIL — STOP: random-клеткам не доверяем (RC=2)",flush=True)
    sys.exit(2)
trained_subset={k:mt.state_dict()[k] for k in mt.state_dict()
                if k in ("model.embed_tokens.weight","model.norm.weight","lm_head.weight")
                or ".input_layernorm." in k or ".post_attention_layernorm." in k}
print(f"trained_subset: {len(trained_subset)} keys (embed+final norm+layer norms+lm_head); блоки/visual/mtp — random",flush=True)
del mt; gc.collect()

for sd in SEEDS:
    torch.manual_seed(sd)
    cfg=AutoConfig.from_pretrained(BASE)
    mr=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32); mr.requires_grad_(False); mr.eval()
    # (a) смоук 1 док×32 токена: finiteness hidden
    stoks=docs2[0][1][:32]
    with torch.no_grad():
        sm=mr(input_ids=torch.tensor([stoks]),output_hidden_states=True)
    hs=sm.hidden_states[L][0,:32]
    fin=bool(torch.isfinite(hs).all())
    print(f"SANITY(a) SMOKE full-random seed={sd}: 1doc({docs2[0][0]})×32tok hidden L{L} "
          f"min={float(hs.min()):.4g} max={float(hs.max()):.4g} finite={'PASS' if fin else 'FAIL'}",flush=True)
    if not fin:
        print(f"SMOKE FAIL seed={sd} — STOP (RC=3)",flush=True)
        sys.exit(3)
    sanity[f"smoke_rand{sd}"]=dict(doc=docs2[0][0],n_tok=32,finite=True,
                                    min=float(hs.min()),max=float(hs.max()))
    res[f"rand{sd}"]=cells(mr,docs2); print(f"rand{sd} готов",flush=True)
    del mr; gc.collect()

for sd in SEEDS:
    torch.manual_seed(sd)
    cfg=AutoConfig.from_pretrained(BASE)
    mh=AutoModelForCausalLM.from_config(cfg,dtype=torch.float32); mh.requires_grad_(False); mh.eval()
    n_copied=0
    with torch.no_grad():
        for name,p in mh.named_parameters():
            if name in trained_subset:
                if tuple(p.shape)!=tuple(trained_subset[name].shape):
                    print(f"HALF seed={sd} SHAPE MISMATCH: {name} — STOP (RC=4)",flush=True); sys.exit(4)
                p.copy_(trained_subset[name]); n_copied+=1
    bad=[k for k in trained_subset if not torch.equal(mh.state_dict()[k],trained_subset[k])]
    if bad:
        print(f"HALF seed={sd} VERIFY FAILED: {bad} — STOP (RC=4)",flush=True); sys.exit(4)
    sanity[f"half{sd}"]=dict(params_copied=n_copied,verified_keys=len(trained_subset),
                             note="блоки побайто = full-random того же сида (один init-поток; проверено torch.equal по всем ключам)")
    res[f"half{sd}"]=cells(mh,docs2); print(f"half{sd} готов (copied={n_copied}, verified={len(trained_subset)})",flush=True)
    del mh; gc.collect()

tab={}
for k,v in res.items():
    tab[k]={m:[float(np.median(v[m])),
               float(np.percentile(v[m],2.5)),float(np.percentile(v[m],97.5))] for m in v}
hv={f"half{sd}":tab[f"half{sd}"]["corpus"][0] for sd in SEEDS}
rv={f"rand{sd}":tab[f"rand{sd}"]["corpus"][0] for sd in SEEDS}
if all(abs(v)<=0.02 for v in hv.values()):
    verdict="learned_survives: |dip|<=0.02 у half-random — «выучено» живёт"
elif any(v<-0.02 for v in hv.values()):
    verdict="metric_measures_input: dip<-0.02 у half-random — метрика меряет вход"
else:
    verdict="out-of-spec: half-random dip>+0.02 — вердикт-правило M9 не покрыто, нужен разбор"
out=dict(
 n_docs=len(docs),
 pool_source="results/e90/LADDER_V2.json meta.nonloopy ∩ nonwiki (канон-пул пина −0.0722, M1)",
 docs=[d[0] for d in docs2],
 pool="nonwiki, len>=64, первые 256 ток, L=23, fp32, пересчёт (не npz)",
 L=L,dtype="float32",
 seeds=SEEDS,
 init_seed_policy="torch.manual_seed(S) перед from_config (M9)",
 half_random_def=dict(
   trained_subset="model.embed_tokens + model.norm + input_layernorm/post_attention_layernorm (×24) + lm_head",
   random="все трансформер-блоки (full/linear-attn, вкл. внутренний norm linear_attn, mlp) + visual + mtp (не в текстовом forward)",
   note="блоки(half_sd) побайто = блоки(rand_sd) при том же сиде"),
 shuffle=dict(seed=7,shared_across_cells=True),
 sanity=sanity,
 pin=dict(trained_corpus=float(tv),target=PIN_DIP,tol=PIN_TOL,ok=True),
 table=tab,
 verdict=dict(
   half_random=verdict,
   half_random_corpus_dips=hv,
   full_random_corpus_dips=rv,
   full_random_token_stat_flag=bool(any(v<-0.02 for v in rv.values())),
   trained_corpus=float(tab["trained"]["corpus"][0])),
 env=dict(torch=torch.__version__,transformers=transformers.__version__,
          n_threads=torch.get_num_threads()))
os.makedirs(ROOT+"/results/e92",exist_ok=True)
json.dump(sanitize(out),open(OUT,"w"),indent=1,ensure_ascii=False)
raw=open(OUT,"rb").read()
mj=json.load(open(MAN))
mj["results/e92/R92_RANDCORPUS.json"]=dict(
    sha256_16=hashlib.sha256(raw).hexdigest()[:16],size=len(raw),
    date=datetime.now().isoformat(timespec="seconds"))
json.dump(mj,open(MAN,"w"),indent=1,ensure_ascii=True)
print("\nR9-2 медианы по клеткам (corpus/shuffled); [2.5;97.5] — в артефакте:")
print(f"{'cell':<10} {'corpus':>9} {'shuffled':>9}")
for k in ("trained","rand11","rand12","half11","half12"):
    print(f"{k:<10} {tab[k]['corpus'][0]:>9.4f} {tab[k]['shuffled'][0]:>9.4f}")
print("VERDICT:",verdict)
print("R9-2 DONE",flush=True)
