# E-75 TMPL×FAMILY (CPU): чиним LFM3-протокол и гоняем тесты пользователя U-D/U-E.
# 1) ВХОД: тексты wiki-пула (декод Qwen-id = канонические тексты серии) → ЭНКОДИРУЮТСЯ КАЖДой
#    МОДЕЛЬЮ СВОИМ токенизатором. В артефакт пишем: tokenizer id, sha текста, декод-проба глазами.
# 2) ФОРМЫ: raw (голый текст) vs tmpl (текст внутри chat-шаблона как user-сообщение; метрики
#    считаются только по зоне текста: NLL маскируется на заголовок шаблона, h-срез по смещению).
# Клетки: {qwen35_08b, lfm3_17b} × {raw, tmpl} × 16 wiki-доков teacher-forced.
# Пре-рег: U-D (внутрифамильный эффект шаблона): |Δdip(qwen tmpl−raw)| и σ-норма; ожидание автора —
#   «различаются здорово». U-E: эффект шаблона LFM против Qwen (если больше — выровненность/шаблон
#   объясняет часть прежнего «межфамильного» расхождения). Честное межфамильное сравнение поля —
#   только в клетках raw-raw (замена отозванной LFM3_open).
import ast, hashlib, json, os, time
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
OUT="results/e75"; os.makedirs(OUT,exist_ok=True)
LAYERS=[11,23]; MAXLEN=448
src=open("experiments/kaggle_e64gpu/e64_gpu.py").read()
tree=ast.parse(src); fns={n.name:ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
ns={"np":np}
for w in ["vpath_units","dcurve","curves_series"]: exec(fns[w],ns)
curves_series=ns["curves_series"]
pool=load_calibration_data("Qwen/Qwen3.5-0.8B",max_length=256,min_length=64,n_samples=16,domain="wiki")
qw0=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
texts=[qw0.decode(d["input_ids"][0].tolist(),skip_special_tokens=True)[:1600] for d in pool]
FN=f"{OUT}/TMPL_FAMILY.json"
res={"provenance":{"texts_sha":hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12],
     "n_texts":len(texts),"sample":" "+texts[0][:80]}}
if os.path.exists(FN): res.update(json.load(open(FN)))
def save(): json.dump(res,open(FN,"w"),default=float)
def wrap(tok,txt):
    try: return tok.apply_chat_template([{"role":"user","content":txt}],add_generation_prompt=True,tokenize=False)
    except Exception: return "USER: "+txt+"\nASSISTANT:"
MODELS=[("qwen","Qwen/Qwen3.5-0.8B"),("lfm","LiquidAI/LFM3-1.7B")]
for tag,mid in MODELS:
    try:
        tk=AutoTokenizer.from_pretrained(mid,trust_remote_code=True)
        m=AutoModelForCausalLM.from_pretrained(mid,trust_remote_code=True,dtype=torch.float32).eval()
    except Exception as e:
        print("!! load",tag,e,flush=True); res.setdefault("errors",[]).append(f"{tag}:{e}"); save(); continue
    caps={}
    hs=[m.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in LAYERS]
    for form in ["raw","tmpl"]:
        ckey=f"{tag}/{form}"
        if ckey in res: print("skip",ckey,flush=True); continue
        rows={}
        for di,txt in enumerate(texts):
            try:
                if form=="raw": hdr=""; body=txt
                else:
                    full=wrap(tk,txt)
                    hdr_s=wrap(tk,"")
                    hdr=full[:len(hdr_s)]   # префикс шаблона до контента
                    body=txt
                ids_h=tk(hdr,add_special_tokens=False)["input_ids"]
                ids_b=tk(body,add_special_tokens=False)["input_ids"][:MAXLEN]
                if form=="tmpl":  # хвост шаблона (ASSISTANT:…) к тексту не относится — режем тело из full
                    ids=(tk(full,add_special_tokens=False)["input_ids"])
                    ids=ids[:len(ids_h)+len(ids_b)]
                    P=len(ids_h)
                else:
                    ids=ids_b; P=0
                t=torch.tensor([ids]); caps.clear()
                with torch.no_grad(): o=m(input_ids=t,use_cache=False)
                lg=o.logits[0,P-1:len(ids)-1].float() if P>0 else o.logits[0,:len(ids)-1].float()
                lb=torch.tensor(ids[P if P>0 else 1:len(ids)])
                nll=float(torch.nn.functional.cross_entropy(lg,lb))
                row={"nll":nll,"n_tok":len(ids),"P":P}
                for L in LAYERS:
                    H=caps[L][0,P if P>0 else 0:len(ids)].to(torch.float32).numpy()
                    c=curves_series(H); row[f"L{L}"]=dict(dip=c["dip"],hump=c["hump"],plateau=c["plateau"])
                rows[f"wiki_{di}"]=row
                print(f"[{ckey}] wiki_{di} nll {nll:.2f} dipL23 {row['L23']['dip']:+.3f}",flush=True)
                res[ckey]=rows; save()
            except Exception as e:
                print("!!",ckey,di,str(e)[:120],flush=True); res.setdefault("errors",[]).append(f"{ckey}/{di}:{e}")
    for h in hs: h.remove()
    res[ckey.split('/')[0]+"_meta"]=dict(tok_mid=mid,tok_vocab=tk.vocab_size,
        decode_probe=tk.decode(tk(texts[0],add_special_tokens=False)["input_ids"][:12])[:70])
    save()
    del m; import gc; gc.collect()
# вердикты
try:
    import statistics as st
    def md(a,b,L="L23"):
        ks=[k for k in res.get(a,{}) if k in res.get(b,{})]
        return float(np.median([res[b][k][L]["dip"]-res[a][k][L]["dip"] for k in ks])),len(ks)
    sdq=float(np.std([res["qwen/raw"][k]["L23"]["dip"] for k in res["qwen/raw"]]))
    sdl=float(np.std([res["lfm/raw"][k]["L23"]["dip"] for k in res["lfm/raw"]]))
    dq,nq=md("qwen/raw","qwen/tmpl"); dl,nl=md("lfm/raw","lfm/tmpl")
    res["verdict"]=dict(U_D_qwen_tmpl_shift=round(dq,4),U_D_qwen_sig=round(dq/sdq,2),
        U_E_lfm_tmpl_shift=round(dl,4),U_E_lfm_sig=round(dl/sdl,2),
        honest_cross_family_raw=dict(qwen=round(float(np.median([res["qwen/raw"][k]["L23"]["dip"] for k in res["qwen/raw"]])),4),
                                     lfm=round(float(np.median([res["lfm/raw"][k]["L23"]["dip"] for k in res["lfm/raw"]])),4)),
        n=(nq,nl))
except Exception as e:
    res["verdict"]=dict(error=str(e)[:200])
save(); print(json.dumps(res["verdict"],indent=1),flush=True); print("E75 DONE",flush=True)
