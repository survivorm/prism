# at16_ceiling.py — потолок чтения (пре-рег §16 + протокол v2 после потери 16k-контекстов)
# ctx8k  = токены из AT8_contexts.json (датасет calib-pins v10)
# ctx16k = тот же текст, продекодированный и склеенный с копией БЕЗ ключевых фраз (ключ один, глубина 0.75)
# probe: forward без кэша, logits только последней строки (lm_head по одному ряду) — память T4-safe
import subprocess, sys, os, json, time, re
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2"])
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
IN="/kaggle/input"
def locate(fname):
    for root,dirs,files in os.walk(IN):
        if fname in files: return os.path.join(root,fname)
    raise FileNotFoundError(fname)
CTX=json.load(open(locate("AT8_contexts.json")))
os.makedirs("/data/Qwen",exist_ok=True); tgt="/data/Qwen/Qwen3.5-0.8B"
from huggingface_hub import snapshot_download
if not os.path.exists(tgt): os.symlink(snapshot_download("Qwen/Qwen3.5-0.8B"),tgt)
m=AutoModelForCausalLM.from_pretrained(tgt,dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
tk=AutoTokenizer.from_pretrained(tgt)
KEYS={"склад":"MKL-4082","сервер":"QRD-7731","лицензи":"ZTR-1195","домофон":"BVX-6620"}
def keypat(w):  # ключ + варианты обрамления
    k=KEYS[w]; return re.compile(re.escape(k))
BARE=[("склад","Вопрос: какой код склада? Ответ:"),("сервер","Вопрос: какой пароль сервера? Ответ:"),
    ("лицензи","Вопрос: какой номер лицензии? Ответ:"),("домофон","Вопрос: какой код домофона? Ответ:")]
QS=[(w+"|bare",q) for w,q in BARE]+[(w+"|nudge","Прочитай текст выше внимательно и ответь строго по фактам из него, без общих слов. "+q) for w,q in BARE]
REFUS=["не могу найти","не могу ответить","не могу обработать","не упомина","не указан","не могу прочитать","к сожалению","i cannot","i can't","unable"]

def last_logits(ids):  # [1,L] -> [1,V] последняя позиция (для префилла без cache-logits)
    with torch.no_grad():
        h=m.model(input_ids=ids).last_hidden_state
        lg=m.lm_head(h[0,-1:])
    return lg.float()

def probe(cids,q):
    x=torch.tensor([cids+tk(q,add_special_tokens=False)["input_ids"]],dtype=torch.long,device="cuda")
    with torch.no_grad():
        # префилл без полных logits (память), затем микро-шаги с kv-кэшем
        h=m.model(input_ids=x).last_hidden_state
        nx=int(m.lm_head(h[0,-1:]).float().argmax())
        out=m(input_ids=x,past_key_values=None,use_cache=True)
        cache=out.past_key_values
        gen=[]
        for i in range(48):
            gen.append(nx)
            out=m(input_ids=torch.tensor([[nx]],device="cuda"),past_key_values=cache,use_cache=True)
            nx=int(out.logits[0,-1].float().argmax())
    ans=tk.decode(gen)
    return ans,ans.lower()

def classify(ans,al,w):
    hit=float(KEYS[w] in ans)
    first=re.search(r'[A-Z]{3}-\d{4}',ans)
    refusal=any(r in al for r in REFUS)
    cls="hit" if hit else ("refusal" if refusal else ("template" if first and first.group()!=KEYS[w] else "garbage"))
    return hit,cls

def build16(ids8):
    txt=tk.decode(ids8)
    txt2=txt
    for w in KEYS: txt2=re.sub(r'[^.]*?'+re.escape(KEYS[w])+r'[^.]*\.',' ',txt2)
    tail=tk(txt2,add_special_tokens=False)["input_ids"]
    while len(tail)<8192: tail=tail+tail
    out=ids8+tail
    return out[:16384]

RES={}
for cond,ids in CTX.items():
    ids8=ids if isinstance(ids,list) else ids["ids"]
    if isinstance(ids8[0],list): ids8=ids8[0]
    c16=build16(ids8)
    for cname,cids in [("8k",ids8),("16k",c16)]:
        key=cond+"_"+cname; RES[key]={}
        for what,q in QS:
            w=what.split("|")[0]
            ans,al=probe(cids,q)
            hit,cls=classify(ans,al,w)
            RES[key][what]={"hit":hit,"cls":cls,"ans":ans[:80]}
            print(key,what,cls,"%.0fs"%(time.time()-t0),flush=True)
        json.dump(RES,open("/kaggle/working/AT_CEILING_v2.json","w"),indent=1,ensure_ascii=False)
summ={}
for k,d in RES.items():
    summ[k]={"hit_bare":round(float(np.mean([v["hit"] for kk,v in d.items() if kk.endswith("bare")])),3),
             "hit_nudge":round(float(np.mean([v["hit"] for kk,v in d.items() if kk.endswith("nudge")])),3)}
print("SUMMARY",json.dumps(summ),flush=True)
json.dump({"res":RES,"summ":summ},open("/kaggle/working/AT_CEILING_v2.json","w"),indent=1,ensure_ascii=False)
print("ALL DONE %.0fs"%(time.time()-t0),flush=True)
