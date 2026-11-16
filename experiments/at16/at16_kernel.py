import subprocess, sys, os, json, hashlib, time, re
t0=time.time()
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==5.15.1","tokenizers==0.22.2","datasets==5.0.1"])
import numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
IN="/kaggle/input"
def locate(fname):
    for root,dirs,files in os.walk(IN):
        if fname in files: return os.path.join(root,fname)
    raise FileNotFoundError(fname)
CTX=json.load(open(locate("AT16_contexts.json")))
print("16k conds:",{k:len(v) for k,v in CTX.items()},flush=True)
os.makedirs("/data/Qwen",exist_ok=True); tgt="/data/Qwen/Qwen3.5-0.8B"
if not os.path.exists(tgt): os.symlink(snapshot_download("Qwen/Qwen3.5-0.8B"),tgt)
m=AutoModelForCausalLM.from_pretrained(tgt,dtype=torch.float32).to("cuda").eval()
tk=AutoTokenizer.from_pretrained(tgt)
KEYS={"склад":"MKL-4082","сервер":"QRD-7731","лицензи":"ZTR-1195","домофон":"BVX-6620"}
BARE=[("склад","Вопрос: какой код склада? Ответ:"),("сервер","Вопрос: какой пароль сервера? Ответ:"),
    ("лицензи","Вопрос: какой номер лицензии? Ответ:"),("домофон","Вопрос: какой код домофона? Ответ:")]
QS=[(w+"|bare",q) for w,q in BARE]+[(w+"|nudge","Прочитай текст выше внимательно и ответь строго по фактам из него, без общих слов. "+q) for w,q in BARE]
REFUS=["не могу найти","не могу ответить","не могу обработать","не упомина","не указан","не могу прочитать","к сожалению","i cannot","i can't","unable"]
out={}
for cond,ids in CTX.items():
    ctxp=torch.tensor([ids],dtype=torch.long,device="cuda")
    for what,q in QS:
        x=torch.cat([ctxp,torch.tensor([tk(q,add_special_tokens=False)["input_ids"]],dtype=torch.long,device="cuda")],dim=1)
        gen=[]
        with torch.no_grad():
            cache=None; o=m(input_ids=x,use_cache=True); cache=o.past_key_values
            for _ in range(48):
                nx=int(o.logits[0,-1].argmax()); gen.append(nx)
                if len(gen)>=48: break
                with torch.no_grad(): o=m(input_ids=torch.tensor([[nx]],device="cuda"),past_key_values=cache,use_cache=True)
        ans=tk.decode(gen)
        al=ans.lower()
        hit=float(KEYS[what] in ans)
        first=re.search(r'[A-Z]{3}-\d{4}',ans)
        refusal=any(r in al for r in REFUS)
        cls="hit" if hit else ("refusal" if refusal else ("template" if first and first.group()!=KEYS[what] else "garbage"))
        out.setdefault(cond,{})[what]={"hit":hit,"cls":cls,"ans":ans[:80]}
        print(cond,what,cls,"%.0fs"%(time.time()-t0),flush=True)
json.dump(out,open("/kaggle/working/AT16_needle.json","w"),indent=1)
summ={c:{"hit_bare":round(float(np.mean([v["hit"] for k,v in d.items() if k.endswith("bare")])),3),
         "hit_nudge":round(float(np.mean([v["hit"] for k,v in d.items() if k.endswith("nudge")])),3),
         "refusals":sum(1 for v in d.values() if v["cls"]=="refusal")} for c,d in out.items()}
print("SUMMARY",json.dumps(summ),flush=True)
print("ALL DONE",time.time()-t0,flush=True)
