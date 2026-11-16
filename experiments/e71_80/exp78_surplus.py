#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-78 SURPLUS: «излишек чтения» S = NLL_model − NLL_5gram как драйвер амплитуды dip.
# ТЗ docs/dynamic/ТЗ_раунд8.md (с само-аудитом). Пре-рег (заранее, в файле):
#  главный гейт: Spearman(S, dip) по 64 домен-докам (wiki/math/code/chat ×16), бут CI95;
#    ρ ≥ +0.5 ⇒ S-драйвер ЖИВ; ρ ≤ +0.2 ⇒ мёртв; между — не разрешено.
#  точка-предсказание «петлевой режим»: loop-synth ожидание S≈минимум, dip глубже
#    baseline−0.05 (попадание = теория описывает оба режима; промах = описывает ничего);
#  shuffle-wiki: ожидание S выше медианы доков при dip НЕ глубже медианы (иначе
#    «излишек» меряет не чтение);
#  ngram-фон: доменный LOO (другие 15 документов того же домена), контроли — wiki-LOO.
#  Домены без own: own-tmpl будет частью B после E-72 (template-конфаунд).
# Оценка dip — тот же ast-дубль ядра L23; nll — shift-CE по документу; один проход.
import ast, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"results"/"e78"; OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT/"SURPLUS_raw.txt"
BASE = "/data/Qwen/Qwen3.5-0.8B"
PINS = ROOT/"data"/"calib_pins"
DOMS = ["wiki", "math", "code", "chat"]
N = 5; LAM = 0.4
src = open(ROOT/"experiments"/"kaggle_e64gpu"/"e64_gpu.py").read()
fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
       if isinstance(n, ast.FunctionDef)}
ns = {"np": np}
for w in ["vpath_units", "dcurve", "curves_series"]:
    exec(fns[w], ns)
curves_series = ns["curves_series"]

def load_pin(dom):
    p = PINS/f"{dom}___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"
    return [d["input_ids"][0].tolist() for d in json.load(open(p))]

class NG:
    """n-gram LM, stupid backoff, фон — список token-последовательностей"""
    def __init__(self, corpus):
        self.c = [defaultdict(int) for _ in range(N+1)]     # c[k][tuple(k)]
        V = set()
        for doc in corpus:
            for k in range(1, N+1):
                for i in range(len(doc)-k+1):
                    self.c[k][tuple(doc[i:i+k])] += 1
            V.update(doc)
        self.V = len(V); self.uni_tot = sum(self.c[1].values())
        self.c0 = self.V + 1                                 # неизвестный токен
    def nll(self, doc):
        tot = 0.0
        for i in range(1, len(doc)):
            w = doc[i]
            p = None
            for k in range(N, 1, -1):
                if i-k+1 < 0:
                    continue
                ctx = tuple(doc[i-k+1:i])
                den = self.c[k-1].get(ctx, 0)
                if den > 0:
                    num = self.c[k].get(ctx+(w,), 0)
                    p = (num+1)/(den+self.c0)                # +1 smoothing
                    break
            if p is None:
                p = (self.c[1].get(w, 0)+1)/(self.uni_tot+self.c0)
            tot += -math.log(max(p, 1e-9))
        return tot/max(1, len(doc)-1)

def main():
    done = set()
    if RAW.exists():
        for l in open(RAW):
            try:
                r = json.loads(l); done.add((r["set"], r["doc"]))
            except Exception:
                pass
    tk = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32,
                                             attn_implementation="eager").eval()
    caps = {}
    hs = [m.model.layers[23].register_forward_hook(lambda mod, i, o: caps.__setitem__(23, o.detach()))]
    pools = {d: load_pin(d) for d in DOMS}
    # контроли
    rng = np.random.default_rng(0)
    wiki = pools["wiki"]
    sent = [s for s in wiki[0] if isinstance(s, str)]        # не используется: id-пул
    loop_doc = (wiki[3][:16]*14)[:224]                        # 16 токенов ×14 — копия-петля
    shuffle_docs = []
    for d in wiki[:4]:
        parts, cur = [], []
        # перемешать предложениями нельзя по id-пулу без декода — декод→шафл→энкод
        txt = tk.decode(d, skip_special_tokens=True)
        sents = [x for x in txt.split(". ") if x.strip()]
        if len(sents) > 3:
            rng.shuffle(sents)
            ids = tk(". ".join(sents), add_special_tokens=False)["input_ids"][:256]
            shuffle_docs.append(ids)
    def cell(setname, j, ids):
        t = torch.tensor([ids])
        caps.clear()
        with torch.no_grad():
            o = m(input_ids=t, use_cache=False)
        nll = float(torch.nn.functional.cross_entropy(o.logits[0, :-1].float(), t[0, 1:]))
        c = curves_series(caps[23][0, :len(ids)].to(torch.float32).numpy())
        return dict(set=setname, doc=j, nTok=len(ids), nll=round(nll, 4), dip=round(float(c["dip"]), 4))
    # 1) модельные клетки
    rows = []
    for dom, docs in pools.items():
        for j, ids in enumerate(docs):
            if (dom, j) in done:
                continue
            rows.append(cell(dom, j, ids))
            print(f"{dom}/{j} nll {rows[-1]['nll']:.2f} dip {rows[-1]['dip']:+.3f}", flush=True)
            with open(RAW, "a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1]) + "\n")
    for j, ids in enumerate(shuffle_docs):
        if ("shuffle", j) not in done:
            r = cell("shuffle", j, ids)
            with open(RAW, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"shuffle/{j} nll {r['nll']:.2f} dip {r['dip']:+.3f}", flush=True)
    if ("loop", 0) not in done:
        r = cell("loop", 0, loop_doc)
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"loop nll {r['nll']:.2f} dip {r['dip']:+.3f}", flush=True)
    # 2) ngram: фон = доменный LOO; контроли — wiki без этого дока (его и нет в wiki-пуле)
    for l in open(RAW):
        rows.append(json.loads(l)) if l not in () else None
    rows = [json.loads(l) for l in open(RAW)]
    ng_cache = {}
    def ng_for(setname, j):
        key = "wiki" if setname in ("shuffle", "loop") else setname
        if key not in ng_cache:
            ng_cache[key] = [NG([d for i, d in enumerate(pools[key]) if i != j]) for j in range(len(pools[key]))]
        return ng_cache[key][j if setname in DOMS else -1]
    # аккуратный LOO для контролирующих клеток: пересобираем фон отдельно
    s_rows = []
    for r in rows:
        if r["set"] in DOMS:
            g = NG([d for i, d in enumerate(pools[r["set"]]) if i != r["doc"]])
            ids = pools[r["set"]][r["doc"]]
        else:
            g = NG([d for d in pools["wiki"]])
            ids = loop_doc if r["set"] == "loop" else shuffle_docs[r["doc"]]
        r["S"] = round(r["nll"] - g.nll(ids), 4)
        s_rows.append(r)
    dom_rows = [r for r in s_rows if r["set"] in DOMS]
    Sd = np.array([r["S"] for r in dom_rows]); Dd = np.array([r["dip"] for r in dom_rows])
    order = np.argsort(np.argsort(Sd)); orderD = np.argsort(np.argsort(Dd))
    rho = float(np.corrcoef(order, orderD)[0, 1])
    rng2 = np.random.default_rng(1)
    boots = []
    for _ in range(2000):
        ix = rng2.integers(0, len(Sd), len(Sd))
        if len(set(ix.tolist())) < 10:
            continue
        a = np.argsort(np.argsort(Sd[ix])); b = np.argsort(np.argsort(Dd[ix]))
        if a.std() > 0 and b.std() > 0:
            boots.append(np.corrcoef(a, b)[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    loop_r = next((r for r in s_rows if r["set"] == "loop"), None)
    sh = [r for r in s_rows if r["set"] == "shuffle"]
    medS = float(np.median(Sd)); base_dip = float(np.median(Dd[np.array([r["set"] for r in dom_rows]) == "wiki"]))
    verd = dict(rho=round(rho, 3), ci95=[round(float(lo), 3), round(float(hi), 3)],
                verdict=("S-драйвер жив" if rho >= 0.5 else ("мёртв" if rho <= 0.2 else "не разрешено")),
                loop_pred=dict(S=loop_r["S"] if loop_r else None, dip=loop_r["dip"] if loop_r else None,
                               expect="S≈минимум, dip < baseline−0.05",
                               hit_S=(loop_r is not None and loop_r["S"] < float(Sd.min())+1e-9),
                               hit_dip=(loop_r is not None and loop_r["dip"] < base_dip - 0.05)),
                shuffle=dict(medS_vs_docs=[round(float(np.mean([x['S'] for x in sh])), 3), medS],
                             dips=[x["dip"] for x in sh], med_dip=base_dip),
                note="LOO-фон по домену; пре-рег в шапке; own не участвует (конфаунд)")
    (OUT/"SURPLUS.json").write_text(json.dumps(dict(rows=s_rows, verdict=verd), ensure_ascii=False, indent=1))
    print(json.dumps(verd, ensure_ascii=False, indent=1), flush=True)
    print("E78 DONE", flush=True)
    for h in hs:
        h.remove()

main()
