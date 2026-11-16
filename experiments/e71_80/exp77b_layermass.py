#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-77b LAYERMASS v2 — фиксы аудита E-77 (H1,H2,H3,H5,H8), пре-рег ДО прогона:
#  H1-фикс: L23 = ЗАКРЕПЛЁННЫЙ trained readout во всех клетках; случайность живёт
#     в слоях 0..22. Все клетки измеряют dip одним и тем же обученным слоем.
#  H5: интерпретация унифицирована — «сколько обученной структуры снизу доходит
#     до фиксированного readout»; вход-грязь suffix остаётся, но направление меры
#     одинаково у всех клеток.
#  H2-фикс: масса = Σ Фробениус по ТЕНЗОРАМ dim>=2, слои 0..22 (A_log/dt_bias
#     патчатся, но в массу не входят — другие единицы).
#  H3-фикс: в RAW пишутся ПО-ДОКОВЫЕ dip; CI = bootstrap(300) по документам.
#  H8-фикс: k=0 публикуется точным числом (ожидание <0.02, репорт маржи).
#  ПРЕ-РЕГ лестницы: «после фикса readout лестница prefix остаётся не-монотонной
#     (max |Δdip| между соседними k > 0.04) ИЛИ хаос объявляется артефактом места
#     меры». Суффикс-предсказание v1 («поздние не дают») понижено: ждём рост
#     dip с k и у suffix — если suffix(k=16..22) остаётся ~0 при trained-читалке,
#     «поздние слои не несут dip» вернётся УСИЛЕННЫМ.
#  Гейты: base=все trained (тавтология-самопроверка, должен совпасть с v1 −0.150);
#     prefix k=22+readout vs base — различие = вклад среднего L22... стоп: k=23==base.
#  Пул — тот же, что в v1 (4B-декод→0.8B, 16 доков) для сопоставимости; pair с
#     E-71A остаётся открытым пунктом (H7).
import ast, hashlib, json, time, os
os.environ.setdefault("OMP_NUM_THREADS","4"); os.environ.setdefault("OPENBLAS_NUM_THREADS","4"); os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"results"/"e77"; OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT/"LAYERMASS2_raw.txt"          # v2 — отдельный файл, v1 не трогаем
BASE = "/data/Qwen/Qwen3.5-0.8B"
READ = int(os.environ.get("E77B_READOUT","23"))     # закреплённый readout
LAYERS = [READ]                                       # мера — только он
src = open(ROOT/"experiments"/"kaggle_e64gpu"/"e64_gpu.py").read()
fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
       if isinstance(n, ast.FunctionDef)}
ns = {"np": np}
for w in ["vpath_units","dcurve","curves_series"]:
    exec(fns[w], ns)
curves_series = ns["curves_series"]
rngG = np.random.default_rng(3)

def main():
    pool = [d for d in load_calibration_data("Qwen/Qwen3.5-4B", max_length=256, min_length=64,
                                             n_samples=16, domain="wiki")]
    tk0 = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    POOL_FP = hashlib.sha256(json.dumps([d["input_ids"][0].tolist() for d in pool]).encode()).hexdigest()[:16]
    texts = [tk0.decode(d["input_ids"][0].tolist()[:48]).strip() for d in pool]
    tk = AutoTokenizer.from_pretrained(BASE)
    xs = [tk(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0][:192] for t in texts]
    m_tr = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32,
                                                attn_implementation="eager").eval()
    cfg = m_tr.config; nL = cfg.num_hidden_layers
    sd_tr = {k: v.detach().clone() for k, v in m_tr.state_dict().items()}
    layer_of = lambda n: int(n.split(".")[2])
    layer_patch = {n for n in sd_tr if ".layers." in n
                   and not n.endswith("layernorm.weight") and "attn.norm.weight" not in n}
    assert layer_of[max(layer_patch, key=layer_of) or 0] < nL
    patch_lo = {n for n in layer_patch if layer_of(n) != READ}     # READ закреплен
    print(f"[v2] readout L{READ} закреплён; патчаемых слоёв: {len({layer_of(n) for n in patch_lo})}", flush=True)
    torch.manual_seed(0)
    rnd_m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                             attn_implementation="eager").to(torch.float32).eval()
    sd_r = rnd_m.state_dict()
    assert set(sd_tr) == set(sd_r)
    RND = {}
    mass_by_layer = {i: 0.0 for i in range(nL)}
    for n in patch_lo:
        a, b = sd_tr[n], sd_r[n]
        if b.dim() >= 2:
            b = b * (float(a.std())/(float(b.std())+1e-12))
            mass_by_layer[layer_of(n)] += float((a-b).pow(2).sum().sqrt())   # H2: только dim>=2
        RND[n] = b.clone()
    del rnd_m
    import gc; gc.collect()
    mass_tot = sum(mass_by_layer.values()); assert mass_tot > 0
    caps = {}
    hs = [m_tr.model.layers[l].register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l, o.detach()))
          for l in LAYERS]
    def dip_docs():
        out = []
        for ids in xs:
            if ids.shape[0] < 40: continue
            t = ids.unsqueeze(0); caps.clear()
            with torch.no_grad():
                m_tr.model(input_ids=t, use_cache=False)
            c = curves_series(caps[READ][0, :ids.shape[0]].to(torch.float32).numpy())
            out.append(round(float(c["dip"]), 5))
        return out
    def set_cell(S):
        S = set(S) | {READ}
        cur = {n: (sd_tr[n] if layer_of(n) in S else RND[n]) for n in patch_lo}
        m_tr.load_state_dict(cur, strict=False)
    def boot_ci(docs):
        d = np.array(docs)
        bs = [float(rngG.choice(d, len(d), replace=True).mean()) for _ in range(300)]
        return [round(float(np.percentile(bs,q)),4) for q in (2.5,97.5)]
    def run(mode,k,seed):
        if mode=="prefix": S=set(range(k))
        elif mode=="suffix": S=set(range(nL-1-k, nL-1))     # окно в 0..22
        else: S=set(np.random.default_rng(1000+seed).choice(nL-1,size=k,replace=False).tolist())
        set_cell(S); t0=time.time()
        docs = dip_docs()
        med = float(np.median(docs)); trained = sorted(set(S)|{READ})
        row = dict(v=2, mode=mode,k=k,seed=seed, mass=round(sum(mass_by_layer[i] for i in S)/mass_tot,4),
                   n_trained=len(trained), readout=READ, dip_med=round(med,4), ci95=boot_ci(docs),
                   dip_docs=docs, pool_fp=POOL_FP, secs=round(time.time()-t0,1))
        with open(RAW,"a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
        print(f"[{mode} k={k} s{seed}] mass {row['mass']:.3f} dip {med:+.4f} CI {row['ci95']}", flush=True)
    have=set()
    if RAW.exists():
        for l in open(RAW):
            r=json.loads(l); have.add((r["mode"],r["k"],r["seed"]))
    if ("base",nL-1,0) not in have:
        m_tr.load_state_dict(sd_tr, strict=True)
        docs=dip_docs()
        row=dict(v=2,mode="base",k=nL-1,seed=0,mass=1.0,n_trained=nL,readout=READ,
                 dip_med=round(float(np.median(docs)),4),ci95=boot_ci(docs),dip_docs=docs,pool_fp=POOL_FP)
        with open(RAW,"a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
        print(f"[base] dip {np.median(docs):+.4f} (v1-самопроверка ≈ −0.150)", flush=True)
        have.add(("base",nL-1,0))
    kg = os.environ.get("E77B_KGRID","prefix:0,2,4,6,8,10,12,13,14,15,16,17,18,20,22|suffix:4,8,12,16,20|random:8,12,16,20")
    plan={}
    for part in kg.split("|"):
        mm,ks=part.split(":"); plan[mm]=[int(x) for x in ks.split(",")]
    for mode in ("prefix","suffix","random"):
        seeds=[0,1,2] if mode=="random" else [0]
        for k in plan.get(mode,[]):
            for s in seeds:
                if (mode,k,s) in have: continue
                run(mode,k,s)
    rows=[json.loads(l) for l in open(RAW)]
    k0=[r["dip_med"] for r in rows if r["mode"]=="prefix" and r["k"]==0]
    if k0: print(f"[гейт k=0 точный] {k0} (ожидание |dip|<0.02; H8 — публикуем маржу)", flush=True)
    pr=sorted([r for r in rows if r["mode"]=="prefix"],key=lambda r:r["k"])
    jumps=[abs(pr[i+1]["dip_med"]-pr[i]["dip_med"]) for i in range(len(pr)-1)]
    print(f"[пре-рег лестницы] prefix соседние Δ: max {max(jumps):.4f} — не-монотонность {'СОХРАНИЛАСЬ (>0.04)' if max(jumps)>0.04 else 'СНЯТА — хаос был артефактом места меры'}", flush=True)
    print("E77b DONE", flush=True)
    for h in hs: h.remove()

main()
