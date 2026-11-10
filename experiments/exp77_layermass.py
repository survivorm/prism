#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-77 LAYERMASS: порог dip по НАКОПЛЕННОЙ МАССЕ обученных слоёв (фальсификатор
# параметризации Т-В(3), ТЗ docs/dynamic/ТЗ_раунд8.md со своим аудитом).
# Случайная модель (fresh-init, per-tensor std-match) + замена dim-2 тензоров k слоёв
# на обученные. Ось X — измеренная масса, не k. Режимы: prefix / suffix / random×3сида.
# Гейты (ТЗ п.3): k=0 в полосе случайного (|dip|<0.02) И k=24 == baseline (построением,
# фиксируется как тавтология-конструкция); несогласованность кривой массы с α-кривой
# E-71A записывается как отдельный результат.
# Патч-правило (аудит п.2): рандомизуются ТОЛЬКО dim==2 тензоры внутри model.layers.*
# (Linear/G-матрицы); layernorm/rmsnorm/эмбеддинги/norm/lm_head — trained везде.
import ast, hashlib, json, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
from prism.calibration import load_calibration_data
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"results"/"e77"; OUT.mkdir(parents=True, exist_ok=True)
RAW = OUT/"LAYERMASS_raw.txt"
BASE = "/data/Qwen/Qwen3.5-0.8B"
LAYERS = [11, 23]
src = open(ROOT/"experiments"/"kaggle_e64gpu"/"e64_gpu.py").read()
fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
       if isinstance(n, ast.FunctionDef)}
ns = {"np": np}
for w in ["vpath_units", "dcurve", "curves_series"]:
    exec(fns[w], ns)
curves_series = ns["curves_series"]

def done_keys():
    if not RAW.exists():
        return set()
    ks = set()
    for l in open(RAW):
        try:
            r = json.loads(l); ks.add((r["mode"], r["k"], r["seed"]))
        except Exception:
            pass
    return ks

def main():
    pool = [d for d in load_calibration_data("Qwen/Qwen3.5-4B", max_length=256, min_length=64,
                                             n_samples=16, domain="wiki")]
    tk0 = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    POOL_FP = hashlib.sha256(json.dumps([d["input_ids"][0].tolist() for d in pool]).encode()).hexdigest()[:16]
    docs = [torch.tensor(tk0.decode(d["input_ids"][0].tolist()[:48]).strip()[:192], dtype=torch.long)
            if False else None for d in pool]
    # тексты пула как в E-71: декод 48 qwen-id, своя токенизация 0.8B, срез 192
    texts = [tk0.decode(d["input_ids"][0].tolist()[:48]).strip() for d in pool]
    tk = AutoTokenizer.from_pretrained(BASE)
    xs = []
    for t in texts:
        ids = tk(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0][:192]
        xs.append(ids)
    m_tr = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32,
                                                attn_implementation="eager").eval()
    cfg = m_tr.config
    nL = cfg.num_hidden_layers
    sd_tr = m_tr.state_dict()
    layer2 = {n for n, p in sd_tr.items() if p.dim() == 2 and n.startswith("model.layers.")}
    layer_of = {n: int(n.split(".")[2]) for n in layer2}
    # полная масса
    mass_tot = 0.0
    # random-словарь: все dim-2 слойные тензоры — fresh-init std-matched
    torch.manual_seed(0)
    rnd_full = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                attn_implementation="eager").to(torch.float32).eval()
    sd_r = rnd_full.state_dict()
    assert set(sd_tr.keys()) == set(sd_r.keys()), "разные state_dict — сборка невалидна"
    RND = {}
    mass_by_layer = {i: 0.0 for i in range(nL)}
    for n in layer2:
        a, b = sd_tr[n], sd_r[n]
        b = b * (float(a.std()) / (float(b.std()) + 1e-12))
        RND[n] = b
        mass_by_layer[layer_of[n]] += float((a - b).pow(2).sum().sqrt())
        sd_r[n] = b
    mass_tot = sum(mass_by_layer.values())
    del rnd_full
    import gc; gc.collect()
    FULL_R = dict(sd_tr)                       # старт: всё trained
    FULL_R.update(RND)                         # + все слойные linear — random

    caps = {}
    hs = [m_tr.model.layers[l].register_forward_hook(lambda mod, i, o, l=l: caps.__setitem__(l, o.detach()))
          for l in LAYERS]

    def dip_cell():
        d11, d23 = [], []
        for ids in xs:
            if ids.shape[0] < 40:
                continue
            t = ids.unsqueeze(0)
            caps.clear()
            with torch.no_grad():
                m_tr(input_ids=t, use_cache=False)
            for L, acc in ((11, d11), (23, d23)):
                if L < nL:
                    c = curves_series(caps[L][0, :ids.shape[0]].to(torch.float32).numpy())
                    acc.append(float(c["dip"]))
        return (float(np.median(d11)) if d11 else None, float(np.median(d23)) if d23 else None, len(d11))

    def set_cell(S):
        """S — множество trained-слоёв; собираем state и грузим in-place"""
        cur = {}
        for n in layer2:
            i = layer_of[n]
            src_t = sd_tr[n] if i in S else RND[n]
            cur[n] = src_t
        m_tr.load_state_dict(cur, strict=False)

    def mass_of(S):
        return round(sum(mass_by_layer[i] for i in S)/mass_tot, 4)

    def run(mode, k, seed):
        if mode == "prefix":
            S = set(range(k))
        elif mode == "suffix":
            S = set(range(nL-k, nL))
        else:
            S = set(np.random.default_rng(1000+seed).choice(nL, size=k, replace=False).tolist())
        set_cell(S)
        t0 = time.time()
        a, b, n = dip_cell()
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(mode=mode, k=k, seed=seed, mass=mass_of(S),
                                    n_trained=len(S), dip_L11=a, dip_L23=b, ndocs=n,
                                    pool_fp=POOL_FP, secs=round(time.time()-t0, 1))) + "\n")
        print(f"[{mode} k={k} s{seed}] mass {mass_of(S):.3f} dip11 {a:+.3f} dip23 {b:+.3f}", flush=True)

    have = done_keys()
    # baseline (всё trained) — первый гейт
    if ("base", 24, 0) not in have:
        m_tr.load_state_dict(sd_tr, strict=True)
        a, b, n = dip_cell()
        with open(RAW, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(mode="base", k=24, seed=0, mass=1.0, dip_L11=a, dip_L23=b,
                                    ndocs=n, pool_fp=POOL_FP)) + "\n")
        print(f"[baseline] dip11 {a:+.3f} dip23 {b:+.3f} (ожидание ≈ −0.174)", flush=True)
        have = done_keys()
    for mode, seeds in (("prefix", [0]), ("suffix", [0]), ("random", [0, 1, 2])):
        for k in range(0, nL+1):
            for s in seeds:
                if (mode, k, s) in have:
                    continue
                run(mode, k, s)
    # гейт k=0
    rows = [json.loads(l) for l in open(RAW)]
    k0 = [r["dip_L23"] for r in rows if r["k"] == 0 and r["mode"] != "base"]
    base = [r["dip_L23"] for r in rows if r["mode"] == "base"]
    if k0 and base:
        ok0 = all(abs(x) < 0.02 for x in k0)
        print(f"[гейт] k=0 dip23 {k0} (<0.02? {ok0}) | base {base}", flush=True)
        print("E77 DONE", flush=True)
    for h in hs:
        h.remove()

main()
