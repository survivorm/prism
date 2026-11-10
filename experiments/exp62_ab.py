#!/usr/bin/env python
"""E-62 A/B: свободная greedy-траектория против teacher-forced на ОДНОМ свежем
запиненном тексте. Если факты батареи (F1 14лаг, osc/exp состав, dip) совпадают на
L23 — слой-эксперимент считается teacher-forced проходом (1 forward вместо 384 шагов).
Пин: n=16/домен, длина 384..512 токенов. Граница (ревью r3 #2): пул min384/max512 НЕ
совпадает с exp47b (min64/max256) НАМЕРЕННО — TF-режиму нужно 255 позиций, а документы
exp47b бывают 64–255 токенов; A/B отвечает «TF ≈ G на одном тексте?», не «те же тексты».
Фаза при NAB=4 не тестируется (strict<3/домен) — в выводе помечается явно."""
import json, os, sys, time
os.environ.setdefault("PRISM_CALIB_PIN_WRITE", "1")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp62_batt_lib import run  # noqa: E402
from exp01_coherence import load_calibration_data  # noqa: E402

MODEL = "/data/Qwen/Qwen3.5-0.8B"
DOMS = ["wiki", "math", "code", "chat"]
NAB, PL, TAB = 4, 64, 192          # доков на домен в A/B; промпт; позиций траектории
LAYERS = [1, 5, 11, 17, 23]       # семантические индексы слоёв; доступ = hidden_states[L+1]; 23 = exp47b-L23
torch.set_num_threads(10)
torch.manual_seed(47)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                         attn_implementation="eager")
m.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
Lset = LAYERS
t0 = time.time()

texts = {}
for dom in DOMS:
    pool = load_calibration_data(MODEL, max_length=512, min_length=384,
                                 n_samples=16, domain=dom)
    texts[dom] = [p["input_ids"][0].tolist() for p in pool[:NAB]]
    assert all(len(t) >= PL + TAB + 1 for t in texts[dom]), "TF-срез не умещается"
    del pool
print(f"[pool] готов, {time.time()-t0:.0f}s", flush=True)

trajG, domG, trajTF, domTF = {}, {}, {}, {}
with torch.no_grad():
    for dom in DOMS:
        for si, ids in enumerate(texts[dom]):
            nm = f"{dom}_{si}"
            # --- режим G: greedy с кэша, u на позициях [prompt_end, gen...]
            pr = torch.tensor([ids[:PL]])
            cache = DynamicCache(config=m.config)
            o = m(input_ids=pr, past_key_values=cache, use_cache=True,
                  output_hidden_states=True)
            h = {L: o.hidden_states[L + 1][0, -1].double() for L in Lset}
            lg = o.logits[0, -1]
            uL = {L: [] for L in Lset}
            hn = {L: [] for L in Lset}
            for step in range(TAB):
                nxt = int(torch.argmax(lg))
                for L in Lset:
                    uL[L].append((h[L] / h[L].norm()).float().numpy())
                    hn[L].append(float(h[L].norm()))
                o = m(input_ids=torch.tensor([[nxt]]), past_key_values=cache,
                      use_cache=True, output_hidden_states=True)
                h = {L: o.hidden_states[L + 1][0, -1].double() for L in Lset}
                lg = o.logits[0, -1]
            for L in Lset:
                trajG[f"{nm}_L{L}"] = (np.stack(uL[L]), np.array(hn[L]))
                domG[f"{nm}_L{L}"] = dom
            # --- режим TF: teacher forcing на полном тексте; ГЕЙТ-выровненный срез
            # PL-1:PL+TAB-1 (G сохраняет позиции 63..63+TAB-1, ревью баг #3)
            T = min(len(ids), PL + TAB + 8)
            seq = torch.tensor([ids[:T]])
            o = m(input_ids=seq, output_hidden_states=True)
            for L in Lset:
                hs = o.hidden_states[L + 1][0, PL - 1: PL + TAB - 1].double().cpu().numpy()
                trajTF[f"{nm}_L{L}"] = (hs / np.linalg.norm(hs, axis=1, keepdims=True),
                                        np.linalg.norm(hs, axis=1))
                domTF[f"{nm}_L{L}"] = dom
            print(f"{nm}: G+TF готовы [{time.time()-t0:.0f}s]", flush=True)

np.savez_compressed("../results/e62/ab_traj.npz",
                    **{f"G_{k}": v[0] for k, v in trajG.items()},
                    **{f"G_{k}_hn": v[1] for k, v in trajG.items()},
                    **{f"TF_{k}": v[0] for k, v in trajTF.items()},
                    **{f"TF_{k}_hn": v[1] for k, v in trajTF.items()})
for tag, tr, dm in (("G", trajG, domG), ("TF", trajTF, domTF)):
    for L in Lset:
        keys = [k for k in tr if k.endswith(f"_L{L}")]
        r = run({k: tr[k] for k in keys}, {k: dm[k] for k in keys}, verbose=False)
        print(f"режим={tag} L{L}: F1={r['F1']} osc={r['n_osc']}/{r['n']} "
              f"clip={r['clip']} strict={r['n_strict']} lag1={r['dip1']}", flush=True)
        if not any(v is not None for v in r["phase"].values()):
            print(f"  (phase: not tested — strict<3/домен при NAB={NAB})", flush=True)
