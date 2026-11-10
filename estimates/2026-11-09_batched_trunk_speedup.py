#!/usr/bin/env python3
# ПРИКИДКА→РЕШЕНИЕ: dip-клетки E-77/E-78 гоняют доки по одному и считают lm_head, который
# для dip не нужен. Две ТОЧНЫЕ оптимизации: (1) батч с right-pad — в каузальной модели
# позиции ≤ len не видят паддинг (проверяем test_equal, не верим на слово); (2) trunk-only
# forward (m.model вместо m) — lm_head 152k×1024 не платится. Решение принимаем по таблице.
# Конвенция замера: пока e72 ест CPU, абсолютные секунды завышены, ОТНОШЕНИЯ валидны (±порядок).
import os; os.environ["OMP_NUM_THREADS"] = "8"
import time, numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(8)
from transformers import AutoModelForCausalLM, AutoTokenizer
BASE = "/data/Qwen/Qwen3.5-0.8B"
tk = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="eager").eval()
ids = [tk(("The history of the city began with a market. " * 8), add_special_tokens=False)["input_ids"][:192] for _ in range(16)]
def t_seq_full():
    t0 = time.time()
    with torch.no_grad():
        for x in ids: m(input_ids=torch.tensor([x]))
    return time.time()-t0
def t_seq_trunk():
    t0 = time.time()
    with torch.no_grad():
        for x in ids: m.model(input_ids=torch.tensor([x]))
    return time.time()-t0
def t_batch(bs):
    mx = max(map(len, ids)); P = torch.zeros(16, mx, dtype=torch.long)
    for i, x in enumerate(ids): P[i, :len(x)] = torch.tensor(x)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, 16, bs): m.model(input_ids=P[i:i+bs])
    return time.time()-t0
r = {}
for name, f in [("seq_full", t_seq_full), ("seq_trunk", t_seq_trunk)]:
    r[name] = f(); print(f"{name}: {r[name]:.1f}s")
for bs in (8, 16):
    r[f"trunk_b{bs}"] = t_batch(bs); print(f"trunk_batch{bs}: {r[f'trunk_b{bs}']:.1f}s  (ускорение ×{r['seq_full']/r[f'trunk_b{bs}']:.1f})")
# ТОЧНОСТЬ: батч-vs-seq на скрытых состояниях L23 (те же данные, real-позиции)
caps = {}
hs = [m.model.layers[23].register_forward_hook(lambda mo, i, o: caps.__setitem__('h', o.detach()))]
with torch.no_grad(): m.model(input_ids=torch.tensor([ids[0]]))
a = caps['h'][0, :len(ids[0])].clone()
mx = max(map(len, ids)); P = torch.zeros(2, mx, dtype=torch.long); P[0, :len(ids[0])] = torch.tensor(ids[0])
P[1, :len(ids[1])] = torch.tensor(ids[1])
with torch.no_grad(): m.model(input_ids=P)
b = caps['h'][0, :len(ids[0])]
d = float((a-b).abs().max()/ (a.abs().max()+1e-12))
print(f"batch==seq на реальных позициях: rel maxdiff {d:.2e} ⇒ ТОЧНО: {d < 1e-5}")
for h in hs: h.remove()
