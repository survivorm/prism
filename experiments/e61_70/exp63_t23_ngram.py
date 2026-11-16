#!/usr/bin/env python
"""E-63 T-23 (гипотеза модератора: «хеш-таблица»): какой объём генерации покрывается
чистым n-gram lookup по контексту (prompt-visible 256 + предгенерированное) — REST/
suffix-decoding стиль. Ключ = последние n токенов; предсказание = мажоритарное
продолжение из ранних вхождений. cov / acc|cov / joint по n=2..8; контроль: порядок
контекта перемешан (тот же multiset)."""
import json, os, sys, glob
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from transformers import AutoTokenizer

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
MK = "/data/Qwen/Qwen3.5-0.8B"
z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
tok = AutoTokenizer.from_pretrained(MK)
dom = {n: meta[n]["dom"] for n in meta}


def prompt_ids(dom_name):
    f = glob.glob(f"../data/calib_pins/{dom_name}___data_Qwen_Qwen3.5_0.8B__n16__min384__max512__plain.json")
    d = json.load(open(f[0]))
    return [tok(t, add_special_tokens=False)["input_ids"][:256] for t in d["texts"]]


def lookup_metrics(ctx, off, n):
    """ctx: prompt+gen ids; шаги генерации: t в [off, len-1). Возвращает cov,acc|cov."""
    mp = {}
    cov = hit = tot = 0
    for t in range(off, len(ctx) - 1):
        key = bytes(np.asarray(ctx[t - n + 1:t + 1], dtype=np.int64).tobytes())
        conts = mp.get(key)
        if conts:
            cov += 1
            vals, cts = np.unique(conts, return_counts=True)
            if int(vals[cts.argmax()]) == int(ctx[t + 1]):
                hit += 1
        tot += 1
        s = t - n + 1                      # добавляем вхождение s (после предсказания)
        if s >= 0:
            k2 = bytes(np.asarray(ctx[s:s + n], dtype=np.int64).tobytes())
            if s + n < len(ctx):
                mp.setdefault(k2, []).append(int(ctx[s + n]))
    return cov / max(tot, 1), hit / max(cov, 1)


res = {}
cache = {}
for nm in sorted(meta):
    dm = dom[nm]
    if dm not in cache:
        cache[dm] = prompt_ids(dm)
    si = int(nm.rsplit("_", 1)[-1])
    gen = list(np.asarray(z[f"{nm}_tok"], dtype=np.int64))
    pr = list(cache[dm][si])
    ctx = pr + gen
    off = len(pr)
    for n in (2, 3, 4, 6, 8):
        c, a = lookup_metrics(ctx, off, n)
        rs = np.random.default_rng(4000 + n)
        shuffled = pr + [gen[i] for i in rs.permutation(len(gen))]
        c2, a2 = lookup_metrics(shuffled, off, n)
        res.setdefault(dm, {}).setdefault(n, []).append((c, a, c2, a2))
    print(nm, "ok", flush=True)
out = {dm: {str(n): {"cov": round(float(np.mean([r[0] for r in v])), 3),
                     "acc|cov": round(float(np.mean([r[1] for r in v])), 3),
                     "joint": round(float(np.mean([r[0] * r[1] for r in v])), 3),
                     "shuf_cov": round(float(np.mean([r[2] for r in v])), 3),
                     "shuf_acc": round(float(np.mean([r[3] for r in v])), 3)}
           for n, v in sorted(dv.items())} for dm, dv in res.items()}
print(json.dumps(out, indent=1))
json.dump(out, open("../results/e63/t23_ngram.json", "w"))
