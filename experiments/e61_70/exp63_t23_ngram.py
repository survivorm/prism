#!/usr/bin/env python
"""E-63 T-23 (гипотеза модератора: «хеш-таблица»): какой объём генерации покрывается
чистым n-gram lookup по контексту (prompt-visible 256 + предгенерированное) — REST/
suffix-decoding стиль. Ключ = последние n токенов; предсказание = мажоритарное
продолжение из ранних вхождений. cov / acc|cov / joint по n=2..8; контроль: порядок
контекта перемешан (тот же multiset)."""
import json, os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, ".."))   # = experiments/, чтобы "../results" -> <repo>/results
os.makedirs(os.path.join(_HERE, "..", "..", "results", "e63"), exist_ok=True)   # каталога results/e63 в репо нет
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
MAX_SI = {dm: max(int(n.rsplit("_", 1)[-1]) for n in meta if dom[n] == dm) for dm in set(dom.values())}


def prompt_ids(dom_name):
    # Изначально тут был один литерал __n16__min384__max512; такого pin-файла в data/calib_pins
    # нет ни для одного домена (проверено 20.11). Поэтому — цепочка: целевой конфиг → единственный
    # существующий n16-набор → любой n16; выбор пишется в артефакт (PIN_USED), а не молча подменяется.
    want = f"../data/calib_pins/{dom_name}___data_Qwen_Qwen3.5_0.8B__n16__min384__max512__plain.json"
    alts = [f"../data/calib_pins/{dom_name}___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json",
            f"../data/calib_pins/{dom_name}___data_Qwen_Qwen3.5_0.8B__n12__min192__max512__plain.json"]
    alts += sorted(glob.glob(f"../data/calib_pins/{dom_name}___*.json"))[:1]
    for p in [want] + alts:
        if os.path.exists(p):
            d = json.load(open(p))
            PIN_USED[dom_name] = os.path.basename(p)
            return [tok(t, add_special_tokens=False)["input_ids"][:256] for t in d["texts"]]
    raise SystemExit(f"нет pin-файла для домена {dom_name} (нужен {os.path.basename(want)} или n16-аналог)")


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
PIN_USED = {}
PIN_CYCLE = {}
for nm in sorted(meta):
    dm = dom[nm]
    if dm not in cache:
        cache[dm] = prompt_ids(dm)
    si = int(nm.rsplit("_", 1)[-1])
    gen = list(np.asarray(z[f"{nm}_tok"], dtype=np.int64))
    n_pins = len(cache[dm])
    PIN_CYCLE[dm] = n_pins <= MAX_SI[dm]          # pin-текстов меньше, чем документов → индекс по кругу
    pr = list(cache[dm][si % n_pins])
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
print(json.dumps({"pins_used": PIN_USED, "pin_cycle": PIN_CYCLE, "res": out}, indent=1))
json.dump({"pins_used": PIN_USED, "pin_cycle": PIN_CYCLE, "res": out}, open("../results/e63/t23_ngram.json", "w"), default=float)
