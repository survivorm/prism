#!/usr/bin/env python
"""E-63 T-1: ритм в САМИХ токенах. p(same)(k) и cos-автокорр эмбеддингов на ID-потоке
генерации: real vs blk8 vs iid. Пик у real на лаге T=2π/ω*, сохранённый blk8 и убитый iid
=> «контент-часы» (ритм в тексте); пика нет => состояние добавляет ритм сверх текста."""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import json, numpy as np
from safetensors import safe_open

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
MK = "/data/Qwen/Qwen3.5-0.8B"
KM = 24
z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
dom = {n: meta[n]["dom"] for n in meta}
names = sorted(meta)

with safe_open(f"{MK}/model.safetensors-00001-of-00001.safetensors", "pt") as f:
    key = [k for k in f.keys() if "embed_tokens" in k][0]
    E = f.get_tensor(key).float().numpy().astype(np.float32)
En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
del E


def curves(ids):
    ids = np.asarray(ids)
    ps = np.array([(ids[:-k] == ids[k:]).mean() for k in range(1, KM + 1)], float)
    U = En[ids]
    c = np.array([float((U[:-k] * U[k:]).sum(1).mean()) for k in range(1, KM + 1)])
    return ps, c


def blocks(ids, blk, rng):
    nb = len(ids) // blk
    idx = rng.permutation(nb)
    out = np.concatenate([ids[b * blk:(b + 1) * blk] for b in idx]) if nb else ids
    return np.concatenate([out, ids[nb * blk:]]) if len(ids) % blk else out


res = {}
for tag in ("real", "blk8", "iid"):
    ps_by = {d: [] for d in ("wiki", "math", "code", "chat")}
    c_by = {d: [] for d in ("wiki", "math", "code", "chat")}
    for i, nm in enumerate(names):
        ids = np.asarray(z[f"{nm}_tok"], dtype=np.int64)   # ID генерации (сопоставлены 64/64)
        if len(ids) < 48:
            continue
        if tag == "blk8":
            ids = blocks(ids, 8, np.random.default_rng(5000 + i))
        elif tag == "iid":
            ids = np.random.default_rng(6000 + i).permutation(ids)
        ps, c = curves(ids)
        ps_by[dom[nm]].append(ps)
        c_by[dom[nm]].append(c)
    res[tag] = {d: {"psame": np.median(np.array(v), 0).round(4).tolist(),
                    "cos": np.median(np.array(w), 0).round(4).tolist()}
               for d, v, w in ((d, ps_by[d], c_by[d]) for d in ps_by)}
    print(tag, "готово", flush=True)

W, M, C, H = res["real"], None, None, None
import math
print(f"{'k':>3} {'psR':>7} {'psB8':>7} {'psIID':>7} | {'cosR':>7} {'cosB8':>7} {'cosIID':>7} (медиана по всем докам)")
for k in range(KM):
    pr = np.median([res[t]["psame"][d][k] for d in res[t]["psame"] for t in [tag]][:0] or
                   [res[tag]["psame"][d][k] for tag in ("real",) for d in res[tag]["psame"]])
    pb = np.median([res["blk8"]["psame"][d][k] for d in res["blk8"]["psame"]])
    pi = np.median([res["iid"]["psame"][d][k] for d in res["iid"]["psame"]])
    cr = np.median([res["real"]["cos"][d][k] for d in res["real"]["cos"]])
    cb = np.median([res["blk8"]["cos"][d][k] for d in res["blk8"]["cos"]])
    ci = np.median([res["iid"]["cos"][d][k] for d in res["iid"]["cos"]])
    print(f"{k+1:>3} {pr:>7.4f} {pb:>7.4f} {pi:>7.4f} | {cr:>7.3f} {cb:>7.3f} {ci:>7.3f}")
print("\nпо доменам, cos-автокорр токенов (k=1..8):")
for d in ("math", "code", "chat", "wiki"):
    for tag in ("real", "blk8", "iid"):
        print(f"  {d:<5}{tag:<5}", [round(x, 3) for x in res[tag]["cos"][d][:8]])
os.makedirs("../results/e63", exist_ok=True)
json.dump(res, open("../results/e63/t1_tokrhythm.json", "w"))
