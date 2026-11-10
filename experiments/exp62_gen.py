#!/usr/bin/env python
"""E-62 gen: greedy-проход по тем же промптам exp47b, но сохраняем unit-hidden
L∈{2,6,12,18,23}+hnorm на каждой позиции траектории + ТОКЕНЫ. Гейты: (1) sha
сгенерированных токенов == json gen_sha (иначе траектория не та — док в quarantine);
(2) L23 u vs существующий npz: медианный кос-угол > 0.999. Экономия: hidden_states
за один проход дают все слои; forward только генерация (0.8B CPU, ~30–50 с/док)."""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

def sha(xs):
    # Байт-в-байт exp47b._sha: sha256(repr-строки списка) (ревью E-62, баг #1)
    return hashlib.sha256(str(xs).encode()).hexdigest()[:12]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--layers", default="1,5,11,17,23")  # семантические (0-based) индексы слоёв
    ap.add_argument("--only", default="", help="только эти домены, напр. wiki")
    ap.add_argument("--n-start", type=int, default=0)
    ap.add_argument("--out", default="../results/e62/gen_layers_qwen.npz")
    ap.add_argument("--ref-json", default="../results/exp47b_gen_kinematics.json")
    a = ap.parse_args()
    torch.set_num_threads(10)
    torch.manual_seed(47)          # как exp47b (RNG_SEED=47)
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from exp01_coherence import load_calibration_data  # та же выборка, что exp47b
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32,
                                             attn_implementation="eager")
    m.eval()
    tok = AutoTokenizer.from_pretrained(a.model)
    n_layers = int(m.config.num_hidden_layers)
    Lset = sorted({int(x) for x in a.layers.split(",")})
    assert max(Lset) <= n_layers - 1, f"слой {max(Lset)} больше n_layers-1={n_layers-1}"
    # ВАЖНО (ревью E-62, баг #2): hidden_states имеет длину n_layers+1; выход слоя L —
    # hidden_states[L+1]; exp47b «L23» = hidden_states[n_layers].
    for L in Lset:
        assert L + 1 <= n_layers
    ref = {f"{d['dom']}_{d['si']}": d["gen_sha"] for d in json.load(open(a.ref_json))["docs"]}
    ref_p = {f"{d['dom']}_{d['si']}": d["prompt_sha"] for d in json.load(open(a.ref_json))["docs"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out_npz, meta = {}, {}
    t0 = time.time()
    for dom in ([x.strip() for x in a.only.split(",")] if a.only else ("wiki", "math", "code", "chat")):
        pool = load_calibration_data(a.model, max_length=256, min_length=64,
                                     n_samples=a.n_docs, domain=dom)
        for si in range(a.n_start, a.n_docs):
            nm = f"{dom}_{si}"
            ids = pool[si]["input_ids"][:, :a.prompt_len]
            p_ok = sha(pool[si]["input_ids"][0, :a.prompt_len].tolist()) == ref_p.get(f"{dom}_{si}")
            if not p_ok:
                print(f"{dom}_{si}: ВНИМАНИЕ: prompt_sha не совпал", flush=True)
            toks = ids[0].tolist()
            uL = {L: [] for L in Lset}
            hns = {L: [] for L in Lset}
            with torch.no_grad():
                cache = DynamicCache(config=m.config)
                o = m(input_ids=ids, past_key_values=cache, use_cache=True,
                      output_hidden_states=True)
            h = {L: o.hidden_states[L + 1][0, -1].double() for L in Lset}
            logits = o.logits[0, -1]
            for step in range(a.max_new):
                nxt = int(torch.argmax(logits))
                for L in Lset:
                    uL[L].append((h[L] / h[L].norm()).float().numpy())
                    hns[L].append(float(h[L].norm()))
                toks.append(nxt)
                with torch.no_grad():
                    o = m(input_ids=torch.tensor([[nxt]]), past_key_values=cache,
                          use_cache=True, output_hidden_states=True)
                h = {L: o.hidden_states[L + 1][0, -1].double() for L in Lset}
                logits = o.logits[0, -1]
            gen = toks[a.prompt_len:]
            gsha = sha(gen)
            ok = (ref.get(nm) == gsha) and p_ok
            meta[nm] = dict(dom=dom, si=si, sha_ok=bool(ok), gen=gsha)
            for L in Lset:
                out_npz[f"{nm}_L{L}_u"] = np.stack(uL[L]).astype(np.float16)
                out_npz[f"{nm}_L{L}_hn"] = np.array(hns[L], dtype=np.float32)
            out_npz[f"{nm}_tok"] = np.array(gen, dtype=np.int32)
            print(f"{nm}: sha_ok={ok} [{time.time()-t0:.0f}s]", flush=True)
        del pool
    np.savez_compressed(a.out, **out_npz)
    json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"))
    n_ok = sum(v["sha_ok"] for v in meta.values())
    print(f"ГОТОВО: sha_ok {n_ok}/{len(meta)}; {time.time()-t0:.0f} c", flush=True)

if __name__ == "__main__":
    main()
