#!/usr/bin/env python
"""E-62 LFM3: регенерация 64 траекторий exp48b со слоями {1,5,11,17,23} (выходы
decoder-слоёв 0-based; L23 = вход финальной RMSNorm = exp48b-«L23»).
Точная политика greedy exp48b: logits = (g_n·rms(h)) @ W_емб^T (tied, g=w),
хук на каждый целевой слой (output_hidden_states LFM3 не поддерживается).
Гейты: prompt_sha и gen_sha против exp48b json; L23-контроль cos с npz.
Запуск: OMP=10 nice -n 15 ../.venv/bin/python -u exp62_gen_lfm3.py"""
import hashlib, json, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def sha(xs):
    return hashlib.sha256(str(xs).encode()).hexdigest()[:12]


def main():
    os.environ.setdefault("PRISM_CALIB_PIN_WRITE", "1")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/LiquidAI/LFM3-1.7B")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--layers", default="1,5,11,17,23")
    ap.add_argument("--out", default="/data/prism_kernels/exp62/gen_layers_lfm3.npz")
    ap.add_argument("--ref-json", default="../results/exp48b_lfm3_gen_kinematics.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    torch.set_num_threads(10)
    torch.manual_seed(47)
    from transformers import AutoTokenizer, DynamicCache
    import transformers as _T
    # LFM3: load_calibration_data вызывает AutoTokenizer без trust_remote_code —
    # интерактивный prompt в nohup = deadlock. Тенируем from_pretrained с дефолтом
    # trust_remote_code=True (явно переданный аргумент сохраняет приоритет).
    for _n in ("AutoTokenizer", "AutoConfig"):
        _f = getattr(_T, _n).from_pretrained
        setattr(_T, _n, "from_pretrained",
                staticmethod(lambda p, _f=_f, **kw:
                             _f(p, **{**kw, "trust_remote_code": kw.get("trust_remote_code", True)}))) \
            if False else setattr(getattr(_T, _n), "from_pretrained",
                staticmethod(lambda p, _f=_f, **kw:
                             _f(p, **{**kw, "trust_remote_code": kw.get("trust_remote_code", True)})))
    from prism.calibration import load_calibration_data
    from exp48b_lfm3_gen_kinematics import load_lfm3   # та же загрузка
    dev = torch.device("cpu")
    m = load_lfm3(a.model, dev)
    nrm = m.model.embedding_norm
    eps = float(nrm.variance_epsilon)
    g_n = nrm.weight.detach().float()
    W = m.get_input_embeddings().weight.detach()
    Lset = sorted(int(x) for x in a.layers.split(","))
    nl = int(m.config.num_hidden_layers)
    assert max(Lset) <= nl - 1, f"слой {max(Lset)} > {nl - 1}"
    caps = {}

    def _mk(L):
        def _hook(_mod, _inp, out):
            caps[L] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().double()
        return _hook
    for L in Lset:
        m.model.layers[L].register_forward_hook(_mk(L))
    ref = {f"{d['dom']}_{d['si']}": d["gen_sha"] for d in json.load(open(a.ref_json))["docs"]}
    ref_p = {f"{d['dom']}_{d['si']}": d["prompt_sha"] for d in json.load(open(a.ref_json))["docs"]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out_npz, meta = {}, {}
    done = set()
    if os.path.exists(a.out):          # RESUME: подхватить уже сгенерированные
        old = np.load(a.out, allow_pickle=True)
        for k in old.files:
            out_npz[k] = old[k]
        if os.path.exists(a.out.replace(".npz", "_meta.json")):
            meta = json.load(open(a.out.replace(".npz", "_meta.json")))
            done = {n for n, v in meta.items() if v.get("sha_ok")}
        print(f"[resume] готово {len(done)} доков", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for dom in ([x.strip() for x in a.only.split(",")] if a.only else ("wiki", "math", "code", "chat")):
            pool = load_calibration_data(a.model, max_length=256, min_length=64,
                                         n_samples=a.n_docs, domain=dom)
            for si in range(a.n_docs):
                nm = f"{dom}_{si}"
                if nm in done:
                    continue
                p_ok = sha(pool[si]["input_ids"][0, :a.prompt_len].tolist()) == ref_p.get(nm)
                if not p_ok:
                    print(f"{nm}: ВНИМАНИЕ: prompt_sha не совпал", flush=True)
                ids = pool[si]["input_ids"][:, :a.prompt_len]
                toks = ids[0].tolist()
                uL = {L: [] for L in Lset}
                hnL = {L: [] for L in Lset}
                cache = DynamicCache(config=m.config)
                m(input_ids=ids, past_key_values=cache, use_cache=True)
                h = {L: caps[L].clone() for L in Lset}
                for step in range(a.max_new):
                    for L in Lset:
                        uL[L].append((h[L] / h[L].norm()).float().numpy())
                        hnL[L].append(float(h[L].norm()))
                    hh = h[max(Lset)]
                    xn = hh.float() * torch.rsqrt(hh.pow(2).mean() + eps).float()
                    lg = ((xn * g_n) @ W.T).float()
                    nxt = int(lg.argmax())
                    toks.append(nxt)
                    m(input_ids=torch.tensor([[nxt]]), past_key_values=cache,
                      use_cache=True)
                    h = {L: caps[L].clone() for L in Lset}
                gen = toks[a.prompt_len:]
                ok = (ref.get(nm) == sha(gen)) and p_ok
                meta[nm] = dict(dom=dom, si=si, sha_ok=bool(ok), gen=sha(gen))
                for L in Lset:
                    out_npz[f"{nm}_L{L}_u"] = np.stack(uL[L]).astype(np.float16)
                    out_npz[f"{nm}_L{L}_hn"] = np.array(hnL[L], dtype=np.float32)
                out_npz[f"{nm}_tok"] = np.array(gen, dtype=np.int32)
                print(f"{nm}: sha_ok={ok} [{time.time()-t0:.0f}s]", flush=True)
            del pool
            np.savez_compressed(a.out, **out_npz)      # per-domain checkpoint
            json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"))
            print(f"[{dom}] checkpoint: {sum(v['sha_ok'] for v in meta.values())} "
                  f"ok / {len(meta)}", flush=True)
    np.savez_compressed(a.out, **out_npz)
    json.dump(meta, open(a.out.replace(".npz", "_meta.json"), "w"))
    print(f"ГОТОВО: sha_ok {sum(v['sha_ok'] for v in meta.values())}/{len(meta)}; "
          f"{time.time()-t0:.0f} c", flush=True)


if __name__ == "__main__":
    main()
