#!/usr/bin/env python
"""E-48b (P1-перенос): кинематика листа LFM3-1.7B при СВОБОДНОЙ ГЕНЕРАЦИИ.

Протокол — docs/leaf_dynamics.md §9 (E-48) + §5.2 (P1): те же метрики exp47_common,
greedy max_new шагов, loop-detect, тривиальные конкуренты. Отличия от exp47b (Qwen):
  * h — вход финальной нормы: hook на последний decoder-слой (у LFM3 output_hidden_states
    не гарантирует порядок tuple — не полагаемся); g = w (без 1+w); head tied;
  * U128 — СВЕЖИЙ частотный базис LFM3 на своём пуле (Qwen-константа бессмысленна:
    другой токенайзер); один базис на все домены;
  * динамический conv-кэш требует шимов transformers 5.15 (см. exp48a docstring) —
    проверяем на месте: prefill+2 decode-шага, route <1e-3, иначе LOUD crash;
  * генерация СВОИМИ шагами (argmax + cache feed), как в exp47b.

Дескриптивные вопросы (чисел ещё нет; вердикты не публикуем — сверяем с Qwen-числами §8):
  loops? a/s/p уровня? in-loop динамика? тривиальные конкуренты?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp47_common as C  # noqa: E402

RNG_SEED = 47


def load_lfm3(path, device):
    """Шимы совместимости transformers 5.15.1 x remote-code lfm3 (детали — exp48a)."""
    import json as _json
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    from transformers.cache_utils import LinearAttentionLayer  # type: ignore
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with open(os.path.join(path, "config.json")) as fh:
        cfg.layer_types = list(_json.load(fh)["layer_types"])
    _orig = LinearAttentionLayer.update_conv_state

    def _ucs(self, cs, state_idx=0, **kw):
        return _orig(self, cs.squeeze(1) if cs.ndim == 4 else cs, state_idx, **kw)

    LinearAttentionLayer.update_conv_state = _ucs
    m = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=torch.float32,
                                             trust_remote_code=True)
    return m.eval().to(device)


def _sha(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def _step(V_prev, u_prev, u_cur, A, eps, h_cur):
    Vb, _ = C.V16_from_J(C.J_analytic(A, h_cur, eps)[0])
    if V_prev is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), Vb
    s, p, _P, _Th = C.frame_kinematics(V_prev, Vb)
    a = float(torch.acos(((u_prev * u_cur).sum()).clamp(max=1 - 1e-12)))
    du = u_cur - (u_prev @ u_cur) * u_prev
    ndu = float(du.norm())
    g = float("nan")
    if ndu > 1e-9:
        q = torch.linalg.qr(torch.cat([V_prev, Vb], dim=1))[0]
        g = float((q.T @ (du / ndu)).norm())
    return a, s, p, g, Vb


def main(model="/data/LiquidAI/LFM3-1.7B", n_docs=8, max_new=384, prompt_len=64,
         device="cpu", results_dir="results"):
    torch.set_num_threads(4)
    torch.manual_seed(RNG_SEED)
    dev = torch.device(device)
    from transformers import AutoTokenizer, DynamicCache  # type: ignore
    from prism.calibration import build_global_token_set, load_calibration_data  # noqa: E402

    t0 = time.time()
    m = load_lfm3(model, dev)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    nrm = m.model.embedding_norm
    eps = float(nrm.variance_epsilon)
    g_n = nrm.weight.detach().float()                    # LFM3: g = w, НЕ 1+w
    W = m.get_input_embeddings().weight.detach()         # head tied
    tied = m.lm_head.weight.data_ptr() == W.data_ptr()

    # --- пул + свежий базис (один на все домены) ---
    pools = {}
    for dom in ("wiki", "math", "code", "chat"):
        pools[dom] = load_calibration_data(model, max_length=256, min_length=64,
                                           n_samples=n_docs, domain=dom)
        assert len(pools[dom]) >= n_docs, f"{dom}: пул короче {n_docs}"
    U128 = build_global_token_set(m, [d for v in pools.values() for d in v], K=128)
    U = torch.tensor(U128, device=dev)
    A = C.head_M(W[U].cpu().float(), g_n.cpu()).to(dev)  # [128,D] fp64

    # --- хук + самопроверки на ДЕКОДЕ (conv-кэш — рискмест) ---
    cap = {}

    def _hook(_mod, _inp, out):
        cap["x"] = (out[0] if isinstance(out, tuple) else out)[0, -1].detach().double()

    m.model.layers[-1].register_forward_hook(_hook)

    probe = torch.randint(1000, 100000, (1, 8), device=dev)
    with torch.no_grad():
        cache = DynamicCache(config=m.config)
        m(input_ids=probe, past_key_values=cache, use_cache=True)
        nxt = probe[:, -1:]
        for _ in range(2):
            out = m(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[0, -1].argmax().view(1, 1)
        rt = float((cap["x"].float() * torch.rsqrt(cap["x"].pow(2).mean() + eps) @ A.T
                    - out.logits[0, -1, U].cpu().double()).abs().max())
    print(f"[decode-smoke] route max|J-route - logits_U| = {rt:.2e}; tied={tied}", flush=True)
    assert rt < 1e-2, f"маршрут на декоде не сошёлся: {rt}"

    docs_out, feats_npz = [], {}
    for dom in ("wiki", "math", "code", "chat"):
        pool = pools[dom]
        for si in range(n_docs):
            ids = pool[si]["input_ids"][:, :prompt_len].to(dev)
            prompt_sha = _sha(pool[si]["input_ids"][0, :prompt_len].tolist())
            with torch.no_grad():
                cache = DynamicCache(config=m.config)
                m(input_ids=ids, past_key_values=cache, use_cache=True)
            h = cap["x"].to(dev)                          # [D] double, pre-final-norm
            feats = {"a": [], "s": [], "p": [], "g": [], "ent": [], "top1": [],
                     "eq_prev": [], "hnorm": []}
            u = h / h.norm()
            V_prev = None
            tokens = ids[0].tolist()
            V_stack, u_stack = [], []
            for step in range(max_new):
                z = (h * torch.rsqrt(h.pow(2).mean() + eps)) @ A.T     # [128] fp64 (U-подпространство)
                pfull = torch.softmax(z.float(), dim=-1)               # на U — конкурирующие фичи
                feats["ent"].append(float(-(pfull.clamp_min(1e-12) *
                                            pfull.clamp_min(1e-12).log()).sum()))
                feats["top1"].append(float(pfull.max()))
                feats["hnorm"].append(float(h.norm()))
                # полный argmax по всем 128k логитам (head = embed.weight, tied);
                # срез по U годится только для метрик на U-массе, но не для argmax
                xn = h.float() * torch.rsqrt(h.pow(2).mean() + eps).float()
                nxt = int((xn.to(dev) @ W.to(dev).T).argmax())
                feats["eq_prev"].append(int(nxt == tokens[-1]))
                u_stack.append(u.float().cpu().numpy())
                tokens.append(nxt)
                with torch.no_grad():
                    m(input_ids=torch.tensor([[nxt]], device=dev),
                      past_key_values=cache, use_cache=True)
                h_new = cap["x"].to(dev)
                u_new = h_new / h_new.norm()
                a, s, pl, gg, V_cur = _step(V_prev, u, u_new, A, eps, h_new)
                feats["a"].append(a); feats["s"].append(s)
                feats["p"].append(pl); feats["g"].append(gg)
                V_prev, u = V_cur, u_new
                V_stack.append(V_cur.float().cpu().numpy())
                h = h_new
            gen_tokens = tokens[prompt_len:]
            loops = C.find_loops(gen_tokens)
            aa = np.array(feats["a"], dtype=float)
            ss = np.array(feats["s"], dtype=float)
            pp = np.array(feats["p"], dtype=float)
            in_mask = np.zeros(max_new, dtype=bool)
            for onset, _j, e, p_, k in loops:
                in_mask[min(onset, max_new - 1): min(e, max_new)] = True
            acf = [C.autocorr_peak(torch.from_numpy(ss[min(o):min(e, len(ss))] -
                  np.nanmean(ss)), p_) for o, _j, e, p_, k in loops]
            docs_out.append({
                "dom": dom, "si": si, "prompt_sha": prompt_sha,
                "n_loops": len(loops),
                "loops": [[o, j, e, p_, k] for o, j, e, p_, k in loops],
                "a_med": float(np.nanmedian(aa)), "s_med": float(np.nanmedian(ss)),
                "p_med": float(np.nanmedian(pp)), "g_med": float(np.nanmedian(feats["g"])),
                "excess_a": C.band_power(torch.from_numpy(np.nan_to_num(aa))),
                "excess_s": C.band_power(torch.from_numpy(np.nan_to_num(ss))),
                "in_loop": {"a": float(np.nanmedian(aa[in_mask])) if in_mask.any() else None,
                            "s": float(np.nanmedian(ss[in_mask])) if in_mask.any() else None,
                            "p": float(np.nanmedian(pp[in_mask])) if in_mask.any() else None,
                            "acf_peak_by_loop": [None if x != x else float(x) for x in acf]},
                "trivial": {"ent_med": float(np.median(feats["ent"])),
                            "top1_med": float(np.median(feats["top1"])),
                            "eq_prev_rate": float(np.mean(feats["eq_prev"]))},
                "gen_sha": _sha(gen_tokens),
            })
            feats_npz[f"{dom}_{si}"] = {k: np.array(v, dtype=np.float32)
                                        for k, v in feats.items()}
            feats_npz[f"{dom}_{si}_V"] = np.stack(V_stack).astype(np.float16)
            feats_npz[f"{dom}_{si}_u"] = np.stack(u_stack).astype(np.float16)
            d0 = docs_out[-1]
            print(f"{dom} s{si}: loops={d0['n_loops']} a={d0['a_med']:.3f} s={d0['s_med']:.3f} "
                  f"p={d0['p_med']:.3f} g={d0['g_med']:.3f} ent={d0['trivial']['ent_med']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    os.makedirs(results_dir, exist_ok=True)
    out = {"params": {"protocol": "docs/leaf_dynamics.md §9 + §5.2", "model": str(model),
                      "n_docs": n_docs, "max_new": max_new, "prompt_len": prompt_len,
                      "decoding": "greedy", "seed": RNG_SEED,
                      "U128": "fresh freq basis on LFM3 pool", "U128_head": U128[:8],
                      "route_decode": rt, "tied": bool(tied)},
           "docs": docs_out}
    with open(os.path.join(results_dir, "exp48b_lfm3_gen_kinematics.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(os.path.join(results_dir, "exp48b_lfm3_gen_kinematics.npz"), **feats_npz)
    print(f"saved. total {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/LiquidAI/LFM3-1.7B")
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--results-dir", default=os.path.join(_HERE, "..", "results"))
    a = ap.parse_args()
    main(model=a.model, n_docs=a.n_docs, max_new=a.max_new, prompt_len=a.prompt_len,
         device=a.device, results_dir=a.results_dir)
