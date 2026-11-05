#!/usr/bin/env python
"""E-47b (P1): кинематика листа L23 при СВОБОДНОЙ ГЕНЕРАЦИИ (Kaggle T4 / локально).

Постановка/критерии (K1.1–K1.4, пререгистрированы) — docs/leaf_dynamics.md §5.2,§6;
числа P0 и прогноз (насыщение s=0.383·tanh(a/0.715), выход с полки на мелком шаге) — §8.1.

Протокол:
  * 4 домена x n_docs документов; промпт = первые prompt_len токенов калибровочного пула
    (те же датасеты, что строили кэш exp27b; точное совпадение документов НЕ гарантируется
    потоком датасета — пишем sha256 промптов в артефакт);
  * greedy-генерация max_new шагов с DynamicCache; на каждый шаг берём h = hidden_states[n_layers]
    (выход L23 = вход финальной RMSNorm; НЕ last_hidden_state — тот уже post-norm);
  * кинематика из exp47_common: a_t, s_t (главные углы V16), p_t, g_t (доля движения u,
    видимая листу: ||Pi_{span(Va∪Vb)} du|| / ||du||), Omega_t; тривиальные фичи-конкуренты:
    энтропия logits, top-1 prob, флаг «токен=предыдущий»;
  * ПЕТЛИ: find_loops по сгенерированным токенам (onset = начало 3-го повтора);
  * СОХРАНЯЕМ СЫРЬЁ (npz): фичи per step, V16 fp16 [T,16,D], u fp16 — весь P2/P3-анализ
    пересобирается офлайн без перепрогона.

Базис U128 — КОНСТАНТА из кэша exp27b (источник истины всех опубликованных чисел;
источник: exp27b_jcache_L23.pt, pool_fp wiki 0cc16ec344e2 / math e71cc282c85d / code
209e83b1898b / chat 2afac288344a), с двумя самопроверками на лету:
  (i)  маршрут: H_head @ W_lm^T vs model(...).logits < 1e-3 (как exp45b) — падает громко;
  (ii) частотный базис, пересобранный с пула этим же датасетом — пересечение в % (дескриптив;
       поток датасета мог сдвинуться — на выводы не влияет, U128 фиксирован константой).

Kaggle: сигнатура main(model=..., device=..., results_dir=...) — драйвер билда подставляет
локальный путь модели/устройство/каталог. Локально: аргументы CLI.

Запуск (локально, CPU медленно): ../.venv/bin/python exp47b_gen_kinematics.py --n-docs 2
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
_ROOT = os.path.dirname(_HERE)          # Kaggle: бандл примонтирован как <root>/{prism,experiments}
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import exp47_common as C   # noqa: E402
from prism.calibration import build_global_token_set, load_calibration_data  # noqa: E402

RNG_SEED = 47
# U128 — частотный базис exp27b (константа-источник; см. докстринг)
U128 = [0, 3, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 30, 32, 82, 87, 198, 220, 264, 271, 279, 303, 310, 314, 318, 321, 328, 348,
        351, 353, 357, 364, 369, 383, 393, 411, 413, 421, 423, 424, 430, 440, 449, 458,
        466, 471, 494, 498, 506, 513, 524, 539, 579, 599, 615, 628, 635, 641, 653, 660,
        668, 694, 695, 716, 725, 735, 748, 777, 791, 799, 821, 864, 948, 984, 1000, 1040,
        1066, 1076, 1088, 1092, 1116, 1132, 1172, 1220, 1238, 1317, 1330, 1332, 1345,
        1358, 1381, 1472, 1518, 1521, 1532, 1602, 1608, 1680, 1754, 1831, 1973, 2107,
        2228, 2394, 2397, 2424, 2574, 2582, 2785, 2957, 2972, 3018, 3067, 4370, 4558,
        56014, 248044, 248046]
POOL_FP_EXPECT = {"wiki": "0cc16ec344e2", "math": "e71cc282c85d",
                  "code": "209e83b1898b", "chat": "2afac288344a"}


def _sha(x) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()[:12]


def _step(V_prev, u_prev, u_cur, A, eps, h_cur):
    """Фичи перехода: V/u текущего шага + (a,s,p,g); при V_prev=None (первый шаг)
    кинематика не определена — nan, но V/u считаем."""
    Vb, _ = C.V16_from_J(C.J_analytic(A, h_cur, eps)[0])
    if V_prev is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), Vb
    Va = V_prev
    s, p, _P, _Th = C.frame_kinematics(Va, Vb)
    a = float(torch.acos(((u_prev * u_cur).sum()).clamp(max=1 - 1e-12)))
    du = u_cur - (u_prev @ u_cur) * u_prev
    ndu = float(du.norm())
    g = float("nan")
    if ndu > 1e-9:
        q = torch.linalg.qr(torch.cat([Va, Vb], dim=1))[0]
        g = float((q.T @ (du / ndu)).norm())
    return a, s, p, g, Vb


def main(model="/data/Qwen/Qwen3.5-0.8B", n_docs=8, max_new=384, prompt_len=64,
         device="cpu", results_dir="results", do_verify_basis=True,
         rtn_bits=None, rtn_group=128, dtype="float32", load_in_8bit=False):
    torch.set_num_threads(4)
    torch.manual_seed(RNG_SEED)
    dev = torch.device(device)
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # type: ignore

    t0 = time.time()
    kw = {}
    if load_in_8bit:      # честный PTQ (bnb int8): trunk квант, embed/head точные
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                  device_map="cuda:0")
    m = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype),
                                             attn_implementation="eager", **kw)
    m.eval()
    if not load_in_8bit:
        m.to(dev)
    if rtn_bits:      # full-model RTN (тест A/B честный: меняется и динамика u).
        lev = 2 ** (rtn_bits - 1) - 1
        nq = 0
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight.data
                wf = w.reshape(w.shape[0], -1, rtn_group).double()
                sc = wf.abs().amax(-1, keepdim=True).clamp(min=1e-12) / lev
                mod.weight.data = ((wf / sc).round().clamp(-lev, lev) * sc).reshape_as(w).to(w.dtype)
                nq += 1
        print(f"[rtn] int{rtn_bits} per-group({rtn_group}) деQuantized Linear-слоёв: {nq}", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    n_layers = int(m.config.num_hidden_layers)
    norm = m.model.norm
    eps = float(getattr(norm, "variance_epsilon",
                        getattr(norm, "eps", m.config.rms_norm_eps)))
    # e52k падал здесь на GPU: `pre`/`raw` ниже снимаются на CPU, а `wn` оставался на
    # cuda:0, и проверка g-режима (голый w против 1+w) давала cuda/cpu mismatch. wn дальше
    # употребляется только через .cpu() (C.head_M, feats), поэтому CPU-размещение безопасно.
    wn = norm.weight.detach().float().cpu()
    # эмпирика семантики RMSNorm: gemma-style g=1+w против голого g=w (Qwen2/3)
    with torch.no_grad():
        prb = torch.randint(1, 1000, (1, 8), device=dev)
        pre_c = []
        hk = norm.register_forward_hook(lambda _m, inp, _o: pre_c.append(inp[0].detach()))
        raw = m.model(input_ids=prb).last_hidden_state[0, -1].detach().float().cpu()
        hk.remove()
        pre = pre_c[0][0, -1].detach().float().cpu()   # raw уже cpu — device-консистентность (e52k cuda fix)
        r = (pre.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        e_w = float((wn * r * pre - raw).abs().max())
        e_1w = float(((1.0 + wn) * r * pre - raw).abs().max())
        g_mode = "w" if e_w <= e_1w else "1p_w"
        assert min(e_w, e_1w) < 1e-3, f"norm не закрыт ни одной g: w={e_w:.1e} 1+w={e_1w:.1e}"
        print(f"[norm] g-режим: {g_mode} (err w={e_w:.1e}, 1+w={e_1w:.1e})", flush=True)
    g_n = (wn if g_mode == "w" else 1.0 + wn)
    W = m.lm_head.weight.detach()
    U = torch.tensor(U128, device=dev)
    A = C.head_M(W[U].cpu().float(), g_n.cpu()).to(dev)      # [128,D] fp64 на устройстве
    tied = m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()

    # маршрут: H_head @ W_lm[U]^T == logits[U]
    with torch.no_grad():
        probe = torch.randint(1, 1000, (1, 8), device=dev)
        hs = m.model(input_ids=probe).last_hidden_state[0, -1].float()
        rt = float((hs @ W[U].T.float().cpu().to(dev)
                    - m(input_ids=probe).logits[0, -1, U].float()).abs().max())
    print(f"[route] max|H@W_U^T - logits_U| = {rt:.2e}; tied={tied}", flush=True)
    assert rt < 1e-3, f"маршрут логитов не сошёлся: {rt}"

    docs_out, feats_npz = [], {}
    basis_overlap = None
    for dom in ("wiki", "math", "code", "chat"):
        pool = load_calibration_data(model, max_length=256, min_length=64,
                                     n_samples=n_docs, domain=dom)
        assert len(pool) >= n_docs
        if do_verify_basis and dom == "wiki":       # дорого: один домен
            fresh = build_global_token_set(m, pool[:n_docs], K=len(U128))
            basis_overlap = len(set(fresh) & set(U128)) / len(U128)
            print(f"[basis] overlap fresh-vs-cache U128: {basis_overlap:.2%}", flush=True)
        for si in range(n_docs):
            ids = pool[si]["input_ids"][:, :prompt_len].to(dev)
            prompt_sha = _sha(pool[si]["input_ids"][0, :prompt_len].tolist())
            # префиксный проход: h_23 промпта (последняя позиция) + cache
            with torch.no_grad():
                cache = DynamicCache(config=m.config)
                out = m(input_ids=ids, past_key_values=cache, use_cache=True,
                        output_hidden_states=True)
            h = out.hidden_states[n_layers][0, -1].double().to(dev)   # h_23 (pre-final-norm)
            logits = out.logits[0, -1]
            feats = {"a": [], "s": [], "p": [], "g": [], "ent": [], "top1": [],
                     "eq_prev": [], "hnorm": []}
            u = h / h.norm()
            V_prev = u_prev = None
            # Кадр при промптной позиции — отдельно от V_prev (NaN первого шага не трогаем).
            V_at_cur = C.V16_from_J(C.J_analytic(A, h, eps)[0])[0]
            tokens = ids[0].tolist()
            V_stack, u_stack = [], []
            for step in range(max_new):
                p = torch.softmax(logits.float(), dim=-1)
                ent = float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum())
                nxt = int(torch.argmax(logits))
                feats["ent"].append(ent)
                feats["top1"].append(float(p.max()))
                feats["eq_prev"].append(int(nxt == tokens[-1]))
                feats["hnorm"].append(float(h.norm()))
                u_stack.append(u.float().cpu().numpy())
                tokens.append(nxt)
                with torch.no_grad():
                    out = m(input_ids=torch.tensor([[nxt]], device=dev),
                            past_key_values=cache, use_cache=True,
                            output_hidden_states=True)
                h_new = out.hidden_states[n_layers][0, -1].double().to(dev)
                logits = out.logits[0, -1]
                u_new = h_new / h_new.norm()
                a, s, pl, gg, V_cur = _step(V_prev, u, u_new, A, eps, h_new)
                feats["a"].append(a); feats["s"].append(s)
                feats["p"].append(pl); feats["g"].append(gg)
                V_prev, u_prev = V_cur, u
                u = u_new
                # Аудит E46+ (A6): писался V_cur = кадр СЛЕДУЮЩЕЙ позиции, то есть
                # сохранённые u[i] и V[i] были сдвинуты на шаг. Теперь оба на позиции i.
                V_stack.append(V_at_cur.float().cpu().numpy())
                V_at_cur = V_cur
                h = h_new
            gen_tokens = tokens[prompt_len:]
            loops = C.find_loops(gen_tokens)
            aa = np.array(feats["a"], dtype=float)
            ss = np.array(feats["s"], dtype=float)
            pp = np.array(feats["p"], dtype=float)
            # посегментная сводка (in-loop по ДЕТЕРМИНИРОВАННОМУ детектору; вне — всё остальное)
            in_mask = np.zeros(max_new, dtype=bool)
            for lp in loops:
                on0 = min(lp[0], max_new - 1)
                in_mask[on0: min(lp[2], max_new)] = True
            acf = [C.autocorr_peak(torch.from_numpy(
                       ss[min(lp[0], len(ss)-1):min(lp[2], len(ss))] - np.nanmean(ss)), lp[3])
                   for lp in loops]
            r_seg = float("nan")
            if in_mask.any() and (~in_mask).sum() > 20:
                x, y = ss[in_mask], aa[in_mask]
                ok = ~(np.isnan(x) | np.isnan(y))
                if ok.sum() > 5:
                    x = x - x.mean(); y = y - y.mean()
                    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
                    r_seg = float((x * y).sum() / d) if d > 0 else float("nan")
            docs_out.append({
                "dom": dom, "si": si, "prompt_sha": prompt_sha,
                "n_loops": len(loops),
                "loops": [list(lp) for lp in loops],
                "a_med": float(np.nanmedian(aa)), "s_med": float(np.nanmedian(ss)),
                "p_med": float(np.nanmedian(pp)),
                "g_med": float(np.nanmedian(feats["g"])),
                "excess_a": C.band_power(torch.from_numpy(np.nan_to_num(aa))),
                "excess_s": C.band_power(torch.from_numpy(np.nan_to_num(ss))),
                "in_loop": {"a": float(np.nanmedian(aa[in_mask])) if in_mask.any() else None,
                            "s": float(np.nanmedian(ss[in_mask])) if in_mask.any() else None,
                            "p": float(np.nanmedian(pp[in_mask])) if in_mask.any() else None,
                            "acf_peak_by_loop": [None if x != x else float(x) for x in acf],
                            "r_s_a_in": r_seg if r_seg == r_seg else None},
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
        del pool

    os.makedirs(results_dir, exist_ok=True)
    out = {"params": {"protocol": "docs/leaf_dynamics.md §5.2 (P1)", "model": str(model),
                      "n_docs": n_docs, "max_new": max_new, "prompt_len": prompt_len,
                      "decoding": "greedy", "seed": RNG_SEED,
                      "U128_source": "exp27b cache constant (pool_fp см. в докстринге)",
                      "basis_overlap_fresh": basis_overlap,
                      "route_check": rt, "tied": bool(tied)},
           "docs": docs_out}
    with open(os.path.join(results_dir, "exp47b_gen_kinematics.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, allow_nan=False)
    feats_npz["g_norm"] = g_n.cpu().numpy().astype(np.float32)   # источник истины для e52a
    feats_npz["eps"] = np.float64(eps)
    np.savez_compressed(os.path.join(results_dir, "exp47b_gen_kinematics.npz"), **feats_npz)
    print(f"saved. total {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rtn-bits", type=int, default=None,
                    help="full-model RTN: симметрично per-group квантовать ВСЕ Linear (dequant in-place)")
    ap.add_argument("--rtn-group", type=int, default=128)
    ap.add_argument("--results-dir", default=os.path.join(_HERE, "..", "results"))
    a = ap.parse_args()
    main(model=a.model, n_docs=a.n_docs, max_new=a.max_new, prompt_len=a.prompt_len,
         device=a.device, results_dir=a.results_dir,
         rtn_bits=a.rtn_bits, rtn_group=a.rtn_group)
