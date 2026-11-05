#!/usr/bin/env python
"""E-52c: симптом-ось для различения историй A (τ*=0.5) и B (τ=ёмкость).

Симптом считается НА ТОМ ЖЕ координатном срезе, что и τ: 128-строковая голова
A=g·W[U128]. Для трассовой последовательности u_t (e47b, жадная генерация)
строится «head-greedy» целевая токенизация варианта: tok_t = argmax U-надмножества
по логитам fp-головы (общая для всех вариантов — общая шкала), затем
  nll(var) = −(1/T) Σ log softmax(A_var·u_t)[tok_t],  acc = mean(argmax == tok).
Это не PPL полного vocab, а «разрешающая способность рид-аута» на том же
срезе: flat_* в E-45 на нём и давали PPL +7374%. Если B верна, (Δτ, nll) по
дозам RTN лежат на одной монотонной линии, а flat_* — выброс «сломанный выход
без потери разрешения» решается отдельно; если A — U-образность по τ.

g: из npz трассы (свежие) или --g-mode {auto,w,1pw} (auto: w~0 ⇒ 1+w).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import exp47_common as C  # noqa: E402
from exp49_leaf_info import load_model  # noqa: E402
from exp52a_perturb_axis import rtn  # noqa: E402

TMAX = 160


def head_from_ckpt(mp, U):
    files = sorted(glob.glob(os.path.join(mp, "*.safetensors")))
    HC = ("lm_head.weight", "model.language_model.lm_head.weight",
          "model.embed_tokens.weight", "model.language_model.embed_tokens.weight")
    NC = ("model.norm.weight", "model.language_model.norm.weight")
    head = normw = None
    from safetensors import safe_open
    for f in files:
        with safe_open(f, framework="pt") as sf:
            ks = set(sf.keys())
            if head is None:
                head = next((sf.get_tensor(c) for c in HC if c in ks), None)
            if normw is None:
                normw = next((sf.get_tensor(c) for c in NC if c in ks), None)
        if head is not None and normw is not None:
            break
    assert head is not None and normw is not None, mp
    return head.float(), normw.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--npz", required=True); ap.add_argument("--docs-json", required=True)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--g-mode", default="npz", choices=("npz", "auto"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.join(_HERE, "..")
    ns = argparse.Namespace(model="qwen")
    _, U, _, npz0, _ = load_model(ns)
    z = np.load(a.npz, allow_pickle=True)
    head, normw = head_from_ckpt(a.model_dir, U)
    if a.g_mode == "npz" and "g_norm" in z.files:
        g = torch.from_numpy(z["g_norm"].astype(np.float64)); mode = "npz"
    else:
        mode = "1+w" if float(normw.abs().mean()) <= 0.5 else "w"
        g = (1.0 + normw) if mode == "1+w" else normw
    print(f"[g] {mode} (|w|mean={float(normw.abs().mean()):.2f})", flush=True)
    A_fp = C.head_M(head[U], g)

    variants = [("fp", A_fp)] + [(f"rtn{b}", rtn(A_fp, b)) for b in (8, 4, 2)]
    if a.flat:
        fl = torch.load(os.path.join(root, "results", "exp45_flat_rows.pt"),
                        map_location="cpu", weights_only=False)
        variants += [(nm, C.head_M(fl[nm].float(), g))
                     for nm in ("flat_l0.01", "flat_l0.1", "flat_noleash")]

    docs = json.load(open(a.docs_json))["docs"]
    per = {nm: [] for nm, _ in variants}
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = torch.from_numpy(z[k + "_u"].astype(np.float64))[:TMAX]
        tgt = (u @ A_fp.T).argmax(1)                    # общая целевая токенизация (fp)
        for nm, A in variants:
            lg = u @ A.T
            lp = torch.log_softmax(lg.double(), dim=1)
            nll = float(-lp.gather(1, tgt[:, None]).mean())
            acc = float((lg.argmax(1) == tgt).float().mean())
            per[nm].append({"k": k, "nll": nll, "acc": acc})
        print(k, "done", flush=True)
    summ = {}
    for nm, _ in variants:
        nll = np.array([x["nll"] for x in per[nm]]); acc = np.array([x["acc"] for x in per[nm]])
        n0 = per["fp"][0]["nll"]
        summ[nm] = {"ppl_slice": round(float(np.exp(nll.mean())), 3),
                    "acc": round(float(acc.mean()), 4),
                    "nll_ratio_vs_fp": round(float(np.mean(nll) / n0), 3)}
        print(nm, summ[nm], flush=True)
    json.dump({"model_dir": a.model_dir, "g_mode": mode, "T": TMAX,
               "summary": summ, "per_doc": per}, open(a.out, "w"), indent=1, allow_nan=False)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
