#!/usr/bin/env python
"""E-52a PREFLIGHT: ось отклонения τ = τ* + ε (E52_review, тесты A–C).

Вопрос не «болеет ли», а «куда и на сколько отклоняется τ под батареей
возмущений». Нули/метрика — E-49b (W4, paired по документам, τ-медиана).

Возмущения (дозные семейства, всё детерминировано по сидам):
  FLAT  : готовые строки головы E-45 (flat_l0.01 < flat_l0.1 < flat_noleash,
          доза = глубина уплощения; qwen-only — кэш exp45_flat_rows.pt);
  RTN   : симметричная per-row квантизация A, int8 → int4 → int2 (доза = биты);
  NOISE : мультипликативный шум строк A, σ = 0.005/0.02/0.08 (реад-аут);
  U-NS  : шум скрытых u перед сглаживанием, σ = 0.01/0.02/0.04 (трэнк-прокси).
Тесты: A — знак Δτ (одно семейство ⇒ знак однообразен?); B — монотонность
по дозе (Spearman); C — перенос на вторую модель (сравнение потом по JSON).

К администратору чисел: τ считается top-16 РЕБРАМИ J через спектр JJᵀ [128,128]
(ранг ≤128 ⇒ тождественно [D,D]-варианту exp49b2, в ~50 раз быстрее); T=96,
W=4 — самосогласовано внутри прогона, с baseline E-49b (T=384) абсолютные
уровни не сравниваем — только paired-Δ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import exp47_common as C  # noqa: E402
from exp49_leaf_info import load_model, smoothed_us, tau_triplets, boot_ci  # noqa: E402

TMAX = 96
W = 4


def _seed(s):
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") % (1 << 62)


def rtn(A, bits):
    """Симметричная per-row RTN: scale = row-absmax/(2^(b-1)-1)."""
    lev = 2 ** (bits - 1) - 1
    sc = A.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / lev
    return (torch.clamp((A / sc).round(), -lev, lev) * sc)


def frames16(A, us, eps):
    """Top-16 рам листа через спектр JJᵀ [K,K] (ранг J ≤ K=128 < D)."""
    out = []
    for x in us:
        J, _, _ = C.J_analytic(A, x, eps)
        Gs = J @ J.T
        Gs = 0.5 * (Gs + Gs.T)
        _, Ue = torch.linalg.eigh(Gs)
        V = J.T @ Ue[:, -16:]
        out.append(torch.linalg.qr(V / V.norm(dim=0, keepdim=True))[0])
    return out


def tau_of(A, us, eps):
    t = tau_triplets(frames16(A, us, eps))
    return float(np.median(t)) if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--families", default="flat,rtn,noise,u_ns")
    ap.add_argument("--npz", default=None); ap.add_argument("--docs-json", default=None)
    ap.add_argument("--model-dir", default=None,
                    help="пересобрать A из этого чекпойнта (тест D, без full-load: safetensors slice)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    fams = set(args.families.split(","))

    ns = argparse.Namespace(model=args.model)
    A0, U, eps, npz, jp = load_model(ns)
    g = load_model.last_g
    if args.model_dir:      # тест D: A/g/eps из чекпойнта другого размера
        import glob
        from safetensors import safe_open
        mp = args.model_dir
        cfg = json.load(open(os.path.join(mp, "config.json")))
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        files = sorted(glob.glob(os.path.join(mp, "*.safetensors")))
        head = normw = None
        for f in files:
            with safe_open(f, framework="pt") as sf:
                ks = set(sf.keys())
                for cand in ("lm_head.weight", "model.embed_tokens.weight"):
                    if head is None and cand in ks:
                        head = sf.get_tensor(cand)
                if normw is None and "model.norm.weight" in ks:
                    normw = sf.get_tensor("model.norm.weight")
            if head is not None and normw is not None:
                break
        assert head is not None and normw is not None, f"head/norm не найдены в {mp}"
        g = 1.0 + normw.float()
        A0 = C.head_M(head.float()[U], g)
        print(f"[model-dir] {mp}: A {tuple(A0.shape)}, tied-предпосылка {head.shape[0] > 100000}", flush=True)
    if args.npz:
        npz = args.npz
    docs = json.load(open(args.docs_json or os.path.join(root, jp)))["docs"][: args.n_docs]
    z = np.load(npz, allow_pickle=True)

    base_us = {}
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = torch.from_numpy(z[k + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(z[k].item()["hnorm"], dtype=float)[:TMAX]
        base_us[k] = (u, hn)
        # sanity: рамочный маршрут 128-трюка против [D,D] — один раз
    if True:
        k0 = next(iter(base_us)); u0, hn0 = base_us[k0]
        us0 = smoothed_us(u0, hn0, W)
        J, _, _ = C.J_analytic(A0, us0[5], eps)
        V_small = J.T @ torch.linalg.eigh(J @ J.T)[1][:, -16:]
        V_big = (J.T @ J) @ torch.linalg.eigh((J.T @ J).double())[1][:, -16:]  # [D,D]
        ang = torch.acos(torch.linalg.svdvals(
            torch.linalg.qr(V_small)[0].T @ torch.linalg.qr(V_big)[0]).clamp(-1 + 1e-12, 1 - 1e-12))
        assert float(ang.max()) < 1e-4, f"128-трюк расходится с [D,D]: {float(ang.max())}"  # 1.4e-6实测

    variants = [("base", A0, None)]
    if "flat" in fams and args.model == "qwen":
        fl = torch.load(os.path.join(root, "results", "exp45_flat_rows.pt"),
                        map_location="cpu", weights_only=False)
        for nm in ("flat_l0.01", "flat_l0.1", "flat_noleash"):
            variants.append((nm, C.head_M(fl[nm].float(), g), None))
    if "rtn" in fams:
        for b in (8, 4, 2):
            variants.append((f"rtn{b}", rtn(A0, b), None))
    if "noise" in fams:
        for s in (0.005, 0.02, 0.08):
            xi = torch.randn(A0.shape, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(_seed(f"An{s}")))
            variants.append((f"noise{s}", A0 * (1.0 + s * xi), None))
    u_variants = ([("u_ns0.01", 0.01), ("u_ns0.02", 0.02), ("u_ns0.04", 0.04)]
                  if "u_ns" in fams else [])

    out = {"model": args.model, "W": W, "T": TMAX, "n_docs": len(docs),
           "note": "paired Δτ (variant−base) per doc; 128-трюк верифицирован против [D,D]"}
    taus = {}
    for nm, A, _ in variants:
        vals = {}
        for k, (u, hn) in base_us.items():
            vals[k] = tau_of(A, smoothed_us(u, hn, W), eps)
        taus[nm] = vals
        print(nm, "done", flush=True)
    for nm, s in u_variants:
        xi = torch.randn((1, A0.shape[1]), dtype=torch.float64,
                         generator=torch.Generator().manual_seed(_seed(f"u{nm}")))
        vals = {}
        for k, (u, hn) in base_us.items():
            un = u + s * u.norm(dim=1, keepdim=True) * xi.to(u.dtype)
            vals[k] = tau_of(A0, smoothed_us(un, hn, W), eps)
        taus[nm] = vals
        print(nm, "done", flush=True)

    dom = {k: k.split("_")[0] for k in base_us}
    summ = {}
    for nm in taus:
        if nm == "base":
            summ[nm] = {"tau": float(np.median(list(taus["base"].values())))}
            continue
        dd = [taus[nm][k] - taus["base"][k] for k in base_us
              if taus[nm][k] is not None and taus["base"][k] is not None]
        ddom = {}
        for d0 in set(dom.values()):
            sub = [taus[nm][k] - taus["base"][k] for k in base_us
                   if dom[k] == d0 and taus[nm][k] is not None]
            ddom[d0] = float(np.median(sub)) if sub else None
        summ[nm] = {"d_tau": float(np.median(dd)), "ci": boot_ci(dd),
                    "dom_sign_pos": sum(1 for v in ddom.values() if (v or 0) > 0),
                    "n_dom": len(ddom), "per_dom": ddom,
                    "tau": float(np.median(list(taus[nm].values())))}
        print(nm, summ[nm]["d_tau"], summ[nm]["ci"], flush=True)
    out["per_doc"] = taus
    out["summary"] = summ
    path = args.out or os.path.join(root, "results", f"exp52a_perturb_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
