#!/usr/bin/env python
"""E-49b: NULL-ТЕСТ артефакта сглаживания (к ревью E-49).

Вопрос: τ(W=16)=0.40/0.45 (против 0.92 без сглаживания) — это медленный КАНАЛ траектории
или АРТЕФАКТ перекрывания окон (соседние кадры делят W−1 отсчёта, и τ<1 получается даже
на серии без всякого порядка)?

Условия (одни и те же кадры V_r(W), разные ПОРЯДКИ):
  real      реальный временной порядок;
  shuffle   перестановка кадров целиком (уничтожает весь порядок);
  block     перемешивание блоков длины W (сохраняет внутренний порядок блока — если τ<1
            от локальной гладкости окна, блок-перестановка её сохраняет);
  cross     кадры из ДРУГИХ документов того же домена (сохраняет статистику домена).
Метрики: τ-медиана и лаг-кривая s(lag). NULL = shuffle/block/cross.
ВЫВОД E-49 валиден только если real заметно ниже всех null.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import exp47_common as C  # noqa: E402
from exp49_leaf_info import (load_model, smoothed_us, tau_triplets, med,  # noqa: E402
                             boot_ci, LAGS)

RNG = np.random.default_rng(49)


def frames_of(us, A, eps, r=16):
    out = []
    for x in us:
        J, _, _ = C.J_analytic(A, x, eps)
        Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
        _, Ue = torch.linalg.eigh(Gm)
        V = J.T @ Ue[:, -r:]
        out.append(torch.linalg.qr(V / V.norm(dim=0, keepdim=True))[0])
    return out


def cond(Vs):
    tau = tau_triplets(Vs)
    sl = {str(L): med([C.frame_kinematics(Vs[t], Vs[t + L])[0] for t in range(len(Vs) - L)])
          for L in LAGS}
    return med(tau), sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    A, U, eps, npz, jp = load_model(args)
    docs = json.load(open(os.path.join(root, jp)))["docs"]
    z = np.load(npz, allow_pickle=True)

    out = {"model": args.model}
    for W in (4, 16):
        per = {k: {"tau": [], "sl": []} for k in ("real", "shuffle", "block", "cross")}
        by_dom = {}
        for doc in docs:
            key = f"{doc['dom']}_{doc['si']}"
            f = z[key].item()
            u = torch.from_numpy(z[key + "_u"].astype(np.float64))
            us = smoothed_us(u, np.asarray(f["hnorm"], dtype=float), W)
            Vs = frames_of(us, A, eps)
            T = len(Vs)
            cands = {k: [] for k in per}
            cands["real"].append(Vs)
            cands["shuffle"].append([Vs[i] for i in RNG.permutation(T)])
            nb = max(1, T // W)
            bid = RNG.permutation(nb)
            bl = [Vs[i * W:(i + 1) * W] for i in range(nb)]
            cands["block"].append([v for b in bid for v in bl[b]])
            cands["shuffle"].append([Vs[i] for i in RNG.permutation(T)])   # вторая перестановка
            # cross: случайный другой документ того же домена (кадры чужого ряда)
            others = [d for d in docs if d["dom"] == doc["dom"] and d is not doc]
            ok_ = others[RNG.integers(len(others))]
            okk = f"{ok_['dom']}_{ok_['si']}"
            fo = z[okk].item()
            uo = torch.from_numpy(z[okk + "_u"].astype(np.float64))
            cands["cross"].append(frames_of(smoothed_us(uo, np.asarray(fo["hnorm"], dtype=float), W), A, eps))
            for k, series in cands.items():
                for Vsx in series:
                    t_, sl_ = cond(Vsx)
                    per[k]["tau"].append(t_)
                    per[k]["sl"].append(sl_)
            by_dom.setdefault(doc["dom"], {"real": [], "shuffle": []})
            by_dom[doc["dom"]]["real"].append(per["real"]["tau"][-1])
            by_dom[doc["dom"]]["shuffle"].append(per["shuffle"]["tau"][-2])
            print(f"W{W} {key} done", flush=True)
        out[f"W{W}"] = {
            k: {"tau": round(float(np.nanmedian(v["tau"])), 4),
                "tau_boot95": boot_ci([t for t in v["tau"] if t is not None]),
                "s_lag": {str(L): round(float(np.nanmedian([x[str(L)] for x in v["sl"]])), 4)
                          for L in LAGS}}
            for k, v in per.items()}
        out[f"W{W}_bydom"] = {d: {"real": round(float(np.nanmedian(v["real"])), 3),
                                  "shuf": round(float(np.nanmedian(v["shuffle"])), 3)}
                              for d, v in by_dom.items()}
        print(json.dumps(out[f"W{W}"], indent=1), flush=True)
    path = os.path.join(root, "results", f"exp49b_smooth_null_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
