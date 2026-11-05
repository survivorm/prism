#!/usr/bin/env python
"""E-49b v3: ПОЛ СГЛАЖИВАНИЯ — решающий тест H2 (v2 + фиксы раунда-3 ревью).

Фиксы раунда 3:
  * СИД: синтетические контроли генерируются на РАЗНЫХ сидах (в exp49-функциях сид
    зашит — здесь локальные версии с внешним seed). Прежний CA CI был фиктивным
    (16 копий одного сэмпла).
  * crossmix переименован в interleave_check: арифметика τ на чередовании даёт
    τ ≈ (a+b−c)/a ≈ 2−c/a при малом c, т.е. τ>1 — ЭТО ДИАГНОСТИКА («межкадровые
    прыжки междоковые»), а не null. Настоящий междоковый null — blockmix (блоки
    длины W из другого документа: локальная непрерывность есть, дальнодействия нет).
  * real_minus_CA: bootstrap разности НЕЗАВИСИМЫХ выборок (boot_diff), не 512 пар.
  * s_lag сохраняется (real/block/shuffle/blockmix/CA).
  * assert T % W == 0 на каждом документе; мёртвый код удалён.

ВЕРДИКТ (правило объявлено до новых CA-чисел; τ ниже = когерентнее):
  H2_survives(W):= CI95_paired(τ_real − τ_block) < 0
                  И CI95_bootdiff(τ_real − τ_CA_smooth) < 0
                  И CI95_paired(τ_real − τ_blockmix) < 0
"""
from __future__ import annotations

import argparse
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
from exp49_leaf_info import load_model, smoothed_us, tau_triplets, med, boot_ci, LAGS  # noqa: E402

RNG = np.random.default_rng(49)
N_REP = 2          # реализации на каждый null (real: 1 — парный вердикт по документам)
N_SYN = 16         # синтетических серий на контроль, РАЗНЫЕ сиды


def safe_med(x, nd=4):
    x = np.asarray([v for v in x if v is not None], float)
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), nd) if len(x) else None


def nz(x):
    return x / x.norm()


def synth_white(T, D, seed):
    g = torch.Generator().manual_seed(seed)
    return [nz(torch.randn(D, dtype=torch.float64, generator=g)) for _ in range(T)]


def synth_coh(T, D, seed, delta=0.02):
    g = torch.Generator().manual_seed(seed)
    u = nz(torch.randn(D, dtype=torch.float64, generator=g))
    w = nz(torch.randn(D, dtype=torch.float64, generator=g))
    out = []
    for t in range(T):
        out.append(u.clone())
        u2 = math.cos(delta) * u + math.sin(delta) * w
        z = nz(torch.randn(D, dtype=torch.float64, generator=g))
        w = nz(w + 0.05 * (z - (z @ u) * u - (z @ w) * w))
        u = nz(u2)
    return out


def boot_diff(a, b, B=2000, seed=42):
    """CI разности медиан двух НЕЗАВИСИМЫХ выборок."""
    a = np.asarray([x for x in a if x is not None], float)
    b = np.asarray([x for x in b if x is not None], float)
    if len(a) < 3 or len(b) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for k in range(B):
        d[k] = np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
    return [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]


def frames_of(us, A, eps, r=16):
    out = []
    for x in us:
        J, _, _ = C.J_analytic(A, x, eps)
        Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
        _, Ue = torch.linalg.eigh(Gm)
        V = J.T @ Ue[:, -r:]
        out.append(torch.linalg.qr(V / V.norm(dim=0, keepdim=True))[0].float())
    return out


def cond(Vs):
    tau = med(tau_triplets(Vs))
    sl = {str(L): med([C.frame_kinematics(Vs[t], Vs[t + L])[0] for t in range(len(Vs) - L)])
          for L in LAGS}
    return tau, sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    A, U, eps, npz, jp = load_model(args)
    D = int(A.shape[1])
    docs = json.load(open(os.path.join(root, jp)))["docs"]
    z = np.load(npz, allow_pickle=True)
    out = {"model": args.model, "n_rep": N_REP, "n_syn": N_SYN,
           "verdict_rule": "paired(real−block)<0 AND bootdiff(real−CA)<0 AND paired(real−blockmix)<0"}

    for W in (4, 16):
        fr = {}
        for doc in docs:
            key = f"{doc['dom']}_{doc['si']}"
            f = z[key].item()
            u = torch.from_numpy(z[key + "_u"].astype(np.float64))
            Vs = frames_of(smoothed_us(u, np.asarray(f["hnorm"], dtype=float), W), A, eps)
            assert len(Vs) % W == 0, f"T={len(Vs)} не делится на W={W}"
            fr[key] = Vs
            print(f"W{W} frames {key}", flush=True)

        syn = {"CA_smooth": {"tau": [], "sl": []}, "CB_smooth": {"tau": [], "sl": []}}
        for i in range(N_SYN):
            for kind, fn in (("CA_smooth", synth_white), ("CB_smooth", synth_coh)):
                us_s = smoothed_us(torch.stack(fn(384, D, 2000 + 37 * i)), np.ones(384), W)
                t_, sl_ = cond(frames_of(us_s, A, eps))
                syn[kind]["tau"].append(t_)
                syn[kind]["sl"].append(sl_)
        print(f"W{W} synthetic done", flush=True)

        per = {}
        pair_blk, pair_bmix = {"d": [], "dom": []}, {"d": [], "dom": []}
        for key, Vs in fr.items():
            T = len(Vs)
            dom = key.split("_")[0]
            def put(name, tau, sl):
                per.setdefault(name, {"tau": [], "sl": [], "dom": []})
                per[name]["tau"].append(tau); per[name]["sl"].append(sl); per[name]["dom"].append(dom)
            tr_, sl_ = cond(Vs); put("real", tr_, sl_)
            others = [k for k in fr if k.split("_")[0] == dom and k != key]
            blk_d, bmix_d = [], []
            for j in range(N_REP):
                t_, s_ = cond([Vs[i] for i in RNG.permutation(T)]); put("shuffle", t_, s_)
                bl = [Vs[i * W:(i + 1) * W] for i in range(T // W)]
                bid = RNG.permutation(T // W)
                t_, s_ = cond([v for b in bid for v in bl[b]]); put("block", t_, s_)
                blk_d.append(t_)
                ok_ = fr[others[RNG.integers(len(others))]]
                # blockmix: блоки длины W из чужого документа (без шва: целиковые блоки)
                obl = [ok_[i * W:(i + 1) * W] for i in range(T // W)]
                obid = RNG.permutation(T // W)
                t_, s_ = cond([v for b in obid for v in obl[b]]); put("blockmix", t_, s_)
                bmix_d.append(t_)
                # interleave_check (диагностика, НЕ null: τ>1 ожидаем)
                mixed = [Vs[t] if t % 2 == 0 else ok_[t] for t in range(T)]
                t_, s_ = cond(mixed); put("interleave_check", t_, s_)
            t_, s_ = cond(fr[others[RNG.integers(len(others))]])
            put("other_doc", t_, s_)
            if tr_ is not None and blk_d:
                pair_blk["d"].append(tr_ - float(np.mean(blk_d))); pair_blk["dom"].append(dom)
            if tr_ is not None and bmix_d:
                pair_bmix["d"].append(tr_ - float(np.mean(bmix_d))); pair_bmix["dom"].append(dom)
            print(f"W{W} cond {key}", flush=True)

        agg = {}
        for k, v in list(per.items()) + [(f"syn_{s}", syn[s]) for s in syn]:
            agg[k] = {"tau": safe_med(v["tau"]), "tau_ci": boot_ci(v["tau"]),
                      "s_lag": {str(L): safe_med([x[str(L)] for x in v["sl"]]) for L in LAGS}}
        db = np.asarray(pair_blk["d"], float)
        dm = np.asarray(pair_bmix["d"], float)
        agg["paired_real_minus_block"] = [safe_med(list(db)), boot_ci(list(db))]
        agg["paired_real_minus_blockmix"] = [safe_med(list(dm)), boot_ci(list(dm))]
        agg["real_minus_CA_bootdiff"] = [
            safe_med([float(np.median(per["real"]["tau"]) - np.median(syn["CA_smooth"]["tau"]))])
            if per["real"]["tau"] and syn["CA_smooth"]["tau"] else None,
            boot_diff(per["real"]["tau"], syn["CA_smooth"]["tau"])]
        doms = sorted(set(per["real"]["dom"]))
        agg["by_dom"] = {d: {k: safe_med([t for t, dd in zip(per[k]["tau"], per[k]["dom"]) if dd == d])
                             for k in ("real", "block", "blockmix", "shuffle", "interleave_check")}
                        for d in doms}
        pb, pm, rc = agg["paired_real_minus_block"][1], agg["paired_real_minus_blockmix"][1], \
            agg["real_minus_CA_bootdiff"][1]
        survives = all(isinstance(ci, list) and ci[1] is not None and ci[1] < 0
                       for ci in (pb, pm, rc))
        agg["verdict"] = {"H2_survives": bool(survives),
                          "components": {
                              "block_pass": isinstance(pb, list) and pb[1] is not None and pb[1] < 0,
                              "blockmix_pass": isinstance(pm, list) and pm[1] is not None and pm[1] < 0,
                              "CA_pass": isinstance(rc, list) and rc[1] is not None and rc[1] < 0,
                              "block_vs_CA": ("block_inflated_by_boundaries"
                                              if abs((agg["block"]["tau"] or 0) -
                                                     (agg["syn_CA_smooth"]["tau"] or 0)) < 0.02
                                              else "block_has_structure")},
                          "real": agg["real"]["tau"], "block": agg["block"]["tau"],
                          "blockmix": agg["blockmix"]["tau"], "CA_floor": agg["syn_CA_smooth"]["tau"],
                          "CB_ceiling": agg["syn_CB_smooth"]["tau"],
                          "interleave_check(не null)": agg["interleave_check"]["tau"]}
        out[f"W{W}"] = agg
        print(f"W{W} VERDICT:", json.dumps(agg["verdict"], ensure_ascii=False), flush=True)

    path = os.path.join(root, "results", f"exp49b2_floor_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
