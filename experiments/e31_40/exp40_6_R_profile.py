#!/usr/bin/env python3
"""E-40.6 — профили R_can(l) и R_raw(l) (docs/e40_todo.md §8).

R_can = λmax(E_K)/tr(E_K), E_K = Σ_t P(p_t)J_t(P(p_t)J_t)ᵀ (канонический срез,
сумма по позициям); R_raw — та же формула на raw J (диагностика амплитуды, не
параметр порядка). Критерий (a′)∧(a)∧(b)∧(c); L0 в кэше нет ⇒ вердикт
односторонний, флаг not_testable_no_L0 (фикс (e) аудита). κ(l) — exp38d r=16.
(c) — устойчивость к удалению топ-gauge-моды (top eig корпусного ESG raw).

Запуск: .venv/bin/python experiments/exp40_6_R_profile.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from _e38_cache import load_layer  # noqa: E402
from exp40_common import DOMAINS, LAYERS, N_PER_DOMAIN, ROOT, S, dump_json, header, tag  # noqa: E402

E38D = os.path.join(ROOT, "results", "exp38d_holonomy_spectrum.json")
BODY = [6, 12, 18, 22]
B_BOOT = 1000


def shares(E: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(E)[::-1]
    return float(ev[0] / max(ev.sum(), 1e-300))


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    e38d = json.load(open(E38D))
    per_layer, per_doc = {}, {}
    EK_docs, EK_docs_nog, Eraw_docs = {}, {}, {}
    for L in LAYERS:
        tl = time.time()
        ck = load_layer(L, want=("J", "P"))
        J, P = ck["J"], ck["P"]
        # корпусный ESG raw и его топ-мода (gauge-мода для (c))
        Grow = torch.zeros(J.shape[-2], J.shape[-2], dtype=torch.float64)
        docs = {}
        Jg_all = torch.empty_like(J)
        for di in range(J.shape[0]):
            for t in range(S):
                p = P[di, t].double()
                p = p / p.sum()
                Jd = J[di, t].double()
                Jg_all[di, t] = (Jd * p[:, None] - p[:, None] * (p @ Jd)).float()
                Grow += Jd @ Jd.T
        w, U = torch.linalg.eigh(Grow)
        u_gauge = U[:, -1].double()                     # top ESG mode
        Proj = torch.eye(J.shape[-2], dtype=torch.float64) - torch.outer(u_gauge, u_gauge)
        for d_of, dom in enumerate(DOMAINS):
            for si in range(N_PER_DOMAIN):
                di = d_of * N_PER_DOMAIN + si   # плоский индекс документа
                # (кэш лежит домен-мажорно: (wiki,0..7),(math,0..7),…); раньше
                # сюда попадал индекс ДОМЕНА 0..3, и все 32 «документа» были
                # четырьмя первыми wiki-документами, повторёнными по 8 раз
                d = f"{dom}#{si}"
                Ekg = torch.zeros(J.shape[-2], J.shape[-2], dtype=torch.float64)
                Ekg_ng = torch.zeros_like(Ekg)
                Er = torch.zeros_like(Ekg)
                for t in range(S):
                    Jg = Jg_all[di, t].double()
                    Jrr = J[di, t].double()
                    Ekg += Jg @ Jg.T
                    Ekg_ng += Proj @ (Jg @ Jg.T) @ Proj.T
                    Er += Jrr @ Jrr.T
                docs[d] = {"R_can": shares(Ekg.numpy()), "R_raw": shares(Er.numpy())}
                EK_docs.setdefault(str(L), {})[d] = Ekg.numpy()
                EK_docs_nog.setdefault(str(L), {})[d] = Ekg_ng.numpy()
                Eraw_docs.setdefault(str(L), {})[d] = Er.numpy()
        Rcan = {d: v["R_can"] for d, v in docs.items()}
        Rraw = {d: v["R_raw"] for d, v in docs.items()}
        Rl_can = shares(sum(EK_docs[str(L)].values()))
        Rl_can_ng = shares(sum(EK_docs_nog[str(L)].values()))
        Rl_raw = shares(sum(Eraw_docs[str(L)].values()))
        per_layer[str(L)] = {"R_can_corpus": Rl_can, "R_raw_corpus": Rl_raw,
                             "R_can_nogauge": Rl_can_ng,
                             "label": "natural × canonical/raw × top × right"}
        per_doc[str(L)] = docs
        print(f"L{L}: R_can={Rl_can:.4f} R_raw={Rl_raw:.4f} nog={Rl_can_ng:.4f} "
              f"({time.time()-tl:.0f}s)")
        del ck, J, Jg_all

    prof = {L: per_layer[str(L)]["R_can_corpus"] for L in LAYERS}
    R23 = prof[23]
    a_prime = R23 >= max(prof.values()) - 1e-15
    a = min(prof[l] for l in BODY) <= 0.5 * R23
    # (b) парный bootstrap по документам: trough пересчитывается внутри реплики
    rng = np.random.default_rng(40)
    dkeys = list(per_doc["6"].keys())
    hits = 0
    for _ in range(B_BOOT):
        idx = rng.integers(0, len(dkeys), len(dkeys))
        Rs = {}
        for L in LAYERS:
            acc = np.zeros((128, 128))
            for i in idx:
                acc += EK_docs[str(L)][dkeys[i]]
            Rs[L] = shares(acc)
        peak = max(Rs, key=Rs.get)
        trough = min(BODY, key=lambda l: Rs[l])
        hits += Rs[peak] - Rs[trough] > 0
    b = hits / B_BOOT >= 0.95
    # (c) инвариантность вердикта к удалению gauge-моды (тот же порядок слоёв)
    nog = {L: per_layer[str(L)]["R_can_nogauge"] for L in LAYERS}
    c = (sorted(nog.values(), reverse=True) == sorted(prof.values(), reverse=True)) \
        and max(nog, key=nog.get) == max(prof, key=prof.get)
    n_sat = sum([a_prime, a, b, c])
    vals = [prof[l] for l in LAYERS]
    mono = all(x >= y for x, y in zip(vals, vals[1:])) or all(x <= y for x, y in zip(vals, vals[1:]))
    if n_sat == 4 and not mono:
        verdict = "phase"
    elif n_sat == 3:
        verdict = "ambiguous"
    elif mono:
        verdict = "monotone_or_flat"
    else:
        verdict = "ambiguous"
    verdict += "__one_sided_no_L0"

    kap = {int(L): e38d["metrics"][str(L)]["16"]["kappa_median"] for L in map(str, LAYERS)
           if str(L) in e38d["metrics"] and "16" in e38d["metrics"][str(L)]}
    from scipy.stats import spearmanr
    ls = sorted(kap)
    rho_k = spearmanr([prof[l] for l in ls], [kap[l] for l in ls])
    # partial-out частотой токена (per-doc, PER-LAYER): covariate = mean_t max p_t на СВОЁМ
    # слое; раньше cov бралась только с L23, а поле называлось per-layer (аудит v2 §3.5)
    partial = {}
    for L in LAYERS:
        ck = load_layer(int(L), want=("P",))
        Pdoc = ck["P"].double()
        cov = {}
        for d_of, dom in enumerate(DOMAINS):
            for si in range(N_PER_DOMAIN):
                di = d_of * N_PER_DOMAIN + si   # плоский индекс документа
                # (кэш лежит домен-мажорно: (wiki,0..7),(math,0..7),…; индекс ДОМЕНА
                # сюда попадать не должен — иначе 32 «документа» = четыре wiki, ×8)
                cov[f"{dom}#{si}"] = float(Pdoc[di].max(dim=-1).values.mean())
        del ck
        x = np.array([per_doc[str(L)][d]["R_raw"] - per_doc[str(L)][d]["R_can"] for d in dkeys])
        z = np.array([cov[d] for d in dkeys])
        resid = x - np.polyval(np.polyfit(z, x, 1), z)
        rho = spearmanr(x, z)
        partial[str(L)] = {"rho_diff_freq": float(rho.statistic), "p": float(rho.pvalue),
                           "resid_sd": float(resid.std()), "cov_basis": "per-layer P"}

    tags = [tag(slice="gf", p_slice="p_t", aggregation="ratio_of_means", window="top",
                side="right", norm_type="energy", layer=None, rank=None)]
    rep = header("E-40.6", "docs/e40_todo.md §8",
                 {"exp27b_jcache_L6": "/data/prism_kernels/exp27b/results/exp27b_jcache_L6.pt",
                  "exp38d_holonomy_spectrum": E38D},
                 channel_tags=tags,
                 extra={"per_layer": per_layer, "per_doc": per_doc,
                        "per_doc_S06_R_can": {str(L): {d: v["R_can"] for d, v in per_doc[str(L)].items()}
                                              for L in LAYERS},
                        "profile_can": "monotone" if mono else "non_monotone",
                        "criteria": {"a_prime": bool(a_prime), "a": bool(a),
                                     "b_bootstrap": bool(b), "c_gauge_invariant": bool(c),
                                     "n_satisfied": n_sat},
                        "verdict": verdict,
                        "kappa_per_layer_exp38d_r16": kap,
                        "corr_Rcan_kappa": {"spearman": float(rho_k.statistic),
                                            "p": float(rho_k.pvalue), "layers": ls},
                        "interpretation": ("channels_not_separable"
                                           if abs(rho_k.statistic) > 0.8
                                           else "radial_and_angular_orthogonal"),
                        "partial_out_token_freq_Rraw_minus_Rcan": partial,
                        "deviations_from_spec": [
                            {"id": "D6_no_L0", "description":
                             "L0 нет в кэше (фикс (e)) ⇒ критерий односторонний"},
                            {"id": "D6_gauge_mode", "description":
                             "gauge-мода = топ-мода корпусного raw-ESG (top eig Σ_t JᵀJ)"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_6_R_profile.json")
    dump_json(rep, out)
    print(f"wrote {out}: verdict={verdict}, ρ(R_can,κ)={rho_k.statistic:.2f}")
    return rep


if __name__ == "__main__":
    main()
