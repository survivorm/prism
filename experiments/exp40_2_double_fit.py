#!/usr/bin/env python3
"""E-40.2 — двойной фит C_F / C_top16 (docs/e40_todo.md §4).

Аудит (g): exp31 считал C_F на RAW J ⇒ пересчёт canonical P(p_t)J из кэша exp27b
(не рефит). Побитовый якорь: raw-пересчёт = опубликованные C_med/C_top_med E-31.
Сетка Δ = 1..32 (Δ ≥ 1, целые; is_dyadic=False). 4 формы (экспонента, степень,
растянутая, степень×лог-периодика), AICc/BIC, bootstrap по документам (B=1000,
рефит лучшей формы, percentile CI) — НЕ блочный по Δ. Слой-кривая — медиана по
32 документам (протокол E-34/E-31), пары с sentinel-концом исключаются (E-31).

Побочный выход: per-(doc,layer) S02_xi_exp (ℓ экспоненциального фита canonical
C_F) — заморожен в E-40.1 (l1).

Запуск: .venv/bin/python experiments/exp40_2_double_fit.py   (десятки минут).
"""
from __future__ import annotations

# BLAS/LAPACK-треды пинятся ДО импорта numpy/torch: [128,1024]-матрицы на 12
# тредах обрабатываются в ~30 раз дольше, чем на одном (замер 2026-11-03), а
# два таких процесса на 12 ядрах кладут машину. На числа треды не влияют.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from _e38_cache import load_layer  # noqa: E402
from exp40_common import DOMAINS, LAYERS, N_PER_DOMAIN, ROOT, S, dump_json, header, sentinel_mask, tag  # noqa: E402

DEL = np.arange(1, 33)
E31 = os.path.join(ROOT, "results", "exp31_correlation_function.json")
E39D_KAPPA_FLOOR = 0.0092          # E-39d κ_flat (Procrustes), L23-only anchor
E39D_KAPPA_BASELINE = 0.016         # E-38d/E-39a κ(L23)
B_BOOT = 1000

# ── формы ────────────────────────────────────────────────────────────────
def f_exp(x, A, ell, c0):
    return A * np.exp(-x / ell) + c0


def f_pow(x, A, alpha, c0):
    return A * x ** (-alpha) + c0


def f_str(x, A, ell, beta, c0):
    return A * np.exp(-np.power(x / ell, beta)) + c0


def f_powlog(x, A, alpha, c0, B, lam, phi):
    return A * x ** (-alpha) * (1 + B * np.cos(2 * np.pi * np.log(x) / np.log(lam) + phi)) + c0


FORMS = {
    "exp": dict(f=f_exp, k=3),
    "power": dict(f=f_pow, k=3),
    "stretched": dict(f=f_str, k=4),
    "power_logperiodic": dict(f=f_powlog, k=6),
}


def aicc_bic(y, yhat, k):
    n = len(y)
    rss = float(((y - yhat) ** 2).sum())
    ll = n * np.log(max(rss, 1e-300) / n)
    aic = ll + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
    return rss, aicc, ll + k * np.log(n)


def fit_one(name, x, y):
    """Детерминированный фит формы; powlog — фиксированный мультистарт."""
    spec = FORMS[name]
    c0g = max(float(y[-1]), 0.02)
    Ag = max(float(y[0]) - c0g, 0.05)
    starts = {
        "exp": [(Ag, 8.0, c0g)],
        "power": [(Ag, 0.5, c0g)],
        "stretched": [(Ag, 8.0, 1.0, c0g)],
        "power_logperiodic": [(Ag, al, c0g, Bv, lam, phi)
                              for lam in (1.5, 2.0, 3.0)
                              for phi in (0.0, np.pi / 2)
                              for al in (0.5, 1.0)
                              for Bv in (0.3,)],
    }[name]
    lo = {"exp": [0, 0.5, -0.5], "power": [0, 0.0, -0.5],
          "stretched": [0, 0.5, 0.2, -0.5],
          "power_logperiodic": [0, 0.0, -0.5, 0.0, 1.02, -np.pi]}[name]
    hi = {"exp": [10, 200, 0.9], "power": [10, 6, 0.9],
          "stretched": [10, 200, 4.0, 0.9],
          "power_logperiodic": [10, 6, 0.9, 1.0, 12.0, np.pi]}[name]
    best = None
    for p0 in starts:
        try:
            popt, _ = curve_fit(spec["f"], x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
            rss, aicc, bic = aicc_bic(y, spec["f"](x, *popt), spec["k"])
            if best is None or rss < best[0]:
                best = (rss, popt, aicc, bic)
        except Exception:  # noqa: BLE001 — стартовая точка неудачна, пробуем следующую
            continue
    if best is None:
        return None
    rss, popt, aicc, bic = best
    return {"rss": rss, "params": popt.tolist(), "AICc": float(aicc), "BIC": float(bic)}


def classify(best_name, params):
    if best_name == "exp":
        return "exp_like"
    if best_name == "stretched":
        return "exp_like" if params[2] >= 0.8 else "other"
    if best_name in ("power", "power_logperiodic"):
        return "power_like"
    return "other"


# ── кривые ───────────────────────────────────────────────────────────────
def vtop16(J, r=16):
    G = torch.bmm(J, J.transpose(1, 2))
    w, u = torch.linalg.eigh(G)
    wr = w[..., -r:]
    Ur = u[..., -r:]
    V = torch.bmm(J.transpose(1, 2), Ur)
    return (V / wr.unsqueeze(1).clamp_min(1e-30).sqrt()).contiguous()


def doc_curves(M, Vtop, sent, need_top=True):
    T = M.shape[0]
    Mf = M.flatten(1)
    nrm = Mf.norm(dim=1).numpy()
    Mf_np = Mf.numpy()
    C = np.full(len(DEL), np.nan)
    Ctop = np.full(len(DEL), np.nan)
    for di, d in enumerate(DEL):
        v = (~sent[:T - d] & ~sent[d:]).numpy()
        if not v.any():
            continue
        a = Mf_np[:T - d][v]
        b = Mf_np[d:][v]
        na = nrm[:T - d][v]
        nb = nrm[d:][v]
        C[di] = float(((a * b).sum(axis=1) / np.maximum(na * nb, 1e-30)).mean())
        if need_top:
            vsel = torch.from_numpy(v)
            # [n,r,D]@[n,D,r] = [n,r,r]: матрица главных углов между топ-r
            # правыми подпространствами — как в эталоне E-31
            # (exp31_correlation.principal_angle_max: `Va.transpose(-1,-2) @ Vb`).
            # Прежний порядок (Va @ Vbᵀ) давал [n,D,D] ранга r: сингулярных
            # значений min(D,D)=1024, из них 1008 нулей ⇒ acos = π/2 ⇒ C_top ≡ 0
            # (якорь E-31 не сошёлся бы), и в ~1000 раз дороже по времени.
            Msub = (Vtop[:T - d][vsel].transpose(-1, -2) @ Vtop[d:][vsel])
            sv = torch.linalg.svdvals(Msub).clamp(max=1.0 - 1e-12)
            Ctop[di] = float(torch.cos(torch.acos(sv).max(-1).values).mean())
    return C, Ctop


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    e31 = json.load(open(E31))
    per_layer: dict[str, dict] = {}
    doc_curves_store: dict[str, dict] = {}
    s02: dict[str, dict[str, float]] = {}
    raw_anchor_worst = 0.0
    for L in LAYERS:
        tl = time.time()
        ck = load_layer(L, want=("J", "P"))
        J, P = ck["J"], ck["P"]
        sent = sentinel_mask(L)
        Jgf = torch.empty_like(J)
        for di in range(J.shape[0]):
            for t in range(S):
                p = P[di, t].double()
                p = p / p.sum()
                Jgf[di, t] = (J[di, t].double() * p[:, None]
                              - p[:, None] * (p @ J[di, t].double())).float()
        docs = []
        for d_of, dom in enumerate(DOMAINS):
            for si in range(N_PER_DOMAIN):
                di = d_of * N_PER_DOMAIN + si   # плоский индекс документа
                # (кэш лежит домен-мажорно: (wiki,0..7),(math,0..7),…); раньше
                # сюда попадал индекс ДОМЕНА 0..3, и все 32 «документа» были
                # четырьмя первыми wiki-документами, повторёнными по 8 раз
                d = f"{dom}#{si}"
                Cra, Ca, tra, ta = [], [], [], []
                for which in ("raw", "gf"):
                    M = J[di] if which == "raw" else Jgf[di]
                    # raw-подпространство (дорогое) нужно только для якоря C_top — L23
                    need_top = (which == "gf") or (L == 23)
                    with torch.no_grad():
                        Vt = vtop16(M) if need_top else None
                    C, Ct = doc_curves(M, Vt, sent[di], need_top)
                    (Cra if which == "raw" else Ca).append(C)
                    if need_top:
                        (tra if which == "raw" else ta).append(Ct)
                # Одна кривая на вариант, а не список-обёртка: `Cra`/`Ca` набирались
                # append'ом по циклу `which`, поэтому каждый хранил [array[13]] ⇒
                # nanmedian(axis=0) давал [1,13], и fit_one падал на `float(y[-1])`
                # («only 0-dimensional arrays can be converted»), а json.dump уткнулся бы
                # в ndarray. Якорь E-31 при этом проходил (broadcast [1,13] vs [13]).
                blank = np.full(len(DEL), np.nan)
                docs.append({"d": d, "C_raw": Cra[0].tolist(),
                             "C_top_raw": tra[0].tolist() if tra else blank.tolist(),
                             "C_gf": Ca[0].tolist(),
                             "C_top_gf": ta[0].tolist() if ta else blank.tolist()})
                print(f"  L{L} {d} ({time.time() - tl:.0f} с)", flush=True)
        # якорь: raw-медиана = опубликованная C_med E-31
        med_raw = np.nanmedian([d["C_raw"] for d in docs], axis=0)
        devC = float(np.nanmax(np.abs(med_raw - np.array(e31[str(L)]["C_med"]))))
        if L == 23:
            med_raw_top = np.nanmedian([d["C_top_raw"] for d in docs], axis=0)
            devT = float(np.nanmax(np.abs(med_raw_top - np.array(e31[str(L)]["C_top_med"]))))
        else:
            devT = 0.0
        raw_anchor_worst = max(raw_anchor_worst, devC, devT)
        print(f"L{L}: raw-anchor dev C={devC:.1e} Ctop={devT:.1e}", flush=True)
        assert devC < 2e-4 and devT < 2e-4, f"L{L}: raw-якорь E-31 не совпал (конвейер разъехался)"

        med_gf = np.nanmedian([d["C_gf"] for d in docs], axis=0)
        med_gf_top = np.nanmedian([d["C_top_gf"] for d in docs], axis=0)
        obs = {"C_F": med_gf, "C_top16": med_gf_top}
        res_obs = {}
        for oname, y in obs.items():
            fits = {fn: fit_one(fn, DEL.astype(float), y) for fn in FORMS}
            fits_ok = {k: v for k, v in fits.items() if v is not None}
            order = sorted(fits_ok.items(), key=lambda kv: kv[1]["AICc"])
            best_name, best = order[0]
            gap = order[1]["AICc"] - best["AICc"] if len(order) > 1 else np.inf
            flag = None if gap >= 10 else "not_distinguishable_at_Δ_grid"
            # bootstrap: медианная кривая на ре выборке документов, рефит лучшей формы
            rng = np.random.default_rng(40)
            arr = np.array([d["C_gf" if oname == "C_F" else "C_top_gf"] for d in docs])
            pars, names_best = [], []
            for _ in range(B_BOOT):
                idx = rng.integers(0, arr.shape[0], arr.shape[0])
                yb = np.nanmedian(arr[idx], axis=0)
                fb = fit_one(best_name, DEL.astype(float), yb)
                if fb is not None:
                    pars.append(fb["params"])
                    names_best.append(best_name)
            pars = np.array(pars)
            ci = [[float(np.quantile(pars[:, j], 0.025)), float(np.quantile(pars[:, j], 0.975))]
                  for j in range(pars.shape[1])] if len(pars) else []
            res_obs[oname] = {
                "AICc": {k: v["AICc"] for k, v in fits.items()},
                "BIC": {k: v["BIC"] for k, v in fits.items()},
                "best_form": best_name, "params": best["params"],
                "aicc_gap_to_second": None if not np.isfinite(gap) else float(gap),
                "resolution_flag": flag,
                "bootstrap": {"B": B_BOOT, "n_ok": int(len(pars)),
                              "param_CI95": ci,
                              "refit_only_best_form": True,
                              "note": "рефит только лучшей формы (6p power_logperiodic "
                                      "на 32 точках в бутстрапе неустойчив)"},
                "classification": classify(best_name, best["params"]),
                "channel_label": "natural × canonical × "
                                 + ("full (L+R)" if oname == "C_F" else "top-16 right")
                                 + " × " + ("both" if oname == "C_F" else "right"),
            }
        # S02: per-doc ℓ экспоненты на canonical C_F
        s02[str(L)] = {}
        for d in docs:
            f = fit_one("exp", DEL.astype(float), d["C_gf"])
            s02[str(L)][d["d"]] = None if f is None else f["params"][1]
        doc_curves_store[str(L)] = {d["d"]: {"C_gf": d["C_gf"], "C_top_gf": d["C_top_gf"]}
                                    for d in docs}
        per_layer[str(L)] = res_obs
        del ck, J, Jgf
        print(f"L{L}: best C_F={res_obs['C_F']['best_form']} "
              f"({res_obs['C_F']['classification']}), C_top16="
              f"{res_obs['C_top16']['best_form']} ({res_obs['C_top16']['classification']}), "
              f"{time.time()-tl:.0f}s")

    # исход из 4 пререгистрированных
    clsF = [per_layer[L]["C_F"]["classification"] for L in map(str, LAYERS)]
    clsT = [per_layer[L]["C_top16"]["classification"] for L in map(str, LAYERS)]
    stretched_beta = {L: per_layer[L]["C_F"]["params"][2]
                      for L in map(str, LAYERS)
                      if per_layer[L]["C_F"]["best_form"] == "stretched"}
    if all(c == "exp_like" for c in clsF) and all(c == "exp_like" for c in clsT):
        outcome = "both_exp"
    elif all(c == "exp_like" for c in clsF) and all(c == "power_like" for c in clsT):
        outcome = "two_channel"
    elif all(c == "power_like" for c in clsF) and all(c == "power_like" for c in clsT):
        outcome = "both_power"
    else:
        outcome = "unstable"
    tags = [tag(slice="gf", p_slice="p_t", aggregation="median", window=w, side=sd,
                norm_type="energy", layer=None, rank=None)
            for w, sd in (("full", "mixed"), ("top", "right"))]
    rep = header("E-40.2", "docs/e40_todo.md §4",
                 {"exp31_correlation_function": E31,
                  "exp27b_jcache_L6": "/data/prism_kernels/exp27b/results/exp27b_jcache_L6.pt"},
                 channel_tags=tags,
                 extra={"per_layer": per_layer,
                        "outcome": outcome,
                        "stretched_beta_C_F": stretched_beta,
                        "raw_anchor_worst_dev": raw_anchor_worst,
                        "grid": {"deltas": DEL.tolist(), "is_dyadic": False,
                                 "note": "целочисленная сетка Δ=1..32 (Δ≥1); "
                                         "log-неравномерна — учтена в E-40.3"},
                        "per_doc_S02_xi_exp": s02,
                        "per_doc_canonical_curves": doc_curves_store,
                        "E39d_anchor_L23_only": {"kappa_floor": E39D_KAPPA_FLOOR,
                                                 "kappa_baseline": E39D_KAPPA_BASELINE,
                                                 "corr_f": 0.9992, "source": "E-39d",
                                                 "applies_to_layers": [23]},
                        "corr_f_applicable": False,
                        "deviations_from_spec": [
                            {"id": "D2_bootstrap_refit_best_only",
                             "description": "bootstrap-рефит только лучшей формы "
                                            "(power_logperiodic 6p×32 точки неустойчив)"},
                            {"id": "D2_grid_integers",
                             "description": "Δ целые 1..32 (кэш дискретен); dyadic — нет"}],
                        "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_2_double_fit.json")
    dump_json(rep, out)
    print(f"wrote {out}: outcome={outcome}, raw_anchor_worst={raw_anchor_worst:.1e}")
    return rep


if __name__ == "__main__":
    main()
