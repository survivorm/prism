#!/usr/bin/env python
"""E-50v2: СИСТЕМА ОТСЧЁТА — превращается ли «стохастичность» листа в закон движения.

Гипотеза (пользователь, 2026-11-06): в ПРАВИЛЬНОЙ системе отсчёта то, что выглядит шумом
(s(lag) насыщается, τ≈0.92), складывается в детерминированную динамику. Улика: голономия
0.028 рад/8 шагов при попарных прыжках 0.36 — джитр может быть почти чистой калибровкой.

ПРЕДРЕГИСТРИРОВАННЫЙ вердикт (объявлен до чтения shrink-чисел, при починке по ревью):
    shrink32 = median_drift32(real) / median_drift32(swap-null)   (transport-gauge)
    вердикт по bootstrap-CI отношения медиан: hi<0.3 => coherent (гипотеза жива);
    lo>0.7 => REFUTED; иначе => inconclusive. (CI-версия правилa, ок. раунда-2 ревью E50.)

Gauge: ИСТИННЫЙ ambient transport: Ṽ_t = Q_tᵀ V_t, Q = 𝓡_0…𝓡_{t-1}, 𝓡 — геодезическое
D×D-вращение в главных плоскостях (коэффициенты выведены и сверены с matrix_exp; точная
ранг-r геодезическая ⇒ дрейф ≡ 0 — в run_controls на r=1,4,16).

НУЛЬ (предрегистрирован, ДВА, вердикт по swap-null):
  * swap-local: случайные транспозиции соседних кадров (~25% пар, порядок уничтожен,
    попарные углы остаются МАЛЫМИ — тот же «масштаб скорости», что у реального пути;
    это ЧЕСТНЫЙ null);
  * shuffle-global: полная перестановка — ВЕРХНЯЯ оценка: углы пар >π/2 «складываются»
    (RP¹-фолдинг) и добавляют ложную диффузию, поэтому shrink по нему систематически
    занижен (в пользу гипотезы) — приводится как диагностика, не как вердикт.

ИЗВЕСТНЫЕ ОГРАНИЧЕНИЯ (ревью E50):
  * tokengauge УДАЛЁН: в npz нет id токенов; top1 = вероятность p.max(), не индекс
    (прошлая версия была тихо неверна — подтверждено фактом).
  * Sentinel/alive-срез (paper §3.2) к генеративным трассам неприменим (нет P128_RAW).
  * τ ПОСЛЕ транспорта тривиален мал (транспорт поглощает соседние шаги by design) —
    дискриминант = рост дрейфа, τ_trans приводится как диагностика, не как вердикт.
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
from exp49_leaf_info import load_model, smoothed_us, med, tau_triplets, nz  # noqa: E402

LAGS = (1, 2, 4, 8, 16, 32)
TMAX = 96
N_SHUF = 2
VERDICT_CUTS = (0.3, 0.7)


# ---------------------------------------------------------------- geometry core

def pullback_step(Va, Vb, M):
    """M := 𝓡(Va→Vb)ᵀ M — геодезический ambient pullback.
    exp(+θG)=I+s(pqᵀ−qpᵀ)+(c−1)(ppᵀ+qqᵀ), G=pqᵀ−qpᵀ, q_j=(Vb·Y_j−c_j p_j)/s_j.
    θ<1e-4: вклад O(θ) ничтожен; θ→π: no-op ОСОЗНАННО (метрика subspace-инвариантна,
    знак базиса не наблюдается)."""
    U, Cj, Wt = torch.linalg.svd(Va.T @ Vb, full_matrices=False)
    P = Va @ U
    c = Cj.clamp(-1 + 1e-12, 1 - 1e-12)
    th = torch.acos(c)
    s = torch.sin(th)
    live = (th > 1e-4) & (th < math.pi - 1e-3)
    Q = (Vb @ Wt.T - P * c) / s.where(live, torch.ones_like(s)).unsqueeze(0)
    s = torch.where(live, s, torch.zeros_like(s))
    cm1 = torch.where(live, c - 1.0, torch.zeros_like(c))
    A1, A2 = P.T @ M, Q.T @ M
    return M + P @ (cm1.unsqueeze(1) * A1 + s.unsqueeze(1) * A2) \
             + Q @ (-s.unsqueeze(1) * A1 + cm1.unsqueeze(1) * A2)


def transport_frames(Vs):
    """Ṽ_t = Qᵀ_t V_t с КОРРЕКТНОЙ (правой) композицией: накапливаем FORWARD
    T_t = R_{t-1}···R_0 (последняя ступень снаружи), Qᵀ_t = T_tᵀ = Rᵀ_0···Rᵀ_{t-1}.
    Для любой траектории, собранной из попарных геодезических, Ṽ_t ≡ V_0 точно
    (юнит-тест: random-walk + некоммутирующие 2-ступени). Левая композиция (bug v2)
    совпадает только при коммутирующих ступенях — потому прежние константно-плоские
    контроли её не ловили."""
    D = Vs[0].shape[0]
    T = torch.eye(D, dtype=torch.float64)          # forward-накопление
    out = [Vs[0].clone()]
    for t in range(1, len(Vs)):
        T = pullback_step(Vs[t], Vs[t - 1], T)     # = R(V_{t-1}→V_t) · T
        out.append(T.T @ Vs[t])
    return out


def drift_curve(Vs, lags=LAGS):
    return {L: [C.frame_kinematics(Vs[t], Vs[t + L])[0] for t in range(len(Vs) - L)]
            for L in lags}


def log_slope(dcurve, lags=(2, 4, 8, 16, 32)):
    xs, ys = [], []
    for L in lags:
        v = np.median(dcurve[L]) if L in dcurve and dcurve[L] else np.nan
        if np.isfinite(v) and v > 1e-9:
            xs.append(math.log(L)); ys.append(math.log(v))
    if len(xs) < 3:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def safe_med(x, nd=4):
    x = np.asarray([v for v in x if v is not None], float)
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), nd) if len(x) else None


def boot_ci(v, B=2000):
    v = np.asarray([x for x in v if x is not None], float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return [None, None]
    idx = np.random.default_rng(5).integers(0, len(v), (B, len(v)))
    m = np.median(v[idx], axis=1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


# ---------------------------------------------------------------- юнит-контроли

def run_controls(A, eps, D, T=96):
    """Контроли конвейера на СЫРЫХ рамках + измерение сжатия листа на белом u."""
    res = {}
    Dc = min(D, 400)
    for r in (1, 4, 16):                       # ТОЧНАЯ геодезическая ранги r
        g = torch.Generator().manual_seed(3)
        V0 = torch.linalg.qr(torch.randn(Dc, r, dtype=torch.float64, generator=g))[0]
        E = torch.randn(Dc, r, dtype=torch.float64, generator=g)
        E = E - V0 @ (V0.T @ E)                # q_j ⟂ span(V0) — иначе это НЕ геодезическая
        E = torch.linalg.qr(E)[0]
        th = torch.tensor([0.008 + 0.0004 * j for j in range(r)], dtype=torch.float64)
        geo = [V0 * torch.cos(th * t) + E * torch.sin(th * t) for t in range(T)]
        dg = drift_curve(transport_frames(geo), lags=(32,))
        res[f"geo_r{r}"] = {"drift32": safe_med(dg[32])}
    g2 = torch.Generator().manual_seed(4)      # random frames — насыщенный дрейф
    rnd = [torch.linalg.qr(torch.randn(Dc, 16, dtype=torch.float64, generator=g2))[0]
           for _ in range(T)]
    dr = drift_curve(transport_frames(rnd), lags=(32,))
    res["frames_white"] = {"drift32": safe_med(dr[32])}
    # (2b) ДЕТЕКТОР ПОРЯДКА КОМПОЗИЦИИ: random-walk линии — ступени не коммутируют;
    # при корректной (правой) композиции Ṽ≡V0 точно ⇒ drift32 ≈ 0; при прежней левой — >0.
    g3 = torch.Generator().manual_seed(5)
    uu = nz(torch.randn(Dc, dtype=torch.float64, generator=g3))
    rw = []
    for _ in range(T):
        rw.append(uu.unsqueeze(1).clone())
        uu = nz(uu + 0.05 * torch.randn(Dc, dtype=torch.float64, generator=g3))
    drw = drift_curve(transport_frames(rw), lags=(32,))
    res["rw_order"] = {"drift32": safe_med(drw[32])}
    Vs = []                                    # ИЗМЕРЕНИЕ: белый u сквозь лист
    gg = torch.Generator().manual_seed(7)
    for _ in range(T):
        u = nz(torch.randn(D, dtype=torch.float64, generator=gg))
        J, _, _ = C.J_analytic(A, u, eps)
        Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
        _, Ue = torch.linalg.eigh(Gm)
        v1 = J.T @ Ue[:, -1]
        Vs.append((v1 / v1.norm()).unsqueeze(1))
    dw = drift_curve(transport_frames(Vs), lags=(32,))
    res["leaf_white"] = {"drift32": safe_med(dw[32]),
                         "raw_s1": safe_med([C.frame_kinematics(Vs[t], Vs[t + 1])[0]
                                             for t in range(T - 1)])}
    bad = [f"geo_r{r} drift32={res[f'geo_r{r}']['drift32']}"
           for r in (1, 4, 16)
           if res[f"geo_r{r}"]["drift32"] is None or res[f"geo_r{r}"]["drift32"] > 1e-3]
    if not res["frames_white"]["drift32"] or res["frames_white"]["drift32"] < 1.0:
        bad.append("white drift не насыщен")
    if not res["rw_order"]["drift32"] or res["rw_order"]["drift32"] > 1e-4:
        bad.append(f"ПОРЯДОК КОМПОЗИЦИИ: rw drift32={res['rw_order']['drift32']} (ожидается ~0)")
    res["SELFTEST"] = "ok" if not bad else "ВАЛИТСЯ: " + "; ".join(bad)
    return res


# ---------------------------------------------------------------- данные

def doc_frames(u, hn, A, eps, W, r):
    """Кадры [D,r] сглаженного ряда; r=1 — ВЕРХНЯЯ ЛИНИЯ листа (отдельная линия из
    полного спектра, а не QR-колонка срезанного блока)."""
    us = smoothed_us(u, hn, W)
    top, lines = [], []
    for x in us:
        J, _, _ = C.J_analytic(A, x, eps)
        Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
        _, Ue = torch.linalg.eigh(Gm)
        v1 = J.T @ Ue[:, -1]
        lines.append((v1 / v1.norm()).unsqueeze(1))
        if r > 1:
            V = J.T @ Ue[:, -r:]
            top.append(torch.linalg.qr(V / V.norm(dim=0, keepdim=True))[0])
        else:
            top.append(lines[-1])
    return top, lines


def det_seed(s):
    return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:8], "little")


def boot_ratio_ci(real, swap, B=2000, seed=42):
    r = np.asarray([x for x in real if x is not None], float)
    s = np.asarray([x for x in swap if x is not None], float)
    if len(r) < 3 or len(s) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for k in range(B):
        out[k] = np.median(rng.choice(r, len(r), True)) / max(np.median(rng.choice(s, len(s), True)), 1e-12)
    return [round(float(np.quantile(out, 0.025)), 3), round(float(np.quantile(out, 0.975)), 3)]


def gauge_stats(Vs, seed_key):
    rng = np.random.default_rng(det_seed(seed_key))
    out = {}
    for name, reps in (("real", 1), ("swap", N_SHUF), ("shuf", N_SHUF)):
        acc = {L: [] for L in LAGS}
        taus, slopes = [], []
        for k in range(reps):
            if name == "real":
                W_ = Vs
            else:
                W_ = list(Vs)
                if name == "swap":       # местами меняем СОСЕДНИЕ кадры (без фолдинга)
                    idx = rng.permutation(len(W_) - 1)[: max(1, int(0.25 * len(W_)))]
                    used = set()
                    for i in sorted(int(x) for x in idx):
                        if i not in used and i + 1 not in used:
                            W_[i], W_[i + 1] = W_[i + 1], W_[i]
                            used.update({i, i + 1})
                else:
                    W_ = [Vs[i] for i in rng.permutation(len(Vs))]
            Tt = transport_frames(W_)
            dc = drift_curve(Tt)
            for L in LAGS:
                acc[L].append(safe_med(dc[L]))
            taus.append(safe_med(tau_triplets(Tt)))
            slopes.append(log_slope(dc))
        out[name] = {"drift": {L: safe_med(v) for L, v in acc.items()},
                     "tau_trans": safe_med(taus), "slope": safe_med(slopes, 3)}
    return out


def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0.0)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    A, U, eps, npz, jp = load_model(args)
    D = int(A.shape[1])
    out = {"model": args.model, "TMAX": TMAX, "N_SHUF": N_SHUF,
           "verdict_cuts": list(VERDICT_CUTS)}

    if not args.skip_controls:
        out["controls"] = run_controls(A, eps, D)
        print("CONTROLS:", json.dumps(out["controls"], indent=1), flush=True)
        if out["controls"]["SELFTEST"] != "ok":
            sys.exit("control selftest failed: " + out["controls"]["SELFTEST"])

    docs = json.load(open(os.path.join(root, jp)))["docs"]
    z = np.load(npz, allow_pickle=True)
    rep = {}
    for doc in docs:
        key = f"{doc['dom']}_{doc['si']}"
        f = z[key].item()
        u = torch.from_numpy(z[key + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(f["hnorm"], dtype=float)[:TMAX]
        for W in (1, 16):
            Vs, lines = doc_frames(u, hn, A, eps, W, r=16)
            g = gauge_stats(Vs, key + f"W{W}")
            g1 = gauge_stats(lines, key + f"W{W}r1")
            rep.setdefault(f"W{W}", []).append(dict(
                dom=doc["dom"], key=key,
                drift_real={str(L): g["real"]["drift"][L] for L in LAGS},
                drift_swap={str(L): g["swap"]["drift"][L] for L in LAGS},
                drift_shuf={str(L): g["shuf"]["drift"][L] for L in LAGS},
                slope_real=g["real"]["slope"], slope_swap=g["swap"]["slope"],
                tau_trans_real=g["real"]["tau_trans"], tau_trans_swap=g["swap"]["tau_trans"],
                shrink32=ratio(g["real"]["drift"][32], g["swap"]["drift"][32]),
                shrink32_glob=ratio(g["real"]["drift"][32], g["shuf"]["drift"][32]),
                shrink_by_lag={str(L): ratio(g["real"]["drift"][L], g["swap"]["drift"][L])
                               for L in LAGS[1:]},
                r1_drift32_real=g1["real"]["drift"][32],
                r1_drift32_swap=g1["swap"]["drift"][32]))
        print(key, "done", flush=True)

    summ = {}
    for tag, rows in rep.items():
        sh = [r["shrink32"] for r in rows if r["shrink32"] is not None]
        lsh = [math.log(r["shrink32"]) for r in rows if r["shrink32"] not in (None, 0.0)]
        mr = safe_med([r["drift_real"]["32"] for r in rows])
        ms = safe_med([r["drift_swap"]["32"] for r in rows])
        mg = safe_med([r["drift_shuf"]["32"] for r in rows])
        point = ratio(mr, ms)
        pci = boot_ratio_ci([r["drift_real"]["32"] for r in rows],
                            [r["drift_swap"]["32"] for r in rows])
        verdict = (None if pci[0] is None else
                   "coherent" if pci[1] is not None and pci[1] < VERDICT_CUTS[0] else
                   "REFUTED" if pci[0] > VERDICT_CUTS[1] else "inconclusive")
        summ[tag] = {
            "n": len(rows),
            "drift32_real": mr, "drift32_real_ci": boot_ci([r["drift_real"]["32"] for r in rows]),
            "drift32_swap": ms, "drift32_swap_ci": boot_ci([r["drift_swap"]["32"] for r in rows]),
            "drift32_shufglob": mg,
            "shrink32_median": [safe_med(sh), boot_ci(sh)],
            "shrink32_pooled_point": point, "shrink32_point_ci": pci, "verdict": verdict,
            "shrink32_glob_point": ratio(mr, mg),
            "shrink32_glob_ci": boot_ratio_ci([r["drift_real"]["32"] for r in rows],
                                              [r["drift_shuf"]["32"] for r in rows]),
            "log_shrink32": [safe_med(lsh, 3), boot_ci(lsh)],
            "slope_real": [safe_med([r["slope_real"] for r in rows], 3),
                           boot_ci([r["slope_real"] for r in rows])],
            "slope_swap": [safe_med([r["slope_swap"] for r in rows], 3),
                           boot_ci([r["slope_swap"] for r in rows])],
            "tau_trans_real": safe_med([r["tau_trans_real"] for r in rows], 3),
            "tau_trans_swap": safe_med([r["tau_trans_swap"] for r in rows], 3),
            "shrink_by_lag": {str(L): safe_med([r["shrink_by_lag"][str(L)] for r in rows])
                              for L in LAGS[1:]},
            "by_dom": {d: {
                "real": safe_med([r["drift_real"]["32"] for r in rows if r["dom"] == d]),
                "swap": safe_med([r["drift_swap"]["32"] for r in rows if r["dom"] == d]),
                "shrink": safe_med([r["shrink32"] for r in rows
                                    if r["dom"] == d and r["shrink32"] is not None]),
                "n": sum(1 for r in rows if r["dom"] == d)}
                for d in sorted({r["dom"] for r in rows})},
            "r1_shrink32": safe_med([ratio(r["r1_drift32_real"], r["r1_drift32_swap"])
                                     for r in rows]),
            "r1_diagnostic_only": True,
        }
        print(tag, "VERDICT:", summ[tag]["verdict"], "shrink32:",
              summ[tag]["shrink32_median"][0], summ[tag]["shrink32_median"][1], flush=True)
    out["summary"] = summ
    out["per_doc"] = rep

    path = os.path.join(root, "results", f"exp50_gauges_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
