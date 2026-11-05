#!/usr/bin/env python
"""E-51: ИНВАРИАНТЫ ВЫСШЕГО ПОРЯДКА — закон может быть нелинейным по приращениям.

Мотивация (вопрос пользователя 2026-11-06): все прежние метрики (расстояния, транспорт,
τ) были ЛИНЕЙНЫМИ по приращениям; первое моментное движение побелело. Если поток входит
мультипликативно/насыщающе, структура живёт во 2-м и выше порядках.

Инварианты (базис-инвариантны; линии v1 и 2-плоскости листа; гауз — геодезический
r=1-транспорт e50):
  I1 ORDER   — нематический порядок приращений в гаузе: g_k = <cos(δ_t, δ_{t+k})>,
               δ_t = упр. компонент v_{t+1} ⟂ v_t. Белое⇒0, баллистика⇒+1.
  I2 T-ODD   — ε_t = sign[det(AᵀB)·det(BᵀC)·det(CᵀA)] для троек 2-плоскостей:
               инвариантен к базисам; ⟨ε⟩=0 у ЛЮБОГО обратимого стационарного ряда —
               null следует из симметрии, не из бутстрэпа.
  I3 CR      — проективное кросс-отношение четвёрок ln[|c12 c34| / |c23 c14|]:
               инвариант и к базису, и к перепараметризации; «закон-форма» без скорости.
  I4 ELLIPSE — 2-й момент облака приращений: e = λ1/λ2 ковариации (окно w=8) и дрейф
               главной оси; вытянутое облако = скрытый поток.
  I5 SPECTR  — ряды σ₁..σ₄(J(t)): τ и автокорреляция (порядок 0: присоединённые
               инварианты; null для них — только перестановочный, отмечаем честно).

КОНТРОЛИ: точная геодезическая (I1≈+1, I3≡const) и белый ряд (I1≈0, I2≈0) — до данных.
NULL: swap-local (25% транспозиций соседей; гауз и все инварианты пересчитываются).
ВЕРДИКТ (предрегистрирован): инвариант выживает, если |real − swap| вне bootstrap-CI
swap И знак стабилен минимум в 3/4 доменов.
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
from exp49_leaf_info import load_model, smoothed_us, nz  # noqa: E402
from exp50_gauges import pullback_step, safe_med, boot_ci  # noqa: E402

TMAX = 96
N_SW = 2
SEED = 51


def nzr(x):
    n = x.norm()
    return x / n if n > 1e-12 else x


def gauge_along_lines(vv, mats):
    """Qᵀ-транспорт по линиям vv (ПРАВАЯ композиция, фикс раунда-2 ревью E50:
    накапливаем forward T=R_{t-1}···R_0, pullback = Tᵀ). Возвращает (gvv, gM, gdg),
    gdg_t = Tᵀ_t·δ_t — СЫРОЕ приращение, привезённое в начало координат."""
    D = vv[0].shape[0]
    T = torch.eye(D, dtype=torch.float64)
    gvv = [vv[0].clone()]
    gm = [None if mats is None else mats[0].clone()]
    gdg = []
    for t in range(1, len(vv)):
        v, w = vv[t - 1], vv[t]
        gdg.append(T.T @ nzr(w - v * (v @ w)))       # δ_{t-1} pullback'ом времени t-1
        T = pullback_step(w.unsqueeze(1), v.unsqueeze(1), T)   # forward
        gvv.append(T.T @ w)
        if mats is not None:
            gm.append(T.T @ mats[t])
    return gvv, gm, gdg


# ------------------------------------------------------------------ инварианты

def leaf_bases(x, A, eps):
    """v1 (линия) и 2-фрейм [v1, орт. в2] листа."""
    J, _, _ = C.J_analytic(A, x, eps)
    Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
    _, Ue = torch.linalg.eigh(Gm)
    v1 = nzr(J.T @ Ue[:, -1])
    v2 = nzr(J.T @ Ue[:, -2])
    v2 = nzr(v2 - v1 * (v1 @ v2))
    sig = torch.sqrt(torch.linalg.eigvalsh(Gm)[-4:].flip(0).clamp(min=0))
    return v1, torch.stack([v1, v2], 1), sig.tolist()


def increments(gdg):
    return gdg        # уже привезены в начало координат


def inv_I1(gdg, lags=(1, 2, 4, 8, 16)):
    d = gdg
    return {k: [float(d[t] @ d[t + k]) for t in range(len(d) - k)] for k in lags}


def inv_I2(gpl):
    out = []
    for t in range(len(gpl) - 2):
        A2, B2, C2 = gpl[t], gpl[t + 1], gpl[t + 2]
        s = float(torch.linalg.det(A2.T @ B2) * torch.linalg.det(B2.T @ C2)
                  * torch.linalg.det(C2.T @ A2))
        if abs(s) > 1e-9:
            out.append(float(np.sign(s)))
    return out


def inv_I3(gv):
    out = []
    for t in range(len(gv) - 3):
        a, b, c, d = (gv[t], gv[t + 1], gv[t + 2], gv[t + 3])
        c12, c23, c34, c14 = abs(a @ b), abs(b @ c), abs(c @ d), abs(a @ d)
        if c23 * c14 > 1e-6 and c12 * c34 > 1e-12:
            out.append(math.log(float((c12 * c34) / (c23 * c14))))
    return out


def inv_I4(dg, w=8):
    d = dg
    e, ax = [], []
    for t in range(len(d) - w):
        X = torch.stack(d[t:t + w], 1)
        Cc = (X @ X.T) / w
        ev, WV = torch.linalg.eigh(Cc)
        lam1 = ev[-1].clamp(min=1e-12); lam2 = ev[-2].clamp(min=1e-12)
        e.append(float(lam2 / lam1))                       # ∈ (0,1]: 1 = шар, <<1 = поток
        ax.append(WV[:, -1])
    drift = [float(1 - abs(ax[i] @ ax[0])) for i in range(len(ax))] if ax else []
    return e, drift


def inv_I5(specs, lags=(1, 4, 16)):
    S = np.stack(specs)
    out = {}
    for i in range(S.shape[1]):
        x = S[:, i]
        a = np.abs(np.diff(x))
        m = a[:-1] > 1e-9
        if m.sum() > 5:
            c = np.abs(x[2:] - x[:-2])
            out[f"tau_s{i+1}"] = round(float(np.median((a[:-1][m] + a[1:][m] - c[m]) / a[:-1][m])), 3)
        for L in lags:
            out[f"r{i+1}_L{L}"] = round(float(np.corrcoef(x[:-L], x[L:])[0, 1]), 3)
    return out


def boot_mean_ci(v, B=2000):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) < 3:
        return [None, None]
    idx = np.random.default_rng(SEED).integers(0, len(v), (B, len(v)))
    m = v[idx].mean(axis=1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


# ------------------------------------------------------------------ контроли

def run_controls(T=96, D=400):
    res = {}
    g = torch.Generator().manual_seed(3)
    u0 = nz(torch.randn(D, dtype=torch.float64, generator=g))
    w0 = torch.randn(D, dtype=torch.float64, generator=g); w0 = nz(w0 - u0 * (u0 @ w0))
    series = {"geo": [u0 * math.cos(0.012 * t) + w0 * math.sin(0.012 * t) for t in range(T)],
              "white": None}
    g2 = torch.Generator().manual_seed(4)
    series["white"] = [nz(torch.randn(D, dtype=torch.float64, generator=g2)) for _ in range(T)]
    for kind in ("geo", "white"):
        vv = series[kind]
        rngp = torch.Generator().manual_seed(9)
        pl = []
        for x in vv:
            zz = nz(torch.randn(D, dtype=torch.float64, generator=rngp))
            zz = nz(zz - x * (x @ zz))
            pl.append(torch.stack([x, zz], 1))
        gv, gpl, gdg = gauge_along_lines(vv, pl)
        # I2/I3 — инвариантны (детерминанты Грэма / CR) БЕЗ транспорта; на транспор-
        # тированных рамках они ВЫРОЖДАЮТСЯ (правая композиция: gv ≡ ±v0 точно —
        # тривиализация теоремы E50), поэтому считаются на исходных vv/pl.
        I1 = inv_I1(gdg); I2 = inv_I2(pl); I3 = inv_I3(vv); e, _ = inv_I4(gdg)
        res[kind] = {"I1_l1": safe_med(I1[1]), "I1_l8": safe_med(I1[8]),
                     "I2_mean": round(float(np.mean(I2)), 3) if I2 else None,
                     "I3_sd": round(float(np.std(I3)), 4) if len(I3) > 3 else None,
                     "I4_e": safe_med(e)}
    def gt(x, thr):   # strict-greater со строгим None-обращением (0.0 — валидное число!)
        return x is not None and x > thr
    ok = ((res["geo"]["I1_l1"] is not None and abs(res["geo"]["I1_l1"] - 1.0) < 0.1)
          and (res["geo"]["I3_sd"] is not None and res["geo"]["I3_sd"] < 0.02)
          and (res["geo"]["I4_e"] is not None and res["geo"]["I4_e"] < 0.3)
          and (res["white"]["I1_l8"] is not None and abs(res["white"]["I1_l8"]) < 0.07)
          and gt(res["white"]["I4_e"], 0.7)
          and gt(res["white"]["I3_sd"], 0.2)
          and (res["white"]["I2_mean"] is not None and abs(res["white"]["I2_mean"]) < 0.15))
    res["SELFTEST"] = "ok" if ok else "ВАЛИТСЯ"
    return res


# ------------------------------------------------------------------ данные

def swap_idx(T, rng):
    idx = np.arange(T)
    pos = rng.permutation(T - 1)[: max(1, int(0.25 * T))]
    used = set()
    for i in sorted(int(x) for x in pos):
        if i not in used and i + 1 not in used:
            idx[i], idx[i + 1] = idx[i + 1], idx[i]
            used |= {i, i + 1}
    return idx


def bundle(vv, pl):
    gv, gpl, gdg = gauge_along_lines(vv, pl)
    I1 = inv_I1(gdg); I2 = inv_I2(pl); I3 = inv_I3(vv); e, axd = inv_I4(gdg)
    return dict(I1_1=safe_med(I1[1]), I1_8=safe_med(I1[8]),
                I1_1_all=I1[1], I1_8_all=I1[8],
                I2=round(float(np.mean(I2)), 4) if I2 else None,
                I3_sd=round(float(np.std(I3)), 4) if len(I3) > 3 else None,
                I4_e=safe_med(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")

    out = {"model": args.model, "controls": run_controls()}
    print("CONTROLS:", json.dumps(out["controls"], indent=1), flush=True)
    if out["controls"]["SELFTEST"] != "ok":
        sys.exit("controls failed")
    if args.controls_only:
        return

    A, U, eps, npz, jp = load_model(args)
    docs = json.load(open(os.path.join(root, jp)))["docs"]
    z = np.load(npz, allow_pickle=True)
    rng = np.random.default_rng(SEED)
    rep = {}
    for doc in docs:
        key = f"{doc['dom']}_{doc['si']}"
        f = z[key].item()
        u = torch.from_numpy(z[key + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(f["hnorm"], dtype=float)[:TMAX]
        us = smoothed_us(u, hn, 1)
        v1s, pl, specs = [], [], []
        for x in us:
            v1, P, sig = leaf_bases(x, A, eps)
            v1s.append(v1); pl.append(P); specs.append(sig)
        b = bundle(v1s, pl)
        row = dict(dom=doc["dom"], key=key,
                   I1_1=b["I1_1"], I1_1_ci=boot_mean_ci(b["I1_1_all"]),
                   I1_8=b["I1_8"], I1_8_ci=boot_mean_ci(b["I1_8_all"]),
                   I2=b["I2"], I3_sd=b["I3_sd"], I4_e=b["I4_e"], I5=inv_I5(specs))
        for j in range(N_SW):
            idx = swap_idx(len(v1s), rng)
            bb = bundle([v1s[i] for i in idx], [pl[i] for i in idx])
            row[f"sw{j}"] = {k: bb[k] for k in ("I1_1", "I1_8", "I2", "I3_sd", "I4_e")}
        rep.setdefault(doc["dom"], []).append(row)
        print(key, "done", flush=True)

    def flat(field, swj=None):
        vals = []
        for dom, rows in rep.items():
            for r in rows:
                vals.append(r[field] if swj is None else r[f"sw{swj}"][field])
        return safe_med(vals)

    summ = {}
    for dom, rows in rep.items():
        summ[dom] = {}
        for f in ("I1_1", "I1_8", "I2", "I3_sd", "I4_e"):
            summ[dom][f] = safe_med([r[f] for r in rows])
            summ[dom]["sw_" + f] = safe_med([r[f"sw{j}"][f] for r in rows for j in range(N_SW)])
        summ[dom]["I2_signfrac"] = safe_med([abs((r["I2"] or 0)) for r in rows])
    allr = [r for rows in rep.values() for r in rows]
    summ["_all"] = {}
    for f in ("I1_1", "I1_8", "I2", "I3_sd", "I4_e"):
        summ["_all"][f] = [safe_med([r[f] for r in allr]),
                           boot_ci([r[f] for r in allr])]
        summ["_all"]["sw_" + f] = [safe_med([r[f"sw{j}"][f] for r in allr for j in range(N_SW)]),
                                   boot_ci([r[f"sw{j}"][f] for r in allr for j in range(N_SW)])]
    summ["_all"]["I5_median_tau"] = {f"tau_s{i+1}": safe_med([r["I5"].get(f"tau_s{i+1}") for r in allr])
                                     for i in range(4)}
    summ["_all"]["I5_corr_s1_lag16"] = safe_med([r["I5"].get("r1_L16") for r in allr])
    out["summary"] = summ
    out["per_doc"] = rep
    print(json.dumps(summ["_all"], indent=1), flush=True)
    path = os.path.join(root, "results", f"exp51_higher_order_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
