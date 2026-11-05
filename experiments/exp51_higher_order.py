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

КОНТРОЛИ (раунд-3 ревью П0-1): геодезическая с COHERENT 2-фреймом (данные дают
детерминированный v2 — контроль обязан совпадать по структуре): I1≈+1, I3≡const,
I2=+1 точно (det ≡ +1); белый ряд с INCOHERENT pl: I1≈0, I2≈0, I3_sd>0.2.
Контроль скобкует I2 с двух сторон [0,1].
NULL (трёхступенчатый): sur = step-matched белая прямая (сильный: тот же шаг,
нулевая динамика), sw = 25% транспозиций соседей (локальный порядок), sh = полный
перемешивание (весь порядок); реплики j=0..1, paired по документу.
ВЕРДИКТ (предрегистрирован v3, 2026-11-06, ДО просмотра данных; по каждому
инварианту: значимо = CI разности не содержит 0 И знак согласован в >=3/4
доменов):
  "directional+ordering"    — sur И sw значимы (не step-случайно, порядок важен);
  "directional(order-free)" — только sur (гладкая структура вне локального порядка);
  "local-ordering"          — только sw;
  "long-range-only"         — только sh;
  "domain-dependent"        — overall значимо, но <3/4 доменов (не подтверждён);
  "null-compatible"         — ни против одного нуля не значимо.
Для I5 — только vs sh (у sur нет σ-ряда). dom_need=⌈0.75·ndom⌉; при ndom=4 → 3;
при ndom<4 (подвыборка) критерий вырождается — вердикт на подвыборке не выносится.
CI `I*_ci_DIAGNOSTIC` по позициям — диагностика (позиции внутри дока не iid);
валидные CI — по документам в summary._all.
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
from exp49_leaf_info import load_model, smoothed_us, nz  # noqa: E402
from exp50v3_curvature import pullback_step, safe_med, boot_ci  # noqa: E402
import exp50v3_curvature as E50  # noqa: E402  (synth_white_matched — sur-нуль)

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
    # _DEV (ревью E51 §6): дрейф ортогональности накопленного T — O(D³) один раз
    # на документ; численный шум не должен имитировать нетривиальность.
    dev = float((T.T @ T - torch.eye(D, dtype=torch.float64)).abs().max())
    return gvv, gm, gdg, dev


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

def run_controls(T=256, D=400):
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
        pl = []
        if kind == "geo":
            # П0-1 (ревью-3): COHERENT pl — как в данных (v2 детерминирован), а не
            # случайный zz_t: вращающийся 2-фрейм даёт det(pl_t^T pl_{t+1}) ≡ +1.
            for t in range(T):
                a = 0.012 * t
                pl.append(torch.stack([u0 * math.cos(a) + w0 * math.sin(a),
                                       -u0 * math.sin(a) + w0 * math.cos(a)], 1))
        else:
            # INCOHERENT pl — белая несвязная ориентация: null I2 ≈ 0.
            rngp = torch.Generator().manual_seed(9)
            for x in vv:
                zz = nz(torch.randn(D, dtype=torch.float64, generator=rngp))
                zz = nz(zz - x * (x @ zz))
                pl.append(torch.stack([x, zz], 1))
        gv, gpl, gdg, gdev = gauge_along_lines(vv, pl)
        # I2/I3 — инвариантны (детерминанты Грэма / CR) БЕЗ транспорта; на транспор-
        # тированных рамках они ВЫРОЖДАЮТСЯ (правая композиция: gv ≡ ±v0 точно —
        # тривиализация теоремы E50), поэтому считаются на исходных vv/pl.
        I1 = inv_I1(gdg); I2 = inv_I2(pl); I3 = inv_I3(vv); e, _ = inv_I4(gdg)
        res[kind] = {"I1_l1": safe_med(I1[1]), "I1_l8": safe_med(I1[8]),
                     "I2_mean": round(float(np.mean(I2)), 3) if I2 else None,
                     "I3_sd": round(float(np.std(I3)), 4) if len(I3) > 3 else None,
                     "I4_e": safe_med(e), "gauge_dev": round(gdev, 12)}
    def gt(x, thr):   # strict-greater со строгим None-обращением (0.0 — валидное число!)
        return x is not None and x > thr
    ok = ((res["geo"]["I1_l1"] is not None and abs(res["geo"]["I1_l1"] - 1.0) < 0.1)
          and (res["geo"]["I2_mean"] is not None and abs(res["geo"]["I2_mean"] - 1.0) < 0.1)
          and (res["geo"]["I3_sd"] is not None and res["geo"]["I3_sd"] < 0.02)
          and (res["geo"]["I4_e"] is not None and res["geo"]["I4_e"] < 0.3)
          and (res["white"]["I1_l8"] is not None and abs(res["white"]["I1_l8"]) < 0.07)
          and gt(res["white"]["I4_e"], 0.7)
          and gt(res["white"]["I3_sd"], 0.2)
          and (res["white"]["I2_mean"] is not None and abs(res["white"]["I2_mean"]) < 0.15)
          and max(res["geo"]["gauge_dev"], res["white"]["gauge_dev"]) < 1e-6)
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


def surrogate_lines(v1s, key, j):
    """Step-matched белая прямая (тот же RMS шаг, что у реальных линий v1s) +
    случайный орт-комплемент: нуль «та же скорость, нулевая динамика» (первичный
    нуль E50v3). Ревью E51 §2/§9: swap-25% слаб для I1 на малых лагах и для
    порядка I2/I3 — нужен сильный нуль. Seed — детерминированный от (key, j)
    (ревью-3 П1-2): результат не зависит от порядка вызовов нулей."""
    Vs = torch.stack([v.detach() for v in v1s])[:, :, None]  # [n,D,r=1], torch
    D = Vs.shape[1]
    h = hashlib.sha256(f"{key}|sur{j}|".encode()).digest()
    vs = E50.synth_white_matched(Vs, D, 1, len(v1s),
                                 int.from_bytes(h[:8], "little") % (1 << 62))
    out_v, out_p = [], []
    rg = torch.Generator().manual_seed(int.from_bytes(h[8:16], "little") % (1 << 62))
    for x in vs:
        xt = x.reshape(-1).to(torch.float64)
        zz = torch.randn(D, dtype=torch.float64, generator=rg)
        zz = nz(zz - xt * (xt @ zz))
        out_v.append(xt)
        out_p.append(torch.stack([xt, zz], 1))
    return out_v, out_p


def _stack1(vs):
    return torch.stack([v.detach().reshape(-1) for v in vs])[:, :, None]   # [n,D,1]


def bundle(vv, pl, specs=None):
    gv, gpl, gdg, gdev = gauge_along_lines(vv, pl)
    I1 = inv_I1(gdg); I2 = inv_I2(pl); I3 = inv_I3(vv); e, _dr = inv_I4(gdg)
    return dict(I1_1=safe_med(I1[1]), I1_8=safe_med(I1[8]),
                I1_1_all=I1[1], I1_8_all=I1[8],
                I2=round(float(np.mean(I2)), 4) if I2 else None,
                I3_sd=round(float(np.std(I3)), 4) if len(I3) > 3 else None,
                I4_e=safe_med(e), gauge_dev=round(gdev, 12),
                I5=(inv_I5(specs) if specs is not None else None))


NULL_KEYS = ("I1_1", "I1_8", "I2", "I3_sd", "I4_e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")

    out = {"model": args.model, "controls": run_controls(), "rank_used": 2,
           "note": "I2/I3 на СЫРЫХ vv/pl (basis-invariant, без транспорта); rank=2 "
                   "— проекция листа (главные мода/плоскость), не top-16. Нули: "
                   "sur=step-matched белая прямая (сильный), sw=25% транспозиции "
                   "(локальный порядок), sh=полный shuffle (весь порядок); paired "
                   "по документам, CI не пересекает 0 => значимо"}
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
    cache_p = os.path.join(root, "results", f"exp51_cache_{args.model}.jsonl")
    done_keys = set()
    if os.path.exists(cache_p):
        with open(cache_p) as fh:
            for ln in fh:
                r0 = json.loads(ln)
                rep.setdefault(r0["dom"], []).append(r0)
                done_keys.add(r0["key"])
        print(f"[resume] кэш: {len(done_keys)} доков", flush=True)
    cache_fh = open(cache_p, "a")
    for doc in docs:
        key = f"{doc['dom']}_{doc['si']}"
        if key in done_keys:
            continue
        f = z[key].item()
        u = torch.from_numpy(z[key + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(f["hnorm"], dtype=float)[:TMAX]
        us = smoothed_us(u, hn, 1)
        v1s, pl, specs = [], [], []
        for x in us:
            v1, P, sig = leaf_bases(x, A, eps)
            v1s.append(v1); pl.append(P); specs.append(sig)
        b = bundle(v1s, pl, specs)
        row = dict(dom=doc["dom"], key=key,
                   I1_1=b["I1_1"], I1_1_ci_DIAGNOSTIC=boot_mean_ci(b["I1_1_all"]),
                   I1_8=b["I1_8"], I1_8_ci_DIAGNOSTIC=boot_mean_ci(b["I1_8_all"]),
                   I2=b["I2"], I3_sd=b["I3_sd"], I4_e=b["I4_e"], I5=b["I5"],
                   gauge_dev=b["gauge_dev"])
        S2 = np.asarray(specs, dtype=float)   # П1-1: rank-deficiency v2
        if S2.ndim == 2 and S2.shape[1] > 1:
            row["frac_rankdef"] = round(float(np.mean(S2[:, 1] < 0.01 * S2[:, 0])), 4)
            row["gap_s2s1"] = safe_med((S2[:, 1] / np.maximum(S2[:, 0], 1e-12)).tolist())
            row["gap_s3s2"] = (safe_med((S2[:, 2] / np.maximum(S2[:, 1], 1e-12)).tolist())
                               if S2.shape[1] > 2 else None)
        n_ = len(v1s)
        row["step_real"] = round(E50.mean_step(_stack1(v1s)), 5)   # matching audit
        if (row["step_real"] or 0) < 6e-4:
            row["sur_step_clamped"] = True   # sur шаг упрётся в MIN_STEP=5e-4 (E50v3)
        for j in range(N_SW):
            idx = swap_idx(n_, rng)
            bb = bundle([v1s[i] for i in idx], [pl[i] for i in idx],
                        [specs[i] for i in idx])
            row[f"sw{j}"] = {k: bb[k] for k in NULL_KEYS}
            row[f"sw{j}"]["I5"] = bb["I5"]
            row[f"sw{j}"]["step"] = round(E50.mean_step(_stack1([v1s[i] for i in idx])), 5)
            sh = rng.permutation(n_)
            bb = bundle([v1s[i] for i in sh], [pl[i] for i in sh],
                        [specs[i] for i in sh])
            row[f"sh{j}"] = {k: bb[k] for k in NULL_KEYS}
            row[f"sh{j}"]["I5"] = bb["I5"]
            row[f"sh{j}"]["step"] = round(E50.mean_step(_stack1([v1s[i] for i in sh])), 5)
            sv, sp = surrogate_lines(v1s, key, j)
            bb = bundle(sv, sp)
            row[f"sur{j}"] = {k: bb[k] for k in NULL_KEYS}
            row[f"sur{j}"]["step"] = round(E50.mean_step(_stack1(sv)), 5)
        rep.setdefault(doc["dom"], []).append(row)
        cache_fh.write(json.dumps(row, allow_nan=False) + "\n"); cache_fh.flush()
        print(key, "done", flush=True)

    FIELDS = ("I1_1", "I1_8", "I2", "I3_sd", "I4_e")
    NULLS = ("sur", "sw", "sh")

    def paired(rows, f, n):
        """paired по документу: реальный минус среднее по репликам нулля j."""
        d = []
        for r in rows:
            if r.get(f) is None:
                continue
            vals = [r[f"{n}{j}"][f] for j in range(N_SW)
                    if r.get(f"{n}{j}", {}).get(f) is not None]
            if vals:
                d.append(r[f] - float(np.mean(vals)))
        return d

    summ = {}
    for dom, rows in rep.items():
        summ[dom] = {f: safe_med([r[f] for r in rows]) for f in FIELDS}
        summ[dom]["I5_tau_s1"] = safe_med([r["I5"].get("tau_s1") for r in rows])
        summ[dom]["frac_rankdef"] = safe_med([r.get("frac_rankdef") for r in rows])
    allr = [r for rows in rep.values() for r in rows]
    s = summ["_all"] = {}
    s["max_gauge_dev"] = max((r["gauge_dev"] for r in allr), default=None)
    s["frac_rankdef_median"] = safe_med([r.get("frac_rankdef") for r in allr])
    s["I2_rankdef_caveat"] = bool((s["frac_rankdef_median"] or 0) > 0.2)
    for f in FIELDS:
        s[f] = [safe_med([r[f] for r in allr]), boot_ci([r[f] for r in allr])]
    # Вердикт v3 (предрегистр. в docstring): overall CI + >=3/4 доменов того же знака
    ndom = len(rep)
    dom_need = max(1, math.ceil(0.75 * ndom))
    verdict = {}
    for f in FIELDS:
        sig, domc = {}, {}
        for n in NULLS:
            d = [x for rows in rep.values() for x in paired(rows, f, n)]
            # point=медиана ⇒ CI медианы (boot_ci), согласованные estimand (ревью §1)
            s[f"vs_{n}_" + f] = [safe_med(d), boot_ci(d)]
            lo, hi = s[f"vs_{n}_" + f][1]
            sig[n] = lo is not None and lo * hi > 0
            dc = 0
            for rows in rep.values():
                dlo, dhi = boot_ci(paired(rows, f, n))   # ПАРА, не [1] (крах qwen-run 05.11)
                if dlo is not None and dlo * dhi > 0:
                    dc += 1
            domc[n] = dc
            s[f"doms_{n}_" + f] = dc
        rob = {n: sig[n] and domc[n] >= dom_need for n in NULLS}
        s[f"sig_{f}"] = {n: bool(sig[n]) for n in NULLS}
        s[f"rob_{f}"] = {n: bool(rob[n]) for n in NULLS}
        s[f"notrobust_{f}"] = bool(any(sig.values()) and not any(rob.values()))
        verdict[f] = ("directional+ordering" if rob["sur"] and rob["sw"] else
                      "directional(order-free)" if rob["sur"] else
                      "local-ordering" if rob["sw"] else
                      "long-range-only" if rob["sh"] else
                      "domain-dependent" if any(sig.values()) else "null-compatible")
    s["dom_need"] = dom_need
    for key in ("tau_s1", "r1_L16"):   # I5: paired против полного shuffle (у sur нет specs)
        d = []
        for r in allr:
            a = r["I5"].get(key)
            bs = [r[f"sh{j}"]["I5"].get(key) for j in range(N_SW)]
            if a is not None and all(b is not None for b in bs):
                d.append(a - float(np.mean(bs)))
        s[f"I5_vs_sh_{key}"] = [safe_med(d), boot_ci(d)]
        lo, hi = s[f"I5_vs_sh_{key}"][1]
        s[f"I5_vs_sh_{key}_sig"] = lo is not None and lo * hi > 0
    s["verdict"] = verdict
    out["summary"] = summ
    out["per_doc"] = rep
    print(json.dumps(summ["_all"], indent=1), flush=True)
    path = os.path.join(root, "results", f"exp51_higher_order_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
