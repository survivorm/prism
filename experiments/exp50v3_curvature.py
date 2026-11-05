#!/usr/bin/env python
"""E-50v3: КРИВИЗНА (ГОЛОНОМИЯ) КАК ЕДИНСТВЕННЫЙ ИНВАРИАНТ гауза.

Концептуальный вывод после раунда-2 ревью E50: транспорт по попарным геодезическим
НАМЕРЕННО тривиализует всё попарное: span(Ṽ_t) ≡ span(V_0) ТОЧНО для ЛЮБОЙ
последовательности рамок (юнит-тесты: geo/white/random-walk drift ≡ 0), и даже O(r)
«спин» на открытом луче поглощается (для геодезической S_t ≡ I). Открытая траектория в
этом гаузе не содержит наблюдаемых. Ненаблюдаемо «дрейф в гаузе» (все прошлые drift/
shrink — артефакт левой композиции, архив SUPERSEDED_*).

Единственный gauge-инвариант — петлевая кривизна: замкнутая петля
V_t→V_{t+1}→…→V_{t+L}→V_t из попарных геодезических даёт остаточное O(r)-вращение
X_loop = V_tᵀ Ṽ_final; κ(t,L) = ||schur(X_loop)||₂. (Это ДРУГОЙ объект, чем
r×r-Procrustes-голономия §10: ambient-геодезический транспорт.)

ПРЕДРЕГИСТРИРОВАННЫЙ вердикт (до первых κ-чисел):
  «кривизна есть» если paired-CI95 по ДОКУМЕНТАМ (n_doc=64, парный: среднее κ по
  позициям внутри документа, перевыборка документов) нижней границей > 0 для L=16.
  Позиционный пул — только диагностика (в нём псевдорепликация, CI занижен ~√80).
ЗАМЕЧАНИЕ ОПРЕДЕЛЕНИЯ: замыкающая сторона петли — КРАТЧАЙшая геодезическая
(углы arccos|σ| свёрнуты в [0, π/2]); при L·θ̄ > π/2 петля = «путь + кратчайший
возврат», т.е. κ(L) — остаточное вращение геодезического полигона (законченная
область), а не «κ траектории». Это по-прежнему gauge-инвариант.
NULL: только swap (локальная перестановка 25% непересекающихся пар — конвенция
наследованная от e49/e50v2 для сопоставимости; разрушает петли с L≲8, сохраняет
маргиналы кадров) — консервативный null для порядковых утверждений.
ГЛАВНЫЙ ЛАГ L=16: фиксирован до чисел; середина диапазона, где swap-null ещё
сильно искажает петли, но окно не съедает половину T.
ПАРА: d16 — средние по позициям ВНУТРИ документа, затем парный bootstrap по
документам; на уровне окна пара НЕ физическая (swap меняет состав позиции) —
поэтому окно усреднено, а не сопряжено.
ЯСНОСТЬ: span-drift ≡ 0 (тривиальность ОТКРЫТОГО транспорта) НЕ противоречит
κ_white > 0: петля возвращается в свой span с остаточным вращением — это и есть κ.
D контролей (300) < D данных (2048) осознанно: контроли проверяют алгебру, не масштаб.
Контроли: точная геодезическая κ≡0 (<1e-5); white-рамки r=16: κ_white(L) —
эталон «гауссова» кривизны, real/swap сравниваются и с ним.
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
from exp49_leaf_info import load_model, smoothed_us  # noqa: E402
from exp50_gauges import pullback_step, safe_med, boot_ci, det_seed  # noqa: E402

TMAX = 96
N_SW = 2
LAGS = (4, 8, 16, 32, 64)
MAIN_LAG = 16


def loop_curvature(Vs, t, L):
    """κ(t,L): петля t→t+L→t транспортом попарных геодезических; остаток в span V_t."""
    V0 = Vs[t]
    r = V0.shape[1]
    T = torch.eye(V0.shape[0], dtype=torch.float64)
    for k in range(t, t + L):
        T = pullback_step(Vs[k + 1], Vs[k], T)        # forward R(V_k→V_{k+1})
    # замыкание: геодезическая V_{t+L} → V_t, т.е. forward R(V_{t+L}→V_t)
    T = pullback_step(V0, Vs[t + L], T)
    Vf = T.T @ V0                                     # привезённый кадр (span = span V0)
    Xl = V0.T @ Vf                                    # r×r, ортогонален с точностью fp
    U_, _, Vh_ = torch.linalg.svd(Xl)                 # полярный множитель (QR даёт signs)
    Q = U_ @ Vh_
    th, _, _ = C.schur_angles(Q)
    return float(torch.sqrt((th * th).sum()))


def kappa_curve(Vs, lags=LAGS):
    out = {}
    for L in lags:
        out[L] = [loop_curvature(Vs, t, L) for t in range(len(Vs) - L)]
    return out


def swap_order(Vs, rng):
    idx = np.arange(len(Vs))
    pos = rng.permutation(len(Vs) - 1)[: max(1, int(0.25 * len(Vs)))]
    used = set()
    for i in sorted(int(x) for x in pos):
        if i not in used and i + 1 not in used:
            idx[i], idx[i + 1] = idx[i + 1], idx[i]
            used |= {i, i + 1}
    return [Vs[i] for i in idx]


def synth_coh(D, r, T, seed):
    """Когерентная синтетика с ИСКРЕННЕЙ кривизной: V_{t+1} = exp(0.02·A_t)V_t,
    A_t — медленно прецессирующая фиксированная скоc-плоскость (шаги малые)."""
    g = torch.Generator().manual_seed(seed)
    V0 = torch.linalg.qr(torch.randn(D, r, dtype=torch.float64, generator=g))[0]
    a = torch.randn(D, r, dtype=torch.float64, generator=g)
    a = a - V0 @ (V0.T @ a)
    A0 = V0 @ a.T - a @ V0.T
    b = torch.randn(D, r, dtype=torch.float64, generator=g)
    b = b - V0 @ (V0.T @ b)
    A1 = V0 @ b.T - b @ V0.T
    Vs = []
    V = V0.clone()
    for t in range(T):
        Vs.append(torch.linalg.qr(V)[0])
        A = math.cos(0.05 * t) * A0 + math.sin(0.05 * t) * A1
        V = torch.matrix_exp(0.02 * A) @ V
    return Vs


def synth_white_matched(Vs_ref, D, r, T, seed):
    """Белые рамки с СОПОСТАВИМЫМ средним шагом (по референсу)."""
    step = float(np.mean([1.0 - float((Vs_ref[t] * Vs_ref[t + 1]).sum())
                          for t in range(len(Vs_ref) - 1)]))
    step = min(max(step, 1e-3), 0.3)
    g = torch.Generator().manual_seed(seed)
    V = torch.linalg.qr(torch.randn(D, r, dtype=torch.float64, generator=g))[0]
    Vs = []
    ang = math.sqrt(step)
    for _ in range(T):
        Vs.append(V.clone())
        z = torch.randn(D, r, dtype=torch.float64, generator=g)
        z = z - V @ (V.T @ z)
        z = torch.linalg.qr(z)[0]
        V = torch.linalg.qr(V * math.cos(ang) + z * math.sin(ang))[0]
    return Vs


def controls():
    res = {}
    D, r, T = 300, 16, 48
    g = torch.Generator().manual_seed(3)
    V0 = torch.linalg.qr(torch.randn(D, r, dtype=torch.float64, generator=g))[0]
    E = torch.randn(D, r, dtype=torch.float64, generator=g)
    E = E - V0 @ (V0.T @ E); E = torch.linalg.qr(E)[0]
    th = torch.tensor([0.008 + 0.0004 * j for j in range(r)], dtype=torch.float64)
    geo = [V0 * torch.cos(th * t) + E * torch.sin(th * t) for t in range(T)]
    res["geo_k16"] = safe_med(kappa_curve(geo, (16,))[16])
    g2 = torch.Generator().manual_seed(4)
    wfr = [torch.linalg.qr(torch.randn(D, r, dtype=torch.float64, generator=g2))[0]
           for _ in range(T)]
    res["white_k16"] = safe_med(kappa_curve(wfr, (16,))[16])
    # обоснование ЗНАКА вердикта: когерентная кривизна против белого с тем же шагом
    coh = synth_coh(D, r, T, 7)
    res["coh_k16"] = safe_med(kappa_curve(coh, (16,))[16])
    wm = synth_white_matched(coh, D, r, T, 8)
    res["white_matched_k16"] = safe_med(kappa_curve(wm, (16,))[16])
    ok = (res["geo_k16"] is not None and res["geo_k16"] < 1e-5
          and res["white_k16"] is not None and res["white_k16"] > 0.3
          and res["coh_k16"] is not None and res["white_matched_k16"] is not None
          and res["coh_k16"] > res["white_matched_k16"])
    res["SELFTEST"] = "ok" if ok else "ВАЛИТСЯ"
    return res


def save_controls(out_dir, model, ctr):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"exp50v3_controls_{model}.json")
    json.dump(ctr, open(p, "w"), indent=1, allow_nan=False)
    return p


def frames_of(u, hn, A, eps, W, r=16):
    us = smoothed_us(u, hn, W)
    out = []
    for x in us:
        J, _, _ = C.J_analytic(A, x, eps)
        Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
        _, Ue = torch.linalg.eigh(Gm)
        V = J.T @ Ue[:, -r:]
        out.append(torch.linalg.qr(V / V.norm(dim=0, keepdim=True))[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    out = {"model": args.model, "main_lag": MAIN_LAG, "controls": controls()}
    print("CONTROLS:", json.dumps(out["controls"], indent=1), flush=True)
    rdir = os.path.join(root, "results")
    save_controls(rdir, args.model, out["controls"])
    if out["controls"]["SELFTEST"] != "ok":
        sys.exit("controls failed")
    if args.controls_only:
        print("WROTE controls-only")
        return

    A, U, eps, npz, jp = load_model(args)
    docs = json.load(open(os.path.join(root, jp)))["docs"]
    z = np.load(npz, allow_pickle=True)
    rep = {}
    for doc in docs:
        key = f"{doc['dom']}_{doc['si']}"
        f = z[key].item()
        u = torch.from_numpy(z[key + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(f["hnorm"], dtype=float)[:TMAX]
        rng = np.random.default_rng(det_seed(key))
        for W in (1, 16):
            Vs = frames_of(u, hn, A, eps, W)
            kr = kappa_curve(Vs)
            row = {"dom": doc["dom"], "key": key,
                   "k_real": {str(L): safe_med(v) for L, v in kr.items()}}
            ks = {L: [] for L in LAGS}
            for _ in range(N_SW):
                for L, vv in kappa_curve(swap_order(Vs, rng)).items():
                    ks[L].append(vv)
            row["k_swap"] = {str(L): safe_med([x for sub in ks[L] for x in sub]) for L in LAGS}
            # ПАРА по ДОКУМЕНТУ: среднее по позициям (для парного bootstrap по n_doc)
            row["d16_mean"] = None
            row["d16"] = [(kr[MAIN_LAG][i] - float(np.mean([ks[MAIN_LAG][j][i]
                            for j in range(N_SW) if i < len(ks[MAIN_LAG][j])])))
                          for i in range(len(kr[MAIN_LAG]))]
            if row["d16"]:
                row["d16_mean"] = float(np.mean(row["d16"]))
            rep.setdefault(f"W{W}", []).append(row)
        print(key, "done", flush=True)

    summ = {}
    for tag, rows in rep.items():
        d16 = [x for r in rows for x in r["d16"]]                    # позиции — диагностикa
        dd = [r["d16_mean"] for r in rows if r["d16_mean"] is not None]  # документы — вердикт
        summ[tag] = {
            "k_real": {str(L): safe_med([r["k_real"][str(L)] for r in rows]) for L in LAGS},
            "k_swap": {str(L): safe_med([r["k_swap"][str(L)] for r in rows]) for L in LAGS},
            "paired_d16_by_doc": [safe_med(dd), boot_ci(dd)],
            "paired_d16_positions_DIAGNOSTIC": [safe_med(d16), boot_ci(d16)],
            "k_white_control": out["controls"].get("white_k16"),
            "k_coh_control": out["controls"].get("coh_k16"),
            "k_white_matched_control": out["controls"].get("white_matched_k16"),
            "n_docs": len(dd),
        }
        lo = summ[tag]["paired_d16_by_doc"][1][0]
        summ[tag]["verdict"] = ("curvature_structure" if (lo is not None and lo > 0)
                                else "none")
        print(tag, json.dumps(summ[tag]), flush=True)
    out["summary"] = summ
    out["per_doc"] = rep
    path = os.path.join(root, "results", f"exp50v3_curvature_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
