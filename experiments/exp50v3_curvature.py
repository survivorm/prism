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
  «кривизна есть» если paired-CI95 по ДОКУМЕНТАМ (n_doc=64 = 4 домена × 16,
  срез exp47b; парный: среднее κ по позициям внутри документа, перевыборка документов) нижней границей > 0 для L=16.
  Позиционный пул — только диагностика (в нём псевдорепликация, CI занижен ~√80).
ЗАМЕЧАНИЕ ОПРЕДЕЛЕНИЯ: замыкающая сторона петли — КРАТЧАЙшая геодезическая
(углы arccos|σ| свёрнуты в [0, π/2]); при L·θ̄ > π/2 петля = «путь + кратчайший
возврат», т.е. κ(L) — остаточное вращение геодезического полигона (законченная
область), а не «κ траектории». Это по-прежнему gauge-инвариант.
NULL И ЗНАК (пере-декларация ДО первых κ-чисел данных, основание — провал
процедурного контроля coh_d16_sign=-0.35 на синтетике, JSON controls):
в мелношаговом режиме (соседние кадры почти параллельны, как в данных) транспозиция
соседних кадров — почти TOждественное преобразование: swap-ноль почти не ломает петли
(L=16 переживает перестановку пары). Поэтому ГЛАВНЫЙ null — sur: суррогат с тем же
средним шагом и случайными направлениями (конструкция валидирована 9/9+generic:
κ_coh/κ_sur ≈ 1000×); прямая процедурная валидация знака: coh_d16_sur_sign > 0 и
white_d16_sur_sign ≈ 0 (те же коды, что к данным).
Swap — ВТОРИЧНЫЙ диагностик d16_swap (чувствительность к локальному порядку).
ТРИХОТОМИЧЕСКИЙ вердикт по paired-CI(Δ), Δ = κ_real − κ_sur (пере-декларация ДО
любых κ_real данных — пер-документные κ в логи не печатались ни разу):
  CI_ло > 0  => "curvature_structure": систематическая кривизна (усилие, force);
  CI_хи < 0  => "sub_geodesic": real глаже step-matched случайности — динамика
                почти геодезическая, кривизна на масштабе L пренебрежима
                (это НЕ провал — это режим «rigid-but-curved», κ→0 с геодезич.
                нулём; предсказание эволюции e47/e49);
  иначе      => "none_random_compatible": случайная динамика.
ОЖИДАНИЕ ДО ДАННЫХ (paper.md E-28, «rigid-but-curved» κ~0.01): на реальных данных
вердикт должен быть sub_geodesic — это УСПЕХ гипотезы (динамика почти геодезическая
в масштабе L относительно step-matched случайности), а НЕ провал; curv_structure
означал бы систематическое «усилие» (force), none — случайность.
ОГРАНИЧЕНИЯ: N_SUR=2 (дисперсия null входит в Δ вдвое легче, при пограничном CI
поднять); sur матчит СРЕДНИЙ шаг, не спектр приращений — следить mean_step_real
vs mean_step_sur и форму k_real(L)/k_sur(L); generic-контроль — прецессирующие
плотные skew (наст. кривизна rank-D; прежняя фиксированная A была 1p-группой=
winding-артефакт, из gates убрано, остаётся отчётом).
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
MIN_STEP = 5e-4
N_SW = 1      # swap — вторичный диагностик (null почти no-op, см. докстринг)
N_SUR = 2     # главный null: step-matched суррогат, 2 реплики
LAGS = (4, 8, 16, 32)   # L=64 убран: 30% времени ради хвоста кривой, не участвующего в вердикте
MAIN_LAG = 16
_DEV = {"xl": 0.0, "q": 0.0}          # надзор ортогональности (п.6-7 ревью)
_DEV_BY = {}                          # per-tag снапшоты (ревью-7 п.2)
_TAG = ["ctrl"]
_EYE16 = None


def loop_curvature(Vs, t, L):
    """κ(t,L): петля t→t+L→t транспортом попарных геодезических; остаток в span V_t.
    ОПРЕДЕЛЕН только для r ≥ 2 (r=1: O(1) дискретен, Xl=±1 ⇒ schur≡0)."""
    V0 = Vs[t]
    r = V0.shape[1]
    T = torch.eye(V0.shape[0], dtype=torch.float64)
    for k in range(t, t + L):
        T = pullback_step(Vs[k + 1], Vs[k], T)        # forward R(V_k→V_{k+1})
    # замыкание: геодезическая V_{t+L} → V_t, т.е. forward R(V_{t+L}→V_t)
    T = pullback_step(V0, Vs[t + L], T)
    Vf = T.T @ V0                                     # привезённый кадр (span = span V0)
    Xl = V0.T @ Vf                                    # r×r, ортогонален с точностью fp
    global _EYE16
    U_, _, Vh_ = torch.linalg.svd(Xl)                 # полярный множитель (QR даёт signs)
    Q = U_ @ Vh_
    r_ = Xl.shape[0]
    if _EYE16 is None or _EYE16.shape[0] != r_:
        _EYE16 = torch.eye(r_, dtype=torch.float64)
    dxl = float((Xl.T @ Xl - _EYE16).abs().max()); dq = float((Q.T @ Q - _EYE16).abs().max())
    _DEV["xl"] = max(_DEV["xl"], dxl); _DEV["q"] = max(_DEV["q"], dq)
    d = _DEV_BY.setdefault(_TAG[0], {"xl": 0.0, "q": 0.0})
    d["xl"] = max(d["xl"], dxl); d["q"] = max(d["q"], dq)
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


def synth_coh(D, r, T, seed, delta=0.02, generic=False):
    """Когерентная синтетика с ИСКРЕННЕЙ кривизной: V_{t+1} = exp(delta·A_t)V_t.
    generic=False: A_t — прецессирующая плоскость (rank ≤ 4 — режим УЖЕ, чем
    ранг-16 данных; знак вердикта вне этого режима — открытый пункт).
    generic=True: A — плотный случайный skew (rank D). κ определён только для r ≥ 2
    (O(1) дискретен: для r=1 Xl=±1, schur≡0)."""
    g = torch.Generator().manual_seed(seed)
    V0 = torch.linalg.qr(torch.randn(D, r, dtype=torch.float64, generator=g))[0]
    if generic:
        A0 = torch.randn(D, D, dtype=torch.float64, generator=g)
        A0 = A0 - A0.T
        A1 = torch.randn(D, D, dtype=torch.float64, generator=g)
        A1 = A1 - A1.T
        Vs = []
        V = V0.clone()
        for t in range(T):
            Vs.append(torch.linalg.qr(V)[0])
            A = math.cos(0.05 * t) * A0 + math.sin(0.05 * t) * A1   # ПРЕЦЕССИЯ: не 1p-группа
            V = torch.matrix_exp(delta * A) @ V                     # => наст. кривизна,
        return Vs                                                   # не winding-артефакт
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
        V = torch.matrix_exp(delta * A) @ V
    return Vs


def synth_white_matched(Vs_ref, D, r, T, seed):
    """Белые рамки с тем же RMS-углом шага (метрика frame_kinematics — та же, в
    которой κ), но не с тем же спектром приращений — матчинг масштаба, не
    распределения. (До фикса раунда-6 «шаг» считался как 1−tr(VᵀV') ≈ −15 и
    clamp'ился в константу — матчинг был мнимый.)"""
    step = mean_step(Vs_ref)                 # = ||Θ||_2 (та же метрика, что κ)
    step = min(max(step, MIN_STEP), 0.6)
    ang_step = step / math.sqrt(r)           # суррогат вращает все r плоскостей на ang:
    ang = ang_step                           # его ||Θ||_2 = √r·ang = step (точный матчинг)
    g = torch.Generator().manual_seed(seed)
    V = torch.linalg.qr(torch.randn(D, r, dtype=torch.float64, generator=g))[0]
    Vs = []
    for _ in range(T):
        Vs.append(V.clone())
        z = torch.randn(D, r, dtype=torch.float64, generator=g)
        z = z - V @ (V.T @ z)
        z = torch.linalg.qr(z)[0]
        V = torch.linalg.qr(V * math.cos(ang) + z * math.sin(ang))[0]
    return Vs


def _d16_of(Vs, rng, nrep=N_SW):
    k0 = kappa_curve(Vs, (MAIN_LAG,))[MAIN_LAG]
    ks = [kappa_curve(swap_order(Vs, rng), (MAIN_LAG,))[MAIN_LAG] for _ in range(nrep)]
    return float(np.mean([k0[i] - np.mean([ks[j][i] for j in range(nrep)
                                           if i < len(ks[j])]) for i in range(len(k0))]))


def _d16_sur_of(Vs, tag, nrep=N_SUR):
    """То же, что вердикт к данным: sur-null (step-matched), среднее по позициям."""
    D = Vs[0].shape[0]; r = Vs[0].shape[1]
    k0 = kappa_curve(Vs, (MAIN_LAG,))[MAIN_LAG]
    ks = [kappa_curve(synth_white_matched(Vs, D, r, len(Vs), det_seed(f"{tag}_s{j}")),
                      (MAIN_LAG,))[MAIN_LAG] for j in range(nrep)]
    return float(np.mean([k0[i] - np.mean([ks[j][i] for j in range(nrep)
                                           if i < len(ks[j])]) for i in range(len(k0))]))


def mean_step(Vs):
    """RMS главный угол между соседними кадрами (метрика frame_kinematics, та же,
    в которой κ). ПреЖняя формула 1−tr(VᵀV') давала ≈−(r−1) при r=16 (ревью-6 П0)."""
    return float(np.mean([C.frame_kinematics(Vs[t], Vs[t + 1])[0]
                          for t in range(len(Vs) - 1)]))


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
    # ЗНАК: робастность по 9 парам (3 сида × 3 шага), и на УРОВНЕ ПРОЦЕДУРЫ
    # (та же swap-процедура + d16, что и к данным)
    coh_by, wm_by, grid = [], [], []
    for s in (7, 17, 27):
        for dl in (0.01, 0.02, 0.04):
            c = synth_coh(D, r, T, s, dl)
            kv = safe_med(kappa_curve(c, (16,))[16])
            kw = safe_med(kappa_curve(synth_white_matched(c, D, r, T, s + 100), (16,))[16])
            coh_by.append(kv); wm_by.append(kw)
            grid.append([dl, s, kv, kw])
    cg = synth_coh(D, r, T, 7, 0.02, generic=True)
    kg = safe_med(kappa_curve(cg, (16,))[16])
    kgw = safe_med(kappa_curve(synth_white_matched(cg, D, r, T, 777), (16,))[16])
    res["generic_coh_k16"] = kg
    res["generic_wm_k16"] = kgw
    res["coh_wm_grid"] = [[dl, s, round(a, 4) if a is not None else None,
                           round(b, 4) if b is not None else None,
                           bool(a is not None and b is not None and a > b)]
                          for dl, s, a, b in grid]
    res["coh_k16_med"] = safe_med(coh_by)
    res["white_matched_k16_med"] = safe_med(wm_by)
    # δ=0.01 — ДИАГНОСТИКА (там κ_coh ~ δ² может не выигрывать у шума), требуем δ≥0.02
    by_delta = {}
    for i, (dl, s, a, b) in enumerate(grid):
        by_delta.setdefault(dl, []).append(a is not None and b is not None and a > b)
    res["sign_ok_by_delta"] = {str(k): all(v) for k, v in by_delta.items()}
    sign_ok = all(by_delta[0.02]) and all(by_delta[0.04])
    rng_s = np.random.default_rng(det_seed("sign_ctrl"))
    coh0 = synth_coh(D, r, T, 7, 0.02)
    res["coh_d16_sign"] = round(_d16_of(coh0, rng_s), 4)              # swap-процедура (диагн.)
    res["white_matched_d16_sign"] = round(_d16_of(synth_white_matched(coh0, D, r, T, 8),
                                                  rng_s), 4)
    # ПРЯМАЯ процедурная валидация ГЛАВНОГО нуля (та же sur-процедура, что к данным)
    res["coh_d16_sur_sign"] = round(_d16_sur_of(coh0, "coh0"), 4)
    res["white_d16_sur_sign"] = round(_d16_sur_of(synth_white_matched(coh0, D, r, T, 8),
                                                  "wm0"), 4)
    ok = (res["geo_k16"] is not None and res["geo_k16"] < 1e-5
          and res["white_k16"] is not None and res["white_k16"] > 0.3
          and sign_ok
          and res["coh_d16_sur_sign"] > 0.01
          and abs(res["white_d16_sur_sign"]) < 0.05
          and res["white_matched_d16_sign"] is not None
          and abs(res["white_matched_d16_sign"]) < 0.05)
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
        rng = None
        for W in (1, 16):
            _TAG[0] = f"W{W}"                                      # per-tag DEV
            rng = np.random.default_rng(det_seed(f"{key}_W{W}"))   # независимость W-проходов
            Vs = frames_of(u, hn, A, eps, W)
            kr = kappa_curve(Vs)
            D_ = Vs[0].shape[0]; r_ = Vs[0].shape[1]
            row = {"dom": doc["dom"], "key": key,
                   "k_real": {str(L): safe_med(v) for L, v in kr.items()}}
            ks = {L: [] for L in LAGS}
            for _ in range(N_SW):
                for L, vv in kappa_curve(swap_order(Vs, rng)).items():
                    ks[L].append(vv)
            row["k_swap"] = {str(L): safe_med([x for sub in ks[L] for x in sub]) for L in LAGS}
            ku = {L: [] for L in LAGS}                      # ГЛАВНЫЙ null: sur
            steps_sur = []
            for j in range(N_SUR):
                sur = synth_white_matched(Vs, D_, r_, len(Vs), det_seed(f"{key}_W{W}_sur{j}"))
                steps_sur.append(mean_step(sur))
                for L, vv in kappa_curve(sur).items():
                    ku[L].append(vv)
            row["k_sur"] = {str(L): safe_med([x for sub in ku[L] for x in sub]) for L in LAGS}
            row["mean_step_real"] = round(mean_step(Vs), 5)   # контроль матчинга null'а
            row["mean_step_sur"] = round(float(np.mean(steps_sur)), 5)
            row["step_clamped"] = bool(row["mean_step_real"] < MIN_STEP)  # П3 ревью-7
            # ПАРА по ДОКУМЕНТУ: среднее по позициям (для парного bootstrap по n_doc)
            row["d16_mean"] = None
            row["d16_swap_mean"] = None
            def _mdiff(knull):
                if not knull[MAIN_LAG]:
                    return None
                return float(np.mean([kr[MAIN_LAG][i]
                                      - float(np.mean([knull[MAIN_LAG][j][i]
                                                       for j in range(len(knull[MAIN_LAG]))
                                                       if i < len(knull[MAIN_LAG][j])]))
                                      for i in range(len(kr[MAIN_LAG]))]))
            dsv = _mdiff(ks)
            row["d16_swap_mean"] = dsv
            row["d16"] = _mdiff(ku)                          # verdit-параметр: real − sur
            row["d16_mean"] = row["d16"]
            rep.setdefault(f"W{W}", []).append(row)
        print(key, "done", flush=True)

    summ = {}
    for tag, rows in rep.items():
        dd = [r["d16_mean"] for r in rows if r["d16_mean"] is not None]  # документы — вердикт
        summ[tag] = {
            "k_real": {str(L): safe_med([r["k_real"][str(L)] for r in rows]) for L in LAGS},
            "k_swap": {str(L): safe_med([r["k_swap"][str(L)] for r in rows]) for L in LAGS},
            "k_sur": {str(L): safe_med([r["k_sur"][str(L)] for r in rows]) for L in LAGS},
            "paired_d16_by_doc": [safe_med(dd), boot_ci(dd)],          # real − sur (ГЛАВНЫЙ)
            "paired_swap_by_doc_DIAGNOSTIC":
                [safe_med([r["d16_swap_mean"] for r in rows
                           if r["d16_swap_mean"] is not None]),
                 boot_ci([r["d16_swap_mean"] for r in rows
                          if r["d16_swap_mean"] is not None])],
            "mean_step_real": safe_med([r["mean_step_real"] for r in rows], 5),
            "mean_step_sur": safe_med([r["mean_step_sur"] for r in rows], 5),
            "k_white_control": out["controls"].get("white_k16"),
            "k_coh_control": out["controls"].get("coh_k16_med"),
            "k_white_matched_control": out["controls"].get("white_matched_k16_med"),
            "coh_d16_sign_control": out["controls"].get("coh_d16_sign"),
            "max_nonorthogonality": _DEV_BY.get(tag, dict(_DEV)),  # per-tag (ревью-7 п.2)
            "n_docs": len(dd),
        }
        lo, hi = summ[tag]["paired_d16_by_doc"][1]
        mean_d = summ[tag]["paired_d16_by_doc"][0]
        summ[tag]["verdict"] = ("inconclusive" if (lo is None or mean_d is None) else
                                "curvature_structure" if lo > 0 else
                                "sub_geodesic" if hi < 0 else
                                "none_random_compatible")
        print(tag, json.dumps(summ[tag]), flush=True)
    out["summary"] = summ
    out["per_doc"] = rep
    path = os.path.join(root, "results", f"exp50v3_curvature_{args.model}.json")
    json.dump(out, open(path, "w"), indent=1, allow_nan=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
