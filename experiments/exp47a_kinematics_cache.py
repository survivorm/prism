#!/usr/bin/env python
"""E-47a (P0): кинематика листа L23 на кэше exp27b (teacher-forcing, 4096 позиций).

Полная постановка, математика и ПРЕРЕГИСТРИРОВАННЫЕ критерии — docs/leaf_dynamics.md
(§5.1 протокол, §6 критерии K0.1–K0.4; написаны до чисел). Здесь — механика:

  вход   : exp27b jcache L23 (JR, H, U128, ...), дефолтный каталог — тот же, что у E-28
           (/data/prism_kernels/exp27b/results: кэш с дописанным P128_RAW);
           веса — Qwen3.5-0.8B (tied lm_head; RMSNorm g=1+w, eps=1e-6).
  методы : exp47_common (J(h)=A-rho(Au)u^T, V16 через eigh JJ^T fp64, углы Шюра R_t,
           a_t=angle(u_t,u_{t+1}), планарность p_t, выравнивание e_t с span{u_t,u_{t+1}},
           геодезическая кривизна Ω_t [M-3], полосный excess FFT в полосе периодов 4..32).
  selfcheck (K0.2, всё assert'ами): (i) J_analytic vs JR на 8 ключах rel < 1e-5;
           (ii) масштабный пол V ≤ 1e-5; (iii) доля позиций с gap(σ16>σ17) ≥ 95%.
  вердикты (K0.1, K0.3): считаются МЕХАНИЧЕСКИ из чисел по критериям §6, seed 47.

Замечание о порядке: кэш хранит t в ОБРАТном порядке позиций документа
(JR[t] = позиция S-1-t); соседние статистики (углы, автокорреляции, FFT-полоса) от
инверсии времени не меняются — направление времени здесь не используется.

Запуск (CPU, ~1 мин):  cd experiments && ../.venv/bin/python exp47a_kinematics_cache.py
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

import exp28_leaf_structure as E28          # noqa: E402  (load_layer)
import exp47_common as C                    # noqa: E402

torch.set_num_threads(4)                    # правило пользователя 09-30
RNG_SEED = 47


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x - x.mean(); y = y - y.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def boot_ci(vals, B: int = 10_000, seed: int = RNG_SEED):
    """Bootstrap-CI медианы по документам (парное пересэмплирование доменов НЕ делаем —
    доменов 4; вывод K0.3 опирается на CI по документам внутри домена, как зафиксировано)."""
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def _nn(o):
    """NaN/Inf -> None (конвенция проекта: JSON строго RFC8259, allow_nan=False)."""
    if isinstance(o, dict):
        return {k: _nn(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_nn(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/data/prism_kernels/exp27b/results")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--model-dir", default="/data/Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results",
                                                  "exp47a_kinematics_cache.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    ck = E28.load_layer(args.cache_dir, args.layer)
    D, S, n, doms = ck["D"], ck["S"], ck["n_samples"], ck["domains"]
    U128 = ck["U128"] if torch.is_tensor(ck["U128"]) else torch.tensor(ck["U128"])

    m = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32,
                                             attn_implementation="eager")
    m.eval()
    tied = m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    norm = m.model.norm
    eps = float(norm.eps)
    g = 1.0 + norm.weight.detach().float()          # Qwen: (1+w) — см. leaf_dynamics §2.1
    A = C.head_M(m.lm_head.weight.detach()[U128].float(), g)   # [128,D] fp64

    # ---------------- selfchecks (K0.2) ----------------
    sc = {"tied_head": bool(tied), "eps": eps}
    rel_max = 0.0
    g16_frac_ok = 0
    n_pos = 0
    floor_max = 0.0
    gen = torch.Generator().manual_seed(RNG_SEED)
    keys = [(d, s, t) for d in doms for s in range(n) for t in range(S)]
    check_idx = torch.randint(0, len(keys), (8,), generator=gen).tolist()
    for i in check_idx:
        dom, si, t = keys[i]
        h = ck["H"][(dom, si, t)].double()
        Jr = ck["JR"][(dom, si, t)].double()
        J, rho, u = C.J_analytic(A, h, eps)
        r_scalar = float(torch.rsqrt(h.pow(2).mean() + eps))
        rel_max = max(rel_max, float((J * r_scalar - Jr).norm() / Jr.norm()))
    sc["max_rel_J_vs_JR"] = rel_max
    assert rel_max < 1e-5, f"идентичность J не выполнена: {rel_max}"

    for i in check_idx[:4]:
        dom, si, t = keys[i]
        h = ck["H"][(dom, si, t)].double()
        V0, sig = C.V16_from_J(C.J_analytic(A, h, eps)[0])
        V1, _ = C.V16_from_J(C.J_analytic(A, 3.7 * h, eps)[0])
        ang = float(torch.acos(torch.linalg.svdvals(V0.T @ V1).clamp(max=1 - 1e-12)).max())
        floor_max = max(floor_max, ang)
        gap = float(sig[15] - sig[16]) if len(sig) > 16 else float("nan")
        n_pos += 1
        g16_frac_ok += int(gap > 0)
    sc["scale_floor_max_rad"] = floor_max
    assert floor_max <= 1e-5, f"пол метода слишком высок: {floor_max}"

    # gap на всех позициях (агрегат) — отдельным быстрым проходом по s16/s17 из спектра JJ^T
    gap_pos = 0
    for dom, si, t in keys[::7]:                     # подвыборка ~585 позиций
        h = ck["H"][(dom, si, t)].double()
        _, sig = C.V16_from_J(C.J_analytic(A, h, eps)[0])
        gap_pos += int(float(sig[15]) > float(sig[16]))
    sc["gap16_positive_frac"] = gap_pos / len(keys[::7])
    assert sc["gap16_positive_frac"] >= 0.95, "gap16 < 0 чаще 5% — Δκ-интерпретации невалидны (E-45 кр.0)"

    # ---------------- основной цикл ----------------
    docs = []
    pair_as = []        # дескриптив: пул пар (a_t, s_t) по всем позициям — гистограмма 2D
    for dom in doms:
        for si in range(n):
            hs = torch.stack([ck["H"][(dom, si, t)] for t in range(S)]).double()
            u = C.u_series(hs)
            a = C.angle_series(u)                                   # [S-1]
            om = C.omega_geodesic(u)                                # [S-2]
            s_ser, p_ser, e_ser = [], [], []
            for t in range(S - 1):
                Vt, _ = C.V16_from_J(C.J_analytic(A, hs[t], eps)[0])
                Vt1, _ = C.V16_from_J(C.J_analytic(A, hs[t + 1], eps)[0])
                s, p, P, _th = C.frame_kinematics(Vt, Vt1)
                s_ser.append(s); p_ser.append(p)
                e_ser.append(C.align_to_u_plane(P, u[t], u[t + 1]))
            s_a = np.array(s_ser); p_a = np.array(p_ser); e_a = np.array(e_ser)
            a_a = a.numpy().astype(float); om_a = om.numpy().astype(float)
            pair_as.extend(zip(a_a.tolist(), s_a.tolist()))
            thr = np.quantile(a_a[~np.isnan(a_a)], 0.10)            # пары a>p10 (K0.1)
            sel = a_a > thr
            docs.append({
                "dom": dom, "si": si,
                "a_med": float(np.nanmedian(a_a)), "s_med": float(np.nanmedian(s_a)),
                "p_med": float(np.nanmedian(p_a)), "e_med": float(np.nanmedian(e_a)),
                "om_med": float(np.nanmedian(om_a)),
                "r_pear": pearson(s_a[sel], a_a[sel]),
                "r_spear": spearman(s_a[sel], a_a[sel]),
                "excess_a": C.band_power(torch.from_numpy(a_a - np.nanmean(a_a))),
                "excess_s": C.band_power(torch.from_numpy(s_a - np.nanmean(s_a))),
            })
            print(f"{dom} s{si}: a={docs[-1]['a_med']:.4f} s={docs[-1]['s_med']:.4f} "
                  f"p={docs[-1]['p_med']:.3f} e={docs[-1]['e_med']:.3f} "
                  f"r={docs[-1]['r_pear']:.3f} exc_a={docs[-1]['excess_a']:.2f}", flush=True)

    # дескриптив (без вердиктов): квантильная 2D-гистограмма (a,s) — проверка гипотезы
    # НАСЫЩЕНИЯ: при a ≳ 0.6 рад расстояние V16 почти не растёт ⇒ корреляция r(s,a)
    # обязана быть низкой в prompt-соседствах и может вернуться в мелководей генерации
    aa = np.array([x for x, _y in pair_as]); ss = np.array([y for _x, y in pair_as])
    ok = ~(np.isnan(aa) | np.isnan(ss))
    qa = np.quantile(aa[ok], np.linspace(0, 1, 13))
    qs = np.quantile(ss[ok], np.linspace(0, 1, 13))
    H, _, _ = np.histogram2d(aa[ok], ss[ok], bins=[qa, qs])
    cond_mean_s = [float(np.nanmean(ss[ok][(aa[ok] >= qa[i]) & (aa[ok] < qa[i + 1])]))
                   for i in range(len(qa) - 1)]

    # ---------------- вердикты (механически, по §6) ----------------
    r_med = float(np.nanmedian([d["r_pear"] for d in docs]))
    by = {dm: [d for d in docs if d["dom"] == dm] for dm in doms}
    exc = {dm: [d["excess_a"] for d in v] for dm, v in by.items()}
    ci = {dm: boot_ci(exc[dm]) for dm in doms}
    med = {dm: float(np.nanmedian(exc[dm])) for dm in doms}
    others = [e for dm in ("wiki", "code") for e in exc[dm]]
    ratio = med["math"] / max(float(np.nanmedian(others)), 1e-12)
    ci_m, ci_o = boot_ci(exc["math"]), boot_ci(others)
    overlap = not (ci_m[1] < ci_o[0] or ci_o[1] < ci_m[0])
    verdicts = {
        "K0.1": {"r_s_a_median": r_med,
                 "verdict": ("кинематика=динамика u" if r_med >= 0.8 else
                             "частичная" if r_med >= 0.5 else "P1 ОПРОВЕРГНУТА")},
        "K0.2": {**sc, "verdict": "ok"},
        "K0.3": {"excess_math_med": med["math"], "excess_others_med": float(np.nanmedian(others)),
                 "ratio": ratio, "ci_math": ci_m, "ci_others": ci_o, "ci_overlap": overlap,
                 "verdict": ("преамбула аттрактора" if (ratio >= 1.5 and not overlap)
                             else "периодичности в промпт-режиме нет")},
        "K0.4": {dm: {"a": boot_ci([d["a_med"] for d in v]),
                      "s": boot_ci([d["s_med"] for d in v]),
                      "p": boot_ci([d["p_med"] for d in v]),
                      "e": boot_ci([d["e_med"] for d in v]),
                      "exc_a": ci[dm]} for dm, v in by.items()},
    }

    out = _nn({"params": {"protocol": "docs/leaf_dynamics.md §5.1 (P0)", "cache_dir": args.cache_dir,
                      "layer": args.layer, "D": D, "S": S, "n_per_dom": n, "domains": doms,
                      "pool_fp": str(ck.get("pool_fp")), "seed": RNG_SEED,
                      "note": "all-позиции (E-28 срез), t-порядок кэша обратен позиции — "
                              "на соседние статистики не влияет"},
           "docs": docs, "descriptive": {"a_edges": qa.tolist(), "s_edges": qs.tolist(),
                       "hist_a_s": H.astype(int).tolist(), "mean_s_by_abin": cond_mean_s,
                       "note": "дескриптив без порогов; e_t ≡ π/2+O(s) — структурно "
                               "(v ∈ row J_t ⟂ ker J_t ∋ u_t), метрика выравнивания пуста"},
           "verdicts": verdicts})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, allow_nan=False, indent=1)
    print(json.dumps({k: v for k, v in verdicts.items() if k != "K0.4"},
                     ensure_ascii=False, indent=1, default=str))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
