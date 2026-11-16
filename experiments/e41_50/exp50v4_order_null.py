#!/usr/bin/env python
"""E-50v4: ПЕРЕКЛАССИФИКАЦИЯ вердикта кривизны на облако-корректных null'ах.

Пере-предрег (ЗАПИСАН ДО первых чисел v4; основание — E50_review round-2 разбор
sur/swap, docs/E50v3_analysis.md):

Данные: те же трассы exp47b (qwen, 64 дока, T=96, W∈{1,16}, L=16, κ-медиана).
Null'ы (все — пер-док, детерминированные сиды det_seed(..._v4...)):
  swap : 25% соседних свапов, N_SW=3 (реплик поднято 1→3 — только дисперсия);
  sur  : step-matched изотропный, N_SUR=4 (2→4 — по выводу раунда-2);
  knn  : НОВЫЙ облако-сохраняющий null — жадный гамильтонов путь по kNN-графу
         кадров (sim = ||VᵀV||_F², k=8): облако и соседство сохранены ТОЧНО,
         временнóй порядок сломан сильнее, чем у swap (не только пары);
  rev  : обращение последовательности — САМОКОНТРОЛЬ (геодезический полигон
         при обратном обходе даёт обратное вращение с тем же ||schur||₂ ⇒
         κ_rev ≡ κ_real; нарушение = баг пайплайна, не физика).

ВЕРДИКТЫ (парной bootstrap по 64 документам, CI95, пер-док среднее по позициям):
  Δ_knn   = κ_real − mean κ_knn  :  CI_lo>0 → "order_curvature" (порядок несёт
                                   избыточную кривизну сверх облака);
                                  CI_hi<0 → "sub_geodesic_order";
                                  иначе   → "order_none".
  Δ_manif = mean κ_knn − mean κ_sur : CI_lo>0 → флаг "manifold_curved"
                                  (кривизна принадлежит облаку кадров, не порядку).
  гейт    : Δ_rev = κ_real − κ_rev обязан иметь CI ∋ 0, иначе прогон невалиден.
Диагностика: Spearman(κ_real, κ_knn) по документам (≈1 ⇒ облако объясняет всё).
Что НЕ меняется: κ, L=16, парность, метрика шага. Меняются: вердит-параметр
(sur → knn) и состав null'ов. Прошлые κ_real не пересчитываются — сверяются.

v4.1 (аудит 2026-11-07, ДО вердиктных чисел; основание — контрольный пробник на
2 доках, diag'ности null-семейства, не данных вердикта):
  1) knn-путь без мэтчинга шага: средний шаг ×2 от реального ⇒ «площадь»
     полигона ×4 ⇒ κ_knn надут на ~×8-12 (wiki_0: 0.0084-0.0127 vs real 0.0010).
     Лечится band-knn: переходы только с sim в полосе [q25,q75] соседних пар
     реального пути; step_ratio и fallback_frac пишутся в диагностику.
  2) rev-гейт был позиции́м (сравнивал РАЗНЫЕ окна); теперь точное парное
     сравнение петли t с перевёрнутой петлёй T-1-L-t: макс|diff| обязан быть
     <1e-6 (пробник: 9.5e-10 — пайплайн замкнут, гейт теперь ловит баги).
  3) NaN-санитайз per-doc полей (allow_nan=False иначе роняет финальный dump
     после часов счёта).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")     # урок E-40.5: 1 тред быстрее на мелочи
import argparse, json, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..")):
    if _p not in sys.path:
        _p and sys.path.insert(0, _p)
import exp50v3_curvature as V3  # noqa: E402
from exp49_leaf_info import load_model  # noqa: E402

N_SW4, N_SUR4, N_KNN4, K_NN = 3, 4, 3, 8


def knn_path(Vs, rng, k=K_NN, band=None):
    """Жадный гамильтонов путь по kNN-графу кадров: облако точно, порядок сломан.
    band=(lo,hi): ОБЛАКО-шаг-мэтчд (аудит 2026-11-07): брать только невизитых
    с sim в полосе sim соседних пар реального пути — иначе самоизбегающий обход
    шагает ~2 временны́ми шагами (шаг ×2 ⇒ «площадь» полигона ×4 ⇒ κ надут
    артефактом). Возвращает (путь, n_fallback)."""
    Vt = torch.stack([v.double() for v in Vs])            # [T,D,r]
    M = torch.einsum("tar,ubs->turs", Vt, Vt)              # [T,T,r,r] = V_aᵀ V_b
    S = (M * M).sum((-1, -2))                              # ‖V_aᵀV_b‖_F² ∈ [0,r]
    T = len(Vs)
    S.fill_diagonal_(-1.0)
    seen = [0]
    left = set(range(1, T))
    cur = 0
    nfb = 0
    while left:
        if band is None:
            cand = torch.topk(S[cur], min(k, T)).indices.tolist()
            cand = [c for c in cand if c in left]
            if cand:
                nxt = int(rng.choice(cand))
            else:
                nxt = int(rng.choice(sorted(left))); nfb += 1
        else:
            lo, hi = band
            cand = [j for j in left if lo <= float(S[cur, j]) <= hi]
            if cand:
                nxt = int(rng.choice(cand))
            else:
                mid = 0.5 * (lo + hi)
                nxt = min(left, key=lambda j: abs(float(S[cur, j]) - mid))
                nfb += 1
        seen.append(nxt); left.discard(nxt); cur = nxt
    return [Vs[i] for i in seen], nfb


def sim_band(Vs):
    """Полоса [q25,q75] sim-значений соседних пар реального пути."""
    Vt = torch.stack([v.double() for v in Vs])
    M = torch.einsum("tar,ubs->turs", Vt, Vt)
    S = (M * M).sum((-1, -2))
    adj = torch.diagonal(S, offset=1)
    q1, q3 = torch.quantile(adj.double(), torch.tensor([0.25, 0.75], dtype=adj.dtype))
    return float(q1), float(q3)


def _fin(x):
    x = float(x)
    return x if np.isfinite(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "results",
                                                  "exp50v4_order_null_qwen.json"))
    ap.add_argument("--n-docs", type=int, default=0)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--n-shards", type=int, default=3)
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    A, U, eps, npz, jp = load_model(args)
    docs = json.load(open(os.path.join(root, jp)))["docs"]
    if args.n_docs:
        docs = docs[: args.n_docs]
    if args.shard >= 0:
        docs = [d for i, d in enumerate(docs) if i % args.n_shards == args.shard]
    z = np.load(npz, allow_pickle=True)
    per = {"W1": [], "W16": []}
    for doc in docs:
        key = f"{doc['dom']}_{doc['si']}"
        f = z[key].item()
        u = torch.from_numpy(z[key + "_u"].astype(np.float64))[:V3.TMAX]
        hn = np.asarray(f["hnorm"], dtype=float)[:V3.TMAX]
        for W, tag in ((1, "W1"), (16, "W16")):
            V3._TAG[0] = tag
            rng = np.random.default_rng(V3.det_seed(f"{key}_{tag}_v4"))
            Vs = V3.frames_of(u, hn, A, eps, W)
            kr = V3.kappa_curve(Vs, (V3.MAIN_LAG,))[V3.MAIN_LAG]
            step_real = V3.mean_step(Vs)
            band = sim_band(Vs)
            def mpos(knull):
                return float(np.mean([kr[i] - float(np.mean([knull[j][i] for j in range(len(knull))
                                                            if i < len(knull[j])]))
                                      for i in range(len(kr))]))
            def reps(n):
                return []
            ks = {L: [] for L in (V3.MAIN_LAG,)}
            ksur, kknn = [], []
            nfb_tot, step_ratio = 0, []
            for _ in range(N_SW4):
                ks[V3.MAIN_LAG].append(V3.kappa_curve(V3.swap_order(Vs, rng), (V3.MAIN_LAG,))[V3.MAIN_LAG])
            for j in range(N_SUR4):
                sur = V3.synth_white_matched(Vs, Vs[0].shape[0], Vs[0].shape[1], len(Vs),
                                             V3.det_seed(f"{key}_{tag}_v4sur{j}"))
                ksur.append(V3.kappa_curve(sur, (V3.MAIN_LAG,))[V3.MAIN_LAG])
            for j in range(N_KNN4):
                pth, nfbj = knn_path(Vs, rng, band=band)
                nfb_tot += nfbj
                step_ratio.append(V3.mean_step(pth) / max(step_real, 1e-9))
                kknn.append(V3.kappa_curve(pth, (V3.MAIN_LAG,))[V3.MAIN_LAG])
            kvr = V3.kappa_curve(Vs[::-1], (V3.MAIN_LAG,))[V3.MAIN_LAG]
            # ТОЧНЫЙ гейт обращения: полигон перевёрнутого пути на позиции i есть
            # полигон реального пути на старте T-1-L-i (петля → P^{-1}, углы Шюра равны)
            Tn = len(Vs)
            d_rev_exact = max(abs(kr[i] - kvr[Tn - 1 - V3.MAIN_LAG - i]) for i in range(len(kr)))
            per[tag].append({
                "dom": doc["dom"], "key": key,
                "k_real": _fin(V3.safe_med(kr)),
                "d_swap": mpos(ks[V3.MAIN_LAG]),
                "d_knn": mpos(kknn),
                "d_rev": _fin(d_rev_exact),
                "knn_step_ratio": _fin(float(np.median(step_ratio))),
                "knn_fallback_frac": _fin(nfb_tot / float(N_KNN4 * (len(Vs) - 1))),
                "manif": float(np.mean([float(np.mean([kknn[j][i] for j in range(len(kknn))
                                                       if i < len(kknn[j])]))
                                        - float(np.mean([ksur[j][i] for j in range(len(ksur))
                                                         if i < len(ksur[j])]))
                                        for i in range(len(kr))])),
            })
        print(key, "done", flush=True)

    out = {"params": {"N_SW": N_SW4, "N_SUR": N_SUR4, "N_KNN": N_KNN4, "k": K_NN,
                      "knn": "band(sim q25..q75 соседних пар) v4.1",
                      "L": V3.MAIN_LAG, "prereg": "docstring v4 + audit v4.1"},
           "summary": {}, "per_doc": per}
    for tag, rows in per.items():
        s = {}
        for fld, name in (("d_knn", "order"), ("manif", "manifold"), ("d_swap", "swap"),
                          ("d_rev", "reversal_gate")):
            v = np.array([r[fld] for r in rows])
            lo, hi = V3.boot_ci(v)
            med = float(np.median(v))
            if fld == "d_rev":
                s[name] = {"med": med, "ci": [lo, hi], "ok": bool(med is not None and med < 1e-6)}
            elif fld == "manif":
                s[name + "_flag"] = bool(lo is not None and lo > 0)
                s[name] = {"med": med, "ci": [lo, hi]}
            else:
                verdict = ("curvature" if lo is not None and lo > 0 else
                           "sub_geodesic" if hi is not None and hi < 0 else "none")
                s[name + "_verdict"] = verdict
                s[name] = {"med": med, "ci": [lo, hi],
                           "pos_docs": int((v > 0).sum()), "n": len(v)}
        kr = np.array([r["k_real"] for r in rows])
        out["summary"][tag] = s
        out["summary"][tag]["k_real_med"] = float(np.median(kr))
        out["summary"][tag]["knn_step_ratio_med"] = float(np.median(
            [r["knn_step_ratio"] for r in rows if r["knn_step_ratio"] is not None] or [float("nan")]))
        out["summary"][tag]["knn_fallback_frac_mean"] = float(np.mean(
            [r["knn_fallback_frac"] for r in rows if r["knn_fallback_frac"] is not None] or [float("nan")]))
        print(tag, json.dumps(s, ensure_ascii=False)[:400], flush=True)
    json.dump(out, open(args.out, "w"), indent=1, allow_nan=False)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
