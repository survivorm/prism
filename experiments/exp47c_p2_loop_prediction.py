#!/usr/bin/env python
"""E-47c (P2): ПРЕДСКАЗАНИЕ петли листом vs тривиальные конкуренты.

Критерии — docs/leaf_dynamics.md §6 (P2), зафиксированы до чисел:
  AUC >= 0.8, фора >= 5 токенов, И ΔAUC против тривиальных фич >= 0.05;
  иначе — «лист не добавляет» (пишется так же отчётливо).

Дизайн (фиксируется здесь, до запуска):
  * данные: СЫРЬЁ P1 без перепрогона — exp47b npz (Qwen) и exp48b npz (LFM3),
    4 домена × 16 документов, фичи per step;
  * target: y_t^(Δ) = «петля объявлена детектором» (onset-маска find_loops) в каком-либо
    шаге (t, t+Δ] — предсказываем ОБЪЯВЛЕНИЕ петли в горизонте Δ;
  *листьевые-фичи (leaf): [a_t, s_t, p_t] + rolling-медианы w=16 => 9;
  * тривиальные (triv): [ent_t, top1_t, eq_prev_t] + rolling-средние w=16 (9) +
    acf_recent = max|autocorr| последних 24 значений s_t на лагах 2..12 (1) => 10;
  * all: leaf ∪ triv;
  * модель: логистическая регрессия (L2, C=1, class_weight=balanced), стандартизация по
    трейну; AUC — по пулу шагов каждого фолда;
  * разделение: ГЛАВНОЕ cross-domain (train 3 домена → test 4-й, 4 фолда); sweep Δ=5/16/32/64;
    вердикт по Δ=16 (пререгистрировано), фора = max Δ с leaf-AUC>=0.8.
Ограничения (честно): фича e (выравненность du с плоскостью листа) в npz не писалась —
используются {a,s,p}; «петля» = детектор (включает wiki-коллапс), как в K1.2; шаги внутри
микропетель имеют a≈0 — это фича, не баг.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = {
    "qwen": ("/data/prism_kernels/exp47b/exp47b_gen_kinematics.npz",
             "results/exp47b_gen_kinematics.json"),
    "lfm3": ("/data/prism_kernels/exp48b/exp48b_lfm3_gen_kinematics.npz",
             "results/exp48b_lfm3_gen_kinematics.json"),
}
# СЕЛФ-РЕВИЮ (2026-11-05): в exp48b ent/top1 считались по U-128 подвектору softmax, а у
# Qwen — по полному vocab (0.02 бит против 6.7 бит на тех же шагах!). triv-фичи LFM3
# пересчитаны из сохранённых u по полному vocab (шаг 8, zero-order hold) — /tmp/e48b.
TRIV_FIX = {"lfm3": "/tmp/e48b/lfm3_triv_fullfix.npz"}
W = 16
DELTS = (5, 16, 32, 64)
MAIN_DELT = 16


def roll_med(x, w=W):
    return np.array([np.nanmedian(x[max(0, i - w + 1):i + 1]) for i in range(len(x))])


def acf_recent(x, w=24, lags=tuple(range(2, 13))):
    out = np.zeros(len(x))
    for i in range(len(x)):
        seg = x[max(0, i - w + 1):i + 1]
        if np.isfinite(seg).sum() <= 4:
            continue
        seg = np.nan_to_num(seg - np.nanmean(seg))
        den = float((seg * seg).sum())
        if den > 0:
            out[i] = max(abs(float((seg[:-l] * seg[l:]).sum() / den)) for l in lags)
    return out


def build_feats(f, triv_fix=None):
    a, s, p = (np.asarray(f[k], float) for k in ("a", "s", "p"))
    ent, top1, eq = (np.asarray(f[k], float) for k in ("ent", "top1", "eq_prev"))
    if triv_fix is not None:
        # ZOH: полновocab-фичи пересчитаны каждый 8-й шаг -> растягиваем на сетку шагов
        ent = np.repeat(ent, 8)[:len(ent) * 8][:len(a)]
        top1 = np.repeat(top1, 8)[:len(top1) * 8][:len(a)]
    leaf = np.column_stack([a, s, p] + [roll_med(m) for m in (a, s, p)])
    triv = np.column_stack([ent, top1, eq] + [roll_med(m) for m in (ent, top1, eq)]
                           + [acf_recent(s)])
    return leaf, triv


def onset_mask(doc, T):
    m = np.zeros(T, dtype=bool)
    for o, _j, e, _p, _k in doc["loops"]:
        m[min(o, T - 1):min(e, T)] = True
    return m


def y_forward(mask, delta):
    """y_t = 1 если маска истинна в (t, t+delta]."""
    T = len(mask)
    cs = np.concatenate([[0], np.cumsum(mask)])
    hi = np.minimum(np.arange(T) + delta + 1, T)
    return (cs[hi] - cs[np.arange(T) + 1]) > 0


def auc(y, score):
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r1 = ranks[y].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-9
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit((Xtr - mu) / sd, ytr.astype(int))
    return clf.decision_function((Xte - mu) / sd)


def stack(sets, ks, g):
    if g == "leaf":
        X = np.vstack([sets[k][0] for k in ks])
    elif g == "triv":
        X = np.vstack([sets[k][1] for k in ks])
    else:
        X = np.vstack([np.hstack([sets[k][0], sets[k][1]]) for k in ks])
    y = np.concatenate([sets[k][2] for k in ks])
    assert X.shape[0] == y.shape[0], (X.shape, y.shape)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6), y


def run_model(sets, keys):
    res = {g: {} for g in ("leaf", "triv", "all")}
    for test_dom in sorted({k.rsplit("_", 1)[0] for k in keys}):
        tr = [k for k in keys if not k.startswith(test_dom + "_")]
        te = [k for k in keys if k.startswith(test_dom + "_")]
        Xtr = {g: stack(sets, tr, g) for g in res}
        for g in res:
            (Xtr_g, ytr), (Xte, yte) = Xtr[g], stack(sets, te, g)
            res[g][test_dom] = auc(yte, fit_predict(Xtr_g, ytr, Xte))
    for g in res:
        vals = [v for v in res[g].values() if v == v]
        res[g]["pooled_mean"] = float(np.mean(vals)) if vals else float("nan")
    return res


def main():
    root = os.path.join(_HERE, "..")
    out = {}
    for name, (npz, jpath) in SOURCES.items():
        docs = json.load(open(os.path.join(root, jpath)))["docs"]
        z = np.load(npz, allow_pickle=True)
        fix = np.load(TRIV_FIX[name]) if name in TRIV_FIX else None
        sets = {}
        for doc in docs:
            key = f"{doc['dom']}_{doc['si']}"
            f = z[key].item()
            tf = None
            if fix is not None:
                tf = True
                f = dict(f); f["ent"] = fix[key + "_ent"]; f["top1"] = fix[key + "_top1"]
            leaf, triv = build_feats(f, triv_fix=tf)
            T = leaf.shape[0]
            m = onset_mask(doc, T)
            for delta in DELTS:
                sets.setdefault(delta, {})[key] = (leaf, triv, y_forward(m, delta)[:T])
        rep = {}
        for delta in DELTS:
            r = run_model(sets[delta], list(sets[delta]))
            rep[delta] = {g: round(r[g]["pooled_mean"], 3) for g in ("leaf", "triv", "all")}
            rep[delta]["folds_leaf"] = {k: round(v, 3) for k, v in r["leaf"].items()
                                        if k != "pooled_mean"}
        m16 = rep[MAIN_DELT]
        leaf_auc, triv_auc, all_auc = m16["leaf"], m16["triv"], m16["all"]
        lead = max([d for d in DELTS if rep[d]["leaf"] >= 0.8], default=0)
        verdict = ("ЛИСТ ДОБАВЛЯЕТ" if (leaf_auc >= 0.8 and lead >= 5
                                        and all_auc - triv_auc >= 0.05)
                   else "лист не добавляет (по пререгистрации §6 P2)")
        out[name] = {
            "auc_cross_domain_by_delta": rep,
            "verdict": {"AUC_leaf_d16": leaf_auc, "AUC_triv_d16": triv_auc,
                        "AUC_all_d16": all_auc, "delta_auc_all_vs_triv": round(all_auc - triv_auc, 3),
                        "lead_tokens_leaf_ge08": lead, "verdict": verdict},
        }
        print(name, verdict, "| leaf/triv/all @d16:", leaf_auc, triv_auc, all_auc, flush=True)
    path = os.path.join(root, "results", "exp47c_p2_loop_prediction.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("WROTE", path)


if __name__ == "__main__":
    main()
