#!/usr/bin/env python
"""E-46-verdicts: чтение «лист как носитель» по пререгистрационному критерию, с bootstrap по документам.

Статистика. Любая проекция 1024-мерного h в 16 измерений уничтожает почти всё, поэтому абсолютный
KL ничего не решает: сравниваем только с конкурентами той же размерности и нормируем на «потолок»
потери своей рамы:

    score(variant) = KL(variant) / KL(leaf_lag0)   ≥ 1, «во сколько раз хуже, чем своя рама»

Вывод по пререгистрации (PLAN.md п.7): лист — носитель, если score(leaf_lag1) достоверно ниже
минимума score по null'ам {haar, pca_doc, pca_doc_raw, pca_win} (парный bootstrap по документам,
CI не накрывает 0 в долях) И лестница лагов монотонна (KL растёт с Δ). Если хоть один PCA-конкурент
не хуже — «носителя нет», и это пишется тем же шрифтом.

Запуск:  cd experiments && ../.venv/bin/python exp46_verdicts.py
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "exp46_frame_as_memory.json.gz")
NULLS = ("haar", "pca_doc", "pca_doc_raw", "pca_win")
# leaf_perm — та же совокупность рам, разложенная по позициям случайно внутри
# документа. Если она не хуже leaf_lag1, то работает не носительство, а «любая
# рама этого документа ловит нужную энергию». В гейт п.7 НЕ входит (добавлена до
# чисел, диагностика).
DIAG = ("leaf_perm",)
N_BOOT = 2000
SEED = 0


def log(*a):
    print(*a, flush=True)


def per_doc_median(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["doc"]].append(r[key])
    return {d: float(np.median(v)) for d, v in sorted(by.items())}


def boot_diff(a, b, n=N_BOOT, seed=SEED):
    """Парный bootstrap разности средних по документам (a и b — словари doc→значение)."""
    docs = sorted(set(a) & set(b))
    xa, xb = np.array([a[d] for d in docs]), np.array([b[d] for d in docs])
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(docs), len(docs))
        ds.append(xa[i].mean() - xb[i].mean())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(docs=len(docs), mean_diff=float(xa.mean() - xb.mean()),
                ci=[float(lo), float(hi)],
                wins_docs=float(np.mean([a[d] < b[d] for d in docs])))


def main():
    with gzip.open(RES, "rt") as f:
        d = json.load(f)
    rows = d["rows"]
    log(f"строк {len(rows)}, документов {len({r['doc'] for r in rows})}, "
        f"проектор-чек {d['projector_check']}")

    # нормировка на потолок своей рамы (по каждой позиции)
    by_pos = defaultdict(dict)
    for r in rows:
        by_pos[(r["doc"], r["t"])][r["variant"]] = r
    tab = defaultdict(list)
    for key, v in by_pos.items():
        if "leaf_lag0" not in v:
            continue
        base = max(v["leaf_lag0"]["kl"], 1e-12)
        base_nr = max(v["leaf_lag0"]["kl_nr"], 1e-12)
        for name, r in v.items():
            tab[name].append(dict(doc=key[0], t=key[1], kl=r["kl"], score=r["kl"] / base,
                                  kl_nr=r["kl_nr"], score_nr=r["kl_nr"] / base_nr,
                                  energy=r["energy"],
                                  top5=r["top5"], dnll=r["nll_proj"] - r["nll_true"]))

    log("\n== медианы по всем позициям (score = KL / KL своей рамы; _nr — с восстановлением нормы) ==")
    log(f"{'вариант':>14s} {'KL':>9s} {'score':>7s} {'KL_nr':>9s} {'score_nr':>9s} "
        f"{'энерг.':>7s} {'top5':>5s} {'ΔNLL':>7s}")
    order = ([f"leaf_lag{lag}" for lag in d["lags"]] + list(NULLS) + ["mean_dir"] + list(DIAG))
    med = {}
    for name in order:
        if name not in tab:
            continue
        def mz(k):
            return np.array([x[k] for x in tab[name]])
        med[name] = dict(kl=per_doc_median(tab[name], "kl"),
                         score=per_doc_median(tab[name], "score"),
                         score_nr=per_doc_median(tab[name], "score_nr"),
                         dnll=per_doc_median(tab[name], "dnll"))
        log(f"{name:>14s} {np.median(mz('kl')):9.4f} {np.median(mz('score')):7.3f} "
            f"{np.median(mz('kl_nr')):9.4f} {np.median(mz('score_nr')):9.3f} "
            f"{np.median(mz('energy')):7.3f} {np.median(mz('top5')):5.2f} "
            f"{np.median(mz('dnll')):7.3f}")

    for field, tag in (("score", "KL"), ("score_nr", "KL с восстановлением нормы")):
        log(f"\n== п.7 ({tag}): лист против null'ей, парный bootstrap по документам ==")
        prim = "leaf_lag1"
        ok_all = True
        for nl in NULLS:
            if nl not in med:
                continue
            bs = boot_diff(med[prim][field], med[nl][field])
            better = bs["ci"][1] < 0
            ok_all = ok_all and better
            log(f"  {prim} − {nl:<11s}: Δ = {bs['mean_diff']:+.3f} "
                f"CI [{bs['ci'][0]:+.3f}, {bs['ci'][1]:+.3f}]  лист лучше на "
                f"{100 * bs['wins_docs']:.0f} % документов → "
                f"{'лист лучше' if better else ('null лучше' if bs['ci'][0] > 0 else 'НЕТ различия')}")
        log("  ИТОГ: " + ("лист лучше ВСЕХ null'ей (носитель есть)" if ok_all else
                          "носителя нет: как минимум один null не хуже листа"))
        for nl in DIAG:
            if nl not in med or prim not in med:
                continue
            bs = boot_diff(med[prim][field], med[nl][field])
            bs_nr = boot_diff(med[prim]["score_nr"], med[nl]["score_nr"])
            log(f"  [диагн] {prim} − {nl:<9s}: Δ(score) = {bs['mean_diff']:+.3f} "
                f"CI [{bs['ci'][0]:+.3f}, {bs['ci'][1]:+.3f}], Δ(score_nr) = "
                f"{bs_nr['mean_diff']:+.3f} CI [{bs_nr['ci'][0]:+.3f}, {bs_nr['ci'][1]:+.3f}] → "
                + ("«любая рама документа» ХУЖЕ листа хотя бы в одном варианте"
                   if (bs['ci'][1] < 0 or bs_nr['ci'][1] < 0)
                   else "«любая рама документа» НЕ ХУЖЕ листа: привязка рама↔позиция сама по "
                        "себе ничего не даёт"))

    log("\n== лестница лагов (монотонность = медленное вращение рамы) ==")
    lags = [f"leaf_lag{lag}" for lag in d["lags"]]
    docs = sorted(med[lags[0]]["kl"])
    per_doc = {lg: np.array([med[lg]["kl"][dd] for dd in docs]) for lg in lags}
    for lg in lags:
        log(f"  {lg:>12s}: медианный KL = {np.median(per_doc[lg]):.4f}")
    mono = np.mean([all(np.diff([per_doc[lg][i] for lg in lags]) >= -1e-9)
                    for i in range(len(docs))])
    log(f"  документов с немонотонной лестницей: {100 * (1 - mono):.0f} %")
    sp = []
    for i in range(len(docs)):
        y = np.array([per_doc[lg][i] for lg in lags])
        # аудит v6: метрика называлась Spearman, но считалась как Пирсон по
        # рангу лага против СЫРОГО значения KL; теперь ранги и того, и другого
        sp.append(np.corrcoef(np.arange(len(lags)), np.argsort(np.argsort(y)))[0, 1])
    log(f"  Spearman(lag, KL) по документам (ранги обеих осей): медиана {np.median(sp):+.3f}, "
        f"доля > 0: {100 * np.mean(np.array(sp) > 0):.0f} %")
    log("  замечание: монотонность лестницы — это то же предсказание, что стоит за κ; если её нет,")
    log("  «медленно вращающаяся рама» не подтверждается, каким бы хорошим ни был lag1 против null'ей")


if __name__ == "__main__":
    main()
