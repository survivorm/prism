#!/usr/bin/env python
"""E-45b-analysis: чтение таблицы «лист ↔ поведение» с честными нулями.

Две ловушки, под которые таблица ломается, если не смотреть:

1. Автокорреляция позиций. Соседние позиции документа похожи (и Δ_SVD(t), и энтропия, и
   «копирование» тянутся блоками), поэтому обычный критерий Спирмена с p от n≈4000 даёт
   «значимость» на пустом месте. Нулевая гипотеза здесь — перестановка БЛОКАМИ по 8 позиций
   внутри документа: она сохраняет внутриблочную структуру и убивает только дальнюю связь.
   Сообщаем z по блочному нулю, а не наивное p.

2. Разные основания в разных масштабах. align = ‖P_V ŵ‖ зависит от того, сколько энергии базис
   вообще захватывает, поэтому сравнивать лист и PCA можно только ВНУТРИ основания:
   score(basis) = align_next / align_rand (тот же базис, случайный токен вместо настоящего).
   Абсолютные числа между основаниями не сравниваются.

Запуск:  cd experiments && ../.venv/bin/python exp45b_analysis.py
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "exp45b_leaf_table.json.gz")
BLOCK = 8                      # длина блока перестановки (позиции внутри документа)
N_PERM = 1000
GAP_PREREG = 0.01       # пререгистрация E-45b: «позиции с зазором меньше — невалидны»
# Единица в пререгистрации оказалась догадкой, а не измерением: измеренный зазор на r=16
# равен 0.0087 (процентиля 0.0078…0.0097, E-45b на свежем тексте и 0.00875 на кэше в E-45g),
# то есть порог 0.01 объявляет невалидными 98.9-99.7 % позиций и весь анализ пуст (n = 0).
# E-45g (`results/exp45g_cut_stability.json`) померил настоящее основание: при шуме ε ≤ 1e-06
# правый базис V₁₆ не двигается (0.00°) и κ не меняется, а порядок σ₁₆ > σ₁₇ не ломается даже
# при ε = 0.25. Значит валидность разреза = «зазор больше численного пола», а не «больше 1 %».
GAP_MIN = 1e-4            # порог валидности по измеренному численному полу (E-45g)
GAP_NOISE_FLOOR = 1e-6    # ε, при котором V₁₆ и κ ещё не шевелятся (E-45g, ε-скан)
SEED = 0

FEATS = ("d1_r16", "k2_r16", "pr_sigma", "g_uni", "j_norm", "gap_r16", "s16_over_s1",
         "h_norm", "align_next")
TARGETS = ("nll", "dnll", "ent", "rank", "in5", "copy16")


def log(*a):
    print(*a, flush=True)


def shift_next(rows, key):
    """Значение key на СЛЕДУЮЩЕЙ позиции того же документа (на последней — NaN)."""
    out = np.full(len(rows), np.nan)
    where = defaultdict(dict)
    for i, r in enumerate(rows):
        where[(r["domain"], r["doc"])][r["t"]] = i
    for _k, m in where.items():
        for t, i in m.items():
            j = m.get(t + 1)
            if j is not None:
                out[i] = rows[j][key]
    return out


def masked_blocks(m2, doc_of):
    """Блоки подряд идущих позиций внутри документа, выраженные в индексах МАССИВА ПОСЛЕ маски.

    Строки таблицы записаны по документам и позициям, поэтому соседство индексов = соседство
    позиций; блоки короче BLOCK остаются как есть — перестановка обязана накрывать все позиции.
    """
    idx = np.flatnonzero(m2)
    if len(idx) == 0:
        return np.empty(0, int), []
    blocks, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1 and doc_of[b] == doc_of[a] and len(cur) < BLOCK:
            cur.append(b)
        else:
            blocks.append(cur)
            cur = [b]
    blocks.append(cur)
    pos = [np.searchsorted(idx, np.asarray(b)) for b in blocks]
    return idx, pos


def block_perm_corr(x_all, y_all, mask, doc_of, rng):
    """Спирмен по замаскированным позициям и z против перестановки блоков внутри документов."""
    idx, pos = masked_blocks(mask, doc_of)
    if len(idx) < 3 * BLOCK:
        return float("nan"), float("nan"), (float("nan"), float("nan")), 0
    x, y = x_all[idx], y_all[idx]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    c = float(np.corrcoef(rx, ry)[0, 1])
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        order = rng.permutation(len(pos))
        perm = np.concatenate([pos[k] for k in order])
        null[p] = np.corrcoef(rx[perm], ry)[0, 1]
    z = float((c - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    lo, hi = np.percentile(null, [2.5, 97.5])
    return c, z, (float(lo), float(hi)), len(idx)


def main():
    with gzip.open(RES, "rt") as f:
        d = json.load(f)
    rows, meta = d["rows"], d["meta"]
    log(f"строк {len(rows)}, доменов {meta['domains']}, r_main={meta['r_main']}, "
        f"PPL таблицы {meta.get('ppl_table', float('nan')):.2f}")
    ht = np.array([r["has_target"] for r in rows], dtype=bool)
    doms = np.array([r["domain"] for r in rows])

    def col(k):
        return np.array([r[k] for r in rows], dtype=np.float64)

    log("\n== 0. валидность разреза (κ и V_r имеют смысл только при зазоре на r) ==")
    gap = col("gap_r16")
    for dom in sorted(set(doms)):
        m = doms == dom
        log(f"  {dom:>5s}: медиана gap_r16 = {np.median(gap[m]):.4f}, "
            f"доля позиций с gap < {GAP_MIN:g} = {np.mean(gap[m] < GAP_MIN) * 100:.1f} %")
    log(f"  пререгистрационный порог {GAP_PREREG}: невалидно {np.mean(gap < GAP_PREREG) * 100:.1f} %"
        f" — при нём анализ пуст, поэтому валидность берётся по численному полу E-45g")
    log(f"  медианный зазор {np.median(gap):.5f} = ×{np.median(gap) / GAP_NOISE_FLOOR:.0f} над "
        f"шумом, при котором V₁₆ и κ ещё не двигаются; невалидно по {GAP_MIN:g}: "
        f"{np.mean(gap < GAP_MIN) * 100:.1f} %")
    log(f"  всего: доля невалидных позиций = {np.mean(gap < GAP_MIN) * 100:.1f} % "
        f"(на них выводы «лист разрушен/укреплён» не опираем)")

    log("\n== 1. медианы по доменам ==")
    keys = ("d1_r16", "k2_r16", "pr_sigma", "g_uni", "j_norm", "h_norm", "nll", "ent",
            "copy16", "align_next", "align_rand", "align_pca_next")
    log("  " + f"{'домен':>6s} " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for dom in sorted(set(doms)):
        m = (doms == dom) & ht
        log("  " + f"{dom:>6s} " + " ".join(f"{np.median(col(k)[m]):9.4f}" for k in keys))
    m = ht
    log("  " + f"{'ВСЕ':>6s} " + " ".join(f"{np.median(col(k)[m]):9.4f}" for k in keys))

    log("\n== 1b. конфаунд-аудит (п.8 пререгистрации): лист против ‖h‖ ==")
    rng = np.random.default_rng(SEED)
    doc_of = np.array([f"{r['domain']}#{r['doc']}" for r in rows])   # 1-D строки: из list-of-tuples numpy делает 2-D массив, и == возвращает вектор (первый прогон упал именно здесь)
    hn, kk = col("h_norm"), col("k2_r16")
    m3 = ht & np.isfinite(hn) & np.isfinite(kk) & (gap >= GAP_MIN)
    c_hn, z_hn, _ci, n_hn = block_perm_corr(hn, kk, m3, doc_of, rng)
    log(f"  Spearman(κ_локальная, ‖h‖) = {c_hn:+.3f} (блочный z = {z_hn:+.1f}, n = {n_hn})")
    trigger = abs(c_hn) > 0.5
    msg = ("СРАБОТАЛ ⇒ все связи репортятся только частичными, с ‖h‖ как ковариатой"
           if trigger else "не сработал")
    log(f"  порог |ρ| > 0.5 по пререгистрации: {msg}")
    if trigger:
        # частичная корреляция Спирмена: из обеих величин вычитается линейный прогноз по рангу ‖h‖
        rzh = np.argsort(np.argsort(hn)).astype(np.float64)
        rzf = {}

        def resid(v):
            r = np.argsort(np.argsort(np.where(np.isfinite(v), v, 0.0))).astype(np.float64)
            mx, my = rzh[m3].mean(), r[m3].mean()
            b = float(np.dot(rzh[m3] - mx, r[m3] - my) / max(np.dot(rzh[m3] - mx, rzh[m3] - mx), 1e-9))
            return r - (my + b * (rzh - mx))

        log("  парные частичные связи (лист(t) → поведение(t+1), ковариата ‖h‖):")
        rows2 = []
        for f_ in FEATS:
            rf = resid(col(f_))
            for tg in TARGETS:
                y = resid(shift_next(rows, tg))
                m4 = m3 & np.isfinite(shift_next(rows, tg))
                c2, z2, _c2, n2 = block_perm_corr(rf, y, m4, doc_of, rng)
                rows2.append((f_, tg, c2, z2, n2))
        rows2.sort(key=lambda r: -abs(r[3]))
        for f_, tg, c2, z2, n2 in rows2[:8]:
            log(f"    {f_:>12s} → {tg:<7s}: ρ_част = {c2:+.3f}  z = {z2:+5.1f}  n = {n2}")
        log(f"  … частичных пар |z| > 3: {sum(1 for r in rows2 if abs(r[3]) > 3)} из {len(rows2)}")

    log("\n== 2. лист(t) против поведения(t+1): Спирмен и z по блочной перестановке ==")
    log("   (наивный p при n≈4000 был бы значим на чём угодно; здесь z против нуля «блоки по "
        f"{BLOCK} позиций перемешаны внутри документа», {N_PERM} перестановок)")
    valid = ht & (gap >= GAP_MIN) & np.isfinite(gap)
    res = []
    for f_ in FEATS:
        x = col(f_)
        for tg in TARGETS:
            y = shift_next(rows, tg)
            m2 = valid & np.isfinite(x) & np.isfinite(y)
            c, z, _ci, n = block_perm_corr(x, y, m2, doc_of, rng)
            res.append((f_, tg, c, z, n))
    res.sort(key=lambda r: -abs(r[3]))
    for f_, tg, c, z, n in res[:12]:
        log(f"  {f_:>12s} → {tg:<7s}: ρ = {c:+.3f}  z = {z:+5.1f}  n = {n}  "
            f"{'ОТДЕЛИМО от блочного нуля' if abs(z) > 3 else 'не отделимо'}")
    log(f"  … всего пар {len(res)}, из них |z| > 3: {sum(1 for r in res if abs(r[3]) > 3)} "
        f"(при {len(res)} парах ожидаемое число ложных |z|>3 ≈ 0.002 × {len(res)} = "
        f"{0.0027 * len(res):.1f})")

    log("\n== 3. «лист как носитель следующего токена» против PCA того же ранга ==")
    an, ar = col("align_next"), col("align_rand")
    pn, pr = col("align_pca_next"), col("align_pca_rand")
    haar = (meta["r_main"] / 1024.0) ** 0.5
    ah = col("align_haar")
    log(f"  Haar-ориентир √(r/D) = {haar:.4f}, измеренный "
        f"align_haar = {np.median(ah):.4f} — механический уровень ранг-r подпространства")
    log(f"  align_rand (тот же базис, случайный токен) = "
        f"{np.median(ar):.4f} = ×{np.median(ar) / max(np.median(ah), 1e-9):.2f} от Haar, "
        f"align_pca_rand = {np.median(pr):.4f}: строки словаря анизотропны, "
        f"поэтому знаменатель скоринга — он, а не Haar")
    with np.errstate(divide="ignore", invalid="ignore"):
        s_leaf, s_pca = an / np.maximum(ar, 1e-9), pn / np.maximum(pr, 1e-9)
    rng2 = np.random.default_rng(1)
    docs = np.array([f"{r['domain']}#{r['doc']}" for r in rows])
    uniq = sorted(set(docs.tolist()))
    dmed = {k: np.median(s_leaf[docs == k]) for k in uniq}
    dpca = {k: np.median(s_pca[docs == k]) for k in uniq}
    dif = np.array([dmed[k] - dpca[k] for k in uniq])
    boot = np.array([np.mean([dmed[k] for k in rng2.choice(uniq, len(uniq))]) -
                     np.mean([dpca[k] for k in rng2.choice(uniq, len(uniq))])
                     for _ in range(2000)])
    log(f"  score(лист) медианой по позициям = {np.median(s_leaf[ht]):.3f}, "
        f"score(PCA) = {np.median(s_pca[ht]):.3f}")
    log(f"  разница по документам = {dif.mean():+.3f}, bootstrap-CI [{np.percentile(boot, 2.5):+.3f}, "
        f"{np.percentile(boot, 97.5):+.3f}], лист лучше на {100 * np.mean(dif > 0):.0f} % документов")
    log("  вывод: " + ("лист содержит строку считывания следующего токена сильнее, чем PCA того же "
                       "ранга (внутрибазисная мера)" if np.percentile(boot, 2.5) > 0 else
                       "PCA того же ранга не хуже — «носитель» приписать листу нельзя"))
    log("  напоминание: PCA построен по всему документу (включая будущие позиции) — это вопрос о "
        "носителе представления, а не о прогнозе; симметрично и в E-46")


if __name__ == "__main__":
    main()
