

# AT0_3_r31_REVIEW
Ревьюер: reviewer-subagent (точечные сверки по §A–§D, репозиторий не читал по технике)

ВЕРДИКТ: PASS_WITH_MINORS

N1: закрыто. (a) строка 1033: `abs(vc[0] - meta_dip23)` — сравнивается dip, пересчитанный из (U,hn) (1026–1030), с продюсерским `curves_meta[23]['dip']`, а не h с U*hn — тавтология убрана; (b) порог `tpu_dev >= 1e-3` (1037); (c) `tpu_n == 0` → только vacuous-лог (1035–1036), не fail. Dip — скалярная характеристика кривой, пересчёт из (u,hn) не тождественен h, чек реальный.

N2: закрыто. A2: L96m/C96m — фильтр `K == 96` (1132–1134), primary = k1l96/k1c96, full-pool + K_composition — secondary. A3: тест и aggregate — только k96rows. Числа: K_comp clean {32:2, 96:51} → clean K96=51; loopy r3=41 и nL(K96)=41 → весь loopy пул K96, т.е. на loopy-стороне KS-пул и location-пул тождественно совпадают; на clean-стороне совпадение «те же доки, что в KS» опирается на комментарий A2 (код KS во фрагменте нет), числовых противоречий нет (0.3305 vs 0.3354 — эффект 2 K32-доков). §P↔§B: 30/41 vs 0/51, p=2.18e-07, k1 0.0738/0.3305 (4.48×) — совпадает.

N3: закрыто. Гейт A3: `p_two_sided_exact <= ALPHA` ∧ `median_concentration > 1` ∧ `frac_conc_above_1 >= 0.8`; min_concentration из гейта исключён. По §P: p=2.18e-07 ≤ 0.05, median=6.06, frac=1.0 → confirmed верно. Robust к одному выбросу: медиана по n=41 сдвигается максимум на один ранг, frac≥0.8 допускает до 8 доков с conc≤1 (при min=2.43 запас есть).

N4: закрыто. Все пункты A4 на месте и не противоречат §P: [ИСТОРИЧЕСКАЯ]-пометки prereg, docstring (two-sided для signed / right-tail для KS), guard `k1l96 and k1l96 > 1e-6 else None`, «гипотеза, не результат» в summary, forced_outcome у v_tangent, status_short H1 + detail, per_doc_conc_p_format.

N5: закрыто. store_sha=190439a0fce981a2 = artifact sha, match=True; per_doc=96; no NaN/inf; manifest claim_sha OK; все гейты PASS. §P↔§B расхождений нет (location, period, H1 — идентичные числа). sha с диском не сверял — по технике репозиторий не читается, сверка текстовая.

НОВЫЕ:
- [MINOR] A1: TPU-док без `curves_meta[23]['dip']` молча исключается (1031 `continue`) — tpu_n занижается, критичный гейт может пройти на меньшем n без видимости. Фикс: считать skipped и логить его в той же строке.
- [MINOR] A3: вычисление p0/p_two вынесено за фрагмент («...»); K96-фильтр сузил пул — проверить, что exact-binom при nL+nC==0 (smoke) не даёт ZeroDivisionError/NaN; при отсутствии добавить guard → test=None.

VERDICT=PASS_WITH_MINORS
AT0_3_R31_DONE
---

## Ответ исполнителя (r3.1.2) — оба MINOR закрыты, перепуск

1. **A1 (silent skip)**: добавлен счётчик `tpu_skip` — логится в строке гейта и в
   `gates.G0a_tpu_curves.n_skipped`. Полный прогон: **n=32 skipped=0** (ни один TPU-док
   не пропущен — опасения не материализовались, но теперь видимы).
2. **A3 (binom при пустом пуле)**: guard уже был — `if 0 < d < n:` при n=0 даёт False,
   деления нет. Добавлен явный комментарий.

**Наружу вышел структурный дефект при проверке MINOR**: блок G0a-гейта стоял ПОСЛЕ
`if args.smoke: return` — в смоуке гейт вообще не исполнялся, vacuous-ветка
(tpu_n==0) не была протестирована (AGENTS.md чек-лист #2). Блок перенесён ДО
smoke-return; смоук теперь проверяет vacuous-ветку:
`G0a_tpu_curves: нет TPU-доков в подмножестве (smoke) — гейт vacuous (skipped=0)`.

Перепуск (ветки гейта исполнялись — AGENTS.md чек-лист #7): новый store
`16c9a58f26d13e6c` (sha match, gates ALL PASS). Все verdict-секции **бит-в-бит
идентичны** предыдущему артефакту `190439a0` (сравнение выполнено): G0a n=32
skipped=0 max_abs_dev=2.083e-04; H1 refuted; H2 confirmed (30/41 vs 0/51, p=2.18e-07);
k1 K96 0.0738/0.3305 (4.48×). NaN/inf: нет; manifest claim_sha: OK.

EXEC_DONE
