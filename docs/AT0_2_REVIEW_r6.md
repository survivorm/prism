# AT0_2 — ответы на раунд 6 (docs/AT0_2_REVIEW.md, 14.11)

Формат: принято / отклонено с аргументом / самокоррекция. Код — `experiments/AT0_2_shape.py`
(раунд 2, B=199, полный прогон запущен 14.11).

## E1. Estimand drift (Σ-band vs shape-distance)

**Частично отклонено + принято по существу.**

Цитата ревью «в ТЗ T0.2: суррогатная полоса для Σ. Пре-рег: Σ_real внутри полосы blk-
суррогатов ⇒ избытка нет» — **устаревшая**: текущий ТЗ (docs/TZ_add_0-3.md §1.5 T0.2, v1.2/v1.3)
говорит прямо:

> «Σ инвариантна к перестановкам ТОЧНО (G7) ⇒ тест «Σ против полосы» **ВЫРОЖДЕН**.
> Estimand: **ФОРМА профиля при фиксированной сумме**.»

И algebra верна: Σx и c(0) инвариантны к ЛЮБОЙ перестановке строк ⇒ Σ_null(id/blk) ≡ Σ_real
ТОЧНО для всех сидов — «полоса blk-суррогатов» — точка, и «Σ_real внутри полосы» — тавтология.
(Численно подтверждено: sum_invariance_max_dev ≤ 1e-12 в агрегатах, G7-гейт |r_null − r_real|
≤ 1e-9 — был с round 1.)

Принято (что реально было дырой — явная декларация отсутствовала):
1. `estimand` — явный блок в pre-reg и claim: T0.2 = форма при фиксированной сумме; Σ-тест
   вырожден (G7) — записано, а не подразумевается; AR-null — Σ своё, mismatch отчитан
   (sum_real_mean / sum_null_mean / sum_mismatch), с real не сравнивается (ТЗ).
2. AR-null Σ-mismatch (r_real ≈ 0.008 vs r_null ≈ 0.14–0.28) — теперь в каждой AR-клетке
   (r_real, sum_mismatch) и в агрегате.

## E2. blk-null без seam-exclusion — не чистый locality-null

**Принято целиком + найден новый баг (самокоррекция ниже).**

1. Формула w(k,b) = max(0, 1−k/b) верна (доля within-block пар лага k; граница O(1/N)).
   Wording исправлен: blk-вердикт читается как «real differs from unmasked block-shuffle
   null (damped within-block, seamed)» — НЕ как «локальность ≤ b не объясняет». В pre-reg
   и claim: blk — damped/seamed null; locality-прибор — P3-ladder (seam-excluded, AT0_1).
2. Фикс (a) ревью «primary blk = seam-excluded within-block» **алгебраически вырожден**:
   внутри блоков blk-перестановка не меняет НИЧЕГО (блоки целыми), поэтому
   within-block(blk-null) ≡ within-block(real) — D ≡ 0 тождественно. Это теперь
   fail-loud identity-check в каждой клетке (G-seam: max_dev ≤ 1e-12) — он же доказательство
   того, что вариант (a) не может быть null'ом.
3. **Самокоррекция (баг round 1, пойман новый identity-check'ом)**: в `null_blk_seam_excl`
   (rounds 1–5) «within-block» считался разбивкой y по ФИКСИРОВАННОЙ сетке y[i·b:(i+1)·b].
   Короткий последний блок (N % b ≠ 0) после перестановки стоит на случайном месте —
   все последующие блоки в сетке сдвинуты, и «within-block» перестрачивал границы двух
   соседних блоков. b=2 проходил (382 = 2·192 — без короткого блока), b=4/8/16/32 —
   загрязнены (max_dev до 0.41 rel). Итог: опубликованные в rounds 1–5 seam_excluded
   prof_mean для b=4/8/16/32 слегка неверны (второстепенный оценщик, в вердикты не входил;
   y-ряды perm-null'ов не затронуты — они пересчитаны бит-в-бит, см. «репродукция»).
   Фикс: границы по пермутированным блокам (pos-курсор) + assert pos == N.

## E3. B_null = 20 → p-floor 1/21 (saturated counter)

**Принято.** B_NNULL = 199 (p_floor = 1/200 = 0.005; стандарт T0.1 — NNULL=199, p_perm =
0.005 per-test, ТЗ v1.2 header). Полные артефакты пересчитаны (прогон запущен). Дополнительно:
- `exceedance_count` per-cell (raw) + `exceedance_count_hist` в агрегате — raw-счётчики
  отчитаны, frac_cells — вторичное (не primary evidence);
- сиды 0..19 репродуктивны из round 1 (гейт репродукции — см. ниже, «самокоррекция по гейту»).

## E4. Aggregate verdict: separate CI → paired difference

**Принято.** Вердикт теперь по paired delta: delta_d = D_real(doc) − D_null(doc) (per-doc
mean по L, кластер = док), CI — cluster-бутстрап по докам (B=1000, seed=117). Separate-CI
(D_real_mean/_ci, D_null_mean/_ci) — отчитано как вторичное. Marginal-клетки ревью
(ratio 1.07–1.25) теперь решаются CI_delta, а не «D_real чуть выше not_testable floor».

## E5. AR diagnostics

**Принято.** В каждой AR-клетке: `ar_max_root` (spectral radius companion, per-dim),
`n_nonstationary` (|root| > 1), `burn_mix_ratio` = 256·(1−max_root), `sigma2_clip_frac`,
`rho_real_1to4` vs `rho_null_1to4_mean`, `sum_mismatch` (r_real vs r_null_mean). Wording:
AR1/AR2 refuted ≈ тавтология при k3,4-excess real (по построению — AR1/AR2 не могут
воспроизвести hump вне своего подпространства); AR4 — информативен только при ok
diagnostics (max_root < 1, clip_frac низкие).

## E6. Primary pool не закреплён

**Принято.** `primary_pool = "verified_nondeg"` (37 доков), `pool_roles` явный:
- primary: `aggregates_primary` (verified ∩ non-degen);
- sensitivity: `aggregates_verified_only` (80), `aggregates` (all_96);
- `corpus_status = "provenance_limited"` (G16: нет токенов).
Решающий пул при расхождении — primary (записано в G0_sources.note).

## E7. Degenerate (loopy) docs в агрегате

**Принято.** primary исключает degen-доки (43 из 96); `degen_docs_n` + per-doc degen-флаги
в store; `degen_note`: AT0_3 — loopy несут периодическую лаговую структуру, агрегат с loopy
= «full-pool shape», не «clean text shape». Strata loopy/clean/remainder — через per-doc
флаги (отдельный per-stratum bootstrap не добавлен — YAGNI: primary/nondeg покрывает
claim, strata-разрез отчитан численно).

## E8. h_raw_gen зависит от не-верифицированного hn

**Принято.** Per-cell флаг `provenance_limited` (gen h_raw/v_tangent); в claim:
`object_provenance` (h_raw: state верифицирован конструкцией / gen provenance_limited;
du_unit: из U только, hn-независим; v_tangent: U·hn) и `primary_objects` (du_unit — primary
на gen; state-объекты — верифицированы; gen h_raw/v_tangent — sensitivity).

## M1. Wilson > 1 (float artifact)

**Принято.** clamp [0,1] в wilson_ci.

## M2. D_bands_L1 — citation trap

**Принято.** Rename: `tested_D_center_l1`/`tested_D_center_l2` (тестируемые),
`descriptive_D_bands_L1` (reportable only).

## M3. Multiple comparisons

**Принято.** Pre-reg `primary_family`: pool=primary × objects {du_unit + state h_raw/
v_tangent} × nulls {iid, blk32, ar4} = 9 тестов; FWER-коррекция не применяется — объявлено
заранее; остальное — sensitivity.

## M4. not_testable floor динамический

**Принято.** DELTA_MIN = 0.05 — pre-registered (зафиксирован в хедере ДО прогона):
порядок эффекта T0.1 по полосам (dip ≈ 0.02–0.06 на полосу; L1-расстояние по 5 полосам
того же порядка). |Δmean| < DELTA_MIN → not_testable; half-CI отчитан рядом (не замена).

## M5. confirmed_anomaly branch semantics

**Принято.** В paired-формулировке семантика однозначна: confirmed_anomaly = Δmean ≤
−DELTA_MIN и CI_Δ < 0 — real ЗНАЧИМО ближе к центру null, чем типичный null-сид (paired
разность, не separate-CI) — флаг «более типичен, чем типично», НЕ сигнал структуры.

---

## Самокоррекции (итог)

1. **Seam-boundary баг** (E2, п. 3) — latent с round 1, пойман новым identity-check'ом;
   опубликованные seam-prof для b=4/8/16/32 rounds 1–5 — слегка загрязнены (корректны
   заново в round 2).
2. **Гейт репродукции (E3)**: per-seed `Dnull_s = ‖P_s − C‖` зависит от B по построению
   (C — среднее по всем B сидов), поэтому бит-в-бит-сравнение Dnull между B=20 и B=199 —
   неправильный гейт (я сначала так проверил и получил ложный «сбой» max 0.056).
   Правильный гейт — per-seed объекты, НЕ зависящие от B: `r_null`, `DtoN` + real-клетки.
   Проверено на 4 клетках (math_3 L5 h_raw, math_3 L23 v_tangent, chat_1 L11 du_unit,
   own_w0 L17 h_raw): **все бит-в-бит (max|Δ| = 0.00e+00)**; отдельно пересчёт B=20
   текущим кодом воспроизводит store round 1 на всех 20 сидах iid (max|Δ| = 0.00e+00).
3. **E1**: цитата ТЗ в ревью — из более ранней версии; текущий ТЗ v1.2 уже declares
   вырождение Σ-теста. Estimand shape-при-фиксированной-сумме — канон, а не «drift».

## Что не принято (с аргументом)

- **E2-вариант (a)** как primary: «primary blk = seam-excluded within-block estimator» —
  алгебраически D ≡ 0 (блоки нетронуты) — null из него не получится; проверено
  identity-check'ом в каждой клетке (fail-loud).
- **E3-альтернатива** («отчитывать raw exceedance count и не использовать frac как
  primary») — принята ВМЕСТЕ с B=199, а не вместо: raw-счётчики отчитаны, НО p_floor =
  0.005 даёт B=20 некалиброванный (1 seed = 4.76 p.p.) — при ~10–14 ч прогона честнее
  пересчитать (стандарт T0.1 уже был NNULL=199 — раунд 1 просто его не использовал).

## Статус

Код round 2: само-аудит + смоук (RC=0, все гейты включая новый G-seam) + дизайн-тесты
(paired verdict на синтетике: refuted/not_testable/confirmed_anomaly; AR roots; seam
identity при N%b≠0; репродукция) — зелёные. Полный прогон B=199 запущен 14.11
(setsid nohup, OMP=8; бенч ~35 с/клетку → ~10–14 ч). Вердикты round 2 — по завершении.
