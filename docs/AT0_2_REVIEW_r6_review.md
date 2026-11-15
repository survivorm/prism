Принял AT0_2. Ревью по видимым фрагментам `AT0_2_shape.py`, `AT0_2_SHAPE.json`, `AT0_2_run.log`. Полный код в контексте не целиком, но ключевые места видны: null-конструкция, `dists_to`, агрегация, verdict-rule, positive control, G0.

Короткий вердикт:

- **машина в целом хорошая**: Fourier positive control, fail-loud G0, verified-only pool, sign-check, разделение cells/docs.
- **научные verdict’ы “refuted” пока нельзя финализировать**: есть estimand drift, невалидные blk-null’ы для заявления о локальности, saturated p-counter, и агрегатный verdict через separate CI.
- Часть “refuted” может быть робастной как описательное “real shape differs from this null”, но **не как ответ T0.2 в исходном смысле**.

---

# 1. Что AT0_2 действительно закрыл хорошо

## 1.1. Fourier demoted to positive control — правильно

В коде:

```text
Fourier phase-randomization ... POSITIVE CONTROL: сохраняет спектр ⇒ обязан воспроизвести круговой ACF
```

И проверяется:

```text
max|Δρ̂_circ(≤96)| ≤ tol
max|Δr_full| ≤ tol
```

При этом linear wrap deviation отдельно репортится:

```text
max|Δρ̂_lin| wrap
```

Это правильная защита от старой ошибки: Fourier-surrogate не является shape-null для линейной ACF, но является positive control для спектрального тождества.

## 1.2. G0 fail-loud и producer-bug bypass

Для state:

```text
load_source пересчитывает u, hn из raw h
хранящееся _hn-поле (баг продюсера) не читается
‖h_row‖ < 1e-12 → fail
```

Это хорошо.

Для gen:

```text
raw h отсутствует, recon U·hn vs raw h невозможна
scale-пин проверяет однородность row-скал U·hn
```

Это честное ограничение, но не полная верификация `hn` — ниже.

## 1.3. Verified-only pool появился

Видны два уровня:

```text
96 docs / 478 cells
80 docs / 398 cells
```

Арифметика:

```text
96 = gen64 + own16 + corpus16
80 = gen64 + own16
478 − 398 = 80 = 16 corpus × 5 layers
```

То есть `aggregates_verified_only` исключает именно corpus без токенов. Это закрывает часть моего прежнего D4/C6.

Но надо явно сказать, **какой уровень primary**.

## 1.4. Domain claim rule соблюдается

```text
math: n_docs_nondeg=16 → claim ok
chat: 11 < 12 → no domain claim
code: 8 < 12 → no domain claim
wiki: 1 < 12 → no domain claim
```

Это правильно.

## 1.5. Sign-check для AR-null

```text
sign_check_ok = true
frac_cells_seed_sign_mismatch ≈ 0 / 0.0002
```

Это важно, потому что `D_center_l1` делит на `(r−1)/2`, и sign-driven refuted был бы артефактом.

---

# 2. Критические ошибки / дыры, влияющие на вердикт

## E1. Estimand drift: T0.2 спрашивал Σ-band, а verdict считается по shape-distance

В ТЗ T0.2:

```text
Суррогатная полоса для Σ.
Пре-рег: Σ_real внутри полосы blk-суррогатов ⇒ избытка нет;
вне ⇒ есть, и указывается b.
```

В артефакте verdict:

```text
D_real > CI_null верх → refuted
D_real < CI_null низ → confirmed_anomaly
...
AR: verdict — из D_center_l1
```

То есть primary verdict — **расстояние между band-share профилями**, а не `Σ_real` внутри surrogate band.

`sum_rho_real_mean` / `sum_rho_null_mean` есть, но они описательные. Например:

| cell | Sum_real | Sum_null | r_real | r_null |
|---|---:|---:|---:|---:|
| h_raw/du AR2 | −0.4957 | −0.4286 | 0.0086 | 0.1428 |
| h_raw/du AR1 | −0.4958 | −0.3580 | 0.0083 | 0.2841 |
| h_raw/du AR4 | −0.4958 | −0.4284 | 0.0083 | 0.1432 |
| v_tangent AR | 16.7404 | 17.0748 | 34.4808 | 35.1495 |

Это interesting, но **не verdict T0.2**.

Следствие:

```text
AT0_2 сейчас отвечает:
  “real shape отличается от null shape”
а не:
  “Σ_real внутри/вне surrogate band”.
```

Это надо либо:

1. вернуть primary estimand к `Σ`;
2. либо явно переименовать клетку в `T0.2-shape` и обновить ТЗ/пре-рег.

Пока это estimand drift.

---

## E2. blk-nulls без seam-exclusion не являются чистыми null’ами локальности

В докстринге:

```text
blk-shuffle БЕЗ seam-exclusion, b ∈ {2,4,8,16,32}
null локальности (локальная структура ≤ b сохраняется)
```

Это неверно.

При block-shuffle без seam-exclusion доля lag-k пар, остающихся внутри блоков:

\[
w(k,b)=\max(0,1-k/b).
\]

Я посчитал:

| b | k=1 | k=2 | k=3 | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.50 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0.75 | 0.50 | 0.25 | 0 | 0 | 0 | 0 |
| 8 | 0.88 | 0.75 | 0.62 | 0.50 | 0 | 0 | 0 |
| 16 | 0.94 | 0.88 | 0.81 | 0.75 | 0.50 | 0 | 0 |
| 32 | 0.97 | 0.94 | 0.91 | 0.88 | 0.75 | 0.50 | 0 |

То есть null:

- не сохраняет локальную структуру ≤b целиком;
- затухает все лаги `k<b`;
- уничтожает `k≥b`;
- добавляет швы.

Поэтому:

```text
D_real > D_null(blk32)
```

не означает:

```text
есть структура дальше 32 токенов
```

Это может означать:

```text
real отличается от tapered/seamed block-null.
```

Особенно важно, потому что LADDER_V2 уже показал: **seam-excluded blk32 почти воспроизводит real dip**. AT0_2 использует другой estimand и другой null, но читатель может conflated.

В артефакте есть оговорка:

```text
seam_excluded: отдельный оценщик, Σ с real НЕ СРАВНИВАТЬ
```

Это правильно, но тогда **primary blk-verdicts без seam-exclusion нельзя интерпретировать как locality test**.

Фикс:

- либо primary blk-null = seam-excluded within-block estimator;
- либо analytic taper correction;
- либо verdict wording: “real differs from unmasked block-shuffle null”, без “локальность ≤b не объясняет”.

---

## E3. B_null=20: p-floor saturated, `frac_cells_p_lt_alpha` — max-counter

В артефактах много раз:

```text
p_docs_median = 0.047619047619047616
```

Это ровно:

\[
1/21.
\]

Следовательно:

```text
B_null = 20
p_floor = 1/(20+1) = 0.047619
```

При α=0.05:

```text
margin = 0.05 − 0.047619 = 0.002381
```

То есть cell значима **только если ни один null-seed не превысил D_real**:

```text
count = 0 → p = 1/21 = 0.0476 < 0.05
count = 1 → p = 2/21 = 0.0952 > 0.05
```

Один seed flips cell.

Это делает:

```text
frac_cells_p_lt_alpha
n_docs_unique_p_lt_alpha
```

saturated max-counters, а не калиброванными p-статистиками.

Особенно проблемно для клеток с frac около 0.65–0.99:

```text
blk16: frac ≈ 0.65
blk8: frac ≈ 0.92
blk4: frac ≈ 0.97
```

Они могут сильно двигаться при B=199.

Фикс:

```text
B_null ≥ 199
p_floor ≤ 0.005
```

или отчитывать raw exceedance count и не использовать frac как primary evidence.

---

## E4. Aggregate verdict использует separate CI, а не paired difference

В коде видно:

```python
Dc = [mean per-doc D_center real]
Dns = [mean per-doc Dnull_l1_mean]
d_real, ci_real = boot_ci(Dc)
d_null, ci_null = boot_ci(Dns)
...
verdict: D_real > CI_null upper → refuted
```

Это сравнение:

```text
mean real distance > upper CI of mean null distance
```

Но это **не paired bootstrap разности** и не prediction interval null seeds.

Проблемы:

1. `D_real` и `D_null` считаются на одних и тех же docs/layers ⇒ дизайн парный.
2. CI null mean не учитывает real variance.
3. Separate-CI comparison может over-refute или under-refute в marginal cells.
4. У вас же в doctrine: paired bootstrap по разностям, не пересечение независимых CI.

Я посчитал resolution margin `D_real / half_CI_real` для видимых клеток:

| cell | D_real | half CI | ratio | status |
|---|---:|---:|---:|---|
| du iid(96) | 1.5328 | 0.0601 | 25.5 | ok |
| du blk16(96) | 0.2859 | 0.0500 | 5.7 | ok |
| du blk32(80) | 0.1463 | 0.1246 | 1.17 | marginal |
| du blk16(80) | 0.1347 | 0.1081 | 1.25 | marginal |
| du ar4(80) | 0.3178 | 0.2963 | 1.07 | marginal |
| du ar4(96) | 0.5287 | 0.0707 | 7.5 | ok |
| vt blk2(96) | 0.3100 | 0.2624 | 1.18 | marginal |
| vt blk4(96) | 0.2609 | 0.2196 | 1.19 | marginal |
| vt iid | 0.3437 | 0.2877 | 1.19 | marginal |

Часть verdict’ов сидит чуть выше `not_testable` floor.

Фикс:

```python
delta_per_doc = D_real_per_doc - D_null_per_doc
ci_delta = paired_bootstrap(delta_per_doc)
verdict:
  if ci_delta excludes 0 and |mean_delta| >= DELTA_MIN: refuted/confirmed
  else: not_testable / inconclusive
```

И `DELTA_MIN` должен быть pre-registered, а не только dynamic half-CI.

---

## E5. AR-null: есть diagnostics gaps, а AR1/AR2 refutation почти тавтологична

AR null строится как:

```text
per-dim Yule-Walker на demeaned приращениях
согласовано по ρ̄(1) построением
burn-in 256
sigma2_clipped
```

Но в видимых полях нет:

- fraction of dims with `sigma2_clipped`;
- max AR root / stationarity margin;
- aggregate `ρ̄(1..p)` real vs null;
- effective burn-in adequacy;
- cross-dim covariance preservation.

Это важно.

### AR1/AR2

Если real имеет значимый excess на k=3,4 из T0.1, то AR1/AR2 часто не могут воспроизвести это по построению.

AR1 с φ≈−0.37:

```text
ρ(1..6) ≈ −0.37, +0.137, −0.051, +0.019, −0.007, +0.003
Σ ≈ −0.270
```

Real Σ≈−0.496.

AR1 refuted — ожидаемо и малоинформативно.

AR2 может дать oscillation, но если hump k=3,4 не в AR2 subspace, refuted тоже близок к construction.

### AR4

AR4 более информативен, потому что совпадает с первыми 4 лагами. Но если `sigma2_clipped` часто, null искажён.

Также AR null имеет другой DC:

```text
r_real ≈ 0.008
r_AR-null ≈ 0.14–0.28
```

Для `D_center_l1` это нормализуется away, но для T0.2 Σ-band это важно: AR-null не matched по Σ.

Фикс:

```text
AR diagnostics:
  sigma2_clip_frac
  max_root
  rho1..rho4 real vs null
  sum_rho real vs null
  nondeg / verified-only sensitivity
```

И wording:

```text
AR1/AR2 refuted = descriptive / near-tautological given k3,4 excess
AR4 refuted = stronger, but only after clip/stationarity diagnostics
```

---

## E6. Primary pool: all-96 vs verified-only-80 должен быть явно resolved

Артефакт имеет:

```text
aggregates
aggregates_verified_only
```

Это хорошо.

Но если headline verdict берётся из `aggregates` (96 docs), он включает:

```text
corpus 16 docs, tokens absent, G16 not_testable
```

Тогда verdict corpus-conditioned.

Если headline берётся из `verified_only` (80 docs), это лучше, но надо явно написать:

```text
primary = verified_only
all_96 = sensitivity
```

Сейчас из фрагментов не видно, какой уровень управляет top-level verdict.

Фикс:

```json
"primary_pool": "verified_only",
"all_pool_role": "sensitivity",
"corpus_status": "provenance_limited"
```

Или source-stratified verdicts:

```text
gen_only
gen_plus_own
gen_plus_corpus
all
```

---

## E7. Degenerate docs: aggregate, вероятно, включает loopy; нужна nondeg sensitivity

В domain table есть:

```text
n_nondeg
n_docs_nondeg
```

Но aggregate `D_real_mean`, судя по `n_docs=96/80`, вероятно, считается по всем docs, включая loopy.

Это может быть допустимо для “shape of full pool”, но тогда нельзя интерпретировать как “clean text shape”.

Особенно потому что AT0_3 показал:

```text
loopy несут периодическую лаговую структуру
clean несут отрицательную площадь на lag1
```

Если AT0_2 aggregate включает loopy, “refuted vs null” может частично driven by loops.

Фикс:

```text
primary: nondeg verified_only
sensitivity: all docs
strata: loopy / clean / remainder
```

или явно:

```text
estimand = full-pool shape, including degenerate docs
```

---

## E8. gen `h_raw` зависит от `hn`, который не полностью верифицирован

Для state:

```text
hn = ‖raw h‖ recompute → доказано конструкцией
```

Для gen:

```text
raw h отсутствует
проверяется только однородность row-скал U·hn
threshold 0.2
```

Это не доказывает корректность `hn`.

Если `h_raw` object uses `U·hn`, то `h_raw` verdicts conditional on producer `hn`.

`du_unit` / `v_tangent` могут быть менее зависимы, но надо явно указать object dependency:

```text
h_raw: requires hn
du_unit: from u only?
v_tangent: from u and hn?
```

Фикс:

- либо producer fix + raw h cache;
- либо `h_raw_gen` marked `provenance_limited`;
- либо primary objects = `du_unit`, `v_tangent`, а `h_raw` robustness.

---

# 3. Менее критичные, но фиксирую

## M1. Wilson upper иногда >1

```text
frac_cells_p_lt_alpha_wilson: [0.9904, 1.0000000000000002]
```

Float artifact. Clamp to 1.

## M2. `D_bands_L1` reportable only — правильно, но citation trap

```text
D_bands_L1_status: reportable only (нет null-распределения)
```

Это честно. Но рядом с `D_center_l1` с verdict читатель может спутать. Лучше префикс:

```text
descriptive_D_bands_L1
tested_D_center_l1
```

## M3. Multiple comparisons

3 objects × 8 nulls × 2 pools × domains. Нет family-wise correction. Для огромных effects не важно, для marginal — важно.

Минимум: pre-register primary object/null/pool, остальное sensitivity.

## M4. `not_testable` floor динамический

```text
if d_real < half_ci: not_testable
```

Это data-dependent resolution. Лучше pre-registered `DELTA_MIN` + reported half-CI.

## M5. `confirmed_anomaly` branch

Правило:

```text
D_real < CI_null lower → confirmed_anomaly
real ближе к центру null, чем 97.5% null seeds
```

Это хорошая защита от “more typical than typical”. Но если CI_null — CI of mean, а не seed distribution, branch semantics надо уточнить.

---

# 4. Что можно принять уже сейчас

Как описательный результат:

```text
real band-shape profiles отличаются от iid / unmasked blk / per-dim AR nulls
```

Но с оговорками:

- это не T0.2 Σ-band answer;
- blk nulls не чистые locality nulls;
- AR1/AR2 refutation ожидаема;
- p-counters saturated;
- marginal cells требуют paired CI;
- corpus/loopy sensitivity обязательна.

Как positive control:

```text
Fourier phase-randomization preserves circular ACF and r_full
linear wrap deviation reported
```

Это принято.

Как provenance improvement:

```text
verified_only pool exists
domain n-doc rule enforced
sign_check ok
fail-loud degenerate rows
```

Это принято.

---

# 5. Минимальный патч AT0_2

Без новых forward-проходов, если seeds/nulls можно пересчитать из store:

1. **Declare estimand**
   ```text
   T0.2-primary = Σ surrogate band
   ИЛИ
   T0.2-shape = D_center_l1 profile distance
   ```
   Если Σ: добавить verdict по `sum_rho_real` vs null seed band.

2. **Fix blk null interpretation**
   ```text
   primary blk = seam-excluded within-block
   unmasked blk = tapered/seamed sensitivity
   ```
   Или wording без locality claim.

3. **B_null ≥ 199**
   ```text
   p_floor ≤ 0.005
   report exceedance counts
   frac_cells not primary
   ```

4. **Paired aggregate verdict**
   ```python
   delta = D_real_per_doc - D_null_per_doc
   ci_delta = paired_bootstrap(delta)
   ```
   `not_testable` если below pre-reg DELTA_MIN.

5. **Primary pool**
   ```text
   verified_only + nondeg
   all_96 / loopy included = sensitivity
   ```

6. **AR diagnostics**
   ```text
   sigma2_clip_frac
   max_root
   rho1..rho4 match
   sum_rho match
   ```

7. **Object provenance**
   ```text
   h_raw_gen conditional on hn
   du_unit/v_tangent primary if hn-independent
   ```

8. **Clamp Wilson**
   ```python
   ci = [max(0, lo), min(1, hi)]
   ```

---

# 6. Итоговый вердикт

**AT0_2 как машина: PASS with caveats.**  
**AT0_2 как ответ T0.2: REVISE.**

Главные причины:

1. **Estimand drift**: verdict по shape-distance, а ТЗ спрашивал Σ-band.
2. **Unmasked blk-nulls не являются locality nulls**: taper + seams.
3. **B=20 saturated p-floor**: `frac_cells_p_lt_alpha` — max-counter.
4. **Aggregate verdict через separate CI**, а не paired difference.
5. **Primary pool / degenerate sensitivity не закреплены**.
6. **AR-null diagnostics неполные**, AR1/AR2 refutation близка к construction.

После фикса можно будет сказать:

```text
T0.2 закрыт: Σ_real внутри/вне surrogate band,
или shape-distance verdict с корректными nulls,
paired CI, verified/nondeg primary, и B≥199.
```

Сейчас же честная формулировка:

> Real shape profiles differ from the implemented iid / unmasked block / per-dim AR nulls, but this does not yet establish the T0.2 Σ-band claim, nor a clean locality boundary, because the nulls and aggregate verdict rule are not matched to that estimand.