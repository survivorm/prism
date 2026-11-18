# AT8_2b — ТЗ (пре-рег): второй стимул (loop2) — «повторение» vs «фраза»

**Статус: пре-рег (TZ v1.0, 17.11).** Следующий шаг линии AT8_2-R2 по вердикту
ревью r4 («РАСШИРЯЕМ») и ответу архитектора (`docs/AT8_2_review_2.md`,
sha256 `abda6f2a…`) — финальной позиции по R2-линии.

## 1. Одна строка и вопрос

R2-вердикт `NO_LOOP_SPECIFIC_EFFECT` (machine r2d) bounds контраст
`e_spec = loop − fillerA`, который одновременно несёт **«запоминаемое vs
новое»** И **«повторение vs не-повторение»** (TZ R2 §10, r2d-1.1). 2b
разделяет эти гипотезы вторым стимулом — **другой** фразой, повторённой n раз:

- **H_phr (фраза-канал):** специфическое *содержание* повторяемой фразы
  (запоминание именно этой фразы) создаёт пост-эффект.
- **H_rep (повторение-канал):** *повторение любого* текста (не
  содержимое) создаёт пост-эффект.

Вопрос: какой из каналов (ни один / один / оба) несёт специфический локальный
пост-эффект поля на разрешении филлер-вариативности (δ R2)?

## 2. Главный сдвиг дизайна (от R2)

- **Две фразы** (loop1 = R2-фраза, loop2 = новая, §3) → новый контраст
  `e_phr = loop1 − loop2`: обе стороны — блоки 9n повторов, **идентичные
  позиции и длина** → чистый контраст содержимого фраз (без конфаунда
  «вставка/сдвиг» из R2-M2).
- **R2-store заморожен и переиспользуется**: ctrl, loop1 (R2 `loop`),
  fillerA, fillerB — читаются из `results/AT8/AT8_2_R2_store.json` (sha §10),
  новые форварды — **только loop2: 14 доков × 4 дозы = 56** (R2: 182).
  Позиционная тождественность гарантирована построением: те же доки, тот же
  `cut`, тот же `L_n = 9n`, те же филлеры (тот же `filler_map`).
- Вердикт-машина R2 (per-doc δ, CI-правила, первичная доза, парный
  inside-гейт, finite-гейт, doc-bootstrap) — **без изменений**; меняется
  только набор контрастов и дерево вердикта (§11).

## 3. Стимулы

- **loop1 (зафиксирован R2):** `" The class will be held at the park."` —
  `PHRASE1_IDS = [561, 523, 668, 381, 5467, 506, 279, 5926, 13]` (9 токенов;
  13 = «.»).
- **loop2 (новый, зафиксирован ЭТИМ пре-регом):**
  `" Three new buses wait near a quiet station."` —
  `PHRASE2_IDS = [14095, 491, 32275, 3655, 3043, 264, 11012, 7803, 13]`
  (9 токенов; верифицировано токенизатором Qwen3.5-0.8B при подготовке
  пре-рега).
- **Критерии выбора phrase2 (пререг, не постфактум):** 8 слов + «.» =
  9 токенов (ровно как phrase1 → тот же L_n на всех дозах → те же филлеры);
  **пересечение контент-токенов с phrase1 = ∅** (единственный общий токен —
  13 «.»); простое декларативное предложение с общим регистром (как phrase1).
  **Фраза НЕ выбирается по CE/эффекту** — выбор по форме, зафиксирован до
  прогона; обучаемость (block-CE) — диагностика (§9), не фильтр.
- **S_stimulus (гейт):** `L_n(n) = 9n` для обоих лупов на каждом n;
  `assert tk(PHRASE_i) == PHRASE_i_IDS` в `run_main`; `phrase_token_check`
  (оба ID-списка) в claim/манифесте (F1).

## 4. Корпус и пул (заморожены, из R2)

- **Тот же пул R2: 14 доков** (wiki 4 / math 2 / code 4 / chat 4), те же
  `pool_fps`, те же `docs_sha256` (token-fingerprint, M4), те же `cut = T//2`,
  тот же `filler_map` (same_pool_cross_doc, M3). Пул **не пересобирается** —
  2b продолжение того же эксперимента (сравниваемость с R2 обязательна).
- Пул-гейт 2b: из R2-store `n_docs = 14`, 4 домена, ≥2/домен — **проверка**
  (read-only), не сборка; несоответствие → NOT_RESOLVED (причина в артефакте).

## 5. Геометрия (идентична R2)

```
T      = длина контроля (из R2-store, 192..512)
cut    = T // 2            (из R2-store)
L_n    = 9 * n             (объединение phrase1/phrase2 — равные длины)
inj_index(cut, n, d) = cut + L_n + d     (loop/filler)
ctrl_index(d)        = cut + d           (ctrl)
H      = 64
near   = d ∈ [0, 15]
far    = d ∈ [48, 63]
```

Post-ряды loop1/fillerA/fillerB — из R2-store (option A, M1: шов
`cos_raw[cut+L_n−1]` вне окна — без изменений). Post-ряды loop2 — те же
индексы (новое условие, тот же cut/L_n).

## 6. Слои

Тот же набор R2: `{L6, L12, L18, L19, L23}`, primary — **L23 по
преемственности** (TZ R2 §4, r2d-1.2: теория не предсказывает максимума на
readout-слое, данные R2 не выделяют ни один слой — слои диагностика, тег-
правила §11).

## 7. Условия и контрасты

Условия на (док, n): `ctrl`, `loop1`, `loop2`, `fillerA`, `fillerB`
(4 из 5 — R2-store, loop2 — новый прогон). Контрасты (per-позиция, per-слой,
`cos_raw` lag-1):

```
e_spec1 = loop1 − fillerA          (R2, заморожен — e_spec R2)
e_spec2 = loop2 − fillerB          (новый)
e_null  = fillerA − fillerB        (R2, заморожен)
e_phr   = loop1 − loop2            (новый; phrase-канал)
e_rep   = (e_spec1 + e_spec2) / 2  (новый; repetition-канал)
```

- **e_phr**: обе стороны — повторения (9n токенов), позиции и длины
  идентичны → контраст содержимого. **Нуль: `e_null`** — «два разных
  НЕ-повторённых блока той же длины»: ровно та же типология разницы
  («разное содержимое», без повторения) → одна статистика на обеих сторонах
  сравнения (AGENTS чк-лист п.1, F4).
- **e_rep**: «повторение vs филлер» в среднем по двум фразам (поз.
  идентичны: loop1 и loop2 имеют тот же cut/L_n; fillerA/B — те же).
- `e_spec1/e_spec2` — диагностика (дозовые кривые per-фраза); в вердикт не
  идут (первичные — e_phr/e_rep).
- **Шов (M1)** — диагностика для loop2: `seam2(n) = cos_raw_loop2[cut+L_n−1]
  − cos_raw_fillerB[cut+L_n−1]` (R2: −0.055…−0.081, знак противоположен
  near[0] — контроль, что новый стимул ведёт себя как старый).

## 8. Разрешение δ (F4 — та же статистика, что и сравниваемые сводки)

```
per doc i, dose n:  near_null_i,n = p90 |e_null_i,n(d)|, d ∈ [0,15]
                    far_null_i,n  = mean |e_null_i,n(d)|, d ∈ [48,63]
δ_near = 95th pct {near_null_i,n}  (56 значений, 14 док × 4 дозы)
δ_far  = 95th pct {far_null_i,n}
```

- Вычисляется 2b-машиной **из R2-store** (единый пайплайн с R2). e_null
  заморожен → **ожидаемо bit-идентично R2: δ_near = 0.1480, δ_far = 0.0332**
  — **fail-loud**: отклонение > 1e-6 → стоп (store повреждён/подменён).
- `δ_min` неконечен/нулевой → NOT_RESOLVED (гейт S_null, как R2).
- Один δ для e_phr и e_rep (оба сравниваются с нулём «разных блоков» — e_null;
  свойственный нуль e_phr — именно e_null, §7).

## 9. Доза

- **Первичная доза: `n = 16` пререгистрирована** (F2; скан по дозам —
  диагностика; если CI_low вторичной дозы > δ — в причине reported max,
  вердикт по первичной — конвенция r2c).
- **I_loop2(n) = Σ CE по токенам блока loop2** (новое; I_loop1 — из R2-store,
  заморожен: 59.2/61.7/63.8/64.6 nat).
- **Дозовый тег loop2** (по док-медианным, как R2 §9): `DOSE_FLAT` /
  `DOSE_SATURATING` / `DOSE_MONOTONIC` / `DOSE_NOT_RESOLVED` (NaN →
  `DOSE_NOT_RESOLVED`, isfinite-гейт, r2d-2.3).
- **Learnability-parity (диагностика, тег):** если `block-CE_loop2(16) /
  block-CE_loop1(16) > 3` ИЛИ `< 1/3` (док-медианы) → тег `LEARN_ASYMM`
  (phrase2 обучается принципиально иначе → e_phr-контраст смешан с
  разницей обучаемости; вердикт не меняется, в причине). Пререг-ожидание:
  block-CE phrase2(16) ~ 0.3–1.0 nat (как phrase1 0.38–0.51 — простая
  декларативная фраза).

## 10. Гейты (до вердикта)

1. **R2-store integrity (новый, главный):** sha256
   `results/AT8/AT8_2_R2_store.json` == **`78419a69a2edbc767248b3a8f9512e1961
   ea82c154970772c66c35e2a42b16c0`**; `__config__`-fingerprint (model, layers,
   n_dose, phrase_ids, docs_sha256, pool_fps, H) == константы 2b; `n_docs=14`,
   4 домена, ≥2/домен. Провал → стоп (SystemExit), не NOT_RESOLVED: 2b без
   целого R2-store не имеет смысла.
2. **S_stimulus:** phrase_token_check (оба), L_n = 9n, конечность.
3. **Finite-гейт (M5):** ce + cos_raw/cos_state/norm_delta по всем 5 условиям
   × слоям (loop2 — полный; R2-условия — spot-проверка конечности) + все 5
   контрастов. Провал → NOT_RESOLVED.
4. **G_inside (F5, парный — основной):** на n=16, per doc:
   `loop1 < fillerA ∧ loop1 < fillerB ∧ loop2 < fillerA ∧ loop2 < fillerB`.
   Гейт: ≥ `ceil(0.75 × 14) = 11` доков; иначе `INSENSITIVE_CELL` (вердикт
   про эффект не выносится). Абсолютные пороги (loop < 1.0, fillerA ≥ 2.0) —
   вторичная диагностика (`inside_abs_pass`): R2 — 13/14 (miss chat_2
  fillerA 1.982), 2b проверяет loop2.
5. **S_null:** δ конечен (> 0), §8.

## 11. Сводки и вердикт

Сводки (L23, каждый n) — те же агрегаторы R2 (p90 near / mean far; асимметрия
зафиксирована r2d-1.3 — в данных 2b та же оговорка):

```
near_phr_n = mean_i ( p90 |e_phr_i,n(d)|,  d ∈ [0,15] )     и far_phr_n (mean, [48,63])
near_rep_n = mean_i ( p90 |e_rep_i,n(d)|,  d ∈ [0,15] )     и far_rep_n (mean, [48,63])
```

(агрегация — среднее по докам; единица бутстрапа — док, §12.)

### Вердиктный словарь (2b)

- `INSENSITIVE_CELL` — G_inside < 11/14 (клетка не различает вставки/лупы).
- `NO_2B_SPECIFIC_EFFECT` — и e_phr, и e_rep не превышают δ **по первичной
  дозе** (max по дозам — в причине, диагностика). Чтение: ни «запоминание
  конкретной фразы», ни «повторение как таковое» не создают измеримого
  пост-эффекта сверх филлер-нуля — сильное чтение R2-`NO_LOOP`.
- `PHRASE_SPECIFIC` — только e_phr значим по первичной дозе. Чтение: эффект
  в содержимом конкретной фразы (запоминание), не в повторении.
- `REPETITION_SPECIFIC` — только e_rep значим. Чтение: эффект в повторении
  любого текста.
- `BOTH_SPECIFIC` — оба значимы: тег по каждому (substructure).
- `NOT_RESOLVED` — мало доков / противоречие слоёв / far между «≤δ с CI>0» и
  «>δ» / δ велик.

### Substructure (для каждого значимого контраста C ∈ {phr, rep}, первичная доза)

```
CI_low(C_near − C_far) > 0  И  CI_high(C_far) ≤ δ_far  → _DECAYS
CI_low(C_far) > δ_far                               → _PERSISTS
иначе                                                 → _AMBIGUOUS (→ NOT_RESOLVED, если единственный значимый)
```

Композит: `<исход> + <substructure phr> + <substructure rep> + <слойный тег>
+ <дозовый тег loop2> [+ LEARN_ASYMM] (n=16)`.

### Слойный тег (как R2 §10, на n=16; «значим» = `near_L > δ_L` по
соответствующему контрасту; значимость по ОБЪЕДИНЕНИЮ контрастов)

`FIELD_RESPONSE` / `READOUT_ONLY` / `INTERNAL_ONLY` / `LAYER_INCONSISTENT`
(правила — TZ R2; на 2b — по контрасту, который значим; оба значимы —
объединение).

### Дерево вердикта (порядок жёсткий)

```
1. Гейты §10: R2-store integrity; S_stimulus; finite; δ конечен.
   провал → NOT_RESOLVED (причина в артефакте) [R2-store integrity → стоп].
2. G_inside < 11/14 → INSENSITIVE_CELL.
3. Первичная доза n=16, L23:
   S_phr := CI_low(near_phr) > δ_near ;  S_rep := CI_low(near_rep) > δ_near
   S_phr ∧ S_rep            → BOTH_SPECIFIC (+substructure каждого)
   S_phr only               → PHRASE_SPECIFIC (+substructure)
   S_rep only               → REPETITION_SPECIFIC (+substructure)
   neither                  → NO_2B_SPECIFIC_EFFECT
   (substructure _AMBIGUOUS по единственному значимому → NOT_RESOLVED)
4. + слойный тег, 5. + дозовый тег loop2, 6. + LEARN_ASYMM (если).
```

Интерпретация положительных исходов — о релаксации/персистентности канала,
НЕ о солитонной упругости (как R2).

## 12. Статистика

- Единица ресемплинга: **документ**; `doc-bootstrap, B = 2000`, seed 0.
- Ресемплируются: `near_phr/far_phr/near_phr−far_phr`, `near_rep/...` — по
  **одним** докам (пара́ность, r2d-2.5: диагностика не независимость).
- CI95 (перцентили 2.5/97.5); значимость near−far — CI целиком > 0.
- `prism.nulls_and_stats` (paired_bootstrap) + numpy.

## 13. Артефакт

- `results/AT8/AT8_2b.json` (claim): вердикт + сводки + CI + гейты +
  `machine_subrevision` + r2_store_sha256 + pool_rule/pool_per_domain
  (r2c-конвенция) + seam2 + I_loop1/I_loop2 + inside_pass/inside_abs_pass.
- `results/AT8/AT8_2b_store.json` (новый, только loop2): на (док, n):
  `ce[t]`, `ce_block`, `i_loop`, per-слой `cos_raw/cos_state/norm_delta` +
  `__config__` (model, layers, n_dose, phrase1_ids, phrase2_ids,
  r2_store_sha256, docs_sha256, pool_fps, H, revision). Оценка ~3.8 MB
  (<10 MB → **в git**; если >10 MB — gitignored + бэкап, как R2).
- `results/AT8/AT8_2b_series.npz` — строится **из обоих сторов** (R2 + 2b,
  M7; 238 ключей). ~13 MB > 10 MB → gitignored, бэкап
  `/data/prism_backups/at8_2b/`, sha256 в манифесте.
- `results/AT8/AT8_2b_manifest.json`: r2_store_sha256 (зафиксирован §10),
  store/npz/log sha256, script_sha256, code-fingerprint, model, attn_impl,
  threads, seed, **full_run-блок** (n_fwd = 56 факт из лога+стора, wall из
  DONE-строки, wall_source=log — r2c/r2d-конвенция).
- `logs/AT8/AT8_2b_run.log` — формат-контракт (r2d-2.8): строка на форвард
  `  <док> n=<n> loop2: len=… CE=… [block_CE=…]` (56 строк; `\bCE=` не
  матчит `block_CE=`), финальная `AT8_2b_DONE verdict=<вердикт>
  mode=<мод> (wall <сек>s)`. Trackio run: `at8_2b_r1` (config со
  started_at/revision/script_sha/seed; стадии pool-check/forward/analysis;
  status ok/crash в finally).

## 14. Цена

- Новые форварды: **56** (14 доков × 4 дозы × loop2). Измеренная скорость R2:
  ~14 с/фд (2596 s / 182, OMP=6, LA≈18) → **~13 мин** + модель/анализ ~3 мин
  → **~15–20 мин CPU** (в 3–4× дешевле R2).
- Запуск: `./run_one.sh at8_2b ...` (setsid nohup, OMP=6, nice-10, watchdog,
  память-гейт avail ≥ 10G). **Одна CPU-задача за раз** (AGENTS).

## 15. Smoke-план (до полного прогона)

1. 1 домен (wiki) × 2 дока × n=2 — 2 форварда loop2:
   - R2-store integrity-гейт проходит (sha, config-fingerprint, 14 доков);
   - δ пересчитан из R2-store: **0.1480/0.0332 в точности** (fail-loud anchor);
   - phrase_token_check для обоих фраз (assert);
   - CE loop2-block < 1.0 (обучаемость phrase2, §9); finite-гейт PASS;
   - `e_phr` конечен, `mean_d e_phr ≈ 0` (после сдвига — два loop'а одного
     порядка); seam2 в диапазоне R2 (±0.05…−0.09, знак как R2);
   - resume: прерванный smoke-прогон доходит до конца (56-план).
2. Синтетические store'ы (§16) → машина даёт ожидаемые вердикты (6/6).
3. Дегенерации: подменённый R2-store (sha-mismatch → стоп, SystemExit);
   NaN в loop2 (finite → NOT_RESOLVED); loop2 = loop1 (e_phr ≡ 0 →
   NO_2B/INSPECT — контроль, что контраст не артефакт).

## 16. Синтетические тесты (6 сценариев, как R2 test_synth)

| Сценарий | e_phr | e_rep | Ожидание |
|---|---|---|---|
| no_effect | ~0 | ~0 | `NO_2B_SPECIFIC_EFFECT` |
| phrase_decays | near>δ, far≤δ, near−far>0 | ~0 | `PHRASE_SPECIFIC_DECAYS` |
| phrase_persists | near>δ, far>δ | ~0 | `PHRASE_SPECIFIC_PERSISTS` |
| rep_decays | ~0 | near>δ, far≤δ, near−far>0 | `REPETITION_SPECIFIC_DECAYS` |
| both | near>δ | near>δ | `BOTH_SPECIFIC_*` |
| insensitive | — | — (inside_pass=3) | `INSENSITIVE_CELL` |

## 17. Пререг-чтения исходов

- `NO_2B_SPECIFIC_EFFECT` — R2-`NO_LOOP` усиливается: и «запоминание фразы»,
  и «повторение» не создают пост-эффекта сверх филлер-нуля (на пре-рег-
  разрешении; отсутствие не доказано — конвенция r2d-1.5).
- `PHRASE_SPECIFIC` — канал: содержание конкретной фразы. Согласуется с
  ответом архитектора «после ~2 повторов фраза выучена» (block CE 0.38–0.51)
  — но CE ≠ пост-эффект; теперь разделены.
- `REPETITION_SPECIFIC` — канал: само повторение (структура, не контент).
- `BOTH_SPECIFIC` — два независимых канала; substructure каждого — в claim.
- `LEARN_ASYMM` — phrase2 обучается иначе phrase1: e_phr читается с
  оговоркой «содержимое + обучаемость».

## 18. Границы (не делать)

- R2-store — **read-only** (sha зафиксирован §10); R2-вердикт/числа не
  пересчитываются, R2-документы (TZ/ответки/results) — история, не
  переписывать.
- Новые форварды — **только loop2 (56)**; ctrl/fillerA/fillerB/loop1 не
  перегоняются (сравниваемость с R2).
- Пул не пересобирается (14 доков R2, тот же filler_map).
- Фраза2 не меняется после начала прогона (пре-рег §3).
- paper.md не трогается; новые слои/дозы/фразы — только отдельным пре-регом.

## 19. Пререг-поля (сводка)

| Поле | Значение |
|---|---|
| phrase1_ids | [561, 523, 668, 381, 5467, 506, 279, 5926, 13] (R2, frozen) |
| phrase2_ids | [14095, 491, 32275, 3655, 3043, 264, 11012, 7803, 13] |
| пул | 14 доков R2 (4/2/4/4), r2_store_sha 78419a69… |
| условия-форварды | loop2 only: 14 × 4 дозы = 56 |
| контрасты | e_phr = loop1−loop2; e_rep = (e_spec1+e_spec2)/2; e_null (нуль) |
| δ | 95th pct per-doc near/far e_null (fail-loud: 0.1480/0.0332) |
| первичная доза | n = 16 (пререг; скан — диагностика) |
| near/far | p90 d∈[0,15] / mean d∈[48,63], H=64, cut+L_n+d |
| слои | {6,12,18,19,23}, primary L23 (преемственность) |
| bootstrap | doc, B=2000, seed 0, пара́ность |
| inside-гейт | парный (4 неравенства) ≥ 11/14 |
| вердикты | NO_2B / PHRASE / REPETITION / BOTH + _DECAYS/_PERSISTS/_AMBIGUOUS |
| цена | ~56 форвардов ≈ 15–20 мин CPU |
