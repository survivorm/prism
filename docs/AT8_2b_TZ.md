# AT8_2b — ТЗ (пре-рег): второй стимул (loop2) — «повторение» vs «фраза», факторный дизайн 2×2

**Статус: пре-рег (TZ v1.1, 18.11).** v1.0 (17.11) → v1.1 по ревью r1
(`docs/AT8_2b_review_r1.md`, sha `620d2207…`; ответы —
`docs/AT8_2b_review_r1_answers.md`): факторизация 2×2 {периодичность ×
содержимое} с ортогональными REP/PHR/INT и **собственными нулями** (N_perm,
N_content); shuffled-условия S1/S2/S1'; доз-сплит (PHR primary n=2, REP
primary n=16); cos_raw остаётся primary (замер CE-альтернативы — ответы §4);
INT — отдельный исход; **слияние с AT8_3 отложено** до закрытия 8_3 F1–F5
(ответы §9).

## 1. Одна строка и вопрос

R2-вердикт `NO_LOOP_SPECIFIC_EFFECT` (machine r2d) bounds контраст
`e_spec = loop − fillerA`, который одновременно несёт **«запоминаемое vs
новое»** И **«повторение vs не-повторение»** (TZ R2 §10, r2d-1.1). 2b
разделяет эти гипотезы вторым стимулом — **другой** фразой, повторённой n
раз, и факторным дизайном, где каждый канал меряется контрастом со своим
нулём:

- **H_phr (фраза-канал):** специфическое *содержание* повторяемой фразы
  (запоминание именно этой фразы) создаёт пост-эффект.
- **H_rep (повторение-канал):** *повторение (периодичность) любого* текста —
  при **том же лексическом составе** (сломана только периодичность) —
  создаёт пост-эффект.

Вопрос: какой из каналов (ни один / один / оба) несёт специфический локальный
пост-эффект поля при разрешении филлер-вариативности (нули из shuffled-
условий, §8)?

## 2. Главный сдвиг дизайна (от R2 и от 2b v1.0)

- **2×2 {периодичность × содержимое}:** клетки `L1` (loop1, R2-store), `L2`
  (loop2, новый), `S1`/`S2` (shuffle phrase1×n / phrase2×n — новые), плюс
  `S1'` (вторая независимая выборка shuffle phrase1×n — новый; нужен для
  чистого нуля REP, ответы п.3).
- **Ортогональные контрасты** (коэффициенты ±1/2, попарно
  неперекрывающиеся): `REP`, `PHR`, `INT` (§7); каждый вердиктный контраст —
  со **своим нулём того же семейства** (§8).
- **Дозы: n ∈ {2, 16}** (PHR primary n=2 — контент жив; REP primary n=16 —
  максимальная доза периодичности; n=4/8 не гоняются — R2-плато, §9).
- **R2-store заморожен и переиспользуется**: ctrl, loop1, fillerA, fillerB —
  read-only из `results/AT8/AT8_2_R2_store.json` (sha §10) на n=2 и n=16.
- Новые форварды: **4 условия × 14 доков × 2 дозы = 112** (R2: 182).
- Вердикт-машина R2 (per-doc δ, CI-правила, парный inside-гейт, finite-гейт,
  doc-bootstrap) — сохраняется; меняются контрасты, нули, дозы, дерево
  (§11).

## 3. Стимулы

- **loop1 (зафиксирован R2):** `" The class will be held at the park."` —
  `PHRASE1_IDS = [561, 523, 668, 381, 5467, 506, 279, 5926, 13]` (9 токенов;
  13 = «.»).
- **loop2 (новый, зафиксирован пре-регом v1.0, без изменений):**
  `" Three new buses wait near a quiet station."` —
  `PHRASE2_IDS = [14095, 491, 32275, 3655, 3043, 264, 11012, 7803, 13]`
  (9 токенов; верифицировано токенизатором Qwen3.5-0.8B; пересечение
  контент-токенов с phrase1 = ∅, общий только 13 «.»).
- **Shuffled-условия (новые, зафиксированы ЭТИМ пре-регом):**
  - `L_i(n)` = токен-последовательность `PHRASE_i_IDS × n` (длина `L_n = 9n`);
  - `S_i(n)` = `L_i(n)[π_i]`, где `π_i = default_rng(seed_i).permutation(L_n)` —
    **детерминированная перестановка** (мультисет, длина, позиции — те же;
    сломана только периодичность). Сиды **пререг**: `seed(S1)=1`,
    `seed(S2)=2`, `seed(S1')=3`;
  - `S1'(n)` = второй независимый draw: `default_rng(3).permutation(L_n)` от
    `L1(n)`.
- **Критерии phrase2** — как в v1.0 (форма, не CE/эффект; обучаемость —
  диагностика §9).
- **Гейты S_stimulus + S_shuffle:**
  - `S_stimulus`: `L_n(n) = 9n` для L1/L2/S1/S2/S1'; `assert tk(PHRASE_i) ==
    PHRASE_i_IDS`; **multiset-check**: `sorted(S_i) == sorted(L_i)` для всех
    i (assert в `run_main`; `phrase_token_check`/`shuffle_multiset_check` в
    claim/манифесте);
  - `S_shuffle` (урок F5 по AT8_3 r3): для каждого (S_i, n): (а) **окно не
    периодично с периодом 9** (`¬∀k: S[k] == S[k+9]`); (б) max count любой
    9-граммы в окне `[0, 9n)` — **per-(S_i, n) таблица в артефакте** (не один
    максимум по 144, как в 8_3 v1.2). Провал (а) → NOT_RESOLVED (сид —
    константа пре-рега; автосид нет).

## 4. Корпус и пул (заморожены, из R2)

- **Тот же пул R2: 14 доков** (wiki 4 / math 2 / code 4 / chat 4), те же
  `pool_fps`, те же `docs_sha256`, те же `cut = T//2`, тот же `filler_map`
  (same_pool_cross_doc, M3). Пул **не пересобирается** (сравниваемость с R2).
- Пул-гейт 2b (read-only-проверка из R2-store): `n_docs = 14`, 4 домена,
  ≥2/домен; несоответствие → NOT_RESOLVED.

## 5. Геометрия (идентична R2)

```
T      = длина контроля (из R2-store, 192..512)
cut    = T // 2            (из R2-store)
L_n    = 9 * n             (L1/L2/S1/S2/S1' — равные длины и мультисеты пар)
inj_index(cut, n, d) = cut + L_n + d     (все 5 стимульных условий)
ctrl_index(d)        = cut + d           (ctrl)
H      = 64
near   = d ∈ [0, 15]
far    = d ∈ [48, 63]
```

Post-ряды L1/fillerA/fillerB — из R2-store (option A, M1: шов
`cos_raw[cut+L_n−1]` вне окна — без изменений). Post-ряды L2/S1/S2/S1' — те
же индексы (новые условия, тот же cut/L_n).

## 6. Слои

Тот же набор R2: `{L6, L12, L18, L19, L23}`, primary — **L23 по
преемственности** (как v1.0).

## 7. Условия и контрасты

Условия на (док, n∈{2,16}): `ctrl`, `L1`, `L2`, `S1`, `S2`, `S1'`,
`fillerA`, `fillerB` — 4 из R2-store (ctrl, L1, fillerA, fillerB), 4 новых
(L2, S1, S2, S1').

**Вердиктные контрасты** (per-позиция d, per-слой, `cos_raw` lag-1):

```
REP = ((L1 + L2) − (S1 + S2)) / 2     (главный эффект периодичности)
PHR = ((L1 + S1) − (L2 + S2)) / 2     (главный эффект содержимого)
INT = ((L1 − S1) − (L2 − S2)) / 2     (взаимодействие; report-only)
```

(клетки (L1, L2, S1, S2); коэффициенты ±1/2; попарная ортогональность:
скалярные произведения = 0.)

**Нули** (per-позиция d, per-слой):

```
N_perm    = S1 − S1'                 (шум перестановки: тот же мультисет,
                                      та же структура, другой draw)
N_content = S1 − S2                  (контентное различие без периодичности)
```

Назначение: **REP ← N_perm** (чистый нуль: N_content несёт контентный эффект,
тестируемый PHR — нуль REP не может им быть); **PHR ← N_content**; **INT —
нуля нет** (report-only + диагностика допущения о нулях: значимый INT =
контентный эффект в луповой строке ≠ в непериодической — тег
`INTERACT_SIGNIFICANT`, §11).

**Диагностика (R2-сравниваемость, в вердикт не идут):**

```
e_spec1 = L1 − fillerA               (R2, заморожен — e_spec R2)
e_spec2 = L2 − fillerB               (новый)
e_null  = fillerA − fillerB          (R2, заморожен)
e_rep_legacy = (e_spec1 + e_spec2)/2 (v1.0-определение — диагностика)
```

- **Шов (M1) — диагностика для L2** (как v1.0): `seam2(n) =
  cos_raw_L2[cut+L_n−1] − cos_raw_fillerB[cut+L_n−1]` (R2: −0.055…−0.081;
  контроль, что новый стимул ведёт себя как старый).

## 8. Разрешение δ (F4-урок: нуль — того же семейства, что вердиктный)

```
per doc i, dose n:
  near_perm_i,n   = p90 |N_perm_i,n(d)|,   d ∈ [0,15];   far_perm   = mean|·|, d ∈ [48,63]
  near_content_i,n = p90 |N_content_i,n(d)|, d ∈ [0,15]; far_content = mean|·|, d ∈ [48,63]

δ_REP  = 95th pct {near_perm_i,n}      (28 значений: 14 док × 2 дозы)
δ_far_REP  = 95th pct {far_perm_i,n}
δ_PHR  = 95th pct {near_content_i,n}   (28 значений)
δ_far_PHR = 95th pct {far_content_i,n}
```

- Вычисляются 2b-машиной **из новых форвардов** (S1, S1', S2) — единый
  пайплайн. `δ_min` неконечен/нулевой → NOT_RESOLVED (гейт S_null).
- **Integrity-якорь (fail-loud):** пересчёт из R2-store legacy-δ (e_null) —
  **ожидаемо bit-идентично R2: 0.1480 / 0.0332**; отклонение > 1e-6 → стоп
  (store повреждён/подменён). Это проверка пайплайна, **не** порог вердикта.

## 9. Доза

- **PHR primary: `n = 2`** (пререг). Обоснование (механика насыщения,
  зафиксирована до прогона): I_loop1(n) = 59.2/61.7/63.8/64.6 (n=2/4/8/16,
  док-медианы; +9% — `DOSE_FLAT` R2); кривая обучения из R2-store: r1 CE
  4.66 → r2 0.38 → r3 0.13 nat/tok (n16) — фраза выучена ко 2–3 повтору →
  на n=16 контентное различие L1/L2 минимально; на n=2 — максимально.
- **REP primary: `n = 16`** (пререг) — максимальная доза периодичности (16
  повторов).
- Обе дозы, оба контраста — перекрёстная диагностика (2×2 дозовая таблица в
  claim). n=4/n=8 не гоняются (R2-плато; ответы п.5a).
- **I_loop2(n) = Σ CE по токенам блока L2** (новое; I_loop1 — из R2-store,
  заморожен).
- **Дозовый тег L2** (по док-медианным, как R2 §9): `DOSE_FLAT` /
  `DOSE_SATURATING` / `DOSE_MONOTONIC` / `DOSE_NOT_RESOLVED` (NaN →
  `DOSE_NOT_RESOLVED`, isfinite-гейт, r2d-2.3).
- **Learnability-parity (диагностика, тег):** если `block-CE_L2(16) /
  block-CE_L1(16) > 3` ИЛИ `< 1/3` (док-медианы) → тег `LEARN_ASYMM`.
  Пререг-ожидание: block-CE L2(16) ~ 0.3–1.0 nat (L1: 0.38–0.51). Структурная
  версия того же вопроса — INT (§7, §11).

## 10. Гейты (до вердикта)

1. **R2-store integrity (главный):** sha256
   `results/AT8/AT8_2_R2_store.json` == **`78419a69a2edbc767248b3a8f9512e1961
   ea82c154970772c66c35e2a42b16c0`**; `__config__`-fingerprint (model, layers,
   n_dose, phrase_ids, docs_sha256, pool_fps, H) == константы 2b; `n_docs=14`,
   4 домена, ≥2/домен. Провал → стоп (SystemExit), не NOT_RESOLVED.
2. **S_stimulus:** phrase_token_check (оба), shuffle_multiset_check (S1/S2/S1'),
   L_n = 9n, конечность.
3. **S_shuffle:** период-9-чек (а) + per-(S_i,n) таблица max 9-gram-count (б)
   (§3). Провал (а) → NOT_RESOLVED.
4. **Finite-гейт (M5):** ce + cos_raw/cos_state/norm_delta по всем 8 условиям
   × слоям (новые — полный; R2-условия — spot-проверка) + все контрасты (3
   вердиктных + 2 нуля + 4 legacy). Провал → NOT_RESOLVED.
5. **G_inside (F5, парный — основной):** на n=16, per doc:
   `L1 < fillerA ∧ L1 < fillerB ∧ L2 < fillerA ∧ L2 < fillerB`. Гейт: ≥
   `ceil(0.75 × 14) = 11` доков; иначе `INSENSITIVE_CELL`. Абсолютные пороги
   (loop < 1.0, fillerA ≥ 2.0) — вторичная диагностика
   (`inside_abs_pass`): R2 — 13/14 (miss chat_2 fillerA 1.982).
6. **S_null:** δ_REP, δ_PHR конечны (> 0), §8.

## 11. Сводки и вердикт

Сводки (L23, каждый n; для C ∈ {REP, PHR, INT, N_perm, N_content, e_spec1,
e_spec2, e_null, e_rep_legacy}):

```
near_C_n = mean_i ( p90 |C_i,n(d)|,  d ∈ [0,15] )     и far_C_n (mean, [48,63])
```

(агрегация — среднее по докам; единица бутстрапа — док, §12.)

### Дерево вердикта (порядок жёсткий)

```
1. Гейты §10: integrity; S_stimulus; S_shuffle; finite; δ конечны.
   провал → NOT_RESOLVED (причина в артефакте) [integrity → стоп].
2. G_inside < 11/14 → INSENSITIVE_CELL.
3. Вердиктная клетка (L23; ровно 2 проверки — множественность зафиксирована):
   S_phr := CI_low(near_PHR(n=2))  > δ_PHR
   S_rep := CI_low(near_REP(n=16)) > δ_REP
   S_phr ∧ S_rep → BOTH_SPECIFIC (+substructure каждого)
   S_phr only    → PHRASE_SPECIFIC (+substructure)
   S_rep only    → REPETITION_SPECIFIC (+substructure)
   neither       → NO_2B_SPECIFIC_EFFECT
   (substructure _AMBIGUOUS по единственному значимому → NOT_RESOLVED)
4. Substructure (для каждого значимого C, его первичная доза):
   CI_low(C_near − C_far) > 0  И  CI_high(C_far) ≤ δ_far(C) → _DECAYS
   CI_low(C_far) > δ_far(C)                                  → _PERSISTS
   иначе                                                     → _AMBIGUOUS
5. INT (отчёт, не вердикт): near_INT(n=2), near_INT(n=16) + CI (L23);
   CI_low(near_INT(n=16)) > δ_REP → тег INTERACT_SIGNIFICANT
   («эффект повторения контент-зависим» — вердикт читается с оговоркой).
6. + слойный тег (на n=16; «значим» = near_C_L > δ_L(C); объединение
   {REP, PHR} — правила TZ R2 §10)
7. + дозовый тег L2, + LEARN_ASYMM (если).
```

**Вердиктный словарь** — как v1.0 (`NO_2B_SPECIFIC_EFFECT` /
`PHRASE_SPECIFIC` / `REPETITION_SPECIFIC` / `BOTH_SPECIFIC` /
`INSENSITIVE_CELL` / `NOT_RESOLVED`) + `_DECAYS`/`_PERSISTS`/`_AMBIGUOUS` +
теги `INTERACT_SIGNIFICANT`, `LEARN_ASYMM`. Чтения — §17.

**Диагностика без вердиктной роли** (явно): INT, e_spec1/e_spec2/
e_rep_legacy/e_null, все слои ≠ L23, доза ≠ primary для каждого контраста.

## 12. Статистика

- Единица ресемплинга: **документ**; `doc-bootstrap, B = 2000`, seed 0.
- Ресемплируются **одними доками** (пара́тность, r2d-2.5): near/far для
  REP, PHR, INT + N_perm, N_content + near−far по первичным дозам.
- CI95 (перцентили 2.5/97.5); значимость near−far — CI целиком > 0.
- `prism.nulls_and_stats` (paired_bootstrap) + numpy.

## 13. Артефакт

- `results/AT8/AT8_2b.json` (claim): вердикт + сводки (REP/PHR/INT + нули +
  legacy) + CI + гейты + δ_REP/δ_PHR/δ_far_* + **legacy-δ anchor
  (0.1480/0.0332)** + `machine_subrevision` + r2_store_sha256 +
  pool_rule/pool_per_domain (r2c) + seam2 + I_loop1/I_loop2 +
  inside_pass/inside_abs_pass + **shuffle-таблица** (per-(S_i,n) max
  9-gram-count + период-9-чек) + seeds (1/2/3) + INT-сводки.
- `results/AT8/AT8_2b_store.json` (новый, только 4 новых условия): на
  (док, n∈{2,16}): `ce[t]`, `ce_block`, per-слой `cos_raw/cos_state/
  norm_delta` + `i_loop` (L2) + `__config__` (model, layers, doses,
  phrase1_ids, phrase2_ids, shuffle_seeds, r2_store_sha256, docs_sha256,
  pool_fps, H, revision). Оценка ~1–2 MB (<10 MB → **в git**; если >10 MB —
  gitignored + бэкап, как R2).
- `results/AT8/AT8_2b_series.npz` — из обоих сторов (R2 + 2b; M7); ~13 MB →
  gitignored, бэкап `/data/prism_backups/at8_2b/`, sha256 в манифесте.
- `results/AT8/AT8_2b_manifest.json`: r2_store_sha256 (§10), store/npz/log
  sha256, script_sha256, code-fingerprint, model, attn_impl, threads, seed,
  **full_run-блок** (n_fwd = 112 факт из лога+стора, wall из DONE-строки,
  wall_source=log — r2c/r2d-конвенция).
- `logs/AT8/AT8_2b_run.log` — формат-контракт (r2d-2.8): строка на форвард
  `  <док> n=<n> <cond>: len=… CE=… [block_CE=…]` (112 строк; `\bCE=` не
  матчит `block_CE=`), финальная `AT8_2b_DONE verdict=<вердикт> mode=<мод>
  (wall <сек>s)`. Trackio: `at8_2b_r1` (config со started_at/revision/
  script_sha/seed; стадии pool-check/forward/analysis; status ok/crash в
  finally).

## 14. Цена

- Новые форварды: **112** (4 условия × 14 доков × 2 дозы). Измеренная
  скорость R2: ~14.3 с/фд (2596 s / 182, OMP=6) → **~27 мин** + модель/
  анализ ~3 мин → **~30–35 мин CPU** (в 1.6× R2).
- Запуск: `./run_one.sh at8_2b ...` (setsid nohup, OMP=6, nice-10, watchdog,
  память-гейт avail ≥ 10G). **Одна CPU-задача за раз** (AGENTS).

## 15. Smoke-план (до полного прогона)

1. 1 домен (wiki) × 2 дока × n=2 — 8 форвардов (L2, S1, S2, S1'):
   - R2-store integrity-гейт проходит (sha, fingerprint, 14 доков);
   - legacy-δ пересчитан из R2-store: **0.1480/0.0332 в точности**
     (fail-loud anchor — целостность пайплайна);
   - phrase_token_check + shuffle_multiset_check (assert); S_shuffle
     период-9-чек PASS + таблица max-count;
   - block-CE L2(n=2) ∈ [0.5, 2 × block-CE_L1(n=2)] (обучаемость L2;
     справка R2: block-CE_L1(n=2) = 3.29 док-медианой — v1.0-порог «< 1.0»
     был неверен для n=2 и исправлен);
   - finite-гейт PASS; δ_REP/δ_PHR > 0; e_spec1 воспроизводит R2 near
     (L23, n16: 0.1020 ± 1e-6 — cross-check стора);
   - resume: прерванный smoke доходит до конца.
2. Синтетические store'ы (§16) → машина даёт ожидаемые вердикты (8/8).
3. Дегенерации:
   - подменённый R2-store (sha-mismatch → стоп, SystemExit);
   - NaN в L2 (finite → NOT_RESOLVED);
   - L2 := L1 (PHR ≡ 0 — контроль, что контент-контраст не артефакт);
   - S1 := L1 (дегенерация shuffle → S_shuffle гейт срабатывает).

## 16. Синтетические тесты (8 сценариев, как R2 test_synth)

| Сценарий | PHR | REP | INT | Ожидание |
|---|---|---|---|---|
| no_effect | ~0 | ~0 | ~0 | `NO_2B_SPECIFIC_EFFECT` |
| phr_decays | near>δ, far≤δ, near−far>0 | ~0 | ~0 | `PHRASE_SPECIFIC_DECAYS` |
| phr_persists | near>δ, far>δ | ~0 | ~0 | `PHRASE_SPECIFIC_PERSISTS` |
| rep_decays | ~0 | near>δ, far≤δ, near−far>0 | ~0 | `REPETITION_SPECIFIC_DECAYS` |
| rep_persists | ~0 | near>δ, far>δ | ~0 | `REPETITION_SPECIFIC_PERSISTS` |
| both | near>δ | near>δ | ~0 | `BOTH_SPECIFIC_*` |
| interact_only | ~0 | ~0 | near>δ | `NO_2B + INTERACT_SIGNIFICANT` |
| insensitive | — | — (inside_pass=3) | — | `INSENSITIVE_CELL` |

## 17. Пререг-чтения исходов

- `NO_2B_SPECIFIC_EFFECT` — R2-`NO_LOOP` усиливается: ни содержимое фразы
  (на n=2, где контент жив), ни периодичность (на n=16) не создают
  пост-эффекта сверх shuffled-нулей (на пре-рег-разрешении; отсутствие не
  доказано — конвенция r2d-1.5).
- `PHRASE_SPECIFIC` — канал: содержание конкретной фразы. Согласуется с
  ответом архитектора «после ~2 повторов фраза выучена» (block CE 0.38–0.51)
  — CE внутри блока ≠ пост-эффект; теперь разделены (и меряются на дозе,
  где контент ещё жив).
- `REPETITION_SPECIFIC` — канал: само повторение (периодичность при том же
  лексическом составе), не контент.
- `BOTH_SPECIFIC` — два канала; substructure каждого — в claim.
- `INTERACT_SIGNIFICANT` — эффект повторения контент-зависим (одна фраза
  «садится», другая нет): REP-вердикт читается с этой оговоркой.
- `LEARN_ASYMM` — phrase2 обучается иначе phrase1 (block-CE ratio > 3×):
  PHR читается с оговоркой «содержимое + обучаемость».

## 18. Границы (не делать)

- R2-store — **read-only** (sha зафиксирован §10); R2-вердикт/числа не
  пересчитываются; R2-документы — история, не переписывать.
- Новые форварды — **только L2/S1/S2/S1' (112), n ∈ {2,16}**;
  ctrl/loop1/fillerA/fillerB не перегоняются.
- Пул не пересобирается (14 доков R2, тот же filler_map).
- Фразы не меняются (пре-рег §3); shuffle-сиды (1/2/3) не меняются;
  автосидов нет (провал S_shuffle — NOT_RESOLVED, не пере-сид).
- **Слияние с AT8_3 — не в этом пре-реге.** 8_3 review r3
  (`docs/AT8_3_review.md`, патч-лист 1–8): 5 вердикт-критических пунктов
  (1–5) открыты в TZ v1.2 (H_eval=97; знаменатель/LOW_POWER; CI-таблица/гейт;
  строгий >0.5 или блокирующий WIDE_CI; доз-зависимый гейт филлеров) —
  слияние унаследовало бы их. Объединённый пре-рег — после закрытия 8_3
  (архитектор предложил расписать: условия, контрасты с нулями, дерево
  REP/PHR/INT + локализация 8_3, store с h-векторами, смета). До утверждения
  объединённого пре-рега standalone-прогон 2b может быть запущен отдельно
  (112 фд ≈ 30 мин); если объединённый пре-рег утверждён до запуска 2b —
  standalone пропускается (явное решение в PLAN).
- paper.md не трогается; новые слои/дозы/фразы — только отдельным пре-регом.

## 19. Пререг-поля (сводка, v1.1)

| Поле | Значение |
|---|---|
| phrase1_ids | [561, 523, 668, 381, 5467, 506, 279, 5926, 13] (R2, frozen) |
| phrase2_ids | [14095, 491, 32275, 3655, 3043, 264, 11012, 7803, 13] |
| shuffles | S_i = L_i[π_i], π_i = default_rng(seed_i).permutation(9n); seeds: S1=1, S2=2, S1'=3 (frozen) |
| пул | 14 доков R2 (4/2/4/4), r2_store_sha 78419a69… |
| условия-форварды | L2, S1, S2, S1': 14 × 2 дозы × 4 = 112 |
| дозы | n ∈ {2, 16}; PHR primary n=2; REP primary n=16 |
| контрасты | REP/PHR/INT (ортогональные, ±1/2); нули: REP ← N_perm = S1−S1', PHR ← N_content = S1−S2; INT report-only |
| legacy (диагностика) | e_spec1/e_spec2/e_null/e_rep_legacy (R2-сравниваемость) |
| δ | δ_REP/δ_PHR = 95th pct per-doc p90\|N_*\| (28 значений); legacy-δ 0.1480/0.0332 — integrity-якорь |
| near/far | p90 d∈[0,15] / mean d∈[48,63], H=64, cut+L_n+d |
| слои | {6,12,18,19,23}, primary L23 (преемственность) |
| bootstrap | doc, B=2000, seed 0, пара́ность (все контрасты + нули одними доками) |
| inside-гейт | парный (4 неравенства, n=16) ≥ 11/14 |
| shuffle-гейт | период-9-чек + per-(S_i,n) max 9-gram-count (артефакт) |
| вердикты | NO_2B / PHRASE / REPETITION / BOTH + _DECAYS/_PERSISTS/_AMBIGUOUS; теги INTERACT_SIGNIFICANT, LEARN_ASYMM |
| множественность | вердиктная поверхность = ровно 2 проверки (S_phr@n2, S_rep@n16; L23) |
| цена | ~112 форвардов ≈ 30–35 мин CPU |
