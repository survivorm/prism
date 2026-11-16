# AT0_4 exec-review r3 — ответки (C1–C6, R1–R3)

Объект: `docs/AT0_4_EXEC_REVIEW.md` (14.11, «инженерно PASS with caveats / scientific REVISE»).
Ответы по пунктам: принято / принято с модификацией / спорю (с аргументом) / самокоррекция.
Исправления: `experiments/at0/AT0_4_mix.py` (sha `72d5f658…`, METRICS_VERSION v2) +
`experiments/AT0_4_rebuild_r3.py` (rebuild БЕЗ форвардов) → новый store
`4c0a1db2e7984621` (full sha `4c0a1db2e798462138aba970ec636634b2d96ff6f994e99762d45d3bc09d13d0`), claim `AT0_4_MIX.json`, manifest, `AT0_4_rebuild_r3.log`.
Проверено, что ревьюер смотрел актуальные файлы: disk sha `a4b7a138…` (момент ревью) =
sha из `AT0_4_rebuild.log`/`AT0_4_EXEC_REVIEW_r2.md` §1; цитаты кода совпадают с диском.

## C1. Estimand drift: `r` ≠ профиль; own ≠ corpus — **ПРИНЯТО**

Факт, не спор: P1–P5 тестируют `r`/`f_DC`/`r_max`; профиль `ρ̄(k)` хранится в store
(112×96) описательно и не тестируется; вход — только `own_w0..7` (G16: P2_corpus 0/16 с
токенами). Фикс wording (в трёх местах):
- docstring `AT0_4_mix.py`: «Скоуп (exec-review r3 C1): own-doc r/DC carrier pilot …
  corpus-клетка T0.4 остаётся открытой»;
- `store.meta.scope` + `claim.meta.scope` (то же);
- `verdict.summary` начинается «AT0_4 (3×2 own-doc r/DC carrier pilot; профиль ρ̄(k) —
  описательно, НЕ тестируется в P1–P5; corpus T0.4 — open): …».
Prereg `docs/AT0_4_prereg.md` НЕ переписан (замороженная пре-регистрация — история;
исправление скоупа — в docstring/meta/summary и здесь).

## C2. P4 median маскирует doc-level эффект — **ПРИНЯТО**

Подтверждено числами: own_w4 ≥ 0.1 во всех 6 live-инстансах (0.1298/0.1188/0.1277/0.1245/
0.1314/0.1258), own_w0 в 2/6 (0.1101, 0.1002). Добавлено:
- `P4.doc_order_flags`: счётчик ≥ порога по всем live-инстансам + флаг
  `systematic` (count == n_live) / `partial` / `none` — own_w4 `6/6 systematic`,
  own_w0 `2/6 partial`, остальные `0/6 none` (полный список по 8 докам);
- summary: «random order-invariant» УБРАНО → «random: null по медианному тесту 6/6;
  doc-level order sensitivity: own_w4 (6/6) systematic; own_w0 (2/6) partial».
Это закрывает и КОСМЕТИКУ r2 (ведущее слово «invariant»).
**Бонус (C6.5, token_provenance):** оба outlier-дока — degenerate low-vocab: own_w0
max_share 0.930 (93% один токен, min_period 1, decode «3-hour class … BBQ»), own_w4
max_share 0.940 (94% один токен — «1000000000000…», min_period 1). Order-sensitivity
концентрируется именно на вырожденных текстах — интерпретация 6/6 без нового прогона.

## C3. P3/P5 «confirmed» без inference — **ПРИНЯТО (и подтверждено числом)**

Измерено по store: P3 per-doc = 6+/2− (own_w0 −0.0535, own_w4 −0.0397 — те же
degenerate-доки, что в C2), exact sign test (биномиальный, двусторонний) p = 74/256 =
**0.2890625** ≥ α=0.05 → «confirmed» (знак медианы 0.0158) был overclaim. Реализовано
(модуль `build_p3_p5`, единый источник для main и rebuild):
- exact sign test (нули исключаются; n=0 → p=1);
- paired bootstrap CI медианы (B=2000, seed 117 — серийный boot-сид AT0_1):
  P3 CI [−0.0221, 0.0208] — содержит 0;
- `DELTA_MIN`: r = 0.01 (1/10 от P4-порога 0.1 — априорный масштаб «большого эффекта»
  серии для r), f_DC = 0.01/383 (G1: f_DC = r/N);
- вердикт-лестница: `confirmed` (p<α И |med|≥Δmin) / `descriptive` (med>0, но нет p или
  |med|<Δmin) / `null` (med≤0 — объявленный нулевой исход пре-рега) / `not_testable`.
  P3 и P5: **confirmed → descriptive** (ожидаемое изменение, зафиксировано в PLAN ДО
  rebuild; rebuild ассертит именно этот исход, fail-loud).
- P5: вычисляется независимо (свои числа: med 4.127e-05, p = 0.2890625), но помечено
  `mirror_of: "P3"` + note «алгебраическое зеркало P3 (G3: f_DC = r/N, S4 ≤ 1e-9) — НЕ
  независимое свидетельство»; в summary — «mirror P3 по G3 — не независимый сигнал».
- `alpha: 0.05` теперь в артефакте (был «заявлен, но не использовался»).

## C4. P1 band/vacuous без null — **ПРИНЯТО С МОДИФИКАЦИЕЙ + САМОКОРРЕКЦИЯ**

Принято: добавлены diagnostics (per инстанс: `c0_stats` median/max, `reachable_frac`,
`low_motion_docs_c0_le_4`) и **синтетический нуль** (`synthetic_null_r`, чистый numpy,
B=200, seed 117, детерминирован; в артефакте `P1.synthetic_null`):
- iid-сфера (u_t независимы на S^1023): r med **0.003 ≈ 1/N**, c0 ≈ 766 ≈ 2N,
  r_max ≈ 0.005 — P1 vacuous и по r_max, и по r;
- iid low-motion c0*=2: r med 1.002 [0.935, 1.079], band_frac 1.00, reachable 1.00;
- iid low-motion c0*=4: r med 0.995, band_frac 1.00, **reachable 0.58 — c0*=4 СИДИТ
  на границе r_max=1** (c0 ~ 4±0.004 → r_max ~ 1±0.001);
- smooth drift (u_t = unit(u_0 + t·0.01·v)): r med 191.6, band_frac 0 — дрейф вне бэнда.

Возражения/уточнения:
1. **Самокоррекция (я):** в черновике ответа я утверждал, что «r ~ 1/N» из ревью
   неверно (ссылаясь на prereg v3 «для некоррелированных приращений E[r]=1»). Дизайн-
   тест показал: **ревьюер прав для null-модели iid-ПОЗИЦИЙ на сфере** — ΣΔu =
   u_end − u_0 (телескопирование), ‖u_end−u_0‖² ≈ 2, c0 ≈ 2N → r ≈ 1/N. E[r]=1 верно
   только для модели iid-ПРИРАЩЕНИЙ (low-motion случайные шаги, где u не нормируются —
   априорный смысл «ортогонального марша»). Обе модели теперь в артефакте с числами;
   prereg-v3-формулировка не отменяется, но охватывала только increments-модель.
2. Бэнд [0.75, 1.5] НЕ arbitrary — это сигнатура low-motion increments-режима
   (null: r ≈ 1, p05/p95 ≈ 0.93/1.08 → бэнд ≈ ±3σ). Но ревью прав в главном:
   «подтверждение P1 возможно только на low-motion траекториях» — и это **объявлено a
   priori в prereg v3.1** («vacuous(r_max<1) — наиболее вероятный исход для случайной
   модели; подтверждение возможно только если случайный ствол даёт «короткие» шаги —
   объявлено заранее, не подгонка»). Измерено: 3/3 vacuous, c0 = 13…393 (r_max 0.013…
   0.31) — на 3–100× выше low-motion гейта c0≤4. «Отбор low-motion paths» — реальная
   оговорка, но она предобъявлена, а не обнаружена постфактум; теперь закреплена
   `band_note` + `synthetic_null` в артефакте.
3. Positive controls (iid sphere / smooth walk / drift) — сделаны; «collapsed path»
   (= degenerate c0=0) покрывается существующей `degenerate`-меткой G_nd, отдельный
   null не нужен (r не определён).

## C5. P2 «embed_head_contributes» misleading — **ПРИНЯТО**

Подтверждено: h = hidden_states[24] — ДО lm_head; разница R1/R2 в r идёт только от
input-эмбеддинга (head действует на logits после L+1). Фикс:
- ветка вердикта переименована: `embed_head_contributes` → `tied_embed_contributes_to_r`
  (в текущем артефакте недостижима: все 3 сида «same» → `trunk_carries` — НЕ ИЗМЕНИЛСЯ);
- `P2.cell_note` (модульная константа `P2_CELL_NOTE`, единый источник): «hidden-state r
  (hidden_states[24], ДО lm_head) идентифицирует только вклад tied input-эмбеддинга:
  lm_head действует на logits после L+1 и не входит в hidden-траекторию; «embed+head
  contributes» было бы неверно (r3 C5)»;
- summary: «R1 vs R2 (real): trunk_carries (hidden-state r — input-эмбеддинг, не head)».
Второе замечание C5 («same r ≠ same profile») — в скоупе C1: профиль не тестируется
(описательно), не переименовывать.

## C6. Gates/provenance/checkpoint

### C6.1 S5 telescope — нормы, не вектор — **ПРИНЯТО (код) + обоснование без ре-форварда**

Код: `du_metrics` теперь считает `telescope = ‖ΣΔu − (u_N − u_0)‖` (векторный остаток),
`METRICS_VERSION` v1→v2 (ckpt-инвалидация для БУДУЩИХ прогонов). Дизайн-тест: на
синтетике vector-telescope = 1.95e-15, G1-тождество 3.16e-16.
**Текущий артефакт НЕ пересчитан** — и это обосновано, а не проигнорировано:
(a) S5 на нём — норма-форма, PASS (≤1e-9); (b) векторный остаток — чистая fp64-
округление телескопической суммы: ‖ΣΔu − (u_N−u_0)‖ ≤ γ_383·Σ‖Δu_t‖ ≤ γ_383·N·√c0_max,
γ_383 ≈ 4.2e-14, N = 383, c0_max = 392.83 (из store) → ≤ **3.2e-10 < 1e-9**;
(c) u-вектора в store не хранятся, пересчёт без форвардов невозможен, а ре-форвард
(112 шт., ~4 ч) ради укрепления уже-пройденного на 3 порядка гейта — не по цене.
Граница закреплена в `provenance.aggregation_rebuild_r3.s5_note`.

### C6.2 ckpt resume без script sha — **ПРИНЯТО (подтверждено кодом)**

`save_ckpt` писал `script_sha256`, а `load_ckpt` проверял только `metrics_version` и
`sigma_table_sha256` — ревью прав. Фикс: в mismatch-условие добавлено
`j.get("script_sha256") != ckpt_script_sha` (fail → полный прогон, лог с двумя shas).
Влияние на текущий артефакт нулевое (ckpt удалён после успешного прогона).

### C6.3 `g.get("ok", True)` — **ПРИНЯТО**

→ `all(g.get("ok") is True …)`. В текущем артефакте у всех 6 гейтов (`S1..S5`, `G_nd`)
`ok` эксплицитно true — «ALL PASS» в run.log действителен; фикс — для будущих прогонов.

### C6.4 mix — одна перестановка — **ПРИНЯТО (declaration)**

`meta.mix_note`: «mix = одна фиксированная перестановка на док (seed 7) — без
permutation-seed sensitivity (новых форвардов нет); эффект mix может зависеть от
конкретной перестановки». 2–3 перестановки/док — новые форварды (§6 ревью) — вне
агрегационного патча.

### C6.5 decode/token statistics — **ПРИНЯТО (сделано, см. C2-бонус)**

`token_provenance` в store и claim: per док `n_uniq`, `max_share`, `rep8`, `rep16`,
`min_period_90pct` (p=1..16, порог 0.9), `decode_48` (первые 48 токенов). Decode —
через `tokenizers` (0.22.2, локальный `tokenizer.json`), НЕ transformers — venv-
transformers сломан параллельной сессией (huggingface-hub 2.4.0 требует <2.0); чинить
venv не мое право, rebuild работает с transformers-стубом (data-only, зафиксировано в
`provenance.venv_note` и rebuild-логе).

## R1. Manifest обновлён? — **ПОДТВЕРЖДЕНО (и для этого rebuild)**

Цепочка после rebuild r3 (холодный пересчёт с диска): `sha256(store/4c0a1db2e7984621.
json)` = имени файла = `claim.store.sha256` = `manifest.store_sha256`;
`manifest.claim_sha256` = `sha256(claim)`. 4/4.

## R2. Old store deletion safe? — **ПОДТВЕРЖДЕНО (с одним честным ограничением)**

Новый store содержит старое как подмножество (rebuild-ассерт: изменены ТОЛЬКО
prereg P1–P5, verdict.summary, verdict.pretest_results, provenance, +новый ключ
token_provenance; meta/gates/per_doc/profiles БИТ-неизменны). Full sha заменяемого
store (`f1cc7265…`) закреплён в `provenance.aggregation_rebuild_r3.replaced_store_sha256` **и
восстановим из git**: `git show 61483f3:results/AT0/store/f1cc7265dd40c874.json | sha256sum`
= `f1cc7265dd40c8744174fe3d6e0889199907c5652012970100639205c440ee4d` (бит-в-бит, проверено).
**Но:** full sha предшественника `71d3b5d868ed00c7` (store эпохи r1, удалён rebuild'ом
r1/r2) НЕвосстановим — файл никогда не коммитился (r2 §1: «все AT0_4-файлы untracked» на
момент r1), на диске и в git только 16-хекс-префикс (= content-addressed имя серии
AT0_1–3). Цепочка закреплена по 16-хекс-звеньям + полным shas доступных звеньев в
`provenance.script_diff_chain` (f59035c0…→92effe83…→a4b7a138…→bf80334d…→текущий→этот
rebuild; полные 64-хекс звеньев 1–4 — в `AT0_4_EXEC_REVIEW_r2.md` и commit 61483f3).
С 61483f3 все store-файлы в git — дальше цепочка полный-ша-полный (каждый rebuild
записывает full sha заменяемого, а store коммитится вместе с rebuild'ом).

## R3. Hardcoded suffix — **СОГЛАСЕН (fail-loud, ок)**

Rebuild r3 в духе того же паттерна: ожидаемые вердикты (P3/P5 «descriptive», p =
0.2890625, P1 vacuous×3, P2 trunk_carries, outliers 2/6 + 6/6) ассертятся ДО записи;
любое расхождение — краш без записи (проверено: два краша в процессе разработки —
оба до записи, артефакт не пострадал).

## Изменения вердиктов (итог)

| Блок | Было | Стало |
|---|---|---|
| P3 | confirmed (знак медианы) | **descriptive** (p=0.2890625, CI ⊃ 0; med 0.0158 ≥ Δmin) |
| P5 | confirmed (mirror) | **descriptive** (mirror P3; p=0.2890625) |
| P1 | vacuous ×3 | vacuous ×3 + c0_stats/reachable_frac/low_motion/synthetic_null/band_note |
| P2 | trunk_carries | trunk_carries + cell_note (head не в hidden-траектории) |
| P4 | null ×6, outliers | null ×6 + **doc_order_flags** (own_w4 6/6 systematic, own_w0 2/6 partial) |
| summary | «random order-invariant» | «null по медианному тесту + doc-level sensitivity», own-doc скоуп |

## Не сделано (осознанно)

- **Новые форварды** — запрещены скоупом (ревью §5 — агрегация; §6 — что требует
  форвардов). Corpus-клетка T0.4, mix-permutation sensitivity, re-run под METRICS_VERSION
  v2 — остаются открытыми (§6 ревью подтверждает).
- **Смоук `AT0_4_mix.py`** — невозможен в текущем venv (transformers сломан
  параллельной сессией; модуль импортирует его на топе). Вместо смока: py_compile обоих
  скриптов + дизайн-тесты всех новых функций (sign test: 8/8→0.0078125, 6/2→0.2890625,
  7/1→0.0703125; bootstrap-детерминированность; build_p3_p5 confirmed/descriptive/null
  ветки на синтетике; du_metrics vector-telescope 1.95e-15 + degenerate-ValueError;
  synthetic_null инварианты) + rebuild как end-to-end прогон всех изменённых путей
  (main-логика = те же модульные функции, что вызваны rebuild'ом). Перед следующим
  forward-прогоном смоук обязателен в рабочем venv.
- **Коммит** — по протоколу серии (prereg §9: «коммит — только после внешнего ревью») и
  по паттерну AT0_1/AT0_2 (коммит по команде юзера после ревью-цикла) — ждёт.

## Самокоррекции (мои, этого раунда)

1. C4 «r ~ 1/N»: сначала считал неверным (prereg v3 «E[r]=1»); для iid-сферических
   ПОЗИЦИЙ — верен (телескопирование). Обе null-модели теперь в артефакте.
2. c0*=4 в low-motion null: ожидал reachable ≈ 1; фактически 0.58 — граница r_max=1.
   Ассерт в rebuild'е — интервал (0.3, 0.7), не точность.
3. Rebuild: два краша на своих ассертах (порядок ключей P4-блока; отсутствие f_dc в
   cells-минимуме) — оба пойманы ДО записи; артефакт повреждён не был.
