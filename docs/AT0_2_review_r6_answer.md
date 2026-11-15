# AT0_2 — ответка на внешнее ревью r6 (docs/dynamic/AT0_2_r6_REVIEW.md)

**Статус:** VERDICT: PASS (MAJOR: 0, MINOR: 2) — все 5 пунктов «Что проверить при ответке»
закрыты; перепуск НЕ требуется (все пункты — документация/следующий run).
Дата: 2026-11-15. Прогон: round 2 (ревью r6: E1–E8, M1–M5), gates ALL PASS.
Пины: claim file sha `b3cad772b3b3190f5f0c552c189e528d59676b2f8bd050e163580349eb2d83d7`;
store canonical sha `f7cc011ef856a6dc30dcb1727dbd2948f61ca7a0d96cec119549eedb9340db77`
(файл 222.8MB — **не в git**, правило >10MB; закреплён sha в manifest и в ревью);
код `e58b7d19…` (закоммичен в `649cc04`). Все числа ниже пересчитаны из артефакта
(`results/AT0/AT0_2_SHAPE.json`), не из головы.

## По пунктам

### 1. MINOR-1 — иерархия D_null нарушена для v_tangent (all_96) — **согласен; object-dependent, reportable-only**

Числа сверены с claim (`aggregates.v_tangent[null].D_null_mean`) — **точное совпадение** с
таблицей ревью:

| null | iid | ar1 | ar2 | ar4 | blk2 | blk4 | blk8 | blk16 | blk32 |
|---|---|---|---|---|---|---|---|---|---|
| D_null_mean | 0.0412 | 0.0049 | 0.0064 | 0.0085 | 0.0556 | 0.0558 | 0.0441 | 0.0464 | 0.0538 |

Механизм подтверждён: `D_null` — разброс null-сидов вокруг **центра null**, а не «сила
null-гипотезы». Для `v_tangent` реальная ACF быстро спадает (профиль сконцентрирован в
полосе k=1), поэтому AR-null (Yule-Walker) воспроизводит ACF real с точностью до fit-error
⇒ сиды лежат близко к центру ⇒ малый D_null (ar1 0.0049). Разброс iid/blk-сидов задан шумом
в долях k=1..8 и не следует «иерархии» null-типов — цепь iid>ar1>blk2 для v_tangent рвётся,
blk-цепь не монотонна. Для `h_raw`/`du_unit` (hump-ACF) иерархия полная (таблица ревью) —
противоречий нет.

**Действие:** код и артефакт не менять (reportable-only, не входило в verdict, не гейт).
Правило для отчётов: иерархию D_null **не утверждать** для `v_tangent`; помечать как
object-dependent (зафиксировано здесь; в финальном пользовательском отчёте — в п. 4).
Перепуск не требуется.

### 2. MINOR-2 (Z1) — α=0.05 не зафиксирован в хедере — **согласен; фикс в следующем run**

Проверено: в claim α отсутствует (только `delta_min: 0.05`; `meta` = env/field_sha256/
data_sha256/gen_sha_ok/decode_probe/script_sha256/n_docs/n_docs_verified/confounds_note/
layers/objects/smoke). **Действие:** при следующем run добавить поле `alpha: 0.05` в
claim-хедер/meta. Перепуск round 2 не требуется: frac-метрики descriptive-only (риск мал —
как в ревью).

### 3. Z2 — таблица TZ §1.5 T0.2 — **сделано**

`docs/TZ_add_0-3.md`, §1.5, T0.2, строка blk-shuffle: роль «null локальности» заменена на
«damped/seamed null (НЕ null локальности)» с wording'ом r6-E2: verdict читается как
«real differs from unmasked block-shuffle null (damped within-block, seamed)» — НЕ как
«локальность ≤ b не объясняет»; locality-прибор — P3-ladder (seam-excluded, AT0_1).
Прочие строки таблицы и примечание «seam-excluded варианты: только как отдельный оценщик,
Σ с real НЕ СРАВНИВАТЬ» — без изменений. Файл до правки был чистым (git diff пуст) —
в коммит попадает только эта правка.

### 4. N6 — пользовательский отчёт: verdict v_tangent = not_testable (primary) — **зафиксировано**

Сверено с claim:
- **primary** (`aggregates_primary`, verified ∩ non-degen, 37 доков, n_cells=185):
  `v_tangent` Δmean < 0.05 по **всем 9 null** (iid 0.0444, blk2 0.0374, blk4 0.0318,
  blk8 0.0273, blk16 0.0231, blk32 0.0181, ar1 0.0377, ar2 0.0294, ar4 0.0283) ⇒ **9/9
  not_testable**. (В ревью «3/3 (Δmean 0.0444/0.0181/0.0283)» — это подмножество
  iid/blk32/ar4 (экстремумы); остальные 6 тоже < 0.05 — проверено.)
- **all_96** (`aggregates`, 96 доков, 478 клеток): `v_tangent` — **refuted**
  (iid Δmean 0.3025, CI_Δ [0.082, 0.592]; blk2 0.2541; ar1 0.3311).

Финальная формулировка (зафиксирована здесь и для итогового отчёта):
**verdict v_tangent = not_testable (решающий пул — primary, по пре-регу); расхождение с
all_96 (refuted) — раскрыто: пулы дают противоположные ответы из-за corpus/degen-доков,
входящих только в all_96.** `h_raw`/`du_unit` — refuted во всех пулах (primary Δmean
iid 1.1946/1.1963 — вне разночтений).

### 5. Z3 — формулировка «participation ratio» — **проверено, не переносится**

Grep по docs/: в документах AT0_2 (prereg/claim/ревью) метрика = `tested_D_center_l1`
(L1 band-share distance, rename M2) — формулировка «participation ratio» там отсутствует.
Она присутствует только (а) в самом ревью-промпте `AT0_2_r6_REVIEW_prompt.md` (строки 12,
66) — это запись промпта, в docs не переносится; (б) в файлах AT0_1/AT0_5, где
participation ratio — фактическая метрика (d_eff AT0_1 / diagonal PR AT0_5) —
конфликта нет. В отчётах AT0_2 использовать «L1 band-share distance (tested_D_center_l1)».

## Замечания (не блокируют) — приняты, без действия

- **N3:** run.log и AT0_2_r6_stdout.log различаются одной строкой («field selfcheck: OK» —
  print из `pf._selfcheck()` не попадает в LOG). run.log — tracked-канон. Косметично,
  согласен.
- **N4:** claim.gates G0/G0_sources/G1_field/G7_store — raw-value гейты без ok-флага;
  PASS установлен пересчётом (таблицы 1a–1e/2b ревью). Учтено.
- **N5:** all_96 n_cells=478 (не 480): 2 L1-absent-клетки; «480» в промпте — все клетки
  включая absent. Подтверждено (n_cells=478 в claim).

## Итог

Все 5 пунктов «что проверить при ответке» закрыты (1 — зафиксировано object-dependent;
2 — правило для следующего run; 3 — TZ-правка выполнена; 4 — verdict зафиксирован;
5 — проверено). Перепуск round 2 не требуется; артефакты round 2 (claim/manifest/логи)
коммитятся с этой ответкой, ревью-файлом и TZ-правкой (явные пути; store 222.8MB — не в
git, sha закреплён в manifest).

AT0_2_R6_ANSWER_DONE
