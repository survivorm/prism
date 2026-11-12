# Ревью E94/S5 r2 — проверка закрытия 5 MINOR из r1. Ты ревьюер, не исполнитель.

Ревью r1 (`docs/dynamic/2_REVIEW_E94_S5_exec_r1.md`) вынесло PASS_WITH_MINORS с 5 FINDINGS.
Код и артефакт перепущены. Твоя задача — проверить, что **каждый из 5 пунктов закрыт**
по конкретному полю артефакта/строке кода, и что **нет регрессии** (вердикт и числа
не изменились, кроме предписанных правок).

## ЛИМИТ

Ответ **≤ 3000 символов**. Цитаты ≤ 2 строк. Читай ТОЛЬКО файлы из списка.
Не запускай GPU, не пересчитывай корпус; допустим точечный python для 1–2 сверок.

## Что читать

1. `docs/dynamic/2_REVIEW_E94_S5_exec_r1.md` — FINDINGS (объект сверки).
2. `results/e94/S5.json` — новый артефакт.
3. `experiments/exp94_s5_dryrecon.py` — новый код (точечно: summarize(), negative_controls(),
   блок control_no_dry и prompt_brk в main(), lines_cited).

## Чек-лист (по одному пункту на каждый MINOR)

**F1.** `flips_explained_by_dry` теперь считается как «flip ∧ bucket==EXPLAINED» (не TP
конфьюжна). Укажи строку кода. В артефакте: 16 при 16 флипах и UNEXPLAINED=0.

**F2.** `control_no_dry` — сравнение НАБОРОВ (doc:pos): поле `mismatch_positions`
в артефакте равно набору флипов (16 позиций, сверь хотя бы 3 с `flips_detail`).
`expected_eq_obs_flips` — результат сравнения множеств, не счетов.

**F3.** Промпт-иды из brk зафиксированы и сверены: `vocab_text_check.prompt_brk_ids`
(10 id), `prompt_brk_ids_verified` (каждый с our/server-текстом и affected),
`prompt_brk_ids_unverifiable_from_raw`. Все 10 сверены? `any_penalty_affected` учитывает
и промпт-токены?

**F4.** Цитаты строк исправлены: lines_cited содержит 3393 (не 3394) и 3459 (не 3457).
Сверь по исходнику `~/work/llama.cpp/src/llama-sampler.cpp`: 3393 = `min(|tokens|,
dry_penalty_last_n)`? 3459-3461 = `if (rep_limit < dry_allowed_length) return;`?

**F5.** `negative_controls` прогнан по всем 16 r1-документам: `scope` =
{rep: r1, docs: 16, positions: 6144}; все три варианта `caught=true`; структура
`variants`/`scope` записана в JSON.

**Регрессия.** Сравни ключевые числа с r1-артефактом (закоммиченного нет — сверяй со
stdout r1: buckets EXPLAINED=6144, flips 16, g_need_max 0.994864, vocab 16 mismatches,
H0_supported=true). Что изменилось сверх предписанных 5 правок? Допустимо ли?

## Формат (строго)

Отчёт запиши в `docs/dynamic/2_REVIEW_E94_S5_exec_r2.md`:

```
ВЕРДИКТ: PASS | PASS_WITH_MINORS | BLOCK
F1: <закрыт? строка кода, поле артефакта>
F2: <...>
F3: <...>
F4: <...>
F5: <...>
РЕГРЕССИЯ: <...>
FINDINGS: (≤5, если есть)
```

Последняя строка stdout: `S5_R2_DONE`
