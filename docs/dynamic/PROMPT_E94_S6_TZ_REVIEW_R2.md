# Ревью r2 ТЗ E94/S6 (после правок по r1)

Ты — внешний ревьюер. Твоя задача — проверить, что поправки r1-ревью внесены
корректно и полностью, и нет ли новых проблем в ИЗМЕНЁННЫХ местах. Не пересматривай
снова всю ТЗ целиком (r1 уже дал PASS_WITH_MINORS по базовой части); фокус — diff-область.

## Файлы

- ТЗ: `docs/dynamic/0_PACKET_E94_S6_TZ.md` (актуальная версия, после правок)
- r1-ревью (исходные findings): `docs/dynamic/2_REVIEW_E94_S6_TZ_r1.md`
- Код (для сверки формулировок гейтов с реализацией): `experiments/exp94_s6_dry0.py`
- База: `results/e94/Q2.json`, `results/e94/Q1.json` (только если нужно сверить числа)

## Что проверять (по finding-ам r1)

1. **MAJOR-1 (echo-гейт)**: в §2/§3 есть обязательный эхо-гейт — `generation_settings.dry_multiplier == 0.0`
   на первом live-запросе И офлайн из raw при пересборке; нарушение → rc=14; в артефакт —
   эхо первого запроса (dry_multiplier/dry_base/dry_penalty_last_n). Сверь: (a) формулировка в ТЗ
   совпадает с кодом (`check_echo` в main_run — каждый ответ вкл. кэш; `rebuild_from_raw` — `echo_offline`;
   `run_from_raw` → HALT/rc=14 при n_bad>0), (b) в §6 контракте артефакта есть `echo_first_request`
   и `gates.echo_offline`, (c) rc=14 в RUN_RC-строке §3 покрывает эхо.
2. **MINOR-1 (R1-разбивка)**: в §4 R1 — требование разбивки RESOLVED по `top1_agree_cpu`
   (шумовая составляющая, вершина≠cpu). В коде: `verdict_rules(..., per_position)` считает
   `resolved_by_top1_agree_cpu` и упоминает её в detail R1. Сверь соответствие.
3. **MINOR-2 (G2-фразировка)**: §3 G2 теперь разделяет: спайк ДО прогона — жёсткий гейт
   (rc=2/13, прогон не запускается), G2 основной фазы — мягкий (g2_violations, не убивают).
   Сверь с кодом: `spike_cache` → fail → finish rc=2/13 (прогон не запускается); G2 main —
   только в `rebuild_from_raw` (g2_violations в артефакт).
4. **MINOR-3 (q2_emitted_rank_in_s6)**: §4 эффект-сайд — `rank_reported` для RESOLVED тавтологично 1,
   добавлен `q2_emitted_rank_in_s6` (ранг Q2-эмиссента в top_logprobs отчёта S6). В коде:
   `effect_side(per_position, resolved, q2_emitted)`; Q2-эмиссент берётся из Q2-raw тем же
   `position_metrics_s6` (не из артефакта). Проверь, что в §6 контракт артефакта включает
   `q2_emitted` (в baseline) и `q2_emitted_rank_in_s6` (в effect_side).
5. **MINOR-4 (ids448)**: §1 явно описывает конкатенацию `ids448(doc) = ids(prompts.json[doc], 64) +
   gen_ids(REPLAY[doc], 384)` и `prompt(k) = ids448[:k]`. Сверь с `load_data` в
   `experiments/exp94_q2.py` (импорт в S6-код).
6. **MINOR-5 («чистый»)**: §4-Контроль и §7-смоук определяют «чистый» как самосогласованность
   (`emitted == top1_reported`), не CPU-референс. Сверь: определение flip в §4 — именно по
   собственному отчёту; смоук-ожидания в §7 используют ту же метрику (в коде: `m["flip"]`).
7. **MINOR-6 (решение утром)**: в §4 есть строка — после R1 канон E94 = body_template S6
   (`s6_sha` в S6.json), прежние артефакты (Q1/Q2/Q2rank/Q3/S3b) помечаются «DRY-on».
8. **MINOR-7 (no_probs)**: §4 flip-определение явно — 'no_probs' (пустой top_logprobs) НЕ флип
   (конвенция Q2rank), 'absent' (эмитента нет в непустом списке) — флип. Сверь с кодом
   `position_metrics_s6` (ветка no_probs: flip=False) и с конвенцией в `experiments/exp94_q2rank.py`
   (`fl = [r for r in rows if r["rank"] not in (1, "no_probs")]`).

## Формат ответа

Напиши в `docs/dynamic/2_REVIEW_E94_S6_TZ_r2.md`:

```
ВЕРДИКТ: PASS | PASS_WITH_MINORS | BLOCK

По каждому из 8 пунктов: ЗАКРЫТО / НЕ ЗАКРЫТО / ЧАСТИЧНО + 1-2 предложения (что именно сверено,
на каких строках/функциях).

Новые findings (если есть): [MAJOR|MINOR] §N: ...
```

Только чтение. Не меняй файлы, кроме итогового `2_REVIEW_E94_S6_TZ_r2.md`.
Финальная строка файла — вердикт без лишнего текста.
