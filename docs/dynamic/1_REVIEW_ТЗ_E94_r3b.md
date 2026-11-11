

VERDICT: REVISE

Примечание: во время ревью ТЗ обновлён до r4 (8056f0c). Исправленные в r4 пункты помечены CLOSED; остальное открыто в r4. Новые проверки касались только r4-изменений (полей, которые r4 мог бы упустить).

## BLOCKER (1)
B1. §Запросы/Q2 «p_max = exp(tokens[0].logprob), top2 из top_logprobs» → массив tokens[] заполняется только при return_tokens:true (default false); в whitelist полей запроса return_tokens нет → tokens[] пуст, logprobs/top_logprobs отсутствуют, Q1/Q2 не считаются.
server-context.cpp:1839,2108; server-task.h:54; server-schema.cpp:34; README.md:591; test_completion.py проверяет tokens только с return_tokens.
Исправление: return_tokens:true или переход на completion_probabilities[]. CLOSED в r4: completion_probabilities[] — реальное поле (server-task.cpp:360; запись {id, token, bytes, logprob, top_logprobs} — server-task.cpp:282-300), печатается только при stream:false и непустом probs_output (server-task.cpp:358). Остаток: зафиксировать stream:false во всех /completion-запросах.

## MAJOR (4)
M1. §Device-гейт «GET /metrics: decode_device содержит SYCL/Iris» → эндпоинт требует флаг --metrics, struct server_metrics не содержит device-полей; decode_device/device_name — 0 совпадений grep в tools/server.
server-context.cpp:4651-4656; server-common.h:440-490. Исправление: независимое доказательство устройства. CLOSED в r4 (fd-доказательство /proc/<pid>/fd → /dev/dri).
M2. §Q2 «k = 65..448» → сдвиг с обоих концов: k=65 пропускает gen_p1[0] (условие = 64 id), k=448 не имеет CPU-пары (gen_ids кончается j=383 → k=447).
exp81b_replay.py:42-53. Исправление: k = 64..447. CLOSED в r4.
M3. §Спайк-гейт кэша «k=64/k=128: tokens_cached ≥ k-2» → первый запрос в свежий контекст попадает в пустой кэш (tokens_cached=0) → ложный RUN_RC=2 до единого измерения.
Исправление: гейт со второго (повторного) запроса того же k. CLOSED в r4 (гейт со 2-го запроса).
M4. §Статистика «медиана |Δrep8|», n=16, насыщенность 13/16 в 0.839–0.974 → пермутационный null формально корректен (перестановка фиксированного вектора rep8, Spearman c avg-ranks на связях — без смещения), но мощность низкая: один док сдвигает медиану на ~1 ранг, вывод по p-значению одному не безопасен.
Исправление: преддекларировать в ТЗ: отчитывать rho, p, полный per-doc массив и bootstrap-CI медианы; запрет выводов «механизм» без effect size. Открыто в r4 (не в списке фиксов).

## MINOR (4)
m1. §Q2 «top-2 hit rate» → top_logprobs:2 даёт ровно 2 значения (server-common.cpp:1381-1389): промах = ранк ≥3, цензура (реальный ранк 3..V неизвестен). Метрика честна, если подписана «miss = rank ≥3, censored». Открыто в r4: добавить подпись.
m2. §Prompt-count гейт: tokens_evaluated = n_prompt_tokens (server-task.cpp:349), но назначение n_prompt_tokens при cache_prompt не подтверждено на уровне строки → если считаются только новые токены, гейт (RUN_RC=13) лжесрабатывает на 2-х запросах Q2.
Unverifiable: исполнителю проверить на спайке (k=128 дважды: 2-й запрос tokens_cached ≥ 126 при tokens_evaluated = 128) и зафиксировать в FORMAT.md.
m3. §Perplexity-fallback (fact-check §2): ТЗ молчит о покрытии позиций — пишутся только first..n_ctx-2 при first=n_ctx/2 (perplexity.cpp:544,617-620), вход требует tokens ≥ 2·n_ctx (perplexity.cpp:480-484), uint16-квантизация → док 448 id надо пэдить до 2·n_ctx; при n_ctx=512 доступно лишь j≈192..383 (теряется ~половина позиций). Преддекларировать padding и покрытие в NOTES заранее.
m4. §Модель: r4 меняет на /data/Qwen/qwen3.5-0.8B-f16.gguf — существование файла unverifiable (no shell); проверить до старта. Для Q2 смена оправдана (движок ближе к CPU-fp32-референсу), RUN_RC=14 гейт на CPU-стороне и не затрагивается.

## Что добавил бы (≤5)
1. В спайке: проверка, что первый ответ содержит completion_probabilities[0].logprob и top_logprobs[1] — иначе STOP до Q1.
2. Зафиксировать stream:false во всех /completion-запросах (единственный режим выдачи completion_probabilities, server-task.cpp:358).
3. Bootstrap-CI медианы |Δrep8| в артефакт (n=16, насыщенность).
4. Преддекларированное покрытие позиций perplexity-fallback (j≈192..383 при n_ctx=512) в NOTES.
5. В NOTES: seed 8181 при temperature 0 — no-op, не баг, просто лишний параметр.

## Что проверил
- prompt как массив id: server-common.cpp:963-974 (json_prompt.get<llama_tokens>()), common_tokenize отсутствует; BOS только для string-prompt (комментарий :851) → совпадает с HF add_special_tokens=False.
- legacy /completion мимо чат-билдера: oaicompat_chat_params_parse server-common.cpp:1134, применение :1338; echo отклоняется (:1034-1035 → 500).
- Ответ /completion: server-task.cpp:340-362 — tokens_evaluated(:349), generation_settings(:350; to_json несёт repeat/presence/frequency penalty, xtc, mirostat, repeat_last_n — :58-137), stop_type(:354), tokens_cached(:356), timings(:357), completion_probabilities(:360).
- top_logprobs без logprobs → исключение (server-common.cpp:1381-1389); top_logprobs:2 → ровно 2 значения.
- return_tokens: server-schema.cpp:34, server-task.h:54, server-context.cpp:1839,2108, README.md:591.
- /metrics: требует --metrics (server-context.cpp:4651-4656); struct server_metrics (server-common.h:440-490) без device-полей; decode_device/device_name — 0 совпадений в tools/server.
- /tokenize: эндпоинт есть (server.cpp:272); request-поле «content» (server-context.cpp:5072-5081), add_special по умолчанию false (:5074) — совпадает с HF-пинами; ответ {"tokens"}. Фикс r4 на content корректен (это request-поле).
- Флаги: -sm :2834, -mg :2885, -fa :1756, -c :1641, -np :2546, -ngl :2816, --log-disable :3891 (common/arg.cpp) — все есть.
- Фантомы: anti_reverse, penalty_last_n (request-поле), dry_penalty, dynatop_p, echo, completion_level, tokenize_start, tokenize_count, n_input_pos — 0 совпадений grep в tools/server, include, common (форк 0b8cfb8fe).
- Сдвиг: exp81b_replay.py:42-53 (nxt=argmax(logits[0,-1]), p1=softmax[nxt]) → gen_p1[j]=p_max после prompt+gen[:j], позиция k=64+j; гейт 1e-3 ловит сдвиг на 1 позицию (|Δ|~0.01–0.1), бит-в-бит не гарантирует — для цели достаточно.
- REPLAY.json: верхние dict'и gen_ids (строка 854), gen_p1 (строка 7032); prompts.json: docs — массив key/ids, pin_file_sha16 083c732ff67c04f8, roundtrip 16/16.
- Q2 CPU-сторона: один forward по ids[:448] даёт logits всех позиций 0..447 → gen_p1[0..383] — корректно.
- run_one.sh: BUSY печатается с exit 0 (строки 9-10) → rc неоднозначен, требование ТЗ «читать LAUNCHED|BUSY» корректно; RUN_ONE_MIN_MEM-override, mem-poll 240×60s.
- PLAN.md: i915/IGC нестабилен после rapid kill/relaunch → пауза ≥5 s обоснована.
- Unverifiable: существование /data/Qwen/qwen3.5-0.8B-f16.gguf; decode 6.3 t/s (замер r4); n_prompt_tokens при cache_prompt (на уровне строки).

## Риск, который оставляю открытым
Семантика tokens_evaluated/n_prompt_tokens при cache_prompt не подтверждена на уровне строки: если считаются только uncached-токены, prompt-count гейт (RUN_RC=13) лжесрабатывает на повторных запросах Q2. Проверить на спайке (B1-чек + повтор k=128) и зафиксировать в FORMAT.md до Q1/Q2.

TZREV3B_DONE
