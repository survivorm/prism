VERDICT: REVISE

## BLOCKER (2)

1. **§2.1 — фантомная readiness-строка.** Цитата: stderr-гейт по строке «server listening». Эта строка не существует в форке: `grep -rn 'server listening'` по всему дереву = 0 попаданий; реальная стартовая запись — SRV_INF `listening on %s` (tools/server/server.cpp:515). Гейт никогда не сработает → smoke зависает в ожидании готовности, до запросов не дойдёт. Фикс: гейт = GET /health (публичный роут, server.cpp:246) с таймаутом N с; stderr-строка — либо удалить, либо prefix `listening on` + ожидаемый URL (только INFO, не гейт).

2. **§8.1 vs §8.2 — список имён взят из сериализации, а §8.2 требует слать «только имена из п.1».** §8.1 перечисляет имена из `task_params::to_json` (tools/server/server-task.cpp:44-75 и 96-130: seed, temperature, top_k, top_p, min_p, repeat_penalty, presence_penalty, frequency_penalty, dry_multiplier, mirostat, max_tokens, n_predict, n_keep, stream, logit_bias, ...) — это набор sampling-параметров, а не поля запроса. В нём НЕТ `prompt`, хотя сервер требует его в теле: `if (!body.contains("prompt"))` (tools/server/server-common.cpp:1022). Правило §8.2 «только имена из п.1» делает запрос неконструируемым: валидное тело собрать по этому списку невозможно. Фикс: в §8.2 явно: `тело = prompt + имена из §8.1`; в §8.1 подписать список как «имена параметров (эхо в response), не полный набор полей запроса».

## MAJOR (3)

3. **§8.3 — «в ответе нет эха sampling-параметров» неверно.** `to_json_non_oaicompat` (server-task.cpp:~340-360) включает `generation_settings` — сериализацию всех параметров (server-task.cpp:44-75), и по умолчанию `to_json` имеет only_metrics=false (server-task.h:101). Ответ /completion несёт полное эхо (temperature, top_k, ...). Фикс: утверждение §8.3 убрать или заменить на «response содержит generation_settings; для проверки он не используется».

4. **§2b/§2.2b — `response.finish_reason` существует только в oaicompat-ответах.** Нон-OAI-ответ /completion содержит `stop_type`, `stopping_word`, `tokens_evaluated`, `tokens_predicted`, `tokens`, `prompt`, `tokens_cached`, `timings` (server-task.cpp:~340-360; dispatch TASK_RESPONSE_TYPE: server-task.cpp:~318-333). Запрос по TZ идёт в /completion (не-OAI-роуты: server.cpp:253-254), finish_reason там не будет. Фикс: заменить `finish_reason` на `stop_type` (и где нужно `tokens_evaluated`), либо явно перевести запросы на /v1/completions (server.cpp:255) и принять OAI-названия (prompt_tokens, finish_reason).

5. **§8.x — утверждение «grep tools/server/*.cpp = 0» для `penalty_last_n`/`dry_penalty` неверно.** В tools/server есть ЧЛЕНЫ: `{"repeat_last_n", sampling.penalty_last_n}` и `{"dry_penalty_last_n", sampling.dry_penalty_last_n}` (server-task.cpp:55, 61 и 106, 113). Нуль-хитов нет — речь идёт только об именах JSON-полей запроса (в них эти имена не входят; поля — `repeat_last_n`/`dry_penalty_last_n`). Фикс: переформулировать: «имена полей X не входят в тело запроса (поля — Y)», ссылаясь на server-schema.cpp; не смешивать имена членов struct с именами полей.

## MINOR (3)

6. **§2.1 canon-строка.** Заимствована из e93 CLI-вызова (experiments/e93_sycl_smoke.sh: llama-completion, `-e -ngl 99 -sm none -mg 0 --load-mode auto -fa on --temp 0 -s 1 -t 4`). Для llama-server: нет `-m <gguf>` (обязателен), а `-e` — короткая форма `--escape` (common/arg.cpp:1854), принимается без ошибки, но для сервера бессмысленна (`--embedding`, arg.cpp:3495, короткой формы не имеет). Фикс: canon = `-m /data/Qwen/qwen3.5-0.8B-f16.gguf -ngl 99 -sm none -mg 0 --load-mode auto -fa on -c 1024 -t 4 --port 8091` без `-e`; остальные флаги верифицированы: -t:1519, -c:1641, -fa:1756, -lm:2718, -ngl:2816, -sm:2834, -mg:2885, --host/--port: arg.cpp:3341/3348 (только LLAMA_EXAMPLE_SERVER).

7. **§8.1 — сверка имён покрывает только запрос, а ответные поля не названы, хотя уже известны из source:** `tokens_evaluated` (prompt-side count, server-task.cpp:350), `stop_type`/`stopping_word` (вместо finish_reason). Фикс: добавить строку с именами response-полей — сэкономит один спайк-раунд.

8. **§8.1 — не описана связка logprobs↔top_logprobs.** logprobs:true мапится в `n_probs = top_logprobs` (дефолт 20); top_logprobs без logprobs=true → throw (server-common.cpp:1381-1389). Фикс: пометка в §8.1: «logprobs и top_logprobs слать только парой».

## Что добавил бы (≤5)
1. Readiness-гейт на GET /health с таймаутом; убрать stderr-гейт (оставить логи INFO).
2. Разделить §8.1 на «поля запроса» и «имена параметров (эхо task_params::to_json)».
3. Таблицу response-имён для нон-OAI /completion (tokens_evaluated, stop_type, ...).
4. Canon-строку сервера с `-m <gguf>` и без `-e`.
5. Правило «logprobs+top_logprobs — парой».

## Что проверил
- Флаги common/arg.cpp (grep -n в /home/survivor/work/llama.cpp): -t:1519, -c:1641, -fa:1756, -e/--escape:1854, --no-cnv:1905, --load-mode/-lm:2718, -ngl:2816, -sm:2834, -mg:2885, --embedding:3495 (без короткой формы); --host/--port: arg.cpp:3341/3348, только LLAMA_EXAMPLE_SERVER; 'port' в tools/server/server-config.cpp — 0 hits.
- `grep -rn 'server listening'` по всему форку — 0 hits; реальная строка: server.cpp:515 `listening on %s`.
- Роуты server.cpp: GET /health:246 (public), POST /completion:253 (legacy), /completions:254, /v1/completions:255 (oai), /chat/completions:256, POST /tokenize:272.
- server-task.cpp: dispatch to_json (~318-333), to_json_non_oaicompat (~340-360: tokens, tokens_evaluated, tokens_predicted, generation_settings, prompt, stop_type, stopping_word, tokens_cached, timings; в token — id/logprob/top_logprobs), task_params::to_json (44-75, 96-130); server-task.h:101 (only_metrics=false по умолчанию), :342 (generation_params).
- server-common.cpp:1022 (обязательность `prompt`), :1381-1389 (logprobs↔top_logprobs), :1264-1341 (применение chat-шаблона в общем билдере, :1341 `llama_params["prompt"] = chat_params.prompt`).
- **Статус «применяет ли /completion чат-шаблон» — PARTIAL (не проверено до конца):** шаблон применяется в общем билдере параметров (server-common.cpp:1338 `common_chat_templates_apply` → :1341), а /tokenize (server-context.cpp:5068) ходит через `tokenize_mixed` без шаблона; идёт ли именно /completion (legacy, server.cpp:253) через общий билдер или через legacy-путь — end-to-end не проверено.
- Данные: results/e93/REPLAY.json — gate/gen_ids/gen_p1 (16x384), segments/onset/p1_med (по 16), key rep8 отсутствует (rep8 = mean(segments[8]) — вычисляемое); prompts.json — 16 docs w0-w15, n_tokens=64, roundtrip_fail=0, meta → data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json (sha16 083c732ff67c04f8, transformers 5.15.1 CPU); exp81b_replay.py: правило `ids = tk(text, add_special_tokens=False)['input_ids'][:64]`, off=64, 12 окон, seed 8181/2000.
- grep tools/server/*.cpp: `penalty_last_n|dry_penalty_last_n` → server-task.cpp:55,61,106,113 (члены sampling.*); `chat_template` → server-common.cpp:1264-1338, server-chat.cpp:579,639, server-context.cpp:1451+ (init templates, /tokenize-обработчик 5068+).
- **unverifiable:** end-to-end путь /completion через общий билдер параметров (см. PARTIAL выше); тело OAI-ответов /v1/completions (прочитан только dispatch); server не запускался, запросы не слались — все проверки статические по source и данным.

## Вердикт в одно предложение
r2 закрыл все текстовые находки r1, но ТЗ в текущем виде не исполнимо: фантомный readiness-гейт (B1) и самопротиворечие §8.1/§8.2 — «только имена из п.1», где список взят из сериализации task_params и не содержит обязательного `prompt` (B2) — до REVISE запуск не имеет смысла.

TZREV2A_DONE
