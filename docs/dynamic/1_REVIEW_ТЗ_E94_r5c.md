# Ревью ТЗ E94 r5 (коммит c14ddac) — ревьюер C, слепой по форме

VERDICT: REVISE (1 MAJOR: спайк-гейт §2.2 всё ещё на мёртвом поле `tokens_cached`, противоречие внутри §2; одна строка текста)

## CLOSED (19) — 10 из r3a (3B/5M/3m), 9 из r3b (1B/4M/4m); совпадения r3a/r3b посчитаны по одному разу в списках
- r3a-B1: device-гейт `/metrics`/`decode_device` → fd-доказательство `/proc/<pid>/fd` → `/dev/dri/*`, RUN_RC=10, `--metrics` лишь вторично (§2) — CLOSED; `/dev/dri` подтверждён на диске (card1, renderD128).
- r3a-B2: якорь Q4_K_M → `/data/Qwen/qwen3.5-0.8B-f16.gguf` (§1, §8.6) — CLOSED; `ls`: f16 1485.5M на месте, Q4_K_M в /data/Qwen нет.
- r3a-B3: кэш-гейт ложно срабатывал на 1-м запросе → «со второго запроса дока» + /slots-поля (§2) — CLOSED.
- r3a-M1: k=65..448 → k=64..447, j=k-64, k=448 не шлём (§2.2) — CLOSED; сдвиг чист (см. «Логика ядра»).
- r3a-M2: `tokens[]` = int-массив, вероятности в `completion_probabilities[]` (§2) — CLOSED; grep: server-task.cpp:360, условие `!stream && !probs_output.empty()` :358.
- r3a-M3: фантом `timings.n_prompt` — имени нигде в тексте r5; гейты на /slots + клиентский wall-clock (§2, §2.2) — CLOSED.
- r3a-M4: `/tokenize {prompt}` → `{content}` (§2.4) — CLOSED; grep: server-context.cpp:5071-5076 `body.count("content")` → `tokenize_mixed(..., body.at("content"), ...)`, `add_special` дефолт false.
- r3a-M5: смета 280 t/s → 6.3 t/s, ~33 мин, потолки Q1 40/Q2 40/общий 90, «это НЕ 15 мин» (§2.2) — CLOSED.
- r3a-m1: `content_txt` → `content` (§2.1) — CLOSED; grep: `content` в ответе server-task.cpp:343.
- r3a-m2: `seed:8181` при temp 0 — помечено «no-op, не баг» (§2) — CLOSED.
- r3b-B1: `return_tokens:true` в whitelist, id из `completion_probabilities[].id`, кросс-чек `[p.id] == tokens` иначе RUN_RC=2, `stream:false` везде (§2, §8.1) — CLOSED; grep: server-schema.cpp:34, server-task.h:54 (default false), server-context.cpp:1839.
- r3b-M1: device-гейт → fd — CLOSED (совпало с r3a-B1).
- r3b-M2: k=65..448 → k=64..447 — CLOSED (совпало с r3a-M1).
- r3b-M3: кэш-гейт с 1-го запроса → со 2-го — CLOSED (совпало с r3a-B3).
- r3b-M4: мощность n=16 → per-doc массивы, bootstrap-CI медианы (10000, seed 8181), rho+p null, запрет «эффект/механизм» при CI∋0 или ES < ширины полосы 0.839–0.974 (§8.3) — CLOSED.
- r3b-m1: top-2 censored подпись «miss = ранк ≥3, истинный ранг цензурирован» + доля `len(top_logprobs)<2` отдельным числом (§8.4) — CLOSED.
- r3b-m2: prompt-count при cache_prompt → `n_prompt_tokens = slot.task->n_tokens()` = полная длина промпта, а не «новые» токены (§2) — CLOSED: проверено по коду, server-context.cpp:2116 ровно это присваивает; живой спайк k=128 → FORMAT.md оставлен как runtime-чек.
- r3b-m3: perplexity-fallback — покрытие first..n_ctx-2 (first=n_ctx/2), вход ≥2·n_ctx, uint16, n_ctx=512 → j≈192..383, padding ДО запуска (§8.5) — CLOSED.
- r3b-m4: существование f16 gguf → §8.6 + мой независимый `ls` (размеры совпадают до знака: 1485.5M/2958.6M) — CLOSED.

## OPEN (1)
- r3a-m3: §2.1 «repro_rate … (одинаковые флаги)» — формулировка r5 не уточнила (косметика; смысл понятен: идентичные запросы r=1/r=2).

## FIXED-WRONG (1) — сломано/оставлено текстом r4/r5
- §2.2 спайк-гейт: «со второго запроса `tokens_cached ≥ k-2`, а клиентский wall-clock …» — сам §2 объявляет `tokens_cached` мёртвым: = `slot.prompt.n_tokens()`, т.е. длине промпта (server-context.cpp:2118, проверено), и `tokens_cached ≥ k-2` тавтологично (k ≥ k-2, истинно уже с ПЕРВОГО запроса — квалификатор «со второго» неуместен). r4/r5 пересобрали Q2-гейт кэша на /slots-поля, но спайк-гейт переписали наполовину: условие деградировало до wall-clock-одного. MAJOR (обязательный гейт до метрик; литеральный исполнитель получит валидное, но ничего не измеряющее условие). Чинить: переписать на те же поля Q2-гейта (`n_prompt_tokens_processed ≤ 2`, `n_prompt_tokens_cache ≥ k-2` при `n_prompt_tokens == k`) или явно только на wall-clock.

## Новые BLOCKER/MAJOR/MINOR
- MAJOR: тот же, что в FIXED-WRONG (не дублирую). BLOCKER и MINOR новых: нет.
- Фантомы r4/r5 (return_tokens, completion_probabilities, GET /slots + 3 поля, /dev/dri, gguf-якорь, /health, content в /tokenize, generation_settings, stop_type, tokens_evaluated) — ни одного: все имена и строки подтверждены grep/ls (ниже). Семантика /slots-гейта корректна: n_prompt_processed — обработанные БЕЗ кэша токены (server-context.cpp:3393-3394 сброс в 0 при n_past>0, :3491 += новых), n_prompt_cached = n_past (:3393) → «≤2» и «≥k-2» меряют именно переиспользование префикса.

## Логика ядра
- ids-промпт 448 = 64 prompt + 384 gen; probe k=64..447: позиция k = распределение после ids[:k]; gen_p1[j] = p_max после prompt+gen[:j], j=0..383 ⇒ j=k-64 покрывает ровно gen_p1[0..383] (k=64→j=0; k=447→j=383 — последний gen-токен); k=448 корректно не шлём (за последним токеном предсказывать нечего). prompt-count гейт `tokens_evaluated == len(отправленных)` стабилен при cache_prompt (n_prompt_tokens = полная длина, :2116). CPU: один forward по ids[:448] даёт logits на 0..447, позиция k ↔ logits[k]; гейт max|Δ|<1e-3 против REPLAY.gen_p1 зафиксирован. Связность: OK.

## Что проверил (выводы команд)
- `ls -la /data/Qwen/*.gguf`: `qwen3.5-0.8B-f16.gguf 1485.5M`, `qwen3.5-0.8B-f32.gguf 2958.6M` (плюй чужой K2-Horizon-7B-Q8_0.gguf; Q4_K_M нет).
- `ls /dev/dri`: `by-path/ card1 renderD128` — fd-гейт исполним.
- `grep -n "n_prompt_tokens_processed" tools/server/server-context.cpp` → `:701:res["n_prompt_tokens_processed"] = stats.n_prompt_processed;`; рядом `:700 n_prompt_tokens = prompt.tokens.size()`, `:702 n_prompt_tokens_cache = stats.n_prompt_cached`.
- `grep -n "n_prompt_tokens" server-context.cpp` → `:2116 res->n_prompt_tokens = slot.task->n_tokens()`, `:2117 res->n_prompt_tokens_cache = slot.stats.n_prompt_cached`, `:2118 res->n_tokens_cached = slot.prompt.n_tokens()` (мёртвость подтверждена строкой).
- `grep -n "return_tokens" server-schema.cpp server-task.h server-context.cpp` → :34 / :54 (default false) / :1839 — цитаты ТЗ совпали.
- `grep -n "completion_probabilities" server-task.cpp` → :360 (non-OAI) и :1068 (OAI); условие печати `if (!stream && !probs_output.empty())` — :358.
- `grep -n "generation_settings|stop_type|tokens_evaluated|tokens_cached" server-task.cpp` → :349 `tokens_evaluated = n_prompt_tokens`, :350 `generation_settings = generation_params.to_json()`, :354 `stop_type`, :356 `tokens_cached = n_tokens_cached`, :357 `timings = stats.to_json()` — таблица полей §2 ТЗ верна.
- `grep -n '"/health"|"/slots"' server.cpp` → :246 /health, :285 /slots — цитаты ТЗ совпали.
- Чтение server-context.cpp:5071-5076: `/tokenize` читает `body.count("content")` → `tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special)`, `add_special` дефолт false — совпадает с HF-пинами; ТЗ цитирует :5072-5077 (сдвиг на 1 строку, не дефект).
- Чтение server-context.cpp:3393-3394/3491: `n_prompt_cached = n_past`, `n_prompt_processed = 0` при старте, `+= n_tokens_out` после графа — семантика Q2-кэш-гейта корректна.
- Unverifiable без запуска сервера: живое переиспользование префикса между запросами (ради этого спайк-гейт), поведение n_past при cache_prompt:false после Q1 (гейт сознательно пропускает 1-й запрос дока).

TZREVC_DONE
