VERDICT: REVISE

## BLOCKER (3)

1. **§2 — device-гейт неисполним: `/metrics` без `--metrics` мёртв, `decode_device` не существует.**
   Цитата: «устройства — GET `/metrics` (`decode_device` должен содержать SYCL/Iris, иначе `RUN_RC=10`)».
   Ломается двояко: (a) в этом форке `/metrics` работает только с флагом `--metrics` — без него
   endpoint вернёт ошибку «This server does not support metrics endpoint. Start it with `--metrics`»
   (server-context.cpp:4652-4656), а канон-команда §2 флага не содержит; (b) строка `decode_device`
   даёт 0 grep-попаданий по всему дереву, а вывод `/metrics` — Prometheus-текст с фиксированными
   counters/gauges (server-task.cpp:1523-1590), полей о устройствах там нет. Итог: гейт как написан
   всегда «устройство не найдено» → `RUN_RC=10` на каждом старте (или исполнитель молча скинет гейт).
   Фикс (измеримо): device-гейт = grep лога сервера по `loaded SYCL backend` (ggml-backend-reg.cpp:259)
   и/или SYCL-таблице устройств (ggml-sycl.cpp:226-245); `--metrics` в канон-команде не нужен.

2. **§1 — якорная модель отсутствует на диске.**
   Цитата: «`/data/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf`».
   Ломается: `ls /data/Qwen/` — Q4_K_M GGUF нет (find -iname '*Q4_K_M*' по /data — только чужие
   модели); `/data/Qwen/Qwen3.5-0.8B/` — HF-каталог (safetensors 1.7 G, нет GGUF). Канал e93, из
   которого ТЗ берёт ВСЮ смету, — `/data/Qwen/qwen3.5-0.8B-f16.gguf` (experiments/e93_sycl_smoke.sh:9,
   results/e93/sycl/bench_all.txt:1; файл на месте, 1485 MB). Запуск с якорем из ТЗ невозможен
   (или требует неконтролируемого шага квантизации + новой сметы).
   Фикс (измеримо): якорь = `/data/Qwen/qwen3.5-0.8B-f16.gguf` (канал e93, тайминги валидны как есть) — либо явный шаг «сделать Q4_K_M» с новой сметой и бенчем до Q1.

3. **§2 — гейт кэша `tokens_cached ≥ k-2` ложно срабатывает на первом запросе каждого дока.**
   Цитата: «**гейт кэша (для Q2):** `tokens_cached ≥ k-2` и отсутствие роста `timings` prompt-части с `k`».
   Ломается: первый Q2-запрос дока (k=64/65) не может ударить в кэш — после Q1 в слоте чужая
   последовательность, общий префикс ≈ 0 → `tokens_cached = 0 < k-2` → гейт падает на ПЕРВОМ
   запросе каждого дока и на самом спайке (k=64 — первый Q2-запуск) → ложный `RUN_RC=2` «маршрут
   закрыт» при работающем кэше.
   Фикс (измеримо): `tokens_cached ≥ k-2` применять только со второго запроса дока (k ≥ k_first+1);
   спайк-гейт оставить на росте prompt-части timings: при живом кэше новых токенов на k=64 ≈ 64,
   на k=128 ≈ 65 (prompt_ms ~константа); при мёртвом — 64 → 128 (линейный рост).

## MAJOR (5)

4. **§2.2 — off-by-one: диапазон k не совпадает с «всеми позициями 64..447» и gen_p1.**
   Цитата: «для каждого `k = 65..448` → позиция `k−1`» + «все позиции 64..447» + «`gen_p1[j]` — …
   teacher-forced позиция `k = 64 + j`».
   Ломается: по собственной конвенции ТЗ gen_p1[j] ↔ k=64+j, т.е. ровно gen_p1[0..383] покрывает
   k=64..447. Фактический k=65..448: позиция 64 (= gen_p1[0]) не пробуется вовсе, а k=448 предсказывает
   токен ЗА 448-id потоком (референса нет). Утверждение «все позиции 64..447» ложно — фактически 65..448.
   Фикс (измеримо): `k = 64..447` (384 запроса/док, ровно gen_p1[0..383]); CPU-гейт 1e-3 не трогать.

5. **§2/§2.1/§2.2 — схема ответа: `tokens[]` в этом форке — массив int, а не объекты с logprob.**
   Цитата: «`tokens[]` с `id`/`logprob`/`top_logprobs`» / «`p_max = exp(tokens[0].logprob)`,
   `top1_id = tokens[0].id`, `top2` из `top_logprobs`».
   Ломается: в ответе нон-OAI `tokens` = `llama_tokens` — просто массив id (server-task.h:322,
   server-task.cpp:344, server-context.cpp:2103-2106: `res->tokens = slot.generated_tokens`).
   Пер-позиционные `logprob`/`top_logprobs` лежат в `completion_probabilities[]`
   (server-task.cpp:359-361, probs_vector_to_json:282-302: {id, logprob, top_logprobs:[…]}),
   все 384 позиции при non-stream + n_probs>0 (server-context.cpp:2136-2143). `tokens[0].logprob`
   не существует → извлечение Q1/Q2 по тексту ТЗ невозможно (ТЗ при этом утверждает «выдуманных
   имён в ТЗ нет»).
   Фикс (измеримо): id-стрим — `tokens[]`; позиция i: `p_max = exp(completion_probabilities[i].logprob)`,
   `top1_id = completion_probabilities[i].id`, `top2 = completion_probabilities[i].top_logprobs[0]`
   (при n_predict:1 — ровно 1 элемент).

6. **§2.2 — фантом `timings.n_prompt` в спайк-гейте кэша.**
   Цитата: «по `timings` ответа `n_prompt`/`prompt_ms` должны остаться ~константой».
   Ломается: `timings` = server_slot_stats::to_json (server-common.cpp:63-75) — поля:
   `cache_n, prompt_n, prompt_ms, prompt_per_token_ms, prompt_per_second, predicted_n, predicted_ms,
   predicted_per_token_ms, predicted_per_second`; `n_prompt` там нет (правильно `prompt_n`).
   Фикс (измеримо): гейт по `prompt_n` (~1, новые токены), `cache_n` (~k−1, растёт), `prompt_ms`
   (~константа).

7. **§2.4 — запрос `/tokenize` принимает поле `content`, а не `prompt`.**
   Цитата: «16 запросов `/tokenize {prompt: <текст дока>}`».
   Ломается: обработчик читает только `body["content"]` (server-context.cpp:5072-5077); тело
   `{prompt: …}` → токенов нет → Q4-запросы падают/пустые, `tok_drift` не считается.
   Фикс (измеримо): `/tokenize {content: <текст дока>}` (ответ — массив id).

8. **§3 — смета: «pp ≈ 280 t/s» — это f16 pp512-бенч, а не замер канала; Q1 занижен ~в 2 раза.**
   Цитата: «e93-смета (загрузка 21 с, pp ≈ 280 t/s, decode ≈ 6–11 t/s) → Q1 ≈ 15 мин».
   Ломается: реальные замеры e93 (results/e93/sycl/NOTES.md:22-23 из smoke_err.log): prompt eval
   **55.9 t/s**, decode **6.28 t/s** (load 21138 мс — первичный JIT). 280 t/s — строка f16 pp512 из
   bench (bench_all.txt: 279.9±3.8), не канал. Q1 при 6.28 t/s: 32 req × 384 ток ≈ 33 мин
   (оптимум 11 t/s → ≈19 мин), а не 15 — смета в NOTES будет неверна.
   Фикс (измеримо): пересчитать смету из smoke-цифр (Q1 ≈ 30–33 мин) либо замерить спайком; явно
   указать, что потолок 40 мин — на фазу Q2, а у Q1 потолка нет (или добавить потолок Q1).

## MINOR (3)

9. **§2.1 — фантом `content_txt`.** Поле ответа — `content` (server-task.cpp:344); grep 'content_txt'
   по дереву = 0. Фикс: `content_txt` → `content`.
10. **§2.1 — `seed: 8181` при `temperature: 0`** не влияет на жадный стрим; в NOTES пометить, что
    воспроизводимость проверяется r=1/r=2 (repro_rate), а не сидом.
11. **§2.1 — «repro_rate … (одинаковые флаги)»** — неясная фраза; читать как «r=1 и r=2 — идентичные
    запросы (те же поля)». Уточнить формулировку.

## Что добавил бы (≤5)

1. Таблицу реальных полей ответа нон-OAI `/completion` (`content`, `tokens` int[],
   `completion_probabilities[]`, `tokens_evaluated`, `tokens_cached`, `timings{cache_n,prompt_n,
   prompt_ms,…}`, `generation_settings`, `stop_type`) — половина находок этого ревью — имена, таблица
   экономит спайк-раунд.
2. Device-гейт через grep лога (`loaded SYCL backend`, ggml-backend-reg.cpp:259) + дублирование
   строк в FORMAT.md.
3. Явный потолок времени Q1 и смету по фазам из smoke-цифр (55.9 / 6.28 t/s).
4. Проверку в спайке: `len(completion_probabilities) == n_predict` (384 у Q1, 1 у Q2) — дешёвое
   подтверждение схемы одним запросом.
5. Правило идемпотентности для `raw/probe_<doc>.jsonl`: строка на позицию не перезаписывается
   (аналог Q1) — сейчас это оговорено только для q1_*.json.

## Что проверил

- llama.cpp `~/work/llama.cpp` (HEAD 0b8cfb8fe), статически, сервер НЕ запускался:
  - grep 'content_txt' по tools/server = 0; `{"content",…}` server-task.cpp:344.
  - `tokens` в ответе = `llama_tokens` (server-task.h:322; server-context.cpp:2103-2106);
    `completion_probabilities` — server-task.cpp:359-361, probs_vector_to_json:282-302;
    probs_output — все сгенерированные токены, non-stream, n_probs>0 (server-context.cpp:2136-2143).
  - `timings` — server_slot_stats::to_json, server-common.cpp:63-75 (каменные имена полей выше).
  - `/metrics`: роут server.cpp:248; гейт `params.endpoint_metrics`, ошибка «Start it with `--metrics`»
    server-context.cpp:4652-4656; вывод — Prometheus-текст, фикс. набор (server-task.cpp:1523-1590);
    grep 'decode_device' по tools/server, common, ggml/src = 0.
  - Строка загрузки бэкенда: ggml-backend-reg.cpp:259; SYCL-таблица устройств: ggml-sycl.cpp:226-245.
  - Поля запроса (server-schema.cpp): cache_prompt:31, n_predict:44, xtc_probability:104,
    repeat_last_n:126, dry_base:142, mirostat:157, seed:176, alias logprobs:180 — список §2 ТЗ чист.
  - `/tokenize`: тело — `content` (server-context.cpp:5072-5077); ответ {content, tokens}.
  - BOS/массив id: подтверждено ранее (server-common.cpp:851, 963-974) — не перепроверял.
- Файлы: `/data/Qwen/` — qwen3.5-0.8B-f16.gguf (1485 MB) и -f32.gguf на месте; GGUF *Q4_K_M* под
  /data — нет (find -iname); Qwen3.5-0.8B/ — HF safetensors.
- e93: NOTES.md:22-23 (55.9/6.28 t/s, load 21138 мс), bench_all.txt:1 (f16), smoke.sh:9 (модель).
- results/e93/prompts.json: 16 доков × 64 ток; цепь пинов цела: sha256(pin_file)[:16] =
  083c732ff67c04f8 (то, что в meta как pin_file_sha16); 16/16 ids воспроизводятся токенизатором
  (add_special_tokens=False) из текстов пина; text==decode(ids) 16/16.
- results/e81/REPLAY.json: верхние ключи [C_tail_vs_rep8, docs, gate, gate_pass, gen_ids, gen_p1] —
  gen_ids/gen_p1 верхние словари, как в §1 ТЗ; rep8 = mean(segments['8']), off=64, 12×32 (exp81b).
- run_one.sh: BUSY → exit 0, LAUNCHED в stdout, RUN_ONE_MIN_MEM, RUN_RC в /tmp/<name>.log (совпадает с §3 ТЗ); df: /home 14 G, /tmp 15 G свободны (≥6 G — ок).
- **unverifiable (сервер не запускался):** реальное поведение prefix-кэша между запросами (именно
  ради этого спайк-гейт), тело ответа end-to-end (все утверждения о схеме — из source),
  воспроизводимость gen_p1 свежим transformers-proходом (гейт 1e-3 — runtime).

## Вердикт в одно предложение

r3 убрал текстовые дефекты r2 и якорь формул честный, но три гейта делают прогон не исполнимым в
исписанном виде (мёртвый device-гейт, несуществующая Q4_K_M-модель, ложный срыв кэш-гейта на первом
запросе дока), а схема ответа описана тремя фантомными именами — REVISE.

TZREV3A_DONE
