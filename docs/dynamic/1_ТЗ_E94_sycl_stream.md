# ТЗ E94 — llama.cpp-SYCL (iGPU) против CPU-референса: тот же стрим, та же полка

Статус: **r2** после ревью-лупа r1 (1_REVIEW_ТЗ_E94_r1a.md, 1_REVIEW_ТЗ_E94_r1b.md).
Цель, вопросы Q1–Q3 и инварианты из r1 сохранены; механизм заменён целиком
(по BLOCKER r1a-1 / r1b-1), метрики и гейты доопределены. Ревью r2 сверяет
ДЕЛЬТУ против r1 и перечитывает целое.

## 0. Постановка

Тот же вопрос, что в r1: та же модель (Qwen3.5-0.8B f16 GGUF), те же 16 wiki-
доков, тот же greedy 384 шага, движок llama.cpp SYCL (iGPU) против
transformers-CPU-референса `results/e81/REPLAY.json`. Q1 — префиксы id-стримов;
Q2 — rep8 по 16 докам (основная метрика — медиана |Δrep8|; rho — вспомогательная
с null); Q3 — p1 на выровненных позициях.

Оговорки измерения (в шапку NOTES): (a) «greedy» в этом билде = argmax ПОСЛЕ
top-k(40)/top-p(0.95)/min-p(0.05) при temperature=0 (r1a-F15) — расхождения на
низковероятных позициях ожидаемы; (b) сравнение мер, а не битов; (c) exploratory,
GO/NO-GO нет, «механизмов» в NOTES не пишем.

## 1. Единственные источники

- Промпты: `data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__
  plain.json`, правило загрузки дословно из exp81b_replay.py:24–27:
  `ids_i = tk(texts[i], add_special_tokens=False)['input_ids'][:64]` (ПЕРВЫЕ 64).
  Сверка: `results/e93/prompts.json` (meta.rule, roundtrip 16/16) — id 1-в-1;
  в шапку артефакта sha256(pin-файла)[:16] (ожидается 083c732ff67c04f8) и sha256
  списка 16×64 id. `results/e64/tpu` — НЕ источник промптов (там только метрики).
- Референс: `results/e81/REPLAY.json`: `gen_ids` (16×384), `gen_p1`,
  `docs.w*.segments["8"]` (12 сегментов). Скаляр rep8 := mean(segments[8])
  (ровно C-строка exp81b: `float(np.mean(res["docs"][k]["segments"][8]))`).
- Модель GGUF: `/data/Qwen/qwen3.5-0.8B-f16.gguf`; HF-веса `/data/Qwen/
  Qwen3.5-0.8B` — только токенизатор. vocab GGUF 248,044 (r1a-F8).
- Билд: `~/work/llama.cpp/build/bin` (тот же, что e93; в шапку ОБА git_sha:
  prism и llama.cpp-форка). Флаги сервера — канон e93: source setvars.sh,
  ZES_ENABLE_SYSMAN=1, `-ngl 99 -sm none -mg 0 --load-mode auto -e -fa on
  -c 1024 -t 4 --port 8091`. `-no-cnv` не нужен (raw-эндпоинт /completion;
  проверить спайком, п.3.1). DMMV не ставить. Правки форка запрещены (read-only).

## 2. Метод (один llama-server на весь прогон)

2.0 Pre-flight (часть харнесса, ДО сервера): pgrep -f
'llama-(completion|server|bench|perplexity)' минус своё дерево — чужой живой
llama = ждать поллом 30 с (макс 2 ч), иначе аборт с записью; ss -ltn — 8091
свободен, иначе аборт (не занимать чужой порт). ps-снимок сосуществования
( exp91/e83 ) — в NOTES как операционный факт.
2.1 Запуск сервера ОДИН раз: stderr → `results/e94/runs/server.stderr.log`.
Гейт старта: в stderr есть `[TEMP-SDPA] Intel(R) Iris(R) Xe Graphics` и
`server listening`; ждём готовности поллом /health.
2.2 На каждый док w0..w15, раунды A и B (оба с ИДЕНТИЧНЫМИ параметрами):
  a) `POST /tokenize` с точным текстом пина → сравнить с HF-id (правило §1).
     Поле `prompt_match` ∈ {ok, mismatch}; mismatch → док НЕ интерпретируется
     в Q1/Q3 (prefix_len такого дока — не метрика движка), в Q2-медиану идёт
     только при prompt_match=ok; число mismatch — в шапку.
  b) `POST /completion` {prompt: текст, n_predict: 384, temperature: 0,
     logprobs: true, top_logprobs: 2, cache_prompt: false, stream: false,
     repeat_penalty: 1.0, frequency_penalty: 0.0, presence_penalty: 0.0,
     dry_multiplier: 0.0, xtc_probability: 0.0, mirostat: 0} — ВСЕ
     sampling-штрафы обнуляются ЯВНО: server может подхватить их из GGUF
     метадатов модели (common.cpp:1267 читает samplers.penalty_repeat), а
     штраф повторов исказил бы именно повторный стимул; факт применения
     проверяется, чем позволит лог-уровень сервера (spike FORMAT.md зафиксирует,
     печатается ли строка sampler-цепочки и при каком -v; если не печатается —
     единственный источник истины: сам запрос + сохранённый sample ответа). Остальные поля запроса, если схема этого
     форка их не знает (спайк), НЕ засылать: unknown-поле может уронить 400.
     Ожидаемая цепочка при temp=0: top-k(40)/top-p(0.95)/min-p(0.05) без
     штрафов — как в e93 smoke (задокументировать в NOTES).
     → полный ответ сохраняется СЫРЫМ в `results/e94/runs/w{i}_{A|B}.json`.
     Из него: `tokens[].id` — id-стрим; p1_llama[t] := exp(tokens[t].logprob);
     margin[t] := exp(top2.logprob) − exp(top1.logprob) (r1b-8; на выровненных
     позициях top1==выбранный, это anti-flip страж, не «маргин»).
     Формат-гейт на ответ: len(ids)==384 (или стоп-токен раньше — зафиксировать
     фактическую длину и причину по response.finish_reason), поле prompt-стороны
     (usage/timing prompt tokens — точное имя уточняется спайком) == 64,
     иначе док бракуется (аналог «prompt eval / 64 tokens», r1a-2b).
  c) пауза ≥2 с между запросами; сбой/ханг запроса → один retry (повтор того же
     запроса), повтор в runs/ с суффиксом `_retry`; ханг сервера → перезапуск
     сервера (пауза ≥5 с, IGC-flaky), один раз за прогон, затем аборт. Всё в
     NOTES как операционные факты.
2.3 Метрики (только из сохранённых сырых ответов + референса):
  Q1: prefix_len := длина общего префикса prompt+gen id (llama: prompt-id из
     /tokenize ok-доков; cpu: 64 id + gen_ids[i]) по gen-позициям 0..383.
  Q2: rep8_llama по формуле exp81b ТОЧНО: flags=rep_flags(prompt_ids+gen_ids,8)
     по всей 448-последовательности; segments[k]=mean(flags[64+a:64+a+32]),
     a=0..11; rep8_scalar=mean(segments). Функцию rep_flags скопировать из
     exp81b дословно; off=64 ассертится (prompt длины 64). Отчёты: медиана
     |Δrep8| по ok-докам (основная), Spearman rho по 16 точкам рангами со
     СРЕДНИМИ при тайах + пермутацонный null 2000 (seed 8181, перестановка
     одной стороны, перцентили 2.5/97.5) — по образцу блока C exp81b; число
     различных rep8-значений в каждой стороне (диагностика тайов).
  Q3: на общих префиксах (id_llama[t]==id_cpu[t] ⇒ один id в обеих softmax):
     corr(p1_llama, p1_cpu) по Пирсону; гейты: n_aligned≥32 И std(p1_llama)≥1e-4
     И std(p1_cpu)≥1e-4, иначе «не измерено (prefix=X)». Отчёт per-doc и сводка.
  Q4 (детерминизм): repro := доля совпавших gen-id между A и B на 384 позициях,
     per-doc; медиана по 16. Порог интерпретации: медиана repro <0.95 → в шапку
     NON-DETERMINISTIC, Q1/Q3 помечаются «не интерпретируются» (Q2 не трогает —
     rep8 по обоим раундам, в артефакт обе оценки).
2.4 Смерть сервера: по завершении 32 запросов — kill по PID, пауза. pkill
запрещён.

## 3. Спайк ( Writer-B , до кода метрик; один живой сервер)

3.1 Микро-сессия на сервере (флаги §1): /health; POST /tokenize на текст w0;
POST /completion на w0 с n_predict=32 и logprobs:true, top_logprobs:2 → полный
фактический JSON (обрезать до 3 токенов для образца) в
`results/e94/spike/sample_completion.json`; проверка, что raw-эндпоинт НЕ
применяет chat-шаблон (prompt-token-count == 64-префикс; для микропрогона
допустим сокращённый промпт, но счётчик сверяется с /tokenize). Запись: имена
полей id/logprob/top_logprobs/prompt-count/finish_reason, формат ошибок,
`FORMAT.md`.
3.2 Детерминизм-микро: тот же запрос ×2 → доля несовпадений id (полный r=2
по всем 16 придёт основным прогоном; здесь только грубая проверка механизма).
3.3 Результат спайка — FORMAT.md + sample; ТЗ НЕ правится (r1a-9); расхождение
с ожиданиями харнесса — дефект харнесса, правит Writer-A на само-аудите.

## 4. Артефакты и гейты харнесса

`results/e94/`: `runs/` (server.stderr.log, w{i}_{A|B}[,retry].json сырые),
`spike/`, `stream.json`, `NOTES.md`.
`stream.json`: шапка {git_sha:{prism,llama_fork}, gguf_sha12, pin_sha16,
ids_sha256, флаги сервера, vocab, n_requests, prompt_match число,
repro_медиана, NON-DETERMINISTIC flag}; на документ: prompt_match, ids_llama
(A,B — 2×384 int), p1_llama (2×384 float), margin (A), rep8_llama(A,B),
rep8_cpu, prefix_len, n_aligned, p1 corr+std, repro-doc.
Гейты харнесса (жёсткие, проверяются скриптом, не глазами): 32/32 ответов с
format-гейтом (len==384 или фикс. стоп, prompt-count==64); server stderr гейт
(§2.1); ни одного чужой-llama пересечения (pre-flight журнал); размер id/p1
стрим в stream.json == 384; повторный расчёт rep8/prefix из сырых runs/
совпадает со stream.json (само-пересчёт встроен в харнесс: режим --verify).
Научных порогов нет (exploratory).

## 5. Ограничения и логистика

- НЕ трогать: процессы/файлы exp91-e83, форк (read-only), чужие results/.
  pkill запрещён; свои — по PID.
- Один llama на машину: pre-flight §2.0 обязателен и документируется.
- Прогон через `RUN_ONE_MIN_MEM=3 ./run_one.sh e94 ...`; в stdout читать
  LAUNCHED/BUSY (BUSY даёт rc=0!), лог /tmp/e94.log; не перепускать дважды без
  разбора лога.
- Смета: загрузка cold ~21 с + 32×(~45–65 с tg + ~0.3 с pp) ≈ 25–35 мин +
  паузы; ориентир зависания: >2.5 ч без 32 ответов.
- Wall-time и операционные ретраи — в NOTES (не наука).

## 6. Исполнители (после закрытия ревью r2)

- Writer-A: `experiments/e94_sycl_stream.py` — pre-flight, lifecycle сервера,
  32 запроса, парсинг, метрики §2.3, stream.json + --verify, вызов через
  run_one-обёртку `experiments/e94_run.sh`.
- Writer-B: спайк §3 → `results/e94/spike/FORMAT.md` + sample (+ микро-сервер
  убить по PID). Пишет только в results/e94/spike/.
- Слияние: парсер/имена полей A подгоняются под FORMAT.md на само-аудите
  (grep-проверка имён полей против sample). Затем: смоук (1 док ×A,
  n_predict=32 отдельным запуском не делать — смоук = харнесс в режиме
  --limit 1), полный прогон run_one, само-ревью артефакта (--verify +
  ручная выборка), ВНЕШНИЙ ревью кода+артефакта, коммит.

## 7. Чек-лист ревью r2 (сверяем дельту)

1. Исчезли ли r1-BLOCKER: фантом-флаги (--logits-all, --dump-logits, -cd),
   чат-шаблон (ныне: raw-эндпоинт + prompt-count-гейт + /tokenize pre-gate)?
2. Реп-8: точная копия формулы exp81b, off=64, mean(segments[8]); null-модель
   и тай-коррекция в Q2; медиана |Δ| как основная?
3. Q3 гейты (n≥32, std>eps), p1=один id (r1a-12), margin переопределён?
4. Детерминизм: r=2 на всех 16, порог 0.95, судьба Q1/Q3 прописана?
5. Пересчитываемость: сырые runs/ + id/p1-стримы + режим --verify?
6. Гейты измеримы (файлы, строки stderr, pgrep, порт, LAUNCHED/BUSY)?
7. Не появилось ли новых фантомов: поля ответов сервера (точное имя prompt-count
   не придумано, а вынесено в спайк), эндпоинт /health, cache_prompt=false
   поддерживается ли этим форком (если нет — спайк зафиксирует, харнесс не
   шлёт неподдерживаемое поле)?
8. Прочее: всё, что ломает одноисполнительность или делает Q невыполненным.

## 8. Уточнения r2.1 (пост-коммитная сверка имён полей с исходниками)

1. Реальные имена полей запроса `/completion` (grep `tools/server/server-task.cpp:97-130`):
   `temperature, top_k, top_p, min_p, top_n_sigma, xtc_probability, xtc_threshold, typical_p,
   repeat_last_n, repeat_penalty, presence_penalty, frequency_penalty, dry_multiplier, dry_base,
   dry_allowed_length, dry_penalty_last_n, dry_sequence_breakers, mirostat, mirostat_tau, mirostat_eta,
   adaptive_target, adaptive_decay, stop, max_tokens, n_predict, n_keep, n_discard, ignore_eos, stream,
   logit_bias, n_probs, min_keep, seed`. Плюс `logprobs`/`top_logprobs` (`server-common.cpp:1381-1388` → `n_probs`)
   и `cache_prompt` (`server-schema.cpp:31`).
2. Фантомов в запросе быть не должно: полей `anti_reverse`, `penalty_last_n`, `dry_penalty`, `dynatop_p`
   в этом форке нет (grep `tools/server/*.cpp` = 0; совпадения — только в vendored `tools/mtmd/.../llama-sweep`).
   В §2b засылать только имена из п.1.
3. `task_params::to_json(only_metrics)` используется для метрик, поэтому **эха применённых sampling-параметров
   в ответе `/completion` нет** → проверка штрафов делается поведенчески в спайке: тот же `w0` с
   `repeat_penalty: 1.1`; id-поток не изменился → штрафы в билде не применяются (записать в NOTES как
   ограничение; заодно подтверждает, что основной penalty-free поток не искажён).
