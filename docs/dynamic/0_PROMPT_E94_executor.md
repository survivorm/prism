# ИСПОЛНИТЕЛЬ E94 — sycl_stream: одна llama-server SYCL-сессия, id+logprobs по 16 докам

Рабочая директория: `/home/survivor/work/prism`. Ты — единственный исполнитель эксперимента E94.
Любое сомнение решается консервативно: в сторону STOP с записью в NOTES, а не тихой подгонкой.

## 0. Что прочитать ПЕРВЫМ ДЕЛОМ (по порядку, целиком)
1. `docs/dynamic/1_ТЗ_E94_sycl_stream.md` — это ТЗ r2, единственный источник требований (разделы §0–§9; §9 —
   журнал правок, не requirement).
2. `docs/dynamic/1_REVIEW_ТЗ_E94_r1a.md` и `1_REVIEW_ТЗ_E94_r1b.md` — ревью, породившие r2.
3. `experiments/exp81b_replay.py` — дословные формулы `rep_flags`, `spear`, `load()` и константы.
4. `experiments/exp81d_gen.py` — `load_docs()`.
5. `results/e93/prompts.json` (структура `{meta, docs}`) и `results/e93/sycl/NOTES.md` (канал, флаги, тайминги).
6. `docs/STATIC/llama.cpp_docs/SYCL.md`, `~/work/llama.cpp/PLAN.md` (i915/IGC-предосторожности, SDPA-гейт).
7. `run_one.sh` (как работает лончер) и `AGENTS.md` (чек-лист само-аудита).

## 1. Разрешено / запрещено
- Создать: `experiments/exp94_sycl_stream.py`, всё под `results/e94/`, `docs/dynamic/0_E94_EXEC_NOTES.md`.
- Запуск: только `RUN_ONE_MIN_MEM=3 ./run_one.sh e94 "<cmd>"` (одна сессия = один лонч; внутри скрипта
  свои подпроцессы сервера запрещены только вне логики §4 ниже — сервер поднимает скрипт сам, это его дети).
- Запрещено: править ТЗ/чужие скрипты/`~/work/llama.cpp`; `git add/commit`; `pkill` чужих процессов
  (свой сервер убивать только по своему PID, `kill <pid>`, затем пауза ≥5 с); любые флаги llama,
  не найденные grep'ом в local tree (`~/work/llama.cpp/common/arg.cpp` или `tools/server/server-config.cpp`) —
  отсутствие флага ДОКАЗЫВАТЬ строкой grep, а не памятью; правки llama.cpp запрещены (TZ §1).
- Числа в отчёт — только из артефактов/логов. `RC=0` без артефакта = провал, не успех.

## 2. Реализация `experiments/exp94_sycl_stream.py`
Две фазы: `spike` (шпажорка, ~4 запроса) и `full`. Общие требования:
- **Один** `llama-server` на инстанс-сессию: `-m /data/Qwen/Qwen3.5-0.8B-Q4_K_M.gguf -sm none -mg 0 -ngl 99
  -fa on -c 2048 -np 1 --host 127.0.0.1 --port <PORT: free> --log-disable` (проверить точные имена флагов
  сервера grep'ом; если `-sm/-mg` не приняты сервером — зафиксировать и использовать принятые эквиваленты).
  stdout/stderr сервера → файл `results/e94/server.log` (НИКОГДА в pipe).
- Готовность: GET `/health` (существует — проверить grep server.cpp) с таймаутом; после готовности — GET
  `/metrics`: извлечь `n_prompt/n_decode` и **устройства** (`embd/prompt/decode`), записать в артефакт.
  Device-гейт: `decode_device` должен содержать SYCL/Iris; иначе `RUN_RC=10` с текстом гейта.
- **Prompt pre-gate** на каждом доке: POST `/tokenize {prompt: <текст из prompts.json>}` → id; сравнить с
  HF-id; равно → `prompt_match=1`. В `/completion` слать **`prompt` = raw-текст** (не messages, не chat),
  `n_predict: 384`, `temperature: 0`, `stream: false`, `cache_prompt: false`, `logprobs: true`,
  `top_logprobs: 2` и **явно обнулённые штрафы** ИСКЛЮЧИТЕЛЬНО проверенными именами
  (`tools/server/server-task.cpp:97-130`): `repeat_penalty: 1.0`, `presence_penalty: 0.0`,
  `frequency_penalty: 0.0`, `repeat_last_n: 0`, `dry_multiplier: 1.0`, `dry_base: 1.0`, `xtc_probability: 0.0`,
  `mirostat: 0`, `seed: <фикс.>`; `top_k`/`top_p`/`min_p` НЕ трогать (цепочка как в e93 — см. ТЗ §2b).
  Запрещено слать поля, которых нет в этом списке: `anti_reverse`, `penalty_last_n`, `dry_penalty`,
  `dynatop_p` — их в форке нет (grep = 0, совпадения только в vendored llama-sweep).
  Поведенческий контроль (один запрос в spike, см. ТЗ §8.3): тот же `w0` с `repeat_penalty: 1.1` —
  сравнить id-поток; совпал → штрафы не применяются, записать в NOTES.
- Ответ: сохранить **сырым** JSON в `results/e94/raw/<doc>__r<r>.json`; извлечь `tokens[]` (id, logprob,
  `top_logprobs`/`top_probs` — имя ключа подтвердить по живому ответу, `server-task.cpp:296`), text.
  Проверить `len(tokens)` ≥ минимум; записать `content_txt = completion` и проверить
  `decode(tokens.id) == content_txt` (иначе `reencode_ok=0`).
- Идемпотентность: существующий raw-файл → не перезапрашивать; метаданные пересобираются из raw.
- Таймаут запроса 900 с; ошибка HTTP/timeout → `errors.append` и продолжение по остальным докам; если ≥4
  доков упало — `RUN_RC=11`.
- Сервер: `try/finally` с graceful kill по PID, `wait`, пауза ≥5 с перед любым новым стартом; при падении
  сервера посередине — перезапуск не более 2 раз, иначе `RUN_RC=12`.
- Память: пик RSS скрипта и сервера (`/proc/<pid>/status` или resource.getrusage) — записать.

### spike (`--phase spike`)
1. `w0`: id-поток r=1; затем `w0` r=2 → `repro_rate` = доля совпадающих id по позициям 0..min(len)−1.
2. Из raw w0: `len(tokens)`, `len(content_txt)`, распределение `len(top_logprobs)` по позициям
   (доля позиций с ровно 2), `p1 := exp(p_logprob)` на prompt-позициях.
3. **Три независимых прогона p1 по 8prompt-позициям w0** (3 запроса с одним prompt, `n_predict:1`):
   `p1_illuminator` — медиана |Δ| между прогонами и против CPU `p1[0]` доков REPLAY.json (для w0).
4. Валидация парсера на заведомо «уродливых» токенах: синтетический unicode-тест decode-consistency
   (CJK из w8, U+200B из w12) — decode(tokens) vs content_txt побайтово.
5. `results/e94/FORMAT.md`: формат, число позиций `1+len(prompt)`, ответы на все пункты ТЗ §4, все отклонения.
6. If `repro_rate < 0.95` или гейты/format не сходятся → `RUN_RC=2` (TZ: переписать метрику через perplexity
   teacher-forcing — НЕ делать, только задокументировать и остановиться).

### full (`--phase full`)
- 16 доков × r=2 (идемпотентно), затем метрики ТОЛЬКО по `prompt_match=1` докам:
  - Q1: prefix_len по HF-id (prompt + gen id), нормировка `/448`; per-doc таблица + медиана + `frac_prefix_gt_0_9`.
  - Q2: rep8 (формула exp81b, `off = len(prompt)=64`), `rep8_delta` llama−CPU; главная — медиана |Δ|;
    Spearman **с тай-коррекцией** (пересчитывать самостоятельно, avg-ranks), плюс пермутационный null
    (2000 перестановок, seed 8181, один и тот же набор доков) → эмпирическая p и доля null-максимумов ≥
    наблюдаемого rho; p1: aligned-позиции по `reencode_ok`+roundtrip (gen-позиции 0..96 против REPLAY `gen_p1`),
    `p1_agree` = mean(|Δ|<0.05) с `n_aligned`, медиана |Δ|, Spearman+null.
  - Q3 (только доки с `n_aligned≥32` и std rep8/CPU > 0.01 по обеим сторонам): margin := `exp(top1)−exp(top2)`
    из `top_logprobs` llama, против аналогичного CPU-margin там, где он есть (если CPU-margin нет в REPLAY.json —
    зафиксировать «N/A: source absent», НЕ выдумывать; rho против rep8_delta тогда не считается);
    rep8 CPU-vs-llama Spearman+null по всем 16 докам с `prompt_match=1`.
  - Гейт интерпретируемости: если >5% id-позиций расходятся между r=1/r=2 по медиане доков → Q1/Q3 помечать
    `non-interpretable (determinism)` и не обсуждать как эффект движка.
- Артефакт `results/e94/SYCL_STREAM.json` по ТЗ §6 + `results/e94/NOTES.md`: вердикты Q1/Q2/Q3
  (число + «интерпретируемо/нет» + причина), лимиты (в один абзац каждый), все отклонения от ТЗ,
  смета времени, список grep-доказательств использованных флагов.

## 3. Порядок работы (не переставлять)
1. Чтение (§0) → список всех флагов/полей, которые будут использованы, с grep-доказательствами в заметки.
2. Написание скрипта.
3. **Само-аудит кода ДО запуска** — чек-лист AGENTS.md: одинаковая статистика на сторонах сравнения; ветки
   guard'ов; потребители изменённых типов; json.dump-sanitize (numpy→float, NaN); никаких мёртвых параметров;
   главная ловушка — метрики считаются от НЕ-raw источника (запрещено: пересчёт из сводных полей).
4. Смоук: `RUN_ONE_MIN_MEM=3 ./run_one.sh e94 "python3 -u experiments/exp94_sycl_stream.py --phase spike --docs w0"`
   в `logs/daemons/e94_smoke.log`; прочитать лог ЦЛИКОМ; проверить `results/e94/FORMAT.md` и raw-файлы.
5. Полный spike (все 4 запроса шпажорки) тем же способом; прочитать лог целиком.
6. Если spike-гейты пройдены — full: `RUN_ONE_MIN_MEM=3 ./run_one.sh e94 "python3 -u experiments/exp94_sycl_stream.py --phase full" > logs/daemons/e94.log 2>&1`
   и ожидание завершения СВОИМ блокирующим циклом (poll файла артефакта + `pgrep -x python3`, sleep 30),
   не более 2.5 ч; по завершении прочитать лог целиком.
7. Само-ревью артефакта: пересчитать 3 числа вручную (prefix_len одного дока, rep8 одного дока, один margin)
   и сверить с артефактом; проверить отсутствие NaN в JSON и что все цифры отчёта есть в артефакте.
8. Не коммитить. LAST LINE stdout финального отчёта: `E94_EXEC_DONE` (успех) или
   `E94_FAIL <причина>`; в `0_E94_EXEC_NOTES.md` — то же последнее строкой.

## 4. Что писать в отчёт
RC каждой фазы, путь артефакта, вердикты Q1/Q2/Q3 (числа из артефакта, с n и гейтами), что бы ты сам
проверил первым при внешнем ревью, отклонения от ТЗ. Без гипотез о механизме — только измерения.
