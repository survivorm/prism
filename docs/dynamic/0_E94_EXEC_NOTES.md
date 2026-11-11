# E94 EXEC NOTES — фаза Q1 (исполнитель, 12.11 ~01:20) + пересборка артефакта (отчёт ~01:40)

## Дефект артефакта и его фикс (без GPU, без сервера)
- Корень: в `run_q1()` per_doc копился в локальный список и **никогда не присваивался**
  `R["per_doc"]` → в `results/e94/Q1.json` был `"per_doc": []` при живых сводках.
  Фикс корневой: 1 строка `R["per_doc"] = per_doc` в `run_q1` + новый режим
  `--phase q1 --from-raw` (`run_q1_from_raw` в `exp94_q1.py`, флаг в `exp94_sycl_stream.py`):
  пересборка per_doc и сводок ИСКЛЮЧИТЕЛЬНО из `/tmp/e94raw/q1_*.json` (0 запросов к
  серверу, 0.97 c); поля прогона (port=8110, server_pid=169854, rc=0, halt_reason=null,
  t_total_s=851.68, fd-поля, body_template) сохранены из исходного артефакта.
- Сводка: единая `make_summary(ok)` для run и from-raw (одинаковая статистика, те же
  сиды). Добавлены пер-доковые агрегаты: `median_prefix_len=35.0`, `frac_prefix_full=0.3125`
  (5/16 доков с prefix_len==384), `n_identical_docs=16` (r1==r2 во всех 16), per-doc
  `prefix_len`/`repro_rate`/`n_tokens_r*` — в `per_doc[16]` (пересчёт любой цифры из
  raw+per_doc возможен; медиана |Δrep8| вручную из per_doc = 0.018229166666666685 —
  совпадает со сводкой).
- Кросс-чек (обязательный гейт): мой per_doc vs независимый пересчёт supervisor'а
  `/tmp/q1_independent_perdoc.json` — **16/16: prefix_len точное равенство, max|Δ| по
  rep8_llama/rep8_cpu/delta = 0.0** (< 1e-12). Подгонки не было — формулы совпали.
  Результат записан в артефакт: `independent_crosscheck = {pass: true, prefix_len_exact_matches: 16, max_abs_delta_rep8: 0.0, missing: []}`.
- Запуск фикса: `python3 -u experiments/exp94_sycl_stream.py --phase q1 --from-raw`
  → `E94 q1 from-raw: ok=16/16 crosscheck pass=True prefix_exact=16 max_d=0.0`, RUN_RC=0.
- Артефакт: `results/e94/Q1.json` (per_doc=16, сводки, independent_crosscheck, t_rebuild_s).

## Что сделано (Q1, ровно по ТЗ §2.1 + steering; Q2/Q3/кэш-гейт НЕ тронуты)
- Новый код: `experiments/exp94_q1.py` (244 строки, `run_q1()`), плюс 3 строки в
  `experiments/exp94_sycl_stream.py` (`--phase q1` → `sys.exit(run_q1())`; spike-ветка не тронута).
  Запуск ровно командой из ТЗ: `RUN_ONE_MIN_MEM=3 ./run_one.sh e94 "python3 -u experiments/exp94_sycl_stream.py --phase q1"`.
- Тело запроса — канон (B) из `results/e94/FORMAT.md` дословно: `n_probs:20, top_logprobs:20`
  (реальное поле — n_probs, top_logprobs сервером игнорируется, но остаётся в теле по канону),
  снятое прореживание `top_k:400, top_p:1.0, min_p:0.0, typical_p:1.0, xtc_probability:0.0`,
  нулевые штрафы (NEUTRAL из exp94_sycl_common), `temperature:0, stream:false, return_tokens:true,
  cache_prompt:false, seed:8181, n_predict:384`, prompt = 64 HF-ид из `results/e93/prompts.json`.
  16 доков w0..w15 × r=2, ОДИН сервер на всю фазу (port 8110, pid 169854).
- Идемпотентность: raw `/tmp/e94raw/q1_<doc>_r<r>.json` существует → не запрашивается (ветка
  `get_response` проверена смоуком).
- Метрики — только из raw, формулы дословно exp81b (через exp94_sycl_common):
  `prefix_len` vs `REPLAY.gen_ids` (/384); `rep8` по конкатенации 64+384 (off=64, 12×32,
  mean(segments['8'])) для llama- и CPU-стрима; главная — медиана |Δrep8| + bootstrap 10000
  (seed 8181, 2.5/97.5); вспомогательно Spearman avg-ranks + перм-null 2000 (seed 8181) + число
  различных rep8; `repro_rate` r1/r2, гейт медианы 0.95, метка `non-interpretable (determinism)`
  при провале. Только числа+гейты, без «механизм»-выводов.
- Артефакт `results/e94/Q1.json`: per-doc (prefix_len, rep8_llama, rep8_cpu, Δ, repro_rate,
  n_tokens, errors) + summary + `halt_reason` при STOP. `sys.exit(rc)` обязательный; RUN_RC:
  0 ok · 10 device-гейт (fd после health) · 11 ≥4 упавших дока · 13 prompt-count (tokens_evaluated≠64)
  · 1 другое.
- Само-аудит одной перечиткой (AGENTS.md): (1) статистика одинаковая на сторонах — одна функция
  rep8_segments для обеих, bootstrap/perm — одна процедура из common; (2) guard'ы — halt-каскад
  (server died before/between/during), per-doc ошибки не halt'ят, rc-маппинг по префиксу halt,
  fd-поток join'ится в finally; (3) каскад None — repro_rate=None при пропуске r, excluded из
  медианы; null_ci может быть None → san()→null; (4) json — весь R через san(), float-касты;
  (5) setsid через run_one; (6) метрики только из raw (resp_cache из get_response).
- Смоук (офлайн, без сервера, 1 прогон): 16/16 docs ok; rep8_cpu сверка против замороженных
  `REPLAY.docs.w*.segments['8']` — max|Δ| < 1e-12 (конвенция подтверждена); prefix/repro сверены
  с наивным циклом; тело — ровно whitelist-поля канона; `san`+`json.dumps` без ошибок.
  (Первые 2 падения смоука — баг моих синтетических ожиданий, не кода: см. steering; 1 правка
  конструкции синтетики, 1 повторный прогон.)

## Запуск (факты)
- pid обёртки (run_one) = **169841**, pid llama-server = **169854**, port 8110.
- лончер: `LAUNCHED e94 wrapper=169841 log=/tmp/e94.log`; свой сервер-процесс — ребёнок скрипта,
  kill только по своему PID (в finally `srv.kill()`).
- лог: `/tmp/e94.log`; серверный лог: `/tmp/e94q1_server.log`; raw: `/tmp/e94raw/q1_*.json`.
- Дежурство ~4 мин: health ok (load 14.1 s), fd-гейт пройден (`/dev/dri/renderD128` после health),
  w0..w3 r1/r2 зелёные (384 токена, `tokens_evaluated=64`, prompt-count гейт проходит на каждом),
  темп ≈30 с/док → 32 запроса ≈16 мин, потолок фазы 40 мин не угрожает. Чужих llama-server нет
  (`pgrep -x` до старта и по дежурству).

## Хайен-раунд по ревью r1 (12.11 ~02:40; GPU/сервер НЕ тронуты, только /tmp/e94raw)
Артефакт перегнан ЧИСТО из raw: `python3 -u experiments/exp94_sycl_stream.py --phase q1 --from-raw`
→ `ok=16/16 crosscheck pass=True prefix_exact=16 max_d=0.0 idempotent=True`, RUN_RC=0, ~1 c;
llama-процессов нет (ps до запуска). Четыре правки `experiments/exp94_q1.py`, одна на пункт:
1. `summary.n_identical_docs` → **`n_docs_repro_identical=16`** (это r1≡r2), рядом
   **`n_docs_prefix_full=5`** (prefix_len==384: w9, w10, w11, w12, w15).
2. **`spearman.p_definition`**: two-sided на |rho|, MC-доля |rho_perm| ≥ |rho_obs| без +1,
   пермутируется только llama-массив против фикс. cpu, avg-ranks, n_perm=2000, seed 8181;
   **`spearman.p_mc_se=0.00122`** = sqrt(p(1-p)/n_perm). MC-ошибка того же порядка, что и само p:
   корректная формулировка «p ≲ 0.005», не «p=0.003».
3. **`interpretability="not interpretable as an effect"`** + **`interpretability_reason`**
   (CI медианы |Δrep8| [0, 0.0325846] накрывает 0; 0.0182292 < ширины насыщенной полосы
   0.839–0.974 (0.135) — полоса цитата из ревью r1) + **`bifurcation_note`**: w4 Δrep8=−0.9505
   = единичная бифуркация (1/16) — расхождение движков f16/SYCL против fp32/CPU в
   низкоуверенных позициях (argmax-flip при temp=0), **не** «iGPU теряет повторы».
4. Идемпотентность (закрытие MAJOR-1'): `run_q1_from_raw` делает ДВЕ пересборки с диска
   (`rebuild_from_raw` ×2) и пишет **`idempotence={runs: 2, artifact_identical: true,
   sha256: 8e64191d69ec8c8cdce13e4b6189872d61505f39b39bf0578fff787fb788044e}`** — sha256 от
   канонического `json.dumps` (sort_keys, компактные separators) сводки+per_doc. Пересборки
   идентичны — источника не-детерминизма не найдено, числа не правились.
5. Чек значений (пункт 5): `median_abs_delta_rep8=0.018229166666666685`,
   `bootstrap_ci_2.5_97.5=[0.0, 0.03258463541666623]`, `spearman.rho=0.7039771994004667`,
   `spearman.p_perm=0.003` — после перегонки ВСЕ СОШЛИ с прежним артефактом (4/4 точное
   равенство; находки нет — код дерева воспроизводит артефакт, что согласуется с проверкой
   MAJOR-1 supervisor'ом).

## Что осталось (следующий прогон, НЕ в этот)
- Q2 (teacher-forced id-probe, подset по скоупу спайка) и Q3 (margin vs rep8 в каждом движке) —
  отложены, не реализованы.
- Кэш-гейт на 448-id префиксах (ТЗ §2.2: k ∈ {64,128,192,448}, порядок 64,64,128,128,192,192,448,448,
  `timings.cache_n ≥ k−2` со второго запроса, канон-тело (B), cache_prompt:true) — отложен, не тронут.
  Спайк показал сломанный /slots-гейт (task_prev: n_prompt_tokens=64, processed=0, cache=0) →
  гейт по `timings.cache_n` из ответа, а не /slots.
- Arm B (CPU-f16: тот же бинарник и тот же f16 GGUF, но `-ngl 0 -sm none`, отдельный сервер) —
  отложен, не тронут: нужен для разделения «бэкенд против точности» (f16/SYCL vs f16/CPU vs
  fp32/CPU); без него бифуркацию w4 нельзя атрибутировать между SYCL-бэкендом и f16-точностью.
- RUN_RC=0 без артефакта = прогон не выполнен: проверка `results/e94/Q1.json` обязательна после финиша
  (артефакт пересчитываемый: per_doc=16, кросс-чек с независимым пересчётом pass=true,
  идемпотентность 2/2).

E94_EXEC_DONE phase=q1_hygiene rc=0
