# Внешнее ревью: trackio-миграция v1.9 (код + артефакт)

CWD: /home/survivor/work/prism. Читай только, НЕ изменяй файлы. Задача — ревью-ФАЙЛ:
`docs/dynamic/trackio_mig_v19_REVIEW.md` (едиственный файл, который можно создать).

## Объекты
- Код: `experiments/trackio_migration.py` (v1.9: N1 is_json_log fix, N2 EMPTY=101,
  N3 реестр 129/145, N4 --append, N5 merge_hf_db из бэкапа, rc-фикс, skip-идемпотентность)
- ТЗ: `docs/TZ_trackio_migration.md` §9 (аддендум v1.9 + r1-правки)
- Артефакт: `results/trackio_migration_report.json` (rc=0, 129 runs, 7451 metrics, 18 warnings)
- DB: `.trackio/prism.db` (139 имён / 149 configs), HF-бэкап `/tmp/recovered/hf_trackio_prism_backup_*.db`

## Контекст
Bulk-миграция всех прогонов prism в trackio. 15.11-попытка не завершилась (quiescence).
16.11: v1.9-аддендум → TZ-ревью (REVISE, 1 MAJOR: протухший числовой контракт + pre-flight;
3 MINOR) → правки → тест на копиях DB (3 прогона: полный, идемпотентный) → реальный прогон
RUN_RC=0. Параллельная сессия была активна (AT2b, AT3 r4, AT-chain): инвентарь двигался —
at2b-лог перенесён в results/AT2b/, AT3_run.log переписан r3→r4 — реестр донастроен pre-flight'ом.

## Чек-лист (каждый пункт: OK / ПРОБЛЕМА + число/строка)
1. **N1 (is_json_log):** фикс `lstrip("[").lstrip(",").rstrip("]").rstrip(",")` — корректен?
   Ложные срабатывания на текст-логах реестра? (Проверь: `python3` + функция на всех 129
   регистровых логах — JSON только T4-логи + dубли + xet.)
2. **N2 (EMPTY=101):** список в коде == факт (пересчитай dry-run'ом)? e93 не пуст?
3. **Реестр 129/145:** арифметика сходится (find vs known)? Пути всех 13 новых run'ов существуют?
   at2b (results/AT2b/AT2b_run.log), at3_r4 (results/AT3_run.log)?
4. **--append:** идемпотентность (skip по name+log, вкл. абсолютный путь); FATAL без флага;
   счётчик-проверки не применяются при all-skip?
5. **merge_hf_db:** ATTACH из бэкапа (не uri!), схема-сверка, skip существующих run_id,
   fail-loud; бэкап реально создан? HF DB на диске не тронут (mtime)?
6. **Артефакт (report):** rc=0, 0 errors, md5_mismatch= [], 18 warnings = только marker-dev?
   n_runs=129, registry-статистика 145/129/145?
7. **DB репо:** 139 имён / 149 configs; HF-имена (at05_selftest_tmp, at0_9_t21_r1,
   at3_margin_r1/r2/r3/r4) присутствуют; e17_lora_distill имеет метрики (bridge/kl —
   spot-check через sqlite: SELECT ... WHERE run_name='e17_lora_distill' LIMIT 3)?
   Дупликатные имена — только ожидаемые (at07_loop_r1×5, at0_4_rebuild_r6/r6b×2,
   at0_9_t21_r1×2, at3_margin_r1×3, at3_margin_r2×2)?
8. **ТЗ vs код vs артефакт:** согласованы (N1-N5, pre-flight, формула ожидаемого итога)?
9. **AGENTS.md чек-лист:** одинаковая статистика / guard'ы / каскад / json-sanitize /
   долгие прогоны / калибровки — нарушения в v1.9-изменениях?

## Формат ревью-файла
Шапка (объекты + sha проверенных: `sha256sum experiments/trackio_migration.py
results/trackio_migration_report.json .trackio/prism.db`), разделы по чек-листу,
MAJOR/MINOR/NIT, ИТОГ: `VERDICT: PASS | REVISE | FAIL` (PASS = можно коммитить).
ПОСЛЕДНЯЯ СТРОКА ФАЙЛА: `REVIEW_TRACKIO_MIG_DONE` (и последняя строка твоего ответа).

## Правила
Числа — только из артефактов/логов/DB. Не запускай эксперименты. Не трогай DB и код.
Сомнение → STOP-запись в ревью.
