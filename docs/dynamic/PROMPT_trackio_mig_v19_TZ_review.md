# Внешнее ревью TZ: аддендум v1.9 «завершение миграции в trackio»

CWD: /home/survivor/work/prism. Ты — внешний ревьюер. Читай только, НЕ изменяй файлы.

## Объект
`docs/TZ_trackio_migration.md` — секция **«## 9. Аддендум v1.9 (16.11)»** (в конце файла) +
контекст §3 (реестр), §4 (экстракторы/счётчики), §5 (протокол), §7 (само-ревью), §8 (порядок).

## Суть задачи (что пользователь просит)
Bulk-миграция всех прогонов prism в trackio (`.trackio/prism.db`) ни разу не завершилась
(15.11 — срыв на quiescence). Нужно: загнать ВСЕ старые прогоны (115 из снапшота 15.11 +
новые после него) в канонический DB проекта, не потеряв live-записи (10 run'ов в двух DB).

## Факты, которые аддендум опирается (проверь, что они правдивы — команды в конце)
1. Bulk-прогон 15.11 не дошёл до trackio (report: quiescence error, n_runs: None).
2. `is_json_log` (в `experiments/trackio_migration.py`) не распознаёт формат Kaggle-лога
   (1-я строка `[{…}`) — все T4-логи трактуются как текст → 0 метрик → счётчик-чек упал бы.
   Фикс предложен: lstrip "[". Dry-run: 116 старых run'ов с фиксом воспроизводят ВСЕ 25
   счётчиков v1.8 (TZ_EXPECTED_HITS) ровно.
3. Текст-логи не экстрактятся → `empty_metrics=True` у всех них. Факт dry-run: 99 пустых
   из 127 (v1.8-список был 12 — никогда не проверялся полным прогоном).
4. На диске 143 .log; реестр v1.8 покрывает 116 run'ов (114 уник. файлов); 12 .log не покрыты.
5. Live-записи: репо-DB 5 run'ов, HF-cache-DB 5 run'ов (trackio 0.45.1, py 3.12.9
   /opt/miniconda3/bin/python3.12, user-site).

## Чек-лист
1. **N1 (is_json_log фикс):** корректен ли `lstrip("[").lstrip(",").rstrip("]").rstrip(",")`
   для всех реальных логов? Есть ли ложные срабатывания: текст-лог, чья первая непустая строка
   начинается с `[` и после strip'а — с `{` (например, строка вида `[{...}]` в обычном тексте)?
   Проверь по нескольким текст-логам (e45*, at0_*, AT3) и Kaggle-логам.
2. **N2 (EMPTY-список 99):** методология dry-run честна? Края: e93_sycl_bench (bench-метрики —
   НЕ пустой), e27b_early (табличный — пустой). Правильно ли, что проверка остаётся строгим
   равенством множеств (а не «содержит»)?
3. **N3 (реестр v1.9):** арифметика 127/143 = 125+5+13 сходится? Оправданы ли имена/пути
   (at0_5_g8fail, at0_6_v2_2, at0_6_v2r1_partial, at3_r1/r3 — два разных run'а из двух файлов)?
   Оправдан ли exclusion `logs/AT8/AT8_1_run.log` (stdout-дубль T4-прогона)? Допуск дупликаций
   имён run_name — приемлем?
4. **N4 (--append):** семантика (DB есть + append → не удалять; DB нет + append → как fresh;
   DB есть без флага → FATAL) — корректна и не ломает idempotency-гарантии §5.2?
5. **N5 (консолидация HF-cache DB):** риски копирования строк (схемы, run_id-коллизии,
   sqlite-блокировки при live-писателе); достаточно ли backup? Альтернатива — НЕ мигрировать,
   а зафиксировать в отчёте — какая лучше?
6. **Полнота цели «все старые прогоны»:** что НЕ попадёт в trackio и почему допустимо
   (ранние at07-прогоны 15.11 ×3 без выживших логов; тестовый /tmp/trackio_full 14.11;
   smoke-артефакты)? Есть ли другие «потерянные» прогоны, которые можно/нужно добавить?
7. **Протокол:** интерпретатор /opt/miniconda3/bin/python3.12 (не .venv — там нет trackio)
   — согласуется с AGENTS.md/run_one.sh?

## Формат ответа
По чек-листу; классификация MAJOR/MINOR/INFO; вердикт PASS/REVISE/FAIL.
Последняя строка ответа ровно: `TZ_REVIEW_DONE verdict=<PASS|REVISE|FAIL> major=<N> minor=<N>`

## Команды для проверки (read-only)
- `cd /home/survivor/work/prism`
- `grep -n "def is_json_log" -A 8 experiments/trackio_migration.py`
- `head -c 100 logs/prism-exp17-lora-distill.log; echo; head -c 100 logs/e45b_leaf_table.log`
- `head -3 logs/AT0/AT0_6_run.log logs/AT3_run.log logs/AT8/AT8_1_run.log`
- `python3 -c "import sqlite3; d=sqlite3.connect('.trackio/prism.db'); print(d.execute('SELECT DISTINCT run_name FROM configs').fetchall())"`
- `python3 -c "import sqlite3; d=sqlite3.connect('/home/survivor/.cache/huggingface/trackio/prism.db'); print(d.execute('SELECT DISTINCT run_name FROM configs').fetchall())"`
- `find . -name "*.log" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./box/*" | wc -l`
- `sed -n '/## 9. Аддендум/,$p' docs/TZ_trackio_migration.md`
- `/opt/miniconda3/bin/python3.12 /tmp/trackio_dryrun.py` (dry-run счётчиков; /tmp/trackio_dryrun2.py — empty/run_failed)
