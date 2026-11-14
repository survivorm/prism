# PROMPT_AT0_4_exec_review_r1 — task-брюф r1 (зафиксирован post-hoc)

ЗАМЕТКА: файл PROMPT_AT0_4_exec_review.md, на который ссылалась task-строка, НЕ СОЗДАВАЛСЯ
(проверено по session-записи b5f18094: первый read ревьюера — ошибка «файл нет», далее
ревьюер сам нашёл все входы на диске). Ниже — task-брюф дословно (session b5f18094, entry 6)
+ фактический набор входов, который ревьюер прочитал сам. Файл фиксируется post-hoc как
аудит-запись, не как «исходный промпт».

## Task-брюф (дословно)
```
Внешнее ревью AT0_4 (код+артефакт). Читай и выполняй ПОЛНОСТЬЮ файл
/home/survivor/work/prism/PROMPT_AT0_4_exec_review.md — там цель, входы, чек-лист,
формат, бюджет (≤30 вызовов) и финальный маркер AT0_4_EXEC_DONE. Пиши ТОЛЬКО
/home/survivor/work/prism/docs/AT0_4_EXEC_REVIEW.md (overwrite после каждого раздела),
чат — только короткие подтверждения.
```

## Фактические входы (по session-записи b5f18094, tool-вызовы)
- AGENTS.md, docs/MARKERS.md — протоколы
- docs/AT0_4_prereg.md — TZ v3/v3.1
- experiments/AT0_4_mix.py (sha 92effe83… на момент ревью) — код
- results/AT0/AT0_4_MIX.json, AT0_4_manifest.json, AT0_4_run.log, store/71d3b5d868ed00c7.json — артефакт
- docs/AT0_4_TZ_REVIEW_r2.md — TZ-ревью r2
- git status/log, sha256sum, stat, pyc-аудит, независимые пересчёты P4/P3/P2/σ-таблицы
  (.venv/bin/python) — верификация числами

Итог: docs/AT0_4_EXEC_REVIEW.md, VERDICT=PASS_WITH_MINORS, маркер AT0_4_EXEC_DONE
(sha256 7c9751b8f439eb88d8b4aff9e085dbab2582de8bf15bc8d7c75aef6c7633049d).
