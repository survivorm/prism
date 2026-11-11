# subagent (проектное расширение pig)

Оркестрация headless-субагентов без shell-хоря:
- `subagent_spawn(prompt_file|prompt, model='obleth/ngenix-llm:low', tools='read,bash,write', cwd, marker)` — detached-запуск, журнал `.pig/jobs/<id>.json`;
- `subagent_status(id?)` — статусы+хвост вывода (done_marker / done_rc0 / done_rcX / dead — трупы детектируются, урок «тихих RC=0»);
- `subagent_wait(id, timeout_s)` — блокировка до конца.

Загрузка: `pig -a` (траст проекта) или `-e .pig/extensions/subagent`. Промпты — файлами
(/tmp или docs/dynamic/orch/), не инлайном (урок отравленных cmdline).
