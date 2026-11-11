# subagent (проектное расширение pig)

Оркестрация headless-субагентов без shell-хоря:
- `subagent_spawn(prompt_file|prompt, model='obleth/ngenix-llm:low', tools='read,bash,write', cwd, marker)` — detached-запуск, журнал `.pig/jobs/<id>.json`;
- `subagent_status(id?)` — статусы+хвост вывода (done_marker / done_rc0 / done_rcX / dead — трупы детектируются, урок «тихих RC=0»);
- `subagent_wait(id, timeout_s)` — блокировка до конца.

Загрузка: `pig -a` (траст проекта) или `-e .pig/extensions/subagent`. Промпты — файлами
(/tmp или docs/dynamic/orch/), не инлайном (урок отравленных cmdline).

## Почему самописное, а не готовое (проверено 11.11, не выдумано)
Реестр pig за шлюзом — 401. Через donsetch/npm найдены и провалидированы здесь:
- `pi-subagents@0.88.0` (nicobailon): `pig install --validate-only` → «Node extension
  bundle/index.js has no Pi-compatible default export» (собран под более новый pi-API,
  чем Pig-совместимость).
- `@tintinweb/pi-subagents@0.23.2`: register handshake EOF в packed-node (тот же сбой,
  что pi-blackhole/pi-mcp-adapter на этой машине: stripTypesScriptTypes worker).
Вывод: наpig-сборке этой машины TS-расширения субагентов не стартуют; python-SDK вариант
работает (сквозная проба SMOKE_DONE). Если Pig-рантайм починят/обновятся бандлы —
замена на готовое = `pig install npm:pi-subagents`, это расширение retire.
