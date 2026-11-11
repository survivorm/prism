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

## 11.11 вечер: ОБНОВЛЕНО — готовые субагенты поставлены
pig обновлён npm-ом 0.2.0+pi0.87.1 → 0.7.0+pi1.5.0; на новом рантайме 
`pig install npm:pi-subagents` (глобально) валидируется и работает. Модели зафиксированы:
subagents.defaultModel=obleth/ngenix-llm + agentOverrides (reviewer/oracle/worker/scout)
в ~/.pi/agent/settings.json И ~/.pig/agent/settings.json (расширение читает pi-путь).
Кастомные агенты: .pi/agents/{prism-reviewer,prism-runner}.md (зеркало .pig/agents —
расширение ищет pi-пути). Проба: prism-reviewer проверил MARKERS M5/M6 — ок.
Это расширение = fallback (детектор трупов/RC-маркеров); основной путь — subagent-инструмент.
