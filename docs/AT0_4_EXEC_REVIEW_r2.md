# AT0_4_EXEC_REVIEW_r2 — closure-ревью фикса (к r1)

Ревьюер: внешний (prism-reviewer), 2026-11-14.
Объекты: `experiments/AT0_4_mix.py` (a4b7a138…), `experiments/AT0_4_rebuild_p4.py`
(bf80334d…), `results/AT0/store/f1cc7265dd40c874.json` (новый store),
`results/AT0/AT0_4_MIX.json` (claim), `AT0_4_manifest.json`, `AT0_4_rebuild.log`.
База: `docs/AT0_4_EXEC_REVIEW.md` (r1, PASS_WITH_MINORS; условия П1/П2 — §4).
Скоуп: ТОЛЬКО closure-проверка — выполнены ли П1/П2 и не сломало ли правки остальное;
числа/гейты/P1-vacuous/P2-trunk_carries из r1 не пересматриваются.

## 1. П1 (diff-цепочка): ЗАКРЫТ

**Value-neutrality (проверено прочтением конца main() в a4b7a138):** порядок записей в
текущем коде — store (`store_path`) → claim (`OUT_JSON`) → manifest (`MAN_JSON`) →
финальные две logline («AT0_4 gates: ALL PASS», «AT0_4 DONE») → RUN_LOG последним
(комментарий в коде прямо называет баг v3.0: в баг-версии run.log писался до финальных
строк и «ALL PASS»/«DONE» терялись). SHA-вычисления (`store_sha`, `claim`-blob,
`manifest.claim_sha256`) выполняются на стадии формирования blob'ов — ДО любой записи,
и не зависят от того, где пишется лог. Значит, перенос записи RUN_LOG (f59035c0→92effe83)
не может изменить байты store/claim/manifest: единственное наблюдаемое отличие версий =
две строки в run.log, которых в баг-версии физически не могло быть. Это согласуется с
доказательствами r1 (run.log на диске mtime 05:32:03 — ПОЗЖЕ артефактов 05:19:54, и
содержит «ALL PASS»/«DONE» — т.е. восстановлен продюсером из stdout-редакта nohup).
**Заключение: изменение f59035c0→92effe83 value-neutral; перепуск прогона не требуется.**

**Цепочка (3 звена, shas сверены с диском):**
- `f59035c024fa3349…` — скрипт, считавший 112 forward (`store.meta.script_sha256`,
  прогон 05:19:54); оригинал потерян (untracked, pyc откомпилирован от новой версии —
  hash-match невозможен, зафиксировано в r1).
- `92effe839acec490…` — единственное изменение = перенос записи RUN_LOG (см. выше).
- `a4b7a1385d74fd…` — fix-раунд (текущий диск, sha подтверждён: sha256 файла = a4b7a138…).
  Изменения r1-условий: build_p4/build_summary на module-уровень, P4 per_doc/
  per_doc_outliers/outlier_note, summary-пометки, S2_tied фактические значения, med().
- `bf80334d2e5128…` — rebuild-скрипт (sha подтверждён диском; см. §4).

**Где закреплена:** store.meta (звено 1), r1-документ §2 П1 (оба shas + гипотеза),
сообщение продюсера, этот ревью; `provenance.aggregation_rebuild.forward_window`
зафиксирует окно форвардов (04:20:00..05:11:00, 112 forward, kill→resume; 05:14
resume-проход — replay без форвардов; артефакты v1 05:19:54). Реbuild-лог закрепляет
только текущие shas (a4b7a138/bf80334d), звено 1 — только в r1-документе/сообщениях —
см. новое замечание 5.1. Git: все AT0_4-файлы untracked, коммит ещё не сделан —
финальный носитель (коммит-месседж по r1 П1) проверяется продюсером при коммите;
процессный гейт r1 исполнён до минимума возможного без коммита.

## 2. П2 (P4 per_doc + summary): ЗАКРЫТ — числами

**Блок `store.prereg.P4_random_order_insensitive` (store f1cc7265…):**
- Все 6 live-инстансов (R1/R2 × S11/12/13) имеют `per_doc` по 8 доков, `threshold=0.1`,
  `median_abs_diff`, `verdict`. Замечаний not_testable/divergent нет (как в r1).
- `per_doc_outliers = {"own_w0": "2/6", "own_w4": "6/6"}` — ТОЧНО как в r1-условии;
  `outlier_note = "null по медиане, но per-doc исключения ≥0.1: own_w0 2/6, own_w4 6/6"`;
  `n_token_statistics_flags = 0`.
- **Медианы и вердикты неизменны:** пересчёт `np.median(per_doc)` для всех 6 инстансов =
  бит-в-бит сохранённым значениям: 0.001826 / 0.000960 / 0.001814 / 0.000853 / 0.000743 /
  0.001080 (диапазон 0.0007–0.0018 = диапазон r1 «0.0007–0.0018 → null×6»); вердикты
  null×6; `pretest_results.P4` = null×6, flags=0. Независимый пересчёт per_doc из
  `store.per_doc` (|r_real − r_mix| по real/mix-клеткам) = бит-в-бит значениям per_doc
  блока P4 во всех 6 инстансах (и через rebuild-ассерт в AT0_4_rebuild.log).
- **Сверка per-doc с r1 (§2 П2):** own_w4 во ВСЕХ 6: 0.1298/0.1188/0.1277/0.1245/0.1314/
  0.1258 (r1: 0.1188–0.1314, max 0.1314 ✓); own_w0: 0.1101 и 0.1002 ≥ 0.1, остальные
  0.0989/0.0911/0.0909/0.0860 (r1: «0.1101, 0.1002; остальные 0.086–0.099» ✓).

**Summary (`store.verdict.summary` = claim, сверено):**
`AT0_4 (3×2): R2×real — S11 vacuous, S12 vacuous, S13 vacuous; R1 vs R2 (real):
trunk_carries; R0 real-vs-mix: r confirmed (med 0.016), f_DC confirmed (med 0.000,
mirror P3 по G3 — не независимый сигнал); random order-invariant: 6/6 null (медианный
тест; per-doc исключения: own_w0 (2/6), own_w4 (6/6)), 0 token-statistics флаг(ов),
0 not_testable`
— содержит «медианный тест» ✓, per-doc исключения ✓, P5 «mirror P3 по G3» ✓
(пометка r1-П3 при П2-правке исполнена). Префикс до f_DC идентичен прежнему summary
(rebuild-ассерт, подтверждён строкой выше против r1-цитаты).

**Незменность остального:** rebuild-ассерт (лог): в store изменились ТОЛЬКО
`prereg.P4_…`, `provenance`, `verdict` (в verdict — только summary; pretest_results
неизменён). Проверено: meta/gates/per_doc/prereg/verdict claim ≡ store (§4); σ-таблица,
gates S1–S5/G_nd — не тронуты rebuild-кодом (тот читает только per_doc/gates.G_nd и
пишет ровно три поля).

## 3. Код фикса: чек-лист 3 — чисто

1. **py_compile:** `experiments/AT0_4_mix.py` и `experiments/AT0_4_rebuild_p4.py` —
   оба компилируются (`python -m py_compile` OK).
2. **build_p4/build_summary — единственные источники P4/summary:** по grep весь
   `p4` в main — это `p4 = build_p4(cells, docs_run, seeds_run, inst_not_testable,
   divergent)` (строка ~668), `n_flag = p4["n_token_statistics_flags"]` и
   `build_summary(p1, p2_cell, p3, p5, p4)`; других писателей P4/summary нет.
   Rebuild-скрипт импортирует тот же модуль (`importlib` по файлу) — единый источник
   логики подтверждён конструктивно.
3. **`med` — один, на module-уровне:** ровно одна `def med` (строка 251, не вложенная;
   полный список `def` в файле перепроверен); вызывают P2 (641), P3 (658), P5 (678)
   и build_p4. `P4_THR = 0.1` (строка 93, вместе с P2_THR).
4. **S2_tied гейт = фактические значения:** в цикле прогонов `s2_results[f"R{c}_{inst}"]
   = tied_check(model)` (data_ptr lm_head/embed), FATAL RC=4 при провале;
   `gates["S2_tied"] = {"trained": bool(s2_trained), **s2_results, "ok": ...}` —
   литерала True больше нет (r1-П5 исполнен). В артефакте: trained=true, R1/R2 ×3 true,
   ok=true — значения, а не заглушки (прошедший прогон = пройденный check).
5. **p4-компрехеншн в verdict защищён:** `{k: v["verdict"] for k, v in p4.items() if
   isinstance(v, dict) and "verdict" in v}` — ключи-не-даты (`n_token_statistics_flags`,
   `per_doc_outliers`, `outlier_note`) отфильтрованы; аналогичная защита в
   `build_p4`/`build_summary` (`isinstance(v, dict) and "per_doc" in v`).

Дополнительно: rebuild-скрипт не правит store на месте (deep-copy `store_orig` +
финальный ассерт «изменены ТОЛЬКО P4/provenance/verdict» + ассерт
pretest_results-неизменности) — защита от случайного повреждения данных есть.

## 4. Артефакт: чек-лист 4 — чисто

1. **SHA-цепочка 4/4 (пересчитано по байтам диска):**
   - `sha256(store/f1cc7265dd40c874.json)` = `f1cc7265dd40c8744174fe3d6e0889199907c5652012970100639205c440ee4d`
     = префиксу имени файла ✓
   - `claim.store.sha256` = тот же sha, `claim.store.path` = `store/f1cc7265dd40c874.json` ✓
   - `manifest.store_sha256` = тот же sha ✓
   - `manifest.claim_sha256` = `sha256(results/AT0/AT0_4_MIX.json)` ✓
2. **claim ≡ store:** meta, gates, per_doc, prereg, verdict — все 5 секций идентичны
   (claim = подмножество store без profiles/provenance, как и задумано).
3. **Profiles 112×96:** 14 групп (R0_trained×{real,mix} + R1/R2×S11/12/13×{real,mix})
   × 8 доков = 112 профилей, каждый — ровно 96 значений (K=96) ✓.
4. **Rebuild-маркировка:** `manifest.aggregation_rebuild = true` ✓;
   `store.provenance.aggregation_rebuild = {date: 2026-11-14T06:36:18, forward_window,
   reason, script_sha256}` — forward_window зафиксирован (112 forward 04:20:00..05:11:00;
   05:14 resume-проход — replay без форвардов; артефакты v1 05:19:54) ✓,
   `script_sha256 = bf80334d2e51…` = sha rebuild-скрипта на диске ✓.
   `store.meta.script_sha256 = f59035c024fa3349…` (forward-прогон) — согласовано:
   meta описывает прогон, rebuild описан в provenance.
5. **Старый store удалён:** `results/AT0/store/71d3b5d868ed00c7*` — отсутствует
   (ls: No such file); в каталоге store новый f1cc7265dd40c874.json единственный
   AT0_4-артефакт.
6. **Замечание по пути claim:** claim лежит в `results/AT0/AT0_4_MIX.json` (не в
   store/) — так и было задумано (claim — корневой артефакт, store — content-addressed);
   r1-ссылка на `store/AT0_4_MIX.json` в промпте ревью — опечатка, на диске —
   `results/AT0/AT0_4_MIX.json`.

## 5. НОВЫЕ (2, оба — косметика)

- **[КОСМЕТИКА] results/AT0/AT0_4_rebuild.log + git (нет коммита) / звено f59035c0↔92effe83
  закреплено только вне rebuild-артефактов.** Rebuild-лог фиксирует лишь текущие shas
  (a4b7a138, bf80334d); звено, о котором идёт речь в П1 (diff между скриптом артефакта
  и диском на момент r1), закреплено в r1-документе §2 П1 (оба shas + гипотеза),
  сообщении продюсера и этом ревью — но не в rebuild-логе/provenance. Коммит AT0_4-
  файлов ещё не сделан (все untracked), так что финальный носитель по r1 П1 —
  коммит-месседж — пока не существует. Почему важно: будущий читатель увидит
  meta.script_sha256 = f59035c0, байты которого на диске отсутствуют, и без цепочки
  не сможет заключить value-neutrality заново. Чинить: при коммите — перечислить в
  месседже 3-звенную цепочку (f59035c0→92effe83: перенос записи run.log, value-neutral;
  92effe83→a4b7a138: fix-раунд; a4b7a138→bf80334d: rebuild) либо добавить строку
  про цепочку в rebuild-лог/provenance.
- **[КОСМЕТИКА] store/claim verdict.summary / лид «random order-invariant: 6/6 null».**
  Та же фраза содержит «per-doc исключения: own_w0 (2/6), own_w4 (6/6)», что
  противоречит первому слову «invariant» (r1-спор #1: «нечувствительна к порядку» —
  неверно для w4, там ≥0.1188 во всех 6 инстансах). Скобка-квалификатор спасает
  читаемость, но начало предложения всё ещё обобщает на все доки. Чинить: при
  следующей пересборке — «null по пре-рег-тесту (медиана): 6/6 (per-doc исключения:
  own_w0 (2/6), own_w4 (6/6))». Не блокирует: условие r1 П2 (per_doc + пометка w4/w0
  в note/summary) исполнено буквально, вердикт-клетки не меняются.

**Итог чек-листов:** 1 — П1 закрыт (§1); 2 — П2 закрыт числами (§2); 3 — чисто (§3);
4 — чисто (§4); новых БЛОКЕР/ПРАВКА — нет (2×КОСМЕТИКА).
Smoke после fix-раунда ревьюером НЕ прогонялся (бюджет; требует модели); claim
продюсера «value-строки bit-идентичны» подкреплён rebuild-ассертами (медианы P4
bit-идентичны, префикс summary неизменен) и моим независимым пересчётом медиан —
остаточный риск нулевой по данным (см. residual risks в отчёте).

VERDICT=PASS
AT0_4_EXEC_R2_DONE
