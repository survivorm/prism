# AT0_4 — closure-ревью фикса (r2 к docs/AT0_4_EXEC_REVIEW.md)

Роль: внешний ревьюер (prism-reviewer). Задача — ТОЛЬКО closure-проверка: выполнены ли
ДО коммита условия (обязательные П1, П2) и не сломало ли правки остальное. Не пересматривай
весь цикл заново (числа/гейты/P1-vacuous/P2-trunk_carries уже верифицированы в r1).

## Вход
1. `docs/AT0_4_EXEC_REVIEW.md` — r1, VERDICT: PASS_WITH_MINORS; условия перед коммитом = §4:
   - **П1**: зафиксировать diff f59035c0 (артефакт) ↔ 92effe83 (диск); если diff шире
     переноса записи run.log — перепуск прогона.
   - **П2**: добавить per_doc в блок P4 + пометку про w4/w0 в note/summary (пересборка
     агрегации без пере-форварда).
2. `experiments/AT0_4_mix.py` (текущий, sha a4b7a138…) — код ПОСЛЕ фикса.
3. `experiments/AT0_4_rebuild_p4.py` (sha bf80334d…) — одноразовый rebuild артефакта.
4. Артефакт: `results/AT0/store/f1cc7265dd40c874.json` (новый; старый 71d3b5d868ed00c7
   удалён), `AT0_4_MIX.json` (claim), `AT0_4_manifest.json`, `AT0_4_rebuild.log`.

## Факты продюсера (проверяй по диску, не доверяй)
- Diff-цепочка (закрепляется в коммит-месседже): **f59035c0** (скрипт, считавший 112
  forward; store.meta.script_sha256) → **92effe83** (единственное изменение: блок записи
  RUN_LOG перенесён с позиции ДО финальных двух logline на позицию ПОСЛЕ них — фикс
  «run.log без ALL PASS/DONE»; в ОБЕХ версиях запись стоит ПОСЛЕ записей store/claim/
  manifest и после всех sha-вычислений ⇒ value-neutral для данных; оригинал f59035c0
  не сохранился (untracked), hash-match невозможен) → **a4b7a138** (fix-раунд: build_p4/
  build_summary вынесены в module-функции + P4 per_doc/per_doc_outliers/outlier_note +
  summary-пометки + S2_tied фактические значения + med() на module-уровень) → **bf80334d**
  (rebuild-скрипт: пересборка агрегации из store БЕЗ пере-форвардов, добавление
  provenance.aggregation_rebuild с forward-окном).
- Rebuild-ассерты (из AT0_4_rebuild.log): 6/6 медиан и вердиктов P4 bit-идентичны;
  per_doc_outliers = {own_w0: 2/6, own_w4: 6/6} — независимым пересчётом; store diff =
  только {prereg.P4, provenance, verdict.summary}; sha-цепочка 4/4.
- Smoke после fix-раунда: все value-строки bit-идентичны предшествующим смокам
  (P2 0.0041 same; P3 −0.0535 null; P4 0.1101/0.0909; S4 3.856e-16; S5 4.441e-16).

## Чек-лист (≤15 вызовов инструментов)
1. **П2 закрыт?** В `store.f1cc7265…`: у всех 6 инстансов P4 есть `per_doc` (8 доков),
   `per_doc_outliers == {"own_w0": "2/6", "own_w4": "6/6"}`, `outlier_note`; в summary —
   «медианный тест» + per-doc исключения + P5 «mirror P3 по G3»; медианы P4 и вердикты
   null×6 неизменны (сопоставь с блоком P4 в r1-описании / пересчитай 2-3 клетки по
   per_doc из store).
2. **П1 закрыт?** Diff-цепочка закреплена (rebuild-лог + provenance + это сообщение
   ревью); изменение f59035c0→92effe83 действительно value-neutral (прочитай конец
   main(): порядок записей store/claim/manifest/RUN_LOG в ТЕКУЩЕМ коде + аргумент, что в
   f59035c0 запись была только раньше).
3. **Код фикса не сломал:** py_compile (bash); build_p4/build_summary — единственные
   источники P4/summary (grep потребителей p4); `med` — один, на module-уровне; S2_tied
   гейт = фактические значения; p4-компрехеншн в verdict защищён (isinstance/«verdict» in v).
4. **Артефакт целостен:** sha-цепочка 4/4 по файлам; claim.per_doc ≡ store.per_doc;
   profiles 112×96; manifest.aggregation_rebuild = true; provenance.aggregation_rebuild
   содержит forward_window и sha rebuild-скрипта; старый store 71d3b5d868ed00c7 отсутствует
   в results/AT0/store/.
5. До 2 НОВЫХ замечания формата «- [БЛОКЕР|ПРАВКА|КОСМЕТИКА] где / что / почему / чинить».

## Формат вывода
Пиши ТОЛЬКО `docs/AT0_4_EXEC_REVIEW_r2.md` (overwrite после каждого раздела), чат — только
короткие подтверждения. Структура:
```
# AT0_4_EXEC_REVIEW_r2 — closure-ревью фикса (к r1)
Ревьюер: внешний (prism-reviewer), дата.
## 1. П1 (diff-цепочка): закрыт/не закрыт + что проверено
## 2. П2 (P4 per_doc + summary): закрыт/не закрыт + числами
## 3. Код фикса: чек-лист 3
## 4. Артефакт: чек-лист 4
## 5. НОВЫЕ (≤2)
VERDICT=PASS|REVISE|FAIL
AT0_4_EXEC_R2_DONE
```
Критерий PASS: П1 и П2 закрыты, чек-листы 3–4 чисты, нет БЛОКЕР/ПРАВКА в новых.
Финальная строка файла — ТОЧНО `AT0_4_EXEC_R2_DONE`.
