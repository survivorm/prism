# Повторное ревью (round 2b): AT0_2b v2.1 — MINOR-фиксы

Роль: внешний ревьюер. Ревью round 2 (docs/dynamic/AT0_2b_REVIEW_r2.md) дало PASS_WITH_MINORS
(4 MINOR + 3 NIT). Исполнитель применил фиксы (v2.1): store
b7e90c9366d9bbad → **15ad06bdae5c14c5** (claim file sha 014bf176246540aa).

Задача: независимая верификация, что (1) изменения v2→v2.1 — только заявленные (НУЛЬ
численных изменений в cells/агрегатах), (2) все 4 MINOR + 3 NIT закрыты по существу,
(3) цепочка пинов согласована.

## Файлы
- v2 store (superseded): `results/AT0/store/b7e90c9366d9bbad.json`
- v2.1 store (текущий): `results/AT0/store/15ad06bdae5c14c5.json`
- claim (текущий): `results/AT0/AT0_2b_RA.json` (file sha 014bf176246540aa)
- manifest: `results/AT0/AT0_2b_manifest.json`; tracked лог: `results/AT0/AT0_2b_run.log`;
  stdout: `/tmp/at02b_r2b.log` (RC=0, маркер ALL PASS)
- Скрипт: `experiments/AT0_2b_ra_split.py` (v2.1)
- Ответка: `docs/dynamic/AT0_2b_REVIEW_r2_answer.md`

## Обязательные проверки
1. **Числовой diff v2→v2.1:** json-сравнение ВСЕХ листьев (кроме meta). Критерий:
   `cells` — bit-eq (canonical JSON sort_keys); агрегаты (все 15 пул×слой) — bit-eq;
   gates — bit-eq КРОМЕ gate_max.gram (v2: 0.0 инициализация → v2.1: 6.439293542825908e-15 —
   измеренный max, пересчитай по store сам); meta — только 4 ключа (data_sha256-ключи,
   gate_max, hash_convention, script_sha256). Любой иной числовой diff = MAJOR.
2. **MINOR-1:** gate_max.gram = max per-cell gram по store (6.44e-15)?
3. **MINOR-2:** tracked run.log содержит «AT0_2b gates: ALL PASS» (последняя строка)?
   Совпадает с stdout-логом (по числу строк/содержимому)?
4. **MINOR-3:** data_sha256-ключи = basenames; location-invariant (пересчёт canonical sha
   из копии в /tmp дал бы тот же store sha? — можно не перепускать: проверь, что ключи не
   содержат абсолютных путей и hash_convention описывает правило).
5. **MINOR-4:** пре-рег §2 — «corpus 82..254»?
6. **NIT-1/2/3:** мёртвая D убрана (grep); assert len(r)==len(a) в _boot_ci_joint;
   except-список decode-пробы расширен.
7. **Цепочка:** manifest.claim_sha256 = sha256 файла claim; manifest.store_file_sha256 =
   sha256 файла store; claim.meta.script_sha256 = sha256 файла скрипта; store_sha256 =
   canonical payload hash (пересчитай).
8. **Регрессия:** verdict'ы 15× NO_DH_DC; primary L23 contrib_R med −0.0015142104413634894 /
   contrib_A med 1.0015142104413635 (как в v2); n_absent = 2; primary = 37.

## Ограничения
- НЕ менять файлы репо (чтение + /tmp; запись — только addendum в
  docs/dynamic/AT0_2b_REVIEW_r2.md).
- Числа — только из фактических пересчётов.

## Формат вывода
В конец docs/dynamic/AT0_2b_REVIEW_r2.md дописать секцию:

```
## 5. Addendum round 2b (v2.1, store 15ad06bd)
- Числовой diff v2→v2.1: <состав>
- MINOR-1..4 + NIT: <закрыт/не закрыт — доказательство>
- Цепочка: <OK/NA> | Регрессия: <OK/NA>
- Вердикт: PASS/FAIL
```

Финальный маркер в stdout: `REVIEW_DONE`.
