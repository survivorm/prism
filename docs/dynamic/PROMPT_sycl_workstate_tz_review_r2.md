# Промпт: ВТОРОЙ проход ревью ТЗ SYCL-WORKSTATE (дельта)

Ты — тот же независимый ревьюер. Первый проход: `docs/dynamic/TZ_sycl_workstate_REVIEW.md`
(VERDICT: REVISE, 3 MINOR + 3 NIT). ТЗ переправлено. Проверь ДЕЛЬТУ:

1. **MINOR 1.1** (гигиена вне гейтов) — закрыт? (см. G6(b)(c)(d) в §4 и §3.6)
2. **MINOR 3.1** (область LD_PRELOAD) — закрыт? (§6)
3. **MINOR 4.1** (производитель артефакта + sanitize) — закрыт? (§3.7, G6(a))
4. **NIT 2.1** (цифры GLIBCXX) — закрыт? (§2 ENV-B)
5. **NIT 4.2** (формула «ноль abort» в G1) — закрыт?
6. **NIT 4.3** (неоднозначность G4) — закрыт?

Проверь также, что правки не ввели новых противоречий (сверь §3 с §4, §6 с §3.2).

Файл: `/home/survivor/work/prism/docs/TZ_sycl_workstate.md` (новый sha256 пересчитай).

Результат: ПЕРЕЗАПИШИ `/home/survivor/work/prism/docs/dynamic/TZ_sycl_workstate_REVIEW.md`
как ревью v2 (в заголовке — v2, дата, НОВЫЙ sha256 ТЗ; кратко: статус каждого MINOR/NIT
закрыт/нет; новые находки, если появились; VERDICT: PASS|REVISE|FAIL; финальная строка
REVIEW_DONE). ТЗ не редактируй.
