# Ревью E94/S5 r3 — сверка двух отчётных правок после r2-PASS. Ты ревьюер, не исполнитель.

После PASS в r2 в `experiments/exp94_s5_dryrecon.py` внесены ДВЕ чисто отчётные
правки (числа менять не должны):
1. **Явные нули в buckets**: `summarize()` теперь выводит все 4 корзины
   (EXPLAINED/TIE/UNEXPLAINED/INCONCLUSIVE) с нулями; per_doc — тоже.
2. **Vocab scope**: в `runs.*.vocab_text_check` добавлено поле `scope`
   (серверный текст виден только для id из top_logprobs ответа).

## ЛИМИТ
Ответ **≤ 1500 символов**. Читай только: `experiments/exp94_s5_dryrecon.py`
(точечно: summarize, per_doc, vocab_text_check) и `results/e94/S5.json`.

## Чек-лист

**C1.** Изменён только отчёт: buckets/per_doc — те же суммы (6144), flips 16,
g_need_max 0.994864 (r1), H0_supported=true. Нет ли нового кода в логике
(classify/dry_apply/вердикт)?

**C2.** Не ломает ли явные нули сравнения в `negative_controls` (разность buckets
между референсом и вариантами) и в verdict (b.get(...))?

**C3.** Точка: `per_doc` теперь dict (не Counter) — все потребители per_doc в файле
совместимы (json.dump, сравнения)?

Формат — в `docs/dynamic/2_REVIEW_E94_S5_exec_r3.md`:
```
ВЕРДИКТ: PASS | BLOCK
C1: <...>
C2: <...>
C3: <...>
```
Последняя строка stdout: `S5_R3_DONE`
