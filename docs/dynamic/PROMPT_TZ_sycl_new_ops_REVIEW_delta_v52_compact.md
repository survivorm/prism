# Промпт: ДЕЛЬТА-ревью ТЗ sycl_new_ops v5.2 (компактный, бюджет ≤5 tool-вызовов)

Ты — независимый ревьюер. Задача: проверить ДЕЛЬТУ v5.2 ТЗ sycl_prism eigjac
(параллельный матчинг XLA → ПОСЛЕДОВАТЕЛЬНЫЙ циклический Jacobi, 1 work-item
на матрицу, 1 sweep = 1 запуск ядра, sweep-луп на хосте, host-MGS f64,
C-ABI + offdiag_curve).

## Строгий бюджет (чтобы не упереться в лимит вывода)
НЕ читай файлы целиком. Используй ТОЛЬКО:
1. `sed -n '1,40p' /home/survivor/work/prism/docs/TZ_sycl_new_ops.md` — ревизия/история
2. `sed -n '185,265p' /home/survivor/work/prism/docs/TZ_sycl_new_ops.md` — §2.3 eigjac + C-ABI + начало §2.4/2.5
3. `grep -n "eigjac\|curve\|MGS\|sweep" /home/survivor/work/prism/docs/TZ_sycl_new_ops.md` — перекрёстная сверка
4. (опционально) `sed -n '1,40p' /home/survivor/work/prism/docs/dynamic/TZ_sycl_new_ops_REVIEW.md` — контекст прошлых MINOR (m-28 и т.д.)
5. `write` файла с ревью.

Не читай .cpp, не запускай код — это ревью ТЕХНИЧЕСКОГО ЗАДАНИЯ, не кода.

## Что проверить (чек-лист)
А. ПОСЛЕДОВАТЕЛЬНОСТЬ/НЕПРЕРЕВНОСТЬ:
   - v5.2 согласован со старым текстом ТЗ (нет противоречий: порядок vals,
     offdiag-семантика, ограничения, watchdog §2.1/§5)?
   - "1 sweep = 1 запуск ядра (work-item b обрабатывает матрицу b)" + "сweep-луп
     на хосте: после каждого sweep хост читает offdiag[b] (8·batch B) и
     останавливается" — реализуемо и не противоречит определению offdiag
     ("относительная ‖offdiag(A)‖_F/‖A‖_F на остановке")?
   - offdiag_curve в C-ABI: "offdiag_rel матрицы 0 ПОСЛЕ каждого sweep (до
     остановки; после — финальное значение)" — нет ли двусмысленности с
     длиной буфера (curve_len) при ранней остановке?
Б. АЛГОРИТМ:
   - циклический порядок d=1..n-1, i=0..n-1-d: ровно n(n-1)/2 ротаций,
     каждая пара один раз — верно?
   - "малый корень t = (d−a)/(2b), t = sign(t)/(|t|+√(1+t²)), fudge-guard
     XLA (|b| ≤ 0.1·eps·min(|a|,|d|) ⇒ ротация не применяется)" — корректно
     как спецификация? (внимание: в формуле переменная d конфликтует с
     циклическим d — это MINOR на имя, проверь)
   - сходимость "4–5 sweep'ов до 1e-4, 4–6 до 1e-5 на всех 6 реальных
     jcache" — заявлен замер; проверь, что тест §2.7 согласован (max_sweeps=64,
     offdiag < 1e-4 на остановке).
   - host-MGS f64: "МGS 1 проход, ortho ~5e-15, subdist 1.4e-4 → 8.4e-5" —
     указано ли, ЧТО именно ре-ортогонализуется и как это согласуется с
     детерминизмом (бит-в-бит)?
В. ДЕТЕРМИНИЗМ/WATCHDOG:
   - "один work-item на матрицу, фиксированный порядок, локальной памяти НЕТ,
     редукции — только offdiag (фикс. порядок) и host-MGS ⇒ бит-в-бит" —
     все источники недетерминизма перечислены?
   - "1 запуск = 1 sweep ≈ 3с (n=128, iGPU)" — согласуется с hang-check
     ~20с и max_sweeps=64 (64×3с = 192с — это ОК для одного вызова или
     нужен ли watchdog-порог на суммарное время)?
Г. C-ABI / PYTHON:
   - `prism_sycl_eigjac(A, batch, n, max_sweeps, tol, vals, vecs, offdiag,
     offdiag_curve, curve_len)` — offdiag_curve/curve_len опциональны
     (NULL/0) — корректно? tol — double, max_sweeps — int64?
   - python: `ps.eigjac(A, max_sweeps=12, tol=1e-5, curve=None)`;
     "reference-режим (numpy): curve → ValueError (понятие sweep отсутствует
     — ЧЕСТНО)" — согласовано с семантикой curve?
   - ограничение "1e-6 ≤ tol ≤ 1e-3" vs "умолчание 1e-5" vs тест
     "max_sweeps=64" — нет ли конфликта с заявленным "4–6 sweep'ов до 1e-5"?
Д. ЗАКРЫТИЕ ПРОШЛЫХ MINOR: m-28 (fp32-дно, замер) — закрыт? Другие
   MINOR из TZ_sycl_new_ops_REVIEW.md, относящиеся к eigjac — не сломаны
   дельтой?

## Формат вывода (ОДИН write, файл
/home/survivor/work/prism/docs/dynamic/TZ_sycl_new_ops_REVIEW_delta_v52.md):

```
# ДЕЛЬТА-ревью ТЗ sycl_new_ops v5.2 (eigjac: cyclic Jacobi)
Дата: ...
Файл: docs/TZ_sycl_new_ops.md, sha256: f6e4f2d12b3bf4e6bcbf8ed210a73e0487cf4ee63d6dc90391b144c5718f5fc2 (463 строки)
Scope: дельта v5.1→v5.2 (§2.3 eigjac + C-ABI + python curve)
VERDICT: PASS | REVISE | FAIL
Findings: (F1..Fn, каждый: ID, MAJOR|MINOR|NIT, что, где (строка), почему, рекомендация)
Checklist: А/Б/В/Г/Д — построчно OK/проблема
Краткое обоснование VERDICT (3-5 строк)
TZ_REVIEW_v52_DONE
```

Финальный маркер в КОНЦЕ файла: `TZ_REVIEW_v52_DONE`.
Будь компактным: не цитируй большие куски ТЗ, пиши только findings.
