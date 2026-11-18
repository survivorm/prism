# Промпт: КОД-ревью eigjac v5.3 (реализация TZ_sycl_new_ops §2.3/2.5/2.7)

Ты — независимый ревьюер КОДА (не ТЗ). Задача: проверить реализацию eigjac v5.3
на СООТВЕТСТВИЕ TZ v5.3 и на БАГИ (детерминизм, race, OOB, ABI, согласованность
C↔python↔тесты).

## Файлы (читай ТРЕЗВО, не всё — только eigjac-секции)
1. C-ядро + host: `sed -n '1305,1570p' /home/survivor/work/prism/sycl_prism/src/prism_sycl.cpp`
   (eigjac_v_init, eigjac_sweep_chunk, eigjac_mgs_host, sort_eigpairs_f64_host, eigjac_impl)
2. C-ABI: `grep -n -A20 "prism_sycl_eigjac" /home/survivor/work/prism/sycl_prism/src/prism_sycl.h`
   + `grep -n -A15 "^int prism_sycl_eigjac" /home/survivor/work/prism/sycl_prism/src/prism_sycl.cpp`
3. Python: `grep -n -B2 -A65 "def eigjac" /home/survivor/work/prism/sycl_prism/py/prism_sycl.py`
   + `grep -n -A6 '"eigjac"' /home/survivor/work/prism/sycl_prism/py/prism_sycl.py` (биндинг)
4. Тесты: `grep -n -A45 "def test_eigjac" /home/survivor/work/prism/sycl_prism/py/test_new_ops.py`
5. TZ-спец: `sed -n '207,290p' /home/survivor/work/prism/docs/TZ_sycl_new_ops.md` (§2.3 + C-ABI)
   и `sed -n '340,380p'` (§2.7 тесты)

Бюджет: ≤8 tool-вызовов. НЕ запускай код, НЕ читай gemm/eigtop/eigbot.

## Чек-лист (каждый пункт: OK / проблема)
А. **Детерминизм по строению** (TZ: 1 WI/матрицу, фикс. порядок, нет local mem,
   редукции фикс. порядок): в eigjac_sweep_chunk — каждый work-item пишет ТОЛЬКО
   свою матрицу (A_[...], V_[...] с b из id)? Нет shared/global-состояния между
   WI'ами? offdiag-редукция — внутри одного WI (последовательный цикл)? Чанкинг
   (b0..b0+chunk) не нарушает независимость матриц?
Б. **Ротация (математика)**: th = sign(t)/(|t|+√(1+t²)), t=(dd−a)/(2·bq); c, s;
   обновление A (кросс-строки/колонки + 2×2-блок + offdiag=0) и V (колонки i,j
   ВКЛЮЧАЯ блок {i,j}×{i,j}) — формулы JᵀAJ / V·M корректны? fudge-guard:
   `bq==0 || |bq| <= 0.1·eps_f32·min(|a|,|dd|)` — skip? (TZ: |A_ij| ≤ 0.1·eps·min(|A_ii|,|A_jj|))
В. **Сортировка + MGS (host)**: sort_eigpairs_f64_host — пары (λ,столбец) вместе,
   stable, по возрастанию; eigjac_mgs_host — MGS 1 проход f64, write-back в f32,
   порядок фиксирован (кол по кол)? MGS ДО или ПОСЛЕ сортировки — и важно ли?
   (TZ: (1) сортировка, (2) MGS — так ли в eigjac_impl?)
Г. **Sweep-луп и стоп**: 1 sweep = ⌈batch/C⌉ чанков; стоп — ВСЕ off64[b] < tol
   ИЛИ max_sweeps; off64 читается ПЕРЕД стоп-проверкой (после sweep)? curve:
   hCurve[sw−1] = off64[0] внутри лупа; fill [last..curve_len) = off64[0]
   (last = min(stop_sw, max_sweeps)) — при стопе по max_sweeps (stop_sw=0?)
   fill не затирает? (Проверь: stop_sw=0 ⇒ last = min(0, max_sweeps) = 0 ⇒
   fill [0..curve_len) = off64[0] — ПЕРЕПИШЕТ всю кривую финальным значением!
   Это баг или нет? TZ F3: «если k < curve_len — curve[k..curve_len) = offdiag[0]»,
   где k = число реально выполненных sweep'ов. При стопе по max_sweeps k = max_sweeps,
   и fill должен быть [max_sweeps..curve_len). Как это в коде?)
Д. **f32 device / f64 host**: dOff/dDiag — Dev<float> (f32)? hOff/hVals — f64?
   Перенос f32→f64 корректен? offdiag_rel f32 (bound ~2e-3) — согласуется с
   стоп-критерием (максимум лишний sweep — TZ F11)?
Е. **C-ABI ↔ python**: сигнатура C (10 арг: A, batch, n, max_sweeps, tol, vals,
   vecs, offdiag, offdiag_curve, curve_len) = биндинг python (10 argtypes)?
   NULL/0 для curve без кривой — python передаёт c_ptr=None, c_len=0 — C
   проверяет `hCurve != nullptr`? (ctypes None → nullptr — да?)
   int64_t/-double-типы совпадают?
Ж. **Тесты покрывают TZ §2.7**: curve (F3-контракт: не-возрастание, curve[k−1]==off[0],
   fill), sweep-time guard (н=128 чанк C, n=256 чанк C/4, C≥256, fail-loud),
   fudge-guard (a диагональ, b guard-skip), det бит-в-бит, jcache (relλ 1e-3/1e-2,
   subdist 5e-2, ortho 1e-4, off < 1e-4, max_sweeps=64), random-SPD — все на месте?
З. **AGENTS.md-ловушки**: guard'ы (fudge-guard, tot>0, curve_len<max_sweeps) —
   designed-сценарии протестированы? json (eigjac_tuning.json, eigjac_jcache.json)
   — numpy-типы/NaN? (float()-касты?) Одинаковая статистика (нет сравнений — lib)?

## Формат (ОДИН write в /home/survivor/work/prism/docs/dynamic/AT0_3_CODE_REVIEW_r1.md):
```
# КОД-ревью AT0_3 eigjac v5.3 (r1)
Дата: ...
Скоп: src/prism_sycl.cpp (eigjac-секция), .h, py/prism_sycl.py (eigjac+биндинг),
py/test_new_ops.py (eigjac-тесты); TZ v5.3 §2.3/2.5/2.7.
VERDICT: PASS | REVISE | FAIL
Findings: (C1..Cn: ID, MAJOR|MINOR|NIT, что, где (файл:строка), почему, фикс)
Checklist А–З: построчно
Краткое обоснование VERDICT (3-5 строк)
CODE_REVIEW_AT03_r1_DONE
```
Компактно (каждый finding ≤4 строки). Маркер в конце файла: `CODE_REVIEW_AT03_r1_DONE`.
