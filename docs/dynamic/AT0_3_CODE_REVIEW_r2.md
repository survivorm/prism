# ДЕЛЬТА-ревью AT0_3 eigjac — r2 (только правки по r1)
Дата: 2026-11-18
Скоп: дельта r1→r2 = sycl_prism/py/test_new_ops.py (3 ханка: 259–291, 402–421, 425–457);
сверка со src/prism_sycl.cpp (рабочее дерево), py/prism_sycl.py, TZ §2.7, findings r1 (C1–C4).
Статическое ревью; запуск — пункт 4 (pending).

VERDICT: PASS

## 0. Проверка дельты
- `git diff HEAD --stat -- sycl_prism/` = ДВА файла: py/test_new_ops.py (+71/−20) и
  src/prism_sycl.cpp (+6/−6). По .cpp — ровно ОДИН ханк, строки 1552–1560: fill-фикс
  `k = (stop_sw > 0) ? stop_sw : max_sweeps` — тот, что уже сверен в r1 (чек-лист Г, cpp:1559).
  Утверждение «C-код не менялся с r1» ПОДТВЕРЖДЕНО: рабочее дерево идентично состоянию,
  которое читал r1. Коммит HEAD (3e4c0bd) ещё содержит старый fill
  (`last = min(stop_sw, max_sweeps)`, при stop_sw==0 fill отсутствует вовсе —
  curve[max_sweeps..curve_len) не инициализируется), фикс живёт только в рабочем дереве.
- Остальные файлы в git status (docs/AT0_2b_*, AT2d/AT3/AT8_2, experiments/AT3.py, ...) —
  чужой стрим, не читались.

## 1. Закрытие findings r1
- **C1 (MINOR) — ЗАКРЫТО.** test_eigjac_convergence_jcache переписан (test_new_ops.py:425–457)
  в точность под TZ §2.7 m-28 (TZ_sycl_new_ops.md:373–376): на каждый jcache ОДИН вызов
  `ps.eigjac(g32, max_sweeps=16, tol=1e-5, curve=curve)` с буфером `np.empty(16)` (py:437–438),
  ровно 2 реальных jcache — L6 d0t0 и L23 d0t0 (py:433, 435), assert
  `sweeps_to is not None and sweeps_to <= 16` (py:443–444), артефакт
  results/AT0_3/eigjac_convergence.json (py:456–457) с sweeps_to_1e-5/wall_s/offdiag_final/
  curve_per_sweep (все float()/int — JSON-безопасно). Старая конструкция (7 прогонов
  k=1..64, буфера нет, assert «<1e-4 на 64») удалена. Контракт TZ соблюдён построчно.
- **C2 (MINOR) — ЗАКРЫТО.** В curve-регрессии добавлен `assert curve2[0] > off2[0]`
  (test_new_ops.py:289). Проверка лова: при баге «fill [0..curve_len) = off[0]»
  curve2[0] == off2[0] → ассерт ПРОВАЛИВАЕТСЯ ✓. Важно: главная часть test_eigjac_curve
  (py:257–271) к этому багу слепа — при полной переписи буфера curve[0] == off[0],
  неубывание, curve[k−1]==off[0] и fill-ассерт все проходят (при bug k=1, все
  тождественно). Значит, новый ассерт — несущая, а не дублирующая проверка.
  (Нюанс о фактическом стоп-режиме этой матрицы — C5/C6 ниже.)
- **C3 (NIT) — ЗАКРЫТО.** Новый test_eigjac_zero_and_curve_size (py:405–422).
  (a) Нулевая матрица: в ядре все ротации skip (bq==0, cpp:1392), off2=0, tot=0 →
  `pOff[b] = (tot > 0.f) ? sqrtf(off2/tot) : 0.f` даёт ровно 0.0 (cpp:1444) → стоп после
  1 sweep (0 < tol, cpp:1551–1553); diag=0 → vals=0 (строго, f64); V инициализирован
  базисом (cpp:1369–1377), MGS базис не трогает (dot=0, nrm=1 — строго) → vecs=I;
  NaN невозможно. Ассерты теста (off[0]==0.0, no NaN, vals=0, vecs≈I, py:413–418)
  совпадают с C-поведением. (b) `curve=np.empty(3)`, max_sweeps=16 → ValueError
  на python-стороне ДО C (py/prism_sycl.py:604–607, c_arr.size < max_sweeps),
  `pytest.raises(ValueError, match="curve")` (py:419–420) ловит — сообщение
  «prism_sycl: eigjac: curve — f64 1D, size >= max_sweeps...» содержит "curve" ✓.
- **C4 (NIT) — НЕ ПРАВилось, как и предписано** (ревьюер r1 сверил структурно;
  ассерты jcache-теста вне выреза по бюджету — статус «оставить как есть» принят).

## 2. Оффс-один (самонаходка при реализации C2)
C-код (cpp:1543–1560): в лупе sw=1..max_sweeps после sweep sw пишется
`hCurve[sw-1] = off64[0]` (cpp:1549); стоп при ВСЕХ off64 < tol — stop_sw = sw (cpp:1551–1553);
fill `hCurve[k..curve_len) = off64[0]`, `k = (stop_sw>0) ? stop_sw : max_sweeps` (cpp:1559–1560).
Новый тест (py:266): `k = argmax(curve < tol) + 1`.
- Корректность: пусть i* — первый индекс с curve[i*] < tol. curve[i*] — это offdiag ПОСЛЕ
  sweep i*+1 (индексация curve[sw−1]), значит для batch=1 стоп случается после sweep i*+1,
  т.е. stop_sw = i*+1 = k. Тогда curve[k−1] = curve[i*] = off64[0] — ТОТ ЖЕ f64, что
  копится в hOff[0] (cpp:1575), ⇒ |curve[k−1] − off[0]| = 0 < 1e-15, fill [k..64) ✓.
  Цифры прогона: curve[6]=1.66e-5 ≥ 1e-5, curve[7]=2.46e-8 < 1e-5 ⇒ i*=7, k=8,
  стоп после sweep 8, curve[7] == off[0] — согласуется с C-кодом. k = i+1 КОРРЕКТЕН.
- Прежний тест: с этими же цифрами old-k = 7, old-ассерт |curve[6] − off[0]| ≈ 1.66e-5 > 1e-15
  — на здоровом коде он ПАДАЛ, а не «проходил слепым». Проходил только в дегенеративных
  ветках: (i) нет сходимости до tol (k=64: curve[63] — последний per-sweep = финал —
  тождественно, и слеп к fill-багу); (ii) сходимость за 1 sweep (i*=0, old-k=0 →
  curve[−1] — тоже слеп). Ветка «curve[i−1] случайно == curve[i]» из промпта — мёртвая:
  curve[i*−1] ≥ tol > curve[i*], равенство невозможно. Итог: old-ассерт проверял позицию
  ПЕРЕД значением на стопе (сломан в нормальной ветке + слеп в дегенеративных) — фикс верен.
- Условие эквивалентности: k = argmax+1 точен ТОЛЬКО при batch=1 (curve ведёт matrix 0,
  а стоп требует ВСЕ матрицы < tol; при batch>1 i*+1 — нижняя граница stop_sw).
  Оба теста с curve используют batch=1 (py:252–256: (1,128,128); py:435: [None,:,:]) — OK.
- Другие места: grep "argmax" по файлу — ровно 2 вхождения: py:266 (curve) и
  py:442 (jcache, `sweeps_to = argmax(conv) + 1`) — конвенция «+1» согласована;
  других индексов curve/оффс-один в test-файле нет (grep "curve" — 45 совпадений,
  все в test_eigjac_curve и test_eigjac_convergence_jcache).

## 3. Новые findings
- **C5 | NIT | test_new_ops.py:275–288, 408–409 — неточные комментарии/docstring в
  новых блоках (поведение не затрагивается).** (1) «стоп по max_sweeps» (py:275–276,
  277, 281–282, 292): матрица A*50 на самом деле сходится к tol=1e-6 ровно на sweep 8
  (off_rel масштаб-инвариантен, curve2 ≈ curve: curve[7]=2.46e-8 < 1e-6 < curve[6]=1.66e-5)
  ⇒ стоп по tol со stop_sw=8>0, НЕ по max_sweeps. Позиция fill та же (k=8), ассерты
  валидны — неточен только комментарий. (2) «off ≈ O(10)» (py:285): off — ОТНОСИТЕЛЬНОЕ
  ‖offdiag‖_F/‖A‖_F ≤ 1 (cpp:1444, py:579); после sweep 1 это O(0.1–0.5), не O(10).
  (3) docstring (a): «offdiag — абсолютный, деления на fro нет» (py:408–409): фактически
  offdiag — относительное sqrt(off2/tot), деление ЕСТЬ, но защищено guard (tot > 0.f)
  (cpp:1444), при нуле даёт 0.0 — вывод «NaN невозможно» верен, обоснование неверно.
  Чинить: правка трёх комментариев (1 строка каждый), severity NIT.
- **C6 | NIT | ветка fill при ИСТИННОМ стопе по max_sweeps (stop_sw==0, k=max_sweeps,
  cpp:1559–1560) не покрыта тестами с curve.** Все прогоны с curve-буфером сходятся
  в пределах max_sweeps (curve2 — ровно на последнем sweep по tol). Сам луп со
  stop_sw==0 покрывается sweep_time_guard (max_sweeps=1, dense-матрица), но там
  curve=None — т.е. комбинация «curve != nullptr И стоп по max_sweeps» не проверялась:
  именно этот случай рождал в HEAD-коммите мусорный хвост (нет fill вовсе). Покрыть:
  матрица с off_rel > 1e-6 после 8 sweep'ов (tol ограничен снизу 1e-6, сузить нельзя —
  нужна медленная сходимость, напр. кластер близких собственных значений). Чинить:
  опционально, severity NIT (r1 чек-лист Г проверил ветку статически).

## 4. Пункт 5 — запуск
**pending: 19/19 подтвердит прогон отдельно ДО коммита.** На момент ревью
/tmp/AT0_3_shim_full.log: collected 19 items, test_gemm_small PASSED,
test_gemm_big в процессе (т.е. 2/19; eigjac-тесты стоят 5–13-ми в порядке файла
и ещё не стартуют). Статические пункты 0–3 полные. GPU (iGPU): device виден в sycl-ls,
queue не создаётся («No device of requested type available») — драйвер в деградированном
состоянии, GPU-прогоны невозможны (блокер драйвера, не кода).

## 5. Чек-лист: что читал
- git: `git status --short`; `git diff HEAD --stat -- sycl_prism/`;
  `git diff HEAD -- sycl_prism/src/prism_sycl.cpp` (1 ханк, cpp:1552–1560);
  `git diff HEAD -- sycl_prism/py/test_new_ops.py` (3 ханка: py:259–291, 402–421, 425–457);
  `git show HEAD:sycl_prism/py/test_new_ops.py` (старый test_eigjac_curve, py:245–281 в HEAD);
  `git log --oneline -5` (HEAD=3e4c0bd).
- sycl_prism/src/prism_sycl.cpp:1330–1579 — eigjac-секция: комментарий v5.3 (1334–1357),
  EIGJAC_CHUNK (1361), v_init (1363–1384), sweep_chunk (1386–1445: fudge-guard 1392,
  ротация 1396–1437, offdiag_rel + guard tot>0 1438–1444), mgs_host (1447–1472),
  sort (1474–1496), impl (1498–1579: ограничения 1500–1512, чанки 1538–1540,
  sweep-луп 1541–1553, fill 1555–1560, D2H/sort/MGS/hOff 1562–1577).
- sycl_prism/py/prism_sycl.py:571–644 — биндинг eigjac (docstring 572–590, форма 592–593,
  ограничения 595–599, curve: asarray/ndim/size 601–607, contiguity 608–611,
  reference 613–620, call 631–642); py:148–150 — по r1 (не перечитывал).
- sycl_prism/py/test_new_ops.py:246–291 (test_eigjac_curve: k=266, C2-ассерт 289,
  регрессия 275–291), 405–422 (test_eigjac_zero_and_curve_size), 425–457
  (test_eigjac_convergence_jcache: argmax 442), 39 (DT_SET), 42–44 (DOMAIN/RESULTS),
  27 (import pytest); grep "argmax" (266, 442), grep "curve" (45 совпадений).
- docs/TZ_sycl_new_ops.md:360–384 — TZ §2.7: m-28 (373–376), sweep-time guard (377–383),
  F3 curve-контракт (384–388).
- docs/dynamic/AT0_3_CODE_REVIEW_r1.md — целиком (findings C1–C4, чек-лист А–З).
- /tmp/AT0_3_shim_full.log — tail (2/19 на момент ревью).

Краткое обоснование VERDICT: дельта r2 — только test_new_ops.py; C-код идентичен
состоянию r1 (в diff vs HEAD — только тот же fill-фикс, что сверен в r1 чек-листом Г).
C1 переписан под TZ m-28 построчно, C2-ассерт численно ловит fill-перепись
(и несущий — главная часть к ней слепа), C3 согласован с C-кодом по guard tot>0,
C4 не тронут по предписанию. Оффс-один k=i+1 корректен для batch=1 (оба теста),
второе argmax согласовано, других оффс-один нет. Новые findings — только NIT
(C5: три неточных комментария; C6: ветка fill при stop_sw==0 не покрыта).
Пункт 5 pending — прогон 19/19 подтверждается отдельно ДО коммита.

AT0_3_CODE_REVIEW_r2_DONE
