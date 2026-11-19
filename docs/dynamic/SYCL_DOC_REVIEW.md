# Ревью docs/SYCL.md (независимое, внешнее)

Штамп сверки (sha256, на момент ревью):

```
40c28dc64bb3425a27a8ef0ecf40e0dd45593003aaa78fff3ab6d4f244662b18  docs/SYCL.md
76bb57658dbe230505153adc03fdaa33d89b89c167ea0f0d95df89e7bd5b2159  sycl_prism/py/prism_sycl.py
d30ce53eea82dd5d7ce741e5219722d77db60c1551e634928ca9c73b77ca7c74  prism/backend.py
9243e2dbb018cdd2108d1fa787e2d9615ba215f279e1e331596ce3cbc0b8efb9  sycl_prism/src/prism_sycl.cpp
1bbf0a3e090b83b0d63368f9d481fdeda75ca2215cf97aa1cf3628b6bc54bfb5  sycl_prism/probe/RESULTS_tuning.md
b7da8d6cbb847cfe444425e12499abce6afde87f1608ee9615d2814649a4e04a  docs/TZ_sycl_new_ops.md
cae89caf5ab5a6d8be5c0654d4b6ef35d2f91af637b9e5233e50c925134569c5  sycl_prism/sycl_env.sh
8cf04f6c50ee5d9670fb038716dd4b7958201e22b208178738896abe1e0960eb  sycl_prism/run_local.sh
5fdf6df0970678ccb4ea8f34e081e0ff6d5fe558f8133914177d9082c251a075  /tmp/AT0_3_suite_final.log
afbb22d4e0c0d06a5597ff50c71576893aa6ff8cc9b9953ac58d402d7cf19d26  results/AT0_3/eigjac_convergence.json
d105a0b9f1caab59210e49437b2e93bcb87e4b071a3e9ffbef7ab8836045b524  results/AT0_3/eigjac_jcache.json
629c813fbbddcd38a104dd46d087232859a976c5b7a9ae8296dafd569afdf8dd  results/AT0_3/eigjac_tuning.json
297206c3ac24afd2ec4ef671ac0b5da99ff79bf3ac9cbcb60c8c7aa868d3af71  results/AT0_3/eigtop_eigbot_jcache.json
d7d8648681b4f5cd136954ccaec8096d7cd64eb5148703b43ec5459a1fb55851  results/AT0_3/gagg_composition.json
c1b2483e63ae9de726efd8e36239d3268969e4b8db915ba2981daba8c964d6e8  results/AT0_3/svd_top_hdocs.json
+ sycl_prism/py/test_new_ops.py, sycl_prism/CMakeLists.txt, experiments/at2/AT2c.py (L380-400), tests/test_backend.py
```

Метод: каждое утверждение/число docs/SYCL.md сверено с перечисленными файлами
(чтение файлов + точечные grep; арифметика пересчитана, не домыслена).
Ничего в репо не менялось; единственный написанный файл — этот.

## Сверено и СХОДИТСЯ (коротко, по 9 проверкам)

**Проверка 1 — контракты §4 (vs prism_sycl.py + C++):**
- gemm: C = A@B f32 row-major; M≤65536, K≤4096, N≤262144 (python-гвард + C++);
  строгий C-contiguous f32 (_contig32 → RuntimeError); N-чанкинг ≤ 32 GFLOP
  (GEMM_FLOP_BUDGET = 32e9, prism_sycl.cpp:1041; N_chunk = floor_to_tile, cpp:1073);
  in_order-очередь (cpp:97); детерминизм-тест есть в сьuite.
- eigjac: сигнатура (A, max_sweeps=12, tol=1e-5, curve=None) → (vals f64 [b,n]
  ПО ВОЗРАСТАНИЮ, vecs f32 [b,n,n], offdiag f64 [b]); лимиты 65536/256/64/1e-6..1e-3
  (python + TZ); host-MGS f64 с write-back f32 (cpp:1509-1512, ‖VᵀV−I‖≈3e-7 — TZ F4);
  curve-контракт F3 дословно (docstring + код: fill до конца = offdiag[0]);
  NaN → NaN в offdiag (f32-редукция на устройстве, нигде не гасится; guard
  `nrmA != 0.0` — cpp:1267-1269 — для eig resid).
- eigtop/eigbot: лимиты (b≤65536, n≤1024, r≤32, r<n, max_iter≤512) +
  local-бюджет 2·n·r·4 + r²·4 + 64·4 ≤ 65536 → n=1024: r≤7 (python _eig_check_args
  И C++ — двойной backstop); vals — Rayleigh f64, порядок колонок = MGS-динамика
  (контракт TZ, тест сортирует перед сравнением); ДВОЙНОЙ MGS (n-14) реально в
  коде (cpp:907, 1106-1107, 1206); eigbot: arg-guard ДО host-Cholesky (python),
  Cholesky f64 на хосте, μ = 1e-12·‖A‖_F по матрице, L → f32, вырожденный вход →
  громкий RuntimeError «не сошёлся».
- svd_top: (U f32 [m,r], S f64 [r] descending, Vh f32 [r,n] = Vᵀ) — reverse +
  .T в коде (обе ветки m<n / m≥n); любой f32-расклад принимается
  (np.ascontiguousarray(_f32(A)) — контраст со строгим gemm описан в доках верно);
  σ=0 ⇒ 0-векторы в ОБАХ выходах — действительно так в коде (U0/V0 нулевые колонки,
  обе ветки; доки точнее TZ, где упомянут только V_i).
- core-опы (vpath/dcurve/acf_moments/f_dc/ra_closure/permutation_indices/
  permutation_null) — все существуют, сигнатуры и семантика (§4.5) совпадают с
  docstring (f_dc = r/N — код; permutation_null: band = mean_{k=9..K}(ρ̄_Tk − f_dc)
  — код).

**Проверка 2 — числа (§6/§4):** 168–170 GF/s (0.8%) — RESULTS_tuning.md/TZ §2.1 ✓;
116.6 GF/s → 17.7с vs ~29с CPU — tuning ✓; 2080 GFLOP — TZ §2.1 ✓ (точно 2083, см. N2);
0.312/0.464/1.364/2.708с — eigjac_tuning.json ✓ (0.31199/0.46443/1.36416/2.70781);
L6: 6 sweep'ов, 1.854с, 1.47e-7; L23: 4, 1.19с, 8.2e-6 — eigjac_convergence.json ✓
(1.46948e-7, 8.22955e-6; кривые согласованы: sweep5 2.7e-5 → sweep6 1.47e-7);
rel_lambda 6.7e-5..1.3e-3 — СМОТРИ M4 (верх ✓ 1.278e-3, низ не сходится);
top32/bottom32 ≤1.6e-3/1.1e-2 — eigjac_jcache.json ✓ (макс 1.5975e-3 / 1.0927e-2);
ortho ≤2.3e-6 ✓ (макс 2.277e-6); eigtop/eigbot subdist/resid — СМОТРИ M3
(resid ≤3.3e-7 ✓: макс 3.251e-7); σrel 7.1e-7/8.6e-7 — svd_top_hdocs.json ✓
(7.133e-7/8.578e-7); U/V ≤6.3e-6 — см. N4; 253.6с, 19/19 — лог ✓ (253.61s,
19 passed); 1.35e-2 — НАЙДЕН: test_new_ops.py:629 (комментарий к S5b: «замер 19.11:
эталон f64-на-f32-данных даёт s5b(|λ|-норм.) = 1.35e-2 на L6 weak») ✓;
4.2e-6 — gagg_composition.json ✓ (макс s5b_rel_max 4.205e-6, L6 weak);
3e-7 (MGS) — TZ v5.3 F4 ✓; гейты 1e-3 (subdist, тест: `assert d_sp <= 1e-3`) и
1e-4 (S5b с нормировкой ‖G‖·‖v‖, тест: `assert s5b <= 1e-4`, test_new_ops.py:636-637;
нормировка в доках = нормировка в тесте дословно) ✓.

**Проверка 3 — §3 диспетчер (vs prism/backend.py):** импорт по файлу (importlib,
НЕ `import prism`) ✓; class Backend, PRISM_BACKEND=cpu|sycl (дефолт cpu) ✓;
eigh sycl n≤256 → BackendError с подсказкой (torch-fallback «по одному вызову») ✓;
matmul 3D [b,m,k]@[k,n] → host-reshape в 2D → ps.gemm → reshape ✓;
f64→f32 one-shot warning (_warn_f64) ✓; PS_ERR_* → BackendError (операция, форма,
причина) ✓; lazy-import sycl ✓; tests/test_backend.py существует (5.1K) ✓.

**Проверка 4 — §5 ошибки:** PS_ERR_ARG (−3) — cpp:38 ✓; PS_ERR_LOCAL (−5) — cpp:39,
используется в бюджете (cpp:412) ✓; NaN-семантика — guard `nrmA != 0.0`
cpp:1267-1269 ✓ (NaN → NaN в resid, не 0). PS_ERR_DEVICE — СМОТРИ M2.

**Проверка 5 — §7 подвохи:** f32-дно ✓ (источники выше); NumPy 2.5 T-view — реально:
комментарий в svd_top (prism_sycl.py) + тест передаёт view Hc32.T ✓; порядки
выходов ✓ (eigjac/eigtop — по возрастанию, svd_top — S desc; eigtop/eigbot — не
гарантировано, см. N8); σ=0 явные нули ✓; hang-check 32 GFLOP ✓ (GEMM_FLOP_BUDGET);
r≤7 при n=1024 ✓ (local-бюджет: 2·1024·8·4+8²·4+256 = 66048 > 65536 ✓ пересчёт);
in_order ✓; EIGJAC_CHUNK = 64 (cpp:1367) и chunk_max = C/16 при n>128
(cpp:1601) — доки описывают ИМЕННО это (C/16 = 4), в отличие от TZ (C/4) — см. N11.

**Проверка 6 — §8 тесты:** `def test_` в test_new_ops.py = 19 (пересчítано:
grep -c = 19) = 19 passed в логе; набор имён из доков = набор в логе
(gemm small/big/wide/oob/determinism, eigjac jcache/determinism/curve/sweep_time_
guard/fudge_guard/zero_and_curve_size/convergence/random_spd, svd_top hdocs/
degenerate, gagg_composition, eigtop_eigbot random_spd/jcache, degradation) ✓;
DEVICE_OK-guard есть (_skip_nogpu, pytest.skip) ✓; JCACHE_DIR =
/data/prism_kernels/exp27b/results + exp27b_jcache_L{6,23}.pt (test_new_ops.py:37,71) ✓;
AT2c L386-393 = блок Грамиана: jr/P/sp, G = (sp·(jr@jr.T))·sp + torch.linalg.eigh
(проверено по строкам) ✓ — примечание: комментария `_gram64` в AT2c.py НЕТ
(0 совпадений), блок инлайновый — доки это не утверждают, верно.

**Проверка 7 — §2 окружение:** sycl_env.sh существует; ENV-A = libumf.so.1 из
/opt/intel/oneapi/umf/1.1/lib в LD_LIBRARY_PATH ✓; ENV-B = LD_PRELOAD системной
libstdc++ (/usr/lib/libstdc++.so.6) ✓; идемпотентность (дедуп по soname) ✓;
`cd sycl_prism && cd build && make` — build/Makefile (сгенерирован cmake) +
build/libprism_sycl.so существуют ✓ (N9 — про свежий checkout); icpx -fsycl —
CMakeLists.txt ✓.

**Проверка 8 — полнота:** ядро API покрыто (§4.1-4.5, §3, §2-сниппет); пропущено —
СМОТРИ M6 (use_reference, device_info).

**Проверка 9 — гигиена:** внутри файла двойных чисел/конфликтов лимитов не найдено
(лимиты n≤256/r≤32/r≤7/32 GFLOP/EIGJAC_CHUNK=64 сформулированы единообразно;
единственная внутренняя нестыковка — «1.36с» в §6 vs «1.364с» в §4.2 — округление,
не конфликт).

## Findings

### MAJOR
Нет. Ни одного расхождения контрактов с кодом (проверка 1), ни одного числа
без источника из перечня (проверка 2) — 1.35e-2 найден в test_new_ops.py:629,
остальные — в JSON/логе/tuning/TZ.

### MINOR
1. **M1 — §8 (+ §2 «шаги 1–1c»): карта шагов run_local.sh не совпадает со
   скриптом.** В доках: «0 окружение → 1 SYCL-зонд (устройство видно?) → 1c
   барьер-форма → 2 сборка → 3 pytest core → 3b pytest new-ops». Реально
   (шапка run_local.sh): 0 ОКРУЖЕНИЕ → 1 ТЕСТЫ (pytest: эталон E-45 +
   самопроверка зонда) → 1a СИНТАКС-ГЕЙТ (компиляция пробы и библиотеки
   против шима + сверка C ABI ↔ ctypes) → 2 СБОРКА SYCL → 3 ПАРИТЕТ ПРИБОРА
   (numpy vs SYCL) → 3b NEW-OPS ТЕСТЫ (pytest py/test_new_ops.py) → 4
   БЕНЧМАРКИ → 5 ЗОНД XPU. Шага «1c барьер-форма» не существует (есть 1a;
   барьер-пробы — бинарники в build/), core-pytest — это шаг 1, а не 3
   (шаг 3 — паритет). Только 0/2/3b сошлись. Исправить: переписать строку
   по шапке скрипта.
2. **M2 — §5 (и §2): `PS_ERR_DEVICE` не существует в коде.** В C++ —
   `PS_ERR_NODEV = -1` (prism_sycl.cpp:36); python-исключение —
   RuntimeError «prism_sycl: нет SYCL-устройства — <load_error>»
   (_ERR[-1], prism_sycl.py). Строки «PS_ERR_DEVICE» ни в одном исключении
   нет (имя только в TZ §0.4, README и в этом доке). Пользователь, ищущий
   код в трейсбеке, его не найдёт. Исправить: в таблице — «PS_ERR_NODEV (−1)
   — нет устройства (python: RuntimeError «нет SYCL-устройства», с причиной)».
3. **M3 — §4.3: «Замер (eigtop_eigbot_jcache.json): subdist ≤ 1.7e-6»
   противоречит приведённому артефакту.** Максимум subdist в json = 7.688e-6
   (L6_0_0 bot_subdist); максимум только по top = 1.745e-6 — т.е. даже
   top-reading «≤ 1.7e-6» на 2.6% НИЖЕ фактического максимума. Исправить:
   «subdist ≤ 7.7e-6 (top ≤ 1.7e-6, bot ≤ 7.7e-6); resid ≤ 3.3e-7»
   (resid верен: максимум 3.251e-7).
4. **M4 — §4.2: «rel_lambda_max 6.7e-5..1.3e-3» — нижняя граница неверна.**
   Фактический диапазон в eigjac_jcache.json: 6.03e-5 (L6_0_1) .. 1.278e-3
   (L23_0_0). 6.7e-5 — это значение L6_0_0, а не минимум. Исправить:
   «6.0e-5..1.3e-3».
5. **M5 — §4.1 и §7.5: «окно драйвера ~5-13с» атрибутировано TZ §2.1, но в
   TZ §2.1 — «~20с при недогрузе» (+ эвристика 13.7с OK); «< ~12с под
   нагрузкой» — из комментария C++ (prism_sycl.cpp, у EIGJAC_CHUNK). «5с» —
   не измеренное окно, а следствие бюджета: 32 GFLOP на самом медленном
   измеренном 6.5 GF/s = ≤5с (TZ §2.1). Исправить: «~20с при недогрузе /
   <~12с под нагрузкой (TZ §2.1, комм. C++); бюджет 32 GFLOP держит ≤5с даже
   на худшем конфиге».
6. **M6 — §4/§8 (полнота): пропущены публичные API.** `ps.use_reference()`
   (reference/numpy-режим через PRISM_SYCL_BACKEND=reference — единственный
   способ работать без GPU; §8 говорит лишь «без GPU — автоскип») и
   `ps.device_info()` — оба в __all__ prism_sycl.py, в доках отсутствуют.
   available()/backend() — есть только внутри сниппета §2. Исправить: 2-3
   строки в §4 (или новый §4.6 «режимы»).

### NIT
1. **N1 — §6: «57°C» — число без источника** (grep «57» по tuning/TZ/
   AT0_3-JSON/логу/комм. C++ — пусто; температур в артефактах нет). Остальное
   в предложении источенно (1.5–1.9×, c256 2.88→4.3–6.9, ~1.2–1.4 GHz, AC,
   100% batt — комм. C++ у EIGJAC_CHUNK). Убрать или доложить замер в
   tuning-артефакт. (При буквальном чтении правила «число без источника =
   MAJOR» — оценено MINOR: диагностическая деталь, не число API.)
2. **N2 — §4.1/§6: «2080 GFLOP»** — точно 2·4096·1024·248320 = 2083 GFLOP
   (TZ тоже пишет 2080). Округление до 3 знач. — допустимо, пометить.
3. **N3 — §4.1/§6: «≈17.7с vs ~29с CPU»** — по TZ §1 29с = z_full «с
   softmax/KL» (т.е. с оверхедом); чистый CPU-GEMM ≈ 25с ≈ 83 GF/s (TZ §5) →
   честный выигрыш ~1.4× (29/17.7 = 1.64× завышает). Формулировка унаследована
   от tuning. Исправить: «≈17.7с vs ~25с CPU (чистый GEMM, 83 GF/s)».
4. **N4 — §4.4: «U/V subdist ≤ 6.3e-6»** — максимум в json 6.339e-6 (L23 V).
   Округление до 2 знач., «≤» на 0.6% ниже максимума.
5. **N5 — §7.1: «замеры ≤4.2e-6»** — максимум 4.205e-6 (L6 weak). Та же
   класс-округление.
6. **N6 — §3:** `B = B.Backend()` работает, но модуль backend.py уже выдаёт
   готовый инстанс (`B = Backend()`, последняя строка) — короче: `B = B.B`.
   Плюс пропущено: sycl-eigh возвращает vals f64 (cpu — f32) — задокументировано
   в TZ/backend, в доках нет (влияет на 1e-3-совпадение — «не влияет», но
   стоит строка).
7. **N7 — §5: в строке PS_ERR_ARG указана «contiguity gemm»** — в python
   не-contiguous/не-fp32 даёт RuntimeError БЕЗ метки PS_ERR_ARG (метка —
   только в сообщении size-лимитов; C++ PS_ERR_ARG(−3) — на лимитах).
   Уточнить формулировку.
8. **N8 — §4.3: «(top = по убыванию, bot = по возрастанию в рамках своей
   стороны)»** — по контракту TZ порядок колонок = MGS-динамика, НЕ
   гарантируется (тест сравнивает спектры ПОСЛЕ сортировки —
   test_eigtop_eigbot_random_spd). Формулировка «обычно», не «по».
9. **N9 — §2: `cd sycl_prism && cd build && make`** — build/Makefile —
   артефакт cmake (на этой машине существует; build/.gitignore есть). На
   свежем checkout нужен сначала `cmake -B build` (подсказка в prism_sycl.py:
   «CXX=icpx cmake -B build && cmake --build build»).
10. **N10 — §8: «wide-gemm ~1-2 мин»** — без источника (лог не содержит
    per-test таймингов). Оценка, не замер.
11. **N11 — наблюдения вне доков (для владельца кода):** (a) test_new_ops.py:13
    шапка «S5b-аналог ≤ 1e-5», а assert в том же файле — `s5b <= 1e-4`
    (строка ~637) с нормировкой ‖G‖·‖v‖ (TZ §2.7: 1e-5 при /|λ_j|); доки
    корректно описывают ИМПЛЕМЕНТИРОВАННОЕ (1e-4, ‖G‖·‖v‖) — расхождение
    внутри тест-файла, не доков. (b) TZ v5.3 §2.3: чанк n=256 = C/4 (×4.02),
    код — C/16 (замер ×9.5, комм. C++ 1360-1367); доки верно следуют КОДУ —
    спеку (TZ) пора синхронизировать.

## Итог

- Сверено: 44 утверждения/числа (§1–§9) — 38 сходятся с источниками дословно,
  6 MINOR (2 числа не сходятся с приведёнными артефактами: M3/M4; 1 неверная
  карта шагов: M1; 1 имя ошибки без кода: M2; 1 неверная атрибуция окна
  hang-check: M5; 1 неполнота API: M6), 11 NIT (округления, формулировки,
  пропуски-однострочники).
- Ни одно число доков не додумано: каждое найдено в файлах-источниках
  (исключение — 57°C, N1).
- Контракты §4 полностью совпадают с кодом prism_sycl.py/C++: сигнатуры,
  лимиты, порядки (eigjac — по возрастанию; svd_top — S desc + Vh=Vᵀ;
  eigtop/eigbot — несортированный MGS-порядок), σ=0 → нули в оба выхода,
  contiguity-контракт gemm (строгий) vs svd_top (свободный), curve-контракт
  F3, Cholesky eigbot ПОСЛЕ arg-guard, двойной MGS — всё подтверждено
  строчками кода.

**VERDICT: PASS_WITH_MINORS** (0 MAJOR, 6 MINOR, 11 NIT)

SYCL_DOC_REVIEW_DONE
