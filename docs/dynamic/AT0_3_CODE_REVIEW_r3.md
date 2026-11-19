# КОД+артефакт-ревью AT0_3 SYCL — r3 (дельта vs HEAD df81a26)
Дата: 2026-11-19
Скоп: `git diff HEAD -- sycl_prism/` — 4 файла (src/prism_sycl.cpp +162/−99*, py/prism_sycl.py,
py/test_new_ops.py, run_local.sh) + 6 артефактов results/AT0_3/ + лог /tmp/AT0_3_suite_final.log
(19/19 PASSED, 253.61s, RC=0). *diffstat: +254/−87 суммарно по 4 файлам.
Предмет — векторизованный eigjac (sycl::float4), gemm N-чанкинг-переписывание, NaN-resid,
EIGJAC_CHUNK 256→64 (перетюнинг 19.11), py/test-фиксы. r1/r2 (C1–C6) закрыты ранее;
здесь — только дельта r2→рабочее дерево.

## 0. sha256 ключевых файлов (на момент ревью)
```
9243e2dbb018cdd2108d1fa787e2d9615ba215f279e1e331596ce3cbc0b8efb9  sycl_prism/src/prism_sycl.cpp
76bb57658dbe230505153adc03fdaa33d89b89c167ea0f0d95df89e7bd5b2159  sycl_prism/py/prism_sycl.py
06c70d0031d7f7b0e61fb462de950e38760613675ab5ca172d186245194371eb  sycl_prism/py/test_new_ops.py
8cf04f6c50ee5d9670fb038716dd4b7958201e22b208178738896abe1e0960eb  sycl_prism/run_local.sh
afbb22d4e0c0d06a5597ff50c71576893aa6ff8cc9b9953ac58d402d7cf19d26  results/AT0_3/eigjac_convergence.json
d105a0b9f1caab59210e49437b2e93bcb87e4b071a3e9ffbef7ab8836045b524  results/AT0_3/eigjac_jcache.json
629c813fbbddcd38a104dd46d087232859a976c5b7a9ae8296dafd569afdf8dd  results/AT0_3/eigjac_tuning.json
297206c3ac24afd2ec4ef671ac0b5da99ff79bf3ac9cbcb60c8c7aa868d3af71  results/AT0_3/eigtop_eigbot_jcache.json
d7d8648681b4f5cd136954ccaec8096d7cd64eb5148703b43ec5459a1fb55851  results/AT0_3/gagg_composition.json
c1b2483e63ae9de726efd8e36239d3268969e4b8db915ba2981daba8c964d6e8  results/AT0_3/svd_top_hdocs.json
```

## VERDICT: PASS_WITH_MINORS

Блокеров нет: gemm-чанкинг, float4-ротации, NaN-guards, curve-fill, svd-ветки, tuning-гейты
и артефакты — все сверены численно и согласованы. 1 MINOR (TZ-контракт C/4 не обновлён под
измеренное C/16) + 4 NIT.

## 1. Findings

**F-r3-1 | MINOR | docs/TZ_sycl_new_ops.md:238–239, 379, 382 — TZ-контракт для n=256
отстал от кода: «чанк C/4 (×4.02)» и гейт «C ≥ 256», код — C/16 и C=64.**
- Где: TZ §2.3 (238–239: «н > 128 ⇒ чанк = C/4 (EIGJAC_CHUNK/4) … твёрдый тест §2.7 —
  256: чанк C/4»), §2.7 (379: «n=256 (F9): чанк C/4», 382: «И C ≥ 256»). Код:
  cpp:1355–1363 (`EIGJAC_CHUNK = 64` + замер), cpp:1596–1598 (`chunk_max = (n>128) ?
  EIGJAC_CHUNK/16 : EIGJAC_CHUNK` = 4), test (C_FIXED=64, C256 = C_FIXED//16 = 4,
  гейты «n128_c64 ≤ 4с», «n256_c4 ≤ 4с», «C_measured == C_FIXED»).
- Что не так: три числа контракта (C/4, ×4.02, C ≥ 256) не совпадают с реализацией
  (C/16, ×9.5, C == 64). Причина измерительная и честная (комментарий cpp:1355–1363:
  bandwidth-contention, t_sweep(n=256): c16=3.6с, c32=4.9с; в состоянии перетюнинга
  iGPU медленнее 1.5–1.9× ⇒ c16 ≈ 5.4–6.8с > 4с, c4 ≈ 2.7–3.4с — финальный замер
  n256_c4 = 2.708с в eigjac_tuning.json). Направление консервативное (чанк 4 вместо 16 —
  медленнее по wall, безопаснее к hang-check), но TZ — контракт, и по TZ «твёрдый тест»
  n=256 проверял бы чанк 16, а не 4.
- Чинить: правка TZ §2.3/§2.7 (C/4 → C/16, ×4.02 → замер ×9.5 с указанием
  /tmp/AT0_3_tsweep_reps.json; «C ≥ 256» → «C == EIGJAC_CHUNK (фиксация тюнинга)»);
  alternatively — пометка в TZ «дельта 19.11: см. комментарий у EIGJAC_CHUNK». Код менять
  не нужно. Серьёзность: MINOR (нет ошибки в коде; расхождение контракт/реализация).

**F-r3-2 | NIT | py/prism_sycl.py:682–684 — svd_top: лишний ascontiguousarray +
недокументированный f64→f32-каст.**
- `a = np.ascontiguousarray(_f32(A))`: `_f32` (py:221–225) уже возвращает
  `np.ascontiguousarray(...)` — внешний вызов — no-op-обёртка. Главное: `_f32` тихо кастит
  любой dtype в f32 — f64-вход в svd_top усечётся без предупреждения, тогда как TZ §2.1
  для ps.gemm требует PS_ERR_ARG на не-fp32, а TZ §2.4 декларирует «SVD A[m,n] f32».
  Комментарий (682–683) обосновывает только layout-расслабление (Hc32.T — view), dtype —
  молча. Чинить: строка в docstring «f64-вход — каст в f32 (как B.matmul)» либо строгая
  f32-проверка; убрать внешний ascontiguousarray.
- Там же (py:707–708, 731–732): комментарий «маска-индексация по НЕ-contiguous слайсу
  возвращает T-view (стрейды (4, 512))» — неточен: слайс U_r = U1[:, m−r:] C-contiguous,
  не-contiguous РЕЗУЛЬТАТ маски. Проверено на numpy 2.5.2 (venv репо):
  `U[:, 121:][:, mask].flags['C_CONTIGUOUS'] == False`, strides (4, 512) ⇒ сам
  ascontiguousarray-каст НЕСУЩИЙ (без него gemm получил бы F-стрейды), только формулировка
  комментария врет. Чинить: «mask-индексация в numpy 2.5 возвращает non-contiguous view».

**F-r3-3 | NIT | py/prism_sycl.py:534–562 — тройное дублирование аргс-чека eigtop/eigbot.**
- Тело нового `_eig_check_args` (534–548) пословно идентично началу `_eig_call` (550–562);
  обе копии дублируют C++-бэкстоп в eig_impl (cpp:1280–1295: range + local-бюджет).
  eigbot проходит чек дважды (helper ДО Cholesky — его смысл — PS_ERR_ARG вместо
  «Matrix is not positive definite»; затем _eig_call). Поведение корректно, числа везде
  одинаковые (65536/1024/32/512/65536B). Чинить: `_eig_call` вызывает `_eig_check_args`
  (−12 строк); не критично.

**F-r3-4 | NIT | src/prism_sycl.cpp:1267–1269 — сломанная индентация комментария
(2 пробела вместо 4) в eig_postprocess у NaN-guard'а. Косметика, код не затрагивает.**

**Примечания к промпту (не findings по коду):**
- Пункт 3 промпта называет блок-эталон «_rel_eig» — фактически изменён `_gemm_check`
  (тот же BLK=8192-механизм; `_rel_eig` не тронут).
- Промпт: gagg «subdist ≤ 6.6e-4» — в финальном артефакте max subdist = 2.74e-5
  (L23 weak); 6.6e-4 — видимо, из промежуточного прогона. Оба ≤ 1e-3 — гейт пройден.

## 2. Обязательные проверки (10 пунктов)

### 2.1 gemm N-чанкинг (C++) — ОК
- Индексация (cpp:945–1031): `B = dB.p + c0` (элемент (0,c0) row-major [K,Nfull]);
  B-чтение `B[(k0+rb)*Nfull + col0 + cb]` с guard `(k0+rb < K && col0 + cb < Nc)`
  (cpp:987–989) — даёт именно B[k][c0+col], т.к. col0+cb < Nc ⇒ c0+col0+cb < c0+Nc ≤ N.
  C-запись `C[row*Nfull + c0 + col0 + lc]` с guard `(row < M && col0 + lc < Nc)`
  (cpp:1028–1029) — в пределах чанка, не выходит за N. Старый баг (B со смещением
  c0*K и Nc-стрейдом) убран.
- Чанк-луп (cpp:1073–1080): nch = floor(32e9/(2·M·K)) → floor_to_tile(TC=64), min TC
  (нет бесконечного цикла); c0 += nch; Nc = min(nch, N−c0) ⇒ покрытие [0,N) ровно,
  записи чанков в C не пересекаются (несомещаемые столбцы [c0, c0+Nc)). Детерминизм:
  in_order-очередь, порядок чанков фиксирован, per-элементные операции неизменны.
- Арифметика: M=4096,K=1024 ⇒ 32e9/8.39e6 = 3814.7 → 3776 (59 тайлов) — совпадает с TZ §2.1;
  M=512,K=1024 ⇒ 30517 → 30464 ⇒ N=248320 → 9 чанков; N=248321 → последний чанк Nc=30465
  (guard col0+cb<Nc ловит +1).
- Связь с тестами: test_gemm_wide (512×1024@1024×248320), test_gemm_oob (33×64@64×97 —
  M не кратно 128, N не кратно 64; 4096×1024@1024×248321), test_gemm_determinism
  (512×1024@1024×4096, 2× bit-exact) — все PASSED в логe (5/5 gemm).

### 2.2 eigjac float4-ветка — ОК
- Формулы ротации идентичны скалярным построчно (cpp:1424–1437 vs 1440–1486):
  th = copysignf(1,tt)/(|tt|+√(1+tt²)), c = 1/√(1+th²), s = th·c;
  ni = c·ai − s·aj, nj = s·ai + c·aj — те же операции на элемент в float4 и k_scalar.
  Fudge-guard (cpp:1410–1413: bq==0 || |bq| ≤ 0.1·eps_f32·min(|a|,|dd|) ⇒ skip) и
  чтение a/bq/dd ДО k-лупа (cpp:1408–1410) — без изменений (r1 чек-лист Б).
- Ветвление: v4ok = выравнивание строк (b·n²+i·n)%4==0 для обеих строк (при n=128/256
  всегда истинно — n кратно 4, USM-база 64B; защита от нечётного n); чистая группа
  [k,k+3] ∩ {i,j} = ∅ ⇔ `k+3 < klo || k > khi` (cpp:1446) — корректно; нечистые группы —
  4×k_scalar (включая блок {i,j}×{i,j} в VT — баг-фикс драфта v5.2 сохранён);
  хвост k = n4·4..n−1 — скаляр (cpp:1481). n=128/256 — хвост пуст.
- A-колонки в float4-ветке пишутся strided-скалярами A_[(k+m)·n + i] = ni4.m() — те же
  4 значения, что k_scalar (симметрия A[k,i]=A[i,k] на живом A). Диагональ rt1/rt2
  (A[i,i] = a − th·bq; A[j,j] = dd + th·bq; A[i,j]=A[j,i]=0) — после k-лупа, на
  неперезаписанных a/bq/dd.
- V--layout: dV → dVT (строки = векторы), инициализация v_init не тронута (I —
  симметрична); финальный транспозиция vecs[b,i,j] = vt[b,j,i] (cpp:1632–1637) —
  корректна: строка VT[j] = вектор j ⇒ столбец vecs[j] = вектор j (контракт TZ:
  столбцы = собственные векторы). Затем sort/MGS — на столбечных vecs, без изменений.
- offdiag-редукция (cpp:1493–1503) не тронута: фикс. порядок, tot/off2 по всей матрице,
  guard `(tot != 0.f)` — см. 2.6. Fill curve (cpp:1615–1621): `k = (stop_sw>0) ?
  stop_sw : max_sweeps`, fill [k..curve_len) = off64[0] — не сломан векторизацией
  (r2-логика к=stop_sw сохранена дословно; hCurve[sw−1] пишется в лупе как раньше).
- Детерминизм: per-element операции и порядок k (0..n−1, группы по 4 в том же порядке)
  неизменны ⇒ бит-в-бит; test_eigjac_determinism PASSED (лог).

### 2.3 EIGJAC_CHUNK=64 — ОК (MINOR — см. F-r3-1)
- Согласованность: cpp:1363 constexpr 64 ↔ test C_FIXED=64 (test:351) ↔ комментарии
  (cpp:1327–1329, 1355–1363; test docstring) — триада консистентна. chunk_max n=256:
  cpp:1597 → 64/16 = 4 ↔ test C256 = 64//16 = 4.
- Гейт 4с сохранён: assert table["n128_c64"] ≤ 4.0, assert table["n256_c4"] ≤ 4.0,
  loud-fail при C < 64 («n128_c64 in table») и drift-гейт C_measured == C_FIXED.
  (Старый гейт TZ «C ≥ 256» заменён — часть F-r3-1.)
- Обоснование перетюнинга в комментарии: замеры 19.11 (c256 = 4.3–6.9с > 4с в состоянии
  перетюнинга; c1024 — DEVICE_LOST; c64 = 1.0–1.9с) + ссылка /tmp/AT0_3_tsweep_reps.json —
  численно согласуется с финальным eigjac_tuning.json (см. 2.9).

### 2.4 svd_top — ОК (NIT — F-r3-2)
- TZ §2.4: S descending — `S = sigma[::-1].astype(f64)`, U_out/U и V0/Vh_out reverse-ятся
  в обеих ветках; Vh = Vᵀ (torch-конвенция) — `Vh_out = V[:, ::-1].T` / `V0[:, ::-1].T`.
  σ=0 ⇒ явные 0-векторы в ОБАХ выходах (m<n: U0 + V; m≥n: U + V0) — TZ §2.7 degenerate;
  clamp ДО sqrt (np.where(l_asc ≤ 1e-12·lam_max, 0, sqrt(max(l_asc,0)))).
- Ветки симметричны: m<n — G = A·Aᵀ (m×m), U_r = eigenv(G), V = AᵀU_r/σ; m≥n — G = AᵀA
  (n×n), V_r = eigenv(G), U = AV_r/σ. Порядок: l_asc — r последних по возрастанию,
  reverse — descending; U_r[:,k] ↔ σ_k ↔ V[:,k] — согласовано (проверено по индексам:
  U_out[:, i] = U0[:, r−1−i], S[i] = σ_{r−1−i}).
- ascontiguousarray — ВСЕ пути gemm: at = ascontiguousarray(a.T); Vn = gemm(at,
  ascontiguousarray(U_r[:, nz])); Un = gemm(a, ascontiguousarray(V_r[:, nz]));
  eigjac(G[np.newaxis, ...]) — G contiguous, newaxis — contiguous view. gagg-тест:
  ps.gemm(Jagg32, ascontiguousarray(Jagg32.T)) — T-view (стрейды (32,4)) — каст нужен.
  Проверено на numpy 2.5.2: mask-индексация U_r[:, nz] → C_CONTIGUOUS=False, strides
  (4,512) — касты несущие, не формальные.
- Вход-расклад: `_f32` вместо строгих `_contig32` — сознательное отклонение, обосновано
  комментарием (тест TZ §2.7 передаёт Hc32.T); TZ-позиция «f32-вход» сохранена
  (f64-каст — единственный побочный эффект, F-r3-2).

### 2.5 eigbot guard — ОК (NIT — F-r3-3)
- Порядок (py:511–531): _contig32 → форма → _eig_check_args (PS_ERR_ARG/PS_ERR_LOCAL)
  → reference-ветка → host-Cholesky → _eig_call. r≥n ⇒ PS_ERR_ARG ДО Cholesky
  (раньше — «Matrix is not positive definite»). Покрыто: test_degradation
  `ps.eigbot((1,8,8), 8)` → match "PS_ERR_ARG" (PASSED). reference-режим: чек ДО
  ref.eig_bottom — ошибка консистентна в CI без GPU.
- Дублирование (helper + _eig_call + C++ eig_impl): избыточно, но не ломает — см. F-r3-3.
- C++-бэкстоп eig_impl: local-бюджет (cpp:1291–1297: 2·n·r·4 + r²·4 + 64·4 ≤ 65536,
  n=1024,r=8 ⇒ 66048 — m-12b) + range-чеки — на месте.

### 2.6 NaN-resid (eig_postprocess) — ОК (NIT — F-r3-4)
- `hResid[b] = (nrmA != 0.0) ? sqrt(res2)/nrmA : 0.0` (cpp:1266–1269): нулевая матрица
  → nrmA = 0.0 → resid = 0.0 (не NaN); NaN во входе → nrmA = NaN, NaN != 0.0 = true →
  sqrt(NaN)/NaN = NaN — честный сигнал; отрицательное nrmA невозможно (fro-норма).
- eigjac: `pOff[b] = (tot != 0.f) ? sqrtf(off2/tot) : 0.f` (cpp:1503) — та же семантика:
  нулевая матрица → ровно 0.0 (test_eigjac_zero_and_curve_size (a): off[0] == 0.0,
  PASSED), NaN → NaN (test_degradation: `np.isnan(off[0])`, PASSED).
- TZ §2.7 «NaN в A → NaN в resid/offdiag (не тихий мусор)» — теперь выполняется и для
  eigtop (раньше NaN > 0.0 = false ⇒ глушился в 0.0 — assertion в test_degradation
  существовал с b389d5c, но падал; фикс guard — причина 19/19 в финальном логe).

### 2.7 Тесты §2.7 — ОК
- S5b: новая метрика ‖Gagg·v − λv‖/(‖G‖·‖v‖) ≤ 1e-4 — масштабно-инвариантна (делитель
  ‖G‖·‖v‖, не |λ|); замеры max 4.205e-6 (L6 weak) / 2.25e-6 (L23 weak) — запас 24×/44×.
  Обоснование смены /|λ|→/‖G‖: f32-квант данных Jagg32 доминирует на слабых λ — эталон
  «f64-математика на f32-данных» уже даёт 1.35e-2 по старой метрике (замер 19.11,
  в комментарии py:622–627) — честно, не «подогнано под проход».
- d[j]-фикс (py:615): старое `assert d < 1e-3` при d ∈ R^128 — ValueError «truth value
  of an array» на ПЕРВОМ же элементе (тест всегда падал); `d[j]` — значение на argmin —
  корректно. Это реальный баг-фикс, а не полировка.
- Tuning-гейты (test_eigjac_sweep_time_guard): монотонный hang-check-гард
  `4·t(prev) − 0.3 > 4 ⇒ break` — не пропускает валидный C: при ×4-переходе чанков
  (64→256→…) условие 4·t(prev) > 4.3 ⇒ оценка t(next) ≈ 4·t(prev) > 4.3с > гейта;
  при ×16-переходе (1→16) условие слабее, но направление то же (skip только при
  t(1) > 1.025с, тогда t(16) ≥ ~16·t(1) ещё больше). C_measured при skip = последний
  измеренный чанк с t ≤ 4с — именно C по правилу TZ; гейт C_measured == C_FIXED
  принудит перетюнинг при смене состояния GPU. В финальном прогоне: skip на 256
  (4·1.364−0.3 = 5.156 > 4) — artifact без c256/c1024/c4096 — согласуется.
- m-28/curve/zero-тесты НЕ тронуты дельтой (diff по test_new_ops.py — ханки только:
  _gemm_check, sweep_time_guard, gagg, random_spd) — покрытие r2 сохранено.
- test_eigtop_eigbot_random_spd: конструктор переписан — Q·diag(λ)·Qᵀ (QR гаусса ×
  фиксированный спектр: top-8 ratio 0.8, mid 20→18, bot 16·0.8ⁱ) вместо wishart;
  выведение в docstring: wishart даёт ratio 0.987–0.994 ⇒ 512 итераций недостаточно
  даже в f64 (замер subdist 2.6e-3…8.1e-3). Спектры сравниваются ПОСЛЕ np.sort —
  согласовано с TZ §2.2 (λ_j = Rayleigh колонки j, порядок колонок не гарантирован).
  Отделённость: min(top) 26.84 > max(mid) 20 > min(mid) 18 > max(bot) 16 — без
  случайных кластеров.
- test_eigjac_jcache: per-λ классификация «кластерный» (сосед в 10% по спектру) с
  допусками 1e-2/1e-3 — согласовано с артефактом (max relλ = 1.278e-3 — кластерный λ,
  иначе assert 1e-3 упал бы); подпространства/орто/offdiag — как в TZ.

### 2.8 run_local.sh 3b — ОК
- Шаг 3b вставлен между 3 (паритет) и 4 (бенчмарки); шапка-комментарий обновлена.
  `"$PY" -m pytest py/test_new_ops.py -q || echo "!!! new-ops тесты упали"` — не убивает
  скрипт (конвенция шага 1: `|| echo «!!! тесты упали — не идти дальше»`); без
  build/libprism_sycl.so — явный пропуск с подсказкой (сборка — шаг 2); DEVICE_OK-guard
  ВНУТРИ тестов (_skip_nogpu) — без GPU тихий skip, не падение. Остальные шаги не
  затронуты (diff +9 строк: 1 в шапке, 8 в теле).

### 2.9 Артефакты vs лог — ОК
- results/AT0_3/ — ровно 6 файлов (ls), лишнего нет. Все JSON валидны: только
  float()/round-значения, нет numpy-типов/NaN/None (json.dump-ловушка AGENTS.md п.4).
- eigjac_tuning.json: n128_c1=0.312, c16=0.464, c64=1.364 (все ≤ 4с), n256_c4=2.708 (≤ 4с);
  c256+ отсутствуют — монотонный skip (2.7). C_measured = 64 = C_FIXED — гейт C_measured
  == C_FIXED пройден (19/19 в логe).
- eigjac_convergence.json: L6 — curve [0.1256, 0.0244, 0.00383, 0.000374, 2.72e-5,
  1.469e-7], первый < 1e-5 на индексе 5 ⇒ sweeps_to = 6 (арифметика argmax+1 ✓), ≤ 16 ✓;
  L23 — [0.1179, 0.01029, 0.000491, 8.23e-6] ⇒ sweeps_to = 4 ✓. offdiag_final ==
  последний элемент curve в обоих случаях ✓. Wall-time согласован с ранним стопом:
  L6: 6 launch × t(1)=0.312 ≈ 1.87с vs замер 1.854с; L23: 4 × 0.312 ≈ 1.25с vs 1.19с.
  TZ-оценка «4–6 sweep'ов до 1e-5» — валидирована (6 и 4).
- eigjac_jcache.json: max relλ = 1.278e-3 (≤ 1e-2 кластерный), max subdist = 1.093e-2
  (≤ 5e-2), max ortho = 2.28e-6 (≤ 1e-4), max offdiag = 8.23e-6 (< 1e-4) — все 6 (L,d,t).
- gagg_composition.json: max subdist = 2.74e-5 (≤ 1e-3), max s5b_rel = 4.205e-6
  (≤ 1e-4) — 6 групп (2 слоя × top/weak/weakmid).
- svd_top_hdocs.json: max σrel = 8.58e-7 (≤ 1e-3), max subdist = 6.34e-6 (≤ 1e-3);
  зеркальная симметрия точная: mt_n_U_subdist == V_subdist, mt_n_V_subdist == U_subdist
  (L6: 3.793e-6/3.517e-6 перестановка) — cross-check U(Aᵀ)=V(A) выполняется.
- eigtop_eigbot_jcache.json: Ritz-предельные условия по всем 6 (L,d,t):
  ritz_max_bot ≤ λ_r(1+1e-3) — самое близкое: L23_1_0: 3.34496e-7 vs 3.34496e-7
  (избыток 1.59e-6 ≤ 1e-3 ✓); ritz_max_top ≥ λ_r(1−1e-3) — с большим запасом.
- Лог /tmp/AT0_3_suite_final.log: collected 19 = 5 gemm + 8 eigjac + 2 svd + 1 gagg +
  2 eigtop/bot + 1 degradation — ровно тесты файла; 19 passed in 253.61s, RC=0.

### 2.10 AGENTS.md-ловушки
1. Одинаковая статистика на сторонах: gemm — тот же f32-вход → f64-эталон (f32 точно
   представимо в f64); gagg — f32-путь vs f64-эталон НАМЕРЕННО (TZ §2.7: «f64 → f32 как
   в AT2c», сверка с eigh(Gagg f64)) — квант данных задокументирован в S5b-комментарии
   (эталон f64-на-f32-данных = 1.35e-2 — пол данных, не ядра). ОК.
2. Guard'ы протестированы: fudge-guard (a)(b) — test_eigjac_fudge_guard; tot>0 —
   zero-тест (off[0]==0.0); Cholesky/args — degradation (eigtop+eigbot r≥n → PS_ERR_ARG,
   eigtop n=1024,r=8 → PS_ERR_LOCAL, eigjac n=300 → PS_ERR_ARG, gemm f64 → PS_ERR_ARG,
   gemm K=4097 → PS_ERR_ARG); curve-size — zero-тест (b) ValueError. Все PASSED.
   Не покрыт (предсуществующее, не требование TZ-теста): громкий fail eigbot-Cholesky
   на не-SPD входе с ВАЛИДНЫМИ аргс — см. residual risks.
3. Каскад фиксов: сигнатуры C ABI не менялись (eigjac/gemm/eigtop/eigbot) — внешних
   потребителей нет; eigjac_sweep_chunk — один вызов (cpp:1606), gemm_tile — один
   (cpp:1078); dV→dVT — все места (v_init-вызов, sweep, D2H) переименованы.
4. json.dump: см. 2.9. 7. Артефакты vs код: ключи JSON (s5b_rel_max, mt_n_*,
   rel_lambda_max, curve_per_sweep) — ровно текущие тестовые имена. 9. Мелочи:
   F-r3-2/3/4. 5/6/10 — Н/Д для ревью.

## 3. Чек-лист: что читал/проверял
- git: `git status --short`, `git diff HEAD --stat -- sycl_prism/` (+162/−99 по 4 файлам
  суммарно +254/−87), полные диффы 4 файлов (/tmp/r3_{cpp,py,test}.diff + run_local.sh),
  `git show HEAD:b389d5c-state` по test_degradation (NaN-ассерты existed до дельты).
- sycl_prism/src/prism_sycl.cpp: 938–1090 (gemm_tile+gemm_impl целиком), 1260–1320
  (eig_postprocess NaN-guard, eig_impl backstop), 1322–1395 (комментарии v5.3,
  EIGJAC_CHUNK), 1396–1512 (sweep: fudge-guard, float4/скаляр, offdiag), 1605–1655
  (sweep-луп, fill, VT→vecs транспозиция, sort/MGS); grep callers.
- sycl_prism/py/prism_sycl.py: 221–232 (_f32), 418–430 (_contig32), 470–562
  (eigtop/eigbot/_eig_check_args/_eig_call), 660–745 (svd_top целиком).
- sycl_prism/py/test_new_ops.py: 96–245 (_gemm_check, gemm-тесты, _eigjac_jcache_layer),
  332–420 (sweep_time_guard целиком), 423–770 (fudge/zero/m-28/svd/gagg/eig/degradation).
- docs/TZ_sycl_new_ops.md: §2.1 (137–177), §2.2 (178–213), §2.3 (214–296), §2.4 (297–322),
  §2.5 (323–338), §2.7 (356–413); grep C/4 (67, 238–239, 379, 382).
- docs/dynamic/AT0_3_CODE_REVIEW_r2.md — целиком (C5/C6, off-by-one r2); AGENTS.md
  чек-лист (7–49); docs/MARKERS.md (шапка).
- /tmp/AT0_3_suite_final.log — весь файл; results/AT0_3/*.json — все 6 целиком.
- Запущено: numpy 2.5.2 (venv репо) — проверка mask-индексации/стрейдов; arithmetica
  (nch, sweeps_to, wall-times, Ritz-избытки) — пересчёт вручную.

## 4. Residual risks
- Перетюнинг C=64 привязан к состоянию iGPU 19.11 (batt/AC, 1.2–1.4 ГГц): на другой
  машине C_measured может уйти от 64 — гейт C_measured == C_FIXED упадет громко
  (дизайн, не тихое замедление), но потребует пересборки константы.
- TZ §2.3/§2.7 (C/4, C ≥ 256) не обновлены (F-r3-1) — до правки TZ формально противоречит
  коду; будущие ревьюеры, сверяющиеся с TZ, получат ложный finding.
- eigbot: не-SPD вход с валидными аргс (Cholesky-fail → RuntimeError) — поведение
  задокументировано TZ, но designed-теста нет (предсуществующий разрыв, не дельта).
- Монотонный гард (4·t−0.3) — эвристика на линейности t(c)∝c; при аномальной
  сублинейности (кэш-эффекты на очень больших чанках) теоретически можно пропустить
  чанк, проходящий 4с — на практике ×4-переходы дают запас (условие skip: t(prev) > 1.075с).

VERDICT: PASS_WITH_MINORS

SYCL_R3_REVIEW_DONE
AT03R3_REVIEW_DONE
