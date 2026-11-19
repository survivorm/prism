# SYCL-прибор (sycl_prism) — инструкция к API «для себя»

> Статус: **22/22 теста PASSED на iGPU (2026-11-19, RC=0, 235.95с)** — лог
> `/tmp/AT0_3_suite_r4fix.log` (22 = 19 TZ §2.7 + 3 канарейки само-аудита r4;
> интеграция `tests/test_backend.py` 13/13 в sycl и 13/13 в cpu),
> артефакты `results/AT0_3/*.json`.
> Полная спека контрактов: `docs/TZ_sycl_new_ops.md` (v5.4). Короткий README:
> `sycl_prism/README.md`. Этот файл — рабочая шпаргалка: как зовут, что дают,
> где подвох, сколько реально быстро.
>
> **Статус (19.11):** внешнее ревью r3 (код+артефакт) закрыто —
> PASS_WITH_MINORS, 0 MAJOR (docs/dynamic/AT0_3_CODE_REVIEW_r3.md), findings
> закрыты; цифры ниже — из артефактов финального прогона.

---

## 1. Что это

C ABI (`sycl_prism/src/prism_sycl.cpp`, icpx -fsycl) + python-обёртка на
ctypes (`sycl_prism/py/prism_sycl.py`). Входы/выходы — numpy f32/f64;
torch-тензоры конвертируются прозрачно. Устройство: iGPU Intel Iris Xe
(96 EU, Level-Zero), общий RAM с CPU.

**Ключевая идея прибора**: тяжёлые линейно-алгебраические операции prism
(AT2c/E-45) идут на iGPU, тяжёлые по памяти/вырождению — на хосте в f64
(Cholesky в eigbot, MGS в eigjac/eigtop — host-side, lift SVD — host-side).

## 2. Окружение и сборка (один раз)

```bash
source sycl_prism/sycl_env.sh     # ЕДИНСТВЕННЫЙ источник окружения (идемпотентен)
cd sycl_prism && cd build && make # icpx -fsycl → build/libprism_sycl.so
# build/ — gitignore'нут; на свежем checkout сначала: CXX=icpx cmake -B build && cmake --build build
```

`source sycl_env.sh` обязателен — без него карта не видна. Два неочевидных
фикса внутри (ENV-A: `libumf.so.1` из `/opt/intel/oneapi/umf/1.1/lib` в
`LD_LIBRARY_PATH`; ENV-B: `LD_PRELOAD` системной libstdc++ — python-процесс
иначе тянет свой старый GLIBCXX раньше L0-загрузчика → 0 устройств).
Диагностика: `sycl_prism/probe/` + шаги 1 (самопроверка зонда) и 5 (зонд XPU)
`run_local.sh` (устройство видно?).

Проверка без кода:
```bash
python -c "import sys; sys.path.insert(0,'sycl_prism/py'); import prism_sycl as ps; print(ps.backend(), ps.available())"
# → sycl True  (иначе RuntimeError с причиной — PS_ERR_NODEV, не тайный сбой)
```

## 3. Диспетчер для экспериментов (как зовут из prism)

`prism/backend.py` (импортировать по файлу, как field.py — НЕ `import prism`):

```python
import importlib.util
spec = importlib.util.spec_from_file_location("backend", "prism/backend.py")
be = importlib.util.module_from_spec(spec); spec.loader.exec_module(be)
B = be.B                            # готовый инстанс (режим: PRISM_BACKEND=cpu|sycl, дефолт cpu)
C  = B.matmul(A, Bt)                # torch → torch; 3D батч reshaped в 2D внутри
w, V = B.eigh(M)                    # cpu: torch.linalg.eigh (vals f32); sycl: ps.eigjac (vals f64, n≤256!)
U, S, Vh = B.svd_top(A, r)
```

Влияние f64/f32 на vals: sycl-eigh отдаёт vals f64 (cpu — f32) — на 1e-3-совпадение
не влияет (задокументировано в TZ §2.6).

Ограничения диспетчера: `eigh` на sycl — **n ≤ 256** (eigjac), иначе
`BackendError` с подсказкой (torch-fallback «по одному вызову» невозможен по
контракту — эксперимент решает явно). f64-вход → f32 + one-shot warning.
Ошибки `PS_ERR_*` → `BackendError` (операция, форма, причина).
Тест: `tests/test_backend.py` (cpu-режим, без GPU).

## 4. Низкоуровневый API (`import prism_sycl as ps`)

### 4.1 `ps.gemm(A, B) → C [M,N] f32`
- `C = A@B`, row-major, fp32. A [M,K] (M≤65536, K≤4096), B [K,N] (N≤262144).
- **Входы — строго C-contiguous f32** (view/`.T`/f64 → python-RuntimeError с
  причиной; метки PS_ERR_ARG на python-стороне нет — только в C++ на лимитах).
- Hang-check: N-чанкинг ВНУТРИ вызова, каждый launch ≤ 32 GFLOP (TZ §2.1:
  окно ~20с при недогрузе / <~12с под нагрузкой (комм. C++); бюджет держит
  launch ≤5с даже на самом медленном измеренном конфиге; 2083 GFLOP одним
  launch'ем = рантайм 15+с = зависание). Детерминизм: in_order-очередь + фиксированный порядок чанков →
  **бит-в-бит повторный запуск** (test_gemm_determinism: `np.array_equal`).
- Конфиг тайлов зафиксирован тюнингом (128×64 kc16 db; probe/RESULTS_tuning.md).
- **Реальная скорость** (TZ §2.1/§5, RESULTS_tuning.md): sustained @1250 MHz
  N=4096: **168–170 GF/s** (разброс 0.8%); wide 4096×1024×248320 (реальный
  z_full AT2c, 2083 GFLOP): **116.6 GF/s → ≈17.7с** (чистый CPU-GEMM ≈25с / 83
  GF/s — TZ §5; ~29с в TZ §2.1 — z_full-стадия с softmax/KL).

### 4.2 `ps.eigjac(A, max_sweeps=12, tol=1e-5, curve=None) → (vals, vecs, offdiag)`
- Полный спектр симметричного батча. A — `[batch, n, n]` f32 (batch≤65536,
  **n≤256**, 1≤max_sweeps≤64, 1e-6≤tol≤1e-3).
- `vals` — **f64 [batch,n] ПО ВОЗРАСТАНИЮ** (порядок torch).
- `vecs` — f32 [batch,n,n], столбцы — собственные векторы (host-MGS f64,
  write-back f32: ‖VᵀV−I‖ ≈ 3e-7 при n=128).
- `offdiag` — f64 [batch], отн. ‖offdiag(A)‖_F/‖A‖_F на остановке — **ЧЕСТНЫЙ
  отчёт сходимости**; стоп по max_sweeps без tol → не тихий мусор, а честное
  остаточное число. NaN во входе → NaN в offdiag (сигнал, не ноль).
- `curve` — f64 1D C-contiguous, size ≥ max_sweeps: заполняется на месте,
  `curve[s-1]` = offdiag_rel матрицы 0 после sweep s; хвост (от k до конца) = offdiag[0].
- Чанкинг: `EIGJAC_CHUNK = 64` (перетюнинг 19.11: c256 деградировал до 4.3–6.9с
  > гейта 4с; n=256: чанк C/16 = 4). Замер (results/AT0_3/eigjac_tuning.json):
  n128: c1=0.312с, c16=0.464с, c64=1.364с; n256: c4=2.708с — все ≤ 4с гейта.
- Сходимость на реальных jcache (eigjac_convergence.json): L6: 6 sweep'ов до
  1e-5, wall 1.854с, offdiag_final 1.47e-7; L23: 4 sweep'а, 1.19с, 8.2e-6.
- **Точность** (eigjac_jcache.json): rel_lambda_max 6.0e-5..1.3e-3 (f32-вход!),
  top32/bottom32 subdist ≤ 1.6e-3 / 1.1e-2, ortho ≤ 2.3e-6.
- Для одной матрицы: `ps.eigjac(G[None, :, :])` → снять `[0]`.

### 4.3 `ps.eigtop(A, r, max_iter=512)` / `ps.eigbot(A, r, max_iter=512) → (vals, vecs, resid)`
- A — `[batch, n, n]` f32 (n≤1024, **r≤32, r<n**, max_iter≤512).
  Local-memory бюджет (само-аудит r4, m3): `2*n*r*4 + (WG+1)*4 ≤ 65536`
  (WG=64; члена `r²*4` в ядре НЕТ) → **n=1024: r≤7** (PS_ERR_LOCAL, громко).
- `vals` — f64 [batch,r]: Rayleigh λ_j = ⟨v_j, A v_j⟩; **порядок колонок —
  динамика MGS, НЕ гарантируется** (обычно top — по убыванию, bot — по
  возрастанию; тесты сравнивают спектры после сортировки).
- `vecs` — f32 [batch,n,r] (двойной MGS → ортогональность ~eps).
- `resid` — f64 [batch]: подпространственный остаток
  ‖AV − V(VᵀAV)‖_F/‖A‖_F — ЧЕСТНЫЙ отчёт, **не критерий стопа** (стоп = max_iter).
  NaN → NaN.
- **eigbot-специфика**: A должен быть SPD после сдвига. Внутри: host f64
  `L = chol(A + μI)`, μ = 1e-12·‖A‖_F (inverse iteration), L кастится в f32.
  Вырожденный вход → host-Cholesky падает громко («не positive definite») —
  это сигнал данных, не баг. Arg-guard (PS_ERR_ARG) — **ДО** Cholesky.
- Замер (eigtop_eigbot_jcache.json): subdist ≤ 7.7e-6 (top ≤ 1.7e-6, bot ≤ 7.7e-6),
  resid ≤ 3.3e-7.
  Сходимость (TZ §2.2): eigtop 9–130 итераций на jcache.

### 4.4 `ps.svd_top(A, r) → (U f32 [m,r], S f64 [r], Vh f32 [r,n])`
- Top-r SVD через малую сторону: G = A·Aᵀ (m<n) или Aᵀ·A (m≥n) → gemm + eigjac
  + lift на хосте (f64). **m,n ≤ 1024 И min(m,n) ≤ 256** (малая сторона идёт
  в eigjac, его лимит n≤256; большая — в gemm≤1024): A[128,1024] OK,
  A[300,400] — PS_ERR_ARG с явным сообщением (M2, само-аудит r4);
  1 ≤ r ≤ min(m,n).
- **Выходы C-contiguous и без отрицательных шагов** (`_pos_strides`): REVERSE
  порядка torch делается копией, а не view — потребитель `torch.from_numpy`
  отрицательные шаги не принимает (M1, само-аудит r4).
- Конвенция **torch**: S — descending; `Vh = Vᵀ` (не V).
- **Вход — любой f32-расклад** (view/`.T`/f64 OK — это host-композиция, не
  low-level gemm): внутри `_f32(A)` (f64→f32-каст + C-контигуация).
- **Вырожденный ранг: σ=0 ⇒ явные 0-векторы в ОБАХ выходах** (и U, и Vh) —
  собственный вектор при σ=0 произволен, прибор фиксирует нуль.
- Замер (svd_top_hdocs.json, H-доки [128,1024] r=7): σrel 7.1e-7 (L6) / 8.6e-7
  (L23), U/V subdist ≤ 6.34e-6.

### 4.5 Core-операции (были до new-ops, TZ workstate)
| вызов | что |
|---|---|
| `ps.vpath(u, hn)` | ортогонализованный тангенс: (V [T-1,D] f32, keep [T-1] uint8) |
| `ps.dcurve(P, ks, stat="median")` | кривая per-t косинусов → f64 [nk] |
| `ps.acf_moments(x, K)` | моментный ACF, словарь как `prism.field.acf_moments` |
| `ps.f_dc(x)` | f_dc = r/N (инвариантна к перестановке — пересчёту не подлежит) |
| `ps.ra_closure(u, hn)` | (closure_max, du_u_cross_max) |
| `ps.permutation_indices(n, seed, blk=2)` | индексы block-blk перестановки (1:1 с AT0_1_accounting) |
| `ps.permutation_null(x, fdc, seeds, blk=2, K=96)` | перестановочный null: dict {band, e3, e4} — band = mean_{k=9..K}(ρ̄_Tk(k) − f_dc), e3/e4 = ρ̄_Tk(3)/(4) − f_dc; протокол AT0_1_accounting._perm_stat |

### 4.6 Режимы и диагностика
| вызов | что |
|---|---|
| `ps.available()` | bool: устройство видно (или reference-режим) |
| `ps.backend()` | "sycl" / "reference" / "unavailable" |
| `ps.device_info()` | dict: src_rev, name, backend, driver, device_type, workgroup/local mem, fp16/bf16/subgroups, barrier_form |
| `ps.use_reference()` | переключает на numpy-f64 эталон (`PRISM_SYCL_BACKEND=reference`) — единственный режим работы без GPU (для сверки и CI) |

## 5. Ошибки — громкие, никогда тихо

| код | когда |
|---|---|
| `PS_ERR_ARG (−3)` | лимиты (размеры, r, max_iter) — сообщение с формами; NB: не-f32/не-contiguous вход в gemm — python-RuntimeError БЕЗ метки PS_ERR_ARG (проверка на python-стороне) |
| `PS_ERR_LOCAL` | local-memory бюджет ядра (eigtop/eigbot: n=1024 → r≤7) |
| `PS_ERR_NODEV (−1)` | нет устройства — python: RuntimeError «prism_sycl: нет SYCL-устройства — <причина>» (libumf/libstdc++/LD_LIBRARY_PATH) |
| NaN во входе | → NaN в resid/offdiag (сигнал, **не** ноль — eig_postprocess: guard `nrmA != 0.0`) |

Правило: любой «мусор» на выходе — это либо ошибка данных (вырожденный SPD,
f32-дно), либо баг ядра. Тихого прогона не существует.

## 6. Реальные скорости (только из замеров)

| операция | замер | источник |
|---|---|---|
| gemm N=4096 sustained | 168–170 GF/s (0.8%) | RESULTS_tuning.md, TZ §2.1 |
| gemm wide 4096×1024×248320 (z_full, 2083 GFLOP) | 116.6 GF/s ≈ 17.7с (чистый CPU-GEMM ≈25с / 83 GF/s — TZ §5) | TZ §2.1/§5 |
| eigjac n128 (чанк 64) | 1.36с | eigjac_tuning.json |
| eigjac n256 (чанк 4) | 2.71с | eigjac_tuning.json |
| eigjac L6-матрица 128×128 | 6 sweep'ов, 1.85с | eigjac_convergence.json |
| полный сьют 22 теста | 235.95с | AT0_3_suite_r4fix.log |

**Деградирующий режим** (19.11, зафиксирован): iGPU за день замедлился в
1.5–1.9× (c256: 2.88с → 4.3–6.9с). Частоты норм (~1.2–1.4 GHz), AC,
100% batt — не питание. Отсюда и перетюнинг EIGJAC_CHUNK 256→64. Если
гейт 4с снова пробило — понижать чанк (правило TZ §2.7), не латать таймаут.
Политика фиксации (v5.4): EIGJAC_CHUNK=64 — под ХУЖОЕ измеренное состояние;
тест требует только C_measured ≥ C_FIXED. 19.11-день восстановления:
c256 = 3.843с ≤ 4с → C_measured=256 — остаёмся на 64 (стоимость на матрицу
при c64/c256 идентична в замерах, 14.8/15.0 мс). GPU медленнее → C_measured < 64
— громкий fail — понижать константу.

## 7. Подвохи (вычитаны кровью — не повторять)

1. **f32-дно данных ≠ баг ядра.** Квант f32-входа (Jagg32) доминирует на
   слабых λ: эталон «f64-математика на f32-данных» уже даёт с5b(|λ|-норм.)
   1.35e-2 на L6 weak (19.11-замер). Геометрический контракт — subdist (≤1e-3);
   масштабно-инвариантный остаток ‖Gv−λv‖/(‖G‖·‖v‖) ≤ 1e-4 (замеры ≤ 4.205e-6,
   max L6 weak, gagg_composition.json).
   Нормировка /|λ| на слабых λ — НЕГОДНА (раздувание 1/λ^1.5).
2. **NumPy 2.5: маска-индексация возвращает T-view (без копии).**
   `U_r[:, nz]` (маска по строкам колоночного среза) — транспонированный
   view со стрейдами (4, 512); gemm отвернёт. Явный `np.ascontiguousarray(...)`
   перед каждым gemm.
3. **Порядки выходов**: eigjac/eigtop — vals по возрастанию (torch-конвенция
   eigh); svd_top — S descending (torch-конвенция svd). Смешивать нельзя.
4. **σ=0 / вырожденное**: 0-векторы явные (svd_top — оба выхода; eigbot —
   громкий Cholesky). Не гадать, что «произвольный вектор» — ноль.
5. **Hang-check**: не отдавать > 32 GFLOP одним launch'ем (драйвер убивает
   ~20с при недогрузе / <~12с под нагрузкой; бюджет держит launch ≤5с даже на
   самом медленном конфиге). gemm чанкует сам; в экспериментах — следить за form-фактором.
6. **r≤7 при n=1024** (local-memory). n≤256 — eigjac потолок.
7. **Детерминизм — инвариант**: in_order + фиксированный порядок чанков.
   Любой «случайный» разброс по прогонам — баг (test_*_determinism:
   `np.array_equal`, бит-в-бит).
8. **Правки после старта прогона** — только если ветки не взяты (AGENTS.md):
   артефакт ≠ stdout; JSON санитируется (float()/round, None→null).
9. **GPU-тайминги нестабильны при деградации** — перед перетюнингом смотреть
   частоты/температуру (adl), питание (AC/batt), не домысливать «шум».

## 8. Тесты

- **22-тестовый сьют** (TZ §2.7: 19 + 3 канарейки само-аудита r4):
  `python -m pytest py/test_new_ops.py -q`
  (без GPU — автоскип, DEVICE_OK-guard; wide-gemm — самый медленный тест;
  полный прогон ~4 мин). Покрывает: gemm (small/big/wide/oob/determinism), eigjac
  (jcache/determinism/curve/curve_onplace_contract/timing/fudge/zero/convergence/
  random_spd), svd_top (hdocs/degenerate/output_contract), gagg-композиция,
  eigtop/eigbot (random_spd/jcache), degradation, guards_small_n_and_mem (r4).
- **run_local.sh** — шаги: 0 окружение → 1 тесты (pytest: эталон E-45 +
  самопроверка зонда) → 1a синтакс-гейт (компиляция пробы и библиотеки против
  шима + сверка C ABI ↔ ctypes) → 2 сборка SYCL → 3 паритет прибора (numpy vs
  SYCL) → 3b new-ops тесты (pytest py/test_new_ops.py) → 4 бенчмарки → 5 зонд XPU.
- Референсы: **jcache-чекпоинты** `/data/prism_kernels/exp27b/results/exp27b_jcache_L{6,23}.pt`
  (пресчитанные, вне репо; формулы Грамиана — ТОЧНО как AT2c L386-393).
  Тесты сравнивают iGPU-вывод с f64-эталонами + assert'ы-гейты; «сходимость
  чисел» = зелёный сьют, не ручной diff.

## 9. Файлы

```
sycl_prism/
  src/prism_sycl.cpp        # C ABI (gemm/eigjac/eigtop/eigbot + core) — EIGJAC_CHUNK=64
  py/prism_sycl.py          # ctypes-обёртка (этот API)
  py/test_new_ops.py        # 22-тестовый сьют (TZ §2.7 + само-аудит r4)
  py/jcache/  (если есть)   # локальные чекпоинты
  build/                    # make → libprism_sycl.so
  probe/                    # зонды устройства/тюнинг (RESULTS_tuning.md — источник чисел)
  sycl_env.sh               # ЕДИНСТВЕННОЕ окружение
  run_local.sh              # полный локальный прогон
../docs/TZ_sycl_new_ops.md  # контракты v5.4 (полная спека)
../prism/backend.py         # диспетчер B.matmul/B.eigh/B.svd_top
../results/AT0_3/*.json     # артефакты последнего прогона
```
