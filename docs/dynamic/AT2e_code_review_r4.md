# ВНЕШНЕЕ РЕВЬЮ (r4): AT2e код-дифф после PASS-вердикта r5

**Объект:** `experiments/at2/AT2e.py` (sha256 = 2137a7f912e719d0be17f7c5ccf47323012f9384e22cbce16c34c2db05c9efd2 —
**сверено, совпадает** с заявленным), `docs/AT2e_prereg.md` (r5.2), `docs/AT2e_response_r5.md`,
`experiments/kaggle_at2e/build_kernel.py`, `experiments/kaggle_at2e/at2e_t4_canary.py` (+ `at2e_t4_main.py`).

Метод: полное чтение всех 5 файлов (2143 + 951 + 122 + 80 + 2176 строк) + git-diff TZ vs r5.1-коммит (f06b142) +
статический AST-анализ + исполненные юнит-тесты ТОЧНЫХ фрагментов кода (вырезка из файла, не копии):
(T1/T1b/T1c) linear-гейт verify_vstar, (T2) np.median-guard + screen_ok, (T3/T3b) nonlin_probe_1c со счётчиком
fwd, (T4) AST-сверка сигнатур canary-вызовов. Полный прогон AT2e.py НЕ запускался (запрещено ревью).

---

## A. U1(b): κ-флаги — гейт в normal-ветке `verify_vstar` — **PASS**

- **A1 (κ-флаги до ветвления, доступны в normal): PASS.** `kflags_hi` вычисляется на строках 589–592
  (цикл `for k in range(H + 1)` по `hi["q"]/hi["kappa"]`, т.е. по замеру a_hi — уже посчитанному в
  per_amp-цикле, 0 доп. forward), ДО определения `mode` (строка 596) и до ветки
  `if mode == "normal":` (строка 600). В normal-ветке доступен как локал; также дублируется в отчёт
  `out["kappa_flags_hi"]` (строка 593) и `out["linear"]["kappa_flags_hi"]`.
- **A2 (гейт реально бьёт): PASS — проверено исполнением точного фрагмента (строки 589–614).**
  T1: моки L1=0.000, L2=2.000 (без κ-гейта linear был бы True), κ=0, q=1.1e-4 > 0, floor=1e-13 →
  `any(kflags_hi) = True`, **`linear.ok = False`**. T1b (контроль ложного срабатывания): κ = q/2
  (линейный режим, 2κ = q) → `linear.ok = True`. T1c: below_floor-ветка, κ=0 → `kappa_fail = True`,
  `linear.ok = False` (симметрия с normal сохранена). Формула флага совпадает с TZ:
  `abs(2κ − q)/max(q, floor_q) > 0.5` (строка 591).
- **A3 (TZ §3.5 = код): PASS.** TZ строка 239: «r3 (U1): + κ-флаги (|2κ_s−q_s|/max(q_s,floor) ≤ 0.5,
  все s = t..t+H, при a_hi) — ГЕЙТ в normal-режиме (0 доп. forward, κ уже посчитан; симметрия с
  below_floor): K2-окно (k=4,8) не остаётся непроверенным — FOUND не может опираться на нелинейное
  K2-откликовое окно» — дословно соответствует коду 589–592 + 603–607
  (`linear_ok = bool(L1 <= 0.10 and 1.9 <= L2 <= 2.1 and not any(kflags_hi.values()))`).
  L1/L2 (K1-объект, tau=1e-2) — строки 604–605; `lambda_full_report` — отчёт, не гейтится (строка 612)
  — согласовано с V5.
- **A4 (эффект — консервативное сужение): PASS.** `cell_verdict`: `if (not v["verify"]["linear"]["ok"]
  or ...) return "NOT_TESTABLE"` (строка ~1101) — FOUND требует `linear.ok = True`; в normal-режиме
  `linear.ok` теперь конъюнктивен с `not any(kflags_hi.values())`, где kflags покрывают k=0..8,
  включая K2 (k=4,8) → FOUND не может опираться на нелинейное K2-откликовое окно. Направление —
  сужение (больше NOT_TESTABLE), расширение FOUND невозможно. Потребители `kflags_hi` (каскад, AGENTS п.3):
  normal-гейт (607), отчёт verify (593, 609), S10-smoke-чек (`c["verify"]["kappa_flags_hi"] is not None`) —
  все проверены, поле ставится в обоих режимах.

## B. U2: `nonlin_probe_1c(fwd, cell, b0, vstar=None)` — v*-слот — **PASS**

- **B1 (dirs — пары, v* — отдельный слот): PASS.** Строки 963–965: `dirs = [(j, b0[j]) for j in
  range(32)]` + `if vstar is not None: dirs.append(("vstar", np.asarray(vstar, dtype=np.float64)))`.
  Ключи `f"{j}_{rel}"` (строка 973): «0_0.003»…«31_0.01» + «vstar_0.003»/«vstar_0.01» — коллизий нет
  (проверено исполнением: T3b — vstar=None → 64 ключа без «vstar_*»).
- **B2 (+16 ф.-э. = 4 кл. × 1 v* × 2 ампл. × 2 fwd): PASS — проверено исполнением точной функции со
  счётчиком fwd.** T3: с vstar → **132 fwd/клетку** (2 ампл. × 33 dirs × 2 fwd); без vstar (T3b) →
  **128** (2 × 32 × 2). Дельта = 4 fwd/клетку = 2 ампл. × 1 v* × 2 fwd; × 4 клетки (t32, doc ∈ wiki/math)
  = **16 ф.-э.** Основной блок: 4 × 128 = 512. Цикл 966–984: `for rel in PROBE_AMPS_REL: for j, vec in
  dirs: zp = fwd.fwd(+); zm = fwd.fwd(−)` — ровно 2 fwd на (dir, амплитуду).
- **B3 (вызов — только живые клетки): PASS.** `sec_cells` (строки 1697–1699) фильтрует
  `"ascent" in c` — мёртвые клетки (rec_min, eps0≤0, ранний выход run_cell) исключены; `vstar_c =
  np.asarray(c["ascent"]["v"], dtype=np.float64)` (строка 1725) и вызов `nonlin_probe_1c(fwd_c, ctx1c,
  b0_c, vstar_c)` (строка 1726) — только по `sec_cells`. M5-примечание в коде (строка 1724) — resume:
  list из JSON → ndarray. `ctx1c` несёт все ключи, используемые функцией (L/t/hnorm/pwin/z0).
- **B4 (TZ §6 смета 1c): PASS.** TZ строки 647–650: «4 кл. × 2 доп. амплитуды (3e-3, 1e-2) × 32 × 2 =
  **512** … **+ 4 кл. × v* × 2 ампл. × 2 = 16 (r3 U2)** ≈ 0.62 ч»; worst-case строка 652: «≈ **9926**
  ф.-э. (r3: 5686+1856+1856+528, +16 = v* из U2)». Арифметика верифицирована: 512+16=528;
  5686+1856+1856+528 = 9926.

## C. U3: dirs парами (idx, vec) — шаблон по ВСЕМ спискам-фильтрам — **PASS**

Проверенные места и статус («метка по исходному индексу» — инцидент pos_spec r3):

| # | место (строки) | статус |
|---|---|---|
| 1 | `pos_spec`: `ref_dirs16[j]` / `ref_lams_t[j]` / `sibling_refs[j]` (735–759) — один порядок (фрейм shared per (doc,L), B4), без перенумерации; **rstar-выравнивание** (762–766): `ratio_ref[sum(1 for j in range(rstar_j) if ref_lams_t16[j] > 0)]` — индекс конечного ratio после M5-фильтра, совпадает с порядком `ratio_ref` (строится `zip`-ом в том же порядке) | безоп. (V2, проверено T2) |
| 2 | `cext_ref`: `dirs = [(int(j), vec)]` (997–1005), цикл `for j, v in dirs`, метка `dir_idx: j`, `eig: w[j]` (1007–1008) — j = индекс собственного вектора, из которого вектор построен (`JR.T @ u[:, j]`) — **свой** индекс, дедуп `idxs` через `dict.fromkeys` (1005) | безоп. (U3) |
| 3 | `run_cell` B0-батч: `b0[j]` + `b0_labels[j]` (1146–1157) — метки фикс. список 32, B0 нигде не фильтруется/не перенумеруется | безоп. |
| 4 | `ascent` / `secondary_k2`: `v0_idx = argmax` по 32 в порядке B0, `b0[v0_idx]` (528, 849) — оригинальный индекс, без перенумерации | безоп. |
| 5 | `random_reference`: `dirs[j]` и `out[j]` в одном цикле `for j in range(n_random)` (633–649) — порядок идентичен; ndarray без меток | безоп. |
| 6 | `stage_1d`: `b0[j]` / `b0q_p[j]` в одном цикле (817–820), ascent на `b0q_p` тем же порядком | безоп. |
| 7 | `nonlin_probe_1c`: пары (j, vec) + меточный слот "vstar" (963–965) — последнее место, переведённое на пары; коллизий ключей нет (T3/T3b) | безоп. (U2/U3) |
| 8 | `kflags` / `kflags_hi` — dict по k (589–592, 616–619) | безоп. |
| 9 | `Z_hi[k][j]` 2-D (1150–1154), `matrices_from_Z` (цикл по k) | безоп. |
| 10 | `per_dir` в rec.b0 (1245–1252): `{"j": j, "label": b0_labels[j], ...}` — j = оригинальный индекс в том же порядке B0 | безоп. |
| 11 | `gates["G_pos_spec_ref"]` (1775–1780) — читает сохранённые поля, меток-индексов нет | безоп. |

**Вывод: других «метка по исходному индексу» мест нет** — согласуется с аудитом в response_r5 (per_dir /
kflags / sib_refs + nonlin_probe_1c как последнее место).

## D. Гигиена — **PASS**

- **D1 (np.median guard, все мёртвы → no warning, screen_ok=False): PASS — проверено исполнением
  точного фрагмента pos_spec (строки 750–771 + 775–780).** T2 «all-dead (16×0)»: `n_ref_finite=0`,
  `ratio_ref_p10=None`, `rstar=None`, `f_inflation=None`, **RuntimeWarnings=[]** (guard
  `med_ref = float(np.median(ref_lams_t16)) if ref_lams_t16 else 0.0`, строка 770 — np.median([]) не
  вызывается). T2 «empty (n_ref=0, sibling-мёртвая клетка, r3 V1)»: то же, RuntimeWarnings=[]. T2b:
  screen_ok при всех None-рациях = **False** (строки 775–778: конъюнкция с `is not None`).
- **D2 (n_ref_finite в G_pos_spec_ref): PASS.** pos_spec: `res["n_ref_finite"] = n_finite` (строка 757);
  gates-блок: `"n_ref_finite": c["pos_spec"].get("n_ref_finite", 0)` (строка 1777) + `ratios_finite`
  (1778–1780) — согласовано со схемой TZ §13.
- **D3 (G-zero оговорка амплитуды, TZ §3.8): PASS.** TZ строки 316–318: «амплитуда патча — a_hi клетки
  L12 (1e-3·‖h₁₂‖), НЕ ε·‖h₂₃‖». Код строка 1686: `gz = g_zero(fwd_g, 23, 32, b0_g[0], cell_g["a_hi"])`,
  где `cell_g` — клетка wiki L12 t32 → `a_hi = A_HI_REL * ‖h_{12,32}‖` (1683–1685). Совпадает.

## E. T4-режим (новый, не ревьюился ранее) — **PASS**

- **E1 (`AT2E_NO_JCACHE=1` → jcache=None, нигде нет обращения вне guard'а): PASS.**
  S1: 1439–1446 (`no_jcache` → `jc_sha = None` + лог SKIP, assert только в else-ветке);
  загрузка: 1453 `jcache = None if (SMOKE or no_jcache) else load_jcache_l12()`;
  S1b: 1459 `if not SMOKE and jcache is not None:` (assert pool_fp только внутри);
  cext: 1729 `if jcache is None:` → лог SKIP, `elif "cext" not in stage_done:` (1731) —
  `cext_ref(..., jcache, ...)` вызывается только когда jcache загружен;
  gates: 1766 `"S1b_pool_fp": pool_fp_ok` (None в T4).
  Полный grep `jcache` (20 вхождений): 69/225 (константа/функция), 985–993 (тело cext_ref — вызывается
  только из ветки с jcache), 1439–1463/1729–1738 (guard-ветки), 1764 (отчётное поле), 2015 (лог-строка
  canary). **Ни одного `jcache[...]` вне guard'а.** Наблюдение (не дефект): в T4-артефакте
  `gates["S1_sha"]["jcache_l12"]` = False (None == JCACHE_SHA) — инфо-поле, SKIP зафиксирован в логе
  строкой 1442; интерпретировать как сработавший гейт нельзя.
- **E2 (`PRISM_KAGGLE_ROOT` → ROOT в kernel'е): PASS.** Строки 62–65: `ROOT = (_KAGGLE_ROOT if
  _KAGGLE_ROOT else os.path.abspath(...))` — тернарник ленивый: в kernel'е env задан (bootstrap
  строка 27 `setdefault("PRISM_KAGGLE_ROOT", "/kaggle/working")` + envVars-метаданные kaggle_run.sh) →
  else-ветка (с `__file__`) НЕ вычисляется → NameError в notebook невозможен. Локально env не задан →
  корень репо (как было). `RESULTS = ROOT/results/AT2e` (строка 78) → в kernel'е `/kaggle/working/results/AT2e`
  (вытягивается kaggle_run.sh).
- **E3 (`AT2E_SCRIPT_SHA`, условие `"__file__" in globals()`): PASS — оба пути.** Строки 1416–1417
  (main) и 2002–2003 (canary): `script_sha = (sha256_file(os.path.abspath(__file__)) if "__file__" in
  globals() else (os.environ.get("AT2E_SCRIPT_SHA") or "embedded-no-sha"))`.
  (а) Kernel: kaggle_run.sh упаковывает скрипт в **notebook с 1 кодовой ячейкой** (`kernel_type:
  notebook`) — в Jupyter-ячейке `__file__` не определён → env-путь; bootstrap строка 29:
  `AT2E_SCRIPT_SHA = "2137a7f912e7…"` = sha256 локального AT2e.py на момент билда (сверено:
  `sha256sum` файла = 2137a7f9…). Лог-строка 1418 фиксирует «из env (локальный AT2e.py)».
  (б) Локально: `__file__` определён (запуск как скрипт) → `sha256_file(os.path.abspath(__file__))` —
  sha файла на диске. Проверено: `abspath` корректен для относительного пути.
- **E4 (canary_mode — сигнатуры, гейты, клетки, провал): PASS.**
  Сигнатуры (AST-сверка вызовов в canary_mode vs def): `run_cell(9 pos)` vs def 12 pos
  (force_pos_spec/sibling/force_1d = дефолты) — строка 2051; `b0_full(2/2)` — 2039/2063; `probe(6/6)` —
  2066; `vjp_leaf(8/8)` — 2070; `measure_floor(3/3)` — 2043; `baseline(need_hidden=...)` — 2025/2029/2060.
  VRAM-гейт ≤ 12 GB: `ok_vram = bool(vram_peak <= 12.0)` (2075), `vram_peak = torch.cuda.
  max_memory_allocated()/2**30` после клеток (2056), `reset_peak_memory_stats()` (2033) после
  4-doc-baselines → пик за cells **с 4-doc baseline в памяти** (footprint 4×128×248320×4 B ≈ 0.5 GB —
  baselines в dict'ах `fwds`/`baselines`, 2021–2027, до.cells и не освобождаются до конца).
  GPU-дет: двойной forward md5 (2028–2030: `z0b, _ = fwds["wiki"].baseline(need_hidden=False)` vs
  сохранённый md5). Обе клетки L11+L12: цикл `for li, L in enumerate(LAYERS)` (2049) → `run_cell(fwds
  ["wiki"], "wiki", L, 32, li*8, ...)` — cell_id 0 и 8 (канон. wiki-идентификаторы main-сетки).
  `ok_cells = all("ascent" in c and c["eps0"] > 0 for c in cells)` (2077) — полный пайплайн с retain-
  двойным backward (ascent). Провал: 2108–2112 — `AT2E_T4_CANARY_FAIL` + `trackio_finish("interrupted")`
  + `raise SystemExit(1)`; успех — `AT2E_T4_CANARY_DONE`. canary T4-only: `assert DEVICE == "cuda"`
  (2001). Trackio в kernel: `AT2E_TRACKIO=0` (bootstrap 28) → `trackio_start` no-op (137–139), atexit
  no-op. S1 в canary: pool sha assert + fla guard (2011–2015), jcache не загружается вовсе.
- **E5 (build_kernel.py): PASS.**
  (1) Порядок: `out = bootstrap + src` (строка 66) — bootstrap (pip-pin transformers==5.15.1 /
  tokenizers==0.22.2, snapshot_download модели, копирование pool из dataset, env, argv) ПЕРЕД кодом;
  env читается кодом при импорте/исполнении того же cell — после bootstrap-утверждений. pip ДО
  `import transformers` (импорт в коде на строке ~50).
  (2) ASSERT встроенного текста: строка 69 `assert back[len(bootstrap):] == src` + строка 71
  `assert f'"AT2E_SCRIPT_SHA"] = "{sha}"' in back`. Сверено с дисками: `at2e_t4_canary.py` и
  `at2e_t4_main.py` содержат ПОБайтово локальный AT2e.py (шапка после bootstrap-маркера + 1 пустая
  строка; body → sha256 2137a7f9…, body==src: True для обоих кейлов).
  (3) sys.argv: canary `["at2e_t4", "--canary"]` (строка 30 кейла), main `["at2e_t4", "main"]` —
  main-ветка кода: `"--reanalysis" in sys.argv` = F, `"--canary" in sys.argv` = F, `"--resume" in
  sys.argv` = F → полный main-путь, аргумент "main" не читается.
  Провенанс-константы: pool sha в builder = sha256 локального файла (сверено: ded634f7… = константа
  POOL_SHA; локальный файл на диске даёт тот же sha), jcache sha 976079ce… = JCACHE_SHA (локальный
  файл — тот же). pin transformers==5.15.1 = локальная версия (проверено: 5.15.1).
- **E6 (T4 не меняет CPU-путь и правило решения): PASS.** Все T4-ветки — новые и задраны
  (а) env, не заданным локально: `AT2E_NO_JCACHE` (1439), `PRISM_KAGGLE_ROOT` (62), `AT2E_SCRIPT_SHA`
  (1416/2002 — локально `__file__`-ветка), `AT2E_TRACKIO` (137), `AT2E_MODEL_DIR/POOL/JCACHE`
  (65–69 — дефолты прежние); (б) `--canary` (1436–1438) — `canary_mode` недостижим локально без
  флага. Локальный main идёт по прежним веткам: sha-гейт jcache (else-ветка 1444–1446), jcache
  загружается (1453), S1b работает (1459–1463), cext_ref работает (1731–1741). Функции правила
  решения — `cell_verdict` / `global_verdict` / `stage1b_verdict` — чистые функции от сохранённых
  метрик: ни env, ни device, ни T4-состояний не содержат. Единственное изменение правила решения в
  диффе — U1(b) (A) — это принятый патч-лист r5 (консервативное сужение FOUND в normal-режиме,
  рекомендованный ревью-вариант (б)), не T4-инфраструктура.

## F. Согласованность TZ ↔ код — **REVISE** (1 минор: устаревшие хвосты worst-case в TZ r5.2)

- **F1 (§3.5 V5 + U1(b) = код): PASS** — см. A3 (TZ 239–241 = код 589–614; L1/L2 на K1-объекте —
  TZ 235–237 = код 604–605; below_floor — TZ 243–247 = код 615–628; fallback-правило → NOT_TESTABLE —
  TZ 230–233 = код 1160–1166 + `fallback_grid_triggered` в cell_verdict).
- **F2 (§5 = §3.11.5, NOT_TESTABLE, ≥8): PASS.** §3.11.5 (TZ ~417–421): «вердикт клетки — NOT_TESTABLE
  (одна метка для этого случая в §3.11 и §5, r5 гигиена; клетка НЕ считается в "denom+companion OK ≥ 8"
  для NOT_FOUND_IN_SUBSPACE)» = §5 (TZ 461–465: «…§3.11.5 — вердикт клетки NOT_TESTABLE (единая метка
  §3.11.5/§5, r5-гигиена; r3: синхронизировано — формулировка "сохраняет стадию-1-метку" убрана), клетка
  НЕ считается в ≥8 для NOT_FOUND_IN_SUBSPACE»). Формулировка «сохраняет стадию-1-метку» действительно
  убрана (git-diff f06b142→HEAD). Код: `cell_verdict` — `or (v.get("stage1d") and v["stage1d"]["test_guard"])`
  → NOT_TESTABLE; `global_verdict` — `n_ok8 = sum(1 for c in cells if ... and c["verdict"] !=
  "NOT_TESTABLE")` — тест-страж-клетки исключается из ≥8. Согласовано.
- **F3 (§6 смета = код): PASS.** Строка стадии 1: 4+1024+1024+832+64+2048+208+128+34+256+32+32 =
  **5686** (TZ 633; T4-вариант 5558 = 5686−128 cext — TZ 633, код: cext SKIP при jcache=None);
  1d: 16×(64+52) = **1856** (TZ 636); 1b: 1248+464+144 = **1856** (TZ 640–642; 144 = 4 кл. × (2 dir × 2 fwd
  + 16 ref × 2 fwd) — совпадает с ценой pos_spec в коде: v*+v0 в t' 4 fwd + 16 пробы × 2 = 32);
  1c: 512+16 = **528** (TZ 647–650, код — см. B2); worst: 5686+1856+1856+528 = **9926** (TZ 652).
  Наблюдения (пред-существующие, из r5.1, не блокеры): (i) финальный FD-only-проход ascent
  (2 fwd/клетку, ≤32 ф.-э.) и null floor-пре-пасс (16 fwd) не в таблице — завышение точности сметы,
  не смена правила; (ii) TZ-формула null pos-spec «4 × (2·2 + 16·2) × 2 = 144» содержит лишнее «× 2»
  (4×36×2 = 288 ≠ 144) — число 144 верно, опечатка в записи формулы (строка 642, была в r5.1).
- **F4 (внутренняя консистентность TZ r5.2 — НОВЫЙ наход): REVISE.** В TZ r5.2 остались два
  устаревших хвоста worst-case **9910/«9.91k»** (9910 = 5686+1856+1856+**512** — т.е. стадия 1 c
  новым 5686, но 1c **без** U2-дополнения +16; актуальное worst — 9926):
  (а) шапка, строка ~79: «…worst case ≈ **9.91k** ф.-э. ≈ 11.9 ч» → заменить на «≈ **9926** ф.-э.»;
  (б) §6b, строка 590: «run_one.sh (смета §6: **5686–9910** ф.-э. ≈ 6.85–11.9 ч; …)» →
  «**5686–9926** ф.-э.».
  Canonical-строка §6 (652) и ответ-файл (§2: «9926») верны; в response_r5 §1(U2) «worst-case
  9910→9926» — «9910» — тот же устаревший промежуточный (актуальное "до": 9894 из TZ r5.1; ответка
  разовая, docs-hygiene — правка необязательна). Правило решения и числа-гейты не затронуты — чистая
  текст-синхронизация, правка = TZ-sub-revision (r5.2 → r5.2.1) по п.10 AGENTS, код не меняется.

---

## Итог

| пункт | вердикт |
|---|---|
| A. U1(b) — κ-гейт в normal | PASS |
| B. U2 — v*-слот в nonlin_probe_1c (+16 ф.-э.) | PASS |
| C. U3 — dirs парами, все списки-фильтры | PASS |
| D. Гигиена (median-guard / n_ref_finite / G-zero) | PASS |
| E. T4-режим (NO_JCACHE / ROOT / SCRIPT_SHA / canary / builder / CPU-путь) | PASS |
| F. TZ ↔ код (§3.5, §5=§3.11.5, §6 смета) | REVISE (F4: 2 строки хвостов 9910/9.91k → 9926) |

Код (2137a7f9) — все 4 пункта патч-листа r5 + гигиена применены корректно; T4-инфраструктура не
касается CPU-пути и правила решения (только новые задраные ветки + canary); ключевые утверждения
response_r5 подтверждены исполнением точных фрагментов, не чтением. Единственное замечание —
текстовое (TZ F4), не кодовое.

VERDICT: REVISE

REVIEW_AT2E_CODE_R4_DONE
