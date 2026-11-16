# AT0_5 ARTIFACT REVIEW — раунд r5 (пост-прогон, внешнее ревью)

Дата: 2026-11-15. Ревью по файлам на диске (чтение + лёгкие python-проверки чисел).
Прогон: run `at05_gencheck_r1`, `RUN_RC=0`, gates `ALL PASS`, verdict `branch2`.

## 0. sha256sum ревьюируемых файлов

| файл | sha256 |
|---|---|
| experiments/at0/AT0_5_gencheck.py | `c0c3dd4267dc37027e397cd9085868996d2aa17f3a5843b9c3dd90d9799cfaa0` |
| docs/AT0_5_prereg.md | `986f51b74cf63bc2ddd6ae8abaaa20cc6d36c6bd15ebd2254b8a991977646ef0` |
| results/AT0/AT0_5_GENCHECK.json (claim) | `173c62adf9252f81f2fc65b54c5a521e4c328f6e6c2426e47b32823a1a866a5d` |
| results/AT0/store/4732f98c6291fc8c7c41c3d47407eb0fcce1c009420068b9f83be78341952fcb.json (store) | `4732f98c6291fc8c7c41c3d47407eb0fcce1c009420068b9f83be78341952fcb` |
| results/AT0/AT0_5_run.log | `7e016b4d13f67e79778bbb7333f5feb2e6bb28582795522798bb1c864b597db6` |
| results/AT0/AT0_5_manifest.json | `185cb2dc65971875773662717811f7200c6ebe0c2e31203f2163d059ee0b67eb` |
| /tmp/at05_r1.log (лог прогона) | `b1eca1af73642d6176bf6ae1f177c0f11ae87c11eda078da96650d64bb4f15f8` |
| docs/AT0_5_review.md (входное r4.1) | `f48cb29f55ef62a1e6c192cd952bce574ed7ec8955be86a14250d34353b4e427` |

Сцепка хешей: `claim.store_sha256 == sha256(store)` ✓; `manifest.claim_sha256 == sha256(claim)` ✓;
`manifest.store_sha256 == sha256(store)` ✓ (пересчитано hashlib-проверкой).

## 1. Чек-лист

### 1.1 Одинаковая статистика на обеих сторонах — ПРОВЕРЕНО
- `S0` гейт (experiments/at0/AT0_5_gencheck.py:802-814): exec-сравнение `field.acf_moments` vs
  `rho_metrics` на 2 матрицах, assert `dev ≤ 1e-14`. В артефакте: `gates.S0.max_dev = 0.0`, ok (claim).
  Лог: `S0 ok` (/tmp/at05_r1.log:18). Один прибор подтверждён гейтом, а не утверждением.
- `sim_mats` реально вызывает `rho_metrics` (нет тихой дубликации формул). Каскад-баг разупаковки
  (3→2 элемента возврата после удаления `fdcs`): **все** call-sites перепроверены grep'ом —
  только 2-элементные разупаковки (`mats, _ = sim_mats(...)` в null_cell ~:718, `ma/mb = sim_mats(...)[0]`
  в calibration ~:760-761, `gm, gagg` в ladder-части). Трехэлементных разупаковок и потребителей
  `fdcs` больше нет.

### 1.2 Ветки и guard'ы — ПРОВЕРЕНО (ветки живые, exercised в прогоне)
- `dcma1_fit` / `dcma1_fit_tail`: ветки `f_le_0` / `out_of_model` / GN — живы; exercised:
  `n_f_le_0` = 49/49/50/50/50, `n_out_of_model` = 1/1/0/0/0 (per layer, store).
- `build_paths`: K=0 skip и k<16 fail-loud — exercised: claim.meta.skipped = [own_w0 L1, own_w4 L1],
  причина `T0.1 K=0 (fp16_l1_degraded provenance)`; лог: `skip 2`, `478 клеток из 480`.
- `var_ladder`: `bic None` (sign ≤ 0 из slogdet) → fallback в selection (experiments/at0/AT0_5_gencheck.py:599-600).
- `coverage`: None-metrics для коротких клеток (plat_lo/plat_hi = None при k < границы) — обрабатываются
  как non-covered; `n_short_k_lt_33 = 0` в G8 (claim).
- `oos_fit`: пер-cell `n` в нормировке `(1−k/n)` (experiments/at0/AT0_5_gencheck.py:856-888, docstring + код).
- `fit_scale`: M=0/E=0 edge — см. M1 ниже (флаги `dc_dir`).
- `fullcov_sim`: chol + SVD-fallback (experiments/at0/AT0_5_gencheck.py:503-525); exercised:
  `how: {'chol': 255, 'svd': 0}` (лог :31).
- trackio atexit-страховка: experiments/at0/AT0_5_gencheck.py:982-984 (`import atexit` + finish при краше);
  exercised: лог `trackio: finish ok (status=ok, verdict=branch2)`.

### 1.3 Каскад фиксов — ПРОВЕРЕНО
- Старые имена `dcma1_sim_d` / `dcma1_sim` (без `_fit`): grep — 0 вхождений в коде (только
  упоминания в комментариях/docstring-истории). `sum96`: 0 вхождений (переименовано в `sumK`,
  :321, :564, :657).
- Все потребители `fit["f"]/fit["a"]` получают DC-компоненту через `fit_scale`: part1 sim,
  `null_cell` (:710-716), `fullcov_sim` (:519), calibration (:750-759) — единый контракт
  «total power сохранён, DC-доля = fitted f» (fit_scale, :467-489).

### 1.4 json.dump / артефакты — ПРОВЕРЕНО
- claim и store валидный JSON, `NaN/Infinity` = 0 в обоих (grep-подсчёт).
- Сцепка хешей — п.0 ✓.

### 1.5 Калибровки читаются первыми — ПРОВЕРЕНО
- S2: mean = 0.000593, target = 0, 0.52 SE (лог :18; claim: 0.0005929285665609565) — null-контроль ноль ✓.
- S4: r_even384 = 0.0, r_odd383 = 2.61e-03 = 1/N ✓ (лог :19; claim S4).
- S9: 478 клеток, max_dev = 0, `k_mismatch=[]`, `n_mismatch=[]`, absent_T01=[] (лог :20; claim S9,
  store-пин 23281c64… ✓).
- G8 (а): median|tail_excess|/sd_emp = 1.04 ≤ 3.0 (claim: 1.035542240671972). Это РЕЗУЛЬТАТ, не ноль:
  эффект-сайз ~1.04 sd_emp по определению нормальный для flat null (excess без структуры ~ N(0,1)
  в sd_emp-масштабе) — не сигнал, не «шум, который можно игнорировать». G8 (б): 1.00 / 0.974 / 0.974
  ≥ 0.90 ✓.
- **Ключевой claim перепроверен** (детали — п.4, раздел «between-doc артефакт»): per-doc Δh
  (h_raw) плато ≈ 0 и f_dc ≈ 0; число 0.06 — уровень DC объекта v_tangent, не between-doc-пулинг Δh.
  Формулировка «T0.1 пул-плато 0.06 = between-doc артефакт» верна по существу с уточнением
  (артефакт выбора объекта, не пулинга по докам Δh).

### 1.6 Числа артефакта vs лог — ПРОВЕРЕНО (spot-check 12 чисел, все совпадают)
| число | лог (/tmp/at05_r1.log) | артефакт | |
|---|---|---|---|
| S2 mean | 0.000593 (:18) | 0.0005929285665609565 (claim) | ✓ |
| S9 max_dev | 0.00e+00 (:20) | 0.0 (claim) | ✓ |
| G8 (а) ratio | 1.04 (:22) | 1.035542240671972 (claim) | ✓ (округление) |
| G8 (б) fracs | 1.00/0.97/0.97 (:22) | 1.0/0.97448/0.97… (claim) | ✓ |
| per-layer frac (5 слоёв) | 0.157/0.216/0.255/0.176/0.078 (8/11/13/9/4 из 51) (:23) | store per_layer — то же | ✓ |
| CI95 (5 слоёв) | [0.082,0.280]…[0.031,0.185] (:23) | store — то же; независимый Wilson-пересчёт: [0.0817,0.2801]…[0.0309,0.1850] | ✓ |
| src_cov L1 | P1_own 0.0 / P2 0.214 / gen 0.139 (:24) | store frac_covered_by_source — то же | ✓ |
| d_eff diag/full | 336/38, 277/37, 322/40, 510/43, 635/25 (:24) | store d_eff_median/d_eff_full_median — то же | ✓ |
| OOS | 0.412/0.588/0.608/0.510/0.373 (:26) | store frac_covered_oos — то же | ✓ |
| selected_dist | 190/34/254 (сумма 478) (:25) | store ladder — то же | ✓ |
| AR(1) beta/R2 | 0.275 / 0.076 (:27) | store 0.27458 / 0.07618 | ✓ |
| calibration + fullcov | marg 0.950/0.945/0.955 joint 0.855 (L1); full-cov 0.627/0.725/0.784/0.804/0.725 (:29-31) | store calibration/fullcov_calibration — то же | ✓ |
- wall_sec: claim 46857.5 с; лог-старт 2026-11-15T07:29:11 → ожидаемый конец 20:30:09 (13.0 ч) —
  согласуется с trackio-завершением (trackio DB gitignored, пересчитывать не требовалось).
- run.log (артефакт) vs /tmp/at05_r1.log: идентичны, кроме trackio-wrapper-строк (баннер trackio,
  `field selfcheck: OK …` — print из field._selfcheck, строки `RUN_RC=0` и trackio-finish).
  Числовые строки — бит-в-бит.

### 1.7 Протокол — ПРОВЕРЕНО
- `RUN_RC=0` в конце /tmp/at05_r1.log ✓; `AT0_5 gates: ALL PASS` в конце run.log ✓;
  gates в claim все `ok: true` (G1′, G0, S0, S2–S9, S6, S7, G8_null, G2) ✓.
- Хеш-цепь claim→store→manifest — п.0 ✓.

### 1.8 Вердикт-оверкейм — ПРОВЕРЕНО (с пометкой о null-зависимости, см. п.3)
- Текст вердикта (claim.prereg.verdict): «branch2: fitted DC+MA(1) per-doc НЕ покрывает дип+плато
  на всех слоях (below: [1, 5, 11, 17, 23]; empty primary-пул: []) ⇒ структура сверх DC+MA(1) есть,
  T3 получает работу (in-sample GOF; generative-claim не делается)».
  Нет слов «generative», «out-of-sample», «воспроизводит» без оговорки — формулировка в пределах
  in-sample GOF ✓.
- Per-layer fracs — из eligible = non-degen ∩ K96 (51/layer; код :1268-1270, пул-строка лога :24) ✓.
- **Устойчивость к выбору null — см. раздел 4**: branch1 не поддерживается НИ одним из null
  (primary и full-cov) — отрицательный результат robust; сам лейбл branch2 (vs inconclusive)
  null-зависим — вердикт вынесен по registered primary null, как и pre-рег правило.

### 1.9 Прочее — ПРОВЕРЕНО
- Seed-формулы == prereg (docs/AT0_5_prereg.md:281-283):
  cal = `SEED_CAL_{A,B} + 5·idx + li` (код :757-759); null sim = `SEEDN + NSEED·(5·idx+li) + seed_i`
  (код :716); perm = `1000000 + 25·idx + 5·li + seed_i` (код :699). Часть 1/2: 100000/200000 +
  5·idx + li (заголовок :44, :74) — совпадает с prereg :42, :72.
- Мёртвого кода, дедупликационных и тайп-несовместимостей при проходе не найдено
  (см. также п.1.3).

## 2. Обработка входного ревью r4.1 (docs/AT0_5_review.md)

### 2.1 r4.1 §2 — must-fix (новые C1–C3)

| пункт | статус | доказательство (file:line) |
|---|---|---|
| **C1** — CI для frac_covered + source-coverage + 3-х веточный вердикт | **закрыт** | `wilson_ci` (gencheck.py:527-533); per_layer: `frac_covered_ci95` (:1294), `frac_covered_by_source` (:1295, расчёт :1277-1283); вердикт: CI lo>0.5 → above / hi<0.5 → below / содержит 0.5 → **inconclusive** / пустой пул → branch2 (:1307-1343). Exercised: в логе каждый слой «CI95[…] below» (:23) — ветка inconclusive живая (условие проверяется на каждом слое). |
| **C2** — d_eff_full + калибровка real coverage (хотя бы один из 3 вариантов) | **закрыт** | `d_eff_full = (trΣ)²/‖Σ‖_F²` (gencheck.py:1059-1063), отчёт diag/full per layer (:1285-1286) и в калибровке (:770-771); выбран вариант 2 — **covariance-matched Gaussian null**: `fullcov_sim` (L·Lᵀ = Σ_c, :503-525) + reportable `fullcov_calibration` (артефакт: 0.627–0.804) + `fullcov_note` (не-гауссовость измеряет G8-v3 эмпирический пул). |
| **C3.1** — seed-коллизии calibration | **закрыт** | `SEED_CAL_A + 5·idx_of[nm] + li` / `SEED_CAL_B + 5·idx_of[nm] + li` (gencheck.py:757-759), комментарий про старые коллизии (:754). Уникальность: 5·idx+li инъективно на idx 0..95, li 0..4. |
| **C3.2** — null seeds без layer | **закрыт** | null sim: `SEEDN_OFF + NSEED·(5·idx + li) + seed_i` (gencheck.py:716); perm: `SEED_PERM_BASE + 25·idx + 5·li + seed_i` (gencheck.py:699, `SEED_PERM_BASE = 1000000` :199); декларируется «НЕ paired design» (docstring :678-679). Формулы совпадают с prereg addendum r5 (:281-283). |

### 2.2 r4.1 §3 — should-fix (M1–M5)

| пункт | статус | доказательство |
|---|---|---|
| **M1** — fit_scale M=0, f>0 | **закрыт** | gencheck.py:477-486: `M>0` → mvec-scale; `M=0, f>0` → направление `svec/‖svec‖` (ось макс. дисперсии) + флаг `dc_dir`; `E=0` → «degenerate». Контракт «DC-доля = fitted f» не нарушается. В прогоне: dc_dir = 433× «none (f=0)» + 45× «mvec» — M=0-ветка не срабатывала (все real mvec ≠ 0), но ветка живая и объявлена. |
| **M2** — имя sum96 врёт для short cells | **закрыт** | `sumK` (gencheck.py:321, :564, :657); grep `sum96` — 0 вхождений. |
| **M3** — gate_s5/s8 глобальный N | **закрыт** | gate_s5: `n = dh.shape[0]; k = min(K, n-1)` (gencheck.py:871-872); gate_s8: `field.acf_moments(dh, n - 1)` (gencheck.py:903, :907). |
| **M4** — burn-in в VAR/AR1 sim | **закрыт (declare)** | r4.1 разрешал «добавить burn-in ИЛИ declare». Declare: gencheck.py:1386 — «no burn-in (M4): VAR нач. нули (первые p шагов — нач.-условный артефакт, ≤ p…)» в reportable-ноуте части 2. |
| **M5** — null-лестница на real scores? | **закрыт** | null_cell: `ladder = var_ladder({r: scores[r][perm]})` (gencheck.py:~727) — рядоки РЕАЛЬНЫХ PCA-проекций, переставленные тем же blk-perm; docstring :679-684 объясняет корректность null (порядок времени разорван ⇒ временные корреляции уничтожены; базис r-проекций фиксирован). Real scores без перестановки нигде не используются. |

### 2.3 r4.1 §1 — прежние C1–C7 (закрыты в r4; re-check в r5-коде)

| прежний пункт | статус | доказательство |
|---|---|---|
| **C1** S2 target 0 | **закрыт** | gate_s2: `target = 0.0` + docstring про demeaned-артефакт (gencheck.py:795-819); прогон: mean=0.000593 ok. |
| **C2** D-null из fitted-DC | **закрыт** | `dcma1_sim_d_fit` + `fit_scale` (gencheck.py:467-501); циркулярная версия (real m/s as-is) удалена. |
| **C3** in-sample GOF wording | **закрыт** | текст вердикта (gencheck.py:1330-1343): «in-sample GOF; generative-claim не делается»; OOS — только `plat_hi_oos` (oos_fit docstring: дип не OOS, циркулярен). |
| **C5** K-правило / short cells | **закрыт** | `k = min(96, n−1)`, T0.1 K-алфавит {0,24,32,96}, S9 K/N-сверка (claim: k_mismatch/n_mismatch пустые), primary = non-degen ∩ K96, short — sensitivity, T0.1 K=0 skip (claim.meta.skipped). |
| **C6** VAR BIC full df | **закрыт** | gencheck.py:599-600: `bic = n·logdet(Σ) + (p·r²+r)·ln n`, Σ = EᵀE/n (slogdet, full df); `bic_tz` — sensitivity (:585). |
| **C7** gates/RC | **закрыт** | `ok is True` assert-и; при FAIL store+log пишутся, claim/manifest нет, RUN_RC=1 (main-блок). Прогон: `RUN_RC=0`, `AT0_5 gates: ALL PASS`. |
| S4 docstring odd/even | **закрыт** | gate_s4 (gencheck.py:838-849): r_even=0, r_odd=1/N, прогон 0.0 / 2.61e-03. |

## 3. Замечания

**CRITICAL:** нет.

**MAJOR:** нет.

**MINOR:**
1. **MINOR-1** — текст вердикта (claim.prereg.verdict) не несёт явной оговорки, что покрытие
   null-зависимо: primary (registered) null даёт 0.078–0.255 (branch2), full-cov null —
   0.627–0.804 (под ним вердикт был бы «inconclusive»: L1 CI95 [0.477, 0.754] содержит 0.5).
   Числа full-cov в артефакте (`fullcov_calibration` + `fullcov_note`), вердикт вынесен по
   registered primary null согласно pre-рег правилу — т.е. процедурно корректно; для paper/ TZ
   рекомендуется одна строка: «coverage null-зависим (primary 0.08–0.26 vs full-cov 0.63–0.80);
   branch2 — по registered primary null; branch1 не поддерживается ни одним из null».
   Точка: экспериментальный файл править не требуется (фикс опционален — только формулировка).
2. **MINOR-2** — joint coverage null-on-null 0.75–0.87 < 0.95: объяснено корреляцией трёх метрик
   (calibration_note: «joint — эмпирический (метрики коррелированы)») и не является требованием;
   отмечено для полноты — при чтении артефакта не интерпретировать joint как откалиброванную
   95%-покрытость.
3. **MINOR-3** — wall_sec (claim: 46857.5 с) закреплён только в claim/trackio; в claim.meta нет
   `started_at/finished_at` (они — в trackio config, gitignored). Провенанс времени — trackio +
   timestamp первой строки лога (2026-11-15T07:29:11); согласуется, но не дублируется в артефакте.

## 4. Научная интерпретация branch2

### 4.1 Between-doc артефакт: перепроверка ключевых чисел

Пересчёт по T0.1 store (23281c64…) и AT0.5 store (4732f98c…), non-degen, K96, 51 док/слой:

- **Per-doc Δh (h_raw), объект AT0_5**: плато k=33..96 — медианы по докам
  L1 −0.000013, L5 +0.000017, L11 −0.000007, L17 −0.000016, L23 +0.000051; **максимум по всем
  докам 0.00284** (top-доки: chat_5, wiki_13/own_w13, code_11, math_1/4). Per-doc f_dc = r/N —
  медианы 2e-6…1.6e-5. AT0.5 real — бит-в-бит те же числа (S9 max_dev = 0).
- **f_le_0 = 49/49/50/50/50** per layer: per-doc DC в Δh практически отсутствует ✓.
- **Число 0.06**: это DC-уровень объекта **v_tangent** — f_DC = r_v/N = 24.902/383 = 0.0650 (пул)
  и 21.846/383 = 0.0570 (P2), TZ_add_0-3.md:216; per-doc v: плато 0.059–0.105, f_dc(v) = 0.075–0.129.
  Для Δh пул-уровень ≈ 0 (per-doc медианы ≈ 0 ⇒ пул ≈ 0).

**Вывод:** «T0.1 пул-плато 0.06 = between-doc артефакт» — подтверждено по существу с уточнением:
0.06 — артефакт **выбора объекта** (v_tangent несёт per-doc DC, Δh — нет), а не эффект пулинга
по докам для Δh. Для AT0_5 (моделирует Δh) fitted-DC null с f ≈ 0 согласован с per-doc данными —
null не смещён.

### 4.2 Устойчивость вердикта к выбору null

| null (покрытие 3 метрик, primary-пул) | L1 | L5 | L11 | L17 | L23 | вердикт по pre-рег правилу |
|---|---|---|---|---|---|---|
| **primary** (independent D, fitted DC+MA(1); registered) | 0.157 | 0.216 | 0.255 | 0.176 | 0.078 | **branch2** (все below, CI hi < 0.5) |
| **full-cov** (Gaussian, L·Lᵀ = Σ_c; reportable) | 0.627 | 0.725 | 0.784 | 0.804 | 0.725 | **inconclusive** (L1: CI95 [0.477, 0.754] содержит 0.5; остальные above) |

- **Robust**: отрицательный результат — branch1 («DC+MA(1) покрывает ≥50% на всех слоях с CI lo > 0.5»)
  **не поддерживается ни одним из двух null**. Ни primary, ни full-cov не дают all-above.
- **Null-зависим**: сам лейбл «branch2 vs inconclusive» зависит от null: при full-cov null
  4 из 5 слоёв above, L1 в неопределённости. Причина расхождения — cross-dim ковариация:
  d_eff_full = 14–43 против d_eff_diag = 223–635 (strong cross-dim, per layer в артефакте);
  independent-D null не несёт cross-dim структуры, и именно она вносит основную долю
  «непокрытия». Иными словами, часть «структуры сверх DC+MA(1)» — это **cross-dim ковариация Δh**,
  а не автокорреляционная форма (AUC-форма: dip покрывается почти во всех клетках — не-покрытие
  сконцентрировано в plat_lo/plat_hi, ~30–37 и ~23–33 из 51, в обе стороны, max |excess| ≤ 0.0067).
- **Оговорка** (MINOR-1): вердикт в артефакте вынесен по registered primary null, как и pre-рег
  правило; оговорка о null-зависимости в текст вердикта не включена — для paper/TZ добавить строку
  из MINOR-1.

### 4.3 Что остаётся T3

1. **Cross-dim структура**: full-cov null закрывает 0.63–0.80 покрытия — остаток сверх
   covariance-matched Gaussian (не-гауссовость + остаток по форме) — предмет T3: измерить,
   сколько остаётся при полном Σ (G8-v3 эмпирический пул — первый замер: sd_emp 1.6–5.5× выше
   sim-шума, т.е. не-гауссовость существенна).
2. **Горб (bump)**: T0.1 — объект-зависимый excess ±0.012–0.014; AT0_5 dip/plat-метрики горб
   не изолируют (k=3,4 внутри plat_lo) — T3 получает точечный тест горба (per T0.1 G8-блок).
3. **v-объект**: AT0_5 моделирует Δh; per-doc DC объекта v_tangent (0.057–0.129) не проверяется
   DC+MA(1)-гипотезой — если T3 выходит на v, нужен отдельный fitted-DC null (феномен 0.06 живёт там).
4. **OOS**: plat_hi_oos 0.373–0.608 (reportable) — in-sample GOF не переносится на held-out
   формально; это ожидаемо (a из ρ̂(1) — циркулярность дипа) и уже объявлено в вердикте.

## 5. Итог

Все критические пункты чисты: статистика одна на обеих сторонах (S0), ветки живы и exercised,
каскад-баги отсутствуют, артефакты валидны и хеш-сцеплены, лог == артефакт, протокол (RUN_RC=0,
ALL PASS) соблюдён, все must-fix r4.1 (C1–C3) и should-fix (M1–M5) закрыты в коде с доказательствами,
вердикт не превышает in-sample GOF. Замечания — 3 MINOR (все опциональны).

**VERDICT: PASS**

AT0_5_REVIEW_R5_DONE
