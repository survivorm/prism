# Внешнее ревью AT0_2 r6 — артефакт полного прогона T0.2 shape (round 2, B=199)

Роль: независимый верификатор (НЕ автор). Прогнан: 15.11, gates ALL PASS.
Артефакты: `results/AT0/AT0_2_SHAPE.json` (claim), `results/AT0/store/f7cc011ef856a6dc.json` (store, 222.8MB),
`results/AT0/AT0_2_manifest.json`, `results/AT0/AT0_2_run.log` / `AT0_2_r6_stdout.log`.
Код: `experiments/at0/AT0_2_shape.py` (1214 LOC), прибор `prism/field.py`.
Все числа ниже — из фактических пересчётов (numpy 2.5.3, /tmp), не из головы.

## Таблица проверок

| # | Проверка | Итог | Ключевые числа |
|---|----------|------|----------------|
| 1a | sha256 script/claim/store == заявленные | PASS | e58b7d19… / b3cad772… / f7cc011e… — пересчитано, совпадение |
| 1b | manifest.claim_sha256 == sha(claim); manifest.store_sha256 == sha(store) | PASS | оба совпали |
| 1c | manifest self-hash (convention: blob ДО вставки поля) | PASS | sha256(json.dumps(man без manifest_sha256, indent=1)) = 51a0ae58… = заявлено; файл == реконструкция побайтово |
| 1d | data_sha256 (gen/own/corpus) пересчёт + presence в gates и store.meta | PASS | gen f142f297…1349fdd7, own 783e1633…d58198, corpus 5f42a084…ec125 — все 3 пересчитаны и совпали; gen_sha_ok 64/64 (gen_meta.json: 64/64) |
| 1e | field.py sha-пин | PASS | bcc607c561f95e3ed7…c26396532 — пересчитано; в claim.gates.G1_field и store.meta.field_sha256 |
| 1f | store.prereg == claim.prereg; store.meta == claim.meta | PASS | json-равенство: True / True (хедер не дрейфует) |
| 2a | gates: все PASS | PASS | G0a_hn_field ok=True (state 3.33e-16, gen unit 2.43e-04, scale 0.0231), G2 ok=True (1440/1440, absent 6), G8_gram bad=0, G7_invariance bad=0, positive_control ok=True (4.441e-16 ≤ 1e-6; 2.345e-13 ≤ 2e-4). G0/G0_sources/G1_field/G7_store — raw-value гейты (без ok-флага) — PASS установлен пересчётом (1a–1e, 2b) |
| 2b | G0-декомпозиция: decode-пробы, corpus 0/16 → not_testable (G16) | PASS | gen decode_probe chat_0 (384 ток.), own 16/16 (probe own_w0, 384 ток.), corpus 0/16 → not_testable; text-пробы в log == claim.gates.G0_sources |
| 2c | degen 43/96 == флаги T0.1 (23281c64cc418ff1.json, per_doc degen_tz) | PASS | 43/43 полное совпадение, 0 расхождений в обе стороны |
| 2d | absent ровно own_w0\|L1, own_w4\|L1 — 6 клеток (2 дока × 3 объекта) | PASS | ровно 6: {own_w0\|L1, own_w4\|L1} × 3 объекта; L1-TPU пересчёт: own_w0 dev 0.8836, own_w4 dev 0.9211 (> 1e-3); остальные 14 own-доков ≤ 3.9e-5; flags == T0.1 per_doc.l1_tpu |
| 2e | store: 3 объекта × 480 клеток (96×5), per-cell ключи полные | PASS | 480/480/480; key-set единый для всех 1434 real-клеток (9 top-ключей, 12 null-ключей; 0 неконформных) |
| 3a | пулы: primary 37/185, verified_only 80, all_96 96 | PASS | n_docs/n_cells в каждом (obj,null) слоте: primary 37/185 ×27, verified_only 80/398 ×27, all_96 96/478 ×27 (478 = 480 − 2 absent на объект; «480» в промпте = все клетки включая absent) |
| 3b | ЛИНЕЙНОСТЬ: delta_mean == D_real_mean − D_null_mean (tol 1e-12), 3 пула × 3 × 9 | PASS | 0 нарушений из 81 (точное, diff = 0.0) |
| 3c | CI-методика | PASS (методика) | percentile-бутстрап B=1000 seed=117 (boot_ci, строка ~631) — асимметрична по конструкции, симметрия НЕ требуется; frac — Wilson с clamp [0,1] (M1): 81 CI все в [0,1], 0 инверсий |
| 3d | Агрегаты: полный независимый пересчёт из store (средние + все CI + frac + Wilson + exceedance-гистограммы + sum-invariance) | PASS | max\|Δ\| = 0.0 — bit-exact на всех 81 слотах (при точном run-порядке итерации: sorted(docs) × LAYERS [1,5,11,17,23]) |
| 3e | Monotonicity null-иерархии (all_96, reportable) | MINOR | h_raw и du_unit: iid>ar1>blk2>blk4>blk8>blk16>blk32 полностью монотонны (0.2783>0.2551>0.2124>…>0.1003 / 0.2752>0.2589>0.2118>…>0.1006); v_tangent — нарушение (см. MINOR-1) |
| 4a | D_real пересчёт от нуля (2 gen + 1 own, не degen) | PASS | math_3 L5 h_raw, code_2 L11 v_tangent (gen), own_w13 L5 h_raw (own): собственный direct per-lag einsum ACF (без FFT/cumsum): c0/r/sum_rho/rho96/prof5/tail ≤ 1.3e-15 rel (tol 1e-9); N = 383 = 384−1 |
| 4b | D_null(iid) пересчёт от нуля (seed'ы из meta, B=199 из store) | PASS | math_3: tested_D_center_l1 bit-exact (1.4030072441052281), per-seed r_null bit-exact, Dnull_l1_mean 4.5e-16 rel, p_doc=0.005/exceed=0 точны; code_2: ≤5.3e-15 rel. Store хранит per-null per_seed-распределения ({Dnull,DtoN,r_null,sum_rho} ×199) |
| 4c | delta/frac для 1 клетки ↔ log-строка | PASS | math_3 h_raw iid: per-doc delta (5 L) = 1.215896 (из store; L5 сверен от нуля); пул all_96 Δmean 1.2569 (log) = bootstrap-mean по 96 per-doc значениям — пересчёт bit-exact |
| 4d | log ↔ claim: 27 строк (3×9) == claim.aggregates (all_96) | PASS | 0 рассогласований на 162 полях (Δmean 4 знака, CI full-precision, D_real 4 знака, frac 2 знака, verdict); якоря промпта: h_raw iid 1.2569 ✓, v_tangent blk16 0.0922 ✓, D_real 0.1385 ✓ |
| 5 | r6-ревью E1–E8, M1–M5 закрыты | PASS | см. раздел «r6-пункты» (path'и в claim/коде) |
| 6 | код: assert-самоаудит / детерминизм / NaN-sanitize / границы | PASS | см. раздел «Код» |
| 7 | пре-рег: хедер == docs/AT0_2_prereg.md; DELTA_MIN/α/9 null/пулы | PASS (+2 замечания) | см. раздел «Пре-рег» (замечания Z1–Z2) |

## Развёрнутые результаты

### 1. Provenance-цепочка
- `sha256sum`: script e58b7d1911cb…375bc868, claim b3cad772b3b3…eb2d83d7, store f7cc011ef856…edb9340db77 — все совпадают с заявленными (manifest + claim.meta.script_sha256).
- claim.store_sha256 == sha256(store-файла) ✓; manifest.claim_sha256 == sha256(claim-файла) ✓; store-файл: sha256[:16] == имя файла ✓.
- Self-hash manifest: convention «по blob ДО вставки поля» воспроизведена: `json.dumps(man без manifest_sha256, ensure_ascii=False, indent=1)` → sha256 = 51a0ae583760…cd617777 == заявлено; побайтовая реконструкция файла == факт.
- Data sha (пересчёт): gen f142f297ec16209d1520cf6f4ee46fdd2543b0d73f566a69d93c26771349fdd7, P1_own 783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198, P2_corpus 5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125 — совпадают с claim.gates.G0.data_sha256 и store.meta.data_sha256. gen_meta.json: sha_ok 64/64.
- field.py: bcc607c561f95e3ed7aaf07677d088159aeb572b804c2f32fc9c5f4c26396532 ✓.
- store.prereg == claim.prereg (json-eq True) — хедер-скрипт не дрейфует между store и claim.

### 2. Gates (claim.gates)
Все гейты green. Декомпозиция:
- G0: 3 data-sha + gen_sha_ok 64/64 (пересчитано по gen_meta.json) + decode-пробы (gen chat_0, own_w0 — текст в log == claim; corpus 0/16 токенов → not_testable (G16), reason зафиксирован).
- G0a_hn_field: state unit-dev 3.33e-16 (regression-guard), gen unit-dev 2.43e-04 (fp16 _u), scale-dev 0.0231; residual_risk (пермутированный/константный _hn) задокументирован в claim.
- G2: 1440/1440 = 96×5×3; absent = 6 = 2 дока × 3 объекта; per-cell set uniform.
- G7-invariance: per-seed пересчёт |sum_rho_null − sum_rho_real|/max(1,|·|) по всем 1434 real-клеткам × 6 perm-null × 199 сидов: max = 4.552e-15 ≤ 1e-9 (совпадает с max sum_invariance_max_dev в агрегатах, 4.5519e-15).
- G8: 0 клеток с gram_resid ≥ 1e-10 (перескан по всему store).
- Positive control: max\|Δρ̂_circ\| = 4.441e-16 ≤ 1e-6, max\|Δr_full\| = 2.345e-13 ≤ 2e-4 (перескан по всему store == claim); wrap-отчёт 2.109e-1 (свойство суррогата, не гейт).
- G-seam (E2): max_dev within-block identity = 1.95e-15 ≤ 1e-12 (перескан: 0 клеток > 1e-12).

### 3. Пулы и арифметика
- n_docs/n_cells в каждом (obj,null) слоте: primary 37/185, verified_only 80/398, all_96 96/478 — во всех 27 слотах каждого пула (478 = 480 − 2 L1-absent на объект).
- ЛИНЕЙНОСТЬ: 0 нарушений из 81, точное (delta_mean == D_real_mean − D_null_mean в machine-precision).
- Полный пересчёт агрегатов из store (свой код, run-порядок): bit-exact на всех 81 слотах — средние, D_real_ci/D_null_ci/delta_ci (percentile bootstrap B=1000 seed=117), frac + Wilson, exceedance_count_hist, n_docs_unique_p_lt_alpha, p_docs_median, sum_real/sum_null/sum_invariance. Max\|Δ\| = 0.0.
  (Промежуточный аномальный diff 0.0084 в CI пойман как артефакт МОЕГО порядка итерации: store dump'ится с sort_keys → chat_10 < chat_1, бутстрап зависит от порядка per-doc списка; в run-порядке — 0.0. Не дефект артефакта.)
- Verdict-правило: ре-дербивация по пре-рег правилу (|Δ|<0.05 → not_testable; Δ≥0.05 ∧ CI>0 → refuted; Δ≤−0.05 ∧ CI<0 → confirmed_anomaly; иначе inconclusive) на 81 слоте: 0 рассогласований.
- Monotonicity: h_raw/du_unit — полная иерархия iid > ar1 > blk2 > blk4 > blk8 > blk16 > blk32 (0.2783>0.2551>0.2124>0.1523>0.1257>0.1054>0.1003; 0.2752>0.2589>0.2118>0.1525>0.1255>0.1055>0.1006). v_tangent — нарушение, см. MINOR-1.

### 4. Независимый пересчёт от нуля
- Клетки: math_3|L5 h_raw (gen), code_2|L11 v_tangent (gen), own_w13|L5 h_raw (own) — все не-degen.
- (a) Real: собственный direct per-lag einsum ACF (не FFT, не cumsum): все поля ≤ 1.3e-15 rel (c0 — bit-exact). N=383 = min(384, T)−1.
- (b) D_null(iid) от нуля: 199 сидов, seed = 2026 + doc_idx·1009 + s (doc_idx из sorted(96 доков): math_3=57, code_2=24, own_w13=69): math_3 — tested_D_center_l1 bit-exact, per-seed r_null bit-exact (max|Δ|=0.0), Dnull_l1_mean 4.5e-16 rel, p_doc=0.005, exceedance 0 — точны; code_2 — ≤5.3e-15 rel. Третья клетка: внутренняя согласованность store (mean(per_seed Dnull) == Dnull_l1_mean до 1e-12, p_doc пересчитан из per_seed, n_valid=199).
- (c) per-doc delta math_3 h_raw iid = 1.215896 (5 L из store, L5 сверен от нуля); пул all_96 Δmean = 1.2569 (log/claim) — bootstrap по 96 per-doc значениям, пересчёт bit-exact.
- (d) log↔claim: 27 строк × 6 полей — 0 рассогласований. Якоря: h_raw iid Δmean 1.2568570905181 (log 1.2569) ✓; v_tangent blk16 Δmean 0.09215167417537133 (log 0.0922) ✓, D_real 0.1385283215096689 (log 0.1385) ✓.
- E3-репродукция r1→r6 (store 5b05537017013a4f, B=20): на 4 клетках ответки (math_3|L5 h_raw, math_3|L23 v_tangent, chat_1|L11 du_unit, own_w0|L17 h_raw): real-блоки bit-exact (max|Δ|=0.0); per-seed r_null и DtoN, s=0..19, ×9 null'ов — bit-exact (max|Δ|=0.0, 480 значений на клетку). Dnull отличается (max 0.091) — ровно как объявлено в ответке (зависит от B через C=mean по B сидам; НЕ гейт).

### 5. r6-ревью (E1–E8, M1–M5) — закрытость в артефакте/коде
| Пункт | Статус | Path / доказательство |
|-------|--------|----------------------|
| E1 (estimand: форма при фиксированной сумме; Σ-тест вырожден) | закрыт | store.prereg.estimand; агрегаты: sum_verdict="degenerate (ТЗ v1.2 G7…)" во всех 45 perm-null слотах; sum_invariance_max_dev ≤ 4.6e-15 (перескан всего store: 4.552e-15); AR-клетки: sum_mismatch/r_real/r_null_mean/sum_rho_null_mean (0 отсутствующих) |
| E2 (blk = damped/seamed null; identity-check) | закрыт | store.prereg.nulls["blk{2,4,8,16,32}"] (w(k,b)=max(0,1−k/b), «НЕ чистый null локальности»); per-cell seam_excluded_identity_max_dev ≤ 1.95e-15 во всех 1434 клетках (гейт ≤1e-12, fail-loud assert строка 988); код: null_blk_seam_excl с pos-курсором по пермутированным блокам + assert pos==N (строка 301) |
| E3 (B=199, p_floor=0.005, raw exceedance, репродукция сидов 0..19) | закрыт | store.meta.env.b_null=199; per-cell exceedance_count + агрегат exceedance_count_hist (пересчёт bit-exact); репродукция r1→r6: r_null/DtoN/real bit-exact на 4 клетках ответки (перепроверено: max\|Δ\|=0.0) |
| E4 (paired verdict: delta_d per-doc, cluster-bootstrap по докам) | закрыт | aggregate() строки ~660–700: delta_d = per-doc (D_real − D_null), boot_ci(B=1000, seed=117); агрегаты: delta_mean/delta_ci/verdict; ре-дербивация правила на 81 слоте: 0 рассогласований |
| E5 (AR diagnostics) | закрыт | per-cell: ar_max_root, n_nonstationary, burn_mix_ratio, sigma2_clip_frac, rho_real_1to4, rho_null_1to4_mean, sum_mismatch (0 отсутствующих в 4302 AR-клетках); агрегат: sign_check_ok, frac_cells_seed_sign_mismatch; wording в note (AR1/AR2 ≈ тавтология, AR4 — conditional) |
| E6 (primary pool закреплён) | закрыт | claim.primary_pool="verified_nondeg" (37 доков); pool_roles явный; aggregates_primary/verified_only/aggregates; G0_sources.note: «ЕСЛИ verdict'ы расходятся — решает primary» |
| E7 (degen-доки вне primary) | закрыт | claim.degen_docs_n=43 (== T0.1 per_doc degen_tz, сверено); per-cell degen-флаги (43 доков, 0 расхождений с T0.1); degen_note (loopy = full-pool shape) |
| E8 (object provenance) | закрыт | per-cell provenance_limited: ровно gen×{h_raw,v_tangent} (128 клеток True, 0 неконформных из 1440 — 6 absent-клеток без флага, по конструкции status-only); claim.object_provenance + primary_objects (du_unit — primary на gen) |
| M1 (Wilson clamp [0,1]) | закрыт | wilson_ci (строка ~612): max(0,…)/min(1,…); 81 Wilson-CI в claim: все в [0,1], 0 инверсий |
| M2 (rename tested/descriptive) | закрыт | per-cell tested_D_center_l1/l2, descriptive_D_bands_L1 (reportable only) — key-set uniform по store |
| M3 (primary family = 9 тестов, FWER не применяется — объявлено) | закрыт | store.prereg.primary_family; aggregates_primary: 9/9 тестов присутствуют с verdict'ами (6 refuted, 3 not_testable — v_tangent iid/blk32/ar4: Δmean 0.0444/0.0181/0.0283 < 0.05) |
| M4 (DELTA_MIN=0.05 pre-registered) | закрыт | хедер строка ~107 (B_NNULL/DELTA_MIN константы), pre-reg doc round 2, агрегаты: delta_min=0.05 во всех 81 слоте |
| M5 (confirmed_anomaly semantics) | закрыт | store.prereg.verdict + verdict_note в агрегатах; в r6-результатах confirmed_anomaly нет (все refuted/not_testable) — ветка не срабатывала, семантика зафиксирована |

### 6. Код (experiments/at0/AT0_2_shape.py)
- **Assert-самоаудит останавливает запись**: все fail-loud assert'ы (G2:963, G8:976, G7-inv:977, G-seam:988, positive-control:999) предшествуют блоку записи (store:1024, claim:1093, manifest:1183, run.log:1209); `--smoke` возвращает ДО записей (1013, «SMOKE: артефакты НЕ пишем»). Сбой любого assert ⇒ артефакты не пишутся. ✓
- **Детерминизм**: все 6 call-site `np.random.default_rng` сессионно засеяны (363 AR-eps, 410 perm-null, 449 seam, 538 fourier (+1000+s), 631 boot (seed=117), 823 test-pin (123)); нет global `np.random.*`, нет `time`, нет hash-итерации; документы итерируются `sorted(docs)`, ключи store — `sort_keys=True`; пересчёт агрегатов bit-exact — эмпирическое подтверждение. ✓
- **NaN/Inf sanitize**: jclean (field.py:161) NaN/Inf → None; raw-скан claim: 0 токенов NaN/Infinity. ✓
- **Границы**: field.py — importlib read-only + `pf._selfcheck()`; данные — `a1.load_source`/`np.load` read-only; AT0_1-скрипты не модифицируются; writes — только 4 AT0_2-артефакта. ✓

### 7. Пре-рег-соответствие
- Хедер скрипта == docs/AT0_2_prereg.md (round 2 секция): bands {1},{2},{3..8},{9..32},{33..96} + tail-скаляр; нормировка (r−1)/2 полные лаги, AR — своя сумма; distance L1-primary/L2-descriptive/descriptive_D_bands_L1 (M2-имена); B=199, p_floor 1/200, seed=2026+doc_idx·1009+s, fourier B=3; 9 null-типов (NULL_NAMES) + fourier positive control; seam-excluded отдельный оценщик + identity ≤1e-12; verdict paired, DELTA_MIN=0.05, B_BOOT=1000/SEED_BOOT=117; пулы 37/80/96; object provenance; primary family 9; G-гейты. Расхождений нет. Round 1 секция дока — историческая (B=20), док сам декларирует «источники правды: хедер + ответка». ✓
- DELTA_MIN=0.05 ✓; 9 null-типов ✓; пулы 37/80/96 ✓.
- Замечание Z1: α=0.05 (ALPHA) зафиксирован в коде и pre-reg doc (round 1: «p_doc < 0.05»), но НЕ явно в хедере (единственном источнике правды round 2). Риск мал: frac — descriptive only, в verdict не входит.
- Замечание Z2: TZ_add_0-3.md §1.5 T0.2 таблица всё ещё гласит «blk — null локальности»; r6 E2 переопределил wording в damped/seamed (задокументировано в хедере/pre-reg/ответке с алгеброй w(k,b)). TZ не обновлялся; артефакт самосогласован.
- Замечание Z3 (промпт↔артефакт): промпт называет метрику «D (participation ratio) пер-дим дисперсий»; фактическая D артефакта = tested_D_center_l1 — L1-расстояние 5-вектора band-долей real от центра null (согласно TZ «ФОРМА профиля при фиксированной сумме»). Промпт деферирует «по формуле из скрипта» — все сверки выполнены по формуле скрипта, расхождений не найдено. Для отчётов: не переносить формулировку «participation ratio» из этого промпта в docs.

## Находки

MAJOR: нет.

MINOR:
1. **Monotonicity null-иерархии нарушена для v_tangent (all_96)** — reportable-only, не гейт.
   Where: claim.aggregates[v_tangent][*].D_null_mean.
   Заявлено (промпт): D_null(iid) > D_null(ar1) > blk2 > …
   Пересчитано (v_tangent): iid 0.0412, ar1 0.0049, ar2 0.0064, ar4 0.0085, blk2 0.0556, blk4 0.0558, blk8 0.0441, blk16 0.0464, blk32 0.0538.
   Нарушения: (a) ar1 (0.0049) < blk2 (0.0556) — цепь iid>ar1>blk2 рвётся; (b) blk-цепь не монотонна (0.0556≈0.0558 > 0.0441 < 0.0464 < 0.0538).
   Почему: Dnull = разброс null-сидов вокруг центра null. У v_tangent реальная ACF быстро спадает (v̂ ≈ белая, профиль сконцентрирован в полосе k=1), поэтому AR-null (Yule-Walker) воспроизводит ACF real с точностью до fit-error → сиды лежат близко к центру (малый Dnull); разброс iid/blk сидов задан шумом в долях k=1..8 и не следует «силе» null-иерархии. Для h_raw/du_unit (hump-ACF) иерархия полная.
   Чинить: не требуется (не входило в verdict); в отчёте не утверждать иерархию для v_tangent либо пометить как object-dependent.
2. **α=0.05 не зафиксирован в хедере** (Z1): только в коде (ALPHA) и pre-reg doc round 1. Чинить: добавить строку в хедер при следующем run (frac descriptive-only — риск мал).

Замечания (не блокируют):
- N3: run.log и AT0_2_r6_stdout.log различаются одной строкой «field selfcheck: OK …» (print из pf._selfcheck() не попадает в LOG). run.log — tracked-канон; расхождение космо́тическое.
- N4: claim.gates G0/G0_sources/G1_field/G7_store — raw-value гейты без ok-флага (PASS установлен пересчётом, таблица 1a–1e/2b).
- N5: all_96 n_cells=478 (не 480): 2 L1-absent на объект; «480» в промпте — все клетки включая absent.
- N6: расхождение verdict'ов по пулам реально существует для v_tangent (primary: 3/3 not_testable; all_96: refuted) — обработано по пре-регу («решает primary»); в финальном отчёте verdict v_tangent = not_testable (primary), расхождение раскрыть.

## Вердикт: PASS

Provenance-цепочка, gates, пулы, арифметика (линейность точная; агрегаты bit-exact от пересчёта), независимый пересчёт от нуля (real ≤1.3e-15 rel; iid-null bit-exact/≤5.3e-15), log↔claim (162/162 полей), r6-пункты E1–E8/M1–M5 (все закрыты с path'ами), код (asserts before writes, детерминизм, sanitize, границы) и пре-рег (хедер == док) — подтверждены пересчётами. MAJOR: 0. MINOR: 2 (monotonicity v_tangent — reportable; α не в хедере).

### Что проверить при ответке
1. MINOR-1: если иерархия D_null собирается фигурировать как reportable-инвариант — обосновать object-dependent характер (v_tangent) или убрать из заявлений; verdict'ы не затрагивает.
2. MINOR-2 (Z1): α=0.05 в хедер следующего run.
3. Z2: обновить таблицу TZ §1.5 T0.2 (blk — damped/seamed, не «null локальности») — артефакт уже согласован с r6-переопределением.
4. N6: в пользовательском отчёте зафиксировать primary-verdict v_tangent = not_testable (Δmean 0.0444/0.0181/0.0283 < 0.05) и раскрыть расхождение с all_96 (refuted) — пулы дают противоположные ответы, решающий пул = primary.
5. Z3: не переносить формулировку «participation ratio» из ревью-промпта в docs — фактическая метрика = L1 band-share distance (tested_D_center_l1).

AT0_2_R6_REVIEW_DONE
