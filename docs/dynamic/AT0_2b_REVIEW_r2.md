# AT0_2b R/A-split — внешнее ревью round 2 (код + артефакт)

Ревьюер: независимая верификация r2-патча (C1–C7, M1–M3). Дата: 2026-11-15.
Метод: все числа — из фактических пересчётов (в /tmp, репо не модифицировалось, кроме этого файла).
Правка репо: только этот файл. Исполнители: `/home/survivor/.local/bin/python3.12` (py3.12.9, numpy 2.5.3 — тот же интерпретатор, что в руне, по строчке лога).

## 0. Сверка входных sha (до любых выводов)

| файл | sha256 | сверка |
|---|---|---|
| `experiments/AT0_2b_ra_split.py` | `0de271e27a930bdd05f35723678ea19873eb1a45207a47d9724a59078b8bd816` | = claim.meta.script_sha256 = manifest.script_sha256 ✓ |
| `prism/field.py` | `bcc607c561f95e3ed7aaf07677d088159aeb572b804c2f32fc9c5f4c26396532` | = claim.meta.field_sha256 ✓ |
| `results/AT0/AT0_2b_RA.json` (claim) | `9cc2a7cb71a11d8c3e29d32b6581d9f1cfadc345f0660b9afab54ea6cffdaf02` | = manifest.claim_sha256 = manifest.claim_file_sha256 ✓ |
| `results/AT0/store/b7e90c9366d9bbad.json` | `e7ad8c0578096d77979e77528522d3e220841ba47d80fa970b65e6561706e997` | = manifest.store_file_sha256 = claim.store_file_sha256 ✓ |
| `results/AT0/AT0_2b_manifest.json` | `a971bbf38452a3e6858f015d5a2ca223f7b0b885aa688936bcabdac1fa7a844f` | = «file sha» из /tmp/at02b_r2.log ✓ |
| gen npz | `f142f297ec16209d…` | = claim.meta.data_sha256 ✓ |
| P1_own.npz | `783e16338cdcb652…` | = claim.meta.data_sha256 ✓ |
| P2_corpus.npz | `5f42a08442a5d3bc…` | = claim.meta.data_sha256 ✓ |

## 1. Таблица: пункты ревью round 1 → статус в r2 → доказательство

| # | Пункт round 1 | Статус в r2 | Доказательство |
|---|---|---|---|
| C1 | verdict игнорирует вклад в net displacement Δh | **ЗАКРЫТ** | `contrib_R = share_rad + 0.5·share_cross`, зеркало — в `cell_ra`; `_carrier_verdict` работает по contrib (CI_low > 0.5 ∧ med > med), не по f_dc компонент. Идентичность `contrib = share + 0.5·cross` — **бит-точно (max |Δ| = 0.0)** по всем 478 present-клеткам; `contrib_R + contrib_A = 1` — max |Δ| = 4.26e-14 (fp64-округление 3-суммы; «ТОЧНО» — в математическом смысле). Полные пересчёты из npz (load_source + `pf.ra_split` + точная `cell_ra` скрипта) для 5 клеток (chat_0\|L23, math_3\|L11, math_3\|L23, own_w1\|L23, cor_w2\|L5): **0 расхождений при rel-1e-12**, включая shares/contrib. Ветка CANCELLED (cross_med < −0.5 ∧ обе shares > 0.5) — ДО carrier-веток, как в пре-реге; фактически достижима (loopy L23: cross_med −1.13, shares 0.57/1.53 — блокируется только вышестоящим NO_DH_DC-гейтом, порядок — по пре-регу). |
| C2 | нет DC-теста самого Δh / нет gate | **ЗАКРЫТ** | Per-cell `fdc_dh/p_fdc_dh/fdc_dh_perm_median/dc_present_dh` — в store, все 478 клеток. Агрегаты `dh_frac_dc/p_fdc_dh_med/fdc_dh_med` — во всех 15 (пул×слой). **Gate сработал: dh_frac_dc = 0.0 везде ⇒ verdict = NO_DH_DC везде** (15/15); p_fdc_dh_med = 1.0, dc_present_dh=False по всем 478 клеткам. Пересчёт `sign_flip_fdc` (тот же сид 2026+doc_idx·1009+s, doc_idx глобальный) для 5 клеток — **бит-точно** (fdc_dh, p, perm_median). Калибровка в main **в логе**: «iid median 2.617e-03 ~ 1/383; DC p=0.0050» (∈ [0.5/383, 2/383]; ≤ 1/(B+1)). |
| C3 | frac_dc/1.1 без uncertainty, нет n_present | **ЗАКРЫТ** | В агрегатах `radial/angular_n_present, _n_dc_present, _absent_frac` (все 15). Bootstrap-CI (B_BOOT=199, seed_ci = 9001+pool_idx·1009+L, pool_idx = index в sorted(pools): all_96=0, loopy=1, primary=2): пересчёт для **primary L23** (seed 9001+2·1009+23 = 11019) — **бит-точно** с claim: contrib r [−0.06860715875647427, 0.024533554334704142], a [0.9754664456652962, 1.0686071587564745], ratio [−0.06420241357573499, 0.025153514534121135]; аналогично boot_ci_fdc. CI **совместный**: один `idx` на итерацию на (R, A, ratio) — подтверждено (в каждой из 199 итераций med_r+med_a=1 до 6.7e-16; независимые извлечения дают другие границы — свойство детектируемо). `dominance_verdict` — старое правило v1 без изменений (1.1, без CI), переименовано в `*_DOM` по пре-регу v2, помечено descriptive. |
| C4 | sign-flip null — mean-direction, не ACF-matched | **ЗАКРЫТ** | `meta.null_note`: «MEAN-DIRECTION null (v2 C4; sign-flip Rademacher, B=199) … ГРАНИЦА: null НЕ сохраняет ACF/telescoping — «DC present» = направленный mean drift, НЕ spectral DC (не мостить к T0.1/T0.2)»; `claim.prereg.null` то же. В вердиктах/интерпретации смешения со spectral DC T0.1 нет (все вердикты NO_DH_DC; interpretation прямо разделяет «плато T0.1 = персистентность v_tangent, не DC Δh»). |
| C5 | n_docs может считать absent | **ЗАКРЫТ** | `n_cells_total/n_cells_present` + per-component `n_present/absent_frac`. Семантика правильная: all_96 L1: 96/94 (absent_frac 0.02083 = 2/96); loopy L1: 43/41; primary: 37/37. Полная пер-агрегация всех 15 (пул×слой) из store: **0 расхождений** с claim (медианы, frac, CI, кривые rho16/cross med16, вердикты). |
| C6 | decode probe skipped → provenance_limited | **ЗАКРЫТ (ok-ветка)** | `meta.decode_probe_status = "ok"` (проба: decode_probe[chat_0] = ' theme version.\nThe theme version is the'…), `meta.provenance_limited = null` (правило: при ok — null). `gates.provenance_limited = null`. `pool_note` сохраняет «Δh gen — provenance_limited (hn из файла, scale-пин G0a — тот же остаточный риск, что h_raw AT0_2); own/corpus — hn = ‖raw h‖ (верифицировано конструкцией)». |
| C7 | self-hash semantics store/manifest | **ЗАКРЫТ** | Пересчитано: canonical store sha (store_sha256=None, canonical JSON sort_keys) = `b7e90c93…` — **совпадает** с claim/manifest; canonical manifest sha (без self-поля) = `98ac37f7…` — совпадает; manifest **не содержит** свой file_sha (только canonical `manifest_sha256`); file sha store/claim = sha256 фактических файлов (пересчитано). Конвенция записана в `meta.hash_convention` — честно описывает разницу canonical-vs-file и почему manifest не содержит свой file_sha. (Оговорка про location-зависимость — MINOR-3 ниже.) |
| M1 | energy-additivity gate | **ЗАКРЫТ** | Per-cell `energy_add_rel` — все 478 клеток (обе компоненты present везде). Max = `5.278410046211474e-16` = claim.gates.gate_max.energy = лог (5.28e-16). Tolerance `ENERGY_TOL = 1e-12` в коде. Тест-сценарий (exec **точного фрагмента** из файла): real-данные → 1.97e-16, no raise (правильно); искусственный баг ra_split (A × 1.5) → **gate реально срабатывает**: `ValueError: G6 energy-additivity: 1.250e+00 > 1e-12`; absent-компонента → guard без краша, поле не пишется. |
| M2 | vt_plateau не в verdict | **ЗАКРЫТ** | `meta.vt_plateau_note` = «DESCRIPTIVE anchor (якорь плато T0.1), НЕ часть вердикта (v2 M2)» + повтор в pool_note; `vt_plateau_med` только в агрегатах (primary L23: 0.046057472127962365); входы `_carrier_verdict` не включают vt_plateau. |
| M3 | absent_frac не агрегируется | **ЗАКРЫТ** | `radial/angular_absent_frac` в агрегатах (в этом руне = 0.0 — absent-компонент нет ни в одной клетке; поле присутствует и корректно посчитано: all_96 L1 0.02083 = L1-absent). |
| — | Итог round 1: «PASS как R/A decomposition / REVISE verdict rule» | **REVISE-часть закрыта** | Вердикт-правило carrier (пре-рег §5 канал 2) реализовано бит-точно (порядок: not_testable → NO_DH_DC → CANCELLED → RADIAL/ANGULAR → INCONCLUSIVE; пороги MIN_PRESENT=5, dh_frac ≤ 0.5, cross < −0.5 ∧ shares > 0.5, CI_low > 0.5 ∧ med > med). Все 14 branch-тестов `_carrier_verdict` (exec точной функции, синтетические статы) — OK, включая границы (dh_frac=0.5 → NO_DH_DC; ci_low=0.5 ровно → INCONCLUSIVE; CANCELLED при cross=−0.49 не срабатывает). |

## 2. Независимый аудит (свои проверки сверх round-1 списка)

### 2.1 Provenance-цепочка — полная
manifest.claim_sha256 = sha256(claim файла) ✓; manifest.store_file_sha256 = sha256(store файла) ✓;
claim.meta.script_sha256 = sha256(скрипта) ✓; field_sha256 ✓; data_sha256 (gen/own/cor) ✓;
decode_probe_status = ok (не skipped) ✓; canonical store sha пересчитан сам ✓ (см. §1 C7).
claim.store_sha256 = manifest.store_sha256 = b7e90c93… (store filename = sha16 — консистентно).

### 2.2 Гейты и инварианты (store vs claim vs лог)
- Клеток: **480** (96×5) ✓; absent ровно 2: `own_w0|L1`, `own_w4|L1` (L1_degraded) ✓; pin `assert n_absent == 2` в коде; в пре-реге v2 §2 зафиксировано: «n_absent = 2 (пин, fail-loud; v2: 6 в v1 — арифметическая ошибка, поймана fail-loud при первом полном прогоне)» ✓.
- primary = **37** (пин `assert len(pools["primary"]) == 37`) ✓; loopy = 43, degen_docs 43 (без corpus — согласуется с AT0_2-флагами).
- gate max (пересчёт по store-клеткам = claim.gates.gate_max = лог, точное совпадение):
  closure `2.447324235011452e-15` / add `1.78442252313248e-16` / orth `8.881784197001252e-16` / energy `5.278410046211474e-16`. Все ≪ tolerances (1e-12/1e-12/1e-9/1e-12).
- `gate_max.gram = 0.0` — **НЕ измеренный максимум** (см. MINOR-1).
- Калибровки в main — в логе (см. C2). `field selfcheck: OK` — в stdout-логе.
- Маркер «AT0_2b gates: ALL PASS» — в /tmp/at02b_r2.log, RUN_RC=0 ✓ (в tracked run.log — см. MINOR-2).

### 2.3 Детерминизм (перепуск в изолированной копии /tmp/at02b_det)
Копия: скрипт+field.py (sha идентичны), gen (абсолютный путь /data/… — тот же файл), own/cor npz+meta, .hf_cache (tokenizer), интерпретатор py3.12.9/np2.5.3, thr=4. Прогон ~5 мин, RUN_RC=0.
- **Все 480 клеток — бит-идентичны** между двумя run'ами: canonical sha **cells-пейлоада** `8c6d830772275aaf58c5b04fc6a7a11191be477ffbd69975e4fcc602dc6e142d` (оба раза).
- Лог перепуска — построчно идентичен оригиналу (все агрегаты/вердикты/гейты), кроме строки с store-шарой.
- Canonical sha **всего store** различается (`1ec17d008546c8ae` vs `b7e90c9366d9bbad`) **исключительно** из-за ключей `meta.data_sha256` (ROOT-относительные абсолютные пути: /home/survivor/work/prism/… vs /tmp/at02b_det/…; значения sha файлов — те же). После нормализации ключей весь store-пейлоад бит-идентичен (`2484272d…`). Т.е. **вычисления полностью детерминированы**; location-зависимость — свойство hash-конвенции (MINOR-3).
- Claim перепуска идентичен оригиналу за теми же полевых store-шарами/именом.

### 2.4 Код: ветки/guard'ы, json, запись
- `cell_ra`: absent-ветки (dh_zero / L1_degraded / компонент degenerate) — покрываются; `nH = 0` → shares/contrib = None (в этом руне не срабатывает — все 478 клеток nH > 0); guard'ы `c0 <= 0 or c0 < 1e-6·scale` — точный сценарий (absent-компонента) протестирован exec'ом энергетического фрагмента (без краша).
- `_boot_ci_joint`: пустые списки → None (тест OK); `ma = 0` → NaN → ratio CI = None (тест OK); jointness — один rng/idx на итерацию (подтверждено численно, §1 C3). **Латентно** (не сработало в этом артефакте, 0 absent-компонент): r/a списки строятся независимо и паруются по позиции (`n = min(len(r), len(a))`) — если бы разные доки теряли разные компоненты, парность «joint» нарушилась бы без предупреждения (см. NIT-2).
- `_carrier_verdict`/`_dominance_verdict`: None-guard'ы (not_testable) — протестированы (14+7 кейсов, все OK); порядок веток = пре-рег.
- `json.dump`: store/claim — через `pf.jclean` (NaN/inf → None, numpy → python) ✓; manifest — только plain-строки ✓.
- Запись артефактов — **после** всех assert'ов/raise'ов (G2-пин 480, n_absent==2, primary==37, гейты в cell_ra) ✓.
- Dead code: существенных нет. `N, D = x.shape` в `sign_flip_fdc` — `D` не используется (NIT-1).
- decode-проба: try/except только `(ImportError, ModuleNotFoundError)` — другие сбои (FileNotFoundError на .hf_cache и т.п.) упали бы в краш, а не в «skipped» (NIT-3; в этом руне проба ok — не сработало).

### 2.5 Ответ/интерпретация (item 14)
Все 15 вердиктов = NO_DH_DC — подтверждено артефактом (dh_frac_dc = 0.0, p_fdc_dh_med = 1.0, fdc_dh_med 1.07e-5…5.6e-5 по пулам/слоям — тот же порядок, что v1-самокоррекция «f_dc(Δh) ≈ 1.5e-5», которая в claim явно подписана как v1). Формулировка «NO_DH_DC ⇒ нечего разводить (территория T0.5/VAR)» — согласуется с self-коррекцией пре-рега v1 §1. Заявлений, не подтверждённых артефактом, не найдено. Descriptive-канал (dominance_verdict) отдельно и честно: RADIAL_DOM на loopy L1/L11/L17/L23 и all_96 L1 (frac_R до 0.93) — не смешивается с primary-вердиктом.

## 3. Свои находки

**MAJOR: не найдено.**

- **MINOR-1. `gate_max.gram = 0.0` — инициализация, а не измеренный максимум.**
  `main()`: `n_gate = {"closure": 0.0, "add": 0.0, "orth": 0.0, "gram": 0.0, "energy": 0.0}`; обновляются только closure/add/orth/energy. Фактический max gram по store: **6.439e-15** (956 компонент-значений, все < 1e-10 — гейт работает fail-loud и не срабатывал). Числа в claim.gates.gate_max.gram / meta.gate_max.gram и логе (там gram не выводится) не отражают измеренный max. К вердиктам не относится (все гейты зелёные; store несёт per-cell gram).
- **MINOR-2. Tracked `results/AT0/AT0_2b_run.log` — stdout минус последние 3 строки.**
  `RUN_LOG` пишется **до** `logline("store: …")`, `logline("written: …")`, `logline("AT0_2b gates: ALL PASS")`. Маркер «AT0_2b gates: ALL PASS» и строки provenance-записи есть только в stdout (/tmp/at02b_r2.log). Manifest note говорит «run_log = stdout (tracked)» — неточно. Все гейт/калибровочные числа в tracked-логе присутствуют.
- **MINOR-3. Canonical store sha — location-зависимый (не задокументировано в hash_convention).**
  `meta.data_sha256` использует ROOT-относительные абсолютные пути как **ключи** словаря; перенос репо ⇒ другой canonical store sha при бит-идентичных клетках (измерено: cells-пейлоад совпадает, весь store — только после нормализации ключей). Для верификатора, пересчитывающего sha из другого места, нужен неявный шаг нормализации, отсутствующий в `meta.hash_convention`. Варианты фикса: ключи «basename → sha» (location-invariant) или записать правило нормализации в конвенцию.
- **MINOR-4. Пре-рег §2: «corpus 123..254» — фактически corpus N ∈ [82, 254].**
  cor_w8: N=82 (K=81), cor_w11: N=94 (K=93) — видно в store (rho96-длины {96, 81, 93}). Правило K = min(96, N−1) реализовано и описано корректно; corpus-доки не входят в primary/loopy (loopy = 43, без cor_*), на вердикты не влияет — неточность описания данных в пре-реге.

**NIT (не MINOR, для чистоты):**
1. `sign_flip_fdc`: `N, D = x.shape` — `D` не используется (мёртвая переменная).
2. `_boot_ci_joint`: lатентный desync r/a списков при разных absent-компонентах в разных доку (не сработал: 0 absent-компонент в 478 клетках); стоит assert len(r) == len(a) + совпадение документов.
3. decode-проба: except только ImportError/ModuleNotFoundError; FileNotFoundError/другие → краш, а не skipped (в этом руне ok).

## 4. Итог

r2 **полностью закрывает** round-1 замечания C1–C7, M1–M3: вердикт-правило carrier реализовано бит-точно по пре-регу §5 канал 2 (порядок веток, пороги, сиды CI и f_dc-нуля), Δh-DC gate сработал (dh_frac_dc = 0.0 ⇒ NO_DH_DC 15/15), n_present/absent_frac/совместный bootstrap-CI (пересчёт primary L23 — бит-точно), null-граница (mean-direction, не ACF-matched) записана, provenance (ok/null/Δh-gen) честен, hash-конвенция C7 задокументирована и верифицирована (canonical + file sha пересчитаны), energy-гейт реально ловит искусственный баг (exec точного фрагмента). Provenance-цепочка (script/field/data/claim/store/manifest) сходится на всех уровнях; 480 клеток, n_absent=2, primary=37 — пины на месте; полные пересчёты из npz (5 клеток, точная `cell_ra`) — бит-точно; детерминизм подтверждён перепуском в изолированной копии (клетки бит-идентичны).

Найдено 4 MINOR (диагностическая пометка gram, хвост tracked-лога, location-зависимость store sha, неточность пре-рега про corpus N) — ни одно не трогает вердикт, числа или воспроизводимость вычислений.

**VERDICT: PASS_WITH_MINORS**

(Финальный маркер: REVIEW_DONE — напечатан в stdout ревюера.)

## 5. Addendum round 2b (v2.1, store 15ad06bd)

Ревьюер: независимое повторное ревью v2.1 (МИНОР/НИТ-фиксы). Дата: 2026-11-15.
Метод: все числа — из фактических пересчётов (/home/survivor/.local/bin/python3.12, py3.12.9,
numpy 2.5.3; рабочая папка /tmp/at02b_r2b_review/). Репо не модифицировано, кроме этого файла.

### 5.0 Входные sha (v2.1)

| файл | sha256 | сверка |
|---|---|---|
| `experiments/AT0_2b_ra_split.py` (v2.1) | `99ab651f0f575816445652f987f68ebea99753f2f5f526b326a3b88e9181b673` | = claim.meta.script_sha256 = manifest.script_sha256 ✓ |
| `prism/field.py` | `bcc607c561f95e3ed7aaf07677d088159aeb572b804c2f32fc9c5f4c26396532` | не изменён c round 2 ✓ |
| `results/AT0/store/15ad06bdae5c14c5.json` (file) | `f8615708d78b61a4de556ef88ac6f0ca4c31fc964da27869d06db9d204e0d883` | = manifest.store_file_sha256 = claim.store_file_sha256 ✓ |
| `results/AT0/AT0_2b_RA.json` (claim, file) | `014bf176246540aac4d8162fc799ce90f66824b32dfeb5c8bf3e292d45747ddf` | = manifest.claim_sha256 (=claim_file_sha256) ✓ |
| `results/AT0/AT0_2b_manifest.json` (file) | `a100dcf972cb63625f05ae2081a65e79905b1451354c1b0a82a82ee873c18711` | = «file sha» из stdout-лога v2.1 ✓ |

Примечание по провенансу: в промпте ревюера «stdout: /tmp/at02b_r2.log» — это лог ПРОГОНА v2
(03:51; store b7e90c93, claim 9cc2a7cb). Фактический stdout v2.1-прогона — `/tmp/at02b_r2b.log`
(04:45; store 15ad06bd, claim 014bf176, `RUN_RC=0`, маркер ALL PASS; лок `run_one_at02b_r2b.lock`).
На вердикт не влияет — оба лога на месте, v2.1-лог согласуется с артефактами.

### 5.1 Числовой diff v2→v2.1 (рекурсивный, ВСЕ листья)

- **cells: bit-eq** — canonical JSON (sort_keys) идентичен, sha16 `8c6d830772275aaf` (тот же
  cells-пейлоад, что в round 2 §2.3: `8c6d830772275aaf58c5b04f…`).
- **Агрегаты (15 пул×слой): bit-eq** — пересчёт из store-клеток ТОЧНЫМ кодом v2.1 (exec
  фрагментов `_pool_stats`/`_boot_ci_joint`/`_dominance_verdict`/`_carrier_verdict` + константы
  из файла; B_BOOT=199, seed_ci=9001+pool_idx·1009+L): 15/15 bit-eq с claim; пересчёт из v2-клеток
  == пересчёт из v2.1-клеток (15/15); прямой кросс-чек: claim v2.1 primary L23
  `boot_ci_contrib` = [r −0.06860715875647427…0.024533554334704142, a 0.9754664456652962…
  1.0686071587564745, ratio −0.06420241357573499…0.025153514534121135] — **bit-в точности с
  задокументированными в round 2 (v2) числами**.
- **Гейты: bit-eq КРОМЕ gate_max.gram** — v2: `0.0` (инициализация) → v2.1:
  `6.439293542825908e-15`; остальные 4 (closure 2.447324235011452e-15, add 1.78442252313248e-16,
  orth 8.881784197001252e-16, energy 5.278410046211474e-16) — без изменений;
  claim.gates.gate_max == store.meta.gate_max (зеркало, проверено).
- **meta: ровно 4 ключа** — `data_sha256` (ключи: абсолютные пути → basenames; 3 значения-sha —
  без изменений), `gate_max` (gram), `hash_convention` (текст: дописана v2.1-заметка),
  `script_sha256` (0de271e2… → 99ab651f…).
- `store_sha256` (self-hash, топовый ключ): b7e90c93… → 15ad06bd… — необходимое следствие смены
  meta (canonical hash покрывает meta; конвенция C7), не числовой diff.
- **Любых иных отличий нет** (рекурсивный leaf-diff: 480 клеток, все 15 агрегатов, весь прочий
  meta/claim-зеркал — 0 расхождений).

**Числовой diff: НУЛЬ числовых изменений** (единственная числовая разница — gate_max.gram,
замена инициализации на измеренный максимум; см. MINOR-1).

### 5.2 MINOR-1..4 + NIT

- **MINOR-1 — ЗАКРЫТ.** Пересчёт по store: max per-cell gram = `6.439293542825908e-15` по
  956 компонент-значениям (478 present-клеток × 2), все < 1e-10 (GRAM_TOL-гейт не срабатывал,
  fail-loud на месте); == claim.gates.gate_max.gram == store.meta.gate_max.gram. Код:
  `n_gate["gram"] = max(n_gate["gram"], x["gram"])` в том же цикле, где GRAM_TOL-гейт.
- **MINOR-2 — ЗАКРЫТ (с NIT-пометкой).** Последняя строка tracked `AT0_2b_run.log` =
  `AT0_2b gates: ALL PASS`; строки `store: …`/`written: …` — в tracked-логе (23 строки).
  Сравнение с stdout v2.1 (/tmp/at02b_r2b.log, 25 строк): разница = ровно (a) строка
  `field selfcheck: OK …` — print из пинненого field.py, не проходящий через logline()
  (информация продублирована в G1′-строке: «_selfcheck ok»), и (b) финальная `RUN_RC=0` —
  дописывает run_one.sh, не скрипт. Все 23 строки tracked-лога присутствуют в stdout
  дословно и в том же порядке. NIT-пометка: manifest-note «run_log = stdout (tracked)»
  по-прежнему не буквальна (1 строка field.py не в tracked) — уровень NIT, не блокирует.
- **MINOR-3 — ЗАКРЫТ.** Ключи data_sha256 = basenames:
  `gen_layers_qwen.npz / P1_own.npz / P2_corpus.npz` (нет '/' в ключах); абсолютных путей
  НЕТ ни в одном payload (store/claim/manifest, regex-проверка); hash_convention явно
  документирует правило: «v2.1 MINOR-3: ключи data_sha256 = basenames файлов
  (location-invariant: перенос репо не меняет canonical store sha)». Следствие проверено:
  canonical store sha зависит только от содержимого payload → пересчёт из копии в /tmp
  даёт тот же sha (`15ad06bdae5c14c5…` — пересчитано, §5.4).
- **MINOR-4 — ЗАКРЫТ.** Пре-рег §2: «K = min(96, N−1), N = T_states−1 (gen/own 383;
  **corpus 82..254** — v2.1 MINOR-4: 123..254 в v2 — неточно; фактически cor_w8 N=82,
  cor_w11 N=94)». Кросс-чек по store (поле N клетки): corpus N min=82 (cor_w8), max=254;
  gen/own N=383; правило K=min(96,N−1) — 0 нарушений по всем 478 present-клеткам.
- **NIT-1 — ЗАКРЫТ.** `N, D = x.shape` устранена: `sign_flip_fdc` — `N = x.shape[0]`;
  мёртвого `D` в скрипте нет (grep `, D =` — пусто).
- **NIT-2 — ЗАКРЫТ (сильнее, чем просили).** `_boot_ci_joint`: `if len(r) != len(a):
  raise ValueError("…desync r/a…")` — fail-loud, строки 208–209.
- **NIT-3 — ЗАКРЫТ.** decode-проба: `except (ImportError, ModuleNotFoundError, OSError,
  ValueError)` → «skipped», не краш (строка 369).

### 5.3 Цепочка пинов

- manifest.claim_sha256 = sha256(claim файла) = `014bf176…` ✓ (claim_file_sha256 — то же)
- manifest.store_file_sha256 = sha256(store файла) = `f8615708…` ✓
- claim.meta.script_sha256 = sha256(скрипта) = `99ab651f…` ✓ (manifest.script_sha256 — то же)
- claim.meta.field_sha256 = sha256(field.py) ✓; data_sha256 = sha256 фактических npz (gen/own/cor) ✓
- **store_sha256 = canonical payload hash — пересчитан** (store_sha256=None, canonical JSON
  sort_keys): `15ad06bdae5c14c599d9d275d800a5fc2240c96789519419fd2ff2eeccf65c6e`
  = store-поле = claim.store_sha256 = manifest.store_sha256; имя файла = sha16 ✓
  (v2 sanity: canonical = b7e90c9366d9bbad… ✓)
- manifest.manifest_sha256 = canonical (без self-поля) = `2aa55ab5…` ✓
- claim.store_file_sha256 = `f8615708…` ✓; claim.meta.* == store.meta.* по всем общим ключам ✓

**Цепочка: OK.**

### 5.4 Регрессия

- 15/15 verdict = **NO_DH_DC** ✓ (dh_frac_dc = 0.0, p_fdc_dh_med = 1.0 — как в v2)
- primary L23: **contrib_R_med = −0.0015142104413634894**, **contrib_A_med = 1.0015142104413635**
  (bit-в точности с v2) ✓
- **n_absent = 2** (own_w0|L1, own_w4|L1; 480 клеток) ✓; **primary = 37** (96 − 16 corpus −
  43 degen) ✓

**Регрессия: OK.**

### 5.5 Итог

Изменения v2→v2.1 — только заявленные: НУЛЬ числовых изменений в cells/агрегатах (bit-eq,
независимый пересчёт 15/15 + кросс-чек с числами v2 из round 2); единственная числовая
разница — gate_max.gram (измеренный max вместо инициализации, MINOR-1); meta — ровно 4 ключа
+ self-hash. Все 4 MINOR + 3 NIT закрыты по существу. Цепочка пинов сходится на всех уровнях
(script/field/data/claim/store/manifest; canonical-шейфы пересчитаны). Регрессия чисел и
вердиктов — бит-в точности с v2. Единственная оставшаяся шероховатость — NIT-пометка
(manifest-note «run_log = stdout» не буквальна на 1 строку field.py) — не трогает гейты,
числа, вердикты или воспроизводимость.

**VERDICT: PASS**

(Финальный маркер: REVIEW_DONE — напечатан в stdout ревюера.)
