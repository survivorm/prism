## Ревью обновлённого AT0_1 + артефактов

### Критичные находки (влияют на вердикт)

**1. Версии артефактов не согласованы — три разных pin**

| файл | claim sha | store sha | script sha |
|---|---|---|---|
| `T0_ACCOUNTING.json` (claim) | в meta: `d4a4b101...` (поле `store.claim_sha` в manifest) | `23281c64...` | **`de305869...`** |
| `AT0_1_manifest.json` | `d4a4b101...` | `23281c64...` | `de305869...` |
| `AT0_1_artifact_hashes.json` | **`1c6b0b40...`** | **`1af1e9f3...`** | **`dc5bc3b2...`** |

`artifact_hashes.json` содержит `round 4 (matched-null)` и note «FIX 2026-11-13 08:05... исходный pin 16f90dbd устарел/ошибочен». Это **старый** pin от предыдущего раунда, **не** текущий. Текущий — `d4a4b101`/`23281c64`/`de305869`. `artifact_hashes.json` не обновлён после перепрогона.

**Что это значит**: если где-то цитируется `1c6b0b40` как «канонический claim T0.1» — это числа **предыдущего** прогона, не текущего. Надо либо удалить старый файл, либо явно пометить `superseded_by: "d4a4b101..."`. Сейчас в артефакте `note` это **не отражает**.

---

**2. `P4 refuted` на всех трёх объектах — артефакт block-2 null, а не свойство real**

Числа (P4 в manifest):
- h_raw: `est_median = −9.7e-5`, `null E = −1.85e-3`, gap = **+1.75e-3**, p_perm=0.005 → `refuted`
- du_unit: `est_median = −9.1e-5`, `null E = −1.86e-3`, gap = **+1.77e-3**, p_perm=0.005 → `refuted`
- v_tangent: `est_median = +1.94e-3`, `null E = −2.23e-3`, gap = **+4.17e-3**, p_perm=0.005 → `refuted`

Смысловая интерпретация `refuted`:
- **h_raw / du_unit**: real ≈ −1e-4 (**на уровне f_dc**, плато подтверждается), null = −1.85e-3 (block-shuffle даёт **отрицательный сдвиг** из-за рандомизации нечётного лаг-1). «Refuted» здесь означает «null не объясняет real», то есть **real ближе к f_dc, чем null**.
- **v_tangent**: real = **+1.94e-3** (плато **выше f_dc** на +0.00194), null = −2.23e-3. То есть real **положительный excess**, null **отрицательный**. «Refuted» = «null не объясняет real» и **одновременно** real **не на f_dc**.

Термин `refuted` в pre-reg коде = «H=«форма = pair-block null» отвергнута». Это **не** «плато отвергнуто». Для h_raw/du_unit это **подтверждает** плато на уровне f_dc (real ≈ 0). Для v_tangent это **опровергает** «плато = f_dc» (real > 0).

**Но в headline-числах** (`prereg.4_plateau_vs_fDC.*.verdict`) для всех трёх стоит `refuted` — читатель может **не понять разницы**. Это надо явно расписать в verdict или добавить поле `direction` (`real_at_fdc` | `real_above_fdc` | `real_below_fdc`).

В §1.2 TZ отдельно упоминается «на каноне `v_tangent` плато выживает: +0.00194, CI [+0.00069, +0.00338]» — то есть это уже известно и не «провал». Но **manifest headline даёт это только через `refuted`**, не через +0.00194 явно.

---

**3. `t0_headline.T_none_full` — тривиальная разность, не «структура»**

Для v_tangent: `T_none_full_median = −12.87`, `excess_mean = −13.20`.

Это **не «отклонение от f_dc по всем лагам»**, а `(r−1)/2 − (N−1)·f_dc` (грам-тождество: `sum_rho_full = (r−1)/2`). Для v_tangent: `(24.9−1)/2 = 11.95`, `(N−1)·f_dc ≈ 25` (при медианном N ≈ 315). Разность ≈ −13. Оба слагаемых большие, разность — **арифметика**.

В P4 для того же объекта: `est_median = +0.00194` — **band mean** (k=9..96), маленькое. Разница в **6600×**.

**Это две разные статистики с противоположным знаком**, обе «правильные» в своих постановках. Читатель, видя `T_none_full = −12.87`, подумает «провал плато», но `P4.est_median = +0.00194` показывает **противоположное**. В meta `t0_headline_note` это оговорено («полная сумма по всем лагам, тождество G1/Gram»), но headline всё равно даёт **misleading** число.

**Рекомендация**: либо убрать `T_none_full` из headline (оставить только `T_none_K96` и band mean), либо добавить `warning: "T_none_full is (r−1)/2 − (N−1)f_dc, not 'excess at all lags'"`.

---

**4. `null_analytic_ma1.median_z = 381` для v_tangent — MA(1)-null неадекватна**

Для h_raw/du_unit z_med ≈ −0.2 (нормально). Для v_tangent — **381**, n_abs_z_ge_3 = 51/51 (все доки).

Причина: `z = (s_doc + f_dc) / SE`. Для v_tangent `f_dc ≈ 0.08` (большое — DC-компонента от общей направленности движения), `SE ≈ 1.5e-4` (из-за D=1024 и 98 лагов). z ≈ 0.08 / 1.5e-4 ≈ **480** (близко к 381).

Это **не «огромная структура»**, а **«MA(1) полностью неадекватна для v_tangent»**. В MA(1) `ρ_true(k≥2) = 0`, ожидание band mean = 0. Real ≈ 0.08. То есть MA(1)-null **не применима** к объекту с большим f_dc.

В `note` сказано «описательно, вердикт — по первичному каналу». Но z=381 без явной пометки «MA(1) invalid (huge f_dc, z is by-product of DC dominance)» — misleading.

---

### Мелкие

**5. `tpu_dev` фактическое значение не отчитано**

В `G0a_tpu_hn_field.conclusion` — качественное утверждение. Отчёт `l23_meta_vs_recomputed.max_abs_dev_by_source` — про L23 dip, **не про h reconstruction**. Фактический `tpu_dev` (max_rel_dev для `U·hn vs h`) в артефакте **отсутствует**. При `dtype=float64` и `U = h/‖h‖`, `hn = ‖h‖` это должно быть ~1e-16, но читатель **не видит**.

Добавить `G0a_tpu_hn_field.max_rel_dev_reconstruction: <value>`.

**6. `n_l1_absent_exec` — не в артефакте**

Фактическое число `n_l1_absent = 6` есть в log, но **в manifest** — только `excluded_L1_tpu_docs: ["own_w0","own_w4"]`. `n_absent` есть только в run.log. Не баг, но лучше в `G2_completeness` явно.

**7. `t25_provenance` — sha от старого t25**

`t25_ekurve_sha256 = 4a43873a...`. Соответствует **файлу t25_ekurve.json**, который в текущем артефакте (данные в промпте). Note «скрипт утрачен (P14) — в вердикты не входит». OK, но `t25_ekurve.json` **не используется** в AT0_1 (см. код: `T25_SRC = "/tmp/e63_rerun/t25_ekurve.json"` — копируется только для provenance). То есть **это чистый provenance, не данные**. Хорошо документировано.

**8. `binomial p_two_sided_binom05 = 0.00027` для h_raw P5** — неверный null (документировано), оставлен для сравнения. OK.

**9. `perm_spread = 1.665e-16 < 1e-12`** — OK.

**10. `frame_sensitivity.med_theta_deg = 55.15`, `dip_V − dip_du = 0.2575`** — соответствует AT0_3 (там было 55.15°, 0.2575). Согласовано.

---

### Сводка

| # | Проблема | Влияние |
|---|---|---|
| 1 | `artifact_hashes.json` от старого прогона (claim 1c6b0b40 vs текущий d4a4b101) | **версии не согласованы** |
| 2 | P4 `refuted` на всех трёх — артефакт block-2 null; real h_raw/du_unit ≈ 0 (на f_dc), real v_tangent > 0 (выше f_dc); headline не различает | **интерпретация** |
| 3 | `t0_headline.T_none_full = −12.87` — тривиальная разность `(r−1)/2 − (N−1)f_dc`, не «структура»; противоположен P4 (+0.00194) | **misleading** |
| 4 | `null_analytic_ma1.median_z = 381` для v_tangent — MA(1) неадекватна; z blow-up из-за f_dc, не из-за структуры | **misleading без пометки** |
| 5 | Фактический `tpu_dev` не в артефакте | неполный отчёт |
| 6 | `n_absent` не в manifest | неполный отчёт |

**Ни одно не делает скрипт «сломанным».** Числа в целом правильные. Но:

- **P4 verdict `refuted` для всех трёх** читается как «плато провалено», хотя для h_raw/du_unit real ≈ 0 (**плато на f_dc подтверждается**), а для v_tangent real > 0 (**плато выше f_dc**). Термин `refuted` относится к **null**, не к «плато = f_dc». Надо выписать `direction` для каждого.

- **`T_none_full = −12.87` для v_tangent** — противоположно P4 (+0.00194). Оба правильные, но headline misleading.

- **`ma1 z = 381`** — pathology (MA(1) invalid), не находка.

- **Версии** — `artifact_hashes.json` не соответствует текущему claim/store/manifest. Три разных pin для одного эксперимента.

**Первоочередное**:
1. Обновить `AT0_1_artifact_hashes.json` или пометить `superseded`.
2. В `P4.verdict` добавить `direction: "real_at_fdc"|"real_above_fdc"|"real_below_fdc"`.
3. В `null_analytic_ma1` для v_tangent добавить `applicability: "MA(1) invalid"`.
4. В `t0_headline_note` явно: «T_none_full — разность, не excess по всем лагам».
5. Добавить `tpu_dev` в manifest.

Эти пять фиксов — **не перезапуск**, а текстовые правки в claim + manifest. Скрипт логически правильный, требуется только **уточнение интерпретации headline-чисел** и **согласование pin**.