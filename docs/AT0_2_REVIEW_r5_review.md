## Ревью обновлённого `AT0_2_shape.py` (после правок)

### Что исправлено правильно

**1. `G0a_tpu_hn_field` заменён на честный `g0a_hn_gate`.** Это главная правка. Прошлый вариант был тавтологией (`U*hn == h` при `U = h/‖h‖`, `hn = ‖h‖` — тождество). Текущий:

- `state`: unit-pin документирован как **регресс-гард** (`load_source` пересчитывает u/hn из raw h; если конвейер начнёт читать хранимый `_hn` — при коррупции `‖u‖≠1` в O(1)). Это **работает как защита от будущих регрессий**, честно описано.
- `gen_layers`: unit-pin на поле `_u` (замер fp16 ~2.4e-4) + **scale-pin** на однородность `U·hn`. Остаточный риск «константный/пермутированный `_hn` недетектируем» **зафиксирован в claim** — не скрыт.

**2. `_d_bands_l1` — reportable-only.** Явно прописано «нет null-распределения — p-value не вычисляется», `D_bands_L1_status` в `aggregate`. Убрана возможность интерпретации как теста.

**3. AR sign-проверка.** В `aggregate` добавлены `frac_cells_seed_sign_mismatch` (per-cell, по сидам) и `sign_check_ok: mism.mean() <= 0.1`. `note` теперь условный: «интерпретация refuted как 'про band-форму' валидна при sign_check_ok». Это **корректный фикс** к прошлому замечанию: вместо голого «знаки совпадают на aggregate mean» — реальная per-cell оценка.

**4. `confirmed_anomaly` interpretation.** Разъяснение добавлено: «real ближе к центру null, чем 97.5% null-сидов: p_doc=0.975, при null такое вероятно с вероятностью 2.5% — флаг, НЕ сигнал структуры». Корректно.

**5. `G0_sources.note`.** Правило решения прописано: «ЕСЛИ verdict'ы расходятся — решает verified-only».

**6. `band_sums_direct` — инвариант задокументирован.** Комментарий объясняет, почему `lo > hi` guard не нужен (после `hi = min(hi, N-1)` и `continue` для `lo > N-1`, инвариант `lo ≤ hi` держится для текущих BANDS). Корректно для текущего BANDS.

---

### Остающиеся мелкие проблемы

**A. `g0a_hn_gate` — `state` tolerance 1e-9 может дать ложный fail на «малых» h**

```python
tol = 5e-3 if is_gen else 1e-9
...
if udev > tol:
    raise ValueError(...)
```

Для state: `h = np.asarray(z[...], dtype=float64)`, `hn = ‖h_row‖`, `u = h/max(hn, 1e-12)`. Если hn > 1e-12, `‖u‖ = 1 ± ~1e-15` — проходит. Если hn < 1e-12 (маловероятно, но возможно для близких к нулю строк), `u = h/1e-12` → `‖u‖ << 1` → **fail**.

**В текущих данных не срабатывает** (h не бывает <1e-12). Но если какая-то модель/L1-degraded док имеет экстремально малые состояния, check упадёт. Это скорее всего desired (fail-loud на degenerate), но стоит осознать.

**B. `g0a_hn_gate` — `meas["state"]` инициализация `max_unit_dev: 0.0`**

Если `docs` пуст или все отфильтрованы (не наш случай, но hypothetically), `max_unit_dev=0.0` — log покажет `0.00e+00` и gate PASS. Если это impossible случай, то OK; если возможен — стоит `raise` на пустоту. Minor.

**C. `sign_check_ok` threshold 0.1 — произвольный**

```python
"sign_check_ok": bool(mism.mean() <= 0.1),
```

10% — reasonable, но не обосновано в артефакте. Стоит добавить в `note`: «порог 0.1 — эмпирический: при mean 0.02 (текущие данные) sign-mismatch минимальный; при 0.5 — знаки случайны и refuted был бы sign-driven». Сейчас интерпретация «валидна при sign_check_ok» без явного обоснования границы.

**D. `g0a_hn_gate` — `gen` scale-pin ловит CV ≥ 0.63, но не ловит пермутацию**

Явно задокументировано в `residual_risk`. Но если критично: `_hn` может быть пермутирован без изменения `mean`/`rms`/`CV`. Дополнительный guard мог бы проверять корреляцию `_hn` с `‖_u*_hn‖` поэлементно, но это пересчёт h (что и так делается). Не блокер.

**E. `band_sums_direct` инвариант — fragile если BANDS расширится**

```python
hi = min(hi, N - 1)
if lo > N - 1:
    continue
# comment: guard lo > hi был бы мёртвым кодом
```

Для текущих BANDS верно. Если в будущем добавят band типа `(0, 5)` или `(10, 2)` — guard нужен. Стоит `assert lo <= hi` перед `t1 = N - hi` (защита от будущих изменений, не runtime-cost).

**F. `G0a_hn_field` `state` — уточнить что именно verify**

Комментарий:
```python
"""state: load_source пересчитывает u, hn из raw h (npz-поле _hn — баг продюсера —
НЕ читается; load_source:186-188, проверено чтением кода). Unit-пин здесь
структурно истинен = регрессионный guard: если конвейер начнёт читать хранимые
поля, u перестанет быть unit (deviation O(1) state / 1.63× gen) — пин упадёт."""
```

Честно. Но в `gates["G0a_hn_field"]["state"]["check"]` — тот же текст. Читатель артефакта, не знающий контекста, может не понять, что unit-pin на state проверяет **только нормализацию**, а не hn. Добавить в gates краткое: `"verifies": "load_source (row-norm recompute); hn correctness proven by construction"`.

---

### Что осталось из прошлого ревью (не затронуто)

**G. `t25_ekurve_sha256` в AT0_2 отсутствует (это AT0_1).** AT0_2 использует только `GEN, OWN, COR`, не `t25`. OK.

**H. `n_docs` в `meta` = 96 (all), но `n_docs_verified` не отчитан явно.** `G0_sources.note` говорит «verified-only (80) — aggregates_verified_only», но нет поля `n_docs_verified: 80` в meta. Если downstream читатель хочет «сколько верифицированных доков», надо копать в `aggregates_verified_only[obj][null]["n_docs"]`. Minor.

**I. `confounds` — не отчитаны явно в AT0_2.** В AT0_3 был `confounds` (L/C × источник). В AT0_2 конфундов меньше (только `P2_corpus`), но `aggregates_verified_only` этого не описывает — просто числа. OK, но стоит добавить `meta.confounds_note`.

---

### Проверка на новые баги после правок

**J. `g0a_hn_gate` — state `tol = 1e-9`, gen `tol = 5e-3`.** Проверено: state (h recomputed from raw, `‖u‖=1` up to ~1e-15) → pass; gen (`_u` fp16, dev ~2.4e-4) → pass. OK.

**K. `sign_check_ok` при пустом `per_seed` — `np.mean([])` = nan.** Но `sel` отфильтрован по `p_doc is not None`, что требует `n_valid > 0`, что требует непустой `per_seed`. Так что не срабатывает. OK.

**L. `null_ar_batch` — `np.roll` correct.** Проверено вручную для p=2. Правильно.

**M. `band_shares_vec` — `denom` не может быть 0 для null (per seed `sgn ≥ DEGEN_SUMRHO`).** Guard `ok = np.abs(sgn) >= DEGEN_SUMRHO`. OK.

**N. `g0a_hn_gate` — runs before L1-degraded detection.** L1-degraded docs (own_w0, own_w4) are still checked. Since L1-degraded means fp16 noise in signal, h itself is fine → unit-pin passes. OK.

**O. `aggregate` — `D_bands_L1` per-cell mean.** `db = np.array([...])`, `db.mean()`. If `sel` is empty, `db.mean()` on empty array → nan warning (but not crash). `sel` is `p_doc is not None`, which requires non-empty. OK.

---

### Сводка

| # | Замечание | Критичность |
|---|---|---|
| A | state tol 1e-9 может дать fail на degenerate h | minor (edge case) |
| B | `max_unit_dev=0.0` initial для пустой docs | minor |
| C | `sign_check_ok` threshold 0.1 не обоснован | minor |
| D | gen scale-pin не ловит пермутацию | задокументировано |
| E | `band_sums_direct` invariant fragile | minor (defensive) |
| F | state check описание «verify» → уточнить | косметика |
| H | `n_docs_verified` не в meta headline | minor |
| I | `confounds` не отчитаны явно в AT0_2 | minor |

**Вердикт**: обновление **существенно улучшило** скрипт. Тавтологичная проверка заменена честным gate с явным описанием что именно verify и residual risk. AR sign проверка добавлена. `D_bands_L1` помечен как reportable-only. `aggregates_verified_only` имеет явное правило решения.

**Оставшиеся пункты — minor**. Ни один не блокирует артефакт. Рекомендации к уточнению:
- A: `if udev > tol: raise` уже fail-loud; для state tolerant 1e-9 это ok
- C: добавить в `note` обоснование порога 0.1 (например, «при пороге 0.5 refuted становится sign-driven»)
- E: `assert lo <= hi` — 1 строка, защита от будущих BANDS
- F: `"verifies"` поле в `gates["G0a_hn_field"]["state"]`

**Общий итог**: AT0_2 после правок — **честный артефакт**, все критические замечания из прошлого ревью адресованы, gate соответствует декларации, вердикты корректны.