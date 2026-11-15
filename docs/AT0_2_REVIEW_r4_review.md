## Ревью обновлённого AT0_2_shape.py

### Исправлено правильно

**Domain gate (n≥12)** — теперь `n_docs_nondeg` (документы, а не клетки), с явным комментарием «ТЗ §0.3: n — документы (wiki = 1 док)». ✓

**G0_sources** — per-source статус, `not_testable` для corpus, `decode_probe` для own. ✓

**`aggregates_verified_only`** — corpus-excluded версия, отдельное поле. ✓

**`aggregate_levels_note`** — явно различает «клетки» (frac_cells) и «документы» (n_docs_unique). ✓

**AR sign documentation** — добавлены `sum_rho_real_mean`, `sum_rho_null_mean`, `sign_mismatch`, `D_bands_L1` (unnormalized band distance). Note: «в этих данных знак суммы AR совпадает с real». ✓ (частично, см. ниже)

---

### Критичное: G0a_tpu_hn_field **не проверяет то, что заявлено**

```python
for nm, d in docs.items():
    for L in LAYERS:
        u, _ = d["layers"][L]
        dev = float(np.abs(np.linalg.norm(u, axis=1) - 1.0).max())
        tol = 5e-3 if d["source"] == "gen_layers" else 1e-9
        if dev > tol:
            raise ValueError(
                f"u не unit ... — hn не row-norm'ы (G0a_tpu_hn_field), h_raw недостоверен")
```

**Логика ошибочна**. `u = h / ‖h‖` по построению (`load_source` нормирует). Тогда `‖u‖ = 1` **всегда**, независимо от того, правильный ли `hn`:
- **Case A (правильный hn = ‖h_row‖)**: `u = h/‖h‖`, `hn = ‖h‖`, `u*hn = h` — корректно.
- **Case B (битый hn = ‖h_col‖)**: `u = h/‖h‖` (всё ещё unit), `hn = ‖h_col‖`, `u*hn ≠ h` — сломано.

В **обоих** случаях `‖u‖ = 1` → **проверка проходит в обоих**. Значит, она не ловит случай B, ради которого написана.

**AT0_3 делает правильно**: сравнивает `U * hn` с raw `h`, загруженным напрямую из npz:
```python
h = np.asarray(z[f"{nm}_L{L}"], dtype=np.float64)
dev = float(np.max(np.abs(U * hn[:, None] - h) / np.maximum(np.abs(h), 1e-12)))
```

**AT0_2 — регрессия** относительно AT0_3. Заявление в error message («hn не row-norm'ы») не соответствует реальной проверке.

**Fix** (заменить проверку):
```python
# G0a_tpu_hn_field: проверить, что u * hn == raw h (не только ‖u‖=1)
for nm, d in docs.items():
    z = np.load(OWN if d["source"] == "P1_own" else COR, allow_pickle=True)
    for L in LAYERS:
        u, hn = d["layers"][L]
        h_raw = np.asarray(z[f"{nm}_L{L}"], dtype=np.float64)
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        dev = float(np.max(np.abs(U * hn[:, None] - h_raw) / np.maximum(np.abs(h_raw), 1e-12)))
        if dev > (5e-3 if d["source"] == "gen_layers" else 1e-6):
            raise ValueError(f"hn реконструкция сломана ({nm}, L{L}, dev {dev:.3e})")
```

**Или** честно переименовать в `G0a_unit_pin` и в `gates` явно написать `"checks": "‖u‖=1 only; hn correctness NOT verified"`.

---

### Существенное: AR refuted по `D_center_l1`, который знак-зависим

Код документирует:
```python
"note": ("... refuted — про band-форму (распад vs структура), не про знак; ...")
```

Но **вердикт** в `aggregate` вычисляется из `D_center_l1`:
```python
Dc = [float(np.mean(v)) for v in per_doc_c.values()]
d_real, ci_real = boot_ci(Dc)
...
elif d_real > ci_null[1]:
    verdict = "refuted"
```

`D_center_l1` = `‖P_real − C‖`, где `P` — `bandsums / c0 / sgn`. Для AR с `sgn_AR < 0` (возможно, если per-cell знак не совпадает) `P_AR` — отрицательный → L1 с `P_real(+)` огромен → `refuted` тривиально.

**`D_bands_L1`** (sign-safe) посчитан и записан, но **не входит в verdict**. Note говорит «refuted — про band-форму», но computation — про `D_center_l1`, который включает sign-mismatch.

**Что делать** (одно из):
- **(a) Verdict из `D_bands_L1`**: `d_real = mean(D_bands_L1)`, `d_null = mean(D_bands_L1_null)` — но `D_bands_L1` сейчас нет null-distribution, только point estimate.
- **(b) Fail-loud при sign-mismatch**: `if sign_mismatch_aggregate or sign_mismatch_per_cell_frac > 0.1: verdict = "not_testable_sign_mismatch"`.
- **(c) Оставить, но в `verdict_note` для AR явно писать**: `"note": "verdict computed from D_center_l1 (sign-normalized); see D_bands_L1 for sign-safe comparison"`.

Сейчас `sign_mismatch` — только поле-индикатор; verdict не меняется. Читатель артефакта, видя `ar1: refuted`, не знает, тривиально это или нет. `D_bands_L1_mean` в aggregate есть, но не headline.

**Мера критичности**: для AR(1) с φ<0 (что реалистично) sign mismatch **возможен**. Если в текущих данных `sign_mismatch=false` на всех клетках — verdict честный. Но `sign_mismatch` считается на aggregate mean, не per-cell. Cell-level знак может отличаться от aggregate знака.

---

### Мелкие, но конкретные

**1. `band_sums_direct` — нет guard `lo > hi`**

```python
for j, (lo, hi) in enumerate(bands):
    hi = min(hi, N - 1)
    if lo > N - 1:
        continue
    t1 = N - hi
    S1 = C[:, hi + 1:N + 1, :] - C[:, lo:N - hi + lo, :]
```

Если `lo > hi` (после `min`), то `S1` = `C[hi+1:N+1]` (длина `N-hi`) − `C[lo:N+1]` (длина `N-lo+1`) — **broadcasting fail** (разные длины). Для N=384 все полосы OK, но при N<34 — потенциальный краш.

**Fix**: добавить `if lo > hi: continue` сразу после `hi = min(hi, N-1)`.

**2. `sign_mismatch` — aggregate-only**

```python
sr = np.array([cc["real"]["sum_rho"] for _, cc in sel])
sn = np.array([cc["nulls"][nm]["sum_rho_null_mean"] for _, cc in sel])
"sign_mismatch": bool(np.sign(sr.mean()) != np.sign(sn.mean())),
```

Это сравнение **средних**. Если 50% клеток имеют sr>0 и 50% sr<0, `sr.mean()` может быть ≈0 и `np.sign(0) = 0` — mismatch с `sn.mean()>0` даст true, но реально «знаки смешаны», а не «все противоположны».

**Fix**: использовать моду-знак или добавить `frac_cells_sign_mismatch`:
```python
per_cell_mismatch = np.mean(np.sign(sr) != np.sign(sn))
"frac_cells_sign_mismatch": float(per_cell_mismatch),
```

**3. `aggregate_verified_only` — только агрегаты, не вердикты**

В `claim` есть `aggregates_verified_only`, но **`verdict` остался один**. Если verified (80 доков) и unverified (96) дают разные verdict — это видно только при сравнении полей `verdict` в `agg[obj][nm]` vs `agg_v[obj][nm]`. Стоит добавить в `verdict` headline поле типа `"verdict_verified_only": agg_v["du_unit"]["blk32"]["verdict"]` для каждой значимой пары.

Или явно в G0_sources note: «headline `aggregates` = 96 docs (including not_testable corpus); `aggregates_verified_only` = 80 docs; если они расходятся — решает verified».

**4. `h_raw` для gen в check**

В цикле cell-прогона:
```python
if d["source"] == "gen_layers":
    # gen-only check: mean(hn) ≈ RMS(h)·√D
    if abs(hn.mean() - rms * np.sqrt(D_)) / (rms * np.sqrt(D_)) > 0.2:
        raise ValueError(...)
```

Для `gen` правильная проверка (scale test), но она **внутри цикла по клеткам**, не в gate-секции. Если check упадёт на 32-м доке, 31 док уже посчитан — wasted work. Стоит вынести в gate до цикла (как `G0a_tpu_hn_field` в AT0_3).

**5. `D_bands_L1` в `aggregate` — нет null-распределения**

`D_bands_L1_mean` — mean across cells. Но нет `D_bands_L1_null` (null distribution) для сравнения. Значит `D_bands_L1` не testable в текущей форме, только reportable.

**Fix**: либо (a) убрать из aggregate (оставить в cell-only), либо (b) добавить permutation distribution.

**6. `verdict_note` для `confirmed_anomaly`**

```python
elif d_real < ci_null[0]:
    verdict = "confirmed_anomaly"
    verdict_note = (f"D_real {d_real:.2e} < CI_null низ {ci_null[0]:.2e}: "
                    "real более «типичен», чем null-вариативность")
```

Это значит, что real **ближе к центру null**, чем типичный null sample. Interpretation не очевидна. Что это значит по существу? Real форма **более усреднённая** чем null? Или real форма **более extreme** чем null (в смысле специфичности)? Неочевидно.

Pre-reg говорит `confirmed_anomaly` как отдельный исход, но что он означает — не описано. Стоит в pre-reg добавить интерпретацию.

---

### Что проверено корректно

- **Pins** `acf_full` vs direct, `acf_full_many` vs batch, `band_sums_direct` vs per-lag, `r` vs FFT — все работают.
- **`dists_to`** — правильная логика (D_center = ‖P_real − C‖, D'_s = ‖P_s − C‖, p = #{D'_s ≥ D_center}/(n+1)). ✓
- **`null_fourier`** DC/Nyquist fix, phase на промежуточных — правильный spectrum-preserving суррогат.
- **`boot_ci`** — percentile, seed=117, кластер на доке (в `aggregate`), корректный.
- **`wilson_ci`** — стандартная формула, корректно.
- **Positive control fail-loud** — `assert pc_ok` до интерпретации null. ✓
- **SMOKE** не пишет артефакты. ✓
- **`aggregate_verified_only`** — exclude работает корректно.

---

### Сводка

| # | Проблема | Критичность | Fix |
|---|---|---|---|
| **A** | G0a_tpu_hn_field проверяет `‖u‖=1`, не `U*hn==h_raw` | **критично**: check не ловит то, для чего написан | сравнение `U*hn` vs raw h из npz |
| **B** | AR verdict из `D_center_l1` (sign-dependent); `D_bands_L1` не используется в verdict | средняя: документация vs реализация | explicit note или смена metric |
| **C** | `band_sums_direct` без `lo > hi` guard | мелкая (latent) | `if lo > hi: continue` |
| **D** | `sign_mismatch` на aggregate mean, не per-cell | мелкая | добавить `frac_cells_sign_mismatch` |
| **E** | `verdict` не различает verified vs full pool в headline | мелкая | добавить поле |
| **F** | gen-hn check в cell-цикле, не в gate | мелкая | вынести до цикла |
| **G** | `confirmed_anomaly` — pre-reg не описывает интерпретацию | документация | добавить |
| **H** | `D_bands_L1` без null distribution | документация | убрать из aggregate или добавить |

**A — самое важное**. Это не косметика: код **заявляет**, что проверяет hn-баг, но фактически проверяет только `‖u‖=1`, что не ловит hn-баг. Читатель артефакта видит `G0a_tpu_hn_field: unit-pin ok` и думает «hn проверены» — а они не проверены.

**B** — вердикт `refuted` для AR в текущих данных правдив (sign_mismatch=false на aggregate), но mechanism вычисления не совпадает с заявленной интерпретацией. Если в будущих прогонах (разные корпуса, разные модели) sign mismatch случится, `refuted` будет trivial. Стоит либо проверить per-cell `sign_mismatch`, либо использовать sign-safe metric для verdict.

**Остальные** — мелкие, но стоит закрыть.

**Одна мысль.** Обновление демонстрирует хороший паттерн: «критика принята → фикс + явная пометка r2-фикс». Но A показывает риск: исправление может быть **декларативным** (комментарий говорит одно, код делает другое). Правильный workflow для gate-checks: перед фиксом написать **тест**, который должен провалиться на текущем коде, а после фикса — пройти. Для hn-бага такой тест = «искусственно сломать hn, проверить, что gate падает». Сейчас gate на искусственно сломанном hn **не упадёт** (потому что `‖u‖=1` остаётся).