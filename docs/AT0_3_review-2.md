## Ревью `AT0_3_area.py` (r3.1)

### Что правильно улучшено (без оговорок)

**1. Тавтологичный `G0a_tpu_hn_field` → реальный `G0a_tpu_curves`.** Главная правка. Проверка `dip_L23(recomputed from U, hn) == meta_curves[23].dip` — реальный тест согласованности `load_source` с продюсером; `U*hn == h` устранён. Провенанс покрывается SHA-сверкой `match_AT0_1`. Документация честная: `r3_note` фиксирует, что прошлая проверка была тавтологией.

**2. `location_contrast` на K=96 pool.** `_fracs96(L96m, key)` фильтрует к тем же докам, что в primary KS. Full-pool + K-composition — secondary, явно. `ratio` guarded (`if k1l96 and k1l96 > 1e-6`).

**3. `H2` gate: `median_concentration > 1 ∧ frac_conc_above_1 ≥ 0.8`** вместо `min_concentration > 1`. Robust к одному выбросу; gate задокументирован в `verdict.H2_period_structure.gate`. `min` остался описательно.

**4. `prereg.sign_correction` / `prediction` помечены как исторические.** Явно `[ИСТОРИЧЕСКАЯ справка о правке v1.0→v1.1, НЕ актуальная гипотеза]` / `[Исходное предсказание v1.1; … это предсказание опровергнуто …]`. Актуальные тесты — в `test`/`decision`.

**5. `perm_two_sample_stat` docstring.** Теперь: «two-sided для signed (median_diff, mean); right-tail для unsigned (KS)». Явно указывает, что KS — не two-sided.

**6. `binom_pmf` convention.** «as-or-less-likely», не scipy binomtest (там doubled-min). Задокументировано в `pdet["test"]["note"]`.

**7. `stratum_summary` — `K_composition`.** Теперь видно K=32/96 состав стратумов, что делает comparison secondary-vs-primary прозрачным.

---

### Критический баг: `h1_short` ломает `kill` case

```python
h1_v = pof.get("verdict", "n/a")
h1_short = h1_v.split(":")[0].strip()
```

Для `kill`-verdict строка начинается с `"kill по пре-регу: распределения неразличимы..."`. `split(":")[0].strip()` даёт `"kill по пре-регу"`, **не** `"kill"`. Затем:

```python
h1_clauses.get(h1_short, 'n/a')
```

`"kill по пре-регу"` не в словаре (`keys = {"refuted", "kill", "confirmed"}`) → **`'n/a'`**. Так что `summary` содержит `… — n/a. G10: …` вместо осмысленного клауза. И `H1_period2.status = "kill по пре-регу"` — нестандартный словарь (сравните с `H2_period_structure.status = "confirmed"|"not_confirmed"`).

Для `confirmed`/`refuted` split работает (нет пробелов в префиксе). Только `kill` ломается — тот самый, который актуален в текущих данных (KS p ~0.965).

**Фикс**:
```python
if h1_v.startswith("confirmed"):   h1_short = "confirmed"
elif h1_v.startswith("refuted"):   h1_short = "refuted"
elif h1_v.startswith("kill"):      h1_short = "kill"
else:                              h1_short = "other"
```

Или `h1_short = h1_v.split(":")[0].split()[0].lower()`.

---

### Существенное: `summary` — TypeError при `None` значениях

```python
f"... clean — на лаге 1 (k1 {k1c96:.3f} vs loopy {k1l96:.3f}{ratio_txt}) ..."
```

Если `k1l96 = None` (нет loopy с K=96 и отриц. площадью), `f"{None:.3f}"` — **TypeError**. Сейчас в данных есть 39 K=96 loopy с `k1_frac > 0`, так что не падает. Но `_fracs96` возвращает `None` для пустого списка — latent crash.

**Фикс**:
```python
def _fmt(x): return f"{x:.3f}" if x is not None else "n/a"
f"... k1 {_fmt(k1c96)} vs loopy {_fmt(k1l96)}{ratio_txt}"
```

---

### Существенное: `concentration_test` — один и тот же seed для всех доков

```python
def concentration_test(rho, p, K, b=B_PERM, seed=SEED_PERM):
    ...
    rng = np.random.default_rng(seed)
```

Все per-doc permutation-тесты используют `seed=117`. PRNG state идентичен на входе, дивергенция после первого `rng.choice(K, n_neg, ...)` — но для документов с одинаковыми K и `n_neg` первый choice может дать **коррелированные** результаты. 39 loopy docs с одинаковым seed → потенциальная cross-doc корреляция нулевых распределений. Это нарушает «per-doc независимость» null'а. В других местах серия использует `SEED_BASE + doc_idx * SEED_STRIDE`; здесь конвенция нарушена.

**Фикс**: добавить `doc_seed` параметр:
```python
def concentration_test(rho, p, K, b=B_PERM, seed=SEED_PERM, doc_seed=0):
    rng = np.random.default_rng(seed + doc_seed)
```
И в `per_doc`: `concentration_test(..., doc_seed=hash(nm) % 99991)` или через `doc_idx` (строкой идёт первая шкала). Согласуется с `null_ar_batch`/`run_cell` (там `SEED_BASE + doc_idx*SEED_STRIDE + s`).

Влияние: p-values per-doc могут быть слегка correlated. При `B=1000` и `n_neg ~ 20` первый choice даёт `C(K, n_neg)` вариантов — после первого шага PRNG diverges. Скорее всего minor, но это **методологическая ошибка** — null должен быть независим.

---

### Среднее: `pagg.get('median_concentration', float('nan'))` — «nan» в тексте

```python
"finding": (f"... (dom_residues; медиана концентрации "
            f"{pagg.get('median_concentration', float('nan')):.2f}× базы) — ...")
```

Если `pagg` пуст (нет detected loopy K=96), `.get()` возвращает `nan`, форматирование даёт `"nan×"` в finding. Cosmetic, но сбивает. **Фикс**: `f"{pagg['median_concentration']:.2f}" if pagg.get('median_concentration') is not None else "n/a"`.

---

### Среднее: `H2_period_structure.status = "not_confirmed"` для случая «нет теста»

```python
h2_ok = bool(pdet.get("test", {}).get("p_two_sided_exact", 1) <= ALPHA)
if h2_ok and pagg:
    h2_ok = (pagg.get("median_concentration", 0) > 1
             and pagg.get("frac_conc_above_1", 0) >= 0.8)
```

Если `pdet["test"]` отсутствует (не сработал `if 0 < d < n`, например при `d = n` — все detected), `.get("p_two_sided_exact", 1)` = 1 > ALPHA → `h2_ok = False` → status `"not_confirmed"`. То есть **отсутствие теста** интерпретируется как **отрицательный результат**. Для current data `d < n` (часть clean не detected), так что не срабатывает. Но edge case.

**Фикс**: различить:
```python
if "test" not in pdet:
    h2_status = "not_testable (test не определено: 0<d<n не выполнено)"
elif not h2_ok_gate:
    h2_status = "not_confirmed"
else:
    h2_status = "confirmed"
```

---

### Мелкие

**1. `binom_pmf` при underflow** (`pmf_obs = 0`): sum of 0s → `p_two = 0`, что неверно. Для n=39 не критично (не underflow), но при больших n — будет. Задокументировать.

**2. `stratum_summary` — `K_composition` только для доков с negative area.** Не показывает K=32/96 для absent/no-area docs. Для текущих данных `n_noarea = 0` в обеих стратумах, но edge case невиден.

**3. `binomial test` — `p_two` при `dL=0`** требует `pmf_obs = binom_pmf(0, nL, p0)` — OK, но если `nL` мало, граница точная.

**4. `verdict.H2_period_structure.gate` — задокументирован, но без `note`** о том, что `min_concentration` — secondary, не gate. `pagg["min_concentration"]` в aggregate, не в gate. Читатель артефакта увидит `min=0.85` при `status=confirmed` — если не знает, что gate на `median`, будет сбит.

---

### Сводка

| # | Замечание | Критичность |
|---|---|---|
| **1** | `h1_short = h1_v.split(":")[0].strip()` — `"kill по пре-регу"` вместо `"kill"`; `h1_clauses.get` возвращает `'n/a'`; `H1_period2.status` нестандартный | **критично** — ломает `kill`-case в summary и вердикте |
| **2** | `summary` `{k1l96:.3f}` — TypeError при `None` | средняя — latent crash |
| **3** | `concentration_test` — same seed для всех docs | средняя — методологическая |
| **4** | `H2_period_structure.status = "not_confirmed"` при `test` absent | средняя — edge case |
| **5** | `pagg.get(..., nan)` → `"nan×"` в finding | косметика |
| **6** | `H2_period_structure.gate` — без note о min-secondary | косметика |
| **7** | `binom_pmf` underflow при больших n | задокументировать |

**Главное**: #1 — реальная **регрессия** r3.1: `kill`-case (главный в текущих данных) сломан. `summary` содержит `— n/a` вместо осмысленного клауза. Починка — 4 строки.

**После фикса #1**: скрипт **готов** — все критические замечания прошлых ревью адресованы, тавтология удалена, пулы согласованы, gate задокументирован. #2–#4 — желательные, но не блокеры для текущих данных (не срабатывают).

**Итоговая рекомендация**: исправить #1 (`startswith`-based dispatch), #2 (`_fmt` для None), #3 (`doc_seed` в `concentration_test`). Перезапуск не требуется — можно поправить `verdict.summary` и `H1_period2.status` в claim-артефакте постфактум (числа верные, сломана только текстовая диспетчеризация).