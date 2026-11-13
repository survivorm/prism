## Ревью обновлённого `AT0_3_area.py` (r3)

### Что правильно улучшено (без оговорок)

- **`LAYER_TYPE` сверен с config** (`G15_layer_types`) — hardcoded `(11, 23)` валидируется по `config.text_config.layer_types`. Не тавтология: если конфиг изменится, gate упадёт fail-loud.
- **`copy_rate` off-by-one** исправлен (`[3:383]` вместо `[4:383]`).
- **`neg_fractions` floor** `-1e-9` вместо `0.0` — правильная защита от fp-шума.
- **KS-тест как primary** (r3 §3.2) — корректная реакция на «одинаковое распределение» из пре-рега (mean был слеп к бимодальности).
- **`period_structure` (H2)** — новый тест на периодическую структуру, ортогональный H1. Включает per-doc перестановочный `concentration_test`, агрегаты `mean/median/min_concentration`, `dom_residues`, `anti_frac`. Это правильное дополнение к KS.
- **`forced_outcome` для v_tangent** — `b1>0.8 ⇒ odd_frac≡1` пойман и заблокирован от цитирования.
- **`verdict` разложен на H1/H2 + confounds + location_contrast + summary** — структура честная.
- **`G10_forced_outcomes`** — задокументировано, что `odd_frac` слеп к p≥3 (периоды лупов 5/7/8/24).

---

### Критичное: `G0a_tpu_hn_field` в AT0_3 — **тавтология**

```python
for L in LAYERS:
    h = np.asarray(z[f"{nm}_L{L}"], dtype=np.float64)
    u, hn = d["layers"][L]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    dev = float(np.max(np.abs(U * hn[:, None] - h) / np.maximum(np.abs(h), 1e-12)))
```

`u` и `hn` возвращены `load_source` из `h` этого же `z`. В `load_source` (`AT0_1_accounting.py`):
```python
hn = np.linalg.norm(h, axis=1)   # = ‖h_row‖
u = h / hn[:, None]              # = h/‖h‖
```

Тогда `‖u‖ = 1`, `U = u`, `U * hn = h` — **точно**, до float64 (~1e-15). Проверка проходит на **любых** h_npz, включая сломанные. `max_rel_dev ≈ 1e-16`, `ok=True` всегда.

**Это fake check.** Настоящая валидация — `match_AT0_1: {k: g0_sha[k] == at01_shas[k]}` — сверка SHA с AT0_1 (где уже была валидация row-reconstruction). Если SHA совпал, source идентичен AT0_1, и `U*hn == h` **не даёт новой информации**.

**Фикс** (один из):
- (a) Удалить `tpu_dev`-цикл, заменить `"live_check"` на `"по match_AT0_1: {match: true} — источник верифицирован через SHA сверку"`. Провенанс полностью покрыт SHA-сверкой.
- (b) Оставить, но сделать **нетривиальным**: проверить `dip_recomputed_from_h == dip_from_meta_curves` (как в AT0_1 `gate_g0a_tpu`). Это проверяет, что h — действительно то, из чего строились curves.

Сейчас `G0a_tpu_hn_field` **декларирует** проверку hn-поля, но **проверяет тавтологию**. Это ровно та проблема, которую AT0_2 ранее отозвал в другом месте (`unit`-пин, не hn-чек).

---

### Существенное: `location_contrast` смешивает пулы

```python
k1l = _med(strata[PRIMARY_OBJ]["loopy"], "k1_frac")
```

`strata[PRIMARY_OBJ]["loopy"]` = `stratum_summary(cells, (obj, PRIMARY_L), L)` на **полном пуле L = 41 док**, не на K=96.

`k1_frac = neg[0] / A_neg`, где `A_neg = Σ_{k=1..K} min(rho(k), 0)`. Для K=32 доков A_neg использует **только 32 лага**, для K=96 — 96. Знаменатель разный ⇒ `k1_frac` разный **для одного и того же дока при разных K**.

При этом primary-test (KS) использует **только K=96 доков** (`k96(L)` фильтр). Значит:

- KS p-value → K=96 pool (≈39 loopy, 12 clean?)
- `location_contrast` → полный pool (41 loopy, 53 clean)

**Если K-состав loopy и clean разный**, цифры `k1l = 0.076` vs `k1c = 0.336` могут быть частично артефактом K-гетерогенности, а не свойством loopy/clean.

Из AT0_1 P4: `pool4_n = 51` из 53 nondeg — только 2 дока K=32. Значит **почти все в K=96 пуле**. Влияние невелико, но не показано.

**Фикс**:
1. `location_contrast` на **K=96 pool** (те же доки, что в `obj_test`), или
2. Явно отчитаться K-составом каждого стратума, чтобы читатель видел, что разница минимальна.

Сейчас в `location_contrast` есть `"pool": "strata (полный пул 41/53, не K=96)"` — это honest disclosure, но **без последствия** (что k1_frac разный при разном K).

---

### Существенное: `period_structure` detection-rate на смешанном пуле

```python
def per_doc(nm):
    ...
    K = c["K"]
    p = detect_period(rho, K)
```

`detect_period` ищет `p ∈ [2, K//2]`:
- K=32 → detects p ∈ 2..16
- K=96 → detects p ∈ 2..48

**Больше лагов = больше шанс детекции** (и false positives, и true positives). Detection rate на смешанном пуле confounded by K-distribution. Если loopy имеет больше K=32 доков, чем clean, detection rate ниже у loopy — **противоположно ожиданию**.

Затем `binom_pmf`-тест на `dL` vs `dC` предполагает независимость детекций — но если K разный, вероятности детекции per-doc разные, что нарушает `H0: p_detect одинаковый`.

**Фикс**:
- Фильтровать `period_structure` к K=96 pool (`k96(L)` и `k96(C)`), или
- Отчитываться detection rate per K (K=96: x/39 loopy; K=32: y/2 loopy), и тест — только на K=96.

Также **документировать K-состав стратумов в `results["strata"]`** — сейчас `stratum_summary` не возвращает K-распределение.

---

### Существенное: `H2_period_structure` gate через `min_concentration > 1` — fragile

```python
if h2_ok and pagg:
    h2_ok = pagg.get("min_concentration", 0) > 1
```

`min_concentration = min(concentration across detected loopy)`, где `concentration = max_share * p` — выше 1 означает «выше uniform».

Если **один** loopy-док имеет `concentration = 0.99` (чуть ниже uniform) при median 5, весь H2 становится "not_confirmed".

**Фикс**: `median_concentration > 1` или `frac_above_baseline ≥ 0.8` (робстнее к выбросам). Или явно в пре-реге прописать, что gate — "min" (строгий критерий).

Сейчас `median` считается и отчитывается (`median_concentration`), но **не используется в gate**. Nonsensical: зачем считать, если не проверяется.

---

### Существенное: prereg `sign_correction` — stale

```
"sign_correction": "период-2 ρ(k)=(−1)^k ⇒ отриц. площадь на нечётных k (осн. k=1)..."
```

Это описание **исходного** H1 предсказания (loopy → odd k). Но r3 **переопределил** primary test как KS на распределения, а H1 предсказание "loopy odd" проверяется через `location_contrast` (где loopy НЕ на k=1). Плюс H2 (периоды) показывает, что loopy не на k=1, а на фазе dom_residues.

Поле `sign_correction` в артефакте **читается как актуальный пре-рег**, но на самом деле это **историческая справка** о правке v1.0→v1.1. Читатель, не знающий контекста, может подумать, что `sign_correction` = текущая гипотеза.

**Фикс**: переименовать в `historical_note` или добавить в начало: `"[историческое; актуальный тест — см. test/decision]"`.

Аналогично `prediction: "loopy: отриц. площадь на нечётных k; clean: в хвосте k≥9"` — это **исходное** предсказание. По нему verdict = `refuted` (loopy на хвосте, не на нечётных). Что корректно отражено в `verdict.H1_period2.status`. Но в `prereg.prediction` нет пометки, что эта prediction **провалена** — можно подумать, что она актуальна.

---

### Среднее: `perm_two_sample_stat` docstring путает

```python
"""... p = (1 + #{|stat_perm| >= |stat_obs|})/(b+1), two-sided."""
```

Для **signed** statistic (median_diff) формула `|s_perm| ≥ |s_obs|` — двусторонняя, верно.
Для **unsigned** statistic (KS) — `|s_perm| = s_perm`, `|s_obs| = s_obs`, формула = right-tail `s_perm ≥ s_obs`, что и есть KS-тест. **Не two-sided**.

Код правильный, но docstring говорит "two-sided" для обоих. Для KS это **не two-sided** — это один-хвостовой right-tail. Путает читателя.

**Фикс**: `"two-sided for signed stat (median_diff), right-tail for unsigned stat (KS)"`.

---

### Среднее: `binom_pmf` two-sided convention

```python
pmf_obs = binom_pmf(dL, nL, p0)
p_two = sum(binom_pmf(i, nL, p0) for i in range(nL + 1)
            if binom_pmf(i, nL, p0) <= pmf_obs + 1e-15)
```

Это **"sum of outcomes as or less likely than observed"**. Валидная, но **нестандартная** convention. Другие стандарты:
- (a) `2 * min(P(X ≤ dL), P(X ≥ dL))`, capped at 1
- (b) `Σ_{i: |i - E| ≥ |dL - E|} pmf(i)`

Все три дают разные числа, особенно при малых n и extreme proportions. Без явной пометки читатель может сравнить с scipy `binomtest` — там convention (a), и числа будут разные.

**Фикс**: явно в note: `"двусторонний через вероятность (as-or-less-likely), не scipy binomtest (там doubled-min convention)"`.

---

### Мелкие

**1. `k1c / k1l` ratio без guard.** Если k1l → 0, ratio → ∞. Отчёт `{k1c / k1l:.1f}×` может выдать огромные числа. Сейчас k1l = 0.076 (безопасно), но общий guard: `if k1l > 1e-6 else None`.

**2. `results["strata"]["v_tangent"]["clean"]["odd_frac"]["median"] = 1.0`** — чисто потому что `b1 ≈ 1`, и `odd_frac = neg[0]/A_neg`. Это **forced**, не находка. В `verdict` и `forced_outcome` помечено, но в `results["strata"]` — просто median = 1.0 без пометки. Стоит добавить `"forced_outcome": true` в `stratum_summary` для v_tangent clean.

**3. `verdict.summary` — интерпретационное расширение**

> "лупы размазывают плату периодически по хвосту (на фазе границы лупа)"

"на фазе границы лупа" — интерпретация `dom_residues`. Данные показывают, что mass concentrate на `dom_residues mod p`, но не идентифицируют это как "loop boundary". Честнее: "концентрация A_neg на фазе dom_residues mod p (интерпретация как loop boundary — гипотеза, не результат)".

**4. `G0a_provenance.corpus` — известный confound, документирован, но не исправлен**

Корпус (16 доков) включён в clean (C), т.к. `degen()` для corpus срабатывает только по `mcos` (`mshare` недоступен без токенов). Конфаунд L/C × источник объявлен в `confounds`. **Мера влияния не оценена** (source-strata описательно, n<12). Не баг, но результат слабее, чем звучит.

**5. `period_structure` использует `L`, `C` (полный пул)** — уже отмечено выше. `stratum_summary` и `_k96` не согласованы.

**6. `per_doc_conc_p` — tuple `(p, p_perm)`** — читатель должен знать, что второй элемент — p-value concentration test. Имя `p_perm` двусмысленно. OK, но нужно в артефакте явно написать (`"per_doc_conc_p_format": "(p_detected, p_value_concentration)"`).

---

### Сводка

| # | Замечание | Влияние |
|---|---|---|
| 1 | **`G0a_tpu_hn_field` тавтологичен** (`U*hn == h` тождественно) | **fake check**, не валидирует |
| 2 | `location_contrast` смешивает пулы (K=96 KS vs full-pool medians) | умеренная — confounded K |
| 3 | `period_structure` detection rate на смешанном K-пуле | confounded K |
| 4 | `H2` gate = `min_concentration > 1` (fragile to outliers) | не переоценивать |
| 5 | `prereg.sign_correction` / `prediction` — stale (v1.0) | misleading |
| 6 | `perm_two_sample_stat` docstring "two-sided" для KS | misleading |
| 7 | `binom_pmf` convention не помечена | сравнения с scipy сломаются |
| 8 | `k1c/k1l` без guard при k1l→0 | latent |
| 9 | `verdict.summary` — "фаза границы лупа" интерпретативна | overreach |

**Первоочередное**:

1. **`G0a_tpu_hn_field`** — либо удалить, либо заменить на реальную проверку (`dip_recomputed == dip_meta`). Сейчас это **самый серьёзный дефект**: gate выглядит как проверка hn, но валидирует тождество.
2. **Пул в `location_contrast`/`period_structure`** — привести к K=96 (или явно отчитаться K-составом).
3. **`prereg`** — пометить `sign_correction`/`prediction` как "историческое/опровергнутое", не актуальное.

**Всё вместе**: `results["verdict"]` в целом правильный (H1 refuted, H2 confirmed, location различена в хвост/на лаг-1), но:

- H1 refuted подкреплён **KS p=0.965** (нулевой), не p<0.05. То есть **распределения неразличимы**, не «направление противоположно». Это **`kill`**, не `refuted`. `refuted` = «различимо, направление против» — но в текущем коде `refuted` вызывается при `KS p≤0.05 AND median L<C`. При KS p=0.965 → **`kill`**.

Проверяю логику:
```python
elif ks["p"] <= ALPHA:
    if of["median_L"] > of["median_C"]:
        of["verdict"] = "confirmed..."
    else:
        of["verdict"] = "refuted..."
else:
    of["verdict"] = "kill по пре-регу..."
```

По run.log: `KS d=0.1416 p=0.9650` — значит **`kill`**, не `refuted`. Соответствует коду.

**Но** в моём чтении `verdict.summary` я мог неверно прочитать. Проверю: `"H1_period2": {"status": pof.get("verdict", pt.get("verdict")), ...}` — берёт из `pof["verdict"]`, что при KS p=0.965 → `kill по пре-регу`. Значит summary = "kill". OK. Моё предыдущее замечание про "refuted" неверно — надо перепроверить.

Перечитываю. Ага, `h1_short = h1_v.split(":")[0]` → `"kill по пре-регу"`. И summary начинается:
```python
h1_lead = (f"Период-2 как носитель отриц. площади: подтверждён" if h1_short == "confirmed"
           else f"Период-2 как носитель отриц. площади: не подтверждён")
```

При `h1_short = "kill"` → `h1_lead = "не подтверждён"`. Тогда **`not confirmed`**, не "refuted". Верно.

Но `H1_period2.status` в `verdict`:
```python
"H1_period2": {
    "status": pof.get("verdict", pt.get("verdict")),
    ...
}
```

`pof["verdict"]` = полный текст `"kill по пре-регу: распределения неразличимы (KS p>0.05); период-2 не подтверждён; G10: estimand слеп к периодам p≥3..."`. Значит `status` = длинная строка, не короткое `kill`. Читатель, желающий видеть `kill | refuted | confirmed`, увидит полный текст. Не bug, но inconsistent с другими вердиктами в `verdict`.

**Действительно**: раздел `verdict` в артефакте разный по структуре для H1 (`status` = полный текст) и H2 (`status` = короткое `"confirmed"|"not_confirmed"`). Стоит унифицировать.

---

### Окончательный вердикт по скрипту

**Скрипт лучше предыдущей версии**, но содержит **один критичный fake check** (`G0a_tpu_hn_field`) и **несколько путаниц в пулах** (K=96 vs full), которые могут усилить/ослабить выводы при малых различиях. Основной вывод (H1: period-2 не подтверждён; H2: периодическая структура у loopy подтверждена) — **корректен по данным**, но `location_contrast` (k1/tail) требует K-фильтрации для точной сопоставимости с KS.

**Наиболее серьёзный фикс**: убрать тавтологичный `G0a_tpu_hn_field` (или заменить на реальную сверку `dip_recomputed == dip_meta` из meta-файла). Затем — привести `location_contrast` и `period_structure` к K=96 пулу (или отчитаться K-составом стратумов). `prereg` — пометить stale-поля.

**Все фиксы — текстовые/логические правки в артефакте**, перезапуск не обязателен, если пулы уже фильтруются в `obj_test` и можно просто пересчитать medians на K=96 подмножестве. Но честнее — перезапуск с правильными пулами.