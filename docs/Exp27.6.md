# ТЗ: Эксперимент `Exp 27.6`
## Domain isolation, block structure, and null-baselines for the Exp 27.5 verdicts

---

## 1. Название и статус

```text
Exp 27.6: P0 addendum to Exp 27.5
```

Рабочее имя:

```text
exp27_6_isolation.py
```

**Статус:** постобработка кэша `exp27_jcache.pt` и уже сохранённых `exp27_pair_matrix_S64K64p0.json`, `exp27_5_postprocess.json`. GPU не требуется. Все вычисления — ≤ 30 минут CPU.

**Зачем:** Exp 27.5 закрывает статистику и кластеризацию, но оставляет открытыми шесть P0-дыр из прошлого раунда, без которых вердикты «домен не работает», «кластеров нет», «ARI ≈ 0» **не защищены**. Exp 27.6 закрывает только эти шесть.

---

## 2. Список задач (по приоритету)

| # | Задача | Откуда | Время |
|---|---|---|---|
| T1 | Domain isolation: T_domain без math-домена | §3.1 | 2 мин |
| T2 | Bock 4×4 на K^G | §3.2 | 3 мин |
| T3 | Bootstrap CI на T_domain и T_ent | §3.3 | 5 мин |
| T4 | Permutation-null для ARI | §3.4 | 3 мин |
| T5 | Dip p-value / bootstrap-null | §3.5 | 5 мин |
| T6 | Eigengap спектра K^F, K^G | §4.1 | 2 мин |
| T7 | Spearman ρ(layer, C_a) с bootstrap-CI | §4.7 | 2 мин |
| T8 | Per-sample r_eff(J_i), ‖h_i‖₂ | §4.4 | 5 мин |

T8 — опционально, но дёшево и даёт attribution-поле для Exp 27.5 §8.2.

---

## 3. Входные данные

```
results/exp27_jcache.pt                          # JR, H, P128, U128
results/exp27_pair_matrix_S64K64p0.json          # 32×32 K^F per layer
results/exp27_5_postprocess.json                 # ari, dip, halfsplit
```

Основной срез для всех задач — **S=64, K=128, pos=0**, если не указано иное. Совпадает с main slice Exp 27.5.

---

## 4. T1: Domain isolation (без math)

### 4.1. Постановка

В Exp 27.5 метрика

\[
T_{\text{domain}} = C_{\text{within}} - C_{\text{cross}}
\]

считается на всех 4 доменах. Если math-math пары систематически ниже (что видно из `per_domain.mean` в pair_matrix JSON: math 0.038 при code 0.108 на L22), то `C_within` смещён вниз, а `T_domain` завышен или занижен в зависимости от того, как math-other пары ложатся в `C_cross`.

### 4.2. Определения

Пусть \(\mathcal D = \{w, m, c, h\}\) — 4 домена. Для подмножества \(\mathcal D' \subset \mathcal D\):

\[
C_{\text{within}}^{\mathcal D'} = \frac{1}{|P_{\text{within}}^{\mathcal D'}|} \sum_{(i,j) \in P_{\text{within}}^{\mathcal D'}} C_{ij}
\]

\[
C_{\text{cross}}^{\mathcal D'} = \frac{1}{|P_{\text{cross}}^{\mathcal D'}|} \sum_{(i,j) \in P_{\text{cross}}^{\mathcal D'}} C_{ij}
\]

\[
T_{\text{domain}}^{\mathcal D'} = C_{\text{within}}^{\mathcal D'} - C_{\text{cross}}^{\mathcal D'}
\]

Считаем для **четырёх** вариантов:

- \(\mathcal D' = \{w, m, c, h\}\) — baseline (совпадает с Exp 27.5)
- \(\mathcal D' = \{w, c, h\}\) — **без math**
- \(\mathcal D' = \{w, m, c\}\) — без chat
- \(\mathcal D' = \{w, m, h\}\) — без code

Все четыре обязательны, чтобы различить «math выброс» от «какой-то домен выброс».

### 4.3. Метрика на K^G (не только K^F)

Повторить для K^G = cos(J_i J_i^T, J_j J_j^T) — потому что Exp 27.5 §9 работает именно с ней, и isolation-вердикт должен быть согласован с кластеризацией.

Для каждой матрицы (K^F, K^G) — свой T_domain.

### 4.4. Deliverable

```json
"T1_isolation": {
  "6": {
    "K_F": {"all": 0.0, "no_math": 0.0, "no_chat": 0.0, "no_code": 0.0},
    "K_G": {"all": 0.0, "no_math": 0.0, "no_chat": 0.0, "no_code": 0.0}
  },
  ...
}
```

### 4.5. Gate

- Если \(T_{\text{domain}}^{\text{no-math}} \ge 0.5 \cdot T_{\text{domain}}^{\text{all}}\) на всех слоях — math не доминирует, вердикт Exp 27.5 устойчив.
- Если \(T_{\text{domain}}^{\text{no-math}} < 0.2 \cdot T_{\text{domain}}^{\text{all}}\) хотя бы на одном слое — предыдущий вывод о «слабой доменной оси» артефакт math. Записать явно.

---

## 5. T2: Bock 4×4 на K^G

### 5.1. Постановка

`per_domain.mean` в pair_matrix JSON даёт **диагональ** блока (d, d). `cross_domain_mean` — одно число на все 6 off-diagonal блоков. Нужна полная 4×4 матрица средних.

### 5.2. Определение

\[
\bar K^G_{d_1 d_2} = \frac{1}{64 \cdot 64} \sum_{i \in d_1} \sum_{j \in d_2} K^G_{ij}
\]

при \(d_1 \ne d_2\). Для \(d_1 = d_2\) — использовать существующие `per_domain.mean`.

Все 16 ячеек на каждый слой. Диагональ симметрична по построению.

### 5.3. Deliverable

```json
"T2_bock": {
  "6": {
    "matrix": [[0.0063, 0.0, 0.0, 0.0],
               [0.0, 0.0069, 0.0, 0.0],
               [0.0, 0.0, 0.0161, 0.0],
               [0.0, 0.0, 0.0, 0.0109]],
    "domains": ["wiki", "math", "code", "chat"]
  }
}
```

### 5.4. Gate

- Диагональ \(K^G_{dd}\) должна быть > off-diagonal в среднем, если домены работают.
- Если \(\bar K^G_{mm} \ll \bar K^G_{dd}\) для всех \(d\) и \(\bar K^G_{md} \ll \bar K^G_{d_1 d_2}\) для \(d_1, d_2 \ne m\) — math изолирован. Это ожидаемый исход, и он **должен быть явно записан**.

---

## 6. T3: Bootstrap CI на T_domain и T_ent

### 6.1. Постановка

p_perm в Exp 27.5 §7.2 отвечает на «отличимо ли T от нуля при случайной разметке». CI отвечает на «насколько точно измерено T». Нужно и то и другое.

### 6.2. Алгоритм

Bootstrap по 32 сэмплам:

1. Ресэмплировать индексы \(\mathcal I^{(b)} \subset \{1, \dots, 32\}\) с возвращением, \(|\mathcal I^{(b)}| = 32\).
2. Построить \(C^{(b)}\) как подматрицу \(C[\mathcal I^{(b)}, \mathcal I^{(b)}]\).
3. Пересчитать \(T_{\text{domain}}^{(b)}\) по реальным доменным меткам.
4. \(B = 1000\) итераций.
5. Отчёт: mean, p2.5, p50, p97.5.

То же для \(T_{\text{ent}}\) (§8.2 Exp 27.5) с квартильными метками.

Важно: ресэмплировать **сэмплы**, не пары. Иначе пул пар внутри бутстрепа скоррелирован (пара (i, j) и (i, k) делят i).

### 6.3. Deliverable

```json
"T3_bootstrap": {
  "6": {
    "T_domain": {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_valid_boot": 998},
    "T_ent": {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
  }
}
```

### 6.4. Gate

- Если CI на T_domain включает 0 на всех слоях — доменная ось статистически неотличима от нуля при N=32. Записать явно.
- Если CI не включает 0, но ширина CI ≥ |mean| — значимо, но слабо (small effect). Отметить как предварительное.
- Если CI не включает 0 и ширина CI ≤ 0.5 · |mean| — сильный результат.

---

## 7. T4: Permutation-null для ARI

### 7.1. Постановка

В `exp27_5_postprocess.json` есть `ari: 0.007…0.029`. Это ARI между кластерами и доменными метками. Без null-модели нельзя сказать, «мало» это или «на уровне случайности».

### 7.2. Алгоритм

1. Зафиксировать кластерные метки \(\pi_K\) (например, из Leiden с фиксированным seed; в Exp 27.5 кластеризация уже сделана).
2. \(B_{\text{perm}} = 1000\) раз перемешать доменные метки \(\sigma^{(b)}\).
3. Пересчитать \(ARI(\pi_K, \sigma^{(b)})\).
4. Отчёт: null-mean, null-p95, real-ARI.
5. p-value: доля \(ARI^{(b)} \ge ARI_{\text{real}}\).

**Важно:** не пересчитывать кластеризацию при каждом бутстрепе. Метки \(\pi_K\) фиксированы, перемешиваются только домены.

### 7.3. Deliverable

```json
"T4_ari_null": {
  "6": {"ari_real": 0.007, "ari_null_mean": 0.001, "ari_null_p95": 0.015, "p_value": 0.32}
}
```

### 7.4. Gate

- Если `p_value ≥ 0.05` на всех слоях — ARI неотличим от случайной разметки. Это формальный вердикт: **доменные метки не выравнены с кластерной структурой**.
- Если `p_value < 0.05` хотя бы на одном слое — есть слабый доменный сигнал, требующий проверки на K^G (Exp 27.5 §9).

---

## 8. T5: Dip p-value / bootstrap-null

### 8.1. Постановка

В `exp27_5_postprocess.json` есть `dip: 0.04…0.25`. Это статистика Хартигана. Без критического значения или p-value интерпретации нет.

### 8.2. Вариант A: нормативное значение

Для N=496 и однородного распределения критическое значение dip ≈ 0.02–0.03 (литературные данные для Hartigan dip на непрерывных распределениях, n=500). Если `dip > 0.05`, гипотеза унимодальности отвергается.

Порог 0.05 применяется ко всем значениям dip из JSON.

### 8.3. Вариант B: bootstrap-null

Если хочется свой null для конкретных данных:

1. Взять эмпирическое распределение \(K^G\) off-diagonal значений.
2. Сгенерировать \(B = 500\) синтетических выборок размера 496 из **одномодального** распределения с тем же mean и std (например, Beta или lognormal, подобранная по первым двум моментам).
3. Для каждой посчитать dip.
4. p-value = доля \(dip^{(b)} \ge dip_{\text{real}}\).

Вариант B дороже, но не зависит от предположений о нормативном значении.

### 8.4. Deliverable

```json
"T5_dip": {
  "6": {
    "S64_pos0": {"dip": 0.071, "dip_null_p95": 0.048, "reject_unimodal": true},
    "S128_mid32_33": {"dip": 0.047, "dip_null_p95": 0.048, "reject_unimodal": false}
  }
}
```

### 8.5. Gate

- Все значения `dip > 0.05` на S=64, pos=0 — bimodality умеренно поддержана. Проверить, что пик dip сдвинут к тому же k, что и silhouette (Exp 27.5 §9.3).
- Если dip на S=128, mid32_33 ниже порога — bimodality может быть артефактом выбора среза.

---

## 9. T6: Eigengap спектра

### 9.1. Постановка

Silhouette в Exp 27.5 §9.3 может вернуть `S_max = 0.15` при k=3, но без eigengap нельзя сказать, есть ли вообще после λ₁ заметный gap. Если спектр плоский — silhouette меряет шум.

### 9.2. Алгоритм

Для каждой матрицы \(K^F_{ij}\) (и отдельно \(K^G_{ij}\), если Exp 27.5 §9 работает на ней):

1. Симметризация: \(\tilde K = (K + K^T)/2\) (в pair_matrix уже симметрична, но проверить).
2. Eigendecomposition: \(\tilde K = V \Lambda V^T\).
3. Отсортировать λ по убыванию.
4. Отчёт:
    - λ_1, λ_2, …, λ_31
    - eigengap ratios: \(g_i = \lambda_i / \lambda_{i+1}\) для \(i = 1, \dots, 5\)
    - effective rank: \(r_{\text{eff}}(K) = \exp(-\sum_i \hat\lambda_i \log \hat\lambda_i)\)
5. **Критерий:** существует ли \(i \in \{2, 3, 4\}\) с \(g_i \ge 1.5\)?
    - Да: кандидат на \(k = i\).
    - Нет: кластерной структуры в спектре нет, silhouette/gap пропускаем.

### 9.3. Deliverable

```json
"T6_eigengap": {
  "6": {
    "lambda_top": [0.95, 0.42, 0.31, 0.28, 0.15, ...],
    "gap_ratios": {"2/3": 1.35, "3/4": 1.11, "4/5": 1.87},
    "argmax_gap": 4,
    "r_eff_K": 8.4,
    "has_structure": false
  }
}
```

### 9.4. Gate

- Если `has_structure = false` на всех слоях — Exp 27.5 §9 можно сократить до отчёта «silhouette не считается, потому что спектр плоский». Это **сильный отрицательный результат**.
- Если `has_structure = true` на поздних слоях — silhouette/gap пересчитать только на них. Ранние слои закрыть.

---

## 10. T7: Spearman ρ(layer, C_a) с bootstrap-CI

### 10.1. Постановка

C_a(pos-pk) растёт от 0.077 (L6) до 0.22 (L22), см. `halfsplit` из `exp27_5_postprocess.json` (`mean_ci` 0.013 → 0.029; фактически C_a растёт в 3×). Это монотонный тренд, но без CI и без p-value при n=6 нельзя отличить «монотонность» от «малая выборка».

### 10.2. Алгоритм

1. Взять 6 пар (layer, C_a) на S=64, K=64, pos=0 (совпадает с main slice Exp 27.5).
2. Spearman ρ(layer, C_a).
3. Bootstrap по слоям: \(B = 1000\) раз ресэмплировать 6 точек с возвращением, пересчитать ρ.
4. p-value: permutation-null. Перемешать порядок C_a среди 6 слоёв, \(B = 1000\), доля \(|\rho^{(b)}| \ge |\rho_{\text{real}}|\).

### 10.3. Deliverable

```json
"T7_depth_spearman": {
  "C_a_pospk": {"rho": 0.94, "ci_lo": 0.6, "ci_hi": 1.0, "p_perm": 0.002},
  "C_a_samepk": {"rho": 0.89, "ci_lo": 0.4, "ci_hi": 1.0, "p_perm": 0.005},
  "C_a_raw": {"rho": -0.31, "ci_lo": -0.9, "ci_hi": 0.7, "p_perm": 0.58}
}
```

### 10.4. Gate

- Если CI на ρ для C_a(pos-pk) ≥ 0.5 — монотонный тренд защищён, даже при n=6. Это формализует вывод «когерентность растёт с глубиной» из Exp 27 §7.3.
- Если CI включает 0 — тренд неотличим от шума. Записать явно.

---

## 11. T8: Per-sample ковариаты для attribution

### 11.1. Постановка

Exp 27.5 §8.2 хочет attributions кластеров (если найдутся) к энтропии, r_eff, ‖J‖. Сейчас в JSON есть только агрегаты. Нужны per-sample значения для всех 32 сэмплов × 6 слоёв.

### 11.2. Список метрик

Из `exp27_jcache.pt`, срез S=64, K=128, pos=0:

| Метрика | Формула | Источник |
|---|---|---|
| E_i | ‖J_i‖²_F | JR |
| ℓ_i | log(E_i + ε) | JR |
| r_eff(J_i) | exp(−Σ λ̂ log λ̂), λ̂ из σ²(J_i) | JR |
| r_stable(J_i) | ‖J_i‖²_F / σ²_max | JR |
| ‖h_i‖₂ | норма hidden state | H |
| g_uni,i | ‖QJ_i‖²_F / E_i, Q=uu^T | P128 |
| H_i | −Σ p̂ log p̂ на U128 | P128 |
| σ_z,i | std(z_i[U128]) | нужно пересчитать из logits |

σ_z,i требует загрузки логитов — их в кэше нет, только P128 (post-softmax). Если логитов нет, σ_z,i помечается `not_cached`.

### 11.3. Deliverable

```json
"T8_per_sample": {
  "6": {
    "wiki_0": {"E": 0.0, "logE": 0.0, "r_eff": 0.0, "r_stable": 0.0, "norm_h": 0.0, "g_uni": 0.0, "H": 0.0},
    ...
  }
}
```

### 11.4. Применение

Из этого строится всё, что Exp 27.5 §8.2 называет «attribution»:

- корреляция cluster-label с H_i
- корреляция cluster-label с r_eff(J_i)
- сравнение распределений признаков между найденными кластерами

Без T8 attribution кластеров в Exp 27.5 §14.3 невозможна.

---

## 12. Выходной формат

Единый JSON:

```text
results/exp27_6_isolation.json
```

Структура:

```json
{
  "params": {"S": 64, "K": 128, "pos": 0, "seed": 42, "B_perm": 1000, "B_boot": 1000},
  "T1_isolation": {...},
  "T2_bock": {...},
  "T3_bootstrap": {...},
  "T4_ari_null": {...},
  "T5_dip": {...},
  "T6_eigengap": {...},
  "T7_depth_spearman": {...},
  "T8_per_sample": {...}
}
```

Один PNG:

```text
results/exp27_6_bock_4x4.png
```

4 панели (по одному на слой 6/12/18/22), на каждой — heatmap 4×4 блоков K^G. Плюс диагональ K^F для сравнения.

---

## 13. Критерий готовности

Exp 27.6 считается завершённым, если:

1. Все 8 задач T1–T8 имеют выход в JSON.
2. Для каждой из задач явно записан **вердикт** в текстовом поле: `verdict: "pass" | "fail" | "inconclusive"`.
3. Обновлены вердикты Exp 27.5:
    - если T1 (no-math) ломает доменный вердикт — отметить в §14.3 Exp 27.5.
    - если T4 (ARI null) подтверждает «домены не работают» — обновить §5 таблицу Exp 27.5.
    - если T6 (eigengap) `has_structure = false` — удалить из Exp 27.5 §9 весь блок silhouette/gap как неприменимый, заменить на «спектр плоский, структуры нет».

---

## 14. Порядок исполнения

```
1. T1 → T2 (domain isolation и блоки)      — 5 мин
2. T6 (eigengap)                            — 2 мин
3. Если T6 has_structure: T3, T4, T5        — 13 мин
   Иначе: пропустить T3/T4/T5, идти к T7
4. T7 (depth-C_a)                           — 2 мин
5. T8 (per-sample)                          — 5 мин
```

T2 — единственная задача, которая работает с K^G и требует pair_matrix; всё остальное — на K^F и per_sample массивах.

---

## 15. Минимальный псевдокод

```python
jcache = load(results/exp27_jcache.pt)
pair_matrix = load(results/exp27_pair_matrix_S64K64p0.json)
labels = {"wiki_0": "wiki", ...}  # 32 метки

# T1, T2 — из pair_matrix (уже содержат C_b и per_domain)
T1 = isolation(pair_matrix, domains_exclude=["math", "chat", "code"])
T2 = bock_4x4(pair_matrix)
save_png_T2(T2)

# T6 — из pair_matrix
eigengap = spektrum(pair_matrix)
has_structure = any(g > 1.5 for g in eigengap.ratios)
if has_structure:
    T3 = bootstrap_ci(pair_matrix, labels, n=1000)
    T4 = permutation_ari(pair_matrix, labels, n=1000)
    T5 = dip_null(pair_matrix, n=500)

# T7 — из halfsplit_baseline
T7 = spearman_depth_ca(jcache, S=64, K=64, pos=0, n_boot=1000)

# T8 — из jcache, для каждого сэмпла × слой
T8 = per_sample_features(jcache, S=64, K=128, pos=0)

save(results/exp27_6_isolation.json)
```

---

## 16. Ожидаемые результаты (для проверки, что эксперимент осмысленный)

| Задача | Ожидание | Проверка |
|---|---|---|
| T1 | `T_domain^(no-math) / T_domain^(all)` ∈ [0.4, 1.0] | если <0.2 — артефакт |
| T2 | `K^G_mm` ниже `K^G_dd` на 3× | если нет — math не выброс |
| T3 | CI на T_domain шире 0.05 | если уже — неожиданно точно |
| T4 | p_value ∈ [0.3, 0.7] (chance) | если <0.05 — домены работают |
| T5 | dip > 0.05 на поздних слоях | если <0.05 — унимодально |
| T6 | eigengap заметен на L18–L23 | если нет нигде — плоский |
| T7 | ρ(layer, C_a^pospk) > 0.8 | если <0.5 — тренд слабый |
| T8 | r_eff(J_i) варьируется в 5× между сэмплами | если нет — per-sample не нужно |

Если после T1–T8 вердикты Exp 27.5 совпадут с ожиданиями — работа Exp 27 закрыта. Если разойдутся — есть конкретная точка, где план требует пересборки.

---

## 17. Итог

Exp 27.6 — компактное приложение к Exp 27.5, закрывающее ровно шесть P0-дыр:

- **изоляция math** (T1) — иначе «домен не работает» не защищено;
- **блоки 4×4** (T2) — иначе math vs остальные не разделить;
- **bootstrap CI** (T3) — иначе p_perm ≠ точность;
- **permutation-null ARI** (T4) — иначе ARI ≈ 0 без интерпретации;
- **dip-нуль** (T5) — иначе bimodality не интерпретируется;
- **eigengap** (T6) — иначе silhouette может мерить шум.

Плюс два дешёвых дополнения:

- **ρ(layer, C_a) с CI** (T7) — формализует вывод о глубине;
- **per-sample features** (T8) — открывает attribution для Exp 27.5 §14.

Все восемь задач — ≤ 30 минут CPU на кэше, без новых forward-проходов.