# ТЗ на новые модули `prism/`, часть 3: `noise.py` + `permutation.py` + `topology.py`

**Область:** три независимых модуля. `noise.py` — generic калибровка шумового пола и валидация сигнала; `permutation.py` — permutation max-statistic (FWER) и cluster-based alternative; `topology.py` — zigzag persistence и generic проекции в подпространства.

**Ключевые решения (зафиксированы):**

1. **`noise.py` — без «GDN» в именах.** `calibrate_noise_floor(metric_fn, n_repeats)` и `validate_signal(signal, floor, k=5)` — generic. GDN-политика (кросс-версия, детерминированный режим, full-attn control) — в `experiments/exp_T0.py`. Модуль **не знает** о модели, GDN, `Qwen3.5` — только callable и числа.
2. **`noise.py` — не устанавливает seeds по умолчанию.** Цель — измерить **естественную** вариантность `metric_fn`. Если `metric_fn` детерминирован — шумовой пол `≈ 0`, и `validate_signal` вырождается в простой порог. Это **ожидаемое** поведение, задокументировано.
3. **`permutation.py` — только permutation-статистика и null-генератор; никаких forward-проходов.** Null генерируется paired-перестановкой меток внутри документа (стандартный paired permutation test). Метрика — **уже посчитанные скаляры** (из `MetricCache`).
4. **`topology.py` — optional зависимости.** `giotto-tda` и/или `dionysus` — не обязательны для `import prism`; модуль импортируется, но функции, использующие PH, требуют установки бэкенда. Явная ошибка `ImportError` с инструкцией `pip install`, а не молчаливый `ImportError` при `import prism.topology`.
5. **`topology.py` — только zigzag PH и generic проекции.** `compare_tail_vs_pca` и специфичные диаграммы — в `experiments/exp_T1.py`.

---

# Модуль 5: `prism/noise.py`

## 5.1. Назначение

Generic калибровка шумового пола вычисления и валидация сигнала против этого пола. **Никакой зависимости от модели, GDN или конкретных экспериментов.** Только `Callable[[], float]` и числа.

## 5.2. Публичный API

```python
def calibrate_noise_floor(
    metric_fn: Callable[[], float],
    n_repeats: int = 10,
    reset_between: bool = True,
    verbose: bool = False,
) -> dict: ...

def validate_signal(
    signal: float,
    floor: dict,
    k: float = 5.0,
) -> dict: ...
```

## 5.3. Контракты по функциям

### `calibrate_noise_floor(metric_fn, n_repeats=10, reset_between=True, verbose=False)`

| Аспект | Значение |
|---|---|
| Цель | Измерить **естественную** вариантность `metric_fn()` при повторных вызовах без изменения весов/данных |
| `metric_fn` | `Callable[[], float]` — pure-функция без побочных эффектов; **не должна** устанавливать seeds (иначе вариантность будет 0; это допустимо, но не даёт шумового пола) |
| `n_repeats` | `≥ 2` (иначе std не определён); рекомендуется 10 |
| `reset_between` | Если `True` (по умолчанию): между вызовами `torch._dynamo.reset()` (если доступно), `torch.cuda.empty_cache()` (если доступно), `gc.collect()`. Это сбрасывает кэши компиляции/памяти, увеличивая естественную вариантность (что нам и нужно). Если `False` — вызовы подряд (может дать std ≈ 0 из-за warm cache) |
| Seeds | **Не** устанавливаются. `calibrate_noise_floor` не вызывает `torch.manual_seed`/`np.random.seed` — иначе шумовой пол был бы 0 |
| Возврат | `dict` с ключами: `mean` (float), `std` (float, `ddof=0`), `values` (list[float], `len=n_repeats`), `is_deterministic` (bool, `std < 1e-10`), `n_repeats`, `reset_between` |
| `is_deterministic` | `True` — сигнал: `metric_fn` детерминирован в данной конфигурации. `validate_signal` всё равно работает, но `k·std ≈ 0`, и любое ненулевое `signal` даёт «significant» — это **корректно**, задокументировать |
| NaN/Inf в values | `ValueError` с указанием, на каком повторе. Это ловит «шумовой пол» от деления на ноль внутри `metric_fn` |
| Ошибки | `n_repeats < 2` → `ValueError`; `metric_fn()` возвращает не скаляр (Tensor/list) → `TypeError` (требуется `.item()`) |
| Побочные эффекты | `metric_fn` может мутировать состояние модели (например, in-place `copy_`); **responsibility caller'а** — сделать `metric_fn` pure. `calibrate_noise_floor` не защищает от этого |
| Verbose | Печать `[i/N] value` для диагностики |

**Почему `reset_between=True` по умолчанию:** без сброса кэшей `torch._dynamo` и CUDA memory allocator, повторные вызовы часто дают идентичные результаты (warm cache), и мы измерим 0, даже если реальный источник вариантности (например, GDN ordering) существует. Сброс даёт **реалистичный** шумовой пол.

**Что именно измеряется:** `calibrate_noise_floor` — это не «сколько шума в модели», а «сколько вариантности в **повторном вычислении** `metric_fn`». Если `metric_fn` включает forward + Jacobian, шумовой пол отражает недетерминизм forward/backward (GDN ordering, CUDA atomic adds, thread scheduling). Если `metric_fn` читает только из кэша — шумовой пол ≈ 0, и это правильно: кэш детерминирован.

### `validate_signal(signal, floor, k=5.0)`

| Аспект | Значение |
|---|---|
| Цель | Решить, значим ли `signal` относительно шумового пола `floor` |
| `signal` | `float` — наблюдаемая величина (например, `κ_U` или `ε_gap`) |
| `floor` | `dict` из `calibrate_noise_floor` (ключи `mean`, `std`, `is_deterministic`) |
| `k` | `> 0` — число «сигм» для значимости; по умолчанию `5.0` (5-σ правило) |
| Возврат | `dict` с ключами: `verdict` (`"significant"` / `"noise"` / `"no_floor"`), `ratio` (float, `|signal − mean| / std`), `deviation` (float, `signal − mean`), `threshold` (float, `k · std`), `is_deterministic` (bool, из floor) |
| Вердикт `"significant"` | `|signal − mean| > k · std` (при `is_deterministic=True` — любое `|signal − mean| > 1e-10`) |
| Вердикт `"noise"` | `|signal − mean| ≤ k · std` (и `is_deterministic=False`) |
| Вердикт `"no_floor"` | `is_deterministic=True` и `|signal − mean| ≤ 1e-10` (нечего калибровать; это отдельный случай: `metric_fn` детерминирован, `signal` тоже ~0) |
| `std == 0` и `is_deterministic=False` | `RuntimeError` — невозможно (по определению `is_deterministic=True` при `std < 1e-10`) |
| `floor` без ключа | `KeyError` с явным сообщением |
| `k ≤ 0` | `ValueError` |

**Интерпретация:**
- `"significant"` — `signal` выходит за 5σ естественной вариантности; можно утверждать реальный эффект.
- `"noise"` — `signal` внутри шумового пола; **нельзя** утверждать эффект.
- `"no_floor"` — `metric_fn` детерминирован и `signal` тоже; ничего не происходит; либо `signal` действительно 0, либо `signal` мал в детерминированной системе (не баг, но и не «значимый эффект»).

## 5.4. Внутренние детали

- **`_reset_caches()`** — приватная: `try: torch._dynamo.reset() except: pass`; `try: torch.cuda.empty_cache() except: pass`; `gc.collect()`. Не бросает, если что-то недоступно.
- **`_validate_floor(floor)`** — приватная: проверяет наличие `mean`, `std`, `is_deterministic`.
- **Импорты** — `import torch`, `import gc`, `from collections.abc import Callable`. Никаких `datasets`, `transformers`.
- **Совместимость с `deterministic_mode`** из `cache.py` — `calibrate_noise_floor` **не** должен вызываться внутри `deterministic_mode(strict=True)`, иначе `is_deterministic=True` и шумовой пол 0. Задокументировать: `strict` режим — для других целей.

## 5.5. Взаимодействие с существующим

- **`cache.deterministic_mode`** — **ортогональна**: `calibrate_noise_floor` измеряет вариантность; `deterministic_mode` её устраняет. Не вызывать `calibrate_noise_floor` внутри `deterministic_mode(strict=True)` — иначе шумовой пол 0 (это корректно, но не даёт «естественного» пола).
- **`spectral.py`, `holonomy.py`** — `metric_fn` обычно `lambda: eigengap(spectrum_from_gramian(cross_positional_gramian(J_t, J_tp)))` или `lambda: holonomy_accumulate(...)[0]`. `noise.py` их не импортирует; работает с **любым** `Callable[[], float]`.
- **`paper.md` §8.8** — прямая ссылка в docstrings.
- **GDN-политика (в `experiments/exp_T0.py`)** — использует `validate_signal` + решает, что делать с `"noise"`/`"significant"`. `noise.py` не знает, что такое GDN.

## 5.6. Selfcheck (`tests/test_noise.py`)

1. **Deterministic `metric_fn`:** `metric_fn = lambda: 42.0`; `floor = calibrate_noise_floor(metric_fn, n_repeats=5, reset_between=False)`; `assert floor["std"] < 1e-10`; `assert floor["is_deterministic"]`.
2. **Noisy `metric_fn`:** `rng = np.random.default_rng(0)`; `metric_fn = lambda: rng.normal(0, 1)`; `floor = calibrate_noise_floor(metric_fn, n_repeats=100)`; `assert 0.5 < floor["std"] < 1.5` (порядок 1).
3. **`is_deterministic=False` for noisy:** `assert not floor["is_deterministic"]` для (2).
4. **`validate_signal` significant:** `floor = {"mean": 0.0, "std": 1.0, "is_deterministic": False}`; `v = validate_signal(10.0, floor, k=5.0)`; `assert v["verdict"] == "significant"`; `v["ratio"] ≈ 10.0`.
5. **`validate_signal` noise:** `v = validate_signal(2.0, floor, k=5.0)`; `assert v["verdict"] == "noise"`.
6. **`validate_signal` deterministic:** `floor = {"mean": 42.0, "std": 0.0, "is_deterministic": True}`; `v = validate_signal(42.0 + 1e-9, floor)`; `assert v["verdict"] == "no_floor"`; `v = validate_signal(50.0, floor)`; `assert v["verdict"] == "significant"`.
7. **`n_repeats < 2` → ValueError.**
8. **NaN в values → ValueError:** `metric_fn = lambda: float('nan')`; `calibrate_noise_floor(...)` → `ValueError` с сообщением, содержащим «nan».
9. **Non-scalar return → TypeError:** `metric_fn = lambda: [1.0, 2.0]` → `TypeError`.
10. **`reset_between=True` vs `False`:** на `torch.randn` (без seed) `reset_between=False` даёт std > 0, но **меньше**, чем `reset_between=True` (если есть кэш) — или равно (если `torch.randn` всегда использует глобальный RNG). Проверить на синтетике с явной зависимостью от кэша (например, `metric_fn` читает из `functools.lru_cache`) — `reset=False` даёт 0, `reset=True` не даёт (если lru_cache не сбрасывается; на практике `metric_fn` без lru_cache — оба дают std > 0). Тест: `assert calibrate_noise_floor(..., reset_between=True)["std"] >= 0`.
11. **`k ≤ 0` → ValueError.**

## 5.7. Ограничения и риски

- **`calibrate_noise_floor` не защищает от мутаций в `metric_fn`.** Если `metric_fn` меняет веса модели (`copy_`), повторные вызовы дадут разное поведение, и шумовой пол будет мешать все мутации вместе. **Ответственность caller'а** — сделать `metric_fn` pure (сохранять/восстанавливать состояние внутри).
- **`reset_between=True` увеличивает стоимость** (перекомпиляция `torch._dynamo`). Для `n_repeats=10` на T4 — минуты; приемлемо.
- **`is_deterministic=True` ≠ «сигнал отсутствует».** Это значит «вычисление детерминировано, шумовой пол 0, любой сигнал — реальный эффект». Не путать.
- **Не измеряет GDN-специфичный шум отдельно.** Если нужно — caller добавляет в `metric_fn` `enable_grad` + GDN-обёртку; `noise.py` остаётся generic.
- **Не сравнивает два шумовых пола** (например, base vs edit). Для этого — `validate_signal` применяется дважды; `noise.py` не делает «paired» сравнение двух floor'ов.

## 5.8. Что НЕ входит в `noise.py`

- GDN-политика — `experiments/exp_T0.py`.
- `deterministic_mode` — `cache.py`.
- Cross-version проверки — `experiments/exp_T0.py` (вызывают `calibrate_noise_floor` на разных `transformers`).
- Сравнение двух floor'ов — caller делает вручную.
- Кэширование floor'ов — caller сохраняет в `MetricCache` (не внутри `noise.py`).

---

# Модуль 6: `prism/permutation.py`

## 6.1. Назначение

Permutation-статистика для FWER-контроля над послойными профилями. **Никаких forward-проходов** — работает с уже посчитанными скалярами (из `MetricCache`) и `metric_fn` над переставленными метками. Null-генератор — paired permutation (внутри документа). Cluster-based permutation — альтернатива с повышенной мощностью.

## 6.2. Публичный API

```python
def permutation_max_statistic(
    observed: np.ndarray,                # [n_layers]
    null_samples: np.ndarray,            # [B, n_layers]
    alpha: float = 0.05,
) -> dict: ...

def generate_null(
    base_values: np.ndarray,             # [n_docs] or [n_docs, n_layers]
    edit_values: np.ndarray,             # [n_docs] or [n_docs, n_layers], same shape
    metric_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],  # ([n_docs], [n_docs]) -> [n_layers]
    n_permutations: int = 1000,
    seed: int = 0,
) -> np.ndarray: ...                     # [n_permutations, n_layers]

def cluster_based_permutation(
    observed: np.ndarray,                # [n_layers]
    null_samples: np.ndarray,            # [B, n_layers]
    threshold: float = 2.0,
    alpha: float = 0.05,
) -> dict: ...
```

## 6.3. Контракты по функциям

### `permutation_max_statistic(observed, null_samples, alpha=0.05)`

| Аспект | Значение |
|---|---|
| Цель | FWER-контроль (Westfall–Young max-statistic) над профилем из `n_layers` |
| `observed` | `[n_layers]`, наблюдаемая статистика (например, `ε_gap(edit) − ε_gap(base)` по слоям) |
| `null_samples` | `[B, n_layers]`, нуль из `generate_null`; `B ≥ 100` |
| `alpha` | `∈ (0, 1)`; по умолчанию 0.05 |
| Возврат | `dict` с ключами: `p_value` (float), `max_observed` (float), `null_max_distribution` (`[B]`), `significant` (bool, `p_value < alpha`), `alpha`, `n_layers`, `B` |
| Алгоритм | `max_obs = max_l |observed[l]|`; `max_null[b] = max_l |null_samples[b, l]|`; `p = (1 + Σ_b [max_null[b] ≥ max_obs]) / (B + 1)` |
| `p` — точное | Формула `(1 + k) / (B + 1)` — стандартная для permutation (Wright et al., 2015); не использует асимптотику, точное для дискретного нуля |
| Форма `observed` | `[n_layers]`; если `[n_layers, ...]` — `ValueError` (требуется плоский профиль) |
| Форма `null_samples` | `[B, n_layers]`; `null_samples.shape[1] != len(observed)` → `ValueError` |
| NaN/Inf | `ValueError` (тихо игнорировать — скрывает баги в `generate_null`) |
| `B < 100` | `ValueError` (слишком мало для FWER) |
| `alpha` вне `(0, 1)` | `ValueError` |
| `max_obs == 0` | `p = 1.0` (нет эффекта; корректно, если `max_null ≥ 0` всегда) |

**Почему max-statistic, а не BH-FDR:** BH-FDR предполагает (или требует) независимость/PRDS; слои L18–L23 сильно коррелированы. Max-statistic **не требует независимости** и контролирует FWER при произвольной корреляции. Это выбор, а не компромисс — задокументировать в docstring.

### `generate_null(base_values, edit_values, metric_fn, n_permutations=1000, seed=0)`

| Аспект | Значение |
|---|---|
| Цель | Сгенерировать null-распределение `[B, n_layers]` для paired permutation test |
| `base_values` | `[n_docs]` или `[n_docs, n_layers]` — по одному значению на документ (или вектор на слой); **base** без правки |
| `edit_values` | Той же формы — **edited** |
| `metric_fn` | `Callable[[np.ndarray, np.ndarray], np.ndarray]` — `(base_perm, edit_perm) -> [n_layers]`; должна быть pure |
| Paired permutation | Для каждой перестановки `b`: для каждого документа `i` независимо бросаем «монетку» (`rng.random() < 0.5`); если орёл — `(base_perm[i], edit_perm[i]) = (base[i], edit[i])`; если решка — `(edit[i], base[i])`. Метки `base`/`edit` переставляются внутри пары |
| Семантика | Под H0 (base и edit exchangeable) paired permutation сохраняет структуру пар; это стандартный paired test |
| `metric_fn` вызывается | `B` раз с `(base_perm, edit_perm)`; результат `[n_layers]`; стек → `[B, n_layers]` |
| Seed | `np.random.default_rng(seed)` — детерминирован; `seed` записывается в возврат? Нет — caller сохраняет в `meta.json` |
| Форма `base_values` | `[n_docs]` → передаётся в `metric_fn` как `[n_docs]`; `metric_fn` сама решает, как агрегировать в `[n_layers]` (например, `median` по докам на каждый слой, если на входе `[n_docs, n_layers]` — не нужно; на входе `[n_docs]` — `metric_fn` возвращает `[n_layers]` через `[base_perm, edit_perm] → [n_layers]`, если метрика per-layer) |
| Форма `edit_values` | Должна совпадать с `base_values` |
| `len(base) != len(edit)` | `ValueError` |
| `n_docs < 2` | `ValueError` (paired test требует ≥2 пар) |
| `n_permutations < 100` | `ValueError` |
| NaN/Inf | `ValueError` |

**Важно про `metric_fn`:** caller отвечает за то, чтобы `metric_fn` возвращала **ту же** статистику, что `observed` в `permutation_max_statistic`. Если `observed = median(edit − base) по докам`, то `metric_fn(perm_base, perm_edit) = median(perm_edit − perm_base) по докам`. Если `metric_fn` возвращает per-layer, а `observed` — скаляр, тест сломается семантически; `permutation.py` проверяет только shapes, не семантику.

**Пример `metric_fn` для ε:**
```python
def metric_fn(base_vals, edit_vals):  # [n_docs, n_layers] each
    # median across docs, per layer
    return np.median(edit_vals - base_vals, axis=0)  # [n_layers]
```
`observed` тогда — `np.median(edit_values − base_values, axis=0)` — тот же расчёт без перестановки.

### `cluster_based_permutation(observed, null_samples, threshold=2.0, alpha=0.05)`

| Аспект | Значение |
|---|---|
| Цель | FWER-контроль с использованием корреляции соседних слоёв (Maris & Oostenveld, 2007) |
| `threshold` | `> 0` — порог для формирования кластеров; слои с `|observed[l]| > threshold` объединяются в кластеры по соседству (`|l − l'| = 1`) |
| `null_samples` | Как в `permutation_max_statistic`; предполагается, что null центрирован (`mean ≈ 0`); функция **не** центрирует |
| Алгоритм | 1. Threshold `observed`, найти кластеры (contiguous runs). 2. Для каждого кластера: statistic = `Σ |observed[l]|` по слоям. 3. Для каждого `b`: threshold `null_samples[b]`, найти кластеры, statistic каждого; взять `max` по кластерам → `max_cluster_null[b]`. 4. Для каждого observed-кластера: `p = (1 + Σ_b [max_cluster_null[b] ≥ observed_stat]) / (B + 1)`. |
| Возврат | `dict` с ключами: `clusters` (list of dicts: `{"layers": [...], "statistic": float, "p_value": float, "significant": bool}`), `null_max_cluster_distribution` (`[B]`), `alpha`, `n_layers`, `B` |
| Нет кластеров в `observed` | Возврат `{"clusters": [], ...}`; не ошибка (нет сигнала) |
| Нет кластеров в `null_samples[b]` | `max_cluster_null[b] = 0.0` (корректно: null не даёт кластеров, observed-кластер значим) |
| `threshold` ≤ 0 | `ValueError` |
| `alpha` вне `(0, 1)` | `ValueError` |
| NaN/Inf | `ValueError` |
| Форма | Как в `permutation_max_statistic` |
| `B < 100` | `ValueError` |

**Когда использовать:** если `permutation_max_statistic` даёт `p` чуть выше `alpha` (например, 0.06), но эффект визуально есть на 2–3 соседних слоях — cluster-based может дать `p < 0.05` за счёт группировки. **Не** использовать как primary — primary `permutation_max_statistic` (консервативнее, чище); cluster-based — дополнительная диагностика.

## 6.4. Внутренние детали

- **`_find_clusters(values, threshold)`** — приватная: `mask = |values| > threshold`; ищет contiguous runs; возвращает `list[list[int]]` (индексы слоёв). Используется в `cluster_based_permutation`.
- **`_paired_permutation_indices(n_docs, rng)`** — приватная: `signs = rng.random(n_docs) < 0.5` (bool); возвращает `signs`; `base_perm[i] = base[i] if signs[i] else edit[i]`; `edit_perm[i] = edit[i] if signs[i] else base[i]`. Это **paired**: каждый документ переставляется независимо.
- **Импорты** — `import numpy as np`, `from collections.abc import Callable`. Никаких `torch`, `scipy` (кроме `scipy.stats` если понадобится для `t`-значений в cluster-based — см. §6.3; в базовом варианте не нужно, т.к. используется `Σ |observed|`).
- **Про `cluster_based_permutation` и `t`:** стандартная формулировка Maris & Oostenveld использует `t`-статистику, а не `observed`-значения. Здесь мы используем `|observed[l]|` как statistic (проще, устойчивее; caller должен передать уже нормированные значения, например, `t` из парного теста). Задокументировать: «caller должен передать нормированную статистику (например, Cohen's d или t); функция использует `|values|` как расстояние».
- **Детерминизм:** `rng = np.random.default_rng(seed)`; никаких глобальных сидов.

## 6.5. Взаимодействие с существующим

- **`cache.MetricCache`** — источник `base_values`, `edit_values` (через `bulk_get(metric, "base", layer)` и `bulk_get(metric, "readout_pull", layer)`). `permutation.py` **не** импортирует `cache.py` — caller читает и передаёт `np.ndarray`.
- **`cache.bootstrap_ci`** — комплементарен: bootstrap даёт CI, permutation даёт p-value. Оба — в `experiments/exp_eps.py` и `exp_T2.py`.
- **`paper.md` §8.9, §13** — прямая ссылка в docstrings.
- **`experiments/exp_T0.py`** — вызывает `permutation_max_statistic` для каждого `metric × edit`; результат в JSON.

## 6.6. Selfcheck (`tests/test_permutation.py`)

1. **`permutation_max_statistic` trivial (observed = 0):** `observed = np.zeros(6)`; `null_samples = rng.normal(0, 1, (100, 6))`; `p ≈ 1.0`; `assert not significant`.
2. **`permutation_max_statistic` extreme:** `observed = np.zeros(6); observed[2] = 100.0`; `null_samples = rng.normal(0, 1, (100, 6))`; `p ≈ 0.0` (в пределах `1/(B+1)`); `assert significant`.
3. **`permutation_max_statistic` p formula:** `observed = np.array([1.0, 0, 0])`; `null = np.zeros((100, 3))` (все null = 0); `p = (1 + 100) / 101 ≈ 1.0` (все `max_null ≥ max_obs` ложно, т.к. `0 < 1`; на самом деле `k = 100`, p = 1.0). Проверить формулу.
4. **`permutation_max_statistic` correlation:** `observed = np.array([1.0] * 6)`; `null_samples = rng.normal(0, 1, (100, 6))`; `max_obs = 1.0`; `null_max = max(|null[b]|)`; `p = mean(null_max ≥ 1.0)`; проверить, что `p` существенно **больше**, чем если бы слои были независимы (`p` для одного слоя = 0.32; для max по 6 коррелированным — ближе к 0.32, чем 0.32⁶, если корреляция положительная — это свойство max-statistic).
5. **`generate_null` determinism:** `base = rng.normal(0, 1, 10)`; `edit = base + 0.5`; `metric_fn = lambda b, e: np.array([np.median(e − b)])`; два вызова с `seed=0` → идентичные `[B, 1]`.
6. **`generate_null` paired structure:** `base = np.array([1.0, 1.0, 1.0])`; `edit = np.array([2.0, 2.0, 2.0])`; `metric_fn = lambda b, e: np.array([np.median(e − b)])`; для каждой перестановки `b`, `median(e_perm − b_perm)` равно `median(sign · 1)` где `sign ∈ {+1, −1}` по каждому документу. При `n_docs=3`, возможные значения `median(sign · 1)`: `+1`, `−1`, `0` (если 1 vs 2); проверить распределение.
7. **`generate_null` shape mismatch:** `base.shape != edit.shape` → `ValueError`.
8. **`generate_null` n_docs < 2** → `ValueError`.
9. **`generate_null` non-1D metric_fn output:** `metric_fn` возвращает `[]` (пустой) → `ValueError`.
10. **`cluster_based_permutation` single cluster:** `observed = np.array([0, 3, 4, 0, 0, 0])`; `null_samples = rng.normal(0, 1, (100, 6))`; `threshold = 2.0`; `assert len(clusters) == 1`; `clusters[0]["layers"] == [1, 2]`; `clusters[0]["statistic"] == 7.0`.
11. **`cluster_based_permutation` two clusters:** `observed = np.array([3, 4, 0, 0, 5, 6])`; `assert len(clusters) == 2`.
12. **`cluster_based_permutation` no clusters:** `observed = np.zeros(6)`; `assert clusters == []`.
13. **`cluster_based_permutation` p for strong observed:** `observed = np.array([0, 10, 10, 0, 0, 0])`; `null = rng.normal(0, 1, (100, 6))`; `p ≈ 0`; `significant`.
14. **`threshold <= 0` → ValueError.**
15. **`B < 100` → ValueError** для обеих функций.

## 6.7. Ограничения и риски

- **`generate_null` предполагает, что `metric_fn` — paired-статистика.** Если caller передаёт независимую статистику (не парную), paired permutation даст неверный null. Задокументировать: `metric_fn` должна быть функцией **парных** значений (`base[i]`, `edit[i]` для одного и того же документа).
- **`metric_fn` может быть дорогой** — `B=1000` × стоимость `metric_fn`. Для метрик на кэшах — секунды. Для forward-проходов — часы. Caller должен обеспечить, что `metric_fn` работает на кэшах, не на модели.
- **`cluster_based_permutation` использует `Σ |values|`** — не стандартный `t`-test; caller должен нормировать значения (например, `Cohen's d`). Задокументировать.
- **`p` из `permutation_max_statistic` — для всего профиля**, не per-layer. Для per-layer — `bootstrap_ci` на каждом слое или `cluster_based_permutation`.
- **Correlation между слоями не моделируется явно** — max-statistic поглощает её консервативно. Cluster-based использует её для повышения мощности, но требует нормированной статистики.

## 6.8. Что НЕ входит в `permutation.py`

- Чтение из `MetricCache` — caller делает.
- Computation `metric_fn` — caller передаёт.
- BH-FDR — **сознательно** не реализуется (max-statistic + cluster-based покрывают FWER; BH-FDR не подходит для коррелированных слоёв).
- Multiple-testing коррекция на уровне экспериментов (5 edits × 6 слоёв × 4 метрики) — за пределами модуля (caller решает, применять ли дополнительную коррекцию к `p` из `permutation_max_statistic`).
- Генерация null для других нулевых гипотез (не paired) — только paired.

---

# Модуль 7: `prism/topology.py`

## 7.1. Назначение

Zigzag persistent homology по последовательности точечных облаков (слоёв) + generic проекции в подпространства. **Optional зависимости:** `giotto-tda` и/или `dionysus`. Модуль импортируется без них; функции, требующие PH, бросают `ImportError` с инструкцией.

## 7.2. Публичный API

```python
def zigzag_persistence(
    point_clouds: list[np.ndarray],     # [X_6, X_12, ..., X_23], each [n_points, d]
    maxdim: int = 1,
    metric: str = "euclidean",
    backend: str = "giotto",            # "giotto" | "dionysus"
) -> dict: ...

def project_to_subspace(
    h: np.ndarray,                       # [N, D] or [D]
    basis: np.ndarray,                   # [r, D] or [D, r] — rows or cols, disambiguated
) -> np.ndarray: ...                     # [N, r] or [r]

def project_to_pca_subspace(
    h_batch: np.ndarray,                 # [N, D]
    n_components: int = 15,
) -> tuple[np.ndarray, np.ndarray]: ...  # ([N, n_components], [n_components, D] — components)
```

## 7.3. Контракты по функциям

### `zigzag_persistence(point_clouds, maxdim=1, metric="euclidean", backend="giotto")`

| Аспект | Значение |
|---|---|
| Цель | Zigzag PH по последовательности точечных облаков (слоёв), с прямыми и обратными отображениями между соседними облаками |
| `point_clouds` | `list[np.ndarray]`, `len ≥ 2`; каждый `[n_l, d_l]`; `d_l` **может различаться** (разные слои могут иметь разные подпространства — см. ниже), но `n_l` (число точек) — **одинаковый** на всех облаках (это одни и те же позиции/документы) |
| `maxdim` | `0` или `1`; `≥ 2` → `ValueError` (H2 нестабилен при `N ~ 128`) |
| `metric` | `"euclidean"` / `"cosine"` / `"precomputed"` (последний — если `point_clouds` уже distance matrices, но это редко; default `"euclidean"`) |
| `backend` | `"giotto"` (default) или `"dionysus"`; если бэкенд не установлен → `ImportError` с `pip install` |
| Отображения между облаками | **Zigzag требует явных maps.** В базовом варианте: если `d_l == d_{l+1}` (одно и то же подпространство), maps — identity. Если разные — нужна проекция; **caller отвечает** за приведение к общему пространству (например, PCA-сжатие до `d=15` на всех слоях) **до** вызова `zigzag_persistence`. Функция не делает проекцию неявно |
| Возврат | `dict` с ключами: `dgms` (list of diagrams, `dgms[k]` — `[n_features, 2]` birth/death for dimension `k`), `n_layers`, `maxdim`, `backend`, `metric`, `n_points` |
| Формат `dgms[k]` | `np.ndarray` `[n_features, 2]`; `∞` в `death` — numpy `inf` |
| `n_points < 5` | `ValueError` (PH на < 5 точках тривиальна) |
| NaN/Inf в облаках | `ValueError` |
| Детерминизм | `giotto-tda` и `dionysus` детерминированы по значениям (порядок features может зависеть от реализации — не полагаться) |

**Про zigzag vs обычную PH:** обычная PH требует **вложенной** фильтрации (`K_0 ⊂ K_1 ⊂ ...`). Zigzag допускает **обратные** шаги (`K_i → K_{i+1} → K_i`), что позволяет отслеживать рождение/смерть признаков при переходе между слоями (облака не вложены). Это ключевое отличие, зафиксировать в docstring.

**Почему `d_l` может различаться:** если caller сжимает каждое облако до `d=15` через PCA отдельно — `d_l = 15` для всех. Если caller использует хвостовые моды (E-30b, `r_tail ≈ 10–20`), `d_l = r_tail` одинаково для всех. Если caller использует разные подпространства по слоям — `d_l` различается, и **caller обязан** привести к общему `d` через PCA/проекцию **до** zigzag. `zigzag_persistence` не делает это неявно.

### `project_to_subspace(h, basis)`

| Аспект | Значение |
|---|---|
| Цель | Проекция `h` на подпространство, заданное `basis` |
| `h` | `[N, D]` или `[D]` |
| `basis` | `[r, D]` (строки — базисные векторы; **default**) **или** `[D, r]` (столбцы); дизамбигуация по соглашению: если `basis.shape[1] == h.shape[-1]` — rows; если `basis.shape[0] == h.shape[-1]` — cols; если оба — `ValueError` с указанием формы |
| Возврат | `[N, r]` или `[r]` — проекция `h @ basis.T` (для rows) или `h @ basis` (для cols) |
| Ортонормальность `basis` | **Не проверяется** (caller отвечает; для `top_r_svd` `V_r` — уже ортонормирован; для произвольного `basis` — caller обеспечивает) |
| dtype | `float32` (вход `float64` — конверсия) |
| Ошибки | Несовместимые формы → `ValueError`; `basis.dim() != 2` → `ValueError` |
| Зачем нужна | **Generic**: хвостовые моды (`basis = tail_basis` из E-30b), top-r SVD (`basis = V_r`), произвольный базис. Не привязана к конкретному источнику базиса |

### `project_to_pca_subspace(h_batch, n_components=15)`

| Аспект | Значение |
|---|---|
| Цель | PCA-сжатие `h_batch` до `n_components` |
| `h_batch` | `[N, D]` |
| Возврат | `(h_proj, components)` — `h_proj: [N, n_components]`, `components: [n_components, D]` (строки — главные компоненты, ортонормированы) |
| Алгоритм | `h_centered = h_batch − mean`; `U, S, Vt = svd(h_centered, full_matrices=False)`; `components = Vt[:n_components]`; `h_proj = h_centered @ components.T` (центрированные проекции) |
| Центрирование | **Да** — PCA без центрирования некорректен; возвращаемое `h_proj` — центрированное. Если caller хочет нецентрированное — добавить `mean` обратно вручную; задокументировать |
| `n_components > min(N, D)` | `ValueError` |
| `N < 2` | `ValueError` |
| dtype | `float32` |
| Детерминизм | `torch.linalg.svd`/`np.linalg.svd` детерминирован по значениям; знаки могут различаться между версиями — не полагаться |

## 7.4. Внутренние детали

- **`_import_giotto()`** — приватная: `try: from gtda.diagrams import ZigzagPersistence; except ImportError: raise ImportError("giotto-tda is required for backend='giotto'. Install: pip install giotto-tda")`. Аналогично `_import_dionysus()`.
- **`_validate_point_clouds(point_clouds)`** — приватная: `len ≥ 2`, каждая `np.ndarray` `2D`, `n_points` одинаковое, `d ≥ 1`, `n_points ≥ 5`, нет NaN/Inf.
- **`_to_common_dim(point_clouds, target_d)`** — **не реализуется** в модуле (caller приводит к общему `d` через `project_to_pca_subspace` перед вызовом `zigzag_persistence`). Задокументировать в docstring.
- **Импорты** — `import numpy as np`. `giotto`/`dionysus` — лениво внутри `_import_*`. Модуль импортируется без них.
- **Про `precomputed` metric:** `giotto-tda` ZigzagPersistence принимает `point_clouds` как list of arrays; для `metric="precomputed"` — list of distance matrices. Реализовать как передачу в бэкенд без конверсии; caller отвечает за формат.

## 7.5. Взаимодействие с существующим

- **`spectral.top_r_svd`** — `V_r` из `top_r_svd(J, r)` используется как `basis` в `project_to_subspace`. `topology.py` **не** импортирует `spectral`, но caller (в `experiments/exp_T1.py`) делает:
  ```python
  U, S, V = top_r_svd(J, r)
  h_proj = project_to_subspace(h, V)
  ```
- **`esg.compute_esg_per_sample`** — источник `tail_basis` (top-r eigenvectors `JJ^T`, `r ≥ 119` из E-30b). `topology.py` **не** импортирует `esg`; caller передаёт `tail_basis` явно.
- **`cache.MetricCache`** — не используется (PH даёт диаграммы, не скаляры; если нужно скалярное резюме — caller считает `total_persistence` вручную и кладёт в `MetricCache`).
- **`paper.md` §8.10** — прямая ссылка в docstrings.
- **`experiments/exp_T1.py`** — обёртка: подготовка `point_clouds` (проекции на tail / PCA), вызов `zigzag_persistence`, сравнение диаграмм (`compare_tail_vs_pca` — в `experiments/`, не в `prism/topology.py`).

## 7.6. Selfcheck (`tests/test_topology.py`)

Тесты, не требующие `giotto-tda` (будут `skip` если не установлен):

1. **`project_to_subspace` rows:** `h = np.random.randn(10, 8)`; `basis = np.eye(8)[:3]` (3 строки); `h_proj = project_to_subspace(h, basis)`; `assert h_proj.shape == (10, 3)`; `assert np.allclose(h_proj, h[:, :3])`.
2. **`project_to_subspace` cols:** `basis = np.eye(8)[:, :3]` (8 строк, 3 столбца); `h_proj = project_to_subspace(h, basis)`; `assert h_proj.shape == (10, 3)`; то же значение.
3. **`project_to_subspace` 1D h:** `h = np.random.randn(8)`; `h_proj = project_to_subspace(h, basis)`; `assert h_proj.shape == (3,)`.
4. **`project_to_subspace` ambiguous:** `basis = np.eye(8)` (8×8); `h = np.random.randn(8, 8)`; обе интерпретации совместимы → `ValueError` с указанием формы.
5. **`project_to_subspace` invalid basis:** `basis = np.random.randn(8)`; `project_to_subspace(h, basis)` → `ValueError`.
6. **`project_to_pca_subspace`:** `h = np.random.randn(100, 20)`; `h_proj, comps = project_to_pca_subspace(h, n_components=5)`; `assert h_proj.shape == (100, 5)`; `assert comps.shape == (5, 20)`; `assert np.allclose(comps @ comps.T, np.eye(5), atol=1e-5)` (ортонормированность).
7. **`project_to_pca_subspace` preserves variance:** `assert h_proj.var(axis=0).sum() / h.var(axis=0).sum() > 0.5` (для случайных данных 5 из 20 компонент дают заметную долю дисперсии).
8. **`project_to_pca_subspace` n_components > min(N, D) → ValueError.**
9. **`project_to_pca_subspace` N < 2 → ValueError.**

Тесты, требующие `giotto-tda` (`skipif` если не установлен):

10. **`zigzag_persistence` circle:** `theta = np.linspace(0, 2π, 30)`; `X = np.stack([cos θ, sin θ], axis=1)`; `point_clouds = [X, X, X]`; `result = zigzag_persistence(point_clouds, maxdim=1, backend="giotto")`; `assert len(result["dgms"]) == 2`; `assert result["dgms"][1].shape[0] ≥ 1` (H1 — окружность).
11. **`zigzag_persistence` trivial:** `X = np.random.randn(30, 2)`; `point_clouds = [X, X, X]`; `result = zigzag_persistence(point_clouds, maxdim=1)`; `result["dgms"][1].shape[0] == 0` (нет H1).
12. **`zigzag_persistence` with two distinct clouds:** `X1, X2 = random`; `point_clouds = [X1, X2]`; `result` не падает; диаграммы корректного формата.
13. **`zigzag_persistence` n_points < 5 → ValueError.**
14. **`zigzag_persistence` maxdim > 1 → ValueError.**
15. **`zigzag_persistence` no backend installed:** если `giotto` не установлен — `ImportError` с сообщением, содержащим `pip install giotto-tda`.
16. **`zigzag_persistence` backend="dionysus"** (если `dionysus` установлен) — симметрично (10).

## 7.7. Ограничения и риски

- **Zigzag требует общих maps между облаками.** Если `d_l` различается между слоями — caller обязан привести к общему `d` (через `project_to_pca_subspace` или `project_to_subspace`). Функция не делает это неявно.
- **`giotto-tda` и `dionysus` — тяжёлые зависимости.** Первый тянет `scikit-learn`, `scipy`, `plotly`; второй — C++ extension. `import prism.topology` **не** импортирует их (ленивая загрузка).
- **`ZigzagPersistence` в `giotto-tda` может быть медленным** для `n_points > 200` — при `N=128` приемлемо; для `N=512+` — минуты.
- **Знаки диаграмм и порядок features** могут различаться между версиями бэкендов; не полагаться.
- **`maxdim=1` — единственный разумный выбор** при `N ~ 128`. `H2` требует `N ~ 500+` для стабильности; для наших данных `H2` — шум. Задокументировать.
- **`∞` в диаграмме:** `np.inf` — стандартно для H0 (essential component). Caller должен обрабатывать `inf` (например, для `total_persistence` — игнорировать или использовать `max_finite`).
- **`project_to_pca_subspace` центрирует** — если caller ожидал нецентрированные проекции, будет ошибка масштаба. Задокументировать.

## 7.8. Что НЕ входит в `topology.py`

- `compare_tail_vs_pca` — `experiments/exp_T1.py`.
- `total_persistence` — caller считает вручную (или добавляет в `experiments/`).
- Приведение разных `d_l` к общему `d` — caller через `project_to_pca_subspace`.
- Загрузка `tail_basis` из `E-30b` кэша — caller.
- Визуализация диаграмм — caller.
- `giotto-tda`/`dionysus` как hard dependency — optional, ленивый импорт.

---

# Сводный чек-лист для этой части

| Пункт | Модуль | Проверка |
|---|---|---|
| `calibrate_noise_floor` (deterministic, noisy, NaN, non-scalar) | `noise.py` | selfcheck §5.6.1–5.6.9 |
| `calibrate_noise_floor` не устанавливает seeds (вариантность > 0 на случайном) | `noise.py` | selfcheck §5.6.2 |
| `validate_signal` (significant, noise, no_floor, deterministic) | `noise.py` | selfcheck §5.6.4–5.6.6 |
| `permutation_max_statistic` (trivial, extreme, correlation) | `permutation.py` | selfcheck §6.6.1–6.6.4 |
| `generate_null` (determinism, paired, mismatch, n_docs < 2) | `permutation.py` | selfcheck §6.6.5–6.6.9 |
| `cluster_based_permutation` (single cluster, two clusters, none, strong) | `permutation.py` | selfcheck §6.6.10–6.6.14 |
| `project_to_subspace` (rows, cols, 1D, ambiguous, invalid) | `topology.py` | selfcheck §7.6.1–7.6.5 |
| `project_to_pca_subspace` (shape, orthonormal, variance) | `topology.py` | selfcheck §7.6.6–7.6.9 |
| `zigzag_persistence` (circle, trivial, two clouds, errors) — под `giotto` | `topology.py` | selfcheck §7.6.10–7.6.16 (skip если не установлен) |
| `import prism.topology` работает без `giotto`/`dionysus` | `topology.py` | `python -c "import prism.topology"` без бэкендов |
| `noise.py` **не** упоминает GDN/Qwen3.5 | `noise.py` | `grep -i "gdn\|qwen" prism/noise.py` — пусто |
| `permutation.py` **не** делает forward-проходы | `permutation.py` | `grep "torch" prism/permutation.py` — пусто (только `numpy`) |
| `topology.py` **не** импортирует `giotto`/`dionysus` на верхнем уровне | `topology.py` | `grep "^from gtda\|^import gtda\|^import dionysus" prism/topology.py` — пусто |
| `permutation.py` **не** реализует BH-FDR | `permutation.py` | grep: `bh\|fdr\|benjamini` — пусто |
| `cache.MetricCache` **не** импортируется в `permutation.py` | `permutation.py` | grep: `from .cache import` — пусто |
| `spectral.top_r_svd` **не** импортируется в `topology.py` | `topology.py` | grep: `from .spectral import` — пусто |
| `experiments/exp_T1.py` использует `project_to_subspace` + `zigzag_persistence` | `experiments/` | ручная проверка |
| `experiments/exp_eps.py` использует `permutation_max_statistic` + `bootstrap_ci` | `experiments/` | ручная проверка |
| `experiments/exp_T0.py` использует `calibrate_noise_floor` + `validate_signal` + GDN-политику | `experiments/` | ручная проверка |

**Критерий готовности:**
- Все selfcheck из §5.6, §6.6, §7.6 проходят (с `skip` для `giotto`-тестов, если бэкенд не установлен).
- `git diff` по `bridge.py`, `esg.py`, `jacobian.py`, `calibration.py`, `diagnostics.py`, `layer_selection.py`, `cache_io.py`, `spectral.py`, `cache.py`, `holonomy.py`, `interventions.py` — пустой (кроме уже согласованных правок).
- `import prism.noise`, `import prism.permutation`, `import prism.topology` работают без `datasets`, `giotto-tda`, `dionysus`.
- Существующие тесты (`tests/test_*.py`) — зелёные.

---

*Следующая часть ТЗ: `pipeline.py` (`ExperimentSpec` + `run_spec`) — финальный инфраструктурный модуль. Затем — `experiments/exp_T0.py` … `exp_delta.py` (тонкие обёртки, каждая ~50–100 строк, используют только `prism/*`).*