# ТЗ на новые модули `prism/`, часть 1: `spectral.py` + `cache.py`

**Область:** два фундаментальных модуля, от которых зависят все остальные новые модули (`holonomy`, `interventions`, `noise`, `permutation`, `topology`). Порядок — сначала математические примитивы (`spectral`), затем инфраструктура (`cache`). Оба — только примитивы с ≥2 потребителями; конкретные композиции — в `experiments/`.

**Принцип:** не дублировать существующее. `esg.py` уже даёт `_spectral_decomposition` (eigh + sort) — `spectral.py` **переиспользует** его, не переписывает. `cache_io.py` уже даёт `find_targets_dir` — `cache.py` **использует** его для поиска, не дублирует.

---

# Модуль 1: `prism/spectral.py`

## 1.1. Назначение

Консолидация спектральных метрик, которые сейчас разбросаны по `experiments/` и частично дублируются в `esg.py`. Все функции — чистые (без состояния), принимают спектр или матрицу, возвращают скаляр/тензор. **Никаких forward-проходов, никакой зависимости от модели.** Это математика поверх уже посчитанных `J` / `G`.

## 1.2. Публичный API

```python
# SVD-примитивы
def top_r_svd(J: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def spectrum_from_gramian(G: torch.Tensor, ascending: bool = False) -> torch.Tensor: ...

# Спектральные метрики (принимают spectrum — [K] или [n], НЕ матрицу)
def eigengap(spectrum: torch.Tensor) -> float: ...
def entropy_effective_rank(spectrum: torch.Tensor) -> float: ...
def stable_rank(spectrum: torch.Tensor) -> float: ...
def participation_ratio(spectrum: torch.Tensor) -> float: ...
def snr_top1_vs_rest(spectrum: torch.Tensor) -> float: ...   # см. §1.5 — определение уточнить
def gauge_share(spectrum: torch.Tensor, eigenvectors: torch.Tensor, uniform: torch.Tensor | None = None) -> float: ...
def gauge_share_top1(spectrum: torch.Tensor, eigenvectors: torch.Tensor) -> float: ...
```

**Никаких `from_model`, `compute_*_for_layer` — только чистые функции над тензорами.** Это критично для переиспользования: та же `eigengap` работает и для `Δ_U(t, t')`, и для ESG, и для произвольной PSD-матрицы.

## 1.3. Контракты по функциям

### `top_r_svd(J, r)`

| Аспект | Значение |
|---|---|
| Вход | `J: [K, D]`, `r: int`, `1 ≤ r ≤ min(K, D)` |
| Выход | `(U_r, S_r, V_r)` — `U_r: [K, r]`, `S_r: [r]` (убывание), `V_r: [r, D]` |
| Семантика | `J ≈ U_r @ diag(S_r) @ V_r` (truncated SVD, `torch.linalg.svd(full_matrices=False)`) |
| dtype | Вход `float32`/`float64` → выход того же dtype; вход `fp16/bfloat16` → приведение к `float32` (для численной стабильности SVD) |
| device | Тот же, что у `J` |
| Ошибки | `r < 1` или `r > min(K, D)` → `ValueError`; `J.dim() != 2` → `ValueError` |
| Детерминизм | `torch.linalg.svd` детерминирован по значениям; **знаки** `U`, `V` могут зависеть от версии LAPACK — потребители не должны опираться на знаки (использовать `|cos|`, `|align|` и т.п.) |

**Зачем нужна:** holonomy (`U_r`, `V_r`), ε (топ-r спектр без полного `eigh`), T1' (top-r проекция для PCA-сжатия).

**Не заменяет** `esg._spectral_decomposition` (тот возвращает **все** eigenvalues + векторы, отсортированные). Если нужен полный спектр — `spectrum_from_gramian`.

### `spectrum_from_gramian(G, ascending=False)`

| Аспект | Значение |
|---|---|
| Вход | `G: [K, K]` симметричная (PSD не обязательно — `Δ_U` симметрична, но не всегда PSD при `t ≠ t'` из-за знакопеременных скалярных произведений) |
| Выход | `[K]` — eigenvalues, sorted |
| Семантика | `torch.linalg.eigvalsh` (только собственные значения — быстрее, чем `eigh` на ~2×) |
| Сортировка | `ascending=False`: убывание (λ₁ ≥ λ₂ ≥ …) — соответствует конвенции paper §6. `ascending=True`: возрастание — для некоторых численных проверок |
| dtype | Всегда `float64` при вычислении (для стабильности `eigvalsh` на плохо обусловленных `Δ_U`), результат приводится к dtype входа |
| Ошибки | `G.dim() != 2` или `G.shape[0] != G.shape[1]` → `ValueError` |
| Симметрия | Проверка `torch.allclose(G, G.T)` с `atol=1e-6`; если нет — `ValueError` (не молчаливая симметризация — это скрывает баги в `Δ_U`) |

**Зачем нужна:** все метрики ниже принимают `spectrum` — этот метод стандартно получает его из `Δ_U` или `G`.

**Не путать** с `esg._spectral_decomposition(G, n_components)` — тот возвращает **top-n + векторы** (нужно для `gauge_share`), а этот — **все** eigenvalues без векторов (нужно для `entropy_effective_rank`).

### `eigengap(spectrum)`

| Аспект | Значение |
|---|---|
| Вход | `spectrum: [K]`, `K ≥ 2`, в **любом** порядке (функция сама сортирует) |
| Выход | `λ₁ − λ₂` (скаляр `float`) |
| Семантика | Разность двух наибольших собственных значений; мера «спектральной резкости» |
| Ошибки | `K < 2` → `ValueError` |
| Отрицательные λ | Не отбрасываются (важно: `Δ_U(t,t')` при `t ≠ t'` может иметь отрицательные λ; для «правильного» `eigengap` Δ_U должно быть PSD, но функция не навязывает) |

**Зачем нужна:** ε — падение `eigengap(Δ_U)` — сигнал спектрального Goodhart.

### `entropy_effective_rank(spectrum)`

| Аспект | Значение |
|---|---|
| Вход | `spectrum: [K]` |
| Выход | `exp(−Σ p_k log p_k)`, `p_k = λ_k / Σλ` — эффективный ранг `r_eff` (paper §6) |
| Обработка отрицательных λ | **Отбрасываются** (`λ_k ≤ 0 → p_k = 0`); если все λ ≤ 0 → возврат `0.0` с `RuntimeWarning` |
| Обработка `λ_k = 0` | `p_k log p_k → 0` (предел); реализовать через `torch.where(p > 0, p * log(p), 0)` |
| Нормировка | `Σ p_k = 1` после отбрасывания отрицательных |
| Ошибки | Пустой spectrum → `ValueError` |

**Соответствие paper:** `r_eff` из §6, §15; используется в E-14, E-27.6.

### `stable_rank(spectrum)`

| Аспект | Значение |
|---|---|
| Вход | `spectrum: [K]` |
| Выход | `Σλ / λ₁` (убывание, `ascending=False` внутри) |
| Семантика | Отношение полной энергии к максимуму — «насколько спектр не сосредоточен в одной моде» (paper §6) |
| Обработка `λ₁ = 0` | Возврат `0.0` (все λ нулевые); `RuntimeWarning` |
| Отрицательные λ | Не отбрасываются (для `Δ_U` может быть смешанный спектр; stable_rank определён как отношение) |
| Отличие от `PR` | **Не синоним**: для `λ = (4,1,1,1,1)`: `r_stable = 8/4 = 2`, `PR = 64/18 ≈ 3.6` (paper §6) |

### `participation_ratio(spectrum)`

| Аспект | Значение |
|---|---|
| Вход | `spectrum: [K]` |
| Выход | `(Σλ)² / Σλ²` |
| Обработка `Σλ² = 0` | Возврат `0.0` |
| Отрицательные λ | Не отбрасываются (PR определён на всех λ) |

**Соответствие paper:** PR в §6, §17 (E-14).

### `snr_top1_vs_rest(spectrum)` — **требует уточнения**

**Проблема:** в paper §6 сказано «SNR — leave-one-out отношение "сигнал" к "шуму" в спектре». **Точная формула в paper не специфицирована.** В E-14 числа: 0.14 → 3.65 (L0 → L21), 199.3 (L23). Нужно:

1. **Найти в `experiments/exp14_snr_pr.py`** (или эквиваленте) точную формулу, которую использовал оригинальный E-14.
2. Если код недоступен — реализовать **наиболее вероятную** формулу `λ₁ / mean(λ₂..λ_K)` и назвать её `snr_top1_vs_rest`, явно задокументировав: «definition under review; differs from E-14 if original was different».
3. Если E-14 использовал `λ₁ / (Σλ_{-1})` или `λ₁ / std(λ_{-1})` — назвать соответственно.

**Контракт (при реализации `snr_top1_vs_rest`):**

| Аспект | Значение |
|---|---|
| Вход | `spectrum: [K]`, `K ≥ 2` |
| Выход | `λ₁ / mean(λ₂..λ_K)`, где λ₁ — **наибольшее положительное** |
| Обработка `mean(λ_{-1}) = 0` | Возврат `inf` (или `float('inf')`) |
| Обработка `K = 1` | `ValueError` (нечего делить) |
| Отрицательные λ | Не отбрасываются (для ESG спектр PSD, но для `Δ_U` — нет) |

**Требование к имплементатору:** если удастся найти оригинальную формулу E-14 — реализовать её и назвать `snr_e14`. Обе функции (`snr_top1_vs_rest` и `snr_e14`) могут сосуществовать; в `experiments/` использовать ту, что воспроизводит числа E-14, и явно указать в `meta.json`, какая используется.

**Не блокирует:** модуль можно реализовать без `snr`, если явно зафиксировать в `__init__.py` отсутствие. Но `E-T0`/`E-ε` будут ссылаться на `SNR` — если функции нет, использовать `eigengap` как прокси. Решение — за пользователем.

### `gauge_share(spectrum, eigenvectors, uniform=None)`

| Аспект | Значение |
|---|---|
| Вход | `spectrum: [K]`, `eigenvectors: [K, K]` (столбцы — `u_k`, соответствуют `spectrum[k]`), `uniform: [K] | None` (по умолчанию `1/√K · ones(K)`) |
| Выход | `Σ_k (u_k^T uniform)² λ_k / Σ_k λ_k` (точное определение paper §6) |
| Семантика | Доля энергии `JJ^T`, лежащая на uniform-направлении в token-space (paper §6, «gauge_share (ESG-версия)») |
| Нормировка `uniform` | Если задан — приводится к `L2-норме 1` (не обязательно, но рекомендуется); если не задан — `1/√K` |
| Сортировка | `spectrum` и `eigenvectors` должны быть согласованы (i-й eigenvalue ↔ i-й столбец); проверка `len(spectrum) == eigenvectors.shape[0]` |
| Отрицательные λ | Не отбрасываются (для ESG λ ≥ 0, но контракт не требует) |
| Ошибки | Несовпадение размеров → `ValueError`; `Σ λ = 0` → `0.0` с предупреждением |
| Отличие от `g_uni` (§8) | `g_uni = ‖QJ‖_F² / ‖J‖_F²` — доля энергии **сырого** J; `gauge_share` — доля энергии **`JJ^T`** на uniform. Разные объекты (paper §6, §8); не путать |

**Соответствие paper:** точное определение `gauge_share` (§6). Top-1 proxy — отдельная функция.

### `gauge_share_top1(spectrum, eigenvectors)`

| Аспект | Значение |
|---|---|
| Вход | Те же, что `gauge_share` |
| Выход | `λ₁ · (u₁^T uniform)² / Σ_k λ_k` — top-1 приближение (paper §6) |
| Семантика | То, что отчитывается в E-10a как колонка `gauge_share_0` |
| Отличие от `gauge_share` | Учитывает только `u₁`; `gauge_share` — все моды |

**Соответствие paper:** E-10a `gauge_share_0` — это `gauge_share_top1`. Задокументировать явно.

## 1.4. Внутренние детали реализации

- **Все функции — pure, без состояния.** Никаких кэшей внутри `spectral.py` (кэш — в `cache.py`).
- **`torch.linalg.svd` / `eigvalsh`** — единственные вызовы; никаких `numpy` (для консистентности device/dtype).
- **Проверки на `NaN/Inf`**: в `eigengap`, `gauge_share` — если вход содержит `NaN` → `ValueError` с указанием индекса. Это ловит битые `J` из кэша до того, как они уйдут в permutation-тесты.
- **`float64` для внутренних вычислений** в `spectrum_from_gramian`, `gauge_share` (плохо обусловленные `Δ_U` могут давать λ с ошибкой 1e-3 в float32); вход/выход — в dtype входа.

## 1.5. Взаимодействие с существующим

- **`esg.py`**: `esg._spectral_decomposition(G, n_components)` возвращает `(ev, evec)` — **не** дублируется. `spectral.spectrum_from_gramian` — дополнение для случаев, когда нужен полный спектр. Если `esg._spectral_decomposition` называется с `n_components=K` — это эквивалент `spectrum_from_gramian` + векторы. **Никаких изменений в `esg.py`** — `spectral.py` просто не дублирует.
- **`experiments/exp14_snr_pr.py`** (существующий): если в нём есть локальные `_entropy_effective_rank`, `_participation_ratio`, `_snr` — **перенести** их в `spectral.py` и в `exp14` заменить на импорт. Не копировать. Если код экспериментов недоступен — переписать по формулам paper §6 + комментарий «originally in exp14, moved to prism/spectral.py».
- **`paper.md`** (ТЗ на правку §8.5–§8.10): ссылки на `spectral.py` — не нужны, paper описывает математику, `spectral.py` — реализация. Обратная ссылка: `spectral.py` docstrings → paper §6, §8.5, §8.9 (соответствующие определения).

## 1.6. Selfcheck (`tests/test_spectral.py`)

1. **`top_r_svd`:** `J = torch.randn(8, 16, dtype=torch.float64)`; `U, S, V = top_r_svd(J, r=4)`; `assert torch.allclose(U @ torch.diag(S) @ V, J, atol=1e-4)` — нет, это **не** точное равенство для truncated SVD; правильная проверка: `‖J − U S V‖_F` мала относительно `‖J‖_F`, и `U S V` — ранг r. Сравнить с `torch.linalg.svd(J, full_matrices=False)` — top-r компоненты совпадают до знака.
2. **`spectrum_from_gramian`:** `G = J @ J.T`; `spectrum_from_gramian(G)` = `(torch.linalg.svd(J).S**2)`, отсортированные по убыванию — сравнить.
3. **`eigengap`:** `spectrum = torch.tensor([4.0, 2.0, 1.0])`; `assert eigengap(spectrum) == 2.0`. С неотсортированным — то же значение.
4. **`entropy_effective_rank`:** `spectrum = torch.tensor([0.25, 0.25, 0.25, 0.25])` (равномерный rank-4); `assert abs(r_eff - 4.0) < 1e-6`. С rank-1 `[1, 0, 0, 0]`: `assert abs(r_eff - 1.0) < 1e-6`.
5. **`stable_rank` vs `participation_ratio`:** `spectrum = torch.tensor([4.0, 1.0, 1.0, 1.0, 1.0])`; `assert stable_rank ≈ 8/4 = 2.0`; `assert participation_ratio ≈ 64/18 ≈ 3.555`.
6. **`snr_top1_vs_rest`:** `spectrum = torch.tensor([10.0, 1.0, 1.0])`; `assert snr_top1_vs_rest ≈ 10.0` (10 / mean(1,1)).
7. **`gauge_share`:** `U = torch.eye(K)` (eigenvectors = стандартный базис); `spectrum = torch.tensor([1.0] * K)`; uniform = `1/√K`; `assert gauge_share(spectrum, U, uniform) ≈ 1.0` (все λ на uniform). С `U` случайной ортогональной — сравнить с `Σ_k (u_k^T uniform)² λ_k / Σ λ_k` (прямой расчёт).
8. **`gauge_share_top1`:** `gauge_share_top1(spectrum, U)` = `λ₁ · (u₁^T uniform)² / Σλ`. При `U = eye(K)`, `spectrum = [1,0,...,0]`: `gauge_share_top1 = 1/K` (uniform = `1/√K`, `(u₁^T uniform)² = 1/K`).
9. **Ошибки:** `eigengap(torch.tensor([1.0]))` → `ValueError`; `top_r_svd(J, r=K+1)` → `ValueError`; `spectrum_from_gramian(несимметричная G)` → `ValueError`.
10. **NaN:** `spectrum = torch.tensor([1.0, float('nan'), 2.0])` → `eigengap` → `ValueError`.

## 1.7. Ограничения и риски

- **Знаки сингулярных векторов** не стабильны между версиями LAPACK — все потребители (`holonomy`, `ε`) должны использовать `|cos|` или Procrustes-остатки (инвариантные к знаку).
- **`snr_top1_vs_rest`** — определение под вопросом; если E-14 использует другую формулу, числа в `E-ε` не совпадут с E-14. Задокументировать явно.
- **`gauge_share` с `eigenvectors` из `top_r_svd`** — если `r < K`, `Σλ` в знаменателе ≠ trace полного `JJ^T`; функция ожидает **полный** спектр. Для top-r — использовать `gauge_share_top1` (или передавать `Σλ` явно как `total_trace`). **Контракт:** `spectrum` и `eigenvectors` должны покрывать **все** моды, не top-r.
- **`entropy_effective_rank` с отрицательными λ** (для `Δ_U(t,t')` при `t ≠ t'`) отбрасывает их → `r_eff` может быть меньше, чем «ожидалось». Это семантически корректно (отрицательная «энергия» не имеет смысла), но задокументировать.

## 1.8. Что НЕ входит в `spectral.py`

- Никаких `compute_*_for_layer`, `for_model` — это `experiments/`.
- Никаких кэшей — `cache.py`.
- Никаких `gauge_share` на `JJ^T` от сырого `J` (это `g_uni` в §8 — отдельно, если понадобится в будущем).
- Никакого `SNR` сверх `snr_top1_vs_rest` (если E-14 использует другую формулу — отдельная функция, не в этом PR).

---

# Модуль 2: `prism/cache.py`

## 2.1. Назначение

Инфраструктура кэширования для новой серии: SVD по `(edit, layer, doc, pos)`, скалярные метрики по `(metric, edit, layer, doc)`, bootstrap CI, детерминированный режим, итератор по позиционным парам. **Не заменяет `cache_io.py`** — `cache_io.find_targets_dir` используется для поиска существующих кэшей (Kaggle FUSE), `cache.py` — для **записи** и логики кэша.

**Ключевое отличие от `cache_io`:** `cache_io` — read-only, поиск путей. `cache.py` — read+write, ключи, атомарная запись, версионирование.

## 2.2. Публичный API

```python
# Кэши
class SVDCache:
    def __init__(self, root: str, r: int, dtype: torch.dtype = torch.float32, version: str = "v1"): ...
    def get(self, edit: str, layer: int, doc: int, pos: int) -> tuple[Tensor, Tensor, Tensor] | None: ...
    def put(self, edit: str, layer: int, doc: int, pos: int, U: Tensor, S: Tensor, V: Tensor) -> None: ...
    def has(self, edit: str, layer: int, doc: int, pos: int) -> bool: ...
    def key(self, edit: str, layer: int, doc: int, pos: int) -> str: ...

class MetricCache:
    def __init__(self, root: str, version: str = "v1"): ...
    def get(self, metric: str, edit: str, layer: int, doc: int | None = None) -> float | None: ...
    def put(self, metric: str, edit: str, layer: int, doc: int | None, value: float) -> None: ...
    def has(self, metric: str, edit: str, layer: int, doc: int | None = None) -> bool: ...
    def bulk_get(self, metric: str, edit: str, layer: int) -> dict[int, float]: ...
    def bulk_put(self, metric: str, edit: str, layer: int, values: dict[int, float]) -> None: ...

# Статистика
def bootstrap_ci(values: Sequence[float], B: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]: ...
def bootstrap_ci_paired(values_a: Sequence[float], values_b: Sequence[float], B: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]: ...

# Детерминизм
@contextmanager
def deterministic_mode(seed: int = 0, strict: bool = False): ...

# Итераторы
def iter_position_pairs(S: int, mode: str = "adjacent", stride: int = 1) -> Iterable[tuple[int, int]]: ...
```

## 2.3. Контракты по компонентам

### `SVDCache`

| Аспект | Значение |
|---|---|
| Ключ | `(edit, layer, doc, pos, r, dtype, version)` — SHA-256 от строки `f"{edit}|L{layer}|d{doc}|p{pos}|r{r}|{dtype}|v{version}"` |
| Путь | `{root}/{edit}/L{layer:02d}/d{doc:04d}_p{pos:04d}_r{r}_{version}.pt` |
| Формат файла | `torch.save({"U": U, "S": S, "V": V, "meta": {"edit": ..., "layer": ..., "doc": ..., "pos": ..., "r": ..., "dtype": str, "version": ..., "torch": torch.__version__}})`. `weights_only=True` при загрузке (как в `bridge._inverse_gramian`). |
| Атомарная запись | Пишем в `tmp`-файл, затем `os.replace(tmp, final)` — чтобы прерванная запись не оставила полу-файл |
| Валидация при `put` | `U.shape == (K, r)`, `S.shape == (r,)`, `V.shape == (r, D)`; `S` убывает (`S[:-1] >= S[1:]`); `dtype(U) == dtype(S) == dtype(V) == self.dtype`; `U`/`V` ортонормированы (`‖U^T U − I‖_F < 1e-5`); `r <= min(K, D)` |
| Валидация при `get` | Если meta содержит `torch != torch.__version__` — **предупреждение**, не ошибка (SVD совместим по формату). Если `r`, `dtype`, `version` не совпадают — возврат `None` (ключ другой) |
| Ошибки | `put` при несовпадении shapes/dtype → `ValueError`; `get` при битом файле → `RuntimeError` с указанием пути (не молчаливый `None` — это скрывает коррупцию) |
| In-memory кэш | LRU на `cache_size=256` записей (по умолчанию); параметр `cache_size` в `__init__` |
| Device | Дисковый кэш — CPU; при `get` результат переносится на device вызывающего явно (не хранится device в файле — кэш device-независим) |

**Почему SHA-256 от строки, а не от содержимого:** `(edit, layer, doc, pos)` — это **идентификатор запроса**, а не данные. Одинаковый запрос → одинаковый ключ. Версия `torch` и `r` включены, чтобы апгрейд не давал молчаливое «попадание в старый кэш».

**Что НЕ в `SVDCache`:** forward-проходы, применение edit — это `interventions.py` + `experiments/`. `SVDCache` — только read/write.

### `MetricCache`

| Аспект | Значение |
|---|---|
| Ключ | `(metric, edit, layer, doc, version)` — JSON-файл `{root}/{metric}/{edit}/L{layer:02d}.json`, содержащий `{"doc_0": value, "doc_1": value, ..., "meta": {...}}` |
| Формат значения | `float` (включая `inf`, `nan` — сохраняются как `"inf"`, `"nan"`; при загрузке — обратное) |
| Атомарная запись | `tmp` + `os.replace` (как в `SVDCache`) |
| `doc=None` | Для скаляров на слой (медиана по документам); ключ `"__layer_aggregate__"` в том же JSON |
| `bulk_get` | Возврат `{doc: value}` для всех доступных документов; отсутствующие документы не включены (не `None`) |
| Валидация | `put` при `value` не `float` (например, `Tensor`) → `ValueError` (требуется явное `.item()` — не молчаливая конверсия) |
| Версионирование | `version` в ключе; смена `version` → новый файл |
| Ошибки | `get` при битом JSON → `RuntimeError` |

**Почему JSON, а не `.pt`:** скаляры — маленькие, JSON human-readable, легко мержить в git и инспектировать; `torch.load` для скаляров — оверинжиниринг. Для сравнения: `SVDCache` — `.pt` (большие тензоры).

### `bootstrap_ci(values, B=1000, alpha=0.05, seed=0)`

| Аспект | Значение |
|---|---|
| Вход | `values: Sequence[float]`, `len ≥ 2`; `B ≥ 100`; `alpha ∈ (0, 1)` |
| Выход | `(lo, hi)` — percentile CI (`alpha/2` и `1−alpha/2` перцентили распределения бутстреп-статистик) |
| Статистика | **Median** (по умолчанию) — устойчива к выбросам в спектральных метриках. Параметр `statistic: Callable = np.median` — возможность заменить на `mean` |
| Ресэмплинг | С **возвращением** (стандартный bootstrap), `len(values)` элементов на итерацию |
| Seed | `np.random.default_rng(seed)` — детерминирован |
| `len < 2` | `ValueError` (нечего ресэмплировать) |
| `NaN` в values | `ValueError` (тихо игнорировать NaN — скрывает баги) |
| Возврат | `tuple[float, float]`; если все значения одинаковые — `(v, v)` |

**Соответствие paper:** paper §13 «Bootstrap CI: B = 1000 ресэмплингов (по сэмплам, не по парам), percentile-CI». **Замечание:** paper говорит «по сэмплам», эта функция — generic. Для paired (doc-level) — `bootstrap_ci_paired`.

### `bootstrap_ci_paired(values_a, values_b, B=1000, alpha=0.05, seed=0)`

| Аспект | Значение |
|---|---|
| Вход | `values_a`, `values_b` — **одинаковой длины**, парные (например, metric до/после edit на одном документе) |
| Выход | CI для **разности** `median(a) − median(b)` (или `median(a − b)` в paired-режиме) |
| Ресэмплинг | Индексы пар ресэмплируются **вместе** (`idx = rng.integers(0, N, N)`; `a[idx]`, `b[idx]`) |
| `len(a) != len(b)` | `ValueError` |
| Пустые | `ValueError` |

**Зачем:** ε и T2' сравнивают base vs edit **на одних и тех же документах** — paired CI даёт более узкие интервалы (правильно), чем независимые bootstrap-ы.

### `deterministic_mode(seed=0, strict=False)`

| Аспект | Значение |
|---|---|
| Контекстный менеджер | `@contextmanager` |
| Действия при входе | `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)` |
| `strict=True` | Дополнительно: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True` |
| Действия при выходе | **Восстановить** исходные состояния (`torch.random.get_rng_state()`, `torch.backends.cudnn.deterministic` и т.д.) — **не оставить** глобальное состояние изменённым |
| Вложенность | Поддерживается; восстановление — к состоянию до входа в **этот** контекст |
| Предупреждение | Если `strict=True` и `torch.are_deterministic_algorithms_enabled()` уже True — предупреждение «redundant» |

**Зачем:** σ_GDN, cross-version проверки, permutation-тесты. **Не заменяет** `_clone_gdn_states_on_entry` из `jacobian.py` — это разные уровни (seed vs in-place клонирование).

### `iter_position_pairs(S, mode="adjacent", stride=1)`

| Аспект | Значение |
|---|---|
| Вход | `S ≥ 2`, `mode ∈ {"adjacent", "strided", "all"}` |
| Выход | `Iterable[tuple[int, int]]` — пары `(t, t')` с `0 ≤ t < t' < S` |
| `mode="adjacent"` | `[(t, t+stride) for t in range(0, S−stride, stride)]` — пары соседних (с шагом stride) |
| `mode="strided"` | `[(t, t+stride) for t in range(0, S−stride)]` — все пары с фиксированным шагом |
| `mode="all"` | `[(t, t') for t in range(S) for t' in range(t+1, S)]` — все пары (для S=128 — 8128 пар; дорого) |
| `stride < 1` | `ValueError` |
| `S < 2` | `ValueError` |
| Детерминизм | Порядок фиксирован (t возрастает, для `all` — t' возрастает) |

**Зачем:** ε использует `adjacent`, T2' использует `adjacent` для `Δ_U` baseline, T1' использует `strided` для топологической фильтрации. `all` — для полноты, но не рекомендуется.

## 2.4. Внутренние детали реализации

- **`cache_io.find_targets_dir` не дублируется:** `SVDCache.__init__` вызывает `find_targets_dir(root)` для проверки, что `root` существует; если нет — создаёт (в `write`-режиме). Если `root` — read-only (Kaggle FUSE), `put` → `RuntimeError` с понятным сообщением. Это критично, потому что на Kaggle input-директории FUSE read-only.
- **Атомарная запись:** `tmp = f"{final}.tmp.{os.getpid()}.{uuid4().hex[:8]}"`; `os.replace(tmp, final)`. На Windows `os.replace` атомарен для файлов в одной ФС — этого достаточно.
- **LRU для `SVDCache`:** `OrderedDict` с `move_to_end`; вытеснение при `len > cache_size`.
- **`MetricCache` не имеет in-memory кэша** — файлы маленькие, чтение с диска дешёвое; кэширование усложнило бы consistency при параллельной записи.
- **`deterministic_mode(strict=True)`** — при `torch.use_deterministic_algorithms(True)` некоторые операции GDN могут бросить `RuntimeError` (нет детерминированной реализации); это **ожидаемое поведение**, задокументировать: «strict режим может ломать GDN; использовать только для non-GDN сценариев».
- **`bootstrap_ci` seed:** по умолчанию 0; для воспроизводимости в `meta.json` фиксировать `seed`, `B`, `alpha`.

## 2.5. Взаимодействие с существующим

- **`cache_io.find_targets_dir`** — используется в `SVDCache.__init__` и `MetricCache.__init__` для проверки/создания `root`. **Не дублировать.**
- **`bridge._gramian_cache`** — существующий in-memory кэш Грамиана; **не трогать** (он для `(E_S^T E_S)^{-1}`, не для SVD/metrics). `SVDCache` — отдельный.
- **`esg._spectral_decomposition`** — `SVDCache` **не** использует его (там eigh на `K×K`; здесь SVD на `K×D`). Это **разные** операции: SVD нужен для `U_r`, `V_r` (holonomy), eigh — для спектра `JJ^T` (метрики). Не путать.
- **`paper.md` §13** — `bootstrap_ci` реализует «B=1000, percentile». Параметр `statistic` не из paper, но добавлен для гибкости; default `median` соответствует «bootstrap по сэмплам» в контексте paper (медиана по документам).
- **`experiments/exp27_5_postprocess.py`** (существующий) — если там есть локальный `_bootstrap`, заменить на `prism.cache.bootstrap_ci`. Не дублировать.

## 2.6. Selfcheck (`tests/test_cache.py`)

1. **`SVDCache` round-trip:** `put` → `get` → `assert torch.allclose(U, U_loaded)` и т.д. для S, V.
2. **`SVDCache` key collision:** `put(edit="base", layer=6, doc=0, pos=0)` и `put(edit="base", layer=6, doc=0, pos=1)` — разные файлы, оба `has == True`.
3. **`SVDCache` version mismatch:** `SVDCache(root, r=4, version="v1")` → `put`; `SVDCache(root, r=4, version="v2")` → `has == False`.
4. **`SVDCache` r mismatch:** `SVDCache(root, r=4)` и `SVDCache(root, r=8)` — разные ключи, не пересекаются.
5. **`SVDCache` validation:** `put` с неотсортированным `S` → `ValueError`; с `U` не ортонормированным → `ValueError`.
6. **`SVDCache` атомарность:** симулировать падение посреди записи (monkey-patch `torch.save` на raise) → файл не создан (только `.tmp`); следующий `put` работает.
7. **`MetricCache` round-trip:** `put("eigengap", "base", 6, 0, 2.5)` → `get` → `2.5`. `bulk_get` возвращает `{0: 2.5}`.
8. **`MetricCache` `nan`/`inf`:** `put(..., float('nan'))` → `get` → `math.isnan`. `put(..., float('inf'))` → `get` → `math.isinf`.
9. **`MetricCache` doc=None:** `put("eigengap", "base", 6, None, 2.5)` → `get("eigengap", "base", 6, None)` → `2.5`; `bulk_get("eigengap", "base", 6)` **не включает** `None`-агрегат (отдельный ключ `__layer_aggregate__`).
10. **`bootstrap_ci` determinism:** `bootstrap_ci([1, 2, 3, 4, 5], B=100, seed=0)` дважды → одинаковый выход.
11. **`bootstrap_ci` constant:** `bootstrap_ci([5.0, 5.0, 5.0], B=100)` → `(5.0, 5.0)`.
12. **`bootstrap_ci_paired` vs independent:** на парных данных `bootstrap_ci_paired` даёт более узкий CI, чем разность независимых bootstrap-ов (проверить на синтетике).
13. **`deterministic_mode` restore:** `state_before = torch.random.get_rng_state()`; `with deterministic_mode(seed=42): torch.randn(1)`; `state_after = torch.random.get_rng_state()`; `assert torch.equal(state_before, state_after)`.
14. **`deterministic_mode` вложенность:** два вложенных контекста с разными seed — внутренний использует свой seed, внешний восстанавливается.
15. **`iter_position_pairs`:** `S=4, mode="adjacent", stride=1` → `[(0,1), (1,2), (2,3)]`. `mode="all"` → `[(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]`.
16. **`iter_position_pairs` errors:** `S=1` → `ValueError`; `stride=0` → `ValueError`.

## 2.7. Ограничения и риски

- **`SVDCache` требует знания `doc` и `pos`** — вызывающий должен передавать их явно; это правильно (не угадывать из `input_ids`).
- **`MetricCache` — глобальный по `root`**: несколько процессов, пишущих в один `root` с разными `metric` — не конфликтуют (разные файлы). Один процесс, пишущий в один `metric` с разными `doc` из разных потоков — **гонка** (атомарность `os.replace` спасает от полу-записи, но не от потери последнего значения). Задокументировать: **один процесс на `MetricCache` на `(metric, edit, layer)`**.
- **`bootstrap_ci` statistic=median** — для спектральных метрик с тяжёлыми хвостами (SNR может быть `inf`) median устойчив; для `mean` — предупредить в docstring.
- **`deterministic_mode(strict=True)`** — может ломать GDN (нет детерминированных ядер); использовать только для non-GDN сценариев или с `_clone_gdn_states_on_entry` в обёртке (клонирование — отдельно от seed).
- **`iter_position_pairs(mode="all")`** — `O(S²)`; для `S=128` — 8128 пар; предупредить в docstring и рекомендовать `adjacent`/`strided`.

## 2.8. Что НЕ входит в `cache.py`

- Применение edit к модели — `interventions.py`.
- Forward-проходы для заполнения кэша — `experiments/exp_T0.py`.
- Параллельная запись из нескольких процессов — не поддерживается (документировать).
- `cache_io.find_targets_dir` — не дублируется; используется.

---

# Сводный чек-лист для этой части

| Пункт | Модуль | Проверка |
|---|---|---|
| `top_r_svd`, `spectrum_from_gramian`, `eigengap`, `entropy_effective_rank`, `stable_rank`, `participation_ratio`, `gauge_share`, `gauge_share_top1` реализованы | `spectral.py` | selfcheck §1.6 (10 тестов) |
| `snr_top1_vs_rest` реализован с честной пометкой «definition under review» | `spectral.py` | selfcheck §1.6.6; либо явно исключён с записью в `__init__.py` |
| `SVDCache` (get/put/has/key, валидация, атомарность) | `cache.py` | selfcheck §2.6.1–2.6.6 |
| `MetricCache` (get/put/bulk, nan/inf, doc=None) | `cache.py` | selfcheck §2.6.7–2.6.9 |
| `bootstrap_ci`, `bootstrap_ci_paired` | `cache.py` | selfcheck §2.6.10–2.6.12 |
| `deterministic_mode` (restore, nesting) | `cache.py` | selfcheck §2.6.13–2.6.14 |
| `iter_position_pairs` (adjacent/strided/all, errors) | `cache.py` | selfcheck §2.6.15–2.6.16 |
| `cache_io.find_targets_dir` **не дублируется**, используется | `cache.py` | grep: `from .cache_io import find_targets_dir` |
| `esg._spectral_decomposition` **не дублируется**, не путается с `spectrum_from_gramian` | `spectral.py` | docstring: явное различие (eigh на K×K vs SVD на K×D) |
| `bridge._gramian_cache` **не трогается** | оба | git diff по `bridge.py` — пустой |
| `experiments/exp14_snr_pr.py` локальные метрики **перенесены** в `spectral.py`, не скопированы | `spectral.py` | если код exp14 доступен — diff пустой после замены на импорт |
| `paper.md §6, §8.5, §8.9, §13` — ссылки в docstrings | оба | ручная проверка |

**Критерий готовности:** все selfcheck проходят; `git diff` по `bridge.py`, `esg.py`, `jacobian.py`, `calibration.py`, `diagnostics.py`, `layer_selection.py`, `cache_io.py` — пустой; `import prism.spectral` и `import prism.cache` работают; существующие тесты (`tests/test_*.py`) — зелёные.

**Отклонение от исходного критерия «без `datasets`» (review `new_code_1_review.md` §3, P2.3):** `import prism.spectral`/`prism.cache` идёт через `prism/__init__.py`, который импортирует `calibration` (→ `datasets` на модульном уровне). Решение (директива от 2026-07-18, audit `docs/changed_review.md`): **без lazy-import** — `load_calibration_data` ядро публичного API, тишое отсутствие `datasets` дало бы NameError глубоко в эксперименте; вместо lazy — явный комментарий в `prism/__init__.py`. Следствие: окружение (CI, Kaggle) **обязано** иметь `datasets` (установлена в `requirements`). Если это неприемлемо — PEP 562 `__getattr__` для `calibration` (отдельное решение, не в этом PR).

---

*Следующая часть ТЗ: `holonomy.py` + `interventions.py` (зависят от `spectral.top_r_svd` и `bridge.py` соответственно). `noise.py`, `permutation.py`, `topology.py`, `pipeline.py` — после.*