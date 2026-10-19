# ТЗ на новые модули `prism/`, часть 2: `holonomy.py` + `interventions.py`

**Область:** два модуля, которые зависят от `spectral.top_r_svd` (часть 1) и `bridge.py`/`jacobian.py` (существующие). `holonomy.py` — чистые геометрические примитивы (без модели). `interventions.py` — спецификация правок и их применение к модели (с моделью).

**Ключевые архитектурные решения (зафиксированы до деталей):**

1. **`holonomy.py` — только примитивы.** `procrustes_align`, `holonomy_accumulate`, `holonomy_along_path`, `holonomy_along_2d_loop`. Конкретные path (`W₀→W₁₀₀→W₀`) и loop (`V₁,V₂` vs `PC₁,PC₂`) — в `experiments/exp_T2.py`.
2. **`interventions.py` — `delta_W` как `[K, D_S]`, не `[V, D_S]`.** Правки применяются только к `lm_head.weight[token_set]`; остальные строки не меняются. Это спасает память (512 KB vs 600 MB на edit) и соответствует семантике (`J` зависит только от `token_set` строк).
3. **`apply_edit` — context manager с `try/finally`.** Модель обязана восстанавливаться даже при исключении. Это **отдельный** примитив от `apply_edit_and_cache`; второй использует первый.
4. **`apply_edit_and_cache` кэширует только `h_t` и `p_t`, не `J`.** J реконструируется на лету через `compute_jacobian_for_tokens(h_override=h_t[l][t])` + `SVDCache` — это дешевле по диску в ~30× и позволяет любой `r` без пересчёта.

---

# Модуль 3: `prism/holonomy.py`

## 3.1. Назначение

Геометрические примитивы для вычисления голономии (Procrustes-остатка вдоль замкнутого цикла) в **пространстве подпространств** (Грассманиан `Gr(r, K)` для левых `U_r` и `Gr(r, D_S)` для правых `V_r`). Никаких forward-проходов, никакой зависимости от модели. **Только математика над тензорами.**

**Что такое голономия здесь:** для последовательности ортонормированных подпространств `U_0, U_1, ..., U_{N-1}, U_0` (замкнутый цикл в пространстве параметров/активаций) мы хотим измерить «накопленное скручивание» — насколько композиция Procrustes-выравниваний между соседними подпространствами отличается от тождественного преобразования. Это gauge-инвариантная величина: если все `U_i` заменить на `U_i Q_i` (смена базиса внутри подпространства), `κ` не меняется.

## 3.2. Публичный API

```python
def procrustes_align(
    U1: torch.Tensor, U2: torch.Tensor,
) -> tuple[torch.Tensor, float]: ...

def holonomy_accumulate(
    U_list: list[torch.Tensor],
    closed_loop: bool = True,
) -> tuple[float, list[float]]: ...

def holonomy_along_path(
    path_fn: Callable[[float], tuple[torch.Tensor, torch.Tensor]],
    thetas: Sequence[float],
    closed_loop: bool = True,
) -> dict: ...

def holonomy_along_2d_loop(
    base: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    epsilon: float,
    n_theta: int,
    measure_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> dict: ...
```

**Никаких `parameter_space_holonomy` / `representation_holonomy_sens` / `_canon`** — это композиты, живущие в `experiments/exp_T2.py`. Здесь — только примитивы, которые эти композиты используют.

## 3.3. Контракты по функциям

### `procrustes_align(U1, U2)`

| Аспект | Значение |
|---|---|
| Вход | `U1, U2: [K, r]` — ортонормированные (`U^T U ≈ I_r`), одинаковой формы |
| Выход | `(R, residual_sq)`, `R: [r, r]` ортогональная, `residual_sq: float` |
| Семантика | `R` минимизирует `‖U1 @ R − U2‖_F`; `residual_sq = ‖U1 R − U2‖_F²` |
| Алгоритм | `M = U2^T @ U1` (`[r, r]`); `P, s, Q^T = svd(M)`; `R = Q @ P^T`; `residual_sq = max(0, 2r − 2·Σs)` |
| Знаковая неоднозначность SVD | `R` **инвариантна** к одновременной смене знака столбцов `U1, U2` (проверка в selfcheck). `residual_sq` тоже инвариантен |
| Вырожденный SVD | Если `s_i = s_j` для `i ≠ j`, `R` не единственна; `residual_sq` всё равно определён корректно. **Не** бросать ошибку; документировать в docstring, что для вырожденного случая `R` — одна из допустимых |
| Ортонормальность входа | Проверка `‖U^T U − I‖_F < 1e-4` (для float32; для float64 — `1e-6`); если нет — `ValueError` (молчаливая работа с неортонормированными даёт неверный `R`) |
| dtype | Вход `float32`/`float64`; `M = U2^T U1` вычисляется в `float64` для стабильности SVD; результат приводится к dtype `U1` |
| Ошибки | Форма не совпадает → `ValueError`; `U1.dim() != 2` → `ValueError`; неортонормированные → `ValueError` |

**Формула residual:** `‖U1 R − U2‖_F² = ‖U1 R‖_F² + ‖U2‖_F² − 2 tr(R^T U1^T U2) = 2r − 2 tr(M^T R)`. При `R = Q P^T` и `M = P Σ Q^T`: `tr(M^T R) = tr(Q Σ P^T Q P^T) = tr(Σ P^T P) = tr(Σ) = Σs`. Значит `residual_sq = 2r − 2 Σs` — это **точная** формула, не приближение. `clamp(min=0)` — только против floating-point `−1e-16`.

### `holonomy_accumulate(U_list, closed_loop=True)`

| Аспект | Значение |
|---|---|
| Вход | `U_list: list[[K, r]]`, `len ≥ 2`; `closed_loop: bool` |
| Выход | `(kappa, per_step_residuals)` — `kappa: float`, `per_step_residuals: list[float]` |
| Семантика | `R_total = R_{0,1} @ R_{1,2} @ ... @ R_{N-2,N-1}` (и `@ R_{N-1,0}` при `closed_loop`); `kappa = ‖R_total − I_r‖_F` |
| Порядок умножения | **Левое умножение в порядке цикла:** `R_total = R_{0,1}`, затем `R_total = R_total @ R_{1,2}`, и т.д. Это соответствует композиции отображений `U_0 → U_1 → ... → U_0` |
| `closed_loop=False` | Открытый путь: `R_total = R_{0,1} @ ... @ R_{N-2,N-1}`; `kappa` измеряет накопленный параллельный перенос (не голономию в строгом смысле; задокументировать в docstring) |
| `len(U_list) == 2` | Одна пара `(U_0, U_1)`; при `closed_loop=True` — две пары (туда-обратно); `kappa ≈ 0` всегда (обратный Procrustes — точное обращение) |
| `len(U_list) < 2` | `ValueError` |
| `per_step_residuals` | `residual_sq` от каждого `procrustes_align` в порядке цикла; `len = len(U_list) - 1` или `len(U_list)` при `closed_loop=True` |
| dtype | Внутри `R_total` — `float64`; `kappa` возвращается как `float` (Python) |

**Замечание о композиции:** `R_{i,i+1}` — это матрица, такая что `U_{i+1} ≈ U_i @ R_{i,i+1}`. Тогда полный цикл: `U_0 @ R_{0,1} @ R_{1,2} @ ... @ R_{N-1,0} ≈ U_0`. Если `R_total = R_{0,1} R_{1,2} ... R_{N-1,0} ≈ I`, цикл тривиален (плоское расслоение). `kappa = ‖R_total − I‖_F` измеряет отклонение от тривиальности. При `closed_loop=False` полный цикл не замыкается, но `kappa` всё равно измеряет накопленный «сдвиг».

### `holonomy_along_path(path_fn, thetas, closed_loop=True)`

| Аспект | Значение |
|---|---|
| Вход | `path_fn: float → (U_r, V_r)`, где `U_r: [K, r]`, `V_r: [r, D_S]`; `thetas: Sequence[float]` (`len ≥ 2`); `closed_loop: bool` |
| Выход | `dict` с ключами: `kappa_U`, `kappa_V`, `per_step_U`, `per_step_V`, `thetas` |
| Семантика | Для каждого `θ ∈ thetas` вызывает `path_fn(θ)` → `(U_r, V_r)`; накапливает `holonomy_accumulate` по `U` и по `V` |
| `path_fn` ответственность | **Вся модельная логика:** установка весов, forward, SVD, взятие top-r. `holonomy_along_path` не знает о модели |
| Замыкание цикла | `closed_loop=True` → последний элемент `thetas` **не** должен совпадать с первым (это добавит тривиальный Procrustes `I` в конце, что не влияет на κ, но дублирует измерение). `holonomy_accumulate` сам замыкает цикл, добавляя `R_{N-1,0}` |
| Ошибки | `len(thetas) < 2` → `ValueError`; `path_fn` возвращает не кортеж → `TypeError` с понятным сообщением |
| Детерминизм | Полностью определяется `path_fn` и `thetas` |

**Зачем нужна:** parameter-space holonomy (цикл `W₀→W₁₀₀→W₀`) — это `path_fn(θ) = setup_model_and_compute_UV(θ)`, `thetas = linspace(0, 2, n_steps+1)`. Обёртка в `experiments/exp_T2.py` реализует `path_fn` с `lm_head`-оптимизацией (`J(θ) = W(θ) C_l`).

### `holonomy_along_2d_loop(base, b1, b2, epsilon, n_theta, measure_fn)`

| Аспект | Значение |
|---|---|
| Вход | `base: [D]` или `[1, 1, D]` — базовая точка; `b1, b2: [D]` — два ортонормированных направления (проверка `|b1·b2| < 1e-4`, `‖b1‖ ≈ 1`, `‖b2‖ ≈ 1`); `epsilon: float > 0`; `n_theta: int ≥ 4`; `measure_fn: [D] → (U_r, V_r)` |
| Выход | `dict` с ключами: `kappa_U`, `kappa_V`, `per_step_U`, `per_step_V`, `h_perturbed` (list of `[D]` — для диагностики) |
| Петля | `h(θ) = base + ε (cos θ · b1 + sin θ · b2)`, `θ ∈ {0, 2π/n_theta, ..., 2π(n_theta-1)/n_theta}` (без `2π`, замыкание делает `holonomy_accumulate`) |
| `measure_fn` ответственность | Forward, SVD, top-r. `holonomy_along_2d_loop` только формирует `h(θ)` и вызывает `measure_fn` |
| `n_theta` | Минимум 4 (иначе петля — квадрат, аппроксимация грубая); рекомендуется 16. При `n_theta < 4` → `ValueError` |
| `epsilon` | Должен быть `> 0` и сопоставим с масштабом `‖base‖` (например, `0.1 · ‖base‖`); проверка `epsilon > 0` → иначе `ValueError` |
| Проверка `b1, b2` | `‖b1 − 1‖ < 1e-4`, `‖b2‖ ≈ 1`, `|b1·b2| < 1e-4`; если нет — `ValueError` (иначе петля не ортогональна и `κ` не интерпретируется как голономия) |
| `base` форма | `[D]`, `[1, D]`, `[1, 1, D]` — нормализуется к `[D]`; `b1, b2` — `[D]` |
| Ошибки | Не ортонормированные `b1, b2` → `ValueError`; `n_theta < 4` → `ValueError`; `epsilon ≤ 0` → `ValueError` |

**Зачем нужна:** representation holonomy. Для `sens`: `b1 = V1`, `b2 = V2` (top-2 правых сингулярных вектора `J(base)`). Для `canon`: `b1 = PC1`, `b2 = PC2` (главные компоненты `h_batch`). Обёртки — в `experiments/exp_T2.py`.

## 3.4. Внутренние детали реализации

- **`_validate_orthonormal(U, tol)`** — приватная функция; проверяет `‖U^T U − I‖_F < tol`. Используется в `procrustes_align`.
- **`_check_loop_basis(b1, b2, tol)`** — приватная; проверяет `‖b1‖, ‖b2‖ ≈ 1`, `|b1·b2| < tol`. Используется в `holonomy_along_2d_loop`.
- **Все внутренние SVD** (`procrustes_align`) — в `float64`; результат приводится к `float32` при возврате (для консистентности с остальными модулями).
- **`torch.linalg.svd` в `procrustes_align`** — `full_matrices=False` (матрица `M` — `[r, r]`, полная и так).
- **Никаких состояний** (нет кэшей, нет глобалов). Каждый вызов — pure.
- **`per_step_residuals`** — для диагностики: если `kappa` велик, но `per_step_residuals` все малы, ошибка в композиции; если `kappa` мал, а `per_step` один велик, ошибка в одном шаге.

## 3.5. Взаимодействие с существующим

- **`spectral.top_r_svd`** — используется в `path_fn` / `measure_fn`, **не** в `holonomy.py`. `holonomy.py` работает с уже готовыми `U_r`, `V_r`.
- **`esg.py`** — не пересекается. ESG — про усреднённые грамианы, holonomy — про последовательность подпространств. Не путать.
- **`paper.md` §8.6, §8.7** — прямая ссылка в docstrings.
- **`experiments/exp28_leaf_structure.py`** (существующий, `κ` Procrustes) — если там есть локальный `_procrustes` / `_accumulate`, заменить на `prism.holonomy.*`. **Не дублировать.** Проверить через `grep -rn "procrustes\|holonomy" experiments/`.

## 3.6. Selfcheck (`tests/test_holonomy.py`)

1. **`procrustes_align` identity:** `U = torch.eye(8)[:, :4]`; `R, res = procrustes_align(U, U)`; `assert torch.allclose(R, torch.eye(4), atol=1e-6)`; `assert abs(res) < 1e-10`.
2. **`procrustes_align` rotation:** `Q, _ = torch.linalg.qr(torch.randn(4, 4))`; `R, res = procrustes_align(U, U @ Q)`; `assert torch.allclose(R, Q, atol=1e-5)`; `res ≈ 0`.
3. **`procrustes_align` sign invariance:** `U2 = U1.clone(); U2[:, 0] = -U2[:, 0]`; `R, res = procrustes_align(U1, U2)`; сравнить с `procrustes_align(U1, U1)` — `R` отличается (не I), но `res ≈ 0` (расстояние между подпространствами 0). Это **проверка семантики**: Procrustes выравнивает подпространства, а не базисы.
4. **`procrustes_align` residual formula:** `U1 = torch.randn(8, 4); U1, _ = torch.linalg.qr(U1)`; `U2 = torch.randn(8, 4); U2, _ = torch.linalg.qr(U2)`; `R, res = procrustes_align(U1, U2)`; `assert abs(res - torch.norm(U1 @ R - U2).pow(2).item()) < 1e-6`.
5. **`procrustes_align` non-orthonormal → ValueError.**
6. **`holonomy_accumulate` trivial:** `U_list = [U, U, U]`; `kappa, per_step = holonomy_accumulate(U_list)`; `assert kappa < 1e-8`; `per_step` все ≈ 0.
7. **`holonomy_accumulate` constant rotation:** `U_list = [U @ Q^i for i in range(n)]` с фиксированной `Q`; при `closed_loop=False`: `kappa ≈ ‖Q^{n-1} − I‖`; при `closed_loop=True`: `kappa ≈ ‖Q^n − I‖`.
8. **`holonomy_accumulate` closed vs open:** для нетривиального пути, `kappa(closed) ≠ kappa(open)` — sanity check.
9. **`holonomy_along_path` constant path:** `path_fn = lambda θ: (U, V)` для всех θ; `kappa_U ≈ 0`, `kappa_V ≈ 0`.
10. **`holonomy_along_path` linear rotation:** `path_fn(θ) = (U @ Q(θ), V)` с `Q(θ) = exp(θ A)` для `A` антисимметричной; `kappa_U ≈ ‖exp(2π A) − I‖` при `closed_loop=True` и `thetas = linspace(0, 2π, n)`.
11. **`holonomy_along_2d_loop` zero epsilon (не должно падать, но κ=0 по смыслу):** тест на `epsilon > 0`; с `epsilon=1e-10` — κ ≈ 0.
12. **`holonomy_along_2d_loop` trivial measure:** `measure_fn = lambda h: (U_const, V_const)`; κ ≈ 0.
13. **`holonomy_along_2d_loop` orthonormal basis check:** `b1 = b2` → `ValueError`; `‖b1‖ = 2` → `ValueError`.
14. **`n_theta < 4` → ValueError.**
15. **Gauge-invariance:** для `U_list` и `U_list' = [U_i @ Q_i for i in range(n)]` с ортогональными `Q_i`: `kappa(U_list) ≈ kappa(U_list')` (в пределах численной ошибки). Это **фундаментальное свойство** голономии; его нарушение — баг.

## 3.7. Ограничения и риски

- **`residual_sq` при вырожденном SVD:** `R` не единственна, но `residual_sq` — да. Потребители не должны полагаться на `R` при `s_i ≈ s_j`.
- **Плохая обусловленность:** при `‖U1 U2^T‖` близком к сингулярному (все `s_i ≈ 0`), `R` вычислительно нестабильна; `residual_sq` всё равно близок к `2r` (максимум). Предупреждение в docstring.
- **`holonomy_along_path` с `path_fn`, который медленный** — вся стоимость в `path_fn`; функция не кэширует.
- **`closed_loop=True` с `thetas` длины 2** — тривиальный цикл (туда-обратно); κ ≈ 0 всегда. Потребитель должен использовать `n_steps ≥ 3`.
- **`holonomy_along_2d_loop` с `epsilon` большим `‖base‖`** — петля уходит далеко от рабочей точки; `measure_fn` может дать вырожденные `U`, `V`. Задокументировать: `epsilon ≲ 0.2·‖base‖`.

## 3.8. Что НЕ входит в `holonomy.py`

- `parameter_space_holonomy` — `experiments/exp_T2.py`.
- `representation_holonomy_sens` / `_canon` — `experiments/exp_T2.py`.
- Работа с `model`, `lm_head` — `interventions.py` и `experiments/`.
- `SVDCache` — `cache.py`.

---

# Модуль 4: `prism/interventions.py`

## 4.1. Назначение

Спецификация правок (`EditSpec`) и их применение к модели. Пять типов правок для T0: `base`, `readout_pull`, `random_ortho`, `random_matched`, `tail_only`. Плюс `apply_edit` (context manager) и `apply_edit_and_cache` (с сохранением `h_t`, `p_t`).

**Ключевые решения:**

1. **`delta_W` как `[K, D_S]`, применяется к `lm_head.weight[token_set]`.** Не `[V, D_S]`. Экономия памяти в ~1200×.
2. **`EditSpec` знает свой `token_set`.** Не пересчитывается при `apply_edit`.
3. **`apply_edit` восстанавливает модель в `finally`.** Даже при исключении.
4. **`apply_edit_and_cache` кэширует только `h_t`, `p_t`** (не `J`); J реконструируется через `compute_jacobian_for_tokens(h_override=...)` + `SVDCache`.

## 4.2. Публичный API

```python
@dataclass(frozen=True)
class EditSpec:
    name: str
    delta_W: torch.Tensor | None       # [K, D_S] or None for base
    token_set: tuple[int, ...]          # frozen tuple for hashability
    target_layer: str                   # "lm_head" (only supported)
    norm_frobenius: float
    projection_onto_top_r: float        # ||P_topr delta_W||_F / ||delta_W||_F
    seed: int | None
    meta: dict                          # extra info (target_J_p hash, lr, steps, etc.)


def make_readout_pull(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    token_set: list[int],
    target_J_p: torch.Tensor,           # [K, D_S]
    n_steps: int = 100,
    lr: float = 1e-3,
    kl_leash_weight: float = 0.1,
    seed: int = 0,
    verbose: bool = False,
) -> EditSpec: ...


def make_random_ortho(
    delta_W_ref: torch.Tensor,          # [K, D_S]
    top_r_basis: torch.Tensor,           # [K, r]
    token_set: list[int],
    seed: int,
) -> EditSpec: ...


def make_random_matched(
    delta_W_ref: torch.Tensor,          # [K, D_S]
    top_r_basis: torch.Tensor,           # [K, r]
    token_set: list[int],
    seed: int,
) -> EditSpec: ...


def make_tail_only(
    delta_W_ref: torch.Tensor,          # [K, D_S]
    top_r_basis: torch.Tensor,           # [K, r]
    token_set: list[int],
) -> EditSpec: ...


def make_base(token_set: list[int]) -> EditSpec: ...


@contextmanager
def apply_edit(model, edit: EditSpec): ...


def apply_edit_and_cache(
    model,
    edit: EditSpec,
    samples: list[dict],                 # [{"input_ids": [1,S], "attention_mask": [1,S]}, ...]
    layers: list[int],
    cache_dir: str,
    overwrite: bool = False,
) -> dict: ...
```

## 4.3. Контракты по компонентам

### `EditSpec` (dataclass)

| Поле | Тип | Значение |
|---|---|---|
| `name` | `str` | `"base"` / `"readout_pull"` / `"random_ortho"` / `"random_matched"` / `"tail_only"`; используется как имя поддиректории в кэше |
| `delta_W` | `Tensor \| None` | `[K, D_S]`, `None` для `base`; **относится к `lm_head.weight[token_set]`** |
| `token_set` | `tuple[int, ...]` | frozen для hashability; `len == delta_W.shape[0]` |
| `target_layer` | `str` | `"lm_head"` (только поддерживаемый; другие — `NotImplementedError` при `apply_edit`) |
| `norm_frobenius` | `float` | `‖delta_W‖_F`; `0.0` для `base` |
| `projection_onto_top_r` | `float` | `‖P_topr delta_W‖_F / ‖delta_W‖_F`; `0.0` для `base` |
| `seed` | `int \| None` | для `random_*`; `None` для детерминированных |
| `meta` | `dict` | произвольная метаинформация (для `readout_pull`: `target_J_p_hash`, `lr`, `n_steps`, `kl_leash_weight`) |

**Frozen=True:** `EditSpec` — immutable value object; все поля нельзя менять после создания. `meta` — `dict`, но не мутируется (копируется при создании).

### `make_readout_pull(...)`

| Аспект | Значение |
|---|---|
| Цель | Воспроизвести E-16a: 100 шагов Adam на `lm_head.weight[token_set]`, минимизируя `(1 − cos(J_S_p, target_J_p)) + kl_leash_weight · KL(p_base ‖ p_edit)` |
| `target_J_p` | `[K, D_S]` — целевой gauge-free Якобиан (например, из 4B-модели; или из другого слоя base-модели; выбор — за `exp_T0.py`) |
| Loss | `(1 − cos).mean() + kl_leash_weight · KL`; `cos` по строкам (`F.cosine_similarity`), KL по `token_set`-распределениям |
| `p_base` | Вычисляется один раз в начале (forward, softmax, slice `[token_set]`); **не** пересчитывается на каждом шаге |
| `J_S_p` на каждом шаге | Вычисляется через `compute_hidden_jacobian_rows(model, ..., token_set, p_base, W_rows, create_graph=True)` из `jacobian.py` (переиспользование, не дублирование) |
| Adam state | Только `lm_head.weight[token_set]` (срез); полная матрица не оптимизируется |
| Восстановление | `try/finally`: `W0 = lm_head.weight[token_set].detach().clone()`; в `finally` — `copy_(W0)` |
| `requires_grad` | Сохранить исходное состояние `lm_head.weight.requires_grad`; восстановить в `finally` |
| Детерминизм | `torch.manual_seed(seed)` в начале; `torch.use_deterministic_algorithms` **не** включать (Adam на GDN-модели может бросить) |
| Возврат | `EditSpec` с `delta_W = (lm_head.weight[token_set].detach() - W0)`; `norm_frobenius`, `projection_onto_top_r = 0.0` (top-r базис не известен на этом этапе; caller обновит через `_with_projection` — см. ниже) |
| Ошибки | `target_J_p.shape != (K, D_S)` → `ValueError`; `n_steps < 1` → `ValueError`; `lr <= 0` → `ValueError` |

**Про `projection_onto_top_r`:** `make_readout_pull` не знает `top_r_basis` (он нужен для `random_*`). Caller (в `exp_T0.py`) после создания `readout_pull` и вычисления `top_r_basis` (из base ESG) должен **обновить** `projection_onto_top_r` для контроля. Проще всего: `EditSpec` создаётся с `projection_onto_top_r=0.0`, затем `dataclasses.replace(edit, projection_onto_top_r=...)` — но `frozen=True` разрешает `replace`. Документировать это в docstring.

**Альтернатива:** добавить параметр `top_r_basis: Tensor | None = None` в `make_readout_pull`; если задан — вычислить `projection_onto_top_r` сразу. Это чище. **Принять эту альтернативу:** `top_r_basis` опциональный; если не задан — `projection_onto_top_r=0.0`.

### `make_random_ortho(...)`

| Аспект | Значение |
|---|---|
| Цель | Случайная правка той же `‖·‖_F`, **ортогональная** к `top_r_basis` (т.е. `P_topr ΔW ≈ 0`) |
| Алгоритм | 1. `G = torch.randn(K, D_S)`; 2. `P_topr = top_r_basis @ top_r_basis.T` — проектор `[K, K]`; 3. `G_proj = G − P_topr @ G` — ортогональная компонента; 4. Если `‖G_proj‖_F < 1e-6·‖G‖_F` (почти всё в top-r) → `RuntimeError` (нельзя построить контроль; caller должен уменьшить `r` или взять другую базу); 5. `delta_W = G_proj · (‖delta_W_ref‖_F / ‖G_proj‖_F)` — масштабирование |
| Детерминизм | `torch.manual_seed(seed)`; `G` — единственный источник случайности |
| Проверка после | `‖delta_W‖_F ≈ ‖delta_W_ref‖_F` (в пределах 1e-6); `‖P_topr delta_W‖_F / ‖delta_W‖_F < 1e-3`; если нет — `RuntimeError` с диагностикой |
| `norm_frobenius` | `‖delta_W‖_F` |
| `projection_onto_top_r` | `‖P_topr delta_W‖_F / ‖delta_W‖_F` — ожидается ≈ 0 |
| Ошибки | `delta_W_ref.shape != top_r_basis.shape[0] × D_S` → `ValueError`; `top_r_basis` не ортонормирован → `ValueError` |

**Почему `[K, K]` проектор:** `top_r_basis` — `[K, r]`, ортонормирован по столбцам (`U^T U = I_r`). Тогда `P = U U^T` — `[K, K]` проектор на span(U). `G − P G` — компонента `G` в ортогональном дополнении (в `R^K`). **Это не то же самое, что проекция `[K, D_S]` матрицы;** здесь `P` применяется к столбцам `G` (каждый столбец `[K]` проецируется в `R^K`). Это корректная операция: `ΔW[token_set]` — `[K, D_S]`, каждая строка — токен; `U` — базис в `R^K` (token-space), `P` проецирует векторы длины `K`. Умножение `P @ G` — `[K, K] @ [K, D_S]` = `[K, D_S]`, что и нужно.

### `make_random_matched(...)`

| Аспект | Значение |
|---|---|
| Цель | Случайная правка той же нормы, **сохраняющая** проекцию на `top_r_basis`, но с рандомизированной остальной частью |
| Алгоритм | 1. `ref_proj = top_r_basis @ (top_r_basis.T @ delta_W_ref)` — `[K, D_S]`, проекция reference на top-r; 2. `ref_residual = delta_W_ref − ref_proj`; 3. `G = torch.randn(K, D_S)`; `G_residual = G − top_r_basis @ (top_r_basis.T @ G)` — случайный в ортогональном дополнении; 4. `delta_W = ref_proj + G_residual · (‖ref_residual‖_F / ‖G_residual‖_F)` — сохраняет проекцию reference, случайная — только ортогональная часть |
| Проверка после | `‖delta_W‖_F ≈ ‖delta_W_ref‖_F`; `‖P_topr delta_W − P_topr delta_W_ref‖_F < 1e-4·‖delta_W_ref‖_F`; `projection_onto_top_r ≈ projection_onto_top_r(delta_W_ref)` |
| Ошибки | Как в `random_ortho`; плюс `‖ref_residual‖_F < 1e-6·‖delta_W_ref‖_F` (весь reference в top-r, нечего рандомизировать) → `RuntimeError` |

**Отличие от `random_ortho`:** `random_ortho` даёт **нулевую** проекцию на top-r; `random_matched` даёт **ту же**, что у reference. Сравнение этих двух контролей разделяет «эффект нормы» и «эффект попадания в сигнал»: если `κ_U` растёт только у `random_matched`, дело в top-r компоненте; если у обоих — дело в норме.

### `make_tail_only(...)`

| Аспект | Значение |
|---|---|
| Цель | Правка **только** хвостовых мод ESG (ортогональных `top_r_basis`) |
| Алгоритм | `delta_W = delta_W_ref − top_r_basis @ (top_r_basis.T @ delta_W_ref)` — сохраняет только хвост |
| Норма | `‖delta_W‖_F = ‖ref_residual‖_F ≤ ‖delta_W_ref‖_F`; **не** нормируется (tail-only — это именно то, что осталось, а не «как будто» полная норма) |
| `projection_onto_top_r` | `≈ 0` (по построению) |
| Ошибки | Как в `random_matched`; если `‖ref_residual‖_F < 1e-6·‖delta_W_ref‖_F` → `RuntimeError` (нечего оставлять) |

**Замечание:** `tail_only` — не «случайная правка в хвосте», а именно **reference минус top-r проекция**. Это контроль на специфичность: если Goodhart-эффект есть и у `readout_pull`, и у `tail_only`, а у `random_ortho` — нет, значит важна структура **reference**, а не просто попадание в хвост. Если эффект есть только у `readout_pull` — важна top-r компонента.

### `make_base(token_set)`

| Аспект | Значение |
|---|---|
| Возврат | `EditSpec(name="base", delta_W=None, token_set=tuple(token_set), target_layer="lm_head", norm_frobenius=0.0, projection_onto_top_r=0.0, seed=None, meta={})` |
| Назначение | Единообразная сигнатура с остальными edit-ами; `apply_edit(model, base)` — no-op |

### `apply_edit(model, edit)` (context manager)

| Аспект | Значение |
|---|---|
| Вход | `model` (Qwen3_5ForCausalLM), `edit: EditSpec` |
| Действие | `model.lm_head.weight[edit.token_set] += edit.delta_W` (in-place, `torch.no_grad()`); `yield`; в `finally` — восстановить |
| Восстановление | `W_orig = model.lm_head.weight[edit.token_set].detach().clone()` до правки; в `finally` — `model.lm_head.weight[edit.token_set].copy_(W_orig)` |
| `requires_grad` | Сохранить `model.lm_head.weight.requires_grad`; восстановить |
| `base` edit | `delta_W=None` → правка не применяется; `yield` без изменений (для единообразия с остальными) |
| `target_layer != "lm_head"` | `NotImplementedError` с сообщением «only lm_head supported in v6.0» |
| Вложенность | **Не поддерживается** (один edit за раз); повторный вход → `RuntimeError` (это защита от случайного `with apply_edit(e1): with apply_edit(e2):`) |
| Детерминизм | Правка детерминирована (не random в этой функции) |
| Ошибки | `delta_W.shape != (len(token_set), D_S)` → `ValueError` (проверка в начале) |

**Почему `requires_grad` восстанавливается:** если caller передал модель в `apply_edit` в тренировочном режиме (`requires_grad=True`), после выхода модель должна остаться в том же режиме, даже если внутри мы делали `no_grad()`. Это стандартная гигиена context manager.

### `apply_edit_and_cache(model, edit, samples, layers, cache_dir, overwrite=False)`

| Аспект | Значение |
|---|---|
| Действие | Для каждого `(sample_idx, sample)` и каждого `layer ∈ layers`: forward, сохранить `h_t` (`[S, D_S]`) и `p_t` (`[S, K]`) в `cache_dir/{edit.name}/L{layer:02d}/doc{sample_idx:04d}_{h,p}.pt` |
| Формат `h.pt` | `torch.save({"h": h_tensor, "meta": {...}})`; `h_tensor` — `[S, D_S]`, `dtype=model_dtype` (или `float32` — см. ниже) |
| Формат `p.pt` | `torch.save({"p": p_tensor, "meta": {...}})`; `p_tensor` — `[S, K]`, `float32` (для стабильности softmax при последующих операциях) |
| Мета | `{"edit": edit.name, "layer": layer, "doc": sample_idx, "S": S, "D_S": D_S, "K": K, "torch": torch.__version__, "model_dtype": str(model_dtype), "transformers": transformers.__version__, "seed": edit.seed}` |
| Forward | `model(**sample, output_hidden_states=True, use_cache=False)`; `p_t` = `softmax(model.lm_head(h_t)[:, :, token_set])` |
| Применение edit | `with apply_edit(model, edit):` — модель восстанавливается в `finally` |
| `overwrite=False` | Если файл уже существует — **не** перезаписывать (пропустить, но `assert` что shape/meta совпадают; иначе `RuntimeError`) |
| `overwrite=True` | Перезаписать (атомарно: `tmp` + `os.replace`) |
| Возврат | `dict` с метаданными о кэше: `{"root": cache_dir, "edit": edit.name, "n_docs": len(samples), "layers": layers, "S": S, "D_S": D_S, "K": K, "files": [...]}` |
| GDN | **Никакой** `_clone_gdn_states_on_entry` — мы в `no_grad`, in-place мутации безопасны (backward не вызывается). Документировать |
| Ошибки | `samples` пустой → `ValueError`; `layers` содержит невалидный индекс → `ValueError` (проверка `0 <= layer < n_layers`); `cache_dir` read-only (Kaggle FUSE) и `overwrite=True` → `RuntimeError` с явным сообщением (не молчаливый сбой) |
| `h` dtype | `float32` (не `model_dtype`), если модель в `fp16` — конверсия при сохранении; это упрощает downstream (все эксперименты работают в `float32`) |

**Почему `h_t` в `float32`:** downstream (`compute_jacobian_for_tokens` с `h_override`) приведёт `h` к `model_dtype` через `_normalize_h_override`; `float32` на диске — точнее и совместимее. Потеря памяти: 128 × 1024 × 4 = 512 KB vs 256 KB в fp16 — приемлемо.

**Почему `p_t` в `float32`:** softmax-значения; `gauge-free` проекция `P(p)` чувствительна к точности `p`; `float32` — стандарт.

**Почему не кэшируется `J`:** J реконструируется через `compute_jacobian_for_tokens(h_override=h_t[l][t], prefix_cache=...)`. Стоимость — один decoder-pass на позицию; для `S=128`, `K=128` — секунды. Диск экономится в 30×.

## 4.4. Внутренние детали реализации

- **`_validate_edit_spec(edit, D_S)`** — приватная; проверяет `delta_W.shape == (len(token_set), D_S)` (если `delta_W is not None`), `target_layer == "lm_head"`.
- **`_projection_fraction(delta_W, top_r_basis)`** — `‖P_topr delta_W‖_F / ‖delta_W‖_F`; используется в `make_*` для заполнения `projection_onto_top_r`.
- **`_scale_to_norm(delta_W, target_norm)`** — `delta_W * (target_norm / ‖delta_W‖_F)`; используется в `make_random_ortho` и `make_random_matched`.
- **`_resolve_lm_head_weight(model)`** — `model.lm_head.weight` (flat) или `model.lm_head.weight` через wrapper; в Qwen3.5-0.8B оба варианта дают один и тот же `lm_head`. Fallback: `model.get_output_embeddings().weight` (для экзотических wrapper-ов).
- **Атомарная запись кэша** — `tmp = f"{final}.tmp.{os.getpid()}.{uuid4().hex[:8]}"`; `torch.save(...); os.replace(tmp, final)`.
- **Чтение `meta.json` при `overwrite=False`** — если есть, сравнить с текущим `edit.name`, `layer`, `doc`, `S`, `D_S`, `K`; если не совпадает — `RuntimeError`. Если совпадает — пропустить (файлы `h.pt`/`p.pt` не перезаписываются).

## 4.5. Взаимодействие с существующим

- **`bridge.jacobian_gap`** — **не используется** в `make_readout_pull` (loss строится через `compute_hidden_jacobian_rows` + `F.cosine_similarity`). Причина: `jacobian_gap` работает на уже готовых `J`; для оптимизации нам нужны `create_graph=True` строки, что даёт `compute_hidden_jacobian_rows`.
- **`jacobian.compute_hidden_jacobian_rows`** — используется в `make_readout_pull` (единственная точка, где нужен `create_graph=True`); `chunk_size` — параметр; для GDN — обёрнут в `_clone_gdn_states_on_entry` **внутри** этой функции (уже сделано в `jacobian.py`).
- **`jacobian.compute_jacobian_for_tokens`** — используется в `apply_edit_and_cache`? **Нет**; `apply_edit_and_cache` делает простой forward без градиентов, только для `h_t`, `p_t`. J реконструируется в `experiments/`, а не в `apply_edit_and_cache`.
- **`cache_io.find_targets_dir`** — **не используется** в `interventions.py` (это для read-only поиска существующих кэшей; интервенции пишут новые). Если пользователь передаёт `cache_dir` с `overwrite=False` и хочет использовать существующий `exp27b`-кэш — это ответственность `exp_T0.py`, не `interventions.py`.
- **`_clone_gdn_states_on_entry`** — **не нужен** в `apply_edit_and_cache` (`no_grad`); **нужен** в `make_readout_pull` (внутри `compute_hidden_jacobian_rows`, который обёрнут; но `make_readout_pull` сам должен быть обёрнут в `torch.enable_grad()` для backward через `lm_head.weight`).
- **`paper.md` §E-16a** — `make_readout_pull` воспроизводит протокол; `paper.md` §19.A — ссылки на `EditSpec`.
- **`experiments/exp16a_readout_pull.py`** (если существует) — после появления `prism.interventions.make_readout_pull` **удалить дублирующую логику**, оставить тонкую обёртку. Проверить `grep -rn "readout_pull\|lm_head.*Adam" experiments/`.

## 4.6. Selfcheck (`tests/test_interventions.py`)

1. **`make_base`:** `edit = make_base([1, 2, 3])`; `edit.delta_W is None`; `edit.norm_frobenius == 0.0`.
2. **`make_random_ortho` norm:** `delta_W_ref = torch.randn(K, D_S)`; `top_r_basis = qr(randn(K, r))`; `edit = make_random_ortho(delta_W_ref, top_r_basis, ..., seed=0)`; `abs(edit.norm_frobenius - delta_W_ref.norm()) < 1e-5`.
3. **`make_random_ortho` projection:** `P = top_r_basis @ top_r_basis.T`; `assert (P @ edit.delta_W).norm() / edit.delta_W.norm() < 1e-4`.
4. **`make_random_ortho` determinism:** два вызова с `seed=0` → идентичные `delta_W`.
5. **`make_random_matched` projection preserved:** `edit = make_random_matched(delta_W_ref, top_r_basis, ...)`; `assert (P @ (edit.delta_W − delta_W_ref)).norm() < 1e-4 * delta_W_ref.norm()`.
6. **`make_random_matched` norm:** `abs(edit.norm_frobenius - delta_W_ref.norm()) < 1e-5`.
7. **`make_tail_only`:** `edit = make_tail_only(delta_W_ref, top_r_basis, ...)`; `assert (P @ edit.delta_W).norm() / edit.delta_W.norm() < 1e-4`; `assert edit.delta_W.norm() < delta_W_ref.norm()` (строго, если reference имеет ненулевую top-r компоненту).
8. **`make_readout_pull`:** на tiny-модели с `n_steps=3`, `target_J_p = random`, `assert edit.delta_W.shape == (K, D_S)`; `assert model.lm_head.weight[token_set]` восстановлен после (побитово).
9. **`make_readout_pull` Adam updates:** `edit.norm_frobenius > 0` (Adam что-то изменил).
10. **`make_readout_pull` model restoration:** `W_before = model.lm_head.weight.clone()`; `make_readout_pull(...)`; `assert torch.equal(model.lm_head.weight, W_before)`.
11. **`make_readout_pull` `requires_grad` restoration:** `model.lm_head.weight.requires_grad_(False)`; `make_readout_pull(...)`; `assert not model.lm_head.weight.requires_grad`.
12. **`apply_edit` no-op for base:** `edit = make_base(...)`; `W_before = model.lm_head.weight.clone()`; `with apply_edit(model, edit): pass`; `assert torch.equal(model.lm_head.weight, W_before)`.
13. **`apply_edit` restores on exception:** `try: with apply_edit(model, edit): raise RuntimeError; except: pass`; `assert torch.equal(model.lm_head.weight, W_before)`.
14. **`apply_edit` model state visible inside:** `with apply_edit(model, edit): assert not torch.equal(model.lm_head.weight[token_set], W_before[token_set])`.
15. **`apply_edit` nesting → RuntimeError.**
16. **`apply_edit` wrong layer → NotImplementedError.**
17. **`apply_edit_and_cache` round-trip:** на tiny-модели с 2 samples; `cache = apply_edit_and_cache(model, edit, samples, layers=[1], cache_dir=tmpdir)`; `assert os.path.exists(f"{tmpdir}/{edit.name}/L01/doc0000_h.pt")`; load, `assert h.shape == (S, D_S)`; `assert p.shape == (S, K)`.
18. **`apply_edit_and_cache` model restored:** `W_before`; `apply_edit_and_cache(...)`; `assert torch.equal(model.lm_head.weight, W_before)`.
19. **`apply_edit_and_cache` overwrite=False skips:** повторный вызов с `overwrite=False` — файлы не перезаписаны (mtime не изменился).
20. **`apply_edit_and_cache` meta consistency:** `meta.json` содержит `edit`, `layer`, `doc`, `S`, `D_S`, `K`, `torch`, `transformers`.

## 4.7. Ограничения и риски

- **`make_readout_pull` с `target_J_p` из внешнего источника** — caller отвечает за семантическую согласованность (`K`, `D_S`, gauge-free, тот же `token_set`). `make_readout_pull` проверяет только `shape`; не проверяет `J`-семантику.
- **Adam на GDN** — `compute_hidden_jacobian_rows` обёрнут в `_clone_gdn_states_on_entry`; но Adam state и backward через GDN — empirically работают на T4; на других версиях `transformers` могут быть проблемы. Документировать.
- **`delta_W` как `[K, D_S]`** — это **соглашение**; если кто-то применит `delta_W` ко всей `lm_head.weight` — ошибка. `apply_edit` проверяет shape и применяет только к `token_set` строкам; но `EditSpec` может быть сохранён на диск и переиспользован неверно — caller должен знать конвенцию.
- **`apply_edit_and_cache` не кэширует `J`** — это решение; downstream должен реконструировать. Если downstream реконструирует часто — использовать `SVDCache` (часть 1). Это **намеренно**, чтобы не дублировать хранилище `J` с `SVDCache`.
- **`make_tail_only` с `‖ref_residual‖_F ≈ 0`** — reference целиком в top-r; tail-only не имеет смысла → `RuntimeError`. Это защита от бессмысленного контроля; caller должен либо уменьшить `r`, либо выбрать другой reference.
- **`make_readout_pull` с `kl_leash_weight` > 0** — требует корректной `p_base`; если модель в eval и `p_base` вычислен корректно, KL — валидная метрика. Но при `kl_leash_weight` слишком большом Adam может сойтись к `delta_W ≈ 0` — это не баг, а сигнал; caller должен выбрать разумное значение (E-16a: 0.1).

## 4.8. Что НЕ входит в `interventions.py`

- `SVDCache` — `cache.py` (часть 1).
- `compute_jacobian_for_tokens` с `h_override` — `jacobian.py` (уже в существующем, добавлено в предыдущей правке).
- `bridge.jacobian_gap` — не используется здесь.
- Конкретные цели для `make_readout_pull` (`target_J_p`) — выбор в `exp_T0.py`.
- Forward для получения `J` — downstream, не здесь.

---

# Сводный чек-лист для этой части

| Пункт | Модуль | Проверка |
|---|---|---|
| `procrustes_align` (identity, rotation, sign-invariance, residual) | `holonomy.py` | selfcheck §3.6.1–3.6.5 |
| `holonomy_accumulate` (trivial, constant rotation, open vs closed) | `holonomy.py` | selfcheck §3.6.6–3.6.8 |
| `holonomy_along_path` (constant, linear rotation) | `holonomy.py` | selfcheck §3.6.9–3.6.10 |
| `holonomy_along_2d_loop` (trivial measure, orthonormal check, errors) | `holonomy.py` | selfcheck §3.6.11–3.6.15 |
| Gauge-invariance под заменой базиса `U_i → U_i Q_i` | `holonomy.py` | selfcheck §3.6.15 (фундаментальное свойство) |
| `EditSpec` frozen, `delta_W: [K, D_S]`, `token_set: tuple` | `interventions.py` | selfcheck §4.6.1 |
| `make_base` no-op | `interventions.py` | selfcheck §4.6.1, §4.6.12 |
| `make_readout_pull` (loss, Adam, restore, `requires_grad`) | `interventions.py` | selfcheck §4.6.8–4.6.11 |
| `make_random_ortho` (norm, projection, determinism) | `interventions.py` | selfcheck §4.6.2–4.6.4 |
| `make_random_matched` (projection preserved, norm) | `interventions.py` | selfcheck §4.6.5–4.6.6 |
| `make_tail_only` (zero top-r projection, smaller norm) | `interventions.py` | selfcheck §4.6.7 |
| `apply_edit` (no-op, restore on exception, visible inside, nesting, wrong layer) | `interventions.py` | selfcheck §4.6.12–4.6.16 |
| `apply_edit_and_cache` (round-trip, restore, overwrite=False, meta) | `interventions.py` | selfcheck §4.6.17–4.6.20 |
| `bridge.jacobian_gap` **не дублируется**; `compute_hidden_jacobian_rows` **используется** | `interventions.py` | grep: `from .jacobian import compute_hidden_jacobian_rows` |
| `compute_jacobian_for_tokens(h_override=...)` **не используется** в `apply_edit_and_cache` (только в downstream) | `interventions.py` | grep: нет `compute_jacobian_for_tokens` в `interventions.py` |
| `cache_io.find_targets_dir` **не используется** в `interventions.py` | `interventions.py` | grep: нет `from .cache_io` |
| `_clone_gdn_states_on_entry` **не оборачивает** `apply_edit_and_cache` (no_grad) | `interventions.py` | grep: нет `_clone_gdn` в `apply_edit_and_cache` |
| `experiments/exp16a_readout_pull.py` (если есть) — **обёртка**, не дублирование | `experiments/` | ручная проверка |
| `experiments/exp28_leaf_structure.py` локальные `_procrustes` — **заменены** на `prism.holonomy` | `holonomy.py` | grep: нет локального `_procrustes` в `exp28` |

**Критерий готовности:**
- Все selfcheck из §3.6 и §4.6 проходят.
- `git diff` по `bridge.py`, `esg.py`, `jacobian.py`, `calibration.py`, `diagnostics.py`, `layer_selection.py`, `cache_io.py`, `spectral.py` (часть 1) — пустой (кроме уже согласованных правок).
- `import prism.holonomy` и `import prism.interventions` работают без `datasets`.
- Существующие тесты (`tests/test_*.py`) — зелёные.

**Отклонение от критерия «без `datasets`» (принято при исполнении, 2026-07):** как и для части 1 (`new_code_1.md`, P2.3): `import prism.holonomy`/`prism.interventions` идёт через `prism/__init__.py` → `calibration` → `datasets`. Директива: без lazy — окружение обязано иметь `datasets`. Отдельно: TZ selfcheck §3.6.7/§3.6.10 «closed κ ≈ ‖Q^n − I‖ / ‖exp(2πA) − I‖» **опровергнут численно** для групповых орбит (закрытый цикл по орбите имеет κ = 0: замыкающий шаг — точное обратное к накопленному произведению); эти значения — для *открытого* пути. Зафиксировано в docstring `holonomy_accumulate` и в тестах (`test_accumulate_constant_rotation`, `test_along_path_linear_rotation`); не-тривиальный closed κ ≠ 0 только для геометрических (не-орбитных) путей.

---

*Следующая часть ТЗ: `noise.py` + `permutation.py` + `topology.py` (три независимых модуля, зависят от `spectral`/`cache`, но не друг от друга). `pipeline.py` (ExperimentSpec + run_spec) — после. `experiments/exp_T0.py` … `exp_delta.py` — финальная часть, тонкие обёртки.*