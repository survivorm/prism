# ТЗ на правку существующих файлов `prism/` для серии v6.0

**Область:** только существующие модули. Новые модули (`holonomy.py`, `interventions.py`, `spectral.py` и т.д.) — отдельное ТЗ. Здесь: минимальные, обратно совместимые правки, без которых новая серия не запустится.

**Принцип:** любая правка существующего файла должна (а) сохранять поведение для существующих вызовов, (б) быть покрыта selfcheck, (в) не дублировать логику, которая уже есть в `bridge.py` / `cache_io.py` / `layer_selection.py`.

---

## 1. Сводная таблица правок

| Файл | Тип правки | Зачем нужна | Риск для существующих вызовов |
|---|---|---|---|
| `prism/jacobian.py` | `h_override` в `compute_jacobian_for_tokens`; respect `h` в `create_graph=True` в `compute_hidden_jacobian` | Representation holonomy требует `J(h_l(θ))` без пересчёта forward | Нет при `h_override=None` / `h=None` |
| `prism/esg.py` | `cross_positional_gramian(J_t, J_tp)` | `Δ_U(t, t')` для спектрального Goodhart (ε) | Нет; новая функция |
| `prism/calibration.py` | `build_global_token_set` (если отсутствует) | T0 строит `token_set` для intervention-кэшей | Нет; новая функция |
| `prism/__init__.py` | Публичные импорты | Единая точка входа для `experiments/` | Нет; namespace-файл |

---

## 2. `prism/jacobian.py`

### 2.1. Цель правки

Две независимые, но связанные задачи:

1. **`compute_jacobian_for_tokens`** должен уметь принимать **готовый `h`** (вместо того, чтобы извлекать его из full forward). Это ядро representation holonomy: для каждой θ нужно вычислить `J(h_l + ε(cos θ·V₁ + sin θ·V₂))`. Без `h_override` пришлось бы делать новый forward на каждую θ, что удорожает эксперимент в `n_θ ×`.

2. **`compute_hidden_jacobian`** уже имеет параметр `h` (для переиспользования `h` из KL-forward в exp17), но в `create_graph=True` пути он **молча игнорируется** — код безусловно перезаписывает `h` из forward. Это баг совместимости: docstring обещает «reuse a pre-computed h», а поведение не соответствует. Правка — сделать `h` respected в обоих путях.

### 2.2. Изменение 1: `h_override` в `compute_jacobian_for_tokens`

**Текущая сигнатура:**

```python
def compute_jacobian_for_tokens(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_set: list[int],
    prefix_cache=None,
) -> torch.Tensor:
```

**Новая сигнатура (обратно совместимая):**

```python
def compute_jacobian_for_tokens(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_set: list[int],
    prefix_cache=None,
    h_override: torch.Tensor | None = None,
) -> torch.Tensor:
```

**Семантика `h_override`:**

- Если `h_override is None` — **поведение побитово идентично текущему**. `h` извлекается из full forward, как и раньше. Никаких изменений.
- Если `h_override is not None` — используется **вместо** h, извлечённого из forward. `prefix_cache` по-прежнему строится из `input_ids[:, :-1]` (если не передан явно). Full forward (`_extract_hidden_state`) **не выполняется** — это экономит один forward на вызов.

**Валидация `h_override`** (в порядке проверки, до любых вычислений):

1. Принимаются формы `[1, 1, D]`, `[1, D]`, `[D]`. Нормализуется к `[1, 1, D]` через `reshape(1, 1, -1)`. Если форма не сводится — `ValueError`.
2. `D` должно совпадать с `model.config.hidden_size` (или `text_config.hidden_size` для wrapper). Если нет — `ValueError`.
3. Устройство: должно совпадать с `input_ids.device`. Если нет — `ValueError` (не автоматический перенос, чтобы не скрывать ошибки).
4. dtype: приводится к `model_dtype` через `.to(model_dtype)` (как в текущей ветке `h = h.to(model_dtype)`).
5. `h_override` **клонируется** (`h_override.detach().clone().requires_grad_(True)`), чтобы:
    - не мутировать тензор вызывающего (если он передал лист);
    - получить свежий лист (иначе `autograd.grad` пожалуется на non-leaf);
    - не тащить паразитные `CopyBackwards`-узлы.

**Изменение в теле** (псевдокод):

```python
# было:
if prefix_cache is None:
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
h = _extract_hidden_state(model, input_ids, attention_mask, layer_idx)
h = h.to(model_dtype).detach().requires_grad_(True)

# станет:
if prefix_cache is None:
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
if h_override is None:
    h = _extract_hidden_state(model, input_ids, attention_mask, layer_idx)
    h = h.to(model_dtype).detach().requires_grad_(True)
else:
    h = _normalize_h_override(h_override, model, input_ids)  # новая приватная функция
```

где `_normalize_h_override` реализует валидацию выше и возвращает `[1, 1, D]` detached-leaf.

**Важно:** `prefix_cache` строится **всегда** (не зависит от `h_override`), потому что контекст (позиции `0..S-2`) не меняется при переопределении последней позиции. Это согласуется с текущим поведением: `_run_prefix_pass` всегда вызывается, если `prefix_cache is None`.

**Документировать в docstring:**

> `h_override`: если задан — используется как скрытое состояние последней позиции на слое `layer_idx`, вместо извлечения из full forward. `prefix_cache` по-прежнему строится по `input_ids[:, :-1]`. Вызывающий отвечает за согласованность `h_override` и `prefix_cache` (например, `h_override` из той же последовательности, что и `input_ids`). Форма: `[1, 1, D]`, `[1, D]` или `[D]`. Device должен совпадать с `input_ids.device`. dtype приводится к dtype модели. Тензор клонируется и detach-ится.

**Почему не `h` для консистентности с `compute_hidden_jacobian`:** см. §2.3 — там параметр уже `h`, и переименование сломало бы обратную совместимость. Асимметрия имён (h_override здесь, h там) — намеренная, отражает разное происхождение параметра: в `compute_hidden_jacobian` `h` уже был, а здесь `h_override` — новое расширение. Зафиксировать в docstring обоих.

### 2.3. Изменение 2: respect `h` в `compute_hidden_jacobian` (create_graph=True)

**Текущая проблема.** В `create_graph=True` ветке:

```python
if create_graph:
    S = input_ids.shape[1]
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
    with torch.enable_grad():
        outputs = model(...)          # ← full forward всегда
    h = outputs.hidden_states[layer_idx + 1][:, -1:, :]  # ← безусловная перезапись
    ...
```

Переданный `h` игнорируется. В `create_graph=False` ветке он respected:

```python
S = input_ids.shape[1]
prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
if h is None:
    h = _extract_hidden_state(model, input_ids, attention_mask, layer_idx)
```

**Правка.** Сделать `create_graph=True` ветку симметричной:

```python
if create_graph:
    S = input_ids.shape[1]
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)

    if h is None:
        with torch.enable_grad():
            outputs = model(...)
        h = outputs.hidden_states[layer_idx + 1][:, -1:, :]  # [1, 1, D]
    # normalize: [1, D] / [D] -> [1, 1, D]
    h = _normalize_h_for_hidden_jacobian(h, model, input_ids)
    h = h.to(model_dtype)
    if h.grad_fn is None:
        h = h.detach().requires_grad_(True)

    # дальше — как было (h_norm через norm или decoder-to-norm, J_norm)
```

**Ключевое различие при `h_override + create_graph=True`:**

- Если `h` был получен из forward (h is None на входе) — `h.grad_fn` не None, градиенты текут к параметрам модели. **Поведение идентично текущему.**
- Если `h` передан (h_override) и disconnected — `h.grad_fn` is None, fallback делает его листом. Градиенты текут **только к h**, но **не** к параметрам модели. Это желаемое поведение для representation holonomy (нас интересует J как функция h, а не параметров). **Документировать в docstring** — иначе это footgun для тех, кто ожидал тренировки.

**Не переименовывать `h` → `h_override`:** параметр `h` уже используется (exp17, exp18, exp19 могут полагаться на `h=` kwarg). Переименование сломает обратную совместимость. Асимметрия с `compute_jacobian_for_tokens` — намеренная, зафиксировать в docstring.

**Не добавлять `h_override` алиас:** это усложнит API без пользы. Если очень нужно единообразие — отдельный PR после v6.0.

### 2.4. Изменение 3: документация

В docstring `compute_jacobian_for_tokens` добавить блок «Interaction with `compute_hidden_jacobian`»:

> `compute_jacobian_for_tokens` принимает `h_override`; `compute_hidden_jacobian` принимает `h`. Оба параметра означают «использовать этот h вместо вычисления из forward», но имена разные по историческим причинам. При `h_override=None` (или `h=None`) поведение побитово идентично версии до v6.0.

### 2.5. Selfcheck (новый файл `prism/tests/test_jacobian_h_override.py`)

1. **Identity:** `h = _extract_hidden_state(...)`; `J_a = compute_jacobian_for_tokens(..., h_override=None)`; `J_b = compute_jacobian_for_tokens(..., h_override=h)`. `assert torch.allclose(J_a, J_b, atol=1e-6)`.
2. **Shape normalization:** `h` в формах `[D]`, `[1, D]`, `[1, 1, D]` дают одинаковый J.
3. **Device mismatch:** `h_override` на CPU, модель на CUDA (или наоборот) → `ValueError` с понятным сообщением.
4. **D mismatch:** `h_override` с `D+1` → `ValueError`.
5. **Detach safety:** передать `h_override, requires_grad=True`; после вызова `assert not h_override.requires_grad` (не мутировали) и `J.grad_fn is None` (J — тензор, не граф, т.к. `h` был detached).
6. **`create_graph=True` + `h=None`:** identity с текущим поведением — `J_norm_a = compute_hidden_jacobian(..., create_graph=True)`; `J_norm_b = compute_hidden_jacobian(..., create_graph=True, h=None)` — равны.
7. **`create_graph=True` + `h` (leaf):** `J_norm = compute_hidden_jacobian(..., create_graph=True, h=h_leaf)`; `assert J_norm.requires_grad`; `J_norm.sum().backward()`; `assert h_leaf.grad is not None`; `assert any(p.grad is not None for p in model.parameters())` — **False** (граф не связан с параметрами; это ожидаемое поведение).

**Фикстура:** маленькая модель `Qwen3_5ForCausalLM` (или mock с `hidden_size=32`), `S=8`, `K=4`, `layer_idx=1`. CPU.

---

## 3. `prism/esg.py`

### 3.1. Цель правки

Добавить `Δ_U(t, t')` — кросс-позиционный Грамиан в token-space. Это ядро **ε** (спектральный Goodhart) и baseline для **T2'** (holonomy). `esg.py` — правильное место, потому что это уже «файл про Грамианы»; отдельный `gramian.py` ради одной функции — оверинжиниринг.

### 3.2. Новая функция

```python
def cross_positional_gramian(
    J_t: torch.Tensor,
    J_tp: torch.Tensor,
) -> torch.Tensor:
    """Cross-positional Gramian Δ_U(t, t') = J_t @ J_tp.T.

    Args:
        J_t:  [K, D] gauge-free Jacobian at position t.
        J_tp: [K, D] gauge-free Jacobian at position t'.

    Returns:
        [K, K] symmetric Gramian (symmetric iff t == t').

    Note:
        - NOT a pooling operation: each call uses exactly two positions.
          See module docstring for intra-trajectory pooling (compute_esg)
          and cross-sample pooling validity (E-0.1).
        - When t == t', result equals the inner product part of
          compute_esg([J_t]) for N=1: compute_esg([J_t])[0][0] ==
          eigenvalue_0(Δ_U(t, t)). This is checked in selfcheck.
        - Caller must ensure J_t, J_tp come from the same model / edit /
          token_set; the function does not check semantic compatibility,
          only shape / device / dtype (via _validate).
    """
    _validate([J_t, J_tp])
    return J_t.float() @ J_tp.float().T
```

**Детали:**

- Использует существующий `_validate(jacobians: list)` — переиспользование, не дублирование.
- `.float()` — потому что `_build_gramian` в `esg.py` уже работает в float, и `Δ_U` должен быть консистентен.
- Никакого `eigengap` внутри — это работа `prism/spectral.py` (отдельное ТЗ). `cross_positional_gramian` возвращает **матрицу**, а не метрики. Это важно для переиспользования: тот же `Δ_U` может использоваться для `ε_gap`, для проекции на top-r (holonomy baseline), для визуализации.

### 3.3. Модификация module docstring

Добавить абзац:

> **Cross-positional Gramian (`cross_positional_gramian`).** `Δ_U(t, t') = J_t J_{t'}^T` — token-корреляции между позициями. При `t = t'` совпадает с per-position ESG. Используется в `E-ε` (спектральный Goodhart) и как baseline для `E-T2'` (holonomy). Не путать с `compute_esg`, который пулит по траектории.

### 3.4. Selfcheck (новый файл `prism/tests/test_cross_positional_gramian.py`)

1. **Identity с ESG при t=t':** `J = torch.randn(K=8, D=16)`; `G = cross_positional_gramian(J, J)`; `ev, _ = compute_esg([J], n_components=8)`; `assert torch.allclose(torch.linalg.eigvalsh(G), ev, atol=1e-6)` (с учётом сортировки).
2. **Симметрия:** `Δ_U(t, t').T == Δ_U(t', t)`.
3. **Shape / device / dtype validation:** `_validate` уже проверяет; дублирующий тест — `J1` shape `(K, D)`, `J2` shape `(K+1, D)` → `ValueError`.
4. **float32 output:** даже если вход fp16/fp64, выход float32 (важно для численной стабильности при последующем `eigvalsh`).

---

## 4. `prism/calibration.py`

### 4.1. Цель правки

Убедиться, что `build_global_token_set` существует и доступен для T0. Если он уже есть (по аудиту «судя по импортам, есть»), правка = 0 строк.

### 4.2. Условная правка

**Проверить:** `grep -n "def build_global_token_set" prism/calibration.py`.

**Если отсутствует** — добавить, дословно как в `paper.md` §C.7:

```python
def build_global_token_set(
    model,
    calibration_samples: list[dict],
    K: int = 64,
) -> list[int]:
    """Union of top-K logits across calibration samples, sorted by ID.

    Mirrors paper.md §C.7. Deterministic (no sampling); returns sorted list
    so token_set is comparable across runs.

    Args:
        model: Qwen3_5ForCausalLM (any size).
        calibration_samples: list of {"input_ids": [1, S], "attention_mask": [1, S]}.
        K: top-K per sample.

    Returns:
        sorted list of token IDs, length <= K (could be < K if fewer unique
        tokens across samples; typically == K).
    """
    from collections import Counter

    token_counter: Counter = Counter()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for sample in calibration_samples:
            outputs = model(
                input_ids=sample["input_ids"].to(device),
                attention_mask=sample["attention_mask"].to(device),
            )
            logits = outputs.logits[:, -1, :]  # [1, V]
            top_indices = logits.topk(K, dim=-1).indices[0]
            token_counter.update(top_indices.tolist())
    return sorted(token_id for token_id, _ in token_counter.most_common(K))
```

**Проверки в теле (важно, потому что paper C.7 их не делает):**
- `len(calibration_samples) > 0` — иначе пустой token_set.
- `K <= model.config.vocab_size` — иначе `topk` кинет.

**Если существует** — правка = 0 строк; в `__init__.py` добавить в `__all__` (см. §5). Зафиксировать в CHANGELOG ТЗ «проверено, отсутствует/присутствует».

### 4.3. Selfcheck (если добавляли)

1. **Determinism:** два вызова на тех же `calibration_samples` → идентичный `token_set`.
2. **Sorted:** `assert token_set == sorted(token_set)`.
3. **Cardinality:** `len(token_set) <= K`.
4. **Reproduces E-27b:** если в репо есть сохранённый `token_set` из E-27b, `assert set(token_set_e27b) == set(build_global_token_set(model, same_samples, K))`. Если не совпадает — это важный сигнал (модель обновилась, K изменился и т.д.); не «фиксить», а задокументировать.

---

## 5. `prism/__init__.py`

### 5.1. Цель правки

Сделать `prism` единой точкой входа для `experiments/`, чтобы экспериментальные скрипты не импортировали из подмодулей напрямую. Это упрощает рефакторинг.

### 5.2. Содержимое

```python
"""PRISM: Jacobian-based analysis of transformer models.

Public API. Import from ``prism`` rather than submodules when possible.
Backward-compatible: ``from prism.jacobian import ...`` continues to work.
"""

from .bridge import jacobian_gap
from .esg import (
    compute_esg,
    compute_esg_per_sample,
    cross_positional_gramian,  # new in v6.0
)
from .jacobian import (
    compute_jacobian_for_tokens,
    compute_hidden_jacobian,
    softmax_jacobian,
    uniform_centered,
)
from .calibration import (
    load_calibration_data,
    extract_embeddings,
    uniform_layer_mapping,
    get_layer_types,
    build_global_token_set,  # new or existing
)
from .diagnostics import (
    run_layer_diagnostics,
    run_identity_test,
    run_heatmap,
)
from .layer_selection import select_pairs, classify_pair
from .cache_io import find_targets_dir

__all__ = [
    "jacobian_gap",
    "compute_esg",
    "compute_esg_per_sample",
    "cross_positional_gramian",
    "compute_jacobian_for_tokens",
    "compute_hidden_jacobian",
    "softmax_jacobian",
    "uniform_centered",
    "load_calibration_data",
    "extract_embeddings",
    "uniform_layer_mapping",
    "get_layer_types",
    "build_global_token_set",
    "run_layer_diagnostics",
    "run_identity_test",
    "run_heatmap",
    "select_pairs",
    "classify_pair",
    "find_targets_dir",
]
```

**Замечания:**

- Список экспортов отражает **public**-функции существующих модулей; приватные (`_validate`, `_inverse_gramian`, `_spectral_decomposition`, `_save_results`, `_plot_heatmap`, `_nearest_preimage`, `_DOMAIN_DATASETS`) — **не** экспортируются.
- Если в `calibration.py` / `diagnostics.py` / `layer_selection.py` окажутся другие public-функции (например, `run_heatmap` может быть private), список скорректировать по факту. **Правило:** экспорт = то, что вызывается из `experiments/` или из публичных docstring.
- Новые модули (`holonomy`, `spectral`, `interventions`, `noise`, `permutation`, `topology`, `cache`, `pipeline`) **не** импортируются здесь, пока не будут реализованы — иначе `ImportError` при импорте `prism`. Их экспорты — в ТЗ на новые модули.

### 5.3. Selfcheck

1. `python -c "import prism; print(prism.__all__)"` — без ошибок.
2. `python -c "from prism import *"` — все имена из `__all__` доступны.
3. `python -c "from prism.jacobian import compute_jacobian_for_tokens"` — работает (backward compat).
4. `python -c "from prism import compute_esg, cross_positional_gramian"` — работает.

---

## 6. Обратная совместимость и ограничения

### 6.1. Что НЕ меняется

- `bridge.py`, `cache_io.py`, `diagnostics.py`, `layer_selection.py` — **ноль изменений**.
- `esg.py`: `compute_esg`, `compute_esg_per_sample`, `_build_gramian`, `_spectral_decomposition` — **ноль изменений**; добавлена одна новая функция.
- `jacobian.py`: `_deep_clone_cache`, `_clone_gdn_states_on_entry`, `_run_decoder_pass`, `_compute_jacobian_rows`, `softmax_jacobian`, `uniform_centered`, `_last_layer_jacobian` — **ноль изменений**.
- `calibration.py`: существующие функции — **ноль изменений**; одна новая (условно).

### 6.2. Что может сломаться (и как проверить)

| Сценарий | Что проверить | Ожидание |
|---|---|---|
| Старый вызов `compute_jacobian_for_tokens(model, l, ids, mask, ts)` | `h_override` не передан → `None` | Побитово идентично v5.0 |
| Старый вызов `compute_jacobian_for_tokens(model, l, ids, mask, ts, prefix_cache=cache)` | `prefix_cache` передан, `h_override` не передан | Побитово идентично v5.0 |
| Старый вызов `compute_hidden_jacobian(..., create_graph=True)` | `h` не передан → `None` | Побитово идентично v5.0 (forward, как раньше) |
| Старый вызов `compute_hidden_jacobian(..., create_graph=True, h=h_precomputed)` | `h` передан | **Поведение меняется:** теперь h respected, а не перезаписывается. Это **баг-фикс**, но если кто-то полагался на старое (ошибочное) поведение — заметит. В CHANGELOG ТЗ явно указать. |
| `from prism.esg import compute_esg` | Импорт работает | Без изменений |
| `from prism import compute_esg` | Новый импорт | Работает (если `__init__.py` обновлён) |
| `from prism import cross_positional_gramian` | Новый экспорт | Работает |

### 6.3. Порядок имплементации

1. `esg.py` — `cross_positional_gramian` (наименьший риск; 10 строк + selfcheck).
2. `jacobian.py` — `h_override` в `compute_jacobian_for_tokens` (§2.2).
3. `jacobian.py` — respect `h` в `create_graph=True` (§2.3).
4. `calibration.py` — `build_global_token_set` (проверить наличие; добавить при отсутствии).
5. `__init__.py` — экспорты.
6. Прогнать все существующие эксперименты (`experiments/exp1*.py` … `exp19*.py`) на smoke-тесте (N=1, K=4) — убедиться, что нет регрессий.

**Критерий готовности:**

- Все selfcheck из §2.5, §3.4, §4.3, §5.3 проходят.
- `git diff` по `bridge.py`, `cache_io.py`, `diagnostics.py`, `layer_selection.py` — пустой.
- `git diff` по `esg.py`: только добавленная функция + docstring-абзац.
- `git diff` по `jacobian.py`: `h_override` + respect `h` + docstring-правки.
- Существующие тесты (если есть `prism/tests/`) — зелёные.

### 6.4. Что НЕ входит в это ТЗ

- Новые модули (`holonomy.py`, `interventions.py`, `spectral.py`, `noise.py`, `permutation.py`, `topology.py`, `cache.py`, `pipeline.py`) — отдельное ТЗ.
- Изменения в `paper.md` — отдельное ТЗ (уже согласовано).
- `experiments/exp_T0.py` и т.д. — после новых модулей.
- Переименование `h` → `h_override` в `compute_hidden_jacobian` — отложено, обратная несовместимость.

---

## 7. Сводный чек-лист ревьюера

- [ ] `compute_jacobian_for_tokens` с `h_override=None` побитово идентичен v5.0 (тест: `git stash`, прогнать smoke exp1).
- [ ] `compute_jacobian_for_tokens` с `h_override=h_from_forward` даёт тот же J, что без override (selfcheck §2.5.1).
- [ ] `h_override` с неверным device/dtype/shape → `ValueError` с понятным сообщением.
- [ ] `compute_hidden_jacobian(create_graph=True)` с `h=None` побитово идентичен v5.0.
- [ ] `compute_hidden_jacobian(create_graph=True)` с `h=leaf` даёт J с grad_fn, градиенты текут к h, не к модели (selfcheck §2.5.7).
- [ ] `cross_positional_gramian(J, J)` спектрально равен `compute_esg([J])` (selfcheck §3.4.1).
- [ ] `Δ_U(t, t').T == Δ_U(t', t)` (selfcheck §3.4.2).
- [ ] `build_global_token_set` либо уже есть, либо добавлен дословно по paper C.7; selfcheck §4.3.
- [ ] `__init__.py` не падает при `import prism`; `__all__` соответствует public-функциям.
- [ ] `git diff` по «неизменяемым» файлам — пустой.
- [ ] CHANGELOG: явно указано, что `compute_hidden_jacobian(create_graph=True, h=h)` меняет поведение (баг-фикс).

---

*ТЗ на существующие файлы. Следующий шаг — ТЗ на новые модули (`prism/holonomy.py` и т.д.) с учётом того, что примитивы из `bridge.py`, `esg.py`, `cache_io.py` уже переиспользуются, а не дублируются.*