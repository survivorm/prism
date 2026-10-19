# Аудит исполнения `prism/` по ТЗ `docs/changed_functions.md`

Проверил построчно. В целом исполнение аккуратное, 3 из 4 файлов соответствуют ТЗ; документированные отклонения — корректные баг-фиксы самого ТЗ. Но нашёл **два реальных несоответствия** (одно из них — регрессия) и **один неясный пункт** в отчёте.

---

## ✅ Что сделано верно (сверка с ТЗ)

### `prism/jacobian.py` — §2 ТЗ

- **`h_override` в `compute_jacobian_for_tokens`** (§2.2): сигнатура, семантика, ветвление `if h_override is None / else` — всё совпадает. `prefix_cache` строится всегда (если не передан), `_extract_hidden_state` пропускается при `h_override` — ✅.
- **Валидация** (§2.2): `_normalize_h_override` проверяет shape → D → device → dtype → detach+clone+requires_grad, в правильном порядке, без auto-move — ✅.
- **Respect `h` в `create_graph=True`** (§2.3): `if h is None: <forward>; h = h.reshape(...)` — корректный баг-фикс. `h.grad_fn is None and h.requires_grad` → in-place (не `detach().requires_grad_`), это правильно для получения `.grad` у тензора вызывающего — ✅.
- **Docstring «Interaction with compute_hidden_jacobian»** (§2.4) — ✅.
- **`_normalize_h_override` fallback `D_expected`** через `text_config` для wrapper-моделей — корректное расширение (не было в ТЗ, но необходимо) — ✅.

### `prism/esg.py` — §3 ТЗ

- **`cross_positional_gramian(J_t, J_tp)`** — сигнатура, `_validate([J_t, J_tp])`, `J_t.float() @ J_tp.float().T` — совпадает с ТЗ §3.2 **байт-в-байт** — ✅.
- **Docstring-абзац в module docstring** — семантически совпадает с ТЗ §3.3 — ✅.
- Существующие `compute_esg`, `compute_esg_per_sample`, `_build_gramian`, `_spectral_decomposition` — не тронуты — ✅.

### `prism/calibration.py` — §4 ТЗ

- `build_global_token_set` уже была; сравнение с ТЗ §4.2 — семантически эквивалентна — ✅.

### `prism/__init__.py` — §5 ТЗ

- Экспортирует все имена из ТЗ §5.2 + `compute_hidden_jacobian_rows`, `bridge_rows_backward`, `h_norm_at_perturbed_h` (это существующие public-функции `jacobian.py`, ТЗ §5.2 разрешал «экспорт = то, что вызывается из experiments/») — ✅.
- Новые модули (`holonomy`, `interventions`, и т.д.) не импортируются — ✅.

### Документированные отклонения от ТЗ — все корректны

1. **`tests/` вместо `prism/tests/`** — репо уже использует `tests/`; следование конвенции репо правильнее буквы ТЗ — ✅.
2. **§2.5.7 «граф не связан с параметрами» — недостижимо**: J_norm = d(h_norm)/d(h_l) считается decode-проходом через слои `layer_idx+1..N-1`, их параметры получают градиенты. Моё ТЗ было **неверно**; имплементатор корректно исправил тест — ✅.
3. **§2.5.5 «`assert not h_override.requires_grad`»** — тоже **неверно** в моём ТЗ: клон не мутирует флаг вызывающего. Корректный контракт — «значения и флаг не мутируются, J без графа при `create_graph=False`» — ✅.

---

## ❌ Несоответствие №1 (регрессия): `__init__.py` — нет try/except для `calibration`

**Отчёт §3 утверждает:**
> `calibration` под try/except ImportError (опция `datasets` не всегда стоит).

**Код `__init__.py`:**
```python
from .calibration import (
    load_calibration_data,
    extract_embeddings,
    uniform_layer_mapping,
    get_layer_types,
    build_global_token_set,
)
```

`try/except` **отсутствует**. При этом `calibration.py` имеет на верхнем уровне:
```python
from datasets import load_dataset
```

**Следствие:** `import prism` **упадёт**, если `datasets` не установлен. До v6.0 `__init__.py` был пустой (только docstring), и `import prism` работал тривиально. Сейчас — регрессия: `from prism import compute_esg` требует `datasets`.

**Что делать:** либо
(а) добавить `try/except ImportError` вокруг `from .calibration import ...` (и тогда `calibration`-экспорты отсутствуют в `__all__`, если `datasets` нет — `__all__` нужно делать динамическим или через `hasattr`), либо
(б) убрать `from .calibration import ...` из `__init__.py` и оставить эти функции импортируемыми только из `prism.calibration`, либо
(в) убрать `from datasets import load_dataset` с верхнего уровня `calibration.py`, сделав ленивый импорт внутри `load_calibration_data` / `_load_dataset_for_domain`.

Вариант (в) — предпочтительный: `prism.calibration` остаётся импортируемым без `datasets`; ошибка возникает только при фактическом вызове `load_calibration_data` — что и есть правильное поведение.

**Если решено не фиксить — отчёт §3 нужно исправить**, убрать заявление про try/except, и зафиксировать в CHANGELOG: «`import prism` теперь требует `datasets`».

---

## ❌ Несоответствие №2: `esg.py` — отчёт §1 описывает не тот код

**Отчёт §1:**
> ### 1. `prism/esg.py` — новая `cross_positional_gramian(J)`
> - `G_{ab} = <J_a, J_b> / d`; для L23: [128,128], L22: [192,192] (без батчинга).

**Код:**
```python
def cross_positional_gramian(
    J_t: torch.Tensor,
    J_tp: torch.Tensor,
) -> torch.Tensor:
    """Cross-positional Gramian Δ_U(t, t') = J_t @ J_tp.T.
    ...
    """
    _validate([J_t, J_tp])
    return J_t.float() @ J_tp.float().T
```

**Расхождения:**
1. **Сигнатура:** отчёт описывает `cross_positional_gramian(J)` (один аргумент), код — `(J_t, J_tp)` (два). ТЗ §3.2 требовал два аргумента — код соответствует ТЗ, отчёт — нет.
2. **Нормировка:** отчёт описывает `G_{ab} = <J_a, J_b> / d`, код — `J_t @ J_tp.T` без `/d`. ТЗ §3.2 не требовал `/d` — код соответствует ТЗ, отчёт — нет.
3. **Форма выхода:** отчёт говорит `[128,128]` для L23 и `[192,192]` для L22 — если `K=128` для всех слоёв E-27b (как заявлено в paper §E-27b), то L22 тоже должен давать `[128,128]`, не `[192,192]`. Неясно, откуда `[192,192]`.

**Что делать (приоритет — выяснить и зафиксировать):**

- **Про `/d`:** либо (а) добавить `/d` в код (тогда selfcheck §3.4.1 «равенство с `compute_esg`» перестанет проходить без поправки: `eigvalsh(J J^T) == eigvalsh(compute_esg([J]))`, но `eigvalsh(J J^T / d) != eigvalsh(compute_esg([J]))`); либо (б) убрать `/d` из отчёта. Мой ТЗ исходил из (б); отчёт утверждает (а). Нужно явное решение, потому что для `ε_gap` нормировка на `d` не меняет относительный спектр (все λ умножаются на `1/d`), но `eigengap` в абсолютных числах меняется на `1/d` — а именно абсолютные числа сравниваются с `σ_GDN` в T2'. Если решено оставить `/d` в коде — нужно обновить ТЗ §3.2 и selfcheck §3.4.1.
- **Про `[192,192]`:** либо это опечатка (должно быть `[128,128]` для всех слоёв), либо `S` (позиции) различается по слоям в кэше, который использовался для проверки — если да, зафиксировать это в `meta.json`. Если `K` различается — это несовместимо с E-27b и требует проверки, что `token_set` один на все слои.

---

## ⚠️ Неясность №3: `_normalize_h_override` — порядок `.to()` и `.detach().clone()`

Мелочь, но: код делает
```python
h = h.to(model_dtype)              # 1. dtype cast
return h.detach().clone().requires_grad_(True)  # 2. detach + clone
```

Если `h.dtype != model_dtype`, шаг 1 создаёт промежуточный тензор (dtype-cast), шаг 2 — ещё один (clone). Итого 2 новых тензора. Можно
```python
return h.detach().clone().to(model_dtype).requires_grad_(True)
```
чтобы получить 1 новый тензор (clone, потом cast) — но на `dtype=float32` (типичный для CPU-экспериментов) это no-op, потому что `to(same_dtype)` возвращает тот же tensor. Не баг, но лишний аллокат в fp16-сценарии. Приоритет — низкий.

---

## 📋 Сводная таблица

| Пункт | ТЗ | Код | Отчёт | Статус |
|---|---|---|---|---|
| `h_override` в `compute_jacobian_for_tokens` | ✅ | ✅ | ✅ | OK |
| `_normalize_h_override` (shape → D → device → dtype → detach) | ✅ | ✅ | ✅ | OK |
| Respect `h` в `compute_hidden_jacobian(create_graph=True)` | ✅ | ✅ | ✅ | OK (баг-фикс) |
| In-place `h` (leaf с requires_grad) | не специфицировано | ✅ | ✅ | Корректный фикс ТЗ |
| `cross_positional_gramian(J_t, J_tp)` без `/d` | ✅ | ✅ | ❌ (описывает `/d` и один аргумент) | **Разобраться** |
| `cross_positional_gramian` форма L22 `[192,192]` | не специфицировано | возвращает `[K,K]` | `[192,192]` | **Уточнить** |
| `build_global_token_set` | проверка наличия | уже есть | «уже есть» | OK |
| `__init__.py` try/except для `calibration` | не требовалось | ❌ отсутствует | ✅ заявлено | **Регрессия / несоответствие отчёта** |
| `__init__.py` экспорт всех public | ✅ | ✅ (+3 существующих) | ✅ | OK |
| Тесты в `tests/` вместо `prism/tests/` | рекомендовано `prism/tests/` | `tests/` | отклонение зафиксировано | OK (конвенция репо) |
| §2.5.7 «граф не связан с параметрами» | ❌ (моё ТЗ было неверно) | ✅ фикс | отклонение зафиксировано | OK |
| §2.5.5 «not h_override.requires_grad» | ❌ (моё ТЗ было неверно) | ✅ фикс | отклонение зафиксировано | OK |

---

## 🎯 Что нужно сделать до закрытия ТЗ

1. **Решить про `try/except` в `__init__.py`.** Приоритетный вариант — ленивый `from datasets import load_dataset` внутри `calibration.load_calibration_data`, чтобы `prism.calibration` импортировался без `datasets`. Если не делать — исправить отчёт §3 (убрать заявление про try/except) и записать в CHANGELOG: «`import prism` теперь требует `datasets`».
2. **Зафиксировать решение про `/d`.** Или добавить в код (и обновить selfcheck + ТЗ §3.2), или убрать из отчёта §1. Сейчас код и отчёт противоречат друг другу.
3. **Проверить `[192,192]` для L22.** Если это опечатка — исправить отчёт. Если реально `K=192` для L22 — значит `token_set` различается по слоям, что нарушает инвариант E-27b (`K = |S| = 128` для всех слоёв); это серьёзно, нужно понять, откуда.
4. **Опционально:** переупорядочить `_normalize_h_override` в `h.detach().clone().to(model_dtype)` — экономит один аллокат в fp16, не влияет на корректность.

После этих четырёх пунктов реализация соответствует ТЗ и отчёту одновременно. Сейчас код — ТЗ-консистентен, но отчёт местами описывает не тот код, а `__init__.py` имеет реальную регрессию.