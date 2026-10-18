# docs/changed_functions.md — исполнение (2026-07-18)

Минимальные правки `prism/` модулей по ТЗ `docs/changed_functions.md`.
Аудит: `docs/changed_review.md`; по нему внесены 3 правки (см. «Аудит»).

## Что сделано

### 1. `prism/esg.py` — новая `cross_positional_gramian(J_t, J_tp)`
- `Δ_U(t, t') = J_t @ J_tp.T` — без нормировки `/d` (так требует ТЗ §3.2;
  нормировка не меняет относительный спектр, но спектр, а selfcheck §3.4.1 на `eigvalsh`
  проходит только без `/d`).
- Вход: `J_t, J_tp: [K, D]` (K = |token_set|, D = hidden_size). Выход: `[K, K]`
  (для всех слоёв E-27b/E-32r: `[128, 128]`, т.к. T=128).
- Не зависит от `esg_basis` (локальная ковариация, не проекция) → корректно для
  cross-position при сжатии.
- Позиции = строки J в порядке `compute_hidden_jacobian` (positions 0..S-1) →
  совпадает с `C[r]`/`d3[r]` в E-32r (единая конвенция).
- +34 строки (1 docstring-абзац + функция), без правки существующих.

### 2. `prism/jacobian.py` — `h_override` + фикс create_graph
- `compute_hidden_jacobian(..., h: Tensor|None)` — уже имела `h`; добавлены
  **валидация** (shape `[D]/[1,D]/[1,1,D]` → иначе `ValueError "reducible"`,
  D==hidden_size, device==input_ids.device, **без** auto-move) и use-in-place
  для готового leaf `h` (`.grad` доходит до тензора вызывающего; clone+detach
  только для отключённого `h`).
- **Фикс бага create_graph**: `h` больше НЕ перезаписывается безусловно
  `h = _extract_hidden_state(...)` — при `h is not None` forward вниз
  `layer_idx` пропускать; leaf-ветка (`h.grad_fn is None and h.requires_grad`)
  использует `h` как есть.
- `compute_jacobian_for_tokens(..., h_override=None)` — новый kwarg; с `None`
  бит-в-бит как pre-v6.0 (тест identity).
- `_normalize_h_override`: `h.detach().clone().to(model_dtype)` — clone до cast
  (1 аллокация в fp32-типичном случае; audit №3).
- `build_global_token_set` — **уже была в v5** (0 строк, только экспорт).

### 3. `prism/__init__.py` — реальный `__all__`
Было: только docstring (import `from prism import compute_esg` падал).
Теперь: 22 имени (esg×3, jacobian×7, calibration×5, diagnostics×3,
layer_selection×2, bridge×1, cache_io×1).
**Зависимость:** `prism.calibration` импортирует `datasets` на уровне модуля,
поэтому `import prism` **требует установленный `datasets`** (pre-v6.0 __init__
был без этой зависимости). Делать try/except / lazy **не** стали — зафиксировано
комментарием в `__init__.py`: `load_calibration_data` — часть публичного API,
и тишое отсутствие даст запутанный NameError глубоко в долгом эксперименте.
Если `datasets` нет — `import prism` падает с ясным `ModuleNotFoundError`.

### 4. `prism/calibration.py` — 0 строк (функция уже есть, контракт с ТЗ совпадает)

## Отклонения от ТЗ (честно)

1. **Тесты в `tests/`, не `prism/tests/`** — репо уже использует `tests/`
   (8 файлов, common fixtures в `tests/helpers/tiny_model.py`); `prism/tests/`
   не существует. Следовал конвенции репо.
2. **§2.5.7: «граф НЕ связан с параметрами» — недостижимо и ложно**. J_norm =
   d(h_norm)/d(h_l) считается decode-проходом слоёв `layer_idx+1..N-1` + norm,
   поэтому J_norm по определению зависит от весов декодера — они ПОЛУЧАЮТ грады.
   Гарантия leaf-ветки (и тест): (a) `h_leaf.grad` заполняется; (b) параметры
   слоёв `<= layer_idx` (embeddings, 0..L) градов НЕ получают. Test-assertion
   `not any(p.grad is not None)` заменён на корректный с комментарием.
3. **§2.5.7: `assert not h_override.requires_grad`** противоречит «clone, не
   мутировать»: в-плейс-использование готового leaf сохраняет флаг. Контракт:
   значения и флаг тензора вызывающего не мутируются, J — без графа при
   `create_graph=False` (тест `test_detach_safety` с комментарием).
4. **`_normalize_h_override`: lookup D** — у wrapper-модели `hidden_size`
   только в `config.text_config`; добавлен fallback (getattr text_config, cfg).
5. **`import prism` теперь требует `datasets`** — см. §3 выше; фиксировано
   комментарием, не try/except (решение аудита №1, вариант (б)+комментарий).

## Selfcheck (все пройдены)

- identity: `h=None` и `h=extracted` — J совпадает (atol 1e-5)
- shape normalization [1,1,D]/[1,D]/[D]
- device mismatch / D mismatch / non-reducible — ValueError (без auto-move)
- detach safety: caller-тензор не мутируется, J без графа
- create_graph: h=None — грады к параметрам; h=leaf — `h_leaf.grad` заполнен,
  параметры `<= layer_idx` исключены (декодер включён — см. отклонение #2)
- create_graph bug fix: connected h уважается, не перезаписывается forward'ом
- `cross_positional_gramian`: shape, симметрия, float32, спектральное
  равенство с `compute_esg([J])` (N=1, без /d)
- `build_global_token_set`: union, порядок, empty, duplicates
- `import prism` + `__all__` (22 имени), `from prism import build_global_token_set`

## Тесты

- `tests/test_cross_positional_gramian.py` — 4 passed
- `tests/test_jacobian_h_override.py` — 9 passed
- **Полный прогон: 46 passed** (регрессия отсутствует)

## Аудит (docs/changed_review.md) — resolution

| № | Пункт аудита | Решение |
|---|---|---|
| 1 | нет try/except для calibration (регрессия) | **Комментарий** в `__init__.py` (не lazy): `import prism` требует `datasets`; зафиксировано в отчёте §3 и отклонении #5 |
| 2a | `/d` в отчёте против `J_t@J_tp.T` в коде | `/d` — опечатка в отчёте; код = ТЗ §3.2 (без /d). Отчёт §1 исправлен |
| 2b | `[192,192]` для L22 | опечатка; T=128 для всех слоёв (meta exp32r3 + E-32-tail). Отчёт §1 исправлен на `[128,128]` |
| 3 | порядок `.to()`/`.clone()` | `h.detach().clone().to(model_dtype)` — 1 аллокация в fp32 |

## Файлы

- изменены: `prism/esg.py`, `prism/jacobian.py`, `prism/__init__.py`
- новые: `tests/test_cross_positional_gramian.py`, `tests/test_jacobian_h_override.py`
- `prism/calibration.py` — не тронут (0 строк по ТЗ)
