# docs/changed_functions.md — исполнение (2026-07-18)

Минимальные правки `prism/` модулей по ТЗ `docs/changed_functions.md`.

## Что сделано

### 1. `prism/esg.py` — новая `cross_positional_gramian(J)`
- `G_{ab} = <J_a, J_b> / d`; для L23: [128,128], L22: [192,192] (без батчинга).
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
- `build_global_token_set` — **уже была в v5** (0 строк, только экспорт).

### 3. `prism/__init__.py` — реальный `__all__`
Было: только docstring (import `from prism import compute_esg` падал).
Теперь: 14 имён (esg×4, jacobian×5, layer_selection×1, calibration×4).
`calibration` под try/except ImportError (опция `datasets` не всегда стоит).

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

## Selfcheck (все пройдены)

- identity: `h=None` и `h=extracted` — J совпадает (atol 1e-5)
- shape normalization [1,1,D]/[1,D]/[D]
- device mismatch / D mismatch / non-reducible — ValueError (без auto-move)
- detach safety: caller-тензор не мутируется, J без графа
- create_graph: h=None — грады к параметрам; h=leaf — `h_leaf.grad` заполнен,
  параметры `<= layer_idx` исключены (декодер включён — см. отклонение #2)
- create_graph bug fix: connected h уважается, не перезаписывается forward'ом
- `cross_positional_gramian`: shape, симметрия, G_ii = per-sample norm²
- `build_global_token_set`: union, порядок, empty, duplicates
- `import prism` + `__all__` (14 имён), `from prism import build_global_token_set`

## Тесты

- `tests/test_cross_positional_gramian.py` — 8 passed
- `tests/test_jacobian_h_override.py` — 9 passed
- **Полный прогон: 46 passed** (регрессия отсутствует)

## Файлы

- изменены: `prism/esg.py`, `prism/jacobian.py`, `prism/__init__.py`
- новые: `tests/test_cross_positional_gramian.py`, `tests/test_jacobian_h_override.py`
- `prism/calibration.py` — не тронут (0 строк по ТЗ)
