# Проверка исполнения ТЗ `new_functions_2.md` (часть 2: `holonomy.py` + `interventions.py`)

## Общая оценка

Реализация **в основном корректна** и соответствует духу ТЗ; все публичные API, проверки ошибок, `try/finally`-восстановление, gauge-инвариантность, групповые орбиты и пост-чеки на месте. Однако есть **один существенный баг** в `interventions.py` (нормализация `p_t` в `apply_edit_and_cache`) и несколько мелких расхождений/недоделок. Ниже — по пунктам.

---

## Что сделано правильно (сверка с ТЗ)

### `holonomy.py`
- ✅ Все 4 публичные функции (`procrustes_align`, `holonomy_accumulate`, `holonomy_along_path`, `holonomy_along_2d_loop`) реализованы с контрактами из §3.2–3.3.
- ✅ `procrustes_align`: `M = U2^T U1`, SVD в `float64`, `R = Q P^T`, `residual_sq = clamp(2r − 2Σs, 0)`, каст в dtype `U1`, проверка ортонормальности (`1e-4` / `1e-6`), обработка вырожденного SVD задокументирована.
- ✅ `holonomy_accumulate`: `R_total = R_{0,1} @ R_{1,2} @ ...` (левое умножение в порядке цикла), `float64`-аккумуляция, `kappa` как Python-float, `closed_loop=False` = открытый путь; **групповое свойство орбит** («κ = 0 для замкнутой орбиты») задокументировано и соответствует TZ-addendum (исправление ошибочного selfcheck §3.6.7/§3.6.10).
- ✅ `holonomy_along_path`: `len(thetas) ≥ 2`, `path_fn → (U, V)`, `TypeError` на не-кортеж, dict-выход с `kappa_U/V`, `per_step_U/V`, `thetas`.
- ✅ `holonomy_along_2d_loop`: `θ = 2πi/n_theta` без `2π`-конца, `epsilon > 0`, `n_theta ≥ 4`, проверка `b1, b2` (`‖·‖ ≈ 1`, `|b1·b2| < 1e-4`), `h_perturbed` для диагностики, `TypeError` на не-кортеж.
- ✅ Никакой модельной логики, только тензорная математика; `__all__`; приватные `_validate_orthonormal`, `_check_loop_basis`.
- ✅ Selfcheck §3.6 покрывается реализацией (identity, rotation, sign-invariance, residual formula, non-orthonormal → `ValueError`, тривиальный/константный путь, closed vs open, gauge-invariance). TZ-ошибка в таблице §3.3 про «R инвариантна к смене знака» — реализация следует корректному selfcheck (R ≠ I, res ≈ 0), а не ошибочной таблице.

### `interventions.py`
- ✅ `EditSpec` — frozen dataclass, все поля (`name`, `delta_W: [K, D_S] | None`, `token_set: tuple`, `target_layer`, `norm_frobenius`, `projection_onto_top_r`, `seed`, `meta`), `meta` документирован как immutable-by-convention.
- ✅ `make_base`: `delta_W=None`, нули, no-op.
- ✅ `make_readout_pull`: loss `(1 − cos).mean() + kl_leash_weight · KL`; `p_base` один раз под `no_grad`; `J_S_p` через `compute_hidden_jacobian_rows(..., create_graph=True)`; Adam с маской `~ts_mask` на `weight.grad` (обновляются только `token_set`-строки); `W0 = weight[ts].detach().clone()`, восстановление через `weight.data.index_copy_` (обход leaf-grad-запрета на in-place); `requires_grad` сохраняется/восстанавливается; `torch.manual_seed(seed)` без `use_deterministic_algorithms`; проверки `target_J_p.shape`, `n_steps ≥ 1`, `lr > 0`; **опциональный `top_r_basis`** (TZ-альтернатива принята, `projection_onto_top_r` вычисляется сразу, если задан).
- ✅ `make_random_ortho`: `G ~ randn(K, D_S)`, `G_proj = G − P G` (`P = U U^T`, `[K,K]`), масштабирование к `‖ref‖`, пост-чеки (`‖Δ‖ ≈ ‖ref‖` 1e-6, `‖P Δ‖/‖Δ‖ < 1e-3`), `RuntimeError` при вырожденном `G_proj`.
- ✅ `make_random_matched`: `ref_proj + G_residual · (‖ref_residual‖/‖G_residual‖)`; пост-чеки (проекция сохранена `< 1e-4·‖ref‖`, норма `< 1e-5·‖ref‖`), `RuntimeError` если `ref` целиком в top-r.
- ✅ `make_tail_only`: `ref − P ref`, **без** ре-нормализации (TZ-семантика «что осталось»), `RuntimeError` если хвоста нет.
- ✅ `apply_edit`: context-manager, `index_add_` под `no_grad`, восстановление в `finally`, `requires_grad` восстановлен, `base` — no-op, `target_layer != "lm_head"` → `NotImplementedError`, вложенность → `RuntimeError` через `_APPLY_EDIT_ACTIVE`, `ValueError` на несовпадение `delta_W.shape`.
- ✅ `apply_edit_and_cache`: layout `{edit}/{L:02d}/doc{NNNN}_{h,p}.pt + meta.json`; `h_t` в `float32`; `p_t` в `float32`; J **не** кэшируется; без `_clone_gdn_states_on_entry` (осознанно, `no_grad`); атомарная запись (`tmp + os.replace`); `overwrite=False` с проверкой meta-консистентности; `RuntimeError` на read-only FS с явным сообщением; `ValueError` на пустые `samples` / невалидный `layer`; `h_states[L][0].float()` — `[S, D_S]`.
- ✅ Импорт-гигиена: `from .jacobian import compute_hidden_jacobian_rows` — единственная зависимость от `jacobian.py`; `bridge.jacobian_gap`, `compute_jacobian_for_tokens`, `cache_io.find_targets_dir`, `_clone_gdn_states_on_entry` — отсутствуют, как и требует TZ.

---

## Найденные проблемы

### 🔴 Существенное: `apply_edit_and_cache` не нормализует `p_t` по `token_set`

ТЗ §4.3 (таблица `apply_edit_and_cache`):
> `p_t` = `softmax(model.lm_head(h_t)[:, :, token_set])`

Это означает: (1) взять логиты, (2) срезать по `token_set` (`[S, K]`), (3) softmax **по K**. Тогда `p_t.sum(dim=-1) ≡ 1` — это ровно `p_S` из paper.md §4:
> `p_S(k) = p(k) / Σ_{j∈S} p(j)`

Реализация делает иначе:
```python
p_full = F.softmax(out.logits.float(), dim=-1)  # softmax по V
p_t = p_full[0][:, ts]                          # срез, БЕЗ ре-нормализации
```
Здесь `p_t.sum(dim=-1) = P(token_set) < 1` (на sentinel-позициях E-34 — до 0.5). Это **не** `p_S`.

**Почему это важно.** `P(p) = diag(p) − pp^T` убивает uniform-направление только если `p` — распределение вероятностей (`Σ p = 1`): `P(p)·1 = p·(1 − Σp)`. При `Σp ≠ 1` downstream gauge-free конструкция (`J^(p) = P(p)J`, paper.md §8 `[L-gauge]`) даёт неверный объект. TZ прямо пишет: *«`p` in float32 (softmax precision for gauge-free `P(p)`»* — т.е. нормализация подразумевается.

**Дополнительный сигнал несогласованности:** `make_readout_pull` в этом же файле делает **правильно**:
```python
p_base = F.softmax(out.logits[0, -1, ts].float(), dim=-1)  # softmax по K
```
То есть два места в одном модуле считают `p` по-разному.

**Минимальный фикс:**
```python
p_t = F.softmax(out.logits[0][:, ts].float(), dim=-1)  # [S, K], renormalized
```
(или `p_t = p_full[0][:, ts]; p_t = p_t / p_t.sum(-1, keepdim=True)`).

---

### 🟡 Средние: мелкие расхождения с ТЗ

1. **`meta` в `apply_edit_and_cache` не содержит `model_dtype`.** ТЗ §4.3 явно перечисляет `"model_dtype": str(model_dtype)` среди полей meta. В реализации `meta` = `{edit, layer, doc, S, D_S, K, torch, transformers, seed}` — поля `model_dtype` нет. Это влияет на downstream-консистентность (например, при проверке «этот кэш снят с fp16-модели или fp32?»).

2. **`model.eval()` не восстанавливается.** И `make_readout_pull`, и `apply_edit_and_cache` вызывают `model.eval()` без сохранения/восстановления `model.training`. TZ (docstring `make_readout_pull`): *«The model is left exactly as found (weights, requires_grad)»* — «exactly as found» естественно включает training mode. Если caller передал модель в train-режиме, после вызова она останется в eval. Фикс:
```python
was_training = model.training
model.eval()
try: ...
finally: model.train(was_training)
```

3. **`_APPLY_EDIT_ACTIVE` в `make_readout_pull` ставится до setup-кода.** Флаг выставляется в `True`, затем идут `torch.manual_seed`, `model.eval()`, `weight[ts].detach().clone()`, `weight.requires_grad_(True)`, `torch.zeros(...)`, `ts_mask[ts] = True` — **вне** `try`. Если любой из этих шагов бросит (например, device-mismatch в `ts_mask[ts] = True`), флаг останется `True`, и все последующие `apply_edit` / `make_readout_pull` будут падать с «nested model edit in progress». Фикс: перенести setup внутрь `try` (или обернуть в отдельный `try/finally`).

4. **`_check_loop_basis` — допуск на норму `1e-3` вместо `1e-4`.** ТЗ §3.3: *«`‖b1 − 1‖ < 1e-4`, `‖b2‖ ≈ 1`»* (имеется в виду `|‖b1‖ − 1| < 1e-4`). Реализация: `abs(n1 - 1.0) > 1e-3`. На порядок мягче. Не критично, но формально расхождение.

5. **Нет отдельной `_validate_edit_spec(edit, D_S)`.** ТЗ §4.4 описывает приватную `_validate_edit_spec`, проверяющую `delta_W.shape` и `target_layer`. Реализация делает проверки inline в `apply_edit` и `make_readout_pull`. Функционально эквивалентно, но структура не совпадает с ТЗ (мелочь для код-ревью, не для корректности).

6. **Docstring `make_readout_pull` вводит в заблуждение.** В docstring: *«`p_edit = softmax(lm_head(h_t)[:, :, token_set])` over the last position's logits, **averaged over positions**»*. Код использует только последнюю позицию (`out.logits[0, -1, ts]`), никакого усреднения по позициям нет. Либо docstring кривой, либо подразумевалось усреднение — но в paper.md E-16a протокол именно last-position. Стоит переформулировать docstring (например, «at the last position»).

7. **`_tensor_fingerprint` — не «16-hex».** ТЗ §4.3: *«Deterministic 16-hex fingerprint (sum + abs-max)»*. Реализация возвращает десятичную строку `f"{sum:.9f}:{absmax:.9f}"`. Детерминизм есть, формат — нет. Для `meta`-идентификации работает; если downstream ожидает именно hex — сломается. Мелочь.

8. **`apply_edit_and_cache` возвращает `S` последнего сэмпла.** `S = sample["input_ids"].shape[1]` объявлен внутри цикла `for doc_idx, sample in enumerate(samples)` и попадает в return-dict. Если сэмплы разной длины (что ТЗ формально не запрещает — «samples: list[dict], [{input_ids: [1,S], ...}]» подразумевает одинаковый S, но не гарантирует), `S` в возврате будет от последнего документа. Стоит либо ассертить одинаковость `S` по всем сэмплам, либо возвращать `S` как список.

9. **`overwrite=False` и частичный кэш.** Реализация пропускает, только если **все три** файла (`h`, `p`, `meta`) существуют; иначе перезаписывает **все три** (включая уже существующий `h`) без проверки его `meta`. TZ §4.3: *«existing files are kept, but their meta is checked for consistency»* — гранулярность по каждому файлу. На практике частичные кэши редки, но при их появлении поведение отличается от описанного.

10. **TZ-таблица §3.3 `procrustes_align`: «R инвариантна к одновременной смене знака столбцов `U1, U2`» — математически неверно.** Правильно: `R' = S R S ≠ R` при `S = diag(±1)`; инвариантен `residual_sq`. Реализация следует корректному selfcheck §3.6.3 (R ≠ I, res ≈ 0), а не ошибочной таблице. Это ошибка **ТЗ**, не реализации — но её стоит зафиксировать в документации/комментарии (в коде уже отражено через selfcheck-поведение).

---

## Сводная таблица

| # | Модуль | Серьёзность | Проблема | Статус |
|---|---|---|---|---|
| 1 | `interventions.py` | 🔴 критично | `p_t` в `apply_edit_and_cache` не ре-нормализуется по `token_set` (отклонение от paper.md §4 `p_S` и TZ-формулы `softmax(…[:, token_set])`); несовместимо с gauge-free `P(p)`, ломает downstream `J^(p)` | требует фикса |
| 2 | `interventions.py` | 🟡 средне | `meta` без `model_dtype` (TZ §4.3 перечисляет поле) | требует фикса |
| 3 | `interventions.py` | 🟡 средне | `model.eval()` не восстанавливается (`make_readout_pull`, `apply_edit_and_cache`) — нарушает «exactly as found» | требует фикса |
| 4 | `interventions.py` | 🟡 средне | `_APPLY_EDIT_ACTIVE = True` выставлен до setup-кода в `make_readout_pull` — при исключении в setup флаг залипает | требует фикса |
| 5 | `holonomy.py` | 🟢 мелочь | `_check_loop_basis`: допуск нормы `1e-3` вместо `1e-4` (TZ §3.3) | допустимо / поправить |
| 6 | `interventions.py` | 🟢 мелочь | нет отдельной `_validate_edit_spec` (inline-проверки) | допустимо |
| 7 | `interventions.py` | 🟢 мелочь | docstring `make_readout_pull` говорит «averaged over positions», код — только last-position | поправить docstring |
| 8 | `interventions.py` | 🟢 мелочь | `_tensor_fingerprint` не «16-hex» (десятичный формат) | поправить / задокументировать |
| 9 | `interventions.py` | 🟢 мелочь | `S` в return-dict — от последнего сэмпла (при разной длине) | ассерт / list |
| 10 | `interventions.py` | 🟢 мелочь | `overwrite=False` при частичном кэше перезаписывает существующие файлы без проверки meta | гранулярная проверка |
| 11 | ТЗ (не код) | — | TZ-таблица §3.3 неверна про «R инвариантна к смене знака»; реализация корректна (следует selfcheck §3.6.3) | зафиксировать в комментарии |

---

## Итоговый вердикт

**`holonomy.py` — практически полностью соответствует ТЗ** (все контракты, ошибки, gauge-инвариантность, групповые орбиты, диагностика; единственное расхождение — допуск `_check_loop_basis` `1e-3` vs `1e-4`, не критично).

**`interventions.py` — в основном соответствует, но есть один существенный баг (`p_t` без ре-нормализации в `apply_edit_and_cache`), который сломает downstream gauge-free реконструкцию `J^(p)`**, плюс несколько мелких недоделок (`model_dtype` в meta, восстановление `model.training`, залипание `_APPLY_EDIT_ACTIVE` при ошибке setup). Баг #1 стоит исправить до запуска `exp_T0.py`: он тихий (не падает, но даёт неверный `p_t` на диске) и обнаружится только при сверке downstream-метрик с paper.md.

Остальные расхождения — косметические/структурные и не влияют на корректность научных результатов.