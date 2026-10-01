## Остаточные ошибки

### exp27b_dense.py

**1. `fwd` пересобирается на каждом слое — 6× лишних forward'ов**

```python
for l in layers:
    ...
    fwd: dict[tuple, dict] = {}
    for dom in domains:
        for si in range(n_samples):
            if all((dom, si, t) in JR for t in range(S)):
                continue
            fwd[(dom, si)] = _forward_dense(m, pool[dom][si], S, layers)
```

`fwd` локальный внутри цикла по слоям. При холодном старте для каждого из 6 слоёв делается полный forward всех 32 сэмплов → **192 forward'а вместо 32**. При возобновлении из частичного ckpt (слой L готов, слой L+1 нет) — то же самое.

`_forward_dense` возвращает hs для всех `layers` сразу, поэтому forward достаточно сделать **один раз** на сэмпл и раздать его всем слоям. Это не баг (результат корректный), но ~6× потеря времени на GPU. Для 12–16h T4 это критично.

**2. `f["logits"]` — [S, V] float32 на сэмпл в RAM**

Для Qwen3.5 vocab ≈ 150k: `128 × 150000 × 4 = 77 MB` на сэмпл × 32 = **~2.5 GB** в `fwd`. Комментарий «RAM-safe» верен только если V маленький; на реальном Qwen3.5 0.8B (vocab 151936) это упирается в лимит.

`f["logits"]` используется ровно для одного: `torch.softmax(f["logits"][S-1-t])[U128]`. То есть на вход идут только softmax-вероятности на **top-128**, остальное выбрасывается. Можно сохранить сразу `p128_all[S, 128]`:

```python
def _forward_dense(m, s, S, layers):
    ...
    logits = out.logits[0].float()          # [S, V] на GPU
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)   # [S, V] на GPU
        p128 = p[:, U128]                    # [S, 128] на GPU
    return {"p128": p128.cpu(), "hs": hs}
```

Это снижает RAM в ~1000×, заодно убирает повторный softmax по [V] внутри внутреннего цикла `t`.

**3. `DynamicCache(config=m.config)` — версия-зависимо**

Комментарий утверждает, что без `config` получаем `layers=[]` и IndexError. Это поведение конкретной версии transformers. Проверить на целевой версии (или хотя бы ветвиться):

```python
try:
    pc = DynamicCache(config=m.config)
except TypeError:
    pc = DynamicCache()
    pc.layers = [None] * m.config.num_hidden_layers   # версия-специфично
```

Гарантировать это тестом `--s-dense 4` (сразу попадаем в `t = S-1 = 3` ≠ 0) — но нужен именно `t = S-1` (позиция 0), т.е. надо, чтобы во внутреннем цикле был `t == S-1`. При `s_dense=4` это `t=3`, а условие `S - t == 1` → `t == S-1 = 3`. Сработает. ✓

**4. `fwd.pop` внутри двойного цикла**

```python
for dom in domains:
    for si in range(n_samples):
        for t in range(S):
            ...
        fwd.pop((dom, si), None)
```

`pop` вне цикла `t` (по отступу — на уровне `si`). Корректно. Но если `fwd[(dom, si)]` не создан (все `t` уже в JR) — `pop` с default `None` вернёт None, не упадёт. ✓

---

### jacobian.py

**5. `_compute_jacobian_rows` — цикл по K с отдельным `autograd.grad`**

```python
for i in range(len(token_set)):
    grad = torch.autograd.grad(outputs=logits_subset[i], inputs=h, ...)[0]
    J[i] = grad[0, 0, :]
```

Для K=128 это 128 backward'ов. `torch.autograd.grad` поддерживает `is_grads_batched`:

```python
J = torch.autograd.grad(
    outputs=logits_subset,                # [K]
    inputs=h,                             # [1, 1, D]
    grad_outputs=torch.eye(K, device=h.device),   # [K, K]
    retain_graph=False,
    is_grads_batched=True,
)[0]                                      # [K, 1, 1, D]
J = J[:, 0, 0, :]
```

Это один vmap-проход вместо K. На T4 для [K=128, D=1024] и небольшого decoder-tail обычно 5–20× быстрее. Точность: для линейной (в смысле grad-выхода) функции результат идентичен с точностью до fp-порядка.

Альтернативно `torch.func.vmap(torch.func.grad(...))` — но требует отсутствия in-place в forward, а здесь есть `_clone_gdn_states_on_entry` (in-place на кэше). `is_grads_batched` безопаснее: он работает на уровне autograd, не требует функциональности forward'а.

**6. `_build_causal_masks` вызывается один раз на decode-pass**

Ок. Но `create_causal_mask` с `inputs_embeds=dummy_embeds` и `past_key_values=cache` — если кэш большой (S=128), эта функция не должна зависеть от размера кэша для одного query-токена. Скорее всего ок, но стоит проверить, что для S=128 и `position_ids = 127` маска корректна. В коде комментарий об этом есть, значений не проверял.

**7. `softmax_jacobian` — корректно, но: `p` может не быть нормирован**

```python
J_bar = (p[:, None] * J).sum(dim=0, keepdim=True)
return p[:, None] * (J - J_bar)
```

Формула верна для любого `p` (не обязательно нормированного), при условии что caller нормирует перед вызовом. В `exp27_noise_floor.py` это делается (`.sum()` normalize). В `_metrics` тоже. ОК.

**8. `_deep_clone_cache` — fallback для неизвестного типа**

```python
if isinstance(layer, DynamicLayer):
    ...
else:
    # GDN (LinearAttentionLayer). Clone every state field dynamically...
    nl = LinearAttentionLayer()
```

Если `DynamicCache(config=...)` создаёт placeholder-слои другого типа (например, `None` или конфиг-специфичные), ветка `else` обработает их как `LinearAttentionLayer` и создаст **пустые** GDN-слои. При последующем обращении GDN-слоя к cache в `_run_decoder_pass` это может дать тихую неконсистентность (несоответствие числа слоёв в кэше и модели). Проверить тип `cache.layers[0]` при `DynamicCache(config=...)` на целевой версии.

---

## Место для оптимизации (не ошибки)

1. **`_forward_dense` на GPU**: `out.hidden_states` — tuple из n_layers тензоров [1, S, D]. Индексация `out.hidden_states[l+1][0].float().cpu()` спускает на CPU **только нужные слои**, но остальные живут в GPU до выхода из функции. Для n_layers=24, S=128, D=1024, fp32 это ~12 MB. Не критично, но при больших S (256) уже ощутимо. Можно `del out` сразу после вытаскивания hs нужных слоёв.

2. **`_load_pool` — dedup полный `data`**:

```python
seen, ded = set(), []
for x in data:
    t = tuple(x["input_ids"][0].tolist())
    ...
```

`tuple([...])` — материализация списка; для L=256 это 256-элементный tuple × 32 × (n*64) сэмплов. Работает, но `hash(bytes)` дешевле. Микрооптимизация.

3. **`_ranks(Jr)` для K=128, D=1024 делает `svdvals` полного J**. 24576 раз × 6 слоёв. На CPU это тяжело. Можно перейти на `torch.linalg.svdvals(J, driver='gesvd')` или `svd_lowrank` — но точность top-r нужна. Оставить как есть.

4. **`_metrics` в exp27_noise_floor.py — `softmax_jacobian` вызывается для каждого (dom, si, S, p1, p2) многократно**. Например, `J1p = softmax_jacobian(J1, pk1)` внутри тройного цикла по парам. Если несколько пар делят один (dom, si, S, p), можно кэшировать. Минорно.

5. **`_j_configs` возвращает (S, K_of_128, pos)** — везде K=128, никогда не свёртывается. K вынесен в отдельную ось в `_metrics`, но `cfgs` хранит только 128. Это by design (J считается один раз при U_128, K-подмножества — срезы строк). Комментарий есть, ОК.

---

## Приоритеты

| Что | Тип | Влияние |
|---|---|---|
| `fwd` не пересобирать per-layer | баг-эффективности | **6× на скорость exp27b** |
| Хранить `p128_all` вместо `logits` | RAM | **~1000×** в `fwd` |
| `is_grads_batched=True` в `_compute_jacobian_rows` | скорость jacobian | **5–20×** на backward |
| Verify `DynamicCache(config=...)` и тип placeholder-слоёв | совместимость | риск NameError/тихой неконсистентности |

Математика и структура (`_ranks`, `softmax_jacobian`, `_last_layer_jacobian`, MAD-границы, resume-логика после `all(...)`) — корректны. Ошибок уровня «не компилируется / не запускается» не вижу при условии, что `DynamicCache(config=...)` работает на целевой версии transformers.