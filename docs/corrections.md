## Самопроверка: что агент не поймёт или сломает

Перечитал v3.0 глазами агента, который видит код Qwen3.5 впервые. Нашёл пять мест, где он споткнётся.

### 1. `use_cache=True` в вызове слоя — лишний и опасный

В v3.0 написано:

```python
layer_output = layer(
    x,
    position_embeddings=position_embeddings,
    attention_mask=mask,
    position_ids=text_position_ids,
    past_key_values=cache_clone,
    use_cache=True,  # ← НЕ НУЖНО
)
```

Сигнатура `Qwen3_5DecoderLayer.forward()` **не принимает** `use_cache`. Он улетит в `**kwargs` и может вызвать неожиданное поведение. Решение: убрать. Слой решает, использовать кэш или нет, по `past_key_values is not None`.

```python
layer_output = layer(
    x,
    position_embeddings=position_embeddings,
    attention_mask=mask,
    position_ids=text_position_ids,
    past_key_values=cache_clone,
)
```

### 2. Структура `DynamicCache` для Qwen3.5 не описана

Агент будет делать `copy.deepcopy(prefix_cache)`, не зная, что внутри. Для Qwen3.5 кэш содержит **два типа состояний**:

- **Attention-слои:** `cache.key_cache[layer_idx]`, `cache.value_cache[layer_idx]` — тензоры `[B, H, S, D]`
- **GDN-слои:** `cache.layers[layer_idx].conv_states[0]` — тензор `[1, conv_dim, kernel_size - 1]`, и `cache.layers[layer_idx].recurrent_states[0]` — тензор `[1, num_v_heads, head_k_dim, head_v_dim]`

Агент должен знать, что `deepcopy` клонирует **оба типа**. И что GDN-состояние **нельзя обрезать** на один токен — это агрегированная матрица.

Добавить в документ:

```python
# Структура кэша Qwen3.5 (DynamicCache):
# Для full_attention слоёв:
#   cache.key_cache[layer_idx]: [B, num_kv_heads, seq_len, head_dim]
#   cache.value_cache[layer_idx]: [B, num_kv_heads, seq_len, head_dim]
# Для linear_attention (GDN) слоёв:
#   cache.layers[layer_idx].conv_states[0]: [1, conv_dim, conv_kernel_size - 1]
#   cache.layers[layer_idx].recurrent_states[0]: [1, num_v_heads, head_k_dim, head_v_dim]
```

### 3. Маска внимания для одного токена с кэшем

В v3.0 написано `mask = None`. Для GDN при `seq_len=1` это работает (рекуррентная маска тривиальна). Для attention при `seq_len=1` с кэшем это **тоже работает**, потому что один токен attend-ит на все кэшированные ключи без ограничений.

Но агент может перестраховаться и начать строить маску вручную. Добавить явное указание:

```python
# При seq_len=1 с кэшем маска НЕ НУЖНА.
# Attention: один токен attend-ит на все кэшированные KV.
# GDN: рекуррентное состояние уже содержит контекст.
mask = None
```

### 4. Позиционные эмбединги: форма `[4, B, S]`

В Qwen3.5 `position_ids` имеют форму `[4, B, S]` (text, temporal, height, width). Для текстового входа без изображений все 4 компоненты одинаковы. В v3.0 я написал:

```python
position_ids = torch.tensor([[[S - 1]]], device=input_ids.device)
position_ids = position_ids.expand(4, 1, 1)
```

Это корректно, но агент может не понять, зачем `4`. Добавить комментарий:

```python
# Qwen3.5 использует MRoPE с 4 компонентами: text, temporal, height, width.
# Для текстового входа все 4 одинаковы.
# При seq_len=1 с кэшем позиция последнего токена = S - 1 (0-indexed).
position_ids = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=input_ids.device)
text_position_ids = position_ids[0]  # [1, 1]
position_embeddings = model.model.rotary_emb(h, position_ids[1:])  # [3, 1, 1]
```

### 5. `copy.deepcopy` может быть медленным

Для 32 слоёв и $S = 2048$ кэш занимает ~100–200 MB. `deepcopy` работает, но медленно. Альтернатива — ручное клонирование:

```python
import copy

def clone_cache(cache):
    """Быстрое клонирование кэша без deepcopy."""
    new_cache = DynamicCache(config=cache.config)
    for layer_idx in range(len(cache.key_cache)):
        if layer_idx < len(cache.key_cache) and cache.key_cache[layer_idx] is not None:
            new_cache.key_cache.append(cache.key_cache[layer_idx].clone())
            new_cache.value_cache.append(cache.value_cache[layer_idx].clone())
    # Для GDN слоёв
    if hasattr(cache, 'layers'):
        for layer_idx, layer_cache in cache.layers.items():
            new_cache.layers[layer_idx].conv_states = [s.clone() for s in layer_cache.conv_states]
            new_cache.layers[layer_idx].recurrent_states = [s.clone() for s in layer_cache.recurrent_states]
    return new_cache
```

Но для v1 `deepcopy` допустим. Оптимизация — после валидации.

---

## Итого: что добавить в v3.0

| Проблема | Где | Исправление |
|----------|-----|------------|
| `use_cache=True` в вызове слоя | Шаг 1, Проход 3 | Убрать параметр |
| Структура DynamicCache не описана | Раздел 3 | Добавить блок с описанием структуры кэша |
| Маска для seq_len=1 | Шаг 1, Проход 3 | Явный комментарий: `mask = None` корректен |
| Форма position_ids `[4, B, S]` | Шаг 1, Проход 3 | Комментарий про MRoPE |
| `deepcopy` медленный | Шаг 1, Проход 3 | Примечание: для v1 допустим, оптимизация потом |

Ни одно из этих замечаний не ломает математику. Все пять — инженерные уточнения, чтобы агент не споткнулся.

---

## Название

Суть проекта: Якобиан — это **призма**, которая разлагает поведение модели на компоненты чувствительности. Пространство словаря — это **спектр**, через который мы сравниваем модели разного размера. Мы пропускаем «белый свет» модели через призму Якобиана и видим, какие компоненты как работают.

**Prism.**

Подзаголовок: *Jacobian Bridge for Cross-Scale Decoding Alignment.*

Рабочее название для кода: `prism-bridge` или просто `prism`.

```
prism/
├── jacobian.py          # compute_jacobian_for_tokens
├── bridge.py            # jacobian_bridge, проекция через E
├── esg.py               # Empirical Sensitivity Gramian
├── diagnostics.py       # тепловая карта, бейзлайны KL/CKA
├── calibration.py       # калибровочные данные, глобальный набор U
└── experiments/
    ├── exp1_identity.py
    ├── exp2_heatmap.py
    ├── exp3_random_proj.py
    └── ...
```

Горжусь? Да. Это красивая идея — использовать пространство словаря как универсальную систему координат для сравнения моделей, которые «думают» в разных размерностях. И ты эту идею сгенерировал. Я только довёл до формул.

Так что нет. Ты не «только генератор идеи». Без твоего вопроса про Якобианы из старшей модели в младшую через эмбеддинги — ничего бы этого не было. Я бы продолжал писать про CKA и GPTQ.

**Prism** — наш проект.