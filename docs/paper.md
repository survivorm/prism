# Якобианов Мост v3.0: Техническое задание на реализацию

> **Статус:** Рабочая гипотеза. Ни одно утверждение об улучшении качества модели не подтверждено экспериментально. Математический аппарат (размерности, градиенты) корректен. Ожидаемые эффекты — гипотезы. Документ написан для проверки этих гипотез.

---

## 0. Что реализуем и зачем

### 0.1. Суть метода

Даны две модели одного семейства (Qwen3.5-8B и Qwen3.5-0.8B), общий словарь. Метод сравнивает **чувствительность декодирования** — то, как скрытые состояния каждой модели влияют на финальные логиты. Сравнение происходит через пространство словаря, которое является общей системой координат для обеих моделей.

### 0.2. Цели (по приоритету)

| Приоритет | Цель | Что получаем | Статус |
| ----------- | ------ | ------------- | -------- |
| **P0** | Диагностика: послойная карта чувствительности декодирования | Тепловая карта $\mathcal{L}_{\text{bridge}}$ по слоям | Гипотеза |
| **P1** | Карта чувствительности для квантования: какие направления защищать | Собственные векторы ESG по слоям | Гипотеза |
| **P2** | Трансформация младшей модели | Лосс для оптимизации | Гипотеза, отложено до результатов P0 |
| **P3** | Регуляризатор при дообучении | Дополнительный лосс | Гипотеза, отложено до результатов P0/P2 |

**Реализуем P0 и P1.** P2 и P3 — только если P0 покажет осмысленную структуру.

### 0.3. Ограничения версии 1

- **Tied weights:** $W = E$ (в Qwen3.5 выполнено: `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`)
- **Общий словарь:** обе модели используют один токенизатор
- **Текстовый вход:** без изображений/видео (только `Qwen3_5ForCausalLM`)
- **CPU:** Intel Iris Xe / SYCL, 64GB RAM. `torch.set_num_threads(4)`. float32 вместо float16 (fp16 на CPU слишком медленный).
- **Qwen3.5-0.8B API:** `model.model` — это `Qwen3_5TextModel` (нет `language_model`). Для Qwen3.5-4B/8B: `model.model.language_model`.
- **Модель:** Qwen3.5-4B (32 слоя, $D=2560$) вместо 8B для экспериментов

---

## 1. Математические определения

### 1.1. Модели и матрицы

- Старшая модель $\mathcal{M}_L$: скрытая размерность $D_L$, $n_L$ слоёв, словарь $V$
- Младшая модель $\mathcal{M}_S$: скрытая размерность $D_S < D_L$, $n_S$ слоёв, тот же словарь $V$
- Матрица эмбедингов $E \in \mathbb{R}^{V \times D}$: строка $E[i]$ — эмбединг токена $i$
- Для tied weights: матрица анэмбединга $W = E$

### 1.2. Чувствительность декодирования (Якобиан)

**Определение.** Для слоя $l$, позиции $t$:

$$J_l^{(t)} = \frac{\partial z_t}{\partial h^l_t} \in \mathbb{R}^{V \times D}$$

где $z_t \in \mathbb{R}^V$ — логиты, $h^l_t \in \mathbb{R}^D$ — скрытое состояние.

На практике вычисляется только для фиксированного набора токенов $\mathcal{U}$ (см. раздел 2.1):

$$J_l^{(t)}\big|_{\mathcal{U}} \in \mathbb{R}^{|\mathcal{U}| \times D}$$

### 1.3. Лемма о размерностях (Якобианов Мост)

**Лемма.** Пусть $J_L \in \mathbb{R}^{|\mathcal{U}| \times D_L}$, $J_S \in \mathbb{R}^{|\mathcal{U}| \times D_S}$, $E_L \in \mathbb{R}^{V \times D_L}$, $E_S \in \mathbb{R}^{V \times D_S}$. Тогда:

**(а)** Проекция в пространство словаря: $\tilde{J}_L = J_L \cdot E_L^T \in \mathbb{R}^{|\mathcal{U}| \times V}$

**(б)** Целевой Якобиан для младшей модели: $J_S^{\text{target}} = \tilde{J}_L \cdot E_S \in \mathbb{R}^{|\mathcal{U}| \times D_S}$

**(в)** $J_S$ и $J_S^{\text{target}}$ имеют одинаковую размерность $|\mathcal{U}| \times D_S$ и непосредственно сравнимы.

> **⚠️ ГИПОТЕЗА H3.** Проекция через $E_L^T \cdot E_S$ сохраняет семантический смысл для моделей одного семейства. Не доказано. Проверяется Экспериментом 3 (абляция со случайной матрицей).

### 1.4. Коррекция через псевдообратную Грамианову матрицу

**Проблема.** Для реальных эмбеддингов $E^T \cdot E \neq I$ (неортогональность). Это значит, что при $J_L = J_S$ и $E_L = E_S$ (одна и та же модель) лосс не равен нулю — проектция $J \cdot E^T \cdot E$ вращает векторы.

**Решение.** Вместо $J_L \cdot E_L^T$ использовать:

$$\tilde{J}_L = J_L \cdot E_L^T \cdot E_L \cdot G_L^{-1} = J_L \cdot G_L \cdot G_L^{-1} = J_L$$

где $G_L = E_L^T E_L$ и $G_L^{-1} = \text{pinv}(G_L)$ (псевдообратная через SVD для числовой стабильности).

На практике: $J_L$ проецируется через $E_L^T \cdot E_L \cdot \text{pinv}(E_L^T E_L)$, что при идентичных моделях даёт $\mathcal{L}_{\text{bridge}} \approx 10^{-7}$.

**Оптимизация.** $G_L^{-1}$ вычисляется один раз и кэшируется на диск (`results/.gramians/`) с хешем по форме и содержимому $E_L$.

> **Опыт:** без коррекции $\mathcal{L}_{\text{bridge}} \approx 0.6\text{—}0.85$. С коррекцией: $\approx 10^{-7}$.

### 1.5. Empirical Sensitivity Gramian (ESG)

**Определение.** Для слоя $l$ на $N$ калибровочных сэмплах:

$$\bar{G}_l = \frac{1}{N} \sum_{i=1}^{N} J_l^{(i)} \cdot \left(J_l^{(i)}\right)^T \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{U}|}$$

Спектральное разложение: $\bar{G}_l = U \Lambda U^T$. Собственные векторы $u_1, ..., u_{|\mathcal{U}|}$ — главные направления чувствительности декодирования в пространстве топ-$|\mathcal{U}|$ логитов.

> **Примечание.** Это НЕ Neural Tangent Kernel в классическом смысле (Jacot et al., 2018). NTK — градиенты по параметрам. ESG — градиенты по активациям. Термин NTK не использовать.

### 1.6. Функция потерь

**Определение.** Якобианов лосс для слоя $l$:

$$\mathcal{L}_{\text{bridge}}(l) = \frac{1}{|\mathcal{U}|} \sum_{i \in \mathcal{U}} w_i \cdot \left(1 - \frac{J_S[i] \cdot J_S^{\text{target}}[i]}{\|J_S[i]\| \cdot \|J_S^{\text{target}}[i]\| + \epsilon}\right)$$

где:

- $w_i = p_L(i)$ — вероятность токена $i$ в распределении старшей модели
- $\epsilon = 10^{-8}$
- Строки $J_S$ и $J_S^{\text{target}}$ соответствуют **одним и тем же токенам** из $\mathcal{U}$ (выравнивание гарантировано)

Если $\|J_S[i]\| < \tau$ (порог затухания, $\tau = 10^{-6}$): строка $i$ исключается из суммы (маскируется).

---

## 2. Алгоритм

### 2.1. Шаг 0: Подготовка

#### Калибровочные данные

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-8B")
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

calibration_samples = []
for sample in dataset:
    tokens = tokenizer(sample["text"], max_length=2048, truncation=True, return_tensors="pt")
    if tokens["input_ids"].shape[1] >= 512:
        calibration_samples.append(tokens)
    if len(calibration_samples) >= 32:
        break
```

#### Глобальный набор токенов $\mathcal{U}$

Top-K токены **фиксируются заранее** по частоте на калибровочном наборе. Это гарантирует, что строки Якобианов выровнены по одним и тем же токенам на всех сэмплах.

```python
import torch
from collections import Counter

def build_global_token_set(model, calibration_samples, K=256):
    """Собирает глобальный набор токенов из top-K логитов на всех сэмплах."""
    token_counter = Counter()
    model.eval()
    
    with torch.no_grad():
        for sample in calibration_samples:
            outputs = model(
                input_ids=sample["input_ids"].to(model.device),
                attention_mask=sample["attention_mask"].to(model.device),
            )
            # Берём логиты последнего токена
            logits = outputs.logits[:, -1, :]  # [1, V]
            top_indices = logits.topk(K, dim=-1).indices[0]  # [K]
            token_counter.update(top_indices.tolist())
    
    # Берём топ-|U| самых частых токенов
    U_size = min(K, len(token_counter))
    U = [token_id for token_id, _ in token_counter.most_common(U_size)]
    return sorted(U)
```

**Размер $\mathcal{U}$:** Ожидаемо $|\mathcal{U}| \leq K$. Для типичных текстов перекрытие top-K между сэмплами высоко, поэтому $|\mathcal{U}|$ будет близок к $K$.

#### Соответствие слоёв

Для версии 1 — равномерное:

```python
def uniform_layer_mapping(n_layers_large, n_layers_small):
    mapping = {}
    for i in range(n_layers_large):
        j = round(i * (n_layers_small - 1) / (n_layers_large - 1))
        mapping[i] = j
    return mapping
```

#### Загрузка моделей

```python
import torch
from transformers import AutoModelForCausalLM

# CPU: float32 обязателен! fp16 на CPU в 10x медленнее.
torch.set_num_threads(4)

model_large = AutoModelForCausalLM.from_pretrained(
    "/data/Qwen/Qwen3.5-4B",
    torch_dtype=torch.float32,  # float32 на CPU!
    attn_implementation="eager",  # важно: не flash_attention, нужны hidden_states
)
model_large.eval()

model_small = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype=torch.float32,
    attn_implementation="eager",
)
model_small.eval()

# Извлекаем матрицы эмбедингов
# Qwen3.5-0.8B: model.model — это Qwen3_5TextModel (нет language_model)
# Qwen3.5-4B/8B: model.model.language_model — Qwen3_5TextModel
E_large = model_large.model.embed_tokens.weight.detach()  # [V, D_L]
E_small = model_small.model.embed_tokens.weight.detach()  # [V, D_S]
```

### 2.2. Шаг 1: Вычисление Якобиана (с корректным кэшем)

> **КРИТИЧНО:** Кэш должен содержать состояние **ДО** последнего токена, а не после. Два отдельных прохода.

```python
import copy

def compute_jacobian_for_tokens(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,       # [1, S]
    attention_mask: torch.Tensor,   # [1, S]
    token_set: list[int],           # глобальный набор U
) -> torch.Tensor:
    """
    Вычисляет Якобиан логитов по скрытому состоянию слоя layer_idx
    для последнего токена, только для токенов из token_set.
    
    Возвращает: J [len(token_set), D] — Якобиан (ненормализованный)
    """
    model.eval()
    S = input_ids.shape[1]
    
    # ============================================================
    # ПРОХОД 1: Префикс (все токены кроме последнего) → кэш
    # ============================================================
    with torch.no_grad():
        prefix_outputs = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
            output_hidden_states=False,
        )
    prefix_cache = prefix_outputs.past_key_values
    
    # ============================================================
    # ПРОХОД 2: Полная последовательность → скрытые состояния
    # ============================================================
    with torch.no_grad():
        full_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    
    # Извлекаем скрытое состояние последнего токена на слое layer_idx
    # hidden_states[0] = эмбединги, hidden_states[l+1] = выход слоя l
    h = full_outputs.hidden_states[layer_idx + 1][:, -1:, :]  # [1, 1, D]
    h = h.float().detach().requires_grad_(True)
    
    # ============================================================
    # ПРОХОД 3: Прогон h через оставшиеся слои с кэшем ПРЕФИКСА
    # ============================================================
    # Клонируем кэш, чтобы не испортить оригинал
    cache_clone = copy.deepcopy(prefix_cache)
    
    # Получаем оставшиеся слои
    # model.model.layers содержит все декодерные слои
    remaining_layers = model.model.layers[layer_idx + 1:]
    norm_layer = model.model.norm
    lm_head = model.lm_head
    
    # Позиционные данные для последнего токена
    # Qwen3.5 использует MRoPE с 4 компонентами: text, temporal, height, width.
    # Для текстового входа все 4 одинаковы.
    # При seq_len=1 с кэшем позиция последнего токена = S - 1 (0-indexed).
    position_ids = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=input_ids.device)  # [4, 1, 1]
    text_position_ids = position_ids[0]  # [1, 1]
    
    # Позиционные эмбединги (RoPE)
    # Для последнего токена с кэшем нужны cos/sin для позиции S-1
    position_embeddings = model.model.rotary_emb(h, position_ids[1:])
    
    x = h  # [1, 1, D]
    
    for i, layer in enumerate(remaining_layers):
        actual_layer_idx = layer_idx + 1 + i
        layer_type = model.config.layer_types[actual_layer_idx]
        
        # При seq_len=1 с кэшем маска НЕ НУЖНА.
        # Attention: один токен attend-ит на все кэшированные KV.
        # GDN: рекуррентное состояние уже содержит контекст.
        mask = None
        
        # Сигнатура layer.forward() не принимает use_cache —
        # слой решает сам по past_key_values is not None.
        layer_output = layer(
            x,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            position_ids=text_position_ids,
            past_key_values=cache_clone,
        )
        x = layer_output
    
    # ============================================================
    # Нормализация и логиты
    # ============================================================
    x = norm_layer(x)
    logits = lm_head(x[:, -1, :])  # [1, V]
    
    # ============================================================
    # Вычисление градиентов для токенов из token_set
    # ============================================================
    logits_subset = logits[0, token_set]  # [|U|]
    
    # Векторизованный градиент: один backward вместо |U| вызовов
    grad_outputs = torch.eye(len(token_set), device=logits.device, dtype=logits.dtype)
    
    J = torch.autograd.grad(
        outputs=logits_subset,
        inputs=h,
        grad_outputs=grad_outputs,
        retain_graph=False,
    )[0]  # [1, 1, D] → нужно извлечь для каждого токена
    
    # J имеет форму [len(token_set), 1, 1, D] из-за grad_outputs
    # Но на самом деле autograd.grad вернёт [1, 1, D] для каждого вызова
    # Нужно использовать vmap или цикл
    
    # Fallback: цикл (медленно, но надёжно)
    J = torch.zeros(len(token_set), h.shape[-1], device=h.device)
    for i in range(len(token_set)):
        grad = torch.autograd.grad(
            outputs=logits_subset[i],
            inputs=h,
            retain_graph=(i < len(token_set) - 1),
        )[0]  # [1, 1, D]
        J[i] = grad[0, 0, :]  # [D]
    
    return J  # [|U|, D], ненормализованный
```

> **⚠️ ЗАМЕЧАНИЕ ПО ПРОИЗВОДИТЕЛЬНОСТИ.** Цикл по $|\mathcal{U}|$ с `autograd.grad` — медленно. Для ускорения использовать `torch.func.jacrev` (PyTorch 2.1+) или `torch.func.vmap`. В первой реализации допустим цикл. Оптимизация — после валидации корректности.

> **⚠️ GDN DEEP CLONE.** `copy.deepcopy(prefix_cache)` не работает для GDN слоёв — `recurrent_state` и `conv_state` мутируют in-place через `cache_params.update_recurrent_state(...)` и `torch_causal_conv1d_update`. Решение: ручной клон всех тензоров в кэше:
>
> ```python
> def _deep_clone_cache(cache):
>     """Клонирует DynamicCache, учитывая GDN in-place мутации."""
>     new_cache = DynamicCache()
>     for layer in cache:
>         if hasattr(layer, 'keys'):  # DynamicLayer
>             new_cache.append(DynamicLayer(
>                 keys=[k.clone() for k in layer.keys],
>                 values=[v.clone() for v in layer.values],
>             ))
>         elif hasattr(layer, 'recurrent_states'):  # LinearAttentionLayer
>             new_conv = {k: v.clone() for k, v in layer.conv_states.items()}
>             new_recur = {k: v.clone() for k, v in layer.recurrent_states.items()}
>             new_cache.append(LinearAttentionLayer(
>                 conv_states=new_conv,
>                 recurrent_states=new_recur,
>             ))
>     return new_cache
> ```

> **⚠️ ЗАМЕЧАНИЕ ПО ПАМЯТИ.** Для длинных последовательностей это ~1-2 GB. Если не хватает памяти — уменьшить $S$ до 512.

### 2.3. Шаг 2: Якобианов Мост

```python
def jacobian_bridge(
    J_L: torch.Tensor,       # [|U|, D_L] — Якобиан старшей модели
    J_S: torch.Tensor,       # [|U|, D_S] — Якобиан младшей модели
    E_L: torch.Tensor,       # [V, D_L] — эмбеддинги старшей
    E_S: torch.Tensor,       # [V, D_S] — эмбеддинги младшей
    probs_L: torch.Tensor,   # [|U|] — вероятности токенов в старшей модели
    threshold: float = 1e-6, # порог затухания градиента
) -> float:
    """
    Вычисляет Якобианов лосс между старшей и младшей моделями.
    
    Возвращает: scalar — значение лосса
    """
    # Проекция в пространство словаря
    # J_L: [|U|, D_L], E_L^T: [D_L, V] → J_L @ E_L^T: [|U|, V]
    J_L_projected = J_L.float() @ E_L.float().T  # [|U|, V]
    
    # Обратная проекция в пространство младшей модели
    # J_L_projected: [|U|, V], E_S: [V, D_S] → J_L_projected @ E_S: [|U|, D_S]
    J_S_target = J_L_projected @ E_S.float()  # [|U|, D_S]
    
    # Маска затухших градиентов
    J_S_norms = J_S.float().norm(dim=1)       # [|U|]
    J_target_norms = J_S_target.norm(dim=1)    # [|U|]
    valid_mask = (J_S_norms > threshold) & (J_target_norms > threshold)  # [|U|]
    
    if valid_mask.sum() == 0:
        return 1.0  # максимальный лосс, если всё затухло
    
    # Косинусное сходство по валидным строкам
    cos_sim = torch.nn.functional.cosine_similarity(
        J_S.float()[valid_mask],
        J_S_target[valid_mask],
        dim=1,
    )  # [n_valid]
    
    # Взвешивание по вероятностям старшей модели
    weights = probs_L[valid_mask]
    weights = weights / (weights.sum() + 1e-8)  # нормализация весов
    
    loss = (weights * (1.0 - cos_sim)).sum().item()
    
    return loss
```

### 2.4. Шаг 3: Диагностика (Цель P0)

```python
def run_diagnostics(
    model_large, model_small,
    calibration_samples,
    token_set,
    layer_mapping,
):
    """
    Вычисляет L_bridge для всех пар слоёв.
    Возвращает: dict {(l_L, l_S): loss}
    """
    E_L = model_large.model.embed_tokens.weight.detach()
    E_S = model_small.model.embed_tokens.weight.detach()
    
    results = {}
    
    for l_L, l_S in layer_mapping.items():
        losses = []
        
        for sample in calibration_samples:
            input_ids = sample["input_ids"].to(model_large.device)
            attention_mask = sample["attention_mask"].to(model_large.device)
            
            # Якобиан старшей модели
            J_L = compute_jacobian_for_tokens(
                model_large, l_L, input_ids, attention_mask, token_set
            )
            
            # Якобиан младшей модели
            J_S = compute_jacobian_for_tokens(
                model_small, l_S, input_ids, attention_mask, token_set
            )
            
            # Вероятности токенов из token_set в старшей модели
            with torch.no_grad():
                logits_L = model_large(input_ids=input_ids).logits[:, -1, :]
                probs_L = torch.softmax(logits_L[0, token_set], dim=-1)
            
            loss = jacobian_bridge(J_L, J_S, E_L, E_S, probs_L)
            losses.append(loss)
        
        results[(l_L, l_S)] = sum(losses) / len(losses)
    
    return results
```

### 2.5. Шаг 4: ESG для карты чувствительности (Цель P1)

```python
def compute_esg(
    model,
    layer_idx: int,
    calibration_samples,
    token_set,
    n_components: int = 32,
):
    """
    Вычисляет Empirical Sensitivity Gramian для слоя layer_idx.
    
    Возвращает:
        eigenvalues: [n_components] — собственные значения
        eigenvectors: [n_components, |U|] — собственные векторы
    """
    all_J = []
    
    for sample in calibration_samples:
        input_ids = sample["input_ids"].to(model.device)
        attention_mask = sample["attention_mask"].to(model.device)
        
        J = compute_jacobian_for_tokens(
            model, layer_idx, input_ids, attention_mask, token_set
        )
        all_J.append(J)  # [|U|, D]
    
    # ESG: (1/N) * sum_i (J_i @ J_i^T)
    # J_i: [|U|, D] → J_i @ J_i^T: [|U|, |U|]
    |U| = len(token_set)
    G = torch.zeros(|U|, |U|, device=all_J[0].device)
    
    for J in all_J:
        G += J.float() @ J.float().T
    G /= len(all_J)
    
    # Спектральное разложение
    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    
    # Сортировка по убыванию
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components].T  # [n_components, |U|]
    
    return eigenvalues, eigenvectors
```

---

## 3. Специфика Qwen3.5

### 3.1. Гибридная архитектура

`config.layer_types[layer_idx]` определяет тип слоя:

- `"full_attention"` → `Qwen3_5Attention` (GQA)
- `"linear_attention"` → `Qwen3_5GatedDeltaNet` (GDN)

### 3.2. GDN: рекуррентное состояние

GDN поддерживает рекуррентное состояние $S_t \in \mathbb{R}^{H \times D_k \times D_v}$:

```
S_t = g_t * S_{t-1} + k_t * delta_t^T
output_t = S_t * q_t
```

Кэш GDN содержит:

- `conv_state`: `[1, conv_dim, conv_kernel_size - 1]`
- `recurrent_state`: `[1, num_v_heads, head_k_dim, head_v_dim]`

### 3.3. Мутация кэша

**Критично:** `torch_causal_conv1d_update` мутирует `conv_state` in-place (`conv_state.copy_(...)`). GDN обновляет `recurrent_state` через `cache_params.update_recurrent_state(...)`.

**Решение:** Всегда клонировать кэш перед использованием в вычислении Якобиана:

```python
cache_clone = copy.deepcopy(prefix_cache)
```

### 3.4. Tied weights

В Qwen3.5: `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`.

Это означает $W = E$, и проекция через $E$ корректна для логитов.

### 3.5. MRoPE

Qwen3.5 использует Multi-dimensional RoPE (MRoPE) с `position_ids` формы `[4, B, S]` (text, temporal, height, width). Для текстового входа без изображений: все 4 компоненты одинаковы.

При вычислении Якобиана для последнего токена с кэшем: `position_ids = S - 1`.

### 3.6. RMSNorm

`Qwen3_5RMSNorm` использует формулу `output = (x / rms(x)) * (1 + weight)`. Вес инициализирован нулями, поэтому начальное поведение — идентичность. Это влияет на масштаб Якобиана, но не на направление (косинусное сходство инвариантно к масштабу).

---

## 4. Протокол валидации

### Эксперимент 1: Тест на идентичных моделях (ОБЯЗАТЕЛЬНЫЙ)

Загрузить Qwen3.5-0.8B дважды. Вычислить $\mathcal{L}_{\text{bridge}}$ для всех пар слоёв.

**Ожидание:** $\mathcal{L}_{\text{bridge}} < 0.01$ для всех пар.

**Если не выполняется:** ошибка в реализации. Искать в:

- Индексации слоёв (`hidden_states[l+1]` vs `hidden_states[l]`)
- Кэше (префикс vs полная последовательность)
- Выравнивании токенов

### Эксперимент 2: Neighbor windows (пропорциональная vs end-mapping)

Qwen3.5-4B → Qwen3.5-0.8B. Вместо полной карты (дорого на CPU) — якорные слои 4B с локальными окнами слоёв 0.8B:

| Якорь 4B | Окно слоёв 0.8B |
| --- | --- |
| 0 | {0, 1, 2} |
| 8 | {0, 3, 6} |
| 16 | {8, 10, 12} |
| 24 | {16, 17, 18} |
| 30 | {21, 22, 23} |

**Интерпретация:**

- минимум у пропорционального партнёра $p(l)$ → реальное функциональное соответствие;
- минимум у end-mapping $e(l) = l - 8$ → «хвостовая» конфаунд;
- плоско и высоко → сигнал слабый.

**Бонус:** Якорь 0 — для $l_L < 8$ концевого партнёра нет у младшей модели. Если конфаунд прав, у слоя 0 не будет хорошего партнёра (лосс высокий). Если нет — минимум у 0.

Полная $32 \times 24$ карта — как фигура для статьи, после валидации.

### Эксперимент 3: Абляция со случайной матрицей

Вместо $E_L^T$ использовать случайную гауссову матрицу $R \in \mathbb{R}^{D_L \times V}$ с нормированными строками.

```python
R_random = torch.randn(D_L, V)
R_random = R_random / R_random.norm(dim=0, keepdim=True)
```

**Если $\mathcal{L}_{\text{bridge}}$ не изменится:** проекция через эмбеддинги не несёт семантической нагрузки. Мост бессмыслен.

**Если изменится значительно:** проекция через словарь работает. Гипотеза H3 подтверждается.

### Эксперимент 4: Чувствительность к $|\mathcal{U}|$

$|\mathcal{U}| = 64, 128, 256, 512$.

**Ожидание:** $\mathcal{L}_{\text{bridge}}$ стабилизируется при $|\mathcal{U}| \geq 128$.

### Эксперимент 5: Перестановка эмбеддингов

Случайно переставить строки $E_L$ (и $W_L$, но для tied weights это одна матрица).

**Если метод сломается:** мост опирается на семантику эмбеддингов. Хорошо.

**Если не сломается:** мост не использует семантику. Плохо.

### Эксперимент 6: Бейзлайн KL-дистилляция

Вычислить KL-дивергенцию между логитами старшей и младшей модели на тех же данных:

```python
def kl_baseline(model_large, model_small, sample):
    with torch.no_grad():
        logits_L = model_large(**sample).logits[:, -1, :]
        logits_S = model_small(**sample).logits[:, -1, :]
    
    p_L = F.softmax(logits_L / 2.0, dim=-1)  # temperature = 2.0
    log_p_S = F.log_softmax(logits_S / 2.0, dim=-1)
    
    return F.kl_div(log_p_S, p_L, reduction='batchmean').item()
```

**Если $\mathcal{L}_{\text{bridge}}$ коррелирует с KL:** мост не добавляет информации. Ценность только в послойной диагностике.

**Если не коррелирует:** мост улавливает структурную информацию, недоступную KL.

### Эксперимент 7: Бейзлайн CKA

Вычислить Linear CKA между скрытыми состояниями $h_L^l$ и $h_S^{\phi(l)}$:

```python
def cka_baseline(model_large, model_small, layer_L, layer_S, sample):
    with torch.no_grad():
        H_L = model_large(**sample, output_hidden_states=True).hidden_states[layer_L + 1]
        H_S = model_small(**sample, output_hidden_states=True).hidden_states[layer_S + 1]
    return linear_cka(H_L, H_S)
```

**Если $\mathcal{L}_{\text{bridge}}$ коррелирует с CKA:** мост не добавляет информации сверх сравнения представлений.

**Если не коррелирует:** мост улавливает информацию о декодировании, а не о представлениях.

### Эксперимент 8: SVD-абляция проекции

Вместо полной $E^T$ использовать усечённую проекцию через топ-$r$ сингулярных компонент:

```python
U, S, Vt = torch.linalg.svd(E_L, full_matrices=False)
E_L_truncated = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r, :]  # [V, D_L], rank r
```

**Если метод устойчив к усечению:** он опирается на главные семантические направления.

**Если ломается:** он полагается на шумовые компоненты.

---

## 5. Приоритеты реализации

| Этап | Что | Оценка трудоёмкости |
| ------ | ----- | ------------------- |
| **1** | Загрузка моделей, калибровочные данные, глобальный набор $\mathcal{U}$ | 1 день |
| **2** | `compute_jacobian_for_tokens` с корректным кэшем (2 прохода) | 2-3 дня |
| **3** | `jacobian_bridge` (проекция + лосс) | 1 день |
| **4** | Эксперимент 1 (идентичные модели) | 0.5 дня |
| **5** | Эксперимент 2 (тепловая карта) | 1 день |
| **6** | Эксперименты 3, 6, 7 (абляции и бейзлайны) | 2 дня |
| **7** | ESG для карты чувствительности | 1 день |
| **8** | Эксперименты 4, 5, 8 (дополнительные абляции) | 2 дня |

**Итого:** ~10-12 рабочих дней на полный протокол.

**Минимальный жизнеспособный эксперимент:** Этапы 1-4 (3-4 дня). Если Эксперимент 1 не проходит — ищем баг. Если проходит — переходим к Этапу 5.

---

## 6. Гипотезы и их статус

| ID | Гипотеза | Как проверяется | Приоритет |
| ---- | ---------- | ---------------- | ----------- |
| H1 | Якобиан кодирует осмысленную «чувствительность декодирования» | Эксп. 2 (тепловая карта), Эксп. 7 (CKA) | P0 |
| H2 | Top-$ | \mathcal{U} | $ покрывает основную вероятностную массу | Эксп. 4 (чувствительность к $ | \mathcal{U} | $) | P0 |
| H3 | Проекция через $E_L^T \cdot E_S$ сохраняет семантику | Эксп. 3 (случайная матрица), Эксп. 5 (перестановка) | P0 |
| H4 | Минимизация $\mathcal{L}_{\text{bridge}}$ улучшает модель | НЕ проверяется в v1. Отложено. | P2 |
| H5 | ESG кодирует глобальную стратегию декодирования | Эксп. 2 + ESG на разных калибровочных наборах | P1 |

---

## 7. Известные ограничения

1. **Локальность Якобиана.** Якобиан — касательная в конкретной точке. Агрегация по $N$ сэмплам частично решает проблему, но не полностью.
2. **Неортогональность $E$.** Матрица эмбедингов не ортогональна. Проекция $J \cdot E^T$ не изометрична. Проверяется Экспериментом 8.
3. **Ранние слои.** Для $l < n/4$ градиент может затухнуть. В первой реализации: если $\|J[i]\| < 10^{-6}$, строка маскируется.
4. **Float16 точность.** Градиенты вычисляются в float32 (`h = h.float()`), но веса модели остаются в float16. Это может давать неточности. При проблемах: загрузить модель в float32.
5. **Память.** `copy.deepcopy(prefix_cache)` для $S = 2048$ может требовать 1-2 GB. При нехватке: уменьшить $S$ до 512.

---

*Документ подготовлен для передачи агенту-разработчику. Версия 3.0. Все гипотезы явно маркированы. Протокол валидации самодостаточен.*
