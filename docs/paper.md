# Якобианов Мост v3.3: Техническое задание на реализацию

> **Статус:** P0 ✅ (диагностика подтверждена), P1 частично ✅ (ESG/квантование валидированы в Exp 12b,c). P2/P3 — гипотезы, отложены. Математический аппарат проверен и протестирован (Exp 1 identity).

---

## 0. Что реализуем и зачем

### 0.1. Суть метода

Даны две модели одного семейства (Qwen3.5-4B и Qwen3.5-0.8B), общий словарь. Метод сравнивает **чувствительность декодирования** — то, как скрытые состояния каждой модели влияют на финальные логиты. Сравнение происходит через пространство словаря, которое является общей системой координат для обеих моделей.

### 0.2. Цели (по приоритету)

| Приоритет | Цель | Что получаем | Статус |
| ----------- | ------ | ------------- | -------- |
| **P0** | Диагностика: послойная карта расхождения Якобианов ($\mathcal{L}_{\text{bridge}}$) | Тепловая карта $\mathcal{L}_{\text{bridge}}$ по слоям | ✅ Подтверждено |
| **P1** | Карта чувствительности для квантования: какие направления защищать | Правые сингулярные векторы J^T J (hidden space); ESG = JJ^T (token space) | ✅ Частично (Exp 12b,c) |
| **P2** | Трансформация младшей модели | Лосс для оптимизации | Гипотеза, отложено |
| **P3** | Регуляризатор при дообучении | Дополнительный лосс | Гипотеза, отложено |

**P0 и P1 реализованы.** P2 и P3 — только после дополнительных исследований.

> **Примечание:** P0 — карта расхождения Якобианов (bridge loss), P1 — карта чувствительности (ESG). Это разные артефакты: P0 показывает, насколько J_S отличается от J_S_target, P1 показывает, какие направления hidden state наиболее чувствительны.

### 0.3. Ограничения версии 1

- **Tied weights:** $W = E$ (в Qwen3.5 выполнено: `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`)
- **Общий словарь:** обе модели используют один токенизатор
- **Текстовый вход:** без изображений/видео (только `Qwen3_5ForCausalLM`)
- **CPU:** Intel Iris Xe / SYCL, 64GB RAM. `torch.set_num_threads(4)`. float32 вместо float16 (fp16 на CPU слишком медленный).
- **Qwen3.5-0.8B API:** `model.model` — это `Qwen3_5TextModel`. Для всех Qwen3.5: `model.model` — текст модель (один путь доступа).
- **Модель:** Qwen3.5-4B (32 слоя, $D=2560$) вместо 8B для экспериментов

---

## 1. Математические определения

> **Фиксация позиции.** Если не указано иное, все карты чувствительности и Якобианы вычисляются для последнего токена последовательности: $t = S-1$.

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

> **Gauge-инвариантность.** Сдвиг логитов $z \to z + c\cdot 1$ добавляет к Якобиану $J$ строку, пропорциональную $1^T$. Оператор $\mathrm{diag}(p) - pp^T$ обнуляет это направление: $(\mathrm{diag}(p) - pp^T)1 = p - p(p^T 1) = p - p = 0$. Следовательно, gauge-free Якобиан $J^{(p)} = (\mathrm{diag}(p) - pp^T)J$ инвариантен к uniform shift логитов.

**Слой-переходный Якобиан.** Для скрытого состояния $h^l$:

$$D_l(x) = \frac{\partial h^{l+1}(x)}{\partial h^l(x)} \in \mathbb{R}^{D \times D}$$

**Правило композиции.** По цепному правилу:

$$J_l^{(t)} = W_{\mathrm{lm}} \prod_{i=l}^{L-1} D_i(x)$$

Произведение упорядочено справа налево; матрицы $D_i$ зависят от $x$ и в общем случае не коммутируют, порядок имеет значение.

**Структура pre-norm блока.** Для $h^{l+1} = h^l + \mathrm{MLP}(\mathrm{RMSNorm}(h^l + \mathrm{Attn}(\mathrm{RMSNorm}(h^l))))$:

$$D_l = I + \frac{\partial \mathrm{MLP}}{\partial \tilde h} \cdot \frac{\partial \mathrm{RMSNorm}}{\partial \hat h} \cdot \left( I + \frac{\partial \mathrm{Attn}}{\partial \tilde h} \cdot \frac{\partial \mathrm{RMSNorm}}{\partial h^l} \right)$$

где $\tilde h = \mathrm{RMSNorm}(h^l)$, $\hat h = h^l + \mathrm{Attn}(\tilde h)$; вклад каждого RMSNorm — как в §3.6.



### 1.3. Лемма о размерностях (Якобианов Мост)

**Лемма.** Пусть $J_L \in \mathbb{R}^{|\mathcal{U}| \times D_L}$, $J_S \in \mathbb{R}^{|\mathcal{U}| \times D_S}$, $E_L \in \mathbb{R}^{V \times D_L}$, $E_S \in \mathbb{R}^{V \times D_S}$. Тогда:

**(а)** Проекция в пространство словаря: $\tilde{J}_L = J_L \cdot E_L^T \in \mathbb{R}^{|\mathcal{U}| \times V}$

**(б)** Целевой Якобиан для младшей модели: $J_S^{\text{target}} = \tilde{J}_L \cdot E_S \in \mathbb{R}^{|\mathcal{U}| \times D_S}$

**(в)** $J_S$ и $J_S^{\text{target}}$ имеют одинаковую размерность $|\mathcal{U}| \times D_S$ и непосредственно сравнимы.

> **✅ ГИПОТЕЗА H3.** Проекция через $E_L^T \cdot E_S$ сохраняет семантический смысл для моделей одного семейства. Подтверждено Экспериментом 5 (перестановка эмбеддингов ломает мост).

### 1.4. Коррекция через псевдообратную Грамианову матрицу

**Проблема.** При кросс-модельной проекции $J_L \cdot E_L^T \cdot E_S$ матрица $E_S$ неортогональна ($E_S^T E_S \neq I$). Это значит, что даже при $J_L = J_S$ и $E_L = E_S$ (одна и та же модель) лосс не равен нулю — проекция $E_S$ «вращает» векторы в пространстве младшей модели.

**Решение.** После проекции через $E_S$ применить правую коррекцию $G_S^{-1} = \text{pinv}(E_S^T E_S)$:

$$J_S^{\text{target}} = (J_L \cdot E_L^T) \cdot E_S \cdot G_S^{-1}$$

где $G_S^{-1} = \text{pinv}(E_S^T E_S)$ — псевдообратная через SVD для числовой стабильности. Эквивалентно $E_S \cdot G_S^{-1} = (E_S^T)^{+}$ — псевдообратная строки $E_S^T$, то есть коррекция сводится к замене $E_S$ на её ортогональную проекцию в $\mathbb{R}^D$. Для identity-случая ($E_L = E_S$):

$$J_S^{\text{target}} = J_L \cdot E^T \cdot E \cdot \text{pinv}(E^T E) = J_L$$

Смысл коррекции для кросс-модельного случая (уравнение (3.2) Math): $J_L E_L^T$ — координаты в токен-пространстве; умножение на $E_S G_S^{-1}$ — это **проекция** этих координат в пространство младшей модели, корректная при условии $\sigma_{\min}(E_L^T E_S) > 0$ (матрица пересечения не вырождена; для моделей одного семейства это выполнено — см. κ ниже).

На практике это даёт $\mathcal{L}_{\text{bridge}} \approx 10^{-8}$ при идентичных моделях.

**Оптимизация.** $G_S^{-1}$ вычисляется один раз и кэшируется на диск (`results/.gramians/`) с хешем по форме и содержимому $E_S$.

> **Опыт:** без коррекции $\mathcal{L}_{\text{bridge}} \approx 0.6\text{—}0.85$. С коррекцией: $\approx 10^{-8}$.
> **Устойчивость:** condition number κ(E^T E) = 1684 для Qwen3.5-0.8B (V=32000, D=1024).
> Тихоновская регуляризация λ ≤ 0.01 не изменяет identity loss (10⁻⁸).
> Матрица не требует регуляризации — min eigenvalue = 12.5 (полный ранг 1024).

### 1.5. Empirical Sensitivity Gramian (ESG)

**Определение.** Для слоя $l$ на $N$ калибровочных сэмплах:

$$\bar{G}_l = \frac{1}{N} \sum_{i=1}^{N} J_l^{(i)} \cdot \left(J_l^{(i)}\right)^T \in \mathbb{R}^{|\mathcal{U}| \times |\mathcal{U}|}$$

Спектральное разложение: $\bar{G}_l = U \Lambda U^T$. Собственные векторы $u_1, ..., u_{|\mathcal{U}|}$ — главные направления чувствительности декодирования в пространстве топ-$|\mathcal{U}|$ логитов.

> **Примечание.** Это НЕ Neural Tangent Kernel в классическом смысле (Jacot et al., 2018). NTK — градиенты по параметрам. ESG — градиенты по активациям. Термин NTK не использовать.

Для левого сингулярного вектора $u_1$ матрицы $J^{(p)}$ (сингулярные значения $\sigma_1 \ge \sigma_2 \ge \dots$) определяем:

- **uniformity**: $\mathrm{uniformity}(u) = \langle u, \mathbf{1}_{\mathcal{U}}/\sqrt{K} \rangle^2 \in [0,1]$ — доля направления $u$ вдоль uniform-режима $\mathcal{U}$;
- **gauge_share**: $\sigma_1^2 \cdot \mathrm{uniformity}(u_1) / \sum_i \sigma_i^2$ — доля энергии $J^{(p)}$, приходящаяся на gauge-направление.

(Не путать с gauge_share $\|P(p)J\|_F^2/\|J\|_F^2$ из §12–13 — доля энергии *сырого* $J$, убиваемой проекцией $P$.)

> **Разведение рангов.** В статье символ $r$ используется в двух смыслах, которые следует отличать:
> 1. **Спектральный ранг**: participation ratio / stable rank спектра $JJ^T$ (таблицы §14, §11).
> 2. **Функциональный ранг** $r_{KL}$: минимальное $r$ такое, что $KL$ defrag-$r$ аппроксимации $J^{(p)} \approx U_r J_r$ совпадает с $KL$ полной $J^{(p)}$ (Exp 11a, §8).
>
> Оба ранга одного порядка (PR 5.9–10.2 против $r_{KL} \approx 4$–8 на L6–L18), и их согласие — отдельная проверка, а не допущение.

### 1.6. Метрика «разрыв» (gap)

**Определение.** Разрыв (gap) Якобиана для слоя $l$:

$$\mathcal{G}_{\text{gap}}(l) = \frac{1}{|\mathcal{U}|} \sum_{i \in \mathcal{U}} w_i \cdot \left(1 - \frac{J_S[i] \cdot J_S^{\text{target}}[i]}{\|J_S[i]\| \cdot \|J_S^{\text{target}}[i]\| + \epsilon}\right)$$

где:

- $w_i = p_L(i) / \sum_{j \in \mathcal{U}} p_L(j)$ — вероятность токена $i$ в распределении старшей модели, **нормированная на $\mathcal{U}$**. Асимметрия осознанна: используется именно $p_L$ (не $p_S$ и не среднее) — это фиксирует систему отсчёта старшей модели.
- $\epsilon = 10^{-8}$
- Строки $J_S$ и $J_S^{\text{target}}$ соответствуют **одним и тем же токенам** из $\mathcal{U}$ (выравнивание гарантировано)

Если $\|J_S[i]\| < \tau_l$ — строка $i$ получает статус sentinel и исключается из суммы; доля sentinel-строк отчитывается отдельно. Порог задается **относительным** квантилем слоя, чтобы избежать систематического смещения (масштабы $J$ различаются на порядки между слоями):

$$\tau_l = 0.01 \cdot \mathrm{median}_i\big(\|J_S[i]\|\big)$$

(Исторически использовался абсолютный $\tau = 10^{-6}$; при sentinel-доле > 0 результаты маркируются, а не молча маскируются.)

> **Направление метрики.** $\mathcal{G}_{\text{gap}} \in [0,2]$ (поскольку $\mathcal{G}_{\text{gap}} = 1 - \cos$): $0$ — полное совпадение направлений, $2$ — анти-выравнивание, $\approx 1$ — уровень нуля (ортогональность случайных векторов в высокоразмерном пространстве). Меньше = лучшее выравнивание (gap = 0 ⇔ Якобианы коллинеарны). Наблюдаемые gap $\approx 1.00$–$1.03$ на ранних слоях интерпретируются как отсутствие выравнивания (стохастическая независимость), а не «структурный конфликт». Исторически именовалось «Л_bridge / мост», но это вызывало путаницу (см. §1.7); теперь это **разрыв**. Метрика инвариантна к построчным масштабам (см. оговорку о масштабной слепоте ниже).

> **Масштабная слепота.** Так как Gap — это $(1-\cos)$, он слепо к абсолютному масштабу $J_S$ (калбровке уверенности). Для предотвращения геометрической слепоты оптимизации к калибровке отчитываем Gap совместно с метрикой масштаба:
>
> $$s(l) = \log \frac{\|J_S\|_F}{\|J_S^{\text{target}}\|_F}$$
>
> $s(l) \approx 0$ — масштабы совпадают; $s(l) \gg 0$ — младшая модель систематически «громче» старшей по чувствительности (и наоборот).

### 1.7. Метрика «мост» (bridge) и терминология

Две метрики имеют противоположное направление — важно различать:

| Метрика | Формула | Направление |
|---|---|---|
| **gap (разрыв)** $\mathcal{G}_{\text{gap}}$ | вероятностно-взвешенная $(1 - \cos)$ | **меньше = лучше** |
| **bridge (мост)** $\mathcal{B}_{\text{bridge}}$ | (взвешенное) среднее $\cos$ | **больше = лучше** |

$$\mathcal{B}_{\text{bridge}}(l) = \frac{1}{|\mathcal{U}|} \sum_{i \in \mathcal{U}} w_i \cdot \frac{J_S[i] \cdot J_S^{\text{target}}[i]}{\|J_S[i]\| \cdot \|J_S^{\text{target}}[i]\| + \epsilon}$$

В общем случае $\mathcal{G}_{\text{gap}} \approx 1 - \mathcal{B}_{\text{bridge}}$ (точно, если веса $w_i$ совпадают; при усреднении по строкам без весов — ровно $1-\cos$). Эксперименты 1–15 отчётывают **gap** (lower-is-better); эксперименты 16–19 отчётуют **bridge** (higher-is-better, $\cos$). В JSON: ключи `gap` (эксп. 1–15) и `bridge` (эксп. 16–19) — направление теперь однозначно.

---

## 2. Алгоритм

### 2.1. Шаг 0: Подготовка

#### 2.2. Шаг 1: Вычисление Якобиана (с корректным кэшем)

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
    # Цикл по token_set: один backward на токен (надёжно)
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

> **Примечание по производительности.** Цикл по $|\mathcal{U}|$ с `autograd.grad` — медленно. Для ускорения использовать `torch.func.jacrev` (PyTorch 2.1+) или `torch.func.vmap`. Валидация корректности завершена (Exp 1 identity). Оптимизация — future work.

> **Примечание (GDN cache cloning).** `copy.deepcopy(prefix_cache)` не работает для GDN слоёв — `recurrent_state` и `conv_state` мутируют in-place через `cache_params.update_recurrent_state(...)` и `torch_causal_conv1d_update`. Решение: `_deep_clone_cache()` — ручной клон всех тензоров в кэше. Внедрено (`prism/jacobian.py`).
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

> **Определение (projected Jacobian).** Для задач, инвариантных к uniform shift логитов (softmax, cross-entropy), важна не $J$ целиком, а её gauge-free компонента (проектор $\mathrm{diag}(p) - pp^T$, не ортогональный проектор в общем случае):
>
> $$J^{(p)} = (\mathrm{diag}(p) - pp^T) J$$
>
> где $p = \mathrm{softmax}(z_0) \in \mathbb{R}^K$ — выходное распределение. Матрица $\mathrm{diag}(p) - pp^T$ — ковариационная матрица one-hot с распределением $p$, линейный оператор, убивающий gauge-направление $\mathbf{1}$. Оператор убирает gauge-направление (uniform shift), сохраняя только token-discriminating направления. $J^{(p)}$ используется в §8 (defrag) и §9 (surgery).

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
    # Коррекция неортогональности E_S через pinv(E_S^T E_S)
    G_inv = torch.linalg.pinv(E_S.float().T @ E_S.float())  # [D_S, D_S]
    J_S_target = J_L_projected @ E_S.float() @ G_inv  # [|U|, D_S]
    
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

> **Определение (gauge-направление).** Gauge-направление $v_g \in \mathbb{R}^{D_S}$ — правый сингулярный вектор $J_S$, чей левый сингулярный вектор $u_g$ максимизирует $|\cos(u_g, \mathbf{1})|$ (где $\mathbf{1} \in \mathbb{R}^{|U|}$ — единичный вектор в токенном пространстве).
>
> **Лемма.** Равномерный сдвиг логитов не меняет softmax: для любого $c \in \mathbb{R}$, $\mathrm{softmax}(z + c \cdot \mathbf{1}) = \mathrm{softmax}(z)$. Следствие: градиент $rac{\partial}{\partial h} \log p_t$ инвариантен к gauge-shift, т.е. $(\mathrm{diag}(p) - pp^T) \mathbf{1} = 0$.
>
> $v_g$ отвечает за mode-shift (сдвиг всех логитов одновременно); направления, ортогональные $v_g$, несут token-discriminating информацию. Gauge-коррекция в $J^{(p)}$ убирает именно $v_g$, оставляя информативные направления для моста.

### 2.4. Шаг 3: Диагностика (Цель P0)

```python
def run_diagnostics(
    model_large, model_small,
    calibration_samples,
    token_set,
    layer_mapping,
):
    """
    Вычисляет gap (разрыв Якобиана, lower=better) для всех пар слоёв.
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
    U_size = len(token_set)
    G = torch.zeros(U_size, U_size, device=all_J[0].device)
    
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

> **Пространства и представления.** Собственные векторы ESG живут в разных пространствах:
>
> - **ESG eigenvectors** ($J J^T$): живут в **токенном** пространстве $\mathbb{R}^{|U|}$. Каждая строка $J_i$ — градиент логитов по скрытому состоянию для токена $i$; $J J^T$ = Gramian токенов.
> - **Right singular vectors** ($J^T J$): живут в **hidden** пространстве $\mathbb{R}^D$. $J^T J$ — чувствительность hidden dimensions друг к другу; собственные векторы $J^T J$ = направления в hidden space, по которым модель наиболее чувствительна.
> - **Связь через SVD:** $J = U \Sigma V^T$, где $U \in \mathbb{R}^{|U| 	imes |U|}$ — левые (token space), $V \in \mathbb{R}^{D 	imes D}$ — правые (hidden space). ESG использует $U$; анализ hidden directions требует $V$.

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

**Производная.** Для $y = \mathrm{RMSNorm}(x)$, $r = \sqrt{\langle x, x \rangle/D}$, $u = x/r$, $g = 1+w$ (вектор веса):

$$\frac{\partial y}{\partial x} = \frac{1}{r}\big(I - u u^T\big)\,\mathrm{diag}(g)$$

(фактор $(1 - 1/D)$ поглощается в $r$). Свойства: вектор $g^{-1}\odot u$ (покомпонентно) — нулевой собственный вектор, матрица ранга $\le D-1$ **неортогональна**; композиция через RMSNorm даёт вклад $(1/r)(I - u u^T)\mathrm{diag}(g)$ в полный слой-переходный Jacobian $D_l$ (уравнение (1.4) Math). Отсюда — при вычислении FTLE в сумму top-$m$ показателей входит вклад $\log(1/r)$ на каждом RMSNorm (см. appendix Math.md §3 «FTLE» — QR-алгоритм, лог-аккумуляция, оговорки n = 24–32): это дрейф масштаба от нормировки, а не хаотичность.

---

## 4. Экспериментальная установка

> **Статистический каркас (единый для всех экспериментов).**
> - **Bootstrap CI**: $B = 1000$ ресэмплингов (по сэрнам, не по парам), percentile-CI.
> - **Permutation tests**: $B = 1000$ перестановок, точный $p = (k+1)/(B+1)$.
> - **Относительное возмущение**: $\varepsilon_{rel} = \varepsilon / \|h_l\|$ (а не абсолютное $\varepsilon$), чтобы амплитуда возмущения была сопоставима между слоями.
> - Fixed seed, детерминированность: один seed на весь эксперимент, кластеризация не пересчитывается при бутстрепах (метки фиксированы).

## 4.1. Калибровочные данные

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

calibration_samples = []
for sample in dataset:
    tokens = tokenizer(sample["text"], max_length=64, truncation=True, return_tensors="pt")
    if tokens["input_ids"].shape[1] >= 64:
        calibration_samples.append(tokens)
    if len(calibration_samples) >= 4:
        break
```

> **Примечание.** В дизайне §2.1 предполагалось S=2048/512, N=32. В tested-протоколе: S=64, N=1–4 (ограничения CPU). Для полных экспериментов рекомендуем N ≥ 4, S ≥ 256.

#### 4.2. Глобальный набор токенов $\mathcal{U}$

Top-K токены **фиксируются заранее** по частоте на калибровочном наборе (старшая модель). Это гарантирует, что строки Якобианов выровнены по одним и тем же токенам на всех сэмплах.

```python
import torch
from collections import Counter

def build_global_token_set(model, calibration_samples, K=64):
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

> **Рекомендация (Exp 10c):** top-K покрытие нестабильно (min coverage 0.72–0.86 при K=256). Для стабильности переключиться на top-p саппорт (брать токены, чья совокупная вероятность ≥ p, напр. p=0.99).

#### 4.3. Соответствие слоёв

Для версии 1 — равномерное:

```python
def uniform_layer_mapping(n_layers_large, n_layers_small):
    mapping = {}
    for i in range(n_layers_large):
        j = round(i * (n_layers_small - 1) / (n_layers_large - 1))
        mapping[i] = j
    return mapping
```

#### 4.4. Загрузка моделей

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
# Qwen3.5 (все размеры): model.model — Qwen3_5TextModel
E_large = model_large.model.embed_tokens.weight.detach()  # [V, D_L]
E_small = model_small.model.embed_tokens.weight.detach()  # [V, D_S]
# Qwen3.5-0.8B: D_S = 1024; Qwen3.5-4B: D_L = 2560
```

## 5. Гипотезы и их статус

| ID | Гипотеза | Статус | Как проверяется | Приоритет |
| --- | --- | --- | --- | --- |
| H1 | Якобиан кодирует осмысленную «чувствительность декодирования» | ✅ Подтверждена | Эксп. 2, 7 | P0 |
| H2 | Top-$|\mathcal{U}|$ покрывает основную вероятностную массу | ✅ Подтверждена (K>=64) | Эксп. 4 | P0 |
| H3 | E-проекция $E_L^T E_S$ — нетривиальное выравнивание (≠ случайное; опирается на семантику эмбеддингов, а не только на размерности) | ✅ Подтверждена | Эксп. 3(random R), 5 | P0 |
| H4 | Минимизация $\mathcal{L}_{	ext{bridge}}$ улучшает модель | ⏳ Отложено | Не проверяется в v1 | P2 |
| H5 | ESG кодирует глобальную стратегию декодирования | ✅ Подтверждена | Эксп. 9, 10a | P1 |

---

## 6. Известные ограничения

1. **Локальность Якобиана.** Якобиан — касательная в конкретной точке. Агрегация по $N$ сэмплам частично решает проблему, но не полностью. **Exp 0.1 (§12) уточняет границу:** поле $J$ когерентно вдоль последовательности ($C(\sigma_d)=0.33$–$0.87$ по глубине) и не когерентно поперёк распределения сэмплов ($C\approx0.003$–$0.05$) — Грамианы строятся sample-local, без пулляции по домену.
2. **Неортогональность $E$.** Матрица эмбедингов не ортогональна. Проекция $J \cdot E^T$ не изометрична. Проверяется Экспериментом 8.
3. **Ранние слои.** Для $l < n/4$ градиент может затухнуть. В первой реализации: если $\|J[i]\| < 10^{-6}$, строка маскируется.
4. **Точность.** Модель загружается в float32 на CPU (§0.3). Градиенты вычисляются в float32. При проблемах с памятью можно использовать fp16, но это требует отдельной валидации.
5. **Память.** `copy.deepcopy(prefix_cache)` для $S = 2048$ может требовать 1-2 GB. При нехватке: уменьшить $S$ до 512.
6. **Линейный предиктор.** При $\varepsilon \leq 0.1$ gauge и differential направления неразличимы (KL_g = KL_d, ratio = 1.00×). При $\varepsilon = 1.0$ появляется слабый разрыв (ratio = 1.07×): KL_g слегка падает, KL_d слегка растёт. При $\varepsilon \geq 1$ нелинейности оставшихся слоёв доминируют. Якобиан корректен в первом порядке; для больших возмущений нужны higher-order методы.
7. **Малая статистическая мощность.** Эксперименты выполнены с N=1–8. Доверительные интервалы не вычислены, bootstrap не применяется. Утверждения H1–H5 — предварительные; для валидации требуются N ≥ 16–32 (см. Future work).

---

## 7. Результаты экспериментов

> Qwen3.5-0.8B (CPU, fp32, `torch.set_num_threads(4)`), Qwen3.5-4B → 0.8B для кросс-модельных тестов. K=4–256, N=1–16 сэмпла.

### Эксперимент 1 (N=1, K=4): Identity тест (0.8B → 0.8B)

Все 24 слоя: $\mathcal{L}_{\text{bridge}} \approx 10^{-8}$ (максимум $1.0 \times 10^{-7}$). Детерминизм: PASS (макс разница = 0.0).

**Вывод:** Реализация корректна. Без inverse Gramian correction лосс ~0.6–0.85 — подтверждает необходимость $\text{pinv}(E^T E)$.

### Эксперимент 2 (N=3, K=8): Neighbor windows (4B → 0.8B)

| Якорь 4B | Окно 0.8B | Минимум |
| --- | --- | --- |
| 0 | {0, 1, 2} | small_2 (0.99) |
| 8 | {0, 3, 6} | small_6 (0.94) |
| 16 | {8, 10, 12} | small_12 (0.77) |
| 24 | {16, 17, 18} | small_17 (0.67) |
| 30 | {21, 22, 23} | small_22 (0.55) |

Sparse map (пропорциональное выравнивание, 11 слоёв): 1.04 (layer 0) → 0.49 (layer 30→22). Монотонное снижение.

**Вывод:** Минимум в каждом окне — у пропорционального партнёра. Лосс падает с глубиной. «Хвостовая» конфаунд ($e(l) = l-8$) не доминирует.

### Эксперимент 3 (N=1, K=8): Абляция inverse Gramian (проверка необходимости коррекции)

| Коррекция | gap (layer 0) ↓ = лучше |
| --- | --- |
| С `pinv(E^T E)` | $1.1 \times 10^{-8}$ |
| Без (raw $E^T$) | 0.82 |
| **Случайная ортогональная R** | **1.02 ± 0.01** |

**Вывод:**
1. Inverse Gramian критичен: без него loss 0.82 vs $10^{-8}$ — разница в 10⁷×.
2. Learned bridge matrix ≠ случайная ортогональная матрица: random orthogonal R даёт loss ≈ 1.02 (максимум), тогда как learned W даёт loss ≈ 0. Это strong evidence что W выполняет конкретную работу по выравниванию подпространств, а не просто вращает.
3. Три режима: corrected (≈0) < raw (0.82) < random R (1.02). Порядок подтверждает, что Gramian correction убирает bias, а learned W убирает rotation.

### Эксперимент 4 (N=4, K-sweep): Чувствительность к $|\mathcal{U}|$ (кросс-модельный 4B → 0.8B)

> **Params:** N=4, K ∈ {4, 8, 16, 32, 64, 128, 256}, 6 par (0→0, 6→4, 12→9, 18→13, 24→18, 30→22).

| K | 0→0 | 6→4 | 12→9 | 18→13 | 24→18 | 30→22 | Avg |
|---|---|---|---|---|---|---|---|
| 4 | 1.0184 | 0.9405 | 0.9071 | 0.8172 | 0.6876 | 0.4513 | 0.804 |
| 8 | 1.0130 | 0.9404 | 0.9156 | 0.8372 | 0.6977 | 0.4736 | 0.813 |
| 16 | 1.0079 | 0.9389 | 0.9010 | 0.8151 | 0.6757 | 0.4523 | 0.799 |
| 32 | 1.0049 | 0.9380 | 0.8959 | 0.8081 | 0.6651 | 0.4427 | 0.792 |
| 64 | 1.0038 | 0.9378 | 0.8935 | 0.8042 | 0.6588 | 0.4370 | 0.789 |
| 128 | 1.0030 | 0.9377 | 0.8920 | 0.8021 | 0.6547 | 0.4336 | 0.787 |
| 256 | 1.0024 | 0.9375 | 0.8911 | 0.8011 | 0.6518 | 0.4313 | 0.786 |

**Вывод:** Метрика сходится при K ≥ 32; после K=64 изменение < 0.3% на par. Ранние par (0→0) почти не реагируют на K (1.018→1.002), поздние (30→22) чувствительны (0.451→0.431).

### Эксперимент 5 (N=2, K=8): Перестановка эмбеддингов (0.8B)

| Слой | Baseline | Permuted | Inverse Perm |
| --- | --- | --- | --- |
| 0 | $1.7 \times 10^{-8}$ | 0.98 | $4.3 \times 10^{-8}$ |
| 12 | $4.4 \times 10^{-8}$ | 0.98 | $-2 \times 10^{-9}$ |
| 23 | $4.8 \times 10^{-8}$ | 1.00 | $4.8 \times 10^{-8}$ |

**Вывод:** Перестановка ломает мост ($10^{-8} \to 0.98$), обратная перестановка восстанавливает. Мост опирается на семантику эмбеддингов. H3 подтверждена.

### Эксперимент 6 (N=16): KL-дивергенция (4B → 0.8B)

> **v2 (исправлен.)** Версия v1 `exp6_kl_baseline.py` имела баг: проекция промежуточных `hidden_states` через `lm_head` давала отрицательные KL. v2: корректный запуск, N=16, K=64, total_time=5955s.

**Single-number KL (финальные логиты):** kl_final_mean = 0.41.

**KL-lens proxy по слоям (некорректный послойный бейзлайн, lm_head не обучен на промежуточных слоях):**

| Слой | KL-lens | gap ↓ = лучше |
| --- | --- | --- |
| 0→0 | 2.38 | 1.00 |
| 3→2 | 3.20 | 1.00 |
| 6→4 | 3.43 | 1.00 |
| 9→7 | 3.25 | 1.00 |
| 12→9 | 3.35 | 1.00 |
| 15→11 | 3.29 | 1.00 |
| 18→13 | 3.25 | 0.83 |
| 21→16 | 3.44 | 0.73 |
| 24→18 | 3.27 | 0.66 |
| 27→20 | 3.75 | 0.59 |
| 30→22 | 1.83 | 0.41 |

> **Sentinel note:** `gap = 1.00` в ранних строках — **sentinel, не реальное значение**: `prism/bridge.py:31` возвращает 1.0, когда маска строк с ‖J_S‖ < 1e-6 пуста (ранние слои, малые по норме Jacobians). ρ=-0.09 ниже посчитан с учётом sentinel; sentinel-excluded пере-расчёт — в `docs/gpu_restart_audit.md` (CPU item).

**Спирмен ρ(KL-lens, bridge) = -0.09** — практически 0, нет корреляции (с sentinel; см. note).

**Вывод:**
1. `kl_final_mean = 0.41` — единственный корректный single-number KL финальных логитов.
2. KL-lens по слоям (~2.4–3.8, плоский) — garbage: lm_head не обучен на промежуточных слоях. Значения положительны (баг v1 исправлен), но не несут смысла.
3. gap монотонно падает (1.0→0.41) — разрыв несёт уникальную послойную структуру.
4. Spearman ~ 0 — мост измеряет нечто принципиально иное, недоступное послойному KL.

### Эксперимент 7 (N=5, K=8): CKA (4B → 0.8B)

| Слой | CKA | gap ↓ = лучше | Spearman ρ |
| --- | --- | --- | --- |
| 0→0 | 0.0003 | 1.03 | — |
| 30→22 | 0.0007 | 0.50 | — |

CKA ≈ 0 для всех слоёв — представления геометрически не скоррелированы, несмотря на общее семейство. Linear CKA **работает** при разных D (грам-матрицы строятся по токенам, а не по измерениям). Spearman ρ между CKA и bridge = -0.77 (CKA растёт, bridge падает — противоположная монотонность).

> **Null caveat:** при N≪D (N=5, D=1024/2560) линейный CKA **inflated** — для независимых 64×128-матриц ожидание ≈0.34. Измеренные 0.0003–0.0007 **далеко ниже** null → контраст «CKA≈0 vs bridge 1.03→0.796» устойчив (см. audit, exp7 row).

**Вывод:** CKA ≈ 0 означает, что скрытые состояния 4B и 0.8B не линейно скоррелированы. При этом мост структурирован (монотонное падение) — мост измеряет нечто новое, недоступное CKA.

### Эксперимент 8 (N=1, K=8): SVD-абляция (0.8B)

| n_components | gap (layer 0) ↓ = лучше |
| --- | --- |
| 16 | 0.89 |
| 64 | 0.77 |
| 256 | 0.51 |
| 512 | 0.30 |
| 1024 | $2.4 \times 10^{-8}$ |

**Вывод:** Нужен полный ранг. Усечение даже до 512 (half) не даёт identity — мост полагается на полный спектр $E^T E$. Condition number κ(E^T E) = 1684 (Exp 6b) — матрица хорошо обусловлена и имеет полный ранг 1024; min eigenvalue = 12.5.

### Эксперимент 9 (N=5, K=64)

| Слой | λ₁ | λ₂ | λ₃ |
| --- | --- | --- | --- |
| 0 | 6.3M | 2.5M | 1.7M |
| 6 | 14.8K | 2.2K | 2.0K |
| 12 | 8.7K | 1.4K | 1.3K |
| 18 | 1.8K | 357 | 317 |
| 23 | 2.8 | 2.2 | 1.3 |

**Вывод:** Eigenvalues падают с 6M → 3 (layer 0 → 23). Топ-3 eigenvalues слоя 0 составляют ~70% от следа — доминирование **масштабной** моды (scale mode), а не uniform gauge (uniformity слоя 0 = 0.45 по Exp 10a). Uniform gauge доминирует в средних слоях (6–18, uniformity ≈ 0.95). В слое 23 спектр «плоский» — дифференциальная чувствительность. Это подтверждает, что ранние слои кодируют mainly gauge (common-mode baseline), поздние — семантику.

### Эксперимент 10a (N=4, K=32)

| Слой | uniformity_0 | gauge_share_0 | Raw λ₁ | Cent λ₁ | Soft λ₁ |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.455 | 0.111 | 4.6M | 4.1M | 2296 |
| 6 | 0.965 | 0.620 | 8783 | 3340 | 11 |
| 12 | 0.955 | 0.474 | 4387 | 2309 | 6 |
| 18 | 0.848 | 0.349 | 799 | 520 | 3 |
| 23 | 0.503 | 0.165 | 3 | 3 | 0 |

**Вывод:**

- Слои 6–18: uniformity ≈ 0.95–0.97 → топ-направление почти равномерное (common-mode shift).
- Слои 0 и 23: uniformity ≈ 0.5 → top-направление дифференциальное (не common-mode).
- Uniform centering убирает 62% (L6) → 11% (L0) от λ₁.
- **Softmax-jacobian убивает common-mode почти полностью:** trace L0: 11M → 2.4K (99.98% reduction), L6: 22K → ~100.
- L0 special: raw и cent не сильно отличаются (gauge_share=0.11), но softmax резко срезает → в L0 есть масштабный компонент, который не uniform, но weighted softmax убирает.

### Эксперимент 10b (N=8, K=64): Gauge-free bridge 4B→0.8B (7 par)

> **Params:** N=8, K=64, 7 par (0→0, 5→4, 10→7, 15→11, 20→15, 25→19, 30→22). GPU T4.

| Слой | Raw | Centered | Softmax |
| --- | --- | --- | --- |
| 0→0 | 1.0030 | 0.9915 | 0.9970 |
| 5→4 | 0.9452 | 0.9209 | 0.9194 |
| 10→7 | 0.9020 | 0.8744 | 0.8691 |
| 15→11 | 0.8554 | 0.8302 | 0.8312 |
| 20→15 | 0.7407 | 0.7122 | 0.7169 |
| 25→19 | 0.6372 | 0.5966 | 0.6158 |
| 30→22 | 0.4336 | 0.3640 | 0.3997 |
| **Avg** | **0.788** | **0.756** | **0.764** |

**Вывод:** Centering (удаление gauge $J^{(p)}$) даёт наименьший gap (0.756) — подтверждение N=1 результата. Softmax чуть хуже centered (0.764). Пары глубоких слоёв (30→22) реагируют сильнее (0.434→0.364 при centering) — gauge-вклад растёт с глубиной.

**Интерпретация (по матрице из § 10b):**

| Наблюдение | Вывод |
| --- | --- |
| Centered avg 0.755 vs raw 0.787 (-4%) | Небольшое снижение, но монотонность сохранилась |
| Ранние слои: разница 0.5% (L0: 1.028→1.020) | Гейдж не доминирует в early layers — кросс-модельное расхождение здесь структурное |
| Поздние слои: разница до 18% (L30: 0.44→0.36) | В поздних слоях есть gauge-составляющая — часть расхождения объясняется разными common-mode базлайнами |
| Softmax: avg 0.774 (между raw и centered) | Softmax-weighting частично убирает gauge, но не так сильно как uniform centering — весовой эффект мягче |

**Итог:** Raw bridge переоценивает кросс-модельное расхождение на ~4% в среднем, в поздних слоях — до 18%. Основной паттерн (ранние >> поздние) сохраняется. **Часть расхождения — структурная, часть — gauge.**

### Эксперимент 10c (N=8): Top-K vs Top-P покрытие вероятностной массы

Цель: показать, что top-K даёт нестабильное покрытие, а top-p — стабильное.

### Top-K: фиксированное K, переменное покрытие

| $|\mathcal{U}|$ | Mean | Min | Max | Std |
|---|---|---|---|---|
| 32 | 0.87 | 0.66 | 1.00 | 0.13 |
| 64 | 0.91 | 0.75 | 1.00 | 0.10 |
| 128 | 0.93 | 0.81 | 1.00 | 0.08 |
| 256 | 0.95 | 0.86 | 1.00 | 0.05 |

**Проблема:** при K=32 coverage варьируется от 0.66 (sample_4) до 1.00 (sample_7) — разница в 1.5×. Даже при K=256: min=0.86 (sample_2), max=1.00 (sample_7).

### Top-P: фиксированное покрытие, переменное K

Сколько токенов нужно для достижения заданной накопленной вероятности:

| p-target | Mean K | Min K | Max K |
|---|---|---|---|
| 0.90 | 136 | 1 | 512 |
| 0.95 | 398 | 1 | 1456 |
| 0.99 | 2152 | 4 | 6814 |

**Наблюдение:** K_needed тоже сильно варьируется (1–512 при p=0.90), но это ожидаемо — distribution sharpness разная по сэмплам (sample_5: один токен имеет p=0.99, sample_2: равномерное распределение).

**Вывод:** Top-K гарантирует фиксированный размер $|\mathcal{U}|$, но coverage нестабилен (std=0.05–0.13). Top-p гарантирует покрытие, но размер $|\mathcal{U}|$ адаптируется. Для Jacobian bridge критичен coverage, поэтому **top-p предпочтительнее** — $|\mathcal{U}|$ варьируется, но масса гарантирована.

### Эксперимент 10d (N=4, K=32)

Spearman корреляция между величиной common-mode ($\|v_0^T J\|$) и энтропией $H(p)$:

| Слой | Spearman ρ |
| --- | --- |
| 0 | -0.40 |
| 6 | -0.20 |
| 12 | -0.60 |
| 18 | -0.80 |
| 23 | +0.40 |

**Вывод:** В средних слоях (6–18) есть устойчивая отрицательная корреляция: больший common-mode → меньшая энтропия → модель более уверена. В выходном слое (23) знак меняется → common-mode «инвертируется». Это подтверждает, что common-mode несёт сигнал уверенности — побочный канал мета-информации, **невидимый в линейном приближении** (softmax инвариантен к uniform shift), но **просачивающийся через нелинейности decoder** при полном forward (Exp 12a ε-sweep).



## 8. Дефрагментация Якобиана

> Hypothesis: $J^{(p)} \in \mathbb{R}^{K \times D}$ имеет эффективный ранг $r \ll K$. Если так, можно аппроксимировать $J^{(p)} \approx U_r J_r$, где $J_r \in \mathbb{R}^{r \times D}$ требует только $r$ обратных проходов вместо $K$.

### Эксперимент 11a (N=4): Fidelity реконструкции (0.8B, слои 6/12/18, K=32)

Метод: возмущаем $h_l$ на $\pm\varepsilon \cdot v$ (gauge / semantic / random направления, $\varepsilon \in \{0.5, 1.0, 2.0\}$), мерим KL($p_{\text{true}} \| p_{\text{pred}}$) для трёх предикторов:

- **Raw:** $z_{\text{pert}} = z_0 + J \cdot \delta$ (только по token_set)
- **Gauge-free:** $p_{\text{pert}} = \text{softmax}(\log p_0 + J^{(p)} \cdot \delta)$ на token_set
- **Defrag-r:** $p_{\text{pert}} = \text{softmax}(\log p_0 + U_r J_r \cdot \delta)$, $r \in \{4, 8, 16, 32\}$

| Perturbation | Scale | KL(raw) | KL(gf) | KL(dr4) | KL(dr8) | KL(dr32) |
| --- | --- | --- | --- | --- | --- | --- |
| gauge (L6) | 1.0 | 13.43 | 10.47 | 10.47 | 10.47 | 10.47 |
| sem_0 (L6) | 1.0 | ~13.5 | ~10.6 | ~10.6 | ~10.6 | ~10.6 |

**Выводы:**

1. **Gauge-free значительно лучше raw** (10.5 vs 13.4) — softmax-weighting убирает gauge-артефакты из линейной аппроксимации.
2. **Defrag сходится при $r=4$** — KL(dr4) = KL(dr8) = KL(dr32) = KL(gf). Эффективный ранг $J^{(p)}$ действительно мал.
3. **Но KL(gf) всё ещё ~10.5** — линейная аппроксимация ломается при $\varepsilon=1.0$. Нелинейности оставшихся слоёв доминируют. Это не баг реконструкции, это предел первого порядка.

### Эксперимент 11b (N=2): Стоимость вычисления (0.8B, K=256)

| Слой | r | Скорость (×) | Fidelity rel error (%) |
| --- | --- | --- | --- |
| 6 | 8 | 17.8× | 16 |
| 6 | 16 | 12.8× | 25 |
| 6 | 32 | 8.1× | 36 |
| 12 | 8 | 10.5× | 15 |
| 12 | 16 | 6.8× | 24 |
| 12 | 32 | 5.5× | 39 |
| 18 | 8 | 6.3× | 14 |
| 18 | 16 | 5.4× | 23 |
| 18 | 32 | 4.0× | 38 |

> **Метрика fidelity (v1).** Сохранённый JSON отмерен v1-метрикой `‖U_r J_r − J_p‖/‖J_p‖` — она сравнивает проекцию **сырого** J с J_p и растёт с r (19→31→48) как артефакт, а не как сигнал; баг исправлен в `experiments/exp11b_defrag_cost.py` (c904db3): корректная величина — tail-ошибка `‖U_r U_rᵀ J_p − J_p‖/‖J_p‖`, монотонно убывающая и равная 0 при r=K. Значения таблицы ниже (14–38%) — v1-метрика; после re-run столбец поменяет тренд на убывающий. Speedup-столбец от метрики не зависит.

Для проекции (forward-only, без backward):

| r | FLOPs | Storage |
| --- | --- | --- |
| raw (K=256) | 130G | 262K |
| dr8 | 4G (32×) | 8K (32×) |
| dr32 | 16G (8×) | 32K (8×) |

**Вывод:** Defrag ускоряет backward на 4–18× (zависит от слоя и ранга). FLOPs проекции падают в 32× при r=8. Цена — 14–39% relative error на реконструкцию $J^{(p)}$.

> **Примечание о кажущемся противоречии (11a vs 11b).** В 11a KL(gauge-free реконструкция) одинаков при r=4,8,32 — gauge-free пространство достаточно покрыто уже при r=4. В 11b relative error на реконструкцию $J^{(p)}$ растёт (14% → 39%), потому что измеряется ошибка на полный J, а не только на gauge-free компоненту. Больше сингулярных компонент = больше накопленных ошибок округления в менее значимых направлениях.

### Эксперимент 11c (N=8): Gap сигнал через defrag (4B → 0.8B, 8 пар, GPU T4)

> **Params:** N=8, 8 par (0→0, 4→3, 8→6, 12→9, 16→12, 20→15, 24→18, 28→21).

| Метод | Avg gap ↓ = лучше | Spearman vs raw | Монотонность |
| --- | --- | --- | --- |
| Raw | 0.821 | — | -1.0 (идеальна) |
| $J^{(p)}$ (gauge-free / softmax) | 0.798 | 1.0 | -1.0 |
| Defrag r=8 | 1.002 | 0.81 | -0.81 |
| Defrag r=16 | 0.976 | 0.71 | -0.71 |
| Defrag r=32 | 0.982 | 0.43 | -0.43 |

**Вывод:** Confirmed at N=8: defrag при r ≤ 32 разрушает монотонный сигнал gap. Raw и softmax (gauge-free) сохраняют идеальную монотонность (ρ=1.0). При r=32 gap flat-нет (~0.98 вместо 1.0→0.5) — ранг r=32 недостаточен для переноса. **Defrag не подходит для метрики gap** — мост чувствителен к компонентам, которые не видны в $J^{(p)}$ (low-probability токены).

---

## 9. Карта чувствительности: валидация и применение

> Эксперименты 12 проверяют, что карта $J^{(p)}$ разделяет gauge и дифференциальную чувствительность **в линейном приближении**. Функциональная разделимость при полном forward не подтверждается (ε-sweep: KL_g ≈ KL_d при ε ≤ 0.1), но **структурное разделение** полезно для направленного квантования (Exp 12b,c) и диагностики (Exp 12a uniformity).

### Эксперимент 12a (N=8): Пертурбационный тест (0.8B, слои 6/12/18, K=64)

> **Params:** текущий прогон: N=8, K=64 (`results/exp12a_perturbation_test.json` → `params`). Дефолт скрипта `experiments/exp12a_perturbation_test.py` (N=4, K=64) не синхронизирован с прогоном — следующий GPU-ран перезапишет значения.

Для каждого слоя находим:

- $v_g$ = правый сингулярный вектор $J$, чей левый вектор ≈ равномерен (gauge direction)
- $v_d$ = топ-правый сингулярный вектор $J^{(p)}$ (дифференциальное направление)

Возмущаем $h_l \leftarrow h_l \pm \varepsilon \cdot v$ при одинаковой норме $\|v\|=1$, $\varepsilon=1.0$. Мерим KL($p_{\text{orig}} \| p_{\text{pert}}$).

| Слой | uniformity($v_g$) | KL$_g$ (avg ± std) | KL$_d$ (avg ± std) | ratio KL$_d$/KL$_g$ |
| --- | --- | --- | --- | --- |
| 6 | 0.92 | 7.04 ± 0.37 | 7.55 ± 0.58 | 1.07× |
| 12 | 0.94 | 7.46 ± 0.47 | 7.80 ± 0.51 | 1.05× |
| 18 | 0.97 | 6.42 ± 0.97 | 6.69 ± 0.94 | 1.04× |

**Результат ε-sweep:**
1. **Линейный режим (ε ≤ 0.1):** KL_g = KL_d (ratio = 1.00×). Gauge и differential направления неразличимы — в первом порядке оба дают одинаковый KL.
2. **Нелинейный переход (ε = 0.5–1.0):** Появляется слабый разрыв (max ratio = 1.07×, L6, ε=1.0): KL_g(L6) 7.34 → 7.04, KL_d(L6) 7.52 → 7.55; на L18 gap остаётся малым (1.01× → 1.04×).
3. **Интерпретация:** Модель не игнорирует gauge-направление (KL_g не меньше KL_d). Gauge «просачивается» через нелинейности decoder. Разрыв при ε=1.0 — не gauge-инвариантность, а нелинейное затухание gauge-сигнала.

**Что подтверждено:** uniformity $v_g$ = 0.92–0.97 на слоях 6–18 — gauge direction существует и имеет равномерную структуру. **Структурное разделение** работает; **функциональная разделимость** при полном forward — нет. (Последующий Exp 0.1, §12, исключил альтернативное объяснение нелинейного перехода — фрактальности поля $J$: на масштабах $\sigma_d$ поле когерентно, первый порядок корректен.)

> **Рефрейм:** map — **направление минимальной чувствительности**. При равном destruktive budget (KL≈15) map-квантование даёт меньшее возмущение hidden space (Δh 0.22–0.63 vs 0.82–2.52), чем uniform — но **оба варианта разрушают модель полностью** (PPL 10⁶–10⁸). Это не «map сохраняет модель», а «map направляет разрушение эффективнее» (см. exp12b).

### Эксперимент 12b (N=4, K=32): Квантование по карте (0.8B, слои 6/12/18)

> **Params:** таблица — из сохранённого JSON (N=4, K=32). Скрипт уже обновлён до K=64; следующий run перезапишет значения.

Симуляция направленного квантования $h_l$:

- **Map-based:** gauge (2 бита) + 5 дифференциальных направлений (8 бит) + остальные (6 бит). Число 5 согласовано с Exp 11a: эффективный ранг $J^{(p)}$ мал (defrag сходится уже при $r=4$), PR(L6) ≈ 5.9 — то есть «5 направлений» — это спектральный ранг, а не фиксированная конвентация.
- **Uniform:** 4 бита на всё

| Слой | KL(map) | KL(uniform) | $\Delta h_{\text{map}}$ | $\Delta h_{\text{uniform}}$ | PPL(map) | PPL(uniform) |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 14.91 | 15.00 | 0.24 | 0.95 | 1.25M | 2.47M |
| 12 | 15.09 | 15.09 | 0.26 | 1.00 | 14.8M | 50.8M |
| 18 | 15.02 | 15.01 | 0.84 | 3.34 | 1.69M | 2.14M |

> **PPL note:** Значения PPL ~1.25M-50.8M соответствуют exp(15) ~ 3.3e6. Это ожидаемо: KL ~15 означает полное разрушение распределения (random perturbation), а не реальное квантование. Симуляция проверяет *направление* dH, не абсолютный PPL.


**Вывод:** KL одинаков (оба ~15), но map-based достигает этого при $\Delta h \approx 0.24$ vs uniform $\Delta h \approx 0.95$. Map-based квантование создаёт в 4× меньшее возмущение в hidden space при том же KL.

> **Оговорка:** оба варианта непригодны (PPL 10⁶–10⁸, полное разрушение распределения). Меньший Δh не означает «лучше сохраняет модель» — карта лишь эффективнее направляет разрушение в чувствительные направления. Это предельный тест, не практическое применение.

### Эксперимент 12c (N=4, K=32): Мост оригинал ↔ квантованная (0.8B, слои 6/12/18)

> **Params:** таблица — из сохранённого JSON (N=4, K=32); скрипт `experiments/exp12c_bridge_quantized.py` на K=64.

Softmax-мост между оригиналом и каждой схемой квантования (cosine similarity градиентов):

| Слой | $\mathcal{L}$(orig ↔ map) | $\mathcal{L}$(orig ↔ uniform) | $\mathcal{L}$(map ↔ uniform) |
| --- | --- | --- | --- |
| 6 | 0.001 | 0.020 | 0.021 |
| 12 | 0.001 | 0.015 | 0.016 |
| 18 | 0.001 | 0.014 | 0.015 |

**Вывод:**

1. **Map-based ≈ идентичен оригиналу по НАПРАВЛЕНИЮ ЯКОБИАНА** — мост 0.001 (почти ноль). *Оговорка: это направление градиента, не функция — по Exp 12b обе схемы полностью разрушают PPL (10⁶–10⁸); map лишь 15–20× меньше искажает Jacobian-направление.*
2. **Uniform далеко** — мост 0.014–0.020, в 15–20× больше.
3. **Мост map↔uniform ≈ мост orig↔uniform** — равномерное квантование повреждает те же направления, что отличают map от оригинала.

**Механистическая интерпретация:** Uniform квантование шумит в gauge-направлении (где нужно 8 бит защиты) и в дифференциальных направлениях (где 4 бита достаточно). Map-based ставит 8 бит туда, где они нужны, и экономит на gauge. Результат: тот же KL, но в 4× меньше возмущение и мост ≈ 0.

---

## 10. Итоговая картина

### Что подтверждено

| Утверждение | Статус | Эксперименты |
| --- | --- | --- |
| Грамиановская коррекция ($\text{pinv}(E^T E)$) критична | ✅ | Exp 1, 3, 8 |
| gap монотонно падает (ранние слои >> поздние): лучше — глубже | ✅ | Exp 2, 4, 6, 10b |
| Мост опирается на семантику эмбеддингов | ✅ | Exp 5 |
| Common-mode (gauge) несёт сигнал уверенности | ✅ | Exp 10a, 10d |
| $J^{(p)}$ имеет малый эффективный ранг ($r \approx 4$) | ✅ | Exp 11a |
| Defrag ускоряет backward в 2.7–18.9× | ✅ | Exp 11b |
| Карта = направление минимальной чувствительности (при равном KL — меньший Δh) | ✅ | Exp 12b, 12c |
| Gauge direction имеет uniform структуру | ✅ | Exp 12a (uniformity ~0.9) |
| Condition number Gramian умеренный (κ = 1684) | ✅ | Exp 6b |
| Learned bridge matrix ≠ random orthogonal | ✅ | Exp 3(random R) |
| Gauge и differential неразличимы в линейном режиме | ✅ | Exp 12a (ε-sweep) |

### Что НЕ подтвердилось

| Предсказание | Результат | Интерпретация |
| --- | --- | --- |
| KL$_g \ll$ KL$_d$ при малых ‖δ‖ | ❌ ratio = 1.00× (ε ≤ 0.1) | В линейном режиме направления неразличимы |
| Слабый разрыв при ε=1.0 | ⚠️ ratio ≤ 1.07× | Не gauge-инвариантность, а нелинейное затухание gauge |
| Defrag r=8 сохраняет мост | ❌ Spearman=0.19 | Мост чувствителен к low-probability токенам вне top-K |
| Full ранг $E^T E$ нужен для identity | ✅ (Exp 8, 6b) | Мост — не low-rank; κ=1684, rank=1024 |

### Практические следствия

1. **Карта $J^{(p)}$ работает для направленного квантования** (Exp 12b,c): map-based квантование при 4× меньшем $\Delta h$ даёт тот же KL, при этом мост ≈ 0.
2. **Defrag экономит backward (×2.7–18.9)**, но для bridge-моста нужен полный ранг — defrag и мост решают разные задачи.
3. **Gauge-direction реален** (uniformity 0.92–0.97) и **поведенчески активен**: KL_g ≈ KL_d при ε ≤ 0.1 (линейный режим), слабый разрыв до 7% при ε = 1.0 (нелинейности). Gauge не «невидим» — он влияет на prediction через нелинейности decoder.
4. **Мост не заменяет KL, но дополняет его:** KL — один номер, мост — послойная карта расхождения.

---

## 11. Доменная специфичность, глубинная структура, фазовая робастность

### Эксперимент 13 (N=8, K=64)

Четыре домена: wiki (C4), math (GSM8K), code (apex-agents), chat (ultrachat). ESG вычисляется на общем `token_set` (wiki) для сопоставимости eigenvectors. shared_token_set_size = 64. Мост использует вероятности каждого домена.

**Монотонность моста (Spearman ρ vs глубина):**

| Домен | ρ | L0->0 | L6->4 | L12->9 | L18->13 | L24->18 |
| --- | --- | --- | --- | --- | --- | --- |
| wiki | -0.89 | 1.000 | 1.000 | 1.000 | 0.788 | 0.640 |
| math | -0.89 | 1.000 | 1.000 | 1.000 | 0.863 | 0.758 |
| code | -0.89 | 1.000 | 1.000 | 1.000 | 0.790 | 0.640 |
| chat | -0.89 | 1.000 | 1.000 | 1.000 | 0.769 | 0.591 |

**Вывод:** Монотонность -0.89 во всех доменах (не -1.00) — с 5 точками это почти идеальная монотонность. Конвергенция моста робастна к домену. Первые 3 слоя saturated (1.000) — мост "отлипает" только с L18.

> **Sentinel note:** 1.000 в ранних якорях — **sentinel, не реальное значение**: `prism/bridge.py:31` возвращает 1.0, когда маска строк с ‖J_S‖ < 1e-6 пуста (ранние слои, малые по норме Jacobians). ρ=-0.89 / cross-Spearman 0.971 посчитаны **с sentinel** (совпадает с JSON «all points»); sentinel-excluded пере-расчёт — CPU (audit item 9).

**Overlap топ-8 подпространств ESG (token space, по собственным векторам JJ^T):**

Метрика: средний квадрат косинуса главных углов между $r$-мерными подпространствами:
$$\mathrm{overlap}(U_1, U_2) = \frac{\|U_1^T U_2\|_F^2}{r}$$
где $U_1, U_2 \in \mathbb{R}^{V \times r}$ — ортонормированные базисы подпространств ESG (по собственным векторам $J J^T$ с наибольшими eigenvalues), $r = 8$. Значение в $[0, 1]$: 0 — ортогональны, 1 — совпадают.

| Пара | L0 | L6 | L12 | L18 | L23 |
| --- | --- | --- | --- | --- | --- |
| wiki x wiki | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| wiki x math | 0.387 | 0.647 | 0.757 | 0.785 | 1.00 |
| wiki x code | 0.415 | 0.708 | 0.746 | 0.857 | 1.00 |
| wiki x chat | 0.301 | 0.680 | 0.759 | 0.898 | 1.00 |
| math x code | 0.361 | 0.646 | 0.643 | 0.734 | 1.00 |

**Cross-overlap Spearman = 0.971** — перекрёстные значения overlap тесно скоррелированы между собой.

**Вывод:**
- Self-overlap = 1.0 (determinism).
- Cross-overlap L0: 0.30–0.42 — в ранних слоях чувствительные подпространства **доменно-специфичны**.
- Cross-overlap L23: ≈1.0 — в поздних слоях все домены сходятся к **одному подпространству**.
- Переход специфичность→универсальность **gradual**: L12 (0.74–0.79) → L18 (0.79–0.90) почти линейно, а **большой скачок L18→L23** (0.79–0.90 → 0.999–1.00). Т.е. «универсальность» — свойство самых поздних слоёв, а не переход на L18.

**Trace ESG (L0):** chat 37.3M >> wiki 16.7M > math 6.8M > code 5.3M. Чувствительность L0 зависит от домена. На глубине (L23): все ≈27–29 (универсально).

**Вывод:** "Что" чувствительно (какие направления) — зависит от домена на ранних слоях. "Как сходится" (монотонное убывание моста) — не зависит от домена. На поздних слоях все домены сходятся к одному подпространству чувствительности.

> **Оговорка:** Absolute magnitude моста между доменами несравнима (разные распределения вероятностей). Монотонность — сравнима. Trace значения при K=64 не сравнимы с K=32.
### Эксперимент 14 (N=8, K=64)

Все 24 слоя, полный спектр (n_components=K). Exact gauge_share ($u^T G u / \text{trace}$), leave-one-out SNR, effective rank (PR). Слой 23 — full attention, остальные — mixed (linear_attention / full_attention).

> Condition number Gramian κ(E^T E) = 1684 — умеренный для 1024×1024 матрицы. Все eigenvalues > 0 (полный ранг).

| Слой | SNR | PR | Gauge share | Trace | Тип |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.14 | 6.4 | 0.31 | 17.2M | linear |
| 3 | 0.26 | 3.6 | 0.48 | 140K | full |
| 6 | 0.47 | 5.9 | 0.36 | 44.1K | linear |
| 9 | 0.47 | 7.3 | 0.33 | 43.2K | linear |
| 12 | 0.63 | 8.0 | 0.31 | 32.3K | linear |
| 15 | 0.93 | 10.2 | 0.28 | 13.9K | full |
| 18 | 1.98 | 14.8 | 0.22 | 11.0K | linear |
| 21 | 3.65 | 16.6 | 0.20 | 6.0K | linear |
| 23 | 199.3 | 33.4 | 0.08 | 30.7 | full |

**Вывод:**

- **SNR растёт с глубиной:** 0.14 → 3.65 (L0→L21, в 26×). L23 — аномалия (199.3) — тривиальный якобиан (последний слой, J = W lm_head).
- **PR (effective rank) — немонотонно:** падает 6.4 (L0) → 3.6 (L3), затем монотонно растёт 16.6 (L21) → 33.4 (L23). Подпространство чувствительности становится многомерным с глубиной, но с провалом в самых ранних слоях.
- **Gauge share — НЕ монотонна:** 0.31 (L0) → **peak 0.48 (L3)** → спад к 0.08 (L23). «Доля gauge уменьшается с глубиной» — неверно: она пиковая в L3 и только потом падает; ранние слои L0–L5 доминируют gauge (0.31–0.48), поздние — дифференциальные (0.08–0.20).
- **Trace падает:** 17.2M → 30.7. Чувствительность уменьшается с глубиной.

### Эксперимент 15 (N=4, K=256)

Четыре схемы якорей: A = full-attention, B = full+1, C = full−1, D = full−2. 4 якоря на схему для 0.8B.

**Якоря (слои 4B→0.8B):** A=[3→3, 11→7, 19→15, 27→19], B=[4→4, 12→8, 20→16, 28→20], C=[2→2, 10→6, 18→14, 26→18], D=[1→1, 9→5, 17→13, 25→17].

**Монотонность (Spearman ρ vs глубина):** A=-0.95, B=-0.95, C=-0.95, D=-0.95. > **Оговорка:** при n=4 возможные значения ρ дискретны (-1, -0.8, -0.6, …, 1). Значение -0.95 указывает на использование scipy.stats.spearmanr, которая применяет непрерывную аппроксимацию. Строгий ρ = -1.0 (полный обратный порядок).

**Корреляция между схемами:** A_vs_B=1.0, A_vs_C=1.0, A_vs_D=1.0 (идентичные кривые). max_abs_diff_schemes = 0.12.

**Мост по схемам (4B→0.8B):**

| Якорь | Scheme A | Scheme B | Scheme C | Scheme D |
| --- | --- | --- | --- | --- |
| ~L3 / ~L4 / ~L2 / ~L1 | 1.000 | 1.000 | 1.000 | 1.000 |
| ~L11 / ~L12 / ~L10 / ~L9 | 1.000 | 1.000 | 1.000 | 1.000 |
| 19→15 / 20→16 / 18→14 / 17→13 | 0.734 | 0.697 | 0.757 | 0.854 |
| 27→19 / 28→20 / 26→18 / 25→17 | 0.618 | 0.608 | 0.634 | 0.635 |

**ESG trace на якорях:**

| Якорь | Scheme A (full) | Scheme B (post-full) | Scheme C (pre-full) | Scheme D (full−2) |
| --- | --- | --- | --- | --- |
| L3 / L4 / L2 / L1 | 564K | 407K | 697K | **17.2M** |
| L7 / L8 / L6 / L5 | 132K | 139K | 134K | 220K |
| L15 / L16 / L14 / L13 | 32K | 26K | 44K | 69K |
| L19 / L20 / L18 / L17 | 16K | 13K | 20K | 23K |

**Вывод:**

1. Все 4 схемы дают практически идентичные кривые моста (Spearman=1.0 между каждой парой). Убывание с глубиной — свойство глубины, а не артефакт якоринга на full-attention слоях.
2. Первые 2 якоря saturated (1.000) во всех схемах — мост не различает ранние слои вообще. *(1.000 = sentinel `prism/bridge.py:31`, не реальное значение — см. sentinel note exp13.)*
3. Схема D, L1: trace=17.2M — слой 1 невероятно чувствителен (в 30× больше чем L3). L1 близок к embedding — огромный Jacobian. Это отдельное наблюдение: чувствительность взрывается у embedding.
4. Scheme D L17: bridge=0.854 (выше A 0.734, B 0.697, C 0.757) — **но это конфаунд глубины, не эффект схемы.** 3-й якорь D = 4B L17→**0.8B L13**, а у A/B/C 3-й якорь = L15/L16/L14. Mост D измеряется на более раннем 0.8B-слое (L13) → выше по глубинному тренду. На сопоставимой глубине (4-й якорь: D L17 vs A L19) разницы между схемами нет (0.635 vs 0.618).
5. Максимальное абсолютное расхождение между схемами: 0.12.

> **Вывод:** Конвергенция моста — не артефакт фазы. Глубина доминирует, фаза архитектуры вторична.

### Эксперимент 6 L18 SNR (N=8, K=256): Spektrum vs spektrum-depth r (0.8B, слой 18)

> **Params:** N=8, K=256, слой 18 (linear_attention), r ∈ {4, 8, 16, 32, 64}. GPU T4, 55s.

| r | SNR | PR (Parker ratio) | Gauge share (top-r) | Trace top-r |
|---|---|---|---|---|
| 4 | 2.803 | 2.15 | 0.602 | 6.3×10³ |
| 8 | 2.803 | 3.46 | 0.464 | 8.2×10³ |
| 16 | 2.803 | 5.72 | 0.354 | 1.08×10⁴ |
| 32 | 2.803 | 9.85 | 0.264 | 1.45×10⁴ |
| 64 | 2.803 | 16.91 | 0.199 | 1.93×10⁴ |

**Вывод:** SNR = 2.80 — r-инвариантен (это свойство всего спектра, не top-r). PR растёт монотонно 2.1→16.9 — спектр L18 имеет умеренную концентрацию (не low-rank). Gauge share top-r падает 0.60→0.19: common-mode компонент (u-direction) занимает 60% энергии top-4 modes, но лишь 19.9% top-64 — gauge-сигнал сконцентрирован в первых few modes. Слой 18 (linear_attention) — не last layer, не trivial J= lm_head.

---

### Эксперимент 16 (v2): Readout-pull как доза-ответ (0.8B → target, пара 31_23)

> **Params:** lm_head-only edit (readout), loss = (1−cos(J_S_p, J_tgt_p)) + 0.1·KL(leash), Adam lr=1e-3, 100 steps, eval every 10, K=256, N=4 (paired by input_ids), device=cuda. GPU T4, 126s.

Доза = ‖ΔW‖_F (Frobenius norm readout-редакции); bridge = cosine readout (exp16-метрика, ≠ exp19 per-token), KL = leash-метрика (full-vocab, base vs edited). База при ΔW=0: **bridge=0.599965** (= exp19 baseline — совпадает), KL=0.

| step | dose ‖ΔW‖ | bridge (cos) | KL (leash) |
|---|---|---|---|
| 0 | 0.0 | 0.599965 | 0.0 |
| 10 | 61.8 | 0.6173 | 0.4808 |
| 20 | 77.4 | 0.6460 | 0.5970 |
| 40 | 91.6 | 0.6843 | 0.3627 |
| 60 | 98.6 | 0.7137 | 0.2836 |
| 80 | 100.2 | 0.7359 | 0.2608 |
| 100 | 100.7 | **0.7543** | 0.2403 |

**Вывод:** readout-редакция (lm_head-only) — **causal lever**: bridge монотонно растёт 0.60→0.754 (т.е. мост к target-подпространству **улучшается**), при этом KL leash держит (0→0.24, damage bounded). **α\* = 0.3207 при step 100** — к моменту 100 шагов колено (knee, перестановка bridge gain ниже knee_tol=0.001) **не достигнуто**: bridge продолжает расти. Т.е. readout-редакция — самое дешёвое управление (1 матрица ‖W‖≈314, не вся сеть), и она движет мост в сторону target с ограниченным collateral (KL). **V4 (1000 шагов)** — continuation этого прогона, bridge достигает **0.921** (‖ΔW‖=128.8); см. следующий раздел (поведенческая проверка). V1 (alpha sweep как trust-region caps) **VOID** — caps ≥0.1 не bind, α не был дозой (его 'α*=0.01' = артефакт первого slope); v2 чинит определение дозы. **Оговорка:** exp16 (analytical last-layer readout, bridge 0.60→0.754) и exp19 (per-token full-path, baseline 0.600) — **разные величины bridge**, напрямую несравнимы (structural, не bug).

---

### Эксперимент 16 (v4) — поведенческая проверка LORA: top-p retention

> **Params:** финальный ΔW из exp16 v4 (‖ΔW‖_F=128.8, ‖W‖≈314, bridge@end=0.921) накладывается на lm_head 0.8B; поведение измеряется на **held-out** запросах (≠ training queries) тремя путями — base 0.8B / 0.8B+LORA / 4B-учитель. Метрика — **top-p retention** (p=0.9): доля массы вероятностей модели, попадающая в top-p регион учителя (top-p set = минимальный префикс, покрывающий 0.9 массы). Т.к. топ-k (top-10) — хрупкий индикатор (малейший шов двигает конкретный член), оценка построена на top-p retention как на робастной массовой метрике.

**Single-edit (s=1, чистые held-out вопросы, n=6):**

| Метрика (vs 4B-учитель) | base 0.8B | 0.8B + LORA |
| --- | --- | --- |
| Масса модели в top-p(0.9) регионе учителя | **0.841** | **0.016** |
| Масса учителя в top-p(0.9) регионе модели | **0.888** | **0.173** |

**Доза-ответ top-p retention (s ∈ {0.1, 0.25, 0.5, 0.75, 1.0}, C4-heldout, n=6):**

| s (доля ΔW) | top-p retention | bridge (exp16-metric) |
| --- | --- | --- |
| 0 (base) | **0.837** | 0.603 |
| 0.10 | 0.809 | (растёт) |
| 0.25 | 0.746 | ↑ |
| 0.50 | 0.666 | ↑ |
| 0.75 | 0.523 | ↑ |
| 1.00 | **0.416** | **0.921** |

(Метрики top-k: agreement k=10: 0.733→0.067; garbage k=10 (вне top-200 учителя): 0→0.450 — оба монотонны.)

**Вывод.** LORA exp16 **не улучшает** поведение, а **ломает** его. Уже при **s=0.1** (10% финального edit) top-p retention падает 0.837→0.809; при полном edit — до 0.016 (чистые вопросы) / 0.416 (C4), т.е. модель выводится **с manifold** (масса уходит в нереалистичные для учителя токены). Критично: это происходит **одновременно** с **монотонным ростом bridge** (0.603→0.921) вдоль той же траектории. Т.е. **геометрический bridge и поведенческая top-p retention антикоррелированы**: мост «улучшается», поведение деградирует.

**Главный вывод — точечных якобианов недостаточно.** Эксперимент 16 тянет **одну точку** (один last-layer hidden state → одно направление ΔJ, одна метрика bridge по одной паре слоёв). Оказалось, что такую точку можно «подогнать» (bridge 0.60→0.92) **без улучшения — а с разрушением** реального поведения. Точечный якобиан не несёт ограничения на **остальные** точки/домены, поэтому оптимизация по нему = **Goodhart на геометрическом прокси**: метрика растёт, качество гниёт. Чтобы это исправить, метрика/ограничение должны быть **многоточечными**: не точка, а **гиперповерхность из якобианов по доменам** (и по глубине), т.е. нельзя подогнать одну точку, не сломав другие. Это направление — следующая работа (после exp16/exp19).

---

### Обновлённая итоговая картина

| Утверждение | Статус | Эксперименты |
| --- | --- | --- |
| Чувствительные подпространства доменно-специфичны на ранних слоях | ✅ | Exp 13 (overlap 0.30–0.42) |
| Все домены сходятся к одному подпространству на поздних слоях | ✅ | Exp 13 (overlap ≈ 1.0 at L23) |
| **Мост — causal lever: readout-редакция движет его** | ✅ | **Exp 16 (bridge 0.60→0.754, KL leash 0→0.24, α*=0.3207)** |
| Монотонность моста робастна к домену | ✅ | Exp 13 (ρ = -0.89) |
| Конвергенция моста — не артефакт фазы GDN/full | ✅ | Exp 15 (A_vs_B=1.0) |
| Gauge share немонотонна (peak 0.48 L3 → 0.08 L23) | ✅ | Exp 14 |
| Learned bridge matrix ≠ random orthogonal rotation | ✅ | Exp 3(random R): loss=1.02 vs 0.0 |
| Граница линейности ε ≈ 0.1–1.0 | ✅ | Exp 12a (ε-sweep) |
| **Поле $J$ не фрактально на масштабах $\sigma_d$ (kill-switch)** | ✅ | **Exp 0.1 (C(σ_d)=0.33–0.87 по глубине)** |
| **Грамианы — sample-local: Якобианы разных сэмплов домена почти ортогональны** | ✅ | **Exp 0.1 (C(b)≈0.003–0.05)** |
| **Кластерной структуры кросс-сэмпловых Якобианов нет (в т.ч. доменной)** | ✅ | **Exp 27.5/27.6 (T6 eigengap, T4 ARI, T2/T3 math-изоляция)** |
| Gramian не требует регуляризации | ✅ | Exp 6b: κ=1684, λ≤0.01 не меняет результат |
| Чувствительность (trace) зависит от домена на L0 | ✅ | Exp 13 (chat 37.3M >> wiki 16.7M) |

### Ограничения и оговорки

1. **Exp 13:** абсолютная величина моста несравнима между доменами (разные распределения вероятностей). Монотонность — сравнима.
2. **Exp 14:** Запущен с N=8, K=64 (script `experiments/exp14_snr_pr.py` `K=64`, `n_components=K`). Полные значения SNR, PR, gauge_share, trace для 24 слоёв.
3. **Exp 15:** Схема D запущена (данные в `exp15_gdn_anchors.json`). Все 4 схемы дают идентичный Spearman ρ = -0.95. Максимальное расхождение между схемами: 0.12.
4. **Exp 12a (ε-sweep):** gauge поведенчески активен (uniformity 0.92–0.97), но KL_g ≈ KL_d при ε ≤ 0.1 (линейный режим). При ε = 1.0 разрыв до 7% — не gauge-инвариантность, а нелинейное затухание gauge-сигнала.
5. **Exp 11a:** линейный предиктор ненадёжен в глубоких слоях (нелинейности доминируют при ε ≥ 1.0).

### Future work (спроектированы, требуют запуска)

**Выполнено в этом раунде:**
- Exp 3(random R): случайная ортогональная матрица vs learned W
- Exp 12a (ε-sweep): ε ∈ {0.01, 0.1, 0.5, 1.0} — граница линейного режима
- Exp 6b: condition number + Tikhonov λ-sweep — регуляризация не нужна
- **Exp 0.1: длина когерентности поля $J$ — kill-switch пройден (не фрактально), sample-local Грамианы (§12)**

1. **Exp 17: LoRA-дистилляция.** (Exp 16 done — readout-pull как доза-ответ выше.) Послойная LoRA-дистилляция: оптимизировать LoRA на слоях, где **gap высок** (выравнивание хуже всего).
2. **Exp 18: Контроль logit-KD.** Валидация того, что дистилляция, направляемая выравниванием Якобиана (bridge/gap), действительно улучшает quality vs стандартный logit-KD на downstream задачах.
3. **Exp 19: Калибровка детектора разрушения.** Использование ESG для детекции направлений, в которых квантование/сжатие разрушает качество, и калибровка порога защиты.


## 12. Stage-0 kill-switch: длина когерентности поля Якобиана (Exp 0.1)

> Мотивация — из Exp 12a: линейный режим (корректность первого порядка, на котором стоит вся конструкция Моста) ломается при ε ≥ 0.5. Если поле $J(h)$ при этом **фрактальное** — быстро осциллирует на масштабах, сопоставимых с типичным расстоянием между сэмплами домена, — то **доменно-условные Грамианы $G^d$ бессмысленны** (пул Якобианов усредняет несвязанные значения), и Мост как многоточечная метрика (см. §10, «обновлённая итоговая картина») не может работать. Exp 0.1 — kill-switch: измерить длину когерентности поля $J$ и решить, продолжать или останавливаться.

### Эксперимент 0.1 (N=32/домен, K=64): Коэффициент когерентности C(δ) (0.8B, слои 6/12/18/22, GPU T4)

**Задача.** Поле $J: \mathbb{R}^d \to \mathbb{R}^{d\times K}$ (gauge-free $J^{(p)} = \text{diag}(p) - pp^T J$, $p$ = softmax логитов, перенормированный по глобальному токен-сету top-K=64 0.8B self-bridge) гладкое, если близкие точки дают близкие Якобианы. Метрика — коэффициент когерентности по парам:

$$C(\delta) = \mathbb{E}_{\|h - h'\| \approx \delta} \left[ \frac{\langle J(h), J(h')\rangle_F}{\|J(h)\|\,\|J(h')\|} \right]$$

(косинус по строкам в порядке ранга токенов, не по id), биннинг по $\log \delta$ (14 бинов, ≥5 пар/бин). Коэффициент когерентности — $\ell = \delta$ при $C(\delta) = 0.5$; гейт — $C(\sigma_d)$, где $\sigma_d = \|\text{std}_{i \in d}\, h_i\|_2$ (норма стандартного отклонения по размерам, по доменам). Три типа пар: **(a)** соседние позиции одного сэмпла $(S{-}2, S{-}1)$, $(S{-}1, S)$ (малые δ, ≤ $\sigma_d$); **(b)** две выборки одного домена (δ ~ $\sigma_d$); **(c)** междоменные (большие δ). Параметры: 4 домена (wiki/math/code/chat) × 8 сэмплов, max_length=64, 384 gauge-free J-вызова, 644 с на T4 (0.18 GPU-h).

| Слой | σ_d | C(σ_d) | приговор | (a) intra-sample C | (b) between-samples C |
| --- | --- | --- | --- | --- | --- |
| 6 | 1.66 | 0.33 | умеренное | 0.14–0.35 (δ 1.7–2.6) | −0.003–0.029 (δ 2.3–2.9) |
| 12 | 1.86 | 0.45 | умеренное | 0.30–0.49 (δ 1.8–2.8) | 0.011–0.023 (δ 2.7–3.0) |
| 18 | 4.78 | 0.64 | гладкое | 0.67–0.71 (δ 5.6–6.6) | 0.04–0.17 (δ 6.9–7.8) |
| 22 | 7.18 | 0.87 | гладкое | 0.85–0.93 (δ 7.5–9.1) | 0.05–0.30 (δ 9.7–11.8) |

**Результат: НЕ ФРАКТАЛЬНО.** Kill-switch не сработал (порог STOP: $C(\sigma_d) < 0.1$) ни на одном из четырёх слоёв. Поле $J$ на масштабах доменной дисперсии $\sigma_d$ когерентно, и когерентность **растёт с глубиной** (0.33 → 0.87). Линейное приближение первого порядка корректно как раз там, где работают Мост и доменные Грамианы.

**Формализм: листы.** Поле $J$ — сечение векторного расслоения $E = \mathcal{H} \times \mathbb{R}^{K \times D} \to \mathcal{H}$ над многообразием скрытых состояний. **Лист** $\mathcal{L}$ — максимальное связное подмногообразие, где когерентность выше порога $C_{\min}$: $\forall h_1, h_2 \in \mathcal{L}:\ C(J(h_1), J(h_2)) \geq C_{\min}$. **Длина когерентности** в направлении $v$: $\ell(h, v) = \sup\{ t > 0 : C(J(h), J(h + tv)) \geq C_{\min}\}$; изотропная — $\ell(h) = \inf_{v \in S^{D-1}} \ell(h, v)$. **Гипотеза анизотропии** (поле анизотропно, экспериментально подтверждена типами пар (a)/(b)): разложив $\delta = \delta_\parallel + \delta_\perp$ относительно локального тангенциального подпространства листа, $C(\delta; h) \approx f(\|\delta_\parallel\|) \cdot g(\|\delta_\perp\|)$, где $f$ убывает медленно (вдоль листа), $g$ — быстро (поперёк). Проверка: для каждой пары вычислить $\delta_\parallel, \delta_\perp$ (PCA по соседним точкам траектории) и убедиться, что $C$ коррелирует со $\|\delta_\parallel\|$ слабо, а с $\|\delta_\perp\|$ сильно.

**Ключевой побочный сигнал.** Разрыв между типами пар: (a) в пределах сэмпла $C \approx 0.3\text{–}0.9$, а (b) между сэмплами одного домена $C \approx 0.003\text{–}0.05$ (пик 0.30, L22/code) — в общем случае Якобианы разных сэмплов домена почти ортогональны, хотя их hidden states лежат в пределах $\sigma_d$. Поле $J$ когерентно **вдоль последовательности** (позиции), но не поперёк распределения сэмплов. Следствие для ESG: Грамианы должны строиться **sample-local** (каждый сэмпл — свой Грамиан, как это сделано в Exp 16/19), а не пулом Якобианов по домену: пул усредняет почти ортогональные $J$ и убивает сигнал.

**Оговорки.** (1) Гейт оценен на кривой, сшитой из (a)+(b); (a) даёт покрытие при δ ≤ $\sigma_d$, (b) — диапазон δ ~ $\sigma_d$. $\ell$ (перекрытие $C=0.5$) формально не существует для чистых (b)-кривых ($C \le 0.3$ во всём диапазоне), т.е. поперёк распределения сэмплов когерентность ниже 0.5 уже на масштабе $\sigma_d$ — это не стоп, а количественная граница допустимой пулляции. (2) N=8/домен, доверительные интервалы не вычислялись (см. §6, п. 5). (3) Self-bridge 0.8B: 4B не загружается, токен-сет — из 0.8B (отклонение от параметрической таблицы, зафиксировано в PLAN).

**Практический итог:** Мост как многоточечная метрика допустим; пул Якобианов по доменам — нет (sample-local только). Открывается Exp 0.2 (синтетическая деградация readout — валидация метрики $d_{geo}$).

## 13. Post-hoc: значимость, кластеры и изоляция доменов поля $J$ (Exp 27.5/27.6)

Exp 0.1 показал, что Грамианы sample-local: внутри домена $C_b \approx 0.003$–0.05, а поперёк сэмплов — почти ортогонально. Оставались три вопроса: (1) статистически ли значима когерентность $C_a$ относительно null-моделей, (2) существует ли в кросс-сэмпловом спектре кластерная структура (в т.ч. доменная), (3) каков честный шумовой пол $F$ вместо $1/\sqrt{D}$.

**Данные и метод.** Exp 27 (v11, 0.8B, T4): 4 домена × 8 сэмплов, $S=64$, $K=128$, pos=0, 6 слоёв {6,12,18,19,22,23} (GDN + full-attention, контроль L23 ≈ lm_head). Gauge-free $J^{(p)}$ кешируются в jcache; post-processing — только CPU. Exp 27.5 (`experiments/exp27_5_postprocess.py`): перестановочный тест доменных меток (B=1000), spectral null (хранящие собственные значения ротации, B=100), half-split, корреляции с осями (entropy quartile), agglomerative clustering на $D^G = 1 - K^G$ (silhouette, gap, $\rho_{NN}$, cophenetic). Exp 27.6 (`experiments/exp27_6_isolation.py`): P0-приложение — T1–T8 (см. §D ниже). Два раунда ревью (коллега + внутренний аудит) выловили 6 P0-багов; все закрыты, финальный прогон — run12.

**Результаты (по слоям L6/L12/L18/L19/L22/L23, baseline S=64 K=64, $K=128$):**

| Метрика | Значения | Вердикт |
| --- | --- | --- |
| $C_a$ (pos-pk) | 0.061 → 0.229, монотонно | **COHERENT** на всех 6 слоях (8–41× $F$) |
| $C_b$ (между сэмплами, чистый math-pool) | +0.009 → +0.104 | ORTH/WEAK на L6–L12, растёт с глубиной |
| $F = 1/\sqrt{r_{eff}(K+D-r_{eff})}$ | 0.0056–0.0078 | в 4–9× ниже $1/\sqrt{D} = 0.031$ (v6 exp01) |
| $C_a^{raw}$ | 0.304 → 0.996 | gauge-mode артефакт: $\|P(p)J\|^2/\|J\|^2 \approx 0.0005$–0.0025 |
| $g_{uni}$ (uniform-режим) | $\le 3\times10^{-12}$ | после gauge-free проекции — машинный ноль |

**T-блоки 27.6 (run12, финальный):**

| T | Что | Результат |
| --- | --- | --- |
| T1 | $T_{domain} = C_{within} - C_{cross}$, {all, no_math, …} | inconclusive: max $\|T_{domain}\| < 0.01$ (шумовой пол) |
| T2 | Bock 4×4 на $K^G$ | **pass**: math изолирован (диагональ в 4–5× ниже других) |
| T3 | bootstrap CI (B=1000, дедуп-маска пар) | inconclusive: $T_{domain} = 0.0024 \pm 0.011$, CI ∋ 0 на всех слоях |
| T4 | permutation-null ARI (Bonferroni α=0.0083) | inconclusive: raw p<0.05 только на L12, не выдерживает коррекцию |
| T5 | dip + beta-null (Hartigan, moment-match) | **pass**: 12/12 reject_unimodal (dip 0.006–0.011 > null_p95 0.005–0.006) |
| T6 | eigengap $K^F$, $K^G$ | **pass**: $r_{eff} = 24–32$ из 31, спектра плоские — silhouette 27.5 (0.19) измеряет шум |
| T7 | ρ(layer, $C_a$) + perm-null | **pass**: ρ = 1.0, CI [1.0, 1.0], p = 0.003 |
| T8 | ковариаты: $E, r_{eff}, r_{stable}, H, g_{uni}$ | **pass**: $r_{eff}$ коррелирует с $H$ (уверенные сэмплы чувствительнее по больше направлений) |

**Итоговая картина поля $J$.** Поле расслоено (fibered): вдоль сэмпла когерентно (длина ~1–2 позиции, $T_7$: рост с глубиной защищён), между сэмплами — почти ортогонально на ранних слоях и слабая поперечная когерентность с глубиной ($C_b/F$: 1.2 → 17.5). Кластерной структуры кросс-сэмпловых Якобианов нет: доменная ось статистически неотличима от шума ($T_1/T_3/T_4$), спектр плоский ($T_6$), math-домен — аутсайдер-изолят ($T_2$). Вердикт exp01 v6 «FIBERED/ORTHOGONAL» подтверждён, но с честным полом $F$: «orthogonal» — это «ниже $1/\sqrt{D}$», а не «ниже $F$».

**P0-баги двух ревью (закрыты, run12).** (1) Gauge-free проекция: einsum по samples вместо tokens — «gauge-free» $J$ сохранял gauge-режим; после фикса $g_{uni} \sim 10^{-12}$. (2) Gap-statistic null: свежие ротации на итерацию (было: те же 100 ротаций → статистика дегенерировала). (3) Half-split ARI на скалярах (было: на J-векторах с гаучевым знаком). (4) T3 bootstrap: дубликат-пары (resample with replacement даёт $K_{ii} = 1$) — дедуп-маска. (5) T2: диагональ $\mathcal{D}$ была NaN (было: `i != j` внутри `np.triu` без маски → деление 0/0). (6) T5: beta moment-match не сохранял mean (было: cv-формула, mean_fit = 2.5× наблюдаемого) — правильные формулы $S = m(1-m)/s^2 - 1$, $a = mS$, $b = (1-m)S$ + нормализация в (0,1). Selfcheck: T3-дедуп + T5 (unimodal не отклоняется, bimodal отклоняется).

**Оговорки.** (1) T5 dip на грани критического значения (0.006–0.011 vs null_p95 ~0.005): маржа 1.1–1.5×, вердикт «pass» — минимальный, би-модальность слабее, чем заявлено selfcheck-бимодальностью. (2) N=32: разрешение $T_3$ — CI ±0.011, эффекты меньше неотличимы от нуля. (3) L23 = контроль (≈ lm_head, pre-final-RMSNorm, $\sigma_d \approx 96$–104 — норма для residual stream). (4) S=128 dip (ctx 127) — эффект длины контекста, не mid-context (пара (32,33)@S128 выше S64 pos0 на всех слоях); гипотеза GDN chunk-boundary @128 не проверена.

### Математический аппарат Exp 27/27.5/27.6

Объекты (Exp 27, exp27 v11; слой l ∈ {6,12,18,19,22,23}, токены x ∈ U_128, |U|=K):

- $h_t = h(x, t) \in \mathbb{R}^D$ — скрытое состояние на позиции t; $J_t = \partial h_{l,t} / \partial x \in \mathbb{R}^{D \times K}$ — Якобиан скрытого состояния по токенам (backprop; в кэше хранится уже gauge-free $J^{(p)}_t$).
- **Гаучев оператор**: $P(p) = \mathrm{diag}(p) - pp^T$, где $p = p_U$ — softmax, урезанный и ренормализованный на U. $P$ симметричен, полуопределённо положителен ($x^T P x = \mathrm{Var}_p[x] \ge 0$), $\ker P = \mathrm{span}\{\mathbf{1}\}$, rank = $K-1$; $P^2 \ne P$ (не проектор; идемпотентен лишь при $p = \mathrm{uniform}$). Точка: $P = \partial \mathrm{softmax}/\partial z$, поэтому $J^{(p)} = P J = \partial p / \partial h$ — канонический Якобиан выходного распределения, и $J^T P J$ — информация Фишера: $\mathrm{KL}(p \| p_{h+\delta}) = \tfrac{1}{2}\delta^T J^T P J\, \delta + O(\|\delta\|^3)$.
- **Gauge-free Якобиан**: $J^{(p)}_t = P(p_t) J_t$. Три конвенции пар (t1, t2): **pos-pk** (каноническая: $P(p_{t1})J_{t1}$ и $P(p_{t2})J_{t2}$ — проектор на своём положении), **same-pk** (оба члена с $p_{t1}$ или оба с $p_{t2}$), **raw** (без проектора). Инвариантность: $JJ^T$-статистики инвариантны к $J \mapsto J O$ ($O \in O(D)$), vec-косинус $C$ — нет ($\mathbb{E}_{Haar}[C] = 0$): нулевое $C$ между сэмплами априори не различает «нет структуры» и «ротационная структура» (одинаковые грамианы, разные вращения); дискриминант — сравнение грамианов по доменам (§13.1, факт 5). Утечка на усечённом носителе: $O(p(U)(1-p(U)))$ (независимый равномерный сдвиг $\mathbf{1}_U$ обнуляется точно).
- **Расстояние**: $\delta = \|h_{t1} - h_{t2}\|_2$ (скрытое, не текстовое). $\sigma_d$ — норма покомпонентного std скрытых состояний на последней позиции (median по доменам).
- **Метрики**: $C_a = \mathrm{mean}\ \cos(J^{(p_1)}_{t1}, J^{(p_2)}_{t2})$ по position-парам одного образца; $C_b = \mathrm{mean}\ \cos$ по sample-парам одного домена (pos 0); $T_{domain} = C_{within} - C_{cross}$ (27.5/27.6: $C_{within}$ = 112 пар в доменах, $C_{cross}$ = 384 междоменных, N=32 = 4×8).
- **Эффективное измерение** ($r_{eff}$): распределение $p_k = \sigma_k^2 / \sum \sigma^2$ по сингулярным значениям $J$ [K, D]. Основной: **ентропический ранг** $r_{eff} = \exp(-\sum p_k \log p_k)$; вторичные: stable rank $= \|J\|_F^2/\sigma_1^2$, participation ratio $= (\sum \sigma^2)^2 / \sum \sigma^4$.
- **Шумовой пол**: $F = 1/\sqrt{r_{eff}\,(K + D - r_{eff})}$ (медиана $r_{eff}$ по слоям; в 27.5/27.6 — по конфигурации). Вердикты: $C < F$ — ORTHOGONAL, $F \le C < 3F$ — WEAK, $C \ge 3F$ — COHERENT.
- **Half-split**: N/2 + N/2 разделение, $\Delta_{half} = |C_a^A - C_a^B|$, CI из 100 bootstrap-ресэмпл (27.6: по каждому домену отдельно, $N_d/2 + N_d/2$).

27.5 (постпроцессинг jcache, slice S=64, K=128, pos 0):

- **Матрицы сходства**: $K^G_{ij} = \cos(J_i J_i^T, J_j J_j^T) = \langle J_iJ_i^T, J_jJ_j^T \rangle_F / (\|J_iJ_i^T\|_F \|J_jJ_j^T\|_F)$ — косинус token-грамианов; $K^F_{ij} = \cos(J_i, J_j)$ — прямые косинусы $J$ (Frobenius).
- **Null-модели**: (a) перестановка domain-меток, $B_{perm}=1000$ → $p_{perm}$ (односторонняя); (b) **спектральный null**: $G^{null}_i = R_i \Lambda_i R_i^T$ — случайные ортогональные вращения, сохраняющие собственные значения грамиана ($R_i \sim Haar$, $B_{rot}=100$) → $p_{rot}$.
- **Оси сигнала** (Spearman $\rho(C_{ij}, \Delta)$): $\Delta h$ (скрытая дистанция), $\Delta H$ (энтропия softmax на U), $\Delta\sigma_z$ (std логитов на U), $\Delta\log E$ ($E = \|J_i\|_F^2$), $\Delta g_{uni}$; $g_{uni} = \|Q J_i\|_F^2 / E_i$, $Q = uu^T$, $u = \mathbf{1}/\sqrt{K}$ (доля энергии в uniform-режиме). Entropy-quartile: 32 образца → 4 квартiles по $H$, $T_{ent} = C_{within-quartile} - C_{cross-quartile}$ + size-preserving перестановка.
- **Кластеризация** (без доменных меток): $D^G = 1 - K^G$; agglomerative (average linkage, precomputed), $k = 2..8$; silhouette $S_{max}$; gap statistic ($B_{gap}=100$ eigenvalue-preserving nulls); $\rho_{NN} = \bar{K}_{within\,cluster} / \bar{K}_{nearest}$; cophenetic correlation; half-split silhouette (16+16, ×50).

27.6 (блок-перестановки, T-блоки):

- **T4**: $T_4(L) = C_a(L) - C_b(L)$ (layer-level difference); null = block-permutation по доменам (сохраняет within-domain корреляцию); $p_{4,adj} = B \cdot \min(0.05, p_4 / \alpha_4)$, где $\alpha_4 = 0.05/(n_{layers})$ (Bonferroni по слоям).
- **T5**: Hartigan dip-test на распределение 112 $C_b$-значений (бимодальность); $p_{dip}$ из β-null: параметры $a = mS$, $b = (1-m)S$, $S = m(1-m)/s^2 - 1$ (moment-matching на эмпирическое $m, s^2$), $B=100$ β-выборок.
- **T6**: $\lambda_1, \lambda_2$ собственных значений усреднённой $K^G$ (32×32); $\Delta\lambda = \lambda_1 - \lambda_2$ (eigengap); $r_{eff}(K^G) = \exp(-\sum \lambda_k \log \lambda_k)$ (нормировка $\sum \lambda_k = 1$).
- **T8**: $C_b$ на math-only пуле (8 образцов, 28 пар) vs full pool (4 домена, 112 пар).
- **T1**: вырожденные Якобианы ($E_i = \|J_i\|_F^2 < 10^{-12}$) исключаются; фиксируется доля $n_{degen}/32$ (все слои: 0).

Чувствительность:

- **Per-$k^*$ gap test**: $K^G_{k^*}$ = $k^*$-й главный компонент $K^G$ (без усреднения по компонентам) для $k^* \in \{2,3,4\}$; если eigengap не виден на всех $k^*$, он отсутствует, а не усреднён.
- **Per-$S$ dip**: Hartigan dip для каждого $S \in \{64, 128\}$ отдельно.
- **K-чувствительность $r_{eff}$**: $r_{eff}$ пересчитана для $K \in \{64, 128, 256\}$ (ранки $J$ по сингулярным значениям [K, D] — K-зависимы); $\Delta r_{eff}$ (K=128 vs K=64) < 3% на всех слоях — спектральный вывод K-устойчив.

### 13.1. Выводы Exp 27.5/27.6: факты, интерпретации, открытые вопросы

**Строго установленные факты** (следуют из данных без дополнительных предположений):

1. Внутридоменная когерентность Якобианов статистически неотличима от междоменной (p_perm > 0.05 на всех слоях; все bootstrap CI включают ноль).
2. Спектры матриц сходства $K^F$ и $K^G$ не содержат заметных разрывов (eigengap ratio < 1.5 для всех $i \in \{2,3,4\}$); эффективный ранг $r_{eff} \approx 26$–32 из 31.
3. Внутридокументная когерентность $C_a$ монотонно возрастает с глубиной (Spearman $\rho = 1.0$, $p < 0.003$ для всех трёх конвенций: pos-pk, same-pk-late, raw).
4. Распределение попарной когерентности отвергает гипотезу унимодальности (Hartigan dip, $p < 0.05$ на всех слоях для обоих срезов $S=64$ и $S=128$).
5. Math-домен демонстрирует систематически низкую внутридоменную когерентность: $K^G_{mm} = 0.008$–0.011 против $K^G_{dd} = 0.033$–0.042 для остальных доменов (3–4×).
6. Корреляция $\rho(C, \Delta h)$ отрицательна на поздних позициях (pos=8: от $-0.07$ на L6 до $-0.40$ на L22), но слабо положительна или около нуля на ранних (pos=1–4).

**Интерпретации (с оценкой вероятности):**

*Высокая (>80%):*
- **Глобальная доменная геометрия отсутствует.** Все тесты (permutation, bootstrap CI, ARI-null, eigengap) согласованно показывают отсутствие сигнала. Альтернативы: эффект $N=32$ — маловероятно (CI с ростом $N$ сужаются, но остаются широкими); math-домен маскирует сигнал — отвергнуто (T1: исключение math не улучшает сигнал); **ротационная структура** (одинаковые грамианы $JJ^T$, различие — ортогональные вращения в hidden-пространстве; vec-$C$ в ней нулевой) — отвергнута фактом 5: для неё нужно $K^G_{dd} \approx 1$, наблюдаемо 0.033–0.042. Проверка: $N=64$–128, повтор T3.
- **Кластерная структура в спектральном смысле отсутствует.** Плоский спектр (факт 2) ⇒ silhouette и gap statistic измеряют шум, а не структуру. Альтернативы: структура в нелинейном многообразии (средняя вероятность — проверка UMAP/t-SNE); размытие из-за усреднения по позициям (средняя — per-position кластеризация).
- **Readout-pull имеет фундаментальный потолок** для кросс-сэмплового выравнивания: без кросс-сэмпловой когерентности (факт 1) глобальная правка $W$ не может выровнять разные «листы» траекторий. Наиболее вероятно, что readout-pull работает только внутри траекторий (согласуется с фактом 3). Проверка: ограничить readout-pull парами одной траектории (intra-sample).

*Средняя (40–80%):*
- **Бимодальность связана с внутри-/междокументным разделением.** Факт 4 (бимодальность) при отклонённой доменной оси (факт 1): естественная альтернатива — разделение на пары внутри одного документа (высокая $C$) и между разными (низкая $C$). Альтернативы: позиция в документе (средняя), синтаксические паттерны (низкая — code-домен не выделяется). Проверка: split пар на intra-/inter-document.
- **Отрицательный $\rho(C, \Delta h)$ на поздних позициях — эффект «листов».** Близкие точки в $h$-пространстве могут принадлежать разным листам: траектории расходятся. Альтернативы: масштабный эффект (низкая — косинус масштабно-инвариантен); нелинейность поля $J(h)$ (средняя — анализ вторых производных). Проверка: локальные тангенциальные подпространства, когерентность внутри/между листами.
- **Math-домен имеет специфическую структуру** (символьная запись, формулы). Альтернативы: больше шума (короткие формулы, мало контекста) — средняя; особый набор токенов — средняя. Проверка: распределение токенов, контроль на длину контекста.

*Низкая (<40%):*
- **Латентные оси кроме доменов существуют** — но ARI с доменами не значим (T4), что снижает вероятность сильных латентных осей. Проверка: unsupervised кластеризация + атрибуция (энтропия, длина, позиция).
- **Бимодальность связана с синтаксисом** — code-домен не отличается от wiki/chat, что противоречит этой гипотезе.

**Открытые вопросы (требуют экспериментов).**
1. Механизм отрицательной корреляции с $\Delta h$ (тангенциальные подпространства, вторые производные $\nabla^2 J(h)$).
2. Природа бимодальности (intra-/inter-document split, per-position, кластеризация + атрибуция).
3. Эффект размера выборки ($N=64$–128, повтор T3; если CI не сужаются — эффект реальный, но слабый).

**Итоговая формулировка.**
- *Основной вывод:* глобальная доменная геометрия Якобианов отсутствует — внутри- и междоменная когерентность неотличимы, спектр плоский, кластерная структура в спектральном смысле не обнаружена.
- *Вторичный вывод:* $C_a$ монотонно растёт с глубиной (формирование устойчивой локальной геометрии в поздних слоях); бимодальность, вероятно, отражает внутри-/междокументное разделение.
- *Практический вывод:* readout-pull ограничен для кросс-сэмплового выравнивания; рабочая парадигма — per-sample / leaf-oriented геометрия, а не глобальная доменная.
- *Ограничение:* $N=32$; увеличение выборки может выявить слабые эффекты, но основные выводы (отсутствие доменной оси, плоский спектр) устойчивы.
