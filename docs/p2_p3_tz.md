# p2_p3_tz.md — ТЗ на P2/P3/H4

> **Предпосылка:** P2/P3/H4 — следующие шаги после валидации моста. Сейчас мост — только диагностический инструмент. Цель: превратить его в оптимизационный.
>
> **Зависимость:** P2 и P3 требуют H4. H4 — гипотеза, которая проверяется экспериментально (Exp 16–17 в Future work). P2 и P3 — проектирование; H4 — проверка.
>
> **Ключевая техническая проблема:** Bridge loss = cos_sim(J_S, J_S_target). J_S = d(logits_S)/d(h_S) — первый порядок. Чтобы оптимизировать веса модели, нужен dJ_S/dw = d²(logits_S)/d(h_S)dw — второй порядок (third-order tensor: [tokens, hidden_dim, model_params]). Прямая оптимизация весов невозможна (O(V·D·params) памяти).

---

## T1: Анализ вариантов оптимизации (проектирование, не код)

**Цель:** Определить, КАК оптимизировать младшую модель через bridge loss, учитывая невозможность прямого backprop через J_S.

**Варианты (требуют сравнения):**

### Вариант A: LoRA adapters на младшей модели
- Добавить LoRA-адаптеры к каждому слою 0.8B (Q- и V-веса attention, FFN)
- J_S вычисляется с LoRA-адаптерами → backprop через J_S → градиенты по LoRA-весам
- **Плюсы:** Мало параметров (<1% от модели), совместимо с PEFT
- **Минусы:** J_S backprop всё ещё требует second derivative (J_S зависит от h, h зависит от LoRA-весов через forward pass)
- **Стоимость:** ~10-100× дороже обычного fine-tuning (из-за Jacobian computation)

### Вариант B: Projection matrix W_proj
- Для каждого слоя l найти W_proj^(l) ∈ R^{D_S×D_S} так, чтобы J_S @ W_proj^(l) ≈ J_S_target
- Решается как least squares: W_proj = pinv(J_S^T @ J_S) @ J_S^T @ J_S_target
- Затем применять W_proj к hidden states: h_S' = h_S @ W_proj
- **Плюсы:** Закрытое решение, не требует backprop через модель
- **Минусы:** Непараметрическое, не обучаемое, только layer-wise

### Вариант C: Proxy loss через активации
- Вместо J_S оптимизировать что-то дешевле, что коррелирует с bridge loss
- Например: MSE между hidden states (стандартная дистилляция) + bridge loss как регуляризатор на подмножество слоёв
- Или: bridge loss только для финальных слоёв (где J_S маленький и дешёвый)
- **Плюсы:** Значительно дешевле
- **Минусы:** Корреляция proxy ↔ bridge loss не гарантирована

### Вариант D: Bridge loss как регуляризатор (P3)
- Fine-tune младшую модель на стандартной задаче (next-token prediction)
- Добавить bridge loss как регуляризатор: L_total = L_NTP + λ · L_bridge
- Bridge loss вычисляется на отдельном шаге (не на каждом sample)
- **Плюсы:** Стандартный pipeline, bridge loss улучшает quality (если H4 верна)
- **Минусы:** Два forward passes per step (NTP + bridge), медленнее

**Результат T1:** Выбрать 1-2 варианта и обосновать выбор (стоимость, сложность, ожидаемый эффект). Написать 1-2 страницы анализа.

**Критерии выбора:**
1. Вычислительная стоимость (CPU vs GPU, время)
2. Количество оптимизируемых параметров
3. Совместимость с существующим кодом
4. Ожидаемое улучшение quality

---

## T2: Проектирование pipeline (на основе выбора T1)

**Цель:** Написать детальный pipeline оптимизации.

### Шаги:

1. **Data pipeline:** Калибровочные данные → token_set → Jacobian computation
   - Используем существующий `compute_jacobian_for_tokens` из `prism/jacobian.py`
   - N сэмплов, K токенов

2. **Loss computation:**
   - J_S и J_S_target через `jacobian_bridge` из `prism/bridge.py`
   - Для LoRA: J_S вычисляется с LoRA-адаптерами, backprop по LoRA-весам
   - Для proxy: bridge loss на subset слоёв

3. **Optimization:**
   - Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
   - Epochs: 1-3 (bridge loss быстро стабилизируется)
   - Batch: 1 sample per step (Jacobian computation не масштабируется по batch)

4. **Validation:**
   - Bridge loss на validation set (не train)
   - PPL на validation set
   - Downstream tasks (если H4 проверена)

**Результат T2:** Pseudo-code + архитектурная диаграмма.

---

## T3: Реализация prototype (минимальный рабочий пример)

**Цель:** Рабочий код на 0.8B модели, один слой.

### Что реализовать:

1. **BridgeLossModule:** PyTorch Module, который:
   - Принимает model, layer_idx, input_ids, attention_mask, token_set
   - Вычисляет J_S через compute_jacobian_for_tokens
   - Вычисляет J_S_target через jacobian_bridge
   - Возвращает scalar loss (differentiable по model parameters)

2. **Train loop:**
   - Загрузка 0.8B + LoRA (если Variant A)
   - 100 шагов на одном слое
   - Мониторинг: bridge loss per step, PPL per 10 steps

3. **Evaluation:**
   - Bridge loss before vs after
   - PPL before vs after
   - Если PPL ухудшается — регуляризатор работает слишком сильно

**Результат T3:** Python script + результаты (bridge loss и PPL before/after).

**Критерий успеха:** Bridge loss ↓, PPL не ухудшается (или улучшается ≤ 1%).

---

## T4: Расширение на все слои + validation

**Цель:** Рабочий pipeline на всех 24 слоях.

### Что реализовать:

1. **Multi-layer bridge loss:**
   - L_total = sum_l L_bridge(l) (равномерные веса)
   - Или: L_total = sum_l w_l · L_bridge(l) (веса по вероятностям старшей модели)

2. **Scaling:**
   - Сколько сэмплов нужно? (1-4 на CPU, 8-16 на GPU)
   - Сколько шагов? (100-500)
   - Как часто валидировать? (каждые 10 шагов)

3. **Validation:**
   - Bridge loss на train vs validation
   - PPL на train vs validation
   - Downstream tasks (GSM8K, MMLU — если H4 проверена)

**Результат T4:** Script + результаты + выводы.

---

## T5: Проверка H4 (Minimizing bridge loss improves model)

**Цель:** Экспериментально подтвердить/опровергнуть H4.

### Эксперимент:

1. **Baseline:** 0.8B без оптимизации → PPL на validation set
2. **Bridge-optimized:** 0.8B после bridge loss оптимизации → PPL на validation set
3. **Control:** 0.8B после стандартной next-token prediction → PPL на validation set

**Критерий:** Bridge-optimized PPL ≤ Baseline PPL (или улучшение ≥ 1%)

**Если H4 верна:** P2/P3 готовы к production.
**Если H4 неверна:** Мост — только диагностика. P2/P3 остаются отложенными.

---

## Порядок и приоритеты

| Шаг | Что | Статус | Зависит от |
|---|---|---|---|
| T1 | Анализ вариантов | ⏳ | — |
| T2 | Проектирование pipeline | ⏳ | T1 |
| T3 | Prototype (1 слой) | ⏳ | T2 |
| T4 | All layers + validation | ⏳ | T3 |
| T5 | Проверка H4 | ⏳ | T4 |

**Бюджет:** T1-T2: проектирование (1-2 дня). T3-T5: реализация + эксперименты (3-5 дней).

**Блокировщик:** GPU access (Jacobian computation на CPU prohibitively expensive для оптимизации). Если только CPU — T3-T5 неfeasible.

