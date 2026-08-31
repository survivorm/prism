# p2_tz.md — P2: Трансформация младшей модели через bridge loss

> **Куда запускать:** Kaggle GPU (T4×2 или P100).
> **Зависимости:** `prism/` (jacobian.py, bridge.py, calibration.py).

---

## Основная серия P2: Exp16–19

**Полное ТЗ уже в `exp10_article_supplement.md`:**

| Эксперимент | Файл | Что |
|---|---|---|
| Stage 0 | `exp16_stage0_targets.py` | Кэширование целевых якобианов (~7 пар слоёв × n_samples) |
| Exp16 | `exp16_readout_pull.py` | Пилот без обучения — правка только `lm_head`, свип по α |
| Exp19 | `exp19_destruction.py` | Калибровка детектора разрушения — 4 сценария при α* |
| Exp17 | `exp17_lora_distill.py` | Послойная LoRA-дистилляция с bridge-loss |
| Exp18 | `exp18_logit_kd.py` | Контроль — logit-KD при том же бюджете |

**Порядок:** 16 → 19 → 17 → 18 (калибровка детектора до доверия к метрикам).

**Оркестратор:** `experiments/run_bridge_exps.py` — опции `--skip-stage0`, `--skip-exp16` и т.д.

**Что в `exp10_article_supplement.md` и НЕ дублируется здесь:**
- Методология транспорта якобианов (формулы)
- Батарея метрик (8 метрик: bridge_per_pair, KL_ed_L, P_ed(A_L), rank_top10_L_in_ed, hit5_in_U, KL_ed_orig, PPL_ed_orig, gsm8k_accuracy)
- Device-конфигурация (GPU vs CPU: dtype, n_samples, K, α-свип, GD steps)
- Runtime-guard (замер первого якобиана; если >60 с — уменьшение K и n_samples вдвое)
- Три сценария интерпретации (pull улучшает / bridge падает / разрушение)

---

## Уникальные эксперименты P2 (НЕ в supplement)

### P2.5: Projection matrix (без LoRA, без обучения)

**Что:** Для каждого слоя найти W_proj^(l) ∈ R^{D_S×D_S} так, чтобы J_S @ W_proj^(l) ≈ J_S_target. Закрытое решение (least squares).

**Почему уникально:** Самое дешёвое тестирование H4. Нет backprop, нет training. Если projection matrix не улучшает PPL → H4, скорее всего, неверна → можно не запускать Exp17–19.

**Pipeline:**
1. Stage 0 (или взять из кэша): J_S и J_S_target на N=4 сэмплах
2. Для каждого слоя l:
   - W_proj^(l) = pinv(J_S^T @ J_S) @ J_S^T @ J_S_target
   - Применить к hidden states: h' = h @ W_proj^(l)
3. Замерить PPL с W_proj vs baseline PPL

**Критерий:**
- Если PPL с W_proj ≈ baseline → projection работает (H4 prelim подтверждена)
- Если PPL с W_proj >> baseline → projection ломает модель (H4 prelim опровергнута)

**Код:** `./kaggle_kernel/p2_exp5_projection_matrix.py`

**Проверка:**
- [ ] PPL baseline записан
- [ ] PPL с W_proj записан
- [ ] ΔPPL записан
- [ ] Вывод: W_proj улучшает/не улучшает модель → запускать/не запускать Exp17–19

---

## Порядок и рекомендации

| Шаг | Что | Зависит от | Приоритет |
|---|---|---|---|
| P2.5 | Projection matrix (закрытое решение) | — | P0 (самый дешёвый, быстрый ответ на H4) |
| Stage 0 | Кэширование Jacobians | — | P0 (см. supplement) |
| Exp16 → 19 → 17 → 18 | Основная серия | Stage 0, P2.5 (recom) | P0 |

**Рекомендация:** Запустить P2.5 ПЕРВЫМ (1 час GPU). Если projection matrix не улучшает PPL → H4, вероятно, неверна, можно не запускать основную серию (экономия ~20–40 часов GPU).

