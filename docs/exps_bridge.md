Ниже — полное ТЗ для агента: pull-эксперименты Exp16–19, упакованные под Kaggle с автоматическим фоллбеком на CPU. Ключевая архитектурная идея, удешевляющая всё: **целевые якобианы старшей модели считаются один раз (Stage 0) и кэшируются**, дальше все эксперименты гоняют только малую модель — это и делает CPU-фоллбек тривиальным.

---

# ТЗ: Pull-эксперименты (Exp16–19). Kaggle + CPU fallback

## 0. Контекст и цели (вставить в шапку каждого скрипта)

Мы используем Jacobian bridge не как диагностик, а как **инструмент интервенции**: правим веса малой модели (Qwen3.5-0.8B) так, чтобы её чувствительность чтения $J_S$ приближалась к транспорту якобианов старшей (Qwen3.5-4B). Гипотеза: «эластичные» (ограниченные по норме) правки улучшают чтение без разрушения модели. Установка метрик: **KL к оригиналу — поводок (ограничение), не цель**; цели — сеньорные метрики; когерентность — оконная метрика (union top-n обеих моделей).

## 1. Архитектура запуска

**Stage 0 — `exp16_stage0_targets.py`** (один раз, дорого, можно на CPU ночью):
Для каждой пары слоёв `(l_L, l_S)` из `uniform_layer_mapping(n_L, n_S)` (шаг как в exp11c, 7 пар), для `n_samples` калибровочных сэмплов:
```python
J_L = compute_jacobian_for_tokens(mL, l_L, ids, attn, token_set)     # K×D_L
J_tgt = ((J_L @ E_L.T) @ E_S) @ G_S_inv                              # транспорт в пространство S
J_tgt_p = softmax_jacobian_like(J_tgt, p)                            # gauge-free: p[:,None]*(J - (p[:,None]*J).sum(0,True))
J_S_p   = softmax_jacobian_like(compute_jacobian_for_tokens(mS, l_S, ...), p)
torch.save({"J_tgt_p": ..., "J_S_p": ..., "p": ..., "pair": ...}, f"targets/{pair}_{si}.pt")
```
`token_set` = top-K логитов старшей на первом сэмпле (K=256 GPU / 128 CPU), `p` = softmax логитов старшей на `token_set`. `G_S_inv = (E_S^T E_S)^{-1}` (float64 для обратимости).

**Stage 1 — exp16/17/18/19**: читают кэш, гоняют только малую модель → влезают в CPU и в одну T4.

## 2. Kaggle-упаковка

```
kaggle_kernel/
├── kernel-metadata.json   # ПОКА НЕ ТРОГАТЬ
├── kernel2.ipynb              # оркестратор: device detect → конфиг → stage0 (если нет кэша) → 16→17→18→19
├── kernel2.ipynb              # оркестратор: device detect → конфиг → stage0 (если нет кэша) → 16→17→18→19
├── exp16_*.py ... exp19_*.py
└── prism/                 # пакет целиком + prism/metrics.py (новый, см. §4)
```
Модели: `/kaggle/input/qwen35-08b`, `/kaggle/input/qwen35-4b`, иначе HF-хаб. Кэш таргетов: если в `/kaggle/input/prism-targets/` нет — считать в Stage 0 и сохранить в `/kaggle/working/targets/` (переживёт рестарт сессии как output). CLI: `kaggle kernels push -p kaggle_kernel`; `kaggle kernels status ...`; `kaggle kernels output ... --force`.

## 3. Устройство-конфиг и фоллбек

| Параметр | GPU (T4/P100) | CPU |
|---|---|---|
| dtype малой | fp32 | fp32 |
| n_samples | 4 | 2 |
| K (token set) | 256 | 128 |
| α-свип Exp16 | {0.01,0.03,0.1,0.3,1.0} | {0.03,0.1,0.3} |
| GD steps | 100 | 40 |
| Exp17 порядки | late→early И early→late | только late→early, пары [23,15,7] |
| Exp18 GSM8K | 20 задач few-shot | пропуск (только метрики батареи) |

Runtime-guard: замер первого якобиана малой; если >60 c — один раз уменьшить K и n_samples вдвое, WARNING. Чекпоинты: JSON пишется после каждого эксперимента и после каждого слоя Exp17. `torch.manual_seed(42)`, `attn_implementation="eager"`.

**Реализовано:** `exp16_stage0_targets.py` с `_gpu_config()` (n_samples, K, dtype), runtime-guard, checkpoint-файл `.checkpoint.json`, `torch.manual_seed(42)`, `attn_implementation="eager"` во всех файлах.

## 4. Общий модуль `prism/metrics.py` — батарея метрик

- [x] Статус выполнения

`eval_battery(m_edited, m_orig, mL, prefixes) -> dict`:
- `A_L`, `A_S` = top-15 **и** top-p 0.9 (оба варианта репортить) логитов старшей / оригинальной малой на префиксах; `U = A_L ∪ A_S`; контекст: `|A_S ∩ A_L|`, `|U|`.
- Цели: `bridge_per_pair` (на кэш-таргетах, gauge-free), `KL_ed_L` = KL(p_ed ‖ p_L), `P_ed(A_L)`, средний ранг top-10 старшей в p_ed (должен падать).
- Когерентность: `hit5_in_U` (доля top-5 p_ed, попавших в U — «попало в любые указанные = хорошо»), `P_ed(U)`.
- Поводок: `KL_ed_orig`, PPL на 2 натуральных последовательностях.
- Поведение (GPU): GSM8K few-shot accuracy.

## 5. Exp16 — пилот без обучения: readout-only pull

- [x] Статус выполнения

Файл: `experiments/exp16_readout_pull.py`. Правим **только `lm_head`**: `W' = W + ΔW`, `ΔW=0`, Adam lr 1e-3, лосс = bridge(J_tgt_p, J_S_p(θ)) + 0.1·KL(p_θ‖p_orig); после каждого шага проекция trust-region: `if ‖ΔW‖_F > α·‖W‖_F: ΔW *= α·‖W‖_F/‖ΔW‖_F`. Свип по α. CPU: α={0.03,0.1,0.3}, steps=40. GPU: α={0.01,0.03,0.1,0.3,1.0}, steps=100. JSON: `per_alpha: {alpha, battery...}, alpha_star`.

## 6. Exp17 — послойная LoRA-дистилляция моста

- [x] Статус выполнения

Файл: `experiments/exp17_lora_distill.py`. LoRA r=4 на `lm_head` + `o_proj` целевого слоя. Порядок: late→early (всегда), early→late (GPU). CPU: шаги=40, пары subset [23,15,7]. GPU: шаги=100. После каждого слоя: freeze → update orig logits cache → следующий слой.

## 7. Exp18 — контроль: logit-KD при том же бюджете

- [x] Статус выполнения

Файл: `experiments/exp18_logit_kd.py`. Та же LoRA-конфигурация и шаги, лосс = KL(p_θ ‖ p_L) на префиксах (без bridge). Сравнение: **Δbridge на единицу ΔKL_ed_orig** у Exp17 vs Exp18. Автоматически читает результаты exp17 и считает ratio.

## 8. Exp19 — калибровка детектора разрушения

- [x] Статус выполнения

Файл: `experiments/exp19_destruction.py`. На α* из Exp16 четыре правки (последний слой): (a) random ΔW той же нормы; (b) 10×α*; (c) gauge component; (d) differential component. Ранжирование по bridge.

## 9. Порядок и отчёт

- [x] Статус выполнения

Файл: `experiments/run_bridge_exps.py` — оркестратор полного пайплайна (Stage0 → 16 → 19 → 17 → 18). Опции `--skip-stage0`, `--skip-exp16` и т.д. Итоговый print: таблица «сценарий × батарея» + α* + отношение Δbridge/ΔKL для 17 vs 18.

**Интерпретация:** (1) pull улучшает и bridge, и поведение → sensitivity distillation как новая KD; (2) bridge падает, поведение стоит → проблема малой в репрезентациях, не в readout; (3) ломается при малых α → readout жёстко связан с репрезентациями, мост — диагностик, не скальпель. Все три публикуемы.

Готов уточнить любой пункт, если агент упрётся в размерности (особенно `G_S_inv` и проекцию trust-region).

## 10. Дополнение статьи

- [x] Статус выполнения

Файл: `docs/exp10_article_supplement.md`. Методология, транспорт якобианов, батарея метрик, device-конфигурация, интерпретация результатов.