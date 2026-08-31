# p3_tz.md — P3: Bridge loss как регуляризатор при fine-tuning

> **Куда запускать:** Kaggle GPU (T4×2 или P100).
> **Отличие от P2:** P2 (Exp16–19) = только bridge loss. P3 = стандартный NTP loss + bridge loss как регуляризатор.
> **Зависимости:** `prism/`, `kaggle_kernel/` (calibration.py, bridge.py, jacobian.py).

---

## Серия P3: Fine-tuning с bridge loss регуляризатором

### P3.1: Baseline — plain fine-tuning (без bridge)

**Что:** Fine-tune 0.8B с LoRA на 100 сэмплах C4, только NTP loss. **Не дублирует** ничего из P2.

**Параметры:**
- LoRA: rank=8, alpha=16, target_modules=['q_proj', 'v_proj']
- Data: 100 сэмплов C4 (seq_len=64)
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Steps: 200 (2 эпохи, batch=1)
- Eval: PPL на 16 validation сэмплах

**Код:** `./kaggle_kernel/p3_exp1_baseline_ft.py`

---

### P3.2: Fine-tuning с bridge loss на 1 слое

**Что:** L_total = L_NTP + λ·L_bridge(layer 12). λ=0.1.

**Почему 1 слой:** Jacobian computation дорогой (~158 сек на CPU, ~5-10 сек на GPU). 1 слой — компромисс.

**Код:** `./kaggle_kernel/p3_exp2_ft_bridge_1layer.py`

---

### P3.3: Fine-tuning с bridge loss на 4 слоях

**Что:** L_total = L_NTP + λ·sum(L_bridge(l)) для l ∈ {6, 12, 18, 23}. λ=0.05.

**Код:** `./kaggle_kernel/p3_exp3_ft_bridge_4layers.py`

---

### P3.4: λ grid search

**Что:** λ ∈ {0.01, 0.05, 0.1, 0.5}. 4 слоя. Steps=100.

**Код:** `./kaggle_kernel/p3_exp4_lambda_search.py`

---

### P3.5: Сравнение P2 vs P3

**Что:** PPL после P2 (Exp17, bridge-only) vs PPL после P3 (NTP + bridge) vs baseline (P3.1).

**H4 критерии:**
- P3 PPL < P3.1 PPL → bridge-регуляризатор **полезен** → H4 подтверждена
- P3 PPL ≈ P3.1 PPL → bridge-регуляризатор **нейтрален** → H4 не подтверждена
- P3 PPL > P3.1 PPL → bridge-регуляризатор **вредит** → H4 опровергнута

**Код:** `./kaggle_kernel/p3_exp5_comparison.py`

---

## Порядок и стоимость

| Эксперимент | Что | Зависит от | GPU время |
|---|---|---|---|
| P3.1 | Baseline FT | — | ~30 мин |
| P3.2 | FT + bridge, 1 слой | P3.1 | ~2 ч |
| P3.3 | FT + bridge, 4 слоя | P3.2 | ~6 ч |
| P3.4 | λ search | P3.3 | ~18 ч (4 λ × 100 steps) |
| P3.5 | Сравнение | P2.4, P3.4 | ~1 ч |

**Итого:** ~27 часов GPU времени.

**Параллельность:** P3.1–P3.2 запустить параллельно с P2 (Stage 0, Exp16). P3.3–P3.5 — после P3.2 + P2.5.
