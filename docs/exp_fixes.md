# Исправления экспериментов (код + запуски)

## 🔷 Требуют Kaggle T4×2 (4B + 0.8B)

### E2. CKA identity-контроль ✅ КОД ГОТОВ
**Файл:** `experiments/exp2a_cka_identity.py` — новый
**Что:** CKA(4B→4B) ≈ 1 и CKA(0.8B→0.8B) ≈ 1
**Один запуск.** Если ≈ 1 → CKA работает.
```bash
# На Kaggle (T4×2):
cd /home/survivor/work/prism
python experiments/exp2a_cka_identity.py --n-samples 5
```
**Результат →** `results/exp2a_cka_identity.json`

### E4. top-p support + random-negative токены ✅ КОД ГОТОВ
**Файл:** `experiments/exp4_top_p_tokens.py` — новый
**Что:** uniform vs top-p=0.92 vs random токены
**Один запуск.**
```bash
# На Kaggle (T4×2):
cd /home/survivor/work/prism
python experiments/exp4_top_p_tokens.py --n-samples 4
```
**Результат →** `results/exp4_top_p_tokens.json`


### E7. N-апгрейд ключевых экспериментов 📋 КОМАНДЫ
**Статус:** Код готов, нужно просто `--n-samples 16` или `32`:
```bash
# CPU (0.8B only):
.venv/bin/python experiments/exp2_heatmap.py --n-samples 16
# Kaggle (4B + 0.8B):
python experiments/exp10d_common_mode.py --n-samples 16
python experiments/exp13_domain_esg.py --n-samples 16
python experiments/exp15_gdn_anchors.py --n-samples 16
```

---

## Инфраструктура

### I1. T4×2 setup
```
cuda:0: 4B fp32
cuda:1: 0.8B fp32
```

### I2. jacrev/vmap вместо цикла по |U|
- Заменить manual token loop на `jax.jacrev` или `torch.func.jacrev` + `vmap`
- Ускоряет ×|U| (~200× при |U|=64–128)

---

## Приоритетный порядок
2. **E2+E4** → Kaggle T4×2
3. **E7** → N-апгрейд (массовый, требует времени)
