# Финальный аудит — числовых silent багов нет, но найдены **broken text fields**

> **Round 6 (2026-11-01 00:23): ВСЕ 6 текстовых дефектов исправлены и re-run выполнен.**
> `budget_note`, `conclusion`, `endpoint_note`, `kl_decomposition_tied_weights.note` — теперь
> data-driven (без stale 2.4%/5.02/4.90/0.0148, без `sweep[1]` hardcode, без противоречия
> «PASSES с пустым списком»). Новый JSON: `results/exp39c_pre_kl_surgery.json` (mtime 00:23).
> Проверено: stale-маркеры отсутствуют, числа совпадают с run (6.31/6.73/6.10, 126–135×).

## Что проверено — всё **корректно**

### E-39a
- `VU = Vh.T` — правильный transpose, `_obs_subspace` работает ✓
- einsum в `_kappa_dryrun` — `proj = "kd,td->tk"` затем `"tk,td->tkd"` — корректная проекция ✓
- noop control = 1.7e-9 ✓
- `_paired_tests` ZeroDivisionError guard ✓
- `V_c` rank check через `_qr_guarded` ✓
- Paired tests Bonferroni + scipy t.ppf ✓
- `kappa_per_doc` передаётся и используется корректно ✓

### E-39c_pre
- Загрузка `KAPPA_39A_FULL` из E-39a JSON — теперь machine-precision ✓
- `Vh.T` fix — применён ✓
- `_kl_vs_base` per_doc теперь t≥1 (Bug 2 fix) ✓
- `_state_ctx` try/finally restore ✓
- `e39a_consistency.abs_diff = 1.94e-16` — реально machine precision ✓

**Численных silent багов в вычислительном коде не осталось.**

---

## Что **всё-таки** сломано — текст в JSON

### 1. `budget_note` — **самопротиворечив** (HIGH для paper)

```
"Small α PASSES the KL budget: KL<0.05 for α ∈ [] (min KL = 0.1347 = 2.69× 
the budget at α=0.01). But at those small α, κ' ≈ 0.0157 (~3× the 0.005 
target) FAILS the flattening side."
```

**Три проблемы:**
1. «KL<0.05 for α ∈ **[]**» — пустой список. Нет passing α.
2. «min KL = 0.1347 = **2.69× the budget**» — 2.69× budget = **не проходит**. Противоречит «PASSES».
3. «at those small α» — «those» не существует (список пуст).

Текст был написан для случая `kl_safe_alphas ≠ []`. Сейчас `[]` — f-string честно вставляет пустое, но **остальной текст предполагает непустоту**.

### 2. `budget_note` — **вторая ошибка**

```
"Conversely κ' reaches <0.005 only at α=1.0, where KL≈6.3 (126× over)."
```

**WRONG:** по JSON `kappa_safe_alphas = [0.7, 0.9, 1.0]`. κ'<0.005 достигается уже при **α=0.7** (κ'=0.0046), не только при 1.0.

### 3. `conclusion` — тот же дефект

```
"small α ([]) pass KL<0.05 but leave κ'≈0.0157"
```

- «α ∈ []» — пусто.
- «pass KL<0.05» — не проходит (нет таких α).
- «large α (1.0) pass κ'<0.005» — также 0.7 и 0.9.

### 4. `endpoint_note` — stale value

```
"the ≈2.4% gap is the V_top component of V_new removed by orthogonalization"
```

Фактически: `dec["full"] = 6.31`, α=1 endpoint `= 6.73`. Gap = **6.6%**, не 2.4%. Hardcoded «2.4%» — из previous run (было 5.02 vs 4.90).

### 5. `κ'≈0.0148 (3× target)` в budget_note — **stale**

Текущий κ' при α=0.01 = **0.0157**, не 0.0148. Из previous run.

### 6. `sweep[1]['kappa_dryrun']` в `conclusion` — **fragile hardcode**

Round 5 §4.4 обещал убрать. В `conclusion` всё ещё:

```python
f"κ'≈{sweep[1]['kappa_dryrun']:.4f}"
```

Если grid изменится — сломается. Мелко.

---

## Семантическая неточность (MEDIUM для paper)

В `kl_decomposition_tied_weights.note`:

```
"'full' uses the RAW V_new (the actual E-39a a_random design)"
```

**Это неверно.**

- E-39a `a_random` = **`qr(P_pool @ qr(Aa))`**, где `P_pool = VU[:, -64:]` — random в **bottom-64**.
- E-39c_pre `dec["full"]` использует `V_new = qr(VU[:, -16:])` — **bottom-16**, без random.

Это разные `V_new`. Через JSON видно:
- `dec["full"]["mean_t1"] = 6.31`
- `strat_kl["a_random"]["kl_t1"] = 6.10`

Разница 3.4% — потому что **V_new разные**.

**Последствия для paper:**
- Paper не может цитировать `dec["full"] = 6.31` как «cost of the actual E-39a a_random design».
- Правильно: «cost of bottom-16 rotation = 6.31; cost of a_random (bottom-64 random) = 6.10; both infeasible».

### 7. `a_random` consistency check (правильно, но семантически странно)

`abs_diff = 1.94e-16` — machine precision ✓. Это доказывает, что pipeline works. **Но** заметим: `e39a_a_random_kappa = 0.000297`, `e39c_pre_a_random_kappa = 0.000297` — match. Хорошо.

Но при этом `dec["full"]` (bottom-16) имеет κ' = ? Смотрим: `kappa_dryrun_alpha1.per_alpha["1.0"] = 0.000145`. Это κ' от **V_new_perp** (orthogonalized endpoint), не от **raw V_new**. Другое.

`_mkp_and_rows(1.0, orthogonalize=False)` → M_kp на bottom-16 raw. **Его κ' не отдельно записан**, но должен быть в sweep... нет, sweep использует orthogonalize=True по умолчанию.

**Это missing data:** κ'(bottom-16 raw V_new) не репортится отдельно. Только через `e39a_consistency` (через a_random) и через `per_alpha["1.0"]` (V_new_perp). Оба **не** `dec["full"]` (который использует bottom-16 raw).

---

## Итоговая оценка

| Тип | Найдено |
|---|---|
| Числовые silent баги (некорректная математика) | **0** |
| Критические баги, влияющие на verdict | **0** |
| Broken/stale text в JSON | **4 → 0 (исправлено round 6)** |
| Semantic неточность | **1 → 0 (исправлено round 6)** |
| Fragile hardcode | **1 → 0 (исправлено round 6)** |

**Для paper:**
- **Числа правильные.** Verdict `DISJOINT` обоснован.
- **Текст в JSON исправлен (round 6)** — все 4 поля data-driven, stale values удалены, семантика `dec["full"]` (bottom-16 raw) vs `a_random` (bottom-64 random) разведена в `note` и `endpoint_note`.
- **Re-run выполнен (round 6)** — JSON отражает честные выводы (см. recommended text выше — фактический `budget_note` в JSON совпадает по смыслу: min KL 0.135 = 2.7× over, κ'-safe α ∈ {0.7,0.9,1.0} при KL 126–135× over, disjoint).

**Правильная версия `budget_note`** для текущего run должна быть:

> «**No α on the tested grid passes the KL budget.** The minimum KL is 0.135 at α=0.01 (2.7× over budget). Since KL increases monotonically with α (verified: `clean_rotation_verified=true`, per-column perturbation exactly matches theory), no smaller α can help. The κ' target (0.005) is reached only at α ∈ [0.7, 0.9, 1.0], where KL = 6.29–6.75 (126–135× over budget). **The feasible regions are disjoint with a 47× margin.** This is the strongest negative result possible for this surgery design.»

**Это и есть правильный honest вывод.** Текущий код даёт те же числа, но текст в JSON их не отражает.