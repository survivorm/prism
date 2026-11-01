# Ревью E-39c_pre v4 (round 4): α-grid extended + per-α κ' + trade-off

## 1. Round 4 fixes — verified ✅

| Fix | Check |
|---|---|
| α-grid extended below 0.1 (0.01, 0.02, 0.05, 0.08) | ✅ 11 точек |
| Per-α κ' dry-run | ✅ `kappa_dryrun` в каждой точке |
| trade-off computed | ✅ `trade_off` block |
| QR-blend removed | ✅ `pert_col_observed ≈ pert_col_expected` точно |
| `rho_noise` convention aligned | ✅ `0.01416` matches E-39a |
| `V_c` rank check | ✅ `v_c_rank_ok: true` |
| `V_new_perp` computed once | ✅ in closure |
| a_random consistency check | ✅ `abs_diff = 0.00024` |

**Механически всё правильно.**

---

## 2. Главный результат — **regions DISJOINT, отрицательный результат окончателен**

### 2.1. Данные (round 4)

| α | KL(t≥1) | row_diag | pert_col / theory | κ' | KL<0.05 | κ'<0.005 |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0 / 0 | 0.0159 | ✓ | ✗ |
| **0.01** | 0.0043 | 0.073 | 0.1001 / 0.1001 | 0.0157 | ✓ | ✗ |
| 0.02 | 0.0086 | 0.104 | 0.1418 / 0.1418 | 0.0156 | ✓ | ✗ |
| 0.05 | 0.0227 | 0.164 | 0.2250 / 0.2250 | 0.0151 | ✓ | ✗ |
| **0.08** | 0.0401 | 0.208 | 0.2858 / 0.2858 | 0.0148 | ✓ | ✗ |
| 0.1 | 0.0628 | 0.233 | 0.3204 / 0.3204 | 0.0147 | ✗ | ✗ |
| 0.2 | 0.3864 | 0.334 | 0.4595 / 0.4595 | 0.0134 | ✗ | ✗ |
| 0.5 | 2.0217 | 0.551 | 0.7654 / 0.7654 | 0.0099 | ✗ | ✗ |
| 0.7 | 2.4293 | 0.679 | 0.9511 / 0.9511 | 0.0076 | ✗ | ✗ |
| 0.9 | 3.7836 | 0.825 | 1.1694 / 1.1694 | 0.0053 | ✗ | ✗ |
| **1.0** | 4.9045 | 0.982 | 1.4142 / 1.4142 | 0.0041 | ✗ | ✓ |

### 2.2. **Blend теперь идеально чистый** ✅

`pert_col_observed_mean ≈ pert_col_observed_max ≈ pert_col_expected` **для всех α** — все 16 колонок имеют **идентичную** пертурбацию `√(2−2√(1−α))`. **QR-артефакт (Householder sign-flips) устранён.** Это подтверждает round-3 фикс.

**Следствие:** `row_diag` теперь **монотонен** (0 → 0.982). `KL` тоже **монотонен** (0 → 4.905). Никаких dips в α=0.1 или α=0.7. **Round-3 "genuine loss-landscape" гипотеза опровергнута — non-monotonicity была QR-артефактом, а не свойством loss landscape.** Спасибо за честное retraction.

### 2.3. **Trade-off — окончательный отрицательный результат**

**KL-safe α:** `[0.01, 0.02, 0.05, 0.08]` — все малые.
**κ'-safe α:** `[1.0]` — только полная surgery.
**Both safe:** `[]` — **пусто.**

Проверка magnitude:

- При α=0.08 (max KL-safe): κ' = 0.01477 = **3.0× target** (0.005)
- При α=0.9 (near κ'-safe): KL = 3.78 = **76× budget** (0.05)

**Regions полностью disjoint с огромным margin (3× и 76×).** Это не "тонкая грань", это **фундаментальная несовместимость** в данной конструкции.

---

## 3. Критический inconsistency — **JSON не соответствует коду** (HIGH)

### 3.1. Факт

**.py файл** (line ~740):
```python
"budget_note": (f"Small α PASSES the KL budget: KL<{KL_PASS} for "
                f"α ∈ {kl_safe_alphas} (min KL = {min_kl_nonzero:.4f} "
                f"= {min_kl_nonzero / KL_PASS:.2f}× the budget at "
                f"α={min_alpha_nonzero}). But at those small α, "
                f"κ' ≈ {sweep[1]['kappa_dryrun']:.4f} "
                f"(~{sweep[1]['kappa_dryrun'] / KAPPA_TARGET:.0f}× the "
                f"{KAPPA_TARGET} target) FAILS the flattening side. ...")
```

**Ожидаемый вывод:**
> "Small α PASSES the KL budget: KL<0.05 for α ∈ [0.01, 0.02, 0.05, 0.08] (min KL = 0.0043 = 0.09× the budget at α=0.01). But at those small α, κ' ≈ 0.0157 (~3× the 0.005 target) FAILS the flattening side. ..."

**JSON** (что фактически в файле):
> "min KL over α>0 = 0.0043 = 0.09× the 0.05 budget at α=0.01 — **FAILS**."

### 3.2. Проблема

Это **два разных текста**. И JSON-версия **внутренне противоречива**: "0.09× the budget" = 0.09·0.05 = 0.0045 < 0.05 = **проходит бюджет**. Но текст говорит "**FAILS**". Reader увидит "0.09× under budget — FAILS" и решит, что либо budget_note сломан, либо 0.05 не является threshold.

**Причина:** JSON — от предыдущей версии кода (round 4 partial fix). .py — уже обновлён (round 4 final). Скрипт был переписан, но **не перезапущен** с обновлённым кодом — JSON остался от промежуточного прогона.

### 3.3. Severity

- **Paper-level HIGH:** если paper цитирует `budget_note` из JSON, получит противоречивое утверждение.
- **Код-level MEDIUM:** сам код правильный, но output stale.

**Fix:** перезапустить скрипт с текущим кодом. JSON обновится.

**Проверка после перезапуска:** `budget_note` должен начинаться с "Small α PASSES the KL budget", не "min KL ... FAILS".

---

## 4. Средние замечания

### 4.1. `runtime_s = 1285s` (~21 мин) — iteration стоимость (MEDIUM)

Round 3: 305s. Round 4: 1285s. **4.2× рост** из-за:
- 4 дополнительных α-forward (275s → 375s)
- 11 × `_kappa_dryrun` (каждый ~40-50s = 500s)
- 3 × per-strategy `_kappa_dryrun` = 150s
- Остальное = KL decomposition + base

**Проблема:** `_kappa_dryrun` вызывается **11 раз** на α-sweep. Каждый вызов SVD 32×128 раз.

**Fix (опционально, non-blocking):** κ'(α) — гладкая функция от α. Можно измерить только на 4-5 точках (α ∈ {0.01, 0.1, 0.5, 0.9, 1.0}) и интерполировать. Но для paper лучше полный grid, так что это trade-off.

### 4.2. `kl_crossing_alpha = 0.01` — misnomer (LOW)

Семантика: `min(kl_safe_alphas) = 0.01` — smallest α that PASSES, но это не "crossing point" (crossing = где KL=0.05, between α=0.08 и α=0.1). Reader может подумать, что α=0.01 — точка пересечения budget.

**Fix:** переименовать `kl_crossing_alpha` → `kl_min_passing_alpha` или добавить `kl_budget_crossing_estimate: ~0.085` (интерполяция).

### 4.3. V_new_perp endpoint (α=1) KL = 4.90 vs raw V_new full = 5.02 (LOW)

Разница 2.4% — малая, но в paper §4.4 это два разных числа. `endpoint_note` документирует. OK, но нужна ясность: **какой α используется в headline full-surgery?**

- `kl_full_nats_per_token.mean_real_tokens_t_ge_1 = 5.017` — это **raw V_new** (α=1, orthogonalize=False)
- `alpha_blend_sweep.points[α=1].kl_t1 = 4.904` — это **V_new_perp** (α=1, orthogonalize=True)

Оба infeasible, но для consistency paper должен явно цитировать **5.02** для full surgery и **4.90** для orthogonalized α=1 endpoint. **Сейчас это только в `endpoint_note`, что может быть пропущено.**

### 4.4. `sweep[1]['kappa_dryrun']` в `budget_note` — fragile (LOW)

```python
f"κ' ≈ {sweep[1]['kappa_dryrun']:.4f} "
```

`sweep[1]` — hardcoded index для α=0.01. Если grid изменится, это станет α=0.02 (или что-то ещё). Хрупко.

**Fix:** явно `sweep_by_alpha = {p["alpha"]: p for p in sweep}; alpha01_kap = sweep_by_alpha[min_alpha_nonzero]["kappa_dryrun"]`.

### 4.5. `V_a` consistency — очень хорошо, но `KAPPA_39A_FULL = 0.0042` (LOW)

`a_random.kappa_dryrun = 0.003965` vs `KAPPA_39A_FULL = 0.0042`. Diff = 0.00024. Откуда 0.0042? Из E-39a JSON: a_random kappa = 0.00396, kappa_median_all_triples = 0.00399. Ни одно не 0.0042. 0.0042 — вероятно старая quoted value.

**Fix:** установить `KAPPA_39A_FULL = 0.00396` (actual a_random from E-39a JSON) или явно документировать rounding. Diff 0.00024 малый, не критично.

---

## 5. Что paper может утверждать (окончательно)

### 5.1. Strong claim (все confirmed):

1. **Full M_K-surgery (α=1, raw V_new):** KL = **5.02** nats, **100×** над бюджетом. Row damage 98%. **Catastrophic.**
2. **Tied-weights decomposition:** embed_only = 4.01 (80%) >> head_only = 1.24 (20%). Hidden drift 87.7%. **Surgery fundamentally non-readout-only.**
3. **All 3 V_new strategies fail at α=1:** 4.71–5.51. **Structural, not V_new-specific.**
4. **Untied head-only cost = 1.24** (25× budget) — даже лучший случай далеко.
5. **α-blend verified clean** (round-3 QR-artifact retracted): monotone row_diag, monotone KL, per-column perturb = theory exactly.
6. **Trade-off disjoint:** KL-safe α ≤ 0.08; κ'-safe α = 1.0; **crossing point непроходим**. Max KL-safe α=0.08 gives κ'=0.0148 (3× target); κ'-safe α=0.9 gives KL=3.78 (76× budget).

### 5.2. Окончательный verdict

> **E-39c_pre (definitive negative):** M_K-surgery at L23 is infeasible both in full rotation and in the tested partial-α family. Full surgery: KL = 5.02 (100× budget), tied-weight dominant embed term (4.01, 80%), hidden drift 88%. Clean α-blend: KL-safe region [α ≤ 0.08] and κ'-safe region [α = 1.0] are **disjoint** — at max KL-safe α=0.08, κ' = 0.0148 (3× the 0.005 target); at min κ'-safe α=1.0, KL = 4.90 (98× budget). **The route to L23 leaf flattening via M_K-surgery is refuted within the tested construction.** A conceptually different approach (e.g., LoRA training with flatness loss, E-39d) is required.

---

## 6. Итог

**Блокеры:**

1. **§3.1** — JSON stale / mismatch с .py. `budget_note` в JSON содержит противоречивый "0.09× the budget — FAILS". Перезапустить скрипт.

**Средние:**

2. **§4.2** — `kl_crossing_alpha` naming misleading. Fix: rename или add interpolation estimate.
3. **§4.4** — `sweep[1]` fragile index в budget_note. Fix: lookup by value.

**Мелочи:**

4. **§4.1** — runtime 21 min (documentation, non-blocking).
5. **§4.3** — 5.02 vs 4.90 (paper consistency).
6. **§4.5** — `KAPPA_39A_FULL=0.0042` vs actual 0.00396.

**Оценка: 8.5/10.**

- **Blend cleanup** — отлично. QR-artifact retraction honest.
- **Trade-off** — mathematically conclusive negative result.
- **Runtime** — выросло до 21 min.
- **JSON/code mismatch** — единственный блокер, лёгкий fix.

**Главное достижение цепочки:** E-39a → E-39c_pre v1 → v2 → v3 → v4 — это textbook-quality iterative refinement. Каждый round поймал важный дефект:
- v1: `M_kp/g` bug (round 1)
- v2: tied-weights decomposition (round 2)
- v3: QR-blend artifact (round 3)
- v4: α-grid extension + per-α κ' (round 4)

**Для paper v6.0:** результат **conclusive negative** — surgery refuted, следующая ставка должна быть E-39d (LoRA training) с flatness-loss, которая **не модифицирует embed** (LoRA на отдельных слоях вместо full W_lm edit). Это единственный оставшийся path.