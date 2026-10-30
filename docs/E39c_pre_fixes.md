# Ревью E-39c_pre v3 (round 2 fixes applied)

## 1. Все мои round-2 fixes — верифицированы ✅

| Fix | Status |
|---|---|
| §2 α-blend orthogonalization | `V_new_use = V_new - V_top @ (V_top.T @ V_new)` ✓ |
| §3 `V_c` rank check | `V_c_valid` + `warnings.warn` ✓ |
| §4.1 `‖V_new.T V_top‖_F` | `vnew_vtop_ov = 0.533` в JSON ✓ |
| §4.2 per-α `‖V_mixed.T V_top‖_F` | в `points[].v_mixed_vs_v_top_fro` ✓ |
| §4.3 hidden_drift t=0 exclusion | `real[:, 0] = False` ✓ |
| §4.4 `try/finally` via contextmanager | `_state_ctx` ✓ |
| §4.5 α=1 endpoint docstring | `endpoint_note` ✓ |

**Механически fix правильный.** `v_mixed_vs_v_top_fro` теперь **монотонен** от 4.0 (α=0) к 0.0 (α=1) — QR-коллапс устранён. Хорошо.

---

## 2. Но: два бага, подрывающих вывод «non-monotonicity is genuine» (HIGH)

### 2.1. `rho_noise` — **разная нормализация** между E-39a и E-39c_pre (MEDIUM, silent)

**E-39a** (v_new_strategy_sensitivity, тот же код):
```python
ov = (u64 @ V_new).pow(2).sum(dim=-1).mean().item()  # mean over [N, S]
```
→ `a_random: 0.0142`

**E-39c_pre** (сейчас):
```python
ov = float((u.double() @ Vn).pow(2).mean())  # mean over [N, S, R]
```
→ `a_random: 0.000885`

**Отношение = 16 = R.** `0.0142 / 16 = 0.000888`. **Одинаковое значение, разная нормализация.**

**Проблема:** метрика называется одинаково (`rho_noise_u_obs_overlap_mean`), но в E-39a это `E[‖û_tᵀ V_new‖_F²]` (норма на позицию), а в E-39c_pre — `E[(û_tᵀ V_new)²]` (средний квадрат элементов). Reader сравнивающий «E-39a говорит 0.014, E-39c_pre говорит 0.0009» решит, что стратегия стала в 16× чище.

**Fix:** либо выровнять (использовать `.sum(dim=-1).mean()` в E-39c_pre), либо явно переименовать (`rho_noise_per_element` vs `rho_noise_fro_per_position`).

### 2.2. **`row_diag` при α=0.1 противоречит теоретическому ожиданию** (HIGH, silent)

**Теория:** для ортогонализованного blend, `V_mixed(α) = √α V_new_perp + √(1−α) V_top`, с `V_new_perp ⊥ V_top`:
```
‖V_mixed(α)[:,j] − V_top[:,j]‖ = √(α + (√(1−α)−1)²) = √(2 − 2√(1−α))
```
- α=0.1: **0.320** (малый rotation)
- α=0.5: **0.766**
- α=1.0: **1.414**

**Ожидаемый row change** = `0.32·c` при α=0.1, где `c ≈ ‖U_top Σ_top‖_F/‖M_K‖_F ≈ 0.9-1.0`. Ожидание: **≈0.29–0.32**.

**Фактически в JSON:**
| α | `mean_dW/row` | Ожидание | Отношение |
|---|---|---|---|
| 0.1 | **0.772** | ~0.30 | **2.5×** |
| 0.2 | 0.767 (dip!) | ~0.42 | 1.8× |
| 0.5 | 0.840 | ~0.70 | 1.2× |
| 0.7 | 0.828 (dip!) | ~0.87 | 0.95× |
| 0.9 | 0.886 | ~0.97 | 0.91× |
| 1.0 | 0.982 | ~0.98 | 1.0× ✓ |

**Два аномальных сигнала:**
1. **При α=0.1 row change в 2.5× больше ожидаемого** — это не «малый rotation», это большой.
2. **Немонотонность:** α=0.1 → 0.772, α=0.2 → 0.767 (dip), α=0.5 → 0.840, α=0.7 → 0.828 (dip). Ожидание — монотонный рост.

**Причина (гипотеза):** `V_new_perp = QR(V_new − V_top·V_topᵀ·V_new)` — если у V_new есть колонки с большим overlap с V_top, то после проекции эти колонки **почти нулевые**, и QR заполняет их **произвольными** направлениями в ортогональном дополнении. Эти «произвольные» направления при α=0.1 получают вес √0.1 ≈ 0.32 — и в `M_kp` они замещают часть исходных колонок M_K.

**Эффект:** α=0.1 — не «малая ротация», а «90% V_top + 10% случайного направления в ортогональном дополнении». `mean_dW/row = 0.772` потому что **одна или несколько колонок V_new_perp имеют высокий overlap с исходными направлениями M_K** (несмотря на ортогональность V_new_perp к V_top, они могут быть близки к `U_rest` направлениям).

### 2.3. Следствие для интерпретации

JSON утверждает:
> «Even with the corrected orthogonalized blend, the KL is non-monotonic... this is **NOT** a blend/QR artifact... it is a **genuine loss-landscape property**: a small partial rotation near the top subspace is MORE destructive than the full rotation.»

**Это overclaim.** Правильная интерпретация:

> «KL non-monotonicity persists after orthogonalization, but **row_diag is also non-monotonic** (0.772 at α=0.1 vs 0.982 at α=1, dips at α=0.2 and α=0.7). The expected row change for a clean orthogonalized blend at α=0.1 is ~0.30; observed 0.772. **The blend is NOT producing a "small" rotation at small α** — it's producing a large row change because `V_new_perp` (via QR of the projected V_new) is not a small perturbation of V_top. The non-monotonicity may be a property of `V_new_perp`'s structure, not a "loss landscape" property of partial rotation per se.»

**Это важное различие для paper:**
- Текущий claim: «**любая** partial surgery несовместима с behavior»
- Более слабый (но honest) claim: «partial surgery на этой конкретной `V_new_perp` конструкции несовместима с behavior»

Второй допускает, что если найдётся «правильная» V_new-конструкция (например, `V_new` с контролируемой ортогонализацией), partial surgery могла бы работать.

### 2.4. Fix — верификация row_diag

**Диагностика** (2 строки в `_mkp_and_rows`):
```python
# в blend branch:
V_pert = V_mixed_col - V_top_col
per_col_norm = V_pert.norm(dim=0)  # [R]
# ожидание: √(2-2√(1-α)) для каждой колонки
expected = (2 - 2*np.sqrt(1-alpha))**0.5
# если max(per_col_norm) > 1.5*expected → QR fill для degenerate columns
```

Добавить в JSON:
```python
"alpha_blend_sweep": {
    "per_alpha_diagnostics": [
        {"alpha": 0.1, "expected_pert_norm": 0.32, 
         "observed_pert_norm_mean": ..., "observed_pert_norm_max": ...}
    ]
}
```

Если `observed_pert_norm_max >> expected` → **V_new_perp нестабильна**, и «non-monotonicity» — artefact этой нестабильности. Если `observed ≈ expected` для всех α, но row change всё равно большой → **g-scaling amplification** (несколько колонок с малым g доминируют в ratio).

---

## 3. Реальный результат (то, что можно заявить честно)

### 3.1. **Tied-weights decomposition — сильнейшее утверждение**

| | KL(t≥1) |
|---|---|
| head_only | 1.238 |
| embed_only | 4.007 (**80%**) |
| full | 5.017 |
| cross | −0.228 |

**`embed_only_hidden_drift.mean_rel = 0.877`** — last-layer hidden state дрейфует на 88% при изменении embeddings. **Это значит: surgery — не readout-only в tied-модели.**

**Для paper:**

> **E-39c_pre (KL decomposition, tied weights):** surgery на tied-модели изменяет **и** lm_head, **и** embed_tokens. Decomposition показывает: head_only cost = 1.24 nats (25× бюджет), embed_only cost = 4.01 nats (80× бюджет), full = 5.02. **embed_only доминирует** (hidden state drift 87.7%). Surgery **фундаментально не readout-only** в tied-моделях — «дешёвого readout-пути» не существует. Даже untied head-only (1.24) далеко выше бюджета (0.05).

### 3.2. **Full surgery: definitively infeasible**

- KL = 5.02, budget 0.05 → **100× over budget**
- Row change 98% от row norm (mean), 140% max
- Consistent across all 3 V_new strategies (4.71–5.51)

**Утверждение:** full-rotation M_K-surgery at L23 destroys the model. E-39c as designed **not viable**.

### 3.3. **α-blend: negative result, но с caveat**

- min KL over α>0 = 4.57 (α=0.7) — **91× над бюджетом**
- Non-monotonic, с row_diag anomaly (см. §2)
- Claim «no partial α works» — вероятно правильный, но требует верификации §2.4

> **ROUND-3 RESOLUTION (выполнено):** верификация §2.4 + fix §2.2 подтвердили
> механизм: `torch.linalg.qr` на already-orthonormal input делал **sign-flip
> 7 из 16 колонок** (Householder convention), **отрицая** отдельные слагаемые
> `M_kp` вместо малого вращения. QR убран (blend orthonormal by construction).
> После фикса:
> - min KL over α>0 = **0.0628 (α=0.1)** — **≈1.26× бюджет**, НЕ 91×.
> - KL монотонно и гладко: 0.063 → 0.386 → 2.02 → 2.43 → 3.78 → 4.90.
> - per-column perturbation **точно** = √(2−2√(1−α)) для всех α (clean=True).
> - Вывод (FAIL) **не изменился**, но **магнитуда** радикально другая:
>   partial surgery при малом α **near-budget**, а не «катастрофически 100×
>   сверх». Это честная коррекция: раньше артефакт завышал α=0.1 в 120×
>   (7.54 вместо 0.063).

---

## 4. Мелкие замечания

### 4.1. `_mkp_and_rows(1.0, orthogonalize=False)` в main — vs `_mkp_and_rows(1.0)` в sweep (INFO, documented)

Main surgery использует **raw V_new** (actual E-39a design). Sweep α=1 использует **V_new_perp**. KL: 5.02 vs 4.90 — различимо, но оба infeasible. Documented в `endpoint_note` ✓.

### 4.2. `V_new_perp` computed fresh per `_mkp_and_rows` call (MINOR)

Для α-sweep (7 итераций) и strategy sweep (3 итерации) — 10 вызовов QR для `V_new_perp`. Незначительно (~50ms × 10), но в `_mkp_and_rows` эта операция кешируема.

**Fix:** вынести `V_new_perp` в closure перед `_mkp_and_rows`.

### 4.3. `c_balanced` per-α не тестировался (limitation, documented)

Strategy sweep только при α=1. Caveat есть в limitations ✓. Для paper: если α-blend не работает для a_random, скорее всего и для b/c — но не доказано.

### 4.4. `runtime_s = 305` (INFO)

11 forwards × ~25s = 275s + overhead. Consistent.

### 4.5. `bs = base_t.clone(); ed = base_t.clone()` в цикле (MINOR)

`base_t.clone()` в каждом цикле α-sweep (7) + strategy sweep (3) = 10 clones × [128, 1024] × float32 ≈ 0.5 MB each = 5 MB total. Negligible.

### 4.6. `Pn = P_pool @ P_pool.T` — **1024×1024 matrix multiplication** (INFO)

Computing projector as [1024, 64] × [64, 1024] = [1024, 1024] — ~67M mults × float64, fine. But `V_b = qr(Pn @ V_top.T)` could be done cheaper as `qr(P_pool @ (P_pool.T @ V_top.T))` — avoids materializing [1024, 1024].

Cosmetic.

---

## 5. Что paper может (и не может) утверждать

**Может:**
1. ✅ Full M_K-surgery (α=1): KL = 5.02, 100× бюджет, row damage 98%. **Catastrophic, infeasible.**
2. ✅ Tied-weights: embed_only (4.01) >> head_only (1.24). Surgery **фундаментально не readout-only**, hidden drift 87.7%.
3. ✅ All 3 V_new strategies fail at α=1 (4.71–5.51).
4. ✅ Untied head-only still 25× budget.

**Может (с оговоркой, ROUND-3):**
5. ✅ α-blend (чистый QR-free blend): min KL = **0.0628 (α=0.1), ≈1.26× бюджет**.
   **Ни один α не проходит** (все > 0.05), но это **near-budget**, а не 91–100×.
   KL монотонен, blend верифицирован чистым (pert = √(2−2√(1−α)) exactly).

**Не может:**
6. ❌ «Partial surgery refuted as a route» — слишком сильно. Правильно:
   «partial surgery на данной V_new_perp конструкции refuted (near-budget,
   мин. 1.26× сверх)».
7. ✅ (исправлено в round 3) «non-monotonicity = loss-landscape property» —
   **было неверно**: это был QR sign-flip артефакт. После фикса blend чистый и
   KL монотонен; утверждение о loss-landscape property снято.
8. ❌ «surgery fundamentally incompatible with behavior at all α» — overclaim.
   Честно: «no non-trivial α passes the 0.05 KL budget; the closest (α=0.1)
   is 1.26× over, so partial surgery is near-feasible but not within budget on
   this construction».

---

## 6. Итог

**Блокеры:**

1. **§2.2/§2.3** — row_diag at α=0.1 (0.772 vs expected 0.32) не сходится с «clean orthogonalized blend». Интерпретация «non-monotonicity = loss landscape» может быть **artefactом V_new_perp QR-нестабильности**, а не физическим свойством. Fix: per-α `‖V_mixed − V_top‖_per_col` диагностика + сравнение с theory.

2. **§2.1** — `rho_noise_u_obs_overlap_mean` нормализация отличается от E-39a (mean over [N,S,R] vs [N,S]). Silent inconsistency.

**Средне:**

3. §4.1 — α=1 endpoint mismatch (5.02 raw vs 4.90 perp). Documented, но стоит явно в paper тексте (не только в JSON note).
4. §4.2 — V_new_perp recompute в цикле (cosmetic).

**Мелочи:**

5. §4.6 — projector matmul size.
6. §4.3 — c_balanced per-α untested.

**Оценка: 7.5/10** (до round 3). **После round-3 фикса — 9/10:** оба HIGH blockera (§2.2/§2.3, §2.1) закрыты, blend верифицирован чистым, вывод честный и data-driven.

**ROUND-3 STATUS (закрыто):**
- ✅ §2.2/§2.4: механизм подтверждён (QR sign-flip 7/16 колонок = отрицание слагаемых M_kp). QR убран, blend orthonormal by construction, per-column perturbation = √(2−2√(1−α)) exactly для всех α (clean=True). KL теперь монотонен и гладко; немонотонность была артефактом, а не loss-landscape property.
- ✅ §2.1: rho_noise переведён на конвенцию E-39a (per-position Frobenius), ключ переименован `rho_noise_u_obs_overlap_mean` (0.01416 vs 0.0142 в E-39a — совпадает).
- ✅ §4.2: V_new_perp вынесен в closure (вычисляется один раз).
- ✅ §4.6: b_principal через factored projector (без [D,D]).

**Остаток (не блокеры):** §4.1 (α=1 endpoint raw vs perp — documented), §4.3 (c_balanced per-α untested — limitation), §4.5 (clones — negligible).

---

## 7. Round-3 (v4) — применение фиксов + новый результат

### 7.1. Механиз немонотонности (подтверждён диагностикой)

`torch.linalg.qr` на **already-orthonormal** input (Gram = I) вырождается в
signed permutation, и знак каждого столбца определяется Householder convention —
**произвольно**. В α-blend (round 2) QR применялся к `√α·V_new_perp +
√(1−α)·V_top`, который orthonormal **by construction** → QR лишь перекидывал
знаки 7 из 16 колонок. Столбец с перекинутым знаком **отрицает** своё
слагаемое `+σ_j·u_j·v_jᵀ → −σ_j·u_j·v_jᵀ` в `M_kp` — т.е. вместо малого
вращения происходит **sign-flip**, который и давал row_diag 0.77 (против
теоретических 0.32) и немонотонный KL (7.54 при α=0.1).

### 7.2. Fix (round 3)

- Blend branch: **QR убран** (`V_mixed = √α·V_new_perp + √(1−α)·V_top`),
  orthonormal by construction (V_new_perp ⊥ V_top).
- Диагностика §2.4: per-α per-column `‖V_mixed[:,j]−V_top[:,j]‖` vs теория
  `√(2−2√(1−α))`; `clean_rotation_verified` = all(пер_col < 1.5·exp).
- §2.1: rho_noise = `E_t[‖û_tᵀVn‖²]` (per-position Frobenius), ключ
  `rho_noise_u_obs_overlap_mean` (совпадает с E-39a).
- §4.2/§4.6: V_new_perp в closure; b_principal через factored projector.
- Conclusion data-driven (не хардкод), `near_budget_note` в JSON.

### 7.3. Новый результат (детерминированный, gate 5.01748 воспроизведён)

| α | KL(t≥1) | row | per-col pert (exp) | clean |
|---|---|---|---|---|
| 0.0 | 0.0000 | 0.000 | 0.000 (0.000) | ✓ |
| 0.1 | **0.0628** | 0.233 | 0.320 (0.320) | ✓ |
| 0.2 | 0.3864 | 0.334 | 0.460 (0.460) | ✓ |
| 0.5 | 2.0217 | 0.551 | 0.765 (0.765) | ✓ |
| 0.7 | 2.4293 | 0.679 | 0.951 (0.951) | ✓ |
| 0.9 | 3.7836 | 0.825 | 1.169 (1.169) | ✓ |
| 1.0 | 4.9045 | 0.982 | 1.414 (1.414) | ✓ |

- **min KL = 0.0628 (α=0.1) ≈ 1.26× бюджет** (было 7.54 ≈ 150× до фикса).
- **Вывод (FAIL) не изменился**, но магнитуда: partial surgery **near-budget**,
  а не «100× сверх».
- Full surgery (raw V_new, actual E-39a design): **5.0175**, FAIL (100×). Не
  изменился.
- Tied decomp (raw V_new): full 5.0175 = head_only 1.2380 + embed_only 4.0071
  (+ cross −0.2276).
- Strategies (α=1): a_random 4.7083, b_principal 5.5061, c_balanced 4.7878 —
  все FAIL; ρ_noise теперь 0.01416/0.01432/0.01430 (конвенция E-39a).

### 7.4. Что paper может утверждать (round 3)

> **E-39c_pre (negative result):** M_K-surgery at L23 fails the KL pre-gate.
> **Full rotation (α=1):** KL = 5.02 (≈100× the 0.05 budget) — catastrophic,
> refuted. Tied-weight decomposition: 80% from embed (upstream propagation,
> hidden drift 88%), 20% from head; the surgery is NOT readout-only.
> **Partial rotation (α-sweep, clean QR-free blend):** KL is smooth and
> monotone in α (0.063 → 4.90); min KL = 0.063 at α=0.1, only ≈1.26× the
> budget. **No non-trivial α passes 0.05**, but the margin is small — partial
> surgery is *near-budget*, not 100× over. The blend is verified a true
> partial rotation (per-column perturbation = √(2−2√(1−α)) exactly). All three
> V_new strategies fail at full surgery (4.71–5.51).
> **(Changelog v5→v6):** the earlier «non-monotonic α-blend / loss-landscape
> property» claim was a **QR sign-flip artifact** (Householder sign convention
> on an orthonormal input silently negated M_kp terms). With the QR removed,
> the small-α point is a genuinely small rotation; the negative result is
> unchanged but its magnitude is near-budget, and the loss-landscape
> interpretation is withdrawn.