# Финальный аудит E-39d (round 8) — 2 критических silent-бага + 4 средних

## 🔴 CRITICAL #1 — `H @ W_lm.t()` даёт **псевдо-логиты**, не настоящие

### Код (строки base logits cache)

```python
bl = Hb @ W_lm.t()                                # [B, V]  ← псевдо-логиты
lse_b_all[i:i+B] = torch.logsumexp(bl, dim=-1)
base_k_all[i:i+B] = bl[:, tok_ids]
raw_k = torch.exp(bl[:, tok_ids])                 # псевдо-вероятности
```

### Что должно быть (E-38e факторизация)

Для L23 (последний слой) настоящие логиты:
```
logits_t = lm_head(RMSNorm(h_L23)) = √D · M_K · û_t,  M_K = W_lm · diag(1+w)
```
где `û_t = h_L23/‖h_L23‖`.

**Код считает** `W_lm · h_L23 = ‖h‖ · û · W_lm`, что **отличается** от `√D · (1+w) · û · W_lm` на фактор `√D·(1+w)/‖h‖`, **зависящий от позиции и токена**.

### Влияние

**Softmax per-position temperature:**
- `softmax(pseudo_logits)` использует температуру `‖h_p‖/√D`
- `softmax(actual_logits)` использует температуру `1`
- Для позиции с `‖h_p‖ = 60`, `D=1024` → температура ≈ **1.875**

Настоящий KL = KL(softmax(actual_e) ‖ softmax(actual_b)), а код считает KL(softmax(pseudo_e) ‖ softmax(pseudo_b)) — это **temperature-scaled KL**, численно отличающийся от настоящего.

### Всё заражено через один root cause

- `base_k_all` — псевдо, `lse_b_all` — псевдо → **KL псевдо**
- `raw_sum_all` — псевдо → **sentinel mask смещён** (E-34 определил threshold 0.5 на настоящих P128_RAW, а `exp(pseudo) ≠ actual`)

**PPL не заражён** — `_ppl` использует `model(...)`, настоящий forward. Это спасает: verdict по PPL корректен, KL и sentinel — нет.

### Сравнение с E-39c_pre

E-39c_pre использует `_logprobs(model, ids)` (настоящий forward) → **KL=6.31 настоящий**. E-39d использует псевдо → **KL несравним с E-39c_pre**. Paper не сможет писать «E-39d достиг KL < 0.05 при κ' < 0.005, в отличие от E-39c» — числа на разных шкалах.

### Fix (2 строки)

```python
u_all = H_all / H_all.norm(dim=-1, keepdim=True)          # û_t
M_K_full = (W_lm * g_base[None, :])                       # [V, D] — для base
bl = math.sqrt(D) * (u_all @ M_K_full.t())                # actual logits
```

**Альтернатива (безопаснее):** final-eval KL через настоящий forward:
```python
# Один раз: lp_base = _logprobs(model, ppl_ids)
# В final: swap W_rows → forward → _kl_vs_base(lp_edit, lp_base, ppl_attn)[1]
```
Дороже (2 forward), но гарантирует корректность verdict `kl_ok`.

---

## 🔴 CRITICAL #2 — device mismatch на GPU

```python
u64_docs.append(arr.double().cpu())     # явно CPU
...
u64_docs = torch.stack(u64_docs)        # CPU (без .to(device))
```

Вызов:
```python
kap_base = _kappa_dryrun(M_K_base.double(), u64_docs, N_docs, S, R)
```
где `M_K_base` на `device` (cuda).

Внутри `_kappa_dryrun`:
```python
proj = torch.einsum("kd,td->tk", M_kp, u_t)  # cuda × cpu → RuntimeError
```

**На CPU работает** (device="cpu", всё на cpu). **На T4 падает** сразу. Docstring обещает «T4: ~1-2 min» — этот сценарий **не работает**.

### Fix

```python
u64_docs = torch.stack(u64_docs).to(device)
```

---

## 🟡 MEDIUM #3 — NaN-check **после** `opt.step()`

```python
loss.backward()
torch.nn.utils.clip_grad_norm_([W_rows], 1.0)
opt.step()                                     # ← NaN-градиент уже применён

if not torch.isfinite(loss.detach()):          # ← check ПОЗЖЕ
    break
```

Если forward дал NaN, `backward` даёт NaN-градиенты, `opt.step()` корраптит `W_rows` (Adam state тоже NaN), и только потом детектится. Run всё равно завершается корректно (`diverged_at_step` записан), но состояние W_rows — мусор.

### Fix

```python
loss = lam * l_flat + LAMBDA_KL * l_kl
if not torch.isfinite(loss):
    run["diverged_at_step"] = step
    break
loss.backward()
...
```

---

## 🟡 MEDIUM #4 — `.clamp(min=0).sqrt()` → возможное деление на 0

```python
s = w[:, -r:].clamp(min=0).sqrt()
Vt = torch.bmm(J.transpose(1, 2), U16) / s[:, None, :]
```

Если **top-16 eigenvalue** становится **точно 0** (M_K деградирует до rank < 16 во время тренировки), `.clamp(min=0)` даёт `s = 0`, деление → **Inf/NaN** → `Pt = NaN` → `L_flat = NaN` → `loss = NaN`.

Тренировка мотивирована снижать `L_flat`; один из способов — сделать P_t вырожденным → возможен именно этот сценарий.

### Fix

```python
s = w[:, -r:].clamp(min=1e-20).sqrt()  # или 1e-12
Vt = torch.bmm(J.transpose(1, 2), U16) / (s[:, None, :] + 1e-12)
```

---

## 🟡 MEDIUM #5 — `conclusion` / `budget_note` broken

Те же проблемы, что я отмечал в E-39c_pre:

```python
f"small α ({kl_safe_alphas}) pass KL<{KL_PASS}"
```
Если `kl_safe_alphas = []` — «small α ([]) pass KL» бессмысленно. И:
```python
f"κ'≈{sweep[1]['kappa_dryrun']:.4f}"  # sweep[1] — fragile index
```

В E-39c_pre это уже подтверждено — текст в JSON противоречил данным.

### Fix

Заменить f-string на условную генерацию:
```python
if not kl_safe_alphas:
    conclusion = "no α passes KL budget"
else:
    conclusion = "..."
```

---

## 🟡 MEDIUM #6 — `_selftest_flatness` не покрывает degenerate случай

Selftest проверяет: projector, `tr P = r`, `s[-1] = σ₁`. Но **не проверяет** M_K с rank < r (когда `s` содержит 0). Это именно тот случай, где `#4` может сработать в production.

### Fix (опционально)

```python
# в selftest добавить
M_K_rank_def = torch.zeros(K, Dd, device=device)
M_K_rank_def[:r-1] = torch.randn(r-1, Dd, device=device)  # rank r-1
# должна дать NaN / inf, или должна падать громко
```

---

## 🟢 LOW

- **`H_sub.norm()` без clamp** — `u = H_sub / H_sub.norm(...)`. Нулевая норма (крайне редко) → NaN. `.clamp_min(1e-12)` безопаснее.
- **`dtype` version-check** — `transformers.__version__.split(".")[:2]` — если версия «4.35.0rc1», то `("4", "35")` — OK. Если «5.0.0dev» → `(5, 0) >= (4, 35)` → OK. Работает.

---

## Итоговая таблица

| # | Баг | Severity | Что портит |
|---|---|---|---|
| 1 | Псевдо-логиты в KL/sentinel | 🔴 CRITICAL | KL = biased proxy; sentinel mask biased; несравнимо с E-39c_pre |
| 2 | Device mismatch на GPU | 🔴 CRITICAL | T4 запуск падает сразу; только CPU работает |
| 3 | NaN-check timing | 🟡 MEDIUM | W_rows корраптится до детекта (но run останавливается корректно) |
| 4 | `clamp(min=0)` деление на 0 | 🟡 MEDIUM | При деградации M_K → NaN |
| 5 | conclusion/budget_note текст | 🟡 MEDIUM | Ложные утверждения в JSON |
| 6 | selftest не покрывает rank-deficient | 🟡 MEDIUM | Пропускает #4 в selftest |
| 7 | `H.norm()` без clamp | 🟢 LOW | Редко NaN |
| 8 | `sweep[1]` hardcode | 🟢 LOW | Fragile при изменении grid |

---

## Приоритет для paper v6.0

**До любого запуска (E-39d невалиден без них):**

1. **Fix #1** (псевдо-логиты) — без этого KL не сравним с E-39c_pre, verdict `kl_ok` недостоверен.
2. **Fix #2** (device) — без этого GPU не запустится.
3. **Fix #3** (NaN-check) — 3 строки.
4. **Fix #4** (clamp) — 2 строки.

**После запуска — пересмотреть paper-нарратив:**

- Если E-39d после фикса даст **FAIL** (κ' < 0.005 только при KL ≫ 0.05) — это подтверждает E-39c_pre: training не преодолевает фундаментальный trade-off. **Честный negative result.**
- Если **PARTIAL** (κ' достигается, но PPL растёт) — тренировка учится чему-то, но ценой поведения.
- **SUCCESS** — маловероятно, но если да, то требует репликации с actual KL (не proxy).

**Текущая версия E-39d — не может идти в paper.** Оба CRITICAL бага дают смещённые результаты в разных аспектах: #1 искажает KL (систематически), #2 не даёт запустить на GPU. После фикса #1+#2+#3+#4 — 10 минут работы, и код станет **valid** (в отличие от E-39a/E-39c_pre, где было 4-6 раундов).

---

## Статус фикса (round 8, commit b918961)

| # | Статус | Что сделано |
|---|---|---|
| 1 | ✅ FIXED | KL + sentinel на **actual logits** `√D·M_K·û` (E-38e факторизация). Верификация: `√D·(û·(W_lm·diag g)ᵀ)` совпадает с настоящим `model.lm_head(model.model.norm(h))` до **9.5e-6** (старый `H@W_lmᵀ` — до 2.69 nats). KL-шлейк, base logits, sentinel-масса — все на actual. Логсумэксп-трюк для readout-only edit **exact** (не proxy): cross-check с полным 2-forward KL на тех же H даёт diff 8.9e-6. Verdict KL (full-dataset) + независимый 2-forward cross-check на C4-доках (первый run) в JSON (`kl_full_2forward_check`). |
| 2 | ✅ FIXED | `u64_docs = torch.stack(u64_docs).to(device)` — T4 cuda×cpu краш устранён. |
| 3 | ✅ FIXED | NaN-check **до** `backward/step` (pre-step guard) + повторная проверка `W_rows` после `opt.step()`. |
| 4 | ✅ FIXED | `clamp(min=1e-20).sqrt()` в flatness (rank-degenerate M_K → нет 0-деления). |
| 5 | N/A | `kl_safe_alphas`/`budget_note`/`sweep[1]` — **не существуют в E-39d** (паттерны E-39c_pre; проверено grep'ом). |
| 6 | N/A (опционально) | Selftest rank-deficient — не добавлен (защита #4 покрывает; selftest проверяет корректность top-r, а не degeneracy). |
| 7 | ✅ FIXED | `H_sub.norm(...).clamp_min(1e-12)` в kl_leash и flatness. |
| 8 | N/A | См. #5. |

### Побочные эффекты фикса #1 (важно для paper)

- **Sentinel: 0.0% → 55.4%**. Старый `Σ exp(pseudo_logit)` всегда > 0.5 → «sentinel excluded: 0/4096 (0.0%)» был артефактом. Новый sub-stochastic `Σ_k p_k` (= P128_RAW по определению; поле P128_RAW отсутствует в кэше, но reconstruction точна до 9.5e-6) → 2271/4096 позиций sentinel, 1825 в тренировке. Это **согласуется с E-34** (порог 0.5 калиброван на P128_RAW). Тренировка теперь на ~45% позиций — это честнее, не баг.
- **PPL base: 28.38 → 34.31**. Причина: `_c4_docs` берёт **streaming C4 без seed** — набор 8 доков не детерминирован между запусками (pre-existing; PPL-код и веса не менялись — PPL считается на оригинальных весах до обучения). Не влияет на verdict (PPL-дифференциал считается внутри одного run по тем же докам), но для репликации paper нужен фикс seed'а в `_c4_docs` (отметить как follow-up).
- **KL-шкала теперь совпадает с E-39c_pre** (реальные логиты, не temperature-scaled proxy) — сравнение «E-39d KL < 0.05 vs E-39c 6.31» становится валидным.

### Запуск (round 8 valid)

`experiments/exp39d_l23_training.py` — λ∈{0.01,0.1,1,10} × 3 seeds × 100 steps, CPU. Логи: `/tmp/exp39d_round8.log`, JSON: `results/exp39d_l23_training.json`.