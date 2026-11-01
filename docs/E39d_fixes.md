# Финальный аудит E-39d (round 8) — критических нет, но нашёл 1 stale-комментарий и 4 medium

## ✅ Проверено — вычисления корректны

### Fix #1 (actual logits) — **математически верен**

Алгебраически проверил: для L23 (последний слой, вход в final norm):
```
h_norm = h_L23 / rms(h_L23) · (1+w) ≈ √D · û · (1+w)
logit_v = W_lm[v,:] · h_norm = √D · W_lm[v,:] · diag(1+w) · û
        = √D · M_K[v,:] · û,   M_K = W_lm·diag(1+w)
```
Код: `SQRT_D * (u_all @ M_K_full.t())` — правильно. Старая форма `H @ W_lm.t()` отличалась на `‖h‖/(√D·g)` — position/token-dependent. ✓

### Fix #2 (u64_docs на device) — **исправлено**
```python
u64_docs = torch.stack(u64_docs).to(device)  # было .cpu() в конце
```
`.cpu()` внутри loop, `.to(device)` после stack. ✓

### Fix #3 (NaN guard) — **правильно**
Pre-step check + post-step param check. ✓

### Fix #4 (sqrt guard) — **работает**
`clamp(min=1e-20).sqrt()` → max 1e-10 → нет NaN. ✓ (см. issue #4 ниже)

### Fix #5 (H-norm clamp) — **применено**
В `u_all`, `u`, `u64_docs` — но **не везде** (см. issue #2).

### Other verifications
- `_c4_docs` — исправлено через persistent iterator ✓
- `flatness_loss` — top-r SVs правильные, Vt orthonormal, Pt rank-r projector ✓
- `_selftest_kl` — тестирует logsumexp trick ✓
- `_selftest_flatness` — тестирует top-r SVs ✓
- Verdict требует все три критерия ✓
- `kl_f_fwd` cross-check (first run only) ✓
- Post-step `if not isfinite(W_rows).all()` — lовит corrputed state ✓

---

## 🟡 MEDIUM #1 — **stale comment: `l_flat is ~1e3` математически невозможно**

```python
# l_flat is ~1e3 on this data while l_kl is ~1e-1, so WITHOUT this the
# spec λ-grid {0.01..10} would have flat dominate at every λ (KL leash
# never binds). Normalising makes λ a meaningful relative weight.
```

**Математический факт** (проверено):
- Vt имеет orthonormal columns → `trace(Pt) = r = 16`
- `‖Pt‖_F² = r = 16` для rank-r projector
- `sum_norm_sq = P·16` (не P·1e3)
- `l_flat = 16 − ‖P̄‖_F²` ∈ **[0, 16]**

**Max l_flat = 16, не 1e3.** Comment устарел — вероятно, до fix #1a/#1b (когда Vt был garbage-mixture, и `Pt` не был projector).

**Влияние:**
- Comment вводит в заблуждение о роли normalizer
- Reader подумает «λ-grid откалиброван под l_flat ~1e3», но фактический l_flat ∈ [0, 16]

**Fix:** обновить comment:
> l_flat ∈ [0, r=16] by construction (rank-r projectors). Base l_flat is expected O(1e-3)–O(16) depending on leaf flatness; normalizer keeps λ-sweep scale-invariant.

---

## 🟡 MEDIUM #2 — **нет clamp в `u64_docs`**

```python
arr = arr / arr.norm(dim=-1, keepdim=True)  # NO clamp
u64_docs.append(arr.double().cpu())
```

Если какая-то `h_t` имеет **ровно** норму 0 (unlikely для trained model, но defensive), `arr = 0/0 = NaN`. `_kappa_dryrun` с NaN → `κ_base = NaN` → всё заражено.

**Fix:**
```python
arr = arr / arr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
```

---

## 🟡 MEDIUM #3 — **surrogate награждает rank deficiency**

Если `M_K` деградирует до rank `< 16`:
- `s[-i] → 0` для некоторых i → `clamp(1e-20)` даёт 1e-10 → `Vt = J^T u / 1e-10`
- `J^T u = 0` (u в null space) → `Vt = 0/1e-10 = 0` → `Pt` имеет **меньший rank** → `‖Pt‖_F² < 16`
- **`l_flat` уменьшается** (surrogate "улучшился")
- Но `κ'` через SVD (правильная метрика) → **растёт** (top-16 включает произвольные null-space directions)

**KL leash должен предотвратить катастрофическую rank loss** (logits станут почти нулевыми для всех K токенов → KL взорвётся). Но **mild rank loss** (rank 15 из 16) может пройти: KL растёт на ~10%, surrogate уменьшается. Это training-game, а не реализация flattening.

**Fix (опционально):** penalize rank deficiency:
```python
if (s < 1e-6).any():
    # rank < r — flatness metric unreliable; add penalty
    rank_penalty = r - (s > 1e-6).sum(1).float().mean()
    # или: clamp l_flat to be >= min value
```

**Severity:** MEDIUM, но только если training реально пойдёт в эту область. Обычно KL leash доминирует.

---

## 🟡 MEDIUM #4 — **docstring "9.5e-6" не валидируется runtime кодом**

```python
# Verified: √D·(û·(W·diag g)ᵀ) matches model.lm_head(model.model.norm(h))
# to 9.5e-6 (the old form was off by 2.69 nats).
```

**Runtime проверка в коде** (§cross-check first run):
```python
kl_f_fwd = ...  # 2-forward KL on C4 PPL docs
kl_f = ...      # cached-H formula on jcache positions
if abs(kl_f_fwd - kl_f) > max(0.05, 0.2 * max(...)):
    WARN
```

Это **сравнение на РАЗНЫХ данных** (C4 vs jcache). Порог 20% — sanity check по magnitude, не formula-correctness.

Формула **алгебраически верна** (я проверил), но **9.5e-6 claim не доказан этим кодом**. Возможно, это результат offline-разработки — тогда стоит указать в docstring «offline verified during development; runtime validates magnitude on held-out data».

**Severity:** LOW-MEDIUM (не bug, но overclaiming).

---

## 🟢 LOW

### #5 — `kl_full` (jcache) vs `kl_full_2forward_check` (C4) — разная семантика

```python
"kl_full": kl_f,                    # mean KL over jcache positions
"kl_full_2forward_check": kl_f_fwd, # mean KL over C4 PPL docs
```

Два разных числа, один label «full». Paper должен знать, что это разные datasets. Minor confusion risk.

### #6 — `SEEDS[0]` hardcoded в `torch.manual_seed(SEEDS[0])` перед normalizer

```python
torch.manual_seed(SEEDS[0])  # после этого norm_idx=random deterministic
```

Если `SEEDS` изменится, normalizer использует новый seed. OK, но стоит комментарий про «normalizer uses SEEDS[0] to keep norm_idx deterministic across runs».

### #7 — `_selftest_flatness` не покрывает degenerate case

Проверяет only полноценный rank-K M_K. Не проверяет M_K с rank < r. Именно этот случай **породил бы issue #3**. Расширение selftest'а:
```python
M_K_lowrank = torch.randn(K, Dd, device=device)
M_K_lowrank[r:] = 0  # rank r
# должна дать l_flat = 0 (small rank projectors), но κ' высокий
```

---

## Итог

**После round 8 код вычислительно КОРРЕКТЕН** по всем CRITICAL issues:
- ✅ Actual logits (fix #1)
- ✅ Device consistency (fix #2)
- ✅ NaN guard (fix #3)
- ✅ Sqrt guard (fix #4)
- ✅ H-norm clamp в u_all (fix #5, но см. #2)

**Осталось (не блокируют запуск, но снижают честность):**

| # | Issue | Impact |
|---|---|---|
| 1 | «l_flat ~1e3» stale | Misleading comment |
| 2 | No clamp в u64_docs | Potential NaN (rare) |
| 3 | Rank deficiency rewarded | Surrogate game (KL leash guard) |
| 4 | «9.5e-6» not runtime-verified | Overclaiming |
| 5-7 | LOW/misc | Minor |

**Рекомендация перед paper v6.0:**

1. **Fix #1** — переписать comment (5 мин). Иначе paper-рецензент поймает inconsistency.
2. **Fix #2** — clamp (1 мин), правильность кода.
3. **Fix #3** — отметить в limitations (rank deficiency is not covered by the surrogate; KL leash is the only guard).
4. **Fix #4** — либо runtime verify, либо переформулировать docstring.

**Ожидаемый результат E-39d после fix:**

Скорее всего **FAIL или низкий SUCCESS rate**:
- Fundamental trade-off (E-39c_pre): κ' < 0.005 requires KL ≫ 0.05
- Training может найти **лучшее решение**, чем algebra **если** есть структура, которую linear rotation не может exploit
- Но — E-39c_pre показал, что trade-off **структурный** (all α disjoint), не «optimizer не нашёл»
- Training с W_lm trainable — **та же степень свободы** (131K params) что и rotation (16×1024 subspace). Может найти немного другое, но не фундаментально проходить trade-off

**Приоритетный прогноз:** E-39d даст 0 SUCCESS, несколько PARTIAL (κ' < 0.005, но PPL/KL проваливаются), подтверждая E-39c_pre negative conclusion. **Это будет честный результат**, подтверждающий, что проблема структурная.

**Если E-39d даст SUCCESS** — это будет сюрприз и важный результат: «training finds structure that algebra misses». Тогда **обязательна репликация** с 3+ seeds и cross-model, чтобы убедиться, что это не seed-luck.

**Текущая версия E-39d — после исправления #1 и #2 — готова к запуску**. Запускать 12 runs × 100 steps = ~10-15 min на CPU, ~2 min на T4. Результат стоит получить до paper v6.0 — это закроет блок 4.8 §E-39d и позволит написать окончательный verdict по E-39 линии.

---

# Round 9 — disposition (применено к коду + верификация)

Round 9: 4 MEDIUM + 3 LOW, **критических нет**. Ни один не меняет динамику обучения (комментарии, defensive-clamp, selftest, документация) → **запущенный run остаётся valid**, правки применены для след. запусков.

| # | Issue | Статус | Что сделано |
|---|---|---|---|
| M#1 | «l_flat ~1e3» stale comment | ✅ Fixed | Comment переписан: l_flat ∈ [0, r=16] по построению (rank-r projectors, ‖P_t‖²=tr P_t=r). Реальное base l_flat ≈ **0.187** (live run) — O(0.1), не 1e3. (См. ниже про 3575.4 в старом логе.) |
| M#2 | No clamp в `u64_docs` | ✅ Fixed | `arr / arr.norm(dim=-1,keepdim=True).clamp_min(1e-12)` — defensive: 0-норма h_t → 0/0=NaN → κ'=NaN → весь sweep заражён. |
| M#3 | Rank deficiency rewarded (surrogate game) | 📝 Documented, **не чинится кодом** | Это **inherent limitation** surrogate (не bug): rank-degenerate M_K сжимает ‖P_t‖² и «улучшает» l_flat, но κ' (SVD-метрика) растёт. Единственный guard — KL leash (logits K токенов → 0 → KL взрывается). Добавлен **degenerate-case finite-check** в `_selftest_flatness` (rank0 / rank-r → l_flat finite) + комментарий в коде. Penaltу в loss НЕ добавляю (изменит динамику, не требуется для вердикта). |
| M#4 | «9.5e-6» not runtime-verified | ✅ Fixed | Docstring переформулирован: 9.5e-6 — **offline-верификация при разработке**; runtime-чеки проверяют **формулу** (vs full-softmax KL в `_selftest_kl`) и **магнитуду** на held-out C4 (kl_f_fwd), но **не** пере-доказывают 9.5e-6. |
| L#5 | kl_full (jcache) vs kl_full_2forward (C4) | ✅ Fixed | Комментарий: два числа на **разных** популяциях (jcache train positions vs C4 PPL docs) → same-ORDER magnitude check, **не** equality; порог max(0.05, 20%) отражает это. |
| L#6 | `SEEDS[0]` hardcoded в normalizer | ✅ Fixed | Комментарий: нормализатор детерминирован через `SEEDS[0]` (одинаковый norm_idx между runs). |
| L#7 | `_selftest_flatness` не покрывает degenerate | ✅ Fixed | Добавлен degenerate-check (rank0=0.0, rankr=0.0064, оба finite). |

## Два false-positive, которые я поймал при верификации (важно для честности)

**1. Спурные NaN в `kl_leash` были багом МОИХ тест-скриптов, а не кода.**
При проверке я написал `H/H.norm(-1, keepdim=True)` — аргумент `-1` попал в `p` (L_p-норма), а не в `dim` → `u` получилась ~7392 (вместо 1.0) → логиты ~5e10 → NaN. Это **тот же bug, что был в round 1**. Реальный код **корректен** (`dim=-1`, строки 147/229/383).

С **корректной** нормализацией на **реальных** jcache данных `kl_leash` совпадает с точным full-vocab KL до **≤3e-6** на всех масштабах edit (1.0→5.0), все значения finite. Запущенный run работает при frozen base p_topK≈**0.42–0.52**.

**2. В формуле `kl_leash` есть (теоретический) float32-cancellation edge при p_topK_base→1**: член `exp(lse_b−m)−Σexp(b_k−m) = (1−p_topK_base)·exp(lse_b−m) → 0 → log(0) → NaN`. Но этот режим **не достигается** (frozen base p_topK≈0.42–0.52, член остаётся ~0.48·exp(…) > 0). Guard (`clamp`) **НЕ добавлен** — он no-op в реальном режиме, а NaN-guard в training-цикле (round 8 #3) — backstop. Для paper: упомянуть как known float32 edge, не для данного run.

## Про «3575.4» в старом логе (red herring)

Старый лог `/tmp/exp39d_run.log` (08:47) показывает `flatness normalizer (base l_flat): 3575.4` и `sentinel excluded: 0/4096`. Это **pre-round-8 v1 код** (до fix #1a/#1b top-r и до actual-logits) — garbage Vt (не projector → ‖P_t‖²≫r) и старый sentinel. **Live run** (19:xx, round 8): `base l_flat = 0.1869`, `sentinel 55.4%`, κ' base 0.015923 — всё согласовано с audit. 3575.4 — артефакт старой версии, не текущий код.

## Состояние live run (на момент правок)

PID 3917103, ~1.5ч из ~2.7ч, на λ=0.1 s2. Ранние результаты: κ' падает 0.0159→0.0044 (λ=0.01), KL_full 0.029<0.05, PPL +12–106%. 2-forward KL (0.069, C4) vs cached-H (0.029, jcache) — same order (разные популяции). Правки в этот запуск **не влияют** (процесс уже загрузил старый .py в память).

## Итог round 9

Код вычислительно КОРРЕКТЕН (CRITICAL закрыты в round 8, верифицировано заново). Round 9 — только честность/robustness: 5 фиксов применено (M#1, M#2, M#4, L#5, L#6, L#7), 1 задокументирован как limitation (M#3), 2 false-positive отклонено с доказательствами. Self-тесты проходят (KL brute + p_topK~0.5; flatness top-r + degenerate-finite).