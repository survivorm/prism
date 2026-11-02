# Аудит v2 `exp39b_ptq_perturbation.py` против E39.md §2

Проверял построчно, сравнивая с предыдущей версией и с ТЗ. Ниже — что исправлено, что осталось, и что появилось нового.

## 1. Исправлено корректно (высокий приоритет)

| Пункт | Было | Стало | Проверка |
|---|---|---|---|
| **P0.1** | `t[:s_len]` резал **символы**, а не токены | Полный текст → `tokenizer(max_length=s_len, truncation=True)` | ✅ |
| **P0.2** | Padding в GPTQ-гессиане | `X = X_all[mask]`, `mask = attn_cal.bool()` | ✅ — теперь `n_real ≈ 16384`, а не ~6000 |
| **P0.3** | PPL = mean NLL | `ppl = exp(Σ NLL / n_real)` | ✅ — `nll_sum / max(n_real, 1)`, потом `np.exp` |
| **P1.4** | `SEED=0` не использовался | `random/np/torch.manual_seed` + `use_deterministic_algorithms` | ✅ (best-effort) |
| **P1.5** | `J_rec, J_cache, H, M_K_base` висели | `del ...; torch.cuda.empty_cache()` | ✅ |
| **P1.6** | Gate только абсолютный | `κ < 0.02 AND ratio ≤ 1.25` | ✅ |
| **P1.7** | `int8` назывался GPTQ-8bit | `int8_rtn`, документировано | ✅ |

P0.1–P0.3 — это те самые «тихие» баги, которые бы дали неверный `ΔPPL` и неверный GPTQ-H. Теперь они устранены правильно.

## 2. Частично исправлено

### P0.4 — двойной RMSNorm: **check, а не fix**
Код проверяет `lm_head(last_hidden_state) == model(...).logits` на 4 документах, но:
- проверка **только warning** (`ref_err > 2.0` → `print`), не `raise`;
- если проверка провалится, PPL всё равно посчитается на `last_hidden_state` напрямую → тихо неверный.

Для стандартного HF Qwen `last_hidden_state` действительно post-final-RMSNorm, так что проверка должна пройти. Но логичнее было бы `raise` при `ref_err > 2.0` — тогда мы гарантированно не получим неверный PPL.

Дополнительно: `lref.float() @ W_lm64.float().T` создаёт fp32-копию `W_lm64` (~1 ГБ) и `[4,128,V]` fp32 (~500 МБ) — на T4 терпимо, но лишнее. Лучше `W_lm32 = W_lm.float()` один раз.

### P0.5 — tied weights: **документировано, но не исправлено**
`"readout_only_ptq": True` в JSON — хорошо. Но `ppl["gptq4"]` — это PPL «если квантовать только lm_head, embeddings оставить fp16», а не PPL реального GPTQ tied-модели. Это **занижает** деградацию. Для gate (κ) не важно, но `ppl_delta_pct_gptq4_vs_base` нельзя читать как «ΔPPL модели после GPTQ-4bit». В JSON это теперь есть — но в `print` и в тексте отчёта стоит повторить.

### P1.4 — детерминизм на CPU
`torch.use_deterministic_algorithms(True)` — только в `if dev.type == "cuda"`. На CPU не выставлен. CPU-ops обычно детерминированы, но формально ТЗ §2.5 требует deterministic. Мелочь.

## 3. Не исправлено (осталось)

### P1.1 — для L<23 `_recompute_J` — синтетика
Скрипт для **всех** слоёв `L_CACHED` строит `J_base = _recompute_J(M_K_base, H_L)` и `Jq = _recompute_J(M_K_q, H_L)`. Для L=23 это точный Jacobian (self-check ~1e-7). Для L=6/12/18/19/22 — это «readout-linearized J»: `c_t · M_K · (I − û_t û_tᵀ)` с `û_t` от слоя L, но **без промежуточных слоёв** `D_i`. То есть:

- `metrics[6]["fp16"]["kappa"]` — это **не κ(L6)** реального листа, а κ синтетической матрицы;
- `metrics[6]["gptq4"]["kappa"]` — это «что делает квантование M_K с этой синтетикой»;
- `metrics[6]["fp16"]["cos_ug_1k"]` — cos прокси-`u_g` синтетики, а не реального `J_6`.

В комментарии это указано: «exact at L23; convention for L<23». Но JSON не несёт флага. **Silent** риск: читатель JSON решит, что `kappa(L6)` — это κ реального листа L6.

**Фикс**: добавить в каждый `metrics[L][name]` поле `"synthetic_readout_j": L != 23`. Или считать L<23 только как `readout_effect`, а κ/β/ cos — только для L23.

### P1.2 — `_cos_ug_1k` — proxy, а не точный `u_g`
`v = JJᵀ·1; u_g = v/‖v‖; cos = ⟨u_g, 1/√K⟩`. В E-33r это работало как cheap-proxy (align 0.85–0.99 с точным `u_top`). В E37.md определение `u_g` — это **левый сингулярный вектор, максимизирующий |cos(u, 1)|**. Разница небольшая (proxy ≈ u_top на 85–99%), но числа не будут сравнимы с E-37 один-в-один.

**Фикс**: при желании считать exact через `SVD(J)` — но для gate это не критично, только для отчётности.

### P1.3 — нет sentinel-фильтра
E39.md §14.2: flatness-metrics на non-sentinel. Скрипт использует все позиции. Однако E-39a baseline 0.0162 считался тоже на всех triples, так что **согласованность с E-39a важнее** — не фильтровать здесь правильно. Но в документе стоит явно записать: «sentinel не фильтруется для консистентности с E-39a».

### P1.8 — L19 лишний
`L_CACHED = [6,12,18,19,22,23]`, gate — `L_GATE = [6,12,18,22,23]`. L19 попадает в JSON `metrics[19]`, но не в gate. Не баг, но лучше либо убрать, либо явно пометить `"extra": True`.

## 4. Новые находки

### N1 — `lp_t[real]` с mask на CPU, tensor на CUDA
```python
real = attn_ppl_cpu[c0:c1, 1:].bool()  # CPU
lp_t = lp[:, :-1, :].gather(2, target.unsqueeze(2)).squeeze(2)  # CUDA
nll_sum += float((-lp_t[real]).sum().item())
```
В большинстве версий PyTorch CPU-boolean-mask для CUDA-тензора **должен работать** (индексы — это «данные», не тензор), но в некоторых версиях/сборках это может выбросить `RuntimeError: Expected all tensors to be on the same device`. Надёжнее:

```python
real = attn_ppl_dev[c0:c1, 1:].bool()  # GPU-версия
# или
real = attn_ppl_cpu[c0:c1, 1:].bool().to(lp_t.device)
```
Это не «тихий» баг (упадёт громко), но стоит поправить до запуска.

### N2 — нет self-check для κ и β/σ₁
Self-check есть только для J-реконструкции (~1e-7). Но:
- `metrics[23]["fp16"]["kappa"]` должен быть ≈ 0.0162 (E-38d / E-39a). Если fp32-SVD здесь даст 0.025, gate может переключиться `pass → moderate` — результат изменится.
- `metrics[23]["fp16"]["beta_sigma1"]` должен быть ≈ 0.882 (E-38e).

Оба числа уже известны из предыдущих экспериментов и могли бы быть assert-проверены. Сейчас их нет → мы узнаем о расхождении только по факту запуска.

**Фикс**:
```python
assert abs(metrics[23]["fp16"]["kappa"] - 0.0162) < 0.005, \
    f"κ(L23) fp16 = {metrics[23]['fp16']['kappa']:.4f}, expected ~0.0162"
assert abs(metrics[23]["fp16"]["beta_sigma1"] - 0.882) < 0.05, \
    f"β/σ₁(L23) fp16 = {metrics[23]['fp16']['beta_sigma1']:.3f}, expected ~0.882"
```

### N3 — CPU + fp16 = медленно
`torch_dtype=torch.float16` задан безусловно. На CPU (а E-39 серия — CPU-first) fp16-forward в 5–10× медленнее fp32. Для PPL 96 форвардов по `[8, 128]` это часы. Стоит:
```python
model_dtype = torch.float16 if dev.type == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_P, torch_dtype=model_dtype, ...)
```
При этом `W_lm.double()` всё равно даёт fp64 — self-check и метрики не пострадают.

### N4 — `Wq.float()` в цикле PPL
```python
for name, Wq in {"base": W_lm64, ...}.items():
    ...
    logits_c = h @ Wq.float().T
```
Для `base` (W_lm64 fp64) `Wq.float()` создаёт fp32-копию ~1 ГБ **на каждом из 4 чанков**. Итого 4 аллокации × 1 ГБ + 4 освобождения. На T4 может привести к фрагментации. Лучше один раз: `W_lm32 = W_lm.float()` — и для base использовать `W_lm32` (fp32-матрица для matmul, при этом fp64 остаётся только для сборки `M_K`).

### N5 — `invperm` не используется
```python
perm = torch.argsort(H.diagonal(), descending=True)
invperm = torch.argsort(perm)  # dead code
```
Мёртвая переменная. Не баг, но загрязняет.

### N6 — C4 streaming может быть медленным
`_c4_docs` стримит полные C4-документы (иногда тысячи токенов) и токенизирует полный текст. Для 128 доков калибровки и 32 PPL — терпимо, но **скорость** на CPU может быть 10–30 мин. Стоит либо использовать `streaming=False` с кэшем, либо ограничить `max_length` в токенизаторе ещё до полной токенизации (он уже ограничен `truncation=True, max_length=s_len`, но полная токенизация длинного текста всё равно выполняется — HF-токенизатор не может «обрезать» без токенизации).

**Фикс** (опционально): `tokenizer(texts, max_length=s_len, truncation=True, ...)` **без** `padding="max_length"` в первом проходе — просто чтобы получить длину; не критично.

## 5. Итоговая таблица

| Приоритет | Пункт | Статус |
|---|---|---|
| **P0.1** | char-slice → token-slice | ✅ исправлено |
| **P0.2** | padding в GPTQ-H | ✅ исправлено |
| **P0.3** | PPL = mean NLL | ✅ исправлено |
| **P0.4** | двойной RMSNorm | ⚠️ warning вместо error; P0.4-check memory-waste |
| **P0.5** | tied-weights readout-only | ⚠️ документировано (`readout_only_ptq: True`), не исправлено |
| **P1.1** | L<23 synthetic J | ⚠️ документировано в комментарии, но **не в JSON** |
| **P1.2** | `_cos_ug_1k` — proxy | ⚠️ осталось (consistent с E-33r) |
| **P1.3** | нет sentinel-фильтра | ⚠️ согласовано с E-39a; стоит записать |
| **P1.4** | seed / deterministic | ✅ на CUDA; на CPU не выставлено |
| **P1.5** | del после self-check | ✅ |
| **P1.6** | gate absolute | ✅ добавлено relative |
| **P1.7** | int8 → int8_rtn | ✅ |
| **P1.8** | L19 extra | ⚠️ осталось |
| **N1** | `lp_t[real]` cross-device | ⚠️ возможен runtime-error |
| **N2** | нет κ/β self-check | ⚠️ рекомендуется добавить |
| **N3** | CPU+fp16 медленно | ⚠️ исправить до запуска |
| **N4** | `Wq.float()` в цикле | ⚠️ оптимизация памяти |
| **N5** | `invperm` unused | ⚠️ cleanup |
| **N6** | C4 streaming медленный | ⚠️ терпимо |

## 6. Что я бы сделал до запуска (минимальный набор)

1. **N1**: `real = attn_ppl_dev[c0:c1, 1:].bool()` (использовать GPU-копию).
2. **N3**: `torch_dtype = torch.float32 if dev.type == "cpu" else torch.float16`.
3. **N2**: добавить assert на `metrics[23]["fp16"]["kappa"]` ≈ 0.0162 и `beta_sigma1` ≈ 0.882.
4. **N4**: `W_lm32 = W_lm.float()` один раз вне PPL-цикла.
5. **P1.1**: добавить `"synthetic_readout_j": L != 23` в каждый `metrics[L][name]`.
6. **P0.4**: `raise` вместо `warn`, если `ref_err > 2.0` (или хотя бы явный `print` с «PPL is INVALID»).
7. **N5**: убрать `invperm`.

После этого — скрипт будет не только без «тихих» багов, но и с громкой защитой от дрейфа self-check в будущем.

## 7. Резюме

**P0-баги (P0.1–P0.3), которые реально портили числа, — устранены правильно.** P0.4/P0.5 — частично (check + документирование). Остались в основном P1-уровня: (a) L<23 synthetic J — «тихий» риск интерпретации JSON, (b) `_cos_ug_1k` proxy, (c) отсутствие self-check для κ/β. **Новый баг N1** (cross-device mask) — не «тихий», но потенциально runtime-error. N3 (CPU+fp16) — performance. N2/N4 — quality-of-life.

После минимального набора из §6 скрипт можно запускать: результаты E-39b будут корректны, gate на κ(L23) — осмыслен, `ΔPPL` — это настоящий `exp(NLL)`, GPTQ-H — чистый.
## 8. Раунд 2 — закрытие и прогон (v12)

Минимальный набор §6 + P1.3/P1.8 применены, прогон доведён до результата.

| Пункт | Итог |
|---|---|
| N1, N2, N3, N4, N5, P1.1, P1.8, P0.4 (`raise`) | ✅ применено |
| P1.3 | ✅ зафиксировано в шапке скрипта (sentinel-фильтр отсутствует сознательно, согласовано с E-38d/E-39a) |
| P1.2 | принято как есть: `cos(u_g, 1/√K)` — proxy, согласованный с E-33r/E-38e |
| P0.5 | принято как ограничение: readout-only PTQ; граница вынесена в paper §6 п.18 (deployment на tied-весах ≠ readout-слайс) |
| N6 | терпимо: весь прогон 532 с на T4 |

**Ещё два бага, всплывших при доведении (не из ревью):**
- **CUDA OOM**: полные копии `W_lm` `[151936,1024]` в fp64 (3 уровня ≈ 3.7 ГБ) держались на GPU
  одновременно с base/quant `J` (по 2.15 ГБ) → падение в `_recompute_J`. Фикс: копии весов живут
  на CPU, в цикл по слоям уходит только `M_K [128,1024]`; PPL поднимает по одному fp32-уровню;
  `_recompute_J` пишет в срез выхода без лишного `[nc,S,K,D]` временного.
- **HTTP 409 на `kaggle kernels push`**: `id` в metadata ≠ slug из `title`. Канонический slug —
  `e-39b-ptq-perturbation-gate-gptq-4-bit-lm-head`; исправлено в metadata и в `push_all.py`.

**Прогон (v12, T4, 532 с): гейт `pass`.** Самопроверки: base-J vs кэш max 7.77e-07 (tol 1e-5),
base κ(L23) = 0.01623 ≡ E-38d, β/σ₁ = 0.8821 ≡ E-38e. L23: κ base 0.016233 / int8 0.016245 /
gptq4 0.016043 (ratio 0.988), β/σ₁ 0.8821→0.8822, cos 0.9168→0.9171, J_rel_drift gptq4 0.1205,
PPL 30.4753 → 30.7498 (+0.90%). Текст результата: `docs/E39.md` §2.7, карточка — `docs/ref-exp.md`
(E-39b), следствия — `docs/paper.md` §4.8/§5/§6 п.18.
