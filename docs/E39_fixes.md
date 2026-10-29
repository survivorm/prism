# Ревью E-39b (`exp39b_kl_surgery.py`)

## 0. Фундаментальное несоответствие ТЗ

**Скрипт НЕ реализует E-39b из `E39.md` §2.**

ТЗ §2 явно описывает эксперимент про **GPTQ-4bit**:
- §2.2: «Стандартный PTQ (GPTQ/AWQ) квантует `lm_head.weight` → `M_K` → `J_t`»
- §2.4: «GPTQ 4-bit (group_size=128, desc_act=True, damp=0.01). Восстановить fp32 эмуляцию: `W_q = dequantize(W_q)`»
- §2.4: слои `{6, 12, 18, 22, 23}`, метрики `β/σ₁(M_K)`, `cos(u_g, 1/√K)`, `κ(L)`, `J_rel_drift`, PPL
- §2.6: 3 режима по κ после PTQ: `<0.02`, `0.02–0.10`, `>0.10`

**Что делает скрипт:**
- Никакого GPTQ (нет group_size, desc_act, damp, dequantize)
- Только L23, только хирургия `M_K`
- Метрика — средний KL(P_edit ‖ P_base), порог 0.05; метрик `β/σ₁`, `cos(u_g,1/√K)`, PPL нет
- **Это гибрид идей E-39c (§3, surgery) с KL-гейтом, не описанным в ТЗ вообще.**

Возможные объяснения: (a) ТЗ в файле устарел; (b) ID `E-39b` занят другим экспериментом; (c) фазы переупорядочены и скрипт — это pre-gate для E-39c, названный `E-39b` по ошибке. В любом случае **это надо прояснить до запуска**: либо переименовать, либо реализовать §2.

---

## 1. Критический тихий баг — **неверный delta в W_lm**

### 1.1. Суть

```python
M_K = build_M_K(W_lm, list(U128), g).double().detach()  # M_K[k,d] = W_lm[U128[k],d] * g[d]
...
M_kp = U_top @ torch.diag(s_top) @ V_new.T + M_rest    # surgery result, в M_K-пространстве
dM = (M_kp - M_K)[None, :].float()                      # delta в M_K-пространстве
W_lm.data[list(U128)] += dM[0]                          # ← BUG
```

Дельта `dM` живёт в пространстве `M_K = W_lm·diag(g)`, но прибавляется к `W_lm` напрямую. Правильно (и это явно прописано в ТЗ E-39c §3.3, шаг 6):

```python
W_lm.data[list(U128)] = (M_kp / g[None, :]).float()
# или, эквивалентно по дельте:
W_lm.data[list(U128)] += (dM / g[None, :]).float()
```

### 1.2. Последствия

Достигнутая `M_K_new` в модели:

```
W_lm_new = W_lm_old + dM
M_K_new  = W_lm_new · g = M_K_old + (M_kp − M_K_old)·g
         = M_K_old·(1−g) + M_kp·g
```

- При `g ≡ 1` (необученный RMSNorm) — случайно работает.
- При `g ≠ 1` — **эффективная хирургия масштабирована по строкам на `g`**. Модель делает НЕ то, что показывает `kappa_after_surgery_27b_cache` (последнее считается из `M_kp`, а не из достигнутой `M_K_new`).

**Это тихо ломает всё:**

1. `kappa_after_surgery_27b_cache ≈ 0.004` — верно для **задуманной** хирургии.
2. Фактический `κ'` модели — другой, не измеряется.
3. Измеренный KL — для **фактической (неверной)** хирургии.
4. Вывод «surgery flatten до κ'=0.004 при цене KL=0.05» **не подтверждён**: измеряется не то, что рекламируется.

Защита `assert g.min() > 1e-6` проверяет только `g > 0`, **не** `g ≡ 1`. Так что баг молча проходит.

**Severity: HIGH — центральный результат эксперимента недостоверен.**

**Fix:**
```python
assert abs(g - 1).max() < 1e-3, f"γ ≠ 1 (max |γ−1| = {(g-1).abs().max():.3g}); surgery formula must divide by γ"
W_lm.data[list(U128)] = (M_kp / g[None, :]).to(W_lm.dtype)
```
(либо явно `dM / g` для delta-варианта).

---

## 2. Второй тихий баг — **разрыв между κ' и KL**

```python
u = (H / H.norm(...))         # û_t от base-модели (кэш)
...
Jp = M_kp[None,None] - Mcu[...,None]*u64[:,:,None,:]  # J' от задуманной M_kp с base û_t
kap_new, ... = _kappa_from_J(Jp.float(), S_LEN, R)
```

κ' считается по формуле `J' = M_kp·(I − û_t û_t^T)` с `û_t` **из base-модели**. Но после `W_lm[U128] += dM` (tied weights) меняется `embed_tokens`, а значит и `û_t` **в реальной edited-модели**. То есть:

- `kappa_after_surgery_27b_cache` = κ' **dry-run**.
- KL — мерится на **реальной модели** с другими `û_t`.
- **Никакой связи между числом κ' и числом KL нет** — они описывают разные объекты.

Даже если бы БАГ-1 был исправлен, `κ'` в JSON = «предсказание по формуле», а не измерение. Для честной проверки надо: (а) после edit прогнать forward, (б) вытащить `h_t`, (в) посчитать `J_t = W_lm_new[U128]·(I − û_t^new û_t^new,T)`, (г) из него `κ'`. Иначе результат — «dry-run рядом с реальным KL».

**Fix:** либо (а) добавить post-edit κ-верификацию, либо (б) в `notes` явно написать «κ' — dry-run preview, не измерение», а в `gate_pass` опираться только на KL (это уже так), но и в выводах не писать «surgery flattened to κ'=…».

---

## 3. Тихий баг — `row_diag` смешивает пространства

```python
row_old = W_lm[list(U128)].double().cpu()      # [128, D], W_lm-пространство
drow = (M_kp - M_K)                             # [128, D], M_K-пространство
dratio = drow.norm(dim=-1) / row_old.norm(dim=-1).clamp_min(1e-9)
```

Числитель в M_K-пространстве, знаменатель в W_lm-пространстве → диагностика «доля изменения в строке» неверна на множитель `g`. При `g ≠ 1` `mean_drow_over_row` систематически смещён.

**Fix:** либо `drow_correct = drow / g[None, :]`, либо `row_old = M_K` (тогда в M_K-пространстве оба).

---

## 4. Громкие / средние

### 4.1. Missing κ'-consistency assert

```python
print(f"κ (27b cache, J'): {kap_new:.5f} (39a dry-run ≈ 0.0042)")
```

Никакой проверки — только печать. Если κ' выйдет 0.05 (surgery не сработала), скрипт всё равно продолжит и вынесет `gate_pass` по KL. Нужно:
```python
if abs(kap_new - 0.0042) > 0.003:
    warnings.warn(f"κ' mismatch: {kap_new:.5f} vs 39a dry-run 0.0042")
    res["consistency"] = "MISMATCH"
```

### 4.2. Memory footprint

`Jp = M_kp[None,None] - Mcu[...]*u64[...]` даёт **double [32,128,128,1024] = 4.2 GB**, потом `Jp.float()` — ещё 2.1 GB. На 64 GB CPU сработает, но:
- На 32 GB машине — OOM.
- `_kappa_from_J` внутри итерирует по docs, материализация всего `Jp` не нужна.

**Fix:** считать per-doc в цикле, или сразу в float32, или хранить в `float16` (для SVD top-16 достаточно).

### 4.3. Device mismatch потенциальный

```python
row_old = W_lm[list(U128)].double().cpu()      # на CPU
drow = (M_kp - M_K)                             # на устройстве модели
dratio = drow.norm(dim=-1) / row_old.norm(...)  # CPU vs device → runtime error
```

Сейчас скрипт CPU-only, но при переезде на GPU — упадёт. Или, если `g` не CPU — `dM/g` упадёт.

### 4.4. `V_new` хардкодит fallback-ветку E-39a

```python
_, sU, VU = torch.linalg.svd(U, full_matrices=False)
P_null = VU[:, -R:]
```

E-39a явно различал `null_idx` (истинное нуль-пространство) и `VU[:, -R:]` (bottom-SVs). E-39b безусловно использует bottom-SVs. Для типичного `U: [4096, 1024]` (rank 1024) это совпадает, но:
- Если E-39a когда-нибудь получит истинное нуль-пространство (например, при малом N), E-39b будет использовать **другой** `V_new` → κ' разойдётся.
- Проверки `assert dim(P_null) == R` нет — если `VU.shape[1] < R` (теоретически), `VU[:, -R:]` даст меньше столбцов, а `torch.eye(R)` в assert провалится поздно.

**Fix:** либо импортировать `_obs_subspace(VU, sU, R)` из E-39a, либо явно повторить ветку и заassert-ить.

### 4.5. `W_lm.data[list(U128)] += dM[0]` — in-place edit shared tensor

Для tied weights `W_lm is embed_tokens`. `+=` in-place меняет и `embed_tokens` — заявлено в notes, но:
- После forward-1 (`lp_base`) кэш активаций испорчен не будет (нет кэша), ок.
- Но если модель внутри хранит какую-то копию или shared-views — возможны сюрпризы.
- **Безопаснее**: `W_lm.data[list(U128)] = (W_lm[list(U128)] + dM[0] / g[None,:])` без `+=` (in-place на подмножестве строк в shared-tensor может нечисто отработать с некоторыми реализациями autograd-views, хотя `no_grad` контекст спасает).

Не блокер, но лучше явное присваивание.

---

## 5. Методологические тихие смещения

### 5.1. Padding в mean KL

```python
enc = tokenizer(texts, truncation=True, max_length=s_len, padding="max_length", ...)
...
kl = (p_edit * (lp_edit - lp_base)).sum(dim=-1)      # [N, S]
kl_mean = float(kl.mean())                            # включает padded-позиции
```

C4 wiki: часть документов короче 128 токенов. Их padded-позиции — EOS-pad, где модель предсказывает EOS на `[pad]*` контексте. KL там — это «шум на EOS». Включается в среднее. Может искажать и вверх, и вниз.

**Fix:** считать `mean` по реальным токенам (`attention_mask`), отчитывать отдельно `mean_without_padding` и `mean_all`.

### 5.2. Позиция t=0 в mean KL

`kl.mean()` включает `t=0`. На t=0 контекста нет; logits ≈ prior, KL ≈ «насколько edit сдвинул prior». В то время как основная часть KL — из контекстных позиций. Вклад t=0 непропорционален. **Fix:** отчитывать с/без t=0.

### 5.3. «Fresh C4 disjoint from 27b/39a cache» — не проверено

```python
notes: "Fresh C4 docs (streaming, first 8) — disjoint from the 27b/39a training cache"
```

`load_dataset("allenai/c4", "en", split="train", streaming=True)` берёт **первые** 8 документов потока. Если 27b cache тоже брался из первых N C4 train, будет **пересечение**. Утверждение «disjoint» не верифицировано ни хешем, ни seed-offset.

**Fix:** либо использовать `split="validation"` для eval, либо `ds.skip(SKIP).take(8)` с явным SKIP > N_train_cache, либо сверять sha256 текстов.

### 5.4. Threshold 0.05 nats/token не обоснован

`KL_PASS = 0.05` — введён в скрипте. В ТЗ §0.1 и E-16 используется 0.1 (KL leash) и 0.24. Почему 0.05 — не объяснено. Это может быть и «строгий» порог, и «произвольный». **Fix:** либо обосновать в комментарии (сравнение с E-16 KL leash), либо взять 0.1 (согласованность с E-16).

### 5.5. Tied-weights effect не отмечен как limitation

Notes: «single row-edit realizes both Δlm and Δembed of the (untied) spec — `J_t = M_K(I−ûûᵀ)` moves consistently». Это **неверное утверждение**: untied-спека «поменять только lm_head, `û_t` не менять» не реализуется tied-исполнением — tied меняет и embed, и `û_t`, и forward. «Consistently» — натяжка.

E-39c ТЗ §3.5 честно пишет: «Для E-39c: это серьёзное ограничение. Возможно, использовать не-tied версию… или принять это как часть эксперимента». E-39b это не отражает в limitations. **Fix:** добавить в `limitations`.

---

## 6. Мелочи

- `from _e38_cache import S, D, load_layer` — `S` не используется (есть `S_LEN`). Мелочь.
- `_logprobs` возвращает `log_softmax` по всему словарю [V=151936] — ок, но `lp_base` [8,128,151936] fp32 = 622 MB × 2 (base+edit) + p_edit ещё 622 MB → ~2.5 GB пик на KL. Приемлемо, но стоит упомянуть.
- `vocab` в `params` — только для отчёта, не участвует в метриках.
- `assert W_lm.shape[1] == D` — проверяет только dim, не vocab. Ок.
- `_kappa_from_J` использует `Jp.float()` — float32 SVD. Для цели κ ≈ 0.004 это на грани float32-точности. Не критично, но стоит помнить при κ' < 0.003.
- `dratio` диагностика не помечена как «в M_K-пространстве» — читатель может решить, что это доля изменения W_lm.
- `m_k_norm.before == m_k_norm.after` должен быть точно (Frobenius сохраняется при ортогональном повороте топ-16) — это хороший **тест на корректность** `M_kp` построения; сейчас просто репортится, не assert-ится.

---

## 7. Резюме

**Блокеры перед запуском:**

1. **§0** — решить, что такое E-39b: GPTQ (как в ТЗ §2) или KL-gated surgery (как в скрипте). Сейчас они несовместимы.
2. **§1 (Q1)** — исправить `W_lm[U128] += dM` → `W_lm[U128] = M_kp / g` (или `+= dM/g`). Без этого весь эксперимент невалиден; KL меряет не ту хирургию, κ' — не той модели.
3. **§2 (Q2)** — либо добавить post-edit κ' измерение, либо явно в `notes` отделить dry-run κ' от реального KL.

**Сильные рекомендации:**

4. **§3** — `row_diag` привести к одному пространству.
5. **§4.1** — assert/warning на κ' vs E-39a.
6. **§4.2** — считать `Jp` per-doc в float32 (память 4.2 GB → ~0.3 GB на doc).
7. **§5.1–5.2** — отчитывать KL с/без padding и с/без t=0.
8. **§5.3** — либо verification disjointness, либо явно в `limitations`.
9. **§5.5** — добавить tied-weights в `limitations` (как в E-39c ТЗ §3.5).

**Мелкое:**

10. Убрать импорт `S`, если не используется.
11. `KL_PASS` обосновать или привести к 0.1 (согласованно с E-16).
12. `m_k_norm` assert-ить на равенство (Frobenius preserved) — хороший smoke-test.

**Главный вывод.** Один тихий баг (§1, delta в неправильном пространстве) + один методологический разрыв (§2, κ' — dry-run, KL — real) вместе делают **центральный результат эксперимента недостоверным**. Оба исправимы, но без них ни `gate_pass`, ни `kappa_after_surgery_27b_cache` нельзя интерпретировать как «эластичность уплощена ценой KL=0.05».

# ТЗ: правка E39.md §12 и `exp39l_drift_behavior.py`

**Версия:** draft для ревью
**Дата:** 2026-10-27
**Связано с:** E-39l (paper v5.0 §6 п.6, §12 E39.md), E-38i, E-16b, E-39d/e
**Статус:** блокирует публикацию E-39l в paper §5 synthesis

---

## 0. Контекст и цель

### 0.1. Что решаем

Текущая E-39l (E39.md §12 + `exp39l_drift_behavior.py`) **не может дать валидный ответ** на вопрос «предсказывает ли drift листа поведенческую деградацию». Причина — design flaw, не bug: обе серии (`J_rel_drift_max(s)`, `top-p retention(s)`) монотонны по одному и тому же аргументу `s_frac = step/1000`, поэтому Spearman между ними **тождественно −1** независимо от данных. Скрипт по конструкции получит `ρ = −1`, `verdict = "drift_predictive (Goodhart geometric)"` — вывод, который **не следует из данных** и попадёт в §5 synthesis как «установлено».

### 0.2. Что фиксим

Два уровня:
1. **E39.md §12** — заменить однотестовый дизайн на многотестовый с явным различением «trivial monotone co-drive» и «predictive beyond trend»; переформулировать вердикты.
2. **`exp39l_drift_behavior.py`** — реализовать новый дизайн, устранить тихие баги (silent skip, hardcoded E-16b, отсутствие anti-control, лживые docstrings).

### 0.3. Что НЕ фиксим

- Идею join E-38i × E-16b — она верна как идея.
- Пути к файлам (`results/exp38i_factorization.json`, `results/exp39l_drift_behavior.json`).
- Требование CPU-only и deterministic.

### 0.4. Acceptance summary

После правок:
- `verdict` в JSON не может принимать значение `drift_predictive` **без** явного `rho_residual` и `anti_control_passed = true`.
- Скрипт явно сообщает `n_measured`, `n_interpolated`, `monotonicity_ok`, `rho_all`, `rho_residual`, `rho_anti_control`.
- При `ρ_res ≈ 0` вердикт — `shared_trend_only`, а не `drift_predictive`.

---

## 1. Диагноз (зачем переписывать, а не «улучшать»)

### 1.1. Root cause

Spearman инвариантен к монотонным преобразованиям. Если `drift(s)` строго возрастает, а `retention(s)` строго убывает по `s ∈ {s_1,…,s_n}`, то `rank(drift) = rank(s)` (с точностью до направления), `rank(retention) = −rank(s)`, и `ρ(drift, retention) = −1` **при любых значениях**, согласованных с монотонностью.

**Следствие:** тест не различает «drift предсказывает retention» и «оба убывают/возрастают с s». Обе гипотезы дают `ρ = −1`.

### 1.2. Усугубляющий фактор

9 из 11 точек в E-38i-join — **интерполированы** `np.interp` по 6 узлам E-16b. Внутри каждого сегмента retention строго линейна по `s`. Значит `ρ = −1` гарантирован на 9 точках **по построению**, а на двух оставшихся (step 0, 1000) — по монотонности. Остаточная информация о связи: 0.

### 1.3. Почему нельзя «просто добавить точек»

Даже с 100 измеренными точками `(drift(s), retention(s))` при монотонности обеих **ρ = −1** останется. Нужен **другой тест**, не инвариантный к монотонной перепараметризации.

### 1.4. Что будет, если оставить как есть

`paper.md §5` синтез пополнится строкой «Установлено [И]: drift листа предсказывает поведенческую деградацию (Goodhart геометрический)». При ревью это можно опровергнуть одним абзацем (§1.1). Paper v6.0 рискует получить reject по этому пункту.

---

## 2. Правки в E39.md §12

### 2.1. Новый §12.2 — Определения

**§12.2.1 (Predictive beyond trend).** Будем говорить, что `drift` *predictive beyond trend* для `retention`, если после удаления монотонного тренда `s ↦ drift` и `s ↦ retention` остатки коррелируют значимо отрицательно:

```
ρ_res = Spearman(resid_drift, resid_retention) < -0.3,  p < 0.10
```

где `resid_*` — остатки после изотонической (или монотонной loess) аппроксимации по `s_frac`.

**§12.2.2 (Shared trend only).** Противоположный случай: `|ρ_res| < 0.2`. Это означает: **единственное, что несут обе серии — общий монотонный тренд по s**; специфической связи drift→retention нет.

**§12.2.3 (Тривиальная корреляция).** `ρ_all = Spearman(drift(s), retention(s))` на полном наборе точек. Помечается как «trivial» (в смысле §1.1); репортится для прозрачности, **не используется** в вердикте.

**§12.2.4 (Anti-control).** Для каждой точки `i` пары `(drift(s_i), retention(s_{π(i)}))`, где `π` — случайная перестановка. `ρ_anti = median_{B=1000} |Spearman|`. Если `ρ_anti` близко к `|ρ_all|`, тест не различает сигнал и шум; иначе — различимо, но всё ещё может быть тривиальным по §1.1.

### 2.2. Новый §12.3 — Протокол (заменяет текущий §12.3)

**Primary metric** (решает вердикт):
1. Загрузить `drift(s)` для `s ∈ E38I_STEPS`.
2. Загрузить `retention(s)` **из источника E-16b** (не hardcode; см. §2.4).
3. Отфильтровать `s`-точки, где drift определён (fail-loud при пропусках; см. §2.5).
4. Проверить монотонность `drift(s)` (см. §2.5).
5. Изотонически детрендить обе серии по `s_frac`.
6. Вычислить `ρ_res = Spearman(resid_drift, resid_retention)`, `p_res`.
7. Вычислить `ρ_all` (для отчёта, помеченный как «trivial»).
8. Вычислить `ρ_anti` (anti-control, B=1000).

**Secondary** (информативные, если первичный даст `|ρ_res| < 0.2` и хочется дожать):
- **§12.3.1 Half-split** (при наличии `n ≥ 12`): подогнать `retention ≈ f(drift)` (isotonic) на нечётных индексах `s`, измерить `ρ_res` на чётных. Если `n < 12` — skip с явной пометкой.
- **§12.3.2 Seed-noise** (при наличии ≥2 seeds на точку E-16b): коррелировать seed-флуктуации drift и retention. **Сейчас недоступно** — E-16b n=6 без повторностей. Пометка в `limitations`.

**Что не входит в primary:** `ρ_all` — только репорт. `ρ_anchors` (только 2 измеренные точки) — репорт с пометкой «n=2, не информативен».

### 2.3. Новый §12.4 — Интерпретации (заменяет текущий §12.4)

Вердикт вычисляется по `ρ_res` **только** при `anti_control_passed = true` (`ρ_anti > 0.3` или явная диагностика: «anti-control не подтверждает, что тест различает сигнал и шум»).

| `ρ_res` | `p_res` | `ρ_anti` | Вердикт |
|---|---|---|---|
| `≤ −0.5` | `< 0.05` | `> 0.3` | `drift_predictive_beyond_trend` (сильный) |
| `≤ −0.3` | `< 0.10` | `> 0.3` | `drift_predictive_beyond_trend` (умеренный) |
| `|ρ_res| < 0.2` | any | any | `shared_trend_only (drift not predictive)` |
| `0.2 ≤ ρ_res < 0.3` | any | any | `inconclusive` |
| `> 0.3` | any | any | `paradoxical (проверить знак тренда)` |
| any | any | `≤ 0.3` | `anti_control_insufficient (тест не различает)` |

**Правило:** вердикт `drift_predictive_beyond_trend` **не может** быть вынесен, если `anti_control_passed = false` или если `n_measured < 2` или если `monotonicity_ok = false` (см. §2.5, немонотонность меняет интерпретацию `ρ_res`).

### 2.4. Новый §12.4.5 — Semantics of s=0

E-16b: `retention(s=0) = 0.837`, не 1.0. Это означает, что `s=0` в E-16b — **не базовая модель**, а состояние после E-16a. Два обязательных требования к скрипту:

1. **Сверить semantics** `s=0` в E-16b и `step=0` в E-38i. Если `E-38i step=0` — базовая модель (drift=0 относительно base), а E-16b `s=0` — уже отредактированная, то **склейка в `s_frac=0` некорректна**.
2. В `caveats` явно: «E-16b retention(0)=0.837 — состояние после E-16a, не base; drift(s=0) в E-38i может быть относительно другой точки отсчёта. Join at s_frac=0 требует явной проверки, иначе исключить точку».

**Рекомендация:** исключить `s_frac=0` из join (оставить n=10) до верификации semantics. Если верификация возможна — включить, но пометить.

### 2.5. Новый §12.4.7 — Обязательные проверки (fail-loud)

До подсчёта `ρ_res`:

1. **Загрузка E-16b из источника.** Если файл `results/exp16b_*.json` существует — загрузить, `assert` совпадение endpoints `(s=0, s=1.0)` с hardcoded (погрешность 0.01); при mismatch — `warnings.warn`, использовать источник. Если файла нет — использовать hardcoded, явно в `caveats`.
2. **Монотонность drift.** `assert` `drift(s_i) ≤ drift(s_{i+1}) + 1e-6` для всех i. При нарушении — `monotonicity_ok = false`, вердикт `inconclusive (monotonicity violated — test may be informative)`.
3. **Масштаб s.** `assert max(E38I_STEPS) == 1000`. Иначе — fail-loud, потому что `s_frac = step/1000` некорректен.
4. **n_min.** `assert n_joined ≥ 6`. При меньшем — fail-loud (не запускать).
5. **Missing steps.** `warnings.warn` при пропусках, `res["missing_steps"] = [...]`.

### 2.6. Новый §12.5 — Sources of truth

Явно указать:
- `results/exp38i_factorization.json` — обязателен, ключ `per_state[str(step)]["J_rel_drift_max"]`. Если схема изменится — `KeyError` (громко), не silent skip.
- `results/exp16b_*.json` — желателен; схема `{"s": [...], "retention": [...]}`. Если нет — hardcoded fallback с явной пометкой в JSON.

### 2.7. Новый §12.6 — Limitations (для paper)

Добавить в paper §6 «Ограничения»:

> **E-39l:** два независимых ограничения. (а) Обе серии монотонны по `s_frac`, поэтому `ρ_all = −1` — **тривиальная корреляция**, не информативна; используется только `ρ_res` (после детрендинга) и anti-control. (б) `ρ_res` измеряет *связь остатков*, но **не причинность**: если оба ряда ведомы третьим фактором (например, деградацией `W_lm`), `ρ_res = 0` при наличии каузальной связи. E-39l отвечает на вопрос «есть ли специфическая связь drift↔retention сверх тренда», но не «drift вызывает деградацию». Для каузального теста нужен E-39d/e (training с drift-loss vs flatness-loss).

---

## 3. Правки в `exp39l_drift_behavior.py`

### 3.1. Структурные изменения (аддитивные)

| Раздел | Что добавляется |
|---|---|
| Загрузка E-16b | `_load_e16b()` с fallback на hardcoded + `assert` endpoints |
| Метрики | `_detrend(s, y)` (isotonic), `_residual_spearman`, `_anti_control` |
| Verdict | `_verdict(rho_res, p_res, rho_anti, n_measured, monotonicity_ok)` |
| JSON output | новые поля: `rho_residual`, `rho_all_trivial`, `rho_anti_control`, `n_measured`, `n_interpolated`, `monotonicity_ok`, `anti_control_passed`, `e16b_source`, `verdict_confidence` |
| Assertions | `assert n_joined >= 6`, `assert max(E38I_STEPS) == 1000`, monotonicity check |
| Warnings | missing steps, E-16b mismatch, non-monotone drift |

### 3.2. Ключевые фрагменты (псевдокод)

```python
# --- load E-16b from source (with fallback) ---
def _load_e16b(repo, fallback):
    src = os.path.join(repo, "results", "exp16b_dose_response.json")
    if not os.path.exists(src):
        return dict(fallback), {"source": "hardcoded", "verified": False}
    data = json.load(open(src))
    # verified: assert endpoints
    for s, r in [(0.0, 0.837), (1.0, 0.416)]:
        got = float(np.interp(s, data["s"], data["retention"]))
        assert abs(got - r) < 0.01, f"E-16b mismatch at s={s}: {got} vs {r}"
    return dict(zip(data["s"], data["retention"])), {"source": src, "verified": True}


# --- detrend (isotonic, simplest monotone fit) ---
from sklearn.isotonic import IsotonicRegression
def _detrend(s, y):
    iso = IsotonicRegression(increasing='auto').fit(s, y)
    return y - iso.predict(s)


# --- anti-control: shuffle retention across s, B=1000 ---
def _anti_control(s, drift, retention, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(B):
        perm = rng.permutation(len(retention))
        r, _ = stats.spearmanr(drift, retention[perm])
        rhos.append(r)
    return float(np.median(np.abs(rhos))), float(np.percentile(np.abs(rhos), 95))


# --- verdict (strict, no drift_predictive without anti-control) ---
def _verdict(rho_res, p_res, rho_anti, n_measured, monotone_ok):
    if not monotone_ok:
        return "inconclusive (drift non-monotone — rho_res may be informative, see diagnostics)"
    if rho_anti <= 0.3:
        return "anti_control_insufficient (test does not distinguish signal from noise)"
    if n_measured < 2:
        return "insufficient_measured_anchors"
    if rho_res <= -0.5 and p_res < 0.05:
        return "drift_predictive_beyond_trend_strong"
    if rho_res <= -0.3 and p_res < 0.10:
        return "drift_predictive_beyond_trend_moderate"
    if abs(rho_res) < 0.2:
        return "shared_trend_only (drift not predictive beyond trend)"
    if rho_res > 0.3:
        return "paradoxical (check sign, detrending failure)"
    return "inconclusive"
```

### 3.3. Порядок проверок в `main()`

1. Загрузить E-38i JSON. **Fail-loud** при отсутствии/неправильной схеме.
2. Загрузить E-16b (source или fallback). Записать `e16b_source` в результат.
3. `assert max(E38I_STEPS) == 1000`.
4. Filter, `warnings.warn` при missing steps. `assert n_joined ≥ 6`.
5. Проверить монотонность drift. `monotonicity_ok = bool(...)`.
6. Detrend обе серии.
7. `rho_res, p_res = _residual_spearman(resid_drift, resid_retention)`.
8. `rho_all, _ = _spearman(drift, retention)` — только для отчёта.
9. `rho_anti, _ = _anti_control(...)`.
10. `verdict = _verdict(...)`.
11. `verdict_confidence = "high" if anti_control_passed and n_measured ≥ 2 and monotone_ok else "low"`.

### 3.4. Новый JSON schema (v2)

Сохраняется обратная совместимость полей. Добавляются:

```json
{
  "experiment": "E-39l: drift vs behavior (v2, non-tautological design)",
  "tz": "docs/E39.md §12 (v2)",
  "verdict": "...",
  "verdict_confidence": "high|low",
  "metrics": {
    "rho_residual": {"rho": -0.31, "p": 0.09},
    "rho_all_trivial": {"rho": -1.000, "note": "tautological — both monotone in s_frac"},
    "rho_anti_control": {"median_abs": 0.12, "p95_abs": 0.34},
    "anti_control_passed": true,
    "monotonicity_ok": true,
    "n_joined": 10,
    "n_measured": 2,
    "n_interpolated": 8,
    "missing_steps": [],
    "s_frac_excluded": [0.0]
  },
  "e16b_source": {"path": "results/exp16b_...json", "verified": true},
  "points": [ ... ],
  "caveats": [
    "rho_all = -1.000 is tautological (§12.2.3)",
    "s_frac=0 excluded: E-16b retention(0)=0.837 is post-E-16a, not base (§12.4.5)",
    "Only 2 of 10 points measured; 8 interpolated (§12.3)",
    "rho_residual measures shared residual structure, not causation (§12.6)"
  ]
}
```

### 3.5. Что НЕ меняется

- Требование CPU-only, deterministic.
- Пути к файлам.
- `E38I_STEPS = [0, 10, 20, 40, 80, 90, 100, 200, 400, 800, 1000]`.
- `E16B` hardcoded — как fallback, не как основной источник.

### 3.6. Docstring (`_spearman`)

```python
def _spearman(x, y):
    """Spearman rho + two-sided p (t-distribution for small n, via scipy)."""
```

(заменяет «normal approx», что неверно).

### 3.7. Scatter plot

ТЗ §12.3 просит scatter. Добавить в конец `main()`:

```python
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(drift, retention, c="C0", label="joined points")
    for r in rows:
        ax.annotate(f"s={r['s_frac']:.2f}", (r["J_rel_drift_max"], r["retention_interp"]),
                    fontsize=7, alpha=0.6)
    ax.set_xlabel("J_rel_drift_max"); ax.set_ylabel("top-p retention (interp)")
    ax.set_title(f"E-39l v2: rho_all={rho_all:+.2f} (trivial), rho_res={rho_res:+.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp39l_scatter.png"), dpi=120)
    plt.close(fig)
except ImportError:
    res["scatter_plot"] = "matplotlib not installed, skipped"
```

---

## 4. Acceptance criteria

Скрипт проходит правку, если:

- [ ] Запускается на текущих `results/exp38i_*.json` без ручных правок.
- [ ] `verdict` никогда не равен `drift_predictive` (без суффикса `beyond_trend`) — старые ярлыки `drift_predictive (Goodhart geometric)` удалены.
- [ ] JSON содержит все поля из §3.4.
- [ ] При искусственном нарушении монотонности drift (тест: подать `drift = [0.5, 0.4, 0.3, ...]` в mock) — `monotonicity_ok = false` и вердикт `inconclusive`.
- [ ] При подаче пустого E-16b-источника (`results/exp16b_*.json` с другим набором endpoints) — `assert`/`warn` срабатывает.
- [ ] При `n_joined < 6` — fail-loud (исключение), не silent skip.
- [ ] При `max(E38I_STEPS) != 1000` — fail-loud.
- [ ] `caveats` содержит все четыре пункта из §3.4.
- [ ] Scatter plot создаётся (или явная пометка «skipped»).

**Дополнительно** (желательно, но не блокер):
- [ ] Anti-control с B=1000 отрабатывает за < 1 с на CPU.
- [ ] `_detrend` использует `sklearn.isotonic.IsotonicRegression` — если sklearn недоступен, fallback на линейный detrend (`np.polyfit` degree=1) с записью в `caveats`.

---

## 5. Риски и ожидаемые результаты

### 5.1. Вероятный исход

**Наиболее вероятно:** `ρ_res ≈ 0` (обе серии несут только общий s-тренд), вердикт `shared_trend_only`. Это **честный** результат: E-39l не подтверждает «drift предсказывает поведение» **и не опровергает**; он показывает, что текущие данные не позволяют это утверждать.

**Для paper §5:** строка «Установлено [И]: drift predictive» **удаляется**. Заменяется на:
> **Не установлено:** E-39l не различает «drift предсказывает retention» и «оба убывают с s». Специфическая связь (после детрендинга) `ρ_res ≈ 0`, механизм остаётся открытым.

Это **негативный, но валидный** результат. Он не блокирует paper v6.0, но требует переформулировки §5 и §6.

### 5.2. Альтернативный исход

**Если `ρ_res < −0.3`, p < 0.10, anti-control passed:** можно писать «drift predictive beyond trend», но **не** «Goodhart geometric» — последнее требует каузального теста (E-39d/e, см. §2.6 paper). Формулировка: «drift carries information beyond shared trend; causal test deferred to E-39d/e».

### 5.3. Отказоустойчивость

Если `results/exp16b_*.json` не существует и endpoints hardcoded не совпадают с paper §4.8 (0.837, 0.416) — **fail-loud** до запуска (проверка в `_load_e16b`).

---

## 6. Порядок выполнения

1. **Правка E39.md §12** (30 мин): переписать §12.2–§12.6 по §2.1–§2.7.
2. **Правка `exp39l_drift_behavior.py`** (2 ч): §3.1–§3.7.
3. **Sanity-run на mock-данных** (30 мин): проверить все acceptance criteria §4, особенно failure modes (monotonicity violation, missing steps, endpoint mismatch).
4. **Production run** (5 мин CPU): на текущих `results/exp38i_*.json`.
5. **Обновление paper.md §5, §6** (30 мин): переформулировать synthesis-строку и limitation.

**Блокирует:** paper v6.0, E-39d/e (если планируется заменить flatness-loss на drift-loss по результатам E-39l — сейчас **нельзя**, вывод будет `shared_trend_only`).

**Не блокирует:** E-39a, E-39b, E-39c, E-39d/e как отдельные эксперименты.

---

## 7. Что это ТЗ явно НЕ решает

1. **Каузальность.** Даже при сильном `ρ_res` E-39l не различает «drift → degradation» и «третий фактор → и drift, и degradation». Для каузального теста нужен E-39d/e (training с drift-loss vs flatness-loss на downstream).
2. **Power.** n=10 (или 11) даёт широкий CI на `ρ_res`. `|ρ_res| < 0.5` при p ≈ 0.1 — не «доказательство отсутствия связи», а «недостаточная мощность».
3. **Per-doc variability.** E-16b n=6, E-38i per-position max over 32 docs — агрегация разная. Даже при `ρ_res` значимом — это связь агрегатов, не per-position.
4. **Пересборка E-16b.** Если E-16b перезапустить с per-doc retention, можно получить n × 6 точек и уменьшить интерполяцию. Это — отдельный эксперимент, не входит в scope.

---

## 8. Ссылки

- `E39.md §12` — текущий дизайн E-39l
- `paper.md §4.8` (E-16b), §5 synthesis, §6 ограничения
- `exp39l_drift_behavior.py` — текущая реализация
- `results/exp38i_factorization.json`, `results/exp16b_*.json` (если существует)
- Ревью `exp39l_drift_behavior.py` от 2026-10-27 (диагноз §0–§5)