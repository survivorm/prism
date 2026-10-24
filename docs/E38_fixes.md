# Ревью `exp16_readout_pull.py` — сохранение данных + эксперимент + совместимость с E-38h

## 1. Что сохраняется (контракт данных)

### 1.1 Полный набор артефактов

| Файл | Поле | Формат | Размер |
|---|---|---|---|
| `exp16_dW_snapshots.pt` | `W_orig` | `W_orig[token_set]` fp16 `[256,1024]` | ~0.5 MB |
| | `snapshots[step]` | `(W_curr − W_orig)[token_set]` fp16 `[256,1024]` | ~0.5 MB × 100 |
| | `token_set` | `list[int]` | — |
| | `step` | `int` | — |
| | `final_dose` | fp32 scalar | — |
| `exp16_lm_head_final.pt` | `delta_W` | `(W_curr − W_orig)` fp16 `[V,1024]` | ~300 MB |
| | `step`, `bridge`, `dose` | — | — |
| `exp16_lm_head_knee_info.pt` | `knee_step`, `knee_dose`, `knee_bridge` | — | — |
| `exp16_readout_pull.json` | `curve` | `[{step, dose, bridge, KL}, ...]` | — |

### 1.2 **P0-проблема для потребителя (E-38h), не в E-16**

E-16 сохраняет **`snapshots[step] = ΔW = W − W_orig`** (см. код строки 488–491 и docstring).

E-38h (`exp38h_two_channel.py`, строки 48–50):
```python
snaps_all, raw = load_snapshots()
W_orig = raw["W_orig"].to(torch.float64)
dW = {k: snaps_all[k] - W_orig for k in ks}     # ← (ΔW) − W_orig = W_curr − 2·W_orig
```

Это вычитает `W_orig` **дважды**. Правильно: `dW = {k: snaps_all[k] for k in ks}` (снапшот уже ΔW).

**Следствие для E-16:** сохранение корректно. Проблема — контракт не задокументирован в `snap_payload` рядом с полем (только в комментарии), и потребитель его нарушил. В `snap_payload` стоит добавить:
```python
"schema": "snapshots[step] = (lm_head.weight − W_orig)[token_set]; dW = snapshots[step] (NOT − W_orig)"
```
а также в `exp38_hi_common.load_snapshots` — комментарий и, желательно, проверку `‖snaps[k]‖ ≪ ‖W_orig‖` (ΔW мал по сравнению с W_orig).

### 1.3 **P1: `W_orig` в fp16 — потеря точности для E-38h**

`snap_payload["W_orig"] = W_orig[token_set].to(torch.float16)` — W_orig компоненты `O(0.05)` (lm_head Qwen3.5-0.8B). fp16 абсолютная точность ~`5e-5`. ΔW компоненты `O(0.5)`, точность ~`5e-4`. Относительная ошибка ΔW ~`1e-3`. Для SVD (`A_ref = dW[ks[0]]` — минимальный ΔW = step 10) сингулярные векторы ошибочны на ~`1e-3/σ_min`. При `σ_16/σ_1 ~ 0.01` ошибка ~`0.1 rad`. Для Procrustes/defect в E-38h это значимо.

Рекомендация: `W_orig` хранить в fp32 (~1 MB), `snapshots` оставить fp16 (ΔW уже O(0.5)). Или обосновать в `meta`, что точность fp16 достаточна для Procrustes (тогда с явным числом).

### 1.4 **P2: промежуточный полный тензор при снапшоте**

```python
dW_snap = (lm_head.weight.data - W_orig)[token_set].to(torch.float16).cpu()
```

`lm_head.weight.data - W_orig` — полный `[V,1024]` fp32 (~622 MB на GPU) — создаётся перед срезом. Лучше:
```python
dW_snap = (lm_head.weight.data[token_set] - W_orig[token_set]).to(torch.float16).cpu()
```

### 1.5 **P2: `final_dose` fp32 vs `dose` fp16**

`snap_payload["final_dose"] = float((lm_head.weight.data - W_orig).norm())` — fp32, полный.
`pt_final["dose"] = float(delta_W.float().norm())` — из fp16 `delta_W`, полный.
Значения различаются на ~`1e-3`. Не bug, но потребитель может выбрать не то поле. Унифицировать или задокументировать.

### 1.6 **P2: K_STEPS не согласован со steps на CPU**

`_gd_steps(cpu) = 40`, `_snap_every = 10`. На CPU снапшоты есть только для steps ∈ {10,20,30,40}. E-38h `K_STEPS = [10,20,40,80,90,100,200,400,800,1000]` и `assert not missing` → **AssertionError** на CPU-прогоне E-16.

E-38h docstring утверждает «100 steps, 10..1000» — то есть потребитель ожидает `steps=1000`. Стоит в `meta` E-16 явно: «E-38h требует steps ≥ 1000». Или адаптировать E-38h `K_STEPS` к доступным снапшотам.

## 2. Эксперимент

### 2.1 **P1: tied weights — «readout-only» неверно для Qwen3.5-0.8B**

В paper §3.1: `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}` — `W = E` (тот же `nn.Parameter`).

E-16 редактирует `lm_head.weight` (Adam `[lm_head.weight]`). Если tied, это **тот же** Parameter, что `embed_tokens.weight`, значит forward меняется: `input_ids → embed (=W) → layers → norm → lm_head (=W)`. h_s зависит от W, J_norm зависит от h_s, J_S = W @ J_norm — двойная зависимость.

E-16 docstring: «Edit **only** `lm_head`». Для tied-модели это не «только», а «lm_head **и** embed». E-38h (docstring): «two-channel split applied to ΔW — computed EXACTLY from ΔW alone (h_t invariant under a readout-only edit)» — **неверно**, если tied.

Проверить на практике: `_resolve_lm_head_weight` в `interventions.py` (не в E-16), но E-16 использует `mS.lm_head.weight` напрямую. При tied это один Parameter.

Стоит:
- Либо проверить `lm_head.weight is embed_tokens.weight` и явно документировать эффект;
- Либо развязать веса перед экспериментом (`mS.lm_head.weight = nn.Parameter(mS.lm_head.weight.data.clone())`) — тогда «readout-only» строго верно. Это меняет поведение модели (в исходной tied-модели lm_head и embed связаны), но соответствует E-16 v1 («only lm_head»).

Как минимум — зафиксировать в `meta` как ограничение и учесть в интерпретации E-38h (ΔW top/tail влияет на forward через embed, не только на readout).

### 2.2 **P1: `h_s` не detached → backward через весь трансформер**

```python
out = mS(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
h_s = out.hidden_states[last_layer + 1]       # grad_fn до embeddings
J_norm = compute_hidden_jacobian(mS, last_layer, ids, attn, h=h_s)
J_S = lm_head.weight[token_set] @ J_norm
cos_sim = ...cosine_similarity(J_S_p, tgt)...
loss = (1.0 - cos_sim).mean() + kl_weight * kl_loss
loss.backward()
```

`compute_hidden_jacobian` для last_layer возвращает `_last_layer_jacobian(h_s, norm)` **вне** `no_grad` (внутри функции только `h = h.to(dtype)` под `no_grad`, сам `return _last_layer_jacobian(...)` — вне). J_norm имеет `grad_fn` до `h_s` и до `norm.weight`. `loss.backward()` распространяет градиент до параметров всего трансформера через `h_s`, хотя `optimizer = Adam([lm_head.weight])` их не обновляет.

Последствия:
- Лишние compute;
- `.grad` у параметров трансформера накапливается (только `lm_head` в `optimizer.zero_grad()`);
- Не влияет на корректность (gradient по W правильный), но `optimizer.step()` для lm_head не зависит от этих `.grad`.

Исправление — `h_s.detach()` при подаче в `compute_hidden_jacobian` в loop. В `battery()` — уже под `no_grad`, там OK.

Стоит также `J_norm.detach()` в loop (он не должен зависеть от h_s):
```python
J_norm = compute_hidden_jacobian(mS, last_layer, ids, attn, h=h_s.detach())
```

### 2.3 **P2: battery() на каждом шаге на GPU**

`_eval_every(gpu) = 1`, `battery()` запускает `2·N` forward + J_norm. Для 1000 шагов = 2000 forward + 2000 J_norm ≈ несколько десятков минут. Не bug, но «readout-only» эксперимент стоит того, чтобы battery вызывалась реже (например, `_eval_every(gpu) = 5` — она всё равно только метрика, не влияет на optimization).

### 2.4 **P1: pairing_ok — только первая ось**

```python
ok = (
    t.get("input_ids") is not None
    and t["input_ids"].shape[0] == ids.shape[0]
    and bool((t["input_ids"].view(-1) == ids.view(-1)).all())
)
```

`shape[0]` — только первая размерность. Если `targets[si]["input_ids"]` = `[S]` (без батча), а `samples[si]["input_ids"]` = `[1, S]`, то `shape[0] = S` vs `1` → False, pairing ложно провален. `view(-1)` сравнение может дать True (одинаковые элементы). Но первая проверка **убивает** пару ложно.

Проверять полную форму `t["input_ids"].shape == ids.shape` или только `view(-1)` (при уверенности, что layouts совместимы). Сейчас молча пропускает bridge-term для таких пар — метрика занижена.

### 2.5 KL, cosine, backward — корректно

- `F.kl_div(log_softmax(orig), log_softmax(out), log_target=True)` = `KL(p_ed ‖ p_orig)` — соответствует paper и E-16 docstring.
- `loss = (1 - cos_sim).mean() + kl_weight * kl_loss` — правильный loss per-sample.
- `loss.backward()` внутри цикла + `optimizer.step()` вне — правильно (accumulation по samples).
- `optimizer.zero_grad()` в начале каждого шага — правильно.

### 2.6 α* (knee) — корректно, но зависимо от `eval_every`

```python
for i in range(1, len(curve)):
    db = curve[i]["bridge"] - curve[i - 1]["bridge"]
    if db >= knee_tol:
        alpha_star = curve[i]["dose"] / W_norm
        knee_step = curve[i]["step"]
```

- `curve[0]` — baseline (step 0), `curve[i]` для i≥1 — после optimizer.step.
- На GPU: `eval_every=1`, α* определён точно.
- На CPU: `eval_every=10`, шаг детекции — 10. Менее точный knee, но это зафиксировано в `meta` через `eval_every` в `result`.

Корректно. Но `knee_step` передаётся потребителю (`exp16_lm_head_knee_info.pt`), а E-38h опирается на «knee_step=84, dose≈100». На CPU `knee_step` будет кратен 10 — а **84 не кратен 10**. То есть на CPU knee = 80 или 90, не 84. E-38h принимает knee_step как данное, значит важно запускать E-16 с `eval_every=1` (GPU).

Стоит в `meta` E-16 и в E-38h зафиксировать: «knee_step=84 — только при eval_every=1».

### 2.7 Baseline и curve — согласованы

`baseline = battery()` (при `ΔW=0`), `curve = [{"step": 0, "dose": 0.0, **baseline}]`. Затем в loop `curve.append(...)`. `curve[0]["bridge"]` = baseline bridge. `db = curve[1]["bridge"] - curve[0]["bridge"]` — прирост после первого eval-окна. OK.

### 2.8 seed=42 vs seed=0

Все E-38 используют `seed=0`, E-16 — `torch.manual_seed(42)`. Для E-16 собственно seed влияет только на Adam? Нет — Adam детерминирован при фиксированных данных и порядке. Seed влияет на `load_calibration_data` (сэмплы), инициализацию (нет — модель загружается), dropout (eval — нет). Так что seed=42 фактически влияет только на случайную выборку сэмплов. Не bug, но несогласованно с остальными E-38 (в `meta` стоит зафиксировать).

## 3. Соответствие ТЗ E-16

| Пункт | Реализация | Статус |
|---|---|---|
| Dose-response, одна unclamped Adam run | `optimizer = Adam([lm_head.weight])`, без клампа | ✅ |
| Dose = ‖ΔW‖_F, α = ‖ΔW‖/‖W‖ | `dose = ...norm()`, `alpha = dose / W_norm` | ✅ |
| Loss = (1−cos) + kl_weight·KL | совпадает | ✅ |
| KL leash: `KL(p_ed ‖ p_orig)` | `F.kl_div(log_softmax(orig), log_softmax(ed), log_target=True)` | ✅ |
| Target pair — last-layer, asserted | `target_pair` выбор + `ValueError` если нет | ✅ |
| Baseline при ΔW=0 в curve | `curve[0]` = baseline | ✅ |
| α* = последний step с Δbridge ≥ knee_tol, иначе `None` | реализовано | ✅ |
| v1 voided (caps ≥ 0.1 не bind) — исправлено на dose-response | нет caps | ✅ |
| Pairing via input_ids, no silent | `pairing_ok` per sample | ⚠️ первая ось (2.4) |
| Snapshot grid для E-38h | `snap_every=10`, `[10..1000]` | ✅ (но CPU steps=40 → недостаточно) |
| «Edit only lm_head» | при tied — затрагивает embed | ⚠️ P1 (2.1) |
| Deterministic seed | seed=42, но нет `use_deterministic_algorithms` | ⚠️ (не критично для E-16) |

## Приоритетный список

**В E-16 (исправить):**
1. **P1** — задокументировать `snapshots = ΔW` контракт в `snap_payload` (полем `schema`/`note`), а не только в комментарии, чтобы потребитель (E-38h) не читал неверно.
2. **P1** — задокументировать tied-эффект: изменение `lm_head` меняет `embed`; «readout-only» неточен. Либо развязать веса, либо явно в `meta`.
3. **P1** — `h_s.detach()` в loop перед `compute_hidden_jacobian` (или `J_norm.detach()`).
4. **P1** — pairing_ok: проверять `t["input_ids"].shape == ids.shape` (полная форма), а не первую ось.
5. **P2** — сохранять `W_orig` в fp32 (для точности SVD в E-38h); `snapshots` оставить fp16.
6. **P2** — снапшот через `(W_curr[token_set] − W_orig[token_set])`, без промежуточного `[V, D]`.
7. **P2** — зафиксировать в `meta`: CPU (`steps=40`) не даёт снапшотов для E-38h; `knee_step=84` достижим только при `eval_every=1` (GPU).
8. **P2** — унифицировать `final_dose` / `dose` (fp32 vs fp16).

**В E-38h (потребителе — критично, P0):**

9. **P0** — `dW = {k: snaps_all[k] for k in ks}` — **убрать** вычитание `W_orig` (снапшоты уже ΔW; сейчас E-38h строит `W_curr − 2·W_orig`). До этого исправления результаты E-38h (SVD, Procrustes, `A_ref`, energy_top) **невалидны**.
10. **P1** — `K_STEPS` должен пересекаться с фактическими снапшотами (сейчас `assert not missing` упадёт на CPU-прогоне E-16).

# Ревью `exp38i_factorization.py`

## P0 — тот же баг «двойного вычитания», что в E-38h

### 1. `dW_shared = snaps_all[k] − W_orig256` — W_orig вычитается дважды

`exp38_hi_common.load_snapshots`:
```python
snaps = {int(k): v.to(torch.float64) for k, v in d["snapshots"].items()}
```

`snapshots[k]` из E-16 — это уже `ΔW = (W_curr − W_orig)[token_set]` (см. `exp16_readout_pull.py`, стр. `dW_snap = (lm_head.weight.data - W_orig)[token_set]`).

E-38i:
```python
dW_shared = snaps_all[k][shared_idx256] - W_orig256[shared_idx256]
W_ed[shared_idxU] = W_ed[shared_idxU] + dW_shared
```

Подстановка:
`W_ed = W_orig + (ΔW − W_orig) = W_curr − W_orig`

Это **не** `W_curr`. Правильно:
```python
dW_shared = snaps_all[k][shared_idx256]         # snaps = ΔW, не W_curr
```

**Следствия:**
- `MK_rel_drift_max ≈ 1` (вместо `≈ 0.045`): `MK = ΔW·γ`, `MK0 = W_orig·γ`, `‖MK − MK0‖ ≈ ‖W_orig‖` (поскольку `‖ΔW‖ ≪ ‖W_orig‖`).
- `J_rel_drift_max ≈ 1` (вместо `≈ 0.045`): аналогично для J.
- `states[k]` — не реальное состояние после E-16, а разность `W_curr − W_orig`.

**Вердикт не пострадал** (см. п.2), но все отчётные числа `J_drift_max` / `MK_drift_max` — бессмысленны. Это тот же P0, что в `exp38h_two_channel.py`.

## P0 — вердикт тавтологичен (не баг математики, но неинформативность)

### 2. Residual ≡ 0 для любого W — «preservation» не проверяет редактирование

`factor_residual` сравнивает `W @ Jnorm` с closed form `(c/r)M(I−ûûᵀ)`. Но это **одна и та же формула**:

- `Jnorm = (c/r)·diag(γ)·P`, `W @ Jnorm = (c/r)·(W·diag(γ))·P = (c/r)·M·P`
- closed form `(c/r)(M − outer(M@u, u))` раскрывается в `(c/r)·M·P`

Результат — **тождественно нуль** (с точностью fp64 ~`1e-15`). Порог `RESID_TOL = 1e-6` в ~`10⁹` раз больше численного шума.

**Следствие.** `verdict = "factorization preserved"` вернётся **независимо от** `W_ed` — включая багнутый `W_ed = W_curr − W_orig` из п.1, включая `W_ed = 0`, включая любой мусор. Это не проверка леммы после редактирования, а перепроверка алгебраического тождества. Лемма `[M-1]` не «сохраняется» и не «нарушается» под редактированием — она выполняется тождественно для любой пары `(W, h)`.

**Что имело бы смысл тестировать (и почему это важно для E-38h):**
- `ΔJ(s) = c·ΔM_K(s)·P` — «линейный отклик» J на ΔW при фиксированном `h`. Если `h` действительно инвариантен (см. п.3), `ΔJ(s)` должен **точно** равняться `c·ΔW·diag(γ)·P`, и `‖ΔJ(s) − c·ΔW·diag(γ)·P‖ / ‖ΔJ(s)‖ ≈ 0` — вот это проверка «структурного факта», на который опирается E-38h. Сейчас этого нет.
- `J(s) @ h_t = c·M_K(s)·P·h_t = 0` — следствие `P·û = 0`. Если `h_t` инвариантен, это тождество верно для любого `W`. Если tie-weights (п.3), `h` меняется, и это уже не тождество.

## P1 — `h_invariant: True` — ложное утверждение для tied-модели

### 3. `W_lm = embed_tokens` → правка `lm_head` меняет `embed_tokens` → `h` меняется

Docstring (ключевой для всей логики):
> «the edit changes ONLY lm_head, so h_t, r_t, û_t, J_norm(t) are INVARIANT across steps»

И код:
```python
"h_invariant": True,  # structural: edit touches only lm_head
```

Для Qwen3.5-0.8B веса **tied** (`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`, paper §3.1). Это явно подтверждается кодом:
```python
W_full = emb[U]  # tied-embedding rows for U128 (== model W_lm, verified)
```
`emb = embed_tokens.weight`, и код его отождествляет с `W_lm`.

**Следствие.** В реальной модели `input_ids → embed(=W) → layers → norm → lm_head(=W)`. При редактировании `W_lm` меняется и `embed`, поэтому `h_t` **изменяется** на всех позициях (входные эмбеддинги другие → все скрытые состояния другие). Формально:

- `h_t(s)` при tied weights **не** инвариантен;
- `r_t(s)` **не** инвариантен;
- `û_t(s)` **не** инвариантен;
- `J_norm(t, s)` **не** инвариантен (зависит от `h_t(s)`).

E-38i использует `h` из кэша (исходная модель), поэтому в вычислении residual это не проявляется — но это **не та модель**, которая получится после E-16-редактирования. Если бы E-38i делал реальный forward с `W_ed`, `h` был бы другой.

**Что это значит:**
- Формулировка «h_t invariant» — ложна для текущей модели.
- Проверка «preservation» неинформативна дважды: (а) residual ≡ 0 (п.2), (б) `h` не «сохраняется» реально, просто код его не обновляет.
- Для E-38h это **критично**: docstring E-38h опирается на «h_t invariant under a readout-only edit» для точного `ΔJ = ΔM_K·(I−ûûᵀ)`. Под tied weights это приближение, а не тождество. E-38h докстринг: «computed EXACTLY from ΔW alone (h_t invariant under a readout-only edit)» — **неверно**.

**Варианты фикса (P1):**
- Явно в `meta` E-38i: `"weights": "tied (Qwen3.5-0.8B); readout-only edit changes embed_tokens too ⇒ h_t in the REAL model changes; the identity is checked on the cached h only"`.
- Переопределить «readout-only» через **untied** копию: `mS.lm_head.weight = nn.Parameter(mS.lm_head.weight.data.clone())` перед E-16 — тогда «h invariant» строго верно (это меняет поведение модели, но соответствует E-16 v1 «Edit only lm_head»).
- Или переписать E-38h на вычисление `ΔJ` **из реального forward** с `W_ed` (дорого, но честно).

## P1 — только 13 из 128 токенов U128 затронуты

Docstring (честно):
> «only 13 rows intersect U128… the other 115 rows are edit-invariant by construction».

Это ограничение, не баг. Но:
- `MK_rel_drift_max` — максимум по 128 токенам; у 115 из них drift тождественно 0 (строки не тронуты), у 13 — какой-то. Максимум = максимум по 13, но с bug `W_curr − W_orig` (п.1) всё равно даст `≈ 1` — то есть **баг п.1 маскирует** это ограничение: без бага drift был бы `O(0.045)` для 13 токенов, что могло бы удивить (маленький эффект, локальный), а с багом получается `O(1)`, и ограничение менее заметно.
- Стоит отчитывать `MK_rel_drift_median` и `MK_rel_drift_only_shared` — по-хорошему, не по всем 128.

## P2 — прочие замечания

### 4. `states = {0: W_full}` — общая ссылка

`W_full` не `.clone()`, так что `states[0]` — тот же тензор. Пока никто его не мутирует — ОК. Но стоит `.clone()` для симметрии (или явно в комментарии: «W_full is not modified»).

### 5. `j_raw`/`factor_residual` вызываются с `W_ed` [128, 1024], не [V, 1024]

`W_lm` — [V, D], но эффективный `W_ed` для U128 — [128, D]. `j_raw(W_ed, h, gamma)` даёт [128, D] J (для 128 выбранных токенов). Комментарий в docstring: «token-by-token on U128» — согласовано. ОК.

### 6. `torch.eye(W.shape[1])` в `factor_residual` — [D, D] на каждый вызов

`factor_residual` вызывается 11 раз (s ∈ {0, 10, …, 1000}), `I` — [1024, 1024] fp64 = 8 MB. Не критично, но `I` кэшируется между вызовами или заменяется на `P = I - uu^T` через `torch.outer(u, u)`, без явной identity:
```python
P = torch.eye(D) - torch.outer(u, u)   # или: Jnorm = (c/r)·γ[:, None]·(δ - u u^T)
```
Мелочь.

### 7. `jdr.append(rel.max().item())` — max по токенам

`J_rel_drift_max` — максимум по 128 токенам, из которых 115 имеют drift = 0, 13 — реальные. Отчитывать `jdr` по-токенно или хотя бы медиану/среднее по shared-токенам было бы информативнее. Сейчас `max` завышен (даже без бага — по 13 токенам с большим весом ΔW).

### 8. `assert not missing` — упадёт на CPU-прогоне

`K_STEPS` включает 80, 90, …, 1000. `_gd_steps(cpu) = 40`, снапшоты только для {10, 20, 30, 40}. На CPU `assert not missing` даст AssertionError — громко, но E-38i не сможет работать с CPU-прогоном E-16. Стоит либо адаптировать `K_STEPS` к фактическим снапшотам, либо явно в docstring требовать `--steps 1000` в E-16.

### 9. `float(D) ** 0.5` в `j_raw` — декоративно

`D = 1024`, `sqrt(1024) = 32.0` — точно в fp64. Не баг.

### 10. `load_model_tensors()` — HF cache path

`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/model*.safetensors` — жёстко привязано к кэшу HF. Если модель скачана в другое место (Kaggle), `glob` вернёт пустой список → `sorted([])[0]` → IndexError. Стоит fallback на `AutoModelForCausalLM` или явный параметр.

### 11. `deterministic: True` — корректно

Нет случайности: SVD детерминирован, fp64 арифметика детерминирована. OK.

### 12. `torch.outer(M @ u, u)` — [K, D], эффективно; OK

Для K=128, D=1024 — это [128, 1024], не проблема.

## Соответствие ТЗ

| Утверждение | Реализация | Статус |
|---|---|---|
| «apply E-16 ΔW snapshots» | `snaps_all[k]` из `load_snapshots` | ⚠️ двойное вычитание (P0) |
| «check factorization J = c M_K (I−ûûᵀ)» | `factor_residual` | ✅ (но проверка тавтологична, п.2) |
| «h_t, û_t, J_norm invariant» | `h_invariant: True` | ❌ ложно для tied (P1) |
| «residual at numeric floor» | `≤ 1e-6` | ⚠️ верно, но тривиально (P0) |
| «leaf drift: relative change of J / M_K per step» | `J_rel_drift_max`, `MK_rel_drift_max` | ❌ бессмысленны из-за P0 (п.1) |
| «13 rows intersect U128» | `shared_rows_256_U128: 13` | ✅ |
| «verdict: preserved» | `resid_max ≤ 1e-6` | ⚠️ тавтологично (P0) |

## Итоговый приоритетный список

**P0:**
1. `dW_shared = snaps_all[k][shared_idx256]` — убрать `− W_orig256[...]` (снапшоты уже ΔW). Без этого `W_ed = W_curr − W_orig`, все drift-метрики неверны.
2. Переосмыслить verdict: `factorization_residual ≤ 1e-6` тривиально; либо заменить на тест **линейного отклика** `‖ΔJ(s) − c·ΔW·diag(γ)·P‖ / ‖ΔJ(s)‖ ≈ 0` (это то, что реально проверяет E-38h-гипотезу), либо явно в `meta` записать «this test re-verifies the identity, not preservation under edit».

**P1:**
3. `h_invariant: True` — ложно для tied. Задокументировать в `meta`, что E-38i использует **кэшированный** `h` (исходная модель), а в реальной модели после E-16 `h` меняется (tied: `lm_head = embed_tokens`).
4. `J_rel_drift_max` / `MK_rel_drift_max` — отчитывать `median`/`mean` и отдельно по 13 shared-токенам, не только `max` по всем 128.

**P2:**
5. `W_full.clone()` для `states[0]`.
6. Кэшировать `I` в `factor_residual` (или заменить на `δ − uuᵀ` через `torch.outer`).
7. Fallback для `load_model_tensors` при отсутствии HF-кэша.
8. Зафиксировать в docstring требование `--steps 1000` в E-16 (CPU `steps=40` даёт недостаточно снапшотов).

**Связь с E-38h:**
- E-38h `dW = snaps_all[k] - W_orig` — тот же P0, тот же фикс.
- E-38h `j_response_channel_split` опирается на «h invariant», которое неверно для tied; после фикса P0 эффект будет `O(0.045)`, а не `O(1)`, но само наличие линейного отклика — приближение.

Оба скрипта нужно перезапустить после фикса P0; текущие `MK_rel_drift` / `J_rel_drift` / `energy_top` / `defect` в E-38h невалидны.

## Статус исполнения (2026-10-24)

**Сделано:**
- **E-38i P0:** `dW_shared = snaps_all[k][shared_idx256]` (без `− W_orig256`); verdict переписан — явно «тавтологично, leaf drifts carry the information, h_t invariance FALSE for tied model»; `W_full.clone()` (P2.5).
- **E-38h P0:** `dW = snaps_all[k]` (без `− W_orig`); meta: оговорка tied-весов (h_t — cached-h аппроксимация).
- **hi_common:** guard `cos(snap, W_orig) < 0.5` — ловит двойное вычитание (cos ≈ −1); norm-vs-dose guard **удалён как невалидный** для 256-среза (‖snap[10]‖ = 2.658 vs full-dose 61.79 — разные срезы; для 256-строк ‖ΔW‖ ≪ ‖W‖, но dose — по всей матрице).
- **Пересчёт (локально, CPU):** E-38i — residual = 5.48e-16 (fp64-пол), J_rel_drift max 0.355 (s=10) → 1.152 (s=1000), median_shared = 0.656, MK совпадает (1.152) — согласуется с bridge 0.60→0.91 кривой E-16. E-38h — энергия top-16: ΔW 0.735→0.280, ΔJ 0.717→0.487 по s; дефект ассоциативности top 0.323/0.313/0.312 vs tail 0.350/0.313/0.319 — оба канала Abelian (сценарий 4).
- **E-16 (для будущих прогонов):** текущий прогон валиден (steps=1000, eval_every=1, снапшоты 100/100), перезапуск НЕ требуется; P1-фиксы E-16 (W_orig-снапшот в .pt, eval_every=1 по умолчанию, top-1 по частоте, meta, docstring tied) — в коде.
- **ref-exp.md:** карточки E-38h/E-38i + индекс обновлены на v2.

**Оговорка:** P1.3/E-38h «эффект будет O(0.045)» — в реальности J_rel_drift = 0.36→1.15 (линейный отклик на cached-h — приближение, не тождество; tied-веса).
