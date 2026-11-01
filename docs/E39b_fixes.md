Ниже — аудит `exp39b_ptq_perturbation.py` против `E39.md §2` и связанных карточек. Я не запускал код, только читал. Главный вывод: **код в целом следует структуре ТЗ, но есть несколько «тихих» багов, которые могут сильно исказить именно результат E-39b — особенно PPL, GPTQ-гессиан и вывод gate**. Часть из них не упадёт с ошибкой, а просто даст неверные числа.

## 1. Соответствие ТЗ — кратко

| Пункт ТЗ | В коде | Статус |
|---|---|---|
| Qwen3.5-0.8B, fp16 | `MODEL`, `torch_dtype=float16` | ✅ |
| Слои `{6,12,18,22,23}` | `L_CACHED` + лишний `19`; gate — `L_GATE` | ⚠️ L19 лишний, но не ломает gate |
| GPTQ 4-bit, `group_size=128`, `desc_act=True`, `damp=0.01` | `_quantize_gptq` | ✅ по смыслу |
| fp16 / 8-bit / 4-bit | fp16, int8 RTN, gptq4 | ⚠️ int8 — RTN, не GPTQ-8bit |
| Калибровка 128 C4 | `N_CAL=128`, `S_CAL=128` | ❌ фактически короткие тексты + padding |
| PPL на C4 holdout | есть | ❌ считается NLL, не PPL; плюс double RMSNorm |
| `J_t = c_t M_K (I − ûûᵀ)` | `_recompute_J` | ✅ точно на L23; ❌ для L<23 это не настоящий Jacobian |
| Метрики κ, β/σ₁, cos(u_g,1/√K), drift | есть | ⚠️ κ/drift на L<23 — синтетика; cos — proxy, не точный `u_g` |
| Gate `<0.02 / <0.10` | есть | ⚠️ абсолютный порог, не проверка `≤ +25% base` |
| Детерминизм | `SEED=0`, но не используется | ❌ |
| Tied weights | не учтены в PPL | ❌ PPL — readout-only, не полный GPTQ tied-модели |

## 2. Критические «тихие» баги, портящие результат

### P0.1. `_c4_docs` режет текст по символам, а не по токенам
В `_c4_docs`:
```python
texts.append(t[:s_len])
...
tokenizer(..., max_length=s_len, padding="max_length")
```
`t[:s_len]` — это **128 символов**, а не 128 токенов. Для английского C4 это ~20–30 токенов, остальное — padding. Последствия:
- калибровочный набор GPTQ фактически в 4–6 раз меньше, чем заявлено;
- PPL считается на очень коротких последовательностях;
- растёт шум и падает мощность.

**Исправление:** не резать текст до токенизации. Передавать полный `t`, а `max_length=s_len` делать в токенизаторе.

### P0.2. GPTQ-гессиан включает padding-позиции
Хук снимает `lm_head` input для всей матрицы `[B,S,D]`, затем:
```python
X = torch.cat(captured, dim=1).reshape(-1, D)
```
`attention_mask` (`attn_cal`) **не применяется**. В `X` попадают padding-позиции. Это тихо портит `H = XᵀX` и, следовательно, GPTQ-квантование. При коротких текстах из P0.1 padding может быть большинством.

**Исправление:** фильтровать по маске:
```python
X = captured[0][attn_cal.bool()]  # [n_real_tokens, D]
```
и логировать `n_real_tokens`.

### P0.3. PPL считается как mean NLL, а не как perplexity
`_ppl_from_logits` возвращает:
```python
lp_target[real].mean().abs().item()
```
Это средний negative log-likelihood, не `exp(NLL)`. В JSON ключ называется `ppl`, в gate используется `ppl_delta_pct`. Если base NLL ≈ 2.5, а quant NLL ≈ 2.6, то NLL-дельта ≈ 4%, а PPL-дельта ≈ 10.5%. То есть **ΔPPL систематически занижается**, и вывод «PPL вырос всего на 1–2%» может быть ложным.

**Исправление:** возвращать `exp(mean_nll)` или явно переименовать в `nll` и в gate считать `exp`.

### P0.4. Двойной final RMSNorm в PPL
В PPL:
```python
out = model.model(input_ids=..., attention_mask=...)
h_last = out.last_hidden_state  # комментарий: "pre-norm"
h_normed = _lm_module(model).norm(h_last.float())
```
В стандартных HF-моделях Qwen2/Qwen3 `model.model(...).last_hidden_state` — это **уже post-final-RMSNorm**. Если так и здесь, то `norm` применяется второй раз. Это тихо искажает `h_normed`, а значит все PPL и `ppl_delta_pct`.

Проверка:
```python
with torch.no_grad():
    logits_model = model(input_ids=..., attention_mask=...).logits
    logits_manual = model.lm_head(out.last_hidden_state)
assert torch.allclose(logits_model, logits_manual, atol=...)
```
Если `logits_manual` совпадает с `logits_model`, то `out.last_hidden_state` уже нормирован и повторный `norm` — баг.

**Исправление:** либо использовать `h_last` как есть, либо снимать `lm_head` input хуком и для PPL тоже, чтобы гарантированно совпадать с GPTQ-калибровкой.

### P0.5. Tied weights: PPL — не полный GPTQ, а readout-only
Qwen3.5-0.8B tied: `lm_head.weight == embed_tokens.weight`. В коде квантуется только `W_lm`, а `h_normed` берётся от **базовой** модели. То есть эмулируется «квантованный только readout», но не реальный GPTQ tied-модели, где квантование `lm_head` меняет и `embed_tokens`, а значит и все `h`. E39.md §14.1 это признаёт как limitation, но скрипт нигде не фиксирует, что PPL — readout-only. Это может сильно занизить деградацию PPL.

**Исправление:** либо явно писать `readout_only_ptq`, либо квантовать embed_tokens и пересчитывать forward.

## 3. Методологические тихие искажения

### P1.1. Для L<23 `_recompute_J` — не настоящий Jacobian
Формула `J = c M_K (I − ûûᵀ)` точна только на L23. Для L6/L12/L18/L19/L22 скрипт строит «синтетический readout-only J» из `H_L`, игнорируя оставшиеся слои. Поэтому:
- `κ(L)` для L<23 — не κ реального листа, а κ синтетической матрицы;
- `β/σ₁` для L<23 использует `û_t` не того слоя, где определён `M_K`;
- `cos(u_g,1/√K)` и `J_rel_drift` для L<23 — тоже синтетика.

Gate стоит на L23, и там всё точно. Но JSON с `metrics[L]` для L<23 может быть прочитан как «κ на слое L», что неверно.

**Исправление:** либо считать L<23 только как `readout_linearized` с явным префиксом, либо ограничить β/σ₁ и cos только L23, а для L<23 использовать кэшированный `J` и реальный `J_q` через полный путь.

### P1.2. `_cos_ug_1k` — proxy, а не точный `u_g`
Код:
```python
v = (J Jᵀ) @ ones
u_g = v / ||v||
cos = <u_g, ones>
```
Это cheap-proxy `JJᵀ·1/||·||`, а не левый сингулярный вектор `u_g`, максимизирующий `|cos(u_g,1)|`. В E-33r такой proxy использовался как приближение, но E-39b ТЗ просит `cos(u_g,1/√K)`. Для отчёта это может дать смещение.

**Исправление:** считать `u_g` через SVD `J = UΣVᵀ`, выбирать левый вектор с максимальным `|cos(u,1)|`, затем `cos`.

### P1.3. Нет sentinel-фильтра
E39.md §14.2 говорит, что flatness-метрики считаются на non-sentinel. Скрипт использует все позиции. Для L23 κ baseline 0.0162 в E-39a считался на всех triples, так что это не обязательно ломает gate, но это отклонение от общего ограничения серии.

### P1.4. Нет seed и deterministic
`SEED=0` объявлен, но `torch.manual_seed(SEED)` нигде нет. `torch.use_deterministic_algorithms(True)` тоже нет. GPTQ/SVD на GPU могут быть недетерминированными. E39.md §2.5 прямо требует deterministic.

### P1.5. Утечка памяти после self-check
После self-check остаются жить `J_rec`, `J_cache`, `H`, `M_K_base` — это ~4+ ГБ на GPU. В per-layer цикле добавляются `J_base` и `Jq`. На T4 может не упасть, но риск OOM и лишнее давление. Нужно `del J_rec, J_cache, H, M_K_base` и `torch.cuda.empty_cache()`.

### P1.6. Gate — абсолютный, не относительный
ТЗ: `< 0.02` трактуется как `≤ +25% от base`. Код проверяет только `kap23_gptq < 0.02`. При base 0.0162 это почти то же, но формально не проверяет ratio. Лучше добавить `kap_ratio < 1.25` как дополнительное условие.

### P1.7. `int8` — RTN, не GPTQ-8bit
ТЗ говорит про «fp16 GPTQ vs 8-bit GPTQ vs 4-bit». Код делает `int8 RTN`. Это не то же самое. Если 8-bit нужен как контроль GPTQ, надо либо явно назвать `int8_rtn`, либо реализовать GPTQ-8bit.

### P1.8. `L19` лишний
`L_CACHED` включает L19, ТЗ — `{6,12,18,22,23}`. Gate не страдает, но JSON получает лишний слой. Лучше либо убрать, либо пометить `extra`.

## 4. Что я бы проверил ассертами перед запуском

1. **PPL double norm:**
   ```python
   assert torch.allclose(model.lm_head(out.last_hidden_state), model(...).logits)
   ```
   Если не совпадает — `out.last_hidden_state` не то, что идёт в `lm_head`.

2. **Маска калибровки:**
   ```python
   print("cal real tokens:", attn_cal.sum().item(), "of", attn_cal.numel())
   assert attn_cal.sum().item() > 0.5 * attn_cal.numel()
   ```
   С текущим `t[:128]` это почти наверняка провалится.

3. **PPL — это PPL:**
   ```python
   assert ppl["base"] > 1.0  # если это NLL, будет ~2–4; если PPL, ~10–30
   ```
   Лучше явно `ppl = math.exp(nll)`.

4. **L23 self-check уже есть** — хорошо. Дополнительно:
   ```python
   assert abs(metrics[23]["fp16"]["kappa"] - 0.0162) < 0.003
   assert abs(metrics[23]["fp16"]["beta_sigma1"] - 0.882) < 0.05
   ```
   Если β/σ₁ сильно уехал — возможно, `g` или `M_K` не те.

5. **`J_rel_drift` для fp16:**
   ```python
   assert metrics[23]["fp16"]["j_rel_drift_max"] < 1e-5
   ```

6. **Реальное число токенов в `X`:**
   ```python
   print("X rows:", X.shape[0], "expected real ~", attn_cal.sum().item())
   ```

## 5. Итоговая приоритизация

**До запуска обязательно исправить:**
1. `_c4_docs`: не резать по символам; токенизировать полный текст до `max_length`.
2. GPTQ Hessian: фильтровать `X` по `attention_mask`.
3. PPL: `exp(mean NLL)`, а не mean NLL.
4. PPL: убрать двойной `final RMSNorm` (проверить, что `out.last_hidden_state` уже post-norm).
5. Явно зафиксировать tied-weights limitation: PPL — readout-only, не полный GPTQ.

**Желательно исправить:**
6. Для L<23 не отчитывать `κ`, `β/σ₁`, `cos` как «настоящие»; либо считать только L23, либо явно `readout_linearized`.
7. `_cos_ug_1k` — точный `u_g` через SVD.
8. Seed + deterministic.
9. Удалять `J_cache`/`J_rec` после self-check.
10. Sentinel-фильтр или явная оговорка.
11. Gate: добавить `kap_ratio ≤ 1.25`.
12. Переименовать `int8` → `int8_rtn` или сделать GPTQ-8bit.

**Не критично, но стоит поправить:**
13. Убрать L19 или помечать extra.
14. `_ppl_from_logits` — не брать `abs()` от среднего, а `-mean`.
15. Логировать `n_real_tokens` для калибровки и PPL.

Если исправить P0.1–P0.5, результаты E-39b могут заметно измениться — особенно `ΔPPL` и вывод gate. Сейчас скрипт выглядит как аккуратная заготовка, но именно PPL и GPTQ-калибровка — те места, где «тихие» ошибки наиболее опасны.