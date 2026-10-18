## Краткий вердикт

Скрипт **структурно соответствует ТЗ E32t.md**: реализованы `C_b(r)` через Gram-трюк, контроль `off²(r)` по алгебре E-30b, кубический fit без константы, агрегация median/IQR, пилот L18/L22, `--layers`, meta с пометкой, что hidden-space sweep не сделан.

Но **запускать как есть нельзя**: есть **две критические ошибки в коде**, **одна критическая методологическая несостыковка** с ТЗ и **ошибка в selfcheck**. Математика ядра `C_b` и `off2_cum` в основном корректна; основной риск — что `b` считается не для того оператора, с чьими ESG-модами он сравнивается.

---

## 1. Соответствие ТЗ E32t.md

| Пункт ТЗ | Статус | Комментарий |
|---|---|---|
| `C_b(r)=‖P_r b_off P_r‖_F²/‖b_off‖_F²`, Gram-трюк | ✅ | `cb_curve` корректен: `‖P_r b P_r‖_F² = ‖U_r^T b U_r‖_F²`, double cumsum по `M=(U^T b_off U)²` |
| `b = a2` из кубического fit без const | ⚠️ | Формально `a2` извлекается правильно, но см. п. 3: forward — только decoder-tail, не полный путь |
| `b_off = b − diag(b)` | ✅ | Корректно |
| Контроль `off²(r)` по E-30b из того же JR | ✅ | `off2_cum` алгебраически верен: `W=P⊙(U−pe)`, `G=(WᵀW)²⊙λλᵀ`, `off²=‖A_r‖²−‖diag A_r‖²` |
| Per-doc median по валидным позициям → median/IQR по докам | ⚠️ | Для кривых — да. Но per-doc Spearman считается по **mean** кривым, а не median, как в ТЗ |
| Spearman(C_b, C_off²) per doc | ❌ | Функция `spearman` реализована неверно — все Spearman-числа невалидны |
| Overlap top-5, tail_share_b | ✅ | Логика есть, но зависит от сломанного Spearman/verdict |
| T1/T2/T3 verdict | ❌ | `verdict` падает при наличии `exp30_full_spectrum.json`: `ref_ts` — это кривая, а не скаляр |
| Hidden-space sweep `b^(r)` | ⚠️ | Не сделан; в meta помечено. ТЗ называет его secondary — допустимо, но это не полное соответствие |
| «b уже посчитан, без форвардов, минуты на слой» | ❌ | Скрипт **пересчитывает** `a2` 16-точечным ε-sweep с форвардами; это 30–60 мин, а не минуты. В docstring это честно оговорено, но ТЗ §3.4 расходится с артефактом `exp32r2_hessian.json` |
| Пилот L18/L22, `--layers` | ✅ | Есть |
| Self-contained, без `prism.*` | ✅ | Есть |
| CPU-safe | ⚠️ | Нет `torch.set_num_threads(12)`, но это мелочь |

---

## 2. Что математически корректно

1. **`cb_curve`** — корректен.  
   `P_r = U_r U_r^T`, поэтому  
   `‖P_r b_off P_r‖_F² = ‖U_r^T b_off U_r‖_F²`,  
   а `M_b[k,k']=(u_k^T b_off u_{k'})²` и double cumsum дают нужную кумуляту. Sign-invariance обеспечена квадратом.

2. **`off2_cum`** — выглядит корректно.  
   `P(p)=diag(p)−pp^T`, `W=P(p)U`, `A_r=W_r Λ_r W_r^T`,  
   `‖A_r‖²=Σ_{k,l≤r} λ_k λ_l (w_k·w_l)²`,  
   `diag(A_r)_i=Σ_{k≤r} λ_k W_{i,k}²`.  
   Код `W=P[:,None]*(U−pe[None,:])` — это ровно `P(p)U`, не ошибка.

3. **`fit_all` в основном пайплайне** — корректен.  
   При `e_abs=ε_rel·‖h‖` он возвращает `a1=c1/n`, `a2=c2/n²`, что соответствует `dz=a1 e+a2 e²+a3 e³`. То есть `b=a2` извлекается правильно.

4. **RMSNorm-tail** в `load_decoder` — корректен: `x·rsqrt(mean(x²)+eps)·(1+w)`, затем `lm_head`.

5. **Сентинел** по `P128_RAW.sum(-1)<0.5` — соответствует конвенции E-34.

---

## 3. Критические ошибки

### 3.1. `spearman` реализован неверно — все Spearman-метрики невалидны

```python
rx = np.argsort(np.argsort(x)) - (np.arange(n) + 1) / (n + 1)
```

Вычитается вектор, зависящий от **индекса массива**, а не от ранга. Спирмен должен быть корреляцией рангов; вычитание `(i+1)/(n+1)` меняет корреляцию и зависит от порядка элементов. Для одинаковых `x` selfcheck проходит, но для разных кривых значения не равны Spearman.

**Следствия:**
- `spearman_agg`, `spearman_docs_med`, `spearman_pos_med`;
- `off2_vs_e30b_spearman` — то есть проверка воспроизведения E-30b;
- вердикты T1/T2;
- все числа `Spearman` в JSON невалидны.

**Исправление:** `scipy.stats.spearmanr` или `rankdata` + `np.corrcoef`.

---

### 3.2. `verdict` падает, если `exp30_full_spectrum.json` есть

В `main`:

```python
e30b[l] = e30b_raw['layers'][str(l)]['off2_cum_med']
...
agg['verdict'] = verdict(agg, e30b.get(l))
```

То есть в `verdict` передаётся **кривая** `off2_cum_med`, а внутри:

```python
ref_ts = e30b_ref if e30b_ref is not None else layer_agg['tail_share_off2']
...
abs(ts - ref_ts) < 0.25
```

Если `e30b_ref` — список/массив, то `ts - ref_ts` даёт массив, и `and`/`abs(...) < 0.25` приводят к `TypeError`/`ValueError`. Для L18/L22 при наличии E30B скрипт, скорее всего, **упадёт на verdict**.

**Исправление:**  
`ref_ts = 1.0 - e30b_ref[117]`, либо передавать уже скаляр `tail_share_off2`, либо явно считать `tail_share` из кривой.

---

### 3.3. Selfcheck `fit_all` неверен — `--selfcheck` не должен проходить

В `selfcheck`:

```python
en = er[None,None,None,:] * nrm[:,None,None,None]
dz_t = (a1t * nrm[:,None,None])[...,None] * en
     + (a2t * nrm[:,None,None]**2)[...,None] * en**2
     + ...
```

Но `en = er·nrm`, поэтому первый член равен `a1t·er·nrm²`, второй `a2t·er²·nrm⁴` и т.д.  
`fit_all` вернёт `a2 = a2t·nrm²`, а не `a2t`. При `nrm≠1` assertion `np.allclose(a2, a2t)` упадёт.

Правильно:

```python
dz_t = a1t[None,:,:,None] * en
     + a2t[None,:,:,None] * en**2
     + a3t[None,:,:,None] * en**3
```

**Важно:** основной `fit_all` в пайплайне не сломан; сломан именно selfcheck. Но это значит, что заявленное «selfcheck: 7/7 OK» не соответствует коду.

---

### 3.4. Главная методологическая несостыковка: `b` — это Hessian decoder-tail, а `U`/`ESG` — из full-path `JR`

Это самое серьёзное для научного вывода.

`load_decoder.forward(hs)` делает только:

```python
x = RMSNorm(hs)
logits = x @ W_U^T
```

То есть для слоя `l < 23` это **не полный forward** `h_l → слои l+1..23 → norm → lm_head`, а только **decoder-tail**.  
При этом `ESG = JR @ JR.T`, где `JR` из jcache — по E-32r.3.1 это **full-path Jacobian** `h_l → ... → lm_head`.

То есть в `doc_layer`:

- `b = a2` — Гессиан **tail-readout** вдоль направлений `JR`;
- `U` — собственные векторы `ESG` **full-path** `JR`;
- `off2_cum` — контроль по `ESG` **full-path** `JR`.

Для L23 `JR ≡ J_tail` (E-32r.3.1: cos = 1.0000), поэтому L23 чист.  
Для L18/L22 это смешение двух операторов. E-32r.3.1 даёт:

`cos(JR, J_tail)`: L6 0.105, L12 0.237, L18 0.598, L19 0.642, L22 0.789, L23 1.0000.

Значит:
- **L18** — сравнение `b_tail` с `U_full` особенно сомнительно;
- **L22** — умеренно сомнительно;
- **L23** — корректно.

ТЗ E32t §3.2 определяет `b_ji=½(J^T e_i)^T H_j (J^T e_i)/(ESG)_ii` как будто `J` и `H_j` — один и тот же оператор. В скрипте `J` — full-path `JR`, а `H_j` — tail. Это **не точный TЗ-объект**, кроме L23.

**Что делать:**
- либо считать `b` на полном forward `h_l → ... → lm_head`;
- либо считать `U`/`off²` из `J_tail` и явно называть объект `b_tail`;
- либо ограничить пилот L23;
- либо в отчёте явно писать, что для L18/L22 `C_b` — это «tail-Hessian в full-path ESG-базисе», а не TЗ-`b`.

---

## 4. Средние/мелкие замечания

1. **Per-doc Spearman считается по mean, а не median.**  
   В ТЗ: per-doc median по валидным позициям → median/IQR по докам.  
   В коде:
   ```python
   sp = spearman(cb_t[v].mean(0), off2_t[v].mean(0))
   ```
   Для `spearman_doc_mean` это mean. Для финального `spearman_agg` используется median-кривая, но `spearman_docs_med` — mean-based. Лучше считать per-doc Spearman по `np.median(cb_t[v],axis=0)` и `np.median(off2_t[v],axis=0)`.

2. **`neg_frac_per_r_b` всегда 0.**  
   `C_b` — кумулята суммы квадратов, поэтому `ΔC_b ≥ 0` по построению. Поле как «аналог E-30b neg_frac» неинформативно. Не ошибка, но не стоит интерпретировать.

3. **`fit_all` использует `np.linalg.inv(A0.T@A0)`.**  
   Для 16 точек и 3 базисных функций это работает, но `lstsq` был бы устойчивее. Не критично.

4. **`load_decoder` использует `dtype=torch.float32`.**  
   В зависимости от версии transformers аргумент может называться `torch_dtype`; в v5 встречается `dtype`. Мелочь, но версионно-зависимая.

5. **Нет `torch.set_num_threads(12)`.**  
   ТЗ говорит CPU-safe 12 threads; скрипт этого не делает.

6. **`bool(e30b.get(l))` в meta.**  
   Если `off2_cum_med` — список, `bool(list)` работает; если numpy array — упадёт. В текущем JSON, скорее всего, список, но лучше явно `len(...)>0`.

---

## 5. Итоговая оценка

**Математика ядра — в основном корректна:**
- `C_b` через Gram-трюк — да;
- `off2_cum` по E-30b — да;
- `fit_all` в основном пайплайне — да;
- `b=a2` извлекается правильно.

**Код — содержит критические баги:**
1. `spearman` неверен → все Spearman-метрики и T1/T2 невалидны;
2. `verdict` падает на `ref_ts`-кривой;
3. `selfcheck` для `fit_all` неверен;
4. `b` считается на decoder-tail, а `U`/`off²` — на full-path `JR`; для L18/L22 это смешение операторов.

**Соответствие ТЗ — частичное:**
- token-space `C_b` и контроль `off²` реализованы;
- hidden-space sweep не сделан (помечено, secondary);
- «без форвардов, минуты» не выполнено — скрипт пересчитывает `a2` с форвардами;
- per-doc median для Spearman частично нарушен.

**Рекомендация:** перед запуском исправить `spearman`, `verdict`, `selfcheck`; решить, что такое `b` — tail-Hessian или full-path Hessian. Если нужен именно TЗ-`b` для L18/L22, надо либо гонять полный forward, либо пересчитывать `U`/`off²` из `J_tail` и явно переименовать объект. Иначе результаты L18 будут смешаны, L22 — умеренно, L23 — чисто.