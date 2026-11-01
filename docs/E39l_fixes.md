# Ревью E-39l v2 (rounds 1+2 fixes)

## 1. Главный результат

**Verdict: `inconclusive (drift non-monotone)`** — честный и правильный.

Проверил данные:
```
drift:  0.355 → 0.569 → 0.685 → 0.700 → 0.699 → 0.698 → 0.708 → 0.684 → 1.197 → 1.152
s_frac: 0.01   0.02    0.04    0.08    0.09    0.10    0.20    0.40    0.80    1.00
```

Drift **реально немонотонен**: локальные провалы на 0.09, 0.10, 0.40, 1.00. `mono_inc=False`, `mono_dec=False` → `monotonicity_ok=False` → verdict `inconclusive`. **Это правильный honest result**, а не bug.

---

## 2. Что сделано хорошо (rounds 1+2)

| Fix | Verified |
|---|---|
| Anti-control на residual | `_anti_control_residual(resid_drift, resid_ret)` ✓ |
| Permutation significance | `significant = |rho_res| > p95_anti` (0.273 < 0.636 → False) ✓ |
| NaN → None (`_safe`) | `allow_nan=False` при dump ✓ |
| Detrend pair consistency | `_detrend_pair` — один метод для обеих серий ✓ |
| Sign-flip sensitivity | `detrend_sensitivity` с quad vs linear ✓ |
| Confidence по frac_measured | `frac=0.2 → "low"` ✓ |
| Один json.dump | ✓ |
| Retention_source | measured=[0.1, 1.0], interp=8 ✓ |
| Null distribution | p10/p50/p95/p99 ✓ |
| savefig try/except | ✓ |
| parametric vs permutation warn | ✓ |

**Sign-flip детектирован:** quadratic rho=−0.273, linear rho=+0.127 → `sign_consistent=False`. **Это важный finding** — residual correlation зависит от detrend метода, значит артефакт.

---

## 3. Критический тихий баг — **verdict противоречит caveat** (HIGH)

### 3.1. Факт

**Verdict:**
> `"inconclusive (drift non-monotone — rho_res may be informative, see diagnostics)"`

**Caveat:**
> `"rho_res SIGN flips across detrend choices ... — result is a detrend artifact, not a robust signal; **do not interpret**"`

**Противоречие:** verdict говорит «rho_res may be informative», caveat — «do not interpret». Reader (или paper-generation pipeline), читающий только `verdict`, получит неправильный сигнал.

### 3.2. Причина

`sign_flip` вычисляется **после** `_verdict`:

```python
verdict = _verdict(rho_res, p_res, p95_anti, significant, monotonicity_ok)  # line A
...
sens = {...}
sign_flip = any(...)   # line B (позже)
if sign_flip:
    caveats.append("... do not interpret ...")   # line C (ещё позже)
```

`_verdict` не знает про sign_flip — не может его учесть.

### 3.3. Почему это тихо портит результат

В paper §5 synthesis читатель увидит `verdict` в первую очередь. Он прочитает «drift non-monotone — rho_res may be informative», решит «есть потенциальный сигнал, нужны доп. данные», **пропустив** caveat «do not interpret».

В **текущем** JSON это менее критично, потому что `monotone_ok=False` сработал первым и verdict всё равно `inconclusive`. Но логическая ошибка есть: если данные станут монотонными (drift monotone), verdict всё равно не учтёт sign_flip.

### 3.4. Fix

```python
# Вычислить sign_flip ДО _verdict
sens = _compute_detrend_sensitivity(sf, drift, ret, caveats)
sign_flip = sens and not sens.get("sign_consistent", True)

# Передать в verdict
verdict = _verdict(rho_res, p_res, p95_anti, significant, monotonicity_ok, sign_flip)

def _verdict(..., sign_flip=False):
    if np.isnan(rho_res):
        return "inconclusive (degenerate residuals)"
    if sign_flip:
        return ("inconclusive (rho_res SIGN FLIPS across detrend choices "
                "— detrend artifact, not a robust signal; do not interpret)")
    if not monotone_ok:
        return "inconclusive (drift non-monotone — see diagnostics)"
    ...
```

**Priority: HIGH.** Это единственный дефект, который реально может испортить paper.

---

## 4. Средний баг — **`interpolated_points` использует exact `in` для float** (MEDIUM)

```python
measured_points = [r["s_frac"] for r in rows
                   if any(abs(r["s_frac"] - s0) < 1e-9 for s0 in e16b_nodes)]
interpolated_points = [r["s_frac"] for r in rows
                       if r["s_frac"] not in measured_points]
```

**Первая строка** — толерантная (`abs < 1e-9`). **Вторая** — exact `in` для list of floats.

В текущих данных все `s_frac` representable (`100/1000=0.1` точно, `1000/1000=1.0` точно), поэтому OK. Но если добавить `s=300 → 0.29999...`, может **не совпасть** с `measured_set` из-за float error → s_frac попадёт и в measured (по толерантной проверке), и в interpolated (по exact `in` — `0.29999 not in [0.3]`).

Результат: `n_measured + n_interpolated > n_joined`, `fraction_measured > 1`.

**Fix:**
```python
interpolated_points = [r["s_frac"] for r in rows
                       if not any(abs(r["s_frac"] - s0) < 1e-9 for s0 in e16b_nodes)]
```

Или:
```python
measured_set = {round(p, 6) for p in measured_points}
interpolated_points = [r["s_frac"] for r in rows
                       if round(r["s_frac"], 6) not in measured_set]
```

---

## 5. Мелочи

### 5.1. `_detrend` — dead code (LOW)

```python
def _detrend(s, y, caveats):
    """... ТОЛЬКО для одной серии — для пары используйте _detrend_pair ..."""
```

Определён, но не используется нигде в `main()`. `_detrend_pair` и `_detrend_poly` — используются. `_detrend` — dead. Удалить или явно помечены как unused (или использовать для одного из sensitivity branches).

### 5.2. `caveats` sign_flip добавляется после verdict (LOW, следствие §3)

После fix §3.4 эта проблема исчезает.

### 5.3. `sens` в JSON — только при `len(rhos) > 1` (OK)

```python
if len(rhos) <= 1:
    sens = None
```
Если один из методов даёт NaN, sens=None → нет sign_consistent поля → нет caveat. Правильно.

### 5.4. `_verdict` tier "moderate" — underpowered (информативно)

Проверил: `moderate` требует `significant AND −0.5 ≤ rho_res ≤ −0.3`, но `significant` требует `|rho_res| > p95 ≈ 0.64`. Значит `|rho_res| > 0.64 AND |rho_res| ≤ 0.5` → пусто. Caveat это фиксирует:

> "moderate tier unreachable at this n"

✓ Правильно.

### 5.5. `_safe(rho_res)` для печати

```python
print(f"E-39l v2: rho_res={_safe(rho_res)}, ...")
```

`_safe` применяется для print — OK, NaN → None → "None". Но `_safe(rho_res)` возвращает `None` (не string "None"). f-string выдаст "None". OK.

### 5.6. `scatter_plot` fallback на exception

```python
except Exception as e:
    res["scatter_plot"] = f"savefig failed: {e}"
```

Catches all exceptions — good for robustness, но можно тихо съесть real bug (например, `plot` с NaN). Стоит `except (IOError, OSError) as e:` — целевые IO-ошибки. Но `catch-all` тоже OK для non-critical plot.

### 5.7. JSON output — `detrend_sensitivity.quadratic.rho_res = -0.2727` vs `metrics.rho_residual.rho = -0.2727` — consistency ✓

Значит primary rho_res использует quadratic detrend (из `_detrend_pair` fallback). Consistency ✓.

### 5.8. **`rho_residual.p = 0.446` parametric** vs `significant=False` permutation — **согласованы** (0.446 > 0.10, not significant). Caveat про divergence добавлен, но в текущем случае divergence **не случился** — warn про `p_res < 0.05 AND not significant` не сработал. ✓

---

## 6. Что paper может теперь сказать

**E-39l v2 честный результат:**

1. **`monotonicity_ok=False`** — drift немонотонен в s_frac. Это самостоятельный finding: J_rel_drift_max не монотонно растёт с dose.
2. **`rho_res` не определён** — sign flip (quad: −0.27, linear: +0.13) + не значим ни в одном варианте (p95 null = 0.64).
3. **`shared_trend_only` не достигнут** — |rho_res| = 0.27 > 0.2, но p95 не превышен. Falls в "inconclusive (underpowered)".
4. **Verdict**: `inconclusive (drift non-monotone)` — primary; sign-flip — secondary (fix §3.4 сделает его primary).

**Для paper §5 synthesis:**
> «E-39l: drift-vs-behavior корреляция **не установлена** (verdict inconclusive). Drift немонотонен в dose (0.35→0.70→1.20→1.15 с локальными провалами); residual correlation после детрендинга знак-флипается между quadratic и linear fit (−0.27 vs +0.13), т.е. rho_res — артефакт детренд-выбора. Гипотеза «drift predictive beyond trend» **не поддержана** и **не опровергнута** — недостаточная мощность + немонотонность. Кausal test — E-39d/e.»

Это **negative/inconclusive result**, но честный. Не блокирует paper, но paper не может заявить «Goodhart is geometric» на основе E-39l.

---

## 7. Итог

**Блокер:**

1. **§3** — verdict contradicts caveat about sign_flip. Fix: ~10 строк (compute sign_flip before verdict, pass to `_verdict`).

**Средне:**

2. **§4** — `interpolated_points` exact `in` для floats. Fix: 1 строка (tolerant check).
3. **§5.1** — `_detrend` dead code.

**Что можно оставить:**

4. §5.6 catch-all exception — OK для non-critical plot.
5. §5.3 sens=None fallback — правильно.

**Оценка v2 (rounds 1+2): 8.5/10.** После фикса §3 — **9/10**, готов к paper. §4 — preventative fix (текущие данные не триггерят), но стоит сделать для robustness.

**Хорошее научное поведение:** скрипт **не соврал**. Результат `inconclusive` — правильный. Это ровно тот результат, который paper должен иметь — **honest negative**.