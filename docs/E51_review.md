## Статус: запускать можно

Все P0/P1 предыдущего раунда закрыты:

| Пункт | Статус |
|---|---|
| `T=256` в controls (white `I2_mean` variance) | ✅ P(ложного провала) ≈ 1.5% вместо 14% |
| `I1_1_ci_DIAGNOSTIC` / `I1_8_ci_DIAGNOSTIC` | ✅ Правильно помечено |
| `step_real` / `step` в `sw{j}`/`sh{j}`/`sur{j}` — matching audit | ✅ |
| `notrobust_{f}` — overall sig ≠ robust sig | ✅ |
| ndom<4 caveat в docstring | ✅ |
| `increments` — удалён | ✅ |

---

## Что осталось (P2, не блокирует)

### 1. Median vs mean в point estimate и CI

```python
s[f"vs_{n}_" + f] = [safe_med(d), boot_mean_ci(d)]
```

`safe_med(d)` — медиана; `boot_mean_ci(d)` — CI **среднего**. Для симметричных распределений разницы нет, для скошенных — point и CI могут расходиться (медиана уходит от среднего). Если `lo*hi > 0` на mean-CI, но медиана близка к 0 — решение о значимости нестабильно.

**Fix** (1 строка): `s[f"vs_{n}_" + f] = [safe_med(d), boot_ci(d)]` — `boot_ci` в E50 уже даёт CI медианы, согласован с `safe_med`. Или наоборот, использовать `np.mean(d)` как point — тогда mean+mean CI консистентны.

Это не баг — но для формальной публикации цифры «медиана ≠ 0 с CI mean» лучше не оставлять.

### 2. `notrobust_{f}` не отличает «overall sig» от «domain-dependent»

```python
s[f"notrobust_{f}"] = bool(any(sig.values()) and not any(rob.values()))
```

Если `sig[sur]=True`, `rob[sur]=False` (в 2/4 доменах), то `notrobust=True`. Но если `sig[sur]=False`, `sig[sw]=False`, `sig[sh]=True` (только sh значим, но в 2 доменах) — тоже `notrobust=True`, хотя это **другой** паттерн. Читатель не отличит.

**Fix**: добавить `which_notrobust` — какая именно null-модель дала overall sig без robustness. Или сохранить `sig` и `domc` в `s` явно — сейчас они в summary (`doms_{n}_{f}`), но `sig[n]` (overall) нигде не сохранён.

```python
s[f"sig_{f}"] = {n: sig[n] for n in NULLS}
s[f"rob_{f}"] = {n: rob[n] for n in NULLS}
```

Одна строка в цикле. Тогда читатель видит полную картину.

### 3. `I5` — только `vs_sh`, без `sig`-определения

Для I5 в `summ["_all"]` есть `I5_vs_sh_tau_s1` и `I5_vs_sh_r1_L16`, но нет `sig`-определения (как для остальных FIELDS). По docstring «Для I5 — только vs sh», но решение `sig[sh] = lo*hi > 0` не выводится.

**Fix**: добавить явно:
```python
for key in ("tau_s1", "r1_L16"):
    ...
    lo, hi = s[f"I5_vs_sh_{key}"][1]
    s[f"I5_vs_sh_{key}_sig"] = lo is not None and lo * hi > 0
```

### 4. `surrogate_lines` — нет аудита step после clamp

`E50.synth_white_matched` делает `step = min(max(step, 5e-4), 0.6)`. Если real `mean_step < 5e-4` (очень гладкие данные), surrogate получит 5e-4 — в 5+ раз крупнее. Это будет видно в `row["step_real"]` vs `row["sur{j}"]["step"]`, но лучше добавить явный флаг:

```python
if row["step_real"] < 6e-4:
    row["sur_step_clamped"] = True
```

P3 — можно и без этого, если проверить `step_real` на первом прогоне.

### 5. `_stack1` без `.detach()`

```python
def _stack1(vs):
    return torch.stack([v.reshape(-1) for v in vs])[:, :, None]
```

vs в `surrogate_lines` использует `.detach()` внутри. В `row["step_real"]` — нет. Не баг (v1s отдеточены), но консистентность: лучше добавить.

---

## Прогноз

После запуска ожидаем (по paper.md E-28, E-31):

| Инвариант | Ожидание |
|---|---|
| I1 | `directional(order-free)` — гладкость есть, но 25% swap слабо ломает малые лаги |
| I2 | `directional` или `domain-dependent` — coherent v2 |
| I3 | `null-compatible` — реальная траектория не идеально геодезическая, I3 не отличает её от sur |
| I4 | `directional+ordering` — сильнейший сигнал (вытянутое облако приращений) |
| I5 | зависит от домена; скорее `long-range-only` или значимо vs sh |

**Если I3 окажется `null-compatible`** — это **честный** результат, а не провал. Он означает: «I3 ловит геодезичность, а реальная траектория не идеально геодезическая — так что I3 не отличает её от sur». Это информация, а не ошибка.

**Если I4 = `directional+ordering`, I1 = `directional(order-free)`, I2 = `directional`** — дифференцированный ответ, показывающий, что разные инварианты ловят разные аспекты структуры. Это содержательно.

---

## Pre-launch checklist

1. **P0-проверка**: `E50.synth_white_matched(Vs, D, 1, T, seed)` принимает `[T, D, 1]` тензор — импортировать и вызвать один раз в REPL:
   ```python
   import torch, exp50v3_curvature as E50
   Vs = torch.randn(10, 400, 1, dtype=torch.float64)
   out = E50.synth_white_matched(Vs, 400, 1, 10, 42)
   print(len(out), out[0].shape)  # должен быть (10, (400, 1))
   ```
   Если работает — запускать.

2. **P1-фиксы** (опциональные, но полезные):
  - `boot_ci` вместо `boot_mean_ci` в `vs_{n}_{f}` (п.1)
  - `sig_{f}`/`rob_{f}` в summary (п.2)
  - `I5_vs_sh_{key}_sig` (п.3)

3. **Запуск `--controls-only`** и проверка:
  - `SELFTEST: ok`?
  - `geo I2_mean ≈ +1.0`?
  - `white I2_mean ≈ 0` (|·|<0.15)?
  - `gauge_dev < 1e-6`?

4. **Полный прогон** и чтение:
  - `summary._all.verdict` — дифференцированная картина по 5 инвариантам;
  - `summary._all.notrobust_{f}` — какие инварианты «overall sig, но не robust»;
  - `per_doc[*].step_real` vs `per_doc[*].sur{j}.step` — matching audit (должны совпадать);
  - `summary._all.max_gauge_dev` — должно быть ~1e-10 для r=1, D=1024.

---

## Итог

Скрипт хорошо проработан после трёх раундов ревью. Оставшиеся P2 — косметика для читателя, не влияют на валидность чисел. После P0-проверки `synth_white_matched` на `[T,D,1]` — запускать.