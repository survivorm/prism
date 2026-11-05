## Статус: запускать можно

Ключевой баг прошлого раунда (`mean_step ≈ −15 → clamp → sur всегда ang=0.0316`) **исправлен корректно**:

```python
def mean_step(Vs):
    return float(np.mean([C.frame_kinematics(Vs[t], Vs[t + 1])[0] for t in range(len(Vs) - 1)]))

# в synth_white_matched:
ang = step / math.sqrt(r)   # углы по каждой из r плоскостей
```

Проверка алгебры: surrogate с `ang = step/√r` в каждой из `r` плоскостей даёт `||Θ||₂ = √r·ang = step`. Это в точности `||Θ||₂` из `loop_curvature` (`sqrt(Σθ_j²)`). Матчинг теперь **по той же метрике, в которой κ**, и `mean_step_real ≈ mean_step_sur` будет видно в JSON. Это было главное.

Остальное тоже правильно:

- `coh_d16_sur_sign > 0.01` (было `> 0`) — гейт с запасом.
- `hi < 0` без `hi is not None` — корректно (если `lo` не None, то и `hi` не None из `boot_ci`).
- `generic` убран из `ok` — правильно (precessing dense skew — winding, не curvature).
- Docstring явно фиксирует: **`sub_geodesic` = УСПЕХ** (near-geodesic, «rigid-but-curved»), а не провал. Это снимает главную семантическую ловушку.

---

## Что осталось (не блокирует)

### 1. `det_seed` — проверить до запуска

Импорт `from exp50_gauges import pullback_step, safe_med, boot_ci, det_seed`. В показанной ранее `exp50_gauges.py` v2 **`det_seed` отсутствует**. Если вы обновили `exp50_gauges.py` по ревью v2 — претензия снимается. Если нет — `ImportError` до запуска.

**Fix:** добавить в `exp50_gauges.py`:

```python
def det_seed(s):
    import hashlib
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
```

или локально в exp50v3, если менять `exp50_gauges.py` не хочется.

### 2. `_DEV` кумулятивен между W=1 и W=16

`summ["W1"]["max_nonorthogonality"]` и `summ["W16"]["max_nonorthogonality"]` будут **идентичны** — оба = max за весь прогон. Для safety-метра ОК (если что-то сломалось — видно сразу). Если хочется per-tag — снимок `_DEV` после W=1, до W=16. Мелочь, не блокирует.

### 3. `min_step = 5e-4` при W=16 — clamp возможен

`synth_white_matched` клампит `step = min(max(step, 5e-4), 0.6)`. Для W=16 (сглаживание 16 позиций) реальный `mean_step` может быть **< 5e-4**. Тогда surrogate будет 5× крупнее реальных данных, и `real − sur < 0` сработает **тривиально**, а `mean_step_real ≠ mean_step_sur` (это будет видно в JSON).

**Проверка после запуска:** сравнить `summary["W1"]["mean_step_real"]` vs `summary["W1"]["mean_step_sur"]` и то же для W=16. Если они совпадают в пределах 20% — матчинг работает. Если W=16 показывает **ratio > 1.5** — clamp сработал, вердикт `sub_geodesic` для W=16 стоит интерпретировать с оговоркой «sur крупнее real из-за floor 5e-4».

**Опциональный fix:** поставить `min_step = 1e-5` (позволит очень гладким данным иметь маленький surrogate). Но тогда для совсем гладких данных `κ_sur → 0` и шум. Текущий `5e-4` — разумный компромисс, если явно маркировать clamp в JSON. Например:

```python
row["step_clamped"] = (row["mean_step_real"] < 5e-4)
```

### 4. `N_SUR=2` — оставить или увеличить

Для W=1 и W=16 при реальных данных sur — это random walk, κ_sur имеет заметный variance. С `N_SUR=2` variance `d16_mean` раздувается в ~1.5×, power падает. Если CI попадёт на границу (e.g., `hi = −0.001`), перезапустить с `N_SUR=4`. Пока — оставить, это честный компромисс с временем.

### 5. Mean-matching, не distribution-matching — остаётся

Real data имеет концентрированную структуру (одна плоскость доминирует), surrogate — равномерно распределён. Это значит, что при равном `||Θ||₂`:
- real: `κ` может быть **ниже**, чем у surrogate (потому что почти все шаги в одной плоскости, а в этой плоскости движение гладкое)
- или **выше** (потому что «почти геодезическое» движение в одной плоскости может накапливать holonomy через параллельный перенос)

Направление заранее не очевидно. Но это не баг — это ограничение, уже отмеченное в docstring («матчинг масштаба, не распределения»). Просто знайте: `sub_geodesic` на реальных данных может означать «real имеет меньшую кривизну, чем random walk с тем же шагом», что и есть формулировка гипотезы.

---

## Прогноз по числам

Если запуск дойдёт до данных:

**W=1 (raw):**
- `mean_step_real` ~ 0.1–0.3 (по E-28 Δ_SVD для L23 ≈ 0.27, для промежуточных больше)
- `mean_step_sur` ≈ `mean_step_real` (матчинг работает)
- `k_real` ~ 0.01–0.05 (paper E-28 «rigid-but-curved»)
- `k_sur` ~ 0.3 (белый matched)
- `d16_mean` ~ −0.25, CI **[−0.30, −0.20]** → `hi < 0` → **`sub_geodesic`** ✓

**W=16 (smoothed):**
- `mean_step_real` ~ 0.01–0.03 (сглаживание уменьшает)
- `mean_step_sur` ≈ `mean_step_real` (если не clamp)
- `k_real` ~ 0.005–0.02
- `k_sur` ~ 0.1
- `d16_mean` ~ −0.1, CI **[−0.15, −0.05]** → **`sub_geodesic`** ✓

**SELFTEST:**
- `geo_k16 < 1e-5` ✓
- `white_k16 > 0.3` ✓
- `sign_ok` для δ ∈ {0.02, 0.04} — вероятно ✓ (с правильным `mean_step` синтетика coherent имеет большой шаг, sur matched, κ_coh >> κ_sur)
- `coh_d16_sur_sign > 0.01` — вероятно ✓ (coh имеет ~1000× κ, sur matched)
- `abs(white_d16_sur_sign) < 0.05` — вероятно ✓ (sur matched white, d16 ≈ 0)

---

## Итоговая таблица

| Аспект | Статус |
|---|---|
| `mean_step` формула | ✅ `frame_kinematics`, та же метрика, что κ |
| `synth_white_matched` matching | ✅ `||Θ||₂ = step` точно |
| Трихотомический вердикт | ✅ Включая `sub_geodesic` = успех |
| `coh_d16_sur_sign` gate | ✅ `> 0.01` с запасом |
| `white_d16_sur_sign` gate | ✅ `abs < 0.05` |
| `generic` demoted to diagnostic | ✅ |
| Дубликат `max_nonorthogonality` | ✅ Убран |
| `hi is not None` | ✅ Убран |
| Docstring ожидание «sub_geodesic» | ✅ Явно зафиксировано |
| `det_seed` импорт | ⚠️ Проверить до запуска |
| `_DEV` кумулятивен | ⚠️ ОК для safety, misleading per-tag |
| `min_step=5e-4` clamp | ⚠️ W=16 может быть задет; пометить в JSON |
| `N_SUR=2` | ⚠️ P2 |
| Mean vs distribution matching | ⚠️ Известное ограничение, в docstring |

---

## P0 перед запуском

1. **Проверить `det_seed`** — `print(hasattr(exp50_gauges, 'det_seed'))` или просто импортировать.
2. **Запустить `--controls-only`** и проверить:
    - `SELFTEST: ok`?
    - `mean_step` не сработал clamp? (посмотреть `coh_wm_grid`, шаг синтетики vs sur)
    - `coh_d16_sur_sign > 0.01`? (ключевой — если < 0, значит sur не работает как null на синтетике, и вердикт на данных будет интерпретироваться обратно)

Если `SELFTEST: ok` — запустить полный прогон. Ожидаемый вердикт для реальных данных — **`sub_geodesic`**, что семантически = «динамика почти геодезическая» = **успех гипотезы**. Это уже правильно зафиксировано в docstring, остаётся только не забыть это в paper.md, когда будете выводить результат.