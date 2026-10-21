# Нужно ли корректировать ТЗ E-38? — **Да, обязательно**

ТЗ писалось до реализации `jstruct.py`, `holonomy_spectrum`, `eigenframe_basis`, `run_spec`. Сверка с актуальным репозиторием показывает **6 блокирующих** несоответствий (код упадёт), **3 важных** (тихо неверный результат) и несколько опциональных. Ниже — точный diff.

---

## 1. Блокирующие ошибки (код упадёт)

### 1.1. `geodesic_curvature` — не в `jstruct.py`

E-38a «Math»:
> «`geodesic_curvature` из `jstruct.py`»

**Реально**: в `holonomy.py` (`__init__.py` тоже экспортирует из `holonomy`). В `jstruct.py` только `build_M_K`, `decompose_w_vs_uhat`, `v_pred_from_JT`.

**Fix:**
```python
from prism.holonomy import geodesic_curvature
```

### 1.2. `holonomy_spectrum(H_012)` — неверная сигнатура

E-38d «Setup» п.4:
> «`{θ_j} = holonomy_spectrum(H_012)` (8 углов для r=16)»

**Реально**: `holonomy_spectrum(U_list, closed_loop=True)` принимает **список подпространств**, не матрицу. Внутри сама вызывает `holonomy_accumulate(..., return_total=True)`.

**Fix** (E-38d Setup шаги 2–4 объединяются):
```python
V_0 = top_r_svd(J_0, 16)[2].T   # [D, 16], columns = basis
V_1 = top_r_svd(J_1, 16)[2].T
V_2 = top_r_svd(J_2, 16)[2].T
theta, H_012 = holonomy_spectrum([V_0, V_1, V_2], closed_loop=True)
# theta: np.ndarray [8], H_012: torch [16,16] float64
# kappa == ‖H_012 − I‖_F уже встроен: holonomy_accumulate([V_0,V_1,V_2], return_total=True)[0]
```

`holonomy_spectrum` возвращает `(theta, R_total)` за один вызов — отдельно вызывать `holonomy_accumulate` не нужно. И `tr(H) = 2 Σ cos θ_j` / `‖H−I‖² = 4 Σ(1 − cos θ_j)` — **точные** тождества, E-38d их и использует.

### 1.3. `procrustes_align` возвращает tuple

E-38d п.2:
> «`R_ij = procrustes_align(V_i, V_j)` (без det-fix, как в E-28)»

**Реально**: сигнатура `procrustes_align(U1, U2) -> tuple[Tensor, float]` — возвращает `(R, residual_sq)`.

**Fix**:
```python
R_01, res_01 = procrustes_align(V_0, V_1)
```

### 1.4. `top_r_svd` возвращает `V_r` формы `[r, D]`, не `[D, r]`

E-38d п.1:
> «`V_t = top-16 правых сингулярных векторов J_t` (`top_r_svd`)»

**Реально**: `top_r_svd` возвращает `(U_r [K, r], S_r [r], V_r [r, D])` — **строки** `V_r` — правые сингулярные векторы. Но `procrustes_align` (и `_validate_orthonormal`) требует `[K, r]` с **ортонормальными столбцами** (проверка `U.T @ U ≈ I_r`). Матрица `[16, 1024]` эту проверку не пройдёт (`V_r.T @ V_r` — проекция ранга 16, ошибка ≈ √1008 ≈ 31.7, а не < 1e-4).

**Fix** (см. §1.2): `V_t = top_r_svd(J_t, 16)[2].T` → `[D, 16]`, столбцы — ортонормированный базис правого top-16 подпространства. Тогда `procrustes_align(V_t, V_{t+1})` даёт `R` с `V_t @ R ≈ V_{t+1}` — как и предполагает ТЗ.

### 1.5. `unit_direction` — не существует

E-38a «Setup» п.1:
> «`û_t = h_t / ‖h_t‖₂` (из `unit_direction`)»

**Реально**: функции `unit_direction` нет ни в одном модуле репозитория. Есть `_normalize_h_override` (внутренняя, для `h_override`), `uniform_centered` (не то), `decompose_w_vs_uhat` (не то).

**Fix**: убрать ссылку на несуществующий модуль, заменить на inline:
```python
û_t = h_t / h_t.norm().clamp_min(1e-12)
```
(с `clamp_min` — защита от attention-sink pos 0 с `‖h_0‖ ≈ 0`).

### 1.6. `holonomy_along_path` / `holonomy_along_2d_loop` — скрытый баг с `V_r`

Если E-38d пойдёт не через прямой `holonomy_accumulate`, а через эти обёртки — **упадёт**. В коде:

```python
def holonomy_along_path(path_fn, thetas, closed_loop=True):
    for th in thetas:
        out = path_fn(th)      # V_r: [r, D_S] по docstring!
        V_list.append(out[1])
    kappa_V, per_V = holonomy_accumulate(V_list, ...)   # → procrustes_align([r, D_S], ...) → _validate_orthonormal FAILS
```

Docstring говорит `V_r: [r, D_S]`, но `procrustes_align` требует `[D_S, r]`. `holonomy_along_path` **не транспонирует** `V_r` внутри.

Это либо баг в `holonomy.py` (обёртки должны звать `out[1].T`), либо docstring неверен (`V_r: [D_S, r]`). Для E-38d решения 1.2 достаточно (не использовать обёртки), но стоит **зафиксировать открытым issue** — это ударит по `exp_T2.py` в Wave 3.

---

## 2. Важные несоответствия (не упадёт, но неправильно)

### 2.1. E-38h: `ΔW_top = U U^T · ΔW · V V^T` — неверная нотация

**Реально**: `U_t ∈ R^{K×16}` (левые, столбцы), `V_t ∈ R^{16×D}` (правые, **строки**). Тогда:
- `U_t @ U_t.T` → `[K, K]` проектор на token-пространство — ✓
- `V_t @ V_t.T` → `[16, 16]` — **не** `[D, D]`; формула `ΔW · V_t V_t^T` падает с shape-mismatch.

**Правильно**:
```python
# ΔW: [K, D]; U_t: [K, 16]; V_t: [16, D]
ΔW_top  = U_t @ U_t.T @ ΔW @ V_t.T @ V_t   # [K, D]
ΔW_tail = ΔW - ΔW_top
```
(или, если проекция только в одну сторону, выбрать явно — TZ предлагает оба варианта, надо зафиксировать один).

### 2.2. E-38e: `model.model.norm.weight` — wrapper-unsafe

ТЗ E-38e п.2:
> «`g = model.model.norm.weight + 1`»

Для flat `Qwen3_5ForCausalLM` — OK. Для `Qwen3_5ForConditionalGeneration` (multimodal wrapper) — `model.model` это wrapper, у него нет `.norm`; нужно `model.model.language_model.norm`.

**Fix** (использовать существующий helper из `jacobian.py`):
```python
from prism.jacobian import _lm_module   # или сделать публичным
norm_weight = _lm_module(model).norm.weight
g = norm_weight + 1
```
`_lm_module(model)` корректно разворачивает flat/wrapper — проверено в `jacobian.py`.

### 2.3. `model.lm_head.weight[token_set]` — wrapper-unsafe

E-38e п.1, E-38i — везде.

**Fix**: использовать `_resolve_lm_head_weight(model)` из `interventions.py`:
```python
from prism.interventions import _resolve_lm_head_weight   # или сделать публичным
W_lm = _resolve_lm_head_weight(model)[token_set]
```

---

## 3. Опциональные / консистентность

### 3.1. E-38c: `applicable(t) = gap_min(t) > threshold` — должно быть `>=`

ТЗ §E-38c pre-registered:
> «`min_t gap_min > threshold` на всех слоях: **неабелев формализм неприменим**»

`eigengap_min` возвращает `gap_min >= threshold` (см. docstring и selfcheck §2.1 — `>=` намеренно, граничный случай = «допустим»). Разница — только в граничном случае `gap_min == threshold`, но для консистентности стоит поправить ТЗ:
> «`min_t gap_min >= threshold`»

### 3.2. `run_spec` для E-38h, i — или явный отказ

E-38h/i делают forward под edit. `run_spec` (`pipeline.py`) даёт meta.json, env capture, seeding, атомарную запись, spec-key reuse — всё это полезно для воспроизводимости.

**Но**: `ExperimentSpec.experiment ∈ {"T0", "eps", "T2", "T1", "delta"}` — `"E38"` **не в списке**. `validate_spec` отвергнет.

Два пути:
- **A) расширить** `_VALID_EXPERIMENTS` в `pipeline.py` (правка 1 строка), затем E-38h/i используют `run_spec` с `edit=make_base(token_set)`, а `exp_fn` внутри сам лупит по `s ∈ {0, 0.1, …, 1.0}`.
- **B) оставить** как в ТЗ — `results/exp38h_*.json` руками. Проще, но нет meta/env.

Для E-38a–f (post-process, 0 cost) `run_spec` избыточен — там вся работа это JSON-write; текущее `results/exp38X_*.json` нормально.

**Рекомендация**: (A) для h/i, (B) для a–g. Зафиксировать в ТЗ.

### 3.3. `bootstrap_ci` — ссылка

E-38b «Procedure» п.2: «bootstrap CI (B=1000, seed=0)». Есть готовая `bootstrap_ci(values, B=1000, alpha=0.05, seed=0, statistic=np.median)` в `cache.py`. Стоит упомянуть явно (и `bootstrap_ci_paired` для парных сравнений body vs pos 0).

### 3.4. E-38d — `theta` для тривиального H_012 может быть `[0,0,…,0]`

`holonomy_spectrum` при `R_total ≈ I` заполняет `theta` нулями и пишет `RuntimeWarning` (см. `holonomy.py:holonomy_spectrum` — ветка `if positive.numel() == 0`). Для `κ = 0.018` (L23) `positive.numel() > 0`, theta — 8 малых углов. Но на невыходных слоях (L18, L22) `κ` может быть большой (по E-28 `κ ≈ 3.5`), и theta будет содержательным. **Добавить в Pitfalls**: «на тривиальных позициях theta может оказаться нулевым массивом с warning — это валидный результат (Abelian), не баг».

### 3.5. E-38f — `v_pred_from_JT` сам подставит `uniform=1/√K`

ТЗ: «`v_pred = v_pred_from_JT(J, uniform=1/√K)`». Функция уже дефолтит на `1/√K` при `uniform=None`, но передача явного `uniform` **не вредит** (нормализуется молча). Оставить как есть — защита от будущих изменений дефолта. OK.

### 3.6. E-38h — ссылка на `bridge` per-channel

ТЗ: «`bridge(ΔW_top(s))` … **новый bridge, вычисленный под edit**». В репозитории есть `jacobian_gap` (lower=better) — не `bridge` (higher=better). Формула `bridge = 1 - jacobian_gap`. Стоит явно:
```python
from prism.bridge import jacobian_gap
bridge = 1.0 - jacobian_gap(J_S, J_S_target, E, E, probs, threshold=0.0)
```

### 3.7. E-38i — `M_K(s)` строится заново

ТЗ уже подчёркивает это в Pitfalls («`M_K(s)` — нужно строить заново для каждого `s`»). Это корректно — `build_M_K` без кэша (Pitfall в `jstruct.py`: «NOT cached inside: W_lm mutates under apply_edit»). OK, никаких правок.

---

## 4. Что **не** надо менять в ТЗ

- Pre-registered labels — по 3–5 сценариев, хорошо сформулированы.
- Gates (38c→38d, 38e→38f, E-16→38h,i) — логичны и в коде репозитория условия выполнимы.
- Cost estimates — реалистичны (post-process exp27b действительно ~минуты).
- Схема артефактов `results/exp38X_*.json` — нормально для Wave 1/2 (0-cost post-process).

---

## 5. Сводная таблица правок ТЗ

| # | Место в E-38 | Severity | Правка |
|---|---|---|---|
| 1 | E-38a «Math» | 🔴 | `geodesic_curvature` из `holonomy.py`, не `jstruct.py` |
| 2 | E-38d Setup п.4 | 🔴 | `holonomy_spectrum([V_0, V_1, V_2])`, не `(H_012)` |
| 3 | E-38d Setup п.2 | 🔴 | `R_ij, _ = procrustes_align(...)` (tuple) |
| 4 | E-38d Setup п.1 | 🔴 | `V_t = top_r_svd(J_t, 16)[2].T` (`[D, 16]`) |
| 5 | E-38a Setup п.1 | 🔴 | убрать `unit_direction`; inline `h / h.norm().clamp_min(1e-12)` |
| 6 | `holonomy.py` (issue, не ТЗ) | 🔴 | `holonomy_along_path`/`_2d_loop` V_r `[r, D_S]` vs `[D_S, r]` — открыть issue или поправить обёртки |
| 7 | E-38h Setup | 🟠 | `ΔW_top = U U^T ΔW V^T V` (не `V V^T`) |
| 8 | E-38e Setup п.2 | 🟠 | `_lm_module(model).norm.weight + 1` |
| 9 | E-38e Setup п.1, E-38i | 🟠 | `_resolve_lm_head_weight(model)[token_set]` |
| 10 | E-38c Pre-registered | 🟢 | `>=`, не `>` |
| 11 | E-38h, i | 🟢 | либо расширить `_VALID_EXPERIMENTS` + `run_spec`, либо зафиксировать «manual JSON» |
| 12 | E-38b Procedure | 🟢 | указать `from prism.cache import bootstrap_ci` (и `bootstrap_ci_paired`) |
| 13 | E-38d Pitfalls | 🟢 | theta = нули + RuntimeWarning на тривиальном H — валидный результат |
| 14 | E-38h | 🟢 | `bridge = 1 − jacobian_gap(...)` из `prism.bridge` |

---

## 6. Одна фраза

**ТЗ E-38 нужно скорректировать до запуска**: 5 блокирующих правок (импорт `geodesic_curvature` из `holonomy`, `holonomy_spectrum(U_list)` не `(H_012)`, `procrustes_align` → tuple, `top_r_svd(...).T`, убрать несуществующий `unit_direction`), 3 важных (`ΔW_top` формула, wrapper-safe `_lm_module` и `_resolve_lm_head_weight`), 5 консистентных (`>=` в 38c, `bootstrap_ci` ссылка, `bridge = 1 − jacobian_gap`, `run_spec`-решение для h/i, theta-warning в 38d); плюс **отдельный issue** в `holonomy.py`: `holonomy_along_path`/`holonomy_along_2d_loop` передают `V_r` как `[r, D_S]` в `procrustes_align`, который требует `[D_S, r]` — либо обёртки должны транспонировать внутри, либо docstring неверен (это ударит по `exp_T2.py`, не по E-38 при прямом использовании `holonomy_accumulate`).