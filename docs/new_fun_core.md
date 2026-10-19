# Исправленный `new_fun_core.md`

Изменения относительно предыдущей версии:
1. **§0** — добавлен принцип «не дублировать семантически» с явными ссылками на существующие `spectral.py`/`holonomy.py`.
2. **§2.1 `eigengap_min`** — сигнатура `sigma` → `spectrum` (eigenvalues, как у `eigengap`/`stable_rank`/`participation_ratio`); алгоритм без `sigma ** 2`; edge cases и тесты переписаны под спектр; docstring явно разводит с `eigengap` (это **не** `min(eigengap(...))`).
3. **§1.2 `holonomy_spectrum`**, **§1.4 `eigenframe_basis`** — в docstring добавлена сноска «NOT a substitute for `top_r_svd`, because …».
4. **§3.2 `decompose_w_vs_uhat`**, **§3.3 `v_pred_from_JT`** — в docstring добавлена сноска «NOT a generic projection; distinct from `topology.project_to_subspace`».
5. **§6** — добавлены пункты «не писать», закрывающие оставшиеся риски путаницы.
6. **§7** — обновлена итоговая фраза.

---

```markdown
# ТЗ для агента: ядро функций Фазы 1

## 0. Принципы

- **Не дублировать существующее.** В репозитории уже есть `procrustes_align`, `holonomy_accumulate`, `holonomy_along_path`, `holonomy_along_2d_loop` (`holonomy.py`), `top_r_svd`, `spectrum_from_gramian`, `eigengap`, `entropy_effective_rank`, `stable_rank`, `participation_ratio`, `snr_top1_vs_rest`, `snr_e14`, `snr_e14_parts`, `gauge_share`, `gauge_share_top1` (`spectral.py`), `apply_edit` (`interventions.py`), `SVDCache`/`MetricCache`/`deterministic_mode` (`cache.py`). Всё новое — импортируется или расширяется, не переписывается.
- **Docstring-нота против путаницы.** Если новую функцию можно спутать с существующей (та же операция над похожим объектом), в её docstring обязательна одна строка `NOT a substitute for <existing>, because <reason>`. Это относится к парам: `eigengap_min` vs `eigengap`; `eigenframe_basis` vs `top_r_svd`; `decompose_w_vs_uhat`/`v_pred_from_JT` vs будущий `topology.project_to_subspace` (появится позже, но читатель увидит обе).
- **Никаких новых геометрических объектов.** Только формализация уже существующих (`H_{012}`, `Ω`, `M_K`, `û_t^⊥`). Moratorium на WZ-терминологию активен.
- **Стиль существующих модулей:** docstring на английском, type hints, `ValueError`/`RuntimeError` с диагностикой (первое неверное значение/индекс), тесты через `pytest` в `tests/`.
- **Порядок файлов в ТЗ — порядок реализации.** Каждый следующий файл опирается на предыдущий.
- **Каждая функция — тонкая** (5–30 строк), без классов, без кэш-слоёв внутри (кроме явно оговорённых).

---

## 1. Файл: `prism/holonomy.py` (расширение существующего)

### 1.1. `holonomy_accumulate` — расширить параметром `return_total`

**Смысл:** добавить опциональный возврат полной матрицы `R_total ∈ SO(r)` для последующего спектрального анализа. Существующий вызов `(κ, per_step)` не меняется.

**Сигнатура (изменение):**
```python
def holonomy_accumulate(
    U_list: Sequence[torch.Tensor],
    closed_loop: bool = True,
    return_total: bool = False,
) -> tuple[float, list[float]] | tuple[float, list[float], torch.Tensor]:
```

**Алгоритм:** существующий, плюс:
```
if return_total:
    return kappa, per_step, R_total   # R_total already computed internally, float64
return kappa, per_step
```

**Краевые условия:** без изменений. `R_total` — уже существующая внутренняя переменная, dtype `float64`, shape `[r, r]`.

**Тесты** (в `tests/test_holonomy_ext.py`):
- Backward-compat: `holonomy_accumulate(U_list)` возвращает 2-tuple.
- `holonomy_accumulate([U, U], closed_loop=True, return_total=True)` → `κ ≈ 0`, `R_total ≈ I` (tol 1e-6).
- Композиция: `R_total` из 3-элементного `U_list` == `R_01 @ R_12 @ R_20` (ручное произведение через `procrustes_align`, tol 1e-6).
- `R_total^T @ R_total ≈ I` (ортонормальность, tol 1e-6).
- `‖R_total − I‖_F == κ` (точное тождество, tol 1e-6).

**Импорты:** без новых.

---

### 1.2. `holonomy_spectrum` — новый

**Смысл:** спектр `{θ_j}_{j=1..r//2}` углов вращения из `R_total` (лемма [M-2] в ТЗ математики). Заменяет «свёртку в скаляр κ» на полную информацию о вращении.

**Отношение к существующему.** `holonomy_spectrum` — **convenience wrapper**, а не новый объект: внутри вызывает `holonomy_accumulate(U_list, closed_loop=True, return_total=True)` и делает пост-обработку (`torch.linalg.eigvals` → `angle` → фильтр `+1`-оси → dedup `{e^{iθ}, e^{-iθ}}` → sort). Это **не** замена `holonomy_accumulate`: κ по-прежнему считается там; спектр — производная величина. Использовать `spectrum_from_gramian`/`eigvalsh` **нельзя**: `R_total ∈ SO(r)` — ортогональная, но не симметричная; нужен `eigvals` (комплексный).

**Сигнатура:**
```python
def holonomy_spectrum(
    U_list: Sequence[torch.Tensor],
    closed_loop: bool = True,
) -> tuple[np.ndarray, torch.Tensor]:
    """Spectral angles of R_total; convenience wrapper over
    ``holonomy_accumulate(..., return_total=True)``.

    NOT a substitute for the scalar κ (``holonomy_accumulate()[0]``):
    κ collapses R_total to one number; this returns the full angle set
    used in paper [M-2] / §6.

    Returns (θ: np.ndarray [r//2] sorted ascending ∈ (0, π], R_total: [r,r] float64)."""
```

**Алгоритм:**
```
_, _, R_total = holonomy_accumulate(U_list, closed_loop, return_total=True)
eigvals = torch.linalg.eigvals(R_total)           # complex
angles = torch.angle(eigvals)                    # (-π, π]
positive = angles[angles > 1e-6]                 # drop +1 axis (and numerical zeros)
theta_unique = torch.unique(positive.round(decimals=8))  # dedupe {e^{iθ}, e^{-iθ}}
theta_np = theta_unique.sort().values.numpy().astype(np.float64)
return theta_np, R_total
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `r` (из `U_list[0].shape[1]`) — чётное | ожидаем `len(θ) == r//2` |
| `r` нечётное | допускается (тривиальная ось +1); `len(θ) == (r-1)//2`; предупреждение `RuntimeWarning` |
| `R_total` не ортогональна (`‖R^T R − I‖_F > 1e-4`) | `RuntimeError` с диагностикой |
| собственное значение `|λ| < 1 − 1e-6` (не на единичной окружности) | `RuntimeWarning`; продолжить (числовая неустойчивость) |
| дубликаты углов (кратные собственные значения) | `RuntimeWarning` о неоднозначности базиса, вернуть `unique` |

**Тесты** (`tests/test_holonomy_ext.py`):
- `R = I` (все `U_i` идентичны) → `θ = [0, 0, ..., 0]` (все `≈ 0`).
- `R = blockdiag(Rot(θ), Rot(θ), I_{r-4})` через прямую конструированную `U_list` → `θ = [θ, θ]` (sorted).
- `R = blockdiag(Rot(π), I_{r-2})` → `θ = [π]`.
- `R = −I_r` (r=4) → `θ = [π, π]`.
- Тождество: `tr(R_total) == 2 Σ_j cos(θ_j)` (tol 1e-6).
- Тождество: `‖R_total − I‖_F² == 4 Σ_j (1 − cos θ_j)` (tol 1e-6).

**Импорты:** `import numpy as np` (уже используется в `cache.py`, но не в `holonomy.py` — добавить).

---

### 1.3. `geodesic_curvature` — новый (заменяет ранее предложенную `spherical_excess`)

**Смысл:** дискретная геодезическая кривизна кривой `û_t` на сфере `S^{D-1}` по тройке `(u1, u2, u3)` — локальный Abelian-агностик для сравнения с `κ` (лемма [M-3] в исправленной формулировке). **Не** дефект неравенства треугольника.

**Сигнатура:**
```python
def geodesic_curvature(
    u1: torch.Tensor, u2: torch.Tensor, u3: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Discrete geodesic curvature κ_g ∈ R≥0 at u2, from three unit vectors."""
```

**Алгоритм:**
```
# Все три — [D], unit (‖u_i‖ = 1)
t1 = u1 - <u1, u2> * u2       # tangent at u2, toward u1
t3 = u3 - <u3, u2> * u2       # tangent at u2, toward u3
n1, n3 = ‖t1‖, ‖t3‖
if n1 < eps or n3 < eps:
    return 0.0                # degenerate: u1 or u3 collinear with u2
cos_turn = clip(<t1, t3> / (n1 * n3), -1, 1)
turn = π - arccos(cos_turn)
arc = (‖u1 - u2‖ + ‖u3 - u2‖) / 2
return turn / arc
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `u_i` не unit (`|‖u_i‖−1| > 1e-4`) | `ValueError` (с указанием индекса и значения) |
| `D < 2` | `ValueError` |
| `u1 == u2` или `u2 == u3` (норма тангенса < eps) | вернуть `0.0` (не ошибка; кривая «вырождена» в тривиальном смысле) |
| `u1 == u3` (туда-обратно) | `turn = π − arccos(1) = 0`; `arc > 0` → `κ_g = 0` |
| дуга нулевая (`arc < eps`) | `RuntimeError` (все три точки совпадают — неинформативно) |
| `D = 1` | `ValueError` (невозможно построить касательные) |

**Тесты** (`tests/test_holonomy_ext.py`):
- Три точки на геодезической (большой круг `S^2`): `u1=(1,0,0)`, `u2=(0,1,0)`, `u3=(−1,0,0)` → `κ_g ≈ 0` (tol 1e-6).
- Три точки на малой окружности радиуса `R = 0.5` (в плоскости `z = √(1−R²)`): `κ_g ≈ 1/R = 2.0` (tol 5%).
- Три базисных вектора `e1, e2, e3` в `R^3` → `κ_g ≈ (π/2)/√2 ≈ 1.1107` (tol 1e-4, ручной расчёт).
- `u1 == u2` (коллинеарность) → `0.0` (не исключение).
- Не-unit вход → `ValueError`.
- `D=1` → `ValueError`.

**Импорты:** `import math` (уже есть в `holonomy.py`? проверить; если нет — добавить).

---

### 1.4. `eigenframe_basis` — новый

**Смысл:** канонический ортонормированный базис в `û_t^⊥`, в котором ограничение `M_K^T M_K` диагонально (конвенция [M-5]). Используется для однозначного разложения `H_{012}` на компоненты.

**Отношение к существующему.** **Не** замена `spectral.top_r_svd`: `top_r_svd(J, r)` возвращает правые сингулярные векторы **`J`** (общая `K×D`-матрица); `eigenframe_basis(M_K, û)` возвращает собственные векторы **симметричной `P_⊥ M_K^T M_K P_⊥`** (размерность `D×D`, объект в `û^⊥`). Разные операторы, разная размерность, разная задача (`top_r_svd` — сжатие `J`; `eigenframe_basis` — фиксация базиса для `H_{012}`, конвенция `[D-holonomy-frame]`). Вызывающий, которому нужны правые сингулярные векторы `J`, использует `top_r_svd`; вызывающий, которому нужен канонический базис в `û^⊥`, — `eigenframe_basis`.

**Сигнатура:**
```python
def eigenframe_basis(
    M_K: torch.Tensor,   # [K, D]
    uhat: torch.Tensor,  # [D] unit
) -> torch.Tensor:
    """Canonical basis B [D, D-1] orthonormal, B^T uhat = 0, ordered by
    decreasing eigenvalue of P_⊥ M_K^T M_K P_⊥.

    This is a CONVENTION per [M-5]; the basis is data-determined up to
    permutation and sign, but the decomposition of H_{012} using it is
    fixed once chosen.

    NOT a substitute for spectral.top_r_svd: the operator is the symmetric
    P_⊥ M_K^T M_K P_⊥ (D×D, in uhat^⊥), NOT J (general K×D). top_r_svd
    returns right singular vectors of J; this returns eigenvectors of the
    gauge-projected M_K Gramian. Consumers needing right singular vectors
    of J must call top_r_svd directly."""
```

**Алгоритм:**
```
uhat = uhat.reshape(-1)
A = M_K.T @ M_K                         # [D, D]
A_proj = A - torch.outer(uhat, A @ uhat) - torch.outer(A @ uhat, uhat) + torch.dot(uhat, A @ uhat) * torch.outer(uhat, uhat)
                                        # = P_⊥ A P_⊥
eigvals, eigvecs = torch.linalg.eigh(A_proj)
mask = eigvals.abs() > 1e-8             # исключить тривиальную û-ось
eigvals_pos = eigvals[mask]
eigvecs_pos = eigvecs[:, mask]
order = torch.argsort(eigvals_pos, descending=True)
B = eigvecs_pos[:, order]                # [D, D-1]
B = B / B.norm(dim=0, keepdim=True)      # renormalize (eigh returns unit, но для устойчивости)
return B
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `uhat` не unit (`|‖uhat‖−1| > 1e-4`) | `ValueError` (не нормализовать молча — вызывающий должен передать unit) |
| `M_K.shape[1] != uhat.shape[0]` | `ValueError` |
| `D < 2` | `ValueError` |
| Тривиальная û-ось не отделяется (`|λ_min| > 1e-8`, т.е. `M_K^T M_K` не вырождена на û) | `RuntimeWarning` (конвенция нарушена; но B всё равно вернуть, исключив наименьшую λ) |
| Кратные собственные значения (degenerate) | `RuntimeWarning` о неоднозначности выбора базиса внутри вырожденного блока |

**Тесты** (`tests/test_holonomy_ext.py`):
- `M_K = I_D`, `uhat = e1` → `B` любая ортонормированная в `e1^⊥`; проверить: `B^T B ≈ I_{D-1}`, `B^T e1 ≈ 0`.
- `M_K = diag(2, 1, 0.5, ...)`, `uhat = e1` → `B[:, 0] = e2` (наибольшее λ), `B[:, 1] = e3`, ...
- Реконструкция: `B @ diag(λ_rest) @ B^T ≈ P_⊥ A P_⊥` (tol 1e-5).
- `uhat` не unit → `ValueError`.
- `M_K.shape` mismatch → `ValueError`.

**Импорты:** без новых (torch).

---

## 2. Файл: `prism/spectral.py` (расширение существующего)

### 2.1. `eigengap_min` — новый

**Смысл:** pre-flight проверка вырождения топ-k (диагностика [M-4]). Возвращает минимальный зазор `λ_i − λ_{i+1}` внутри топ-k и флаг «формализм применим / не применим».

**Отношение к существующему.** **Не** обёртка над `eigengap` и **не** `min(eigengap(...))`:
- `eigengap(spectrum)` возвращает **один** первый зазор `λ₁ − λ₂` (top-1 gap, paper §6);
- `eigengap_min(spectrum)` возвращает **минимум по всем** последовательным зазорам `λ_i − λ_{i+1}` + флаг `applicable` (pre-flight [M-4], paper §E.4).

Разные задачи, разные возвраты (float vs tuple[float, bool]), разные пороги. Импортировать `eigengap` **не нужно и не следует**. **Принимает те же входные данные** (eigenvalues — `σ²`, не сингулярные значения), что и `eigengap`/`stable_rank`/`entropy_effective_rank`/`participation_ratio`, — это снимает риск путаницы `σ vs λ = σ²` в типичном пайплайне `spectrum_from_gramian → eigengap_min`.

**Сигнатура:**
```python
def eigengap_min(
    spectrum: torch.Tensor,      # [k] eigenvalues (NOT singular values), descending not required
    eps_frac: float = 0.01,      # threshold fraction of λ_1 = max(spectrum)
) -> tuple[float, bool]:
    """Pre-flight [M-4]: min consecutive eigenvalue gap inside top-k.

    Returns (gap_min, applicable), applicable = gap_min >= eps_frac * spectrum[0]
    (after sorting descending; so spectrum[0] is λ_1 = max).

    NOT min(eigengap(...)): eigengap returns only the FIRST gap λ₁−λ₂;
    this returns the MINIMUM over ALL consecutive gaps λ_i − λ_{i+1}.
    Same input convention as eigengap/stable_rank/participation_ratio
    (eigenvalues, i.e. σ²), so it can follow spectrum_from_gramian
    directly without a manual sqrt(). Do NOT pass singular values here.
    """
```

**Алгоритм:**
```
s = torch.sort(spectrum, descending=True).values
gaps = s[:-1] - s[1:]              # [k-1] consecutive gaps in eigenvalues
gap_min = float(gaps.min())
threshold = float(eps_frac * s[0])     # s[0] = λ_1 = max
applicable = gap_min >= threshold      # >=, not > (см. ТЗ, §2.1 «уточнение»)
return gap_min, applicable
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `len(spectrum) < 2` | `ValueError` |
| `spectrum` содержит NaN/Inf | `ValueError` (указать индекс) |
| `spectrum` содержит отрицательные значения | `ValueError` (для [M-4] pre-flight нужны собственные значения `J·J^T` или `J^T·J`, которые ≥ 0; отрицательное значение — сигнал бага в caller) |
| `spectrum[0] == 0` (все λ = 0, отсортированы descending) | вернуть `(0.0, False)`, `RuntimeWarning` |
| `eps_frac ≤ 0` | `ValueError` |
| `eps_frac ≥ 1` | `ValueError` |

**Уточнение по `>=` vs `>`.** Используем `>=` (не `>`), чтобы точное равенство `gap_min == threshold` классифицировалось как applicable. Это согласуется с [M-4]: «gap_min > ε → формализм неприменим; gap_min < ε → допустим, не обязателен» — граница `== ε` трактуется как «допустим, не обязателен» (нет строгого «вырождения»), т.е. `applicable = True`. Это осознанный выбор, а не артефакт.

**Тесты** (`tests/test_spectral_ext.py`):
- `spectrum = [100.0, 81.0, 64.0, 49.0]` (= σ² для σ=[10,9,8,7]) с `eps_frac=0.01` → `(15.0, True)` (gaps = [19, 17, 15], min = 15; threshold = 100·0.01 = 1; 15 ≥ 1 → True).
- Тот же `spectrum`, `eps_frac=0.2` → `(15.0, False)` (threshold = 20; 15 ≥ 20 → False).
- `spectrum = [10.0, 9.0, 8.0]`, `eps_frac=0.1` → `(1.0, True)` (граница `==`: gaps = [1, 1], gap_min = 1; threshold = 10·0.1 = 1; 1 ≥ 1 → True — проверяет `>=`).
- `spectrum = [10.0, 9.0, 8.0]`, `eps_frac=0.099` → `(1.0, False)` (threshold = 0.99; 1 ≥ 0.99 → True — стоп, это `True`, не `False`. Пересчёт: threshold = 10·0.099 = 0.99; 1 ≥ 0.99 → True. Этот тест не различает `>=`/`>`; заменить на `eps_frac=0.101`: threshold = 1.01; 1 ≥ 1.01 → False).
    - Итог: `spectrum = [10.0, 9.0, 8.0]`, `eps_frac=0.101` → `(1.0, False)`.
- `spectrum = [100.02, 100.0, 99.98]` → `gap_min ≈ 0.02`, `False` при `eps_frac=0.01` (threshold ≈ 1.0002).
- `spectrum = [100.0, 25.0, 25.0]` → `gap_min = 0.0`, `False` (дубликаты λ).
- `spectrum = [10.0]` (k=1) → `ValueError`.
- `spectrum = [10.0, -1.0, 5.0]` → `ValueError` (отрицательное значение).
- `eps_frac=0` → `ValueError`; `eps_frac=1` → `ValueError`.
- `spectrum = [10.0, float('nan'), 5.0]` → `ValueError` (non-finite).
- `spectrum = [0.0, 0.0, 0.0]` → `RuntimeWarning`, возврат `(0.0, False)`.

**Импорты:** без новых (torch, `pytest` для тестов).

---

## 3. Файл: `prism/jstruct.py` (новый)

### 3.1. `build_M_K` — новый

**Смысл:** построить `M_K = W_lm[S] · diag(g) ∈ R^{K×D}` (лемма [M-1], фундаментально для L23).

**Сигнатура:**
```python
def build_M_K(
    W_lm: torch.Tensor,     # [V, D]
    token_set: Sequence[int],
    g: torch.Tensor,        # [D] — RMSNorm weight (1 + w, НЕ сырой w)
) -> torch.Tensor:          # [K, D]
```

**Алгоритм:**
```
ts = torch.tensor(token_set, device=W_lm.device, dtype=torch.long)
M_K = W_lm[ts] * g.unsqueeze(0)   # broadcast
return M_K
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `len(token_set) == 0` | `ValueError` |
| `max(token_set) >= V` или `min(token_set) < 0` | `ValueError` (указать неверный индекс) |
| `W_lm.shape[1] != g.shape[0]` | `ValueError` |
| `g` содержит NaN/Inf | `ValueError` |

**Не кэшировать внутри функции.** `W_lm` меняется под `apply_edit`; кэширование ключом по `data_ptr` опасно. Если нужно кэшировать в эксперименте — вызывающий код отвечает за инвалидацию.

**Тесты** (`tests/test_jstruct.py`):
- `W_lm = [[1,2],[3,4],[5,6]]`, `token_set=[0,2]`, `g=[1,1]` → `M_K = [[1,2],[5,6]]`.
- `g=[2,3]` с тем же → `M_K = [[2,6],[10,18]]`.
- `token_set=[0, 0]` (дубликаты) — допустимо, вернёт `[W_lm[0], W_lm[0]]` (без ошибки; caller отвечает за уникальность).
- `token_set=[5]` при `V=3` → `ValueError`.
- Пустой `token_set` → `ValueError`.
- `W_lm.shape[1] != len(g)` → `ValueError`.

**Импорты:** `from typing import Sequence`, torch.

---

### 3.2. `decompose_w_vs_uhat` — новый

**Смысл:** разложение `w = α û + β v_⊥` (лемма [M-1], шаг 1). `v_⊥` — unit-вектор (или ноль при β=0).

**Отношение к существующему / будущему.** **Не** generic-проекция: возвращает кортеж `(α, β, v_⊥)` — коэффициент rank-1, норму остатка и **единичный** residual-вектор. Отличается от `topology.project_to_subspace` (когда появится): (а) работает в `[D]`, а не `[N, D]`; (б) возвращает residual, а не проекцию; (в) residual нормирован. **Не заменяется** `project_to_subspace`, и наоборот.

**Сигнатура:**
```python
def decompose_w_vs_uhat(
    w: torch.Tensor,     # [D]
    uhat: torch.Tensor,  # [D] unit
) -> tuple[float, float, torch.Tensor]:
    """Rank-1 split w = α·uhat + β·v_perp (paper [M-1], step 1).

    Returns (α: float, β: float ≥ 0, v_perp: [D] unit if β > 0 else zeros).

    NOT a generic projection (cf. topology.project_to_subspace): this
    returns a rank-1 coefficient + residual norm + unit residual direction,
    working on a single vector [D]. project_to_subspace returns
    h @ basis.T for [N, D] inputs and does not compute a residual."""
```

**Алгоритм:**
```
α = float(torch.dot(w, uhat))
r = w - α * uhat
β = float(r.norm())
if β > 1e-12:
    v_perp = r / β
else:
    v_perp = torch.zeros_like(w)
return α, β, v_perp
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `uhat` не unit (`|‖uhat‖−1| > 1e-4`) | `ValueError` |
| `w.shape != uhat.shape` | `ValueError` |
| `w` содержит NaN/Inf | `ValueError` |
| `β ≈ 0` | вернуть `v_perp = zeros_like(w)` (не ошибка; caller интерпретирует) |

**Тесты** (`tests/test_jstruct.py`):
- `w = 3·û + 4·v_⊥` (v_⊥ unit, ⊥û) → `(3.0, 4.0, v_⊥)` (tol 1e-6).
- `w = û` → `(1.0, 0.0, 0)`.
- `w = 0` → `(0.0, 0.0, 0)`.
- `w ⊥ û`, `‖w‖ = 5` → `(0.0, 5.0, w/5)`.
- Не-unit `uhat` → `ValueError`.

**Импорты:** torch.

---

### 3.3. `v_pred_from_JT` — новый

**Смысл:** вычислить `v_pred = J^T·u / ‖J^T·u‖` с `u = 1/√K` (лемма [M-1], шаг 2). **Никакого `J^+`** (он подчёркивает хвост, а не gauge).

**Отношение к существующему / будущему.** **Не** generic-проекция: (а) работает по **транспонированной** оси (`J^T·u`, `u ∈ R^K`), а не по `h @ basis.T`; (б) возвращает **нормированный** unit-вектор; (в) при `‖J^T u‖ ≈ 0` бросает `RuntimeError` (это [M-1]-вырождение, а не тривиальный ноль). Не заменяется `topology.project_to_subspace` — тот делает `[N, D] @ [D, r]`, другая ось и другой возврат.

**Сигнатура:**
```python
def v_pred_from_JT(
    J: torch.Tensor,             # [K, D]
    uniform: torch.Tensor | None = None,   # [K] unit, default 1/√K
    eps: float = 1e-10,
) -> torch.Tensor:               # [D] unit
    """Normalized transpose projection v_pred = J^T·u / ‖J^T·u‖, u = 1/√K
    (paper [M-1], step 2).

    NOT a generic projection: transpose-side of J, returns a UNIT vector,
    raises on ‖J^T u‖ ≈ 0 (lemma [M-1] degeneracy). Cf.
    topology.project_to_subspace (h @ basis.T, [N, D] → [N, r])."""
```

**Алгоритм:**
```
K = J.shape[0]
if uniform is None:
    uniform = torch.full((K,), 1.0 / math.sqrt(K), device=J.device, dtype=J.dtype)
else:
    uniform = uniform / uniform.norm()   # normalize silently (documented)
x = J.T @ uniform                # [D]
nx = x.norm()
if nx < eps:
    raise RuntimeError(
        f"v_pred_from_JT: ‖J^T u‖ = {nx:.2e} < {eps:.0e}; "
        "this is the [M-1] degeneracy (β≈0); see lemma"
    )
return x / nx
```

**Краевые условия:**

| Условие | Поведение |
|---|---|
| `J.dim() != 2` | `ValueError` |
| `uniform.shape != (K,)` | `ValueError` |
| `uniform` содержит нулевую норму | `ValueError` |
| `J^T u ≈ 0` | `RuntimeError` с диагностикой ([M-1] degeneracy) |
| `K == 0` | `ValueError` |

**Тесты** (`tests/test_jstruct.py`):
- `J = I_K` (D=K, квадратная identity) → `v_pred = uniform` (tol 1e-10).
- `J` случайная, `uniform = 1/√K`, `D=K` → `‖v_pred‖ = 1`, `v_pred ∝ J^T·uniform` (проверка направления через косинус).
- `J = M_K @ (I − ûû^T)` с random `M_K` и unit `û` → `cos(v_pred, û) ≈ 0` (tol 1e-6) — это [M-1](2) на синтетике.
- `J = 0` (нулевая матрица) → `RuntimeError`.
- `uniform` неверной формы → `ValueError`.
- `uniform` не единичной нормы → нормализуется молча (документировано); результат корректен.

**Импорты:** `import math`, torch.

---

## 4. Тесты (в `tests/`)

Три файла, каждый — после соответствующего модуля.

### 4.1. `tests/test_holonomy_ext.py`
Покрывает: `holonomy_accumulate` (extension), `holonomy_spectrum`, `geodesic_curvature`, `eigenframe_basis`. Стиль — `pytest`, `torch.testing.assert_close` для тензоров, `pytest.approx` для скаляров, `pytest.raises` для исключений, `pytest.warns` для предупреждений.

### 4.2. `tests/test_spectral_ext.py`
Покрывает: `eigengap_min` (9 тестов, перечислены в §2.1). Проверка, что существующие `spectral.py`-тесты (`test_spectral.py`) по-прежнему проходят (не сломаны).

### 4.3. `tests/test_jstruct.py`
Покрывает: `build_M_K`, `decompose_w_vs_uhat`, `v_pred_from_JT`.

**Общие требования к тестам:**
- Детерминированность: `torch.manual_seed(seed)` в начале каждого теста с рандомом.
- Понятные имена: `test_<function>_<edge_case>`.
- Один тест — одна проверка (или плотно связанная группа).
- Тесты быстрые (< 1 сек на тест, всего < 30 сек).

---

## 5. Порядок реализации

1. **`holonomy.py`** — расширение `holonomy_accumulate` (`return_total`).
2. **`tests/test_holonomy_ext.py`** — тесты для расширения.
3. **`holonomy.py`** — `holonomy_spectrum`, `geodesic_curvature`, `eigenframe_basis`.
4. **`tests/test_holonomy_ext.py`** — дописать тесты.
5. **`spectral.py`** — `eigengap_min`.
6. **`tests/test_spectral_ext.py`** — тесты.
7. **`jstruct.py`** — новый файл, три функции.
8. **`tests/test_jstruct.py`** — тесты.
9. **Прогон:** `pytest tests/test_holonomy_ext.py tests/test_spectral_ext.py tests/test_jstruct.py tests/test_spectral.py -v` — все зелёные.

---

## 6. Что НЕ делать

- **Не писать `procrustes_rotation`** — есть `procrustes_align`.
- **Не писать `right_singular_directions`** — есть `top_r_svd`.
- **Не писать `holonomy_matrix`** (композицию одной тройки) — `holonomy_accumulate(..., return_total=True)` даёт `R_total`.
- **Не писать `project_onto_subspace`** — `V @ V.T @ X` пишется в эксперименте; generic `project_to_subspace` появится в `topology.py`, здесь не дублируется.
- **Не реализовывать `eigengap_min` как обёртку над `eigengap`** — это **другая** функция (min по всем зазорам vs первый зазор); импортировать `eigengap` не нужно и не следует.
- **Не путать `eigengap_min(spectrum)` с `eigengap(spectrum)`**: обе принимают eigenvalues (не singular values); первая — pre-flight [M-4] (`min` по всем зазорам, tuple-возврат), вторая — top-1 gap (paper §6, float-возврат). **Не** передавать в `eigengap_min` сингулярные значения (`σ`) — только `σ²` (eigenvalues).
- **Не вызывать `spectrum_from_gramian` внутри `holonomy_spectrum`** — `R_total ∈ SO(r)` не симметрична; нужен `torch.linalg.eigvals` (комплексный), а не `eigvalsh`.
- **Не заменять `eigenframe_basis` на `top_r_svd`** — разные операторы: `P_⊥ M_K^T M_K P_⊥` (симметричная, `D×D`, в `û^⊥`) vs `J` (общая, `K×D`). Вызывающий, которому нужны правые сингулярные векторы `J`, использует `top_r_svd`; `eigenframe_basis` — только для канонического базиса `[D-holonomy-frame]`.
- **Не вводить термин «Wilczek–Zee» / «κ_WZ»** в код — `holonomy_spectrum` возвращает нейтральные `θ_j`.
- **Не кэшировать `build_M_K`** внутри функции — `W_lm` меняется под edit.
- **Не трогать** `interventions.py`, `cache.py`, `esg.py` — использовать как есть.
- **Не менять сигнатуры** существующих публичных функций (`procrustes_align`, `top_r_svd`, `holonomy_along_path`, `holonomy_along_2d_loop`, `eigengap`, `spectrum_from_gramian`, `gauge_share` и т.д.) — только `holonomy_accumulate` расширяется обратно-совместимо.
- **Не добавлять** функции 13–16 (тест 2) — отложены до исхода теста 1b.

---

## 7. Одна фраза

ТЗ — три файла (`holonomy.py` — 3 функции + 1 расширение, `spectral.py` — 1 функция, `jstruct.py` — 3 функции), все тонкие и без новых геометрических объектов; `eigengap_min` принимает **eigenvalues** (`spectrum`), как и существующий `eigengap` (НЕ singular values — снимает риск путаницы σ vs λ = σ²); в docstring каждой функции явно разведены пары, которые читатель может спутать (`eigengap_min` vs `eigengap`, `eigenframe_basis` vs `top_r_svd`, `decompose_w_vs_uhat`/`v_pred_from_JT` vs будущий `project_to_subspace`); три тестовых файла с синтетикой (дуга большого круга → κ_g = 0, малая окружность → κ_g = 1/R, `J = M_K(I−ûû^T)` → `v_pred ⊥ û`, тождества `tr(H) = 2Σcosθ`); порядок — `holonomy` → `spectral` → `jstruct`, затем тесты; moratorium на терминологию WZ и функции теста 2 сохраняется.
```