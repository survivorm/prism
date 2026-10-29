## Короткий ответ

**Отчётные числа и вердикты E-38d (κ, max_theta, ρ(trH, Ω), consistency_pass/redflag, effectively_abelian/non_abelian_candidate/mixed_spectrum, spherical/third_mechanism/partial) на текущем кэше корректны** — код реализует заявленный контракт `holonomy_spectrum` и `procrustes_align` с правильной композицией `H = R01·R12·R20`. Но есть **два «тихих» бага**, которые на текущих данных (κ ∈ [0.016, 3.99]) не проявляются, а на «плоском» H сломают метрику/вердикт, и несколько косметических.

---

## 1. «Тихие» баги, ломающие тривиальный случай

### 1.1. Метрика `trivial` мертва (`np.all(NaN <= 1e-9) = False`)

```python
theta_flat = _spectrum_batched(torch.tensor(Htot))[0]   # [B, R//2]
trivial = np.all(theta_flat <= 1e-9, axis=1)
```

`_spectrum_batched` инициализирует `theta = np.full((B, half), np.nan)` и заполняет только первые `take.size < half` (или `== half`) значения. Для тривиального H ≈ I (`pos.size == 0`) все `theta = NaN`. Тогда:

- `np.all([NaN, NaN, ...] <= 1e-9) = np.all([False, False, ...]) = False`
- `trivial = False` **всегда**, когда `theta` содержит NaN (то есть когда angles < 1e-6 — а это и есть тривиальный случай!)

**Последствия:**
- `trivial_frac = 0` в JSON на всех слоях — систематически **неверно**.
- `theta["median_per_j"]` и `theta["hist_8_angles"]` **исключают** тривиальные триплеты (они NaN → фильтруются `np.isfinite(th_masked)`) — смещение распределения.
- `n_dropped_nan` считает тривиальные триплеты как «потерянные».
- **Вердикт** для полностью тривиального H: `max_theta = np.nanmax([NaN, ..., NaN]) = NaN` → `NaN < 0.1 = False`, `NaN > 0.5 = False` → **`mixed_spectrum`**, хотя должно быть `effectively_abelian`.

**На текущих данных не проявляется:** κ(L23) ≈ 0.016, angles ≈ 0.009 > 1e-6 → `pos.size = half`, `theta_flat` без NaN, `max_theta ≈ 0.009` → `effectively_abelian` **правильно** (но по «случайной» причине — angles чуть выше порога 1e-6). Для истинно тривиального H вердикт будет неверным.

**Фикс:** `trivial = np.all(np.isnan(theta_flat) | (theta_flat <= 1e-9), axis=1)` (или `trivial = n_ang == 0`, где `n_ang` уже возвращается из `_spectrum_batched`).

### 1.2. Одиночный −1 в H может потеряться в `_spectrum_batched`

```python
angs = np.angle(np.linalg.eigvals(Hn[bi]))   # (-π, π]
pos = angs[angs > 1e-6]
```

Для вещественного собственного значения −1 LAPACK может вернуть `−1 + 0j` (angle = **+π**) или `−1 − 0j` (angle = **−π**). Во втором случае `−π > 1e-6` ложно, и угол **отбрасывается**. `holonomy_spectrum` этого не допускает (там `.abs()` → оба ±π → π):

```python
angles = torch.angle(eigvals).abs()   # [0, π]
```

Это ровно тот баг, про который предупреждает docstring `holonomy_spectrum`: **«the old `angles > 1e-6` filter silently dropped −π and could lose an entire π block»**. В `_spectrum_batched` он **воспроизведён**.

**Последствия:** пиковый угол π из lone-−1 или 2D-блока с θ = π может отсутствовать в `theta_flat`. Влияет на `hist_8_angles`, `median_per_j`, `max_theta`, и — если θ = π единственный большой угол — на вердикт (`effectively_abelian` вместо `non_abelian_candidate`).

**На текущих данных не проявляется:** углы ≈ 0 (L23) или ≈ 1.0 (L6), π-блоков нет (Procrustes-полярные факторы близки к I, det ≈ +1, спектр H далёк от π). Но для H с истинным π-блоком (возможен, если какой-то шаг Procrustes — отражение) — тихая потеря угла.

**Фикс:** заменить `pos = angs[angs > 1e-6]` на `pos = np.abs(angs); pos = pos[pos > 1e-6]`, как в `holonomy_spectrum`. Или симметрично обрабатывать ±π.

---

## 2. Разночтения с `holonomy_spectrum` (контракт)

Докстринг `_spectrum_batched` заявляет «mirrors `prism.holonomy.holonomy_spectrum`'s binning contract». Но:

| Аспект | `holonomy_spectrum` | `_spectrum_batched` | Совпадает? |
|---|---|---|---|
| Выборка одного угла из пары | `.abs()` (→ [0, π]) | `> 1e-6` (→ (0, π]) | Для θ ∈ (0, π) да; для π — **нет** (баг 1.2) |
| Тривиальный H → theta | `torch.zeros(expected)` (нули) | `NaN` padding | **нет** (баг 1.1) |
| Предупреждения (odd r, degenerate, non-unit) | есть | нет | диагностика потеряна |
| Повторяющиеся углы (degenerate) | multiplicity сохраняется | `np.repeat` сохраняет | ✅ |

То есть контракт **не полностью** зеркалится, и это причина багов 1.1/1.2.

---

## 3. Мелкие несоответствия (не влияют на результат)

### 3.1. `idiff_max` — идентичность `‖H−I‖² = Σ2(1−cos θ)` верна только для **ортогонального** H

```python
angs = np.angle(np.linalg.eigvals(Htot))
idiff_max = np.abs(kappa_flat**2 - (2*(1 - np.cos(angs))).sum(axis=1)).max()
```

`cos(angle(λ))` = `cos θ`, но `Re(λ) = |λ| cos θ`. `‖H−I‖² = 2r − 2 tr(H) = Σ 2(1 − Re λ) = Σ 2(1 − |λ| cos θ)`. Совпадение с `Σ 2(1 − cos θ)` — **только при `|λ| = 1`**. Для ортогонального H (текущие данные, fp64) — `idiff_max ~ 1e-15`. Для не-ортогонального H (теоретически невозможно, т.к. R из SVD полярный = ортогонален, но численно при fp32-входе в M) — `idiff_max ~ 1e-7..1e-6`. Комментарий «exact identity» **неточен** — это identity для ортогональных H; для текущего fp64-пайплайна ок.

### 3.2. `OUT` (module-level) — мёртвый

```python
OUT = os.path.join(REPO, "results", "exp38d_holonomy_spectrum.json")
# OUT is the LOCAL default; main() overrides it on Kaggle (see OUT_PATH).
```

`main()` использует только `OUT_PATH`, `OUT` нигде не читается. Мёртвая константа.

### 3.3. `results_dir="results"` делает `/kaggle/working` ветку мёртвой

```python
if results_dir:  # "results" — truthy
    out_dir = results_dir
elif os.path.isdir("/kaggle/working"):  # никогда
    out_dir = ...
```

При вызове `main()` (без аргументов) `results_dir = "results"` — truthy → elif недостижим. На Kaggle CWD = `/kaggle/working`, так что `"results"` == `/kaggle/working/results` — эффект тот же. Но priority «explicit > Kaggle > repo» из докстринга **не реализован**: explicit всегда выигрывает, Kaggle не проверяется.

**Фикс:** `if results_dir is None and os.path.isdir("/kaggle/working"): ... elif results_dir is None: ... else: ...` — или явно поменять дефолт на `None`.

### 3.4. `rho_at` docstring: `[n_docs, n_triples]` — на самом деле `[n_triples, n_docs]`

```python
def _tn(flat):
    return flat.reshape(N, n_triples).T   # → [n_triples, N]
```

`trH[t, m]` в цикле `for t in range(n_triples)` — корректно по коду, но docstring `_tn`/`rho_at` вводит в заблуждение (пишет «doc=axis0, position=axis1», а фактически position=axis0, doc=axis1). Косметика, но при чтении кода легко ошибиться.

### 3.5. `n_ang` (возврат `_spectrum_batched`) не используется

`theta_flat = _spectrum_batched(...)[0]` — второй элемент отброшен. `n_ang` мог бы зафиксировать trivial/не-trivial (см. фикс 1.1). Мёртвый возврат.

### 3.6. `torch.tensor(Htot)` — лишний round-trip

`_spectrum_batched` внутри делает `Hn = H.numpy()`. `Htot` уже numpy — конвертация в torch избыточна (не баг, efficiency).

### 3.7. `median_per_j` при полностью NaN-слайсе — warning + NaN

`np.nanmedian` на all-NaN слайсе выдаёт RuntimeWarning и NaN. В JSON сериализуется как `"nan"` (контракт). Докстринг это признаёт («median is the string 'nan'»). OK, но при большом r (128) большинство слайсов пусто → много warning-ов в лог.

---

## 4. Что проверено и **корректно**

| Аспект | Статус |
|---|---|
| `_procrustes_batched(V_a, V_b)` = `V_a @ R ≈ V_b` — та же конвенция, что `prism.holonomy.procrustes_align` | ✅ |
| `H = R01 @ R12 @ R20` — правильная циклическая композиция (right-multiply) | ✅ |
| `einsum("bij,bjk,bkl->bil", R01, R12, R20)` = `R01 @ R12 @ R20` | ✅ |
| `residual_sq = 2r − 2Σ s_i` — та же формула, что `procrustes_align` | ✅ |
| `f0 = i_doc * Ss + t_pos` — корректная плоская индексация `[N*S]` | ✅ |
| `V0, V1, V2 = V[f0], V[f0+1], V[f0+2]` — позиции `t, t+1, t+2` внутри документа | ✅ |
| `chunk` sizing — под бюджет T4 16 GB (проверено: R=16 → chunk=N; R=128 → chunk=5) | ✅ |
| `deg_h` / `deg_arc` — комплексная маска дегенерации Ω (по нормам и дугам) | ✅ |
| `omega_all = ... .T` → `[T, N]` — согласовано с `_tn(...)` | ✅ |
| Вердикты `effectively_abelian / mixed_spectrum / non_abelian_candidate` (пороги 0.1/0.5) | ✅ |
| `spherical_geometry / third_mechanism / partial` (пороги 0.7/0.3) | ✅ |
| `consistency_e28`: 5e-3 = pre-registered, 1e-3 = diagnostic | ✅ |
| `degen_sweep` (0.7/0.9/0.95 × 2r) — проверка артефакта порога | ✅ |
| `H_012_doc0` npz: `f0 = arange(n_triples)` = doc 0 (правка v2: было `f0 = Ss + t` = doc 1) | ✅ |
| `Vt` жив до npz-дампа (не `del Vt` до `if l == 23`) | ✅ |
| `identity_check_max_absdiff ≈ 1e-15` для ортогонального H (текущие данные) | ✅ |
| Детерминизм (`np.random.seed(SEED)`, `torch` без рандома) | ✅ |
| JSON strict + atomic write | ✅ |
| Профиль `profile_kappa` использует `.median()` — kappa без NaN, приемлемо | ✅ (мелочь: не `.nanmedian()`) |

---

## 5. Сверка с `paper.md` §4.4 / `ref-exp.md` E-38d

Paper:
> «κ (медиана): 3.99 / 3.48 / 2.39 / 2.24 / 1.64 / 0.0162 (L6→L23); ρ_median(trH, Ω): 0.484 / 0.347 / 0.241 / 0.344 / 0.251 / 0.358; degenerate_frac = 0.0 на всех слоях; L18/L22 = non_abelian_candidate, third_mechanism; L6/L12/L19 = non_abelian_candidate, partial; L23 = consistency_pass, effectively_abelian, partial».

Code выдаёт именно это (проверено по логике вердиктов). ✅

Единственное, чего нет в paper: `trivial_frac`. В JSON оно есть, но систематически 0 из-за бага 1.1. Если paper на него ссылается — цифра неверна. Если нет — баг латентный.

`anomaly_count 1397–4032` в paper — **не соответствует** никакому полю в JSON `exp38d`. Возможно, из более ранней версии. Стоит проверить источник.

---

## 6. Итоговая таблица

| Аспект | Статус |
|---|---|
| Отчётные κ, ρ, вердикты на текущих данных | ✅ Корректны |
| Заявленный контракт `holonomy_spectrum` — соблюдён | ⚠️ Не полностью (баги 1.1, 1.2) |
| `trivial` как метрика | ❌ Мёртвая (NaN padding) |
| Отбрасывание lone −1 (angle −π) | ⚠️ Латентный баг |
| `max_theta = NaN` для тривиального H | ⚠️ → `mixed_spectrum` вместо `effectively_abelian` |
| `idiff_max` — «exact identity» | ⚠️ Только для ортогонального H (текущие — да) |
| `OUT` unused, `results_dir` dead elif | ⚠️ Косметика |
| `rho_at` docstring shape | ⚠️ Косметика (неверно) |
| Композиция H, Procrustes, einsum, npz, indexing | ✅ |
| Вердикты и пороги | ✅ |

---

## 7. Рекомендации перед следующим прогоном

1. **`_spectrum_batched`**: заменить `pos = angs[angs > 1e-6]` на `pos = np.abs(angs); pos = pos[pos > 1e-6]` — убрать latent bug с −π (баг 1.2).
2. **`trivial`**: `trivial = np.all(np.isnan(theta_flat) | (theta_flat <= 1e-9), axis=1)` — иначе метрика мертва (баг 1.1). Или использовать `n_ang == 0` (уже возвращается).
3. **`max_theta`**: для all-NaN `theta_flat` вернуть `0.0` вместо `NaN` (сохранив `n_ang == 0` как индикатор тривиальности) — чтобы вердикт тривиального H был `effectively_abelian`.
4. **`_spectrum_batched`**: для `pos.size == 0` возвращать `zeros` вместо NaN — совпадёт с контрактом `holonomy_spectrum` (тогда фикс 2 не нужен).
5. **`idiff_max`**: обновить комментарий — «valid for orthogonal H (|λ| = 1); for non-orthogonal H it measures the orthogonality drift `Σ cos θ (|λ|−1)`» — или использовать `1 − Re(λ)` вместо `1 − cos(angle λ)` для полной точности.
6. **`results_dir`**: дефолт `None` (явно передавать `"results"` из `if __name__ == "__main__"`), чтобы Kaggle-ветка стала достижимой.
7. **`OUT`**: убрать (мёртвая константа).
8. **`rho_at` docstring**: `[n_triples, n_docs]`, не `[n_docs, n_triples]`.
9. **`paper.md`**: проверить происхождение `anomaly_count 1397–4032` — такого поля в JSON нет.
10. **`trivial_frac` в JSON**: после фикса 2 убедиться, что значение совпадает с paper (если paper на него ссылается).

## Короткий ответ

**Основные числа E-38e на текущем кэше корректны и соответствуют `paper.md`** (α/β ≈ 0.196, β/σ₁ ≈ 0.882, r_t/√D ≈ 4.077, over_sigma_g_OLD ≈ 0.216, gate=`mixed`, cons=`consistent_with_E33r`). Логика `[L-gauge-J]` (β/σ₁ ≈ cos(u_g, 1/√K)), factorization cross-check (β vs ‖J^T 1/√K‖) и трёхуровневая разметка (gate/cons/gate_failed) реализованы правильно. **Критических багов нет.**

Но есть **несколько «тихих» путей**, которые на повреждённом кэше или в граничных случаях могут исказить отчёт:

---

## 1. «Тихие» баги (латентные, на текущем кэше не проявляются)

### 1.1. Нет upfront-проверки `torch.isfinite(H).all()` / `torch.isfinite(J23).all()`

Если в кэше есть Inf: `norms = H.norm(dim=-1)` даёт Inf, `valid = Inf >= 1e-6` → True; `u = H / Inf` — Inf/Inf = NaN или конечные/Inf = 0. Дальше:
- `w2 = u @ w` — 0 или NaN
- `beta_vec = ‖w − w2·u‖` — либо ‖w‖ ≈ 15, либо NaN
- `resid = ‖J @ u‖ / ‖J‖` — 0/Inf или NaN

**Последствия:** `cons_ratio`, `jtu_ratio`, `ratio_median` могут стать NaN (→ `undefined` или silent-redflag) или получить ложные значения, при том что `nan_frac = 0` (norms=Inf ≥ 1e-6 → valid=True). Это баг маскировки: **`nan_frac` не отловит Inf**. Docstring говорит «fail loud вместо тихой деградации», но этого нет.

**Фикс:** `if not torch.isfinite(H).all(): raise ValueError(...)` и аналогично для `J23`.

### 1.2. `ratio_median` не капнут; капнут только `pos0_ratio`

```python
POS0_CAP = 1e6
pos0_ratio_capped = bool(np.isfinite(pos0_ratio) and pos0_ratio > POS0_CAP)
if pos0_ratio_capped:
    pos0_ratio = float(POS0_CAP)
```

Но общий `ratio_median`:
```python
ratio_median = float(np.nanmedian(ratios_docs))  # без капа
```

`ratios_docs[i] = nanmedian(ratio[i])` — per-doc median over S positions. Если в одном доке >S/2 позиций с β ≈ 0 (например, û ∥ w), per-doc median → huge (≈ |α|/1e-12 ≈ 1e15). Тогда `ratio_median = median(huge, 0.2, 0.2, ...)` — median robust для 1 bad doc (31 good → median ≈ 0.2). Но если ≥ 17 из 32 docs имеют β ≈ 0 в >64 позиций, `ratio_median` → huge → `gate_failed` (E-38f скипнут). Not corruption per se, но **false negative** (E-38f зря пропускается).

Docstring упоминает эту возможность («v_1(M_K) ∥ û») но кап не применяет. На текущем кэше β ≈ 12.7, catastrofa не случается.

**Фикс:** симметрично капнуть `ratio_median` (или капнуть `ratio` до `ratio.clamp(max=POS0_CAP)` перед агрегацией).

### 1.3. `r_t_expected` (all positions) vs `jtu_ratio` (valid positions) — разные наборы

```python
r_t_expected = float(np.nanmedian(norms.numpy()) / sqrtD)  # включает invalid (нормы ~1e-15)
jtu_ratio = float(np.nanmedian(beta_vs_JtU.numpy()))       # median over all positions; NaN в invalid
```

`norms` содержит нормы на ВСЕХ (doc, t), включая дегенеративные (~1e-15). `nanmedian` их не отбрасывает (они конечны). Так что `r_t_expected` = median(‖h‖)/√D по всем позициям. `jtu_ratio` — median по valid только. Если fraction дегенератов велика, значения могут слегка расходиться. На текущем кэше нет дегенератов — расхождение ~0.

**Фикс:** `r_t_expected = float(np.nanmedian(norms[valid].numpy()) / sqrtD)` — один и тот же набор.

### 1.4. `beta_scaled` использует `norms.clamp_min(1e-12)`, а не valid-маску

```python
beta_scaled = (sqrtD * beta_vec / norms.clamp_min(1e-12)).detach()
```

`beta_vec` = NaN на invalid (via `torch.where`). `NaN / 1e-12 = NaN`. `nanmedian` отбрасывает — OK. Но `norms.clamp_min` на valid позиции с ‖h‖ = 5e-7 (между 1e-12 и 1e-6) даёт 5e-7 (не клипает), и u там = корректный unit vector. Но `valid = norms >= 1e-6` — такие позиции = invalid. Так что beta_vec = NaN там. `nanmedian` отбрасывает. Good.

Но — если ‖h‖ = 5e-13 (между 1e-12 и 1e-6), clamp_min(1e-12) = 1e-12. `u = H / 1e-12` — если ‖H‖ = 5e-13, то u ≈ H · 2e12 — норм не 1, а ≈ 2e12 · 5e-13 = 1 — всё-таки unit? Нет: `‖u‖ = ‖H‖/1e-12 = 5e-13/1e-12 = 0.5`. Не unit. Но `valid = False` → masked. OK.

**Единственная проблема:** если `norms` = 0 точно (clamp до 1e-12), `u = 0/1e-12 = 0`. `w2 = 0 @ w = 0`. `valid = False` → masked. OK.

### 1.5. `pos0_ratio` при numel=1 — no-op, но `median()` на 1 элементе работает. Guard `if pos0_vals.numel() else NaN` — если 1 элемент, обрабатывается. OK.

---

## 2. Семантические несоответствия (не портят числа, но вводят в заблуждение)

### 2.1. `paper.md` говорит «α/β = 0.196», код — `|α|/|β|`

```python
"ratio_alpha_beta": {
    "median_over_docs": ratio_median,
```

`ratio_median = median over docs of median_t(|α|/β)` — **unsigned**. Paper пишет «α/β = 0.196» без модуля. Если α отрицателен в части позиций, `median(α/β)` ≠ `median(|α|/β)`. На текущем кэше совпадает (0.196 в обоих случаях), значит α > 0 в медианном случае. Но docstring и paper должны быть согласованы: либо переименовать в `ratio_abs` или явно написать в paper.

**Влияние:** нет на текущих данных; при смене кэша или слоя (где α часто отрицателен) результаты paper и кода разойдутся.

### 2.2. `nan_frac > 0.5 → gate = "degenerate"` перезаписывает `gate_failed`/`gauge_blind_sep`/`mixed`

```python
if nan_frac > 0.5:
    gate = "degenerate"
```

Если `nan_frac = 0.6` и одновременно `ratio_median = 15` (→ было бы `gate_failed`), мы видим только `degenerate`. Информация о том, что было бы в нормальном случае, потеряна. Для E-38f всё равно скип (`gate_failed in (gate_failed, degenerate)` — оба ведут к скипу). Но диагностически было бы полезно сохранить оба состояния.

### 2.3. `consistency_undefined` vs `consistency_redflag` — обе возможны при `nan_frac ∈ (0, 0.5]`

```python
cons = ("consistency_consistent_with_E33r" if abs(cons_ratio - 0.9) <= TOL_CONSISTENCY
        else ("consistency_undefined" if not np.isfinite(cons_ratio)
              else "consistency_redflag"))
```

Если `nan_frac = 0.4` и `cons_ratio = 0.5` (далеко от 0.9), это `consistency_redflag` — семантически «модель плохая/структура сломана». Но docstring отмечает: «redflag one-sided — fires also when v_1(M_K) ∥ û». Так что `redflag` при `cos << 0.9` — это не обязательно bad model, это может быть просто «alignment factor ≠ 0.9» (model-specific). Мета говорит: «redflag = 'the check did not confirm', not 'model is bad'». Правильно, но `paper.md` §4.4 может читаться как «redflag = проблема». На текущих данных `cons = consistent`, не redflag. OK, но семантически это ловушка.

### 2.4. `pos0_median` vs `pos0_ci` — нет такой проблемы в E-38e (это было в E-38b для C_b). Skip.

### 2.5. `ci_note` говорит «the old flat percentile over N·S positions overstated precision ~√(S)×». Код использует per-doc medians → правильно.

---

## 3. Что проверено и **корректно**

| Аспект | Статус |
|---|---|
| `w = M_K^T @ (1/√K)` — правильное определение w | ✅ |
| `α = u @ w`, `β = ‖w − αû‖` — rank-1 разложение | ✅ |
| `cons_ratio = median(β)/σ₁(M_K) ≈ cos(u_g, 1/√K) ≈ 0.9` — правильная нормализация (после P0-фикса) | ✅ |
| `over_sigma_g_OLD = median(√D·β/‖h‖)/σ₁` — правильная диагностика «старой» ошибки | ✅ |
| `beta_vs_JtU = β / ‖J^T(1/√K)‖ ≈ r_t/√D` — правильный cross-check факторизации | ✅ |
| `r_t_expected = median(‖h‖)/√D` — правильно (но minor см. 1.3) | ✅ |
| Gate-пороги (10, 0.1) — соответствуют docstring | ✅ |
| `consistency_undefined` guard при NaN — P2-фикс применён | ✅ |
| `pos0_valid` / `pos0_ratio_capped` / `pos0_degenerate_frac` — guards | ✅ |
| `bootstrap_ci` на finite per-doc medians | ✅ |
| `del M_K, H, J23, ck23` — правильный порядок освобождения памяти | ✅ |
| `del u, w2, beta_vec, JtU, ...` — освобождение big tensors | ✅ |
| `j_uhat_resid = ‖J·û‖/‖J‖` — правильный диагностический кросс-чек û-as-input | ✅ |
| `g = norm.weight + 1` — правильно для RMSNorm Qwen3.5 (paper §3.1) | ✅ |
| `build_M_K` — W_lm[ts] · g, правильная размерность и dtype | ✅ |
| `_resolve_lm_head_weight` — fallback на `get_output_embeddings()` | ✅ |
| `_lm_module` — flat/wrapper обрабатываются | ✅ |
| JSON strict (sanitize + allow_nan=False + atomic write) | ✅ |
| Детерминизм (`torch.manual_seed` без рандома в этом скрипте) | ✅ |

---

## 4. Сверка с `paper.md` E-38e

| Paper | Code ожидаемое | Совпадает |
|---|---|---|
| σ_g = 14.347 | `svdvals(M_K)[0]` | ✅ (структурно) |
| α/β = 0.196, CI [0.143, 0.232] | `ratio_median`, `ci` | ✅ (с оговоркой 2.1 про abs) |
| β (медиана) = 12.656 | `beta.median` | ✅ |
| β/σ₁ = 0.882 | `cons_ratio` | ✅ |
| r_t/√D = 4.077 | `r_t_expected` | ✅ |
| over_sigma_g_OLD = 0.216 | `over_sigma_g_OLD` | ✅ |
| gate = mixed, gate_failed = false | `gate = "mixed"` | ✅ |
| consistency = consistent_with_E33r | `cons_ratio = 0.882 ∈ [0.75, 1.05]` | ✅ |

`paper.md` соответствует коду. Никаких расхождений, кроме 2.1 (α/β vs |α|/|β|).

---

## 5. Модули `jstruct.py` / `interventions.py` / `cache.py` / `jacobian.py`

Прошёлся по используемым функциям — **корректны**:

- `build_M_K` — правильная индексация `W_lm[ts]`, cast `g` к `W_lm.dtype`, проверки NaN/range/empty. OK.
- `decompose_w_vs_uhat` — E-38e **не использует** (векторизовал inline). Но если бы использовал — функция корректна: требует unit `uhat`, возвращает `(α: float, β: float ≥ 0, v_⊥: [D] unit)`. OK.
- `_resolve_lm_head_weight` — OK.
- `_lm_module` — OK (flat + wrapper).
- `bootstrap_ci` — checked earlier, OK.
- `json_dump` — checked earlier, OK.

Никаких критических багов в поддерживающих модулях, затрагивающих E-38e.

---

## 6. Итоговая таблица

| Аспект | Статус |
|---|---|
| Отчётные числа воспроизводятся (paper.md: 0.196, 0.882, 4.077, 0.216) | ✅ |
| Критические баги, портящие числа на текущем кэше | ❌ Нет |
| Логика gate/cons/gate_failed | ✅ |
| Consistency check после P0-фикса (β/σ₁, не cβ/σ₁) | ✅ |
| Factorization cross-check (β vs ‖J^T u‖) | ✅ |
| Inf в H/J23 (нет isfinite guard) | ⚠️ Латентный |
| `ratio_median` не капнут (pos0 — капнут) | ⚠️ Латентный |
| `r_t_expected` (all) vs `jtu_ratio` (valid) — разные наборы | ⚠️ Minor |
| Paper "α/β" vs code "\|α\|/\|β\|" | ⚠️ Документационное |
| `nan_frac > 0.5 → gate = "degenerate"` перезаписывает | ⚠️ Диагностическое (не влияет на skip) |
| `consistency_redflag` может false-fire при cos ≠ 0.9 | ⚠️ Семантика (модель-специфично) |
| Детерминизм, atomic write, strict JSON | ✅ |

---

## 7. Рекомендации перед следующим прогоном

1. **`torch.isfinite(H).all()` / `torch.isfinite(J23).all()` upfront** — fail loud на corrupt cache (баг 1.1).
2. **Капнуть общий `ratio`** до `POS0_CAP` перед агрегацией (баг 1.2) — или симметрично капнуть `ratio_median` и записать `ratio_median_capped` в JSON.
3. **`r_t_expected = nanmedian(norms[valid].numpy()) / sqrtD`** — единый набор (баг 1.3).
4. **Переименовать `ratio_alpha_beta.median_over_docs` в `ratio_abs_over_abs`** или задокументировать в `meta`, что это модуль отношения (баг 2.1).
5. **Сохранять оба gate-значения** при `nan_frac > 0.5` (например, `gate_raw` и `gate`), чтобы диагностика не терялась (баг 2.2).
6. **В `meta.redflag_semantics`** — уже правильно, что redflag ≠ bad model. Но `paper.md` §4.4 стоит перепроверить — нет ли интерпретации «redflag = проблема».

**Ни одна из рекомендаций не портит текущие числа.** Всё — про будущую устойчивость на повреждённых кэшах и на других моделях/слоях.

## Короткий ответ

**Числовые результаты на текущем кэше воспроизводятся** (defect_top ≈ 3e-16, defect_tail 0.14–0.27, energy_top 0.735→0.280, energy_top(ΔJ) 0.717→0.487, P0-фикс композиции `R23 @ R12` применён). Но есть **одна серьёзная семантическая проблема**, из-за которой вывод «top абелев, tail неабелев» — частично **тавтология**, и несколько латентных/косметических.

---

## 1. Главное: `defect(top) = 0` гарантирован конструкцией, а не измерен

Это ключевое.

При фиксированных `P_L, P_R` (обе — проекторы ранга 16):
```python
A_top(s) = P_L · dW[s] · P_R = U_ref · M(s) · V_ref^T,   M(s) [16×16]
```
где `M(s) = U_ref^T · dW[s] · V_ref`.

SVD: `M(s) = P · σ(s) · Q^T`. Тогда
```
A_top(s) = U_ref P σ (V_ref Q)^T
```
Правые сингулярные векторы `A_top(s)` = `V_ref · Q`. То есть **span(V_ref Q) = span(V_ref)** для любого `s` и любого Q. **Правый top-16 фрейм `A_top(s)` — константа `span(V_ref)`** для всех s.

Следствие: `procrustes_align(Q_top(a), Q_top(b))` возвращает **чистую смену базиса** 16×16, и композиция `R23·R12` = `R13` **точно** (с точностью до машинного эпсилон). Отсюда `defect(top) ≈ 3e-16` — не измерение «абелевости топ-канала», а **математическое тождество**, заложенное конструкцией фиксированного проектора.

Аналогично для левого фрейма `U_top(s) = U_ref · P`.

**Что это значит:**

- Paper пишет: «top строго абелев, tail неабелев; старый вывод "оба Abelian" опровергнут» — как **находку**. Но top-абелевость — **тавтология** при данном определении канала.
- «Неабелев в top-16» (сценарий 2 из E-38d) в этой постановке **невозможен** — не потому что данных нет, а потому что фиксированный проектор делает фрейм постоянным.
- Настоящий вопрос «является ли top-16 подпространство **непроецированного** ΔW неабелевым?» — **не задан этим экспериментом**. Он измеряет другое: "стабилен ли фрейм проецированного канала", что тривиально.

**Что paper/E-38d действительно мог бы утверждать:** «tail-канал (остаток после фиксированного проектора) неабелев (defect 0.14–0.27), тогда как top-канал тривиально абелев **по определению фиксированного проектора**». Это честная формулировка. Текущая — вводит в заблуждение.

---

## 2. Латентные баги

### 2.1. Семантика `gamma` в `Jnorm` не проверена

```python
Jnorm = (D ** 0.5 / r0) * (torch.eye(D) - torch.outer(u0, u0)) * gamma[:, None]
```

`Jnorm = diag(gamma) · (I − ûû^T) · (√D/r_t)`. Для `∂h_norm/∂h` (RMSNorm) правильно `diag(1+w) · (I − ûû^T) · (1/rms)` с `rms = ‖h‖/√D`. Значит `gamma` должен быть `1 + w`, а не `w`. Я не вижу `load_model_tensors()` — если он возвращает сырой `w` (обычно ≈ 0 в Qwen3.5 на старте), то `Jnorm ≈ 0`, `Jt = A @ Jnorm ≈ 0`, `energy_top(ΔJ) = 0/0 = NaN`.

Paper даёт 0.717→0.487 — не NaN. Значит **либо** `gamma = 1+w`, **либо** результат посчитан в старой версии. Нужно проверить `load_model_tensors()` / `exp38_hi_common.py`.

### 2.2. `top_r_subspaces` возвращает V или U — важно для семантики полей JSON

Комментарии:
- `Qt = top_r_subspaces(A_top, R)` → «right frame» (hidden, [1024, r])
- `Ut = top_r_subspaces(A_top.t(), R)` → «left frame» (token, [256, r])

Это соответствует возврату **V** (правые сингулярные векторы). Если функция возвращает **U** (левые), комментарии и JSON-поля `right_hidden` / `left_token` будут **инвертированы** (правая сторона станет левой). Defect-вычисления остаются корректными (обе стороны симметричны), но семантика имён — нет.

Нужно проверить `exp38_hi_common.top_r_subspaces`.

### 2.3. `A_tail(s=10)` близок к вырождению

`energy_top(10) = 0.735` → `A_tail(10)` несёт ~0.265 энергии. `rank(A_tail(10))` может быть < 16, и тогда часть правых сингулярных векторов — произвольный LAPACK-basis нуль-пространства. `procrustes_align` не упадёт (SVD-ортогонализация), но фиксированные столбцы могут вращаться между `s`, внося шум в tail-defect.

Paper даёт стабильные 0.14–0.27 — значит **на текущих данных** rank ≥ 16 (или вращение мало). Но **проверки `rank(A_tail) == 16` нет**. Латентная хрупкость.

### 2.4. `e_top` и `jresp[k]["energy_top"]` без guard на ноль

```python
e_top = float((A_top.pow(2).sum() / A.pow(2).sum()))
```

Если `A = 0` → `0/0 = NaN` (молча). `dW[10]` ненулевой — не проявляется. Но defensive `if A.pow(2).sum() > 0` или `clamp_min` не помешал бы.

### 2.5. `defect()` принимает `side`, но не использует

```python
def defect(framedict, a, b, c, side):
```
`side` в теле не фигурирует. Косметика.

### 2.6. `raw` из `load_snapshots()` загружается, но не используется

```python
snaps_all, raw = load_snapshots()
```
По комментарию — только schema-check внутри `load_snapshots`. Но держит ~1 MB (`W_orig` [256,1024] fp32) без нужды. Косметика.

---

## 3. Что проверено и **корректно**

| Аспект | Статус |
|---|---|
| `prod = R23 @ R12` (P0-фикс композиции) | ✅ |
| Defect = `‖R23 R12 − R13‖_F / r` | ✅ |
| Правый/левый фрейм, обе стороны | ✅ (если `top_r_subspaces` возвращает V) |
| `P_L = U_t U_t^T`, `P_R = V_t V_t^T` из `dW[10]` | ✅ |
| 3 triples как в pre-registration | ✅ |
| `energy_top(ΔW) ∈ [0, 1]` (ортогональные P_L, P_R) | ✅ |
| `energy_top(ΔJ)` — honest diagnostic note (может быть > 1) | ✅ |
| Deviation от TZ (fixed projector vs J23) — документирован | ✅ |
| `tied_weights` оговорка (ΔJ = ΔW @ J_norm — приближение) | ✅ |
| BLOCKED (per-channel bridge, top-p retention) | ✅ |
| Atomic write + strict JSON | ✅ |
| Matches paper (defect, energy_top) | ✅ |

---

## 4. Сверка с `paper.md` §4.3 / E-38h

| Paper | Code | Match |
|---|---|---|
| defect top 3.1e-16/3.5e-16/3.1e-16 | `defects['top']['right_hidden']` | ✅ (если `top_r_subspaces` = V) |
| defect tail 0.253/0.144/0.174 (right) | `defects['tail']['right_hidden']` | ✅ |
| defect left/token top 4.2e-16, tail 0.268/0.152/0.175 | `defects[side]['left_token']` | ✅ |
| energy_top(ΔW) 0.735→0.280 | `per[k]['energy_top']` | ✅ |
| energy_top(ΔJ) 0.717→0.487 | `jresp[k]['energy_top']` | ✅ |
| «top строго абелев, tail неабелев» | ✅ (значения, но см. §1) | ⚠️ семантика |

---

## 5. Итоговая таблица

| Аспект | Статус |
|---|---|
| **Tautology в "top-абелевости"** (фиксированный P_L, P_R → константный фрейм) | ⚠️ **Серьёзно** — paper может утверждать больше, чем измерено |
| `gamma = 1+w` vs `w` в Jnorm | ⚠️ Нужно проверить |
| `top_r_subspaces` возврат V vs U → invert JSON labels? | ⚠️ Нужно проверить |
| `A_tail(s=10)` rank < 16 → unstable tail defect | ⚠️ Латентный |
| `e_top`, `energy_top(ΔJ)` без zero-guard | ⚠️ Косметика |
| `defect(side)` unused param | ⚠️ Косметика |
| `raw` unused | ⚠️ Косметика |
| Композиция, формула defect, проектор, JSON | ✅ |

---

## 6. Рекомендации

1. **В `paper.md` §4.3 и `E38.md` (E-38h) явно указать**: `defect(top) ≈ 0` — **следствие фиксированного проектора**, а не свидетельство «top-канал абелев по данным». Настоящий вопрос (неабелевость top-16 **непроецированного** ΔW) требует отдельного эксперимента с `A_top(s) = top_r(dW[s])` (без фиксированного P).
2. **Проверить `load_model_tensors()`**: возвращает ли `gamma = 1 + w` (правильно для Qwen3.5 RMSNorm) или сырой `w`. Добавить assert: `gamma.min() > 0` (если `1+w` с `w` инициализированным в 0, `1+0 = 1 > 0`; если `w` инициализирован в 0, `w.min() = 0` — не отловит). Лучше: `assert gamma.mean() ≈ 1` (для Qwen3.5 на старте) — или явно документировать `gamma = 1+w`.
3. **Проверить `exp38_hi_common.top_r_subspaces`**: возвращает V или U? Если V — комментарии правильные. Если U — исправить комментарии, JSON-поля `right_hidden` / `left_token` переименовать/поменять местами.
4. **Добавить `rank(A_top(s))`, `rank(A_tail(s))` в JSON** — для аудита стабильности фреймов при малых дозах.
5. **Zero-guard для `e_top` и `jresp["energy_top"]`** — `if A.pow(2).sum() > 0 else NaN` с явной записью.
6. **Убрать неиспользуемые `side` и `raw`** — cosmetic.
7. **Добавить `meta.tautology_warning`** — как явное предупреждение потребителю JSON.

## Короткий ответ

**Основные числа и вердикты на текущем кэше воспроизводятся** (overlap = 0.0596, random = 0.0996, ρ_our = −0.041, ρ_lens = −0.927, lens_valid = false, verdict = `at_random,lens_validity_suspect`, tz = `jlens_not_tail_modes`). P0-фикс (удаление null completion, primary = overlap) применён, MEDIUM-4/7 (doc-level bootstrap, guard на `size ≥ 2`) реализованы.

Но есть **одно серьёзное семантическое несоответствие** (метка `jlens_not_tail_modes` при ρ_lens = −0.927), **одно расхождение paper ↔ code** (`completion_sensitive`, `rho_our_random_qr`), и несколько косметических/латентных.

---

## 1. Серьёзное: `jlens_not_tail_modes` при ρ_lens = −0.927

Код:
```python
tz_preregistered_outcome = (
    "jlens_not_tail_modes" if rho_our_med < 0.3 and rho_lens_med < 0.3
    else ("symmetric" if min(rho_our_med, rho_lens_med) > 0.7 else "partial"))
```

При `rho_lens_med = −0.927`:
- `−0.927 < 0.3` = True
- Значит, `jlens_not_tail_modes` срабатывает

Paper (`E38.md`):
> «ρ>0.7 symmetric / ρ<0.3 not_tail_modes»

То есть **pre-registered правило** `ρ < 0.3 = not_tail_modes`. Код ему следует. Но семантически `ρ_lens = −0.927` означает **сильную отрицательную корреляцию** `S(t)` vs `frac_tail(t)`. При концентрированном распределении (`S` низкое) `frac_tail` высокое — то есть **энергия концентрируется в хвосте**, что и есть **tail-sparsity**. Это **прямо противоположно** `not_tail_modes`.

Paper сам признаёт проблему в нарративе:
> «dual-сигнал **односторонний**: ρ_our не значимо (CI через 0) при ρ_lens = −0.927»

Но метка `jlens_not_tail_modes` (буквально «не в tail-модах») этому нарративу противоречит. Метка говорит «оба не в tail»; факт говорит «lens сильно в tail, наш — нет».

**Последствия:**
- В `paper.md` §4.3 / E-38g это читается как «симметрия с Anthropic не подтверждена, потому что **оба** оператора не имеют tail-структуры» — это неверная интерпретация `ρ_lens = −0.927`.
- Правильная интерпретация: «ρ_lens = −0.927 (сильная tail-sparsity у lens), ρ_our ≈ 0 (нет tail-sparsity у нашего J), **асимметрия** — у lens есть tail-структура, у нашего нет».

**На числах это не баг** (код честно применяет pre-registered критерий `< 0.3`), но **метка семантически некорректна** для отрицательных ρ. Это повторяет паттерн из E-38c, где метка `well_separated` ↔ `не применим` инвертирована относительно определения.

**Фикс:** либо уточнить pre-registration (`|ρ| < 0.3 = not_tail_modes`, `ρ < −0.3 = tail_sparse`, `ρ > 0.3 = anti_tail_sparse`), либо переименовать метку в `rho_our_low_rho_lens_strong_negative` (описательно) и записать интерпретацию в `outcomes.outcome_note`.

---

## 2. Расхождение paper ↔ code: `completion_sensitive`, `rho_our_random_qr`

`ref-exp.md` / `E38.md`:
> «ρ_our(S, frac_tail) = −0.041, CI [−0.120, +0.009] (не значимо; **sensitivity к completion: −0.166 при random-QR — completion_sensitive = false**)»

В коде / JSON:
- **Нет** поля `completion_sensitive`
- **Нет** поля `rho_our_random_qr`
- Код **не вычисляет** random-QR completion (P0-фикс удалил null completion)

P0-фикс (docs/E38_fixes2.md) явно удалил null completion как «делающий метрику вырожденной». Логично: если null completion удалён, то и sensitivity-чек на completion теряет смысл. **Но paper не обновили** — он ссылается на поля, которых больше нет.

**Последствия:** потребитель JSON ищет `completion_sensitive` → `KeyError`. В `paper.md` §4.3 эта строка читается как подтверждённый результат, но в коде её нет.

**Фикс:** либо восстановить проверку (например, сравнить ρ_our при двух случайных QR-дополнениях до 1024 — если |Δρ_our| < 0.05, `completion_sensitive: false`), либо удалить из paper упоминание.

---

## 3. Латентные / косметические

### 3.1. Нет `torch.isfinite(H).all()` / `torch.isfinite(J_our).all()` upfront
Тот же класс, что в E-38e:
- `H` с Inf → `c2_o = (H @ V_our)**2` с Inf → `p = Inf/Inf = NaN` → `spearmanr` = NaN → guard `np.isfinite(r_o)` отбросит → `rho_our` медиана по меньшему набору.
- `c2_l` (lens) аналогично.
- `overlap_t` содержит NaN — `nanmedian` отбросит, но `overlap_med` по меньшему набору.

Fail-loud отсутствует. Не проявляется на текущем кэше (H finite), но при повреждении — тихое занижение.

### 3.2. `overlap_t = np.empty((N, Ss))` вместо `np.full(..., np.nan)`
Заполняется в цикле по всем `(i, t)`, но `np.empty` при пропуске (крайне маловероятно) даст мусор, а не NaN. Defensive — `np.full`.

### 3.3. `S = 128` импортирован из `_e38_cache`, но не используется
`N, Ss, Kk, _ = J_our.shape` — `Ss` берётся из данных. `S` — мёртвый импорт. Косметика.

### 3.4. `load_layer(T_LAYER)` — default `want=("J", "P", "H")`, но `P` не используется
`want=("J", "H")` сэкономил бы память (`P` — [32, 128, 128] fp32 ≈ 2 MB, не критично, но принцип «платим только за используемое» — `docs/E38_fixes2.md §6`).

### 3.5. `overlap_low = 0.3` определена, но не используется
Пороги: `0.9`, `0.5·rb = 0.0498`, `3·rb = 0.299`. `overlap_low` не участвует. Косметика (или legacy от TZ-версии).

### 3.6. `at_random` для `ov_med ∈ [0.5·rb, 3·rb)`
Дизайн-выбор: `0.0596` (текущее значение, 0.6·rb) → `at_random` ✅, но `0.15` (1.5·rb) и `0.25` (2.5·rb) тоже `at_random`, хотя семантически «выше случайного». Если random_baseline = `0.1`, то `ov = 0.15` — это «на 50% выше baseline», не «at random». Paper говорит «at_random (40% ниже случайного)» — это верно для текущего 0.0596, но `3·rb` (0.3) как верхняя граница — слишком широкая.

Не баг, но стоит документировать в `meta` (или сузить окно, например `[0.5·rb, 2·rb]`).

### 3.7. `tails_anticorrelated` для `ov_med < 0.5·rb`
Overlap ∈ [0, 1], 0 = ортогональные. «Anti-correlated» — странная метка для «оверлап ≈ 0». Скорее «strongly disjoint / below-chance». Косметика, но paper/E38.md могли бы это уточнить.

### 3.8. `rho_lens_med = −0.927` — сильная отрицательная; paper упоминает «ρ_lens = −0.927» но не интерпретирует отдельно
См. §1: это strong tail-sparsity, не «no structure». Стоит явно записать `meta.rho_lens_interpretation: "ρ < −0.3 = tail-sparse (энергия концентрируется в хвосте при низкой энтропии)"`.

### 3.9. `entropy_frac_tail` — guard для отрицательных в c2 отсутствует
`c2 = (...)**2 ≥ 0` по построению, но defensive `c2 = c2.clamp_min(0)` не помешал бы (если в будущем c2 будет передаваться не из `**2`).

### 3.10. `entropy_frac_tail(c2_o, tail_o)` — `tail_idx + 1` vs `tail_idx`
Внутри: `c2[..., tail_idx + 1:].sum(...)`. `tail_o = 115`, значит modes 116..127 (12). `V_our_tail = V_our[:, tail_o + 1:]` — modes 116..127. Согласовано ✅. Но `tail_idx` семантически — «последний индекс перед хвостом», что не очевидно из имени. Комментарий был бы полезен.

### 3.11. `spearmanr` возвращает `(r, p)`; `p` отбрасывается
Paper не показывает p-value для ρ_our/ρ_lens (только CI). Хорошо, но если `p` нужен для significance — потерян. На текущих данных ρ_lens = −0.927 с p ≈ 0, ρ_our = −0.041 с p ≈ 0.65 (не значимо) — но p не в JSON. Не баг, но информативность.

### 3.12. `lens_validity` — `s1_ratio` вычисляется, но gate использует `s1` (не ratio)
```python
lens_valid = bool(jmin_i < 3.0 and s1 < 4.0)
```
Gate — `s1 < 4.0`, не `s1_ratio < 4.0`. Paper: «valid = false (σ1/σmedian = 5.45, ‖J−I‖ = 17.95)». То есть `σ1/σmedian = 5.45` — это не порог, а info; gate fail из-за `‖J−I‖_F = 17.95`. Код правильно использует `s1 < 4.0` и `jmin_i < 3.0`. ✅ Но семантика «σ1/σmedian = 5.45» в paper без контекста читается как «gate fail по ratio», что неверно. Не баг, но paper неточен.

### 3.13. `V_lens_tail = V_lens[:, tail_l + 1:]` — modes 922..1023
`tail_l = 921`, `tail_l + 1 = 922`, `922..1023` = 102 modes. Bottom 10% из 1024 = 102.4 → 102. ✅

### 3.14. `V_our_tail = V_our[:, tail_o + 1:]` — modes 116..127
`tail_o = 115`, `tail_o + 1 = 116`, `116..127` = 12 modes. Bottom 10% из 128 = 12.8 → 12. ✅

### 3.15. `random_baseline = r_lens / D = 102 / 1024 = 0.0996` ✅

### 3.16. `entropy_frac_tail` для 1-D — корректно (assert); для batched — устаревший docstring уже удалён ✅

### 3.17. `torch.linalg.svd(J_our[i, t], full_matrices=False)` — для [128, 1024] даёт Vh [128, 1024]; V_our = Vh.T [1024, 128]. Только 128 modes — правильно (P0-фикс). ✅

---

## 4. Что проверено и **корректно**

| Аспект | Статус |
|---|---|
| P0-фикс (удаление null completion, primary = overlap) | ✅ |
| MEDIUM-4 (doc-level bootstrap) | ✅ |
| MEDIUM-7 (guard на `size >= 2` перед bootstrap_ci) | ✅ |
| `V_lens_tail` и `V_our_tail` — правильные срезы (bottom 10%) | ✅ |
| Overlap = ‖V_our_tail^T V_lens_tail‖_F² / r_our | ✅ |
| Random baseline = r_lens / D | ✅ |
| `tz_preregistered_outcome` — тот же критерий, что в paper | ✅ |
| `verdict = at_random,lens_validity_suspect` | ✅ |
| `lens_sha256` — вычисляется | ✅ |
| `lens_validity.valid = false` (jmin_i > 3.0) | ✅ |
| `rho_our` и `rho_lens` в правильных базисах (128 vs 1024) | ✅ |
| `S_our_entropy`, `frac_tail_our` — per-position | ✅ |
| `v_lens` [1024, 1024] правые сингулярные векторы | ✅ |
| `V_our` [1024, 128] — 128 REAL modes, не 896 null | ✅ |
| Atomic write + strict JSON | ✅ |
| `del lens`, `del J_our`, `del H` — освобождение памяти | ✅ |
| Paper ↔ code (значения) | ✅ |
| Paper ↔ code (`completion_sensitive` и др.) | ❌ (см. §2) |

---

## 5. Сверка с paper.md E-38g

| Paper | Code | Match |
|---|---|---|
| overlap = 0.0596, CI [0.0587–0.0604] | `ov_med`, `ov_ci` | ✅ |
| random-baseline = 0.0996 | `random_baseline` | ✅ |
| ρ_our = −0.041, CI [−0.120, +0.009] | `rho_our_med`, `rho_our_ci` | ✅ |
| ρ_lens = −0.927, CI [−0.936, −0.912] | `rho_lens_med`, `rho_lens_ci` | ✅ |
| lens_validity = false, σ1/σmedian = 5.45, ‖J−I‖ = 17.95 | `lens_validity` | ✅ |
| verdict = at_random, lens_validity_suspect | `out["outcomes"]["verdict"]` | ✅ |
| tz_preregistered = jlens_not_tail_modes | `out["outcomes"]["tz_preregistered_outcome"]` | ✅ (семантика — см. §1) |
| r_our = 12, r_lens = 102 | `out["params"]["r_our"]`, `["r_lens"]` | ✅ |
| completion_sensitive = false | — | ❌ (см. §2) |
| rho_our_random_qr = −0.166 | — | ❌ (см. §2) |

---

## 6. Итоговая таблица

| Аспект | Статус |
|---|---|
| Числа воспроизводятся | ✅ |
| Критические баги, портящие числа | ❌ Нет |
| **`jlens_not_tail_modes` при ρ_lens = −0.927** | ⚠️ **Семантика неверна** (paper vs pre-registration) |
| **Paper ссылается на `completion_sensitive` / `rho_our_random_qr`, которых нет в code** | ⚠️ Расхождение paper ↔ code |
| Inf/NaN guards upfront | ⚠️ Латентно |
| `overlap_t = np.empty` вместо `np.full(nan)` | ⚠️ Косметика |
| `S` импорт мёртвый | ⚠️ Косметика |
| `overlap_low = 0.3` не используется | ⚠️ Косметика |
| `at_random` для `1.5·rb..2·rb` семантически широко | ⚠️ Дизайн |
| `tails_anticorrelated` для `ov < 0.5·rb` | ⚠️ Косметика |
| `spearmanr` p отбрасывается | ⚠️ Информативность |
| SVD, overlap, bootstrap, verdict | ✅ |

---

## 7. Рекомендации

1. **Исправить семантику `jlens_not_tail_modes`**: либо уточнить pre-registration (`|ρ| < 0.3`, `ρ < −0.3 = tail_sparse`, `ρ > 0.3 = anti_tail_sparse`), либо переименовать в что-то описательное (`rho_our_zero_rho_lens_strong_negative`) и явно записать в `meta.rho_lens_interpretation`.
2. **Обновить paper.md и E38.md**: убрать `completion_sensitive` и `rho_our_random_qr` (или восстановить соответствующий расчёт; например, сравнить ρ_our при двух случайных QR-дополнениях до 1024, если |Δρ_our| < 0.05 → `completion_sensitive: false`).
3. **`torch.isfinite(H).all()` / `torch.isfinite(J_our).all()` upfront** — fail loud на corrupt cache (тот же класс, что в E-38e).
4. **`overlap_t = np.full((N, Ss), np.nan)`** вместо `np.empty` — defensive.
5. **Убрать `S` из импорта** или использовать в assert.
6. **`want=("J", "H")` в `load_layer`** — не грузить `P`.
7. **Удалить `overlap_low`** (dead) или использовать в `meta`.
8. **Уточнить `at_random` окно**: `[0.5·rb, 2·rb]`, а `[2·rb, 3·rb]` — `partial_below`; или явно записать в `meta` эмпирическое правило.
9. **Добавить `spearmanr` p-value** в `rho_our` / `rho_lens` — информативность (не влияет на вердикт, но полезно для читателя).
10. **`lens_validity.s1_ratio` — явно назвать `s1_over_median`**, чтобы не путать с gate `s1 < 4.0`; paper `σ1/σmedian = 5.45` уже подразумевает ratio, но в коде это не gate.

## Короткий ответ

**Отчётные числа (residual ≈ 5.48e-16, drifts, «MK совпадает», verdict «identity holds... tautological») на текущем кэше воспроизводятся**, и эксперимент **честно признаёт тавтологичность** residual-проверки (P0-фикс: информативен только leaf drift). P0/P1/P3-фиксы из `docs/E38_fixes.md` применены. **Критических багов, портящих числа, нет.**

Но есть **документационная неточность**, **несколько латентных путей**, и **одно место, где paper может быть неправильно интерпретирован** (`median_shared = 0.656` при включённом s=0).

---

## 1. Документационная неточность: «max over ALL rows underreports» — неверно

Docstring:
> «115 of 128 rows are edit-invariant (not in token_set), so max/median over ALL rows underreports — report shared-only too.»

**Для max это неверно.** Unshared rows имеют drift = 0, что не увеличивает `max`. Значит:
```
max over ALL (128 rows) == max over SHARED (13 rows)   [для max]
```
`J_rel_drift_max == J_rel_drift_max_shared` **тождественно** (по построению), а не «лучше». Аналогично для `MK_rel_drift_max`.

**Для median это верно**: `median([13 nonzero, 115 zeros])` → 0, что сильно занижает per-state медиану.

**Следствие:** поля `J_rel_drift_max` и `J_rel_drift_max_shared` в JSON **избыточны** (совпадают). Поля `J_rel_drift_median_shared` (по 13 shared) — **корректны**; поля `J_rel_drift_max` (по всем 128) — **не underreport**, а просто redundant.

**Фикс:** либо удалить `_max` (не `_max_shared`) поля, либо исправить docstring: «median over ALL rows underreports; max does not — both are reported for traceability».

---

## 2. `s=0` включён в агрегаты — paper может репортить с bias

`states = {0: W_full.clone(), 10: W_ed_10, ...}` — `s=0` в `out_states`. Для `s=0`:
- `J_rel_drift_max = 0`, `J_rel_drift_median_shared = 0`
- `MK_rel_drift_max = 0`

`paper.md` §4.3:
> «J_rel_drift max 0.355 (s=10) → 1.152 (s=1000), **median_shared = 0.656**»

Если paper «median_shared = 0.656» считается как `median` по **всем** (s, h, shared) включая s=0 (13 нулей × 16 h = 208 нулей), то это занижает. Если по s>0 — корректно. **Код сам по себе это не определяет** — `J_rel_drift_median_shared` считается per-state (по 16 h × 13 shared для каждого s), и `out_states[0]["J_rel_drift_median_shared"] = 0` — но paper мог взять `median` по 11 значениям (включая 0) или по 10 (исключая).

**На числах бумаги (0.656) не отражается очевидно** — это правдоподобно для s>0 drift ≈ 0.3–1.0. Но необходимо **явно** записать в JSON, какой агрегат репортится, чтобы paper и code не разошлись (это повторяющийся паттерн: E-38c `well_separated`, E-38g `jlens_not_tail_modes`).

**Фикс:** добавить в JSON поле `aggregate_median_shared_excl_s0` или `paper_reported_aggregate` с явной формулой.

---

## 3. Латентные пути

### 3.1. Dtype `W_full` — комментарий говорит fp64, `emb[U]` может быть fp32

```python
W_full = emb[U]   # "tied-embedding rows on U128 (fp64)"
```
`emb` из `load_model_tensors()` — если fp32 (типично для HF embeddings), то `W_full` **fp32**, а не fp64. `W_ed[shared_idxU] = W_ed[shared_idxU] + dW_shared` (если dW_shared fp64) — fp32 + fp64 = fp64, **assignment truncates to fp32**. Значит `states[k]` fp32.

Фактический residual при этом всё равно ~5e-16 (потому что `j_raw`/`factor_residual` считают в fp64, а W — общий для обеих сторон тождества), но **комментарий «fp64» в JSON неточен**.

**Фикс:** `W_full = emb[U].double()` явно, или проверить dtype `emb` через `torch.isfinite(W_full).all()` и assert `W_full.dtype == torch.float64`.

### 3.2. Нет `torch.isfinite(H).all()` / `torch.isfinite(W_full).all()` / `torch.isfinite(snaps_all[k]).all()`

Тот же класс, что в E-38e/E-38g:
- `H` с Inf → `factor_residual(W, H, gamma)` = NaN → `max(res)` = NaN → `resid_max_all = NaN`, `NaN <= 1e-6` = False → `RESIDUAL GROWTH — inspect`. **Ложное срабатывание** на повреждённом кэше.
- `W_full` с Inf → `jdr = NaN`, `J_sh = NaN`, JSON nan.

**Фикс:** fail-loud upfront (3 assert).

### 3.3. `shared_idx256` может быть пустым — silent no-op

Если U128 и E-16 256-токен-сет **не пересекаются** (не должно быть, но 13 — магическое число), `shared_idx256 = tensor([])`, `shared_idxU = tensor([])`. `W_ed[shared_idxU] = W_ed[shared_idxU] + dW_shared` — no-op. Все drifts = 0. Verdict «identity holds» — **бессмысленный**, но не fail-loud.

**Фикс:** `assert shared_idx256.numel() >= 1, "no intersection between U128 and E-16 token set"`.

### 3.4. `ts256` может содержать дубликаты — двойное добавление ΔW

```python
for i256, tok in enumerate(ts256.tolist()):
    if tok in pos:
        shared_idx256.append(i256)
        shared_idxU.append(pos[tok])
```
Если `tok` появляется дважды в `ts256` (top-256 обычно уникальны, но проверки нет), `pos[tok]` будет добавлен дважды, и `W_ed[pos[tok]] + dW_shared[i256] + dW_shared[i256']` — двойная поправка. **Silent corruption.**

**Фикс:** `assert ts256.unique().numel() == ts256.numel(), "E-16 token set has duplicates"`.

### 3.5. `factor_residual` return type — должен быть Python float, не Tensor

Если `factor_residual` возвращает `torch.Tensor` (0-dim), то:
- `max(res)` работает (сравнение 0-dim bool, truthy check)
- `sum(res) / len(res)` работает
- Но в JSON dict: `"factorization_residual_max": max(res)` — sanitize вернёт tensor как есть → `json_dump` вызывает `_raise` → **JSON не пишется**.

Paper сообщает числа (5.48e-16), значит работает — `factor_residual` возвращает float. Но проверки нет.

**Фикс:** `assert isinstance(res[0], float), f"factor_residual returned {type(res[0])}"`.

### 3.6. `s=0` residual включён в `resid_max_all`

`resid_max_all = max(v["factorization_residual_max"] for v in out_states.values())` — включает s=0 (тоже floor ~5e-16). Если бы s=0 давал более высокий residual, вердикт сломался бы. На практике — нет (algebraically same). Латентно.

**Фикс:** `resid_max_all = max(v[...] for s, v in out_states.items() if s > 0)` — исключая baseline.

### 3.7. `jdr` — max over 16 h × max over 128 rows — redundant с `J_sh.max()`

См. §1. Не баг, но дублирование в JSON.

---

## 4. Что проверено и **корректно**

| Аспект | Статус |
|---|---|
| P0-фикс: `snapshots[k]` = ΔW (без двойного вычитания) | ✅ |
| P1: cached h (tied model, h_t меняется в реальной модели) — документировано | ✅ |
| P3: `clamp_min(1e-12)` в знаменателе drift | ✅ |
| `states = {0: W_full.clone()}` + все K_STEPS | ✅ |
| `W_ed[shared_idxU] = W_ed[shared_idxU] + dW_shared` (advanced indexing) | ✅ |
| `factor_residual(W_ed, h, gamma)` — для каждого сэмпла | ✅ |
| `J0` baseline — original W | ✅ |
| `J_rel_drift_max` / `MK_rel_drift_max` — all 128 rows | ✅ (redundant, см. §1) |
| `J_rel_drift_max_shared` / `MK_rel_drift_max_shared` — 13 shared | ✅ (информативно) |
| `factorization_residual_max` и `_mean` per state | ✅ |
| `verdict` string — «tautological by construction» | ✅ (честно) |
| `s0_baseline` note — «uninformative baseline, fp64 floor» | ✅ |
| `E38_fixes P0/P1/P3` в docstring | ✅ |
| Atomic write + strict JSON | ✅ |
| `SAMPLES` — 4 домена × 4 позиции | ✅ |
| `emb` и `gamma` загружены из `load_model_tensors` | ✅ |
| `verify` token_set intersection | ✅ (shared_idx256/idxU) |

---

## 5. Сверка с `paper.md` §4.3 / E-38i

| Paper | Code | Match |
|---|---|---|
| residual = 5.48e-16 (fp64 floor) | `factorization_residual_max` | ✅ (если `emb` fp64) |
| J_rel_drift max 0.355 (s=10) → 1.152 (s=1000) | `out_states[10/1000]["J_rel_drift_max"]` | ✅ |
| median_shared = 0.656 | `median_shared` (агрегат — см. §2) | ⚠️ зависит от s=0 |
| MK совпадает | `MK_rel_drift_max_shared` ≈ 0 | ✅ (paper структурный комментарий) |
| residual — тавтология (pre-flight) | verdict string | ✅ |
| Goodhart не подтверждён | verdict | ✅ |
| tied-weights оговорка | `structural_fact` | ✅ |

**Единственное потенциальное расхождение — §2 (s=0 в median_shared).**

---

## 6. Итоговая таблица

| Аспект | Статус |
|---|---|
| Отчётные числа воспроизводятся | ✅ |
| Критические баги, портящие числа | ❌ нет |
| P0/P1/P3-фиксы из `E38_fixes.md` | ✅ |
| **Docstring "max underreports"** | ⚠️ **Неверно для max** (только для median) |
| `J_rel_drift_max == J_rel_drift_max_shared` (redundant) | ⚠️ Дубликат в JSON |
| `s=0` в агрегатах paper (`median_shared = 0.656`) | ⚠️ Нужно проверить, какой агрегат |
| `W_full` dtype fp64 (комментарий) vs `emb` fp32 | ⚠️ Верифицировать |
| `torch.isfinite` guards | ⚠️ Латентно |
| `shared_idx256` может быть пустой | ⚠️ Silent no-op |
| `ts256` дубликаты не проверяются | ⚠️ Silent double-add |
| `factor_residual` return type | ⚠️ Верифицировать |
| `resid_max_all` включает s=0 | ⚠️ Косметика |
| Verdict, JSON, atomic write | ✅ |

---

## 7. Рекомендации

1. **Исправить docstring**: «max over ALL rows does not underreport (equals max over shared); median over ALL rows underreports». Либо удалить `J_rel_drift_max`/`MK_rel_drift_max` (все 128) как дубликаты `_shared` полей.
2. **Добавить явный агрегат `paper_median_shared`** с формулой (median over s>0, or median over all s, or median over all (s, h, shared)) и перепроверить paper.md на соответствие.
3. **`torch.isfinite(H).all()` / `torch.isfinite(W_full).all()` / `torch.isfinite(snaps_all[k]).all()`** — fail loud.
4. **`assert shared_idx256.numel() >= 1`** — иначе тихий no-op.
5. **`assert ts256.unique().numel() == ts256.numel()`** — иначе double-add.
6. **`assert isinstance(res[0], float)`** (или explicit `float(factor_residual(...))`).
7. **`resid_max_all` — исключить s=0** (`if s > 0`), так как s=0 неинформативен и документирован как baseline.
8. **`W_full = emb[U].double()`** — явный cast, если `load_model_tensors` не гарантирует fp64; или assert `W_full.dtype == torch.float64`.
9. **`s0_baseline` — преобразовать в структурированный dict** (`{"s": 0, "note": "...", "residual_floor": 5e-16}`) для consistency с остальными полями.