# Проверка на корректность, соответствие ТЗ и тихие ошибки

Ниже — послойный аудит по трём файлам с разбивкой «соответствует ТЗ / отклонение / потенциальная тихая ошибка». Ключевые пункты помечены severity.

---

## 1. `exp38a_uhat_dynamics.py`

### Соответствует ТЗ
- `û_t = h_t / ‖h_t‖₂` inline с `clamp_min(1e-12)` — как в ТЗ. ✓
- `s(t) = ‖û_{t+1} − û_t‖`, `κ_g(t) = geodesic_curvature(û_{t−1}, û_t, û_{t+1})` — вызовы импортированы из `prism.holonomy`, не из `jstruct`. ✓
- Порог аномалии `|median(seg) − median(body)| / IQR(body) > 2`. ✓
- Профили `profile_S`/`profile_K` для визуализации. ✓
- L18 (full) vs L19 (GDN) отдельно — `l18_vs_l19_gdn`. ✓
- `pos0` записывается отдельным блоком. ✓

### Отклонения от ТЗ (документированы в docstring, но не безобидные)

**[MEDIUM] Хвостовой сегмент `tail`:**
- ТЗ: `tail = {S−2, S−1}` (две позиции).
- Код: `tail = {S−2}` (одна позиция), `BODY_END = S−2`.
- Причина корректна: `s` имеет длину `S−1` (индексы 0..S−2), `κ_g` имеет индексы 1..S−2; позиции `S−1` в этих полях физически нет. Docstring это фиксирует. **ТЗ содержит ошибку, код прав** — но это отклонение от pre-registered интерпретации, которое стоит явно упомянуть в отчёте.

**[MEDIUM] Словарь вердиктов:**
- ТЗ pre-registers три метки: `edges_special`, `edges_typical`, `edges_atypical`.
- Код добавляет `mixed` и `edges_atypical_candidate`. В статусе E-38a есть строка «L6 — mixed: tail-скорость аномальна», т.е. эта метка реально попала в отчёт. Это нарушение pre-registration — либо пере-формулировать, либо явно указать как post-hoc расширение.

**[LOW] Асимметричный порог для `edges_atypical_candidate`:**
- Для «special/mixed» используется `anomaly_ratio > 2.0` (`T_ANOM`).
- Для «atypical candidate» — `anomaly_ratio > 0.5`.
- ТЗ использует `<<` без порога; разумная интерпретация — тот же `T_ANOM=2.0`. Код ослабляет порог в 4 раза для «нижнего» направления. Метка `candidate` частично компенсирует, но асимметрия может дать false-positive `edges_atypical_candidate` там, где head просто чуть ниже body. Агрегат `edges_typical` в текущем прогоне от этого не пострадал, но если 3+ слоя окажутся `edges_atypical_candidate`, агрегат «поедет».

**[LOW] Обработка pos 0:**
- ТЗ: «Исключать из медиан **или** записывать как отдельный сегмент».
- Код: записывает в блок `pos0`, **но не исключает** из `head`. Если ТЗ имело в виду «исключить и записать», поведение расходится. С учётом того, что `edges_typical` — итог, вероятно, `pos0` не доминирует в `head`, но стоит пере-проверить отдельно для L22/L6 (там аномалии ближе к краям).

### Потенциальные тихие ошибки

**[HIGH] `geodesic_curvature` может упасть на точном нуле:**  
При `‖h_t‖ = 0` (attention sink) `norms.clamp_min(1e-12)` даёт `u = 0`, не единичный вектор → `geodesic_curvature` бросает `ValueError` на unit-check (`abs(n - 1.0) > 1e-4`). Это **громкий** крах, не тихий, но — крах посреди прогона. Для E-38a на pos 0 всё прошло, значит, точных нулей в кэше нет; но для других экспериментов серии (E-38d на L22, если считать `Ω` по `û_t`) риск тот же. Рекомендация: пропускать тройки с `‖h_t‖ < 1e-10` (NaN в `kappa`) и фиксировать в `meta.nan_count`, как в E-28 («дегенерации → NaN и исключаются»).

**[MEDIUM] `norms.median(dim=-1)` включает pos 0 в медиану:**  
`pos0_ratio = norms[:, 0] / norms.median(dim=-1).values` — медиана по всем 128 позициям, включая pos 0. При сильном attention sink медиана сместится, `ratio` будет завышен. Эффект мал (1 из 128 позиций), но при диагностике «насколько pos 0 аномален» систематически завышает аномальность. Лучше `norms[:, 1:].median(dim=-1)`.

**[LOW] `json.dump(..., default=float)`:**  
`np.bool_` в JSON превратится в `1.0` вместо `true`. Сейчас все булевы поля явно обёрнуты `bool(...)` — проверено, тихой конверсии нет. Но если кто-то добавит поле из `numpy` без `bool()`, получит не-булев JSON молча.

**[LOW] Tie-breaking при агрегате:**  
`aggregate = max(counts, key=counts.get)` при равенстве голосов выбирает первый по порядку вставки (слой 6 → 12 → …). Не задокументировано. При `edges_typical: 3, mixed: 3` результат непредсказуем с точки зрения ТЗ.

---

## 2. `holonomy.py`

### Соответствует ТЗ (E-38d)
- `procrustes_align` возвращает `(R, residual_sq)` — tuple, как в ТЗ. ✓
- `holonomy_accumulate(..., return_total=True)` возвращает `(kappa, per_step, R_total)` — ТЗ ждёт `H_012` из `holonomy_spectrum`, отдельный вызов не нужен. ✓
- `R_total` в float64. ✓
- `geodesic_curvature` — «дискретная геодезическая кривизна, не дефект неравенства» (E38.md прямо это специфицирует). ✓
- Procrustes **без det-fix** (полярный фактор, не SO(r)-проекция) — как в ТЗ. ✓
- Gauge-инвариантность `U_i → U_i Q_i` — `procrustes_align` инвариантен к joint sign-flip; κ инвариантна к смене базиса. ✓

### Отклонения

**[MEDIUM] `geodesic_curvature` ≠ `Ω` из `paper_old.md` [M-3]:**  
`paper_old.md` определяет `Ω(t) = |A(u1,u2) + A(u2,u3) − A(u1,u3)|` (дефект треугольного неравенства). `E38.md` и код реализуют `turn/arc` (геодезическая кривизна). Формально это разные объекты; при коллинеарной тройке оба дают 0, но численно различаются. `E38.md` прав («не дефект неравенства»), `paper_old.md` — устаревшая формулировка `[M-3]`. **Кросс-документная несогласованность** — при переносе результатов E-38d в бумагу `[M-3]` надо переписать под код, иначе ревьюер поймает расхождение.

### Потенциальные тихие ошибки

**[HIGH] `holonomy_spectrum` может молча потерять π-углы (lone −1 / −I-блок):**

```python
angles = torch.angle(eigvals)   # (-π, π]
positive = angles[angles > 1e-6]
```

Для собственного значения `λ = −1` (реального, imag = 0 или ±tiny):
- `torch.angle(-1 + 0j) = atan2(0, -1) = +π` — попадает в `positive`.
- `torch.angle(-1 - 0j) = atan2(-0, -1) = −π` — **отфильтровывается**.

LAPACK-вычисление ортогональной матрицы даёт реальные λ с imag ≈ ±tiny. Знак зависит от fp-шума. Для пары `{−1, −1}` (2D-поворот на π) возможен сценарий **оба λ с imag < 0**: обе −π, обе отфильтрованы, π-пара теряется молча (если других положительных углов нет — `expanded = zeros(expected)`, «плоский» вердикт вместо π).

Механика:
- В `holonomy_spectrum` есть `elif expanded.numel() != expected: warnings.warn(...)` — предупреждение сработает, **если** `positive.numel() > 0`. Но не «missing π», а generic mismatch.
- Если все углы отфильтрованы (`positive.numel() == 0`), код идёт в `torch.zeros(expected)` — это **тихая ошибка**: реальный `κ ≈ 2` (для lone −1) репортится как `θ = [0,…,0]`, `κ ≈ 0`.

В E-38d отчёт `tr(H)` берётся из матрицы (статус это явно фиксирует: «след считается из матрицы H, всегда точно»), поэтому κ **не** потерян. Но `{θ_j}` спектр может быть неполным — что ломает «тождества» `tr = 2Σcos θ`, `‖H−I‖² = 4Σ(1−cos θ)`. Статус фиксирует `idiff=4.0` для случая, когда π **есть** в theta — а если π **не** попал, idiff будет другим.

Рекомендация:
```python
positive = angles.abs() > 1e-6   # вместо angles > 1e-6
# далее работать с abs-значениями, π не различается ±
```
и/или явно ловить `λ ≈ −1` до фильтра. Это единственный содержательный кандидат на «тихую ошибку» в `holonomy.py`.

**[MEDIUM] `procrustes_align` при почти ортогональных подпространствах — R численно нестабилен:**
Docstring честно предупреждает: «residual_sq → 2r, consumers should not rely on R». В E-38d это может дать «мусорный» `H_012` на позициях с дегенерацией. ТЗ (E-38d Pitfalls) требует: «detect and exclude the triple» при крупном `residual_sq`. В `exp38d`-скрипте (не приложен, но по статусу судя) — **надо убедиться**, что эта фильтрация есть: `residual_sq` близко к `2r` должно давать `NaN`/пропуск, а не молча попадать в `{θ_j}`.

**[LOW] `holonomy_spectrum` — `rounded(decimals=8)` для малых углов:**
Угол `θ ≈ 1.49e-6` (чуть выше фильтра `> 1e-6`) округляется до `0.00000149` → остаётся. Угол `θ ≈ 1.00000001e-6` округлится до `0.00000100` — ок. Но `θ ≈ 1.0000004e-6` и `θ ≈ 1.0000005e-6` сольются в одну «моду» — очень edge case, для E-38d не критично.

---

## 3. `_e38_cache.py`

### Соответствует ТЗ
- Формат `JR[(dom,si,t)]` → `J[N,S,K,D]`, `P`, `H`. ✓
- Порядок доменов `wiki, math, code, chat`. ✓
- `gauge_free_J` соответствует `P(p)J = p[:,None]·(J − (Jᵀp)[None,:])`. ✓ Проверено аналитически.
- `torch.empty` полностью заполняется — если ключа нет, будет `KeyError` (громко), а не тихая дырка в тензоре. ✓

### Потенциальные тихие ошибки

**[MEDIUM] Пиковая память на слой ~4.2 GB:**
- `ck` (~2.1 GB) + стек `J` (~2.1 GB) одновременно в памяти, `del ck` освобождает `ck`, но не `J`.
- Если среда ≤ 4 GB free в момент загрузки — `OOM` (громко, но посреди прогона, между слоями). Статус E-38a говорит «выполнено» — значит, ресурсов хватило, но для других потребителей (`exp38d` может быть тяжелее) стоит зафиксировать минимальные требования.

**[LOW] `torch.load(..., weights_only=False)`:**
- В PyTorch 2.6+ `weights_only=True` — дефолт и настоятельно рекомендуется; `False` открывает pickle-инъекции. Для внутреннего кэша допустимо, но стоит добавить комментарий, что это осознанное решение, а не копипаст.

**[LOW] `gauge_free_J` предполагает `P` уже нормирован (`sum=1`):**
Docstring `_e38_cache.py` говорит `P128` — normalized (sum=1). Если подать `P128_RAW` (sub-stochastic), результат будет «полу-gauge-free» — **тихая ошибка** (метрики посчитаются, но семантика другая). В `load_layer` берётся именно `P128`, так что в текущем прогоне всё верно; но интерфейс `gauge_free_J(J, P)` не проверяет `P.sum()≈1` и не задокументирован как «требует нормированного P». Для E-38b/c/e (которые тоже используют `_e38_cache`) — стоит добавить assert в `load_layer`.

---

## 4. Сводная таблица

| Файл | ТЗ-соответствие | Отклонения | Тихие ошибки (severity) |
|---|---|---|---|
| `exp38a_uhat_dynamics.py` | в основном ✓ | tail={S−2} вместо {S−2,S−1} (обосновано); словарь вердиктов расширен (mixed, atypical_candidate); atypical threshold=0.5 vs 2.0; pos0 включён в head | падение `geodesic_curvature` при `‖h‖=0` (**H**, громко); медиана `pos0_ratio` включает pos0 (**M**); tie-breaking агрегата (**L**) |
| `holonomy.py` | ✓ (E-38d), кроме [M-3]-несогласованности с paper_old.md | `geodesic_curvature` ≠ Ω-defect из paper_old.md [M-3] (E38.md прав) | **потеря π-углов при `torch.angle = −π`** (**H**, тихо, если нет других +углов); `procrustes_align` при дегенерации R (M, документировано) |
| `_e38_cache.py` | ✓ | — | пик памяти 4.2 GB/слой (M, громко); `gauge_free_J` не валидирует `P.sum()≈1` (L); `weights_only=False` (L) |

---

## 5. Минимальные правки, которые я бы внёс перед прогоном E-38d

1. **`holonomy_spectrum`**: заменить `angles[angles > 1e-6]` → `angles[angles.abs() > 1e-6]`, работать с `abs(angles)` в `positive`. Для π-бина: `pi_mask = uniques > π − 1e-6` уже есть; при `λ=−1` в паре `{e^{iθ}, e^{-iθ}}` с `θ=π` и `counts` нечётном — `counts//2` теряет единицу, `clamp(min=1)` это компенсирует, но лучше явно: `mult = (counts + 1) // 2` для π-бина (нечётный counts означает либо пару + шум, либо lone −1).
2. **`geodesic_curvature` + E-38a/d**: явный NaN-пропуск для `‖h_t‖ < 1e-10` (как E-28) с записью `nan_count` в `meta`.
3. **`exp38a`**: `norms[:, 1:].median(dim=-1)` для `pos0_ratio` (исключить сам pos0 из знаменателя).
4. **`exp38a` verdict**: либо убрать `edges_atypical_candidate` (использовать pre-registered `edges_atypical` с порогом 2.0), либо задокументировать `0.5` как осознанный post-hoc порог в `meta` и в `docs/E38_fixes.md`.
5. **`_e38_cache.load_layer`**: `assert P.sum(dim=-1).allclose(1.0, atol=1e-3)` перед возвратом (защита от случайной подстановки `P128_RAW`).
6. **`paper_old.md` §[M-3]**: обновить формулу `Ω` под реализацию `turn/arc` (или переименовать в `paper_old.md` `Ω_defect` явно, а `Ω` оставить для `geodesic_curvature`) — иначе при переносе E-38d в статью ревьюер поймает несоответствие определений.

Всё остальное — `procrustes_align` (формула `R = V Uᵀ` с `M = U2ᵀ U1`), `holonomy_accumulate` (порядок сомножителей), `gauge_free_J` (`p·(J − Jᵀp)` = `P(p)J`), `residual = 2r − 2Σσ`, JSON-сериализация с `default=float`, `torch.empty` + полное заполнение — **корректно** и проверено аналитически.

# Аудит `exp38b_position_audit.py` и `cache.py`

## 1. `exp38b_position_audit.py` — соответствие ТЗ E-38b

### Соответствует ТЗ ✓
- `C_b(t)` — cross-doc Frobenius cosine **gauge-free** J (через `gauge_free_J`), не raw J. ✓
- `PR_modes(t)` = `participation_ratio(sv.S**2)` — PR по собственным значениям `J^(p) J^(p)^T` (§6 paper: «PR для спектра `JJ^T`»). ✓
- `bootstrap_ci(values, B=1000, seed=0, statistic=np.median)` — импорт из `prism.cache`, сигнатура совпадает. ✓
- Body = `[16, S−2]` (включая S−2) — совпадает с pitfall-note ТЗ («Сегмент `body` включает `t = S−2`»). ✓
- pos0 baseline отдельно. ✓
- Профиль по t (`profiles`) для L22-визуализации. ✓
- `median-of-medians` = per-doc median по остальным 31, затем median по docs — обосновано в docstring (≠ flat median по 32·31 парам). ✓

### Отклонения от ТЗ (severity)

**[HIGH] `align_K1` меряется на `J^(p)`, а не на `K = J − J^(p)`:**
- ТЗ называет метрику `align_K1` (отсылка к E-37). E-37 (`paper_old.md` §18) определяет `align_K1` как `|cos(top-left-singular **K**, 1/√K)|` где `K = J − P(p_t)·J` — **убитая** компонента.
- Код: `sv = svd(Jt)` где `Jt = Jgf[:, t] = P(p)J` — **оставшаяся** компонента. Это другой объект.
- Следствие: значения несопоставимы с E-37-числами `0.895 (L18) → 0.710 (L22)`. В статусе заявлено «align_K1 как в E-37» — но числа будут другими по построению.
- Оговорка: если ТЗ буквально читать как «`align_K1` для **J**» (как написано в Setup), код формально прав; но имя + Pitfall-ссылки на E-37 допускают именно `K`.

**[HIGH] Bootstrap CIs — независимые, а не парные; ТЗ требует `bootstrap_ci_paired`:**
- ТЗ §Procedure: «для парного сравнения body vs pos 0 — `bootstrap_ci_paired`».
- Код: два независимых `bootstrap_ci`:
   - `pos0_ci` — по 32 значениям (per-doc medians).
   - `body_ci` — по 3552 значениям (`32 docs × 111 positions`, flattened).
- Это **не** paired: не учитывает корреляцию pos0↔body для одного и того же документа; при N=32 vs 3552 body-CI становится в ~√(3552/32) ≈ 10× уже, чем должен быть при корреляции. Сравнение двух таких CI даёт ложноположительные срабатывания.

**[HIGH] Вердикт `l22_position_sensitive` — статистически некорректен:**
```python
lo, hi = r["PR_modes"]["body_ci"]
pos0_pr_med = r["PR_modes"]["pos0_median"]
if not (lo <= pos0_pr_med <= hi):
    reasons.append("l22_position_sensitive")
```
- Body CI уже в ~10× (см. выше). Поэтому любая малая разница pos0↔body даст выход за CI → ложная метка.
- Правильно (по ТЗ): per-doc body = median по t в `[16,126]`; затем `bootstrap_ci_paired(pos0_metrics, body_metrics)` по 32 парам — и проверка, что CI разности **исключает 0**.

**[MEDIUM] `pos0` для L22-вердикта — вычисленный, а не 4.6:**
- ТЗ буквально: «`PR_modes` на L22 = 4.6 на `pos 0` … если на середине другое — L22-anomaly position-sensitive».
- Код заменяет 4.6 на `pos0_median` из собственного кэша. Docstring ссылается на `docs/E38_fixes.md §2.3` (файла нет — проверить нельзя). Причина разумна (4.6 в `paper_old.md` §20 — это min-over-positions из E-30, а E-30 определён иначе: `PR_modes = (Σw)²/Σw²` по off²-энергии, а не PR по λ). Но это отклонение от ТЗ-literal и может расходиться с pre-registered интерпретацией.

**[MEDIUM] Нет проверки, что `pos0_pr_median ≈ 4.6`:**
- `PR_MODES_L22_PAPER = 4.6` фиксируется в `meta`, но не сравнивается с `pos0_median`. Если вычисленный pos0-PR на L22 окажется, например, 20 (потому что 4.6 — это не PR-J^(p), а E-30-PR_modes), — это **тихо** уйдёт в отчёт. Рекомендую `if abs(pos0_med - 4.6) / 4.6 > 0.2: meta.warning = "pos0_PR_mismatch"`.

**[MEDIUM] Silent floor при вырожденном J:**
```python
cos_mat = (Jf @ Jf.T) / (nrm[:, None] * nrm[None, :]).clamp_min(1e-30)
```
- Если `‖Jf[i]‖ ≈ 0` (вырожденный документ/позиция), `Jf @ Jf.T[i,:] ≈ 0`, деление на `1e-30` даёт `cos ≈ 0` — молча. В отличие от E-38a, где `geodesic_curvature` **упадёт** на non-unit векторе. Несогласованная обработка degenerate между E-38a и E-38b.
- `1e-30` — слишком слабый порог: `nrm ≈ 1e-5` даст `cos ≈ 1e-5 · J_iJ_j/nrm → плавающий`. Рекомендую `nrm < 1e-6` → NaN и исключение (как E-28).

**[LOW] `torch.eye(N)` внутри цикла:**
- Пересоздаётся 128 раз на слой. Вынести за цикл — не баг, но шум.

### Тихие ошибки

**[MEDIUM] `norms` в E-38a vs E-38b — несогласованная политика degenerate:**  
См. выше. В рамках серии E-38 это создаёт несогласованность: `edges_typical` может быть следствием того, что в E-38a degenerate-triple → `nan` (не влияет на медиану), а в E-38b degenerate-doc → `cos=0` (влияет на медиану).

**[LOW] `json.dump(..., default=float)` — та же ловушка `np.bool_` → `1.0`:**  
Проверено — все текущие bool-поля явно обёрнуты `bool(...)`. Но при добавлении полей — риск.

**[LOW] `torch.linalg.svd(Jt, full_matrices=False)` для `Jt` с вырожденным спектром:**  
На практике rank(Jgf) может быть < 128 (E-35: `eigengap_{16/17}=1.035` — не дегенерация, но близко). SVD вернёт 128 значений, но часть будет ~ 0. PR — не проблема (PR инвариантен к нулевым λ в спектре? Нет: PR = (Σλ)²/Σλ² — нули не меняют, если всего Σλ > 0). OK.

### Проверка логики median-of-medians

```python
iu, ju = torch.nonzero(mask).T          # row-major, iu[i] — строка, ju[i] — столбец
cos_docs = cos_mat[iu, ju].reshape(N, N - 1).median(dim=1).values
```
- `torch.nonzero(~eye)` возвращает пары в row-major: `(0,1),(0,2),…,(0,N-1),(1,0),(1,2),…`.
- `reshape(N, N-1)` группирует по строкам → row i = `cos[i, all except i]`.
- `median(dim=1)` = per-doc median по остальным 31. ✓ Корректно.

### Проверка `align_K1` broadcasting

```python
al_docs = (sv.U * one[None, :, None]).sum(dim=-1).abs().max(dim=-1).values
```
- `sv.U` [N, K, K], `one` [K] → `one[None,:,None]` [1,K,1].
- `sv.U * one` [N,K,K] — `[i, j, k] = U[i, j, k] · one[k]`.
- `sum(dim=-1)` → `[N,K]` — `[i,j] = <U[i, :, j], one>` = cos(U-столбец j, uniform).
- `.abs().max(dim=-1).values` → `[N]` = max_k. ✓ (При условии, что столбцы U — сингулярные векторы, что верно для `sv.U[..., :, k]`.)

Но: `sv.U` — сингулярные векторы **в порядке убывания σ**. Так что `U[:,0]` — топ-режим; `U[:,k]` для k>0 — следующие моды. `.max(dim=-1)` ищет **самый uniform-режим** среди всех — это E-37-конвенция `u_g`, но отклонение от ТЗ-literal «top-left-singular».

---

## 2. `cache.py` — вспомогательная инфраструктура

### Соответствует спецификации ✓
- `bootstrap_ci(values, B=1000, alpha=0.05, seed=0, statistic=np.median)` — точно совпадает с ТЗ E-38b. ✓
- `bootstrap_ci_paired` — реализован с парной ресемплировкой индексов. ✓
- `_atomic_write` (tmp + `os.replace`) — корректно. ✓
- `deterministic_mode` — сохраняет/восстанавливает RNG-state (torch / cuda / numpy / random), включая вложенность. ✓
- `iter_position_pairs` — fixed order, детерминированно. ✓
- `MetricCache` JSON — nan/inf → строки, восстановление обратно. ✓
- `SVDCache.put` — валидирует dtype, shape, descending S, orthonormality (atol=1e-5). ✓

### Потенциальные тихие ошибки

**[MEDIUM] `bootstrap_ci` — i.i.d. resampling поверх (doc × position):**
- Функция сама по себе корректна. Но её использование в `exp38b` для body (32 docs × 111 positions = 3552 «независимых» значения) — статистически невалидно, если позиции внутри документа коррелированы (а они коррелированы, E-31: `C_F(1) = 0.30–0.996`).
- Рекомендация: добавить в `cache.py` функцию `bootstrap_ci_block(values, blocks, ...)` (cluster bootstrap по doc-id) и/или явное предупреждение в docstring `bootstrap_ci`: «i.i.d. assumption — do NOT pass flattened (sample × time) without checking within-group correlation».

**[LOW] `stats = np.array([statistic(s) for s in samples])` — Python loop:**
- Для B=1000 и n=3552 — ~3.5M вызовов `np.median` (0.5–1 s). Не критично, но если `statistic = np.median`, то `np.median(samples, axis=1)` в 50× быстрее. Косметика.

**[LOW] `_check_value` принимает `float`, а `np.float32` — нет:**
```python
if not isinstance(value, float):
    raise ValueError(...)
```
`np.float64` — subclass of `float` в CPython (проходит). `np.float32` — **не** subclass (падает). Это защита от потери точности при JSON-сериализации, но может неожиданно сломать вызывающий код с `torch.Tensor.item()` (возвращает Python float для fp64/fp32 — ок) vs `np.float32(...)` — упадёт. Задокументировано в docstring, но `np.float32`-кейс стоит упомянуть.

**[LOW] `_is_writable` — race condition:**
- Между `open(probe, "w")` и `os.remove(probe)` другой процесс мог удалить файл → `OSError` при `os.remove`. Docstring покрывает: «ONE process per MetricCache — no parallel writes». Приемлемо.

**[LOW] `bootstrap_ci` с мульти-выходной `statistic`:**
- `stats = np.array([statistic(s) for s in samples])` — если `statistic` возвращает array длины D, получится [B, D]; `np.percentile(stats, [lo, hi])` → [2, D]; `lo, hi = ...` — распаковка в массивы; `float(lo)` упадёт (громко). Не тихая ошибка, но неэлегантно.

**[LOW] `deterministic_mode` — `cuda_states` фиксируется только на входе:**
- Если CUDA станет доступна внутри контекста (теоретически), `set_rng_state_all(None)` упадёт. Практически не случается. Игнорируемо.

### Нет HIGH-severity тихих ошибок в `cache.py` — код зрелый, валидация строгая.

---

## 3. Сводная таблица

| Файл | ТЗ-соответствие | Отклонения | Тихие ошибки (severity) |
|---|---|---|---|
| `exp38b_position_audit.py` | в основном ✓ | `align_K1` на `J^(p)` вместо `K` (**H**); CIs не paired (**H**); `l22_position_sensitive` — независимые CIs вместо paired (**H**); pos0 вычислен вместо 4.6 (**M**); нет cross-check pos0↔4.6 (**M**) | `clamp_min(1e-30)` → cos=0 молча при degenerate (**M**); `torch.eye` в цикле (**L**) |
| `cache.py` | ✓ | — | i.i.d. bootstrap на (doc × position) — использование в exp38b (**M**, документировать); `np.float32` не проходит `_check_value` (**L**) |

---

## 4. Минимальные правки (до публикации E-38b-результатов)

1. **P0 — paired bootstrap для body vs pos0** (и для L22-вердикта):
```python
# per-doc body: median over t in [16,126]
body_per_doc = torch.stack(body_pr_list).median(dim=0).values   # [32]
diff_lo, diff_hi = bootstrap_ci_paired(pos0_pr.numpy(), body_per_doc.numpy(),
                                        B=1000, seed=0)
if not (diff_lo <= 0 <= diff_hi):
    reasons.append("l22_position_sensitive")
```
Аналогично для `C_b` и `align_K1`.

2. **P0 — `align_K1` на `K = J − J^(p)`**, если нужна сопоставимость с E-37. Иначе — переименовать в `align_gf` (не вводить в заблуждение). Оба варианта — записать в `meta.naming`.

3. **P1 — degenerate-J guard**:
```python
nrm = Jf.norm(dim=1)
deg = nrm < 1e-6
# … cos_docs = torch.where(deg[:,None] | deg[None,:], nan, cos_mat) …
# затем torch.nanmedian
```
+ `nan_count` в `meta`.

4. **P1 — cross-check pos0 PR vs 4.6**:
```python
ratio = pos0_pr_med / PR_MODES_L22_PAPER
meta["pos0_PR_vs_paper"] = ratio
if abs(ratio - 1) > 0.2:
    warnings.warn(f"L22 pos0 PR = {pos0_pr_med:.2f} vs paper 4.6 "
                  f"(ratio {ratio:.2f}) — 4.6 may be a different definition")
```

5. **P2 — задокументировать в `cache.py` docstring `bootstrap_ci`** предупреждение про i.i.d. и/или добавить `bootstrap_ci_block(values, group_ids, ...)`.

6. **P2 — `torch.eye(N)` вынести** за цикл; `mask = ~torch.eye(N, dtype=torch.bool)` — один раз на слой.

---

## 5. Дополнительное наблюдение (кросс-документное)

`paper_old.md` §18 (E-37) пишет: «$align_{K1} = 0.71$–0.90 (11–14× истинного нуля)». Если E-38b-код даёт на L22 (pos0) другое значение (не 0.71), это **не** ошибка — потому что E-37 мерил на `K = J − P J`, а E-38b — на `P J`. Но в отчёте `exp38b_position_audit.json` это различие должно быть явно в `meta`, иначе читатель/ревьюер сделает ложный вывод о несогласованности E-37 vs E-38b. В текущем виде `meta.note` говорит только про `align_K1 = max_k |<u_k, 1/√K>| over LEFT singular vectors (E-37 u_g convention)` — не уточняет, что это на `J^(p)`, а не на `K`. **Добавить в `meta`: `"align_K1_target": "J^(p) (gauge-free J), NOT K = J − J^(p); NOT comparable to E-37 numbers"`**.

# Аудит `spectral.py` и `exp38c_degeneracy_preflight.py`

## 1. `spectral.py` — математическое ядро

### Соответствует математике (`paper_old.md` §6, §8) ✓

| Функция | Формула в коде | Формула в paper | Проверка |
|---|---|---|---|
| `stable_rank` | `Σλ / λ₁` | §6: `trace(JJᵀ)/λ₁` | ✓ |
| `participation_ratio` | `(Σλ)² / Σλ²` | §6: `(Σλ_k)²/Σλ_k²` | ✓ |
| `entropy_effective_rank` | `exp(−Σ p_k log p_k)`, `p_k=λ_k/Σλ` | §6 | ✓ |
| `eigengap_min` | `min_i(λ_i − λ_{i+1})`, threshold `= eps_frac·λ₁` | §6/E38c: `min_i(σ_i² − σ_{i+1}²)`, `threshold = eps_frac·σ_1²` | ✓ |
| `gauge_share` | `Σ_k (u_kᵀu)² λ_k / Σλ` | §6 | ✓ |
| `top_r_svd` | `U[:, :r], S[:r], V[:r, :]` | E-38d: «`V_r` формы `[r, D]`, строки = правые сингулярные» | ✓ |

**Дополнительно верно:**
- `stable_rank` docstring исправляет арифметическую ошибку paper §6 (`PR = 64/18 ≈ 3.555` → правильно `64/20 = 3.2`). ✓ Полезно — при переносе в статью пригодится.
- `snr_e14_parts` валидирован против `exp14_snr_pr.json` (бит-в-бит). ✓
- Sign convention — LAPACK-зависимость, не полагаться на raw sign — задокументировано и соответствует E-38d pitfalls. ✓
- `spectrum_from_gramian` **не** делает silently симметризацию (`atol=1e-6` + rtol default 1e-5 — не пропустит явно несимметричную Δ_U). ✓

### Отклонения / тихие ловушки

**[HIGH — архитектурная ловушка, не баг] `eigengap_min` семантика `well_separated` vs TZ `applicable`:**

TZ E-38c содержит **внутреннее противоречие**:
- Setup step 4: `applicable(t) = gap_min(t) >= threshold` — т.е. «gap большой ⇒ применим».
- Pre-registered: `min_t gap_min >= threshold` на всех слоях ⇒ «**не**абелев формализм **не**применим».

Первое — **инверсия** [M-4] из `paper_old.md`; второе — **правильно** по [M-4] (`gap_min < ε` = почти-вырождение = применим). Код **осознанно** разрешает противоречие: возвращает нейтральное `well_separated` и явно пишет в docstring: «It is NOT the [M-4] verdict… Callers decide the mapping». **Код прав, ТЗ противоречиво.**

**Но:** downstream-читатель JSON (включая будущего ревьюера статьи) может интерпретировать `fraction_well_separated` как `fraction_applicable` и получить **противоположный вердикт**. Это нельзя оставлять только в docstring — в `out["meta"]` должна быть явная фраза про направление, иначе вся серия будет miscited.

**[MEDIUM] `eigengap_min` возвращает `(gap_min, well_separated)` без `threshold_used`:**  
Caller вынужден реконструировать `threshold = eps_frac * spectrum[0]`, чтобы отчитаться в JSON. В `exp38c` это сделано через глобальную медиану (см. ниже), а не per-(t, doc). Возврат `(gap_min, well_separated, threshold)` снял бы класс ошибок.

**[MEDIUM] `eigengap_min` на `torch.float32` (не cast внутри функции):**  
`_check_1d_finite` проверяет finite, но `s[:-1] - s[1:]` при `s[15] ≈ σ_1² * 1e-3` в float32 даёт абсолютную погрешность ~`1e-7 · σ_1²` (у float32 `eps ≈ 1.2e-7`). Для `eps_frac = 0.005` это близко к границе. `spectral_from_gramian` **делает** `.double()` внутри; `eigengap_min` — **нет**. Асимметрия API. Рекомендация: либо `.double()` в `eigengap_min` при `eps_frac ≤ 0.01`, либо явно задокументировать float32 floor.

**[LOW] `top_r_svd` — silent NaN-пропагация:**  
Нет проверки `torch.isfinite(J).all()`. При заражённом кэше (уже случалось в E-27.6: NaN в диагонали `D` после бага `i != j` внутри `np.triu`) `svdvals` даст NaN → `sigma²` = NaN → `eigengap_min` **упадёт** на `_check_1d_finite` (громко). Приемлемо, но `top_r_svd` как публичный API стоит защитить.

**[LOW] `gauge_share` — dtype mismatch `uniform` vs `eigenvectors`:**  
`torch.full(..., dtype=eigenvectors.dtype)` — ок, но при смешанных dtype caller'ов (например, `eigenvectors` float64, `spectrum` float32) внутри `coeffs * spectrum` PyTorch промоутит без предупреждения. Не баг, но может маскировать источник потери точности.

---

## 2. `exp38c_degeneracy_preflight.py` — соответствие ТЗ E-38c

### Соответствует ТЗ ✓
- `sv(J_t) → σ_1..σ_16`, `gap_min = min_i(σ_i² − σ_{i+1}²)`, `threshold = eps_frac·σ_1²` — через `eigengap_min(lam[i], e)`. ✓
- `eps_frac = 0.01` main + sweep `{0.005, 0.01, 0.02}`. ✓
- L23 помечен `informative=False`, L18/L22 — `True`. ✓ (Architectural layer distinction сохранён.)
- Per-layer, не усреднено по слоям. ✓
- `median_t` / `median_doc` агрегации. ✓
- Verdicts: `non_abelian_not_applicable`, `near_degeneracy` (+ `near_degeneracy_layers`), `l23_architectural`. ✓
- Детерминировано, `default=float` в JSON. ✓

### Отклонения от ТЗ (severity)

**[HIGH] Verdict-логика использует **глобальную медиану** `σ_1²`, а не per-t `σ_1²`:**

```python
ref_threshold = EPS_MAIN * float(np.median(s1sq))     # median over ALL (t, doc)
...
all_not_applicable = all(r["min_t_gap_min_over_threshold"] >= 1.0 ...)
near_degen = [l for l, r in inf.items() if r["min_t_gap_min_over_threshold"] < 1.0]
```

ТЗ E-38c Setup step 3: `threshold = eps_frac · σ_1²` — **per (t, doc)**, а step 4: `applicable(t) = gap_min(t) >= threshold` — с этим per-t threshold. Pre-registered: `min_t gap_min >= threshold` — `threshold` в контексте per-t слоя, но по смыслу «на каждом t сравнить с локальным порогом, затем взять min по t».

Код **делает правильно** для `fraction_well_separated`:
```python
g, a = eigengap_min(lam[i], e)  # threshold внутри = e * lam[i,0] = e*σ_1²[i] per (t,doc)
```
но **verdict** считает отношение `min_t (median_doc gap_min)` к `median_{t,doc}(σ_1²)` — глобальный масштаб. Если `σ_1²(t)` сильно варьируется по t (провал у L22 — uniformity_0 = 0.504, E-33r; профиль в `s1sq_profile_t` для этого есть!), глобальная медиана даст **неверное** отношение.

Для текущего прогона на L18/L22 `min gap ratio = 0.08–0.09 << 1.0`, вердикт робастен к выбору порога. Но в отчёте `meta` не сказано, что вердиктная метрика использует **глобальный** масштаб — только `s1sq_profile_t` для диагностики. Правильнее:

```python
s1sq_t = np.median(s1sq, axis=1)             # per t median over docs
ratio_t = gap_t_med / (EPS_MAIN * s1sq_t)    # per t ratio
min_ratio = ratio_t.min()                    # <-- для verdict
```
+ сохранить и `min_ratio` (per-t), и `min_t_gap_min_over_threshold` (глобальный) в JSON с пометкой, какой из них используется в вердикте.

**[MEDIUM] Название переменной `appl_td` хранит `well_separated`, а не `applicable`:**

```python
appl_td = {e: np.zeros((Ss, N), dtype=bool) for e in EPS_SWEEP}
...
appl_td[e][t, i] = a   # a = well_separated, не applicable
...
"fraction_well_separated": {e: float(np.median(appl_td[e].mean(axis=1))) ...}
```

Code работает корректно (label в JSON — правильный `well_separated`), но **имя переменной** — «appl» — прямое приглашение будущему мейнтейнеру «исправить» инверсией и внести [M-4]-баг. Также `l23_appl` (см. ниже) содержит `well_separated`. Рекомендация: переименовать `appl_td → ws_td`, `l23_appl → l23_well_sep`, и добавить в код явный комментарий:

```python
# well_separated ≡ (gap_min >= eps*sigma_1^2). Per [M-4], formalism is
# APPLICABLE where this is False. Do NOT rename to "applicable".
```

**[MEDIUM] `l23_appl` содержит `well_separated`, но имя говорит `applicable`:**

```python
l23_appl = per_layer["23"]["min_t_gap_min_over_threshold"] >= 1.0
l1822_not = all(per_layer[l]["min_t_gap_min_over_threshold"] < 1.0 for l in ("18", "22"))
if l23_appl and l1822_not:
    verdicts["l23_architectural"] = True
```

- `l23_appl = True` означает «L23 well-separated» ⇒ по [M-4] **формализм НЕприменим** на L23.
- `l1822_not = True` означает «L18/L22 near-degenerate» ⇒ по [M-4] **формализм применим** на L18/L22.
- TZ pre-registered: «L23 **применим**, L18/L22 нет: **архитектурная особенность**».

Код fires verdict в противоположном направлении от TZ-фразы «L23 применим», но семантически соответствует **физике** L23 (lm_head — full-rank, `eigengap_{16/17} = 1.035`, E-35). Т.е. TZ pre-registered outcome #3 либо содержит ту же инверсию слова «применим», что и step 4, либо имеет в виду «L23 хорошо разделён — вот это архитектурная особенность». В любом случае — `l23_architectural` в текущем прогоне **не сработал** (L23 тоже near_degenerate с ratio 0.08–0.09 по статусу? Проверить: статус говорит `near_degeneracy (L18, L22)`, значит L23 в near_degen не попал, но `l23_appl` требует `ratio >= 1.0` на L23; если L23 тоже `< 1.0`, `l23_appl = False` и verdict не fires).

**[MEDIUM] Float32 SVD — потеря малых gap-ов в чувствительном sweep:**

`torch.linalg.svdvals(J[:, t])` — J в float32. Для σ_16/σ_1 ~ 1e-3 (типично для K=128, D=1024), `σ_16² ~ 1e-6 σ_1²`. Абсолютная точность float32 SVD ~ `eps_fp32 · σ_1 ≈ 1.2e-7 σ_1`, т.е. `σ²` точность ~ `2.4e-7 σ_1²`. При `eps_frac = 0.005` threshold = `5e-3 σ_1²` — gap-noise ~ `5e-5 · threshold` — ок. Но **gap_min** между σ_15² и σ_16² может быть ~ noise → значение `gap_min` смещено. Для `0.005` sweep это может дать false near-degenerate. Рекомендация: `svdvals(J[:, t].double())` — K=128, D=1024, N=32: ~0.05 с на t, × 128 t × 6 layers → +40 с, приемлемо.

**[LOW] Невекторизованный inner loop — 73k вызовов `eigengap_min`:**

`for i in range(N): for e in EPS_SWEEP: eigengap_min(lam[i], e)` — каждый вызов делает `torch.sort` на [16]. Векторизация:
```python
s, _ = lam.sort(dim=-1, descending=True)   # [N, R]
gaps = s[:, :-1] - s[:, 1:]
gap_min = gaps.min(dim=-1).values          # [N]
thr = s[:, 0]                              # [N]
ws = gap_min >= e * thr[:, None]           # [N] per e
```
~50–100× быстрее. Косметика для CPU-прогона 2 мин, но полезно для чувствительного sweep. **[LOW]**

**[LOW] `np.median(s1sq)` на большом массиве — ок, но `ref_threshold` единственный скаляр:**  
Как отмечено в MEDIUM выше — неверный масштаб для вердикта. Уже покрыто.

---

## 3. Сводная таблица

| Файл | ТЗ-соответствие | Отклонения | Тихие ошибки (severity) |
|---|---|---|---|
| `spectral.py` | ✓ математика верна | — | `eigengap_min` возвращает `well_separated` (нейтрально, но противоположно TZ `applicable`) — **H** для интерпретации downstream; `eigengap_min` без `.double()` при малых `eps_frac` — **M**; `top_r_svd` без finite-check — **L**; dtype-mismatch в `gauge_share` — **L** |
| `exp38c_degeneracy_preflight.py` | в основном ✓ | verdict использует **глобальную** медиану `σ_1²`, а не per-t — **H**; `l23_appl` имя инвертировано относительно TZ-fraзы — **M** | `appl_td` хранит `well_separated` — **M** (footgun); float32 SVD для `sweep@0.005` — **M**; невекторизованные циклы — **L** |

---

## 4. Минимальные правки до переноса E-38c в статью

1. **P0 — per-t threshold в verdict:**
```python
s1sq_t = np.median(s1sq, axis=1)                    # [S]
ratio_t = gap_t_med / (EPS_MAIN * s1sq_t)           # [S]
r["min_t_gap_min_over_threshold_per_t"] = float(ratio_t.min())
r["min_t_gap_min_over_threshold_global"] = float(
    gap_t_med.min() / (EPS_MAIN * float(np.median(s1sq))))
r["threshold_scale_used"] = "per_t"                 # для verdict
all_not_applicable = all(r["min_t_gap_min_over_threshold_per_t"] >= 1.0
                         for r in per_layer.values())
near_degen = [l for l, r in inf.items()
              if r["min_t_gap_min_over_threshold_per_t"] < 1.0]
```

2. **P0 — переименовать переменные (anti-footgun):**
  - `appl_td` → `ws_td`
  - `l23_appl` → `l23_well_sep`
  - В `out["meta"]` явно: `"well_separated ≠ applicable: per [M-4] formalism is APPLICABLE where well_separated=False; E-38c JSON reports well_separated (inverse of applicability)"`.

3. **P1 — `.double()` для `svdvals` в sweep `{0.005, 0.01, 0.02}`:**
```python
sv = torch.linalg.svdvals(J[:, t].double())  # [N, K]
sv16 = sv[:, :R].to(torch.float32)           # back to fp32 for downstream
```
+ `meta["svd_dtype"] = "float64"` — для ревьюера.

4. **P1 — `eigengap_min` возвращает `threshold_used`:**
```python
def eigengap_min(spectrum, eps_frac=0.01):
    ...
    threshold = float(eps_frac * s[0])
    return gap_min, gap_min >= threshold, threshold
```
Caller'ы (все текущие + будущие) явно видят, какой порог использован.

5. **P2 — векторизовать inner loop** (для воспроизводимости sweep при больших N в будущих E-38*, где N может быть 256+).

6. **P2 — добавить `eigengap_min(..., dtype="float64")` keyword** — или явно `.double()` внутри при `eps_frac ≤ 0.01`. Это устранит зависимость от того, что caller сам не забыл `.double()`.

---

## 5. Кросс-документные замечания

- **`paper_old.md` §6 E-28/E-35:** «`eigengap_{16/17}(M_K) = 1.035`» — это зазор между `σ_16` и `σ_17` **матрицы `M_K`**, не внутренние зазоры `J_t`. E-38c впервые измеряет внутренние — статус явно это фиксирует. ✓
- **`paper_old.md` [M-4]:** формулировка «`min_t gap_min < ε` → formalism применим» совпадает с E-38c pre-registered (не с TZ E-38c Setup step 4). Код следует [M-4]. В отчёте статьи нужно явно указать, что `well_separated` в E-38c — инверсия [M-4]-«applicable», чтобы не путать с E-35-«eigengap_16/17 non-degenerate».
- **Статус E-38c:** «near_degeneracy (L18, L22; min gap ratio 0.08–0.09) → неабелев формализм применим» — **согласован** с [M-4] и с кодом (при условии, что «min gap ratio» = per-t или global — не указано; с учётом ratio ≪ 1 вердикт не меняется). Рекомендую в `exp38_summary.json` сохранить **обе** метрики (`..._per_t`, `..._global`) для трассируемости.

# Аудит `exp38d_holonomy_spectrum.py`

Код в целом следует ТЗ E-38d, но имеет три содержательных проблемы (одна из них — **HIGH** на сценарий «молча пропущенных позиций») и несколько тихих ловушек. Ниже — послойно.

---

## 1. Соответствие ТЗ E-38d ✓

| Пункт ТЗ | Реализация |
|---|---|
| `V_t = top_r_svd(J_t, 16)[2].T` → `[D, 16]` | ✓ `.T.contiguous()` |
| `procrustes_align(V_i, V_j)` возвращает tuple, residual_sq = `[1]` | ✓ |
| `holonomy_spectrum([V_0, V_1, V_2], closed_loop=True)` | ✓ |
| `kappa = ‖H_012 − I‖_F` из матрицы, не из `{θ_j}` | ✓ `kappa_all` из `np.linalg.norm(Hn − I)` |
| `tr(H) = 2Σ cos θ_j`, `‖H−I‖²_F = 4Σ(1−cos θ_j)` — **точные** тождества | ✓ через `ang_all = np.angle(eigvals(Htot))` (все r, а не `r//2` — учтён lone −1) |
| `Ω(t) = geodesic_curvature(û_t, û_{t+1}, û_{t+2})` | ✓ |
| Degenerate step → исключить тройку | ✓ `DEGEN_RES_FRAC = 0.9·2r = 28.8` |
| Trivial `H ≈ I` → valid Abelian, не bug | ✓ `trivial[t,i] = th.max() < 1e-9` |
| L23 consistency vs E-28 `κ = 0.018` | ✓, диагностика с 5e-3 (rationale в docstring) |
| Deterministic | ✓ детерминизм (`spearmanr` — детерминирован, см. ниже) |

**Плюсы, которые стоит отметить:**
- `theta_all` NaN-padded, а не zero-padded — корректно для mixed spectrum (`nanmedian/nanmax` работают; zero-padding завысил бы `spread`). ✓
- `trH` считается из матрицы, а не из `2Σcosθ` — не ломается на случае «< r//2 углов». ✓
- `idiff` через **все r собственных значений** — корректно обходит lone −1 (не 8, а 4 в `‖H−I‖²`, но `2(1−cos π) = 4` — точно). ✓
- `consistency_e28`: `within_tol_5e-3` (pre-registered) отдельно от `within_tol_1e-3` (diagnostic), с обоснованием — честно. ✓

---

## 2. HIGH-severity проблемы

**[HIGH-1] Порог `DEGEN_RES_FRAC = 0.9·2r = 28.8` может исключить почти все нетривиальные тройки на L6–L22.**

`residual_sq = 2r − 2Σ σ_i(V_iᵀV_j) = 2r − 2Σ cos(principal angles)`.
E-28 даёт `Δ_SVD ≈ 1.549 rad ≈ 88.7°` на L6–L22 — т.е. **главный** principal angle ≈ π/2. Но `residual_sq` зависит от **всех** r углов.

Два режима:
- **(A) Top-углы π/2, остальные малы** (несколько общих мод): `Σ cos ≈ 3` → `residual_sq ≈ 32 − 6 = 26 < 28.8` → **не degenerate**, попадает в ρ.
- **(B) Все углы ≈ π/2** (полностью случайные подпространства): `residual_sq ≈ 31.3 > 28.8` → **degenerate**, исключено.

Статус E-38d рапортует `ρ(L18) = 0.24–0.25`, `ρ(L22) = 0.24–0.25` — значит, **часть** троек проходит. Но **`degenerate_frac` не упомянут** в статусе. Если на L18/L22 `degenerate_frac > 0.5`, ρ посчитан по патологически малой подвыборке (только «почти выровненные» тройки) — и вывод «третьего механизма» (ρ < 0.3) может быть артефактом отбора.

**Что делать:** в отчёте явно указать `degenerate_frac` по слоям (есть в JSON) и распределение нон-дегенератных `residual_sq`. Если медиана `residual_sq` на L6–L22 близка к 28.8 — порог произвольный и требует sweep {0.7·2r, 0.9·2r, 0.95·2r}.

---

**[HIGH-2] π-loss bug из `holonomy.py` взаимодействует с `trivial`/`spread`.**

Код ловит `wlist` (warnings), но **не инспектирует их**:
```python
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    th, Htot = holonomy_spectrum(...)
```

Комментарий: *"The 'R_total in w.message' branch was dead code"* — верно, но это не отменяет других warning'ов. В частности, ранее замеченный π-loss (`angles > 1e-6` теряет `−π`, когда imag = −0):
- `H = diag(−1, −1, 1, …, 1)` (180° rotation в 2D-плоскости), `κ = √8 ≈ 2.83`.
- `angles = [−π, −π, 0, …, 0]` (sign im = −0) → `positive = []` → `expanded = zeros(8)`.
- `th = [0]*8`, `float(th.max()) = 0 < 1e-9` → **`trivial = True` (WRONG)**.
- `theta_all[t, i] = zeros` → `spread` → `effectively_abelian` (WRONG — это 180° rotation).

**`idiff_max` этого не ловит** — identity `||H−I||² = 2Σ(1−cos ang_j)` выполняется для `ang_j = ±π` точно: `cos(±π) = −1`, `2(1−(−1)) = 4`; `κ² = 8 = 4 + 4`. Значит, `idiff ≈ 0` даже при потерянных θ. Только `kappa_median` (из матрицы) корректен.

На текущем прогоне L23 (rigid, κ ≈ 0.016) π-ротаций, скорее всего, нет — вердикт не пострадал. Но:
- `trivial_frac` в JSON **завышен** для любой будущей серии, где H имеет большие углы;
- при переиспользовании `exp38d` в edit-сценариях (E-38h/i follow-up) риск реального miscategorize.

**Что делать:**
1. Исправить `holonomy.py` (`angles.abs() > 1e-6`, работать с `abs(angles)`).
2. В `exp38d`: **инспектировать `wlist`** — если у warnings есть «expected X angles, got Y» или «split across ±π boundary» — инкрементить `anomaly_count` в out. Это громкая защита от π-loss без зависания на диагностику.
3. Верифицировать: для каждой (t, i), где `kappa² > 0.1` и `th.max() < 1e-6`, записывать в `meta.possible_pi_loss_count`.

---

**[HIGH-3] `json.dump(..., default=float)` — NaN даёт невалидный JSON.**

```python
json.dump(out, open(OUT, "w"), indent=1, default=float)
```
- `default` вызывается только для **не-сериализуемых** объектов. `float('nan')` сериализуется как литерал `NaN` (Python `json` default `allow_nan=True` — не RFC 8259).
- Значения `theta.median_per_j` (для j без данных → `nanmedian → NaN`), `rho_med = NaN` при пустом `rho` — **попадают в JSON как `NaN`**.
- `cache.py` использует `_json_dump_value` для того же — консистентность нарушена.
- Многие downstream-парсеры (Python strict, Rust `serde_json`, JS `JSON.parse`) **отвергнут** такой файл.

**Что делать:** либо явно `_json_dump_value` (как в `cache.py`) для каждой float-конвертации, либо `allow_nan=False` с предварительным `float('nan') → None` в полях, где NaN допустим, либо `default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o`.

---

## 3. MEDIUM-severity проблемы

**[M-1] `Ω(t)` может упасть на нулевой норме `h`.**
```python
u = H / norms.clamp_min(1e-12)[:, :, None]
...
omega_all[t, i] = geodesic_curvature(u[i, t], u[i, t + 1], u[i, t + 2])
```
Если `‖h_t‖ ≈ 0` (attention sink pos 0, как в E-38a), `u = 0`, `geodesic_curvature` бросает `ValueError` (unit-check). Не обёрнуто в try/except — **валит весь прогон слоя**. E-38a на pos 0 прошёл только потому, что там clamp дал не-нулевое значение; на L22/pos0 в другой момент может не повезти.

**Рекомендация:** pre-filter `min(‖h_t‖, ‖h_{t+1}‖, ‖h_{t+2}‖) < 1e-10` → `omega_all[t,i] = np.nan`; сохранить `n_omega_skipped` в out.

---

**[M-2] `H012_save` тихо кастуется в float32, но комментарий обещает 8 байт (float64).**
```python
H012_save.append(Htot.float().clone())      # каст в float32
...
# ~126x16x16x8B  ← 8B = float64
```
`Htot` — float64 `[16,16]`. `.float()` → float32. При future использовании (κ_fiber через eigenframe) loss of precision на 7 значащих цифр.

**Рекомендация:** `H012_save.append(Htot.clone())` (keep float64). 126×16×16×8B ≈ 258 KB → JSON ~500 KB, приемлемо.

---

**[M-3] Docstring врёт про permutation-тест.**
> *"Spearman permutation p uses seed=0."*

`scipy.stats.spearmanr` — детерминированная, **не** permutation. `np.random.seed(SEED)` в начале **не влияет** на `spearmanr`. P-value (возвращается как второй элемент, но отбрасывается `r_, _ = ...`) считается асимптотической t-аппроксимацией.

**Рекомендация:** убрать фразу; сохранять `p_med = median(p_values)` в out — при N ≈ 32 non-degenerate docs это полезная статистика.

---

**[M-4] `np.histogram(theta_flat, ...)` тихо отбрасывает NaN.**
`theta_flat` содержит NaN (для позиций с mixed spectrum). `np.histogram` с NaN — все сравнения False → NaN не попадает ни в один bin, `sum(hist) < theta_flat.size`. Информационная потеря.

**Рекомендация:** `n_dropped_nan = theta_flat.size - hist.sum()` → out. Или считать отдельный `frac_trivial_or_short` = доля позиций с < 8 углов.

---

**[M-5] `theta_all[t, i, :len(th)] = th` может бросить shape-error, если `len(th) > 8`.**
`holonomy_spectrum` в degenerate-спектре может вернуть `expanded.numel() > expected` (например, каждая пара углов встречается 2× → `mult = counts = 2`, `expanded` содержит duplicates). Warning есть, но length может превысить 8 → **runtime error**, не silent. В текущем прогоне не наблюдалось (`κ` мал → `θ_j` уникальны). Но с **`wlist` не инспектируется** — если такой триггер случится, крах произойдёт посреди слоя.

**Рекомендация:** `assert len(th) <= R//2, f"unexpected theta length {len(th)}"` с явным логом.

---

## 4. LOW-severity

**[L-1] `trivial[t,i] = float(th.max()) < 1e-9`** — работает при `th = zeros(8)` (R_total ≈ I), но fragile: зависит от того, что `holonomy_spectrum` **возвращает zeros правильной длины** в trivial-случае, а не пустой массив. С π-loss фиксом (abs > 1e-6) — будет корректно. Без фикса — риск `trivial` для 180° ротаций.

**[L-2] `spearmanr(trH_all[t, m], omega_all[t, m])` — фиксированный seed не нужен; `t % 4`-сэмплирование позиций (32 позиции из 126) — произвольное, не обосновано.** Нет permutation CI для ρ. Для pre-registered «ρ > 0.7» / «ρ < 0.3» — вероятно OK, но ρ = 0.24 на грани `partial` (не `third_mechanism < 0.3`), и CI мог бы попасть в `partial` — вердикт чувствителен к шуму. В статусе L18/L22 = «третьего механизма **не** нашли» (ρ = 0.24–0.25, `ρ < 0.3` → third_mechanism) — то есть статус **утверждает** third_mechanism, а запись в статусе «третьего механизма не нашли» — противоречие между кодом и статусом. Стоит пере-проверить: что именно попадает в `verdicts["18"]` и `verdicts["22"]`.

**[L-3] `float(th.max())` при `th` пустом** — как обсуждалось, маловероятно при `r=16`, но незащищено.

**[L-4] `rho_med = NaN` при пустом `rho` — попадает в JSON как `NaN`.** См. HIGH-3.

---

## 5. Кросс-проверка со статусом E-38d

Статус: *«L18/L22: non_abelian_candidate, ρ(tr H, Ω) = 0.24–0.48 (сферическая геометрия ρ > 0.7 НЕ подтверждена; максимум L6 0.48, L18/L22 0.24–0.25)»*.

Согласуется с кодом:
- `spread > 0.5` → `non_abelian_candidate` — но статус говорит `non_abelian_candidate` для L18/L22; это верно только если `spread > 0.5`. Учитывая `κ ≈ 0.5–1.5` на L6–L22 (E-28), `{θ_j}` шире нуля, но **не** «один θ ≫ других» (E38.md ждёт «один θ_1 ≫ θ_2..θ_8» для неабелева). `spread > 0.5` — слабый критерий; статус корректно говорит «кандидат», не «неабелев».
- `rho < 0.3` → `third_mechanism` — статус «сферическая геометрия НЕ подтверждена» ✓. Но в самом статусе фраза «третьего механизма не найден» — **противоречит** коду (`ρ < 0.3` = `third_mechanism` label). Либо статус неточен, либо в коде `third_mechanism` — это positive label для «сферическая не работает», и статус его переформулирует. **Проверить:**

```python
v.append("spherical_geometry" if rho_m > 0.7
         else ("third_mechanism" if rho_m < 0.3 else "partial"))
```

`rho_m = 0.24` → `third_mechanism` (literal). Статус пишет «не подтверждена» — это интерпретация `third_mechanism` как «κ измеряет что-то третье», что по ТЗ = «интерпретация требует пересмотра» (не «не найден»). Так что статус и код расходятся в семантике ярлыка. Стоит зафиксировать в `docs/E38_fixes.md`, что `third_mechanism ≡ "не сферическая"`, а не «третий механизм идентифицирован».

---

## 6. Минимальные правки (приоритет)

**P0:**
1. **`holonomy.py`**: `angles.abs() > 1e-6`, работать с `abs(angles)` — устраняет π-loss.
2. **`exp38d`**: инспектировать `wlist` — счётчик `anomaly_count` (сколько warnings на слой).
3. **`exp38d`**: `json.dump` → кастомный `_json_default` для NaN/Inf → `"nan"/"inf"` (или `allow_nan=False` с pre-conversion).
4. **`exp38d`**: явно отчитаться по `degenerate_frac` (уже в JSON, но нужно пере-проверить: не > 0.5 ли на L6–L22); при необходимости — sweep порога `{0.7·2r, 0.9·2r, 0.95·2r}`.

**P1:**
5. `Ω`: pre-filter `‖h‖ < 1e-10` → NaN.
6. `H012_save`: сохранять float64 (`.clone()` без `.float()`).
7. `assert len(th) <= R//2` перед записью в `theta_all`.
8. Сохранять `p_med` от `spearmanr`.

**P2:**
9. `np.histogram` — `n_dropped_nan` в out.
10. Docstring: убрать «permutation p uses seed=0».
11. `trivial` — после π-fix пере-проверить порог `1e-9` (маловероятно, что π-ротации редкие; для L23 — почти наверняка нет, для будущих серий — возможно).

---

## 7. Итоговая сводка

| Категория | Проблема |
|---|---|
| **HIGH** | порог `28.8` может исключить большинство L6–L22 троек → ρ на предвзятой подвыборке (нужен `degenerate_frac` и sweep) |
| **HIGH** | π-loss в `holonomy.py` → `theta`/`trivial`/`spread` молча неверны для 180° ротаций; `wlist` не инспектируется |
| **HIGH** | NaN в JSON (`json.dump(..., default=float)` не конвертирует NaN) — невалидный JSON для downstream |
| MEDIUM | `Ω` не guard'ируется от `‖h‖ ≈ 0` (крах слоя) |
| MEDIUM | `H012_save` кастуется в float32 (комментарий врёт про 8B) |
| MEDIUM | docstring врёт про permutation seed (не используется) |
| MEDIUM | `p`-value от `spearmanr` отбрасывается |
| MEDIUM | `np.histogram` тихо отбрасывает NaN |
| MEDIUM | `theta_all[.., :len(th)] = th` может бросить shape-error в degenerate-spectrum случае |
| LOW | `trivial` fragile на пустом `th`; `t % 4` не обосновано; `third_mechanism` label семантически неоднозначен |

# Аудит `exp38e_MK_diagnostic.py`, `jstruct.py`, `interventions.py`, `jacobian.py`

Разбивка по файлам, severity: **HIGH** (искажает научный вывод), **MEDIUM** (может дать ложный/смещённый результат), **LOW** (стиль/edge-case).

---

## 1. `exp38e_MK_diagnostic.py` — **главный файл серии E-38e**

### 🔴 HIGH-1: Consistency check `cons_ratio` — **неверная нормализация, статус `consistency_redflag` скорее всего ложно-положительный**

Код:
```python
sigma_g = float(torch.linalg.svdvals(M_K)[0])          # σ₁(M_K), NOT σ_g(J)
...
beta_scaled = (sqrtD * beta_vec / norms)               # c·β, c = √D/r_t
cons_ratio = float(np.median(beta_scaled.numpy()) / sigma_g)
cons = "consistency_consistent_with_E33r" if abs(cons_ratio - 1.0) <= 0.15 else "consistency_redflag"
```

Из [L-gauge-J] (paper_old.md Прил. E.1, п. 3): `c·β = ‖J^T(1/√K)‖ = σ_g(J)·cos(u_g, 1/√K)`, где `σ_g(J)` — сингулярное число **J**, не `M_K`.

Соотношение: `σ_g(J) ≈ c·σ₁(M_K)` (поскольку `J = c·M_K·(I − ûû^T)` и `û` не в топ-направлении `M_K` — подтверждено `|α|/|β| = 0.195`).

Тогда:
```
cons_ratio = c·β / σ₁(M_K) = σ_g(J)·cos / (σ_g(J)/c) = c·cos ≈ 0.2–0.4
```
**Ожидаемое значение — `c·cos`, а НЕ 1.** Порог `|cons_ratio − 1| ≤ 0.15` требует `cons_ratio ∈ [0.85, 1.15]`, что при `c ≈ 0.25` **физически невозможно**.

Статус E-38e: `consistency_redflag ((√D/r)|β|/σ_g = 0.216, β/‖Jᵀu‖=4.08)`. Ratio 0.216 — это ровно `c·cos` для `c ≈ 0.24`, `cos ≈ 0.9` — **согласуется с точной факторизацией**, а не с её нарушением.

**Что делать:**
```python
# Правильный нормализатор — σ_g(J) ≈ (√D/r_t)·σ₁(M_K); считаем per (t, doc):
r_t = norms[i, t]
sigma_g_J = (sqrtD / r_t) * sigma_g          # σ_g(J) локально
cons_ratio = median(c·β / sigma_g_J)         # = median(cos(u_g, 1/√K)) ≈ 0.9
cons = "consistency_ok" if abs(cons_ratio - 0.9) <= 0.15 else "consistency_redflag"
```
Или проще: считать `cons_ratio = median(c·β) / median(σ₁(J))` где `σ₁(J)` из `svdvals(J23[i,t])` (можно раз в `per_doc` цикле).

**Риск:** статус E-38e `consistency_redflag` — этот вывод сейчас стоит в `paper_old.md §21` (desync) и в `E38.md` Summary как свидетельство нарушения M-1. Если проверка ложная — вывод требует пере-проверки.

---

### 🔴 HIGH-2: Комментарий к `β/‖J^T u‖ = 4.08` **неверен** (но сам факт — признак точной факторизации)

```python
"consistency_factorization": {
    "median_beta_over_JtU": float(np.median(beta_vs_JtU.numpy())),
    "note": "β_t / ‖J_t^T 1/√K‖; = 1 under exact "
            "J = M_K(I − û û^T)"},                    # ← WRONG
```

Из [L-gauge-J] `J^T·(1/√K) = c·β·v_⊥` ⇒ `‖J^T(1/√K)‖ = c·β`. Тогда `β / ‖J^T u‖ = 1/c = r_t/√D`, **не 1**.

Статус: `β/‖Jᵀu‖=4.08` ⇒ `r_t/√D = 4.08` ⇒ `r_t = 4.08·32 ≈ 130` — типичное значение для L23 (RMSNorm выход, `σ_d = 96–104` по E-27.6, но per-position `r_t` может быть больше). То есть **`4.08` — это ровно признак точной факторизации**, а не её нарушения!

**Что делать:** исправить комментарий:
```python
"note": "β / ‖J^T (1/√K)‖ = r_t/√D under exact J = M_K(I − ûû^T) "
        "(not 1); status 4.08 → r_t ≈ 130, consistent with L23 scale"
```

И в статусе E-38e фраза «β/‖Jᵀu‖=4.08» должна интерпретироваться как **подтверждение** факторизации, а не как дополнительный «redflag». Тогда E-38e и E-38f могут указывать в разные стороны — что важно проверить.

---

### 🔴 HIGH-3: `token_set = ck23["U128"]` — нет проверки на соответствие модели

```python
token_set = ck23["U128"]          # из exp27b cache
assert len(token_set) == 128      # ← проверяет только длину
...
W_lm_full = _resolve_lm_head_weight(model)       # [V, D]
M_K = build_M_K(W_lm_full, token_set, g)         # build_M_K проверяет 0 ≤ idx < V
```

`build_M_K` проверяет `0 ≤ idx < V`, но **не** проверяет, что `W_lm_full[ts]` — семантически те же токены, что при построении кэша. Если модели имеют разные tokenizer-версии или cache-файл построен на другой модели (Qwen3.5-0.8B vs 0.8B-b, или т.п.), — индексируются неверные строки **молча**.

**Что делать:**
```python
V_model = W_lm_full.shape[0]
V_cache = ck23.get("V", None) or V_model    # если кэш хранит V
assert V_model == V_cache, f"vocab mismatch: model {V_model} vs cache {V_cache}"
# Плюс sanity: min/max idx
assert 0 <= min(token_set) and max(token_set) < V_model
out["meta"]["token_set_range"] = [min(token_set), max(token_set)]
```

---

### 🟡 MEDIUM-4: `pos0_ratio` может быть смещён attention-sink

```python
pos0_ratio = float(ratio[:, 0].median())
```
Если `‖h_0‖ ≈ 0` (attention sink): `u = 0`, `α = w·0 = 0`, `β = ‖w − 0‖ = ‖w‖`, `ratio = 0`. Медиана по 32 docs может быть «размыта» несколькими документами с маленьким `‖h_0‖`.

**Что делать:**
```python
norms0 = norms[:, 0]
valid_pos0 = norms0 > 1e-6           # отбрасываем degenerate
pos0_ratio = float(ratio[valid_pos0, 0].median())
out["meta"]["pos0_degenerate_frac"] = float((~valid_pos0).float().mean())
```

---

### 🟡 MEDIUM-5: `norms.clamp_min(1e-12)` — silent floor

Тот же паттерн, что E-38a/b/c/d: `‖h‖ ≈ 0` не даёт `NaN` и не падает, а даёт `u = 0`, `α = 0`, `β = ‖w‖`. Это тихо попадает в медианы. Рекомендация: pre-filter `norms < 1e-6` → `NaN`, `np.nanmedian` в агрегате, `nan_frac` в meta.

---

### 🟢 LOW-6: `torch.ones(128) / np.sqrt(128.0)` — mixed float/numpy

```python
w = (M_K.T @ (torch.ones(128) / np.sqrt(128.0)))
```
`torch.ones(128)` — float32; `np.sqrt(128.0)` — numpy float64. PyTorch type-promotion скорее всего даст корректный результат, но стилистически лучше:
```python
u_K = torch.full((128,), 1.0 / math.sqrt(128), dtype=M_K.dtype)
w = M_K.T @ u_K
```

---

## 2. `jstruct.py` — вспомогательный модуль

### 🟢 LOW-1: `decompose_w_vs_uhat` — `β ≤ 1e-12` → zeros

```python
if beta > 1e-12:
    v_perp = r / beta
else:
    v_perp = torch.zeros_like(w)
```
`zeros_like(w)` — не единичный вектор, но `‖zeros‖ = 0`. Downstream должен проверять `β > 0`. Документировано. OK, но edge-case `β = 0` (w ∥ û) — особая точка: если β = 0, `cos(v_g, v_pred)` из E-38f непредсказуем (v_pred = 0).

### 🟢 LOW-2: `v_pred_from_JT` — silent normalization non-unit `uniform`

Docstring: «A non-unit uniform is normalized silently». Если caller передаёт случайно не единичный uniform, тихо работает. Рекомендация: `if abs(n − 1.0) > 0.1: warn`.

### 🟢 LOW-3: `v_pred_from_JT` — `eps=1e-10` для degeneracy check

```python
if nx < eps:   # eps = 1e-10
    raise RuntimeError(...)
```
Слишком слабый порог: `nx = 1e-9` даст нормализацию шумного вектора, но не упадёт. Для `r_t ~ 100` и `β ~ 0.1` типичный `nx = c·β ~ 0.02` — далеко от порога. Но при малом `r_t` (short contexts) может сработать. Рекомендация: `eps=1e-6` или относительный `eps = √(eps_fp32) ≈ 3e-4 · max(1, ‖J‖)·‖uniform‖`.

---

## 3. `interventions.py` — инфраструктура для E-38h/i (не E-38e)

### 🟡 MEDIUM-1: `_tensor_fingerprint` — коллизии возможны

```python
s = int(t.sum().item() * 1e9) & 0xFFFFFFFF    # 32-bit truncation
m = int(t.abs().max().item() * 1e9) & 0xFFFFFFFF
```
Используется в `meta["target_J_p_hash"]`. Коллизии → cache invalidation fails silently. Рекомендация: SHA-256 от tensor bytes (`t.cpu().numpy().tobytes()`), или явно пометить `"diagnostic_fingerprint_not_unique"` в meta.

### 🟡 MEDIUM-2: `_APPLY_EDIT_ACTIVE` — module-level global blocks threads

```python
_APPLY_EDIT_ACTIVE = False
...
global _APPLY_EDIT_ACTIVE
if _APPLY_EDIT_ACTIVE:
    raise RuntimeError(...)
_APPLY_EDIT_ACTIVE = True
```
Concurrency across threads (e.g. multi-threaded server) → все `apply_edit` serialize/error. Для batch-экспериментов OK; для production — проблема. Рекомендация: заменить на `threading.local()` или `contextvars.ContextVar`.

### 🟡 MEDIUM-3: `apply_edit_and_cache` — нет `attention_mask.all()` check

```python
p_t = F.softmax(out.logits[0][:, ts].float(), dim=-1)   # [S, K]
```
`out.logits[0]` включает padded positions (attention_mask=0) — `p_t` для них тихо нормализуется к «мусорным» значениям. TZ говорит «only unpadded single sequences», а `compute_jacobian_for_tokens` этот check делает (`attention_mask.all()`). `apply_edit_and_cache` — нет.

**Что делать:**
```python
for sample in samples:
    if not bool(sample["attention_mask"].all()):
        raise ValueError("apply_edit_and_cache: unpadded single sequences only")
```

### 🟢 LOW-4: `make_readout_pull` — Adam state на full `[V, D_S]`

```python
params = [weight]   # full [V, D_S]
```
Docstring честно: «~1200× savings apply to persisted delta, not optimizer state». Adam держит `m`, `v` на `[V, D_S]` (~1.2 GB для 0.8B). Для CPU-прогона — заметный memory hit. Документировано. ✓

---

## 4. `jacobian.py` — низкоуровневая инфраструктура

### 🟡 MEDIUM-1: `_extract_hidden_state` — convention `hidden_states[layer_idx + 1]`

```python
h = outputs.hidden_states[layer_idx + 1][:, -1:, :]   # output of layer layer_idx
```
Это `h^{layer_idx+1}` в терминах hidden_states, но с комментарием «[l+1] = output of layer l» — то есть это **выход слоя `layer_idx` = вход в слой `layer_idx+1`**.

Вопрос: соответствует ли это `kaggle_kernel/results/exp27b_jcache_L*.pt` поле `H`? Если E-27 строил `H` из `hidden_states[layer_idx]` (вход слоя `layer_idx`), то E-38* имеет off-by-one. Проверить не могу (файла нет), но **проверка возможна**:

```python
# Sanity check для L23 raw J = c·M_K·(I−ûû^T) требует û = INPUT к L23
# Тогда ‖J·û‖ ≈ 0 (проектор убивает û)
resid = (J23[i,t] @ u[i,t]).norm() / J23[i,t].norm()
# Если resid << 1 — convention правильная; если resid ~ 1 — off-by-one.
```

Статус E-38e `|α|/|β| = 0.195` **физически разумен** только если û — правильный вход L23. Значит, вероятно, convention правильная. Но **проверить явно** стоит.

### 🟡 MEDIUM-2: `_deep_clone_cache` — `isinstance(v, list)` может пропустить non-list containers

```python
for k, v in vars(cache).items():
    if k == "layers" or isinstance(v, list):    # tuple/ModuleList silently copied
        continue
```
Если `layers` — `tuple` или `nn.ModuleList` (версия-зависимо), `deepcopy` упадёт или тихо сработает. Рекомендация: `if k == "layers" or hasattr(v, "__iter__") and not isinstance(v, (str, bytes))`.

### 🟡 MEDIUM-3: `_clone_gdn_states_on_entry` — positional `initial_state` ломается

```python
def rec(q, k, v, g, beta, *args, initial_state=None, output_final_state=True, ...):
    ...
    return orig_rec(q, k, v, g, beta, *args, initial_state=initial_state, ...)
```
Если оригинал зовётся с `initial_state` **позиционно** (`orig_rec(q,k,v,g,beta, state, ...)`), wrapper получит `state` в `*args`, `initial_state=None`, и передаст `orig_rec(..., *args, initial_state=None, ...)` → **TypeError** (multiple values). Это **LOUD**, документировано («fails loud»). Но: если версия transformers передаёт `initial_state` позиционно и caller ожидает silent cloning — получит крах. Рекомендация: явный `if len(args) > 0: raise NotImplementedError("positional initial_state not supported")` с понятным сообщением.

### 🟡 MEDIUM-4: `h.grad_fn is None` + `requires_grad=False` — silent detach

```python
if h.grad_fn is None:
    if h.requires_grad:
        pass                                    # use leaf in place
    else:
        h = h.detach().requires_grad_(True)     # NEW leaf — gradient to caller's tensor is LOST
```
Если caller ожидал накопления `.grad` на их `h` и передал leaf без `requires_grad=True` (забыл), — gradient тихо не пойдёт к их tensor. Документировано, но `_forward_h_norm` (вызывается через `compute_hidden_jacobian_rows`) делает то же:

```python
if h.grad_fn is None:
    h = h.detach().requires_grad_(True)
```
**Что делать:** для `h_override` в `compute_jacobian_for_tokens` — есть явный `_normalize_h_override` с проверками; для `compute_hidden_jacobian` — нет. Рекомендация: аналогичная валидация.

### 🟢 LOW-5: `_run_decoder_pass` — `attention_mask=None` в `_build_causal_masks`

```python
"attention_mask": None,   # в mask_kwargs
```
Работает для Qwen3.5 (проверено), но модели с обязательным mask — silent error. Документировано как version-dependent.

---

## Сводная таблица

| Файл | ТЗ | Отклонения | Тихие ошибки |
|---|---|---|---|
| **exp38e** | ✓ (с σ_g-ноансом) | σ_g check — **H** (неверная нормализация, ложно-положительный `consistency_redflag`); `β/‖J^Tu‖=1` комментарий — **H** (на деле = r_t/√D, 4.08 подтверждает точную факторизацию) | `token_set` из cache не валидирован — **H**; `pos0_ratio` без guard — **M**; `norms.clamp_min(1e-12)` — **M** |
| **jstruct.py** | ✓ | — | `β ≤ 1e-12` zeros — **L**; silent normalization `uniform` — **L**; `eps=1e-10` weak — **L** |
| **interventions.py** | ✓ | — | `_tensor_fingerprint` collisions — **M**; `_APPLY_EDIT_ACTIVE` global blocks threads — **M**; `apply_edit_and_cache` без `attention_mask.all()` — **M** |
| **jacobian.py** | ✓ | — | `_extract_hidden_state` off-by-one convention — **M** (кросс-файловая проверка); `_deep_clone_cache` `isinstance(list)` — **M**; positional `initial_state` TypeError — **M** (LOUD); `h.grad_fn is None` silent detach — **M** |

---

## Минимальные правки (приоритет)

**P0 — до переноса E-38e в статью:**
1. **Исправить `consistency check`** (нормализатор `σ_g(J)` = `(√D/r_t)·σ₁(M_K)`, ожидаемое ≈ `cos(u_g, 1/√K) ≈ 0.9`, толеранс 0.15). Пере-запустить E-38e; если `consistency_ok` — статус `consistency_redflag` отозвать (или зафиксировать как «ложное срабатывание старой проверки»).
2. **Исправить комментарий `β/‖J^T u‖ = 1`** на «= r_t/√D» и в статусе интерпретировать 4.08 как **подтверждение** факторизации, а не как нарушение.
3. **Валидация `token_set` vs model vocab** (assert `V_model == V_cache`).

**P1:**
4. `pos0` guard и `norms < 1e-6` → NaN + `nan_frac`.
5. `apply_edit_and_cache` — `attention_mask.all()` assert.
6. `_deep_clone_cache` — `hasattr(v, "__iter__") and not isinstance(v, (str, bytes))`.
7. `_extract_hidden_state` convention — cross-check `‖J·û‖/‖J‖` в E-38e/E-38f (закрывает off-by-one).

**P2:**
8. `_tensor_fingerprint` → SHA-256 (или документировать «диагностика»).
9. `_APPLY_EDIT_ACTIVE` → `threading.local()` если возможна многопоточность.
10. `jacobian.py` — `h_override` валидация в `compute_hidden_jacobian` (аналог `_normalize_h_override`).

---

## Кросс-документное замечание

Статус E-38e (`consistency_redflag`) и статус E-38f (`lemma_broken: Δcos = −0.224`) **ссылаются на одну гипотезу [M-1]**, но:
- E-38f — **прямой** тест `cos(v_g, v_pred) vs cos(u_g, 1/√K)` — надёжен.
- E-38e — **косвенный** тест через σ_g — вероятно, false-positive из-за бага нормализации.

Если после фикса E-38e покажет `consistency_ok`, а E-38f всё равно `lemma_broken` — гипотеза [M-1] **нарушена в прямом смысле** (v_g ≠ J^Tu направление), но σ_g-соотношение держится. Это тонкий, но важный нюанс для статьи: «gauge-направление v_g и униформ-направление 1/√K связаны через σ_g, но соответствующий `v_⊥` из `J^Tu` не коллинеарен `v_g`» — это третий режим, не покрытый [M-1], и стоит его выделить.

# Аудит `exp38f_structural_consistency.py`

## 🔴 HIGH-1: Ключевой баг `U @ one` вместо `U.T @ one` — **производит ложный `lemma_broken`**

```python
c = (U @ one).abs()                 # ← BUG
k = int(c.argmax())
ug = U[:, k]
vg = Vt[k]
```

`U` — [K, K], `U[:, j]` — j-й левый сингулярный вектор (по конвенции `torch.linalg.svd`). Нужно `c_k = u_k · 1/√K = Σ_i U[i, k]·one[i] = (U.T @ one)[k]`.

Код же считает `(U @ one)[i] = Σ_j U[i, j]·one[j]` — смешивает **компоненту i** и **индекс сингулярного вектора j**. Это не `u_k · one`, а «строка i матрицы U, свёрнутая с one». Смысл — «row-overlap с uniform», не «svector-overlap с uniform».

Следствия:
1. **`c.argmax()` выбирает не тот k.** `ug = U[:, k]` — валидный левый сингулярный вектор, но **не тот**, что максимизирует `|u_k·1/√K|`.
2. **`cos_ug_uniform` репортится неверно.** Reported `0.2286` — это `c[k] = row_k(U)·one`, а не `|u_g·1/√K|`.
3. **`v_g = Vt[k]` не тот, что определён в paper §8 / TZ.** `cos(v_g, v_pred) = 0.0044` — валидное |cos| двух векторов, но для «не того» v_g.

**Числовая проверка через независимые данные:** E-10a для L23 даёт `uniformity_0 = 0.503` ⇒ `|u_1 · 1/√K| = √0.503 ≈ 0.71` для **top-моды**. Если `u_g ≈ u_1` (что E-33r утверждает: `align = 0.989`), то после фикса `c_g` должно быть **≈ 0.71**, не 0.23.

**Алгебраическая проверка через [L-gauge-J]:** для любого k, `cos(v_k, v_pred) = σ_k c_k / ‖J^T u‖ ≥ σ_k c_k / σ_1`. При g = 1 (top-мода): `cos(v_g, v_pred) ≥ c_g`. То есть после фикса `cos(v_g,v_pred) ≥ 0.71`, и `Δcos ≥ 0` (**положительный**). Текущий `Δcos = −0.224` — знак, несовместимый с леммой → **результат от бага, а не от нарушения леммы**.

**Fix:**
```python
c = (U.T @ one).abs()   # c_k = u_k · 1/√K
k = int(c.argmax())
ug = U[:, k]            # k-й левый svector
vg = Vt[k]              # k-й правый svector
cos_ug_one = float(c[k])   # теперь = |u_g · 1/√K| — правильно
```

После фикса ожидаемо: `cos_ug ≈ 0.71`, `cos_vg_vp ∈ [0.71, 1]`, `Δcos ∈ [0, 0.29]` — вердикт сместится в `lemma_confirmed` или `intermediate`.

**Кросс-следствие для статьи:** статус E-38f (`lemma_broken: Δcos = −0.224`) и вывод №1 в `E38.md` Summary («Унификация M-1 **не подтверждается**») **основаны на этом баге**. После фикса E-38f вывод может **развернуться**. Аналогично E-38e (`consistency_redflag`), где я нашёл независимый баг нормализации. **Обе опоры главного вывода — с багами.**

---

## 🔴 HIGH-2: Pre-registered `Δcos > +0.1 = methodological_anomaly` — **слишком строго, противоречит [L-gauge-J]**

По лемме: `Δcos = cos(v_g, v_pred) − c_g ∈ [0, 1 − c_g]`. При `c_g = 0.3` лемма разрешает `Δcos ≤ 0.7`. TZ-правило «Δcos > 0.1 = аномалия» отвергнет **валидные** результаты.

Только `Δcos < 0` — действительно невозможно при точной факторизации (это то, что нужно пометить как `lemma_broken`). Всё `≥ 0` — совместимо с леммой (с разной степенью «доминирования top-моды»).

**Рекомендация:** переформулировать в `meta` и в pre-registered интерпретации:
- `Δcos < −0.05`: lemma_broken (несовместимо с точной факторизацией).
- `0 ≤ Δcos ≤ 1 − c_g + ε`: lemma_confirmed (в допустимом диапазоне).
- `Δcos > 1 − c_g + ε`: методическая аномалия (числовой сбой).

---

## 🟡 MEDIUM-3: Bootstrap CI — **i.i.d.-based, не doc-level; «CI [−0.2243, −0.2231]» — иллюзорная точность**

```python
dc_ci = list(bootstrap_ci(dc[~np.isnan(dc)], B=B, seed=SEED))
```
`dc` — плоский массив `N·S = 4096` (doc × position). Позиции внутри документа **коррелированы** (E-31: `C_F(1) = 0.30–0.996`). I.i.d.-bootstrap по 4096 значениям даёт CI в ~√(4096/32) ≈ 11× уже истинного.

Тот же баг, что в E-38b (я его отмечал в предыдущем аудите). ТЗ E-38f процедура: «Bootstrap CI Δcos по docs» — код делает по (doc, pos), не по docs.

**Fix:**
```python
per_doc_dcos = np.array([
    float(np.nanmedian(dc[i * Ss:(i + 1) * Ss])) for i in range(N)
])
dc_ci = list(bootstrap_ci(per_doc_dcos[~np.isnan(per_doc_dcos)], B=B, seed=SEED))
dc_med = float(np.nanmedian(per_doc_dcos))
```

При N = 32 и коррелированных позициях честный CI будет в разы шире; вердикт может стать `intermediate` (не однозначно `lemma_broken`).

---

## 🟡 MEDIUM-4: `np.median` в `profile_dc_per_docblock` (не `nanmedian`)

```python
profile.append(float(np.median(row_dc)))
```
`row_dc` — список из `Ss` значений, может содержать NaN (при `vp = None` для вырожденных J^T u). `np.median` с NaN → **NaN**. Профиль молча получает NaN-провалы, которые трудно отличить от «медиана = NaN по существу».

**Fix:** `float(np.nanmedian(row_dc))`.

---

## 🟡 MEDIUM-5: `has_data` допускает 1 non-NaN, но `bootstrap_ci` требует ≥ 2

```python
has_data = nan_count < dc.size           # True при 1 non-NaN
dc_ci = list(bootstrap_ci(dc[~np.isnan(dc)], B=B, seed=SEED)) if has_data else [nan, nan]
```

Если `nan_frac = 1 − 1/4096` (единственный non-NaN), `bootstrap_ci` бросает `ValueError: need >= 2 values`. Не обёрнуто в try — **крах всего эксперимента** на краевом случае.

**Fix:**
```python
valid_dc = dc[~np.isnan(dc)]
has_data = valid_dc.size >= 2
```

---

## 🟡 MEDIUM-6: Нет assert на форму `Jt`

```python
Jt = J[i, t]
U, _, Vt = torch.linalg.svd(Jt, full_matrices=False)
...
c = (U @ one).abs()   # one длины K = 128; если Jt.shape[0] != 128 → RuntimeError
```

`one = torch.ones(K)` при `K = 128` (константа файла). Если в кэше `K ≠ 128` (например, другой прогон), крах с неинформативным сообщением. Плюс `_e38_cache.K = 128`, `exp38e` asserts `len(token_set) == 128` — но `exp38f` — нет.

**Fix:** `assert Jt.shape[0] == Kk, f"Jt rows {Jt.shape[0]} != K {Kk}"`.

---

## 🟡 MEDIUM-7: `exp38e` gate читается из файла, но `E-38e` сам баговый

```python
gate_ok = not g["outcomes"].get("gate_failed", False)
```

`E-38e` reports `mixed` (`|α|/|β| = 0.195 < 10`), gate проходит. ✓ Но `E-38e` содержит независимый баг нормализации (`σ_g = σ_1(M_K)` вместо `σ_g(J)`), который я отметил в предыдущем аудите. После фикса E-38e gate не изменится (он на `|α|/|β|`, не на consistency), так что этот пункт — только про прозрачность: если `E-38e` пере-запускают, `exp38f` продолжит корректно.

**Не критично.**

---

## 🟢 LOW-8: `torch.sqrt(torch.tensor(float(Kk)))` — тянет torch в мусор

```python
one = torch.ones(Kk) / torch.sqrt(torch.tensor(float(Kk)))
```
`math.sqrt(Kk)` даёт Python float, `torch.ones(Kk) / math.sqrt(Kk)` создаёт один tensor. Стиль.

---

## 🟢 LOW-9: `profile` — только 4 doc-блока (i=0,8,16,24)

Не баг, но `np.median` по 4 документам не отражает 32; в JSON поле `profile_dc_per_docblock` может быть интерпретировано неверно. Плюс уже отмеченная проблема с NaN. Рекомендация: явно `meta["profile_docs"] = [0, 8, 16, 24]` или расширить до всех 32.

---

## 🟢 LOW-10: Naming `cos_ug_uniform` в metrics — после фикса нормально, до фикса — вводит в заблуждение

Поле `cos_ug_uniform` в JSON сейчас = row-метрика, не `|u_g · 1/√K|`. После фикса станет правильной. Никаких JSON-изменений не нужно (значение просто станет корректным), но в `paper_old.md` любая ссылка на «E-38f cos_ug = 0.23» должна быть отозвана.

---

## Сводная таблица

| Категория | Проблема |
|---|---|
| **HIGH** | `U @ one` вместо `U.T @ one` — производит ложный `lemma_broken`, вероятно **развернёт главный вывод** после фикса |
| **HIGH** | Pre-registered `Δcos > +0.1 = anomaly` противоречит [L-gauge-J] (лемма допускает `Δcos ∈ [0, 1−c_g]`) |
| MEDIUM | Bootstrap CI — i.i.d. по (doc, pos), а не doc-level; «CI [−0.2243, −0.2231]» — иллюзорная точность |
| MEDIUM | `np.median(row_dc)` в профиле (NaN) — должно быть `np.nanmedian` |
| MEDIUM | `has_data` допускает 1 non-NaN → `bootstrap_ci` крах |
| MEDIUM | Нет assert на `Jt.shape[0] == K` |
| MEDIUM | `exp38e` gate читается, но E-38e сам баговый (не блокирует, но для трейсинга) |
| LOW | `torch.sqrt(torch.tensor(float(Kk)))` → `math.sqrt` |
| LOW | `profile` — только 4 doc |

---

## Кросс-влияние на выводы статьи

| Эксперимент | Заявленный вывод | Реальный статус |
|---|---|---|
| **E-38e** | `consistency_redflag` — нарушение M-1 | **Скорее ложно-положительный** (неверная нормализация σ_g, я отметил в прошлом аудите) |
| **E-38f** | `lemma_broken: Δcos = −0.224` — разрыв M-1 | **Ложно из-за `U @ one`**; после фикса может дать `lemma_confirmed` |
| **`paper_old.md §21`** | «Унификация M-1 не подтверждается» | Опирается на E-38e + E-38f, **оба с багами** |
| **`E38.md` Summary п.1** | «Унификация M-1 не подтверждается» | То же |
| **`paper_old.md Прил. B.2 H1–H8`** | (не помечено) | При пере-записи `H1 (Jacobian meaningful)` и `H3 (E-projection)` — не задеты; но `[L-gauge-J]` (лемма M-1) — **под вопросом** |

**P0 перед любой публикацией:** пере-запустить E-38e (с исправлением `σ_g(J)`) и E-38f (с `U.T @ one` + doc-level CI). Только после этого формулировать вывод о статусе M-1. Текущее «не подтверждается» — ненадёжно.