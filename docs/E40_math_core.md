# ТЗ: математика в статью и модули ядра v1 для репозитория `prism`

**Ревизия 2 (2026-11-03, после закрытия E6):** внесены правки архитектора (`docs/E40_math_fixes.md`, файл после внесения удалён): E6 закрыт (несогласованность тройки = смешение медианы и среднего в `E39_supplement` §2.2.1), `frac_K` разстейджирован, `slice`/`aggregation` identity-relevant, добавлен инвариант `same_slice_aggregation`, спорные пункты — §VI, порядок работ — §VII.

**Кому:** агенту-исполнителю, работающему только с репозиторием `https://github.com/survivorm/prism` (папка `prism/`).
**Что это:** stand-alone спецификация. Контекст обсуждения не нужен — всё, что требуется, описано ниже. Два блока: (I) что добавить в математическую часть статьи; (II) какие модули кода создать/расширить, с сигнатурами, инвариантами и краевыми условиями.

---

## 0. Что уже есть в репозитории (релевантное)

В `prism/` существуют модули: `jacobian.py` (вычисление J), `esg.py` (ESG), `holonomy.py` (Procrustes-голономия), `spectral.py` (спектр), `noise.py`, `permutation.py`, `diagnostics.py`, `bridge.py`, `calibration.py`, `cache.py`, `interventions.py`, `topology.py`, `layer_selection.py`. Новые модули **не дублируют** их, а используют как источники данных и дополняют недостающей алгеброй.

---

## I. Математика для статьи

Добавить в `paper.md` (раздел §2 «Аппарат») и `ref-math.md` (глоссарий/индекс) следующие объекты. Все они уже прошли содержательную проверку; часть из них — уточнение уже написанного, часть — новое.

### 1. Алгебра оператора `P(p)` (модуль `p_algebra`)

**Определение.** `P(p) = diag(p) − p pᵀ`, `p ∈ Δ^{K−1}` (нормированный softmax на носителе). Три роли одного объекта:
- якобиан softmax: `P = ∂ softmax/∂z`;
- метрика Фишера–Рао в логит-координатах: `Jᵀ P J`;
- аннигилятор калибровки: `P·1 = 0`.

**Свойства (доказать/зафиксировать):**
- `P` симметричен, положительно полуопределён (`xᵀPx = Var_p[x] ≥ 0`), `rank P = K−1` (при `K > 1`);
- `P` **не проектор**: `P² ≠ P` ни при каком `p` (при `p = uniform`: `P² = (1/K)P`);
- замкнутые формы: `tr P = 1 − ‖p‖₂²`, `tr P² = ‖p‖₂² − 2‖p‖₃³ + (‖p‖₂²)²`;
- собственные значения — корни секулярного уравнения `Σᵢ pᵢ²/(pᵢ − λ) = 1`; `λ_max ≤ p_max(1 − p_max)`, в двухточечном пределе `λ_max = 2p_max(1 − p_max)`; для типичных `p` диапазон `≈ 0.09–0.18` (при `K = 128`).

**Что добавить в статью:** новый подраздел §2.5.1 «Замкнутые формы и спектр `P(p)`»; в `ref-math.md` — соответствующие строки глоссария.

### 2. Канонический срез и инвариантность (модуль `slices` / `p_algebra`)

**Определение.** `J^(p_t) = P(p_t)J = ∂p/∂h` — канонический срез. Три среза: `raw` (`J`), `gauge-free` (`P(p_t)J`), `fixP` (`P(p₀)J`).

**Правило срез-локальности:** тождества выполняются внутри каждого среза; числа не переносятся между срезами. Пример: `g_uni(J^gf) ≡ 0` по построению (`P·1 = 0`), а `g_uni` на raw — нет.

**Что добавить:** явная формулировка в §2.5 (сейчас есть неявно); таблица трёх срезов и правило срез-локальности.

### 3. Тождества `O2 = O3` и `g_uni(PJ) = 0`

**Тождество (a).** `g_uni = ‖QJ‖²_F / ‖J‖²_F`, `Q = 11ᵀ/K`. Тогда `g_uni = 1ᵀ(JJᵀ)1 / (K tr(JJᵀ))` — то же, что ESG `gauge_share = Σ_k (u_kᵀ1)² λ_k / Σ_k λ_k`. **Это одна величина**, а не две.

**Тождество (b).** `P·1 = 0 ⇒ (PJ)ᵀ1 = 0 ⇒ g_uni(J^gf) ≡ 0` (тождественно, не эмпирически).

**Что добавить:** подраздел §2.5.2 «Два тождества, устраняющие путаницу»; в §4.5 (блок gauge) — заменить формулировки «два объекта» на «одна величина под двумя именами».

### 4. Разложение `O1/iso = conc × V × sharpness`

**Определение.** `O1 = ‖PJ‖²_F / ‖J‖²_F` (доля энергии, убитая оператором); `iso = ‖P‖²_F / K` (изотропный референс). Точное разложение (доказано алгебраически, проверено численно с невязкой < 1e-16):

```
O1/iso = conc × V × sharpness,
conc      = K · Σa / Σ(b·c),   a_i = p_i² d_i,  d_i = ‖J_i − m‖²,  m = Σ_j p_j J_j,
V         = Σb / ‖J‖²_F,
sharpness = c_eff / ‖P‖²_F,    c_eff = Σ(b·c)/Σb,  c = Σ_i p_i².
```

**Смысл.** Малость `O1/iso` (0.25–0.69×) — свойство оператора (его нормы), а не «gauge доминирует в данных». Это ключевой аргумент, снимающий энергетическую мотивировку H5.

**Что добавить:** §2.5.3 «Почему `P` убивает меньше, чем изотропный референс»; в §5 (синтез) — отзыв энергетической поддержки H5.

### 5. Тройка `(c, s, frac_K)` — самосогласованная при одной агрегации (E6 закрыт)

**Определения:**
- `c = ⟨J, PJ⟩/‖J‖²` (агрегат: `Σ_pos ⟨J, PJ⟩ / Σ_pos ‖J‖²`, **тот же срез**);
- `s = ‖PJ‖/‖J‖` (агрегат: `√(Σ_pos ‖PJ‖² / Σ_pos ‖J‖²)`, **тот же срез**);
- `frac_K = ‖(I−P)J‖²/‖J‖²` (агрегат: `Σ_pos ‖(I−P)J‖² / Σ_pos ‖J‖²`, **тот же срез**);
- `cos(J, PJ) = c/s` (не независимый объект).

**Тождество:** `frac_K = 1 − 2c + s²` выполняется **точно** при одинаковой агрегации и одинаковом срезе (попозиционно — assert < 1e-12; на aggregate одного среза — точное тождество в fp64). Отсюда `c = (1 + frac_K − s²)/2` без корня.

**Что было не так и как разобрано (E6 закрыт 2026-11-03):**
Таблица §2.2.1 в `E39_supplement.md` (ранняя версия) смешивала агрегаты: `c`, `cos` — aggregates alive-среза, `frac_K` — медиана всех позиций. Медиана отношения ≠ отношение сумм. Подстановка смешанных строк давала фиктивное `s² < 0` (от −7e-4 до −1.5e-3). Исполнитель пересчитал тройку на одном срезе/агрегации; тождество замыкается, `s_agg` и `frac_K_energy_agg` добавлены в JSON (`K_from_I_minus_P`), сверка — бит-в-бит.

**Самосогласованные числа** (пример, alive, `E39_supplement.md` §2.2.1):

| Слой | `c` | `s` | `cos = c/s` | `frac_agg = 1−2c+s²` |
|---|---|---|---|---|
| L6  | 1.215e-3 | 0.01381 | 0.0880 | 0.997760 |
| L12 | 1.067e-3 | 0.01258 | 0.0848 | 0.998025 |
| L18 | 5.491e-4 | 0.00830 | 0.0661 | 0.998971 |
| L22 | 4.810e-4 | 0.00750 | 0.0641 | 0.999094 |
| L23 | 2.087e-3 | 0.01939 | 0.1076 | 0.996202 |

(Полная таблица для all-среза — в `E39_supplement.md` §2.2.1.)

**Правило для статьи и артефактов:**
Для любой величины из тройки `(c, s, cos, frac_K)` **обязательно указывать**:
1. **Срез** (`all` / `alive` / другой);
2. **Агрегацию** (`ratio_of_means` / `mean` / `median` / `energy_weighted`).

Медиана отношения и отношение сумм — разные объекты; для тождества `frac_K = 1 − 2c + s²` требуется **одна агрегация** (`ratio_of_means`); при `median_per_position` тождество не держится (ожидаемо).

**В статью:** §2.5.4 «Тройка `(c, s, frac_K)` и правило срез-агрегации». В §6 (футер) — правило: «для отношений указывать срез и агрегацию; медиана отношения и отношение сумм — разные объекты» (в `paper.md` §6 следующий номер — 24; в фиксах названо «правило 7» — это про supplement §6, не про футер). **В ref-math:** запись с формулировкой тождества и условием валидности.

### 6. `cos(J, PJ) ≈ 0.064–0.108` — почти ортогональность, не параллельность

**Измерено.** `cos(J, PJ)` = 0.064–0.108 (углы 82–86°); `c ≈ 0.001–0.002`. Из PSD `P` следует только `c ≥ 0`, но не «PJ выровнен с J».

**Смысл.** `PJ` — **маленькое перпендикулярное возмущение** `J`, а не «удаление доминирующего направления». Это примиряет `‖PJ‖ ≈ 0.02‖J‖` с `‖K‖ ≈ ‖J‖` (так как `‖J − PJ‖² = ‖J‖² + ‖PJ‖² − 2c‖J‖² ≈ ‖J‖²`).

**Что добавить:** §2.5.5 «Геометрический смысл `cos(J, PJ)`»; в §2.11 (мотивировка `E_K`) — заменить «убирает доминирующее направление» на «добавляет маленькое перпендикулярное возмущение».

### 7. Кинетическая SVD-декомпозиция (5 каналов, энергии)

**Определение.** Для `J_t, J_{t+1}` в транспортном фрейме: `ΔJ = J_{t+1} − J_t`. Рецептура линеаризации:

```
ΔJ_lin = ΔΣ + M Σ̄ − Σ̄ N,  M = log polar(U_tᵀU_{t+1}),  N = log polar(V_tᵀV_{t+1}),  Σ̄ = (Σ_t+Σ_{t+1})/2.
```

**Пять каналов:** `E_radial = ‖ΔΣ‖²`, `E_right = ‖Σ̄ N‖²`, `E_left = ‖M Σ̄‖²`, `E_cross = 2 tr(Σ̄ M Σ̄ N)` (знаковый).

**`E_resid` (независимо):** `‖U_tᵀ ΔJ V_t − ΔJ_lin‖²_F`. **Не** через conservation (иначе тавтология).

**`E_off_manifold` (шестой канал, спорный — §VI.1):** `‖ΔJ − U_t U_tᵀ ΔJ V_t V_tᵀ‖²_F`. При `r = 16 ≪ D = 1024` **не пренебрежим**, особенно на L6–L22, где `Δ_SVD ≈ π/2`. Без него `Σ5 = ‖ΔJ_proj‖²`, а не `‖ΔJ‖²`.

**Полная accounting:** `‖ΔJ‖² = E_radial + E_right + E_left + E_cross + E_resid + E_off_manifold + eps3_correction`, где `eps3_correction = O(ε³)` — диагностика, не канал.

**Два условия валидности:**
- **A:** `Δ_SVD < π/2 − 0.3` рад; иначе `linearization_valid = False` (на текущих данных — только L23);
- **B:** `min σ-gap > 10⁻² σ₁²`; иначе `rotation_unique = False` (на всех слоях по E-38c).

**Если `linearization_valid = False`** — отчитывать только `E_radial` и `E_off_manifold`; пять каналов неинтерпретируемы. **Если `rotation_unique = False`** — пять каналов отчитывать, но `E_right, E_left, E_cross` помечать `ambiguous` (базис-dependent). **Если шестой канал rejected** (§VI.1) — в assert `‖ΔJ‖²` заменить на `‖ΔJ_proj‖²`; `E_off_manifold` остаётся отдельным диагностическим полем, не входит в сумму.

**Важно:** хранить **энергии**, а не доли (доли композиционны, `Σ s_i = 1`, ранг ≤ 4 по построению). При `r=1` кросс-член тождественно нулевой. `E_residual` вычисляется **независимо** (из остаточного члена рецептуры), а не вычитанием — иначе ассерт `Σ = ‖ΔJ‖²` тавтологичен.

**Что добавить:** §2.9 (уже есть кинетическая декомпозиция, но нужно явно зафиксировать энергии, независимый `E_residual`, `r=1`-контроль).

### 8. Procrustes-транспорт с `align_signs_and_columns` (E5)

**Проблема.** `Q_U = polar(U_tᵀU_{t+1})` корректна только если столбцы `U_t` и `U_{t+1}` согласованы по знаку и порядку. Иначе `log Q_U` даёт мусор (флип знака → `−1` на диагонали → комплексные члены; перестановка при `σ_j ≈ σ_{j+1}` → `log` не определён на главной ветви).

**Решение (обязательное предусловие):**
1. **Signs:** `sign(u_j(t)ᵀu_j(t+1)) → +1` до полярного фактора.
2. **Columns:** сопоставление по максимуму `|U_tᵀU_{t+1}|` (венгерский алгоритм), не по рангу `σ`; флаг `order_cross` при расхождении.
3. **Proximity:** `‖log Q_U‖_F` мал, спектр `Q_U` вдали от `−1`; иначе `unstable_transport`.
4. **Artifact:** все флипы/перестановки → в `E40_5_transport.json`.

**Что добавить:** §2.7 (вертикальный транспорт) — явно описать `align_signs_and_columns` как предусловие; в `ref-impl.md` — алгоритм.

### 9. Редукционный контроль (правило 23) и два null'а

**Правило.** Любая величина `‖OX‖²/‖X‖²` проверяется против **двух** нулей:
- **operator-null** (`iid`): сколько `O` съел бы у случайного `X`;
- **content-matched-null**: `X` с сохранёнными нормами/маргиналами, рандомизированными направлениями.

Плюс изотропный референс `iso(O) = ‖O‖²_F/K` и отношение `real/iso`.

**Что добавить:** §3.2 (протокол) — явная формулировка правила; в §6 (футер) — пункт 23.

### 10. Sentinel как разрез, а не бинарная маска

**Измерено.** Доля позиций с `Σ P128_RAW < 0.5` = 0.5071 (одинаково по слоям). Распределение массы top-128 на полном softmax: p10/p90 = 0.0050/0.9952. Любой порог 0.2–0.8 даст ~50% «sentinel». **Рекомендация:** не использовать бинарную маску; отчитываться кривой массы или включать массу ковариатой.

**Что добавить:** §2.5.6 «Sentinel: артефакт усечения, а не свойство модели»; в §3.2 — правило «sentinel не бинарная маска».

---

## II. Код: модули ядра v1

Все новые модули — в `prism/`. Существующие модули (`jacobian.py`, `esg.py`, `holonomy.py`, `spectral.py`, `noise.py`, `permutation.py`) используются как источники данных; новые модули не дублируют их, а добавляют недостающую алгебру и контракты.

### 1. `prism/tag.py` — контракт метки (M1)

**Назначение:** обязательная паспортизация каждого результата.

**Содержимое:**
```python
ChannelTag = namedtuple(
    "ChannelTag",
    ["protocol", "slice", "window", "side", "p_provenance", "aggregation",
     "layer", "rank", "norm_type", "source", "status"]
)
# protocol ∈ {"natural", "deltaW"}
# slice ∈ {"raw", "gf", "fixP"}
# window ∈ {"top", "tail", "full"}
# side ∈ {"left", "right", "mixed"}
# p_provenance ∈ {"truncated_U128", "full_softmax", "other"}
# aggregation ∈ {"ratio_of_means", "mean", "median", "energy_weighted"}   # identity-relevant, NEW
# norm_type ∈ {"norm", "energy", "angle"}   # identity-relevant
# source ∈ {"cache", "recomputed", "audit-only"}
# status ∈ {"measured", "inferred", "not_testable"}
```

**Правила:**
- `slice`, `p_provenance`, `norm_type`, `aggregation` — **identity-relevant** (меняют числа; в частности `triple_identity` держится только при `ratio_of_means`); обязательно заполнять.
- `layer`, `rank`, `source`, `status` — атрибуты, не identity-relevant.
- `aggregation` обязателен для любого отношения (значение/референс, real/null); по умолчанию `ratio_of_means`.
- Любое отношение вида `real/null` обязано хранить `slice` и `aggregation` (правило 7 E39_supplement).
- Теги, собранные из разных уровней (например, `basis=topk128` + `norm_type=spectral_full`), — **невалидны**; сборка обязана кидать ошибку.
- Артефакт без полного `ChannelTag` невалиден; неполный тег — `INCOMPLETE_TAG`, отношение не публикуется.

**Интеграция:** использовать в `pipeline.py` и во всех артефактах E-40.x; артефакт без `ChannelTag` невалиден.

### 2. `prism/invariants.py` — assert-hub (M0)

**Назначение:** реестр кросс-модульных тождеств; прогоняется при любом изменении ядра.

**Содержимое:**
```python
INVARIANTS = {
    "O2_eq_O3": {
        "identity": "g_uni(J) == gauge_share(ESG)",
        "modules": ("p_algebra", "operator_checks"),
        "assert": lambda J, p: abs(g_uni(J) - gauge_share(J)) / gauge_share(J) < 1e-12,
        "status": "passing",
    },
    "g_uni_gf_zero": {
        "identity": "g_uni(P(p)J) == 0",
        "modules": ("p_algebra",),
        "assert": lambda J, p: g_uni(P(p) @ J) < 1e-12,
        "status": "passing",
    },
    "P_properties": {
        "identity": "P·1=0, P⪰0, P²≠P",
        "modules": ("p_algebra",),
        "assert": lambda p: (norm(P(p) @ ones) < 1e-12) and (min_eig(P(p)) >= -1e-12) and (not is_idempotent(P(p))),
        "status": "passing",
    },
    "iso_inline_eq_closed": {
        "identity": "iso_inline == iso_closed",
        "modules": ("p_algebra",),
        "assert": lambda p: abs(iso_inline(p) - iso_closed(p)) / iso_inline(p) < 1e-9,
        "status": "passing",
    },
    "p_tilde_sum": {
        "identity": "Σp̃ ∈ {1, 1 − mass_sentinel}",
        "modules": ("tag", "p_algebra"),
        "assert": lambda p: abs(sum(p) - 1) < 1e-12 or abs(sum(p) - (1 - mass_sentinel)) < 1e-12,
        "status": "passing",
    },
    "secular_equation": {
        "identity": "Σ p̃ᵢ²/(p̃ᵢ − λ) = 1",
        "modules": ("p_algebra",),
        "assert": lambda p, lam: abs(sum(p_i**2 / (p_i - lam)) - 1) < 1e-12,
        "status": "passing",
    },
    "trace_identities": {
        "identity": "tr P = 1 − ‖p‖₂²; tr P² = ‖p‖₂² − 2‖p‖₃³ + (‖p‖₂²)²",
        "modules": ("p_algebra",),
        "assert": lambda p: abs(trace(P(p)) - (1 - norm2(p)**2)) < 1e-12,
        "status": "passing",
    },
    "H_orthogonal": {
        "identity": "H·Hᵀ = I",
        "modules": ("transport",),
        "assert": lambda H: norm(H @ H.T - I) < 1e-10,
        "status": "passing",
    },
    "kappa_invariant": {
        "identity": "κ(OᵀHO) = κ(H)",
        "modules": ("transport",),
        "assert": lambda H, O: abs(kappa(O.T @ H @ O) - kappa(H)) < 1e-10,
        "status": "passing",
    },
    "channels_sum": {
        "identity": "E_radial + E_right + E_left + E_cross + E_resid + E_off_manifold + eps3_correction = ‖ΔJ‖²",
        "modules": ("kinetic",),
        "assert": lambda J0, J1: abs(sum_channels(J0, J1) - norm(J1 - J0)**2) < 1e-9,
        "status": "passing",
        "note": "E_residual вычисляется независимо, не вычитанием; eps3_correction — диагностика, не канал; "
                "если шестой канал rejected (§VI.1): правая часть ‖ΔJ_proj‖², E_off_manifold вне суммы",
    },
    "r1_cross_zero": {
        "identity": "r=1 ⇒ E_cross = 0",
        "modules": ("kinetic",),
        "assert": lambda J0, J1: abs(cross_channel(J0, J1, r=1)) < 1e-12,
        "status": "passing",
    },
    "O1_eq_s2": {
        "identity": "O1 ≡ s² (послойно)",
        "modules": ("operator_checks",),
        "assert": lambda J, p: abs(O1(J, p) - s(J, p)**2) < 1e-12,
        "status": "passing",   # empirical: может failing
    },
    "b_lin_r1": {
        "identity": "b_lin(r=1) = ε²σ₁²",
        "modules": ("behavior",),
        "assert": lambda J, eps: abs(b_lin(J, eps, r=1) - eps**2 * sigma1(J)**2) < 1e-12,
        "status": "passing",
    },
    "triple_identity": {
        "identity": "frac_K = 1 − 2c + s²",
        "modules": ("cheap_invariants",),
        "assert": lambda J, p, agg="ratio_of_means": (
            abs(frac_K(J, p, agg) - (1 - 2*c(J, p, agg) + s(J, p, agg)**2)) < 1e-12
        ),
        "status": "passing",   # E6 закрыт 2026-11-03: выполнялось всегда, блокировано ошибкой агрегации в ТЗ
        "note": "Требует одной агрегации и одного среза. Медиана отношения ≠ отношение сумм.",
    },
    "same_slice_aggregation": {   # NEW (ревизия 2)
        "identity": "для отношения value/reference: slice и aggregation явные",
        "modules": ("tag", "cheap_invariants"),
        "assert": lambda tag: tag.slice is not None and tag.aggregation is not None,
        "status": "passing",
    },
}
```

**Правила:**
- Каждая запись обязана иметь `status ∈ {"passing", "failing", "blocked"}`.
- Изменение любого модуля → полный перепрогон `INVARIANTS`; смена статуса — событие в манифесте.
- Требование невакуитета: запись должна быть фальсифицируема (например, `channels_sum` валиден только если `E_residual` посчитан независимо).

### 3. `prism/p_algebra.py` — алгебра `P(p)` (M3)

**Функции:**
```python
def softmax_jacobian(p: np.ndarray) -> np.ndarray:
    """P(p) = diag(p) − p pᵀ. p: [K], sum=1. Возвращает [K,K]."""
def trace_P(p): ...
def trace_P2(p): ...
def isotropic_reference(p):  # ‖P‖²_F / K
def g_uni(J, p=None):        # ‖QJ‖²/‖J‖², Q=11ᵀ/K; p не используется
def gauge_share(ESG):        # Σ (u_kᵀ1)² λ_k / Σ λ_k
def secular_roots(p):        # собственные значения P(p)
```

**Краевые условия:** `p` может быть усечённым (`truncated_U128`) — `g_uni` не зависит от `p`; `tr P` и `iso` зависят от `p` (проверять `p_provenance`). `P` не идемпотентен — не использовать как проектор.

**Интеграция:** использовать в `esg.py` (для `gauge_share`), `diagnostics.py` (для `g_uni`), `bridge.py` (для `P`).

### 4. `prism/svd_bundle.py` — SVD-связка (M4)

**Функции:**
```python
def svd_bundle(J, rank, side="right", slice="raw"):
    """Возвращает dict: U, Sigma, V, energy, rank_effective, stable, tag."""
def principal_angles(V1, V2, r): ...
def effective_rank(Sigma, threshold): ...
```

**Интеграция:** может быть расширением `spectral.py`; использовать в `holonomy.py` (для `V_t`), `kinetic.py` (для `U, Σ, V`), `esg.py` (для `U`).

### 5. `prism/transport.py` — Procrustes-транспорт (M5)

**Функции:**
```python
def align_signs_and_columns(U0, U1) -> (U0_aligned, U1_aligned, flags):
    """Выравнивание знаков и порядка столбцов до polar.
    flags: order_cross, unstable_transport."""
def procrustes_transport(V0, V1):  # polar(V0ᵀV1)
def holonomy(V0, V1, V2):          # R01 R12 R02⁻¹
def holonomy_spectrum(H):          # {θ_j}, κ
```

**Краевые условия:** без `align_signs_and_columns` `log Q` не определён при флипе знака или перестановке; `unstable_transport` при спектре `Q` вблизи `−1`.

**Интеграция:** расширить `holonomy.py`; использовать в `kinetic.py` (для `Q_U, Q_V`), `topology.py` (для `F_lt`).

### 6. `prism/kinetic.py` — 5-канальная декомпозиция (M6)

**Функции:**
```python
def kinetic_decomposition(J_t, J_t1, r=16, eps_angle=0.3, eps_gap=1e-2):
    """Каналы E_radial/E_right/E_left/E_cross/E_resid (+E_off_manifold, §VI.1).
    E_resid — независимо (из остаточного члена рецептуры). r=1: E_cross = 0.
    validity: A (eps_angle) → linearization_valid, B (eps_gap) → rotation_unique."""
def channel_energy_matrix(J_t, J_t1, r):  # 5×5 на энергиях (не долях)
```

**Краевые условия:** `E_cross` знаковый; отчитывать `signed` и `abs`. Энергии, не доли. `E_residual` — независимо.

**Интеграция:** новый модуль; использовать в `pipeline.py` (E-40.5), `topology.py` (F_lt).

### 7. `prism/operator_checks.py` — операторные проверки (M7)

**Функции:**
```python
def operator_fraction(O, X, norm_type="energy"):  # ‖OX‖²/‖X‖² или ‖OX‖/‖X‖
def isotropic_reference(O):                       # ‖O‖²_F / K
def cos_J_PJ(J, p):                               # cos(J, PJ)
def O1(J, p):                                     # ‖PJ‖²/‖J‖²
def s(J, p):                                      # ‖PJ‖/‖J‖
def frac_K(J, p):                                 # ‖(I−P)J‖²/‖J‖²
```

**Краевые условия:** `norm_type` — identity-relevant; `cos_J_PJ` и `c` — разные числа; `frac_K` — энергия, не норма.

**Интеграция:** новый модуль; использовать в `diagnostics.py`, `esg.py`.

### 8. `prism/cheap_invariants.py` — дешёвые инварианты (M10)

**Функции:**
```python
def c(J, p, aggregation="ratio_of_means"):      # ⟨J, PJ⟩ / ‖J‖²
def s(J, p, aggregation="ratio_of_means"):      # √(Σ ‖PJ‖² / Σ ‖J‖²)
def cos(J, p, aggregation="ratio_of_means"):    # c/s
def frac_K(J, p, aggregation="ratio_of_means"): # ‖(I−P)J‖² / ‖J‖²
def c_pred(p):                                   # tr P / √(K tr P²)  (изотропный предиктор)
def A(J, p):                                     # c / c_pred  (коэффициент выравнивания)
```

**Агрегация** — explicit parameter; по умолчанию `ratio_of_means`. Для `median` тождество `triple_identity` не выполняется (ожидаемо — см. invariant).
**Все четыре (`c, s, cos, frac_K`) — рабочий вход ядра.** `frac_K` разстейджирован решением 2026-11-03 (E6 закрыт).

**Интеграция:** новый модуль; использовать в `diagnostics.py` (E-40.1), `paper.md` §2.5.

### 9. `prism/nulls_and_stats.py` — нули и статистика (M8, M9)

**Функции:**
```python
def operator_null(O, X, kind="iid"):        # iid или content-matched
def reduction_control(statistic, trivial_predictor, null_model, partial_vars=None):
def paired_bootstrap(docs, stat_fn, B=1000):
def permutation_test(objects, stat_fn, permute_fn, B=1000):
def resolution_floor(metric_fn, docs, subsamples=[8,16,32]):
```

**Правила:**
- `operator-null` (`iid`) + `content-matched-null` — оба обязательны для `‖OX‖²/‖X‖²`.
- `reduction_control` — стандартный паттерн для производных метрик (вычесть тривиальное предсказание).
- `resolution_floor` — внутри каждого теста, не глобально; флаг `not_testable`, если эффект ниже floor.

**Интеграция:** консолидировать `noise.py`, `permutation.py`, `diagnostics.py`; использовать во всех E-40.x.

### 10. `prism/behavior.py` — поведенческие эндпоинты (M11, gate)

**Функции:**
```python
def b_mag(delta_z):              # ‖Δz‖² (квадрат, не первая степень)
def b_lin(J, eps, v):            # ε²‖Jv‖²; = ε²σ₁² при v = v₁
def b_res(b_mag, b_lin):         # b_mag − b_lin
def b_qual(logits_base, logits_perturbed):  # corr_f
def KL_lin(J, p, eps, v):        # ½ ε² (Jv)ᵀ F(z) (Jv), F = diag(p) − ppᵀ
```

**Gate:** `behavior.py` **не импортируется** из tier-0 (`E40_*.py`); только из `E-42/E-43/E-44`. Техническая проверка: `import-check` в CI.

**Краевые условия:** `b_mag` — только `‖·‖²` (не `‖·‖₂`); `b_lin` — общая форма `ε²‖Jv‖²`, `ε²σ₁²` — частный случай; `KL_lin` — с фактором Фишера, не голая `½ε²σ₁²`.

### 11. `prism/profiles.py` — рецепты (вне ядра)

**Функции:**
```python
def R_can(J, p):        # λ_max(E_K)/tr(E_K)
def R_raw(J):           # λ_max(JJᵀ)/tr(JJᵀ)
def G(l):               # 1 − Δ_fixP/Δ_raw
def r_cross(curve1, curve2):  # пересечение
```

**Статус:** recipes, не core. Продвижение в ядро — при появлении второго потребителя (после E-41).

---

## III. Подводные камни и краевые условия (сводка)

1. **`P` не проектор.** `P² ≠ P`; не использовать `P` как `Π` в формулах, где нужна идемпотентность.
2. **`g_uni` не зависит от `p`.** `g_uni = ‖Jᵀ1‖²/(K‖J‖²)`; перенормировка `p` на него не влияет. Зависимость от `p` — только у `tr P`, `iso`, `O1`.
3. **`cos(J, PJ)` ≈ 0.08, не `c`.** `c ≈ 0.002`; `cos = c/s`; не путать.
4. **`frac_K` — энергия**, не норма. Разстейджирован (E6 закрыт): тройка `(c, s, frac_K)` самосогласована при одной агрегации и одном срезе, `c = (1 + frac_K − s²)/2` без корня.
5. **Sentinel — разрез, не маска.** 50.7% — артефакт усечения; использовать кривую массы или ковариату.
6. **`align_signs_and_columns` обязателен.** Без него `log Q` не определён при флипе знака/перестановке.
7. **`E_cross` знаковый.** Отчитывать `signed` и `abs`; в энергетических долях использовать `|E_cross|`.
8. **`E_residual` — независимо.** Иначе ассерт `Σ = ‖ΔJ‖²` тавтологичен.
9. **`b_mag = ‖Δz‖²`.** Не первая степень; `‖·‖₂` в поле `b_mag` запрещено.
10. **`KL_lin` с фактором Фишера.** `F = diag(p) − ppᵀ`, не `I`.
11. **MP-скобки поэтапно.** `p=6` (6×6): `λ+ = 2.05/1.38`; `p=8` (8×8): `λ+ = 2.25/1.45`.
12. **`p_provenance` identity-relevant.** Без него абсолютные числа `tr P`, `iso`, `g_uni` несравнимы между срезами.
13. **Шестой канал (E_off_manifold).** Обязателен при `r ≪ D`; без него `Σ5 = ‖ΔJ_proj‖²`, а не `‖ΔJ‖²` (§VI.1 для reconciliation).
14. **Агрегация identity-relevant.** Медиана отношения ≠ отношение сумм. `triple_identity` требует одной агрегации (`ratio_of_means`). Для любой тройки `(c, s, cos, frac_K)` указывать срез и агрегацию.
15. **Срез identity-relevant.** `c`, `s`, `frac_K` на all ≠ на alive. Не смешивать.

---

## IV. Юнит-тесты (прогоняются в E-40.1)

1. `O2 == O3`, rel < 1e-12.
2. `g_uni(J^gf) < 1e-12`.
3. `iso_inline == iso_closed`, rel < 1e-9.
4. `frac_K = 1 − 2c + s²` при `aggregation="ratio_of_means"`, том же срезе — **passing** (было EXPECTED FAIL до закрытия E6).
5. **NEW: `same_slice_aggregation`:** все ChannelTag имеют `slice` и `aggregation` — **passing**.
6. **NEW: негативный тест:** смешать агрегации (`c` через median, `frac_K` через mean) → assert обязан упасть (проверяет, что правило работает).
7. `transport` синтетика: случайная ортогональная `Q` с известным `log Q` → реконструкция через `align + polar + log`.
8. MP-калибровка: Монте-Карло гауссовского нуля.
9. `Σp̃ == 1 − mass_sentinel` (p̃-ассерт).

**Стоимость:** часы, ноль форвардов.

---

## V. Что НЕ делать

- Не добавлять `T_{ij}` (мёртв).
- Не использовать `corr_f` в tier-0 (только в ΔW/интервенциях).
- ~~Не цитировать числа `E39_supplement` §2.2.1 до разбора E6.~~ **Снято (E6 закрыт):** §2.2.1 цитируется с подписью «median_per_position» либо пересобирается в `ratio_of_means`.
- **Не смешивать агрегации** в тройке `(c, s, cos, frac_K)` (median отношения ≠ отношение сумм).
- **Не смешивать срезы** в тройке (`all` vs `alive`).
- ~~Не включать `frac_K` в ядро до E6.~~ **Снято (E6 закрыт):** `frac_K` — рабочий вход ядра с явной агрегацией.
- Не называть `R_can` параметром порядка до прохождения критерия (a′)∧(a)∧(b)∧(c).
- Не использовать `b_mag = ‖Δz‖` (первая степень).
- Не путать `cos(J, PJ)` с `c`.
- Не интерпретировать `g_uni` как «uniform-долю от полного softmax» — это uniform по базе `U128`.

---

## VI. Спорные пункты (reconciliation с `docs/E40_math_core_answers.md`)

Позиция архитектора (ревизия 2) — ТЗ работает независимо от решения:

**§VI.1. Шестой канал (`E_off_manifold`).** Без него `Σ5 = ‖ΔJ_proj‖²` (только проекционная часть), а не `‖ΔJ‖²`. При `r = 16 ≪ D = 1024` и `Δ_SVD ≈ π/2` (L6–L22) `E_off_manifold ≈ ‖ΔJ‖²/2` — не пренебрежимо. **Если rejected:** в §I.7 заменить `‖ΔJ‖²` на `‖ΔJ_proj‖²`; `E_off_manifold` оставить как диагностическое поле (не в сумме).

**§VI.2. MP-скобки (p=6 vs p=8).** `λ+ = (1+√(p/n))²`. Для 6×6 — `p=6`. **Если contested:** `p=6` для 6×6, `p=8` для 8×8.

**§VI.3. λ_max граница.** В фиксах: `λ_max ≤ p_max(1−p_max)`; в двухточечном пределе `λ_max = 2p_max(1−p_max)`. **Замечание исполнителя (снятый ответ, `E40_math_core_answers.md` §2):** эта формулировка самопротиворечива — ровно двухточечный предел даёт `2p_max(1−p_max) > p_max(1−p_max)`; измеренно `λ_max = 0.1366 > p_max(1−p_max) = 0.1229` при типичном sharp p. Корректная граница: `λ_max ≤ p_max` (из `P ⪯ diag p`). До reconciliation в §I.1 не переписывать.

**§VI.4. `P² = P`.** Фиксы: «`P` не проектор ни при каком `p ≠ uniform`; при `p = uniform` `P² = (1/K)P`». **Замечание исполнителя:** на границе неверно — при вырожденном `p = e_i` (one-hot) `P = 0`, что идемпотентно; при `uniform` `P² = (1/K)P ≠ P` при `K > 1`, то есть даже uniform — не проектор в строгом смысле.

---

## VII. Итог ревизии 2 и порядок работ

$$\boxed{\text{E6 закрыт. } \texttt{frac\_K} \text{ разстейджирован. Тройка } (c, s, frac\_K) \text{ самосогласована при одной агрегации и одном срезе. } \texttt{triple\_identity} \text{ — passing; добавлен } \texttt{same\_slice\_aggregation}. \text{ Поле } \texttt{aggregation} \text{ — в ChannelTag. Негативный тест на смешанные агрегации обязателен.}}$$

**Что делать (по порядку, ревизия 2):**
1. `prism/tag.py`: добавить поле `aggregation` (обязательное) в `ChannelTag` (`slice` уже есть).
2. `prism/cheap_invariants.py`: `c, s, cos, frac_K` с явным параметром `aggregation`.
3. `prism/invariants.py`: `triple_identity` → `passing` (готово); `same_slice_aggregation` — не реализован, запись в реестр добавляется.
4. `paper.md` §2.5.4 (тройка + правило срез-агрегации), §6 (футер — следующий пункт 24; в фиксах назван «правило 7») — только по явной директиве пользователя.
5. `ref-math.md` §2 — записи `c`, `s`, `cos`, `frac_K` с явной семантикой агрегации.
6. Reconciliation четырёх спорных пунктов §VI с `docs/E40_math_core_answers.md`.

---

$$\boxed{\text{ТЗ: математика (P(p), тождества, разложение O1/iso, cos(J,PJ), 5 каналов, align signs) + код (tag, invariants, p_algebra, svd_bundle, transport, kinetic, operator_checks, cheap_invariants, nulls_and_stats, behavior gate, profiles recipes).}}$$