# ТЗ на корректировку серии E-38 по факту JSON vs paper

Документ структурирован по принципу «где → что → на что менять → почему». Основан на сверке 8 JSON-файлов (`exp38a-i`) с `paper.md`, `ref-exp.md`, `E38.md`. **Все числа в JSON — верны**; менять нужно **интерпретации и формулировки**. Один пункт (`trivial_frac` в E-38d) требует пере-генерации JSON, остальные — правки текста и/или латентных багов.

---

## A. Фактические ошибки: числа paper ≠ JSON

### A.1. E-38b — L22 неверно помечен `no_artifact`

| | |
|---|---|
| **Где** | `paper.md` §4.3 (E-38b); `ref-exp.md` карточка E-38b; `E38.md` E-38b section |
| **JSON (факт)** | `"22": "l22_position_sensitive"`; `PR_modes.paired_diff_ci_pos0_minus_body = [-0.445, -0.045]` (CI исключает 0); warning: `"L22 pos0 PR = 1.41 vs paper 4.6 (ratio 0.31)"` |
| **Текст (неверно)** | «L22 (GDN): paired bootstrap … PR_modes [−0.305, +0.012] … → **no_artifact**. Старая строка v4.0 «L22 — position_sensitive» **опровергнута** новым протоколом.» |
| **Заменить на** | «L22: **`l22_position_sensitive`** — paired CI для PR_modes = [−0.445, −0.045], исключает 0. Paper's 4.6 для pos0 PR **не воспроизводится** (computed = 1.41, ratio = 0.31) — вероятно, другая величина (min-over-positions из paper §20, не pos0).» |
| **Удалить** | фразу «опровергнута новым протоколом» |
| **Причина** | числа и CI в paper не совпадают с JSON; warning в JSON явно фиксирует несоответствие 4.6 |

---

### A.2. E-38g — ρ_our неверное значение, интерпретация «не значимо» ложная

| | |
|---|---|
| **Где** | `paper.md` §4.3 (E-38g); `ref-exp.md`; `E38.md` |
| **JSON (факт)** | `rho_our_S_frac_tail.median = -0.2428`, `ci = [-0.334, -0.182]` — CI **исключает 0**, значимо |
| **Текст (неверно)** | «ρ_our = **−0.041**, CI [−0.120, +0.009] (не значимо; sensitivity к completion: **−0.166** при random-QR — **completion_sensitive = false**)» |
| **Заменить на** | «ρ_our = **−0.243**, CI [−0.334, −0.182] — **значимо** (CI ∌ 0): tail-sparsity есть и у нашего оператора. ρ_lens = −0.927 — значительно сильнее. Оба оператора tail-структурны; lens — сильнее.» |
| **Удалить** | `completion_sensitive: false`, `rho_our_random_qr: −0.166` — **полей нет в JSON** |
| **Причина** | (а) числа paper устарели; (б) поля `completion_sensitive` / `rho_our_random_qr` удалены P0-фиксом (null completion снят) — paper ссылается на несуществующее |

---

### A.3. E-38g — метка `jlens_not_tail_modes` семантически противоположна смыслу ρ_lens = −0.927

| | |
|---|---|
| **Где** | `paper.md` §4.3 (E-38g); `E38.md`; `ref-exp.md`; `exp38g_jlens_link.json` `outcomes.tz_preregistered_outcome` |
| **JSON (факт)** | `tz_preregistered_outcome = "jlens_not_tail_modes"` при `rho_lens_med = −0.927`. Код применяет pre-registered критерий `< 0.3`, не различая «≈ 0» (нет корреляции) и «≪ 0» (сильная отрицательная = tail-sparse) |
| **Проблема** | ρ_lens = −0.927 означает **сильную tail-sparsity** (энергия концентрируется в хвосте при низкой энтропии). Метка «not_tail_modes» говорит **обратное**. Тот же класс бага, что `[M-4]` инверсия |
| **Заменить на (paper/E38.md)** | «pre-registered rule `|ρ| < 0.3` не различает «нет корреляции» и «сильная отрицательная корреляция». Факт: ρ_lens = −0.927 = **сильная tail-sparsity у lens**; ρ_our = −0.243 = слабая, но значимая. Метка `jlens_not_tail_modes` в paper **не соответствует** значению — переименовать в **`rho_our_weak_rho_lens_strong_negative`** (описательно)» |
| **Заменить на (exp38g, если хотим оставить pre-registered)** | Добавить в `meta.rho_interpretation`: `"rho < -0.3 = strong tail-sparsity; -0.3 < rho < 0.3 = no correlation; rho > 0.3 = anti-tail-sparsity"` — и перегенерировать `tz_preregistered_outcome` с учётом знака |
| **Причина** | label противоречит смыслу числа; читатель paper решит «lens не tail-sparse», хотя ρ_lens = −0.93 говорит обратное |

---

### A.4. E-38c — `[M-4]` инвертирован в paper/E38.md относительно definition и code

| | |
|---|---|
| **Где** | `paper.md` §4.4 (E-38c выводы); `ref-exp.md`; `E38.md` Status; `exp38c_degeneracy_preflight.json` `meta.direction`/`meta.note` (уже правильно) |
| **JSON (факт)** | `min_ratio_t = 0.071/0.089/0.072/0.082/0.077/0.078` (все ≪ 1.0); `aggregate = "near_degeneracy"`. Definition (`ref-math.md` §4 E.4): `< ε` → **применим** |
| **Текст (неверно)** | «[M-4] pre-flight **не пройден**: … non-Abelian формализм **неприменим** … WZ-мораторий **сохраняется**» |
| **Заменить на** | «[M-4] pre-flight **пройден** (near-degeneracy везде → формализм **применим**); применимость **ограничена хвостом** спектра (gap_min драйвится хвостовыми зазорами, топ-моды well-separated — E-38h). WZ-мораторий **снят для tail-анализа**, сохраняется для top-mods.» |
| **Дополнительно** | В `paper.md` §4.4 удалить формулировку «Пройден ⟺ доля well-separated позиций > 0» — она **не является** критерием `[M-4]` (обратное направление). `fraction_ws = 0` = все near-degenerate = **прохождение** |
| **Причина** | прямое противоречие ref-math §4 E.4 и коду `exp38c` |

---

### A.5. E-38c — формулировка «fraction_ws = 0 → near_degeneracy» неоднозначна

| | |
|---|---|
| **Где** | `paper.md` §4.4; `E38.md`; `ref-exp.md` E-38c |
| **Текст** | «fraction_well_separated = 0 на всех слоях → near_degeneracy» |
| **Проблема** | `fraction_ws = 0` означает «ни один (t, doc) не well-separated» = **все** near-degenerate = **прохождение** [M-4]. `fraction_ws > 0` означало бы «есть well-separated точки, формализм не везде применим». Текущая формулировка читается как «нет well-separated точек → плохо» |
| **Заменить на** | «`fraction_well_separated = 0` = все точки near-degenerate = [M-4] **пройден** (формализм применим во всех точках). `fraction_ws > 0` означало бы «есть well-separated точки, формализм применим не везде».» |

---

## B. Тавтологии, помеченные как эмпирические результаты

### B.1. E-38h — `defect_top ≈ 0` — тождество фиксированного проектора

| | |
|---|---|
| **Где** | `ref-exp.md`; `E38.md`; `exp38h_two_channel.py` (`meta.tautology_warning` — ДОБАВЛЕНО, 2026-10-27) |
| **JSON (факт)** | `top.right_hidden = {0.0, 0.0, 0.0}`; `top.left_token = {0.0, 0.0, 0.0}` — exact 0.0 в текущем JSON (re-run 2026-10-27; старый прогон 2026-10-26: 3.1e-16–4.2e-16) — машинный эпсилон **для всех triples и s** |
| **Проблема** | При **фиксированных** P_L, P_R (оба ранга 16) `A_top(s) = U_ref·M(s)·V_refᵀ` имеет правый фрейм `span(V_ref)` **постоянным по s**. `procrustes_align(Q_top(a), Q_top(b))` — чистая смена базиса, композиция `R23·R12 = R13` **точно**. Это **следствие определения канала**, не измерение абелевости |
| **Добавить в paper/E38.md перед результатами** | «**Тавтология (фикс. проектор)**: `defect_top ≈ 0` — следствие фиксированных P_L, P_R (оба ранга 16): правый фрейм A_top(s) постоянен по s, композиция R23·R12 = R13 тождественна. Это **не** эмпирическое свидетельство абелевости top-канала; сценарий «неабелев в top-16» (E-38d scenario 2) в этой постановке **невозможен**. Информативна только tail-сторона: `defect_tail = 0.14–0.27` — измеренный неабелев сигнал.» |
| **Добавить в `exp38h_two_channel.py`** | `meta.tautology_warning: "defect_top=0 is a consequence of the fixed P_L/P_R, not a measurement"` |
| **Причина** | сейчас читается как «top Abelian по данным», на деле — гарантия конструкцией |

---

### B.2. E-38i — `residual = 5.5e-16` — алгебраическое тождество

| | |
|---|---|
| **Где** | `paper.md` §4.3 (E-38i); `ref-exp.md`; `E38.md`; `exp38i_factorization.py` docstring |
| **JSON (факт)** | `residual_max = 5.45–5.48e-16` для всех s. Код сам в verdict пишет «tautological by construction», но в paper/E38.md это может читаться как «факторизация сохранена под edit» |
| **Добавить в paper/E38.md перед результатами** | «**Тавтология**: residual = 5.5e-16 — **алгебраическое тождество** для любого W (факторизация `J = c·M(I−ûûᵀ)` closed form = dense W @ J_norm). Проверка подтверждает корректность пайплайна, **не является свидетельством «сохранения под edit»**. Информативен только **drift**: `J_rel_drift_max` 0.355→1.152 (s=10→1000), `median_shared` 0.219→0.656 — лист «перетягивается» в соответствии с ростом bridge 0.60→0.91 в E-16.» |
| **Добавить в `exp38i_factorization.py`** | `out["tautology_warning"] = "residual = 0 for any W; drift carries the information"` — явное поле в JSON |
| **Причина** | сейчас meta говорит «verdict: identity holds», что звучит как empirical finding |

---

### B.3. E-38b — `align_K1` на gauge-free J измеряет null-space alignment, не residual gauge

| | |
|---|---|
| **Где** | `paper.md` §4.3 (E-38b); `E38.md`; `exp38b_position_audit.py` (поле `align_K1`) |
| **JSON (факт)** | `meta.naming` уже предупреждает: «align_K1 computed on the gauge-free J (kept component) with max-over-modes; NOT comparable to E-37's 0.895→0.710» |
| **Проблема** | `1/√K ∈ ker J^(p)` по построению (P(p)·1 = 0). `max_k |⟨U_k, 1/√K⟩|` на `J^(p)` = **alignment нуль-пространства** с uniform, **не** residual gauge. Значения 0.90–0.97 — «насколько хорошо P(p) убил uniform» (базисно-зависимая величина, LAPACK) |
| **Добавить в paper/E38.md** | «`align_K1` на gauge-free J измеряет **alignment нуль-пространства** J^(p) с uniform — то есть «насколько хорошо P(p) выполнил работу», **не** «сколько gauge осталось». Значения 0.90–0.97 — не «gauge присутствует». Вердикт `align_edge_effect` не срабатывает на текущих данных.» |
| **В `exp38b_position_audit.py`** | Переименовать поле в JSON `align_K1` → `null_space_uniform_alignment` (или добавить `meta.align_K1_semantics` с явной формулировкой) |
| **Причина** | семантически неверная интерпретация в paper |

---

### B.4. E-38d vs E-38h — `max_theta = π` (top-r J_t) vs `defect_top ≈ 0` (top-r ΔW проектор) — **не противоречат**

| | |
|---|---|
| **Где** | `ref-exp.md`; `E38.md` |
| **JSON (факт)** | E-38d: `max_theta = 3.1416` на L6–L22 (top-r **спектра J_t**, position-dependent). E-38h: `defect_top ≈ 0` (≤ 4e-16; exact 0.0 в текущем JSON, 2026-10-27) (top-r **проекторного разбиения ΔW** с фиксированными P_L, P_R) |
| **Проблема** | читатель видит «E-38h: top Abelian» и «E-38d: top non-Abelian» — противоречие |
| **Добавить в paper/E38.md** | «**Не противоречит**: E-38d измеряет `{θ_j}` на top-r **спектра J_t** (position-dependent правые сингулярные векторы) → `max_theta = π` → non-Abelian. E-38h измеряет defect на top-r **проекторном разбиении ΔW** с **фиксированными** P_L, P_R → defect ≈ 0 → тривиально Abelian. Разные объекты: J_t vs канал ΔW.» |
| **Причина** | без явного разделения два результата выглядят как противоречие, хотя они о разных вещах |

---

## C. Латентные баги в коде (не портят текущие числа, исправить до следующего прогона)

### C.1. `exp38a_uhat_dynamics.py` — `kapp` инициализация zeros
```python
kapp = np.zeros((N, S - 1), dtype=np.float64)  # kapp[:, 0] never filled
```
**Заменить на** `np.full((N, S - 1), np.nan, dtype=np.float64)`.
**Причина**: `kapp[:, 0]` = 0 вместо NaN — если когда-нибудь срез включит `t=0`, будет ложная плоская кривизна.

### C.2. `exp38a_uhat_dynamics.py` — NaN-нормы не считаются в `kappa_nan_count`
```python
if float(norms[i, t - 1:t + 2].min()) < 1e-10:
```
**Проблема**: если `norms` содержит NaN, `min()` = NaN, `NaN < 1e-10 = False` → идёт в `geodesic_curvature` с NaN → возвращает NaN → счётчик не увеличивается.
**Заменить на** `if not torch.isfinite(norms[i, t-1:t+2]).all() or float(norms[i, t-1:t+2].min()) < 1e-10:`.

### C.3. `exp38a_uhat_dynamics.py` — `s` не защищён от вырожденных норм
```python
u = H / norms.clamp_min(1e-12)
```
**Проблема**: при `‖h_t‖ = 0` → `u = 0` → `s = 1.0` вместо NaN.
**Заменить на** `u = H / torch.where(norms >= 1e-10, norms, torch.nan)[:, :, None]`.

### C.4. `exp38a_uhat_dynamics.py` — `pos0_ratio` при нулевой норме `h0` считается «sink», а не NaN
```python
pos0_ratio = norms[:, 0] / norms[:, 1:].nanmedian(dim=-1).values.clamp_min(1e-12)
```
**Проблема**: `‖h0‖ = 0` → ratio = 0 → попадает в `frac_norm_h0_below_0.1x_median` как «sink».
**Заменить на** `pos0_ratio = torch.where(norms[:, 0] > 1e-12, ..., torch.nan)`.

### C.5. `exp38e_MK_diagnostic.py` — нет `torch.isfinite(H).all()` / `torch.isfinite(J23).all()`
**Добавить** после `H = ck23["H"]`, `J23 = ck23["J"]`:
```python
if not torch.isfinite(H).all(): raise ValueError("non-finite H in cache")
if not torch.isfinite(J23).all(): raise ValueError("non-finite J23 in cache")
```

### C.6. `exp38e_MK_diagnostic.py` — `ratio_median` не капнут (pos0 капнут)
**Добавить**: `ratio = ratio.clamp(max=POS0_CAP)` перед `ratios_docs = ...`; сохранить `ratio_median_capped` в JSON.

### C.7. `exp38e_MK_diagnostic.py` — `r_t_expected` (all) vs `jtu_ratio` (valid) — разные наборы
**Заменить**: `r_t_expected = float(np.nanmedian(norms[valid].numpy()) / sqrtD)`.

### C.8. `exp38g_jlens_link.py` — нет `torch.isfinite(H).all()` / `torch.isfinite(J_our).all()`
**Добавить** после загрузки H/J_our.

### C.9. `exp38g_jlens_link.py` — `at_random` слишком широкое окно `[0.5·rb, 3·rb]`
**Проблема**: `random_baseline = 0.0996`; `at_random` покрывает `[0.05, 0.30]`. `ov = 0.25` (2.5× rb) попадает в `at_random`, хотя выше случайного.
**Варианты**: (а) сузить до `[0.5·rb, 2·rb]`; (б) задокументировать в `meta.verdict_rule`, что `at_random` здесь = «в пределах фактора 3 от случайного» (уже частично есть в `random_baseline_note`). Предпочтительно (а) + явная оговорка.

### C.10. `exp38i_factorization.py` — `shared_idx256` может быть пустым → silent no-op
**Добавить** после построения `shared_idx256`:
```python
if shared_idx256.numel() == 0:
    raise ValueError("U128 and E-16 token sets do not intersect")
if ts256.unique().numel() != ts256.numel():
    raise ValueError("E-16 token set has duplicates (double-add would corrupt)")
```

### C.11. `exp38i_factorization.py` — `factor_residual` return type не проверен
**Добавить** `assert isinstance(res[0], float), f"factor_residual returned {type(res[0])}"` — иначе `json_dump` упадёт на 0-dim tensor без явной причины.

### C.12. `exp38i_factorization.py` — `resid_max_all` включает s=0
**Изменить** на `max(v["factorization_residual_max"] for s, v in out_states.items() if s > 0)` — s=0 неинформативен и документирован как baseline.

### C.13. `exp38d_holonomy_spectrum.py::_spectrum_batched` — `trivial_frac` мёртв (NaN-padding)
```python
trivial = np.all(theta_flat <= 1e-9, axis=1)
```
**Проблема**: `np.all([NaN, ...] <= 1e-9)` = False для тривиальных H (все углы ниже порога → NaN padding).
**Заменить на** `trivial = np.all(np.isnan(theta_flat) | (theta_flat <= 1e-9), axis=1)` **или** `n_ang == 0` (уже возвращается из `_spectrum_batched`).

### C.14. `exp38d_holonomy_spectrum.py::_spectrum_batched` — lone −1 (angle −π) отбрасывается
```python
pos = angs[angs > 1e-6]
```
**Проблема**: LAPACK может вернуть `−1 − 0j` → angle = −π → `−π > 1e-6 = False` → угол теряется. `holonomy_spectrum` этого не допускает (`.abs()`).
**Заменить на** `pos = np.abs(angs); pos = pos[pos > 1e-6]` — синхронизировать с `holonomy_spectrum`.

---

## D. Документация / схема JSON

### D.1. E-38g — удалить `completion_sensitive` / `rho_our_random_qr` из paper
- **Где**: `paper.md` §4.3, `ref-exp.md` E-38g.
- **Факт**: полей нет в `exp38g_jlens_link.json` (P0-фикс снял null completion).
- **Действие**: удалить оба поля из текста paper/ref-exp **или** восстановить их в коде (сравнить ρ_our при двух random-QR completions до 1024 modes; если |Δρ_our| < 0.05 → `completion_sensitive: false`). Рекомендация: **удалить** — P0-фикс обосновал отказ от null completion, проверка бессмысленна.

### D.2. E-38d — `anomaly_count 1397–4032` в `ref-exp.md` не соответствует JSON
- **Где**: `ref-exp.md` E-38d.
- **Факт**: в `exp38d_holonomy_spectrum.json` нет поля `anomaly_count`.
- **Действие**: удалить упоминание или найти источник (возможно, из ранней версии JSON).

### D.3. E-38i — `J_rel_drift_max` == `J_rel_drift_max_shared` (redundancy)
- **Где**: `exp38i_factorization.py` docstring: «115 of 128 rows are edit-invariant, so max/median over ALL rows underreports».
- **Факт**: для **max** это неверно — unshared rows дают drift = 0, что **не увеличивает** max. `J_rel_drift_max ≡ J_rel_drift_max_shared` тождественно.
- **Изменить docstring**: «max over ALL rows **не** underreports (equals max over shared rows); median over ALL rows underreports (115 zeros dilute) — поэтому отчитываем median_shared».

### D.4. `E38.md` Status для E-38d–g — противоречие с JSON
- **Где**: `E38.md` (E-38d/e/f/g помечены «⏸ не запускается до [M-4]»).
- **Факт**: в JSON есть результаты для E-38d, E-38e, E-38f, E-38g.
- **Действие**: либо обновить статус на «✅ выполнен», либо явно: «Запущены несмотря на формальный статус [M-4] — поскольку near-degeneracy означает применимость, а не блокировку».

### D.5. E-38h — `top_r_subspaces` возврат V/U — верифицировать
- **Где**: `exp38_hi_common.top_r_subspaces`.
- **Проблема**: комментарии в `exp38h` предполагают возврат V (правые); если функция возвращает U — JSON-поля `right_hidden` / `left_token` инвертированы.
- **Действие**: добавить явный docstring/assert о возвращаемой стороне.

---

## E. Итоговая сводка правок

| # | Где | Действие |
|---|---|---|
| A.1 | paper/ref-exp/E38.md (E-38b) | L22: `no_artifact` → **`l22_position_sensitive`**; удалить «опровергнута» |
| A.2 | paper/ref-exp/E38.md (E-38g) | ρ_our −0.041 → **−0.243**; CI [−0.120, +0.009] → **[−0.334, −0.182]**; «не значимо» → «значимо» |
| A.3 | paper/E38.md/exp38g (E-38g) | `jlens_not_tail_modes` → **`rho_our_weak_rho_lens_strong_negative`** (или уточнить pre-registration с `|ρ|`) |
| A.4 | paper §4.4 / E38.md Status / ref-exp (E-38c) | `[M-4]` **«не пройден / неприменим»** → **«пройден / применим (tail-only); мораторий снят для tail»** |
| A.5 | paper §4.4 (E-38c) | Переформулировать «fraction_ws = 0 → near_degeneracy» → «fraction_ws = 0 = все near-degenerate = [M-4] пройден» |
| B.1 | paper/ref-exp/E38.md + `exp38h.py` meta (E-38h) | Явно: `defect_top ≈ 0` — **тавтология фикс. проектора**; информативен только tail |
| B.2 | paper/ref-exp/E38.md + `exp38i.py` meta (E-38i) | Явно: residual = 0 — **алгебраическое тождество**; информативен только drift |
| B.3 | paper/E38.md + `exp38b.py` meta (E-38b) | `align_K1` = **null-space alignment**, не residual gauge; переименовать/задокументировать |
| B.4 | paper/E38.md (E-38d + E-38h) | Явно разделить: E-38d (top-r J_t, non-Abelian) ≠ E-38h (top-r ΔW проектор, тривиально) |
| C.1–C.14 | `exp38a/d/e/g/i.py` | Латентные баги: NaN-init, isfinite guards, clamp, asserts, angle sync, trivial_frac |
| D.1 | paper/ref-exp (E-38g) | Удалить `completion_sensitive` / `rho_our_random_qr` |
| D.2 | ref-exp (E-38d) | Удалить `anomaly_count 1397–4032` |
| D.3 | `exp38i.py` docstring | Уточнить: max не underreports; median underreports |
| D.4 | `E38.md` Status | Обновить статус E-38d–g (уже запущены) |
| D.5 | `exp38_hi_common.top_r_subspaces` | Добавить docstring/assert о возвращаемой стороне V/U |

---

## F. Порядок выполнения

**Приоритет 1** (числовые расхождения paper↔JSON): A.1, A.2, A.4, D.1 — **до любых других правок**, иначе paper публикует неверные числа.

**Приоритет 2** (семантика/тавтологии): B.1, B.2, A.3, B.4, B.3 — меняют интерпретацию.

**Приоритет 3** (латентные баги): C.1–C.14, C.13–C.14 (после пере-генерации E-38d JSON).

**Приоритет 4** (документация): D.2–D.5.

**Ни один эксперимент не «опровергнут».** Три вывода paper (L22, ρ_our, [M-4]) требуют перезаписи; три «подтверждения» (E-38h top, E-38i residual, E-38d trivial_frac) понижены до тавтологии/bug; несколько латентных багов исправляются до следующего прогона.