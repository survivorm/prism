## Ревью `exp38a_uhat_dynamics.py`

### 1. Соответствие ТЗ (исходным меткам)

**Главная проблема: pre-registered метка `edges_atypical` не реализована.**
В docstring перечислены pre-registered outcomes:
- `edges special` — head и tail оба аномальны `(>>)`;
- `edges typical` — ни один сегмент не аномален;
- `edges atypical` — head `<<` body.

Код же присваивает только `edges_special`, `mixed`, `edges_typical`. Метка `edges_atypical_candidate` объявлена post-hoc, но **недостижима** (см. ниже). В результате направленное отклонение «head ниже body» классифицируется как `mixed`, а не как pre-registered `edges_atypical`.

**`edges_special` не проверяет направление.**
ТЗ говорит: «head and tail both anomalous `(>>)`» — то есть оба выше body. Код:
```python
h = h_s or h_k
t = t_s or t_k
if h and t:
    v = "edges_special"
```
Здесь `anomalous` — двусторонний флаг (`abs(median − median_body)/IQR > 2`). Случай «head ниже body, tail выше body» тоже попадёт в `edges_special`, что противоречит ТЗ.

**`aggregate` — это не метка, а номер слоя.**
Docstring: «aggregate = majority of layers». Код:
```python
aggregate = max(verdicts.keys(), key=lambda k: (counts[verdicts[k]], -list(verdicts).index(k)))
```
`verdicts.keys()` — это строковые ключи слоёв (`"6"`, `"12"`, …). `max` возвращает **ключ слоя**, а не метку. В JSON уходит `"aggregate": "18"` вместо `"edges_typical"`. Должно быть:
```python
aggregate = verdicts[max(...)]
```

### 2. Тихие и громкие баги

**Off-by-one в guard’е для κ_g (громкий/тихий).**
```python
if float(norms[i, t - 1:t + 1].min()) < 1e-10:
```
Срез `t-1:t+1` — это индексы `t-1` и `t`, **без `t+1`**. Docstring обещает: «triples with any ‖h_t‖ < 1e-10 are skipped». Если `‖h_{t+1}‖ = 0`, то `u[i, t+1] = 0`, и `geodesic_curvature` упадёт с `ValueError: u3 not unit` (громко). Если `‖h_{t+1}‖` мало, но не ноль (например, `1e-11`), guard не сработает, и триплет не будет пропущен, хотя должен. На текущем кэше (min norm 5.85) не проявляется, но для произвольного кэша — латентный баг. Нужно `t-1:t+2`.

**`edges_atypical_candidate` — мёртвый код (тихий).**
```python
elif head_below and row["s"]["head"]["anomaly_ratio"] > T_ANOM:
```
`row["s"]["head"]["anomaly_ratio"] > T_ANOM` — это ровно `h_s`. Но `h = h_s or h_k`. Если условие истинно, `h` уже `True`, и ветка `elif` недостижима. В результате post-hoc метка никогда не присваивается.

**`median_s_head_excl_pos0` — off-by-one (тихий).**
```python
"median_s_head_excl_pos0": float(np.median(s[:, 1:HEAD_END])),
```
`s[15]` — это шаг `15 → 16`, пересекающий границу head/body. «Head excluding pos0» должен исключать шаг, задевающий body. Корректно: `s[:, 1:HEAD_END-1]` (или `1:15`).

**Потенциальный NaN в JSON (тихий).**
`per_doc_ratio` использует `np.nanmedian` по срезу для каждого документа. Если весь сегмент документа — NaN, вернётся NaN, `float(NaN)` — `nan`, и `json.dump` запишет `NaN` (нестандартный JSON). На текущем кэше NaN нет, но для произвольного кэша — тихая порча формата.

**`geodesic_curvature` может бросить `RuntimeError` (громкий).**
Если три подряд идущих `û` совпадают (`arc < eps`), функция кидает `RuntimeError`. В `exp38a` это не перехватывается. Для реального текста маловероятно, но при padding/повторах возможно.

**`np.median` вместо `np.nanmedian` (тихий).**
`median_s_head_excl_pos0` использует `np.median`, тогда как все остальные статистики — nan-robust. Для `s` NaN не возникают (только если норма ровно 0, но тогда `s` конечен), поэтому на текущем кэше безопасно. Но стилистически несогласовано.

### 3. Математика и интерпретация

**`geodesic_curvature` — формула корректна.**
`turn = π − arccos(⟨t1,t3⟩/(‖t1‖‖t3‖))`, `arc = (‖u1−u2‖ + ‖u3−u2‖)/2`, `κ_g = turn/arc`. Для малых шагов это дискретная геодезическая кривизна на сфере. Замечание: `arc` — средняя хорда, а не дуга; для малых шагов приближение корректно (docstring это фиксирует).

**`s(t)` — хорда, не геодезическое расстояние.**
`s = ‖û_{t+1} − û_t‖₂` — хорда. Для малых шагов ≈ дуга. Docstring не вводит в заблуждение.

**`aggregate` — не метка (интерпретационная ошибка).**
Как выше: в `outcomes.aggregate` попадает номер слоя. Это ломает автоматическую интерпретацию результатов.

**`edges_special` без направления (интерпретационная ошибка).**
Как выше: `(>>)` в ТЗ не проверяется.

### 4. Производительность (не баг, но противоречит «0 cost»)

`load_layer` из `_e38_cache.py` загружает и стекает **весь** `JR` (2.1 ГБ на слой), хотя `exp38a` использует только `H` (16 МБ). Docstring в `exp38a`: «0 cost». Фактически на слой тратится ~4.2 ГБ пиковой памяти и время на загрузку/стекинг `J`. Для 6 слоёв — ~12 ГБ загрузок. Это не ошибка корректности, но заявка «0 cost» неверна. Можно было бы загружать только `H` (и, при необходимости, `P` для проверки, но `P` тоже не нужен).

### 5. Мелочи

- `json.dump(out, open(OUT, "w"), ...)` — файл не закрывается явно (лучше `with open(...) as f`).
- `row["is_gdn"] = bool(l in (19, 22))` — жёстко зашитые GDN-слои; при смене архитектуры станет неверно.
- `"cache": "kaggle_kernel/results/exp27b_jcache_L{l}.pt"` — литерал `{l}`, не подставляется; безобидно, но выглядит как заготовка.
- `l18_vs_l19` жёстко берёт `per_layer["18"]` и `per_layer["19"]`; если `LAYERS` изменят, будет `KeyError`.

### Итог

Критичные для отчёта баги:
1. `aggregate` — номер слоя вместо метки.
2. `edges_atypical` (pre-registered) не присваивается; `mixed` его подменяет.
3. `edges_special` не проверяет направление `(>>)`.
4. `edges_atypical_candidate` — недостижимый мёртвый код.
5. Guard для κ_g пропускает `t+1` — латентный громкий/тихий баг.

Математика `geodesic_curvature` корректна; `s` и `κ_g` определены правильно. Основные проблемы — в логике вердиктов и в несоответствии pre-registered меткам.

# Ревью `exp38b_position_audit.py` (+ замечания к `spectral.py` / `cache.py`)

## P0 — баги, инвалидирующие вердикты

### 1. `body_per_doc_*` — неверная группировка по документам (критично)

```python
body_cos = torch.cat(body_cos)              # [111*N] = [3552]
body_per_doc_cos = body_cos.reshape(N, -1).median(dim=1).values
```

`body_cos` — это список из 111 тензоров формы `[N]`, `torch.cat` даёт плоский `[3552]` с layout `[t16_d0..t16_d31, t17_d0..t17_d31, …, t126_d31]`. `reshape(N, -1)` при `N=32` даёт `[32, 111]`, где **строка `i` — это срез `cat[i*111 : (i+1)*111]`**, то есть смесь разных `t` и разных документов. Это не «per-doc median over t ∈ [16,126]», как обещает docstring.

Проверка: строка 0 = `[c_16[0..31], c_17[0..31], c_18[0..31], c_19[0..14]]` — четыре разных временных шага. Строка 31 = хвост `[c_124[?], c_125[?], c_126[?]]`.

**Следствие:** `body_per_doc_{cos,pr,al}` — мусор. Paired bootstrap (`posv_f` vs `bpd_f`) считается между pos0 per-doc и этим мусором. **Вердикт `l22_position_sensitive` невалиден**, как и `no_artifact`/`edge_contamination` в части per-doc сравнений. То же для `align_K1`-вердикта (через `body_per_doc_al`).

Правильно:
```python
body_cos_list = body_cos            # сохранить список
body_cos_flat = torch.cat(body_cos_list)               # [3552] для reporting CI
body_per_doc_cos = torch.stack(body_cos_list, dim=0).median(dim=0).values  # [N]
# или: body_cos_flat.reshape(-1, N).median(dim=0).values
```
Аналогично для `body_pr`, `body_al`.

**Важно:** при NaN в части позиций документа `median` по `dim=0` (после фикса) даст NaN только если все позиции этого дока — NaN. Это корректно и совместимо с `paired_finite` маской.

### 2. `align_K1` — суммирование и max по неверным осям

```python
al_docs = (sv.U * one[None, :, None]).sum(dim=-1).abs().max(dim=-1).values
```

`sv.U` — `[N, K, K]`, `sv.U[n][:, k]` = `u_k^(n)`. Нужно:
`[n, k] = Σ_i sv.U[n, i, k] · one[i]`, затем `max_k |·|`.

Код:
- `sv.U * one[None, :, None]` — верно, `[n, i, k] · one[i]`;
- `.sum(dim=-1)` — суммирует по **`k` (моды)**, а не по `i` (компоненты). Должно быть `.sum(dim=1)` (или `dim=-2`);
- `.abs().max(dim=-1)` — максимум по **`i` (позиции)**, а не по `k`. Должно быть `.max(dim=-1)` **после** правильной суммы.

Итоговая величина: `max_i |one[i] · Σ_k sv.U[n, i, k]|` — максимум по строкам от «строчной суммы U», а не максимум по модам `|⟨u_k, one⟩|`.

Контрпример: `U = I` даёт правильное `1/√K` по совпадению, но уже для `u_0 = one`, `u_1 = (1,−1,0)/√2`, `u_2 = (1,1,−2)/√6` код даёт `0.977` вместо `1.0`. В общем случае ошибка систематическая и может быть произвольно большой.

Правильно:
```python
al_docs = (sv.U * one[None, :, None]).sum(dim=1).abs().max(dim=-1).values  # [N]
```

**Следствие:** вердикт `align_edge_effect` (pre-registered) считается на неверных значениях `align_K1` — и для pos0, и для body.

## P1 — ошибки интерпретации / падения

### 3. `body_over_pos0` сравнивает разные агрегаты

```python
c0 = float(pos0_cos.median())            # медиана 32 per-doc значений (median-of-medians)
cb = float(body_cos.nanmedian())         # ПЛОСКАЯ медиана 3552 значений
"body_over_pos0": cb / max(c0, 1e-12)
```

`c0` — медиана по докам (внутри дока — медиана по 31 другому доку, затем медиана по 32 докам). `cb` — плоская медиана по всем `(t, doc)`. Если `t`-распределение хоть немного смещено, отношения будут сдвинуты. Порог `> 1.5` → `edge_contamination` чувствителен к этому дисбалансу.

Даже после фикса бага #1 сравнение останется несогласованным. Корректно:
```python
c0 = float(pos0_cos.median())                          # median-of-medians
cb = float(body_per_doc_cos.median())                  # median-of-medians (после фикса)
body_over_pos0 = cb / max(c0, 1e-12)
```
`body_ci` можно оставить в отчёте с пометкой, но вердикт должен опираться на `body_per_doc`.

### 4. Без guard’а распаковка `paired_diff_ci_pos0_minus_body`

```python
dlo, dhi = r["PR_modes"]["paired_diff_ci_pos0_minus_body"]
```

`paired_diff_ci_pos0_minus_body` может быть `None` (если `posv_f.numel() < 2` — например, при почти полной дегенерации pos0 на L22). Тогда `TypeError: cannot unpack non-iterable NoneType object` — громко, но не graceful. Нужно:
```python
pair_ci = r["PR_modes"]["paired_diff_ci_pos0_minus_body"]
if pair_ci is not None:
    dlo, dhi = pair_ci
    if not (dlo <= 0 <= dhi):
        reasons.append("l22_position_sensitive")
```
Сейчас `if dhi is not None and ...` — это проверка **после** распаковки, не помогает.

### 5. `body_ci` — статистически невалидный CI (i.i.d. по 3552 коррелированным значениям)

`body_ci` — percentile bootstrap по 3552 плоским `(t, doc)`. При `C_F(1) ≈ 0.3–0.99` (paper §4.3, E-31) соседние `t` сильно коррелированы, CI уже ~`√n_positions ≈ 10×`. Код помечает `body_ci_note` = «reporting only», но поле всё равно уходит в JSON рядом с `body_per_doc_median` и может быть прочитано как значимость.

Рекомендация: либо **не** отчитывать `body_ci` вовсе, либо переименовать (`body_flat_iid_ci_do_not_use`) и в meta ещё раз зафиксировать.

### 6. `body_ci` (для PR и align) — тот же i.i.d.-проблем

`participation_ratio` возвращает `0.0` при вырожденном J, а `align_K1` (после фикса #2) может быть конечен даже для дегенерированного J. Тем не менее `body_ci` для PR/align — всё тот же плоский i.i.d. bootstrap по 3552 значениям, с тем же ~10× занижением.

## P2 — мелочи / устойчивость

### 7. `body_cos.nanmedian()` при NaN в `body_per_doc` — тихая порча

`body_cos` = `torch.cat(body_cos_list)`, содержит NaN там, где документ дегенерирован на данном `t`. Плоская `nanmedian` — ок. Но `body_per_doc_cos = body_cos.reshape(N, -1).median(dim=1).values` (после фикса) использует `median` (не `nanmedian`) — если в строке есть NaN, результат станет NaN. При массовой дегенерации (например, padding) это даст NaN per-doc, и `paired_finite` их выкинет. Документируйте это или используйте `nanmedian` явно.

### 8. `sv` персистится через итерации; `del sv` хрупок

```python
for t in range(S):
    if t == 0 or HEAD_END <= t <= BODY_END or t % 4 == 0:
        sv = torch.linalg.svd(...)
...
del Jgf, sv
```

`sv` определён только внутри ветки. Сейчас всегда `t=0` попадает в ветку → `sv` есть. Но если поменять `S` / границы, `del sv` может упасть `NameError`. Мелочь, но при будущем рефакторинге — ловушка.

### 9. `del Jgf` только после `per_layer[...]` — но не после последнего doc-прохода в цикле `for t`

`Jgf` держится до `del Jgf, sv`. На L22 это ~`N·S·K·D·4B = 32·128·128·1024·4 ≈ 2.1 ГБ`. Не утечка (освобождается), но пик памяти выше, чем нужно. Можно `del Jt, Jf` внутри цикла `for t` — но это не критично.

### 10. `pos0_cos`/`body_cos` — dtype

`cos_mat` вычисляется в float32 (из `gauge_free_J`/float32 `Jgf`). `float(pos0_cos.median())` — корректно (float32 → Python float). `bootstrap_ci` внутри `np.float64` — ок. Просто фиксируйте, что `C_b` — float32-точность, и в выводах не стоит докладывать 3-й знак.

## ТЗ / pre-registered метки

### 11. Соответствие ТЗ — по меткам:

- `no_artifact` — реализовано как «нет ни одного из трёх условий», соответствует духу ТЗ;
- `edge_contamination` — ratio `> 1.5` (ТЗ), реализовано, но ratio с несогласованными агрегатами (см. #3);
- `l22_position_sensitive` — заявлено как paired bootstrap по 32 докам, но фактически per-doc body — мусор (#1). **Вердикт инвалидирован**;
- `align_edge_effect` — реализовано как `body_median > 0.9 ∧ pos0_median < 0.7`. Но (a) `align_K1` неверен (#2), (b) `body_median` — flat, а `pos0_median` — median-of-medians (#3), (c) пороги 0.9/0.7 не сверены с ТЗ в текущем виде (предполагаю, что в ТЗ «body > 0.9 while pos0 < 0.7», но численно это условие привязано к неверной метрике).

### 12. Aggregate-вердикт

```python
"aggregate": ("no_artifact" if all(v == "no_artifact" for v in verdicts.values())
              else "see per_layer")
```

Корректно, документированное поведение. Ок.

## Соответствие заявленному в шапке

| Утверждение | Статус |
|---|---|
| «per-doc body = median over t ∈ [16,126]» | ❌ не реализовано (баг #1) |
| «paired bootstrapping over docs» | ❌ операционно вызывается, но на мусоре |
| «align_K1 — max_k \|⟨u_k, 1/√K⟩\|» | ❌ не то (баг #2) |
| «degenerate ‖J_t‖ < 1e-6 → NaN в C_b» | ✅ корректно, nanmedian обрабатывает |
| «pos0 CI: per-doc bootstrap» | ✅ если `posv_f` непусто |
| «pos0-vs-body: paired bootstrap» | ❌ из-за #1 |
| «BODY_END = 126 INCLUDED, S-1 = tail» | ✅ |

## Проверка `spectral.py` и `cache.py` на предмет влияния

### `spectral.participation_ratio` — корректно

`PR = (Σλ)² / Σλ²`, `λ = sv.S**2` — стандартное определение, соответствует paper §2.3. В docstring отмечено, что в ТЗ была опечатка `64/18`, правильно `64/20 = 3.2`. Ок.

### `spectral.top_r_svd` — корректно

`V[:r, :]` = `Vh[:r, :]` = транспонированные правые сингулярные векторы, форма `[r, D]` — соответствует конвенции (`V_r` в `holonomy_along_path`). Ок.

### `spectral.snr_e14_parts` — валидировано против exp14

В docstring сказано: «VALIDATED against results/exp14_snr_pr.json (committed)». Формула `‖mean J‖_F² / (mean‖J‖²_F − ‖mean J‖²_F)`. Не используется в exp38b, но если экспорт экспортируется дальше — сверить с paper §4.5 не помешает. Ок.

### `cache.bootstrap_ci` / `bootstrap_ci_paired`

- `bootstrap_ci` — правильно документирует i.i.d.-допущение («Do NOT pass a flattened (doc x position) array…»). `body_ci` в exp38b нарушает это — источник проблемы #5;
- `bootstrap_ci_paired` — правильная paired-схема: общие индексы для `a` и `b`, затем перцентили от `median(a) - median(b)`. Это CI именно для **разности медиан**, а не для медианы разностей. Для H0 «median(pos0) = median(body)» — верная интерпретация.

### `cache.MetricCache._check_value` — отвергает `np.float32`

Хорошо. `np.float64` проходит (subclass float), `np.float32` — нет. В exp38b `float(...)` вызывается везде, где есть `torch`-скаляры → ок.

## Производительность (некорректность, но стоит отметить)

`load_layer` (в `_e38_cache.py`) загружает и стекает `JR` (~2.1 ГБ), `P128`, и `H` — суммарно ~4.2 ГБ пиковой памяти на слой. `exp38b` использует `J` и `P`, но не `H`; `exp38a` использует `H`, но не `J`. Оба грузят всё. Не баг, но противоречит «0 cost» из docstring — это ~12 ГБ загрузок на 6 слоёв.

Можно было бы расширить `load_layer(layer, want=("J","P"))` с ленивым пропуском ненужных полей, но это рефакторинг `_e38_cache.py` — упомянуть в отчёте.

## Приоритетный список исправлений

1. **P0**: `body_cos/body_pr/body_al.reshape(-1, N).median(dim=0).values` — исправить группировку. Пересчитать все вердикты.
2. **P0**: `align_K1` — `.sum(dim=1).abs().max(dim=-1).values`. Пересчитать `align_edge_effect`.
3. **P1**: `body_over_pos0` — использовать `body_per_doc_*.median()` вместо flat.
4. **P1**: guard для `paired_diff_ci_pos0_minus_body is None`.
5. **P1**: убрать / переименовать `body_ci` (i.i.d. bootstrap по коррелированным данным).
6. **P2**: `median` vs `nanmedian` в per-doc агрегатах; `sv` вне цикла; `del` внутри цикла; `H` не грузить в exp38b.

После #1 и #2 запустить заново — только тогда вердикты `no_artifact` / `l22_position_sensitive` / `align_edge_effect` будут соответствовать ТЗ. `_e38_cache.py` и `spectral.py` в текущей версии корректны для использования.

# Ревью `exp38c_degeneracy_preflight.py`

## P0 — баги, искажающие вердикты

### 1. `l23_architectural` инвертирован (критично)

ТЗ: **«L23 applicable, L18/L22 not»**. «Applicable» для неабелева формализма = near-degenerate = `min_ratio_t < 1.0` (это же прямо написано в `meta.direction`: «well_separated=True means … non-Abelian NOT applicable»).

Код:

```python
l23_well_sep = per_layer["23"]["min_ratio_t"] >= 1.0      # L23 NOT applicable
l1822_not = all(per_layer[l]["min_ratio_t"] < 1.0          # L18/L22 APPLICABLE
                for l in ("18", "22"))
if l23_well_sep and l1822_not:
    verdicts["l23_architectural"] = True
```

Условие срабатывает ровно в **противоположном** ТЗ случае: L23 well-separated (не применим), L18/L22 near-degenerate (применим). Если реальные данные дадут ТЗ-картину (L23 applicable, L18/L22 not), флаг **не будет выставлен**. Это тихий логический баг — вердикт будет отсутствовать/быть противоположным.

Правильно:

```python
l23_applicable = per_layer["23"]["min_ratio_t"] < 1.0
l1822_not_applicable = all(per_layer[l]["min_ratio_t"] >= 1.0
                           for l in ("18", "22"))
if l23_applicable and l1822_not_applicable:
    verdicts["l23_architectural"] = True
else:
    verdicts["l23_architectural"] = False   # стабильная схема JSON
```

### 2. `outcomes` может не содержать `aggregate` (тихий пропуск исхода)

```python
if all_not_applicable:
    verdicts["aggregate"] = "non_abelian_not_applicable"
elif near_degen:
    verdicts["aggregate"] = "near_degeneracy"
    ...
if l23_well_sep and l1822_not:
    verdicts["l23_architectural"] = True
```

Если:
- `all_not_applicable` = False (какой-то слой < 1.0),
- `near_degen` пуст (L18/L22 оба ≥ 1.0, а < 1.0 только L6/L12/L19/L23),

то **ни одна ветка не выполняется**, и в `outcomes` нет `aggregate`. Это возможно: `near_degen` фильтрует только `informative` (L18, L22), а `all_not_applicable` смотрит на все шесть слоёв. Пример: L6 near-degenerate, L18/L22 well-separated → `all_not_applicable=False`, `near_degen=[]`, `aggregate` отсутствует.

Нужен fallback (например, `"inconclusive"` / `"mixed"`) и явная фиксация в ТЗ/мете.

### 3. Header противоречит коду и `meta` о метрике вердикта

Header:

> «The pre-registered verdicts below are driven by `min_t_gap_min_over_threshold` (direct comparison)…»

`meta`:

> «Verdicts use `min_ratio_t` (per-t threshold = eps*sigma_1^2 at that t); the global-scale ratio `min_t_gap_min_over_threshold` is reported for reference only»

Код использует `min_ratio_t`. Это прямая документальная несовместимость. Если ТЗ действительно требует **глобальный** порог (`min_t_gap_min >= threshold`, threshold = eps · median σ₁² по (t,doc)), то код неверен. Если ТЗ требует **per-t** порог (что утверждает комментарий со ссылкой на `docs/E38_fixes.md part-4 §2, HIGH`), то header устарел и вводит в заблуждение.

Нужно привести header, `meta` и код к одной формулировке и явно записать, какой именно порог pre-registered.

### 4. `ratio_t` — отношение медиан, а не медиана отношений (методологическое расхождение)

```python
gap_t_med = np.median(gap_min_td, axis=1)          # median_doc gap
s1sq_t_med = np.median(s1sq, axis=1)               # median_doc sigma_1^2
ratio_t = gap_t_med / (EPS_MAIN * s1sq_t_med)      # ratio-of-medians
```

Комментарий утверждает: «threshold = eps·σ₁² is local to each (t, doc)». Локальное сравнение — это `gap_min(t,d) >= eps·σ₁²(t,d)` **для каждого (t,d)**, затем агрегация по docs (медиана/доля). Текущий `ratio_t` сравнивает медиану `gap_min` с медианой `σ₁²` — это не то же самое, если `σ₁²` сильно варьируется по документам при данном `t`.

При этом `fraction_well_separated` **уже** использует per-doc локальное сравнение (`ws_td`), а вердикт — нет. Это внутренняя несогласованность. Надёжнее:

```python
per_doc_ratio = gap_min_td / (EPS_MAIN * s1sq)     # [S, N], NaN при s1sq=0
ratio_t = np.nanmedian(per_doc_ratio, axis=1)      # median_doc локального отношения
min_ratio_t = float(np.nanmin(ratio_t))
```

или, ещё точнее к формулировке «local to each (t, doc)»: вердикт по доле `(t,d)` с `gap_min < eps·σ₁²` (`1 - mean(ws_td[EPS_MAIN])`). Выбор нужно сверить с ТЗ и зафиксировать.

### 5. Нулевой `σ₁²` → «well_separated = True» (тихая ошибка на вырожденном J)

```python
thr_base = s_sorted[:, 0]            # sigma_1^2
ws = gap_min_vec >= e * thr_base
```

При `thr_base = 0` (все сингулярные значения нулевые) и `gap_min_vec = 0`: `0 >= e*0 = 0` → **True**. То есть полностью вырожденный J классифицируется как «well separated», хотя это максимальная дегенерация.

`spectral.eigengap_min` этот случай обрабатывает специально: `s[0] == 0 → (0.0, False)`. Векторизованная реализация с ней расходится.

Правильно:

```python
valid = thr_base > 0
ws = torch.zeros_like(gap_min_vec, dtype=torch.bool)
ws[valid] = gap_min_vec[valid] >= e * thr_base[valid]
```

Аналогично `ratio_t`: при `s1sq_t_med = 0` сейчас будет `0/0 = NaN`, а `NaN` молча выпадает из `< 1.0` и `>= 1.0` — слой не попадает ни в `near_degen`, ни в `all_not_applicable`. Нужен явный guard (`inf` / отдельный статус).

## P1 — устойчивость и соответствие ТЗ

### 6. `eigengap_min` импортирован, но не используется

```python
from prism.spectral import eigengap_min  # noqa: E402
```

Векторизация (`P2` в комментарии) оправдана, но тогда:
- либо добавьте assert/тест, что `gap_min_vec` совпадает с `eigengap_min(lam[i])` на случайных спектрах (включая нулевой и near-tie);
- либо удалите неиспользуемый импорт, чтобы не создавать иллюзию валидации.

Сейчас расхождение на нулевом спектре (п.5) как раз показывает, что «same math as per-doc eigengap_min» — не совсем правда.

### 7. `l23_architectural` — флаг, а не aggregate-метка

ТЗ перечисляет три pre-registered **outcome**: `non_abelian_not_applicable`, `near_degeneracy`, `l23_architectural`. Код кладёт первые две в `aggregate`, а третью — в отдельный булев ключ. Это несимметрично: `l23_architectural` не может быть «aggregate»-значением, и при сценарии «L23 applicable, L18/L22 not» в `aggregate` вообще ничего нет (п.2).

Лучше:

```python
if l23_applicable and l1822_not_applicable:
    verdicts["aggregate"] = "l23_architectural"
elif ...
```

и отдельно оставить per-layer флаги, если нужно.

### 8. `all_not_applicable` включает L23 и неинформативные слои

По ТЗ «on ALL layers» — формально верно. Но L23 тривиален (`J = W_lm_head`), а L6/L12/L19 не информативны. Если L6 near-degenerate, а L18/L22 — нет, `all_not_applicable=False`, `near_degen=[]` → нет aggregate (п.2). Стоит либо явно ограничить `all_not_applicable` информативными слоями (если ТЗ допускает), либо добавить fallback и записать это в `meta`.

### 9. Нет guard’а `R <= min(K, D)`

```python
sv16 = sv[:, :R]
```

При `K < R` (или `D < R`) срез молча вернёт меньше столбцов, `gaps` станет короче, `gap_min_vec` всё равно посчитается, но семантика «top-16» нарушится. Для текущего кэша `K=128, D=1024`, но добавьте:

```python
if min(Kk, Dd) < R:
    raise ValueError(...)
```

### 10. `json.dump(..., default=float)` — потенциально тихое `np.bool_ → 1.0`

В `exp38a` для этого сделан `_json_default`, который **бросает** на `np.bool_`. Здесь `default=float` молча конвертирует `np.bool_` в `1.0`/`0.0`. Сейчас `l23_architectural` — Python `True`, `informative` — Python `bool`, так что баг не срабатывает, но защита слабее, чем в соседнем эксперименте. Лучше переиспользовать `_json_default`.

### 11. Деление на ноль в `ref_threshold` / `min_t_gap_min_over_threshold`

```python
ref_threshold = EPS_MAIN * float(np.median(s1sq))
...
"min_t_gap_min_over_threshold": float(gap_t_med.min() / ref_threshold)
```

При `median(s1sq) = 0` — `inf`/`NaN`. Для текущего кэша (min ‖h‖ = 5.85) не срабатывает, но добавьте guard и пометку в `meta`.

### 12. `np.median` без nan-robust в `gap_t_med`, `s1sq_t_med`

Если J содержит NaN (битый кэш), `np.median` вернёт NaN для соответствующего `t`; далее `ratio_t` → NaN, вердикты молча выпадут. `spectral.eigengap_min` явно проверяет finite. Стоит либо использовать `np.nanmedian`, либо явно проверять `np.isfinite(J).all()` в начале слоя.

## P2 — мелочи

- `open(OUT, "w")` без `with` — файл не закрывается явно (как в `exp38a`).
- `del J` после `per_layer[str(l)] = ...` — можно освобождать раньше (после цикла `for t`), пик памяти ~2.1 ГБ на слой.
- `sv` остаётся от последней итерации — безвредно, но при рефакторинге легко получить `NameError`, если `t=0` перестанет попадать в ветку.
- `s1sq_profile_t` (128 значений) и `profile_gap_min_t` (32 значения, `[::4]`) — разная длина; в `meta` это не отмечено, читатель JSON может споткнуться.
- `fraction_well_separated` — ключи float; после `json.dump`/`load` станут строками `"0.005"`, `"0.01"`, `"0.02"`. Для воспроизводимости лучше сразу писать строковые ключи или список пар.
- `l23_architectural` отсутствует в `outcomes`, если условие ложно; для стабильной схемы JSON лучше всегда писать `False`.

## Соответствие ТЗ

| Pre-registered outcome | Реализация | Статус |
|---|---|---|
| `non_abelian_not_applicable` — `min_t_gap_min >= threshold` на всех слоях | `all_not_applicable` по `min_ratio_t >= 1.0` на всех 6 слоях | ⚠️ зависит от того, что ТЗ называет `threshold` (per-t vs global); header и код расходятся (п.3) |
| `near_degeneracy` — `< threshold` на L18 или L22 | `near_degen` по `min_ratio_t < 1.0` для `informative` | ✅ по смыслу, но `ratio_t` — отношение медиан, а не локальное сравнение (п.4) |
| `l23_architectural` — L23 applicable, L18/L22 not | инвертировано (п.1) | ❌ P0 |
| `fraction_well_separated = median_t(mean_doc ws)` | реализовано ровно так | ✅ |
| `min_t_gap_min = min_t(median_doc gap_min)` | реализовано | ✅ |
| «Layers are NOT averaged» | per-layer | ✅ |
| L23 тривиален, информативны L18/L22 | `informative` = L18/L22 | ✅ |

## Итоговый приоритетный список

1. **P0** — исправить инверсию `l23_architectural` (п.1).
2. **P0** — добавить fallback `aggregate` (п.2) и решить, является ли `l23_architectural` aggregate-меткой (п.7).
3. **P0** — согласовать header/`meta`/код по метрике вердикта (`min_ratio_t` vs `min_t_gap_min_over_threshold`) (п.3).
4. **P0** — защита от `σ₁² = 0`: `ws=False`, `ratio=inf/NaN` с явной обработкой (п.5).
5. **P1** — заменить `ratio-of-medians` на `median-of-per-doc-ratios` (или на долю нарушений `ws`), если ТЗ действительно требует локального сравнения (п.4).
6. **P1** — guard `R <= min(K,D)`, guard `ref_threshold > 0`, nan-robust медианы (п.9–12).
7. **P2** — `_json_default` вместо `default=float`, `with open`, стабильные ключи, `l23_architectural=False` всегда (п.10, мелочи).

После исправлений п.1–4 перезапустить — только тогда `outcomes` будут соответствовать ТЗ. Векторизованная реализация `gap_min` корректна для положительного спектра, но расходится с `spectral.eigengap_min` на нулевом — это стоит либо покрыть тестом, либо явно обработать (п.5–6).

# Ревью `exp38d_holonomy_spectrum.py`

## P0 — баги, искажающие вердикты

### 1. Метрика `spread` для вердикта abelian/non-abelian сломана (критично)

```python
theta_flat = theta_all[~degenerate]
spread = float(np.nanmax(theta_flat) - np.nanmin(theta_flat))
...
v.append("effectively_abelian" if sp < 0.1
         else ("non_abelian_candidate" if sp > 0.5 else "mixed_spectrum"))
```

`holonomy_spectrum` по контракту **выбрасывает тривиальные +1-оси** (см. docstring функции: «+1-axes dropped»). Значит `theta_flat` содержит только нетривиальные углы. Тогда:

- `H = I` (все +1) → 8 нулей → spread = 0 → «effectively_abelian» ✓
- **`H = diag(e^{iθ}, e^{−iθ}, 1, …, 1)`, θ = 0.3** → в `theta_flat` один угол `[0.3]` → spread = 0 → «effectively_abelian» ✗ (при этом κ = 2√2·sin(0.15) ≈ 0.42, что далеко от нуля)
- **8 одинаковых пар по θ = 0.5** → 8 углов 0.5 → spread = 0 → «effectively_abelian» ✗ (κ ≈ 1.98)

То есть `spread` не измеряет **величину** вращения, только разброс между нетривиальными углами. Все реальные слои (E-28: κ = 0.018…3.96, «rigid-but-curved») будут лейблиться как «effectively_abelian» вне зависимости от того, что реально происходит.

Заменить на магнитуду, например:
```python
max_theta = float(np.nanmax(theta_flat)) if theta_flat.size else 0.0
v.append("effectively_abelian" if max_theta < 0.1
         else ("non_abelian_candidate" if max_theta > 0.5 else "mixed_spectrum"))
```
или прямо на `kappa_median` (тоже magnitude, уже отчитан, семантически совпадает с «H ≈ I = abelian»).

### 2. `np.nanmax` на пустом `theta_flat` — латентный крэш

```python
spread = float(np.nanmax(theta_flat) - np.nanmin(theta_flat))
```

Если все триплеты вырождены (на текущем кэше — нет, но контракт допускает), `theta_flat` — пустой массив, `np.nanmax(empty)` → `ValueError: zero-size array to reduction operation fmax`. Громко, но не graceful. Нужен guard:
```python
if theta_flat.size:
    spread = float(np.nanmax(theta_flat) - np.nanmin(theta_flat))
else:
    spread = float("nan")
```
Плюс отдельная запись в meta при `theta_flat.size == 0`.

### 3. `_json_default` — тихий catch-all `str(o)` (P0 по стилю E-38a)

```python
def _json_default(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    return str(o)
```

- `np.float32` **не является** подклассом Python `float` → fall-through в `str(o)`; для `np.float32('nan')` это даёт `"nan"` — по счастливой случайности совпадает с нужным форматом. Но это именно совпадение, а не корректность.
- `np.bool_` → `str(o)` = `"True"`/`"False"` (строка) — **тихая конверсия**. Именно то, от чего защищался `_json_default` в E-38a.
- Любой неизвестный тип → `str(o)` — тихая утечка.

Сейчас в `out` всё приведено через `float(...)`/`int(...)`, кроме `list(np.percentile(...))` (возвращает `np.float64` — subclas of float, проходит). Так что сработает, но гарантия слабая. Скопировать `_json_default` из E-38a (бросать на `np.bool_`, преобразовывать `np.floating`/`np.integer` явно).

## P1 — расхождения с ТЗ / документацией

### 4. Header vs код: tolerance для L23 consistency

Header (docstring, строка про J = RAW):
> «same convention as E-28, which gave κ = 0.018 on L23 — consistency check, tol 1e-3»

`meta.note`:
> «1e-3 is tighter than the rounding of the cited 0.018»

Код:
```python
"within_tol_5e-3": abs_diff < 5e-3,
"within_tol_1e-3": abs_diff < 1e-3}
```
Вердикт использует **5e-3**. То есть header устарел и противоречит вердикту. Либо обновить header на 5e-3 (с указанием, что 1e-3 — диагностика), либо явно в докстринге зафиксировать, что pre-registered tolerance = 5e-3.

### 5. Pre-registered метки не перечислены явно

Docstring называет pre-registered gate (E-38c → near_degeneracy), но **не** перечисляет pre-registered исходы E-38d. При этом код присваивает:
- `consistency_pass` / `consistency_redflag`;
- `effectively_abelian` / `non_abelian_candidate` / `mixed_spectrum`;
- `spherical_geometry` / `third_mechanism` / `partial`.

Из них `non_abelian_candidate` и `partial` — по духу post-hoc расширения (нет пороговой формулы в docstring). Согласно прецеденту E-38a («`mixed` / `edges_atypical_candidate` — post-hoc extensions; новый лейбл нельзя вводить молча»), эти метки нужно либо явно объявить pre-registered (с формулой и порогом), либо записать в `meta` как post-hoc.

### 6. `n_anom` — смешение семантик

```python
for w in wlist:
    if "trivial" not in str(w.message):
        n_anom += 1
```

Docstring: «counts (expected/got angle mismatch, ±π boundary split)». Фактически `holonomy_spectrum` кидает ещё:
- «lone -1 eigenvalue (reflection axis)» — **может быть законным** (без det-fix Procrustes действительно даёт отражения);
- «degenerate spectrum (repeated angles)» — амбивалентно.

Оба не являются «±π boundary split», но инкрементируют `n_anom`. Если это диагностика — пометить в meta, что в счётчик входят также reflection и degenerate. Если важно различать — разбить на счётчики по типам.

### 7. `kappa_ci` и `trivial_frac` включают вырожденные триплеты

- `kappa_ci = np.percentile(kappa_all, [2.5, 97.5])` — на вырожденных `res ≈ 2r` R плохо обусловлен, `kappa` может быть произвольным. CI получается «шумным».
- `trivial_frac = float(trivial.mean())` — тривиальность на вырожденных триплетах почти никогда не сработает (R почти ортогонален → H не ≈ I), так что `trivial_frac` **занижен** относительно «доли тривиальных среди валидных».

Рекомендация: считать оба на `~degenerate`, а валовое значение оставить как diagnostics.

### 8. Порог `m.sum() >= 8` в `rho_at` не задокументирован

```python
if m.sum() >= 8:
    r_, p_ = spearmanr(...)
```

`n_docs = 32`; при 25% выживших `rho` по 8 точкам — очень шумно. Порог произвольный. Либо явно в `meta`, либо поднять (например, `>= 16`).

## P2 — мелочи / устойчивость

### 9. `np.random.seed(SEED)` не используется

`spermanr` детерминирован, permutation не выполняется. Seed декоративный. Docstring это честно фиксирует («seed is irrelevant for it»), но вызов `np.random.seed` только путает. Удалить или оставить с комментарием `# noqa`.

### 10. `trivial` threshold 1e-9 rad

Порог `th.max() < 1e-9` — произволен. Для `H = I` в float64 нули должны быть ~1e-15. Для `κ = 0.018` (E-28) углы ~ 0.006 — не тривиальны. ОК для текущего кэша, но при более шумном SVD (float32) порог 1e-9 может ложно ловить нетривиальные H как тривиальные. Не bug, но стоит зафиксировать в meta.

### 11. `theta_all[~degenerate]` — 3D vs 2D маска

`theta_all` — 3D `[n_triples, N, R//2]`, `~degenerate` — 2D `[n_triples, N]`. Numpy делает правильный выбор (первые две оси), но это неявное поведение — стоит комментарий (`# 2D mask on 3D array → selects (t,i) rows`).

### 12. `H_012_doc0` — 124 × 16 × 16 float в JSON

~31k float. Это ~1 МБ на слой (только L23). Заявлено «for future κ_fiber recomputation». Хранить лучше в отдельном `.npz`, чтобы не раздувать `exp38d_holonomy_spectrum.json` (парсинг и так небыстрый).

### 13. `float(np.median(kappa_all))` в отчёте

`kappa_median` — плоская медиана по всем `(t, i)` включая вырожденные. Для отчёта в консоли — ок, для JSON — уже `kappa_ci` (тоже с дегенератами). Согласовать: либо везде включать (и в meta), либо везде исключать.

### 14. `n_positions` vs `n_docs`

`rho_at` возвращает `len(rs)` = число позиций `t` (шаг 4) с `m.sum() >= 8`. В JSON ключ `n_positions` — верно. Но `n_docs` нигде не отчитан (внутри `m` всегда ≤ 32). Не bug, но для диагностики полезно.

### 15. `degen_sweep` — `frac` не целое, ключи float

`f"{frac}x2r"` даёт `"0.7x2r"`, `"0.9x2r"`, `"0.95x2r"`. В JSON ключи — строки. ОК, но для машинной обработки лучше `{"frac": 0.7, "degenerate_frac": ..., "rho_median": ...}`.

### 16. `if len(th) > R // 2: raise`

Корректно (громко), но комментарий «P1-7: loud, not a shape crash mid-layer» — единственный «P1-7» в файле, отсылки к `docs/E38_fixes.md part-4 §4` нет. Либо добавить ссылку, либо убрать псевдо-идентификатор.

### 17. `Htot.clone()` — `.detach()` не нужен, но `.cpu()` не помешал бы

`Htot` — CPU-тензор (R_total в holonomy_spectrum строится на `torch.eye(..., dtype=torch.float64)` по умолчанию на CPU). Но если когда-нибудь `holonomy_spectrum` начнёт возвращать тензор на GPU (сейчас нет), `.tolist()` упадёт. Добавить `.cpu()`:
```python
H012_save.append(Htot.detach().cpu().clone())
```

## Соответствие ТЗ / paper

| Утверждение | Реализация | Статус |
|---|---|---|
| «triples with any step_res > 0.9·2r → EXCLUDED from ρ» | `degenerate` + `rho_at` фильтр | ✅ |
| «trivial H → valid Abelian result, recorded» | `trivial` + `trivial_frac` | ✅ |
| «‖H−I‖² = 4Σ(1−cos θ_j)» | через `eigvals(H)` (все r собственных значений) | ✅ (корректно, с учётом `holonomy_spectrum` контракта) |
| «identity_check_max_absdiff» | есть | ✅ |
| «J = RAW (E-28 convention)» | `ck["J"]` без gauge-free | ✅ |
| «per-TZ degenerate step near 2r» | `step_res > 0.9*2r` | ✅ |
| «θ: NaN padding, mixed spectrum < 8» | `theta_all = nan` + `:len(th)` | ✅ |
| «Spearman p — asymptotic, not permutation» | `meta` + `p_med` | ✅ |
| «E-28 κ(L23) = 0.018, tol 1e-3» | header 1e-3, код 5e-3 | ❌ header устарел |
| pre-registered outcome list | не перечислен явно | ⚠️ |

## Итоговый приоритетный список

1. **P0** — заменить `spread` для вердикта abelian/non-abelian на `max_theta` (или `kappa_median`). Иначе вердикт не отражает вращения.
2. **P0** — guard для `theta_flat.size == 0` в `np.nanmax/nanmin`.
3. **P0** — переписать `_json_default` по образцу E-38a (бросать на `np.bool_`; явно обрабатывать `np.floating`/`np.integer`).
4. **P1** — согласовать tolerance (1e-3 vs 5e-3) в header и коде.
5. **P1** — явно перечислить pre-registered метки (или пометить post-hoc: `non_abelian_candidate`, `partial`).
6. **P1** — исключить `degenerate` из `kappa_ci` и `trivial_frac` (или отчитаться явно).
7. **P1** — разобрать `n_anom` по типам warning.
8. **P2** — `with open(...)`, `.detach().cpu().clone()` для `H012_save`, `if m.sum() >= 8` в meta, `H_012_doc0` в отдельный `.npz`.

После исправления п.1 вердикты будут отражать реальную величину вращения, а не только разброс по нетривиальным модам. Всё остальное — сохраняет уже сделанные правильные решения (identity через все r собственных значений, NaN padding, degenerate exclusion из ρ, документированное различие asymptotic p vs permutation).

# Ревью `exp38e_MK_diagnostic.py`

## Контекст: что делает скрипт

1. Загружает `Qwen3.5-0.8B` (CPU, fp32), строит `M_K = W_lm[token_set] · diag(1+w)` (из tie-weight через lm_head и финального RMSNorm).
2. Вычисляет `w = M_K^T · 1/√K` — [1024] вектор.
3. По каждой (doc, t) раскладывает `w = α_t û_t + β_t v_⊥`: `α = ⟨w, û⟩`, `β = ‖w − α û‖`.
4. Сравнивает:
    - `ratio_median = median |α|/|β|` → gate;
    - `cons_ratio = median(β)/σ_1(M_K)` ≈ 0.9;
    - `jtu_ratio = median(β/‖J^T·1/√K‖)` ≈ 1/c = r_t;
    - `j_uhat_resid = median ‖J·û‖/‖J‖` ≈ 0 (факторизация).
5. Вердикты: `gate_failed` / `mixed` / `gauge_blind_sep`; `consistency_consistent_with_E33r` / `consistency_redflag`.

Все 5 pre-registered меток реализованы. Логика в целом соответствует ТЗ, но есть существенные баги и пара концептуальных оговорок.

## P0 — баги, влияющие на вывод (или падающие)

### 1. `np.median` / `.mean()` на NaN-содержащих `alpha_abs` / `beta_abs`

```python
w2 = torch.where(valid, w2, torch.tensor(float("nan")))
beta_vec = torch.where(valid, beta_vec, torch.tensor(float("nan")))
...
alpha_abs = w2.abs().numpy()
beta_abs  = beta_vec.numpy()
...
"alpha": {"median_abs": float(np.median(alpha_abs)),
          "mean_abs":   float(alpha_abs.mean())},
"beta":  {"median":     float(np.median(beta_abs)),
          "mean":       float(beta_abs.mean())},
```

`alpha_abs` / `beta_abs` содержат `NaN` в позициях с `‖h‖ < 1e-6`. `np.median` и `.mean()` **не nan-robust**: при наличии хотя бы одного NaN результат — NaN. На L23 (min ‖h‖ = 5.85) в текущем кэше `nan_frac = 0`, поэтому сейчас не проявляется — но это латентный баг: если хоть один (doc, t) вырожден, весь блок `alpha`/`beta` уходит в JSON как `null`/NaN. Заменить на `np.nanmedian` / `np.nanmean`.

### 2. `pos0_ratio` на потенциально пустом срезе

```python
pos0_ratio = float(ratio[:, 0][pos0_valid].median())
```

Если для всех документов `‖h_0‖ < 1e-6` (сильный sink), `pos0_valid` — все `False`, `.median()` на пустом тензоре в новых версиях torch бросает `RuntimeError: median(): Expected reduction dim 0 to have non-zero size`, в старых — тихо даёт `NaN`, дальше JSON получает NaN. Нужен guard:

```python
pos0_vals = ratio[:, 0][pos0_valid]
pos0_ratio = float(pos0_vals.median()) if pos0_vals.numel() else float("nan")
```

### 3. Мёртвая проверка `ck23.get("V")` — «HIGH-3» guard не работает

```python
if ck23.get("V") is not None:
    assert ck23["V"] == V_model, ...
```

`_e38_cache.load_layer` возвращает `{"J", "P", "H", "U128"}` — **ключа `V` нет и в исходном `ck`**. Проверка всегда `False`, никогда не исполняется. Комментарий обещает «HIGH-3: the cache's U128 must index the SAME model's vocab» — а на деле остаётся только `assert 0 <= min(token_set) and max(token_set) < V_model`, которая проверяет диапазон, но **не** соответствие вокабуляра. Silent failure: если `U128` из другого чекпоинта с тем же `V` (пере-токенизация / другая модель того же размера), проверка пройдёт, а `M_K` будет собрана по чужим строкам.

Варианты: либо убрать мёртвый код (и пере-определить «HIGH-3» через сравнение с реальным токенайзером модели), либо добавить `V` в `_e38_cache.load_layer` при следующем обновлении кэша.

### 4. `default=float` не защищает JSON от NaN/Inf

```python
json.dump(out, open(OUT, "w"), indent=1, default=float)
```

`default=float` вызывается только для типов, которые `json` не умеет сам. `float('nan')` и `float('inf')` **являются** Python float и сериализуются как `NaN` / `Infinity` (невалидный RFC-8259 JSON). В exp38a и exp38d это уже починено через `_json_default` (бросает на `np.bool_`, превращает `inf`/`nan` в строки). Здесь стоит переиспользовать тот же `_json_default` — иначе любая комбинация (вырожденный pos0, `nan_frac > 0`) даёт битый JSON молча.

## P1 — математика и интерпретация

### 5. `cons_ratio ≈ 0.9` — тихая предпосылка

```python
cons_ratio = float(np.nanmedian(beta_vec.numpy()) / sigma_g)  # = median β / σ_1(M_K)
cons = ("consistency_consistent_with_E33r"
        if abs(cons_ratio - 0.9) <= TOL_CONSISTENCY else "consistency_redflag")
```

Цепочка: `β = ‖J^T u‖/c` (из [L-gauge-J]); проверка `β/σ_1(M_K) ≈ 0.9` требует

$$
\frac{\beta}{\sigma_1(M_K)} = \frac{\|J^T u\|}{\sigma_g(J)} \cdot \frac{\sigma_g(J)}{c\,\sigma_1(M_K)} \approx \cos\cdot\frac{\sigma_g(J)}{c\,\sigma_1(M_K)}
$$

Чтобы это ≈ `cos`, нужно `σ_g(J) ≈ c·σ_1(M_K)`, т.е. **`v_1(M_K) ⊥ û`** (топ-направление `M_K` выживает после проекции). Если `v_1(M_K)` значимо лежит вдоль `û`, `σ_g(J) << c·σ_1(M_K)` и `cons_ratio << 0.9` → `consistency_redflag` сработает, хотя `cos` может быть 0.9. Т.е. `redflag` ≠ «модель плохая», а «проверка не сработала из-за выравнивания `v_1(M_K)` с `û`».

Docstring описывает это как «the correct local ratio», но не помечает, что проверка **односторонне консервативна** (нижняя граница для cos). Стоит в `meta` явно записать: `consistency_redflag` может означать не модель, а `v_1(M_K) ∦ û`. Сейчас это `meta.consistency_semantics` частично описывает (говорит про `≤1`), но не про ложные срабатывания при `v_1(M_K) ∥ û`.

### 6. `jtu_ratio ≈ 4` — ожидание завышено и вводит в заблуждение

```python
"note": "β_t / ‖J_t^T 1/√K‖ = r_t/√D (NOT 1) под exact "
        "J = M_K(I − ûû^T): ... A value ~4 CONFIRMS the exact factorization "
        "(r_t ≈ 4·√D)"
```

Строго: `β/‖J^T u‖ = 1/c = r_t = ‖h‖/√D`. «≈ 4» означает `‖h‖ ≈ 4·√D ≈ 128` — это **эмпирическое ожидание конкретного масштаба**, а не математическое следствие. Для типичного скрытого состояния `‖h‖ ≈ 50–100` реалистичное значение `jtu_ratio ≈ 1.5–3`, и это **тоже** подтверждает факторизацию. Код не проверяет `≈ 4` — просто отчитывает значение. Но комментарий читателя собьёт: если при запуске `jtu_ratio ≈ 2`, вывод «≈ 4 CONFIRMS» будет прочитан как «не подтверждено», хотя это не так.

Заменить формулировку: «ожидание `jtu_ratio ≈ ‖h‖/√D`, значение определяется масштабом скрытых состояний; факторизация подтверждается, если `jtu_ratio` согласуется с `median(‖h‖)/√D` — проверьте по `s1sq_profile_t` / нормам, `~1–4` типично». Либо добавить в JSON поле `r_t_expected = median(‖h‖)/√D`, чтобы сравнение было явным, а не косвенным.

### 7. `sigma_g` — это `σ_1(M_K)`, не `σ_g(J)`

Имя `sigma_g` в paper обозначает gauge-сингулярное значение `J`. Здесь это `σ_1(M_K)`. Формулы работают (докстринг всё оговаривает), но имя путает: тот, кто читает JSON, ожидает `σ_g(J)`, а получает `σ_1(M_K)`. Переименовать в `sigma_1_MK` (или в `sigma_top_MK`) — тогда `consistency_sigma_g` тоже переименовать и явно связать через `σ_g(J) ≈ c·σ_1(M_K)` в комментарии.

### 8. `nan_frac = 1.0` → все медианы NaN молча

Если по какой-то причине `valid` пуст целиком (массовая дегенерация), `np.nanmedian` вернёт NaN с `RuntimeWarning` от numpy — не падает. `cons_ratio`, `ratio_median`, `jtu_ratio`, `j_uhat_resid_median` — все NaN. `np.isfinite` не проверяется. `cons = redflag` (потому что `abs(nan - 0.9) > 0.15` — False, но и не True… стоп, `nan > 0.15` в Python — False; значит `abs(nan - 0.9) <= 0.15` — False; `else` → redflag). Худший случай — громко: `gate_failed` или `mixed` по NaN (сравнение `nan > 10` — False → упадёт в `mixed`/`gauge_blind_sep`). Стоит добавить guard: `if nan_frac > 0.5: verdict = "degenerate"`.

## P2 — мелочи / надёжность

### 9. `torch.manual_seed(SEED)` в main не используется

В теле `exp38e` нет случайных операций: загрузка весов детерминирована, `M_K`, `w`, `α`, `β`, `cons_ratio`, `ratio` — детерминированные. Seed декоративен. Не баг, но его вызов создаёт ложное ощущение «воспроизводимости через seed» — в реальности `determinism` гарантирован отсутствием randomness, а не seed.

### 10. `pos0_ratio` может быть огромным (не NaN)

`ratio = |α|/|β|clamp_min(1e-12)`. Если в каком-то (doc, 0) `β ≈ 0` (w почти параллелен û), `ratio ≈ |α|/1e-12` — `1e9` или больше. `median` по 32 докам может дать `1e9`. `float(np.float32(1e9))` — валидный JSON, но значение бессмысленное. Стоит клиповать `pos0_ratio` или отчитывать с диагностическим флагом (`pos0_ratio_capped: true`).

### 11. `torch.ones(128)` / `np.sqrt(128.0)` — хардкод `K`

```python
w = M_K.T @ (torch.ones(128) / torch.sqrt(torch.tensor(128.0)))
...
JtU = torch.einsum("ntkd,k->ntd", J23, torch.ones(128) / np.sqrt(128.0))
```

`assert len(token_set) == 128` гарантирует правильность при текущем кэше, но при смене `_e38_cache.K` (другой token_set) упадёт с невнятной ошибкой. Заменить на `K = len(token_set)` / `Kt = torch.ones(K) / np.sqrt(K)`.

### 12. `M_K`, `H`, `J23` не освобождаются явно

`del ck23` освобождает dict, но `H` и `J23` продолжают держать ~2.15 GB до выхода из функции. Для одного слоя не критично; для будущего расширения на L18/L22 превратится в утечку. `del M_K` тоже не сделан (~0.5 MB, но зачем держать). Мелочь.

### 13. `attn_implementation="eager"` — не нужно для этой задачи

`exp38e` не делает forward — все величины берутся из cache (H) или из весов (lm_head, norm). `attn_implementation` влияет только на forward, который здесь не запускается. Не баг, но вводит в заблуждение: выглядит как «нужна совместимость с Jacobian-проходом», а её нет. Можно удалить (или оставить с комментарием «не используется, для консистентности с E-28»).

### 14. `torch.manual_seed(SEED)` до `from transformers import AutoModelForCausalLM`

Модуль `transformers` при импорте может инициализировать RNG (крайне маловероятно, но). Неважно, потому что никакой forward не запускается; но сама позиция «seed до импорта» слабая идиома.

### 15. `ck23["U128"]` — list, но `build_M_K` ожидает `Sequence[int]`

`token_set = ck23["U128"]` — `list`. `build_M_K(W_lm_full, token_set, g)` — `Sequence[int]`, ок. `torch.tensor(token_set, ...)` внутри `build_M_K` — тоже ок. Всё работает; просто убедитесь, что `U128` — `list[int]`, а не `torch.Tensor` (иначе `torch.tensor(tensor_of_long)` даст тревожное предупреждение). Из `_e38_cache` — это `list(ck["U128"])`, так что safe.

## Соответствие ТЗ

| Pre-registered outcome | Реализация | Статус |
|---|---|---|
| `gate_failed` — `median \|α\|/\|β\| > 10` | `ratio_median > 10` | ✅ |
| `mixed` — `≈ 1` (0.1–10) | `0.1 ≤ ratio_median ≤ 10` | ✅ |
| `gauge_blind_sep` — `< 0.1` | `ratio_median < 0.1` | ✅ |
| `consistency_consistent_with_E33r` — `\|β_t\|/σ_1(M_K)` в 15% от 0.9 | `abs(cons_ratio - 0.9) <= 0.15` | ✅ (см. п.5 — односторонне консервативно) |
| `consistency_redflag` — иначе | `else` | ✅ |
| Проверка `σ_1(M_K)` | `svdvals(M_K)[0]` | ✅ |
| Единственный слой L23 | `per_layer = {"L23": ...}` | ✅ |
| Легенда о «старой проверке» | поля `over_sigma_g_OLD`, `over_sigma_g_OLD_note` | ✅ |
| Проверка `‖J·û‖/‖J‖` | `j_uhat_resid_median` | ✅ |

Проблемы ТЗ-соответствия — только интерпретационные (п.5, п.6): pre-registered метки применяются буквально, но их семантика (что именно означает «consistency_redflag») описана неполно.

## Приоритетный список исправлений

1. **P0** — `np.nanmedian` / `np.nanmean` для `alpha_abs` / `beta_abs` (п.1).
2. **P0** — guard на пустой `pos0_vals` (п.2).
3. **P0** — удалить/заменить мёртвый `ck23.get("V")` (п.3).
4. **P0** — `_json_default` из exp38a/exp38d вместо `default=float` (п.4).
5. **P1** — задокументировать в `meta`, что `consistency_redflag` возможен при `v_1(M_K) ∥ û` (п.5).
6. **P1** — исправить комментарий про «jtu_ratio ≈ 4» на формулу `r_t = ‖h‖/√D`, `~1–4` в зависимости от масштаба (п.6).
7. **P1** — переименовать `sigma_g` → `sigma_1_MK` (п.7).
8. **P1** — guard `nan_frac > 0.5` → отдельный verdict `degenerate` (п.8).
9. **P2** — `K = len(token_set)` вместо хардкода 128 (п.11); клип/флаг для `pos0_ratio` (п.10); `del M_K, H, J23` перед выходом (п.12).

`jstruct.build_M_K`, `jacobian._lm_module`, `interventions._resolve_lm_head_weight` — используются корректно (`g = norm.weight + 1` для Qwen3.5 RMSNorm с `output = (x/rms)·(1+w)`, `M_K = W_lm[ts]·diag(g)` — точное соответствие формуле `[M-1]`). После исправления P0-пунктов скрипт будет давать корректный JSON; интерпретация (P1) требует уточнения формулировок, но не меняет значений.

# Ревью `exp38f_structural_consistency.py` (+ JSON)

Теперь есть код — и он **подтверждает** математическую проблему, которую я поднял в ревью JSON. Более того, я нашёл ещё пару багов.

## P0 — математика: правило вердикта не покрывает `g ≠ 1`

### 1. `Δcos ∈ [0, 1−c_g]` требует, чтобы `g = 1` (топ-мода была модой с максимальной alignment)

Вывод в `meta.note`/docstring:
> «per [L-gauge-J], cos(v_k, v_pred) = σ_k c_k / ‖J^T u‖, so under exact factorization Δcos ∈ [0, 1 − c_g]»

Формула `cos(v_k, v_pred) = σ_k c_k / ‖J^T u‖` — **верна** (из `J^T u = V Σ U^T u`).

Но из неё `Δcos = c_g (σ_g / ‖J^T u‖ − 1) ≥ 0` следует **только если** `σ_g ≥ ‖J^T u‖`. А `‖J^T u‖² = Σ σ_k² c_k² ≤ σ_1²`. Значит `σ_g ≥ ‖J^T u‖` требует `σ_g = σ_1` (и `σ_k c_k = 0` для `k ≠ g`).

Код же выбирает `g = argmax_k |c_k|` **без проверки** `g == 1`:

```python
c = (U.T @ one).abs()
k = int(c.argmax())            # ← может быть ≠ 0
```

Если мода с максимальным `|c_k|` — **не** топ-мода, `σ_g < σ_1`, и `Δcos` может быть **отрицательным**, хотя `J` невырожден и лемма `J^T u = c β v_⊥` выполнена точно.

**Контрпример.** `J = diag(1, 0.5, …)`, `u = (0.6, 0.8, …)`:

- `c_1 = 0.6`, `c_2 = 0.8` ⇒ `g = 2`, `c_g = 0.8`, `σ_g = 0.5`.
- `‖J^T u‖ = √(1·0.36 + 0.25·0.64) ≈ 0.722`.
- `cos(v_g, v_pred) = 0.5·0.8 / 0.722 ≈ 0.554`, `Δcos ≈ −0.246`.

Правило `Δcos < −0.05 → lemma_broken` даст **ложный** `lemma_broken` на валидной конфигурации. (Мягкий контрпример `diag(1, 0.9)` с `u = (0.6, 0.8)` даёт `Δcos ≈ −0.032`, попадает в зону `[−0.05, 0)` и проскальзывает, но это просто везение допуска.)

**Следствие.** В текущем прогоне `Δcos = 0.103 > 0` — совместимо с тем, что `g = 1` типичен, но нигде не проверено. Правило нужно либо

1. **ограничить режимом `g = 1`** (или `σ_g ≥ ‖J^T u‖`): `g≠1` — отдельный исход (`g_not_top_mode`) / исключение из статистики; либо
2. **заменить на per-instance проверку** `Δcos ∈ [0, 1−c_g + ε]` для каждого `(doc, t)` — тогда нарушение для конкретной позиции и будет `lemma_broken`; либо
3. **прямо проверить лемму**: `‖J^T u − c β v_⊥‖ / (c β) < tol` (тождество, не эмпирическое неравенство). Если цель — проверить именно лемму, это единственный корректный тест; проверка `cos(v_g, v_pred)` — про **эмпирическое** `v_⊥ ∥ v_g`, не про лемму.

### 2. Verdict названия вводят в заблуждение

`verdict = "lemma_confirmed"` при `Δcos ∈ [0, 1−c_g+ε]` означает «эмпирическое выравнивание `v_⊥ ≈ v_g` совместимо с леммой», а не «лемма подтверждена». Лемма либо выполняется тождественно (для точной факторизации `J = c M_K (I−ûû^T)`), либо нет — её нельзя «подтвердить» эмпирическим `cos`. Переименовать: `empirical_alignment_within_lemma_range` или аналог.

### 3. Верхняя граница `1 − c_g + ε` — тривиальна

`Δcos ≤ 1 − c_g + 0.05` эквивалентно `cos(v_g, v_pred) ≤ 1 + 0.05`. То есть «confirmed» ловит всё, что не попало в «broken». Всё нетривиальное содержание — в нижней границе `Δcos ≥ 0`, которая ложна вне `g = 1` (п.1).

## P0 — код

### 4. `default=float` в `json.dump` — тихий невалидный JSON

```python
json.dump(out, open(OUT, "w"), indent=1, default=float)
```

При `verdict = "insufficient_data_nan"` (например, `valid.size < 2` → `dc_ci = [nan, nan]`, `dc_med = nan`) `json.dump` запишет литерал `NaN` — невалидный RFC-8259 JSON. `default=float` вызывается только для типов, которые `json` не умеет, а Python `float('nan')` — умеет (и пишет как `NaN`).

В `exp38a` и `exp38d` уже сделан `_json_default`, который превращает `inf`/`nan` в строки. Здесь тот же баг, что был в `exp38e`, и он воспроизводится **именно в ветке `insufficient_data_nan`**. Переиспользовать `_json_default`.

### 5. Гейт молча проходит при отсутствии `GATE_FILE`

```python
gate_ok = True
if os.path.exists(GATE_FILE):
    g = json.load(open(GATE_FILE))
    gate_ok = not g["outcomes"].get("gate_failed", False)
```

Если `exp38e_MK_diagnostic.json` не сгенерирован, `gate_ok = True` — E-38f работает без гейта. ТЗ: «Gate: run only if E-38e did not report gate_failed». Отсутствие E-38e — это **не** «E-38e прошёл». Нужно `gate_ok = False` при отсутствии (с предупреждением) или явно в `meta.gate_file_missing = True`.

## P1 — интерпретация

### 6. `mean` и `median` — разные единицы

```python
"dcos": {"median": dc_med, "ci": dc_ci,
         "ci_note": "doc-level: per-doc median, bootstrap over 32 docs",
         "mean": float(np.nanmean(dc)), ...}
```

`dc_med` — медиана от per-doc медиан (32 значения). `mean` — `nanmean(dc)` по всем `N × S = 4096` значениям. Разные единицы, но `ci_note` документирует только `median`/`ci`. Ввести `mean_note` или заменить `mean` на `mean_per_doc` (медиана по докам от per-doc средних).

### 7. Глобальная `upper = 1 − cu_med + eps` не ловит per-t аномалии

`cu_med` — медиана `c_g` по всем `(doc, t)`. Верхняя граница `1 − cu_med + eps` — это медианная граница. Если для отдельных `t` `c_g(t) ≈ 0.99`, их персональный `Δcos` может быть, скажем, `0.04` (что `≤ 1−0.99+0.05 = 0.06` — OK), но также может быть `0.10` (`> 0.06` — методологическая аномалия для этой позиции). Поскольку вердикт считается по `dc_med` (медиана), персональные аномалии усредняются.

Добавить в `metrics`:
```python
"frac_positions_above_upper": float((dc > upper_per_t).mean())
```
где `upper_per_t = 1 − c_g_per_t + eps` — поэлементно. Это даст более чувствительный индикатор, чем глобальный медианный.

### 8. Документированный допуск `[-0.05, 0)` ничем не обоснован

В `meta.note`: «Δcos < −0.05 = broken (impossible exactly)». Если `Δcos < 0` «невозможно точно», то **любое** `Δcos < 0` (а не `< −0.05`) должно давать broken. Допуск `0.05` нужен только для численного шума (float32), но код делает SVD в float32 (`torch.linalg.svd` на float32) без приведения к float64 — шум порядка `1e-7`, не `5e-2`. Значит, допуск `0.05` в 5·10⁵ раз больше численного шума, то есть он не про численность, а про «мы не уверены в теореме». Это — замаскированный post-hoc (как `edges_atypical_candidate` в E-38a).

Явно записать в `meta`, что `0.05` — **статистический допуск** (не численный), и что при `Δcos ∈ [−0.05, 0)` вердикт фактически подтверждён только эмпирически, а не по лемме.

## P2 — мелочи

- **`K = 128` в начале файла не используется.** Код использует `Kk = J.shape[2]`. Удалить.
- **`S` (импортированный из `_e38_cache`) не используется.** Используется `Ss = J.shape[1]`. Удалить.
- **`vp = None` + `cv = float("nan")`** — счётчик `nan_count` включает эти позиции. `np.nanmedian` их исключает. ОК, но стоит отчитывать `n_vpred_degenerate` отдельно (сколько раз `v_pred_from_JT` поднял `RuntimeError`), это диагностически полезно.
- **`cos_vg_vp` и `cos_ug_one`** — плоские списки `N·S`; медианы считаются `np.nanmedian`. `cos_vg_vp` содержит NaN (для дегенеративных), `np.nanmedian` справляется. `cos_ug_one` NaN не содержит (`c[k]` — из SVD, конечен). ОК.
- **`for i in range(N): ... if i % 8 == 0: profile.append(...)`** — `profile` имеет длину 4 (доки 0, 8, 16, 24), а `profile_docs = list(range(0, N, 8))` — тоже 4. ОК.
- **`dc = np.array(dcos).reshape(N, Ss)`** — если `Ss = 128`, `N = 32`, то `dcos` длина `4096`, `reshape(32, 128)` — OK.
- **`v_pred_from_JT(Jt, one)`** — `one` передаётся как `uniform`, но `v_pred_from_JT` нормализует внутри (и `one` уже единичный). Двойная нормализация — no-op, но неявная. Мелочь.
- **`scipy.stats.spearmanr`** — не используется (в отличие от exp38d). OK.

## Соответствие ТЗ / pre-registered

Pre-registered исходы явно не перечислены (в коде — `verdict ∈ {insufficient_data_nan, lemma_broken, lemma_confirmed, methodological_anomaly}`, что близко к `meta.note`).

| Исход | Реализация | Статус |
|---|---|---|
| `insufficient_data_nan` | `not has_data` | ✅ |
| `lemma_broken` — `Δcos < −0.05` | `dc_med < −eps` | ⚠️ премиса неверна вне `g=1` (п.1) |
| `lemma_confirmed` — `Δcos ≤ 1−c_g+0.05` | `dc_med ≤ upper` | ⚠️ тривиально для верхней границы (п.3) |
| `methodological_anomaly` — иначе | `else` | ✅ |

Формально вердикты соответствуют заявленному правилу; правило математически неполно.

## Приоритетный список

1. **P0** — добавить проверку `g == 1` (или `σ_g ≥ ‖J^T u‖`) перед применением правила; иначе — отдельный исход `g_not_top_mode` или исключение из статистики.
2. **P0** — `_json_default` из exp38a/exp38d вместо `default=float` (воспроизводит `NaN` в ветке `insufficient_data_nan`).
3. **P0** — `gate_ok = False` при отсутствии `GATE_FILE`.
4. **P1** — переименовать `verdict = "lemma_confirmed"` → `empirical_alignment_in_range` (или аналог), чтобы не путать с проверкой самой леммы.
5. **P1** — добавить `frac_positions_above_upper` (per-t проверка).
6. **P1** — `mean` и `median` в разных единицах: пометить в `ci_note` или заменить `mean`.
7. **P1** — задокументировать, что `0.05` — статистический допуск, а не численный.
8. **P2** — удалить неиспользуемые `K`, `S`; добавить `n_vpred_degenerate`.

# Ревью `exp38g_jlens_link.py`

## P0 — `frac_tail_our` (и `S_our`) включают 896 null-мод; вторичная метрика `rho_our` не несёт информации

### 1. `frac_tail_our` — не «bottom 10% of 128 σ», а «12 real + 896 null»

Docstring (SECONDARY (a)):
> «frac_tail_our(t) = energy in our tail modes (bottom 10% of the 128 σ)»

Код:

```python
tail_o = int((1 - TAIL_FRAC) * Kk)         # int(0.9*128) = 115
V_our = Vh_o.T.contiguous()                # [D, Kk] = [1024, 128]
V_null = null_space(V_our.T.numpy())       # [D, D-Kk] = [1024, 896]
V_full = torch.cat([V_our, V_null], 1)     # [D, D] = [1024, 1024]
c2_o = (H[i, t] @ V_full) ** 2             # [1024]
S_o[t], Ft_o[t] = entropy_frac_tail(c2_o, tail_o)
```

Внутри `entropy_frac_tail(c2, tail_idx)`:
```python
ft = c2[..., tail_idx + 1:].sum(dim=-1) / tot.squeeze(-1)
```

`c2[116:]` = индексы `116..1023` = **12 реальных tail-мод** (`116..127`) **+ 896 null-мод** (`128..1023`). То есть `frac_tail_our` считается по 908 модам, а не по 12.

**Магнитуда.** Для типичного `h ∈ R^1024` с примерно равномерным распределением энергии по ортонормальному базису `(H·v_k)² ≈ ‖h‖²/1024`, сумма по 896 null-модам ≈ `0.875·‖h‖²`. Значит:

- `frac_tail_our ≈ 0.87 + (вклад 12 реальных tail)` — почти константа от `t`;
- `S_our ≈ log(1024) ≈ 6.93` — тоже почти константа;
- `rho_our = spearmanr(S_our, frac_tail_our)` → 0 или `nan` (нет вариации).

То есть **вторичная метрика, включая её `tz_preregistered_outcome`, неинформативна**: null-моды доминируют и «размывают» сигнал.

### 2. `rho_our_alt` (MEDIUM-5 sensitivity check) тоже сломан — тем же путём

```python
G = torch.randn(D - Kk, D, ...)             # [896, 1024]
Q_alt, _ = torch.linalg.qr(G.T)             # [1024, 896]
V_alt_full = torch.cat([V_our, Q_alt], 1)   # [1024, 1024]
c2_alt = (H[i, t] @ V_alt_full) ** 2
S_alt_t, _ = entropy_frac_tail(c2_alt, tail_o)
```

`S_alt_t` тоже считается по 1024 модам, включая 896 «заполнителей». Поскольку обе версии (`S_our` и `S_alt`) одинаково null-доминированы, `rho_our_alt ≈ rho_our ≈ 0` независимо от того, какой completion выбран. `completion_sensitive = False` — ложно-отрицательный результат: check «не чувствителен к completion» пройден, но обе версии одинаково бессмысленны.

### 3. `tz_preregistered_outcome` систематически смещён

```python
tz_preregistered_outcome = (
    "jlens_not_tail_modes" if rho_our_med < 0.3 and rho_lens_med < 0.3
    else ("symmetric" if min(rho_our_med, rho_lens_med) > 0.7 else "partial"))
```

Так как `rho_our_med ≈ 0` (или `nan`) практически всегда, условие сводится к:
- `jlens_not_tail_modes` ⇔ `rho_lens_med < 0.3` (только lens);
- `symmetric` ⇔ `rho_lens_med > 0.7` (только lens, поскольку `min(0, rho_lens)` = 0);
- `partial` — всё остальное.

То есть **весь «pre-registered» исход определяется только `rho_lens_med`**, хотя ТЗ требовало «both rho high» / «both low». `rho_our` в решении не участвует.

### 4. Исправление — минимум

Считать `frac_tail_our` и `S_our` по реальным модам, отдельно от null:
```python
# только 128 реальных мод V_our
c2_real = (H[i, t] @ V_our) ** 2        # [128]
S_our = entropy(c2_real)
frac_tail_our = c2_real[tail_o+1:].sum() / c2_real.sum()
```
и `rho_our` сравнивать с `rho_lens` только если обе метрики определены на «своих» реальных модах. В `meta` явно записать, что метрики определены на разных пространствах (128 vs 1024 моды) — они семантически сравнимы («спарсити в спектральном базисе»), но не по абсолютной величине.

Для PRIMARY (overlap) — код **корректен** (`V_our_tail = V_our[:, tail_o+1:]`, только 12 реальных мод, nulls не участвуют). P0 касается только SECONDARY.

## P1 — pre-registered label `tails_disjoint` не реализован

Docstring:

> ```
> overlap_median > 0.9 → our_tail_within_lens_tail
> overlap_median < 0.3 → tails_disjoint
> else                  → partial_overlap
> ```

Код:
```python
if ov_med > OVERLAP_HIGH: verdict = "our_tail_within_lens_tail"
elif ov_med < 0.5 * random_baseline: verdict = "tails_anticorrelated"
elif ov_med < 3 * random_baseline: verdict = "at_random"
else: verdict = "partial_overlap"
```

- `tails_disjoint` **не встречается** в коде — pre-registered label заменён на `tails_anticorrelated` + `at_random`.
- `partial_overlap` теперь занимает диапазон `[3·baseline, 0.9]`, а не `[0.3, 0.9]`.

`MEDIUM-3` документирует, что `overlap ≈ 0.1` — «random independence, not disjointness», и что порог `0.3` нужно переинтерпретировать. Это разумно, но по прецеденту E-38a (`mixed`/`edges_atypical_candidate` — «новый label нельзя вводить молча»), замена pre-registered label должна быть явно в `meta` зафиксирована как **post-hoc rename**, а не только как математическое уточнение.

## P1 — `V_full.shape` не проверяется

```python
V_null = torch.from_numpy(null_space(V_our.T.numpy())).float()
V_full = torch.cat([V_our, V_null], dim=1)
```

`scipy.linalg.null_space(V_our.T)` при неточном float32-SVD может дать ранг 127 вместо 128 → `V_null` 897 столбцов → `V_full` 1025 столбцов → `c2_o` длиной 1025. Далее `c2_o[116:].sum()` тихо посчитает другой набор мод; `S_o` — энтропию по 1025 значениям. Никакой assert это не поймает.

Добавить:
```python
if V_full.shape != (D, D):
    raise RuntimeError(f"V_full {V_full.shape} != ({D},{D}); null_space degenerate")
```
Аналогично для `V_alt_full` (хотя там shape предсказуем — QR даёт ровно 896 столбцов; но assert не повредит).

## P1 — загрузка `lens["J"]`: разная семантика для dict / tensor / list

```python
if hasattr(lens["J"], "keys"):
    assert T_LAYER in lens["J"], ...
    J_l = lens["J"][T_LAYER].float()
else:
    assert len(lens["J"]) > T_LAYER, ...
    J_l = lens["J"][T_LAYER].float()
```

- если `lens["J"]` — dict с ключами `"22"` (строка) вместо `22`, `T_LAYER in lens["J"]` = False → AssertionError с ключами в сообщении — громко.
- если `lens["J"]` — `np.ndarray` формы `[24, D, D]`, `hasattr(..., "keys")` = False → `lens["J"][T_LAYER]` даст `np.ndarray [D,D]` → `.float()` падает с `AttributeError: 'numpy.ndarray' object has no attribute 'float'` — громко, но невнятно.

Не P0, но стоит нормализовать: `torch.as_tensor(lens["J"][T_LAYER], dtype=torch.float32)`.

## P1 — `lens_valid` gate проверяет только `s1 < 4` и `‖J−I‖_F < 3`, но не дегенерацию спектра

```python
lens_valid = bool(jmin_i < 3.0 and s1 < 4.0)
```

ТЗ (docstring): «validity gate: J_22 spectrum bounded and moderately close to identity (σ1 < 4, ‖J−I‖_F < 3)». Реализовано ровно так. Но `MEDIUM-6` в `meta.note` честно оговаривает, что gate не ловит дегенерацию (`V_lens_tail` неоднозначен внутри вырожденного подпространства). Соответствие ТЗ — да; ограничение — документировано. Это не баг, а долг.

## P1 — `entropy_frac_tail`: `keepdim=True` и `assert c2.dim() == 1` противоречат

Docstring:
> «keepdim=True makes the 1-D and batched [B, n_modes] cases behave identically»

Код:
```python
assert c2.dim() == 1, f"entropy_frac_tail expects 1-D c2, got {c2.dim()}D"
```

Assert запрещает batched, хотя docstring обещает batched. Одно из двух устарело. В `exp38g` вызывается только 1-D — тест не поймает.

## P2 — `T_LAYER = 22` несовместим с остальными E-38 (L23) и с paper §4.5

Все E-38a–e — про L23 (последний слой, аналитический `J = W_lm`). E-38g использует L22. Paper §4.5 (E-37):

> «L22 — провал (GDN)»

То есть L22 — самый проблемный слой для интерпретации (GDN, нестабильные метрики). Docstring не объясняет выбор L22: «priority layer per TZ». Если ТЗ действительно говорит L22 — ок; но стоит явно зафиксировать в `meta`, что:
- L22 — GDN, position-averaging (в лизне) может терять recurrent-структуру;
- paper §4.5 уже помечает L22 как аномальный;
- остальные E-38 сделаны на L23, поэтому «symmetry with Anthropic» здесь сравнивает несравнимые слои.

## P2 — `np.random.seed(SEED)` не используется

Все стохастические операции — через `bootstrap_ci(seed=SEED)` (свой `default_rng`) и `torch.Generator().manual_seed(...)` для random QR. `np.random.seed` — no-op. Как в exp38e/f.

## P2 — `s1_ratio` — мёртвая диагностика

```python
s1_ratio = s1 / max(float(sigma_l.median()), 1e-30)
```

Записывается в JSON, но не используется в `lens_valid` (там только `s1` и `jmin_i`). Либо убрать, либо добавить в `lens_valid` (например, `s1_ratio < 20` как sanity-check на «spectrum not catastrophically flat»).

## P2 — `rho_our_alt_med` при всех `nan` даёт `nan` без предупреждения

`np.nanmedian(all_nan) → RuntimeWarning + nan`. Затем `completion_sensitive = np.isfinite(nan) and ...` = False. Тихая обработка; хорошо, что не падает, но стоит логировать.

## Сводка по PRIMARY vs SECONDARY

| Компонент | Статус |
|---|---|
| `overlap_t`, `V_our_tail`, random_baseline | ✅ корректно |
| `verdict` (overlap-based) | ✅ корректно (с точностью до rename pre-registered label) |
| `V_lens`, `V_lens_tail`, `rho_lens`, `frac_tail_lens`, `S_lens` | ✅ корректно |
| `frac_tail_our`, `S_our`, `rho_our`, `rho_our_alt` | ❌ null-доминированы (P0) |
| `tz_preregistered_outcome` | ❌ систематически смещён (P0, следствие) |
| `completion_sensitive` | ❌ ложно-отрицательный (P0, следствие) |

## Соответствие ТЗ

| Утверждение | Реализация | Статус |
|---|---|---|
| «PRIMARY (b): direct subspace check» | `overlap_t` через `V_our_tail` и `V_lens_tail` | ✅ |
| «`overlap = ‖V_our_tail^T V_lens_tail‖²_F / r_our`» | ровно так | ✅ |
| «random baseline = `r_lens/D`» | ровно так | ✅ |
| «SECONDARY (a): dual sparsity/tail correlation» | реализовано, но null-доминировано | ❌ P0 |
| `overlap > 0.9 → our_tail_within_lens_tail` | ✅ | ✅ |
| `overlap < 0.3 → tails_disjoint` | заменено на `at_random`/`tails_anticorrelated` | ⚠️ P1 (post-hoc rename, не отмечен) |
| `else → partial_overlap` | ✅ (но диапазон шире из-за rename выше) | ⚠️ |
| `+ lens_validity_suspect` при провале gate | ✅ | ✅ |
| Lens fit на Hindi, docs English | `meta.note` фиксирует | ✅ |
| Deterministic seed=0 | ✅ (SVD LAPACK; bootstrap seeded) | ✅ |

## Приоритетный список

1. **P0** — `frac_tail_our` и `S_our`: считать по 128 реальным модам `V_our`, отдельно от null; либо `c2_o[:Kk]` вместо `c2_o` для обеих метрик. Пересчитать `rho_our`, `rho_our_alt`, `tz_preregistered_outcome`.
2. **P0** — после фикса (1) перепроверить `rho_our_alt` / `completion_sensitive`: сейчас ложно-отрицательный.
3. **P1** — зафиксировать в `meta` post-hoc замену `tails_disjoint` на `at_random`/`tails_anticorrelated` (по прецеденту E-38a).
4. **P1** — assert `V_full.shape == (D, D)` (и `V_alt_full` тоже).
5. **P1** — `entropy_frac_tail`: убрать assert или docstring про batched.
6. **P1** — нормализовать загрузку `lens["J"]` (`torch.as_tensor(..., dtype=float32)`).
7. **P2** — `meta` про L22 (GDN + paper §4.5 «провал»), про несовместимость с L23-центричным E-38.
8. **P2** — убрать `np.random.seed(SEED)`, `s1_ratio` (или использовать), предупреждение при all-nan `rho_our_alt`.

PRIMARY-метрика (overlap) в текущем виде корректна и даёт осмысленную проверку ТЗ; SECONDARY — нужно исправить (P0). Verdict, посчитанный по overlap, не затронут P0, но `tz_preregistered_outcome` сейчас смещён и должен быть перевычислен после фикса.
