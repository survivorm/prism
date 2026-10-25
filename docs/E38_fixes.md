# Ревью исправленных `exp38a_uhat_dynamics.py` + `_e38_cache.py` + `holonomy.py`

## Что проверено (верификация фиксов)

| Ранее найденный баг | Фикс | Проверка |
|---|---|---|
| P0 `aggregate` возвращал ключ слоя | `best_layer = max(...); aggregate = verdicts[best_layer]` | ✓ |
| P0 `edges_atypical` недостижим | новая ветка `elif h and not head_above` | ✓ (но см. п.4 ниже) |
| P0 `edges_special` без `(>>)` | `if h and t and head_above and tail_above` | ✓ |
| P0 `edges_atypical_candidate` — dead code | удалён из docstring и кода | ✓ |
| P0 guard для κ_g пропускал `t+1` | `norms[i, t-1:t+2]` | ✓ |
| P1 `median_s_head_excl_pos0` off-by-one | `s[:, 1:HEAD_END-1]` = `s[:, 1:15]` | ✓ |
| P1 `l18_vs_l19` KeyError | `if "18" in per_layer and "19" in per_layer` | ✓ |
| P2 `json.dump(open(OUT))` без `with` | `with open(OUT, "w")` | ✓ |
| P2 `_json_default` не защищал от np-скаляров | теперь raise для unknown, "nan"/"inf" для NaN/inf floats | ⚠️ см. п.1 |
| P2 двойная загрузка J в E-38a | новый `load_layer_H(layer)` | ✓ |
| P2 `load_layer` без выборочных полей | `load_layer(layer, want=(...))` | ✓ |

Математика `geodesic_curvature` не менялась — по-прежнему корректна. `_e38_cache` рефакторинг корректный. `holonomy.py` без изменений (это хорошо — стабильный API).

## P1 — новый баг (фактически был и раньше, но сейчас документирован как фикс)

### 1. `_json_default` **не** предотвращает `NaN` в JSON — документированный guard не работает

Docstring в коде:
> «NaN in a per-doc segment (an all-NaN segment → nanmedian NaN) must not emit the non-standard NaN literal into the JSON (valid RFC-8259; same convention as exp38d's _json_default)»

Код:
```python
def _json_default(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    raise TypeError(...)

json.dump(out, f, indent=1, default=_json_default)
```

**Проблема:** `json.dump`'s `default=` вызывается **только** для объектов, которые `json` не умеет сериализовать. `float('nan')` — сериализуем (как литерал `NaN`), поэтому `default` для него **не вызывается**:

```python
>>> json.dumps(float('nan'), default=lambda x: "CUSTOM")
'NaN'   # default не вызван
```

То есть при наличии NaN в `out` получится `NaN` в JSON — тот самый невалидный литерал, от которого guard должен защищать. Guard сработает только для `np.bool_`/`np.float32` (не подкласс `float`) — для них `default` вызывается, потому что encoder сам не умеет.

**Почему это P1, а не P2:** docstring явно обещает защиту от NaN, а фактически её нет. На текущем кэше (`min ‖h‖ = 5.85`, `kappa_nan_count = 0`) не проявляется, но на любом кэше с одним нулевым `‖h‖` в сегменте JSON станет невалидным **молча**, вопреки документированному контракту.

**Фикс:** пре-санитизация перед `json.dump`:
```python
def _sanitize(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o

json.dump(_sanitize(out), f, indent=1, allow_nan=False)
```
`allow_nan=False` дополнительно делает так, что **если** какой-то NaN проскочит санитайзер, вылетит `ValueError` — громко. Либо использовать `prism.cache._json_dump_value` / образец из `exp38d` (там `default=_json_default` работает аналогично — а значит тот же баг есть и в exp38d; см. следующий пункт).

### 1b. Тот же баг в `exp38d_holonomy_spectrum.py`

`exp38d` использует **тот же** `_json_default`:
```python
def _json_default(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    return str(o)
json.dump(out, open(OUT, "w"), indent=1, default=_json_default)
```

По той же причине NaN/floats из `out` (а в exp38d `theta` содержит NaN-padding → `nanmedian` по theta вернёт float) сериализуются как `NaN` literal. `default` для `float('nan')` не вызывается. `str(o)` — вообще опасный fallback (тихо конвертирует `np.bool_` в `"True"`).

Стоит исправить одновременно (и, если экспортируется `_json_default`, то в общем модуле — например, `prism.json_utils`).

## P2 — новые или неполные фиксы

### 2. `edges_atypical` — направление берётся только из `s`

```python
head_above = row["s"]["head"]["median"] > row["s"]["body"]["median"]
...
elif h and not head_above:
    v = "edges_atypical"
```

`h = h_s or h_k` — но `head_above` использует **только** `s`. Если `h_s=False`, `h_k=True` (аномалия только в κ), и при этом `s.head > s.body`, получим `mixed`, а не `edges_atypical`, хотя κ-anomalous head может быть именно «ниже body» (kappa — мера поворота, «straighter head» может быть семантически «<< body»).

Это не баг в строгом смысле (ТЗ не специфицирует, в каком канале брать направление), но стоит зафиксировать в `meta`:
> «`edges_atypical` direction is taken from `s` only; if the anomaly is kappa-driven and s-direction disagrees, the label falls to `mixed`»

Иначе читатель JSON может интерпретировать `mixed` как «оба направления неясны», а не как «направление s и κ расходятся».

### 3. `seg_stats`: all-NaN body → `anomaly_ratio = inf` → ложно-аномальный

```python
ratio = abs(med - med_b) / iqr_b if iqr_b > 0 else (
    float("inf") if med != med_b else 0.0)
```

Если `body` полностью NaN: `iqr_b = nan` (`np.nanpercentile` возвращает NaN для all-NaN с `RuntimeWarning`); `iqr_b > 0` → `nan > 0` — **False**. Уходим в `else`: `med != med_b` — если `med` тоже NaN, `NaN != NaN` → `True` → `inf`. `anomalous = inf > 2.0` → `True`.

То есть полностью деградировавший сегмент классифицируется как **аномальный**, хотя правильная семантика — «неопределённый» или «не аномальный». На текущем кэше не проявляется (`kappa_nan_count = 0`).

Рекомендация:
```python
if np.isnan(med) or np.isnan(med_b) or np.isnan(iqr_b):
    ratio = float("nan")
    anomalous = False
elif iqr_b > 0:
    ratio = abs(med - med_b) / iqr_b
    anomalous = ratio > T_ANOM
else:
    ratio = float("inf") if med != med_b else 0.0
    anomalous = ratio > T_ANOM
```

Сейчас `anomalous = bool(ratio > T_ANOM)` — а `bool(nan > 2)` = `False`, так что `anomalous=False`. Значит вердикт неверный только в части `ratio=inf` при `med`, `med_b` NaN (это возможно).

### 4. `kappa_nan_count` не участвует в вердикте

Если весь κ-канал NaN (массовая дегенерация `‖h‖ < 1e-10` — например, при отравленном кэше или при padding в другом сценарии), `segs(kapp, ...)` — всё NaN → `seg_stats` → `median=nan`, `anomalous=False` (см. п.3). Вердикт для κ-канала — `edges_typical`. Слой, где κ не измерен, классифицируется как «типичный» — это **противоположность** содержательного вывода.

Фикс: добавить в `outcomes` поле `kappa_measured_fraction = 1 - kappa_nan_count/(N*(S-2))`, и в вердикт не доверять κ-каналу при `fraction < 0.5`. Либо в `meta` явно: «if kappa_nan_count > 0.1 · total, the κ channel is not informative».

### 5. `is_gdn` — hardcoded `(19, 22)` — **фактически лжёт сейчас**, а не «при изменении LAYERS» (P1, не P2)

```python
row["is_gdn"] = bool(l in (19, 22))
```

Сверено с `model.config.layer_types` (Qwen3.5-0.8B, 24 слоя, `full_attention`/`linear_attention`): GDN-множество = {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22}. Среди изучаемых слоёв: **GDN = L6, L12, L18, L22; full attention = L19, L23**. Хардкод `(19, 22)` инвертирован на 4 из 6 слоёв: L6/L12/L18 помечены как non-GDN при том, что это GDN, и L19 помечен как GDN при том, что это full attention. Выводы для опубликованного JSON:
- `is_gdn` в `exp38a_uhat_dynamics.json` ложен для L6/L12/L18/L19 (4/6 слоёв);
- блок `l18_vs_l19_gdn` (max_abs_profile_diff 0.056) — **не** «GDN vs full»: L18 и L19 — оба full attention; сравнение само по себе валидно (профильные отличия двух full-слоёв), но его **название** ложно.

Фикс: перенести GDN-множество в `_e38_cache.py` константой рядом с `LAYERS` (значение из `config.layer_types`, загрузка модели не нужна) и пересчитать `is_gdn` в JSON; числа `l18_vs_l19_gdn` не меняются — переименовать (убрать `_gdn`).

### 6. `"cache": "kaggle_kernel/results/exp27b_jcache_L{l}.pt"` — литерал `{l}`

Косметика, но выглядит как незаконченная f-string. Если читатель JSON сверяет путь — вводит в заблуждение.

### 7. `norms[:, 1:].median(dim=-1).values.clamp_min(1e-12)` — NaN не клампится

`pos0_ratio = norms[:, 0] / (...).clamp_min(1e-12)`. `torch.clamp_min` на NaN возвращает NaN — если `norms[:, 1:]` содержит NaN (H-кэш с NaN), `pos0_ratio` = NaN. `pos0_ratio.median()` = NaN. `float(NaN)` → NaN в JSON (см. п.1). Защита — `nan_to_num` или `nanmedian`. Не критично (H без NaN в текущем кэше), но закрывает цикл с п.1.

## Что **не** является проблемой

- `except (RuntimeError, ValueError)` вокруг `geodesic_curvature` — действительно defensive. Для `u = H / norms.clamp_min(1e-12)`: при `norms ≥ 1e-10` (что и есть при отсутствии skip’а) `u` unit с точностью fp32 ~1e-7, `geodesic_curvature` не выбросит. Для `norms < 1e-10` — мы skip. Clamp-сценарий не достижим. ✓
- `head/body/tail` индексы для `s` и `kappa`: `s[:, 0:16]` — 16 значений; `kappa[:, 1:16]` — 15 значений (позиция 0 не считается, т.к. `kapp[:, 0]` — padding нулями). Body: `s[:, 16:126]`, `kappa[:, 16:126]` — 110 значений. Tail: `s[:, 126:127]`, `kappa[:, 126:127]` — 1 значение. Позиции согласованы, перекрытий нет. ✓
- `profile_S` — `range(0, 127, 4)` = `[0, 4, ..., 124]`; `s[:, 124]` существует. ✓
- `profile_K` — `range(1, 128, 4)` = `[1, 5, ..., 125]`; `kapp[:, 125]` существует. ✓
- `aggregate` tie-breaking — L6 (индекс 0) выигрывает при равных counts, потому что `-list(verdicts).index(k)` даёт `-0` для L6 и `-1` для L12; `-0 > -1`, `max` выбирает L6. Соответствует документированному «first in LAYERS order». ✓
- `_e38_cache.load_layer(want=...)` — `bad_want` проверка, `P` normalization guard, `out["U128"]` всегда включается. ✓
- `_e38_cache.load_layer_H` — H-only, ~2.1 GB peak (вместо 4.2 GB). ✓
- `_e38_cache.jcache_path` — multi-fallback (local → Kaggle datasets → Kaggle slug). Если ничего не найдено, возвращает `local` (torch.load упадёт громко). ✓
- `holonomy.py` — без изменений. `geodesic_curvature`, `procrustes_align`, `holonomy_spectrum` — корректны. ✓

## Итоговый приоритетный список

**P1 — исправить до отчёта:**
1. `_json_default` не предотвращает NaN в JSON — нужна рекурсивная пре-санитизация + `allow_nan=False`. Тот же баг в `exp38d` (и, вероятно, везде, где используется этот паттерн).
5. `is_gdn = (19, 22)` фактически ложен на 4/6 слоёв (config: GDN = L6/L12/L18/L22, full = L19/L23) — ложные метаданные в опубликованном JSON; плюс ложное название `l18_vs_l19_gdn` (оба full attention).

**P2 — исправить перед публикацией JSON:**
2. `seg_stats` для all-NaN body: `anomaly_ratio = inf`, `anomalous = True` — ошибочная классификация. Добавить явный `if nan in {med, med_b, iqr_b}: anomalous=False`.
3. `edges_atypical` использует только s-направление, не κ — зафиксировать в `meta`.
4. `kappa_nan_count` не влияет на вердикт — добавить `kappa_measured_fraction` и не доверять κ-каналу при малой доле.
6. `"cache": "...L{l}.pt"` — литерал, не f-string.

**P3 — косметика / латентные:**
7. `pos0_ratio` с NaN в `norms[:, 1:]` — `clamp_min` не санитизирует NaN. `nan_to_num` или `torch.nanmedian`.
8. `_json_default` определён внутри `main()` — вынести в модуль для переиспользования (как в `cache.py`/`spectral.py`).

**Вывод по исправлениям:** все P0/P1 из прошлого ревью (aggregate, edges_atypical, edges_special direction, dead code, guard t+1, off-by-one, l18_vs_l19, with open) — **исправлены корректно**. Появились два P1: не работающий NaN-guard (п.1) и фактическая ошибка `is_gdn` (п.5, обнаружена при повторной сверке с `config.layer_types`). После исправления п.1 и п.2 файл можно запускать на реальном кэше — текущий `kappa_nan_count=0` и `min‖h‖=5.85` означают, что p.1/p.2 не искажают текущие числа, но контракт «NaN → valid JSON» нужно соблюсти до следующего прогона с другим кэшем. П.5 искажает **уже опубликованный** JSON (метаданные, не измерения).

# Ревью исправленных `cache.py` + `spectral.py` + `exp38b_position_audit.py`

## Что исправлено в exp38b (верификация)

| Ранее найденный баг | Фикс | Проверка |
|---|---|---|
| P0-1 `body_per_doc` группировка (смешение t/doc) | `torch.stack(body_cos_list, dim=0).nanmedian(dim=0)` | ✅ |
| P0-2 `align_K1` оси (sum по модам, max по компонентам) | `.sum(dim=1).abs().max(dim=-1)` | ✅ |
| P1-3 `body_over_pos0` — несогласованные агрегаты | `pos0.nanmedian()` vs `body_per_doc.nanmedian()` | ✅ |
| P1-4 guard на `paired_diff_ci is None` | `if pair_ci is not None` | ✅ |
| P1-5 `body_ci` — i.i.d. по 3552 коррелированным | переименован в `body_flat_iid_ci_do_not_use` + note | ✅ |
| P2-9 `del Jgf` после использования | `del Jgf` в конце итерации | ✅ |
| P2-7 `nanmedian` для per-doc | `.nanmedian(dim=0)` | ✅ |
| `load_layer(want=("J","P"))` — H не грузится | ✅ |

Арифметика `C_b`, `PR_modes`, `align_K1`, paired bootstrap — корректна. `cache.bootstrap_ci_paired` даёт CI для **разности медиан** на одних и тех же доках — правильная paired-схема.

## P0 — новый баг: `del Jraw, ck` → `NameError` (регрессия, введённая ПОСЛЕ публикации JSON)

```python
ck = load_layer(l, want=("J", "P"))
Jraw, P = ck["J"], ck["P"]          # [N, S, K, D], [N, S, K]
del ck                              # ← имя ck удалено
N = Jraw.shape[0]
Jgf = gauge_free_J(Jraw, P)         # [N, S, K, D]
del Jraw, ck                        # ← NameError: name 'ck' is not defined
```

`del` для уже удалённого имени бросает `NameError`. Текущий файл **упадёт на первом слое**, не записав JSON. Нужно просто `del Jraw` (или удалить строку, поскольку `Jraw` всё равно перезаписывается в следующей итерации).

**Хронология (проверено по git):** опубликованный `results/exp38b_position_audit.json` (10-23 23:01) создан из коды эпохи 204a2f6, где было единственное `del Jraw, ck` (валидно). Строка `del ck` добавлена коммитом 7f29482 (10-24 17:25) — с этого момента файл не запускается; аудит E38_fixes2 (d562f75, 10-24 21:07) этого не поймал. Иными словами: опубликованные числа валидны, текущий репо-код — сломан, пересчёт JSON в текущем виде невозможен без фикса.

## P1 — `_json_default` не предотвращает `NaN` literal (унаследовано)

Тот же баг, что был в `exp38a`/`exp38d` (отмечен в предыдущем ревью):

```python
def _json_default(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    raise TypeError(...)
json.dump(out, f, indent=1, default=_json_default)
```

`json.dump`'s `default=` вызывается **только** для типов, которые encoder не умеет сериализовать. `float('nan')` — умеет (пишет `NaN` literal), поэтому `default` для него **не вызывается**. Guard сработает только для `np.bool_`/`np.float32`, но не для `float('nan')`.

Источники NaN в `out`:
- `pos0_median = float(posv.nanmedian())` — если все pos0 NaN (сильный sink);
- `body_median = float(body_flat.nanmedian())` — если все body NaN;
- `body_per_doc_median = float(bpd_f.nanmedian())` — если все surviving docs NaN;
- `pos0_PR_vs_paper_ratio = ratio` — если `pos0_pr_med = NaN`;
- `body_over_pos0 = cb / max(c0, 1e-12)` — если `c0 = NaN` → `NaN`;
- `prof_cos/prof_pr/prof_al` — `float(cos_docs.nanmedian())` при all-NaN.

При текущем кэше: опубликованный JSON валиден (NaN-литералов нет — единственный `NaN` в файле — текст в строке `meta.note`; проверено), а путь NaN **реально проходил**: `nan_count(L23) = 1` — одна дегенерированная пара (doc, t) с `‖J_t‖ < 1e-9` дала NaN в C_b и была корректно исключена медианами (paired_n = 32 на всех слоях). Все per-doc медианы в том прогоне конечны, поэтому NaN в `out` не попал — но это свойство кэша, не кода: для произвольного кэша с all-NaN сегментом — молча невалидный JSON (см. также п.1 куска 1 про механизм: `default` для `float('nan')` не вызывается).

**Фикс — пре-санитизация + `allow_nan=False`:**

```python
def _sanitize(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict):  return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):  return [_sanitize(v) for v in o]
    return o

json.dump(_sanitize(out), f, indent=1, allow_nan=False)
```

`allow_nan=False` дополнительно делает так, что **если** NaN проскочит, вылетит `ValueError` — громко. Тот же фикс нужен в `exp38a`/`exp38d`.

## P2 — не критично, но стоит исправить перед публикацией

### 1. `nan_count` не участвует в вердикте

Если целый слой дегенерирован (`nan_count` ≈ N·S), `PR_modes` = 0.0 для всех доков (по конвенции `participation_ratio`), `body_over_pos0` может дать произвольное значение. Вердикт `no_artifact`/`edge_contamination` будет построен на шуме. (На текущем кэше `nan_count` = 0..1, т.е. режим «целый слой» не достигнут.) Стоит:
- либо добавить `frac_nan = nan_count / (N * S)` и при `frac_nan > 0.5` — отдельный вердикт `degenerate_layer`;
- либо явно в `meta`: «nan_count is diagnostic only; verdicts assume nan_count = 0 (as in the current cache)».

### 2. `pos0_PR_vs_paper_ratio` может быть NaN

```python
ratio = pos0_pr_med / PR_MODES_L22_PAPER
if abs(ratio - 1) > 0.2:
    warnings_.append(...)
```

Если `pos0_pr_med = NaN`, `ratio = NaN`; `abs(NaN - 1) > 0.2` → `False` (NaN comparisons — False), так что warning молчит. Но `ratio` попадёт в JSON как `NaN` literal (см. P1). Добавить guard: `if not np.isfinite(ratio): continue` или `if np.isfinite(ratio) and abs(ratio - 1) > 0.2`.

### 3. `body_over_pos0` при `c0 = NaN`

`cb / max(c0, 1e-12)` — `max(NaN, 1e-12)` в Python возвращает `NaN` (max с NaN возвращает первый аргумент, если сравнение не сработало — на самом деле `max(NaN, 1e-12)` в CPython: `NaN > 1e-12` = False, поэтому `max` возвращает `1e-12`... стоп, `max(a, b)` = `a if a > b else b`. `NaN > 1e-12` → False, значит `max` вернёт `1e-12`. Значит `cb / 1e-12` = большое число или NaN. `cb` может быть NaN (если body все NaN). Тогда `NaN / 1e-12 = NaN` → JSON literal. Латентно).

Рекомендация: `body_over_pos0 = cb / c0 if (np.isfinite(c0) and c0 > 0) else float("nan")`.

### 4. `params.cache = "exp27b_jcache_L{l}.pt"` — литерал, не f-string

Косметика; читатель JSON может принять за шаблон.

### 5. `del Jgf` — только после записи, но `Jraw` освобождён раньше

После фикса P0 (убрать `del Jraw, ck`): `Jraw` держится до переопределения в следующей итерации. Если память важна (2.1 GB на слой), добавить `del Jraw` после `gauge_free_J`.

### 6. `sv` персистится через итерации

`sv` определяется внутри `if t == 0 or ...`. В конце не удаляется. Не проблема (перезаписывается), но при будущем рефакторинге — ловушка.

## `cache.py` — без изменений, корректно

- `bootstrap_ci`: `if not np.all(np.isfinite(arr))` → ValueError на NaN. В `exp38b` передаются отфильтрованные `posv_f`/`body_f`. ✅
- `bootstrap_ci_paired`: `np.median(a[idx], axis=1) - np.median(b[idx], axis=1)` — правильная paired-схема (общие индексы). ✅
- `_json_dump_value` для `MetricCache` — строка `"nan"` для NaN. Но `MetricCache` в `exp38b` не используется. ✅
- `iter_position_pairs` — не используется в `exp38b`, но корректен. ✅

## `spectral.py` — без изменений, корректно

- `participation_ratio`: `(Σλ)² / Σλ²`, `s2 == 0 → 0.0`. Для дегенерированного J (все σ=0) возвращает 0.0 — это документированное поведение, соответствует `exp38b` (`"PR on a degenerate J returns 0.0"`). ✅
- `top_r_svd`, `eigengap_min`, `snr_e14_parts` — не используются в `exp38b`, но корректны. `snr_e14_parts` явно валидирован против `exp14_snr_pr.json`. ✅
- `_check_1d_finite` — корректный guard против NaN в спектре.

## Соответствие ТЗ (exp38b)

| Исход | Реализация | Статус |
|---|---|---|
| `no_artifact` — `C_b(body) ≤ 1.5·C_b(pos0)` ∧ (L22: paired CI ∋ 0) ∧ ¬(align body > 0.9 ∧ pos0 < 0.7) | `reasons = []`, если пусто → `no_artifact` | ✅ |
| `edge_contamination` — `C_b(body) > 1.5·C_b(pos0)` | `body_over_pos0 > 1.5` | ✅ |
| `l22_position_sensitive` — L22 paired CI excludes 0 | `pair_ci` guard + `not (dlo <= 0 <= dhi)` | ✅ |
| `align_edge_effect` — body > 0.9 ∧ pos0 < 0.7 | `body_median > 0.9 and pos0_median < 0.7` | ✅ |
| `body = [16, S-2]` (126 INCLUDED) | `HEAD_END <= t <= BODY_END` | ✅ |
| `pos0` — baseline, не в body | `t == 0` отдельно | ✅ |
| CI: per-doc bootstrap / paired bootstrap | `bootstrap_ci` / `bootstrap_ci_paired` | ✅ |
| Degenerate ‖J_t‖ < 1e-9 → NaN в C_b | `deg = nrm < 1e-9`, NaN через `torch.where` | ✅ |
| PR degenerate → 0.0 | `participation_ratio` | ✅ |

## Итоговый приоритетный список

**P0 — скрипт упадёт:**
1. `del Jraw, ck` — `ck` уже удалён строкой выше → `NameError`. Заменить на `del Jraw`.

**P1 — исправить до отчёта:**
2. `_json_default` не работает для `float('nan')` — нужна рекурсивная пре-санитизация + `allow_nan=False`. Тот же баг в `exp38a`/`exp38d`.

**P2 — перед публикацией JSON:**
3. `pos0_PR_vs_paper_ratio` при `pos0_pr_med = NaN` — `ratio = NaN`, warning молчит, NaN в JSON. Guard `if np.isfinite(ratio) and abs(ratio - 1) > 0.2`.
4. `body_over_pos0` при `c0 = NaN` → `NaN`. Явный guard: `if np.isfinite(c0) and c0 > 0`.
5. `nan_count` не участвует в вердикте — добавить `frac_nan` и/или явно в `meta`.
6. `params.cache` — литерал `{l}`, не f-string.
7. `del Jraw` после `gauge_free_J` (освободить 2.1 GB до следующей итерации).

**cache.py / spectral.py:** в контексте `exp38b` — проблем нет, изменений не требуется. Единственное замечание — `cache.py` содержит правильный `_json_dump_value` (строки для NaN) — его можно вынести в общий `prism.json_utils` и переиспользовать в `exp38a`/`exp38b`/`exp38d` вместо локальных `_json_default`, которые не работают.

После P0-фикса (п.1) `exp38b` запустится; после P1 (п.2) — будет давать валидный JSON при любом кэше.

# Ревью исправленного `exp38c_degeneracy_preflight.py`

## Верификация фиксов из прошлого ревью

| Баг | Фикс | Проверка |
|---|---|---|
| **P0-1** `l23_architectural` инвертирован | `l23_applicable = min_ratio_t < 1.0`, `l1822_not_applicable = all(>= 1.0)`, `l23_arch = bool(l23_applicable and l1822_not_applicable)` | ✅ |
| **P0-2** нет fallback для aggregate | добавлены `elif l23_arch: "l23_architectural"` и `else: "inconclusive"` | ✅ |
| **P0-3** header противоречит коду/meta по метрике вердикта | header теперь «driven by min_ratio_t … (the header previously claimed the verdict used the global-scale … which contradicted the code)» | ✅ |
| **P0-4/P1-4** ratio-of-medians → median-of-ratios | `per_doc_ratio = gap_min_td / (EPS_MAIN * s1sq); ratio_t = np.median(per_doc_ratio, axis=1)` | ✅ |
| **P0-5/P1-5** нулевой `σ_1²` → `well_separated=True` | `valid = thr_base > 0`, `wse[valid] = ...` | ✅ |
| **P1-6** неиспользуемый `eigengap_min` | импорт удалён, комментарий про zero-spectrum edge | ✅ |
| **P1-7** `l23_architectural` — флаг, не aggregate | одновременно `verdicts["aggregate"] = "l23_architectural"` и `verdicts["l23_architectural"] = l23_arch` | ✅ |
| **P1-9** guard `R <= min(K,D)` | `raise ValueError` | ✅ |
| **P1-11** guard `ref_threshold > 0` | `global_ratio = ... if ref_threshold > 0 else inf` | ✅ |
| **P1-12** NaN в J / nan-robust медианы | `if not torch.isfinite(J).all(): raise`, `np.nanmedian(...)` | ✅ |
| **P2** строковые ключи `fraction_well_separated` | `str(e)` | ✅ |
| **P2** `profile_gap_min_t` vs `s1sq_profile_t` разная длина | задокументировано `profile_gap_min_t_note` | ✅ |
| **P0** `del ck` дважды (как было в exp38b) | `del ck` один раз после `J = ck["J"]`; в конце итерации только `del J` | ✅ отсутствует |

Логика вердиктов проверена по всем комбинациям:
- `all_not_applicable` ∧ `l23_arch` — невозможно (противоречие по L23);
- `near_degen` ∧ `l23_arch` — невозможно (противоречие по L18/L22);
- случай L6 < 1.0, L18/L22 ≥ 1.0, L23 ≥ 1.0 — `inconclusive` (новый fallback работает);
- случай L6 < 1.0, L18/L22 ≥ 1.0, L23 < 1.0 — `l23_architectural` (игнорирует L6, соответствует ТЗ).

## P1 — тот же баг, что в exp38a/b/d: `_json_default` не работает для `float('nan')` / `float('inf')`

```python
def _json_default(o):
    if isinstance(o, np.bool_):
        raise TypeError(...)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    return str(o)
json.dump(out, f, indent=1, default=_json_default)
```

`json.dump`'s `default=` вызывается **только** для типов, которые encoder не умеет сериализовать. `float` (включая `float('nan')` и `float('inf')`) — умеет (пишет литералы `NaN` / `Infinity`). Значит:

- `isinstance(o, (float, np.floating)) and not np.isfinite(o)` — **никогда не выполняется** для Python `float` и для `np.float64` (subclass);
- ветка срабатывает **только** для `np.float32` NaN/Inf (не подкласс `float`).

**Источники NaN/Inf в текущем `out`:**
- `global_ratio = inf` при `s1sq_global_med = 0` (все `J` нулевые) — `float('inf')` уйдёт как `Infinity`;
- `min_ratio_t = float(ratio_t.min())` — если `per_doc_ratio` had NaN, `np.median` вернёт NaN, `float(nan)` → `NaN` literal. Сейчас маскируется через `np.where(np.isfinite(...), ..., 0.0)`, но маска неполная (см. п.3 ниже);
- `min_t_gap_min = float(gap_t_med.min())` — теоретически NaN при all-NaN столбце.

**Фикс — тот же, что нужно применить в exp38a/b/d:** пре-санитизация + `allow_nan=False`:

```python
def _sanitize(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [_sanitize(v) for v in o]
    return o

json.dump(_sanitize(out), f, indent=1, allow_nan=False)
```

`allow_nan=False` гарантирует **громкий** `ValueError`, если какая-то NaN/Inf проскочит мимо санитайзера.

## P2 — `argmin_t` использует не тот агрегат, что вердикт

```python
"argmin_t": int(gap_t_med.argmin()),  # ← argmin of MEDIAN GAP
...
"verdict_metric": "min_ratio_t (per-t threshold)",
```

Вердикт определяется через `ratio_t.argmin()`, но `argmin_t` — это `gap_t_med.argmin()`. Эти два могут указывать на разные `t`, если `σ_1²` сильно варьируется по `t`. `argmin_t` в JSON читается как «позиция, определившая вердикт», но это не так.

**Фикс:** либо переименовать в `argmin_gap_t`, либо добавить `argmin_ratio_t = int(ratio_t.argmin())`. Второе лучше — это позиция, определившая вердикт.

## P2 — маска `np.isfinite` → 0.0 маскирует и `inf` тоже

```python
per_doc_ratio = gap_min_td / (EPS_MAIN * s1sq)   # [S, N]
per_doc_ratio = np.where(np.isfinite(per_doc_ratio), per_doc_ratio, 0.0)
```

Случаи:
- `gap=0, s1sq=0` → `0/0 = NaN` → маска 0.0. **Правильно** (максимальная дегенерация).
- `gap>0, s1sq=0` → математически невозможно (`σ_1=0 ⇒ все σ_i=0 ⇒ gap=0`).
- `gap=0, s1sq>0` → `0/positive = 0`. **Правильно** (максимальная дегенерация).
- `gap>0, s1sq>0` но `EPS_MAIN * s1sq` **underflow** в float32 → `gap/0 = inf` → маска 0.0. **Неправильно** — это случай **сильной** сепарации (gap >> threshold), а маскируется в максимальную дегенерацию.

Underflow `EPS_MAIN * s1sq` в float32 случился бы при `s1sq < ~1e-40` (float32 denormal), но деление здесь происходит в **float64**: `s1sq = np.zeros((Ss, N))` создаётся float64 (значения из torch float32 кастятся при присваивании), значит подпроток только при `s1sq < ~1e-308` — недостижимо для реального Якобиана (σ₁² ~ O(1)). Иными словами: `inf` в `per_doc_ratio` на практике недостижим, и на всех достижимых входах маска верна (`0/0 → NaN → 0.0` = максимальная дегенерация; при `s1sq > 0` знаменатель неотрицателен). Контракт всё же неточен: если бы `inf` появился, он маскировался бы в «максимальную дегенерацию» — в неверную сторону (это случай **сильной** сепарации). Для точности контракта (не для текущих чисел) маскировать только NaN:

```python
denom = EPS_MAIN * s1sq
per_doc_ratio = gap_min_td / np.where(denom > 0, denom, 1.0)  # x/1, не x/0
per_doc_ratio = np.where(denom > 0, per_doc_ratio, 0.0)       # 0/0 → 0.0
per_doc_ratio = np.where(np.isnan(per_doc_ratio), 0.0, per_doc_ratio)
```

Либо, если проверка на `inf` требуется только как защита от неизвестных случаев, заменить `np.isfinite` на `~np.isnan` (NaN-маска) и оставить `inf` как «очень далеко от дегенерации» — тогда он естественно пройдёт `ratio_t.min() > 1` и не повлияет на вердикт.

## P3 — устаревший комментарий

```python
sv = torch.linalg.svdvals(J[:, t])  # [N, K] desc
sv16 = sv[:, :R]                    # [N, R] float64   ← неверно
```

`sv` — float32 (J — float32). Комментарий «float64» — stale от предыдущей (reverted) версии. Не влияет на корректность, но противоречит собственному комментарию ниже («float32 SVD, same as the original run»).

## Проверка математики и float32-обоснования

**float32 в SVD.** Docstring/комментарий утверждает: «float32 SVD noise in sigma^2 is ~5e-7 relative — 10^4x below the eps_frac=0.005 threshold». Строго:

- Абсолютная ошибка σ_i ≈ `eps_f32 · σ_1 · κ_i`, где κ_i — локальное число обусловленности;
- Ошибка σ_i² ≈ `2 σ_i · eps_f32 · σ_1 · κ`;
- Ошибка gap = σ_i² − σ_{i+1}² ≈ `2 eps_f32 · σ_1² · κ`;
- Порог = `eps_frac · σ_1² = 5e-3 · σ_1²`.

При κ ~ 10 ошибка ~`2e-6 · σ_1²`, что действительно ≪ 5e-3 · σ_1². **Обоснование корректно для κ ≲ 1000**. Для сильно ill-conditioned J (κ ~ 10⁵) ошибка σ_16² и σ_17² может стать сопоставимой с их разницей. В таком случае вердикт `well_separated` для конкретного (t, doc) может флипнуться. Но это лишь влияет на `fraction_well_separated` (диагностический отчёт), не на `min_ratio_t` (усреднение по 32 докам). Для практических `J` из Qwen3.5 — κ обычно ≲ 10². Приемлемо.

**`np.nanmedian` vs `np.median`.** Замена на `np.nanmedian` не вредит (все данные конечны — проверено `isfinite(J)`).

**`global_ratio`.** Комментарий «reported for reference only» — корректно; вердикт не использует.

## Соответствие ТЗ

| Пункт | Реализация | Статус |
|---|---|---|
| `non_abelian_not_applicable` — все слои `>= 1.0` | `all_not_applicable` | ✅ |
| `near_degeneracy` — L18 или L22 `< 1.0` | `near_degen` | ✅ |
| `l23_architectural` — L23 `< 1.0`, L18/L22 `>= 1.0` | `l23_arch` | ✅ (инверсия исправлена) |
| `inconclusive` — fallback | `else` | ✅ (новый) |
| `fraction_well_separated = median_t(mean_doc ws)` | `np.nanmedian(ws_td[e].mean(axis=1))` | ✅ |
| `min_t_gap_min = min_t(median_doc gap_min)` | `float(gap_t_med.min())` | ✅ |
| `Layers not averaged` | per-layer | ✅ |
| `L23 trivial by construction` | `informative` | ✅ |
| `eps_frac = 0.01`, sweep `{0.005, 0.01, 0.02}` | `EPS_MAIN`, `EPS_SWEEP` | ✅ |
| `well_separated` semantics inverted vs `applicable` | `meta.direction` | ✅ |
| `verdict uses min_ratio_t`, not global ratio | header + code + meta согласованы | ✅ |
| string keys для `fraction_well_separated` | `str(e)` | ✅ |
| `profile_gap_min_t` длина S/4 | `gap_t_med[::4]` + note | ✅ |

## Итоговый приоритетный список

**P1 — до отчёта:**
1. `_json_default` не работает для `float('nan')`/`float('inf')` — нужна рекурсивная пре-санитизация + `allow_nan=False`. Тот же баг в exp38a/b/d. Особенно важно для `global_ratio = inf` при `s1sq_global_med = 0`.

**P2 — перед публикацией JSON:**
2. `argmin_t` использует `gap_t_med.argmin()`, а не `ratio_t.argmin()` — добавить `argmin_ratio_t` (позиция, определяющая вердикт), оставить `argmin_t` как `argmin_gap_t` или задокументировать.
3. Маска `np.isfinite(per_doc_ratio) → 0.0` маскирует и `inf` (underflow `EPS_MAIN * s1sq`) в «максимальную дегенерацию» — маскировать только NaN, а inf оставить как «≥ 1.0» (или явно guard).

**P3 — косметика:**
4. `sv16 = sv[:, :R]  # [N, R] float64` — stale comment; sv — float32.

## Верификация исправлений из exp38b (за пределами этого файла)

`exp38c` не содержит P0-бага `del ck` из `exp38b`. `del ck` один раз, `del J` один раз. ✓

## Общий вывод

**Все 12 заявленных фиксов из прошлого ревью действительно исправлены**, включая инверсию `l23_architectural`, добавление fallback `inconclusive`, согласование header/code/meta по метрике вердикта, median-of-ratios, mask для нулевого `σ_1²`, и guards для `R <= min(K,D)` / `ref_threshold > 0`. Логика вердиктов проверена — все edge cases обработаны согласованно, `l23_architectural` и `near_degeneracy` взаимоисключающие.

**Остался один унаследованный P1** — `_json_default` не работает для `float`-NaN/Inf. Он **не проявляется** в текущем прогоне (J конечен, `s1sq_global_med > 0`), но контракт «valid JSON for any cache» нарушен. Рекомендую исправить одновременно в exp38a, exp38b, exp38c, exp38d и вынести в `prism/json_utils.py`.

# Ревью `exp38e_MK_diagnostic.py` (исправленная версия)

## Верификация заявленных фиксов

| Пункт | Фикс | Проверка |
|---|---|---|
| **P0** `np.median`/`mean` на NaN в α/β | `np.nanmedian` / `np.nanmean` | ✅ |
| **P0** `pos0_ratio` на пустом срезе | `if pos0_vals.numel()` | ✅ |
| **P0** мёртвый `ck23.get("V")` | удалён, задокументирован в `meta.vocab_guard` | ✅ |
| **P0** `default=float` пропускает NaN | `_json_default` | ❌ (см. P1-1) |
| **P1** `consistency_redflag` — односторонний | `meta.redflag_semantics` | ✅ |
| **P1** `jtu_ratio ≈ 4` — завышено | `r_t_expected = median(‖h‖)/√D`, комментарий про `~1–4` | ✅ |
| **P1** `sigma_g` → `sigma_1_MK` | переименовано | ✅ |
| **P1** `nan_frac > 0.5` → отдельный verdict | `gate = "degenerate"` | ✅ (но см. P1-3) |
| **P2** `pos0_ratio` без cap | `POS0_CAP = 1e6` + флаг | ✅ |
| **P2** `K = 128` hardcoded | `K = len(token_set)` | ✅ |
| **P2** `del M_K, H, J23, ck23` | добавлено | ✅ |
| **P2** `ck23["U128"]` list[int] | OK | ✅ |

## P1 — новый баг: `bootstrap_ci` падает при NaN в `ratios_docs`

```python
ratios_docs = [float(np.nanmedian(ratio[i].numpy())) for i in range(N)]
...
per_layer = {
    "L23": {
        "ratio_alpha_beta": {
            "median_over_docs": ratio_median,
            "ci": list(bootstrap_ci(np.array(ratios_docs), B=B, seed=SEED)),  # ← crash
            ...
        },
        ...
    }
}
```

`cache.bootstrap_ci`:
```python
if not np.all(np.isfinite(arr)):
    raise ValueError("bootstrap_ci: NaN/Inf in values")
```

**Сценарий отказа:** один документ полностью дегенерирован (`norm < 1e-6` на всех S позициях) → `ratio[i]` — все NaN → `np.nanmedian` → NaN → `ratios_docs[i] = NaN` → `bootstrap_ci` бросает `ValueError`.

**Важно:** этот crash происходит **до** `gate = "degenerate"` (per_layer строится раньше, чем gate) — то есть guard `nan_frac > 0.5`, который был добавлен именно для такого случая, **недостижим**: скрипт падает раньше.

На текущем кэше (min ‖h‖ = 5.85) `nan_frac = 0`, все `ratios_docs` конечны → не проявляется. Но контракт «gated by nan_frac > 0.5» нарушен.

**Фикс:**
```python
finite_ratios = [x for x in ratios_docs if np.isfinite(x)]
ci = list(bootstrap_ci(np.array(finite_ratios), B=B, seed=SEED)) if len(finite_ratios) >= 2 else [float("nan"), float("nan")]
"ci": ci,
```

## P1 — `_json_default` не работает для Python `float('nan')` (унаследовано из exp38a/b/d)

```python
def _json_default(o):
    if isinstance(o, np.bool_):
        raise TypeError(...)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    raise TypeError(...)
```

**Проблема:** `json.dump`'s `default=` вызывается **только** для типов, которые encoder не умеет сериализовать. Python `float('nan')` сериализуем (пишет `NaN` literal) → `_json_default` **не вызывается**. Ветка `isinstance(o, float) and not np.isfinite(o)` **мёртвая** для Python float. `np.float64` — подкласс `float`, тоже сериализуем, поэтому тоже не попадает в `default`. Работает только для `np.float32` (не подкласс).

**Источники NaN/Inf в `out`:**
- `cons_ratio = float(np.nanmedian(beta_vec.numpy()) / sigma_1_MK)` — если все beta NaN → NaN; если `sigma_1_MK = 0` → inf/NaN;
- `ratio_median = float(np.nanmedian(ratios_docs))` — NaN при all-NaN;
- `alpha.median_abs`, `beta.median` — NaN при all-NaN;
- `pos0_ratio` — NaN если все pos0 NaN;
- `j_uhat_resid` — NaN если `valid.sum() == 0`.

**Фикс — пре-санитизация + `allow_nan=False`** (тот же баг в exp38a/b/d):
```python
def _sanitize(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [_sanitize(v) for v in o]
    return o
json.dump(_sanitize(out), f, indent=1, allow_nan=False)
```
`allow_nan=False` — громкий `ValueError`, если NaN проскочит мимо (defense-in-depth).

**Особо важно:** `cons_ratio` при `sigma_1_MK = 0` даёт `inf`, и это `float('inf')` (не `np.floating`) → JSON получает литерал `Infinity` → невалидный RFC-8259. При `_json_default` это **не ловится**.

## P1 — `gate = "degenerate"` не останавливает E-38f

```python
"outcomes": {"gate": gate, "consistency": cons,
             "gate_failed": gate == "gate_failed"},
```

В `exp38f_structural_consistency.py`:
```python
gate_ok = not g["outcomes"].get("gate_failed", False)
```

При `gate = "degenerate"` → `gate_failed = False` → E-38f **процессит** дегенерированные данные. Docstring exp38f: «Gate: run only if E-38e did not report gate_failed». `degenerate` — новое состояние, семантически эквивалентное по последствиям для E-38f (все медианы неопределены), но не эквивалентное по контракту.

**Фикс — либо в exp38e:**
```python
"gate_failed": gate in ("gate_failed", "degenerate"),
```
**либо в exp38f:**
```python
gate_ok = g["outcomes"]["gate"] not in ("gate_failed", "degenerate")
```

## P2 — `resid[valid].median()` без guard

```python
resid = (torch.einsum("ntkd,ntd->ntk", J23, u).norm(dim=-1)
         / J23.norm(dim=(-1, -2)).clamp_min(1e-12))
j_uhat_resid = float(resid[valid].median())
```

Если `valid.sum() == 0` (все `norm < 1e-6`) → `resid[valid]` пустой → `.median()` бросает `RuntimeError` на новых torch, молча даёт NaN на старых. Тот же класс бага, что был исправлен для `pos0_vals`. Параллельный guard:
```python
resid_valid = resid[valid]
j_uhat_resid = float(resid_valid.median()) if resid_valid.numel() else float("nan")
```

Не проявляется на текущем кэше, и в `nan_frac = 1.0` сценарии скрипт упадёт раньше в `bootstrap_ci` (P1-1). Поэтому P2, а не P1.

## P2 — `cons = "consistency_redflag"` при NaN `cons_ratio`

```python
cons = ("consistency_consistent_with_E33r" if abs(cons_ratio - 0.9) <= TOL_CONSISTENCY
        else "consistency_redflag")
```

При `cons_ratio = NaN` → `abs(NaN - 0.9) <= 0.15` = False → `cons = "consistency_redflag"`. Молчаливое присвоение `redflag` вместо явного `undefined`. Если `gate = "degenerate"` и `cons = "redflag"` — читатель может подумать, что обе проверки независимо провалились. Лучше:
```python
cons = ("consistency_consistent_with_E33r" if abs(cons_ratio - 0.9) <= TOL_CONSISTENCY
        else ("consistency_undefined" if not np.isfinite(cons_ratio) else "consistency_redflag"))
```

## P2 — прочее

- **`torch.manual_seed(SEED)`** — декоративно (нет случайных операций: model loading, детерминированные вычисления). Не критично, но создаёт иллюзию контроля.
- **`attn_implementation="eager"`** — не используется (нет forward pass). Убрать или пометить `# for protocol consistency only`.
- **`w` после `del M_K, H, J23, ck23`** — `w` [1024] и `u` [N,S,D] остаются живы до конца функции. `u` ~2.1 GB при N=32, S=128, D=1024. Добавить `del u, w2, beta_vec, u` перед dump (не критично, процесс завершается сразу).

## Итоговый приоритетный список

**P1 — исправить до отчёта:**
1. `bootstrap_ci(np.array(ratios_docs), ...)` падает при NaN → guard на finite subset (P1-1).
2. `_json_default` не работает для Python `float('nan')`/`float('inf')` — нужна пре-санитизация + `allow_nan=False` (P1-2). Тот же баг в exp38a/b/d.
3. `gate = "degenerate"` не пропагируется в `gate_failed` → E-38f процессит дегенерированные данные (P1-3).

**P2 — перед публикацией:**
4. `resid[valid].median()` без guard.
5. `cons = "consistency_redflag"` при NaN `cons_ratio` → отдельная метка `consistency_undefined`.
6. `del u, w2, beta_vec` перед dump (уменьшить peak на ~2 GB до exit).
7. `torch.manual_seed` / `attn_implementation` — пометить как «не используется» или удалить.

## Сводка

Все **P0** и большинство **P1** из прошлого ревью — исправлены корректно. Остались три P1: `bootstrap_ci` на NaN (новый), `_json_default` на Python float (унаследованный и перенесённый в exp38e), `gate = "degenerate"` не пропагируется в E-38f (новый). Плюс два P2. Все три P1 не проявляются на текущем кэше (min norm 5.85, `nan_frac = 0`, `sigma_1_MK > 0`), но нарушают контракт «valid JSON and gated E-38f for ANY cache». После их исправления exp38e будет robust к произвольному кэшу.

# Ревью исправленного `exp38f_structural_consistency.py`

## Верификация фиксов

| Пункт | Фикс | Проверка |
|---|---|---|
| **P0** гейт fail-open при отсутствии `GATE_FILE` | `gate_file_missing`, `gate_ok = not gate_file_missing` | ✅ |
| **P1** `_json_default` в `main` | вынесен на модульный уровень | ✅ (но см. P1-1 ниже) |
| **P2** `with open(OUT, "w")` | `with open(OUT, "w") as f` + `json.dump(..., f, ...)` | ✅ |
| **P2** `os.makedirs` для skipped-ветки | добавлен | ✅ |
| **P0/P1** критические математические проблемы (`g ≠ top mode`, rename verdict, `frac_positions_above_upper`) | — | **❌ НЕ исправлены** |

## P0 — главный баг остался: код не ограничивает `g` топ-модой

Meta теперь утверждает:

> «the lemma J^T u = c β v_⊥ is an exact identity — the cos test measures the EMPIRICAL alignment v_⊥ ∥ v_g, **restricted to g = top mode** (Δcos ≥ 0 is impossible to violate otherwise — P0, docs/E38_fixes2.md)»

Код:

```python
c = (U.T @ one).abs()
k = int(c.argmax())          # ← никакой проверки k == 0
ug = U[:, k]
vg = Vt[k]
```

**Ограничения на `g = top mode` в коде нет.** `k = argmax_k |c_k|` может дать `k ≠ 0`, и это не отфильтровывается, не логируется, не отбрасывается.

**Сценарий отказа (тот же, что в прошлом ревью):** `J = diag(σ_1, σ_2, …)` с `σ_1 > σ_2`, `u = 1/√K` не совпадает с `u_1`:

- если `|⟨u_1, u⟩| < |⟨u_2, u⟩|`, то `k = 2` (не топ-мода);
- `cos(v_g, v_pred) = σ_2 c_2 / ‖J^T u‖`;
- `‖J^T u‖² = σ_1² c_1² + σ_2² c_2² + …`;
- если `c_2` доминирует, `‖J^T u‖ ≈ σ_2 c_2 · √(1 + (σ_1 c_1/(σ_2 c_2))²) > σ_2 c_2`, значит `cos < 1`;
- `Δcos = cos − c_g < c_g(σ_2/‖J^T u‖) − c_g` — может быть `<< 0`.

Конкретный контрпример: `J = diag(1, 0.5)`, `u = (0.6, 0.8)` — я приводил в предыдущем ревью. `k = 2`, `σ_g = 0.5`, `‖J^T u‖ ≈ 0.722`, `cos ≈ 0.554`, `c_g = 0.8`, `Δcos ≈ −0.246`. Правило `dc_med < -eps → lemma_broken` даст ложный `lemma_broken` на конфигурации, где лемма выполнена точно.

**Следствие:** meta содержит **ложное утверждение** («restricted to g = top mode»), а вердикт может быть ложно-отрицательным (`lemma_broken` при корректной математике). Это тот самый P0, который был отмечен в прошлом ревью и не исправлен.

**Фикс (один из):**

1. **Фильтр `k == 0`:**
   ```python
   if k != 0:
       cv = float("nan")
       n_g_not_top += 1
       continue
   ```
   С отдельным счётчиком `n_g_not_top` и `g_not_top_frac` в `metrics.dcos`. Вердикт — на подмножестве с `g == 0`.

2. **Переформулировать проверку без `argmax`:** использовать `v_g = Vt[0]` (топ-мода по определению paper §2.5), а `c_g` — `|u_1 · 1/√K|`. Тогда `Δcos ≥ 0` следует из леммы автоматически, и тест честен.

3. **Прямая проверка тождества** `‖J^T u − c β v_⊥‖ / (c β) < tol` — но для этого нужно измеренное `β, v_⊥` (из E-38e), т.е. другой протокол.

В любом случае, `Δcos < 0` — **не** нарушение леммы вне режима `g=top`; это математическое свойство `σ_g < σ_1`. Правило вердикта должно это учитывать или исключать такие `(doc, t)`.

## P1 — `_json_default` по-прежнему не работает для Python `float('nan')`

```python
def _json_default(o):
    if isinstance(o, np.bool_):
        raise TypeError(...)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    raise TypeError(...)
json.dump(out, f, indent=1, default=_json_default)
```

`json.dump`'s `default=` вызывается **только** для типов, которые encoder не умеет. Python `float('nan')`/`float('inf')` — умеет (пишет `NaN`/`Infinity` literal). Значит ветка `isinstance(o, float) and not np.isfinite(o)` **мёртвая для Python float**. Работает только для `np.float32` (не подкласс `float`), но не для `np.float64` (подкласс).

**Источники NaN/Inf в `out` при `has_data == False`:**

```python
dc_med = float("nan")
dc_ci = [float("nan"), float("nan")]
...
"median": dc_med,     # → NaN literal
"ci": dc_ci,          # → [NaN, NaN]
```

В skipped-ветке (`gate_file_missing` или `gate_failed`):
```python
json.dump({"params": {"seed": SEED, "gate": "gate_failed"},
           "outcomes": {"skipped": True},
           ...
```
Тут NaN нет — безопасно.

Но в основной ветке при `not has_data` (все документы с NaN `Δcos` — теоретически при `‖J_t‖ ≈ 0` везде) `dc_med = nan`, `dc_ci = [nan, nan]` попадут в JSON как литералы. Валидность нарушена.

Это тот же баг, что в `exp38a/b/c/d/e`. Фикс — пре-санитизация + `allow_nan=False`:

```python
def _sanitize(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [_sanitize(v) for v in o]
    return o
json.dump(_sanitize(out), f, indent=1, allow_nan=False)
```

`allow_nan=False` — громкий `ValueError` если NaN проскочит.

## P1 — `verdict = "lemma_confirmed" / "lemma_broken"` не переименован

Метки остались прежними. Meta теперь признаёт: «measures the EMPIRICAL alignment v_⊥ ∥ v_g» — но само поле `verdict` при этом говорит «lemma_confirmed». Программный потребитель (E-38f consumer / следующий скрипт / reader JSON) увидит `lemma_confirmed` и интерпретирует как «лемма подтверждена», что **неверно**.

По прецеденту E-38a (`mixed`/`edges_atypical_candidate` — «новый label нельзя вводить молча»), замена меток должна сопровождаться переименованием поля и явной записью в `meta.labels_renamed`. Сейчас `meta.note` описывает семантику, но `verdict` остаётся «lemma_confirmed».

**Фикс:** `verdict = "empirical_alignment_within_range"` / `"empirical_alignment_violated"` / `"methodological_anomaly"` / `"insufficient_data_nan"`, с записью в `meta` о переименовании из pre-registered `lemma_confirmed`/`lemma_broken`.

## P1 — `frac_positions_above_upper` всё ещё не добавлено

Вердикт по-прежнему опирается на `dc_med` (медиана per-doc медиан). Если для отдельных `(doc, t)` `Δcos > 1 − c_g + 0.05` — методологическая аномалия для этой позиции, — она исчезает в медиане.

**Фикс:** добавить поэлементную проверку:
```python
upper_per_t = 1.0 - np.array(cos_ug_one).reshape(N, Ss) + eps
frac_above = float((dc > upper_per_t).mean())
```
в `metrics.dcos.frac_positions_above_upper`. Это диагностика чувствительности, не вердикт (как и `degen_sweep` в E-38d).

## P1 — `mean` и `median` — разные агрегаты, не задокументировано

```python
"median": dc_med,                                    # median-of-per-doc-medians
"ci": dc_ci,                                         # bootstrap over per-doc medians
"mean": float(np.nanmean(dc)),                       # flat mean over 4096 values
```

`dc` — `[N, S]` flattened (`np.array(dcos).reshape(N, Ss)`), так что `mean` — плоское среднее по `N·S`, `median` — медиана медиан. Units несопоставимы. `ci_note` документирует только `ci`, но `mean` остаётся недокументированным.

**Фикс:** либо `mean_per_doc = float(np.nanmean([np.nanmean(dc[i]) for i in range(N)]))` (тоже median-of-means, согласовано), либо `mean_note: "flat mean over 4096 values — not comparable to median (median-of-medians)"`.

## P1 — `eps = 0.05` — статистический допуск, а не численный

Meta (старая и новая):
> «Δcos < −0.05 = lemma_violation … negative is impossible exactly»

Если «impossible exactly», то `Δcos < 0` (а не `< −0.05`) должно давать `lemma_broken`. `0.05` в ~10⁵ раз больше float32-шума SVD (~`1e-7`). Значит, `0.05` — статистический допуск, признающий, что эмпирически `Δcos` может отклоняться от 0. Это делает вердикт «confirmed» на `Δcos ∈ [−0.05, 0)` — **фактически эмпирическим**, а не лемматическим.

Не исправлено. В `meta.note` явно не записано, что `0.05` — статистический, не численный допуск.

## P2 — dead code

- **`ug = U[:, k]`** — присваивается, никогда не используется (только `vg = Vt[k]` и `c[k]`). Удалить или использовать (например, для `j_uhat_resid`-подобной проверки).
- **`K = 128`** на модульном уровне — не используется (используется `Kk = J.shape[2]`).
- **`from _e38_cache import S`** — `S` не используется; `Ss = J.shape[1]` локально.

## P2 — `n_vpred_degenerate` не отчитывается

При `vp is None` (RuntimeError из `v_pred_from_JT`) `cv = NaN`, эти `(doc, t)` неотличимы от «нормальных» NaN. Стоит добавить счётчик:

```python
n_vpred_degenerate = 0
...
try:
    vp = v_pred_from_JT(Jt, one)
except RuntimeError:
    vp = None
    n_vpred_degenerate += 1
```
и в `metrics.dcos.vpred_degenerate_count`.

## Сводка

| Категория | Число |
|---|---|
| Исправлено корректно | 3 (gate fail-closed, `_json_default` на модуль, `with open`) |
| **Не исправлено, P0** | 1 (`g = top mode` не проверяется; meta ложно утверждает обратное) |
| **Не исправлено, P1** | 5 (`_json_default` NaN literal, verdict rename, `frac_positions_above_upper`, mean/median units, `eps=0.05` justification) |
| Новые P2 | 3 (`ug` dead, `K` dead, `n_vpred_degenerate` missing) |

## Итоговый приоритетный список

**P0 (обязательно до отчёта):**
1. Ограничить тест режимом `g = top mode` (фильтр `k != 0` → NaN + счётчик), либо переформулировать через `v_g = Vt[0]`. Сейчас meta содержит **ложное** утверждение «restricted to g = top mode», а вердикт может быть ложно-отрицательным.

**P1 (до отчёта):**
2. `_json_default` не предотвращает NaN literal для Python `float('nan')` — пре-санитизация + `allow_nan=False`. Особенно критично для `not has_data` ветки (`dc_med = nan`, `dc_ci = [nan, nan]`).
3. Переименовать `verdict` (`lemma_confirmed` → `empirical_alignment_within_range` и т.д.) + `meta.labels_renamed` (по прецеденту E-38a).
4. Добавить `frac_positions_above_upper` (per-`(doc, t)` проверка).
5. Привести `mean` к тем же единицам, что `median`, или задокументировать.
6. Записать в `meta`, что `eps = 0.05` — статистический, не численный допуск.

**P2:**
7. Удалить `ug`, `K = 128`, `import S`; добавить `n_vpred_degenerate`.

## Общий вывод

Из трёх P0 из прошлого ревью (математическая ошибка `g ≠ top`, `_json_default` NaN, `gate_file_missing`) **исправлен только gate**. Два из трёх не исправлены, причём критический математический дефект теперь **маскируется** meta-утверждением «restricted to g = top mode», которое код не реализует. Это хуже, чем было: раньше было явно видно, что нет ограничения; теперь читатель meta поверит, что ограничение есть.

Остальное (rename, `frac_positions`, units) — не исправлено и не заявлено исправленным. Скрипт можно запускать, но при появлении non-top-mode-aligned `(doc, t)` (что не исключено для edge-документов и/или L23 при малых `‖h‖`) он выдаст `lemma_broken` на валидной математике. После P0-фикса (п.1) скрипт станет валидным; после P1 — устойчивым и корректно интерпретируемым.

# Ревью исправленного `exp38h_two_channel.py` + `exp38_hi_common.py`

## Верификация заявленных фиксов

| Пункт | Фикс | Проверка |
|---|---|---|
| **P0** двойное вычитание `W_orig` | `dW = {k: snaps_all[k] for k in ks}` | ✅ |
| **P1** tied-weights документация | `data.tied_weights` | ✅ |
| **P1** схема снапшотов | `data.schema` | ✅ |
| **P2** guard в `load_snapshots` | добавлен `assert cos < 0.5` | ❌ формула не ловит баг (см. P1-2) |

## P0 — композиция в `defect` — неверный порядок

```python
def defect(framedict, a, b, c, side):
    R12, _ = procrustes_align(framedict[b], framedict[a])   # V_b → V_a
    R23, _ = procrustes_align(framedict[c], framedict[b])   # V_c → V_b
    R13, _ = procrustes_align(framedict[c], framedict[a])   # V_c → V_a
    prod = R12 @ R23
    return float(((prod - R13).norm() / R))
```

**Семантика `procrustes_align(U1, U2)`** (docstring `holonomy.py`): возвращает `R` с `U1 @ R ≈ U2`, т.е. `R` — коэффициенты выражения `U2` в базисе `U1`. Тогда:
- `R12` — коэффициенты выражения `V_a` в базисе `V_b` (`V_b R12 ≈ V_a`);
- `R23` — `V_c R23 ≈ V_b`;
- `R13` — `V_c R13 ≈ V_a`.

**Композиция `V_c → V_b → V_a`:** `V_c R23 R12 ≈ V_b R12 ≈ V_a`, т.е. **`R13` должен равняться `R23 @ R12`**, а не `R12 @ R23`.

**Следствия:**
- Для **Abelian** случая (`R12, R23` коммутируют) `R12 R23 = R23 R12` — обе формулы совпадают, defect = 0. ✓
- Для **некоммутирующего** случая `R12 R23 ≠ R23 R12` — код считает `‖R12 R23 − R13‖/r`, что не равно `‖R23 R12 − R13‖/r`.
- Для **«одно подпространство, разные базисы»** (`V_a = V Q_a`, `V_b = V Q_b`, `V_c = V Q_c`): `R12 = Q_b^T Q_a`, `R23 = Q_c^T Q_b`, `R13 = Q_c^T Q_a`. Правильная композиция `R23 R12 = Q_c^T Q_b Q_b^T Q_a = Q_c^T Q_a = R13` → **defect = 0** (что и должно быть — bundle «плоский»). Код даёт `R12 R23 = Q_b^T Q_a Q_c^T Q_b ≠ Q_c^T Q_a` → **defect ≠ 0**. **Ложноположительный «не-Abelian» на тривиальном случае.**

**Согласованность с `holonomy_accumulate`:** там `R_total = R_total @ R_step` с `R_step = procrustes(V_i, V_{i+1})` — правильный порядок для forward-композиции (`V_0 R_0 R_1 = V_2`). В `defect` — обратный, но с аргументами, дающими `R_{b→a}` вместо `R_{a→b}`. Т.е. две ошибки «сократились» до одной.

**Правильно — один из вариантов:**

```python
def defect(framedict, a, b, c, side):
    R_ab, _ = procrustes_align(framedict[a], framedict[b])   # V_a → V_b
    R_bc, _ = procrustes_align(framedict[b], framedict[c])   # V_b → V_c
    R_ac, _ = procrustes_align(framedict[a], framedict[c])   # V_a → V_c
    prod = R_ab @ R_bc                                        # V_a R_ab R_bc ≈ V_c
    return float(((prod - R_ac).norm() / R))
```

или, если хотите сохранить текущие `procrustes(b, a)`:
```python
prod = R23 @ R12   # вместо R12 @ R23
```

**Согласованность с docstring:** `‖R12 R23 − R13‖_F / r` — формула в docstring соответствует КОДУ, но обе неверны. Строку docstring надо тоже исправить.

## P1 — `load_snapshots` assert не ловит баг, для которого написан

```python
cos = float((s @ Wo.T).norm() / (s.norm() * Wo.norm()))
assert cos < 0.5, (...)
```

`s @ Wo.T` — матрица [256, 256] всех попарных скалярных произведений строк. `‖s @ Wo.T‖_F / (‖s‖_F ‖Wo‖_F)` — норма **произведения матриц**, а не Frobenius-inner-product скаляра `⟨s, Wo⟩ = tr(s Wo^T)`.

**Численная проверка:**
- Для правильного `s = ΔW` (случайный малый): `‖s Wo^T‖_F ≈ ‖s‖ ‖Wo‖ / √n` (n=1024), `cos ≈ 1/32 ≈ 0.031`.
- Для «баг»-случая `s = W_curr ≈ Wo`: `‖s Wo^T‖_F = ‖Wo Wo^T‖_F = √(Σσ_k⁴)`, `‖s‖ ≈ ‖Wo‖ = √(Σσ_k²)`. Для [256, 1024] с σ ≈ 32 (MP-распределение): `cos ≈ √(256·32⁴) / (256·32²) = 32²·16 / (256·32²) = 16/256 = 1/16 = 0.0625`.
- Для double-sub `s = −Wo`: тот же `≈ 0.0625`.

**Обе ветки дают `cos ≈ 0.06 << 0.5` — assert никогда не срабатывает.** Комментарий «double subtraction gives cos ≈ −1» неверен: используемая норма `‖·‖_F` всегда неотрицательна, а `‖s @ Wo^T‖_F` — это **норма матрицы** [256, 256], не скалярное произведение.

**Фикс — Frobenius inner product скаляра:**
```python
inner = float((s * Wo).sum())              # ⟨s, Wo⟩_F
cos = abs(inner) / (s.norm() * Wo.norm())
assert cos < 0.5, (...)
```
- `s = c·Wo`: `|⟨c·Wo, Wo⟩| = |c| ‖Wo‖²`, `cos = 1` — срабатывает ✓.
- `s = −c·Wo`: `cos = 1` ✓.
- `s = ΔW`: `⟨ΔW, Wo⟩` — скалярное произведение двух «случайно ориентированных» [256, 1024], `cos ≈ 1/512 = 0.002` — проходит ✓.

Либо простое относительное сравнение:
```python
assert float(s.norm() / Wo.norm()) < 0.5, (...)
```
Для `ΔW ≈ 0.045·Wo` — проходит; для `W_curr ≈ Wo` или `−Wo` — 1.0, срабатывает. ✓

## P2 — `energy_top(ΔJ)` может превышать 1

```python
Jtt = (P_L @ dW[k] @ P_R) @ Jnorm           # ΔJ_top
Jt = dW[k] @ Jnorm                          # ΔJ = ΔJ_top + ΔJ_tail
jresp[k] = {"energy_top": float((Jtt.pow(2).sum() / Jt.pow(2).sum()))}
```

`ΔJ_top` и `ΔJ_tail` — не ортогональны в общем случае (обе — проекции `ΔJ` через `Jnorm` из разных `ΔW`-каналов). Возможна **anti-alignment** `ΔJ_top ≈ −ΔJ_tail`, тогда `‖ΔJ‖ << ‖ΔJ_top‖`, и `energy_top > 1`. Метрика перестаёт быть «долей» (share) в `[0, 1]`.

Для `energy_top(ΔW)` проблемы нет — `P_L A P_R` — проекция, `‖A_top‖ ≤ ‖A‖` всегда. Но для `energy_top(ΔJ)` — может выйти за `[0, 1]`.

**Фикс:** либо добавить диагностику (`energy_top_above_one: bool`), либо переформулировать как `‖ΔJ_top‖² / (‖ΔJ_top‖² + ‖ΔJ_tail‖²)` (положительная доля по определению, но теряет смысл «доли в ΔJ»).

В `meta` явно: «`energy_top(ΔJ)` is a share of ΔJ only when ΔJ_top ⊥ ΔJ_tail; may exceed 1 in the anti-aligned case — diagnostic only».

## P2 — dead code (не исправлено с прошлого ревью)

- `Us, sU, Vt = torch.linalg.svd(A_ref, ...)` — `sU` не используется.
- `U128, ck = load_u128()` — `U128` не используется (только `ck["H"]`).
- `W_orig = raw["W_orig"].to(torch.float64)  # noqa: F841 (schema check only)` — никакой schema check не использует `W_orig` (schema check — внутри `load_snapshots`). Просто dead load.
- `Jtail = Jt - Jtt` — не входит в `jresp[k]`.
- `h0 = ck["H"][("wiki", 0, 60)].double()` — позиция 60 не обоснована (P2 из прошлого ревью).

## P2 — отсутствует `_json_default`, но JSON не пострадает

`json.dump(out, f, indent=2, ensure_ascii=False)` без `default`. Все поля `out` — Python-скаляры (проверено), так что сериализуется. Но:
- `defect` возвращает `float(...)` — если `procrustes_align` когда-нибудь вернёт NaN (не должно, т.к. он проверяет orthonormality), попадёт `NaN`-литерал.
- На текущем коде риск нулевой, но паттерн небезопасен — если кто-то добавит поле с `np.float32` или torch-tensor, `json.dump` бросит `TypeError` (fail-loud, ОК) — но `float('nan')` пройдёт как литерал.

Стоит добавить (пре-санитизация + `allow_nan=False`) для консистентности с остальными E-38 файлами. Но это не срочно.

## Проверка математики, которая корректна

- `top_r_subspaces(A.t(), R)` даёт [256, 16] левых сингулярных векторов `A`. ✓
- `top_r_subspaces(A, R)` даёт [1024, 16] правых. ✓
- `P_L = U_t U_t^T`, `P_R = V_t V_t^T` — ортогональные проекции. ✓
- `energy_top(ΔW) = ‖P_L A P_R‖²/‖A‖²` — проекция, `≤ 1`. ✓
- `Jnorm = (√D/r0)(I − u0 u0^T) diag(γ)` — соответствует `[M-1]` с `gamma` в правильном порядке (out-dim scaling). ✓
- `ΔJ_top = (P_L ΔW P_R) @ Jnorm` — согласовано с `j_raw`. ✓
- `load_snapshots` возвращает `snaps[k] = ΔW[k]` (после `to(float64)`), consumer не вычитает `W_orig`. ✓ (за вычетом P1-2 про assert).

## Итоговый приоритетный список

**P0 — исправить до отчёта:**
1. `defect`: порядок композиции неверный. `prod = R12 @ R23` должен быть `prod = R23 @ R12`, или переопределить `R_ab, R_bc, R_ac` через `procrustes(V_a, V_b)` и т.д. Согласовать docstring.

**P1 — исправить до отчёта:**
2. `load_snapshots` assert: `‖s @ Wo.T‖_F / (‖s‖ ‖Wo‖)` — не тот норм. Заменить на `|⟨s, Wo⟩_F| / (‖s‖_F ‖Wo‖_F)`, либо на `s.norm()/Wo.norm() < 0.5`. Текущий assert никогда не срабатывает для full-rank `Wo` (`cos ≈ 0.06` в обоих случаях).

**P2 — перед публикацией:**
3. `energy_top(ΔJ)` — доля, но может превышать 1 при anti-aligned каналах. Задокументировать или переформулировать.
4. Dead code: `sU`, `U128`, `W_orig`, `Jtail`.
5. `h0` (position 60) — обосновать или усреднить по выборке.
6. `_json_default` — добавить для консистентности (не критично на текущих данных).

## Сводка

**P0 из прошлого ревью (двойное вычитание) исправлен корректно.** Появился/обнаружен **новый P0** — `defect` композиция — и **новый P1** — `load_snapshots` assert. Оба в одном файле, обе относятся к одной области (`procrustes` + `non-commutativity`), и оба не покрыты тестами (иначе были бы пойманы сразу).

До исправления P0-1 все результаты `non_commutativity.values.{top,tail}.{right_hidden,left_token}` — недействительны как associativity-defect; это либо

# Ревью `exp38i_factorization.py` (исправленная версия)

## Верификация заявленных фиксов

| Пункт | Фикс | Проверка |
|---|---|---|
| **P0** двойное вычитание `W_orig256` | `dW_shared = snaps_all[k][shared_idx256]` | ✅ |
| **P0** тавтологичность вердикта | новый `verdict`: «identity holds … tautological by construction» | ✅ |
| **P1** tied-weights `h_invariant` | `structural_fact` описывает корректно | ✅ |
| **P1** shared-only drift | `J_rel_drift_{max,median}_shared`, `MK_rel_drift_{max,median}_shared` | ✅ |
| **P2** `states[0] = W_full.clone()` | ✅ | ✅ |
| **P2** `del Jraw, ck` NameError | теперь `W_full = emb[U]`, никаких `del ck` | ✅ (отсутствует) |

## P1 — унаследованный (из shared): `load_snapshots` assert не выполняет свою функцию

```python
# exp38_hi_common.py
cos = float((s @ Wo.T).norm() / (s.norm() * Wo.norm()))
assert cos < 0.5, (...)
```

Формула вычисляет **норму матричного произведения** `‖s @ Wo.T‖_F / (‖s‖_F ‖Wo‖_F)`, а не **Frobenius inner product** `|⟨s, Wo⟩_F| / (‖s‖_F ‖Wo‖_F)`.

Численная оценка для `s`, `Wo` ∈ `[256, 1024]`, `‖s‖_F = ‖Wo‖_F = 512σ`:
- Правильный `s = ΔW` (случайный): `E[‖s @ Wo.T‖_F] ≈ ‖s‖_F ‖Wo‖_F / √D` → ratio ≈ `1/√1024 = 1/32 ≈ 0.031`.
- Баг `s = −W_orig` (двойное вычитание): `‖Wo Wo.T‖_F / ‖Wo‖_F² = √(Σσ⁴)/Σσ² ≈ 1/√256 ≈ 0.0625` (для uniform rank-256).
- Баг `s = W_curr` (не вычли `W_orig`): `cos ≈ 0.0625` тоже.

**Все три случая дают `cos << 0.5` → assert проходит всегда, баг не ловится.** Комментарий «double subtraction gives cos ≈ −1» — неверен: `‖·‖_F` неотрицательна, а матричное произведение — не то же, что скалярное. Assert никогда не срабатывает на реальных весах (разве что при rank-1 `Wo`).

Этот же assert стоит в `exp38h`, где я его уже отмечал. `exp38i` его не избегает.

**Фикс** (проще всего по норме):
```python
assert float(s.norm() / Wo.norm()) < 0.5, (
    f"snapshot step {k}: ‖snap‖/‖W_orig‖ = {float(s.norm()/Wo.norm()):.3f} "
    "— snapshots[k] is ΔW (small perturbation, ≈0.045·‖W_orig‖); "
    "a ratio ≈ 1 means the snapshot is W_curr, not ΔW")
```
или через Frobenius inner product:
```python
inner = float((s * Wo).sum())
cos = abs(inner) / (s.norm() * Wo.norm())
assert cos < 0.5, (...)
```

## P2 — `verdict_rule` противоречит `verdict`

```python
"verdict_rule": "max_s residual <= %.0e => preserved" % RESID_TOL,
"verdict": verdict,   # "identity holds ... tautological by construction ..."
```

Вердикт говорит «tautological», а `verdict_rule` — «preserved». Это ровно та тавтология, от которой ушли в docstring — но она возрождается в теле JSON. Читатель, обращающийся к `verdict_rule` (наиболее машинно-читаемому полю), получит старую семантику.

**Фикс:**
```python
"verdict_rule": "max_s residual <= %.0e => identity holds on edited W "
                "rows (residual at numeric floor; tautological — drift "
                "metrics carry the information)" % RESID_TOL,
```

## P2 — `W_orig256` — stale ключ в `data`

```python
"data": {
    "W_orig256": "exp16_dW_snapshots.pt (fp16->fp64), step grid",
    ...
```

`W_orig256` больше не читается (после P0-фикса). Ключ описывает то, чего в коде нет. Переименовать в `"snapshots"` / `"delta_source"`, либо удалить.

## P2 — отсутствует `_json_default`; при NaN получится `NaN` literal

```python
json.dump(out, f, indent=2, ensure_ascii=False)
```

`out` весь построен на `float(...)`, `bool(...)`, `int(...)`, `str`, `dict`, `list`. Для текущих данных (min ‖h‖ = 5.85, все `h` невырождены) NaN не появится. Но потенциально:

- `jdr.append(rel.max().item())` — если `Ji0[k]` имеет нулевую строку, `rel[k] = nan/inf`, `max` → `nan`;
- `MK_rel_drift_max = float(((...)).max())` — аналогично при `MK0[k] = 0`;
- `J_sh.median()` — `nan` при наличии `nan` в 13 значениях.

**Фикс** (как в exp38a–g): пре-санитизация + `allow_nan=False`:
```python
def _sanitize(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "nan" if o != o else ("inf" if o > 0 else "-inf")
    if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [_sanitize(v) for v in o]
    return o
json.dump(_sanitize(out), f, indent=2, ensure_ascii=False, allow_nan=False)
```

Единый фикс для всех E-38 скриптов; лучше вынести в `prism/json_utils.py`.

## P3 — `s = 0` в `out_states` — избыточная запись

```python
states = {0: W_full.clone()}
for k in K_STEPS: ...
...
for s, W_ed in states.items():
    ...
```

`s = 0` попадает в `out_states` со всеми drift = 0 (`J = J0`, `MK = MK0`). Это uninformative baseline; он занимает место в JSON и в `resid_max_all = max(v["factorization_residual_max"] for v in out_states.values())` — но для `s = 0` residual ≈ 1e-15 (fp64 precision), не влияет на вердикт.

Либо явно пометить (`"s=0 is baseline; included for reference"` в `meta`), либо исключить из `out_states` (baseline не нужен — drift = 0 по построению).

## P3 — отсутствие guard'а на нулевые строки в знаменателях drift

```python
rel = (Ji - Ji0).norm(dim=1) / Ji0.norm(dim=1)
...
MK_sh = ((MK - MK0).norm(dim=1) / MK0.norm(dim=1))[shared_idxU]
```

Если `Ji0[k] = 0` (строка J для токена k нулевая) или `MK0[k] = 0`, то деление даёт `inf`/`nan` → `max` → `nan` → `float(nan)` → JSON literal. Для Qwen3.5-0.8B trained embeddings/head строки ненулевые (это не zero-init), и `j_raw` даёт `(√D/r) · M · (I − uu^T)`, где `r > 0` и `M` — trained. Но контракт `factor_residual` формально допускает любой `W`, включая нулевые строки — стоит либо `clamp_min(1e-12)` в знаменателе, либо явный `isfinite` guard.

## Проверка математики

- `j_raw(W, h, gamma) = (√D/r) · M · (I − uu^T)`, `M = W·γ` — соответствует `[M-1]` с `γ` (out-dim scaling, `gamma[None,:]`). ✓
- `factor_residual(W, h, gamma)` — сравнение `W @ J_norm` с closed form; `Jnorm = (√D/r) · P · γ[:,None]` (`P` — in-dim projector). Согласовано с `j_raw` (там `M = W·γ`, `M(I−uu^T)` = `W·γ·P`; здесь `W · (γ[:,None]·P)` — то же). ✓
- `MK = W_ed · γ[None,:]` — та же формула. ✓
- `J_rel_drift_shared` — max по 13 shared токенам, по 16 позициям, `torch.stack(...).max()` — даёт max по 16×13 = 208 значениям. ✓
- `MK_rel_drift_shared` — `[13]`, `.max()` / `.median()`. ✓

Всё корректно.

## Соответствие ТЗ

| Пункт | Реализация | Статус |
|---|---|---|
| Apply E-16 ΔW snapshots | `snaps_all[k]` (P0 fix) | ✅ |
| Check factorization `J = c M_K (I−ûû^T)` token-by-token on U128 | `factor_residual` по 16 сэмплам | ✅ (тривиально) |
| «h_t invariance FALSE for tied model» | `structural_fact` в JSON | ✅ |
| «Verifies algebraic identity, not preservation» | `verdict` + `identity` | ✅ |
| «Leaf drift is informative» | `J_rel_drift_*`, `MK_rel_drift_*` | ✅ |
| Shared-only drift (13/128) | `*_shared` поля | ✅ |
| 13 rows shared, 115 invariant | `shared_rows_256_U128` в `data` | ✅ |
| `s=0` baseline | присутствует, uninformative | ⚠️ P3 |
| Fp64 precision | весь `torch.float64` | ✅ |

## Итоговый приоритетный список

**P1 — исправить до запуска exp38h/exp38i:**
1. `load_snapshots` assert в `exp38_hi_common.py` не работает: формула `‖s @ Wo.T‖_F / (‖s‖ ‖Wo‖)` даёт ≈0.03–0.06 в обоих случаях (правильный ΔW и баг), `0.5` не достигается. Заменить на `‖s‖/‖Wo‖ < 0.5` или на Frobenius inner product. Без этого `exp38h`/`exp38i` не получают защиту от повторения P0.

**P2 — перед публикацией JSON:**
2. `verdict_rule` — заменить «preserved» на «identity holds on edited W rows (tautological)», чтобы не противоречить полю `verdict`.
3. `W_orig256` в `data` — удалить / переименовать в `snapshots`.
4. Добавить `_sanitize` + `allow_nan=False` — консистентность с остальными E-38 и защита от NaN literal (при нулевых строках в `J0`/`MK0`).

**P3:**
5. `s=0` в `out_states` — либо исключить, либо задокументировать как baseline.
6. `clamp_min` в знаменателях `Ji0.norm(dim=1)` / `MK0.norm(dim=1)`, или `isfinite` guard.

## Общий вывод

**P0 (двойное вычитание) из прошлого ревью исправлен корректно.** Новых багов нет. Осталось: (a) один унаследованный из shared P1 — assert, который не выполняет роль, для которой написан (влияет на `exp38h` и `exp38i` одинаково); (b) два P2 — `verdict_rule` противоречит вердикту и `W_orig256` устарел; (c) NaN-контракт (`_json_default`); (d) мелкие P3.

При исправлении п.1 (assert) exp38h и exp38i получают реальную защиту от повторения P0-класса ошибок; без него `load_snapshots` комментирует схему, но не проверяет её. Сами результаты `exp38i` (после P0-фикса) уже семантически корректны — «identity holds, tautological, drift carried by `*_shared`» — и готовы к публикации после устранения P2-противоречия в `verdict_rule`.