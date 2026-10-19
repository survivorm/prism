# Проверка адекватности и соответствия ТЗ

## Итоговый вердикт

Реализация **в основном адекватна и соответствует ТЗ**, но есть **один существенный риск** (формула `snr_e14` не валидирована против E-14), **одно задокументированное отклонение** от критерия готовности (`import prism.spectral` требует `datasets`) и несколько **мелких несоответствий/недочётов** (device в `gauge_share`, экспорт `snr_e14_parts`, TZ-ошибки в selfcheck/пути).

---

## 1. `spectral.py` — соответствие ТЗ §1

| Функция | ТЗ | Реализация | Статус |
|---|---|---|---|
| `top_r_svd` | fp16/bf16 → float32, r-валидация, sign-caveat | ✅ `.float()` при `_LOW_PREC`, `1 ≤ r ≤ min(K,D)`, docstring про LAPACK-знаки | ✅ |
| `spectrum_from_gramian` | `eigvalsh` в float64, симметрия `atol=1e-6`, ascending | ✅ `eigvalsh(G.double())`, проверка симметрии, но добавлен `rtol=1e-6` | ⚠️ rtol расширяет допуск (для больших λ может пропустить асимметрию — TZ просил только atol) |
| `eigengap` | K ≥ 2, λ₁−λ₂, любой порядок | ✅ | ✅ |
| `entropy_effective_rank` | exp(−Σ p log p), negatives dropped, all≤0→0.0+warn | ✅ фильтр `spectrum>0`, warn+0.0 | ✅ |
| `stable_rank` | Σλ/λ₁, λ₁=0→0.0+warn | ✅ | ✅ + **бонус:** в docstring корректно исправлена ошибка ТЗ (`PR` для `(4,1,1,1,1)` = 64/**20** = 3.2, а не 64/18≈3.555) |
| `participation_ratio` | (Σλ)²/Σλ², negatives не отбрасываются | ✅ | ✅ |
| `snr_top1_vs_rest` | λ₁/mean(λ₂..), λ₁≤0 — не оговорено, K=1→ValueError, mean=0→inf | ✅ + добавлен `ValueError` при λ₁≤0 | ✅ (доп. проверка не противоречит ТЗ) |
| `snr_e14` / `snr_e14_parts` | «найти оригинальную формулу, задокументировать» | Формула `‖mean J‖²_F / (E‖J‖²_F − ‖mean J‖²_F)` — sample-based, docstring честно маркирует как «original E-14» | ⚠️ **не валидировано против E-14** (0.14→3.65, L23=199.3) — риск для downstream |
| `gauge_share` | Σ(u_kᵀu)²λ/Σλ, uniform L2, default 1/√K, shape (K,K) columns | ✅, docstring про `esg`-конвенцию (ROWS → передавать `eig.T`) | ⚠️ **device**: `uniform = torch.full((K,), ...)` создаётся на CPU; для GPU `eigenvectors` — падение (см. ниже) |
| `gauge_share_top1` | λ₁(u₁ᵀu)²/Σλ, `spectrum[0]` должен быть λ_max | ✅, docstring про descending-конвенцию | ⚠️ тот же device-баг |
| `_check_1d_finite` | ТЗ §1.4: NaN/Inf в `eigengap`, `gauge_share` → ValueError с индексом | Применён ко **всем** функциям, сообщает индекс | ✅ (строже ТЗ, чем требуется) |

### ⚠️ Существенная проблема: device в `gauge_share`/`gauge_share_top1`

```python
uniform = torch.full((K,), 1.0 / math.sqrt(K))   # CPU!
coeffs = (eigenvectors.T @ uniform) ** 2         # RuntimeError, если eigenvectors на GPU
```

ТЗ §1.4 требует «Никаких numpy (для консистентности device/dtype)». Для GPU-тензоров эти функции упадут. Фикс — одна строка: `uniform = torch.full((K,), 1/math.sqrt(K), device=eigenvectors.device, dtype=eigenvectors.dtype)`.

### ⚠️ `snr_e14` — открытый риск (не баг, но требует валидации)

Комментарий в коде «MOVED here, not copied» заявлен, но: (а) нельзя проверить из предоставленных файлов, был ли реальный перенос; (б) формула не проверена против заявленных в paper чисел (0.14→3.65, аномалия L23=199.3). Если формула не совпадает — E-T0/E-ε не воспроизведут E-14. Это надо пометить в `meta.json` (ТЗ §1.3 «snr_top1_vs_rest» это разрешает) и по возможности сделать прогон на тех же J, что дали paper-числа.

### Мелкие замечания

- **`stable_rank` при `lam1 < 0`**: проверка `if lam1 == 0` пропустит случай `lam1 = -1e-9` (не реально для ESG, но `Δ_U` может дать смешанный спектр) — вернёт огромное отрицательное `r_stable`. Minor.
- **`__init__.py` не экспортирует `snr_e14_parts`** (есть в модуле и его `__all__`, но не в пакетном `__all__`). Не критично: доступ через `prism.spectral.snr_e14_parts`, но несогласованно.

---

## 2. `cache.py` — соответствие ТЗ §2

| Компонент | ТЗ | Реализация | Статус |
|---|---|---|---|
| `SVDCache.key` | SHA-256 строки `edit\|L\|d\|p\|r\|dtype\|v` | ✅ совпадает | ✅ |
| `SVDCache._path` | `.../d{doc:04d}_p{pos:04d}_r{r}_{v}.pt` | ✅ | ✅ |
| `SVDCache.put` валидация | shapes, S descending, dtype, ортонорм (1e-5), r≤min(K,D) | ✅ все проверки на месте | ✅ |
| `SVDCache.get` | version/torch warn, r/dtype/version mismatch → None, битый файл → RuntimeError | ✅ | ✅ |
| Атомарная запись | tmp+`os.replace`, уборка tmp при ошибке | ✅ `_atomic_write` | ✅ |
| `_ensure_root` | использовать `find_targets_dir`, иначе создать | ✅ `from .cache_io import find_targets_dir` | ✅ |
| `MetricCache` ключ | `(metric, edit, layer, doc, version)`, JSON, `doc=None` → `__layer_aggregate__` | ✅ | ✅ |
| `nan/inf` в JSON | строки `"nan"/"inf"/"-inf"` | ✅ `_json_dump_value`/`_json_load_value` | ✅ |
| `bulk_get` исключает aggregate | ✅ (фильтр `startswith("doc_")`) | ✅ |
| `bootstrap_ci` | median default, B≥100, alpha∈(0,1), NaN→ValueError, seed | ✅ | ✅ |
| `bootstrap_ci_paired` | индексы ресэмплируются вместе, длина = | ✅ | ✅ |
| `deterministic_mode` | save/restore torch/np/random + CUDA, strict, nesting | ✅, `try/finally`, восстановление в обратном порядке | ✅ |
| `iter_position_pairs` | adjacent/strided/all, ошибки | ✅ (selfcheck 2.6.15–2.6.16 проходят) | ✅ |

### ⚠️ Несогласованность `dtype` в ключе vs пути `SVDCache` (TZ-проблема)

TZ §2.3:
- Ключ: `(edit, layer, doc, pos, r, **dtype**, version)`
- Путь: `..._r{r}_{version}.pt` — **dtype в пути отсутствует**

Следствия в реализации:
- `has()` проверяет только существование файла (по пути без dtype) → **`has()==True` при несовпадении dtype**, но `get()` вернёт `None` (сверяет `meta["dtype"]`).
- Два инстанса `SVDCache` с разным dtype будут писать в **один файл** и перетирать друг друга.

Реализация **буквально следует ТЗ**, но ТЗ здесь противоречиво (dtype в ключе логически требует dtype в пути). Это TZ-баг, не реализационный; в selfcheck §2.6.4 тестируется только `r`-mismatch (в пути есть), dtype-mismatch не тестируется — баг не всплывает.

### ⚠️ `MetricCache` — «version в ключе → новый файл» не работает (TZ-проблема)

TZ §2.3:
> «`version` в ключе; смена `version` → новый файл»

Но путь: `{root}/{metric}/{edit}/L{layer:02d}.json` — **version в пути отсутствует**. Реализация буквально следует пути и перезаписывает `meta["version"]` при каждом `put`. Значит:
- при смене версии старые документы остаются, meta помечается новой версией → **смешение v1/v2 данных под меткой v2**;
- «новый файл» не создаётся.

Реализация следует ТЗ по пути, но ТЗ внутренне противоречиво. Риск: если пользователь сменит `version` в `root` без очистки — получит mixed-version метрики. Стоит либо добавить `{version}/` в путь `MetricCache`, либо документировать, что `root` должен быть версионно-чистым.

### Мелкие замечания

- **`SVDCache.get` переводит `U,S,V` в `self.dtype`** после загрузки — no-op, если meta-dtype совпал; если файл испорчен несоответствующим dtype, ошибки не будет (молчаливое приведение). Minor.
- **`SVDCache.key` импортирует `hashlib` внутри функции** — работает, но стилистически лучше вынести наверх.
- **`MetricCache._save`** при `nan/inf`: `json.dump` пишет `"nan"/"inf"` как строки — корректно, но `sort_keys=True` сортирует `"doc_N"` лексикографически (`doc_10` < `doc_2`); не влияет на семантику, но для читаемости diff'ов в git стоит `doc_{N:04d}`.
- **`iter_position_pairs(mode="all")`** возвращает генератор — соответствует `Iterable`, ок.
- **`deterministic_mode`**: восстановление идёт в фиксированном порядке (random → np → torch → cuda); если одно из восстановлений бросит, остальные не выполнятся. Крайне маловероятно, но `try/except` вокруг каждого было бы надёжнее. Minor.

---

## 3. `__init__.py` — соответствие критерию готовности

| Критерий ТЗ | Состояние |
|---|---|
| `from .cache_io import find_targets_dir` (не дублируется) | ✅ |
| `esg._spectral_decomposition` не дублируется; различие `eigh K×K` vs `SVD K×D` в docstring | ✅ (явно в шапке `spectral.py`) |
| `bridge.py`, `esg.py`, `jacobian.py`, `calibration.py`, `diagnostics.py`, `layer_selection.py`, `cache_io.py` — не тронуты | ✅ (по предоставленным файлам не видно обратных импортов) |
| `import prism.spectral` / `import prism.cache` **без `datasets`** | ❌ **не выполнено** (см. ниже) |
| `snr_top1_vs_rest` + `snr_e14` в публичном API | ✅ оба экспортированы |

### ❌ Критерий «`import prism.spectral` без `datasets`» не выполнен

`prism/__init__.py` импортирует `calibration`, который на уровне модуля делает `from datasets import load_dataset`. Значит, **любой** `import prism.spectral` (как и `import prism.cache`) сначала выполняет `prism/__init__.py` → требует `datasets`.

В `__init__.py` есть комментарий (audit 2026-07-18), что это **осознанное** решение: «`load_calibration_data` —核心 public API entry point; silent absence → NameError deep in long experiment». То есть это **задокументированное отклонение от ТЗ**, сделанное после написания ТЗ. Формально критерий не выполнен, но отклонение обосновано.

**Что делать:** либо обновить ТЗ (убрать критерий «без datasets»), либо ввести lazy-import `calibration` через `__getattr__` (PEP 562) — тогда `import prism.spectral` не потянет `datasets`. Выбор — за командой; в текущем виде это **риск для unit-тестов `spectral.py`/`cache.py`** в окружениях без `datasets` (например, в CI).

---

## 4. Ошибки в самом ТЗ (реализация их корректно обходит/отмечает)

1. **ТЗ §1.6.5**: `λ = (4,1,1,1,1)`, `assert participation_ratio ≈ 64/18 ≈ 3.555` — **неверно**, `Σλ² = 16+1+1+1+1 = 20`, `PR = 64/20 = 3.2`. Реализация это отмечает в docstring `stable_rank` — ✅ хорошо.
2. **ТЗ §2.3 `SVDCache` путь без dtype** — противоречие с ключом (см. выше).
3. **ТЗ §2.3 `MetricCache` «version → новый файл»** — путь не содержит version (см. выше).
4. **ТЗ §1.3 `gauge_share`**: «`spectrum` и `eigenvectors` должны покрывать все моды» — реализация это проверяет через `shape != (K,K)`, но не проверяет, что `spectrum` — **полный** (не top-r). Если кто-то передаст top-r спектр + полную матрицу векторов — пройдёт, но `Σλ` будет не trace. Стоит добавить `total_trace`-параметр или явную проверку `len(spectrum) == eigenvectors.shape[1] == K`. Minor.

---

## 5. Что нужно исправить (по приоритету)

### P0 — блокирующее для GPU-прогонов
1. **`gauge_share` / `gauge_share_top1`**: `uniform` создавать на `eigenvectors.device` с `dtype=eigenvectors.dtype`.

### P1 — валидация корректности
2. **`snr_e14`**: прогнать против J-кэша E-14 (если доступен) и проверить 0.14→3.65; иначе — пометить в `meta.json` как «candidate, unvalidated» и в downstream использовать `snr_top1_vs_rest` как прокси (ТЗ §1.3 это разрешает).

### P2 — консистентность и CI
3. **`import prism.spectral` без `datasets`**: либо lazy-import `calibration`, либо обновить ТЗ-критерий (в текущем виде CI без `datasets` не сможет импортировать `spectral`/`cache`).
4. **`SVDCache` / `MetricCache`**: добавить `dtype` (для SVD) и `version` (для Metric) в путь — либо явно задокументировать, что `root` должен быть моно-версионным/моно-dtype. Сейчас «`has()`/`get()` рассогласованы при dtype mismatch» и «version mismatch → смешение» — потенциальные баги в проде.

### P3 — косметика
5. `snr_e14_parts` — добавить в `__init__.__all__` (или явно оставить внутренним, но тогда убрать из `spectral.__all__` для консистентности).
6. `spectrum_from_gramian` — `rtol=1e-6` либо убрать, либо задокументировать как «TZ + допуск на большие матрицы».
7. `SVDCache.key` — вынести `import hashlib` наверх.
8. `MetricCache` — форматировать doc-ключи как `doc_{N:04d}` для стабильного git-diff.

---

## 6. Резюме одной строкой

**Реализация адекватна ТЗ и в основном соответствует ему; блокирующих багов два — device в `gauge_share*` (P0, GPU) и невыполненный критерий «`import prism.spectral` без `datasets`» (P2, но осознанно задокументирован); существенный риск один — `snr_e14` не валидирован против E-14 (P1); остальное — мелкие несоответствия и TZ-ошибки, которые реализация корректно обходит/комментирует.**