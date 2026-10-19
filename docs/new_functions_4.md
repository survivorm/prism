# ТЗ на `prism/pipeline.py` — финальный инфраструктурный модуль

**Область:** единая спецификация эксперимента (`ExperimentSpec`) и thin-driver (`run_spec`), который оркестрирует запуск. **Не содержит логики экспериментов** — только валидация, окружение, seeds, кэш путей, meta.json, обработка исключений. Конкретные эксперименты (`exp_T0.py`, `exp_eps.py`, `exp_T2.py`, `exp_T1.py`, `exp_delta.py`) реализуют `Callable[[ExperimentSpec], dict]` и вызываются через `run_spec`.

**Ключевые решения (зафиксированы):**

1. **`run_spec` не загружает модель.** Модель — ответственность `exp_fn`. `run_spec` только записывает `model_name` в meta.
2. **`ExperimentSpec` — frozen dataclass, но `config: Mapping` и `to_key()` для hashability.** `dataclass(frozen=True)` с `dict` не hashable; `config` хранится как `MappingProxyType`, а `to_key()` возвращает SHA-256 от сериализованного spec (без `cache_root`/`output_root`).
3. **`run_spec` идемпотентен при `skip_if_exists=True`.** Если `meta.json` уже есть и spec-ключ совпадает — возвращает загруженные артефакты без повторного запуска.
4. **`run_spec` **не** устанавливает `deterministic_mode(strict=True)` по умолчанию.** `strict=True` может ломать GDN; `strict=False` (только seeds) — безопасно. `strict` включается явно через `spec.config["strict"] = True` (для non-GDN сценариев).
5. **Все NaN/Inf в JSON сохраняются как строки `"nan"`/`"inf"`/`"-inf"`.** Стандартный `json.dump` пишет `NaN`/`Infinity`, что не валидный JSON; кастомный encoder.
6. **`run_spec` возвращает `OutputArtifacts`**, а не голый dict — чтобы caller имел доступ к `path`, `meta`, `spec_key` без повторного чтения с диска.

---

## 1. Публичный API

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str                     # "T0" | "eps" | "T2" | "T1" | "delta"
    edit: EditSpec
    layers: tuple[int, ...]
    token_set: tuple[int, ...]
    n_docs: int
    S: int
    r: int = 16
    n_theta: int = 16
    n_steps: int = 20
    epsilon_ratio: float = 0.1
    B_perm: int = 1000
    B_boot: int = 1000
    k_sigma: float = 5.0
    alpha: float = 0.05
    seed: int = 0
    cache_root: str = "results/.cache"
    output_root: str = "results"
    model_name: str = "Qwen/Qwen3.5-0.8B"
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def to_key(self) -> str: ...        # SHA-256 hex (12 chars) of serialized spec, excluding paths
    def to_meta_dict(self) -> dict: ... # JSON-serializable, excluding config values that aren't JSON


@dataclass(frozen=True)
class OutputArtifacts:
    spec: ExperimentSpec
    results: dict                       # experiment-specific results (JSON-serializable)
    meta: dict                          # auto-populated environment + timing + versions
    path: str                           # where meta.json was saved
    spec_key: str                        # == spec.to_key()
    reused: bool                         # True if loaded from disk, not recomputed


def run_spec(
    spec: ExperimentSpec,
    exp_fn: Callable[[ExperimentSpec], dict],
    output_subdir: str | None = None,
    skip_if_exists: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> OutputArtifacts: ...


def run_many(
    specs: Sequence[ExperimentSpec],
    exp_fn: Callable[[ExperimentSpec], dict],
    output_subdir_fn: Callable[[ExperimentSpec], str] | None = None,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> list[OutputArtifacts]: ...


def validate_spec(spec: ExperimentSpec) -> None: ...      # raises ValueError on invalid
def load_artifacts(path: str) -> OutputArtifacts: ...     # inverse of save
def default_output_subdir(spec: ExperimentSpec) -> str: ...  # "{edit}/{L?}-{L?}_r{r}_seed{seed}"
```

---

## 2. `ExperimentSpec` — контракт

### 2.1. Поля

| Поле | Тип | Default | Валидация |
|---|---|---|---|
| `experiment` | `str` | (required) | ∈ `{"T0", "eps", "T2", "T1", "delta"}` |
| `edit` | `EditSpec` | (required) | `edit.token_set == spec.token_set` (как `tuple`) |
| `layers` | `tuple[int, ...]` | (required) | Все `0 ≤ l < n_layers` (n_layers из `model_name`; проверка при `validate_spec`, если модель загружена — иначе доверяем caller'у) |
| `token_set` | `tuple[int, ...]` | (required) | `len == edit.delta_W.shape[0]` (если `delta_W is not None`); `len ≥ 2` |
| `n_docs` | `int` | (required) | `≥ 1` |
| `S` | `int` | (required) | `≥ 2` |
| `r` | `int` | `16` | `≥ 1`; `≤ min(len(token_set), D_S)` — проверка при `validate_spec`, если `D_S` известен (иначе caller отвечает) |
| `n_theta` | `int` | `16` | `≥ 4` (для T2'); для остальных — игнорируется |
| `n_steps` | `int` | `20` | `≥ 2` (для T2'); для остальных — игнорируется |
| `epsilon_ratio` | `float` | `0.1` | `> 0` (для T2'); для остальных — игнорируется |
| `B_perm` | `int` | `1000` | `≥ 100` |
| `B_boot` | `int` | `1000` | `≥ 100` |
| `k_sigma` | `float` | `5.0` | `> 0` |
| `alpha` | `float` | `0.05` | `∈ (0, 1)` |
| `seed` | `int` | `0` | `≥ 0` |
| `cache_root` | `str` | `"results/.cache"` | не пустой |
| `output_root` | `str` | `"results"` | не пустой |
| `model_name` | `str` | `"Qwen/Qwen3.5-0.8B"` | не пустой |
| `config` | `Mapping[str, Any]` | `{}` | JSON-serializable values (для meta) |

### 2.2. `to_key()`

| Аспект | Значение |
|---|---|
| Цель | Стабильный идентификатор spec для кэширования `meta.json` |
| Алгоритм | `json.dumps(spec.to_meta_dict(), sort_keys=True)` → `hashlib.sha256(...).hexdigest()[:12]` |
| **Исключается** | `cache_root`, `output_root` (пути не влияют на результаты) |
| **Включается** | `experiment`, `edit.name`, `edit.token_set`, `edit.norm_frobenius` (для контроля), `layers`, `token_set`, `n_docs`, `S`, `r`, `n_theta`, `n_steps`, `epsilon_ratio`, `B_perm`, `B_boot`, `k_sigma`, `alpha`, `seed`, `model_name`, `config` |
| Коллизии | 12 hex chars = 48 bits; при `B=1000` spec-ов в серии — коллизия невозможна; полный хеш — в meta для верификации |
| Детерминизм | `sort_keys=True`; `config` сериализуется как `dict(sorted(...))` |

### 2.3. `to_meta_dict()`

Возвращает JSON-serializable `dict` без `cache_root`/`output_root`. `config` сериализуется как `{str(k): _json_safe(v) for k, v in sorted(config.items())}`, где `_json_safe` конвертирует `np.ndarray` → `list`, `torch.Tensor` → `list` (с `.detach().cpu().tolist()`), `float('nan')` → `"nan"`, `float('inf')` → `"inf"`, `-inf` → `"-inf"`.

**Не бросает** при несериализуемых значениях в `config`; вместо этого пишет `"<unserializable: {type}>"` — это для диагностики, а не для строгой валидации.

---

## 3. `run_spec` — контракт

### 3.1. Сигнатура и семантика

| Аспект | Значение |
|---|---|
| Вход | `spec: ExperimentSpec`, `exp_fn: Callable[[ExperimentSpec], dict]`, `output_subdir: str \| None = None`, `skip_if_exists: bool = True`, `dry_run: bool = False`, `verbose: bool = True` |
| Выход | `OutputArtifacts` |
| Путь сохранения | `{output_root}/{experiment}/{output_subdir or default_output_subdir(spec)}/meta.json` + `results.json` |
| `default_output_subdir(spec)` | `f"{edit.name}/L{min(layers):02d}-{max(layers):02d}_r{r}_seed{seed}"` |
| `exp_fn` ответственность | Вся экспериментальная логика; возвращает `dict` с результатами (JSON-serializable; `np.ndarray`/`torch.Tensor` конвертируются `_json_safe` при сохранении); **не пишет на диск** — `run_spec` пишет |
| `skip_if_exists=True` | Если `{path}/meta.json` существует **и** `meta["spec_key"] == spec.to_key()` **и** `meta["complete"] == True` — загрузить и вернуть `OutputArtifacts(reused=True)`. Иначе — перезапуск |
| `skip_if_exists=False` | Всегда перезапуск (перезапись) |
| `dry_run=True` | Валидирует spec, печатает план (`spec.to_meta_dict()`, путь), **не** вызывает `exp_fn`, возвращает `OutputArtifacts` с `results={}`, `meta={...}`, `path=""`, `reused=False` |
| `verbose=True` | Печать `[run_spec] experiment={...} key={...} path={...}` в начале и `[run_spec] done in {duration:.1f}s` в конце |
| Seeds | `with deterministic_mode(seed=spec.seed, strict=spec.config.get("strict", False)):` — обёртка вокруг `exp_fn`. **Не** `strict=True` по умолчанию (ломает GDN) |
| Исключение в `exp_fn` | `try/except`: сохранить `meta["complete"] = False`, `meta["error"] = str(exc)`; **перебросить** исключение после сохранения meta (caller видит ошибку, но диагностика на диске остаётся) |
| `KeyboardInterrupt` | Сохранить `meta["complete"] = False`, `meta["error"] = "KeyboardInterrupt"`; **перебросить** (не глотать Ctrl-C) |
| Атомарность | `tmp = path + ".tmp"; json.dump; os.replace(tmp, path)` — прерванная запись не оставляет полу-файл |
| Кодировка | `utf-8`; `ensure_ascii=False` (для читаемости; ключи — ASCII, значения могут быть Unicode) |

### 3.2. Meta.json (автозаполнение)

`run_spec` **добавляет** в meta (перезаписывает одноимённые ключи, если `exp_fn` их не вернул):

```json
{
  "spec": {...},                       // to_meta_dict()
  "spec_key": "...",                   // to_key()
  "spec_key_full": "...",              // полный SHA-256 (для верификации)
  "experiment": "...",
  "edit_name": "...",
  "env": {
    "torch": "...",
    "transformers": "...",
    "numpy": "...",
    "python": "...",
    "device": "cpu" | "cuda:0" | "mps",
    "git_commit": "..." | "unknown",
    "timestamp_start": "ISO8601",
    "timestamp_end": "ISO8601",
    "duration_sec": 12.3
  },
  "complete": true,
  "error": null,
  "reused": false
}
```

- `device`: определяется `torch.cuda.is_available()`, `torch.backends.mps.is_available()`, иначе `"cpu"`. **Не** device модели (это в `exp_fn`).
- `git_commit`: `subprocess.run(["git", "rev-parse", "HEAD"], ...)`; при ошибке — `"unknown"` (не падать).
- `timestamp_start`/`timestamp_end`: `datetime.now(timezone.utc).isoformat()`.
- `duration_sec`: `time.perf_counter()` разница.
- `complete`: `False` при исключении, `True` при успехе.

### 3.3. Results.json

`exp_fn` возвращает `dict`. `run_spec` конвертирует через `_json_safe` рекурсивно и сохраняет в `{path}/results.json`. Если `exp_fn` вернул не dict — `TypeError` (до записи meta; meta с `complete=False`, `error="expected dict from exp_fn"`).

**`_json_safe(obj)`** рекурсивно:
- `np.ndarray` → `obj.tolist()` (с конверсией NaN/Inf)
- `torch.Tensor` → `obj.detach().cpu().tolist()` (с конверсией)
- `float('nan')` → `"nan"`, `float('inf')` → `"inf"`, `float('-inf')` → `"-inf"`
- `np.float32/64` → `float(obj)`
- `np.int32/64` → `int(obj)`
- `dict` → рекурсивно; ключи — `str(k)`
- `list`/`tuple` → рекурсивно
- `None`, `bool`, `int`, `str`, `float` — как есть
- иначе → `f"<unserializable: {type(obj).__name__}>"`

### 3.4. `load_artifacts(path)`

| Аспект | Значение |
|---|---|
| Вход | `path: str` — путь к `meta.json` (или директории, содержащей `meta.json`) |
| Выход | `OutputArtifacts` с `results` из `results.json` (если есть), `meta` из `meta.json`, `path`, `spec_key`, `reused=True` |
| `spec` | Реконструируется из `meta["spec"]`; `edit` реконструируется как `EditSpec` (delta_W отсутствует в meta по соображениям размера — только `edit.name`, `norm_frobenius`, `projection_onto_top_r`). **`delta_W` в реконструированном `EditSpec` — `None`.** Это ограничение: `load_artifacts` не восстанавливает правку, только метаданные. Документировать |
| `results.json` отсутствует | `results = {}`; `meta["results_available"] = False` |
| Битый JSON | `RuntimeError` с указанием пути |

### 3.5. `run_many`

| Аспект | Значение |
|---|---|
| Вход | `specs: Sequence[ExperimentSpec]`, `exp_fn`, `output_subdir_fn: Callable[[ExperimentSpec], str] \| None`, `skip_if_exists`, `verbose` |
| Действие | Для каждого `spec`: `run_spec(spec, exp_fn, output_subdir=output_subdir_fn(spec) if output_subdir_fn else None, skip_if_exists=skip_if_exists, verbose=verbose)` |
| Исключение в одном spec | Прервать batch (не глотать); caller может обернуть в `try/except` для partial batch |
| Возврат | `list[OutputArtifacts]` |

**Зачем `run_many`:** для T0 (5 edits), ε (5 edits × metrics), T2' (3 edits). Не критичен, но экономит boilerplate. `output_subdir_fn` позволяет разделить результаты по edit name, если разные `exp_fn` вызываются для разных edits.

---

## 4. Внутренние детали реализации

### 4.1. Приватные функции

- **`_validate_spec(spec)`** — вызывается `run_spec` в начале; реализует все проверки из §2.1. При ошибке — `ValueError` с указанием поля.
- **`_ensure_output_dir(path)`** — `os.makedirs(path, exist_ok=True)`; при `PermissionError` (Kaggle FUSE read-only) — `RuntimeError` с явным сообщением.
- **`_write_json_atomic(path, obj)`** — `tmp = path + f".tmp.{os.getpid()}.{uuid4().hex[:8]}"`; `json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_safe)`; `os.replace(tmp, path)`.
- **`_read_json(path)`** — `json.load`; проверка `complete` (если `False` — `RuntimeError`).
- **`_detect_env()`** — возвращает `dict` с `torch`, `transformers`, `numpy`, `python`, `device`, `git_commit`; не бросает.
- **`_json_safe(obj)`** — рекурсивная конверсия (см. §3.3).
- **`_load_or_reuse(meta_path, spec_key)`** — если `meta_path` существует и `meta["spec_key"] == spec_key` и `meta["complete"]` — вернуть `OutputArtifacts(reused=True)`, иначе `None`.
- **`_expected_output_path(spec, output_subdir)`** — `os.path.join(spec.output_root, spec.experiment, output_subdir or default_output_subdir(spec))`.
- **`default_output_subdir(spec)`** — `f"{spec.edit.name}/L{min(spec.layers):02d}-{max(spec.layers):02d}_r{spec.r}_seed{spec.seed}"`. Публичная, потому что caller может использовать для поиска.
- **`_print_plan(spec, output_path)`** — для `dry_run`/`verbose`.

### 4.2. Импорты

- `import json`, `import os`, `import hashlib`, `import subprocess`, `import time`, `import uuid`
- `from collections.abc import Callable, Mapping, Sequence`
- `from dataclasses import dataclass, field`
- `from datetime import datetime, timezone`
- `from types import MappingProxyType`
- `from typing import Any`
- `import numpy as np`, `import torch` (для `_json_safe` и `_detect_env`; transformers — ленивый `try/except` в `_detect_env`)
- **`from .interventions import EditSpec`** — реконструкция в `load_artifacts`
- **`from .cache import deterministic_mode`** — обёртка seeds

**Никаких `from .spectral`, `.holonomy`, `.noise`, `.permutation`, `.topology`** — `pipeline.py` не знает об экспериментах.

### 4.3. Порядок выполнения `run_spec`

1. `_validate_spec(spec)` — бросить `ValueError` при ошибке.
2. `output_path = _expected_output_path(spec, output_subdir)`.
3. Если `dry_run`: `_print_plan(spec, output_path)`; вернуть `OutputArtifacts(spec, {}, {}, "", spec.to_key(), False)`.
4. `_ensure_output_dir(output_path)`.
5. Если `skip_if_exists`: `existing = _load_or_reuse(f"{output_path}/meta.json", spec.to_key())`; если не None — вернуть `existing`.
6. `meta = {"spec": spec.to_meta_dict(), "spec_key": spec.to_key(), "spec_key_full": _full_hash(spec), "experiment": spec.experiment, "edit_name": spec.edit.name, "env": _detect_env() | {"timestamp_start": now_iso()}, "complete": False, "error": None}`.
7. `t0 = time.perf_counter()`.
8. `try: with deterministic_mode(seed=spec.seed, strict=spec.config.get("strict", False)): results = exp_fn(spec)`.
9. `except (Exception, KeyboardInterrupt) as e: meta["error"] = f"{type(e).__name__}: {e}"; meta["env"]["duration_sec"] = time.perf_counter() - t0; _write_json_atomic(f"{output_path}/meta.json", _json_safe(meta)); raise`.
10. `meta["complete"] = True`; `meta["env"]["timestamp_end"] = now_iso()`; `meta["env"]["duration_sec"] = time.perf_counter() - t0`; `_write_json_atomic(f"{output_path}/results.json", _json_safe(results))`; `_write_json_atomic(f"{output_path}/meta.json", _json_safe(meta))`.
11. `if verbose: print("[run_spec] done in ...")`.
12. `return OutputArtifacts(spec, results, meta, f"{output_path}/meta.json", spec.to_key(), False)`.

**Почему meta пишется дважды (до и после `exp_fn`):** первый раз — чтобы при падении/`Ctrl-C` была диагностика (`complete=False`); второй — с финальным `complete=True` и временем. Атомарность гарантирует, что промежуточное состояние не читается как финальное.

---

## 5. Взаимодействие с существующим

| Модуль | Использование |
|---|---|
| `interventions.EditSpec` | Импорт для реконструкции в `load_artifacts`; `spec.edit` — ссылка на существующий `EditSpec` |
| `cache.deterministic_mode` | Обёртка seeds; `strict` из `spec.config` |
| `bridge`, `esg`, `jacobian`, `calibration`, `diagnostics`, `layer_selection`, `cache_io` | **Не импортируются** — `pipeline.py` их не знает |
| `spectral`, `holonomy`, `noise`, `permutation`, `topology` | **Не импортируются** — используются внутри `exp_fn`, а не в `pipeline.py` |
| `paper.md` §19.A | Ссылка на spec-структуру в docstring |
| `experiments/exp_T0.py` … `exp_delta.py` | Вызывают `run_spec`; каждая реализует `exp_fn: Callable[[ExperimentSpec], dict]` |

**Ключевое:** `pipeline.py` — самый «тонкий» модуль; если он импортирует что-то из экспериментов — это архитектурная ошибка.

---

## 6. Selfcheck (`tests/test_pipeline.py`)

1. **`validate_spec` valid:** собрать spec из реальных `EditSpec` (`make_base`), валидные `layers`, `token_set`; `validate_spec(spec)` — не бросает.
2. **`validate_spec` invalid experiment:** `experiment="unknown"` → `ValueError` с «experiment».
3. **`validate_spec` edit.token_set mismatch:** `edit.token_set = (1,2)`, `spec.token_set = (1,2,3)` → `ValueError`.
4. **`validate_spec` layers out of range:** `layers=(100,)` с моделью 24 слоя (если `n_layers` известен из `model_name` — hardcoded проверка; иначе caller отвечает — **уточнить в ТЗ**: `pipeline.py` **не** знает `n_layers`; проверка только `l ≥ 0` и `len ≥ 1`. Полная проверка `l < n_layers` — в `exp_fn`). **Исправление:** `validate_spec` проверяет `0 ≤ l` и `len(layers) ≥ 1`; `n_layers` — ответственность `exp_fn` (модель загружена только там).
5. **`validate_spec` r > K:** `r > len(token_set)` → `ValueError`.
6. **`validate_spec` alpha out of range:** `alpha=0` или `1` → `ValueError`.
7. **`to_key` determinism:** два spec с одинаковыми полями → одинаковый `to_key`.
8. **`to_key` sensitivity:** `seed=0` vs `seed=1` → разные keys.
9. **`to_key` path-insensitive:** `cache_root="a"` vs `"b"` → одинаковый key.
10. **`to_key` config-sensitive:** `config={"x": 1}` vs `{"x": 2}` → разные keys.
11. **`run_spec` basic:** dummy `exp_fn = lambda spec: {"x": 42}`; `run_spec(spec, exp_fn, output_root=tmpdir)`; `assert artifacts.results == {"x": 42}`; `assert os.path.exists(f"{tmpdir}/{spec.experiment}/{default_output_subdir(spec)}/meta.json")`; `assert artifacts.meta["complete"] == True`.
12. **`run_spec` skip_if_exists:** второй вызов с тем же spec → `artifacts.reused == True`; `exp_fn` **не** вызван (проверить через counter).
13. **`run_spec` skip_if_exists=False:** второй вызов → `reused == False`; `exp_fn` вызван дважды.
14. **`run_spec` dry_run:** `exp_fn` **не** вызван; `artifacts.results == {}`; `artifacts.path == ""`.
15. **`run_spec` exp_fn returns non-dict:** `exp_fn = lambda spec: 42` → `TypeError`; `meta["complete"] == False`; `meta["error"]` содержит «expected dict».
16. **`run_spec` exp_fn raises:** `exp_fn = lambda spec: (_ for _ in ()).throw(ValueError("boom"))`; `pytest.raises(ValueError)`; `meta["complete"] == False`; `meta["error"]` содержит «boom»; файл `meta.json` существует.
17. **`run_spec` NaN/Inf в results:** `exp_fn = lambda spec: {"x": float('nan'), "y": float('inf')}`; `assert artifacts.results["x"] == "nan"` (после `_json_safe`); `assert artifacts.results["y"] == "inf"`; JSON-файл валиден (можно прочитать стандартным `json.load`).
18. **`run_spec` np.ndarray/torch.Tensor в results:** `exp_fn = lambda spec: {"arr": np.array([1, 2]), "t": torch.tensor([3.0, 4.0])}`; результат — списки.
19. **`run_spec` seeds determinism:** `exp_fn = lambda spec: {"r": float(torch.randn(1))}`; два запуска с `seed=0` → одинаковый `r`; с `seed=1` → разный.
20. **`run_spec` strict=True:** `spec.config = MappingProxyType({"strict": True})`; `exp_fn = lambda spec: {"r": float(torch.randn(1))}`; работает (или `RuntimeError` на GDN — но dummy `exp_fn` не использует модель, значит OK).
21. **`load_artifacts` round-trip:** `run_spec` → `load_artifacts(artifacts.path)` → `assert loaded.results == artifacts.results`; `assert loaded.meta["spec_key"] == artifacts.spec_key`; `assert loaded.reused == True`.
22. **`run_many`:** 3 specs; `run_many(specs, exp_fn, output_root=tmpdir)` → 3 artifacts; все файлы на диске.
23. **`run_many` с output_subdir_fn:** `output_subdir_fn = lambda spec: spec.edit.name`; файлы в разных поддиректориях.
24. **`_json_safe` recursive:** `_json_safe({"a": [np.array([1, float('nan')])], "b": {"c": torch.tensor([2.0])}})` → `{"a": [[1, "nan"]], "b": {"c": [2.0]}}`.
25. **`default_output_subdir`:** `spec.layers = (6, 12, 18)`; `default_output_subdir(spec)` == `"base/L06-18_r16_seed0"` (для `edit.name="base"`, `r=16`, `seed=0`).

---

## 7. Ограничения и риски

- **`run_spec` не загружает модель** — `model_name` в spec только для meta. Если `exp_fn` использует другую модель, meta будет неверным (это ответственность `exp_fn`).
- **`load_artifacts` не реконструирует `edit.delta_W`** — в meta хранятся только `edit.name`, `norm_frobenius`, `projection_onto_top_r`. Для повторного применения edit нужен явный `EditSpec` от caller'а. Задокументировать.
- **`skip_if_exists=True` требует, чтобы `meta.json` имел `spec_key` и `complete=True`** — старые meta (до v6.0) не пройдут (нет поля `spec_key`); они будут перезаписаны. Это ожидаемое поведение.
- **`deterministic_mode` не гарантирует побитовую воспроизводимость GDN** — только seeds. Для GDN-побитовости нужна `_clone_gdn_states_on_entry` + `torch.use_deterministic_algorithms(True)` внутри `exp_fn`, что может ломать GDN. `noise.py` измеряет остаточную вариантность.
- **`dry_run` не валидирует `config`** — `config` может содержать несериализуемые значения; `dry_run` их не трогает (только печатает `spec.to_meta_dict()`, где `_json_safe` уже применён).
- **`run_many` при исключении в одном spec прерывает batch** — caller может обернуть в `try/except` для partial. `run_many` не пытается продолжить.
- **`meta.json` и `results.json` могут быть большими** — `results` содержит все per-layer/per-doc значения. Для T0 (5 edits × 6 layers × 32 docs) — единицы MB; приемлемо. Для больших N — caller должен агрегировать в `exp_fn` до возврата (медианы, CI, а не сырые значения).

---

## 8. Что НЕ входит в `pipeline.py`

- Загрузка модели — `exp_fn`.
- Реализация экспериментов — `exp_T0.py` … `exp_delta.py`.
- CLI / argparse — `experiments/run.py` (или `__main__.py`) поверх `run_spec`.
- Параллельный запуск нескольких spec — caller (например, `concurrent.futures`); `pipeline.py` — синхронный.
- Визуализация — `exp_*` или отдельные `plot_*.py`.
- Кэширование **результатов** `exp_fn` (не только meta) — caller через `run_spec` с `skip_if_exists` (что и есть кэш по spec-key).
- Cross-experiment orchestration (T0 → ε → T2') — `experiments/run_all.py` поверх `run_many`.

---

## 9. Сводный чек-лист

| Пункт | Проверка |
|---|---|
| `ExperimentSpec` frozen, `to_key()` стабилен и path-insensitive | selfcheck §6.7–6.10 |
| `validate_spec` ловит все невалидные поля | selfcheck §6.1–6.6 |
| `run_spec` basic, skip/reuse, dry_run | selfcheck §6.11–6.14 |
| `run_spec` обрабатывает исключения (meta с error, re-raise) | selfcheck §6.15–6.16 |
| `run_spec` сериализует NaN/Inf/ndarray/Tensor | selfcheck §6.17–6.18 |
| `run_spec` seeds determinism | selfcheck §6.19–6.20 |
| `load_artifacts` round-trip | selfcheck §6.21 |
| `run_many` batch | selfcheck §6.22–6.23 |
| `_json_safe` recursive | selfcheck §6.24 |
| `default_output_subdir` | selfcheck §6.25 |
| `pipeline.py` **не** импортирует `spectral`, `holonomy`, `noise`, `permutation`, `topology` | grep: нет `from .spectral` и т.д. |
| `pipeline.py` импортирует `interventions.EditSpec`, `cache.deterministic_mode` | grep: есть оба |
| `exp_*` — тонкие обёртки, `run_spec(spec, exp_fn)` | ручная проверка |
| `experiments/run.py` — CLI поверх `run_many` (опционально) | не входит в критерий |

**Критерий готовности:**
- Все selfcheck §6 проходят.
- `git diff` по `bridge.py`, `esg.py`, `jacobian.py`, `calibration.py`, `diagnostics.py`, `layer_selection.py`, `cache_io.py`, `spectral.py`, `cache.py`, `holonomy.py`, `interventions.py`, `noise.py`, `permutation.py`, `topology.py` — пустой (кроме уже согласованных правок).
- `import prism.pipeline` работает без `datasets`, `giotto-tda`, `dionysus`.
- `python -c "from prism import ExperimentSpec, run_spec, OutputArtifacts"` работает (экспорты в `__init__.py` добавлены).

---

## 10. Порядок интеграции (после реализации)

1. Добавить экспорты `ExperimentSpec`, `OutputArtifacts`, `run_spec`, `run_many`, `validate_spec`, `load_artifacts`, `default_output_subdir` в `prism/__init__.py` (не в `try/except` — `pipeline.py` не зависит от `datasets`).
2. `experiments/exp_T0.py`: реализует `run(spec: ExperimentSpec) -> dict` (создание 5 `EditSpec`, кэширование `h_t`/`p_t` через `apply_edit_and_cache`, GDN-политика через `noise` + `validate_signal`); `if __name__ == "__main__": run_spec(spec, run)`.
3. `experiments/exp_eps.py`: `run(spec) -> dict` (для каждого edit: `spectral_profile` через `cross_positional_gramian` + `eigengap` + `bootstrap_ci`, `permutation_max_statistic`).
4. `experiments/exp_T2.py`: `run(spec) -> dict` (`parameter_space_holonomy` через `holonomy_along_path`; `representation_holonomy_sens/canon` через `holonomy_along_2d_loop`; сравнение с `σ_GDN`).
5. `experiments/exp_T1.py`: `run(spec) -> dict` (`project_to_subspace`/`project_to_pca_subspace` + `zigzag_persistence`; `compare_tail_vs_pca`).
6. `experiments/exp_delta.py`: `run(spec) -> dict` (4D-портрет через `spectral` + `diagnostics`).

**Каждый `exp_*.py` — ≤100 строк**, только композиция `prism.*`. Если `exp_*.py` содержит >200 строк или свою математику — сигнал, что `prism/` недодекомпозирован.

---

*Финальная часть ТЗ на инфраструктуру. После реализации всех модулей (`spectral`, `cache`, `holonomy`, `interventions`, `noise`, `permutation`, `topology`, `pipeline`) — серия `experiments/exp_T0.py` … `exp_delta.py` пишется в 1–2 дня, потому что вся математика и инфраструктура уже покрыты selfcheck'ами и контрактами.*