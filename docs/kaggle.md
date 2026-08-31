# Kaggle Kernel: отладка и деплой

## Структура

```bash
kaggle_kernel/
  build_kernel.py      # билдер одного эксперимента (--exp, --out, --id, --title)
  push_all.py          # пушер (KERNELS dict: dir, slug, id, title, exp_file)
  kernel-exp13/        # per-experiment: kernel.ipynb + kernel-metadata.json
  kernel-exp14/
  kernel-exp15/
  kernel-exp6/
  kernel-exp16s/       # exp16_stage0_targets
  ...
```

```bash
# Build + push одного эксперимента:
uv run python3 kaggle_kernel/push_all.py --exp exp16_stage0

# Build + push всех:
uv run python3 kaggle_kernel/push_all.py --all
```

`build_kernel.py`:

- встраивает `prism.*` через `importlib + inspect.getsource()`
- стрипит `from prism.X import ...` (single-line и multi-line, с `\s*` для индентированных импортов)
- удаляет `argparse` и `if __name__` блоки
- renames `main()` → `main_{exp_name}`
- runner cell: `inspect.signature(main)` → kwargs по сигнатуре

## Отладка

```bash
# Пуш:
uv run python3 kaggle_kernel/push_all.py --exp exp16_stage0

# Статус + логи:
kaggle kernels status <slug>
kaggle kernels logs <slug> | grep -E '\[OK\]|\[FAIL\]|RuntimeError|TypeError' | tail -20
```

## Ошибки и паттерны

### 1. CUDA OOM (4B model, single T4)

**Симптом**: `CUDA out of memory. 14.43 GiB of 14.56 GiB`

**Причина**: fp32 (4B × 4 bytes) + backward ≈ 18GB > 16GB T4.

**Фикс**: fp16 для large model.

```python
dtype_L = torch.float16 if device_L.startswith("cuda") else torch.float32
mL = AutoModelForCausalLM.from_pretrained(..., torch_dtype=dtype_L)
```

### 2. Device mismatch (samples on CPU, model on GPU)

**Симптом**: `RuntimeError: Expected all tensors to be on the same device, cuda:X and cpu!`

**Причина**: `load_calibration_data()` возвращает samples на CPU.

**Фикс**: `.to(model.device)` перед `compute_jacobian_for_tokens`.

```python
J = compute_jacobian_for_tokens(m, layer,
    s["input_ids"].to(m.device), s["attention_mask"].to(m.device), ts)
```

### 3. dtype mismatch (fp32 hidden state, fp16 model)

**Симптом**: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`

**Причина**: `h.float()` (hardcoded float32) + модель в fp16.

**Фикс**: `h.to(model_dtype)` в `compute_jacobian_for_tokens`.

```python
model_dtype = next(model.parameters()).dtype
h = h.to(model_dtype).detach().requires_grad_(True)
```

### 4. GPU tensor → numpy → TypeError

**Симптом**: `TypeError: can't convert cuda:X device type tensor to numpy`

**Причина**: `.numpy()` / `.tolist()` на GPU-тензоре без `.cpu()`.

**Где искать**: `grep -rn '\.numpy()\|\.tolist()' experiments/ kaggle_kernel/`

**Фикс**: `.cpu()` перед `.numpy()` / `.tolist()`.

```python
E[0].cpu().numpy().tobytes()       # _gramian_key
ev.cpu().tolist()                   # JSON save
```

### 5. G_inv disk cache на CPU, умножение с GPU

**Симптом**: device mismatch после загрузки G_inv с диска.

**Причина**: `torch.load()` грузит на CPU; cached G_inv ≠ device(E).

**Фикс**: `.to(E.device)` при возврате из `_inverse_gramian`.

```python
return _gramian_cache[key].to(E.device)
```

### 6. torch.ones(K) на CPU, eigenvectors на GPU

**Симптом**: `cuda:X and cpu! (when checking argument for argument tensor in method wrapper_CUDA__dot)`

**Причина**: `torch.ones(K)` → CPU по умолчанию; eigenvectors от `compute_esg()` на GPU.

**Фикс**: `device=eigenvectors.device`.

```python
torch.ones(K, device=eigenvectors.device) / torch.sqrt(...)
```

### 7. UnboundLocalError: AutoModelForCausalLM

**Симптом**: `cannot access local variable 'AutoModelForCausalLM'`

**Причина**: `from transformers import AutoModelForCausalLM` внутри `main()` → Python считает локальной переменной. Если что-то падает до этой строки — переменная unbound.

**Фикс**: импорт на уровне модуля (выживает стрип prism-импортов).

### 8. kernel.py vs kernel.ipynb

**Симптом**: Kaggle исполняет kernel.py вместо kernel.ipynb.

**Фикс**: удалить `kernel.py` и `kernel_flat.py` из `kaggle_kernel/`.

### 9. P100 incompatible with PyTorch 2.x

**Симптом**: `CUDA capability sm_60 is too low`

**Причина**: Kaggle P100 = sm_60, PyTorch требует sm_70+.

**Фикс**: `kernel-metadata.json` → `machine_shape: NvidiaTeslaT4`. Push: `kaggle kernels push --accelerator gpu_t4`.

### 10. pip install без -U → старый transformers

**Симптом**: `ModuleNotFoundError: No module named 'LinearAttentionLayer'`

**Причина**: `pip install` без `-U` пропускает уже установленные пакеты.

**Фикс**: `!pip install -q -U transformers` после установки torch.

### 11. strip_prism_imports — индентированные импорты

**Симптом**: `ImportError: from prism.calibration import ...` не стрипнуты.

**Причина**: regex `\n?from prism.X` не ловит `from prism.X` (отступы внутри функций).

**Фикс**: `\s*` после `\n?` в regex.

```python
re.sub(r"\n?\s*from prism\.\w+ import [^\n]*\n?", "", code)
```

### 12. log_probs = torch.zeros(1) → device mismatch

**Симптом**: `cuda:X and cpu!` при累加 log_probs.

**Фикс**: `torch.zeros(1, device=model.device)`.

### 13. os.path.isdir() лжёт на kernel_sources

**Симптом**: директория видна в `os.listdir('/kaggle/input/')`, но `os.path.isdir('/kaggle/input/<slug>/path')` → `False`.

**Причина**: Kaggle FUSE mount — listdir работает, isdir нет.

**Фикс**: проверяй через `os.listdir(parent) + basename`.

```python
def resolve_model(hf_id):
    slugs = os.listdir("/kaggle/input")
    for slug in slugs:
        p_sub = os.path.join("/kaggle/input", slug)
        contents = os.listdir(p_sub)
        for model_dir in contents:
            # ... match and return
    # fallback: HF download
```

### 14. Модель сохранена, токенайзер нет

**Симптом**: `ValueError: Couldn't instantiate the backend tokenizer`

**Причина**: `model.save_pretrained()` сохраняет веса, а не токенайзер.

**Фикс**: отдельно сохранить токенайзер.

```python
model.save_pretrained(dst)
transformers.AutoTokenizer.from_pretrained(hf_id).save_pretrained(dst)
```

### 15. fp16 overflow → NaN в J_tgt_p

**Симптом**: все элементы J_tgt_p NaN после transport chain.

**Причина**: `torch.float16` + `autograd.grad` через deep chain rule = overflow (~10^4 range).

**Фикс**: `torch.bfloat16` — тот же размер (2 bytes), диапазон как fp32 (~10^38).

```python
dtype_L = torch.bfloat16 if device.startswith("cuda") else torch.float32
```

### 16. retain_graph → OOM при accumulate + backward

**Симптом**: `CUDA out of memory` при `loss.backward(retain_graph=True)` в цикле по samples.

**Причина**: каждый decoder-pass graph (~12GB) хранится до последнего `.backward()`.

**Фикс**: `.backward()` per-sample (gradients accumulate automatically), или `.item()` + del + empty_cache.

### 17. `device == "cuda"` не ловит `"cuda:0"`

**Симптом**: GPU-конфиг не применяется, идёт CPU fallback (dtype float32 → OOM).

**Фикс**: `device.startswith("cuda")`.

### 18. runtime guard уменьшает K посередине

**Симптом**: size mismatch `softmax_jacobian` (K=128 vs 256).

**Причина**: guard проверил время, уменьшил K, но J_L уже вычислен с K=256.

**Фикс**: guard BEFORE J_L computation — отдельный probe.

### 19. h.detach() → zero gradient на lm_head

**Симптом**: bridge loss есть, но lm_head не обновляется (delta_W = 0).

**Причина**: `compute_jacobian_for_tokens` делает `h.detach().requires_grad_(True)` — градиент не проходит через модель.

**Вывод**: bridge loss purely diagnostic. Для оптимизации lm_head нужен KL leash или direct loss.

## Массовый поиск проблем

```bash
# Найти все .numpy() без .cpu():
grep -rn '\.numpy()' experiments/ | grep -v '.cpu()'

# Найти все .tolist() без .cpu():
grep -rn '\.tolist()' experiments/ | grep -v '.cpu()'

# Найти torch.ones/zeros без device=:
grep -rn 'torch\.ones\|torch\.zeros' experiments/ | grep -v 'device='

# Найти .to() после extract_embeddings:
grep -rn 'extract_embeddings\|\.to(device' experiments/
```

## Архитектура Notebook

| Ячейка | Содержимое |
| -------- | ----------- |
| 0 | `!pip install` deps |
| 1 | imports + GPU detection + `resolve()` |
| 2 | `prism.jacobian` (inline) |
| 3 | `prism.bridge` (inline) |
| 4 | `prism.esg` (inline) |
| 5 | `prism.calibration` (inline) |
| 6 | exp13 (stripped, `main_13`) |
| 7 | exp14 (stripped, `main_14`) |
| 8 | exp15 (stripped, `main_15`) |
| 9 | runner cell — JOBS loop |

Все функции `prism.*` доступны глобально после выполнения ячейки. Импорт `from prism.X` в эксп-файлах удалён, имена используются как globals.

## GPU allocation pattern

```python
if num_gpus >= 2:
    device_L, device_S = "cuda:0", "cuda:1"
    dtype_L = dtype_S = torch.float32
elif num_gpus == 1:
    device_L = device_S = "cuda:0"
    dtype_L = torch.float16  # 4B model, avoid OOM
    dtype_S = torch.float32
else:
    device_L = device_S = "cpu"
    dtype_L = dtype_S = torch.float32
    torch.set_num_threads(4)
```

## kernel-metadata.json

```json
{
  "kernel_type": "notebook",
  "is_gpu_enabled": true,
  "machine_shape": "NvidiaTeslaT4",
  "accelerator": "gpu",
  ...
}
```

**Не удалять `machine_shape: NvidiaTeslaT4`** — без него Kaggle может назначить P100.
