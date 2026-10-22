## 2. Архитектурный контекст: Qwen3.5

Архитектурные факты, влияющие на корректность вычислений. Производные — в §8.

**Гибридная архитектура.** `config.layer_types[layer_idx]` определяет тип слоя: `"full_attention"` → `Qwen3_5Attention` (GQA); `"linear_attention"` → `Qwen3_5GatedDeltaNet` (GDN). Типы чередуются.

**GDN: рекуррентное состояние.** GDN поддерживает рекуррентное состояние `S_t ∈ R^{H × D_k × D_v}`:

```
S_t = g_t * S_{t-1} + k_t * delta_t^T
output_t = S_t * q_t
```

Кэш GDN содержит `conv_state: [1, conv_dim, conv_kernel_size − 1]` и `recurrent_state: [1, num_v_heads, head_k_dim, head_v_dim]`. GDN — не трансформер: обратный проход идёт через рекуррентную цепочку, а не через attention-матрицы.

**Мутация кэша.** `torch_causal_conv1d_update` мутирует `conv_state` in-place (`conv_state.copy_(...)`); GDN обновляет `recurrent_state` через `cache_params.update_recurrent_state(...)`. Поэтому ни `copy.deepcopy(prefix_cache)`, ни ручной клон не чинят in-place-запись в сохранённый узел графа — это делает только `_clone_gdn_states_on_entry()`. Используется `_deep_clone_cache()` (ручной клон всех тензоров + объектные атрибуты кэша; `deepcopy` на 5.15.1 тоже даёт независимые копии — ручной клон нужен для detach и устойчивости к переименованию полей слоёв), см. §9 и Приложение C.

**Tied weights.** `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}` — т.е. `W = E`, и проекция через `E` корректна для логитов. Защита: `extract_embeddings` (§12) при каждом вызове проверяет `data_ptr`-тождество `lm_head.weight` и `embed_tokens.weight` и при их расхождении выдаёт `UserWarning` — молчаливое использование не той матрицы анэмбеддинга для нетянутых чекпоинтов превращается в громкий сигнал (в рамках статьи все модели — с tied weights).

**MRoPE.** Multi-dimensional RoPE с `position_ids` формы `[4, B, S]` (text, temporal, height, width); для текстового входа все 4 компоненты одинаковы. При вычислении Якобиана для последнего токена с кэшем: `position_ids = S − 1`.

**RMSNorm — не идентичность.** `Qwen3_5RMSNorm`: `output = (x / rms(x)) * (1 + weight)`; вес инициализирован нулями, поэтому на старте `1 + w = 1` и RMSNorm — чистая нормализация `y = x/RMS(x) ≠ x`. Производная — §8 `[L-rms]`.

**Ограничения версии 1:** CPU (Intel Iris Xe / SYCL, 64 GB, `torch.set_num_threads(4)`, float32 — fp16 на CPU на порядок медленнее); текстовый вход (`Qwen3_5ForCausalLM`); `model.model` — `Qwen3_5TextModel` (единый путь для flat-моделей всех размеров; для мультимодального wrapper `Qwen3_5ForConditionalGeneration` путь — `model.model.language_model`, обрабатывается `_lm_module()`, `prism/jacobian.py`); dtype Якобиана — dtype модели (`model_dtype`, fp16 на GPU — `h.to(model_dtype)`); в экспериментах — Qwen3.5-4B (32 слоя, `D_L = 2560`) и Qwen3.5-0.8B (24 слоя, `D_S = 1024`).

---

# Часть II. Математический аппарат

> В Части II — только математика: определения, леммы, свойства. Результаты экспериментов здесь не фигурируют; статус валидации — в Приложении B и §20. Допустимы ссылки вида «см. E-5» без раскрытия результата.

## 4. Якобиан и модели
