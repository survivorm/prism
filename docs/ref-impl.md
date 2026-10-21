# ref-impl.md — Реализация: листинги и инфраструктура

Листинги C.1–C.9 (код) и инфраструктурные детали: вычисление Якобиана (быв. §ref-impl §1),
кэш, данные, 𝒮, layer mapping, загрузка моделей (быв. §ref-impl §4). Канонический текст,
вырезан из paper.md v4.0.

---


# Метод

## 1. Вычисление Якобиана

> **КРИТИЧНО:** Кэш должен содержать состояние **ДО** последнего токена, а не после. Два отдельных прохода.

```python
def compute_jacobian_for_tokens(
    model, layer_idx: int,
    input_ids: torch.Tensor,       # [1, S]
    attention_mask: torch.Tensor,  # [1, S]
    token_set: list[int],          # глобальный набор 𝒮
) -> torch.Tensor:
    """
    Якобиан логитов по скрытому состоянию слоя layer_idx
    для последнего токена, только для токенов из token_set.
    Возвращает: J [len(token_set), D].
    """
    # 3 прохода: (1) префикс -> кэш; (2) полный прогон -> h_l;
    # (3) replay h_l через оставшиеся слои с клоном кэша префикса.
    # Ключевые оговорки: кэш — ДО последнего токена; clone через
    # _deep_clone_cache (deepcopy не чинит in-place запись, §ref-impl §1).
    # Полный код — C.1
```

> Реализации `_deep_clone_cache`, `compute_esg_spectrum`, `compute_bridge` — в Приложении C (C.2–C.4).

> **Производительность.** Строки Якобиана вычисляются **одним батченным backward** (`grad_outputs=torch.eye(K)`, `is_grads_batched=True`) — не циклом по токенам; 3–15× быстрее на T4 (реализовано, `prism/jacobian.py:_compute_jacobian_rows`). Для плотного `[D, D]` Якобиана (`compute_hidden_jacobian`) цикл по `D` строк остаётся — `vmap`/`jacrev` там future work.

**GDN cache cloning.** Ни `copy.deepcopy(prefix_cache)`, ни ручной клон не чинят in-place-запись в сохранённый узел графа (на 5.15.1 `deepcopy` даёт независимые копии — мутации изолированы; проблему в-place решает только `_clone_gdn_states_on_entry`, ниже). `_deep_clone_cache()` — ручной клон всех тензоров в кэше через **прямое присваивание атрибутов** (конструкторные kwargs в обоих типах слоёв молча отбрасываются — листинг C.2) плюс **объектные атрибуты кэша** (`vars(cache)`: на 5.15.1 — флаг `offloading`, в 4.x — `_seen_tokens`) и assert `get_seq_length()` после клона (на 5.15.1 длина выводится из `keys.shape`; на версии с объектным учётом длины assert превращает тихую порчу в громкий сбой). Внедрено (`prism/jacobian.py`).

**In-place мутация сохранённых узлов графа.** Detach не решает проблему: in-place `copy_` (оптимизация статических адресов cudagraph в transformers 5.15) попадает на объект, сохранённый рекуррентным циклом для backward, — version-счётчик узла повышается в **любом** backward через декод (не только при `create_graph`). Решение — `_clone_gdn_states_on_entry()`: monkey-patch ядер `torch_recurrent_gated_delta_rule` / `causal_conv1d_update`, клонирующий входное состояние **на входе ядра** (объект чтения ≠ объект записи), с восстановлением оригиналов после декода. Активен во всех градо-декодах (логит-Якобиан §ref-impl §1, hidden-Jacobian, `bridge_rows_backward`); в `no_grad`-прогонах клон избыточен, но безвреден. **Гарантия действия патча — версионно-зависимая, эмпирическая**: патч подменяет атрибуты модуля `modeling_qwen3_5`, а GDN `forward` на 5.15.1 разрешает ядра как свободные переменные этого модуля (вызовы L457/L506); если ядро вызывается через алиас в другом модуле, патч молча не действует — сигнал обнаружения будет громкий (autograd-ошибка в месте, который патч защищает, вернётся). Патч **реентерабелен** (вложенный контекст — no-op, клон идемпотентен); обёртки фиксируют только `initial_state`/`cs`, хвост сигнатуры ядер прокидывается без изменений (устойчиво к добавлению позиционных параметров; удалённый/переименованный параметр — громкий `TypeError`).

**Аналитический Якобиан последнего слоя** (`_last_layer_jacobian`): `J = ∂h_norm/∂h` без backward (RMSNorm: `[L-rms]`; LayerNorm: аналог с `diag(w)`). Вход `h` может прийти как `[1, S, D]` (полные hidden_states) — схлопывается до последней позиции **до** `reshape(1,-1)`: иначе identity размера `[S·D, S·D]` (например `[65536, 65536]` = 16 GB) → OOM на GPU / XLA-аборт на TPU.

> **⚠️ ЗАМЕЧАНИЕ ПО ПАМЯТИ.** Для длинных последовательностей это ~1–2 GB. Если не хватает памяти — уменьшить $S$ до 512.

**Ограничение на вход.** Только одна последовательность **без паддинга**: replay-фаза строит causal-маски от длины кэша с `attention_mask=None`, поэтому padding-токены в префиксе молча считались бы валидным контекстом. Ограничение зафиксировано в коде (`ValueError` в `compute_jacobian_for_tokens`, v23.2).

**Версионно-зависимое поведение (vmap через GDN-ядра).** Батченный backward проходит через хвост декодера с `linear_attention`-слоями (рекуррентные ядра `torch_recurrent_gated_delta_rule` / `causal_conv1d_update`). Идентичность построчному циклу проверена на синтетических графах; на реальных ядрах — эмпирический факт для transformers 5.15 (прогоны на T4), **фолбэка на построчный цикл нет**: апгрейд transformers, ломающий batched backward через GDN, даст runtime-ошибку, а не тихую деградацию.

**Ключевые свойства.** (1) Глубинный доступ к `h_l` через `hidden_states[layer_idx + 1]` (проход 2) без изменения модели. (2) KV cache делает forward $O(S \cdot D)$, а backward — $O(D^2)$ на слой (без re-forward по $S$). (3) Один батченный backward со всеми $K$ строками (`retain_graph=False`; см. «Производительность») — граф не удерживается между строками. (4) `position_ids = S−1` для последнего токена при использовании KV cache (MRoPE).

## 2. Jacobian bridge

Полный алгоритм (§paper §3, §paper §3, §paper §3, §paper §3) — листинг `jacobian_gap` (с Грамиан-коррекцией `pinv(E_S^T E_S)` и sentinel-маской) — Приложение C.

**Свойство (identity).** При $M_S = M_L$, $E_S = E_L$: $J_L \cdot E^T \cdot E \cdot \mathrm{pinv}(E^T E) = J_L$ (в пределах pinv-условия, [L-gram]), $J_L \approx J_S$ (детерминизм), $\mathcal{G}_{\text{gap}} \approx 0$ ([L-dim] + [L-gram]). Это — базовая самопроверка корректности всей цепочки (E-1: gap ≈ 10⁻⁸ на всех 24 слоях).

## 3. Диагностика P0 и ESG P1

**P0 (карта расхождения).** Для всех пар слоёв $(l_L, l_S)$: вычислить $\mathcal{G}_{\text{gap}}(l)$, $\mathcal{B}_{\text{bridge}}(l)$. Выход: монотонная карта по глубине (E-2, E-6, E-13, E-15). Направления: `gap` — меньше = лучше, `bridge` — больше = лучше (§paper §3).

**P1 (карта чувствительности для квантования).** Для каждого слоя $l$:

1. $J_S(l)$ — Якобиан, §ref-impl §1.
2. **Защита** (hidden space): правые сингулярные векторы $J_S^T J_S$ — направления в $h$, наиболее чувствительные к изменениям.
3. **Эксплойт** (token space): собственные векторы ESG $\bar{G}_l = JJ^T$.

**ESG-вычисление.** `N` Якобианов одной траектории (позиции сэмпла) — внутри-траекторный пул; **sample-local** по сэмплам (E-0.1: меж-сэмпловый пул невалиден — Якобианы разных сэмплов почти ортогональны; агрегация по сэмплам — только скалярные метрики, §paper §3):

$$\bar{G}_l = \frac{1}{N} \sum_i J^{(i)} (J^{(i)})^T \quad (\text{усреднение по сэмплам, не пул})$$

Спектральное разложение — стандартное (матрица симметричная, $K \times K$). Метрики: SNR, PR, gauge share — §paper §3.

**Связь с P2/P3.** P2 реализован (Exp 17–19, §paper §9): лосс $\mathcal{L}_{p2} = \sum_l \alpha_l \cdot \mathcal{G}_{\text{gap}}(l)$ — bridge-лосс на matrix-free per-sample J-rows (chunked backward, HVP), оптимизирует LoRA r=4 на lm_head + o_proj(l_S). P3 (регуляризатор) — гипотеза, отложена: $\mathcal{L}_{p3} = \sum_l \beta_l \, \|J_S(l) - J_S^{\text{target}}(l)\|_F^2$.

---

## 4. Данные, 𝒮 (глобальный токен-сет), layer mapping, загрузка

**Калибровочные данные.** C4 (`allenai/c4`, `en`, `train`, streaming): токенизация `max_length=64`, отбор сэмплов ≥64 токенов. В дизайне §2.1 (v1) предполагалось S=2048/512, N=32; в tested-протоколах: **S=64, N=1–4** (ограничения CPU). Для полных экспериментов рекомендуем N ≥ 4, S ≥ 256. Доменные эксперименты (E-0.1, E-13, E-27): 4 домена — wiki (C4), math (OpenMathInstruct-2), code (apex-agents), chat (ultrachat).

**Глобальный набор токенов $\mathcal{S}$ (support).** Токены отбираются **заранее** по частоте: union top-`K_max` токенов по калибровочному набору (старшая модель). Это гарантирует, что строки Якобианов выровнены по одним и тем же токенам на всех сэмплах. Фактический размер `K = |𝒮|` (перекрытия top-`K_max` высоки; типовым случаем является `K = K_max`).

> **Рекомендация (E-10c):** top-K покрытие нестабильно (min coverage 0.86 при K=256, 0.66 при K=32). Для стабильности — top-p саппорт (токены с совокупной вероятностью ≥ p, напр. p=0.99).

**Сопоставление слоёв (v1 — равномерное).**

```python
def uniform_layer_mapping(n_layers_large, n_layers_small):
    mapping = {}
    for i in range(n_layers_large):
        j = round(i * (n_layers_small - 1) / (n_layers_large - 1))
        mapping[i] = j
    return mapping
```

E-15 проверяет фазовую робастность: 4 схемы якорей (full/full±1/full−2) дают идентичные кривые (Spearman 1.0 между парами, max diff 0.12).

**Загрузка моделей.** CPU, float32 обязателен (fp16 на CPU ~10× медленнее), `torch.set_num_threads(4)`, `attn_implementation="eager"` (нужны `hidden_states`). Эмбеддинги: `model.model.embed_tokens.weight` — `E_large [V, 2560]`, `E_small [V, 1024]`.


Код вынесен из основного текста (§ref-impl §1–§ref-impl §4, §ref-exp §2). В теле — только сигнатуры и ключевые комментарии.

## C.1. Вычисление Якобиана

```python
def compute_jacobian_for_tokens(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,       # [1, S]
    attention_mask: torch.Tensor,   # [1, S]
    token_set: list[int],           # глобальный набор 𝒮, K = len(token_set)
    prefix_cache=None,              # опциональный переиспользуемый кэш префикса (v23.1)
) -> torch.Tensor:
    """
    Якобиан логитов по скрытому состоянию слоя layer_idx
    для последнего токена, только для токенов из token_set.
    Возвращает: J [K, D] (ненормализованный, dtype модели).
    """
    model.eval()
    S = input_ids.shape[1]

    # ============================================================
    # ПРОХОД 1: Префикс (все токены кроме последнего) -> кэш.
    # КРИТИЧНО: кэш = состояние ДО последнего токена (§ref-impl §1).
    # Переиспользуется, если вызывающий передаёт prefix_cache
    # (exp17: h из KL-forward не дублируется).
    # ============================================================
    if prefix_cache is None:
        with torch.no_grad():
            prefix_cache = model(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=True,
                output_hidden_states=False,
            ).past_key_values

    # ============================================================
    # ПРОХОД 2: Полная последовательность -> скрытое состояние
    # ============================================================
    # hidden_states[0] = эмбеддинги, hidden_states[l+1] = выход слоя l
    h = model(input_ids=input_ids, attention_mask=attention_mask,
              output_hidden_states=True).hidden_states[layer_idx + 1][:, -1:, :]
    # dtype модели (fp16 на GPU; .float() дал бы mismatch в декодере)
    model_dtype = next(model.parameters()).dtype
    h = h.to(model_dtype).detach().requires_grad_(True)

    # enable_grad: декодерный проход и batched backward обязаны работать
    # даже из torch.no_grad() вызывающего (eval-циклы exp18/19).
    with torch.enable_grad():
        logits = _run_decoder_pass(model, prefix_cache, h, S, layer_idx)
        J = _compute_jacobian_rows(logits, h, token_set)
    return J  # [K, D], ненормализованный


def _run_decoder_pass(model, prefix_cache, h, S, layer_idx):
    """Декодит h через оставшиеся слои с клоном кэша префикса."""
    # ручной клон прямым присваиванием атрибутов (C.2); in-place запись
    # в сохранённый узел графа чинит _clone_gdn_states_on_entry (§ref-impl §1)
    cache_clone = _deep_clone_cache(prefix_cache)
    text_model = _lm_module(model)  # flat: model.model; mm-wrapper: .language_model

    # MRoPE: 4 компонента, все = S-1 (позиция последнего токена, 0-indexed)
    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
    text_position_ids, rope_position_ids = pos[0], pos[1:]
    position_embeddings = text_model.rotary_emb(h, rope_position_ids)

    x = h  # [1, 1, D]
    n_layers = model.config.num_hidden_layers
    start = layer_idx + 1

    # Маски строится ЯВНО (create_causal_mask для full_attention,
    # create_recurrent_attention_mask для linear_attention/GDN) —
    # не "mask=None": при seq_len=1 с кэхом это путь, на котором
    # трансформеры 5.15 ожидает маску по layer_type.
    causal_mask_mapping = _build_causal_masks(text_model, text_position_ids, cache_clone, x)

    with _clone_gdn_states_on_entry():  # in-place copy_ vs сохранённые узлы графа (§ref-impl §1)
        for i in range(start, n_layers):
            layer_type = model.config.layer_types[i]
            x = text_model.layers[i](
                x,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=text_position_ids,
                past_key_values=cache_clone,
            )

    x = text_model.norm(x)
    return model.lm_head(x[:, -1, :])  # [1, V]


def _compute_jacobian_rows(logits, h, token_set):
    """ОДИН батченный backward вместо цикла по токенам (3–15x на T4).

    grad_outputs=eye(K) даёт все K строк за один проход (vmap-style);
    верифицировано идентичным циклу на синтетических графах.
    """
    K = len(token_set)
    J = torch.autograd.grad(
        outputs=logits[0, token_set],
        inputs=h,
        grad_outputs=torch.eye(K, device=h.device, dtype=h.dtype),
        is_grads_batched=True,
        retain_graph=False,
    )[0]
    return J[:, 0, 0, :]  # [K, D]
```
## C.2. _deep_clone_cache (GDN in-place мутации)

```python
def _deep_clone_cache(cache):
    """Deep-клон DynamicCache. КРИТИЧНО: прямое присваивание атрибутов,
    НЕ конструкторские kwargs.

    Оба типа слоёв глотают незнакомые kwargs: DynamicLayer(keys=...) и
    LinearAttentionLayer(conv_states=..., ...) молча ОТБРАСЫВАЮТ состояние
    и дают пустой кэш (каждый слой видит только текущий токен / GDN считает
    токен первым в последовательности). Поэтому — атрибуты напрямую.

    * DynamicLayer (full attention): keys / values / is_initialized / dtype /
      device. Без них update() лениво инициализирует пустой кэш.
    * LinearAttentionLayer (GDN): все state-поля КОПИРУЮТСЯ ДИНАМИЧЕСКИ
      через обход vars(layer) (не хардкод имён полей 5.15.1 — другие
      версии transformers называют часть полей иначе): тензоры — clone+detach,
      словари тензоров (conv_states, recurrent_states, ...) — поэлементно,
      скаляры/flag-dicts (has_previous_state и т.п.) — deepcopy.

    Объектные атрибуты САМОГО кэша (vars(cache): на 5.15.1 — offloading, в 4.x
    — _seen_tokens) копируются после цикла по слоям; assert get_seq_length()
    после клона — тихую порчу на версии с объектным учётом длины превращает
    в громкий сбой.

    deepcopy(prefix_cache) на 5.15.1 ТАКЖЕ даёт независимые копии (мутации
    изолированы); ручной клон нужен для (а) detach (нет паразитных
    CopyBackwards-узлов в enable_grad-декоде) и (б) устойчивости к
    переименованию полей слоёв. Ни один клон не чинит in-place-запись в
    сохранённый узел графа — этим занимается _clone_gdn_states_on_entry.
    """
    import copy as _copy

    def _cl(t):
        return t.clone().detach() if t is not None else None

    new_cache = DynamicCache()
    for idx in range(len(cache.layers)):
        layer = cache.layers[idx]
        if isinstance(layer, DynamicLayer):
            # НЕ vars()-обход: поля full-attention стабильны между версиями,
            # переименование падает ГРОМКО (AttributeError), а не тихо.
            # GDN-ветка ниже ОБЩАЯ — поля GDN действительно дрейфуют.
            nl = DynamicLayer()
            nl.keys = _cl(layer.keys)
            nl.values = _cl(layer.values)
            nl.is_initialized = layer.is_initialized
            nl.dtype = getattr(layer, "dtype", None)
            nl.device = getattr(layer, "device", None)
            for k, v in vars(layer).items():       # forward-compat safety net
                if k not in vars(nl):
                    setattr(nl, k, _copy.deepcopy(v))
            new_cache.layers.append(nl)
        else:  # LinearAttentionLayer (GDN): generic vars()-обход
            nl = LinearAttentionLayer()
            for k, v in vars(layer).items():
                if isinstance(v, torch.Tensor):
                    setattr(nl, k, _cl(v))
                elif isinstance(v, dict) and v and all(
                    isinstance(x, torch.Tensor) for x in v.values()
                ):
                    setattr(nl, k, {kk: _cl(xv) for kk, xv in v.items()})
                else:
                    setattr(nl, k, _copy.deepcopy(v))
            new_cache.layers.append(nl)
    # Объектное состояние кэша (offloading на 5.15.1; 4.x: _seen_tokens):
    # свежий DynamicCache() сбросил бы его; list-атрибуты не копируем
    # (layers уже собран заново; алиасинг чужого списка не допустим).
    for k, v in vars(cache).items():
        if k == "layers" or isinstance(v, list):
            continue
        setattr(new_cache, k, _copy.deepcopy(v))
    # Страховка: клон отчитывается о той же длине последовательности.
    assert new_cache.get_seq_length() == cache.get_seq_length()
    return new_cache

# Детач: префиксный кэш — чистые данные (построен под no_grad); detach убирает
# бессмысленные CopyBackwards-узлы в enable_grad-декоде. НО in-place copy_ из
# update_recurrent_state / causal_conv1d_update (cudagraph-оптимизация
# transformers 5.15) detach НЕ чинит — in-place запись падает на объект,
# сохранённый ядром для backward. Этим занимается _clone_gdn_states_on_entry
# (клон на входе ядра: объект чтения ≠ объект записи), см. §ref-impl §1.
```

## C.3. jacobian_gap (из `prism/bridge.py`)

```python
def jacobian_gap(
    J_L: torch.Tensor,       # [K, D_L] — Якобиан старшей модели
    J_S: torch.Tensor,       # [K, D_S] — Якобиан младшей модели
    E_L: torch.Tensor,       # [V, D_L] — эмбеддинги старшей
    E_S: torch.Tensor,       # [V, D_S] — эмбеддинги младшей
    probs_L: torch.Tensor,   # [K] — вероятности токенов в старшей модели
    threshold: float = 0.0,  # абсолютный порог на ОБЕ norms; дефолт 0.0 =
                             # sentinel-only (маскируются только нулевые строки).
                             # Decay-маска: вызывающий сам передаёт абсолютный τ;
                             # относительный τ_l = 0.01·median(‖J_S[i]‖) (§paper §3, E-6)
                             # вычисляется вызывающим.
) -> float:
    """Gap (lower=better) = weighted (1 - cos) по валидным строкам.

    Терминология v4.0: gap = 1 - bridge при равных весах. Вызывающий,
    отчитывающийся по «bridge», инвертирует: bridge = 1 - jacobian_gap(...).
    Legacy-алиас `jacobian_bridge` удалён (имя bridge + направление GAP —
    ловушка инверсии метрики); инверсия живёт в экспериментальных ядрах.
    """
    G_inv = _inverse_gramian(E_S)       # pinv(E_S^T E_S), диск-кэш (§paper §3)
    J_L_proj = _project_to_vocab(J_L, E_L)      # [K, D_L] @ [D_L, V] → [K, V]
    J_S_target = _project_to_small(J_L_proj, E_S, G_inv)  # [K, V] → [K, D_S]
    mask = _valid_mask(J_S, J_S_target, threshold)
    if mask.sum() == 0:
        return 1.0  # sentinel (§paper §3)
    return _weighted_cosine_loss(J_S, J_S_target, probs_L, mask)


def _inverse_gramian(E: torch.Tensor) -> torch.Tensor:
    """(E^T E)^{-1} через torch.linalg.pinv (SVD); кэш: память + диск
    (results/.gramians/), ключ = SHA-256(shape|device|E[0]|E[-1]|10 mid-rows) — см. §paper §3."""
    key = _gramian_key(E)
    if key not in _gramian_cache:
        path = os.path.join(_GRAMIAN_DIR, f"{key}.pt")
        if os.path.exists(path):
            _gramian_cache[key] = torch.load(path, weights_only=True)
        else:
            _gramian_cache[key] = torch.linalg.pinv(E.float().T @ E.float())
            os.makedirs(_GRAMIAN_DIR, exist_ok=True)
            torch.save(_gramian_cache[key], path)
    return _gramian_cache[key].to(E.device)


def _valid_mask(J_S, J_target, threshold):
    """Маска по обеим нормам: (‖J_S[i]‖ > τ) & (‖J_target[i]‖ > τ)."""
    n1 = J_S.float().norm(dim=1)
    n2 = J_target.float().norm(dim=1)
    return (n1 > threshold) & (n2 > threshold)


def _weighted_cosine_loss(J_S, J_target, probs, mask) -> float:
    cos_sim = torch.nn.functional.cosine_similarity(
        J_S.float()[mask], J_target[mask], dim=1,
    )  # eps=1e-8 по умолчанию
    w = probs[mask] / (probs[mask].sum() + 1e-8)   # нормировка по незамаскированным (§paper §3)
    return (w * (1.0 - cos_sim)).sum().item()
```

## C.4. run_layer_diagnostics (`prism/diagnostics.py`)

```python
def run_layer_diagnostics(
    model_large, model_small,
    token_set, layer_mapping, samples,
    results_dir="results", exp_name="exp2_heatmap",
) -> dict[str, float]:
    """Gap (lower=better) для всех пар слоёв; E-1 (identity) и E-2 (heatmap).

    E-1 — run_identity_test: ta же модель и как large, и как small,
    uniform_layer_mapping → gap ≈ 10⁻⁸. Результат сохраняется в
    results/<exp_name>.json.
    """
    E_L, _ = extract_embeddings(model_large)
    E_S, _ = extract_embeddings(model_small)
    results = {}
    for l_L, l_S in layer_mapping.items():
        losses = []
        for sample in samples:
            input_ids = sample["input_ids"].to(model_large.device)
            attention_mask = sample["attention_mask"].to(model_large.device)
            J_L = compute_jacobian_for_tokens(
                model_large, l_L, input_ids, attention_mask, token_set)
            J_S = compute_jacobian_for_tokens(
                model_small, l_S, input_ids, attention_mask, token_set)
            # Вероятности токенов из token_set в старшей модели (§paper §3)
            with torch.no_grad():
                logits_L = model_large(input_ids=input_ids).logits[:, -1, :]
                probs_L = torch.softmax(logits_L[0, token_set], dim=-1)
            loss = jacobian_gap(J_L, J_S, E_L, E_S, probs_L)
            losses.append(loss)
        results[f"{l_L}->{l_S}"] = sum(losses) / len(losses)
    _save_results(results, exp_name, results_dir)
    return results
```

## C.5. ESG

```python
def compute_esg(jacobians: list[torch.Tensor], n_components: int = 32):
    """ESG от списка Якобианов ОДНОЙ траектории (позиций одного документа).

    Пул: G = (1/N) Σ_i J_i J_i^T,  J_i: [K, D] → G: [K, K].
    Пул ВНУТРИ сэмпла валиден (C_a = 8–41× F, E-27.6); междосеамплыный
    пул НЕВАЛЕН (E-0.1: C_b ≈ 0.003–0.05, J почти ортогональны) —
    по сэмплам агрегируем только скаляры (λ, PR, SNR, r_eff).

    Returns: eigenvalues [min(n_components, K)] (по убыванию), eigenvectors [min(n_components, K), K]
    """
    _validate(jacobians)  # формы и device
    G = _build_gramian(jacobians)  # (1/N) Σ J_i J_i^T
    return _spectral_decomposition(G, n_components)


def compute_esg_per_sample(sample_jacobians, n_components=32):
    """Sample-local ESG: один Грамиан на сэмпл, без междосеамплыного пула.

    Args: sample_jacobians: list по сэмплам, каждый — list по позициям, J: [K, D]
    Returns: (eigenvalues: list[n_samples] × [n_components],
              eigenvectors: list[n_samples] × [n_components, K])
    """
    eigenvalues, eigenvectors = [], []
    for Js in sample_jacobians:
        ev, evec = compute_esg(Js, n_components=n_components)
        eigenvalues.append(ev); eigenvectors.append(evec)
    return eigenvalues, eigenvectors


def _build_gramian(jacobians):
    U, device = jacobians[0].shape[0], jacobians[0].device
    G = torch.zeros(U, U, device=device, dtype=torch.float)
    for J in jacobians:
        G += J.float() @ J.float().T
    return G / len(jacobians)


def _spectral_decomposition(G, n_components):
    eigenvalues, eigenvectors = torch.linalg.eigh(G)
    idx = eigenvalues.argsort(descending=True)
    return (eigenvalues[idx][:n_components],
            eigenvectors[:, idx][:, :n_components].T)  # [n_components, K]
```

## C.6. Калибровочные данные

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

calibration_samples = []
for sample in dataset:
    tokens = tokenizer(sample["text"], max_length=64, truncation=True, return_tensors="pt")
    if tokens["input_ids"].shape[1] >= 64:
        calibration_samples.append(tokens)
    if len(calibration_samples) >= 4:
        break
```

## C.7. Глобальный набор токенов 𝒮

```python
import torch
from collections import Counter

def build_global_token_set(model, calibration_samples, K=64):
    """Собирает глобальный набор токенов из top-K логитов на всех сэмплах."""
    token_counter = Counter()
    model.eval()

    device = next(model.parameters()).device
    with torch.no_grad():
        for sample in calibration_samples:
            outputs = model(
                input_ids=sample["input_ids"].to(device),
                attention_mask=sample["attention_mask"].to(device),
            )
            # Берём логиты последнего токена
            logits = outputs.logits[:, -1, :]  # [1, V]
            top_indices = logits.topk(K, dim=-1).indices[0]  # [K]
            token_counter.update(top_indices.tolist())

    # Топ-K самых частых токенов (most_common(K) ограничивает K сам)
    U = [token_id for token_id, _ in token_counter.most_common(K)]
    return sorted(U)
```

## C.8. Соответствие слоёв

```python
def uniform_layer_mapping(n_layers_large, n_layers_small):
    mapping = {}
    for i in range(n_layers_large):
        j = round(i * (n_layers_small - 1) / (n_layers_large - 1))
        mapping[i] = j
    return mapping
```

## C.9. Загрузка моделей

```python
import torch
from transformers import AutoModelForCausalLM

# CPU: float32 обязателен! fp16 на CPU в 10x медленнее.
torch.set_num_threads(4)

model_large = AutoModelForCausalLM.from_pretrained(
    "/data/Qwen/Qwen3.5-4B",
    torch_dtype=torch.float32,  # float32 на CPU!
    attn_implementation="eager",  # важно: не flash_attention, нужны hidden_states
)
model_large.eval()

model_small = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    torch_dtype=torch.float32,
    attn_implementation="eager",
)
model_small.eval()

# Извлекаем матрицы эмбедингов
# Qwen3.5 (все размеры): model.model — Qwen3_5TextModel
E_large = model_large.model.embed_tokens.weight.detach()  # [V, D_L]
E_small = model_small.model.embed_tokens.weight.detach()  # [V, D_S]
# Qwen3.5-0.8B: D_S = 1024; Qwen3.5-4B: D_L = 2560
```

---
