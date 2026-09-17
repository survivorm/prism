# ТЗ: Exp 27 — Калибровка шумового пола и устойчивости поля J

## 1. Название

**Exp 27: Noise-floor calibration and robustness sweep for the gauge-free Jacobian field J^(p)(x).**

## 2. Обоснование

Exp 0.1 v6 дал три факта, устойчивых к коду:
- gauge_share ≈ 0.001;
- иерархия `C_a(raw) > C_a(same-pk) > C_a(pos-pk)` (0.33/0.26/0.08 на L6 → 0.92/0.85/0.22 на L22);
- σ_d растёт с глубиной, `L_p90(pos-pk)` падает в 8× между L12 и L22.

Но **ни один вывод о «fibered / orthogonal / weakly aligned» не защищён**, потому что:
- noise floor не определён (используется 1/√D = 0.031, что завышено в 3–8× относительно возможного effective dimension);
- все сэмплы обрезаны до S=64 — L18-аномалия (C_b > C_a) может быть truncation artifact;
- позиции только {last, last-1, last-2} — coherence length не измерен;
- K=64 не проверен на устойчивость;
- половина использованных слоёв (6, 12, 18, 22) — **linear_attention** (GDN). Ни один full-attention слой (3, 7, 11, 15, 19, 23) не проверен. Это selection bias, способный объяснить всю «глубинную» динамику как GDN-specific.

Exp 27 закрывает эти пять дыр. Без него любая надстройка (bridge-статистика, quantization, MoE) строится на неизвестном шумовом поле.

## 3. Математический аппарат

Используем только то, что уже определено в paper.md и реализовано в `prism/jacobian.py`, `prism/bridge.py`.

### 3.1. Якобиан

$$J_l^{(t)} = \frac{\partial z_t}{\partial h^l_t} \in \mathbb{R}^{V \times D}, \quad J_l^{(t)}\big|_{\mathcal U} \in \mathbb{R}^{K \times D}$$

Gauge-free (paper.md §2.2):

$$J^{(p)} = (\mathrm{diag}(p) - pp^T)\,J, \quad p = \mathrm{softmax}(z_0)\big|_{\mathcal U},\ \text{renormalized}$$

Три конвенции (v6):
- **pos-pk**: `P(p_t)J_t` — свой проектор на каждой позиции;
- **same-pk**: `P(p_{t_1})J_{t_1}`, `P(p_{t_1})J_{t_2}` — один проектор на пару;
- **raw**: `J_t`, `J_{t_2}` без проектора.

### 3.2. Косинус

$$C(J, J') = \frac{\langle J, J'\rangle_F}{\|J\|_F\,\|J'\|_F}$$

Метрики когерентности (paper.md §1.6–1.7, v6):
- $C_a$ — intra-sample, пары соседних позиций;
- $C_b$ — intra-domain, пары сэмплов на позиции last.

### 3.3. Effective rank (новое)

Для gauge-free $J \in \mathbb{R}^{K \times D}$ с сингулярными числами $\sigma_1 \geq \dots \geq \sigma_r$:

**Энтропийный ранг** (primary):

$$r_{\text{eff}}(J) = \exp\left(-\sum_i \hat\sigma_i^2 \log \hat\sigma_i^2\right), \quad \hat\sigma_i = \frac{\sigma_i}{\sqrt{\sum_j \sigma_j^2}}$$

**Stable rank** (secondary): $r_{\text{stable}} = \|J\|_F^2 / \|J\|_2^2$.

**Participation ratio** (secondary): $r_{\text{PR}} = (\sum \sigma_i^2)^2 / \sum \sigma_i^4$.

### 3.4. Noise floor

Для двух независимых случайных единичных векторов в $\mathbb{R}^d$: $\mathrm{std}[C] = 1/\sqrt{d}$.

Для $J^{(p)}$, живущего в эффективном подпространстве размерности $r_{\text{eff}}(K + D - r_{\text{eff}})$:

$$\mathcal F = \frac{1}{\sqrt{r_{\text{eff}}\,(K + D - r_{\text{eff}})}}$$

Считаем $r_{\text{eff}}$ **на каждом слое** как медиану по сэмплам. Это заменяет 1/√D из v6.

### 3.5. Half-split reliability

N=32 сэмпла (4 домена × 8). Разбиваем случайно 16/16, считаем $\bar C_a$ на каждой половине. 100 бутстреп-ресэмплов. CI = перцентили $|C_a^{(A)} - C_a^{(B)}|$.

### 3.6. Критерии интерпретации (pre-registered)

| Уровень | Условие |
|---|---|
| ORTHOGONAL | $C < \mathcal F$ |
| WEAK | $\mathcal F \le C < 3\mathcal F$ |
| COHERENT | $C \ge 3\mathcal F$ |

Применяется к $C_a$(pos-pk), $C_a$(same-pk), $C_a$(raw), $C_b$ отдельно на каждом слое.

## 4. Граничные условия

### 4.1. Фиксировано

- **Модель:** Qwen3.5-0.8B, float32, `attn_implementation="eager"`, GPU.
- **Токенный набор:** вложенный. Сначала строится $K_{\max}=128$ по top-128 финальных логитов калибровочного пула. Для $K \in \{32, 64, 128\}$ берётся top-K из этого набора. Проектор $P(p_K)$ — по пере-нормированному $p$ на соответствующем подмножестве.
- **Позиции:** отсчитываются от конца: pos = 0, 1, 2, 4, 8, 16. pos=0 = last.
- **gauge convention:** три (raw, same-pk, pos-pk) для каждой пары.
- **Домены:** wiki, math, code, chat. N=8 на домен.
- **Токенизатор:** Qwen, `truncation=True`. Для S-sweep — **левая обрезка** (сохраняет начало текста, отбрасывает конец).

### 4.2. Слои

Множество слоёв: **{6, 12, 18, 19, 22, 23}**.

- 6, 12, 18, 22 — linear_attention (совпадает с v6, для сравнения);
- 19, 23 — full_attention (закрывает selection bias);
- 23 = последний слой (J ≈ W lm_head, тривиальный — контроль).

Слои 3, 7, 11, 15 добавлять не обязательно — они linear, а полный набор из 6 слоёв даёт приемлемую статистику.

### 4.3. Свипы

Три независимых свипа, не полное произведение:

**S-sweep** (fix K=64, pos ∈ {0,1,2}):
$$S \in \{32, 64, 128, 256\}$$

**K-sweep** (fix S=64, pos ∈ {0,1,2}):
$$K \in \{32, 64, 128\}$$

**Position-sweep** (fix S=64, K=64):
$$\text{pos} \in \{0, 1, 2, 4, 8, 16\}$$

Плюс **half-split bootstrap** на baseline (S=64, K=64, pos=0).

### 4.4. Требование к данным

Все сэмплы должны иметь ≥256 токенов после токенизации. Если короче — redraw из C4/GSM8K/apex/ultrachat. Это гарантирует, что S=256 не упирается в длину текста.

## 5. Измеряемые величины

Для каждой конфигурации (layer, S, K, pos):

1. $r_{\text{eff}}$, $r_{\text{stable}}$, $r_{\text{PR}}$ — медиана по 32 сэмплам.
2. $\mathcal F = 1/\sqrt{r_{\text{eff}}(K + D - r_{\text{eff}})}$.
3. $C_a$(pos-pk), $C_a$(same-pk), $C_a$(raw) — средние и std.
4. $C_b$(pos-pk) — среднее и std.
5. $\sigma_d$, $L_{p90}$(pos-pk).
6. $\cos(p_t, p_{t+1})$ — распределение.
7. gauge_share.
8. Half-split CI на baseline.

## 6. Возможные расхождения интерпретации (исключаем)

### 6.1. Асимметрия same-pk

В v6 same-pk использует `P(p_{t_1})` — проектор более ранней позиции. Проверяем и `P(p_{t_2})`. Если результаты отличаются >20% — фиксируем **обе** конвенции в отчёте, не выбираем одну.

### 6.2. Truncation vs position

Если $C_a$ зависит от S, есть две гипотезы:
- (a) **position effect**: чем глубже в текст, тем иначе J;
- (b) **truncation effect**: последняя позиция обрезанного текста всегда «end-of-fragment».

Различить: сравнить $C_a(S=64, \text{pos}=0)$ с $C_a(S=128, \text{pos}=64)$ (одна и та же абсолютная позиция, разный S). Если равны — position. Если различаются — truncation. **Pre-registered.**

### 6.3. K-зависимость

gauge_share и $C_a$ могут зависеть от K. Если gauge_share **растёт** с K — это артефакт token set, а не свойство поля. Если стабилен — свойство.

### 6.4. L18-аномалия

$C_b(L18) > C_a(L18)$ в v6. Три возможных объяснения:
- (a) **end-of-context**: последняя позиция всех сэмплов обрезанных текстов похожа;
- (b) **GDN property**: linear_attention на L18 даёт специфическую динамику;
- (c) **шум**: разница на границе CI.

Различить:
- (a) vs (b): сравнить с full-attention слоем L19. Если аномалии нет на L19 — GDN-specific.
- (a) vs (c): half-split CI.

### 6.5. r_eff зависит от позиции

Если $r_{\text{eff}}$(pos=0) ≠ $r_{\text{eff}}$(pos=16), noise floor разный на разных позициях. Считаем $\mathcal F$ **per position**, а не глобально.

## 7. Интерпретация результата (pre-registered)

### 7.1. По noise floor

| Условие | Вывод | Действие |
|---|---|---|
| $C_a$(pos-pk) < $\mathcal F$ на всех слоях | Дифференциальная часть J — шум | Эксп. 24, 26 закрыть; bridge пересмотреть |
| $C_a$(pos-pk) ∈ [$\mathcal F$, 3$\mathcal F$] на всех слоях | Слабый сигнал, на границе | Bridge остаётся, но с оговорками |
| $C_a$(pos-pk) > 3$\mathcal F$ хотя бы на одном слое | Реальная когерентность | Продолжаем по слоям, где прошло |

### 7.2. По S-sweep

| Наблюдение | Вывод |
|---|---|
| $C_a$ не зависит от S | Поле устойчиво, выводы защищены |
| $C_a$ растёт с S | Больше контекста → больше когерентность; truncation подозрение |
| $C_a$ падает с S | Position effect; L18-аномалия может быть truncation |

### 7.3. По position-sweep

Если $C_a(d)$ убывает с $d$ монотонно — есть coherence length. Если плато — нет позиционной структуры. Если немонотонно — периодичность (проверяем GDN-period 4).

### 7.4. По K-sweep

| Наблюдение | Вывод |
|---|---|
| gauge_share, $C_a$ стабильны | K=64 репрезентативен |
| gauge_share растёт с K | token-set artifact, надо фиксировать K |
| $C_a$ растёт с K | Хвост распределения содержит сигнал |

### 7.5. По half-split

| Условие | Вывод |
|---|---|
| CI($C_a$) уже $\bar C_a$ в 3+ раза | Сигнал стабилен, N=8 достаточно |
| CI($C_a$) $\approx \bar C_a$ | Сигнал sample-dependent, нужно N ≥ 32 |

## 8. Deliverables

1. **JSON** `results/exp27_noise_floor.json`:
    - per (layer, S, K, pos): r_eff (три определения), floor, $C_a$ (три конвенции), $C_b$, σ_d, L_p90, cos_p, gauge_share;
    - half-split CI;
    - manifest: git hash, model hash, token_set hash.

2. **PNG**:
    - heatmap $C_a$(pos-pk) / $\mathcal F$ по (layer × S);
    - heatmap $C_a$ / $\mathcal F$ по (layer × pos);
    - $r_{\text{eff}}$ по слоям (bar chart, три определения);
    - half-split distribution.

3. **Отчёт** `docs/exp27_findings.md`:
    - pre-registered verdicts (таблица);
    - что защищено, что нет;
    - список следующих экспериментов с обоснованием.

## 9. Что НЕ делается в Exp 27

- Bridge-статистика (Exp 26).
- MoE-фикация.
- Кластеризация листов (Exp 24).
- Кросс-модельные сравнения (4B).
- Quantization.

Всё это — **после** закрытия noise floor. Порядок: Exp 27 → интерпретация → Exp 26/24/25 по результатам.

## 10. Бюджет

- J-вычислений: ~2500 (6 слоёв × 32 сэмпла × ~13 конфигураций).
- GPU T4: ~2–3 часа.
- CPU постобработка: минуты.

## 11. Критерий остановки

Exp 27 считается завершённым, когда:
- для всех (layer, S, K, pos) вычислен $\mathcal F$;
- half-split CI на baseline получен;
- pre-registered verdicts присвоены;
- обновлён `docs/exp27_findings.md`.

**До этого момента** ни один документ не должен содержать фразы «field is fibered», «orthogonal», «smooth», «weakly aligned» без ссылки на конкретные числа и $\mathcal F$.