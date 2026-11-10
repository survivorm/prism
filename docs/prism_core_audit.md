# Аудит ядра `prism/` (31 файл, 7947 строк)

Дата: 11.11 (рабочее дерево восстановлено 21:19; аудит пофайлово на
`prism/` = GitHub main 5b52eda). Проведён: пофайловое чтение всех 31 модуля `prism/*.py`,
сверка с эталоном `docs/ref-math.md` (31 критерий), прогон тестов
(`tests/test_operator_invariants.py`, `test_transport_kinetic.py`,
`test_audit_core_v2.py`, `test_cache.py` и др.), проверка off-by-one/отсчётов,
проекций, guards и скоростей. Методология та же, что у аудита
`experiments/` (чистка 2026-11-09): найденное фиксируется, однозначные баги
чинятся в том же коммите, неоднозначное — на решение пользователя.

**Примечание о коммитах**: фиксы P2-1/P2-2/P3-1 и этот док попали в коммит
`bbab642` параллельной сессии (общее дерево, широкий `git add`); сообщение
того коммита их не описывает. Отдельный коммит `85f4e69` несёт только
PLAN.md (к моменту коммита файлы уже были в `bbab642`).

## P1 — «не та сущность» / расхождение с ref-math (НЕ чинится без решения)

### P1-1. `cheap_invariants.c_pred` — подпись обещает «изотропный предиктор c»,
формула — спектральная плоскостность P

- Модуль-докстринг: «adds the isotropic predictor of c:
  `c_pred(p) = tr P / sqrt(K · tr P²)`».
- Что это на самом деле: `(tr P)/‖P‖_F · 1/√K` — коэффициент выравнивания
  собственного спектра P (по К–Б по вектору собственных значений
  `(tr P)² ≤ K·‖P‖²_F`, равен 1 тождественно только при `P ∝ I` —
  недостижимо для `P(p)`: у него нулевое собственное значение). Это объект,
  зависящий только от p, и он НЕ предиктор c.
- Настоящий изотропный предиктор: для изотропного J (iid-вхождения ИЛИ
  ортонормальные строки) `c = ⟨J,PJ⟩/‖J‖² = tr P/K` (вывод:
  E[J_ij J_kj] = σ²δ_ik → числитель σ²·trP, знаменатель KDσ²; ортонорм.
  строки: tr(PJJᵀ)/K = trP/K — то же).
- Численно (p из кэша exp27b, K=128, Σp² = 0.4686): `c_pred_код = 0.158`,
  `trP/K = 0.0041`, измеренные c (alive) 4.8e-4…2.1e-3 — в 2–8× от trP/K,
  в 70–330× ниже c_pred_код.
- В E-40-доке (`docs/models/corr/3-4.md`) A := K·tr(PG)/(trP·trG) = c/(trP/K),
  A=1 ⟺ изотропия. Ядро `A_alignment = c/c_pred_код` = A из дока делить на
  √(K/trP²) ≈ 38 (на реальном p). Разные объекты, не масштабы.
- `tests/test_operator_invariants.py::test_c_pred_and_A` фиксирует только
  тавтологию (`A == c/c_pred`, `0 < c_pred ≤ 1`) — семантику не проверяет.
- **Решение пользователю**: либо формула → `trP/K` (совпадение с E-40-доком и
  с «изотропный предиктор»), либо переименовать/переподписать функцию в
  спектральную плоскостность P. Не чинить без решения — меняет значение
  опубликованного объекта A_alignment.

### P1-2. `docs/ref-math.md`, строка `gauge_share (ESG)` — в спектральной
формуле нет 1/K

- ref-math: «`Σ_k (u_k^T 1)²λ_k / Σ_k λ_k`» — с **ненормированным** 1.
- Ядро всюду с 1/K: `spectral.gauge_share` (uniform = 1/√K, т.е.
  `(u_kᵀuniform)² = (u_kᵀ1)²/K`), `p_algebra.gauge_share = 1ᵀG1/(K·tr G)`,
  paper `g_uni = ‖Jᵀ1‖²/(K‖J‖²_F)`. Тождество O2≡O3 закреплено
  (max расхождение 7.7e-16) именно при 1/K.
- Та же строка ref-math даёт top-1 proxy `λ₁·uniformity(u₁)/trace`, где
  `uniformity(u) = (uᵀ(1/√K))²` — 1/K там ЕСТЬ. Формула противоречит
  собственному proxy в той же строке.
- Как записано, формула даёт значение в K раз больше ядра.
- Не чинится (граница аудита: docs/ не трогать кроме этого дока). Фикс:
  добавить 1/K (или записать через `u_uni = 1/√K`).

## P2 — расходившиеся guard/флаги (исправлено в этом коммите)

### P2-1. `behavior.KL_lin` — фиксированный допуск 1e-8 (audit3_math п. 2Б
не добит по параллельному пути)

`p_algebra._check_p` уже исправлен на адаптивный `max(1e-8, K·eps32)`
(≈3.5e-6 при K=128) после замеров: fp32-строки кэша несут |Σp−1| до 2.8e-7
(медиана 4.0e-8, p99 1.7e-7), фиксированный 1e-8 отвергал 88 % реальных
позиций. `KL_lin` остался со старым фиксированным 1e-8 и отбрасывал те же
строки. **Фикс**: `KL_lin` теперь проходит p через тот же `_check_p`
(единый код-путь, заодно получает проверки finiteness/отрицательных/ndim).

### P2-2. `kinetic.channel_energy_matrix` — нет проверки `sigma_perm_mismatch`

`kinetic_decomposition` считает `permU != permV` (макс. совпадение колонок
U- и V-фреймов даётся разными перестановками) и помечает вывод «ambiguous».
`channel_energy_matrix` (5×5-грамма каналов) переставляет `S1r` по
U-перестановке (`S1r = S1[:r][fU["perm"]]`, M2), а V-фрейм выравнивает своей
собственной перестановкой, которую **выбрасывает**, и сверки permU == permV
нет. При почти вырожденном верхнем спектре σ каналы dSig (U-порядок) и N
(V-порядок) тихо смешиваются. Замыкание `sum(G) == ‖ΔJ‖²` при этом
выполняется (Пифагор по разложению ΔJ = embed(B) + A5), поэтому ассерт
`exp40_5_kinetic.py` (tol 1e-10) это НЕ ловит. **Фикс**: fail-loud
`ValueError` при `permU != permV` (та же семантика «ambiguous», что в
`kinetic_decomposition`; молча возвращать смешанную грамма — не вариант).
Сценарий проверен сконструированным случаем (см. «Проверки фиксов»).

## P3 — docstring (одна строка, исправлено)

### P3-1. `interventions.py:154` — «KL(p_base ‖ p_edit)»

Сводная строка пишет один порядок операндов, код
(`F.kl_div(log_p_base_full, log_p_edit_full)` = Σ p_edit·log(p_edit/p_base))
и детализация в строке 159 («FULL-vocabulary KL(p_edit ‖ p_base) … operand
order as the Kaggle kernel») — другой. **Фикс**: сводная строка →
`KL(p_edit ‖ p_base)`. Код (published E-16a) не тронут.

## P4 / скорость — только к акту, не тронуто

- `layer_selection.select_pairs` — ветка
  `if (n_L-1, n_S-1) not in pairs: pairs.append(...)` недостижима:
  n_S−1 ∈ selected всегда, и `mapping[n_L−1] = n_S−1` ровно (монотонность
  round(i·(n_S−1)/(n_L−1)), максимум — на последнем l_L). Вредless
  belt-and-suspenders, баг не является — не удалять.
- `exp38_hi_common.load_model_tensors` — хардкод абсолютного HF-пути
  (`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots`) +
  `sorted(glob)[0]`: хрупок при нескольких снапшотах/ревизиях.
  Экспериментомодуль E-38h/i, практический потолок — к акту.
- `cache.py:421` (`bootstrap_ci`) — Python-цикл по B=1000 выборкам:
  `np.array([statistic(s) for s in samples])`. Для дефолтного
  `statistic=np.median` вся стадия векторизуется одним
  `np.median(samples, axis=1)` (10–50× на этом шаге); доминирует цикл
  перестановок — чинить, когда bootstrap станет горячим.
- `jacobian._run_decoder_pass` / `_run_decoder_to_norm` — дублирующийся
  хвост декодера (разница в последнем шаге: norm vs +lm_head). Работает,
  не трогать.

## Проверено чисто

**Off-by-one / отсчёт**
- `hidden_states[l+1]` (0 = embeddings) — конвенция консистентна во всех
  5 местах `jacobian.py` (`_extract_hidden_state`, оба пути
  `compute_hidden_jacobian`, `_forward_h_norm`).
- prefix = позиции 0..S−2 (`input_ids[:, :-1]`); позиция последнего токена
  S−1; `start = layer_idx + 1` в обоих декодерах.
- `last_position_logits` — документ-гард от двойного счёта последнего
  токена (кэш полного forward'а вместо prefix'а).
- `cache.iter_position_pairs`: adjacent t = 0, stride, … < S−stride;
  strided — все t < S−stride; all — все пары. Границы верны (S=8, stride=1:
  7 пар).
- `select_pairs`: nearest preimage, tie → меньший l_L (строгое `<` сохраняет
  первый).
- B1 (`interventions`, h_states[L+1]) — исправлен и задокументирован.

**Математика vs ref-math**
- Замкнутые формы: `tr P = 1−Σp²`, `‖P‖²_F = Σp²−2Σp³+(Σp²)²`, P PSD и НЕ
  проектор (граница p=e_i — задокументирована), спектр — корни
  секкулярного уравнения с измеренным потолком (11.4 % позиций —
  неразрешимые в fp64 полюса → отказ, не баг; fallback `eigvalsh`),
  `λ_max ≤ p_max`.
- L23-факторизация `J(t) = (√D/r_t)·M_K(I−û_tû_tᵀ)`, `M_K = W_lm·diag(γ)`:
  верифицирована прямым индексным расчётом (γ — hidden-размерность,
  масштабирует **столбцы** W — `W * gamma[None, :]`; в
  `exp38_hi_common.factor_residual` `P * gamma[:, None]` — `diag(γ)P`,
  строки — тоже верно).
- `J_norm` RMSNorm (`r·diag(1+w)·(I−xhat·xhatᵀ/D)`) и LayerNorm — по
  ref-math; оценка на выходе нормировки (ошибка ~0.87) задокументирована.
- `softmax_jacobian = (diag p − ppᵀ)J` — идентично строкам
  `compute_hidden_jacobian_rows` / `bridge_rows_backward`.
- `bridge_objective` = метрика (audit4 M1): оптимизируется и
  отчитывается одна и та же функция; chunk-инвариантность (нормировка
  снаружи) задокументирована.
- O2≡O3 (g_uni ≡ ESG gauge_share) — закреплено (7.7e-16); invariants —
  21 идентичность + negative controls + Jensen pin.
- κ: `κ² = 2r − 2tr H` безусловно; `procrustes_gap` — ловушка
  транспонированного оптимума задокументирована (audit B5: 0.499 vs 0.405);
  SVD берётся по ПРЯМОМУ произведению `Q2ᵀQ1` — правильно.
- `principal_angles` — `_check_frame` принудительно ортонормальность
  (audit M10: без неё клип SVD-значений в [0,1) продаёт не-косинусы).

**Проекции**
- `topology.project_to_subspace`: квадратная база — явный ValueError
  (двусмысленность), не тихий выбор; `project_to_pca_subspace` — numpy-only
  по проекту (не на `top_r_svd`, чтобы не тащить torch).
- `jstruct.v_pred_from_JT` — вырождение ‖Jᵀu‖ < eps → RuntimeError
  ([M-1], β≈0), не NaN.

**CI / guards («не в доверительном интервале»)**
- `deterministic_mode` — полное сохранение/восстановление всех RNG
  (torch, cuda all, numpy, random); strict — опционально (GDN-кейсы).
- `permutation_test`: p = (1+#{null ≥ obs})/(B+1), fail-loud на опечатку в
  `alternative` (audit M7); `resolution_floor` — full-size bootstrap
  (audit M5: без замены m=n — перестановка, разброс тождественно 0).
- `json_utils` — строгий RFC-8259: `allow_nan=False` + `default=_raise`
  форсируются, `np.bool_` → TypeError, атомарная запись (tmp + replace).
- `_deep_clone_cache` — guard длины последовательности (audit Q12:
  `assert` убивает `-O`; здесь явный raise); GDN in-place — клон на входе
  ядра (read object ≠ written object), реентерабельно (счётчик).
- `h_override` — контракт device/dtype/shape, clone+detach до каста.
- `exp38_hi_common.load_snapshots` — signed cosine
  `|⟨s,Wo⟩|/(‖s‖‖Wo‖) < 0.5`: старый guard (норма матричного произведения)
  никогда не срабатывал (≈1/√D и ≈1/√256 оба «правильны»), новый
  разделяет «ΔW» (0.02–0.26) от «W_orig не вычтен» (≈+1).
- `SVDCache` — полная верификация на put (dtype/shape/уменьшающиеся σ/
  ортонормальность U и V), атомарные записи, LRU + диск; `MetricCache` —
  только Python float (np.float32 отклоняется — guard точности).

**Скорость**
- `_compute_jacobian_rows` — ОДИН batched backward (grad_outputs=eye(K),
  is_grads_batched) вместо K backward: 3–15× (эмпирическое свойство
  transformers 5.15, при поломке — runtime error, не тихая деградация).
- `bridge_rows_backward` / `_compute_jacobian_rows_matrixfree` — chunked
  (A4, OOM-фикс v22): пик = chunk_size+1 вторых графов, не K; per-chunk
  градиенты ≡ одному backward полной взвешенной потери.
- h-реюс (v23.1): KL-forward'у из exp17 не пересчитывается полный forward.
- ESG-граммианы — один BLAS matmul (A = J перестановкой → A @ Aᵀ), не
  einsum по позициям.
- Кэши: jcache (по слоям), gramian-LRU (cap 8 — под E-27.5 grid),
  SVDCache, MetricCache (bulk-опы без агрегатов); atomic writes везде.
- `bootstrap_ci` — индексы векторизованы (B, n) одним `rng.integers`.
- `noise.calibrate_noise_floor` — повторный прогон метрики (детерминированный
  кэш → пол ≈ 0, честно).

## Проверки фиксов (AGENTS.md чек-лист: designed-сценарий guard'а
протестирован exec, не глазами)

- P2-1: p из fp32-кэша (|Σp−1| = 2.8e-7) теперь проходит `KL_lin`;
  обрезанная строка (Σp = 0.49) всё ещё отбрасывается с понятной ошибкой.
- P2-2: сконструированный случай permU ≠ permV (J_t = diag(2,1),
  J_t1 с переставленными U-столбцами, V без изменений) → `ValueError`;
  контрольный случай permU == permV → грамма возвращается, замыкание
  `sum(G) == ‖ΔJ‖²` (tol 1e-10).
- P3-1: текст; код не тронут.
- Прогон: `tests/test_operator_invariants.py`, `test_transport_kinetic.py`,
  `test_audit_core_v2.py`, `test_cache.py`, `test_bridge.py`,
  `test_bridge_rows_backward.py` — зелёные.

## Не тронуто (границы)

- `experiments/`, `kaggle_kernel/` — аудитировались/не входят в ядро.
- `docs/` кроме этого дока: P1-2 (ref-math 1/K) — только к акту.
- `c_pred` (P1-1) — ждёт решения пользователя, значения A_alignment не
  менялись.
