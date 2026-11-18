# AT2e (T2.2b) — пре-рег r2: каноническая клетка T2 — подъём отношения O_H^fut/(C+εI) per (l,t)

Теория: docs/AT2e_theory.md. Ответ на ревью теории: docs/AT2e_response.md (приняты все 4 дефекта,
D1–D3 с правками; r2 — ответ на ревью TZ r1: 5 блокеров B1–B5 + 5 мажорных M1–M5, все приняты,
см. ниже). Ответки проекта на теорию: 12 инвариантов theory §VII + 5 из response/r2
(guard окружения, search-effort-matched референс, companion-гейт на абсолютный q̄_fut, единый
ансамбль B₀/референс, знаменатель выше пола) = **17 инвариантов** (§7).

Контент T2: ∃v, (l,t): направление слабо видно текущему ответу (q_t мало), но сильно видно будущему
(q̄_fut велико) — скретчпад-гипотеза в канонической форме аддентума T2. Стадия 1 — existence pilot:
нижняя граница λ_max|_B + подъём отношения в полном пространстве (response §0: ширина проб не имеет
мощности найти максимизатор; поиск перенесён из ширины в глубину). Кандидат ≠ «скретчпад доказан»
(теория §IX): на стадии 1 запрещены формулировки «скретчпад доказан» / «T2 закрыта».

Ключевые конструкции (решения, зафиксированные ДО кода):
(1) **Один патч → все считывания** (теория §III): h_l^t ← h_l^t + a·v, ‖v‖=1; логиты на s = t..t+H;
    q_s = Δz_sᵀ P_s Δz_s, P_s = diag(p_s) − p_s p_sᵀ (ПОЛНЫЙ вокабуляр, точная Fisher, инвариант 3);
    числитель и знаменатель из ОДНОГО forward (инвариант 2).
(2) **Подъём вместо ширины** (response §0): B₀ = 32 направления (8 top-PCA + 8 bottom-PCA +
    16 gQR per-cell seed; r2 B3: Адамар/Уолш исключён) — стартовое подпространство и сертификат
    нижней границы; дальше 8 итераций градиента обобщённого Рэлея (∇λ ∝ (vᵀ(C+εI)v)·Ov − λ(C+εI)v,
    ε = 1e-2·ε₀ — r2 B2) в ПОЛНОМ R^1024. Итерация = 2 FD-forward + 1 grad-fwd + 2 backward
    (vector-leaf: Ov — все k≥1 сразу, Cv — k=0; замер 6.5 ф.-э.).
    C-extremes ИЗ кэша ИЗ B₀ УБРАНЫ (response §3): кэш не нужен для эстиманда вовсе; C-extremes L12 —
    report-only reference (мост к AT2c/AT2d, не гейт).
(3) **Референс search-effort-matched** (response §0/§4): первичное сравнение — rank-тест: v* против
    N=64 случайных направлений (те же амплитуда, P, клетка; p_cell = 1/(N+1) при H0). Вторичное — null
    random-weights × real text с ИДЕНТИЧНОЙ процедурой поиска (условная стадия 1b). Сравнение
    max поиска со средним случайных ЗАПРЕЩЕНО (порядковая статистика форсирует гейт, инвариант 14).
(4) **Эстиманд** (теория §II, dose-free нормировка — правка r1):
    λ_ε(v; a, τ) = H·q̄_fut(v;a) / (q_t(v;a) + a²·τ·λ_max(C_t|_B)),  τ ∈ {1e-3, 1e-2, 5e-2},
    q_s(v;a) = ‖P_s·(a·J_s v)‖² ∝ a²; a = ε_a·‖h_t‖, ε_a ∈ {3e-4, 1e-3} (linear-режим) +
    probe {3e-3, 1e-2} (nonlinearity, report-only). a²-масштаб в знаменателе обязателен: иначе
    λ зависит от a через ε/a² и «согласие λ между амплитудами» бессмысленно (правка r1).
(5) **Слои L11 (full) + L12 (GDN)** (response §3): соседние глубины (12-vs-11 слоёв выше) —
    type-контроль чистый; L23 — G-zero (архитектурный нуль, детектор протечки окна).
(6) **Клетки = 16**: {L11, L12} × {wiki/0, math/0, code/0, chat/0} × {t=32, t=96}.
    Базлайны = 4 forward (shared по слоям/позициям; output_hidden_states → PCA + logits → P_s).
(7) **Split-half по k** (теория §IV/V): первичное — подбор v* на окне {k=1,2}, валидация на {k=4,8};
    ветка SHORT_HORIZON_ONLY (response §2): окно {1,2} — FOUND, валидация {4,8} провалена →
    SHORT_HORIZON_ONLY (не NOT_FOUND). Обратное (поиск {4,8} → валидация {1,2}) — secondary отчёт.
(8) **Вердикты стадии 1**: per-cell NOT_TESTABLE / FOUND / SHORT_HORIZON_ONLY / REDUNDANT /
    POSITION_GENERIC / NOT_FOUND; global SCRATCHPAD_CANDIDATE / REDUNDANT_DOMINANT / DEAD_WEIGHT /
    NOT_FOUND_IN_SUBSPACE / NOT_RESOLVED. SCRATCHPAD_CANDIDATE (из ≥2 разных доков, r2 B4) →
    условная стадия 1b (null) → LEARNING / ARCHITECTURE / NULL_NOT_TESTABLE.

Thread-pinning OMP/OPENBLAS/MKL=4 + torch 4 тр. ДО импортов (бит-идентичность форвардов, как в
AT2b/AT2c/AT2d). CPU, eager, **fla отключён guard'ом** (инвариант 13 — без него краш: triton 3.8.0
без CPU-driver, fla 0.5.x CUDA-only, прямой forward падает в triton GDN-l2norm).

Режимы:
  AT2e_SMOKE=1 : smoke (1 док wiki/0, L12, t=32: B₀-пробы на 2 амплитудах, подъём 2 итерации,
                 верификация, random N=8, VJP↔FD-проверка, G-zero, floor, PCA/gQR, ε₀, rank,
                 двойной прогон md5) — ≈ 200 forward ≈ 15 мин;
  (без флагов) : main-прогон at2e_s1_r1 (trackio; стадия 1 ≈ 5.4k ф.-э. ≈ 6.4 ч CPU; стадии 1b/1c —
                 условные/отчётные, в том же прогоне ≈ +1.9 ч максимум; worst case ≈ 7.0k ф.-э. ≈ 8.4 ч);
  --reanalysis : диск-реанализ frozen raw-артефакта БЕЗ форвардов (если понадобится).

---

## 0. Клетка одной строкой

Патчим ОДНО состояние в ОДНОЙ позиции (h_l^t += a·v), читаем логиты на t..t+8 с полной точной
Fisher P = diag(p)−ppᵀ; ищем направление, максимизирующее ОТНОШЕНИЕ будущей видимости к текущей
(подъём обобщённого Рэлея, не сэмплирование хвоста), сравниваем с search-effort-matched референсом
(rank против 64 случайных + идентичный поиск на random-модели), пер- (l,t), с линейным гейтом по
амплитудам, гейтом здоровья знаменателя (знаменатель выше численного пола) и split-half по k.
Дешёвая стадия даёт нижнюю границу: «найдено» — существование
установлено; «не найдено» — «не найдено в B₀+подъёме», НЕ «нет».

## 1. Входы

| вход | путь / константа | гейт |
|---|---|---|
| модель | /data/Qwen/Qwen3.5-0.8B (Qwen3.5-0.8B, 24 слоя, D=1024, 248320 vocab) | S1 (см. ниже) |
| doc-пул | /data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt; 4 домена × 8 × 128 int64; используем si=0: wiki/0, math/0, code/0, chat/0 | S1-sha: sha256 = ded634f7d16cf33a9bb565de5cbc3faa288cfc90b15cdfa3e42365695fec4e03 |
| kэш L12 (report-only cext_ref) | /data/prism_kernels/exp27b/results/exp27b_jcache_L12.pt | S1-sha: sha256 = 976079ce1d3c84c7016a236683f86f5fee1ce6571ff6edd53cf4f6f9ac228a24 |
| U128 (только для cext_ref-меток) | из jcache L12 (128 vocab-id) | — |
| SEED | 0 (directions, random ref, bootstrap); SEED_NULL = 1234 (random-init null) | фикс |

S1: модель загружается from_pretrained, torch.float32, attn_implementation="eager"; ноль кастомных
весов; sha-гейты на 2 файловых входах ДО любого форварда; провал = стоп, артефакта нет.
**Кэш J (jcache) НЕ используется для никаких эстимандов** (инвариант 10) — только jcache_L12 для
report-only C-extremes reference (мост к AT2c/AT2d; метка cext_ref = direction-only bridge, НЕ
сравниваемо с AT2d U128-pooled gbar — разный P).

Доки: 4 (по одному на домен, si=0 из пула) — power стадии 1 декларирован как поиск (rank-тест),
НЕ как CI по докам (4 юнита — bootstrap по докам бессмыслен; кросс-док-генерализация =
консистентность вердиктов по 16 клеткам, §5).

## 2. Базисы (B₀ и направления)

B₀ per cell (32 направления, ‖·‖=1, все f64-считаются, f32-применяются):
- **8 top-PCA**: SVD центрированной матрицы hidden-состояний слоя L на 128 позициях дока
  (128×1024, из baseline-forward output_hidden_states — 0 доп. форвардов); top-8 главных направлений.
- **8 bottom-PCA**: последние 8 из 127-мерного sample-пространства (ранк ≤ 127; «bottom» — произвольный
  базис sample-null-space). Метка обязательна: **sample_null_space**, НЕ «каноническое
  низковариантное подпространство» (response §2).
- **16 gQR (per-cell seed)**: RandomState(SEED + cell_id).standard_normal(16, 1024) → reduced QR,
  ортонормальные направления = строки R (Q (16,16) — НЕ направления). Метка gqr_cellseed.
  **r2 (B3): строки Sylvester–Hadamard (Уолш) исключены**: (a) ансамбль отличается от гауссового
  референса — нарушена обмениваемость rank-теста при H₀; (b) фиксированность по всем 16 клеткам —
  коррелированные траектории и FOUND, FWER-декларация при независимости неверна; (c) метка «Haar»
  ложна (AT2c/AT2d — настоящий QR-Хаар, другой объект под одним именем). Семейство ансамбля
  СОВПАДАЕТ у случайной части B₀ и референса (§3.6) — обмениваемость при H₀.

Свойства: B₀ зависит от (дока, слоя, per-cell seed) — position-специфичность через PCA +
отсутствие межклеточной фиксированности через per-cell gQR (r2: фиксированность Адамара снята).
Каждое направление бинортемуализуется (Gram–Schmidt на случай вырождения, флаг rank_deficient).
Состав B₀ пишется в артефакт (f64-байты + sha256 + gqr_seed per cell) — инвариант 11: «не найдено»
отчитывается как «не найдено в ЭТОМ B₀+подъёме».

C-extremes (report-only, L12, t=32, 4 дока): top-8 + bottom-8 ненулевого спектра K=128-рестрикта
из kэша L12 — 16 направлений на (док, t); λ оценивается 2 FD-forward на направление (128 forward).
Метка: cext_ref_L12_U128cache. НЕ гейт, НЕ эстиманд, не входит в вердикты.

## 3. Пайплайн: baseline, патч, считывание, поиск

### 3.1 Baseline (per док, 1 forward, no_grad)
- logits z⁰ [128, 248320] (полный вокабуляр);
- p_s = softmax(z⁰_s) для всех s (для P_s и P_t);
- hidden-состояния L11, L12 на всех 128 позициях (PCA, ‖h_t‖);
- ‖h_t‖ per (L, t) — для относительных амплитуд (инвариант 12);
- κ_base-проверка: det — двойной baseline-прогон md5 (дет-гейт, §7 G_det).

### 3.2 Patch-forward (одна позиция, центральные разности)
Патч: выход слоя L в позиции t: out[:, t, :] += a·v (in-place на clone — autograd-верно, проверено:
inplace ≡ functional-хук точно, pre-study r1). a = ε_a·‖h_t‖ (инвариант 12).
Центральные: z⁺ = F(+a·v), z⁻ = F(−a·v) → Δz_s^{cent} = (z⁺_s − z⁻_s)/2, s = t..t+H
(инвариант: центральные, точность O(a²)). **Критичная индексация (r1 self-audit): текущий ответ =
logit в позиции t (предсказывает токен t+1) — s = t, k = 0; будущее = s = t+k, k = 1..H.**
J_s = ∂z_s/∂h_l^t: J_0 (позиция t — текущий, знаменатель), J_k, k≥1 (будущее, числитель).
Из z⁺, z⁻, z⁰ (бесплатно из тех же forward):
- Δz_s⁺ = z⁺_s − z⁰_s (one-sided, для κ и floor-оценок), Δz_s^{cent} = (z⁺_s − z⁻_s)/2 (для q);
- q_s = Δz_s^{cent}ᵀ P_s Δz_s^{cent}, P_s = diag(p_s) − p_s p_sᵀ (полный вокабуляр, f64-аккумуляция);
- κ_s = KL(softmax(z⁺_s) ‖ softmax(z⁰_s)) (f64); линейный флаг: |2κ_s − q_s|/max(q_s, floor_q) > 0.5
  (инвариант 4: q И κ отчитываются оба; κ ≈ ½q в линейном режиме — классический результат
  pullback-Фисхера через softmax);
- q_t (знаменатель) и q_{t+k}, k=1..8 (числитель) — из ОДНОГО forward-пара (инвариант 2).

### 3.3 Матрицы из проб (per cell, из B₀-прохода)
Для каждого направления b ∈ B₀ и всех s: J_s b — из 2 forward (центральные, s = t..t+H, k = 0..8).
Тогда (k-нумерация: k = s − t):
- [O_H|_B]_{jj'} = Σ_{k=1..H} (J_k b_j)ᵀ P_{t+k} (J_k b_{j'}) / a² (f64) — будущие k≥1;
- [C_t|_B]_{jj'} = (J_0 b_j)ᵀ P_t (J_0 b_{j'}) / a² (f64) — ТЕКУЩИЙ (k=0, позиция t);
- ε₀(cell) = λ_max(C_t|_B) — из 32×32 eigh (f64); τ-свип ε = τ·ε₀, τ ∈ {1e-3, 1e-2, 5e-2} (инвариант 5);
- **подъём (r2, B2): ε_ascent = τ_primary·ε₀ = 1e-2·ε₀** — тот же объект, что выбор v₀ и вердикт
  (поиск и эстиманд согласованы; ε₀ без τ максимизирует vᵀOv, а не отношение);
- нижняя граница: λ_max|_B = max обобщённого RQ на B₀ (32×32 обобщённая задача, f64) — отчёт per cell;
- per-k профиль q_k per направление (k = 0..8) — бесплатно из тех же forward.

### 3.4 Подъём (ascent, per cell)
- v₀ = argmax_{b∈B₀} λ_ε(b; a₁, τ=1e-2) (первичная τ — середина свипа; выбор зафиксирован).
- Итерация i = 1..8 (v — единичный, полный R^1024):
  1. 2 FD-forward (±a₁·v) → J_s v, s = t..t+H (k = 0..8; a₁ = рабочая амплитуда: primary 1e-3,
     fallback 3e-3 — r2 B1);
  2. VJP через vector-leaf (замер r1): ОДИН grad-forward (патч = 1024-dim лист d,
     requires_grad) + 2 backward (retain_graph):
     backward₁: loss = Σ_{k=1..8} c_kᵀ·z_{t+k} → d.grad = Σ_{k=1..8} J_kᵀ c_k = Ov
     (c_k = P_{t+k}(J_k v), 248320-dim, плотный — из baseline-логитов);
     backward₂: loss = c_0ᵀ·z_t → d.grad = J_0ᵀ c_0 = Cv (k=0 — знаменатель C);
     итерация = 2 FD-fwd + 1 grad-fwd + 2 bwd ≈ 8.2 + 3.8 + 16.1 ≈ 28 s ≈ 6.5 ф.-э.
     (замер r1, стабилен по повторам; оценка response «20 ф.-э.» 3× консервативна —
     vector-leaf сворачивает 9 VJP в 2 backward; эстиманд не меняется: те же Ov, Cv).
  3. ∇λ ∝ (vᵀ(C_t+εI)v)·Ov − λ·(C_t+εI)v, **ε = τ_primary·ε₀ = 1e-2·ε₀** (r2 B2: тот же ε, что в
     v₀ и вердикте); **адаптивный нормированный шаг (r2 M2)**: g = ∇λ; v ← normalize(v + η·g/‖g‖);
     η₀ = 0.2; после каждой итерации: λ_i < λ_{i−1} → η/=2 (мин 0.01); λ_i > λ_{i−1} →
     η = min(0.5, 1.5·η); ‖g‖ < 1e-8 → ранний стоп (ascent_converged). Детерминированно,
     0 доп. forward (λ_{i+1} — из прохода следующей итерации, amortized);
     флаг ascent_not_converged если |λ₈−λ₇|/max(λ₇,1e-9) > 1e-3;
  4. λ(i) вычисляется из J_k v того же прохода (amortized — 0 доп. forward).
- v* = argmax_{i} λ(v_i) по траектории (траектория целиком в артефакте: v_i (f64 sha), λ_i, ‖Δv‖).

### 3.5 Верификация v* (per cell, 4 forward)
- 2 амплитуды (ε_a = 3e-4, 1e-3 — первичная сетка, **подтверждена pre-meas B1, pre-study r2 §8**;
  **fallback-сетка {1e-3, 3e-3} — правило: медиана по B₀ q_t(a₂)/floor < 100 → сдвиг сетки
  ДО main; при активации: L1/L2 → отчёт, линейность судится κ-флагами при 1e-3,
  3e-3 — на границе нелинейности, явное признание (r2 B1)**) × центральные (2 forward) → q_t, q̄_fut,
  λ_ε(τ) per τ per ε_a;
- **linear-гейт L1** (активный режим): |λ(3e-4) − λ(1e-3)|/λ ≤ 0.10 (согласие λ — правка r1, response §2 добавка);
- **linear-гейт L2** (активный режим): slope(q) = log(q(1e-3)/q(3e-4))/log((1e-3)/(3e-4)) ∈ [1.9, 2.1];
- **fallback-режим (если активирован, r2 B1)**: L1/L2 → отчёт; линейность — κ-флаги при a₁:
  FOUND требует отсутствие срабатывающих κ-флагов (|2κ_s−q_s|/max(q_s,floor) ≤ 0.5 для всех
  s = t..t+H) — единственный оставшийся линейный тест на границе нелинейности;
- **denom-гейт v\* (r2 B1)**: q_t(v*; a₂) > 10·floor_q → ρ осмыслен (FOUND/REDUNDANT по ρ-rank, §5);
  q_t(v*) ≤ 10·floor_q → флаг **denom_below_floor**: L1/L2 → отчёт, вердикт — rank-тест + companion +
  split-half + специфичность (сигнатура скретчпада: невидим в текущем — ниже шума);
- companion-гейт (инвариант 15): q̄_fut(v*) > max(10·floor_q, 1e-12) — абсолютный, иначе λ_ε велик
  может быть ε-артефактом при микро-числителе (response §2 добавка к 3);
- κ-флаги per k (инвариант 4); доля ε в знаменателе a²·τ·ε₀/(q_t + a²·τ·ε₀) per τ — отчёт (r2 M1,
  показывает режим: q_t-доминирование vs ε-доминирование).

### 3.6 Random-референс (per cell, 128 forward)
- N = 64 гауссовых направления: RandomState(SEED + cell_id + 1000).standard_normal(64, 1024) →
  reduced QR, строки R — то же семейство ансамбля, что случайная часть B₀ (r2 B3: обмениваемость
  при H₀; per-cell seed — независимость клеток);
- per направление: q_k per k (все k), λ_ε(τ) per τ, ρ (при q_t > 10·floor), per окна {1,2}, {4,8};
- **rank-тест (первичный, инвариант 14)**: per τ: rank_τ(v*) среди {v*, r_1..r_64} по λ_ε(τ);
  p_cell(τ) = 1/(N+1) = 1/65 при H0 (v* неотличим от случайного);
  **FWER (r2 B4, пересчёт)**: P(rank-1 где-то в 16 клетках | H₀) = 1−(1−1/65)^16 ≈ 22% — это
  per-cell-аларм-рейт, НЕ FWER глобального гейта. FWER глобального гейта (≥4 клетки FOUND/SHO,
  FOUND ≥2, И из ≥2 разных доков, §5) ≤ 2.0%: per-док p = 1−(1−1/65)^4 ≈ 6.01% (верхняя граница
  при независимости внутри дока), P(≥2 из 4 доков) = 1 − (1−p)⁴ − 4p(1−p)³ ≈ 2.0%;
- effort-match: N=64 ≈ полное число оценённых направлений в поиске (32 пробы + 8 шагов + верификация)
  (response §0).

### 3.7 Split-half (per cell, 0 доп. forward)
- Первичное: поиск v* на окне K1 = {k=1,2} (λ_{ε}^{K1} = 2·q̄_{k∈{1,2}}/(q_t + a²ε)); валидация:
  rank по λ_{ε}^{K2}, K2 = {k=4,8}, против тех же 64 random (их λ^{K2} — бесплатно из 3.6);
- rank-условия split-half (полное per-cell определение FOUND/SHO с companion/scratchpad/
  специфичностью — §5):
  FOUND (полный) = rank-1 ≥2/3 τ на K1 И rank-1 ≥2/3 τ на K2;
- SHORT_HORIZON_ONLY = rank-1 ≥2/3 τ на K1, провал на K2 (response §2);
- secondary отчёт: обратное (поиск {4,8} → валидация {1,2}) — 8 итераций на подмножестве 4 клеток
  (L11/L12 × 2 дока, t=32) — 4×8×6.5 = 208 ф.-э. (0.25 ч) — входит в стадию 1 как отчётный блок.

### 3.8 G-zero (L23, 2 forward)
Патч выхода слоя 23 в позиции t (±a₁, wiki/0, t=32): после layers[23] только per-position
final-norm + head → логиты t+k (k≥1) НЕ зависят от позиции t (свойство графа, не эвристика —
теория §VI). Гейт: max_k ‖Δz_{t+k}‖ ≤ 1e-6 (дет-нуль; протечка окна/пропагации).

### 3.9 Числовой пол (floor_q, 8 forward)
a_floor = 1e-9·‖h_t‖ (Δz ниже fp32-разрешения логитов ~4e-6): q(a_floor) = уровень fp32-шума.
8 forward: 4 дока × 2 слоя (t=32). floor_q = max по 8 (per cell берётся тот же floor — консервативно).
**r2 (M4): floor_q := max(floor_measured, q_quant), q_quant = (2⁻²³·max|z_t|)²·p_max** — при
a_floor центральная разность двух fp32-логитов часто побитово равна → floor_measured = 0, и
companion-гейт не вырождается в хардкод 1e-12, а опирается на явную границу квантования
(пер-компонентный ulp-шум, распространённый через Var_p ≤ (макс. ошибка)²·p_max).
Companion-гейт (§3.5) против max(10·floor_q, 1e-12).

### 3.10 Positional specificity (со-первичное условие для позитивных ветвей, r2 M5, ≤32 forward)
Для каждой клетки, прошедшей rank (K1) + companion: тот же v* в другой позиции того же дока
(t=32↔96), 2 forward (±a₁): **pos_spec_ratio = λ^{K1}(v*; t')/λ^{K1}(v*; t) ≤ 0.5 — условие
FOUND и SHORT_HORIZON_ONLY** (r2 M5: именно этот контроль различает «скретчпад в точке» и
generic-направление модели; при B4 (≥2 дока) он — со-первичный, отчётного статуса недостаточно).
Ratio > 0.5 → ветка **POSITION_GENERIC** (направление — свойство модели, не точки; отчёт;
null стадии 1b различает обучение/архитектуру). Отчитывается также all-k ratio (отчёт).
Цена: только для клеток-кандидатов; worst case 16 × 2 = 32 forward.

### 3.11 C-extremes reference (report-only, L12, t=32, 128 forward)
16 kэш-направлений per док (top-8+bottom-8 nonzero-C, K=128-рестрикт) × 2 forward: λ_ε(τ), ρ, q.
Мост к AT2c/AT2d: «вот λ у старых армов, вот λ у найденного v*». Метка cext_ref_L12_U128cache
(направление из U128-кэша, P — полный вокабуляр AT2e — НЕ сравниваемо с AT2d gbar по числам).

## 4. Контроли

| контроль | где | цена |
|---|---|---|
| ε_a-свип / linear-гейты L1, L2 + режимы (r2 B1) | §3.5 | 2-я амплитуда в B₀-проходе (1024 ф.-э. уже в смете) |
| κ-флаги (2κ ≈ q) | §3.2 | 0 (из тех же forward) |
| split-half по k ({1,2}→{4,8}) | §3.7 | 0 (из тех же forward); secondary 208 ф.-э. |
| random direction, effort-matched (N=64, gQR-семейство, r2 B3) | §3.6 | 2048 ф.-э. |
| random weights × real text (null) | стадия 1b, УСЛОВНАЯ | 1136 ф.-э. (1.4 ч) |
| позиционная специфичность (со-первичный для FOUND, r2 M5) | §3.10 | ≤32 ф.-э. (для клеток-кандидатов) |
| тип слоя (L11 full vs L12 GDN) | design | 0 (оба в сетке) |
| G-zero L23 | §3.8 | 2 ф.-э. |
| детерминизм (двойной прогон md5) | smoke + main (подмножество) | smoke: в smoke; main: 8 forward на финальном чек-блоке |
| множественные рестарты подъёма | НЕ в стадии 1 (теория: стадия 2) | декларировано |
| температурный sham | НЕ в стадии 1 (функциональный тест — стадия 3) | декларировано |

## 5. Дерево вердиктов (стадия 1)

### Per-cell (16 клеток)
- **NOT_TESTABLE**: провал линейных гейтов (L1/L2, активный режим) ИЛИ denom-гейта клетки
  (медиана по B₀ q_t(a₂) ≤ 10·floor_q) ИЛИ companion-гейта (q̄_fut(v*) < 10·floor) →
  «не тестирована», НЕ refuted (теория §V.0, r2 B1).
- **FOUND**: companion OK И rank-1 ≥2/3 τ (K1) И rank-1 ≥2/3 τ (K2) И линейность (активный режим:
  L1 OK; fallback-режим: κ-флаги не срабатывают, §3.5)
  И [ρ(v*) > p90_ref OR q_t(v*) ≤ 10·floor_q (denom_below_floor)] И pos_spec_ratio ≤ 0.5.
  p90_ref — перцентиль ρ(v*) в распределении ρ 64 random-референсов (r2 B5: референс измеренный,
  не prior; scratchpad-семантика: невидим в текущем ИЛИ аномально читаем в будущем).
- **SHORT_HORIZON_ONLY**: companion OK, rank-1 ≥2/3 τ (K1), провал K2, те же scratchpad-условия
  [ρ > p90_ref OR denom_below_floor] И pos_spec_ratio ≤ 0.5 (response §2 — эффект живёт только на
  k=1..2; t+1 — тоже будущее; слабый, но содержательный исход).
- **REDUNDANT**: companion OK, rank-1 ≥2/3 τ (K1), q_t(v*) > 10·floor_q И ρ-rank(v*) ∈ [p10_ref, p90_ref]
  (r2 B5: сейчас видно + будущее доминирует, отношение типичное — redundancy с видимым каналом,
  не скретчпад. Generically ρ<1 (затухание ‖J_k v‖) — ветка достижима, в отличие от prior «ρ≈1»).
- **POSITION_GENERIC**: companion OK, rank-1 (K1) ≥2/3 τ, pos_spec_ratio > 0.5 (r2 M5 — свойство
  модели, не точки; не учитывается в глобальном кандидате; отчёт).
- **NOT_FOUND (резидуальная ветка)**: companion OK и ни одна позитивная ветка не накрывает —
  включая rank > 1 (все τ) на K1 и rank-1 с отсутствием scratchpad-сигнатуры
  (ρ-rank < p10_ref при q_t > 10·floor: сейчас видно, будущее аномально низко — не скретчпад).

### Global (по 16 клеткам)
- **SCRATCHPAD_CANDIDATE**: ≥4 cells FOUND/SHORT_HORIZON_ONLY (FOUND ≥2) И **из ≥2 разных доков**
  (r2 B4: внутридоковая репликация — не генерализация; наши 4 дока = 4 домена (wiki/math/code/chat),
  значит ≥2 дока ⇒ автоматически ≥2 домена). → запускает стадию 1b (null) в том же прогоне.
- **REDUNDANT_DOMINANT**: 0 FOUND, и ≥8 клеток REDUNDANT (per-cell, выше; r2 B5: ρ-rank по
  измеренному референсу, ноль доп. форвардов — ρ случайных уже считается).
- **DEAD_WEIGHT**: ≥8 клеток с q_t И q̄_fut ниже пола для всех v ∈ B₀ ∪ {v*} → в B₀ нечего нести.
- **NOT_FOUND_IN_SUBSPACE**: 0 FOUND, пол/мощность достаточны, паттерны REDUNDANT/DEAD/POSITION_GENERIC
  не накрывают → «не найдено в B₀+подъёме»; T2 НЕ закрывается; стадия 2 (полный подъём на подвыборке,
  рестарты) — отдельное ТЗ (теория §VIII).
- **NOT_RESOLVED**: иначе.

### После стадии 1b (условно)
- **LEARNING**: null НЕ воспроизводит (λ_null(v*_null) rank > 1 ИЛИ q̄_fut ниже пола) → сигнал
  свойство обучения.
- **ARCHITECTURE**: null воспроизводит (λ_null rank-1 ≥2/3 τ при сопоставимых отн. статистиках)
  → сигнал свойство архитектуры.
- **NULL_NOT_TESTABLE**: отклик null в шумовом поле (пред-замер ‖h_t‖/‖PΔz‖ ниже пола; масштаб
  ε_a_null до 0.1 включительно; всё равно ниже пола) — честный исход, не «спектра нет» (response §4.2).

Запрет (теория §IX, инвариант): на стадии 1/1b НЕТ «скретчпад доказан» / «T2 закрыта».
Кандидат ≠ скретчпад: нужны адресуемость (стадия 3), устойчивость, полный null.

## 6. Масштаб и смета (замеры pre-study r1, 18.11)

Замеры (Qwen3.5-0.8B, CPU, 4 тр., eager, fla-guard):
- forward (no-grad, 128 tok): **4.33 s**; peak RSS 5.12 GB.
- VJP (1 backward через 24 слоя, full-vocab логиты): **8.41 s = 1.94× forward**; peak RSS **10.92 GB**.
  Pre-budget гейт response §5: VJP ≤ 3× forward И RSS ≤ 20 GB → **ПРОЙДЕН**, идём в подъём.
- VJP↔FD: rel err 0.2–0.9% (a=1e-3, full-v + 3 базисных направления, /tmp/at2e_vjp_diag.log).
- **VJP vector-leaf (рабочий паттерн подъёма, замер r1, /tmp/at2e_vjp3.log)**: dense 9-позиционный
  cotangent — backward = **8.9 с = 1.94× forward** (стабильно: 7.9–8.9 с по повторам,
  warm-up-эффекта нет); retain_graph-двойной backward = 16.1 с; RSS_peak = **13.9 GB**
  (гейт 20 GB пройден); итерация подъёма = 2 FD + 1 grad-fwd + 2 bwd ≈ 28 с ≈ **6.5 ф.-э.**
- **VJP↔FD при рабочей амплитуде a=1.78e-3** (/tmp/at2e_vjp4.log): v-направление: Ov rel 0.22%,
  Cv rel 4.2%; e_0: Ov 0.39%, Cv 9.3%; e_512: Ov 22.8% (|x|=2.5e-4 — FD-шум на микро-величине,
  не ошибка VJP: VJP точен по autograd). Профиль: величина > 0.03 → rel ≤ 4%; микро-величины —
  FD-шумовы. Определяет гейт S5 (§11).
- JVP↔FD при unit-амплитуде: 1–6% (нелинейность при больших a — ожидаемо).
- ‖h_t‖ (живой замер r1, все 4 дока × L11/L12 × t∈{32,96}):
  L11: wiki 1.777/2.199, math 2.053/1.770, code 2.298/2.316, chat 2.061/2.348;
  L12: wiki 1.763/2.255, math 1.943/1.836, code 2.237/2.158, chat 2.166/2.270
  (t32/t96). Оба слоя ~1.8–2.4 → a = ε_a·‖h_t‖: при ε_a=1e-3 → a ≈ 1.8e-3–2.4e-3 (оба слоя).
  Кэш: L6 1.62–12.85, L18 4.33–8.10, L19 4.24–8.92, L22 5.85–11.80 (контекст глубины).
- ‖J_1 v‖ (full logits, unit v): L12 45–51, L19 19–20; P-метрика (полный вокабуляр):
  q_t(a=1e-3) ~ 1e-5..1e-6 порядок (замеры r1) — согласуется с r2 (max q_t по B₀ при a=5.3e-4:
  9.7e-7; ×(1.76/0.53)² ≈ 1.1e-5 при a=1.76e-3, top-PCA направления).
- **Pre-meas B1 (r2, pilot-клетки wiki/0 t=32, a = 3e-4·‖h_t‖, 32 направления B₀ per клетка,
  /tmp/at2e_premeas_b1.log)**: L11 (‖h‖=1.763): q_t min/med/max = 3.4e-8/8.2e-8/9.7e-7;
  L12 (‖h‖=2.161): 2.5e-8/9.4e-8/8.0e-7. floor (a_floor=1e-9·‖h_t‖, центральные): замер
  4.6e-13/7.5e-13, квант-граница (2⁻²³·max|z_t|)²·p_max = 6.3e-13 → floor_used = 6.3e-13/7.5e-13
  (r1-оценка «~1e-11» — верхняя; прямой замер r2 в 1.5 порядка ниже, запас 5+ порядков при любом
  из значений). q̄_fut при a_small: L11 1.7e-10..1.3e-9, L12 2.8e-10..1.3e-9 — все 32 направления
  обоих слоёв: q_t > 10·floor И q̄_fut > 10·floor (frac = 1.00).
  **Решение по правилу B1: медиана q_t(3e-4)/floor = 1.30e5 (L11) / 1.25e5 (L12) ≫ 100 →
  первичная сетка {3e-4, 1e-3} ПОДТВЕРЖДАЕТСЯ, fallback НЕ активируется.**
- triton 3.8.0: CPU-driver отсутствует; fla 0.5.x — CUDA-only; без guard'а прямой forward крашится
  (triton GDN-l2norm) — инвариант 13 обязателен.

### 6b. Окружение исполнения (T4 main, CPU fallback)

- **Main (стадия 1 + условные 1b/1c) — Kaggle T4**: kernel через `kaggle_run.sh` (push→poll→pull),
  fp32, eager, fla-guard (инвариант 13 — тот же численный путь, что CPU; на T4 triton/fla могли бы
  работать — их избегаем осознанно для консистентности пути).
- Модель в kernel'е: `huggingface_hub.snapshot_download("Qwen/Qwen3.5-0.8B")` (паттерн kaggle_at16);
  интернет в kernel включён.
- **Пулы доков — константой в скрипте** (12.9 KB; 4 дока × 128 tok) + sha-гейт на встроенные токены
  (те же sha, что §1). Jcache L12 (2 GB) **не грузится на GPU** → cext_ref вынесен из T4-main в
  CPU-follow-up (report-only; исполняется, если кандидат найден).
- **T4-canary ДО main** (10–15 мин, тот же код, wiki/0, t=32, ОБЕ клетки L11+L12): (1)
  retain-двойной backward на каждой клетке — полный подъёмный проход (2 FD + 1 grad-fwd + 2 bwd,
  dense 9-позиционный cotangent); **VRAM_peak max ≤ 12 GB — явный fallback-порог (r2 M3):** CPU-
  замер с градиентом = 13.9 GB RSS; 16 GB T4 — впритык; порог оставляет 4 GB запаса; провал =
  OOM-риск в середине main после часов работы; (2) детерминизм на GPU — двойной forward md5;
  (3) wall-цена forward/backward → фиксация T4-бюджета. Провал canary (VRAM > 12 GB ИЛИ det) →
  CPU fallback (бюджет §6 выше, один ночной прогон, watchdog; avail ≥ 10G до лонча).
- **trackio в kernel НЕ пишется** (правило проекта: локальная БД, лишний канал отказа) — JSON-лог с
  маркерами (stages, started_at/finished_at, forward_s/vjp_s замеренные в canary), бэкилл
  `experiments/trackio_migration.py` после pull. Числа — из лога артефакта, не из trackio.
- Worst case wall: T4 ≈ 30–90 мин (canary-замер); CPU ≈ 8.4–12 ч (ночь).

Смета стадии 1 (ф.-э. = forward-эквиваленты, 4.33 s CPU; VJP = 1.94× учтён в итерации):

| блок | расчёт | ф.-э. |
|---|---|---|
| baseline | 4 дока × 1 | 4 |
| B₀, ε_a=1e-3 | 16 кл. × 32 × 2 | 1024 |
| B₀, ε_a=3e-4 (linear-гейт) | 16 × 32 × 2 | 1024 |
| подъём (vector-leaf, замер r1) | 16 × 8 ит. × 6.5 | 832 |
| верификация v* (2 амплитуды) | 16 × 4 | 64 |
| random-референс (N=64) | 16 × 64 × 2 | 2048 |
| split-half secondary (4 кл.) | 4 × 8 × 6.5 | 208 |
| cext_ref (L12, t32, 4 дока × 16 × 2) | 128 | 128 |
| G-zero + floor | 2 + 8 | 10 |
| positional specificity (максимум) | 16 × 2 | 32 |
| **итого стадия 1** | | **5374 ≈ 6.4 ч CPU** |

Стадия 1b (null, УСЛОВНО — только при SCRATCHPAD_CANDIDATE): 4 клетки × (32×2×2 + 8×6.5 + 4 +
64×2 = 284) = **1136 ≈ 1.4 ч** (арифметика response §5 исправлена: 420→284/клетка при
vector-leaf подъёме; в response «≈360» — опечатка, 128+160+4+128 = 420).
Стадия 1c (nonlinearity-probe, отчёт): 4 кл. × 2 доп. амплитуды (3e-3, 1e-2) × 32 × 2 =
**512 ≈ 0.6 ч**.

**Worst case: ≈ 7022 ф.-э. ≈ 8.4 ч CPU** — либо один ночной прогон через run_one.sh,
либо **Kaggle T4** (§6b: canary → main ≈ 30–90 мин wall).

## 7. Инварианты (17; «оптимизация» любого = другая клетка)

1. **Одна позиция, не окно** (theory §VII.1). Окно = отклик системы, не частная производная.
2. **Числитель и знаменатель — из одного forward-пара** (theory §VII.2).
3. **Полный вокабуляр, точная P = diag(p) − ppᵀ** (theory §VII.3). U128-pooled / диагональ —
   другой объект.
4. **q (оператор) и κ (отклик) отчитываются оба** (theory §VII.4); флаг |2κ−q|/max(q,floor) > 0.5.
5. **ε относительно λ_max(C_t|_B), свип τ ∈ {1e-3, 1e-2, 5e-2}; подъём — при ε = 1e-2·ε₀ (primary
   τ), тот же объект, что v₀ и вердикт** (r2 B2: поиск и эстиманд согласованы; theory §VII.5 +
   response §4.4: λ_max(C|_B) — нижняя граница полного λ_max(C), отчитывается как таковая).
6. **ρ нормировано на H (mean, не sum)** (theory §VII.6); **распределение ρ референса
   ИЗМЕРЕННО (N=64), не prior** (r2 B5: prior «ρ≈1» снят — generically ρ<1 из-за затухания,
   ветка REDUNDANT на нём недостижима).
7. **Подбор и проверка v* разделены** (split-half, theory §VII.7).
8. **Teacher forcing, не свободная генерация** (theory §VII.8).
9. **l ≤ 22; L23 только как нуль-гейт** (theory §VII.9).
10. **Кэш — только для выбора проб/report-only; ни одно вердиктное число из кэша** (theory §VII.10;
    в r1: кэш вообще не нужен для эстимандов — только cext_ref report-only).
11. **Ограничение на span(B₀) отчитывается как нижняя граница**, состав B₀ в артефакте
    (theory §VII.11). «Не найдено» ≠ «нет».
12. **Амплитуда относительная к ‖h_t‖** (a = ε_a·‖h_t‖, theory §VII.12).
13. **Guard окружения: `sys.modules["fla"] = None` ДО импорта transformers** (response §6).
    Без guard'а forward крашится (triton 3.8.0 без CPU-driver, fla 0.5.x CUDA-only).
    Инвариант окружения, не опция: определяет численный путь и саму исполнимость.
14. **Референс search-effort-matched** (response §6). Сравнение max поиска со средним случайных
    запрещено (порядковая статистика форсирует гейт). Rank-тест (N=64) + null с идентичным поиском.
15. **Companion-гейт на абсолютный q̄_fut выше численного пола** (response §6). Иначе большое λ_ε
    может быть ε-артефактом при микро-числителе.
16. **Случайная часть B₀ и референс — один ансамбль (gaussian QR, per-cell seed)** (r2 B3).
    Адамар/Уолш исключён: расхождение ансамблей (обмениваемость rank-теста при H₀), межклеточная
    фиксированность (корреляция FOUND, FWER), ложная метка «Haar».
17. **Знаменатель выше пола** (r2 B1/M4): клетка testable только при медиане по B₀ q_t(a₂) > 10·floor_q;
    для v* — q_t(v*) > 10·floor_q либо режим denom_below_floor (линейные гейты → отчёт).
    floor_q = max(замер, (2⁻²³·max|z_t|)²·p_max) — явная граница квантования при degenerate-замере.

Центральные разности + f64-аккумуляция квадратичных форм — часть инварианта 2/3 (точная P).
Thread-pinning (OMP/OPENBLAS/MKL=4 ДО импортов, torch 4 тр.) — дет-условие инварианта 1 (как в AT2d).

## 8. Pre-study r1 (замеры ДО фиксации сметы; НЕ гейты)

1. forward = 4.33 s, RSS 5.12 GB (no-grad); VJP = 8.41 s = 1.94×, RSS 10.92 GB (с градиентом)
   → pre-budget гейт response §5 пройден.
2. VJP↔FD rel err 0.2–0.9% (full-v, j∈{0,512,1023}) — reverse-mode через модель верен;
   «расхождение 0.0975 vs −0.1354» из первого замера — баг test-harness'а (dвойное применение
   vt: (JᵀPw)·(vt⊙vt) вместо (JᵀPw)·vt), НЕ баг VJP.
3. In-place патч (out.clone(); out[:,t,:] += d) autograd-верен: inplace ≡ functional-хук точно
   (оба −0.136166) — патч-конструкция AT2d достаточна для градиентов.
4. JVP (forward-mode) через GDN+full-attn работает на torch-пути (guard): rel err 1–6% при
   unit-амплитуде (нелинейность); при a~1e-3 — линейно.
5. ‖h_t‖ (все 4 дока × L11/L12 × t∈{32,96}): оба слоя ~1.8–2.4 (таблица в §6);
6. ‖J_1 v‖ full-logits: L12 45–51, L19 19–20; P128-субсет (128 id дока): ‖J_1ᵀPw‖ ~ 4.2.
   Полный-вокабулярный P (diag(p)−ppᵀ) — q_t(1e-3) ~ 1e-5..1e-6 (замер в смоке, floor-сверка).
7. triton 3.8.0 CPU-driver отсутствует (backends: amd/nvidia); fla 0.5.x ops — triton-ядра;
   без guard'а — RuntimeError «0 active drivers» на GDN-l2norm. AT2d-форварды работали ТОЛЬКО
   благодаря guard'у (sys.modules["fla"]=None в шапке AT2d.py).
8. Vector-leaf VJP (рабочий паттерн): итерация 6.5 ф.-э., dense cotangent 1.94×, RSS 13.9 GB
   (замеры в §6). **Урок test-harness'а (2 бага, оба пойманы в r1)**: (а) FD-forward без no_grad
   тащит графы параметров в c_k → чужие графы освобождаются чужим backward → «Trying to backward
   through the graph a second time»; (б) hook без remove в цикле → старые листья цепляются в новые
   графы. В коде эксперимента: **FD-пары — всегда no_grad; c_k — .detach(); hook — remove в
   finally после каждого forward** (входит в само-аудит кода).
9. Промежуточный замер «dense backward 25–28 с (6×)» — артефакт бага (а), не свойство torch:
   чистый dense backward = 7.9–8.9 с = 1.94×, стабилен по повторам.
10. **Pre-meas B1 (r2, ДО фиксации амплитудной сетки)**: pilot-клетки wiki/0 t=32, L11+L12,
   32 направления B₀ (8 top-PCA + 8 bottom-PCA + 16 gQR) при a = 3e-4·‖h_t‖ + floor при
   a_floor = 1e-9·‖h_t‖ (по 67 forward на клетку: 32×2 + baseline + 2 floor). Результат: все 32
   направления обоих слоёв выше 10·floor (по q_t и по q̄_fut); медиана/floor = 1.3e5/1.25e5 ≫ 100
   → первичная сетка {3e-4, 1e-3} подтверждена, fallback {1e-3, 3e-3} не активируется (логика
   fallback остаётся в TZ как правило; срабатывание = пересмотр сетки ДО main). Лог:
   results/AT2e/prestudy_r2/at2e_premeas_b1.log (pct-поля в лог-хедере считались по
   неотсортированному массиву — баг pre-study-скрипта, починен; канонические перцентили — от
   sorted-списка в том же логе).

## 9. Ожидания (не гейты)

- Если T2 жива: v* найдётся на ≥4 клетках (ранние/средние слои, позиции вне «горячих» t),
  q_t ниже пола или близка (denom_below_floor), λ_ε — экстремальный ранг (≤ 1/65) при
  τ=1e-2..5e-2, ρ(v*) > p90_ref или denom_below_floor, split-half держит (эффект на k=2..8),
  pos_spec_ratio ≤ 0.5. (r2 M1: абсолютный порог «λ_ε ≥ 3–5» убран — в q_t-доминируемом режиме
  λ → H·ρ, порог не интерпретируем без указания режима; rank-тест масштабно-свободен; режим
  виден из отчёта доли ε в знаменателе per cell per τ).
- Если T2 мертва: λ_max|_B ≈ референс (rank > 1), REDUNDANT_DOMINANT (ρ-rank ∈ [p10,p90] по
  референсу, r2 B5) или DEAD_WEIGHT.
- null: GDN random-модель — контрактивность (g = −exp(A_log)·softplus < 0, ~0.5/step) → отклик
  может лечь в пол; пред-замер обязателен (response §4.2); full-attention (L11) — безопаснее.
- C1b-эффект (позиционная зависимость): bottom-PCA B₀ должен давать лучшие ранги на t=32/96,
  чем stationary (проверка через per-(d,t) PCA).

## 10. Что НЕ делается (scope стадии 1)

- Ни одного «скретчпад доказан» / «T2 закрыта» (запрет theory §IX).
- Стадия 2 (полный подъём с рестартами на подвыборке) — только если NOT_FOUND_IN_SUBSPACE,
  отдельное ТЗ.
- Стадия 3 (функциональная адресуемость, argmax-флипы, температурный sham, полный null) —
  только при SCRATCHPAD_CANDIDATE, отдельное ТЗ.
- Open-loop генерация (T7), солитон (T8) — другие клетки.
- Кросс-док bootstrap CI (4 юнита — бессмысленно); генерализация = консистентность 16 клеток.
- Новые кэши; multi-GPU/multi-node (T4-один — main-окружение, §6b); multi-seed trained
  (1 seed; power декларирован через rank-тест).

## 11. Смоки (AT2e_SMOKE=1; до main, отдельный шаг)

1 док (wiki/0), L12, t=32 (≈ 200 forward ≈ 15 мин):
- S1: sha-гейты (pool + jcache_L12) + модель загрузилась + guard активен (import fla → None).
- S2: baseline det — двойной forward md5 логитов (дет-гейт).
- S3: B₀ конструирование — 32 направления, unit-norm (max отклонение < 1e-9), Gram–Schmidt
  флоги, метки top_pca/bottom_pca/sample_null_space/gqr_cellseed (+ gqr_seed per cell, r2 B3);
  B₀-пробы на 2 амплитудах (32×2×2 = 128 forward).
- S4: ε₀ = λ_max(C_t|_B) > 0 (f64); нижняя граница λ_max|_B вычислена (32×32 eigh).
- S5: VJP↔FD при рабочей амплитуде a₁: полный v-направление (SEED), rel_err ≤ 0.05 для Ov И Cv
  (гейт смока; pre-study r1: 0.22–4.2%, /tmp/at2e_vjp4.log; микро-направления e_j НЕ гейтятся —
  FD-шум на |x| < 0.03).
- S6: подъём 2 итерации (адаптивный нормированный шаг, r2 M2) — траектория λ не NaN, v единичный,
  ‖Δv‖ > 0, η-траектория отчёт.
- S7: верификация — linear-гейты L1/L2 вычисляемы (значения отчёт; активный/fallback-режим по
  pre-study r2); **denom-гейт: медиана по B₀ q_t(a₂)/floor вычислима и > 1 (отчёт значения,
  r2 B1)**; companion-гейт: q̄_fut(v*) > max(10·floor_q, 1e-12) вычислим;
  floor_q = max(замер (1 forward, a_floor), q_quant = (2⁻²³·max|z_t|)²·p_max) (r2 M4)
  и floor_q < q(a₁) (иначе NOT_TESTABLE-режим задокументирован).
- S8: random N=8 (16 forward) — rank ∈ [1, 9] вычислим; распределение ρ референса вычислено
  (p10/p50/p90, для гейтов §5, r2 B5).
- S9: G-zero (2 forward) — max_k ‖Δz_{t+k}‖ ≤ 1e-6 (гейт).
- S10: κ-флаги — |2κ−q|/max(q,floor) вычислим (значения отчёт).
- S11: двойной прогон подмножества артефакта (B₀-пробы + верификация) — md5 идентичны (дет-гейт).
- S12: JSON-артефакт валиден (numpy-типы/NaN sanitize), schema-покрытие, статус ok.
Провал любого = артефакта нет, стоп, фикс + пересмоук.

## 12. Провенанс

- Модель: Qwen/Qwen3.5-0.8B локальная (/data/Qwen/Qwen3.5-0.8B), фп32, eager; версия
  transformers зафиксирована в config артефакта (pip freeze-строка).
- Пул: exp27b_pool_ids_L23.pt (sha в §1) — те же доки, что AT2b/AT2c/AT2d (pool_fp cross-check
  с kэшем L12: совпадение pool_fp — гейт S1b).
- Kэш L12 (cext_ref только): exp27b_jcache_L12.pt (sha в §1).
- Замеры r1: /tmp/at2e_vjp_test.log, /tmp/at2e_vjp_diag.log, /tmp/at2e_vjp3.log, /tmp/at2e_vjp4.log,
  /tmp/at2e_jnorm.log, /tmp/at2e_jvp_test2.log, /tmp/at2e_l11_hnorm.log, /tmp/at2e_vjp_fd.log
  (копии в results/AT2e/prestudy_r1/ при коммите).
- Замеры r2 (pre-meas B1: q_t(3e-4) распределение по B₀, pilot-клетки): at2e_premeas_b1.log
  (копия в results/AT2e/prestudy_r2/).
- AT2d: raw-артефакт results/AT2d/AT2d.json + AT2d_v2.json (REFUTED, frozen) — контекст, не вход.

## 13. Схема артефакта (results/AT2e/AT2e_s1_r1.json)

```
header: {run, run_revision r1 (TZ r2), script_sha, seed, device=cpu|t4-kaggle, model,
         transformers_version,
         grid_mode: primary|fallback (r2 B1),
         canary: {vram_peak_gb per layer, det_md5_ok, fwd_s, bwd_s} (если T4), started_at,
         finished_at, status, log_md5, forward_s=4.33, vjp_s=8.41}
gates: {S1_sha_pool, S1_sha_jcache, S1b_pool_fp, G_det (md5 baseline double),
        G_zero (max|Δz|), G_floor (floor_q per (d,L): {measured, quant, used}),
        G_vjp_fd (rel err per j), G_denom per cell {median_b0_qt_over_floor, vstar_qt_over_floor},
        G_linear per cell {L1, L2, mode: active|fallback, kappa_flags}}
config: {eps_a: [3e-4, 1e-3] (primary) | [1e-3, 3e-3] (fallback), taus: [1e-3, 1e-2, 5e-2],
         tau_primary: 1e-2, H_windows: [[1,2],[4,8],[1..8]], N_random: 64,
         ascent: {iters 8, eta0 0.2, eta_min 0.01, adaptive: halve/double x1.5, gnorm_stop 1e-8,
                  tol 1e-3}, a_floor_rel: 1e-9, floor_quant: 2^-23*max|z_t|^2*p_max (M4),
         gqr: {b0_seed_offset: 0, ref_seed_offset: 1000, per_cell_seed: SEED + cell_id},
         pos_spec_threshold: 0.5 (M5), rho_gates: {p10, p90} по референсу (B5),
         pool_sha, jcache_sha, layers [11,12,23], positions [32,96], docs [wiki/0,math/0,
         code/0,chat/0]}
cells: per (L, dom, t) {
  hnorm, floor_q {measured, quant, used},
  b0: {sha256 (f64), gqr_seed, per direction: {label: top_pca|bottom_pca|sample_null_space|gqr,
         q_k [8], lambda_tau [3], windows {K1, K2, all}}},
  eps0, lambda_max_B0 (нижняя граница),
  ascent: {traj: [ {i, v_sha256, lambda, delta_v_norm, eta} x8 ], v0_source,
           eps_ascent (= 1e-2*eps0, B2), convergence_flag},
  vstar: {sha256, lambda_tau [3], q_t, q_bar_fut, rho (при q_t>10*floor),
          rho_rank_ref (percentile, B5), denom_eps_share_tau [3] (M1),
          denom_below_floor (flag, B1),
          linear {L1, L2, mode}, kappa_flags [9] (k=0..8), companion_ok,
          rank_tau {K1: [3], K2: [3]} (первичный + валидация),
          windows {K1: lambda_tau, K2: lambda_tau, per_k [8]}},
  random_ref: {N=64, sha256 seeds, max_lambda_tau [3], rank_vstar per tau,
               distribution: [min, p25, med, p75, max] per tau,
               rho_distribution: [p10, p25, med, p75, p90]},
  pos_spec: {t_prime, lambda_K1_tprime, ratio, spec_ok} (для клеток-кандидатов, M5),
  verdict: NOT_TESTABLE|FOUND|SHORT_HORIZON_ONLY|REDUNDANT|POSITION_GENERIC|NOT_FOUND
}
cext_ref: per (dom, t=32) {per direction: lambda_tau, rho, q_t; label cext_ref_L12_U128cache}
nonlin_probe: per (4 кл., 2 доп. амплитуды): q(a) curve [3e-4, 1e-3, 3e-3, 1e-2] per v* + B0-subset
null (если 1b): {seed_null, hnorm, response_floor_check, per cell: как cells,
  relative: {lambda_ratio_trained_vs_null, eps_scale_factor, split_half_rho},
  verdict: LEARNING|ARCHITECTURE|NULL_NOT_TESTABLE}
verdicts: {per_cell: {...}, global: SCRATCHPAD_CANDIDATE|..., final: (если 1b) ...}
manifest: {script_sha, log_sha, artifact_sha, trackio_run, started_at, finished_at}
```

Двухпроходный артефакт (raw + manifest, как AT2d r6): raw пишется первым, manifest вторым,
md5-сцепка; маркер AT2E_MAIN_DONE — в лог-файл (не в stdout).

## 14. Коммит-план

1. `AT2e: TZ r1 (pre-рег) + pre-study r1 замеры` — **сделан (2a5c652)**.
2. `AT2e: TZ r2 (ревью r1: B1-B5, M1-M5) + pre-study r2` — docs/AT2e_prereg.md +
   results/AT2e/prestudy_r2/ (q_t(3e-4) pre-meas: решение по амплитудной сетке).
3. (после ревью-лупа, PASS) `AT2e: код r1 + smoke PASS` — experiments/at2/AT2e.py + smoke-логи +
   smoke-артефакт (CPU).
4. `AT2e: T4 canary PASS` — kaggle kernel canary (обе клетки, retain-двойной backward, VRAM ≤ 12 GB)
   + лог + VRAM/det-числа в pre-study-папку.
5. (после main) `AT2e: main r1 результаты` — results/AT2e/ + AT2e_results.md + trackio-бэкилл
   (T4: JSON-лог; CPU: trackio как обычно).
