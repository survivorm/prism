# AT2d (T2.2) — пре-рег r4: причинная (интервенционная) state-клетка

**Статус: ТЗ r4 — на внешнее ревью.** История ревизий: r1 — внутренний
черновик (воркер умер на output-limit без файла/маркера, зафиксировано).
r2 — внешнее ревью (4 блокера + M5–M9 + m1–m8) → r3 — редизайн
(ответка docs/AT2d_review_r2_answers.md, P0-замеры §8). **r3 — внешнее
ревью (docs/AT2d_review.md, раунд 3, sha b394ef5457bbbd6f): 5 блокеров
(B1–B5) + M1–M8 — PRINATO по всем; ответка docs/AT2d_review_r3_answers.md;
P0'-замеры (disk-only) — §8.** r4 — редизайн по ревью r3: residual-аппарат
(ĉ/excess_res/S-energy) снят, заменён per-doc амплитудным матчингом +
температурным sham; REFUTED — non-superiority; L23 — G-zero; окно —
вариант A (63 значения); gbar — полная конструкция.

Связь с цепочкой T2: T2.1 (readout r1, floor-dominiрована) → T2.1b (AT2b,
SUBSPACE_CONTROL_FAIL: intercept-конфаунд per-doc базисов) → T2.1c (AT2c,
NOT_RESOLVED, control_ok=True: **H1 readout-часть отклонена** — weak-tail хуже
max(pca,haar) на всех (L,k)≥1; top доминирует readout'ом на всех (L,k);
G̸-проба §11.7: истинные слабо-наблюдаемые направления ≤ floor на k≥1) →
**T2.2 (этот документ) — НЕ readout-клетка, а интервенция**: zero-ablation
J-компонент state h_l и замер ΔCE на БУДУЩИХ позициях — тест «weak-сторона
causally используется моделью для предсказания будущего, даже если не
линейно читаема».

## 0. Клетка одной строкой (r4)

Тот же T2-контур (32 дока × 128-префикс, слои L∈{6,12,18,22} (GDN) + {19}
(full attention), стационарный J-базис из кэша exp27b), но:
(а) **P_future-хук (вариант A)**: абляция ТОЛЬКО на позициях t<64 (0..63),
метрика ΔCE ТОЛЬКО по позициям 64..126 (цели 65..127, **63 значения**) —
позиция 63 (последняя абляционная) в метрику не входит; бины target-индексные
65-77/78-90/91-103/104-116/117-127; конвенция: «позиция» = logit-позиция
(предсказывает токен t+1), target = позиция+1 (ревью r3 B5 — off-by-one r3);
(б) **первичный sham = per-doc гауссов ортонормальный k-фрейм randG** (каждый
док — свой фрейм: draw-дисперсия в doc-bootstrap, ревью r2 B1);
(в) **первичные эстиманды (оба ко-первичные, per-doc)**:
  **excess_match(арм, док) = ΔCE(арм, док) − ΔCE(randG, тот же док)** —
  все армы α-матчнуты к dose sham'а (post-norm ‖δn‖², НЕ raw E — ревью r3 M1/B2;
  «matched» = БАЗОВЫЙ randG-ран α=1, dose совпадает by construction, §3);
  **excess_cal(арм, док) = ΔCE(арм, док) − [CE(τ*·z_B) − CE(z_B)]** —
  температурный sham: τ* решает H(softmax(τ*·z_B)) = H(p_arm) per-doc, окно
  64..126; считается из B_plain/arm-логитов самого прогона, 0 доп. форвардов
  (ревью r3 M3); ĉ/energy-фит/субпоулинг армов **нет** (ревью r3 B1/B2);
(г) арм weak_Gbar (CANONICAL G̸-базис, **полная конструкция** на всех 4096 (d,t)
— ревью r3 M4) — в сетке, secondary branch; verdict-арм — weak_Gagg
(по dose согласован с randG — P0': dose 0.11–1.32 vs 0.11–1.58).
(д) **L23 = G-zero** (P_future: assert |ΔCE| ≤ 1e-6 — архитектурный нуль,
детектор протечки окна) + L23-P-контроль (direction specificity, ревью r3 B4).

## 1. Входы

1. **Кэш exp27b** (как в AT2c, SHA-гейт S1 — 7 файлов, дословно таблица
   AT2c-пре-рега §7): `exp27b_jcache_L{6,12,18,19,22,23}.pt` +
   `exp27b_pool_ids_L23.pt` (AT2d использует L6/12/18/22/19 + pool_ids;
   L23 — G-zero + P-контроль §4; L19 — в кэше с JR+P128, проверено 17.11).
   **P128_RAW** (субстохастический row полного распределения; нужен для
   KL_pred, M1) есть в кэшах L6/12/18/22, **в L19-кэше отсутствует**
   (старый ckpt; P128_RAW идентичен по слоям — из baseline-логитов; spot-check
   L12 vs L6: max|Δ| = 0.0, P0') → для L19 берётся из L6-кэша, зафиксировано.
2. **Токены 32 дока**: `exp27b_pool_ids_L23.pt` → `ids[dom][si]` (32 × 128
   int, forward-порядок i=0..127). Pool-верификация: `meta.pool_fp` (per
   domain sha256[:12]) сверяется с `pool_fp`, сохранённым в jcache-ckpt
   (гейт G2b). Пул НЕ реконструируется из потока (load_calibration_data/pin
   не вызывается — токены из кэша дословно; закрывает риск недетерминизма
   Hub-стрима).
3. **Модель**: `/data/Qwen/Qwen3.5-0.8B` (24 слоя, hidden 1024, V = 248320,
   гибридная: linear_attention (GDN) + full_attention каждый 4-й слой;
   layer_types из config.json: full на 0-based {3,7,11,15,19,23}, остальные
   linear — **AT2d-слои: L6/12/18/22 = GDN, L19 = full attention** (M9).
   **После layers[23] — только final RMSNorm (per-position) + LM head
   (per-position, tied embeddings): позиционного смешения нет** ⇒
   ΔCE_P_future(L23) ≡ 0 бит-в-бит (основа G-zero, §4). Загрузка ТОЧНО как
   `experiments/e21_30/exp27b_dense.py` (собрал кэш): `AutoModelForCausalLM.
   from_pretrained(..., torch_dtype=float32, attn_implementation="eager")`.
   **fla на этой машине НЕ установлен** → GDN-слои идут torch-reference
   (на T4, где собран кэш, был fla 0.4.2) — расхождение учтено гейтом G2c.
4. **Окружение**: thread-pinning OMP/OPENBLAS/MKL_NUM_THREADS=4 +
   `torch.set_num_threads(4)` ДО импортов numpy/torch (дословно AT2c/AT2b;
   бит-идентичность форвардов на 4 тр. верифицирована замером: max|Δlogits|
   = 0 между повторами). .venv: python 3.13, torch 2.13.0+cu130,
   transformers 5.15.1, без fla.

## 2. Базисы (для ablation-проекций)

Все базисы — ортонормированные наборы направлений в пространстве state
(1024), вычисленные ИЗ КЭША (без новых форвардов), J-базис дословно по
AT2c-коду (experiments/at2/AT2c.py: строки 440–484; G32/HAAR 366–371/431):

- **Jagg** = mean_{d,t} JR_{d,t} ([128,1024], f64); **Gagg** = JaggᵀJagg
  ([1024,1024], rank ≤ 128); `torch.linalg.eigh`; order по убыванию λ;
  nz = {i: λ_i > 1e-12·λ_max} (ненулевые, по возрастанию).
- **top-k**: первые k по убыванию (k=4: λ-индексы 0..3; k=8: 0..7).
- **weak-k (VERDICT-арм, конвенция AT2c)**: первые k из nz по возрастанию
  (слабейшие ненулевые, weak-tail — конвенция T2.1/AT2b/AT2c).
- **pca-k** (per (L, док)): SVD центрированного H_doc (H_t − mean_t H_t,
  полный 128) → первые k правых сингулярных векторов (дословно AT2c).
- **haar-k** (глобальный фиксированный): G32 = standard_normal((1024,32))
  (rng = np.random.default_rng(0) — дословно AT2c), HAAR_k = QR(G32[:, :k]).
- **weak_Gbar-k (CANONICAL арм, secondary branch) — ПОЛНАЯ конструкция**
  (ревью r3 M4): G̸ = mean_{d,t} Jᵀ diag(P128) J на **ВСЕХ 4096 (d,t)**
  per слой (f64-аккумуляция: per-doc f32-batch-граммы → f64-сумма; ~3 мин/
  слой, P0'). eigh, nz, weak_Gbar = первые k из nz по возрастанию (k=4/8).
  Субсэмпл r3 (8 позиций × 32 дока) — **report-only**: главные углы
  full-vs-subsample в артефакте (P0': 64–87° на k=4 — субсэмпл практически
  другой базис; canonical-арм r3 был бы чужим).
- **randG-k (ПЕРВИЧНЫЙ sham, PER-DOC)**: для каждого дока d: A_d =
  standard_normal((1024, k)) (rng = np.random.default_rng(200_000 + L·10 +
  k + 1000·d)), V_d = QR(A_d). Фреймы в артефакте (orth + sha per (L,k,d)).
  P0' (правильное окно 0..63): mean 0.00351–0.00400 (theo 0.00391), per-doc
  sd 4.2e-4–9.0e-4 (E_raw), dose-sd 0.02–0.36 — draw-дисперсия входит в
  doc-bootstrap (per-doc фреймы).
- **α-лестница randG** (ревью r3 B1/M7): ТОТ ЖЕ per-doc фрейм randG,
  масштабирование h' = h − α·VVᵀh при **α ∈ {0.5, 1, 2, 4}** (α=1 — базовый
  sham из сетки; 0.5/2/4 — лестница dose–response: dose ∝ α² в первом
  порядке ⇒ doses {0.25, 1, 4, 16}×D_randG). Лестница — прямой тест
  softening-канала без рычага и R² (заменяет S-energy гейт).

**Доза (M1)**: функциональная пертурбация — **post-norm dose**
D(арм, док) = mean_{pos 0..63} ‖(I−û_t û_tᵀ)·(δh_t/√mean(h_t²))·g‖²,
δh_t = −VVᵀh_t (α=1), û_t = h_t/‖h_t‖, **g = weight input_layernorm
СЛЕДУЮЩЕГО слоя** (model.model.layers[L+1].input_layernorm.weight — первый
преобразующий норм после хука; L23 → final norm). Первый порядок по
RMSNorm (изменение знаменателя не учитывается — оговорка). Raw E
(mean ‖Vᵀh‖²/‖h‖²) и **KL_pred = ½ mean_{pos} ‖√P128_RAW·(JR·δh)‖²**
(pooled-Fisher; P128_RAW — субстохастический row полного распределения) —
отчётные ковариаты (disk-only). Oговорки: U128-pooled basis ≠ полный
вокабуляр, диагональный P128_RAW ≠ полная Fisher, первый порядок.

**Амплитудное матчинг (первичный контраст, по построению)**: для каждого
не-sham арма (top, weak, gbar, pca, haar) и дока: α(арм, L, док) =
√(D_randG(док)/D_arm(док)) (первый порядок, closed form, disk-only).
Ограничение α ∈ [0.05, 5]: за пределами — флаг `unmatched`, арм-док
отчитывается при α=1, исключается из excess_match (остаётся в отчёте).
Остаточное несовпашение dose после матчинга отчитывается per arm-док
(P0': α_gbar4 ≈ 0.30–0.40 на L6/12/18/19, 1.68 на L22 [1.15–2.49]; pca
α ≈ 0.05–0.2 — у границы).

r_final = 7 AT2c — **не используется** (M8: rematching readout-клетки;
интервенция матчится по dose, не по d_eff — провенанс в §12). weak_mid и
randJ — **выбиты** (weak_mid заменён weak_Gbar; randJ — draw-artifact,
P0/pre-study §8.9). tw — **выбит** (M8 r2: 2k против k — механический
excess ≈ −0.13; реинтро с randG(2k) — только отдельным решением).

## 3. Пайплайн: форвард, хук, метрика

**Хук** (проверен pre-study: слой Qwen3_5DecoderLayer возвращает plain
Tensor; хук на `model.model.layers[L]`):
- **Вариант P_future (ПЕРВИЧНЫЙ, verdict; ВАРИАНТ A, ревью r3 B5)**:
  h' = h − αVVᵀh (α=1 для base-армов; α-matched для verdict-армов),
  применяется ТОЛЬКО при t < 64 (позиции 0..63); метрика ΔCE — ТОЛЬКО по
  позициям **64..126 (logit-позиции) = цели 65..127 = 63 значения**.
  Позиция 63 (последняя абляционная) в метрику НЕ входит by construction.
  Biны per-position ΔCE (target-индексные, 5): **65-77, 78-90, 91-103,
  104-116, 117-127** (13,13,13,13,11). G-zero (L23, §4) независимо
  верифицирует окно: любой сдвиг на 1 позицию даст ненулевой ΔCE на L23.
- **Вариант P (persistent; ТОЛЬКО smoke-DS2 (2 дока) и L23-контроль)**:
  h' на всех 128 позициях; ΔCE по 127 значениям (0..126). Из main-сетки
  5 слоёв ВЫВЕДЕН (M7 r3: 1920 форвардов report-only при несущем
  первичном эстиманде).
- (r2-вариант R — reflection — из main-сетки выбит: R/P = 4.00/4.17,
  предсказан из P; сохраняется ТОЛЬКО в смоке как диагностик
  квадратичности: ratio R/P, ожидание ~4.)
- GDN-внутренние state напрямую НЕ трогаются (только residual stream h_l);
  модифицированный residual продолжается следующими слоями, GDN-слои
  обновляют свой рекуррентный state от изменённого входа — часть
  causation-пути (зафиксировано).

**Конвенция позиционирования кэша** (ревью r2 NIT #7; **баг M2 r3
исправлен в P0' и зафиксирован здесь**): ключ кэша H[(dom,si,t)] = позиция
**127−t** (t-инверсия builder'а exp27b_dense.py: hs[S−1−t]). В КОДЕ:
Hf[doc, pos] = cache[(dom, si, 127−pos)] — forward-порядок ВСЕГДА.
G2c-сверка: H_fwd[L+1][:, pos] vs H_cache[(dom,si, 127−pos)] на всех 128
позициях. P0-скрипты r3 брал Hd[:64] в кэш-индексах = позиции 64..127 —
**в r4 все окна в forward-индексах**, обе половины отчитываются.

**Baseline (B)**: полный форвард дока (128 токенов) без хука: CE_B(d) по
softmax logits (f32). **Полные логиты z_B [128, 248320] удерживаются в RAM
per док (~127 MB) и сбрасываются после вычисления sham-величин** (M3:
температурный sham — из B_plain-логитов ПРОГОНА, не из кэша — в кэше только
pooled U128/P128/P128_RAW/JR/H). Отчёт: mean по 127 значениям (0..126) и
отдельно по окну 64..126 (m2: позиция 0 — нулевой контекст).

**Метрики** (per (L, k, арм, док); первичное окно P_future):
- **ΔCE** = CE_arm − CE_B (nats/токен) — P_future: среднее по **63**
  значениям (позиции 64..126); P: по 127. (Отрицательный ΔCE = абляция
  улучшила предсказание — softening overconfidence, pre-study §8.)
- **D(арм, док)** — post-norm dose (§2) + raw E + KL_pred — отчёт,
  per-doc (окно 0..63).
- **excess_match(арм, док) = ΔCE(арм, док) − ΔCE(randG, док)** —
  ПЕРВИЧНЫЙ контраст (per-doc пара: тот же док; матчинг: dose(арм, α_matched)
  = dose(randG, α=1) per док by construction ⇒ «matched-randG» — это БАЗОВЫЙ
  randG-ран (α=1) из лестницы, доп. ран НЕ нужен; draw-дисперсия в бутстрапе).
  Valid по построению (одна dose, один док). Для pca при α у границы [0.05] —
  см. §2 (unmatched-флаг).
- **excess_cal(арм, док)** — ПЕРВИЧНЫЙ (ко-): τ*(арм, док) — решение
  mean_{pos 64..126} H(softmax(τ·z_B)) = mean_{pos 64..126} H(p_arm(док))
  (H монотонно убывает по τ ∈ (0,∞): бисекция [0.01, 100], tol 1e-9;
  p_arm — из логитов arm-форварда, полные). excess_cal = ΔCE(арм, док) −
  [CE(τ*·z_B) − CE(z_B)] (окно 64..126). τ>0 не меняет argmax ⇒ второе
  слагаемое — чистый энтропийный эффект; «абляция хуже температурного
  эквивалента» = удалена используемая информация (формулировка H1).
  0 доп. форвардов.
- **dCE_correct / dCE_wrong** (co-primary, M3): per-doc ΔCE на позициях
  окна 64..126, где baseline ПРАВИЛЕН (argmax p_B == target) / НЕПРАВИЛЕН.
  «weak causally используется» обязано проявляться на правильных позициях,
  не как общее смягчение (CE_plain ≈ 14.7 > ln V = 12.42 — пул
  преимущественно confidently-wrong).
- **KL(арм, док)**: mean по позициям 64..126 KL(p_arm ‖ p_base) (полный
  вокабуляр, из логитов; скаляр в артефакт). excess_KL(арм) = KL(арм) −
  KL(randG, α=1) — сопутствующий (разведён с ΔCE: «для выхода» vs
  «для правильного ответа»).
- **Per-position ΔCE**: 5 бинов (§3, target-индексные) per (дока, арм) —
  отчёт (затухание/послесвечение).
- **α-лестница randG** (per (L,k,док), α ∈ {0.5,1,2,4}): ΔCE(α) —
  dose–response кривая; отчёт + SOFTENING_DOMINATED-тест (§5): где лежит
  точка weak (matched) относительно кривой при своей dose.
- **norm_ratio** (report-only): mean_t ‖h'_t‖/‖h_t‖ на окне 0..63.

**Статистика**: unit = док (32). Doc-bootstrap B=1000, seed 0
(np.random.default_rng(0)): ресемплинг 32 дока с повтором; **все пер-док
эстиманды (ΔCE, excess_match, excess_cal, τ*, D, KL, бины, dCE_correct) —
per-doc величины, бутстрап усредняет пер-док значения ресемплированных
доков — БЕЗ перефита** (ĉ нет; пары «арм + его randG-фрейм + базовый
randG-ран + z_B» ресемплируются вместе — та же выборка дока, пары
сохраняются;
draw-дисперсия sham'а в бутстрапе). CI95 = перцентили 2.5/97.5;
односторонние верхние/нижние — перцентили 97.5/2.5. Отчёт: mean, CI95,
per-doc таблицы (ΔCE, excess_match, excess_cal, D, E_raw, KL_pred, KL,
dCE_correct, dCE_wrong, α, bins5).

## 4. Контроли

1. **G1 (hook no-op, bit-exact)**: baseline прогоняется ДВАЖДЫ per док:
   B_plain (хук не зарегистрирован) и B_hook (хук зарегистрирован, но no-op:
   пустой V → h' = h). Гейт: max|Δlogits| = 0 на всех 32 дока. Промах = стоп.
2. **G4 (sham responsiveness, ревью r3 M6 — определение без разрыва)**:
   pass, если |ΔCE_P_future(randG4, α=1)| ≥ 0.005 на **≥4 из 10** (L,k);
   <4 клеток — флаг **DEAD_CELL**, вердикт принуждён к NOT_RESOLVED с
   фиксацией в артефакте (защита от тихого plumbing-срыва; НЕ жёсткий гейт).
   Количество отчитывается. (0.005 — пре-регистрация из pre-study-оценки
   |ΔCE_P_future(randG)| ≈ 0.02–0.06.)
3. **G-zero (L23, ревью r3 B4/B5 — новый)**: L23 = последний слой; после
   него — только final-norm + head (per-position) ⇒ ΔCE_P_future(L23) ≡ 0
   бит-в-бит для любого арма (проверено по архитектуре, §1.3). Гейт: на
   обоих армах {top4, randG4} × 32 дока (P_future): **max|Δlogits[t≥64]| ≤
   1e-6 И |ΔCE_P_future(L23)| ≤ 1e-6**. Единственная проверка, ловящая
   протечку окна: если в метрику попадёт позиция 63, L23-абляция даст на
   ней ненулевой ΔCE (маржа 1e-6 vs ~0.01–0.5). Промах = стоп (метрика
   битва — артефакта нет).
4. **L23-P-контроль (direction specificity, ревью r3 B4)**: L23 под
   ВАРИАНТОМ P (persistent, все 128 позиций — там абляция L23 реально бьёт
   по текущим логитам; J там = readout): 32 дока × {top4, randG4} × P =
   64 форварда. Отчёт: ΔCE, KL, excess_match. Ожидание (не гейт):
   |ΔCE_P(top4)| > |ΔCE_P(randG4)| — клетка direction-specific.

## 5. Дерево вердиктов (r4 — по первичному P_future, variant A)

Предварительно: S1, G1, G2a, G2b, G2c, G3, S9, Ortho, **G-zero**, G4 —
по гейтам. Иначе — артефакта нет, стоп (G4 — флаг, не стоп).
Все ветки — по **excess_match(weak_Gagg) И excess_cal(weak_Gagg)** per (L,k),
10 клеток (5 L × 2 k), doc-bootstrap CI.

- **H1-causal → `WEAK_CAUSAL_SIGNAL`** (M7 r2 + M3 r3): **≥2 из 10 клеток**:
  CI_lower(excess_match(weak)) > 0 И CI_lower(excess_cal(weak)) > 0
  (односторонние 97.5%, per-cell α = 0.025/10 = 0.0025, Bonferroni;
  FWER ≥2 ложных ≈ C(10,2)·0.0025² ≈ 0.3%) И на той же клетке парные
  per-doc CI_lower (excess_match(weak) − excess_match(pca)) > 0 И
  (excess_match(weak) − excess_match(haar)) > 0 (том же α; пара считается
  на доках, где ОБА арма матчнуты; n < 8 — пара не оценивается на клетке,
  отчёт) И **точечная excess_match(weak, correct-позиции) > 0** на той же
  клетке (CI отчёт; M3: пул confidently-wrong не вердикт сам по себе).
  → «weak-сторона mean-J спектра causally специфична (не лучше по dose,
  лучше по функции); readout (T2.1c) негативен, causal-контраст
  положительный». (Согласие ОБОИХ эстимандов — консервативная
  пре-регистрация; риск разжижения задекларирован: два независимых
  механизма (sham-контраст и temp-sham) обязаны сказать одно.)
- **`WEAK_CAUSAL_REFUTED_MEANJ`** (ревью r3 B3 — non-superiority, ветка
  достижима под нулём): на **ВСЕХ 10 клетках**: CI_upper(excess_match(
  weak)) < δ_min И CI_upper(excess_cal(weak)) < δ_min (односторонние 97.5%,
  **δ_min = 0.010 — фиксирован ДО прогона**; вывод: 1.96·SE, SE ≈
  0.029/√32 ≈ 0.0051 — pre-study per-doc sd ΔCE(weak4); P0' noise-флора
  контраста не даёт (требует форвардов) — фактическая SE отчитывается
  прогоном, δ_min не пересматривается post-hoc). → «weak-сторона mean-J
  спектра causally НЕ специфична (эффект меньше различимого)».
  Достигаемость под H₀: P(cell) = Φ(0.010/0.0051) ≈ 0.977 → P(все 10) ≈ 0.8.
- **`SOFTENING_DOMINATED`** (заменяет ENERGY_DOMINATED; ревью r3 B1 — без
  рычага/R²/порога): **≥4 из 10 клеток**: CI95(excess_match(weak))
  накрывает 0 (direction неразличим с random-направлением той же dose).
  Отчёт: положение точки weak (matched) на α-лестнице randG per клетка
  (dose–response). → «клетка мерит амплитуду возмущения (softening-канал),
  не направление; excess-контрасты — ноль в пределах CI». T2 этой клеткой
  НЕ решается (readout-часть — AT2c).
- **Secondary branch (pre-registered, canonical базис full, ревью r3 M4)**:
  - `WEAK_CAUSAL_SIGNAL_GBAR`: ≥2 из 10 клеток CI_lower(excess_match(weak_
    Gbar)) > 0 И CI_lower(excess_cal(weak_Gbar)) > 0 (α/10; парные — как в
    SIGNAL; α-matched, §2).
  - `WEAK_CAUSAL_REFUTED_GBAR`: на всех 10 клетках CI_upper < δ_min (оба
    эстиманда).
- **Закрытие T2** (по базисам): T2 закрывается как «не подтверждено»
  (решение за супервайзером), ЕСЛИ: (i) readout отклонена на обоих базисах
  (AT2c: weak_Gagg — §11.3 CI; weak_Gbar — §11.7 probe ≤ floor на k≥1), И
  (ii) causal: WEAK_CAUSAL_REFUTED_MEANJ (и GBAR, если применимо).
  Частичные случаи → T2 НЕ закрывается, отчёт по базисам.
- **`NOT_RESOLVED`** (иначе): смешанный случай (weak > 0 на каких-то
  клетках, но не ≥2; CI через 0 без SOFTENING_DOMINATED-порога; DEAD_CELL;
  unmatched-флаги) — отчёт со всеми CI.

**Ожидаемые consistency-отчёты (не гейты)**:
- **excess_match(top) ≈ 0** (top не специфичен per dose; r2-ожидание
  «excess_G(top) < 0» было энергетическим артефактом — исправлено в r3,
  сохраняется в r4 в виде: top (matched) не отличим от random-направления
  той же dose).
- **α-лестница randG**: монотонная (в первом порядке квадратичная)
  dose–response — отчёт per (L,k); отклонение = нелинейный информационный
  канал (аналог DS3).
- **L19 (full attention) vs L6/12/18/22 (GDN)**: вариация по типу
  внимания (M9; отчёт, не предсказание).
- **L23-P**: |ΔCE_P(top4)| > |ΔCE_P(randG4)| (direction specificity).
- **KL vs ΔCE-разведение**: excess_KL(weak) vs excess_match(weak) —
  «для выхода» vs «для правильного ответа» (M6 r2 — отчёт).
- **Стратификация (co-primary)**: excess на correct vs wrong позициях
  (M3 r3).
- **обе половины энергии** (E_early/E_late, D_early — отчёт per арм/док;
  M2 r3).

## 6. Масштаб (r4-сетка; цена-нейтрально vs r3)

- 32 дока (4 домена × 8), 128-префикс. L ∈ {6,12,18,22,19} (5: GDN×4 +
  full×1), k ∈ {4,8}.
- Arm-runs per (L,k,док) под P_future: 9 — {top, weak, weak_Gbar(α-matched),
  pca, haar} (все 5 α-matched per док) + randG {α=1 (base), α=0.5, α=2,
  α=4} (лестница).
- Форварды: 32×5×2×9 (2880) + L23: 32×{top,randG}×{P_future (G-zero), P}
  (128) + B_plain 32 + B_hook 32 = **3072**. Замер 3.6–4.2 с/форвард →
  **≈ 3.6 ч** + bootstrap (минуты) + загрузка (2–3 мин) + G̸-full-конструкция
  (~3 мин/слой, disk-only) ≈ **4.0–4.5 ч** total (r3: 3968 ≈ 5–5.5 ч —
  ДЕШЕВЛЕ). Fallback: wall > 9 ч — стоп и разбор.
- Smoke-сетка (§11): 2 фиксируемых дока (wiki,0),(wiki,1), L6, k=4:
  9 arm-runs (P_future) + {top4, randG4}×P (DS2) + R(top4, P_future) +
  B_plain + B_hook = 2×(9+2+1+2) = 28 форвардов/прогон; S9-пара = 56
  ≈ 4 мин.

## 7. Гейты (до записи артефакта; провал = артефакта нет, стоп)

- **S1**: sha256 7 файлов кэша (таблица AT2c S1, дословно).
- **G1**: hook no-op бит-в-бит (max|Δlogits| = 0, 32 дока).
- **G2a**: структура pool_ids: 4 домена × 8 дока × 128 токенов, int-типы.
- **G2b**: pool_fp per домен (meta pool_ids) == pool_fp per домен в
  jcache-ckpt L6 (все 4 совпадают).
- **G2c (state provenance, все позиции)**: per (док, L∈{6,12,18,22,19}) cos
  на ВСЕХ 128 позициях (H_fwd[L+1][:,pos] vs H_cache[(dom,si,127−pos)]):
  гейт min ≥ 0.999 И отчёт median (pre-study: min 0.99999976, медиана 1.0).
  Промах = стоп.
- **S9 (детерминизм, смоки)**: два smoke-прогона → md5 артефакта бит-в-бит
  (дословно AT2c S9).
- **Ortho (f64)**: ‖VᵀV − I‖ < 1e-6 на всех (L,k,арм,док), f64 (замер:
  f64 ≤ 2e-15; ‖VVᵀ−I‖₂=1 всегда — не тот норм). Стоп при провале.
- **G-zero (новый, §4.3)**: L23 P_future max|Δlogits[t≥64]| ≤ 1e-6 И
  |ΔCE| ≤ 1e-6 (64 runs). Стоп при провале.
- **G4 (§4.2)**: флаг (не стоп): ≥4/10 клеток |ΔCE_P_future(randG4)| ≥ 0.005.
- **G3 (trackio)**: run **at2d_t22_r1** (первый main-прогон клетки; TZ-r4 в
  config): t=0 config (started_at, device=cpu, revision="TZ-r4", script_sha,
  seed=0, k, layers, n_docs=32, boot_b=1000, model, venv, log, delta_min=
  0.010) + stage-точки (load / prov / gbar_full / forwards-L{…} /
  l23_control / ladder / temp_sham / bootstrap / artifact) + финальная (t =
  duration, finished_at, status: ok|crash|interrupted, log_md5 — в
  finally/atexit; crash — НЕ молча).

## 8. Pre-study + P0 + P0'-замеры (17.11; НЕ гейты — основа дизайна)

Pre-study (L6, 32 дока; CPU 4 тр., f32 eager, без fla; скрипты /tmp/at2d_
{speed,hooktest2,power,energy,pos,reflect,seedtest}.py):

1. **Скорость**: форвард 128 токенов 3.3–3.8 с; ~4.2 с/форвард с хуками.
2. **Детерминизм**: повторные форварды бит-идентичны (max|Δlogits| = 0).
3. **State provenance (G2c)**: cos min 0.99999976 (code,2,L18, позиция 127),
   медиана 1.0, 128 пар (32 дока × 4 слоя). L23 точное совпадение.
4. **Хук**: plain Tensor; no-op бит-в-бит; абляция распространяется на
   downstream (max|Δh| = 18.7 на L23 при L6-абляции).
5. **CE_plain ≈ 14.7 nats/токен** (wiki doc0) > ln V = 12.42 — модель в
   среднем «confidently wrong»; механизм «улучшения» от абляции — softening:
   corr(CE_plain_i, ΔCE_i) = −0.31 (doc0, top4).
6. **Все армы дают ОТРИЦАТЕЛЬНЫЙ ΔCE_P** (32 дока, L6): top4 −0.150±0.157,
   weak4 −0.027±0.029, randJ4(draw1) −0.154, top8 −0.245±0.175,
   weak8 −0.101±0.059, randJ8(draw1) −0.214.
7. **Энергия E** (full-окно): top4 0.0102, weak4 0.0037, randJ4 0.0067,
   top8 0.0201, weak8 0.0095, randJ8 0.0168; k/D = 0.0039 ≈ E(weak4).
   ΔCE_P ≈ c·E (c ≈ −10…−15) — softening-канал; **c по армам −7.3/−10.6/
   −12.2/−12.7/−14.7/−23.0 — single-c опровергнут собственным pre-study**
   (ревью r3 B2a).
8. **Варианты**: R ≈ 4×P (top4: 4.00; top8: 4.17) — квадратичный отклик;
   nr ≈ P — норм-канал незначим.
9. **randJ — draw-artifact**: excess(weak4−randJ4) по 3 seed'ам:
   +0.127/+0.020/−0.005 → randG — первичный sham, randJ — выбит.
10. **L6-вывод**: weak не выделен явно относительно J-draw'ов; вопрос main:
    специален ли weak относительно направлений той же DOSE (r4: matched +
    temp-sham).

**P0 раунда 2** (disk-only, /tmp/at2d_p0_diag.py — **окно было СМЕЩЕНО
на 64 позиции (баг M2 r3): его числа — только для истории**; пересмотрено
P0'): sham-дисперсия (per-t rel 26–33%, window-средняя вторична к doc-SE),
energy-фит c=−12.93 R²=0.804 (pre-study 6 точек — single-c уже не плотен),
R/P=4.00/4.17, gbar-энергии (неправильное окно), per-doc sd E(weak4).

**P0' раунда 3 (disk-only, /tmp/at2d_p0r_diag.py, /tmp/at2d_p0r.json,
P0R_DIAG_DONE; ПРАВИЛЬНОЕ окно 0..63, обе половины, full-gbar, dose,
KL_pred)** — основа r4:
- **P0'.1 — энергии/dose/KL_pred (mean32±sd32, окно 0..63)**:
  L6: top4 E 0.0127±0.0015 / D 0.464 / KLp 0.641; weak4 0.0045±0.0009 /
  0.106 / 0.625; gbar4_f 0.0821±0.0152 / 0.703 / 1.662; pca4 0.1603±0.0272 /
  **3.549** / 31.24; haar4 0.0038 / 0.106 / 0.396; randG4 0.0040±0.0011 /
  0.110 / 0.861. L12: top4 0.0166 / 2.399 / 0.829; weak4 0.0036 / 0.551 /
  0.667; gbar4_f 0.0709 / 8.482 / 13.61; pca4 0.1888 / **31.21** / 34.35;
  haar4 0.0029 / 0.642 / 0.358; randG4 0.0038 / 0.851 / 0.845. L18: top4
  0.0172 / 0.889 / 2.89; weak4 0.0032 / 0.172 / 0.458; gbar4_f 0.1465 /
  2.354 / 41.89; pca4 0.2144 / 9.176 / 30.32; haar4 0.0029 / 0.164 / 0.711;
  randG4 0.0038 / 0.215 / 1.063. L19: top4 0.0204 / 8.794 / 3.086; weak4
  0.0031 / 1.319 / 0.503; gbar4_f 0.0887 / 16.12 / 99.30; pca4 0.2506 /
  72.57 / 27.99; haar4 0.0033 / 1.428 / 0.724; randG4 0.0036 / 1.577 / 1.112.
  L22: top4 0.0195 / 1.899 / 7.615; weak4 0.0025 / 0.182 / 0.225; gbar4_f
  **0.0017** / 0.109 / 0.313; pca4 0.1753 / 13.00 / 50.50; haar4 0.0031 /
  0.267 / 0.933; randG4 0.0035 / 0.301 / 1.471. E_late (64..127): ±10–30%
  от E_early (weak4 стабилен: 0.00449/0.00451 L6).
- **P0'.2 — рычаг (B1) на правильном окне**: E²-доля arмов (k=4): L6
  pca ≈ 0.79 / gbar ≈ 0.21; L12 pca ≈ 0.87; L18 pca ≈ 0.68; L19 pca ≈ 0.88;
  L22 pca ≈ 0.98 (gbar ничтожен: 0.0005). **Механический R² S-energy-гейта**
  (закон ΔCE=cE + шум arm-mean, c ∈ {−15,−13,−10}, se ∈ {0.005..0.03}):
  **R² = 0.9977…1.0000 на ВСЕХ 5 слоях, 3/3 c при каждом se** → гейт
  r3 (≥0.85 на ≥4/10) срабатывал бы гарантированно ⇒ снят (α-лестница).
- **P0'.3 — gbar FULL vs SUBSAMPLE (M4)**: главные углы (k=4): L6
  37.8–79.4°, L12 28.5–72.6°, L18 29.6–71.8°, L19 29.6–63.8°, L22
  65.0–87.2°; k=8 max 86.5–88.8° → субсэмпл ≈ другой базис → **full-
  конструкция** (M4). P128_RAW идентичен по слоям (L12 vs L6 max|Δ| = 0.0).
- **P0'.4 — мощности/матчинг**: per-doc sd E(weak4) 4.2e-4–9.0e-4, dose-sd
  0.02–0.36 (L19); **α_gbar4 = √(D_randG/D_gbar): mean 0.398 [0.295–0.551]
  L6, 0.317 L12, 0.301 L18, 1.679 [1.147–2.486] L22, 0.314 L19**; dose(
  weak4) 0.11–1.32 ≈ dose(randG4) 0.11–1.58 — **weak_Gagg по dose
  согласован с sham'ом даже при α=1**. δ_min = 0.010 (1.96·SE, SE ≈
  0.029/√32 — pre-study per-doc sd ΔCE(weak4); P0' noise-флора контраста
  не даёт — требует форвардов; фиксирован ДО прогона, факт. SE — отчёт).

## 9. Ожидания (не гейты)

1. **excess_match(top) ≈ 0** на большинстве (L,k) (top не специфичен per
   dose; см. §5).
2. **|ΔCE_P_future(randG4, α=1)| ≈ 0.02–0.06** — клетка откликается (G4).
3. **α-лестница**: ΔCE(randG, α) монотонно (в 1-м порядке ∝ α²) — отчёт;
   отклонение = нелинейный канал.
4. **excess_cal(weak)**: без обязательства (сигнал либо non-superiority).
5. **L19 vs GDN-слои**: вариация по типу внимания (отчёт).
6. **L23-P**: |ΔCE_P(top4)| > |ΔCE_P(randG4)| (direction specificity).
7. **Вероятен исход SOFTENING_DOMINATED или NOT_RESOLVED** (честный
   отрицательный про клетку: softening-канал доминирует в pre-study,
   single-c не плотен — специфичности weak в pre-study не видно).

## 10. Что НЕ делается (scope, r4)

- Никакой генерации: только forward на заданных 128-префиксах.
- Никакой интервенции во внутренние GDN/recurrent state напрямую (только
  residual stream; обновление GDN-state от изменённого входа — часть
  causation-пути, §3).
- **tw, weak_mid, randJ** — выбиты (§2).
- **R-вариант в main** (квадратичный, предсказан из P; smoke-диагностика).
- **Вариант P в main-сетке 5 слоёв** (M7 r3: 1920 ф. report-only; остаётся
  только smoke-DS2 + L23-контроль).
- Никакого nr-варианта. Никаких k ∈ {1,2,16}; никаких L, кроме
  {6,12,18,22,19} (L11 не в кэше).
- Никакого per-position разбора в вердикт (бины + стратификация — отчёт,
  дCE_correct — co-primary в отчёте и SIGNAL-условии).
- Никакого ĉ/energy-фита/субпоулинга армов (B1/B2 r3 — сняты).
- Никакого GPU/TPU (CPU-only, один прогон за ночь по AGENTS).

## 11. Смоки (до main-прогона, отдельный шаг; AT2d_SMOKE=1)

Сетка: 2 **фиксируемых** дока (wiki,0),(wiki,1), L6, k=4, 9 arm-runs
(P_future: top4/weak4/gbar4_f/pca4/haar4 α-matched + randG4 {1,0.5,2,4}) +
{top4, randG4}×P (DS2) + R(top4, P_future) + B_plain + B_hook =
28 форвардов/прогон.

1. **S9 (детерминизм)**: два smoke-прогона → md5 артефакта бит-в-бит.
2. **G1 (hook no-op)**: max|Δlogits| = 0 (стоп иначе).
3. **DS1 (хук действует)**: max|Δh| на выходе хука (P_future, top4, per док)
   ≥ 0.1 (pre-study: 0.18) — стоп иначе.
4. **DS2 (plumbing + softening; ревью r3 M5 — на P, где знаки измерены)**:
   mean по 2 smoke-докам **ΔCE_P(top4) < −0.03** (pre-study −0.150±0.157,
   full-window — знак измерен) — стоп иначе. ΔCE_P(randG4) и ВСЕ
   P_future-значения — report-only (знак future не установлен — ложный
   STOP исключён).
5. **DS3 (квадратичность, диагностика, НЕ стоп)**: R/P (top4, 2 дока) —
   отчёт, ожидание 2..6 (pre-study 4.0); отклонение = нелинейный канал.
6. **DS4 (temp-sham plumbing, НОВЫЙ, НЕ стоп)**: на smoke-доках τ*(top4)
   найден (бисекция сходится, τ* ∈ [0.1, 10]) И excess_cal(top4) −
   excess_match(top4) отчитан (два механизма дают сопоставимые значения —
   plumbing sham'а работает). Стоп только при несходимости.
7. **DS5 (α-matching plumbing, НОВЫЙ, НЕ стоп)**: per smoke-док
   α(weak4), α(gbar4_f) ∈ [0.05, 5] И остаточное несовпашение dose < 25% —
   отчёт; вне границ — флаг (smoke-артефакт).
8. **Ortho (f64)**: ‖VᵀV−I‖ < 1e-6 (стоп).
9. **G2c** на 2 smoke-доках: min по 128 позициям ≥ 0.999 (стоп).
10. **G-zero smoke (1 док, top4, P_future)**: |ΔCE| ≤ 1e-6 (стоп иначе —
    окно/метрика биты ДО main; полный G-zero — 32 дока в main).
11. **randG-фреймы per-doc**: записаны в smoke-артефакт (orth + sha).

## 12. Провенанс (что на чём стоит)

- **Кэш exp27b** (T4, fla 0.4.2, transformers 5.12.1): S1 sha + G2b pool_fp
  + G2c state provenance (3 независимые верификации, §7). P128_RAW — из
  L6-кэша для L19 (идентичен по слоям, spot-check max|Δ| = 0.0, P0').
- **J-базис**: код AT2c.py:440–484 (Jagg/Gagg/eigh/sort) — дословный copy
  в AT2d.py. **G̸-базис: ПОЛНАЯ конструкция** (все 4096 (d,t) per слой, f64)
  — код P0' (gbar_from в /tmp/at2d_p0r_diag.py); субсэмпл (AT2c §11.7) —
  только для углов в артефакте. **AT2d не использует r_final AT2c (=7):
  rematching readout-клетки не переносится на интервенцию; матчинг — по
  post-norm dose (§2, M8 r3).**
- **Токены**: pool_ids из кэша (G2a/G2b), не реконструкция.
- **Pre-study + P0 + P0'**: §8 (скрипты /tmp, повторяемые, disk-only).
  P0 (r3) — окно смещено (M2 r3) — только история; дизайн r4 стоит на
  P0'.
- **AT2c-ревью раундов 8–9 (17.11)**: docs/AT2c_review_r8_answers.md /
  docs/AT2d_review_r3_answers.md — основание weak_Gbar-арма (full) и
  secondary branch.
- **AT2d-ревью**: раунд 2 (4 блокера) → r3; **раунд 3 (B1–B5, M1–M8,
  docs/AT2d_review.md sha b394ef5457bbbd6f) → r4** (этот документ);
  все блокирующие пункты закрыты редизайном (цена-нейтрально: 3072 ф. ≈
  4.0–4.5 ч vs r3 3968 ≈ 5–5.5 ч).
- **ЗАВИСИМОСТЬ ОТ AT2b/AT2c-КОДА** (как в r2): если в базисах/кэше/доках
  найден серьёзный баг — перепуск с исправленным кодом.
- **Модель**: /data/Qwen/Qwen3.5-0.8B (weights sha — в артефакте meta;
  layer_types — в meta; архитектура после L23 — per-position только, §1.3).
- **Окружение**: .venv (torch 2.13.0+cu130, transformers 5.15.1, без fla),
  CPU 4 тр., python 3.13.

## 13. Схема артефакта (results/AT2d/AT2d.json)

```
{
 "meta": {revision:"r4", script_sha, seed:0, k:[4,8],
          layers:[6,12,18,22,19],
          layer_types: {L: "gdn"|"full"} (все 24 из config),
          n_docs:32, boot_b:1000, model, venv, wall_sec, started_at,
          finished_at, trackio_run, log_md5, model_weights_sha (per file),
          at2c_script_sha, p0r_sha,
          gbar_full: {ts:"4096", docs:32, dtype:f64},
          window: {ablate:"0..63", metric_pos:"64..126", targets:"65..127",
                   n:63, bins_targets:[65-77,78-90,91-103,104-116,117-127]},
          delta_min:0.010},
 "gates": {S1: {file: sha}, G1_hook_noop: {max_logit_diff, per_doc, ok},
           G2a_pool_ids: {...}, G2b_pool_fp: {...},
           G2c_state_prov: {cos_min_all_pos, cos_median, worst, ok},
           ortho_f64: {max_norm_diff, ok},
           randG_frames: {L|k: {d: {sha, ortho_ok}}},
           s9_smoke: {md5_pair, ok},
           smoke_ds: {DS1, DS2, DS3_ratio_R_P, DS4_tsham, DS5_alpha,
                      ortho, G2c_smoke, Gzero_smoke},
           g_zero: {per run (2 arma × 32 дока): max_logit_diff_t_ge_64,
                    dce, ok},
           g4: {n_cells_responsive, threshold:0.005, ok/DEAD_CELL}},
 "baseline": {per doc: CE_B_full127, CE_B_win63, H_B_win},
 "arms": {"L{L}k{k}": {"P_future": {"<арм>": {per_doc:[32]: {dCE, D,
          E_raw, KL_pred, KL, excess_match, excess_cal, tau_star, alpha,
          dose_residual, dCE_correct, dCE_wrong, bins5:[5]}},
          mean, ci95, excess_match_mean, excess_match_ci95,
          excess_cal_mean, excess_cal_ci95,
          excess_match_correct_mean, excess_match_correct_ci95}},
          "ladder": {per_doc α∈{0.5,1,2,4}: dCE},
          "pairwise": {per_doc: (weak-pca), (weak-haar), n_matched}},
          "P": {...L23 control only...}}, ...},
 "l23": {"P_future": {top4/randG: per_doc dCE, max_logit_diff},
          "P": {top4/randG: per_doc dCE, KL, ratio}},
 "expectations": {excess_match_top_near_zero: per (L,k) bool,
          randG_responsive: per (L,k) |ΔCE|, ladder_monotone: per (L,k),
          l19_vs_GDN: per metric, l23_specificity: bool},
 "verdict": "...", "verdict_detail": {per (L,k): excess_match(weak)
          mean/ci95, excess_cal(weak) mean/ci95, pairwise vs pca/haar,
          correct-pos point-est, per branch, n_unmatched},
 "deviations": []
}
```

Smoke-артефакт: results/AT2d_smoke/AT2d_smoke{,_det}.json (S9-пара).
Артефакт ~0.5–1 MB (per-doc × бины × стратификация × лестница) — в git.

## 14. Коммит-план

Код, ТЗ (r1–r4), ревью-файлы (docs/AT2d_review.md + ответки r2/r3),
results (лог + артефакт + smoke) — в git явными путями. CJK-гейт,
маркеры ревью — по AGENTS.
