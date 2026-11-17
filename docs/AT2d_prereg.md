# AT2d (T2.2) — пре-рег r3: причинная (интервенционная) state-клетка

**Статус: ТЗ r3 — на внешнее ревью.** r1 — внутренний черновик (не рассматривался
внешним ревьюером: воркер умер на output-limit без файла/маркера, зафиксировано).
r2 — внешнее ревью (docs/AT2d_review.md, 17.11): 4 блокера (sham-дисперсия не в CI,
energy-фит R²=0.80, current/future смешение, неканонический базис в REFUTED) +
M5–M9 + m1–m8 → **r3 — редизайн по ревью** (ответка: docs/AT2d_review_r2_answers.md;
P0-замеры: §8, скрипт /tmp/at2d_p0_diag.py, диск-only).

Связь с цепочкой T2: T2.1 (readout r1, floor-dominiрована) → T2.1b (AT2b,
SUBSPACE_CONTROL_FAIL: intercept-конфаунд per-doc базисов) → T2.1c (AT2c,
NOT_RESOLVED, control_ok=True: **H1 readout-часть отклонена** — weak-tail хуже
max(pca,haar) на всех (L,k)≥1; top доминирует readout'ом на всех (L,k);
G̸-проба §11.7: истинные слабо-наблюдаемые направления ≤ floor на k≥1) →
**T2.2 (этот документ) — НЕ readout-клетка, а интервенция**: zero-ablation
J-компонент state h_l и замер ΔCE на БУДУЩИХ позициях — тест «weak-сторона
causally используется моделью для предсказания будущего, даже если не
линейно читаема» (readout может пропустить state, который *используется*,
а не *хранится*).

## 0. Клетка одной строкой (r3)

Тот же T2-контур (32 дока × 128-префикс, слои L∈{6,12,18,22} (GDN) + {19}
(full attention), стационарный J-базис из кэша exp27b), но:
(а) **P_future-хук**: абляция применяется ТОЛЬКО на позициях t<64, метрика
ΔCE — ТОЛЬКО по позициям 64..126 (цели 64..127) — current-логиты
абляционного окна в метрику не входят by construction (ревью B3);
(б) **первичный sham = per-doc гауссов ортонормальный k-фрейм randG**
(каждый док — свой фрейм: draw-дисперсия входит в doc-bootstrap, ревью B1);
(в) **первичный эстиманд = energy-residual**: excess_res = ΔCE − ĉ·E,
где ĉ — коэффициент energy-фита ΔCE~E (softening-канал; ревью B2) —
«специфичность за единицу энергии», не «разность с одним draw»;
(г) арм weak_Gbar (canonical G̸-базис) — в сетке, secondary branch
(ревью B4); verdict-арм — weak_Gagg (энергетически согласован с sham'ом,
сравним с AT2c).

## 1. Входы

1. **Кэш exp27b** (как в AT2c, SHA-гейт S1 — 7 файлов, дословно таблица
   AT2c-пре-рега §7): `exp27b_jcache_L{6,12,18,19,22,23}.pt` +
   `exp27b_pool_ids_L23.pt` (AT2d использует L6/12/18/22/19 + pool_ids;
   L23 — positive control §13; L19 — в кэше с JR+P128, проверено 17.11).
2. **Токены 32 дока**: `exp27b_pool_ids_L23.pt` → `ids[dom][si]` (32 × 128
   int, forward-порядок i=0..127). Pool-верификация: `meta.pool_fp` (per
   domain sha256[:12]) сверяется с `pool_fp`, сохранённым в jcache-ckpt
   (гейт G2b). Пул НЕ реконструируется из потока (load_calibration_data/pin
   не вызывается — токены из кэша дословно; закрывает риск недетерминизма
   Hub-стрима).
3. **Модель**: `/data/Qwen/Qwen3.5-0.8B` (24 слоя, hidden 1024, гибридная:
   linear_attention (GDN) + full_attention каждый 4-й слой; layer_types из
   config.json: full на 0-based {3,7,11,15,19,23}, остальные linear —
   **AT2d-слои: L6/12/18/22 = GDN, L19 = full attention** (M9: L-контраст
   теперь между типами внимания, не «по глубине»)). Загрузка ТОЧНО как
   `experiments/e21_30/exp27b_dense.py` (собрал кэш): `AutoModelForCausalLM.
   from_pretrained(..., torch_dtype=float32, attn_implementation="eager")`.
   **fla на этой машине НЕ установлен** → GDN-слои идут torch-reference
   (на T4, где собран кэш, был fla 0.4.2) — расхождение учтено гейтом G2c
   (замер: cos ≥ 0.99999976 на всех 32×4, медиана 1.0).
4. **Окружение**: thread-pinning OMP/OPENBLAS/MKL_NUM_THREADS=4 +
   `torch.set_num_threads(4)` ДО импортов numpy/torch (дословно AT2c/AT2b;
   бит-идентичность форвардов на 4 тр. верифицирована замером: max|Δlogits|
   = 0 между повторами). .venv: python 3.13, torch 2.13.0+cu130,
   transformers 5.15.1, без fla.

## 2. Базисы (для ablation-проекций)

Все базисы — ортонормированные наборы направлений в пространстве state
(1024), вычисленные ИЗ КЭША (без новых форвардов), дословно по AT2c-коду
(experiments/at2/AT2c.py: строки 440–484 для J-базиса; 366–371/431 для
G32/HAAR):

- **Jagg** = mean_{d,t} JR_{d,t} ([128,1024], f64); **Gagg** = JaggᵀJagg
  ([1024,1024], rank ≤ 128); `torch.linalg.eigh`; order по убыванию λ;
  nz = {i: λ_i > 1e-12·λ_max} (ненулевые, по возрастанию).
- **top-k**: первые k по убыванию (k=4: λ-индексы 0..3; k=8: 0..7).
- **weak-k (VERDICT-арм, конвенция AT2c)**: первые k из nz по возрастанию
  (слабейшие ненулевые, weak-tail — конвенция T2.1/AT2b/AT2c).
- **pca-k** (per (L, док)): SVD центрированного H_doc (H_t − mean_t H_t) →
  первые k правых сингулярных векторов (дословно AT2c pca-арм).
- **haar-k** (глобальный фиксированный): G32 = standard_normal((1024,32))
  (rng = np.random.default_rng(0) — дословно AT2c), **HAAR_k = QR(G32[:,
  :k])** (k=4: 4 ст., k=8: 8 ст.).
- **weak_Gbar-k (CANONICAL арм, secondary branch)**: **G̸ = mean_{d,t}
  Jᵀ diag(P128) J** (средняя грам-наблюдаемость; P128 — диагональные веса
  [128] из кэша; накопление f64; субсэмпл 8 позиций t∈{0,16,…,112} × 32
  дока — AT2c §11.7 строил по всем 4096 (d,t), для report-арма
  субсэмпл достаточен, зафиксировано). eigh, nz, weak_Gbar = первые k из
  nz по возрастанию (к=4/8). P0-замер энергий: E(weak_Gbar4) = 0.064–0.114
  (L6/12/18/19), 0.002 (L22) — **не энергетически согласован с randG
  (k/D=0.0039)** → для weak_Gbar валиден ТОЛЬКО residual-эстиманд (sham-
  контраст механически ≈ −0.85, не интерпретируется).
- **randG-k (ПЕРВИЧНЫЙ sham, PER-DOC)**: для каждого дока d: A_d =
  standard_normal((1024, k)) (rng = np.random.default_rng(200_000 + L·10 +
  k + 1000·d)), **V_d = QR(A_d)**. Каждому доку — свой фрейм: draw-
  дисперсия sham'а ВХОДИТ в doc-bootstrap как per-doc разброс (ревью B1:
  shared-фрейм даёт общую смещённость, не покрываемую бутстрапом).
  Изотропность: E[V_d] ≈ k/D по среднему (P0.1: mean 0.00385–0.00400 vs
  theo 0.00391); draw-дисперсия ИЗМЕРЕНА (P0.1: per-doc sd 8.2e-5–1.25e-4
  на окне t<64; в ΔCE-единицах 1–1.6e-3 при |c|≈12.9 — вторична к doc-SE
  5.1e-3, но теперь покрыта CI). Фреймы записываются в артефакт
  (orth-проверка + sha per (L,k,d)).
  (r2-формулировка «дисперсия draw к draw мала (изотропия)» — неверна и
  заменена замером: изотропность даёт несмещённость среднего, не малую
  дисперсию; per-t rel-разброс 26–33% по замеру.)

r_final = 7 — зафиксировано AT2c main-прогоном (S5a rematched, все слои);
AT2d не пересчитывает d_eff (та же клетка). weak_mid и randJ — **выбыты**
из r3 (weak_mid заменён weak_Gbar; randJ — draw-artifact, P0/pre-study §8.9;
J-спектр как вопрос — follow-up с multi-draw). tw — **выбит** (M8: 2k
направлений против sham-размерности k — механический excess ≈ −0.13;
возможный реинтро с randG(2k) — только отдельным решением).

## 3. Пайплайн: форвард, хук, метрика

**Хук** (проверен pre-study: слой Qwen3_5DecoderLayer возвращает plain
Tensor; хук на `model.model.layers[L]`):
- **Вариант P_future (ПЕРВИЧНЫЙ, verdict)**: h' = h − V Vᵀh, применяется
  ТОЛЬКО при t < 64 (позиции 0..63); метрика ΔCE — ТОЛЬКО по позициям
  64..126 (target-токены 64..127; 64 значения). Current-логиты окна
  0..63 в метрику НЕ входят by construction (ревью B3: T2-эстиманд —
  future-only).
- **Вариант P (report, consistency с pre-study)**: h' = h − V Vᵀh на всех
  128 позициях; ΔCE по 127 значениям (0..126).
- (r2-вариант R — reflection — из main-сетки ВЫБЫТ: ‖δh_R‖ = 2‖δh_P‖ ⇒ при
  квадратичном отклике (pre-study R/P = 4.00/4.17) R предсказан из P
  тождественно — «независимая проверка» иллюзорна, DS2 тавтологичен
  (ревью M5). R сохраняется ТОЛЬКО в смоке как диагностик квадратичности:
  ratio R/P, ожидание ~4, отклонение = нелинейный информационный канал.)
- GDN-внутренние state напрямую НЕ трогаются (только residual stream
  h_l); модифицированный residual продолжается следующими слоями, GDN-
  слои обновляют свой рекуррентный state от изменённого входа — это
  часть causation-пути, который мерим (не дефект конструкции; зафиксировано).

**Конвенция позиционирования кэша** (ревью B-раунд 2, NIT #7): ключ кэша
H[(dom,si,t)] = позиция **127−t** (t-инверсия builder'а exp27b_dense.py:
hs[S−1−t]). G2c-сверка: H_fwd[L+1][:, pos] vs H_cache[(dom,si, 127−pos)] —
на всех 128 позициях (m1: r2 мерил только pos=127).

**Baseline (B)**: полный форвард дока (128 токенов) без хука:
CE_B(d) по softmax logits (f32). Отчёт: mean по 127 значениям (0..126),
И отдельно по окну 64..126 (m2: позиция 0 — нулевой контекст, другой
режим; primary-окно — 64..126).

**Метрики** (per (L, k, арм, вариант, док)):
- **ΔCE** = CE_arm − CE_B (nats/токен) — для P_future: среднее по 64
  значениям (позиции 64..126); для P: по 127.
  (Отрицательный ΔCE = абляция УЛУЧШИЛА предсказание — softening
  overconfidence, pre-study §8.)
- **E(arm, doc)** (ОВЕР-релиз report-only → КОВАРИАТА, ревью B2):
  E_win = mean_{t<64} ‖Vᵀh_t‖²/‖h_t‖² (окно абляции; per-doc — m3).
- **excess_res(арм, doc) = ΔCE(арм, doc) − ĉ(L,k)·E(арм, doc)** —
  ПЕРВИЧНЫЙ контраст: специфичность за единицу энергии (остаток
  energy-фита). ĉ(L,k) — коэффициент фита через ноль ΔCE_P_future ~ E
  per-doc по ВСЕМ 6 армам (192 точки на (L,k)); неопределённость ĉ
  входит в per-doc величину и покрывается бутстрапом.
- **excess_G(арм, doc) = ΔCE(арм, doc) − ΔCE(randG, doc)** — вторичный
  (per-doc пара: фрейм sham'а тот же док — draw-дисперсия в бутстрапе).
  Валиден для энергетически согласованных армов (weak_Gagg: E≈0.0025–0.0045
  vs k/D=0.0039; top — с нота об E≈2.5–5.5× больше). Для weak_Gbar —
  механический (≈ −0.85), НЕ интерпретируется, в отчёт с пометкой.
- **KL(арм, doc)** (ревью M6): mean по позициям 64..126 KL(p_arm ‖ p_base)
  (softmax-распределения, полный вокабуляр; вычисляется и сбрасывается,
  только скаляр в артефакт). Инвариантен к «правильности» — меряет
  величину функционального возмущения. excess_KL(арм) = KL(арм) −
  KL(randG) — сопутствующий контраст (разведён с ΔCE: «для выхода» vs
  «для правильного ответа»).
- **Стратификация ΔCE** (M6): per-doc ΔCE на позициях, где baseline
  PРАВИЛЕН (argmax p_base == target) vs НЕПРАВИЛЕН (позиции 64..126).
  «weak causally используется» обязано проявляться как ухудшение на
  правильных позициях, а не как общее смягчение.
- **Per-position ΔCE**: 5 бинов (64-76, 77-89, 90-102, 103-115, 116-127)
  per (дока, арм) (m4: затухание/послесвечение, прогрев).
- **norm_ratio** (report-only): mean_t ‖h'_t‖/‖h_t‖ на окне 0..63.

**Статистика**: unit = док (32). Doc-bootstrap B=1000, seed 0
(np.random.default_rng(0)): ресемплинг 32 дока с повтором; на каждом
бутстреп-семпле пересчитываются mean ΔCE/excess_res/excess_G/KL per
(L,k,арм) (арм и его per-doc randG-фрейм ресемплингуются ВМЕСТЕ — та же
выборка дока, пары сохраняются; per-doc фрейм ⇒ draw-дисперсия sham'а в
бутстрапе). CI95 = перцентили 2.5/97.5 (per-cell α = 0.05/10 для SIGNAL-
ветки — см. §5). Отчёт: mean, CI95, per-doc таблицы (ΔCE, excess_res, E,
KL).

**Energy-фит (S-energy, ревью B2)**: per (L,k):
- **ĉ_doc(L,k)** — через ноль, per-doc, все 6 армов (192 точки) →
  эстиманд excess_res.
- **R²_arm(L,k)** — фит ΔCE̅_arm ~ E_arm по 6 арм-средним (через ноль) →
  S-energy гейт (аналог AT2c S6c: «что мерит клетка»).

## 4. Контроли

1. **G1 (hook no-op, bit-exact)**: baseline прогоняется ДВАЖДЫ per док:
   B_plain (хук не зарегистрирован) и B_hook (хук зарегистрирован, но
   no-op: пустой V → h' = h). Гейт: max|Δlogits| = 0 на всех 32 дока.
   (Проверено pre-study: no-op хук даёт max|Δlogits| = 0.) Промах = стоп.
2. **G4 (sham responsiveness, sanity)**: |ΔCE_P_future(randG)| ≥ 0.005 на
   ≥8 из 10 (L,k) (pre-study-оценка через energy-фит: |ΔCE_P(randG)| ≈
   0.05–0.11). Нарушение на ВСЕХ = флаг «клетка не откликается», вердикт
   принуждается к NOT_RESOLVED с фиксацией (защита от тихого plumbing-
   срыва; НЕ жёсткий гейт).
3. **L23 positive control (m5)**: L23 = readout-слой (J там = readout);
   абляция top-J на L23 обязана давать БОЛЬШЕ damage, чем randG (64
   форварда: 32 дока × {top, randG} × P_future). Отчёт: ΔCE, KL, excess.
   Ожидание (не гейт): |ΔCE(top)| > |ΔCE(randG)| — клетка direction-
   specific, не «любое возмущение». Calibration-free (L23 в кэше).

## 5. Дерево вердиктов (r3 — симметричное, по первичному P_future)

Предварительно: S1, G1, G2a, G2b, G2c, S9, Ortho, S-energy = по гейтам.
Иначе — артефакта нет, стоп. Все ветки — по excess_res(weak_Gagg) per
(L,k), 10 клеток (5 L × 2 k), doc-bootstrap CI.

- **S-energy гейт → `ENERGY_DOMINATED`** (M7/m7: содержательный исход,
  не молчаливый excess): R²_arm(L,k) ≥ 0.85 на ≥4 из 10 клеток → «клетка
  меряет амплитуду возмущения (softening-канал), не направление; excess-
  контрасты — остатки energy-фита, отчёт». (Порог 0.85 пре-регистрирован:
  pre-study 6-точечный фит дал 0.804; 192-точечный per-doc фит при
  сохранении закона будет плотнее; 0.85 разделяет «закон» и «специфику».)
  T2 этой клеткой НЕ решается (readout-часть — AT2c).
- **H1-causal → `WEAK_CAUSAL_SIGNAL`** (M7: ≥2 клетки + Bonferroni +
  парные CI): **≥2 из 10 клеток**: CI_lower(excess_res(weak)) > 0 при
  per-cell α = 0.05/10 (Bonferroni; FWER ≥2 ложных ≈ C(10,2)·0.005² ≈ 1.2%)
  И на той же клетке парные CI (excess_res(weak) − excess_res(pca)) И
  (excess_res(weak) − excess_res(haar)) CI_lower > 0 при том же α. →
  «weak-сторона mean-J спектра causally специфична (не лучше по энергии,
  лучше по функции); readout (T2.1c) негативен, causal-контраст
  положительный».
- **`WEAK_CAUSAL_REFUTED_MEANJ`** (симметрия M7: ВСЕ клетки, не точечные
  оценки): на ВСЕХ 10 клетках CI_upper(excess_res(weak)) ≤ 0 (per-cell
  двусторонний α = 0.05). → «weak-сторона mean-J спектра causally НЕ
  специфична».
- **Secondary branch (pre-registered, canonical базис, ревью B4)**:
  - `WEAK_CAUSAL_SIGNAL_GBAR`: ≥2 из 10 клеток CI_lower(excess_res(weak_
    Gbar)) > 0 (α/10) — сильный результат: канонические слабо-
    наблюдаемые направления causally специфичны, хотя readout-ом ≤ floor
    (AT2c §11.7).
  - `WEAK_CAUSAL_REFUTED_GBAR`: на всех 10 клетках CI_upper ≤ 0 —
    canonical-часть тоже отклонена.
- **Закрытие T2** (формулировка B4-исправления — по базисам, не «два
  независимых дизайна»): T2 закрывается как «не подтверждено» (решение за
  супервайзером), ЕСЛИ: (i) readout отклонена на обоих базисах (AT2c:
  weak_Gagg — §11.3 CI; weak_Gbar — §11.7 probe ≤ floor на k≥1), И (ii)
  causal: WEAK_CAUSAL_REFUTED_MEANJ (и GBAR, если применимо). Частичные
  случаи (readout-отрицательно + causal NOT_RESOLVED; или MEANJ-refuted +
  GBR-неотрицательно) → T2 НЕ закрывается, отчёт по базисам.
- **`NOT_RESOLVED`** (иначе): смешанный случай (weak > 0 на каких-то
  клетках, но не ≥2; CI через 0; G4-флаг) — отчёт со всеми CI.

**Ожидаемые consistency-отчёты (не гейты)**:
- **excess_res(top) ≈ 0** — ИСПРАВЛЕНИЕ r2-ожидания: r2 предсказывал
  «excess_G(top) < 0 (top causally дёшев)» — это был энергетический
  артефакт (E(top) 2.5–5.5× больше E(randG) → больший softening → более
  отрицательный ΔCE → отрицательный excess_G механически). В residual-
  рамке честное ожидание: top не специфичен per unit energy (readout-
  доминирование не переносится в causation per unit perturbation).
  Систематический excess_res(top) > 0 = новый пост-хок факт.
- **L19 (full attention) vs L6/12/18/22 (GDN)**: если softening/
  специфичность различаются по типу внимания — содержательный отчёт
  (M9: теперь L-контраст осмыслен, не «профиль по глубине»).
- **L23 control**: |ΔCE(top)| > |ΔCE(randG)| (direction specificity).
- **KL-фит** (ΔCE vs KL по E): если KL-резидивалы ≠ ΔCE-резидивалы —
  разведение эстимандов (M6) — отчёт.
- **Стратификация**: специфичность weak (если есть) — на правильных vs
  неправильных позициях (M6).

## 6. Масштаб (зафиксирован замером, §8; r3-сетка)

- 32 дока (4 домена × 8), 128-префикс.
- L ∈ {6,12,18,22,19} (5: GDN×4 + full×1), k ∈ {4,8}.
- Армы: 6 (top, weak, weak_Gbar, pca, haar, randG-per-doc).
- Варианты: 2 (P_future — первичный, P — report).
- Форварды: 32×5×2×6×2 (3840) + 32 (B_plain) + 32 (B_hook, G1) + 64
  (L23 control: 32×{top,randG}×P_future) = **3968**. Замер 3.6–4.2 с/
  форвард → **≈ 4.6 ч** + bootstrap (минуты) + загрузка (2–3 мин) +
  G̸-конструкция (1 мин) ≈ **5–5.5 ч** total. Fallback: wall > 9 ч —
  стоп и разбор (масштаб фиксируется; урезание сетки — только по явному
  решению).
- Smoke-сетка (§11): 2 фиксируемых дока (wiki,0),(wiki,1) — (ревью B-
  раунд 2, MINOR #2: случайный выбор давал P(флипа DS2) ≈ 9%), L6, k=4,
  все 6 армов × 2 варианта + R-диагностика (top4) + B_plain + B_hook =
  2×(12+1+2) = 30 форвардов/прогон; S9-пара = 60 ≈ 4.5 мин.

## 7. Гейты (до записи артефакта; провал = артефакта нет, стоп)

- **S1**: sha256 7 файлов кэша (таблица AT2c S1, дословно).
- **G1**: hook no-op бит-в-бит (max|Δlogits| = 0, 32 дока).
- **G2a**: структура pool_ids: 4 домена × 8 дока × 128 токенов, int-типы.
- **G2b**: pool_fp per домен (meta pool_ids) == pool_fp per домен в
  jcache-ckpt L6 (все 4 совпадают).
- **G2c (state provenance, m1: все позиции)**: per (док, L∈{6,12,18,22,
  19}) cos на ВСЕХ 128 позициях (H_fwd[L+1][:,pos] vs H_cache[(dom,si,
  127−pos)]): гейт min ≥ 0.999 И отчёт median (pre-study: min
  0.99999976 на позиции 127, медиана 1.0). Промах = стоп.
- **S9 (детерминизм, смоки)**: два smoke-прогона → md5 артефакта
  бит-в-бит (дословно AT2c S9).
- **Ortho (m6: f64)**: ‖VᵀV − I‖ < 1e-6 на всех (L,k,арм,док),
  вычисляется в **f64** (замер: f64 ≤ 2e-15, f32-накопление ~4e-6 —
  у порога; r2-формула §2 «‖VVᵀ−I‖» исправлена на ‖VᵀV−I‖ — VVᵀ для
  k-фрейма rank-k проекция, ‖VVᵀ−I‖₂=1 всегда). Стоп при провале.
- **S-energy (новый, §3/§5)**: R²_arm(L,k) computed per (L,k); гейт-
  флаг ENERGY_DOMINATED (≥4 из 10 клеток ≥ 0.85) — не стоп, ветка
  вердикта.
- **G3 (trackio)**: run at2d_t22_r1: t=0 config (started_at, **device=
  cpu**, revision, script_sha, seed=0, k, layers, n_docs=32, boot_b=1000,
  model, venv, log) + stage-точки (load / prov / Gbar / forwards-L{…} /
  l23_control / energy_fit / bootstrap / artifact) + финальная (t=
  duration, **finished_at, status: ok|crash|interrupted, log_md5** — в
  finally/atexit; crash — НЕ молча: дописать status=crash и finish()).

## 8. Pre-study + P0-замеры (17.11; НЕ гейты — основа дизайна)

Pre-study (L6, 32 дока; .venv, CPU 4 тр., f32 eager, без fla; скрипты
/tmp/at2d_{speed,hooktest2,power,energy,pos,reflect,seedtest}.py):

1. **Скорость**: форвард 128 токенов 3.3–3.8 с; ~4.2 с/форвард с хуками.
2. **Детерминизм**: повторные форварды (plain и ablated) бит-идентичны
   (max|Δlogits| = 0).
3. **State provenance (G2c)**: cos min 0.99999976 (code,2,L18, позиция
   127), медиана 1.0, 128 пар (32 дока × 4 слоя). L23 точное совпадение.
4. **Хук**: plain Tensor; no-op бит-в-бит; абляция распространяется на
   downstream (max|Δh| = 18.7 на L23 при L6-абляции); DS1-основание:
   max|Δh| = 0.18 (wiki si0, top4).
5. **CE_plain ≈ 14.7 nats/токен** (wiki doc0) > ln V = 12.42 — модель в
   среднем «confidently wrong»; механизм «улучшения» от абляции —
   softening: corr(CE_plain_i, ΔCE_i) = −0.31 (doc0, top4) (M6).
6. **Все армы дают ОТРИЦАТЕЛЬНЫЙ ΔCE_P** (32 дока, L6): top4 −0.150±0.157,
   weak4 −0.027±0.029, randJ4(draw1) −0.154, top8 −0.245±0.175,
   weak8 −0.101±0.059, randJ8(draw1) −0.214.
7. **Энергия E** (full-окно): top4 0.0102, weak4 0.0037, randJ4 0.0067,
   top8 0.0201, weak8 0.0095, randJ8 0.0168; k/D = 0.0039 ≈ E(weak4).
   **ΔCE_P ≈ c·E (c ≈ −10…−15) — softening-канал** (B2).
8. **Варианты**: R ≈ 4×P (top4: −0.600 vs −0.150 = 4.00; top8: −1.022
   vs −0.245 = 4.17) — **квадратичный отклик** (M5: ‖δh_R‖=2‖δh_P‖);
   nr ≈ P (−0.145 vs −0.150) → норм-канал незначим.
9. **randJ — draw-artifact**: excess(weak4−randJ4) по 3 seed'ам:
   +0.127/+0.020/−0.005 → randG — первичный sham, randJ — выбит.
10. **L6-вывод pre-study**: weak не выделен явно относительно J-draw'ов;
    главный вопрос main: специален ли weak относительно энергетически
    согласованных направлений (теперь — per-doc, residual-эстиманд).

**P0-замеры раунда 2** (disk-only из кэша, 17.11; скрипт /tmp/at2d_p0_
diag.py, результат /tmp/at2d_p0.json):

- **P0.1 — sham-дисперсия на реальных states** (B1; 200 draw × 8 дока,
  окно t<64, k=4): mean 0.00385–0.00400 (theo 0.00391); per-doc sd по
  draw'ам 8.2e-5–1.25e-4 (ΔCE-единицы при |c|=12.9: 1.1–1.6e-3); between-
  doc 2.9e-5–6.0e-5; pooled 1.0e-3–1.3e-3 (per-t rel-разброс 26–33%).
  Вывод: r2-утверждение «малая дисперсия (изотропия)» неверно; draw-
  дисперсия window-усреднением подавлена ~√64 и вторична к doc-SE
  (5.1e-3), но shared-фрейм оставляет общую смещённость вне бутстрапа →
  **per-doc фреймы** (в r3).
- **P0.2 — energy-фит** (B2; pre-study 6 точек L6, через ноль): c =
  Σ(E·ΔCE)/ΣE² = **−12.93**; R² = **0.804**; остатки: top4 −0.0181,
  weak4 **+0.0208** (≈ заявленный excess +0.023 — весь сигнал в остатке
  energy-фита), randJ4 −0.0674, top8 +0.0149, weak8 +0.0218, randJ8
  +0.0032. Циркулярность r2: randG в pre-study НЕ прогонялся, его ΔCE
  «0.05–0.11» — предсказание этим же фитом → в r3 randG per-doc в сетке,
  S-energy гейт пре-регистрирован.
- **P0.3 — квадратичность** (M5): R/P = 4.00 (top4), 4.17 (top8) → R
  предсказан из P; R из main-сетки выбит (smoke-диагностика).
- **P0.4 — G̸-базисы и энергии** (B4; субсэмпл 8 поз × 32 дока, f64):
  E_win(окно t<64, mean32±sd32): L6 top4 0.0126, weak4 0.0045,
  weak_Gbar4 0.0726; L12 0.0170/0.0036/0.0637; L18 0.0199/0.0033/0.1144;
  L22 0.0199/0.0025/**0.0020**; L19 0.0216/0.0033/0.0760 (weak_Gbar8:
  0.0778/0.0666/0.1203/0.0042/0.0820). Вывод: canonical weak-арм НЕ
  underpowered (энергия 16–27× больше weak_Gagg на L6/12/18/19; L22 —
  наоборот 0.002) и НЕ энергетически согласован с randG → для него только
  residual-эстиманд; verdict-арм остаётся weak_Gagg.
- **P0.5 — per-doc разброс E(weak4)** (мощность residual-эстиманда):
  sd32 = 0.00059–0.00102 → |c|·sd_E = 7.6e-3–1.3e-2 vs per-doc sd ΔCE
  (weak4) 0.029: энерг-канал объясняет значительную часть between-doc
  вариации, per-doc шум 0.026–0.029 → SE 32-докового mean excess_res ≈
  0.0046–0.005 → детектируемый excess_res ≥ 0.01–0.015 (z ≥ 2.2–3.3);
  pre-study остаток weak4 +0.0208 — в зоне детектируемости. Мощность
  ДОСТАТОЧНА для реалистичного сигнала.

## 9. Ожидания (не гейты)

1. **excess_res(top) ≈ 0** на большинстве (L,k) (top не специфичен per
   unit energy; см. §5 — исправление r2-ожидания).
2. **|ΔCE_P_future(randG)| ≈ 0.02–0.06** (окно-версия energy-фита; pre-
   study full-окно 0.05–0.11 × доля окна) — клетка откликается (G4).
3. **S-energy**: R²_arm на большинстве клеток 0.7–0.9 (pre-study 0.804) —
   вероятен исход ENERGY_DOMINATED (честный отрицательный результат про
   клетку).
4. **weak (Gagg)**: без обязательства (H1-causal либо подтвердится ≥2
   клетками, либо отклонится на всех).
5. **weak_Gbar (G̸)**: без обязательства, secondary branch (canonical).
6. **L19 vs GDN-слои**: вариация по типу внимания (отчёт, не
   предсказание).
7. **L23 control**: |ΔCE(top)| > |ΔCE(randG)| (direction specificity).

## 10. Что НЕ делается (scope, r3)

- Никакой генерации: только forward на заданных 128-префиксах.
- Никакой интервенции во внутренние GDN/recurrent state напрямую (только
  residual stream; обновление GDN-state от изменённого входа — часть
  causation-пути, §3).
- **tw** (M8: 2k vs k — механический excess; реинтро с randG(2k) — только
  отдельным решением).
- **weak_mid, randJ** (заменены/выбты, §2).
- **R-вариант в main** (M5: квадратичный, предсказан из P; smoke-
  диагностика только).
- Никакого nr-варианта (pre-study §8.8).
- Никаких k ∈ {1,2,16}; никаких L, кроме {6,12,18,22,19} (L11 не в кэше —
  JR/P128 потребовали бы новых jacobian-форвардов, не price-neutral).
- Никакого per-position разбора в вердикт (бины + стратификация — отчёт).
- Никаких GPU/TPU (CPU-only, один прогон за ночь по AGENTS).

## 11. Смоки (до main-прогона, отдельный шаг; AT2d_SMOKE=1)

Сетка: 2 **фиксируемых** дока (wiki,0),(wiki,1) (pre-study значения
wiki si0: DS1 0.18; DS2 −0.150/−0.600; wiki si1 — пересчитывается на
смоке как baseline-отчёт), L6, k=4, 6 армов × 2 варианта + R(top4) +
B_plain + B_hook (30 форвардов/прогон).

1. **S9 (детерминизм)**: два smoke-прогона → md5 артефакта бит-в-бит.
2. **G1 (hook no-op)**: max|Δlogits| = 0 (стоп иначе).
3. **DS1 (хук действует)**: max|Δh| на выходе хука (P_future, top, per
   док) ≥ 0.1 (pre-study: 0.18) — стоп иначе.
4. **DS2 (plumbing + softening активен, замена r2-DS2)**: mean по 2
   smoke-докам: ΔCE_P_future(top4) < 0 И ΔCE_P_future(randG4) < 0 — стоп
   иначе (plumbing/метрика сломаны; r2-версия «|R|>|P|» тавтологична при
   квадратичном законе — M5/m8).
5. **DS3 (квадратичность, диагностика, НЕ стоп)**: R/P (top4, 2 дока) —
   отчёт, ожидание 2..6 (pre-study 4.0); отклонение = нелинейный канал.
6. **Ortho (f64)**: ‖VᵀV−I‖ < 1e-6 (стоп).
7. **G2c** на 2 smoke-доках: min по 128 позициям ≥ 0.999 (стоп).
8. **randG-фреймы per-doc**: записаны в smoke-артефакт (orth + sha).

## 12. Провенанс (что на чём стоит)

- **Кэш exp27b** (T4, fla 0.4.2, transformers 5.12.1): S1 sha + G2b
  pool_fp + G2c state provenance (3 независимые верификации, §7).
- **J-базис**: код AT2c.py:440–484 (Jagg/Gagg/eigh/sort) — дословный copy
  в AT2d.py; тот же r_final = 7 (AT2c S5a). **G̸-базис**: /tmp/at2c_
  gram_check.py конструкция (AT2c §11.7) + субсэмпл-векторизация
  /tmp/at2d_p0_diag.py; AT2c-версия (все 4096 (d,t)) — для сверки
  report-арма.
- **Токены**: pool_ids из кэша (G2a/G2b), не реконструкция.
- **Pre-study + P0**: замеры §8 (скрипты в /tmp, повторяемые) — основа
  r3-дизайна (per-doc sham, P_future, residual-эстиманд, S-energy, L19,
  G̸-арм). Pre-study/P0 — L6 только (+ P0 по всем 5 слоям, disk-only).
- **AT2c-ревью раунда 8 (17.11)**: docs/AT2c_review_r8_answers.md — F1
  (ratio закрыт readout-замером) и G̸-probe (mean-J ≠ mean observability
  spectrum) — основание weak_Gbar-арма и secondary branch.
- **AT2d-ревью раунда 2 (17.11)**: docs/AT2d_review.md — B1–B4, M5–M9,
  m1–m8; ответка docs/AT2d_review_r2_answers.md; все блокирующие пункты
  закрыты редизайном r3 (цена-нейтрально: 3968 форвардов ≈ 5–5.5 ч vs
  r2 4160 ≈ 5–6 ч).
- **ЗАВИСИМОСТЬ ОТ AT2b/AT2c-КОДА** (как в r2): если в базисах/кэше/
  доках найден серьёзный баг — перепуск с исправленным кодом.
- **Модель**: /data/Qwen/Qwen3.5-0.8B (weights sha — в артефакте meta;
  layer_types — в meta, M9).
- **Окружение**: .venv (torch 2.13.0+cu130, transformers 5.15.1, без fla),
  CPU 4 тр., python 3.13.

## 13. Схема артефакта (results/AT2d/AT2d.json)

```
{
 "meta": {revision:"r1", script_sha, seed:0, k:[4,8],
          layers:[6,12,18,22,19],
          layer_types: {L: "gdn"|"full"} (все 24 из config, M9),
          n_docs:32, boot_b:1000, model, venv, wall_sec, started_at,
          finished_at, trackio_run, log_md5, model_weights_sha (per file),
          at2c_script_sha, gbar_subsample: {ts:[8], docs:32, dtype:f64}},
 "gates": {S1: {file: sha}, G1_hook_noop: {max_logit_diff, per_doc, ok},
           G2a_pool_ids: {...}, G2b_pool_fp: {...},
           G2c_state_prov: {cos_min_all_pos, cos_median, worst, ok},
           ortho_f64: {max_norm_diff, ok},
           randG_frames: {L|k: {d: {sha, ortho_ok}}},
           s9_smoke: {md5_pair, ok},
           smoke_ds: {DS1, DS2, DS3_ratio_R_P, ortho, G2c_smoke},
           s_energy: {R2_arm per (L,k), c_doc per (L,k), n_cells_ge_0.85,
                      flag}},
 "baseline": {per doc: CE_B_full127, CE_B_win64},
 "arms": {"L{L}k{k}": {"P_future": {"<арм>": {per_doc:[32]: {dCE, E,
          excess_res, excess_G, KL, dCE_correct, dCE_wrong, bins5:[5]}},
          mean, ci95, excess_res_mean, excess_res_ci95,
          excess_G_mean, excess_G_ci95, excess_KL_mean}},
          "P": {...report...}}, ...},
 "l23_control": {per doc: top/randG: dCE, KL, mean, ci95, ratio},
 "expectations": {excess_res_top_near_zero: per (L,k) bool,
          randG_responsive: per (L,k) |ΔCE|, L19_vs_GDN: per metric,
          l23_specificity: bool},
 "verdict": "...", "verdict_detail": {per (L,k): excess_res(weak)
          mean/ci95, pairwise vs pca/haar, per branch},
 "deviations": []
}
```

Smoke-артефакт: results/AT2d_smoke/AT2d_smoke{,_det}.json (S9-пара).
Артефакт ~0.5–1 MB (per-doc × 5 бинов × стратификация) — в git входит.

## 14. Коммит-план

Код, ТЗ (r1+r2+r3-заметки), ревью-файлы (docs/AT2d_review.md + ответка),
results (лог + артефакт + smoke) — в git явными путями. CJK-гейт,
маркеры ревью — по AGENTS.
