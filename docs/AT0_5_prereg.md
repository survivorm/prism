AT0_5 — T0.5 «Генеративная проверка (disk-only)» — пре-рег (текст = хедер скрипта,
docs/AT0_5_prereg.md)

Дата: 13.11.2026. ТЗ: docs/TZ_add_0-3.md §1.5 T0.5 (sha см. в артефакте).
Контекст (T0.1): DC+MA(1) refuted на пуле (p_perm = 1/199), но объясняет ≈93% уровня
плато; остаток 6.2–6.6% статистически надёжен ⇒ estimand T0.5 — **per-doc** (G10):
воспроизводит ли минимальная линейная модель пути (DC+MA(1), параметры fit-нуты на
доке) dip+плато **у каждого дока**. Вердикт — IN-SAMPLE GOF (ревью r4 C3): a из ρ̂(1)
⇒ дип циркулярен по построению; held-out/generative-claim НЕ делается (OOS — только
plat_hi_oos, reportable). Оба исхода публикуемы.

Клетки: 96 доков (gen 64 + P1_own 16 + P2_corpus 16) × L ∈ {1,5,11,17,23} = 480 − skips
(k < 16; T0.1 K=0: own_w0/own_w4 L1, fp16_l1_degraded).
Объект: joint (v̂_t, log‖Δh_t‖), h = u·hn (load_source AT0_1; TPU/corpus — raw fp16,
‖row‖ пересчитан; построение U/h — ТОЧНО машина T0.1: U = u/max(‖u‖,1e-12), h = U·hn),
Δh_t = h_{t+1} − h_t, t = 0..382 (N = 383). v̂_t = Δh_t/‖Δh_t‖ (guard ‖Δh_t‖ > 1e-12,
вырожденный шаг — fail-loud), n_t = log‖Δh_t‖.

Часть 1 (РЕШАЮЩАЯ) — DC+MA(1) per-doc. Модель (G8): x_t = m + (ε_t − aε_{t−1}),
ε ~ N(0,1) iid, m = const (DC-компонент; f_DC-плато безцентр. ACF возникает из m²-
вклада в c(k)). ACF в безцентровом T-нормировочном согласии (поле.acf_moments):
ρ̄(1) = (1−1/N)(f − (1−f)a/(1+a²)), ρ̄(k≥2) = (1−k/N)·f — DC-компонент несёт
наклон (1−k/N), а не плоское f (ревью р2). m² = f(1+a²)/(1−f). Фит per-doc (pooled
запрещён — E-56): (f, a) — LS к ACF Δh (T-нормировка, k=1..96): f = mean_k ρ̂(k)/(1−k/N)
(k=2..96); a из ρ̂(1)/(1−1/N) = f − (1−f)a/(1+a²) closed-form (корень с |a| < 1 —
стационарность MA); 1 шаг
Gauss-Newton с backtracking (≤8 halving); выход из области (0≤f<1, |a|<1) исключён
(backtracking принимает только доменные шаги, начальные доменные — raise недостижим);
out_of_model (disc ≤ 0, d ≥ (1−f)/2 — unit-корень/вне области) — reportable flag;
acf_r2 — R² ACF-фита (reportable). f ≤ 0 ⇒ f заморожен = 0 (GN по f
не выполняется, gn_skipped), a — из closed-form с (1−f) = 1 (чистый MA(1), если ρ̂(1) < 0; a = 0, если
ρ̂(1) ≥ 0) — flag f_le_0 = «DC нет». OOS-вариант (reportable, разделение GOF/out-of-sample):
fit k=2..48 (нормировка (1−k/n) — PER-CELL n) → симуляция → тест plat_hi_oos (k=49..96)
— ОДНА метрика, слабая проверка;
дип НЕ тестируется OOS (a из ρ̂(1) ⇒ циркулярно); основной вердикт — по TZ (fit
k=2..96). Симуляция — D-МЕРНЫЙ null из FIT-НУТЫХ параметров (ревью р3 + р4 C2):
x_{t,j} = m′_j + s′_j·(ε_{t,j} − aε_{t−1,j})/√(1+a²); масштаб: total power сохраняем
(Σ(m′²+s′²) = Σ(m²+s²) = E), DC-доля = fit-нутый f (Σm′² = f·E, Σs′² = (1−f)·E),
анизотропия сохраняется (направление m_j, форма s_j); f = 0 ⇒ m′ = 0. Старая версия
(real m/s as-is, f не передавался) давала f_DC_sim = f_DC_real BY CONSTRUCTION
(f_dc = ‖mean‖²/mean‖x‖²) ⇒ покрытие плато было циркулярным. B = 200 путей длины n_i,
seed = 100000 + 5·doc_idx + layer_idx (layer_idx = 0..4 по (1,5,11,17,23)). Метрики (все scale-invariant): dip = ρ̂(1), plat_lo = mean
ρ̂(k), k=9..32, plat_hi = mean ρ̂(k), k=33..96 — для real и каждого sim-пути.
Per-doc covered = real (dip, plat_lo, plat_hi) все ∈ [q_0.025, q_0.975] sim-
распределения (короткая клетка — по доступным полосам, sensitivity). Агрегат:
per-layer frac_covered по PRIMARY-пулу (non-degen ∩ K96; degen_tz — флаг T0.1 store;
degen-пул и short-клетки (16 ≤ k < 96) — отдельные отчёты, в вердикт не входят;
per-source стратификация gen/own/corpus — reportable; corpus: h-данные есть, токены
0/16 — G16 not_testable = только provenance-оговорка). КАЛИБРОВКА (ревью r4 C4,
reportable, не гейт): d_eff per-cell = participation ratio per-dim дисперсий
(анизотропия: d_eff ≪ 1024 ⇒ effective dof мал); null-on-null: 2 независимые партии
B=200 из fit-нутых (f,a) — партия A → квантили [q2.5,q97.5], партия B → marginal/joint
покрытие (первые K96 non-degen доки per layer); quantile MC error при B=200 ≈ 0.19σ
(meta). ВЕРДИКТ (IN-SAMPLE GOF, ревью r4 C3 — generative-claim не делается): branch1
«три моды мертвы как in-sample generative объяснение» (T3 остаётся только горб) ⇔
frac_covered ≥ 0.5 на КАЖДОМ слое по eligible клеткам; пустой primary-пул на слое ⇒
branch2; иначе branch2: структура сверх DC+MA(1) есть, T3 получает работу.
Reportable: r_sim/f_DC_sim/Σ96_sim/bump_sim по sim-пути + gaps real−sim; per-док
(f, a, m, acf_r2, f_fit_vs_real, d_eff).

Часть 2 (reportable) — лестница VAR per-doc. PCA(v̂) per (doc,L) (SVD 383×1024).
Сетка r ∈ {8,16,32} × p ∈ {1,2,4}: VAR(p) на r-проекции v̂, LS-фит (intercept;
регрессия x_t = c + Σ_i A_i x_{t−i}; n = 383−p — число LS-выборок; SSR = Σ_t‖e_t‖²).
BIC (PRIMARY, ревью r4 C6) = n·ln det Σ + (p·r²+r)·ln n — гауссов likelihood,
Σ = EᵀE/n (full df: A_i — r×r); bic_tz = n·ln(SSR/n) + (p·r+r)·ln n (TZ —
SENSITIVITY, reportable: trace-SSR теряет cross-output covariance, TZ-df
недоучитывает A_i; ДОП. смещение: BIC по СУММАРНОМУ SSR — низкодисперсионные
компоненты почти ничего не платят за свои df ⇒ selected смещён к r=32 сильнее, чем
df-объяснение). selected (r,p) = argmin BIC (primary); selected_tz — по TZ
(reportable). Норма —
AR(1): n_t = α + βn_{t−1} + ε (LS; reportable β, R²; в BIC-выбор не входит).
Генеративный путь с selected (r,p): B = 100 (seed = 200000 + 5·doc_idx + layer_idx):
v̂̃ ∈ R^r — ЕДИНИЧНОЕ направление сгенерированного пути (X/‖X‖ по оси r; VAR,
нач. усл. нули ⇒ первые p шагов ‖X_t‖ = 0 — артефакт нач. условий, ≤ p/383 ≈ 1%
ACF-массы, задокументировано),
ñ_t (AR(1), нач. ñ_0 = mean(n_real), e ~ N(0, sd_res)), Δh̃_t =
exp(ñ_t)·v̂̃_t (r-мер) → метрики (r, f_DC, dip, plat_lo, plat_hi, Σ96) + gaps real−sim.
|β| ≥ 1 ⇒ ar1_unstable flag, симуляция пропускается. Примечание: real D = 1024,
sim D = r — gap = «доля структуры в r главных проекциях», НЕ точное воспроизведение.
DC+MA(1) в лестнице — L0, ВНЕ BIC-выбора (другое семейство: MA-ошибка vs AR-регрессия)
— отчитывается ACF-fit R² (reportable).

Часть 3 (гейт G8) — negative control. Blk-shuffle Δh (b = 8 по t; 383 = 47×8 + 7,
хвост 7 остаётся на месте; 5 сидов 1001..1005; перестановка блоков = перестановка
строк v̂/scores — проекция коммутирует с перестановкой строк, SVD один per (doc,L)).
Свойство блок-нуля: внутри-блок структура (k < b) сохраняётся — BIC-лестница на null
закономерно может выбирать p > 1 (это сохранённая локальная структура, не баг).
«Плоская кривая» операционализуется по ДАЛЕКОМУ хвосту: при k > b пары всегда в
разных блоках ⇒ корреляции нет. Gate (G8-v3, 15.11): (а) median |tail_excess|
(plat_hi − (1−k̄/N)·f_DC, k=33..96, k̄ = 64.5) по (doc,L,seed) ≤ 3·sd_EMP, где
sd_EMP = std эмпирического null-распределения (пул по всем клеткам). Параметрическая
D-мерная DC+MA(1)-симуляция занижает флуктуацию null в 1.6–5.5× (реальные строки Δh —
не гауссовы по колонкам: концентрация row-power ⇒ дисперсия произведений пар больше,
чем в sim с iid-колонками) — sd_null (sim) отчитывается как калибровочная
диагностика, НЕ как масштаб гейта; (б) sim-среднее (TAIL-FIT: f на k=33..96, a из
лаг-1; внутри-блок структура k<8 сохранена по построению и из fit исключена) RAW —
согласование с null-статистиками на эмпирическом масштабе: frac клеток с |null −
sim_mean| ≤ 3·sd_pool_stat ≥ 90% (dip, plat_lo, plat_hi — RAW; без base-вычитания:
sample f_dc sim перегружен шумом среднего MA(1) ≈ V/n ≫ f — баг G8-v3).
Reportable: распределение selected (r,p) на null + BIC-сводка + raw-покрытие sim.
Нарушение (а)/(б) ⇒ баг пайплайна.

Нормировка (TZ): все отчитываемые метрики scale-invariant (ρ̂, r = 1+2Σρ̂ (G1), f_DC
(G3), dip, plateau, bump). Σ/S(0) — ТОЛЬКО как r; абсолютные Σ и ‖Δh‖² не сравниваются
real-vs-sim (разные D).

Гейты (fail-loud, до вердикта):
G0: sha256 3 npz + gen_sha 64/64 + decode-probe 3 строки/gen + 1 строка/own;
    P2_corpus — not_testable (G16, 0/16 tok).
G1′: field.py sha-префикс bcc607c561f95e3e + field._selfcheck().
G2: все клетки (480 − skips; skips: k < 16, T0.1 K=0), per-cell ключи полные.
S0: rho_metrics ≡ field.acf_moments (exec на 2 матрицах, точность 1e-14).
S1: телескоп (TZ §1.7) per (doc,L): ‖ΣΔu − (u_T−u_0)‖ < 1e-6.
S2: iid-нуль (ТЗ §1.7, D5): DC+MA(1) (f=0,a=0), B=2000: mean ρ̂(1) = 0 ± 3·SE.
    UNCENTERED iid: E[ρ̂(k)] = 0 ТОЧНО (симметрия по знаку одной координаты);
    −1/(N−1) — target DEMEANED ACF, не наше соглашение (ревью r4 C1). D13: старый
    «срыв на 2.81 SE» был артефактом wrong target — тот же seed 42: mean = +0.000593
    = 0.52 SE от 0; tol 3·SE сохранён как консервативный запас.
S3: AR(1) φ=0.5, B=2000: mean ρ̂(1) ∈ [0.3, 0.6] (ориентир +0.414).
S4: period-2: x_t = (−1)^t: чётный N=384 ⇒ r = 0; нечётный N=383 ⇒ r = 1/N
    (краевой член, свойство uncentered T-норм. ACF; старый докстринг «r=0» для odd N
    неверен — ревью r4 C7.3).
S5: Fourier-суррогат (positive control): 1 док L23, seed=7; фазовая рандомизация с
    сопряжённой симметрией ⇒ КРУГОВОЙ ACF сохраняется с точностью ≤ 1e-6
    (max |c_circ(сурр.) − c_circ(ориг.)|; круговой r самих данных — O(1) из-за
    DC-плато, так что «r ≤ 1e-6» на данных невозможно — (r) в TZ = расхождение);
    ЛИНЕЙНЫЙ ACF отличается wrap-around хвостом — отчитывается отдельно, «свойство,
    не баг» (урок AT0_2: линейный 1e-6 для spectrum-preserving суррогата
    конструктивно недостижим).
S6: gram_resid < 1e-10 на всех 480 клетках (G8-аудит field.acf_moments).
S7: r = N·f_DC (G3) ≤ 1e-12 на всех 480 клетках.
S8: Σρ̂_full перм-инвариантность: 1 док L23, 5 сидов, spread < 1e-12.
S9 (ЯКОРЬ): ACF Δh per (doc,L) == T0.1 store (results/AT0/store/23281c64cc418ff1.
    json, per_doc[doc].layers[L].h_raw.mean.rho) ≤ 1e-12; якоря 478 (own_w0/own_w4
    L1 — T0.1 K=0, L1-TPU деградация — skip, в клетках не участвуют); + K/N-сверка
    с T0.1 (ревью r4 C5.1): N_store == n; K-алфавит {0,24,32,96}: K=96 ⇔ k=96,
    K=32 ⇔ 32≤k≤95, K=24 ⇔ 24≤k≤31, K=0 ⇒ клетки нет. Пин — на STORE: sha256
    23281c64cc418ff1fedad2a5013c4bf7c8b7dd7680ac8d4fa7cf746c62fd1732 (бит-в-бит к TZ);
    claim-файл T0_ACCOUNTING.json (TZ-шина d4a4b101…) был перезаписан AT0_1 r5
    (коммит 6c1cdf3) — его sha только отчитывается.

Числа-ссылки (не пересчитывать без причины): T0.1 плато ≈ f_DC ≈ 0.057–0.065
(МЕДИАНА ПУЛА, не per-doc значение: f_DC per-doc варьируется, например до ~0 —
сверять per-doc с пулом нельзя);
избыток +0.00194 (v_tangent); горб +0.0121/−0.0134/−0.0137; E_null blk-2 = −0.002228;
аналит. E_null = −(1−f)/(N−1).

Формат: claim results/AT0/AT0_5_GENCHECK.json; store results/AT0/store/<sha>.json;
run.log; manifest. RUN_RC=0 + gates ALL PASS — обязательны; гейт с отсутствующим/
False ok ⇒ RUN_RC=1, claim/manifest НЕ пишутся (ревью r4 C7).
## Addendum r4 (фиксы после внешнего ревью docs/AT0_5_review.md, 13.11.2026,
sha 71f4a77f91b070ed302227c03ed6819a9bc57d5f8c82c2709a54197e49ee63fc)
Эксперимент не запускался (0 артефактов) — все правки ДО первого полного прогона:
- C1: S2 target 0 (uncentered iid: E=0 точно), не −1/(N−1) (demeaned-target);
  D13 «срыв 2.81 SE» был артефактом wrong target (тот же seed: 0.52 SE от 0).
- C2: D-null из FIT-НУТЫХ (f,a) (вариант A): total power сохранён, DC-доля = f_fit;
  plateau-покрытие не циркулярно; f_fit_vs_real — reportable.
- C3: вердикт переименован в in-sample GOF; generative-claim не делается
  (OOS — только plat_hi_oos, reportable).
- C4: d_eff per-cell + null-on-null calibration (2 партии B=200: A → квантили,
  B → marginal/joint coverage) + quantile MC error (≈0.19σ при B=200) — reportable.
- C5.1: k = min(96, n−1), n = min(T,384)−1 (= N T0.1 store; 0 расхождений);
  старая min(96, n//4) была ложна (82/480 расхождений; S9 rho-якорь не ловил).
  S9 + K/N-сверка с T0.1 (K-алфавит {0,24,32,96}).
- C5.2: primary-пул = non-degen ∩ K96 (все 3 полосы); short (16≤k<96) —
  sensitivity; eligibility явная (n_eligible_k96 per layer).
- C5.3: per-source стратификация (gen/own/corpus) per layer; corpus h-данные
  ЕСТЬ (0/16 tok = G16, только provenance-оговорка) — corpus в primary-пуле.
- C6: BIC primary = n·ln det Σ + (p·r²+r)·ln n (logdet, full df); bic_tz —
  sensitivity; selected — по primary, selected_tz — по TZ.
- C7: ok строго is True (гейт без ok — FAIL); RUN_RC=1 без claim/manifest при FAIL;
  S4-докстринг odd/even; OOS (1−k/n) — per-cell n (был глобальный N=383 — баг).
- Чистка: dcma1_sim (1-D, мёртвый код), mёртвые параметры (k в sim_d_fit,
  seed в r_of).

## G8-v2 ре-спека (15.11, ДО перепуска; первый прогон умер на G8 — fail-loud, до артефактов)

**Сбой v1 (измерено):** median |tail_excess|/sd_null = 4.29 > 3.0 и frac_dcma1_cov = 0.091
< 0.90 (n = 2400 null-клеток). Триаж (не баг пайплайна: S0/S5/S8/S9-гейты все ок, S9
бит-в-бит T0.1): G8-фит (f, a) делался на ПОЛНОМ null-ACF (k=2..k), но blk-null
**сохраняет внутри-блок структуру (k < b = 8) по построению** — DC+MA(1) вынужден
«впитывать» её (BIC сам выбирает p > 1 в 917/2400) → (f, a) специфицированы неверно →
D-мерная sim хвоста (k=33..96) в неверном масштабе → sd_null/coverage ложны.

**Решение (G8-v2):** null-симуляция G8 строится от **tail-fit**:
- f — ЛС-фит ТОЛЬКО на хвостовой области k = 33..min(96, K) (PLAT_HI — та область,
  которую тестирует часть 1; модель DC+MA(1) при k ≥ 2 линейна по f: ρ(k) = f(1−k/n),
  член a влияет ТОЛЬКО на лаг 1 ⇒ для хвоста a неиндентифицируема и на sd_null не
  влияет);
- a — closed-form из лаг-1 (как в v1; реалистичность sim, на хвост не влияет);
- Gate НЕ МЕНЯЕТСЯ: (а) median |tail_excess|/sd_null ≤ 3.0 (D-мерная sim, tail-fit);
  (б) coverage null-статистик sim-распределением ≥ 0.90 (fit = tail-fit).
- Reportable (НЕ гейт): full-fit (f_full, a_full) + BIC-выбор p на полном ACF —
  внутри-блок структура (k < 8) сохранена по построению и ожидается p > 1; отчёт
  с разложением «full vs tail».
- Часть 1 (реальные данные) НЕ ТРОНУТА: full-fit — D-совместный дизайн для данных
  (коротко-лаг структура реального Δh — часть модели, а не артефакт шифла).
- Свойство блок-нуля для хвоста (обоснование корректности sim): при k ≥ b = 8 пары
  (x_t, x_{t+k}) всегда в РАЗНЫХ блоках (блочная перестановка ⇒ блоки независимы) ⇒
  хвост null = iid-внесение с сохранёнными per-dim m/s + DC-базовая (1−k/N)·f_DC;
  tail-fit f ≈ уровень DC-базовой (≈0.05) ⇒ sim ≈ DC + почти iid ⇒ sd_null верен.

## G8-v3 ре-спека (15.11, ДО перепуска; G8-v2 всё ещё не калиброван — измерено)

**Сбой v2 (измерено на 3 доках, ДО прогона):** после tail-fit (G8-v2) расхождение
остаётся, но уже в **масштабе флуктуации**: sd_null (D-мерная DC+MA(1)-sim, B=200)
систематически МЕНЬШЕ эмпирической null-флуктуации (200 перестановок) в 1.6–5.5×:
chat_0 L1: sim 0.00011 vs emp 0.00061 (5.5×); math_0 L5: 0.00012 vs 0.00018 (1.6×);
code_0 L11: 0.00013 vs 0.00059 (4.4×). Причина: sim строит строки как u_блок +
iid-по-колонкам MA(1)-шум (гауссовы колонки ⇒ CLT-сглаживание произведений пар),
а реальные строки Δh = Δ(U·hn) — НЕ гауссовы по колонкам (концентрация row-power)
⇒ дисперсия Σ_{t,j} x_tj x_{t+k,j} под блочной перестановкой больше, чем в sim.
Проверка на реальном null: 200 перестановок chat_0 L1 — tail_excess:
mean −0.00018, sd 0.00061, q025/q975 = −0.00129/+0.00091 (сцентрован, симметричен);
dip null ≈ −0.36 ± 0.007 (внутри-блок сохранён); plat_lo null ≈ 0 ± 0.003.
Т.е. NULL ЗДОРОВ (сцентрован) — ложит G8 ложил СИМУЛЯЦИЯ, а не null.

**Решение (G8-v3):** масштаб гейта — ЭМПИРИЧЕСКОЕ null-распределение; sim — проверка
среднего. Гейты:
- (а) median |tail_excess| (по (doc,L,seed), 2400 клеток) ≤ 3·sd_EMP, где
  sd_EMP = std tail_excess в пуле (по всем null-клеткам). Смысл: null сцентрован и
  структура не «протекает» в хвост; константная структура δ во всех клетках смещает
  медиану без раздувания поул-sd ⇒ гейт ловит δ > 3σ. (При healthy null:
  median|X| ≈ 0.67·σ → ratio ≈ 0.5–0.8.)
- (б) sim-СРЕДНЕЕ согласовано с null-статистиками на эмпирическом масштабе:
  frac клеток с |null_stat − sim_mean| ≤ 3·sd_pool_stat ≥ 0.90 по трём статистикам:
  dip (raw), plat_lo_ex, tail_excess (excess = stat − f_dc·(1−k̄/n)). Смысл: fit→sim
  пайплайн воспроизводит уровень, который fit обещает (баг «sim игнорирует f»
  пойман); raw-покрытие sim-распределением (v1/v2) — reportable.
- sd_null (sim) — reportable калибровочная диагностика (ожидаемо занижен;
  emp/sim ratio отчитывается), НЕ масштаб гейта.
- Всё остальное G8 (blk-shuffle, tail-fit f/a, selected-distribution, full-fit
  reportable) — без изменений от G8-v2.

**Верификация дизайна (15.11, 8 клеток × 5 сидов, реальные доки):** (а) ratio =
0.825 ≤ 3 PASS; (б) dip 1.00, plat_lo_ex 1.00, tail_excess 0.97 ≥ 0.90 PASS.
Боковой результат (reportable-кандидат): code_0 L11 имеет РЕАЛЬНЫЙ tail-excess
+0.0056 (≈10× своей null-флуктуации 0.0006) — возможная подлинная длинно-лаг
структура; это объект частей 1/2, не дефект null.

## G8-v3.1 поправка (15.11, ДО перепуска; баг в G8-v3 (б))

**Баг (измерено при верификации v3):** sample f_dc симуляции (fdcs = field.f_dc(X_b))
перегружен шумом СРЕДНЕГО MA(1) по t: x_{t,j} = m2_j + s2_j·u_{t,j}, sample mean по t
≠ m2_j (sd ≈ √(V·(1+2ρ₁)/n) per column; Σ по D колонкам даёт ≈ V/n ≈ 0.0001–0.0024),
пока fit f ≈ 0.0003–0.0017. Измерено (chat_0 L1, seed 1001, B=500): f = 0.00043,
mean fdcs = 0.00117 (2.7×f), mean excess_sim = −0.00062 (систематически отрицателен,
тогда как E[sim tail] = f(1−k/n) > 0). Причина: «excess» sim = plat_hi_sim −
fdcs·(1−k̄/n) вычитает ОШИБОЧНО БОЛЬШУЮ базу.

**Поправка (G8-v3.1):** (б) — RAW-сравнение: |null_stat − sim_mean| ≤ 3·sd_pool_stat
по dip, plat_lo, plat_hi (без base-вычитания; DC у null и так ≈ 0, а base-вычитание
для sim некорректно по причине выше). sd_null (reportable) = std(plat_hi_sim −
f_fit·(1−k̄/n)) — base = КОНСТРУКЦИОННЫЙ f (E[rho_sim(k≥2)] = f(1−k/n) точно, т.к.
MA(1): E[u_t u_{t+k}] = 0 при k ≥ 2). (а) без изменений.

**Верификация v3.1 (15.11, 8 клеток × 5 сидов):** (а) med_ratio = 0.825 ≤ 3 PASS;
(б) raw: dip frac=1.00 (med_gap 0.00027), plat_lo frac=1.00, plat_hi frac=0.97
(≤3·sd_pool=0.975 по 40 клеткам) PASS. См. также замер: raw sim plat_hi mean ≈
f(1−k̄/n) + 0.00015-шум (B=500, chat_0 L1: raw mean 0.00036 = analytic f(1−k̄/n)
0.00036 — совпадает; смещение −0.00062 было полностью от fdcs-base).

## Addendum r5 (15.11.2026: фиксы внешнего ревью docs/AT0_5_review.md r4.1,
sha f48cb29f55ef62a1e6c192cd952bce574ed7ec8955be86a14250d34353b4e427 + G8-v3.1
ре-дизайн пользователя + trackio по AGENTS)
Эксперимент не был завершен ни одним прогоном (первый полный прогон убит SIGTERM
на nulls-фазе — RUN_RC=143, артефактов нет) — все правки ДО первого засчитанного
прогона.
- C1: per-layer Wilson 95% CI для frac_covered (frac_covered_ci95); source-coverage
  (frac_covered_by_source: кто покрывается, не только кто в пуле; corpus в
  primary — per-source split обязателен). ВЕРДИКТ 3-Х ВЕТВОЧНЫЙ: CI lo > 0.5 на
  всех слоях → branch1; слой с CI hi < 0.5 или пустой primary-пул → branch2;
  CI содержит 0.5 → inconclusive (frac ≈ 0.5 в пределах шума, n_eligible ~50 ⇒
  CI ±0.14).
- C2: d_eff_diag (per-dim дисперсии) + d_eff_full = (trΣ)²/‖Σ‖_F² (ПОЛНАЯ
  ковариация centred Δh) — оба per-cell + медианы per layer. Full-cov null
  (опция 2 ревью): y_t = m′ + L·u_t, u ~ MA(1) независимый, L Lᵀ = Σ_c (chol,
  SVD-fallback) на всех eligible клетках → per-layer coverage под full-cov null
  (сравнение с primary independent-null coverage). Оставшаяся не-гауссовость —
  измеряет G8-v3 эмпирический пул.
- C3: seed-коллизии: calibration SEED_CAL_{A,B} + 5·idx + li (было idx + li:
  (1,0) == (0,1)); null-sim SEEDN + NSEED·(5·idx + li) + seed_i (было без layer);
  block_perm = 1000000 + 25·idx + 5·li + seed_i (было 1001 + seed_i — один pattern
  на все doc/layer; теперь НЕ paired design).
- M1: fit_scale — M = 0, f > 0: направление DC = svec/‖svec‖ (flag dc_dir:
  mvec | svec | none | degenerate), контракт «DC-доля = fit-нутый f» не ломается.
- M2: sum96 → sumK (short-клетки: сумма по k=1..K_i, не всегда 96).
- M3: gate_s5/s8 — per-cell n/k (был глобальный N=383/K=96 — short-first smoke
  ломался).
- M4: burn-in НЕ добавлен — задекларировано (VAR нач. нули: первые p шагов —
  нач.-условный артефакт ≤ 1% ACF-массы; AR(1) ñ_0 = mean(n_real) quasi-stationary;
  |β|→1 — transient возможен; part2 reportable, не влияет на вердикт).
- M5: подтверждено (не баг): null-лестница scores[r][perm] — РЯДОКИ реальных
  PCA-проекций, переставленные тем же perm: порядок времени разорван ⇒ временные
  корреляции уничтожены; базис фиксирован real-данными (изоляция ordering от
  выбора базиса). Задокументировано в docstring null_cell.
- G8-v3.1 (15.11, пользователь): (а) масштаб гейта = ЭМПИРИЧЕСКОЕ null-
  распределение (sd_emp пул по клеткам; параметрическая sim занижает флуктуацию
  null в 1.6–5.5× — реальные строки Δh не гауссовы по колонкам; sd_null sim —
  reportable); (б) sim-среднее TAIL-FIT (f на k=33..96 — dcma1_fit_tail, внутри-
  блок k<8 из fit исключена) RAW согласовано с null-статистиками:
  frac |null − sim_mean| ≤ 3·sd_pool_stat ≥ 0.90 (без base-вычитания: sample f_dc
  sim перегружен шумом среднего MA(1) ≈ V/n — баг v3).
- trackio (AGENTS: обязателен для ВСЕХ локальных CPU экспериментов): run
  at05_gencheck_r1, project prism, group AT0, TRACKIO_DIR=.trackio; t=0 — config
  (started_at, revision, script_sha, seed-формулы, device, log); ступенчатые точки
  {stage, stage_s}: gates, real_metrics, cells_part12, nulls, calibration;
  финиш — finished_at + status ok|fail|crash + log_md5; краш — atexit-страховка
  (status=crash + finish()). smoke (тест, не эксперимент) — без trackio.
- Каскад-баг (найден само-аудитом): пользовательский ре-факторинг sim_mats
  (3-элементный → 2-элементный возврат, удаление fdcs) оставил call-site
  `gm, _, gagg = sim_metrics(...)` с 3-разупаковкой — краш в cells-loop; исправлено
  на `gm, gagg = ...`.

## Addendum r6 (15.11): внешнее ревью артефакта r5 — PASS, ответка по MINOR

Внешнее ревью прогона r1: docs/dynamic/AT0_5_ARTIFACT_REVIEW_r5.md
(sha 7cbb0afd9e831cc7ba388e2e2d10943f05fbe05c9f55d0d9fdf41f43c12e05df,
VERDICT: PASS, 0 CRITICAL / 0 MAJOR / 3 MINOR, маркер AT0_5_REVIEW_R5_DONE).
Прогон: RUN_RC=0, gates ALL PASS, verdict=branch2 (frac 0.078–0.255, все CI95 hi < 0.5).

Ответка по MINOR (все опциональны, none блокирует):
- **MINOR-1** (null-зависимость покрытия) — ПРИНЯТО как оговорка в TZ/paper.
  Формулировка: «coverage null-зависим (registered primary 0.08–0.26 vs full-cov
  0.63–0.80); branch2 вынесен по registered primary null согласно pre-рег правилу;
  branch1 не поддерживается НИ ОДНИМ из null (ни primary, ни full-cov)».
  В текст claim.prereg.verdict не внесено (артефакт неизменяем post-run); фиксируется
  здесь — при цитировании вердикта использовать эту оговорку.
- **MINOR-2** (joint null-on-null 0.75–0.87 < 0.95) — ПРИНЯТО: joint — ЭМПИРИЧЕСКАЯ
  покрываемость коррелированных метрик, НЕ откалиброванная 95%; уже зафиксировано в
  claim calibration_note; интерпретировать как маркер, не как уровень значимости.
- **MINOR-3** (started_at/finished_at не в claim.meta) — ФИКС КОДОМ для будущих ран
  (claim.meta.started_at/finished_at; r1-артефакт не переписан — провенанс времени r1:
  trackio config started_at + первая строка лога 2026-11-15T07:29:11, wall_sec 46857.5 —
  согласуется).

## Addendum r6.1: ключевые числа прогона r1 (из артефакта)

- Вердикт: branch2 (все 5 слоёв below: L1 0.157 [0.082,0.280], L5 0.216 [0.125,0.346],
  L11 0.255 [0.155,0.389], L17 0.176 [0.096,0.303], L23 0.078 [0.031,0.185]).
- f_le_0 = 49–50/51 per layer: per-doc f_dc median ≈ 1e-5, 0% клеток > 0.01.
  T0.1 пул-плато ≈ 0.06 — НЕ within-doc DC и НЕ between-doc пулинг Δh: это DC-уровень
  ОБЪЕКТА h_raw (v_tangent r_v/N ≈ 0.057–0.065) — артефакт выбора объекта (reвью r5 §4.1).
- dip (ρ(1) med ≈ −0.375) покрыт в 99.5% eligible (2/419 вне полосы) — циркулярно.
  Не-покрытие: plat_lo 376, plat_hi 319 (gap med ≈ 0 — real в центре sim-распределения,
  но sim-полоса ±0.00036/±0.00018 vs real sd 0.0051/0.0014 — d_eff-расхождение, r4-C2/C4).
- d_eff: diag 223–635, full 14–43 (strong cross-dim correlation).
- full-cov null coverage: L1 0.627, L5 0.725, L11 0.784, L17 0.804, L23 0.725
  (how: chol 255, svd 0).
- null-on-null: marginal 0.91–0.98 (CI калиброван), joint 0.75–0.87 (MINOR-2).
- OOS (reportable, plat_hi): 0.373–0.608. Лестница part2: r32p1 190 / r32p2 34 /
  r32p4 254. AR(1) норма: β med 0.275, R2 med 0.076, unstable 0.
- G8 null: 2390 (doc,L,seed); (а) ratio 1.04 (sd_emp 4.7e-4, sd_null sim med 1.2e-4);
  (б) dip 1.00 / plat_lo 0.97 / plat_hi 0.97.
- S2 mean 5.9e-4 target 0; S9 478/478 max_dev 0 k_mis=[]; skips 2 (own_w0/w4 L1,
  T0.1 K=0 fp16_l1_degraded). wall 46857.5 s (≈13 ч).
- store 4732f98c6291fc8c7c41c3d47407eb0fcce1c009420068b9f83be78341952fcb (1.35MB),
  claim 173c62adf9252f81, sha-цепь claim→store→manifest сходится (reвью r5 §2).

## Что остаётся T3 (по ревью r5 §4.3)

1. Cross-dim структура сверх covariance-matched Gaussian: full-cov null закрывает
   0.63–0.80, остаток 20–37% — не-гауссовость + остаток по форме (G8-v3 эмпирический
   пул: sd_emp 1.6–5.5× выше parametric sim).
2. Форма плато пер-доков: real per-doc plat_lo/hi имеют sd 0.0051/0.0014 (sd-распределение
   доков), DC+MA(1) даёт только ±0.0004/±0.0002 — нужна модель, покрывающая per-doc
   плато-амплитуды (не только их среднее).
3. Оговорка о null-зависимости (MINOR-1) — обязательна при любом цитировании
   branch2.
